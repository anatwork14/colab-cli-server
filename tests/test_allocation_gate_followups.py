"""Offline follow-up tests for the allocation approval gate (Opus review of Issue #25).

Covers: colab_server `run` gating (lock, global cap, registry record, nonce phrase, prompt
interrupts, TOCTOU re-hash), workspace fingerprint coverage, HMAC key recovery and permissions,
strict approval-id matching, and `project approve --json` with a non-tty stdout.

Every test replaces the upstream `colab` executable and the operator terminal with mocks.
No test runs a real `colab` binary, touches the network or reads a real /dev/tty.
"""
from __future__ import annotations

import contextlib
import errno
import io
import json
import os
import re
import shutil
import socket
import stat
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import colab_server
from colab_adapter import cli
from colab_adapter.approval import (
    ApprovalStore,
    KIND_SERVER_RUN,
    count_active_managed,
    plan_fingerprint,
)
from colab_adapter.errors import AllocationError
from colab_adapter.project_sessions import ProjectSessionState
from colab_adapter.provider import (
    ApprovalRequiredError,
    ColabProcessLock,
    ManagedSessionRegistry,
)
from tests.approval_helpers import fake_guard_spawner, grant_v2
from tests.test_allocation_approval_gate import GateFixture

APPROVAL_TTY = "colab_adapter.approval._stdio_is_tty"
APPROVAL_STDIN_TTY = "sys.stdin.isatty"
APPROVAL_OPEN_TTY = "colab_adapter.approval._open_tty"
APPROVAL_READER = "colab_adapter.approval.read_tty_phrase"
PHRASE_IN_PROMPT = re.compile(r"type exactly: (ALLOCATE [^\n]*)\n")


def _cp(cmd, returncode=0):
    import subprocess

    return subprocess.CompletedProcess(args=cmd, returncode=returncode, stdout="", stderr="")


class ServerRunFixture(GateFixture):
    """Drives colab_server.main(["run", ...]) against a temp registry with a counting fake `colab run`."""

    def setUp(self):
        super().setUp()
        self.run_calls = []
        self.run_returncode = 0
        self.run_exception = None
        self.run_delay = 0.0
        self.run_hook = None
        self.prompts = []
        self.prompt_effect = None
        self.observed_states = []
        self.script = self.root / "train.py"
        patches = [
            mock.patch.object(colab_server, "_state_registry",
                              side_effect=lambda: ManagedSessionRegistry(self.registry_path)),
            mock.patch.object(colab_server.shutil, "which", return_value="/fake/bin/colab"),
            mock.patch.object(colab_server.subprocess, "run", side_effect=self.fake_run),
            mock.patch(APPROVAL_TTY, return_value=True),
            mock.patch(APPROVAL_READER, side_effect=self.fake_prompt),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def fake_run(self, cmd, **kwargs):
        self.run_calls.append(list(cmd))
        self.observed_states.append(sorted(
            info["state"] for sid, info in self.registry.get_all_managed().items() if sid.startswith("colab-server-run-")
        ))
        if self.run_hook:
            self.run_hook()
        if self.run_delay:
            import time

            time.sleep(self.run_delay)
        if self.run_exception is not None:
            raise self.run_exception
        return _cp(cmd, self.run_returncode)

    def fake_prompt(self, prompt, **_kwargs):
        self.prompts.append(prompt)
        if self.prompt_effect is not None:
            return self.prompt_effect(prompt)
        match = PHRASE_IN_PROMPT.search(prompt)
        return match.group(1) if match else ""

    def run_server(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            code = colab_server.main(["run", "--gpu", "T4", "--script", str(self.script)])
        return code, err.getvalue()

    def server_records(self):
        return {sid: info for sid, info in self.registry.get_all_managed().items()
                if sid.startswith("colab-server-run-")}

    def register_busy_session(self, session_id="colab-v2-busy-aaaaaaaa"):
        self.registry.register(session_id, "job-busy", "T4", "busy_proj")


class TestServerRunGate(ServerRunFixture):
    def test_active_managed_session_blocks_run_without_prompt_or_colab_run(self):
        self.register_busy_session()
        code, err = self.run_server()
        self.assertEqual(code, 4)
        self.assertEqual(self.run_calls, [], "no `colab run` may start while a managed session is active")
        self.assertEqual(self.prompts, [], "the operator must not be asked to approve a refused run")
        self.assertIn("global cap", err)

    def test_successful_run_registers_record_then_finalizes_it_outside_the_cap(self):
        code, _ = self.run_server()
        self.assertEqual(code, 0)
        self.assertEqual(len(self.run_calls), 1)
        self.assertEqual(self.observed_states, [["PROVISIONING"]], "record must exist while colab run is live")
        records = self.server_records()
        self.assertEqual(len(records), 1)
        (sid, info), = records.items()
        self.assertRegex(sid, r"^colab-server-run-[0-9a-f]{8}$")
        self.assertEqual(info["state"], ProjectSessionState.TERMINATED.value)
        self.assertEqual(count_active_managed(self.registry), 0)

    def test_failed_run_finalizes_record_as_failed_and_frees_the_cap(self):
        self.run_returncode = 1
        code, _ = self.run_server()
        self.assertEqual(code, 1)
        (info,) = self.server_records().values()
        self.assertEqual(info["state"], ProjectSessionState.FAILED.value)
        self.assertEqual(count_active_managed(self.registry), 0)
        self.run_returncode = 0
        code, _ = self.run_server()
        self.assertEqual(code, 0, "a finalized failed run must not block the next approved run")
        self.assertEqual(len(self.run_calls), 2)

    def test_exception_during_run_finalizes_record_as_failed(self):
        self.run_exception = OSError("simulated launch failure")
        with self.assertRaises(OSError):
            self.run_server()
        (info,) = self.server_records().values()
        self.assertEqual(info["state"], ProjectSessionState.FAILED.value)
        self.assertEqual(count_active_managed(self.registry), 0)

    def test_approval_is_consumed_by_the_run(self):
        self.run_server()
        approvals = list(self.manager.approvals.directory.glob("apr-*.json"))
        self.assertEqual(len(approvals), 1)
        self.assertEqual(json.loads(approvals[0].read_text())["status"], "consumed")
        self.assertEqual(json.loads(approvals[0].read_text())["kind"], KIND_SERVER_RUN)

    def test_phrase_carries_a_per_approval_nonce(self):
        self.run_server()
        self.assertEqual(len(self.prompts), 1)
        self.assertRegex(self.prompts[0], r"type exactly: ALLOCATE T4 [0-9a-f]{8}\n")

    def test_two_concurrent_runs_with_one_approval_each_and_cap_one_run_once(self):
        self.run_delay = 0.5
        barrier = threading.Barrier(2)
        codes = []
        codes_lock = threading.Lock()

        def worker():
            barrier.wait()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = colab_server.main(["run", "--gpu", "T4", "--script", str(self.script)])
            with codes_lock:
                codes.append(code)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertEqual(len(self.run_calls), 1, "at most one `colab run` may start under cap 1")
        self.assertEqual(sorted(codes), [0, 4])
        self.assertEqual(count_active_managed(self.registry), 0)

    def test_ctrl_c_at_prompt_refuses_cleanly_without_traceback(self):
        self.prompt_effect = self._raise(KeyboardInterrupt())
        code, err = self.run_server()
        self.assertEqual(code, 3)
        self.assertEqual(self.run_calls, [])
        self.assertEqual(self.server_records(), {})
        self.assertIn("REFUSED", err)

    def test_eof_at_prompt_refuses_cleanly_without_traceback(self):
        self.prompt_effect = self._raise(EOFError())
        code, err = self.run_server()
        self.assertEqual(code, 3)
        self.assertEqual(self.run_calls, [])
        self.assertEqual(self.server_records(), {})

    def test_wrong_phrase_is_refused_and_nothing_runs(self):
        self.prompt_effect = lambda prompt: "ALLOCATE T4 deadbeef"
        code, _ = self.run_server()
        self.assertEqual(code, 3)
        self.assertEqual(self.run_calls, [])
        self.assertEqual(self.server_records(), {})

    def test_script_changed_during_prompt_is_refused_by_rehash(self):
        def tamper(prompt):
            self.script.write_text("print('swapped after approval')\n", encoding="utf-8")
            match = PHRASE_IN_PROMPT.search(prompt)
            return match.group(1)

        self.prompt_effect = tamper
        code, _ = self.run_server()
        self.assertEqual(code, 3)
        self.assertEqual(self.run_calls, [], "a script changed after approval must never reach colab run")
        self.assertEqual(self.server_records(), {})

    def test_lock_busy_refuses_with_exit_4_and_no_prompt(self):
        held = ColabProcessLock(self.registry_path.parent / "colab_gpu.lock")
        self.assertTrue(held.acquire(blocking=False))
        try:
            code, _ = self.run_server()
        finally:
            held.release()
        self.assertEqual(code, 4)
        self.assertEqual(self.prompts, [])
        self.assertEqual(self.run_calls, [])

    @staticmethod
    def _raise(exc):
        def effect(_prompt):
            raise exc
        return effect


class TestServerRunNoRegressionOnExistingRunPaths(unittest.TestCase):
    """The default registry/lock location must follow the registry directory (never the real home)."""

    def test_lock_path_follows_registry_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ManagedSessionRegistry(Path(tmp) / "managed_sessions.json")
            lock = colab_server._state_lock(registry)
            self.assertEqual(lock.lock_path, Path(tmp) / "colab_gpu.lock")


class TestFingerprintCoverage(GateFixture):
    def fp(self):
        return plan_fingerprint(self.spec)

    def test_fingerprint_ignores_artifacts_state_logs_and_bytecode(self):
        before = self.fp()
        artifacts = self.root / "artifacts"
        artifacts.mkdir()
        (artifacts / "model.pt").write_bytes(b"pulled checkpoint")
        (self.root / "run.log").write_text("log line\n", encoding="utf-8")
        (self.root / "nested").mkdir()
        (self.root / "nested" / "debug.log").write_text("more\n", encoding="utf-8")
        state = self.root / ".colab_experiments" / "approvals"
        state.mkdir(parents=True)
        (state / "apr-x.json").write_text("{}", encoding="utf-8")
        (self.root / "__pycache__").mkdir()
        (self.root / "__pycache__" / "train.cpython-38.pyc").write_bytes(b"\x00")
        (self.root / "helper.pyc").write_bytes(b"\x00")
        self.assertEqual(self.fp(), before)

    def test_fingerprint_ignores_custom_artifacts_dir_inside_root(self):
        spec = self.make_spec(artifacts_local_dir=Path("out_dir"))
        (self.root / "out_dir").mkdir()
        before = plan_fingerprint(spec)
        (self.root / "out_dir" / "result.json").write_text("{}", encoding="utf-8")
        self.assertEqual(plan_fingerprint(spec), before)

    def test_fingerprint_changes_after_covered_source_edit(self):
        before = self.fp()
        (self.root / "train.py").write_text("print('changed')\n", encoding="utf-8")
        self.assertNotEqual(self.fp(), before)

    def test_fingerprint_changes_after_new_covered_source_file(self):
        before = self.fp()
        (self.root / "utils.py").write_text("X = 1\n", encoding="utf-8")
        self.assertNotEqual(self.fp(), before)

    def test_reuse_still_works_after_artifacts_are_pulled(self):
        approval_id = grant_v2(self.manager, self.spec, 1800.0)
        first = self.manager.up(self.spec, lease_seconds=1800.0, approval_id=approval_id)
        artifacts = self.root / "artifacts"
        artifacts.mkdir()
        (artifacts / "pulled.pt").write_bytes(b"new artifact")
        again = self.manager.up(self.spec, lease_seconds=1800.0)
        self.assertEqual(again.session_id, first.session_id)
        self.assertEqual(self.new_count(), 1)

    def test_no_reuse_with_live_session_fails_cap_and_keeps_fresh_approval(self):
        approval_id = grant_v2(self.manager, self.spec, 1800.0)
        self.manager.up(self.spec, lease_seconds=1800.0, approval_id=approval_id)
        (self.root / "artifacts").mkdir()
        (self.root / "artifacts" / "pulled.pt").write_bytes(b"x")
        fresh = grant_v2(self.manager, self.spec, 1800.0)
        with self.assertRaises(Exception) as ctx:
            self.manager.up(self.spec, lease_seconds=1800.0, approval_id=fresh, reuse_active=False)
        self.assertEqual(getattr(ctx.exception, "reason", None), "global_cap_exceeded")
        self.assertEqual(self.new_count(), 1)
        self.assertEqual(self.record(fresh)["status"], "approved")


class TestApprovalKeyRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="key-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.directory = Path(self.tmp) / "approvals"
        self.store = ApprovalStore(self.directory)
        self.key_path = self.directory / ".hmac.key"

    def request(self):
        return self.store.request(kind=KIND_SERVER_RUN, plan={"x": 1}, gpu="T4", summary="s", phrase_code="")

    def test_empty_key_file_is_recreated(self):
        self.directory.mkdir(mode=0o700)
        self.key_path.write_bytes(b"")
        os.chmod(self.key_path, 0o600)
        self.request()
        self.assertEqual(len(self.key_path.read_bytes()), 32)

    def test_short_key_file_is_recreated(self):
        self.directory.mkdir(mode=0o700)
        self.key_path.write_bytes(b"0123456789")
        os.chmod(self.key_path, 0o600)
        self.request()
        self.assertEqual(len(self.key_path.read_bytes()), 32)

    def test_insecure_key_permissions_are_tightened(self):
        self.request()
        os.chmod(self.key_path, 0o644)
        self.request()
        self.assertEqual(stat.S_IMODE(os.stat(self.key_path).st_mode), 0o600)

    def test_symlinked_key_is_refused_as_corrupt(self):
        self.directory.mkdir(mode=0o700)
        target = Path(self.tmp) / "elsewhere.key"
        target.write_bytes(os.urandom(32))
        os.symlink(target, self.key_path)
        with self.assertRaises(ApprovalRequiredError) as ctx:
            self.request()
        self.assertEqual(ctx.exception.reason, "corrupt")

    def test_concurrent_key_creation_yields_one_key(self):
        self.directory.mkdir(mode=0o700)
        barrier = threading.Barrier(4)
        keys = []
        keys_lock = threading.Lock()

        def worker():
            store = ApprovalStore(self.directory)
            barrier.wait()
            key = store._key()
            with keys_lock:
                keys.append(key)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual(len(keys), 4)
        self.assertEqual(len(set(keys)), 1)
        self.assertEqual(len(self.key_path.read_bytes()), 32)


class TestApprovalIdStrictness(GateFixture):
    def test_approval_id_with_trailing_newline_is_refused(self):
        record = self.manager.request_approval(self.spec, 1800.0)
        with mock.patch(APPROVAL_READER) as reader:
            self.assert_refused(lambda: self.manager.approve(record["approval_id"] + "\n"), "not_found")
        reader.assert_not_called()


class TestJsonApproveWithoutStdoutTty(GateFixture):
    def setUp(self):
        super().setUp()
        manifest = {
            "schema_version": 1,
            "project": {"name": "gate_proj", "root": "."},
            "runtime": {"gpu": "T4", "timeout_seconds": 600},
            "execution": {"entrypoint": "train.py"},
        }
        self.manifest = self.root / "colab-project.json"
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        patcher = mock.patch.object(cli, "ProjectSessionManager", return_value=self.manager)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.pair = socket.socketpair()
        self.addCleanup(lambda: [s.close() for s in self.pair])

    def test_reader_accepts_tty_stdin_with_non_tty_stdout_when_allowed(self):
        phrase = "ALLOCATE T4 abcd1234"
        self.pair[1].sendall((phrase + "\n").encode("utf-8"))
        with mock.patch(APPROVAL_STDIN_TTY, return_value=True), \
                mock.patch(APPROVAL_OPEN_TTY, side_effect=lambda: os.dup(self.pair[0].fileno())), \
                contextlib.redirect_stdout(io.StringIO()):
            from colab_adapter.approval import read_tty_phrase

            self.assertEqual(read_tty_phrase("> ", require_stdout_tty=False), phrase)

    def test_reader_still_refuses_non_tty_stdout_by_default(self):
        with mock.patch(APPROVAL_STDIN_TTY, return_value=True), mock.patch(APPROVAL_OPEN_TTY) as open_tty, \
                contextlib.redirect_stdout(io.StringIO()):
            from colab_adapter.approval import read_tty_phrase

            with self.assertRaises(ApprovalRequiredError) as ctx:
                read_tty_phrase("> ")
        self.assertEqual(ctx.exception.reason, "non_tty")
        open_tty.assert_not_called()

    def test_project_approve_json_piped_stdout_succeeds(self):
        code, out, _ = self.run_cli(["project", "approve-request", "--manifest", str(self.manifest),
                                     "--lease-seconds", "1800", "--json"])
        self.assertEqual(code, 0)
        view = json.loads(out)
        self.pair[1].sendall((view["phrase"] + "\n").encode("utf-8"))
        with mock.patch(APPROVAL_STDIN_TTY, return_value=True), \
                mock.patch(APPROVAL_OPEN_TTY, side_effect=lambda: os.dup(self.pair[0].fileno())):
            code, out, err = self.run_cli(["project", "approve", view["approval_id"], "--json"])
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["status"], "approved")
        self.assertEqual(self.record(view["approval_id"])["status"], "approved")

    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()


if __name__ == "__main__":
    unittest.main()
