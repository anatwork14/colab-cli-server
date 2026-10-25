"""Offline tests for the V2 allocation approval gate (Issue #25).

Every test injects a mock ``run_cmd_fn`` (or ``_run_cli``) that counts ``colab new`` calls.
No test runs a real ``colab`` binary, touches the network or uses a real terminal.

Only original public names are imported at module level, so the no-approval cases can be shown
failing against the pre-gate code. Approval helpers are imported in the tests that need them.
"""
from __future__ import annotations

import errno
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from colab_adapter import cli
from colab_adapter.errors import AllocationError, UnmanagedSessionError
from colab_adapter.project_manifest import ProjectSpec
from colab_adapter.project_sessions import ProjectSession, ProjectSessionManager, ProjectSessionState
from colab_adapter.provider import (
    ApprovalRequiredError,
    ColabGPUProvider,
    ColabProcessLock,
    DuplicateAllocationError,
    JobSpec,
    ManagedSessionRegistry,
)
from colab_adapter.tracker import ExperimentTracker
from tests.approval_helpers import grant_record, grant_v1, grant_v2

REPO_ROOT = Path(__file__).resolve().parent.parent
APPROVAL_TTY = "colab_adapter.approval._stdio_is_tty"
APPROVAL_OPEN_TTY = "colab_adapter.approval._open_tty"
APPROVAL_CLOCK = "colab_adapter.approval._utcnow"


def _completed(cmd, returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=cmd, returncode=returncode, stdout="ok", stderr=stderr)


class GateFixture(unittest.TestCase):
    """Project with a real entrypoint, a temporary registry and a counting fake `colab` backend."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gate-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = Path(self.tmp) / "project"
        self.root.mkdir()
        (self.root / "train.py").write_text("print('train')\n", encoding="utf-8")
        self.spec = self.make_spec()
        self.registry_path = Path(self.tmp) / "managed_sessions.json"
        self.lock_path = Path(self.tmp) / "colab.lock"
        self.calls_log = Path(self.tmp) / "new_calls.log"
        self.registry = ManagedSessionRegistry(self.registry_path)
        self.lock = ColabProcessLock(self.lock_path)
        self.new_calls = []
        self.commands = []
        self.new_returncode = 0
        self.new_exception = None
        self.new_delay = 0.0
        self.manager = self.make_manager()

    def make_spec(self, **overrides):
        kwargs = dict(name="gate_proj", root=self.root, entrypoint="train.py", gpu="T4", timeout_seconds=600.0)
        kwargs.update(overrides)
        return ProjectSpec(**kwargs)

    def make_manager(self):
        return ProjectSessionManager(registry=self.registry, lock=self.lock, run_cmd_fn=self.fake_colab)

    def fake_colab(self, cmd, **kwargs):
        self.commands.append(list(cmd))
        if len(cmd) > 1 and cmd[1] == "new":
            self.new_calls.append(list(cmd))
            with open(self.calls_log, "a", encoding="utf-8") as fh:
                fh.write("new\n")
            if self.new_delay:
                time.sleep(self.new_delay)
            if self.new_exception is not None:
                raise self.new_exception
            return _completed(cmd, returncode=self.new_returncode, stderr="Quota exceeded" if self.new_returncode else "")
        return _completed(cmd)

    def new_count(self):
        return len(self.new_calls)

    def record_path(self, approval_id):
        return self.manager.approvals.directory / f"{approval_id}.json"

    def record(self, approval_id):
        with open(self.record_path(approval_id), encoding="utf-8") as fh:
            return json.load(fh)

    def seed_session(self, session_id, state, project_id="proj-gate_proj", gpu="T4", fingerprint=None, lease=3600):
        now = datetime.now(timezone.utc)
        session = ProjectSession(
            session_id=session_id,
            project_id=project_id,
            project_name="gate_proj",
            state=state,
            accelerator=gpu,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
            lease_seconds=float(lease),
            expires_at=(now + timedelta(seconds=lease)).isoformat(),
            remote_root="/content/colab-workspaces/gate_proj",
            approved_fingerprint=fingerprint,
        )
        with self.registry._lock():
            data = self.registry._read_locked()
            data[session_id] = session.to_dict()
            self.registry._atomic_write_locked(data)

    def assert_refused(self, fn, reason):
        with self.assertRaises(ApprovalRequiredError) as ctx:
            fn()
        self.assertEqual(ctx.exception.reason, reason)
        return ctx.exception


class TestNoApprovalNoAllocation(GateFixture):
    def test_up_without_approval_makes_zero_new_calls(self):
        self.assert_refused(lambda: self.manager.up(self.spec, lease_seconds=1800.0), "missing")
        self.assertEqual(self.new_count(), 0)

    def test_up_without_reuse_and_without_approval_makes_zero_new_calls(self):
        self.assert_refused(lambda: self.manager.up(self.spec, reuse_active=False), "missing")
        self.assertEqual(self.new_count(), 0)

    def test_unknown_approval_id_makes_zero_new_calls(self):
        self.assert_refused(
            lambda: self.manager.up(self.spec, approval_id="apr-" + "0123456789abcdef" * 2), "not_found"
        )
        self.assertEqual(self.new_count(), 0)

    def test_malformed_approval_id_makes_zero_new_calls(self):
        self.assert_refused(lambda: self.manager.up(self.spec, approval_id="../../etc/x"), "not_found")
        self.assertEqual(self.new_count(), 0)

    def test_require_approval_knob_is_gone(self):
        with self.assertRaises(TypeError):
            ProjectSessionManager(registry=self.registry, lock=self.lock, run_cmd_fn=self.fake_colab,
                                  require_approval=False)


class TestApprovalLifecycle(GateFixture):
    def test_valid_approval_allows_exactly_one_new_and_is_consumed(self):
        approval_id = grant_v2(self.manager, self.spec, 1800.0)
        session = self.manager.up(self.spec, lease_seconds=1800.0, approval_id=approval_id)
        self.assertEqual(session.state, ProjectSessionState.READY.value)
        self.assertEqual(session.approval_id, approval_id)
        self.assertEqual(self.new_count(), 1)
        self.assertEqual(self.record(approval_id)["status"], "consumed")

    def test_exact_reuse_needs_no_approval_and_no_new_call(self):
        approval_id = grant_v2(self.manager, self.spec, 1800.0)
        first = self.manager.up(self.spec, lease_seconds=1800.0, approval_id=approval_id)
        again = self.manager.up(self.spec, lease_seconds=1800.0)
        self.assertEqual(first.session_id, again.session_id)
        self.assertEqual(self.new_count(), 1)

    def test_consumed_approval_cannot_be_reused(self):
        approval_id = grant_v2(self.manager, self.spec, 1800.0)
        self.manager.up(self.spec, lease_seconds=1800.0, approval_id=approval_id)
        # Stop the managed session so the global cap does not mask the consumed-approval check.
        self.manager.stop(self.manager.find_reusable_session("proj-gate_proj", "T4", self.current_fp()).session_id)
        self.assert_refused(
            lambda: self.manager.up(self.spec, lease_seconds=1800.0, approval_id=approval_id, reuse_active=False),
            "consumed",
        )
        self.assertEqual(self.new_count(), 1)

    def test_expired_approval_is_refused(self):
        approval_id = grant_v2(self.manager, self.spec, 1800.0)
        later = datetime.now(timezone.utc) + timedelta(days=1)
        with mock.patch(APPROVAL_CLOCK, return_value=later):
            self.assert_refused(
                lambda: self.manager.up(self.spec, lease_seconds=1800.0, approval_id=approval_id), "expired"
            )
        self.assertEqual(self.new_count(), 0)

    def test_expired_pending_request_cannot_be_approved(self):
        record = self.manager.request_approval(self.spec, 1800.0)
        later = datetime.now(timezone.utc) + timedelta(days=1)
        with mock.patch(APPROVAL_CLOCK, return_value=later), mock.patch(APPROVAL_TTY, return_value=True), \
                mock.patch("colab_adapter.approval.read_tty_phrase", return_value=record["phrase"]):
            self.assert_refused(lambda: self.manager.approve(record["approval_id"]), "expired")

    def test_forged_record_fails_hmac_check(self):
        record = self.manager.request_approval(self.spec, 1800.0)
        approval_id = record["approval_id"]
        # Forge: mark it approved by hand without the operator phrase and without a valid MAC.
        data = self.record(approval_id)
        data["status"] = "approved"
        data["approved_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.record_path(approval_id), "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        self.assert_refused(lambda: self.manager.up(self.spec, lease_seconds=1800.0, approval_id=approval_id),
                            "invalid_mac")
        self.assertEqual(self.new_count(), 0)

    def test_forged_mac_value_is_refused(self):
        approval_id = grant_v2(self.manager, self.spec, 1800.0)
        data = self.record(approval_id)
        data["mac"] = "0" * 64
        with open(self.record_path(approval_id), "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        self.assert_refused(lambda: self.manager.up(self.spec, lease_seconds=1800.0, approval_id=approval_id),
                            "invalid_mac")
        self.assertEqual(self.new_count(), 0)

    def test_changed_gpu_inside_approved_record_fails_hmac(self):
        approval_id = grant_v2(self.manager, self.spec, 1800.0)
        data = self.record(approval_id)
        data["gpu"] = "A100"
        data["plan"]["gpu"] = "A100"
        with open(self.record_path(approval_id), "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        self.assert_refused(lambda: self.manager.up(self.spec, lease_seconds=1800.0, approval_id=approval_id),
                            "invalid_mac")
        self.assertEqual(self.new_count(), 0)

    def test_record_and_key_permissions_are_owner_only(self):
        approval_id = grant_v2(self.manager, self.spec, 1800.0)
        directory = self.manager.approvals.directory
        self.assertEqual(stat.S_IMODE(os.stat(directory).st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(os.stat(self.record_path(approval_id)).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(directory / ".hmac.key").st_mode), 0o600)

    def current_fp(self):
        from colab_adapter.approval import plan_fingerprint

        return plan_fingerprint(self.spec)


class TestPlanBinding(GateFixture):
    def assert_plan_mismatch(self, spec):
        approval_id = grant_v2(self.manager, self.spec, 1800.0)
        self.assert_refused(lambda: self.manager.up(spec, lease_seconds=1800.0, approval_id=approval_id),
                            "plan_mismatch")
        self.assertEqual(self.new_count(), 0)
        self.assertEqual(self.record(approval_id)["status"], "approved", "a rejected approval must not be consumed")

    def test_wrong_gpu_is_refused(self):
        self.assert_plan_mismatch(self.make_spec(gpu="A100"))

    def test_wrong_project_is_refused(self):
        self.assert_plan_mismatch(self.make_spec(name="other_proj"))

    def test_changed_workspace_file_fingerprint_is_refused(self):
        approval_id = grant_v2(self.manager, self.spec, 1800.0)
        (self.root / "train.py").write_text("print('changed')\n", encoding="utf-8")
        self.assert_refused(lambda: self.manager.up(self.spec, lease_seconds=1800.0, approval_id=approval_id),
                            "plan_mismatch")
        self.assertEqual(self.new_count(), 0)

    def test_lease_above_approved_maximum_is_refused_then_smaller_lease_works(self):
        approval_id = grant_v2(self.manager, self.spec, 1800.0)
        self.assert_refused(lambda: self.manager.up(self.spec, lease_seconds=3600.0, approval_id=approval_id),
                            "lease_exceeded")
        self.assertEqual(self.new_count(), 0)
        session = self.manager.up(self.spec, lease_seconds=900.0, approval_id=approval_id)
        self.assertEqual(session.state, ProjectSessionState.READY.value)
        self.assertEqual(self.new_count(), 1)

    def test_wrong_kind_is_refused(self):
        from colab_adapter.approval import KIND_V1_JOB, digest_of, v2_plan

        approval_id = grant_v2(self.manager, self.spec, 1800.0)
        self.assert_refused(
            lambda: self.manager.approvals.verify_and_consume(
                approval_id, kind=KIND_V1_JOB, plan_digest=digest_of(v2_plan(self.spec, "colab"))
            ),
            "wrong_kind",
        )


class TestFailedAllocationConsumesApproval(GateFixture):
    def test_failed_allocation_consumes_approval_and_retry_is_rejected(self):
        approval_id = grant_v2(self.manager, self.spec, 1800.0)
        self.new_returncode = 1
        with self.assertRaises(AllocationError):
            self.manager.up(self.spec, lease_seconds=1800.0, approval_id=approval_id)
        states = [info["state"] for info in self.registry.get_all_managed().values()]
        self.assertEqual(states, [ProjectSessionState.FAILED.value])
        self.assertEqual(self.record(approval_id)["status"], "consumed")

        self.new_returncode = 0
        self.assert_refused(lambda: self.manager.up(self.spec, lease_seconds=1800.0, approval_id=approval_id),
                            "consumed")
        self.assertEqual(self.new_count(), 1)

    def test_timeout_consumes_approval_and_needs_reconciliation(self):
        approval_id = grant_v2(self.manager, self.spec, 1800.0)
        self.new_exception = subprocess.TimeoutExpired(cmd=["colab", "new"], timeout=180.0)
        with self.assertRaises(AllocationError):
            self.manager.up(self.spec, lease_seconds=1800.0, approval_id=approval_id)
        states = [info["state"] for info in self.registry.get_all_managed().values()]
        self.assertEqual(states, [ProjectSessionState.NEEDS_RECONCILIATION.value])
        self.assertEqual(self.record(approval_id)["status"], "consumed")
        self.assertEqual(self.new_count(), 1)


class TestReuseRules(GateFixture):
    def test_legacy_session_without_fingerprint_is_not_reused(self):
        self.seed_session("colab-v2-legacy-aaaaaaaa", ProjectSessionState.READY.value, fingerprint=None)
        self.assert_refused(lambda: self.manager.up(self.spec, lease_seconds=1800.0), "missing")
        self.assertEqual(self.new_count(), 0)

    def test_session_with_other_fingerprint_is_not_reused(self):
        self.seed_session("colab-v2-other-aaaaaaaa", ProjectSessionState.READY.value, fingerprint="0" * 64)
        self.assert_refused(lambda: self.manager.up(self.spec, lease_seconds=1800.0), "missing")
        self.assertEqual(self.new_count(), 0)

    def test_session_with_other_gpu_is_not_reused(self):
        from colab_adapter.approval import plan_fingerprint

        self.seed_session("colab-v2-gpu-aaaaaaaa", ProjectSessionState.READY.value, gpu="A100",
                          fingerprint=plan_fingerprint(self.spec))
        self.assert_refused(lambda: self.manager.up(self.spec, lease_seconds=1800.0), "missing")
        self.assertEqual(self.new_count(), 0)

    def test_expired_lease_session_is_not_reused(self):
        from colab_adapter.approval import plan_fingerprint

        self.seed_session("colab-v2-old-aaaaaaaa", ProjectSessionState.READY.value,
                          fingerprint=plan_fingerprint(self.spec), lease=-60)
        self.assert_refused(lambda: self.manager.up(self.spec, lease_seconds=1800.0), "missing")
        self.assertEqual(self.new_count(), 0)


class TestStopScope(GateFixture):
    def test_stop_refuses_unmanaged_session_and_runs_no_command(self):
        with self.assertRaises(UnmanagedSessionError):
            self.manager.stop("colab-user-session-1")
        self.assertEqual(self.commands, [])

    def test_stop_only_affects_registered_session(self):
        approval_id = grant_v2(self.manager, self.spec, 1800.0)
        managed = self.manager.up(self.spec, lease_seconds=1800.0, approval_id=approval_id)
        self.assertTrue(self.manager.stop(managed.session_id))
        stop_targets = [c[c.index("-s") + 1] for c in self.commands if len(c) > 1 and c[1] == "stop"]
        self.assertEqual(stop_targets, [managed.session_id])


class TestGlobalCap(GateFixture):
    def test_global_cap_blocks_second_project_and_keeps_approval_unconsumed(self):
        self.registry.register("colab-v2-other-aaaaaaaa", "job-x", "T4", "other_project")
        approval_id = grant_v2(self.manager, self.spec, 1800.0)
        with self.assertRaises(DuplicateAllocationError) as ctx:
            self.manager.up(self.spec, lease_seconds=1800.0, approval_id=approval_id)
        self.assertEqual(getattr(ctx.exception, "reason", None), "global_cap_exceeded")
        self.assertEqual(self.new_count(), 0)
        self.assertEqual(self.record(approval_id)["status"], "approved")

    def test_cap_counts_stopping_and_nonterminal_states(self):
        from colab_adapter.approval import count_active_managed

        self.seed_session("colab-v2-s-aaaaaaaa", ProjectSessionState.STOPPING.value)
        self.assertEqual(count_active_managed(self.registry), 1)
        self.seed_session("colab-v2-t-aaaaaaaa", ProjectSessionState.TERMINATED.value, project_id="p2")
        self.assertEqual(count_active_managed(self.registry), 1)


class TestConcurrentUp(GateFixture):
    def test_double_up_with_threads_and_one_approval_calls_new_once(self):
        approval_id = grant_v2(self.manager, self.spec, 1800.0)
        self.new_delay = 0.3
        barrier = threading.Barrier(2)
        outcomes = []
        outcome_lock = threading.Lock()

        def worker():
            manager = ProjectSessionManager(
                registry=self.registry,
                lock=ColabProcessLock(self.lock_path),
                run_cmd_fn=self.fake_colab,
            )
            barrier.wait()
            try:
                result = ("ok", manager.up(self.spec, lease_seconds=1800.0, approval_id=approval_id))
            except Exception as exc:  # noqa: BLE001 - classify below
                result = ("err", exc)
            with outcome_lock:
                outcomes.append(result)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertEqual(self.new_count(), 1)
        self.assertTrue(any(kind == "ok" for kind, _ in outcomes))
        for kind, value in outcomes:
            if kind == "err":
                self.assertIsInstance(value, (DuplicateAllocationError, ApprovalRequiredError), repr(value))

    def test_double_up_across_two_processes_calls_new_once(self):
        approval_id = grant_v2(self.manager, self.spec, 1800.0)
        child = Path(self.tmp) / "gate_child.py"
        child.write_text(CHILD_SCRIPT, encoding="utf-8")
        args = [
            sys.executable, str(child), str(REPO_ROOT), str(self.root), str(self.registry_path),
            str(self.lock_path), approval_id, str(self.calls_log),
        ]
        procs = [subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
        outputs = []
        for proc in procs:
            out, err = proc.communicate(timeout=120)
            self.assertEqual(proc.returncode, 0, err)
            outputs.append(json.loads(out.strip().splitlines()[-1]))
        with open(self.calls_log, encoding="utf-8") as fh:
            self.assertEqual(fh.read().count("new"), 1)
        self.assertTrue(any(o.get("ok") for o in outputs), outputs)
        for o in outputs:
            if not o.get("ok"):
                self.assertIn(o.get("error"), ("DuplicateAllocationError", "ApprovalRequiredError"), o)


CHILD_SCRIPT = r'''
import json
import subprocess
import sys
import time
from pathlib import Path

repo, root, registry_path, lock_path, approval_id, calls_log = sys.argv[1:7]
sys.path.insert(0, repo)
from colab_adapter.project_manifest import ProjectSpec
from colab_adapter.project_sessions import ProjectSessionManager
from colab_adapter.provider import ColabProcessLock, ManagedSessionRegistry

def fake_colab(cmd, **kwargs):
    if len(cmd) > 1 and cmd[1] == "new":
        with open(calls_log, "a", encoding="utf-8") as fh:
            fh.write("new\n")
        time.sleep(0.5)
    return subprocess.CompletedProcess(cmd, 0, "ok", "")

spec = ProjectSpec(name="gate_proj", root=Path(root), entrypoint="train.py", gpu="T4", timeout_seconds=600.0)
mgr = ProjectSessionManager(
    registry=ManagedSessionRegistry(Path(registry_path)),
    lock=ColabProcessLock(Path(lock_path)),
    run_cmd_fn=fake_colab,
)
try:
    s = mgr.up(spec, lease_seconds=1800.0, approval_id=approval_id)
    print(json.dumps({"ok": True, "state": s.state}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": type(exc).__name__, "reason": getattr(exc, "reason", None)}))
'''


class TestTtyReader(GateFixture):
    def grant_pending(self):
        return self.manager.request_approval(self.spec, 1800.0)

    def test_non_tty_refused_before_any_tty_access(self):
        record = self.grant_pending()
        with mock.patch(APPROVAL_TTY, return_value=False), mock.patch(APPROVAL_OPEN_TTY) as open_tty:
            with self.assertRaises(ApprovalRequiredError) as ctx:
                self.manager.approve(record["approval_id"])
        self.assertEqual(ctx.exception.reason, "non_tty")
        self.assertIn("stdin is not a tty", str(ctx.exception))
        open_tty.assert_not_called()
        self.assertEqual(self.record(record["approval_id"])["status"], "pending")
        self.assert_refused(
            lambda: self.manager.up(self.spec, lease_seconds=1800.0, approval_id=record["approval_id"]),
            "not_approved",
        )
        self.assertEqual(self.new_count(), 0)

    def test_enxio_without_controlling_terminal_is_refused(self):
        record = self.grant_pending()
        enxio = OSError(errno.ENXIO, os.strerror(errno.ENXIO))
        with mock.patch(APPROVAL_TTY, return_value=True), mock.patch(APPROVAL_OPEN_TTY, side_effect=enxio):
            with self.assertRaises(ApprovalRequiredError) as ctx:
                self.manager.approve(record["approval_id"])
        self.assertEqual(ctx.exception.reason, "no_controlling_tty")
        self.assertEqual(self.record(record["approval_id"])["status"], "pending")

    def test_wrong_phrase_keeps_request_pending(self):
        record = self.grant_pending()
        with mock.patch("colab_adapter.approval.read_tty_phrase", return_value="ALLOCATE T4 wrong"):
            self.assert_refused(lambda: self.manager.approve(record["approval_id"]), "phrase_mismatch")
        self.assertEqual(self.record(record["approval_id"])["status"], "pending")

    def test_exact_phrase_approves(self):
        record = self.grant_pending()
        grant_record(self.manager.approvals, record)
        self.assertEqual(self.record(record["approval_id"])["status"], "approved")

    def test_real_reader_refuses_piped_stdin(self):
        # Under unittest stdin is not a terminal; the reader must refuse without touching /dev/tty.
        record = self.grant_pending()
        with mock.patch(APPROVAL_OPEN_TTY) as open_tty:
            with mock.patch("sys.stdin.isatty", return_value=False):
                self.assert_refused(lambda: self.manager.approve(record["approval_id"]), "non_tty")
        open_tty.assert_not_called()


class TestV1Gate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gate-v1-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.script = Path(self.tmp) / "train.py"
        self.script.write_text("print('ok')\n", encoding="utf-8")
        self.registry = ManagedSessionRegistry(Path(self.tmp) / "managed_sessions.json")
        self.provider = ColabGPUProvider(
            cli_executable="/usr/local/bin/colab",
            tracker=ExperimentTracker(log_dir=Path(self.tmp) / "logs"),
            registry=self.registry,
            lock=ColabProcessLock(Path(self.tmp) / "v1.lock"),
        )
        self.cli_calls = []
        availability = {"authenticated": True, "active_sessions": [], "unmanaged_sessions": [], "error": None}
        patches = [
            mock.patch.object(self.provider, "check_availability", return_value=availability),
            mock.patch.object(self.provider, "get_usage", return_value={"success": True, "raw_output": ""}),
            mock.patch.object(self.provider, "_run_cli", side_effect=self.fake_cli),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def fake_cli(self, args, timeout=None):
        self.cli_calls.append(list(args))
        return _completed(args, returncode=0)

    def prepare(self, **kwargs):
        spec = JobSpec(project_name="p", script_path=self.script, gpu="T4",
                       output_download_dir=Path(self.tmp) / "out", **kwargs)
        return self.provider.prepare_job(spec)

    def launches(self):
        return [c for c in self.cli_calls if c and c[0] in ("run", "new")]

    def test_hand_built_approved_job_is_rejected_without_allocation(self):
        from colab_adapter.provider import ApprovedJob

        prep = self.prepare()
        forged = ApprovedJob(prepared_job=prep, approval_code=prep.approval_nonce, approved_by="tester",
                             fingerprint=prep.fingerprint, approved_at="now")
        with self.assertRaises(ApprovalRequiredError) as ctx:
            self.provider.submit_job(forged)
        self.assertEqual(ctx.exception.reason, "missing")
        self.assertEqual(self.launches(), [])
        self.assertEqual(self.registry.get_all_managed(), {})

    def test_unknown_approval_id_is_rejected(self):
        from colab_adapter.provider import ApprovedJob

        prep = self.prepare()
        forged = ApprovedJob(prepared_job=prep, approval_code=prep.approval_nonce, approved_by="tester",
                             fingerprint=prep.fingerprint, approved_at="now",
                             approval_id="apr-" + "0123456789abcdef" * 2)
        with self.assertRaises(ApprovalRequiredError) as ctx:
            self.provider.submit_job(forged)
        self.assertEqual(ctx.exception.reason, "not_found")
        self.assertEqual(self.launches(), [])

    def test_pending_request_is_rejected(self):
        from colab_adapter.approval import KIND_V1_JOB, v1_job_plan
        from colab_adapter.provider import ApprovedJob

        prep = self.prepare()
        pending = self.provider._approval_store().request(
            kind=KIND_V1_JOB, plan=v1_job_plan(prep), gpu="T4", summary="pending", phrase_code=prep.approval_nonce,
        )
        forged = ApprovedJob(prepared_job=prep, approval_code=prep.approval_nonce, approved_by="tester",
                             fingerprint=prep.fingerprint, approved_at="now", approval_id=pending["approval_id"])
        with self.assertRaises(ApprovalRequiredError) as ctx:
            self.provider.submit_job(forged)
        self.assertEqual(ctx.exception.reason, "not_approved")
        self.assertEqual(self.launches(), [])

    def test_approval_for_another_job_is_rejected(self):
        from colab_adapter.provider import ApprovedJob

        prep_a = self.prepare()
        other = Path(self.tmp) / "other.py"
        other.write_text("print('different')\n", encoding="utf-8")
        prep_b = self.provider.prepare_job(JobSpec(project_name="p", script_path=other, gpu="T4",
                                                   output_download_dir=Path(self.tmp) / "out2"))
        approval_for_a = grant_v1(self.provider, prep_a)
        forged = ApprovedJob(prepared_job=prep_b, approval_code=prep_b.approval_nonce, approved_by="tester",
                             fingerprint=prep_b.fingerprint, approved_at="now", approval_id=approval_for_a)
        with self.assertRaises(ApprovalRequiredError) as ctx:
            self.provider.submit_job(forged)
        self.assertEqual(ctx.exception.reason, "plan_mismatch")
        self.assertEqual(self.launches(), [])

    def test_granted_approval_is_consumed_by_submit(self):
        from colab_adapter.approval import KIND_V1_JOB, v1_job_plan
        from colab_adapter.provider import ApprovedJob

        prep = self.prepare()
        approved = ApprovedJob(prepared_job=prep, approval_code=prep.approval_nonce, approved_by="tester",
                               fingerprint=prep.fingerprint, approved_at="now", approval_id=grant_v1(self.provider, prep))
        self.provider.submit_job(approved)
        self.assertEqual(len(self.launches()), 1)
        record = self.provider._approval_store()._read(approved.approval_id)
        self.assertEqual(record["status"], "consumed")
        with self.assertRaises(ApprovalRequiredError) as ctx:
            self.provider._approval_store().verify_and_consume(
                approved.approval_id, kind=KIND_V1_JOB, plan_digest=_digest(v1_job_plan(prep)))
        self.assertEqual(ctx.exception.reason, "consumed")


def _digest(plan):
    from colab_adapter.approval import digest_of

    return digest_of(plan)


class TestCliGate(GateFixture):
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

    def run_cli(self, argv):
        import contextlib
        import io

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_up_without_approval_exits_3_with_json_and_no_allocation(self):
        code, _, err = self.run_cli(["project", "up", "--manifest", str(self.manifest), "--json"])
        self.assertEqual(code, 3)
        payload = json.loads(err)
        self.assertEqual(payload["reason"], "missing")
        self.assertEqual(self.new_count(), 0)

    def test_request_approve_and_up_flow(self):
        code, out, _ = self.run_cli(["project", "approve-request", "--manifest", str(self.manifest),
                                     "--lease-seconds", "1800", "--json"])
        self.assertEqual(code, 0)
        view = json.loads(out)
        self.assertEqual(view["status"], "pending")
        self.assertEqual(self.new_count(), 0)

        with mock.patch("colab_adapter.approval.read_tty_phrase", return_value=view["phrase"]):
            code, out, _ = self.run_cli(["project", "approve", view["approval_id"], "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["status"], "approved")

        code, out, err = self.run_cli(["project", "up", "--manifest", str(self.manifest), "--lease-seconds", "1800",
                                       "--approval-id", view["approval_id"], "--json"])
        self.assertEqual(code, 0, out + err)
        self.assertEqual(json.loads(out)["state"], ProjectSessionState.READY.value)
        self.assertEqual(self.new_count(), 1)

        # A second allocation attempt is refused by the shared cap (exit 4) before the spent approval is checked.
        code, _, err = self.run_cli(["project", "up", "--manifest", str(self.manifest), "--lease-seconds", "1800",
                                     "--approval-id", view["approval_id"], "--no-reuse", "--json"])
        self.assertEqual(code, 4)
        self.assertEqual(json.loads(err)["reason"], "global_cap_exceeded")
        self.assertEqual(self.new_count(), 1)

    def test_approve_without_terminal_exits_3(self):
        code, _, _ = self.run_cli(["project", "approve-request", "--manifest", str(self.manifest), "--json"])
        self.assertEqual(code, 0)
        approval_id = next(p.stem for p in self.manager.approvals.directory.glob("apr-*.json"))
        with mock.patch(APPROVAL_TTY, return_value=False):
            code, _, err = self.run_cli(["project", "approve", approval_id, "--json"])
        self.assertEqual(code, 3)
        # The summary lines precede the JSON error object on stderr.
        self.assertEqual(json.loads(err[err.index("{"):])["reason"], "non_tty")


if __name__ == "__main__":
    unittest.main()
