"""Offline tests for `project exec` semantics (F1).

The argument of `project exec` is PYTHON code run by the remote kernel (`colab exec -f`). `--working-dir`
is a Python prologue (`os.chdir`), and `--shell` wraps a shell command in a Python payload that prints an
`EXIT_CODE <n>` marker. Tests run the generated Python locally through the fake `colab` (which executes
the `-f` file with python3). No real `colab` binary is ever run.
"""
import ast
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from colab_adapter.errors import LeaseExpiredError, TransportError
from colab_adapter.project_sessions import ProjectSession, ProjectSessionManager, build_exec_code
from colab_adapter.provider import ColabProcessLock, ManagedSessionRegistry
from colab_adapter.transports.official_cli import OfficialCliTransport
from tests.approval_helpers import fake_guard_spawner
from tests.colab_fakes import fake_env, write_fake_colab

SID = "colab-v2-exec-sem01"


def _iso(offset_s):
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_s)).isoformat()


class ExecFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="exec-sem-test-"))
        self.addCleanup(shutil.rmtree, str(self.tmp), True)
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.state_dir = self.tmp / "state"
        (self.state_dir / "alive").mkdir(parents=True)
        self.calls_file = self.tmp / "calls.jsonl"
        self.fake = write_fake_colab(self.tmp / "bin" / "colab")
        patcher = mock.patch.dict(os.environ, fake_env(self.state_dir, self.calls_file, self.home))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.registry = ManagedSessionRegistry(self.tmp / "registry" / "managed_sessions.json")
        self.lock = ColabProcessLock(self.tmp / "colab.lock")
        self.work = self.tmp / "work dir with spaces"
        self.work.mkdir()
        self.put_session(SID)

    def put_session(self, sid):
        now = datetime.now(timezone.utc).isoformat()
        session = ProjectSession(
            session_id=sid, project_id="proj-exec", project_name="exec", state="READY", accelerator="T4",
            created_at=now, updated_at=now, lease_seconds=900.0, expires_at=_iso(900),
            remote_root="/content/exec", approved_fingerprint="fp", approval_id="apr",
        )
        with self.registry._lock():
            data = self.registry._read_locked()
            data[sid] = session.to_dict()
            self.registry._atomic_write_locked(data)

    def manager(self, transport=None):
        kwargs = dict(registry=self.registry, lock=self.lock, spawn_fn=fake_guard_spawner(self))
        if transport is None:
            transport = OfficialCliTransport(registry=self.registry, cli_path=str(self.fake), auth_mode="oauth2")
        return ProjectSessionManager(transport=transport, auth_mode="oauth2", **kwargs)

    def calls(self):
        import json
        if not self.calls_file.exists():
            return []
        return [json.loads(x) for x in self.calls_file.read_text(encoding="utf-8").splitlines() if x]


class TestPayloadBuilder(unittest.TestCase):
    def test_plain_code_is_passed_through_unchanged(self):
        self.assertEqual(build_exec_code("print(1)"), "print(1)")

    def test_working_dir_is_a_python_prologue_not_a_shell_cd(self):
        code = build_exec_code("print('x')", working_dir="/content/run dir")
        self.assertEqual(code, "import os\nos.chdir('/content/run dir')\nprint('x')")
        self.assertNotIn("cd ", code)
        ast.parse(code)

    def test_shell_payload_is_valid_python_for_hostile_quoting(self):
        commands = [
            "echo it's",
            'printf "%s\\n" "a \\"b\\""',
            "true && echo ok",
            "echo $HOME; echo ${PATH:+set}",
            "echo line1\necho line2",
            "exit 0",
        ]
        for command in commands:
            with self.subTest(command=command):
                ast.parse(build_exec_code(command, working_dir="/content", shell=True))

    def test_shell_payload_uses_repr_so_no_text_is_interpolated_raw(self):
        command = "echo it's && echo $X\nexit 0"
        code = build_exec_code(command, shell=True)
        self.assertIn(repr(command), code)
        self.assertIn("EXIT_CODE", code)


class TestWorkingDirValidation(ExecFixture):
    def test_unsafe_working_dirs_are_rejected_before_any_cli_call(self):
        bad = ["relative/dir", "", "/content/a\nprint(1)", "/content/a\rb", "/content/\x00x", "/content/../etc"]
        transport = mock.MagicMock()
        mgr = self.manager(transport=transport)
        for wd in bad:
            with self.subTest(working_dir=wd):
                with self.assertRaises((TransportError, ValueError)):
                    build_exec_code("print(1)", working_dir=wd)
                with self.assertRaises((TransportError, ValueError)):
                    mgr.exec(SID, "print(1)", working_dir=wd)
        transport.exec_code.assert_not_called()
        self.assertEqual(self.calls(), [], "no colab command may run for an invalid working dir")


class TestExecRunsPython(ExecFixture):
    def test_working_dir_prologue_changes_directory_on_the_kernel(self):
        mgr = self.manager()
        code = "import os\nprint('CWD=' + os.getcwd())"
        res = mgr.exec(SID, code, working_dir=str(self.work), timeout=60.0)
        self.assertTrue(res["success"], res)
        self.assertIn("CWD=" + str(self.work.resolve()), res["output"])
        self.assertNotIn("cd ", " ".join(" ".join(c) for c in self.calls()))

    def test_plain_python_exec_without_working_dir(self):
        mgr = self.manager()
        res = mgr.exec(SID, "print('plain-ok')", timeout=60.0)
        self.assertTrue(res["success"])
        self.assertIn("plain-ok", res["output"])
        self.assertNotIn("exit_code", res)

    def test_kernel_traceback_on_stderr_is_reported_as_failure(self):
        """Upstream prints kernel tracebacks to stderr and still exits 0; the adapter must see them."""
        mgr = self.manager()
        res = mgr.exec(SID, "raise RuntimeError('boom-marker')", timeout=60.0)
        self.assertFalse(res["success"])
        self.assertTrue(res["has_traceback"])
        self.assertIn("boom-marker", res["output"])


class TestShellMode(ExecFixture):
    def test_shell_quotes_ampersands_dollars_and_reports_exit_code_marker(self):
        command = "printf '%s|%s\\n' \"it's \\\"quoted\\\"\" 'semi;colon' && echo done $((1+2))\necho second\nexit 3"
        res = self.manager().exec(SID, command, shell=True, timeout=60.0)
        self.assertEqual(res["exit_code"], 3)
        self.assertFalse(res["success"])
        self.assertIn("it's \"quoted\"|semi;colon", res["output"])
        self.assertIn("done 3", res["output"])
        self.assertIn("second", res["output"])
        self.assertIn("EXIT_CODE 3", res["output"])

    def test_shell_exit_zero_is_success(self):
        res = self.manager().exec(SID, "true", shell=True, timeout=60.0)
        self.assertEqual(res["exit_code"], 0)
        self.assertTrue(res["success"])

    def test_shell_runs_in_working_dir(self):
        res = self.manager().exec(SID, "pwd", shell=True, working_dir=str(self.work), timeout=60.0)
        self.assertEqual(res["exit_code"], 0)
        self.assertIn(str(self.work.resolve()), res["output"])

    def test_shell_stderr_is_visible_in_output(self):
        res = self.manager().exec(SID, "echo oops 1>&2; exit 2", shell=True, timeout=60.0)
        self.assertEqual(res["exit_code"], 2)
        self.assertIn("oops", res["output"])

    def test_missing_marker_is_never_success(self):
        transport = mock.MagicMock()
        transport.exec_code.return_value = {"success": True, "output": "hello\n", "has_traceback": False}
        res = self.manager(transport=transport).exec(SID, "echo hello", shell=True, timeout=60.0)
        self.assertIsNone(res["exit_code"])
        self.assertFalse(res["success"])

    def test_marker_must_be_a_whole_line(self):
        transport = mock.MagicMock()
        transport.exec_code.return_value = {"success": True, "output": "xEXIT_CODE 0", "has_traceback": False}
        res = self.manager(transport=transport).exec(SID, "echo", shell=True, timeout=60.0)
        self.assertIsNone(res["exit_code"])
        self.assertFalse(res["success"])


if __name__ == "__main__":
    unittest.main()
