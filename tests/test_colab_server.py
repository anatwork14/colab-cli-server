"""Offline unit tests. No access to Google or billable cloud resources."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import colab_server as server


class TestColabServer(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.script = Path(self.directory.name) / "probe.py"
        self.script.write_text("print('ok')\n", encoding="utf-8")
        # Approval records go to a temporary registry directory, never the real home directory.
        from colab_adapter.provider import ManagedSessionRegistry

        registry = ManagedSessionRegistry(Path(self.directory.name) / "managed_sessions.json")
        patcher = mock.patch.object(server, "_state_registry", return_value=registry)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_plan_does_not_spawn_subprocess_or_allocate(self):
        with mock.patch.object(server.subprocess, "run") as run:
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = server.main(["plan", "--gpu", "T4", "--script", str(self.script)])
        self.assertEqual(code, 0)
        run.assert_not_called()
        self.assertIn("PLAN ONLY", stdout.getvalue())

    def test_missing_script_rejected_before_execution(self):
        with mock.patch.object(server.subprocess, "run") as run:
            with contextlib.redirect_stderr(io.StringIO()):
                code = server.main(["run", "--gpu", "T4", "--script", "missing.py"])
        self.assertEqual(code, 2)
        run.assert_not_called()

    def test_noninteractive_run_is_refused(self):
        with mock.patch.object(server.shutil, "which", return_value="/usr/bin/colab"):
            with mock.patch.object(server.sys.stdin, "isatty", return_value=False):
                with mock.patch.object(server.subprocess, "run") as run:
                    with contextlib.redirect_stderr(io.StringIO()):
                        code = server.main(["run", "--gpu", "T4", "--script", str(self.script)])
        self.assertEqual(code, 3)
        run.assert_not_called()

    def test_wrong_confirmation_is_refused(self):
        with mock.patch.object(server.shutil, "which", return_value="/usr/bin/colab"):
            with mock.patch.object(server.sys.stdin, "isatty", return_value=True):
                with mock.patch("colab_adapter.approval.read_tty_phrase", return_value="ALLOCATE A100"):
                    with mock.patch.object(server.subprocess, "run") as run:
                        with contextlib.redirect_stderr(io.StringIO()):
                            code = server.main(["run", "--gpu", "T4", "--script", str(self.script)])
        self.assertEqual(code, 3)
        run.assert_not_called()

    def test_approved_run_invokes_official_cli_once(self):
        with mock.patch.object(server.shutil, "which", return_value="/usr/bin/colab"):
            with mock.patch.object(server.sys.stdin, "isatty", return_value=True):
                with mock.patch("colab_adapter.approval.read_tty_phrase", return_value="ALLOCATE T4"):
                    with mock.patch.object(server.subprocess, "run", return_value=mock.Mock(returncode=0)) as run:
                        code = server.main(["run", "--gpu", "T4", "--script", str(self.script)])
        self.assertEqual(code, 0)
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[:5], ["colab", "--auth=oauth2", "run", "--gpu", "T4"])
        self.assertEqual(cmd[-1], str(self.script.resolve()))

    def test_provider_flag_must_precede_subcommand(self):
        cmd = server.cli_command("adc", "usage")
        self.assertEqual(cmd, ["colab", "--auth=adc", "usage"])

    def test_timeout_validation(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code = server.main(["plan", "--gpu", "T4", "--script", str(self.script), "--timeout", "-1"])
        self.assertEqual(code, 2)

    def test_read_only_usage(self):
        with mock.patch.object(server.shutil, "which", return_value="/usr/bin/colab"):
            with mock.patch.object(server.subprocess, "run", return_value=mock.Mock(returncode=0)) as run:
                code = server.main(["--auth", "adc", "usage"])
        self.assertEqual(code, 0)
        run.assert_called_once_with(["colab", "--auth=adc", "usage"], check=False)


if __name__ == "__main__":
    unittest.main()
