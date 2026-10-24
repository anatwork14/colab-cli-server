"""Comprehensive reliability and security tests for ColabGPUProvider.

Tests all 14 failure modes using mocks without making unapproved cloud allocations:
1. Missing dependencies
2. Incorrect Python version
3. Missing OAuth authentication
4. Invalid job script
5. Unsupported GPU request
6. Denied GPU approval
7. Noninteractive provisioning refusal
8. GPU allocation error
9. Remote execution failure
10. Missing checkpoint / output verification
11. Interrupted execution / cancellation
12. Cleanup failure
13. Duplicate job submission
14. Existing unrelated sessions
"""
from __future__ import annotations

import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from colab_adapter.provider import (
    AllocationError,
    ApprovalRequiredError,
    ApprovedJob,
    ArtifactMissingError,
    AuthenticationError,
    CleanupError,
    ColabGPUProvider,
    DependencyMissingError,
    DuplicateAllocationError,
    InvalidJobSpecError,
    JobSpec,
    PreparedJob,
    PythonVersionError,
    RemoteExecutionError,
    UnsupportedGPUError,
)
from colab_adapter.tracker import ExperimentTracker


class TestColabReliabilityAndSecurity(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.log_dir = Path(self.temp_dir.name) / "logs"
        self.tracker = ExperimentTracker(log_dir=self.log_dir)

        self.script_file = Path(self.temp_dir.name) / "train.py"
        self.script_file.write_text("print('training ok')", encoding="utf-8")

        self.reqs_file = Path(self.temp_dir.name) / "requirements.txt"
        self.reqs_file.write_text("torch\n", encoding="utf-8")

        self.provider = ColabGPUProvider(cli_executable="/usr/local/bin/colab", tracker=self.tracker)

    # 1. Missing dependencies
    def test_missing_cli_dependency(self):
        with mock.patch("shutil.which", return_value=None):
            p = ColabGPUProvider(cli_executable=None, tracker=self.tracker)
            with self.assertRaises(DependencyMissingError) as ctx:
                p._get_cli()
            self.assertIn("Official Colab CLI executable not found", str(ctx.exception))

    # 2. Incorrect Python version
    def test_incorrect_python_version(self):
        spec = JobSpec(project_name="p", script_path=self.script_file, gpu="T4")
        with mock.patch("sys.version_info", (3, 7, 0)):
            with self.assertRaises(PythonVersionError):
                self.provider.prepare_job(spec)

    # 3. Missing OAuth authentication
    def test_missing_oauth_authentication(self):
        spec = JobSpec(project_name="p", script_path=self.script_file, gpu="T4")
        prep = self.provider.prepare_job(spec)
        approved = ApprovedJob(
            prepared_job=prep,
            approval_code=prep.approval_nonce,
            approved_by="tester",
            approved_at="now",
        )
        # Mock Path.exists to return False for token.json
        with mock.patch.object(Path, "exists", return_value=False):
            with self.assertRaises(AuthenticationError) as ctx:
                self.provider.submit_job(approved)
            self.assertIn("OAuth credentials not configured", str(ctx.exception))

    # 4. Invalid job script
    def test_invalid_job_script(self):
        spec = JobSpec(project_name="p", script_path="nonexistent_script.py", gpu="T4")
        with self.assertRaises(InvalidJobSpecError):
            self.provider.prepare_job(spec)

        # Non-python file
        txt_file = Path(self.temp_dir.name) / "test.txt"
        txt_file.write_text("not python", encoding="utf-8")
        spec2 = JobSpec(project_name="p", script_path=txt_file, gpu="T4")
        with self.assertRaises(InvalidJobSpecError):
            self.provider.prepare_job(spec2)

    # 5. Unsupported GPU request
    def test_unsupported_gpu_request(self):
        spec = JobSpec(project_name="p", script_path=self.script_file, gpu="RTX4090")
        with self.assertRaises(UnsupportedGPUError) as ctx:
            self.provider.prepare_job(spec)
        self.assertIn("unsupported", str(ctx.exception).lower())

    # 6. Denied GPU approval
    def test_denied_gpu_approval(self):
        spec = JobSpec(project_name="p", script_path=self.script_file, gpu="T4")
        prep = self.provider.prepare_job(spec)
        with mock.patch("sys.stdin.isatty", return_value=True):
            with mock.patch("builtins.input", return_value="NO"):
                with self.assertRaises(ApprovalRequiredError):
                    self.provider.approve_job_interactive(prep)

    # 7. Noninteractive provisioning refusal
    def test_noninteractive_provisioning_refusal(self):
        spec = JobSpec(project_name="p", script_path=self.script_file, gpu="T4")
        prep = self.provider.prepare_job(spec)
        with mock.patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(ApprovalRequiredError) as ctx:
                self.provider.approve_job_interactive(prep)
            self.assertIn("stdin is not a tty", str(ctx.exception))

    # 8. GPU allocation error
    def test_gpu_allocation_error(self):
        spec = JobSpec(project_name="p", script_path=self.script_file, gpu="A100", stateful=True)
        prep = self.provider.prepare_job(spec)
        approved = ApprovedJob(
            prepared_job=prep,
            approval_code=prep.approval_nonce,
            approved_by="tester",
            approved_at="now",
        )
        with mock.patch.object(self.provider, "check_availability", return_value={"authenticated": True, "active_sessions": []}):
            with mock.patch.object(self.provider, "_run_cli") as mock_cli:
                # new command fails with ResourceExhausted
                mock_cli.return_value = subprocess.CompletedProcess(
                    args=["colab", "new"],
                    returncode=1,
                    stdout="",
                    stderr="ResourceExhausted: No A100 GPU available",
                )
                with self.assertRaises(AllocationError) as ctx:
                    self.provider.submit_job(approved)
                self.assertIn("Failed to allocate session", str(ctx.exception))

    # 9. Remote execution failure
    def test_remote_execution_failure(self):
        spec = JobSpec(project_name="p", script_path=self.script_file, gpu="T4", stateful=False)
        prep = self.provider.prepare_job(spec)
        approved = ApprovedJob(
            prepared_job=prep,
            approval_code=prep.approval_nonce,
            approved_by="tester",
            approved_at="now",
        )
        with mock.patch.object(self.provider, "check_availability", return_value={"authenticated": True, "active_sessions": []}):
            with mock.patch.object(self.provider, "_run_cli") as mock_cli:
                # returncode 0 but traceback inside stdout
                mock_cli.return_value = subprocess.CompletedProcess(
                    args=["colab", "run"],
                    returncode=0,
                    stdout="Traceback (most recent call last):\n  File 'train.py', line 1\nZeroDivisionError",
                    stderr="",
                )
                with self.assertRaises(RemoteExecutionError) as ctx:
                    self.provider.submit_job(approved)
                self.assertIn("Remote error detected", str(ctx.exception))

    # 10. Missing checkpoint / output verification
    def test_missing_checkpoint_verification(self):
        spec = JobSpec(
            project_name="p",
            script_path=self.script_file,
            gpu="T4",
            stateful=True,
            expected_outputs=["/content/model.pt"],
            output_download_dir=Path(self.temp_dir.name) / "out",
        )
        prep = self.provider.prepare_job(spec)
        approved = ApprovedJob(
            prepared_job=prep,
            approval_code=prep.approval_nonce,
            approved_by="tester",
            approved_at="now",
        )
        with mock.patch.object(self.provider, "check_availability", return_value={"authenticated": True, "active_sessions": []}):
            with mock.patch.object(self.provider, "_run_cli") as mock_cli:
                # new -> ok, exec -> ok, download -> fails
                def side_effect(cmd_args, *args, **kwargs):
                    cmd = cmd_args[0]
                    if cmd == "new":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="Session created", stderr="")
                    elif cmd == "exec":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="Finished", stderr="")
                    elif cmd == "download":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=1, stdout="", stderr="FileNotFoundError: /content/model.pt")
                    elif cmd == "stop":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="Stopped", stderr="")
                    return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="", stderr="")

                mock_cli.side_effect = side_effect
                with self.assertRaises(ArtifactMissingError) as ctx:
                    self.provider.submit_job(approved)
                self.assertIn("could not be downloaded", str(ctx.exception))

    # 11. Interrupted execution / cancellation
    def test_interrupted_execution(self):
        spec = JobSpec(project_name="p", script_path=self.script_file, gpu="T4", stateful=True)
        prep = self.provider.prepare_job(spec)
        approved = ApprovedJob(
            prepared_job=prep,
            approval_code=prep.approval_nonce,
            approved_by="tester",
            approved_at="now",
        )
        with mock.patch.object(self.provider, "check_availability", return_value={"authenticated": True, "active_sessions": []}):
            with mock.patch.object(self.provider, "_run_cli") as mock_cli:
                # new succeeds, exec raises KeyboardInterrupt, stop called in finally
                def side_effect(cmd_args, *args, **kwargs):
                    cmd = cmd_args[0]
                    if cmd == "new":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="Session created", stderr="")
                    elif cmd == "exec":
                        raise KeyboardInterrupt("User interrupted")
                    elif cmd == "stop":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="Stopped", stderr="")
                    return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="", stderr="")

                mock_cli.side_effect = side_effect
                with self.assertRaises(KeyboardInterrupt):
                    self.provider.submit_job(approved)

                # Verify stop was called to clean up session
                stop_calls = [c for c in mock_cli.call_args_list if "stop" in c[0][0]]
                self.assertTrue(len(stop_calls) > 0)

    # 12. Cleanup failure
    def test_cleanup_failure_detection(self):
        spec = JobSpec(project_name="p", script_path=self.script_file, gpu="T4", stateful=True)
        prep = self.provider.prepare_job(spec)
        approved = ApprovedJob(
            prepared_job=prep,
            approval_code=prep.approval_nonce,
            approved_by="tester",
            approved_at="now",
        )
        with mock.patch.object(self.provider, "check_availability", return_value={"authenticated": True, "active_sessions": []}):
            with mock.patch.object(self.provider, "_run_cli") as mock_cli:
                def side_effect(cmd_args, *args, **kwargs):
                    cmd = cmd_args[0]
                    if cmd == "new":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="Created", stderr="")
                    elif cmd == "exec":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="Done", stderr="")
                    elif cmd == "stop":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=1, stdout="", stderr="Internal server error stopping session")
                    return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="", stderr="")

                mock_cli.side_effect = side_effect
                with self.assertRaises(CleanupError) as ctx:
                    self.provider.submit_job(approved)
                self.assertIn("Failed to stop session", str(ctx.exception))

    # 13. Duplicate job submission
    def test_duplicate_job_submission(self):
        spec = JobSpec(project_name="p", script_path=self.script_file, gpu="T4")
        prep = self.provider.prepare_job(spec)
        approved = ApprovedJob(
            prepared_job=prep,
            approval_code=prep.approval_nonce,
            approved_by="tester",
            approved_at="now",
        )
        # Mock check_availability to show an active session already running
        with mock.patch.object(
            self.provider,
            "check_availability",
            return_value={"authenticated": True, "active_sessions": ["existing-colab-session-1"]},
        ):
            with self.assertRaises(DuplicateAllocationError) as ctx:
                self.provider.submit_job(approved)
            self.assertIn("Active Colab sessions exist", str(ctx.exception))

    # 14. Existing unrelated sessions
    def test_existing_unrelated_sessions_handled_in_cleanup(self):
        with mock.patch.object(
            self.provider,
            "check_availability",
            return_value={"authenticated": True, "active_sessions": ["stray-session-a", "stray-session-b"]},
        ):
            with mock.patch.object(self.provider, "_run_cli") as mock_cli:
                mock_cli.return_value = subprocess.CompletedProcess(args=["colab", "stop"], returncode=0, stdout="Stopped", stderr="")
                res = self.provider.cleanup()
                self.assertEqual(res["stopped"], ["stray-session-a", "stray-session-b"])
                self.assertEqual(res["failed"], [])

    # 15. Insufficient disk space check before download
    def test_insufficient_disk_space_before_download(self):
        from colab_adapter.provider import DiskSpaceError, check_disk_space

        with mock.patch("shutil.disk_usage") as mock_usage:
            # 100 MB free, min required 500 MB
            mock_usage.return_value = mock.Mock(free=100 * 1024 * 1024)
            with self.assertRaises(DiskSpaceError) as ctx:
                check_disk_space(Path(self.temp_dir.name), min_free_mb=500)
            self.assertIn("Insufficient disk space", str(ctx.exception))

    # 16. Subprocess command timeout
    def test_subprocess_timeout_raises_remote_execution_error(self):
        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["colab", "run"], timeout=5)):
            with self.assertRaises(RemoteExecutionError) as ctx:
                self.provider._run_cli(["run"], timeout=5)
            self.assertIn("timed out after 5s", str(ctx.exception))

    # 17. Stale session reconciliation
    def test_reconcile_stale_sessions(self):
        with mock.patch.object(
            self.provider,
            "check_availability",
            return_value={"authenticated": True, "active_sessions": ["stale-1", "stale-2"]},
        ):
            with mock.patch.object(self.provider, "_run_cli") as mock_cli:
                mock_cli.return_value = subprocess.CompletedProcess(args=["colab", "stop"], returncode=0, stdout="Stopped", stderr="")
                res = self.provider.reconcile_stale_sessions(auto_stop=True)
                self.assertEqual(res["stale_sessions"], ["stale-1", "stale-2"])
                self.assertEqual(res["stopped"], ["stale-1", "stale-2"])

    # 18. Malformed CLI output during sessions
    def test_malformed_cli_output_in_availability(self):
        with mock.patch.object(Path, "exists", return_value=True):
            with mock.patch.object(self.provider, "_run_cli") as mock_cli:
                mock_cli.return_value = subprocess.CompletedProcess(
                    args=["colab", "sessions"],
                    returncode=0,
                    stdout="[WARNING] Unexpected warning\n[colab] No active sessions found on server.\n",
                    stderr="",
                )
                res = self.provider.check_availability()
                self.assertTrue(res["authenticated"])
                self.assertEqual(res["active_sessions"], ["[WARNING] Unexpected warning"])


if __name__ == "__main__":
    unittest.main()
