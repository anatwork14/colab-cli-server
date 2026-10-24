"""Comprehensive reliability, concurrency, and security tests for ColabGPUProvider.

Tests all failure modes, session ownership, tamper resistance, and security controls offline:
1. Missing dependencies
2. Incompatible Python version
3. Missing OAuth2 credentials
4. Application Default Credentials (ADC) mode verification (present & missing)
5. Invalid job script validation
6. Unsupported GPU request
7. Interactive human approval & refusal when stdin is not a TTY
8. Job tampering rejection (script/inputs modified after approval)
9. Cross-process concurrency locking (fcntl.flock preventing simultaneous allocation)
10. GPU allocation error handling
11. Remote execution failure & traceback detection
12. Missing checkpoint / output verification
13. Insufficient disk space check before download
14. Subprocess command timeout handling
15. Error preservation: cleanup failure does NOT mask primary execution error
16. Error preservation: usage query failure does NOT mask primary execution error
17. Structured session line parsing (exact name extraction & CLI notification filtering)
18. Session ownership: cleanup only stops managed sessions, leaves unmanaged sessions untouched
19. Stale session reconciliation: only touches managed sessions in stale states
20. Interrupted execution teardown
21. Deep secret sanitization in ExperimentTracker (OAuth codes, tokens, private keys)
22. File permission enforcement (0600 permissions on experiments.jsonl and registry)
"""
from __future__ import annotations

import io
import os
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
    ColabProcessLock,
    DependencyMissingError,
    DiskSpaceError,
    DuplicateAllocationError,
    InvalidJobSpecError,
    JobSpec,
    JobTamperedError,
    ManagedSessionRegistry,
    PreparedJob,
    PythonVersionError,
    RemoteExecutionError,
    UnsupportedGPUError,
    check_disk_space,
    parse_session_lines,
)
from colab_adapter.tracker import ExperimentMetadata, ExperimentTracker, sanitize_text


class TestColabAdapterSecurityAndReliability(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.base_dir = Path(self.temp_dir.name)

        self.log_dir = self.base_dir / "logs"
        self.tracker = ExperimentTracker(log_dir=self.log_dir)
        self.registry = ManagedSessionRegistry(registry_path=self.base_dir / "managed_sessions.json")
        self.lock = ColabProcessLock(lock_path=self.base_dir / "test.lock")

        self.script_file = self.base_dir / "train.py"
        self.script_file.write_text("print('training ok')", encoding="utf-8")

        self.reqs_file = self.base_dir / "requirements.txt"
        self.reqs_file.write_text("torch\n", encoding="utf-8")

        self.input_file = self.base_dir / "data.csv"
        self.input_file.write_text("1,2,3,4\n", encoding="utf-8")

        self.provider = ColabGPUProvider(
            cli_executable="/usr/local/bin/colab",
            tracker=self.tracker,
            registry=self.registry,
            lock=self.lock,
        )

    # 1. Missing CLI dependency
    def test_missing_cli_dependency(self):
        with mock.patch("shutil.which", return_value=None):
            p = ColabGPUProvider(cli_executable=None, tracker=self.tracker, registry=self.registry, lock=self.lock)
            with self.assertRaises(DependencyMissingError) as ctx:
                p._get_cli()
            self.assertIn("Official Colab CLI executable not found", str(ctx.exception))

    # 2. Incompatible Python version
    def test_incorrect_python_version(self):
        spec = JobSpec(project_name="p", script_path=self.script_file, gpu="T4")
        with mock.patch("sys.version_info", (3, 7, 0)):
            with self.assertRaises(PythonVersionError):
                self.provider.prepare_job(spec)

    # 3. Missing OAuth2 credentials
    def test_missing_oauth_authentication(self):
        spec = JobSpec(project_name="p", script_path=self.script_file, gpu="T4")
        prep = self.provider.prepare_job(spec)
        approved = ApprovedJob(
            prepared_job=prep,
            approval_code=prep.approval_nonce,
            approved_by="tester",
            fingerprint=prep.fingerprint,
            approved_at="now",
        )
        with mock.patch.object(Path, "exists", return_value=False):
            with self.assertRaises(AuthenticationError) as ctx:
                self.provider.submit_job(approved)
            self.assertIn("OAuth2 credentials missing", str(ctx.exception))

    # 4. ADC authentication mode
    def test_adc_mode_authenticated_and_missing(self):
        adc_provider = ColabGPUProvider(
            auth_mode="adc",
            cli_executable="/usr/local/bin/colab",
            tracker=self.tracker,
            registry=self.registry,
            lock=self.lock,
        )
        # Test missing ADC credentials
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(Path, "exists", return_value=False):
                avail = adc_provider.check_availability()
                self.assertFalse(avail["authenticated"])
                self.assertIn("Application Default Credentials missing", avail["error"])

        # Test present ADC credentials
        with mock.patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/tmp/fake_adc.json"}):
            with mock.patch.object(Path, "exists", return_value=True):
                with mock.patch.object(adc_provider, "_run_cli") as mock_cli:
                    mock_cli.return_value = subprocess.CompletedProcess(
                        args=["colab", "--auth=adc", "sessions"], returncode=0, stdout="", stderr=""
                    )
                    avail = adc_provider.check_availability()
                    self.assertTrue(avail["authenticated"])
                    self.assertEqual(avail["auth_mode"], "adc")

    # 5. Invalid job script
    def test_invalid_job_script(self):
        spec = JobSpec(project_name="p", script_path="nonexistent.py", gpu="T4")
        with self.assertRaises(InvalidJobSpecError):
            self.provider.prepare_job(spec)

        txt_file = self.base_dir / "test.txt"
        txt_file.write_text("text", encoding="utf-8")
        spec2 = JobSpec(project_name="p", script_path=txt_file, gpu="T4")
        with self.assertRaises(InvalidJobSpecError):
            self.provider.prepare_job(spec2)

    # 6. Unsupported GPU request
    def test_unsupported_gpu_request(self):
        spec = JobSpec(project_name="p", script_path=self.script_file, gpu="V100")
        with self.assertRaises(UnsupportedGPUError) as ctx:
            self.provider.prepare_job(spec)
        self.assertIn("unsupported", str(ctx.exception).lower())

    # 7. Interactive human approval and non-interactive refusal
    def test_interactive_approval_and_noninteractive_refusal(self):
        spec = JobSpec(project_name="p", script_path=self.script_file, gpu="T4")
        prep = self.provider.prepare_job(spec)

        # Refusal when not a TTY
        with mock.patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(ApprovalRequiredError) as ctx:
                self.provider.approve_job_interactive(prep)
            self.assertIn("stdin is not a tty", str(ctx.exception))

        # Rejection when wrong phrase entered
        with mock.patch("sys.stdin.isatty", return_value=True):
            with mock.patch("builtins.input", return_value="NO"):
                with self.assertRaises(ApprovalRequiredError):
                    self.provider.approve_job_interactive(prep)

        # Success when exact phrase entered
        expected_phrase = f"ALLOCATE T4 {prep.approval_nonce}"
        with mock.patch("sys.stdin.isatty", return_value=True):
            with mock.patch("builtins.input", return_value=expected_phrase):
                approved = self.provider.approve_job_interactive(prep)
                self.assertEqual(approved.approval_code, prep.approval_nonce)
                self.assertEqual(approved.fingerprint, prep.fingerprint)

    # 8. Job tampering rejection
    def test_job_tampering_rejected(self):
        spec = JobSpec(
            project_name="p",
            script_path=self.script_file,
            gpu="T4",
            requirements_path=self.reqs_file,
            input_files={str(self.input_file): "/content/data.csv"},
        )
        prep = self.provider.prepare_job(spec)
        approved = ApprovedJob(
            prepared_job=prep,
            approval_code=prep.approval_nonce,
            approved_by="tester",
            fingerprint=prep.fingerprint,
            approved_at="now",
        )

        # Verify integrity initially passes
        self.provider.verify_job_integrity(approved)

        # Tamper with script file
        self.script_file.write_text("print('malicious tamper')", encoding="utf-8")
        with self.assertRaises(JobTamperedError) as ctx:
            self.provider.submit_job(approved)
        self.assertIn("Script file was modified", str(ctx.exception))

    # 9. Concurrency protection via cross-process lock
    def test_concurrency_protection_via_lock(self):
        spec = JobSpec(project_name="p", script_path=self.script_file, gpu="T4")
        prep = self.provider.prepare_job(spec)
        approved = ApprovedJob(
            prepared_job=prep,
            approval_code=prep.approval_nonce,
            approved_by="tester",
            fingerprint=prep.fingerprint,
            approved_at="now",
        )

        # Simulate another process holding the lock
        external_lock = ColabProcessLock(lock_path=self.base_dir / "test.lock")
        self.assertTrue(external_lock.acquire(blocking=False))
        try:
            with self.assertRaises(DuplicateAllocationError) as ctx:
                self.provider.submit_job(approved)
            self.assertIn("holding the GPU execution lock", str(ctx.exception))
        finally:
            external_lock.release()

    # 10. GPU allocation error handling
    def test_gpu_allocation_error(self):
        spec = JobSpec(project_name="p", script_path=self.script_file, gpu="A100", stateful=True)
        prep = self.provider.prepare_job(spec)
        approved = ApprovedJob(
            prepared_job=prep,
            approval_code=prep.approval_nonce,
            approved_by="tester",
            fingerprint=prep.fingerprint,
            approved_at="now",
        )
        with mock.patch.object(self.provider, "check_availability", return_value={"authenticated": True, "active_sessions": []}):
            with mock.patch.object(self.provider, "_run_cli") as mock_cli:
                mock_cli.return_value = subprocess.CompletedProcess(
                    args=["colab", "new"],
                    returncode=1,
                    stdout="",
                    stderr="ResourceExhausted: No A100 GPU available",
                )
                with self.assertRaises(AllocationError) as ctx:
                    self.provider.submit_job(approved)
                self.assertIn("Failed to allocate session", str(ctx.exception))

    # 11. Remote execution failure & traceback detection
    def test_remote_execution_failure_detection(self):
        spec = JobSpec(project_name="p", script_path=self.script_file, gpu="T4", stateful=False)
        prep = self.provider.prepare_job(spec)
        approved = ApprovedJob(
            prepared_job=prep,
            approval_code=prep.approval_nonce,
            approved_by="tester",
            fingerprint=prep.fingerprint,
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
                self.assertIn("Remote error detected in execution logs", str(ctx.exception))

    # 12. Missing checkpoint verification
    def test_missing_checkpoint_verification(self):
        spec = JobSpec(
            project_name="p",
            script_path=self.script_file,
            gpu="T4",
            stateful=True,
            expected_outputs=["/content/model.pt"],
            output_download_dir=self.base_dir / "out",
        )
        prep = self.provider.prepare_job(spec)
        approved = ApprovedJob(
            prepared_job=prep,
            approval_code=prep.approval_nonce,
            approved_by="tester",
            fingerprint=prep.fingerprint,
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
                    elif cmd == "download":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=1, stdout="", stderr="FileNotFoundError")
                    elif cmd == "stop":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="Stopped", stderr="")
                    return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="", stderr="")

                mock_cli.side_effect = side_effect
                with self.assertRaises(ArtifactMissingError) as ctx:
                    self.provider.submit_job(approved)
                self.assertIn("could not be downloaded", str(ctx.exception))

    # 13. Disk space check
    def test_disk_space_check(self):
        with mock.patch("shutil.disk_usage") as mock_usage:
            mock_usage.return_value = mock.Mock(free=100 * 1024 * 1024)  # 100 MB free
            with self.assertRaises(DiskSpaceError) as ctx:
                check_disk_space(self.base_dir, min_free_mb=500)
            self.assertIn("Insufficient disk space", str(ctx.exception))

    # 14. Subprocess timeout handling
    def test_subprocess_timeout_raises_remote_execution_error(self):
        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["colab", "run"], timeout=5)):
            with self.assertRaises(RemoteExecutionError) as ctx:
                self.provider._run_cli(["run"], timeout=5)
            self.assertIn("timed out after 5s", str(ctx.exception))

    # 15. Error preservation: cleanup failure does not mask execution error
    def test_cleanup_failure_does_not_mask_execution_error(self):
        spec = JobSpec(project_name="p", script_path=self.script_file, gpu="T4", stateful=True)
        prep = self.provider.prepare_job(spec)
        approved = ApprovedJob(
            prepared_job=prep,
            approval_code=prep.approval_nonce,
            approved_by="tester",
            fingerprint=prep.fingerprint,
            approved_at="now",
        )
        with mock.patch.object(self.provider, "check_availability", return_value={"authenticated": True, "active_sessions": []}):
            with mock.patch.object(self.provider, "_run_cli") as mock_cli:
                def side_effect(cmd_args, *args, **kwargs):
                    cmd = cmd_args[0]
                    if cmd == "new":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="Created", stderr="")
                    elif cmd == "exec":
                        # Primary execution error
                        return subprocess.CompletedProcess(args=cmd_args, returncode=1, stdout="", stderr="Kernel panic")
                    elif cmd == "stop":
                        # Cleanup error
                        return subprocess.CompletedProcess(args=cmd_args, returncode=1, stdout="", stderr="Stop network failure")
                    return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="", stderr="")

                mock_cli.side_effect = side_effect
                # Must raise RemoteExecutionError (the primary failure), NOT CleanupError
                with self.assertRaises(RemoteExecutionError) as ctx:
                    self.provider.submit_job(approved)
                self.assertIn("Kernel panic", str(ctx.exception))

                # Verify cleanup error was still recorded in metadata
                job_record = self.tracker.get_job(prep.job_id)
                self.assertIsNotNone(job_record)
                self.assertIn("Stop network failure", job_record.get("cleanup_error", ""))

    # 16. Error preservation: usage error does not mask execution error
    def test_usage_error_does_not_mask_execution_error(self):
        spec = JobSpec(project_name="p", script_path=self.script_file, gpu="T4", stateful=False)
        prep = self.provider.prepare_job(spec)
        approved = ApprovedJob(
            prepared_job=prep,
            approval_code=prep.approval_nonce,
            approved_by="tester",
            fingerprint=prep.fingerprint,
            approved_at="now",
        )
        with mock.patch.object(self.provider, "check_availability", return_value={"authenticated": True, "active_sessions": []}):
            with mock.patch.object(self.provider, "get_usage", return_value={"success": False, "error": "Usage quota API 500"}):
                with mock.patch.object(self.provider, "_run_cli") as mock_cli:
                    mock_cli.return_value = subprocess.CompletedProcess(args=["colab", "run"], returncode=1, stdout="", stderr="Script crashed")
                    with self.assertRaises(RemoteExecutionError) as ctx:
                        self.provider.submit_job(approved)
                    self.assertIn("Script crashed", str(ctx.exception))

                    job_record = self.tracker.get_job(prep.job_id)
                    self.assertIsNotNone(job_record)
                    self.assertEqual(job_record.get("usage_error"), "Usage quota API 500")

    # 17. Structured session line parsing
    def test_structured_session_line_parsing(self):
        raw_output = """
[colab] Syncing remote assignments...
[managed-job-1] https://colab.google.com/runtime/1 | Hardware: T4 | Shape: STANDARD | Variant: GPU
[user-session-2] https://colab.google.com/runtime/2 | Hardware: A100 | Shape: HIGH_MEM | Variant: GPU | Status: IDLE
[?] https://colab.google.com/runtime/3 | Hardware: NONE | Shape: STANDARD | Variant: DEFAULT
[colab] No active sessions found on server.
"""
        parsed = parse_session_lines(raw_output)
        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[0]["name"], "managed-job-1")
        self.assertEqual(parsed[1]["name"], "user-session-2")
        self.assertEqual(parsed[2]["name"], "?")

    # 18. Session ownership: cleanup only touches managed sessions
    def test_cleanup_only_touches_managed_sessions(self):
        # Register one managed session
        self.registry.register("my-managed-sess", "job-1", "T4", "proj")

        # Mock CLI output showing both a managed session and an unmanaged user session
        cli_output = """
[my-managed-sess] endpoint-1 | Hardware: T4 | Shape: STANDARD | Variant: GPU
[unrelated-user-session] endpoint-2 | Hardware: A100 | Shape: HIGH_MEM | Variant: GPU
"""
        with mock.patch.object(self.provider, "_check_auth_credentials", return_value=(True, None)):
            with mock.patch.object(self.provider, "_run_cli") as mock_cli:
                def side_effect(cmd_args, *args, **kwargs):
                    if cmd_args[0] == "sessions":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout=cli_output, stderr="")
                    elif cmd_args[0] == "stop":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="Stopped", stderr="")
                    return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="", stderr="")

                mock_cli.side_effect = side_effect
                res = self.provider.cleanup()

                # MUST only stop the managed session
                self.assertEqual(res["stopped"], ["my-managed-sess"])
                self.assertEqual(res["unmanaged_ignored"], ["unrelated-user-session"])

                # Verify colab stop was NEVER called on unrelated-user-session
                stop_calls = [c[0][0][2] for c in mock_cli.call_args_list if c[0][0][0] == "stop"]
                self.assertIn("my-managed-sess", stop_calls)
                self.assertNotIn("unrelated-user-session", stop_calls)

    # 19. Stale session reconciliation
    def test_reconcile_stale_sessions_ignores_unmanaged(self):
        self.registry.register("stale-managed", "job-1", "T4", "proj")
        self.registry.update_state("stale-managed", "INTERRUPTED")

        cli_output = """
[stale-managed] endpoint-1 | Hardware: T4 | Shape: STANDARD | Variant: GPU
[unmanaged-work] endpoint-2 | Hardware: T4 | Shape: STANDARD | Variant: GPU
"""
        with mock.patch.object(self.provider, "_check_auth_credentials", return_value=(True, None)):
            with mock.patch.object(self.provider, "_run_cli") as mock_cli:
                def side_effect(cmd_args, *args, **kwargs):
                    if cmd_args[0] == "sessions":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout=cli_output, stderr="")
                    elif cmd_args[0] == "stop":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="Stopped", stderr="")
                    return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="", stderr="")

                mock_cli.side_effect = side_effect
                rec = self.provider.reconcile_stale_sessions(auto_stop=True)
                self.assertEqual(rec["stale_managed"], ["stale-managed"])
                self.assertEqual(rec["stopped"], ["stale-managed"])
                self.assertEqual(rec["unmanaged_ignored"], ["unmanaged-work"])

    # 20. Interrupted execution teardown
    def test_interrupted_execution_teardown(self):
        spec = JobSpec(project_name="p", script_path=self.script_file, gpu="T4", stateful=True)
        prep = self.provider.prepare_job(spec)
        approved = ApprovedJob(
            prepared_job=prep,
            approval_code=prep.approval_nonce,
            approved_by="tester",
            fingerprint=prep.fingerprint,
            approved_at="now",
        )
        with mock.patch.object(self.provider, "check_availability", return_value={"authenticated": True, "active_sessions": []}):
            with mock.patch.object(self.provider, "_run_cli") as mock_cli:
                def side_effect(cmd_args, *args, **kwargs):
                    cmd = cmd_args[0]
                    if cmd == "new":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="Created", stderr="")
                    elif cmd == "exec":
                        raise KeyboardInterrupt("Interrupted")
                    elif cmd == "stop":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="Stopped", stderr="")
                    return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="", stderr="")

                mock_cli.side_effect = side_effect
                with self.assertRaises(KeyboardInterrupt):
                    self.provider.submit_job(approved)

                stop_calls = [c for c in mock_cli.call_args_list if c[0][0][0] == "stop"]
                self.assertTrue(len(stop_calls) > 0)

    # 21. Deep secret sanitization in ExperimentTracker
    def test_sensitive_secrets_deeply_sanitized_in_tracker(self):
        leak_text = (
            "Exception occurred with token=ya29.a0AfH6SM... and OAuth code: "
            "4/0AXlqoi4_fakeCode1234567890abcdef and Bearer ghp_12345678901234567890abcdef\n"
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAz...fakeKey\n-----END RSA PRIVATE KEY-----"
        )
        meta = ExperimentMetadata(
            job_id="job-sec-test",
            project_name="sec",
            requested_accelerator="T4",
            start_time="now",
            completion_time="now",
            job_state="FAILED",
            script_path="/path/with?secret=supersecretpassword123",
            script_revision="rev",
            dependencies=[],
            dataset_identifier="data?token=secrettokenvalue123",
            output_artifact_paths=[],
            success=False,
            error_summary=leak_text,
            cleanup_status="CLEANED",
        )
        self.tracker.record_job(meta)
        logged = self.tracker.get_job("job-sec-test")
        self.assertIsNotNone(logged)

        # Check that none of the raw secrets leaked
        self.assertNotIn("ya29.a0AfH6SM", logged["error_summary"])
        self.assertNotIn("4/0AXlqoi4_fakeCode", logged["error_summary"])
        self.assertNotIn("ghp_12345678901234567890abcdef", logged["error_summary"])
        self.assertNotIn("BEGIN RSA PRIVATE KEY", logged["error_summary"])
        self.assertNotIn("supersecretpassword123", logged["script_path"])
        self.assertNotIn("secrettokenvalue123", logged["dataset_identifier"])

        # Check redaction placeholders present
        self.assertIn("[REDACTED_GOOGLE_TOKEN]", logged["error_summary"])
        self.assertIn("[REDACTED_OAUTH_CODE]", logged["error_summary"])
        self.assertIn("[REDACTED_GITHUB_TOKEN]", logged["error_summary"])
        self.assertIn("[REDACTED_PRIVATE_KEY]", logged["error_summary"])

    # 22. File permission enforcement
    def test_tracker_file_permissions(self):
        meta = ExperimentMetadata(
            job_id="job-perm-test",
            project_name="perm",
            requested_accelerator="T4",
            start_time="now",
            completion_time="now",
            job_state="COMPLETED",
            script_path="train.py",
            script_revision="rev",
            dependencies=[],
            dataset_identifier=None,
            output_artifact_paths=[],
            success=True,
            error_summary=None,
            cleanup_status="CLEANED",
        )
        self.tracker.record_job(meta)
        mode = oct(self.tracker.jsonl_path.stat().st_mode & 0o777)
        self.assertEqual(mode, "0o600")


if __name__ == "__main__":
    unittest.main()
