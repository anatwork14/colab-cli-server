"""Offline unit tests for V2 Remote Transports (Official CLI & Guarded SSH) (M3 / G3)."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from colab_adapter.errors import (
    SessionNotFoundError,
    SshHandshakeError,
    TransportError,
    TransportUnavailableError,
    UnmanagedSessionError,
    UpstreamCliError,
)
from colab_adapter.provider import ManagedSessionRegistry
from colab_adapter.transports import (
    OfficialCliTransport,
    SshProxyTransport,
)


class TestTransports(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="colab_test_m3_")
        self.reg_path = os.path.join(self.tmp_dir, "test_registry.json")
        self.registry = ManagedSessionRegistry(self.reg_path)
        self.session_id = "test-session-m3"
        self.registry.register(
            session_id=self.session_id,
            job_id="job-m3",
            gpu="T4",
            project_name="proj-m3",
        )
        self.config_path = os.path.join(self.tmp_dir, "sessions.json")
        Path(self.config_path).write_text(
            json.dumps({self.session_id: {"name": self.session_id, "endpoint": "https://colab"}}),
            encoding="utf-8",
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_official_cli_refuses_unmanaged_session(self):
        transport = OfficialCliTransport(registry=self.registry)
        with self.assertRaises(UnmanagedSessionError):
            transport.list_dir("unmanaged-session-999")
        with self.assertRaises(UnmanagedSessionError):
            transport.upload_file("unmanaged-session-999", "/tmp/local", "remote")
        with self.assertRaises(UnmanagedSessionError):
            transport.download_file("unmanaged-session-999", "remote", "/tmp/local")
        with self.assertRaises(UnmanagedSessionError):
            transport.exec_code("unmanaged-session-999", "print(1)")

    def test_official_cli_upload_and_download(self):
        transport = OfficialCliTransport(registry=self.registry, cli_path="/usr/bin/colab")
        local_file = Path(self.tmp_dir) / "data.txt"
        local_file.write_text("sample content", encoding="utf-8")
        dest_file = Path(self.tmp_dir) / "downloaded.txt"

        with patch("subprocess.run") as mock_run:
            mock_res = MagicMock()
            mock_res.returncode = 0
            mock_res.stdout = "[colab] Uploaded 'data.txt' to 'remote/data.txt'\n"
            mock_res.stderr = ""
            mock_run.return_value = mock_res

            # 1. Test upload
            transport.upload_file(self.session_id, local_file, "remote/data.txt")
            args_upload = mock_run.call_args_list[-1][0][0]
            self.assertIn("upload", args_upload)
            self.assertIn("-s", args_upload)
            self.assertIn(self.session_id, args_upload)
            self.assertIn(str(local_file), args_upload)

            # 2. Test download
            mock_res.stdout = "[colab] Downloaded 'remote/data.txt' to 'dest'\n"
            transport.download_file(self.session_id, "remote/data.txt", dest_file)
            args_download = mock_run.call_args_list[-1][0][0]
            self.assertIn("download", args_download)
            self.assertIn("-s", args_download)
            self.assertIn(self.session_id, args_download)

    def test_official_cli_list_dir_and_remove(self):
        transport = OfficialCliTransport(registry=self.registry, cli_path="/usr/bin/colab")

        with patch("subprocess.run") as mock_run:
            mock_res = MagicMock()
            mock_res.returncode = 0
            mock_res.stdout = "[colab] Listing files\nfile1.py\nfile2.py\nsrc/\n"
            mock_res.stderr = ""
            mock_run.return_value = mock_res

            items = transport.list_dir(self.session_id, "content")
            self.assertEqual(items, ["file1.py", "file2.py", "src/"])

            mock_res.stdout = "[colab] Deleted remote/file1.py\n"
            transport.remove_file(self.session_id, "remote/file1.py")
            args_rm = mock_run.call_args_list[-1][0][0]
            self.assertIn("rm", args_rm)
            self.assertIn("remote/file1.py", args_rm)

    def test_official_cli_exec_code_and_traceback_detection(self):
        transport = OfficialCliTransport(registry=self.registry, cli_path="/usr/bin/colab")

        with patch("subprocess.run") as mock_run:
            mock_res = MagicMock()
            mock_res.returncode = 0
            mock_res.stdout = "Loss: 0.042\nFinished\n"
            mock_res.stderr = ""
            mock_run.return_value = mock_res

            res = transport.exec_code(
                self.session_id,
                "print('Loss: 0.042')",
                timeout=45.0,
                env={"CUDA_VISIBLE_DEVICES": "0"},
            )
            self.assertTrue(res["success"])
            self.assertFalse(res["has_traceback"])

            args = mock_run.call_args_list[-1][0][0]
            self.assertIn("exec", args)
            self.assertIn("-s", args)
            self.assertIn("--env", args)
            self.assertIn("CUDA_VISIBLE_DEVICES=0", args)

            # Detect traceback in output
            mock_res.stdout = "Traceback (most recent call last):\n  File 'train.py', line 1, in <module>\nRuntimeError: CUDA OOM\n"
            res_fail = transport.exec_code(self.session_id, "raise RuntimeError")
            self.assertFalse(res_fail["success"])
            self.assertTrue(res_fail["has_traceback"])

    def test_official_cli_session_not_found_handling(self):
        transport = OfficialCliTransport(registry=self.registry, cli_path="/usr/bin/colab")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1,
                cmd=["colab", "ls", "-s", self.session_id],
                stderr="[colab] Session 'test-session-m3' not found.\n",
            )
            with self.assertRaises(SessionNotFoundError):
                transport.list_dir(self.session_id)

    def test_ssh_transport_refusal_when_disabled(self):
        ssh_transport = SshProxyTransport(
            registry=self.registry,
            allow_ssh=False,
            colab_config_path=self.config_path,
        )
        self.assertFalse(ssh_transport.check_health(self.session_id))
        with self.assertRaises(TransportUnavailableError) as ctx:
            ssh_transport.exec_code(self.session_id, "print(1)")
        self.assertIn("SSH transport is disabled", str(ctx.exception))

    def test_ssh_transport_safe_proxy_invocation_without_s_flag(self):
        ssh_transport = SshProxyTransport(
            registry=self.registry,
            allow_ssh=True,
            colab_config_path=self.config_path,
            cli_path="/usr/bin/colab",
        )
        self.assertTrue(ssh_transport.check_health(self.session_id))
        temp_dir = tempfile.mkdtemp(dir=self.tmp_dir)
        cmd = ssh_transport._prepare_safe_proxy_command(self.session_id, temp_dir)
        self.assertNotIn("-s", cmd)
        self.assertNotIn("--session", cmd)
        self.assertIn("--config", cmd)
        self.assertIn("ssh", cmd)
        self.assertIn("--proxy-mode", cmd)

    def test_official_cli_upload_creates_parent_dir_on_500(self):
        transport = OfficialCliTransport(registry=self.registry, cli_path="/usr/bin/colab")
        local_file = Path(self.tmp_dir) / "train.py"
        local_file.write_text("print('ok')", encoding="utf-8")

        with patch("subprocess.run") as mock_run:
            err_500 = subprocess.CalledProcessError(
                returncode=1,
                cmd=["colab", "upload"],
                stderr="[colab] Upload failed: 500 Server Error: Internal Server Error\n",
            )
            ok_exec = MagicMock(returncode=0, stdout="", stderr="")
            ok_upload = MagicMock(returncode=0, stdout="[colab] Uploaded\n", stderr="")
            mock_run.side_effect = [err_500, ok_exec, ok_upload]

            transport.upload_file(self.session_id, local_file, "/content/colab-workspaces/proj/train.py")
            self.assertEqual(mock_run.call_count, 3)

    def test_upload_scenario_1_genuine_missing_parent_directory(self):
        """1. Genuine missing remote parent directory -> creates dir and retries upload once, succeeding."""
        transport = OfficialCliTransport(registry=self.registry, cli_path="/usr/bin/colab")
        local_file = Path(self.tmp_dir) / "train.py"
        local_file.write_text("print('ok')", encoding="utf-8")

        with patch("subprocess.run") as mock_run:
            err_500 = subprocess.CalledProcessError(
                returncode=1,
                cmd=["colab", "upload"],
                stderr="[colab] Upload failed: 500 Server Error: /api/contents/workspace/train.py\n",
            )
            probe_out = 'COLAB_MKDIR_STATUS:{"existed": false, "isdir": true}\n'
            ok_exec = MagicMock(returncode=0, stdout=probe_out, stderr="")
            ok_upload = MagicMock(returncode=0, stdout="[colab] Uploaded\n", stderr="")
            mock_run.side_effect = [err_500, ok_exec, ok_upload]

            transport.upload_file(self.session_id, local_file, "/content/workspace/models/train.py")
            self.assertEqual(mock_run.call_count, 3)

            # First call was upload
            self.assertIn("upload", mock_run.call_args_list[0][0][0])
            # Second call was exec (probe)
            self.assertIn("exec", mock_run.call_args_list[1][0][0])
            # Third call was retry upload
            self.assertIn("upload", mock_run.call_args_list[2][0][0])

    def test_upload_scenario_2_http_500_unrelated_to_directories(self):
        """2. HTTP 500 unrelated to directories -> does NOT retry upload, raises original UpstreamCliError."""
        transport = OfficialCliTransport(registry=self.registry, cli_path="/usr/bin/colab")
        local_file = Path(self.tmp_dir) / "train.py"
        local_file.write_text("print('ok')", encoding="utf-8")

        # Case A: Probe indicates parent directory already existed (existed=True)
        with patch("subprocess.run") as mock_run:
            err_500 = subprocess.CalledProcessError(
                returncode=1,
                cmd=["colab", "upload"],
                stderr="[colab] Upload failed: 500 Server Error: Internal Server Error\n",
            )
            probe_out = 'COLAB_MKDIR_STATUS:{"existed": true, "isdir": true}\n'
            ok_exec = MagicMock(returncode=0, stdout=probe_out, stderr="")
            mock_run.side_effect = [err_500, ok_exec]

            with self.assertRaises(UpstreamCliError) as ctx:
                transport.upload_file(self.session_id, local_file, "/content/workspace/models/train.py")
            self.assertEqual(mock_run.call_count, 2)  # upload + exec, NO retry
            self.assertIn("500 Server Error", str(ctx.exception))

        # Case B: CLI error message has no 500 or upload failure (e.g. unhandled internal CLI error)
        with patch("subprocess.run") as mock_run:
            err_other = subprocess.CalledProcessError(
                returncode=1,
                cmd=["colab", "upload"],
                stderr="[colab] Network handshake failed with status 502 Bad Gateway\n",
            )
            mock_run.side_effect = err_other

            with self.assertRaises(UpstreamCliError):
                transport.upload_file(self.session_id, local_file, "/content/workspace/models/train.py")
            self.assertEqual(mock_run.call_count, 1)  # only initial upload, zero exec calls

    def test_upload_scenario_3_directory_creation_fails(self):
        """3. Directory creation fails (exec_code raises UpstreamCliError) -> preserves and raises original upload UpstreamCliError."""
        transport = OfficialCliTransport(registry=self.registry, cli_path="/usr/bin/colab")
        local_file = Path(self.tmp_dir) / "train.py"
        local_file.write_text("print('ok')", encoding="utf-8")

        with patch("subprocess.run") as mock_run:
            err_upload = subprocess.CalledProcessError(
                returncode=1,
                cmd=["colab", "upload"],
                stderr="[colab] Upload failed: 500 Server Error\n",
            )
            err_exec = subprocess.CalledProcessError(
                returncode=1,
                cmd=["colab", "exec"],
                stderr="[colab] exec command failed internally\n",
            )
            mock_run.side_effect = [err_upload, err_exec]

            with self.assertRaises(UpstreamCliError) as ctx:
                transport.upload_file(self.session_id, local_file, "/content/workspace/models/train.py")
            self.assertEqual(mock_run.call_count, 2)  # upload + exec, NO retry
            # Verify original error context is preserved, NOT the exec error
            self.assertIn("Upload failed: 500 Server Error", str(ctx.exception))

    def test_upload_scenario_4_directory_creation_reports_unsuccessful_execution(self):
        """4. Directory creation reports unsuccessful execution -> does NOT retry upload, raises original UpstreamCliError with mkdir_recovery_failed in details."""
        transport = OfficialCliTransport(registry=self.registry, cli_path="/usr/bin/colab")
        local_file = Path(self.tmp_dir) / "train.py"
        local_file.write_text("print('ok')", encoding="utf-8")

        with patch("subprocess.run") as mock_run:
            err_upload = subprocess.CalledProcessError(
                returncode=1,
                cmd=["colab", "upload"],
                stderr="[colab] Upload failed: 500 Server Error\n",
            )
            traceback_out = (
                "Traceback (most recent call last):\n"
                "  File '/tmp/tmp123.py', line 5, in <module>\n"
                "    os.makedirs(p, exist_ok=True)\n"
                "PermissionError: [Errno 13] Permission denied: '/content/workspace/models'\n"
            )
            exec_fail = MagicMock(returncode=0, stdout=traceback_out, stderr="")
            mock_run.side_effect = [err_upload, exec_fail]

            with self.assertRaises(UpstreamCliError) as ctx:
                transport.upload_file(self.session_id, local_file, "/content/workspace/models/train.py")
            self.assertEqual(mock_run.call_count, 2)  # upload + exec, NO retry
            self.assertIn("mkdir_recovery_failed", ctx.exception.details)
            self.assertIn("PermissionError", ctx.exception.details["mkdir_recovery_failed"])

    def test_upload_scenario_5_runtime_disappears_during_retry(self):
        """5. Runtime disappears during retry (SessionNotFoundError) -> raises SessionNotFoundError, never allocates a VM."""
        transport = OfficialCliTransport(registry=self.registry, cli_path="/usr/bin/colab")
        local_file = Path(self.tmp_dir) / "train.py"
        local_file.write_text("print('ok')", encoding="utf-8")

        # Case A: Runtime disappears during exec_code
        with patch("subprocess.run") as mock_run:
            err_upload = subprocess.CalledProcessError(
                returncode=1,
                cmd=["colab", "upload"],
                stderr="[colab] Upload failed: 500 Server Error\n",
            )
            err_exec_disappeared = subprocess.CalledProcessError(
                returncode=1,
                cmd=["colab", "exec"],
                stderr="[colab] Session 'test-session-m3' not found.\n",
            )
            mock_run.side_effect = [err_upload, err_exec_disappeared]

            with self.assertRaises(SessionNotFoundError):
                transport.upload_file(self.session_id, local_file, "/content/workspace/models/train.py")
            self.assertEqual(mock_run.call_count, 2)

        # Case B: Runtime disappears during retry upload
        with patch("subprocess.run") as mock_run:
            err_upload = subprocess.CalledProcessError(
                returncode=1,
                cmd=["colab", "upload"],
                stderr="[colab] Upload failed: 500 Server Error\n",
            )
            probe_out = 'COLAB_MKDIR_STATUS:{"existed": false, "isdir": true}\n'
            ok_exec = MagicMock(returncode=0, stdout=probe_out, stderr="")
            err_retry_disappeared = subprocess.CalledProcessError(
                returncode=1,
                cmd=["colab", "upload"],
                stderr="[colab] No active sessions found\n",
            )
            mock_run.side_effect = [err_upload, ok_exec, err_retry_disappeared]

            with self.assertRaises(SessionNotFoundError):
                transport.upload_file(self.session_id, local_file, "/content/workspace/models/train.py")
            self.assertEqual(mock_run.call_count, 3)

    def test_upload_scenario_6_parent_directory_already_exists_or_root(self):
        """6. Parent directory already exists (existed: true or root /content/file.py) -> does not retry upload, raises original UpstreamCliError."""
        transport = OfficialCliTransport(registry=self.registry, cli_path="/usr/bin/colab")
        local_file = Path(self.tmp_dir) / "train.py"
        local_file.write_text("print('ok')", encoding="utf-8")

        # Case A: Root /content/train.py has parent /content (trivial directory), no exec or retry
        with patch("subprocess.run") as mock_run:
            err_upload = subprocess.CalledProcessError(
                returncode=1,
                cmd=["colab", "upload"],
                stderr="[colab] Upload failed: 500 Server Error\n",
            )
            mock_run.side_effect = [err_upload]

            with self.assertRaises(UpstreamCliError):
                transport.upload_file(self.session_id, local_file, "/content/train.py")
            self.assertEqual(mock_run.call_count, 1)  # only initial upload, no exec or retry

        # Case B: Parent existed on remote (existed: true)
        with patch("subprocess.run") as mock_run:
            err_upload = subprocess.CalledProcessError(
                returncode=1,
                cmd=["colab", "upload"],
                stderr="[colab] Upload failed: 500 Server Error\n",
            )
            probe_out = 'COLAB_MKDIR_STATUS:{"existed": true, "isdir": true}\n'
            ok_exec = MagicMock(returncode=0, stdout=probe_out, stderr="")
            mock_run.side_effect = [err_upload, ok_exec]

            with self.assertRaises(UpstreamCliError):
                transport.upload_file(self.session_id, local_file, "/content/workspace/models/train.py")
            self.assertEqual(mock_run.call_count, 2)  # upload + probe, NO retry

    def test_upload_scenario_7_invalid_remote_path_rejected_before_cli(self):
        """7. Invalid remote path (path traversal /content/../etc/passwd or null byte) -> rejected before any CLI invocation with TransportError."""
        transport = OfficialCliTransport(registry=self.registry, cli_path="/usr/bin/colab")
        local_file = Path(self.tmp_dir) / "train.py"
        local_file.write_text("print('ok')", encoding="utf-8")

        invalid_paths = [
            "/content/../etc/passwd",
            "../secret.txt",
            "models/../../root",
            "/content/\x00nullbyte",
            "/content/newline\nfile.py",
            "/content/carriage\rfile.py",
            "",
            "   ",
        ]
        for bad_path in invalid_paths:
            with patch("subprocess.run") as mock_run:
                with self.assertRaises(TransportError) as ctx:
                    transport.upload_file(self.session_id, local_file, bad_path)
                mock_run.assert_not_called()
                self.assertIn("Invalid or unsafe remote path", str(ctx.exception))

    def test_upload_scenario_8_timeout_during_remote_directory_check(self):
        """8. Timeout during remote directory check (TransportError on exec_code) -> raises original UpstreamCliError without retry."""
        transport = OfficialCliTransport(registry=self.registry, cli_path="/usr/bin/colab")
        local_file = Path(self.tmp_dir) / "train.py"
        local_file.write_text("print('ok')", encoding="utf-8")

        with patch("subprocess.run") as mock_run:
            err_upload = subprocess.CalledProcessError(
                returncode=1,
                cmd=["colab", "upload"],
                stderr="[colab] Upload failed: 500 Server Error\n",
            )
            timeout_exec = subprocess.TimeoutExpired(
                cmd=["colab", "exec"],
                timeout=30.0,
            )
            mock_run.side_effect = [err_upload, timeout_exec]

            with self.assertRaises(UpstreamCliError) as ctx:
                transport.upload_file(self.session_id, local_file, "/content/workspace/models/train.py")
            self.assertEqual(mock_run.call_count, 2)  # upload + probe, NO retry
            self.assertIn("Upload failed: 500 Server Error", str(ctx.exception))

    def test_upload_scenario_9_repeated_500_after_recovery_retries_at_most_once(self):
        """9. Repeated HTTP 500 after recovery -> retries at most once, raises UpstreamCliError with original_upload_error and recovery_attempted=True in details."""
        transport = OfficialCliTransport(registry=self.registry, cli_path="/usr/bin/colab")
        local_file = Path(self.tmp_dir) / "train.py"
        local_file.write_text("print('ok')", encoding="utf-8")

        with patch("subprocess.run") as mock_run:
            err_upload1 = subprocess.CalledProcessError(
                returncode=1,
                cmd=["colab", "upload"],
                stderr="[colab] Upload failed: 500 Server Error (first try)\n",
            )
            probe_out = 'COLAB_MKDIR_STATUS:{"existed": false, "isdir": true}\n'
            ok_exec = MagicMock(returncode=0, stdout=probe_out, stderr="")
            err_upload2 = subprocess.CalledProcessError(
                returncode=1,
                cmd=["colab", "upload"],
                stderr="[colab] Upload failed: 500 Server Error (second try)\n",
            )
            mock_run.side_effect = [err_upload1, ok_exec, err_upload2]

            with self.assertRaises(UpstreamCliError) as ctx:
                transport.upload_file(self.session_id, local_file, "/content/workspace/models/train.py")
            self.assertEqual(mock_run.call_count, 3)  # upload + probe + retry (at most once)
            self.assertTrue(ctx.exception.details.get("recovery_attempted"))
            self.assertIn("500 Server Error (first try)", ctx.exception.details.get("original_upload_error", ""))

    def test_upload_scenario_10_unmanaged_session_rejected_immediately(self):
        """10. Unmanaged session -> rejected immediately with UnmanagedSessionError before any CLI call."""
        transport = OfficialCliTransport(registry=self.registry, cli_path="/usr/bin/colab")
        local_file = Path(self.tmp_dir) / "train.py"
        local_file.write_text("print('ok')", encoding="utf-8")

        with patch("subprocess.run") as mock_run:
            with self.assertRaises(UnmanagedSessionError):
                transport.upload_file("unregistered-session-xyz", local_file, "/content/workspace/train.py")
            mock_run.assert_not_called()

    def test_upload_scenario_11_normal_successful_upload(self):
        """11. Normal successful upload -> single CLI call, zero exec_code calls."""
        transport = OfficialCliTransport(registry=self.registry, cli_path="/usr/bin/colab")
        local_file = Path(self.tmp_dir) / "train.py"
        local_file.write_text("print('ok')", encoding="utf-8")

        with patch("subprocess.run") as mock_run:
            ok_upload = MagicMock(returncode=0, stdout="[colab] Uploaded 'train.py'\n", stderr="")
            mock_run.return_value = ok_upload

            transport.upload_file(self.session_id, local_file, "/content/workspace/train.py")
            self.assertEqual(mock_run.call_count, 1)
            args = mock_run.call_args[0][0]
            self.assertIn("upload", args)
            self.assertNotIn("exec", args)

    def test_upload_scenario_12_no_implicit_vm_allocation(self):
        """12. No implicit VM allocation -> verify new or run is never present in any CLI invocation across all failure modes."""
        transport = OfficialCliTransport(registry=self.registry, cli_path="/usr/bin/colab")
        local_file = Path(self.tmp_dir) / "train.py"
        local_file.write_text("print('ok')", encoding="utf-8")

        sequences = [
            # 1: Successful upload
            [MagicMock(returncode=0, stdout="ok", stderr="")],
            # 2: 500 then successful recovery
            [
                subprocess.CalledProcessError(1, ["colab", "upload"], stderr="500 Server Error\n"),
                MagicMock(returncode=0, stdout='COLAB_MKDIR_STATUS:{"existed": false, "isdir": true}\n', stderr=""),
                MagicMock(returncode=0, stdout="ok", stderr=""),
            ],
            # 3: Repeated 500
            [
                subprocess.CalledProcessError(1, ["colab", "upload"], stderr="500 Server Error\n"),
                MagicMock(returncode=0, stdout='COLAB_MKDIR_STATUS:{"existed": false, "isdir": true}\n', stderr=""),
                subprocess.CalledProcessError(1, ["colab", "upload"], stderr="500 Server Error\n"),
            ],
            # 4: Directory already exists
            [
                subprocess.CalledProcessError(1, ["colab", "upload"], stderr="500 Server Error\n"),
                MagicMock(returncode=0, stdout='COLAB_MKDIR_STATUS:{"existed": true, "isdir": true}\n', stderr=""),
            ],
            # 5: Directory creation fails
            [
                subprocess.CalledProcessError(1, ["colab", "upload"], stderr="500 Server Error\n"),
                subprocess.CalledProcessError(1, ["colab", "exec"], stderr="exec failed\n"),
            ],
        ]

        for seq in sequences:
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = seq
                try:
                    transport.upload_file(self.session_id, local_file, "/content/workspace/sub/train.py")
                except Exception:
                    pass

                self.assertGreater(mock_run.call_count, 0)
                for call in mock_run.call_args_list:
                    cmd_args = call[0][0]
                    self.assertNotIn("new", cmd_args)
                    self.assertNotIn("run", cmd_args)


if __name__ == "__main__":
    unittest.main()
