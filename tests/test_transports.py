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


if __name__ == "__main__":
    unittest.main()
