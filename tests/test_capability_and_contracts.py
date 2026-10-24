"""Offline tests for V2 Upstream CLI compatibility probe, error taxonomy and SSH safety guard."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from colab_adapter.capability import CapabilityProbe, CliCapabilities, SshSafetyGuard
from colab_adapter.errors import (
    ArtifactValidationError,
    AutoAllocationBlockedError,
    ColabV2Error,
    ManifestValidationError,
    SessionNotFoundError,
    SshHandshakeError,
    TransportError,
    TransportUnavailableError,
    UnmanagedSessionError,
    UpstreamCliError,
    UpstreamCliNotFoundError,
    WorkspaceSyncError,
)
from colab_adapter.provider import (
    ColabProviderError,
    ManagedSessionRegistry,
)


class TestV2ContractsAndCapabilityProbe(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="colab_test_m1_")
        self.reg_path = os.path.join(self.tmp_dir, "test_registry.json")
        self.registry = ManagedSessionRegistry(self.reg_path)
        self.config_path = os.path.join(self.tmp_dir, "sessions.json")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_error_taxonomy_hierarchy_and_serialization(self):
        err = AutoAllocationBlockedError(
            "Auto allocation refused", details={"session": "ghost-session"}
        )
        self.assertIsInstance(err, ColabV2Error)
        self.assertIsInstance(err, ColabProviderError)
        self.assertEqual(err.details["session"], "ghost-session")
        d = err.to_dict()
        self.assertEqual(d["error_type"], "AutoAllocationBlockedError")
        self.assertEqual(d["message"], "Auto allocation refused")
        self.assertEqual(d["details"], {"session": "ghost-session"})

        # Handshake error
        hs_err = SshHandshakeError("Upgrade failed", http_status=404)
        self.assertIsInstance(hs_err, TransportError)
        self.assertEqual(hs_err.http_status, 404)
        self.assertEqual(hs_err.to_dict()["details"]["http_status"], 404)

        # Other error types
        self.assertIsInstance(ManifestValidationError("bad schema"), ColabV2Error)
        self.assertIsInstance(WorkspaceSyncError("sync fail"), ColabV2Error)
        self.assertIsInstance(ArtifactValidationError("missing pt"), ColabV2Error)
        self.assertIsInstance(SessionNotFoundError("no session"), TransportError)

    def test_capability_probe_offline_with_mocked_cli(self):
        mock_cli_path = "/usr/local/bin/colab"

        def fake_run(cmd, capture_output=True, text=True, check=True, timeout=10.0):
            res = MagicMock()
            res.returncode = 0
            if "version" in cmd:
                res.stdout = "Version: 0.7.5\n"
            elif "--help" in cmd and "ssh" not in cmd:
                res.stdout = """
Colab CLI
╭─ Commands ───────────────────╮
│ download   Download a file   │
│ exec       Execute code      │
│ new        Create session    │
│ run        Run script        │
│ sessions   List sessions     │
│ ssh        Connect via SSH   │
│ stop       Stop session      │
│ upload     Upload a file     │
│ usage      Show usage        │
╰──────────────────────────────╯
"""
            elif "ssh" in cmd and "--help" in cmd:
                res.stdout = """
Usage: colab ssh [OPTIONS]
  --proxy-mode  Act as an OpenSSH ProxyCommand-compatible bridge
  --session -s  Session name
"""
            else:
                res.stdout = ""
            return res

        with patch("os.path.isfile", return_value=True), patch(
            "subprocess.run", side_effect=fake_run
        ) as mock_subprocess:
            probe = CapabilityProbe(cli_path=mock_cli_path)
            caps = probe.probe()

            self.assertEqual(caps.version, "0.7.5")
            self.assertTrue(caps.supports_ssh)
            self.assertTrue(caps.supports_ssh_proxy_mode)
            self.assertTrue(caps.supports_upload)
            self.assertTrue(caps.supports_download)
            self.assertTrue(caps.supports_exec)
            self.assertTrue(caps.supports_sessions)
            self.assertTrue(caps.supports_usage)

            # Subprocess spy test: verify NO allocating commands ('new', 'run', or bare 'ssh') were called!
            for call_item in mock_subprocess.call_args_list:
                args = call_item[0][0]
                self.assertNotIn("new", args, "Probing must never call 'colab new'")
                self.assertNotIn("run", args, "Probing must never call 'colab run'")
                if "ssh" in args:
                    self.assertIn(
                        "--help", args, "SSH should only be inspected with --help"
                    )

    def test_capability_probe_missing_binary(self):
        probe = CapabilityProbe(cli_path="/nonexistent/colab_binary")
        with self.assertRaises(UpstreamCliNotFoundError):
            probe.probe()

    def test_ssh_safety_guard_refuses_when_disabled(self):
        guard = SshSafetyGuard(
            registry=self.registry,
            colab_config_path=self.config_path,
            allow_ssh=False,
        )
        with self.assertRaises(TransportUnavailableError) as ctx:
            guard.check_ssh_allowed_for_session("session-1")
        self.assertIn("SSH transport is disabled", str(ctx.exception))

    def test_ssh_safety_guard_refuses_unmanaged_session(self):
        guard = SshSafetyGuard(
            registry=self.registry,
            colab_config_path=self.config_path,
            allow_ssh=True,
        )
        with self.assertRaises(UnmanagedSessionError) as ctx:
            guard.check_ssh_allowed_for_session("unmanaged-session")
        self.assertIn("unmanaged session", str(ctx.exception))

    def test_ssh_safety_guard_blocks_auto_allocation_when_session_missing_from_colab_store(
        self,
    ):
        # Register session in adapter registry
        self.registry.register(
            session_id="managed-session-123",
            job_id="job-1",
            gpu="T4",
            project_name="proj",
        )
        # But sessions.json does not exist or does not contain managed-session-123
        guard = SshSafetyGuard(
            registry=self.registry,
            colab_config_path=self.config_path,
            allow_ssh=True,
        )
        with self.assertRaises(AutoAllocationBlockedError) as ctx:
            guard.check_ssh_allowed_for_session("managed-session-123")
        self.assertIn("CRITICAL SAFETY VIOLATION PREVENTED", str(ctx.exception))
        self.assertIn("would automatically provision a new Google Colab VM", str(ctx.exception))

    def test_ssh_safety_guard_permits_when_registered_and_in_colab_store(self):
        session_id = "valid-managed-session"
        self.registry.register(
            session_id=session_id,
            job_id="job-2",
            gpu="T4",
            project_name="proj",
        )
        # Create valid sessions.json
        Path(self.config_path).write_text(
            json.dumps({session_id: {"name": session_id, "endpoint": "https://foo"}}),
            encoding="utf-8",
        )
        guard = SshSafetyGuard(
            registry=self.registry,
            colab_config_path=self.config_path,
            allow_ssh=True,
        )
        # Should not raise any error
        guard.check_ssh_allowed_for_session(session_id)


if __name__ == "__main__":
    unittest.main()
