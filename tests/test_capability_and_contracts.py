"""Offline tests for V2 Upstream CLI compatibility probe, error taxonomy and SSH safety guard."""
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from colab_adapter.capability import (
    CapabilityProbe,
    CliCapabilities,
    SshSafetyGuard,
    _strip_ansi,
    probe_cli_capabilities,
)
from colab_adapter.errors import (
    ArtifactValidationError,
    AutoAllocationBlockedError,
    ColabV2Error,
    ManifestValidationError,
    SessionNotFoundError,
    SshAutoAllocationRiskError,
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

    def test_secret_redaction_in_errors(self):
        secret_token = "ya29.a0AfH6SMB_secret_access_token_12345"
        oauth_code = "4/0AfH6SMB_authorization_code_abcdef"
        err = UpstreamCliError(
            f"Failed with token {secret_token}",
            details={
                "cmd": ["colab", "--token", secret_token, "--auth", oauth_code],
                "api_key": "super_secret_key_999",
            },
        )
        msg = str(err)
        d = err.to_dict()
        self.assertNotIn(secret_token, msg)
        self.assertNotIn(secret_token, str(d))
        self.assertNotIn(oauth_code, str(d))
        self.assertIn("[REDACTED]", msg)
        self.assertEqual(d["details"]["api_key"], "[REDACTED]")

    def test_capability_probe_offline_with_mocked_cli(self):
        mock_cli_path = "/usr/local/bin/colab"

        def fake_run(cmd, capture_output=True, text=True, check=True, timeout=10.0):
            res = MagicMock()
            res.returncode = 0
            if "version" in cmd:
                res.stdout = "Version: 0.7.5\n"
            elif "--help" in cmd and "ssh" not in cmd:
                res.stdout = """\x1b[1mColab CLI\x1b[0m
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
            self.assertIsInstance(caps.available_commands, tuple)
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

    def test_strip_ansi_helper(self):
        raw = "\x1b[31;1mError: \x1b[0mSomething went wrong \x1b[32m[OK]\x1b[0m"
        clean = _strip_ansi(raw)
        self.assertEqual(clean, "Error: Something went wrong [OK]")

    def test_live_installed_colab_cli_probe_if_available(self):
        real_colab = shutil.which("colab")
        if real_colab and os.path.isfile(real_colab):
            probe = CapabilityProbe(cli_path=real_colab)
            caps = probe.probe()
            self.assertIn(".", caps.version)
            self.assertIsInstance(caps.available_commands, tuple)
            self.assertIn("upload", caps.available_commands)
            self.assertIn("download", caps.available_commands)
            self.assertIn("exec", caps.available_commands)
            self.assertIn("sessions", caps.available_commands)
            self.assertTrue(caps.supports_ssh_proxy_mode)

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
        self.registry.register(
            session_id="managed-session-123",
            job_id="job-1",
            gpu="T4",
            project_name="proj",
        )
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
        Path(self.config_path).write_text(
            json.dumps({session_id: {"name": session_id, "endpoint": "https://foo"}}),
            encoding="utf-8",
        )
        guard = SshSafetyGuard(
            registry=self.registry,
            colab_config_path=self.config_path,
            allow_ssh=True,
        )
        # Passes precheck
        guard.check_ssh_allowed_for_session(session_id)

    def test_ssh_safety_guard_builds_safe_invocation_without_s_flag(self):
        session_id = "safe-session-456"
        self.registry.register(
            session_id=session_id,
            job_id="job-3",
            gpu="T4",
            project_name="proj",
        )
        session_data = {"name": session_id, "endpoint": "https://foo", "token": "tok"}
        Path(self.config_path).write_text(
            json.dumps({session_id: session_data}),
            encoding="utf-8",
        )
        guard = SshSafetyGuard(
            registry=self.registry,
            colab_config_path=self.config_path,
            allow_ssh=True,
            cli_path="/usr/bin/colab",
        )
        isolated_dir = tempfile.mkdtemp(dir=self.tmp_dir)
        cmd = guard.build_safe_ssh_proxy_invocation(session_id, isolated_dir)

        # Invariant checks:
        # 1. Must NOT include "-s" or "--session"
        self.assertNotIn("-s", cmd, "Safe invocation MUST NOT pass '-s' to avoid auto-allocation")
        self.assertNotIn("--session", cmd, "Safe invocation MUST NOT pass '--session'")
        # 2. Must include --config pointing to the isolated single-session config
        self.assertIn("--config", cmd)
        config_idx = cmd.index("--config") + 1
        isolated_config_file = cmd[config_idx]
        self.assertTrue(os.path.isfile(isolated_config_file))
        # 3. Isolated config must contain ONLY the single session
        isolated_content = json.loads(Path(isolated_config_file).read_text(encoding="utf-8"))
        self.assertEqual(list(isolated_content.keys()), [session_id])
        # 4. Command must specify ssh --proxy-mode
        self.assertIn("ssh", cmd)
        self.assertIn("--proxy-mode", cmd)

    def test_adversarial_session_disappearance_toctou_upstream_behavior(self):
        """Proves that upstream _select_proxy_session(session=None) fails closed
        if the session disappears, whereas passing session='NAME' would call _auto_create_session.
        """
        import sys
        if sys.version_info < (3, 10):
            self.skipTest("Upstream colab_cli requires Python >=3.10 (uses pattern matching)")

        uv_path = Path("/mnt/ssd/genos2/.local/share/uv/tools/google-colab-cli/lib/python3.12/site-packages")
        if uv_path.exists() and str(uv_path) not in sys.path:
            sys.path.insert(0, str(uv_path))

        try:
            from colab_cli.commands import ssh as upstream_ssh
        except ImportError:
            self.skipTest("colab_cli not installed in this Python environment")

        # Case A: When session='missing_name' is passed to upstream _select_proxy_session:
        with patch.object(upstream_ssh, "_session_exists", return_value=False), \
             patch.object(upstream_ssh, "_auto_create_session") as mock_auto_create:
            mock_auto_create.return_value = MagicMock()
            upstream_ssh._select_proxy_session("missing_name", gpu="T4", tpu=None)
            self.assertTrue(
                mock_auto_create.called,
                "Confirmed vulnerability: passing '-s missing_name' triggers _auto_create_session!"
            )

        # Case B: When session=None is passed (our safe invocation without '-s'):
        with patch.object(upstream_ssh, "_auto_create_session") as mock_auto_create_b, \
             patch.object(upstream_ssh, "_resolve_session") as mock_resolve:
            mock_resolve.side_effect = SystemExit(2)
            with self.assertRaises(SystemExit):
                upstream_ssh._select_proxy_session(None, gpu="T4", tpu=None)
            self.assertFalse(
                mock_auto_create_b.called,
                "Verified invariant: omitting '-s' NEVER calls _auto_create_session!"
            )

    def test_cli_capabilities_available_commands_immutable_tuple(self):
        """Verify CliCapabilities converts available_commands to immutable tuple, preventing .append()."""
        caps = CliCapabilities(
            cli_path="/usr/bin/colab",
            version="1.0.0",
            available_commands=["download", "exec", "new", "run", "ssh"],
            supports_ssh=True,
            supports_ssh_proxy_mode=True,
            supports_upload=True,
            supports_download=True,
            supports_exec=True,
            supports_sessions=True,
            supports_usage=True,
        )
        self.assertIsInstance(caps.available_commands, tuple)
        self.assertEqual(caps.available_commands, ("download", "exec", "new", "run", "ssh"))
        with self.assertRaises(AttributeError):
            caps.available_commands.append("unauthorized_command")

    def test_probe_cli_capabilities_strips_ansi(self):
        """Verify probe_cli_capabilities strips ANSI escape codes from rich terminal output."""
        def fake_run(cmd, **kwargs):
            res = MagicMock()
            res.returncode = 0
            if "version" in cmd:
                res.stdout = "\x1b[32;1mVersion: 0.8.0\x1b[0m\n"
            elif "--help" in cmd and "ssh" not in cmd:
                res.stdout = "\x1b[1mColab CLI\x1b[0m\n╭─ \x1b[34mCommands\x1b[0m ─╮\n│ \x1b[33mupload\x1b[0m   Upload file │\n│ \x1b[33mexec\x1b[0m     Exec code   │\n│ \x1b[33mssh\x1b[0m      SSH bridge  │\n╰─────────────╯\n"
            elif "ssh" in cmd and "--help" in cmd:
                res.stdout = "\x1b[36m--proxy-mode\x1b[0m"
            else:
                res.stdout = ""
            return res

        with patch("os.path.isfile", return_value=True), patch("subprocess.run", side_effect=fake_run):
            caps = probe_cli_capabilities(cli_path="/usr/bin/colab")
            self.assertEqual(caps.version, "0.8.0")
            self.assertIn("upload", caps.available_commands)
            self.assertIn("exec", caps.available_commands)
            for cmd_name in caps.available_commands:
                self.assertNotIn("\x1b", cmd_name, f"Command '{cmd_name}' contains ANSI escape sequence")
            self.assertTrue(caps.supports_ssh_proxy_mode)

    def test_prepare_isolated_ssh_config_permissions_and_invariants(self):
        """Verify prepare_isolated_ssh_config creates a 0700 private directory, 0600 sessions.json with only target session, and copies auth config."""
        session_id = "test-isolated-session-789"
        self.registry.register(
            session_id=session_id,
            job_id="job-isolated-1",
            gpu="T4",
            project_name="proj-isolated",
        )
        session_data = {"name": session_id, "endpoint": "https://colab.google.com/session/789", "token": "tok789"}
        Path(self.config_path).write_text(
            json.dumps({session_id: session_data, "other-session": {"name": "other-session"}}),
            encoding="utf-8",
        )
        # Create non-session settings.json in same dir as config
        settings_path = Path(self.config_path).parent / "settings.json"
        settings_path.write_text(json.dumps({"theme": "dark", "client_version": "1.0.0"}), encoding="utf-8")

        guard = SshSafetyGuard(
            registry=self.registry,
            colab_config_path=self.config_path,
            allow_ssh=True,
            cli_path="/usr/bin/colab",
        )

        iso_dir = guard.prepare_isolated_ssh_config(session_id)
        self.assertTrue(iso_dir.is_dir())

        # Check directory permissions: 0700
        dir_mode = stat.S_IMODE(os.stat(iso_dir).st_mode)
        self.assertEqual(dir_mode, 0o700, f"Expected 0700 permissions on isolated directory, got {oct(dir_mode)}")

        # Check sessions.json existence and 0600 permissions
        sessions_json = iso_dir / "sessions.json"
        self.assertTrue(sessions_json.is_file())
        file_mode = stat.S_IMODE(os.stat(sessions_json).st_mode)
        self.assertEqual(file_mode, 0o600, f"Expected 0600 permissions on sessions.json, got {oct(file_mode)}")

        # Check contents: ONLY target session
        content = json.loads(sessions_json.read_text(encoding="utf-8"))
        self.assertEqual(list(content.keys()), [session_id])
        self.assertNotIn("other-session", content)

        # Check non-session config copied safely
        iso_settings = iso_dir / "settings.json"
        self.assertTrue(iso_settings.is_file())
        self.assertEqual(stat.S_IMODE(os.stat(iso_settings).st_mode), 0o600)

    def test_session_disappearance_immediately_after_preflight_blocks_with_zero_allocation(self):
        """Verify TOCTOU race: if session disappears from local store immediately after preflight, invocation fails closed with SshAutoAllocationRiskError and zero allocations."""
        session_id = "toctou-target-session"
        self.registry.register(
            session_id=session_id,
            job_id="job-toctou",
            gpu="T4",
            project_name="proj-toctou",
        )
        session_data = {"name": session_id, "endpoint": "https://colab.google.com/session/toctou"}
        Path(self.config_path).write_text(
            json.dumps({session_id: session_data}),
            encoding="utf-8",
        )

        guard = SshSafetyGuard(
            registry=self.registry,
            colab_config_path=self.config_path,
            allow_ssh=True,
            cli_path="/usr/bin/colab",
        )

        # Preflight check passes
        guard.check_ssh_allowed_for_session(session_id)

        # Adversarial race: session file is pruned right after preflight check
        Path(self.config_path).write_text(json.dumps({}), encoding="utf-8")

        # Invocation must detect session disappearance immediately and fail closed
        with self.assertRaises(SshAutoAllocationRiskError) as ctx:
            guard.build_safe_ssh_proxy_invocation(session_id, isolated_dir=tempfile.mkdtemp(dir=self.tmp_dir))
        self.assertIn("CRITICAL SAFETY VIOLATION PREVENTED", str(ctx.exception))



if __name__ == "__main__":
    unittest.main()
