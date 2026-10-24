"""Upstream CLI capability detection and SSH safety guard.

Provides:
1. CapabilityProbe: offline inspection of google-colab-cli version and feature flags (zero VM allocation).
2. SshSafetyGuard: strict invariant checks preventing 'colab ssh' from auto-allocating a missing session.
"""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Set

from .errors import (
    AutoAllocationBlockedError,
    TransportUnavailableError,
    UnmanagedSessionError,
    UpstreamCliError,
    UpstreamCliNotFoundError,
)
from .provider import ManagedSessionRegistry


@dataclass(frozen=True)
class CliCapabilities:
    """Immutable record of detected upstream CLI capabilities."""

    cli_path: str
    version: str
    available_commands: List[str]
    supports_ssh: bool
    supports_ssh_proxy_mode: bool
    supports_upload: bool
    supports_download: bool
    supports_exec: bool
    supports_sessions: bool
    supports_usage: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CapabilityProbe:
    """Inspects the local google-colab-cli installation without allocating resources."""

    def __init__(self, cli_path: Optional[str] = None):
        self.cli_path = cli_path or shutil.which("colab")

    def probe(self) -> CliCapabilities:
        """Probe the CLI binary offline. Never calls run, new, or ssh."""
        if not self.cli_path or not os.path.isfile(self.cli_path):
            raise UpstreamCliNotFoundError(
                f"google-colab-cli executable not found at '{self.cli_path}'."
            )

        version = self._get_version()
        commands = self._get_commands()
        supports_ssh = "ssh" in commands
        supports_ssh_proxy_mode = self._check_ssh_proxy_mode() if supports_ssh else False

        return CliCapabilities(
            cli_path=self.cli_path,
            version=version,
            available_commands=sorted(commands),
            supports_ssh=supports_ssh,
            supports_ssh_proxy_mode=supports_ssh_proxy_mode,
            supports_upload="upload" in commands,
            supports_download="download" in commands,
            supports_exec="exec" in commands,
            supports_sessions="sessions" in commands,
            supports_usage="usage" in commands,
        )

    def _run_cmd(self, args: List[str], timeout: float = 10.0) -> str:
        cmd = [self.cli_path] + args
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout,
            )
            return res.stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise UpstreamCliError(
                f"Failed to invoke '{' '.join(cmd)}': {e}",
                details={"cmd": cmd, "stderr": getattr(e, "stderr", None)},
            )

    def _get_version(self) -> str:
        out = self._run_cmd(["version"])
        match = re.search(r"Version:\s*([0-9a-zA-Z\.\-]+)", out)
        if match:
            return match.group(1).strip()
        return out.strip() or "unknown"

    def _get_commands(self) -> Set[str]:
        out = self._run_cmd(["--help"])
        # Commands are listed in a table / section: "│ command_name" or "  command_name  "
        commands = set()
        in_commands = False
        for line in out.splitlines():
            if "Commands" in line:
                in_commands = True
                continue
            if in_commands:
                # Format is typically "│ console         Connect to..." or "  console   ..."
                match = re.search(r"[│\s]+([a-z][a-z0-9\-]+)\s{2,}", line)
                if match:
                    cmd_name = match.group(1).strip()
                    commands.add(cmd_name)
        return commands

    def _check_ssh_proxy_mode(self) -> bool:
        try:
            out = self._run_cmd(["ssh", "--help"])
            return "--proxy-mode" in out
        except UpstreamCliError:
            return False


class SshSafetyGuard:
    """Guards against upstream 'colab ssh' auto-allocating missing sessions."""

    def __init__(
        self,
        registry: ManagedSessionRegistry,
        colab_config_path: Optional[str] = None,
        allow_ssh: bool = False,
    ):
        self.registry = registry
        self.config_path = Path(
            colab_config_path
            or os.environ.get("COLAB_CLI_CONFIG")
            or os.path.expanduser("~/.config/colab-cli/sessions.json")
        )
        self.allow_ssh = allow_ssh

    def check_ssh_allowed_for_session(self, session_name: str) -> None:
        """Verifies that an SSH connection to session_name will not auto-allocate a VM.

        Raises:
            TransportUnavailableError: If SSH transport is globally disabled.
            UnmanagedSessionError: If the session is not owned in the adapter registry.
            AutoAllocationBlockedError: If the session is missing from colab-cli store,
                which would trigger upstream CLI auto-provisioning.
        """
        if not self.allow_ssh:
            raise TransportUnavailableError(
                "SSH transport is disabled for this project workspace. "
                "Use the verified official CLI fallback transport."
            )

        if not self.registry.is_managed(session_name):
            raise UnmanagedSessionError(
                f"Refusing SSH transport for unmanaged session '{session_name}'. "
                "The adapter only interacts with its own managed sessions."
            )

        # Check colab-cli local session store file
        if not self._session_in_colab_cli_store(session_name):
            raise AutoAllocationBlockedError(
                f"CRITICAL SAFETY VIOLATION PREVENTED: Session '{session_name}' is not found "
                f"in colab-cli local store ({self.config_path}). Upstream 'colab ssh -s {session_name}' "
                "would automatically provision a new Google Colab VM without authorization! "
                "Operation blocked."
            )

    def _session_in_colab_cli_store(self, session_name: str) -> bool:
        if not self.config_path.exists():
            return False
        try:
            content = self.config_path.read_text(encoding="utf-8")
            if not content.strip():
                return False
            data = json.loads(content)
            return session_name in data
        except Exception:
            # If the config file cannot be safely read, fail closed!
            return False
