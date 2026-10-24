"""Upstream CLI capability detection and SSH safety guard.

Provides:
1. CapabilityProbe: offline inspection of google-colab-cli version and feature flags (zero VM allocation).
2. SshSafetyGuard: strict fail-closed invariants preventing 'colab ssh' from auto-allocating a missing session.
"""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .errors import (
    AutoAllocationBlockedError,
    TransportUnavailableError,
    UnmanagedSessionError,
    UpstreamCliError,
    UpstreamCliNotFoundError,
)
from .provider import ManagedSessionRegistry

_ANSI_REGEX = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences from rich/colored terminal outputs."""
    return _ANSI_REGEX.sub("", text)


@dataclass(frozen=True)
class CliCapabilities:
    """Immutable record of detected upstream CLI capabilities."""

    cli_path: str
    version: str
    available_commands: Tuple[str, ...]
    supports_ssh: bool
    supports_ssh_proxy_mode: bool
    supports_upload: bool
    supports_download: bool
    supports_exec: bool
    supports_sessions: bool
    supports_usage: bool

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["available_commands"] = list(self.available_commands)
        return d


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
            available_commands=tuple(sorted(commands)),
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
            return _strip_ansi(res.stdout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            stderr = _strip_ansi(getattr(e, "stderr", "") or "")
            raise UpstreamCliError(
                f"Failed to invoke '{' '.join(args)}': {e}",
                details={"args": args, "stderr": stderr},
            )

    def _get_version(self) -> str:
        out = self._run_cmd(["version"])
        match = re.search(r"Version:\s*([0-9a-zA-Z\.\-]+)", out)
        if match:
            return match.group(1).strip()
        return out.strip() or "unknown"

    def _get_commands(self) -> Set[str]:
        out = self._run_cmd(["--help"])
        commands = set()
        in_commands = False
        for line in out.splitlines():
            if "Commands" in line:
                in_commands = True
                continue
            if in_commands:
                # Format typically: "│ command-name    Description" or "  command-name  Description"
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
    """Enforces fail-closed invariants preventing 'colab ssh' from auto-allocating a session.

    CRITICAL ARCHITECTURAL FINDING:
    Upstream 'colab ssh --proxy-mode -s NAME' checks if NAME is in the session store;
    if absent, it invokes `_auto_create_session(name=NAME)` which creates a remote VM!
    A preflight existence check alone cannot eliminate TOCTOU race conditions.

    SAFE EXECUTION INVARIANT:
    To ensure ZERO chance of auto-allocation:
    1. Direct 'colab ssh -s NAME' is STRICTLY FORBIDDEN.
    2. Proxy mode MUST be invoked WITHOUT '-s': 'colab --config <isolated> ssh --proxy-mode'.
    3. The isolated config MUST contain only the approved, managed session entry.
    4. When '-s' is omitted, upstream '_select_proxy_session(session=None)' resolves the unique
       active session. If the session has disappeared (or is empty), it fails closed with Exit(1)
       without ever calling '_auto_create_session()'.
    5. SSH transport remains DISABLED by default (allow_ssh=False) for autonomous execution,
       using official CLI upload/exec transport as the fail-closed fallback.
    """

    def __init__(
        self,
        registry: ManagedSessionRegistry,
        colab_config_path: Optional[str] = None,
        allow_ssh: bool = False,
        cli_path: Optional[str] = None,
    ):
        self.registry = registry
        self.config_path = Path(
            colab_config_path
            or os.environ.get("COLAB_CLI_CONFIG")
            or os.path.expanduser("~/.config/colab-cli/sessions.json")
        )
        self.allow_ssh = allow_ssh
        self.cli_path = cli_path or shutil.which("colab") or "colab"

    def check_ssh_allowed_for_session(self, session_name: str) -> None:
        """Verifies session ownership and local store presence.

        Note: A precheck alone is not a guarantee if passing '-s NAME'.
        Use `build_safe_ssh_proxy_invocation` to prevent TOCTOU auto-allocation.
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

    def build_safe_ssh_proxy_invocation(
        self,
        session_name: str,
        isolated_dir: Union[str, Path],
    ) -> List[str]:
        """Builds a fail-closed, race-free SSH proxy command without '-s'.

        Creates an isolated session config file containing ONLY the single approved session.
        Invokes: ['colab', '--config', isolated_config_path, 'ssh', '--proxy-mode']
        
        Because '-s' is NOT passed, upstream code resolves session=None, which CANNOT
        enter the `_auto_create_session` branch if the session was pruned or missing.
        """
        self.check_ssh_allowed_for_session(session_name)

        session_entry = self._get_session_data(session_name)
        if not session_entry:
            raise AutoAllocationBlockedError(
                f"Session '{session_name}' data could not be retrieved from local store."
            )

        isolated_path = Path(isolated_dir) / f"isolated_colab_session_{session_name}.json"
        # Write single session store
        isolated_data = {session_name: session_entry}
        isolated_path.write_text(json.dumps(isolated_data, indent=2), encoding="utf-8")

        # Command intentionally OMITS '-s NAME' to prevent upstream auto-creation!
        return [
            self.cli_path,
            "--config",
            str(isolated_path.resolve()),
            "ssh",
            "--proxy-mode",
        ]

    def _session_in_colab_cli_store(self, session_name: str) -> bool:
        data = self._read_config_data()
        return session_name in data

    def _get_session_data(self, session_name: str) -> Optional[Dict[str, Any]]:
        data = self._read_config_data()
        return data.get(session_name)

    def _read_config_data(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {}
        try:
            content = self.config_path.read_text(encoding="utf-8")
            if not content.strip():
                return {}
            return json.loads(content)
        except Exception:
            # Corrupted or unreadable config fails closed!
            return {}
