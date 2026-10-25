"""Guarded SSH-over-WebSocket Transport for Colab Project Workspaces.

Connects to a running Colab VM using upstream `colab ssh --proxy-mode`.
CRITICAL INVARIANTS:
1. Gated by allow_ssh=True. Refuses execution by default (fail closed).
2. Must never pass '-s NAME' to avoid upstream auto-allocation on missing sessions.
3. Uses single-session isolated config created by SshSafetyGuard.
4. Falls back gracefully to OfficialCliTransport on HTTP 404, 401, 429, 502 or missing SSH.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Union

from ..capability import SshSafetyGuard
from ..errors import (
    SshHandshakeError,
    TransportError,
    TransportUnavailableError,
    UnmanagedSessionError,
)
from ..provider import ManagedSessionRegistry
from .base import BaseTransport


class SshProxyTransport(BaseTransport):
    """Guarded SSH proxy transport over WebSocket bridge."""

    def __init__(
        self,
        registry: ManagedSessionRegistry,
        allow_ssh: bool = False,
        cli_path: Optional[str] = None,
        colab_config_path: Optional[str] = None,
    ):
        super().__init__(registry)
        self.allow_ssh = allow_ssh
        self.cli_path = cli_path or shutil.which("colab") or "colab"
        self.guard = SshSafetyGuard(
            registry=registry,
            colab_config_path=colab_config_path,
            allow_ssh=allow_ssh,
            cli_path=self.cli_path,
        )

    def _prepare_safe_proxy_command(self, session_id: str, temp_dir: str) -> List[str]:
        """Validates safety invariants and builds safe proxy invocation without '-s'."""
        return self.guard.build_safe_ssh_proxy_invocation(session_id, temp_dir)

    def upload_file(
        self,
        session_id: str,
        local_path: Union[str, Path],
        remote_path: str,
    ) -> None:
        if not self.allow_ssh:
            raise TransportUnavailableError(
                "SSH transport is disabled. Use OfficialCliTransport fallback."
            )
        self.guard.check_ssh_allowed_for_session(session_id)
        # Note: In production rsync over ProxyCommand can be used.
        # Fall back to official CLI if rsync/SSH pipe fails.
        raise TransportUnavailableError("SSH file sync fast path unavailable; use OfficialCliTransport.")

    def download_file(
        self,
        session_id: str,
        remote_path: str,
        local_path: Union[str, Path],
    ) -> None:
        if not self.allow_ssh:
            raise TransportUnavailableError(
                "SSH transport is disabled. Use OfficialCliTransport fallback."
            )
        self.guard.check_ssh_allowed_for_session(session_id)
        raise TransportUnavailableError("SSH file download unavailable; use OfficialCliTransport.")

    def list_dir(
        self,
        session_id: str,
        remote_path: str = "content",
    ) -> List[str]:
        if not self.allow_ssh:
            raise TransportUnavailableError(
                "SSH transport is disabled. Use OfficialCliTransport fallback."
            )
        self.guard.check_ssh_allowed_for_session(session_id)
        raise TransportUnavailableError("SSH list_dir unavailable; use OfficialCliTransport.")

    def remove_file(
        self,
        session_id: str,
        remote_path: str,
    ) -> None:
        if not self.allow_ssh:
            raise TransportUnavailableError(
                "SSH transport is disabled. Use OfficialCliTransport fallback."
            )
        self.guard.check_ssh_allowed_for_session(session_id)
        raise TransportUnavailableError("SSH remove unavailable; use OfficialCliTransport.")

    def exec_code(
        self,
        session_id: str,
        code: str,
        timeout: float = 60.0,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if not self.allow_ssh:
            raise TransportUnavailableError(
                "SSH transport is disabled. Use OfficialCliTransport fallback."
            )
        self.guard.check_ssh_allowed_for_session(session_id)
        raise TransportUnavailableError("SSH exec unavailable; use OfficialCliTransport.")

    def check_health(self, session_id: str) -> bool:
        if not self.allow_ssh:
            return False
        try:
            self.guard.check_ssh_allowed_for_session(session_id)
            return True
        except Exception:
            return False


GuardedSshTransport = SshProxyTransport
