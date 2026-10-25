"""Official CLI Transport for Colab Project Workspaces.

Uses Google's official google-colab-cli commands:
- colab upload -s <session> <local> <remote>
- colab download -s <session> <remote> <local>
- colab exec -s <session> -f <script>
- colab ls -s <session> <path>
- colab rm -s <session> <path>

Guaranteed fail-closed: upstream commands fail with exit code 1 if session is missing;
they NEVER auto-allocate a new VM.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Union

from ..capability import _strip_ansi
from ..errors import (
    RemoteExecutionError,
    SessionNotFoundError,
    TransportError,
    UnmanagedSessionError,
    UpstreamCliError,
    UpstreamCliNotFoundError,
)
from ..provider import ManagedSessionRegistry
from .base import BaseTransport


class OfficialCliTransport(BaseTransport):
    """Primary transport implementation backed by official google-colab-cli."""

    def __init__(
        self,
        registry: ManagedSessionRegistry,
        cli_path: Optional[str] = None,
        auth_mode: str = "oauth2",
    ):
        super().__init__(registry)
        self.cli_path = cli_path or shutil.which("colab") or "colab"
        self.auth_mode = auth_mode

    def _verify_session_managed(self, session_id: str) -> None:
        if not self.registry.is_managed(session_id):
            raise UnmanagedSessionError(
                f"Refusing transport operation for unmanaged session '{session_id}'. "
                "The adapter only interacts with its own registered sessions."
            )

    def _run_cmd(
        self,
        args: List[str],
        timeout: float = 60.0,
        stdin_input: Optional[str] = None,
    ) -> str:
        cmd = [self.cli_path, f"--auth={self.auth_mode}"] + args
        try:
            res = subprocess.run(
                cmd,
                input=stdin_input,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout,
            )
            return _strip_ansi(res.stdout)
        except subprocess.TimeoutExpired as e:
            raise TransportError(
                f"Command timed out after {timeout}s: {' '.join(args[:3])}",
                details={"args": args[:3], "timeout": timeout},
            )
        except subprocess.CalledProcessError as e:
            stdout = _strip_ansi(e.stdout or "")
            stderr = _strip_ansi(e.stderr or "")
            combined = f"{stdout}\n{stderr}".strip()

            if "not found" in combined.lower() or "no active sessions found" in combined.lower():
                raise SessionNotFoundError(
                    f"Session not found in Colab: {combined}",
                    details={"args": args[:3], "output": combined},
                )
            raise UpstreamCliError(
                f"Upstream CLI command failed with code {e.returncode}: {combined}",
                details={"returncode": e.returncode, "args": args[:3], "output": combined},
            )

    def upload_file(
        self,
        session_id: str,
        local_path: Union[str, Path],
        remote_path: str,
    ) -> None:
        self._verify_session_managed(session_id)
        local_p = Path(local_path).resolve()
        if not local_p.is_file():
            raise TransportError(f"Local file to upload does not exist: {local_p}")

        args = ["upload", "-s", session_id, str(local_p), remote_path]
        self._run_cmd(args, timeout=300.0)

    def download_file(
        self,
        session_id: str,
        remote_path: str,
        local_path: Union[str, Path],
    ) -> None:
        self._verify_session_managed(session_id)
        local_p = Path(local_path).resolve()
        local_p.parent.mkdir(parents=True, exist_ok=True)

        args = ["download", "-s", session_id, remote_path, str(local_p)]
        self._run_cmd(args, timeout=300.0)

    def list_dir(
        self,
        session_id: str,
        remote_path: str = "content",
    ) -> List[str]:
        self._verify_session_managed(session_id)
        args = ["ls", "-s", session_id, remote_path]
        out = self._run_cmd(args, timeout=30.0)
        items = []
        for line in out.splitlines():
            line = line.strip()
            if line and not line.startswith("[colab]"):
                items.append(line)
        return items

    def remove_file(
        self,
        session_id: str,
        remote_path: str,
    ) -> None:
        self._verify_session_managed(session_id)
        args = ["rm", "-s", session_id, remote_path]
        self._run_cmd(args, timeout=30.0)

    def exec_code(
        self,
        session_id: str,
        code: str,
        timeout: float = 60.0,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        self._verify_session_managed(session_id)

        # Write code to temporary script to pass with -f
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
            tf.write(code)
            tmp_script = tf.name

        try:
            args = ["exec", "-s", session_id, "-f", tmp_script]
            if env:
                for k, v in env.items():
                    args.extend(["--env", f"{k}={v}"])

            out = self._run_cmd(args, timeout=timeout)
            # Detect remote traceback or failure in output
            has_traceback = "Traceback (most recent call last):" in out
            return {
                "success": not has_traceback,
                "output": out,
                "has_traceback": has_traceback,
            }
        finally:
            if os.path.exists(tmp_script):
                os.remove(tmp_script)

    def check_health(self, session_id: str) -> bool:
        try:
            self._verify_session_managed(session_id)
            self.list_dir(session_id, "content")
            return True
        except Exception:
            return False
