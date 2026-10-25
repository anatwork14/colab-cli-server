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
import json
import os
from pathlib import Path, PurePosixPath
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


def _validate_remote_path(remote_path: str) -> str:
    """Validate remote path against empty values, null bytes, newlines, and path traversal."""
    if not isinstance(remote_path, str) or not remote_path.strip():
        raise TransportError(f"Invalid or unsafe remote path: {remote_path!r}")
    if "\x00" in remote_path or "\n" in remote_path or "\r" in remote_path:
        raise TransportError(f"Invalid or unsafe remote path: {remote_path!r}")
    if ".." in PurePosixPath(remote_path).parts:
        raise TransportError(f"Invalid or unsafe remote path: {remote_path!r}")
    return remote_path


EXEC_UPSTREAM_MARGIN_S = 15


class OfficialCliTransport(BaseTransport):
    """Primary transport implementation backed by official google-colab-cli."""

    _validate_remote_path = staticmethod(_validate_remote_path)

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
        remote_path = _validate_remote_path(remote_path)
        local_p = Path(local_path).resolve()
        if not local_p.is_file():
            raise TransportError(f"Local file to upload does not exist: {local_p}")

        args = ["upload", "-s", session_id, str(local_p), remote_path]
        try:
            self._run_cmd(args, timeout=300.0)
        except UpstreamCliError as orig_err:
            err_str = str(orig_err)
            err_lower = err_str.lower()

            is_500 = ("500 server error" in err_lower) or (
                "500" in err_str and ("/api/contents/" in err_str or "upload failed" in err_lower)
            )
            is_disconnect_or_auth = any(
                term in err_lower
                for term in ("not found", "unauthorized", "401", "403", "disconnected", "quota")
            )
            parent_dir = str(PurePosixPath(remote_path).parent)
            is_nontrivial_parent = parent_dir not in ("", ".", "/", "/content", "content")

            if not (is_500 and not is_disconnect_or_auth and is_nontrivial_parent):
                raise orig_err

            self._verify_session_managed(session_id)

            probe_code = (
                "import json, os\n"
                f"p = {parent_dir!r}\n"
                "existed = os.path.isdir(p)\n"
                "if not existed:\n"
                "    os.makedirs(p, exist_ok=True)\n"
                'print("COLAB_MKDIR_STATUS:" + json.dumps({"existed": existed, "isdir": os.path.isdir(p)}))\n'
            )

            try:
                probe_res = self.exec_code(session_id, probe_code, timeout=30.0)
            except SessionNotFoundError:
                raise
            except Exception:
                raise orig_err

            if not isinstance(probe_res, dict) or not probe_res.get("success", False):
                out = probe_res.get("output", "") if isinstance(probe_res, dict) else ""
                if not hasattr(orig_err, "details") or not isinstance(orig_err.details, dict):
                    orig_err.details = {}
                orig_err.details["mkdir_recovery_failed"] = out
                raise orig_err

            output = probe_res.get("output", "")
            if "COLAB_MKDIR_STATUS:" in output:
                try:
                    marker = "COLAB_MKDIR_STATUS:"
                    status_part = output.split(marker, 1)[1].strip()
                    status_line = status_part.splitlines()[0] if status_part else ""
                    status_obj = json.loads(status_line)
                    if status_obj.get("existed") is True or not status_obj.get("isdir", False):
                        raise orig_err
                except (json.JSONDecodeError, KeyError, IndexError, ValueError):
                    raise orig_err

            try:
                self._run_cmd(args, timeout=300.0)
            except UpstreamCliError as retry_err:
                if not hasattr(retry_err, "details") or not isinstance(retry_err.details, dict):
                    retry_err.details = {}
                retry_err.details["original_upload_error"] = str(orig_err)
                retry_err.details["recovery_attempted"] = True
                raise retry_err from orig_err

    def download_file(
        self,
        session_id: str,
        remote_path: str,
        local_path: Union[str, Path],
    ) -> None:
        self._verify_session_managed(session_id)
        remote_path = _validate_remote_path(remote_path)
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
        remote_path = _validate_remote_path(remote_path)
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
            # Upstream `colab exec` defaults to a 30 s kernel timeout, which kills long training runs.
            # Pass our budget explicitly, minus a margin so upstream reports its own timeout first.
            upstream_timeout = max(1, int(timeout) - EXEC_UPSTREAM_MARGIN_S)
            args = ["exec", "-s", session_id, "-f", tmp_script, "--timeout", str(upstream_timeout)]
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
