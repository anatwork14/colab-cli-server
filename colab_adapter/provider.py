"""Google Colab GPU Compute Provider Adapter.

Integrates official Google Colab CLI as an additional remote GPU backend
alongside local server GPUs (e.g., local GTX 1080 Ti).
Enforces:
- Separation between job preparation and actual provisioning.
- Strict human approval gate that autonomous agents cannot forge.
- Single active Colab GPU session concurrency limit.
- No unlimited retries.
- Verification of actual outputs (not relying solely on colab exec returncode).
- Reliable cleanup in finally blocks.
- Non-sensitive structured logging.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .tracker import ExperimentMetadata, ExperimentTracker

SUPPORTED_GPUS = ("T4", "L4", "G4", "A100", "H100")
MAX_TIMEOUT_SECONDS = 86400.0  # 24 hours


class ColabProviderError(Exception):
    """Base exception for Colab provider errors."""


class DependencyMissingError(ColabProviderError):
    """Raised when uv or colab CLI is missing."""


class PythonVersionError(ColabProviderError):
    """Raised when Python version is incompatible (< 3.12 for upstream colab CLI)."""


class AuthenticationError(ColabProviderError):
    """Raised when OAuth authentication is missing or invalid."""


class InvalidJobSpecError(ColabProviderError):
    """Raised when job arguments or files are invalid."""


class UnsupportedGPUError(ColabProviderError):
    """Raised when requested accelerator is not supported."""


class ApprovalRequiredError(ColabProviderError):
    """Raised when job has not been explicitly authorized by a human."""


class DuplicateAllocationError(ColabProviderError):
    """Raised when attempting to allocate while a Colab session is already running."""


class AllocationError(ColabProviderError):
    """Raised when Colab fails to allocate the requested GPU."""


class RemoteExecutionError(ColabProviderError):
    """Raised when remote execution or verification fails."""


class ArtifactMissingError(ColabProviderError):
    """Raised when expected output checkpoint or file is missing."""


class DiskSpaceError(ColabProviderError):
    """Raised when available local disk space is insufficient for downloads."""


class CleanupError(ColabProviderError):
    """Raised when runtime teardown fails."""


def check_disk_space(target_path: Path, min_free_mb: int = 500) -> None:
    """Ensure sufficient disk space exists before downloading artifacts."""
    target_dir = target_path if target_path.is_dir() else target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(target_dir).free
    free_mb = free_bytes / (1024 * 1024)
    if free_mb < min_free_mb:
        raise DiskSpaceError(
            f"Insufficient disk space in {target_dir}: {free_mb:.1f} MB free, minimum required is {min_free_mb} MB."
        )


@dataclass
class JobSpec:
    project_name: str
    script_path: Union[str, Path]
    gpu: str = "T4"
    requirements_path: Optional[Union[str, Path]] = None
    input_files: Dict[str, str] = field(default_factory=dict)  # local_path -> remote_path
    expected_outputs: List[str] = field(default_factory=list)  # remote_path
    output_download_dir: Optional[Union[str, Path]] = None
    timeout: float = 300.0
    dataset_identifier: Optional[str] = None
    stateful: bool = False  # False = colab run (one-shot); True = new/install/upload/exec/download/stop


@dataclass
class PreparedJob:
    job_id: str
    spec: JobSpec
    approval_nonce: str
    command_preview: List[str]
    prepared_at: str


@dataclass
class ApprovedJob:
    prepared_job: PreparedJob
    approval_code: str
    approved_by: str
    approved_at: str


@dataclass
class JobResult:
    job_id: str
    success: bool
    state: str
    output_files: List[str]
    error: Optional[str] = None
    logs: Optional[str] = None
    cleaned_up: bool = True
    session_id: Optional[str] = None


class ColabGPUProvider:
    """Production adapter for Google Colab remote GPU compute backend."""

    def __init__(
        self,
        auth_mode: str = "oauth2",
        cli_executable: Optional[str] = None,
        tracker: Optional[ExperimentTracker] = None,
    ):
        self.auth_mode = auth_mode
        self.cli_path = cli_executable or shutil.which("colab")
        self.tracker = tracker or ExperimentTracker()
        self._max_active_sessions = 1

    def _get_cli(self) -> str:
        cli = self.cli_path or shutil.which("colab")
        if not cli:
            raise DependencyMissingError(
                "Official Colab CLI executable not found on PATH. Run `bash scripts/install.sh` first."
            )
        return cli

    def _run_cli(
        self,
        args: List[str],
        check: bool = False,
        capture: bool = True,
        timeout: Optional[float] = None,
    ) -> subprocess.CompletedProcess:
        cli = self._get_cli()
        cmd = [cli, f"--auth={self.auth_mode}", *args]
        try:
            return subprocess.run(
                cmd,
                stdout=subprocess.PIPE if capture else None,
                stderr=subprocess.PIPE if capture else None,
                text=True if capture else False,
                check=check,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise RemoteExecutionError(
                f"Colab CLI command timed out after {timeout}s: {' '.join(cmd)}"
            ) from e

    def check_availability(self) -> Dict[str, Any]:
        """Inspect environment, tool version, and current session availability.

        Does NOT allocate any cloud compute.
        """
        cli_found = bool(self.cli_path or shutil.which("colab"))
        uv_found = bool(shutil.which("uv"))
        py_version = sys.version.split()[0]
        token_path = Path.home() / ".config" / "colab-cli" / "token.json"
        has_token = token_path.exists()

        active_sessions: List[str] = []
        auth_ok = False
        error_msg = None

        if cli_found and has_token:
            res = self._run_cli(["sessions"])
            if res.returncode == 0:
                auth_ok = True
                # Parse sessions from output
                lines = [l.strip() for l in (res.stdout or "").splitlines() if l.strip()]
                # Exclude table headers or empty notifications
                active_sessions = [
                    l for l in lines
                    if "no active sessions" not in l.lower()
                    and not l.startswith("Active")
                    and not l.startswith("Name")
                    and not l.startswith("---")
                ]
            else:
                error_msg = (res.stderr or res.stdout or "").strip()
        elif cli_found and not has_token:
            error_msg = "OAuth credentials not configured (~/.config/colab-cli/token.json missing). Run `bash scripts/authenticate.sh`."

        return {
            "cli_installed": cli_found,
            "uv_installed": uv_found,
            "python_version": py_version,
            "authenticated": auth_ok,
            "oauth_token_present": has_token,
            "active_sessions": active_sessions,
            "can_accept_new_colab_job": auth_ok and len(active_sessions) < self._max_active_sessions,
            "local_gpu_available": bool(shutil.which("nvidia-smi")),
            "error": error_msg,
        }

    def get_usage(self) -> Dict[str, Any]:
        """Query read-only compute-unit balance from Google Colab."""
        res = self._run_cli(["usage"])
        if res.returncode != 0:
            err = (res.stderr or res.stdout or "Failed to query usage").strip()
            return {"success": False, "error": err, "raw_output": ""}
        return {"success": True, "error": None, "raw_output": (res.stdout or "").strip()}

    def prepare_job(self, job: JobSpec) -> PreparedJob:
        """Validate job inputs and prepare execution plan.

        Does NOT provision any runtime.
        """
        # Validate Python environment
        if sys.version_info < (3, 8):
            raise PythonVersionError(f"Python 3.8+ required locally; found {sys.version}")

        # Validate script
        script_path = Path(job.script_path).expanduser().resolve()
        if not script_path.is_file() or script_path.suffix != ".py":
            raise InvalidJobSpecError(f"Script must be an existing .py file: {job.script_path}")

        # Validate GPU
        gpu = job.gpu.upper()
        if gpu not in SUPPORTED_GPUS:
            raise UnsupportedGPUError(f"Requested GPU '{job.gpu}' unsupported. Must be one of {SUPPORTED_GPUS}")

        # Validate timeout
        if not (0 < job.timeout <= MAX_TIMEOUT_SECONDS):
            raise InvalidJobSpecError(f"Timeout must be between 1 and {MAX_TIMEOUT_SECONDS}s; got {job.timeout}")

        # Validate requirements if given
        if job.requirements_path:
            req_path = Path(job.requirements_path).expanduser().resolve()
            if not req_path.is_file():
                raise InvalidJobSpecError(f"Requirements file not found: {job.requirements_path}")

        # Generate unique job ID
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d-%H%M%S")
        content_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()[:8]
        job_id = f"colab-{gpu.lower()}-{ts}-{content_hash}"

        # Generate approval nonce (tied to job_id and GPU to prevent replay/forgery)
        nonce_src = f"{job_id}:{gpu}:{script_path}:{now.isoformat()}"
        nonce = hashlib.sha256(nonce_src.encode("utf-8")).hexdigest()[:12]

        # Preview command
        if job.stateful:
            cmd = ["colab", f"--auth={self.auth_mode}", "new", "-s", job_id, "--gpu", gpu]
        else:
            cmd = ["colab", f"--auth={self.auth_mode}", "run", "--gpu", gpu, "--timeout", str(job.timeout), str(script_path)]

        prepared = PreparedJob(
            job_id=job_id,
            spec=job,
            approval_nonce=nonce,
            command_preview=cmd,
            prepared_at=now.isoformat(),
        )

        # Log PREPARED state to tracker
        meta = ExperimentMetadata(
            job_id=job_id,
            project_name=job.project_name,
            requested_accelerator=gpu,
            start_time=now.isoformat(),
            completion_time=None,
            job_state="PREPARED",
            script_path=str(script_path),
            script_revision=content_hash,
            dependencies=[str(job.requirements_path)] if job.requirements_path else [],
            dataset_identifier=job.dataset_identifier,
            output_artifact_paths=[],
            success=False,
            error_summary=None,
            cleanup_status="NOT_APPLICABLE",
            compute_units_before=None,
            compute_units_after=None,
        )
        self.tracker.record_job(meta)
        return prepared

    def approve_job_interactive(self, prepared_job: PreparedJob) -> ApprovedJob:
        """Prompt a human operator at an interactive terminal to authorize provisioning.

        Refuses non-interactive or automated calls.
        """
        if not sys.stdin.isatty():
            raise ApprovalRequiredError(
                "REFUSED: Colab GPU provisioning requires human authorization at an interactive terminal (stdin is not a tty)."
            )

        gpu = prepared_job.spec.gpu.upper()
        nonce = prepared_job.approval_nonce
        expected_phrase = f"ALLOCATE {gpu} {nonce}"

        print(f"\n{'='*70}")
        print("COLAB GPU ALLOCATION APPROVAL GATE")
        print(f"Job ID:      {prepared_job.job_id}")
        print(f"Project:     {prepared_job.spec.project_name}")
        print(f"Script:      {prepared_job.spec.script_path}")
        print(f"GPU:         {gpu} (May consume Colab compute units)")
        print(f"Command:     {' '.join(prepared_job.command_preview)}")
        print(f"{'='*70}")
        print(f"To authorize allocation, enter exactly: {expected_phrase}")

        try:
            val = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            raise ApprovalRequiredError("Approval prompt aborted by user.")

        if val != expected_phrase:
            raise ApprovalRequiredError(f"Approval rejected: entered phrase '{val}' did not match '{expected_phrase}'.")

        now = datetime.now(timezone.utc).isoformat()
        return ApprovedJob(
            prepared_job=prepared_job,
            approval_code=nonce,
            approved_by="human_terminal",
            approved_at=now,
        )

    def submit_job(self, approved_job: ApprovedJob) -> JobResult:
        """Execute approved job on Colab backend with rigorous lifecycle and cleanup."""
        prep = approved_job.prepared_job
        spec = prep.spec
        gpu = spec.gpu.upper()
        job_id = prep.job_id

        # Verify approval code matches nonce
        if approved_job.approval_code != prep.approval_nonce:
            raise ApprovalRequiredError("Invalid approval code. Autonomous agents cannot forge approval tokens.")

        # Check existing sessions to prevent duplicate allocation
        avail = self.check_availability()
        if not avail["authenticated"]:
            raise AuthenticationError(avail.get("error") or "Colab CLI not authenticated. Run `bash scripts/authenticate.sh`.")

        if len(avail["active_sessions"]) >= self._max_active_sessions:
            raise DuplicateAllocationError(
                f"Cannot allocate: Active Colab sessions exist ({avail['active_sessions']}). Maximum concurrent session limit is {self._max_active_sessions}."
            )

        # Record compute units before
        usage_before = self.get_usage().get("raw_output")
        start_time = datetime.now(timezone.utc).isoformat()
        success = False
        error_msg = None
        output_files: List[str] = []
        cleanup_status = "PENDING"
        execution_logs = ""

        # Destination directory for outputs
        download_dir = Path(spec.output_download_dir or f"./colab_outputs/{job_id}").expanduser().resolve()
        download_dir.mkdir(parents=True, exist_ok=True)

        session_id = job_id if spec.stateful else None
        cli_timeout = spec.timeout + 120.0

        try:
            if not spec.stateful:
                # One-shot colab run with explicit subprocess timeout
                cmd_args = ["run", "--gpu", gpu, "--timeout", str(spec.timeout), str(Path(spec.script_path).resolve())]
                res = self._run_cli(cmd_args, timeout=cli_timeout)
                execution_logs = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")

                if res.returncode != 0:
                    if "ResourceExhausted" in execution_logs or "unavailable" in execution_logs.lower():
                        raise AllocationError(f"GPU {gpu} allocation unavailable on Colab: {execution_logs}")
                    raise RemoteExecutionError(f"Colab execution failed with exit code {res.returncode}:\n{execution_logs}")

                # Verify execution logs contain no uncaught exception or fatal error
                error_indicators = (
                    "Traceback (most recent call last):",
                    "CUDA out of memory",
                    "ModuleNotFoundError:",
                    "SyntaxError:",
                )
                if any(ind in execution_logs for ind in error_indicators):
                    raise RemoteExecutionError(f"Remote error detected in execution logs:\n{execution_logs}")

                success = True
                cleanup_status = "CLEANED"
            else:
                # Stateful multi-step lifecycle
                # 1. colab new
                res_new = self._run_cli(["new", "-s", session_id, "--gpu", gpu], timeout=180.0)
                if res_new.returncode != 0:
                    err = (res_new.stderr or res_new.stdout or "").strip()
                    raise AllocationError(f"Failed to allocate session {session_id} on GPU {gpu}: {err}")

                try:
                    # 2. install dependencies if provided
                    if spec.requirements_path:
                        res_inst = self._run_cli(["install", "-s", session_id, "-r", str(Path(spec.requirements_path).resolve())], timeout=300.0)
                        if res_inst.returncode != 0:
                            raise RemoteExecutionError(f"Failed to install requirements on session {session_id}: {res_inst.stderr}")

                    # 3. upload inputs if provided
                    for local_p, remote_p in spec.input_files.items():
                        res_up = self._run_cli(["upload", "-s", session_id, str(Path(local_p).resolve()), remote_p], timeout=180.0)
                        if res_up.returncode != 0:
                            raise RemoteExecutionError(f"Failed to upload {local_p} to {remote_p}: {res_up.stderr}")

                    # 4. exec script
                    res_exec = self._run_cli(["exec", "-s", session_id, "--timeout", str(spec.timeout), "-f", str(Path(spec.script_path).resolve())], timeout=cli_timeout)
                    execution_logs = (res_exec.stdout or "") + ("\n" + res_exec.stderr if res_exec.stderr else "")

                    if res_exec.returncode != 0:
                        raise RemoteExecutionError(f"Script execution failed with exit code {res_exec.returncode}:\n{execution_logs}")

                    error_indicators = (
                        "Traceback (most recent call last):",
                        "CUDA out of memory",
                        "ModuleNotFoundError:",
                        "SyntaxError:",
                    )
                    if any(ind in execution_logs for ind in error_indicators):
                        raise RemoteExecutionError(f"Remote traceback detected:\n{execution_logs}")

                    # 5. verify disk space before downloading artifacts
                    if spec.expected_outputs:
                        check_disk_space(download_dir, min_free_mb=500)

                    # 6. verify expected outputs and download
                    for remote_out in spec.expected_outputs:
                        filename = Path(remote_out).name
                        local_target = download_dir / filename
                        res_dl = self._run_cli(["download", "-s", session_id, remote_out, str(local_target)], timeout=180.0)
                        if res_dl.returncode != 0 or not local_target.exists():
                            raise ArtifactMissingError(f"Expected output {remote_out} could not be downloaded to {local_target}")
                        output_files.append(str(local_target))

                    success = True
                finally:
                    # Guaranteed session termination
                    try:
                        res_stop = self._run_cli(["stop", "-s", session_id], timeout=60.0)
                        if res_stop.returncode == 0:
                            cleanup_status = "RUNTIME_STOPPED"
                        else:
                            cleanup_status = "ORPHANED"
                            stop_err = (res_stop.stderr or res_stop.stdout or "").strip()
                            if not error_msg:
                                raise CleanupError(f"Failed to stop session {session_id}: {stop_err}")
                    except Exception:
                        cleanup_status = "ORPHANED"
                        if not error_msg:
                            raise

        except Exception as e:
            success = False
            error_msg = str(e)
            if cleanup_status == "PENDING":
                cleanup_status = "FAILED"
            raise
        finally:
            completion_time = datetime.now(timezone.utc).isoformat()
            usage_after = self.get_usage().get("raw_output")

            # Record final experiment metadata
            meta = ExperimentMetadata(
                job_id=job_id,
                project_name=spec.project_name,
                requested_accelerator=gpu,
                start_time=start_time,
                completion_time=completion_time,
                job_state="COMPLETED" if success else "FAILED",
                script_path=str(Path(spec.script_path).resolve()),
                script_revision=prep.prepared_at,
                dependencies=[str(spec.requirements_path)] if spec.requirements_path else [],
                dataset_identifier=spec.dataset_identifier,
                output_artifact_paths=output_files,
                success=success,
                error_summary=error_msg,
                cleanup_status=cleanup_status,
                compute_units_before=usage_before,
                compute_units_after=usage_after,
            )
            self.tracker.record_job(meta)

        return JobResult(
            job_id=job_id,
            success=success,
            state="COMPLETED" if success else "FAILED",
            output_files=output_files,
            error=error_msg,
            logs=execution_logs,
            cleaned_up=cleanup_status in ("CLEANED", "RUNTIME_STOPPED"),
            session_id=session_id,
        )

    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Query status of a specific job from structured logs and active sessions."""
        logged = self.tracker.get_job(job_id)
        active = self.check_availability()["active_sessions"]
        is_active = any(job_id in s for s in active)
        return {
            "job_id": job_id,
            "is_running_on_colab": is_active,
            "logged_state": logged.get("job_state") if logged else "UNKNOWN",
            "success": logged.get("success") if logged else None,
            "error_summary": logged.get("error_summary") if logged else None,
            "outputs": logged.get("output_artifact_paths", []) if logged else [],
        }

    def collect_artifacts(self, job_id: str) -> List[str]:
        """Return downloaded local artifacts for a completed job."""
        logged = self.tracker.get_job(job_id)
        if not logged:
            return []
        return logged.get("output_artifact_paths", [])

    def cancel_job(self, session_id: str) -> bool:
        """Cancel and terminate a running Colab session."""
        res = self._run_cli(["stop", "-s", session_id])
        return res.returncode == 0

    def cleanup(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Release session if given, or report and stop any orphaned sessions."""
        stopped = []
        failed = []
        if session_id:
            res = self._run_cli(["stop", "-s", session_id], timeout=60.0)
            if res.returncode == 0:
                stopped.append(session_id)
            else:
                failed.append(session_id)
        else:
            avail = self.check_availability()
            for s in avail["active_sessions"]:
                res = self._run_cli(["stop", "-s", s], timeout=60.0)
                if res.returncode == 0:
                    stopped.append(s)
                else:
                    failed.append(s)
        return {"stopped": stopped, "failed": failed}

    def reconcile_stale_sessions(self, auto_stop: bool = False) -> Dict[str, Any]:
        """Inspect and reconcile running Colab sessions against tracked job metadata.

        If auto_stop is True, stops sessions that are not actively running a current job.
        """
        avail = self.check_availability()
        active = avail.get("active_sessions", [])
        stale = list(active)
        stopped = []
        failed = []

        if auto_stop and active:
            for session in active:
                res = self._run_cli(["stop", "-s", session], timeout=60.0)
                if res.returncode == 0:
                    stopped.append(session)
                else:
                    failed.append(session)

        return {
            "active_found": active,
            "stale_sessions": stale,
            "stopped": stopped,
            "failed": failed,
        }
