"""Google Colab GPU Compute Provider Adapter.

Integrates official Google Colab CLI as an additional remote GPU backend
alongside local server GPUs (e.g., local GTX 1080 Ti).

Key Safety & Reliability Controls:
- Session Ownership: Only manages and cleans up adapter-registered sessions. Never stops unrelated user sessions.
- Approval Guard: Binds cryptographic input hashes (script, requirements, dataset) at preparation time;
  detects file tampering before dispatch. Documents that in-process approval is a usability safeguard,
  requiring an isolated, separately protected executor for autonomous agents.
- Concurrency Protection: Cross-process file locking (fcntl.flock) and single active session limit.
- Subprocess Timeouts: Generous wall-clock buffer around remote kernel timeouts to avoid hangs while allowing teardown.
- Error Preservation: Primary execution errors are never masked by subsequent cleanup or usage query failures.
- Credential Security: Deep sanitization of all logged output and strict 0600 file permissions.
- Dual Auth Support: Supports both OAuth2 and Application Default Credentials (ADC).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .tracker import ExperimentMetadata, ExperimentTracker, sanitize_text

SUPPORTED_GPUS = ("T4", "L4", "G4", "A100", "H100")
MAX_TIMEOUT_SECONDS = 86400.0  # 24 hours


class ColabProviderError(Exception):
    """Base exception for Colab provider errors."""


class DependencyMissingError(ColabProviderError):
    """Raised when uv or colab CLI is missing."""


class PythonVersionError(ColabProviderError):
    """Raised when Python version is incompatible (< 3.12 for upstream colab CLI)."""


class AuthenticationError(ColabProviderError):
    """Raised when OAuth or ADC authentication is missing or invalid."""


class InvalidJobSpecError(ColabProviderError):
    """Raised when job arguments or files are invalid."""


class UnsupportedGPUError(ColabProviderError):
    """Raised when requested accelerator is not supported."""


class ApprovalRequiredError(ColabProviderError):
    """Raised when job has not been explicitly authorized by a human."""


class JobTamperedError(ColabProviderError):
    """Raised when script, requirements, or inputs were modified after approval."""


class DuplicateAllocationError(ColabProviderError):
    """Raised when attempting to allocate while a Colab session is already running or locked."""


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


class ColabProcessLock:
    """Cross-process and cross-thread lock preventing concurrent Colab allocations."""

    _path_locks: Dict[Path, threading.Lock] = {}
    _path_locks_guard = threading.Lock()

    def __init__(self, lock_path: Optional[Path] = None):
        if lock_path is None:
            lock_path = Path.home() / ".colab_experiments" / "colab_gpu.lock"
        self.lock_path = Path(lock_path)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread_lock = self._get_thread_lock(self.lock_path)
        self._fd: Optional[int] = None
        self._is_held_by_thread = False

    @classmethod
    def _get_thread_lock(cls, path: Path) -> threading.Lock:
        with cls._path_locks_guard:
            resolved = path.resolve()
            if resolved not in cls._path_locks:
                cls._path_locks[resolved] = threading.Lock()
            return cls._path_locks[resolved]

    def acquire(self, blocking: bool = False) -> bool:
        # Prevent reentrancy across threads sharing this lock path
        if not self._thread_lock.acquire(blocking=blocking):
            return False

        flags = os.O_RDWR | os.O_CREAT
        fd = os.open(str(self.lock_path), flags, 0o600)
        lock_flags = fcntl.LOCK_EX
        if not blocking:
            lock_flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(fd, lock_flags)
            os.write(
                fd,
                f"pid={os.getpid()}\nthread={threading.get_ident()}\ntime={time.time()}\n".encode("utf-8"),
            )
            self._fd = fd
            self._is_held_by_thread = True
            return True
        except (BlockingIOError, OSError):
            os.close(fd)
            self._thread_lock.release()
            return False
        except Exception:
            os.close(fd)
            self._thread_lock.release()
            raise

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        if self._is_held_by_thread:
            self._is_held_by_thread = False
            try:
                self._thread_lock.release()
            except RuntimeError:
                pass

    def __enter__(self) -> "ColabProcessLock":
        if not self.acquire(blocking=False):
            raise DuplicateAllocationError(
                "Another Colab job is currently holding the GPU execution lock. Concurrent allocations are prohibited."
            )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


class ManagedSessionRegistry:
    """Registry tracking Colab sessions created by this adapter.

    Guarantees that the adapter only manages, monitors, and cleans up its own sessions,
    never interfering with unrelated user sessions on the same account.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        if registry_path is None:
            registry_path = Path.home() / ".colab_experiments" / "managed_sessions.json"
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> Dict[str, Any]:
        if not self.registry_path.exists():
            return {}
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _write(self, data: Dict[str, Any]) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(str(self.registry_path), flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise

    def register(self, session_id: str, job_id: str, gpu: str, project_name: str) -> None:
        data = self._read()
        data[session_id] = {
            "session_id": session_id,
            "job_id": job_id,
            "gpu": gpu,
            "project_name": project_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "state": "PROVISIONING",
        }
        self._write(data)

    def update_state(self, session_id: str, state: str) -> None:
        data = self._read()
        if session_id in data:
            data[session_id]["state"] = state
            data[session_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write(data)

    def is_managed(self, session_id: str) -> bool:
        data = self._read()
        return session_id in data

    def get_active_managed(self) -> List[str]:
        data = self._read()
        return [
            s_id for s_id, info in data.items()
            if info.get("state") in ("PROVISIONING", "ACTIVE", "INTERRUPTED", "NEEDS_RECONCILIATION")
        ]

    def get_all_managed(self) -> Dict[str, Any]:
        return self._read()


def parse_session_lines(output: str) -> List[Dict[str, str]]:
    """Parse structured session information from colab CLI output.

    Format from colab_cli:
    [<name>] <endpoint> | Hardware: <accel> | Shape: <shape> | Variant: <variant>[ | Status: <status>]
    """
    parsed = []
    pattern = re.compile(r"^\[(?P<name>[^\]]+)\]\s+(?P<endpoint>\S+)")
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("[colab]"):
            continue
        m = pattern.match(line)
        if m:
            name = m.group("name").strip()
            endpoint = m.group("endpoint").strip()
            # If name is not empty and not the CLI tag
            if name and name.lower() != "colab":
                parsed.append({"name": name, "endpoint": endpoint, "raw_line": line})
    return parsed


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
    fingerprint: str
    script_sha256: str
    reqs_sha256: str
    inputs_sha256: str
    command_preview: List[str]
    prepared_at: str


@dataclass
class ApprovedJob:
    prepared_job: PreparedJob
    approval_code: str
    approved_by: str
    fingerprint: str
    approved_at: str


@dataclass
class JobResult:
    job_id: str
    success: bool
    state: str
    output_files: List[str]
    error: Optional[str] = None
    cleanup_error: Optional[str] = None
    usage_error: Optional[str] = None
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
        registry: Optional[ManagedSessionRegistry] = None,
        lock: Optional[ColabProcessLock] = None,
    ):
        self.auth_mode = auth_mode
        self.cli_path = cli_executable or shutil.which("colab")
        self.tracker = tracker or ExperimentTracker()
        self.registry = registry or ManagedSessionRegistry()
        self.lock = lock or ColabProcessLock()
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

    def _check_auth_credentials(self) -> Tuple[bool, Optional[str]]:
        """Verify presence of credentials for configured auth strategy without cloud allocation."""
        if self.auth_mode == "oauth2":
            token_path = Path.home() / ".config" / "colab-cli" / "token.json"
            if not token_path.exists():
                return False, "OAuth2 credentials missing (~/.config/colab-cli/token.json not found). Run `bash scripts/authenticate.sh`."
            return True, None
        elif self.auth_mode == "adc":
            adc_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            adc_default = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
            if adc_env and Path(adc_env).exists():
                return True, None
            if adc_default.exists():
                return True, None
            # If not explicitly on disk, check if colab --auth=adc sessions can connect
            return False, "Application Default Credentials missing. Run `gcloud auth application-default login`."
        return False, f"Unknown auth mode: {self.auth_mode}"

    def check_availability(self) -> Dict[str, Any]:
        """Inspect environment, tool version, and current session availability.

        Does NOT allocate any cloud compute.
        """
        cli_found = bool(self.cli_path or shutil.which("colab"))
        uv_found = bool(shutil.which("uv"))
        py_version = sys.version.split()[0]
        has_credentials, auth_err = self._check_auth_credentials()

        active_managed_sessions: List[str] = []
        unmanaged_user_sessions: List[str] = []
        auth_ok = False
        error_msg = auth_err

        if cli_found and has_credentials:
            res = self._run_cli(["sessions"], timeout=30.0)
            if res.returncode == 0:
                auth_ok = True
                error_msg = None
                parsed = parse_session_lines(res.stdout or "")
                for s in parsed:
                    name = s["name"]
                    if self.registry.is_managed(name):
                        active_managed_sessions.append(name)
                    else:
                        unmanaged_user_sessions.append(name)
            else:
                error_msg = (res.stderr or res.stdout or "").strip()

        return {
            "cli_installed": cli_found,
            "uv_installed": uv_found,
            "python_version": py_version,
            "auth_mode": self.auth_mode,
            "authenticated": auth_ok,
            "credentials_present": has_credentials,
            "active_sessions": active_managed_sessions,
            "unmanaged_sessions": unmanaged_user_sessions,
            "can_accept_new_colab_job": auth_ok and len(active_managed_sessions) < self._max_active_sessions,
            "local_gpu_available": bool(shutil.which("nvidia-smi")),
            "error": error_msg,
        }

    def get_usage(self) -> Dict[str, Any]:
        """Query read-only compute-unit balance from Google Colab."""
        try:
            res = self._run_cli(["usage"], timeout=30.0)
            if res.returncode != 0:
                err = (res.stderr or res.stdout or "Failed to query usage").strip()
                return {"success": False, "error": err, "raw_output": ""}
            return {"success": True, "error": None, "raw_output": (res.stdout or "").strip()}
        except Exception as e:
            return {"success": False, "error": str(e), "raw_output": ""}

    def _compute_job_hashes(self, job: JobSpec) -> Tuple[str, str, str, str]:
        """Compute cryptographic SHA256 hashes of all job assets and parameters to guarantee integrity."""
        script_path = Path(job.script_path).expanduser().resolve()
        script_sha = hashlib.sha256(script_path.read_bytes()).hexdigest()

        reqs_sha = ""
        if job.requirements_path:
            reqs_path = Path(job.requirements_path).expanduser().resolve()
            reqs_sha = hashlib.sha256(reqs_path.read_bytes()).hexdigest()

        input_hashes = []
        for local_p in sorted(job.input_files.keys()):
            p = Path(local_p).expanduser().resolve()
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            remote_dest = str(job.input_files[local_p])
            input_hashes.append(f"{local_p}->{remote_dest}:{h}")
        inputs_sha = hashlib.sha256(";".join(input_hashes).encode("utf-8")).hexdigest()

        gpu = job.gpu.upper()
        outputs_str = ";".join(sorted(job.expected_outputs))
        out_dir = str(job.output_download_dir or "")
        dataset_id = str(job.dataset_identifier or "")
        proj_name = str(job.project_name)

        fingerprint_src = (
            f"{proj_name}:{script_sha}:{reqs_sha}:{inputs_sha}:{outputs_str}:"
            f"{out_dir}:{dataset_id}:{gpu}:{job.timeout}:{job.stateful}"
        )
        fingerprint = hashlib.sha256(fingerprint_src.encode("utf-8")).hexdigest()
        return script_sha, reqs_sha, inputs_sha, fingerprint

    def prepare_job(self, job: JobSpec) -> PreparedJob:
        """Validate job inputs and prepare execution plan with immutable asset fingerprints.

        Does NOT provision any runtime.
        """
        if sys.version_info < (3, 8):
            raise PythonVersionError(f"Python 3.8+ required locally; found {sys.version}")

        script_path = Path(job.script_path).expanduser().resolve()
        if not script_path.is_file() or script_path.suffix != ".py":
            raise InvalidJobSpecError(f"Script must be an existing .py file: {job.script_path}")

        gpu = job.gpu.upper()
        if gpu not in SUPPORTED_GPUS:
            raise UnsupportedGPUError(f"Requested GPU '{job.gpu}' unsupported. Must be one of {SUPPORTED_GPUS}")

        if not (0 < job.timeout <= MAX_TIMEOUT_SECONDS):
            raise InvalidJobSpecError(f"Timeout must be between 1 and {MAX_TIMEOUT_SECONDS}s; got {job.timeout}")

        if job.requirements_path:
            req_path = Path(job.requirements_path).expanduser().resolve()
            if not req_path.is_file():
                raise InvalidJobSpecError(f"Requirements file not found: {job.requirements_path}")

        for local_p in job.input_files.keys():
            inp_p = Path(local_p).expanduser().resolve()
            if not inp_p.is_file():
                raise InvalidJobSpecError(f"Input file not found: {local_p}")

        script_sha, reqs_sha, inputs_sha, fingerprint = self._compute_job_hashes(job)

        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d-%H%M%S")
        job_id = f"colab-{gpu.lower()}-{ts}-{script_sha[:8]}"
        approval_nonce = fingerprint[:12]

        if job.stateful:
            cmd = ["colab", f"--auth={self.auth_mode}", "new", "-s", job_id, "--gpu", gpu]
        else:
            cmd = ["colab", f"--auth={self.auth_mode}", "run", "-s", job_id, "--gpu", gpu, "--timeout", str(job.timeout), str(script_path)]

        prepared = PreparedJob(
            job_id=job_id,
            spec=job,
            approval_nonce=approval_nonce,
            fingerprint=fingerprint,
            script_sha256=script_sha,
            reqs_sha256=reqs_sha,
            inputs_sha256=inputs_sha,
            command_preview=cmd,
            prepared_at=now.isoformat(),
        )

        meta = ExperimentMetadata(
            job_id=job_id,
            project_name=job.project_name,
            requested_accelerator=gpu,
            start_time=now.isoformat(),
            completion_time=None,
            job_state="PREPARED",
            script_path=str(script_path),
            script_revision=script_sha[:8],
            dependencies=[str(job.requirements_path)] if job.requirements_path else [],
            dataset_identifier=job.dataset_identifier,
            output_artifact_paths=[],
            success=False,
            error_summary=None,
            cleanup_status="NOT_APPLICABLE",
        )
        self.tracker.record_job(meta)
        return prepared

    def approve_job_interactive(self, prepared_job: PreparedJob) -> ApprovedJob:
        """Prompt human operator at interactive terminal to authorize provisioning.

        NOTE ON SECURITY ARCHITECTURE:
        This interactive confirmation ensures human awareness and consent for ordinary operations.
        It is a usability safeguard, not a cryptographic security boundary against untrusted local agents
        sharing the same OS account. For autonomous workflows, deploy a separately protected approval executor.
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
            raise ApprovalRequiredError(f"Approval rejected: entered phrase did not match '{expected_phrase}'.")

        now = datetime.now(timezone.utc).isoformat()
        return ApprovedJob(
            prepared_job=prepared_job,
            approval_code=nonce,
            approved_by="human_terminal",
            fingerprint=prepared_job.fingerprint,
            approved_at=now,
        )

    def verify_job_integrity(self, approved_job: ApprovedJob) -> None:
        """Verify that job assets and parameters have not been modified between preparation and submission.

        NOTE ON SECURITY ARCHITECTURE:
        This integrity verification ensures that an approved JobSpec has not experienced accidental
        tampering, file mutation, or destination path modification prior to remote execution.
        While it validates parameters, input file hashes, remote transfer destinations, outputs,
        and approval nonces, it operates within the Python runtime and does not constitute an
        unforgeable authorization boundary against untrusted local agents running under the same
        OS user account. Autonomous agent workflows requiring cryptographic separation must rely on
        a separately protected approval executor or daemon with non-delegable credentials.
        """
        prep = approved_job.prepared_job
        if not approved_job.approval_code:
            raise JobTamperedError("Missing approval code. Execution rejected.")
        if approved_job.approval_code != prep.approval_nonce:
            raise JobTamperedError("Approval nonce mismatch. Execution rejected.")
        if approved_job.fingerprint != prep.fingerprint:
            raise JobTamperedError("Approved job fingerprint mismatch. Execution rejected.")
        if not approved_job.approved_by:
            raise JobTamperedError("Missing approved_by authorization identity. Execution rejected.")

        script_sha, reqs_sha, inputs_sha, current_fp = self._compute_job_hashes(prep.spec)

        if script_sha != prep.script_sha256:
            raise JobTamperedError("Script file was modified after approval. Execution rejected.")
        if reqs_sha != prep.reqs_sha256:
            raise JobTamperedError("Requirements file was modified after approval. Execution rejected.")
        if inputs_sha != prep.inputs_sha256:
            raise JobTamperedError("Input dataset files or remote destinations were modified after approval. Execution rejected.")
        if current_fp != approved_job.fingerprint:
            raise JobTamperedError("Job configuration or output parameters were altered after approval. Execution rejected.")

    def submit_job(self, approved_job: ApprovedJob) -> JobResult:
        """Execute approved job on Colab backend with cross-process locking, lifecycle guard and cleanup."""
        # 1. Verify tamper resistance
        self.verify_job_integrity(approved_job)

        prep = approved_job.prepared_job
        spec = prep.spec
        gpu = spec.gpu.upper()
        job_id = prep.job_id
        session_id = job_id

        # 2. Acquire cross-process execution lock
        with self.lock:
            # Check availability under lock
            avail = self.check_availability()
            if not avail["authenticated"]:
                raise AuthenticationError(avail.get("error") or "Colab CLI not authenticated.")

            if len(avail["active_sessions"]) >= self._max_active_sessions:
                raise DuplicateAllocationError(
                    f"Cannot allocate: Active managed Colab session already running ({avail['active_sessions']}). Maximum limit is {self._max_active_sessions}."
                )

            # Record compute units before
            usage_before = self.get_usage().get("raw_output")
            start_time = datetime.now(timezone.utc).isoformat()
            primary_error: Optional[BaseException] = None
            cleanup_error: Optional[str] = None
            usage_error: Optional[str] = None
            output_files: List[str] = []
            cleanup_status = "PENDING"
            execution_logs = ""

            download_dir = Path(spec.output_download_dir or f"./colab_outputs/{job_id}").expanduser().resolve()
            download_dir.mkdir(parents=True, exist_ok=True)

            # Pre-register session in managed registry before invoking cloud allocation
            self.registry.register(session_id, job_id, gpu, spec.project_name)

            # Distinguish kernel inactivity timeout from overall wall-clock subprocess limit
            # Kernel timeout: passed to CLI
            # Subprocess timeout: generous wall-clock limit allowing spinup, execution, and teardown
            cli_wall_clock_timeout = spec.timeout + 180.0

            try:
                if not spec.stateful:
                    # One-shot colab run with explicit session name
                    cmd_args = ["run", "-s", session_id, "--gpu", gpu, "--timeout", str(spec.timeout), str(Path(spec.script_path).resolve())]
                    try:
                        res = self._run_cli(cmd_args, timeout=cli_wall_clock_timeout)
                        execution_logs = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")

                        if res.returncode != 0:
                            if "ResourceExhausted" in execution_logs or "unavailable" in execution_logs.lower():
                                raise AllocationError(f"GPU {gpu} allocation unavailable on Colab: {execution_logs}")
                            raise RemoteExecutionError(f"Colab execution failed with exit code {res.returncode}:\n{execution_logs}")

                        error_indicators = (
                            "Traceback (most recent call last):",
                            "CUDA out of memory",
                            "ModuleNotFoundError:",
                            "SyntaxError:",
                        )
                        if any(ind in execution_logs for ind in error_indicators):
                            raise RemoteExecutionError(f"Remote error detected in execution logs:\n{execution_logs}")

                        cleanup_status = "CLEANED"
                        self.registry.update_state(session_id, "TERMINATED")
                    except BaseException as one_shot_err:
                        # Attempt explicit cleanup for the registered session in case of CLI failure or timeout
                        try:
                            res_stop = self._run_cli(["stop", "-s", session_id], timeout=30.0)
                            if res_stop.returncode == 0:
                                cleanup_status = "RUNTIME_STOPPED"
                                self.registry.update_state(session_id, "TERMINATED")
                            else:
                                cleanup_status = "ORPHANED"
                                self.registry.update_state(session_id, "NEEDS_RECONCILIATION")
                                cleanup_error = f"Failed to stop session: {(res_stop.stderr or res_stop.stdout or '').strip()}"
                        except Exception as ce:
                            cleanup_status = "ORPHANED"
                            self.registry.update_state(session_id, "NEEDS_RECONCILIATION")
                            cleanup_error = f"Exception during session stop: {ce}"
                        raise
                else:
                    # Stateful multi-step lifecycle
                    # Step 1: colab new
                    res_new = self._run_cli(["new", "-s", session_id, "--gpu", gpu], timeout=180.0)
                    if res_new.returncode != 0:
                        err = (res_new.stderr or res_new.stdout or "").strip()
                        raise AllocationError(f"Failed to allocate session {session_id} on GPU {gpu}: {err}")
                    self.registry.update_state(session_id, "ACTIVE")

                    try:
                        # Step 2: install dependencies if specified
                        if spec.requirements_path:
                            res_inst = self._run_cli(["install", "-s", session_id, "-r", str(Path(spec.requirements_path).resolve())], timeout=300.0)
                            if res_inst.returncode != 0:
                                raise RemoteExecutionError(f"Failed to install requirements on session {session_id}: {res_inst.stderr}")

                        # Step 3: upload inputs if specified
                        for local_p, remote_p in spec.input_files.items():
                            res_up = self._run_cli(["upload", "-s", session_id, str(Path(local_p).resolve()), remote_p], timeout=180.0)
                            if res_up.returncode != 0:
                                raise RemoteExecutionError(f"Failed to upload {local_p} to {remote_p}: {res_up.stderr}")

                        # Step 4: execute script on remote kernel
                        res_exec = self._run_cli(["exec", "-s", session_id, "--timeout", str(spec.timeout), "-f", str(Path(spec.script_path).resolve())], timeout=cli_wall_clock_timeout)
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
                            raise RemoteExecutionError(f"Remote error detected:\n{execution_logs}")

                        # Step 5: check disk space before downloading
                        if spec.expected_outputs:
                            check_disk_space(download_dir, min_free_mb=500)

                        # Step 6: download and verify artifacts
                        for remote_out in spec.expected_outputs:
                            filename = Path(remote_out).name
                            local_target = download_dir / filename
                            res_dl = self._run_cli(["download", "-s", session_id, remote_out, str(local_target)], timeout=180.0)
                            if res_dl.returncode != 0 or not local_target.exists():
                                raise ArtifactMissingError(f"Expected output {remote_out} could not be downloaded to {local_target}")
                            output_files.append(str(local_target))

                    finally:
                        # Guaranteed teardown of stateful session in inner finally
                        try:
                            res_stop = self._run_cli(["stop", "-s", session_id], timeout=60.0)
                            if res_stop.returncode == 0:
                                cleanup_status = "RUNTIME_STOPPED"
                                self.registry.update_state(session_id, "TERMINATED")
                            else:
                                cleanup_status = "ORPHANED"
                                self.registry.update_state(session_id, "NEEDS_RECONCILIATION")
                                cleanup_error = f"Failed to stop session: {(res_stop.stderr or res_stop.stdout or '').strip()}"
                        except Exception as ce:
                            cleanup_status = "ORPHANED"
                            self.registry.update_state(session_id, "NEEDS_RECONCILIATION")
                            cleanup_error = f"Exception during session stop: {ce}"

            except BaseException as e:
                primary_error = e
                if isinstance(e, KeyboardInterrupt):
                    self.registry.update_state(session_id, "INTERRUPTED")
                elif cleanup_status == "PENDING":
                    cleanup_status = "FAILED"
                    self.registry.update_state(session_id, "NEEDS_RECONCILIATION")
                raise
            finally:
                # Outer finally: query usage & log experiment metadata without masking primary error
                completion_time = datetime.now(timezone.utc).isoformat()
                usage_after = None
                try:
                    usage_res = self.get_usage()
                    if usage_res["success"]:
                        usage_after = usage_res["raw_output"]
                    else:
                        usage_error = usage_res.get("error")
                except Exception as ue:
                    usage_error = str(ue)

                compute_success = (primary_error is None)
                cleanup_success = (cleanup_status in ("CLEANED", "RUNTIME_STOPPED")) and (cleanup_error is None)
                overall_success = compute_success and cleanup_success

                if isinstance(primary_error, KeyboardInterrupt):
                    job_state = "INTERRUPTED"
                elif primary_error is not None:
                    job_state = "FAILED"
                elif not cleanup_success:
                    job_state = "CLEANUP_FAILED"
                else:
                    job_state = "COMPLETED"

                error_summary_val = None
                if primary_error is not None:
                    error_summary_val = str(primary_error)
                elif cleanup_error is not None:
                    error_summary_val = f"Cleanup failed: {cleanup_error}"

                try:
                    meta = ExperimentMetadata(
                        job_id=job_id,
                        project_name=spec.project_name,
                        requested_accelerator=gpu,
                        start_time=start_time,
                        completion_time=completion_time,
                        job_state=job_state,
                        script_path=str(Path(spec.script_path).resolve()),
                        script_revision=prep.script_sha256[:8],
                        dependencies=[str(spec.requirements_path)] if spec.requirements_path else [],
                        dataset_identifier=spec.dataset_identifier,
                        output_artifact_paths=output_files,
                        success=overall_success,
                        error_summary=error_summary_val,
                        cleanup_status=cleanup_status,
                        cleanup_error=cleanup_error,
                        usage_error=usage_error,
                        compute_units_before=usage_before,
                        compute_units_after=usage_after,
                    )
                    self.tracker.record_job(meta)
                except Exception:
                    pass

                # If execution succeeded but cleanup failed, raise CleanupError now
                if primary_error is None and cleanup_error:
                    raise CleanupError(cleanup_error)

            return JobResult(
                job_id=job_id,
                success=overall_success,
                state=job_state,
                output_files=output_files,
                error=error_summary_val if not overall_success else None,
                cleanup_error=cleanup_error,
                usage_error=usage_error,
                logs=execution_logs,
                cleaned_up=cleanup_success,
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
            "cleanup_error": logged.get("cleanup_error") if logged else None,
            "outputs": logged.get("output_artifact_paths", []) if logged else [],
        }

    def collect_artifacts(self, job_id: str) -> List[str]:
        """Return downloaded local artifacts for a completed job."""
        logged = self.tracker.get_job(job_id)
        if not logged:
            return []
        return logged.get("output_artifact_paths", [])

    def cancel_job(self, session_id: str) -> bool:
        """Cancel and terminate a running managed Colab session.

        Refuses to stop sessions not owned by this adapter to protect unmanaged user workloads.
        """
        if not self.registry.is_managed(session_id):
            print(f"Refusing to cancel unmanaged session '{session_id}'. Adapter only controls its own registered sessions.", file=sys.stderr)
            return False
        res = self._run_cli(["stop", "-s", session_id], timeout=60.0)
        if res.returncode == 0:
            self.registry.update_state(session_id, "TERMINATED")
            return True
        return False

    def cleanup(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Release a specific managed session or all active managed sessions.

        NEVER terminates unmanaged user sessions on the account.
        """
        stopped = []
        failed = []
        unmanaged_ignored = []

        if session_id:
            if not self.registry.is_managed(session_id):
                return {
                    "stopped": [],
                    "failed": [session_id],
                    "unmanaged_ignored": [session_id],
                    "error": f"Refusing to stop unmanaged session '{session_id}'. Adapter only controls its own registered sessions.",
                }
            res = self._run_cli(["stop", "-s", session_id], timeout=60.0)
            if res.returncode == 0:
                stopped.append(session_id)
                self.registry.update_state(session_id, "TERMINATED")
            else:
                failed.append(session_id)
        else:
            avail = self.check_availability()
            for s in avail["active_sessions"]:  # active_sessions contains ONLY managed sessions
                res = self._run_cli(["stop", "-s", s], timeout=60.0)
                if res.returncode == 0:
                    stopped.append(s)
                    self.registry.update_state(s, "TERMINATED")
                else:
                    failed.append(s)
            unmanaged_ignored = avail.get("unmanaged_sessions", [])

        return {"stopped": stopped, "failed": failed, "unmanaged_ignored": unmanaged_ignored}

    def reconcile_stale_sessions(self, auto_stop: bool = False) -> Dict[str, Any]:
        """Inspect and reconcile managed Colab sessions against tracked state.

        Only stops sessions registered to this adapter that are in stale states.
        Leaves unmanaged user sessions strictly untouched.
        """
        avail = self.check_availability()
        active_managed = avail.get("active_sessions", [])
        unmanaged = avail.get("unmanaged_sessions", [])

        stale_managed = []
        all_managed = self.registry.get_all_managed()
        for session_name in active_managed:
            info = all_managed.get(session_name, {})
            # If state is PROVISIONING, INTERRUPTED, or NEEDS_RECONCILIATION, it is stale
            if info.get("state") in ("PROVISIONING", "INTERRUPTED", "NEEDS_RECONCILIATION"):
                stale_managed.append(session_name)

        stopped = []
        failed = []
        if auto_stop and stale_managed:
            for session in stale_managed:
                res = self._run_cli(["stop", "-s", session], timeout=60.0)
                if res.returncode == 0:
                    stopped.append(session)
                    self.registry.update_state(session, "TERMINATED")
                else:
                    failed.append(session)

        return {
            "active_managed": active_managed,
            "stale_managed": stale_managed,
            "unmanaged_ignored": unmanaged,
            "stopped": stopped,
            "failed": failed,
        }
