"""Project session lifecycle, lease management, safe attach, and reconciliation.

Part of Milestone M5 (Issue #9) in Colab Project Workspaces V2.
Enforces strict state invariants, collision-resistant session IDs, lease limits,
safe reuse, process locking, and non-interference with unmanaged user sessions.

Verification model (see docs/SECURITY.md): a stop is only recorded as TERMINATED after `colab stop`
exits 0 without a "not found" message AND `colab sessions` no longer lists the id AND, when a baseline
was recorded before `colab new`, `colab usage` active assignments are back at or below that baseline.
Without a baseline the record is marked verification=weak.
"""
import contextlib
import enum
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

from .capability import _strip_ansi
from .errors import (
    AllocationError,
    CleanupError,
    InvalidStateTransitionError,
    LeaseExpiredError,
    RemoteExecutionError,
    SessionNotFoundError,
    TransportError,
    UnmanagedSessionError,
)
from .approval import (
    KIND_V2_UP,
    ApprovalStore,
    approvals_dir_for_registry,
    digest_of,
    ensure_global_cap,
    v2_plan,
)
from .lease_guard import (
    LOCK_WAIT_S,
    _mark_needs_reconciliation,
    child_env,
    ensure_guard_dir,
    epoch_from_iso,
    guard_command,
    guard_log_path,
    guard_process_matches,
    guard_spawn_lock,
    merge_guard_metadata,
    open_private_append,
    validate_session_id,
)
from .project_manifest import ProjectSpec
from .provider import (
    ApprovalRequiredError,
    ColabProcessLock,
    ManagedSessionRegistry,
    is_process_running,
    parse_active_assignments,
)
from .transports.base import BaseTransport
from .transports.official_cli import OfficialCliTransport

LEASE_GUARD_CONFIRM_S = 5.0
# Upstream `colab exec` should time out before our own subprocess budget does.
EXEC_MIN_TIMEOUT_S = 1.0
# Fail-closed path (guard could not start): how long to wait for a busy project lock before giving up.
GUARD_STOP_LOCK_WAIT_S = LOCK_WAIT_S
# A PROVISIONING record older than this is considered abandoned by reconcile.
PROVISIONING_STALE_S = 300.0
STOP_TIMEOUT_S = 60.0
USAGE_TIMEOUT_S = 30.0
SESSIONS_TIMEOUT_S = 60.0
_UNNAMED_SESSION_RE = re.compile(r"^\s*\[\?\]")
EXIT_CODE_MARKER = "EXIT_CODE"
_EXIT_MARKER_RE = re.compile(r"^EXIT_CODE (-?[0-9]+)\s*$", re.MULTILINE)

ORPHAN_WARNING = (
    "NEEDS_RECONCILIATION: a stop failed or could not be verified, or colab new was interrupted. "
    "An unnamed orphan VM may still exist: upstream `colab sessions` lists a server-side assignment that has "
    "no local name only as '[?] <endpoint>'. Check active assignments with `colab --auth=<mode> usage` "
    "(\"Active assignments: N\") and the Colab UI. The record keeps counting toward the global cap until "
    "an operator resolves it; the adapter does not clear it automatically."
)


def _default_spawn(cmd: List[str], **kwargs: Any) -> subprocess.Popen:
    return subprocess.Popen(cmd, **kwargs)


class ProjectSessionState(str, enum.Enum):
    """Lifecycle states for Colab project sessions."""
    PLANNED = "PLANNED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    PROVISIONING = "PROVISIONING"
    READY = "READY"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    TERMINATED = "TERMINATED"
    FAILED = "FAILED"
    NEEDS_RECONCILIATION = "NEEDS_RECONCILIATION"


VALID_TRANSITIONS: Dict[ProjectSessionState, Set[ProjectSessionState]] = {
    ProjectSessionState.PLANNED: {
        ProjectSessionState.AWAITING_APPROVAL,
        ProjectSessionState.PROVISIONING,
        ProjectSessionState.FAILED,
        ProjectSessionState.TERMINATED,
    },
    ProjectSessionState.AWAITING_APPROVAL: {
        ProjectSessionState.PROVISIONING,
        ProjectSessionState.FAILED,
        ProjectSessionState.TERMINATED,
    },
    ProjectSessionState.PROVISIONING: {
        ProjectSessionState.READY,
        ProjectSessionState.FAILED,
        ProjectSessionState.NEEDS_RECONCILIATION,
        ProjectSessionState.TERMINATED,
    },
    ProjectSessionState.READY: {
        ProjectSessionState.RUNNING,
        ProjectSessionState.STOPPING,
        ProjectSessionState.TERMINATED,
        ProjectSessionState.FAILED,
        ProjectSessionState.NEEDS_RECONCILIATION,
    },
    ProjectSessionState.RUNNING: {
        ProjectSessionState.READY,
        ProjectSessionState.STOPPING,
        ProjectSessionState.TERMINATED,
        ProjectSessionState.FAILED,
        ProjectSessionState.NEEDS_RECONCILIATION,
    },
    ProjectSessionState.STOPPING: {
        ProjectSessionState.TERMINATED,
        ProjectSessionState.NEEDS_RECONCILIATION,
        ProjectSessionState.FAILED,
    },
    ProjectSessionState.TERMINATED: set(),
    ProjectSessionState.FAILED: {ProjectSessionState.TERMINATED},
    ProjectSessionState.NEEDS_RECONCILIATION: {
        ProjectSessionState.STOPPING,
        ProjectSessionState.TERMINATED,
        ProjectSessionState.FAILED,
    },
}


class ManagerLockBusyError(RuntimeError):
    """The project lock stayed busy for the whole bounded wait. Nothing was sent to the CLI."""


NOT_FOUND_REASON = "stop reported session not found; runtime may be orphaned"


class StopNotFoundError(CleanupError):
    """`colab stop` exited 0 with a "not found" message, so the session was NOT stopped. `reason` is fixed text."""

    def __init__(self, session_id: str):
        super().__init__(f"Failed to stop session '{session_id}': {NOT_FOUND_REASON}")
        self.reason = NOT_FOUND_REASON


class StopUnverifiedError(CleanupError):
    """`colab stop` exited 0 but the removal could not be verified.

    `listed` is True when `colab sessions` still listed the id (a retry may stop it again) and False when
    only the server-side assignment count was over its baseline (retry verification only; never re-stop).
    """

    def __init__(self, message: str, listed: bool):
        super().__init__(message)
        self.listed = listed


@dataclass
class _Verification:
    ok: bool
    level: Optional[str]
    detail: str
    listed: bool
    count: Optional[int] = None  # active assignments read after the check (None when not read)


def _id_pattern(session_id: str) -> "re.Pattern":
    return re.compile(r"(?<![A-Za-z0-9_\-])" + re.escape(session_id) + r"(?![A-Za-z0-9_\-])")


def _validate_working_dir(working_dir: str) -> str:
    """Working directories must be absolute remote POSIX paths with no control characters and no '..'."""
    if not isinstance(working_dir, str) or not working_dir.startswith("/"):
        raise TransportError("Invalid working directory: must be an absolute remote path.")
    if any(ch in working_dir for ch in ("\n", "\r", "\x00")):
        raise TransportError("Invalid working directory: control characters are not allowed.")
    if ".." in PurePosixPath(working_dir).parts:
        raise TransportError("Invalid working directory: '..' is not allowed.")
    return working_dir


def build_exec_code(command_or_code: str, working_dir: Optional[str] = None, shell: bool = False) -> str:
    """Build the Python source sent to the remote kernel by `colab exec -f`.

    Default (shell=False): the argument IS Python code. A working directory becomes an `os.chdir`
    prologue. shell=True: the argument is a shell command run through `subprocess.run(shell=True)` on
    the managed VM; its stdout, stderr and `EXIT_CODE <n>` are printed to kernel stdout, so the exit
    code survives the upstream CLI (which does not propagate it). Both are built with repr(), so quotes,
    newlines, `&&` and `$` are never interpolated raw.
    """
    if not isinstance(command_or_code, str):
        raise TypeError("command must be a string")
    wd = None if working_dir is None else _validate_working_dir(working_dir)
    if shell:
        return (
            "import subprocess as _colab_sp\n"
            f"_colab_r = _colab_sp.run({command_or_code!r}, shell=True, cwd={wd!r}, "
            "stdout=_colab_sp.PIPE, stderr=_colab_sp.PIPE, universal_newlines=True)\n"
            "print(_colab_r.stdout)\n"
            "if _colab_r.stderr:\n"
            "    print('[stderr]')\n"
            "    print(_colab_r.stderr)\n"
            f"print('{EXIT_CODE_MARKER}', _colab_r.returncode)\n"
        )
    if wd is not None:
        return f"import os\nos.chdir({wd!r})\n" + command_or_code
    return command_or_code


def _attach_shell_exit(result: Dict[str, Any]) -> Dict[str, Any]:
    """Read the EXIT_CODE marker (last whole line). A missing marker is never a success."""
    lines = [ln for ln in (result.get("output") or "").splitlines() if ln.strip()]
    last = _EXIT_MARKER_RE.findall(lines[-1].strip()) if lines else []
    result["shell"] = True
    if len(last) != 1 or lines[-1].strip() != f"{EXIT_CODE_MARKER} {last[0]}":
        # Only a marker on the FINAL non-empty line counts; a marker printed earlier by the command
        # itself (or left behind by truncated output) must never be read as the real exit status.
        result["exit_code"] = None
        result["success"] = False
        return result
    code = int(last[0])
    result["exit_code"] = code
    result["success"] = bool(result.get("success")) and code == 0
    return result


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_stale_provisioning(info: Dict[str, Any], now_ts: float) -> bool:
    try:
        return now_ts - epoch_from_iso(str(info.get("updated_at"))) > PROVISIONING_STALE_S
    except (TypeError, ValueError):
        return True


def _status_warnings(session: "ProjectSession", guard_alive: bool) -> Tuple[List[str], bool]:
    warnings: List[str] = []
    if session.state == ProjectSessionState.NEEDS_RECONCILIATION.value:
        warnings.append(ORPHAN_WARNING)
    if session.state == ProjectSessionState.STOPPING.value:
        warnings.append("STOPPING: a previous stop did not finish. Run 'project reconcile' and check `colab usage`.")
    if (session.state in (ProjectSessionState.READY.value, ProjectSessionState.RUNNING.value)
            and session.is_lease_active() and not guard_alive):
        warnings.append("lease guard is not running while the lease is active. Run 'project exec' or 'project up' "
                        "to respawn it, and check `sessions` and `usage`.")
    return warnings, bool(warnings)


@dataclass
class ProjectSession:
    """Represents a managed remote Colab session bound to a workspace project."""
    session_id: str
    project_id: str
    project_name: str
    state: str
    accelerator: str
    created_at: str
    updated_at: str
    lease_seconds: float
    expires_at: str
    remote_root: str
    owner: str = "adapter"
    pid: int = field(default_factory=os.getpid)
    approved_fingerprint: Optional[str] = None
    approval_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def transition_to(self, new_state: Union[ProjectSessionState, str], reason: Optional[str] = None) -> None:
        """Validate and apply state transition."""
        target_state = ProjectSessionState(new_state)
        current = ProjectSessionState(self.state)
        allowed = VALID_TRANSITIONS.get(current, set())
        if target_state not in allowed and target_state != current:
            raise InvalidStateTransitionError(
                f"Illegal session state transition: '{current.value}' -> '{target_state.value}'. "
                f"Allowed target states: {[s.value for s in allowed]}",
                details={
                    "session_id": self.session_id,
                    "current_state": current.value,
                    "target_state": target_state.value,
                    "reason": reason,
                },
            )
        self.state = target_state.value
        self.updated_at = _iso_now()
        if reason:
            self.metadata["last_transition_reason"] = reason

    def is_lease_active(self, now_ts: Optional[float] = None) -> bool:
        """Check if approved lease duration is still unexpired."""
        if self.state in (ProjectSessionState.TERMINATED.value, ProjectSessionState.FAILED.value):
            return False
        if not self.expires_at:
            return False
        try:
            exp = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00")).timestamp()
            current = time.time() if now_ts is None else now_ts
            return current < exp
        except (ValueError, TypeError):
            return False

    def time_remaining(self, now_ts: Optional[float] = None) -> float:
        """Seconds remaining before approved session lease expires."""
        if not self.expires_at:
            return 0.0
        try:
            exp = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00")).timestamp()
            current = time.time() if now_ts is None else now_ts
            return max(0.0, exp - current)
        except (ValueError, TypeError):
            return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectSession":
        valid_keys = {
            "session_id", "project_id", "project_name", "state", "accelerator",
            "created_at", "updated_at", "lease_seconds", "expires_at", "remote_root",
            "owner", "pid", "approved_fingerprint", "approval_id", "metadata"
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


class ProjectSessionManager:
    """Manages project session lifecycle, leases, and safe execution boundaries."""

    def __init__(
        self,
        registry: Optional[ManagedSessionRegistry] = None,
        lock: Optional[ColabProcessLock] = None,
        transport: Optional[BaseTransport] = None,
        colab_binary: Optional[str] = None,
        run_cmd_fn: Optional[Callable[..., subprocess.CompletedProcess]] = None,
        auth_mode: str = "oauth2",
        spawn_fn: Optional[Callable[..., Any]] = None,
    ):
        self.registry = registry or ManagedSessionRegistry()
        self.lock = lock or ColabProcessLock()
        self.colab_binary = colab_binary or "colab"
        self.auth_mode = auth_mode
        self._run_cmd = run_cmd_fn or subprocess.run
        # Injection point for tests only; production spawns a real detached lease guard.
        self._spawn_fn = spawn_fn or _default_spawn
        # Allocation approval is always enforced inside up(); there is no constructor bypass.
        self.approvals = ApprovalStore(approvals_dir_for_registry(self.registry))
        self.transport = transport or OfficialCliTransport(
            registry=self.registry,
            cli_path=self.colab_binary,
            auth_mode=self.auth_mode,
        )

    @staticmethod
    def generate_session_id(project_name: str) -> str:
        """Create a collision-resistant session name with project tag and random suffix."""
        sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "", project_name.lower())[:12] or "proj"
        suffix = uuid.uuid4().hex[:8]
        return f"colab-v2-{sanitized}-{suffix}"

    def plan_session(self, spec: ProjectSpec, lease_seconds: float = 3600.0) -> ProjectSession:
        """Plan a session offline without allocating cloud resources."""
        session_id = self.generate_session_id(spec.name)
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        exp_iso = datetime.fromtimestamp(now_dt.timestamp() + lease_seconds, timezone.utc).isoformat()
        project_id = getattr(spec, "project_id", None) or f"proj-{spec.name}"
        gpu = getattr(spec, "gpu", "T4")
        remote_root = spec.remote_root or f"/content/colab-workspaces/{spec.name}"

        return ProjectSession(
            session_id=session_id,
            project_id=project_id,
            project_name=spec.name,
            state=ProjectSessionState.PLANNED.value,
            accelerator=gpu,
            created_at=now_iso,
            updated_at=now_iso,
            lease_seconds=lease_seconds,
            expires_at=exp_iso,
            remote_root=remote_root,
            approved_fingerprint=getattr(spec, "manifest_fingerprint", None),
        )

    def get_active_session_for_project(self, project_id: str) -> Optional[ProjectSession]:
        """Find an existing active session belonging to this project if lease is still valid."""
        all_managed = self.registry.get_all_managed()
        for session_id, info in all_managed.items():
            if info.get("project_id") == project_id or info.get("project_name") == project_id:
                state = info.get("state")
                if state in (ProjectSessionState.READY.value, ProjectSessionState.RUNNING.value, "ACTIVE"):
                    session = ProjectSession.from_dict(info)
                    if session.is_lease_active():
                        return session
        return None

    def _v2_plan(self, spec: ProjectSpec) -> Dict[str, Any]:
        return v2_plan(spec, self.colab_binary)

    def _has_live_session(self, project_id: str, gpu: str) -> bool:
        for info in self.registry.get_all_managed().values():
            if info.get("project_id") == project_id and info.get("accelerator") == gpu:
                if info.get("state") in (ProjectSessionState.READY.value, ProjectSessionState.RUNNING.value):
                    if ProjectSession.from_dict(info).is_lease_active():
                        return True
        return False

    def find_reusable_session(self, project_id: str, gpu: str, fingerprint: str) -> Optional[ProjectSession]:
        """Return an active READY/RUNNING session only if project, GPU and plan fingerprint match exactly.

        Legacy sessions without a recorded fingerprint are never reusable without a fresh approval.
        """
        for info in self.registry.get_all_managed().values():
            if info.get("project_id") != project_id or info.get("accelerator") != gpu:
                continue
            if not info.get("approved_fingerprint") or info.get("approved_fingerprint") != fingerprint:
                continue
            if info.get("state") not in (ProjectSessionState.READY.value, ProjectSessionState.RUNNING.value):
                continue
            session = ProjectSession.from_dict(info)
            if session.is_lease_active():
                return session
        return None

    def request_approval(self, spec: ProjectSpec, lease_seconds: float = 3600.0) -> Dict[str, Any]:
        """Write a pending approval for this exact plan. Agent-runnable: no cloud side effects."""
        plan = self._v2_plan(spec)
        lease = float(lease_seconds)
        code = f"{secrets.token_hex(4)} {int(lease)}s"
        summary = (
            f"Allocate a {spec.gpu} Colab runtime for project '{spec.name}' "
            f"(max lease {int(lease)}s); may consume Colab compute units."
        )
        return self.approvals.request(
            kind=KIND_V2_UP,
            plan=plan,
            gpu=spec.gpu,
            summary=summary,
            phrase_code=code,
            max_lease_seconds=lease,
        )

    def approve(self, approval_id: str, require_stdout_tty: bool = True) -> Dict[str, Any]:
        """Operator step: confirm the pending approval at /dev/tty. Never called by agents."""
        return self.approvals.approve(approval_id, require_stdout_tty=require_stdout_tty)

    def up(
        self,
        spec: ProjectSpec,
        lease_seconds: float = 3600.0,
        reuse_active: bool = True,
        approval_id: Optional[str] = None,
    ) -> ProjectSession:
        """Provision or attach a session. A new allocation requires a valid, unconsumed approval.

        After the session is READY (or reused), a detached lease guard is started. If the guard cannot
        be confirmed alive, the session is stopped again (fail closed) and AllocationError is raised.
        Do not Ctrl-C `up`: an interrupt during `colab new` leaves a NEEDS_RECONCILIATION record that
        still counts toward the cap (see ORPHAN_WARNING).
        """
        session = self._up_allocate(
            spec,
            lease_seconds=lease_seconds,
            reuse_active=reuse_active,
            approval_id=approval_id,
        )
        self._guard_or_fail_closed(session)
        return session

    def _up_allocate(
        self,
        spec: ProjectSpec,
        lease_seconds: float = 3600.0,
        reuse_active: bool = True,
        approval_id: Optional[str] = None,
    ) -> ProjectSession:
        """Allocation half of up(): reuse check, global cap, approval verify-and-consume, `colab new`.

        Order inside the ColabProcessLock critical section: reuse check, global cap, approval
        verify-and-consume (single use, before any CLI call), baseline usage read, registry PROVISIONING
        intent, `colab new`.
        """
        if not (0 < float(lease_seconds) <= 86400.0):
            raise ValueError("lease_seconds must be > 0 and <= 86400.")
        project_id = getattr(spec, "project_id", None) or f"proj-{spec.name}"
        gpu = getattr(spec, "gpu", "T4")
        # Fail fast: without an approval, only an exact live session can be reused, so refuse before planning.
        if not approval_id and not self._has_live_session(project_id, gpu):
            raise ApprovalRequiredError(
                f"A human-approved allocation record is required to allocate a {gpu} runtime for "
                f"project '{spec.name}'. Run 'project approve-request', have the operator run "
                "'project approve <id>', then pass --approval-id.",
                reason="missing",
            )
        plan = self._v2_plan(spec)
        fingerprint = plan["fingerprint"]
        plan_digest = digest_of(plan)

        # Reuse of an exact, active match needs no approval and never calls `colab new`.
        if reuse_active:
            existing = self.find_reusable_session(project_id, gpu, fingerprint)
            if existing is not None:
                return existing

        # Acquire cross-process lock to prevent concurrent allocation
        with self.lock:
            if reuse_active:
                existing = self.find_reusable_session(project_id, gpu, fingerprint)
                if existing is not None:
                    return existing

            if not approval_id:
                raise ApprovalRequiredError(
                    f"A human-approved allocation record is required to allocate a {gpu} runtime for "
                    f"project '{spec.name}'. Run 'project approve-request', have the operator run "
                    "'project approve <id>', then pass --approval-id.",
                    reason="missing",
                )
            ensure_global_cap(self.registry)
            self.approvals.verify_and_consume(
                approval_id,
                kind=KIND_V2_UP,
                plan_digest=plan_digest,
                lease_seconds=lease_seconds,
            )

            session = self.plan_session(spec, lease_seconds=lease_seconds)
            session.approved_fingerprint = fingerprint
            session.approval_id = approval_id
            session_id = session.session_id

            # Register intent in registry BEFORE invoking CLI
            now_dt = datetime.now(timezone.utc)
            exp_iso = datetime.fromtimestamp(now_dt.timestamp() + lease_seconds, timezone.utc).isoformat()

            session.expires_at = exp_iso
            # Baseline for server-side verification, read BEFORE `colab new`. None when `usage` is unavailable.
            session.metadata["assignments_before"] = self._assignments_now()
            # Runtimes that already exist without a local name (for example created in the Colab UI); an orphan from a
            # failed `new` is indistinguishable from them except by count, so the baseline is part of the proof.
            session.metadata["unnamed_before"] = self._unnamed_sessions_now()
            session.transition_to(ProjectSessionState.PROVISIONING)

            # Record in registry with full metadata
            with self.registry._lock():
                data = self.registry._read_locked()
                data[session_id] = session.to_dict()
                self.registry._atomic_write_locked(data)

            # Call official CLI to allocate new session: colab new -s <session_id> --gpu <accelerator>
            cmd = [self.colab_binary, f"--auth={self.auth_mode}", "new", "-s", session_id, "--gpu", gpu]
            try:
                res = self._run_cmd(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180.0)
            except subprocess.TimeoutExpired as te:
                self._flag_orphan_risk(session, "Allocation CLI timed out; an unnamed orphan VM may exist")
                raise self._allocation_error(session_id, gpu, "timed out after 180s", False) from te
            except AllocationError:
                raise
            except (FileNotFoundError, PermissionError) as ex:
                # Only the exec of `colab` itself can raise these here (this try covers just the subprocess call):
                # the executable could not be started, so `colab new` never ran and no assignment can exist.
                session.transition_to(ProjectSessionState.FAILED, reason=f"colab could not be executed: {type(ex).__name__}")
                self._commit_session(session)
                raise
            except Exception as ex:
                # Unknown failure while `colab new` may have run: uncertain until proven otherwise.
                proven = self._settle_failed_allocation(session, f"Allocation raised {type(ex).__name__}: {ex}")
                if not proven:
                    raise self._allocation_error(session_id, gpu, f"{type(ex).__name__}: {ex}", False) from ex
                raise
            except BaseException:
                # KeyboardInterrupt / SystemExit during `colab new`: the VM may exist without a local name.
                # The record stays NEEDS_RECONCILIATION (it keeps counting toward the cap); try a stop, re-raise.
                self._flag_orphan_risk(session, "colab new interrupted; an unnamed orphan VM may exist")
                self._best_effort_stop_locked(session)
                raise

            if res.returncode != 0:
                err_msg = (res.stderr or res.stdout or "").strip()
                # A nonzero exit is NOT evidence that no VM was assigned: upstream assigns the VM before it
                # records the local name. Only positive evidence may free the cap.
                proven = self._settle_failed_allocation(
                    session, f"Allocation CLI failed (exit {res.returncode}): {err_msg}"
                )
                raise self._allocation_error(session_id, gpu, err_msg, proven)

            # Transition to READY. A runtime exists now: if READY cannot be recorded, the record must not be left
            # looking idle or vanish from the cap.
            try:
                session.transition_to(ProjectSessionState.READY)
                committed = self._commit_session(session)
            except BaseException as exc:
                self._flag_orphan_risk(
                    session, f"READY could not be recorded ({type(exc).__name__}); a runtime was allocated"
                )
                if isinstance(exc, Exception):
                    # Surface as an uncertain allocation (CLI exit 6 with the orphan warning), not a generic error.
                    raise self._allocation_error(
                        session_id, gpu, f"the READY state could not be recorded ({type(exc).__name__})", False
                    ) from exc
                raise
            if not committed:
                raise self._allocation_error(
                    session_id, gpu, "the session record is missing from the registry after allocation", False
                )
            return session

    def attach(self, session_id: str) -> ProjectSession:
        """Attach to an existing managed session."""
        if not self.registry.is_managed(session_id):
            raise UnmanagedSessionError(
                f"Cannot attach: session '{session_id}' is not registered or owned by this adapter instance.",
                details={"session_id": session_id},
            )

        all_managed = self.registry.get_all_managed()
        info = all_managed.get(session_id)
        if not info:
            raise SessionNotFoundError(
                f"Session '{session_id}' record not found in registry.",
                details={"session_id": session_id},
            )

        session = ProjectSession.from_dict(info)
        if not session.is_lease_active():
            raise LeaseExpiredError(
                f"Session '{session_id}' lease expired at {session.expires_at}.",
                details={"session_id": session_id, "expires_at": session.expires_at},
            )
        return session

    def exec(
        self,
        session_id: str,
        command_or_code: str,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 300.0,
        working_dir: Optional[str] = None,
        shell: bool = False,
    ) -> Dict[str, Any]:
        """Run PYTHON code on a managed session's kernel (or, with shell=True, a shell command on its VM).

        `colab exec` runs its input as kernel Python: the default argument is Python source, and
        `working_dir` becomes an `os.chdir` prologue. With shell=True the command runs through the
        shell on the managed VM only, and success requires an `EXIT_CODE 0` marker. The upstream exit
        status is NOT the command's exit status. Kernel tracebacks are reported as failures.

        The lease guard is (re)started first, so remote work never runs without a deadline guard.
        The timeout is clamped to the remaining lease.
        """
        code = build_exec_code(command_or_code, working_dir=working_dir, shell=shell)
        self.ensure_lease_guard(session_id)
        session = self.attach(session_id)
        remaining = session.time_remaining()
        effective_timeout = min(float(timeout), remaining)
        if effective_timeout < EXEC_MIN_TIMEOUT_S:
            raise LeaseExpiredError(
                f"Only {remaining:.1f}s of lease remain for '{session_id}'; refusing exec.",
                details={"session_id": session_id, "expires_at": session.expires_at},
            )

        session.transition_to(ProjectSessionState.RUNNING)
        with self.registry._lock():
            data = self.registry._read_locked()
            if session_id in data:
                data[session_id] = session.to_dict()
                self.registry._atomic_write_locked(data)

        try:
            result = self.transport.exec_code(
                session_id=session_id,
                code=code,
                env=env,
                timeout=effective_timeout,
            )
            if shell:
                result = _attach_shell_exit(dict(result))
            session.transition_to(ProjectSessionState.READY)
            return result
        except BaseException as e:
            if isinstance(e, KeyboardInterrupt):
                session.transition_to(ProjectSessionState.NEEDS_RECONCILIATION, reason="Interrupted by KeyboardInterrupt")
            else:
                session.transition_to(ProjectSessionState.READY, reason=f"Execution completed with error: {e}")
            raise
        finally:
            self._persist_preserving_guard(session)

    def _persist_preserving_guard(self, session: ProjectSession) -> None:
        """Write the session unless the registry already holds a terminal state.

        The guard may stop the session during a long exec; a stale in-memory copy must not
        resurrect it. Guard metadata written by the guard process is kept as well.
        """
        with self.registry._lock():
            data = self.registry._read_locked()
            current = data.get(session.session_id)
            if current is None:
                return
            if current.get("state") in ("TERMINATED", "FAILED"):
                return
            new_record = session.to_dict()
            merged_md = dict(new_record.get("metadata") or {})
            for key, value in (current.get("metadata") or {}).items():
                if key.startswith("guard_"):
                    merged_md[key] = value
            new_record["metadata"] = merged_md
            data[session.session_id] = new_record
            self.registry._atomic_write_locked(data)

    # ----- upstream CLI helpers (every call goes through self._run_cmd so tests can inject them) -----

    def _cli(self, args: List[str], timeout: float) -> subprocess.CompletedProcess:
        return self._run_cmd(
            [self.colab_binary, f"--auth={self.auth_mode}", *args],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout,
        )

    def _sessions_listed(self, session_id: str) -> Optional[bool]:
        """True/False from `colab sessions`, or None when the check itself failed."""
        try:
            res = self._cli(["sessions"], SESSIONS_TIMEOUT_S)
            out = _strip_ansi(res.stdout or "")
            if res.returncode != 0:
                combined = (out + "\n" + _strip_ansi(res.stderr or "")).lower()
                return False if "no active sessions" in combined else None
            return _id_pattern(session_id).search(out) is not None
        except Exception:  # noqa: BLE001 - unknown means not verified
            return None

    def _assignments_now(self) -> Optional[int]:
        """Active assignments from `colab usage`, or None when unavailable (never guessed as 0)."""
        try:
            res = self._cli(["usage"], USAGE_TIMEOUT_S)
            if res.returncode != 0:
                return None
            return parse_active_assignments(_strip_ansi(res.stdout or ""))
        except Exception:  # noqa: BLE001
            return None

    def _unnamed_sessions_now(self) -> Optional[int]:
        """Number of `[?]` (runtime with no local name) lines in `colab sessions`, or None when unreadable."""
        try:
            res = self._cli(["sessions"], SESSIONS_TIMEOUT_S)
            out = _strip_ansi(res.stdout or "")
            if res.returncode != 0:
                combined = (out + "\n" + _strip_ansi(res.stderr or "")).lower()
                return 0 if "no active sessions" in combined else None
            return sum(1 for line in out.splitlines() if _UNNAMED_SESSION_RE.match(line))
        except Exception:  # noqa: BLE001 - unknown means not proven
            return None

    def _verify_gone(self, session_id: str, baseline: Optional[int]) -> _Verification:
        listed = self._sessions_listed(session_id)
        if listed is None:
            return _Verification(False, None, "could not read `colab sessions`", listed=True)
        if listed:
            return _Verification(False, None, "session still listed by `colab sessions`", listed=True)
        after = self._assignments_now()
        if baseline is not None and after is None:
            # A baseline exists, so an unreadable post-stop usage count is UNVERIFIED, never silently weak.
            return _Verification(False, None, "could not read `colab usage` after stop; baseline is known", listed=False)
        if baseline is None or after is None:
            return _Verification(True, "weak", "usage baseline unavailable; verified with sessions only", listed=False)
        if after <= baseline:
            return _Verification(True, "strong", f"active assignments {after} <= baseline {baseline}",
                                 listed=False, count=after)
        return _Verification(False, None, f"active assignments {after} > baseline {baseline}",
                             listed=False, count=after)

    # ----- registry writes that never flip a newer TERMINATED back -----

    def _commit_session(self, session: ProjectSession) -> bool:
        """Write the in-memory session, re-reading the registry under its lock.

        Refuses when the registry already says TERMINATED and this copy does not. Guard-written
        `guard_*` metadata in the registry is kept.
        """
        with self.registry._lock():
            data = self.registry._read_locked()
            current = data.get(session.session_id)
            if current is None:
                return False
            if (current.get("state") == ProjectSessionState.TERMINATED.value
                    and session.state != ProjectSessionState.TERMINATED.value):
                return False
            new_record = session.to_dict()
            merged = dict(new_record.get("metadata") or {})
            for key, value in (current.get("metadata") or {}).items():
                if key.startswith("guard_"):
                    merged[key] = value
            new_record["metadata"] = merged
            data[session.session_id] = new_record
            self.registry._atomic_write_locked(data)
            return True

    @staticmethod
    def _set_state(session: ProjectSession, target: ProjectSessionState, reason: Optional[str]) -> None:
        try:
            session.transition_to(target, reason=reason)
        except InvalidStateTransitionError:
            # Recovery writes may need a state the normal table does not allow from here; record it explicitly.
            session.state = target.value
            session.updated_at = _iso_now()
            if reason:
                session.metadata["last_transition_reason"] = reason

    # Seconds between the two consecutive "nothing is assigned" readings required before a failed
    # allocation may be recorded as FAILED (a lagging server-side count must not fool a single read).
    FAILURE_SETTLE_SECONDS = 5.0

    def _allocation_failure_evidence(self, session_id: str, baseline: Optional[int],
                                     unnamed_before: Optional[int] = None) -> "tuple":
        """(proven, detail): True only with positive evidence that no assignment remains.

        Requires a known baseline AND a known count of unnamed (`[?]`) runtimes before allocation, and TWO
        consecutive readings separated by FAILURE_SETTLE_SECONDS in which: `colab sessions` is readable and does
        not list the id, `colab usage` is readable and EQUALS the baseline (a different count means something else
        changed concurrently, so absence of our VM cannot be inferred), and no more unnamed `[?]` runtimes exist
        than before. Anything unreadable, unknown, different or inconsistent is NOT proof. Best used with a
        baseline of 0 and no other runtimes on the account.
        """
        if baseline is None:
            return False, "the active-assignment baseline could not be read before allocation"
        if unnamed_before is None:
            return False, "the unnamed-runtime baseline could not be read before allocation"
        for attempt in range(2):
            verdict = self._verify_gone(session_id, baseline)
            if not verdict.ok or verdict.level != "strong":
                return False, verdict.detail
            if verdict.count != baseline:
                return False, (f"active assignments {verdict.count} differ from baseline {baseline}; "
                               "a concurrent change cannot be ruled out")
            unnamed_now = self._unnamed_sessions_now()
            if unnamed_now is None:
                return False, "could not read `colab sessions` for unnamed runtimes"
            if unnamed_now > unnamed_before:
                return False, f"{unnamed_now} unnamed runtime(s) listed, {unnamed_before} before allocation"
            if attempt == 0 and self.FAILURE_SETTLE_SECONDS > 0:
                time.sleep(self.FAILURE_SETTLE_SECONDS)
        return True, f"no assignment remains (two readings, baseline {baseline}, unnamed {unnamed_before})"

    def _settle_failed_allocation(self, session: ProjectSession, reason: str) -> bool:
        """Record a failed allocation. FAILED only when proven empty, else NEEDS_RECONCILIATION (cap kept).

        Never allocates again. Returns True when absence of an assignment was proven.
        """
        baseline = session.metadata.get("assignments_before")
        try:
            proven, detail = self._allocation_failure_evidence(
                session.session_id, baseline, session.metadata.get("unnamed_before")
            )
        except BaseException:  # noqa: BLE001 - evidence gathering must never mask the failure or free the cap
            proven, detail = False, "evidence check was interrupted or failed"
        session.metadata["allocation_failure_evidence"] = detail
        if proven:
            self._set_state(session, ProjectSessionState.FAILED, f"{reason}; {detail}")
            self._commit_session(session)
            return True
        session.metadata["orphan_possible"] = True
        self._set_state(
            session, ProjectSessionState.NEEDS_RECONCILIATION,
            f"{reason}; allocation state uncertain ({detail}); an orphan runtime may exist",
        )
        self._commit_session(session)
        return False

    @classmethod
    def _allocation_error(cls, session_id: str, gpu: str, err_msg: str, proven_empty: bool) -> AllocationError:
        exc = AllocationError(cls._allocation_failure_message(session_id, gpu, err_msg, proven_empty))
        exc.details = {"session_id": session_id, "orphan_possible": not proven_empty}
        exc.orphan_possible = not proven_empty
        return exc

    @staticmethod
    def _allocation_failure_message(session_id: str, gpu: str, err_msg: str, proven_empty: bool) -> str:
        if proven_empty:
            return (
                f"Failed to allocate session '{session_id}' with accelerator '{gpu}': {err_msg} "
                "(two consecutive server-side readings matched the pre-allocation baseline; this lowers the risk of an "
                "orphan but is not an absolute guarantee, so still confirm `colab usage` and the Colab web UI)."
            )
        return (
            f"Allocation of session '{session_id}' ({gpu}) failed or is uncertain: {err_msg}. "
            "POSSIBLE ORPHAN GPU RUNTIME: it could not be proven that no runtime was assigned. The record is "
            "NEEDS_RECONCILIATION and still counts toward the one-session cap; no automatic retry will happen. "
            "Check `colab usage` (Active assignments vs your baseline) and the Colab web UI, stop any runtime "
            "you did not intend, then follow the reconciliation procedure."
        )

    def _flag_orphan_risk(self, session: ProjectSession, reason: str) -> None:
        self._set_state(session, ProjectSessionState.NEEDS_RECONCILIATION, reason)
        session.metadata["orphan_possible"] = True
        self._commit_session(session)

    def _best_effort_stop_locked(self, session: ProjectSession) -> None:
        """Cleanup after an interrupt: must not mask the original exception."""
        try:
            self._stop_locked(session.session_id, session)
        except BaseException:  # noqa: BLE001
            pass

    def _stop_locked(self, session_id: str, session: Optional[ProjectSession] = None) -> bool:
        """Stop a managed session and verify it. The caller must hold self.lock.

        Returns True only when the stop is verified. Raises CleanupError (stop failed, or `colab stop`
        reported "not found") or StopUnverifiedError (exit 0 but not verified). Both leave the record in
        NEEDS_RECONCILIATION, never TERMINATED.
        """
        if session is None:
            session = ProjectSession.from_dict(self.registry.get_all_managed()[session_id])
        baseline = session.metadata.get("assignments_before")
        try:
            session.transition_to(ProjectSessionState.STOPPING)
        except InvalidStateTransitionError:
            pass
        self._commit_session(session)

        res = self._cli(["stop", "-s", session_id], STOP_TIMEOUT_S)
        if res.returncode != 0:
            err_msg = (res.stderr or res.stdout or "").strip()
            self._set_state(session, ProjectSessionState.NEEDS_RECONCILIATION, f"Stop failed: {err_msg}")
            self._commit_session(session)
            raise CleanupError(f"Failed to stop session '{session_id}': {err_msg}")

        # Upstream `colab stop` exits 0 and prints "[colab] Session '<name>' not found." for an unknown name.
        output = _strip_ansi(res.stdout or "") + "\n" + _strip_ansi(res.stderr or "")
        if "not found" in output.lower():
            self._set_state(session, ProjectSessionState.NEEDS_RECONCILIATION, NOT_FOUND_REASON)
            self._commit_session(session)
            raise StopNotFoundError(session_id)

        verdict = self._verify_gone(session_id, baseline)
        if verdict.ok:
            session.metadata["verification"] = verdict.level
            session.metadata["verification_detail"] = verdict.detail
            session.metadata.pop("stop_unverified", None)
            self._set_state(session, ProjectSessionState.TERMINATED, "stopped and verified")
            self._commit_session(session)
            return True
        reason = f"stop returned 0 but removal is not verified: {verdict.detail}"
        # "listed": sessions still lists the id, so a retry may stop again. "usage": only the assignment count
        # is over its baseline, so retries must verify and never re-stop (the stop already succeeded).
        session.metadata["stop_unverified"] = "listed" if verdict.listed else "usage"
        self._set_state(session, ProjectSessionState.NEEDS_RECONCILIATION, reason)
        self._commit_session(session)
        raise StopUnverifiedError(
            f"Stop for '{session_id}' returned success but could not be verified: {verdict.detail}",
            listed=verdict.listed,
        )

    @contextlib.contextmanager
    def _project_lock(self, timeout: Optional[float]) -> Iterator[None]:
        """timeout None: fail immediately when busy (existing behaviour). A number: wait up to that long."""
        if timeout is None:
            with self.lock:
                yield
            return
        deadline = time.monotonic() + float(timeout)
        while not self.lock.acquire(blocking=False):
            if time.monotonic() >= deadline:
                raise ManagerLockBusyError(f"project lock stayed busy for {float(timeout):g}s")
            time.sleep(0.2)
        try:
            yield
        finally:
            self.lock.release()

    def stop(self, session_id: str, *, terminate_guard: bool = True,
             lock_timeout: Optional[float] = None) -> bool:
        """Safely terminate a managed session. Refuses unmanaged sessions.

        Returns True only when the stop is verified (see _stop_locked). After a verified stop the session's
        lease guard is terminated (SIGTERM, verified by cmdline). A failed stop leaves the guard running as
        the deadline backstop. The guard itself passes terminate_guard=False and a lock_timeout.
        """
        if not self.registry.is_managed(session_id):
            raise UnmanagedSessionError(
                f"Refusing to stop unmanaged session '{session_id}'. Adapter only manages its own registered sessions.",
                details={"session_id": session_id},
            )

        with self._project_lock(lock_timeout):
            all_managed = self.registry.get_all_managed()
            info = all_managed.get(session_id, {})
            session = ProjectSession.from_dict(info) if info else None
            if info.get("state") == ProjectSessionState.TERMINATED.value:
                # Already stopped and verified (for example by the lease guard): do not call `colab stop`
                # again, which upstream would answer with a misleading "not found".
                ok = True
            else:
                ok = self._stop_locked(session_id, session)
        if terminate_guard:
            self._terminate_guard(session_id)
        return ok

    def verify_stopped(self, session_id: str, lock_timeout: Optional[float] = None) -> bool:
        """Verification-only retry for a stop that exited 0 but failed the usage baseline check.

        Never issues `colab stop`. Marks the record TERMINATED only when verification now passes.
        """
        with self._project_lock(lock_timeout):
            info = self.registry.get_all_managed().get(session_id)
            if not info:
                return True
            session = ProjectSession.from_dict(info)
            if session.state == ProjectSessionState.TERMINATED.value:
                return True
            verdict = self._verify_gone(session_id, session.metadata.get("assignments_before"))
            if not verdict.ok:
                return False
            session.metadata["verification"] = verdict.level
            session.metadata["verification_detail"] = verdict.detail
            session.metadata.pop("stop_unverified", None)
            self._set_state(session, ProjectSessionState.TERMINATED, "verified after an earlier stop")
            self._commit_session(session)
            return True

    # ----- host-side lease guard -------------------------------------------------------------

    def _guard_pid_of(self, session_id: str) -> Optional[int]:
        info = self.registry.get_all_managed().get(session_id, {})
        pid = (info.get("metadata") or {}).get("guard_pid")
        try:
            return int(pid) if pid is not None else None
        except (TypeError, ValueError):
            return None

    def _terminate_guard(self, session_id: str) -> None:
        pid = self._guard_pid_of(session_id)
        if pid is None or pid == os.getpid():
            return
        if guard_process_matches(pid, session_id):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass

    def start_lease_guard(self, session: ProjectSession, confirm_timeout: float = LEASE_GUARD_CONFIRM_S) -> int:
        """Spawn a detached lease guard for this session and confirm it is alive. Returns its pid.

        Raises AllocationError if spawning fails, the guard exits, or it is not confirmed in time.
        """
        sid = validate_session_id(session.session_id)
        registry_path = Path(self.registry.registry_path)
        deadline = epoch_from_iso(session.expires_at)
        ensure_guard_dir(registry_path)
        # Clear any stale startup marker first: only this incarnation's guard may confirm itself.
        merge_guard_metadata(self.registry, sid, {"guard_started_at": None, "guard_outcome": None})
        log_fd = open_private_append(guard_log_path(registry_path, sid))
        try:
            proc = self._spawn_fn(
                guard_command(sid, registry_path, deadline, self.colab_binary, self.auth_mode),
                stdin=subprocess.DEVNULL,
                stdout=log_fd,
                stderr=log_fd,
                start_new_session=True,
                close_fds=True,
                env=child_env(),
            )
        except Exception as exc:
            raise AllocationError(f"Could not spawn lease guard for '{sid}': {type(exc).__name__}") from exc
        finally:
            os.close(log_fd)

        pid = int(proc.pid)
        merge_guard_metadata(self.registry, sid, {
            "guard_pid": pid,
            "guard_deadline": datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
            "guard_spawned_at": _iso_now(),
        })
        # Confirmation: the child is still running AND the guard itself wrote its startup marker.
        end = time.monotonic() + confirm_timeout
        while True:
            if proc.poll() is not None:
                raise AllocationError(f"Lease guard for '{sid}' exited during startup (code {proc.returncode}).")
            info = self.registry.get_all_managed().get(sid, {})
            meta = info.get("metadata") or {}
            if is_process_running(pid) and self._guard_pid_of(sid) == pid and meta.get("guard_started_at"):
                return pid
            if time.monotonic() >= end:
                if guard_process_matches(pid, sid):
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except OSError:
                        pass
                raise AllocationError(f"Lease guard for '{sid}' was not confirmed alive within {confirm_timeout}s.")
            time.sleep(0.1)

    def ensure_lease_guard(self, session_id: str) -> Dict[str, Any]:
        """Idempotently make sure exactly one live lease guard exists for an active managed lease.

        Respawns a dead guard once. Raises RemoteExecutionError if the guard cannot be (re)started, so
        callers refuse remote work rather than run without a deadline guard.
        """
        if not self.registry.is_managed(session_id):
            raise UnmanagedSessionError(
                f"Refusing lease guard for unmanaged session '{session_id}'.",
                details={"session_id": session_id},
            )
        validate_session_id(session_id)
        with guard_spawn_lock(Path(self.registry.registry_path), session_id):
            info = self.registry.get_all_managed().get(session_id, {})
            session = ProjectSession.from_dict(info)
            if not session.is_lease_active():
                raise LeaseExpiredError(
                    f"Session '{session_id}' lease expired at {session.expires_at}.",
                    details={"session_id": session_id, "expires_at": session.expires_at},
                )
            pid = self._guard_pid_of(session_id)
            if guard_process_matches(pid, session_id):
                return {"guard_pid": pid, "guard_alive": True, "respawned": False}
            try:
                new_pid = self.start_lease_guard(session)
            except Exception as exc:
                raise RemoteExecutionError(
                    f"Lease guard for '{session_id}' is not running and could not be restarted "
                    f"({type(exc).__name__}); refusing remote work without a deadline guard."
                ) from exc
            return {"guard_pid": new_pid, "guard_alive": True, "respawned": True}

    def _guard_or_fail_closed(self, session: ProjectSession) -> None:
        """Used by up(): if the guard cannot be confirmed, flag and stop the session, then raise.

        The record is flagged NEEDS_RECONCILIATION before the stop is attempted, so an interrupt or a busy
        lock can never leave it READY without a guard. An interrupt is re-raised after the best-effort stop.
        """
        sid = session.session_id
        try:
            self.ensure_lease_guard(sid)
        except BaseException as exc:
            _mark_needs_reconciliation(self.registry, sid,
                                       f"lease guard not confirmed ({type(exc).__name__}); stop pending")
            outcome = self._best_effort_stop(sid, lock_wait_s=GUARD_STOP_LOCK_WAIT_S)
            if not isinstance(exc, Exception):
                raise
            verified = outcome == "the session was stopped and verified"
            err = AllocationError(
                f"Lease guard could not be started for '{sid}' ({type(exc).__name__}); {outcome}. "
                "The approval stays consumed."
                + ("" if verified else
                   " POSSIBLE ORPHAN GPU RUNTIME: the session may still be running without a deadline guard; "
                   "check `colab usage` (Active assignments) and the Colab web UI and stop it by hand if needed.")
            )
            err.orphan_possible = not verified
            err.details = {"session_id": sid, "orphan_possible": not verified}
            raise err from exc

    def _best_effort_stop(self, session_id: str, lock_wait_s: float) -> str:
        """Try one verified stop and describe the result accurately for the operator."""
        try:
            self.stop(session_id, terminate_guard=False, lock_timeout=lock_wait_s)
            return "the session was stopped and verified"
        except ManagerLockBusyError:
            return (f"the project lock was busy for {lock_wait_s:g}s, so no stop was issued; "
                    "session flagged NEEDS_RECONCILIATION")
        except StopUnverifiedError:
            return "the stop returned success but could not be verified; session flagged NEEDS_RECONCILIATION"
        except Exception as exc:  # noqa: BLE001 - _stop_locked has already flagged the record
            return f"the stop failed ({type(exc).__name__}); session flagged NEEDS_RECONCILIATION"
        except BaseException:  # noqa: BLE001 - e.g. a second Ctrl-C; the original interrupt is re-raised
            return "the stop was interrupted; session flagged NEEDS_RECONCILIATION"

    def close_session(self, session_id: str) -> bool:
        """Safely close, stop and tear down a managed project session (alias for stop)."""
        return self.stop(session_id)

    def status(self, session_id: str) -> Dict[str, Any]:
        """Return status and lease details for a managed session."""
        if not self.registry.is_managed(session_id):
            return {
                "session_id": session_id,
                "managed": False,
                "error": "Session is not managed by this adapter instance",
            }
        info = self.registry.get_all_managed().get(session_id, {})
        session = ProjectSession.from_dict(info)
        guard_pid = self._guard_pid_of(session_id)
        guard_alive = guard_process_matches(guard_pid, session_id)
        warnings, needs_attention = _status_warnings(session, guard_alive)
        metadata = info.get("metadata") or {}
        return {
            "session_id": session.session_id,
            "project_id": session.project_id,
            "project_name": session.project_name,
            "state": session.state,
            "accelerator": session.accelerator,
            "lease_seconds": session.lease_seconds,
            "expires_at": session.expires_at,
            "time_remaining_seconds": session.time_remaining(),
            "lease_active": session.is_lease_active(),
            "remote_root": session.remote_root,
            "managed": True,
            "guard_pid": guard_pid,
            "guard_alive": guard_alive,
            "guard_deadline": metadata.get("guard_deadline"),
            "guard_outcome": metadata.get("guard_outcome"),
            "seconds_remaining": round(session.time_remaining(), 1),
            "assignments_before": metadata.get("assignments_before"),
            "verification": metadata.get("verification"),
            "orphan_possible": bool(metadata.get("orphan_possible")),
            "warnings": warnings,
            "needs_attention": needs_attention,
        }

    def reconcile_leases(self, auto_stop: bool = True) -> Dict[str, Any]:
        """Audit managed sessions and, unless auto_stop is False, stop the ones that need it.

        Managed ids only. Categories:
        - READY/RUNNING/ACTIVE with an expired lease: stopped (verified) when auto_stop.
        - NEEDS_RECONCILIATION and STOPPING: a stop is attempted when auto_stop.
        - PROVISIONING older than PROVISIONING_STALE_S: a stop is attempted when auto_stop.
        - Healthy READY/RUNNING with an active lease, and fresh PROVISIONING: reported as active.
        Results: stopped (verified now), failed (attempt raised), active, needs_attention (unresolved or
        skipped; includes every NEEDS_RECONCILIATION record), warnings (orphan caveat when any exists).
        """
        stopped: List[str] = []
        failed: List[Dict[str, str]] = []
        active: List[str] = []
        needs_attention: List[Dict[str, str]] = []
        now_ts = time.time()

        with self.lock:
            all_managed = self.registry.get_all_managed()
            for session_id, info in all_managed.items():
                state = info.get("state")
                session = ProjectSession.from_dict(info)
                if state in (ProjectSessionState.READY.value, ProjectSessionState.RUNNING.value, "ACTIVE"):
                    if session.is_lease_active():
                        active.append(session_id)
                        continue
                    category = "lease expired"
                elif state in (ProjectSessionState.NEEDS_RECONCILIATION.value, ProjectSessionState.STOPPING.value):
                    category = f"state {state}"
                elif state == ProjectSessionState.PROVISIONING.value:
                    if not _is_stale_provisioning(info, now_ts):
                        active.append(session_id)
                        continue
                    category = f"PROVISIONING older than {PROVISIONING_STALE_S:g}s"
                else:
                    continue  # PLANNED, AWAITING_APPROVAL, FAILED, TERMINATED: nothing live to reconcile

                if not auto_stop:
                    needs_attention.append({
                        "session_id": session_id, "state": state,
                        "reason": f"{category}; not stopped (--no-auto-stop)",
                    })
                    continue
                try:
                    self._stop_locked(session_id, session)
                    self._terminate_guard(session_id)
                    stopped.append(session_id)
                except Exception as e:  # noqa: BLE001 - reported per session
                    failed.append({"session_id": session_id, "state": state, "error": str(e)})
                    needs_attention.append({
                        "session_id": session_id, "state": state,
                        "reason": f"{category}; stop not verified: {e}",
                    })

            current = self.registry.get_all_managed()
            unresolved = [sid for sid, rec in current.items()
                          if rec.get("state") == ProjectSessionState.NEEDS_RECONCILIATION.value]

        warnings: List[str] = []
        if unresolved:
            warnings.append(f"{ORPHAN_WARNING} Records: {', '.join(sorted(unresolved))}.")
            for sid in unresolved:
                if not any(item["session_id"] == sid for item in needs_attention):
                    needs_attention.append({"session_id": sid, "state": "NEEDS_RECONCILIATION",
                                            "reason": "unresolved; see warnings"})
        return {
            "stopped": stopped,
            "failed": failed,
            "active": active,
            "needs_attention": needs_attention,
            "warnings": warnings,
        }
