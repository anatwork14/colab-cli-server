"""Project session lifecycle, lease management, safe attach, and reconciliation.

Part of Milestone M5 (Issue #9) in Colab Project Workspaces V2.
Enforces strict state invariants, collision-resistant session IDs, lease limits,
safe reuse, process locking, and non-interference with unmanaged user sessions.
"""
import enum
import json
import os
import re
import secrets
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from .errors import (
    AllocationError,
    CleanupError,
    InvalidStateTransitionError,
    LeaseExpiredError,
    RemoteExecutionError,
    SessionNotFoundError,
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
from .project_manifest import ProjectSpec
from .provider import ApprovalRequiredError, ColabProcessLock, ManagedSessionRegistry
from .transports.base import BaseTransport
from .transports.official_cli import OfficialCliTransport


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


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    ):
        self.registry = registry or ManagedSessionRegistry()
        self.lock = lock or ColabProcessLock()
        self.colab_binary = colab_binary or "colab"
        self._run_cmd = run_cmd_fn or subprocess.run
        # Allocation approval is always enforced inside up(); there is no constructor bypass.
        self.approvals = ApprovalStore(approvals_dir_for_registry(self.registry))
        self.transport = transport or OfficialCliTransport(
            registry=self.registry,
            cli_path=self.colab_binary,
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

    def approve(self, approval_id: str) -> Dict[str, Any]:
        """Operator step: confirm the pending approval at /dev/tty. Never called by agents."""
        return self.approvals.approve(approval_id)

    def up(
        self,
        spec: ProjectSpec,
        lease_seconds: float = 3600.0,
        reuse_active: bool = True,
        approval_id: Optional[str] = None,
    ) -> ProjectSession:
        """Provision or attach a session. A new allocation requires a valid, unconsumed approval.

        Order inside the ColabProcessLock critical section: reuse check, global cap, approval
        verify-and-consume (single use, before any CLI call), registry PROVISIONING intent, `colab new`.
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
            now_iso = now_dt.isoformat()
            exp_iso = datetime.fromtimestamp(now_dt.timestamp() + lease_seconds, timezone.utc).isoformat()

            session.expires_at = exp_iso
            session.transition_to(ProjectSessionState.PROVISIONING)

            # Record in registry with full metadata
            with self.registry._lock():
                data = self.registry._read_locked()
                data[session_id] = session.to_dict()
                self.registry._atomic_write_locked(data)

            # Call official CLI to allocate new session: colab new -s <session_id> --gpu <accelerator>
            cmd = [self.colab_binary, "new", "-s", session_id, "--gpu", gpu]
            try:
                res = self._run_cmd(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180.0)
                if res.returncode != 0:
                    err_msg = (res.stderr or res.stdout or "").strip()
                    session.transition_to(ProjectSessionState.FAILED, reason=f"Allocation CLI failed: {err_msg}")
                    with self.registry._lock():
                        data = self.registry._read_locked()
                        if session_id in data:
                            data[session_id] = session.to_dict()
                            self.registry._atomic_write_locked(data)
                    raise AllocationError(
                        f"Failed to allocate session '{session_id}' with accelerator '{gpu}': {err_msg}"
                    )
            except subprocess.TimeoutExpired as te:
                session.transition_to(ProjectSessionState.NEEDS_RECONCILIATION, reason="Allocation CLI timed out")
                with self.registry._lock():
                    data = self.registry._read_locked()
                    if session_id in data:
                        data[session_id] = session.to_dict()
                        self.registry._atomic_write_locked(data)
                raise AllocationError(f"Session allocation timed out after 180s for '{session_id}'") from te
            except Exception as ex:
                if not isinstance(ex, AllocationError):
                    session.transition_to(ProjectSessionState.FAILED, reason=str(ex))
                    with self.registry._lock():
                        data = self.registry._read_locked()
                        if session_id in data:
                            data[session_id] = session.to_dict()
                            self.registry._atomic_write_locked(data)
                raise

            # Transition to READY
            session.transition_to(ProjectSessionState.READY)
            with self.registry._lock():
                data = self.registry._read_locked()
                if session_id in data:
                    data[session_id] = session.to_dict()
                    self.registry._atomic_write_locked(data)

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
    ) -> Dict[str, Any]:
        """Execute command in an attached session with state validation and lock safety."""
        session = self.attach(session_id)

        session.transition_to(ProjectSessionState.RUNNING)
        with self.registry._lock():
            data = self.registry._read_locked()
            if session_id in data:
                data[session_id] = session.to_dict()
                self.registry._atomic_write_locked(data)

        try:
            # Prepend cd to working_dir if provided
            cmd = command_or_code
            if working_dir:
                cmd = f"cd {working_dir} && {command_or_code}"

            result = self.transport.exec_code(
                session_id=session_id,
                code=cmd,
                env=env,
                timeout=timeout,
            )
            session.transition_to(ProjectSessionState.READY)
            return result
        except BaseException as e:
            if isinstance(e, KeyboardInterrupt):
                session.transition_to(ProjectSessionState.NEEDS_RECONCILIATION, reason="Interrupted by KeyboardInterrupt")
            else:
                session.transition_to(ProjectSessionState.READY, reason=f"Execution completed with error: {e}")
            raise
        finally:
            with self.registry._lock():
                data = self.registry._read_locked()
                if session_id in data:
                    data[session_id] = session.to_dict()
                    self.registry._atomic_write_locked(data)

    def _stop_locked(self, session_id: str, session: Optional[ProjectSession] = None) -> bool:
        """Internal worker to stop a session when self.lock is already held."""
        if session:
            try:
                session.transition_to(ProjectSessionState.STOPPING)
                with self.registry._lock():
                    data = self.registry._read_locked()
                    if session_id in data:
                        data[session_id] = session.to_dict()
                        self.registry._atomic_write_locked(data)
            except InvalidStateTransitionError:
                pass

        cmd = [self.colab_binary, "stop", "-s", session_id]
        res = self._run_cmd(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60.0)
        if res.returncode == 0:
            if session:
                session.transition_to(ProjectSessionState.TERMINATED)
                with self.registry._lock():
                    data = self.registry._read_locked()
                    if session_id in data:
                        data[session_id] = session.to_dict()
                        self.registry._atomic_write_locked(data)
            else:
                self.registry.update_state(session_id, ProjectSessionState.TERMINATED.value)
            return True
        else:
            err_msg = (res.stderr or res.stdout or "").strip()
            if session:
                session.transition_to(ProjectSessionState.NEEDS_RECONCILIATION, reason=f"Stop failed: {err_msg}")
                with self.registry._lock():
                    data = self.registry._read_locked()
                    if session_id in data:
                        data[session_id] = session.to_dict()
                        self.registry._atomic_write_locked(data)
            else:
                self.registry.update_state(session_id, ProjectSessionState.NEEDS_RECONCILIATION.value)
            raise CleanupError(f"Failed to stop session '{session_id}': {err_msg}")

    def stop(self, session_id: str) -> bool:
        """Safely terminate a managed session. Refuses unmanaged sessions."""
        if not self.registry.is_managed(session_id):
            raise UnmanagedSessionError(
                f"Refusing to stop unmanaged session '{session_id}'. Adapter only manages its own registered sessions.",
                details={"session_id": session_id},
            )

        with self.lock:
            all_managed = self.registry.get_all_managed()
            info = all_managed.get(session_id, {})
            session = ProjectSession.from_dict(info) if info else None
            return self._stop_locked(session_id, session)

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
        }

    def reconcile_leases(self, auto_stop: bool = True) -> Dict[str, Any]:
        """Audit all managed sessions, detect expired leases, and optionally auto-stop them."""
        stopped = []
        failed = []
        active = []

        with self.lock:
            all_managed = self.registry.get_all_managed()
            for session_id, info in all_managed.items():
                state = info.get("state")
                if state in (ProjectSessionState.READY.value, ProjectSessionState.RUNNING.value, "ACTIVE"):
                    session = ProjectSession.from_dict(info)
                    if not session.is_lease_active():
                        if auto_stop:
                            try:
                                self._stop_locked(session_id, session)
                                stopped.append(session_id)
                            except Exception as e:
                                failed.append({"session_id": session_id, "error": str(e)})
                        else:
                            active.append(session_id)
                    else:
                        active.append(session_id)

        return {
            "stopped": stopped,
            "failed": failed,
            "active": active,
        }
