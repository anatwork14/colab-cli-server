"""Ubuntu-hosted Workspace Broker for secure cross-device session attachment.

Enables authorized laptops, desktops, and mobile clients to attach to running
or completed Colab runs in strictly READ-ONLY mode. Guarantees:
- Zero implicit cloud VM allocation on view or attach.
- Expired or terminated runtimes remain fully inspectable through durable artifacts.
- No Colab OAuth credentials exposed to client devices.
- Multi-client concurrent attachment with audit logging.
- Strict rejection of unmanaged sessions.
"""
from __future__ import annotations

import hmac
import logging
import os
import re
import secrets
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from .errors import (
    BrokerAuthenticationError,
    BrokerError,
    ColabV2Error,
    SessionNotFoundError,
    UnmanagedSessionError,
)

_SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


def _validate_safe_id(name: str, field_name: str = "identifier") -> str:
    """Validate that identifier contains only alphanumeric, dash, dot, or underscore."""
    if not name or not isinstance(name, str) or not _SAFE_ID_PATTERN.match(name) or ".." in name or "/" in name or "\\" in name:
        raise ColabV2Error(
            f"Invalid {field_name} '{name}': contains illegal characters or path traversal sequence.",
            details={field_name: str(name)},
        )
    return name

from .project_sessions import ProjectSession, ProjectSessionManager, ProjectSessionState
from .provider import ManagedSessionRegistry
from .storage.base import ArtifactStore, StoredArtifact, _utc_now_iso
from .storage.manifest import PublicationManifest
from .telemetry import TelemetryEmitter

logger = logging.getLogger(__name__)


@dataclass
class ClientAttachSession:
    """Record of an attached client session."""
    client_id: str
    run_id: str
    mode: str = "READ_ONLY"  # Initial release is strictly READ_ONLY
    attached_at: str = field(default_factory=_utc_now_iso)
    last_seen_at: str = field(default_factory=_utc_now_iso)


class WorkspaceBroker:
    """Central Ubuntu workspace broker coordinating multi-device read-only inspection."""

    def __init__(
        self,
        registry: ManagedSessionRegistry,
        session_manager: Optional[ProjectSessionManager] = None,
        telemetry: Optional[TelemetryEmitter] = None,
        storage: Optional[ArtifactStore] = None,
        auth_tokens: Optional[Set[str]] = None,
        artifacts_dir: Union[str, Path] = "./artifacts",
    ):
        self.registry = registry
        self.session_manager = session_manager
        self.telemetry = telemetry
        self.storage = storage

        if auth_tokens:
            self.auth_tokens = set(auth_tokens)
        else:
            generated = secrets.token_urlsafe(32)
            self.auth_tokens = {generated}

        self.primary_token = next(iter(self.auth_tokens))
        self.artifacts_dir = Path(artifacts_dir).resolve()

        self._clients: Dict[str, ClientAttachSession] = {}
        self._audit_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def verify_token(self, token: Optional[str]) -> bool:
        """Verify client bearer token against authorized set with constant-time compare."""
        if not token or not isinstance(token, str):
            return False
        token_bytes = token.encode("utf-8")
        for auth_token in self.auth_tokens:
            if hmac.compare_digest(token_bytes, auth_token.encode("utf-8")):
                return True
        return False

    def authenticate_client(self, token: Optional[str]) -> None:
        """Raise BrokerAuthenticationError if client token is invalid."""
        if not self.verify_token(token):
            raise BrokerAuthenticationError("Unauthorized: invalid or missing broker bearer token.")

    def list_runs(
        self,
        project_id: Optional[str] = None,
        token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all known active and historical runs."""
        self.authenticate_client(token)
        if project_id:
            _validate_safe_id(project_id, "project_id")

        runs: Dict[str, Dict[str, Any]] = {}


        # 1. Active runs from managed registry
        all_managed = self.registry.get_all_managed()
        for s_id, s_info in all_managed.items():
            proj = s_info.get("project_name", "default")
            if project_id and proj != project_id:
                continue

            r_id = s_info.get("run_id") or s_id
            runs[r_id] = {
                "run_id": r_id,
                "project_id": proj,
                "session_id": s_id,
                "status": s_info.get("state", "UNKNOWN"),
                "accelerator": s_info.get("gpu") or s_info.get("accelerator", "T4"),
                "is_active": s_info.get("state") in ("READY", "RUNNING", "ACTIVE"),
                "created_at": s_info.get("created_at"),
                "expires_at": s_info.get("expires_at"),
                "source": "registry",
            }

        # 2. Historical runs from persisted manifests
        proj_base = self.artifacts_dir / "projects"
        if proj_base.is_dir():
            target_projects = [project_id] if project_id else [p.name for p in proj_base.iterdir() if p.is_dir()]
            for p_name in target_projects:
                runs_dir = proj_base / p_name / "runs"
                if not runs_dir.is_dir():
                    continue
                for r_dir in runs_dir.iterdir():
                    if not r_dir.is_dir():
                        continue
                    r_id = r_dir.name
                    if r_id in runs:
                        continue  # Already represented with active data

                    m_file = r_dir / "manifest.json"
                    manifest = PublicationManifest.load(m_file) if m_file.exists() else None
                    runs[r_id] = {
                        "run_id": r_id,
                        "project_id": p_name,
                        "session_id": None,
                        "status": "TERMINATED",
                        "accelerator": None,
                        "is_active": False,
                        "created_at": manifest.created_at if manifest else None,
                        "expires_at": None,
                        "source": "durable_manifest",
                    }

        return list(runs.values())

    def _find_historical_manifest(self, run_id: str) -> Optional[PublicationManifest]:
        """Find publication manifest for a historical run if on disk."""
        _validate_safe_id(run_id, "run_id")
        proj_base = self.artifacts_dir / "projects"
        if not proj_base.is_dir():
            return None
        for p_dir in proj_base.iterdir():
            if not p_dir.is_dir():
                continue
            m_path = p_dir / "runs" / run_id / "manifest.json"
            if m_path.is_file():
                try:
                    return PublicationManifest.load(m_path)
                except Exception:
                    pass
        return None

    def attach_run(
        self,
        run_id: str,
        client_id: str,
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Attach to an existing run in strictly read-only mode.

        CRITICAL: Never provisions, triggers, or allocates a Colab VM.
        """
        self.authenticate_client(token)
        _validate_safe_id(run_id, "run_id")
        _validate_safe_id(client_id, "client_id")


        # Check managed session registry
        all_managed = self.registry.get_all_managed()
        matching_session_id = None
        session_info = None

        for s_id, s_info in all_managed.items():
            if s_id == run_id or s_info.get("run_id") == run_id:
                matching_session_id = s_id
                session_info = s_info
                break

        historical_manifest = self._find_historical_manifest(run_id)

        # Strictly refuse unmanaged sessions / unknown runs
        if not session_info and not historical_manifest:
            raise UnmanagedSessionError(
                f"Refusing attach: session/run '{run_id}' is unmanaged or not registered.",
                details={"run_id": run_id},
            )

        # Log audit entry
        with self._lock:
            client_session = ClientAttachSession(client_id=client_id, run_id=run_id, mode="READ_ONLY")
            self._clients[f"{client_id}:{run_id}"] = client_session
            self._audit_log.append({
                "action": "ATTACH_READ_ONLY",
                "client_id": client_id,
                "run_id": run_id,
                "timestamp": _utc_now_iso(),
            })

        # Determine runtime availability
        runtime_available = False
        session_state = "TERMINATED"
        project_id = "default"

        if session_info:
            project_id = session_info.get("project_name", "default")
            raw_state = session_info.get("state", "UNKNOWN")
            # Check lease validity if session manager is present
            if self.session_manager and matching_session_id:
                try:
                    s_obj = self.session_manager.attach(matching_session_id)
                    runtime_available = s_obj.is_lease_active()
                    session_state = s_obj.state.value
                except Exception:
                    runtime_available = False
                    session_state = "LEASE_EXPIRED"
            else:
                session_state = raw_state
                runtime_available = raw_state in ("READY", "RUNNING", "ACTIVE")
        elif historical_manifest:
            project_id = historical_manifest.project_id
            runtime_available = False
            session_state = "TERMINATED"


        # Discover artifacts from local manifest
        artifacts = self.get_run_artifacts(run_id, project_id=project_id, token=token)
        latest_checkpoint = None
        for a in artifacts:
            if a.get("artifact_type") == "checkpoint":
                latest_checkpoint = a

        # Telemetry events
        events: List[Dict[str, Any]] = []
        if self.telemetry:
            events = [
                e for e in self.telemetry.get_events()
                if e.get("run_id") == run_id or e.get("session_id") == matching_session_id
            ]

        return {
            "run_id": run_id,
            "project_id": project_id,
            "session_id": matching_session_id,
            "mode": "READ_ONLY",
            "runtime_status": session_state if runtime_available else "RUNTIME_UNAVAILABLE",
            "is_runtime_active": runtime_available,
            "attached_client_id": client_id,
            "attached_at": client_session.attached_at,
            "latest_checkpoint": latest_checkpoint,
            "artifacts_count": len(artifacts),
            "artifacts": artifacts,
            "telemetry_events_count": len(events),
            "audit_note": "Read-only attach verified. Zero GPU resources allocated.",
        }

    def get_run_artifacts(
        self,
        run_id: str,
        project_id: Optional[str] = None,
        token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch list of verified local and Drive-published artifacts."""
        self.authenticate_client(token)
        _validate_safe_id(run_id, "run_id")
        if project_id:
            _validate_safe_id(project_id, "project_id")


        results: List[Dict[str, Any]] = []

        # Find project directory
        proj_base = self.artifacts_dir / "projects"
        candidates = [project_id] if project_id else []
        if not candidates and proj_base.is_dir():
            candidates = [p.name for p in proj_base.iterdir() if p.is_dir()]

        for p_name in candidates:
            run_m_file = proj_base / p_name / "runs" / run_id / "manifest.json"
            if run_m_file.exists():
                manifest = PublicationManifest.load(run_m_file)
                for a in manifest.list_artifacts():
                    results.append(a.to_dict())

        return results

    def request_write_lease(
        self,
        run_id: str,
        client_id: str,
        authorization_token: str,
    ) -> Dict[str, Any]:
        """Future extension point: single-writer lease with human approval.

        Strictly disabled in initial read-only release.
        """
        raise NotImplementedError(
            "Writable cross-device execution is disabled in the initial V2 release. "
            "All cross-device workspace attachments are strictly READ-ONLY."
        )

    def get_audit_log(self, token: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return broker attach and access audit history."""
        self.authenticate_client(token)
        with self._lock:
            return list(self._audit_log)
