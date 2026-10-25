"""Structured training telemetry, experiment provenance, and resume planning.

Part of Milestone M6 (Issue #10) in Colab Project Workspaces V2.
Aligns with M8 Monitoring event ingestion schema (docs/MONITORING_DASHBOARD_V2.md).
Enforces append-only JSONL ledgers, recursive credential redaction (Q12),
accurate failure classification (Q11), and explicit checkpoint resume contracts.
"""
from __future__ import annotations

import fcntl
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .artifacts import ArtifactManifest
from .project_manifest import ProjectSpec
from .tracker import sanitize_data


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TelemetryEvent:
    """Structured telemetry event matching M8 monitoring schema."""
    event_id: str
    run_id: str
    project_id: str
    event_type: str
    emitted_at: str = field(default_factory=_iso_now)
    schema_version: int = 1
    epoch: Optional[int] = None
    step: Optional[int] = None
    total_steps: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    gpu_info: Optional[Dict[str, Any]] = None
    source: str = "adapter"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TelemetryEmitter:
    """Appends sanitized structured training events to local JSONL telemetry ledger."""

    _file_locks: Dict[Path, threading.Lock] = {}
    _file_locks_guard = threading.Lock()

    def __init__(
        self,
        ledger_path: Optional[Union[str, Path]] = None,
        run_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        if ledger_path is None:
            ledger_path = Path.home() / ".colab_experiments" / "telemetry_events.jsonl"
        self.ledger_path = Path(ledger_path).resolve()
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        self.project_id = project_id or "default"
        self._thread_lock = self._get_thread_lock(self.ledger_path)

    @classmethod
    def _get_thread_lock(cls, path: Path) -> threading.Lock:
        with cls._file_locks_guard:
            resolved = path.resolve()
            if resolved not in cls._file_locks:
                cls._file_locks[resolved] = threading.Lock()
            return cls._file_locks[resolved]

    def emit(
        self,
        event_type: str,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        gpu_info: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Emit a structured event, deeply sanitizing any credentials (Q12)."""
        ev_id = f"evt-{uuid.uuid4().hex[:12]}"
        r_id = run_id or self.run_id
        p_id = project_id or self.project_id

        raw_event = {
            "schema_version": 1,
            "event_id": ev_id,
            "run_id": r_id,
            "project_id": p_id,
            "emitted_at": _iso_now(),
            "event_type": event_type,
            "epoch": epoch,
            "step": step,
            "total_steps": total_steps,
            "metrics": metrics or {},
            "gpu_info": gpu_info,
            "source": "trainer_callback",
            "details": details or {},
        }

        # Deep sanitization of credentials/secrets in metrics, details, error messages
        sanitized = sanitize_data(raw_event)

        # Append to JSONL ledger with thread and file lock
        line = json.dumps(sanitized, ensure_ascii=False) + "\n"
        with self._thread_lock:
            fd = os.open(str(self.ledger_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                os.write(fd, line.encode("utf-8"))
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    os.close(fd)
                except OSError:
                    pass

        return sanitized

    def get_events(
        self,
        run_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Read events from ledger, optionally filtering by run_id and/or project_id.

        Records without a project_id never match a named project (they only appear
        when no project_id filter is given).
        """
        if not self.ledger_path.exists():
            return []
        events = []
        with self._thread_lock:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if run_id is not None and record.get("run_id") != run_id:
                            continue
                        if project_id is not None and (record.get("project_id") or "") != project_id:
                            continue
                        events.append(sanitize_data(record))
                    except json.JSONDecodeError:
                        continue
        return events


def plan_resume_from_checkpoint(
    checkpoint: ArtifactManifest,
    spec: ProjectSpec,
    resume_flag: str = "--resume-from",
) -> Dict[str, Any]:
    """Generate deterministic plan to resume training on a newly approved session."""
    target_remote_checkpoint = f"{spec.remote_root}/checkpoints/{checkpoint.rel_path}"
    return {
        "project_id": spec.name,
        "source_checkpoint": {
            "rel_path": checkpoint.rel_path,
            "local_path": checkpoint.abs_path,
            "sha256": checkpoint.sha256,
            "size_bytes": checkpoint.size_bytes,
            "verification_level": checkpoint.verification_level,
        },
        "remote_destination": target_remote_checkpoint,
        "entrypoint": spec.entrypoint,
        "resume_arguments": [resume_flag, target_remote_checkpoint] + spec.args,
        "plan_generated_at": _iso_now(),
    }


def record_interrupted_run(
    error: BaseException,
    cleanup_error: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify execution failure state accurately without masking cleanup errors (Q11)."""
    if isinstance(error, KeyboardInterrupt):
        status = "INTERRUPTED"
    elif cleanup_error:
        status = "CLEANUP_FAILED"
    else:
        status = "FAILED"

    return {
        "status": status,
        "primary_error_type": error.__class__.__name__,
        "primary_error_message": str(error),
        "cleanup_error": cleanup_error,
        "recorded_at": _iso_now(),
    }
