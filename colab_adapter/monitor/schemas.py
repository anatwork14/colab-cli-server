"""Data models and schemas for Colab Monitoring Dashboard (M8)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AccountSample:
    """Snapshot of Google Colab compute units and active session assignments."""
    sample_id: str
    observed_at: str
    collected_at: str = field(default_factory=_iso_now)
    source: str = "colab usage"
    balance_cu: Optional[float] = None
    rate_cu_per_hour: Optional[float] = None
    active_assignments: int = 0
    net_balance_change: Optional[float] = None
    change_type: str = "UNCHANGED"  # "TOP_UP", "SPEND", "UNCHANGED", "UNATTRIBUTED"
    status: str = "OK"  # "OK", "STALE", "ERROR"
    error_message: Optional[str] = None
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AccountSample":
        valid = {
            "sample_id", "observed_at", "collected_at", "source", "balance_cu",
            "rate_cu_per_hour", "active_assignments", "net_balance_change",
            "change_type", "status", "error_message", "schema_version"
        }
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


@dataclass
class HealthStatus:
    status: str  # "HEALTHY", "DEGRADED", "ERROR"
    monitoring_since: str
    last_collected_at: Optional[str]
    sample_count: int
    is_stale: bool
    collector_backoff_level: int
    uptime_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunSummary:
    run_id: str
    project_id: str
    status: str
    start_time: str
    completion_time: Optional[str] = None
    latest_epoch: Optional[int] = None
    latest_step: Optional[int] = None
    latest_metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoint_count: int = 0
    latest_checkpoint_name: Optional[str] = None
    latest_checkpoint_sha256: Optional[str] = None
    drive_publication_status: str = "PENDING"
    drive_file_id: Optional[str] = None
    source_snapshot_available: bool = False
    is_runtime_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactDetail:
    artifact_id: str
    name: str
    artifact_type: str
    size_bytes: int
    sha256: str
    storage_provider: str
    drive_publication_status: str
    remote_file_id: Optional[str] = None
    verified_at: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



@dataclass
class MonitoringSummary:
    current_balance_cu: Optional[float]
    current_rate_cu_per_hour: Optional[float]
    active_assignments: int
    managed_running_runs: int
    total_runs: int
    monitoring_since: str
    last_sample_time: Optional[str]
    is_stale: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
