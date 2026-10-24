"""Base storage protocols and models for V2 cross-device artifact persistence.

Defines the provider-independent storage interface (ArtifactStore) and
immutable StoredArtifact records with multi-tier integrity guarantees.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArtifactType(str, Enum):
    CHECKPOINT = "checkpoint"
    LOG = "log"
    METRICS = "metrics"
    SOURCE_SNAPSHOT = "source_snapshot"
    MANIFEST = "manifest"
    ENVIRONMENT = "environment"
    OUTPUT = "output"


@dataclass(frozen=True)
class StoredArtifact:
    """Immutable record of an artifact committed to an ArtifactStore."""
    artifact_id: str
    storage_provider: str
    path: str
    size_bytes: int
    sha256: str
    project_id: str
    run_id: str
    artifact_type: str = ArtifactType.CHECKPOINT.value
    remote_file_id: Optional[str] = None
    md5: Optional[str] = None
    created_at: str = field(default_factory=_utc_now_iso)
    verified_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredArtifact":
        valid = {
            "artifact_id", "storage_provider", "path", "size_bytes",
            "sha256", "project_id", "run_id", "artifact_type",
            "remote_file_id", "md5", "created_at", "verified_at", "metadata"
        }
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


@runtime_checkable
class ArtifactStore(Protocol):
    """Provider-independent interface for durable artifact storage."""

    @property
    def provider_name(self) -> str:
        """Name of the storage provider (e.g. 'local', 'google_drive')."""
        ...

    def publish(
        self,
        source: Path,
        project_id: str,
        run_id: str,
        artifact_type: str = ArtifactType.CHECKPOINT.value,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StoredArtifact:
        """Publish a local verified artifact into the store idempotently."""
        ...

    def retrieve(
        self,
        artifact_id: str,
        destination: Path,
    ) -> Path:
        """Retrieve a stored artifact by ID to destination path."""
        ...

    def exists(self, artifact_id: str) -> bool:
        """Check if an artifact with the given ID exists in the store."""
        ...

    def get_metadata(self, artifact_id: str) -> Optional[StoredArtifact]:
        """Fetch metadata for an artifact without downloading contents."""
        ...

    def list_artifacts(
        self,
        project_id: str,
        run_id: Optional[str] = None,
    ) -> List[StoredArtifact]:
        """List stored artifacts for a project and optional run."""
        ...
