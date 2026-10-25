"""Base storage protocols and models for V2 cross-device artifact persistence.

Defines the provider-independent storage interface (ArtifactStore) and
immutable StoredArtifact records with multi-tier integrity guarantees.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_logical_artifact_id(project_id: str, run_id: str, logical_path: str) -> str:
    """Normalizes logical_path = logical_path.lstrip('/') and returns f'{project_id}/{run_id}/{logical_path}'."""
    if "@sha256_" in logical_path:
        logical_path = logical_path.split("@sha256_")[0]
    clean_path = logical_path.lstrip("/")
    prefix = f"{project_id}/{run_id}/"
    if project_id and run_id and clean_path.startswith(prefix):
        clean_path = clean_path[len(prefix):].lstrip("/")
    return f"{project_id}/{run_id}/{clean_path}"


def format_revision_artifact_id(project_id: str, run_id: str, logical_path: str, sha256: str) -> str:
    """If logical_path already contains @sha256_, strip the @sha256_... suffix first.
    Normalizes logical_path = logical_path.lstrip('/') and sha256 = sha256.strip().lower(),
    returning f'{project_id}/{run_id}/{logical_path}@sha256_{sha256}'.
    """
    if "@sha256_" in logical_path:
        logical_path = logical_path.split("@sha256_")[0]
    clean_path = logical_path.lstrip("/")
    prefix = f"{project_id}/{run_id}/"
    if project_id and run_id and clean_path.startswith(prefix):
        clean_path = clean_path[len(prefix):].lstrip("/")
    clean_sha = sha256.strip().lower()
    return f"{project_id}/{run_id}/{clean_path}@sha256_{clean_sha}"


def parse_artifact_id(artifact_id: str) -> Tuple[str, str, str, Optional[str]]:
    """Parses either a versioned ID or unversioned logical ID.
    Returns (project_id, run_id, logical_path, sha256_or_None).
    If artifact_id has fewer than 3 / segments (e.g. just 'checkpoints/model.pt' or
    'checkpoints/model.pt@sha256_...'), returns ('', '', logical_path, sha256_or_None).
    """
    raw = artifact_id.strip()
    sha256: Optional[str] = None
    if "@sha256_" in raw:
        base_id, sha_suffix = raw.split("@sha256_", 1)
        sha256 = sha_suffix.strip().lower()
    else:
        base_id = raw

    clean_base = base_id.lstrip("/")
    parts = clean_base.split("/", 2)
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2], sha256
    else:
        return "", "", clean_base, sha256


def format_transfer_id(provider_name: str, project_id: str, run_id: str, logical_path: str, sha256: str) -> str:
    """Normalizes safe_path = logical_path.lstrip('/').split('@sha256_')[0].replace('/', '_')
    and returns f'{provider_name}-{project_id}-{run_id}-{safe_path}-{sha256.strip().lower()[:16]}'.
    """
    clean_path = logical_path.lstrip("/").split("@sha256_")[0]
    prefix = f"{project_id}/{run_id}/"
    if project_id and run_id and clean_path.startswith(prefix):
        clean_path = clean_path[len(prefix):].lstrip("/")
    safe_path = clean_path.replace("/", "_")
    clean_sha = sha256.strip().lower()[:16]
    return f"{provider_name}-{project_id}-{run_id}-{safe_path}-{clean_sha}"


@dataclass(frozen=True)
class ArtifactRevisionIdentity:
    """Strongly-typed canonical identity for an immutable artifact revision."""
    project_id: str
    run_id: str
    logical_path: str
    sha256: str

    @property
    def logical_id(self) -> str:
        return format_logical_artifact_id(self.project_id, self.run_id, self.logical_path)

    @property
    def revision_id(self) -> str:
        return f"sha256_{self.sha256.strip().lower()}"

    @property
    def artifact_id(self) -> str:
        return format_revision_artifact_id(self.project_id, self.run_id, self.logical_path, self.sha256)


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
    logical_path: Optional[str] = None
    revision_id: Optional[str] = None
    is_latest: bool = True

    def __post_init__(self) -> None:
        if self.logical_path is None:
            _, _, parsed_lp, _ = parse_artifact_id(self.artifact_id)
            lp = parsed_lp or (Path(self.path).name if self.path else "")
            object.__setattr__(self, "logical_path", lp)

        if self.revision_id is None and self.sha256:
            object.__setattr__(self, "revision_id", f"sha256_{self.sha256.strip().lower()}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredArtifact":
        valid = {
            "artifact_id", "storage_provider", "path", "size_bytes",
            "sha256", "project_id", "run_id", "artifact_type",
            "remote_file_id", "md5", "created_at", "verified_at", "metadata",
            "logical_path", "revision_id", "is_latest"
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
        include_history: bool = True,
    ) -> List[StoredArtifact]:
        """List stored artifacts for a project and optional run."""
        ...
