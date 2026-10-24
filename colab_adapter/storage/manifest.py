"""Durable publication manifest for tracking verified project artifacts.

Supports recording immutable hashes, environment metadata, git provenance,
and atomic disk persistence with checksum validation.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import StoredArtifact, _utc_now_iso


def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute SHA256 checksum of a file in streaming chunks."""
    p = Path(file_path)
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_file_md5(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute MD5 checksum of a file in streaming chunks (for Drive v3 comparison)."""
    p = Path(file_path)
    hasher = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class PublicationManifest:
    """Manifest describing all published artifacts and provenance for a run."""
    project_id: str
    run_id: str
    manifest_version: int = 1
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    status: str = "IN_PROGRESS"  # IN_PROGRESS, PUBLISHED, PARTIAL, FAILED
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: Optional[bool] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_artifact(self, artifact: StoredArtifact) -> None:
        """Register or update an artifact in the manifest."""
        self.artifacts[artifact.artifact_id] = artifact.to_dict()
        self.updated_at = _utc_now_iso()

    def get_artifact(self, artifact_id: str) -> Optional[StoredArtifact]:
        """Retrieve an artifact by ID from the manifest."""
        data = self.artifacts.get(artifact_id)
        if not data:
            return None
        return StoredArtifact.from_dict(data)

    def list_artifacts(self) -> List[StoredArtifact]:
        """List all stored artifacts in the manifest."""
        return [StoredArtifact.from_dict(d) for d in self.artifacts.values()]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PublicationManifest":
        valid = {
            "project_id", "run_id", "manifest_version", "created_at",
            "updated_at", "status", "git_commit", "git_branch",
            "git_dirty", "environment", "artifacts", "metadata"
        }
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)

    def save(self, dest_path: Union[str, Path]) -> Path:
        """Atomically persist manifest to disk via temporary file and replace."""
        dest = Path(dest_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / f"{dest.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"

        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())

        os.replace(str(tmp), str(dest))
        return dest

    @classmethod
    def load(cls, src_path: Union[str, Path]) -> "PublicationManifest":
        """Load manifest from disk."""
        p = Path(src_path).resolve()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
