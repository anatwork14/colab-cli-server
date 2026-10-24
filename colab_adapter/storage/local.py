"""Local filesystem artifact store implementation.

Provides provider-independent storage on local SSD/HDD with staged atomic commits,
manifest updates, and multi-tier checksum verification.
"""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..errors import ArtifactNotFoundError, ArtifactValidationError
from .base import ArtifactStore, ArtifactType, StoredArtifact, _utc_now_iso
from .manifest import PublicationManifest, compute_file_md5, compute_file_sha256


class LocalArtifactStore:
    """ArtifactStore implementation storing files on the local filesystem."""

    def __init__(self, root_dir: Union[str, Path] = "./artifacts"):
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @property
    def provider_name(self) -> str:
        return "local"

    def _get_run_dir(self, project_id: str, run_id: str) -> Path:
        run_dir = self.root_dir / "projects" / project_id / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _get_manifest_path(self, project_id: str, run_id: str) -> Path:
        return self._get_run_dir(project_id, run_id) / "manifest.json"

    def _load_manifest(self, project_id: str, run_id: str) -> PublicationManifest:
        m_path = self._get_manifest_path(project_id, run_id)
        if m_path.exists():
            return PublicationManifest.load(m_path)
        return PublicationManifest(project_id=project_id, run_id=run_id)

    def publish(
        self,
        source: Union[str, Path],
        project_id: str,
        run_id: str,
        artifact_type: str = ArtifactType.CHECKPOINT.value,
        metadata: Optional[Dict[str, Any]] = None,
        expected_sha256: Optional[str] = None,
    ) -> StoredArtifact:
        """Publish a local file into the store idempotently."""
        src = Path(source).resolve()
        if not src.is_file():
            raise ArtifactValidationError(
                f"Source file to publish does not exist: {src}",
                details={"source": str(src)},
            )

        run_dir = self._get_run_dir(project_id, run_id)
        manifest = self._load_manifest(project_id, run_id)

        # Determine target relative and absolute paths
        target_filename = src.name
        if artifact_type == ArtifactType.CHECKPOINT.value:
            dest_dir = run_dir / "checkpoints"
        elif artifact_type == ArtifactType.SOURCE_SNAPSHOT.value:
            dest_dir = run_dir
        elif artifact_type in (ArtifactType.LOG.value, ArtifactType.METRICS.value):
            dest_dir = run_dir
        else:
            dest_dir = run_dir / "outputs"

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / target_filename
        rel_path = str(dest_file.relative_to(run_dir))

        # Staged copy + atomic rename
        tmp_file = dest_dir / f"{target_filename}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        shutil.copy2(src, tmp_file)

        try:
            sha256 = compute_file_sha256(tmp_file)
            if expected_sha256 and expected_sha256.lower() != sha256.lower():
                raise ArtifactValidationError(
                    f"Checksum mismatch for published file {target_filename}: expected {expected_sha256}, got {sha256}",
                    details={"file": target_filename, "expected": expected_sha256, "actual": sha256},
                )
            size_bytes = tmp_file.stat().st_size
            md5_hash = compute_file_md5(tmp_file)

            os.replace(str(tmp_file), str(dest_file))
        finally:
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass

        artifact_id = f"{project_id}/{run_id}/{rel_path}"
        stored = StoredArtifact(
            artifact_id=artifact_id,
            storage_provider=self.provider_name,
            path=str(dest_file),
            size_bytes=size_bytes,
            sha256=sha256,
            md5=md5_hash,
            project_id=project_id,
            run_id=run_id,
            artifact_type=artifact_type,
            verified_at=_utc_now_iso(),
            metadata=metadata or {},
        )

        manifest.add_artifact(stored)
        manifest.save(self._get_manifest_path(project_id, run_id))
        return stored

    def retrieve(self, artifact_id: str, destination: Union[str, Path]) -> Path:
        """Retrieve an artifact by ID to destination path."""
        meta = self.get_metadata(artifact_id)
        if not meta or not Path(meta.path).exists():
            raise ArtifactNotFoundError(
                f"Artifact '{artifact_id}' not found in local store.",
                details={"artifact_id": artifact_id},
            )

        dest = Path(destination).resolve()
        if dest.is_dir():
            dest = dest / Path(meta.path).name

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / f"{dest.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        shutil.copy2(meta.path, tmp)

        # Verify integrity on retrieval
        retrieved_hash = compute_file_sha256(tmp)
        if retrieved_hash.lower() != meta.sha256.lower():
            if tmp.exists():
                tmp.unlink()
            raise ArtifactValidationError(
                f"Retrieved artifact hash mismatch: recorded {meta.sha256}, actual {retrieved_hash}",
                details={"artifact_id": artifact_id, "recorded": meta.sha256, "actual": retrieved_hash},
            )

        os.replace(str(tmp), str(dest))
        return dest

    def exists(self, artifact_id: str) -> bool:
        meta = self.get_metadata(artifact_id)
        return meta is not None and Path(meta.path).exists()

    def get_metadata(self, artifact_id: str) -> Optional[StoredArtifact]:
        parts = artifact_id.split("/")
        if len(parts) < 3:
            return None
        project_id, run_id = parts[0], parts[1]
        manifest = self._load_manifest(project_id, run_id)
        return manifest.get_artifact(artifact_id)

    def list_artifacts(
        self,
        project_id: str,
        run_id: Optional[str] = None,
    ) -> List[StoredArtifact]:
        if run_id:
            return self._load_manifest(project_id, run_id).list_artifacts()

        # List across all runs for project
        proj_dir = self.root_dir / "projects" / project_id / "runs"
        if not proj_dir.is_dir():
            return []

        results = []
        for r_dir in proj_dir.iterdir():
            if r_dir.is_dir() and (r_dir / "manifest.json").exists():
                m = PublicationManifest.load(r_dir / "manifest.json")
                results.extend(m.list_artifacts())
        return results
