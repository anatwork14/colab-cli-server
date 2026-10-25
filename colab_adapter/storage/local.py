"""Local filesystem artifact store implementation.

Provides provider-independent storage on local SSD/HDD with staged atomic commits,
manifest updates, and multi-tier checksum verification.
"""
from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..errors import ArtifactNotFoundError, ArtifactValidationError
from .base import (
    ArtifactStore,
    ArtifactType,
    StoredArtifact,
    _utc_now_iso,
    format_revision_artifact_id,
    parse_artifact_id,
)
from .manifest import PublicationManifest, compute_file_md5, compute_file_sha256

_SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


def _validate_safe_id(name: str, field: str = "identifier") -> str:
    if not name or not isinstance(name, str) or not _SAFE_ID_PATTERN.match(name) or ".." in name or "/" in name or "\\" in name:
        raise ArtifactValidationError(
            f"Invalid {field} '{name}': contains illegal characters or path traversal sequence.",
            details={field: str(name)},
        )
    return name


class LocalArtifactStore:
    """ArtifactStore implementation storing files on the local filesystem."""

    def __init__(self, root_dir: Union[str, Path] = "./artifacts"):
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @property
    def provider_name(self) -> str:
        return "local"

    def _get_run_dir(self, project_id: str, run_id: str) -> Path:
        _validate_safe_id(project_id, "project_id")
        _validate_safe_id(run_id, "run_id")
        run_dir = (self.root_dir / "projects" / project_id / "runs" / run_id).resolve()
        try:
            run_dir.relative_to(self.root_dir)
        except ValueError:
            raise ArtifactValidationError("Path traversal escape detected.")
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _get_manifest_path(self, project_id: str, run_id: str) -> Path:
        return self._get_run_dir(project_id, run_id) / "manifest.json"

    def _load_manifest(self, project_id: str, run_id: str) -> PublicationManifest:
        m_path = self._get_manifest_path(project_id, run_id)
        if m_path.exists() or Path(str(m_path) + ".bak").exists():
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
        raw_src = Path(source)
        if raw_src.is_symlink():
            raise ArtifactValidationError(
                f"Source cannot be a symlink: {source}",
                details={"source": str(source)},
            )
        if not raw_src.exists() or not raw_src.is_file():
            raise ArtifactValidationError(
                f"Source file to publish does not exist or is not a regular file: {source}",
                details={"source": str(source)},
            )

        src = raw_src.resolve()
        if src.is_symlink() or not src.is_file():
            raise ArtifactValidationError(
                f"Source file to publish is not a regular file: {src}",
                details={"source": str(src)},
            )

        run_dir = self._get_run_dir(project_id, run_id)
        manifest = self._load_manifest(project_id, run_id)

        # Compute sha256 and size_bytes of src first
        src_sha256 = compute_file_sha256(src)
        if expected_sha256 and expected_sha256.strip().lower() != src_sha256.lower():
            raise ArtifactValidationError(
                f"Checksum mismatch for published file {src.name}: expected {expected_sha256}, got {src_sha256}",
                details={"file": src.name, "expected": expected_sha256, "actual": src_sha256},
            )
        size_bytes = src.stat().st_size
        md5_hash = compute_file_md5(src)

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
        latest_dest_file = dest_dir / target_filename
        rel_path = str(latest_dest_file.relative_to(run_dir))

        # Canonical revision artifact ID
        revision_artifact_id = format_revision_artifact_id(project_id, run_id, rel_path, src_sha256)

        # Revision file destination inside content-addressed .revisions/<sha16>/ directory
        # so Path(rev_dest_file).name remains target_filename while never overwriting prior revisions.
        revs_base_dir = dest_dir / ".revisions"
        revs_base_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(revs_base_dir, 0o700)
        except OSError:
            pass

        rev_dest_dir = revs_base_dir / src_sha256[:16]
        rev_dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(rev_dest_dir, 0o700)
        except OSError:
            pass

        rev_dest_file = rev_dest_dir / target_filename

        # Write immutable revision file via atomic copy with 0600 mode and fsync
        tmp_rev = rev_dest_dir / f"{target_filename}.rev.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        try:
            fd_rev = os.open(str(tmp_rev), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with open(fd_rev, "wb") as f_out, open(src, "rb") as f_in:
                shutil.copyfileobj(f_in, f_out)
                f_out.flush()
                os.fsync(f_out.fileno())
            try:
                os.chmod(tmp_rev, 0o600)
            except OSError:
                pass

            verify_sha = compute_file_sha256(tmp_rev)
            if verify_sha.lower() != src_sha256.lower():
                raise ArtifactValidationError(
                    f"Integrity check failed during staging: expected {src_sha256}, got {verify_sha}",
                    details={"file": target_filename, "expected": src_sha256, "actual": verify_sha},
                )
            os.replace(str(tmp_rev), str(rev_dest_file))
            try:
                os.chmod(rev_dest_file, 0o600)
            except OSError:
                pass
        finally:
            if tmp_rev.exists():
                try:
                    tmp_rev.unlink()
                except OSError:
                    pass

        # Also copy to latest_dest_file with 0600 mode and fsync so caller checking unversioned path finds latest file
        tmp_latest = dest_dir / f"{target_filename}.latest.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        try:
            fd_latest = os.open(str(tmp_latest), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with open(fd_latest, "wb") as f_out, open(rev_dest_file, "rb") as f_in:
                shutil.copyfileobj(f_in, f_out)
                f_out.flush()
                os.fsync(f_out.fileno())
            try:
                os.chmod(tmp_latest, 0o600)
            except OSError:
                pass
            os.replace(str(tmp_latest), str(latest_dest_file))
            try:
                os.chmod(latest_dest_file, 0o600)
            except OSError:
                pass
        finally:
            if tmp_latest.exists():
                try:
                    tmp_latest.unlink()
                except OSError:
                    pass

        stored = StoredArtifact(
            artifact_id=revision_artifact_id,
            storage_provider=self.provider_name,
            path=str(rev_dest_file.resolve()),
            size_bytes=size_bytes,
            sha256=src_sha256,
            md5=md5_hash,
            project_id=project_id,
            run_id=run_id,
            artifact_type=artifact_type,
            verified_at=_utc_now_iso(),
            metadata=metadata or {},
            logical_path=rel_path,
            revision_id=f"sha256_{src_sha256.lower()}",
            is_latest=True,
        )

        manifest.add_artifact(stored)
        manifest.save(self._get_manifest_path(project_id, run_id))
        return stored

    def retrieve(
        self,
        artifact_id: str,
        destination: Union[str, Path],
        expected_sha256: Optional[str] = None,
    ) -> Path:
        """Retrieve an artifact by ID to destination path."""
        meta = self.get_metadata(artifact_id)
        if not meta or not Path(meta.path).exists():
            raise ArtifactNotFoundError(
                f"Artifact '{artifact_id}' not found in local store.",
                details={"artifact_id": artifact_id},
            )

        stored_path = Path(meta.path).resolve()
        try:
            stored_path.relative_to(self.root_dir)
        except ValueError:
            raise ArtifactValidationError(
                f"Path traversal escape detected in artifact path: {stored_path}",
                details={"artifact_id": artifact_id, "path": str(stored_path)},
            )

        if stored_path.is_symlink() or not stored_path.is_file():
            raise ArtifactValidationError(
                f"Stored artifact path is invalid or a symlink: {stored_path}",
                details={"artifact_id": artifact_id, "path": str(stored_path)},
            )

        dest_raw = Path(destination)
        if dest_raw.is_symlink():
            raise ArtifactValidationError(
                f"Destination cannot be a symlink: {destination}",
                details={"destination": str(destination)},
            )

        dest = dest_raw.resolve()
        if dest.is_dir():
            filename = Path(meta.logical_path).name if meta.logical_path else Path(meta.path).name
            dest = dest / filename

        if dest.is_symlink():
            raise ArtifactValidationError(
                f"Resolved destination cannot be a symlink: {dest}",
                details={"destination": str(dest)},
            )

        dest.parent.mkdir(parents=True, exist_ok=True)
        part_file = dest.parent / f"{dest.name}.part.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        try:
            fd_part = os.open(str(part_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with open(fd_part, "wb") as f_out, open(meta.path, "rb") as f_in:
                shutil.copyfileobj(f_in, f_out)
                f_out.flush()
                os.fsync(f_out.fileno())
            try:
                os.chmod(part_file, 0o600)
            except OSError:
                pass

            actual_size = part_file.stat().st_size
            # Verify size
            if meta.size_bytes is not None and actual_size != meta.size_bytes:
                if part_file.exists():
                    try:
                        part_file.unlink()
                    except OSError:
                        pass
                raise ArtifactValidationError(
                    f"Retrieved artifact size mismatch: recorded {meta.size_bytes}, actual {actual_size}",
                    details={
                        "artifact_id": artifact_id,
                        "recorded_size": meta.size_bytes,
                        "actual_size": actual_size,
                        "expected_size": meta.size_bytes,
                    },
                )

            # Verify integrity on retrieval (expected_sha256 or recorded meta.sha256)
            target_sha = expected_sha256 or meta.sha256
            retrieved_hash = compute_file_sha256(part_file)
            if target_sha and retrieved_hash.lower() != target_sha.lower():
                if part_file.exists():
                    try:
                        part_file.unlink()
                    except OSError:
                        pass
                raise ArtifactValidationError(
                    f"Retrieved artifact hash mismatch: recorded {target_sha}, actual {retrieved_hash}",
                    details={
                        "artifact_id": artifact_id,
                        "recorded": target_sha,
                        "actual": retrieved_hash,
                        "expected": target_sha,
                    },
                )

            os.replace(str(part_file), str(dest))
            try:
                os.chmod(dest, 0o600)
            except OSError:
                pass
            return dest
        finally:
            if part_file.exists():
                try:
                    part_file.unlink()
                except OSError:
                    pass

    def exists(self, artifact_id: str) -> bool:
        meta = self.get_metadata(artifact_id)
        return meta is not None and Path(meta.path).exists()

    def get_metadata(self, artifact_id: str) -> Optional[StoredArtifact]:
        project_id, run_id, logical_path, _ = parse_artifact_id(artifact_id)
        if not project_id or not run_id:
            return None
        manifest = self._load_manifest(project_id, run_id)
        art = manifest.get_artifact(artifact_id)
        if art is not None:
            return art

        # Fallback to direct path resolution if no manifest exists or artifact not in manifest
        run_dir = (self.root_dir / "projects" / project_id / "runs" / run_id).resolve()
        direct_file = (run_dir / logical_path).resolve()
        if direct_file.is_file():
            try:
                direct_file.relative_to(self.root_dir)
                file_sha = compute_file_sha256(direct_file)
                file_size = direct_file.stat().st_size
                return StoredArtifact(
                    artifact_id=artifact_id,
                    storage_provider=self.provider_name,
                    path=str(direct_file),
                    size_bytes=file_size,
                    sha256=file_sha,
                    project_id=project_id,
                    run_id=run_id,
                    logical_path=logical_path,
                    revision_id=f"sha256_{file_sha.lower()}",
                    is_latest=True,
                )
            except ValueError:
                pass
        return None

    def list_artifacts(
        self,
        project_id: str,
        run_id: Optional[str] = None,
        include_history: bool = True,
    ) -> List[StoredArtifact]:
        if run_id:
            return self._load_manifest(project_id, run_id).list_artifacts(include_history=include_history)

        # List across all runs for project
        proj_dir = self.root_dir / "projects" / project_id / "runs"
        if not proj_dir.is_dir():
            return []

        results = []
        for r_dir in sorted(proj_dir.iterdir()):
            if r_dir.is_dir() and ((r_dir / "manifest.json").exists() or (r_dir / "manifest.json.bak").exists()):
                m = PublicationManifest.load(r_dir / "manifest.json")
                results.extend(m.list_artifacts(include_history=include_history))
        return results
