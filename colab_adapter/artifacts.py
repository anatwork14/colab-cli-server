"""Durable artifact recovery and safe checkpoint verification for Colab Project Workspaces.

Part of Milestone M6 (Issue #10) in Colab Project Workspaces V2.
Enforces multi-tier artifact validation (presence, size, SHA256 hash, ZIP structure,
and safe PyTorch weights_only=True deserialization), staged downloads with atomic rename,
and unmanaged session boundaries.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .errors import ArtifactValidationError, UnmanagedSessionError
from .transports.base import BaseTransport


@dataclass
class ArtifactManifest:
    """Metadata describing a verified local artifact."""
    rel_path: str
    abs_path: str
    size_bytes: int
    sha256: str
    verified_at: str
    verification_level: int
    is_checkpoint: bool = False
    drive_publication_status: Optional[str] = None
    remote_file_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute SHA256 checksum of a file in streaming chunks."""
    p = Path(file_path)
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_local_artifact(
    file_path: Union[str, Path],
    expected_sha256: Optional[str] = None,
    min_size_bytes: int = 1,
    is_checkpoint: bool = False,
) -> ArtifactManifest:
    """Verify local file against multi-tier integrity and safety constraints.

    Tier 1: File existence and non-zero / minimum size.
    Tier 2: SHA256 checksum matching expected hash.
    Tier 3: Safe checkpoint inspection (ZIP format check and optional weights_only load).
    """
    p = Path(file_path).resolve()
    if not p.is_file():
        raise ArtifactValidationError(
            f"Artifact file not found: {p}",
            details={"path": str(p), "check": "existence"},
        )

    size = p.stat().st_size
    if size < min_size_bytes:
        raise ArtifactValidationError(
            f"Artifact file is truncated or empty: {p} ({size} bytes, minimum {min_size_bytes} required)",
            details={"path": str(p), "size_bytes": size, "min_size_bytes": min_size_bytes},
        )

    # Tier 2: Checksum
    actual_hash = compute_file_sha256(p)
    if expected_sha256 is not None:
        if expected_sha256.lower() != actual_hash.lower():
            raise ArtifactValidationError(
                f"Artifact SHA256 checksum mismatch for {p.name}: expected {expected_sha256}, got {actual_hash}",
                details={"path": str(p), "expected_sha256": expected_sha256, "actual_sha256": actual_hash},
            )

    verification_level = 2
    metadata: Dict[str, Any] = {}

    # Check if file looks like a PyTorch checkpoint
    if is_checkpoint or p.suffix.lower() in (".pt", ".pth", ".bin", ".ckpt"):
        is_checkpoint = True
        header = b""
        with open(p, "rb") as f:
            header = f.read(4)

        if header.startswith(b"PK"):
            if not zipfile.is_zipfile(str(p)):
                raise ArtifactValidationError(
                    f"Corrupted ZIP archive for checkpoint {p}: file has ZIP magic bytes but is not a valid archive.",
                    details={"path": str(p)},
                )
            try:
                with zipfile.ZipFile(str(p), "r") as zf:
                    namelist = zf.namelist()
                    metadata["archive_entries"] = namelist[:10]
                    # Verify expected PyTorch archive members
                    has_data_or_pkl = any("data" in n or "pkl" in n for n in namelist)
                    if not has_data_or_pkl:
                        raise ArtifactValidationError(
                            f"PyTorch checkpoint archive missing data/pickle stream: {p}",
                            details={"namelist": namelist[:5]},
                        )
            except (zipfile.BadZipFile, OSError) as bze:
                raise ArtifactValidationError(
                    f"Corrupted ZIP archive for checkpoint {p}: {bze}",
                    details={"path": str(p)},
                ) from bze

        # Level 3: Safe weight deserialization if PyTorch is available locally
        try:
            import torch
            try:
                # Require PyTorch >= 2.0 with weights_only=True
                _ = torch.load(str(p), map_location="cpu", weights_only=True)
                verification_level = 3
                metadata["weights_only_verified"] = True
            except TypeError:
                # Older PyTorch lacks weights_only=True; refuse unsafe unpickling
                metadata["weights_only_verified"] = False
                metadata["warning"] = "PyTorch < 2.0 installed; skipped unsafe unrestricted unpickling"
            except Exception as te:
                raise ArtifactValidationError(
                    f"Safe PyTorch weight loading failed on {p}: {te}",
                    details={"path": str(p), "error": str(te)},
                ) from te
        except ImportError:
            metadata["weights_only_verified"] = False
            metadata["note"] = "Local PyTorch not installed; Level 1 & 2 verified"

    return ArtifactManifest(
        rel_path=p.name,
        abs_path=str(p),
        size_bytes=size,
        sha256=actual_hash,
        verified_at=datetime.now(timezone.utc).isoformat(),
        verification_level=verification_level,
        is_checkpoint=is_checkpoint,
        metadata=metadata,
    )


class ArtifactManager:
    """Handles durable download, multi-tier verification, and atomic commit of remote artifacts."""

    def __init__(
        self,
        transport: BaseTransport,
        local_dest_dir: Optional[Union[str, Path]] = None,
        cloud_store: Optional[Any] = None,
        upload_worker: Optional[Any] = None,
    ):
        self.transport = transport
        self.local_dest_dir = Path(local_dest_dir or "./artifacts").resolve()
        self.local_dest_dir.mkdir(parents=True, exist_ok=True)
        self.cloud_store = cloud_store
        self.upload_worker = upload_worker

    def enqueue_cloud_publish(
        self,
        manifest: ArtifactManifest,
        project_id: str,
        run_id: str,
        artifact_type: str = "checkpoint",
    ) -> Optional[Any]:
        """Enqueue verified artifact for background cloud publication."""
        if not self.upload_worker and not self.cloud_store:
            return None

        if self.upload_worker:
            record = self.upload_worker.enqueue(
                local_path=manifest.abs_path,
                project_id=project_id,
                run_id=run_id,
                artifact_type=artifact_type,
                expected_sha256=manifest.sha256,
                metadata={"verification_level": manifest.verification_level, "rel_path": manifest.rel_path},
            )
            manifest.drive_publication_status = record.status
            return record
        elif self.cloud_store:
            stored = self.cloud_store.publish(
                source=manifest.abs_path,
                project_id=project_id,
                run_id=run_id,
                artifact_type=artifact_type,
                expected_sha256=manifest.sha256,
                metadata={"verification_level": manifest.verification_level},
            )
            manifest.drive_publication_status = "PUBLISHED"
            manifest.remote_file_id = stored.remote_file_id
            return stored
        return None

    def download_and_verify(
        self,
        session_id: str,
        remote_path: str,
        local_filename: Optional[str] = None,
        expected_sha256: Optional[str] = None,
        min_size_bytes: int = 1,
        is_checkpoint: bool = False,
        enqueue_cloud: bool = False,
        project_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> ArtifactManifest:
        """Download remote artifact to a staged temp file, verify it, and atomically commit."""
        filename = local_filename or Path(remote_path).name
        target_path = self.local_dest_dir / filename
        staging_path = self.local_dest_dir / f"{filename}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"

        try:
            # Download to staged temporary location
            self.transport.download_file(
                session_id=session_id,
                remote_path=remote_path,
                local_path=staging_path,
            )

            # Perform multi-tier verification on staged file
            manifest = verify_local_artifact(
                file_path=staging_path,
                expected_sha256=expected_sha256,
                min_size_bytes=min_size_bytes,
                is_checkpoint=is_checkpoint,
            )

            # Atomically commit verified file to target destination
            os.replace(str(staging_path), str(target_path))
            manifest.abs_path = str(target_path)
            manifest.rel_path = filename

            if enqueue_cloud and project_id and run_id:
                try:
                    self.enqueue_cloud_publish(
                        manifest=manifest,
                        project_id=project_id,
                        run_id=run_id,
                        artifact_type="checkpoint" if is_checkpoint else "output",
                    )
                except Exception as ce:
                    # Cloud enqueue failure must NEVER mask successful local verification
                    manifest.drive_publication_status = "FAILED"
                    manifest.metadata["cloud_enqueue_error"] = str(ce)

            return manifest

        except Exception as e:
            # Clean up staged temporary file on any error
            if staging_path.exists():
                try:
                    staging_path.unlink()
                except OSError:
                    pass
            if isinstance(e, (ArtifactValidationError, UnmanagedSessionError)):
                raise
            raise ArtifactValidationError(
                f"Failed to retrieve or verify artifact '{remote_path}' from session '{session_id}': {e}",
                details={"session_id": session_id, "remote_path": remote_path, "error": str(e)},
            ) from e

