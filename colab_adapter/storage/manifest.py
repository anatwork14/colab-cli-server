"""Durable publication manifest for tracking verified project artifacts.

Supports recording immutable hashes, environment metadata, git provenance,
and atomic disk persistence with checksum validation.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..errors import ArtifactValidationError, StorageError
from .base import (
    StoredArtifact,
    _utc_now_iso,
    format_logical_artifact_id,
    format_revision_artifact_id,
    parse_artifact_id,
)


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


def _extract_logical_path(
    art_dict_or_obj: Union[Dict[str, Any], StoredArtifact],
    project_id: str = "",
    run_id: str = "",
) -> str:
    """Extract and normalize logical path from an artifact dict or StoredArtifact."""
    if isinstance(art_dict_or_obj, StoredArtifact):
        lp = art_dict_or_obj.logical_path
        if not lp:
            _, _, parsed_lp, _ = parse_artifact_id(art_dict_or_obj.artifact_id)
            lp = parsed_lp or (Path(art_dict_or_obj.path).name if art_dict_or_obj.path else "")
        p_id = art_dict_or_obj.project_id or project_id
        r_id = art_dict_or_obj.run_id or run_id
    else:
        lp = art_dict_or_obj.get("logical_path")
        if not lp:
            _, _, parsed_lp, _ = parse_artifact_id(art_dict_or_obj.get("artifact_id", ""))
            lp = parsed_lp or (Path(art_dict_or_obj.get("path", "")).name if art_dict_or_obj.get("path") else "")
        p_id = art_dict_or_obj.get("project_id") or project_id
        r_id = art_dict_or_obj.get("run_id") or run_id

    lp = (lp or "").split("@sha256_")[0].lstrip("/")
    prefix = f"{p_id}/{r_id}/"
    if p_id and r_id and lp.startswith(prefix):
        lp = lp[len(prefix):].lstrip("/")
    return lp


def _build_stored_artifact(art_dict: Dict[str, Any]) -> StoredArtifact:
    return StoredArtifact.from_dict(art_dict)


@dataclass
class PublicationManifest:
    """Manifest describing all published artifacts and provenance for a run."""
    project_id: str
    run_id: str
    manifest_version: int = 2
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    status: str = "IN_PROGRESS"  # IN_PROGRESS, PUBLISHED, PARTIAL, FAILED
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: Optional[bool] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    latest_by_path: Dict[str, str] = field(default_factory=dict)
    migration_notes: List[str] = field(default_factory=list)

    def add_artifact(self, artifact: StoredArtifact) -> StoredArtifact:
        """Register or update an artifact in the manifest."""
        # Determine logical_path
        logical_path = artifact.logical_path
        if not logical_path:
            _, _, parsed_lp, _ = parse_artifact_id(artifact.artifact_id)
            logical_path = parsed_lp or (Path(artifact.path).name if artifact.path else "")

        logical_path = logical_path.split("@sha256_")[0].lstrip("/")
        prefix = f"{artifact.project_id}/{artifact.run_id}/"
        if artifact.project_id and artifact.run_id and logical_path.startswith(prefix):
            logical_path = logical_path[len(prefix):].lstrip("/")

        # Check if artifact_id was explicitly versioned with a @sha256_<hash> suffix that does not match sha256
        _, _, _, sha_suffix = parse_artifact_id(artifact.artifact_id)
        if sha_suffix is not None:
            if artifact.sha256 and sha_suffix.strip().lower() != artifact.sha256.strip().lower():
                raise ArtifactValidationError("Artifact ID sha256 suffix does not match artifact sha256")

        # Compute canonical immutable rev_id
        if artifact.sha256:
            rev_id = format_revision_artifact_id(artifact.project_id, artifact.run_id, logical_path, artifact.sha256)
        else:
            rev_id = artifact.artifact_id

        # Conflicting content check: if rev_id already exists in self.artifacts
        if rev_id in self.artifacts:
            existing = self.artifacts[rev_id]
            ex_sha = existing.get("sha256")
            ex_size = existing.get("size_bytes")
            if (artifact.sha256 and ex_sha and ex_sha.strip().lower() != artifact.sha256.strip().lower()) or (
                artifact.size_bytes is not None and ex_size is not None and ex_size != artifact.size_bytes
            ):
                raise ArtifactValidationError(f"Conflicting content for immutable artifact revision {rev_id}")

        # Check if we are updating an already-superseded revision
        existing = self.artifacts.get(rev_id, {})
        is_historical = bool(
            artifact.is_latest is False
            or artifact.superseded_by is not None
            or (artifact.metadata and artifact.metadata.get("is_latest") is False)
            or (existing and (existing.get("is_latest") is False or existing.get("superseded_by")))
        )

        art_created = existing.get("created_at") or artifact.created_at or ""

        # Check existing latest for this logical path to ensure we never demote the real latest
        current_latest_id = self.latest_by_path.get(logical_path)
        current_latest = self.artifacts.get(current_latest_id) if current_latest_id else None

        if current_latest and not is_historical:
            latest_created = current_latest.get("created_at") or ""
            if latest_created and art_created and latest_created > art_created:
                # An existing latest is strictly newer than this artifact, so this artifact cannot become latest
                is_historical = True

        if not is_historical:
            # Prior revisions for the same logical_path:
            # Demote them to is_latest=False and maintain immediate-successor superseded_by
            for k, v in self.artifacts.items():
                if k != rev_id:
                    v_lp = _extract_logical_path(v, self.project_id, self.run_id)
                    if v_lp == logical_path:
                        v["is_latest"] = False
                        v.setdefault("metadata", {})["is_latest"] = False
                        if not v.get("superseded_by"):
                            v["superseded_by"] = str(rev_id)
                        else:
                            v["superseded_by"] = str(v["superseded_by"])
                        v.setdefault("metadata", {})["superseded_by"] = v["superseded_by"]
                        v.setdefault("metadata", {})["latest_revision_id"] = str(rev_id)

        # Construct updated StoredArtifact
        remote_file_id = artifact.remote_file_id if artifact.remote_file_id is not None else existing.get("remote_file_id")
        merged_metadata = dict(existing.get("metadata", {}))
        if artifact.metadata:
            merged_metadata.update(artifact.metadata)

        rev_str = f"sha256_{artifact.sha256.strip().lower()}" if artifact.sha256 else (artifact.revision_id or "")
        created_at = existing.get("created_at") or artifact.created_at
        path_val = artifact.path or existing.get("path", "")
        md5_val = artifact.md5 or existing.get("md5")
        verified_at = artifact.verified_at or existing.get("verified_at")
        storage_provider = artifact.storage_provider or existing.get("storage_provider", "local")
        artifact_type = artifact.artifact_type or existing.get("artifact_type", "checkpoint")

        # Determine superseded_by_val for this artifact
        superseded_by_val: Optional[str] = None
        if existing and existing.get("superseded_by"):
            superseded_by_val = str(existing["superseded_by"])
        elif artifact.superseded_by:
            superseded_by_val = str(artifact.superseded_by)
        elif is_historical:
            # When a historical revision is inserted AFTER a newer one exists,
            # set its superseded_by only if the successor is known in the manifest
            # (the next revision by created order); otherwise leave None but is_latest False.
            newer_revisions = []
            for k, v in self.artifacts.items():
                if k != rev_id:
                    v_lp = _extract_logical_path(v, self.project_id, self.run_id)
                    if v_lp == logical_path:
                        v_created = v.get("created_at") or ""
                        if v_created > art_created:
                            newer_revisions.append((v_created, k))
            if newer_revisions:
                newer_revisions.sort(key=lambda x: (x[0], x[1]))
                superseded_by_val = str(newer_revisions[0][1])

        is_latest_val = False if (is_historical or superseded_by_val is not None) else True
        if superseded_by_val is not None:
            merged_metadata["superseded_by"] = superseded_by_val
        merged_metadata["is_latest"] = is_latest_val

        updated = StoredArtifact(
            artifact_id=rev_id,
            storage_provider=storage_provider,
            path=path_val,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
            project_id=artifact.project_id,
            run_id=artifact.run_id,
            artifact_type=artifact_type,
            remote_file_id=remote_file_id,
            md5=md5_val,
            created_at=created_at,
            verified_at=verified_at,
            metadata=merged_metadata,
            logical_path=logical_path,
            revision_id=rev_str,
            is_latest=is_latest_val,
            superseded_by=superseded_by_val,
        )

        self.artifacts[rev_id] = updated.to_dict()
        if is_latest_val:
            self.latest_by_path[logical_path] = str(rev_id)
        elif self.latest_by_path.get(logical_path) == str(rev_id):
            self.latest_by_path.pop(logical_path, None)
        self.updated_at = _utc_now_iso()
        return updated

    def get_artifact(self, artifact_id: str) -> Optional[StoredArtifact]:
        """Retrieve an artifact by ID from the manifest."""
        # 1. First, if artifact_id in self.artifacts: return StoredArtifact.from_dict(...)
        if artifact_id in self.artifacts:
            return _build_stored_artifact(self.artifacts[artifact_id])

        # 2. Second, if artifact_id is in self.latest_by_path: resolve rev_id and return
        if artifact_id in self.latest_by_path:
            rev_id = self.latest_by_path[artifact_id]
            if rev_id in self.artifacts:
                art_dict = self.artifacts[rev_id]
                if art_dict.get("is_latest") is True:
                    return _build_stored_artifact(art_dict)

        # 3. Third, parse _, _, logical_path, sha_suffix = parse_artifact_id(artifact_id)
        _, _, logical_path, sha_suffix = parse_artifact_id(artifact_id)
        clean_lp = logical_path.split("@sha256_")[0].lstrip("/")
        prefix = f"{self.project_id}/{self.run_id}/"
        if self.project_id and self.run_id and clean_lp.startswith(prefix):
            clean_lp = clean_lp[len(prefix):].lstrip("/")

        if sha_suffix is None and clean_lp in self.latest_by_path:
            rev_id = self.latest_by_path[clean_lp]
            if rev_id in self.artifacts:
                art_dict = self.artifacts[rev_id]
                if art_dict.get("is_latest") is True:
                    return _build_stored_artifact(art_dict)

        # Fallback: scan self.artifacts.values() for matching logical_path
        candidates = []
        for art_dict in self.artifacts.values():
            art_lp = art_dict.get("logical_path")
            if not art_lp:
                _, _, art_lp, _ = parse_artifact_id(art_dict.get("artifact_id", ""))
            art_lp = (art_lp or "").split("@sha256_")[0].lstrip("/")
            art_prefix = f"{art_dict.get('project_id', '')}/{art_dict.get('run_id', '')}/"
            if art_dict.get("project_id") and art_dict.get("run_id") and art_lp.startswith(art_prefix):
                art_lp = art_lp[len(art_prefix):].lstrip("/")

            full_logical = format_logical_artifact_id(
                art_dict.get("project_id", self.project_id),
                art_dict.get("run_id", self.run_id),
                art_lp,
            )
            clean_req = artifact_id.split("@sha256_")[0].lstrip("/")

            matches_path = (
                art_lp == clean_lp
                or Path(art_lp).name == clean_lp
                or full_logical == clean_req
                or Path(full_logical).name == clean_lp
            )

            if matches_path:
                art_sha = art_dict.get("sha256", "").strip().lower()
                if sha_suffix is not None:
                    if art_sha == sha_suffix.strip().lower():
                        return _build_stored_artifact(art_dict)
                else:
                    candidates.append(art_dict)

        if candidates:
            latest_candidate = next((c for c in candidates if c.get("is_latest") is True), None)
            if latest_candidate:
                return _build_stored_artifact(latest_candidate)
            return None

        return None

    def get_latest_artifact(self, logical_path: str) -> Optional[StoredArtifact]:
        """Resolves the latest revision for logical_path (or full unversioned proj/run/logical_path)."""
        clean_lp = logical_path.split("@sha256_")[0].lstrip("/")
        prefix = f"{self.project_id}/{self.run_id}/"
        if self.project_id and self.run_id and clean_lp.startswith(prefix):
            clean_lp = clean_lp[len(prefix):].lstrip("/")
        rev_id = self.latest_by_path.get(clean_lp) or self.latest_by_path.get(logical_path)
        if rev_id and rev_id in self.artifacts:
            return _build_stored_artifact(self.artifacts[rev_id])
        return self.get_artifact(clean_lp) or self.get_artifact(logical_path)

    def get_latest_revision_id(self, logical_path: str) -> Optional[str]:
        """Returns art.revision_id (i.e. 'sha256_<full_sha256>') for the latest artifact at logical_path, or None."""
        art = self.get_latest_artifact(logical_path)
        if art is not None:
            if art.revision_id:
                return art.revision_id
            if art.sha256:
                return f"sha256_{art.sha256.strip().lower()}"
        return None

    def get_latest_sha256(self, logical_path: str) -> Optional[str]:
        """Returns art.sha256 (i.e. '<full_sha256>') for the latest artifact at logical_path, or None."""
        art = self.get_latest_artifact(logical_path)
        if art is not None:
            if art.sha256:
                return art.sha256.strip().lower()
            if art.revision_id and art.revision_id.startswith("sha256_"):
                return art.revision_id[len("sha256_"):].strip().lower()
        return None

    def list_revisions(self, logical_path: Optional[str] = None) -> List[StoredArtifact]:
        """Returns all revisions (filtered by logical_path if provided), sorted chronologically."""
        results: List[StoredArtifact] = []
        for art_dict in self.artifacts.values():
            art = _build_stored_artifact(art_dict)
            if logical_path is not None:
                target_clean = logical_path.split("@sha256_")[0].lstrip("/")
                art_lp = (art.logical_path or "").split("@sha256_")[0].lstrip("/")
                full_lp = format_logical_artifact_id(art.project_id, art.run_id, art_lp)
                prefix = f"{art.project_id}/{art.run_id}/"
                target_stripped = target_clean[len(prefix):].lstrip("/") if target_clean.startswith(prefix) else target_clean

                matches = (
                    art.logical_path == logical_path
                    or art_lp == target_clean
                    or art_lp == target_stripped
                    or Path(art_lp).name == target_clean
                    or Path(art_lp).name == target_stripped
                    or full_lp == target_clean
                    or full_lp == logical_path
                )
                if not matches:
                    continue
            results.append(art)

        return sorted(results, key=lambda a: (a.created_at or "", a.artifact_id))

    def list_artifacts(self, include_history: bool = True) -> List[StoredArtifact]:
        """List stored artifacts in the manifest."""
        if include_history:
            all_arts = [_build_stored_artifact(d) for d in self.artifacts.values()]
            return sorted(all_arts, key=lambda a: (a.created_at or "", a.artifact_id))

        latest_items: Dict[str, StoredArtifact] = {}
        for lp, rev_id in self.latest_by_path.items():
            if rev_id in self.artifacts:
                latest_items[lp] = _build_stored_artifact(self.artifacts[rev_id])

        for d in self.artifacts.values():
            art = _build_stored_artifact(d)
            lp = art.logical_path or parse_artifact_id(art.artifact_id)[2]
            if art.is_latest and lp not in latest_items:
                latest_items[lp] = art

        return sorted(latest_items.values(), key=lambda a: (a.created_at or "", a.artifact_id))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PublicationManifest":
        valid = {
            "project_id", "run_id", "manifest_version", "created_at",
            "updated_at", "status", "git_commit", "git_branch",
            "git_dirty", "environment", "artifacts", "metadata",
            "latest_by_path", "migration_notes"
        }
        raw_artifacts = dict(data.get("artifacts", {}))
        version = data.get("manifest_version", 1)

        needs_migration = (version < 2) or any("@sha256_" not in k for k in raw_artifacts.keys())

        if needs_migration:
            migrated_artifacts: Dict[str, Dict[str, Any]] = {}
            latest_by_path: Dict[str, str] = {str(k): str(v) for k, v in data.get("latest_by_path", {}).items()}
            migration_notes: List[str] = list(data.get("migration_notes", []))
            project_id = data.get("project_id", "")
            run_id = data.get("run_id", "")

            sorted_raw = sorted(
                raw_artifacts.items(),
                key=lambda kv: (kv[1].get("created_at") or "", kv[0]),
            )

            for old_key, art_dict in sorted_raw:
                art_dict = dict(art_dict)
                p_id = art_dict.get("project_id") or project_id
                r_id = art_dict.get("run_id") or run_id
                sha = art_dict.get("sha256", "")

                log_path = art_dict.get("logical_path")
                if not log_path:
                    _, _, parsed_lp, _ = parse_artifact_id(art_dict.get("artifact_id", old_key))
                    log_path = parsed_lp or (Path(art_dict.get("path", "")).name if art_dict.get("path") else "")
                log_path = log_path.split("@sha256_")[0].lstrip("/")
                prefix = f"{p_id}/{r_id}/"
                if p_id and r_id and log_path.startswith(prefix):
                    log_path = log_path[len(prefix):].lstrip("/")

                if sha:
                    new_key = format_revision_artifact_id(p_id, r_id, log_path, sha)
                else:
                    new_key = old_key

                for existing_k, existing_d in migrated_artifacts.items():
                    if existing_k != new_key and existing_d.get("logical_path") == log_path:
                        existing_d["is_latest"] = False
                        existing_d.setdefault("metadata", {})["is_latest"] = False
                        if not existing_d.get("superseded_by"):
                            existing_d["superseded_by"] = str(new_key)
                        else:
                            existing_d["superseded_by"] = str(existing_d["superseded_by"])
                        existing_d.setdefault("metadata", {})["superseded_by"] = existing_d["superseded_by"]
                        existing_d.setdefault("metadata", {})["latest_revision_id"] = str(new_key)

                art_dict["artifact_id"] = str(new_key)
                art_dict["logical_path"] = str(log_path)
                if sha:
                    art_dict["revision_id"] = f"sha256_{sha.strip().lower()}"
                art_dict["is_latest"] = True

                migrated_artifacts[str(new_key)] = art_dict
                latest_by_path[str(log_path)] = str(new_key)

            migration_note = "Migrated v1 unversioned artifact records to v2 immutable revision IDs; pre-migration overwritten historical revisions cannot be fabricated."
            if migration_note not in migration_notes:
                migration_notes.append(migration_note)

            filtered = {k: v for k, v in data.items() if k in valid}
            filtered["manifest_version"] = 2
            filtered["artifacts"] = migrated_artifacts
            filtered["latest_by_path"] = latest_by_path
            filtered["migration_notes"] = migration_notes
            return cls(**filtered)

        filtered = {k: v for k, v in data.items() if k in valid}
        raw_arts = dict(filtered.get("artifacts", {}))
        clean_arts: Dict[str, Dict[str, Any]] = {}
        for k, v in raw_arts.items():
            d = dict(v)
            if "superseded_by" in d and d["superseded_by"]:
                d["superseded_by"] = str(d["superseded_by"])
                d.setdefault("metadata", {})["superseded_by"] = d["superseded_by"]
            clean_arts[str(k)] = d
        filtered["artifacts"] = clean_arts

        if not filtered.get("latest_by_path"):
            lbp: Dict[str, str] = {}
            for k, art_dict in clean_arts.items():
                lp = art_dict.get("logical_path")
                if not lp:
                    _, _, lp, _ = parse_artifact_id(art_dict.get("artifact_id", k))
                if lp and art_dict.get("is_latest", True):
                    lbp[str(lp)] = str(k)
            filtered["latest_by_path"] = lbp
        else:
            filtered["latest_by_path"] = {
                str(k): str(v) for k, v in filtered["latest_by_path"].items()
            }

        return cls(**filtered)

    def save(self, dest_path: Union[str, Path]) -> Path:
        """Atomically persist manifest to disk via temporary file and replace."""
        dest = Path(dest_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True)

        tmp = dest.parent / f"{dest.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        try:
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with open(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            os.replace(str(tmp), str(dest))
            try:
                os.chmod(dest, 0o600)
            except OSError:
                pass
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

        # Also write/update a backup file Path(str(p) + ".bak") with 0600 permissions
        bak_path = Path(str(dest) + ".bak")
        tmp_bak = dest.parent / f"{bak_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        try:
            fd_bak = os.open(str(tmp_bak), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with open(fd_bak, "w", encoding="utf-8") as f_bak:
                f_bak.write(payload)
                f_bak.flush()
                os.fsync(f_bak.fileno())
            try:
                os.chmod(tmp_bak, 0o600)
            except OSError:
                pass
            os.replace(str(tmp_bak), str(bak_path))
            try:
                os.chmod(bak_path, 0o600)
            except OSError:
                pass
        finally:
            if tmp_bak.exists():
                try:
                    tmp_bak.unlink()
                except OSError:
                    pass

        return dest

    @classmethod
    def load(cls, src_path: Union[str, Path], allow_backup_recovery: bool = True) -> "PublicationManifest":
        """Load manifest from disk with optional backup recovery."""
        p = Path(src_path).resolve()
        bak_path = Path(str(p) + ".bak")

        def _recover_from_bak(reason: str) -> Optional["PublicationManifest"]:
            if bak_path.exists():
                try:
                    with open(bak_path, "r", encoding="utf-8") as fb:
                        bak_data = json.load(fb)
                    manifest = cls.from_dict(bak_data)
                    note = f"Recovered manifest from backup file {bak_path.name} due to {reason}."
                    if note not in manifest.migration_notes:
                        manifest.migration_notes.append(note)
                    return manifest
                except Exception:
                    pass
            return None

        if not p.exists():
            if allow_backup_recovery:
                recovered = _recover_from_bak("missing primary manifest")
                if recovered is not None:
                    return recovered
            raise FileNotFoundError(f"Manifest file not found: {p}")

        try:
            try:
                mode = stat.S_IMODE(p.stat().st_mode)
                if mode & 0o077:
                    os.chmod(str(p), 0o600)
            except OSError:
                pass

            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, StorageError) as jde:
            if allow_backup_recovery:
                recovered = _recover_from_bak("corrupted primary manifest")
                if recovered is not None:
                    return recovered
            raise StorageError(
                f"Manifest at {p} is corrupted: {jde}",
                details={"path": str(p), "error": str(jde)},
            ) from jde
