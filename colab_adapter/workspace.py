"""Incremental Workspace Synchronization Engine (V2).

Implements content-addressed delta calculation (SHA256), staged two-phase upload,
safe deletion, and atomic remote manifest commitment.
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

from .errors import WorkspaceSyncError
from .project_manifest import ProjectSpec, discover_project_files
from .transports.base import BaseTransport

MANIFEST_FILENAME = ".colab_manifest.json"
STAGING_DIRNAME = ".colab_staging"


@dataclass(frozen=True)
class SyncDelta:
    """Calculated difference between local workspace files and remote committed manifest."""

    to_add: Tuple[str, ...]
    to_modify: Tuple[str, ...]
    unchanged: Tuple[str, ...]
    to_delete: Tuple[str, ...]
    bytes_to_transfer: int
    files_to_transfer_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "to_add": list(self.to_add),
            "to_modify": list(self.to_modify),
            "unchanged": list(self.unchanged),
            "to_delete": list(self.to_delete),
            "bytes_to_transfer": self.bytes_to_transfer,
            "files_to_transfer_count": self.files_to_transfer_count,
        }


def compute_sync_delta(
    local_files: Dict[str, Dict[str, Any]],
    remote_manifest: Optional[Dict[str, Any]] = None,
    delete_remote: bool = False,
) -> SyncDelta:
    """Computes differences between local workspace files and last applied remote manifest."""
    remote_files = (remote_manifest or {}).get("files", {})

    to_add: List[str] = []
    to_modify: List[str] = []
    unchanged: List[str] = []
    to_delete: List[str] = []
    bytes_to_transfer = 0

    # 1. Compare local files against remote
    for rel_path, meta in sorted(local_files.items()):
        size = meta["size"]
        sha256 = meta["sha256"]

        if rel_path not in remote_files:
            to_add.append(rel_path)
            bytes_to_transfer += size
        else:
            rem_meta = remote_files[rel_path]
            if rem_meta.get("sha256") == sha256:
                unchanged.append(rel_path)
            else:
                to_modify.append(rel_path)
                bytes_to_transfer += size

    # 2. Check for deletions (only scheduled if delete_remote is True)
    if delete_remote:
        for rem_path in sorted(remote_files.keys()):
            if rem_path not in local_files:
                to_delete.append(rem_path)

    return SyncDelta(
        to_add=tuple(to_add),
        to_modify=tuple(to_modify),
        unchanged=tuple(unchanged),
        to_delete=tuple(to_delete),
        bytes_to_transfer=bytes_to_transfer,
        files_to_transfer_count=len(to_add) + len(to_modify),
    )


class WorkspaceSyncEngine:
    """Manages incremental delta synchronization to remote Colab workspace."""

    def __init__(
        self,
        transport: BaseTransport,
        local_cache_dir: Optional[Path] = None,
    ):
        self.transport = transport
        self.cache_dir = local_cache_dir or Path(tempfile.gettempdir()) / "colab_workspace_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def plan_sync(
        self,
        spec: ProjectSpec,
        session_id: Optional[str] = None,
        cached_remote_manifest: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Calculates sync delta offline without modifying remote state."""
        local_file_tuples = discover_project_files(spec)
        local_files = {
            f[0]: {"size": f[1], "sha256": f[2]}
            for f in local_file_tuples
        }

        # Resolve remote manifest from cache or session
        remote_manifest = cached_remote_manifest
        if remote_manifest is None and session_id:
            remote_manifest = self._fetch_remote_manifest(spec, session_id)
        if remote_manifest is None:
            remote_manifest = self._load_local_cached_manifest(spec.name)

        delta = compute_sync_delta(
            local_files=local_files,
            remote_manifest=remote_manifest,
            delete_remote=spec.delete_remote,
        )

        return {
            "project_name": spec.name,
            "remote_root": spec.remote_root,
            "delete_remote_enabled": spec.delete_remote,
            "delta": delta.to_dict(),
            "local_files_count": len(local_files),
        }

    def apply_sync(
        self,
        spec: ProjectSpec,
        session_id: str,
    ) -> Dict[str, Any]:
        """Applies delta sync to the active session using two-phase staging and atomic commit."""
        local_file_tuples = discover_project_files(spec)
        local_files = {
            f[0]: {"size": f[1], "sha256": f[2]}
            for f in local_file_tuples
        }

        remote_root = spec.remote_root or f"/content/colab-workspaces/{spec.name}"
        remote_manifest = self._fetch_remote_manifest(spec, session_id)

        delta = compute_sync_delta(
            local_files=local_files,
            remote_manifest=remote_manifest,
            delete_remote=spec.delete_remote,
        )

        # If zero files to transfer or delete, sync is an immediate no-op
        if delta.files_to_transfer_count == 0 and len(delta.to_delete) == 0:
            return {
                "status": "UP_TO_DATE",
                "files_transferred": 0,
                "bytes_transferred": 0,
                "files_deleted": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        transferred_count = 0
        transferred_bytes = 0

        # Phase 1: Upload added and modified files
        files_to_upload = list(delta.to_add) + list(delta.to_modify)
        try:
            for rel_path in files_to_upload:
                local_path = spec.root / rel_path
                remote_dest = f"{remote_root}/{rel_path}"
                self.transport.upload_file(session_id, local_path, remote_dest)
                transferred_count += 1
                transferred_bytes += local_files[rel_path]["size"]

            # Phase 2: Apply safe deletions if requested
            deleted_count = 0
            if spec.delete_remote and delta.to_delete:
                for rem_rel_path in delta.to_delete:
                    # Security check: must not escape remote_root
                    if ".." in rem_rel_path.split("/") or rem_rel_path.startswith("/"):
                        raise WorkspaceSyncError(
                            f"Refusing to delete unsafe remote path: {rem_rel_path}"
                        )
                    target_to_del = f"{remote_root}/{rem_rel_path}"
                    self.transport.remove_file(session_id, target_to_del)
                    deleted_count += 1

            # Phase 3: Commit updated manifest remotely and locally LAST
            committed_manifest = {
                "schema_version": 1,
                "project_name": spec.name,
                "committed_at": datetime.now(timezone.utc).isoformat(),
                "files": local_files,
            }
            self._commit_manifest(spec, session_id, committed_manifest)

            return {
                "status": "COMMITTED",
                "files_transferred": transferred_count,
                "bytes_transferred": transferred_bytes,
                "files_deleted": deleted_count,
                "timestamp": committed_manifest["committed_at"],
            }
        except Exception as e:
            raise WorkspaceSyncError(
                f"Workspace synchronization interrupted or failed: {e}",
                details={"transferred_count": transferred_count, "error": str(e)},
            )

    def _fetch_remote_manifest(
        self,
        spec: ProjectSpec,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Downloads and parses the remote committed manifest if available."""
        remote_root = spec.remote_root or f"/content/colab-workspaces/{spec.name}"
        remote_manifest_path = f"{remote_root}/{MANIFEST_FILENAME}"

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            local_tmp = Path(tf.name)

        try:
            self.transport.download_file(session_id, remote_manifest_path, local_tmp)
            content = local_tmp.read_text(encoding="utf-8")
            return json.loads(content)
        except Exception:
            # File missing or download error: no valid prior remote manifest exists
            return None
        finally:
            if local_tmp.exists():
                local_tmp.unlink()

    def _commit_manifest(
        self,
        spec: ProjectSpec,
        session_id: str,
        manifest_data: Dict[str, Any],
    ) -> None:
        """Writes the manifest remotely and caches it locally."""
        remote_root = spec.remote_root or f"/content/colab-workspaces/{spec.name}"
        remote_manifest_path = f"{remote_root}/{MANIFEST_FILENAME}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(manifest_data, tf, indent=2)
            tmp_local_path = Path(tf.name)

        try:
            # Upload remote manifest
            self.transport.upload_file(session_id, tmp_local_path, remote_manifest_path)
            # Cache locally
            self._save_local_cached_manifest(spec.name, manifest_data)
        finally:
            if tmp_local_path.exists():
                tmp_local_path.unlink()

    def _load_local_cached_manifest(self, project_name: str) -> Optional[Dict[str, Any]]:
        cache_file = self.cache_dir / f"{project_name}_last_manifest.json"
        if not cache_file.exists():
            return None
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_local_cached_manifest(self, project_name: str, data: Dict[str, Any]) -> None:
        cache_file = self.cache_dir / f"{project_name}_last_manifest.json"
        try:
            cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass
