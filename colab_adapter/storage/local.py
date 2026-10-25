"""Local filesystem artifact store implementation.

Layout per run (``<root>/projects/<project>/runs/<run>/``)::

    manifest.json (+ .bak)            authoritative publication manifest (head + history)
    .locks/manifest.lock              cross-thread/cross-process lock for the manifest
    checkpoints/<name>                mutable alias: copy of the manifest head only (cache)
    checkpoints/.revisions/<sha256>/<name>   immutable revision files, one dir per full sha
    checkpoints/.revisions/<sha16>/<name>    legacy revision dirs (read-only support)

Rules:
* The manifest is authoritative. Every publish is one locked read-modify-write: reload the
  manifest, install the immutable revision, add it, save the manifest, then reconcile the alias.
* Historical revisions are retrieved only by their exact full 64-hex sha256. A versioned id
  never falls back to the mutable alias.
* The alias is written only for the manifest head and is repaired from the head revision
  file under the lock when it disagrees (for example after a crash between manifest save
  and alias write).
* Read paths never create directories. Symlinked directories on the storage path are refused.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import shutil
import stat
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..errors import ArtifactNotFoundError, ArtifactValidationError, StorageError
from .base import (
    ArtifactStore,
    ArtifactType,
    StoredArtifact,
    _utc_now_iso,
    format_revision_artifact_id,
    parse_artifact_id,
)
from .locking import DEFAULT_LOCK_TIMEOUT_S, manifest_lock, manifest_lock_path
from .manifest import PublicationManifest, compute_file_md5, compute_file_sha256
from .validation import (
    validate_artifact_type,
    validate_filename,
    validate_logical_path,
    validate_project_id,
    validate_run_id,
)

MANIFEST_FILENAME = "manifest.json"
REVISIONS_DIR_NAME = ".revisions"
_FULL_SHA_RE = re.compile(r"[0-9a-f]{64}")
_SHA16_LEN = 16
_COPY_CHUNK = 1024 * 1024
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_RESERVED_RUN_DIR_NAMES = frozenset({MANIFEST_FILENAME, MANIFEST_FILENAME + ".bak"})


def _validate_safe_id(name: str, field: str = "identifier") -> str:
    from .validation import validate_identifier

    return validate_identifier(name, field)


# --------------------------------------------------------------------------- #
# Filesystem helpers (no directory creation unless stated)
# --------------------------------------------------------------------------- #

def _lstat_or_none(path: Path) -> Optional[os.stat_result]:
    try:
        return os.lstat(str(path))
    except FileNotFoundError:
        return None
    except NotADirectoryError:
        return None


def _check_real_dir(path: Path, create: bool = False) -> bool:
    """Return True if ``path`` is a real directory. Refuse symlinks and non-directories.

    Returns False when the directory is absent and ``create`` is False. When ``create`` is
    True the directory is created with mode 0700 if missing.
    """
    st = _lstat_or_none(path)
    if st is None:
        if not create:
            return False
        try:
            os.mkdir(str(path), 0o700)
        except FileExistsError:
            pass
        st = os.lstat(str(path))
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise ArtifactValidationError(
            f"Refusing symlinked or non-directory storage path: {path}",
            details={"path": str(path)},
        )
    return True


def _open_regular_nofollow(path: Path) -> int:
    """Open ``path`` read-only without following symlinks; the result must be a regular file."""
    try:
        fd = os.open(str(path), os.O_RDONLY | _O_NOFOLLOW)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ArtifactValidationError(
            f"Stored artifact path is a symlink or unreadable: {path}",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ArtifactValidationError(
                f"Stored artifact path is not a regular file: {path}",
                details={"path": str(path)},
            )
    except BaseException:
        os.close(fd)
        raise
    return fd


def _sha256_of_regular(path: Path) -> Optional[str]:
    """Full SHA-256 of a regular file (no symlink following). None if the file is absent."""
    try:
        fd = _open_regular_nofollow(path)
    except FileNotFoundError:
        return None
    hasher = hashlib.sha256()
    with os.fdopen(fd, "rb") as fh:
        for chunk in iter(lambda: fh.read(_COPY_CHUNK), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _copy_to_private_tmp(src: Path, tmp: Path, expected_sha: str) -> None:
    """Copy ``src`` to a new 0600 file ``tmp`` (O_EXCL, no follow), fsync, verify the hash."""
    fd_out = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW, 0o600)
    with os.fdopen(fd_out, "wb") as f_out:
        with open(str(src), "rb") as f_in:
            shutil.copyfileobj(f_in, f_out, _COPY_CHUNK)
        f_out.flush()
        os.fsync(f_out.fileno())
    try:
        os.chmod(str(tmp), 0o600)
    except OSError:
        pass
    actual = _sha256_of_regular(tmp)
    if actual != expected_sha:
        raise ArtifactValidationError(
            f"Integrity check failed during staging: expected {expected_sha}, got {actual}",
            details={"expected": expected_sha, "actual": actual},
        )


def _accept_identical_or_refuse(final: Path, sha: str) -> None:
    actual = _sha256_of_regular(final)
    if actual != sha:
        raise ArtifactValidationError(
            f"Refusing to overwrite immutable revision with different content: {final}",
            details={"path": str(final), "expected": sha, "actual": actual},
        )


def _normalize_logical_path(project_id: str, run_id: str, logical_path: str) -> str:
    lp = logical_path.split("@sha256_")[0].lstrip("/")
    prefix = f"{project_id}/{run_id}/"
    if lp.startswith(prefix):
        lp = lp[len(prefix):].lstrip("/")
    return validate_logical_path(lp)


def _head_sha(entry: Dict[str, Any], rev_id: str) -> str:
    sha = str(entry.get("sha256") or "").strip().lower()
    if not _FULL_SHA_RE.fullmatch(sha):
        raise StorageError(
            f"Manifest entry {rev_id} has no valid sha256; manifest is inconsistent.",
            details={"artifact_id": rev_id},
        )
    return sha


class LocalArtifactStore:
    """ArtifactStore implementation storing files on the local filesystem."""

    def __init__(self, root_dir: Union[str, Path] = "./artifacts", lock_timeout: float = DEFAULT_LOCK_TIMEOUT_S):
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.lock_timeout = float(lock_timeout)

    @property
    def provider_name(self) -> str:
        return "local"

    # ----------------------------------------------------------------- paths

    def _run_dir(self, project_id: str, run_id: str, create: bool = False) -> Path:
        validate_project_id(project_id)
        validate_run_id(run_id)
        run_dir = (self.root_dir / "projects" / project_id / "runs" / run_id).resolve()
        try:
            run_dir.relative_to(self.root_dir)
        except ValueError:
            raise ArtifactValidationError("Path traversal escape detected.")
        if create:
            run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def _manifest_path(run_dir: Path) -> Path:
        return run_dir / MANIFEST_FILENAME

    @staticmethod
    def _manifest_present(run_dir: Path) -> bool:
        m_path = run_dir / MANIFEST_FILENAME
        return os.path.lexists(str(m_path)) or os.path.lexists(str(m_path) + ".bak")

    @staticmethod
    def _has_revisions(run_dir: Path) -> bool:
        for d in (run_dir, run_dir / "checkpoints", run_dir / "outputs"):
            if os.path.lexists(str(d / REVISIONS_DIR_NAME)):
                return True
        return False

    @staticmethod
    def _dest_rel_dir(artifact_type: str) -> str:
        if artifact_type == ArtifactType.CHECKPOINT.value:
            return "checkpoints"
        if artifact_type in (
            ArtifactType.SOURCE_SNAPSHOT.value,
            ArtifactType.LOG.value,
            ArtifactType.METRICS.value,
        ):
            return ""
        return "outputs"

    # -------------------------------------------------------------- manifest

    def _read_manifest(self, run_dir: Path, project_id: str, run_id: str) -> PublicationManifest:
        """Load the manifest without creating anything. Fails closed if it was lost."""
        if self._manifest_present(run_dir):
            return PublicationManifest.load(self._manifest_path(run_dir))
        if self._has_revisions(run_dir):
            raise StorageError(
                "Publication manifest is missing while immutable revisions exist; "
                "refusing to serve artifacts without the authoritative manifest.",
                details={"project_id": project_id, "run_id": run_id},
            )
        return PublicationManifest(project_id=project_id, run_id=run_id)

    def _load_manifest(self, project_id: str, run_id: str) -> PublicationManifest:
        run_dir = self._run_dir(project_id, run_id, create=False)
        return self._read_manifest(run_dir, project_id, run_id)

    def _load_manifest_for_write(self, run_dir: Path, project_id: str, run_id: str) -> PublicationManifest:
        """Caller holds the manifest lock. Bootstraps an empty manifest for a brand-new run."""
        if self._manifest_present(run_dir):
            return PublicationManifest.load(self._manifest_path(run_dir))
        if self._has_revisions(run_dir):
            raise StorageError(
                "Publication manifest is missing while immutable revisions exist; refusing to publish.",
                details={"project_id": project_id, "run_id": run_id},
            )
        manifest = PublicationManifest(project_id=project_id, run_id=run_id)
        manifest.save(self._manifest_path(run_dir))
        return manifest

    # ------------------------------------------------------- revision files

    def _find_revision_file(self, dest_dir: Path, filename: str, sha: str, verify: bool) -> Optional[Path]:
        """Locate the immutable file for ``sha`` in ``dest_dir``.

        Looks in ``.revisions/<full sha>/`` first and then the legacy ``.revisions/<sha16>/``.
        With ``verify`` the full SHA-256 of the file must equal ``sha``. A mismatch in the
        full-sha directory is corruption and raises; a mismatch in the legacy prefix directory
        is a different revision sharing the prefix and is skipped.
        """
        if not _check_real_dir(dest_dir, create=False):
            return None
        revs = dest_dir / REVISIONS_DIR_NAME
        if not _check_real_dir(revs, create=False):
            return None
        for sub in (sha, sha[:_SHA16_LEN]):
            rev_dir = revs / sub
            if not _check_real_dir(rev_dir, create=False):
                continue
            candidate = rev_dir / filename
            st = _lstat_or_none(candidate)
            if st is None:
                continue
            if not verify:
                if not stat.S_ISREG(st.st_mode):
                    raise ArtifactValidationError(
                        f"Immutable revision path is not a regular file: {candidate}",
                        details={"path": str(candidate)},
                    )
                return candidate
            actual = _sha256_of_regular(candidate)
            if actual == sha:
                return candidate
            if sub == sha:
                raise ArtifactValidationError(
                    f"Immutable revision file is corrupted: expected {sha}, got {actual}",
                    details={"path": str(candidate), "expected": sha, "actual": actual},
                )
        return None

    def _install_revision(self, dest_dir: Path, filename: str, src: Path, sha: str) -> Path:
        """Write the immutable revision (caller holds the lock). Never overwrites different content."""
        existing = self._find_revision_file(dest_dir, filename, sha, verify=True)
        if existing is not None:
            return existing

        revs = dest_dir / REVISIONS_DIR_NAME
        _check_real_dir(revs, create=True)
        rev_dir = revs / sha
        _check_real_dir(rev_dir, create=True)
        try:
            os.chmod(str(revs), 0o700)
            os.chmod(str(rev_dir), 0o700)
        except OSError:
            pass
        final = rev_dir / filename
        if _lstat_or_none(final) is not None:
            raise ArtifactValidationError(
                f"Refusing to overwrite existing immutable revision file with different content: {final}",
                details={"path": str(final)},
            )

        tmp = rev_dir / f".rev.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            _copy_to_private_tmp(src, tmp, sha)
            try:
                os.link(str(tmp), str(final))
            except FileExistsError:
                # Created concurrently by a writer outside this lock; accept only identical content.
                _accept_identical_or_refuse(final, sha)
            except OSError:
                # Hard links unsupported on this filesystem: fall back to an exclusive check + rename.
                if _lstat_or_none(final) is not None:
                    _accept_identical_or_refuse(final, sha)
                else:
                    os.replace(str(tmp), str(final))
        finally:
            try:
                os.unlink(str(tmp))
            except OSError:
                pass
        _fsync_dir(rev_dir)
        return final

    # ------------------------------------------------------------ alias

    def _alias_matches(self, alias: Path, sha: str) -> bool:
        try:
            return _sha256_of_regular(alias) == sha
        except ArtifactValidationError:
            return False

    def _write_alias(self, alias: Path, rev_path: Path, sha: str) -> None:
        """Replace the mutable alias with a verified copy of the head revision (caller holds lock)."""
        parent = alias.parent
        _check_real_dir(parent, create=True)
        tmp = parent / f".alias.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            _copy_to_private_tmp(rev_path, tmp, sha)
            os.replace(str(tmp), str(alias))
        finally:
            try:
                os.unlink(str(tmp))
            except OSError:
                pass

    def _head_record(self, manifest: PublicationManifest, logical_path: str) -> Optional[Tuple[str, str]]:
        rev_id = manifest.latest_by_path.get(logical_path)
        if rev_id is None:
            return None
        entry = manifest.artifacts.get(rev_id)
        if entry is None:
            raise StorageError(
                f"Manifest head {rev_id} has no artifact entry; manifest is inconsistent.",
                details={"artifact_id": rev_id},
            )
        return rev_id, _head_sha(entry, rev_id)

    def _sync_alias_locked(self, run_dir: Path, manifest: PublicationManifest, logical_path: str) -> None:
        """Caller holds the lock. Make the alias match the manifest head, failing closed otherwise."""
        head = self._head_record(manifest, logical_path)
        if head is None:
            return
        _, sha = head
        lp_path = Path(logical_path)
        dest_dir = run_dir / lp_path.parent
        rev_path = self._find_revision_file(dest_dir, lp_path.name, sha, verify=True)
        if rev_path is None:
            raise ArtifactNotFoundError(
                f"Head revision file for {logical_path} is missing; cannot reconcile alias.",
                details={"logical_path": logical_path, "sha256": sha},
            )
        alias = run_dir / logical_path
        if not self._alias_matches(alias, sha):
            self._write_alias(alias, rev_path, sha)

    # ------------------------------------------------------------ publish

    def publish(
        self,
        source: Union[str, Path],
        project_id: str,
        run_id: str,
        artifact_type: str = ArtifactType.CHECKPOINT.value,
        metadata: Optional[Dict[str, Any]] = None,
        expected_sha256: Optional[str] = None,
    ) -> StoredArtifact:
        """Publish a local file into the store idempotently (one locked read-modify-write)."""
        # Validate every untrusted input before any filesystem side effect.
        validate_project_id(project_id)
        validate_run_id(run_id)
        validate_artifact_type(artifact_type)

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
        target_filename = validate_filename(src.name)

        dest_rel_dir = self._dest_rel_dir(artifact_type)
        if not dest_rel_dir and target_filename in _RESERVED_RUN_DIR_NAMES:
            raise ArtifactValidationError(
                f"File name {target_filename!r} is reserved for store metadata.",
                details={"file": target_filename},
            )

        src_sha256 = compute_file_sha256(src)
        if expected_sha256 and expected_sha256.strip().lower() != src_sha256.lower():
            raise ArtifactValidationError(
                f"Checksum mismatch for published file {src.name}: expected {expected_sha256}, got {src_sha256}",
                details={"file": src.name, "expected": expected_sha256, "actual": src_sha256},
            )
        size_bytes = src.stat().st_size
        md5_hash = compute_file_md5(src)

        run_dir = self._run_dir(project_id, run_id, create=True)
        rel_path = f"{dest_rel_dir}/{target_filename}" if dest_rel_dir else target_filename
        revision_artifact_id = format_revision_artifact_id(project_id, run_id, rel_path, src_sha256)
        dest_dir = run_dir / dest_rel_dir if dest_rel_dir else run_dir

        with manifest_lock(self._manifest_path(run_dir), timeout=self.lock_timeout):
            manifest = self._load_manifest_for_write(run_dir, project_id, run_id)
            _check_real_dir(dest_dir, create=True)

            head_rev = manifest.latest_by_path.get(rel_path)
            existing_entry = manifest.artifacts.get(revision_artifact_id)

            if existing_entry is not None and revision_artifact_id != head_rev:
                # Historical replay: verify the immutable file, leave manifest and alias untouched.
                rev_path = self._find_revision_file(dest_dir, target_filename, src_sha256, verify=True)
                if rev_path is None:
                    raise ArtifactNotFoundError(
                        f"Immutable revision file for {revision_artifact_id} is missing.",
                        details={"artifact_id": revision_artifact_id},
                    )
                stored = dataclasses.replace(
                    StoredArtifact.from_dict(existing_entry), path=str(rev_path.resolve()), is_latest=False
                )
                return stored

            if existing_entry is not None:
                # Repeating the head: idempotent; re-verify and repair the alias only.
                rev_path = self._find_revision_file(dest_dir, target_filename, src_sha256, verify=True)
                if rev_path is None:
                    raise ArtifactNotFoundError(
                        f"Head revision file for {revision_artifact_id} is missing.",
                        details={"artifact_id": revision_artifact_id},
                    )
                self._sync_alias_locked(run_dir, manifest, rel_path)
                return dataclasses.replace(
                    StoredArtifact.from_dict(existing_entry), path=str(rev_path.resolve()), is_latest=True
                )

            rev_path = self._install_revision(dest_dir, target_filename, src, src_sha256)

            stored = StoredArtifact(
                artifact_id=revision_artifact_id,
                storage_provider=self.provider_name,
                path=str(rev_path.resolve()),
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
            manifest.save(self._manifest_path(run_dir))

            is_head = manifest.latest_by_path.get(rel_path) == revision_artifact_id
            if is_head:
                self._sync_alias_locked(run_dir, manifest, rel_path)
            return dataclasses.replace(stored, is_latest=is_head)

    # ----------------------------------------------------------- retrieve

    def retrieve(
        self,
        artifact_id: str,
        destination: Union[str, Path],
        expected_sha256: Optional[str] = None,
    ) -> Path:
        """Retrieve an artifact by ID to destination path. A failure leaves the destination untouched."""
        meta = self.get_metadata(artifact_id)
        if not meta or not meta.path or not Path(meta.path).exists():
            raise ArtifactNotFoundError(
                f"Artifact '{artifact_id}' not found in local store.",
                details={"artifact_id": artifact_id},
            )
        _, _, _, requested_sha = parse_artifact_id(artifact_id)
        if requested_sha is not None and meta.sha256 and requested_sha != meta.sha256.strip().lower():
            raise ArtifactValidationError(
                "Retrieved artifact sha256 does not match the requested revision.",
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
            fd_src = _open_regular_nofollow(stored_path)
            try:
                fd_part = os.open(str(part_file), os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW, 0o600)
            except BaseException:
                os.close(fd_src)
                raise
            with os.fdopen(fd_part, "wb") as f_out, os.fdopen(fd_src, "rb") as f_in:
                shutil.copyfileobj(f_in, f_out, _COPY_CHUNK)
                f_out.flush()
                os.fsync(f_out.fileno())
            try:
                os.chmod(part_file, 0o600)
            except OSError:
                pass

            actual_size = part_file.stat().st_size
            if meta.size_bytes is not None and actual_size != meta.size_bytes:
                raise ArtifactValidationError(
                    f"Retrieved artifact size mismatch: recorded {meta.size_bytes}, actual {actual_size}",
                    details={
                        "artifact_id": artifact_id,
                        "recorded_size": meta.size_bytes,
                        "actual_size": actual_size,
                        "expected_size": meta.size_bytes,
                    },
                )

            target_sha = expected_sha256 or meta.sha256
            retrieved_hash = compute_file_sha256(part_file)
            if target_sha and retrieved_hash.lower() != target_sha.lower():
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

    # ----------------------------------------------------------- metadata

    def get_metadata(self, artifact_id: str) -> Optional[StoredArtifact]:
        project_id, run_id, logical_path, sha = parse_artifact_id(artifact_id)
        if not project_id or not run_id:
            return None
        validate_project_id(project_id)
        validate_run_id(run_id)
        if not logical_path:
            return None
        lp = _normalize_logical_path(project_id, run_id, logical_path)
        run_dir = self._run_dir(project_id, run_id, create=False)

        if sha is not None:
            if not _FULL_SHA_RE.fullmatch(sha):
                raise ArtifactValidationError(
                    "Versioned artifact id must carry a full 64-character lowercase hex sha256.",
                    details={"artifact_id": artifact_id},
                )
            return self._get_exact(run_dir, project_id, run_id, lp, sha)
        return self._get_head(run_dir, project_id, run_id, lp)

    def _get_exact(self, run_dir: Path, project_id: str, run_id: str, lp: str, sha: str) -> Optional[StoredArtifact]:
        """Exact historical lookup. Never falls back to the mutable alias."""
        manifest = self._read_manifest(run_dir, project_id, run_id)
        rev_id = format_revision_artifact_id(project_id, run_id, lp, sha)
        lp_path = Path(lp)
        dest_dir = run_dir / lp_path.parent
        is_head = manifest.latest_by_path.get(lp) == rev_id

        entry = manifest.artifacts.get(rev_id)
        if entry is not None:
            located = self._find_revision_file(dest_dir, lp_path.name, sha, verify=False)
            if located is None:
                return None
            return dataclasses.replace(
                StoredArtifact.from_dict(entry), path=str(located.resolve()), is_latest=is_head
            )

        # Manifest has no entry: the immutable file may still exist. Accept it only if the full
        # SHA-256 matches the requested revision.
        located = self._find_revision_file(dest_dir, lp_path.name, sha, verify=True)
        if located is None:
            return None
        return StoredArtifact(
            artifact_id=rev_id,
            storage_provider=self.provider_name,
            path=str(located.resolve()),
            size_bytes=located.stat().st_size,
            sha256=sha,
            project_id=project_id,
            run_id=run_id,
            logical_path=lp,
            revision_id=f"sha256_{sha}",
            is_latest=is_head,
            metadata={"recovered_from": "revisions_dir"},
        )

    def _get_head(self, run_dir: Path, project_id: str, run_id: str, lp: str) -> Optional[StoredArtifact]:
        """Unversioned lookup: the manifest head, verified against its immutable file."""
        if not self._manifest_present(run_dir) and not self._has_revisions(run_dir):
            return self._legacy_lookup(run_dir, project_id, run_id, lp)

        manifest = self._read_manifest(run_dir, project_id, run_id)
        head = self._head_record(manifest, lp)
        if head is None:
            return None
        rev_id, sha = head
        lp_path = Path(lp)
        dest_dir = run_dir / lp_path.parent
        located = self._find_revision_file(dest_dir, lp_path.name, sha, verify=True)
        if located is None:
            entry = manifest.artifacts.get(rev_id) or {}
            if entry.get("storage_provider") not in (None, self.provider_name) and entry.get("remote_file_id"):
                # Head was published by another provider (e.g. Drive) into the shared manifest: there are
                # no local bytes, so return the manifest metadata only. The mutable alias is not touched.
                return dataclasses.replace(StoredArtifact.from_dict(entry), is_latest=True)
            raise ArtifactNotFoundError(
                f"Head revision file for {project_id}/{run_id}/{lp} is missing.",
                details={"artifact_id": rev_id},
            )

        if not self._alias_matches(run_dir / lp, sha):
            with manifest_lock(self._manifest_path(run_dir), timeout=self.lock_timeout):
                fresh = self._read_manifest(run_dir, project_id, run_id)
                self._sync_alias_locked(run_dir, fresh, lp)

        entry = manifest.artifacts[rev_id]
        return dataclasses.replace(StoredArtifact.from_dict(entry), path=str(located.resolve()), is_latest=True)

    def _legacy_lookup(self, run_dir: Path, project_id: str, run_id: str, lp: str) -> Optional[StoredArtifact]:
        """Pre-manifest layout only (no manifest and no .revisions): serve the plain file."""
        direct = run_dir / lp
        try:
            sha = _sha256_of_regular(direct)
        except ArtifactValidationError:
            return None
        if sha is None:
            return None
        return StoredArtifact(
            artifact_id=format_revision_artifact_id(project_id, run_id, lp, sha),
            storage_provider=self.provider_name,
            path=str(direct.resolve()),
            size_bytes=direct.stat().st_size,
            sha256=sha,
            project_id=project_id,
            run_id=run_id,
            logical_path=lp,
            revision_id=f"sha256_{sha}",
            is_latest=True,
        )

    def list_artifacts(
        self,
        project_id: str,
        run_id: Optional[str] = None,
        include_history: bool = True,
    ) -> List[StoredArtifact]:
        validate_project_id(project_id)
        if run_id:
            run_dir = self._run_dir(project_id, run_id, create=False)
            return self._read_manifest(run_dir, project_id, run_id).list_artifacts(include_history=include_history)

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
