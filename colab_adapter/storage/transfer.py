"""Transfer lifecycle state machine and persistent journal for background cloud uploads.

Enforces state transitions:
  LOCAL_STAGED -> LOCAL_VERIFIED -> UPLOAD_PENDING -> UPLOADING -> DRIVE_VERIFIED -> PUBLISHED
Tracks resumable upload sessions, byte offsets, retries, and errors with atomic disk persistence.
"""
from __future__ import annotations

import contextlib
import errno
import json
import logging
import os
import re
import secrets
import shutil
import stat
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

try:
    import fcntl
except ImportError:
    fcntl = None


from ..errors import InvalidStateTransitionError, JournalCorruptionError, JournalLockError
from ..tracker import sanitize_data, sanitize_text
from .base import _utc_now_iso

logger = logging.getLogger(__name__)


class TransferStatus(str, Enum):
    LOCAL_STAGED = "LOCAL_STAGED"
    LOCAL_VERIFIED = "LOCAL_VERIFIED"
    UPLOAD_PENDING = "UPLOAD_PENDING"
    PENDING = "UPLOAD_PENDING"
    UPLOADING = "UPLOADING"
    DRIVE_VERIFIED = "DRIVE_VERIFIED"
    PUBLISHED = "PUBLISHED"
    UPLOAD_FAILED = "UPLOAD_FAILED"
    FAILED = "UPLOAD_FAILED"
    RETRY_PENDING = "RETRY_PENDING"
    CANCELLED = "CANCELLED"


# Allowed state transitions
VALID_TRANSFER_TRANSITIONS = {
    TransferStatus.LOCAL_STAGED: {TransferStatus.LOCAL_VERIFIED, TransferStatus.CANCELLED},
    TransferStatus.LOCAL_VERIFIED: {TransferStatus.UPLOAD_PENDING, TransferStatus.CANCELLED},
    TransferStatus.UPLOAD_PENDING: {TransferStatus.UPLOADING, TransferStatus.CANCELLED},
    TransferStatus.UPLOADING: {
        TransferStatus.DRIVE_VERIFIED,
        TransferStatus.UPLOAD_FAILED,
        TransferStatus.RETRY_PENDING,
        TransferStatus.CANCELLED,
    },
    TransferStatus.RETRY_PENDING: {TransferStatus.UPLOADING, TransferStatus.UPLOAD_FAILED, TransferStatus.CANCELLED},
    TransferStatus.UPLOAD_FAILED: {TransferStatus.RETRY_PENDING, TransferStatus.CANCELLED},
    TransferStatus.DRIVE_VERIFIED: {TransferStatus.PUBLISHED},
    TransferStatus.PUBLISHED: set(),
    TransferStatus.CANCELLED: set(),
}


@dataclass
class TransferRecord:
    """Record of an individual artifact cloud transfer."""
    transfer_id: str
    project_id: str
    run_id: str
    artifact_id: str
    local_path: str
    target_rel_path: str
    size_bytes: int
    sha256: str
    status: str = TransferStatus.LOCAL_STAGED.value
    md5: Optional[str] = None
    resumable_session_url: Optional[str] = None
    bytes_transferred: int = 0
    remote_file_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 5
    next_retry_at: Optional[float] = None
    last_error: Optional[str] = None
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)
    owner_pid: Optional[int] = None
    lease_token: Optional[str] = None
    lease_expires_at: Optional[float] = None
    fencing_epoch: int = 0
    expected_sha256: Optional[str] = None
    source_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.expected_sha256 is None and self.sha256:
            self.expected_sha256 = self.sha256
        if self.source_path is None and self.local_path:
            self.source_path = self.local_path

    def __str__(self) -> str:
        return self.transfer_id

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.transfer_id == other or self.artifact_id == other
        if not isinstance(other, TransferRecord):
            return False
        return (
            self.transfer_id == other.transfer_id
            and self.status == other.status
            and self.sha256 == other.sha256
        )

    def transition_to(self, new_status: TransferStatus, error_message: Optional[str] = None) -> None:
        """Enforce valid state transition."""
        current = TransferStatus(self.status)
        allowed = VALID_TRANSFER_TRANSITIONS.get(current, set())
        if new_status not in allowed:
            raise InvalidStateTransitionError(
                f"Illegal transfer state transition from {current.value} to {new_status.value}",
                details={"transfer_id": self.transfer_id, "from": current.value, "to": new_status.value},
            )
        self.status = new_status.value
        self.updated_at = _utc_now_iso()
        if error_message is not None:
            self.last_error = sanitize_text(str(error_message))

    def to_dict(self, safe: bool = False) -> Dict[str, Any]:
        """Convert record to dict, optionally redacting sensitive session URLs and errors."""
        d = asdict(self)
        if safe:
            if d.get("resumable_session_url"):
                d["resumable_session_url"] = "[REDACTED_SESSION_URL]"
            if d.get("last_error"):
                d["last_error"] = sanitize_data(d["last_error"])
            if d.get("metadata"):
                d["metadata"] = sanitize_data(d["metadata"])
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransferRecord":
        valid = {
            "transfer_id", "project_id", "run_id", "artifact_id", "local_path",
            "target_rel_path", "size_bytes", "sha256", "status", "md5",
            "resumable_session_url", "bytes_transferred", "remote_file_id",
            "retry_count", "max_retries", "next_retry_at", "last_error",
            "created_at", "updated_at", "metadata",
            "owner_pid", "lease_token", "lease_expires_at", "fencing_epoch",
            "expected_sha256", "source_path",
        }
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


def _is_pid_alive(pid: Optional[int]) -> bool:
    """Check if process with given PID is currently running."""
    if pid is None or pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except (ProcessLookupError, OSError):
        return False


class TransferJournal:
    """Persistent, cross-process and thread-safe store for transfer records."""

    _lock_registry: Dict[Tuple[str, str], threading.RLock] = {}
    _lock_registry_lock = threading.Lock()
    _local_state = threading.local()
    _is_pid_alive = staticmethod(_is_pid_alive)

    def __init__(self, journal_path: Union[str, Path]):
        self.path = Path(journal_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(str(self.path.parent), 0o700)
        except OSError:
            pass
        self.lock_path = self.path.with_suffix(".lock")
        self._thread_lock = threading.Lock()
        if not self.path.exists():
            with self._acquire_locks():
                if not self.path.exists():
                    self._write_records({})

    @contextmanager
    def _acquire_locks(self):
        """Cross-thread and cross-process file lock guard. Fails closed if locking fails."""
        if not self._thread_lock.acquire(timeout=30.0):
            raise JournalLockError("Thread lock acquisition timed out.")
        lock_fd = None
        try:
            if fcntl is not None:
                try:
                    lock_fd = os.open(str(self.lock_path), os.O_RDWR | os.O_CREAT, 0o600)
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                except (OSError, AttributeError) as e:
                    raise JournalLockError(f"Failed to acquire exclusive lock on {self.lock_path}: {e}") from e
            yield
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
                except OSError:
                    pass
            self._thread_lock.release()

    def _read_records(self) -> Dict[str, Dict[str, Any]]:
        """Read records from journal. Recovers from .bak or preserves corrupt file without silent wipe."""
        if not self.path.exists():
            return {}
        try:
            mode = stat.S_IMODE(self.path.stat().st_mode)
            if mode & 0o077:
                os.chmod(str(self.path), 0o600)
        except OSError:
            pass

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError) as jde:
            # Preserve corrupt copy with both .bak and standard timestamps
            ts = int(time.time() * 1000)
            corrupt_file_bak = self.path.parent / f"{self.path.name}.corrupt.{ts}.bak"
            corrupt_file = self.path.parent / f"{self.path.name}.corrupt.{ts}"
            try:
                shutil.copy2(str(self.path), str(corrupt_file_bak))
                shutil.copy2(str(self.path), str(corrupt_file))
                os.chmod(str(corrupt_file_bak), 0o600)
                os.chmod(str(corrupt_file), 0o600)
            except Exception:
                pass

            # Attempt recovery from validated backup
            bak = self.path.with_suffix(".bak")
            if bak.exists():
                try:
                    with open(bak, "r", encoding="utf-8") as f_bak:
                        data = json.load(f_bak)
                        logger.warning(f"Journal corruption detected; recovered {len(data)} records from backup {bak}")
                        try:
                            self._write_records(data)
                        except Exception as restore_err:
                            logger.error(f"Failed to restore recovered journal to {self.path}: {restore_err}")
                        return data
                except Exception:
                    pass

            raise JournalCorruptionError(
                f"Transfer journal at {self.path} is corrupted: {jde}. Corrupt copy preserved at {corrupt_file}."
            ) from jde

    def _write_records(self, records: Dict[str, Dict[str, Any]]) -> None:
        """Atomically persist records to journal file with 0600 mode and backup."""
        tmp = self.path.parent / f"{self.path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        payload = json.dumps(records, indent=2, sort_keys=True)

        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())

        os.replace(str(tmp), str(self.path))
        try:
            os.chmod(str(self.path), 0o600)
        except OSError:
            pass

        # Create/update backup file
        try:
            bak = self.path.with_suffix(".bak")
            shutil.copy2(str(self.path), str(bak))
            os.chmod(str(bak), 0o600)
        except OSError:
            pass

        # Fsync parent directory descriptor if supported
        try:
            dfd = os.open(str(self.path.parent), os.O_RDONLY)
            os.fsync(dfd)
            os.close(dfd)
        except OSError:
            pass

    def record_transfer(self, record: TransferRecord) -> None:
        """Create or update a transfer record under lock."""
        with self._acquire_locks():
            data = self._read_records()
            data[record.transfer_id] = record.to_dict()
            self._write_records(data)

    def get_transfer(self, transfer_id: str) -> Optional[TransferRecord]:
        """Fetch transfer record by ID under lock. Supports exact match and legacy ID prefix fallback."""
        with self._acquire_locks():
            data = self._read_records()
            raw = data.get(transfer_id)
            if raw:
                return TransferRecord.from_dict(raw)
            # Legacy fallback: match keys starting with transfer_id + "-"
            # e.g. "mock_cloud-p1-r1-secret_ckpt.pt" matches "mock_cloud-p1-r1-secret_ckpt.pt-a1b2c3d4e5f6"
            prefix = f"{transfer_id}-"
            matching = [v for k, v in data.items() if k.startswith(prefix)]
            if matching:
                matching.sort(key=lambda r: (r.get("updated_at", ""), r.get("created_at", "")))
                return TransferRecord.from_dict(matching[-1])
            return None

    def _load(self) -> Dict[str, Dict[str, Any]]:
        """Reload journal records from disk under lock."""
        with self._acquire_locks():
            return self._read_records()

    def mark_completed(self, transfer_id: str, remote_id: Optional[str] = None) -> Optional[TransferRecord]:
        """Mark a transfer as PUBLISHED and clear its lease under lock."""
        with self._acquire_locks():
            data = self._read_records()
            target_key = transfer_id
            raw = data.get(target_key)
            if not raw:
                prefix = f"{transfer_id}-"
                matching_keys = [k for k in data.keys() if k.startswith(prefix)]
                if matching_keys:
                    matching_keys.sort(key=lambda k: (data[k].get("updated_at", ""), data[k].get("created_at", "")))
                    target_key = matching_keys[-1]
                    raw = data[target_key]
            if not raw:
                return None
            rec = TransferRecord.from_dict(raw)
            rec.status = TransferStatus.PUBLISHED.value
            if remote_id:
                rec.remote_file_id = remote_id
            rec.lease_token = None
            rec.lease_expires_at = None
            rec.owner_pid = None
            rec.updated_at = _utc_now_iso()
            data[rec.transfer_id] = rec.to_dict()
            self._write_records(data)
            return rec

    def list_transfers_for_run(self, project_id: str, run_id: str) -> List[TransferRecord]:
        """Inspect all transfer records and revisions for a given project and run."""
        return self.list_transfers(project_id=project_id, run_id=run_id)

    def get_published_revision(
        self, project_id: str, run_id: str, artifact_id: str
    ) -> Optional[TransferRecord]:
        """Query a specific published revision for a run by full artifact_id (including @sha256_) or base artifact_id."""
        with self._acquire_locks():
            records = self._read_records()
            matches: List[TransferRecord] = []
            for raw in records.values():
                rec = TransferRecord.from_dict(raw)
                if rec.project_id != project_id or rec.run_id != run_id:
                    continue
                if rec.status not in (TransferStatus.PUBLISHED, TransferStatus.PUBLISHED.value):
                    continue
                if rec.artifact_id == artifact_id:
                    return rec
                base_id = rec.artifact_id.split("@sha256_")[0]
                if base_id == artifact_id or rec.target_rel_path == artifact_id:
                    matches.append(rec)
            if matches:
                matches.sort(key=lambda r: (r.updated_at, r.created_at))
                return matches[-1]
            return None

    def list_transfers(
        self,
        project_id: Optional[str] = None,
        run_id: Optional[str] = None,
        status: Optional[TransferStatus] = None,
    ) -> List[TransferRecord]:
        """List transfers matching optional filters under lock."""
        with self._acquire_locks():
            data = self._read_records()
            results = []
            for raw in data.values():
                rec = TransferRecord.from_dict(raw)
                if project_id and rec.project_id != project_id:
                    continue
                if run_id and rec.run_id != run_id:
                    continue
                if status and rec.status != status.value:
                    continue
                results.append(rec)
            return results

    def get_pending_transfers(self) -> List[TransferRecord]:
        """Get all transfers that need to be processed or resumed under lock."""
        with self._acquire_locks():
            data = self._read_records()
            results = []
            for raw in data.values():
                rec = TransferRecord.from_dict(raw)
                if rec.status in (
                    TransferStatus.LOCAL_VERIFIED.value,
                    TransferStatus.UPLOAD_PENDING.value,
                    TransferStatus.UPLOADING.value,
                    TransferStatus.RETRY_PENDING.value,
                ):
                    results.append(rec)
            results.sort(key=lambda r: (r.created_at, r.transfer_id))
            return results

    @contextlib.contextmanager
    def acquire_named_lock(self, lock_name: str, timeout: float = 30.0) -> Iterator[Path]:
        """Cross-process and re-entrant thread lock for a named resource."""
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(lock_name))
        locks_dir = self.path.parent / ".transfer_locks"
        locks_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(str(locks_dir), 0o700)
        except OSError:
            pass

        key = (str(locks_dir.resolve()), safe_id)
        lock_path = locks_dir / f"{safe_id}.lock"

        local_held = getattr(TransferJournal._local_state, "held_locks", None)
        if local_held is None:
            local_held = {}
            TransferJournal._local_state.held_locks = local_held

        if key in local_held and local_held[key]["depth"] > 0:
            local_held[key]["depth"] += 1
            try:
                yield lock_path
            finally:
                local_held[key]["depth"] -= 1
            return

        start_time = time.monotonic()
        with TransferJournal._lock_registry_lock:
            if key not in TransferJournal._lock_registry:
                TransferJournal._lock_registry[key] = threading.RLock()
            rlock = TransferJournal._lock_registry[key]

        remaining = timeout - (time.monotonic() - start_time)
        if not rlock.acquire(timeout=max(0.0, remaining)):
            raise TimeoutError(f"Timed out acquiring thread lock for '{lock_name}' after {timeout:.1f}s")

        fd = None
        flocked = False
        try:
            fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                os.chmod(str(lock_path), 0o600)
            except OSError:
                pass

            poll_interval = 0.05
            while True:
                if fcntl is None:
                    flocked = True
                    break
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    flocked = True
                    break
                except (BlockingIOError, OSError) as e:
                    if isinstance(e, BlockingIOError) or getattr(e, "errno", None) in (
                        errno.EAGAIN,
                        errno.EWOULDBLOCK,
                        errno.EACCES,
                    ):
                        elapsed = time.monotonic() - start_time
                        if elapsed >= timeout:
                            raise TimeoutError(
                                f"Timed out acquiring cross-process lock on {lock_path} for '{lock_name}' after {timeout:.1f}s"
                            ) from e
                        sleep_time = min(poll_interval, max(0.005, timeout - elapsed))
                        time.sleep(sleep_time)
                    else:
                        raise

            local_held[key] = {"depth": 1, "fd": fd}
            try:
                yield lock_path
            finally:
                local_held.pop(key, None)
        finally:
            if flocked and fcntl is not None and fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            rlock.release()

    @contextlib.contextmanager
    def acquire_artifact_lock(self, *args, timeout: float = 30.0) -> Iterator[Path]:
        """Cross-process and re-entrant exclusive lock for an individual artifact transfer."""
        if len(args) == 1:
            lock_id = args[0]
        else:
            lock_id = "-".join(str(a) for a in args)
        with self.acquire_named_lock(f"artifact-{lock_id}", timeout=timeout) as p:
            yield p

    def claim_transfer_lease(
        self,
        transfer_id: str,
        lease_duration_sec: float = 300.0,
        expected_sha256: Optional[str] = None,
    ) -> Tuple[bool, Optional[TransferRecord], str]:
        """Atomically claim durable lease on a transfer record with fencing."""
        with self._acquire_locks():
            records = self._read_records()
            raw = records.get(transfer_id)
            if not raw:
                prefix = f"{transfer_id}-"
                matching_keys = [k for k in records.keys() if k.startswith(prefix)]
                if matching_keys:
                    matching_keys.sort(key=lambda k: (records[k].get("updated_at", ""), records[k].get("created_at", "")))
                    transfer_id = matching_keys[-1]
                    raw = records[transfer_id]
            rec = TransferRecord.from_dict(raw) if raw else None

            # If already published with matching sha256, refuse claim
            if rec is not None and rec.status in (TransferStatus.PUBLISHED, TransferStatus.PUBLISHED.value) and rec.remote_file_id:
                if expected_sha256 is None or not rec.sha256 or rec.sha256.lower() == expected_sha256.lower():
                    return (False, rec, "")

            # Check if record exists and has an active lease held by another live process
            now = time.time()
            if rec is not None and rec.lease_token and rec.lease_expires_at and rec.lease_expires_at > now:
                if _is_pid_alive(rec.owner_pid) and rec.owner_pid != os.getpid():
                    return (False, rec, "")

            lease_token = secrets.token_hex(12)
            fencing_epoch = (rec.fencing_epoch if rec else 0) + 1

            if rec is not None:
                rec.owner_pid = os.getpid()
                rec.lease_token = lease_token
                rec.lease_expires_at = now + lease_duration_sec
                rec.fencing_epoch = fencing_epoch
                rec.updated_at = _utc_now_iso()
                records[rec.transfer_id] = rec.to_dict()
                self._write_records(records)
                return (True, rec, lease_token)
            else:
                return (True, None, lease_token)

    def release_transfer_lease(self, transfer_id: str, lease_token: Optional[str] = None) -> None:
        """Release active transfer lease under lock."""
        with self._acquire_locks():
            records = self._read_records()
            target_key = transfer_id
            raw = records.get(target_key)
            if not raw:
                prefix = f"{transfer_id}-"
                matching_keys = [k for k in records.keys() if k.startswith(prefix)]
                if matching_keys:
                    matching_keys.sort(key=lambda k: (records[k].get("updated_at", ""), records[k].get("created_at", "")))
                    target_key = matching_keys[-1]
                    raw = records[target_key]
            if not raw:
                return
            rec = TransferRecord.from_dict(raw)
            if lease_token is None or rec.lease_token == lease_token:
                rec.lease_token = None
                rec.lease_expires_at = None
                rec.owner_pid = None
                rec.updated_at = _utc_now_iso()
                records[rec.transfer_id] = rec.to_dict()
                self._write_records(records)

