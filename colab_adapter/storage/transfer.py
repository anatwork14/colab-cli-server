"""Transfer lifecycle state machine and persistent journal for background cloud uploads.

Enforces state transitions:
  LOCAL_STAGED -> LOCAL_VERIFIED -> UPLOAD_PENDING -> UPLOADING -> DRIVE_VERIFIED -> PUBLISHED
Tracks resumable upload sessions, byte offsets, retries, and errors with atomic disk persistence.
"""
from __future__ import annotations

import json
import logging
import os
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
from typing import Any, Dict, List, Optional, Union

try:
    import fcntl
except ImportError:
    fcntl = None


from ..errors import InvalidStateTransitionError, JournalCorruptionError, JournalLockError
from .base import _utc_now_iso

logger = logging.getLogger(__name__)


class TransferStatus(str, Enum):
    LOCAL_STAGED = "LOCAL_STAGED"
    LOCAL_VERIFIED = "LOCAL_VERIFIED"
    UPLOAD_PENDING = "UPLOAD_PENDING"
    UPLOADING = "UPLOADING"
    DRIVE_VERIFIED = "DRIVE_VERIFIED"
    PUBLISHED = "PUBLISHED"
    UPLOAD_FAILED = "UPLOAD_FAILED"
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
        if error_message:
            self.last_error = error_message

    def to_dict(self, safe: bool = False) -> Dict[str, Any]:
        """Convert record to dict, optionally redacting sensitive session URLs."""
        d = asdict(self)
        if safe and d.get("resumable_session_url"):
            d["resumable_session_url"] = "[REDACTED_SESSION_URL]"
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransferRecord":
        valid = {
            "transfer_id", "project_id", "run_id", "artifact_id", "local_path",
            "target_rel_path", "size_bytes", "sha256", "status", "md5",
            "resumable_session_url", "bytes_transferred", "remote_file_id",
            "retry_count", "max_retries", "next_retry_at", "last_error",
            "created_at", "updated_at", "metadata"
        }
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


class TransferJournal:
    """Persistent, cross-process and thread-safe store for transfer records."""

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
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError) as jde:
            # Preserve corrupt copy
            corrupt_file = self.path.parent / f"{self.path.name}.corrupt.{int(time.time())}"
            try:
                shutil.copy2(str(self.path), str(corrupt_file))
            except Exception:
                pass

            # Attempt recovery from validated backup
            bak = self.path.with_suffix(".bak")
            if bak.exists():
                try:
                    with open(bak, "r", encoding="utf-8") as f_bak:
                        data = json.load(f_bak)
                        logger.warning(f"Journal corruption detected; recovered {len(data)} records from backup {bak}")
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
        """Fetch transfer record by ID under lock."""
        with self._acquire_locks():
            data = self._read_records()
            raw = data.get(transfer_id)
            if not raw:
                return None
            return TransferRecord.from_dict(raw)

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
            return results

