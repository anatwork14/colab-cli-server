"""Background asynchronous worker for reliable cloud artifact uploads.

Processes transfer queue with bounded concurrency, automatic restart recovery,
per-file retries with backoff, and strict isolation from training job execution status.
"""
from __future__ import annotations

import logging
import queue
import random
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ..errors import (
    ArtifactValidationError,
    DriveAuthenticationError,
    DriveIntegrityError,
    DriveQuotaExceededError,
)
from .base import ArtifactStore, ArtifactType, StoredArtifact
from .manifest import compute_file_sha256
from .transfer import TransferJournal, TransferRecord, TransferStatus

logger = logging.getLogger(__name__)


class ArtifactUploadWorker:
    """Asynchronous background worker processing artifact uploads to cloud storage."""

    def __init__(
        self,
        store: ArtifactStore,
        journal: TransferJournal,
        max_workers: int = 2,
        initial_retry_delay_sec: float = 0.2,
        on_success: Optional[Callable[[StoredArtifact], None]] = None,
        on_failure: Optional[Callable[[TransferRecord, Exception], None]] = None,
    ):
        self.store = store
        self.journal = journal
        self.max_workers = max(1, max_workers)
        self.initial_retry_delay_sec = initial_retry_delay_sec
        self.on_success = on_success
        self.on_failure = on_failure

        self._queue: queue.Queue[TransferRecord] = queue.Queue()
        self._threads: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self._active_transfers: Dict[str, TransferRecord] = {}
        self._scheduled_retries: List[TransferRecord] = []
        self._lock = threading.Lock()

        # Load pending or interrupted transfers from persistent journal
        self._recover_pending_transfers()

    def _recover_pending_transfers(self) -> None:
        """Scan persistent journal and re-enqueue any pending or in-flight transfers."""
        pending = self.journal.get_pending_transfers()
        for record in pending:
            # If it was interrupted while UPLOADING, transition back to RETRY_PENDING or UPLOAD_PENDING
            if record.status == TransferStatus.UPLOADING.value:
                try:
                    record.transition_to(TransferStatus.RETRY_PENDING, error_message="Worker restarted during upload")
                    self.journal.record_transfer(record)
                except Exception:
                    pass
            self._queue.put(record)

    def start(self) -> None:
        """Start worker background threads."""
        self._stop_event.clear()
        for i in range(self.max_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"artifact-upload-worker-{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def stop(self, wait: bool = True, timeout: Optional[float] = 5.0) -> None:
        """Signal worker threads to stop."""
        self._stop_event.set()
        if wait:
            for t in self._threads:
                t.join(timeout=timeout)
        self._threads.clear()

    def _check_scheduled_retries(self) -> None:
        """Promote due retries from scheduled list into the active queue."""
        now = time.time()
        with self._lock:
            due: List[TransferRecord] = []
            remaining: List[TransferRecord] = []
            for rec in self._scheduled_retries:
                if rec.next_retry_at is None or now >= rec.next_retry_at:
                    due.append(rec)
                else:
                    remaining.append(rec)
            self._scheduled_retries = remaining

        for rec in due:
            self._queue.put(rec)

    def enqueue(
        self,
        local_path: Union[str, Path],
        project_id: str,
        run_id: str,
        artifact_type: str = ArtifactType.CHECKPOINT.value,
        expected_sha256: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TransferRecord:
        """Enqueue an artifact for background upload."""
        src = Path(local_path).resolve()
        transfer_id = f"{self.store.provider_name}-{project_id}-{run_id}-{src.name}"

        record = self.journal.get_transfer(transfer_id)
        if not record:
            from .manifest import compute_file_md5, compute_file_sha256
            sha256 = expected_sha256 or compute_file_sha256(src)
            md5_hash = compute_file_md5(src)
            rel_path = f"checkpoints/{src.name}" if artifact_type == ArtifactType.CHECKPOINT.value else src.name

            record = TransferRecord(
                transfer_id=transfer_id,
                project_id=project_id,
                run_id=run_id,
                artifact_id=f"{project_id}/{run_id}/{rel_path}",
                local_path=str(src),
                target_rel_path=rel_path,
                size_bytes=src.stat().st_size,
                sha256=sha256,
                md5=md5_hash,
                status=TransferStatus.LOCAL_VERIFIED.value,
                metadata=metadata or {},
            )
            self.journal.record_transfer(record)

        if record.status in (
            TransferStatus.LOCAL_VERIFIED.value,
            TransferStatus.UPLOAD_PENDING.value,
            TransferStatus.RETRY_PENDING.value,
        ):
            with self._lock:
                if record.transfer_id not in self._active_transfers:
                    self._queue.put(record)

        return record

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            self._check_scheduled_retries()
            try:
                record = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with self._lock:
                if record.transfer_id in self._active_transfers:
                    self._queue.task_done()
                    continue
                self._active_transfers[record.transfer_id] = record

            try:
                # Revalidate local source file integrity before publication
                src_path = Path(record.local_path)
                if not src_path.is_file():
                    raise ArtifactValidationError(f"Local artifact source file missing: {record.local_path}")

                current_sha256 = compute_file_sha256(src_path)
                if record.sha256 and len(record.sha256) == 64 and current_sha256.lower() != record.sha256.lower():
                    raise ArtifactValidationError(
                        f"Local artifact modified before publication: expected {record.sha256}, got {current_sha256}"
                    )

                # Execute publication through store
                stored = self.store.publish(
                    source=record.local_path,
                    project_id=record.project_id,
                    run_id=record.run_id,
                    artifact_type=record.metadata.get("artifact_type", ArtifactType.CHECKPOINT.value),
                    metadata=record.metadata,
                    expected_sha256=record.sha256,
                )
                if self.on_success:
                    self.on_success(stored)
            except Exception as e:
                logger.warning(f"Artifact upload failed for {record.artifact_id}: {e}")
                # Refresh record state from journal
                refreshed = self.journal.get_transfer(record.transfer_id) or record
                refreshed.retry_count += 1
                refreshed.last_error = str(e)

                if refreshed.retry_count <= refreshed.max_retries:
                    backoff = self.initial_retry_delay_sec * (2 ** (refreshed.retry_count - 1)) + random.uniform(0.01, 0.05)
                    refreshed.next_retry_at = time.time() + backoff
                    refreshed.status = TransferStatus.RETRY_PENDING.value
                    self.journal.record_transfer(refreshed)
                    with self._lock:
                        self._scheduled_retries.append(refreshed)
                else:
                    refreshed.status = TransferStatus.UPLOAD_FAILED.value
                    self.journal.record_transfer(refreshed)

                if self.on_failure:
                    self.on_failure(refreshed, e)

            finally:
                with self._lock:
                    self._active_transfers.pop(record.transfer_id, None)
                self._queue.task_done()

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Block until all enqueued items and scheduled retries are processed."""
        start = time.time()
        while not self._queue.empty() or len(self._active_transfers) > 0 or len(self._scheduled_retries) > 0:
            if timeout and (time.time() - start) >= timeout:
                return False
            time.sleep(0.05)
        return True
