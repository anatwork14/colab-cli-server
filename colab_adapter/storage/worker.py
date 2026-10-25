"""Background asynchronous worker for reliable cloud artifact uploads.

Processes transfer queue with bounded concurrency, automatic restart recovery,
per-file retries with backoff, and strict isolation from training job execution status.
"""
from __future__ import annotations

import logging
import os
import queue
import random
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from ..errors import (
    ArtifactValidationError,
    DriveAuthenticationError,
    DriveIntegrityError,
    DriveQuotaExceededError,
)
from ..tracker import sanitize_text
from .base import ArtifactStore, ArtifactType, StoredArtifact
from .manifest import compute_file_md5, compute_file_sha256
from .transfer import TransferJournal, TransferRecord, TransferStatus, _is_pid_alive

logger = logging.getLogger(__name__)

UploadTask = TransferRecord


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
        self._queued_ids: Set[str] = set()
        self._threads: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self._active_transfers: Dict[str, TransferRecord] = {}
        self._scheduled_retries: List[TransferRecord] = []
        self._lock = threading.Lock()

        # Load pending or interrupted transfers from persistent journal
        self.recover_from_journal()

    def _is_enqueued_or_active(self, transfer_id: str) -> bool:
        """Check whether transfer is queued, active, or in scheduled retries. Must hold self._lock."""
        return (
            transfer_id in self._queued_ids
            or transfer_id in self._active_transfers
            or any(r.transfer_id == transfer_id for r in self._scheduled_retries)
        )

    def recover_from_journal(self) -> None:
        """Scan persistent journal and recover pending transfers with lease fencing."""
        pending = self.journal.get_pending_transfers()
        now = time.time()
        with self._lock:
            for record in pending:
                # Skip records whose lease is actively held by another live PID
                is_held_by_other_live_pid = bool(
                    record.lease_token
                    and record.lease_expires_at
                    and record.lease_expires_at > now
                    and _is_pid_alive(record.owner_pid)
                    and record.owner_pid != os.getpid()
                )
                if is_held_by_other_live_pid:
                    continue

                # Immediately recover records whose owner_pid is dead or lease has expired
                if record.status == TransferStatus.UPLOADING.value:
                    try:
                        record.transition_to(TransferStatus.RETRY_PENDING, error_message="Worker restarted during upload")
                        record.lease_token = None
                        record.lease_expires_at = None
                        record.owner_pid = None
                        self.journal.record_transfer(record)
                    except Exception:
                        pass
                if not self._is_enqueued_or_active(record.transfer_id):
                    self._queued_ids.add(record.transfer_id)
                    self._queue.put(record)

    def _recover_pending_transfers(self) -> None:
        """Alias for recover_from_journal."""
        self.recover_from_journal()

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
                if not self._is_enqueued_or_active(rec.transfer_id):
                    self._queued_ids.add(rec.transfer_id)
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
        current_sha256 = expected_sha256 or compute_file_sha256(src)
        current_size = src.stat().st_size
        transfer_id = f"{self.store.provider_name}-{project_id}-{run_id}-{src.name}"

        record = self.journal.get_transfer(transfer_id)
        if not record:
            md5_hash = compute_file_md5(src)
            rel_path = f"checkpoints/{src.name}" if artifact_type == ArtifactType.CHECKPOINT.value else src.name

            record = TransferRecord(
                transfer_id=transfer_id,
                project_id=project_id,
                run_id=run_id,
                artifact_id=f"{project_id}/{run_id}/{rel_path}",
                local_path=str(src),
                target_rel_path=rel_path,
                size_bytes=current_size,
                sha256=current_sha256,
                md5=md5_hash,
                status=TransferStatus.LOCAL_VERIFIED.value,
                metadata=metadata or {},
            )
            self.journal.record_transfer(record)
        else:
            is_sha_changed = bool(
                record.sha256
                and len(record.sha256) == 64
                and record.sha256.lower() != current_sha256.lower()
            )
            is_size_changed = bool(record.size_bytes != current_size)
            if is_sha_changed or is_size_changed:
                record.sha256 = current_sha256
                record.md5 = compute_file_md5(src)
                record.size_bytes = current_size
                record.bytes_transferred = 0
                record.resumable_session_url = None
                record.retry_count = 0
                record.status = TransferStatus.LOCAL_VERIFIED.value
                if metadata:
                    record.metadata.update(metadata)
                self.journal.record_transfer(record)

        if record.status in (
            TransferStatus.LOCAL_VERIFIED.value,
            TransferStatus.UPLOAD_PENDING.value,
            TransferStatus.RETRY_PENDING.value,
        ):
            with self._lock:
                if not self._is_enqueued_or_active(record.transfer_id):
                    self._queued_ids.add(record.transfer_id)
                    self._queue.put(record)

        return record

    def _process_task(self, task: UploadTask) -> bool:
        """Process an individual upload task with cross-process artifact locking and lease fencing."""
        src_path = Path(task.local_path)
        local_sha256: Optional[str] = None
        if src_path.is_file():
            local_sha256 = compute_file_sha256(src_path)
            if task.sha256 and len(task.sha256) == 64 and local_sha256.lower() != task.sha256.lower():
                raise ArtifactValidationError(
                    f"Local artifact modified before publication: expected {task.sha256}, got {local_sha256}"
                )

        with self.journal.acquire_artifact_lock(task.transfer_id, timeout=60.0):
            existing = self.journal.get_transfer(task.transfer_id)
            if (
                existing is not None
                and existing.status in (TransferStatus.PUBLISHED, TransferStatus.PUBLISHED.value)
                and existing.remote_file_id
                and (not local_sha256 or not existing.sha256 or existing.sha256.lower() == local_sha256.lower())
            ):
                logger.info(
                    "Artifact %s was already published by another worker/process; skipping upload.",
                    task.transfer_id,
                )
                if self.on_success:
                    stored = StoredArtifact(
                        artifact_id=existing.artifact_id,
                        storage_provider=self.store.provider_name,
                        path=existing.local_path,
                        size_bytes=existing.size_bytes,
                        sha256=existing.sha256,
                        project_id=existing.project_id,
                        run_id=existing.run_id,
                        artifact_type=existing.metadata.get("artifact_type", ArtifactType.CHECKPOINT.value) if existing.metadata else ArtifactType.CHECKPOINT.value,
                        remote_file_id=existing.remote_file_id,
                    )
                    self.on_success(stored)
                return True

            claimed, existing_rec, lease_token = self.journal.claim_transfer_lease(
                task.transfer_id,
                expected_sha256=local_sha256,
            )
            if not claimed:
                if existing_rec and existing_rec.status in (TransferStatus.PUBLISHED, TransferStatus.PUBLISHED.value):
                    logger.info(
                        "Artifact %s was already published by another worker/process; skipping upload.",
                        task.transfer_id,
                    )
                    if self.on_success:
                        stored = StoredArtifact(
                            artifact_id=existing_rec.artifact_id,
                            storage_provider=self.store.provider_name,
                            path=existing_rec.local_path,
                            size_bytes=existing_rec.size_bytes,
                            sha256=existing_rec.sha256,
                            project_id=existing_rec.project_id,
                            run_id=existing_rec.run_id,
                            artifact_type=existing_rec.metadata.get("artifact_type", ArtifactType.CHECKPOINT.value) if existing_rec.metadata else ArtifactType.CHECKPOINT.value,
                            remote_file_id=existing_rec.remote_file_id,
                        )
                        self.on_success(stored)
                    return True
                logger.info(
                    "Transfer %s lease held by another live process (pid %s)",
                    task.transfer_id,
                    getattr(existing_rec, "owner_pid", None),
                )
                return False

            if not src_path.is_file():
                raise ArtifactValidationError(f"Local artifact source file missing: {task.local_path}")

            try:
                stored = self.store.publish(
                    source=task.local_path,
                    project_id=task.project_id,
                    run_id=task.run_id,
                    artifact_type=task.metadata.get("artifact_type", ArtifactType.CHECKPOINT.value) if task.metadata else ArtifactType.CHECKPOINT.value,
                    metadata=task.metadata,
                    expected_sha256=local_sha256 or task.sha256,
                )

                # Ensure transfer record is transitioned to PUBLISHED if store did not do so
                refreshed = self.journal.get_transfer(task.transfer_id)
                if refreshed and refreshed.status != TransferStatus.PUBLISHED.value:
                    if refreshed.status == TransferStatus.LOCAL_VERIFIED.value:
                        refreshed.transition_to(TransferStatus.UPLOAD_PENDING)
                    if refreshed.status == TransferStatus.UPLOAD_PENDING.value:
                        refreshed.transition_to(TransferStatus.UPLOADING)
                    if refreshed.status == TransferStatus.UPLOADING.value:
                        refreshed.transition_to(TransferStatus.DRIVE_VERIFIED)
                    if refreshed.status == TransferStatus.DRIVE_VERIFIED.value:
                        refreshed.transition_to(TransferStatus.PUBLISHED)
                    if stored.remote_file_id:
                        refreshed.remote_file_id = stored.remote_file_id
                    self.journal.record_transfer(refreshed)

                if self.on_success:
                    self.on_success(stored)
                return True
            finally:
                self.journal.release_transfer_lease(task.transfer_id, lease_token)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            self._check_scheduled_retries()
            try:
                record = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with self._lock:
                self._queued_ids.discard(record.transfer_id)
                if record.transfer_id in self._active_transfers:
                    self._queue.task_done()
                    continue
                self._active_transfers[record.transfer_id] = record

            try:
                success = self._process_task(record)
                if not success:
                    with self._lock:
                        record.next_retry_at = time.time() + 0.5
                        self._scheduled_retries.append(record)
            except Exception as e:
                err_str = sanitize_text(str(e))
                logger.warning(f"Artifact upload failed for {record.artifact_id}: {err_str}")
                # Refresh record state from journal
                refreshed = self.journal.get_transfer(record.transfer_id) or record
                refreshed.retry_count += 1
                refreshed.last_error = err_str

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
                    self._queued_ids.discard(record.transfer_id)
                self._queue.task_done()

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Block until all enqueued items and scheduled retries are processed."""
        start = time.time()
        while True:
            with self._lock:
                queue_empty = self._queue.empty() and len(self._queued_ids) == 0
                active_count = len(self._active_transfers)
                retry_count = len(self._scheduled_retries)
            if queue_empty and active_count == 0 and retry_count == 0:
                return True
            if timeout is not None and (time.time() - start) >= timeout:
                return False
            time.sleep(0.05)
