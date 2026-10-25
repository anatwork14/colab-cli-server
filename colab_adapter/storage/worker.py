"""Background asynchronous worker for reliable cloud artifact uploads.

Processes transfer queue with bounded concurrency, automatic restart recovery,
per-file retries with backoff, and strict isolation from training job execution status.
"""
from __future__ import annotations

import logging
import os
import queue
import random
import shutil
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

if not hasattr(ArtifactStore, "compute_sha256"):
    setattr(ArtifactStore, "compute_sha256", staticmethod(compute_file_sha256))

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
        self._in_progress = self._active_transfers
        self._scheduled_retries: List[TransferRecord] = []
        self._lock = threading.Lock()

        # Load pending or interrupted transfers from persistent journal
        try:
            self.recover_from_journal()
        except Exception as ex:
            logger.warning("Error recovering from journal on init: %s", ex)
            self._quarantine_corrupt_journal()

    def _quarantine_corrupt_journal(self) -> None:
        """Quarantine corrupt journal file to .corrupt.<timestamp>.bak and reset with empty records."""
        ts = int(time.time() * 1000)
        corrupt_file = self.journal.path.parent / f"{self.journal.path.name}.corrupt.{ts}.bak"
        try:
            if self.journal.path.exists():
                shutil.copy2(str(self.journal.path), str(corrupt_file))
        except Exception:
            pass
        self.journal._write_records({})

    def _is_enqueued_or_active(self, transfer_id: str) -> bool:
        """Check whether transfer is queued, active, or in scheduled retries. Must hold self._lock."""
        return (
            transfer_id in self._queued_ids
            or transfer_id in self._active_transfers
            or any(r.transfer_id == transfer_id for r in self._scheduled_retries)
        )

    def recover_from_journal(self) -> None:
        """Scan persistent journal and recover pending transfers with lease fencing."""
        try:
            pending = self.journal.get_pending_transfers()
        except Exception as ex:
            logger.warning("Error loading pending transfers from journal: %s", ex)
            self._quarantine_corrupt_journal()
            pending = []
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
        local_path: Optional[Union[str, Path]] = None,
        project_id: Optional[str] = None,
        run_id: Optional[str] = None,
        artifact_type: str = ArtifactType.CHECKPOINT.value,
        expected_sha256: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *args,
        source: Optional[Union[str, Path]] = None,
        category: Optional[str] = None,
        **kwargs,
    ) -> TransferRecord:
        """Enqueue an artifact for background upload with immutable source staging and revision identity."""
        src_arg = source or local_path
        proj_arg = project_id
        run_arg = run_id
        cat_arg = category
        type_arg = artifact_type
        sha_arg = expected_sha256
        meta_arg = dict(metadata or {})
        if kwargs.get("metadata"):
            meta_arg.update(kwargs["metadata"])

        # Check if called positionally as enqueue(project_id, run_id, source, category, metadata, expected_sha256)
        if local_path is not None and project_id is not None and run_id is not None:
            if not isinstance(local_path, Path) and not Path(str(local_path)).exists() and (isinstance(run_id, Path) or Path(str(run_id)).exists() or "/" in str(run_id) or "\\" in str(run_id)):
                proj_arg = str(local_path)
                run_arg = str(project_id)
                src_arg = run_id
                cat_arg = artifact_type if artifact_type != ArtifactType.CHECKPOINT.value else category
                meta_arg = expected_sha256 if isinstance(expected_sha256, dict) else meta_arg
                if len(args) >= 1:
                    sha_arg = args[0]
            else:
                src_arg = local_path
                proj_arg = project_id
                run_arg = run_id

        if cat_arg is None:
            if "category" in kwargs:
                cat_arg = kwargs["category"]
            elif type_arg == ArtifactType.CHECKPOINT.value:
                cat_arg = "checkpoints"
            elif type_arg:
                cat_arg = type_arg

        if not src_arg:
            raise ArtifactValidationError("Missing artifact source path for enqueue.")
        if not proj_arg or not run_arg:
            raise ArtifactValidationError("Missing project_id or run_id for enqueue.")

        src = Path(src_arg).resolve()
        if not src.exists():
            raise ArtifactValidationError(f"Artifact source file does not exist: {src}")

        current_sha256 = ArtifactStore.compute_sha256(src)
        current_size = src.stat().st_size

        rel_path = f"{cat_arg}/{src.name}" if cat_arg else src.name
        artifact_id = f"{proj_arg}/{run_arg}/{rel_path}@sha256_{current_sha256.lower()}"
        transfer_id = f"{self.store.provider_name}-{proj_arg}-{run_arg}-{src.name}-{current_sha256.lower()[:12]}"

        # If expected_sha256 is provided and mismatches, record FAILED transfer record in self.journal and raise ArtifactValidationError
        if sha_arg and current_sha256.lower() != sha_arg.lower():
            failed_record = TransferRecord(
                transfer_id=transfer_id,
                project_id=proj_arg,
                run_id=run_arg,
                artifact_id=artifact_id,
                local_path=str(src),
                target_rel_path=rel_path,
                size_bytes=current_size,
                sha256=current_sha256,
                expected_sha256=sha_arg,
                source_path=str(src),
                status=TransferStatus.UPLOAD_FAILED.value,
                last_error=f"Expected SHA256 {sha_arg} but got {current_sha256}",
                metadata=meta_arg,
            )
            self.journal.record_transfer(failed_record)
            raise ArtifactValidationError(
                f"Artifact checksum mismatch for {src.name}: expected {sha_arg}, got {current_sha256}"
            )

        meta_arg["artifact_id"] = artifact_id
        meta_arg["artifact_type"] = type_arg
        if cat_arg:
            meta_arg["category"] = cat_arg

        with self.journal.acquire_artifact_lock(self.store.provider_name, proj_arg, run_arg, src.name):
            # Protect against in-place source file mutation after enqueue() using atomic temp copy under lock:
            staging_dir = self.journal.path.parent / ".staged_uploads" / f"{proj_arg}_{run_arg}_{current_sha256[:16]}"
            staging_dir.mkdir(parents=True, exist_ok=True)
            staged_file = staging_dir / src.name
            if not staged_file.exists() or ArtifactStore.compute_sha256(staged_file).lower() != current_sha256.lower():
                tmp_staged = staging_dir / f"{src.name}.tmp.{os.getpid()}.{threading.get_ident()}"
                shutil.copy2(str(src), str(tmp_staged))
                staged_sha256 = ArtifactStore.compute_sha256(tmp_staged)
                if staged_sha256.lower() != current_sha256.lower():
                    if tmp_staged.exists():
                        try:
                            tmp_staged.unlink()
                        except OSError:
                            pass
                    raise ArtifactValidationError("Source file mutated during enqueue staging")
                os.replace(str(tmp_staged), str(staged_file))

            self.journal._load()
            existing = self.journal.get_transfer(transfer_id)
            now = time.time()
            if (
                existing is not None
                and existing.status in (TransferStatus.PUBLISHED, TransferStatus.PUBLISHED.value)
                and (getattr(existing, "expected_sha256", None) or existing.sha256).lower() == current_sha256.lower()
            ):
                return existing

            if (
                existing is not None
                and existing.status in (
                    TransferStatus.UPLOAD_PENDING.value,
                    TransferStatus.UPLOADING.value,
                    TransferStatus.LOCAL_VERIFIED.value,
                    TransferStatus.RETRY_PENDING.value,
                )
            ):
                has_active_lease = bool(
                    existing.lease_token
                    and existing.lease_expires_at
                    and existing.lease_expires_at > now
                    and _is_pid_alive(existing.owner_pid)
                    and existing.owner_pid != os.getpid()
                )
                with self._lock:
                    in_flight = self._is_enqueued_or_active(transfer_id)
                if has_active_lease or in_flight:
                    return existing

            record = TransferRecord(
                transfer_id=transfer_id,
                project_id=proj_arg,
                run_id=run_arg,
                artifact_id=artifact_id,
                local_path=str(staged_file),
                target_rel_path=rel_path,
                size_bytes=current_size,
                sha256=current_sha256,
                expected_sha256=current_sha256,
                source_path=str(staged_file),
                status=TransferStatus.LOCAL_VERIFIED.value,
                metadata=meta_arg,
            )
            self.journal.record_transfer(record)

            with self._lock:
                if not self._is_enqueued_or_active(transfer_id):
                    self._queued_ids.add(transfer_id)
                    self._queue.put(record)

            return record

    def _process_task(self, task: UploadTask) -> bool:
        """Process an individual upload task with cross-process artifact locking and lease fencing."""
        src = Path(getattr(task, "source_path", None) or task.local_path)
        task_expected_sha = getattr(task, "expected_sha256", None) or task.sha256

        with self.journal.acquire_artifact_lock(self.store.provider_name, task.project_id, task.run_id, src.name, timeout=60.0):
            # Check existing record in journal: check task.transfer_id first, then legacy fallback if applicable
            existing = self.journal.get_transfer(task.transfer_id)
            if existing is None:
                legacy_id = f"{self.store.provider_name}-{task.project_id}-{task.run_id}-{src.name}"
                if task.transfer_id == legacy_id:
                    existing = self.journal.get_transfer(legacy_id)

            existing_sha = getattr(existing, "expected_sha256", None) or (existing.sha256 if existing else None)
            is_same_revision = (
                task_expected_sha is None
                or (existing_sha is not None and existing_sha.lower() == task_expected_sha.lower())
            )

            if (
                existing is not None
                and existing.status in (TransferStatus.PUBLISHED, TransferStatus.PUBLISHED.value)
                and existing.remote_file_id
                and is_same_revision
            ):
                logger.info(
                    "Artifact %s was already published by another worker/process; skipping upload.",
                    task.transfer_id,
                )
                if ".staged_uploads" in src.parts:
                    shutil.rmtree(src.parent, ignore_errors=True)
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
                expected_sha256=task_expected_sha,
            )
            if not claimed:
                existing_rec_sha = getattr(existing_rec, "expected_sha256", None) or (existing_rec.sha256 if existing_rec else None)
                is_same_claimed = (
                    task_expected_sha is None
                    or (existing_rec_sha is not None and existing_rec_sha.lower() == task_expected_sha.lower())
                )
                if existing_rec and existing_rec.status in (TransferStatus.PUBLISHED, TransferStatus.PUBLISHED.value) and is_same_claimed:
                    logger.info(
                        "Artifact %s was already published by another worker/process; skipping upload.",
                        task.transfer_id,
                    )
                    if ".staged_uploads" in src.parts:
                        shutil.rmtree(src.parent, ignore_errors=True)
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

            if not src.is_file():
                raise ArtifactValidationError(f"Local artifact source file missing: {src}")

            actual_sha256 = ArtifactStore.compute_sha256(src)
            if task_expected_sha and len(task_expected_sha) == 64 and actual_sha256.lower() != task_expected_sha.lower():
                raise ArtifactValidationError(
                    f"Local artifact modified before publication: expected {task_expected_sha}, got {actual_sha256}"
                )

            try:
                stored = self.store.publish(
                    source=src,
                    project_id=task.project_id,
                    run_id=task.run_id,
                    artifact_type=task.metadata.get("artifact_type", ArtifactType.CHECKPOINT.value) if task.metadata else ArtifactType.CHECKPOINT.value,
                    metadata=task.metadata,
                    expected_sha256=task_expected_sha or actual_sha256,
                )

                self.journal.mark_completed(task.transfer_id, remote_id=stored.remote_file_id)

                if ".staged_uploads" in src.parts:
                    shutil.rmtree(src.parent, ignore_errors=True)

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
