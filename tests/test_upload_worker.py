"""Unit tests for asynchronous background cloud upload worker and recovery.

Part of Colab V2 Cross-Device Workspaces & Google Drive (Issue #21).
"""
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from colab_adapter.errors import ArtifactValidationError
from colab_adapter.storage import (
    ArtifactStore,
    ArtifactType,
    ArtifactUploadWorker,
    StoredArtifact,
    TransferJournal,
    TransferRecord,
    TransferStatus,
)
from colab_adapter.storage.manifest import PublicationManifest, compute_file_sha256


class MockStoreForWorker:
    """Mock store recording calls and simulating successes/failures with optional manifest persistence."""

    def __init__(
        self,
        fail_first_n: int = 0,
        error_message: str = "Simulated transient cloud error",
        manifest_dir: Optional[Union[str, Path]] = None,
    ):
        self.published: List[StoredArtifact] = []
        self.fail_first_n = fail_first_n
        self.error_message = error_message
        self.provider_name = "mock_cloud"
        self.manifest_dir = Path(manifest_dir).resolve() if manifest_dir else None
        self._thread_lock = threading.Lock()

    def publish(
        self,
        source: Path,
        project_id: str,
        run_id: str,
        artifact_type: str = ArtifactType.CHECKPOINT.value,
        metadata: Optional[Dict[str, Any]] = None,
        expected_sha256: Optional[str] = None,
    ) -> StoredArtifact:
        if self.fail_first_n > 0:
            self.fail_first_n -= 1
            raise RuntimeError(self.error_message)

        p = Path(source)
        sha256 = expected_sha256 or compute_file_sha256(p)
        meta = dict(metadata or {})
        artifact_id = meta.get("artifact_id") or f"{project_id}/{run_id}/{p.name}"

        stored_file_path = str(p)
        if self.manifest_dir:
            store_run_dir = self.manifest_dir / project_id / run_id
            files_dir = store_run_dir / "files"
            files_dir.mkdir(parents=True, exist_ok=True)
            rev_file = files_dir / f"{p.name}@{sha256[:12]}"
            latest_file = files_dir / p.name
            shutil.copy2(str(p), str(rev_file))
            shutil.copy2(str(p), str(latest_file))
            stored_file_path = str(rev_file)

        stored = StoredArtifact(
            artifact_id=artifact_id,
            storage_provider=self.provider_name,
            path=stored_file_path,
            size_bytes=p.stat().st_size,
            sha256=sha256,
            project_id=project_id,
            run_id=run_id,
            artifact_type=artifact_type,
            remote_file_id=f"cloud-id-{p.name}-{sha256[:8]}",
            metadata=meta,
        )

        with self._thread_lock:
            self.published.append(stored)

        if self.manifest_dir:
            manifest_file = self.manifest_dir / project_id / run_id / "manifest.json"
            manifest_lock = self.manifest_dir / project_id / run_id / "manifest.lock"
            try:
                import fcntl
            except ImportError:
                fcntl = None
            fd = os.open(str(manifest_lock), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                if fcntl:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                if manifest_file.exists():
                    manifest = PublicationManifest.load(manifest_file)
                else:
                    manifest = PublicationManifest(project_id=project_id, run_id=run_id)
                manifest.add_artifact(stored)
                manifest.save(manifest_file)
            finally:
                if fcntl:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

        return stored

    def retrieve(self, artifact_id: str, destination: Path) -> Path:
        dest = Path(destination).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.manifest_dir:
            parts = artifact_id.split("/")
            if len(parts) >= 2:
                proj, run = parts[0], parts[1]
                m_path = self.manifest_dir / proj / run / "manifest.json"
                if m_path.exists():
                    manifest = PublicationManifest.load(m_path)
                    art = manifest.get_artifact(artifact_id)
                    if art and Path(art.path).exists():
                        shutil.copy2(art.path, str(dest))
                        return dest
        for a in reversed(self.published):
            if a.artifact_id == artifact_id:
                if Path(a.path).exists():
                    shutil.copy2(a.path, str(dest))
                    return dest
        return dest

    def exists(self, artifact_id: str) -> bool:
        if self.manifest_dir:
            parts = artifact_id.split("/")
            if len(parts) >= 2:
                proj, run = parts[0], parts[1]
                m_path = self.manifest_dir / proj / run / "manifest.json"
                if m_path.exists():
                    manifest = PublicationManifest.load(m_path)
                    return manifest.get_artifact(artifact_id) is not None
        return any(a.artifact_id == artifact_id for a in self.published)

    def get_metadata(self, artifact_id: str) -> Optional[StoredArtifact]:
        if self.manifest_dir:
            parts = artifact_id.split("/")
            if len(parts) >= 2:
                proj, run = parts[0], parts[1]
                m_path = self.manifest_dir / proj / run / "manifest.json"
                if m_path.exists():
                    manifest = PublicationManifest.load(m_path)
                    return manifest.get_artifact(artifact_id)
        for a in self.published:
            if a.artifact_id == artifact_id:
                return a
        return None

    def list_artifacts(self, project_id: str, run_id: Optional[str] = None) -> List[StoredArtifact]:
        if self.manifest_dir and run_id:
            m_path = self.manifest_dir / project_id / run_id / "manifest.json"
            if m_path.exists():
                return PublicationManifest.load(m_path).list_artifacts()
        return list(self.published)

    def get_manifest(self, project_id: str, run_id: str) -> Optional[PublicationManifest]:
        if self.manifest_dir:
            m_path = self.manifest_dir / project_id / run_id / "manifest.json"
            if m_path.exists():
                return PublicationManifest.load(m_path)
        return None


def _mp_worker_same_revision_entrypoint(journal_path_str: str, store_dir_str: str, file_path_str: str, project_id: str, run_id: str):
    journal = TransferJournal(journal_path_str)
    store = MockStoreForWorker(manifest_dir=store_dir_str)
    worker = ArtifactUploadWorker(store=store, journal=journal, max_workers=1)
    worker.enqueue(local_path=file_path_str, project_id=project_id, run_id=run_id)
    worker.start()
    try:
        worker.wait_for_completion(timeout=5.0)
    finally:
        worker.stop()


def _mp_worker_diff_revision_entrypoint(journal_path_str: str, store_dir_str: str, file_path_str: str, file_bytes: bytes, project_id: str, run_id: str):
    p = Path(file_path_str)
    p.write_bytes(file_bytes)
    journal = TransferJournal(journal_path_str)
    store = MockStoreForWorker(manifest_dir=store_dir_str)
    worker = ArtifactUploadWorker(store=store, journal=journal, max_workers=1)
    worker.enqueue(local_path=file_path_str, project_id=project_id, run_id=run_id, category="checkpoints")
    worker.start()
    try:
        worker.wait_for_completion(timeout=5.0)
    finally:
        worker.stop()


class TestUploadWorker(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="colab_worker_test_")
        self.addCleanup(lambda: shutil.rmtree(self.test_dir, ignore_errors=True))
        self.journal_file = Path(self.test_dir) / "transfers.json"
        self.journal = TransferJournal(self.journal_file)

    def test_worker_asynchronous_queue_processing(self):
        """Worker asynchronously processes enqueued artifacts in background."""
        store = MockStoreForWorker()
        successes = []
        worker = ArtifactUploadWorker(
            store=store,
            journal=self.journal,
            max_workers=2,
            on_success=lambda a: successes.append(a),
        )
        worker.start()
        self.addCleanup(worker.stop)

        # Create two test files
        f1 = Path(self.test_dir) / "ckpt_1.pt"
        f2 = Path(self.test_dir) / "ckpt_2.pt"
        f1.write_bytes(b"DATA_1")
        f2.write_bytes(b"DATA_2")

        worker.enqueue(f1, project_id="p1", run_id="r1")
        worker.enqueue(f2, project_id="p1", run_id="r1")

        done = worker.wait_for_completion(timeout=3.0)
        self.assertTrue(done)
        self.assertEqual(len(successes), 2)
        self.assertEqual(len(store.published), 2)

    def test_worker_recovers_pending_tasks_on_restart(self):
        """Worker started with existing journal automatically resumes pending tasks."""
        # Pre-seed journal with a pending transfer
        f1 = Path(self.test_dir) / "interrupted.pt"
        f1.write_bytes(b"INTERRUPTED_DATA")

        rec = TransferRecord(
            transfer_id="mock_cloud-p1-r1-interrupted.pt",
            project_id="p1",
            run_id="r1",
            artifact_id="p1/r1/interrupted.pt",
            local_path=str(f1),
            target_rel_path="checkpoints/interrupted.pt",
            size_bytes=f1.stat().st_size,
            sha256="fake_sha",
            status=TransferStatus.UPLOADING.value,
        )
        self.journal.record_transfer(rec)

        store = MockStoreForWorker()
        worker = ArtifactUploadWorker(store=store, journal=self.journal, max_workers=1)
        worker.start()
        self.addCleanup(worker.stop)

        done = worker.wait_for_completion(timeout=3.0)
        self.assertTrue(done)
        self.assertEqual(len(store.published), 1)
        self.assertEqual(store.published[0].artifact_id, "p1/r1/interrupted.pt")

    def test_cloud_failure_preserves_local_file_and_records_error(self):
        """Cloud upload failure records error state without deleting local checkpoint."""
        store = MockStoreForWorker(fail_first_n=10)  # Always fail
        failures = []
        worker = ArtifactUploadWorker(
            store=store,
            journal=self.journal,
            max_workers=1,
            on_failure=lambda r, e: failures.append((r, e)),
        )
        worker.start()
        self.addCleanup(worker.stop)

        f = Path(self.test_dir) / "critical_local.pt"
        f.write_bytes(b"CRITICAL_CHECKPOINT")

        worker.enqueue(f, project_id="p1", run_id="r1")
        worker.wait_for_completion(timeout=2.0)

        # Local file must still exist!
        self.assertTrue(f.exists(), "Local checkpoint must never be deleted on upload failure")
        self.assertGreaterEqual(len(failures), 1)
        record = self.journal.get_transfer(failures[0][0].transfer_id)
        self.assertIn(record.status, (TransferStatus.RETRY_PENDING.value, TransferStatus.UPLOAD_FAILED.value))

    def test_seeded_secrets_in_exceptions_redacted_in_last_error_and_to_dict_safe(self):
        """Seeded secrets in upload exceptions and metadata are redacted in last_error and to_dict(safe=True)."""
        leak_msg = (
            "Drive error with access_token=ya29.secret_access_token_12345678, "
            "refresh_token=1//04ref_token_secret_abcdef, "
            "client_secret=super_secret_client_val "
            "and resumable session URL: "
            "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&session_id=SECRET_123"
        )
        store = MockStoreForWorker(fail_first_n=10, error_message=leak_msg)
        failures = []
        worker = ArtifactUploadWorker(
            store=store,
            journal=self.journal,
            max_workers=1,
            on_failure=lambda r, e: failures.append((r, e)),
        )
        worker.start()
        self.addCleanup(worker.stop)

        f = Path(self.test_dir) / "secret_ckpt.pt"
        f.write_bytes(b"SENSITIVE_CHECKPOINT_DATA")

        worker.enqueue(
            f,
            project_id="p1",
            run_id="r1",
            metadata={
                "id_token": "secret_id_token_val",
                "proxy_secret": "my_proxy_secret_val",
                "safe_param": "safe_val",
            },
        )
        worker.wait_for_completion(timeout=2.0)

        record = self.journal.get_transfer("mock_cloud-p1-r1-secret_ckpt.pt")
        self.assertIsNotNone(record)
        self.assertIsNotNone(record.last_error)

        # Raw secrets must NOT be present in record.last_error
        self.assertNotIn("SECRET_123", record.last_error)
        self.assertNotIn("secret_access_token_12345678", record.last_error)
        self.assertNotIn("1//04ref_token_secret_abcdef", record.last_error)
        self.assertNotIn("super_secret_client_val", record.last_error)

        # Redaction placeholders must be present
        self.assertIn("[REDACTED_UPLOAD_SESSION_URL]", record.last_error)
        self.assertIn("[REDACTED_SECRET]", record.last_error)

        # Test to_dict(safe=True)
        safe_dict = record.to_dict(safe=True)
        self.assertNotIn("SECRET_123", str(safe_dict))
        self.assertNotIn("secret_access_token_12345678", str(safe_dict))
        self.assertNotIn("secret_id_token_val", str(safe_dict))
        self.assertNotIn("my_proxy_secret_val", str(safe_dict))
        self.assertEqual(safe_dict["metadata"]["id_token"], "[REDACTED]")
        self.assertEqual(safe_dict["metadata"]["proxy_secret"], "[REDACTED]")
        self.assertEqual(safe_dict["metadata"]["safe_param"], "safe_val")

    def test_worker_enqueue_multiple_times_deduplicates_and_uploads_once(self):
        """Calling worker.enqueue() multiple times rapidly for the same file deduplicates queue entries."""
        store = MockStoreForWorker()
        successes = []
        worker = ArtifactUploadWorker(
            store=store,
            journal=self.journal,
            max_workers=2,
            on_success=lambda a: successes.append(a),
        )
        worker.start()
        self.addCleanup(worker.stop)

        f = Path(self.test_dir) / "rapid_ckpt.pt"
        f.write_bytes(b"RAPID_CHECKPOINT_DATA")

        # Enqueue rapidly 5 times in a row
        for _ in range(5):
            worker.enqueue(f, project_id="p1", run_id="r1")

        done = worker.wait_for_completion(timeout=3.0)
        self.assertTrue(done)
        self.assertEqual(len(store.published), 1, "Duplicate enqueues must only result in a single upload")
        self.assertEqual(len(successes), 1)

    def test_worker_enqueue_revalidates_and_reuploads_when_checkpoint_modified(self):
        """Updating local checkpoint file causes subsequent enqueue to re-upload new content."""
        store = MockStoreForWorker()
        successes = []
        worker = ArtifactUploadWorker(
            store=store,
            journal=self.journal,
            max_workers=1,
            on_success=lambda a: successes.append(a),
        )
        worker.start()
        self.addCleanup(worker.stop)

        f = Path(self.test_dir) / "evolving_ckpt.pt"
        f.write_bytes(b"VERSION_1_DATA")

        worker.enqueue(f, project_id="p1", run_id="r1")
        self.assertTrue(worker.wait_for_completion(timeout=3.0))
        self.assertEqual(len(store.published), 1)
        self.assertEqual(store.published[0].size_bytes, len(b"VERSION_1_DATA"))

        # Now simulate a newer checkpoint version written to the same filename
        f.write_bytes(b"VERSION_2_UPDATED_LARGER_DATA")

        worker.enqueue(f, project_id="p1", run_id="r1")
        self.assertTrue(worker.wait_for_completion(timeout=3.0))
        self.assertEqual(len(store.published), 2)
        self.assertEqual(store.published[1].size_bytes, len(b"VERSION_2_UPDATED_LARGER_DATA"))

        # Journal record should reflect new size and status
        record = self.journal.get_transfer("mock_cloud-p1-r1-evolving_ckpt.pt")
        self.assertEqual(record.size_bytes, len(b"VERSION_2_UPDATED_LARGER_DATA"))

    def test_acquire_artifact_lock_reentrancy_and_blocking_and_cleanup(self):
        """acquire_artifact_lock is re-entrant within thread, blocks threads/processes, and cleans up."""
        # 1. Re-entrancy within the same thread
        with self.journal.acquire_artifact_lock("transfer_reentrant", timeout=5.0) as p1:
            with self.journal.acquire_artifact_lock("transfer_reentrant", timeout=5.0) as p2:
                with self.journal.acquire_artifact_lock("transfer_reentrant", timeout=5.0) as p3:
                    self.assertEqual(p1, p2)
                    self.assertEqual(p2, p3)
                    self.assertTrue(p1.exists())
                    # Permissions on lock file must be 0600
                    mode = p1.stat().st_mode & 0o777
                    self.assertEqual(mode, 0o600)

        # Check cleanup of thread-local state
        held = getattr(TransferJournal._local_state, "held_locks", {})
        key = (str((self.journal.path.parent / ".transfer_locks").resolve()), "artifact-transfer_reentrant")
        self.assertNotIn(key, held)

        # 2. Blocks concurrent threads in the same process
        thread_started = threading.Event()
        thread_release = threading.Event()
        thread_error: List[Exception] = []

        def holding_thread():
            try:
                with self.journal.acquire_artifact_lock("transfer_thread_block", timeout=5.0):
                    thread_started.set()
                    thread_release.wait(timeout=5.0)
            except Exception as ex:
                thread_error.append(ex)

        t = threading.Thread(target=holding_thread)
        t.start()
        self.assertTrue(thread_started.wait(timeout=3.0))

        # Main thread attempts to acquire same lock; must time out
        with self.assertRaises(TimeoutError):
            with self.journal.acquire_artifact_lock("transfer_thread_block", timeout=0.2):
                pass

        # Release thread and join
        thread_release.set()
        t.join(timeout=3.0)
        self.assertEqual(len(thread_error), 0)

        # Main thread can now acquire without timeout
        with self.journal.acquire_artifact_lock("transfer_thread_block", timeout=2.0) as p:
            self.assertTrue(p.exists())

        # 3. Blocks concurrent separate OS processes
        child_code = (
            "import sys, time\n"
            "from colab_adapter.storage import TransferJournal\n"
            "j = TransferJournal(sys.argv[1])\n"
            "with j.acquire_artifact_lock('transfer_proc_block', timeout=5.0):\n"
            "    sys.stdout.write('LOCKED\\n')\n"
            "    sys.stdout.flush()\n"
            "    sys.stdin.readline()\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", child_code, str(self.journal_file)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            line = proc.stdout.readline().strip()
            self.assertEqual(line, "LOCKED")

            # Main process attempts to acquire; must time out due to cross-process flock
            with self.assertRaises(TimeoutError):
                with self.journal.acquire_artifact_lock("transfer_proc_block", timeout=0.2):
                    pass

            # Signal child process to release lock
            proc.stdin.write("DONE\n")
            proc.stdin.flush()
            proc.wait(timeout=3.0)

            # Main process can now acquire cleanly
            with self.journal.acquire_artifact_lock("transfer_proc_block", timeout=2.0) as p:
                self.assertTrue(p.exists())
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream and not stream.closed:
                    stream.close()

    def test_claim_transfer_lease_and_fencing(self):
        """claim_transfer_lease enforces live PID exclusion, detects PUBLISHED, and reclaims dead PIDs."""
        # 1. Exclusion of duplicate claims across live PIDs
        sub = subprocess.Popen(["sleep", "10"])
        try:
            rec_live = TransferRecord(
                transfer_id="t_live",
                project_id="p1",
                run_id="r1",
                artifact_id="p1/r1/ckpt.pt",
                local_path=str(Path(self.test_dir) / "dummy.pt"),
                target_rel_path="checkpoints/dummy.pt",
                size_bytes=100,
                sha256="hash123",
                status=TransferStatus.UPLOADING.value,
                owner_pid=sub.pid,
                lease_token="live_token_123",
                lease_expires_at=time.time() + 300.0,
                fencing_epoch=1,
            )
            self.journal.record_transfer(rec_live)

            claimed, existing, tok = self.journal.claim_transfer_lease("t_live")
            self.assertFalse(claimed)
            self.assertEqual(tok, "")
            self.assertIsNotNone(existing)
            self.assertEqual(existing.owner_pid, sub.pid)
            self.assertEqual(existing.lease_token, "live_token_123")
        finally:
            sub.terminate()
            sub.wait()

        # 2. Detects already-PUBLISHED records with matching sha256
        rec_pub = TransferRecord(
            transfer_id="t_pub",
            project_id="p1",
            run_id="r1",
            artifact_id="p1/r1/pub.pt",
            local_path=str(Path(self.test_dir) / "pub.pt"),
            target_rel_path="checkpoints/pub.pt",
            size_bytes=200,
            sha256="known_sha_abc",
            status=TransferStatus.PUBLISHED.value,
            remote_file_id="drive_file_id_xyz",
        )
        self.journal.record_transfer(rec_pub)

        claimed, existing, tok = self.journal.claim_transfer_lease("t_pub", expected_sha256="known_sha_abc")
        self.assertFalse(claimed)
        self.assertEqual(tok, "")
        self.assertIsNotNone(existing)
        self.assertEqual(existing.status, TransferStatus.PUBLISHED.value)
        self.assertEqual(existing.remote_file_id, "drive_file_id_xyz")

        # 3. Immediately reclaims leases held by dead PIDs (owner_pid = 99999999)
        rec_dead = TransferRecord(
            transfer_id="t_dead",
            project_id="p1",
            run_id="r1",
            artifact_id="p1/r1/dead.pt",
            local_path=str(Path(self.test_dir) / "dead.pt"),
            target_rel_path="checkpoints/dead.pt",
            size_bytes=300,
            sha256="dead_sha",
            status=TransferStatus.UPLOADING.value,
            owner_pid=99999999,
            lease_token="stale_token_dead",
            lease_expires_at=time.time() + 300.0,
            fencing_epoch=2,
        )
        self.journal.record_transfer(rec_dead)

        claimed, updated, tok = self.journal.claim_transfer_lease("t_dead")
        self.assertTrue(claimed)
        self.assertNotEqual(tok, "stale_token_dead")
        self.assertEqual(updated.owner_pid, os.getpid())
        self.assertEqual(updated.lease_token, tok)
        self.assertEqual(updated.fencing_epoch, 3)

        # Release lease and verify clearance in journal
        self.journal.release_transfer_lease("t_dead", tok)
        refreshed = self.journal.get_transfer("t_dead")
        self.assertIsNone(refreshed.lease_token)
        self.assertIsNone(refreshed.lease_expires_at)
        self.assertIsNone(refreshed.owner_pid)

    def test_two_workers_concurrent_upload_executes_publish_only_once(self):
        """Two ArtifactUploadWorker instances running concurrently on the same artifact publish only once."""
        store = MockStoreForWorker()
        successes_1: List[StoredArtifact] = []
        successes_2: List[StoredArtifact] = []

        worker_1 = ArtifactUploadWorker(
            store=store,
            journal=self.journal,
            max_workers=1,
            on_success=lambda a: successes_1.append(a),
        )
        worker_2 = ArtifactUploadWorker(
            store=store,
            journal=self.journal,
            max_workers=1,
            on_success=lambda a: successes_2.append(a),
        )

        f = Path(self.test_dir) / "concurrent_ckpt.pt"
        f.write_bytes(b"CONCURRENT_SHARED_PAYLOAD")

        # Both workers enqueue the exact same artifact
        worker_1.enqueue(f, project_id="p_shared", run_id="r_shared")
        worker_2.enqueue(f, project_id="p_shared", run_id="r_shared")

        worker_1.start()
        worker_2.start()
        self.addCleanup(worker_1.stop)
        self.addCleanup(worker_2.stop)

        self.assertTrue(worker_1.wait_for_completion(timeout=5.0))
        self.assertTrue(worker_2.wait_for_completion(timeout=5.0))

        # Under concurrency, store.publish must have been executed exactly ONCE
        self.assertEqual(
            len(store.published),
            1,
            f"Expected store.publish called exactly once, but got {len(store.published)}",
        )
        self.assertGreaterEqual(len(successes_1) + len(successes_2), 1)

        # Journal record must be PUBLISHED
        transfer_id = f"mock_cloud-p_shared-r_shared-concurrent_ckpt.pt"
        record = self.journal.get_transfer(transfer_id)
        self.assertIsNotNone(record)
        self.assertEqual(record.status, TransferStatus.PUBLISHED.value)
        self.assertIsNotNone(record.remote_file_id)

    def test_worker_recovery_skips_live_pid_lease_and_recovers_dead_pid(self):
        """recover_from_journal skips records held by live PIDs and immediately recovers dead PIDs."""
        sub = subprocess.Popen(["sleep", "10"])
        try:
            f_live = Path(self.test_dir) / "live_active.pt"
            f_live.write_bytes(b"LIVE_ACTIVE")
            rec_live = TransferRecord(
                transfer_id="mock_cloud-p1-r1-live_active.pt",
                project_id="p1",
                run_id="r1",
                artifact_id="p1/r1/live_active.pt",
                local_path=str(f_live),
                target_rel_path="checkpoints/live_active.pt",
                size_bytes=f_live.stat().st_size,
                sha256="sha_live",
                status=TransferStatus.UPLOADING.value,
                owner_pid=sub.pid,
                lease_token="live_token_xyz",
                lease_expires_at=time.time() + 300.0,
            )
            self.journal.record_transfer(rec_live)

            f_dead = Path(self.test_dir) / "dead_abandoned.pt"
            f_dead.write_bytes(b"DEAD_ABANDONED")
            rec_dead = TransferRecord(
                transfer_id="mock_cloud-p1-r1-dead_abandoned.pt",
                project_id="p1",
                run_id="r1",
                artifact_id="p1/r1/dead_abandoned.pt",
                local_path=str(f_dead),
                target_rel_path="checkpoints/dead_abandoned.pt",
                size_bytes=f_dead.stat().st_size,
                sha256="sha_dead",
                status=TransferStatus.UPLOADING.value,
                owner_pid=99999999,
                lease_token="dead_token_xyz",
                lease_expires_at=time.time() + 300.0,
            )
            self.journal.record_transfer(rec_dead)

            store = MockStoreForWorker()
            worker = ArtifactUploadWorker(store=store, journal=self.journal, max_workers=1)

            # Check queue inside worker: dead abandoned should be enqueued, live active should be skipped
            with worker._lock:
                queued = set(worker._queued_ids)

            self.assertIn("mock_cloud-p1-r1-dead_abandoned.pt", queued)
            self.assertNotIn("mock_cloud-p1-r1-live_active.pt", queued)

            # Start worker and ensure dead abandoned is processed
            worker.start()
            self.addCleanup(worker.stop)
            self.assertTrue(worker.wait_for_completion(timeout=3.0))

            published_ids = [a.artifact_id for a in store.published]
            self.assertIn("p1/r1/dead_abandoned.pt", published_ids)
            self.assertNotIn("p1/r1/live_active.pt", published_ids)
        finally:
            sub.terminate()
            sub.wait()

    def test_scenario_1_same_revision_published_twice_sequentially(self):
        """Scenario 1: Same revision published twice sequentially: Deduplicated cleanly, manifest and journal valid, 1 revision stored."""
        store_dir = Path(self.test_dir) / "store_s1"
        store = MockStoreForWorker(manifest_dir=store_dir)
        worker = ArtifactUploadWorker(store=store, journal=self.journal, max_workers=1)
        worker.start()
        self.addCleanup(worker.stop)

        f = Path(self.test_dir) / "model.pt"
        f.write_bytes(b"REVISION_A_CONTENT")
        expected_sha = compute_file_sha256(f)

        # First publish
        rec1 = worker.enqueue(f, project_id="p1", run_id="r1", category="checkpoints")
        self.assertTrue(worker.wait_for_completion(timeout=3.0))
        self.assertEqual(len(store.published), 1)

        # Second publish of same revision
        rec2 = worker.enqueue(f, project_id="p1", run_id="r1", category="checkpoints")
        self.assertTrue(worker.wait_for_completion(timeout=3.0))

        # Must be deduplicated cleanly - publish called only once!
        self.assertEqual(len(store.published), 1)
        self.assertEqual(rec1.transfer_id, rec2.transfer_id)

        # Journal valid: 1 record
        transfers = self.journal.list_transfers_for_run("p1", "r1")
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].status, TransferStatus.PUBLISHED.value)
        self.assertEqual(transfers[0].sha256, expected_sha)

        # Manifest valid: 1 revision stored
        manifest = store.get_manifest("p1", "r1")
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.get_artifact(rec1.artifact_id).sha256, expected_sha)

    def test_scenario_2_same_revision_published_by_two_threads_concurrently(self):
        """Scenario 2: Same revision published by two threads concurrently: Single logical revision, no corrupt manifest/journal, both succeed."""
        store_dir = Path(self.test_dir) / "store_s2"
        store = MockStoreForWorker(manifest_dir=store_dir)

        worker1 = ArtifactUploadWorker(store=store, journal=self.journal, max_workers=1)
        worker2 = ArtifactUploadWorker(store=store, journal=self.journal, max_workers=1)

        f = Path(self.test_dir) / "model.pt"
        f.write_bytes(b"CONCURRENT_THREAD_REVISION")
        expected_sha = compute_file_sha256(f)

        # Launch concurrent enqueues from two threads
        t1 = threading.Thread(target=lambda: worker1.enqueue(f, project_id="p_th", run_id="r_th", category="checkpoints"))
        t2 = threading.Thread(target=lambda: worker2.enqueue(f, project_id="p_th", run_id="r_th", category="checkpoints"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        worker1.start()
        worker2.start()
        self.addCleanup(worker1.stop)
        self.addCleanup(worker2.stop)

        self.assertTrue(worker1.wait_for_completion(timeout=5.0))
        self.assertTrue(worker2.wait_for_completion(timeout=5.0))

        # Single logical revision published once
        self.assertEqual(len(store.published), 1)

        # Journal valid: 1 transfer record
        transfers = self.journal.list_transfers_for_run("p_th", "r_th")
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].status, TransferStatus.PUBLISHED.value)
        self.assertEqual(transfers[0].sha256, expected_sha)

        # Manifest valid
        manifest = store.get_manifest("p_th", "r_th")
        self.assertIsNotNone(manifest)
        art_key = f"p_th/r_th/checkpoints/model.pt@sha256_{expected_sha}"
        self.assertEqual(manifest.artifacts[art_key]["sha256"], expected_sha)

    def test_scenario_3_same_revision_published_by_two_os_processes_concurrently(self):
        """Scenario 3: Same revision published by two OS processes concurrently: Cross-process lock coordinates access, valid manifest and journal."""
        store_dir = Path(self.test_dir) / "store_s3"
        f = Path(self.test_dir) / "model.pt"
        f.write_bytes(b"MULTIPROC_IDENTICAL_REVISION")
        expected_sha = compute_file_sha256(f)

        ctx = multiprocessing.get_context("spawn")
        p1 = ctx.Process(
            target=_mp_worker_same_revision_entrypoint,
            args=(str(self.journal_file), str(store_dir), str(f), "p_mp3", "r_mp3"),
        )
        p2 = ctx.Process(
            target=_mp_worker_same_revision_entrypoint,
            args=(str(self.journal_file), str(store_dir), str(f), "p_mp3", "r_mp3"),
        )

        p1.start()
        p2.start()
        p1.join(timeout=10.0)
        p2.join(timeout=10.0)

        self.assertEqual(p1.exitcode, 0)
        self.assertEqual(p2.exitcode, 0)

        # Check journal
        transfers = self.journal.list_transfers_for_run("p_mp3", "r_mp3")
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].status, TransferStatus.PUBLISHED.value)
        self.assertEqual(transfers[0].sha256, expected_sha)

        # Check manifest
        store = MockStoreForWorker(manifest_dir=store_dir)
        manifest = store.get_manifest("p_mp3", "r_mp3")
        self.assertIsNotNone(manifest)
        self.assertTrue(any(a["sha256"] == expected_sha for a in manifest.artifacts.values()))

    def test_scenario_4_different_revisions_same_filename_published_sequentially(self):
        """Scenario 4: Different revision, same filename (model.pt), published sequentially (Revision A then Revision B): Both revisions preserved in manifest and journal, latest pointer updated to Revision B, both retrievable!"""
        store_dir = Path(self.test_dir) / "store_s4"
        store = MockStoreForWorker(manifest_dir=store_dir)
        worker = ArtifactUploadWorker(store=store, journal=self.journal, max_workers=1)
        worker.start()
        self.addCleanup(worker.stop)

        f = Path(self.test_dir) / "model.pt"

        # 1. Publish Revision A
        rev_a_bytes = b"REVISION_A_WEIGHTS_V1"
        f.write_bytes(rev_a_bytes)
        sha_a = compute_file_sha256(f)
        rec_a = worker.enqueue(f, project_id="p4", run_id="r4", category="checkpoints")
        self.assertTrue(worker.wait_for_completion(timeout=3.0))

        # 2. Publish Revision B
        rev_b_bytes = b"REVISION_B_WEIGHTS_V2_LARGER"
        f.write_bytes(rev_b_bytes)
        sha_b = compute_file_sha256(f)
        rec_b = worker.enqueue(f, project_id="p4", run_id="r4", category="checkpoints")
        self.assertTrue(worker.wait_for_completion(timeout=3.0))

        # Both revisions in journal
        transfers = self.journal.list_transfers_for_run("p4", "r4")
        self.assertEqual(len(transfers), 2)
        transfer_shas = {t.sha256 for t in transfers}
        self.assertEqual(transfer_shas, {sha_a, sha_b})

        # Query specific revisions via get_published_revision
        pub_a = self.journal.get_published_revision("p4", "r4", rec_a.artifact_id)
        self.assertIsNotNone(pub_a)
        self.assertEqual(pub_a.sha256, sha_a)

        pub_b = self.journal.get_published_revision("p4", "r4", rec_b.artifact_id)
        self.assertIsNotNone(pub_b)
        self.assertEqual(pub_b.sha256, sha_b)

        # Latest pointer query returns Revision B
        latest_pub = self.journal.get_published_revision("p4", "r4", "p4/r4/checkpoints/model.pt")
        self.assertIsNotNone(latest_pub)
        self.assertEqual(latest_pub.sha256, sha_b)

        # Manifest contains both revisions and latest pointer
        manifest = store.get_manifest("p4", "r4")
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.get_artifact(rec_a.artifact_id).sha256, sha_a)
        self.assertEqual(manifest.get_artifact(rec_b.artifact_id).sha256, sha_b)
        self.assertEqual(manifest.get_artifact("p4/r4/checkpoints/model.pt").sha256, sha_b)

        # Both retrievable
        out_a = store.retrieve(rec_a.artifact_id, Path(self.test_dir) / "retrieved_s4_a.pt")
        out_b = store.retrieve(rec_b.artifact_id, Path(self.test_dir) / "retrieved_s4_b.pt")
        out_latest = store.retrieve("p4/r4/checkpoints/model.pt", Path(self.test_dir) / "retrieved_s4_latest.pt")

        self.assertEqual(out_a.read_bytes(), rev_a_bytes)
        self.assertEqual(out_b.read_bytes(), rev_b_bytes)
        self.assertEqual(out_latest.read_bytes(), rev_b_bytes)

    def test_scenario_5_different_revisions_same_filename_published_by_two_os_processes_concurrently(self):
        """Scenario 5: Different revision, same filename (model.pt), published by two OS processes concurrently: Both revisions preserved in TransferJournal and PublicationManifest, neither lost or overwritten!"""
        store_dir = Path(self.test_dir) / "store_s5"
        f1 = Path(self.test_dir) / "p1_dir" / "model.pt"
        f2 = Path(self.test_dir) / "p2_dir" / "model.pt"
        f1.parent.mkdir(parents=True, exist_ok=True)
        f2.parent.mkdir(parents=True, exist_ok=True)

        rev_a_bytes = b"MP_PROCESS_1_REVISION_A_WEIGHTS"
        rev_b_bytes = b"MP_PROCESS_2_REVISION_B_WEIGHTS"
        f1.write_bytes(rev_a_bytes)
        f2.write_bytes(rev_b_bytes)
        sha_a = compute_file_sha256(f1)
        sha_b = compute_file_sha256(f2)

        ctx = multiprocessing.get_context("spawn")
        p1 = ctx.Process(
            target=_mp_worker_diff_revision_entrypoint,
            args=(str(self.journal_file), str(store_dir), str(f1), rev_a_bytes, "p_mp5", "r_mp5"),
        )
        p2 = ctx.Process(
            target=_mp_worker_diff_revision_entrypoint,
            args=(str(self.journal_file), str(store_dir), str(f2), rev_b_bytes, "p_mp5", "r_mp5"),
        )

        p1.start()
        p2.start()
        p1.join(timeout=10.0)
        p2.join(timeout=10.0)

        self.assertEqual(p1.exitcode, 0)
        self.assertEqual(p2.exitcode, 0)

        # Both revisions preserved in TransferJournal
        transfers = self.journal.list_transfers_for_run("p_mp5", "r_mp5")
        self.assertEqual(len(transfers), 2)
        shas = {t.sha256 for t in transfers}
        self.assertEqual(shas, {sha_a, sha_b})
        for t in transfers:
            self.assertEqual(t.status, TransferStatus.PUBLISHED.value)

        # Both revisions preserved in PublicationManifest
        store = MockStoreForWorker(manifest_dir=store_dir)
        manifest = store.get_manifest("p_mp5", "r_mp5")
        self.assertIsNotNone(manifest)
        art_a_id = f"p_mp5/r_mp5/checkpoints/model.pt@sha256_{sha_a}"
        art_b_id = f"p_mp5/r_mp5/checkpoints/model.pt@sha256_{sha_b}"
        self.assertIn(art_a_id, manifest.artifacts)
        self.assertIn(art_b_id, manifest.artifacts)
        self.assertEqual(manifest.artifacts[art_a_id]["sha256"], sha_a)
        self.assertEqual(manifest.artifacts[art_b_id]["sha256"], sha_b)

        # Neither lost: both retrievable
        out_a = store.retrieve(art_a_id, Path(self.test_dir) / "retrieved_s5_a.pt")
        out_b = store.retrieve(art_b_id, Path(self.test_dir) / "retrieved_s5_b.pt")
        self.assertEqual(out_a.read_bytes(), rev_a_bytes)
        self.assertEqual(out_b.read_bytes(), rev_b_bytes)

    def test_scenario_6_source_content_changes_on_disk_immediately_after_enqueue(self):
        """Scenario 6: Source content changes on disk immediately after worker.enqueue(): Verify Revision A is published intact from staged snapshot; mismatched expected_sha256 raises ArtifactValidationError."""
        store = MockStoreForWorker()
        worker = ArtifactUploadWorker(store=store, journal=self.journal, max_workers=1)

        f = Path(self.test_dir) / "model.pt"
        rev_a_bytes = b"REVISION_A_BEFORE_WORKER_STARTS"
        f.write_bytes(rev_a_bytes)
        sha_a = compute_file_sha256(f)

        # Worker is paused / NOT started yet!
        rec = worker.enqueue(f, project_id="p6", run_id="r6", category="checkpoints")

        # Immediately overwrite model.pt on disk with Revision B before worker processes queue
        rev_b_bytes = b"REVISION_B_OVERWRITTEN_ON_DISK"
        f.write_bytes(rev_b_bytes)

        # Now start worker
        worker.start()
        self.addCleanup(worker.stop)
        self.assertTrue(worker.wait_for_completion(timeout=3.0))

        # Revision A must have been published intact from immutable staged snapshot!
        self.assertEqual(len(store.published), 1)
        self.assertEqual(store.published[0].sha256, sha_a)
        self.assertEqual(store.published[0].size_bytes, len(rev_a_bytes))

        # Check expected_sha256 mismatch rejection at enqueue time
        f_bad = Path(self.test_dir) / "bad_sha.pt"
        f_bad.write_bytes(b"DATA")
        wrong_sha = "0000000000000000000000000000000000000000000000000000000000000000"
        with self.assertRaises(ArtifactValidationError):
            worker.enqueue(f_bad, project_id="p6", run_id="r6", expected_sha256=wrong_sha)

        failed_records = self.journal.list_transfers(project_id="p6", run_id="r6", status=TransferStatus.UPLOAD_FAILED)
        self.assertGreaterEqual(len(failed_records), 1)
        self.assertTrue(any("bad_sha.pt" in r.transfer_id for r in failed_records))

    def test_scenario_7_worker_crash_and_journal_recovery(self):
        """Scenario 7: Worker crash before manifest commit / after partial upload: Recovered via recover_from_journal() and completed cleanly."""
        f = Path(self.test_dir) / "crashed_ckpt.pt"
        f.write_bytes(b"CRASHED_CHECKPOINT_DATA")
        sha = compute_file_sha256(f)

        # Seed record in UPLOADING state with dead owner_pid
        transfer_id = f"mock_cloud-p7-r7-crashed_ckpt.pt-{sha[:12]}"
        rec = TransferRecord(
            transfer_id=transfer_id,
            project_id="p7",
            run_id="r7",
            artifact_id=f"p7/r7/checkpoints/crashed_ckpt.pt@sha256_{sha}",
            local_path=str(f),
            target_rel_path="checkpoints/crashed_ckpt.pt",
            size_bytes=f.stat().st_size,
            sha256=sha,
            expected_sha256=sha,
            status=TransferStatus.UPLOADING.value,
            owner_pid=99999999,
            lease_token="dead_crashed_token",
            lease_expires_at=time.time() + 300.0,
            fencing_epoch=1,
        )
        self.journal.record_transfer(rec)

        store = MockStoreForWorker()
        worker = ArtifactUploadWorker(store=store, journal=self.journal, max_workers=1)

        # recover_from_journal automatically resumed pending task
        worker.start()
        self.addCleanup(worker.stop)
        self.assertTrue(worker.wait_for_completion(timeout=3.0))

        # Verify completed cleanly
        self.assertEqual(len(store.published), 1)
        self.assertEqual(store.published[0].sha256, sha)
        final_rec = self.journal.get_transfer(transfer_id)
        self.assertIsNotNone(final_rec)
        self.assertEqual(final_rec.status, TransferStatus.PUBLISHED.value)
        self.assertIsNotNone(final_rec.remote_file_id)

    def test_scenario_8_resume_and_retrieve_older_checkpoint_revision(self):
        """Scenario 8: Resume / retrieve using older checkpoint revision (Revision A) after Revision B is published: Verify store.retrieve(rev_a_id) returns Revision A and store.retrieve(rev_b_id) returns Revision B."""
        store_dir = Path(self.test_dir) / "store_s8"
        store = MockStoreForWorker(manifest_dir=store_dir)
        worker = ArtifactUploadWorker(store=store, journal=self.journal, max_workers=1)
        worker.start()
        self.addCleanup(worker.stop)

        f = Path(self.test_dir) / "model.pt"

        # Publish Revision A
        rev_a_data = b"MODEL_CHECKPOINT_EPOCH_10"
        f.write_bytes(rev_a_data)
        rec_a = worker.enqueue(f, project_id="p8", run_id="r8", category="checkpoints")
        self.assertTrue(worker.wait_for_completion(timeout=3.0))

        # Publish Revision B
        rev_b_data = b"MODEL_CHECKPOINT_EPOCH_20"
        f.write_bytes(rev_b_data)
        rec_b = worker.enqueue(f, project_id="p8", run_id="r8", category="checkpoints")
        self.assertTrue(worker.wait_for_completion(timeout=3.0))

        # Retrieve older Revision A after Revision B is published
        dest_a = Path(self.test_dir) / "dest_epoch10.pt"
        store.retrieve(rec_a.artifact_id, dest_a)
        self.assertEqual(dest_a.read_bytes(), rev_a_data)

        # Retrieve Revision B
        dest_b = Path(self.test_dir) / "dest_epoch20.pt"
        store.retrieve(rec_b.artifact_id, dest_b)
        self.assertEqual(dest_b.read_bytes(), rev_b_data)

    def test_scenario_9_corrupted_journal_quarantined_and_worker_continues(self):
        """Scenario 9: Corrupted journal JSON on disk: Quarantined to .corrupt.<timestamp>.bak and worker continues safely."""
        corrupt_journal_file = Path(self.test_dir) / "corrupt_transfers.json"
        corrupt_journal_file.write_text('{"bad_key": {"broken_json_syntax...')

        corrupt_journal = TransferJournal(corrupt_journal_file)
        store = MockStoreForWorker()

        # Worker initialization must quarantine the corrupt file and continue safely without crashing!
        worker = ArtifactUploadWorker(store=store, journal=corrupt_journal, max_workers=1)
        worker.start()
        self.addCleanup(worker.stop)

        # Verify a .corrupt.*.bak file was created
        corrupt_baks = list(corrupt_journal_file.parent.glob("*.corrupt.*.bak"))
        self.assertGreaterEqual(len(corrupt_baks), 1, "Expected quarantined .corrupt.<timestamp>.bak file")

        # Worker can now enqueue and process new uploads safely
        f = Path(self.test_dir) / "after_corrupt.pt"
        f.write_bytes(b"DATA_AFTER_CORRUPTION_CLEANUP")
        worker.enqueue(f, project_id="p9", run_id="r9")
        self.assertTrue(worker.wait_for_completion(timeout=3.0))

        self.assertEqual(len(store.published), 1)
        record = corrupt_journal.get_transfer(f"mock_cloud-p9-r9-after_corrupt.pt")
        self.assertIsNotNone(record)
        self.assertEqual(record.status, TransferStatus.PUBLISHED.value)


if __name__ == "__main__":
    unittest.main()
