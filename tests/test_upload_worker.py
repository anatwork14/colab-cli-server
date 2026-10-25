"""Unit tests for asynchronous background cloud upload worker and recovery.

Part of Colab V2 Cross-Device Workspaces & Google Drive (Issue #21).
"""
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from colab_adapter.storage import (
    ArtifactStore,
    ArtifactType,
    ArtifactUploadWorker,
    StoredArtifact,
    TransferJournal,
    TransferRecord,
    TransferStatus,
)


class MockStoreForWorker:
    """Mock store recording calls and simulating successes/failures."""

    def __init__(self, fail_first_n: int = 0, error_message: str = "Simulated transient cloud error"):
        self.published: List[StoredArtifact] = []
        self.fail_first_n = fail_first_n
        self.error_message = error_message
        self.provider_name = "mock_cloud"

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
        stored = StoredArtifact(
            artifact_id=f"{project_id}/{run_id}/{p.name}",
            storage_provider=self.provider_name,
            path=str(p),
            size_bytes=p.stat().st_size,
            sha256=expected_sha256 or "dummy_hash",
            project_id=project_id,
            run_id=run_id,
            artifact_type=artifact_type,
            remote_file_id=f"cloud-id-{p.name}",
        )
        self.published.append(stored)
        return stored

    def retrieve(self, artifact_id: str, destination: Path) -> Path:
        return destination

    def exists(self, artifact_id: str) -> bool:
        return any(a.artifact_id == artifact_id for a in self.published)

    def get_metadata(self, artifact_id: str) -> Optional[StoredArtifact]:
        for a in self.published:
            if a.artifact_id == artifact_id:
                return a
        return None

    def list_artifacts(self, project_id: str, run_id: Optional[str] = None) -> List[StoredArtifact]:
        return list(self.published)


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


if __name__ == "__main__":
    unittest.main()
