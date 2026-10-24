"""Unit tests for asynchronous background cloud upload worker and recovery.

Part of Colab V2 Cross-Device Workspaces & Google Drive (Issue #21).
"""
import shutil
import tempfile
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


if __name__ == "__main__":
    unittest.main()
