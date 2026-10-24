"""Unit tests for provider-independent storage abstraction, manifests, and transfer lifecycle.

Part of Colab V2 Cross-Device Workspaces & Google Drive (Issue #21).
"""
import json
import os
import shutil
import stat
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from colab_adapter.artifacts import ArtifactManager
from colab_adapter.errors import (
    ArtifactNotFoundError,
    ArtifactValidationError,
    InvalidStateTransitionError,
    JournalCorruptionError,
)
from colab_adapter.monitor.collector import AccountUsageCollector
from colab_adapter.storage import (
    ArtifactType,
    LocalArtifactStore,
    PublicationManifest,
    StoredArtifact,
    TransferJournal,
    TransferRecord,
    TransferStatus,
    compute_file_md5,
    compute_file_sha256,
)


class TestStorageAbstraction(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="colab_storage_test_")
        self.addCleanup(lambda: shutil.rmtree(self.test_dir, ignore_errors=True))
        self.artifacts_dir = Path(self.test_dir) / "artifacts"
        self.store = LocalArtifactStore(root_dir=self.artifacts_dir)

    def test_local_store_publish_and_retrieve_success(self):
        """Test publishing a verified local file and retrieving it with SHA256 check."""
        src_file = Path(self.test_dir) / "checkpoint_epoch_1.pt"
        payload = b"FAKE_PYTORCH_CHECKPOINT_DATA_12345"
        src_file.write_bytes(payload)

        stored = self.store.publish(
            source=src_file,
            project_id="test_proj",
            run_id="run_001",
            artifact_type=ArtifactType.CHECKPOINT.value,
        )

        self.assertEqual(stored.storage_provider, "local")
        self.assertEqual(stored.size_bytes, len(payload))
        self.assertEqual(stored.sha256, compute_file_sha256(src_file))
        self.assertTrue(self.store.exists(stored.artifact_id))

        # Retrieve
        dest_file = Path(self.test_dir) / "retrieved_ckpt.pt"
        out_path = self.store.retrieve(stored.artifact_id, dest_file)
        self.assertTrue(out_path.exists())
        self.assertEqual(out_path.read_bytes(), payload)

    def test_local_store_sha256_mismatch_rejected(self):
        """Test publishing with incorrect expected_sha256 raises ArtifactValidationError."""
        src_file = Path(self.test_dir) / "corrupt.pt"
        src_file.write_bytes(b"DATA")

        with self.assertRaises(ArtifactValidationError):
            self.store.publish(
                source=src_file,
                project_id="test_proj",
                run_id="run_001",
                expected_sha256="0000000000000000000000000000000000000000000000000000000000000000",
            )

    def test_local_store_nonexistent_artifact_retrieve_fails(self):
        """Test retrieving missing artifact raises ArtifactNotFoundError."""
        with self.assertRaises(ArtifactNotFoundError):
            self.store.retrieve("nonexistent/run/file.pt", Path(self.test_dir) / "out.pt")

    def test_publication_manifest_atomic_save_and_load(self):
        """Test PublicationManifest saves atomically and loads cleanly."""
        manifest = PublicationManifest(
            project_id="mnist",
            run_id="run_999",
            git_commit="abcdef1234567890",
            git_branch="main",
            git_dirty=False,
            environment={"python": "3.12"},
        )
        artifact = StoredArtifact(
            artifact_id="mnist/run_999/model.pt",
            storage_provider="local",
            path="/tmp/model.pt",
            size_bytes=1024,
            sha256="abc123hash",
            project_id="mnist",
            run_id="run_999",
        )
        manifest.add_artifact(artifact)

        m_path = Path(self.test_dir) / "manifest.json"
        saved_path = manifest.save(m_path)
        self.assertTrue(saved_path.exists())

        loaded = PublicationManifest.load(m_path)
        self.assertEqual(loaded.project_id, "mnist")
        self.assertEqual(loaded.run_id, "run_999")
        self.assertEqual(loaded.git_commit, "abcdef1234567890")
        self.assertEqual(len(loaded.list_artifacts()), 1)
        self.assertEqual(loaded.get_artifact("mnist/run_999/model.pt").size_bytes, 1024)

    def test_transfer_lifecycle_state_transitions(self):
        """Test legal and illegal transitions in TransferRecord."""
        rec = TransferRecord(
            transfer_id="t1",
            project_id="p1",
            run_id="r1",
            artifact_id="p1/r1/model.pt",
            local_path="/tmp/model.pt",
            target_rel_path="checkpoints/model.pt",
            size_bytes=100,
            sha256="sha",
            status=TransferStatus.LOCAL_STAGED.value,
        )

        rec.transition_to(TransferStatus.LOCAL_VERIFIED)
        self.assertEqual(rec.status, TransferStatus.LOCAL_VERIFIED.value)

        rec.transition_to(TransferStatus.UPLOAD_PENDING)
        self.assertEqual(rec.status, TransferStatus.UPLOAD_PENDING.value)

        rec.transition_to(TransferStatus.UPLOADING)
        self.assertEqual(rec.status, TransferStatus.UPLOADING.value)

        rec.transition_to(TransferStatus.DRIVE_VERIFIED)
        self.assertEqual(rec.status, TransferStatus.DRIVE_VERIFIED.value)

        rec.transition_to(TransferStatus.PUBLISHED)
        self.assertEqual(rec.status, TransferStatus.PUBLISHED.value)

        # Illegal transition from PUBLISHED to UPLOADING
        with self.assertRaises(InvalidStateTransitionError):
            rec.transition_to(TransferStatus.UPLOADING)

    def test_transfer_journal_persistence_and_recovery(self):
        """Test persistent journal saves records and recovers pending tasks."""
        journal_file = Path(self.test_dir) / "journal.json"
        journal = TransferJournal(journal_file)

        rec1 = TransferRecord(
            transfer_id="t1",
            project_id="p1",
            run_id="r1",
            artifact_id="a1",
            local_path="/p1",
            target_rel_path="t1",
            size_bytes=10,
            sha256="s1",
            status=TransferStatus.UPLOAD_PENDING.value,
        )
        rec2 = TransferRecord(
            transfer_id="t2",
            project_id="p1",
            run_id="r1",
            artifact_id="a2",
            local_path="/p2",
            target_rel_path="t2",
            size_bytes=20,
            sha256="s2",
            status=TransferStatus.PUBLISHED.value,
        )

        journal.record_transfer(rec1)
        journal.record_transfer(rec2)

        # Create new journal instance from same file
        journal2 = TransferJournal(journal_file)
        self.assertEqual(journal2.get_transfer("t1").status, TransferStatus.UPLOAD_PENDING.value)
        self.assertEqual(journal2.get_transfer("t2").status, TransferStatus.PUBLISHED.value)

        pending = journal2.get_pending_transfers()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].transfer_id, "t1")

    def test_corrupt_journal_recovery_repairs_disk_and_sets_0600_permissions(self):
        """Corrupt journal recovery from .bak automatically repairs transfers.json on disk and sets 0600 on .corrupt archive."""
        journal_file = Path(self.test_dir) / "transfers.json"
        journal = TransferJournal(journal_file)

        rec = TransferRecord(
            transfer_id="rec-valid-1",
            project_id="p1",
            run_id="r1",
            artifact_id="p1/r1/model.pt",
            local_path="/tmp/model.pt",
            target_rel_path="checkpoints/model.pt",
            size_bytes=100,
            sha256="abc",
            status=TransferStatus.LOCAL_VERIFIED.value,
        )
        journal.record_transfer(rec)

        # Corrupt the journal file on disk with truncated JSON
        journal_file.write_text('{"rec-valid-1": {"transfer_id": "broken json truncated')

        # Read should detect corruption, preserve .corrupt copy, and recover from .bak
        recovered = journal.get_transfer("rec-valid-1")
        self.assertIsNotNone(recovered, "Journal failed to recover record from backup!")
        self.assertEqual(recovered.transfer_id, "rec-valid-1")

        # Verify transfers.json on disk has been automatically repaired and contains valid JSON
        repaired_data = json.loads(journal_file.read_text(encoding="utf-8"))
        self.assertIn("rec-valid-1", repaired_data)
        self.assertEqual(stat.S_IMODE(journal_file.stat().st_mode), 0o600)

        # Verify a .corrupt.<timestamp> file was created in parent and has 0600 mode
        corrupt_files = list(journal_file.parent.glob("transfers.json.corrupt.*"))
        self.assertGreaterEqual(len(corrupt_files), 1)
        for cf in corrupt_files:
            self.assertEqual(stat.S_IMODE(cf.stat().st_mode), 0o600)

        # Subsequent reads succeed without re-triggering corruption recovery
        prev_count = len(corrupt_files)
        second_read = journal.get_transfer("rec-valid-1")
        self.assertIsNotNone(second_read)
        self.assertEqual(len(list(journal_file.parent.glob("transfers.json.corrupt.*"))), prev_count)

    def test_journal_read_records_tightens_permissions_to_0600(self):
        """_read_records() checks file permissions and tightens to 0600 if group/other bits are set."""
        journal_file = Path(self.test_dir) / "perm_transfers.json"
        journal = TransferJournal(journal_file)

        rec = TransferRecord(
            transfer_id="rec-perm-1",
            project_id="p1",
            run_id="r1",
            artifact_id="p1/r1/model.pt",
            local_path="/tmp/model.pt",
            target_rel_path="checkpoints/model.pt",
            size_bytes=100,
            sha256="abc",
            status=TransferStatus.LOCAL_VERIFIED.value,
        )
        journal.record_transfer(rec)

        # Deliberately loosen permissions to 0644
        os.chmod(str(journal_file), 0o644)
        self.assertEqual(stat.S_IMODE(journal_file.stat().st_mode), 0o644)

        # Reading from journal should tighten permissions back to 0600
        read_rec = journal.get_transfer("rec-perm-1")
        self.assertIsNotNone(read_rec)
        self.assertEqual(stat.S_IMODE(journal_file.stat().st_mode), 0o600)

    def test_concurrent_multithread_writes_preserve_all_records(self):
        """Concurrent multi-thread writes to TransferJournal preserve all records without race conditions or lost updates."""
        journal_file = Path(self.test_dir) / "concurrent_transfers.json"
        journal = TransferJournal(journal_file)

        num_threads = 8
        records_per_thread = 10
        start_event = threading.Event()
        errors = []

        def worker_fn(tid: int):
            start_event.wait()
            try:
                for i in range(records_per_thread):
                    rec = TransferRecord(
                        transfer_id=f"rec-{tid:02d}-{i:02d}",
                        project_id=f"proj_{tid}",
                        run_id=f"run_{i}",
                        artifact_id=f"proj_{tid}/run_{i}/file_{i}.pt",
                        local_path=f"/tmp/file_{tid}_{i}.pt",
                        target_rel_path=f"file_{tid}_{i}.pt",
                        size_bytes=100 + i,
                        sha256=f"sha_{tid}_{i}",
                        status=TransferStatus.LOCAL_VERIFIED.value,
                    )
                    journal.record_transfer(rec)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker_fn, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        start_event.set()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrent worker errors: {errors}")
        records = journal.list_transfers()
        self.assertEqual(len(records), num_threads * records_per_thread)

        record_ids = {r.transfer_id for r in records}
        for tid in range(num_threads):
            for i in range(records_per_thread):
                self.assertIn(f"rec-{tid:02d}-{i:02d}", record_ids)

    def test_get_pending_transfers_sorted_deterministically(self):
        """get_pending_transfers() returns records sorted deterministically by (rec.created_at, rec.transfer_id)."""
        journal_file = Path(self.test_dir) / "sorted_transfers.json"
        journal = TransferJournal(journal_file)

        r1 = TransferRecord(
            transfer_id="b_id",
            project_id="p",
            run_id="r",
            artifact_id="a",
            local_path="/p",
            target_rel_path="t",
            size_bytes=10,
            sha256="s",
            status=TransferStatus.LOCAL_VERIFIED.value,
            created_at="2026-01-01T12:00:00Z",
        )
        r2 = TransferRecord(
            transfer_id="a_id",
            project_id="p",
            run_id="r",
            artifact_id="a",
            local_path="/p",
            target_rel_path="t",
            size_bytes=10,
            sha256="s",
            status=TransferStatus.LOCAL_VERIFIED.value,
            created_at="2026-01-01T12:00:00Z",
        )
        r3 = TransferRecord(
            transfer_id="c_id",
            project_id="p",
            run_id="r",
            artifact_id="a",
            local_path="/p",
            target_rel_path="t",
            size_bytes=10,
            sha256="s",
            status=TransferStatus.LOCAL_VERIFIED.value,
            created_at="2026-01-01T11:00:00Z",
        )

        # Record in arbitrary order: r1, r2, r3
        journal.record_transfer(r1)
        journal.record_transfer(r2)
        journal.record_transfer(r3)

        pending = journal.get_pending_transfers()
        self.assertEqual(len(pending), 3)
        self.assertEqual(pending[0].transfer_id, "c_id")  # earliest created_at (11:00)
        self.assertEqual(pending[1].transfer_id, "a_id")  # 12:00, tiebreak lexicographically
        self.assertEqual(pending[2].transfer_id, "b_id")  # 12:00

    def test_transfer_record_safe_sanitization_and_transition(self):
        """TransferRecord.transition_to sanitizes errors and to_dict(safe=True) redacts secrets."""
        rec = TransferRecord(
            transfer_id="t_sec",
            project_id="p",
            run_id="r",
            artifact_id="a",
            local_path="/p",
            target_rel_path="t",
            size_bytes=10,
            sha256="s",
            status=TransferStatus.LOCAL_STAGED.value,
            resumable_session_url="https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&session_id=SECRET_SESSION_123",
            metadata={
                "proxy_secret": "sensitive_proxy_val",
                "x_api_token": "sensitive_token_val",
                "run_name": "normal_run",
            },
        )

        # Illegal transition with error message containing secrets
        rec.transition_to(
            TransferStatus.LOCAL_VERIFIED,
            error_message="Failure with client_secret=secret_client_key_123 and access_token=ya29.secret_token",
        )
        self.assertNotIn("secret_client_key_123", rec.last_error)
        self.assertNotIn("secret_token", rec.last_error)
        self.assertIn("[REDACTED_SECRET]", rec.last_error)

        # Safe dictionary representation
        safe_dict = rec.to_dict(safe=True)
        self.assertEqual(safe_dict["resumable_session_url"], "[REDACTED_SESSION_URL]")
        self.assertEqual(safe_dict["metadata"]["proxy_secret"], "[REDACTED]")
        self.assertEqual(safe_dict["metadata"]["x_api_token"], "[REDACTED]")
        self.assertEqual(safe_dict["metadata"]["run_name"], "normal_run")
        self.assertNotIn("SECRET_SESSION_123", str(safe_dict))

    def test_download_and_verify_sanitizes_errors_and_cloud_enqueue_exceptions(self):
        """ArtifactManager.download_and_verify sanitizes errors in exceptions and cloud enqueue failures."""
        # 1. Transport error with seeded credentials
        mock_transport = MagicMock()
        mock_transport.download_file.side_effect = RuntimeError(
            "Transport failure with access_token=ya29.secret_down_tok and client_secret=cs_secret_val"
        )
        manager = ArtifactManager(transport=mock_transport, local_dest_dir=self.artifacts_dir)

        with self.assertRaises(ArtifactValidationError) as ctx:
            manager.download_and_verify(
                session_id="sess-sec-1",
                remote_path="/content/secret_model.pt",
            )

        err_str = str(ctx.exception)
        self.assertNotIn("secret_down_tok", err_str)
        self.assertNotIn("cs_secret_val", err_str)
        self.assertIn("[REDACTED]", err_str)
        self.assertNotIn("secret_down_tok", ctx.exception.details.get("error", ""))

        # 2. Cloud enqueue failure with seeded session URL
        mock_transport_ok = MagicMock()
        payload = b"VALID_LOCAL_ARTIFACT_CONTENT"
        def fake_download(session_id, remote_path, local_path):
            Path(local_path).write_bytes(payload)
        mock_transport_ok.download_file.side_effect = fake_download

        mock_worker = MagicMock()
        mock_worker.enqueue.side_effect = RuntimeError(
            "Cloud enqueue failed: https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&session_id=SECRET_CLOUD_SESSION"
        )
        manager_with_cloud = ArtifactManager(
            transport=mock_transport_ok,
            local_dest_dir=self.artifacts_dir,
            upload_worker=mock_worker,
        )

        manifest = manager_with_cloud.download_and_verify(
            session_id="sess-sec-2",
            remote_path="/content/ckpt.pt",
            enqueue_cloud=True,
            project_id="p1",
            run_id="r1",
        )
        self.assertEqual(manifest.drive_publication_status, "FAILED")
        cloud_err = manifest.metadata.get("cloud_enqueue_error", "")
        self.assertNotIn("SECRET_CLOUD_SESSION", cloud_err)
        self.assertIn("[REDACTED_UPLOAD_SESSION_URL]", cloud_err)

    def test_collector_poll_once_sanitizes_error_messages(self):
        """AccountUsageCollector.poll_once sanitizes error messages from non-zero CLI return and exceptions."""
        ledger_path = Path(self.test_dir) / "collector_ledger.jsonl"

        # 1. Non-zero exit code with stderr containing secrets
        def mock_run_with_secret(cmd, **kwargs):
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="CLI failure: access_token=ya29.poll_access_secret and refresh_token=1//poll_ref_secret",
            )

        collector = AccountUsageCollector(ledger_path=ledger_path, run_cmd_fn=mock_run_with_secret)
        sample = collector.poll_once()
        self.assertEqual(sample.status, "ERROR")
        self.assertIsNotNone(sample.error_message)
        self.assertNotIn("poll_access_secret", sample.error_message)
        self.assertNotIn("poll_ref_secret", sample.error_message)
        self.assertIn("[REDACTED_SECRET]", sample.error_message)

        # 2. Exception in runner with credentials
        def mock_run_exception(cmd, **kwargs):
            raise RuntimeError("Subprocess broken: proxy_secret=my_secret_proxy_val")

        collector2 = AccountUsageCollector(ledger_path=ledger_path, run_cmd_fn=mock_run_exception)
        sample2 = collector2.poll_once()
        self.assertEqual(sample2.status, "ERROR")
        self.assertIsNotNone(sample2.error_message)
        self.assertNotIn("my_secret_proxy_val", sample2.error_message)
        self.assertIn("[REDACTED_SECRET]", sample2.error_message)


if __name__ == "__main__":
    unittest.main()
