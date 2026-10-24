"""Unit tests for provider-independent storage abstraction, manifests, and transfer lifecycle.

Part of Colab V2 Cross-Device Workspaces & Google Drive (Issue #21).
"""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from colab_adapter.errors import (
    ArtifactNotFoundError,
    ArtifactValidationError,
    InvalidStateTransitionError,
)
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


if __name__ == "__main__":
    unittest.main()
