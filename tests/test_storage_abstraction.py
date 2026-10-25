"""Unit tests for provider-independent storage abstraction, manifests, and transfer lifecycle.

Part of Colab V2 Cross-Device Workspaces & Google Drive (Issue #21).
"""
import hashlib
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
    StorageError,
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
from colab_adapter.storage.base import (
    ArtifactRevisionIdentity,
    format_logical_artifact_id,
    format_revision_artifact_id,
    format_transfer_id,
    parse_artifact_id,
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

    def test_artifact_revision_identity_and_helper_functions(self):
        """Test ArtifactRevisionIdentity, format_revision_artifact_id, parse_artifact_id, and format_transfer_id."""
        # 1. format_logical_artifact_id
        self.assertEqual(
            format_logical_artifact_id("mnist", "run_1", "checkpoints/model.pt"),
            "mnist/run_1/checkpoints/model.pt",
        )
        self.assertEqual(
            format_logical_artifact_id("mnist", "run_1", "/checkpoints/model.pt"),
            "mnist/run_1/checkpoints/model.pt",
        )
        self.assertEqual(
            format_logical_artifact_id("mnist", "run_1", "mnist/run_1/checkpoints/model.pt"),
            "mnist/run_1/checkpoints/model.pt",
        )
        self.assertEqual(
            format_logical_artifact_id("mnist", "run_1", "checkpoints/model.pt@sha256_deadbeef"),
            "mnist/run_1/checkpoints/model.pt",
        )

        # 2. format_revision_artifact_id
        self.assertEqual(
            format_revision_artifact_id("mnist", "run_1", "checkpoints/model.pt", "ABCDEF1234567890"),
            "mnist/run_1/checkpoints/model.pt@sha256_abcdef1234567890",
        )
        self.assertEqual(
            format_revision_artifact_id("mnist", "run_1", "/checkpoints/model.pt@sha256_oldhash", " 112233AABB "),
            "mnist/run_1/checkpoints/model.pt@sha256_112233aabb",
        )
        self.assertEqual(
            format_revision_artifact_id("mnist", "run_1", "mnist/run_1/checkpoints/model.pt", "112233AABB"),
            "mnist/run_1/checkpoints/model.pt@sha256_112233aabb",
        )

        # 3. parse_artifact_id
        # Versioned full path
        p, r, lp, sha = parse_artifact_id("mnist/run_1/checkpoints/model.pt@sha256_abcdef1234")
        self.assertEqual(p, "mnist")
        self.assertEqual(r, "run_1")
        self.assertEqual(lp, "checkpoints/model.pt")
        self.assertEqual(sha, "abcdef1234")

        # Unversioned full path
        p, r, lp, sha = parse_artifact_id("mnist/run_1/checkpoints/model.pt")
        self.assertEqual(p, "mnist")
        self.assertEqual(r, "run_1")
        self.assertEqual(lp, "checkpoints/model.pt")
        self.assertIsNone(sha)

        # Single or 2-segment path (fewer than 3 segments)
        p, r, lp, sha = parse_artifact_id("checkpoints/model.pt@sha256_abcdef1234")
        self.assertEqual(p, "")
        self.assertEqual(r, "")
        self.assertEqual(lp, "checkpoints/model.pt")
        self.assertEqual(sha, "abcdef1234")

        p, r, lp, sha = parse_artifact_id("model.pt")
        self.assertEqual(p, "")
        self.assertEqual(r, "")
        self.assertEqual(lp, "model.pt")
        self.assertIsNone(sha)

        # 4. format_transfer_id
        t_id = format_transfer_id("google_drive", "proj_1", "run_1", "checkpoints/model.pt", "0123456789abcdef9999")
        self.assertEqual(t_id, "google_drive-proj_1-run_1-checkpoints_model.pt-0123456789abcdef")

        # 5. ArtifactRevisionIdentity dataclass
        ident = ArtifactRevisionIdentity(
            project_id="proj_x",
            run_id="run_y",
            logical_path="checkpoints/weights.pt",
            sha256="CAFE1234BEEF5678",
        )
        self.assertEqual(ident.logical_id, "proj_x/run_y/checkpoints/weights.pt")
        self.assertEqual(ident.revision_id, "sha256_cafe1234beef5678")
        self.assertEqual(ident.artifact_id, "proj_x/run_y/checkpoints/weights.pt@sha256_cafe1234beef5678")

    def test_publish_and_retrieve_multiple_revisions_immutable_history(self):
        """Publish two revisions of checkpoints/model.pt (Revision A & Revision B) to LocalArtifactStore.

        Verifies:
        - Both revisions exist in PublicationManifest and list_artifacts() / list_revisions('checkpoints/model.pt').
        - Revision A has is_latest=False and Revision B has is_latest=True.
        - Retrieving by rev_a.artifact_id returns exact Revision A bytes and SHA256.
        - Retrieving by rev_b.artifact_id returns exact Revision B bytes and SHA256.
        - Retrieving by unversioned 'proj_1/run_1/checkpoints/model.pt' returns Revision B (latest) bytes.
        - Disk files for earlier revisions are never clobbered or destroyed.
        """
        # 1. Publish Revision A
        src_a = Path(self.test_dir) / "source_v1" / "model.pt"
        src_a.parent.mkdir(parents=True, exist_ok=True)
        payload_a = b"PYTORCH_CHECKPOINT_REVISION_A_EPOCH_10"
        src_a.write_bytes(payload_a)

        rev_a = self.store.publish(
            source=src_a,
            project_id="proj_1",
            run_id="run_1",
            artifact_type=ArtifactType.CHECKPOINT.value,
            metadata={"epoch": 10, "val_loss": 0.45},
        )
        self.assertTrue(rev_a.is_latest)
        self.assertEqual(rev_a.logical_path, "checkpoints/model.pt")
        sha_a = compute_file_sha256(src_a)
        self.assertEqual(rev_a.sha256, sha_a)
        self.assertEqual(rev_a.artifact_id, f"proj_1/run_1/checkpoints/model.pt@sha256_{sha_a}")

        # 2. Publish Revision B (different content to same logical checkpoint target)
        src_b = Path(self.test_dir) / "source_v2" / "model.pt"
        src_b.parent.mkdir(parents=True, exist_ok=True)
        payload_b = b"PYTORCH_CHECKPOINT_REVISION_B_EPOCH_20_UPDATED_LARGER_WEIGHTS"
        src_b.write_bytes(payload_b)

        rev_b = self.store.publish(
            source=src_b,
            project_id="proj_1",
            run_id="run_1",
            artifact_type=ArtifactType.CHECKPOINT.value,
            metadata={"epoch": 20, "val_loss": 0.22},
        )
        self.assertTrue(rev_b.is_latest)
        self.assertEqual(rev_b.logical_path, "checkpoints/model.pt")
        sha_b = compute_file_sha256(src_b)
        self.assertEqual(rev_b.sha256, sha_b)
        self.assertEqual(rev_b.artifact_id, f"proj_1/run_1/checkpoints/model.pt@sha256_{sha_b}")
        self.assertNotEqual(rev_a.artifact_id, rev_b.artifact_id)

        # 3. Check PublicationManifest state
        manifest_path = self.artifacts_dir / "projects" / "proj_1" / "runs" / "run_1" / "manifest.json"
        manifest = PublicationManifest.load(manifest_path)
        self.assertEqual(manifest.manifest_version, 2)

        # Both revisions exist in manifest
        all_artifacts = manifest.list_artifacts(include_history=True)
        self.assertEqual(len(all_artifacts), 2)

        # List revisions for "checkpoints/model.pt"
        revisions = manifest.list_revisions("checkpoints/model.pt")
        self.assertEqual(len(revisions), 2)
        self.assertEqual(revisions[0].artifact_id, rev_a.artifact_id)
        self.assertEqual(revisions[1].artifact_id, rev_b.artifact_id)

        # Revision A has is_latest=False and Revision B has is_latest=True
        entry_a = manifest.get_artifact(rev_a.artifact_id)
        entry_b = manifest.get_artifact(rev_b.artifact_id)
        self.assertIsNotNone(entry_a)
        self.assertIsNotNone(entry_b)
        self.assertFalse(entry_a.is_latest)
        self.assertTrue(entry_b.is_latest)
        self.assertEqual(manifest.latest_by_path.get("checkpoints/model.pt"), rev_b.artifact_id)

        # list_artifacts(include_history=False) returns only Revision B
        latest_only = manifest.list_artifacts(include_history=False)
        self.assertEqual(len(latest_only), 1)
        self.assertEqual(latest_only[0].artifact_id, rev_b.artifact_id)

        # 4. Verify physical immutability: both files exist on disk and were NOT overwritten
        self.assertTrue(Path(entry_a.path).exists(), "Revision A disk file was clobbered!")
        self.assertTrue(Path(entry_b.path).exists(), "Revision B disk file does not exist!")
        self.assertNotEqual(entry_a.path, entry_b.path)
        self.assertEqual(Path(entry_a.path).read_bytes(), payload_a)
        self.assertEqual(Path(entry_b.path).read_bytes(), payload_b)

        # Unversioned disk path points to latest Revision B
        unversioned_disk = self.artifacts_dir / "projects" / "proj_1" / "runs" / "run_1" / "checkpoints" / "model.pt"
        self.assertTrue(unversioned_disk.exists())
        self.assertEqual(unversioned_disk.read_bytes(), payload_b)

        # 5. Retrieve by rev_a.artifact_id returns exact Revision A bytes and SHA256
        dest_a = Path(self.test_dir) / "retrieved_v1.pt"
        out_a = self.store.retrieve(rev_a.artifact_id, dest_a)
        self.assertTrue(out_a.exists())
        self.assertEqual(out_a.read_bytes(), payload_a)
        self.assertEqual(compute_file_sha256(out_a), sha_a)

        # 6. Retrieve by rev_b.artifact_id returns exact Revision B bytes and SHA256
        dest_b = Path(self.test_dir) / "retrieved_v2.pt"
        out_b = self.store.retrieve(rev_b.artifact_id, dest_b)
        self.assertTrue(out_b.exists())
        self.assertEqual(out_b.read_bytes(), payload_b)
        self.assertEqual(compute_file_sha256(out_b), sha_b)

        # 7. Retrieve by unversioned "proj_1/run_1/checkpoints/model.pt" returns Revision B (latest) bytes
        dest_latest = Path(self.test_dir) / "retrieved_latest.pt"
        out_latest = self.store.retrieve("proj_1/run_1/checkpoints/model.pt", dest_latest)
        self.assertTrue(out_latest.exists())
        self.assertEqual(out_latest.read_bytes(), payload_b)
        self.assertEqual(compute_file_sha256(out_latest), sha_b)

        # 8. Store get_metadata and exists for both versioned and unversioned
        self.assertTrue(self.store.exists(rev_a.artifact_id))
        self.assertTrue(self.store.exists(rev_b.artifact_id))
        self.assertTrue(self.store.exists("proj_1/run_1/checkpoints/model.pt"))
        meta_unversioned = self.store.get_metadata("proj_1/run_1/checkpoints/model.pt")
        self.assertEqual(meta_unversioned.artifact_id, rev_b.artifact_id)

    def test_conflicting_content_for_immutable_revision_id_raises(self):
        """Conflicting content (size or sha256 mismatch) for the same revision ID raises ArtifactValidationError."""
        manifest = PublicationManifest(project_id="test_proj", run_id="test_run")
        sha_const = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        valid_rev_id = f"test_proj/test_run/checkpoints/model.pt@sha256_{sha_const}"

        initial = StoredArtifact(
            artifact_id=valid_rev_id,
            storage_provider="local",
            path="/tmp/path_1",
            size_bytes=1000,
            sha256=sha_const,
            project_id="test_proj",
            run_id="test_run",
            logical_path="checkpoints/model.pt",
        )
        manifest.add_artifact(initial)

        # 1. Attempting to add artifact with same rev_id but different size_bytes must raise ArtifactValidationError
        conflicting_size = StoredArtifact(
            artifact_id=valid_rev_id,
            storage_provider="local",
            path="/tmp/path_2",
            size_bytes=2000,
            sha256=sha_const,
            project_id="test_proj",
            run_id="test_run",
            logical_path="checkpoints/model.pt",
        )
        with self.assertRaises(ArtifactValidationError):
            manifest.add_artifact(conflicting_size)

        # 2. Attempting to add artifact with same rev_id but different sha256 must raise ArtifactValidationError
        conflicting_sha = StoredArtifact(
            artifact_id=valid_rev_id,
            storage_provider="local",
            path="/tmp/path_3",
            size_bytes=1000,
            sha256="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            project_id="test_proj",
            run_id="test_run",
            logical_path="checkpoints/model.pt",
        )
        with self.assertRaises(ArtifactValidationError):
            manifest.add_artifact(conflicting_sha)

        # 3. Explicitly versioned artifact_id whose @sha256_ suffix does not match sha256 field raises ArtifactValidationError
        mismatched_id_and_hash = StoredArtifact(
            artifact_id="test_proj/test_run/checkpoints/model.pt@sha256_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            storage_provider="local",
            path="/tmp/path_4",
            size_bytes=500,
            sha256="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            project_id="test_proj",
            run_id="test_run",
        )
        with self.assertRaises(ArtifactValidationError):
            manifest.add_artifact(mismatched_id_and_hash)

    def test_automatic_v1_to_v2_manifest_migration(self):
        """Automatic v1 -> v2 manifest migration upgrades manifest_version, populates latest_by_path, and adds notes."""
        v1_manifest_data = {
            "project_id": "legacy_proj",
            "run_id": "legacy_run",
            "manifest_version": 1,
            "created_at": "2026-01-01T10:00:00Z",
            "updated_at": "2026-01-01T10:05:00Z",
            "status": "PUBLISHED",
            "git_commit": "git_commit_hash_12345",
            "git_branch": "main",
            "git_dirty": False,
            "environment": {"python": "3.10"},
            "artifacts": {
                "legacy_proj/legacy_run/checkpoints/checkpoint.pt": {
                    "artifact_id": "legacy_proj/legacy_run/checkpoints/checkpoint.pt",
                    "storage_provider": "local",
                    "path": "/storage/projects/legacy_proj/runs/legacy_run/checkpoints/checkpoint.pt",
                    "size_bytes": 524288,
                    "sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
                    "project_id": "legacy_proj",
                    "run_id": "legacy_run",
                    "artifact_type": "checkpoint",
                    "remote_file_id": "drive-file-legacy-999",
                    "metadata": {"step": 5000},
                }
            },
            "metadata": {"legacy_meta_key": "legacy_val"},
        }

        m_path = Path(self.test_dir) / "v1_manifest.json"
        m_path.write_text(json.dumps(v1_manifest_data, indent=2), encoding="utf-8")

        # Load legacy manifest via PublicationManifest.load()
        loaded = PublicationManifest.load(m_path)

        # 1. Manifest version upgraded to 2
        self.assertEqual(loaded.manifest_version, 2)

        # 2. Canonical revision ID generated as key in artifacts
        expected_rev_id = (
            "legacy_proj/legacy_run/checkpoints/checkpoint.pt@sha256_abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
        )
        self.assertIn(expected_rev_id, loaded.artifacts)

        # 3. latest_by_path populated
        self.assertEqual(loaded.latest_by_path.get("checkpoints/checkpoint.pt"), expected_rev_id)

        # 4. Migration notes recorded
        self.assertGreaterEqual(len(loaded.migration_notes), 1)
        self.assertIn("Migrated v1 unversioned artifact records", loaded.migration_notes[0])

        # 5. Provenance preserved
        self.assertEqual(loaded.git_commit, "git_commit_hash_12345")
        self.assertEqual(loaded.environment["python"], "3.10")
        self.assertEqual(loaded.metadata["legacy_meta_key"], "legacy_val")

        # 6. Lookup by new immutable revision ID
        by_rev = loaded.get_artifact(expected_rev_id)
        self.assertIsNotNone(by_rev)
        self.assertEqual(by_rev.remote_file_id, "drive-file-legacy-999")
        self.assertEqual(by_rev.metadata["step"], 5000)
        self.assertEqual(by_rev.logical_path, "checkpoints/checkpoint.pt")
        self.assertEqual(by_rev.revision_id, "sha256_abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789")
        self.assertTrue(by_rev.is_latest)

        # 7. Lookup by legacy unversioned full ID
        by_legacy = loaded.get_artifact("legacy_proj/legacy_run/checkpoints/checkpoint.pt")
        self.assertIsNotNone(by_legacy)
        self.assertEqual(by_legacy.artifact_id, expected_rev_id)

        # 8. Lookup by logical path
        by_logical = loaded.get_artifact("checkpoints/checkpoint.pt")
        self.assertIsNotNone(by_logical)
        self.assertEqual(by_logical.artifact_id, expected_rev_id)

        # 9. get_latest_artifact
        latest = loaded.get_latest_artifact("checkpoints/checkpoint.pt")
        self.assertIsNotNone(latest)
        self.assertEqual(latest.artifact_id, expected_rev_id)

        # 10. list_revisions and list_artifacts
        revs = loaded.list_revisions("checkpoints/checkpoint.pt")
        self.assertEqual(len(revs), 1)
        self.assertEqual(revs[0].artifact_id, expected_rev_id)

        arts = loaded.list_artifacts()
        self.assertEqual(len(arts), 1)
        self.assertEqual(arts[0].artifact_id, expected_rev_id)

    def test_local_store_retrieve_tampered_size_mismatch_and_dest_preservation(self):
        """Section 6: Tampered artifact size mismatch raises ArtifactValidationError with details, cleans up part file, and preserves dest."""
        src = Path(self.test_dir) / "source_weights.bin"
        payload = b"ORIGINAL_100_BYTES_OF_DATA_" + b"0" * 73
        self.assertEqual(len(payload), 100)
        src.write_bytes(payload)

        stored = self.store.publish(
            source=src,
            project_id="tamper_proj",
            run_id="run_tamper",
            artifact_type=ArtifactType.CHECKPOINT.value,
        )

        dest = Path(self.test_dir) / "output_dest" / "weights.bin"
        dest.parent.mkdir(parents=True, exist_ok=True)
        pre_existing_data = b"PRE_EXISTING_VALID_DEST_CONTENT_SHOULD_REMAIN_UNTOUCHED"
        dest.write_bytes(pre_existing_data)

        stored_file = Path(stored.path)
        self.assertTrue(stored_file.exists())

        # 1. Tamper to be SMALLER than recorded size (50 bytes vs 100 bytes)
        stored_file.write_bytes(b"S" * 50)
        with self.assertRaises(ArtifactValidationError) as ctx_smaller:
            self.store.retrieve(stored.artifact_id, dest)

        err_smaller = ctx_smaller.exception
        self.assertEqual(err_smaller.details.get("recorded_size"), 100)
        self.assertEqual(err_smaller.details.get("actual_size"), 50)
        self.assertEqual(err_smaller.details.get("expected_size"), 100)
        self.assertEqual(err_smaller.details.get("artifact_id"), stored.artifact_id)
        self.assertIn("100", str(err_smaller))
        self.assertIn("50", str(err_smaller))
        # Verify pre-existing valid file at dest remains completely untouched
        self.assertEqual(dest.read_bytes(), pre_existing_data)
        # Verify .part. temporary file was cleanly unlinked
        part_files = list(dest.parent.glob("*.part.*"))
        self.assertEqual(len(part_files), 0, f"Leaked part files: {part_files}")

        # 2. Tamper to be LARGER than recorded size (200 bytes vs 100 bytes)
        stored_file.write_bytes(b"L" * 200)
        with self.assertRaises(ArtifactValidationError) as ctx_larger:
            self.store.retrieve(stored.artifact_id, dest)

        err_larger = ctx_larger.exception
        self.assertEqual(err_larger.details.get("recorded_size"), 100)
        self.assertEqual(err_larger.details.get("actual_size"), 200)
        self.assertEqual(err_larger.details.get("expected_size"), 100)
        self.assertEqual(err_larger.details.get("artifact_id"), stored.artifact_id)
        self.assertIn("100", str(err_larger))
        self.assertIn("200", str(err_larger))
        self.assertEqual(dest.read_bytes(), pre_existing_data)
        self.assertEqual(len(list(dest.parent.glob("*.part.*"))), 0)

        # 3. Tamper with same size (100 bytes) but DIFFERENT SHA256
        stored_file.write_bytes(b"D" * 100)
        with self.assertRaises(ArtifactValidationError) as ctx_hash:
            self.store.retrieve(stored.artifact_id, dest)

        err_hash = ctx_hash.exception
        self.assertEqual(err_hash.details.get("artifact_id"), stored.artifact_id)
        self.assertEqual(dest.read_bytes(), pre_existing_data)
        self.assertEqual(len(list(dest.parent.glob("*.part.*"))), 0)

    def test_local_store_private_permissions_under_zero_umask(self):
        """Section 6: Private local permissions under os.umask(0o000) enforce 0700 on dirs and 0600 on files."""
        old_umask = os.umask(0o000)
        try:
            src = Path(self.test_dir) / "secure_weights.pt"
            src.write_bytes(b"TOP_SECRET_MODEL_PARAMETERS")

            stored = self.store.publish(
                source=src,
                project_id="priv_proj",
                run_id="priv_run",
                artifact_type=ArtifactType.CHECKPOINT.value,
            )

            stored_file = Path(stored.path)
            rev_dir = stored_file.parent
            revs_base = rev_dir.parent
            latest_file = revs_base.parent / stored_file.name

            # .revisions/<sha16> is 0700
            self.assertEqual(stat.S_IMODE(rev_dir.stat().st_mode), 0o700)
            # .revisions base is 0700
            self.assertEqual(stat.S_IMODE(revs_base.stat().st_mode), 0o700)
            # Stored revision file is 0600
            self.assertEqual(stat.S_IMODE(stored_file.stat().st_mode), 0o600)
            # Latest dest file is 0600
            self.assertEqual(stat.S_IMODE(latest_file.stat().st_mode), 0o600)

            # Manifest and backup file are 0600
            manifest_path = self.artifacts_dir / "projects" / "priv_proj" / "runs" / "priv_run" / "manifest.json"
            manifest_bak = Path(str(manifest_path) + ".bak")
            self.assertTrue(manifest_path.exists())
            self.assertTrue(manifest_bak.exists())
            self.assertEqual(stat.S_IMODE(manifest_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(manifest_bak.stat().st_mode), 0o600)

            # Retrieval part_file and dest also enforce 0600
            dest_out = Path(self.test_dir) / "priv_retrieved.pt"
            self.store.retrieve(stored.artifact_id, dest_out)
            self.assertEqual(stat.S_IMODE(dest_out.stat().st_mode), 0o600)
        finally:
            os.umask(old_umask)

    def test_local_store_symlink_and_non_regular_file_rejection(self):
        """Section 6: LocalArtifactStore rejects symlinks and non-regular files during publish and retrieve."""
        real_file = Path(self.test_dir) / "real_file.pt"
        real_file.write_bytes(b"REAL_CONTENT")

        # Symlink source to publish raises ArtifactValidationError
        symlink_src = Path(self.test_dir) / "symlink_src.pt"
        symlink_src.symlink_to(real_file)
        with self.assertRaises(ArtifactValidationError):
            self.store.publish(symlink_src, "p_link", "r_link")

        # Directory source raises ArtifactValidationError
        with self.assertRaises(ArtifactValidationError):
            self.store.publish(self.test_dir, "p_link", "r_link")

        # Publish real file successfully
        stored = self.store.publish(real_file, "p_link", "r_link")

        # Symlink destination in retrieve raises ArtifactValidationError
        symlink_dest = Path(self.test_dir) / "symlink_dest.pt"
        target_dest = Path(self.test_dir) / "target_dest.pt"
        symlink_dest.symlink_to(target_dest)
        with self.assertRaises(ArtifactValidationError):
            self.store.retrieve(stored.artifact_id, symlink_dest)

    def test_section_8_vision_research_5_revisions_history_and_migration_recovery(self):
        """Section 8: 5-revision history A..E under vision-research / experiment-001 with migration, idempotence, and backup recovery."""
        project_id = "vision-research"
        run_id = "experiment-001"

        # 5 distinct payloads for checkpoints/model.pt
        payloads = [
            b"VISION_RESEARCH_WEIGHTS_EPOCH_10_SHA_A",
            b"VISION_RESEARCH_WEIGHTS_EPOCH_20_SHA_B_LARGER_BATCH",
            b"VISION_RESEARCH_WEIGHTS_EPOCH_30_SHA_C_LEARNING_RATE_DECAY",
            b"VISION_RESEARCH_WEIGHTS_EPOCH_40_SHA_D_STRUCTURED_PRUNING",
            b"VISION_RESEARCH_WEIGHTS_EPOCH_50_SHA_E_FINAL_EXPORT",
        ]

        stored_revisions = []
        for i, payload in enumerate(payloads):
            stage_dir = Path(self.test_dir) / f"stage_{i}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            src_file = stage_dir / "model.pt"
            src_file.write_bytes(payload)
            stored = self.store.publish(
                source=src_file,
                project_id=project_id,
                run_id=run_id,
                artifact_type=ArtifactType.CHECKPOINT.value,
                metadata={"epoch": (i + 1) * 10, "version_letter": chr(ord("A") + i)},
            )
            stored_revisions.append(stored)

        rev_a, rev_b, rev_c, rev_d, rev_e = stored_revisions
        hash_e = rev_e.sha256.lower()

        # 1. Verify 5 distinct immutable revisions in PublicationManifest v2
        manifest_path = self.artifacts_dir / "projects" / project_id / "runs" / run_id / "manifest.json"
        manifest = PublicationManifest.load(manifest_path)
        self.assertEqual(manifest.manifest_version, 2)
        self.assertEqual(len(manifest.artifacts), 5)

        # 2. Verify 5 independently addressable revision IDs
        expected_ids = [
            f"vision-research/experiment-001/checkpoints/model.pt@sha256_{r.sha256.lower()}"
            for r in stored_revisions
        ]
        for rev, exp_id in zip(stored_revisions, expected_ids):
            self.assertEqual(rev.artifact_id, exp_id)
            self.assertIn(exp_id, manifest.artifacts)

        self.assertEqual(len(set(expected_ids)), 5)

        # 3. Complete manifest history (is_latest=False and superseded_by on A..D, is_latest=True on E)
        rev_a_obj = manifest.get_artifact(rev_a.artifact_id)
        rev_b_obj = manifest.get_artifact(rev_b.artifact_id)
        rev_c_obj = manifest.get_artifact(rev_c.artifact_id)
        rev_d_obj = manifest.get_artifact(rev_d.artifact_id)

        self.assertEqual(rev_a_obj.superseded_by, rev_b.artifact_id)
        self.assertEqual(rev_b_obj.superseded_by, rev_c.artifact_id)
        self.assertEqual(rev_c_obj.superseded_by, rev_d.artifact_id)
        self.assertEqual(rev_d_obj.superseded_by, rev_e.artifact_id)
        self.assertNotEqual(rev_a_obj.superseded_by, rev_e.artifact_id)

        for rev in [rev_a, rev_b, rev_c, rev_d]:
            art_dict = manifest.artifacts[rev.artifact_id]
            self.assertFalse(art_dict["is_latest"])
            self.assertTrue(art_dict.get("superseded_by"))
            art_obj = manifest.get_artifact(rev.artifact_id)
            self.assertFalse(art_obj.is_latest)
            self.assertTrue(art_obj.superseded_by)
            self.assertTrue(art_obj.metadata.get("superseded_by"))
            self.assertIs(type(art_obj.superseded_by), str)

        # Revision E is latest
        art_e_dict = manifest.artifacts[rev_e.artifact_id]
        self.assertTrue(art_e_dict["is_latest"])
        self.assertIsNone(art_e_dict.get("superseded_by"))
        art_e_obj = manifest.get_artifact(rev_e.artifact_id)
        self.assertTrue(art_e_obj.is_latest)
        self.assertIsNone(art_e_obj.superseded_by)

        # latest_by_path["checkpoints/model.pt"] is canonical str rev_e.artifact_id
        self.assertIs(type(manifest.latest_by_path["checkpoints/model.pt"]), str)
        self.assertEqual(manifest.latest_by_path["checkpoints/model.pt"], rev_e.artifact_id)
        self.assertEqual(manifest.get_latest_revision_id("checkpoints/model.pt"), f"sha256_{hash_e}")
        self.assertEqual(manifest.get_latest_sha256("checkpoints/model.pt"), hash_e)
        latest_art = manifest.get_latest_artifact("checkpoints/model.pt")
        self.assertIsNotNone(latest_art)
        self.assertEqual(latest_art.artifact_id, rev_e.artifact_id)

        # 4. No history loss after reloading store from disk (simulated restart)
        reloaded_store = LocalArtifactStore(root_dir=self.artifacts_dir)
        reloaded_manifest = reloaded_store._load_manifest(project_id, run_id)
        self.assertEqual(len(reloaded_manifest.artifacts), 5)
        self.assertEqual(reloaded_manifest.manifest_version, 2)
        self.assertIs(type(reloaded_manifest.latest_by_path["checkpoints/model.pt"]), str)
        self.assertEqual(reloaded_manifest.latest_by_path["checkpoints/model.pt"], rev_e.artifact_id)
        self.assertEqual(reloaded_manifest.get_latest_revision_id("checkpoints/model.pt"), f"sha256_{hash_e}")
        self.assertEqual(reloaded_manifest.get_latest_sha256("checkpoints/model.pt"), hash_e)

        all_history = reloaded_store.list_artifacts(project_id, run_id, include_history=True)
        self.assertEqual(len(all_history), 5)
        latest_only = reloaded_store.list_artifacts(project_id, run_id, include_history=False)
        self.assertEqual(len(latest_only), 1)
        self.assertEqual(latest_only[0].artifact_id, rev_e.artifact_id)

        # 5. All 5 revisions (A, B, C, D, E) are independently retrievable via store.retrieve(rev_id, dest) and match their original bytes and SHA256
        for i, (rev, expected_payload) in enumerate(zip(stored_revisions, payloads)):
            dest = Path(self.test_dir) / "retrievals" / f"retrieved_rev_{i}.pt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            out = reloaded_store.retrieve(rev.artifact_id, dest)
            self.assertTrue(out.exists())
            self.assertEqual(out.read_bytes(), expected_payload)
            self.assertEqual(compute_file_sha256(out), rev.sha256)

        # Unversioned logical path retrieval returns Revision E
        dest_latest = Path(self.test_dir) / "retrievals" / "retrieved_latest_model.pt"
        out_latest = reloaded_store.retrieve(f"{project_id}/{run_id}/checkpoints/model.pt", dest_latest)
        self.assertTrue(out_latest.exists())
        self.assertEqual(out_latest.read_bytes(), payloads[4])
        self.assertEqual(compute_file_sha256(out_latest), rev_e.sha256)

        # 6. Repeating publication of version E (SHA256 E) is idempotent and does NOT create a 6th duplicate entry
        repeat_src = Path(self.test_dir) / "stage_repeat_e" / "model.pt"
        repeat_src.parent.mkdir(parents=True, exist_ok=True)
        repeat_src.write_bytes(payloads[4])
        repeat_stored = reloaded_store.publish(
            source=repeat_src,
            project_id=project_id,
            run_id=run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
            metadata={"epoch": 50, "note": "repeated publication"},
        )
        self.assertEqual(repeat_stored.artifact_id, rev_e.artifact_id)
        manifest_after_repeat = reloaded_store._load_manifest(project_id, run_id)
        self.assertEqual(len(manifest_after_repeat.artifacts), 5)
        self.assertEqual(len(manifest_after_repeat.list_revisions("checkpoints/model.pt")), 5)

        # 7. Manifest v1-to-v2 migration test: load a legacy v1 manifest JSON, verify it migrates to v2 cleanly without inventing unrecorded historical revisions or fake step numbers
        v1_manifest = {
            "project_id": "legacy_vision",
            "run_id": "legacy_run_01",
            "manifest_version": 1,
            "artifacts": {
                "legacy_vision/legacy_run_01/checkpoints/model.pt": {
                    "artifact_id": "legacy_vision/legacy_run_01/checkpoints/model.pt",
                    "storage_provider": "local",
                    "path": "/some/path/model.pt",
                    "size_bytes": 1024,
                    "sha256": "aaaa0000bbbb1111cccc2222dddd3333aaaa0000bbbb1111cccc2222dddd3333",
                    "project_id": "legacy_vision",
                    "run_id": "legacy_run_01",
                    "artifact_type": "checkpoint",
                    "metadata": {"recorded_param": "val"},
                }
            },
        }
        v1_path = Path(self.test_dir) / "legacy_v1_manifest.json"
        v1_path.write_text(json.dumps(v1_manifest, indent=2), encoding="utf-8")
        migrated = PublicationManifest.load(v1_path)
        self.assertEqual(migrated.manifest_version, 2)
        # Verify it migrates cleanly to exactly 1 revision without inventing unrecorded historical revisions
        self.assertEqual(len(migrated.artifacts), 1)
        # Verify it preserves metadata without inventing fake step numbers
        migrated_art = list(migrated.artifacts.values())[0]
        self.assertNotIn("step", migrated_art.get("metadata", {}))
        self.assertEqual(migrated_art["metadata"]["recorded_param"], "val")
        self.assertEqual(
            migrated.latest_by_path["checkpoints/model.pt"],
            "legacy_vision/legacy_run_01/checkpoints/model.pt@sha256_aaaa0000bbbb1111cccc2222dddd3333aaaa0000bbbb1111cccc2222dddd3333",
        )
        self.assertGreaterEqual(len(migrated.migration_notes), 1)

        # 8. Corrupted manifest.json with valid manifest.json.bak recovers all 5 revisions cleanly
        bak_path = Path(str(manifest_path) + ".bak")
        self.assertTrue(manifest_path.exists())
        self.assertTrue(bak_path.exists())

        # Corrupt primary manifest.json with malformed JSON
        manifest_path.write_text("{\"project_id\": \"vision-research\", broken json [[")

        # allow_backup_recovery=False strictly raises StorageError
        with self.assertRaises(StorageError):
            PublicationManifest.load(manifest_path, allow_backup_recovery=False)

        # allow_backup_recovery=True recovers all 5 revisions cleanly from .bak
        recovered = PublicationManifest.load(manifest_path, allow_backup_recovery=True)
        self.assertEqual(recovered.manifest_version, 2)
        self.assertEqual(len(recovered.artifacts), 5)
        for exp_id in expected_ids:
            self.assertIn(exp_id, recovered.artifacts)
        self.assertIs(type(recovered.latest_by_path["checkpoints/model.pt"]), str)
        self.assertEqual(recovered.latest_by_path["checkpoints/model.pt"], rev_e.artifact_id)
        self.assertEqual(recovered.get_latest_revision_id("checkpoints/model.pt"), f"sha256_{hash_e}")
        self.assertEqual(recovered.get_latest_sha256("checkpoints/model.pt"), hash_e)
        self.assertTrue(any("recovered" in note.lower() for note in recovered.migration_notes))

        # Also verify LocalArtifactStore._load_manifest recovers when manifest.json is corrupted
        store_recovered_manifest = reloaded_store._load_manifest(project_id, run_id)
        self.assertEqual(len(store_recovered_manifest.artifacts), 5)

        # Also verify LocalArtifactStore._load_manifest recovers when manifest.json is deleted
        manifest_path.unlink()
        self.assertFalse(manifest_path.exists())
        self.assertTrue(bak_path.exists())
        store_deleted_recovered = reloaded_store._load_manifest(project_id, run_id)
        self.assertEqual(len(store_deleted_recovered.artifacts), 5)

    def test_canonical_revision_identity_and_chain_semantics(self):
        """Verify Finding F1 fix:
        1. Removal of RevisionRef; plain immutable Python str used everywhere.
        2. Strict immediate-successor superseded_by chain semantics (A->B->C and A->B->C->D->E).
        3. A.superseded_by is never conflated with the latest revision tip E.
        4. Python equality, symmetry, transitivity, and hash() invariants.
        5. Set membership and dict key lookup integrity.
        6. Manifest save/load and to_dict/from_dict round-trip preservation.
        7. Explicit accessor methods get_latest_artifact, get_latest_revision_id, get_latest_sha256.
        """
        project_id = "test_f1_proj"
        run_id = "test_f1_run"
        manifest = PublicationManifest(project_id=project_id, run_id=run_id)

        # Helper to create a StoredArtifact with distinct payload/hash
        def _make_art(letter: str, payload_bytes: bytes) -> StoredArtifact:
            sha = hashlib.sha256(payload_bytes).hexdigest()
            art_id = format_revision_artifact_id(project_id, run_id, "checkpoints/model.pt", sha)
            return StoredArtifact(
                artifact_id=art_id,
                storage_provider="local",
                path=f"/fake/path/{letter}/model.pt",
                size_bytes=len(payload_bytes),
                sha256=sha,
                project_id=project_id,
                run_id=run_id,
                artifact_type=ArtifactType.CHECKPOINT.value,
                logical_path="checkpoints/model.pt",
                metadata={"letter": letter},
            )

        art_a = _make_art("A", b"WEIGHTS_A_11111111")
        art_b = _make_art("B", b"WEIGHTS_B_22222222")
        art_c = _make_art("C", b"WEIGHTS_C_33333333")
        art_d = _make_art("D", b"WEIGHTS_D_44444444")
        art_e = _make_art("E", b"WEIGHTS_E_55555555")

        # --- Phase 1: 3-revision progression A -> B -> C ---
        manifest.add_artifact(art_a)
        self.assertIs(type(manifest.latest_by_path["checkpoints/model.pt"]), str)
        self.assertEqual(manifest.latest_by_path["checkpoints/model.pt"], art_a.artifact_id)
        self.assertEqual(manifest.get_latest_revision_id("checkpoints/model.pt"), art_a.revision_id)
        self.assertEqual(manifest.get_latest_sha256("checkpoints/model.pt"), art_a.sha256)
        rec_a = manifest.get_artifact(art_a.artifact_id)
        self.assertTrue(rec_a.is_latest)
        self.assertIsNone(rec_a.superseded_by)

        manifest.add_artifact(art_b)
        rec_a = manifest.get_artifact(art_a.artifact_id)
        rec_b = manifest.get_artifact(art_b.artifact_id)
        self.assertFalse(rec_a.is_latest)
        self.assertEqual(rec_a.superseded_by, art_b.artifact_id)
        self.assertIs(type(rec_a.superseded_by), str)
        self.assertTrue(rec_b.is_latest)
        self.assertIsNone(rec_b.superseded_by)
        self.assertEqual(manifest.latest_by_path["checkpoints/model.pt"], art_b.artifact_id)

        manifest.add_artifact(art_c)
        rec_a = manifest.get_artifact(art_a.artifact_id)
        rec_b = manifest.get_artifact(art_b.artifact_id)
        rec_c = manifest.get_artifact(art_c.artifact_id)
        self.assertFalse(rec_a.is_latest)
        # Critical: rec_a.superseded_by MUST STILL be art_b, NOT art_c!
        self.assertEqual(rec_a.superseded_by, art_b.artifact_id)
        self.assertNotEqual(rec_a.superseded_by, art_c.artifact_id)
        self.assertFalse(rec_b.is_latest)
        self.assertEqual(rec_b.superseded_by, art_c.artifact_id)
        self.assertTrue(rec_c.is_latest)
        self.assertIsNone(rec_c.superseded_by)
        self.assertEqual(manifest.latest_by_path["checkpoints/model.pt"], art_c.artifact_id)

        # --- Phase 2: Extend to 5-revision chain A -> B -> C -> D -> E ---
        manifest.add_artifact(art_d)
        manifest.add_artifact(art_e)

        rec_a = manifest.get_artifact(art_a.artifact_id)
        rec_b = manifest.get_artifact(art_b.artifact_id)
        rec_c = manifest.get_artifact(art_c.artifact_id)
        rec_d = manifest.get_artifact(art_d.artifact_id)
        rec_e = manifest.get_artifact(art_e.artifact_id)

        # Verify strict immediate-successor chain
        self.assertEqual(rec_a.superseded_by, art_b.artifact_id)
        self.assertEqual(rec_b.superseded_by, art_c.artifact_id)
        self.assertEqual(rec_c.superseded_by, art_d.artifact_id)
        self.assertEqual(rec_d.superseded_by, art_e.artifact_id)
        self.assertIsNone(rec_e.superseded_by)

        # Verify no conflation with latest tip
        self.assertNotEqual(rec_a.superseded_by, art_e.artifact_id)
        self.assertNotEqual(rec_b.superseded_by, art_e.artifact_id)
        self.assertNotEqual(rec_c.superseded_by, art_e.artifact_id)

        # Check types: NO RevisionRef subclass anywhere, strictly plain str or None
        self.assertIs(type(manifest.latest_by_path["checkpoints/model.pt"]), str)
        for rec in [rec_a, rec_b, rec_c, rec_d]:
            self.assertIs(type(rec.superseded_by), str)
            self.assertFalse(rec.is_latest)
            self.assertEqual(rec.metadata.get("superseded_by"), rec.superseded_by)
        self.assertIsNone(rec_e.superseded_by)
        self.assertTrue(rec_e.is_latest)

        # Verify accessors with varied input path normalization
        for path_query in [
            "checkpoints/model.pt",
            "/checkpoints/model.pt",
            f"{project_id}/{run_id}/checkpoints/model.pt",
            f"/{project_id}/{run_id}/checkpoints/model.pt",
            f"{project_id}/{run_id}/checkpoints/model.pt@sha256_unmatched",
        ]:
            latest_obj = manifest.get_latest_artifact(path_query)
            self.assertIsNotNone(latest_obj, f"Failed for query: {path_query}")
            self.assertEqual(latest_obj.artifact_id, art_e.artifact_id)
            self.assertEqual(manifest.get_latest_revision_id(path_query), art_e.revision_id)
            self.assertEqual(manifest.get_latest_sha256(path_query), art_e.sha256)

        # Nonexistent path returns None
        self.assertIsNone(manifest.get_latest_artifact("nonexistent/file.pt"))
        self.assertIsNone(manifest.get_latest_revision_id("nonexistent/file.pt"))
        self.assertIsNone(manifest.get_latest_sha256("nonexistent/file.pt"))

        # --- Phase 3: Python equality, symmetry, transitivity, and hash() invariants ---
        all_ids = [
            art_a.artifact_id, art_b.artifact_id, art_c.artifact_id, art_d.artifact_id, art_e.artifact_id,
            manifest.latest_by_path["checkpoints/model.pt"],
            rec_a.superseded_by, rec_b.superseded_by, rec_c.superseded_by, rec_d.superseded_by,
            f"sha256_{art_e.sha256}", art_e.sha256,
        ]

        for x in all_ids:
            # Reflexivity
            self.assertEqual(x, x)
            # Type is plain str
            self.assertIs(type(x), str)
            for y in all_ids:
                # Symmetry
                self.assertEqual(x == y, y == x)
                # Hash invariant: x == y implies hash(x) == hash(y)
                if x == y:
                    self.assertEqual(hash(x), hash(y))
                for z in all_ids:
                    # Transitivity: x == y and y == z implies x == z
                    if (x == y) and (y == z):
                        self.assertEqual(x, z)

        # Explicitly verify NO magical equality between full artifact ID and short revision digests
        self.assertNotEqual(manifest.latest_by_path["checkpoints/model.pt"], f"sha256_{art_e.sha256}")
        self.assertNotEqual(manifest.latest_by_path["checkpoints/model.pt"], art_e.sha256)
        self.assertNotEqual(rec_a.superseded_by, f"sha256_{art_b.sha256}")
        self.assertNotEqual(rec_a.superseded_by, art_b.sha256)

        # Set membership integrity
        id_set = set(manifest.artifacts.keys())
        self.assertEqual(len(id_set), 5)
        self.assertIn(art_e.artifact_id, id_set)
        self.assertIn(manifest.latest_by_path["checkpoints/model.pt"], id_set)
        self.assertNotIn(f"sha256_{art_e.sha256}", id_set)
        self.assertNotIn(art_e.sha256, id_set)

        # Dict key lookup integrity
        lookup_dict = {k: f"val_{i}" for i, k in enumerate(manifest.artifacts.keys())}
        self.assertEqual(len(lookup_dict), 5)
        self.assertEqual(lookup_dict[manifest.latest_by_path["checkpoints/model.pt"]], "val_4")
        self.assertNotIn(f"sha256_{art_e.sha256}", lookup_dict)
        self.assertNotIn(art_e.sha256, lookup_dict)

        # --- Phase 4: Persistence and round-trip verification ---
        save_path = Path(self.test_dir) / "f1_manifest_test.json"
        manifest.save(save_path)

        loaded_manifest = PublicationManifest.load(save_path)
        deserialized_manifest = PublicationManifest.from_dict(manifest.to_dict())

        for check_m in [loaded_manifest, deserialized_manifest]:
            self.assertEqual(len(check_m.artifacts), 5)
            self.assertIs(type(check_m.latest_by_path["checkpoints/model.pt"]), str)
            self.assertEqual(check_m.latest_by_path["checkpoints/model.pt"], art_e.artifact_id)

            c_a = check_m.get_artifact(art_a.artifact_id)
            c_b = check_m.get_artifact(art_b.artifact_id)
            c_c = check_m.get_artifact(art_c.artifact_id)
            c_d = check_m.get_artifact(art_d.artifact_id)
            c_e = check_m.get_artifact(art_e.artifact_id)

            self.assertIs(type(c_a.superseded_by), str)
            self.assertEqual(c_a.superseded_by, art_b.artifact_id)
            self.assertEqual(c_b.superseded_by, art_c.artifact_id)
            self.assertEqual(c_c.superseded_by, art_d.artifact_id)
            self.assertEqual(c_d.superseded_by, art_e.artifact_id)
            self.assertIsNone(c_e.superseded_by)

            self.assertEqual(check_m.get_latest_revision_id("checkpoints/model.pt"), art_e.revision_id)
            self.assertEqual(check_m.get_latest_sha256("checkpoints/model.pt"), art_e.sha256)
            self.assertEqual(check_m.get_latest_artifact("checkpoints/model.pt").artifact_id, art_e.artifact_id)

    def test_v1_to_v2_migration_chain_and_plain_str(self):
        """Verify that migrating a legacy v1 manifest with multiple sequential revisions
        enforces immediate successor chain semantics and plain str types for all keys and values.
        """
        sha_1 = "1111111111111111111111111111111111111111111111111111111111111111"
        sha_2 = "2222222222222222222222222222222222222222222222222222222222222222"
        sha_3 = "3333333333333333333333333333333333333333333333333333333333333333"

        v1_data = {
            "project_id": "legacy_chain",
            "run_id": "run_chain",
            "manifest_version": 1,
            "artifacts": {
                "legacy_chain/run_chain/checkpoints/model.pt": {
                    "artifact_id": "legacy_chain/run_chain/checkpoints/model.pt",
                    "storage_provider": "local",
                    "path": "/storage/model_1.pt",
                    "size_bytes": 100,
                    "sha256": sha_1,
                    "project_id": "legacy_chain",
                    "run_id": "run_chain",
                    "artifact_type": "checkpoint",
                    "created_at": "2026-01-01T12:00:00Z",
                },
                "legacy_chain/run_chain/checkpoints/model.pt_v2": {
                    "artifact_id": "legacy_chain/run_chain/checkpoints/model.pt",
                    "storage_provider": "local",
                    "path": "/storage/model_2.pt",
                    "size_bytes": 200,
                    "sha256": sha_2,
                    "project_id": "legacy_chain",
                    "run_id": "run_chain",
                    "artifact_type": "checkpoint",
                    "created_at": "2026-01-01T12:01:00Z",
                },
                "legacy_chain/run_chain/checkpoints/model.pt_v3": {
                    "artifact_id": "legacy_chain/run_chain/checkpoints/model.pt",
                    "storage_provider": "local",
                    "path": "/storage/model_3.pt",
                    "size_bytes": 300,
                    "sha256": sha_3,
                    "project_id": "legacy_chain",
                    "run_id": "run_chain",
                    "artifact_type": "checkpoint",
                    "created_at": "2026-01-01T12:02:00Z",
                },
            },
        }

        migrated = PublicationManifest.from_dict(v1_data)
        self.assertEqual(migrated.manifest_version, 2)
        self.assertEqual(len(migrated.artifacts), 3)

        expected_id_1 = f"legacy_chain/run_chain/checkpoints/model.pt@sha256_{sha_1}"
        expected_id_2 = f"legacy_chain/run_chain/checkpoints/model.pt@sha256_{sha_2}"
        expected_id_3 = f"legacy_chain/run_chain/checkpoints/model.pt@sha256_{sha_3}"

        # All artifact keys are plain str
        for k in migrated.artifacts.keys():
            self.assertIs(type(k), str)

        # latest_by_path is plain str
        latest_id = migrated.latest_by_path["checkpoints/model.pt"]
        self.assertIs(type(latest_id), str)
        self.assertEqual(latest_id, expected_id_3)

        art_1 = migrated.get_artifact(expected_id_1)
        art_2 = migrated.get_artifact(expected_id_2)
        art_3 = migrated.get_artifact(expected_id_3)

        self.assertFalse(art_1.is_latest)
        self.assertEqual(art_1.superseded_by, expected_id_2)
        self.assertIs(type(art_1.superseded_by), str)

        self.assertFalse(art_2.is_latest)
        self.assertEqual(art_2.superseded_by, expected_id_3)
        self.assertIs(type(art_2.superseded_by), str)

        self.assertTrue(art_3.is_latest)
        self.assertIsNone(art_3.superseded_by)

        # No conflation of art_1.superseded_by with latest tip expected_id_3
        self.assertNotEqual(art_1.superseded_by, expected_id_3)

        # Accessors work as expected on migrated manifest
        self.assertEqual(migrated.get_latest_artifact("checkpoints/model.pt").artifact_id, expected_id_3)
        self.assertEqual(migrated.get_latest_revision_id("checkpoints/model.pt"), f"sha256_{sha_3}")
        self.assertEqual(migrated.get_latest_sha256("checkpoints/model.pt"), sha_3)



if __name__ == "__main__":
    unittest.main()
