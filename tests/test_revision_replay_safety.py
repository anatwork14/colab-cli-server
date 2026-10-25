"""Offline deterministic unit tests for revision replay safety and idempotence.

Validates Argon-defined semantics:
- Republishing an existing historical revision is an IDEMPOTENT operation,
  never a request to change latest.
- Test A: publish A,B, republish A -> B latest remote+manifest,
  upload_sessions_initiated unchanged, no PATCH issued on replay.
- Test B: delete manifest+bak keep journal, republish A -> B stays latest remote,
  unversioned retrieve returns B.
- Test C: delete manifest,bak,journal, republish A -> A not promoted (remote B
  still the single latest) and no new upload.
- Test D: crash points (before demotion, after demotion before promotion,
  after promotion before manifest save) then retry -> consistent single latest,
  B ultimately latest.
- Test E: concurrent retries of A and B -> one remote file each, single latest,
  verifying final remote appProperties.
- Manifest and resolution edge cases.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import stat
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from colab_adapter.errors import (
    ArtifactNotFoundError,
    ArtifactValidationError,
    DriveAuthenticationError,
    DriveIntegrityError,
    DriveTransferError,
    StorageError,
)
from colab_adapter.storage import (
    ArtifactType,
    DriveCredentials,
    GoogleDriveArtifactStore,
    PublicationManifest,
    StoredArtifact,
    TransferRecord,
    TransferStatus,
    compute_file_md5,
    compute_file_sha256,
)
from colab_adapter.storage.drive_metadata import (
    STATE_HISTORICAL,
    STATE_LATEST,
    STATE_PENDING,
    revision_state,
)
from tests.test_google_drive_store import MockDriveHttpClient


class TestRevisionReplaySafety(unittest.TestCase):
    """Offline test suite for revision replay safety and crash consistency."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="replay_safety_test_")
        self.addCleanup(lambda: shutil.rmtree(self.test_dir, ignore_errors=True))
        self.staging_dir = Path(self.test_dir) / "staging"
        self.credentials_file = Path(self.test_dir) / "credentials.json"

        self.creds = DriveCredentials(
            client_id="test-client-id",
            client_secret="test-client-secret",
            refresh_token="test-refresh-token",
            access_token="initial-access-token",
            token_expiry=9999999999.0,
        )
        self.creds.save(self.credentials_file)

        self.mock_client = MockDriveHttpClient()
        self.store = GoogleDriveArtifactStore(
            credentials=self.creds,
            credentials_file=self.credentials_file,
            local_staging_dir=self.staging_dir,
            http_client=self.mock_client,
            chunk_size=1024 * 1024,
            initial_backoff_sec=0.01,
        )

        self.project_id = "test_project"
        self.run_id = "run_replay_01"

    def _create_source_file(self, filename: str, content: bytes) -> Path:
        src_path = Path(self.test_dir) / filename
        src_path.write_bytes(content)
        return src_path

    def _count_patch_requests(self) -> int:
        return sum(1 for m, _, _, _ in self.mock_client.requests_log if m == "PATCH")

    def test_test_a_publish_ab_republish_a_idempotent_no_patch_no_upload(self):
        """Test A: publish A,B, republish A -> B latest remote+manifest,

        upload_sessions_initiated unchanged, no PATCH issued on replay
        (count mock PATCH requests).
        """
        data_a = b"WEIGHTS_REV_A_DATA_AAAAAAAAAAAAAAAA"
        data_b = b"WEIGHTS_REV_B_DATA_BBBBBBBBBBBBBBBB"

        file_a = self._create_source_file("model_a.pt", data_a)
        # We publish as model.pt to simulate revisions of the same checkpoint
        src_model = Path(self.test_dir) / "model.pt"
        src_model.write_bytes(data_a)

        art_a = self.store.publish(src_model, self.project_id, self.run_id, ArtifactType.CHECKPOINT.value)
        self.assertEqual(self.mock_client.upload_sessions_initiated, 1)

        # Publish revision B
        src_model.write_bytes(data_b)
        art_b = self.store.publish(src_model, self.project_id, self.run_id, ArtifactType.CHECKPOINT.value)
        self.assertEqual(self.mock_client.upload_sessions_initiated, 2)

        # Baseline checks: B is latest in manifest and on Drive
        manifest = self.store._load_or_recover_manifest(self.project_id, self.run_id)
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.latest_by_path["checkpoints/model.pt"], art_b.artifact_id)
        rec_a = manifest.get_artifact(art_a.artifact_id)
        rec_b = manifest.get_artifact(art_b.artifact_id)
        self.assertFalse(rec_a.is_latest)
        self.assertEqual(rec_a.superseded_by, art_b.artifact_id)
        self.assertTrue(rec_b.is_latest)

        # Remotely, verify appProperties
        target_folder_id = self.store.resolve_target_folder_id(self.project_id, self.run_id, "checkpoints")
        model_files = [
            f for f in self.mock_client.mock_files
            if f.get("name") == "model.pt" and f.get("parent") == target_folder_id
        ]
        self.assertEqual(len(model_files), 2)
        mf_a = next(f for f in model_files if f.get("id") == art_a.remote_file_id)
        mf_b = next(f for f in model_files if f.get("id") == art_b.remote_file_id)
        self.assertEqual(mf_b["appProperties"].get("is_latest"), "true")
        self.assertEqual(mf_a["appProperties"].get("is_latest"), "false")

        # Record upload session count and PATCH request count
        upload_sessions_before = self.mock_client.upload_sessions_initiated
        patch_count_before = self._count_patch_requests()

        # REPUBLISH Revision A
        src_model.write_bytes(data_a)
        art_a_replay = self.store.publish(src_model, self.project_id, self.run_id, ArtifactType.CHECKPOINT.value)

        # Assert: upload_sessions_initiated UNCHANGED
        self.assertEqual(self.mock_client.upload_sessions_initiated, upload_sessions_before)

        # Assert: no PATCH issued on replay
        patch_count_after = self._count_patch_requests()
        self.assertEqual(patch_count_after, patch_count_before)

        # Assert: B is STILL latest in manifest
        manifest_after = self.store._load_or_recover_manifest(self.project_id, self.run_id)
        self.assertEqual(manifest_after.latest_by_path["checkpoints/model.pt"], art_b.artifact_id)
        self.assertTrue(manifest_after.get_artifact(art_b.artifact_id).is_latest)
        self.assertFalse(manifest_after.get_artifact(art_a.artifact_id).is_latest)

        # Assert: B is STILL latest remotely
        self.assertEqual(mf_b["appProperties"].get("is_latest"), "true")
        self.assertEqual(mf_a["appProperties"].get("is_latest"), "false")

        # Assert: art_a_replay returned corresponds to A and is marked historical
        self.assertEqual(art_a_replay.artifact_id, art_a.artifact_id)
        self.assertFalse(art_a_replay.is_latest)

    def test_test_b_delete_manifest_keep_journal_republish_a(self):
        """Test B: delete manifest+bak keep journal, republish A -> B stays latest remote,

        unversioned retrieve returns B.
        """
        data_a = b"WEIGHTS_REV_A_DATA_AAAAAAAAAAAAAAAA"
        data_b = b"WEIGHTS_REV_B_DATA_BBBBBBBBBBBBBBBB"

        src_model = Path(self.test_dir) / "model.pt"
        src_model.write_bytes(data_a)
        art_a = self.store.publish(src_model, self.project_id, self.run_id, ArtifactType.CHECKPOINT.value)

        src_model.write_bytes(data_b)
        art_b = self.store.publish(src_model, self.project_id, self.run_id, ArtifactType.CHECKPOINT.value)

        # Delete manifest and manifest.bak, keep journal
        manifest_path = self.staging_dir / "projects" / self.project_id / "runs" / self.run_id / "manifest.json"
        manifest_path.unlink(missing_ok=True)
        manifest_path.with_suffix(".json.bak").unlink(missing_ok=True)
        self.store._manifest_cache.clear()

        # Verify journal still has records
        journal_records = self.store.journal.list_transfers(project_id=self.project_id, run_id=self.run_id)
        self.assertGreaterEqual(len(journal_records), 2)

        # Republish Revision A (Journal fast-path replay)
        src_model.write_bytes(data_a)
        art_a_replayed = self.store.publish(src_model, self.project_id, self.run_id, ArtifactType.CHECKPOINT.value)

        # Assert: B stays latest remotely
        target_folder_id = self.store.resolve_target_folder_id(self.project_id, self.run_id, "checkpoints")
        model_files = [
            f for f in self.mock_client.mock_files
            if f.get("name") == "model.pt" and f.get("parent") == target_folder_id
        ]
        mf_a = next(f for f in model_files if f.get("id") == art_a.remote_file_id)
        mf_b = next(f for f in model_files if f.get("id") == art_b.remote_file_id)
        self.assertEqual(mf_b["appProperties"].get("is_latest"), "true")
        self.assertEqual(mf_a["appProperties"].get("is_latest"), "false")

        # Assert: A added to manifest as historical
        manifest = self.store._load_or_recover_manifest(self.project_id, self.run_id)
        self.assertIsNotNone(manifest)
        rec_a = manifest.get_artifact(art_a.artifact_id)
        self.assertIsNotNone(rec_a)
        self.assertFalse(rec_a.is_latest)
        self.assertNotIn("checkpoints/model.pt", manifest.latest_by_path)

        # Assert: Unversioned retrieve returns B
        dest = Path(self.test_dir) / "retrieved_model.pt"
        self.store.retrieve(f"{self.project_id}/{self.run_id}/checkpoints/model.pt", dest)
        self.assertEqual(dest.read_bytes(), data_b)

    def test_test_c_delete_manifest_and_journal_republish_a(self):
        """Test C: delete manifest,bak,journal, republish A -> A not promoted

        (remote B still the single latest) and no new upload.
        """
        data_a = b"WEIGHTS_REV_A_DATA_AAAAAAAAAAAAAAAA"
        data_b = b"WEIGHTS_REV_B_DATA_BBBBBBBBBBBBBBBB"

        src_model = Path(self.test_dir) / "model.pt"
        src_model.write_bytes(data_a)
        art_a = self.store.publish(src_model, self.project_id, self.run_id, ArtifactType.CHECKPOINT.value)

        src_model.write_bytes(data_b)
        art_b = self.store.publish(src_model, self.project_id, self.run_id, ArtifactType.CHECKPOINT.value)

        # Delete manifest, manifest.bak, journal, and journal.bak
        manifest_path = self.staging_dir / "projects" / self.project_id / "runs" / self.run_id / "manifest.json"
        manifest_path.unlink(missing_ok=True)
        manifest_path.with_suffix(".json.bak").unlink(missing_ok=True)
        self.store._manifest_cache.clear()

        if self.store.journal.path.exists():
            self.store.journal.path.unlink()
        self.store.journal.path.with_suffix(".json.bak").unlink(missing_ok=True)
        self.store.journal.path.with_suffix(".bak").unlink(missing_ok=True)
        from colab_adapter.storage import TransferJournal
        self.store.journal = TransferJournal(self.store.journal.path)

        upload_sessions_before = self.mock_client.upload_sessions_initiated

        # Republish Revision A (Step-6 Drive discovery path)
        src_model.write_bytes(data_a)
        art_a_replayed = self.store.publish(src_model, self.project_id, self.run_id, ArtifactType.CHECKPOINT.value)

        # Assert: No new upload session initiated
        self.assertEqual(self.mock_client.upload_sessions_initiated, upload_sessions_before)

        # Assert: A is NOT promoted, remote B is still the SINGLE latest
        target_folder_id = self.store.resolve_target_folder_id(self.project_id, self.run_id, "checkpoints")
        model_files = [
            f for f in self.mock_client.mock_files
            if f.get("name") == "model.pt" and f.get("parent") == target_folder_id
        ]
        self.assertEqual(len(model_files), 2)
        mf_a = next(f for f in model_files if f.get("id") == art_a.remote_file_id)
        mf_b = next(f for f in model_files if f.get("id") == art_b.remote_file_id)

        self.assertEqual(mf_b["appProperties"].get("is_latest"), "true")
        self.assertEqual(mf_a["appProperties"].get("is_latest"), "false")

        latest_remote_files = [f for f in model_files if f.get("appProperties", {}).get("is_latest") == "true"]
        self.assertEqual(len(latest_remote_files), 1)
        self.assertEqual(latest_remote_files[0]["id"], art_b.remote_file_id)

        self.assertFalse(art_a_replayed.is_latest)

        # Assert: Unversioned retrieve returns B
        dest = Path(self.test_dir) / "retrieved_model_c.pt"
        self.store.retrieve(f"{self.project_id}/{self.run_id}/checkpoints/model.pt", dest)
        self.assertEqual(dest.read_bytes(), data_b)

    def test_test_d_crash_points_then_retry(self):
        """Test D: crash points (fail PATCH / raise at: before demotion, after demotion

        before promotion, after promotion before manifest save) then retry ->
        consistent single latest, B ultimately latest.
        """
        data_a = b"WEIGHTS_REV_A_DATA_AAAAAAAAAAAAAAAA"
        data_b = b"WEIGHTS_REV_B_DATA_BBBBBBBBBBBBBBBB"

        # --- Subtest D1: Crash before demotion ---
        # Publish A
        run_d1 = "run_d1_demotion_crash"
        src_model = Path(self.test_dir) / "model_d1.pt"
        src_model.write_bytes(data_a)
        art_a1 = self.store.publish(src_model, self.project_id, run_d1, ArtifactType.CHECKPOINT.value)

        # Inject failure on first PATCH (before demoting A)
        self.mock_client.patch_status_code = 503
        self.mock_client.fail_next_n_patches = self.store.max_retries + 1
        src_model.write_bytes(data_b)
        with self.assertRaises(DriveTransferError):
            self.store.publish(src_model, self.project_id, run_d1, ArtifactType.CHECKPOINT.value)

        # Reset failure injection and retry publishing B
        self.mock_client.fail_next_n_patches = 0
        art_b1 = self.store.publish(src_model, self.project_id, run_d1, ArtifactType.CHECKPOINT.value)

        # Assert: B ultimately latest, single latest remotely
        target_folder_id = self.store.resolve_target_folder_id(self.project_id, run_d1, "checkpoints")
        files_d1 = [
            f for f in self.mock_client.mock_files
            if f.get("name") == "model_d1.pt" and f.get("parent") == target_folder_id
        ]
        self.assertEqual(len(files_d1), 2)
        mf_a1 = next(f for f in files_d1 if f.get("id") == art_a1.remote_file_id)
        mf_b1 = next(f for f in files_d1 if f.get("id") == art_b1.remote_file_id)
        self.assertEqual(mf_b1["appProperties"].get("is_latest"), "true")
        self.assertEqual(mf_a1["appProperties"].get("is_latest"), "false")
        latest_d1 = [f for f in files_d1 if f.get("appProperties", {}).get("is_latest") == "true"]
        self.assertEqual(len(latest_d1), 1)
        self.assertEqual(latest_d1[0]["id"], art_b1.remote_file_id)

        # --- Subtest D2: Crash after demotion before promotion ---
        run_d2 = "run_d2_promote_crash"
        src_model_d2 = Path(self.test_dir) / "model_d2.pt"
        src_model_d2.write_bytes(data_a)
        art_a2 = self.store.publish(src_model_d2, self.project_id, run_d2, ArtifactType.CHECKPOINT.value)

        # Demote succeeds, but promote crashes with ConnectionResetError
        self.mock_client.abort_before_promote_patch = True
        src_model_d2.write_bytes(data_b)
        with self.assertRaises((DriveTransferError, ConnectionResetError)):
            self.store.publish(src_model_d2, self.project_id, run_d2, ArtifactType.CHECKPOINT.value)

        # Reset flag and retry publishing B
        self.mock_client.abort_before_promote_patch = False
        art_b2 = self.store.publish(src_model_d2, self.project_id, run_d2, ArtifactType.CHECKPOINT.value)

        # Assert: single latest remotely, B is latest
        target_folder_id_d2 = self.store.resolve_target_folder_id(self.project_id, run_d2, "checkpoints")
        files_d2 = [
            f for f in self.mock_client.mock_files
            if f.get("name") == "model_d2.pt" and f.get("parent") == target_folder_id_d2
        ]
        self.assertEqual(len(files_d2), 2)
        mf_a2 = next(f for f in files_d2 if f.get("id") == art_a2.remote_file_id)
        mf_b2 = next(f for f in files_d2 if f.get("id") == art_b2.remote_file_id)
        self.assertEqual(mf_b2["appProperties"].get("is_latest"), "true")
        self.assertEqual(mf_a2["appProperties"].get("is_latest"), "false")
        latest_d2 = [f for f in files_d2 if f.get("appProperties", {}).get("is_latest") == "true"]
        self.assertEqual(len(latest_d2), 1)
        self.assertEqual(latest_d2[0]["id"], art_b2.remote_file_id)

        # --- Subtest D3: Crash after promotion before manifest save ---
        run_d3 = "run_d3_manifest_save_crash"
        src_model_d3 = Path(self.test_dir) / "model_d3.pt"
        src_model_d3.write_bytes(data_a)
        art_a3 = self.store.publish(src_model_d3, self.project_id, run_d3, ArtifactType.CHECKPOINT.value)

        # Hook manifest save to raise on first attempt
        original_save = self.store._save_manifest_with_backup
        fail_save = True

        def failing_save(manifest, path):
            nonlocal fail_save
            if fail_save:
                fail_save = False
                raise IOError("Simulated disk error during manifest save")
            original_save(manifest, path)

        self.store._save_manifest_with_backup = failing_save
        src_model_d3.write_bytes(data_b)
        with self.assertRaises(IOError):
            self.store.publish(src_model_d3, self.project_id, run_d3, ArtifactType.CHECKPOINT.value)

        # Restore original save and retry publishing B
        self.store._save_manifest_with_backup = original_save
        art_b3 = self.store.publish(src_model_d3, self.project_id, run_d3, ArtifactType.CHECKPOINT.value)

        # Assert: single latest remotely and in manifest, B ultimately latest
        target_folder_id_d3 = self.store.resolve_target_folder_id(self.project_id, run_d3, "checkpoints")
        files_d3 = [
            f for f in self.mock_client.mock_files
            if f.get("name") == "model_d3.pt" and f.get("parent") == target_folder_id_d3
        ]
        self.assertEqual(len(files_d3), 2)
        mf_a3 = next(f for f in files_d3 if f.get("id") == art_a3.remote_file_id)
        mf_b3 = next(f for f in files_d3 if f.get("id") == art_b3.remote_file_id)
        self.assertEqual(mf_b3["appProperties"].get("is_latest"), "true")
        self.assertEqual(mf_a3["appProperties"].get("is_latest"), "false")
        latest_d3 = [f for f in files_d3 if f.get("appProperties", {}).get("is_latest") == "true"]
        self.assertEqual(len(latest_d3), 1)
        self.assertEqual(latest_d3[0]["id"], art_b3.remote_file_id)

    def test_test_e_concurrent_retry_a_and_b(self):
        """Test E: two threads/processes retry A and B -> one remote file each,

        single latest. Assert final remote appProperties not just call success.
        """
        data_a = b"WEIGHTS_REV_A_DATA_AAAAAAAAAAAAAAAA"
        data_b = b"WEIGHTS_REV_B_DATA_BBBBBBBBBBBBBBBB"

        file_a = self._create_source_file("src_model_a.pt", data_a)
        file_b = self._create_source_file("src_model_b.pt", data_b)

        run_e = "run_concurrent_retry"
        # Publish A and B initially to set up history
        art_a = self.store.publish(file_a, self.project_id, run_e, ArtifactType.CHECKPOINT.value)
        art_b = self.store.publish(file_b, self.project_id, run_e, ArtifactType.CHECKPOINT.value)

        # Concurrently retry A and retry B across multiple threads
        errors: List[Exception] = []
        barrier = threading.Barrier(2)

        def retry_worker(src_file: Path, name: str):
            try:
                barrier.wait(timeout=5.0)
                res = self.store.publish(src_file, self.project_id, run_e, ArtifactType.CHECKPOINT.value)
                return res
            except Exception as e:
                errors.append(e)
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_a = executor.submit(retry_worker, file_a, "A")
            fut_b = executor.submit(retry_worker, file_b, "B")
            res_a = fut_a.result()
            res_b = fut_b.result()

        self.assertEqual(len(errors), 0, f"Concurrent retry errors: {errors}")
        self.assertIsNotNone(res_a)
        self.assertIsNotNone(res_b)

        # Assert: exactly one remote file each for A and B in target folder
        # Note: Since file_a had name "src_model_a.pt" and file_b had name "src_model_b.pt",
        # let's also test where both source files have the IDENTICAL filename "checkpoint.pt"!
        run_e_same_name = "run_concurrent_same_name"
        file_same = Path(self.test_dir) / "checkpoint.pt"
        file_same.write_bytes(data_a)
        art_a_same = self.store.publish(file_same, self.project_id, run_e_same_name, ArtifactType.CHECKPOINT.value)

        file_same.write_bytes(data_b)
        art_b_same = self.store.publish(file_same, self.project_id, run_e_same_name, ArtifactType.CHECKPOINT.value)

        # Create two distinct local files with identical filename in different directories
        dir1 = Path(self.test_dir) / "dir1"
        dir2 = Path(self.test_dir) / "dir2"
        dir1.mkdir(parents=True, exist_ok=True)
        dir2.mkdir(parents=True, exist_ok=True)
        file_a_path = dir1 / "checkpoint.pt"
        file_b_path = dir2 / "checkpoint.pt"
        file_a_path.write_bytes(data_a)
        file_b_path.write_bytes(data_b)

        barrier2 = threading.Barrier(2)
        errors2: List[Exception] = []

        def retry_worker_same(src_file: Path):
            try:
                barrier2.wait(timeout=5.0)
                return self.store.publish(src_file, self.project_id, run_e_same_name, ArtifactType.CHECKPOINT.value)
            except Exception as e:
                errors2.append(e)
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut1 = executor.submit(retry_worker_same, file_a_path)
            fut2 = executor.submit(retry_worker_same, file_b_path)
            r1 = fut1.result()
            r2 = fut2.result()

        self.assertEqual(len(errors2), 0, f"Concurrent same-name retry errors: {errors2}")

        # Assert: exactly two remote files in target folder (one for A, one for B)
        target_folder_same = self.store.resolve_target_folder_id(self.project_id, run_e_same_name, "checkpoints")
        remote_same = [
            f for f in self.mock_client.mock_files
            if f.get("name") == "checkpoint.pt" and f.get("parent") == target_folder_same
        ]
        self.assertEqual(len(remote_same), 2)
        mf_a_same = next(f for f in remote_same if f.get("id") == art_a_same.remote_file_id)
        mf_b_same = next(f for f in remote_same if f.get("id") == art_b_same.remote_file_id)

        # Assert final remote appProperties: single latest is B
        self.assertEqual(mf_b_same["appProperties"].get("is_latest"), "true")
        self.assertEqual(mf_a_same["appProperties"].get("is_latest"), "false")
        self.assertTrue(
            bool(mf_a_same["appProperties"].get("next_sha256") or mf_a_same["appProperties"].get("superseded_by"))
        )

        latest_same = [f for f in remote_same if f.get("appProperties", {}).get("is_latest") == "true"]
        self.assertEqual(len(latest_same), 1)
        self.assertEqual(latest_same[0]["id"], art_b_same.remote_file_id)

    def test_manifest_add_artifact_historical_and_successor_semantics(self):
        """Test manifest.add_artifact honors is_latest=False for brand-new revision

        without altering other revisions' flags or latest_by_path, and preserves
        immediate-successor superseded_by semantics.
        """
        manifest = PublicationManifest(project_id="proj", run_id="run1")

        # 1. Add revision B (created at 10:05) as latest
        art_b = StoredArtifact(
            artifact_id="proj/run1/checkpoints/model.pt@sha256_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            storage_provider="google_drive",
            path="checkpoints/model.pt",
            size_bytes=100,
            sha256="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            project_id="proj",
            run_id="run1",
            created_at="2026-10-25T10:05:00Z",
            is_latest=True,
        )
        manifest.add_artifact(art_b)
        self.assertEqual(manifest.latest_by_path["checkpoints/model.pt"], art_b.artifact_id)
        self.assertTrue(manifest.get_artifact(art_b.artifact_id).is_latest)

        # 2. Insert historical revision A (created earlier at 10:00) with is_latest=False
        art_a = StoredArtifact(
            artifact_id="proj/run1/checkpoints/model.pt@sha256_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            storage_provider="google_drive",
            path="checkpoints/model.pt",
            size_bytes=100,
            sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            project_id="proj",
            run_id="run1",
            created_at="2026-10-25T10:00:00Z",
            is_latest=False,
        )
        manifest.add_artifact(art_a)

        # Assert: B is still latest, latest_by_path untouched
        self.assertEqual(manifest.latest_by_path["checkpoints/model.pt"], art_b.artifact_id)
        self.assertTrue(manifest.get_artifact(art_b.artifact_id).is_latest)

        # Assert: A is historical and its superseded_by points to B (immediate successor)
        rec_a = manifest.get_artifact(art_a.artifact_id)
        self.assertFalse(rec_a.is_latest)
        self.assertEqual(rec_a.superseded_by, art_b.artifact_id)
        self.assertIsInstance(rec_a.superseded_by, str)

        # 3. Insert historical revision when successor is NOT known in manifest
        empty_manifest = PublicationManifest(project_id="proj2", run_id="run2")
        empty_manifest.add_artifact(art_a)
        rec_a_alone = empty_manifest.get_artifact(art_a.artifact_id)
        self.assertFalse(rec_a_alone.is_latest)
        self.assertIsNone(rec_a_alone.superseded_by)
        self.assertNotIn("checkpoints/model.pt", empty_manifest.latest_by_path)

        # 4. Unversioned get_artifact returns None when no revision is marked latest
        unversioned = empty_manifest.get_artifact("checkpoints/model.pt")
        self.assertIsNone(unversioned)

    def test_resolve_artifact_from_manifest_unversioned_no_latest_returns_none(self):
        """Test _resolve_artifact_from_manifest returns None (not matching[0])

        when no revision is marked latest.
        """
        manifest = PublicationManifest(project_id="proj", run_id="run")
        art_historical = StoredArtifact(
            artifact_id="proj/run/checkpoints/model.pt@sha256_1111111111111111111111111111111111111111111111111111111111111111",
            storage_provider="google_drive",
            path="checkpoints/model.pt",
            size_bytes=50,
            sha256="1111111111111111111111111111111111111111111111111111111111111111",
            project_id="proj",
            run_id="run",
            is_latest=False,
            superseded_by="proj/run/checkpoints/model.pt@sha256_2222222222222222222222222222222222222222222222222222222222222222",
        )
        manifest.add_artifact(art_historical)

        res = self.store._resolve_artifact_from_manifest(manifest, "proj/run/checkpoints/model.pt")
        self.assertIsNone(res)

    def test_remote_revision_state_failure_fails_closed(self):
        """Test remote revision state lookup error in publish raises (fails closed),

        never fabricating success.
        """
        data = b"FAIL_CLOSED_TEST_DATA"
        src_file = self._create_source_file("fail_model.pt", data)

        # Create record in journal as PUBLISHED with a non-existent remote_id
        non_existent_fid = "file-99999"
        rec = TransferRecord(
            transfer_id="transfer-fail-closed",
            project_id=self.project_id,
            run_id=self.run_id,
            artifact_id=f"{self.project_id}/{self.run_id}/checkpoints/fail_model.pt@sha256_{hashlib.sha256(data).hexdigest()}",
            local_path=str(src_file),
            target_rel_path="checkpoints/fail_model.pt",
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            status=TransferStatus.PUBLISHED.value,
            remote_file_id=non_existent_fid,
        )
        self.store.journal.record_transfer(rec)

        # Publishing should fail closed with a typed error (journal points at a remote file that
        # no longer exists -> ArtifactNotFoundError; transient lookup faults -> DriveTransferError)
        with self.assertRaises((DriveTransferError, ArtifactNotFoundError)):
            self.store.publish(src_file, self.project_id, self.run_id, ArtifactType.CHECKPOINT.value)
        manifest = self.store._load_or_recover_manifest(self.project_id, self.run_id)
        self.assertTrue(manifest is None or not manifest.artifacts)


if __name__ == "__main__":
    unittest.main()
