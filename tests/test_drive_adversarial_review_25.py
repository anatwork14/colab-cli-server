"""Adversarial QA suite reviewing Drive v3 storage protocol compliance.

Validates:
1. StrictDriveMock enforcement of Google Drive constraints:
   - 124 UTF-8 bytes key+value limit on custom properties (reproducing reported failure shape)
   - Maximum 30 custom appProperties count
   - PATCH merging with null-deletion
   - Fields parameter filtering and response integrity
2. GoogleDriveArtifactStore revision coordination and recovery:
   - Three-revision (A->B->C) publish under strict limits
   - Long nested checkpoint paths
   - PATCH lost final response convergence
   - Bounded retries on HTTP 429/503
   - Typed error surfacing on HTTP 400 without manifest corruption
   - Fail-closed unversioned retrieval on ambiguous remote latest markers (0 or 2)
   - Backward-compatible retrieval of legacy-metadata files
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from colab_adapter.errors import (
    ArtifactNotFoundError,
    ArtifactValidationError,
    DriveAuthenticationError,
    DriveIntegrityError,
    DriveQuotaExceededError,
    DriveTransferError,
    StorageError,
)
from colab_adapter.storage import (
    ArtifactType,
    DriveCredentials,
    GoogleDriveArtifactStore,
    PublicationManifest,
    TransferStatus,
    compute_file_sha256,
)
from tests.drive_api_mock import StrictDriveMock


class TestDriveAdversarialReview25(unittest.TestCase):
    """Adversarial review tests under StrictDriveMock."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="adversarial_drive_25_")
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

        self.mock_client = StrictDriveMock()
        self.store = GoogleDriveArtifactStore(
            credentials=self.creds,
            credentials_file=self.credentials_file,
            local_staging_dir=self.staging_dir,
            http_client=self.mock_client,
            chunk_size=1024 * 1024,
            initial_backoff_sec=0.01,
        )

    def test_01_reproduce_failure_shape_old_superseded_by_exceeds_124_bytes_rejected(self):
        """(1) Reproduce reported failure shape: OLD property superseded_by is 136 bytes -> StrictDriveMock rejects it."""
        mock = StrictDriveMock()
        mock.mock_files.append({
            "id": "file-seeded-01",
            "name": "model.pt",
            "appProperties": {},
        })

        # Old protocol property format:
        # key: 'superseded_by' (13 UTF-8 bytes)
        # val: 'vision-research/experiment-001/checkpoints/model.pt@sha256_' + 64 hex chars (123 UTF-8 bytes)
        # Total: 136 bytes > 124 bytes
        key = "superseded_by"
        sha_hex = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        val = f"vision-research/experiment-001/checkpoints/model.pt@sha256_{sha_hex}"

        total_bytes = len(key.encode("utf-8")) + len(val.encode("utf-8"))
        self.assertEqual(total_bytes, 136, f"Expected exactly 136 bytes, got {total_bytes}")

        # Assert StrictDriveMock rejects PATCH with HTTP 400
        patch_url = "https://www.googleapis.com/drive/v3/files/file-seeded-01?fields=id,name,appProperties"
        patch_payload = json.dumps({"appProperties": {key: val}}).encode("utf-8")
        status, headers, body = mock.request("PATCH", patch_url, headers={"content-type": "application/json"}, data=patch_payload)

        self.assertEqual(status, 400, f"Expected HTTP 400 Bad Request, got {status}")
        err_data = json.loads(body.decode("utf-8"))
        self.assertIn("error", err_data)
        self.assertIn("124 bytes", err_data["error"]["message"])
        self.assertIn("136 bytes", err_data["error"]["message"])

        # Assert StrictDriveMock also rejects resumable-init POST with HTTP 400
        init_url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable"
        init_payload = json.dumps({"name": "model.pt", "appProperties": {key: val}}).encode("utf-8")
        status_init, _, body_init = mock.request("POST", init_url, headers={"content-type": "application/json"}, data=init_payload)

        self.assertEqual(status_init, 400, f"Expected HTTP 400 Bad Request on resumable-init, got {status_init}")
        err_init = json.loads(body_init.decode("utf-8"))
        self.assertIn("124 bytes", err_init["error"]["message"])

    def test_02_publish_a_b_c_passes_under_strict_mock_single_latest_compliant_props(self):
        """(2) A->B->C publish passes under StrictDriveMock, snapshot has exactly one is_latest='true' and all props <= 124 bytes."""
        project_id = "vision-research"
        run_id = "experiment-001"
        file_path = Path(self.test_dir) / "model.pt"

        # Publish Revision A
        file_path.write_bytes(b"CHECKPOINT_DATA_REVISION_A")
        art_a = self.store.publish(
            source=file_path,
            project_id=project_id,
            run_id=run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
        )

        # Publish Revision B
        file_path.write_bytes(b"CHECKPOINT_DATA_REVISION_B")
        art_b = self.store.publish(
            source=file_path,
            project_id=project_id,
            run_id=run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
        )

        # Publish Revision C
        file_path.write_bytes(b"CHECKPOINT_DATA_REVISION_C")
        art_c = self.store.publish(
            source=file_path,
            project_id=project_id,
            run_id=run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
        )

        # Inspect final snapshot
        snap = self.mock_client.snapshot()
        self.assertEqual(len(snap), 3)

        latest_files = [f for f in snap if f.appProperties.get("is_latest") == "true"]
        self.assertEqual(
            len(latest_files),
            1,
            f"Expected exactly 1 file with is_latest=='true', found {len(latest_files)}: {latest_files}",
        )
        self.assertEqual(latest_files[0].id, art_c.remote_file_id)

        # All custom properties across all files must be <= 124 UTF-8 bytes
        for f in snap:
            for k, v in f.appProperties.items():
                prop_size = len(k.encode("utf-8")) + len(str(v).encode("utf-8"))
                self.assertLessEqual(
                    prop_size,
                    124,
                    f"Property '{k}' on file {f.id} exceeds 124 bytes ({prop_size} bytes)",
                )

    def test_03_long_nested_checkpoint_path_either_succeeds_compliant_or_rejects_before_remote_write(self):
        """(3) Long nested checkpoint path either succeeds with compliant props or rejects before remote write."""
        # Long valid identifiers within _SAFE_ID_PATTERN limits
        # Key "project_id" (10 bytes) + value (118 bytes) = 128 bytes (> 124 bytes limit if stored verbatim)
        long_project_id = "project-" + "x" * 110
        long_run_id = "run-001"
        filename = "модель_weights_checkpoint_тест.pt"
        file_path = Path(self.test_dir) / filename
        file_path.write_bytes(b"LONG_NESTED_PATH_DATA")

        try:
            art = self.store.publish(
                source=file_path,
                project_id=long_project_id,
                run_id=long_run_id,
                artifact_type=ArtifactType.CHECKPOINT.value,
            )
            # Branch A: Successfully published with compliant properties
            snap = self.mock_client.snapshot()
            self.assertGreater(len(snap), 0, "Snapshot should contain published file")
            for f in snap:
                for k, v in f.appProperties.items():
                    prop_size = len(k.encode("utf-8")) + len(str(v).encode("utf-8"))
                    self.assertLessEqual(
                        prop_size,
                        124,
                        f"Custom property '{k}' in file {f.id} exceeds 124 bytes ({prop_size} bytes)",
                    )
        except (DriveTransferError, ArtifactValidationError, StorageError) as exc:
            # Branch B: Rejected with a typed error BEFORE any remote write
            self.assertEqual(
                len(self.mock_client.recorded_posts),
                0,
                f"Expected 0 remote POST requests before write, but {len(self.mock_client.recorded_posts)} were recorded: {self.mock_client.recorded_posts}",
            )
            self.assertEqual(
                len(self.mock_client.recorded_patches),
                0,
                f"Expected 0 remote PATCH requests before write, but {len(self.mock_client.recorded_patches)} were recorded: {self.mock_client.recorded_patches}",
            )

    def test_04_patch_lost_final_response_then_retry_converges_single_latest(self):
        """(4) PATCH lost-final-response then retry -> converges, single latest."""
        project_id = "proj-p4"
        run_id = "run-r4"
        file_path = Path(self.test_dir) / "model.pt"
        file_path.write_bytes(b"DATA_CONVERGENCE_TEST")

        # Configure mock to commit PATCH but lose final response (return 503)
        self.mock_client.lost_final_response_on_patch = True
        self.mock_client.lost_final_response_count = 1
        self.mock_client.lost_final_response_as_exception = False

        # store has max_retries = 0 so caller handles retry
        self.store.max_retries = 0

        # First attempt fails due to lost final response
        with self.assertRaises(DriveTransferError):
            self.store.publish(
                source=file_path,
                project_id=project_id,
                run_id=run_id,
                artifact_type=ArtifactType.CHECKPOINT.value,
            )

        # Retry publishing the same artifact
        art = self.store.publish(
            source=file_path,
            project_id=project_id,
            run_id=run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
        )

        self.assertIsNotNone(art.remote_file_id)

        # Final snapshot has exactly one is_latest == 'true'
        snap = self.mock_client.snapshot()
        latest_files = [f for f in snap if f.appProperties.get("is_latest") == "true"]
        self.assertEqual(len(latest_files), 1)
        self.assertEqual(latest_files[0].id, art.remote_file_id)
        self.assertEqual(latest_files[0].content_sha, compute_file_sha256(file_path))

        # Unversioned retrieval succeeds
        dest = Path(self.test_dir) / "retrieved_converged.pt"
        self.store.retrieve(f"{project_id}/{run_id}/checkpoints/model.pt", dest)
        self.assertEqual(dest.read_bytes(), b"DATA_CONVERGENCE_TEST")

    def test_05_rate_limit_and_service_unavailable_on_patch_retried_bounded(self):
        """(5) 429/503 on PATCH retried bounded."""
        project_id = "proj-p5"
        run_id = "run-r5"
        file_path = Path(self.test_dir) / "model.pt"
        file_path.write_bytes(b"DATA_RETRY_TEST")

        # Subtest 5a: Transient 429 then 503 succeeds within bounded retries
        self.store.max_retries = 2
        self.store.initial_backoff_sec = 0.01
        self.mock_client.fail_operation("patch", 429, count=1)
        self.mock_client.fail_operation("patch", 503, count=1)

        art = self.store.publish(
            source=file_path,
            project_id=project_id,
            run_id=run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
        )
        self.assertIsNotNone(art.remote_file_id)
        self.assertEqual(len(self.mock_client.recorded_patches), 3)
        self.assertEqual(self.mock_client.recorded_patches[0]["status"], 429)
        self.assertEqual(self.mock_client.recorded_patches[1]["status"], 503)
        self.assertEqual(self.mock_client.recorded_patches[2]["status"], 200)

        # Subtest 5b: Persistent 503 fails boundedly
        file_path_2 = Path(self.test_dir) / "model2.pt"
        file_path_2.write_bytes(b"DATA_RETRY_PERSISTENT")
        self.mock_client.recorded_patches.clear()
        self.mock_client.fail_operation("patch", 503, count=10)

        with self.assertRaises(DriveTransferError):
            self.store.publish(
                source=file_path_2,
                project_id=project_id,
                run_id=run_id,
                artifact_type=ArtifactType.CHECKPOINT.value,
            )
        # Verify retries were bounded: exactly max_retries + 1 attempts (3), not 10
        self.assertEqual(
            len(self.mock_client.recorded_patches),
            self.store.max_retries + 1,
            f"Expected {self.store.max_retries + 1} bounded attempts, got {len(self.mock_client.recorded_patches)}",
        )

    def test_06_http_400_on_patch_surfaces_typed_error_and_manifest_not_success(self):
        """(6) HTTP 400 on PATCH surfaces typed error and local manifest does NOT record success/latest."""
        project_id = "proj-p6"
        run_id = "run-r6"
        file_path = Path(self.test_dir) / "model.pt"
        file_path.write_bytes(b"DATA_HTTP_400_TEST")

        self.mock_client.fail_operation("patch", 400, count=1)

        with self.assertRaises(DriveTransferError) as ctx:
            self.store.publish(
                source=file_path,
                project_id=project_id,
                run_id=run_id,
                artifact_type=ArtifactType.CHECKPOINT.value,
            )
        self.assertIn("HTTP 400", str(ctx.exception))

        # Local manifest must NOT record success or latest
        manifest_path = self.staging_dir / "projects" / project_id / "runs" / run_id / "manifest.json"
        if manifest_path.exists():
            manifest = PublicationManifest.load(manifest_path)
            for art in manifest.artifacts:
                self.assertFalse(art.is_latest, "Manifest must not record artifact as latest on PATCH 400 failure")
        cached_manifest = self.store._manifest_cache.get((project_id, run_id))
        if cached_manifest:
            for art in cached_manifest.artifacts:
                self.assertFalse(art.is_latest, "Cached manifest must not record artifact as latest on failure")

        # Transfer record in journal must NOT be PUBLISHED
        transfers = self.store.journal.list_transfers(project_id=project_id, run_id=run_id)
        for t in transfers:
            self.assertNotEqual(t.status, TransferStatus.PUBLISHED.value)

    def test_07_unversioned_retrieve_fails_closed_with_zero_or_two_latest_markers(self):
        """(7) Unversioned retrieve fails closed (DriveIntegrityError) with 0 or 2 latest markers."""
        project_id = "proj-p7"
        run_id = "run-r7"
        folder_id = self.store.resolve_target_folder_id(project_id, run_id, "checkpoints")

        sha1 = "1" * 64
        sha2 = "2" * 64
        data1 = b"CANDIDATE_DATA_1"
        data2 = b"CANDIDATE_DATA_2"

        # Case 7a: 0 latest markers (both have is_latest == 'false')
        self.mock_client.mock_files.append({
            "id": "file-p7-1",
            "name": "model.pt",
            "parent": folder_id,
            "content": data1,
            "size": len(data1),
            "md5Checksum": hashlib.md5(data1).hexdigest(),
            "appProperties": {
                "logical_path": "checkpoints/model.pt",
                "revision_id": f"sha256_{sha1}",
                "sha256": sha1,
                "is_latest": "false",
            },
        })
        self.mock_client.mock_files.append({
            "id": "file-p7-2",
            "name": "model.pt",
            "parent": folder_id,
            "content": data2,
            "size": len(data2),
            "md5Checksum": hashlib.md5(data2).hexdigest(),
            "appProperties": {
                "logical_path": "checkpoints/model.pt",
                "revision_id": f"sha256_{sha2}",
                "sha256": sha2,
                "is_latest": "false",
            },
        })

        dest_0 = Path(self.test_dir) / "dest_0.pt"
        with self.assertRaises(DriveIntegrityError) as ctx_0:
            self.store.retrieve(f"{project_id}/{run_id}/checkpoints/model.pt", dest_0)
        self.assertIn("Ambiguous unversioned Drive recovery", str(ctx_0.exception))
        self.assertFalse(dest_0.exists(), "Fails closed: destination file must not exist")

        # Case 7b: 2 latest markers (both have is_latest == 'true')
        self.mock_client.mock_files[-2]["appProperties"]["is_latest"] = "true"
        self.mock_client.mock_files[-1]["appProperties"]["is_latest"] = "true"

        dest_2 = Path(self.test_dir) / "dest_2.pt"
        with self.assertRaises(DriveIntegrityError) as ctx_2:
            self.store.retrieve(f"{project_id}/{run_id}/checkpoints/model.pt", dest_2)
        self.assertIn("Ambiguous unversioned Drive recovery", str(ctx_2.exception))
        self.assertIn("multiple (2)", str(ctx_2.exception))
        self.assertFalse(dest_2.exists(), "Fails closed: destination file must not exist")

    def test_08_legacy_metadata_files_still_recoverable(self):
        """(8) Legacy-metadata files (old keys logical_path/revision_id/is_latest/sha256) still recoverable."""
        project_id = "proj-p8"
        run_id = "run-r8"
        folder_id = self.store.resolve_target_folder_id(project_id, run_id, "checkpoints")

        data1 = b"LEGACY_DATA_HISTORICAL_REVISION"
        sha1 = hashlib.sha256(data1).hexdigest()
        data2 = b"LEGACY_DATA_LATEST_REVISION"
        sha2 = hashlib.sha256(data2).hexdigest()

        # Seed historical revision
        self.mock_client.mock_files.append({
            "id": "file-leg-1",
            "name": "model.pt",
            "parent": folder_id,
            "content": data1,
            "size": len(data1),
            "md5Checksum": hashlib.md5(data1).hexdigest(),
            "appProperties": {
                "logical_path": "checkpoints/model.pt",
                "revision_id": f"sha256_{sha1}",
                "sha256": sha1,
                "is_latest": "false",
                "project_id": project_id,
                "run_id": run_id,
                "superseded_by": f"{project_id}/{run_id}/checkpoints/model.pt@sha256_{sha2}",
            },
        })

        # Seed latest revision
        self.mock_client.mock_files.append({
            "id": "file-leg-2",
            "name": "model.pt",
            "parent": folder_id,
            "content": data2,
            "size": len(data2),
            "md5Checksum": hashlib.md5(data2).hexdigest(),
            "appProperties": {
                "logical_path": "checkpoints/model.pt",
                "revision_id": f"sha256_{sha2}",
                "sha256": sha2,
                "is_latest": "true",
                "project_id": project_id,
                "run_id": run_id,
            },
        })

        # Unversioned retrieve recovers the single latest legacy file
        dest_unv = Path(self.test_dir) / "legacy_unv.pt"
        self.store.retrieve(f"{project_id}/{run_id}/checkpoints/model.pt", dest_unv)
        self.assertEqual(dest_unv.read_bytes(), data2)

        # Versioned retrieve recovers specific historical revision
        dest_v1 = Path(self.test_dir) / "legacy_v1.pt"
        self.store.retrieve(f"{project_id}/{run_id}/checkpoints/model.pt@sha256_{sha1}", dest_v1)
        self.assertEqual(dest_v1.read_bytes(), data1)

        # Versioned retrieve recovers specific latest revision
        dest_v2 = Path(self.test_dir) / "legacy_v2.pt"
        self.store.retrieve(f"{project_id}/{run_id}/checkpoints/model.pt@sha256_{sha2}", dest_v2)
        self.assertEqual(dest_v2.read_bytes(), data2)

    def test_strict_drive_mock_max_property_count_enforced(self):
        """StrictDriveMock rejects requests attempting to set > 30 custom appProperties."""
        mock = StrictDriveMock()
        mock.mock_files.append({"id": "file-cap-1", "name": "test.pt", "appProperties": {}})

        # 31 properties
        props_31 = {f"k{i}": f"v{i}" for i in range(31)}
        status, _, body = mock.request(
            "PATCH",
            "https://www.googleapis.com/drive/v3/files/file-cap-1",
            data=json.dumps({"appProperties": props_31}).encode("utf-8"),
        )
        self.assertEqual(status, 400)
        err = json.loads(body.decode("utf-8"))
        self.assertIn("Maximum number of custom properties", err["error"]["message"])

    def test_strict_drive_mock_patch_merge_and_null_delete(self):
        """StrictDriveMock PATCH correctly merges properties and removes null keys."""
        mock = StrictDriveMock()
        mock.mock_files.append({
            "id": "file-merge-1",
            "name": "test.pt",
            "appProperties": {"keep": "1", "remove_me": "2"},
        })

        patch_data = {"appProperties": {"remove_me": None, "new_key": "3"}}
        status, _, body = mock.request(
            "PATCH",
            "https://www.googleapis.com/drive/v3/files/file-merge-1",
            data=json.dumps(patch_data).encode("utf-8"),
        )
        self.assertEqual(status, 200)
        resp = json.loads(body.decode("utf-8"))
        self.assertEqual(resp["appProperties"], {"keep": "1", "new_key": "3"})
        self.assertNotIn("remove_me", resp["appProperties"])

    def test_strict_drive_mock_honors_fields_parameter(self):
        """StrictDriveMock honors fields= query parameter and excludes unrequested fields."""
        mock = StrictDriveMock()
        mock.mock_files.append({
            "id": "file-fields-1",
            "name": "test.pt",
            "size": 1234,
            "md5Checksum": "abc123md5",
            "appProperties": {"a": "1"},
        })

        # Request only id and name
        status, _, body = mock.request(
            "GET",
            "https://www.googleapis.com/drive/v3/files/file-fields-1?fields=id,name",
        )
        self.assertEqual(status, 200)
        resp = json.loads(body.decode("utf-8"))
        self.assertIn("id", resp)
        self.assertIn("name", resp)
        self.assertNotIn("size", resp)
        self.assertNotIn("md5Checksum", resp)
        self.assertNotIn("appProperties", resp)

    def test_strict_drive_mock_missing_fields_injection(self):
        """StrictDriveMock supports injecting missing response fields."""
        mock = StrictDriveMock()
        mock.mock_files.append({
            "id": "file-drop-1",
            "name": "test.pt",
            "size": 100,
            "md5Checksum": "abc",
            "appProperties": {"key": "val"},
        })

        mock.drop_response_fields = {"md5Checksum", "appProperties"}
        status, _, body = mock.request(
            "GET",
            "https://www.googleapis.com/drive/v3/files/file-drop-1",
        )
        self.assertEqual(status, 200)
        resp = json.loads(body.decode("utf-8"))
        self.assertNotIn("md5Checksum", resp)
        self.assertNotIn("appProperties", resp)
        self.assertIn("id", resp)


if __name__ == "__main__":
    unittest.main()
