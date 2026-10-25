"""Unit tests for Google Drive compact metadata (124-byte limit) and schema v2 contract.

Validates:
- Validator boundary (exactly 124 UTF-8 bytes OK, 125 rejected)
- Non-string keys and values rejected
- Property count limit (max 30 per file)
- Multi-byte Unicode counted by UTF-8 byte length
- Long nested paths and long project/run IDs produce compliant properties
- Realistic vision-research/experiment-001 A->B->C publish emits only compliant properties
- Rejection (HTTP 400) by mock Drive client if any property exceeds 124 bytes
- Legacy-metadata backward compatibility (read and clean upgrade)
- Canonical successor ID reconstruction
- _remote_revision_state classification and error handling
"""
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from colab_adapter.errors import (
    ArtifactNotFoundError,
    DriveAuthenticationError,
    DriveIntegrityError,
    DriveQuotaExceededError,
    DriveTransferError,
)
from colab_adapter.storage import (
    ArtifactType,
    DriveCredentials,
    GoogleDriveArtifactStore,
)
from colab_adapter.storage.drive_metadata import (
    DRIVE_MAX_APP_PROPERTIES,
    DRIVE_PROPERTY_MAX_BYTES,
    SCHEMA_VERSION,
    STATE_HISTORICAL,
    STATE_LATEST,
    STATE_PENDING,
    build_revision_properties,
    canonical_successor_id,
    is_sha256_hex,
    path_digest,
    props_match_path,
    revision_state,
    successor_sha256,
    validate_drive_properties,
)
from tests.test_google_drive_store import MockDriveHttpClient


class Strict124ByteRejectingMockDriveClient(MockDriveHttpClient):
    """Mock Drive HTTP client that strictly rejects any appProperty exceeding 124 UTF-8 bytes with HTTP 400."""

    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[bytes] = None,
        timeout: float = 30.0,
    ) -> Tuple[int, Dict[str, str], bytes]:
        # 1. Enforce 124-byte limit on resumable upload session initiation POST
        if "uploadType=resumable" in url and method == "POST":
            if data:
                try:
                    meta = json.loads(data.decode("utf-8"))
                    props = meta.get("appProperties")
                    if props and isinstance(props, dict):
                        if len(props) > DRIVE_MAX_APP_PROPERTIES:
                            return 400, {"content-type": "application/json"}, json.dumps({
                                "error": {"message": f"Too many appProperties: {len(props)} > {DRIVE_MAX_APP_PROPERTIES}"}
                            }).encode("utf-8")
                        for k, v in props.items():
                            if not isinstance(k, str) or not isinstance(v, str):
                                return 400, {"content-type": "application/json"}, json.dumps({
                                    "error": {"message": f"Invalid property type: key={type(k)}, val={type(v)}"}
                                }).encode("utf-8")
                            b_len = len(k.encode("utf-8")) + len(v.encode("utf-8"))
                            if b_len > DRIVE_PROPERTY_MAX_BYTES:
                                return 400, {"content-type": "application/json"}, json.dumps({
                                    "error": {"message": f"appProperty '{k}' exceeds {DRIVE_PROPERTY_MAX_BYTES} UTF-8 bytes ({b_len} bytes)"}
                                }).encode("utf-8")
                except json.JSONDecodeError:
                    pass

        # 2. Enforce 124-byte limit on file PATCH requests
        if re.search(r"drive/v3/files/[^/?]+", url) and method == "PATCH":
            if data:
                try:
                    patch_meta = json.loads(data.decode("utf-8"))
                    props = patch_meta.get("appProperties")
                    if props and isinstance(props, dict):
                        for k, v in props.items():
                            if not isinstance(k, str):
                                return 400, {"content-type": "application/json"}, json.dumps({
                                    "error": {"message": "Key must be string"}
                                }).encode("utf-8")
                            if v is not None:
                                if not isinstance(v, str):
                                    return 400, {"content-type": "application/json"}, json.dumps({
                                        "error": {"message": "Value must be string or null"}
                                    }).encode("utf-8")
                                b_len = len(k.encode("utf-8")) + len(v.encode("utf-8"))
                                if b_len > DRIVE_PROPERTY_MAX_BYTES:
                                    return 400, {"content-type": "application/json"}, json.dumps({
                                        "error": {"message": f"PATCH appProperty '{k}' exceeds {DRIVE_PROPERTY_MAX_BYTES} UTF-8 bytes ({b_len} bytes)"}
                                    }).encode("utf-8")
                except json.JSONDecodeError:
                    pass

        return super().request(method, url, headers=headers, data=data, timeout=timeout)


class TestDriveMetadataLimitsAndContract(unittest.TestCase):
    """Offline unit tests for Drive property limits, schema v2 contract, and store integration."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.staging_dir = self.test_dir / "staging"
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.mock_client = Strict124ByteRejectingMockDriveClient()
        self.creds = DriveCredentials(
            client_id="test_client_id",
            client_secret="test_client_secret",
            refresh_token="test_refresh_token",
            access_token="test_access_token",
            token_expiry=9999999999.0,
        )
        self.credentials_file = self.test_dir / "creds.json"
        self.creds.save(self.credentials_file)
        self.store = GoogleDriveArtifactStore(
            credentials=self.creds,
            credentials_file=self.credentials_file,
            local_staging_dir=self.staging_dir,
            http_client=self.mock_client,
            chunk_size=1024 * 1024,
            initial_backoff_sec=0.01,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_validator_boundary_exactly_124_bytes_ok_125_rejected(self):
        """1. Validator boundary: exactly 124 UTF-8 bytes passes, 125 bytes raises DriveTransferError."""
        # 1 byte key + 123 bytes value = 124 bytes -> OK
        valid_props = {"k": "v" * 123}
        validate_drive_properties(valid_props)

        # 1 byte key + 124 bytes value = 125 bytes -> Rejected
        invalid_props = {"k": "v" * 124}
        with self.assertRaises(DriveTransferError) as ctx:
            validate_drive_properties(invalid_props)
        self.assertIn("exceeds 124 UTF-8 bytes (125 bytes)", str(ctx.exception))
        # Ensure value is NOT leaked in error text
        self.assertNotIn("vvv", str(ctx.exception))

        # Split 62 + 62 = 124 bytes -> OK
        validate_drive_properties({"k" * 62: "v" * 62})

        # Split 62 + 63 = 125 bytes -> Rejected
        with self.assertRaises(DriveTransferError):
            validate_drive_properties({"k" * 62: "v" * 63})

    def test_02_validator_non_str_rejected(self):
        """2. Non-string keys and values, or empty keys, are rejected with DriveTransferError."""
        with self.assertRaises(DriveTransferError):
            validate_drive_properties({"key": 123})  # int value

        with self.assertRaises(DriveTransferError):
            validate_drive_properties({"key": True})  # bool value

        with self.assertRaises(DriveTransferError):
            validate_drive_properties({"key": None})  # None value

        with self.assertRaises(DriveTransferError):
            validate_drive_properties({"key": ["list"]})  # list value

        with self.assertRaises(DriveTransferError):
            validate_drive_properties({123: "val"})  # int key

        with self.assertRaises(DriveTransferError):
            validate_drive_properties({"": "val"})  # empty key

    def test_03_validator_count_limit(self):
        """3. Property count cap: 30 properties allowed, 31 rejected."""
        props_30 = {f"k_{i}": f"v_{i}" for i in range(30)}
        validate_drive_properties(props_30)

        props_31 = {f"k_{i}": f"v_{i}" for i in range(31)}
        with self.assertRaises(DriveTransferError) as ctx:
            validate_drive_properties(props_31)
        self.assertIn("maximum count", str(ctx.exception))

    def test_04_validator_unicode_multi_byte_counted_in_bytes(self):
        """4. Unicode multi-byte characters are counted by UTF-8 byte length, not char count."""
        # '€' is 3 bytes in UTF-8
        # key "k" (1 byte) + 41 * 3 bytes (123 bytes) = 124 bytes -> OK
        validate_drive_properties({"k": "€" * 41})

        # key "k" (1 byte) + 41 * 3 bytes + 'a' (1 byte) = 125 bytes -> Rejected
        with self.assertRaises(DriveTransferError):
            validate_drive_properties({"k": "€" * 41 + "a"})

        # Emoji '🔥' is 4 bytes in UTF-8
        # key "k" (1 byte) + 30 * 4 bytes (120 bytes) + "abc" (3 bytes) = 124 bytes -> OK
        validate_drive_properties({"k": "🔥" * 30 + "abc"})

        # key "k" (1 byte) + 30 * 4 bytes (120 bytes) + "abcd" (4 bytes) = 125 bytes -> Rejected
        with self.assertRaises(DriveTransferError):
            validate_drive_properties({"k": "🔥" * 30 + "abcd"})

    def test_05_long_nested_path_and_ids_produce_compliant_props(self):
        """5. Very long nested paths and IDs produce fixed-size, compliant schema v2 properties."""
        long_project = "vision-research-team-alpha-distributed-cluster-project-id" * 2
        long_run = "experiment-run-stage-3-finetuning-with-gradient-accumulation-and-lora" * 2
        long_path = "nested/deeply/within/checkpoints/snapshots/submodules/backbone/" + ("dense_layer_" * 15) + ".pt"
        sha = hashlib.sha256(b"TEST_WEIGHTS").hexdigest()

        # Build properties
        props = build_revision_properties(
            sha256=sha,
            project_id=long_project,
            run_id=long_run,
            logical_path=long_path,
            is_latest=True,
        )

        # Verify all properties comply with 124-byte limit
        validate_drive_properties(props)
        self.assertEqual(props["schema"], "2")
        self.assertEqual(props["sha256"], sha)
        self.assertEqual(props["is_latest"], "true")
        self.assertEqual(props["path_sha256"], path_digest(long_project, long_run, long_path))
        self.assertNotIn("next_sha256", props)

        for k, v in props.items():
            b_size = len(k.encode("utf-8")) + len(v.encode("utf-8"))
            self.assertLessEqual(b_size, DRIVE_PROPERTY_MAX_BYTES)

    def test_06_realistic_publish_a_b_c_emits_only_compliant_props(self):
        """6. Realistic vision-research/experiment-001 A->B->C publish emits only compliant properties.

        Strict mock Drive client rejects any POST or PATCH with properties > 124 bytes.
        """
        project_id = "vision-research"
        run_id = "experiment-001"
        file_path = self.test_dir / "weights.pt"

        # Publish Revision A
        data_a = b"WEIGHTS_REV_A_DATA" * 50
        file_path.write_bytes(data_a)
        art_a = self.store.publish(file_path, project_id, run_id, ArtifactType.CHECKPOINT.value)

        # Publish Revision B
        data_b = b"WEIGHTS_REV_B_DATA_DIFFERENT" * 50
        file_path.write_bytes(data_b)
        art_b = self.store.publish(file_path, project_id, run_id, ArtifactType.CHECKPOINT.value)

        # Publish Revision C
        data_c = b"WEIGHTS_REV_C_DATA_FINAL" * 50
        file_path.write_bytes(data_c)
        art_c = self.store.publish(file_path, project_id, run_id, ArtifactType.CHECKPOINT.value)

        # Verify all three remote files have only compliant properties
        mf_map = {f["id"]: f for f in self.mock_client.mock_files if f.get("id")}
        mf_a = mf_map[art_a.remote_file_id]
        mf_b = mf_map[art_b.remote_file_id]
        mf_c = mf_map[art_c.remote_file_id]

        expected_path_sha = path_digest(project_id, run_id, "checkpoints/weights.pt")

        # Check Revision A
        props_a = mf_a["appProperties"]
        validate_drive_properties(props_a)
        self.assertEqual(props_a["schema"], SCHEMA_VERSION)
        self.assertEqual(props_a["sha256"], art_a.sha256)
        self.assertEqual(props_a["path_sha256"], expected_path_sha)
        self.assertEqual(props_a["is_latest"], "false")
        self.assertEqual(props_a["next_sha256"], art_b.sha256)
        self.assertEqual(revision_state(props_a), STATE_HISTORICAL)
        self.assertNotIn("superseded_by", props_a)
        self.assertNotIn("logical_path", props_a)

        # Check Revision B
        props_b = mf_b["appProperties"]
        validate_drive_properties(props_b)
        self.assertEqual(props_b["schema"], SCHEMA_VERSION)
        self.assertEqual(props_b["sha256"], art_b.sha256)
        self.assertEqual(props_b["path_sha256"], expected_path_sha)
        self.assertEqual(props_b["is_latest"], "false")
        self.assertEqual(props_b["next_sha256"], art_c.sha256)
        self.assertEqual(revision_state(props_b), STATE_HISTORICAL)
        self.assertNotIn("superseded_by", props_b)
        self.assertNotIn("logical_path", props_b)

        # Check Revision C
        props_c = mf_c["appProperties"]
        validate_drive_properties(props_c)
        self.assertEqual(props_c["schema"], SCHEMA_VERSION)
        self.assertEqual(props_c["sha256"], art_c.sha256)
        self.assertEqual(props_c["path_sha256"], expected_path_sha)
        self.assertEqual(props_c["is_latest"], "true")
        self.assertEqual(revision_state(props_c), STATE_LATEST)
        self.assertNotIn("next_sha256", props_c)
        self.assertNotIn("superseded_by", props_c)
        self.assertNotIn("logical_path", props_c)

        # All property pairs strictly <= 124 bytes
        for p in (props_a, props_b, props_c):
            for k, v in p.items():
                self.assertLessEqual(len(k.encode("utf-8")) + len(v.encode("utf-8")), DRIVE_PROPERTY_MAX_BYTES)

    def test_07_legacy_metadata_read_compatibility_and_clean_upgrade(self):
        """7. Legacy properties are read correctly and stripped on subsequent coordination PATCH."""
        project_id = "legacy_proj"
        run_id = "legacy_run"
        folder_id = self.store.resolve_target_folder_id(project_id, run_id, "checkpoints")

        data_legacy = b"LEGACY_REVISION_DATA"
        sha_legacy = hashlib.sha256(data_legacy).hexdigest()

        # Seed mock Drive with a legacy file containing old long keys
        legacy_props = {
            "sha256": sha_legacy,
            "logical_path": "checkpoints/weights.pt",
            "revision_id": f"sha256_{sha_legacy}",
            "project_id": project_id,
            "run_id": run_id,
            "is_latest": "true",
        }
        legacy_file_id = "legacy-file-001"
        self.mock_client.mock_files.append({
            "id": legacy_file_id,
            "name": "weights.pt",
            "parent": folder_id,
            "size": len(data_legacy),
            "md5Checksum": hashlib.md5(data_legacy).hexdigest(),
            "appProperties": dict(legacy_props),
            "content": data_legacy,
        })
        self.mock_client.committed_files[legacy_file_id] = data_legacy

        # Verify props_match_path identifies legacy path
        self.assertTrue(props_match_path(legacy_props, project_id, run_id, "checkpoints/weights.pt"))
        self.assertFalse(props_match_path(legacy_props, project_id, run_id, "checkpoints/other.pt"))

        # Verify successor_sha256 extracts digest from legacy superseded_by
        legacy_historical_props = {
            **legacy_props,
            "is_latest": "false",
            "superseded_by": f"{project_id}/{run_id}/checkpoints/weights.pt@sha256_{sha_legacy}",
        }
        self.assertEqual(successor_sha256(legacy_historical_props), sha_legacy)
        self.assertEqual(revision_state(legacy_historical_props), STATE_HISTORICAL)

        # Unversioned retrieve succeeds from legacy file
        dest_unv = self.test_dir / "legacy_retrieved.pt"
        self.store.retrieve(f"{project_id}/{run_id}/checkpoints/weights.pt", dest_unv)
        self.assertEqual(dest_unv.read_bytes(), data_legacy)

        # Publish new revision B over the legacy revision
        data_new = b"NEW_REVISION_DATA_CLEAN"
        file_path = self.test_dir / "weights.pt"
        file_path.write_bytes(data_new)
        art_new = self.store.publish(file_path, project_id, run_id, ArtifactType.CHECKPOINT.value)

        # Verify legacy file was demoted and legacy keys stripped
        mf_legacy = next(f for f in self.mock_client.mock_files if f.get("id") == legacy_file_id)
        props_after_demote = mf_legacy["appProperties"]
        self.assertEqual(props_after_demote.get("is_latest"), "false")
        self.assertEqual(props_after_demote.get("schema"), SCHEMA_VERSION)
        self.assertEqual(props_after_demote.get("next_sha256"), art_new.sha256)

        # Legacy long keys MUST have been deleted
        self.assertNotIn("logical_path", props_after_demote)
        self.assertNotIn("revision_id", props_after_demote)
        self.assertNotIn("project_id", props_after_demote)
        self.assertNotIn("run_id", props_after_demote)
        self.assertNotIn("superseded_by", props_after_demote)

    def test_08_canonical_successor_id_reconstruction(self):
        """8. canonical_successor_id accurately reconstructs the full canonical ID from schema v2."""
        proj = "my_proj"
        run = "my_run"
        path = "checkpoints/model.pt"
        sha = "a" * 64

        canon = canonical_successor_id(proj, run, path, sha)
        self.assertEqual(canon, f"{proj}/{run}/{path}@sha256_{sha}")

        # Handles leading slash
        canon_slash = canonical_successor_id(proj, run, "/checkpoints/model.pt", sha.upper())
        self.assertEqual(canon_slash, f"{proj}/{run}/{path}@sha256_{sha}")

    def test_09_remote_revision_state_method(self):
        """9. _remote_revision_state accurately queries remote file and classifies state."""
        # 1. State 'latest'
        file_id_latest = "test-fid-latest"
        self.mock_client.mock_files.append({
            "id": file_id_latest,
            "name": "model.pt",
            "appProperties": {
                "schema": "2",
                "sha256": "1" * 64,
                "is_latest": "true",
            },
        })
        st, pr = self.store._remote_revision_state(file_id_latest)
        self.assertEqual(st, STATE_LATEST)
        self.assertEqual(pr["is_latest"], "true")

        # 2. State 'historical'
        file_id_hist = "test-fid-hist"
        self.mock_client.mock_files.append({
            "id": file_id_hist,
            "name": "model.pt",
            "appProperties": {
                "schema": "2",
                "sha256": "2" * 64,
                "is_latest": "false",
                "next_sha256": "3" * 64,
            },
        })
        st_h, pr_h = self.store._remote_revision_state(file_id_hist)
        self.assertEqual(st_h, STATE_HISTORICAL)
        self.assertEqual(pr_h["next_sha256"], "3" * 64)

        # 3. State 'pending'
        file_id_pend = "test-fid-pend"
        self.mock_client.mock_files.append({
            "id": file_id_pend,
            "name": "model.pt",
            "appProperties": {
                "schema": "2",
                "sha256": "4" * 64,
                "is_latest": "false",
            },
        })
        st_p, pr_p = self.store._remote_revision_state(file_id_pend)
        self.assertEqual(st_p, STATE_PENDING)

        # 4. Error classification: 404 raises ArtifactNotFoundError
        with self.assertRaises(ArtifactNotFoundError):
            self.store._remote_revision_state("non-existent-fid")

    def test_10_validate_before_request_rejects_invalid_metadata_early(self):
        """10. validate_drive_properties is called before HTTP requests and blocks invalid metadata."""
        # Test initiate_resumable_upload with oversized property
        with self.assertRaises(DriveTransferError):
            self.store._initiate_resumable_upload(
                filename="model.pt",
                parent_id="folder-1",
                total_size=100,
                app_properties={"k": "v" * 125},
            )

        # Test _patch_remote_file_app_properties with oversized property
        with self.assertRaises(DriveTransferError):
            self.store._patch_remote_file_app_properties(
                remote_id="fid-1",
                app_properties={"k": "v" * 125},
            )

        # Test _patch_remote_file_app_properties with non-string non-None value
        with self.assertRaises(DriveTransferError):
            self.store._patch_remote_file_app_properties(
                remote_id="fid-1",
                app_properties={"k": 999},  # type: ignore
            )


if __name__ == "__main__":
    unittest.main()
