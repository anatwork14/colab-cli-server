"""Tests for Milestone M6: Artifacts, safe checkpoint verification, telemetry and resume.

Verifies:
- Q14: Artifact verification: missing, truncated, empty, and checksum-mismatched files rejected.
- Multi-tier verification: existence, size, SHA256, and ZIP checkpoint structure.
- Staged download with atomic commit; staged cleanup on failure.
- Q12: Credential redaction in telemetry events and error messages.
- Q11: Accurate classification of INTERRUPTED, CLEANUP_FAILED, and FAILED states.
- Explicit resume planning from verified checkpoints.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

from colab_adapter.artifacts import (
    ArtifactManager,
    ArtifactManifest,
    compute_file_sha256,
    verify_local_artifact,
)
from colab_adapter.errors import ArtifactValidationError
from colab_adapter.project_manifest import ProjectSpec
from colab_adapter.telemetry import (
    TelemetryEmitter,
    plan_resume_from_checkpoint,
    record_interrupted_run,
)


class TestArtifactsAndTelemetry(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.artifacts_dir = Path(self.tmp_dir) / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = Path(self.tmp_dir) / "telemetry.jsonl"

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_q14_artifact_verification_missing_file_rejected(self):
        """Q14: Missing artifact file raises ArtifactValidationError."""
        missing = Path(self.tmp_dir) / "nonexistent.pt"
        with self.assertRaises(ArtifactValidationError) as ctx:
            verify_local_artifact(missing)
        self.assertIn("not found", str(ctx.exception))

    def test_q14_artifact_verification_empty_or_truncated_file_rejected(self):
        """Q14: Empty or truncated artifact raises ArtifactValidationError."""
        empty_file = Path(self.tmp_dir) / "empty.bin"
        empty_file.write_bytes(b"")
        with self.assertRaises(ArtifactValidationError) as ctx:
            verify_local_artifact(empty_file, min_size_bytes=10)
        self.assertIn("truncated or empty", str(ctx.exception))

    def test_q14_artifact_verification_sha256_mismatch_rejected(self):
        """Q14: Checksum mismatch raises ArtifactValidationError."""
        data_file = Path(self.tmp_dir) / "weights.bin"
        data_file.write_bytes(b"random-model-weights-data")
        wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000"

        with self.assertRaises(ArtifactValidationError) as ctx:
            verify_local_artifact(data_file, expected_sha256=wrong_hash)
        self.assertIn("checksum mismatch", str(ctx.exception))

    def test_artifact_verification_valid_checksum_and_manifest(self):
        """Valid file with matching SHA256 passes verification and produces ArtifactManifest."""
        payload = b"valid-model-weights-payload-for-test"
        data_file = Path(self.tmp_dir) / "weights.bin"
        data_file.write_bytes(payload)
        expected_hash = hashlib.sha256(payload).hexdigest()

        manifest = verify_local_artifact(data_file, expected_sha256=expected_hash)
        self.assertEqual(manifest.rel_path, "weights.bin")
        self.assertEqual(manifest.sha256, expected_hash)
        self.assertEqual(manifest.size_bytes, len(payload))
        self.assertEqual(manifest.verification_level, 2)

    def test_checkpoint_zip_archive_structure_inspection(self):
        """Validates checkpoint ZIP archive structure checks."""
        ckpt_path = Path(self.tmp_dir) / "model.pt"
        # Create valid zip checkpoint fixture
        with zipfile.ZipFile(str(ckpt_path), "w") as zf:
            zf.writestr("archive/data.pkl", b"test_pickle_data")
            zf.writestr("archive/version", b"3")

        manifest = verify_local_artifact(ckpt_path, is_checkpoint=True)
        self.assertTrue(manifest.is_checkpoint)
        self.assertIn("archive/data.pkl", manifest.metadata.get("archive_entries", []))

        # Test corrupt zip archive raises ArtifactValidationError
        bad_ckpt = Path(self.tmp_dir) / "corrupt.pt"
        bad_ckpt.write_bytes(b"PK\x03\x04corrupted_header_data_not_valid_zip")
        with self.assertRaises(ArtifactValidationError):
            verify_local_artifact(bad_ckpt, is_checkpoint=True)

    def test_artifact_manager_staged_download_and_atomic_commit(self):
        """ArtifactManager downloads to temporary staging file and atomically commits upon verification."""
        mock_transport = MagicMock()
        payload = b"downloaded-model-weights"
        expected_hash = hashlib.sha256(payload).hexdigest()

        def fake_download(session_id, remote_path, local_path):
            Path(local_path).write_bytes(payload)

        mock_transport.download_file.side_effect = fake_download
        manager = ArtifactManager(transport=mock_transport, local_dest_dir=self.artifacts_dir)

        manifest = manager.download_and_verify(
            session_id="colab-v2-sess-1",
            remote_path="/content/output/model.pt",
            expected_sha256=expected_hash,
        )

        final_file = self.artifacts_dir / "model.pt"
        self.assertTrue(final_file.exists())
        self.assertEqual(final_file.read_bytes(), payload)
        self.assertEqual(manifest.sha256, expected_hash)

        # Ensure no temporary staging files left behind
        temp_files = list(self.artifacts_dir.glob("*.tmp.*"))
        self.assertEqual(temp_files, [])

    def test_artifact_manager_staged_download_cleans_up_on_failure(self):
        """On verification failure, staged temporary file is deleted and target is not created."""
        mock_transport = MagicMock()

        def fake_download_bad(session_id, remote_path, local_path):
            Path(local_path).write_bytes(b"bad-data")

        mock_transport.download_file.side_effect = fake_download_bad
        manager = ArtifactManager(transport=mock_transport, local_dest_dir=self.artifacts_dir)

        with self.assertRaises(ArtifactValidationError):
            manager.download_and_verify(
                session_id="colab-v2-sess-1",
                remote_path="/content/output/model.pt",
                expected_sha256="expected_different_sha256",
            )

        final_file = self.artifacts_dir / "model.pt"
        self.assertFalse(final_file.exists())
        temp_files = list(self.artifacts_dir.glob("*.tmp.*"))
        self.assertEqual(temp_files, [])

    def test_q12_telemetry_credential_redaction(self):
        """Q12: TelemetryEmitter deeply scrubs seeded credentials from metrics, details, and errors."""
        emitter = TelemetryEmitter(
            ledger_path=self.ledger_path,
            run_id="run-test-1",
            project_id="proj-redaction",
        )

        # Seeded credential strings
        oauth_token = "ya29.a0AfH6SMAAAAAAA_test_token_secret_12345"
        auth_code = "4/0AX4XfWhhhhh_auth_code_secret_value_12345"
        private_key = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----"

        event = emitter.emit(
            event_type="error",
            epoch=1,
            step=50,
            metrics={"loss": 0.42, "api_token": oauth_token},
            details={
                "error_trace": f"Failed with {auth_code} and key: {private_key}",
                "secret_key": "my_top_secret_pass",
            },
        )

        # Check event returned in-memory
        raw_dump = json.dumps(event)
        self.assertNotIn(oauth_token, raw_dump)
        self.assertNotIn(auth_code, raw_dump)
        self.assertNotIn("-----BEGIN RSA PRIVATE KEY-----", raw_dump)
        self.assertNotIn("my_top_secret_pass", raw_dump)

        # Check persisted JSONL on disk
        persisted = emitter.get_events(run_id="run-test-1")
        self.assertEqual(len(persisted), 1)
        persisted_dump = json.dumps(persisted[0])
        self.assertNotIn(oauth_token, persisted_dump)
        self.assertNotIn(auth_code, persisted_dump)
        self.assertNotIn("my_top_secret_pass", persisted_dump)

    def test_q11_accurate_interrupted_run_classification(self):
        """Q11: Accurate classification of INTERRUPTED, CLEANUP_FAILED, and FAILED states."""
        # 1. KeyboardInterrupt -> INTERRUPTED
        rec_int = record_interrupted_run(KeyboardInterrupt("User cancelled"))
        self.assertEqual(rec_int["status"], "INTERRUPTED")
        self.assertEqual(rec_int["primary_error_type"], "KeyboardInterrupt")

        # 2. Cleanup failure alongside primary failure -> CLEANUP_FAILED without masking
        rec_clean = record_interrupted_run(
            RuntimeError("Remote CUDA out of memory"),
            cleanup_error="Stop command timed out after 60s",
        )
        self.assertEqual(rec_clean["status"], "CLEANUP_FAILED")
        self.assertEqual(rec_clean["primary_error_type"], "RuntimeError")
        self.assertEqual(rec_clean["cleanup_error"], "Stop command timed out after 60s")

        # 3. Standard execution error -> FAILED
        rec_fail = record_interrupted_run(ValueError("Invalid argument"))
        self.assertEqual(rec_fail["status"], "FAILED")
        self.assertEqual(rec_fail["primary_error_type"], "ValueError")

    def test_plan_resume_from_checkpoint(self):
        """Explicit resume planning generates target destination and CLI arguments."""
        ckpt = ArtifactManifest(
            rel_path="checkpoint_epoch_10.pt",
            abs_path=str(self.artifacts_dir / "checkpoint_epoch_10.pt"),
            size_bytes=1024,
            sha256="abc123sha256hash",
            verified_at="2026-10-24T00:00:00Z",
            verification_level=2,
            is_checkpoint=True,
        )
        spec = ProjectSpec(
            name="cifar_exp",
            root=Path(self.tmp_dir),
            entrypoint="train.py",
            args=["--batch-size", "64"],
        )

        plan = plan_resume_from_checkpoint(ckpt, spec)
        self.assertEqual(plan["source_checkpoint"]["sha256"], "abc123sha256hash")
        self.assertIn("--resume-from", plan["resume_arguments"])
        self.assertIn("/content/colab-workspaces/cifar_exp/checkpoints/checkpoint_epoch_10.pt", plan["resume_arguments"])
        self.assertIn("--batch-size", plan["resume_arguments"])


if __name__ == "__main__":
    unittest.main()
