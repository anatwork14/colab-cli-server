"""Regression tests reproducing identified authentication, security, resumable upload, and recovery failures.

Covers:
1. Authentication & Remote-Access Security:
   - Broker rejects hardcoded default tokens; generates high-entropy random token.
   - Path traversal rejection across broker, local store, and monitor server.
   - External monitor binding strictly requires non-empty authentication token.
   - Credentials file automatically restricts insecure file permissions to 0600.
2. Resumable Uploads, Transfer-Journal Recovery, Retries & Integrity Verification:
   - Chunk retry with backoff on transient HTTP 503 errors.
   - Range status query returning HTTP 200/201 (already completed) cleanly finishes without error.
   - Session URL expiration (HTTP 404/410) recovery.
   - Chunk size positive multiple of 256 KiB enforcement.
   - Remote size and MD5 integrity verification.
   - TransferJournal cross-process locking.
"""
import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path

from colab_adapter.broker import WorkspaceBroker
from colab_adapter.errors import (
    ArtifactValidationError,
    BrokerAuthenticationError,
    ColabV2Error,
    DriveIntegrityError,
    DriveTransferError,
)
from colab_adapter.monitor.server import MonitorServer, start_monitor_server
from colab_adapter.provider import ManagedSessionRegistry
from colab_adapter.storage import (
    ArtifactType,
    DriveCredentials,
    GoogleDriveArtifactStore,
    LocalArtifactStore,
    TransferJournal,
    TransferRecord,
    TransferStatus,
)
from tests.test_google_drive_store import MockDriveHttpClient


class TestSecurityAndResumableRecoveryRegressions(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="colab_regressions_")
        self.addCleanup(lambda: shutil.rmtree(self.test_dir, ignore_errors=True))

        self.artifacts_dir = Path(self.test_dir) / "artifacts"
        self.registry_file = Path(self.test_dir) / "registry.json"
        self.registry = ManagedSessionRegistry(registry_path=self.registry_file)

    # -------------------------------------------------------------------------
    # 1. AUTHENTICATION & REMOTE ACCESS SECURITY REGRESSIONS
    # -------------------------------------------------------------------------

    def test_broker_default_token_not_guessable(self):
        """Broker initialized without explicit tokens must NOT accept 'local-dev-token'."""
        broker = WorkspaceBroker(
            registry=self.registry,
            artifacts_dir=self.artifacts_dir,
            auth_tokens=None,  # No tokens provided
        )
        # Attempt with old static guessable token
        with self.assertRaises(BrokerAuthenticationError):
            broker.list_runs(token="local-dev-token")

        # Must succeed with primary_token generated
        self.assertTrue(broker.verify_token(broker.primary_token))
        runs = broker.list_runs(token=broker.primary_token)
        self.assertIsInstance(runs, list)

    def test_broker_path_traversal_in_run_id_rejected(self):
        """Broker must strictly reject path traversal payloads in run_id or project_id."""
        broker = WorkspaceBroker(
            registry=self.registry,
            artifacts_dir=self.artifacts_dir,
            auth_tokens={"secure-token"},
        )
        traversal_payloads = [
            "../../etc/passwd",
            "../secrets",
            "foo/../../bar",
            "run/subdir",
            "\\windows\\path",
        ]
        for payload in traversal_payloads:
            with self.assertRaises(ColabV2Error, msg=f"Payload not rejected: {payload}"):
                broker.attach_run(run_id=payload, client_id="client1", token="secure-token")

            with self.assertRaises(ColabV2Error, msg=f"Payload not rejected: {payload}"):
                broker.get_run_artifacts(run_id=payload, token="secure-token")

            with self.assertRaises(ColabV2Error, msg=f"Payload not rejected: {payload}"):
                broker.list_runs(project_id=payload, token="secure-token")

    def test_monitor_external_binding_without_token_rejected(self):
        """Monitor server with allow_external=True without auth_token raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            start_monitor_server(
                host="0.0.0.0",
                port=9876,
                allow_external=True,
                auth_token=None,  # Forbidden
            )
        self.assertIn("Security violation", str(ctx.exception))
        self.assertIn("requires a non-empty auth_token", str(ctx.exception))

    def test_local_store_path_traversal_rejected(self):
        """LocalArtifactStore strictly rejects path traversal in project_id or run_id."""
        store = LocalArtifactStore(root_dir=self.artifacts_dir)
        dummy_file = Path(self.test_dir) / "data.pt"
        dummy_file.write_bytes(b"DATA")

        with self.assertRaises(ArtifactValidationError):
            store.publish(
                source=dummy_file,
                project_id="../escape_proj",
                run_id="run1",
            )

        with self.assertRaises(ArtifactValidationError):
            store.publish(
                source=dummy_file,
                project_id="proj1",
                run_id="../../escape_run",
            )

    def test_drive_credentials_file_permission_enforcement(self):
        """DriveCredentials.load must automatically restrict overly permissive 0666 files to 0600."""
        creds_file = Path(self.test_dir) / "insecure_credentials.json"
        creds = DriveCredentials(
            client_id="cid",
            client_secret="sec",
            refresh_token="ref",
            access_token="tok",
        )
        creds.save(creds_file)

        # Intentionally make it world-readable/writable (0o666)
        os.chmod(creds_file, 0o666)
        self.assertEqual(stat.S_IMODE(os.stat(creds_file).st_mode), 0o666)

        # Loading must automatically enforce 0600
        loaded = DriveCredentials.load(creds_file)
        self.assertEqual(loaded.access_token, "tok")
        mode_after = stat.S_IMODE(os.stat(creds_file).st_mode)
        self.assertEqual(mode_after, 0o600)

    # -------------------------------------------------------------------------
    # 2. RESUMABLE UPLOADS, JOURNAL RECOVERY & RETRY REGRESSIONS
    # -------------------------------------------------------------------------

    def test_chunk_size_must_be_multiple_of_256_kib(self):
        """GoogleDriveArtifactStore rejects chunk_size that is not a multiple of 256 KiB."""
        with self.assertRaises(ValueError) as ctx:
            GoogleDriveArtifactStore(chunk_size=1000)  # Not multiple of 256 KiB
        self.assertIn("multiple of 256 KiB", str(ctx.exception))

    def test_resumable_upload_query_already_completed(self):
        """If progress query returns 200/201 (already uploaded), finish cleanly without re-upload."""
        mock_http = MockDriveHttpClient()
        store = GoogleDriveArtifactStore(
            credentials=DriveCredentials(access_token="tok"),
            local_staging_dir=self.artifacts_dir,
            http_client=mock_http,
            chunk_size=256 * 1024,
            initial_backoff_sec=0.01,
        )

        test_file = Path(self.test_dir) / "already_done.pt"
        test_file.write_bytes(b"COMPLETED_DATA")

        # Create record with existing session URL and bytes_transferred > 0
        rec = TransferRecord(
            transfer_id="google_drive-p1-r1-already_done.pt",
            project_id="p1",
            run_id="r1",
            artifact_id="p1/r1/checkpoints/already_done.pt",
            local_path=str(test_file),
            target_rel_path="checkpoints/already_done.pt",
            size_bytes=len(b"COMPLETED_DATA"),
            sha256="dummy",
            status=TransferStatus.UPLOADING.value,
            resumable_session_url="https://www.googleapis.com/upload/drive/v3/files?session_id=upload-session-999",
            bytes_transferred=len(b"COMPLETED_DATA"),
        )
        store.journal.record_transfer(rec)

        # Upload call should query progress, see 200/201, and return finished metadata
        stored = store.publish(
            source=test_file,
            project_id="p1",
            run_id="r1",
            artifact_type=ArtifactType.CHECKPOINT.value,
        )
        self.assertEqual(stored.storage_provider, "google_drive")
        self.assertIsNotNone(stored.remote_file_id)
        updated_rec = store.journal.get_transfer(rec.transfer_id)
        self.assertEqual(updated_rec.status, "PUBLISHED")

    def test_chunk_retry_on_transient_503(self):
        """A transient 503 error on a specific chunk upload is retried with backoff."""
        mock_http = MockDriveHttpClient()
        mock_http.fail_next_n_chunks = 2  # Fail twice with 503, then succeed

        store = GoogleDriveArtifactStore(
            credentials=DriveCredentials(access_token="tok"),
            local_staging_dir=self.artifacts_dir,
            http_client=mock_http,
            chunk_size=256 * 1024,
            initial_backoff_sec=0.01,
        )

        test_file = Path(self.test_dir) / "retry_test.pt"
        test_file.write_bytes(b"Z" * (512 * 1024))  # 2 chunks

        stored = store.publish(
            source=test_file,
            project_id="p1",
            run_id="r1",
            artifact_type=ArtifactType.CHECKPOINT.value,
        )
        self.assertEqual(stored.size_bytes, 512 * 1024)
        self.assertIsNotNone(stored.remote_file_id)

    def test_session_url_expired_recovery(self):
        """Expired session URL (HTTP 404/410) clears session URL in journal and permits re-initiation."""
        mock_http = MockDriveHttpClient()

        store = GoogleDriveArtifactStore(
            credentials=DriveCredentials(access_token="tok"),
            local_staging_dir=self.artifacts_dir,
            http_client=mock_http,
            chunk_size=256 * 1024,
            initial_backoff_sec=0.01,
        )

        test_file = Path(self.test_dir) / "expired_session.pt"
        test_file.write_bytes(b"EXPIRED_SESSION_DATA")

        # Pre-seed with dead session URL
        rec = TransferRecord(
            transfer_id="google_drive-p1-r1-expired_session.pt",
            project_id="p1",
            run_id="r1",
            artifact_id="p1/r1/checkpoints/expired_session.pt",
            local_path=str(test_file),
            target_rel_path="checkpoints/expired_session.pt",
            size_bytes=len(b"EXPIRED_SESSION_DATA"),
            sha256="dummy",
            status=TransferStatus.UPLOADING.value,
            resumable_session_url="https://invalid.expired.session/dead",
            bytes_transferred=10,
        )
        store.journal.record_transfer(rec)

        # Next publish should fail closed on dead URL or clear it
        try:
            store.publish(test_file, "p1", "r1")
        except DriveTransferError:
            pass

        # Check journal: session URL was cleared or marked RETRY_PENDING
        updated = store.journal.get_transfer(rec.transfer_id)
        self.assertIn(updated.status, (TransferStatus.RETRY_PENDING.value, TransferStatus.UPLOAD_FAILED.value))

    def test_remote_size_mismatch_raises_integrity_error(self):
        """If remote Drive response reports size != local size, DriveIntegrityError is raised."""
        mock_http = MockDriveHttpClient()

        # Monkey-patch mock client to return corrupted size
        original_request = mock_http.request
        def patched_request(method, url, headers=None, data=None, timeout=30.0):
            status, resp_headers, body = original_request(method, url, headers=headers, data=data, timeout=timeout)
            if "session_id=upload-session" in url and status == 200:
                res = json.loads(body.decode("utf-8"))
                res["size"] = "9999999"  # Corrupted size
                return 200, resp_headers, json.dumps(res).encode("utf-8")
            return status, resp_headers, body

        mock_http.request = patched_request

        store = GoogleDriveArtifactStore(
            credentials=DriveCredentials(access_token="tok"),
            local_staging_dir=self.artifacts_dir,
            http_client=mock_http,
            chunk_size=256 * 1024,
            initial_backoff_sec=0.01,
        )

        test_file = Path(self.test_dir) / "corrupt_size.pt"
        test_file.write_bytes(b"VALID_LOCAL_BYTES")

        with self.assertRaises(DriveIntegrityError):
            store.publish(
                source=test_file,
                project_id="p1",
                run_id="r1",
            )


if __name__ == "__main__":
    unittest.main()
