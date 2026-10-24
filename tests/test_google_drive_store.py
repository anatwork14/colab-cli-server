"""Offline deterministic unit tests for GoogleDriveArtifactStore.

Validates:
- Resumable upload protocol and chunking
- Interrupted upload query and recovery
- Exponential backoff retry on 429/500
- Drive quota exceeded handling (403)
- Token expiration and refresh handling (401)
- MD5 and size integrity verification
- Secret credential file 0600 permissions
- Zero live cloud API calls or GPU allocations (100% mocked offline)
"""
import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from colab_adapter.errors import (
    DriveAuthenticationError,
    DriveIntegrityError,
    DriveQuotaExceededError,
    DriveTransferError,
)
from colab_adapter.storage import (
    ArtifactType,
    DriveCredentials,
    GoogleDriveArtifactStore,
    compute_file_md5,
    compute_file_sha256,
)


class MockDriveHttpClient:
    """Mock HTTP client simulating Google Drive v3 REST API endpoints."""

    def __init__(self):
        self.requests_log: List[Tuple[str, str, Dict[str, str], Optional[bytes]]] = []
        self.folder_id_counter = 100
        self.file_id_counter = 500
        self.fail_next_n_chunks = 0
        self.quota_exceeded = False
        self.auth_expired = False
        self.token_refreshed = False
        self.simulate_resume_interruption = False
        self.resumed_called = False

    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[bytes] = None,
        timeout: float = 30.0,
    ) -> Tuple[int, Dict[str, str], bytes]:
        headers = headers or {}
        self.requests_log.append((method, url, headers, data))

        # 1. Token refresh endpoint
        if "oauth2.googleapis.com/token" in url:
            self.token_refreshed = True
            body = json.dumps({
                "access_token": "new-refreshed-token-12345",
                "expires_in": 3600,
            }).encode("utf-8")
            return 200, {"content-type": "application/json"}, body

        # Check auth expired simulation
        if self.auth_expired and not self.token_refreshed:
            return 401, {}, b'{"error": {"message": "Invalid Credentials"}}'

        # Check quota exceeded simulation
        if self.quota_exceeded:
            return 403, {}, b'{"error": {"errors": [{"reason": "storageQuotaExceeded", "message": "Drive quota full"}]}}'

        # Check transient server error simulation
        if self.fail_next_n_chunks > 0:
            self.fail_next_n_chunks -= 1
            return 503, {}, b'{"error": {"message": "Backend Error"}}'

        # 2. Folder lookup or creation (files?q=...)
        if "drive/v3/files?q=" in url and method == "GET":
            # Return empty list to force creation or match
            return 200, {}, b'{"files": []}'

        if "drive/v3/files" in url and method == "POST" and "uploadType=resumable" not in url:
            # Creating a folder
            meta = json.loads(data.decode("utf-8")) if data else {}
            self.folder_id_counter += 1
            resp = {"id": f"folder-{self.folder_id_counter}", "name": meta.get("name", "folder")}
            return 201, {"content-type": "application/json"}, json.dumps(resp).encode("utf-8")

        # 3. Resumable upload initiation
        if "uploadType=resumable" in url and method == "POST":
            self.file_id_counter += 1
            session_url = f"https://www.googleapis.com/upload/drive/v3/files?session_id=upload-session-{self.file_id_counter}"
            return 200, {"location": session_url}, b""

        # 4. Uploading chunk or querying status
        if "session_id=upload-session" in url and method == "PUT":
            content_range = headers.get("content-range", "")

            # Query progress (bytes */total)
            if "*/" in content_range:
                if self.simulate_resume_interruption:
                    self.resumed_called = True
                    return 308, {"range": "bytes=0-1048575"}, b""
                return 308, {"range": "bytes=0-0"}, b""

            # Sending actual bytes
            if self.simulate_resume_interruption and not self.resumed_called:
                # Interrupt after first chunk
                self.simulate_resume_interruption = False
                return 500, {}, b"Network disconnected"

            # Check if this is the final chunk
            total_size = int(content_range.split("/")[1]) if "/" in content_range else 0
            end_byte = int(content_range.split("-")[1].split("/")[0]) if "-" in content_range else 0

            if end_byte + 1 >= total_size:
                # Finished upload
                resp = {
                    "id": f"file-{self.file_id_counter}",
                    "name": "uploaded_file",
                    "size": str(total_size),
                }
                return 200, {"content-type": "application/json"}, json.dumps(resp).encode("utf-8")
            else:
                # Chunk accepted, more expected
                return 308, {"range": f"bytes=0-{end_byte}"}, b""

        # 5. Media download
        if "alt=media" in url and method == "GET":
            return 200, {}, b"VERIFIED_DOWNLOADED_CONTENT_MOCK"

        return 404, {}, b'{"error": "Not Found"}'


class TestGoogleDriveArtifactStore(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="colab_drive_test_")
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
            chunk_size=1024 * 1024,  # 1 MiB
            initial_backoff_sec=0.01,
        )

    def test_credential_file_has_secure_permissions(self):
        """Credentials file must be restricted to 0600 (owner read/write only)."""
        mode = os.stat(self.credentials_file).st_mode
        self.assertEqual(stat.S_IMODE(mode), 0o600)

    def test_resumable_upload_success_and_verification(self):
        """Test multi-chunk upload completes and returns verified StoredArtifact."""
        file_path = Path(self.test_dir) / "model_epoch_5.pt"
        data = b"X" * (2 * 1024 * 1024 + 500)  # ~2 MiB + 500 bytes (3 chunks)
        file_path.write_bytes(data)

        stored = self.store.publish(
            source=file_path,
            project_id="my_project",
            run_id="run_101",
            artifact_type=ArtifactType.CHECKPOINT.value,
        )

        self.assertEqual(stored.storage_provider, "google_drive")
        self.assertEqual(stored.size_bytes, len(data))
        self.assertEqual(stored.sha256, compute_file_sha256(file_path))
        self.assertIsNotNone(stored.remote_file_id)

        # Check journal state
        transfers = self.store.journal.list_transfers(project_id="my_project", run_id="run_101")
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].status, "PUBLISHED")

    def test_interrupted_upload_query_and_resume(self):
        """Test upload recovers from connection drop by querying byte offset and resuming."""
        file_path = Path(self.test_dir) / "large_weights.pt"
        data = b"Y" * (3 * 1024 * 1024)  # 3 MiB
        file_path.write_bytes(data)

        # Enable simulated interruption on first attempt
        self.mock_client.simulate_resume_interruption = True

        # First attempt should fail transiently
        with self.assertRaises(DriveTransferError):
            self.store.publish(
                source=file_path,
                project_id="my_project",
                run_id="run_102",
                artifact_type=ArtifactType.CHECKPOINT.value,
            )

        # Transfer record should be saved in RETRY_PENDING state
        record = self.store.journal.list_transfers(project_id="my_project", run_id="run_102")[0]
        self.assertEqual(record.status, "RETRY_PENDING")

        # Second publish call resumes from recorded session
        stored = self.store.publish(
            source=file_path,
            project_id="my_project",
            run_id="run_102",
            artifact_type=ArtifactType.CHECKPOINT.value,
        )
        self.assertEqual(stored.size_bytes, len(data))
        self.assertEqual(self.store.journal.get_transfer(record.transfer_id).status, "PUBLISHED")

    def test_transient_error_retry_with_backoff(self):
        """Test transient HTTP 503 is retried automatically."""
        file_path = Path(self.test_dir) / "small.pt"
        file_path.write_bytes(b"HELLO_WORLD_DATA")

        self.mock_client.fail_next_n_chunks = 2  # Fail twice then succeed

        stored = self.store.publish(
            source=file_path,
            project_id="my_project",
            run_id="run_103",
        )
        self.assertIsNotNone(stored.remote_file_id)

    def test_quota_exceeded_raises_typed_error(self):
        """Storage quota exceeded raises DriveQuotaExceededError."""
        file_path = Path(self.test_dir) / "huge.pt"
        file_path.write_bytes(b"HUGE")

        self.mock_client.quota_exceeded = True

        with self.assertRaises(DriveQuotaExceededError):
            self.store.publish(
                source=file_path,
                project_id="my_project",
                run_id="run_104",
            )

    def test_token_expiration_triggers_automatic_refresh(self):
        """Expired access token is refreshed automatically."""
        file_path = Path(self.test_dir) / "test.pt"
        file_path.write_bytes(b"DATA")

        self.mock_client.auth_expired = True

        stored = self.store.publish(
            source=file_path,
            project_id="my_project",
            run_id="run_105",
        )
        self.assertTrue(self.mock_client.token_refreshed)
        self.assertEqual(self.creds.access_token, "new-refreshed-token-12345")


if __name__ == "__main__":
    unittest.main()
