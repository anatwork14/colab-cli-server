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
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

        # Session tracking buffers and committed files
        self.session_buffers: Dict[str, bytearray] = {}
        self.session_to_file_id: Dict[str, str] = {}
        self.committed_files: Dict[str, bytes] = {}
        self.mock_files: List[Dict[str, Any]] = []

    def _get_session_buffer(self, url: str) -> Optional[bytearray]:
        if url in self.session_buffers:
            return self.session_buffers[url]
        m = re.search(r"session_id=([^&]+)", url)
        if m and m.group(1) in self.session_buffers:
            return self.session_buffers[m.group(1)]
        return None

    def _get_session_file_id(self, url: str) -> str:
        if url in self.session_to_file_id:
            return self.session_to_file_id[url]
        m = re.search(r"session_id=([^&]+)", url)
        if m and m.group(1) in self.session_to_file_id:
            return self.session_to_file_id[m.group(1)]
        return f"file-{self.file_id_counter}"

    def stream_download(
        self,
        url: str,
        dest_path: Union[str, Path],
        headers: Optional[Dict[str, str]] = None,
        chunk_size: int = 64 * 1024,
        timeout: float = 60.0,
    ) -> Tuple[int, Dict[str, str], int]:
        status, resp_headers, body = self.request("GET", url, headers=headers, timeout=timeout)
        if status == 200:
            dest = Path(dest_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            body_bytes = body if isinstance(body, (bytes, bytearray)) else (body.encode("utf-8") if isinstance(body, str) else b"")
            dest.write_bytes(body_bytes)
            return status, resp_headers, len(body_bytes)
        return status, resp_headers, 0

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

        # 2. Folder lookup or file lookup in folder (files?q=...)
        if "drive/v3/files?q=" in url and method == "GET":
            q_unquoted = urllib.parse.unquote(url)
            if "mimeType = 'application/vnd.google-apps.folder'" in q_unquoted:
                return 200, {}, b'{"files": []}'
            matched = []
            for f in self.mock_files:
                f_name = f.get("name")
                f_parent = f.get("parent")
                if f_name and f"name = '{f_name}'" in q_unquoted:
                    if f_parent is None or f"'{f_parent}' in parents" in q_unquoted:
                        matched.append({
                            "id": f.get("id", "mock-file-1"),
                            "name": f_name,
                            "size": str(f.get("size", 0)),
                            "md5Checksum": f.get("md5Checksum", ""),
                        })
            return 200, {}, json.dumps({"files": matched}).encode("utf-8")

        if "drive/v3/files" in url and method == "POST" and "uploadType=resumable" not in url:
            # Creating a folder
            meta = json.loads(data.decode("utf-8")) if data else {}
            self.folder_id_counter += 1
            resp = {"id": f"folder-{self.folder_id_counter}", "name": meta.get("name", "folder")}
            return 201, {"content-type": "application/json"}, json.dumps(resp).encode("utf-8")

        # 3. Resumable upload initiation
        if "uploadType=resumable" in url and method == "POST":
            self.file_id_counter += 1
            file_id = f"file-{self.file_id_counter}"
            session_id = f"upload-session-{self.file_id_counter}"
            session_url = f"https://www.googleapis.com/upload/drive/v3/files?session_id={session_id}"
            buf = bytearray()
            self.session_buffers[session_url] = buf
            self.session_buffers[session_id] = buf
            self.session_to_file_id[session_url] = file_id
            self.session_to_file_id[session_id] = file_id
            return 200, {"location": session_url}, b""

        # 4. Uploading chunk or querying status
        if "session_id=upload-session" in url and method == "PUT":
            content_range = headers.get("content-range", "")

            # Query progress (bytes */total)
            if "*/" in content_range:
                if self.simulate_resume_interruption or self.resumed_called:
                    self.resumed_called = True
                    return 308, {"range": "bytes=0-1048575"}, b""
                buf = self._get_session_buffer(url)
                if buf is not None and len(buf) > 0:
                    return 308, {"range": f"bytes=0-{len(buf) - 1}"}, b""
                return 308, {}, b""

            # Parse content-range: bytes start-end/total
            m = re.match(r"bytes\s+(\d+)-(\d+)/(\d+)", content_range)
            if not m:
                return 400, {}, b'{"error": "Invalid Content-Range"}'
            start_byte = int(m.group(1))
            end_byte = int(m.group(2))
            total_size = int(m.group(3))

            buf = self._get_session_buffer(url)
            chunk_data = data or b""

            def _commit_to_buf():
                if buf is not None:
                    if len(buf) < start_byte:
                        fill = chunk_data[0:1] if (chunk_data and chunk_data == chunk_data[0:1] * len(chunk_data)) else b"\x00"
                        buf.extend(fill * (start_byte - len(buf)))
                    if len(buf) < end_byte + 1:
                        buf.extend(b"\x00" * (end_byte + 1 - len(buf)))
                    buf[start_byte : end_byte + 1] = chunk_data

            # Sending actual bytes: simulate network drop after receiving chunk
            if self.simulate_resume_interruption and not self.resumed_called:
                self.simulate_resume_interruption = False
                _commit_to_buf()
                return 500, {}, b"Network disconnected"

            # Fail next n chunk uploads
            if self.fail_next_n_chunks > 0:
                self.fail_next_n_chunks -= 1
                return 503, {}, b'{"error": {"message": "Backend Error"}}'

            _commit_to_buf()

            file_id = self._get_session_file_id(url)

            if end_byte + 1 >= total_size:
                # Finished upload
                committed = bytes(buf) if buf is not None else chunk_data
                self.committed_files[file_id] = committed
                md5_hex = hashlib.md5(committed).hexdigest()
                resp = {
                    "id": file_id,
                    "name": "uploaded_file",
                    "size": str(total_size),
                    "md5Checksum": md5_hex,
                }
                return 200, {"content-type": "application/json"}, json.dumps(resp).encode("utf-8")
            else:
                # Chunk accepted, more expected
                return 308, {"range": f"bytes=0-{end_byte}"}, b""

        # 5. Media download
        if "alt=media" in url and method == "GET":
            m_fid = re.search(r"drive/v3/files/([^/?]+)", url)
            fid = m_fid.group(1) if m_fid else None
            if fid and fid in self.committed_files:
                return 200, {}, self.committed_files[fid]
            return 200, {}, b"VERIFIED_DOWNLOADED_CONTENT_MOCK"

        # 6. File metadata query (GET /drive/v3/files/{file_id})
        if re.search(r"drive/v3/files/[^/?]+", url) and method == "GET" and "?q=" not in url:
            m_fid = re.search(r"drive/v3/files/([^/?]+)", url)
            if m_fid:
                fid = m_fid.group(1)
                if fid in self.committed_files:
                    c_data = self.committed_files[fid]
                    resp = {
                        "id": fid,
                        "name": "uploaded_file",
                        "size": str(len(c_data)),
                        "md5Checksum": hashlib.md5(c_data).hexdigest(),
                    }
                    return 200, {"content-type": "application/json"}, json.dumps(resp).encode("utf-8")

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

        # Pre-resolve folders so failure only happens during chunk upload
        self.store.resolve_target_folder_id("my_project", "run_102", "checkpoints")

        # Set max_retries=1 and fail so first upload attempt exhausts retries
        self.store.max_retries = 1
        self.mock_client.fail_next_n_chunks = 5
        self.mock_client.simulate_resume_interruption = True


        # First attempt should fail transiently after retries
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

        # Clear simulated errors for second attempt
        self.mock_client.fail_next_n_chunks = 0
        self.store.max_retries = 5

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

    def test_partial_308_range_acknowledgment_preserves_committed_content(self):
        """Partial 308 Range acknowledgment followed by remaining bytes matches exact source."""
        file_path = Path(self.test_dir) / "partial_test.bin"
        data = bytes([(i * 7 + 13) % 256 for i in range(256 * 1024)])
        file_path.write_bytes(data)

        partial_acked = False
        original_request = self.mock_client.request

        def patched_request(method, url, headers=None, data=None, timeout=30.0):
            nonlocal partial_acked
            if "session_id=upload-session" in url and method == "PUT" and not partial_acked:
                cr = (headers or {}).get("content-range", "")
                if "*/" not in cr:
                    partial_acked = True
                    buf = self.mock_client._get_session_buffer(url)
                    if buf is not None and data:
                        if len(buf) < 65536:
                            buf.extend(b"\x00" * (65536 - len(buf)))
                        buf[0:65536] = data[0:65536]
                    return 308, {"range": "bytes=0-65535"}, b""
            return original_request(method, url, headers=headers, data=data, timeout=timeout)

        self.mock_client.request = patched_request

        stored = self.store.publish(
            source=file_path,
            project_id="p_partial",
            run_id="r_partial",
        )

        self.assertEqual(stored.size_bytes, len(data))
        self.assertEqual(stored.sha256, compute_file_sha256(file_path))
        committed = self.mock_client.committed_files[stored.remote_file_id]
        self.assertEqual(committed, data)
        self.assertEqual(hashlib.sha256(committed).hexdigest(), compute_file_sha256(file_path))

    def test_308_missing_range_header_retries_from_progress_query(self):
        """308 with missing Range header + progress query returning 0 retries from 0 without skipping."""
        file_path = Path(self.test_dir) / "missing_range.bin"
        data = bytes([(i * 3 + 5) % 256 for i in range(256 * 1024)])
        file_path.write_bytes(data)

        missing_range_sent = False
        original_request = self.mock_client.request

        def patched_request(method, url, headers=None, data=None, timeout=30.0):
            nonlocal missing_range_sent
            if "session_id=upload-session" in url and method == "PUT":
                cr = (headers or {}).get("content-range", "")
                if "*/" not in cr and not missing_range_sent:
                    missing_range_sent = True
                    return 308, {}, b""
            return original_request(method, url, headers=headers, data=data, timeout=timeout)

        self.mock_client.request = patched_request

        stored = self.store.publish(
            source=file_path,
            project_id="p_missing_range",
            run_id="r_missing_range",
        )

        self.assertTrue(missing_range_sent)
        self.assertEqual(stored.size_bytes, len(data))
        committed = self.mock_client.committed_files[stored.remote_file_id]
        self.assertEqual(committed, data)
        self.assertEqual(hashlib.sha256(committed).hexdigest(), compute_file_sha256(file_path))

    def test_308_malformed_and_out_of_bounds_range_raises_transfer_error(self):
        """308 with malformed Range or out-of-bounds Range raises DriveTransferError."""
        file_path = Path(self.test_dir) / "bad_range.bin"
        file_path.write_bytes(b"BAD_RANGE_TEST_DATA" * 100)

        original_request = self.mock_client.request

        # Subcase A: Malformed Range ("bytes=bad")
        def malformed_request(method, url, headers=None, data=None, timeout=30.0):
            if "session_id=upload-session" in url and method == "PUT":
                cr = (headers or {}).get("content-range", "")
                if "*/" not in cr:
                    return 308, {"range": "bytes=bad"}, b""
            return original_request(method, url, headers=headers, data=data, timeout=timeout)

        self.mock_client.request = malformed_request
        with self.assertRaises(DriveTransferError) as ctx1:
            self.store.publish(source=file_path, project_id="p_bad1", run_id="r_bad1")
        self.assertIn("Malformed HTTP 308 Range header", str(ctx1.exception))

        # Subcase B: Out-of-bounds Range ("bytes=0-99999999")
        def out_of_bounds_request(method, url, headers=None, data=None, timeout=30.0):
            if "session_id=upload-session" in url and method == "PUT":
                cr = (headers or {}).get("content-range", "")
                if "*/" not in cr:
                    return 308, {"range": "bytes=0-99999999"}, b""
            return original_request(method, url, headers=headers, data=data, timeout=timeout)

        self.mock_client.request = out_of_bounds_request
        with self.assertRaises(DriveTransferError) as ctx2:
            self.store.publish(source=file_path, project_id="p_bad2", run_id="r_bad2")
        self.assertIn("Impossible HTTP 308 Range offset", str(ctx2.exception))

    def test_duplicate_same_name_same_size_different_content_not_reused(self):
        """Same-name, same-size file with different content is NOT reused; new file uploaded."""
        file_path = Path(self.test_dir) / "model_weights.pt"
        new_data = b"FRESH_NEW_TRAINED_WEIGHTS_CONTENT_V2"
        file_path.write_bytes(new_data)

        stale_data = b"STALE_OLD_CHECKPOINT_DATA_1234567890"
        self.assertEqual(len(new_data), len(stale_data))

        self.mock_client.mock_files.append({
            "id": "old-stale-file-id-999",
            "name": "model_weights.pt",
            "size": len(stale_data),
            "md5Checksum": hashlib.md5(stale_data).hexdigest(),
        })

        stored = self.store.publish(
            source=file_path,
            project_id="p_dup",
            run_id="r_dup",
            artifact_type=ArtifactType.CHECKPOINT.value,
        )

        self.assertNotEqual(stored.remote_file_id, "old-stale-file-id-999")
        self.assertEqual(stored.sha256, compute_file_sha256(file_path))
        self.assertEqual(stored.md5, compute_file_md5(file_path))
        self.assertEqual(self.mock_client.committed_files[stored.remote_file_id], new_data)

    def test_missing_md5_checksum_readback_verification_success_and_failure(self):
        """Missing md5Checksum triggers alt=media readback: succeeds on match, fails on mismatch."""
        file_path = Path(self.test_dir) / "no_md5_file.bin"
        payload_data = b"INTEGRITY_READBACK_TEST_DATA_BYTES"
        file_path.write_bytes(payload_data)

        original_request = self.mock_client.request

        # Subcase A: Missing md5Checksum, readback matches -> succeeds
        def missing_md5_success_request(method, url, headers=None, data=None, timeout=30.0):
            status, resp_headers, body = original_request(method, url, headers=headers, data=data, timeout=timeout)
            if "session_id=upload-session" in url and status == 200:
                resp = {"id": "file-no-md5", "name": "uploaded_file", "size": str(len(payload_data))}
                return 200, resp_headers, json.dumps(resp).encode("utf-8")
            if "drive/v3/files/file-no-md5" in url and method == "GET" and "alt=media" not in url:
                resp = {"id": "file-no-md5", "name": "uploaded_file", "size": str(len(payload_data))}
                return 200, {}, json.dumps(resp).encode("utf-8")
            if "alt=media" in url and method == "GET":
                return 200, {}, payload_data
            return status, resp_headers, body

        self.mock_client.request = missing_md5_success_request
        stored = self.store.publish(source=file_path, project_id="p_md5_ok", run_id="r_md5_ok")
        self.assertEqual(stored.remote_file_id, "file-no-md5")
        self.assertEqual(stored.size_bytes, len(payload_data))

        # Subcase B: Missing md5Checksum, readback corrupted -> raises DriveIntegrityError
        def missing_md5_corrupt_request(method, url, headers=None, data=None, timeout=30.0):
            status, resp_headers, body = original_request(method, url, headers=headers, data=data, timeout=timeout)
            if "session_id=upload-session" in url and status == 200:
                resp = {"id": "file-corrupt-md5", "name": "uploaded_file", "size": str(len(payload_data))}
                return 200, resp_headers, json.dumps(resp).encode("utf-8")
            if "drive/v3/files/file-corrupt-md5" in url and method == "GET" and "alt=media" not in url:
                resp = {"id": "file-corrupt-md5", "name": "uploaded_file", "size": str(len(payload_data))}
                return 200, {}, json.dumps(resp).encode("utf-8")
            if "alt=media" in url and method == "GET":
                return 200, {}, b"CORRUPTED_DOWNLOAD_BYTES"
            return status, resp_headers, body

        self.mock_client.request = missing_md5_corrupt_request
        with self.assertRaises(DriveIntegrityError) as ctx:
            self.store.publish(source=file_path, project_id="p_md5_fail", run_id="r_md5_fail")
        self.assertIn("missing authoritative md5Checksum and failed content readback verification", str(ctx.exception))

    def test_credential_and_session_redaction_in_exception_strings(self):
        """Seeded client_secret, refresh_token, and session_id in errors are redacted."""
        file_path = Path(self.test_dir) / "secret_test.pt"
        file_path.write_bytes(b"SECRET_TEST_DATA")

        secret_val = "SUPER_SECRET_CLIENT_KEY_98765"
        refresh_val = "SECRET_REFRESH_TOKEN_ABCDEF"
        session_val = "upload-session-sensitive-12345"

        original_request = self.mock_client.request

        # Subcase A: 401 with secrets leaked in body
        def leaky_401_request(method, url, headers=None, data=None, timeout=30.0):
            if "drive/v3/files?q=" in url:
                leaky_body = json.dumps({
                    "error": "unauthorized",
                    "client_secret": secret_val,
                    "refresh_token": refresh_val,
                }).encode("utf-8")
                return 401, {}, leaky_body
            return original_request(method, url, headers=headers, data=data, timeout=timeout)

        self.mock_client.request = leaky_401_request
        self.creds.refresh_token = None
        with self.assertRaises(DriveAuthenticationError) as ctx1:
            self.store.publish(source=file_path, project_id="p_sec1", run_id="r_sec1")

        err_str1 = str(ctx1.exception)
        self.assertNotIn(secret_val, err_str1)
        self.assertNotIn(refresh_val, err_str1)
        self.assertIn("[REDACTED]", err_str1)

        # Subcase B: 500 error leaking session_id and upload session URL
        self.creds.refresh_token = "valid-refresh"
        def leaky_500_request(method, url, headers=None, data=None, timeout=30.0):
            if "session_id=upload-session" in url and method == "PUT":
                return 500, {}, f"Server crashed at {url} with session_id={session_val}".encode("utf-8")
            return original_request(method, url, headers=headers, data=data, timeout=timeout)

        self.mock_client.request = leaky_500_request
        self.store.max_retries = 1
        with self.assertRaises(DriveTransferError) as ctx2:
            self.store.publish(source=file_path, project_id="p_sec2", run_id="r_sec2")

        err_str2 = str(ctx2.exception)
        self.assertNotIn(session_val, err_str2)
        records = self.store.journal.list_transfers(project_id="p_sec2", run_id="r_sec2")
        self.assertTrue(len(records) > 0)
        self.assertNotIn(session_val, records[0].last_error or "")


if __name__ == "__main__":
    unittest.main()
