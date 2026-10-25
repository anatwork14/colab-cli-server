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
import concurrent.futures
import threading
import unittest
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    TransferRecord,
    TransferStatus,
    compute_file_md5,
    compute_file_sha256,
)


class MockDriveHttpClient:
    """Mock HTTP client simulating Google Drive v3 REST API endpoints."""

    def __init__(self):
        self.lock = threading.Lock()
        self.requests_log: List[Tuple[str, str, Dict[str, str], Optional[bytes]]] = []
        self.folder_id_counter = 100
        self.file_id_counter = 500
        self.fail_next_n_chunks = 0
        self.quota_exceeded = False
        self.auth_expired = False
        self.token_refreshed = False
        self.simulate_resume_interruption = False
        self.resumed_called = False
        self.upload_sessions_initiated = 0

        # Pagination and listing error simulation controls
        self.list_page_size_override: Optional[int] = None
        self.inject_looping_page_token: bool = False
        self.inject_malformed_page_token: Any = None
        self.fail_list_files_status: Optional[int] = None
        self.fail_list_files_body: Optional[bytes] = None
        self.fail_list_files_exception: Optional[Exception] = None
        self.fail_list_files_at_page: Optional[int] = None
        self.list_files_page_count: int = 0
        self.fail_folder_lookup_status: Optional[int] = None
        self.fail_folder_lookup_body: Optional[bytes] = None
        self.fail_folder_lookup_exception: Optional[Exception] = None

        # Session tracking buffers and committed files
        self.session_buffers: Dict[str, bytearray] = {}
        self.session_to_file_id: Dict[str, str] = {}
        self.session_to_filename: Dict[str, str] = {}
        self.session_to_parent: Dict[str, Optional[str]] = {}
        self.session_to_app_props: Dict[str, Dict[str, str]] = {}
        self.committed_files: Dict[str, bytes] = {}
        self.mock_files: List[Dict[str, Any]] = []
        self.fail_download_network_error = False
        self.corrupt_download_bytes: Optional[bytes] = None
        self.truncate_download_bytes_to: Optional[int] = None

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
        if self.fail_download_network_error:
            raise ConnectionResetError("Simulated network interruption during stream_download")
        status, resp_headers, body = self.request("GET", url, headers=headers, timeout=timeout)
        if status == 200:
            dest = Path(dest_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            body_bytes = body if isinstance(body, (bytes, bytearray)) else (body.encode("utf-8") if isinstance(body, str) else b"")
            if self.truncate_download_bytes_to is not None:
                body_bytes = body_bytes[:self.truncate_download_bytes_to]
            if self.corrupt_download_bytes is not None:
                body_bytes = self.corrupt_download_bytes
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
        with self.lock:
            return self._locked_request(method, url, headers=headers, data=data, timeout=timeout)

    def _locked_request(
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
                if self.fail_folder_lookup_exception is not None:
                    raise self.fail_folder_lookup_exception
                if self.fail_folder_lookup_status is not None:
                    return self.fail_folder_lookup_status, {}, self.fail_folder_lookup_body or b'{"error": "Folder lookup failed"}'
                if self.fail_folder_lookup_body is not None:
                    return 200, {}, self.fail_folder_lookup_body

                matched_folders = []
                for f in self.mock_files:
                    if f.get("mimeType") == "application/vnd.google-apps.folder":
                        f_name = f.get("name")
                        f_parent = f.get("parent")
                        if f_name and f"name = '{f_name}'" in q_unquoted:
                            if (f_parent is None and "'root' in parents" in q_unquoted) or (f_parent and f"'{f_parent}' in parents" in q_unquoted):
                                matched_folders.append({"id": f["id"], "name": f_name})
                return 200, {}, json.dumps({"files": matched_folders}).encode("utf-8")

            # Non-folder files query: parse pagination params
            parsed = urllib.parse.urlparse(url)
            qs = urllib.parse.parse_qs(parsed.query)
            parsed_page_size = qs.get("pageSize", [None])[0]
            parsed_page_token = qs.get("pageToken", [None])[0]

            effective_page_size = self.list_page_size_override or (int(parsed_page_size) if parsed_page_size else None)

            current_page = 1
            offset = 0
            if parsed_page_token:
                m_page = re.match(r"page-(\d+)-offset-(\d+)", parsed_page_token)
                if m_page:
                    current_page = int(m_page.group(1))
                    offset = int(m_page.group(2))
                elif parsed_page_token.startswith("offset-"):
                    offset = int(parsed_page_token.split("-")[1])
                    current_page = (offset // (effective_page_size or 100)) + 1
                else:
                    self.list_files_page_count += 1
                    current_page = self.list_files_page_count
                    if effective_page_size:
                        offset = (current_page - 1) * effective_page_size
            else:
                self.list_files_page_count = 1
                current_page = 1
                offset = 0

            # Check simulated failure for file listing
            if self.fail_list_files_exception is not None:
                if self.fail_list_files_at_page is None or current_page == self.fail_list_files_at_page:
                    raise self.fail_list_files_exception

            if self.fail_list_files_status is not None:
                if self.fail_list_files_at_page is None or current_page == self.fail_list_files_at_page:
                    body = self.fail_list_files_body if self.fail_list_files_body is not None else b'{"error": "Simulated listing failure"}'
                    return self.fail_list_files_status, {}, body

            if self.fail_list_files_body is not None:
                if self.fail_list_files_at_page is None or current_page == self.fail_list_files_at_page:
                    return 200, {}, self.fail_list_files_body

            matched = []
            for f in self.mock_files:
                if f.get("mimeType") == "application/vnd.google-apps.folder":
                    continue
                f_name = f.get("name")
                f_parent = f.get("parent")
                if f_name and f"name = '{f_name}'" in q_unquoted:
                    if f_parent is None or f"'{f_parent}' in parents" in q_unquoted:
                        c_data = f.get("content")
                        if c_data is not None:
                            sz_val = str(len(c_data))
                            md5_val = hashlib.md5(c_data).hexdigest()
                        else:
                            sz_val = str(f.get("size", 0))
                            md5_val = f.get("md5Checksum", "")
                        matched.append({
                            "id": f.get("id", "mock-file-1"),
                            "name": f_name,
                            "size": sz_val,
                            "md5Checksum": md5_val,
                            "appProperties": f.get("appProperties", {}),
                            "description": f.get("description", ""),
                            "createdTime": f.get("createdTime", "2026-01-01T00:00:00Z"),
                        })

            if effective_page_size is not None and effective_page_size > 0:
                page_items = matched[offset : offset + effective_page_size]
                has_more = (offset + effective_page_size) < len(matched)
                resp_dict: Dict[str, Any] = {"files": page_items}

                if self.inject_malformed_page_token is not None:
                    resp_dict["nextPageToken"] = self.inject_malformed_page_token
                elif self.inject_looping_page_token:
                    resp_dict["nextPageToken"] = "looping-page-token"
                elif has_more:
                    next_page_num = current_page + 1
                    next_offset = offset + effective_page_size
                    resp_dict["nextPageToken"] = f"page-{next_page_num}-offset-{next_offset}"

                return 200, {}, json.dumps(resp_dict).encode("utf-8")

            return 200, {}, json.dumps({"files": matched}).encode("utf-8")

        if "drive/v3/files" in url and method == "POST" and "uploadType=resumable" not in url:
            # Creating a folder
            meta = json.loads(data.decode("utf-8")) if data else {}
            self.folder_id_counter += 1
            folder_id = f"folder-{self.folder_id_counter}"
            f_name = meta.get("name", "folder")
            f_parents = meta.get("parents", [])
            f_parent = f_parents[0] if f_parents else None
            self.mock_files.append({
                "id": folder_id,
                "name": f_name,
                "parent": f_parent,
                "mimeType": "application/vnd.google-apps.folder",
            })
            resp = {"id": folder_id, "name": f_name}
            return 201, {"content-type": "application/json"}, json.dumps(resp).encode("utf-8")

        # 3. Resumable upload initiation
        if "uploadType=resumable" in url and method == "POST":
            self.upload_sessions_initiated += 1
            self.file_id_counter += 1
            file_id = f"file-{self.file_id_counter}"
            session_id = f"upload-session-{self.file_id_counter}"
            session_url = f"https://www.googleapis.com/upload/drive/v3/files?session_id={session_id}"
            buf = bytearray()
            self.session_buffers[session_url] = buf
            self.session_buffers[session_id] = buf
            self.session_to_file_id[session_url] = file_id
            self.session_to_file_id[session_id] = file_id

            f_name = "uploaded_file"
            f_parent = None
            f_app_props = {}
            if data:
                try:
                    meta = json.loads(data.decode("utf-8"))
                    f_name = meta.get("name", f_name)
                    f_parents = meta.get("parents", [])
                    f_parent = f_parents[0] if f_parents else None
                    f_app_props = meta.get("appProperties", {})
                except Exception:
                    pass
            self.session_to_filename[session_url] = f_name
            self.session_to_filename[session_id] = f_name
            self.session_to_parent[session_url] = f_parent
            self.session_to_parent[session_id] = f_parent
            self.session_to_app_props[session_url] = f_app_props
            self.session_to_app_props[session_id] = f_app_props
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
                fname = self.session_to_filename.get(url, "uploaded_file")
                fparent = self.session_to_parent.get(url, None)
                fapp_props = self.session_to_app_props.get(url, {})
                self.mock_files.append({
                    "id": file_id,
                    "name": fname,
                    "parent": fparent,
                    "size": total_size,
                    "md5Checksum": md5_hex,
                    "appProperties": fapp_props,
                })
                resp = {
                    "id": file_id,
                    "name": fname,
                    "size": str(total_size),
                    "md5Checksum": md5_hex,
                    "appProperties": fapp_props,
                }
                return 200, {"content-type": "application/json"}, json.dumps(resp).encode("utf-8")
            else:
                # Chunk accepted, more expected
                return 308, {"range": f"bytes=0-{end_byte}"}, b""

        # 5. Media download
        if "alt=media" in url and method == "GET":
            if self.fail_download_network_error:
                raise ConnectionResetError("Simulated network interruption during media download")
            m_fid = re.search(r"drive/v3/files/([^/?]+)", url)
            fid = m_fid.group(1) if m_fid else None
            data_bytes = b"VERIFIED_DOWNLOADED_CONTENT_MOCK"
            if fid and fid in self.committed_files:
                data_bytes = self.committed_files[fid]
            elif fid:
                for mf in self.mock_files:
                    if mf.get("id") == fid and "content" in mf:
                        data_bytes = mf["content"]
                        break
            if self.truncate_download_bytes_to is not None:
                data_bytes = data_bytes[:self.truncate_download_bytes_to]
            if self.corrupt_download_bytes is not None:
                data_bytes = self.corrupt_download_bytes
            return 200, {}, data_bytes

        # 6. File metadata query (GET /drive/v3/files/{file_id})
        if re.search(r"drive/v3/files/[^/?]+", url) and method == "GET" and "?q=" not in url:
            m_fid = re.search(r"drive/v3/files/([^/?]+)", url)
            if m_fid:
                fid = m_fid.group(1)
                for mf in self.mock_files:
                    if mf.get("id") == fid:
                        c_data = mf.get("content")
                        if c_data is not None:
                            sz = str(len(c_data))
                            md5 = hashlib.md5(c_data).hexdigest()
                        else:
                            sz = str(mf.get("size", 0))
                            md5 = mf.get("md5Checksum", "")
                        resp = {
                            "id": fid,
                            "name": mf.get("name", "uploaded_file"),
                            "size": sz,
                            "md5Checksum": md5,
                            "appProperties": mf.get("appProperties", {}),
                            "description": mf.get("description", ""),
                            "createdTime": mf.get("createdTime", "2026-01-01T00:00:00Z"),
                        }
                        return 200, {"content-type": "application/json"}, json.dumps(resp).encode("utf-8")
                if fid in self.committed_files:
                    c_data = self.committed_files[fid]
                    fname = "uploaded_file"
                    fprops = {}
                    for mf in self.mock_files:
                        if mf.get("id") == fid:
                            fname = mf.get("name", fname)
                            fprops = mf.get("appProperties", {})
                            break
                    resp = {
                        "id": fid,
                        "name": fname,
                        "size": str(len(c_data)),
                        "md5Checksum": hashlib.md5(c_data).hexdigest(),
                        "appProperties": fprops,
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

    def test_concurrent_publish_exact_same_file_single_upload_and_idempotent_result(self):
        """Two concurrent workers publishing exact same file result in 1 upload session & 1 remote file."""
        file_path = Path(self.test_dir) / "concurrent_model.pt"
        data = b"SHARED_MODEL_WEIGHTS_CONCURRENT_DATA" * 1000
        file_path.write_bytes(data)

        # Worker 2 has its own GoogleDriveArtifactStore instance pointing to the same staging dir and mock client
        store2 = GoogleDriveArtifactStore(
            credentials=self.creds,
            credentials_file=self.credentials_file,
            local_staging_dir=self.staging_dir,
            http_client=self.mock_client,
            chunk_size=1024 * 1024,
            initial_backoff_sec=0.01,
        )

        barrier = threading.Barrier(2)
        results = [None, None]
        errors = [None, None]

        def worker1():
            try:
                barrier.wait()
                results[0] = self.store.publish(
                    source=file_path,
                    project_id="p_conc",
                    run_id="r_conc",
                    artifact_type=ArtifactType.CHECKPOINT.value,
                )
            except Exception as e:
                errors[0] = e

        def worker2():
            try:
                barrier.wait()
                results[1] = store2.publish(
                    source=file_path,
                    project_id="p_conc",
                    run_id="r_conc",
                    artifact_type=ArtifactType.CHECKPOINT.value,
                )
            except Exception as e:
                errors[1] = e

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertIsNone(errors[0])
        self.assertIsNone(errors[1])
        art1 = results[0]
        art2 = results[1]
        self.assertIsNotNone(art1)
        self.assertIsNotNone(art2)

        # Exactly 1 upload session initiated and 1 file committed on Drive
        self.assertEqual(self.mock_client.upload_sessions_initiated, 1)
        self.assertEqual(len(self.mock_client.committed_files), 1)

        # Both callers receive valid StoredArtifact with the same remote_file_id and sha256
        self.assertEqual(art1.remote_file_id, art2.remote_file_id)
        self.assertEqual(art1.sha256, art2.sha256)
        self.assertEqual(art1.sha256, compute_file_sha256(file_path))
        self.assertEqual(art1.size_bytes, len(data))
        self.assertEqual(art2.size_bytes, len(data))

        # Check journal state
        transfers = self.store.journal.list_transfers(project_id="p_conc", run_id="r_conc")
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].status, "PUBLISHED")
        self.assertEqual(transfers[0].remote_file_id, art1.remote_file_id)

    def test_publish_modified_checkpoint_publishes_new_revision(self):
        """Publishing v1 followed by modifying checkpoint with new bytes publishes v2 with new sha256."""
        file_path = Path(self.test_dir) / "checkpoint.pt"
        data_v1 = b"VERSION_1_WEIGHTS_CONTENT"
        file_path.write_bytes(data_v1)

        stored_v1 = self.store.publish(
            source=file_path,
            project_id="p_rev",
            run_id="r_rev",
            artifact_type=ArtifactType.CHECKPOINT.value,
        )
        self.assertEqual(stored_v1.sha256, compute_file_sha256(file_path))
        self.assertEqual(stored_v1.size_bytes, len(data_v1))
        fid_v1 = stored_v1.remote_file_id

        # Modify checkpoint locally with new bytes (version 2)
        data_v2 = b"VERSION_2_WEIGHTS_CONTENT_UPDATED_BYTES"
        file_path.write_bytes(data_v2)

        stored_v2 = self.store.publish(
            source=file_path,
            project_id="p_rev",
            run_id="r_rev",
            artifact_type=ArtifactType.CHECKPOINT.value,
        )

        # Stored v2 must have version 2's sha256 and distinct remote file ID
        self.assertEqual(stored_v2.sha256, compute_file_sha256(file_path))
        self.assertNotEqual(stored_v2.sha256, stored_v1.sha256)
        self.assertNotEqual(stored_v2.remote_file_id, fid_v1)
        self.assertEqual(stored_v2.size_bytes, len(data_v2))

        # Re-publish version 2 without changing local file: idempotent reuse of v2
        stored_v2_repub = self.store.publish(
            source=file_path,
            project_id="p_rev",
            run_id="r_rev",
            artifact_type=ArtifactType.CHECKPOINT.value,
        )
        self.assertEqual(stored_v2_repub.remote_file_id, stored_v2.remote_file_id)
        self.assertEqual(stored_v2_repub.sha256, stored_v2.sha256)

    def test_concurrent_manifest_locking_no_clobber(self):
        """Concurrent processes publishing distinct files to the same run do not clobber manifest."""
        file_a = Path(self.test_dir) / "weights_a.pt"
        file_a.write_bytes(b"DATA_FILE_A_CONCURRENT")
        file_b = Path(self.test_dir) / "weights_b.pt"
        file_b.write_bytes(b"DATA_FILE_B_CONCURRENT")

        store2 = GoogleDriveArtifactStore(
            credentials=self.creds,
            credentials_file=self.credentials_file,
            local_staging_dir=self.staging_dir,
            http_client=self.mock_client,
            chunk_size=1024 * 1024,
            initial_backoff_sec=0.01,
        )

        barrier = threading.Barrier(2)
        results = [None, None]
        errors = [None, None]

        def publish_a():
            try:
                barrier.wait()
                results[0] = self.store.publish(file_a, project_id="p_man", run_id="r_man")
            except Exception as e:
                errors[0] = e

        def publish_b():
            try:
                barrier.wait()
                results[1] = store2.publish(file_b, project_id="p_man", run_id="r_man")
            except Exception as e:
                errors[1] = e

        t1 = threading.Thread(target=publish_a)
        t2 = threading.Thread(target=publish_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertIsNone(errors[0])
        self.assertIsNone(errors[1])

        # Verify manifest on disk has both artifacts
        manifest_path = self.staging_dir / "projects" / "p_man" / "runs" / "r_man" / "manifest.json"
        manifest = PublicationManifest.load(manifest_path)
        base_art_ids = [a.artifact_id.split("@sha256_")[0] for a in manifest.list_artifacts()]
        self.assertIn("p_man/r_man/checkpoints/weights_a.pt", base_art_ids)
        self.assertIn("p_man/r_man/checkpoints/weights_b.pt", base_art_ids)
        # Verify unversioned lookup works via get_metadata
        self.assertIsNotNone(self.store.get_metadata("p_man/r_man/checkpoints/weights_a.pt"))
        self.assertIsNotNone(self.store.get_metadata("p_man/r_man/checkpoints/weights_b.pt"))

    def test_in_progress_transfer_mismatch_raises_storage_error(self):
        """In-progress/unpublished transfer encountering remote checksum mismatch raises StorageError."""
        file_path = Path(self.test_dir) / "in_progress_test.pt"
        local_data = b"LOCAL_UNPUBLISHED_CONTENT"
        file_path.write_bytes(local_data)
        local_md5 = compute_file_md5(file_path)

        # Pre-record an in-progress transfer in UPLOADING state
        target_folder_id = self.store.resolve_target_folder_id("p_inp", "r_inp", "checkpoints")
        base_tid = f"{self.store.provider_name}-p_inp-r_inp-in_progress_test.pt"
        record = TransferRecord(
            transfer_id=base_tid,
            project_id="p_inp",
            run_id="r_inp",
            artifact_id="p_inp/r_inp/checkpoints/in_progress_test.pt",
            local_path=str(file_path),
            target_rel_path="checkpoints/in_progress_test.pt",
            size_bytes=len(local_data),
            sha256=compute_file_sha256(file_path),
            md5=local_md5,
            status=TransferStatus.UPLOADING.value,
        )
        self.store.journal.record_transfer(record)

        # Remote Drive has a file with the same name, but corrupted / mismatched MD5
        self.mock_client.mock_files.append({
            "id": "corrupt-remote-id-123",
            "name": "in_progress_test.pt",
            "parent": target_folder_id,
            "size": len(local_data),
            "md5Checksum": "00000000000000000000000000000000",
        })

        with self.assertRaises(StorageError) as ctx:
            self.store.publish(file_path, project_id="p_inp", run_id="r_inp")
        self.assertIn("checksum mismatch", str(ctx.exception))

    def test_exclusive_lock_reentrancy_and_timeout(self):
        """_exclusive_lock is re-entrant in the same thread and does not deadlock."""
        with self.store._exclusive_lock("reentrant-lock-test"):
            with self.store._exclusive_lock("reentrant-lock-test"):
                pass

    def test_idempotent_republish_deduplication(self):
        """Re-publishing same content dedupes without uploading; pre-seeded files verified via readback."""
        file_path = Path(self.test_dir) / "idempotent_model.pt"
        content = b"idempotent content for deduplication test"
        file_path.write_bytes(content)

        # 1. First publish
        stored1 = self.store.publish(
            source=file_path,
            project_id="p_idem",
            run_id="r_idem",
            artifact_type=ArtifactType.CHECKPOINT.value,
        )
        self.assertEqual(self.mock_client.upload_sessions_initiated, 1)

        # 2. Second publish of exact same file (in-process deduplication)
        stored2 = self.store.publish(
            source=file_path,
            project_id="p_idem",
            run_id="r_idem",
            artifact_type=ArtifactType.CHECKPOINT.value,
        )
        self.assertEqual(stored2.remote_file_id, stored1.remote_file_id)
        self.assertEqual(stored2.artifact_id, stored1.artifact_id)
        self.assertEqual(self.mock_client.upload_sessions_initiated, 1)

        # 3. Remote folder pre-seeded file deduplication with alt=media readback
        preseeded_data = b"preseeded remote file content on drive"
        preseeded_file = Path(self.test_dir) / "preseeded_weights.pt"
        preseeded_file.write_bytes(preseeded_data)

        target_folder = self.store.resolve_target_folder_id("p_idem2", "r_idem2", "checkpoints")
        self.mock_client.mock_files.append({
            "id": "preseeded-drive-fid-888",
            "name": "preseeded_weights.pt",
            "parent": target_folder,
            "size": len(preseeded_data),
            "md5Checksum": hashlib.md5(preseeded_data).hexdigest(),
            "content": preseeded_data,
        })

        stored3 = self.store.publish(
            source=preseeded_file,
            project_id="p_idem2",
            run_id="r_idem2",
            artifact_type=ArtifactType.CHECKPOINT.value,
        )
        self.assertEqual(stored3.remote_file_id, "preseeded-drive-fid-888")
        self.assertEqual(stored3.sha256, compute_file_sha256(preseeded_file))
        # No upload session was initiated because pre-seeded file was deduplicated via readback!
        self.assertEqual(self.mock_client.upload_sessions_initiated, 1)

    def test_checkpoint_revision_history_and_dual_retrieval(self):
        """Revision persistence, dual retrieval by versioned/unversioned ID, and tampering detection."""
        project_id = "proj_hist"
        run_id = "run_hist"
        file_path = Path(self.test_dir) / "model.pt"

        # Step A: Publish checkpoints/model.pt with content A
        data_a = b"EPOCH_1_WEIGHTS_REVISION_A"
        file_path.write_bytes(data_a)
        sha_a = compute_file_sha256(file_path)

        rev_a = self.store.publish(
            source=file_path,
            project_id=project_id,
            run_id=run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
        )
        self.assertEqual(rev_a.sha256, sha_a)
        self.assertEqual(rev_a.size_bytes, len(data_a))
        self.assertTrue(rev_a.artifact_id.endswith(f"@sha256_{sha_a.lower()}"))
        self.assertIn("proj_hist/run_hist/checkpoints/model.pt", rev_a.artifact_id)

        # Step B: Overwrite local checkpoints/model.pt with content B and publish again
        data_b = b"EPOCH_2_WEIGHTS_REVISION_B_LARGER"
        file_path.write_bytes(data_b)
        sha_b = compute_file_sha256(file_path)

        rev_b = self.store.publish(
            source=file_path,
            project_id=project_id,
            run_id=run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
        )
        self.assertEqual(rev_b.sha256, sha_b)
        self.assertEqual(rev_b.size_bytes, len(data_b))
        self.assertTrue(rev_b.artifact_id.endswith(f"@sha256_{sha_b.lower()}"))

        # Step C: Inspect PublicationManifest
        manifest_path = self.staging_dir / "projects" / project_id / "runs" / run_id / "manifest.json"
        self.assertTrue(manifest_path.exists())
        manifest = PublicationManifest.load(manifest_path)
        all_artifacts = manifest.list_artifacts()
        all_ids = [a.artifact_id for a in all_artifacts]

        # Both Revision A and Revision B exist in manifest
        self.assertIn(rev_a.artifact_id, all_ids)
        self.assertIn(rev_b.artifact_id, all_ids)

        # Their artifact_ids are distinct
        self.assertNotEqual(rev_a.artifact_id, rev_b.artifact_id)

        # Their remote_file_ids are distinct
        self.assertNotEqual(rev_a.remote_file_id, rev_b.remote_file_id)

        # Their sha256 and size_bytes are preserved accurately
        stored_a = manifest.get_artifact(rev_a.artifact_id)
        self.assertIsNotNone(stored_a)
        self.assertEqual(stored_a.sha256, sha_a)
        self.assertEqual(stored_a.size_bytes, len(data_a))

        stored_b = manifest.get_artifact(rev_b.artifact_id)
        self.assertIsNotNone(stored_b)
        self.assertEqual(stored_b.sha256, sha_b)
        self.assertEqual(stored_b.size_bytes, len(data_b))

        # Resolving unversioned ID via store.get_metadata returns Revision B (latest)
        unversioned_id = f"{project_id}/{run_id}/checkpoints/model.pt"
        meta_latest = self.store.get_metadata(unversioned_id)
        self.assertIsNotNone(meta_latest)
        self.assertEqual(meta_latest.artifact_id, rev_b.artifact_id)
        self.assertEqual(meta_latest.sha256, sha_b)

        # Step D: Retrieve Revision A explicitly by rev_a.artifact_id
        dest_a = Path(self.test_dir) / "retrieved_rev_a.pt"
        res_a = self.store.retrieve(rev_a.artifact_id, dest_a)
        self.assertEqual(res_a, dest_a)
        self.assertEqual(dest_a.read_bytes(), data_a)
        self.assertEqual(compute_file_sha256(dest_a), sha_a)

        # Step E: Retrieve Revision B explicitly by rev_b.artifact_id and via unversioned ID
        dest_b = Path(self.test_dir) / "retrieved_rev_b.pt"
        res_b = self.store.retrieve(rev_b.artifact_id, dest_b)
        self.assertEqual(res_b, dest_b)
        self.assertEqual(dest_b.read_bytes(), data_b)
        self.assertEqual(compute_file_sha256(dest_b), sha_b)

        dest_b_unversioned = Path(self.test_dir) / "retrieved_rev_b_unversioned.pt"
        res_b_unv = self.store.retrieve(unversioned_id, dest_b_unversioned)
        self.assertEqual(res_b_unv, dest_b_unversioned)
        self.assertEqual(dest_b_unversioned.read_bytes(), data_b)
        self.assertEqual(compute_file_sha256(dest_b_unversioned), sha_b)

        # Step F: Attempt to retrieve rev_a when remote file on Drive is corrupted/tampered
        self.mock_client.committed_files[rev_a.remote_file_id] = b"CORRUPTED_TAMPERED_BYTES_REVISION_A"
        dest_tampered = Path(self.test_dir) / "retrieved_tampered.pt"
        with self.assertRaises(ArtifactValidationError):
            self.store.retrieve(rev_a.artifact_id, dest_tampered)
        self.assertFalse(dest_tampered.exists(), "Destination file must not be left behind on integrity failure")

    def test_in_flight_source_mutation_staging_safety(self):
        """In-flight local file modification during upload does not corrupt staged upload bytes."""
        project_id = "p_staging"
        run_id = "r_staging"
        file_path = Path(self.test_dir) / "in_flight_model.pt"
        initial_data = b"INITIAL_CLEAN_WEIGHTS_SNAPSHOT" * 100
        file_path.write_bytes(initial_data)
        initial_sha = compute_file_sha256(file_path)

        original_request = self.mock_client.request

        # Mutate the source file on disk while the upload chunk is in flight
        def mutating_request(method, url, headers=None, data=None, timeout=30.0):
            if "session_id=upload-session" in url and method == "PUT":
                # Overwrite local source file during upload
                file_path.write_bytes(b"TRAINING_PROCESS_OVERWRITING_DURING_UPLOAD_IN_PLACE" * 100)
            return original_request(method, url, headers=headers, data=data, timeout=timeout)

        self.mock_client.request = mutating_request
        stored = self.store.publish(
            source=file_path,
            project_id=project_id,
            run_id=run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
        )

        # The uploaded file on Drive must match the initial clean snapshot, not the corrupted mutation!
        self.assertEqual(stored.sha256, initial_sha)
        self.assertEqual(self.mock_client.committed_files[stored.remote_file_id], initial_data)

    def test_source_mutation_during_staging_raises_transfer_error(self):
        """Mutation of source file during staging phase raises ArtifactTransferError."""
        from colab_adapter.storage.google_drive import ArtifactTransferError

        file_path = Path(self.test_dir) / "mutate_during_stage.pt"
        file_path.write_bytes(b"DATA_BEFORE_MUTATION")

        original_stage = self.store._stage_file

        def mutating_stage(src, tmp_staging):
            original_stage(src, tmp_staging)
            tmp_staging.write_bytes(b"MUTATED_DURING_STAGING")
            return len(b"MUTATED_DURING_STAGING"), hashlib.sha256(b"MUTATED_DURING_STAGING").hexdigest()

        self.store._stage_file = mutating_stage
        try:
            with self.assertRaises(ArtifactTransferError) as ctx:
                self.store.publish(
                    source=file_path,
                    project_id="p_mut",
                    run_id="r_mut",
                )
            self.assertIn("Source file mutated during staging", str(ctx.exception))
        finally:
            self.store._stage_file = original_stage


    def test_01_manifest_missing_exact_revision_found_among_multiple_files(self):
        """1. Manifest missing, exact revision @sha256_<FULL_HASH> exists among multiple model.pt files on Drive."""
        project_id = "proj_m1"
        run_id = "run_m1"
        file_path = Path(self.test_dir) / "model.pt"

        # Publish rev 1
        data_1 = b"MODEL_WEIGHTS_EPOCH_1_REV1"
        file_path.write_bytes(data_1)
        sha_1 = compute_file_sha256(file_path)
        art_1 = self.store.publish(
            source=file_path,
            project_id=project_id,
            run_id=run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
        )

        # Publish rev 2
        data_2 = b"MODEL_WEIGHTS_EPOCH_2_REV2_LARGER"
        file_path.write_bytes(data_2)
        sha_2 = compute_file_sha256(file_path)
        art_2 = self.store.publish(
            source=file_path,
            project_id=project_id,
            run_id=run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
        )

        # Destroy local manifest, backup, cache, and journal
        manifest_path = self.staging_dir / "projects" / project_id / "runs" / run_id / "manifest.json"
        bak_path = manifest_path.with_suffix(".json.bak")
        if manifest_path.exists():
            manifest_path.unlink()
        if bak_path.exists():
            bak_path.unlink()
        self.store._manifest_cache.clear()
        self.store.journal._write_records({})

        # Retrieve exact rev 1 by @sha256_<FULL_HASH>
        dest_1 = Path(self.test_dir) / "retrieved_m1_rev1.pt"
        self.store.retrieve(f"{project_id}/{run_id}/checkpoints/model.pt@sha256_{sha_1}", dest_1)
        self.assertEqual(dest_1.read_bytes(), data_1)
        self.assertEqual(compute_file_sha256(dest_1), sha_1)

        # Retrieve exact rev 2 by @sha256_<FULL_HASH>
        dest_2 = Path(self.test_dir) / "retrieved_m1_rev2.pt"
        self.store.retrieve(f"{project_id}/{run_id}/checkpoints/model.pt@sha256_{sha_2}", dest_2)
        self.assertEqual(dest_2.read_bytes(), data_2)
        self.assertEqual(compute_file_sha256(dest_2), sha_2)

    def test_02_manifest_missing_same_name_same_size_different_hashes(self):
        """2. Manifest missing, two model.pt files have the same name and size but different hashes."""
        project_id = "proj_m2"
        run_id = "run_m2"
        folder_id = self.store.resolve_target_folder_id(project_id, run_id, "checkpoints")

        data_a = b"AAAA_16_BYTES_!!"
        data_b = b"BBBB_16_BYTES_??"
        self.assertEqual(len(data_a), len(data_b))
        sha_a = hashlib.sha256(data_a).hexdigest()
        sha_b = hashlib.sha256(data_b).hexdigest()

        # Seed mock Drive folder directly with two same-name, same-size files
        self.mock_client.mock_files.append({
            "id": "file-rev-a",
            "name": "model.pt",
            "parent": folder_id,
            "size": len(data_a),
            "md5Checksum": hashlib.md5(data_a).hexdigest(),
            "appProperties": {
                "sha256": sha_a,
                "logical_path": "checkpoints/model.pt",
                "revision_id": f"sha256_{sha_a}",
            },
            "content": data_a,
        })
        self.mock_client.mock_files.append({
            "id": "file-rev-b",
            "name": "model.pt",
            "parent": folder_id,
            "size": len(data_b),
            "md5Checksum": hashlib.md5(data_b).hexdigest(),
            "appProperties": {
                "sha256": sha_b,
                "logical_path": "checkpoints/model.pt",
                "revision_id": f"sha256_{sha_b}",
            },
            "content": data_b,
        })
        self.mock_client.committed_files["file-rev-a"] = data_a
        self.mock_client.committed_files["file-rev-b"] = data_b

        # Clear local state
        self.store._manifest_cache.clear()
        self.store.journal._write_records({})

        # Exact request for rev A selects A
        dest_a = Path(self.test_dir) / "retrieved_m2_a.pt"
        self.store.retrieve(f"{project_id}/{run_id}/checkpoints/model.pt@sha256_{sha_a}", dest_a)
        self.assertEqual(dest_a.read_bytes(), data_a)

        # Exact request for rev B selects B
        dest_b = Path(self.test_dir) / "retrieved_m2_b.pt"
        self.store.retrieve(f"{project_id}/{run_id}/checkpoints/model.pt@sha256_{sha_b}", dest_b)
        self.assertEqual(dest_b.read_bytes(), data_b)

        # Nonexistent hash fails closed
        dest_bad = Path(self.test_dir) / "retrieved_m2_bad.pt"
        with self.assertRaises(DriveIntegrityError):
            self.store.retrieve(f"{project_id}/{run_id}/checkpoints/model.pt@sha256_{'0' * 64}", dest_bad)
        self.assertFalse(dest_bad.exists())

    def test_03_manifest_missing_unversioned_multiple_files_fails_closed(self):
        """3. Manifest missing, unversioned lookup when two revisions exist on Drive fails closed with DriveIntegrityError."""
        project_id = "proj_m3"
        run_id = "run_m3"
        folder_id = self.store.resolve_target_folder_id(project_id, run_id, "checkpoints")

        data_1 = b"UNVERSIONED_CANDIDATE_1"
        data_2 = b"UNVERSIONED_CANDIDATE_2"
        sha_1 = hashlib.sha256(data_1).hexdigest()
        sha_2 = hashlib.sha256(data_2).hexdigest()

        # Seed two files with NO is_latest=="true" marker
        self.mock_client.mock_files.append({
            "id": "file-unv-1",
            "name": "model.pt",
            "parent": folder_id,
            "size": len(data_1),
            "md5Checksum": hashlib.md5(data_1).hexdigest(),
            "appProperties": {"sha256": sha_1, "logical_path": "checkpoints/model.pt"},
            "content": data_1,
        })
        self.mock_client.mock_files.append({
            "id": "file-unv-2",
            "name": "model.pt",
            "parent": folder_id,
            "size": len(data_2),
            "md5Checksum": hashlib.md5(data_2).hexdigest(),
            "appProperties": {"sha256": sha_2, "logical_path": "checkpoints/model.pt"},
            "content": data_2,
        })

        self.store._manifest_cache.clear()
        self.store.journal._write_records({})

        dest = Path(self.test_dir) / "retrieved_m3.pt"
        with self.assertRaises(DriveIntegrityError) as ctx:
            self.store.retrieve(f"{project_id}/{run_id}/checkpoints/model.pt", dest)
        self.assertIn("Ambiguous unversioned Drive recovery", str(ctx.exception))
        self.assertFalse(dest.exists())

    def test_04_corrupted_manifest_recovered_from_backup(self):
        """4. Corrupted manifest.json with valid manifest.json.bak automatically recovers manifest."""
        project_id = "proj_m4"
        run_id = "run_m4"
        file_path = Path(self.test_dir) / "model.pt"

        # Publish v1 then v2
        file_path.write_bytes(b"MODEL_V1")
        self.store.publish(file_path, project_id, run_id, ArtifactType.CHECKPOINT.value)

        file_path.write_bytes(b"MODEL_V2_LATEST_BYTES")
        art_2 = self.store.publish(file_path, project_id, run_id, ArtifactType.CHECKPOINT.value)

        manifest_path = self.staging_dir / "projects" / project_id / "runs" / run_id / "manifest.json"
        bak_path = manifest_path.with_suffix(".json.bak")
        self.assertTrue(manifest_path.exists())
        self.assertTrue(bak_path.exists())

        # Check backup file permissions are 0600
        mode = stat.S_IMODE(bak_path.stat().st_mode)
        self.assertEqual(mode, 0o600)

        # Corrupt primary manifest
        manifest_path.write_text("{CORRUPTED_JSON_NOT_VALID_SYNTAX...")

        # Clear in-memory manifest cache
        self.store._manifest_cache.clear()

        dest = Path(self.test_dir) / "recovered_from_bak.pt"
        self.store.retrieve(f"{project_id}/{run_id}/checkpoints/model.pt", dest)
        self.assertEqual(dest.read_bytes(), b"MODEL_V2_LATEST_BYTES")

        # Primary manifest must be automatically restored and valid JSON
        self.assertTrue(manifest_path.exists())
        loaded = PublicationManifest.load(manifest_path)
        self.assertIsNotNone(loaded.get_artifact(art_2.artifact_id))

    def test_05_wrong_remote_file_content_raises_integrity_error_and_cleans_tmp(self):
        """5. Wrong remote file returned by Drive mock raises DriveIntegrityError and cleans up temp file."""
        project_id = "proj_m5"
        run_id = "run_m5"
        file_path = Path(self.test_dir) / "model.pt"
        file_path.write_bytes(b"CORRECT_AUTHENTIC_BYTES")
        art = self.store.publish(file_path, project_id, run_id, ArtifactType.CHECKPOINT.value)

        # Tamper committed remote file bytes
        self.mock_client.committed_files[art.remote_file_id] = b"WRONG_TAMPERED_BYTES_FROM_DRIVE"

        dest = Path(self.test_dir) / "dest_m5.pt"
        with self.assertRaises(DriveIntegrityError):
            self.store.retrieve(art.artifact_id, dest)

        self.assertFalse(dest.exists())
        # Verify no .tmp leftover files in destination directory
        tmp_files = list(dest.parent.glob("*.tmp*"))
        self.assertEqual(len(tmp_files), 0, f"Leftover temp files found: {tmp_files}")

    def test_06_corrupted_bytes_from_drive_raises_integrity_error(self):
        """6. Correct remote ID in manifest, but Drive returns corrupted bytes -> raises DriveIntegrityError."""
        project_id = "proj_m6"
        run_id = "run_m6"
        file_path = Path(self.test_dir) / "model.pt"
        file_path.write_bytes(b"GOOD_BYTES_REV6")
        art = self.store.publish(file_path, project_id, run_id, ArtifactType.CHECKPOINT.value)

        self.mock_client.corrupt_download_bytes = b"CORRUPTED_BYTE_STREAM_PAYLOAD"

        dest = Path(self.test_dir) / "dest_m6.pt"
        with self.assertRaises(DriveIntegrityError):
            self.store.retrieve(art.artifact_id, dest)

        self.assertFalse(dest.exists())

    def test_07_omitted_md5_verified_via_full_sha256(self):
        """7. Remote metadata omits md5Checksum -> verified via full SHA256 content readback."""
        project_id = "proj_m7"
        run_id = "run_m7"
        file_path = Path(self.test_dir) / "model.pt"
        data = b"PAYLOAD_WITHOUT_REMOTE_MD5"
        file_path.write_bytes(data)
        art = self.store.publish(file_path, project_id, run_id, ArtifactType.CHECKPOINT.value)

        # Clear MD5 in manifest
        manifest_path = self.staging_dir / "projects" / project_id / "runs" / run_id / "manifest.json"
        manifest = PublicationManifest.load(manifest_path)
        for k in manifest.artifacts:
            manifest.artifacts[k]["md5"] = None
        manifest.save(manifest_path)
        self.store._manifest_cache.clear()

        # Clear MD5 in mock Drive file
        for mf in self.mock_client.mock_files:
            if mf.get("id") == art.remote_file_id:
                mf["md5Checksum"] = ""

        dest = Path(self.test_dir) / "dest_m7.pt"
        self.store.retrieve(art.artifact_id, dest)
        self.assertEqual(dest.read_bytes(), data)
        self.assertEqual(compute_file_sha256(dest), art.sha256)

    def test_08_truncated_download_raises_integrity_error(self):
        """8. Drive returns incomplete content (truncated byte stream) -> raises DriveIntegrityError."""
        project_id = "proj_m8"
        run_id = "run_m8"
        file_path = Path(self.test_dir) / "model.pt"
        file_path.write_bytes(b"LONG_PAYLOAD_DATA_BYTES_TO_BE_TRUNCATED" * 10)
        art = self.store.publish(file_path, project_id, run_id, ArtifactType.CHECKPOINT.value)

        self.mock_client.truncate_download_bytes_to = 25  # smaller than size_bytes

        dest = Path(self.test_dir) / "dest_m8.pt"
        with self.assertRaises(DriveIntegrityError):
            self.store.retrieve(art.artifact_id, dest)

        self.assertFalse(dest.exists())

    def test_09_network_interruption_during_retrieval_raises_transfer_error(self):
        """9. Network interruption during retrieval -> raises DriveTransferError, cleans up .tmp file."""
        project_id = "proj_m9"
        run_id = "run_m9"
        file_path = Path(self.test_dir) / "model.pt"
        file_path.write_bytes(b"DATA_BEFORE_NETWORK_CUT")
        art = self.store.publish(file_path, project_id, run_id, ArtifactType.CHECKPOINT.value)

        self.mock_client.fail_download_network_error = True

        dest = Path(self.test_dir) / "dest_m9.pt"
        with self.assertRaises(DriveTransferError):
            self.store.retrieve(art.artifact_id, dest)

        self.assertFalse(dest.exists())
        tmp_files = list(dest.parent.glob("*.tmp*"))
        self.assertEqual(len(tmp_files), 0, f"Leftover temp files: {tmp_files}")

    def test_10_historical_revision_a_retrieval_after_revision_b_published(self):
        """10. Historical revision A retrieval after revision B is published -> returns exact revision A bytes."""
        project_id = "proj_m10"
        run_id = "run_m10"
        file_path = Path(self.test_dir) / "checkpoint.pt"

        data_a = b"HISTORICAL_REVISION_A_DATA"
        file_path.write_bytes(data_a)
        art_a = self.store.publish(file_path, project_id, run_id, ArtifactType.CHECKPOINT.value)

        data_b = b"LATEST_REVISION_B_NEW_DATA_LONGER"
        file_path.write_bytes(data_b)
        art_b = self.store.publish(file_path, project_id, run_id, ArtifactType.CHECKPOINT.value)

        # Retrieve Rev A by explicit ID
        dest_a = Path(self.test_dir) / "dest_m10_a.pt"
        self.store.retrieve(art_a.artifact_id, dest_a)
        self.assertEqual(dest_a.read_bytes(), data_a)

        # Retrieve Rev B by explicit ID
        dest_b = Path(self.test_dir) / "dest_m10_b.pt"
        self.store.retrieve(art_b.artifact_id, dest_b)
        self.assertEqual(dest_b.read_bytes(), data_b)

        # Retrieve unversioned returns Rev B
        dest_latest = Path(self.test_dir) / "dest_m10_latest.pt"
        self.store.retrieve(f"{project_id}/{run_id}/checkpoints/checkpoint.pt", dest_latest)
        self.assertEqual(dest_latest.read_bytes(), data_b)

    def test_11_recovery_after_process_restart(self):
        """11. Recovery after simulated Ubuntu process restart (fresh store instance on disk directory)."""
        project_id = "proj_m11"
        run_id = "run_m11"
        file_path = Path(self.test_dir) / "model.pt"
        data = b"PROCESS_RESTART_PERSISTENCE_TEST"
        file_path.write_bytes(data)
        art = self.store.publish(file_path, project_id, run_id, ArtifactType.CHECKPOINT.value)

        # Fresh instance
        fresh_store = GoogleDriveArtifactStore(
            credentials=self.creds,
            credentials_file=self.credentials_file,
            local_staging_dir=self.staging_dir,
            http_client=self.mock_client,
            chunk_size=1024 * 1024,
            initial_backoff_sec=0.01,
        )

        dest = Path(self.test_dir) / "dest_m11.pt"
        fresh_store.retrieve(art.artifact_id, dest)
        self.assertEqual(dest.read_bytes(), data)

        dest_unv = Path(self.test_dir) / "dest_m11_unv.pt"
        fresh_store.retrieve(f"{project_id}/{run_id}/checkpoints/model.pt", dest_unv)
        self.assertEqual(dest_unv.read_bytes(), data)

    def test_12_retrieval_failure_preserves_previously_verified_dest_file(self):
        """12. Retrieval failure preserves previously verified local destination file intact."""
        project_id = "proj_m12"
        run_id = "run_m12"
        file_path = Path(self.test_dir) / "model.pt"
        file_path.write_bytes(b"PUBLISHED_DATA")
        art = self.store.publish(file_path, project_id, run_id, ArtifactType.CHECKPOINT.value)

        # Pre-seed destination with good verified data
        dest = Path(self.test_dir) / "dest_m12_existing.pt"
        original_good_data = b"PREVIOUS_VERIFIED_GOOD_DATA_DO_NOT_CORRUPT"
        dest.write_bytes(original_good_data)

        # Make download fail due to corrupted bytes
        self.mock_client.corrupt_download_bytes = b"CORRUPTED_DOWNLOAD_ATTEMPT"

        with self.assertRaises(DriveIntegrityError):
            self.store.retrieve(art.artifact_id, dest)

        # Ensure destination content was preserved completely untouched
        self.assertEqual(dest.read_bytes(), original_good_data)

    def test_13_private_staging_permissions_and_symlink_rejection(self):
        """13. Private staging permissions under os.umask(0): .upload_staging is 0700, files 0600, symlinks rejected."""
        old_umask = os.umask(0)
        try:
            project_id = "proj_m13"
            run_id = "run_m13"
            file_path = Path(self.test_dir) / "open_perms_model.pt"
            file_path.write_bytes(b"WEIGHTS_WITH_0777_PERMS")
            os.chmod(file_path, 0o777)

            art = self.store.publish(file_path, project_id, run_id, ArtifactType.CHECKPOINT.value)

            # Check .upload_staging mode is strictly 0700
            staging_dir = self.staging_dir / ".upload_staging"
            self.assertTrue(staging_dir.exists())
            st_dir_mode = stat.S_IMODE(staging_dir.stat().st_mode)
            self.assertEqual(st_dir_mode, 0o700, f"Expected 0700 for staging_dir, got {oct(st_dir_mode)}")

            # Check backup manifest mode is 0600
            bak_path = self.staging_dir / "projects" / project_id / "runs" / run_id / "manifest.json.bak"
            self.assertTrue(bak_path.exists())
            bak_mode = stat.S_IMODE(bak_path.stat().st_mode)
            self.assertEqual(bak_mode, 0o600, f"Expected 0600 for backup manifest, got {oct(bak_mode)}")

            # Symlink source rejection
            symlink_src = Path(self.test_dir) / "symlink_source.pt"
            symlink_src.symlink_to(file_path)
            with self.assertRaises(ArtifactValidationError):
                self.store.publish(symlink_src, project_id, run_id)

            # Fake .upload_staging outside trusted root is rejected by _is_trusted_staged_file
            fake_dir = Path(self.test_dir) / "fake" / ".upload_staging"
            fake_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            fake_file = fake_dir / "fake.pt"
            fake_file.write_bytes(b"FAKE_STAGED")
            self.assertFalse(self.store._is_trusted_staged_file(fake_file))
        finally:
            os.umask(old_umask)

    # =========================================================================
    # Section 6.4: Full Bounded Pagination Tests (F3)
    # =========================================================================

    def test_6_4_1_single_page_with_no_next_page_token(self):
        """6.4.1. Single page listing with no nextPageToken returns all files."""
        folder_id = "folder-page-1"
        self.mock_client.mock_files.extend([
            {"id": "file-101", "name": "weights.pt", "parent": folder_id, "size": 100, "mimeType": "application/octet-stream"},
            {"id": "file-102", "name": "weights.pt", "parent": folder_id, "size": 200, "mimeType": "application/octet-stream"},
        ])
        files = self.store._list_files_in_folder("weights.pt", folder_id)
        self.assertEqual(len(files), 2)
        self.assertEqual([f["id"] for f in files], ["file-101", "file-102"])

    def test_6_4_2_two_pages_traversal(self):
        """6.4.2. Two pages traversal (nextPageToken on page 1, none on page 2)."""
        folder_id = "folder-page-2"
        self.mock_client.list_page_size_override = 2
        self.mock_client.mock_files.extend([
            {"id": "file-201", "name": "model.bin", "parent": folder_id, "size": 10, "mimeType": "application/octet-stream"},
            {"id": "file-202", "name": "model.bin", "parent": folder_id, "size": 20, "mimeType": "application/octet-stream"},
            {"id": "file-203", "name": "model.bin", "parent": folder_id, "size": 30, "mimeType": "application/octet-stream"},
            {"id": "file-204", "name": "model.bin", "parent": folder_id, "size": 40, "mimeType": "application/octet-stream"},
        ])
        files = self.store._list_files_in_folder("model.bin", folder_id)
        self.assertEqual(len(files), 4)
        self.assertEqual([f["id"] for f in files], ["file-201", "file-202", "file-203", "file-204"])

    def test_6_4_3_five_pages_with_target_on_page_five(self):
        """6.4.3. Five pages traversal where target candidate revision is on page 5."""
        project_id = "proj_p5"
        run_id = "run_p5"
        folder_id = self.store.resolve_target_folder_id(project_id, run_id, "checkpoints")

        # 12 candidate dummy files + 1 target candidate = 13 files across 5 pages of size 3
        self.mock_client.list_page_size_override = 3
        target_data = b"TARGET_REVISION_ON_PAGE_FIVE"
        target_sha = hashlib.sha256(target_data).hexdigest()

        for i in range(12):
            dummy_data = f"DUMMY_REVISION_{i}".encode("utf-8")
            self.mock_client.mock_files.append({
                "id": f"dummy-cand-{i:02d}",
                "name": "model.pt",
                "parent": folder_id,
                "content": dummy_data,
                "mimeType": "application/octet-stream",
                "appProperties": {
                    "sha256": hashlib.sha256(dummy_data).hexdigest(),
                    "is_latest": "false",
                },
            })

        # 13th file is on page 5 (4 pages * 3 items = 12 items on pages 1-4)
        self.mock_client.mock_files.append({
            "id": "target-cand-page5",
            "name": "model.pt",
            "parent": folder_id,
            "content": target_data,
            "mimeType": "application/octet-stream",
            "appProperties": {
                "sha256": target_sha,
                "is_latest": "true",
            },
        })

        # Case A: Versioned retrieval finding target on page 5
        dest_ver = Path(self.test_dir) / "dest_ver_p5.pt"
        self.store.retrieve(f"{project_id}/{run_id}/checkpoints/model.pt@sha256_{target_sha}", dest_ver)
        self.assertEqual(dest_ver.read_bytes(), target_data)

        # Case B: Unversioned retrieval finding is_latest="true" on page 5 (with no local manifest)
        dest_unv = Path(self.test_dir) / "dest_unv_p5.pt"
        self.store.retrieve(f"{project_id}/{run_id}/checkpoints/model.pt", dest_unv)
        self.assertEqual(dest_unv.read_bytes(), target_data)

    def test_6_4_4_duplicate_file_ids_across_pages_deduplicated(self):
        """6.4.4. Duplicate file IDs across pages deduplicated deterministically."""
        folder_id = "folder-dedup"
        self.mock_client.list_page_size_override = 2
        # Page 1: dup-1, file-2
        # Page 2: dup-1, file-3
        self.mock_client.mock_files.extend([
            {"id": "file-dup-1", "name": "data.bin", "parent": folder_id, "size": 10, "mimeType": "application/octet-stream"},
            {"id": "file-2", "name": "data.bin", "parent": folder_id, "size": 20, "mimeType": "application/octet-stream"},
            {"id": "file-dup-1", "name": "data.bin", "parent": folder_id, "size": 10, "mimeType": "application/octet-stream"},
            {"id": "file-3", "name": "data.bin", "parent": folder_id, "size": 30, "mimeType": "application/octet-stream"},
        ])
        files = self.store._list_files_in_folder("data.bin", folder_id)
        self.assertEqual(len(files), 3)
        self.assertEqual([f["id"] for f in files], ["file-dup-1", "file-2", "file-3"])

    def test_6_4_5_repeating_looping_page_token_raises_error(self):
        """6.4.5. Repeating/looping nextPageToken raises DriveTransferError."""
        folder_id = "folder-loop"
        self.mock_client.inject_looping_page_token = True
        self.mock_client.mock_files.extend([
            {"id": "file-l1", "name": "loop.bin", "parent": folder_id, "size": 10, "mimeType": "application/octet-stream"},
            {"id": "file-l2", "name": "loop.bin", "parent": folder_id, "size": 20, "mimeType": "application/octet-stream"},
        ])
        with self.assertRaises(DriveTransferError) as cm:
            self.store._list_files_in_folder("loop.bin", folder_id)
        self.assertIn("loop", str(cm.exception).lower())

    def test_6_4_6_exceeding_max_list_pages_raises_error(self):
        """6.4.6. Exceeding max_list_pages while nextPageToken remains raises DriveTransferError."""
        folder_id = "folder-max-pages"
        custom_store = GoogleDriveArtifactStore(
            credentials=self.creds,
            credentials_file=self.credentials_file,
            local_staging_dir=self.staging_dir,
            http_client=self.mock_client,
            list_page_size=2,
            max_list_pages=2,
            initial_backoff_sec=0.01,
        )
        self.mock_client.mock_files.extend([
            {"id": "f-1", "name": "overflow.pt", "parent": folder_id, "size": 1, "mimeType": "application/octet-stream"},
            {"id": "f-2", "name": "overflow.pt", "parent": folder_id, "size": 2, "mimeType": "application/octet-stream"},
            {"id": "f-3", "name": "overflow.pt", "parent": folder_id, "size": 3, "mimeType": "application/octet-stream"},
            {"id": "f-4", "name": "overflow.pt", "parent": folder_id, "size": 4, "mimeType": "application/octet-stream"},
            {"id": "f-5", "name": "overflow.pt", "parent": folder_id, "size": 5, "mimeType": "application/octet-stream"},
        ])
        with self.assertRaises(DriveTransferError) as cm:
            custom_store._list_files_in_folder("overflow.pt", folder_id)
        self.assertIn("max_list_pages", str(cm.exception))

    def test_6_4_7_exceeding_max_list_candidates_raises_error(self):
        """6.4.7. Exceeding max_list_candidates raises DriveTransferError."""
        folder_id = "folder-max-cands"
        custom_store = GoogleDriveArtifactStore(
            credentials=self.creds,
            credentials_file=self.credentials_file,
            local_staging_dir=self.staging_dir,
            http_client=self.mock_client,
            list_page_size=10,
            max_list_candidates=3,
            initial_backoff_sec=0.01,
        )
        for i in range(5):
            self.mock_client.mock_files.append({
                "id": f"cand-{i}", "name": "cands.pt", "parent": folder_id, "size": 1, "mimeType": "application/octet-stream"
            })
        with self.assertRaises(DriveTransferError) as cm:
            custom_store._list_files_in_folder("cands.pt", folder_id)
        self.assertIn("max_list_candidates", str(cm.exception))

    def test_6_4_8_mid_pagination_failure_raises_error(self):
        """6.4.8. Mid-pagination failure on page 2 (503 or malformed JSON) raises DriveTransferError."""
        folder_id = "folder-mid-fail"
        self.mock_client.list_page_size_override = 2
        for i in range(6):
            self.mock_client.mock_files.append({
                "id": f"mid-{i}", "name": "fail.pt", "parent": folder_id, "size": 1, "mimeType": "application/octet-stream"
            })

        # Subtest A: HTTP 503 on page 2
        self.mock_client.fail_list_files_status = 503
        self.mock_client.fail_list_files_at_page = 2
        with self.assertRaises(DriveTransferError):
            self.store._list_files_in_folder("fail.pt", folder_id)

        # Subtest B: Malformed JSON on page 2
        self.mock_client.fail_list_files_status = None
        self.mock_client.fail_list_files_body = b"NOT_VALID_JSON_ON_PAGE_2"
        self.mock_client.fail_list_files_at_page = 2
        with self.assertRaises(DriveTransferError) as cm:
            self.store._list_files_in_folder("fail.pt", folder_id)
        self.assertIn("Malformed JSON", str(cm.exception))

    def test_6_4_9_malformed_page_token_raises_error(self):
        """6.4.9. Malformed non-string or whitespace nextPageToken raises DriveTransferError."""
        folder_id = "folder-mal-token"
        self.mock_client.inject_malformed_page_token = 12345
        self.mock_client.mock_files.append({
            "id": "f-tok-1", "name": "tok.pt", "parent": folder_id, "size": 1, "mimeType": "application/octet-stream"
        })
        with self.assertRaises(DriveTransferError) as cm:
            self.store._list_files_in_folder("tok.pt", folder_id)
        self.assertIn("Malformed non-string", str(cm.exception))

        self.mock_client.inject_malformed_page_token = "   "
        with self.assertRaises(DriveTransferError) as cm:
            self.store._list_files_in_folder("tok.pt", folder_id)
        self.assertIn("whitespace nextPageToken", str(cm.exception))

    # =========================================================================
    # Section 7.4: Listing Error Classification & No Duplicate Upload Tests (F4)
    # =========================================================================

    def test_7_4_1_listing_http_401_raises_authentication_error(self):
        """7.4.1. HTTP 401 during _list_files_in_folder raises DriveAuthenticationError."""
        self.mock_client.fail_list_files_status = 401
        self.mock_client.auth_expired = True
        with self.assertRaises(DriveAuthenticationError):
            self.store._list_files_in_folder("test.pt", "folder-1")

    def test_7_4_2_listing_http_403_quota_and_forbidden_classification(self):
        """7.4.2. HTTP 403 quota/permission during _list_files_in_folder classified strictly."""
        # Case A: Storage quota exceeded
        self.mock_client.fail_list_files_status = 403
        self.mock_client.fail_list_files_body = b'{"error": {"errors": [{"reason": "storageQuotaExceeded"}]}}'
        with self.assertRaises(DriveQuotaExceededError):
            self.store._list_files_in_folder("test.pt", "folder-1")

        # Case B: Forbidden / permission denied
        self.mock_client.fail_list_files_body = b'{"error": {"errors": [{"reason": "forbidden"}]}}'
        with self.assertRaises(DriveAuthenticationError):
            self.store._list_files_in_folder("test.pt", "folder-1")

    def test_7_4_3_listing_http_server_errors_raise_transfer_error(self):
        """7.4.3. HTTP 429/500/503 during _list_files_in_folder raise DriveTransferError."""
        for code in (429, 500, 503):
            self.mock_client.fail_list_files_status = code
            self.mock_client.fail_list_files_body = f'{{"error": "HTTP {code}"}}'.encode("utf-8")
            with self.assertRaises(DriveTransferError):
                self.store._list_files_in_folder("test.pt", "folder-1")

    def test_7_4_4_listing_network_exception_raises_transfer_error(self):
        """7.4.4. Network timeout or ConnectionResetError during _list_files_in_folder raises DriveTransferError."""
        self.mock_client.fail_list_files_exception = ConnectionResetError("Connection reset by peer")
        with self.assertRaises(DriveTransferError) as cm:
            self.store._list_files_in_folder("test.pt", "folder-1")
        self.assertIn("Connection reset by peer", str(cm.exception))

    def test_7_4_5_listing_malformed_payload_structures_raise_transfer_error(self):
        """7.4.5. Invalid JSON, non-dict JSON, missing/non-list 'files' key raise DriveTransferError."""
        cases = [
            (b"NOT_JSON", "Malformed JSON"),
            (b"[1, 2, 3]", "missing 'files' list"),
            (b'{"no_files": []}', "missing 'files' list"),
            (b'{"files": "not_a_list"}', "missing 'files' list"),
            (b'{"files": ["not_a_dict"]}', "missing string 'id'"),
            (b'{"files": [{"name": "no_id"}]}', "missing string 'id'"),
            (b'{"files": [{"id": 12345}]}', "missing string 'id'"),
        ]
        for body, expected_err in cases:
            self.mock_client.fail_list_files_body = body
            with self.assertRaises(DriveTransferError) as cm:
                self.store._list_files_in_folder("test.pt", "folder-1")
            self.assertIn(expected_err, str(cm.exception))

    def test_7_4_6_listing_valid_empty_files_returns_empty_list(self):
        """7.4.6. Valid HTTP 200 with {"files": []} returns empty list."""
        self.mock_client.mock_files.clear()
        files = self.store._list_files_in_folder("non_existent.pt", "folder-empty")
        self.assertEqual(files, [])

    def test_7_4_7_publish_does_not_initiate_duplicate_upload_on_listing_failure(self):
        """7.4.7. Publish raises DriveTransferError and does NOT initiate upload session on listing failure."""
        project_id = "proj_dup_prev"
        run_id = "run_dup_prev"
        file_path = Path(self.test_dir) / "dup_model.pt"
        data = b"MODEL_DATA_FOR_DUPLICATE_PREVENTION_TEST"
        file_path.write_bytes(data)

        # Pre-resolve target folder
        target_folder_id = self.store.resolve_target_folder_id(project_id, run_id, "checkpoints")

        # Pre-populate remote file as already existing on Drive
        self.mock_client.mock_files.append({
            "id": "existing-drive-file-999",
            "name": file_path.name,
            "parent": target_folder_id,
            "content": data,
            "mimeType": "application/octet-stream",
            "appProperties": {
                "sha256": hashlib.sha256(data).hexdigest(),
                "is_latest": "true",
            },
        })

        initial_upload_sessions = self.mock_client.upload_sessions_initiated

        # Subtest A: Listing fails with HTTP 503
        self.mock_client.fail_list_files_status = 503
        with self.assertRaises(DriveTransferError):
            self.store.publish(file_path, project_id, run_id, ArtifactType.CHECKPOINT.value)

        # Crucial assertion: no upload session was initiated
        self.assertEqual(self.mock_client.upload_sessions_initiated, initial_upload_sessions)

        # Subtest B: Listing fails with malformed JSON
        self.mock_client.fail_list_files_status = None
        self.mock_client.fail_list_files_body = b"GARBAGE_JSON"
        with self.assertRaises(DriveTransferError):
            self.store.publish(file_path, project_id, run_id, ArtifactType.CHECKPOINT.value)

        # Still no upload session was initiated
        self.assertEqual(self.mock_client.upload_sessions_initiated, initial_upload_sessions)

    def test_7_4_8_find_or_create_folder_failure_does_not_send_post(self):
        """7.4.8. _find_or_create_folder failing with HTTP 500/503 raises DriveTransferError and does not POST."""
        self.mock_client.fail_folder_lookup_status = 503
        initial_posts = [r for r in self.mock_client.requests_log if r[0] == "POST"]

        with self.assertRaises(DriveTransferError):
            self.store._find_or_create_folder("fail_folder", parent_id="root")

        current_posts = [r for r in self.mock_client.requests_log if r[0] == "POST"]
        self.assertEqual(len(current_posts), len(initial_posts))

        # Also test malformed JSON on folder lookup
        self.mock_client.fail_folder_lookup_status = None
        self.mock_client.fail_folder_lookup_body = b"NOT_JSON_FOLDER"
        with self.assertRaises(DriveTransferError):
            self.store._find_or_create_folder("fail_folder_2", parent_id="root")

        current_posts = [r for r in self.mock_client.requests_log if r[0] == "POST"]
        self.assertEqual(len(current_posts), len(initial_posts))

    def test_7_4_9_listing_parent_folder_404_raises_artifact_not_found(self):
        """7.4.9. Parent folder HTTP 404 during _list_files_in_folder raises ArtifactNotFoundError."""
        self.mock_client.fail_list_files_status = 404
        self.mock_client.fail_list_files_body = b'{"error": {"code": 404, "message": "File not found"}}'
        with self.assertRaises(ArtifactNotFoundError):
            self.store._list_files_in_folder("test.pt", "missing-folder-id")

    # =========================================================================
    # Additional Verification: Init Param Validation & Error Propagation
    # =========================================================================

    def test_init_pagination_parameter_validation(self):
        """GoogleDriveArtifactStore validates pagination parameters >= 1."""
        with self.assertRaises(ValueError):
            GoogleDriveArtifactStore(list_page_size=0)
        with self.assertRaises(ValueError):
            GoogleDriveArtifactStore(list_page_size=-5)
        with self.assertRaises(ValueError):
            GoogleDriveArtifactStore(max_list_pages=0)
        with self.assertRaises(ValueError):
            GoogleDriveArtifactStore(max_list_candidates=0)

        store = GoogleDriveArtifactStore(
            list_page_size=25,
            max_list_pages=10,
            max_list_candidates=500,
        )
        self.assertEqual(store.list_page_size, 25)
        self.assertEqual(store.max_list_pages, 10)
        self.assertEqual(store.max_list_candidates, 500)

    def test_retrieve_and_exists_error_propagation(self):
        """retrieve() and exists() re-raise DriveTransferError / DriveAuthenticationError."""
        project_id = "proj_err_prop"
        run_id = "run_err_prop"
        dest = Path(self.test_dir) / "err_dest.pt"

        # Folder lookup failure during retrieve() raises DriveTransferError, NOT ArtifactNotFoundError
        self.mock_client.fail_folder_lookup_status = 503
        with self.assertRaises(DriveTransferError):
            self.store.retrieve(f"{project_id}/{run_id}/checkpoints/model.pt", dest)

        # File listing failure during exists() re-raises DriveTransferError, does NOT return False
        self.mock_client.fail_folder_lookup_status = None
        self.mock_client.fail_list_files_status = 503
        with self.assertRaises(DriveTransferError):
            self.store.exists(f"{project_id}/{run_id}/checkpoints/model.pt")

        # 401 during exists() re-raises DriveAuthenticationError
        self.mock_client.fail_list_files_status = 401
        self.mock_client.auth_expired = True
        with self.assertRaises(DriveAuthenticationError):
            self.store.exists(f"{project_id}/{run_id}/checkpoints/model.pt")


if __name__ == "__main__":
    unittest.main()
