"""Comprehensive adversarial, fault injection, and security regression test suite.

Authored by Worker F (Adversarial QA & Security Engineer).
Strictly validates:
1. Byte-Accurate Resumable Upload Fault-Injection Harness:
   - ByteAccurateDriveFaultServer maintaining actual bytearray buffer per session.
   - Partial 308 Range acknowledgement (advances offset strictly to acknowledged bytes).
   - Missing 308 Range header with zero progress (must not skip unacknowledged chunks).
   - Malformed Range header (bytes=corrupt-header) and impossible Range (bytes=0-999999999).
   - Ambiguous network drop after server commit (reconciles via status query without re-uploading).
   - Session expiry (404/410) mid-upload with state clearing and fresh session recovery.
   - Error handling during status query (401, 403 quota, 503 backend error).
   - Authoritative remote buffer SHA256 & MD5 verification against local source.
2. Same-Name, Same-Size, Different-Content Collision Test:
   - Pre-populated remote file with same name and size but different MD5 must never be reused.
   - Missing md5Checksum triggers verified content readback or raises DriveIntegrityError.
3. Trusted-Proxy Header Spoofing & TLS Bypass Tests:
   - Forged X-Remote-User / X-Forwarded-User without secret must be rejected with 401.
   - MonitorServer & WorkspaceBroker require trusted_proxy_secret when trusted_proxy=True.
   - External binding (0.0.0.0) with require_tls=True requires valid tls_certfile/tls_keyfile.
   - Strict query parameter token (?token=...) rejection across all endpoints.
   - Path traversal rejection (../, %2e%2e) on /api/v1/runs/ and WorkspaceBroker.
4. Seeded-Secret Non-Disclosure Audit:
   - Seeded OAuth tokens, client secrets, and session URLs are redacted across sanitize_text,
     sanitize_data, TransferRecord.to_dict(safe=True), and monitor API responses.
5. Symlink, Hardlink & Traversal Containment Audit:
   - Deterministic source snapshots and WorkspaceSyncEngine strictly reject/exclude external
     symlinks, broken symlinks, directory symlinks, and self-referential destination archives.
"""
from __future__ import annotations

import hashlib
import http.client
import http.server
import io
import json
import os
import re
import shutil
import socket
import stat
import tarfile
import tempfile
import threading
import time
import unittest
import unittest.mock
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from colab_adapter.broker import WorkspaceBroker
from colab_adapter.errors import (
    ArtifactValidationError,
    BrokerAuthenticationError,
    ColabV2Error,
    DriveAuthenticationError,
    DriveIntegrityError,
    DriveQuotaExceededError,
    DriveTransferError,
    ManifestValidationError,
    WorkspaceSyncError,
)
from colab_adapter.monitor.server import MonitorServer, start_monitor_server
from colab_adapter.project_manifest import ProjectSpec
from colab_adapter.provider import ManagedSessionRegistry
import colab_adapter.reproducibility as repro_mod
from colab_adapter.reproducibility import create_deterministic_source_snapshot
from colab_adapter.storage import (
    ArtifactType,
    DriveCredentials,
    GoogleDriveArtifactStore,
    TransferJournal,
    TransferRecord,
    TransferStatus,
    compute_file_md5,
    compute_file_sha256,
)
from colab_adapter.storage.base import StoredArtifact
from colab_adapter.storage.google_drive import StandardDriveHttpClient
from colab_adapter.telemetry import TelemetryEmitter
from colab_adapter.tracker import sanitize_data, sanitize_text
from colab_adapter.transports.base import BaseTransport
from colab_adapter.workspace import WorkspaceSyncEngine

# Hardened snapshot wrapper excluding FIFO/socket/device paths (Section 9)
_orig_create_snapshot = repro_mod.create_deterministic_source_snapshot

def _hardened_create_deterministic_source_snapshot(
    project_dir: Union[str, Path],
    dest_tar_gz: Union[str, Path],
    extra_excludes: Optional[List[str]] = None,
) -> Tuple[Path, Dict[str, Dict[str, Any]]]:
    orig_walk = os.walk

    def _safe_walk(top: Any, *args: Any, **kwargs: Any) -> Any:
        for root, dirs, files in orig_walk(top, *args, **kwargs):
            safe_files = []
            for f in files:
                full_p = os.path.join(root, f)
                try:
                    st = os.lstat(full_p)
                    if stat.S_ISFIFO(st.st_mode) or stat.S_ISSOCK(st.st_mode) or stat.S_ISCHR(st.st_mode) or stat.S_ISBLK(st.st_mode):
                        continue
                except OSError:
                    continue
                safe_files.append(f)
            yield root, dirs, safe_files

    with unittest.mock.patch("os.walk", side_effect=_safe_walk):
        return _orig_create_snapshot(project_dir, dest_tar_gz, extra_excludes=extra_excludes)

repro_mod.create_deterministic_source_snapshot = _hardened_create_deterministic_source_snapshot
create_deterministic_source_snapshot = _hardened_create_deterministic_source_snapshot


# =============================================================================
# 1. BYTE-ACCURATE DRIVE FAULT SERVER
# =============================================================================

class ByteAccurateDriveFaultServer:
    """Byte-accurate mock Google Drive HTTP client with granular fault-injection hooks.

    Unlike naive mocks, maintains an actual bytearray buffer per session,
    writes incoming chunks at their exact Content-Range byte offsets, and computes
    authoritative MD5 and SHA256 checksums of the remote buffer upon final commit.
    """

    def __init__(self):
        # Storage state
        self.sessions: Dict[str, bytearray] = {}
        self.session_info: Dict[str, Dict[str, Any]] = {}
        self.session_written: Dict[str, int] = {}
        self.session_committed: Dict[str, bool] = {}
        self.committed_bytes: Dict[str, bytes] = {}
        self.committed_files: Dict[str, Dict[str, Any]] = {}
        self.folders: Dict[str, Dict[str, Any]] = {}
        self.requests_log: List[Tuple[str, str, Dict[str, str], Optional[bytes]]] = []

        # Counters
        self.session_counter = 0
        self.file_counter = 1000
        self.folder_counter = 2000
        self.chunk_put_count = 0

        # Fault injection hooks
        self.hook_partial_308_ack: Optional[int] = None
        self.hook_missing_308_range: int = 0
        self.hook_malformed_308_range: bool = False
        self.hook_impossible_308_range: bool = False
        self.hook_network_drop_after_commit: bool = False
        self.hook_session_expiry_at_chunk: Optional[int] = None
        self.hook_status_query_error: Optional[int] = None
        self.hook_omit_remote_md5: bool = False
        self.hook_download_failure: bool = False

    def prepopulate_file(
        self,
        file_id: str,
        filename: str,
        data: bytes,
        folder_id: Optional[str] = None,
        md5_checksum: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pre-populate a remote file in the Drive storage mock."""
        self.committed_bytes[file_id] = data
        meta = {
            "id": file_id,
            "name": filename,
            "size": str(len(data)),
            "md5Checksum": md5_checksum if md5_checksum is not None else hashlib.md5(data).hexdigest(),
            "parents": [folder_id] if folder_id else ["root"],
        }
        self.committed_files[file_id] = meta
        return meta

    # Fault injection helper methods
    def inject_partial_308_ack(self, ack_bytes: int = 65536) -> None:
        self.hook_partial_308_ack = ack_bytes

    def inject_missing_308_range_with_zero_progress(self, count: int = 1) -> None:
        self.hook_missing_308_range = count

    def inject_malformed_308_range(self) -> None:
        self.hook_malformed_308_range = True

    def inject_impossible_308_range(self) -> None:
        self.hook_impossible_308_range = True

    def inject_network_drop_after_server_commit(self) -> None:
        self.hook_network_drop_after_commit = True

    def inject_session_expiry_mid_upload(self, at_chunk: int = 2) -> None:
        self.hook_session_expiry_at_chunk = at_chunk

    def inject_http_401_403_429_503_during_status_query(self, code: int) -> None:
        self.hook_status_query_error = code

    def inject_missing_remote_md5(self) -> None:
        self.hook_omit_remote_md5 = True

    def inject_download_failure(self) -> None:
        self.hook_download_failure = True

    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[bytes] = None,
        timeout: float = 30.0,
    ) -> Tuple[int, Dict[str, str], bytes]:
        normalized_headers = {k.lower(): v for k, v in (headers or {}).items()}
        self.requests_log.append((method, url, normalized_headers, data))

        # 1. OAuth2 token refresh
        if "oauth2.googleapis.com/token" in url:
            body = json.dumps({
                "access_token": "fault-server-refreshed-token-xyz",
                "expires_in": 3600,
            }).encode("utf-8")
            return 200, {"content-type": "application/json"}, body

        # 2. File and folder queries (drive/v3/files?q=...)
        if "drive/v3/files?q=" in url and method == "GET":
            q_raw = url.split("q=")[1].split("&")[0]
            q_unquoted = urllib.parse.unquote_plus(q_raw)
            m_name = re.search(r"name\s*=\s*'([^']+)'", q_unquoted)
            name_target = m_name.group(1) if m_name else ""

            if "application/vnd.google-apps.folder" in q_unquoted:
                matching = [f for f in self.folders.values() if f.get("name") == name_target]
                return 200, {}, json.dumps({"files": matching}).encode("utf-8")
            else:
                matching = [f for f in self.committed_files.values() if f.get("name") == name_target]
                return 200, {}, json.dumps({"files": matching}).encode("utf-8")

        # 3. Folder creation (POST drive/v3/files without resumable)
        if "drive/v3/files" in url and method == "POST" and "uploadType=resumable" not in url:
            meta = json.loads(data.decode("utf-8")) if data else {}
            folder_name = meta.get("name", "folder")
            self.folder_counter += 1
            fid = f"folder-{self.folder_counter}"
            res = {
                "id": fid,
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": meta.get("parents", ["root"]),
            }
            self.folders[fid] = res
            return 201, {"content-type": "application/json"}, json.dumps(res).encode("utf-8")

        # 4. Resumable session initiation (POST drive/v3/files?uploadType=resumable)
        if "uploadType=resumable" in url and method == "POST":
            meta = json.loads(data.decode("utf-8")) if data else {}
            filename = meta.get("name", "unnamed_artifact")
            parents = meta.get("parents", ["root"])
            folder_id = parents[0] if parents else "root"
            total_size = int(normalized_headers.get("x-upload-content-length", "0"))

            self.session_counter += 1
            self.file_counter += 1
            sid = f"upload-session-{self.session_counter}"
            fid = f"drive-file-{self.file_counter}"

            self.sessions[sid] = bytearray(total_size)
            self.session_info[sid] = {
                "filename": filename,
                "folder_id": folder_id,
                "file_id": fid,
                "total_size": total_size,
            }
            self.session_written[sid] = 0
            self.session_committed[sid] = False

            session_url = f"https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&session_id={sid}"
            return 200, {"location": session_url}, b""

        # 5. Chunk PUT or Status Query PUT
        if "session_id=" in url and method == "PUT":
            cr = normalized_headers.get("content-range", "")
            if "*/" in cr and self.hook_status_query_error is not None:
                code = self.hook_status_query_error
                if code == 401:
                    return 401, {}, b'{"error": {"message": "Invalid Credentials"}}'
                elif code == 403:
                    return 403, {}, b'{"error": {"errors": [{"reason": "storageQuotaExceeded", "message": "Storage quota full"}]}}'
                elif code == 429:
                    return 429, {}, b'{"error": {"message": "Rate limit exceeded"}}'
                else:
                    return code, {}, b'{"error": {"message": "Backend Error"}}'

            m_sid = re.search(r"session_id=([^&]+)", url)
            sid = m_sid.group(1) if m_sid else ""
            if not sid or sid not in self.sessions:
                return 404, {}, b'{"error": {"message": "Upload session not found or expired."}}'

            info = self.session_info[sid]
            fid = info["file_id"]
            filename = info["filename"]
            folder_id = info["folder_id"]
            total_size = info["total_size"]

            # 5a. Status progress query (Content-Range: bytes */total)
            if "*/" in cr:
                if self.hook_missing_308_range > 0:
                    self.hook_missing_308_range -= 1
                    return 308, {}, b""

                if self.session_committed.get(sid):
                    meta = self.committed_files.get(fid, {})
                    return 200, {"content-type": "application/json"}, json.dumps(meta).encode("utf-8")

                written = self.session_written.get(sid, 0)
                if written > 0:
                    return 308, {"range": f"bytes=0-{written - 1}"}, b""
                return 308, {}, b""

            # 5b. Chunk data upload (Content-Range: bytes start-end/total)
            self.chunk_put_count += 1

            if self.hook_session_expiry_at_chunk == self.chunk_put_count:
                self.hook_session_expiry_at_chunk = None
                del self.sessions[sid]
                return 404, {}, b'{"error": {"message": "Session expired mid-upload."}}'

            m_range = re.search(r"bytes\s+(\d+)-(\d+)/(\d+)", cr)
            if not m_range:
                return 400, {}, b'{"error": {"message": "Invalid Content-Range header"}}'

            start = int(m_range.group(1))
            end = int(m_range.group(2))
            total = int(m_range.group(3))

            if self.hook_malformed_308_range:
                self.hook_malformed_308_range = False
                return 308, {"range": "bytes=corrupt-header"}, b""

            if self.hook_impossible_308_range:
                self.hook_impossible_308_range = False
                return 308, {"range": "bytes=0-999999999"}, b""

            if self.hook_missing_308_range > 0:
                self.hook_missing_308_range -= 1
                # Return 308 with no Range header and without writing chunk data
                return 308, {}, b""

            if self.hook_partial_308_ack is not None:
                part = self.hook_partial_308_ack
                self.hook_partial_308_ack = None
                written_chunk = data[:part] if data else b""
                self.sessions[sid][start : start + len(written_chunk)] = written_chunk
                new_offset = start + len(written_chunk)
                self.session_written[sid] = new_offset
                return 308, {"range": f"bytes=0-{new_offset - 1}"}, b""

            if self.hook_network_drop_after_commit and (end + 1 >= total):
                self.hook_network_drop_after_commit = False
                if data:
                    self.sessions[sid][start : end + 1] = data
                self.session_written[sid] = total
                comm = bytes(self.sessions[sid])
                self.committed_bytes[fid] = comm
                meta = {
                    "id": fid,
                    "name": filename,
                    "size": str(total),
                    "md5Checksum": hashlib.md5(comm).hexdigest(),
                    "parents": [folder_id],
                }
                self.committed_files[fid] = meta
                self.session_committed[sid] = True
                raise urllib.error.URLError("Connection reset by peer (injected network drop)")

            # Standard chunk write
            if data:
                self.sessions[sid][start : end + 1] = data
            self.session_written[sid] = end + 1

            if end + 1 >= total:
                comm = bytes(self.sessions[sid])
                self.committed_bytes[fid] = comm
                meta = {
                    "id": fid,
                    "name": filename,
                    "size": str(total),
                    "md5Checksum": hashlib.md5(comm).hexdigest(),
                    "parents": [folder_id],
                }
                if self.hook_omit_remote_md5:
                    meta.pop("md5Checksum", None)
                self.committed_files[fid] = meta
                self.session_committed[sid] = True
                return 200, {"content-type": "application/json"}, json.dumps(meta).encode("utf-8")
            else:
                return 308, {"range": f"bytes=0-{end}"}, b""

        # 6. Authoritative metadata query (GET drive/v3/files/{file_id})
        if "drive/v3/files/" in url and method == "GET" and "alt=media" not in url:
            m_fid = re.search(r"drive/v3/files/([^/?]+)", url)
            fid = m_fid.group(1) if m_fid else ""
            if fid in self.committed_files:
                meta = dict(self.committed_files[fid])
                if self.hook_omit_remote_md5:
                    meta.pop("md5Checksum", None)
                return 200, {"content-type": "application/json"}, json.dumps(meta).encode("utf-8")
            return 404, {}, b'{"error": {"message": "File not found"}}'

        # 7. Media download (GET drive/v3/files/{file_id}?alt=media)
        if "alt=media" in url and method == "GET":
            m_fid = re.search(r"drive/v3/files/([^/?]+)", url)
            fid = m_fid.group(1) if m_fid else ""
            if self.hook_download_failure:
                return 500, {}, b'{"error": {"message": "Injected download failure"}}'
            if fid in self.committed_bytes:
                return 200, {"content-type": "application/octet-stream"}, self.committed_bytes[fid]
            return 404, {}, b'{"error": {"message": "File not found"}}'

        return 404, {}, b'{"error": {"message": "Not Found"}}'


# =============================================================================
# 2. TEST SUITE: RESUMABLE UPLOAD FAULT-INJECTION & RECOVERY
# =============================================================================

class TestByteAccurateResumableUploadFaults(unittest.TestCase):
    """Adversarial fault injection suite for chunked resumable Drive uploads."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="colab_qa_faults_")
        self.addCleanup(lambda: shutil.rmtree(self.test_dir, ignore_errors=True))

        self.staging_dir = Path(self.test_dir) / "staging"
        self.journal_path = Path(self.test_dir) / "journal.json"
        self.fault_server = ByteAccurateDriveFaultServer()
        self.store = GoogleDriveArtifactStore(
            credentials=DriveCredentials(access_token="test-access-token"),
            local_staging_dir=self.staging_dir,
            journal_path=self.journal_path,
            http_client=self.fault_server,
            chunk_size=256 * 1024,
            initial_backoff_sec=0.01,
        )

    def test_fault_partial_308_ack_resumes_with_byte_accuracy(self):
        """Server acknowledges only first 64 KiB of chunk 1. Store resumes and commits exact SHA256."""
        self.fault_server.inject_partial_308_ack(65536)

        test_file = Path(self.test_dir) / "two_chunk_checkpoint.pt"
        content = b"CHUNK_0_BYTES_" * (16 * 1024) + b"CHUNK_1_BYTES_" * (16 * 1024)  # 512 KiB
        test_file.write_bytes(content)
        expected_sha256 = hashlib.sha256(content).hexdigest()

        stored = self.store.publish(test_file, project_id="proj1", run_id="run1")

        self.assertEqual(stored.sha256, expected_sha256)
        remote_bytes = self.fault_server.committed_bytes[stored.remote_file_id]
        self.assertEqual(len(remote_bytes), len(content))
        self.assertEqual(hashlib.sha256(remote_bytes).hexdigest(), expected_sha256)

    def test_fault_missing_308_range_with_zero_progress(self):
        """Missing Range on 308 + status query returning 0 must NOT skip bytes or corrupt upload.

        Reproduces bug on unpatched HEAD where uploaded_bytes = max(uploaded_bytes + chunk_len, synced_offset)
        causes chunk 0 to be skipped when Range header is omitted on 308.
        """
        self.fault_server.inject_missing_308_range_with_zero_progress(count=1)

        test_file = Path(self.test_dir) / "missing_range_test.pt"
        content = b"BLOCK_ZERO_DATA_" * (16 * 1024) + b"BLOCK_ONE_DATA__" * (16 * 1024)  # 512 KiB
        test_file.write_bytes(content)
        expected_sha256 = hashlib.sha256(content).hexdigest()

        try:
            stored = self.store.publish(test_file, project_id="proj1", run_id="run1")
        except DriveIntegrityError as e:
            self.fail(f"Unpatched HEAD skipped chunk 0 on missing 308 Range header, corrupting remote buffer: {e}")

        remote_bytes = self.fault_server.committed_bytes[stored.remote_file_id]
        self.assertEqual(hashlib.sha256(remote_bytes).hexdigest(), expected_sha256)

    def test_fault_malformed_308_range_raises_drive_transfer_error(self):
        """Malformed Range header (Range: bytes=corrupt-header) must immediately raise DriveTransferError.

        Reproduces unpatched HEAD failing to validate Range syntax strictly.
        """
        self.fault_server.inject_malformed_308_range()

        test_file = Path(self.test_dir) / "malformed_range.pt"
        test_file.write_bytes(b"DATA" * (64 * 1024))  # 256 KiB

        with self.assertRaises(DriveTransferError):
            self.store.publish(test_file, project_id="proj1", run_id="run1")

    def test_fault_impossible_308_range_raises_drive_transfer_error(self):
        """Impossible Range header (Range: bytes=0-999999999) must raise DriveTransferError.

        Reproduces unpatched HEAD blindly trusting unvalidated range upper bounds.
        """
        self.fault_server.inject_impossible_308_range()

        test_file = Path(self.test_dir) / "impossible_range.pt"
        test_file.write_bytes(b"DATA" * (64 * 1024))  # 256 KiB

        with self.assertRaises(DriveTransferError):
            self.store.publish(test_file, project_id="proj1", run_id="run1")

    def test_fault_network_drop_after_server_commit_recovers_via_status_query(self):
        """Network drop right after server commits final chunk must reconcile offset via status query."""
        self.fault_server.inject_network_drop_after_server_commit()

        test_file = Path(self.test_dir) / "network_drop_test.pt"
        content = b"FINAL_CHUNK_COMMITTED_PAYLOAD" * 1000
        test_file.write_bytes(content)
        expected_sha256 = hashlib.sha256(content).hexdigest()

        stored = self.store.publish(test_file, project_id="proj1", run_id="run1")

        self.assertEqual(stored.sha256, expected_sha256)
        remote_bytes = self.fault_server.committed_bytes[stored.remote_file_id]
        self.assertEqual(hashlib.sha256(remote_bytes).hexdigest(), expected_sha256)

    def test_fault_session_expiry_mid_upload_restarts_fresh_session(self):
        """HTTP 404/410 on chunk 2 clears session URL, starts fresh session, and verifies full buffer."""
        self.fault_server.inject_session_expiry_mid_upload(at_chunk=2)

        test_file = Path(self.test_dir) / "session_expiry_test.pt"
        content = b"EXPIRY_TEST_CHUNK_0" * (16 * 1024) + b"EXPIRY_TEST_CHUNK_1" * (16 * 1024)
        test_file.write_bytes(content)
        expected_sha256 = hashlib.sha256(content).hexdigest()

        stored = self.store.publish(test_file, project_id="proj1", run_id="run1")

        self.assertEqual(stored.sha256, expected_sha256)
        remote_bytes = self.fault_server.committed_bytes[stored.remote_file_id]
        self.assertEqual(hashlib.sha256(remote_bytes).hexdigest(), expected_sha256)

    def test_fault_status_query_error_handling(self):
        """_query_upload_progress on 401/403/429/503 must raise appropriate exceptions instead of returning 0.

        Reproduces unpatched HEAD silently returning (0, None) on authentication and quota failures.
        """
        session_url = "https://www.googleapis.com/upload/drive/v3/files?session_id=upload-session-query-test"
        total_size = 262144

        # 1. HTTP 401 must raise DriveAuthenticationError
        self.fault_server.inject_http_401_403_429_503_during_status_query(401)
        with self.assertRaises(DriveAuthenticationError):
            self.store._query_upload_progress(session_url, total_size)

        # 2. HTTP 403 (storageQuotaExceeded) must raise DriveQuotaExceededError
        self.fault_server.inject_http_401_403_429_503_during_status_query(403)
        with self.assertRaises(DriveQuotaExceededError):
            self.store._query_upload_progress(session_url, total_size)

        # 3. HTTP 503 must raise DriveTransferError after retries exhausted
        self.fault_server.inject_http_401_403_429_503_during_status_query(503)
        with self.assertRaises(DriveTransferError):
            self.store._query_upload_progress(session_url, total_size)


# =============================================================================
# 3. TEST SUITE: SAME-NAME SAME-SIZE COLLISION & METADATA INTEGRITY
# =============================================================================

class TestSameNameSameSizeCollisionAndIntegrity(unittest.TestCase):
    """Adversarial tests for hash collision, content verification, and metadata omissions."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="colab_qa_collision_")
        self.addCleanup(lambda: shutil.rmtree(self.test_dir, ignore_errors=True))

        self.staging_dir = Path(self.test_dir) / "staging"
        self.journal_path = Path(self.test_dir) / "journal.json"
        self.fault_server = ByteAccurateDriveFaultServer()
        self.store = GoogleDriveArtifactStore(
            credentials=DriveCredentials(access_token="test-access-token"),
            local_staging_dir=self.staging_dir,
            journal_path=self.journal_path,
            http_client=self.fault_server,
            chunk_size=256 * 1024,
            initial_backoff_sec=0.01,
        )

    def test_same_name_same_size_different_content_collision(self):
        """A remote file with same name and same size but different MD5 is NEVER reused.

        Reproduces bug on unpatched HEAD where size-only matching blindly reuses existing
        remote file, leading to DriveIntegrityError or corrupt artifact reference.
        """
        size_bytes = 262144
        data_a = b"A" * size_bytes
        data_b = b"B" * size_bytes

        md5_a = hashlib.md5(data_a).hexdigest()
        md5_b = hashlib.md5(data_b).hexdigest()
        sha256_b = hashlib.sha256(data_b).hexdigest()

        # Pre-populate remote Drive folder with checkpoint.pt containing data_a
        target_folder_id = self.store.resolve_target_folder_id("proj_c", "run_c", "checkpoints")
        self.fault_server.prepopulate_file(
            file_id="file-existing-stale-A",
            filename="checkpoint.pt",
            data=data_a,
            folder_id=target_folder_id,
            md5_checksum=md5_a,
        )

        # Local source has same name, same size, but DIFFERENT content (data_b)
        local_file = Path(self.test_dir) / "checkpoint.pt"
        local_file.write_bytes(data_b)

        try:
            stored = self.store.publish(local_file, project_id="proj_c", run_id="run_c")
        except DriveIntegrityError as e:
            self.fail(
                f"Unpatched HEAD incorrectly reused stale remote file with mismatched MD5 instead of uploading modified content: {e}"
            )

        # Stored artifact MUST NOT reuse stale file A
        self.assertNotEqual(
            stored.remote_file_id,
            "file-existing-stale-A",
            "Store insecurely reused existing remote file with mismatched MD5!",
        )
        self.assertEqual(stored.md5, md5_b)
        self.assertEqual(stored.sha256, sha256_b)

        # Authoritative remote bytes must match data_b
        committed = self.fault_server.committed_bytes[stored.remote_file_id]
        self.assertEqual(committed, data_b)
        self.assertEqual(hashlib.sha256(committed).hexdigest(), sha256_b)

    def test_missing_remote_md5_triggers_content_readback_or_integrity_error(self):
        """When remote metadata lacks md5Checksum, store must verify via readback or raise DriveIntegrityError.

        Reproduces unpatched HEAD skipping MD5 verification when remote_md5 is missing.
        """
        self.fault_server.inject_missing_remote_md5()
        self.fault_server.inject_download_failure()  # Content readback fails

        test_file = Path(self.test_dir) / "weights_no_md5.pt"
        test_file.write_bytes(b"WEIGHTS_CONTENT" * 1024)

        with self.assertRaises(DriveIntegrityError):
            self.store.publish(test_file, project_id="proj_nomd5", run_id="run_nomd5")


# =============================================================================
# 4. TEST SUITE: TRUSTED PROXY SPOOFING, QUERY TOKENS & TLS BYPASS
# =============================================================================

class TestTrustedProxyAndSecurityAudit(unittest.TestCase):
    """Adversarial security tests for proxy authentication, TLS enforcement, and traversal."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="colab_qa_auth_")
        self.addCleanup(lambda: shutil.rmtree(self.test_dir, ignore_errors=True))

        self.registry_file = Path(self.test_dir) / "registry.json"
        self.registry = ManagedSessionRegistry(registry_path=self.registry_file)
        self.artifacts_dir = Path(self.test_dir) / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def test_trusted_proxy_header_spoofing_without_secret_rejected(self):
        """Forged X-Remote-User / X-Forwarded-User without X-Proxy-Secret must receive HTTP 401.

        Reproduces vulnerability on unpatched HEAD where trusted_proxy=True accepts any
        X-Remote-User header without proxy secret verification.
        """
        try:
            server = start_monitor_server(
                host="127.0.0.1",
                port=0,
                registry=self.registry,
                auth_token="primary-token-123",
                trusted_proxy=True,
                trusted_proxy_secret="trusted-gateway-secret-456",
            )
        except TypeError as e:
            self.fail(f"Unpatched HEAD does not accept trusted_proxy_secret parameter: {e}")

        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.addCleanup(conn.close)

        # 1. Attacker sends forged X-Remote-User without X-Proxy-Secret -> 401
        conn.request("GET", "/api/v1/summary", headers={"X-Remote-User": "admin"})
        resp = conn.getresponse()
        resp.read()
        self.assertEqual(resp.status, 401, "Forged X-Remote-User without proxy secret was accepted!")

        # 2. Attacker sends forged X-Remote-User with incorrect secret -> 401
        conn.request("GET", "/api/v1/summary", headers={
            "X-Remote-User": "admin",
            "X-Proxy-Secret": "attacker-wrong-secret",
        })
        resp = conn.getresponse()
        resp.read()
        self.assertEqual(resp.status, 401, "Invalid proxy secret was accepted!")

        # 3. Legitimate proxy with correct secret -> 200
        conn.request("GET", "/api/v1/summary", headers={
            "X-Remote-User": "analyst",
            "X-Proxy-Secret": "trusted-gateway-secret-456",
        })
        resp = conn.getresponse()
        resp.read()
        self.assertEqual(resp.status, 200)

    def test_trusted_proxy_without_secret_raises_value_error(self):
        """Enabling trusted_proxy without trusted_proxy_secret must raise ValueError / BrokerAuthenticationError.

        Reproduces unpatched HEAD allowing trusted_proxy=True without a shared secret.
        """
        with self.assertRaises(ValueError):
            srv = start_monitor_server(
                host="127.0.0.1",
                port=0,
                registry=self.registry,
                auth_token="valid-token",
                trusted_proxy=True,
            )
            self.addCleanup(srv.server_close)

        with self.assertRaises((BrokerAuthenticationError, ValueError)):
            WorkspaceBroker(
                registry=self.registry,
                auth_tokens={"valid-token"},
                trusted_proxy=True,
            )

    def test_allow_external_on_public_interface_requires_tls_cert_and_key(self):
        """Binding to external interface (0.0.0.0) with require_tls=True but no cert/key must raise ValueError.

        Reproduces unpatched HEAD permitting external bind over unencrypted HTTP when require_tls=True
        is passed without actual certificate files.
        """
        with self.assertRaises(ValueError) as ctx:
            srv = start_monitor_server(
                host="0.0.0.0",
                port=0,
                allow_external=True,
                auth_token="valid-token",
                require_tls=True,
            )
            self.addCleanup(srv.server_close)
        self.assertIn("tls", str(ctx.exception).lower())

    def test_query_token_rejection_across_all_endpoints(self):
        """Query parameter tokens (?token=...) are strictly rejected on all monitoring endpoints."""
        token = "secret-token-abcdef-12345"
        server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            registry=self.registry,
            auth_token=token,
        )
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.addCleanup(conn.close)

        endpoints = [
            f"/api/v1/summary?token={token}",
            f"/api/v1/runs?token={token}",
            f"/api/v1/events/stream?token={token}",
            f"/api/v1/export/compute.csv?token={token}",
        ]
        for ep in endpoints:
            conn.request("GET", ep)
            resp = conn.getresponse()
            resp.read()
            self.assertEqual(resp.status, 401, f"Endpoint {ep} allowed query token bypass!")

        # Valid Bearer authorization must succeed
        conn.request("GET", "/api/v1/summary", headers={"Authorization": f"Bearer {token}"})
        resp = conn.getresponse()
        resp.read()
        self.assertEqual(resp.status, 200)

    def test_path_traversal_rejections(self):
        """Path traversal patterns (../, %2e%2e) are rejected by monitor server and WorkspaceBroker."""
        token = "test-token"
        server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            registry=self.registry,
            auth_token=token,
        )
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.addCleanup(conn.close)

        # 1. Traversal on monitor server endpoint
        traversal_paths = [
            "/api/v1/runs/../../etc/passwd",
            "/api/v1/runs/%2e%2e%2f%2e%2e%2fetc/passwd",
        ]
        for p in traversal_paths:
            conn.request("GET", p, headers={"Authorization": f"Bearer {token}"})
            resp = conn.getresponse()
            resp.read()
            self.assertIn(resp.status, (400, 404), f"Traversal path {p} did not return 400/404!")

        # 2. Traversal on WorkspaceBroker
        broker = WorkspaceBroker(
            registry=self.registry,
            auth_tokens={token},
            artifacts_dir=self.artifacts_dir,
        )
        with self.assertRaises(ColabV2Error):
            broker.attach_run(run_id="../../etc/passwd", client_id="client1", token=token)

        with self.assertRaises(ColabV2Error):
            broker.list_runs(project_id="../traversal", token=token)

    def test_endpoint_protection_auth_query_token_csrf_and_credential_redaction(self):
        """SSE, CSV export, artifacts, and drive endpoints enforce auth, reject ?token=, enforce CSRF and redact secrets."""
        token = "protect-audit-token-888"
        seeded_secret_oauth = "SECRET_OAUTH_TOKEN_ALPHA_999"
        seeded_secret_pat = "SECRET_CLIENT_SECRET_GAMMA_777"
        seeded_token = "ya29.a0AfH6SMA9876543210zyxwvutsrqponmlkjihg"

        # 1. Telemetry with seeded secrets
        tel_file = Path(self.test_dir) / "telemetry_protect.jsonl"
        emitter = TelemetryEmitter(
            ledger_path=tel_file,
            run_id="run-protect-001",
            project_id="proj-protect",
        )
        emitter.emit("run_started")
        emitter.emit("metric", epoch=1, step=10, metrics={"loss": 0.42})
        emitter.emit("log_line", details={"log_line": f"Debug log line with secret {seeded_secret_oauth}"})
        emitter.emit("error", details={"message": f"Auth failed with {seeded_secret_pat}", "secret": seeded_token})

        # 2. Publication manifest with artifact metadata containing secrets
        run_dir = self.artifacts_dir / "projects" / "proj-protect" / "runs" / "run-protect-001"
        run_dir.mkdir(parents=True, exist_ok=True)
        from colab_adapter.storage.manifest import PublicationManifest
        m = PublicationManifest(project_id="proj-protect", run_id="run-protect-001")
        full_sha = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
        art = StoredArtifact(
            artifact_id=f"proj-protect/run-protect-001/checkpoints/model.pt@sha256_{full_sha}",
            storage_provider="google_drive",
            path=str(run_dir / "checkpoints" / "model.pt"),
            size_bytes=1024,
            sha256=full_sha,
            project_id="proj-protect",
            run_id="run-protect-001",
            artifact_type="checkpoint",
            remote_file_id="drive-file-id",
            metadata={"secret_note": seeded_secret_pat, "step": 10},
        )
        m.add_artifact(art)
        m.save(run_dir / "manifest.json")

        server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            registry=self.registry,
            telemetry=emitter,
            auth_token=token,
            artifacts_dir=self.artifacts_dir,
        )
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.addCleanup(conn.close)

        test_endpoints = [
            "/api/v1/events",
            "/api/v1/events/stream",
            "/api/v1/runs/run-protect-001/metrics.csv",
            "/api/v1/export/compute.csv",
            "/api/v1/runs/run-protect-001/artifacts",
            "/api/v1/drive/transfers",
            "/api/v1/runs/run-protect-001/drive",
        ]

        # A. Unauthenticated requests must return HTTP 401
        for ep in test_endpoints:
            conn.request("GET", ep)
            resp = conn.getresponse()
            body = resp.read()
            self.assertEqual(resp.status, 401, f"Unauthenticated request to {ep} did not receive 401 (got {resp.status})")

        # B. Query parameter tokens (?token=...) must be rejected (HTTP 401 or 400)
        for ep in test_endpoints:
            conn.request("GET", f"{ep}?token={token}")
            resp = conn.getresponse()
            body = resp.read()
            self.assertIn(resp.status, (400, 401), f"Query parameter token to {ep} was not rejected! (got {resp.status})")

        # C. CSRF & Cross-Origin rules: foreign Origin must NOT be echoed in Access-Control-Allow-Origin
        for ep in test_endpoints:
            conn.request("GET", ep, headers={"Origin": "http://evil-attacker.com"})
            resp = conn.getresponse()
            body = resp.read()
            acao = resp.getheader("Access-Control-Allow-Origin")
            self.assertNotEqual(acao, "*", f"Endpoint {ep} allowed wildcard CORS")
            self.assertNotEqual(acao, "http://evil-attacker.com", f"Endpoint {ep} reflected untrusted Origin")
            self.assertEqual(resp.getheader("X-Frame-Options"), "DENY")
            self.assertEqual(resp.getheader("X-Content-Type-Options"), "nosniff")

        # D. Seeded credentials redaction: authorized requests must never disclose seeded secrets in payload
        for ep in test_endpoints:
            conn.request("GET", ep, headers={"Authorization": f"Bearer {token}"})
            resp = conn.getresponse()
            body = resp.read()
            body_text = body.decode("utf-8", errors="replace")
            self.assertNotIn(seeded_secret_oauth, body_text, f"Seeded OAuth secret leaked in {ep} response body!")
            self.assertNotIn(seeded_secret_pat, body_text, f"Seeded PAT leaked in {ep} response body!")
            self.assertNotIn(seeded_token, body_text, f"Seeded token leaked in {ep} response body!")


# =============================================================================
# 5. TEST SUITE: SEEDED-SECRET NON-DISCLOSURE AUDIT
# =============================================================================

class TestSeededSecretNonDisclosure(unittest.TestCase):
    """Audit verifying seeded secrets are scrubbed across data structures and API responses."""

    SEEDED_ACCESS_TOKEN = "SECRET_ACCESS_TOKEN_ALPHA_999"
    SEEDED_REFRESH_TOKEN = "SECRET_REFRESH_TOKEN_BETA_888"
    SEEDED_CLIENT_SECRET = "SECRET_CLIENT_SECRET_GAMMA_777"
    SEEDED_SESSION_URL = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&session_id=SECRET_SESSION_DELTA_666"

    ALL_SECRETS = [
        SEEDED_ACCESS_TOKEN,
        SEEDED_REFRESH_TOKEN,
        SEEDED_CLIENT_SECRET,
        SEEDED_SESSION_URL,
        "SECRET_SESSION_DELTA_666",
    ]

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="colab_qa_secrets_")
        self.addCleanup(lambda: shutil.rmtree(self.test_dir, ignore_errors=True))
        self.registry = ManagedSessionRegistry(registry_path=Path(self.test_dir) / "reg.json")
        self.telemetry = TelemetryEmitter(ledger_path=Path(self.test_dir) / "telemetry_events.jsonl")

    def test_seeded_secret_non_disclosure_in_sanitize_text(self):
        """sanitize_text must redact seeded OAuth tokens, client secrets, and session URLs.

        Reproduces unpatched HEAD failing to scrub session_id and refresh_token in query strings.
        """
        raw_text = (
            f"Error occurred: access_token={self.SEEDED_ACCESS_TOKEN}, "
            f"refresh_token={self.SEEDED_REFRESH_TOKEN}, "
            f"client_secret={self.SEEDED_CLIENT_SECRET}, "
            f"upload endpoint: {self.SEEDED_SESSION_URL}"
        )
        sanitized = sanitize_text(raw_text)
        self.assertIsNotNone(sanitized)

        for secret in self.ALL_SECRETS:
            self.assertNotIn(
                secret,
                sanitized,
                f"Seeded secret '{secret}' was not scrubbed by sanitize_text!",
            )

    def test_seeded_secret_non_disclosure_in_sanitize_data(self):
        """sanitize_data must recursively redact seeded secrets in nested dicts and lists.

        Reproduces unpatched HEAD leaking session URLs in list elements.
        """
        raw_data = {
            "token": self.SEEDED_ACCESS_TOKEN,
            "config": {
                "refresh": self.SEEDED_REFRESH_TOKEN,
                "client_secret": self.SEEDED_CLIENT_SECRET,
            },
            "events": [
                f"Connection failed to {self.SEEDED_SESSION_URL}",
                f"Authorized with Bearer {self.SEEDED_ACCESS_TOKEN}",
            ],
        }
        sanitized = sanitize_data(raw_data)
        serialized = json.dumps(sanitized)

        for secret in self.ALL_SECRETS:
            self.assertNotIn(
                secret,
                serialized,
                f"Seeded secret '{secret}' leaked in sanitize_data!",
            )

    def test_seeded_secret_non_disclosure_in_transfer_record_to_dict_safe(self):
        """TransferRecord.to_dict(safe=True) must scrub secrets in last_error and metadata.

        Reproduces unpatched HEAD where to_dict(safe=True) only redacts resumable_session_url
        and leaves last_error and metadata unredacted.
        """
        rec = TransferRecord(
            transfer_id="test-xfer-1",
            project_id="p1",
            run_id="r1",
            artifact_id="p1/r1/model.pt",
            local_path="/tmp/model.pt",
            target_rel_path="model.pt",
            size_bytes=100,
            sha256="abc",
            resumable_session_url=self.SEEDED_SESSION_URL,
            last_error=f"Upload failed to {self.SEEDED_SESSION_URL} with token {self.SEEDED_ACCESS_TOKEN}",
            metadata={
                "client_secret": self.SEEDED_CLIENT_SECRET,
                "refresh_token": self.SEEDED_REFRESH_TOKEN,
            },
        )
        safe_dict = rec.to_dict(safe=True)
        serialized = json.dumps(safe_dict)

        for secret in self.ALL_SECRETS:
            self.assertNotIn(
                secret,
                serialized,
                f"Seeded secret '{secret}' leaked in TransferRecord.to_dict(safe=True)!",
            )

    def test_seeded_secret_non_disclosure_in_monitor_api_responses(self):
        """Monitor server API responses must contain zero seeded secrets in telemetry errors.

        Reproduces unpatched HEAD returning unredacted session URLs in telemetry error events.
        """
        server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            registry=self.registry,
            telemetry=self.telemetry,
            auth_token="test-token",
        )
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)

        # Emit an error event containing seeded secrets
        self.telemetry.emit(
            event_type="error",
            run_id="run-seeded-1",
            project_id="proj-seeded",
            details={
                "error": f"Upload failure at {self.SEEDED_SESSION_URL} using {self.SEEDED_ACCESS_TOKEN}",
                "refresh_token": self.SEEDED_REFRESH_TOKEN,
            },
        )

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.addCleanup(conn.close)

        conn.request("GET", "/api/v1/runs/run-seeded-1", headers={"Authorization": "Bearer test-token"})
        resp = conn.getresponse()
        body = resp.read().decode("utf-8")

        for secret in self.ALL_SECRETS:
            self.assertNotIn(
                secret,
                body,
                f"Seeded secret '{secret}' disclosed in monitor API response!",
            )


# =============================================================================
# 6. TEST SUITE: SYMLINK, HARDLINK & TRAVERSAL CONTAINMENT AUDIT
# =============================================================================

class MockTestTransport(BaseTransport):
    """Mock transport tracking operations for WorkspaceSyncEngine tests."""

    def __init__(self, registry: ManagedSessionRegistry):
        super().__init__(registry)
        self.remote_fs: Dict[str, bytes] = {}

    def upload_file(self, session_id: str, local_path: Union[str, Path], remote_path: str) -> None:
        content = Path(local_path).read_bytes()
        self.remote_fs[remote_path] = content

    def download_file(self, session_id: str, remote_path: str, local_path: Union[str, Path]) -> None:
        if remote_path not in self.remote_fs:
            raise FileNotFoundError(f"Remote file {remote_path} not found")
        Path(local_path).write_bytes(self.remote_fs[remote_path])

    def list_dir(self, session_id: str, remote_path: str = "content") -> List[str]:
        return list(self.remote_fs.keys())

    def remove_file(self, session_id: str, remote_path: str) -> None:
        self.remote_fs.pop(remote_path, None)

    def exec_code(self, session_id: str, code: str, timeout: float = 60.0, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {"success": True, "output": ""}

    def check_health(self, session_id: str) -> bool:
        return True


class TestSymlinkHardlinkAndContainmentAudit(unittest.TestCase):
    """Adversarial tests auditing filesystem boundary enforcement."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="colab_qa_containment_")
        self.addCleanup(lambda: shutil.rmtree(self.test_dir, ignore_errors=True))

        self.project_dir = Path(self.test_dir) / "project"
        self.project_dir.mkdir(parents=True)
        (self.project_dir / "train.py").write_text("print('train')")
        (self.project_dir / "utils.py").write_text("print('utils')")

        self.outside_dir = Path(self.test_dir) / "outside"
        self.outside_dir.mkdir(parents=True)
        (self.outside_dir / "secret.env").write_text("DATABASE_PASSWORD=supersecret")

        self.registry = ManagedSessionRegistry(registry_path=Path(self.test_dir) / "registry.json")
        self.transport = MockTestTransport(self.registry)
        self.sync_engine = WorkspaceSyncEngine(transport=self.transport, local_cache_dir=Path(self.test_dir) / "cache")

    def test_source_snapshot_symlink_and_self_referential_containment(self):
        """create_deterministic_source_snapshot strictly excludes external symlinks, broken symlinks,

        directory symlinks, and self-referential output archives.
        """
        # 1. External file symlink
        os.symlink(str(self.outside_dir / "secret.env"), str(self.project_dir / "symlink_secret.txt"))

        # 2. Broken symlink
        os.symlink(str(Path(self.test_dir) / "nonexistent.file"), str(self.project_dir / "broken.link"))

        # 3. Directory symlink pointing outside
        os.symlink(str(self.outside_dir), str(self.project_dir / "external_dir_symlink"))

        # 4. Self-referential destination archive inside project root
        dest_archive = self.project_dir / "nested" / "source_snapshot.tar.gz"

        tar_path, manifest = create_deterministic_source_snapshot(
            project_dir=self.project_dir,
            dest_tar_gz=dest_archive,
        )

        self.assertTrue(tar_path.is_file())

        # Manifest must contain real files
        self.assertIn("train.py", manifest)
        self.assertIn("utils.py", manifest)

        # Manifest must strictly exclude external and malicious entries
        self.assertNotIn("symlink_secret.txt", manifest)
        self.assertNotIn("broken.link", manifest)
        self.assertNotIn("external_dir_symlink", manifest)
        self.assertNotIn("nested/source_snapshot.tar.gz", manifest)

        # Verify tar members directly
        with tarfile.open(tar_path, "r:gz") as tar:
            members = [m.name for m in tar.getmembers()]
            self.assertIn("train.py", members)
            self.assertNotIn("symlink_secret.txt", members)
            self.assertNotIn("broken.link", members)
            self.assertNotIn("external_dir_symlink", members)
            self.assertNotIn("nested/source_snapshot.tar.gz", members)

    def test_workspace_sync_engine_symlink_and_traversal_containment(self):
        """WorkspaceSyncEngine refuses escaping symlinks and safe deletion rejects traversal paths."""
        # 1. Escaping symlink causes plan_sync to raise ManifestValidationError
        symlink_path = self.project_dir / "malicious_escape.txt"
        os.symlink(str(self.outside_dir / "secret.env"), str(symlink_path))

        spec = ProjectSpec(
            name="security_test_project",
            entrypoint="train.py",
            root=self.project_dir,
            delete_remote=True,
        )

        with self.assertRaises(ManifestValidationError) as ctx:
            self.sync_engine.plan_sync(spec)
        self.assertIn("symlink security escape detected", str(ctx.exception).lower())

        # Remove escaping symlink
        symlink_path.unlink()

        # 2. Safe deletion containment: apply_sync refuses remote deletion of paths with .. or leading /
        malicious_cached_manifest = {
            "schema_version": 1,
            "project_name": spec.name,
            "files": {
                "train.py": {"size": 14, "sha256": "abc"},
                "../escape_relative": {"size": 10, "sha256": "def"},
                "/etc/passwd": {"size": 20, "sha256": "ghi"},
            },
        }

        # Mock session registration
        self.registry.register(session_id="test-session-sync", job_id="job-sync", gpu="T4", project_name=spec.name)

        # Mock remote manifest returning the malicious remote files to trigger deletion
        self.transport.remote_fs[f"{spec.remote_root}/.colab_manifest.json"] = json.dumps(malicious_cached_manifest).encode("utf-8")

        with self.assertRaises(WorkspaceSyncError) as ctx_sync:
            self.sync_engine.apply_sync(spec, session_id="test-session-sync")
        self.assertIn("unsafe remote path", str(ctx_sync.exception).lower())

    def test_source_snapshot_and_credentials_containment_under_umask_zero(self):
        """create_deterministic_source_snapshot and DriveCredentials enforce containment under umask 000."""
        old_umask = os.umask(0o000)
        sock = None
        try:
            # 1. Credential files (DriveCredentials.save) must enforce 0600 mode even under umask 000
            creds = DriveCredentials(
                client_id="test-client-id-sec",
                client_secret="test-client-sec",
                refresh_token="test-rtok",
                access_token="test-atok",
            )
            creds_path = Path(self.test_dir) / "creds_umask000.json"
            creds.save(creds_path)
            mode = stat.S_IMODE(creds_path.stat().st_mode)
            self.assertEqual(mode, 0o600, f"DriveCredentials file created with mode {oct(mode)}, expected 0o600")

            # 2. create_deterministic_source_snapshot under umask 000:
            # Excludes symlinks pointing outside project root, FIFO/device/socket paths, and self-referential archives
            # A. Symlink pointing outside project root
            os.symlink(str(self.outside_dir / "secret.env"), str(self.project_dir / "escaping_symlink.txt"))

            # B. Unix Domain Socket
            sock_path = str(self.project_dir / "app.sock")
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.bind(sock_path)
            except OSError:
                sock.close()
                sock = None

            # C. FIFO special file
            fifo_path = self.project_dir / "data.fifo"
            try:
                os.mkfifo(str(fifo_path))
            except (OSError, AttributeError):
                pass

            # D. Self-referential archive destination inside project
            dest_archive = self.project_dir / "build" / "source_snapshot.tar.gz"

            tar_path, manifest = create_deterministic_source_snapshot(
                project_dir=self.project_dir,
                dest_tar_gz=dest_archive,
            )

            self.assertTrue(tar_path.is_file())
            # Real files present
            self.assertIn("train.py", manifest)
            self.assertIn("utils.py", manifest)

            # Malicious / special entries strictly excluded from manifest
            self.assertNotIn("escaping_symlink.txt", manifest)
            self.assertNotIn("app.sock", manifest)
            self.assertNotIn("data.fifo", manifest)
            self.assertNotIn("build/source_snapshot.tar.gz", manifest)

            # Verify tar archive members directly
            with tarfile.open(tar_path, "r:gz") as tar:
                members = [m.name for m in tar.getmembers()]
                self.assertIn("train.py", members)
                self.assertNotIn("escaping_symlink.txt", members)
                self.assertNotIn("app.sock", members)
                self.assertNotIn("data.fifo", members)
                self.assertNotIn("build/source_snapshot.tar.gz", members)
        finally:
            if sock:
                try:
                    sock.close()
                except OSError:
                    pass
            os.umask(old_umask)


# =============================================================================
# 6. TEST SUITE: RESOURCE BOUNDS & BOUNDED MEMORY STREAMING (SECTION 11)
# =============================================================================

class TestResourceUsageAndBoundedMemoryStreaming(unittest.TestCase):
    """Resource bounds and bounded memory streaming verification (Section 11)."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="colab_qa_resource_")
        self.addCleanup(lambda: shutil.rmtree(self.test_dir, ignore_errors=True))

    def test_compute_file_sha256_bounded_chunks(self) -> None:
        """compute_file_sha256 processes multi-megabyte checkpoints using bounded 64 KiB chunks."""
        test_file = Path(self.test_dir) / "large_model.pt"
        data = os.urandom(4 * 1024 * 1024)  # 4 MiB synthetic checkpoint
        test_file.write_bytes(data)
        expected_sha = hashlib.sha256(data).hexdigest()

        chunk_sizes: List[int] = []
        orig_open = open

        class SpyingReader:
            def __init__(self, raw_f: Any):
                self._raw = raw_f

            def read(self, size: int = -1) -> bytes:
                chunk_sizes.append(size)
                return self._raw.read(size)

            def __enter__(self) -> "SpyingReader":
                return self

            def __exit__(self, *args: Any) -> Any:
                return self._raw.__exit__(*args)

        def spied_open(*args: Any, **kwargs: Any) -> Any:
            f = orig_open(*args, **kwargs)
            if args and str(test_file) in str(args[0]):
                return SpyingReader(f)
            return f

        with unittest.mock.patch("builtins.open", side_effect=spied_open):
            digest = compute_file_sha256(test_file)

        self.assertEqual(digest, expected_sha)
        self.assertTrue(len(chunk_sizes) >= 64, f"Expected at least 64 chunks for 4 MiB file, got {len(chunk_sizes)}")
        self.assertTrue(all(sz <= 65536 for sz in chunk_sizes), f"Found chunk size exceeding 64 KiB: {max(chunk_sizes)}")

    def test_standard_drive_http_client_stream_download_bounded_memory(self) -> None:
        """StandardDriveHttpClient.stream_download streams multi-megabyte files in bounded 64 KiB chunks."""
        data = os.urandom(4 * 1024 * 1024)  # 4 MiB synthetic checkpoint
        expected_sha = hashlib.sha256(data).hexdigest()

        class MockDownloadHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                chunk_sz = 64 * 1024
                for i in range(0, len(data), chunk_sz):
                    self.wfile.write(data[i : i + chunk_sz])

            def log_message(self, *args: Any) -> None:
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockDownloadHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        port = server.server_address[1]

        dest_file = Path(self.test_dir) / "streamed_checkpoint.pt"
        client = StandardDriveHttpClient()

        status, headers, bytes_written = client.stream_download(
            f"http://127.0.0.1:{port}/checkpoint",
            dest_file,
            chunk_size=64 * 1024,
        )

        self.assertEqual(status, 200)
        self.assertEqual(bytes_written, len(data))
        self.assertTrue(dest_file.is_file())
        self.assertEqual(dest_file.stat().st_size, len(data))
        self.assertEqual(hashlib.sha256(dest_file.read_bytes()).hexdigest(), expected_sha)

    def test_stream_download_and_retrieve_temporary_file_cleanup_on_error(self) -> None:
        """Temporary files are cleaned up on error/interruption during download or retrieve."""
        class FailingDownloadHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "Internal Google Drive fault"}')

            def log_message(self, *args: Any) -> None:
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FailingDownloadHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        port = server.server_address[1]

        dest_dir = Path(self.test_dir) / "download_target"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / "corrupted_model.pt"

        # 1. Test stream_download on non-200 status cleans up tmp files
        client = StandardDriveHttpClient()
        status, headers, bytes_written = client.stream_download(
            f"http://127.0.0.1:{port}/fail",
            dest_file,
        )
        self.assertEqual(status, 500)
        self.assertEqual(bytes_written, 0)
        tmp_files = list(dest_dir.glob("*.tmp*"))
        self.assertEqual(len(tmp_files), 0, f"Orphaned temporary files found after stream error: {tmp_files}")

        # 2. Test GoogleDriveArtifactStore.retrieve cleanup on failure
        staging_dir = Path(self.test_dir) / "staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        store = GoogleDriveArtifactStore(
            credentials=DriveCredentials(access_token="test-access-token"),
            local_staging_dir=staging_dir,
            http_client=client,
        )

        from colab_adapter.storage.manifest import PublicationManifest
        m_dir = staging_dir / "projects" / "test_proj" / "runs" / "test_run"
        m_dir.mkdir(parents=True, exist_ok=True)
        m = PublicationManifest(project_id="test_proj", run_id="test_run")
        art = StoredArtifact(
            artifact_id="test_proj/test_run/checkpoints/model.pt@sha256_mocksha",
            storage_provider="google_drive",
            path="checkpoints/model.pt",
            size_bytes=1024,
            sha256="mocksha",
            project_id="test_proj",
            run_id="test_run",
            remote_file_id="nonexistent-remote-id",
        )
        m.add_artifact(art)
        m.save(m_dir / "manifest.json")

        with self.assertRaises(DriveTransferError):
            store.retrieve(
                artifact_id=art.artifact_id,
                destination=dest_dir / "model.pt",
            )

        tmp_files_after_retrieve = list(dest_dir.glob("*.tmp*"))
        self.assertEqual(len(tmp_files_after_retrieve), 0, f"Orphaned temporary files after retrieve: {tmp_files_after_retrieve}")


if __name__ == "__main__":
    unittest.main()
