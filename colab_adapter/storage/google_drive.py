"""Google Drive artifact storage adapter with resumable uploads and integrity validation.

Implements the official Google Drive API v3 protocol:
- Resumable chunked uploads for large checkpoints and snapshots
- Interrupted upload recovery via Range queries
- Bounded exponential backoff and retry
- Deterministic folder hierarchy: Colab-Research/projects/<project-id>/runs/<run-id>/...
- Folder ID caching and duplicate prevention
- Multi-tier integrity verification (local SHA256 canonical, remote md5Checksum & size verification)
- Protected local credential storage with 0600 permissions
- Fully testable via injectable DriveHttpClient abstraction (offline-safe)
"""
from __future__ import annotations

import io
import json
import logging
import os
import random
import re
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

from ..errors import (
    ArtifactNotFoundError,
    ArtifactValidationError,
    DriveAuthenticationError,
    DriveIntegrityError,
    DriveQuotaExceededError,
    DriveTransferError,
)
from .base import ArtifactStore, ArtifactType, StoredArtifact, _utc_now_iso
from .manifest import PublicationManifest, compute_file_md5, compute_file_sha256
from .transfer import TransferJournal, TransferRecord, TransferStatus

DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1 MiB chunks (multiple of 256 KiB)
DEFAULT_DRIVE_ROOT_FOLDER = "Colab-Research"

_SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


def _validate_safe_id(name: str, field: str = "identifier") -> str:
    if not name or not isinstance(name, str) or not _SAFE_ID_PATTERN.match(name) or ".." in name or "/" in name or "\\" in name:
        raise ArtifactValidationError(
            f"Invalid {field} '{name}': contains illegal characters or path traversal sequence.",
            details={field: str(name)},
        )
    return name



@dataclass
class DriveCredentials:
    """OAuth2 credentials for Google Drive API."""
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    refresh_token: Optional[str] = None
    access_token: Optional[str] = None
    token_expiry: Optional[float] = None
    token_uri: str = "https://oauth2.googleapis.com/token"

    def is_valid(self) -> bool:
        if not self.access_token:
            return False
        if self.token_expiry and time.time() >= self.token_expiry - 60:
            return False
        return True

    def save(self, file_path: Union[str, Path]) -> None:
        p = Path(file_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token,
            "access_token": self.access_token,
            "token_expiry": self.token_expiry,
            "token_uri": self.token_uri,
        }
        # Secure file permissions (0600)
        fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "DriveCredentials":
        p = Path(file_path).resolve()
        if not p.is_file():
            raise DriveAuthenticationError(f"Drive credentials file not found: {p}")
        try:
            mode = p.stat().st_mode
            if mode & (stat.S_IRWXG | stat.S_IRWXO):
                os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            raise DriveAuthenticationError(f"Failed to read Drive credentials: {e}") from e



class DriveHttpClient(Protocol):
    """Protocol for HTTP interactions with Google Drive API."""

    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[bytes] = None,
        timeout: float = 30.0,
    ) -> Tuple[int, Dict[str, str], bytes]:
        """Send request and return (status_code, headers_dict, body_bytes)."""
        ...


class StandardDriveHttpClient:
    """Default HTTP client using standard library urllib."""

    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[bytes] = None,
        timeout: float = 30.0,
    ) -> Tuple[int, Dict[str, str], bytes]:
        req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_headers = {k.lower(): v for k, v in resp.getheaders()}
                return resp.status, resp_headers, resp.read()
        except urllib.error.HTTPError as e:
            resp_headers = {k.lower(): v for k, v in e.headers.items()}
            return e.code, resp_headers, e.read()
        except urllib.error.URLError as ue:
            raise DriveTransferError(f"Network error contacting Google Drive: {ue}") from ue

    def stream_download(
        self,
        url: str,
        dest_path: Union[str, Path],
        headers: Optional[Dict[str, str]] = None,
        chunk_size: int = 64 * 1024,
        timeout: float = 60.0,
    ) -> Tuple[int, Dict[str, str], int]:
        """Stream response body directly into destination file with bounded memory."""
        req = urllib.request.Request(url, headers=headers or {}, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_headers = {k.lower(): v for k, v in resp.getheaders()}
                if resp.status != 200:
                    return resp.status, resp_headers, 0
                bytes_written = 0
                with open(dest_path, "wb") as f:
                    while True:
                        buf = resp.read(chunk_size)
                        if not buf:
                            break
                        f.write(buf)
                        bytes_written += len(buf)
                    f.flush()
                    os.fsync(f.fileno())
                return resp.status, resp_headers, bytes_written
        except urllib.error.HTTPError as e:
            resp_headers = {k.lower(): v for k, v in e.headers.items()}
            return e.code, resp_headers, 0
        except urllib.error.URLError as ue:
            raise DriveTransferError(f"Network error downloading Drive file: {ue}") from ue


class GoogleDriveArtifactStore:
    """ArtifactStore implementation for Google Drive."""

    def __init__(
        self,
        credentials: Optional[DriveCredentials] = None,
        credentials_file: Optional[Union[str, Path]] = None,
        root_folder_name: str = DEFAULT_DRIVE_ROOT_FOLDER,
        local_staging_dir: Union[str, Path] = "./artifacts",
        journal_path: Optional[Union[str, Path]] = None,
        http_client: Optional[DriveHttpClient] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_retries: int = 5,
        initial_backoff_sec: float = 0.5,
    ):
        self.credentials = credentials
        self.credentials_file = Path(credentials_file).resolve() if credentials_file else None
        self.root_folder_name = root_folder_name
        self.staging_dir = Path(local_staging_dir).resolve()
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        if chunk_size <= 0 or chunk_size % (256 * 1024) != 0:
            raise ValueError(f"chunk_size must be a positive multiple of 256 KiB (262144 bytes), got {chunk_size}")
        self.chunk_size = chunk_size

        self.max_retries = max_retries
        self.initial_backoff_sec = initial_backoff_sec
        self.http_client = http_client or StandardDriveHttpClient()

        journal_file = journal_path or (self.staging_dir / "drive_transfers.json")
        self.journal = TransferJournal(journal_file)

        # In-memory cache for resolved Drive folder IDs: path -> folder_id
        self._folder_cache: Dict[str, str] = {}
        # Local cache of publication manifests: (project_id, run_id) -> PublicationManifest
        self._manifest_cache: Dict[Tuple[str, str], PublicationManifest] = {}

    @property
    def provider_name(self) -> str:
        return "google_drive"

    def _ensure_token(self) -> str:
        """Return valid OAuth access token, refreshing if necessary."""
        if not self.credentials and self.credentials_file and self.credentials_file.exists():
            self.credentials = DriveCredentials.load(self.credentials_file)

        if not self.credentials:
            raise DriveAuthenticationError(
                "No Google Drive credentials configured. Set credentials or credentials_file."
            )

        if self.credentials.is_valid() and self.credentials.access_token:
            return self.credentials.access_token

        # Attempt token refresh if refresh_token is present
        if self.credentials.refresh_token and self.credentials.client_id and self.credentials.client_secret:
            return self._refresh_access_token()

        if self.credentials.access_token:
            return self.credentials.access_token

        raise DriveAuthenticationError("Drive credentials expired and cannot be refreshed (no refresh token).")

    def _refresh_access_token(self) -> str:
        assert self.credentials is not None
        payload = urllib.parse.urlencode({
            "client_id": self.credentials.client_id,
            "client_secret": self.credentials.client_secret,
            "refresh_token": self.credentials.refresh_token,
            "grant_type": "refresh_token",
        }).encode("utf-8")

        status, headers, body = self.http_client.request(
            "POST",
            self.credentials.token_uri,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=payload,
        )
        if status != 200:
            raise DriveAuthenticationError(
                f"Failed to refresh Drive access token (HTTP {status}): {body.decode('utf-8', errors='replace')}"
            )

        data = json.loads(body.decode("utf-8"))
        self.credentials.access_token = data["access_token"]
        expires_in = data.get("expires_in", 3600)
        self.credentials.token_expiry = time.time() + float(expires_in)

        if self.credentials_file:
            self.credentials.save(self.credentials_file)

        return self.credentials.access_token

    def _request_with_retry(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[bytes] = None,
    ) -> Tuple[int, Dict[str, str], bytes]:
        """Execute HTTP request with bounded exponential backoff."""
        token = self._ensure_token()
        hdrs = dict(headers or {})
        hdrs["authorization"] = f"Bearer {token}"

        backoff = self.initial_backoff_sec
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                status, resp_headers, body = self.http_client.request(
                    method, url, headers=hdrs, data=data
                )

                if status == 401:
                    # Token expired; refresh and retry once if possible
                    if attempt < self.max_retries:
                        token = self._refresh_access_token()
                        hdrs["authorization"] = f"Bearer {token}"
                        continue
                    raise DriveAuthenticationError(f"Drive authentication rejected (401): {body.decode('utf-8', errors='replace')}")

                if status == 403:
                    body_text = body.decode("utf-8", errors="replace")
                    if "rateLimitExceeded" in body_text or "userRateLimitExceeded" in body_text:
                        # Rate limit retry
                        pass
                    elif "quotaExceeded" in body_text or "storageQuotaExceeded" in body_text:
                        raise DriveQuotaExceededError(f"Drive storage quota exceeded: {body_text}")
                    else:
                        raise DriveAuthenticationError(f"Drive access forbidden (403): {body_text}")

                if status in (429, 500, 502, 503, 504):
                    if attempt == self.max_retries:
                        raise DriveTransferError(
                            f"Drive request failed after {self.max_retries} retries (HTTP {status}): {body.decode('utf-8', errors='replace')}"
                        )
                    sleep_time = backoff * (1.0 + random.uniform(0.1, 0.5))
                    time.sleep(sleep_time)
                    backoff *= 2.0
                    continue

                return status, resp_headers, body

            except (DriveAuthenticationError, DriveQuotaExceededError):
                raise
            except Exception as e:
                last_error = e
                if attempt == self.max_retries:
                    raise DriveTransferError(f"Drive request failed after {self.max_retries} retries: {e}") from e
                sleep_time = backoff * (1.0 + random.uniform(0.1, 0.5))
                time.sleep(sleep_time)
                backoff *= 2.0

        raise DriveTransferError(f"Drive request retry loop exhausted: {last_error}")

    def _find_or_create_folder(self, folder_name: str, parent_id: Optional[str] = None) -> str:
        """Find existing folder by name and parent or create it safely."""
        cache_key = f"{parent_id or 'root'}::{folder_name}"
        if cache_key in self._folder_cache:
            return self._folder_cache[cache_key]

        # Query existing folder
        q = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        if parent_id:
            q += f" and '{parent_id}' in parents"
        else:
            q += " and 'root' in parents"

        url = f"https://www.googleapis.com/drive/v3/files?q={urllib.parse.quote(q)}&fields=files(id,name)&spaces=drive"
        status, headers, body = self._request_with_retry("GET", url)
        if status == 200:
            data = json.loads(body.decode("utf-8"))
            files = data.get("files", [])
            if files:
                folder_id = files[0]["id"]
                self._folder_cache[cache_key] = folder_id
                return folder_id

        # Folder does not exist, create it
        metadata: Dict[str, Any] = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            metadata["parents"] = [parent_id]

        create_url = "https://www.googleapis.com/drive/v3/files?fields=id,name"
        payload = json.dumps(metadata).encode("utf-8")
        status, headers, body = self._request_with_retry(
            "POST",
            create_url,
            headers={"Content-Type": "application/json; charset=UTF-8"},
            data=payload,
        )
        if status not in (200, 201):
            raise DriveTransferError(
                f"Failed to create Drive folder '{folder_name}' (HTTP {status}): {body.decode('utf-8', errors='replace')}"
            )

        res = json.loads(body.decode("utf-8"))
        folder_id = res["id"]
        self._folder_cache[cache_key] = folder_id
        return folder_id

    def resolve_target_folder_id(self, project_id: str, run_id: str, subfolder: Optional[str] = None) -> str:
        """Resolve full hierarchical Drive folder path:
        Colab-Research -> projects -> <project-id> -> runs -> <run-id> [-> <subfolder>]
        """
        _validate_safe_id(project_id, "project_id")
        _validate_safe_id(run_id, "run_id")
        if subfolder:
            _validate_safe_id(subfolder, "subfolder")

        root_id = self._find_or_create_folder(self.root_folder_name, parent_id=None)

        projects_id = self._find_or_create_folder("projects", parent_id=root_id)
        proj_id = self._find_or_create_folder(project_id, parent_id=projects_id)
        runs_id = self._find_or_create_folder("runs", parent_id=proj_id)
        run_folder_id = self._find_or_create_folder(run_id, parent_id=runs_id)

        if subfolder:
            return self._find_or_create_folder(subfolder, parent_id=run_folder_id)
        return run_folder_id

    def _find_file_in_folder(self, filename: str, folder_id: str) -> Optional[Dict[str, Any]]:
        """Query for existing non-trashed file with matching name in folder to prevent duplicates."""
        safe_name = filename.replace("'", "\\'")
        query = f"name = '{safe_name}' and '{folder_id}' in parents and trashed = false"
        encoded = urllib.parse.quote(query)
        url = f"https://www.googleapis.com/drive/v3/files?q={encoded}&fields=files(id,name,size,md5Checksum)"
        try:
            status, _, body = self._request_with_retry("GET", url)
            if status == 200:
                res = json.loads(body.decode("utf-8"))
                files = res.get("files", [])
                if files:
                    return files[0]
        except Exception:
            pass
        return None

    def _initiate_resumable_upload(
        self,
        filename: str,
        parent_id: str,
        total_size: int,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Initiate resumable upload session and return the upload URL."""
        token = self._ensure_token()
        init_url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&fields=id,name,size,md5Checksum"
        metadata = {
            "name": filename,
            "parents": [parent_id],
        }
        meta_bytes = json.dumps(metadata).encode("utf-8")
        hdrs = {
            "authorization": f"Bearer {token}",
            "content-type": "application/json; charset=UTF-8",
            "x-upload-content-type": content_type,
            "x-upload-content-length": str(total_size),
        }

        status, resp_headers, body = self.http_client.request("POST", init_url, headers=hdrs, data=meta_bytes)
        if status != 200:
            raise DriveTransferError(
                f"Failed to initiate resumable upload for '{filename}' (HTTP {status}): {body.decode('utf-8', errors='replace')}"
            )

        location = resp_headers.get("location")
        if not location:
            raise DriveTransferError(f"Drive API did not return Location header for resumable upload of '{filename}'")
        return location

    def _query_upload_progress(self, session_url: str, total_size: int) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Query Drive for received byte offset or completed upload response."""
        hdrs = {
            "content-range": f"bytes */{total_size}",
            "content-length": "0",
        }
        status, resp_headers, body = self.http_client.request("PUT", session_url, headers=hdrs)
        if status == 308:
            range_hdr = resp_headers.get("range", "")
            if range_hdr and "-" in range_hdr:
                # range format: "bytes=0-1048575"
                m = re.search(r"bytes=0-(\d+)", range_hdr)
                if m:
                    return int(m.group(1)) + 1, None
            return 0, None
        elif status in (200, 201):
            try:
                res = json.loads(body.decode("utf-8"))
                return total_size, res
            except Exception:
                return total_size, {}
        elif status in (404, 410):
            return 0, None
        return 0, None

    def upload_file_resumable(
        self,
        local_path: Union[str, Path],
        target_folder_id: str,
        transfer_record: TransferRecord,
    ) -> Dict[str, Any]:
        """Perform chunked resumable upload with state updates in transfer journal."""
        src = Path(local_path).resolve()
        total_size = src.stat().st_size
        filename = src.name

        # Check if identical file already exists in target folder (idempotency)
        existing = self._find_file_in_folder(filename, target_folder_id)
        if existing and int(existing.get("size", -1)) == total_size:
            transfer_record.bytes_transferred = total_size
            transfer_record.remote_file_id = existing.get("id")
            self.journal.record_transfer(transfer_record)
            return existing

        session_url = transfer_record.resumable_session_url
        if not session_url:
            session_url = self._initiate_resumable_upload(filename, target_folder_id, total_size)
            transfer_record.resumable_session_url = session_url
            self.journal.record_transfer(transfer_record)

        uploaded_bytes = 0
        if transfer_record.bytes_transferred > 0:
            uploaded_bytes, completed_meta = self._query_upload_progress(session_url, total_size)
            if completed_meta:
                transfer_record.bytes_transferred = total_size
                transfer_record.remote_file_id = completed_meta.get("id")
                self.journal.record_transfer(transfer_record)
                return completed_meta
            transfer_record.bytes_transferred = uploaded_bytes
            self.journal.record_transfer(transfer_record)

        with open(src, "rb") as f:
            while uploaded_bytes < total_size:
                f.seek(uploaded_bytes)
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break

                chunk_len = len(chunk)
                start_byte = uploaded_bytes
                end_byte = uploaded_bytes + chunk_len - 1

                hdrs = {
                    "content-range": f"bytes {start_byte}-{end_byte}/{total_size}",
                    "content-length": str(chunk_len),
                    "content-type": "application/octet-stream",
                }

                chunk_backoff = self.initial_backoff_sec
                for attempt in range(self.max_retries + 1):
                    try:
                        status, resp_headers, body = self.http_client.request("PUT", session_url, headers=hdrs, data=chunk)
                    except Exception as ce:
                        # Ambiguous network failure: query server to reconcile committed offset
                        synced_offset, final_res = self._query_upload_progress(session_url, total_size)
                        if final_res:
                            transfer_record.bytes_transferred = total_size
                            transfer_record.remote_file_id = final_res.get("id")
                            self.journal.record_transfer(transfer_record)
                            return final_res
                        if synced_offset > uploaded_bytes:
                            uploaded_bytes = synced_offset
                            transfer_record.bytes_transferred = uploaded_bytes
                            self.journal.record_transfer(transfer_record)
                            break

                        if attempt == self.max_retries:
                            raise DriveTransferError(f"Chunk upload network failure after {self.max_retries} retries: {ce}") from ce
                        time.sleep(chunk_backoff * (1.0 + random.uniform(0.1, 0.4)))
                        chunk_backoff *= 2.0
                        continue

                    if status == 308:
                        range_hdr = resp_headers.get("range", "")
                        m = re.search(r"bytes=0-(\d+)", range_hdr)
                        if m:
                            uploaded_bytes = int(m.group(1)) + 1
                        else:
                            synced_offset, _ = self._query_upload_progress(session_url, total_size)
                            uploaded_bytes = max(uploaded_bytes + chunk_len, synced_offset)

                        transfer_record.bytes_transferred = uploaded_bytes
                        self.journal.record_transfer(transfer_record)
                        break
                    elif status in (200, 201):
                        res = json.loads(body.decode("utf-8"))
                        transfer_record.bytes_transferred = total_size
                        transfer_record.remote_file_id = res.get("id")
                        self.journal.record_transfer(transfer_record)
                        return res
                    elif status in (404, 410):
                        transfer_record.resumable_session_url = None
                        self.journal.record_transfer(transfer_record)
                        # Check whether file was committed before session expired
                        existing_after_expiry = self._find_file_in_folder(filename, target_folder_id)
                        if existing_after_expiry and int(existing_after_expiry.get("size", -1)) == total_size:
                            transfer_record.bytes_transferred = total_size
                            transfer_record.remote_file_id = existing_after_expiry.get("id")
                            self.journal.record_transfer(transfer_record)
                            return existing_after_expiry

                        # Re-initiate upload session
                        session_url = self._initiate_resumable_upload(filename, target_folder_id, total_size)
                        transfer_record.resumable_session_url = session_url
                        uploaded_bytes = 0
                        transfer_record.bytes_transferred = 0
                        self.journal.record_transfer(transfer_record)
                        break
                    elif status in (429, 500, 502, 503, 504):
                        if attempt == self.max_retries:
                            error_msg = f"Resumable upload chunk failed after {self.max_retries} retries (HTTP {status}): {body.decode('utf-8', errors='replace')}"
                            transfer_record.last_error = error_msg
                            self.journal.record_transfer(transfer_record)
                            raise DriveTransferError(error_msg)
                        time.sleep(chunk_backoff * (1.0 + random.uniform(0.1, 0.4)))
                        chunk_backoff *= 2.0
                        # Query upload progress to sync offset
                        synced_offset, final_res = self._query_upload_progress(session_url, total_size)
                        if final_res:
                            return final_res
                        if synced_offset > uploaded_bytes:
                            uploaded_bytes = synced_offset
                            break
                        continue
                    else:
                        error_msg = f"Resumable upload chunk failed (HTTP {status}): {body.decode('utf-8', errors='replace')}"
                        transfer_record.last_error = error_msg
                        self.journal.record_transfer(transfer_record)
                        raise DriveTransferError(error_msg)

        # If zero-byte file or final query
        if total_size == 0:
            hdrs = {
                "content-range": "bytes */0",
                "content-length": "0",
            }
            status, _, body = self.http_client.request("PUT", session_url, headers=hdrs)
            if status in (200, 201):
                res = json.loads(body.decode("utf-8"))
                transfer_record.bytes_transferred = 0
                transfer_record.remote_file_id = res.get("id")
                self.journal.record_transfer(transfer_record)
                return res

        raise DriveTransferError(f"Upload completed without final 200/201 response for '{filename}'")


    def publish(
        self,
        source: Union[str, Path],
        project_id: str,
        run_id: str,
        artifact_type: str = ArtifactType.CHECKPOINT.value,
        metadata: Optional[Dict[str, Any]] = None,
        expected_sha256: Optional[str] = None,
    ) -> StoredArtifact:
        """Publish a local file to Google Drive through verified resumable upload."""
        src = Path(source).resolve()
        if not src.is_file():
            raise ArtifactValidationError(f"Source file does not exist: {src}")

        # Step 1: Verify local SHA256 & MD5
        local_sha256 = compute_file_sha256(src)
        if expected_sha256 and expected_sha256.lower() != local_sha256.lower():
            raise ArtifactValidationError(
                f"Local checksum mismatch before Drive upload: expected {expected_sha256}, got {local_sha256}"
            )
        local_md5 = compute_file_md5(src)
        size_bytes = src.stat().st_size

        # Resolve subfolder
        subfolder = None
        if artifact_type == ArtifactType.CHECKPOINT.value:
            subfolder = "checkpoints"
        elif artifact_type == ArtifactType.OUTPUT.value:
            subfolder = "outputs"

        target_folder_id = self.resolve_target_folder_id(project_id, run_id, subfolder)
        rel_path = f"{subfolder}/{src.name}" if subfolder else src.name
        artifact_id = f"{project_id}/{run_id}/{rel_path}"

        # Initialize transfer record in journal
        transfer_id = f"{self.provider_name}-{project_id}-{run_id}-{src.name}"
        record = self.journal.get_transfer(transfer_id)

        if not record:
            record = TransferRecord(
                transfer_id=transfer_id,
                project_id=project_id,
                run_id=run_id,
                artifact_id=artifact_id,
                local_path=str(src),
                target_rel_path=rel_path,
                size_bytes=size_bytes,
                sha256=local_sha256,
                md5=local_md5,
                status=TransferStatus.LOCAL_STAGED.value,
            )
            self.journal.record_transfer(record)

        # Transition: LOCAL_STAGED -> LOCAL_VERIFIED -> UPLOAD_PENDING -> UPLOADING
        if record.status == TransferStatus.LOCAL_STAGED.value:
            record.transition_to(TransferStatus.LOCAL_VERIFIED)
            self.journal.record_transfer(record)

        if record.status == TransferStatus.LOCAL_VERIFIED.value:
            record.transition_to(TransferStatus.UPLOAD_PENDING)
            self.journal.record_transfer(record)

        if record.status in (TransferStatus.UPLOAD_PENDING.value, TransferStatus.RETRY_PENDING.value):
            record.transition_to(TransferStatus.UPLOADING)
            self.journal.record_transfer(record)

        try:
            drive_file_meta = self.upload_file_resumable(src, target_folder_id, record)

            # Step 2: Authoritative Integrity Verification
            remote_id = drive_file_meta.get("id")
            if not remote_id:
                raise DriveIntegrityError("Drive upload succeeded but no remote file ID was returned.")

            # Never substitute local expected size: fetch authoritative remote metadata if missing
            remote_size_raw = drive_file_meta.get("size")
            remote_md5 = drive_file_meta.get("md5Checksum")

            if remote_size_raw is None or not remote_md5:
                query_url = f"https://www.googleapis.com/drive/v3/files/{remote_id}?fields=id,name,size,md5Checksum"
                try:
                    q_status, _, q_body = self._request_with_retry("GET", query_url)
                    if q_status == 200:
                        authoritative_meta = json.loads(q_body.decode("utf-8"))
                        if authoritative_meta.get("size") is not None:
                            remote_size_raw = authoritative_meta["size"]
                        if authoritative_meta.get("md5Checksum"):
                            remote_md5 = authoritative_meta["md5Checksum"]
                except Exception:
                    pass

            if remote_size_raw is None:
                raise DriveIntegrityError(
                    f"Drive remote file '{remote_id}' missing authoritative size in response and file metadata; cannot verify integrity.",
                    details={"remote_id": remote_id},
                )

            remote_size = int(remote_size_raw)
            if remote_size != size_bytes:
                raise DriveIntegrityError(
                    f"Drive file size mismatch: local {size_bytes} bytes, remote {remote_size} bytes",
                    details={"local_size": size_bytes, "remote_size": remote_size},
                )

            if remote_md5 and remote_md5.lower() != local_md5.lower():
                raise DriveIntegrityError(
                    f"Drive MD5 mismatch: local {local_md5}, remote {remote_md5}",
                    details={"local_md5": local_md5, "remote_md5": remote_md5},
                )

            # Step 3: DRIVE_VERIFIED -> PUBLISHED
            record.transition_to(TransferStatus.DRIVE_VERIFIED)
            self.journal.record_transfer(record)

            record.transition_to(TransferStatus.PUBLISHED)
            record.remote_file_id = remote_id
            self.journal.record_transfer(record)

            stored = StoredArtifact(
                artifact_id=artifact_id,
                storage_provider=self.provider_name,
                path=rel_path,
                size_bytes=size_bytes,
                sha256=local_sha256,
                project_id=project_id,
                run_id=run_id,
                artifact_type=artifact_type,
                remote_file_id=remote_id,
                md5=local_md5,
                verified_at=_utc_now_iso(),
                metadata=metadata or {},
            )

            # Update and persist manifest locally and optionally in Drive
            m_key = (project_id, run_id)
            local_manifest_path = self.staging_dir / "projects" / project_id / "runs" / run_id / "manifest.json"
            if local_manifest_path.exists():
                manifest = PublicationManifest.load(local_manifest_path)
            elif m_key in self._manifest_cache:
                manifest = self._manifest_cache[m_key]
            else:
                manifest = PublicationManifest(project_id=project_id, run_id=run_id)

            existing_art = manifest.get_artifact(artifact_id)
            if existing_art and existing_art.storage_provider == "local":
                # Preserve verified local file path while adding remote Drive file ID
                stored = StoredArtifact(
                    artifact_id=artifact_id,
                    storage_provider="google_drive",
                    path=existing_art.path,
                    size_bytes=size_bytes,
                    sha256=local_sha256,
                    project_id=project_id,
                    run_id=run_id,
                    artifact_type=artifact_type,
                    remote_file_id=remote_id,
                    md5=local_md5,
                    verified_at=_utc_now_iso(),
                    metadata={**existing_art.metadata, **(metadata or {})},
                )

            self._manifest_cache[m_key] = manifest
            manifest.add_artifact(stored)
            manifest.status = "PUBLISHED"
            manifest.save(local_manifest_path)

            return stored


        except Exception as e:
            record.retry_count += 1
            if record.retry_count <= record.max_retries:
                try:
                    record.transition_to(TransferStatus.RETRY_PENDING, error_message=str(e))
                except Exception:
                    record.status = TransferStatus.RETRY_PENDING.value
            else:
                try:
                    record.transition_to(TransferStatus.UPLOAD_FAILED, error_message=str(e))
                except Exception:
                    record.status = TransferStatus.UPLOAD_FAILED.value
            self.journal.record_transfer(record)
            raise

    def retrieve(self, artifact_id: str, destination: Union[str, Path]) -> Path:
        """Download artifact from Google Drive by ID with bounded memory streaming and integrity check."""
        meta = self.get_metadata(artifact_id)
        if not meta or not meta.remote_file_id:
            raise ArtifactNotFoundError(f"Artifact '{artifact_id}' not found in Google Drive registry.")

        dest = Path(destination).resolve()
        if dest.is_dir():
            dest = dest / Path(meta.path).name

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / f"{dest.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"

        download_url = f"https://www.googleapis.com/drive/v3/files/{meta.remote_file_id}?alt=media"
        downloaded_bytes = 0
        fd = None

        try:
            # Check if http_client implements stream_download
            if hasattr(self.http_client, "stream_download"):
                status, _, downloaded_bytes = self.http_client.stream_download(
                    download_url, tmp, headers={"authorization": f"Bearer {self._ensure_token()}"}
                )
                if status != 200:
                    raise DriveTransferError(f"Failed to download Drive file '{meta.remote_file_id}' (HTTP {status})")
                downloaded_sha256 = compute_file_sha256(tmp)
            else:
                # Fallback streaming write with bounded buffer and incremental hash
                status, headers, body = self._request_with_retry("GET", download_url)
                if status != 200:
                    raise DriveTransferError(f"Failed to download Drive file '{meta.remote_file_id}' (HTTP {status})")

                fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                hasher = hashlib.sha256()
                with os.fdopen(fd, "wb") as f:
                    fd = None
                    chunk_sz = 64 * 1024
                    for i in range(0, len(body), chunk_sz):
                        chunk = body[i : i + chunk_sz]
                        f.write(chunk)
                        hasher.update(chunk)
                        downloaded_bytes += len(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                downloaded_sha256 = hasher.hexdigest()

            # Verify integrity
            if downloaded_sha256.lower() != meta.sha256.lower():
                raise DriveIntegrityError(
                    f"Retrieved Drive artifact hash mismatch: recorded {meta.sha256}, got {downloaded_sha256}"
                )
            if meta.size_bytes and downloaded_bytes != meta.size_bytes:
                raise DriveIntegrityError(
                    f"Retrieved Drive artifact size mismatch: expected {meta.size_bytes}, got {downloaded_bytes}"
                )

            os.replace(str(tmp), str(dest))
            return dest

        except Exception:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            raise

    def exists(self, artifact_id: str) -> bool:
        meta = self.get_metadata(artifact_id)
        return meta is not None and meta.remote_file_id is not None

    def get_metadata(self, artifact_id: str) -> Optional[StoredArtifact]:
        parts = artifact_id.split("/")
        if len(parts) < 3:
            return None
        project_id, run_id = parts[0], parts[1]
        m_key = (project_id, run_id)
        if m_key in self._manifest_cache:
            return self._manifest_cache[m_key].get_artifact(artifact_id)

        local_m = self.staging_dir / "projects" / project_id / "runs" / run_id / "manifest.json"
        if local_m.exists():
            manifest = PublicationManifest.load(local_m)
            self._manifest_cache[m_key] = manifest
            return manifest.get_artifact(artifact_id)
        return None

    def list_artifacts(
        self,
        project_id: str,
        run_id: Optional[str] = None,
    ) -> List[StoredArtifact]:
        if run_id:
            m_key = (project_id, run_id)
            if m_key in self._manifest_cache:
                return self._manifest_cache[m_key].list_artifacts()
            local_m = self.staging_dir / "projects" / project_id / "runs" / run_id / "manifest.json"
            if local_m.exists():
                manifest = PublicationManifest.load(local_m)
                self._manifest_cache[m_key] = manifest
                return manifest.list_artifacts()
            return []

        # All runs
        results = []
        proj_dir = self.staging_dir / "projects" / project_id / "runs"
        if proj_dir.is_dir():
            for r in proj_dir.iterdir():
                if r.is_dir() and (r / "manifest.json").exists():
                    manifest = PublicationManifest.load(r / "manifest.json")
                    results.extend(manifest.list_artifacts())
        return results
