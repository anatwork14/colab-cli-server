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

import contextlib
import io
import json
import logging
import os
import random
import re
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple, Union

try:
    import fcntl
except ImportError:
    fcntl = None

import shutil

from ..errors import (
    ArtifactNotFoundError,
    ArtifactValidationError,
    DriveAuthenticationError,
    DriveIntegrityError,
    DriveQuotaExceededError,
    DriveTransferError,
    JournalLockError,
    StorageError,
)
from .base import ArtifactStore, ArtifactType, StoredArtifact, _utc_now_iso
from .drive_metadata import (
    STATE_HISTORICAL,
    STATE_LATEST,
    STATE_PENDING,
    build_revision_properties,
    revision_state,
)
from .manifest import PublicationManifest, compute_file_md5, compute_file_sha256
from .transfer import TransferJournal, TransferRecord, TransferStatus
from .drive_metadata import (
    DRIVE_PROPERTY_MAX_BYTES,
    DRIVE_MAX_APP_PROPERTIES,
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

ArtifactTransferError = DriveTransferError

DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1 MiB chunks (multiple of 256 KiB)
DEFAULT_DRIVE_ROOT_FOLDER = "Colab-Research"

_SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+$")

_NAMED_LOCK_REGISTRY: Dict[str, threading.RLock] = {}
_NAMED_LOCK_REFCOUNTS: Dict[str, int] = {}
_NAMED_LOCK_FDS: Dict[str, int] = {}
_NAMED_LOCK_REGISTRY_MUTEX = threading.Lock()


def _sanitize_drive_error(raw: Union[str, bytes], url: str = "") -> str:
    """Sanitize error messages and URLs by redacting tokens, secrets, and session URLs."""
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    elif isinstance(raw, str):
        text = raw
    else:
        text = str(raw)

    if url:
        sanitized_url = re.sub(
            r"https://www\.googleapis\.com/upload/[^\s\"'>]+",
            "[REDACTED_UPLOAD_SESSION_URL]",
            str(url),
        )
        if str(url) in text:
            text = text.replace(str(url), sanitized_url)

    # 1. Full resumable upload session URLs
    text = re.sub(
        r"https://www\.googleapis\.com/upload/[^\s\"'>]+",
        "[REDACTED_UPLOAD_SESSION_URL]",
        text,
    )

    # 2. Bearer authorization tokens
    text = re.sub(
        r"Bearer\s+[A-Za-z0-9._\-]+",
        "Bearer [REDACTED]",
        text,
        flags=re.IGNORECASE,
    )

    # 3. session_id and upload_id query parameters
    text = re.sub(r"session_id=[^&\s\"'>]+", "session_id=[REDACTED]", text)
    text = re.sub(r"upload_id=[^&\s\"'>]+", "upload_id=[REDACTED]", text)

    # 4. JSON or quoted key-values for secrets
    text = re.sub(
        r'(["\']?(?:access_token|refresh_token|client_secret|id_token)["\']?\s*[:=]\s*)["\'][^"\']*["\']',
        r'\1"[REDACTED]"',
        text,
        flags=re.IGNORECASE,
    )

    # 5. Form-encoded, query string, or non-quoted key-values for secrets
    text = re.sub(
        r'(["\']?(?:access_token|refresh_token|client_secret|id_token)["\']?\s*[:=]\s*)[^\s&,;\'"}]+',
        r'\1[REDACTED]',
        text,
        flags=re.IGNORECASE,
    )

    # 6. Unquoted or whitespace-separated key value pairs: e.g. client_secret 'secret' or client_secret secret
    text = re.sub(
        r'(\b(?:access_token|refresh_token|client_secret|id_token)\s+[\'"]?)[^\s&,;\'"}]+([\'"]?)',
        r'\1[REDACTED]\2',
        text,
        flags=re.IGNORECASE,
    )

    return text


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
        list_page_size: int = 100,
        max_list_pages: int = 50,
        max_list_candidates: int = 5000,
    ):
        self.credentials = credentials
        self.credentials_file = Path(credentials_file).resolve() if credentials_file else None
        self.root_folder_name = root_folder_name
        self.staging_dir = Path(local_staging_dir).resolve()
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        if chunk_size <= 0 or chunk_size % (256 * 1024) != 0:
            raise ValueError(f"chunk_size must be a positive multiple of 256 KiB (262144 bytes), got {chunk_size}")
        self.chunk_size = chunk_size

        if list_page_size <= 0:
            raise ValueError(f"list_page_size must be positive (>= 1), got {list_page_size}")
        if max_list_pages <= 0:
            raise ValueError(f"max_list_pages must be positive (>= 1), got {max_list_pages}")
        if max_list_candidates <= 0:
            raise ValueError(f"max_list_candidates must be positive (>= 1), got {max_list_candidates}")

        self.list_page_size = list_page_size
        self.max_list_pages = max_list_pages
        self.max_list_candidates = max_list_candidates

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

    def compute_sha256(self, file_path: Union[str, Path]) -> str:
        """Compute streaming SHA256 of local file."""
        return compute_file_sha256(file_path)

    def _compute_md5(self, file_path: Union[str, Path]) -> str:
        """Compute streaming MD5 of local file for Drive v3 comparison."""
        return compute_file_md5(file_path)

    @contextlib.contextmanager
    def _exclusive_lock(self, lock_name: str, timeout: float = 60.0):
        """Acquire a named cross-process and cross-thread exclusive lock.

        Delegates to self.journal.acquire_named_lock if supported by the journal,
        otherwise uses a re-entrant thread lock combined with fcntl.flock on
        .transfer_locks/<safe_lock_name>.lock.
        """
        if hasattr(self.journal, "acquire_named_lock"):
            with self.journal.acquire_named_lock(lock_name, timeout=timeout):
                yield
            return

        safe_lock_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", lock_name)
        locks_dir = self.journal.path.parent / ".transfer_locks"
        locks_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(str(locks_dir), 0o700)
        except OSError:
            pass

        lock_file = locks_dir / f"{safe_lock_name}.lock"
        key = str(lock_file.resolve())

        with _NAMED_LOCK_REGISTRY_MUTEX:
            if key not in _NAMED_LOCK_REGISTRY:
                _NAMED_LOCK_REGISTRY[key] = threading.RLock()
            rlock = _NAMED_LOCK_REGISTRY[key]

        start_time = time.time()
        if not rlock.acquire(timeout=timeout):
            raise JournalLockError(f"Thread lock acquisition timed out after {timeout}s for {lock_name}")

        try:
            with _NAMED_LOCK_REGISTRY_MUTEX:
                ref_count = _NAMED_LOCK_REFCOUNTS.get(key, 0)
                if ref_count == 0:
                    fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT, 0o600)
                    try:
                        os.chmod(str(lock_file), 0o600)
                    except OSError:
                        pass
                    if fcntl is not None:
                        flock_acquired = False
                        while (time.time() - start_time) < timeout:
                            try:
                                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                flock_acquired = True
                                break
                            except (BlockingIOError, OSError):
                                time.sleep(0.02)
                        if not flock_acquired:
                            os.close(fd)
                            raise JournalLockError(
                                f"Cross-process file lock acquisition timed out after {timeout}s for {lock_name}"
                            )
                    _NAMED_LOCK_FDS[key] = fd
                _NAMED_LOCK_REFCOUNTS[key] = ref_count + 1

            yield

        finally:
            with _NAMED_LOCK_REGISTRY_MUTEX:
                if key in _NAMED_LOCK_REFCOUNTS:
                    _NAMED_LOCK_REFCOUNTS[key] -= 1
                    if _NAMED_LOCK_REFCOUNTS[key] == 0:
                        del _NAMED_LOCK_REFCOUNTS[key]
                        open_fd = _NAMED_LOCK_FDS.pop(key, None)
                        if open_fd is not None:
                            if fcntl is not None:
                                try:
                                    fcntl.flock(open_fd, fcntl.LOCK_UN)
                                except OSError:
                                    pass
                            try:
                                os.close(open_fd)
                            except OSError:
                                pass
            rlock.release()

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
            sanitized = _sanitize_drive_error(body, self.credentials.token_uri)
            raise DriveAuthenticationError(
                f"Failed to refresh Drive access token (HTTP {status}): {sanitized}"
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
                    raise DriveAuthenticationError(
                        f"Drive authentication rejected (401): {_sanitize_drive_error(body, url)}"
                    )

                if status == 403:
                    body_text = body.decode("utf-8", errors="replace")
                    sanitized_body = _sanitize_drive_error(body, url)
                    if "rateLimitExceeded" in body_text or "userRateLimitExceeded" in body_text:
                        # Rate limit retry
                        pass
                    elif "quotaExceeded" in body_text or "storageQuotaExceeded" in body_text:
                        raise DriveQuotaExceededError(f"Drive storage quota exceeded: {sanitized_body}")
                    else:
                        raise DriveAuthenticationError(f"Drive access forbidden (403): {sanitized_body}")

                if status in (429, 500, 502, 503, 504):
                    if attempt == self.max_retries:
                        raise DriveTransferError(
                            f"Drive request failed after {self.max_retries} retries (HTTP {status}): {_sanitize_drive_error(body, url)}"
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
                    raise DriveTransferError(
                        f"Drive request failed after {self.max_retries} retries: {_sanitize_drive_error(str(e), url)}"
                    ) from e
                sleep_time = backoff * (1.0 + random.uniform(0.1, 0.5))
                time.sleep(sleep_time)
                backoff *= 2.0

        raise DriveTransferError(
            f"Drive request retry loop exhausted: {_sanitize_drive_error(str(last_error), url)}"
        )

    def _find_or_create_folder(self, folder_name: str, parent_id: Optional[str] = None) -> str:
        """Find existing folder by name and parent or create it safely."""
        cache_key = f"{parent_id or 'root'}::{folder_name}"
        if cache_key in self._folder_cache:
            return self._folder_cache[cache_key]

        # Query existing folder
        safe_folder_name = folder_name.replace("'", "\\'")
        q = f"name = '{safe_folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        if parent_id:
            safe_parent = parent_id.replace("'", "\\'")
            q += f" and '{safe_parent}' in parents"
        else:
            q += " and 'root' in parents"

        url = f"https://www.googleapis.com/drive/v3/files?q={urllib.parse.quote(q)}&fields=files(id,name)&spaces=drive"
        try:
            status, headers, body = self._request_with_retry("GET", url)
        except (DriveAuthenticationError, DriveQuotaExceededError, DriveTransferError):
            raise
        except Exception as e:
            raise DriveTransferError(
                f"Failed to query Drive folder '{folder_name}': {_sanitize_drive_error(str(e), url)}"
            ) from e

        if status == 401:
            raise DriveAuthenticationError(
                f"Drive folder lookup unauthorized (401): {_sanitize_drive_error(body, url)}"
            )
        if status == 403:
            body_text = body.decode("utf-8", errors="replace")
            if any(k in body_text for k in ("quotaExceeded", "storageQuotaExceeded", "rateLimitExceeded", "userRateLimitExceeded")):
                raise DriveQuotaExceededError(
                    f"Drive storage quota exceeded during folder lookup (403): {_sanitize_drive_error(body, url)}"
                )
            raise DriveAuthenticationError(
                f"Drive permission denied during folder lookup (403): {_sanitize_drive_error(body, url)}"
            )
        if status != 200:
            raise DriveTransferError(
                f"Failed to query Drive folder '{folder_name}' (HTTP {status}): {_sanitize_drive_error(body, url)}"
            )

        try:
            data = json.loads(body.decode("utf-8"))
        except Exception as je:
            raise DriveTransferError(
                f"Malformed JSON in Drive folder lookup response: {_sanitize_drive_error(body, url)}"
            ) from je

        if not isinstance(data, dict) or "files" not in data or not isinstance(data["files"], list):
            raise DriveTransferError(
                f"Invalid Drive folder lookup payload structure (missing 'files' list): {_sanitize_drive_error(body, url)}"
            )

        files = data["files"]
        if files:
            first_folder = files[0]
            if not isinstance(first_folder, dict) or not first_folder.get("id") or not isinstance(first_folder.get("id"), str):
                raise DriveTransferError(
                    f"Invalid folder entry in Drive lookup response (missing string 'id'): {_sanitize_drive_error(body, url)}"
                )
            folder_id = first_folder["id"]
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
        try:
            status, headers, body = self._request_with_retry(
                "POST",
                create_url,
                headers={"Content-Type": "application/json; charset=UTF-8"},
                data=payload,
            )
        except (DriveAuthenticationError, DriveQuotaExceededError, DriveTransferError):
            raise
        except Exception as e:
            raise DriveTransferError(
                f"Failed to create Drive folder '{folder_name}': {_sanitize_drive_error(str(e), create_url)}"
            ) from e

        if status not in (200, 201):
            raise DriveTransferError(
                f"Failed to create Drive folder '{folder_name}' (HTTP {status}): {_sanitize_drive_error(body, create_url)}"
            )

        try:
            res = json.loads(body.decode("utf-8"))
        except Exception as je:
            raise DriveTransferError(
                f"Malformed JSON in Drive folder creation response: {_sanitize_drive_error(body, create_url)}"
            ) from je

        if not isinstance(res, dict) or not res.get("id") or not isinstance(res.get("id"), str):
            raise DriveTransferError(
                f"Invalid folder creation response (missing string 'id'): {_sanitize_drive_error(body, create_url)}"
            )

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

    def _list_files_in_folder(
        self,
        filename: str,
        folder_id: str = "",
        parent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query for all non-trashed files with matching name in folder across pages."""
        target_folder = parent_id if parent_id is not None else folder_id
        safe_name = filename.replace("'", "\\'")
        safe_folder = target_folder.replace("'", "\\'")
        query = f"name = '{safe_name}' and trashed = false and '{safe_folder}' in parents"
        fields = "nextPageToken,files(id,name,size,md5Checksum,appProperties,description,createdTime)"

        collected_files: List[Dict[str, Any]] = []
        seen_file_ids: set[str] = set()
        seen_tokens: set[str] = set()
        page_token: Optional[str] = None
        pages_fetched = 0

        while True:
            params = [
                f"q={urllib.parse.quote(query)}",
                f"fields={fields}",
                f"pageSize={self.list_page_size}",
            ]
            if page_token is not None:
                params.append(f"pageToken={urllib.parse.quote(page_token, safe='')}")

            url = f"https://www.googleapis.com/drive/v3/files?{'&'.join(params)}"
            try:
                status, _, body = self._request_with_retry("GET", url)
            except (DriveAuthenticationError, DriveQuotaExceededError, DriveTransferError):
                raise
            except Exception as e:
                raise DriveTransferError(
                    f"Drive file listing failed for '{filename}': {_sanitize_drive_error(str(e), url)}"
                ) from e

            if status == 401:
                raise DriveAuthenticationError(
                    f"Drive file listing unauthorized (401): {_sanitize_drive_error(body, url)}"
                )
            if status == 403:
                body_text = body.decode("utf-8", errors="replace")
                if any(k in body_text for k in ("quotaExceeded", "storageQuotaExceeded", "rateLimitExceeded", "userRateLimitExceeded")):
                    raise DriveQuotaExceededError(
                        f"Drive storage quota exceeded during file listing: {_sanitize_drive_error(body, url)}"
                    )
                raise DriveAuthenticationError(
                    f"Drive access forbidden during file listing: {_sanitize_drive_error(body, url)}"
                )
            if status == 404:
                raise ArtifactNotFoundError(
                    f"Drive parent folder '{target_folder}' not found (404): {_sanitize_drive_error(body, url)}"
                )
            if status != 200:
                raise DriveTransferError(
                    f"Drive file listing failed with HTTP {status}: {_sanitize_drive_error(body, url)}"
                )

            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as je:
                raise DriveTransferError(
                    f"Malformed JSON in Drive file listing response: {_sanitize_drive_error(body, url)}"
                ) from je

            if not isinstance(data, dict) or "files" not in data or not isinstance(data["files"], list):
                raise DriveTransferError(
                    f"Invalid Drive file listing payload structure (missing 'files' list): {_sanitize_drive_error(body, url)}"
                )

            pages_fetched += 1

            for item in data["files"]:
                if not isinstance(item, dict) or not item.get("id") or not isinstance(item.get("id"), str):
                    raise DriveTransferError(
                        "Invalid file entry in Drive listing response (missing string 'id')"
                    )
                file_id = item["id"]
                if file_id not in seen_file_ids:
                    seen_file_ids.add(file_id)
                    collected_files.append(item)
                    if len(collected_files) > self.max_list_candidates:
                        raise DriveTransferError(
                            f"Drive file listing exceeded max_list_candidates ({self.max_list_candidates}) for '{filename}'"
                        )

            next_token = data.get("nextPageToken")
            if next_token is None or next_token == "":
                break
            if not isinstance(next_token, str) or not next_token.strip():
                raise DriveTransferError(
                    "Malformed non-string or whitespace nextPageToken in Drive file listing"
                )
            if next_token in seen_tokens:
                raise DriveTransferError(
                    f"Detected repeating nextPageToken loop in Drive file listing: {next_token!r}"
                )
            seen_tokens.add(next_token)

            if pages_fetched >= self.max_list_pages:
                raise DriveTransferError(
                    f"Drive file listing exceeded max_list_pages ({self.max_list_pages}) while nextPageToken is still present"
                )

            page_token = next_token

        return collected_files

    def _find_file_in_folder(
        self,
        filename: str,
        folder_id: str,
        expected_size: Optional[int] = None,
        expected_md5: Optional[str] = None,
        allow_revision_mismatch: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Query for existing non-trashed file with matching name in folder to prevent duplicates."""
        files = self._list_files_in_folder(filename, folder_id)
        if not files:
            return None

        mismatch_reason = None
        for f in files:
            f_sz = f.get("size")
            if expected_size is not None and f_sz is not None and int(f_sz) != expected_size:
                mismatch_reason = f"Remote file '{filename}' size mismatch: expected {expected_size}, remote has {f_sz}"
                continue
            f_md5 = f.get("md5Checksum")
            if expected_md5 is not None and f_md5 and f_md5.lower() != expected_md5.lower():
                mismatch_reason = f"Remote file '{filename}' checksum mismatch: expected MD5 {expected_md5}, remote has {f_md5}"
                continue
            return f

        if mismatch_reason is not None:
            if allow_revision_mismatch:
                return None
            raise StorageError(mismatch_reason)

        return None

    def _initiate_resumable_upload(
        self,
        filename: str,
        parent_id: str,
        total_size: int,
        content_type: str = "application/octet-stream",
        app_properties: Optional[Dict[str, str]] = None,
    ) -> str:
        """Initiate resumable upload session and return the upload URL."""
        if app_properties:
            validate_drive_properties(app_properties)
        token = self._ensure_token()
        init_url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&fields=id,name,size,md5Checksum,appProperties"
        metadata: Dict[str, Any] = {
            "name": filename,
            "parents": [parent_id],
        }
        if app_properties:
            metadata["appProperties"] = app_properties
        meta_bytes = json.dumps(metadata).encode("utf-8")
        hdrs = {
            "authorization": f"Bearer {token}",
            "content-type": "application/json; charset=UTF-8",
            "x-upload-content-type": content_type,
            "x-upload-content-length": str(total_size),
        }

        status, resp_headers, body = self.http_client.request("POST", init_url, headers=hdrs, data=meta_bytes)
        if status != 200:
            body_text = body.decode("utf-8", errors="replace")
            sanitized = _sanitize_drive_error(body, init_url)
            if status == 401:
                raise DriveAuthenticationError(
                    f"Failed to initiate resumable upload for '{filename}' (HTTP 401): {sanitized}"
                )
            if status == 403:
                if "quotaExceeded" in body_text or "storageQuotaExceeded" in body_text:
                    raise DriveQuotaExceededError(f"Drive storage quota exceeded: {sanitized}")
                raise DriveAuthenticationError(f"Drive access forbidden (403): {sanitized}")
            raise DriveTransferError(
                f"Failed to initiate resumable upload for '{filename}' (HTTP {status}): {sanitized}"
            )

        location = resp_headers.get("location")
        if not location:
            raise DriveTransferError(f"Drive API did not return Location header for resumable upload of '{filename}'")
        return location

    def _query_upload_progress(self, session_url: str, total_size: int) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Query Drive for received byte offset or completed upload response."""
        backoff = self.initial_backoff_sec
        for attempt in range(self.max_retries + 1):
            token = self._ensure_token()
            hdrs = {
                "content-range": f"bytes */{total_size}",
                "content-length": "0",
                "authorization": f"Bearer {token}",
            }
            try:
                status, resp_headers, body = self.http_client.request("PUT", session_url, headers=hdrs)
            except (DriveAuthenticationError, DriveQuotaExceededError, DriveTransferError):
                raise
            except Exception as e:
                if attempt == self.max_retries:
                    raise DriveTransferError(
                        f"Upload progress query failed after {self.max_retries} retries: {_sanitize_drive_error(str(e), session_url)}"
                    ) from e
                sleep_time = backoff * (1.0 + random.uniform(0.1, 0.5))
                time.sleep(sleep_time)
                backoff *= 2.0
                continue

            if status == 308:
                range_hdr = resp_headers.get("range", "").strip()
                if not range_hdr:
                    return 0, None
                m = re.match(r"^bytes=0-(\d+)$", range_hdr)
                if not m:
                    raise DriveTransferError(f"Malformed Range header in status query: {range_hdr!r}")
                next_offset = int(m.group(1)) + 1
                if next_offset > total_size:
                    raise DriveTransferError(f"Impossible Range offset {next_offset} > {total_size} in status query")
                return next_offset, None

            elif status in (200, 201):
                try:
                    res = json.loads(body.decode("utf-8"))
                    if not isinstance(res, dict) or "id" not in res:
                        raise ValueError("Missing 'id' or not a dictionary")
                    return total_size, res
                except Exception as je:
                    raise DriveTransferError("Invalid or incomplete JSON response on 200/201 upload completion") from je

            elif status in (404, 410):
                return -1, None

            elif status == 401:
                if attempt < self.max_retries and self.credentials and self.credentials.refresh_token:
                    self._refresh_access_token()
                    continue
                raise DriveAuthenticationError(
                    f"Upload progress query unauthorized (HTTP 401): {_sanitize_drive_error(body, session_url)}"
                )

            elif status == 403:
                body_text = body.decode("utf-8", errors="replace")
                if "rateLimitExceeded" in body_text or "userRateLimitExceeded" in body_text:
                    if attempt == self.max_retries:
                        raise DriveTransferError(
                            f"Upload progress query rate limit exceeded after retries (HTTP 403): {_sanitize_drive_error(body, session_url)}"
                        )
                    sleep_time = backoff * (1.0 + random.uniform(0.1, 0.5))
                    time.sleep(sleep_time)
                    backoff *= 2.0
                    continue
                elif "quotaExceeded" in body_text or "storageQuotaExceeded" in body_text:
                    raise DriveQuotaExceededError(
                        f"Drive storage quota exceeded: {_sanitize_drive_error(body, session_url)}"
                    )
                else:
                    raise DriveAuthenticationError(
                        f"Drive access forbidden (403): {_sanitize_drive_error(body, session_url)}"
                    )

            elif status in (429, 500, 502, 503, 504):
                if attempt == self.max_retries:
                    raise DriveTransferError(f"Upload progress query failed after retries (HTTP {status})")
                sleep_time = backoff * (1.0 + random.uniform(0.1, 0.5))
                time.sleep(sleep_time)
                backoff *= 2.0
                continue

            else:
                raise DriveTransferError(
                    f"Unexpected HTTP {status} during upload progress query: {_sanitize_drive_error(body, session_url)}"
                )

        raise DriveTransferError(
            f"Upload progress query retry loop exhausted for session: {_sanitize_drive_error(session_url)}"
        )

    def _handle_expired_session(
        self,
        filename: str,
        target_folder_id: str,
        total_size: int,
        local_md5: str,
        transfer_record: TransferRecord,
        app_properties: Optional[Dict[str, str]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], str, int]:
        """Recover from expired upload session (HTTP 404/410)."""
        transfer_record.resumable_session_url = None
        self.journal.record_transfer(transfer_record)

        existing = self._find_file_in_folder(
            filename,
            target_folder_id,
            expected_size=total_size,
            expected_md5=local_md5,
            allow_revision_mismatch=True,
        )
        if existing and existing.get("id"):
            existing_id = existing.get("id")
            is_diff = False
            target_sha = transfer_record.sha256 if transfer_record else None
            if target_sha:
                for rec in self.journal.list_transfers(project_id=transfer_record.project_id, run_id=transfer_record.run_id):
                    if rec.remote_file_id == existing_id and rec.sha256 and rec.sha256.lower() != target_sha.lower():
                        is_diff = True
                        break
            if is_diff:
                existing = None
            elif target_sha:
                try:
                    self._verify_remote_file_content(existing_id, total_size, target_sha)
                except Exception:
                    existing = None

        if existing and existing.get("id"):
            transfer_record.bytes_transferred = total_size
            transfer_record.remote_file_id = existing.get("id")
            self.journal.record_transfer(transfer_record)
            return existing, "", total_size

        if app_properties is None and transfer_record and transfer_record.sha256:
            target_rel = transfer_record.target_rel_path or filename or ""
            app_properties = build_revision_properties(
                sha256=transfer_record.sha256,
                project_id=transfer_record.project_id or "",
                run_id=transfer_record.run_id or "",
                logical_path=target_rel,
                is_latest=False,
            )

        new_session_url = self._initiate_resumable_upload(
            filename, target_folder_id, total_size, app_properties=app_properties
        )
        transfer_record.resumable_session_url = new_session_url
        transfer_record.bytes_transferred = 0
        self.journal.record_transfer(transfer_record)
        return None, new_session_url, 0

    def upload_file_resumable(
        self,
        local_path: Union[str, Path],
        target_folder_id: str,
        transfer_record: TransferRecord,
        filename: Optional[str] = None,
        app_properties: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Perform chunked resumable upload with state updates in transfer journal."""
        src = Path(local_path).resolve()
        total_size = src.stat().st_size
        if filename is None:
            if transfer_record and transfer_record.target_rel_path:
                filename = Path(transfer_record.target_rel_path).name
            elif transfer_record and transfer_record.local_path:
                filename = Path(transfer_record.local_path).name
            else:
                filename = src.name
        local_md5 = transfer_record.md5 or compute_file_md5(src)
        if not transfer_record.md5:
            transfer_record.md5 = local_md5

        if app_properties is None and transfer_record and transfer_record.sha256:
            target_rel = transfer_record.target_rel_path or filename or ""
            app_properties = build_revision_properties(
                sha256=transfer_record.sha256,
                project_id=transfer_record.project_id or "",
                run_id=transfer_record.run_id or "",
                logical_path=target_rel,
                is_latest=False,
            )

        # Check if identical file already exists in target folder (idempotency)
        existing = self._find_file_in_folder(
            filename,
            target_folder_id,
            expected_size=total_size,
            expected_md5=local_md5,
            allow_revision_mismatch=True,
        )
        if existing and existing.get("id"):
            existing_id = existing.get("id")
            is_diff = False
            target_sha = transfer_record.sha256 if transfer_record else None
            if target_sha:
                for rec in self.journal.list_transfers(project_id=transfer_record.project_id, run_id=transfer_record.run_id):
                    if rec.remote_file_id == existing_id and rec.sha256 and rec.sha256.lower() != target_sha.lower():
                        is_diff = True
                        break
            if is_diff:
                existing = None
            elif target_sha:
                try:
                    self._verify_remote_file_content(existing_id, total_size, target_sha)
                except Exception:
                    existing = None

        if existing and existing.get("id"):
            transfer_record.bytes_transferred = total_size
            transfer_record.remote_file_id = existing.get("id")
            self.journal.record_transfer(transfer_record)
            return existing

        session_url = transfer_record.resumable_session_url
        if not session_url:
            session_url = self._initiate_resumable_upload(
                filename, target_folder_id, total_size, app_properties=app_properties
            )
            transfer_record.resumable_session_url = session_url
            self.journal.record_transfer(transfer_record)

        uploaded_bytes = 0
        if transfer_record.bytes_transferred > 0:
            synced_offset, completed_meta = self._query_upload_progress(session_url, total_size)
            if completed_meta:
                transfer_record.bytes_transferred = total_size
                transfer_record.remote_file_id = completed_meta.get("id")
                self.journal.record_transfer(transfer_record)
                return completed_meta
            if synced_offset == -1:
                existing_rec, session_url, uploaded_bytes = self._handle_expired_session(
                    filename, target_folder_id, total_size, local_md5, transfer_record, app_properties=app_properties
                )
                if existing_rec:
                    return existing_rec
            else:
                uploaded_bytes = synced_offset
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

                chunk_backoff = self.initial_backoff_sec
                for attempt in range(self.max_retries + 1):
                    token = self._ensure_token()
                    hdrs = {
                        "content-range": f"bytes {start_byte}-{end_byte}/{total_size}",
                        "content-length": str(chunk_len),
                        "content-type": "application/octet-stream",
                        "authorization": f"Bearer {token}",
                    }

                    try:
                        status, resp_headers, body = self.http_client.request("PUT", session_url, headers=hdrs, data=chunk)
                    except (DriveAuthenticationError, DriveQuotaExceededError, DriveTransferError):
                        raise
                    except Exception as ce:
                        # Ambiguous network failure: query server to reconcile committed offset
                        try:
                            synced_offset, final_res = self._query_upload_progress(session_url, total_size)
                        except Exception:
                            synced_offset, final_res = 0, None

                        if final_res:
                            transfer_record.bytes_transferred = total_size
                            transfer_record.remote_file_id = final_res.get("id")
                            self.journal.record_transfer(transfer_record)
                            return final_res
                        if synced_offset == -1:
                            existing_rec, session_url, uploaded_bytes = self._handle_expired_session(
                                filename, target_folder_id, total_size, local_md5, transfer_record, app_properties=app_properties
                            )
                            if existing_rec:
                                return existing_rec
                            break
                        if synced_offset > uploaded_bytes:
                            uploaded_bytes = synced_offset
                            transfer_record.bytes_transferred = uploaded_bytes
                            self.journal.record_transfer(transfer_record)
                            break

                        if attempt == self.max_retries:
                            raise DriveTransferError(
                                f"Chunk upload network failure after {self.max_retries} retries: {_sanitize_drive_error(str(ce), session_url)}"
                            ) from ce
                        time.sleep(chunk_backoff * (1.0 + random.uniform(0.1, 0.4)))
                        chunk_backoff *= 2.0
                        continue

                    if status == 308:
                        range_hdr = resp_headers.get("range", "").strip()
                        if range_hdr:
                            m = re.match(r"^bytes=0-(\d+)$", range_hdr)
                            if not m:
                                raise DriveTransferError(f"Malformed HTTP 308 Range header: {range_hdr!r}")
                            confirmed_next = int(m.group(1)) + 1
                            if not (0 < confirmed_next <= start_byte + chunk_len and confirmed_next <= total_size and confirmed_next >= start_byte):
                                raise DriveTransferError(
                                    f"Impossible HTTP 308 Range offset {confirmed_next} for chunk {start_byte}-{end_byte}/{total_size}"
                                )
                            uploaded_bytes = confirmed_next
                            transfer_record.bytes_transferred = uploaded_bytes
                            self.journal.record_transfer(transfer_record)
                            break
                        else:
                            # range_hdr is empty/missing on status == 308
                            synced_offset, final_res = self._query_upload_progress(session_url, total_size)
                            if final_res is not None:
                                transfer_record.bytes_transferred = total_size
                                transfer_record.remote_file_id = final_res.get("id")
                                self.journal.record_transfer(transfer_record)
                                return final_res
                            if synced_offset < 0:
                                # Expired session sentinel -1
                                existing_rec, session_url, uploaded_bytes = self._handle_expired_session(
                                    filename, target_folder_id, total_size, local_md5, transfer_record, app_properties=app_properties
                                )
                                if existing_rec:
                                    return existing_rec
                                break
                            if synced_offset > total_size or synced_offset > start_byte + chunk_len:
                                raise DriveTransferError(f"Impossible reconciled upload offset {synced_offset}")
                            if synced_offset > start_byte:
                                uploaded_bytes = synced_offset
                                transfer_record.bytes_transferred = uploaded_bytes
                                self.journal.record_transfer(transfer_record)
                                break
                            else:
                                uploaded_bytes = synced_offset
                                transfer_record.bytes_transferred = uploaded_bytes
                                self.journal.record_transfer(transfer_record)
                                if attempt == self.max_retries:
                                    raise DriveTransferError(
                                        f"HTTP 308 missing Range header and server confirmed 0 new bytes after {self.max_retries} retries"
                                    )
                                time.sleep(chunk_backoff * (1.0 + random.uniform(0.1, 0.4)))
                                chunk_backoff *= 2.0
                                continue

                    elif status in (200, 201):
                        try:
                            res = json.loads(body.decode("utf-8"))
                            if not isinstance(res, dict) or "id" not in res:
                                raise ValueError("Missing 'id' or not a dictionary")
                        except Exception as je:
                            raise DriveTransferError(
                                f"Invalid or incomplete JSON response on 200/201 upload completion: {_sanitize_drive_error(body, session_url)}"
                            ) from je
                        transfer_record.bytes_transferred = total_size
                        transfer_record.remote_file_id = res.get("id")
                        self.journal.record_transfer(transfer_record)
                        return res

                    elif status in (404, 410):
                        existing_rec, session_url, uploaded_bytes = self._handle_expired_session(
                            filename, target_folder_id, total_size, local_md5, transfer_record, app_properties=app_properties
                        )
                        if existing_rec:
                            return existing_rec
                        break

                    elif status == 401:
                        if attempt < self.max_retries and self.credentials and self.credentials.refresh_token:
                            token = self._refresh_access_token()
                            continue
                        raise DriveAuthenticationError(
                            f"Resumable upload unauthorized (HTTP 401): {_sanitize_drive_error(body, session_url)}"
                        )

                    elif status == 403:
                        body_text = body.decode("utf-8", errors="replace")
                        if "rateLimitExceeded" in body_text or "userRateLimitExceeded" in body_text:
                            if attempt == self.max_retries:
                                error_msg = f"Resumable upload rate limit exceeded after {self.max_retries} retries: {_sanitize_drive_error(body, session_url)}"
                                transfer_record.last_error = error_msg
                                self.journal.record_transfer(transfer_record)
                                raise DriveTransferError(error_msg)
                            time.sleep(chunk_backoff * (1.0 + random.uniform(0.1, 0.4)))
                            chunk_backoff *= 2.0
                            continue
                        elif "quotaExceeded" in body_text or "storageQuotaExceeded" in body_text:
                            error_msg = f"Drive storage quota exceeded: {_sanitize_drive_error(body, session_url)}"
                            transfer_record.last_error = error_msg
                            self.journal.record_transfer(transfer_record)
                            raise DriveQuotaExceededError(error_msg)
                        else:
                            error_msg = f"Drive access forbidden (403): {_sanitize_drive_error(body, session_url)}"
                            transfer_record.last_error = error_msg
                            self.journal.record_transfer(transfer_record)
                            raise DriveAuthenticationError(error_msg)

                    elif status in (429, 500, 502, 503, 504):
                        if attempt == self.max_retries:
                            error_msg = f"Resumable upload chunk failed after {self.max_retries} retries (HTTP {status}): {_sanitize_drive_error(body, session_url)}"
                            transfer_record.last_error = error_msg
                            self.journal.record_transfer(transfer_record)
                            raise DriveTransferError(error_msg)
                        time.sleep(chunk_backoff * (1.0 + random.uniform(0.1, 0.4)))
                        chunk_backoff *= 2.0
                        synced_offset, final_res = self._query_upload_progress(session_url, total_size)
                        if final_res:
                            transfer_record.bytes_transferred = total_size
                            transfer_record.remote_file_id = final_res.get("id")
                            self.journal.record_transfer(transfer_record)
                            return final_res
                        if synced_offset == -1:
                            existing_rec, session_url, uploaded_bytes = self._handle_expired_session(
                                filename, target_folder_id, total_size, local_md5, transfer_record, app_properties=app_properties
                            )
                            if existing_rec:
                                return existing_rec
                            break
                        if synced_offset > uploaded_bytes:
                            uploaded_bytes = synced_offset
                            transfer_record.bytes_transferred = uploaded_bytes
                            self.journal.record_transfer(transfer_record)
                            break
                        continue

                    else:
                        error_msg = f"Resumable upload chunk failed (HTTP {status}): {_sanitize_drive_error(body, session_url)}"
                        transfer_record.last_error = error_msg
                        self.journal.record_transfer(transfer_record)
                        raise DriveTransferError(error_msg)

        # If zero-byte file or final query
        if total_size == 0:
            token = self._ensure_token()
            hdrs = {
                "content-range": "bytes */0",
                "content-length": "0",
                "authorization": f"Bearer {token}",
            }
            status, _, body = self.http_client.request("PUT", session_url, headers=hdrs)
            if status in (200, 201):
                try:
                    res = json.loads(body.decode("utf-8"))
                    if not isinstance(res, dict) or "id" not in res:
                        raise ValueError("Missing 'id' or not a dictionary")
                except Exception as je:
                    raise DriveTransferError(
                        f"Invalid or incomplete JSON response on zero-byte upload completion: {_sanitize_drive_error(body, session_url)}"
                    ) from je
                transfer_record.bytes_transferred = 0
                transfer_record.remote_file_id = res.get("id")
                self.journal.record_transfer(transfer_record)
                return res

        raise DriveTransferError(f"Upload completed without final 200/201 response for '{filename}'")


    def _verify_remote_file_content(self, remote_id: str, expected_size: int, expected_sha256: str) -> None:
        """Verify content of a remote file via media download and SHA256 comparison."""
        readback_url = f"https://www.googleapis.com/drive/v3/files/{remote_id}?alt=media"
        tmp_readback = self.staging_dir / f".readback_{remote_id}_{uuid.uuid4().hex[:8]}.tmp"
        try:
            if hasattr(self.http_client, "stream_download"):
                rb_status, _, rb_bytes = self.http_client.stream_download(
                    readback_url,
                    tmp_readback,
                    headers={"authorization": f"Bearer {self._ensure_token()}"},
                )
                if rb_status != 200:
                    raise DriveIntegrityError(
                        f"Drive file '{remote_id}' missing authoritative md5Checksum and failed content readback verification."
                    )
                rb_sha256 = compute_file_sha256(tmp_readback)
            else:
                rb_status, _, rb_body = self._request_with_retry("GET", readback_url)
                if rb_status != 200:
                    raise DriveIntegrityError(
                        f"Drive file '{remote_id}' missing authoritative md5Checksum and failed content readback verification."
                    )
                rb_bytes = len(rb_body)
                rb_sha256 = hashlib.sha256(rb_body).hexdigest()

            if rb_bytes != expected_size or rb_sha256.lower() != expected_sha256.lower():
                raise DriveIntegrityError(
                    f"Drive file '{remote_id}' missing authoritative md5Checksum and failed content readback verification."
                )
        except DriveIntegrityError:
            raise
        except Exception as rbe:
            raise DriveIntegrityError(
                f"Drive file '{remote_id}' missing authoritative md5Checksum and failed content readback verification."
            ) from rbe
        finally:
            if tmp_readback.exists():
                try:
                    tmp_readback.unlink()
                except OSError:
                    pass

    def _patch_remote_file_app_properties(
        self, remote_id: str, app_properties: Mapping[str, Optional[str]]
    ) -> Dict[str, Any]:
        """Send a retryable PATCH request to update file appProperties on Google Drive."""
        if not isinstance(app_properties, (dict, Mapping)):
            raise DriveTransferError("Drive custom properties must be provided as a mapping")
        for k in app_properties.keys():
            if not isinstance(k, str) or not k:
                raise DriveTransferError("Drive custom property key must not be empty and must be a string")
        non_none_props = {k: v for k, v in app_properties.items() if v is not None}
        validate_drive_properties(non_none_props)

        url = f"https://www.googleapis.com/drive/v3/files/{remote_id}?fields=id,name,appProperties"
        payload = json.dumps({"appProperties": dict(app_properties)}).encode("utf-8")
        status, headers, body = self._request_with_retry(
            "PATCH",
            url,
            headers={"content-type": "application/json"},
            data=payload,
        )
        if status == 401:
            raise DriveAuthenticationError(
                f"Drive authentication rejected (401): {_sanitize_drive_error(body, url)}"
            )
        if status == 403:
            body_text = body.decode("utf-8", errors="replace")
            sanitized = _sanitize_drive_error(body, url)
            if "quotaExceeded" in body_text or "storageQuotaExceeded" in body_text:
                raise DriveQuotaExceededError(f"Drive storage quota exceeded (403): {sanitized}")
            raise DriveAuthenticationError(f"Drive access forbidden (403): {sanitized}")
        if status != 200:
            raise DriveTransferError(
                f"Failed to patch Drive file properties for '{remote_id}' (HTTP {status}): {_sanitize_drive_error(body, url)}"
            )

        try:
            res = json.loads(body.decode("utf-8"))
        except Exception as e:
            raise DriveTransferError(
                f"Failed to parse PATCH response for '{remote_id}': {_sanitize_drive_error(str(e), url)}"
            ) from e

        if not isinstance(res, dict) or res.get("id") != remote_id:
            raise DriveTransferError(
                f"PATCH response missing expected file id '{remote_id}' (got {res.get('id') if isinstance(res, dict) else 'invalid'})."
            )

        res_props = res.get("appProperties") or {}
        for k, v in app_properties.items():
            if v is None:
                if k in res_props:
                    raise DriveTransferError(
                        f"PATCH verification failed for '{remote_id}': property '{k}' expected to be deleted, but still present in response."
                    )
            else:
                if res_props.get(k) != v:
                    raise DriveTransferError(
                        f"PATCH verification failed for '{remote_id}': property '{k}' expected '{v}', got '{res_props.get(k)}'."
                    )

        return res

    def _coordinate_remote_latest_revision(
        self,
        filename: str,
        target_folder_id: str,
        new_remote_id: str,
        new_sha256: str,
        logical_path: str,
        project_id: str,
        run_id: str,
        artifact_id: str,
    ) -> None:
        """Coordinate remote latest revision markers on Drive.

        Called only after new_remote_id has been uploaded and content-verified.
        Demotes all previous revisions in target_folder_id to is_latest='false'
        and sets next_sha256 to new_sha256 (preserving existing next_sha256 if
        already set). Then promotes new_remote_id to is_latest='true'.
        Removes any legacy custom properties present remotely on demoted/promoted files.
        """
        legacy_keys = ("logical_path", "revision_id", "project_id", "run_id", "superseded_by")

        # Step A: List all existing remote files in target_folder_id with name = filename
        files = self._list_files_in_folder(filename, target_folder_id)

        # Step B: For every remote file f where f.get("id") and f.get("id") != new_remote_id
        for f in files:
            f_id = f.get("id")
            if not f_id or f_id == new_remote_id:
                continue
            old_props = dict(f.get("appProperties") or {})

            # ONLY consider files whose props_match_path(...) is not False (never demote other logical paths)
            if props_match_path(old_props, project_id, run_id, logical_path) is False:
                continue

            # For each other file with revision_state == latest
            if revision_state(old_props) == STATE_LATEST:
                old_sha = old_props.get("sha256")
                if not old_sha and old_props.get("revision_id"):
                    m = re.search(r"sha256_([0-9a-fA-F]{64})", old_props["revision_id"])
                    if m:
                        old_sha = m.group(1).lower()

                if not old_sha:
                    raise DriveTransferError(
                        f"Remote file '{f_id}' marked latest lacks sha256 property; cannot demote."
                    )

                # Preserve existing next_sha256 if already set; do not overwrite
                existing_next = old_props.get("next_sha256") or successor_sha256(old_props)
                nxt_sha = existing_next if (existing_next and is_sha256_hex(existing_next)) else new_sha256

                demoted_props: Dict[str, Optional[str]] = dict(
                    build_revision_properties(
                        sha256=old_sha,
                        project_id=project_id,
                        run_id=run_id,
                        logical_path=logical_path,
                        is_latest=False,
                        next_sha256=nxt_sha,
                    )
                )

                # Drop legacy long keys by sending value None ONLY if present remotely
                for k in legacy_keys:
                    if k in old_props:
                        demoted_props[k] = None

                self._patch_remote_file_app_properties(f_id, demoted_props)

        # Step C: Promote new_remote_id via build_revision_properties(new_sha, ..., True)
        new_remote_file = next((f for f in files if f.get("id") == new_remote_id), None)
        new_existing_props = dict(new_remote_file.get("appProperties") or {}) if new_remote_file else {}
        if new_remote_file is None:
            try:
                _, new_existing_props = self._remote_revision_state(new_remote_id)
            except Exception:
                new_existing_props = {}

        promoted_props: Dict[str, Optional[str]] = dict(
            build_revision_properties(
                sha256=new_sha256,
                project_id=project_id,
                run_id=run_id,
                logical_path=logical_path,
                is_latest=True,
            )
        )

        # Drop legacy long keys by sending value None ONLY if present remotely
        for k in legacy_keys:
            if k in new_existing_props:
                promoted_props[k] = None

        self._patch_remote_file_app_properties(new_remote_id, promoted_props)

    def _remote_revision_state(self, remote_id: str) -> Tuple[str, Dict[str, str]]:
        """Fetch remote file appProperties and classify revision state.

        Returns (revision_state(props), props).
        """
        url = f"https://www.googleapis.com/drive/v3/files/{remote_id}?fields=id,name,appProperties"
        try:
            status, _, body = self._request_with_retry("GET", url)
        except (DriveAuthenticationError, DriveQuotaExceededError, DriveTransferError):
            raise
        except Exception as e:
            raise DriveTransferError(
                f"Failed to fetch remote revision state for '{remote_id}': {_sanitize_drive_error(str(e), url)}"
            ) from e

        if status == 401:
            raise DriveAuthenticationError(
                f"Drive authentication rejected (401) fetching revision state for '{remote_id}': {_sanitize_drive_error(body, url)}"
            )
        if status == 403:
            body_text = body.decode("utf-8", errors="replace")
            sanitized = _sanitize_drive_error(body, url)
            if "quotaExceeded" in body_text or "storageQuotaExceeded" in body_text:
                raise DriveQuotaExceededError(f"Drive storage quota exceeded (403): {sanitized}")
            raise DriveAuthenticationError(f"Drive access forbidden (403): {sanitized}")
        if status == 404:
            raise ArtifactNotFoundError(
                f"Remote file '{remote_id}' not found on Google Drive (HTTP 404): {_sanitize_drive_error(body, url)}"
            )
        if status != 200:
            raise DriveTransferError(
                f"Failed to fetch remote revision state for '{remote_id}' (HTTP {status}): {_sanitize_drive_error(body, url)}"
            )

        try:
            res = json.loads(body.decode("utf-8"))
        except Exception as e:
            raise DriveTransferError(
                f"Invalid JSON response fetching revision state for '{remote_id}': {_sanitize_drive_error(str(e), url)}"
            ) from e

        if not isinstance(res, dict):
            raise DriveTransferError(
                f"Invalid response fetching revision state for '{remote_id}': expected JSON object, got {type(res).__name__}"
            )

        props = dict(res.get("appProperties") or {})
        return revision_state(props), props

    def _save_manifest_with_backup(self, manifest: PublicationManifest, local_manifest_path: Path) -> None:
        """Atomically persist manifest and sync 0600 backup."""
        manifest.save(local_manifest_path)
        try:
            os.chmod(local_manifest_path, 0o600)
        except OSError:
            pass

        bak_path = local_manifest_path.with_suffix(".json.bak")
        bak_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_bak = bak_path.parent / f"{bak_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8")
        fd = os.open(str(tmp_bak), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            with os.fdopen(fd, "wb") as f:
                fd = None
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp_bak, 0o600)
            os.replace(str(tmp_bak), str(bak_path))
            os.chmod(bak_path, 0o600)
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if tmp_bak.exists():
                try:
                    tmp_bak.unlink()
                except OSError:
                    pass

    def _load_or_recover_manifest(self, project_id: str, run_id: str) -> Optional[PublicationManifest]:
        """Load manifest for project_id/run_id, recovering from .json.bak if corrupted or missing."""
        m_key = (project_id, run_id)
        local_m = self.staging_dir / "projects" / project_id / "runs" / run_id / "manifest.json"
        bak_m = local_m.with_suffix(".json.bak")

        # 1. Try loading primary manifest
        if local_m.exists():
            try:
                manifest = PublicationManifest.load(local_m)
                self._manifest_cache[m_key] = manifest
                return manifest
            except Exception:
                # Primary manifest corrupted, try backup
                pass

        # 2. Try recovering from backup
        if bak_m.exists():
            try:
                manifest = PublicationManifest.load(bak_m)
                self._save_manifest_with_backup(manifest, local_m)
                self._manifest_cache[m_key] = manifest
                return manifest
            except Exception:
                pass

        # 3. Memory cache fallback
        if m_key in self._manifest_cache:
            return self._manifest_cache[m_key]

        return None

    def _is_trusted_staged_file(self, src: Path) -> bool:
        """Verify that src is an untampered regular file inside a trusted staging directory."""
        if src.is_symlink() or not src.is_file():
            return False

        try:
            if not stat.S_ISREG(src.lstat().st_mode):
                return False
        except OSError:
            return False

        src_resolved = src.resolve()

        staging_roots = [
            (self.staging_dir / ".upload_staging").resolve(),
        ]
        if hasattr(self, "journal") and hasattr(self.journal, "path") and self.journal.path:
            staging_roots.append((self.journal.path.parent / ".staged_uploads").resolve())

        matched_root = None
        for root in staging_roots:
            if not root.exists() or not root.is_dir():
                continue
            try:
                rel = src_resolved.relative_to(root)
                if rel != Path("."):
                    matched_root = root
                    break
            except ValueError:
                continue

        if matched_root is None:
            return False

        try:
            st = matched_root.stat()
            if st.st_uid != os.getuid():
                return False
            if (st.st_mode & 0o777) != 0o700:
                return False
            if matched_root.is_symlink():
                return False
        except OSError:
            return False

        # Verify no parent directory between staging root and src is a symlink
        curr = src_resolved.parent
        while curr != matched_root and curr != curr.parent:
            if curr.is_symlink():
                return False
            curr = curr.parent

        curr_unresolved = src.parent
        while curr_unresolved != curr_unresolved.parent:
            if curr_unresolved.is_symlink():
                return False
            if curr_unresolved == matched_root:
                break
            curr_unresolved = curr_unresolved.parent

        return True

    def _stage_file(self, src: Path, tmp_staging: Path) -> Tuple[int, str]:
        """Stream-copy source file into private 0600 staging file while calculating SHA256 and size."""
        fd = os.open(str(tmp_staging), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600)
        hasher = hashlib.sha256()
        staged_size = 0
        try:
            with open(src, "rb") as f_src, os.fdopen(fd, "wb") as f_dst:
                fd = None
                chunk_sz = 64 * 1024
                while True:
                    chunk = f_src.read(chunk_sz)
                    if not chunk:
                        break
                    f_dst.write(chunk)
                    hasher.update(chunk)
                    staged_size += len(chunk)
                f_dst.flush()
                os.fsync(f_dst.fileno())
            os.chmod(tmp_staging, 0o600)
            return staged_size, hasher.hexdigest()
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _conflicting_latest_requires_reconcile(
        self, files: List[Dict[str, Any]], my_remote_id: str, my_sha256: str
    ) -> bool:
        """Return True if other files also claim latest AND it is safe to reconcile.

        Reconciliation (demote others, keep this one) is only safe when every conflicting
        'latest' file already names THIS revision as its immediate successor (i.e. an interrupted
        demotion of an older revision). Any other contradiction (e.g. a newer unrelated head)
        cannot be resolved from remote metadata alone: fail closed, never silently pick a head.
        """
        conflicts = [
            f for f in files
            if f.get("id") != my_remote_id and (f.get("appProperties") or {}).get("is_latest") == "true"
        ]
        for f in conflicts:
            if successor_sha256(f.get("appProperties")) != my_sha256.lower():
                raise DriveIntegrityError(
                    "Contradictory remote latest markers: another revision claims latest and does not "
                    "name this revision as its successor; refusing to choose a latest revision."
                )
        return bool(conflicts)

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
        p_src = Path(source)
        if p_src.is_symlink() or not p_src.is_file():
            raise ArtifactValidationError(f"Source file must be a regular file, not a symlink or missing: {source}")
        try:
            if not stat.S_ISREG(p_src.lstat().st_mode):
                raise ArtifactValidationError(f"Source file must be a regular file, not a special file: {source}")
        except OSError as e:
            raise ArtifactValidationError(f"Cannot stat source file: {e}") from e
        src = p_src.resolve()

        # Step 1: Verify local SHA256 & MD5
        local_sha256 = self.compute_sha256(src)
        if expected_sha256 and expected_sha256.lower() != local_sha256.lower():
            raise ArtifactValidationError(
                f"Local checksum mismatch before Drive upload: expected {expected_sha256}, got {local_sha256}"
            )
        local_md5 = self._compute_md5(src)
        local_size = src.stat().st_size

        # Resolve subfolder
        subfolder = None
        if artifact_type == ArtifactType.CHECKPOINT.value:
            subfolder = "checkpoints"
        elif artifact_type == ArtifactType.OUTPUT.value:
            subfolder = "outputs"

        rel_path = f"{subfolder}/{src.name}" if subfolder else src.name
        logical_artifact_id = f"{project_id}/{run_id}/{rel_path}"
        artifact_id = f"{logical_artifact_id}@sha256_{local_sha256.lower()}"
        base_transfer_id = f"{self.provider_name}-{project_id}-{run_id}-{src.name}"
        rev_transfer_id = f"{base_transfer_id}-{local_sha256.lower()[:12]}"
        transfer_id = rev_transfer_id
        lock_name = f"artifact-{self.provider_name}-{project_id}-{run_id}-{src.name}"
        manifest_lock_name = f"manifest-{project_id}-{run_id}"
        m_key = (project_id, run_id)

        staged_src = src
        cleanup_staged = False
        is_already_staged = self._is_trusted_staged_file(p_src)

        with self._exclusive_lock(lock_name):
            # Snapshot src into immutable temporary staging file under lock
            if not is_already_staged:
                staging_dir = self.staging_dir / ".upload_staging"
                if staging_dir.is_symlink():
                    raise ArtifactTransferError(f"Staging directory cannot be a symlink: {staging_dir}")
                staging_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                os.chmod(staging_dir, 0o700)

                staging_target = staging_dir / f"{project_id}_{run_id}_{local_sha256[:16]}_{src.name}"
                tmp_staging = staging_dir / f"{project_id}_{run_id}_{local_sha256[:16]}_{src.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"

                try:
                    staged_size, staged_sha = self._stage_file(src, tmp_staging)
                except Exception as e:
                    if tmp_staging.exists():
                        try:
                            tmp_staging.unlink()
                        except OSError:
                            pass
                    raise ArtifactTransferError(f"Failed to snapshot source file for staging: {e}") from e

                if staged_size != local_size or staged_sha.lower() != local_sha256.lower():
                    if tmp_staging.exists():
                        try:
                            tmp_staging.unlink()
                        except OSError:
                            pass
                    raise ArtifactTransferError("Source file mutated during staging before upload")

                os.replace(str(tmp_staging), str(staging_target))
                os.chmod(staging_target, 0o600)
                staged_src = staging_target
                cleanup_staged = True

            try:
                # 1. Reload journal if supported
                if hasattr(self.journal, "_load"):
                    self.journal._load()

                # 2. Load PublicationManifest under manifest lock
                local_manifest_path = self.staging_dir / "projects" / project_id / "runs" / run_id / "manifest.json"
                with self._exclusive_lock(manifest_lock_name):
                    manifest = self._load_or_recover_manifest(project_id, run_id)
                    if manifest is None:
                        manifest = PublicationManifest(project_id=project_id, run_id=run_id)
                    self._manifest_cache[m_key] = manifest
                    self._manifest_cache[m_key] = manifest

                # 3. Check if exact revision is already in manifest with valid remote_file_id
                existing_art = manifest.get_artifact(artifact_id)
                if existing_art and existing_art.remote_file_id and existing_art.storage_provider == self.provider_name:
                    return existing_art

                # 4. Check journal for this exact revision or pre-enqueued in-progress transfer
                record = self.journal.get_transfer(rev_transfer_id)
                if record is not None:
                    transfer_id = record.transfer_id
                else:
                    base_rec = self.journal.get_transfer(base_transfer_id)
                    if base_rec is not None and (
                        base_rec.status != TransferStatus.PUBLISHED.value
                        or (base_rec.sha256 and base_rec.sha256.lower() == local_sha256.lower())
                    ):
                        record = base_rec
                        transfer_id = base_rec.transfer_id
                    else:
                        transfer_id = rev_transfer_id

                if record is None:
                    # Scan journal for any published record for (project_id, run_id) with matching sha256
                    for rec in self.journal.list_transfers(project_id=project_id, run_id=run_id):
                        if rec.status == TransferStatus.PUBLISHED.value and rec.remote_file_id:
                            if rec.sha256 and rec.sha256.lower() == local_sha256.lower():
                                if Path(rec.local_path).name == src.name or rec.target_rel_path == rel_path:
                                    record = rec
                                    transfer_id = rec.transfer_id
                                    break

                if record is not None and record.status == TransferStatus.PUBLISHED.value and record.remote_file_id:
                    target_folder_id = self.resolve_target_folder_id(project_id, run_id, subfolder)
                    # Replay: remote lookup for revision state (fail closed on failure)
                    state, remote_props = self._remote_revision_state(record.remote_file_id)
                    if state == STATE_HISTORICAL:
                        is_latest_publish = False
                    elif state == STATE_LATEST:
                        # Reconcile if another remote file is also marked latest (inconsistent remote state)
                        other_files = self._list_files_in_folder(src.name, target_folder_id)
                        has_conflicting_latest = self._conflicting_latest_requires_reconcile(
                            other_files, record.remote_file_id, local_sha256
                        )
                        if has_conflicting_latest:
                            self._coordinate_remote_latest_revision(
                                src.name,
                                target_folder_id,
                                record.remote_file_id,
                                local_sha256,
                                rel_path,
                                project_id,
                                run_id,
                                artifact_id,
                            )
                        is_latest_publish = True
                    elif state == STATE_PENDING:
                        # Interrupted transaction: coordinate then add latest
                        self._coordinate_remote_latest_revision(
                            src.name,
                            target_folder_id,
                            record.remote_file_id,
                            local_sha256,
                            rel_path,
                            project_id,
                            run_id,
                            artifact_id,
                        )
                        is_latest_publish = True
                    else:
                        raise DriveTransferError(
                            f"Remote file '{record.remote_file_id}' has unrecognized revision state: {state}"
                        )

                    meta_dict = dict(metadata or {})
                    meta_dict["logical_path"] = rel_path
                    meta_dict["revision_id"] = f"sha256_{local_sha256.lower()}"
                    meta_dict["is_latest"] = is_latest_publish

                    stored = StoredArtifact(
                        artifact_id=artifact_id,
                        storage_provider=self.provider_name,
                        path=rel_path,
                        size_bytes=local_size,
                        sha256=local_sha256,
                        project_id=project_id,
                        run_id=run_id,
                        artifact_type=artifact_type,
                        remote_file_id=record.remote_file_id,
                        md5=local_md5,
                        verified_at=_utc_now_iso(),
                        metadata=meta_dict,
                        is_latest=is_latest_publish,
                    )
                    with self._exclusive_lock(manifest_lock_name):
                        manifest = self._load_or_recover_manifest(project_id, run_id)
                        if manifest is None:
                            manifest = PublicationManifest(project_id=project_id, run_id=run_id)
                        manifest.add_artifact(stored)
                        manifest.status = "PUBLISHED"
                        self._save_manifest_with_backup(manifest, local_manifest_path)
                        self._manifest_cache[m_key] = manifest
                    return stored

                # If record is None, also check legacy unversioned transfer_id for in-progress transfers
                if record is None:
                    legacy_tid = f"{self.provider_name}-{project_id}-{run_id}-{src.name}"
                    record = self.journal.get_transfer(legacy_tid)

                # Check if manifest already contains an earlier revision for the SAME logical path with a DIFFERENT sha256
                has_earlier_revision = any(
                    (
                        a.metadata.get("logical_path") == rel_path
                        or a.path == rel_path
                        or a.artifact_id.startswith(f"{logical_artifact_id}@")
                    )
                    and a.sha256.lower() != local_sha256.lower()
                    for a in manifest.list_artifacts()
                )

                is_new_revision = False
                if (
                    record is not None
                    and record.sha256
                    and record.sha256.lower() != local_sha256.lower()
                    and record.status == TransferStatus.PUBLISHED.value
                ):
                    is_new_revision = True

                allow_revision_mismatch = (
                    has_earlier_revision
                    or is_new_revision
                    or (record is None)
                    or (
                        record.status != TransferStatus.UPLOADING.value
                        and record.status != TransferStatus.UPLOAD_PENDING.value
                    )
                )

                # 5. Ensure remote folder hierarchy exists
                target_folder_id = self.resolve_target_folder_id(project_id, run_id, subfolder)

                # 6. Check remote Drive folder for existing matching file
                existing_file = self._find_file_in_folder(
                    src.name,
                    target_folder_id,
                    expected_size=local_size,
                    expected_md5=local_md5,
                    allow_revision_mismatch=allow_revision_mismatch,
                )
                if existing_file:
                    remote_id = existing_file.get("id")
                    if not remote_id:
                        raise DriveIntegrityError("Drive existing file found but has no ID.")

                    # Check whether existing_remote_id is associated with a DIFFERENT sha256 in manifest or journal
                    is_different_sha = False
                    for art in manifest.list_artifacts():
                        if art.remote_file_id == remote_id and art.sha256.lower() != local_sha256.lower():
                            is_different_sha = True
                            break
                    if not is_different_sha:
                        for rec in self.journal.list_transfers(project_id=project_id, run_id=run_id):
                            if rec.remote_file_id == remote_id and rec.sha256 and rec.sha256.lower() != local_sha256.lower():
                                is_different_sha = True
                                break

                    if is_different_sha:
                        existing_file = None
                    else:
                        already_associated = False
                        for art in manifest.list_artifacts():
                            if art.remote_file_id == remote_id and art.sha256.lower() == local_sha256.lower():
                                already_associated = True
                                break
                        if not already_associated:
                            for rec in self.journal.list_transfers(project_id=project_id, run_id=run_id):
                                if (
                                    rec.remote_file_id == remote_id
                                    and rec.sha256
                                    and rec.sha256.lower() == local_sha256.lower()
                                    and rec.status == TransferStatus.PUBLISHED.value
                                ):
                                    already_associated = True
                                    break

                        # If not already associated or missing md5Checksum, verify content via readback
                        if not already_associated or (local_md5 and not existing_file.get("md5Checksum")):
                            try:
                                self._verify_remote_file_content(remote_id, local_size, local_sha256)
                            except Exception:
                                existing_file = None

                if existing_file:
                    remote_id = existing_file.get("id")
                    if not remote_id:
                        raise DriveIntegrityError("Drive existing file found but has no ID.")

                    state, remote_props = self._remote_revision_state(remote_id)
                    if state == STATE_HISTORICAL:
                        is_latest_publish = False
                    elif state == STATE_LATEST:
                        # Reconcile if another remote file is also marked latest (inconsistent remote state)
                        other_files = self._list_files_in_folder(src.name, target_folder_id)
                        has_conflicting_latest = self._conflicting_latest_requires_reconcile(
                            other_files, remote_id, local_sha256
                        )
                        if has_conflicting_latest:
                            self._coordinate_remote_latest_revision(
                                src.name,
                                target_folder_id,
                                remote_id,
                                local_sha256,
                                rel_path,
                                project_id,
                                run_id,
                                artifact_id,
                            )
                        is_latest_publish = True
                    elif state == STATE_PENDING:
                        # Interrupted publish: coordinate (promote) then add latest
                        self._coordinate_remote_latest_revision(
                            src.name,
                            target_folder_id,
                            remote_id,
                            local_sha256,
                            rel_path,
                            project_id,
                            run_id,
                            artifact_id,
                        )
                        is_latest_publish = True
                    else:
                        raise DriveTransferError(
                            f"Remote file '{remote_id}' has unrecognized revision state: {state}"
                        )

                    if record is None or record.transfer_id != transfer_id:
                        record = TransferRecord(
                            transfer_id=transfer_id,
                            project_id=project_id,
                            run_id=run_id,
                            artifact_id=artifact_id,
                            local_path=str(src),
                            target_rel_path=rel_path,
                            size_bytes=local_size,
                            sha256=local_sha256,
                            md5=local_md5,
                            status=TransferStatus.PUBLISHED.value,
                            remote_file_id=remote_id,
                            bytes_transferred=local_size,
                        )
                    else:
                        record.status = TransferStatus.PUBLISHED.value
                        record.remote_file_id = remote_id
                        record.bytes_transferred = local_size
                        record.sha256 = local_sha256
                        record.md5 = local_md5
                        record.size_bytes = local_size
                        record.updated_at = _utc_now_iso()
                    self.journal.record_transfer(record)

                    meta_dict = dict(metadata or {})
                    meta_dict["logical_path"] = rel_path
                    meta_dict["revision_id"] = f"sha256_{local_sha256.lower()}"
                    meta_dict["is_latest"] = is_latest_publish

                    stored = StoredArtifact(
                        artifact_id=artifact_id,
                        storage_provider=self.provider_name,
                        path=rel_path,
                        size_bytes=local_size,
                        sha256=local_sha256,
                        project_id=project_id,
                        run_id=run_id,
                        artifact_type=artifact_type,
                        remote_file_id=remote_id,
                        md5=local_md5,
                        verified_at=_utc_now_iso(),
                        metadata=meta_dict,
                        is_latest=is_latest_publish,
                    )

                    with self._exclusive_lock(manifest_lock_name):
                        manifest = self._load_or_recover_manifest(project_id, run_id)
                        if manifest is None:
                            manifest = PublicationManifest(project_id=project_id, run_id=run_id)
                        existing_art = manifest.get_artifact(artifact_id)
                        if existing_art and existing_art.storage_provider == "local":
                            stored = StoredArtifact(
                                artifact_id=artifact_id,
                                storage_provider=self.provider_name,
                                path=existing_art.path,
                                size_bytes=local_size,
                                sha256=local_sha256,
                                project_id=project_id,
                                run_id=run_id,
                                artifact_type=artifact_type,
                                remote_file_id=remote_id,
                                md5=local_md5,
                                verified_at=_utc_now_iso(),
                                metadata={**existing_art.metadata, **meta_dict},
                                is_latest=is_latest_publish,
                            )

                        manifest.add_artifact(stored)
                        manifest.status = "PUBLISHED"
                        self._save_manifest_with_backup(manifest, local_manifest_path)
                        self._manifest_cache[m_key] = manifest

                    return stored

                # 7. Not found remotely: perform resumable upload from staged_src
                if not record or record.transfer_id != transfer_id:
                    record = TransferRecord(
                        transfer_id=transfer_id,
                        project_id=project_id,
                        run_id=run_id,
                        artifact_id=artifact_id,
                        local_path=str(src),
                        target_rel_path=rel_path,
                        size_bytes=local_size,
                        sha256=local_sha256,
                        md5=local_md5,
                        status=TransferStatus.LOCAL_STAGED.value,
                    )
                    self.journal.record_transfer(record)
                else:
                    if (record.sha256 and record.sha256.lower() != local_sha256.lower()) or record.size_bytes != local_size:
                        record = TransferRecord(
                            transfer_id=transfer_id,
                            project_id=project_id,
                            run_id=run_id,
                            artifact_id=artifact_id,
                            local_path=str(src),
                            target_rel_path=rel_path,
                            size_bytes=local_size,
                            sha256=local_sha256,
                            md5=local_md5,
                            status=TransferStatus.LOCAL_STAGED.value,
                            resumable_session_url=None,
                        )
                        self.journal.record_transfer(record)

                if record.status == TransferStatus.LOCAL_STAGED.value:
                    record.transition_to(TransferStatus.LOCAL_VERIFIED)
                    self.journal.record_transfer(record)

                if record.status == TransferStatus.LOCAL_VERIFIED.value:
                    record.transition_to(TransferStatus.UPLOAD_PENDING)
                    self.journal.record_transfer(record)

                if record.status in (TransferStatus.UPLOAD_PENDING.value, TransferStatus.RETRY_PENDING.value):
                    record.transition_to(TransferStatus.UPLOADING)
                    self.journal.record_transfer(record)
                elif record.status == TransferStatus.UPLOAD_FAILED.value:
                    record.transition_to(TransferStatus.RETRY_PENDING)
                    self.journal.record_transfer(record)
                    record.transition_to(TransferStatus.UPLOADING)
                    self.journal.record_transfer(record)

                drive_app_props = build_revision_properties(
                    sha256=local_sha256,
                    project_id=project_id,
                    run_id=run_id,
                    logical_path=rel_path,
                    is_latest=False,
                )
                drive_file_meta = self.upload_file_resumable(
                    staged_src, target_folder_id, record, app_properties=drive_app_props
                )

                remote_id = drive_file_meta.get("id")
                if not remote_id:
                    raise DriveIntegrityError("Drive upload succeeded but no remote file ID was returned.")

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
                if remote_size != local_size:
                    raise DriveIntegrityError(
                        f"Drive file size mismatch: local {local_size} bytes, remote {remote_size} bytes",
                        details={"local_size": local_size, "remote_size": remote_size},
                    )

                if remote_md5:
                    if remote_md5.lower() != local_md5.lower():
                        raise DriveIntegrityError(
                            f"Drive MD5 mismatch: local {local_md5}, remote {remote_md5}",
                            details={"local_md5": local_md5, "remote_md5": remote_md5},
                        )
                else:
                    self._verify_remote_file_content(remote_id, local_size, local_sha256)

                if record.status != TransferStatus.PUBLISHED.value and record.status != TransferStatus.DRIVE_VERIFIED.value:
                    record.transition_to(TransferStatus.DRIVE_VERIFIED)
                    self.journal.record_transfer(record)

                self._coordinate_remote_latest_revision(
                    src.name,
                    target_folder_id,
                    remote_id,
                    local_sha256,
                    rel_path,
                    project_id,
                    run_id,
                    artifact_id,
                )

                if record.status != TransferStatus.PUBLISHED.value:
                    record.transition_to(TransferStatus.PUBLISHED)
                    record.remote_file_id = remote_id
                    self.journal.record_transfer(record)

                meta_dict = dict(metadata or {})
                meta_dict["logical_path"] = rel_path
                meta_dict["revision_id"] = f"sha256_{local_sha256.lower()}"
                meta_dict["is_latest"] = True

                stored = StoredArtifact(
                    artifact_id=artifact_id,
                    storage_provider=self.provider_name,
                    path=rel_path,
                    size_bytes=local_size,
                    sha256=local_sha256,
                    project_id=project_id,
                    run_id=run_id,
                    artifact_type=artifact_type,
                    remote_file_id=remote_id,
                    md5=local_md5,
                    verified_at=_utc_now_iso(),
                    metadata=meta_dict,
                )

                with self._exclusive_lock(manifest_lock_name):
                    local_manifest_path = self.staging_dir / "projects" / project_id / "runs" / run_id / "manifest.json"
                    manifest = self._load_or_recover_manifest(project_id, run_id)
                    if manifest is None:
                        manifest = PublicationManifest(project_id=project_id, run_id=run_id)

                    existing_art = manifest.get_artifact(artifact_id)
                    if existing_art and existing_art.storage_provider == "local":
                        stored = StoredArtifact(
                            artifact_id=artifact_id,
                            storage_provider=self.provider_name,
                            path=existing_art.path,
                            size_bytes=local_size,
                            sha256=local_sha256,
                            project_id=project_id,
                            run_id=run_id,
                            artifact_type=artifact_type,
                            remote_file_id=remote_id,
                            md5=local_md5,
                            verified_at=_utc_now_iso(),
                            metadata={**existing_art.metadata, **meta_dict},
                        )

                    manifest.add_artifact(stored)
                    manifest.status = "PUBLISHED"
                    self._save_manifest_with_backup(manifest, local_manifest_path)
                    self._manifest_cache[m_key] = manifest

                return stored

            except Exception as e:
                if record is not None:
                    record.retry_count += 1
                    sanitized_err = _sanitize_drive_error(str(e))
                    if record.retry_count <= record.max_retries:
                        try:
                            record.transition_to(TransferStatus.RETRY_PENDING, error_message=sanitized_err)
                        except Exception:
                            record.status = TransferStatus.RETRY_PENDING.value
                            record.last_error = sanitized_err
                    else:
                        try:
                            record.transition_to(TransferStatus.UPLOAD_FAILED, error_message=sanitized_err)
                        except Exception:
                            record.status = TransferStatus.UPLOAD_FAILED.value
                            record.last_error = sanitized_err
                    self.journal.record_transfer(record)
                raise
            finally:
                if cleanup_staged and staged_src.exists():
                    try:
                        staged_src.unlink()
                    except OSError:
                        pass

    def _resolve_artifact_from_manifest(
        self,
        manifest: PublicationManifest,
        artifact_id: str,
    ) -> Optional[StoredArtifact]:
        """Resolve an artifact from manifest by canonical ID, prefix, or unversioned logical ID."""
        # 1. Exact match
        art = manifest.get_artifact(artifact_id)
        if art is not None:
            return art

        # 2. Versioned ID prefix search: e.g. "proj/run/path@sha256_<hex>"
        if "@sha256_" in artifact_id:
            base_id, sha_prefix = artifact_id.split("@sha256_", 1)
            matching = []
            for a in manifest.list_artifacts():
                a_base = a.artifact_id.split("@sha256_")[0]
                if a_base == base_id and a.sha256.lower().startswith(sha_prefix.lower()):
                    matching.append(a)
            if matching:
                matching.sort(key=lambda x: getattr(x, "created_at", "") or "", reverse=True)
                return matching[0]
            return None

        # 3. Unversioned ID search: e.g. "proj/run/checkpoints/model.pt"
        parts = artifact_id.split("/")
        rel_path = "/".join(parts[2:]) if len(parts) > 2 else ""

        matching = []
        for a in manifest.list_artifacts():
            a_base = a.artifact_id.split("@sha256_")[0]
            if a_base == artifact_id or (rel_path and (a.metadata.get("logical_path") == rel_path or a.path == rel_path)):
                matching.append(a)

        if not matching:
            return None

        matching.sort(key=lambda x: getattr(x, "created_at", "") or "", reverse=True)

        for c in matching:
            if c.metadata.get("is_latest") is True or getattr(c, "is_latest", False) is True:
                return c

        return None

    def _download_and_verify(
        self,
        meta: StoredArtifact,
        dest: Path,
        target_sha: Optional[str] = None,
        expected_full_sha: Optional[str] = None,
        allow_unverified_recovery: bool = False,
    ) -> None:
        """Download remote Drive file to private 0600 temp file, verify size/md5/sha256, atomically replace dest."""
        tmp = dest.parent / f"{dest.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        download_url = f"https://www.googleapis.com/drive/v3/files/{meta.remote_file_id}?alt=media"
        downloaded_bytes = 0
        fd = None

        try:
            if hasattr(self.http_client, "stream_download"):
                fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600)
                os.close(fd)
                fd = None

                try:
                    status, _, downloaded_bytes = self.http_client.stream_download(
                        download_url, tmp, headers={"authorization": f"Bearer {self._ensure_token()}"}
                    )
                except (DriveAuthenticationError, DriveQuotaExceededError, DriveTransferError):
                    raise
                except Exception as de:
                    raise DriveTransferError(
                        f"Network failure downloading file '{meta.remote_file_id}': {_sanitize_drive_error(str(de), download_url)}"
                    ) from de

                if status != 200:
                    if status == 401:
                        raise DriveAuthenticationError(f"Unauthorized downloading file '{meta.remote_file_id}'")
                    if status == 403:
                        raise DriveQuotaExceededError(f"Access forbidden downloading file '{meta.remote_file_id}'")
                    raise DriveTransferError(f"Failed to download Drive file '{meta.remote_file_id}' (HTTP {status})")

                if tmp.exists():
                    os.chmod(tmp, 0o600)
                downloaded_sha256 = self.compute_sha256(tmp)
                downloaded_md5 = self._compute_md5(tmp)
            else:
                try:
                    status, headers, body = self._request_with_retry("GET", download_url)
                except (DriveAuthenticationError, DriveQuotaExceededError, DriveTransferError):
                    raise
                except Exception as de:
                    raise DriveTransferError(
                        f"Network failure downloading file '{meta.remote_file_id}': {_sanitize_drive_error(str(de), download_url)}"
                    ) from de

                if status != 200:
                    if status == 401:
                        raise DriveAuthenticationError(f"Unauthorized downloading file '{meta.remote_file_id}'")
                    if status == 403:
                        raise DriveQuotaExceededError(f"Access forbidden downloading file '{meta.remote_file_id}'")
                    raise DriveTransferError(f"Failed to download Drive file '{meta.remote_file_id}' (HTTP {status})")

                fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600)
                hasher = hashlib.sha256()
                md5_hasher = hashlib.md5()
                with os.fdopen(fd, "wb") as f:
                    fd = None
                    chunk_sz = 64 * 1024
                    for i in range(0, len(body), chunk_sz):
                        chunk = body[i : i + chunk_sz]
                        f.write(chunk)
                        hasher.update(chunk)
                        md5_hasher.update(chunk)
                        downloaded_bytes += len(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                os.chmod(tmp, 0o600)
                downloaded_sha256 = hasher.hexdigest()
                downloaded_md5 = md5_hasher.hexdigest()

            # 1. Byte count integrity check
            if meta.size_bytes is None:
                raise DriveIntegrityError("Retrieved Drive artifact missing expected size_bytes.")
            expected_bytes = int(meta.size_bytes)
            if expected_bytes <= 0:
                if downloaded_bytes != 0:
                    raise DriveIntegrityError(
                        f"Retrieved Drive artifact size mismatch: expected {expected_bytes}, got {downloaded_bytes}"
                    )
            elif downloaded_bytes != expected_bytes:
                raise DriveIntegrityError(
                    f"Retrieved Drive artifact size mismatch: expected {expected_bytes}, got {downloaded_bytes}"
                )

            # 2. MD5 check if present
            if meta.md5:
                if downloaded_md5.lower() != meta.md5.lower():
                    raise DriveIntegrityError(
                        f"Retrieved Drive artifact MD5 mismatch: expected {meta.md5}, got {downloaded_md5}"
                    )

            # 3. SHA256 check
            if target_sha:
                if not downloaded_sha256.lower().startswith(target_sha.lower()):
                    raise DriveIntegrityError(
                        f"Retrieved Drive artifact hash mismatch: recorded {target_sha}, got {downloaded_sha256}"
                    )
                if expected_full_sha and downloaded_sha256.lower() != expected_full_sha.lower():
                    raise DriveIntegrityError(
                        f"Retrieved Drive artifact hash mismatch: expected {expected_full_sha}, got {downloaded_sha256}"
                    )
            elif not allow_unverified_recovery:
                raise DriveIntegrityError(
                    "Unverified Drive recovery refused: local manifest is missing and remote file lacks authoritative SHA256 metadata."
                )

            os.replace(str(tmp), str(dest))
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

    def retrieve(
        self,
        artifact_id: str,
        destination: Union[str, Path],
        expected_sha256: Optional[str] = None,
        allow_unverified_recovery: bool = False,
    ) -> Path:
        """Download artifact from Google Drive by ID with bounded memory streaming and strict integrity check."""
        req_sha: Optional[str] = None
        if "@sha256_" in artifact_id:
            base_art_id, sha_suffix = artifact_id.split("@sha256_", 1)
            req_sha = sha_suffix.strip().lower()
        else:
            base_art_id = artifact_id

        if expected_sha256 is not None:
            target_sha = expected_sha256.strip().lower()
            if req_sha is None:
                req_sha = target_sha
        elif req_sha is not None:
            target_sha = req_sha
        else:
            target_sha = None

        parts = base_art_id.split("/")
        if len(parts) < 3:
            raise ArtifactValidationError(f"Invalid artifact ID format: '{artifact_id}'")

        proj_id, r_id = parts[0], parts[1]
        rel_p = "/".join(parts[2:])
        fname = parts[-1]
        sub_f = parts[2] if len(parts) > 3 else None

        dest = Path(destination).resolve()
        if dest.is_dir():
            dest = dest / fname

        dest.parent.mkdir(parents=True, exist_ok=True)

        meta: Optional[StoredArtifact] = None

        # -----------------------------------------------------------------
        # Case A: Exact revision requested (@sha256_ or expected_sha256)
        # -----------------------------------------------------------------
        if req_sha is not None:
            # 1. Try local manifest / backup / cache
            meta = self.get_metadata(artifact_id)
            if meta and meta.remote_file_id and meta.sha256 and meta.sha256.lower().startswith(req_sha):
                target_sha = meta.sha256.lower()
            else:
                meta = None
                if "@sha256_" not in artifact_id:
                    meta_candidate = self.get_metadata(f"{base_art_id}@sha256_{req_sha}")
                    if meta_candidate and meta_candidate.remote_file_id and meta_candidate.sha256 and meta_candidate.sha256.lower().startswith(req_sha):
                        meta = meta_candidate
                        target_sha = meta.sha256.lower()

            # 2. Check journal if not found in manifest
            if not meta or not meta.remote_file_id:
                for rec in self.journal.list_transfers(project_id=proj_id, run_id=r_id):
                    if rec.status in (TransferStatus.PUBLISHED.value, TransferStatus.DRIVE_VERIFIED.value):
                        if rec.remote_file_id and rec.sha256 and rec.sha256.lower().startswith(req_sha):
                            if rec.target_rel_path == rel_p or Path(rec.local_path).name == fname:
                                meta = StoredArtifact(
                                    artifact_id=f"{base_art_id}@sha256_{rec.sha256.lower()}",
                                    storage_provider=self.provider_name,
                                    path=rec.target_rel_path or rel_p,
                                    size_bytes=rec.size_bytes,
                                    sha256=rec.sha256.lower(),
                                    project_id=proj_id,
                                    run_id=r_id,
                                    remote_file_id=rec.remote_file_id,
                                    md5=rec.md5,
                                    verified_at=rec.updated_at,
                                )
                                target_sha = rec.sha256.lower()
                                break

            # 3. If still not found in manifest/backup/journal, list candidate remote files in Drive folder
            if not meta or not meta.remote_file_id:
                try:
                    f_id = self.resolve_target_folder_id(proj_id, r_id, sub_f)
                except (DriveAuthenticationError, DriveQuotaExceededError, DriveTransferError):
                    raise
                except Exception as fe:
                    raise ArtifactNotFoundError(f"Folder not found on Drive for '{artifact_id}': {fe}") from fe

                candidates = self._list_files_in_folder(fname, f_id)
                if not candidates:
                    raise ArtifactNotFoundError(f"Artifact '{artifact_id}' not found in Google Drive registry.")

                def _candidate_priority(c: Dict[str, Any]) -> int:
                    props = c.get("appProperties") or {}
                    c_sha = props.get("sha256", "").strip().lower()
                    if c_sha and c_sha.startswith(req_sha):
                        return 0
                    return 1

                sorted_candidates = sorted(candidates, key=_candidate_priority)

                matched_any = False
                for cand in sorted_candidates:
                    c_fid = cand.get("id")
                    if not c_fid:
                        continue
                    c_props = cand.get("appProperties") or {}
                    if props_match_path(c_props, proj_id, r_id, rel_p) is False:
                        continue
                    c_size = int(cand["size"]) if cand.get("size") is not None else None
                    c_md5 = cand.get("md5Checksum")
                    cand_meta = StoredArtifact(
                        artifact_id=f"{base_art_id}@sha256_{req_sha}",
                        storage_provider=self.provider_name,
                        path=rel_p,
                        size_bytes=c_size if c_size is not None else 0,
                        sha256=req_sha,
                        project_id=proj_id,
                        run_id=r_id,
                        remote_file_id=c_fid,
                        md5=c_md5,
                    )
                    try:
                        self._download_and_verify(
                            cand_meta,
                            dest,
                            target_sha=req_sha,
                            expected_full_sha=expected_sha256 if (expected_sha256 and len(expected_sha256) == 64) else None,
                            allow_unverified_recovery=False,
                        )
                        matched_any = True
                        return dest
                    except DriveIntegrityError:
                        continue
                    except (DriveAuthenticationError, DriveQuotaExceededError, DriveTransferError):
                        raise

                if not matched_any:
                    raise DriveIntegrityError(
                        f"No remote candidate matched requested SHA256 '{req_sha}' for artifact '{artifact_id}'."
                    )

        # -----------------------------------------------------------------
        # Case B: Latest revision requested (unversioned, req_sha is None)
        # -----------------------------------------------------------------
        else:
            # 1. Resolve latest revision from manifest (or .bak backup)
            meta = self.get_metadata(artifact_id)
            if meta and meta.remote_file_id:
                target_sha = meta.sha256

            # 2. Check journal if manifest is missing/unrecoverable
            if not meta or not meta.remote_file_id:
                matching_transfers = [
                    rec for rec in self.journal.list_transfers(project_id=proj_id, run_id=r_id)
                    if rec.status in (TransferStatus.PUBLISHED.value, TransferStatus.DRIVE_VERIFIED.value)
                    and rec.remote_file_id
                    and (rec.target_rel_path == rel_p or Path(rec.local_path).name == fname)
                ]
                if len(matching_transfers) == 1 and matching_transfers[0].sha256:
                    rec = matching_transfers[0]
                    state, _ = self._remote_revision_state(rec.remote_file_id)
                    if state == STATE_LATEST:
                        meta = StoredArtifact(
                            artifact_id=f"{base_art_id}@sha256_{rec.sha256.lower()}",
                            storage_provider=self.provider_name,
                            path=rec.target_rel_path or rel_p,
                            size_bytes=rec.size_bytes,
                            sha256=rec.sha256.lower(),
                            project_id=proj_id,
                            run_id=r_id,
                            remote_file_id=rec.remote_file_id,
                            md5=rec.md5,
                            verified_at=rec.updated_at,
                        )
                        target_sha = rec.sha256.lower()

            # 3. Query Drive folder
            if not meta or not meta.remote_file_id:
                try:
                    f_id = self.resolve_target_folder_id(proj_id, r_id, sub_f)
                except (DriveAuthenticationError, DriveQuotaExceededError, DriveTransferError):
                    raise
                except Exception as fe:
                    raise ArtifactNotFoundError(f"Folder not found on Drive for '{artifact_id}': {fe}") from fe

                files = self._list_files_in_folder(fname, f_id)
                if len(files) == 0:
                    raise ArtifactNotFoundError(f"Artifact '{artifact_id}' not found in Google Drive registry.")

                latest_candidates = [
                    f for f in files
                    if props_match_path(f.get("appProperties"), proj_id, r_id, rel_p) is not False
                    and (f.get("appProperties") or {}).get("is_latest") == "true"
                ]
                valid_latest_candidates = [
                    f for f in latest_candidates
                    if is_sha256_hex((f.get("appProperties") or {}).get("sha256"))
                ]
                if len(latest_candidates) == 1 and len(valid_latest_candidates) == 1:
                    chosen = valid_latest_candidates[0]
                    app_props = chosen.get("appProperties") or {}
                    chosen_sha = app_props["sha256"].strip().lower()

                    meta = StoredArtifact(
                        artifact_id=f"{base_art_id}@sha256_{chosen_sha}",
                        storage_provider=self.provider_name,
                        path=rel_p,
                        size_bytes=int(chosen["size"]) if chosen.get("size") is not None else None,
                        sha256=chosen_sha,
                        project_id=proj_id,
                        run_id=r_id,
                        remote_file_id=chosen["id"],
                        md5=chosen.get("md5Checksum"),
                        verified_at=_utc_now_iso(),
                    )
                    target_sha = chosen_sha
                elif len(latest_candidates) == 1 and len(valid_latest_candidates) == 0:
                    chosen = latest_candidates[0]
                    chosen_sha = (chosen.get("appProperties") or {}).get("sha256")
                    raise DriveIntegrityError(
                        f"Ambiguous unversioned Drive recovery: latest remote file '{chosen.get('id')}' lacks valid 64-hex sha256 ({chosen_sha!r}) for '{artifact_id}' and local manifest is missing."
                    )
                elif len(latest_candidates) == 0:
                    matching_files = [
                        f for f in files
                        if props_match_path(f.get("appProperties"), proj_id, r_id, rel_p) is not False
                    ]
                    if len(matching_files) == 1 and allow_unverified_recovery:
                        chosen = matching_files[0]
                        if not chosen.get("md5Checksum") or chosen.get("size") is None:
                            raise DriveIntegrityError(
                                "Unverified Drive recovery refused: remote file lacks md5Checksum or size."
                            )
                        meta = StoredArtifact(
                            artifact_id=artifact_id,
                            storage_provider=self.provider_name,
                            path=rel_p,
                            size_bytes=int(chosen["size"]),
                            sha256="",
                            project_id=proj_id,
                            run_id=r_id,
                            remote_file_id=chosen["id"],
                            md5=chosen.get("md5Checksum"),
                            verified_at=_utc_now_iso(),
                        )
                        target_sha = None
                    else:
                        raise DriveIntegrityError(
                            f"Ambiguous unversioned Drive recovery: no remote revision marked latest found for '{artifact_id}' and local manifest is missing."
                        )
                else:
                    raise DriveIntegrityError(
                        f"Ambiguous unversioned Drive recovery: multiple ({len(latest_candidates)}) remote revisions marked latest exist for '{artifact_id}' and local manifest is missing."
                    )

        if not meta or not meta.remote_file_id:
            raise ArtifactNotFoundError(f"Artifact '{artifact_id}' not found in Google Drive registry.")

        # Download and strictly verify
        self._download_and_verify(
            meta,
            dest,
            target_sha=target_sha,
            expected_full_sha=expected_sha256 if (expected_sha256 and len(expected_sha256) == 64) else None,
            allow_unverified_recovery=allow_unverified_recovery,
        )
        return dest

    def exists(self, artifact_id: str) -> bool:
        meta = self.get_metadata(artifact_id)
        if meta is not None and meta.remote_file_id is not None:
            return True
        # Check journal
        parts = artifact_id.split("@sha256_")[0].split("/")
        if len(parts) >= 3:
            proj_id, r_id = parts[0], parts[1]
            fname = parts[-1]
            rel_p = "/".join(parts[2:])
            sub_f = parts[2] if len(parts) > 3 else None
            req_sha = artifact_id.split("@sha256_")[1].strip().lower() if "@sha256_" in artifact_id else None

            for rec in self.journal.list_transfers(project_id=proj_id, run_id=r_id):
                if rec.status in (TransferStatus.PUBLISHED.value, TransferStatus.DRIVE_VERIFIED.value):
                    if rec.remote_file_id and (rec.target_rel_path == rel_p or Path(rec.local_path).name == fname):
                        if req_sha is None or (rec.sha256 and rec.sha256.lower().startswith(req_sha)):
                            return True

            try:
                f_id = self.resolve_target_folder_id(proj_id, r_id, sub_f)
                files = self._list_files_in_folder(fname, f_id)
                if not files:
                    return False
                if req_sha is not None:
                    for f in files:
                        f_props = f.get("appProperties") or {}
                        if props_match_path(f_props, proj_id, r_id, rel_p) is False:
                            continue
                        c_sha = f_props.get("sha256", "").strip().lower()
                        if c_sha and c_sha.startswith(req_sha):
                            return True
                    return False
                for f in files:
                    f_props = f.get("appProperties") or {}
                    if props_match_path(f_props, proj_id, r_id, rel_p) is False:
                        continue
                    return True
                return False
            except (DriveAuthenticationError, DriveQuotaExceededError, DriveTransferError):
                raise
            except Exception:
                return False
        return False

    def get_metadata(self, artifact_id: str) -> Optional[StoredArtifact]:
        parts = artifact_id.split("@sha256_")[0].split("/")
        if len(parts) < 3:
            return None
        project_id, run_id = parts[0], parts[1]
        manifest = self._load_or_recover_manifest(project_id, run_id)
        if manifest:
            return self._resolve_artifact_from_manifest(manifest, artifact_id)
        return None

    def list_artifacts(
        self,
        project_id: str,
        run_id: Optional[str] = None,
    ) -> List[StoredArtifact]:
        if run_id:
            manifest = self._load_or_recover_manifest(project_id, run_id)
            if manifest:
                return manifest.list_artifacts()
            return []

        # All runs
        results = []
        proj_dir = self.staging_dir / "projects" / project_id / "runs"
        if proj_dir.is_dir():
            for r in proj_dir.iterdir():
                if r.is_dir():
                    manifest = self._load_or_recover_manifest(project_id, r.name)
                    if manifest:
                        results.extend(manifest.list_artifacts())
        return results
