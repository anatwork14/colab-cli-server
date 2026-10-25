"""Production-faithful Google Drive v3 REST API mock.

Enforces real Drive constraints:
- Custom property combined key+value <= 124 UTF-8 bytes (HTTP 400 otherwise)
- Max 30 custom appProperties per file (HTTP 400 otherwise)
- PATCH merges appProperties (null deletes a key)
- GET, PATCH, resumable-init honor `fields=` query parameter
- Injectable faults on any operation (init, chunk, patch, list, get, download)
- Failing the Nth PATCH
- 'Lost final response' on PATCH (commits change, then returns 503 or raises ConnectionResetError)
- Dropping / corrupting response fields
- Realistic pagination with pageSize and nextPageToken
- Recording every POST and PATCH for assertions
- `snapshot()` returns final remote state of all files
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from tests.test_google_drive_store import MockDriveHttpClient


class FileSnapshot(dict):
    """Snapshot of a single remote file's state."""

    @property
    def id(self) -> str:
        return self.get("id", "")

    @property
    def name(self) -> str:
        return self.get("name", "")

    @property
    def parent(self) -> Optional[str]:
        return self.get("parent")

    @property
    def appProperties(self) -> Dict[str, str]:
        return self.get("appProperties", {})

    @property
    def content_sha(self) -> Optional[str]:
        return self.get("content_sha")

    @property
    def size(self) -> int:
        return self.get("size", 0)

    @property
    def md5Checksum(self) -> Optional[str]:
        return self.get("md5Checksum")


class SnapshotList(list):
    """List of FileSnapshot objects with convenience lookup methods."""

    def get(self, file_id: str, default: Any = None) -> Any:
        for f in self:
            if f.get("id") == file_id:
                return f
        return default

    def by_id(self, file_id: str) -> Optional[FileSnapshot]:
        return self.get(file_id)

    def by_name(self, name: str) -> List[FileSnapshot]:
        return [f for f in self if f.get("name") == name]

    def values(self) -> List[FileSnapshot]:
        return list(self)

    def items(self) -> List[Tuple[str, FileSnapshot]]:
        return [(f["id"], f) for f in self]

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            res = self.get(item)
            if res is None:
                raise KeyError(f"File id '{item}' not found in snapshot")
            return res
        return super().__getitem__(item)


class StrictDriveMock(MockDriveHttpClient):
    """Deterministic, production-faithful Google Drive v3 REST API mock."""

    MAX_PROPERTY_BYTES = 124
    MAX_APP_PROPERTIES_COUNT = 30

    def __init__(self):
        super().__init__()
        # Operation-level fault queues: op -> list of fault dicts
        self.fault_queues: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)

        # Nth patch failure controls
        self.fail_nth_patch: Optional[int] = None
        self.fail_nth_patch_status: int = 503
        self.fail_nth_patch_body: Optional[bytes] = None
        self.fail_nth_patch_exception: Optional[Exception] = None
        self.total_patches_seen: int = 0

        # Lost final response on PATCH controls
        self.lost_final_response_on_patch: bool = False
        self.lost_final_response_count: int = 0
        self.lost_final_response_status: int = 503
        self.lost_final_response_as_exception: bool = False

        # Field dropping / corruption
        self.drop_response_fields: Set[str] = set()
        self.corrupt_response_fields: Dict[str, Any] = {}

        # Recorded operations for assertions
        self.recorded_patches: List[Dict[str, Any]] = []
        self.recorded_posts: List[Dict[str, Any]] = []
        self.recorded_requests: List[Dict[str, Any]] = []

        # Session tracking for requested fields
        self.session_to_fields: Dict[str, Optional[str]] = {}

    def inject_fault(
        self,
        operation: str,
        status_code: int = 500,
        count: int = 1,
        body: Optional[bytes] = None,
        exception: Optional[Exception] = None,
    ) -> None:
        """Queue an injectable fault on an operation ('init', 'chunk', 'patch', 'list', 'get', 'download')."""
        op = operation.lower().strip()
        for _ in range(count):
            self.fault_queues[op].append({
                "status_code": status_code,
                "body": body,
                "exception": exception,
            })

    def fail_operation(
        self,
        operation: str,
        status_code: int = 500,
        count: int = 1,
    ) -> None:
        """Convenience helper to inject status code fault on operation."""
        self.inject_fault(operation=operation, status_code=status_code, count=count)

    def _default_error_body(self, status: int) -> bytes:
        if status == 401:
            return b'{"error": {"code": 401, "message": "Invalid Credentials", "errors": [{"domain": "global", "reason": "authError", "message": "Invalid Credentials"}]}}'
        if status == 403:
            return b'{"error": {"code": 403, "message": "Drive storage quota exceeded", "errors": [{"domain": "usageLimits", "reason": "storageQuotaExceeded", "message": "Drive storage quota exceeded"}]}}'
        if status == 429:
            return b'{"error": {"code": 429, "message": "Rate limit exceeded", "errors": [{"domain": "usageLimits", "reason": "rateLimitExceeded", "message": "Rate limit exceeded"}]}}'
        if status == 500:
            return b'{"error": {"code": 500, "message": "Backend Error", "errors": [{"domain": "global", "reason": "backendError", "message": "Backend Error"}]}}'
        if status == 503:
            return b'{"error": {"code": 503, "message": "Service Unavailable", "errors": [{"domain": "global", "reason": "backendError", "message": "Service Unavailable"}]}}'
        if status == 400:
            return b'{"error": {"code": 400, "message": "Bad Request", "errors": [{"domain": "global", "reason": "invalid", "message": "Bad Request"}]}}'
        return json.dumps({"error": {"code": status, "message": f"HTTP Error {status}"}}).encode("utf-8")

    def _check_fault_queue(self, op: str) -> Optional[Tuple[int, Dict[str, str], bytes]]:
        if self.fault_queues.get(op):
            fault = self.fault_queues[op].pop(0)
            if fault.get("exception"):
                raise fault["exception"]
            status = fault["status_code"]
            body = fault.get("body")
            if body is None:
                body = self._default_error_body(status)
            return status, {"content-type": "application/json"}, body
        return None

    def _validate_single_property(self, key: Any, value: Any, is_patch: bool = False) -> Optional[str]:
        if not isinstance(key, str):
            return f"Custom property key must be a string, got {type(key).__name__}"
        key_bytes = len(key.encode("utf-8"))
        if key_bytes == 0:
            return "Custom property key must not be empty"
        if key_bytes > self.MAX_PROPERTY_BYTES:
            return f"Property '{key}' key length ({key_bytes} bytes) exceeds limit: {self.MAX_PROPERTY_BYTES} bytes"
        if is_patch and value is None:
            return None
        val_str = value if isinstance(value, str) else str(value)
        val_bytes = len(val_str.encode("utf-8"))
        total = key_bytes + val_bytes
        if total > self.MAX_PROPERTY_BYTES:
            return f"Property '{key}' combined key-value length ({total} bytes) exceeds limit: {self.MAX_PROPERTY_BYTES} bytes"
        return None

    def _validate_properties(self, props: Optional[Mapping[str, Any]], is_patch: bool = False) -> Optional[str]:
        if not props:
            return None
        for k, v in props.items():
            err = self._validate_single_property(k, v, is_patch=is_patch)
            if err:
                return err
        return None

    def _filter_file_dict(self, file_dict: Dict[str, Any], fields_str: Optional[str]) -> Dict[str, Any]:
        res = dict(file_dict)
        for k in self.drop_response_fields:
            res.pop(k, None)
        for k, v in self.corrupt_response_fields.items():
            res[k] = v

        if not fields_str or fields_str.strip() == "*":
            return res

        m = re.search(r"files\(([^)]+)\)", fields_str)
        if m:
            fields_str = m.group(1)

        req_fields = set(f.strip() for f in fields_str.split(",") if f.strip())
        return {k: v for k, v in res.items() if k in req_fields}

    def _find_mock_file(self, fid: str) -> Optional[Dict[str, Any]]:
        for mf in self.mock_files:
            if mf.get("id") == fid:
                return mf
        return None

    def _locked_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[bytes] = None,
        timeout: float = 30.0,
    ) -> Tuple[int, Dict[str, str], bytes]:
        headers = headers or {}

        # 1. Classify operation
        op = "unknown"
        if "oauth2.googleapis.com/token" in url:
            op = "token"
        elif "uploadType=resumable" in url and method == "POST":
            op = "init"
        elif "drive/v3/files" in url and method == "POST" and "uploadType=resumable" not in url:
            op = "folder_create"
        elif "session_id=upload-session" in url and method == "PUT":
            op = "chunk"
        elif "alt=media" in url and method == "GET":
            op = "download"
        elif "drive/v3/files?q=" in url and method == "GET":
            op = "list"
        elif re.search(r"drive/v3/files/[^/?]+", url) and method == "GET" and "?q=" not in url:
            op = "get"
        elif re.search(r"drive/v3/files/[^/?]+", url) and method == "PATCH":
            op = "patch"

        self.recorded_requests.append({
            "method": method,
            "url": url,
            "headers": dict(headers),
            "data": data,
            "op": op,
        })

        # Check faults for download / list / get / chunk early
        if op in ("download", "list", "get"):
            fault_resp = self._check_fault_queue(op)
            if fault_resp is not None:
                return fault_resp

        # 2. Resumable upload initiation (POST uploadType=resumable)
        if op == "init":
            meta = {}
            if data:
                try:
                    meta = json.loads(data.decode("utf-8"))
                except Exception:
                    pass

            # Fault check for init
            fault_resp = self._check_fault_queue("init")
            if fault_resp is not None:
                self.recorded_posts.append({
                    "url": url,
                    "headers": dict(headers),
                    "data": data,
                    "json": meta,
                    "status": fault_resp[0],
                    "is_resumable_init": True,
                })
                return fault_resp

            # Validate appProperties
            props = meta.get("appProperties")
            if props and isinstance(props, dict):
                if len(props) > self.MAX_APP_PROPERTIES_COUNT:
                    msg = f"Maximum number of custom properties ({self.MAX_APP_PROPERTIES_COUNT}) exceeded: {len(props)}"
                    self.recorded_posts.append({
                        "url": url,
                        "headers": dict(headers),
                        "data": data,
                        "json": meta,
                        "status": 400,
                        "is_resumable_init": True,
                    })
                    return 400, {"content-type": "application/json"}, json.dumps({
                        "error": {
                            "code": 400,
                            "message": msg,
                            "errors": [{"domain": "global", "reason": "invalid", "message": msg}],
                        }
                    }).encode("utf-8")

                err = self._validate_properties(props, is_patch=False)
                if err:
                    self.recorded_posts.append({
                        "url": url,
                        "headers": dict(headers),
                        "data": data,
                        "json": meta,
                        "status": 400,
                        "is_resumable_init": True,
                    })
                    return 400, {"content-type": "application/json"}, json.dumps({
                        "error": {
                            "code": 400,
                            "message": err,
                            "errors": [{"domain": "global", "reason": "invalid", "message": err}],
                        }
                    }).encode("utf-8")

            # Record fields requested in query param
            parsed_url = urllib.parse.urlparse(url)
            qs = urllib.parse.parse_qs(parsed_url.query)
            fields_param = qs.get("fields", [None])[0]

            status, resp_hdrs, body = super()._locked_request(method, url, headers=headers, data=data, timeout=timeout)
            session_url = resp_hdrs.get("location", "")
            if session_url:
                self.session_to_fields[session_url] = fields_param
                m_sess = re.search(r"session_id=([^&]+)", session_url)
                if m_sess:
                    self.session_to_fields[m_sess.group(1)] = fields_param

            self.recorded_posts.append({
                "url": url,
                "headers": dict(headers),
                "data": data,
                "json": meta,
                "status": status,
                "is_resumable_init": True,
            })
            return status, resp_hdrs, body

        # 3. Folder creation (POST /drive/v3/files without uploadType=resumable)
        if op == "folder_create":
            meta = {}
            if data:
                try:
                    meta = json.loads(data.decode("utf-8"))
                except Exception:
                    pass
            status, resp_hdrs, body = super()._locked_request(method, url, headers=headers, data=data, timeout=timeout)
            self.recorded_posts.append({
                "url": url,
                "headers": dict(headers),
                "data": data,
                "json": meta,
                "status": status,
                "is_resumable_init": False,
            })
            return status, resp_hdrs, body

        # 4. Chunk upload (PUT session_id=upload-session)
        if op == "chunk":
            fault_resp = self._check_fault_queue("chunk")
            if fault_resp is not None:
                return fault_resp

            status, resp_hdrs, body = super()._locked_request(method, url, headers=headers, data=data, timeout=timeout)
            # If upload just completed (HTTP 200), filter response by stored fields
            if status == 200 and body:
                fields_param = self.session_to_fields.get(url)
                if not fields_param:
                    m_sess = re.search(r"session_id=([^&]+)", url)
                    if m_sess:
                        fields_param = self.session_to_fields.get(m_sess.group(1))
                try:
                    res_dict = json.loads(body.decode("utf-8"))
                    filtered = self._filter_file_dict(res_dict, fields_param)
                    body = json.dumps(filtered).encode("utf-8")
                except Exception:
                    pass
            return status, resp_hdrs, body

        # 5. File PATCH (PATCH /drive/v3/files/{id})
        if op == "patch":
            self.total_patches_seen += 1
            m_fid = re.search(r"drive/v3/files/([^/?]+)", url)
            fid = m_fid.group(1) if m_fid else None
            patch_data = {}
            if data:
                try:
                    patch_data = json.loads(data.decode("utf-8"))
                except Exception:
                    pass
            new_app_props = patch_data.get("appProperties")

            # Check fault queue for "patch"
            fault_resp = self._check_fault_queue("patch")
            if fault_resp is not None:
                self.recorded_patches.append({
                    "file_id": fid,
                    "url": url,
                    "headers": dict(headers),
                    "data": data,
                    "json": patch_data,
                    "app_properties": new_app_props,
                    "status": fault_resp[0],
                })
                return fault_resp

            # Check fail_nth_patch
            if self.fail_nth_patch is not None and self.total_patches_seen == self.fail_nth_patch:
                self.recorded_patches.append({
                    "file_id": fid,
                    "url": url,
                    "headers": dict(headers),
                    "data": data,
                    "json": patch_data,
                    "app_properties": new_app_props,
                    "status": self.fail_nth_patch_status,
                })
                if self.fail_nth_patch_exception:
                    raise self.fail_nth_patch_exception
                return self.fail_nth_patch_status, {"content-type": "application/json"}, (self.fail_nth_patch_body or self._default_error_body(self.fail_nth_patch_status))

            # Check fail_next_n_patches
            if self.fail_next_n_patches > 0:
                self.fail_next_n_patches -= 1
                self.recorded_patches.append({
                    "file_id": fid,
                    "url": url,
                    "headers": dict(headers),
                    "data": data,
                    "json": patch_data,
                    "app_properties": new_app_props,
                    "status": self.patch_status_code,
                })
                return self.patch_status_code, {"content-type": "application/json"}, b'{"error": {"message": "Simulated PATCH failure"}}'

            # Validate 124-byte limit on each property in new_app_props
            if new_app_props and isinstance(new_app_props, dict):
                err = self._validate_properties(new_app_props, is_patch=True)
                if err:
                    self.recorded_patches.append({
                        "file_id": fid,
                        "url": url,
                        "headers": dict(headers),
                        "data": data,
                        "json": patch_data,
                        "app_properties": new_app_props,
                        "status": 400,
                    })
                    return 400, {"content-type": "application/json"}, json.dumps({
                        "error": {
                            "code": 400,
                            "message": err,
                            "errors": [{"domain": "global", "reason": "invalid", "message": err}],
                        }
                    }).encode("utf-8")

            # Look up file to patch
            found_file = self._find_mock_file(fid)
            if found_file is None:
                self.recorded_patches.append({
                    "file_id": fid,
                    "url": url,
                    "headers": dict(headers),
                    "data": data,
                    "json": patch_data,
                    "app_properties": new_app_props,
                    "status": 404,
                })
                return 404, {}, b'{"error": "Not Found"}'

            if "appProperties" not in found_file or not isinstance(found_file["appProperties"], dict):
                found_file["appProperties"] = {}

            # Check max property count after merge
            simulated_props = dict(found_file["appProperties"])
            if new_app_props and isinstance(new_app_props, dict):
                for k, v in new_app_props.items():
                    if v is None:
                        simulated_props.pop(k, None)
                    else:
                        simulated_props[k] = str(v)
                if len(simulated_props) > self.MAX_APP_PROPERTIES_COUNT:
                    msg = f"Maximum number of custom properties ({self.MAX_APP_PROPERTIES_COUNT}) exceeded: {len(simulated_props)}"
                    self.recorded_patches.append({
                        "file_id": fid,
                        "url": url,
                        "headers": dict(headers),
                        "data": data,
                        "json": patch_data,
                        "app_properties": new_app_props,
                        "status": 400,
                    })
                    return 400, {"content-type": "application/json"}, json.dumps({
                        "error": {
                            "code": 400,
                            "message": msg,
                            "errors": [{"domain": "global", "reason": "invalid", "message": msg}],
                        }
                    }).encode("utf-8")

            # Check abort_before_promote_patch
            if self.abort_before_promote_patch and new_app_props and new_app_props.get("is_latest") == "true":
                raise ConnectionResetError("Simulated crash before promotion PATCH")

            # Perform merge with null-delete
            if new_app_props and isinstance(new_app_props, dict):
                for k, v in new_app_props.items():
                    if v is None:
                        found_file["appProperties"].pop(k, None)
                    else:
                        found_file["appProperties"][k] = str(v)

            # Record successful PATCH
            self.recorded_patches.append({
                "file_id": fid,
                "url": url,
                "headers": dict(headers),
                "data": data,
                "json": patch_data,
                "app_properties": new_app_props,
                "status": 200,
            })

            # Check lost final response
            trigger_lost = False
            if self.lost_final_response_count > 0:
                self.lost_final_response_count -= 1
                if self.lost_final_response_count == 0:
                    self.lost_final_response_on_patch = False
                trigger_lost = True
            elif self.lost_final_response_on_patch:
                self.lost_final_response_on_patch = False
                trigger_lost = True

            if trigger_lost:
                if self.lost_final_response_as_exception:
                    raise ConnectionResetError("Connection lost after PATCH committed")
                else:
                    return self.lost_final_response_status, {"content-type": "application/json"}, b'{"error": {"code": 503, "message": "Backend Error (lost response after commit)"}}'

            # Build response honoring fields
            c_data = found_file.get("content")
            if c_data is None and fid in self.committed_files:
                c_data = self.committed_files[fid]

            if c_data is not None:
                sz = str(len(c_data))
                md5 = hashlib.md5(c_data).hexdigest()
            else:
                sz = str(found_file.get("size", 0))
                md5 = found_file.get("md5Checksum", "")

            full_resp = {
                "id": fid,
                "name": found_file.get("name", "uploaded_file"),
                "size": sz,
                "md5Checksum": md5,
                "appProperties": dict(found_file["appProperties"]),
                "description": found_file.get("description", ""),
                "createdTime": found_file.get("createdTime", "2026-01-01T00:00:00Z"),
            }

            parsed_url = urllib.parse.urlparse(url)
            qs = urllib.parse.parse_qs(parsed_url.query)
            fields_param = qs.get("fields", [None])[0]
            filtered_resp = self._filter_file_dict(full_resp, fields_param)
            return 200, {"content-type": "application/json"}, json.dumps(filtered_resp).encode("utf-8")

        # 6. File metadata GET
        if op == "get":
            status, resp_hdrs, body = super()._locked_request(method, url, headers=headers, data=data, timeout=timeout)
            if status == 200 and body:
                parsed_url = urllib.parse.urlparse(url)
                qs = urllib.parse.parse_qs(parsed_url.query)
                fields_param = qs.get("fields", [None])[0]
                try:
                    res_dict = json.loads(body.decode("utf-8"))
                    filtered = self._filter_file_dict(res_dict, fields_param)
                    body = json.dumps(filtered).encode("utf-8")
                except Exception:
                    pass
            return status, resp_hdrs, body

        # 7. File list GET
        if op == "list":
            status, resp_hdrs, body = super()._locked_request(method, url, headers=headers, data=data, timeout=timeout)
            if status == 200 and body:
                parsed_url = urllib.parse.urlparse(url)
                qs = urllib.parse.parse_qs(parsed_url.query)
                fields_param = qs.get("fields", [None])[0]
                try:
                    res_dict = json.loads(body.decode("utf-8"))
                    files_list = res_dict.get("files", [])
                    if fields_param and "files(" in fields_param:
                        filtered_files = [self._filter_file_dict(f, fields_param) for f in files_list]
                        res_dict["files"] = filtered_files
                    elif self.drop_response_fields or self.corrupt_response_fields:
                        filtered_files = [self._filter_file_dict(f, None) for f in files_list]
                        res_dict["files"] = filtered_files
                    body = json.dumps(res_dict).encode("utf-8")
                except Exception:
                    pass
            return status, resp_hdrs, body

        # Default fallback to parent
        return super()._locked_request(method, url, headers=headers, data=data, timeout=timeout)

    def snapshot(self, include_folders: bool = False) -> SnapshotList:
        """Return final remote state of all files (name, parent, appProperties, content_sha)."""
        result = SnapshotList()
        with self.lock:
            for f in self.mock_files:
                if not include_folders and f.get("mimeType") == "application/vnd.google-apps.folder":
                    continue
                fid = f.get("id", "")
                content = self.committed_files.get(fid)
                if content is None:
                    content = f.get("content")

                if content is not None:
                    content_sha = hashlib.sha256(content).hexdigest()
                    size = len(content)
                else:
                    content_sha = f.get("sha256") or f.get("appProperties", {}).get("sha256")
                    size = int(f.get("size", 0))

                props = dict(f.get("appProperties") or {})

                snap = FileSnapshot({
                    "id": fid,
                    "name": f.get("name", ""),
                    "parent": f.get("parent"),
                    "appProperties": props,
                    "content_sha": content_sha,
                    "size": size,
                    "mimeType": f.get("mimeType"),
                    "md5Checksum": f.get("md5Checksum") or (hashlib.md5(content).hexdigest() if content is not None else None),
                })
                result.append(snap)
        return result
