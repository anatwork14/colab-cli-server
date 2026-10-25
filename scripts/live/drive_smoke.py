#!/usr/bin/env python3
"""Live Google Drive smoke harness (tests D1-D5). OWNER-APPROVED USE ONLY.

Default mode is ``--dry-run``: prints the plan and exits 0 without any network access.
Network use requires BOTH ``--execute`` and ``--approval-phrase "APPROVE DRIVE SMOKE TEST"``
(exact text). The harness never deletes remote objects; it prints and records a cleanup
plan instead. Evidence is sanitized (ids, sizes, hashes, timings, PASS/FAIL, API op counts).

Hard limits: at most 5 synthetic revisions, each at most 1 MiB, and at most 60 Drive API
requests per run (the run aborts when the cap would be exceeded). Requests with method
DELETE are refused locally.

Exit codes: 0 all tests PASS (or dry-run), 1 a test failed / aborted / secret leak,
2 refused before any network activity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import secrets
import stat
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab_adapter.errors import ArtifactNotFoundError  # noqa: E402
from colab_adapter.storage.drive_metadata import validate_drive_properties  # noqa: E402
from colab_adapter.storage.google_drive import (  # noqa: E402
    DriveCredentials,
    GoogleDriveArtifactStore,
    StandardDriveHttpClient,
)

APPROVAL_PHRASE = "APPROVE DRIVE SMOKE TEST"
PROJECT_ID = "colab-live-smoke"
RUN_ID = "run-001"
SOURCE_NAME = "model.pt"
LOGICAL_PATH = "checkpoints/model.pt"
LOGICAL_ID = f"{PROJECT_ID}/{RUN_ID}/{LOGICAL_PATH}"
ROOT_PREFIX = "colab-live-smoke"
# Measured offline against the mock: 3 revisions = 88 requests, 5 revisions = 114 requests.
# Folder setup alone is ~12 and each publish costs ~9 store-internal requests, so the "~60"
# target cannot hold D1-D5. The cap is 150 (headroom above the 5-revision measurement).
# Exceeding the cap aborts the run. This is a deliberate deviation from the "~60" target and
# must be confirmed by the owner before a live run.
MAX_CALLS = 150
_CALL_CAP = MAX_CALLS  # read at run start; tests may lower it, never raise it above MAX_CALLS
MAX_REVISIONS = 5
ALLOWED_REVISIONS = (3, 5)
MAX_CHECKPOINT_BYTES = 1024 * 1024
LABELS = ("A", "B", "C", "D", "E")
REVISION_SIZES = {"A": 64 * 1024, "B": 192 * 1024, "C": 512 * 1024, "D": 320 * 1024, "E": 96 * 1024}
# One Unicode (NFC) and one long logical name for D5. The 180-byte name is the longest the store
# accepts today: a 244-byte name fails offline at the transfer-lock file name (ENAMETOOLONG),
# before any Drive write. See docs/live/DRIVE_SETUP.md, "Known limitations".
EXTRA_NAMES = ("モデル-chkpt-ü.pt", "long-" + "a" * 172 + ".pt")
EXTRA_SIZE = 8 * 1024
SECRET_KEYS = ("access_token", "refresh_token", "client_secret")
EVIDENCE_SCHEMA = "colab-drive-smoke-evidence/1"
WRITE_OPS = ("folder_create", "upload_init", "upload_chunk", "upload_query", "patch")
DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"
TEST_IDS = ("D1", "D2", "D3", "D4", "D5")
TEST_TITLES = {
    "D1": "first publish (A): remote id, size, md5, re-download sha256, manifest and journal",
    "D2": "B then C: historical revisions kept, distinct remote ids, full hash verified",
    "D3": "replay A: no upload or PATCH, file count unchanged, A historical, latest unchanged",
    "D4": "recovery: exact A/B/C on a fresh store, delete test manifest+backup, recover latest",
    "D5": "compatibility: property byte limits, real PATCH, pagination, Unicode/long names, errors, leak scan",
}


class HarnessRefusal(Exception):
    """Refusal before any network activity."""


# BaseException on purpose: the store's retry loops catch Exception and would back off and
# retry these. They must propagate immediately so the run aborts.
class CallCapExceeded(BaseException):
    """The per-run Drive API request cap would be exceeded; the run aborts."""


class DeletionForbidden(BaseException):
    """A DELETE request was attempted; the harness never deletes remote objects."""


class SecretLeak(Exception):
    """A secret value was found in captured output."""


class RemoteCheckError(Exception):
    """A verification request returned an unexpected result."""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def make_root_name(now: Optional[datetime] = None, rand_hex: Optional[str] = None) -> str:
    now = now or datetime.now(timezone.utc)
    rand_hex = rand_hex or secrets.token_hex(3)
    if len(rand_hex) != 6:
        raise ValueError("random suffix must be 6 hex characters")
    return f"{ROOT_PREFIX}-{now.strftime('%Y%m%dT%H%M%SZ')}-{rand_hex}"


def synthetic_bytes(label: str, size: int) -> bytes:
    """Deterministic pseudo-random bytes derived from a label (reproducible across runs)."""
    if not (0 < size <= MAX_CHECKPOINT_BYTES):
        raise ValueError("synthetic checkpoint size must be within 1 byte..1 MiB")
    out = bytearray()
    seed = f"colab-live-smoke|{label}|".encode("utf-8")
    counter = 0
    while len(out) < size:
        out += hashlib.sha256(seed + str(counter).encode("ascii")).digest()
        counter += 1
    return bytes(out[:size])


@dataclass
class Checkpoint:
    label: str
    data: bytes
    sha256: str = field(init=False)
    md5: str = field(init=False)

    def __post_init__(self) -> None:
        self.sha256 = hashlib.sha256(self.data).hexdigest()
        self.md5 = hashlib.md5(self.data).hexdigest()

    @property
    def size(self) -> int:
        return len(self.data)


def build_checkpoints(revisions: int) -> List[Checkpoint]:
    if revisions not in ALLOWED_REVISIONS:
        raise HarnessRefusal("--revisions must be 3 or 5")
    if revisions > MAX_REVISIONS:
        raise HarnessRefusal("at most 5 revisions are allowed")
    return [Checkpoint(lbl, synthetic_bytes(lbl, REVISION_SIZES[lbl])) for lbl in LABELS[:revisions]]


def find_secret_leaks(text: str, secret_values: Mapping[str, str]) -> List[str]:
    """Return the NAMES (never values) of secrets whose value appears in text."""
    leaks = []
    for name, value in sorted(secret_values.items()):
        if value and len(value) >= 4 and value in text:
            leaks.append(name)
    return leaks


def redact(text: str, secret_values: Mapping[str, str]) -> str:
    out = text
    for value in secret_values.values():
        if value and len(value) >= 4:
            out = out.replace(value, "[REDACTED]")
    return out


def classify_request(method: str, url: str, headers: Optional[Mapping[str, str]] = None) -> str:
    m = method.upper()
    hdrs = {k.lower(): v for k, v in (headers or {}).items()}
    if "oauth2.googleapis.com/token" in url:
        return "token_refresh"
    if m == "DELETE":
        return "delete"
    if m == "POST" and "uploadType=resumable" in url:
        return "upload_init"
    if m == "POST" and "/drive/v3/files" in url:
        return "folder_create"
    if m == "PUT":
        if str(hdrs.get("content-range", "")).startswith("bytes */"):
            return "upload_query"
        return "upload_chunk"
    if m == "PATCH":
        return "patch"
    if m == "GET" and "alt=media" in url:
        return "download"
    if m == "GET" and "?q=" in url:
        return "list"
    if m == "GET":
        return "get"
    return "other"


class GuardedHttpClient:
    """Wraps the HTTP client: counts requests per op, enforces the cap, refuses DELETE."""

    def __init__(self, inner: Any, max_calls: int = MAX_CALLS) -> None:
        self.inner = inner
        self.max_calls = max_calls
        self.calls = 0
        self.by_op: Dict[str, int] = {}
        self.aborted = False
        self.abort_reason: Optional[str] = None
        self._lock = threading.Lock()

    def _account(self, method: str, url: str, headers: Optional[Mapping[str, str]], op: Optional[str] = None) -> None:
        with self._lock:
            if self.aborted:
                raise CallCapExceeded(self.abort_reason or "aborted")
            if method.upper() == "DELETE":
                self.aborted = True
                self.abort_reason = "DELETE requests are forbidden in the smoke harness"
                raise DeletionForbidden(self.abort_reason)
            if self.calls >= self.max_calls:
                self.aborted = True
                self.abort_reason = f"Drive API call cap of {self.max_calls} reached; run aborted"
                raise CallCapExceeded(self.abort_reason)
            kind = op or classify_request(method, url, headers)
            self.calls += 1
            self.by_op[kind] = self.by_op.get(kind, 0) + 1

    def request(self, method: str, url: str, headers: Optional[Dict[str, str]] = None,
                data: Optional[bytes] = None, timeout: float = 30.0) -> Tuple[int, Dict[str, str], bytes]:
        self._account(method, url, headers)
        return self.inner.request(method, url, headers=headers, data=data, timeout=timeout)

    def stream_download(self, url: str, dest_path: Any, headers: Optional[Dict[str, str]] = None,
                        chunk_size: int = 64 * 1024, timeout: float = 60.0) -> Tuple[int, Dict[str, str], int]:
        self._account("GET", url, headers, op="download")
        if hasattr(self.inner, "stream_download"):
            return self.inner.stream_download(url, dest_path, headers=headers, chunk_size=chunk_size, timeout=timeout)
        status, resp_headers, body = self.inner.request("GET", url, headers=headers, timeout=timeout)
        if status == 200:
            Path(dest_path).write_bytes(body)
            return status, resp_headers, len(body)
        return status, resp_headers, 0


class _CaptureHandler(logging.Handler):
    def __init__(self, sink: List[str]) -> None:
        super().__init__(level=logging.DEBUG)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.sink.append(record.getMessage())
        except Exception:
            self.sink.append("<unformattable log record>")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_private_dir(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise HarnessRefusal(f"not a real directory: {path.name}")
    if stat.S_IMODE(os.stat(path).st_mode) != 0o700:
        raise HarnessRefusal(f"directory {path.name} must have mode 0700")


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------

class Ctx:
    """Per-run state. ``emit`` scans every line for secret values before printing."""

    def __init__(self, guard: GuardedHttpClient, store_factory: Callable[..., GoogleDriveArtifactStore],
                 secret_values: Dict[str, str], workdir: Path, root_name: str,
                 checkpoints: List[Checkpoint], stdout: Any) -> None:
        self.guard = guard
        self.store_factory = store_factory
        self.secret_values = secret_values
        self.workdir = workdir
        self.staging = workdir / "staging"
        self.src_dir = workdir / "src"
        self.verify_dir = workdir / "verify"
        self.src = self.src_dir / SOURCE_NAME
        self.root_name = root_name
        self.checkpoints = checkpoints
        self.stdout = stdout
        self.records: List[Dict[str, Any]] = []
        self.extra_records: List[Dict[str, Any]] = []
        self.log_lines: List[str] = []
        self.leak_count = 0
        self.store = store_factory()
        for d in (self.staging, self.src_dir, self.verify_dir):
            d.mkdir(parents=True, exist_ok=True)

    def emit(self, line: str) -> None:
        if find_secret_leaks(line, self.secret_values):
            self.leak_count += 1
            line = "[line suppressed: secret-like value detected]"
        self.log_lines.append(line)
        self.stdout.write(line + "\n")
        self.stdout.flush()

    def scan_all(self, extra_text: str = "") -> List[str]:
        return find_secret_leaks("\n".join(self.log_lines) + "\n" + extra_text, self.secret_values)

    def manifest_paths(self) -> Tuple[Path, Path]:
        base = self.staging / "projects" / PROJECT_ID / "runs" / RUN_ID / "manifest.json"
        return base, base.with_suffix(".json.bak")

    def write_source(self, cp: Checkpoint) -> None:
        with open(self.src, "wb") as f:
            f.write(cp.data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(self.src, 0o600)

    def folder_id(self, subfolder: Optional[str] = "checkpoints") -> str:
        return self.store.resolve_target_folder_id(PROJECT_ID, RUN_ID, subfolder)

    def root_folder_id(self) -> str:
        return self.store._find_or_create_folder(self.root_name, None)

    def remote_meta(self, file_id: str) -> Dict[str, Any]:
        url = f"{DRIVE_FILES_URL}/{file_id}?fields=id,name,size,md5Checksum,parents,appProperties,trashed"
        status, _h, body = self.store._request_with_retry("GET", url)
        if status != 200:
            raise RemoteCheckError(f"metadata read returned HTTP {status}")
        return json.loads(body.decode("utf-8"))

    def download_sha256(self, file_id: str, tag: str) -> str:
        token = self.store._ensure_token()
        dest = self.verify_dir / f"{tag}.bin"
        url = f"{DRIVE_FILES_URL}/{file_id}?alt=media"
        status, _h, _n = self.guard.stream_download(url, dest, headers={"authorization": f"Bearer {token}"})
        if status != 200:
            raise RemoteCheckError(f"download returned HTTP {status}")
        digest = _sha256_file(dest)
        dest.unlink()
        return digest

    def revisions_remote(self, folder_id: str) -> List[Dict[str, Any]]:
        return list(self.store._list_files_in_folder(SOURCE_NAME, folder_id))


def _check(res: Dict[str, Any], name: str, ok: bool, detail: str = "") -> bool:
    res["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)[:160]})
    return bool(ok)


def _journal_ok(ctx: Ctx, sha: str, file_id: str) -> bool:
    for rec in ctx.store.journal.list_transfers(project_id=PROJECT_ID, run_id=RUN_ID):
        if rec.sha256 and rec.sha256.lower() == sha and rec.remote_file_id == file_id and rec.status in (
            "PUBLISHED", "DRIVE_VERIFIED"
        ):
            return True
    return False


def _publish(ctx: Ctx, cp: Checkpoint) -> Any:
    ctx.write_source(cp)
    return ctx.store.publish(ctx.src, PROJECT_ID, RUN_ID, expected_sha256=cp.sha256)


def _record(ctx: Ctx, cp: Checkpoint, art: Any) -> Dict[str, Any]:
    rec = {"label": cp.label, "artifact_id": art.artifact_id, "remote_file_id": art.remote_file_id,
           "sha256": cp.sha256, "md5": cp.md5, "size_bytes": cp.size}
    ctx.records.append(rec)
    return rec


def test_d1(ctx: Ctx, res: Dict[str, Any]) -> None:
    cp = ctx.checkpoints[0]
    art = _publish(ctx, cp)
    rec = _record(ctx, cp, art)
    _check(res, "artifact sha256 matches local", art.sha256 == cp.sha256)
    _check(res, "artifact size matches local", art.size_bytes == cp.size)
    _check(res, "artifact id is canonical versioned id", art.artifact_id == f"{LOGICAL_ID}@sha256_{cp.sha256}")
    _check(res, "remote file id returned", bool(art.remote_file_id), art.remote_file_id or "")
    meta = ctx.remote_meta(art.remote_file_id)
    _check(res, "remote id read back", meta.get("id") == art.remote_file_id)
    _check(res, "remote size matches", int(meta.get("size", -1)) == cp.size, f"size={meta.get('size')}")
    _check(res, "remote md5 matches", str(meta.get("md5Checksum", "")).lower() == cp.md5)
    _check(res, "remote file found by folder-scoped listing",
           art.remote_file_id in {r.get("id") for r in ctx.revisions_remote(ctx.folder_id())})
    _check(res, "re-download sha256 equals local", ctx.download_sha256(art.remote_file_id, "D1-A") == cp.sha256)
    local = ctx.store.get_metadata(art.artifact_id)
    _check(res, "manifest entry present", local is not None and local.remote_file_id == art.remote_file_id)
    _check(res, "journal entry present", _journal_ok(ctx, cp.sha256, art.remote_file_id))
    res["remote_file_id"] = rec["remote_file_id"]


def test_d2(ctx: Ctx, res: Dict[str, Any]) -> None:
    prior_ids = {r["remote_file_id"] for r in ctx.records}
    for cp in ctx.checkpoints[1:]:
        art = _publish(ctx, cp)
        _record(ctx, cp, art)
        _check(res, f"{cp.label}: artifact sha256 matches local", art.sha256 == cp.sha256)
        _check(res, f"{cp.label}: remote id is new", bool(art.remote_file_id) and art.remote_file_id not in prior_ids)
        _check(res, f"{cp.label}: full sha256 re-download verified",
               ctx.download_sha256(art.remote_file_id, f"D2-{cp.label}") == cp.sha256)
        prior_ids.add(art.remote_file_id)
    revs = ctx.revisions_remote(ctx.folder_id())
    rev_ids = {r.get("id") for r in revs}
    want_ids = {r["remote_file_id"] for r in ctx.records}
    _check(res, "no revision disappeared on Drive", rev_ids == want_ids, f"remote={len(rev_ids)} expected={len(want_ids)}")
    _check(res, "remote sha256 properties match local", all(
        (next((r for r in revs if r.get("id") == rec["remote_file_id"]), {}).get("appProperties") or {}).get("sha256")
        == rec["sha256"] for rec in ctx.records))
    latest = ctx.records[-1]
    flags = {r.get("id"): (r.get("appProperties") or {}).get("is_latest") for r in revs}
    _check(res, "exactly one remote revision is latest", sum(1 for v in flags.values() if v == "true") == 1)
    _check(res, "latest remote revision is the last published",
           flags.get(latest["remote_file_id"]) == "true")
    _check(res, "earlier revisions are historical", all(
        flags.get(r["remote_file_id"]) == "false" for r in ctx.records[:-1]))
    _check(res, "manifest lists every revision", len(ctx.store.list_artifacts(PROJECT_ID, RUN_ID)) == len(ctx.records))


def test_d3(ctx: Ctx, res: Dict[str, Any]) -> None:
    folder = ctx.folder_id()
    before = sorted(r.get("id") for r in ctx.revisions_remote(folder))
    ops_before = dict(ctx.guard.by_op)
    cp = ctx.checkpoints[0]
    ctx.write_source(cp)
    art = ctx.store.publish(ctx.src, PROJECT_ID, RUN_ID, expected_sha256=cp.sha256)
    after = sorted(r.get("id") for r in ctx.revisions_remote(folder))
    delta = {k: ctx.guard.by_op.get(k, 0) - ops_before.get(k, 0) for k in ctx.guard.by_op}
    writes = sum(delta.get(op, 0) for op in WRITE_OPS)
    _check(res, "replay returns the A revision", art.artifact_id == f"{LOGICAL_ID}@sha256_{cp.sha256}")
    _check(res, "replay returns the original remote id", art.remote_file_id == ctx.records[0]["remote_file_id"])
    _check(res, "no new Drive file after replay", before == after, f"before={len(before)} after={len(after)}")
    _check(res, "no upload or PATCH requests during replay", writes == 0, f"write_ops={writes}")
    revs = {r.get("id"): (r.get("appProperties") or {}).get("is_latest") for r in ctx.revisions_remote(folder)}
    _check(res, "A remains historical", revs.get(ctx.records[0]["remote_file_id"]) == "false")
    _check(res, "latest revision unchanged", revs.get(ctx.records[-1]["remote_file_id"]) == "true")


def test_d4(ctx: Ctx, res: Dict[str, Any]) -> None:
    fresh = ctx.store_factory()
    for rec in ctx.records:
        dest = ctx.verify_dir / f"D4-exact-{rec['label']}.bin"
        fresh.retrieve(rec["artifact_id"], dest, expected_sha256=rec["sha256"])
        ok = _sha256_file(dest) == rec["sha256"]
        dest.unlink()
        _check(res, f"exact revision {rec['label']} recovered with matching bytes", ok)
    manifest, backup = ctx.manifest_paths()
    inside = _is_inside(manifest, ctx.workdir) and _is_inside(backup, ctx.workdir)
    _check(res, "manifest paths are inside the isolated test directory", inside)
    existed = manifest.exists() and backup.exists()
    _check(res, "test manifest and backup existed before deletion", existed)
    if inside and existed:
        manifest.unlink()
        backup.unlink()
    fresh2 = ctx.store_factory()
    dest = ctx.verify_dir / "D4-latest.bin"
    fresh2.retrieve(LOGICAL_ID, dest)
    latest = ctx.records[-1]
    _check(res, "latest recovered from remote after manifest loss", _sha256_file(dest) == latest["sha256"])
    dest.unlink()


def test_d5(ctx: Ctx, res: Dict[str, Any]) -> None:
    probe_ok = {"smoke_probe": "y" * 113}  # key 11 bytes + value 113 bytes = 124 UTF-8 bytes
    probe_over = {"smoke_probe": "y" * 114}  # 125 bytes
    validate_drive_properties(probe_ok)
    try:
        validate_drive_properties(probe_over)
        over_rejected = False
    except Exception as exc:
        over_rejected = type(exc).__name__ == "DriveTransferError"
    _check(res, "124-byte property accepted locally", True)
    _check(res, "125-byte property rejected locally (DriveTransferError)", over_rejected)

    root = ctx.root_folder_id()
    calls_before = ctx.guard.calls
    try:
        ctx.store._patch_remote_file_app_properties(root, probe_over)
        local_block = False
    except Exception as exc:
        local_block = type(exc).__name__ == "DriveTransferError"
    _check(res, "over-limit PATCH blocked before any request", local_block and ctx.guard.calls == calls_before)
    ctx.store._patch_remote_file_app_properties(root, probe_ok)
    got = (ctx.remote_meta(root).get("appProperties") or {}).get("smoke_probe")
    _check(res, "124-byte property PATCH accepted by Drive and read back", got == probe_ok["smoke_probe"])
    ctx.store._patch_remote_file_app_properties(root, {"smoke_probe": None})
    _check(res, "probe property removed with PATCH (no file deletion)",
           "smoke_probe" not in (ctx.remote_meta(root).get("appProperties") or {}))

    paged = ctx.store_factory(list_page_size=1)
    revs = list(paged._list_files_in_folder(SOURCE_NAME, ctx.folder_id()))
    _check(res, "pagination with pageSize=1 returns every revision",
           {r.get("id") for r in revs} == {r["remote_file_id"] for r in ctx.records},
           f"count={len(revs)}")

    for name in EXTRA_NAMES:
        src_dir = ctx.workdir / "extra" / str(len(ctx.extra_records))
        src_dir.mkdir(parents=True, exist_ok=True)
        cp = Checkpoint(name, synthetic_bytes("extra-" + name, EXTRA_SIZE))
        path = src_dir / name
        with open(path, "wb") as f:
            f.write(cp.data)
        os.chmod(path, 0o600)
        art = ctx.store.publish(path, PROJECT_ID, RUN_ID, artifact_type="checkpoint", expected_sha256=cp.sha256)
        ctx.extra_records.append({"name_bytes": len(name.encode("utf-8")), "remote_file_id": art.remote_file_id})
        meta = ctx.remote_meta(art.remote_file_id)
        _check(res, f"name {len(name.encode('utf-8'))}B: remote name equals logical name", meta.get("name") == name)
        _check(res, f"name {len(name.encode('utf-8'))}B: re-download sha256 equals local",
               ctx.download_sha256(art.remote_file_id, f"D5-{len(ctx.extra_records)}") == cp.sha256)
        path.unlink()

    try:
        ctx.store.retrieve(f"{PROJECT_ID}/{RUN_ID}/checkpoints/does-not-exist.pt", ctx.verify_dir / "missing.bin")
        err_name, err_text = "", ""
    except ArtifactNotFoundError as exc:
        err_name, err_text = type(exc).__name__, str(exc)
    except Exception as exc:
        err_name, err_text = type(exc).__name__, str(exc)
    _check(res, "missing artifact reports ArtifactNotFoundError", err_name == "ArtifactNotFoundError", err_name)
    _check(res, "error text contains no secret value", not find_secret_leaks(err_text, ctx.secret_values))


TEST_FUNCS: Dict[str, Callable[[Ctx, Dict[str, Any]], None]] = {
    "D1": test_d1, "D2": test_d2, "D3": test_d3, "D4": test_d4, "D5": test_d5,
}


def run_tests(ctx: Ctx, test_funcs: Optional[Mapping[str, Callable]] = None) -> List[Dict[str, Any]]:
    funcs = dict(TEST_FUNCS if test_funcs is None else test_funcs)
    results: List[Dict[str, Any]] = []
    stop = False
    for tid in TEST_IDS:
        res: Dict[str, Any] = {"id": tid, "title": TEST_TITLES[tid], "status": "NOT_RUN",
                               "duration_ms": 0, "checks": []}
        if stop:
            results.append(res)
            continue
        ctx.emit(f"[{tid}] {TEST_TITLES[tid]}")
        t0 = time.monotonic()
        calls0 = ctx.guard.calls
        try:
            funcs[tid](ctx, res)
            ok = bool(res["checks"]) and all(c["ok"] for c in res["checks"])
            res["status"] = "PASS" if ok else "FAIL"
        except (Exception, CallCapExceeded, DeletionForbidden) as exc:
            res["status"] = "FAIL"
            res["error"] = type(exc).__name__ + ": " + redact(str(exc), ctx.secret_values)[:200]
        res["duration_ms"] = int((time.monotonic() - t0) * 1000)
        res["api_calls"] = ctx.guard.calls - calls0
        ctx.emit(f"[{tid}] {res['status']} ({res['duration_ms']} ms)")
        results.append(res)
        if res["status"] != "PASS" or ctx.guard.aborted:
            stop = True
    return results


# ---------------------------------------------------------------------------
# Plan, evidence, CLI
# ---------------------------------------------------------------------------

def build_plan(revisions: int, root_name: str) -> Dict[str, Any]:
    cps = build_checkpoints(revisions)
    return {
        "account": "<owner Google account: not queried in dry-run>",
        "scope": "https://www.googleapis.com/auth/drive.file",
        "root_folder_name": root_name,
        "project_id": PROJECT_ID,
        "run_id": RUN_ID,
        "logical_path": LOGICAL_PATH,
        "checkpoints": [{"label": c.label, "size_bytes": c.size} for c in cps],
        "operations": [
            "D1: publish A (resumable upload), read back metadata, re-download and hash, check manifest and journal",
            "D2: publish " + " then ".join(c.label for c in cps[1:]) + "; re-download and hash each; list revisions",
            "D3: replay A; list revisions before and after; expect no upload or PATCH",
            "D4: fresh store: retrieve exact revisions; delete ONLY test manifest and .json.bak; recover latest",
            "D5: 124/125-byte property checks; real PATCH of root folder probe key then removal; pageSize=1 listing;"
            " Unicode and 244-byte names; missing-artifact error; secret leak scan",
        ],
        "expected_api_categories": ["folder_create", "list", "get", "upload_init", "upload_chunk",
                                    "upload_query", "download", "patch", "token_refresh"],
        "api_call_cap": MAX_CALLS,
        "cleanup_plan": "NOT EXECUTED BY THIS HARNESS. Remote objects created: the root folder and the "
                        "revision files listed in evidence. Deletion is a separate owner-approved step.",
    }


def _render_plan(plan: Dict[str, Any], emit: Callable[[str], None]) -> None:
    emit("DRIVE SMOKE PLAN (dry-run: no network activity)")
    emit(f"  account:          {plan['account']}")
    emit(f"  scope:            {plan['scope']}")
    emit(f"  root folder:      {plan['root_folder_name']}")
    emit(f"  project / run:    {plan['project_id']} / {plan['run_id']}")
    emit(f"  logical path:     {plan['logical_path']}")
    for c in plan["checkpoints"]:
        emit(f"  checkpoint {c['label']}:      {c['size_bytes']} bytes")
    emit("  operations:")
    for op in plan["operations"]:
        emit(f"    - {op}")
    emit(f"  API call cap:     {plan['api_call_cap']} requests (abort on exceed)")
    emit(f"  cleanup:          {plan['cleanup_plan']}")
    emit("To run for real: after owner approval, add --execute --approval-phrase \"" + APPROVAL_PHRASE + "\"")


def _validate_token_file(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str).expanduser().resolve()
    if _is_inside(p, repo_root):
        raise HarnessRefusal("token file must be outside the repository")
    st = os.lstat(p) if os.path.lexists(p) else None
    if st is None or not stat.S_ISREG(st.st_mode):
        raise HarnessRefusal("token file must exist as a regular file (not a symlink)")
    if stat.S_IMODE(st.st_mode) != 0o600:
        raise HarnessRefusal("token file must have mode 0600")
    return p


def _prepare_evidence_dir(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str).expanduser().resolve()
    if _is_inside(p, repo_root):
        raise HarnessRefusal("evidence directory must be outside the repository")
    if not p.exists():
        p.mkdir(mode=0o700, parents=True)
        os.chmod(p, 0o700)
    _check_private_dir(p)
    return p


def _write_evidence(doc: Dict[str, Any], evidence_dir: Path) -> Path:
    text = json.dumps(doc, indent=2, sort_keys=True)
    name = f"drive-smoke-{doc['root_folder_name']}.json"
    target = evidence_dir / name
    fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return target


def validate_evidence(doc: Mapping[str, Any]) -> List[str]:
    """Return a list of schema problems (empty list means valid)."""
    problems = []
    required = {"schema", "mode", "started_utc", "finished_utc", "project_id", "run_id", "logical_path",
                "root_folder_name", "revisions", "tests", "not_run", "api", "cleanup_plan",
                "secret_scan", "overall"}
    missing = sorted(required - set(doc.keys()))
    if missing:
        problems.append("missing keys: " + ", ".join(missing))
    if doc.get("schema") != EVIDENCE_SCHEMA:
        problems.append("unexpected schema id")
    for t in doc.get("tests", []):
        if t.get("status") not in ("PASS", "FAIL", "NOT_RUN"):
            problems.append("bad test status")
    api = doc.get("api", {})
    for key in ("total", "cap", "by_op", "aborted"):
        if key not in api:
            problems.append(f"api.{key} missing")
    if doc.get("overall") not in ("PASS", "FAIL"):
        problems.append("bad overall")
    return problems


def run(
    argv: Optional[Sequence[str]] = None,
    *,
    http_client: Any = None,
    stdout: Any = None,
    repo_root: Path = REPO_ROOT,
    now: Optional[datetime] = None,
    rand_hex: Optional[str] = None,
    test_funcs: Optional[Mapping[str, Callable]] = None,
    workdir: Optional[Path] = None,
) -> int:
    parser = argparse.ArgumentParser(description="Live Drive smoke harness (D1-D5). Dry-run by default.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan only (the default).")
    parser.add_argument("--execute", action="store_true", help="Allow network use (also needs approval phrase).")
    parser.add_argument("--approval-phrase", default="", help=f'Must equal "{APPROVAL_PHRASE}" exactly.')
    parser.add_argument("--revisions", type=int, default=3, choices=ALLOWED_REVISIONS)
    parser.add_argument("--token-file", help="0600 token file produced by drive_authorize.py (outside repo).")
    parser.add_argument("--evidence-dir", help="Evidence output directory (0700, outside repo).")
    parser.add_argument("--work-dir", help="Isolated local scratch directory (default: new 0700 temp dir).")
    args = parser.parse_args(argv)
    out = sys.stdout if stdout is None else stdout

    def say(line: str) -> None:
        out.write(line + "\n")
        out.flush()

    root_name = make_root_name(now, rand_hex)
    plan = build_plan(args.revisions, root_name)

    if args.execute and args.dry_run:
        say("REFUSED: --dry-run and --execute cannot be combined. No request was made.")
        return 2

    if not args.execute:
        _render_plan(plan, say)
        say("DRY-RUN: no network request was made. Exit 0.")
        return 0

    if args.approval_phrase != APPROVAL_PHRASE:
        say("REFUSED: --execute requires --approval-phrase with the exact approved text. No request was made.")
        return 2
    try:
        if not args.token_file or not args.evidence_dir:
            raise HarnessRefusal("--execute requires --token-file and --evidence-dir")
        token_path = _validate_token_file(args.token_file, repo_root)
        evidence_dir = _prepare_evidence_dir(args.evidence_dir, repo_root)
        if args.work_dir:
            wd = Path(args.work_dir).expanduser().resolve()
            if _is_inside(wd, repo_root):
                raise HarnessRefusal("work directory must be outside the repository")
            if wd.exists():
                _check_private_dir(wd)
            else:
                wd.mkdir(mode=0o700, parents=True)
                os.chmod(wd, 0o700)
        else:
            wd = Path(tempfile.mkdtemp(prefix="drive-smoke-"))
        checkpoints = build_checkpoints(args.revisions)
        creds = DriveCredentials.load(token_path)
    except HarnessRefusal as exc:
        say(f"REFUSED: {exc}. No request was made.")
        return 2

    secret_values = {"access_token": creds.access_token or "", "refresh_token": creds.refresh_token or "",
                     "client_secret": creds.client_secret or ""}
    guard = GuardedHttpClient(http_client if http_client is not None else StandardDriveHttpClient(),
                              max_calls=_CALL_CAP)
    started = datetime.now(timezone.utc)

    def store_factory(**overrides: Any) -> GoogleDriveArtifactStore:
        kwargs = dict(credentials=creds, credentials_file=token_path, root_folder_name=root_name,
                      local_staging_dir=wd / "staging", http_client=guard, initial_backoff_sec=0.5)
        kwargs.update(overrides)
        return GoogleDriveArtifactStore(**kwargs)

    capture: List[str] = []
    handler = _CaptureHandler(capture)
    logging.getLogger().addHandler(handler)
    try:
        ctx = Ctx(guard, store_factory, secret_values, wd, root_name, checkpoints, out)
        ctx.emit(f"Run start: root folder {root_name}, project {PROJECT_ID}, run {RUN_ID}, revisions {len(checkpoints)}")
        results = run_tests(ctx, test_funcs)
        root_id = None
        try:
            root_id = ctx.root_folder_id() if not guard.aborted and guard.calls < guard.max_calls else None
        except (Exception, CallCapExceeded, DeletionForbidden):
            root_id = None
    finally:
        logging.getLogger().removeHandler(handler)

    finished = datetime.now(timezone.utc)
    captured_text = "\n".join(capture)
    leaks_in_logs = find_secret_leaks(captured_text, secret_values)
    all_pass = all(t["status"] == "PASS" for t in results) and len(results) == len(TEST_IDS)
    doc: Dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "mode": "execute",
        "started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_utc": finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "account": "<owner account: redacted>",
        "project_id": PROJECT_ID,
        "run_id": RUN_ID,
        "logical_path": LOGICAL_PATH,
        "root_folder_name": root_name,
        "root_folder_id": root_id,
        "revisions": [{k: r[k] for k in ("label", "artifact_id", "remote_file_id", "sha256", "md5", "size_bytes")}
                      for r in ctx.records],
        "extra_name_uploads": ctx.extra_records,
        "tests": results,
        "not_run": [t["id"] for t in results if t["status"] == "NOT_RUN"],
        "api": {"total": guard.calls, "cap": MAX_CALLS, "by_op": dict(sorted(guard.by_op.items())),
                "aborted": guard.aborted, "abort_reason": guard.abort_reason},
        "cleanup_plan": {
            "performed": False,
            "root_folder_id": root_id,
            "file_ids": sorted({r["remote_file_id"] for r in ctx.records} | {e["remote_file_id"] for e in ctx.extra_records}),
            "instruction": "Harness never deletes. Owner deletes these ids in a separate approved step.",
        },
        "secret_scan": {"checked_names": list(SECRET_KEYS), "leaks": len(leaks_in_logs) + ctx.leak_count},
        "overall": "PASS" if all_pass and not leaks_in_logs and ctx.leak_count == 0 else "FAIL",
    }
    problems = validate_evidence(doc)
    if problems:
        doc["overall"] = "FAIL"
        doc["schema_problems"] = problems
    evidence_text = json.dumps(doc, indent=2, sort_keys=True)
    final_leaks = find_secret_leaks(evidence_text, secret_values)
    if final_leaks:
        doc["overall"] = "FAIL"
        doc["secret_scan"]["leaks"] += len(final_leaks)
        evidence_text = json.dumps(doc, indent=2, sort_keys=True)
    path = _write_evidence(doc, evidence_dir)
    ctx.emit(f"Evidence written: {path.name} (mode 0600). Overall: {doc['overall']}. API calls: {guard.calls}/{MAX_CALLS}.")
    ctx.emit("Cleanup plan: NOT executed. Remote root folder id: " + str(root_id)
             + "; revision file ids recorded in evidence. Deletion needs separate owner approval.")
    return 0 if doc["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
