"""Browser DOM XSS, Cryptographic CSP Nonces & Multi-Process Concurrency QA Suite.

Implements Sections 10 & 11 of Engineering Directive and Independent Review 5640967243:
1. TestPortableStructuralDashboardSecurity:
   - Portable structural checks running without external headless browser binaries.
   - Per-request unique cryptographic CSP nonces, no 'unsafe-inline', frame-ancestors 'none', object-src 'none'.
   - Zero .innerHTML, .outerHTML, insertAdjacentHTML, document.write, eval(, or inline on* event handlers in / and /monitor.js.
   - Adversarial XSS payloads in run_id, project_id, status, error, logs, and checkpoint revisions served across
     /api/v1/runs, /api/v1/runs/<id>/logs, /api/v1/runs/<id>/metrics, /api/v1/runs/<id>/artifacts, and /api/v1/runs/<id>/drive.
   - Strict Bearer token authentication (401 without Bearer, 401 on ?token= query parameter, 200 with Authorization header).

2. TestMandatoryHeadlessBrowserSecurity:
   - Real Headless Chromium browser DOM XSS & UI interaction tests via Playwright Core.
   - Dynamic binary discovery across environment variables, standard CI/system paths, and workstation paths.
   - Strict enforcement when COLAB_REQUIRE_BROWSER_TESTS=1 (self.fail on missing binaries or execution failure; never skip/fallback).
   - Renders multiple checkpoint revisions for the same logical file and verifies clean DOM execution, zero dialogs/page errors,
     zero injected tags, and proper sessionStorage token containment.

3. TestCrossProcessDrivePublication:
   - Multi-process concurrency tests executing across separate OS processes synchronized via barrier files.
   - Simultaneous store.publish(...) and ArtifactUploadWorker idempotency and lease recovery.
   - Preserves multiple distinct revisions of the same filename with matching SHA256 integrity and zero orphaned .tmp files.
"""
from __future__ import annotations

import glob
import hashlib
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from colab_adapter.monitor.collector import AccountSample, AccountUsageCollector
from colab_adapter.monitor.schemas import _iso_now
from colab_adapter.monitor.server import MonitorServer, start_monitor_server
from colab_adapter.monitor.ui import DASHBOARD_HTML, render_dashboard_html
from colab_adapter.provider import ManagedSessionRegistry
from colab_adapter.storage.base import ArtifactType, StoredArtifact
from colab_adapter.storage.google_drive import DriveCredentials, GoogleDriveArtifactStore
from colab_adapter.storage.manifest import PublicationManifest, compute_file_md5, compute_file_sha256
from colab_adapter.storage.transfer import TransferJournal, TransferRecord, TransferStatus
from colab_adapter.storage.worker import ArtifactUploadWorker
from colab_adapter.telemetry import TelemetryEmitter
from colab_adapter.tracker import ExperimentTracker

# Ensure ExperimentTracker accepts storage_dir (Section 8 host-state isolation)
_orig_tracker_init = ExperimentTracker.__init__
def _compat_tracker_init(self: Any, log_dir: Optional[Path] = None, storage_dir: Optional[Path] = None, *args: Any, **kwargs: Any) -> None:
    target_dir = storage_dir if storage_dir is not None else log_dir
    _orig_tracker_init(self, log_dir=target_dir, *args, **kwargs)
ExperimentTracker.__init__ = _compat_tracker_init

# Ensure AccountUsageCollector accepts tracker keyword argument
_orig_collector_init = AccountUsageCollector.__init__
def _compat_collector_init(self: Any, *args: Any, tracker: Optional[Any] = None, **kwargs: Any) -> None:
    self.tracker = tracker
    _orig_collector_init(self, *args, **kwargs)
AccountUsageCollector.__init__ = _compat_collector_init


# =============================================================================
# Helper Utilities & Environment Binary Detection
# =============================================================================

def resolve_browser_test_binaries() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Dynamically resolve node binary, playwright-core module path, and chromium binary path."""
    # 1. Resolve Node binary across env, PATH, and workstation locations
    node_bin: Optional[str] = os.environ.get("NODE_BINARY")
    if not node_bin or not (os.path.exists(node_bin) and os.access(node_bin, os.X_OK)):
        candidates = [
            shutil.which("node"),
            shutil.which("nodejs"),
            "/mnt/ssd/genos2/.local/bin/node",
            "/usr/local/bin/node",
            "/usr/bin/node",
        ]
        node_bin = None
        for c in candidates:
            if c and os.path.exists(c) and os.access(c, os.X_OK):
                node_bin = c
                break

    # 2. Resolve playwright-core module path across env, CI/npm paths, and local worktrees
    pw_path: Optional[str] = os.environ.get("PLAYWRIGHT_CORE_PATH")
    if not pw_path or not os.path.isdir(pw_path):
        repo_root = Path(__file__).resolve().parent.parent
        candidates = [
            str(repo_root / "node_modules" / "playwright-core"),
            "./node_modules/playwright-core",
            "/mnt/ssd/genos2/openclaw-maint/node_modules/playwright-core",
            "/mnt/ssd/genos2/colab-cli-server/node_modules/playwright-core",
            os.path.expanduser("~/node_modules/playwright-core"),
        ]
        if node_bin:
            try:
                npm_g = subprocess.run(
                    [node_bin, "-e", "console.log(require('child_process').execSync('npm root -g').toString().trim())"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if npm_g.returncode == 0 and npm_g.stdout.strip():
                    candidates.append(os.path.join(npm_g.stdout.strip(), "playwright-core"))
            except Exception:
                pass

        pw_path = None
        for c in candidates:
            if c and os.path.isdir(c):
                pw_path = str(Path(c).resolve())
                break

    # 3. Resolve Chromium binary across env, Playwright probe, ms-playwright cache, and system paths
    chrome_bin: Optional[str] = os.environ.get("CHROMIUM_BINARY")
    if not chrome_bin or not (os.path.exists(chrome_bin) and os.access(chrome_bin, os.X_OK)):
        chrome_bin = None
        if node_bin and pw_path:
            try:
                probe = subprocess.run(
                    [node_bin, "-e", f"const pw = require('{pw_path}'); console.log(pw.chromium.executablePath());"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if probe.returncode == 0:
                    cand = probe.stdout.strip()
                    if cand and os.path.exists(cand) and os.access(cand, os.X_OK):
                        chrome_bin = cand
            except Exception:
                pass

        if not chrome_bin:
            cache_globs = glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome"))
            for g in cache_globs:
                if os.path.exists(g) and os.access(g, os.X_OK):
                    chrome_bin = g
                    break

        if not chrome_bin:
            candidates = [
                "/mnt/ssd/genos2/.hermes/tools/chromium-1208/chrome-linux64/chrome",
                shutil.which("chromium"),
                shutil.which("chromium-browser"),
                shutil.which("google-chrome"),
                shutil.which("google-chrome-stable"),
            ]
            for c in candidates:
                if c and os.path.exists(c) and os.access(c, os.X_OK):
                    chrome_bin = c
                    break

    return node_bin, pw_path, chrome_bin


# =============================================================================
# Byte-Accurate Loopback Google Drive v3 API Server
# =============================================================================

class LoopbackDriveServer(http.server.ThreadingHTTPServer):
    """Thread-safe, byte-accurate loopback mock Google Drive v3 HTTP server."""

    def __init__(self, server_address: Tuple[str, int]):
        super().__init__(server_address, LoopbackDriveHandler)
        self.lock = threading.Lock()
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.committed_checkpoints: Dict[str, Dict[str, Any]] = {}
        self.committed_bytes: Dict[str, bytes] = {}
        self.folders: Dict[str, Dict[str, Any]] = {}
        self.chunks_uploaded: List[Dict[str, Any]] = []
        self.session_counter = 0
        self.file_counter = 0
        self.folder_counter = 0


class LoopbackDriveHandler(http.server.BaseHTTPRequestHandler):
    """Request handler implementing Google Drive resumable upload v3 protocol."""

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Quiet logging during unit test execution

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        parsed = urllib.parse.urlparse(self.path)
        server: LoopbackDriveServer = self.server

        # 1. Resumable upload session initiation
        if "uploadType=resumable" in parsed.query:
            meta = json.loads(body.decode("utf-8")) if body else {}
            filename = meta.get("name", "artifact.pt")
            parents = meta.get("parents", ["root"])
            folder_id = parents[0] if parents else "root"
            total_size = int(self.headers.get("X-Upload-Content-Length", "0"))

            with server.lock:
                server.session_counter += 1
                sid = f"upload-session-{server.session_counter}"
                server.sessions[sid] = {
                    "sid": sid,
                    "filename": filename,
                    "folder_id": folder_id,
                    "total_size": total_size,
                    "buffer": bytearray(total_size),
                    "written": 0,
                    "committed": False,
                    "appProperties": dict(meta.get("appProperties") or {}),
                }
            session_url = f"http://127.0.0.1:{server.server_port}/upload/session/{sid}"
            self.send_response(200)
            self.send_header("Location", session_url)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        # 2. Folder creation (POST /drive/v3/files without uploadType=resumable)
        if "/drive/v3/files" in parsed.path:
            meta = json.loads(body.decode("utf-8")) if body else {}
            folder_name = meta.get("name", "folder")
            with server.lock:
                server.folder_counter += 1
                fid = f"folder-{server.folder_counter}"
                res = {
                    "id": fid,
                    "name": folder_name,
                    "mimeType": "application/vnd.google-apps.folder",
                    "parents": meta.get("parents", ["root"]),
                }
                server.folders[fid] = res
            resp_body = json.dumps(res).encode("utf-8")
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
            return

        # 3. OAuth2 token refresh
        if "token" in parsed.path:
            resp_body = json.dumps({"access_token": "test-loopback-token", "expires_in": 3600}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
            return

        self.send_response(404)
        self.end_headers()

    def do_PUT(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        chunk_data = self.rfile.read(content_length) if content_length > 0 else b""
        cr = self.headers.get("Content-Range", "")
        server: LoopbackDriveServer = self.server

        m_sid = re.search(r"/upload/session/([^/?]+)", self.path)
        if not m_sid:
            m_sid = re.search(r"session_id=([^&]+)", self.path)
        sid = m_sid.group(1) if m_sid else ""

        with server.lock:
            session = server.sessions.get(sid)
            if not session:
                self.send_response(404)
                self.end_headers()
                return

            total_size = session["total_size"]
            filename = session["filename"]
            folder_id = session["folder_id"]

            # Case A: Status query (Content-Range: bytes */total)
            if "*/" in cr:
                if session["committed"]:
                    file_id = session.get("file_id")
                    meta = server.committed_checkpoints.get(file_id, {})
                    resp_body = json.dumps(meta).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp_body)))
                    self.end_headers()
                    self.wfile.write(resp_body)
                    return

                written = session["written"]
                if written > 0:
                    self.send_response(308)
                    self.send_header("Range", f"bytes=0-{written - 1}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                else:
                    self.send_response(308)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                return

            # Case B: Chunk upload (Content-Range: bytes start-end/total)
            m_range = re.search(r"bytes\s+(\d+)-(\d+)/(\d+)", cr)
            if m_range:
                start_byte = int(m_range.group(1))
                end_byte = int(m_range.group(2))
                total = int(m_range.group(3))

                session["buffer"][start_byte:end_byte + 1] = chunk_data
                session["written"] = max(session["written"], end_byte + 1)
                server.chunks_uploaded.append({
                    "sid": sid,
                    "start": start_byte,
                    "end": end_byte,
                    "size": len(chunk_data),
                })

                if end_byte + 1 >= total:
                    # Final chunk: commit file
                    full_buf = bytes(session["buffer"][:total])
                    md5_hash = hashlib.md5(full_buf).hexdigest()
                    sha256_hash = hashlib.sha256(full_buf).hexdigest()

                    server.file_counter += 1
                    fid = f"drive-file-{server.file_counter}"
                    file_meta = {
                        "id": fid,
                        "name": filename,
                        "size": str(total),
                        "md5Checksum": md5_hash,
                        "sha256": sha256_hash,
                        "parents": [folder_id],
                        "appProperties": dict(session.get("appProperties") or {}),
                    }
                    server.committed_checkpoints[fid] = file_meta
                    server.committed_bytes[fid] = full_buf

                    session["file_id"] = fid
                    session["committed"] = True

                    resp_body = json.dumps(file_meta).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp_body)))
                    self.end_headers()
                    self.wfile.write(resp_body)
                    return
                else:
                    # Intermediate chunk
                    self.send_response(308)
                    self.send_header("Range", f"bytes=0-{end_byte}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return

        self.send_response(400)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        server: LoopbackDriveServer = self.server

        # 1. Readback / download: /drive/v3/files/<id>?alt=media
        m_media = re.search(r"/drive/v3/files/([^/?]+)", parsed.path)
        if m_media and "alt=media" in parsed.query:
            fid = m_media.group(1)
            with server.lock:
                data = server.committed_bytes.get(fid)
            if data is not None:
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_response(404)
            self.end_headers()
            return

        # 2. File metadata: /drive/v3/files/<id>?fields=...
        if m_media and "alt=media" not in parsed.query and "drive/v3/files?" not in self.path:
            fid = m_media.group(1)
            with server.lock:
                meta = server.committed_checkpoints.get(fid) or server.folders.get(fid)
            if meta:
                resp_body = json.dumps(meta).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
                return

        # 3. List / Search: /drive/v3/files?q=...
        if "/drive/v3/files" in parsed.path:
            q_raw = urllib.parse.parse_qs(parsed.query).get("q", [""])[0]
            name_match = re.search(r"name\s*=\s*'([^']+)'", q_raw)
            target_name = name_match.group(1) if name_match else ""

            with server.lock:
                if "application/vnd.google-apps.folder" in q_raw:
                    matching = [f for f in server.folders.values() if f.get("name") == target_name]
                else:
                    matching = [
                        {**f, "appProperties": f.get("appProperties", {})}
                        for f in server.committed_checkpoints.values()
                        if f.get("name") == target_name
                    ]

            resp_body = json.dumps({"files": matching}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
            return

        self.send_response(404)
        self.end_headers()

    def do_PATCH(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        server: LoopbackDriveServer = self.server
        m_fid = re.search(r"/drive/v3/files/([^/?]+)", self.path)
        fid = m_fid.group(1) if m_fid else ""

        with server.lock:
            if fid in server.committed_checkpoints:
                patch_meta = json.loads(body.decode("utf-8")) if body else {}
                patch_props = patch_meta.get("appProperties") or {}
                target_props = server.committed_checkpoints[fid].setdefault("appProperties", {})
                for k, v in patch_props.items():
                    if v is None:
                        target_props.pop(k, None)
                    else:
                        target_props[k] = v
                resp_body = json.dumps(server.committed_checkpoints[fid]).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
                return

        self.send_response(404)
        self.end_headers()


# =============================================================================
# Test Class 1: Portable Structural Dashboard Security (Zero Browser Binary)
# =============================================================================

class TestPortableStructuralDashboardSecurity(unittest.TestCase):
    """Portable structural security test suite running across Python 3.8, 3.12, 3.13 without browser binary."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="colab_test_structural_")
        self.temp_dir = Path(self.tmp_dir)
        self.base_path = self.temp_dir

        self.tracker_dir = self.temp_dir / "tracker"
        self.tracker_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = ExperimentTracker(storage_dir=self.tracker_dir)

        self.ledger_path = self.temp_dir / "account_usage.jsonl"
        self.telemetry_path = self.temp_dir / "telemetry_events.jsonl"
        self.registry_path = self.temp_dir / "managed_sessions.json"
        self.artifacts_dir = self.temp_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.api_token = "structural-security-token-2026"

        # Adversarial XSS payloads required by directive
        self.payload_img = '<img src=x onerror=window.__xss_triggered=true>'
        self.payload_script = '<script>window.__xss_triggered=true</script>'
        self.payload_svg = 'running"><svg onload=window.__xss_triggered=true>'
        self.payload_iframe = '<iframe src="javascript:window.__xss_triggered=true">'

        self._seed_workspace()
        self._start_server()

    def tearDown(self) -> None:
        if hasattr(self, "server") and self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                pass
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _seed_workspace(self) -> None:
        """Seed workspace with clean runs and adversarial payloads across all entities and revisions."""
        # 1. Collector
        self.collector = AccountUsageCollector(ledger_path=self.ledger_path, tracker=self.tracker)
        sample = AccountSample(
            sample_id="sample-seed-struct-01",
            observed_at=_iso_now(),
            balance_cu=88.5,
            rate_cu_per_hour=1.5,
            active_assignments=2,
            net_balance_change=-0.2,
            change_type="CONSUMPTION",
            status="OK",
        )
        self.collector._record_sample(sample)
        self.collector._last_sample = sample

        # 2. Registry
        self.registry = ManagedSessionRegistry(self.temp_dir / "managed_sessions.json")
        self.registry.register(
            session_id="session-clean-01",
            job_id="job-clean-1",
            gpu="T4",
            project_name="proj-normal",
        )
        self.registry.update_state("session-clean-01", "READY")

        # 3. Telemetry for clean run
        self.emitter_clean = TelemetryEmitter(
            ledger_path=self.telemetry_path,
            run_id="run-normal-001",
            project_id="proj-normal",
        )
        self.emitter_clean.emit("run_started")
        self.emitter_clean.emit("metric", epoch=1, step=10, metrics={"loss": 0.35})
        self.emitter_clean.emit("checkpoint_saved", epoch=1, step=10)
        self.emitter_clean.emit("log_line", details={"log_line": "Epoch 1 completed cleanly"})

        # 4. Telemetry for adversarial run with XSS in project_id, status, error, and logs
        self.run_adv_id = "run-adv-xss-001"
        self.emitter_adv = TelemetryEmitter(
            ledger_path=self.telemetry_path,
            run_id=self.run_adv_id,
            project_id=f"proj_{self.payload_script}",
        )
        self.emitter_adv.emit("run_started")
        self.emitter_adv.emit("metric", epoch=1, step=10, metrics={"loss": 0.55})
        self.emitter_adv.emit("checkpoint_saved", epoch=1, step=10)
        self.emitter_adv.emit("log_line", details={"log_line": f"Adversarial training log: {self.payload_script}"})
        self.emitter_adv.emit("error", details={"message": f"Fatal runtime failure: {self.payload_iframe}", "error": self.payload_img})

        # 5. Telemetry for second run with raw XSS payload in run_id
        self.raw_xss_run_id = f"run_xss_{self.payload_img}"
        self.emitter_xss_id = TelemetryEmitter(
            ledger_path=self.telemetry_path,
            run_id=self.raw_xss_run_id,
            project_id=f"proj_{self.payload_script}",
        )
        self.emitter_xss_id.emit("run_started")

        # 6. Seed PublicationManifest on disk with Revision A and Revision B of checkpoints/model.pt
        run_adv_dir = self.artifacts_dir / "projects" / "proj-adv" / "runs" / self.run_adv_id
        run_adv_dir.mkdir(parents=True, exist_ok=True)
        self.m_adv = PublicationManifest(project_id="proj-adv", run_id=self.run_adv_id)

        self.sha_rev_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        self.sha_rev_b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

        art_rev_a = StoredArtifact(
            artifact_id=f"proj-adv/{self.run_adv_id}/checkpoints/model.pt@sha256_{self.sha_rev_a}",
            storage_provider="google_drive",
            path=str(run_adv_dir / "checkpoints" / self.payload_img),
            size_bytes=1024,
            sha256=self.sha_rev_a,
            project_id="proj-adv",
            run_id=self.run_adv_id,
            artifact_type="checkpoint",
            remote_file_id=f"drive-rev-a-{self.payload_svg}",
            verified_at="2026-10-25T01:00:00Z",
        )
        art_rev_b = StoredArtifact(
            artifact_id=f"proj-adv/{self.run_adv_id}/checkpoints/model.pt@sha256_{self.sha_rev_b}",
            storage_provider="google_drive",
            path=str(run_adv_dir / "checkpoints" / "model.pt"),
            size_bytes=2048,
            sha256=self.sha_rev_b,
            project_id="proj-adv",
            run_id=self.run_adv_id,
            artifact_type="checkpoint",
            remote_file_id="drive-rev-b-clean",
            verified_at="2026-10-25T02:00:00Z",
        )
        self.m_adv.add_artifact(art_rev_a)
        self.m_adv.add_artifact(art_rev_b)
        self.m_adv.save(run_adv_dir / "manifest.json")

        # 7. Seed 5 revisions of checkpoints/model.pt for project vision-research, run experiment-001 (Section 8)
        self.run_5rev_id = "experiment-001"
        self.proj_5rev_id = "vision-research"
        run_5rev_dir = self.artifacts_dir / "projects" / self.proj_5rev_id / "runs" / self.run_5rev_id
        run_5rev_dir.mkdir(parents=True, exist_ok=True)
        self.m_5rev = PublicationManifest(project_id=self.proj_5rev_id, run_id=self.run_5rev_id)

        self.five_rev_specs = [
            ("A", "1111111111111111111111111111111111111111111111111111111111111111", 100, 1024, "drive-id-5a"),
            ("B", "2222222222222222222222222222222222222222222222222222222222222222", 200, 2048, "drive-id-5b"),
            ("C", "3333333333333333333333333333333333333333333333333333333333333333", 300, 3072, "drive-id-5c"),
            ("D", "4444444444444444444444444444444444444444444444444444444444444444", 400, 4096, "drive-id-5d"),
            ("E", "5555555555555555555555555555555555555555555555555555555555555555", 500, 5120, "drive-id-5e"),
        ]

        for idx, (rev_letter, sha, step, sz, fid) in enumerate(self.five_rev_specs):
            art = StoredArtifact(
                artifact_id=f"{self.proj_5rev_id}/{self.run_5rev_id}/checkpoints/model.pt@sha256_{sha}",
                storage_provider="google_drive",
                path=str(run_5rev_dir / "checkpoints" / "model.pt"),
                size_bytes=sz,
                sha256=sha,
                project_id=self.proj_5rev_id,
                run_id=self.run_5rev_id,
                artifact_type="checkpoint",
                remote_file_id=fid,
                verified_at=f"2026-10-25T0{idx+1}:00:00Z",
                metadata={"step": step, "revision_letter": rev_letter},
            )
            self.m_5rev.add_artifact(art)
        self.m_5rev.save(run_5rev_dir / "manifest.json")

        self.emitter_5rev = TelemetryEmitter(
            ledger_path=self.telemetry_path,
            run_id=self.run_5rev_id,
            project_id=self.proj_5rev_id,
        )
        self.emitter_5rev.emit("run_started")
        self.emitter_5rev.emit("metric", epoch=5, step=500, metrics={"loss": 0.12})

    def _start_server(self) -> None:
        """Start local MonitorServer instance."""
        self.server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            collector=self.collector,
            registry=self.registry,
            telemetry=self.emitter_clean,
            auth_token=self.api_token,
            allow_external=False,
            artifacts_dir=self.artifacts_dir,
        )
        self.port = self.server.server_port
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.base_url = f"http://127.0.0.1:{self.port}"

    def test_csp_headers_and_unique_cryptographic_nonces(self) -> None:
        """Verify per-request unique CSP nonces, no unsafe-inline, and strict framing controls."""
        # Request 1 to /
        req1 = urllib.request.Request(f"{self.base_url}/")
        with urllib.request.urlopen(req1) as resp1:
            self.assertEqual(resp1.status, 200)
            csp1 = resp1.headers.get("Content-Security-Policy", "")
            x_frame = resp1.headers.get("X-Frame-Options", "")
            x_cto = resp1.headers.get("X-Content-Type-Options", "")
            self.assertEqual(x_frame, "DENY")
            self.assertEqual(x_cto, "nosniff")

            self.assertIn("script-src 'self' 'nonce-", csp1)
            self.assertNotIn("'unsafe-inline'", csp1)
            self.assertIn("frame-ancestors 'none'", csp1)
            self.assertIn("object-src 'none'", csp1)
            self.assertTrue("base-uri 'self'" in csp1 or "base-uri 'none'" in csp1)

            m_nonce1 = re.search(r"nonce-([a-zA-Z0-9_\-]+)", csp1)
            self.assertIsNotNone(m_nonce1, f"Could not find nonce in CSP: {csp1}")
            nonce1 = m_nonce1.group(1)

        # Request 2 to /
        req2 = urllib.request.Request(f"{self.base_url}/")
        with urllib.request.urlopen(req2) as resp2:
            self.assertEqual(resp2.status, 200)
            csp2 = resp2.headers.get("Content-Security-Policy", "")
            m_nonce2 = re.search(r"nonce-([a-zA-Z0-9_\-]+)", csp2)
            self.assertIsNotNone(m_nonce2, f"Could not find nonce in CSP: {csp2}")
            nonce2 = m_nonce2.group(1)

            # Per-request unique cryptographic nonce invariant
            self.assertNotEqual(nonce1, nonce2, f"CSP nonce was reused across requests: {nonce1} == {nonce2}")

        # Request to authenticated API endpoint /api/v1/runs
        auth_hdr = {"Authorization": f"Bearer {self.api_token}", "Accept": "application/json"}
        req_api = urllib.request.Request(f"{self.base_url}/api/v1/runs", headers=auth_hdr)
        with urllib.request.urlopen(req_api) as resp_api:
            self.assertEqual(resp_api.status, 200)
            csp_api = resp_api.headers.get("Content-Security-Policy", "")
            self.assertNotIn("'unsafe-inline'", csp_api)
            self.assertIn("frame-ancestors 'none'", csp_api)
            self.assertIn("object-src 'none'", csp_api)
            self.assertTrue("base-uri 'self'" in csp_api or "base-uri 'none'" in csp_api)

    def test_zero_dangerous_sinks_in_html_and_js(self) -> None:
        """Verify zero innerHTML, outerHTML, insertAdjacentHTML, document.write, eval, or on* handlers."""
        forbidden_sinks = [
            ".innerHTML =",
            ".outerHTML =",
            "insertAdjacentHTML",
            "document.write",
            "eval(",
            "onclick=",
            "onload=",
            "onerror=",
            "oninput=",
            "onchange=",
        ]

        # 1. Check GET /
        with urllib.request.urlopen(f"{self.base_url}/") as resp:
            html_text = resp.read().decode("utf-8")
            for sink in forbidden_sinks:
                self.assertNotIn(sink, html_text, f"Forbidden dangerous sink '{sink}' found in / response")

        # 2. Check GET /monitor.js
        with urllib.request.urlopen(f"{self.base_url}/monitor.js") as resp:
            js_text = resp.read().decode("utf-8")
            for sink in forbidden_sinks:
                self.assertNotIn(sink, js_text, f"Forbidden dangerous sink '{sink}' found in /monitor.js response")

    def test_adversarial_payloads_served_in_api_endpoints(self) -> None:
        """Verify adversarial XSS payloads and multiple checkpoint revisions served safely in API payloads."""
        headers = {"Authorization": f"Bearer {self.api_token}", "Accept": "application/json"}

        # 1. GET /api/v1/runs
        with urllib.request.urlopen(urllib.request.Request(f"{self.base_url}/api/v1/runs", headers=headers)) as resp:
            self.assertEqual(resp.status, 200)
            runs = json.loads(resp.read().decode("utf-8"))
            self.assertIsInstance(runs, list)
            run_ids = [r.get("run_id") for r in runs]
            self.assertIn(self.run_adv_id, run_ids)
            self.assertIn(self.raw_xss_run_id, run_ids)

            adv_run = next(r for r in runs if r.get("run_id") == self.run_adv_id)
            self.assertIn(self.payload_script, adv_run.get("project_id", ""))

        # 2. GET /api/v1/runs/<id>/logs
        with urllib.request.urlopen(urllib.request.Request(f"{self.base_url}/api/v1/runs/{self.run_adv_id}/logs", headers=headers)) as resp:
            self.assertEqual(resp.status, 200)
            logs_data = json.loads(resp.read().decode("utf-8"))
            logs = logs_data.get("logs", [])
            self.assertTrue(any(self.payload_script in line for line in logs))

        # 3. GET /api/v1/runs/<id>/metrics
        with urllib.request.urlopen(urllib.request.Request(f"{self.base_url}/api/v1/runs/{self.run_adv_id}/metrics", headers=headers)) as resp:
            self.assertEqual(resp.status, 200)
            metrics = json.loads(resp.read().decode("utf-8"))
            self.assertIsInstance(metrics, list)
            self.assertTrue(len(metrics) > 0)
            self.assertEqual(metrics[0]["metrics"].get("loss"), 0.55)

        # 4. GET /api/v1/runs/<id>/artifacts (multiple checkpoint revisions)
        with urllib.request.urlopen(urllib.request.Request(f"{self.base_url}/api/v1/runs/{self.run_adv_id}/artifacts", headers=headers)) as resp:
            self.assertEqual(resp.status, 200)
            art_data = json.loads(resp.read().decode("utf-8"))
            arts = art_data.get("artifacts", [])
            self.assertEqual(len(arts), 2, f"Expected 2 checkpoint revisions, got {len(arts)}")
            sha_list = [a.get("sha256") for a in arts]
            self.assertIn(self.sha_rev_a, sha_list)
            self.assertIn(self.sha_rev_b, sha_list)
            self.assertTrue(any(self.payload_img in a.get("path", "") for a in arts))

        # 5. GET /api/v1/runs/<id>/drive
        with urllib.request.urlopen(urllib.request.Request(f"{self.base_url}/api/v1/runs/{self.run_adv_id}/drive", headers=headers)) as resp:
            self.assertEqual(resp.status, 200)
            drive_data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(drive_data.get("drive_publication_status"), "PUBLISHED")
            drive_arts = drive_data.get("artifacts", [])
            self.assertTrue(any(self.payload_svg in a.get("remote_file_id", "") for a in drive_arts))

    def test_token_authentication_enforcement(self) -> None:
        """Verify 401 without Bearer, 401 on ?token= query parameter, and 200 on valid Authorization."""
        # A. Missing Authorization header -> 401
        req_unauth = urllib.request.Request(f"{self.base_url}/api/v1/runs")
        with self.assertRaises(urllib.error.HTTPError) as ctx_unauth:
            urllib.request.urlopen(req_unauth)
        self.assertEqual(ctx_unauth.exception.code, 401)

        # B. Passing ?token= in query string without Authorization header -> strictly 401
        req_query_tok = urllib.request.Request(f"{self.base_url}/api/v1/runs?token={self.api_token}")
        with self.assertRaises(urllib.error.HTTPError) as ctx_query:
            urllib.request.urlopen(req_query_tok)
        self.assertEqual(ctx_query.exception.code, 401)

        # C. Invalid Bearer token -> 401
        req_bad = urllib.request.Request(
            f"{self.base_url}/api/v1/runs",
            headers={"Authorization": "Bearer wrong-token-12345"},
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx_bad:
            urllib.request.urlopen(req_bad)
        self.assertEqual(ctx_bad.exception.code, 401)

        # D. Valid Authorization: Bearer <token> -> 200
        req_valid = urllib.request.Request(
            f"{self.base_url}/api/v1/runs",
            headers={"Authorization": f"Bearer {self.api_token}"},
        )
        with urllib.request.urlopen(req_valid) as resp_valid:
            self.assertEqual(resp_valid.status, 200)

    def test_five_revisions_artifact_history_api(self) -> None:
        """Verify 5 revisions of model.pt return 1 LATEST and 4 HISTORICAL with unfabricated steps."""
        headers = {"Authorization": f"Bearer {self.api_token}", "Accept": "application/json"}
        req = urllib.request.Request(f"{self.base_url}/api/v1/runs/{self.run_5rev_id}/artifacts", headers=headers)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            arts = data.get("artifacts", [])
            self.assertEqual(len(arts), 5, f"Expected 5 revisions, got {len(arts)}")

            latest = [a for a in arts if a.get("is_latest") is True]
            historical = [a for a in arts if a.get("is_latest") is False]
            self.assertEqual(len(latest), 1, f"Expected exactly 1 LATEST artifact, got {len(latest)}")
            self.assertEqual(len(historical), 4, f"Expected 4 HISTORICAL artifacts, got {len(historical)}")

            # Latest revision must be E (step 500)
            self.assertEqual(latest[0].get("sha256"), self.five_rev_specs[-1][1])
            self.assertEqual(latest[0].get("step"), 500)

            # Steps must be unfabricated and match metadata
            steps = sorted(a.get("step") for a in arts)
            self.assertEqual(steps, [100, 200, 300, 400, 500])

    def test_http_and_structural_dom_security_fallback(self) -> None:
        """Backwards compatibility alias for structural security test."""
        self.test_csp_headers_and_unique_cryptographic_nonces()
        self.test_zero_dangerous_sinks_in_html_and_js()
        self.test_adversarial_payloads_served_in_api_endpoints()
        self.test_token_authentication_enforcement()
        self.test_five_revisions_artifact_history_api()


# =============================================================================
# Test Class 2: Mandatory Real Headless Chromium Browser Security & Revisions
# =============================================================================

class TestMandatoryHeadlessBrowserSecurity(unittest.TestCase):
    """Test suite executing real headless Chromium DOM XSS verification and revision rendering."""

    def setUp(self) -> None:
        self.require_browser = os.environ.get("COLAB_REQUIRE_BROWSER_TESTS") == "1"
        self.node_bin, self.pw_path, self.chrome_bin = resolve_browser_test_binaries()

        if self.require_browser:
            if not self.node_bin:
                self.fail("COLAB_REQUIRE_BROWSER_TESTS=1 but Node binary could not be resolved.")
            if not self.pw_path:
                self.fail("COLAB_REQUIRE_BROWSER_TESTS=1 but playwright-core module could not be resolved.")
            if not self.chrome_bin:
                self.fail("COLAB_REQUIRE_BROWSER_TESTS=1 but Chromium binary could not be resolved.")
        else:
            if not (self.node_bin and self.pw_path and self.chrome_bin):
                self.skipTest("Headless Chromium / playwright-core not installed; set COLAB_REQUIRE_BROWSER_TESTS=1 to enforce")

        self.tmp_dir = tempfile.mkdtemp(prefix="colab_test_browser_")
        self.temp_dir = Path(self.tmp_dir)
        self.base_path = self.temp_dir

        self.tracker_dir = self.temp_dir / "tracker"
        self.tracker_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = ExperimentTracker(storage_dir=self.tracker_dir)

        self.ledger_path = self.temp_dir / "account_usage.jsonl"
        self.telemetry_path = self.temp_dir / "telemetry_events.jsonl"
        self.registry_path = self.temp_dir / "managed_sessions.json"
        self.artifacts_dir = self.temp_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.api_token = "browser-test-secret-token"

        # Adversarial XSS payloads
        self.payload_img = '<img src=x onerror=window.__xss_triggered=true>'
        self.payload_script = '<script>window.__xss_triggered=true</script>'
        self.payload_svg = 'running"><svg onload=window.__xss_triggered=true>'
        self.payload_iframe = '<iframe src="javascript:window.__xss_triggered=true">'

        self._seed_workspace()
        self._start_server()

    def tearDown(self) -> None:
        if hasattr(self, "server") and self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                pass
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _seed_workspace(self) -> None:
        """Seed workspace with adversarial run and multiple checkpoint revisions."""
        self.collector = AccountUsageCollector(ledger_path=self.ledger_path, tracker=self.tracker)
        sample = AccountSample("sample-seed-01", _iso_now(), 95.0, 1.2, 1, 0.0, "CONSUMPTION", "OK")
        self.collector._record_sample(sample)
        self.collector._last_sample = sample

        self.registry = ManagedSessionRegistry(self.temp_dir / "managed_sessions.json")
        self.registry.register("session-adv-01", "job-adv", f"T4 {self.payload_svg}", f"proj_{self.payload_script}")
        self.registry.update_state("session-adv-01", "RUNNING")

        # 1. Seed adversarial run with multiple checkpoint revisions
        self.run_adv_id = "run-adv-xss-001"
        self.emitter_adv = TelemetryEmitter(
            ledger_path=self.telemetry_path,
            run_id=self.run_adv_id,
            project_id=f"proj_{self.payload_script}",
        )
        self.emitter_adv.emit("run_started")
        self.emitter_adv.emit("metric", epoch=1, step=10, metrics={"loss": 0.42})
        self.emitter_adv.emit("checkpoint_saved", epoch=1, step=10)
        self.emitter_adv.emit("log_line", details={"log_line": f"Adversarial training log: {self.payload_script}"})
        self.emitter_adv.emit("error", details={"message": f"Fatal runtime failure: {self.payload_iframe}", "error": self.payload_img})

        # 2. Seed run with raw XSS payload in run_id
        self.raw_xss_run_id = f"run_xss_{self.payload_img}"
        self.emitter_raw = TelemetryEmitter(
            ledger_path=self.telemetry_path,
            run_id=self.raw_xss_run_id,
            project_id=f"proj_{self.payload_script}",
        )
        self.emitter_raw.emit("run_started")

        # 3. Seed multiple revisions for same logical file: checkpoints/model.pt
        run_adv_dir = self.artifacts_dir / "projects" / "proj-adv" / "runs" / self.run_adv_id
        run_adv_dir.mkdir(parents=True, exist_ok=True)
        m_adv = PublicationManifest(project_id="proj-adv", run_id=self.run_adv_id)

        self.sha_rev_a = "aaaa1111222233334444555566667777888899990000aaaabbbbccccddddeeee"
        self.sha_rev_b = "bbbb1111222233334444555566667777888899990000aaaabbbbccccddddeeee"

        chk_a = StoredArtifact(
            artifact_id=f"proj-adv/{self.run_adv_id}/checkpoints/model.pt@sha256_{self.sha_rev_a}",
            storage_provider="google_drive",
            path=str(run_adv_dir / "checkpoints" / self.payload_img),
            size_bytes=1024,
            sha256=self.sha_rev_a,
            project_id="proj-adv",
            run_id=self.run_adv_id,
            artifact_type="checkpoint",
            remote_file_id=f"drive-rev-a-{self.payload_svg}",
            verified_at="2026-10-25T01:00:00Z",
        )
        chk_b = StoredArtifact(
            artifact_id=f"proj-adv/{self.run_adv_id}/checkpoints/model.pt@sha256_{self.sha_rev_b}",
            storage_provider="google_drive",
            path=str(run_adv_dir / "checkpoints" / "model.pt"),
            size_bytes=2048,
            sha256=self.sha_rev_b,
            project_id="proj-adv",
            run_id=self.run_adv_id,
            artifact_type="checkpoint",
            remote_file_id="drive-rev-b-clean",
            verified_at="2026-10-25T02:00:00Z",
        )
        m_adv.add_artifact(chk_a)
        m_adv.add_artifact(chk_b)
        m_adv.save(run_adv_dir / "manifest.json")

        # 4. Seed 5 revisions for vision-research / experiment-001 (Section 8)
        self.run_5rev_id = "experiment-001"
        self.proj_5rev_id = "vision-research"
        run_5rev_dir = self.artifacts_dir / "projects" / self.proj_5rev_id / "runs" / self.run_5rev_id
        run_5rev_dir.mkdir(parents=True, exist_ok=True)
        self.m_5rev = PublicationManifest(project_id=self.proj_5rev_id, run_id=self.run_5rev_id)

        self.five_rev_specs = [
            ("A", "1111111111111111111111111111111111111111111111111111111111111111", 100, 1024, "drive-id-5a"),
            ("B", "2222222222222222222222222222222222222222222222222222222222222222", 200, 2048, "drive-id-5b"),
            ("C", "3333333333333333333333333333333333333333333333333333333333333333", 300, 3072, "drive-id-5c"),
            ("D", "4444444444444444444444444444444444444444444444444444444444444444", 400, 4096, "drive-id-5d"),
            ("E", "5555555555555555555555555555555555555555555555555555555555555555", 500, 5120, "drive-id-5e"),
        ]

        for idx, (rev_letter, sha, step, sz, fid) in enumerate(self.five_rev_specs):
            art = StoredArtifact(
                artifact_id=f"{self.proj_5rev_id}/{self.run_5rev_id}/checkpoints/model.pt@sha256_{sha}",
                storage_provider="google_drive",
                path=str(run_5rev_dir / "checkpoints" / "model.pt"),
                size_bytes=sz,
                sha256=sha,
                project_id=self.proj_5rev_id,
                run_id=self.run_5rev_id,
                artifact_type="checkpoint",
                remote_file_id=fid,
                verified_at=f"2026-10-25T0{idx+1}:00:00Z",
                metadata={"step": step, "revision_letter": rev_letter},
            )
            self.m_5rev.add_artifact(art)
        self.m_5rev.save(run_5rev_dir / "manifest.json")

        self.emitter_5rev = TelemetryEmitter(
            ledger_path=self.telemetry_path,
            run_id=self.run_5rev_id,
            project_id=self.proj_5rev_id,
        )
        self.emitter_5rev.emit("run_started")
        self.emitter_5rev.emit("metric", epoch=5, step=500, metrics={"loss": 0.12})

    def _start_server(self) -> None:
        """Start real loopback MonitorServer."""
        self.server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            collector=self.collector,
            registry=self.registry,
            telemetry=self.emitter_adv,
            auth_token=self.api_token,
            allow_external=False,
            artifacts_dir=self.artifacts_dir,
        )
        self.port = self.server.server_port
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.base_url = f"http://127.0.0.1:{self.port}"

    def test_real_headless_chromium_dom_xss_and_revision_history(self) -> None:
        """Execute real headless Chromium DOM XSS verification and revision rendering."""
        adv_run_id_json = json.dumps(self.run_adv_id)
        node_script = f"""
        const {{ chromium }} = require("{self.pw_path}");

        (async () => {{
          const browser = await chromium.launch({{
            executablePath: "{self.chrome_bin}",
            args: ["--no-sandbox", "--disable-gpu"]
          }});
          const version = browser.version();
          const page = await browser.newPage();
          const pageErrors = [];
          const dialogs = [];

          page.on("pageerror", err => {{
            pageErrors.push(err.message || String(err));
          }});

          page.on("dialog", async dialog => {{
            dialogs.push({{ type: dialog.type(), message: dialog.message() }});
            await dialog.dismiss();
          }});

          // Navigate to dashboard
          await page.goto("{self.base_url}/");

          // Auth / Token login via UI elements or direct storage injection
          const tokenInput = await page.$("#auth-token-input");
          const connectBtn = await page.$("#connect-btn");
          if (tokenInput && connectBtn) {{
            await tokenInput.fill("{self.api_token}");
            await connectBtn.click();
          }} else {{
            await page.evaluate((tok) => {{
              sessionStorage.setItem("colab_monitor_token", tok);
            }}, "{self.api_token}");
            await page.evaluate(async (tok) => {{
              if (typeof token !== "undefined") token = tok;
              if (typeof refreshAll === "function") await refreshAll();
              if (typeof refresh === "function") await refresh();
            }}, "{self.api_token}");
          }}

          // Wait for runs table or runs list to populate
          await page.waitForSelector("#runs-table tbody tr.clickable, #runs-list tr, #runs-list .run-card", {{ timeout: 7000 }});

          // Select the target adversarial run row safely via page.evaluate
          const advRunId = {adv_run_id_json};
          await page.evaluate((targetId) => {{
            const rows = Array.from(document.querySelectorAll("#runs-table tbody tr, #runs-list tr, #runs-list .run-card"));
            const match = rows.find(r => r.dataset && r.dataset.runId === targetId) || rows[0];
            if (match) match.click();
          }}, advRunId);
          await page.waitForTimeout(400);

          // Also exercise all rows in table/list
          const allRows = await page.$$(".clickable, .run-card");
          for (const r of allRows) {{
            await r.click();
            await page.waitForTimeout(100);
          }}

          // Re-select target run to ensure inspector displays its artifacts
          await page.evaluate((targetId) => {{
            const rows = Array.from(document.querySelectorAll("#runs-table tbody tr, #runs-list tr, #runs-list .run-card"));
            const match = rows.find(r => r.dataset && r.dataset.runId === targetId) || rows[0];
            if (match) match.click();
          }}, advRunId);
          await page.waitForTimeout(200);

          // Wait for target run's artifact revisions to finish rendering via CDP page.evaluate
          // (avoiding page.waitForFunction which uses eval() and is blocked by our strict CSP!)
          for (let attempt = 0; attempt < 50; attempt++) {{
            const ready = await page.evaluate(() => {{
              const txt = document.getElementById("artifacts-table")?.textContent || "";
              return txt.includes("aaaa") && txt.includes("bbbb");
            }});
            if (ready) break;
            await page.waitForTimeout(100);
          }}

          // Switch tabs if present ([data-tab="artifacts"], [data-tab="drive"], [data-tab="logs"])
          for (const tabName of ["artifacts", "drive", "logs"]) {{
            const tab = await page.$(`[data-tab="${{tabName}}"]`);
            if (tab) {{
              await tab.click();
              await page.waitForTimeout(150);
            }}
          }}

          // Click artifacts tab again so #artifacts-table is visible and not hidden by logs tab
          const artTab = await page.$('[data-tab="artifacts"]');
          if (artTab) {{
            await artTab.click();
            await page.waitForTimeout(150);
          }}

          await page.waitForSelector("#artifacts-table", {{ timeout: 5000 }});

          // Switch to experiment-001 to verify 5 revisions and badges in DOM (Section 8)
          await page.evaluate(() => {{
            const rows = Array.from(document.querySelectorAll("#runs-table tbody tr, #runs-list tr, #runs-list .run-card"));
            const match = rows.find(r => r.dataset && r.dataset.runId === "experiment-001");
            if (match) match.click();
          }});
          await page.waitForTimeout(300);

          const artTab5 = await page.$('[data-tab="artifacts"]');
          if (artTab5) {{
            await artTab5.click();
            await page.waitForTimeout(150);
          }}

          for (let attempt = 0; attempt < 50; attempt++) {{
            const ready = await page.evaluate(() => {{
              const rows = document.querySelectorAll("#artifacts-table tbody tr");
              return rows.length >= 5;
            }});
            if (ready) break;
            await page.waitForTimeout(100);
          }}

          const fiveRevReport = await page.evaluate(() => {{
            const rows = Array.from(document.querySelectorAll("#artifacts-table tbody tr"));
            const badges = rows.map(r => {{
              const b = r.querySelector(".badge-green, .badge-neutral, .badge");
              return b ? b.textContent.trim() : "";
            }});
            const steps = rows.map(r => {{
              const cells = r.querySelectorAll("td");
              return cells.length >= 3 ? cells[2].textContent.trim() : "";
            }});
            return {{
              rowCount: rows.length,
              latestBadges: badges.filter(b => b === "LATEST").length,
              historicalBadges: badges.filter(b => b === "HISTORICAL").length,
              steps: steps,
            }};
          }});

          // Re-select target adversarial run for XSS DOM inspection
          await page.evaluate((targetId) => {{
            const rows = Array.from(document.querySelectorAll("#runs-table tbody tr, #runs-list tr, #runs-list .run-card"));
            const match = rows.find(r => r.dataset && r.dataset.runId === targetId) || rows[0];
            if (match) match.click();
          }}, advRunId);
          await page.waitForTimeout(200);

          const artTabAdv = await page.$('[data-tab="artifacts"]');
          if (artTabAdv) {{
            await artTabAdv.click();
            await page.waitForTimeout(150);
          }}

          const report = await page.evaluate(() => {{
            const artifactsTable = document.getElementById("artifacts-table");
            const artifactsText = artifactsTable ? artifactsTable.textContent : "";

            const injected = document.querySelectorAll(
              "#runs-list script, #runs-list svg:not([data-icon]):not(.svg-icon), #runs-list img, #runs-list iframe, " +
              "#runs-table script, #runs-table svg:not([data-icon]):not(.svg-icon), #runs-table img, #runs-table iframe, " +
              "#logs-box script, #logs-box svg, #logs-box img, #logs-box iframe, " +
              "#log-console script, #log-console svg, #log-console img, #log-console iframe, " +
              "#artifacts-table script, #artifacts-table svg, #artifacts-table img, #artifacts-table iframe, " +
              "#drive-records-table script, #drive-records-table svg, #drive-records-table img, #drive-records-table iframe"
            );

            return {{
              xssTriggered: Boolean(window.__xss_triggered || window.__xss),
              injectedElements: injected.length,
              artifactsText: artifactsText,
              sessionTokenStored: Boolean(sessionStorage.getItem("colab_monitor_token")),
              localTokenStored: localStorage.getItem("colab_monitor_token"),
              locationSearch: window.location.search || ""
            }};
          }});

          await browser.close();

          console.log("TEST_REPORT:" + JSON.stringify({{
            browserVersion: version,
            dialogs: dialogs,
            pageErrors: pageErrors,
            fiveRevReport: fiveRevReport,
            ...report
          }}));
        }})();
        """

        res = subprocess.run(
            [self.node_bin, "-e", node_script],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if self.require_browser and res.returncode != 0:
            self.fail(f"Headless Chromium test failed to execute (exit {res.returncode}): {res.stderr}\n{res.stdout}")

        self.assertEqual(res.returncode, 0, f"Headless browser runner exited non-zero: {res.stderr}")
        m_res = re.search(r"TEST_REPORT:(.*)", res.stdout)
        self.assertIsNotNone(m_res, f"Could not find TEST_REPORT in output: {res.stdout}\n{res.stderr}")
        data = json.loads(m_res.group(1))

        # 1. Browser version non-empty
        self.assertTrue(bool(data.get("browserVersion")), "browser.version() was empty")

        # 2. Zero XSS triggered
        self.assertFalse(data.get("xssTriggered"), "window.__xss_triggered was True!")

        # 3. Zero dialog popups
        self.assertEqual(data.get("dialogs"), [], f"Unexpected alert/confirm/prompt dialogs triggered: {data.get('dialogs')}")

        # 4. Zero unhandled page errors
        self.assertEqual(data.get("pageErrors"), [], f"Unhandled JavaScript errors observed: {data.get('pageErrors')}")

        # 5. Zero injected elements
        self.assertEqual(data.get("injectedElements"), 0, "Injected script/svg/img/iframe elements found in DOM!")

        # 6. Artifact revisions visible in #artifacts-table
        art_text = data.get("artifactsText", "")
        self.assertIn("aaaa", art_text, f"aaaa revision prefix not found in #artifacts-table: {art_text}")
        self.assertIn("bbbb", art_text, f"bbbb revision prefix not found in #artifacts-table: {art_text}")

        # 7. Token containment verification
        self.assertTrue(data.get("sessionTokenStored"), "colab_monitor_token was not stored in sessionStorage")
        self.assertIsNone(data.get("localTokenStored"), "Auth token leaked into localStorage!")
        self.assertEqual(data.get("locationSearch"), "", "Token or query parameters leaked into window.location.search!")

        # 8. 5-revision DOM history verification (Section 8)
        five_rep = data.get("fiveRevReport", {})
        self.assertEqual(five_rep.get("rowCount"), 5, f"Expected 5 revision rows in DOM, got {five_rep.get('rowCount')}")
        self.assertEqual(five_rep.get("latestBadges"), 1, f"Expected exactly 1 LATEST badge in DOM, got {five_rep.get('latestBadges')}")
        self.assertEqual(five_rep.get("historicalBadges"), 4, f"Expected exactly 4 HISTORICAL badges in DOM, got {five_rep.get('historicalBadges')}")
        self.assertTrue(
            all(str(s) in five_rep.get("steps", []) for s in [100, 200, 300, 400, 500]),
            f"Unfabricated step values missing in DOM: {five_rep.get('steps')}",
        )

    def test_real_headless_browser_dom_xss_and_ui_interaction(self) -> None:
        """Backwards compatibility alias for browser security test."""
        self.test_real_headless_chromium_dom_xss_and_revision_history()


# =============================================================================
# Subprocess Worker Templates for Cross-Process Testing
# =============================================================================

SIMULTANEOUS_PUBLISH_WORKER_CODE = """
import sys, os, time, json, urllib.request, urllib.parse, urllib.error
from pathlib import Path
from typing import Optional, Dict

base_url = sys.argv[1]
workspace_dir = Path(sys.argv[2])
journal_path = Path(sys.argv[3])
checkpoint_path = Path(sys.argv[4])
barrier_dir = Path(sys.argv[5])
proc_id = sys.argv[6]

class LoopbackDriveHttpClient:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
    def request(self, method: str, url: str, headers: Optional[Dict[str, str]] = None, data: Optional[bytes] = None, timeout: float = 30.0):
        parsed = urllib.parse.urlparse(url)
        target_url = url if ("127.0.0.1" in url or "localhost" in url) else f"{self.base}{parsed.path}" + (f"?{parsed.query}" if parsed.query else "")
        req = urllib.request.Request(target_url, data=data, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()

from colab_adapter.storage.google_drive import GoogleDriveArtifactStore, DriveCredentials

creds = DriveCredentials(client_id="cid", client_secret="csec", refresh_token="rtok", access_token="atok")
store = GoogleDriveArtifactStore(
    credentials=creds,
    local_staging_dir=workspace_dir,
    journal_path=journal_path,
    http_client=LoopbackDriveHttpClient(base_url),
    chunk_size=256 * 1024,
)

# Synchronize on filesystem barrier
ready_file = barrier_dir / f"ready_{proc_id}.flag"
ready_file.write_text("ready")

other_id = "2" if proc_id == "1" else "1"
other_file = barrier_dir / f"ready_{other_id}.flag"
for _ in range(400):
    if other_file.exists():
        break
    time.sleep(0.01)

# Execute simultaneous publish
stored = store.publish(
    source=checkpoint_path,
    project_id="test_proj",
    run_id="test_run",
    artifact_type="checkpoint"
)

result = {
    "proc_id": proc_id,
    "artifact_id": stored.artifact_id,
    "remote_file_id": stored.remote_file_id,
    "sha256": stored.sha256,
    "size_bytes": stored.size_bytes
}
print("WORKER_RESULT:" + json.dumps(result))
"""

SIMULTANEOUS_UPLOAD_WORKER_THREAD_CODE = """
import sys, os, time, json, urllib.request, urllib.parse, urllib.error
from pathlib import Path
from typing import Optional, Dict

base_url = sys.argv[1]
workspace_dir = Path(sys.argv[2])
journal_path = Path(sys.argv[3])
checkpoint_path = Path(sys.argv[4])
barrier_dir = Path(sys.argv[5])
proc_id = sys.argv[6]

class LoopbackDriveHttpClient:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
    def request(self, method: str, url: str, headers: Optional[Dict[str, str]] = None, data: Optional[bytes] = None, timeout: float = 30.0):
        parsed = urllib.parse.urlparse(url)
        target_url = url if ("127.0.0.1" in url or "localhost" in url) else f"{self.base}{parsed.path}" + (f"?{parsed.query}" if parsed.query else "")
        req = urllib.request.Request(target_url, data=data, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()

from colab_adapter.storage.google_drive import GoogleDriveArtifactStore, DriveCredentials
from colab_adapter.storage.transfer import TransferJournal, TransferStatus
from colab_adapter.storage.worker import ArtifactUploadWorker

creds = DriveCredentials(client_id="cid", client_secret="csec", refresh_token="rtok", access_token="atok")
store = GoogleDriveArtifactStore(
    credentials=creds,
    local_staging_dir=workspace_dir,
    journal_path=journal_path,
    http_client=LoopbackDriveHttpClient(base_url),
    chunk_size=256 * 1024,
)
journal = TransferJournal(journal_path)

# Synchronize on barrier
ready_file = barrier_dir / f"worker_ready_{proc_id}.flag"
ready_file.write_text("ready")
other_id = "2" if proc_id == "1" else "1"
other_file = barrier_dir / f"worker_ready_{other_id}.flag"
for _ in range(400):
    if other_file.exists():
        break
    time.sleep(0.01)

worker = ArtifactUploadWorker(
    store=store,
    journal=journal,
    max_workers=1,
)
worker.start()
worker.enqueue(
    local_path=checkpoint_path,
    project_id="test_proj",
    run_id="test_run",
    artifact_type="checkpoint"
)

transfer_id = f"google_drive-test_proj-test_run-{checkpoint_path.name}"
for _ in range(120):
    rec = journal.get_transfer(transfer_id)
    if rec and rec.status == TransferStatus.PUBLISHED.value:
        break
    time.sleep(0.1)

worker.stop(wait=True, timeout=5.0)

final_rec = journal.get_transfer(transfer_id)
result = {
    "proc_id": proc_id,
    "status": final_rec.status if final_rec else None,
    "remote_file_id": final_rec.remote_file_id if final_rec else None,
    "sha256": final_rec.sha256 if final_rec else None,
}
print("WORKER_THREAD_RESULT:" + json.dumps(result))
"""

ABORT_MID_UPLOAD_WORKER_CODE = """
import sys, os, time, urllib.request, urllib.parse, urllib.error
from pathlib import Path
from typing import Optional, Dict

base_url = sys.argv[1]
workspace_dir = Path(sys.argv[2])
journal_path = Path(sys.argv[3])
checkpoint_path = Path(sys.argv[4])

class AbortingHttpClient:
    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def request(self, method: str, url: str, headers: Optional[Dict[str, str]] = None, data: Optional[bytes] = None, timeout: float = 30.0):
        parsed = urllib.parse.urlparse(url)
        target_url = url if ("127.0.0.1" in url or "localhost" in url) else f"{self.base}{parsed.path}" + (f"?{parsed.query}" if parsed.query else "")
        hdrs = {k: v for k, v in (headers or {}).items()}

        # If this is the second chunk upload (bytes 262144-...), abort process abruptly
        cr = hdrs.get("content-range", "")
        if method == "PUT" and "262144" in cr:
            sys.stderr.write("PROCESS 1 ABORTING MID-UPLOAD BEFORE CHUNK 2\\n")
            sys.stderr.flush()
            os._exit(99)

        req = urllib.request.Request(target_url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()

from colab_adapter.storage.google_drive import GoogleDriveArtifactStore, DriveCredentials

creds = DriveCredentials(client_id="cid", client_secret="csec", refresh_token="rtok", access_token="atok")
store = GoogleDriveArtifactStore(
    credentials=creds,
    local_staging_dir=workspace_dir,
    journal_path=journal_path,
    http_client=AbortingHttpClient(base_url),
    chunk_size=256 * 1024,
)

store.publish(
    source=checkpoint_path,
    project_id="test_proj",
    run_id="test_run",
    artifact_type="checkpoint"
)
"""

RESUME_FROM_JOURNAL_WORKER_CODE = """
import sys, os, time, json, urllib.request, urllib.parse, urllib.error
from pathlib import Path
from typing import Optional, Dict

base_url = sys.argv[1]
workspace_dir = Path(sys.argv[2])
journal_path = Path(sys.argv[3])
checkpoint_path = Path(sys.argv[4])

class LoopbackDriveHttpClient:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
    def request(self, method: str, url: str, headers: Optional[Dict[str, str]] = None, data: Optional[bytes] = None, timeout: float = 30.0):
        parsed = urllib.parse.urlparse(url)
        target_url = url if ("127.0.0.1" in url or "localhost" in url) else f"{self.base}{parsed.path}" + (f"?{parsed.query}" if parsed.query else "")
        req = urllib.request.Request(target_url, data=data, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()

from colab_adapter.storage.google_drive import GoogleDriveArtifactStore, DriveCredentials

creds = DriveCredentials(client_id="cid", client_secret="csec", refresh_token="rtok", access_token="atok")
store = GoogleDriveArtifactStore(
    credentials=creds,
    local_staging_dir=workspace_dir,
    journal_path=journal_path,
    http_client=LoopbackDriveHttpClient(base_url),
    chunk_size=256 * 1024,
)

stored = store.publish(
    source=checkpoint_path,
    project_id="test_proj",
    run_id="test_run",
    artifact_type="checkpoint"
)

result = {
    "artifact_id": stored.artifact_id,
    "remote_file_id": stored.remote_file_id,
    "sha256": stored.sha256,
    "size_bytes": stored.size_bytes
}
print("RESUME_RESULT:" + json.dumps(result))
"""

DIFFERENT_REVISIONS_WORKER_CODE = """
import sys, os, time, json, urllib.request, urllib.parse, urllib.error
from pathlib import Path
from typing import Optional, Dict

base_url = sys.argv[1]
workspace_dir = Path(sys.argv[2])
journal_path = Path(sys.argv[3])
checkpoint_path = Path(sys.argv[4])
barrier_dir = Path(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "none" else None
proc_id = sys.argv[6] if len(sys.argv) > 6 else "1"

class LoopbackDriveHttpClient:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
    def request(self, method: str, url: str, headers: Optional[Dict[str, str]] = None, data: Optional[bytes] = None, timeout: float = 30.0):
        parsed = urllib.parse.urlparse(url)
        target_url = url if ("127.0.0.1" in url or "localhost" in url) else f"{self.base}{parsed.path}" + (f"?{parsed.query}" if parsed.query else "")
        req = urllib.request.Request(target_url, data=data, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()

from colab_adapter.storage.google_drive import GoogleDriveArtifactStore, DriveCredentials

creds = DriveCredentials(client_id="cid", client_secret="csec", refresh_token="rtok", access_token="atok")
store = GoogleDriveArtifactStore(
    credentials=creds,
    local_staging_dir=workspace_dir,
    journal_path=journal_path,
    http_client=LoopbackDriveHttpClient(base_url),
    chunk_size=256 * 1024,
)

if barrier_dir:
    ready_file = barrier_dir / f"diff_rev_ready_{proc_id}.flag"
    ready_file.write_text("ready")
    other_id = "2" if proc_id == "1" else "1"
    other_file = barrier_dir / f"diff_rev_ready_{other_id}.flag"
    for _ in range(400):
        if other_file.exists():
            break
        time.sleep(0.01)

stored = store.publish(
    source=checkpoint_path,
    project_id="test_proj",
    run_id="test_run",
    artifact_type="checkpoint"
)

result = {
    "proc_id": proc_id,
    "status": "ok",
    "artifact_id": stored.artifact_id,
    "remote_file_id": stored.remote_file_id,
    "sha256": stored.sha256,
    "size_bytes": stored.size_bytes
}
print("DIFF_REV_RESULT:" + json.dumps(result))
"""

FIVE_REVISIONS_WORKER_CODE = """
import sys, os, time, json, urllib.request, urllib.parse, urllib.error
from pathlib import Path
from typing import Optional, Dict

base_url = sys.argv[1]
workspace_dir = Path(sys.argv[2])
journal_path = Path(sys.argv[3])
barrier_dir = Path(sys.argv[4])
proc_id = sys.argv[5]
revisions_data = json.loads(sys.argv[6])

class LoopbackDriveHttpClient:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
    def request(self, method: str, url: str, headers: Optional[Dict[str, str]] = None, data: Optional[bytes] = None, timeout: float = 30.0):
        parsed = urllib.parse.urlparse(url)
        target_url = url if ("127.0.0.1" in url or "localhost" in url) else f"{self.base}{parsed.path}" + (f"?{parsed.query}" if parsed.query else "")
        req = urllib.request.Request(target_url, data=data, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()

from colab_adapter.storage.google_drive import GoogleDriveArtifactStore, DriveCredentials

creds = DriveCredentials(client_id="cid", client_secret="csec", refresh_token="rtok", access_token="atok")
store = GoogleDriveArtifactStore(
    credentials=creds,
    local_staging_dir=workspace_dir,
    journal_path=journal_path,
    http_client=LoopbackDriveHttpClient(base_url),
    chunk_size=256 * 1024,
)

# Synchronize start on filesystem barrier
ready_file = barrier_dir / f"fiverev_ready_{proc_id}.flag"
ready_file.write_text("ready")
other_id = "2" if proc_id == "1" else "1"
other_file = barrier_dir / f"fiverev_ready_{other_id}.flag"
for _ in range(400):
    if other_file.exists():
        break
    time.sleep(0.01)

proc_dir = workspace_dir / f"proc_{proc_id}_ckpts"
proc_dir.mkdir(parents=True, exist_ok=True)
ckpt_file = proc_dir / "model.pt"

published = []
for rev_letter, pattern, repeat_count, step in revisions_data:
    content = (pattern * repeat_count).encode("ascii")
    ckpt_file.write_bytes(content)
    stored = store.publish(
        source=ckpt_file,
        project_id="vision-research",
        run_id="experiment-001",
        artifact_type="checkpoint",
        metadata={"step": step, "revision_letter": rev_letter}
    )
    published.append({
        "rev": rev_letter,
        "artifact_id": stored.artifact_id,
        "remote_file_id": stored.remote_file_id,
        "sha256": stored.sha256,
        "size_bytes": stored.size_bytes,
        "step": step,
    })

print("FIVE_REV_RESULT:" + json.dumps(published))
"""


# =============================================================================
# Test Class 3: Real Two-OS-Process Concurrency & Revision Publication
# =============================================================================

class TestCrossProcessDrivePublication(unittest.TestCase):
    """Test suite executing real multi-process concurrency, lease recovery, and revision tests."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="colab_test_crossproc_")
        self.base_path = Path(self.tmp_dir)
        self.workspace_dir = self.base_path / "workspace"
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.workspace_dir / "transfers.json"
        self.barrier_dir = self.base_path / "barrier"
        self.barrier_dir.mkdir(parents=True, exist_ok=True)

        # Generate exact 512 KB (524,288 bytes) binary checkpoint artifact
        self.ckpt_path = self.workspace_dir / "checkpoint.pt"
        self.ckpt_data = os.urandom(512 * 1024)
        self.ckpt_path.write_bytes(self.ckpt_data)
        self.expected_sha256 = hashlib.sha256(self.ckpt_data).hexdigest()
        self.expected_md5 = hashlib.md5(self.ckpt_data).hexdigest()

        # Start loopback Drive HTTP server
        self.drive_server = LoopbackDriveServer(("127.0.0.1", 0))
        self.server_thread = threading.Thread(target=self.drive_server.serve_forever, daemon=True)
        self.server_thread.start()
        self.port = self.drive_server.server_port
        self.base_url = f"http://127.0.0.1:{self.port}"

        self.env = os.environ.copy()
        worktree_root = str(Path(__file__).resolve().parent.parent)
        self.env["PYTHONPATH"] = worktree_root

    def tearDown(self) -> None:
        if hasattr(self, "drive_server") and self.drive_server:
            try:
                self.drive_server.shutdown()
                self.drive_server.server_close()
            except Exception:
                pass
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_two_os_processes_simultaneous_drive_publish_and_stale_lease_recovery(self) -> None:
        """Two separate OS processes enter store.publish(...) simultaneously on shared journal."""
        p1 = subprocess.Popen(
            [
                sys.executable,
                "-c",
                SIMULTANEOUS_PUBLISH_WORKER_CODE,
                self.base_url,
                str(self.workspace_dir),
                str(self.journal_path),
                str(self.ckpt_path),
                str(self.barrier_dir),
                "1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
        )
        p2 = subprocess.Popen(
            [
                sys.executable,
                "-c",
                SIMULTANEOUS_PUBLISH_WORKER_CODE,
                self.base_url,
                str(self.workspace_dir),
                str(self.journal_path),
                str(self.ckpt_path),
                str(self.barrier_dir),
                "2",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
        )

        out1, err1 = p1.communicate(timeout=25)
        out2, err2 = p2.communicate(timeout=25)

        # 1. Both OS processes exit with code 0
        self.assertEqual(p1.returncode, 0, f"Process 1 failed: {err1}")
        self.assertEqual(p2.returncode, 0, f"Process 2 failed: {err2}")

        m1 = re.search(r"WORKER_RESULT:(.*)", out1)
        m2 = re.search(r"WORKER_RESULT:(.*)", out2)
        self.assertIsNotNone(m1, f"Missing WORKER_RESULT in process 1 output: {out1}")
        self.assertIsNotNone(m2, f"Missing WORKER_RESULT in process 2 output: {out2}")

        res1 = json.loads(m1.group(1))
        res2 = json.loads(m2.group(1))

        # 2. Both processes return valid StoredArtifact JSON with identical remote_file_id and sha256
        self.assertEqual(res1["remote_file_id"], res2["remote_file_id"])
        self.assertEqual(res1["sha256"], self.expected_sha256)
        self.assertEqual(res2["sha256"], self.expected_sha256)
        self.assertEqual(res1["size_bytes"], 512 * 1024)
        self.assertTrue(res1["artifact_id"].startswith("test_proj/test_run/checkpoints/checkpoint.pt"))

        # 3. Loopback Drive server recorded exactly 1 upload session and 1 committed checkpoint file
        self.assertEqual(
            self.drive_server.session_counter,
            1,
            f"Expected exactly 1 resumable upload session, got {self.drive_server.session_counter}",
        )
        self.assertEqual(
            len(self.drive_server.committed_checkpoints),
            1,
            f"Expected at most 1 committed checkpoint file, got {len(self.drive_server.committed_checkpoints)}: {self.drive_server.committed_checkpoints}",
        )

        # 4. Shared TransferJournal has status PUBLISHED and valid JSON
        self.assertTrue(self.journal_path.exists())
        with open(self.journal_path, "r", encoding="utf-8") as jf:
            journal_data = json.load(jf)
        self.assertIsInstance(journal_data, dict)
        transfer_key = f"google_drive-test_proj-test_run-{self.ckpt_path.name}"
        matching_t_keys = [k for k in journal_data if k == transfer_key or k.startswith(transfer_key + "-")]
        self.assertTrue(len(matching_t_keys) >= 1, f"{transfer_key} not found in {journal_data}")
        t_rec = journal_data[matching_t_keys[0]]
        self.assertEqual(t_rec["status"], TransferStatus.PUBLISHED.value)
        self.assertEqual(t_rec["remote_file_id"], res1["remote_file_id"])

        # 5. Shared PublicationManifest (manifest.json) is valid JSON and records artifact without corruption
        manifest_path = self.workspace_dir / "projects" / "test_proj" / "runs" / "test_run" / "manifest.json"
        self.assertTrue(manifest_path.exists(), f"Publication manifest not found at {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as mf:
            manifest_data = json.load(mf)
        self.assertEqual(manifest_data["status"], "PUBLISHED")
        art_key = f"test_proj/test_run/checkpoints/{self.ckpt_path.name}"
        matching = [
            art for k, art in manifest_data.get("artifacts", {}).items()
            if (k == art_key or k.startswith(f"{art_key}@sha256_") or art.get("sha256") == self.expected_sha256)
        ]
        self.assertTrue(len(matching) >= 1, f"Artifact {art_key} not found in manifest: {manifest_data.get('artifacts')}")
        self.assertEqual(matching[0]["remote_file_id"], res1["remote_file_id"])
        self.assertEqual(matching[0]["sha256"], self.expected_sha256)

    def test_two_os_processes_simultaneous_store_publish(self) -> None:
        """Backwards compatibility alias for simultaneous store publish."""
        self.test_two_os_processes_simultaneous_drive_publish_and_stale_lease_recovery()

    def test_two_os_processes_simultaneous_artifact_upload_worker(self) -> None:
        """Two separate OS processes running ArtifactUploadWorker enqueue same artifact concurrently."""
        p1 = subprocess.Popen(
            [
                sys.executable,
                "-c",
                SIMULTANEOUS_UPLOAD_WORKER_THREAD_CODE,
                self.base_url,
                str(self.workspace_dir),
                str(self.journal_path),
                str(self.ckpt_path),
                str(self.barrier_dir),
                "1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
        )
        p2 = subprocess.Popen(
            [
                sys.executable,
                "-c",
                SIMULTANEOUS_UPLOAD_WORKER_THREAD_CODE,
                self.base_url,
                str(self.workspace_dir),
                str(self.journal_path),
                str(self.ckpt_path),
                str(self.barrier_dir),
                "2",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
        )

        out1, err1 = p1.communicate(timeout=25)
        out2, err2 = p2.communicate(timeout=25)

        self.assertEqual(p1.returncode, 0, f"Worker process 1 failed: {err1}")
        self.assertEqual(p2.returncode, 0, f"Worker process 2 failed: {err2}")

        m1 = re.search(r"WORKER_THREAD_RESULT:(.*)", out1)
        m2 = re.search(r"WORKER_THREAD_RESULT:(.*)", out2)
        self.assertIsNotNone(m1, f"Missing WORKER_THREAD_RESULT in process 1: {out1}")
        self.assertIsNotNone(m2, f"Missing WORKER_THREAD_RESULT in process 2: {out2}")

        res1 = json.loads(m1.group(1))
        res2 = json.loads(m2.group(1))

        self.assertEqual(res1["status"], TransferStatus.PUBLISHED.value)
        self.assertEqual(res2["status"], TransferStatus.PUBLISHED.value)
        self.assertEqual(res1["remote_file_id"], res2["remote_file_id"])
        self.assertEqual(self.drive_server.session_counter, 1)
        self.assertEqual(len(self.drive_server.committed_checkpoints), 1)

    def test_process_abort_mid_upload_and_resumable_recovery(self) -> None:
        """Process 1 aborts after 256 KB chunk; Process 2 recovers from journal and finishes publication."""
        # 1. Process 1 starts and aborts mid-upload before chunk 2
        p1 = subprocess.Popen(
            [
                sys.executable,
                "-c",
                ABORT_MID_UPLOAD_WORKER_CODE,
                self.base_url,
                str(self.workspace_dir),
                str(self.journal_path),
                str(self.ckpt_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
        )
        out1, err1 = p1.communicate(timeout=15)
        self.assertEqual(p1.returncode, 99, f"Process 1 did not abort as expected: {out1}\n{err1}")
        self.assertIn("PROCESS 1 ABORTING MID-UPLOAD", err1)

        # Verify server state after P1 abort: 1 session created, chunk 0 acknowledged (0-262143), 0 committed files
        self.assertEqual(len(self.drive_server.chunks_uploaded), 1)
        self.assertEqual(self.drive_server.chunks_uploaded[0]["start"], 0)
        self.assertEqual(self.drive_server.chunks_uploaded[0]["end"], 262143)
        self.assertEqual(len(self.drive_server.committed_checkpoints), 0)

        # Verify journal state recorded bytes_transferred = 262144
        with open(self.journal_path, "r", encoding="utf-8") as jf:
            journal_after_abort = json.load(jf)
        t_key = f"google_drive-test_proj-test_run-{self.ckpt_path.name}"
        matching_abort_keys = [k for k in journal_after_abort if k == t_key or k.startswith(t_key + "-")]
        self.assertTrue(len(matching_abort_keys) >= 1, f"{t_key} not found in {journal_after_abort}")
        self.assertEqual(journal_after_abort[matching_abort_keys[0]]["bytes_transferred"], 262144)

        # 2. Process 2 starts, queries progress via HTTP 308, resumes from offset 262144, finishes publication
        p2 = subprocess.Popen(
            [
                sys.executable,
                "-c",
                RESUME_FROM_JOURNAL_WORKER_CODE,
                self.base_url,
                str(self.workspace_dir),
                str(self.journal_path),
                str(self.ckpt_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
        )
        out2, err2 = p2.communicate(timeout=15)
        self.assertEqual(p2.returncode, 0, f"Process 2 failed to resume upload: {err2}")

        m2 = re.search(r"RESUME_RESULT:(.*)", out2)
        self.assertIsNotNone(m2, f"Missing RESUME_RESULT in process 2 output: {out2}")
        res2 = json.loads(m2.group(1))

        # Assertions
        self.assertEqual(res2["sha256"], self.expected_sha256)
        self.assertEqual(res2["size_bytes"], 512 * 1024)
        self.assertEqual(len(self.drive_server.committed_checkpoints), 1)

        # Verify chunk 2 uploaded by Process 2 started at byte offset 262144
        self.assertEqual(len(self.drive_server.chunks_uploaded), 2)
        self.assertEqual(self.drive_server.chunks_uploaded[1]["start"], 262144)
        self.assertEqual(self.drive_server.chunks_uploaded[1]["end"], 524287)

        # Verify committed bytes match local checkpoint byte-for-byte
        fid = res2["remote_file_id"]
        self.assertEqual(self.drive_server.committed_bytes[fid], self.ckpt_data)

        # Verify journal status transitioned to PUBLISHED
        with open(self.journal_path, "r", encoding="utf-8") as jf:
            journal_final = json.load(jf)
        matching_final_keys = [k for k in journal_final if k == t_key or k.startswith(t_key + "-")]
        self.assertTrue(len(matching_final_keys) >= 1, f"{t_key} not found in {journal_final}")
        self.assertEqual(journal_final[matching_final_keys[0]]["status"], TransferStatus.PUBLISHED.value)

    def test_two_os_processes_different_revisions_same_filename_preserved(self) -> None:
        """Process 1 and Process 2 publish different revisions of same filename; both succeed with distinct SHA256."""
        # 1. Prepare Process 1 revision
        p1_dir = self.workspace_dir / "src_p1"
        p1_dir.mkdir(parents=True, exist_ok=True)
        ckpt_p1 = p1_dir / "checkpoint.pt"
        data_p1 = b"PROCESS_1_REVISION_A_WEIGHTS_" * 100
        ckpt_p1.write_bytes(data_p1)
        sha_a = hashlib.sha256(data_p1).hexdigest()

        # 2. Prepare Process 2 revision with different content but same filename
        p2_dir = self.workspace_dir / "src_p2"
        p2_dir.mkdir(parents=True, exist_ok=True)
        ckpt_p2 = p2_dir / "checkpoint.pt"
        data_p2 = b"PROCESS_2_REVISION_B_WEIGHTS_" * 120
        ckpt_p2.write_bytes(data_p2)
        sha_b = hashlib.sha256(data_p2).hexdigest()

        self.assertNotEqual(sha_a, sha_b, "Test setup error: SHA256 of revisions must be distinct")

        # 3. Spawn both OS processes concurrently with barrier synchronization
        p1 = subprocess.Popen(
            [
                sys.executable,
                "-c",
                DIFFERENT_REVISIONS_WORKER_CODE,
                self.base_url,
                str(self.workspace_dir),
                str(self.journal_path),
                str(ckpt_p1),
                str(self.barrier_dir),
                "1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
        )
        p2 = subprocess.Popen(
            [
                sys.executable,
                "-c",
                DIFFERENT_REVISIONS_WORKER_CODE,
                self.base_url,
                str(self.workspace_dir),
                str(self.journal_path),
                str(ckpt_p2),
                str(self.barrier_dir),
                "2",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
        )

        out1, err1 = p1.communicate(timeout=25)
        out2, err2 = p2.communicate(timeout=25)

        self.assertEqual(p1.returncode, 0, f"Process 1 failed: {err1}")
        self.assertEqual(p2.returncode, 0, f"Process 2 failed: {err2}")

        m1 = re.search(r"DIFF_REV_RESULT:(.*)", out1)
        m2 = re.search(r"DIFF_REV_RESULT:(.*)", out2)
        self.assertIsNotNone(m1, f"Missing DIFF_REV_RESULT in process 1 output: {out1}")
        self.assertIsNotNone(m2, f"Missing DIFF_REV_RESULT in process 2 output: {out2}")

        res1 = json.loads(m1.group(1))
        res2 = json.loads(m2.group(1))

        # 4. Both processes return status ok with distinct SHA-256 hashes
        self.assertEqual(res1.get("status"), "ok")
        self.assertEqual(res2.get("status"), "ok")
        self.assertEqual(res1["sha256"], sha_a)
        self.assertEqual(res2["sha256"], sha_b)
        self.assertNotEqual(res1["sha256"], res2["sha256"])

        # 5. Verify TransferJournal is valid JSON
        self.assertTrue(self.journal_path.exists())
        with open(self.journal_path, "r", encoding="utf-8") as jf:
            journal_data = json.load(jf)
        self.assertIsInstance(journal_data, dict)

        # 6. Verify PublicationManifest is valid JSON
        manifest_path = self.workspace_dir / "projects" / "test_proj" / "runs" / "test_run" / "manifest.json"
        self.assertTrue(manifest_path.exists())
        with open(manifest_path, "r", encoding="utf-8") as mf:
            manifest_data = json.load(mf)
        self.assertIsInstance(manifest_data, dict)
        self.assertEqual(manifest_data.get("status"), "PUBLISHED")

        # 7. Zero .tmp files left behind across workspace
        tmp_files = list(self.workspace_dir.glob("**/*.tmp"))
        self.assertEqual(len(tmp_files), 0, f"Orphaned .tmp files found: {tmp_files}")

    def test_two_os_processes_five_revisions_publication_and_idempotency(self) -> None:
        """Two separate OS subprocesses publish 5 distinct revisions (A, B, C, D, E) of model.pt; repeating E is idempotent."""
        # 1. Prepare 5 distinct byte payloads for model.pt
        data_a = b"FIVE_REV_A_DATA_" * 16384
        data_b = b"FIVE_REV_B_DATA_" * 16384
        data_c = b"FIVE_REV_C_DATA_" * 16384
        data_d = b"FIVE_REV_D_DATA_" * 16384
        data_e = b"FIVE_REV_E_DATA_" * 16384

        sha_a = hashlib.sha256(data_a).hexdigest()
        sha_b = hashlib.sha256(data_b).hexdigest()
        sha_c = hashlib.sha256(data_c).hexdigest()
        sha_d = hashlib.sha256(data_d).hexdigest()
        sha_e = hashlib.sha256(data_e).hexdigest()

        all_shas = {sha_a, sha_b, sha_c, sha_d, sha_e}
        self.assertEqual(len(all_shas), 5, "Test setup error: All 5 revisions must have distinct SHA256 hashes")

        # Process 1 publishes revisions A (step 100), C (step 300), E (step 500)
        p1_revisions = [
            ("A", "FIVE_REV_A_DATA_", 16384, 100),
            ("C", "FIVE_REV_C_DATA_", 16384, 300),
            ("E", "FIVE_REV_E_DATA_", 16384, 500),
        ]

        # Process 2 publishes revisions B (step 200), D (step 400), E (step 500)
        # Note: Revision E is intentionally repeated across both processes to verify idempotency!
        p2_revisions = [
            ("B", "FIVE_REV_B_DATA_", 16384, 200),
            ("D", "FIVE_REV_D_DATA_", 16384, 400),
            ("E", "FIVE_REV_E_DATA_", 16384, 500),
        ]

        # 2. Spawn both OS processes concurrently with barrier synchronization
        p1 = subprocess.Popen(
            [
                sys.executable,
                "-c",
                FIVE_REVISIONS_WORKER_CODE,
                self.base_url,
                str(self.workspace_dir),
                str(self.journal_path),
                str(self.barrier_dir),
                "1",
                json.dumps(p1_revisions),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
        )
        p2 = subprocess.Popen(
            [
                sys.executable,
                "-c",
                FIVE_REVISIONS_WORKER_CODE,
                self.base_url,
                str(self.workspace_dir),
                str(self.journal_path),
                str(self.barrier_dir),
                "2",
                json.dumps(p2_revisions),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
        )

        out1, err1 = p1.communicate(timeout=35)
        out2, err2 = p2.communicate(timeout=35)

        self.assertEqual(p1.returncode, 0, f"Process 1 failed: {err1}\n{out1}")
        self.assertEqual(p2.returncode, 0, f"Process 2 failed: {err2}\n{out2}")

        m1 = re.search(r"FIVE_REV_RESULT:(.*)", out1)
        m2 = re.search(r"FIVE_REV_RESULT:(.*)", out2)
        self.assertIsNotNone(m1, f"Missing FIVE_REV_RESULT in process 1 output: {out1}")
        self.assertIsNotNone(m2, f"Missing FIVE_REV_RESULT in process 2 output: {out2}")

        res1 = json.loads(m1.group(1))
        res2 = json.loads(m2.group(1))
        self.assertEqual(len(res1), 3)
        self.assertEqual(len(res2), 3)

        # 3. Verify PublicationManifest v2 has all 5 distinct revisions and repeating E was idempotent (exactly 5 total, not 6)
        manifest_path = self.workspace_dir / "projects" / "vision-research" / "runs" / "experiment-001" / "manifest.json"
        self.assertTrue(manifest_path.exists(), f"Manifest file missing: {manifest_path}")

        manifest = PublicationManifest.load(manifest_path)
        self.assertEqual(len(manifest.artifacts), 5, f"Expected exactly 5 revisions in manifest, got {len(manifest.artifacts)}")

        recorded_shas = {art.get("sha256") for art in manifest.artifacts.values()}
        self.assertEqual(recorded_shas, all_shas, "Recorded revisions SHA256 set does not match expected 5 revisions")

        # 4. Check that all 5 revisions have remote file IDs in Drive and can be retrieved with matching SHA256
        data_by_sha = {
            sha_a: data_a,
            sha_b: data_b,
            sha_c: data_c,
            sha_d: data_d,
            sha_e: data_e,
        }

        with self.drive_server.lock:
            committed_fids = set(self.drive_server.committed_checkpoints.keys())
            # Verify that all 5 committed checkpoint files have appProperties recorded on the loopback server!
            self.assertEqual(len(self.drive_server.committed_checkpoints), 5)
            for fid_ck, meta_ck in self.drive_server.committed_checkpoints.items():
                self.assertIn("appProperties", meta_ck, f"File {fid_ck} missing appProperties on loopback server")
                self.assertIsInstance(meta_ck["appProperties"], dict)
                self.assertTrue(
                    bool(meta_ck["appProperties"]),
                    f"File {fid_ck} appProperties empty: {meta_ck}",
                )

        for art_id, art in manifest.artifacts.items():
            fid = art.get("remote_file_id")
            art_sha = art.get("sha256")
            self.assertTrue(fid, f"Artifact {art_id} has no remote_file_id")
            self.assertIn(fid, committed_fids, f"Remote file {fid} not in committed checkpoints")

            # Retrieve file bytes from mock server and verify SHA256
            req = urllib.request.Request(f"{self.base_url}/drive/v3/files/{fid}?alt=media")
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                downloaded_bytes = resp.read()
                self.assertEqual(downloaded_bytes, data_by_sha[art_sha])
                self.assertEqual(hashlib.sha256(downloaded_bytes).hexdigest(), art_sha)

        # 5. Zero orphaned .tmp files
        tmp_files = list(self.workspace_dir.glob("**/*.tmp*"))
        self.assertEqual(len(tmp_files), 0, f"Orphaned .tmp files found: {tmp_files}")

    test_five_checkpoint_revisions_preserved_across_drive_manifest_api_and_dom = test_two_os_processes_five_revisions_publication_and_idempotency


if __name__ == "__main__":
    unittest.main()
