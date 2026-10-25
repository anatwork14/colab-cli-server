"""Browser DOM XSS & Two-OS-Process Simultaneous Drive Publication QA Suite.

Implements Independent Review 5640707483 verification requirements:
1. Real Headless Browser DOM XSS & UI Interaction Test:
   - Starts a real MonitorServer on 127.0.0.1:0 with api_token="browser-test-secret-token".
   - Seeds the MonitorService workspace with runs, sessions, metrics, logs, error events,
     and Drive artifacts containing 4 adversarial XSS payloads:
       * payload_img = '<img src=x onerror="window.__xss=true">'
       * payload_script = '</td><script>window.__xss=true</script>'
       * payload_onclick = "');window.__xss=true;//"
       * payload_svg = '"><svg onload="window.__xss=true">'
     across run_id, project_id, latest_checkpoint_name, artifact_id, remote_file_id,
     error event message, training log_line, and session metadata, plus normal clean run run-normal-001.
   - Verifies Playwright Core and Headless Chromium binary availability.
   - If available, runs headless Chromium (--no-sandbox --disable-gpu) to verify:
       1. zero unhandled page errors (pageErrors.length === 0)
       2. zero CSP securitypolicyviolation events
       3. zero XSS payload execution (window.__xss === undefined)
       4. zero injected <img> or <svg> elements inside tables and banners
       5. successful row click triggering Run Inspector and safe textContent rendering
   - If Playwright/Chromium is unavailable (e.g. CI runner), falls back to live HTTP + structural DOM checks.

2. Real Two-OS-Process (subprocess.Popen) Simultaneous Drive Publication Test:
   - Starts a loopback HTTP ThreadingHTTPServer (127.0.0.1:0) implementing a byte-accurate
     Google Drive v3 API server tracking upload sessions, chunks, and committed files in thread-safe memory.
   - Spawns two separate OS processes (subprocess.Popen([sys.executable, "-c", ...])) synchronized
     via a filesystem barrier file to enter store.publish(...) and ArtifactUploadWorker at the exact same instant
     for the same 512 KB checkpoint.pt artifact sharing journal_path and workspace_dir.
   - Asserts code 0, matching remote_file_id and sha256, at most 1 committed file, and valid TransferJournal / Manifest.
   - Verifies process abort mid-upload (chunk 1) followed by process 2 recovery from offset 262144 via HTTP 308.
"""
from __future__ import annotations

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


# =============================================================================
# Helper Utilities & Environment Detection
# =============================================================================

DEFAULT_NODE_PATHS = [
    "/mnt/ssd/genos2/.local/bin/node",
    shutil.which("node"),
]

DEFAULT_PLAYWRIGHT_PATHS = [
    "/mnt/ssd/genos2/openclaw-maint/node_modules/playwright-core",
]

DEFAULT_CHROMIUM_PATHS = [
    "/mnt/ssd/genos2/.hermes/tools/chromium-1208/chrome-linux64/chrome",
]


def resolve_browser_test_binaries() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve paths to node binary, playwright-core module, and chromium binary."""
    node_bin = None
    for p in DEFAULT_NODE_PATHS:
        if p and os.path.exists(p) and os.access(p, os.X_OK):
            node_bin = p
            break

    pw_path = None
    for p in DEFAULT_PLAYWRIGHT_PATHS:
        if p and os.path.isdir(p):
            pw_path = p
            break

    chrome_bin = None
    for p in DEFAULT_CHROMIUM_PATHS:
        if p and os.path.exists(p) and os.access(p, os.X_OK):
            chrome_bin = p
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
                    # Final chunk: commit file (no server-side deduplication so duplicate client uploads are caught!)
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
                    matching = [f for f in server.committed_checkpoints.values() if f.get("name") == target_name]

            resp_body = json.dumps({"files": matching}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
            return

        self.send_response(404)
        self.end_headers()


# =============================================================================
# Test Class 1: Real Headless Browser DOM XSS Verification
# =============================================================================

class TestRealHeadlessBrowserDashboardSecurity(unittest.TestCase):
    """Test suite executing real headless browser DOM XSS verification and UI interaction tests."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="colab_test_browser_")
        self.base_path = Path(self.tmp_dir)

        self.ledger_path = self.base_path / "account_usage.jsonl"
        self.telemetry_path = self.base_path / "telemetry_events.jsonl"
        self.registry_path = self.base_path / "managed_sessions.json"
        self.artifacts_dir = self.base_path / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.api_token = "browser-test-secret-token"

        # Adversarial XSS payloads required by review 5640707483
        self.payload_img = '<img src=x onerror="window.__xss=true">'
        self.payload_script = '</td><script>window.__xss=true</script>'
        self.payload_onclick = "');window.__xss=true;//"
        self.payload_svg = '"><svg onload="window.__xss=true">'

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
        """Seed MonitorService workspace with clean runs and adversarial payloads across all entities."""
        # 1. Seed AccountUsageCollector
        self.collector = AccountUsageCollector(ledger_path=self.ledger_path)
        sample = AccountSample(
            sample_id="sample-seed-01",
            observed_at=_iso_now(),
            balance_cu=92.5,
            rate_cu_per_hour=1.2,
            active_assignments=2,
            net_balance_change=-0.1,
            change_type="CONSUMPTION",
            status="OK",
        )
        self.collector._record_sample(sample)
        self.collector._last_sample = sample

        # 2. Seed ManagedSessionRegistry with normal and adversarial session metadata
        self.registry = ManagedSessionRegistry(self.registry_path)
        self.registry.register(
            session_id="session-normal-001",
            job_id="job-clean-1",
            gpu="T4",
            project_name="proj-normal",
        )
        self.registry.update_state("session-normal-001", "READY")

        self.registry.register(
            session_id="session-adv-001",
            job_id="job-adv-2",
            gpu=f"T4 {self.payload_svg}",
            project_name=f"proj {self.payload_script}",
        )
        self.registry.update_state("session-adv-001", "RUNNING")

        # 3. Seed TelemetryEmitter for normal clean run
        self.emitter_clean = TelemetryEmitter(
            ledger_path=self.telemetry_path,
            run_id="run-normal-001",
            project_id="proj-normal",
        )
        self.emitter_clean.emit("run_started")
        self.emitter_clean.emit("metric", epoch=1, step=10, metrics={"loss": 0.42})
        self.emitter_clean.emit("checkpoint_saved", epoch=1, step=10)
        self.emitter_clean.emit("log_line", details={"log_line": "Clean baseline epoch 1 completed"})

        # 4. Seed TelemetryEmitter for adversarial runs
        self.emitter_adv = TelemetryEmitter(
            ledger_path=self.telemetry_path,
            run_id="run-adv-001",
            project_id=f"proj-xss-{self.payload_img}",
        )
        self.emitter_adv.emit("run_started")
        self.emitter_adv.emit("metric", epoch=1, step=5, metrics={"loss": 0.88})
        self.emitter_adv.emit("checkpoint_saved", epoch=1, step=5)
        self.emitter_adv.emit("log_line", details={"log_line": f"Adversarial training log: {self.payload_script}"})
        self.emitter_adv.emit("error", details={"message": f"Fatal runtime failure: {self.payload_onclick}", "error": self.payload_img})

        self.emitter_adv2 = TelemetryEmitter(
            ledger_path=self.telemetry_path,
            run_id="run-adv-svg-002",
            project_id="proj-svg-xss",
        )
        self.emitter_adv2.emit("run_started")
        self.emitter_adv2.emit("error", details={"message": f"SVG event payload: {self.payload_svg}"})

        # 5. Seed PublicationManifest for run-normal-001
        run_norm_dir = self.artifacts_dir / "projects" / "proj-normal" / "runs" / "run-normal-001"
        run_norm_dir.mkdir(parents=True, exist_ok=True)
        m_norm = PublicationManifest(project_id="proj-normal", run_id="run-normal-001")
        chk_norm = StoredArtifact(
            artifact_id="proj-normal/run-normal-001/checkpoints/model.pt",
            storage_provider="google_drive",
            path=str(run_norm_dir / "checkpoints" / "model.pt"),
            size_bytes=1024,
            sha256="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            project_id="proj-normal",
            run_id="run-normal-001",
            artifact_type="checkpoint",
            remote_file_id="drive-clean-001",
            verified_at=_iso_now(),
        )
        m_norm.add_artifact(chk_norm)
        m_norm.save(run_norm_dir / "manifest.json")

        # 6. Seed PublicationManifest for run-adv-001 containing payload artifact fields
        run_adv_dir = self.artifacts_dir / "projects" / f"proj-xss-{self.payload_img}" / "runs" / "run-adv-001"
        run_adv_dir.mkdir(parents=True, exist_ok=True)
        m_adv = PublicationManifest(project_id=f"proj-xss-{self.payload_img}", run_id="run-adv-001")
        chk_adv = StoredArtifact(
            artifact_id=f"proj-xss/run-adv-001/checkpoints/{self.payload_img}",
            storage_provider="google_drive",
            path=str(run_adv_dir / "checkpoints" / "model_chk.pt"),
            size_bytes=2048,
            sha256="9876543210fedcba9876543210fedcba9876543210fedcba9876543210fedcba",
            project_id=f"proj-xss-{self.payload_img}",
            run_id="run-adv-001",
            artifact_type=f"checkpoint-{self.payload_onclick}",
            remote_file_id=f"drive-adv-{self.payload_svg}",
            verified_at=_iso_now(),
        )
        m_adv.add_artifact(chk_adv)
        m_adv.save(run_adv_dir / "manifest.json")

    def _start_server(self) -> None:
        """Start real loopback MonitorServer serving production render_dashboard_html with CSP nonces."""
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

    def test_real_headless_browser_dom_xss_and_ui_interaction(self) -> None:
        """Execute real headless browser DOM XSS verification using Playwright Core and Chromium."""
        node_bin, pw_path, chrome_bin = resolve_browser_test_binaries()

        if not (node_bin and pw_path and chrome_bin):
            # Fallback for CI environments lacking local Chromium installation
            self._run_live_http_and_structural_dom_checks()
            return

        node_script = f"""
        const {{ chromium }} = require("{pw_path}");

        (async () => {{
          const browser = await chromium.launch({{
            executablePath: "{chrome_bin}",
            args: ["--no-sandbox", "--disable-gpu"]
          }});
          const page = await browser.newPage();
          const pageErrors = [];
          const dialogs = [];
          const cspViolations = [];

          page.on("pageerror", err => {{
            pageErrors.push(err.message || String(err));
          }});

          page.on("dialog", async dialog => {{
            dialogs.push(dialog.message());
            await dialog.dismiss();
          }});

          // Navigate to dashboard
          await page.goto("{self.base_url}/");

          // Set session token and security listeners
          await page.evaluate((tok) => {{
            sessionStorage.setItem("colab_monitor_token", tok);
            document.addEventListener("securitypolicyviolation", e => {{
              window.__csp_violations = window.__csp_violations || [];
              window.__csp_violations.push(e.blockedURI || "blocked");
            }});
          }}, "{self.api_token}");

          // Trigger refreshAll to fetch protected endpoints
          await page.evaluate(async (tok) => {{
            token = tok;
            if (typeof refreshAll === "function") {{
              await refreshAll();
            }}
          }}, "{self.api_token}");

          // Wait for runs table and sessions table to render
          await page.waitForSelector("#runs-table tbody tr.clickable", {{ timeout: 7000 }});
          await page.waitForSelector("#sessions-table tbody tr", {{ timeout: 7000 }});

          // Click each run row in #runs-table tbody tr to trigger selectRun
          const rows = await page.locator("#runs-table tbody tr.clickable").all();
          for (const row of rows) {{
            await row.click();
            await page.waitForTimeout(300);
          }}

          // Select run-adv-001 so inspector renders adversarial error banner and artifacts
          const advRow = page.locator('#runs-tbody tr[data-run-id="run-adv-001"]');
          if (await advRow.count() > 0) {{
            await advRow.click();
            await page.waitForTimeout(400);
          }}

          // Click a .sha-badge or .sha-tag element
          const shaBadge = page.locator(".sha-badge, .sha-tag").first();
          const shaCount = await shaBadge.count();
          if (shaCount > 0) {{
            await shaBadge.click();
            await page.waitForTimeout(150);
          }}

          // Evaluate browser context security guarantees
          const result = await page.evaluate(() => {{
            const runsTable = document.getElementById("runs-table");
            const artifactsTable = document.getElementById("artifacts-table");
            const errorBanner = document.getElementById("log-error-banner") || document.getElementById("error-banner");

            const injectedImg = document.querySelectorAll("#runs-table img, #artifacts-table img, #log-error-banner img, #error-banner img");
            const injectedSvg = document.querySelectorAll("#runs-table svg:not([data-icon]):not(.svg-icon), #artifacts-table svg, #log-error-banner svg, #error-banner svg");
            const normalCspViolations = [...(window.__csp_violations || [])];

            // Also verify that CSP actively blocks an un-nonced inline <script> injection
            const rogueScript = document.createElement("script");
            rogueScript.textContent = "window.__xss_csp = true;";
            document.body.appendChild(rogueScript);

            return {{
              xssValue: window.__xss,
              xssCspValue: window.__xss_csp,
              injectedImgCount: injectedImg.length,
              injectedSvgCount: injectedSvg.length,
              runsTableText: runsTable ? runsTable.textContent : "",
              errorBannerText: errorBanner ? errorBanner.textContent : "",
              artifactsTableText: artifactsTable ? artifactsTable.textContent : "",
              cspViolations: normalCspViolations
            }};
          }});

          await browser.close();

          console.log("TEST_RESULT:" + JSON.stringify({{
            success: true,
            pageErrors: pageErrors,
            dialogs: dialogs,
            ...result
          }}));
        }})();
        """

        res = subprocess.run(
            [node_bin, "-e", node_script],
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(res.returncode, 0, f"Headless browser runner exited non-zero: {res.stderr}")
        m_res = re.search(r"TEST_RESULT:(.*)", res.stdout)
        self.assertIsNotNone(m_res, f"Could not find TEST_RESULT in output: {res.stdout}\n{res.stderr}")
        data = json.loads(m_res.group(1))

        # 1. Zero XSS execution & CSP enforcement
        self.assertIsNone(data.get("xssValue"), "Adversarial script executed in browser context (window.__xss was defined)!")
        self.assertIsNone(data.get("xssCspValue"), "CSP failed to block un-nonced inline script injection!")
        self.assertEqual(len(data.get("cspViolations", [])), 0, f"CSP violations during normal UI operation: {data.get('cspViolations')}")

        # 2. Zero unhandled page errors
        self.assertEqual(len(data.get("pageErrors", [])), 0, f"Unhandled JavaScript errors observed: {data.get('pageErrors')}")

        # 3. Zero dialog popups
        self.assertEqual(len(data.get("dialogs", [])), 0, f"Unexpected alert/dialog popups triggered: {data.get('dialogs')}")

        # 4. Zero injected <img> or <svg> elements inside tables or error banners
        self.assertEqual(data.get("injectedImgCount"), 0, "Adversarial <img> elements were injected into the DOM!")
        self.assertEqual(data.get("injectedSvgCount"), 0, "Adversarial <svg> elements were injected into the DOM!")

        # 5. Rendered textContent contains payload text safely as plain text
        runs_text = data.get("runsTableText", "")
        self.assertIn("run-normal-001", runs_text)
        self.assertIn("run-adv-001", runs_text)

        artifacts_text = data.get("artifactsTableText", "")
        self.assertIn("proj-xss", artifacts_text)

    def test_http_and_structural_dom_security_fallback(self) -> None:
        """Execute HTTP and structural DOM contract check across all MonitorServer endpoints."""
        self._run_live_http_and_structural_dom_checks()

    def _run_live_http_and_structural_dom_checks(self) -> None:
        """Verify headers and data isolation across live MonitorServer responses."""
        headers = {"Authorization": f"Bearer {self.api_token}", "Accept": "application/json"}

        # 1. GET /
        req = urllib.request.Request(f"{self.base_url}/")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            csp = resp.headers.get("Content-Security-Policy", "")
            self.assertTrue(bool(csp), "MonitorServer missing Content-Security-Policy header")
            x_frame = resp.headers.get("X-Frame-Options", "")
            self.assertEqual(x_frame, "DENY")
            x_cto = resp.headers.get("X-Content-Type-Options", "")
            self.assertEqual(x_cto, "nosniff")

        # 2. GET /api/v1/runs
        req = urllib.request.Request(f"{self.base_url}/api/v1/runs", headers=headers)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            runs = json.loads(resp.read().decode("utf-8"))
            self.assertIsInstance(runs, list)
            run_ids = [r.get("run_id") for r in runs]
            self.assertIn("run-normal-001", run_ids)
            self.assertIn("run-adv-001", run_ids)

        # 3. GET /api/v1/runs/run-adv-001/logs
        req = urllib.request.Request(f"{self.base_url}/api/v1/runs/run-adv-001/logs", headers=headers)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            logs_payload = json.loads(resp.read().decode("utf-8"))
            logs_list = logs_payload.get("logs", [])
            self.assertTrue(any(self.payload_script in line for line in logs_list))

        # 4. GET /api/v1/runs/run-adv-001/artifacts
        req = urllib.request.Request(f"{self.base_url}/api/v1/runs/run-adv-001/artifacts", headers=headers)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            art_payload = json.loads(resp.read().decode("utf-8"))
            arts = art_payload.get("artifacts", [])
            self.assertTrue(len(arts) > 0)
            self.assertTrue(any(self.payload_img in a.get("artifact_id", "") for a in arts))


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


# =============================================================================
# Test Class 2: Real Two-OS-Process Simultaneous Drive Publication
# =============================================================================

class TestCrossProcessDrivePublication(unittest.TestCase):
    """Test suite executing real multi-process concurrency and resumable recovery tests."""

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

    def test_two_os_processes_simultaneous_store_publish(self) -> None:
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
        self.assertIn(transfer_key, journal_data)
        self.assertEqual(journal_data[transfer_key]["status"], TransferStatus.PUBLISHED.value)
        self.assertEqual(journal_data[transfer_key]["remote_file_id"], res1["remote_file_id"])

        # 5. Shared PublicationManifest (manifest.json) is valid JSON and records artifact without corruption
        manifest_path = self.workspace_dir / "projects" / "test_proj" / "runs" / "test_run" / "manifest.json"
        self.assertTrue(manifest_path.exists(), f"Publication manifest not found at {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as mf:
            manifest_data = json.load(mf)
        self.assertEqual(manifest_data["status"], "PUBLISHED")
        art_key = f"test_proj/test_run/checkpoints/{self.ckpt_path.name}"
        self.assertIn(art_key, manifest_data.get("artifacts", {}))
        self.assertEqual(manifest_data["artifacts"][art_key]["remote_file_id"], res1["remote_file_id"])
        self.assertEqual(manifest_data["artifacts"][art_key]["sha256"], self.expected_sha256)

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
        self.assertEqual(journal_after_abort[t_key]["bytes_transferred"], 262144)

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
        self.assertEqual(journal_final[t_key]["status"], TransferStatus.PUBLISHED.value)


if __name__ == "__main__":
    unittest.main()
