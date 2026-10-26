"""Secure, read-only loopback HTTP server and SSE stream for Colab Monitor (M8).

Guarantees:
- Loopback binding by default (127.0.0.1); refuses public interfaces unless explicit (Q28).
- Strict authentication via Authorization Bearer header or verified reverse-proxy identity; 401 on unauthorized (Q26).
- Strictly read-only: POST/PUT/DELETE return 405 Method Not Allowed (Q29).
- SSE event stream with cursor resume (Q27).
- Deep credential redaction in all responses (Q31).
- Serves embedded responsive SPA dashboard with zero fabricated telemetry (Q30).
"""
from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import ssl
import stat
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from ..provider import ManagedSessionRegistry
from ..telemetry import TelemetryEmitter
from ..tracker import sanitize_data
from .collector import AccountUsageCollector
from .poller import MonitorPoller, clamp_poll_interval
from .schemas import HealthStatus, MonitoringSummary, RunSummary, _iso_now

try:
    from .ui import DASHBOARD_HTML as _EMBEDDED_HTML_DASHBOARD
except ImportError:
    _EMBEDDED_HTML_DASHBOARD = None

try:
    from .ui import render_dashboard_html as _render_dashboard_html
except (ImportError, AttributeError):
    _render_dashboard_html = None

try:
    from .ui import get_dashboard_js as _get_dashboard_js
except (ImportError, AttributeError):
    _get_dashboard_js = None

try:
    from .ui import get_dashboard_css as _get_dashboard_css
except (ImportError, AttributeError):
    _get_dashboard_css = None

if _EMBEDDED_HTML_DASHBOARD is None:
    _EMBEDDED_HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Colab Monitor — V2 Remote GPU Workspaces</title>
  <style nonce="__CSP_NONCE__">
    :root {
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --text-bright: #f0f6fc;
      --text-dim: #8b949e;
      --accent: #58a6ff;
      --green: #3fb950;
      --yellow: #d29922;
      --red: #f85149;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 1rem;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 1.5rem;
    }
    h1 { font-size: 1.3rem; color: var(--text-bright); display: flex; align-items: center; gap: 0.5rem; }
    .badge {
      font-size: 0.75rem;
      padding: 0.2rem 0.5rem;
      border-radius: 999px;
      background: #21262d;
      color: var(--text-dim);
    }
    .badge.active { background: rgba(63, 185, 80, 0.2); color: var(--green); }
    .badge.stale { background: rgba(210, 153, 34, 0.2); color: var(--yellow); }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 1rem;
    }
    .card-title { font-size: 0.8rem; color: var(--text-dim); text-transform: uppercase; margin-bottom: 0.25rem; }
    .card-value { font-size: 1.6rem; font-weight: 600; color: var(--text-bright); }
    .card-subtitle { font-size: 0.75rem; color: var(--text-dim); margin-top: 0.25rem; }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 0.5rem;
      font-size: 0.85rem;
    }
    th, td {
      padding: 0.6rem 0.75rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    th { color: var(--text-dim); font-weight: 600; background: #0b0f14; }
    tr:hover { background: #1c2128; }
    .section-title { font-size: 1.1rem; color: var(--text-bright); margin-bottom: 0.5rem; margin-top: 1rem; }
    .nav { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
    .nav-btn {
      background: var(--card-bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 0.4rem 0.8rem;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.85rem;
    }
    .nav-btn.active { border-color: var(--accent); color: var(--accent); }
    #log-view {
      background: #06090e;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.75rem;
      font-family: monospace;
      font-size: 0.8rem;
      max-height: 250px;
      overflow-y: auto;
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
  <header>
    <h1>Colab Monitor <span class="badge active" id="health-badge">HEALTHY</span></h1>
    <div style="font-size: 0.8rem; color: var(--text-dim);" id="header-meta">Monitoring Active</div>
  </header>

  <div class="grid">
    <div class="card">
      <div class="card-title">Compute Units Balance</div>
      <div class="card-value" id="val-cu">—</div>
      <div class="card-subtitle" id="sub-cu">Account-wide snapshot</div>
    </div>
    <div class="card">
      <div class="card-title">Consumption Rate</div>
      <div class="card-value" id="val-rate">—</div>
      <div class="card-subtitle">CU/hour</div>
    </div>
    <div class="card">
      <div class="card-title">Active Assignments</div>
      <div class="card-value" id="val-assignments">—</div>
      <div class="card-subtitle" id="sub-assignments">Remote Colab sessions</div>
    </div>
    <div class="card">
      <div class="card-title">Managed Runs</div>
      <div class="card-value" id="val-runs">—</div>
      <div class="card-subtitle" id="sub-runs">Running / Total</div>
    </div>
  </div>

  <div class="section-title">Active Managed Sessions</div>
  <div class="card">
    <table id="sessions-table">
      <thead>
        <tr>
          <th>Session ID</th>
          <th>Project</th>
          <th>GPU</th>
          <th>State</th>
          <th>Lease Remaining</th>
          <th>Managed</th>
        </tr>
      </thead>
      <tbody>
        <tr><td colspan="6" style="text-align: center; color: var(--text-dim);">No active sessions</td></tr>
      </tbody>
    </table>
  </div>

  <div class="section-title">Recent Training Runs & Metrics</div>
  <div class="card">
    <table id="runs-table">
      <thead>
        <tr>
          <th>Run ID</th>
          <th>Project</th>
          <th>Status</th>
          <th>Latest Checkpoint</th>
          <th>Drive Status</th>
          <th>Epoch / Step</th>
          <th>Started</th>
        </tr>
      </thead>
      <tbody>
        <tr><td colspan="7" style="text-align: center; color: var(--text-dim);">No training runs recorded</td></tr>
      </tbody>
    </table>
  </div>

  <div class="section-title">Durable Artifacts & Google Drive Publication</div>
  <div class="card" id="artifacts-card">
    <table id="artifacts-table">
      <thead>
        <tr>
          <th>Artifact</th>
          <th>Type</th>
          <th>Size</th>
          <th>SHA256</th>
          <th>Storage</th>
          <th>Drive Status</th>
        </tr>
      </thead>
      <tbody id="artifacts-tbody">
        <tr><td colspan="6" style="text-align: center; color: var(--text-dim);">Select a run or wait for artifacts</td></tr>
      </tbody>
    </table>
  </div>


  <script nonce="__CSP_NONCE__">
    let token = sessionStorage.getItem('colab_monitor_token') || '';

    function getHeaders() {
      return token ? { 'Authorization': 'Bearer ' + token } : {};
    }

    async function promptForToken() {
      const entered = window.prompt("Enter Monitor Authorization Bearer Token:");
      if (entered) {
        token = entered.trim();
        sessionStorage.setItem('colab_monitor_token', token);
        refresh();
      }
    }

    async function refresh() {
      try {
        const res = await fetch('/api/v1/summary', { headers: getHeaders() });
        if (res.status === 401) {
          document.getElementById('header-meta').textContent = '401 Unauthorized: Authorization token required.';
          const badge = document.getElementById('health-badge');
          badge.textContent = 'AUTH REQUIRED';
          badge.className = 'badge stale';
          badge.style.cursor = 'pointer';
          badge.addEventListener('click', promptForToken, { once: true });
          return;
        }
        const data = await res.json();
        const cu = Number(data.current_balance_cu);
        const rate = Number(data.current_rate_cu_per_hour);
        document.getElementById('val-cu').textContent = (data.current_balance_cu != null && Number.isFinite(cu)) ? cu.toFixed(2) : 'N/A';
        document.getElementById('val-rate').textContent = (data.current_rate_cu_per_hour != null && Number.isFinite(rate)) ? rate.toFixed(2) : '0.00';
        document.getElementById('val-assignments').textContent = String(data.active_assignments || 0);
        document.getElementById('val-runs').textContent = String(data.managed_running_runs || 0) + ' / ' + String(data.total_runs || 0);
        document.getElementById('header-meta').textContent = 'Monitoring since ' + String(data.monitoring_since || 'now').slice(0, 19);

        const badge = document.getElementById('health-badge');
        if (data.is_stale) {
          badge.textContent = 'STALE';
          badge.className = 'badge stale';
        } else {
          badge.textContent = 'HEALTHY';
          badge.className = 'badge active';
        }
      } catch (e) {
        console.error('Fetch failed:', e);
      }
    }
    refresh();
    setInterval(refresh, 10000);
  </script>
</body>
</html>
"""


# Unmanifested local files above this size are listed with size only (no sha256).
_LOCAL_HASH_MAX_BYTES = 8 * 1024 * 1024

# fullmatch (not match with "$") so a trailing newline is rejected; ASCII only.
_IDENTIFIER_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_\-\.]{0,127}")


def _is_valid_identifier(value: Optional[str]) -> bool:
    """Validate a project/run selector: bounded ASCII charset, no path traversal."""
    return bool(value) and _IDENTIFIER_RE.fullmatch(value) is not None and ".." not in value


def _event_run_key(ev: Dict[str, Any]) -> Tuple[str, str]:
    """Canonical RunKey (project_id, run_id) for a telemetry event (C4).

    Legacy events without project_id map to project "" which is not a valid
    selector, so they never match a named project (D10).
    """
    return (str(ev.get("project_id") or ""), str(ev.get("run_id") or ""))


def _drop_phantom_legacy_keys(keys: Iterable[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    """Remove legacy keyless RunKeys ('', run_id) whenever a named project owns that run id.

    Legacy events carry no project_id. When a named project also owns the same run id, the
    keyless entry is a phantom duplicate and must not be listed or counted. A legacy key with
    no named owner is kept, so legacy-only runs stay visible.
    """
    key_set = set(keys)
    named_run_ids = {k[1] for k in key_set if k[0]}
    return {k for k in key_set if k[0] or k[1] not in named_run_ids}


# Drive publication states, as recorded by the storage layer. A recorded value outside
# DRIVE_PUBLISHED_STATUS and DRIVE_STATUS_NOT_RECORDED means a transfer was interrupted or is in progress.
DRIVE_STATUS_PUBLISHED = "PUBLISHED"
DRIVE_STATUS_PENDING = "PENDING"
DRIVE_STATUS_NOT_RECORDED = "NOT RECORDED"
_DRIVE_INTERRUPTED_STATES = frozenset({
    "UPLOAD_PENDING", "UPLOADING", "RETRY_PENDING", "UPLOAD_FAILED", "LOCAL_VERIFIED", "PENDING", "FAILED", "PARTIAL",
})
_DRIVE_STATE_FIELDS = ("drive_publication_status", "transfer_status")


def _records_interrupted_drive_transfer(manifest_data: Optional[Dict[str, Any]], events: Iterable[Dict[str, Any]]) -> bool:
    """True only when the manifest or telemetry actually records an interrupted or in-progress Drive transfer.

    The manifest-level IN_PROGRESS default is NOT evidence of a transfer and is ignored.
    """
    if manifest_data:
        if str(manifest_data.get("status") or "").upper() in ("PARTIAL", "FAILED"):
            return True
        raw_artifacts = manifest_data.get("artifacts") or {}
        if isinstance(raw_artifacts, dict):
            for art in raw_artifacts.values():
                if not isinstance(art, dict):
                    continue
                containers = [art, art.get("metadata") if isinstance(art.get("metadata"), dict) else {}]
                for container in containers:
                    for name in _DRIVE_STATE_FIELDS:
                        if str(container.get(name) or "").upper() in _DRIVE_INTERRUPTED_STATES:
                            return True
    for ev in events:
        details = ev.get("details") if isinstance(ev.get("details"), dict) else {}
        for container in (ev, details):
            for name in _DRIVE_STATE_FIELDS:
                if str(container.get(name) or "").upper() in _DRIVE_INTERRUPTED_STATES:
                    return True
    return False


def _drive_publication_state(
    drive_backed: bool,
    manifest_data: Optional[Dict[str, Any]],
    events: Iterable[Dict[str, Any]],
) -> str:
    """PUBLISHED when a Drive-backed artifact is recorded; PENDING when an interrupted or in-progress
    transfer is recorded; otherwise NOT RECORDED (no Drive information, never a guessed PENDING)."""
    if drive_backed:
        return DRIVE_STATUS_PUBLISHED
    if _records_interrupted_drive_transfer(manifest_data, events):
        return DRIVE_STATUS_PENDING
    return DRIVE_STATUS_NOT_RECORDED


def _load_run_manifest(run_dir: Path, project_id: str, run_id: str) -> Optional[Dict[str, Any]]:
    """Load manifest.json for an exact run directory, rejecting symlinks and id mismatches (D10)."""
    m_path = run_dir / "manifest.json"
    if m_path.is_symlink() or not m_path.is_file():
        return None
    try:
        with open(m_path, "r", encoding="utf-8") as mf:
            data = json.load(mf)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("project_id", project_id) != project_id or data.get("run_id", run_id) != run_id:
        return None
    return data


def _inject_csp_nonce(html_str: str, csp_nonce: str) -> str:
    """Inject cryptographic CSP nonce into HTML script and style tags."""
    if "__CSP_NONCE__" in html_str:
        html_str = html_str.replace("__CSP_NONCE__", csp_nonce)

    def _replace_script(m: re.Match) -> str:
        tag = m.group(0)
        if "nonce=" in tag:
            return tag
        return f'<script nonce="{csp_nonce}"' + tag[7:]

    html_str = re.sub(r"<script(\s[^>]*)?>", _replace_script, html_str, flags=re.IGNORECASE)

    def _replace_style(m: re.Match) -> str:
        tag = m.group(0)
        if "nonce=" in tag:
            return tag
        return f'<style nonce="{csp_nonce}"' + tag[6:]

    html_str = re.sub(r"<style(\s[^>]*)?>", _replace_style, html_str, flags=re.IGNORECASE)
    return html_str


def _extract_css_from_html(html: str) -> str:
    """Extract raw CSS rules from <style> blocks in HTML."""
    styles = re.findall(r"<style[^>]*>(.*?)</style>", html, flags=re.DOTALL | re.IGNORECASE)
    if styles:
        return "\n".join(s.strip() for s in styles)
    return ""


def _extract_js_from_html(html: str) -> str:
    """Extract raw JavaScript code from <script> blocks in HTML."""
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, flags=re.DOTALL | re.IGNORECASE)
    if scripts:
        return "\n".join(s.strip() for s in scripts)
    return ""


def _sanitize_path_or_url(url_path: str) -> str:
    """Redact sensitive query parameters (?token=..., ?access_token=...) from URLs/paths."""
    return re.sub(
        r"([?&](?:token|access_token|key|secret|auth|bearer)=)[^&]+",
        r"\1[REDACTED]",
        url_path,
        flags=re.IGNORECASE,
    )


class MonitorRequestHandler(BaseHTTPRequestHandler):
    """Handles read-only authenticated HTTP requests for the monitor."""

    server: "MonitorServer"

    def _send_security_headers(self, csp_nonce: Optional[str] = None) -> None:
        """Send defense-in-depth security headers with strict CSP and zero 'unsafe-inline'."""
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        if csp_nonce:
            self.send_header(
                "Content-Security-Policy",
                f"default-src 'self'; script-src 'self' 'nonce-{csp_nonce}'; style-src 'self' 'nonce-{csp_nonce}'; "
                f"img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            )
        else:
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            )
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")

    def _send_cors_and_security_headers(self, csp_nonce: Optional[str] = None) -> None:
        """Enforce strict CORS and defense-in-depth security headers."""
        origin = self.headers.get("Origin")
        allowed = getattr(self.server, "allowed_origins", None) or [
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ]

        if origin and origin != "*" and origin in allowed and "*" not in origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Authorization, X-API-Token, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")

        self._send_security_headers(csp_nonce=csp_nonce)

    def _is_authenticated(self, query_params: Dict[str, List[str]]) -> bool:
        """Validate credentials from Authorization header, X-API-Token, or trusted gateway."""
        expected_token = getattr(self.server, "auth_token", None)
        trusted_proxy = getattr(self.server, "trusted_proxy", False)
        broker = getattr(self.server, "broker", None)

        # 1. Trusted reverse proxy gateway principal validation
        if trusted_proxy:
            client_ip = self.client_address[0] if self.client_address else ""
            trusted_ips = getattr(self.server, "trusted_proxy_ips", None) or {"127.0.0.1", "::1"}
            proxy_secret = getattr(self.server, "trusted_proxy_secret", None)

            proxy_secret_hdr = self.headers.get("X-Proxy-Secret", "")
            proxy_sig_hdr = self.headers.get("X-Proxy-Signature", "")
            remote_user = (self.headers.get("X-Remote-User") or self.headers.get("X-Forwarded-User") or "").strip()

            if (
                client_ip in trusted_ips
                and proxy_secret
                and remote_user
                and re.match(r"^[a-zA-Z0-9_.@-]{1,128}$", remote_user)
            ):
                secret_bytes = proxy_secret.encode("utf-8")
                secret_valid = bool(
                    proxy_secret_hdr
                    and hmac.compare_digest(proxy_secret_hdr.encode("utf-8"), secret_bytes)
                )
                sig_valid = bool(
                    proxy_sig_hdr
                    and hmac.compare_digest(
                        proxy_sig_hdr.lower(),
                        hmac.new(secret_bytes, remote_user.encode("utf-8"), hashlib.sha256).hexdigest().lower(),
                    )
                )
                if secret_valid or sig_valid:
                    return True

        # 2. Extract Bearer or X-API-Token header
        token: Optional[str] = None
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        elif "X-API-Token" in self.headers:
            token = self.headers.get("X-API-Token", "").strip()

        # 3. Check broker verify_token
        if token and broker and hasattr(broker, "verify_token"):
            if broker.verify_token(token):
                return True

        # 4. Check server auth_token
        if token and expected_token:
            if hmac.compare_digest(token.encode("utf-8"), expected_token.encode("utf-8")):
                return True

        # Auth disabled only when no auth mechanism was configured at all (for local-only test mode)
        if not expected_token and not trusted_proxy and not broker:
            return True

        # Query parameter tokens (?token=...) are strictly rejected
        return False


    def _send_json(self, data: Any, status: int = 200) -> None:
        sanitized = sanitize_data(data)
        body = json.dumps(sanitized, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_and_security_headers(csp_nonce=None)
        self.end_headers()
        self.wfile.write(body)

    def _send_html(
        self,
        body: Union[str, bytes],
        csp_nonce: Optional[str] = None,
        status: int = 200,
    ) -> None:
        encoded = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self._send_cors_and_security_headers(csp_nonce=csp_nonce)
        self.end_headers()
        self.wfile.write(encoded)

    def _send_js(self, body: Union[str, bytes], status: int = 200) -> None:
        encoded = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", "application/javascript; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self._send_cors_and_security_headers(csp_nonce=None)
        self.end_headers()
        self.wfile.write(encoded)

    def _send_css(self, body: Union[str, bytes], status: int = 200) -> None:
        encoded = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", "text/css; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self._send_cors_and_security_headers(csp_nonce=None)
        self.end_headers()
        self.wfile.write(encoded)

    def _send_401(self) -> None:
        self._send_json(
            {"error": "Unauthorized", "message": "Missing or invalid authentication token."},
            status=HTTPStatus.UNAUTHORIZED,
        )

    def _send_405(self) -> None:
        """Reject non-GET methods (Q29)."""
        self._send_json(
            {"error": "Method Not Allowed", "message": "The monitoring dashboard API is strictly read-only."},
            status=HTTPStatus.METHOD_NOT_ALLOWED,
        )

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_and_security_headers(csp_nonce=None)
        self.end_headers()

    def do_POST(self) -> None:
        self._send_405()

    def do_PUT(self) -> None:
        self._send_405()

    def do_DELETE(self) -> None:
        self._send_405()

    def do_PATCH(self) -> None:
        self._send_405()

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        # Serve static HTML/JS dashboard without token prompt on root if public, or check auth
        if path in ("/", "/index.html"):
            csp_nonce = secrets.token_urlsafe(18)
            raw_html: Optional[str] = None

            # If a custom dashboard_html was passed to MonitorServer, respect it
            custom_html = getattr(self.server, "dashboard_html", None)
            if custom_html and custom_html != _EMBEDDED_HTML_DASHBOARD:
                raw_html = custom_html
            elif _render_dashboard_html is not None:
                try:
                    raw_html = _render_dashboard_html(csp_nonce=csp_nonce)
                except TypeError:
                    try:
                        raw_html = _render_dashboard_html()
                    except Exception:
                        raw_html = getattr(self.server, "dashboard_html", _EMBEDDED_HTML_DASHBOARD)
                except Exception:
                    raw_html = getattr(self.server, "dashboard_html", _EMBEDDED_HTML_DASHBOARD)
            else:
                raw_html = getattr(self.server, "dashboard_html", _EMBEDDED_HTML_DASHBOARD)

            final_html = _inject_csp_nonce(raw_html, csp_nonce)
            self._send_html(final_html, csp_nonce=csp_nonce)
            return

        elif path == "/monitor.js":
            js_content = ""
            if _get_dashboard_js is not None:
                try:
                    js_content = _get_dashboard_js()
                except Exception:
                    js_content = ""
            if not js_content:
                dashboard_content = getattr(self.server, "dashboard_html", None)
                if not dashboard_content and _render_dashboard_html is not None:
                    try:
                        dashboard_content = _render_dashboard_html()
                    except Exception:
                        dashboard_content = _EMBEDDED_HTML_DASHBOARD
                js_content = _extract_js_from_html(dashboard_content or _EMBEDDED_HTML_DASHBOARD)
            self._send_js(js_content)
            return

        elif path == "/monitor.css":
            css_content = ""
            if _get_dashboard_css is not None:
                try:
                    css_content = _get_dashboard_css()
                except Exception:
                    css_content = ""
            if not css_content:
                dashboard_content = getattr(self.server, "dashboard_html", None)
                if not dashboard_content and _render_dashboard_html is not None:
                    try:
                        dashboard_content = _render_dashboard_html()
                    except Exception:
                        dashboard_content = _EMBEDDED_HTML_DASHBOARD
                css_content = _extract_css_from_html(dashboard_content or _EMBEDDED_HTML_DASHBOARD)
            self._send_css(css_content)
            return

        # All API endpoints require authentication (Q26)
        if not self._is_authenticated(query):
            self._send_401()
            return

        if path == "/api/v1/health":
            self._handle_health()
        elif path == "/api/v1/summary":
            self._handle_summary(query)
        elif path == "/api/v1/compute/samples":
            self._handle_compute_samples(query)
        elif path == "/api/v1/sessions":
            self._handle_sessions()
        elif path == "/api/v1/runs":
            self._handle_runs(query)
        elif path.startswith("/api/v1/runs/"):
            self._handle_run_detail(path, query)
        elif path == "/api/v1/events/stream":
            self._handle_events_stream(query)
        elif path == "/api/v1/export/compute.csv":
            self._handle_export_csv()
        else:
            self._send_json({"error": "Not Found", "path": path}, status=HTTPStatus.NOT_FOUND)

    def _handle_health(self) -> None:
        col = self.server.collector
        samples = col.get_samples(limit=10)
        latest = samples[-1] if samples else None

        status_str = "HEALTHY"
        if col.is_stale:
            status_str = "DEGRADED"
        if latest and latest.status == "ERROR":
            status_str = "ERROR"

        health = HealthStatus(
            status=status_str,
            monitoring_since=col.monitoring_since,
            last_collected_at=latest.collected_at if latest else None,
            sample_count=len(col.get_samples(limit=10000)),
            is_stale=col.is_stale,
            collector_backoff_level=col.backoff_level,
            uptime_seconds=round(time.time() - self.server.start_time, 1),
        )
        self._send_json(health.to_dict())

    def _handle_summary(self, query: Optional[Dict[str, List[str]]] = None) -> None:
        project = (query or {}).get("project", [None])[0]
        if project is not None and not _is_valid_identifier(project):
            self._send_json(
                {"error": "Invalid project", "message": "Project selector contains illegal characters."},
                status=400,
            )
            return
        col = self.server.collector
        latest = col.get_latest_sample()
        all_managed = self.server.registry.get_all_managed()
        running_count = sum(1 for s in all_managed.values() if s.get("state") in ("READY", "RUNNING", "ACTIVE"))

        events = self.server.telemetry.get_events(project_id=project)
        # Same phantom rule as /runs so total_runs equals the runs list length for event-backed runs.
        unique_runs = len(_drop_phantom_legacy_keys({_event_run_key(e) for e in events if e.get("run_id")}))

        summary = MonitoringSummary(
            current_balance_cu=latest.balance_cu if latest else None,
            current_rate_cu_per_hour=latest.rate_cu_per_hour if latest else None,
            active_assignments=latest.active_assignments if latest else None,
            managed_running_runs=running_count,
            total_runs=unique_runs if project is not None else max(unique_runs, len(all_managed)),
            monitoring_since=col.monitoring_since,
            last_sample_time=latest.observed_at if latest else None,
            is_stale=col.is_stale,
        )
        body = summary.to_dict()
        # Managed sessions carry no project id, so balance/rate/managed counts stay
        # account-wide; only run counts are narrowed by the project filter.
        body["managed_counts_scope"] = "account"
        body["project_filter"] = project
        self._send_json(body)

    def _handle_compute_samples(self, query: Dict[str, List[str]]) -> None:
        from_ts = query.get("from", [None])[0]
        to_ts = query.get("to", [None])[0]
        limit = int(query.get("limit", [1440])[0])

        samples = self.server.collector.get_samples(from_ts=from_ts, to_ts=to_ts, limit=limit)
        self._send_json([s.to_dict() for s in samples])

    def _handle_sessions(self) -> None:
        all_managed = self.server.registry.get_all_managed()
        items = []
        for s_id, info in all_managed.items():
            items.append({
                "session_id": s_id,
                "project_name": info.get("project_name", "unknown"),
                "accelerator": info.get("gpu") or info.get("accelerator", "T4"),
                "state": info.get("state", "UNKNOWN"),
                "managed": True,
                "created_at": info.get("created_at"),
                "expires_at": info.get("expires_at"),
            })
        self._send_json(items)

    # ---- RunKey helpers (C4: RunKey = (project_id, run_id)) -------------

    def _projects_root(self) -> Path:
        artifacts_root = getattr(self.server, "artifacts_dir", Path("./artifacts").resolve())
        return Path(artifacts_root) / "projects"

    def _disk_run_dirs(
        self,
        project_filter: Optional[str] = None,
        run_filter: Optional[str] = None,
    ) -> Dict[Tuple[str, str], Path]:
        """Map RunKey -> run directory found on disk under projects/<project>/runs/<run>."""
        found: Dict[Tuple[str, str], Path] = {}
        proj_base = self._projects_root()
        if proj_base.is_symlink() or not proj_base.is_dir():
            return found
        if project_filter:
            p_dirs = [proj_base / project_filter]
        else:
            p_dirs = sorted(d for d in proj_base.iterdir() if d.is_dir())
        for p_dir in p_dirs:
            # Symlinked project dirs could alias another project's runs (D10)
            if p_dir.is_symlink() or not p_dir.is_dir():
                continue
            runs_dir = p_dir / "runs"
            if runs_dir.is_symlink() or not runs_dir.is_dir():
                continue
            for r_dir in runs_dir.iterdir():
                if r_dir.is_symlink() or not r_dir.is_dir():
                    continue
                if run_filter and r_dir.name != run_filter:
                    continue
                found[(p_dir.name, r_dir.name)] = r_dir
        return found

    def _handle_runs(self, query: Dict[str, List[str]]) -> None:
        proj_filter = query.get("project", [None])[0]
        if proj_filter is not None and not _is_valid_identifier(proj_filter):
            self._send_json(
                {"error": "Invalid project", "message": "Project selector contains illegal characters."},
                status=400,
            )
            return
        events = self.server.telemetry.get_events(project_id=proj_filter)

        runs_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        events_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        manifests_by_key: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
        for ev in events:
            r_id = ev.get("run_id")
            if not r_id:
                continue
            key = _event_run_key(ev)
            events_by_key.setdefault(key, []).append(ev)

            if key not in runs_map:
                runs_map[key] = {
                    "run_id": r_id,
                    "project_id": key[0],
                    "status": "RUNNING",
                    "start_time": ev.get("emitted_at"),
                    "completion_time": None,
                    "latest_epoch": None,
                    "latest_step": None,
                    "latest_metrics": {},
                    "checkpoint_count": 0,
                }

            run = runs_map[key]
            ev_type = ev.get("event_type")
            if ev_type == "run_finished":
                run["status"] = "COMPLETED"
                run["completion_time"] = ev.get("emitted_at")
            elif ev_type == "error":
                run["status"] = "FAILED"
                run["completion_time"] = ev.get("emitted_at")
            elif ev_type == "checkpoint_saved":
                run["checkpoint_count"] += 1

            if ev.get("epoch") is not None:
                run["latest_epoch"] = ev.get("epoch")
            if ev.get("step") is not None:
                run["latest_step"] = ev.get("step")
            if ev.get("metrics"):
                run["latest_metrics"].update(ev.get("metrics", {}))

        # Enrich active runs and discover durable historical runs from disk
        for key, r_dir in self._disk_run_dirs(project_filter=proj_filter).items():
            p_id, r_id = key
            if key not in runs_map:
                runs_map[key] = {
                    "run_id": r_id,
                    "project_id": p_id,
                    "status": "COMPLETED",
                    "start_time": None,
                    "completion_time": None,
                    "latest_epoch": None,
                    "latest_step": None,
                    "latest_metrics": {},
                    "checkpoint_count": 0,
                }

            # Read manifest if present
            m_data = _load_run_manifest(r_dir, p_id, r_id)
            manifests_by_key[key] = m_data
            if m_data is not None:
                try:
                    run = runs_map[key]
                    if not run.get("start_time"):
                        run["start_time"] = m_data.get("created_at")
                    run_arts = self._collect_run_artifacts_and_drive(r_id, manifest_data=m_data, run_dir=r_dir, project_id=p_id)
                    chk_arts = [a for a in run_arts if a.get("artifact_type") == "checkpoint" or a.get("category") == "checkpoint"]
                    if chk_arts:
                        run["checkpoint_count"] = max(run.get("checkpoint_count", 0), len(chk_arts))
                        # Only an artifact the manifest head marks is_latest=True may name the latest
                        # checkpoint. No guessing from order: unknown leaves name/sha null.
                        latest_chk = next((a for a in chk_arts if a.get("is_latest") is True), None)
                        if latest_chk is not None:
                            run["latest_checkpoint_name"] = Path(latest_chk.get("logical_path") or latest_chk.get("path", "")).name
                            run["latest_checkpoint_sha256"] = latest_chk.get("sha256")
                            run["latest_checkpoint_state"] = "known"
                        else:
                            run["latest_checkpoint_name"] = None
                            run["latest_checkpoint_sha256"] = None
                            run["latest_checkpoint_state"] = "unknown"
                    drive_arts = [a for a in run_arts if a.get("storage_provider") == "google_drive" or a.get("remote_file_id") or a.get("remote_available")]
                    if drive_arts:
                        run["drive_publication_status"] = "PUBLISHED"
                        run["drive_file_id"] = drive_arts[-1].get("remote_file_id")
                    if any(a.get("artifact_type") == "source_snapshot" for a in run_arts):
                        run["source_snapshot_available"] = True
                except Exception:
                    pass

        # Check session active status
        all_managed = self.server.registry.get_all_managed()
        active_session_ids = {s_id for s_id, s_info in all_managed.items() if s_info.get("state") in ("READY", "RUNNING", "ACTIVE")}

        # A legacy keyless event stream must not appear as a phantom project next to a named owner of the run.
        # Filter BEFORE counting so phantom keys do not inflate run_id_counts.
        kept_keys = _drop_phantom_legacy_keys(runs_map.keys())
        for legacy_key in [k for k in runs_map if k not in kept_keys]:
            del runs_map[legacy_key]

        # The registry records no project/run id per session, so a session id is
        # attributed to a run only when exactly one run key carries that run id.
        run_id_counts: Dict[str, int] = {}
        for key in runs_map:
            run_id_counts[key[1]] = run_id_counts.get(key[1], 0) + 1

        ordered_keys = sorted(runs_map)
        for key in ordered_keys:
            run = runs_map[key]
            run["is_runtime_active"] = key[1] in active_session_ids and run_id_counts.get(key[1], 0) == 1
            if "drive_publication_status" not in run:
                run["drive_publication_status"] = _drive_publication_state(
                    False, manifests_by_key.get(key), events_by_key.get(key, [])
                )
            if "source_snapshot_available" not in run:
                run["source_snapshot_available"] = False

        self._send_json([runs_map[key] for key in ordered_keys])

    def _collect_run_artifacts_and_drive(
        self,
        run_id: str,
        manifest_data: Optional[Dict[str, Any]] = None,
        run_dir: Optional[Path] = None,
        project_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Collect and normalize run artifacts and historical revisions with drive status.

        The caller must pass the exact run directory (projects/<project>/runs/<run>)
        resolved from a RunKey. No cross-project lookup happens here (C4). Artifact
        ids are always prefixed with "<project>/<run>/" so they are unique per project.
        """
        raw_artifacts: Dict[str, Dict[str, Any]] = manifest_data.get("artifacts", {}) if manifest_data else {}
        latest_by_path: Dict[str, Any] = manifest_data.get("latest_by_path", {}) if manifest_data else {}

        parsed_artifacts: List[Dict[str, Any]] = []

        for art_key, raw_dict in raw_artifacts.items():
            a = dict(raw_dict)
            artifact_id = a.get("artifact_id") or art_key
            a["artifact_id"] = artifact_id
            sha256 = a.get("sha256") or ""
            a["sha256"] = sha256

            # Derive logical_path
            logical_path = a.get("logical_path") or (a.get("metadata") or {}).get("logical_path")
            if not logical_path and artifact_id:
                base_id = artifact_id.split("@sha256_")[0]
                parts = base_id.split("/", 2)
                logical_path = parts[-1] if len(parts) == 3 else base_id
            if not logical_path and a.get("path"):
                logical_path = Path(a["path"]).name
            a["logical_path"] = logical_path or ""

            # Derive revision_id
            revision_id = a.get("revision_id") or (a.get("metadata") or {}).get("revision_id")
            if not revision_id:
                revision_id = f"sha256_{sha256[:12]}" if sha256 else "unversioned"
            a["revision_id"] = revision_id

            # Step: if not present, set to None - never fabricate or guess!
            step_val = (a.get("metadata") or {}).get("step")
            if step_val is None:
                step_val = a.get("step")
            a["step"] = step_val

            # Timestamps
            a["created_at"] = a.get("created_at")
            a["verified_at"] = a.get("verified_at")

            # Remote availability
            a["remote_available"] = bool(a.get("remote_file_id"))

            # Category / type
            if not a.get("category"):
                a["category"] = a.get("artifact_type") or "checkpoint"

            parsed_artifacts.append(a)

        # Deduplicate manifest entries by (logical_path, sha256) preferring versioned artifact IDs
        deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for a in parsed_artifacts:
            key = (a["logical_path"], a["sha256"])
            if key in deduped:
                if "@sha256_" in a.get("artifact_id", "") and "@sha256_" not in deduped[key].get("artifact_id", ""):
                    deduped[key] = a
            else:
                deduped[key] = a
        parsed_artifacts = list(deduped.values())

        # Derive is_latest for each logical_path group ONLY from an authoritative manifest
        # head (manifest latest_by_path). The per-entry "is_latest" flags carried inside
        # manifest entries (e.g. copied from Drive/remote appProperties) are NOT authoritative
        # for display and are overwritten here. When there is no head for the logical path,
        # the head matches no listed revision, or it matches more than one, every revision
        # is unknown (None). Never guess from created_at, step or list order.
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for a in parsed_artifacts:
            groups.setdefault(a["logical_path"], []).append(a)

        for l_path, items in groups.items():
            target_id: Any = None
            if l_path in latest_by_path:
                target = latest_by_path[l_path]
                if isinstance(target, str):
                    target_id = target
                elif isinstance(target, dict):
                    target_id = target.get("artifact_id") or target.get("sha256")
            matches = [
                item for item in items
                if target_id and (item.get("artifact_id") == target_id or item.get("sha256") == target_id)
            ]
            if len(matches) == 1:
                for item in items:
                    item["is_latest"] = item is matches[0]
                    item["latest_state"] = "known"
            else:
                for item in items:
                    item["is_latest"] = None
                    item["latest_state"] = "unknown"

        # Merge with local files found on disk in run_dir (avoiding duplicates and staged files)
        if run_dir and Path(run_dir).is_dir():
            run_path = Path(run_dir).resolve()
            known_shas = {a["sha256"] for a in parsed_artifacts if a.get("sha256")}
            known_paths = set()
            # The mutable alias (checkpoints/<name>) is a cache of a manifest head; never list it as a second artifact.
            known_logical = {str(a.get("logical_path")).lstrip("/") for a in parsed_artifacts if a.get("logical_path")}
            for a in parsed_artifacts:
                p_str = a.get("path")
                if p_str and not p_str.startswith("http"):
                    try:
                        known_paths.add(Path(p_str).resolve())
                    except Exception:
                        pass

            for root, dirs, files in os.walk(run_path):
                dirs[:] = [d for d in dirs if d not in (".revisions", ".upload_staging", ".staged_uploads", ".transfer_locks", ".locks")]
                for fname in files:
                    if (
                        fname in ("manifest.json", "manifest.json.bak")
                        or fname.endswith(".bak")
                        or fname.endswith(".tmp")
                        or ".tmp." in fname
                        or ".part." in fname
                    ):
                        continue
                    if ".sha256_" in fname:
                        continue  # Skip internal revision-staged files
                    fpath = Path(root) / fname
                    try:
                        # Regular files only: never follow or describe a symlink, which may
                        # point into another project's storage.
                        st = os.lstat(fpath)
                        if not stat.S_ISREG(st.st_mode):
                            continue
                        resolved_fpath = fpath.resolve()
                        if resolved_fpath in known_paths:
                            continue
                        if fpath.relative_to(run_path).as_posix() in known_logical:
                            continue
                        f_size = st.st_size
                        f_sha: Optional[str] = None
                        if f_size <= _LOCAL_HASH_MAX_BYTES:
                            hasher = hashlib.sha256()
                            with open(fpath, "rb") as f:
                                for chunk in iter(lambda: f.read(65536), b""):
                                    hasher.update(chunk)
                            f_sha = hasher.hexdigest()
                            if f_sha in known_shas:
                                continue

                        rel = str(fpath.relative_to(run_path))
                        local_art = {
                            # Prefixed exactly once with "<project>/<run>/" below.
                            "artifact_id": rel if project_id else f"{run_id}/{rel}",
                            "logical_path": rel,
                            "revision_id": f"sha256_{f_sha[:12]}" if f_sha else "unversioned",
                            # Unmanifested: no manifest head exists, so latest is unknown.
                            "is_latest": None,
                            "latest_state": "unknown",
                            "step": None,
                            "size_bytes": f_size,
                            "sha256": f_sha,
                            "storage_provider": "local",
                            "remote_file_id": None,
                            "remote_available": False,
                            "created_at": _iso_now(),
                            "verified_at": None,
                            "artifact_type": "checkpoint" if "checkpoint" in rel else "output",
                            "category": "checkpoint" if "checkpoint" in rel else "output",
                        }
                        parsed_artifacts.append(local_art)
                    except Exception:
                        pass

        # Artifact ids must include the owning project and run (D10)
        if project_id:
            id_prefix = f"{project_id}/{run_id}/"
            for a in parsed_artifacts:
                aid = str(a.get("artifact_id") or "")
                if not aid.startswith(id_prefix):
                    a["artifact_id"] = id_prefix + aid

        # Sort the returned artifacts list chronologically
        def _global_sort(a):
            c_at = a.get("created_at") or ""
            s = a.get("step")
            s_num = s if isinstance(s, int) else -1
            return (c_at, s_num, a.get("logical_path") or "", a.get("artifact_id") or "")

        parsed_artifacts.sort(key=_global_sort)
        return parsed_artifacts

    def _resolve_run_key(
        self,
        run_id: str,
        project: Optional[str],
    ) -> Tuple[Optional[Tuple[str, str]], Optional[Tuple[int, Dict[str, Any]]]]:
        """Resolve (project_id, run_id) for a run endpoint.

        Returns (key, None) on success, or (None, (status, body)) for 404/409.
        Without ``project`` the legacy unqualified URL works only when exactly one
        project owns ``run_id``.
        """
        keys: Set[Tuple[str, str]] = {_event_run_key(ev) for ev in self.server.telemetry.get_events(run_id=run_id)}
        keys.update(self._disk_run_dirs(project_filter=project, run_filter=run_id).keys())
        if project is not None:
            keys = {k for k in keys if k[0] == project}
        # Legacy keyless events ("" project) never make a named-project run ambiguous;
        # they only resolve when no named project owns the run.
        if any(k[0] for k in keys):
            keys = {k for k in keys if k[0]}
        if not keys:
            body: Dict[str, Any] = {"error": "Run not found", "run_id": run_id}
            if project is not None:
                body["project"] = project
            return None, (404, body)
        projects = sorted({k[0] for k in keys})
        if len(projects) > 1:
            return None, (409, {"error": "ambiguous_run", "run_id": run_id, "projects": projects})
        return (projects[0], run_id), None

    def _handle_run_detail(self, path: str, query: Dict[str, List[str]]) -> None:
        parts = path.strip("/").split("/")
        # Path format: api/v1/runs/{run_id} or api/v1/runs/{run_id}/{sub_resource}
        if len(parts) < 4:
            self._send_json({"error": "Invalid run endpoint"}, status=400)
            return
        run_id = parts[3]
        if not _is_valid_identifier(run_id):
            self._send_json(
                {"error": "Invalid run_id", "message": "Identifier contains illegal characters or path traversal."},
                status=400,
            )
            return

        project = query.get("project", [None])[0]
        if project is not None and not _is_valid_identifier(project):
            self._send_json(
                {"error": "Invalid project", "message": "Project selector contains illegal characters."},
                status=400,
            )
            return

        key, failure = self._resolve_run_key(run_id, project)
        if failure is not None or key is None:
            status_code, body = failure or (404, {"error": "Run not found", "run_id": run_id})
            self._send_json(body, status=status_code)
            return
        project_id = key[0]

        events = self.server.telemetry.get_events(run_id=run_id, project_id=project_id)
        r_dir = self._disk_run_dirs(project_filter=project_id, run_filter=run_id).get(key)
        manifest_data: Optional[Dict[str, Any]] = (
            _load_run_manifest(r_dir, project_id, run_id) if r_dir is not None else None
        )

        artifacts_list = self._collect_run_artifacts_and_drive(
            run_id, manifest_data=manifest_data, run_dir=r_dir, project_id=project_id
        )
        drive_artifacts = [a for a in artifacts_list if a.get("storage_provider") == "google_drive" or a.get("remote_file_id") or a.get("remote_available")]
        status_str = _drive_publication_state(bool(drive_artifacts), manifest_data, events)

        if len(parts) == 4:
            if not events and not manifest_data and not artifacts_list:
                self._send_json({"error": "Run not found", "run_id": run_id}, status=404)
                return
            self._send_json({
                "run_id": run_id,
                "project_id": project_id,
                "event_count": len(events),
                "artifacts_count": len(artifacts_list),
                "artifacts": artifacts_list,
                "events": events,
            })
        elif len(parts) == 5 and parts[4] == "artifacts":
            self._send_json({"run_id": run_id, "project_id": project_id, "artifacts": artifacts_list})
        elif len(parts) == 5 and parts[4] == "drive":
            self._send_json({
                "run_id": run_id,
                "project_id": project_id,
                "drive_publication_status": status_str,
                "published_count": len(drive_artifacts),
                "artifacts": drive_artifacts,
            })
        elif len(parts) == 5 and parts[4] == "manifest":
            if not manifest_data:
                self._send_json({"error": "Manifest not found for run", "run_id": run_id, "project_id": project_id}, status=404)
                return
            self._send_json(manifest_data)
        elif len(parts) == 5 and parts[4] == "metrics":
            points = [
                {
                    "event_id": e.get("event_id"),
                    "emitted_at": e.get("emitted_at"),
                    "epoch": e.get("epoch"),
                    "step": e.get("step"),
                    "metrics": e.get("metrics", {}),
                }
                for e in events if e.get("metrics")
            ]
            self._send_json(points)
        elif len(parts) == 5 and parts[4] == "logs":
            logs = []
            for e in events:
                if e.get("details", {}).get("log_line"):
                    logs.append(e["details"]["log_line"])
            self._send_json({"run_id": run_id, "project_id": project_id, "logs": logs})
        else:
            self._send_json({"error": "Invalid run endpoint"}, status=404)


    def _handle_events_stream(self, query: Dict[str, List[str]]) -> None:
        """SSE event stream supporting cursor resume (?from_id=...) (Q27)."""
        from_id = query.get("from_id", [None])[0] or self.headers.get("Last-Event-ID")
        project = query.get("project", [None])[0]
        if project is not None and not _is_valid_identifier(project):
            self._send_json(
                {"error": "Invalid project", "message": "Project selector contains illegal characters."},
                status=400,
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._send_cors_and_security_headers(csp_nonce=None)
        self.end_headers()

        events = self.server.telemetry.get_events(project_id=project)
        # Find index of cursor if provided
        start_idx = 0
        if from_id:
            for idx, ev in enumerate(events):
                if ev.get("event_id") == from_id:
                    start_idx = idx + 1
                    break

        for ev in events[start_idx:]:
            ev_id = ev.get("event_id", "")
            data_str = json.dumps(sanitize_data(ev), ensure_ascii=False)
            msg = f"id: {ev_id}\nevent: {ev.get('event_type', 'message')}\ndata: {data_str}\n\n"
            try:
                self.wfile.write(msg.encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break
        self.close_connection = True

    def _handle_export_csv(self) -> None:
        """CSV export of account compute snapshots (Q26)."""
        samples = self.server.collector.get_samples(limit=10000)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "sample_id", "observed_at", "collected_at", "source", "balance_cu",
            "rate_cu_per_hour", "active_assignments", "net_balance_change",
            "change_type", "status"
        ])
        for s in samples:
            writer.writerow([
                s.sample_id, s.observed_at, s.collected_at, s.source, s.balance_cu,
                s.rate_cu_per_hour, s.active_assignments, s.net_balance_change,
                s.change_type, s.status
            ])

        csv_bytes = output.getvalue().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", "attachment; filename=\"colab_compute_usage.csv\"")
        self.send_header("Content-Length", str(len(csv_bytes)))
        self._send_cors_and_security_headers(csp_nonce=None)
        self.end_headers()
        self.wfile.write(csv_bytes)

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress routine request logging to prevent stdout pollution during tests
        pass


class MonitorServer(ThreadingHTTPServer):
    """Threaded HTTP server for Colab Monitoring Dashboard."""

    def __init__(
        self,
        server_address: Tuple[str, int],
        collector: AccountUsageCollector,
        registry: ManagedSessionRegistry,
        telemetry: TelemetryEmitter,
        auth_token: Optional[str] = None,
        allow_external: bool = False,
        artifacts_dir: Optional[Union[str, Path]] = None,
        broker: Optional[Any] = None,
        allowed_origins: Optional[List[str]] = None,
        require_tls: bool = False,
        trusted_proxy: bool = False,
        trusted_proxy_ips: Optional[Set[str]] = None,
        trusted_proxy_secret: Optional[str] = None,
        tls_certfile: Optional[Union[str, Path]] = None,
        tls_keyfile: Optional[Union[str, Path]] = None,
        dashboard_html: Optional[str] = None,
    ):
        host, port = server_address
        # Enforce loopback binding by default (Q28)
        # Set before any validation can raise so shutdown/server_close never hit a missing attribute.
        self._poller: Optional[MonitorPoller] = None
        if not allow_external and host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError(
                f"Security violation: Monitor server attempted to bind to external interface '{host}'. "
                "Only loopback (127.0.0.1) is permitted unless allow_external=True is explicitly set."
            )

        if trusted_proxy:
            if not trusted_proxy_secret or not isinstance(trusted_proxy_secret, str) or not trusted_proxy_secret.strip():
                raise ValueError(
                    "Security violation: trusted_proxy=True requires a non-empty trusted_proxy_secret to authenticate reverse-proxy identity headers."
                )

        if allow_external:
            if not auth_token and not trusted_proxy:
                raise ValueError(
                    "Security violation: Monitor server allow_external=True requires a non-empty auth_token or trusted_proxy=True."
                )
            if not require_tls and not trusted_proxy:
                raise ValueError(
                    "Security violation: External binding over unencrypted HTTP without a trusted identity proxy is rejected. "
                    "Set require_tls=True or trusted_proxy=True."
                )
            if trusted_proxy and host not in ("127.0.0.1", "localhost", "::1"):
                if not (tls_certfile and tls_keyfile):
                    raise ValueError(
                        "Security violation: trusted_proxy mode cannot bind to an unencrypted external interface; "
                        "bind to loopback (127.0.0.1) behind the local reverse proxy or configure real TLS."
                    )
                if trusted_proxy_ips is None:
                    raise ValueError(
                        "Security violation: trusted_proxy mode on external interface requires explicit trusted_proxy_ips."
                    )

        if require_tls or (allow_external and host not in ("127.0.0.1", "localhost", "::1")):
            cert_p = Path(tls_certfile) if tls_certfile else None
            key_p = Path(tls_keyfile) if tls_keyfile else None
            if not cert_p or not key_p or not cert_p.is_file() or not key_p.is_file():
                if allow_external:
                    raise ValueError(
                        "Security violation: External binding over unencrypted HTTP without a trusted identity proxy is rejected. "
                        "Real TLS requires valid tls_certfile and tls_keyfile."
                    )
                raise ValueError("Security violation: Real TLS requires valid tls_certfile and tls_keyfile.")
        elif tls_certfile or tls_keyfile:
            cert_p = Path(tls_certfile) if tls_certfile else None
            key_p = Path(tls_keyfile) if tls_keyfile else None
            if not cert_p or not key_p or not cert_p.is_file() or not key_p.is_file():
                raise ValueError("Security violation: Real TLS requires valid tls_certfile and tls_keyfile.")

        super().__init__((host, port), MonitorRequestHandler)

        if require_tls or (tls_certfile and tls_keyfile):
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(certfile=str(tls_certfile), keyfile=str(tls_keyfile))
            self.socket = context.wrap_socket(self.socket, server_side=True)

        self.collector = collector
        self.registry = registry
        self.telemetry = telemetry
        self.auth_token = auth_token
        self.allow_external = allow_external
        self.allowed_origins = allowed_origins
        self.require_tls = require_tls
        self.trusted_proxy = trusted_proxy
        self.trusted_proxy_ips = set(trusted_proxy_ips) if trusted_proxy_ips is not None else {"127.0.0.1", "::1"}
        self.trusted_proxy_secret = trusted_proxy_secret
        self.tls_certfile = tls_certfile
        self.tls_keyfile = tls_keyfile
        self.dashboard_html = dashboard_html or _EMBEDDED_HTML_DASHBOARD
        self.artifacts_dir = Path(artifacts_dir or "./artifacts").resolve()
        self.broker = broker
        self.start_time = time.time()

    def _stop_poller(self) -> None:
        poller = getattr(self, "_poller", None)
        if poller is not None:
            poller.stop()

    def shutdown(self) -> None:
        self._stop_poller()
        super().shutdown()

    def server_close(self) -> None:
        self._stop_poller()
        super().server_close()


def start_monitor_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    collector: Optional[AccountUsageCollector] = None,
    registry: Optional[ManagedSessionRegistry] = None,
    telemetry: Optional[TelemetryEmitter] = None,
    auth_token: Optional[str] = None,
    allow_external: bool = False,
    artifacts_dir: Optional[Union[str, Path]] = None,
    broker: Optional[Any] = None,
    allowed_origins: Optional[List[str]] = None,
    require_tls: bool = False,
    trusted_proxy: bool = False,
    trusted_proxy_ips: Optional[Set[str]] = None,
    trusted_proxy_secret: Optional[str] = None,
    tls_certfile: Optional[Union[str, Path]] = None,
    tls_keyfile: Optional[Union[str, Path]] = None,
    dashboard_html: Optional[str] = None,
    poll_interval: Optional[float] = 60.0,
) -> MonitorServer:
    """Instantiate the MonitorServer and start its background account poller.

    ``poll_interval`` seconds between read-only ``colab usage`` polls (default 60, clamped to
    at least 15). ``None`` or ``0`` disables polling. The poller stops on ``shutdown()`` or
    ``server_close()``.
    """
    col = collector or AccountUsageCollector()
    reg = registry or ManagedSessionRegistry()
    tel = telemetry or TelemetryEmitter()

    server = MonitorServer(
        server_address=(host, port),
        collector=col,
        registry=reg,
        telemetry=tel,
        auth_token=auth_token,
        allow_external=allow_external,
        artifacts_dir=artifacts_dir,
        broker=broker,
        allowed_origins=allowed_origins,
        require_tls=require_tls,
        trusted_proxy=trusted_proxy,
        trusted_proxy_ips=trusted_proxy_ips,
        trusted_proxy_secret=trusted_proxy_secret,
        tls_certfile=tls_certfile,
        tls_keyfile=tls_keyfile,
        dashboard_html=dashboard_html,
    )
    effective_interval = clamp_poll_interval(poll_interval)
    if effective_interval is not None:
        server._poller = MonitorPoller(col, interval=effective_interval).start()
    return server

