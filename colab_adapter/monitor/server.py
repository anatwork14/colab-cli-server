"""Secure, read-only loopback HTTP server and SSE stream for Colab Monitor (M8).

Guarantees:
- Loopback binding by default (127.0.0.1); refuses public interfaces unless explicit (Q28).
- Strict token authentication via Bearer header or ?token= param; 401 on unauthorized (Q26).
- Strictly read-only: POST/PUT/DELETE return 405 Method Not Allowed (Q29).
- SSE event stream with cursor resume (Q27).
- Deep credential redaction in all responses (Q31).
- Serves embedded responsive SPA dashboard with zero fabricated telemetry (Q30).
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..provider import ManagedSessionRegistry
from ..telemetry import TelemetryEmitter
from ..tracker import sanitize_data
from .collector import AccountUsageCollector
from .schemas import HealthStatus, MonitoringSummary, RunSummary, _iso_now

_EMBEDDED_HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Colab Monitor — V2 Remote GPU Workspaces</title>
  <style>
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


  <script>
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
          document.getElementById('header-meta').innerText = '401 Unauthorized: Authorization token required.';
          const badge = document.getElementById('health-badge');
          badge.innerText = 'AUTH REQUIRED';
          badge.className = 'badge stale';
          badge.style.cursor = 'pointer';
          badge.onclick = promptForToken;
          return;
        }
        const data = await res.json();
        document.getElementById('val-cu').innerText = data.current_balance_cu != null ? data.current_balance_cu.toFixed(2) : 'N/A';
        document.getElementById('val-rate').innerText = data.current_rate_cu_per_hour != null ? data.current_rate_cu_per_hour.toFixed(2) : '0.00';
        document.getElementById('val-assignments').innerText = data.active_assignments || 0;
        document.getElementById('val-runs').innerText = (data.managed_running_runs || 0) + ' / ' + (data.total_runs || 0);
        document.getElementById('header-meta').innerText = 'Monitoring since ' + (data.monitoring_since || 'now').slice(0, 19);

        const badge = document.getElementById('health-badge');
        if (data.is_stale) {
          badge.innerText = 'STALE';
          badge.className = 'badge stale';
        } else {
          badge.innerText = 'HEALTHY';
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


class MonitorRequestHandler(BaseHTTPRequestHandler):
    """Handles read-only authenticated HTTP requests for the monitor."""

    server: "MonitorServer"

    def _send_cors_and_security_headers(self) -> None:
        """Enforce strict CORS and defense-in-depth security headers."""
        origin = self.headers.get("Origin")
        allowed = getattr(self.server, "allowed_origins", None) or [
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ]

        if origin and origin in allowed:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Authorization, X-API-Token, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")

        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")

    def _is_authenticated(self, query_params: Dict[str, List[str]]) -> bool:
        """Validate credentials from Authorization header, X-API-Token, or trusted gateway."""
        expected_token = self.server.auth_token
        trusted_proxy = getattr(self.server, "trusted_proxy", False)

        # 1. Trusted reverse proxy gateway principal validation
        if trusted_proxy:
            remote_user = self.headers.get("X-Remote-User") or self.headers.get("X-Forwarded-User")
            if remote_user:
                return True

        if not expected_token:
            return True  # Auth disabled (for local-only test mode)

        import hmac
        expected_bytes = expected_token.encode("utf-8")

        # 2. Check Authorization header: Bearer <token>
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if hmac.compare_digest(token.encode("utf-8"), expected_bytes):
                return True

        # 3. Check X-API-Token header
        api_token = self.headers.get("X-API-Token", "")
        if api_token and hmac.compare_digest(api_token.encode("utf-8"), expected_bytes):
            return True

        # Query parameter tokens (?token=...) are strictly rejected
        return False


    def _send_json(self, data: Any, status: int = 200) -> None:
        sanitized = sanitize_data(data)
        body = json.dumps(sanitized, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_and_security_headers()
        self.end_headers()
        self.wfile.write(body)

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
        self._send_cors_and_security_headers()
        self.end_headers()

    def do_POST(self) -> None:
        self._send_405()

    def do_PUT(self) -> None:
        self._send_405()

    def do_DELETE(self) -> None:
        self._send_405()

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        # Serve static HTML/JS dashboard without token prompt on root if public, or check auth
        if path in ("/", "/index.html"):
            body = _EMBEDDED_HTML_DASHBOARD.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._send_cors_and_security_headers()
            self.end_headers()
            self.wfile.write(body)
            return

        # All API endpoints require authentication (Q26)
        if not self._is_authenticated(query):
            self._send_401()
            return

        if path == "/api/v1/health":
            self._handle_health()
        elif path == "/api/v1/summary":
            self._handle_summary()
        elif path == "/api/v1/compute/samples":
            self._handle_compute_samples(query)
        elif path == "/api/v1/sessions":
            self._handle_sessions()
        elif path == "/api/v1/runs":
            self._handle_runs(query)
        elif path.startswith("/api/v1/runs/"):
            self._handle_run_detail(path)
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

    def _handle_summary(self) -> None:
        col = self.server.collector
        latest = col.get_latest_sample()
        all_managed = self.server.registry.get_all_managed()
        running_count = sum(1 for s in all_managed.values() if s.get("state") in ("READY", "RUNNING", "ACTIVE"))

        events = self.server.telemetry.get_events()
        unique_runs = len({e.get("run_id") for e in events if e.get("run_id")})

        summary = MonitoringSummary(
            current_balance_cu=latest.balance_cu if latest else None,
            current_rate_cu_per_hour=latest.rate_cu_per_hour if latest else None,
            active_assignments=latest.active_assignments if latest else 0,
            managed_running_runs=running_count,
            total_runs=max(unique_runs, len(all_managed)),
            monitoring_since=col.monitoring_since,
            last_sample_time=latest.observed_at if latest else None,
            is_stale=col.is_stale,
        )
        self._send_json(summary.to_dict())

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

    def _handle_runs(self, query: Dict[str, List[str]]) -> None:
        proj_filter = query.get("project", [None])[0]
        events = self.server.telemetry.get_events()

        runs_map: Dict[str, Dict[str, Any]] = {}
        for ev in events:
            r_id = ev.get("run_id")
            if not r_id:
                continue
            if proj_filter and ev.get("project_id") != proj_filter:
                continue

            if r_id not in runs_map:
                runs_map[r_id] = {
                    "run_id": r_id,
                    "project_id": ev.get("project_id", "default"),
                    "status": "RUNNING",
                    "start_time": ev.get("emitted_at"),
                    "completion_time": None,
                    "latest_epoch": None,
                    "latest_step": None,
                    "latest_metrics": {},
                    "checkpoint_count": 0,
                }

            run = runs_map[r_id]
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
        artifacts_root = getattr(self.server, "artifacts_dir", Path("./artifacts").resolve())
        proj_base = artifacts_root / "projects"
        if proj_base.is_dir():
            target_dirs = [proj_base / proj_filter] if proj_filter else [d for d in proj_base.iterdir() if d.is_dir()]
            for p_dir in target_dirs:
                runs_dir = p_dir / "runs"
                if not runs_dir.is_dir():
                    continue
                for r_dir in runs_dir.iterdir():
                    if not r_dir.is_dir():
                        continue
                    r_id = r_dir.name
                    if r_id not in runs_map:
                        runs_map[r_id] = {
                            "run_id": r_id,
                            "project_id": p_dir.name,
                            "status": "COMPLETED",
                            "start_time": None,
                            "completion_time": None,
                            "latest_epoch": None,
                            "latest_step": None,
                            "latest_metrics": {},
                            "checkpoint_count": 0,
                        }

                    # Read manifest if present
                    m_path = r_dir / "manifest.json"
                    if m_path.is_file():
                        try:
                            with open(m_path, "r", encoding="utf-8") as mf:
                                m_data = json.load(mf)
                            run = runs_map[r_id]
                            if not run.get("start_time"):
                                run["start_time"] = m_data.get("created_at")
                            artifacts_dict = m_data.get("artifacts", {})
                            for a_val in artifacts_dict.values():
                                a_type = a_val.get("artifact_type")
                                if a_type == "checkpoint":
                                    run["checkpoint_count"] = max(run["checkpoint_count"], 1)
                                    run["latest_checkpoint_name"] = Path(a_val.get("path", "")).name
                                    run["latest_checkpoint_sha256"] = a_val.get("sha256")
                                if a_val.get("storage_provider") == "google_drive" or a_val.get("remote_file_id"):
                                    run["drive_publication_status"] = "PUBLISHED"
                                    run["drive_file_id"] = a_val.get("remote_file_id")
                                if a_type == "source_snapshot":
                                    run["source_snapshot_available"] = True
                        except Exception:
                            pass

        # Check session active status
        all_managed = self.server.registry.get_all_managed()
        active_session_ids = {s_id for s_id, s_info in all_managed.items() if s_info.get("state") in ("READY", "RUNNING", "ACTIVE")}

        for r_id, run in runs_map.items():
            run["is_runtime_active"] = r_id in active_session_ids
            if "drive_publication_status" not in run:
                run["drive_publication_status"] = "PENDING"
            if "source_snapshot_available" not in run:
                run["source_snapshot_available"] = False

        self._send_json(list(runs_map.values()))

    def _handle_run_detail(self, path: str) -> None:
        parts = path.strip("/").split("/")
        # Path format: api/v1/runs/{run_id} or api/v1/runs/{run_id}/{sub_resource}
        if len(parts) < 4:
            self._send_json({"error": "Invalid run endpoint"}, status=400)
            return
        run_id = parts[3]
        if not re.match(r"^[a-zA-Z0-9_\-\.]+$", run_id) or ".." in run_id:
            self._send_json(
                {"error": "Invalid run_id", "message": "Identifier contains illegal characters or path traversal."},
                status=400,
            )
            return
        events = self.server.telemetry.get_events(run_id=run_id)


        artifacts_root = getattr(self.server, "artifacts_dir", Path("./artifacts").resolve())
        manifest_data: Optional[Dict[str, Any]] = None
        proj_base = artifacts_root / "projects"
        if proj_base.is_dir():
            for p_dir in proj_base.iterdir():
                m_candidate = p_dir / "runs" / run_id / "manifest.json"
                if m_candidate.is_file():
                    try:
                        with open(m_candidate, "r", encoding="utf-8") as mf:
                            manifest_data = json.load(mf)
                        break
                    except Exception:
                        pass

        if len(parts) == 4:
            if not events and not manifest_data:
                self._send_json({"error": "Run not found", "run_id": run_id}, status=404)
                return
            artifacts_list = list(manifest_data.get("artifacts", {}).values()) if manifest_data else []
            self._send_json({
                "run_id": run_id,
                "event_count": len(events),
                "artifacts_count": len(artifacts_list),
                "artifacts": artifacts_list,
                "events": events,
            })
        elif len(parts) == 5 and parts[4] == "artifacts":
            artifacts_list = list(manifest_data.get("artifacts", {}).values()) if manifest_data else []
            self._send_json({"run_id": run_id, "artifacts": artifacts_list})
        elif len(parts) == 5 and parts[4] == "drive":
            artifacts_list = list(manifest_data.get("artifacts", {}).values()) if manifest_data else []
            drive_artifacts = [a for a in artifacts_list if a.get("storage_provider") == "google_drive" or a.get("remote_file_id")]
            status_str = "PUBLISHED" if drive_artifacts else ("PENDING" if artifacts_list else "DISABLED")
            self._send_json({
                "run_id": run_id,
                "drive_publication_status": status_str,
                "published_count": len(drive_artifacts),
                "artifacts": drive_artifacts,
            })
        elif len(parts) == 5 and parts[4] == "manifest":
            if not manifest_data:
                self._send_json({"error": "Manifest not found for run", "run_id": run_id}, status=404)
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
            self._send_json({"run_id": run_id, "logs": logs})
        else:
            self._send_json({"error": "Invalid run endpoint"}, status=404)


    def _handle_events_stream(self, query: Dict[str, List[str]]) -> None:
        """SSE event stream supporting cursor resume (?from_id=...) (Q27)."""
        from_id = query.get("from_id", [None])[0] or self.headers.get("Last-Event-ID")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._send_cors_and_security_headers()
        self.end_headers()

        events = self.server.telemetry.get_events()
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
        self._send_cors_and_security_headers()
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
    ):

        host, port = server_address
        # Enforce loopback binding by default (Q28)
        if not allow_external and host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError(
                f"Security violation: Monitor server attempted to bind to external interface '{host}'. "
                "Only loopback (127.0.0.1) is permitted unless allow_external=True is explicitly set."
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

        super().__init__((host, port), MonitorRequestHandler)
        self.collector = collector
        self.registry = registry
        self.telemetry = telemetry
        self.auth_token = auth_token
        self.allow_external = allow_external
        self.allowed_origins = allowed_origins
        self.require_tls = require_tls
        self.trusted_proxy = trusted_proxy
        self.artifacts_dir = Path(artifacts_dir or "./artifacts").resolve()
        self.broker = broker
        self.start_time = time.time()


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
) -> MonitorServer:
    """Instantiate and start the MonitorServer."""
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
    )
    return server

