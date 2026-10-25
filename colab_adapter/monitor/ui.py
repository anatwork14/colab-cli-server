"""Embedded responsive HTML5/CSS3/SVG dashboard UI for Colab Monitor (Milestone M8).

Provides:
- Embedded responsive single-page web application (`DASHBOARD_HTML`).
- Strict Bearer token authentication via sessionStorage (no URL token leakage).
- Client-side Blob download for compute CSV export without token in query string.
- Honest accounting: zero fabricated compute units or GPU telemetry.
- Explicit indicators for "No historical compute samples observed yet" and
  "GPU utilization: not measured" when telemetry is absent.
- Full inspection of active managed sessions, training runs, metrics, logs,
  and verified Google Drive durable artifacts.
"""
from __future__ import annotations

import html
import re
from typing import Optional

__all__ = [
    "DASHBOARD_HTML",
    "render_dashboard_html",
    "get_dashboard_css",
    "get_dashboard_js",
]


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Colab Experiment &amp; Compute Monitor</title>
  <style nonce="__CSP_NONCE__">
    :root {
      --bg: #0d1117;
      --card-bg: #161b22;
      --card-header: #21262d;
      --border: #30363d;
      --border-light: #484f58;
      --text: #c9d1d9;
      --text-bright: #f0f6fc;
      --text-dim: #8b949e;
      --accent: #58a6ff;
      --accent-hover: #79b8ff;
      --green: #3fb950;
      --yellow: #d29922;
      --red: #f85149;
      --purple: #bc8cff;
      --font-mono: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: var(--font-sans);
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 1rem 1.5rem 3rem 1.5rem;
    }

    .container {
      max-width: 1440px;
      margin: 0 auto;
    }

    /* Header */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 1rem 0 1.5rem 0;
      border-bottom: 1px solid var(--border);
      margin-bottom: 1.5rem;
      gap: 1rem;
      flex-wrap: wrap;
    }
    .header-brand h1 {
      font-size: 1.5rem;
      font-weight: 600;
      color: var(--text-bright);
      letter-spacing: -0.02em;
    }
    .header-brand .subtitle {
      font-size: 0.85rem;
      color: var(--text-dim);
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      flex-wrap: wrap;
    }
    .header-meta {
      font-size: 0.8rem;
      color: var(--text-dim);
    }

    /* Buttons */
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0.4rem 0.85rem;
      font-size: 0.85rem;
      font-weight: 500;
      border-radius: 6px;
      border: 1px solid var(--border-light);
      cursor: pointer;
      text-decoration: none;
      transition: all 0.15s ease;
      background: var(--card-bg);
      color: var(--text);
      min-height: 34px;
    }
    .btn:hover {
      background: var(--card-header);
      border-color: var(--text-dim);
      color: var(--text-bright);
    }
    .btn-primary {
      background: #238636;
      color: #ffffff;
      border-color: rgba(240, 246, 252, 0.1);
    }
    .btn-primary:hover {
      background: #2ea043;
      color: #ffffff;
    }
    .btn-secondary {
      background: var(--card-header);
      color: var(--text);
    }
    .btn-secondary:hover {
      background: #30363d;
      color: var(--text-bright);
    }
    .btn-success {
      background: #1f6feb;
      color: #ffffff;
      border-color: #388bfd;
    }
    .btn-sm {
      padding: 0.25rem 0.6rem;
      font-size: 0.8rem;
      min-height: 28px;
    }

    /* Badges */
    .badge {
      display: inline-flex;
      align-items: center;
      padding: 0.2rem 0.55rem;
      font-size: 0.75rem;
      font-weight: 600;
      border-radius: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .badge-success { background: rgba(63, 185, 80, 0.15); color: var(--green); border: 1px solid rgba(63, 185, 80, 0.4); }
    .badge-warning { background: rgba(210, 153, 34, 0.15); color: var(--yellow); border: 1px solid rgba(210, 153, 34, 0.4); }
    .badge-danger { background: rgba(248, 81, 73, 0.15); color: var(--red); border: 1px solid rgba(248, 81, 73, 0.4); }
    .badge-info { background: rgba(88, 166, 255, 0.15); color: var(--accent); border: 1px solid rgba(88, 166, 255, 0.4); }
    .badge-purple { background: rgba(188, 140, 255, 0.15); color: var(--purple); border: 1px solid rgba(188, 140, 255, 0.4); }
    .badge-neutral { background: rgba(139, 148, 158, 0.15); color: var(--text-dim); border: 1px solid var(--border); }
    .badge-managed { background: rgba(63, 185, 80, 0.15); color: var(--green); border: 1px solid rgba(63, 185, 80, 0.4); }

    /* Banners */
    .banner {
      padding: 0.75rem 1rem;
      border-radius: 6px;
      font-size: 0.85rem;
      margin-bottom: 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
    }
    .banner-danger { background: rgba(248, 81, 73, 0.1); border: 1px solid var(--red); color: #ff7b72; }
    .banner-warning { background: rgba(210, 153, 34, 0.1); border: 1px solid var(--yellow); color: #e3b341; }
    .banner-inner { display: flex; align-items: center; justify-content: space-between; width: 100%; gap: 1rem; flex-wrap: wrap; }
    .hidden { display: none !important; }

    /* Sections */
    .section {
      margin-bottom: 2rem;
    }
    .section-header {
      margin-bottom: 1rem;
    }
    .section-header h2 {
      font-size: 1.15rem;
      font-weight: 600;
      color: var(--text-bright);
    }
    .section-desc {
      font-size: 0.8rem;
      color: var(--text-dim);
    }

    /* Cards */
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 1.25rem;
      overflow: hidden;
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
      gap: 0.75rem;
      flex-wrap: wrap;
    }
    .card-title-group h3 {
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text-bright);
    }
    .card-subtitle {
      font-size: 0.78rem;
      color: var(--text-dim);
    }

    /* KPI Grid */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 1rem;
      margin-bottom: 1.25rem;
    }
    .kpi-card {
      padding: 1rem 1.1rem;
    }
    .kpi-label {
      font-size: 0.78rem;
      text-transform: uppercase;
      font-weight: 600;
      color: var(--text-dim);
      letter-spacing: 0.04em;
      margin-bottom: 0.4rem;
    }
    .kpi-value {
      font-size: 1.6rem;
      font-weight: 700;
      color: var(--text-bright);
      line-height: 1.2;
      margin-bottom: 0.25rem;
      font-family: var(--font-mono);
    }
    .kpi-meta {
      font-size: 0.78rem;
      color: var(--text-dim);
    }

    /* Chart */
    .chart-card {
      padding: 1rem 1.25rem;
    }
    .chart-container {
      width: 100%;
      min-height: 180px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow-x: auto;
    }
    .chart-svg {
      width: 100%;
      height: auto;
      max-height: 240px;
      display: block;
    }
    .chart-empty {
      color: var(--text-dim);
      font-size: 0.88rem;
      font-style: italic;
      padding: 2.5rem 1rem;
      text-align: center;
    }

    /* Tables */
    .table-card {
      padding: 0;
    }
    .table-responsive {
      width: 100%;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
    }
    .data-table {
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 0.85rem;
    }
    .data-table th {
      background: var(--card-header);
      color: var(--text-dim);
      font-weight: 600;
      padding: 0.65rem 0.85rem;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .data-table td {
      padding: 0.65rem 0.85rem;
      border-bottom: 1px solid var(--border);
      color: var(--text);
      white-space: nowrap;
    }
    .data-table tbody tr {
      transition: background 0.1s ease;
    }
    .data-table tbody tr.clickable {
      cursor: pointer;
    }
    .data-table tbody tr:hover {
      background: rgba(255, 255, 255, 0.02);
    }
    .data-table tbody tr.selected {
      background: rgba(88, 166, 255, 0.08);
      border-left: 3px solid var(--accent);
    }
    .table-empty {
      text-align: center;
      padding: 2rem 1rem;
      color: var(--text-dim);
      font-style: italic;
    }

    /* Filter Bar */
    .filter-bar {
      display: flex;
      align-items: center;
      gap: 1rem;
      margin-bottom: 1rem;
      flex-wrap: wrap;
    }
    .filter-group {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.85rem;
      color: var(--text-dim);
    }
    .filter-group input, .filter-group select {
      background: var(--card-bg);
      border: 1px solid var(--border);
      color: var(--text-bright);
      padding: 0.35rem 0.65rem;
      border-radius: 6px;
      font-size: 0.85rem;
      outline: none;
    }
    .filter-group input:focus, .filter-group select:focus {
      border-color: var(--accent);
    }
    .filter-actions {
      margin-left: auto;
    }

    /* Inspector */
    .empty-panel {
      text-align: center;
      padding: 3rem 1.5rem;
      color: var(--text-dim);
      font-style: italic;
    }
    .inspector-meta-card {
      display: flex;
      gap: 1.5rem;
      padding: 0.85rem 1.25rem;
      margin-bottom: 1rem;
      flex-wrap: wrap;
      background: var(--card-header);
    }
    .meta-item {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      font-size: 0.82rem;
    }
    .meta-label {
      color: var(--text-dim);
      font-size: 0.78rem;
    }
    .inspector-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      margin-bottom: 1rem;
    }
    .inspector-card {
      padding: 1.1rem;
    }
    .inspector-card.full-width {
      grid-column: 1 / -1;
    }
    .log-console {
      background: #090d13;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.75rem 1rem;
      font-family: var(--font-mono);
      font-size: 0.78rem;
      color: #7ee787;
      max-height: 240px;
      min-height: 160px;
      overflow-y: auto;
      white-space: pre-wrap;
      word-break: break-all;
    }

    /* Code & Tag styles */
    .mono { font-family: var(--font-mono); }
    .sha-tag, .sha-badge {
      font-family: var(--font-mono);
      font-size: 0.78rem;
      background: rgba(255, 255, 255, 0.05);
      padding: 0.15rem 0.4rem;
      border-radius: 4px;
      border: 1px solid var(--border);
      cursor: pointer;
      user-select: all;
    }
    .sha-tag:hover, .sha-badge:hover {
      border-color: var(--accent);
      color: var(--text-bright);
    }

    /* Footer */
    footer {
      margin-top: 3rem;
      padding-top: 1.5rem;
      border-top: 1px solid var(--border);
      text-align: center;
      font-size: 0.78rem;
      color: var(--text-dim);
    }

    /* Responsive Mobile & Tablet Rules */
    @media (max-width: 768px) {
      body {
        padding: 0.75rem;
      }
      header {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.75rem;
      }
      .header-actions {
        width: 100%;
        justify-content: flex-start;
      }
      .kpi-grid {
        grid-template-columns: 1fr;
      }
      .inspector-grid {
        grid-template-columns: 1fr;
      }
      .filter-bar {
        flex-direction: column;
        align-items: stretch;
      }
      .filter-group {
        width: 100%;
        justify-content: space-between;
      }
      .filter-group input, .filter-group select {
        flex: 1;
      }
      .filter-actions {
        margin-left: 0;
        width: 100%;
      }
      .filter-actions button {
        width: 100%;
      }
      .data-table th, .data-table td {
        padding: 0.5rem 0.6rem;
        font-size: 0.78rem;
      }
      .inspector-meta-card {
        flex-direction: column;
        gap: 0.5rem;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <!-- Header -->
    <header>
      <div class="header-brand">
        <h1>Colab Monitor</h1>
        <p class="subtitle">V2 Remote GPU Workspaces & Durable Artifacts</p>
      </div>
      <div class="header-actions">
        <span id="health-badge" class="badge badge-info">INITIALIZING</span>
        <span id="header-meta" class="header-meta">Checking health...</span>
        <input type="password" id="auth-token-input" class="hidden" placeholder="Bearer Token" autocomplete="off" />
        <button id="connect-btn" class="hidden" title="Connect">Connect</button>
        <button id="logout-btn" class="hidden" title="Logout">Logout</button>
        <button id="btn-refresh" class="btn btn-secondary btn-sm" title="Refresh all telemetry data">Refresh Now</button>
        <button id="btn-export-csv" class="btn btn-secondary" title="Export compute usage snapshots to CSV">Export Compute CSV</button>
        <button id="btn-set-token" data-testid="btn-auth" class="btn btn-primary" title="Set or manage Bearer token">Auth / Token</button>
      </div>
    </header>

    <!-- Global Banners -->
    <div id="auth-banner" class="banner banner-danger hidden"></div>
    <div id="health-banner" class="banner banner-warning hidden"></div>

    <!-- Section: Account Compute Units & Burn Rate -->
    <section class="section" id="section-compute">
      <div class="section-header">
        <h2>Account Compute Units & Burn Rate</h2>
        <span class="section-desc">Account-level telemetry from official Colab CLI</span>
      </div>

      <div class="kpi-grid">
        <!-- 1. Compute Units Balance -->
        <div class="card kpi-card">
          <div class="kpi-label">Compute Units Balance</div>
          <div class="kpi-value" id="val-cu">--</div>
          <div class="kpi-meta" id="meta-cu">Status: Pending</div>
        </div>

        <!-- 2. Consumption Rate -->
        <div class="card kpi-card">
          <div class="kpi-label">Consumption Rate</div>
          <div class="kpi-value" id="val-rate">--</div>
          <div class="kpi-meta" id="meta-rate">Burn rate per hour</div>
        </div>

        <!-- 3. Active Assignments -->
        <div class="card kpi-card">
          <div class="kpi-label">Active Assignments</div>
          <div class="kpi-value" id="val-assignments">--</div>
          <span id="stat-active-runs" class="hidden">--</span>
          <span id="stat-drive-synced" class="hidden">--</span>
          <span id="stat-transfer-health" class="hidden">--</span>
          <div class="kpi-meta">Allocated remote VMs</div>
        </div>

        <!-- 4. Managed Runs -->
        <div class="card kpi-card">
          <div class="kpi-label">Managed Runs</div>
          <div class="kpi-value" id="val-runs">--</div>
          <div class="kpi-meta">Active / total registered</div>
        </div>

        <!-- 5. Collector Health & Data Freshness -->
        <div class="card kpi-card">
          <div class="kpi-label">Collector Health & Freshness</div>
          <div class="kpi-value" id="val-collector-health">--</div>
          <div class="kpi-meta" id="meta-collector-health">Uptime & polling state</div>
        </div>
      </div>

      <!-- Compute Balance History Chart -->
      <div class="card chart-card">
        <div class="card-header">
          <h3>Compute Balance History</h3>
          <span class="card-subtitle">Last 240 observed account usage samples</span>
        </div>
        <div id="compute-chart-container" class="chart-container">
          <div class="chart-empty">No historical compute samples observed yet</div>
        </div>
      </div>
    </section>

    <!-- Section: Active Managed Sessions -->
    <section class="section" id="section-sessions">
      <div class="section-header">
        <h2>Active Managed Sessions</h2>
        <span class="section-desc">Verified remote GPU leases registered with Workspace Provider</span>
      </div>
      <div class="card table-card">
        <div class="table-responsive">
          <table class="data-table" id="sessions-table">
            <thead>
              <tr>
                <th>Session ID</th>
                <th>Project Name</th>
                <th>Accelerator (GPU)</th>
                <th>State</th>
                <th>Lease Expiration</th>
                <th>Verification</th>
              </tr>
            </thead>
            <tbody id="sessions-tbody">
              <tr><td colspan="6" class="table-empty">Loading active managed sessions...</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- Section: Training Runs & Historical Experiment Navigation -->
    <section class="section" id="section-runs">
      <div class="section-header">
        <h2>Training Runs</h2>
        <span class="section-desc">Real-time telemetry and durable Google Drive artifacts across VM lifecycles</span>
      </div>

      <div class="filter-bar">
        <div class="filter-group">
          <label for="filter-project">Project:</label>
          <input type="text" id="filter-project" placeholder="Filter by project ID..." />
        </div>
        <div class="filter-group">
          <label for="filter-status">Status:</label>
          <select id="filter-status">
            <option value="">All Statuses</option>
            <option value="RUNNING">RUNNING</option>
            <option value="COMPLETED">COMPLETED</option>
            <option value="FAILED">FAILED</option>
            <option value="TERMINATED">TERMINATED</option>
          </select>
        </div>
        <div class="filter-actions">
          <button id="btn-refresh-filter" class="btn btn-sm btn-secondary" title="Refresh telemetry data">Refresh Now</button>
        </div>
      </div>

      <div class="card table-card" id="runs-list">
        <div class="table-responsive">
          <table class="data-table" id="runs-table">
            <thead>
              <tr>
                <th>Run ID</th>
                <th>Project ID</th>
                <th>Status</th>
                <th>Runtime Status</th>
                <th>Epoch / Step</th>
                <th>Latest Metrics</th>
                <th>Checkpoints</th>
                <th>Drive Status</th>
                <th>Source Snapshot</th>
              </tr>
            </thead>
            <tbody id="runs-tbody">
              <tr><td colspan="9" class="table-empty">Loading training runs...</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- Section: Run Inspector Panel -->
    <section class="section" id="section-inspector">
      <div class="section-header">
        <h2>Run Inspector Panel</h2>
        <span class="section-desc" id="inspector-subtitle">Select a run above to inspect live metrics, logs, and artifacts</span>
      </div>

      <div id="inspector-placeholder" class="card empty-panel">
        <p>No run currently selected. Click any row in the Training Runs table above to inspect detailed metrics, logs, and verified Google Drive artifacts.</p>
      </div>

      <div id="inspector-content" class="hidden">
        <!-- Run Meta Bar -->
        <div class="card inspector-meta-card">
          <div class="meta-item">
            <span class="meta-label">Run ID:</span>
            <strong id="insp-run-id" class="mono">--</strong>
          </div>
          <div class="meta-item">
            <span class="meta-label">Project:</span>
            <span id="insp-project-id" class="mono">--</span>
          </div>
          <div class="meta-item">
            <span class="meta-label">Status:</span>
            <span id="insp-status" class="badge">--</span>
          </div>
          <div class="meta-item">
            <span class="meta-label">Runtime:</span>
            <span id="insp-runtime-status" class="badge">--</span>
          </div>
          <div class="meta-item">
            <span class="meta-label">Started:</span>
            <span id="insp-started-at" class="mono">--</span>
          </div>
          <div class="meta-item">
            <span class="meta-label">Completed:</span>
            <span id="insp-completed-at" class="mono">--</span>
          </div>
        </div>

        <div class="inspector-grid">
          <!-- Sub-panel: Loss & Evaluation Metrics -->
          <div class="card inspector-card">
            <div class="card-header">
              <div class="card-title-group">
                <h3>Loss & Evaluation Metrics</h3>
                <span class="card-subtitle">Loss/Metrics progression emitted from training loop</span>
              </div>
              <span id="metric-loss" class="hidden">--</span>
              <span id="metric-step" class="hidden">--</span>
              <span id="gpu-util-badge" class="badge badge-neutral">GPU utilization: not measured</span>
            </div>
            <div id="run-metrics-chart-container" class="chart-container">
              <div class="chart-empty">No metrics recorded for this run</div>
            </div>
          </div>

          <!-- Sub-panel: Execution Logs & Failures Viewer -->
          <div class="card inspector-card">
            <div class="card-header">
              <div class="card-title-group">
                <h3>Execution Logs & Failures Viewer</h3>
                <span class="card-subtitle">Redacted output and runtime error events</span>
              </div>
            </div>
            <div id="logs-box">
              <div id="log-error-banner" class="banner banner-danger hidden"></div>
              <pre id="log-console" class="log-console">No logs available for this run</pre>
            </div>
          </div>
        </div>

        <!-- Sub-panel: Checkpoints & Google Drive Artifacts Table -->
        <div class="card inspector-card full-width">
          <div class="card-header">
            <div class="card-title-group">
              <h3>Checkpoints & Google Drive Artifacts</h3>
              <span class="card-subtitle">Durable storage verification and SHA-256 integrity</span>
            </div>
            <span id="insp-drive-badge" class="badge badge-neutral">Drive Status: PENDING</span>
          </div>
          <div class="table-responsive">
            <table class="data-table" id="artifacts-table">
              <thead>
                <tr>
                  <th>Artifact</th>
                  <th>Revision ID</th>
                  <th>Step</th>
                  <th>Category</th>
                  <th>Size</th>
                  <th>Integrity &amp; Remote Status</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody id="artifacts-tbody">
                <tr><td colspan="7" class="table-empty">No artifacts recorded for this run</td></tr>
              </tbody>
            </table>
            <table class="data-table hidden" id="drive-records-table">
              <thead>
                <tr>
                  <th>Remote File ID</th>
                  <th>Path</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody id="drive-records-tbody"></tbody>
            </table>
          </div>
        </div>

      </div>
    </section>

    <!-- Footer -->
    <footer>
      <p>Colab CLI Server V2 Monitor &bull; Strictly Read-Only Architecture &bull; Loopback Bound</p>
    </footer>
  </div>

  <script nonce="__CSP_NONCE__">
    // State management
    let token = sessionStorage.getItem('colab_monitor_token') || '';
    let selectedRunId = null;
    let cachedRuns = [];

    const SVG_NS = 'http://www.w3.org/2000/svg';

    function getHeaders() {
      if (!token) {
        token = sessionStorage.getItem('colab_monitor_token') || '';
      }
      return token ? { 'Authorization': 'Bearer ' + token, 'Accept': 'application/json' } : { 'Accept': 'application/json' };
    }

    function formatNumber(val, decimals, fallback = '-') {
      if (val === null || val === undefined || val === '') return fallback;
      const n = Number(val);
      return Number.isFinite(n) ? n.toFixed(decimals) : fallback;
    }

    function formatBytes(bytes) {
      if (bytes == null || bytes === '') return '-';
      const n = Number(bytes);
      if (!Number.isFinite(n)) return '-';
      if (n === 0) return '0 B';
      const k = 1024;
      const sizes = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
      const i = Math.floor(Math.log(n) / Math.log(k));
      const safeI = Math.min(Math.max(0, i), sizes.length - 1);
      const val = n / Math.pow(k, safeI);
      return (Number.isFinite(val) ? val.toFixed(2) : '-') + ' ' + sizes[safeI];
    }

    function createSha256Badge(hash) {
      if (!hash) {
        const span = document.createElement('span');
        span.textContent = '-';
        return span;
      }
      const strHash = String(hash);
      const short = strHash.length > 12 ? strHash.slice(0, 8) + '...' + strHash.slice(-4) : strHash;
      const span = document.createElement('span');
      span.className = 'sha-badge';
      span.title = `Click to copy SHA-256: ${strHash}`;
      span.textContent = `${short} 📋`;
      span.addEventListener('click', (e) => {
        e.stopPropagation();
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(strHash);
        }
      });
      return span;
    }

    function renderSha256(hash) {
      return createSha256Badge(hash);
    }

    function createSvgElement(tag, attrs = {}, text = null) {
      const el = document.createElementNS(SVG_NS, tag);
      for (const [k, v] of Object.entries(attrs)) {
        el.setAttribute(k, String(v));
      }
      if (text !== null) {
        el.textContent = String(text);
      }
      return el;
    }

    function updateAuthButtonState() {
      if (!token) {
        token = sessionStorage.getItem('colab_monitor_token') || '';
      }
      const btn = document.getElementById('btn-set-token') || document.getElementById('btn-auth');
      if (!btn) return;
      if (token) {
        btn.textContent = 'Token: Set';
        btn.className = 'btn btn-success';
      } else {
        btn.textContent = 'Auth / Token';
        btn.className = 'btn btn-primary';
      }
    }

    function promptAuthToken() {
      const curInfo = token ? ' (current token configured)' : ' (no token set)';
      const entered = window.prompt(
        'Enter Monitor Authorization Bearer Token (leave empty to clear):' + curInfo,
        token
      );
      if (entered !== null) {
        const clean = entered.trim();
        if (clean) {
          token = clean;
          sessionStorage.setItem('colab_monitor_token', token);
        } else {
          token = '';
          sessionStorage.removeItem('colab_monitor_token');
        }
        updateAuthButtonState();
        hideAuthBanner();
        refreshAll();
      }
    }

    function showAuthBanner(message) {
      const banner = document.getElementById('auth-banner');
      if (banner) {
        const inner = document.createElement('div');
        inner.className = 'banner-inner';

        const span = document.createElement('span');
        const strong = document.createElement('strong');
        strong.textContent = '401 Unauthorized: ';
        span.appendChild(strong);
        span.appendChild(document.createTextNode(String(message || 'Valid Bearer token required.')));

        const btn = document.createElement('button');
        btn.id = 'btn-set-token-banner';
        btn.className = 'btn btn-sm btn-primary';
        btn.textContent = 'Set Bearer Token';
        btn.addEventListener('click', promptAuthToken);

        inner.appendChild(span);
        inner.appendChild(btn);
        banner.replaceChildren(inner);
        banner.classList.remove('hidden');
      }
      const badge = document.getElementById('health-badge');
      if (badge) {
        badge.textContent = 'AUTH REQUIRED';
        badge.className = 'badge badge-danger';
      }
      const meta = document.getElementById('header-meta');
      if (meta) meta.textContent = 'Authentication required';
    }

    function hideAuthBanner() {
      const banner = document.getElementById('auth-banner');
      if (banner) {
        banner.replaceChildren();
        banner.classList.add('hidden');
      }
    }

    function showHealthWarning(status, isStale, backoff, lastCollected) {
      const banner = document.getElementById('health-banner');
      if (!banner) return;
      if (status === 'ERROR' || status === 'DEGRADED' || isStale) {
        const reason = status === 'ERROR' ? 'Collector error detected' : (isStale ? 'Data is STALE' : 'Collector DEGRADED');
        const inner = document.createElement('div');
        inner.className = 'banner-inner';

        const span = document.createElement('span');
        const strong = document.createElement('strong');
        strong.textContent = `Collector Alert [${reason}]: `;
        span.appendChild(strong);
        span.appendChild(document.createTextNode(`Backoff level: ${backoff || 0}. Last collected: ${lastCollected || 'Never'}.`));

        const btn = document.createElement('button');
        btn.className = 'btn btn-sm btn-secondary';
        btn.textContent = 'Retry Now';
        btn.addEventListener('click', refreshAll);

        inner.appendChild(span);
        inner.appendChild(btn);
        banner.replaceChildren(inner);
        banner.classList.remove('hidden');
      } else {
        banner.replaceChildren();
        banner.classList.add('hidden');
      }
    }

    async function exportComputeCsv() {
      try {
        const res = await fetch('/api/v1/export/compute.csv', { headers: getHeaders() });
        if (res.status === 401) {
          showAuthBanner('401 Unauthorized: Bearer token required to export compute data.');
          return;
        }
        if (!res.ok) {
          alert('Export failed: HTTP ' + res.status + ' ' + res.statusText);
          return;
        }
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'colab_compute_usage.csv';
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
      } catch (err) {
        console.error('Export CSV error:', err);
        alert('Error exporting compute CSV: ' + err.message);
      }
    }

    function renderComputeChart(samples) {
      const container = document.getElementById('compute-chart-container');
      if (!container) return;

      container.replaceChildren();

      const emptyDiv = () => {
        const d = document.createElement('div');
        d.className = 'chart-empty';
        d.textContent = 'No historical compute samples observed yet';
        return d;
      };

      if (!samples || samples.length === 0) {
        container.appendChild(emptyDiv());
        return;
      }

      const valid = samples.filter(s => s && s.balance_cu != null);
      if (valid.length === 0) {
        container.appendChild(emptyDiv());
        return;
      }

      const width = 800;
      const height = 220;
      const padLeft = 65;
      const padRight = 30;
      const padTop = 25;
      const padBottom = 40;
      const plotW = width - padLeft - padRight;
      const plotH = height - padTop - padBottom;

      const balances = valid.map(s => Number(s.balance_cu)).filter(n => Number.isFinite(n));
      const rates = valid.map(s => (s.rate_cu_per_hour != null ? Number(s.rate_cu_per_hour) : 0)).filter(n => Number.isFinite(n));

      if (balances.length === 0) {
        container.appendChild(emptyDiv());
        return;
      }

      let minB = Math.min(...balances);
      let maxB = Math.max(...balances);
      if (minB === maxB) { minB -= 5; maxB += 5; }
      minB = Math.max(0, minB - 1);
      maxB = maxB + 1;

      let maxR = rates.length > 0 ? Math.max(...rates, 1.0) : 1.0;

      const n = valid.length;
      const pointsB = [];
      const pointsR = [];

      for (let i = 0; i < n; i++) {
        const x = padLeft + (n > 1 ? (i / (n - 1)) * plotW : plotW / 2);
        const bVal = Number(valid[i].balance_cu);
        const yB = padTop + plotH - ((bVal - minB) / (maxB - minB)) * plotH;
        pointsB.push(`${formatNumber(x, 1)},${formatNumber(yB, 1)}`);

        const r = valid[i].rate_cu_per_hour != null ? Number(valid[i].rate_cu_per_hour) : 0;
        const yR = padTop + plotH - (r / maxR) * plotH;
        pointsR.push(`${formatNumber(x, 1)},${formatNumber(yR, 1)}`);
      }

      const svg = createSvgElement('svg', {
        viewBox: `0 0 ${width} ${height}`,
        class: 'chart-svg',
        preserveAspectRatio: 'xMidYMid meet'
      });

      svg.appendChild(createSvgElement('line', {
        x1: padLeft, y1: padTop, x2: padLeft, y2: padTop + plotH,
        stroke: '#30363d', 'stroke-width': '1'
      }));
      svg.appendChild(createSvgElement('line', {
        x1: padLeft, y1: padTop + plotH, x2: padLeft + plotW, y2: padTop + plotH,
        stroke: '#30363d', 'stroke-width': '1'
      }));
      svg.appendChild(createSvgElement('line', {
        x1: padLeft, y1: padTop, x2: padLeft + plotW, y2: padTop,
        stroke: '#21262d', 'stroke-width': '1', 'stroke-dasharray': '2,2'
      }));

      svg.appendChild(createSvgElement('text', {
        x: padLeft - 8, y: padTop + 10, fill: '#8b949e', 'font-size': '11', 'text-anchor': 'end'
      }, `${formatNumber(maxB, 1)} CU`));
      svg.appendChild(createSvgElement('text', {
        x: padLeft - 8, y: padTop + plotH, fill: '#8b949e', 'font-size': '11', 'text-anchor': 'end'
      }, `${formatNumber(minB, 1)} CU`));

      const tStart = valid[0].observed_at ? String(valid[0].observed_at).slice(11, 19) : 'start';
      const tEnd = valid[n - 1].observed_at ? String(valid[n - 1].observed_at).slice(11, 19) : 'now';

      svg.appendChild(createSvgElement('text', {
        x: padLeft, y: height - 12, fill: '#8b949e', 'font-size': '11', 'text-anchor': 'start'
      }, tStart));
      svg.appendChild(createSvgElement('text', {
        x: width - padRight, y: height - 12, fill: '#8b949e', 'font-size': '11', 'text-anchor': 'end'
      }, tEnd));

      const g = createSvgElement('g', { transform: `translate(${padLeft + 10}, 16)` });
      g.appendChild(createSvgElement('line', {
        x1: '0', y1: '0', x2: '20', y2: '0', stroke: '#58a6ff', 'stroke-width': '2.5'
      }));
      g.appendChild(createSvgElement('text', {
        x: '25', y: '4', fill: '#c9d1d9', 'font-size': '11'
      }, 'Balance CU'));
      g.appendChild(createSvgElement('line', {
        x1: '120', y1: '0', x2: '140', y2: '0', stroke: '#3fb950', 'stroke-width': '1.8', 'stroke-dasharray': '4,4'
      }));
      g.appendChild(createSvgElement('text', {
        x: '145', y: '4', fill: '#c9d1d9', 'font-size': '11'
      }, `Rate CU/hr (max ${formatNumber(maxR, 1)})`));
      svg.appendChild(g);

      svg.appendChild(createSvgElement('polyline', {
        fill: 'none', stroke: '#58a6ff', 'stroke-width': '2.5', points: pointsB.join(' ')
      }));
      svg.appendChild(createSvgElement('polyline', {
        fill: 'none', stroke: '#3fb950', 'stroke-width': '1.8', 'stroke-dasharray': '4,4', points: pointsR.join(' ')
      }));

      container.appendChild(svg);
    }

    function renderMetricsChart(metricsList) {
      const container = document.getElementById('run-metrics-chart-container');
      const gpuBadge = document.getElementById('gpu-util-badge');
      if (!container) return;

      container.replaceChildren();

      const emptyDiv = (text) => {
        const d = document.createElement('div');
        d.className = 'chart-empty';
        d.textContent = text;
        return d;
      };

      // Extract GPU utilization strictly when present
      let measuredGpu = null;
      if (metricsList && metricsList.length > 0) {
        for (let i = metricsList.length - 1; i >= 0; i--) {
          const m = metricsList[i].metrics || {};
          if (m.gpu_util_pct != null) {
            measuredGpu = `${m.gpu_util_pct}%`;
            break;
          }
          if (m.gpu_mem_mb != null) {
            measuredGpu = `${m.gpu_mem_mb} MB`;
            break;
          }
        }
      }

      if (gpuBadge) {
        if (measuredGpu != null) {
          gpuBadge.className = 'badge badge-info';
          gpuBadge.textContent = `GPU Utilization: ${measuredGpu}`;
        } else {
          gpuBadge.className = 'badge badge-neutral';
          gpuBadge.textContent = 'GPU utilization: not measured';
        }
      }

      if (!metricsList || metricsList.length === 0) {
        container.appendChild(emptyDiv('No metrics recorded for this run'));
        return;
      }

      const lossPoints = [];
      const valPoints = [];

      for (let i = 0; i < metricsList.length; i++) {
        const pt = metricsList[i];
        const step = pt.step != null ? pt.step : (pt.epoch != null ? pt.epoch : i);
        const m = pt.metrics || {};
        if (m.loss != null && Number.isFinite(Number(m.loss))) {
          lossPoints.push({ step, val: Number(m.loss) });
        }
        if (m.val_loss != null && Number.isFinite(Number(m.val_loss))) {
          valPoints.push({ step, val: Number(m.val_loss) });
        } else if (m.eval_loss != null && Number.isFinite(Number(m.eval_loss))) {
          valPoints.push({ step, val: Number(m.eval_loss) });
        }
      }

      if (lossPoints.length === 0 && valPoints.length === 0) {
        container.appendChild(emptyDiv('No loss or evaluation metrics recorded for this run'));
        return;
      }

      const allVals = [...lossPoints.map(p => p.val), ...valPoints.map(p => p.val)];
      let minV = Math.min(...allVals);
      let maxV = Math.max(...allVals);
      if (minV === maxV) { minV -= 0.1; maxV += 0.1; }
      minV = Math.max(0, minV * 0.9);
      maxV = maxV * 1.05;

      const width = 600;
      const height = 200;
      const padLeft = 50;
      const padRight = 20;
      const padTop = 25;
      const padBottom = 35;
      const plotW = width - padLeft - padRight;
      const plotH = height - padTop - padBottom;
      const totalCount = Math.max(lossPoints.length, valPoints.length);

      function makePolyline(pts, color) {
        if (pts.length === 0) return null;
        const coords = pts.map((p, idx) => {
          const x = padLeft + (totalCount > 1 ? (idx / (totalCount - 1)) * plotW : plotW / 2);
          const y = padTop + plotH - ((p.val - minV) / (maxV - minV)) * plotH;
          return `${formatNumber(x, 1)},${formatNumber(y, 1)}`;
        });
        return createSvgElement('polyline', {
          fill: 'none', stroke: color, 'stroke-width': '2', points: coords.join(' ')
        });
      }

      const svg = createSvgElement('svg', {
        viewBox: `0 0 ${width} ${height}`,
        class: 'chart-svg',
        preserveAspectRatio: 'xMidYMid meet'
      });

      svg.appendChild(createSvgElement('line', {
        x1: padLeft, y1: padTop, x2: padLeft, y2: padTop + plotH,
        stroke: '#30363d', 'stroke-width': '1'
      }));
      svg.appendChild(createSvgElement('line', {
        x1: padLeft, y1: padTop + plotH, x2: padLeft + plotW, y2: padTop + plotH,
        stroke: '#30363d', 'stroke-width': '1'
      }));

      svg.appendChild(createSvgElement('text', {
        x: padLeft - 8, y: padTop + 10, fill: '#8b949e', 'font-size': '11', 'text-anchor': 'end'
      }, formatNumber(maxV, 2)));
      svg.appendChild(createSvgElement('text', {
        x: padLeft - 8, y: padTop + plotH, fill: '#8b949e', 'font-size': '11', 'text-anchor': 'end'
      }, formatNumber(minV, 2)));

      const g = createSvgElement('g', { transform: `translate(${padLeft + 10}, 16)` });
      if (lossPoints.length > 0) {
        g.appendChild(createSvgElement('line', {
          x1: '0', y1: '0', x2: '16', y2: '0', stroke: '#58a6ff', 'stroke-width': '2'
        }));
        g.appendChild(createSvgElement('text', {
          x: '22', y: '4', fill: '#c9d1d9', 'font-size': '11'
        }, 'Loss'));
      }
      if (valPoints.length > 0) {
        const offX = lossPoints.length > 0 ? 80 : 0;
        g.appendChild(createSvgElement('line', {
          x1: offX, y1: '0', x2: offX + 16, y2: '0', stroke: '#d29922', 'stroke-width': '2'
        }));
        g.appendChild(createSvgElement('text', {
          x: offX + 22, y: '4', fill: '#c9d1d9', 'font-size': '11'
        }, 'Val / Eval Loss'));
      }
      svg.appendChild(g);

      const lineLoss = makePolyline(lossPoints, '#58a6ff');
      if (lineLoss) svg.appendChild(lineLoss);
      const lineVal = makePolyline(valPoints, '#d29922');
      if (lineVal) svg.appendChild(lineVal);

      container.appendChild(svg);
    }

    function renderLogs(logsList, errorEvents) {
      const consoleEl = document.getElementById('log-console');
      const errorBanner = document.getElementById('log-error-banner');
      if (!consoleEl) return;

      if (errorBanner) {
        if (errorEvents && errorEvents.length > 0) {
          const headerDiv = document.createElement('div');
          const strong = document.createElement('strong');
          strong.textContent = 'Runtime Errors Detected:';
          headerDiv.appendChild(strong);

          const items = [headerDiv];
          for (const e of errorEvents) {
            const itemDiv = document.createElement('div');
            const msg = e?.details?.message || e?.details?.error || (typeof e?.details === 'object' ? JSON.stringify(e.details) : String(e?.details || ''));
            itemDiv.textContent = msg;
            items.push(itemDiv);
          }
          errorBanner.replaceChildren(...items);
          errorBanner.classList.remove('hidden');
        } else {
          errorBanner.replaceChildren();
          errorBanner.classList.add('hidden');
        }
      }

      if (!logsList || logsList.length === 0) {
        const emptySpan = document.createElement('span');
        emptySpan.style.color = 'var(--text-dim)';
        emptySpan.style.fontStyle = 'italic';
        emptySpan.textContent = 'No logs available for this run';
        consoleEl.replaceChildren(emptySpan);
        return;
      }

      consoleEl.replaceChildren();
      for (const line of logsList) {
        const lineDiv = document.createElement('div');
        lineDiv.textContent = String(line);
        consoleEl.appendChild(lineDiv);
      }
      consoleEl.scrollTop = consoleEl.scrollHeight;
    }

    function renderArtifacts(artifactsList, driveInfo) {
      const tbody = document.getElementById('artifacts-tbody');
      const driveBadge = document.getElementById('insp-drive-badge');
      if (!tbody) return;

      if (driveBadge) {
        const driveStatus = driveInfo?.drive_publication_status || (artifactsList && artifactsList.some(a => a.storage_provider === 'google_drive' || a.remote_file_id || a.remote_available) ? 'PUBLISHED' : (artifactsList && artifactsList.length ? 'PENDING' : 'DISABLED'));
        driveBadge.textContent = `Drive: ${driveStatus}`;
        if (driveStatus === 'PUBLISHED') driveBadge.className = 'badge badge-purple';
        else if (driveStatus === 'PENDING') driveBadge.className = 'badge badge-warning';
        else driveBadge.className = 'badge badge-neutral';
      }

      tbody.replaceChildren();

      if (!artifactsList || artifactsList.length === 0) {
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.colSpan = 7;
        td.className = 'table-empty';
        td.textContent = 'No artifacts recorded for this run';
        tr.appendChild(td);
        tbody.appendChild(tr);
        return;
      }

      for (const a of artifactsList) {
        const tr = document.createElement('tr');

        // Column 1: Logical path / filename + badge (LATEST vs HISTORICAL)
        const tdPath = document.createElement('td');
        const strongPath = document.createElement('strong');
        strongPath.className = 'mono';
        strongPath.textContent = a.logical_path || a.rel_path || a.name || a.artifact_id || '-';
        tdPath.appendChild(strongPath);

        const badge = document.createElement('span');
        const isLatest = a.is_latest !== false;
        badge.className = isLatest ? 'badge badge-green' : 'badge badge-neutral';
        badge.textContent = isLatest ? 'LATEST' : 'HISTORICAL';
        badge.style.marginLeft = '8px';
        tdPath.appendChild(badge);

        if (a.artifact_id && a.artifact_id !== strongPath.textContent) {
          const divSub = document.createElement('div');
          divSub.className = 'mono';
          divSub.style.fontSize = '0.75rem';
          divSub.style.color = 'var(--text-dim)';
          divSub.textContent = a.artifact_id;
          tdPath.appendChild(divSub);
        }
        tr.appendChild(tdPath);

        // Column 2: Revision ID
        const tdRev = document.createElement('td');
        tdRev.className = 'mono';
        tdRev.textContent = a.revision_id || (a.sha256 ? 'sha256_' + String(a.sha256).substring(0, 12) : 'n/a');
        tr.appendChild(tdRev);

        // Column 3: Step
        const tdStep = document.createElement('td');
        tdStep.textContent = a.step !== undefined && a.step !== null ? String(a.step) : 'unavailable';
        tr.appendChild(tdStep);

        // Column 4: Category
        const tdCat = document.createElement('td');
        const catBadge = document.createElement('span');
        catBadge.className = 'badge badge-info';
        catBadge.textContent = a.category || a.artifact_type || 'checkpoint';
        tdCat.appendChild(catBadge);
        tr.appendChild(tdCat);

        // Column 5: Size
        const tdSize = document.createElement('td');
        tdSize.textContent = formatBytes(a.size_bytes);
        tr.appendChild(tdSize);

        // Column 6: Integrity & Remote Status (VERIFIED / DRIVE / LOCAL + short SHA256)
        const tdStatus = document.createElement('td');
        const isDrive = a.storage_provider === 'google_drive' || Boolean(a.remote_file_id) || Boolean(a.remote_available);
        const statusText = a.verified_at ? 'VERIFIED' : (isDrive ? 'DRIVE' : 'LOCAL');
        const statusBadge = document.createElement('span');
        statusBadge.className = isDrive ? 'badge badge-purple' : 'badge badge-neutral';
        statusBadge.textContent = statusText;
        tdStatus.appendChild(statusBadge);

        if (a.sha256) {
          const shaSpan = createSha256Badge(a.sha256);
          shaSpan.style.marginLeft = '6px';
          tdStatus.appendChild(shaSpan);
        }
        tr.appendChild(tdStatus);

        // Column 7: Timestamp
        const tdTime = document.createElement('td');
        tdTime.className = 'mono';
        const ts = a.verified_at || a.created_at || '-';
        tdTime.textContent = ts !== '-' ? String(ts).slice(0, 19).replace('T', ' ') : '-';
        tr.appendChild(tdTime);

        tbody.appendChild(tr);
      }
    }

    async function selectRun(runId) {
      const safeRunId = String(runId || '');
      selectedRunId = safeRunId;
      document.querySelectorAll('#runs-tbody tr').forEach(row => {
        row.classList.toggle('selected', (row.dataset.runId || row.getAttribute('data-run-id')) === safeRunId);
      });

      const placeholder = document.getElementById('inspector-placeholder');
      const content = document.getElementById('inspector-content');
      const subtitle = document.getElementById('inspector-subtitle');

      if (placeholder) placeholder.classList.add('hidden');
      if (content) content.classList.remove('hidden');
      if (subtitle) subtitle.textContent = `Inspecting run: ${safeRunId}`;

      // Update meta bar from cached run info
      const run = cachedRuns.find(r => String(r.run_id) === safeRunId) || {};
      const elRunId = document.getElementById('insp-run-id');
      if (elRunId) elRunId.textContent = safeRunId;
      const elProjId = document.getElementById('insp-project-id');
      if (elProjId) elProjId.textContent = run.project_id || '-';

      const statusBadge = document.getElementById('insp-status');
      if (statusBadge) {
        const s = String(run.status || 'UNKNOWN');
        const cleanStatus = s.toLowerCase().replace(/[^a-z0-9_-]/g, '');
        statusBadge.textContent = s;
        if (s === 'COMPLETED') statusBadge.className = 'badge badge-success';
        else if (s === 'RUNNING') statusBadge.className = 'badge badge-info';
        else if (s === 'FAILED') statusBadge.className = 'badge badge-danger';
        else if (s === 'TERMINATED') statusBadge.className = 'badge badge-warning';
        else statusBadge.className = `badge badge-${cleanStatus}`;
      }

      const rtBadge = document.getElementById('insp-runtime-status');
      if (rtBadge) {
        const active = Boolean(run.is_runtime_active);
        rtBadge.textContent = active ? 'ACTIVE' : 'TERMINATED / DURABLE';
        rtBadge.className = active ? 'badge badge-success' : 'badge badge-neutral';
      }

      const elStarted = document.getElementById('insp-started-at');
      if (elStarted) elStarted.textContent = run.start_time ? String(run.start_time).slice(0, 19).replace('T', ' ') : '-';
      const elCompleted = document.getElementById('insp-completed-at');
      if (elCompleted) elCompleted.textContent = run.completion_time ? String(run.completion_time).slice(0, 19).replace('T', ' ') : '-';

      // Fetch detail endpoints with URL-encoded runId
      const encodedId = encodeURIComponent(safeRunId);
      try {
        const [detailRes, metricsRes, logsRes, driveRes, artifactsRes] = await Promise.all([
          fetch(`/api/v1/runs/${encodedId}`, { headers: getHeaders() }),
          fetch(`/api/v1/runs/${encodedId}/metrics`, { headers: getHeaders() }),
          fetch(`/api/v1/runs/${encodedId}/logs`, { headers: getHeaders() }),
          fetch(`/api/v1/runs/${encodedId}/drive`, { headers: getHeaders() }),
          fetch(`/api/v1/runs/${encodedId}/artifacts`, { headers: getHeaders() }),
        ]);

        if (detailRes.status === 401) { showAuthBanner(); return; }

        const detail = detailRes.ok ? await detailRes.json() : {};
        const metrics = metricsRes.ok ? await metricsRes.json() : [];
        const logs = logsRes.ok ? await logsRes.json() : {};
        const drive = driveRes.ok ? await driveRes.json() : {};
        const artifacts = artifactsRes.ok ? await artifactsRes.json() : {};

        renderMetricsChart(Array.isArray(metrics) ? metrics : []);
        const errEvents = (detail.events || []).filter(e => e.event_type === 'error');
        renderLogs(logs.logs || [], errEvents);
        renderArtifacts(artifacts.artifacts || detail.artifacts || [], drive);

      } catch (err) {
        console.error('Failed to load run detail:', err);
      }
    }

    function applyFilters() {
      const projFilter = (document.getElementById('filter-project')?.value || '').toLowerCase().trim();
      const statusFilter = document.getElementById('filter-status')?.value || '';

      const filtered = cachedRuns.filter(r => {
        if (projFilter && !r.project_id?.toLowerCase().includes(projFilter)) return false;
        if (statusFilter && r.status !== statusFilter) return false;
        return true;
      });

      renderRunsTable(filtered);
    }

    function renderRunsTable(runs) {
      const tbody = document.getElementById('runs-tbody');
      if (!tbody) return;

      tbody.replaceChildren();

      if (!runs || runs.length === 0) {
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.colSpan = 9;
        td.className = 'table-empty';
        td.textContent = 'No training runs found matching filter';
        tr.appendChild(td);
        tbody.appendChild(tr);
        return;
      }

      for (const r of runs) {
        const tr = document.createElement('tr');
        const runIdStr = String(r.run_id || '');
        tr.dataset.runId = runIdStr;
        tr.className = 'clickable' + (runIdStr === selectedRunId ? ' selected' : '');
        tr.addEventListener('click', () => selectRun(runIdStr));

        const tdRunId = document.createElement('td');
        tdRunId.className = 'mono';
        const strongRunId = document.createElement('strong');
        strongRunId.textContent = runIdStr;
        tdRunId.appendChild(strongRunId);
        tr.appendChild(tdRunId);

        const tdProj = document.createElement('td');
        tdProj.className = 'mono';
        tdProj.textContent = r.project_id || '-';
        tr.appendChild(tdProj);

        const tdStatus = document.createElement('td');
        const statusBadge = document.createElement('span');
        const s = String(r.status || 'UNKNOWN');
        const cleanStatus = s.toLowerCase().replace(/[^a-z0-9_-]/g, '');
        let statusClass = 'badge-neutral';
        if (s === 'COMPLETED') statusClass = 'badge-success';
        else if (s === 'RUNNING') statusClass = 'badge-info';
        else if (s === 'FAILED') statusClass = 'badge-danger';
        else if (s === 'TERMINATED') statusClass = 'badge-warning';
        else statusClass = `badge-${cleanStatus}`;
        statusBadge.className = `badge ${statusClass}`;
        statusBadge.textContent = s;
        tdStatus.appendChild(statusBadge);
        tr.appendChild(tdStatus);

        const tdRt = document.createElement('td');
        const rtBadge = document.createElement('span');
        const active = Boolean(r.is_runtime_active);
        rtBadge.className = active ? 'badge badge-success' : 'badge badge-neutral';
        rtBadge.textContent = active ? 'ACTIVE' : 'TERMINATED / DURABLE';
        tdRt.appendChild(rtBadge);
        tr.appendChild(tdRt);

        const tdEp = document.createElement('td');
        tdEp.className = 'mono';
        const epStr = (r.latest_epoch != null ? `ep ${r.latest_epoch}` : '') +
          (r.latest_step != null ? ` / step ${r.latest_step}` : (r.latest_epoch != null ? '' : '-'));
        tdEp.textContent = epStr;
        tr.appendChild(tdEp);

        const tdMetrics = document.createElement('td');
        tdMetrics.className = 'mono';
        tdMetrics.style.fontSize = '0.78rem';
        const m = r.latest_metrics || {};
        const metricParts = [];
        if (m.loss != null) metricParts.push(`loss: ${formatNumber(m.loss, 3)}`);
        if (m.val_loss != null) metricParts.push(`val: ${formatNumber(m.val_loss, 3)}`);
        else if (m.eval_loss != null) metricParts.push(`eval: ${formatNumber(m.eval_loss, 3)}`);
        if (m.accuracy != null) {
          const accNum = Number(m.accuracy);
          metricParts.push(`acc: ${Number.isFinite(accNum) ? (accNum * 100).toFixed(1) : '-'}%`);
        }
        if (m.gpu_util_pct != null) metricParts.push(`gpu: ${m.gpu_util_pct}%`);
        tdMetrics.textContent = metricParts.length > 0 ? metricParts.join(', ') : '-';
        tr.appendChild(tdMetrics);

        const tdChk = document.createElement('td');
        const chkCount = r.checkpoint_count || 0;
        const chkStr = chkCount > 0 ? String(chkCount) + (r.latest_checkpoint_name ? ` (${r.latest_checkpoint_name})` : '') : '0';
        tdChk.appendChild(document.createTextNode(chkStr));
        if (r.latest_checkpoint_sha256) {
          tdChk.appendChild(document.createTextNode(' '));
          tdChk.appendChild(createSha256Badge(r.latest_checkpoint_sha256));
        }
        tr.appendChild(tdChk);

        const tdDrive = document.createElement('td');
        const driveBadge = document.createElement('span');
        const dStatus = String(r.drive_publication_status || 'PENDING');
        const cleanDrive = dStatus.toLowerCase().replace(/[^a-z0-9_-]/g, '');
        let driveClass = 'badge-neutral';
        if (dStatus === 'PUBLISHED') driveClass = 'badge-purple';
        else if (dStatus === 'PENDING') driveClass = 'badge-warning';
        else driveClass = `badge-${cleanDrive}`;
        driveBadge.className = `badge ${driveClass}`;
        driveBadge.textContent = dStatus;
        tdDrive.appendChild(driveBadge);
        tr.appendChild(tdDrive);

        const tdSrc = document.createElement('td');
        if (r.source_snapshot_available) {
          const srcBadge = document.createElement('span');
          srcBadge.className = 'badge badge-success';
          srcBadge.textContent = 'Available';
          tdSrc.appendChild(srcBadge);
        } else {
          const dimSpan = document.createElement('span');
          dimSpan.className = 'text-dim';
          dimSpan.textContent = '-';
          tdSrc.appendChild(dimSpan);
        }
        tr.appendChild(tdSrc);

        tbody.appendChild(tr);
      }
    }

    function renderSessionsTable(sessions) {
      const tbody = document.getElementById('sessions-tbody');
      if (!tbody) return;

      tbody.replaceChildren();

      if (!sessions || sessions.length === 0) {
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.colSpan = 6;
        td.className = 'table-empty';
        td.textContent = 'No active managed Colab sessions';
        tr.appendChild(td);
        tbody.appendChild(tr);
        return;
      }

      for (const s of sessions) {
        const tr = document.createElement('tr');

        const tdId = document.createElement('td');
        tdId.className = 'mono';
        const strongId = document.createElement('strong');
        strongId.textContent = s.session_id || '-';
        tdId.appendChild(strongId);
        tr.appendChild(tdId);

        const tdProj = document.createElement('td');
        tdProj.className = 'mono';
        tdProj.textContent = s.project_name || 'unknown';
        tr.appendChild(tdProj);

        const tdAcc = document.createElement('td');
        const accBadge = document.createElement('span');
        accBadge.className = 'badge badge-info';
        accBadge.textContent = s.accelerator || 'T4';
        tdAcc.appendChild(accBadge);
        tr.appendChild(tdAcc);

        const tdState = document.createElement('td');
        const stateBadge = document.createElement('span');
        const st = String(s.state || 'UNKNOWN');
        const cleanState = st.toLowerCase().replace(/[^a-z0-9_-]/g, '');
        let stateClass = 'badge-neutral';
        if (st === 'READY' || st === 'RUNNING' || st === 'ACTIVE') stateClass = 'badge-success';
        else if (st === 'TERMINATED' || st === 'FAILED') stateClass = 'badge-danger';
        else stateClass = 'badge-warning';
        stateBadge.className = `badge ${stateClass}`;
        stateBadge.textContent = st;
        tdState.appendChild(stateBadge);
        tr.appendChild(tdState);

        const tdExp = document.createElement('td');
        tdExp.className = 'mono';
        tdExp.textContent = s.expires_at ? String(s.expires_at).slice(0, 19).replace('T', ' ') : '-';
        tr.appendChild(tdExp);

        const tdVer = document.createElement('td');
        const verBadge = document.createElement('span');
        verBadge.className = 'badge badge-managed';
        verBadge.textContent = 'Managed';
        tdVer.appendChild(verBadge);
        tr.appendChild(tdVer);

        tbody.appendChild(tr);
      }
    }

    async function refreshAll() {
      updateAuthButtonState();
      try {
        const [healthRes, summaryRes, samplesRes, sessionsRes, runsRes] = await Promise.all([
          fetch('/api/v1/health', { headers: getHeaders() }),
          fetch('/api/v1/summary', { headers: getHeaders() }),
          fetch('/api/v1/compute/samples?limit=240', { headers: getHeaders() }),
          fetch('/api/v1/sessions', { headers: getHeaders() }),
          fetch('/api/v1/runs', { headers: getHeaders() }),
        ]);

        if (healthRes.status === 401 || summaryRes.status === 401) {
          showAuthBanner('401 Unauthorized: Authorization token required.');
          return;
        }

        hideAuthBanner();

        // 1. Health
        if (healthRes.ok) {
          const h = await healthRes.json();
          const badge = document.getElementById('health-badge');
          if (badge) {
            badge.textContent = h.status || 'HEALTHY';
            if (h.status === 'HEALTHY' && !h.is_stale) badge.className = 'badge badge-success';
            else if (h.status === 'ERROR') badge.className = 'badge badge-danger';
            else badge.className = 'badge badge-warning';
          }
          const meta = document.getElementById('header-meta');
          if (meta) {
            const uptime = h.uptime_seconds != null ? `${Math.floor(Number(h.uptime_seconds) / 60)}m` : '0m';
            meta.textContent = `Uptime: ${uptime} • Samples: ${h.sample_count || 0}`;
          }

          const valCollector = document.getElementById('val-collector-health');
          if (valCollector) {
            valCollector.textContent = h.status || 'HEALTHY';
            valCollector.style.color = h.status === 'HEALTHY' ? 'var(--green)' : (h.status === 'ERROR' ? 'var(--red)' : 'var(--yellow)');
          }
          const metaCollector = document.getElementById('meta-collector-health');
          if (metaCollector) {
            const last = h.last_collected_at ? String(h.last_collected_at).slice(11, 19) : 'None';
            metaCollector.textContent = `Last: ${last} • Backoff: ${h.collector_backoff_level || 0}`;
          }

          showHealthWarning(h.status, h.is_stale, h.collector_backoff_level, h.last_collected_at);
        }

        // 2. Summary
        if (summaryRes.ok) {
          const s = await summaryRes.json();
          const valCu = document.getElementById('val-cu');
          if (valCu) {
            const cuNum = Number(s.current_balance_cu);
            valCu.textContent = (s.current_balance_cu != null && Number.isFinite(cuNum))
              ? `${formatNumber(cuNum, 2)} CU`
              : 'Unavailable';
          }
          const metaCu = document.getElementById('meta-cu');
          if (metaCu) {
            metaCu.textContent = s.is_stale ? 'Warning: Stale snapshot' : 'Official balance';
          }

          const valRate = document.getElementById('val-rate');
          if (valRate) {
            const rateNum = Number(s.current_rate_cu_per_hour);
            valRate.textContent = (s.current_rate_cu_per_hour != null && Number.isFinite(rateNum))
              ? `${formatNumber(rateNum, 2)} CU/hr`
              : '0.00 CU/hr';
          }

          const valAssign = document.getElementById('val-assignments');
          if (valAssign) {
            valAssign.textContent = s.active_assignments != null ? String(s.active_assignments) : '0';
          }

          const valRuns = document.getElementById('val-runs');
          if (valRuns) {
            valRuns.textContent = `${s.managed_running_runs || 0} / ${s.total_runs || 0}`;
          }
        }

        // 3. Compute Samples
        if (samplesRes.ok) {
          const samples = await samplesRes.json();
          renderComputeChart(Array.isArray(samples) ? samples : []);
        }

        // 4. Sessions
        if (sessionsRes.ok) {
          const sessions = await sessionsRes.json();
          renderSessionsTable(Array.isArray(sessions) ? sessions : []);
        }

        // 5. Runs
        if (runsRes.ok) {
          cachedRuns = await runsRes.json();
          applyFilters();
          if (selectedRunId) {
            selectRun(selectedRunId);
          }
        }

      } catch (err) {
        console.error('Refresh failed:', err);
      }
    }

    function initEventListeners() {
      const btnExport = document.getElementById('btn-export-csv');
      if (btnExport) btnExport.addEventListener('click', exportComputeCsv);

      const btnSetToken = document.getElementById('btn-set-token');
      if (btnSetToken) btnSetToken.addEventListener('click', promptAuthToken);

      const btnAuth = document.getElementById('btn-auth');
      if (btnAuth && btnAuth !== btnSetToken) btnAuth.addEventListener('click', promptAuthToken);

      const btnRefresh = document.getElementById('btn-refresh');
      if (btnRefresh) btnRefresh.addEventListener('click', refreshAll);

      const btnRefreshFilter = document.getElementById('btn-refresh-filter');
      if (btnRefreshFilter) btnRefreshFilter.addEventListener('click', refreshAll);

      const btnConnect = document.getElementById('connect-btn');
      if (btnConnect) {
        btnConnect.addEventListener('click', () => {
          const inp = document.getElementById('auth-token-input');
          if (inp && inp.value) {
            token = inp.value.trim();
            sessionStorage.setItem('colab_monitor_token', token);
            updateAuthButtonState();
            refreshAll();
          } else {
            promptAuthToken();
          }
        });
      }

      const btnLogout = document.getElementById('logout-btn');
      if (btnLogout) {
        btnLogout.addEventListener('click', () => {
          token = '';
          sessionStorage.removeItem('colab_monitor_token');
          updateAuthButtonState();
          refreshAll();
        });
      }

      const filterProj = document.getElementById('filter-project');
      if (filterProj) filterProj.addEventListener('input', applyFilters);

      const filterStatus = document.getElementById('filter-status');
      if (filterStatus) filterStatus.addEventListener('change', applyFilters);
    }

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initEventListeners);
    } else {
      initEventListeners();
    }

    // Auto-refresh cadence (10s)
    refreshAll();
    setInterval(refreshAll, 10000);
  </script>
</body>
</html>
"""


def render_dashboard_html(
    title: str = "Colab Experiment & Compute Monitor",
    csp_nonce: str = "",
) -> str:
    """Render the embedded Colab Monitor dashboard HTML document."""
    rendered = DASHBOARD_HTML
    if title:
        rendered = re.sub(
            r"<title>.*?</title>",
            f"<title>{html.escape(title)}</title>",
            rendered,
            count=1,
        )
    if csp_nonce:
        escaped_nonce = html.escape(csp_nonce, quote=True)
        rendered = rendered.replace('nonce="__CSP_NONCE__"', f'nonce="{escaped_nonce}"')
    else:
        rendered = rendered.replace(' nonce="__CSP_NONCE__"', "")
    return rendered


def get_dashboard_css() -> str:
    """Extract embedded CSS stylesheet from DASHBOARD_HTML."""
    match = re.search(r"<style[^>]*>(.*?)</style>", DASHBOARD_HTML, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def get_dashboard_js() -> str:
    """Extract embedded JavaScript client code from DASHBOARD_HTML."""
    match = re.search(r"<script[^>]*>(.*?)</script>", DASHBOARD_HTML, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()
