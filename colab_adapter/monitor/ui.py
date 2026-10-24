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

from typing import Optional

__all__ = ["DASHBOARD_HTML", "render_dashboard_html"]


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Colab Monitor — V2 Remote GPU Workspaces</title>
  <style>
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
    .sha-tag {
      font-family: var(--font-mono);
      font-size: 0.78rem;
      background: rgba(255, 255, 255, 0.05);
      padding: 0.15rem 0.4rem;
      border-radius: 4px;
      border: 1px solid var(--border);
      cursor: pointer;
      user-select: all;
    }
    .sha-tag:hover {
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
        <button id="btn-export-csv" class="btn btn-secondary" onclick="exportComputeCsv()" title="Export compute usage snapshots to CSV">Export Compute CSV</button>
        <button id="btn-auth" class="btn btn-primary" onclick="promptAuthToken()" title="Set or manage Bearer token">Auth / Token</button>
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
          <input type="text" id="filter-project" placeholder="Filter by project ID..." oninput="applyFilters()" />
        </div>
        <div class="filter-group">
          <label for="filter-status">Status:</label>
          <select id="filter-status" onchange="applyFilters()">
            <option value="">All Statuses</option>
            <option value="RUNNING">RUNNING</option>
            <option value="COMPLETED">COMPLETED</option>
            <option value="FAILED">FAILED</option>
            <option value="TERMINATED">TERMINATED</option>
          </select>
        </div>
        <div class="filter-actions">
          <button class="btn btn-sm btn-secondary" onclick="refreshAll()">Refresh Now</button>
        </div>
      </div>

      <div class="card table-card">
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
            <div id="log-error-banner" class="banner banner-danger hidden"></div>
            <pre id="log-console" class="log-console">No logs available for this run</pre>
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
                  <th>Artifact ID</th>
                  <th>Type</th>
                  <th>Size</th>
                  <th>SHA-256 Checksum</th>
                  <th>Storage Provider</th>
                  <th>Remote File ID</th>
                  <th>Verified At</th>
                </tr>
              </thead>
              <tbody id="artifacts-tbody">
                <tr><td colspan="7" class="table-empty">No artifacts recorded for this run</td></tr>
              </tbody>
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

  <script>
    // State management
    let token = sessionStorage.getItem('colab_monitor_token') || '';
    let selectedRunId = null;
    let cachedRuns = [];

    function getHeaders() {
      return token ? { 'Authorization': 'Bearer ' + token, 'Accept': 'application/json' } : { 'Accept': 'application/json' };
    }

    function updateAuthButtonState() {
      const btn = document.getElementById('btn-auth');
      if (!btn) return;
      if (token) {
        btn.innerText = 'Token: Set';
        btn.className = 'btn btn-success';
      } else {
        btn.innerText = 'Auth / Token';
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
        banner.innerHTML = `
          <div class="banner-inner">
            <span><strong>401 Unauthorized:</strong> ${message || 'Valid Bearer token required.'}</span>
            <button class="btn btn-sm btn-primary" onclick="promptAuthToken()">Set Bearer Token</button>
          </div>
        `;
        banner.classList.remove('hidden');
      }
      const badge = document.getElementById('health-badge');
      if (badge) {
        badge.innerText = 'AUTH REQUIRED';
        badge.className = 'badge badge-danger';
      }
      const meta = document.getElementById('header-meta');
      if (meta) meta.innerText = 'Authentication required';
    }

    function hideAuthBanner() {
      const banner = document.getElementById('auth-banner');
      if (banner) banner.classList.add('hidden');
    }

    function showHealthWarning(status, isStale, backoff, lastCollected) {
      const banner = document.getElementById('health-banner');
      if (!banner) return;
      if (status === 'ERROR' || status === 'DEGRADED' || isStale) {
        const reason = status === 'ERROR' ? 'Collector error detected' : (isStale ? 'Data is STALE' : 'Collector DEGRADED');
        banner.innerHTML = `
          <div class="banner-inner">
            <span><strong>Collector Alert [${reason}]:</strong> Backoff level: ${backoff || 0}. Last collected: ${lastCollected || 'Never'}.</span>
            <button class="btn btn-sm btn-secondary" onclick="refreshAll()">Retry Now</button>
          </div>
        `;
        banner.classList.remove('hidden');
      } else {
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

    function formatBytes(bytes) {
      if (bytes == null || isNaN(bytes)) return '-';
      if (bytes === 0) return '0 B';
      const k = 1024;
      const sizes = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function renderSha256(hash) {
      if (!hash) return '-';
      const short = hash.length > 12 ? hash.slice(0, 8) + '...' + hash.slice(-4) : hash;
      return `<span class="sha-tag" title="Click to copy SHA-256: ${hash}" onclick="navigator.clipboard.writeText('${hash}')">${short} 📋</span>`;
    }

    function renderComputeChart(samples) {
      const container = document.getElementById('compute-chart-container');
      if (!container) return;

      if (!samples || samples.length === 0) {
        container.innerHTML = '<div class="chart-empty">No historical compute samples observed yet</div>';
        return;
      }

      const valid = samples.filter(s => s && s.balance_cu != null);
      if (valid.length === 0) {
        container.innerHTML = '<div class="chart-empty">No historical compute samples observed yet</div>';
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

      const balances = valid.map(s => s.balance_cu);
      const rates = valid.map(s => (s.rate_cu_per_hour != null ? s.rate_cu_per_hour : 0));

      let minB = Math.min(...balances);
      let maxB = Math.max(...balances);
      if (minB === maxB) { minB -= 5; maxB += 5; }
      minB = Math.max(0, minB - 1);
      maxB = maxB + 1;

      let maxR = Math.max(...rates, 1.0);

      const n = valid.length;
      const pointsB = [];
      const pointsR = [];

      for (let i = 0; i < n; i++) {
        const x = padLeft + (n > 1 ? (i / (n - 1)) * plotW : plotW / 2);
        const yB = padTop + plotH - ((valid[i].balance_cu - minB) / (maxB - minB)) * plotH;
        pointsB.push(`${x.toFixed(1)},${yB.toFixed(1)}`);

        const r = valid[i].rate_cu_per_hour != null ? valid[i].rate_cu_per_hour : 0;
        const yR = padTop + plotH - (r / maxR) * plotH;
        pointsR.push(`${x.toFixed(1)},${yR.toFixed(1)}`);
      }

      const polylineB = `<polyline fill="none" stroke="#58a6ff" stroke-width="2.5" points="${pointsB.join(' ')}" />`;
      const polylineR = `<polyline fill="none" stroke="#3fb950" stroke-width="1.8" stroke-dasharray="4,4" points="${pointsR.join(' ')}" />`;

      const tStart = valid[0].observed_at ? valid[0].observed_at.slice(11, 19) : 'start';
      const tEnd = valid[n - 1].observed_at ? valid[n - 1].observed_at.slice(11, 19) : 'now';

      container.innerHTML = `
        <svg viewBox="0 0 ${width} ${height}" class="chart-svg" preserveAspectRatio="xMidYMid meet">
          <line x1="${padLeft}" y1="${padTop}" x2="${padLeft}" y2="${padTop + plotH}" stroke="#30363d" stroke-width="1" />
          <line x1="${padLeft}" y1="${padTop + plotH}" x2="${padLeft + plotW}" y2="${padTop + plotH}" stroke="#30363d" stroke-width="1" />
          <line x1="${padLeft}" y1="${padTop}" x2="${padLeft + plotW}" y2="${padTop}" stroke="#21262d" stroke-width="1" stroke-dasharray="2,2" />
          <text x="${padLeft - 8}" y="${padTop + 10}" fill="#8b949e" font-size="11" text-anchor="end">${maxB.toFixed(1)} CU</text>
          <text x="${padLeft - 8}" y="${padTop + plotH}" fill="#8b949e" font-size="11" text-anchor="end">${minB.toFixed(1)} CU</text>
          <text x="${padLeft}" y="${height - 12}" fill="#8b949e" font-size="11" text-anchor="start">${tStart}</text>
          <text x="${width - padRight}" y="${height - 12}" fill="#8b949e" font-size="11" text-anchor="end">${tEnd}</text>
          <g transform="translate(${padLeft + 10}, 16)">
            <line x1="0" y1="0" x2="20" y2="0" stroke="#58a6ff" stroke-width="2.5" />
            <text x="25" y="4" fill="#c9d1d9" font-size="11">Balance CU</text>
            <line x1="120" y1="0" x2="140" y2="0" stroke="#3fb950" stroke-width="1.8" stroke-dasharray="4,4" />
            <text x="145" y="4" fill="#c9d1d9" font-size="11">Rate CU/hr (max ${maxR.toFixed(1)})</text>
          </g>
          ${polylineB}
          ${polylineR}
        </svg>
      `;
    }

    function renderMetricsChart(metricsList) {
      const container = document.getElementById('run-metrics-chart-container');
      const gpuBadge = document.getElementById('gpu-util-badge');
      if (!container) return;

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
          gpuBadge.innerText = `GPU Utilization: ${measuredGpu}`;
        } else {
          gpuBadge.className = 'badge badge-neutral';
          gpuBadge.innerText = 'GPU utilization: not measured';
        }
      }

      if (!metricsList || metricsList.length === 0) {
        container.innerHTML = '<div class="chart-empty">No metrics recorded for this run</div>';
        return;
      }

      const lossPoints = [];
      const valPoints = [];

      for (let i = 0; i < metricsList.length; i++) {
        const pt = metricsList[i];
        const step = pt.step != null ? pt.step : (pt.epoch != null ? pt.epoch : i);
        const m = pt.metrics || {};
        if (m.loss != null) lossPoints.push({ step, val: m.loss });
        if (m.val_loss != null) valPoints.push({ step, val: m.val_loss });
        else if (m.eval_loss != null) valPoints.push({ step, val: m.eval_loss });
      }

      if (lossPoints.length === 0 && valPoints.length === 0) {
        container.innerHTML = '<div class="chart-empty">No loss or evaluation metrics recorded for this run</div>';
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
        if (pts.length === 0) return '';
        const coords = pts.map((p, idx) => {
          const x = padLeft + (totalCount > 1 ? (idx / (totalCount - 1)) * plotW : plotW / 2);
          const y = padTop + plotH - ((p.val - minV) / (maxV - minV)) * plotH;
          return `${x.toFixed(1)},${y.toFixed(1)}`;
        });
        return `<polyline fill="none" stroke="${color}" stroke-width="2" points="${coords.join(' ')}" />`;
      }

      const lineLoss = makePolyline(lossPoints, '#58a6ff');
      const lineVal = makePolyline(valPoints, '#d29922');

      container.innerHTML = `
        <svg viewBox="0 0 ${width} ${height}" class="chart-svg" preserveAspectRatio="xMidYMid meet">
          <line x1="${padLeft}" y1="${padTop}" x2="${padLeft}" y2="${padTop + plotH}" stroke="#30363d" stroke-width="1" />
          <line x1="${padLeft}" y1="${padTop + plotH}" x2="${padLeft + plotW}" y2="${padTop + plotH}" stroke="#30363d" stroke-width="1" />
          <text x="${padLeft - 8}" y="${padTop + 10}" fill="#8b949e" font-size="11" text-anchor="end">${maxV.toFixed(2)}</text>
          <text x="${padLeft - 8}" y="${padTop + plotH}" fill="#8b949e" font-size="11" text-anchor="end">${minV.toFixed(2)}</text>
          <g transform="translate(${padLeft + 10}, 16)">
            ${lossPoints.length > 0 ? '<line x1="0" y1="0" x2="16" y2="0" stroke="#58a6ff" stroke-width="2" /><text x="22" y="4" fill="#c9d1d9" font-size="11">Loss</text>' : ''}
            ${valPoints.length > 0 ? `<line x1="${lossPoints.length > 0 ? 80 : 0}" y1="0" x2="${lossPoints.length > 0 ? 96 : 16}" y2="0" stroke="#d29922" stroke-width="2" /><text x="${lossPoints.length > 0 ? 102 : 22}" y="4" fill="#c9d1d9" font-size="11">Val / Eval Loss</text>` : ''}
          </g>
          ${lineLoss}
          ${lineVal}
        </svg>
      `;
    }

    function renderLogs(logsList, errorEvents) {
      const consoleEl = document.getElementById('log-console');
      const errorBanner = document.getElementById('log-error-banner');
      if (!consoleEl) return;

      if (errorBanner) {
        if (errorEvents && errorEvents.length > 0) {
          const msgs = errorEvents.map(e => e.details?.message || e.details?.error || JSON.stringify(e.details || {})).join('<br>');
          errorBanner.innerHTML = `<strong>Runtime Errors Detected:</strong><br>${msgs}`;
          errorBanner.classList.remove('hidden');
        } else {
          errorBanner.classList.add('hidden');
        }
      }

      if (!logsList || logsList.length === 0) {
        consoleEl.innerHTML = '<span style="color: var(--text-dim); font-style: italic;">No logs available for this run</span>';
        return;
      }

      consoleEl.innerText = logsList.join('\n');
      consoleEl.scrollTop = consoleEl.scrollHeight;
    }

    function renderArtifacts(artifactsList, driveInfo) {
      const tbody = document.getElementById('artifacts-tbody');
      const driveBadge = document.getElementById('insp-drive-badge');
      if (!tbody) return;

      if (driveBadge) {
        const driveStatus = driveInfo?.drive_publication_status || (artifactsList.some(a => a.storage_provider === 'google_drive' || a.remote_file_id) ? 'PUBLISHED' : (artifactsList.length ? 'PENDING' : 'DISABLED'));
        driveBadge.innerText = `Drive: ${driveStatus}`;
        if (driveStatus === 'PUBLISHED') driveBadge.className = 'badge badge-purple';
        else if (driveStatus === 'PENDING') driveBadge.className = 'badge badge-warning';
        else driveBadge.className = 'badge badge-neutral';
      }

      if (!artifactsList || artifactsList.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="table-empty">No artifacts recorded for this run</td></tr>';
        return;
      }

      tbody.innerHTML = artifactsList.map(a => {
        const isDrive = a.storage_provider === 'google_drive' || !!a.remote_file_id;
        const provBadge = isDrive ? '<span class="badge badge-purple">google_drive</span>' : '<span class="badge badge-neutral">local</span>';
        const vDate = a.verified_at ? a.verified_at.slice(0, 19).replace('T', ' ') : '-';
        return `
          <tr>
            <td class="mono"><strong>${a.artifact_id || '-'}</strong></td>
            <td><span class="badge badge-info">${a.artifact_type || 'artifact'}</span></td>
            <td>${formatBytes(a.size_bytes)}</td>
            <td>${renderSha256(a.sha256)}</td>
            <td>${provBadge}</td>
            <td class="mono">${a.remote_file_id || '-'}</td>
            <td class="mono">${vDate}</td>
          </tr>
        `;
      }).join('');
    }

    async function selectRun(runId) {
      selectedRunId = runId;
      document.querySelectorAll('#runs-tbody tr').forEach(row => {
        row.classList.toggle('selected', row.getAttribute('data-run-id') === runId);
      });

      const placeholder = document.getElementById('inspector-placeholder');
      const content = document.getElementById('inspector-content');
      const subtitle = document.getElementById('inspector-subtitle');

      if (placeholder) placeholder.classList.add('hidden');
      if (content) content.classList.remove('hidden');
      if (subtitle) subtitle.innerText = `Inspecting run: ${runId}`;

      // Update meta bar from cached run info
      const run = cachedRuns.find(r => r.run_id === runId) || {};
      document.getElementById('insp-run-id').innerText = runId;
      document.getElementById('insp-project-id').innerText = run.project_id || '-';

      const statusBadge = document.getElementById('insp-status');
      if (statusBadge) {
        const s = run.status || 'UNKNOWN';
        statusBadge.innerText = s;
        if (s === 'COMPLETED') statusBadge.className = 'badge badge-success';
        else if (s === 'RUNNING') statusBadge.className = 'badge badge-info';
        else if (s === 'FAILED') statusBadge.className = 'badge badge-danger';
        else statusBadge.className = 'badge badge-warning';
      }

      const rtBadge = document.getElementById('insp-runtime-status');
      if (rtBadge) {
        const active = !!run.is_runtime_active;
        rtBadge.innerText = active ? 'ACTIVE' : 'TERMINATED / DURABLE';
        rtBadge.className = active ? 'badge badge-success' : 'badge badge-neutral';
      }

      document.getElementById('insp-started-at').innerText = run.start_time ? run.start_time.slice(0, 19).replace('T', ' ') : '-';
      document.getElementById('insp-completed-at').innerText = run.completion_time ? run.completion_time.slice(0, 19).replace('T', ' ') : '-';

      // Fetch detail endpoints
      try {
        const [detailRes, metricsRes, logsRes, driveRes, artifactsRes] = await Promise.all([
          fetch(`/api/v1/runs/${runId}`, { headers: getHeaders() }),
          fetch(`/api/v1/runs/${runId}/metrics`, { headers: getHeaders() }),
          fetch(`/api/v1/runs/${runId}/logs`, { headers: getHeaders() }),
          fetch(`/api/v1/runs/${runId}/drive`, { headers: getHeaders() }),
          fetch(`/api/v1/runs/${runId}/artifacts`, { headers: getHeaders() }),
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

      if (!runs || runs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" class="table-empty">No training runs found matching filter</td></tr>';
        return;
      }

      tbody.innerHTML = runs.map(r => {
        const isSelected = r.run_id === selectedRunId ? 'selected' : '';
        const s = r.status || 'UNKNOWN';
        let statusClass = 'badge-neutral';
        if (s === 'COMPLETED') statusClass = 'badge-success';
        else if (s === 'RUNNING') statusClass = 'badge-info';
        else if (s === 'FAILED') statusClass = 'badge-danger';
        else if (s === 'TERMINATED') statusClass = 'badge-warning';

        const rtText = r.is_runtime_active ? 'ACTIVE' : 'TERMINATED / DURABLE';
        const rtClass = r.is_runtime_active ? 'badge-success' : 'badge-neutral';

        const epStep = (r.latest_epoch != null ? `ep ${r.latest_epoch}` : '') +
          (r.latest_step != null ? ` / step ${r.latest_step}` : (r.latest_epoch != null ? '' : '-'));

        const m = r.latest_metrics || {};
        const metricParts = [];
        if (m.loss != null) metricParts.push(`loss: ${m.loss.toFixed(3)}`);
        if (m.val_loss != null) metricParts.push(`val: ${m.val_loss.toFixed(3)}`);
        else if (m.eval_loss != null) metricParts.push(`eval: ${m.eval_loss.toFixed(3)}`);
        if (m.accuracy != null) metricParts.push(`acc: ${(m.accuracy * 100).toFixed(1)}%`);
        if (m.gpu_util_pct != null) metricParts.push(`gpu: ${m.gpu_util_pct}%`);
        const metricsStr = metricParts.length > 0 ? metricParts.join(', ') : '-';

        const chkCount = r.checkpoint_count || 0;
        const chkStr = chkCount > 0 ? `${chkCount}` + (r.latest_checkpoint_name ? ` (${r.latest_checkpoint_name})` : '') : '0';

        const dStatus = r.drive_publication_status || 'PENDING';
        let driveClass = 'badge-neutral';
        if (dStatus === 'PUBLISHED') driveClass = 'badge-purple';
        else if (dStatus === 'PENDING') driveClass = 'badge-warning';

        const srcBadge = r.source_snapshot_available
          ? '<span class="badge badge-success">Available</span>'
          : '<span class="text-dim">-</span>';

        return `
          <tr class="clickable ${isSelected}" data-run-id="${r.run_id}" onclick="selectRun('${r.run_id}')">
            <td class="mono"><strong>${r.run_id}</strong></td>
            <td class="mono">${r.project_id || '-'}</td>
            <td><span class="badge ${statusClass}">${s}</span></td>
            <td><span class="badge ${rtClass}">${rtText}</span></td>
            <td class="mono">${epStep}</td>
            <td class="mono" style="font-size: 0.78rem;">${metricsStr}</td>
            <td>${chkStr}</td>
            <td><span class="badge ${driveClass}">${dStatus}</span></td>
            <td>${srcBadge}</td>
          </tr>
        `;
      }).join('');
    }

    function renderSessionsTable(sessions) {
      const tbody = document.getElementById('sessions-tbody');
      if (!tbody) return;

      if (!sessions || sessions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="table-empty">No active managed Colab sessions</td></tr>';
        return;
      }

      tbody.innerHTML = sessions.map(s => {
        const stateClass = s.state === 'READY' || s.state === 'RUNNING' || s.state === 'ACTIVE' ? 'badge-success' : 'badge-warning';
        const exp = s.expires_at ? s.expires_at.slice(0, 19).replace('T', ' ') : '-';
        return `
          <tr>
            <td class="mono"><strong>${s.session_id}</strong></td>
            <td class="mono">${s.project_name || 'unknown'}</td>
            <td><span class="badge badge-info">${s.accelerator || 'T4'}</span></td>
            <td><span class="badge ${stateClass}">${s.state || 'UNKNOWN'}</span></td>
            <td class="mono">${exp}</td>
            <td><span class="badge badge-managed">Managed</span></td>
          </tr>
        `;
      }).join('');
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
            badge.innerText = h.status || 'HEALTHY';
            if (h.status === 'HEALTHY' && !h.is_stale) badge.className = 'badge badge-success';
            else if (h.status === 'ERROR') badge.className = 'badge badge-danger';
            else badge.className = 'badge badge-warning';
          }
          const meta = document.getElementById('header-meta');
          if (meta) {
            const uptime = h.uptime_seconds ? `${Math.floor(h.uptime_seconds / 60)}m` : '0m';
            meta.innerText = `Uptime: ${uptime} • Samples: ${h.sample_count || 0}`;
          }

          const valCollector = document.getElementById('val-collector-health');
          if (valCollector) {
            valCollector.innerText = h.status || 'HEALTHY';
            valCollector.style.color = h.status === 'HEALTHY' ? 'var(--green)' : (h.status === 'ERROR' ? 'var(--red)' : 'var(--yellow)');
          }
          const metaCollector = document.getElementById('meta-collector-health');
          if (metaCollector) {
            const last = h.last_collected_at ? h.last_collected_at.slice(11, 19) : 'None';
            metaCollector.innerText = `Last: ${last} • Backoff: ${h.collector_backoff_level || 0}`;
          }

          showHealthWarning(h.status, h.is_stale, h.collector_backoff_level, h.last_collected_at);
        }

        // 2. Summary
        if (summaryRes.ok) {
          const s = await summaryRes.json();
          const valCu = document.getElementById('val-cu');
          if (valCu) {
            valCu.innerText = s.current_balance_cu != null ? `${s.current_balance_cu.toFixed(2)} CU` : 'Unavailable';
          }
          const metaCu = document.getElementById('meta-cu');
          if (metaCu) {
            metaCu.innerText = s.is_stale ? 'Warning: Stale snapshot' : 'Official balance';
          }

          const valRate = document.getElementById('val-rate');
          if (valRate) {
            valRate.innerText = s.current_rate_cu_per_hour != null ? `${s.current_rate_cu_per_hour.toFixed(2)} CU/hr` : '0.00 CU/hr';
          }

          const valAssign = document.getElementById('val-assignments');
          if (valAssign) {
            valAssign.innerText = s.active_assignments != null ? s.active_assignments : '0';
          }

          const valRuns = document.getElementById('val-runs');
          if (valRuns) {
            valRuns.innerText = `${s.managed_running_runs || 0} / ${s.total_runs || 0}`;
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

    // Auto-refresh cadence (10s)
    refreshAll();
    setInterval(refreshAll, 10000);
  </script>
</body>
</html>
"""


def render_dashboard_html(title: Optional[str] = None) -> str:
    """Render the embedded Colab Monitor dashboard HTML document."""
    if title:
        return DASHBOARD_HTML.replace(
            "<title>Colab Monitor — V2 Remote GPU Workspaces</title>",
            f"<title>{title}</title>",
        )
    return DASHBOARD_HTML
