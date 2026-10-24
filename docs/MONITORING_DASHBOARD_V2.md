# M8 — Secure Cross-Device Training & Colab Compute Dashboard

> **Implemented & Verified (2026-10-24).** Embedded responsive web monitoring application (`colab_adapter.monitor.ui`) and authenticated loopback REST/SSE backend (`colab_adapter.monitor.server`). Provides cross-device inspection for running and historical experiments, account compute unit burn rates, and durable Google Drive artifacts with strict read-only security.

---

## 1. Product Objective & Overview

From an authenticated browser on any authorized computer, tablet, or phone, researchers can inspect:
- Live research jobs and historical runs across Colab virtual machine lifecycles.
- Training metrics progression (loss, evaluation loss, accuracy, and throughput).
- Account-wide Colab Compute-Unit (CU) balance and consumption burn rate.
- Historical compute snapshots collected since the monitor service started.
- Active Colab session leases registered with the Workspace Provider.
- Durable Google Drive checkpoints, source code snapshots, and SHA-256 verification hashes.
- Sanitized execution logs and runtime failure events.

### Core Architectural Separation
The dashboard maintains a strict separation of concerns:
1. **Account Billing Telemetry**: The official `colab usage` command reports account-wide CU balance, aggregate consumption rate (CU/hr), and active assignment counts. It does *not* provide an itemized per-job billing ledger. Earlier billing history cannot be reconstructed prior to monitor start time.
2. **Experiment Telemetry**: Running training scripts emit structured lifecycle, step, metric, checkpoint, and log events via `TelemetryEmitter`.
3. **GPU Metrics**: GPU VRAM and utilization are optional remote VM instrumentation. They are displayed **only** when explicitly measured and emitted; otherwise, the interface explicitly displays `GPU utilization: not measured`.
4. **Strictly Read-Only**: The monitor cannot allocate new cloud instances, execute remote commands, cancel sessions, or mutate data. Mutating HTTP methods (`POST`, `PUT`, `DELETE`) return `HTTP 405 Method Not Allowed`.

---

## 2. Dashboard UI Architecture & Panels

The frontend is an embedded single-page application (`colab_adapter.monitor.ui.DASHBOARD_HTML`) implemented in pure HTML5, CSS3, and vanilla JavaScript without external runtime dependencies or CDNs.

```text
Browser Client (Mobile / Laptop / Workstation)
   |
   | [Authorization: Bearer <sessionStorage>] (Strictly 0 URL query tokens)
   v
Localhost HTTP Server (127.0.0.1:8080 or TLS Reverse Proxy)
   |
   +--> GET /                             -> Embedded SPA (ui.py)
   +--> GET /api/v1/health                -> Collector & backoff health
   +--> GET /api/v1/summary               -> Account balance & run totals
   +--> GET /api/v1/compute/samples       -> Observed CU time-series
   +--> GET /api/v1/sessions              -> Managed session registry
   +--> GET /api/v1/runs                  -> Active & historical experiments
   +--> GET /api/v1/runs/{run_id}         -> Run metadata & event stream
   +--> GET /api/v1/runs/{run_id}/metrics -> Emitted training metrics
   +--> GET /api/v1/runs/{run_id}/logs    -> Redacted log tail
   +--> GET /api/v1/runs/{run_id}/drive   -> Google Drive publication status
   +--> GET /api/v1/runs/{run_id}/artifacts -> Verified artifact digests
   +--> GET /api/v1/export/compute.csv    -> Authenticated CSV download
```

### UI Panels

#### 1. Header & Authentication Bar
- **Health Badge**: Displays live health state (`HEALTHY`, `DEGRADED`, `STALE`, or `ERROR`).
- **Metadata**: Shows monitor uptime and total collected samples.
- **Auth / Token Management (`#btn-auth`)**: Opens an accessible modal prompt to set or clear the Bearer token in browser `sessionStorage`. When set, the button updates to `Token: Set`.
- **Export Compute CSV (`#btn-export-csv`)**: Fetches `/api/v1/export/compute.csv` with Bearer authentication and triggers an in-memory `Blob` download (`colab_compute_usage.csv`), completely eliminating sensitive token leakage in URL links.
- **Warning & Auth Banners**: Automatically slides down on `401 Unauthorized` or collector degradation.

#### 2. Account Compute-Unit & Burn-Rate Section
Grid of 5 high-contrast KPI cards:
- **Compute Units Balance**: Displays `current_balance_cu` (e.g. `84.50 CU`) or explicit `Unavailable` state when not yet observed.
- **Consumption Rate**: Displays `current_rate_cu_per_hour` (e.g. `1.90 CU/hr`).
- **Active Assignments**: Current number of allocated Colab virtual machines.
- **Managed Runs**: Displays ratio of `managed_running_runs / total_runs`.
- **Collector Health & Freshness**: Real-time status, backoff level, and last sample collection timestamp.

#### 3. Compute Balance History Chart
- Fetches the last 240 samples from `/api/v1/compute/samples?limit=240`.
- Renders an inline, responsive SVG time-series chart with dual polylines:
  - Solid blue line (`#58a6ff`): Observed `balance_cu`.
  - Dashed green line (`#3fb950`): Consumption rate `rate_cu_per_hour`.
- **Zero Fabrication**: If no historical samples exist, displays `"No historical compute samples observed yet"` (never interpolates fictitious past spending).

#### 4. Active Managed Sessions Table
- Lists all active Colab sessions registered with `ManagedSessionRegistry`.
- Columns: `Session ID`, `Project Name`, `Accelerator (GPU)`, `State`, `Lease Expiration`, and `Verification Badge` (`Managed`).
- Displays an empty state banner when no sessions are currently active.

#### 5. Training Runs & Historical Navigation Table
- Live filtering by `Project ID` and experiment `Status` (`RUNNING`, `COMPLETED`, `FAILED`, `TERMINATED`).
- Displays:
  - `Run ID` and `Project ID`
  - `Status` badge
  - `Runtime Status`: Distinguishes `ACTIVE` running sessions from `TERMINATED / DURABLE` historical records.
  - `Epoch / Step`
  - `Latest Metrics`: Extracted `loss`, `val_loss`, `accuracy`, and measured `gpu_util_pct`.
  - `Checkpoints`: Total saved checkpoints and latest checkpoint filename.
  - `Drive Status`: Google Drive publication badge (`PUBLISHED`, `PENDING`, `DISABLED`).
  - `Source Snapshot`: Indicates whether code snapshot archive is available for exact reproducibility.
- Interactive selection: Clicking any row highlights it and opens the **Run Inspector Panel**.

#### 6. Run Inspector Panel
Provides granular inspection for the selected `run_id`:
- **Run Metadata Bar**: Run ID, project, status, start time, completion time, and runtime continuity.
- **Loss & Evaluation Metrics**:
  - Plots `loss` and `val_loss`/`eval_loss` over steps using inline SVG.
  - **Honest GPU Utilization Indicator**: Displays `GPU Utilization: X%` only when explicitly emitted in telemetry; otherwise renders `GPU utilization: not measured`.
- **Execution Logs & Failures Viewer**:
  - Redacted monospace log tail from `/api/v1/runs/{run_id}/logs`.
  - Prominent runtime error callout banner for failure events.
- **Checkpoints & Google Drive Artifacts Table**:
  - Lists all stored artifacts with `artifact_id`, `artifact_type`, human-formatted file sizes (`KiB`, `MiB`, `GiB`), truncated SHA-256 digests with copy-to-clipboard tooltips, `storage_provider` (`google_drive` vs `local`), remote Google Drive file IDs, and verification timestamps.

---

## 3. Honest Telemetry & Accounting Guarantees

The dashboard strictly enforces honest research accounting:
1. **Zero Fabricated Compute Units**:
   - The official Colab usage command only reports point-in-time snapshots.
   - The dashboard never invents pre-monitoring billing records.
   - Increases in balance are classified as potential top-ups or refunds, never negative spending.
   - Any balance drops occurring while external/unmanaged notebooks exist are labeled unattributed.
2. **Zero Fabricated GPU Telemetry**:
   - Remote GPU instrumentation requires explicit opt-in callbacks on active sessions.
   - When no GPU metric is present in emitted events, the UI explicitly displays `GPU utilization: not measured` rather than showing fake zeroes or guessing utilization.
3. **Cross-VM Historical Continuity**:
   - Historical experiments persisted on disk in `PublicationManifest` (`./artifacts/projects/<project_id>/runs/<run_id>/manifest.json`) remain fully inspectable even after the remote Colab VM has terminated.

---

## 4. Security & Hardening Model

The monitor server and web dashboard adhere to defense-in-depth principles:

### Token Authentication via sessionStorage
- The web application reads bearer tokens strictly from `sessionStorage.getItem('colab_monitor_token')`.
- All API requests and file exports send the `Authorization: Bearer <token>` header.
- **Total Elimination of URL Tokens**: Query parameter authentication (`?token=...`) is strictly prohibited and returns `HTTP 401 Unauthorized`. This prevents token leakage in browser history, proxy access logs, referer headers, and screen sharing.

### Network Interface Binding
- **Loopback by Default**: `MonitorServer` binds to `127.0.0.1` by default and rejects external interface bindings.
- **External Binding Safeguards**: Setting `allow_external=True` strictly requires:
  1. A non-empty cryptographic `auth_token` or `trusted_proxy=True`.
  2. Encryption via `require_tls=True` or a trusted reverse proxy identity gateway (`trusted_proxy=True`).

### Strict CORS & Security Headers
- Wildcard CORS (`Access-Control-Allow-Origin: *`) is strictly prohibited.
- Origin validation is enforced against configured `allowed_origins`.
- Defense-in-depth security headers sent on all responses:
  ```http
  X-Content-Type-Options: nosniff
  X-Frame-Options: DENY
  Referrer-Policy: no-referrer
  Content-Security-Policy: default-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'
  ```

---

## 5. Responsive Cross-Device Layout

The dashboard incorporates responsive CSS design targeting workstations, tablets, and smartphones:
- Responsive viewport meta tag: `<meta name="viewport" content="width=device-width, initial-scale=1.0">`.
- Mobile breakpoint `@media (max-width: 768px)`:
  - Header reorganizes into a flexible vertical column with full-width action buttons.
  - KPI grid collapses smoothly to single-column cards.
  - Data tables enable horizontal touch scrolling (`overflow-x: auto; -webkit-overflow-scrolling: touch;`).
  - Inspector panels stack vertically for natural one-handed scrolling.
  - Touch-friendly button hit targets (`min-height: 34px`, accessible padding).

---

## 6. Secure Production Deployment

To safely monitor Colab experiments from remote devices (such as a laptop or smartphone), deploy the dashboard behind an authenticated reverse proxy or private mesh VPN.

### Option A: Private Mesh VPN (Tailscale / WireGuard) — Recommended
1. Keep the monitor bound to `127.0.0.1:8080`.
2. Connect your host machine and mobile device to your private Tailscale or WireGuard network.
3. Use Tailscale Serve to securely proxy loopback traffic:
   ```bash
   tailscale serve https / http://127.0.0.1:8080
   ```
4. Access the dashboard from your authorized device via your private MagicDNS name with automatic TLS.

### Option B: Reverse Proxy with TLS (Caddy)
Caddy automatically provisions Let's Encrypt certificates and enforces HTTPS:

```caddyfile
monitor.research.internal {
    tls internal

    # Optional Basic Auth or Authelia/OIDC authentication
    basicauth {
        researcher $2a$14$...
    }

    reverse_proxy 127.0.0.1:8080 {
        header_up Host {host}
        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-Proto https
    }
}
```

### Option C: Reverse Proxy with Nginx
```nginx
server {
    listen 443 ssl http2;
    server_name monitor.research.internal;

    ssl_certificate /etc/ssl/certs/monitor.crt;
    ssl_certificate_key /etc/ssl/private/monitor.key;
    ssl_protocols TLSv1.2 TLSv1.3;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE stream buffering configuration
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
        proxy_buffering off;
        proxy_cache off;
    }
}
```

---

## 7. Verification & Offline Testing

The dashboard UI and integration test suite run completely offline without cloud credentials or GPU allocation:

```bash
# Run the complete test suite
python3 -m unittest discover tests -v

# Run the dedicated dashboard UI integration test suite
python3 -m unittest tests/test_dashboard_ui_integration.py -v
```

All 9 integration tests verify:
- HTML5/CSS3/SVG structure, responsiveness, and viewport configuration.
- Strict token handling via `sessionStorage` and absence of URL parameter token extraction.
- Presence of all required KPI cards, session tables, training tables, charts, and artifact views.
- Full-stack REST/CSV conformance against `MonitorServer`, `AccountUsageCollector`, `ManagedSessionRegistry`, `TelemetryEmitter`, and `PublicationManifest`.
- Honest reporting contracts (`No historical compute samples observed yet` and `GPU utilization: not measured`).
