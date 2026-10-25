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
- **Execution Logs & Failures Viewer (`#logs-box` / `#log-console`)**:
  - Redacted monospace log tail from `/api/v1/runs/{run_id}/logs`.
  - Prominent runtime error callout banner for failure events.
- **Checkpoints & Google Drive Artifacts Table (`#artifacts-table`)**:
  - Displays every published checkpoint and artifact revision with clear distinction between latest and historical states.
  - Columns:
    1. **Artifact**: Logical path or filename (`a.logical_path || a.rel_path || a.name`) accompanied by an explicit status badge: `LATEST` (green badge, when `is_latest: true`) or `HISTORICAL` (neutral badge, when `is_latest: false`).
    2. **Revision ID**: Canonical revision hash (`sha256_<12_char_hex>`).
    3. **Step**: Training step emitted with the checkpoint, or explicit `unavailable` (never guessed or fabricated).
    4. **Category**: Artifact category (`checkpoint`, `source_snapshot`, `output`).
    5. **Size**: Human-formatted byte count (`KiB`, `MiB`, `GiB`).
    6. **Integrity & Remote Status**: Verification indicator (`VERIFIED` / `DRIVE` / `LOCAL`) and truncated SHA-256 digest with copy-to-clipboard tooltip.
    7. **Timestamp**: ISO UTC verification or creation timestamp.
  - Supports `PublicationManifest` V2 schema (`manifest_version: 2`, `latest_by_path`, and `migration_notes`) with automatic migration from V1.
  - Backed by `/api/v1/runs/{run_id}/artifacts` and `/api/v1/runs/{run_id}/drive`.
  - Constructed strictly with safe DOM APIs (`document.createElement()`, `.textContent`, `.replaceChildren()`) with zero `.innerHTML` and zero inline event handlers. Verified by real-browser headless Chromium tests in CI.

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
  Content-Security-Policy: default-src 'self'; script-src 'self' 'nonce-<NONCE>'; style-src 'self' 'nonce-<NONCE>'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; img-src 'self' data:; connect-src 'self'
  ```

### Dashboard XSS & CSP Security Architecture
The monitoring UI renders potentially untrusted strings generated during remote training runs (arbitrary experiment names, log lines, traceback details, error messages, and artifact filenames). To guarantee complete immunity against Stored Cross-Site Scripting (XSS):
1. **Zero Inline Event Handlers**:
   - Zero inline event handler attributes (`onclick`, `oninput`, `onchange`, `onsubmit`, `onerror`, `onload`) exist anywhere in the served HTML or dynamic templates.
   - All user interactions and event dispatchers are registered unobtrusively in JavaScript via `.addEventListener()`.
2. **Safe Dynamic DOM Node Construction**:
   - Zero unsafe `innerHTML` string interpolation is used for untrusted run/log/artifact/session data.
   - Dynamic table rows, log consoles, KPI value blocks, and error callouts are constructed exclusively via `document.createElement()` and populated using `.textContent` or `document.createTextNode()`.
3. **Per-Response Cryptographic CSP Nonces**:
   - Every response dynamically generates a high-entropy CSP nonce (`secrets.token_urlsafe(16)`).
   - Scripts and styles are permitted only if tagged with the matching cryptographic nonce (`<script nonce="...">`, `<style nonce="...">`).
   - The permissive `'unsafe-inline'` keyword is completely removed from both `script-src` and `style-src` directives.
   - Restrictive directives enforce strict boundaries: `object-src 'none'`, `base-uri 'none'`, and `frame-ancestors 'none'`.
   - Asset routes `/monitor.js` and `/monitor.css` serve external static JavaScript and CSS resources.

### Cross-Process Artifact Publication Atomicity Context
When researchers inspect artifact publication progress via the dashboard (`/api/v1/runs/<run_id>/drive` or `/api/v1/runs/<run_id>/artifacts`):
- The dashboard queries `PublicationManifest` and `TransferJournal` protected by OS-level `fcntl.flock` locks.
- Status displays accurately reflect the single-host cross-process guarantees: per-artifact locks (`.transfer_locks/<id>.lock`), durable `TransferRecord` leases (`owner_pid`, `lease_token`, `lease_expires_at`, `fencing_epoch`), and dead-PID reclamation.
- Artifact revision safety is guaranteed: identical files deduplicate across processes to a single Drive object, while modified content at the same filename publishes a new verified remote revision rather than displaying corrupt or stale SHA-256 hashes.

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

## 7. Verification Scope, PR Stack Status & Operational Rollback

### Three-Tier Verification Scope

To validate dashboard functionality and security without premature assumptions of live cloud connectivity, verification is partitioned into three distinct tiers:

1. **Tier 1 — Offline Unit, Integration & Adversarial Fault-Injection Suite (Verified in CI & Locally)**:
   - 219 offline tests passing across the test suite (`test_dashboard_ui_integration.py`, `test_monitor_backend.py`, `test_adversarial_security_and_fault_injection.py`, `test_project_e2e.py`).
   - Validates HTML5/CSS3/SVG structure, viewport responsiveness, strict `sessionStorage` token management, total rejection of URL parameter tokens (`?token=...`), authenticated REST/SSE endpoints, and error-handling paths without live cloud VM or Google Drive credentials.
2. **Tier 2 — Local Headless Browser DOM & XSS Verification (Playwright / Chromium)**:
   - Executes automated browser tests using headless Chromium against a live loopback `MonitorServer` (`http://127.0.0.1:<PORT>`).
   - Injects adversarial script payloads (`<script>alert(1)</script>`, `<img src=x onerror=...>`) into telemetry event streams and confirms zero script execution via strict CSP nonces and `.textContent` DOM population.
   - Asserts visual rendering of KPI cards, SVG charts, mobile responsiveness breakpoints, and modal dialogs.
3. **Tier 3 — Owner-Gated Live Environment Checks (Pending Final Owner Signoff)**:
   - Verification requiring privileged external resources:
     - Real Google OAuth credentials and live Google Drive API uploads.
     - Actual Google Colab GPU runtime allocation consuming compute units.
     - Production reverse-proxy deployment with valid external TLS certificates and trusted-proxy identity injection.
   - Staged exclusively for production deployment after explicit owner authorization; never invoked during offline CI runs.

### Release & Merge Status

The monitoring dashboard and associated V2 platform capabilities were merged into `main` via **PR #22 (`670c6bc` / `a1629f9846f19c8d5d4e90469ee9e376b64d98e9`)**:
- **PR #3**: Initial Project Workspace Architecture Proposal.
- **PRs #13–#20**: Milestone M1–M8 implementations (including M8 dashboard backend and UI components).
- **PR #22**: Multi-worker integration, comprehensive test expansion, and documentation reconciliation.

Post-merge hardening delivered under **Issue #23 (`fix/v2-postmerge-hardening-23`)** includes:
- **P1 Verified Google Drive Recovery**: 3-case retrieval (exact `@sha256_<HASH>`, verified latest pointer, fail-closed on ambiguous fallback).
- **P1 Private Checkpoint Staging**: `0700` staging dirs, `0600` files with `fsync`, symlink/device rejection, and canonical containment checks.
- **P2 Local Artifact Retrieval Size-Mismatch Handling**: Preserving `actual_size` on error before unlinking `.part.` temp file.
- **P2 Controlled Colab HTTP 500 Upload Recovery**: Validating remote parent directories and retrying bounded upload once if newly created.
- **Browser Security Host-State Isolation & 5-Revision Multi-Process History**: Strict CSP nonces and verified multi-process history.
- **Mandatory 21-Stage Offline Pipeline Verification**: Comprehensive test in `tests/test_project_e2e.py`.

> [!NOTE]
> All automated tests execute in offline/mock mode (`OFFLINE_RELEASE_READY`). Real cloud operations remain gated on explicit operator approval (`LIVE_VALIDATION_REQUIRES_APPROVAL`).

### Operational Rollback Instructions

In the event of an operational issue or security regression in production or staging:

1. **Terminate Active Daemon Processes**:
   ```bash
   # Stop the monitoring server daemon
   kill $(pgrep -f "colab_adapter.monitor") 2>/dev/null || true
   # Stop the workspace broker daemon
   kill $(pgrep -f "colab_adapter.broker") 2>/dev/null || true
   ```
2. **Clean Stale Transfer Locks**:
   ```bash
   # Remove persistent OS-level lock files
   rm -rf ./artifacts/.transfer_locks/*.lock
   ```
3. **Revert Git Working Tree**:
   ```bash
   # Revert to stable main branch or pre-V2 release tag
   git checkout main
   # Or create a rollback commit reverting the PR merge
   git revert -m 1 <MERGE_COMMIT_SHA>
   ```
4. **Session Isolation**:
   Because `MonitorServer` is strictly read-only and unmanaged sessions are safeguarded by `ManagedSessionRegistry`, stopping the dashboard has zero impact on active Colab runtimes.

## 8. Project-Scoped Monitoring Identity (PR #24, offline)

Implemented in `colab_adapter/monitor/server.py` and `colab_adapter/telemetry.py` on branch `fix/v2-postmerge-hardening-23` (PR #24). Offline tests only; the dashboard has not been validated against a live Colab account.

**Run identity.** A run is identified by a RunKey `(project_id, run_id)`. The key comes from telemetry events (`project_id`, `run_id`) and from the on-disk layout `projects/<project>/runs/<run>`. A bare `run_id` is never treated as unique.

**Selectors**

- `?project=<id>` narrows `/api/v1/summary`, `/api/v1/runs`, `/api/v1/runs/<run_id>` and `/api/v1/events/stream`. An invalid selector returns HTTP 400.
- Without `?project=`, `/api/v1/runs/<run_id>` resolves the run only if exactly one project owns that `run_id`. If several projects own it, the server returns HTTP 409 with `error: "ambiguous_run"` and the list of candidate projects. If none owns it, it returns HTTP 404.
- Legacy telemetry events with no `project_id` never make a named-project run ambiguous. They resolve only when no named project owns the run.

**Telemetry filtering.** `TelemetryEmitter.get_events(project_id=...)` matches the project exactly. Records without a `project_id` never match a named project. They appear only when no project filter is given.

**Filesystem safety.** The run scanner skips symlinked project directories, runs directories and run directories, so one project cannot alias another. It skips symlinked `manifest.json` files and symlinked artifact files. It excludes the internal directories `.revisions`, `.upload_staging`, `.staged_uploads`, `.transfer_locks` and `.locks` from artifact listings.

**Artifact ids.** Artifact ids shown by the monitor are prefixed `<project>/<run>/` for every run that has a project (legacy runs with no project key get `<run>/...` only), so the same logical path in two projects is two distinct entries. Files under a run directory that match a manifest `logical_path` (the mutable latest alias) are never listed a second time.

**Account-wide versus per-run counters**

- Balance, consumption rate, active assignments and managed running counts are account-wide. The summary labels them with `managed_counts_scope: "account"`, and it echoes the applied filter in `project_filter`.
- Only the run counts are narrowed by `?project=`. With a filter, `total_runs` is the number of unique RunKeys in that project. Without a filter it is the larger of the unique RunKey count and the managed-session registry count.
- Managed sessions carry no project id, so per-project billing is not derived.

**`is_runtime_active`** is conservative. It is `true` only when the managed session with that id is in `READY`, `RUNNING` or `ACTIVE` state and exactly one RunKey carries that `run_id`. Otherwise it is `false`, even if a session is running.

**Hashing cost.** For unmanifested local files the monitor computes SHA-256 per request, up to 8 MiB (`_LOCAL_HASH_MAX_BYTES`). Larger files are listed with their size only and no hash.

**Known limits**

- Hashing is repeated on every request, up to 8 MiB per file. This is a known cost, not a cache.
- The SSE stream (`/api/v1/events/stream`) is unfiltered when `?project=` is omitted, so it carries events from every project. Clients that need isolation must pass `?project=`.
- No live Colab or Drive data has been validated through this path.


### Latest-revision state is never guessed (PR #24 follow-up)

`is_latest` is derived only from the manifest's `latest_by_path`. When the head is missing, names no listed revision, or names more than one, every revision of that path gets `is_latest: null` and `latest_state: "unknown"` (`"known"` when a head matched); the dashboard shows `UNKNOWN`. The run list sets `latest_checkpoint_name`/`latest_checkpoint_sha256` only from a revision whose latest flag is true, otherwise `null` with `latest_checkpoint_state: "unknown"`. Remote `is_latest` flags carried inside manifest entries are not authoritative for display. Run counts (`/runs`, `/summary`) are computed after legacy keyless phantom entries are dropped, so they agree for event-backed runs; runs found only on disk (no telemetry) are listed by `/runs` but not counted in `/summary.total_runs`.
