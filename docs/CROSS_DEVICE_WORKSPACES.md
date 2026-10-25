# Cross-Device Workspaces & Session Attachment (V2)

## Motivation

Researchers frequently need to monitor remote Colab training runs from multiple devices (e.g. MacBook, Windows workstation, mobile browser) without:
1. Re-authenticating Google Colab CLI on every device.
2. Sharing sensitive OAuth tokens or private SSH keys.
3. Accidentally provisioning expensive cloud GPU virtual machines merely by viewing status.
4. Losing run visibility once the temporary Colab VM terminates.

The Ubuntu-hosted **Workspace Broker** provides authenticated, multi-client read-only attachment to Colab runs while preserving strict isolation boundaries.

## Architecture

```text
  [MacBook Laptop]       [Windows PC]       [Phone / Tablet]
         \                    |                    /
          \                   |                   /
     Authenticated HTTPS / Tailscale / WireGuard / SSH Tunnel
                              |
                     [Ubuntu Host Server]
                              |
               +------------------------------+
               |       Workspace Broker       |
               |  (Token Auth & Audit Log)    |
               +------------------------------+
                     /                 \
                    /                   \
        [Managed Session Registry]   [Durable Artifact Store]
                   |                             |
      [Active Colab GPU Session]    [Local SSD + Google Drive]
        (Approved Runtime)             (Historical Runs)
```

## Core Guarantees

### 1. Strictly Read-Only in Initial Release
All cross-device inspection actions (`list_runs`, `attach_run`, `get_artifacts`, `get_logs`) are strictly read-only.
Any attempt to invoke writable execution via the broker or monitoring dashboard raises a `NotImplementedError` or returns `HTTP 405 Method Not Allowed`.

### 2. Zero Unintended Cloud VM Allocations
Viewing an existing run or attaching to an expired session **never** triggers `colab up` or auto-provisions a cloud VM.
If an inspected session's lease has expired or the remote VM was terminated:
- The broker reports `runtime_status: "RUNTIME_UNAVAILABLE"` and `is_runtime_active: false`.
- All historical logs, metrics, code snapshots, and Google Drive checkpoints remain fully viewable and downloadable.

### 3. Rejection of Unmanaged Sessions
The broker strictly refuses requests targeting unmanaged sessions (sessions created outside the adapter instance):
- Raises `UnmanagedSessionError` with zero outbound commands dispatched.
- Prevents rogue or unmanaged instances from interfering with researcher accounts.

### 4. Multi-Client Concurrency & Audit History
Multiple independent devices can attach to the same running or historical experiment simultaneously:
- Each client connection is verified via constant-time bearer token validation.
- All attachment events are recorded in an in-memory and persistent audit log with `client_id`, `run_id`, timestamp, and access mode.

## API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/v1/runs` | `GET` | List active and historical runs with checkpoint status and Drive links |
| `/api/v1/runs/<run_id>` | `GET` | Detailed telemetry, event count, and runtime status for a run |
| `/api/v1/runs/<run_id>/artifacts` | `GET` | List verified local and Drive-published artifacts with SHA256 digests |
| `/api/v1/runs/<run_id>/drive` | `GET` | Summary of Google Drive publication state, remote file IDs, and errors |
| `/api/v1/runs/<run_id>/manifest` | `GET` | Full reproducibility manifest (git commit, environment, file hashes) |
| `/api/v1/runs/<run_id>/metrics` | `GET` | Time-series loss and accuracy telemetry points |
| `/api/v1/runs/<run_id>/logs` | `GET` | Filtered standard output and training logs |
| `/api/v1/events/stream` | `GET` | Server-Sent Events (SSE) stream supporting cursor resume (`?from_id=...`) |

All API endpoints strictly require authentication via `Authorization: Bearer <token>`, `X-API-Token: <token>`, or a trusted reverse-proxy identity header (`X-Remote-User`). Query parameter authentication (`?token=...`) is strictly rejected to prevent secret leakage in URLs, browser history, reverse-proxy logs, and referrer headers.

## Remote Access & Deployment Security

1. **Loopback Default Binding**: The monitoring server binds to `127.0.0.1` by default.
2. **Real TLS Socket Encryption & Verified Trusted Proxy**: Binding to an external network interface (`allow_external=True`) strictly requires either:
   - Real TLS socket encryption (`require_tls=True`) with valid `tls_certfile` and `tls_keyfile` wrapped via `ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)` (`TLSv1.2`+), or
   - A verified reverse-proxy identity gateway (`trusted_proxy=True`) configured with a non-empty `trusted_proxy_secret`, peer IP validation against `trusted_proxy_ips` (default `127.0.0.1`, `::1`), `X-Remote-User` format validation (`^[a-zA-Z0-9_.@-]{1,128}$`), and constant-time verification of `X-Proxy-Secret` or HMAC-SHA256 `X-Proxy-Signature`.
   Raw unencrypted HTTP on external interfaces or `trusted_proxy=True` without a shared secret is rejected with `ValueError` / `BrokerAuthenticationError`.
3. **Strict CORS Policy**: Wildcard CORS (`Access-Control-Allow-Origin: *`) is completely eliminated. The server restricts access to same-origin requests or explicitly configured allowed origins, sending `Vary: Origin`.
4. **Persistent Credentials**: Broker tokens can be loaded from and persisted to a protected `tokens_file` with `0600` permissions (parent directory `0700`), ensuring authentication configuration survives service restarts.

---

## Single-Host Cross-Process Atomicity & Publication Guarantees

When multiple local processes (e.g., concurrent training jobs, background upload workers, and the Workspace Broker) interact with workspace state on the host:
1. **File Locks & Durable Leases**: Per-artifact OS-level `fcntl.flock` lock files (`.transfer_locks/<id>.lock`) and durable `TransferRecord` leases (`owner_pid`, `lease_token`, `lease_expires_at`, `fencing_epoch`) prevent concurrent workers from interleaving transfers. Dead-PID lease reclamation (`os.kill(pid, 0)`) prevents orphaned locks from dead processes.
2. **Content-Addressed Revision Safety**: Identical content (`sha256` + `size_bytes`) deduplicates across processes to a single Drive upload, while modified content at the same filename uploads a new verified revision without corrupting state or returning stale hashes.
3. **Manifest Locking**: `PublicationManifest` reads and writes are protected by cross-process `.manifest.lock`, ensuring broker clients always observe consistent snapshot state.
4. **Scope Boundary**: Single-host cross-process atomicity is strictly guaranteed via local filesystem `fcntl.flock` and durable journal leases; multi-host distributed writers without a shared lock coordinator are out of scope for local workspace journals.

---

## Dashboard XSS & CSP Security Architecture

For browser clients accessing the embedded monitoring UI through the broker:
- **Zero Inline Event Handlers**: HTML templates contain zero inline handlers (`onclick`, `oninput`, `onchange`, `onerror`).
- **Safe DOM Construction**: Zero unsafe `innerHTML` interpolation for untrusted run/log/artifact/session data; all dynamic DOM nodes are built via `document.createElement` and `.textContent`.
- **Per-Response Cryptographic CSP Nonces**: Enforces `script-src 'self' 'nonce-...'`, `style-src 'self' 'nonce-...'`, `object-src 'none'`, `base-uri 'none'`, and `frame-ancestors 'none'`, with `'unsafe-inline'` completely removed, alongside static `/monitor.js` and `/monitor.css` asset routes.

---

## Verification Scope, PR Stack Status & Operational Rollback

### Three-Tier Verification Scope

1. **Tier 1 — Offline Unit, Integration & Two-Process Tests (Verified in CI & Locally)**:
   - 219 offline tests verifying all broker methods (`test_workspace_broker.py`, `test_v2_security_and_resumable_recovery.py`, `test_adversarial_security_and_fault_injection.py`, `test_project_e2e.py`).
   - Asserts multi-client concurrency (Client A MacBook and Client B Windows attaching to the same run without interference), unmanaged session rejections, trusted-proxy header validation, and token persistence with 0 cloud VM calls.
2. **Tier 2 — Local Headless Browser DOM & XSS Verification (Playwright / Chromium)**:
   - Headless browser verification against live loopback `MonitorServer`, verifying that broker run telemetry and Drive artifact lists render cleanly without script execution from adversarial logs.
3. **Tier 3 — Owner-Gated Live Environment Checks (Pending Final Owner Signoff)**:
   - Live cloud verification requiring explicit human credentials:
     - Real Google OAuth credentials and live Google Drive cloud API.
     - Live Google Colab GPU runtime allocation.
     - Production reverse-proxy deployment with verified TLS and proxy headers.
   - Strictly excluded from automated CI runs pending owner authorization.

### PR Review Stack Status

The WorkspaceBroker and cross-device attachment subsystem are part of a reconciled review stack:
- **PR #3**: Initial Project Workspace Architecture Proposal.
- **PRs #13–#20**: Milestones M1–M8 implementations (including M7 WorkspaceBroker and M8 web interface).
- **PR #22**: Multi-worker integration, security hardening, and documentation reconciliation.

> [!IMPORTANT]
> PRs #3, #13–#20, and #22 form a reconciled review stack and **remain unmerged into `main`** pending final owner approval and completion of Tier 3 live environment checks.

### Operational Rollback Instructions

If a regression or deployment issue occurs:

1. **Stop Active Daemons**:
   ```bash
   kill $(pgrep -f "colab_adapter.broker") 2>/dev/null || true
   kill $(pgrep -f "colab_adapter.monitor") 2>/dev/null || true
   ```
2. **Clear Stale Locks**:
   ```bash
   rm -rf ./artifacts/.transfer_locks/*.lock
   ```
3. **Revert Git Branch**:
   ```bash
   git checkout main
   # Or revert merge commit
   git revert -m 1 <MERGE_COMMIT_SHA>
   ```
4. **Session Protection**:
   Active Colab sessions are never modified or terminated during rollback, as unmanaged session protection strictly isolates external runtimes.
