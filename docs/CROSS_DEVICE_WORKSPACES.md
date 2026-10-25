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

## Cross-Device Artifact Revision Retrieval & PublicationManifest V2

When secondary workstations (e.g. MacBook, Windows PC) attach to an active or historical run via the Workspace Broker or query `/api/v1/runs/<run_id>/artifacts`:

### 1. Canonical Revision Keying
- Every checkpoint revision published to Google Drive is indexed by its immutable canonical ID:
  `<project_id>/<run_id>/<logical_path>@sha256_<full_sha256>`
- When multiple training epochs overwrite the same filename (e.g., `model.pt`), each distinct revision retains its own permanent remote file ID and SHA256 checksum.

### 2. PublicationManifest V2 Schema
- Manifests on disk follow schema version 2 (`manifest_version: 2`).
- The `latest_by_path` index allows secondary devices to resolve the current active checkpoint for any logical path without inspecting every revision.
- `migration_notes` records all automated schema transformations from older V1 structures.

### 3. Retrieving Latest vs. Specific Historical Revisions
- **Latest Checkpoint**: Querying the unversioned logical path (`checkpoints/model.pt`) uses `latest_by_path` to download the newest verified weights (e.g., Checkpoint B).
- **Historical Revision**: Supplying the versioned artifact ID (`checkpoints/model.pt@sha256_<hash>`) downloads the exact historical weights from that earlier epoch (e.g., Checkpoint A), accompanied by complete SHA256 verification.

### 4. Cross-Process Locking & Staging Protection
- Concurrent background upload processes and broker readers synchronize using `.manifest.lock` via `fcntl.flock`.
- Staged uploads use private temporary files (`*.sha256_*`), ensuring secondary clients never download partially committed artifacts.

### 5. Real-Browser CI Security Testing
- Headless browser verification (Playwright/Chromium) in CI tests multi-client attachment, ensuring `#artifacts-table` displays `LATEST` and `HISTORICAL` badges cleanly without inline event handlers or innerHTML vulnerabilities.

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

### Release & Merge Status

The WorkspaceBroker and cross-device attachment subsystem was merged into `main` via **PR #22 (`670c6bc` / `a1629f9846f19c8d5d4e90469ee9e376b64d98e9`)**:
- **PR #3**: Initial Project Workspace Architecture Proposal.
- **PRs #13–#20**: Milestones M1–M8 implementations (including M7 WorkspaceBroker and M8 web interface).
- **PR #22**: Multi-worker integration, security hardening, and documentation reconciliation.

Post-merge hardening under **Issue #23 (`fix/v2-postmerge-hardening-23`)** delivers:
- **P1 Verified Google Drive Recovery**: 3-case retrieval (exact `@sha256_<HASH>`, verified latest pointer, fail-closed on ambiguous fallback).
- **P1 Private Checkpoint Staging**: `0700` staging dirs, `0600` files with `fsync`, symlink/device rejection, and canonical containment checks.
- **P2 Local Artifact Retrieval Size-Mismatch Handling**: Preserving `actual_size` on error before unlinking `.part.` temp file.
- **P2 Controlled Colab HTTP 500 Upload Recovery**: Validating remote parent directories and retrying bounded upload once if newly created.
- **Browser Security Host-State Isolation & 5-Revision Multi-Process History**: Strict CSP nonces and verified multi-process history.
- **Mandatory 21-Stage Offline Pipeline Verification**: Comprehensive test in `tests/test_project_e2e.py`.

> [!NOTE]
> All automated tests run offline (`OFFLINE_RELEASE_READY`). Real cloud operations remain gated on explicit operator approval (`LIVE_VALIDATION_REQUIRES_APPROVAL`).

## 20-Step Operator Guide for Cross-Device Workspaces & Remote Execution

A step-by-step operator guide using real, supported CLI commands (`python3 colab_server.py ...` and `python3 -m colab_adapter.cli ...`):

1. **Host Environment Verification**: Check host Python version and uv toolchain: `python3 --version && uv --version` (no sudo required).
2. **User-Space Upstream CLI Install**: Install official CLI in user space: `bash scripts/install.sh`.
3. **Headless OAuth Authentication**: Complete Google account authentication in terminal: `bash scripts/authenticate.sh`.
4. **Environment & Diagnostic Pre-Flight**: Run zero-allocation health check: `python3 colab_server.py doctor` (or `python3 -m colab_adapter.cli doctor`).
5. **Compute Units & Session Balance Audit**: Inspect active sessions and compute-unit balance: `python3 colab_server.py usage` and `python3 -m colab_adapter.cli sessions`.
6. **Project Workspace Initialization**: Initialize a research workspace: `python3 -m colab_adapter.cli project init ./research-project --name research-project`.
7. **Source & Dependency Preparation**: Populate local project code (`train.py`, `model.py`, `requirements.txt`). Keep secrets in `.env` (excluded by default).
8. **Project Manifest Validation**: Validate schema and security rules: `python3 -m colab_adapter.cli project validate ./research-project --json`.
9. **Zero-Allocation Offline Execution Plan**: Compute file counts and payload bytes: `python3 -m colab_adapter.cli project plan ./research-project --json`.
10. **Human-Approval-Gated Session Allocation**:
    Provision or attach under lock: `python3 -m colab_adapter.cli project up ./research-project --gpu T4 --lease 3600 --approval-token "ALLOCATE T4 $(date +%s)" --json`.
11. **Initial Workspace Incremental Sync**: Transfer sources to the remote runtime: `python3 -m colab_adapter.cli project sync ./research-project --json`.
12. **Remote Training Execution**: Run training with live logs: `python3 -m colab_adapter.cli project exec ./research-project -- --epochs 10 --batch-size 32 --json`.
13. **Incremental Code Hot-Updates**: Update local code and sync only changed files: `python3 -m colab_adapter.cli project sync ./research-project --json`.
14. **Cross-Device Monitoring Server Launch**: Start read-only dashboard: `python3 -m colab_adapter.cli monitor --host 127.0.0.1 --port 8080 --auth-token "<SECRET_TOKEN>"`.
15. **Google Drive Credential Setup (0600 Permissions)**: Configure `~/.config/colab-cli/drive_credentials.json` with permissions `chmod 0600 ~/.config/colab-cli/drive_credentials.json`.
16. **Multi-Revision Checkpoint Publishing**: Checkpoints (`model.pt`) published at steps 100..500 are stored under canonical `@sha256_<HASH>` identities in `PublicationManifest` v2.
17. **Historical Revision Recovery**: Retrieve a specific historical checkpoint (Revision A) or the latest checkpoint (Revision E) by canonical revision ID with SHA256 verification.
18. **Cross-Device Workspace Attachment**: Secondary developer laptops or mobile devices attach to active runs via Workspace Broker with zero additional VM allocations.
19. **Managed Session Teardown**: Stop the active remote session gracefully: `python3 -m colab_adapter.cli project stop ./research-project --json`.
20. **Lease Reconciliation & Rollback**: Clean up expired leases: `python3 -m colab_adapter.cli project reconcile --auto-stop --json`. If network fails, roll back to verified local checkpoints.

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
