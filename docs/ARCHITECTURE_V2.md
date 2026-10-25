# V2 Architecture — Safe Project-Based Remote GPU Workspace

> **Architecture Specification & Verified Implementation.** Status: `READY_FOR_INDEPENDENT_MERGE_REVIEW`. Extends the V1 provider instead of replacing it. Upstream Google Colab VM remains the actual compute host; this is NOT local CUDA forwarding. All human approval gates strictly preserved (`LIVE_VALIDATION_REQUIRES_APPROVAL`).

## 1. System boundary

~~~text
Human / Research agent / IDE (local Ubuntu)
          |
     Project CLI + Manifest Loader
          |
     Safe Plan + SHA256 Workspace Index
          |
   Human-per-job approval / protected dispatcher
          |
    ProjectSessionOrchestrator (NEW)
      |                  |
 Existing ColabGPUProvider  |-- Registry/lock/usage/tracker (REUSE V1)
      |
    TransportAdapter (NEW)
       |-- Official CLI new/exec/upload/download/stop [baseline]
       |-- SSH-over-WebSocket + rsync [optional capability]
       |
 Google-managed Colab VM / GPU (ephemeral)
       |
  /content/<managed-project-root>
       |
 Output manifest -> verified checkpoints -> host artifact directory
~~~

Ownership rule: only the adapter may allocate/terminate its own managed session; project commands must not mutate an unrelated user's session. Cloud VM remains ephemeral; checkpoints must be durable elsewhere.

## 2. Existing modules and proposed additions

| Module | Status | Responsibility |
| --- | --- | --- |
| colab_adapter/provider.py | EXISTING | ColabGPUProvider, JobSpec, lifecycle, approval, registry, file-level transfer, cleanup |
| colab_adapter/tracker.py | EXISTING | ExperimentMetadata / JSONL |
| colab_adapter/cli.py | EXISTING | doctor, usage, sessions, plan, run, stop, history |
| colab_adapter/project_manifest.py | PROPOSED | Schema parser + normalization, immutable canonical ProjectSpec, validation |
| colab_adapter/workspace_index.py | PROPOSED | File discovery, SHA256 manifest, include/exclude, safe diff |
| colab_adapter/transports/base.py | PROPOSED | Typed operations list/stat/hash/upload/exec/download; error classes |
| colab_adapter/transports/official_cli.py | PROPOSED | Approved-session official upload/download/exec fallback |
| colab_adapter/transports/ssh.py | PROPOSED / OPTIONAL | Existing-session SSH/WebSocket proxy + optional rsync; fail closed |
| colab_adapter/workspace.py | PROPOSED | Remote staging, workspace sync and applied-manifest transactions |
| colab_adapter/project_sessions.py | PROPOSED | Project identity, session attach/up/stop, TTL/lease, cleanup |
| colab_adapter/artifacts.py | PROPOSED | Output manifest, verified download, checkpoint contract |
| colab_adapter/cli.py project subcommands | PROPOSED | Human/developer UX, JSON machine output |
| tests/test_project_*.py | PROPOSED | Offline, deterministic, adversarial fixtures only |

**Python boundary:** stdlib-only host tooling should support Python >=3.8, consistent with current wrapper. Official CLI (>=3.12 runtime) stays installed separately via uv. Any new dependency must be optional and justified.

## 3. Data models (proposed; exact names freeze at M1/M2)

### ProjectSpec (schema_version=1)
- project_id: stable locally generated random identifier, not project name alone
- project_root: canonical absolute path, remote_root: fixed prefix + sanitized project_id
- source patterns, excludes, entrypoint, args, dependency manifests
- resource config: GPU, timeouts, retention policy and optional max compute-unit budget (advisory unless independently enforced)
- artifacts: explicitly declared output patterns + local destination
- hash/fingerprint: normalized, deterministic JSON serialization including mapping and policy values

### WorkspaceManifest
- entries[path] = type=file, size, sha256, optional mode; no symlinks unless explicitly supported safely
- generation, project_id, canonical root, source config digest, schema version, integrity digest
- local desired vs remote last-applied manifest; last-applied never trusted blindly; remote path validation required

### ProjectSession
- project_id, session_id (unique random suffix), owner identity/approval reference, pid + optional process-start identity
- created/updated, lease expiry, GPU, remote_root, state and error/cleanup metadata
- transitions validated; recovered only by exact known session ID and registry ownership
- approval scope includes allocation request, GPU, approved session name, maximal timeout/lease and budget

### ProjectRun
- run_id, project_id, revision, source manifest digest, execution plan digest, remote session ID
- input/output file list, redacted logs, state, transfer counts/bytes, observed usage and cleanup result
- success definition: process semantics + checked remote status + all required artifacts verified + safe cleanup or explicitly active approved lease

## 4. End-to-end sequence (bounded one-shot default)

1. Validate manifest and ignore rules. Collect local hashes without Colab network calls.
2. Render a dry-run plan with allocation/retention requirements, files/bytes, dependency changes, destination and cleanup policy.
3. Human independently approves a **specific** potentially billable request; same-UID agent cannot be trusted to authorize itself. Nothing in a plan creates a runtime.
4. Under provider lock, verify approval fingerprints/TTL, check managed sessions and allocate via official CLI. Register session intent **before** remote creation.
5. On an **already authorized managed session**, test transport capability without auto-creating another VM.
6. Push file delta to a staging location under remote_root, verify hashes/counts, then atomically finalize where supported and commit remote manifest only after successful apply. Remote CLI upload fallback may have weaker atomicity: mitigate with staging, checks and explicit contract.
7. Run entrypoint from remote_root with correct working dir, environment and dependency installation. Capture explicit status JSON plus remote logs. Detect tracebacks / missing status.
8. Download expected outputs to local temporary files, verify hashes/size/status and atomically rename into artifacts destination.
9. Record run metadata, usage snapshots, cleanup outcome; stop session by default. If approved interactive lease exists, retain until bounded expiry then operator handles stop/reconcile.
10. On any failure, preserve primary error; attempt cleanup; mark NEEDS_RECONCILIATION if uncertain. Never stop unmanaged names.

## 5. SSH availability is conditional, not an allocation entitlement

Upstream reference: https://github.com/googlecolab/google-colab-cli/blob/main/docs/06_ssh_access.md

- colab ssh --proxy-mode provides an SSH bridge over WebSocket where exposed on the selected Colab runtime. SSH backend can be missing (HTTP 404), unavailable or credential-restricted (401/403), busy (429), or unhealthy (502). Test these offline.
- **CRITICAL:** upstream colab ssh with -s NAME can create a missing session. A read-only sessions precheck followed by ssh -s is **not** a hard no-auto-create guarantee because the session can disappear between commands. Before implementing autonomous ssh, inspect upstream code for an enforceable no-create option or introduce an external reservation/executor boundary with safe fallback. If neither is viable, do not expose SSH transport to untrusted agents or claim no surprise allocation.
- Using SSH identity keys is permitted only through protected host-side files and a documented threat model; never sync private keys or write them to the repository. Do not disable host-key checking by default without a documented trust alternative.
- No runtime connectivity after user revokes allocation; no hidden retry that creates a replacement Colab VM.
- VS Code Remote-SSH and rsync are OPTIONAL convenience paths and do not guarantee runtime permanence. Prefer short research iterations, not always-on hosting.

## 6. Delta-sync algorithm and transaction boundaries

Local planning:
1. Traverse only the canonical root; normalize POSIX relative paths; reject absolute paths, ../, controls, devices, symlinks pointing outside root, case collisions and path traversal.
2. Apply strong deny patterns first (credential/ssh folders, .git, .env*, token files, venv, caches); deny patterns cannot be overridden by normal includes. User explicitly approves any unusual sensitive asset transfer out-of-band.
3. Hash included regular files (streaming chunks), sort by normalized path, build canonical manifest digest.
4. Diff against **last successfully applied** remote manifest: ADD when absent, MODIFY when digest differs, SAME otherwise; proposed deletion requires explicit opt-in. Report bytes to transfer and removed bytes separately; do not overstate actual transfer savings.

Remote apply:
1. Check exact session ownership, approved remote_root and lease; check remote available disk and upload sizes.
2. Transfer ADD/MODIFY to an adapter-private staging directory. Prefer rsync when SSH transport is validated; fallback per-file upload via official CLI.
3. Validate staged file hashes. Move staged files into target safely (per-file replace); commit applied manifest LAST. A per-file replace is **not an atomic whole-workspace swap**; document this distinction and keep recoverable manifests/snapshots.
4. Only delete remote paths from a previously owned manifest and under allowed root, with explicit approved delete mode; fail closed for symlinks/hardlinks/path escapes.
5. On failure, mark SYNC_FAILED, preserve prior last-applied manifest and staging references for reconciliation; never advertise a completed sync.

**Large data:** prefer explicit dataset references/storage strategy and do not download/reupload gigabytes by default; private data transfer requires project-specific approval and retention plan.

## 7. Concurrency, crash recovery and boundaries

- Reuse V1 cross-thread/flock guard and atomic registry. Do not introduce a second unsynchronized allocation state store.
- Unique project/run/session IDs must include random collision-resistant suffixes; never rely exclusively on a UTC-second timestamp + script digest.
- Project-specific lock may permit multiple local plans, but only a single remote apply/exec on a managed session at a time. Cross-project concurrency bounded by account quota/approval.
- Reconciliation is lock-aware and checks ownership, liveness and bounded stale-age; process PID reuse is possible, so future long-running leases should store process start time/heartbeat.
- Cleanup after SIGKILL/crash may require next-run reconciliation/manual action; do not promise hard guarantees in face of network partition.
- Failures in logs, metrics or usage query must not hide primary job failure. Raw credentials must never enter logs or Git.
- Package import/install on server must resolve to the intended merged repository version rather than silently using a stale sibling copy.

## 8. Approval threat model and privacy

Human approval covers the **operation** (GPU, expected resource usage, project root, transfer selection, retention) rather than merely a nonce printed by an agent. Protect Colab CLI credentials in a separate OS identity/service if agents can execute arbitrary shell. Same-user agent-generated ApprovedJob is **not secure approval**.

- Default unattended action set: manifest validate, plan, hash, diff, doctor, non-allocating status (authenticated read-only queries only after permissions).
- Privileged action set: allocate, SSH connect that might auto-allocate, sync to active session, execute, retain, stop (managed IDs only), external storage mounting.
- Least-privilege file traversal, ignore secrets, private permissions, no public tokens, avoid persisting raw CLI stderr/output.
- No noncompliant keepalive, proxy or web service intended to bypass Colab policy. Verify permitted interactive research use.

## 9. Compatibility / rollout

- Phase 1 (offline): manifest schema + diff + mock official transport and SSH feature detection.
- Phase 2 (no-resource): project CLI plan/status fixtures and approved session orchestration mocks.
- Phase 3 (human-opt-in): live capability smoke on a specifically approved session; never run in CI.
- Rollback: disabling V2 project commands leaves V1 provider, registry and credentials intact. Document migration and avoid overwriting existing host installs or GPU drivers.

## 10. Cross-device monitoring architecture (M8)

The monitor is a **separate local Ubuntu process** and never runs inside a temporary Colab runtime. It collects read-only `colab usage` (account-wide CU balance, aggregate CU/h, active assignment count) and `colab sessions` at a bounded cadence, ingests opt-in structured events from owned training jobs, and stores append-only private local JSONL plus query snapshots.

A responsive read-only SPA displays trends, per-project emitted training curves, logs, checkpoint status and session ownership through authenticated REST/SSE. The web server binds to localhost and requires an operator-approved VPN or TLS + authenticated access gateway to be accessed by other computers. No OAuth tokens in browser, no GPU allocation, no SSH bridging, no live session management or approvals exposed over HTTP.

### Dashboard XSS & CSP Security Architecture
To guarantee immunity against Stored Cross-Site Scripting (XSS) from malicious or untrusted experiment logs, run IDs, session names, and artifact payloads:
1. **Zero Inline Event Handlers**: All inline HTML event attributes (`onclick`, `oninput`, `onchange`, `onerror`) are strictly forbidden and eliminated from the DOM.
2. **Safe DOM Construction**: Zero unsafe `innerHTML` interpolation is permitted for untrusted data (run IDs, log output, traceback details, session IDs, error messages, artifact paths). All dynamic DOM elements are constructed programmatically via `document.createElement()` and populated strictly via `.textContent` or `document.createTextNode()`.
3. **Cryptographic CSP Nonces & Header Hardening**:
   - The web server generates a per-response cryptographically secure random nonce (`secrets.token_urlsafe(16)`).
   - Strict Content Security Policy is enforced:
     ```http
     Content-Security-Policy: default-src 'self'; script-src 'self' 'nonce-<RANDOM_NONCE>'; style-src 'self' 'nonce-<RANDOM_NONCE>'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; img-src 'self' data:; connect-src 'self'
     X-Content-Type-Options: nosniff
     X-Frame-Options: DENY
     ```
   - Legacy `'unsafe-inline'` is completely eliminated from modern script and style execution policies.
   - External asset separation: Static `/monitor.js` and `/monitor.css` routes serve decoupled scripts and stylesheets.

**Source caveat:** balance and rate are aggregated account-wide; a new monitor cannot reconstruct older spend or itemize charges per job. Missing training progress or GPU samples display unavailable. See [MONITORING_DASHBOARD_V2.md](MONITORING_DASHBOARD_V2.md) for UI, source contracts, deployment, authorization and testing.

## 11. Open architectural decisions requiring M1 signoff

1. YAML with optional tested parser versus JSON baseline.
2. SSH transport no-auto-create enforceability (block SSH until confirmed; not a convenience assumption).
3. Remote directory atomicity model with CLI fallback.
4. Cloud runtime lease enforcement without keepalive; what is operator-managed vs automatic.
5. External checkpoint backend and privacy constraints (local SSD only by default).
6. Whether any watch/reload UX is worth Colab's ephemeral lifetime and cost.

Every decision must be logged in progress.md and reflected in acceptance tests.

---

## 12. Durable Storage & Cross-Device Broker Architecture (Issue #21)

```text
  [Mac / Windows / Mobile Client]
                 |
      (Authenticated HTTPS / VPN)
                 v
   +---------------------------+
   |     Workspace Broker      |  <-- Token Auth & Audit Logging
   +---------------------------+
          |             |
          |             v
          |      [ManagedSessionRegistry]
          |             |
          |             v
          |      [Colab GPU Runtime] (Ephemeral VM)
          |             |
          |             | (Local download via Official CLI)
          v             v
   +---------------------------+
   |   Local Artifact Store    |  <-- Multi-Tier Verification (SHA256 + PyTorch)
   |       (`./artifacts/`)    |
   +---------------------------+
                 |
                 | (Asynchronous Queue Worker + Transfer Journal)
                 v
   +---------------------------+
   |  Google Drive Store (v3)  |  <-- Resumable Uploads & Range Query Resume
   |    (`Colab-Research/`)    |
   +---------------------------+
```

### Invariants & Boundaries

1. **Storage Decoupling**: Remote Colab VMs never mount Google Drive for continuous training writes. VM writes locally; checkpoints are transferred to the Ubuntu host and verified before Google Drive publication.
2. **Failure Isolation**: A failure in Google Drive upload never marks the GPU training job as failed. Local verified checkpoints remain intact.
3. **No Allocation on Attach**: Viewing active or terminated runs via the Workspace Broker or M8 UI never provokes a new Colab runtime.
4. **Secret Boundary**: Drive tokens are stored in `0600` files on the host and never transmitted to the browser or Colab VM.

### Single-Host Cross-Process Artifact Publication Atomicity Guarantee

Concurrent processes running on the same host (e.g. parallel background upload workers, training jobs, and CLI publication commands) coordinate publication safely without race conditions or torn writes:

1. **OS-Level File Locking (`fcntl.flock`)**:
   - Each artifact publication acquires an advisory exclusive lock on `.transfer_locks/<transfer_id>.lock`.
   - Manifest updates acquire an exclusive lock on `.manifest.lock`.
2. **Durable Leases & Fencing**:
   - `TransferRecord` entries maintain durable lease metadata: `owner_pid`, `lease_token`, `lease_expires_at`, and `fencing_epoch`.
   - Stale leases held by crashed processes are safely reclaimed after checking process liveness via `os.kill(owner_pid, 0)`. If the PID no longer exists, the lease is reclaimed with an incremented fencing epoch.
3. **Content-Addressed Revision Safety**:
   - Identical content (`sha256` digest and `size_bytes`) deduplicates across concurrent processes to a single Google Drive upload.
   - If a file with the same filename is modified with new content, the system detects checksum differences and uploads a new verified remote revision rather than corrupting state or returning stale cached remote IDs.
4. **Scope Boundary**:
   - Single-host cross-process atomicity is strictly guaranteed via local filesystem `fcntl.flock` locks and durable journal leases.
   - Multi-host distributed writers sharing an NFS/SMB share without a distributed lock coordinator (e.g., etcd, ZooKeeper, or Redis) are explicitly out of scope for local workspace journals.

### Checkpoint Revision History & PublicationManifest V2 Schema

To support iterative checkpointing where consecutive training epochs or evaluation steps overwrite the same logical filename (e.g., `checkpoints/model.pt`), Colab V2 enforces content-addressed revision tracking:

1. **Canonical Revision Identity (`@sha256_<full_sha256>`)**:
   - Every artifact revision is assigned a globally unique, immutable identifier:
     `<project_id>/<run_id>/<logical_path>@sha256_<full_sha256>`
     (e.g., `vision_v2/run-01/checkpoints/model.pt@sha256_3b4f5d133bb7c9c693999b2620703353cba10983fc4aa8162ac1e456fdffa766`).
   - The logical path (`checkpoints/model.pt`) and revision ID (`sha256_<12_char_hex>`) are tracked alongside physical storage keys.

2. **PublicationManifest V2 Schema (`manifest_version: 2`) & Immediate-Successor Chain**:
   - `manifest_version`: Integer schema version set to `2`.
   - `artifacts`: Map of canonical versioned IDs to `StoredArtifact` metadata dicts.
   - `latest_by_path`: Dictionary mapping each `logical_path` to its latest active `artifact_id`.
   - `migration_notes`: Chronological audit list recording schema upgrades and index conversions.
   - **Immediate-Successor `superseded_by` Lineage**: Each historical revision tracks `is_latest: false` and `superseded_by: <immediate_successor_artifact_id>`. The latest revision tracks `is_latest: true` and `superseded_by: null`. Lineage forms a strict chain: `A -> B -> C -> D -> E -> None`.
   - **Programmatic Manifest Accessors**: Typed helper methods `get_latest_artifact(logical_path)`, `get_latest_revision_id(logical_path)`, and `get_latest_sha256(logical_path)` provide direct, verified access to the current head revision.

3. **Two-Phase Remote `is_latest` Coordination Protocol (Drive v3 `PATCH`)**:
   - Revisions are uploaded to Google Drive with `is_latest="false"`.
   - Authoritative content verification (byte length and MD5 / SHA256 readback) is performed first.
   - Once verified, prior revisions are demoted to `is_latest="false"` and `superseded_by="<new_id>"` via Drive v3 `PATCH`.
   - The newly verified revision is then promoted to `is_latest="true"` via Drive v3 `PATCH`.
   - Unverified or failed uploads never claim `is_latest="true"`.

4. **Bounded Pagination & Strict Listing Error Classification**:
   - Drive folder queries use `pageSize=100`, bounded `max_list_pages=50`, and `max_list_candidates=5000`.
   - Tracks visited IDs to detect pagination loops and abort safely without memory exhaustion.
   - Strict error classification: listing operations **never return empty `[]` on HTTP or network errors**, raising typed `DriveAuthenticationError` (401/403), `DriveQuotaExceededError` (429/quota), or `DriveTransferError` (5xx/network/JSON parse).

5. **Automatic V1 to V2 Migration**:
   - When loading legacy V1 manifests with unversioned keys (`<project_id>/<run_id>/<path>`), `PublicationManifest.load()` automatically parses existing entries, generates `@sha256_<full_sha256>` keys, populates `latest_by_path`, sets `manifest_version = 2`, and records a migration note without requiring manual data intervention.

6. **Retrieving Latest vs. Specific Historical Revisions & Catastrophic Recovery**:
   - **Latest Revision**: Clients querying by logical path or unversioned name resolve through `latest_by_path`, retrieving the newest verified weights (e.g. Epoch 2 / Checkpoint B).
   - **Historical Revision**: Clients requesting specific historical revisions supply the full canonical versioned ID (`...@sha256_<hash>`), retrieving earlier states (e.g. Epoch 1 / Checkpoint A) with multi-tier SHA256 integrity validation.
   - **Catastrophic Loss Recovery**: If both `manifest.json`, `.bak`, and journal are destroyed, exact historical revisions are recovered via remote Drive SHA256 filtering; unversioned retrieval succeeds if and only if exactly 1 remote revision claims `is_latest="true"`, failing closed with `DriveIntegrityError` if 0 or >1 remote revisions claim `is_latest="true"`.

7. **Cross-Process Staging Protection**:
   - While an upload is in progress, files are staged under isolated temporary filenames (e.g. `model.pt.sha256_<short_sha>`) under `fcntl.flock` protection.
   - In-progress writes never overwrite existing on-disk revisions until remote and local verification succeed.

8. **Reproducible Real-Browser CI Security Testing**:
   - Dashboard UI rendering is validated using headless browser tests (Playwright/Chromium) verifying that both `LATEST` and `HISTORICAL` badges render cleanly in `#artifacts-table`, with zero `.innerHTML` usage, zero inline event handlers, and strict CSP nonce enforcement.

---

## 13. Verification Scope, PR Stack Status & Operational Rollback

### Three-Tier Verification Scope

To ensure release safety without false assumptions of cloud availability, verification is strictly stratified across three distinct tiers:

1. **Tier 1 — Offline Unit, Integration & Fault-Injection Tests (Verified in CI & Locally)**:
   - 219+ offline tests covering all core modules (`test_project_e2e.py`, `test_adversarial_security_and_fault_injection.py`, `test_google_drive_store.py`, `test_workspace_broker.py`, `test_dashboard_ui_integration.py`, etc.).
   - Executes with zero cloud GPU allocations, zero live Drive cloud calls, and deterministic byte-accurate simulated network faults (308 resume drop, session expiry, token refresh, malformed headers).
2. **Tier 2 — Local Headless Browser DOM & XSS Verification (Playwright / Chromium)**:
   - Automated headless browser test suite running against a live loopback `MonitorServer` (`127.0.0.1`).
   - Verifies real DOM rendering, CSS media query responsiveness, `sessionStorage` token isolation, and immunity to script execution from seeded XSS payloads.
3. **Tier 3 — Owner-Gated Live Cloud & Infrastructure Checks (Pending Owner Signoff)**:
   - Live cloud verification requiring explicit human credentials:
     - Real Google OAuth credentials and live Google Drive API v3 uploads.
     - Live Google Colab GPU runtime allocation (billable compute units).
     - Production reverse-proxy deployment (TLS ingress, DNS resolution, and trusted proxy header propagation).
   - Never executed in automated CI or unapproved environments.

### Release & Merge Status

Colab CLI Server V2 was successfully merged into `main` via **PR #22 (`670c6bc` / `a1629f9846f19c8d5d4e90469ee9e376b64d98e9`)**, consolidating the full platform:
- **PR #3**: Initial Project Workspace Architecture Proposal.
- **PRs #13–#20**: Milestones M1 through M8 implementations (manifest loader, sync engine, SSH transport boundary, telemetry emitter, Google Drive store, WorkspaceBroker, and monitoring dashboard).
- **PR #22**: Multi-worker integration, security hardening, and adversarial regression test suite.

Post-merge hardening delivered under **Issue #23 (`fix/v2-postmerge-hardening-23`)** and **Review `5642047783` (F1–F4)**:
1. **F1 Canonical Versioned Revision Identity & Accessors**: Canonical string IDs (`{project_id}/{run_id}/{logical_path}@sha256_{FULL_SHA256}`), immediate-successor `superseded_by` chain (`A -> B -> C -> D -> E -> None`), and typed accessors `get_latest_artifact()`, `get_latest_revision_id()`, `get_latest_sha256()`.
2. **F2 Two-Phase Remote Coordination Protocol**: Drive upload with `is_latest="false"`, content verification, demote prior revisions via Drive v3 `PATCH`, promote new revision to `is_latest="true"` via `PATCH`.
3. **F3 Bounded Pagination & Strict Error Classification**: `pageSize=100`, bounded `max_list_pages=50`, `max_list_candidates=5000`, loop detection via visited ID tracking, and strict error classification (never returning `[]` on HTTP 401/403/429/5xx or network errors).
4. **F4 Catastrophic Loss Recovery Matrix**: Unversioned latest recovery via single verified remote `is_latest="true"` marker; fail-closed `DriveIntegrityError` when 0 or >1 remote revisions claim `is_latest="true"`; fail-closed `DriveTransferError` / `DriveAuthenticationError` / `DriveQuotaExceededError` on remote listing faults.
5. **P1 Private Checkpoint Staging**: `0700` staging directories, `0600` temporary/staged files with `fsync`, symlink/special-file rejection, canonical staging-root containment validation.
6. **P2 Local Artifact Retrieval Size-Mismatch Error Handling**: capturing `actual_size` before unlinking `.part.` temp file so `ArtifactValidationError` is always raised instead of `FileNotFoundError`.
7. **P2 Controlled Colab HTTP 500 Upload Recovery**: validating remote paths, probing whether parent directory is missing via bounded `exec_code`, checking `result["success"]`, and retrying at most once only when missing directory was newly created.
8. **Browser Security Host-State Isolation & 5-Revision Multi-Process History**: isolated browser test state and verified 5-revision (`SHA256 A..E`) multi-process history.
9. **Mandatory Offline E2E Pipeline Verification**: 21-stage offline lifecycle test and 5-revision catastrophic manifest loss scenario (`test_five_revision_history_and_catastrophic_manifest_loss_e2e`) passing in `tests/test_project_e2e.py`.
10. **Review Status**: `READY_FOR_INDEPENDENT_MERGE_REVIEW`. All human approval gates strictly preserved (`LIVE_VALIDATION_REQUIRES_APPROVAL`).

### Operational Rollback Procedures

If an operational anomaly or regression is detected in staging or deployment:

1. **Stop Active Daemons**:
   ```bash
   # Terminate monitoring server and workspace broker instances
   kill $(pgrep -f "colab_adapter.monitor") 2>/dev/null || true
   kill $(pgrep -f "colab_adapter.broker") 2>/dev/null || true
   ```
2. **Clear Stale Lock & Journal State**:
   ```bash
   # Remove temporary lock files and uncommitted journal staging
   rm -rf ./artifacts/.transfer_locks/*.lock
   rm -f ./artifacts/staging/*.tmp.*
   ```
3. **Git Code Rollback**:
   ```bash
   # Revert to stable main branch or pre-V2 release tag
   git checkout main
   # Or revert specific merge commit
   git revert -m 1 <MERGE_COMMIT_SHA>
   ```
4. **Session Protection**:
   Existing unmanaged Colab sessions remain completely unaffected by rollback, as the adapter never terminates sessions outside `ManagedSessionRegistry`.

