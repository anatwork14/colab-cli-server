# Colab CLI Server — Progress Ledger

> Last plan update: 2026-10-24. Scope: V2 VKong-inspired project/workspace UX **on top of** the existing official Google Colab CLI. This document distinguishes evidence from plans. Update it in the same PR as every substantive implementation change.

## Project status

- **V1: MERGED / verified on GitHub.** PR #2 squash-merged into main at commit 2bfd2194ad8259d785ca59935579eab6bdad052c. Human operator reported server synchronization and 43/43 offline tests on Python 3.8 and 3.12, OAuth diagnostics and no active sessions. The maintainer has not independently inspected the user's running server in this planning PR.
- **V1 live evidence:** earlier explicitly approved Colab T4 one-shot and stateful E2E executions proved basic remote CUDA execution, input upload, artifact download, checksum/ZIP checks and teardown. These tests predate V2 and do **not** validate SSH, directory sync or new development workflows.
- **Dashboard M8: COMPLETE (Offline & local headless browser verified).** Full-stack monitor backend (`colab_adapter/monitor/server.py`, `collector.py`, `schemas.py`) and responsive embedded SPA (`colab_adapter/monitor/ui.py`) are implemented and verified offline and in headless Chromium (`playwright-core`) with strict Bearer/trusted-proxy authentication, per-response cryptographic CSP nonces (`script-src 'self' 'nonce-...'` without `'unsafe-inline'`), zero unsafe `innerHTML` or inline event handlers, TLS enforcement, and deep secret redaction.
- **V2 implementation: COMPLETE across M1–M8 and V2-Drive (`feat/v2-drive-workspaces`).** All V2 milestones (M1–M8) and Cross-Device Workspaces & Google Drive Durable Storage (Issue #21, PR #22) are implemented and verified (**240/240 tests passing on Python 3.12; 239/240 passed + 1 Python>=3.10 AST check skipped on Python 3.8**). Note: PRs #3, #13–#20, and #22 form a reconciled review stack on GitHub and remain **unmerged into `main`** pending owner review. **No live V2 SSH test, GPU rental, or live Google Drive API call has been executed.**
- **Account safety:** NO cloud allocations, SSH auto-provision, Drive mount, paid API, GPU spend or unattended deployment authorized or executed during V2 development and testing.

## Baseline inventory (evidence from main)

| Area | Baseline | V2 gap | State |
| --- | --- | --- | --- |
| Upstream CLI | Official google-colab-cli; reported v0.7.5 on host | Capability probe and graceful version/feature detection | DONE V1 / DONE V2 |
| Provider API | ColabGPUProvider, JobSpec, approval, usage, tracking | Project-aware manifest and reusable workspace operations | DONE V1 / DONE V2 |
| Compute lifecycle | One-shot named run + stateful new/install/upload/exec/download/stop | Explicit attach/resume/reuse/TTL, no unexpected allocation | DONE V1 / DONE V2 |
| Session registry | ManagedSessionRegistry, atomic backups, locks, reconcile | Project/session mapping, leases, resource policy and idempotency | DONE V1 / DONE V2 |
| File transfer | Explicit input_files uploads and explicit expected_outputs downloads | Selective incremental directory sync, dry-run diff and safe deletion | DONE V1 / DONE V2 |
| SSH transport | Upstream documents ssh --proxy-mode | Adapter-managed opt-in capability probe and transport fallback | UPSTREAM ONLY / DONE V2 (Guarded) |
| Checkpoints | Local download, hash/structure/safe inference tiers | Durable remote-to-local artifacts, resume plan and metadata | DONE V1 / DONE V2 |
| Agent safety | Interactive user prompt, non-forgeable boundary expressly absent | Protected approval executor design for autonomous usage | DOC BOUNDARY / DOC & ENFORCED IN CLI |
| Observability | JSONL experiment history, CLI doctor/usage | Structured project events, sync statistics, resource accounting | DONE V1 / DONE V2 |

## Workboard

| ID | Deliverable | Status | Evidence / next action |
| --- | --- | --- | --- |
| M0 | V1 baseline accepted | COMPLETE | PR #2, main commit above |
| M1 | Feature compatibility and security contracts | COMPLETE | Issue #5, docs/UPSTREAM_COMPATIBILITY_V2.md, colab_adapter/errors.py, colab_adapter/capability.py; 55/55 tests pass |
| M2 | Project manifest, validation and safe plan | COMPLETE | Issue #6, colab_adapter/project_manifest.py, tests/test_project_manifest.py; 63/63 tests pass |
| M3 | SSH transport and no-surprise-allocation boundary | COMPLETE | Issue #7, colab_adapter/transports/, tests/test_transports.py; 70/70 tests pass |
| M4 | Content-addressed workspace sync | COMPLETE | Issue #8, colab_adapter/workspace.py, tests/test_sync.py; 74/74 tests pass |
| M5 | Attach/update/execute/stop lifecycle | COMPLETE | Issue #9, colab_adapter/project_sessions.py, tests/test_project_sessions.py; 85/85 tests pass |
| M6 | Artifact/cache/resume and observability | COMPLETE | Issue #10, colab_adapter/artifacts.py, colab_adapter/telemetry.py; 95/95 tests pass |
| M7 | Integration, end-to-end signoff and docs | COMPLETE | Issue #11, colab_adapter/cli.py, tests/test_project_e2e.py; 111/111 tests pass |
| M8 | Secure browser training & compute monitor | COMPLETE | Issue #12, colab_adapter/monitor/, tests/test_monitor_backend.py, tests/test_dashboard_ui_integration.py, tests/test_browser_xss_and_cross_process_qa.py; 240/240 suite pass |
| V2-Drive | Cross-Device Workspaces & Google Drive Artifacts | COMPLETE | Issue #21, PR #22, colab_adapter/storage/, broker, reproducibility; 240/240 tests pass on Python 3.12 (239/240 + 1 skip on Python 3.8) |
| M9 | Cross-provider abstraction (optional) | DEFERRED | Only after Colab V2 is independently usable |


## How an implementer updates this file

For every PR, record: (a) date and branch/commit, (b) tasks finished vs partial, (c) tests run with exact commands and output counts, (d) actual live-VM usage **or explicit none**, (e) observed results vs target values, (f) blockers/risks, (g) next owner. Never mark a milestone COMPLETE merely because the code compiles. Keep this status table consistent with milestones.md.

### Change log
- **2026-10-25 — Review `5640707483` Resolution: Stored DOM XSS Elimination, Strict Nonce CSP, Cross-Process Publication Atomicity & Real Browser/Multi-Process QA (branch `feat/v2-drive-workspaces`):**
  - **P0 Stored DOM XSS Elimination & Strict Cryptographic Nonce CSP (`colab_adapter/monitor/ui.py`, `colab_adapter/monitor/server.py`)**:
    - Removed 100% of `.innerHTML` assignments and 100% of inline event handler attributes (`onclick=`, `oninput=`, `onchange=`, `onerror=`, `onload=`) across `colab_adapter/monitor/ui.py` and `colab_adapter/monitor/server.py`.
    - Refactored `renderRunsTable`, `renderArtifacts`, `renderSessionsTable`, `renderLogs`, `showAuthBanner`, `showHealthWarning`, `createSha256Badge`, `renderComputeChart`, and `renderMetricsChart` to construct DOM and SVG nodes exclusively via `document.createElement`, `document.createElementNS`, `.textContent`, `.replaceChildren()`, and `.addEventListener()`.
    - Hardened `selectRun(runId)` with `encodeURIComponent(String(runId))` on all run detail endpoints and added `formatNumber(val, decimals)` (`Number.isFinite`) validation before all `.toFixed()` calls.
    - Upgraded `MonitorServer` to generate a fresh cryptographic nonce (`secrets.token_urlsafe(18)`) per HTML response and emit `Content-Security-Policy: default-src 'self'; script-src 'self' 'nonce-<NONCE>'; style-src 'self' 'nonce-<NONCE>'; img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'`, completely removing `'unsafe-inline'`. Added static `/monitor.js` and `/monitor.css` asset routes.
  - **P1 Cross-Process Artifact Publication Atomicity & Durable Lease Fencing (`colab_adapter/storage/transfer.py`, `colab_adapter/storage/worker.py`, `colab_adapter/storage/google_drive.py`)**:
    - Extended `TransferRecord` with `owner_pid`, `lease_token`, `lease_expires_at`, and `fencing_epoch`.
    - Implemented re-entrant per-thread and cross-process exclusive `fcntl.flock` locks (`TransferJournal.acquire_named_lock` and `acquire_artifact_lock` on `.transfer_locks/<safe_id>.lock` with `0700` dir and `0600` file permissions) plus durable lease claiming (`claim_transfer_lease` / `release_transfer_lease`) with automatic dead-PID lease reclamation via `_is_pid_alive(pid)` (`os.kill(pid, 0)`).
    - Wrapped `GoogleDriveArtifactStore.publish()` and `PublicationManifest` updates in cross-process locks (`artifact-<id>` and `manifest-<project>-<run>`), performing post-lock journal reload and remote folder deduplication so concurrent processes publishing the same artifact execute at most 1 upload session and receive the same verified `StoredArtifact`, while modified content at the same filename publishes a distinct content-addressed revision (`<base_id>-<sha256[:12]>`).
  - **Real Headless Browser DOM XSS & Two-OS-Process Concurrency QA (`tests/test_browser_xss_and_cross_process_qa.py`)**:
    - Verified in real headless Chromium (`Google Chrome for Testing 145.0.7632.6` + `playwright-core`) against a live loopback `MonitorServer` seeded with `<img src=x onerror=...>`, `</td><script>...`, `');window.__xss=true;//`, and `"><svg onload=...>` across runs, sessions, logs, errors, and Drive artifacts: `window.__xss === undefined`, 0 page errors, 0 CSP violations during normal operation, active CSP blocking of un-nonced inline `<script>` injection, and safe plain-text rendering.
    - Verified with two simultaneous OS processes (`subprocess.Popen`) synchronized via filesystem barrier against a loopback HTTP Drive v3 server (`LoopbackDriveServer`): `session_counter == 1`, `len(committed_checkpoints) == 1`, identical `remote_file_id` and `sha256`, and mid-upload process crash (`os._exit(99)` at 256 KiB) followed by second-process HTTP 308 resume from byte offset `262144`.
    - Full test suite: **240/240 tests passing on Python 3.12** and **239/240 passed + 1 skipped on Python 3.8** (0 failures, 0 errors).
    - **ACTUAL LIVE VM USAGE: NONE** (0 compute units consumed, 0 remote VMs allocated, 0 live Drive API calls made).
    - **PR STACK STATUS**: PRs #3, #13–#20, and #22 remain a reconciled review stack on GitHub and are **not yet merged into `main`**.
    - **STATUS: READY_FOR_INDEPENDENT_REVIEW.**
- **2026-10-25 — V2 Complete Integration, Full-Stack UI, Drive Protocol Hardening & Adversarial QA (branch `feat/v2-drive-workspaces`):**
  - **P0 Trusted-Proxy Authentication & Real TLS Socket Enforcement (`colab_adapter/monitor/server.py`, `colab_adapter/broker.py`)**:
    - Resolved trusted-proxy header spoofing: enabling `trusted_proxy=True` now strictly requires a non-empty `trusted_proxy_secret` (raising `ValueError` / `BrokerAuthenticationError` if omitted), validates client peer IP against `trusted_proxy_ips` (default loopback `127.0.0.1`, `::1`), validates `X-Remote-User` / `X-Forwarded-User` against `^[a-zA-Z0-9_.@-]{1,128}$`, and verifies `X-Proxy-Secret` via `hmac.compare_digest` or `X-Proxy-Signature` via HMAC-SHA256.
    - Enforced real TLS socket encryption: when `require_tls=True` (or `allow_external=True` on a non-loopback host without `trusted_proxy`), `MonitorServer` validates `tls_certfile` and `tls_keyfile` and wraps `self.socket` using `ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)` with `TLSVersion.TLSv1_2` minimum.
  - **P1 Byte-Accurate Google Drive Resumable Upload & Authoritative Verification (`colab_adapter/storage/google_drive.py`)**:
    - Eliminated unacknowledged byte skipping when HTTP 308 omits `Range`: strictly validates `Range` header via `^bytes=0-(\d+)$` and boundary checks (`0 < confirmed_next <= start_byte + chunk_len <= total_size`). When `Range` is absent on 308, reconciles progress via `_query_upload_progress` and retries unacknowledged bytes instead of advancing offset.
    - Hardened `_query_upload_progress` to raise `DriveAuthenticationError` on 401 (after optional token refresh), `DriveQuotaExceededError` on 403 quota errors, `DriveTransferError` on 429/5xx after bounded exponential backoff, and return `-1` sentinel on 404/410 session expiry.
    - Prevented same-name/same-size file collisions by matching both `expected_size` and `expected_md5` in `_find_file_in_folder` and resetting stale journal entries when local file size or SHA256 changes.
    - Added `alt=media` streaming content readback verification fallback in `publish()` when Drive metadata omits `md5Checksum`, raising `DriveIntegrityError` if readback fails or SHA256/size mismatches.
    - Added `_sanitize_drive_error` to scrub Bearer tokens, resumable session URLs (`session_id=`, `upload_id=`), and OAuth secrets across all Drive exceptions and journal error records.
  - **P1 Storage Reliability, Worker Deduplication & Secret Redaction (`colab_adapter/storage/transfer.py`, `worker.py`, `tracker.py`, `collector.py`, `artifacts.py`)**:
    - Updated `TransferRecord.to_dict(safe=True)` to deeply scrub `last_error` (`sanitize_text`) and `metadata` (`sanitize_data`) in addition to redacting `resumable_session_url`.
    - Added in-flight/queued deduplication in `ArtifactUploadWorker.enqueue()` and pre-upload re-verification when local checkpoints are overwritten with new weights.
    - Applied `sanitize_text` / `sanitize_data` across `AccountUsageCollector`, `ArtifactManager`, and `TelemetryEmitter`.
  - **Full-Stack Monitoring Dashboard UI (`colab_adapter/monitor/ui.py`)**:
    - Implemented complete responsive SPA dashboard in `colab_adapter/monitor/ui.py` with `sessionStorage` Bearer auth, Blob CSV export, Compute Units Balance & History SVG chart, Active Managed Sessions table, filterable Training Runs table, and Run Inspector (SVG Loss/Eval metrics chart, honest GPU utilization indicator, log viewer, and SHA256 Google Drive Artifacts table).
  - **M1–M8 Platform Integration & 13-Stage End-to-End Pipeline (`colab_adapter/capability.py`, `tests/test_project_e2e.py`, `tests/test_adversarial_security_and_fault_injection.py`)**:
    - Enforced `CliCapabilities` tuple immutability, `SshSafetyGuard.prepare_isolated_ssh_config` (`0700` dir, `0600` `sessions.json`), and pre-invocation TOCTOU verification raising `SshAutoAllocationRiskError`.
    - Added 13-stage end-to-end integration test (`test_complete_v2_pipeline_with_drive_and_cross_device_monitor`) and 20-test adversarial security & byte-accurate fault-injection suite (`tests/test_adversarial_security_and_fault_injection.py`).
    - Full offline test suite: **218/218 tests passing on Python 3.12** (0 failures, 0 errors, 0 skipped) and **217/218 passed + 1 skipped on Python 3.8** (0 failures, 0 errors).
    - **ACTUAL LIVE VM USAGE: NONE** (0 compute units consumed, 0 remote VMs allocated, 0 live Drive API calls made).
    - **STATUS: READY_FOR_INDEPENDENT_REVIEW.**
- **2026-10-24 — V2 Security Hardening & Resumable Recovery COMPLETE (branch `feat/v2-drive-workspaces`):**
  - **P0 Authentication & Remote Access Security**:
    - Completely removed browser authentication through `?token=` query parameters across all endpoints (`/api/v1/summary`, `/api/v1/runs`, `/api/v1/events/stream`, `/api/v1/export/compute.csv`).
    - Updated embedded dashboard UI to manage bearer token via `sessionStorage` (cleared on tab close, never leaked into URLs, server logs, or referrer headers).
    - Removed `Access-Control-Allow-Origin: *`. Enforced strict same-origin by default, optional explicit origin whitelist, and defense-in-depth security headers (`Content-Security-Policy`, `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`).
    - Enforced secure remote binding: external binding (`allow_external=True`) over unencrypted HTTP without trusted identity proxy is rejected with `ValueError`.
    - Added protected `tokens_file` support to `WorkspaceBroker` with `0600` mode permissions and `0700` parent directory, ensuring authorization survives server restarts.
  - **P1 Google Drive Resumable Uploads & Authoritative Verification**:
    - Strict `Range` header parsing on HTTP 308 responses (`bytes=0-N` correctly updates offset to `N+1`; never assumes whole chunk was committed).
    - Reconciled committed byte offset via status query after ambiguous network errors before resuming.
    - Handled session expiration (`HTTP 404/410`) by clearing session URL and querying target folder to prevent duplicate file objects before re-initiating.
    - Authoritative integrity verification: eliminated local size substitution `remote_size = int(drive_file_meta.get("size", size_bytes))`. Queries remote metadata if missing; raises `DriveIntegrityError` if remote size cannot be verified.
    - Implemented bounded-memory streaming download in `retrieve()` using 64 KiB chunks into a private temporary file (`0600`), incremental SHA256/MD5 hashing, `fsync`, atomic rename, and automatic cleanup on error.
  - **P1 Persistent Transfer Journal & In-Process Retry Scheduler**:
    - Restricted `TransferJournal` directories to `0700` and files/locks to `0600`.
    - Implemented atomic writes with `fsync` and parent directory sync.
    - Replaced silent dictionary resets on corrupted JSON with `.corrupt.<timestamp>` copy preservation and recovery from `.bak` or explicit `JournalCorruptionError`.
    - Implemented in-process exponential backoff retry scheduler in `ArtifactUploadWorker` allowing `RETRY_PENDING` tasks to retry without worker restart.
    - Added pre-publication validation of local source file existence and SHA256 checksum.
  - **P1 Reproducibility Code Snapshots**:
    - Strictly excluded symlinked files and directories by default.
    - Validated canonical resolved paths strictly within project root.
    - Prevented self-referential inclusion of the destination archive itself.
    - Normalized tar entry paths with no leading `/` and no `..` traversal components.
  - **Comprehensive Regression Suite**:
    - Created/expanded `tests/test_v2_security_and_resumable_recovery.py` with 12 focused regression tests covering all P0 and P1 review findings.
    - Full test suite: **154 tests passing** in `11.7s` (100% pass, 0 failures, 1 skipped).
    - Syntax and CLI planning checks (`compileall`, `bash -n`, `colab_server.py plan`) pass cleanly (code 0).
    - **ACTUAL LIVE VM USAGE: NONE** (0 compute units consumed, 0 remote VMs allocated, 0 live Drive API calls made).
    - **STATUS: READY FOR INDEPENDENT SECURITY & CODE REVIEW.**

- **2026-10-24 — Milestone M7 / Gate G7 COMPLETE (branch `feat/v2-m7-integration`):**

  - Integrated complete project CLI subcommands in `colab_adapter/cli.py`: `init`, `validate`, `plan`, `up`, `sync`, `exec`, `status`, `stop`, `reconcile`.
  - Added monitoring CLI subcommands: `snapshot`, `server`.
  - Full `--json` output flags across all commands for automation and agent parsing (Q16).
  - Implemented end-to-end integration test suite in `tests/test_project_e2e.py` covering Scenarios A through F (Q01–Q18):
    - Scenario A: Template project initialization and manifest validation (Q01, Q02).
    - Scenario B: Zero-CLI offline planning and transfer calculation (Q03, Q17).
    - Scenario C: Locked session allocation and idempotent reuse.
    - Scenario D: Incremental two-phase workspace synchronization (Q04, Q05, Q06, Q10).
    - Scenario E: Attached code execution and structured status reporting.
    - Scenario F: Safe session termination and unmanaged session protection (Q08).
  - Full test suite: **111 tests passing** in `5.9s` on Python 3.8 and `6.1s` on Python 3.12 (100% pass, 0 failures).
  - Preserved 100% backward compatibility with V1 baseline (all 43 original tests pass unchanged).
  - **ACTUAL LIVE VM USAGE: NONE** (0 compute units consumed, 0 remote VMs allocated).
- **2026-10-24 — Milestone M8 / Gate G8 COMPLETE (branch `feat/v2-m8-monitor`):**
  - Implemented `colab_adapter/monitor/` package with `AccountSample`, `AccountUsageCollector`, `HealthStatus`, `MonitoringSummary`, `RunSummary`, and `MonitorServer`.
  - Poller dispatches strictly read-only `colab usage` and `colab sessions` queries without GPU allocation (Q19).
  - Ingested 1,440 synthetic snapshots without data loss (Q20).
  - Honest accounting: detects CU top-ups/refunds without negative spend (Q22); distinguishes unattributed account-wide spend when external sessions exist (Q23); immutable `monitoring_since` timestamp (Q24).
  - Resilient error handling: CLI error/timeout backoff and stale indicators (Q25).
  - Authenticated loopback HTTP server with token auth (Q26), SSE event stream with resume cursor (Q27), strict localhost default binding (Q28), and read-only 405 Method Not Allowed enforcement (Q29).
  - Embedded responsive SPA dashboard with real emitted metrics (Q30) and deep credential redaction (Q31).
  - Added unit test suite `tests/test_monitor_backend.py` (10 tests). Full offline suite: **105 tests passed** in `4.90s` on Python 3.8 and `4.84s` on Python 3.12.
  - **ACTUAL LIVE VM USAGE: NONE** (0 compute units consumed, 0 remote VMs allocated).
- **2026-10-24 — Milestone M6 / Gate G6 COMPLETE (branch `feat/v2-m6-artifacts`):**
  - Implemented `colab_adapter/artifacts.py` with `ArtifactManifest`, `ArtifactManager`, and `verify_local_artifact`.
  - Multi-tier artifact verification: file existence, minimum size, chunked streaming SHA256, ZIP structure inspection, and safe PyTorch `weights_only=True` loading (Q14).
  - Built staged atomic download pattern: downloads to `.tmp` file, validates hashes/structure, and renames atomically to final destination; staged files deleted on failure.
  - Implemented `colab_adapter/telemetry.py` with `TelemetryEmitter`, `TelemetryEvent`, and JSONL append-only storage matching M8 monitoring schema.
  - Enforced deep credential and token sanitization across metrics, details, and errors (Q12).
  - Accurate failure classification for `INTERRUPTED`, `CLEANUP_FAILED`, and `FAILED` states without masking errors (Q11).
  - Implemented explicit resume planning (`plan_resume_from_checkpoint`).
  - Added unit test suite `tests/test_artifacts_and_telemetry.py` (10 tests). Full offline suite: **95 tests passed** in `4.83s` on Python 3.8 and `4.29s` on Python 3.12.
  - **ACTUAL LIVE VM USAGE: NONE** (0 compute units consumed, 0 remote VMs allocated).
- **2026-10-24 — Milestone M5 / Gate G5 COMPLETE (branch `feat/v2-m5-sessions`):**
  - Implemented `colab_adapter/project_sessions.py` with `ProjectSession`, `ProjectSessionManager`, `ProjectSessionState`, and `VALID_TRANSITIONS`.
  - Enforced strict state transitions (PLANNED, AWAITING_APPROVAL, PROVISIONING, READY, RUNNING, STOPPING, TERMINATED, FAILED, NEEDS_RECONCILIATION).
  - Implemented collision-resistant session ID generation (`colab-v2-<name>-<uuid4[:8]>`) and offline planning with 0 cloud allocation calls (Q03).
  - Integrated `ColabProcessLock` and `ManagedSessionRegistry` ensuring intent is recorded before subprocess allocation.
  - Implemented session lease enforcement, idempotent session reuse on valid leases, and lock-aware reconciliation (`reconcile_leases`).
  - Hardened unmanaged session protection: attach, exec, and stop fail closed and strictly refuse unmanaged sessions (Q08).
  - Added unit test suite `tests/test_project_sessions.py` (11 tests). Full offline suite: **85 tests passed** in `4.11s` on Python 3.8 and `4.16s` on Python 3.12.
  - **ACTUAL LIVE VM USAGE: NONE** (0 compute units consumed, 0 remote VMs allocated).
- **2026-10-24 — Milestone M4 / Gate G4 COMPLETE (branch `feat/v2-m4-sync`):**
  - Implemented `colab_adapter/workspace.py` with `WorkspaceSyncEngine`, `SyncDelta`, and `compute_sync_delta`.
  - Added chunked content-addressed SHA256 indexing, incremental delta computation (`to_add`, `to_modify`, `unchanged`, `to_delete`), and two-phase staging before commit.
  - Enforced atomic commit order: `.colab_manifest.json` written LAST; failures mid-transfer trigger staged cleanup and raise `WorkspaceSyncError` without false commit.
  - Implemented safe deletion semantics controlled by `delete_remote` flag (default False).
  - Added test suite `tests/test_sync.py` (4 tests). Full offline suite: **74 tests passed** in `3.27s` on Python 3.8 and `3.25s` on Python 3.12 (Q01, Q04: 0 bytes on unchanged project, Q05: only modified file transferred, Q06: delete_remote flag honored, Q07: path safety, Q10: interrupted sync fails closed).
  - **ACTUAL LIVE VM USAGE: NONE** (0 compute units consumed, 0 remote VMs allocated).
- **2026-10-24 — Milestone M3 / Gate G3 COMPLETE (branch `feat/v2-m3-transport`):**
  - Implemented typed `BaseTransport` abstraction in `colab_adapter/transports/base.py`.
  - Built `OfficialCliTransport` implementing upload, download, list_dir, remove_file, exec_code, and health checks using official Google CLI commands. Enforced session registry ownership check before every call.
  - Built `SshProxyTransport` implementing guarded SSH-over-WebSocket using isolated single-session config without `-s`, eliminating TOCTOU auto-allocation risks. Fails closed with typed errors.
  - Added test suite `tests/test_transports.py` (7 tests). Full suite: **70 tests passed** in `3.25s` on Python 3.8 and `4.32s` on Python 3.12 (Q01, Q02, Q08, Q09, Q18).
  - **ACTUAL LIVE VM USAGE: NONE** (0 compute units consumed, 0 remote VMs allocated).
- **2026-10-24 — Milestone M2 / Gate G2 COMPLETE (branch `feat/v2-m2-manifest`):**
  - Implemented typed `ProjectSpec` in `colab_adapter/project_manifest.py` supporting schema_version 1, entrypoints, dependencies, arguments, accelerators, timeouts, and artifact declarations.
  - Built robust manifest parser supporting stdlib JSON natively and PyYAML optionally.
  - Implemented security validations: path canonicalization, path traversal rejection, symlink escape checks, and mandatory secret exclusions (`.git/**`, `.venv/**`, `.env*`, `*token*`, `*.pem`, `*.key`).
  - Implemented `plan_project`: 100% offline file discovery, chunked SHA256 hashing, and transfer byte estimations without invoking subprocesses.
  - Created test suite `tests/test_project_manifest.py` (8 tests). Full suite: **63 tests passed** in `3.58s` on Python 3.8 and `4.29s` on Python 3.12 (Q01, Q02, Q03, Q07, Q17).
  - **ACTUAL LIVE VM USAGE: NONE** (0 compute units consumed, 0 remote VMs allocated).
- **2026-10-24 — Milestone M1 / Gate G1 COMPLETE (branch `feat/v2-m1-contracts`):**
  - Audited official `google-colab-cli v0.7.5` source code and live host behaviors.
  - Documented upstream matrix, error taxonomy, and threat model in `docs/UPSTREAM_COMPATIBILITY_V2.md`.
  - Discovered critical upstream asymmetry: `colab ssh -s NAME` auto-provisions missing sessions, whereas `upload`, `download`, and `exec` fail closed.
  - Resolved TOCTOU race: proved upstream `_select_proxy_session(session=None)` fails closed without calling `_auto_create_session()`. Implemented `SshSafetyGuard.build_safe_ssh_proxy_invocation` using single-session isolated config without `-s`, while keeping SSH disabled by default for autonomous workflows.
  - Created `colab_adapter/errors.py` with typed, serializable V2 error hierarchy and credential/secret redaction.
  - Created `colab_adapter/capability.py` with non-allocating `CapabilityProbe` (ANSI cleaning, immutable tuple capabilities) and fail-closed `SshSafetyGuard`.
  - Added test suite `tests/test_capability_and_contracts.py` (12 tests). Full offline suite: `55/55 tests passed` in `1.55s` on Python 3.8 and 3.12 (Q01, Q02, Q03, Q18).
  - **ACTUAL LIVE VM USAGE: NONE** (0 compute units consumed, 0 remote VMs allocated).
- **2026-10-24 — Monitoring dashboard scope added:** M8 planned for account-wide CU balance/rate snapshots, managed training progress charts, private web UI viewable from any authorized computer. Historical readings before collector installation are unavailable; no UI deployed, no GPU allocated.
- **2026-10-24 — Plan initiated:** defined a V2 extension of the merged V1 provider inspired by GPU-rental developer experience. All V2 work remains NOT_STARTED. No runtime allocation performed by the planning effort.

## Links
- [Detailed milestones](milestones.md)
- [Feature specification](docs/FEATURES_V2.md)
- [Architecture and safety](docs/ARCHITECTURE_V2.md)
- [Expected results and acceptance](docs/EXPECTED_RESULTS_V2.md)
- [Agent implementation contract](docs/AGENT_IMPLEMENTATION_V2.md)
- [Secure training / compute monitor specification](docs/MONITORING_DASHBOARD_V2.md)
- [Original adapter](docs/ADAPTER.md)
