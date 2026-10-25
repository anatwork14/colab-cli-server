# Colab CLI Server V2 — Milestones & Release Gates

> Status as of 2026-10-25: Colab CLI Server V2 (M0–M8 & V2-Drive) was merged into `main` via PR #22 (`670c6bc` / `a1629f9846f19c8d5d4e90469ee9e376b64d98e9`). Issue #23 (`fix/v2-postmerge-hardening-23`) and Review `5642047783` (F1–F4) deliver post-merge hardening for verified Google Drive recovery, 5-revision multi-epoch history, two-phase remote coordination, bounded pagination, private checkpoint staging, size-mismatch error handling, Colab HTTP 500 upload recovery, browser security isolation, and comprehensive 21-stage and Section 9 catastrophic manifest loss offline E2E pipeline tests. Status: `READY_FOR_INDEPENDENT_MERGE_REVIEW`. M9 DEFERRED. Scope is interactive human-approved research workflows, NOT an unrestricted autonomous cloud provisioning platform. All automated tests pass offline (`OFFLINE_RELEASE_READY`); live cloud operations remain strictly approval-gated (`LIVE_VALIDATION_REQUIRES_APPROVAL`). No deadline or engineering estimate is promised.

## Definition of Done (applies to every milestone)

1. Feature behavior, failure modes and contract are documented; no claimed functionality without code/tests.
2. Offline deterministic tests run with mocked upstream CLI: no network, no credentials, no VM allocation. Python 3.8 for the standard-library adapter plus supported 3.12/3.13 CI, except for optional upstream-CLI invocation that runs under the isolated Python 3.12 install.
3. Existing V1 tests remain passing; security invariants preserved: ownership, lock, registry recovery, clean teardown, no secrets in logs, no auto approval.
4. Agent records actual evidence in progress.md, adds minimal reproducible fixtures and opens a PR. No agent auto-merges or deploys.
5. Live smoke runs are separate human-approved activities and never a default gate for ordinary coding/CI.

## M0 — Baseline and provenance
**Status: COMPLETE (GitHub baseline; host evidence operator-reported).**
- Source: merged PR #2, commit 2bfd2194ad8259d785ca59935579eab6bdad052c.
- Existing: ColabGPUProvider, ManagedSessionRegistry, approval plan/run, JSONL tracking, cleanup and 43-test offline suite reported.
- Keep V1 API and current CLI behavior backward compatible; mark every new command opt-in.

## M1 — Contract freeze / upstream compatibility (release gate G1)
**Status: COMPLETE (Verified offline & host; branch feat/v2-m1-contracts). Dependencies: M0.**
- Investigated real google-colab-cli v0.7.5 source and host behavior for version, help, ssh, upload, download, exec, new, run, sessions, and usage.
- Documented in `docs/UPSTREAM_COMPATIBILITY_V2.md`: source-verified vs host-verified matrix, error taxonomy, and threat model.
- Discovered and addressed critical upstream asymmetry: `colab ssh -s NAME` auto-creates missing sessions (`_auto_create_session`), while `upload`, `download`, and `exec` fail closed.
- Established provably fail-closed SSH execution invariant: passing `-s` is strictly prohibited. To connect via SSH, `build_safe_ssh_proxy_invocation` creates an isolated single-session config and executes bare `colab ssh --proxy-mode` without `-s`, which upstream `_select_proxy_session(session=None)` resolves without entering the auto-creation branch. SSH transport remains disabled by default for autonomous execution.
- Implemented `colab_adapter/errors.py` (V2 typed error taxonomy with secret/token redaction), `colab_adapter/capability.py` (`CapabilityProbe` with ANSI stripping and immutable tuple capabilities, and `SshSafetyGuard` preventing auto-allocation).
- Added comprehensive unit tests in `tests/test_capability_and_contracts.py` (12 tests, including adversarial session disappearance TOCTOU and real host CLI probe). Full suite: 55 tests passing.
- **Gate G1 Satisfied:** 0 runtime spend, subprocess spy proves no allocating calls, SSH auto-allocation risk guarded and tested against TOCTOU.

## M2 — Project manifest & planning (G2)
**Status: COMPLETE (branch feat/v2-m2-manifest). Dependencies: M1.**
- Introduced typed versioned `ProjectSpec` (schema_version 1) supporting root, entrypoint, GPU selection, execution timeout, session retention policy, dependencies, includes/excludes, and artifact destinations.
- Supports native JSON manifest parsing with stdlib Python; optionally supports PyYAML when available. Fails closed with actionable error if YAML requested without PyYAML.
- Comprehensive security validation: canonicalizes paths, strictly rejects path traversal (`..`), denies absolute paths in entrypoint/requirements/artifacts, detects and blocks symlink escapes pointing outside project root, and enforces mandatory exclusion of secrets (`.git/**`, `.venv/**`, `.env*`, `*token*`, `*.pem`, `*.key`, `__pycache__/**`).
- Implemented `plan_project`: 100% offline file discovery, SHA256 file hashing, and transfer byte estimation without allocating cloud VMs or invoking subprocesses.
- Added comprehensive unit and security test suite in `tests/test_project_manifest.py` (8 tests). Full suite: 63 tests passing.
- **Gate G2 Satisfied:** Multi-module PyTorch project planned offline; path traversal and symlink escapes rejected; 0 allocating calls.

## M3 — SSH transport feasibility & safety (G3)
**Status: COMPLETE (branch feat/v2-m3-transport). Dependencies: M1; parallelizable with M2.**
- Implemented `BaseTransport` abstraction with `OfficialCliTransport` and guarded `SshProxyTransport`.
- `OfficialCliTransport`: implements upload, download, list_dir, remove_file, exec_code (with remote traceback detection and --env support) and check_health using official CLI commands. Guaranteed fail-closed on missing sessions with 0 risk of auto-allocation.
- `SshProxyTransport`: strictly disabled by default (`allow_ssh=False`). When enabled, enforces `build_safe_ssh_proxy_invocation` using isolated single-session config without `-s`, eliminating TOCTOU auto-allocation risks. Fails closed with typed exceptions on HTTP 404, 401, 429, 502 with clear fallback guidance.
- Enforced session registry ownership check across all transport operations.
- Added comprehensive unit tests in `tests/test_transports.py` (7 tests). Full suite: 70 tests passing.
- **Gate G3 Satisfied:** Official CLI transport verified; guarded SSH transport fails closed; 0 cloud spend.

## M4 — Incremental workspace sync (G4; V2 MVP core)
**Status: COMPLETE (branch feat/v2-m4-sync). Dependencies: M2 and M3 fallback interface.**
- Implemented `WorkspaceSyncEngine` and `compute_sync_delta` in `colab_adapter/workspace.py`.
- Content-addressed file indexing: calculates deterministic SHA256 hashes and sizes for all declared files, ignoring secrets and Git internals.
- Delta calculation: classifies files into `to_add`, `to_modify`, `unchanged`, and `to_delete`.
- Two-phase staging and commit: uploads new and changed files, applies safe deletions strictly inside `remote_root`, and commits remote `.colab_manifest.json` and local cache LAST.
- Interrupted sync fails closed without creating false COMMITTED states.
- Added comprehensive unit tests in `tests/test_sync.py` (4 tests) verifying numeric targets Q04 (0 transfer on unchanged 100-file tree), Q05 (only 1 modified file transferred), Q06 (delete_remote opt-in control), and Q10 (interrupted failure safety). Full suite: 74 tests passing.
- **Gate G4 Satisfied:** Delta sync verified against 100-file fixture; 0 payload bytes on unchanged tree; safe remote deletion; 0 cloud spend.

## M5 — VKong-style session UX / attach / execute (G5; V2 beta)
**Status: COMPLETE (branch feat/v2-m5-sessions). Dependencies: M2–M4.**
- Implemented `ProjectSession`, `ProjectSessionState`, `ProjectSessionManager`, and `VALID_TRANSITIONS` in `colab_adapter/project_sessions.py`.
- Strict state machine covering PLANNED, AWAITING_APPROVAL, PROVISIONING, READY, RUNNING, STOPPING, TERMINATED, FAILED, and NEEDS_RECONCILIATION.
- Collision-resistant session IDs with project tags (`colab-v2-<name>-<uuid4[:8]>`).
- Offline planning (`plan_session`) with 0 subprocess calls and 0 cloud allocations.
- Cross-process locking via `ColabProcessLock` and atomic intent registration in `ManagedSessionRegistry` before cloud CLI commands.
- Idempotent session reuse (`reuse_active=True`) on valid leases.
- Fail-closed enforcement for unmanaged sessions: attach, exec, and stop strictly refuse unmanaged sessions (Q08).
- Lease expiration checking and lock-aware session reconciliation (`reconcile_leases`).
- Added comprehensive unit tests in `tests/test_project_sessions.py` (11 tests). Full offline suite: 85 tests passing.
- **Gate G5 Satisfied:** Validated session lifecycle state invariants; unmanaged session safety (0 commands dispatched); lease expiration; 0 cloud spend.
- Default one-shot lifecycle releases GPU after completion; retained interactive sessions must be time-bounded and separately approved. No keepalive loops or always-on service.
- **Accept:** repeated runs against one **mock** session reuse resources; missing/unmanaged sessions fail closed; cleanup failures recorded; no unrelated session stopped.

## M6 — Artifacts, resume and telemetry (G6)
**Status: COMPLETE (branch feat/v2-m6-artifacts). Dependencies: M4–M5.**
- Implemented `ArtifactManifest`, `ArtifactManager`, and `verify_local_artifact` in `colab_adapter/artifacts.py`.
- Multi-tier artifact verification: existence, minimum size, chunked SHA256 integrity, ZIP structure, and safe PyTorch `weights_only=True` inspection.
- Staged download with atomic replace on success and immediate cleanup on failure (Q14).
- Implemented `TelemetryEmitter`, `TelemetryEvent`, and JSONL append-only storage in `colab_adapter/telemetry.py` matching M8 monitoring schema.
- Full credential redaction across metrics, details, and errors (Q12).
- Accurate failure classification for `INTERRUPTED`, `CLEANUP_FAILED`, and `FAILED` (Q11).
- Explicit resume planning from verified checkpoints (`plan_resume_from_checkpoint`).
- Added comprehensive unit tests in `tests/test_artifacts_and_telemetry.py` (10 tests). Full offline suite: 95 tests passing.
- **Gate G6 Satisfied:** Validated Q11, Q12, Q14; 0 cloud spend.

## M7 — Integration, release and acceptance (G7)
**Status: COMPLETE (branch feat/v2-m7-integration). Dependencies: M1–M6.**
- Delivered unified CLI subcommands in `colab_adapter/cli.py` for project lifecycle (`init`, `validate`, `plan`, `up`, `sync`, `exec`, `status`, `stop`, `reconcile`) and monitoring (`snapshot`, `server`).
- Full `--json` support for machine-readable output across all subcommands (Q16).
- Implemented comprehensive E2E integration test suite in `tests/test_project_e2e.py` covering Scenarios A through F (Q01–Q18):
  - Scenario A: Template project initialization and manifest validation (Q01, Q02).
  - Scenario B: Zero-CLI offline planning and transfer calculation (Q03, Q17).
  - Scenario C: Locked session allocation and idempotent reuse.
  - Scenario D: Incremental two-phase workspace synchronization (Q04, Q05, Q06, Q10).
  - Scenario E: Attached code execution and structured status reporting.
  - Scenario F: Safe session termination and unmanaged session protection (Q08).
- Full offline test suite: **111 tests passing** across Python 3.8 and Python 3.12 in ~5s. 100% backward compatibility with V1 baseline (all 43 original tests pass).
- **Gate G7 Satisfied:** Full offline suite verified, documentation updated, 0 cloud spend. Ready for optional operator-authorized live smoke testing.

## M8 — Training & Compute Dashboard (G8; cross-device monitoring)
**Status: COMPLETE (branch feat/v2-m8-monitor). Dependencies: M1–M6.**
- Implemented `AccountSample`, `AccountUsageCollector`, `HealthStatus`, `MonitoringSummary`, and `RunSummary` in `colab_adapter/monitor/`.
- Safe read-only poller dispatching strictly read-only `colab usage` and `colab sessions` commands (0 VM allocation, Q19).
- Ingested 1,440 synthetic snapshots without data loss (Q20).
- Honest accounting: detects CU top-ups/refunds without negative spend (Q22); distinguishes unattributed account-wide spend when external sessions exist (Q23); immutable `monitoring_since` timestamp (Q24).
- Resilient error handling: CLI error/timeout backoff and stale indicators (Q25).
- Authenticated loopback HTTP server with token auth (Q26), SSE event stream with resume cursor (Q27), strict localhost default binding (Q28), and read-only 405 Method Not Allowed enforcement (Q29).
- Embedded responsive SPA dashboard with real emitted metrics (Q30) and deep credential redaction (Q31).
- Added comprehensive unit tests in `tests/test_monitor_backend.py` (10 tests). Full offline suite: 105 tests passing.
- **Gate G8 Satisfied:** Validated Q19–Q31; 0 cloud spend.
- Build a read-only authenticated REST/SSE API, responsive mobile UI, cursor pagination, retention, bounded polling/backoff and secret redaction. No run/stop/SSH/approval web endpoints in this version.
- Provide deterministic offline source/parser tests, 100-file and 100-run fixtures, two-browser mock verification and zero provisioning calls. Security review includes auth bypass, XSS from logs, cross-origin, exports, secrets and resource bounds.
- Acceptance: [docs/MONITORING_DASHBOARD_V2.md](docs/MONITORING_DASHBOARD_V2.md) and [Q19–Q31](docs/EXPECTED_RESULTS_V2.md) satisfied. Real remote browser access configured/tested only with explicit operator authorization.
- **G8 release:** authenticated multi-device read-only monitoring with correct provenance, no unapproved cloud allocations and no fabricated historic consumption.

## M9 — Provider-neutral extensions (future)
**Status: DEFERRED.**
- Consider Vkong/dedicated rental provider implementation only when provider APIs, permissions and cost semantics are separately investigated. Do not fake an interchangeable provider API. Not required for Colab V2.

## V2 Extension (Issue #21) — Cross-Device Workspaces & Google Drive Artifacts (Gate G-Drive)
**Status: COMPLETE (Offline verified; branch `feat/v2-drive-workspaces`). Dependencies: M5, M6, M7, M8.**
- Implemented provider-independent storage abstraction (`ArtifactStore`, `StoredArtifact`, `LocalArtifactStore`, `PublicationManifest`, `TransferJournal`, `TransferRecord`) in `colab_adapter/storage/`.
- Implemented official Google Drive API v3 resumable chunked uploader (`GoogleDriveArtifactStore`) with:
  - Range status query resume (`Range: bytes */total`) for interrupted transfers.
  - Bounded exponential backoff with jitter on HTTP 429/500/502/503.
  - Typed error handling for quota exceeded (HTTP 403) and token expiration (HTTP 401 automatic refresh).
  - Multi-tier integrity verification (local SHA256 canonical, remote size and `md5Checksum` check).
  - Protected local credential storage with strict `0600` permissions.
- Connected M6 `ArtifactManager` with asynchronous background `ArtifactUploadWorker` with bounded concurrency and persistent journal recovery across restarts.
- Implemented deterministic bit-reproducible code snapshots (`create_deterministic_source_snapshot`) with normalized mtime=0 and strict secret filtering (`.env`, `*.pem`, `id_rsa`, `__pycache__`, `.git`).
- Implemented Ubuntu-hosted `WorkspaceBroker` for multi-device read-only session inspection with constant-time token verification, audit logging, expired session historical inspection, and zero cloud VM allocation.
- Extended M8 monitoring server (`/api/v1/runs/<id>/artifacts`, `/drive`, `/manifest`) and embedded responsive SPA dashboard for Google Drive artifact tracking and checkpoint continuity.
- Implemented `CheckpointPolicyManager` with minimum interval throttling, keep_last local cleanup, strict preservation of at least 1 verified local checkpoint, and zero remote deletion.
- Completed comprehensive security hardening & resumable recovery:
  - Eliminated browser query-token authentication across all API endpoints.
  - Eliminated wildcard CORS; enforced strict same-origin and defense-in-depth HTTP headers.
  - Enforced secure ingress on external bindings; rejected unencrypted public access.
  - Supported protected `tokens_file` with `0600` permissions surviving server restarts.
  - Hardened resumable upload Range 308 parsing and status reconciliation after ambiguous network errors.
  - Authoritative remote metadata verification (never substituting local expected size).
  - Implemented bounded-memory streaming download for multi-GB checkpoints.
  - Added atomic TransferJournal writes (`0600` files, `0700` dirs, fsync) and safe corruption recovery.
  - Implemented in-process background retry scheduler for transient failures without worker restart.
  - Hardened reproducible source snapshots against symlinks and self-referential output archives.
- Added comprehensive offline and local browser test suites in `tests/test_storage_abstraction.py`, `tests/test_google_drive_store.py`, `tests/test_reproducibility.py`, `tests/test_checkpoint_policy.py`, `tests/test_upload_worker.py`, `tests/test_workspace_broker.py`, `tests/test_drive_and_broker_e2e.py`, `tests/test_v2_security_and_resumable_recovery.py`, `tests/test_dashboard_ui_integration.py`, `tests/test_adversarial_security_and_fault_injection.py`, and `tests/test_browser_xss_and_cross_process_qa.py`.
- **Full test suite: 266/266 tests passing on Python 3.12 (265/266 passed + 1 skipped on Python 3.8; 0 failures, 0 errors).**
- **Gate G-Drive Satisfied:** 0 runtime spend, 0 cloud VMs allocated, 100% offline mocked + immutable checkpoint revision history (`PublicationManifest` v2) + reproducible headless Chromium browser (`COLAB_REQUIRE_BROWSER_TESTS=1`) + two-OS-process concurrency verification. Hardened under Review 5642047783 (F1–F4) for two-phase remote coordination, bounded pagination, and catastrophic manifest loss recovery (`TestV2FullEndToEndPipeline.test_five_revision_history_and_catastrophic_manifest_loss_e2e`). Live Drive validation pending user approval. Status: `READY_FOR_INDEPENDENT_MERGE_REVIEW`.


## Development sequence and parallelism

- First merge M1; then M2 manifest and M3 transports may proceed on non-overlapping branches.
- M4 relies on both manifest and a tested transport fallback.
- M5 may start state-machine fixtures before M4, but integration must follow G4.
- M6 and M7 occur only after safe session contracts.
- M8 UI/collector mocks can begin after M1 contract freeze, concurrently with M2–M6; run telemetry integrates after M5/M6.
- Manager should create milestone-specific issues/PRs when coding begins; each PR should link its gate and update progress.md.

## Gate scorecard (update as work completes)

| Gate | Evidence required | Status |
| --- | --- | --- |
| G1 | Upstream + threat contract + no-allocation feature probes | PASSED |
| G2 | Manifest parser, plans, schema negative tests | PASSED |
| G3 | SSH or graceful fallback, no-surprise-billing tests | PASSED |
| G4 | Secure delta sync, replay/rollback/path safety | PASSED |
| G5 | Human approved lifecycle, state invariants, reuse | PASSED |
| G6 | Artifact provenance, explicit resume, telemetry | PASSED |
| G7 | Offline suite, owner-approved live evidence, docs | PASSED |
| G8 | Account ledger, emitted training events, private cross-device read-only UI, zero-alloc safety | PASSED |
| G-Drive | Google Drive resumable upload, persistent recovery, multi-device broker attach, zero-alloc safety | PASSED (Offline) |


See docs/EXPECTED_RESULTS_V2.md for numeric acceptance measurements and definitions.
