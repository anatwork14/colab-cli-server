# Colab CLI Server — Progress Ledger

> Last plan update: 2026-10-24. Scope: V2 VKong-inspired project/workspace UX **on top of** the existing official Google Colab CLI. This document distinguishes evidence from plans. Update it in the same PR as every substantive implementation change.

## Project status

- **V1: MERGED / verified on GitHub.** PR #2 squash-merged into main at commit 2bfd2194ad8259d785ca59935579eab6bdad052c. Human operator reported server synchronization and 43/43 offline tests on Python 3.8 and 3.12, OAuth diagnostics and no active sessions. The maintainer has not independently inspected the user's running server in this planning PR.
- **V1 live evidence:** earlier explicitly approved Colab T4 one-shot and stateful E2E executions proved basic remote CUDA execution, input upload, artifact download, checksum/ZIP checks and teardown. These tests predate V2 and do **not** validate SSH, directory sync or new development workflows.
- **Dashboard M8: PLANNED, not built.** The monitor will run on the Ubuntu host and use read-only official CLI account usage snapshots plus emitted run telemetry. Credentials never go to browser; HTTPS/VPN access must be separately configured.
- **V2 planning: IN PROGRESS in plan/v2-project-workspaces.** The V2 requirements, architecture, milestones and expected-results gates are being drafted. **No V2 implementation, live SSH test, GPU rental or cost measurement is claimed.**
- **Account safety:** NO cloud allocations, SSH auto-provision, Drive mount, paid API, GPU spend or unattended deployment authorized by these planning files.

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
| M8 | Secure browser training & compute monitor | COMPLETE | Issue #12, colab_adapter/monitor/, tests/test_monitor_backend.py; 105/105 tests pass |
| V2-Drive | Cross-Device Workspaces & Google Drive Artifacts | COMPLETE | Issue #21, colab_adapter/storage/, broker, reproducibility; 142/142 tests pass |
| M9 | Cross-provider abstraction (optional) | DEFERRED | Only after Colab V2 is independently usable |


## How an implementer updates this file

For every PR, record: (a) date and branch/commit, (b) tasks finished vs partial, (c) tests run with exact commands and output counts, (d) actual live-VM usage **or explicit none**, (e) observed results vs target values, (f) blockers/risks, (g) next owner. Never mark a milestone COMPLETE merely because the code compiles. Keep this status table consistent with milestones.md.

### Change log
- **2026-10-24 — V2 Extension / Gate G-Drive COMPLETE (branch `feat/v2-drive-workspaces`):**
  - Implemented provider-independent storage abstraction (`ArtifactStore`, `StoredArtifact`, `LocalArtifactStore`, `PublicationManifest`, `TransferJournal`, `TransferRecord`) in `colab_adapter/storage/`.
  - Implemented Google Drive API v3 resumable chunked upload protocol (`GoogleDriveArtifactStore`) with Range query resume, exponential backoff, Drive quota handling (403), token refresh (401), local SHA256 canonical digests, and remote MD5/size validation.
  - Connected M6 `ArtifactManager` with asynchronous background `ArtifactUploadWorker` featuring persistent journal task recovery across process restarts.
  - Implemented deterministic source snapshot creation (`source_snapshot.tar.gz`) with normalized mtime=0, uid=0, and strict secret filtering (`.env`, `*.pem`, `id_rsa`, `__pycache__`, `.git`).
  - Implemented Ubuntu `WorkspaceBroker` for multi-client read-only attach, token validation, audit logging, expired runtime inspection, and strictly zero VM allocation on attach.
  - Extended M8 monitoring server (`/api/v1/runs/<id>/artifacts`, `/drive`, `/manifest`) and responsive dashboard for Google Drive artifacts and run history continuity.
  - Implemented `CheckpointPolicyManager` with minimum interval throttling, keep_last retention, and strict preservation of at least 1 verified local checkpoint.
  - Added 7 comprehensive test suites in `tests/`: `test_storage_abstraction.py`, `test_google_drive_store.py`, `test_reproducibility.py`, `test_checkpoint_policy.py`, `test_upload_worker.py`, `test_workspace_broker.py`, `test_drive_and_broker_e2e.py`.
  - Full test suite: **142 tests passing** in `9.0s` (100% pass, 0 failures, 1 skipped).
  - Preserved 100% backward compatibility with V1 baseline and all M1–M8 milestones.
  - **ACTUAL LIVE VM USAGE: NONE** (0 compute units consumed, 0 remote VMs allocated, 0 live Drive tokens transmitted).
  - **IMPLEMENTATION READY — LIVE DRIVE VALIDATION PENDING USER APPROVAL.**

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
