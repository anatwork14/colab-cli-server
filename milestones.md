# Colab CLI Server V2 — Milestones & Release Gates

> Status as of 2026-10-24: M0 COMPLETE; M1–M8 NOT_STARTED; M9 DEFERRED. Scope is interactive human-approved research workflows, NOT an unrestricted autonomous cloud provisioning platform. No deadline or engineering estimate is promised.

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
**Status: NOT_STARTED. Dependencies: M2–M4.**
- Proposed project commands: up (explicit human approval), attach (existing only), exec, sync, status, logs, stop, reconcile. No silent remote provision from attach/sync/status.
- Separate session state from job state; implement explicit ownership, TTL/retention budget, lease/liveness, per-project lock, idempotent retries, interruption cleanup and stateful reuse.
- Default one-shot lifecycle releases GPU after completion; retained interactive sessions must be time-bounded and separately approved. No keepalive loops or always-on service.
- **Accept:** repeated runs against one **mock** session reuse resources; missing/unmanaged sessions fail closed; cleanup failures recorded; no unrelated session stopped.

## M6 — Artifacts, resume and telemetry (G6)
**Status: NOT_STARTED. Dependencies: M4–M5.**
- Model durable output and checkpoint storage outside temporary Colab VM; metadata records project ID, source revision, config, input/output hashes, GPU class, actual usage snapshot and local artifact path.
- Resume must be explicit and compatible with checkpoint contract; no promise that Colab VM/dataset remains available.
- Report project run states, sync diffs, stage timings, transfer sizes, session age, error class and cleanup state; redact tokens and paths as required.
- **Accept:** mocked interrupted training -> verified checkpoint -> resumed new session; manifest corruption fails closed; logs do not expose secrets.

## M7 — Integration, release and acceptance (G7)
**Status: NOT_STARTED. Dependencies: M1–M6.**
- Deliver cohesive README, AGENTS.md, examples, architecture docs, help output, integration tests and upgrade path from V1; no breaking API without version migration.
- Full offline CI + negative/security tests; compare new workflows to V1 baseline.
- Optional **human-approved** live tests: (a) CPU session SSH capability, (b) T4 multi-module project sync/exec/download, (c) reconnect after interruption, (d) balance and session audit. Human approval required for EACH provision/retention; no automatic live CI.
- **Accept:** all expected-results V2 minimum gates met; owner signs off real Colab behavior; documented feature fallbacks.

## M8 — Training & Compute Dashboard (G8; cross-device monitoring)
**Status: NOT_STARTED.** Read-only account collector, data model and fixture UI may be developed after M1 in parallel with M2–M6; real run-specific metrics integrate after M5/M6.
- Create an Ubuntu-hosted, loopback-only-by-default web monitor accessible from authorized computers through a separately approved encrypted/authenticated VPN or HTTPS access gateway. Never expose raw Colab credentials or an unauthenticated web port.
- Poll only official read-only `colab usage` (account CU balance, aggregate CU/h rate, active assignments) and `colab sessions`; store timestamped JSONL snapshots privately. Do not allocate or keep alive any Colab runtime from dashboard or test suite.
- Build a dashboard with Overview, Training Runs, Run Detail, Compute History, Sessions, History/Artifacts, Health. Charts from true emitted progress/metrics, log tail, checkpoints; optional VRAM/utilization only from existing approved jobs with instrumentation.
- Data-quality contract: account-wide CU snapshots cannot reconstruct all prior usage or exact job spend. Show monitor-start timestamp, collection gaps, top-up/adjustment uncertainty, unknown attribution and stale data.
- Build a read-only authenticated REST/SSE API, responsive mobile UI, cursor pagination, retention, bounded polling/backoff and secret redaction. No run/stop/SSH/approval web endpoints in this version.
- Provide deterministic offline source/parser tests, 100-file and 100-run fixtures, two-browser mock verification and zero provisioning calls. Security review includes auth bypass, XSS from logs, cross-origin, exports, secrets and resource bounds.
- Acceptance: [docs/MONITORING_DASHBOARD_V2.md](docs/MONITORING_DASHBOARD_V2.md) and [Q19–Q31](docs/EXPECTED_RESULTS_V2.md) satisfied. Real remote browser access configured/tested only with explicit operator authorization.
- **G8 release:** authenticated multi-device read-only monitoring with correct provenance, no unapproved cloud allocations and no fabricated historic consumption.

## M9 — Provider-neutral extensions (future)
**Status: DEFERRED.**
- Consider Vkong/dedicated rental provider implementation only when provider APIs, permissions and cost semantics are separately investigated. Do not fake an interchangeable provider API. Not required for Colab V2.

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
| G5 | Human approved lifecycle, state invariants, reuse | NOT_STARTED |
| G6 | Artifact provenance, explicit resume, telemetry | NOT_STARTED |
| G7 | Offline suite, owner-approved live evidence, docs | NOT_STARTED |
| G8 | Account ledger, emitted training events, private cross-device read-only UI, zero-alloc safety | NOT_STARTED |

See docs/EXPECTED_RESULTS_V2.md for numeric acceptance measurements and definitions.
