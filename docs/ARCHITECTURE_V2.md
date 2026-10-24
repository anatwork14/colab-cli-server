# V2 Architecture — Safe Project-Based Remote GPU Workspace

> DESIGN PROPOSAL, 2026-10-24. Implementation status: NOT_STARTED. Extends the V1 provider instead of replacing it. Upstream Google Colab VM remains the actual compute host; this is NOT local CUDA forwarding.

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

**Source caveat:** balance and rate are aggregated account-wide; a new monitor cannot reconstruct older spend or itemize charges per job. Missing training progress or GPU samples display unavailable. See [MONITORING_DASHBOARD_V2.md](MONITORING_DASHBOARD_V2.md) for UI, source contracts, deployment, authorization and testing.

## 11. Open architectural decisions requiring M1 signoff

1. YAML with optional tested parser versus JSON baseline.
2. SSH transport no-auto-create enforceability (block SSH until confirmed; not a convenience assumption).
3. Remote directory atomicity model with CLI fallback.
4. Cloud runtime lease enforcement without keepalive; what is operator-managed vs automatic.
5. External checkpoint backend and privacy constraints (local SSD only by default).
6. Whether any watch/reload UX is worth Colab's ephemeral lifetime and cost.

Every decision must be logged in progress.md and reflected in acceptance tests.
