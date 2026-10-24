# Agent Implementation Contract — Colab Project Workspaces V2

**Audience:** coding managers, implementation workers, independent architecture/security reviewers. **Source of truth:** progress.md, milestones.md, docs/FEATURES_V2.md, docs/ARCHITECTURE_V2.md, docs/EXPECTED_RESULTS_V2.md. All files currently describe **planned**, not implemented, V2 functionality.

## Objective

Evolve the merged V1 ColabGPUProvider into a VKong-inspired project-centric remote GPU development workflow using *official* google-colab-cli (installed separately). Make local research projects executable on managed Colab GPUs with versioned manifest, selective incremental sync, optional guarded SSH, session reuse during explicitly approved leases, artifacts, logs, checkpoints and safe teardown.

The current codebase already handles interactive one-shot/stateful jobs, registry/locking, tracking and Python-source execution. **Extend it; do not rewrite or bypass it.**

## Binding safety instructions

1. Do NOT allocate Colab CPU/GPU/TPU, run colab new/run, connect with ssh where auto-provision is possible, retain a session, call chargeable APIs or use a live Google Drive mount without the user's separate operation-specific approval. **Offline default**.
2. Do NOT run agents with Google OAuth credentials exposed in their prompts, logs or commits; do not forge ApprovedJob, auto-fill interactive approvals, automate a TTY or use same-UID access as authorization.
3. Do NOT modify Ubuntu system Python, CUDA, drivers, local GTX 1080 Ti workloads or running processes. No sudo. Isolate official Python >=3.12 CLI from stdlib wrapper supporting Python >=3.8.
4. Do NOT silently clone full local home directory, private research data, token files, SSH keys, .env, .git or datasets. Fail closed on paths/symlinks.
5. Do NOT stop or alter unmanaged Colab sessions. Never use SSH -s NAME unless a protected wrapper can guarantee authorization and prevent implicit creation; fallback to official exec/upload.
6. No hidden keepalive, billing avoidance, quota circumvention, long-lived inference hosting or promise of permanent Colab GPU.
7. Never merge/deploy automatically. All implementation through small reviewed PRs; do not delete V1 features or rework unrelated repositories.
8. Tests must be offline by default: mock subprocess, SSH and transports; record any live use only after human approval.

## Suggested ownership model

- **Manager / integrator (one):** read documents, audit real source, propose small PR sequence, own API contracts and public CLI, resolve conflicts, enforce acceptance gates.
- **Architecture & security reviewer (independent):** assess SSH implicit allocation, transports, path traversal, session leases and failure recovery before approving a transport design. Reviewer must not silently implement or self-approve.
- **Worker A — schema + tests:** M1/M2 manifest parser, validator, project fixtures, plan CLI, error contracts. Files primarily project_manifest.py, test_project_manifest.py, examples, docs.
- **Worker B — transport:** M1/M3 upstream compatibility probe, official CLI transport, optional SSH-over-WebSocket adapter. Files primarily transports/* and test_transport_*.py. Never provision live session.
- **Worker C — sync engine:** M4 file index/content hashes, remote staging, delta plan and fallback transport. Files primarily workspace_index.py, workspace.py, test_sync_*.py; integrates A and B contracts.
- **Worker D — lifecycle/observability:** M5/M6 project session manager, artifact output metadata, safe reuse/cleanup and metrics. Files project_sessions.py, artifacts.py, test_session_*.py.
- **Worker E — QA/integration:** adversarial fixtures, package import compatibility, offline E2E, upgrade/migration docs, README updates. Files primarily tests, examples, docs. Coordinate collisions.
- Worker parallelism only when API decisions from M1 are frozen and file ownership is non-overlapping; each worker reports evidence to manager. Avoid spawning all workers to edit provider.py and cli.py simultaneously.

## Execution order

### Step 0: inspect and freeze baseline (read-only)
- Confirm main includes commit 2bfd2194ad8259d785ca59935579eab6bdad052c or newer; inspect open PRs and current server package import path before any installation.
- Read complete AGENTS.md, README.md, docs/ADAPTER.md, tests and the five V2 documents.
- Inspect official CLI v0.7.5 flags/source (run -s, ssh --proxy-mode, upload/download, sessions, exec) and error paths; don't trust prior assistant claims without source verification.
- Run offline V1 suite; report exact baseline.
- Publish decisions for manifest format, session/approval boundary, SSH no-auto-create guarantee, remote staging transaction and CLI command naming.

### Step 1: M1/M2
- Implement typed project manifest and deterministic local plan with safe excludes, no expensive mandatory dependencies.
- Add project validate/plan and --json semantics; prove no allocation by monkeypatching ALL upstream invocation paths to fail.
- Demonstrate multi-module project with relative imports and expected outputs in mock.

### Step 2: M3
- Implement Transport interface and official CLI fallback first. SSH transport only behind explicit capability and safe approved-session gate; if SSH may auto-provision, disable it for noninteractive execution and document why.
- Exercise all SSH failure statuses and reconnect/interruption mocks; no new Colab CLI API implementation.

### Step 3: M4
- Implement file index + deterministic diff; stage/verify/commit manifest; provide safe opt-in remote deletion. Test symlink/path traversal/credentials and interrupted sync.
- Differentiate intended transfer from actual transfer; don't claim performance improvements until measured.

### Step 4: M5/M6
- Implement project session lifecycle, attach/update/exec/status/logs/stop, retention lease, cleanup and reconciliation without breaking V1.
- Add checkpoint manifest verification and manual resume on newly approved runtime; record provenance and resource snapshots.

### Step 5: M7
- Add mocked E2E starting with project plan -> approved mock session -> sync -> exec -> checkpoint -> stop, plus fallback 404 and crash scenarios.
- Update all docs, CLI help and agent skill. Run Python 3.8+ regression and CI on 3.12/3.13.
- Only when owner separately authorizes: propose a **minimal** bounded live SSH/Colab test with specific GPU, unit/time ceiling, expected artifacts and cleanup. Stop and audit sessions.

## Pull-request rules and artifact update discipline

- Use branch per milestone: feat/v2-m1-contract, feat/v2-m2-manifest, etc.; base on latest main after the planning PR is merged, or on explicitly documented integration branch if not merged yet.
- PR body must contain scope, requirements IDs, failure modes, non-goals, tests with exact command/result, and whether cloud resources were allocated.
- Update progress.md rows, release gate scorecard in milestones.md, and expected-results observed data in the **same PR** that implements a feature. No future-tense checkbox claims.
- Keep docs links intact, tests hermetic and API backward compatible.
- Manager creates scoped GitHub issues/tasks once ready to implement; issue acceptance references F-IDs and milestone gates. Track blockers in progress.md, not memory alone.
- No PR merge without user/maintainer instruction; no deployment without explicit permission.

## Mandatory evaluation checklist

- No cloud allocations from plan/validate/status or unauthorized SSH? Test via subprocess spy.
- No session creation from attaching missing name? Negative test.
- No unmanaged session stop? Negative test.
- Correct file list, hashes and zero reupload on unchanged fixture?
- Controlled remote delete never escapes project root and requires explicit opt-in?
- Source code and config only; private credentials and datasets are ignored or explicitly authorized?
- Simulated SSH 404, 401, 429, 502 safely falls back without new VM?
- Concurrent apply/exec, session crash and malformed registry fail closed?
- PyTorch checkpoint loading remains safe; interrupted training statuses correct?
- Host Python 3.8 import, CI 3.12/3.13, all original 43 tests remain green?
- Remote GPU remains separated from local CUDA; no local driver edits?
- Live cost, sessions and verification tiers accurately distinguished from offline mocks?

## Definition of delivery

A PR that satisfies the exact milestone gate and documents the observed evidence. If a capability is unavailable upstream, document a clear fallback and mark it unsupported rather than fabricating success.

### Final manager report format
1. PR link, branch, head SHA, changed files.
2. Milestone / feature IDs completed vs remaining.
3. Offline test commands, counts and GitHub CI check results.
4. Backend capabilities actually proven vs mocked vs unsupported.
5. Cloud sessions created, explicit owner approval, compute usage and post-test session state; say **NONE** for offline work.
6. Threat model / residual risks and concrete next PR.
7. Confirm no auto-merge, deployment or local GPU changes.

## References
- https://github.com/googlecolab/google-colab-cli/tree/main/skills/colab-operator
- https://github.com/googlecolab/google-colab-cli/blob/main/docs/06_ssh_access.md
- https://research.google.com/colaboratory/faq.html
