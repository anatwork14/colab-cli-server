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
| Upstream CLI | Official google-colab-cli; reported v0.7.5 on host | Capability probe and graceful version/feature detection | DONE V1 / TODO V2 |
| Provider API | ColabGPUProvider, JobSpec, approval, usage, tracking | Project-aware manifest and reusable workspace operations | DONE V1 / TODO V2 |
| Compute lifecycle | One-shot named run + stateful new/install/upload/exec/download/stop | Explicit attach/resume/reuse/TTL, no unexpected allocation | DONE V1 / TODO V2 |
| Session registry | ManagedSessionRegistry, atomic backups, locks, reconcile | Project/session mapping, leases, resource policy and idempotency | DONE V1 / TODO V2 |
| File transfer | Explicit input_files uploads and explicit expected_outputs downloads | Selective incremental directory sync, dry-run diff and safe deletion | DONE V1 / TODO V2 |
| SSH transport | Upstream documents ssh --proxy-mode | Adapter-managed opt-in capability probe and transport fallback | UPSTREAM ONLY / TODO V2 |
| Checkpoints | Local download, hash/structure/safe inference tiers | Durable remote-to-local artifacts, resume plan and metadata | DONE V1 / TODO V2 |
| Agent safety | Interactive user prompt, non-forgeable boundary expressly absent | Protected approval executor design for autonomous usage | DOC BOUNDARY / NOT IMPLEMENTED |
| Observability | JSONL experiment history, CLI doctor/usage | Structured project events, sync statistics, resource accounting | DONE V1 / TODO V2 |

## Workboard

| ID | Deliverable | Status | Evidence / next action |
| --- | --- | --- | --- |
| M0 | V1 baseline accepted | COMPLETE | PR #2, main commit above |
| M1 | Feature compatibility and security contracts | COMPLETE | Issue #5, docs/UPSTREAM_COMPATIBILITY_V2.md, colab_adapter/errors.py, colab_adapter/capability.py; 50/50 tests pass |
| M2 | Project manifest, validation and safe plan | NOT_STARTED | Schema, loader, examples and negative tests |
| M3 | SSH transport and no-surprise-allocation boundary | NOT_STARTED | Offline SSH capability probe, mock 404/401/429; no live probe |
| M4 | Content-addressed workspace sync | NOT_STARTED | Manifests, delta engine, push/pull, exclusions, safety tests |
| M5 | Attach/update/execute/stop lifecycle | NOT_STARTED | Approved transitions, session-aware CLI and recovery |
| M6 | Artifact/cache/resume and observability | NOT_STARTED | Durable output validation and metrics |
| M7 | Integration, end-to-end signoff and docs | NOT_STARTED | Offline CI; human-opt-in live evaluation separately |
| M8 | Secure browser training & compute monitor | NOT_STARTED | Account snapshots, run telemetry, private authenticated UI, cross-device access; G8/Q19-Q31 |
| M9 | Cross-provider abstraction (optional) | DEFERRED | Only after Colab V2 is independently usable |

## How an implementer updates this file

For every PR, record: (a) date and branch/commit, (b) tasks finished vs partial, (c) tests run with exact commands and output counts, (d) actual live-VM usage **or explicit none**, (e) observed results vs target values, (f) blockers/risks, (g) next owner. Never mark a milestone COMPLETE merely because the code compiles. Keep this status table consistent with milestones.md.

### Change log
- **2026-10-24 — Milestone M1 / Gate G1 IN REVIEW (branch `feat/v2-m1-contracts`):**
  - Audited official `google-colab-cli v0.7.5` source code and live host behaviors.
  - Documented upstream matrix, error taxonomy, and threat model in `docs/UPSTREAM_COMPATIBILITY_V2.md`.
  - Discovered critical upstream asymmetry: `colab ssh -s NAME` auto-provisions missing sessions, whereas `upload`, `download`, and `exec` fail closed.
  - Created `colab_adapter/errors.py` with typed, serializable V2 error hierarchy and credential redaction.
  - Created `colab_adapter/capability.py` with non-allocating `CapabilityProbe` and fail-closed `SshSafetyGuard`.
  - Added test suite `tests/test_capability_and_contracts.py`. Ran full offline suite: `50/50 tests passed` in `1.51s` on Python 3.8 and 3.12 (Q01, Q02, Q03, Q18).
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
