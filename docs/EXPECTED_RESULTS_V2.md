# Expected Results, Metrics & Acceptance Matrix — V2

> DESIGN TARGETS only, **not measured results**. Dated 2026-10-24. Actual measurements must be recorded in progress.md with the test command, machine, fixture and upstream CLI version. Do not imply a guaranteed GPU type, budget rate, latency or VM lifespan.

## User-visible outcomes (release G7)

1. A researcher points the CLI at an ordinary multi-module Python project, validates a manifest and sees precisely which source files/dependencies/output paths would be used before any Colab allocation.
2. The operator authorizes an individual bounded Colab GPU job and receives verified artifacts, status, logs, cleanup disposition and resource-usage snapshots.
3. During an approved interactive session, changing one small source module sends only the changed file (when supported by chosen transport). Unmodified files do not transfer again.
4. Failed/missing SSH support does not force a project rearchitecture: same project can use upstream CLI upload/exec fallback without automatic second allocation.
5. The user's local GTX 1080 Ti remains untouched; Colab compute remains remote. No persistent GPU entitlement or transparent local CUDA is asserted.

## Numeric offline test targets (must pass before release)

| Metric ID | Measurement / fixture | Expected threshold |
| --- | --- | --- |
| Q01 | Existing repository offline unit suite | 100% V1 regression pass, including prior 43 tests |
| Q02 | All new manifest/transport/sync/lifecycle/negative tests | 100% pass on supported CI Python versions |
| Q03 | Plan/validate/diff on mocks | 0 Colab new/run/auto-SSH provision calls |
| Q04 | 100-file fixture unchanged between syncs | 0 transferred files, 0 payload bytes (excluding control metadata) |
| Q05 | 100-file fixture with exactly 1 modified 1KiB source file | Exactly 1 modified source file scheduled for transfer; no other source retransmitted |
| Q06 | 100-file fixture with one deleted file and delete_remote=false | 0 remote delete calls |
| Q07 | Path traversal / symlink escape / secrets fixtures | 100% rejected before transfer (test set must enumerate representative patterns) |
| Q08 | Unknown or unmanaged Colab session | 0 stop/exec/sync commands dispatched |
| Q09 | SSH unsupported/404/401/429/502 simulated | Explicit error and/or safe official CLI fallback; 0 new allocations |
| Q10 | Interrupted transfer / corrupt manifest | No false COMMITTED state; recoverable prior manifest or explicit failure |
| Q11 | Forced job timeout, signal, cleanup failure | Accurate FAILED/INTERRUPTED/CLEANUP_FAILED status; cleanup error separate |
| Q12 | Credential string corpus in logs/tracker/CLI JSON | No known seeded credentials survive sanitization |
| Q13 | Concurrency: 8 threads + 2 mocked processes | No duplicate owned session allocation, no lost registry update |
| Q14 | Artifact verification | All missing/hash-mismatched/truncated outputs rejected |
| Q15 | Default retention | 0 auto-retained sessions after successful one-shot job |
| Q16 | CLI machine format | Stable schema, deterministic ordering and nonzero error exits |
| Q17 | Dry-run authorization | No TTY prompt or secret/OAuth read needed for local manifest validation |
| Q18 | Version compatibility | stdlib adapter imports on Python 3.8; CI on 3.12/3.13; upstream separate Python >=3.12 |

**Sync byte accounting:** Q04 counts source payload bytes sent, not protocol headers, hashes or manifest metadata. Q05 deliberately measures *planned files* offline. Network benchmarks and speedup claims require a later, separately approved real run.

## Live tests (NEVER automatic in CI)

| Scenario | Human approval needed | Evidence to capture | Pass definition |
| --- | --- | --- | --- |
| L1: capability smoke | Explicit approval even for CPU VM | Official CLI version, SSH command result, runtime ID, session stop | SSH works OR explicit documented unsupported outcome with fallback |
| L2: T4 multi-module training | Explicit GPU allocation approval | GPU identity, code version, remote imports, status JSON, output hashes | Remote training executes, outputs verified, no orphan |
| L3: incremental update | Retained-session approval for bounded duration | Initial vs second file diffs, actual upload counts/bytes, logs | Only modified source payload resent where transport supports |
| L4: recovery | Approval for scenario and any new allocation | Interrupted state, resume checkpoint hash, session audit | No unmanaged sessions stopped; resumed result traceable |
| L5: usage audit | Read-only account query | Before/after snapshot, timing, cleanup state | No hidden repeat allocation; no guaranteed cost assumption |

No live test may be conducted just because this document exists. User-approved exact GPU, number of runs, time cap, and acceptable compute consumption must be recorded before execution. Never display account tokens, authorization codes, private remote paths, or proprietary datasets in artifacts.

## Measurable developer experience

- Proposed workflow: project init/validate/plan -> explicit approval -> up -> sync -> exec -> artifacts -> stop.
- Desired cold path: at most one explicit approval boundary and one runtime allocation per configured job, not a script-by-script VM.
- Desired warm path: no second VM allocation while a human-approved lease is active, unless the human separately approves replacement.
- Every command outputs operation_id, selected project/session, planned vs actual file count/bytes, status, output paths, and cleanup status where applicable.
- All user-facing commands have --help, readable errors and optional --json where meaningful; interactive-approval output never exposes secrets.

## Definition of success by capability

| Feature | Minimum passing demonstration | Not required |
| --- | --- | --- |
| Project manifest | 2 multi-file sample projects, invalid cases fail | Supporting every YAML extension |
| SSH | Graceful fallback when unavailable; approved existing-session test optional | SSH on every runtime |
| Incremental sync | Content-addressed delta, replayable and safe deletion | Fastest network throughput |
| Artifact continuity | Verified local checkpoints, resume from new session | Persistent Colab disk |
| Security | No silent allocation; no credential copy; ownership safe | Hostile same-UID agent isolation without protected executor |
| Compatibility | Existing V1 workflows unchanged | Container images on Colab |
| Cost | Resource reads, explicit approval, bounded sessions | Guaranteed compute-unit charge prediction |
| Uptime | Recovery from preemption and interrupted job | Guarantee 24/7 Colab availability |

## Release / reporting template

For each claim, add a result row to progress.md:

| Date | Version/commit | Scenario & command | Environment | Expected | Observed | Result | Colab sessions / actual compute |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TBD | TBD | Q04 unchanged project diff | Mock/offline | 0 source files transferred | TBD | NOT_RUN | 0 / 0 |
| TBD | TBD | L2 remote T4 multi-module training | Live/approval pending | Verified artifacts, 0 orphan | TBD | NOT_RUN | explicit approval required |

**Do not set PASS from intended behavior.** Release G7 requires test evidence, reviewed limitations, and maintainer signoff.
