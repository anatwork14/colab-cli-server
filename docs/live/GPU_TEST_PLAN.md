# Bounded live GPU test plan (live-smoke)

Status: **DRAFT, NOT APPROVED.** No Colab allocation may happen until the owner approves this plan in writing (see [Approval](#4-approval)). This document was prepared offline. Nothing in it has been executed against Colab.

**What the 900 s lease is, and is not.** The lease is a *deadline the host tries to enforce*, not a billing cap and not a guarantee. `project up` starts a detached, session-scoped **lease guard** (`colab_adapter/lease_guard.py`) that, at the deadline, runs `colab --auth=oauth2 stop -s <SID>` for that one session and verifies with `colab sessions` that it is gone (bounded retries; then it flags `NEEDS_RECONCILIATION`). The guard does **not** survive a host reboot, `kill -9`, or loss of OAuth, its verification parses `sessions` output whose real format is **not yet verified against live Colab**, and the lease clock starts at `up` (it includes provisioning time). The operator must always check `sessions` and `usage` themselves after the test. The backstop for a dead guard is `project reconcile` (which auto-stops expired managed sessions) run by the operator.

Project fixtures: `examples/live-smoke/` (manifest `colab-project.json`, `gpu_probe.py`, `train_tiny.py`, `workload_manifest.json`). Offline tests: `tests/test_live_gpu_fixtures.py`.

## 1. Bounds

| Bound | Value |
|---|---|
| Accelerator | 1 x T4 |
| Allocations (`project up` / `colab new`) | at most 1, for the whole plan; one single-use approval = at most one allocation; a failed or timed-out `up` burns the approval and any retry needs a new owner approval |
| Lease | `--lease-seconds 900`: approval maximum and host-side guard deadline (best effort, session-scoped; not a billing cap; verify with `sessions` and `usage`) |
| Sessions touched | only the one session created by G3c |
| Remote workload | `gpu_probe.py` and `train_tiny.py` only; no dataset download, no pip install |
| Training size | 60 steps by default, hard maximum 200 (enforced by the script) |
| Keepalive / polling loops | none; status checks are manual and capped at 3 calls |

## 2. Expected runtime

Steps are deliberately tiny. These are estimates, not measurements; the first live run will replace them.

| Phase | Estimate |
|---|---|
| G1 offline checks (local) | under 1 min |
| G2 baseline usage and sessions | about 10 s |
| G3 approval steps and provisioning (`project up`) | 1 to 5 min (unknown; provisioning dominates; the 900 s lease clock starts when `up` is called) |
| G4 sync and GPU probe | about 1 min |
| G5 tiny training (60 steps) | under 1 min including torch import |
| G6 checkpoint retrieval and hash compare | about 1 min |
| Teardown and final usage | about 1 min |
| **Total wall clock** | **about 6 to 12 min**, within the 900 s lease only if provisioning is fast |

Abort if the remaining lease falls below 300 s before G5 (see abort criteria). The guard will stop the session at the deadline regardless of what is running.

## 3. Cost measurement

CU consumption is measured as an **account-wide delta** from `usage` before G2 and after teardown. It is **not per-job billing**. Other activity on the account during the window is included in the delta, so record the window times and do not attribute the full delta to this test. Do not invent a per-job figure.

## 4. Approval

The approval gate from Issue #25 is implemented (single-use, expiring, plan-bound approval records; see `docs/SECURITY.md`). The owner gives two separate approvals:

1. **Plan approval (chat):** the owner replies with the exact text `APPROVE 1 T4 SESSION — 900 SECONDS — <CU limit>` after reading the current balance. This authorizes the agent to *request* an approval record. It does not by itself allow an allocation.
2. **Operator approval (terminal):** the owner personally runs `project approve <approval-id>` in their own terminal and types the printed phrase (`ALLOCATE T4 <code> 900s`). The phrase is read from `/dev/tty`; piped or non-interactive input is refused. The record is valid for 15 minutes and is consumed by the first `project up` that uses it.

Who runs the paid `project up` (G3c): the **owner (recommended; strongest boundary)** or the agent immediately after the owner's terminal approval. The gate is an interlock, not a security boundary against a process with your OS account (it can read the HMAC key, forge records, or call `colab` directly); do not rely on it against an untrusted agent.

- Owner plan approval (name, date, GPU, lease, max allocations, CU limit): ____________________
- Operator who runs `project approve` and G3c: ____________________
- Optional guard drill chosen (see G7)? yes / no

## 5. Commands

All commands run from the repository root of the integrated branch that contains the approval gate and the lease guard. Every flag below must be re-checked against `python3 -m colab_adapter.cli project <sub> --help` immediately before the run (the approval commands and `--timeout` are new). Do not add flags that are not listed here.

Placeholders:
- `<SID>`: the `session_id` value in the JSON printed by G3c.
- `<APPROVAL_ID>`: the `approval_id` value printed by G3a (form `apr-` plus 32 hex).
- `<SCRATCH>`: a new empty directory for retrieved files (for example, a directory under the agent scratchpad).

### G1: offline gate (local, zero allocation)

```
python3 -m colab_adapter.cli project validate --manifest examples/live-smoke/colab-project.json --json
python3 -m colab_adapter.cli project plan --manifest examples/live-smoke/colab-project.json --json
python3 -m unittest tests/test_live_gpu_fixtures.py -v
```

Expected: `validate` returns `"valid": true`; `plan` reports `"gpu": "T4"`, 3 files (`gpu_probe.py`, `train_tiny.py`, `workload_manifest.json`) and `remote_root` `/content/colab-workspaces/live-smoke`; the unit tests pass (torch-only tests may be skipped locally).
Must not: call any `colab` executable.

### G2: baseline (read-only)

```
python3 -m colab_adapter.cli usage
python3 -m colab_adapter.cli sessions
```

Record the output verbatim (CU balance and the list of existing sessions). Note any managed `live-smoke` session that already exists. Do not touch any session in this step.

### G3: approval and the single allocation

Run G3a, G3b, G3c back to back; the approval expires after 15 minutes and the workspace digest must not change in between (do not edit files under `examples/live-smoke/` between G3a and G3c).

**G3a (agent or operator, no cloud contact):**

```
python3 -m colab_adapter.cli project approve-request --manifest examples/live-smoke/colab-project.json --lease-seconds 900 --json
```

Expected: JSON with `approval_id`, `status: "pending"`, `max_lease_seconds: 900.0`, a `phrase` of the form `ALLOCATE T4 <code> 900s` and an `expires_at` about 15 minutes ahead. No `colab` command runs.

**G3b (operator only, own terminal, interactive):**

```
python3 -m colab_adapter.cli project approve <APPROVAL_ID>
```

The operator reads the summary and plan digest, then types the phrase exactly. Expected: the record becomes `approved`. If the terminal is not interactive the command refuses with `non_tty`; do not work around it.

**G3c (the only allocation; operator recommended):**

```
python3 -m colab_adapter.cli project up --manifest examples/live-smoke/colab-project.json --lease-seconds 900 --approval-id <APPROVAL_ID> --json
```

Expected: exactly one new managed session whose `session_id` is `<SID>` and whose `state` is `READY`; the lease guard has been started (exit code 0). Exit code 3 means the approval was refused (zero `colab new` calls); exit code 4 means the global cap of one active managed session is reached (the approval is not consumed). If `up` fails after the approval was consumed (allocation error, timeout, or the guard could not start, in which case `up` stops the new session itself), the approval is **spent**: run teardown, report, and do **not** request a new approval without the owner's fresh instruction. Never run `project up` a second time.

**G3d (verify the guard before any workload; read-only):**

```
python3 -m colab_adapter.cli project status --session-id <SID> --json
```

Expected: the guard is reported alive with a deadline about 900 s after the allocation and a positive remaining time. If the guard is not alive, run teardown immediately (a session without a live guard has no deadline enforcement).

### G4: sync and GPU probe

```
python3 -m colab_adapter.cli project sync --manifest examples/live-smoke/colab-project.json --session-id <SID> --json
python3 -m colab_adapter.cli project exec --session-id <SID> --working-dir /content/colab-workspaces/live-smoke --timeout 120 --json "python3 gpu_probe.py --require-gpu"
```

Expected output markers (all must appear in the captured output):
- `CUDA available: True`
- `GPU name:` with a T4 device name
- `GPU_PROBE_OK device=cuda:0 checksum=<integer>`

The checksum is exact (small-integer inputs), so the same integer must be reported on every run. Record it. Exit code 2 means CUDA is missing: abort.

### G5: tiny training run

```
python3 -m colab_adapter.cli project exec --session-id <SID> --working-dir /content/colab-workspaces/live-smoke --timeout 300 --json "python3 train_tiny.py --steps 60 --require-gpu"
```

Expected output:
- `CUDA available: True` and `Device: cuda:0`
- 60 lines beginning with `TELEMETRY {` whose `event_type` is `metric`, with steps 1 to 60 and a `loss` metric that is finite
- one `checkpoint_saved` telemetry line and one `run_finished` telemetry line
- `FINAL_LOSS <number>`
- `DONE_OK steps=60 device=cuda:0 final_loss=<number> checkpoint=artifacts/checkpoints/model.pt sha256=<64 hex>`

Verify the DONE_OK line; do not treat exit code 0 alone as success (AGENTS.md rule 9). The final loss should be clearly lower than the first step's loss. A final loss that is not lower is recorded as a finding, not as a failure of the plan.

### G6: checkpoint retrieval and hash compare

```
python3 -m colab_adapter.cli project exec --session-id <SID> --working-dir /content/colab-workspaces/live-smoke --timeout 120 --json "sha256sum artifacts/checkpoints/model.pt && cat artifacts/checkpoints/model.pt.sha256 && base64 -w0 artifacts/checkpoints/model.pt"
```

Procedure:
1. Save the JSON output to `<SCRATCH>/g6_exec.json`.
2. Extract the base64 payload and decode it to `<SCRATCH>/model.pt`. If the output is truncated or the payload is incomplete, mark G6 as failed and do not retry with a new allocation.
3. Compute the local SHA-256 of `<SCRATCH>/model.pt`.
4. Compare three values: the local digest, the remote `sha256sum` digest, and the digest in the `.sha256` sidecar. All three must be identical and equal the `sha256=` value printed by G5.
5. Optional: load the file with `torch.load(..., map_location="cpu")` on a machine that has torch, and confirm the keys `model_state_dict`, `step` (60) and `seed` (1234). Skip this if torch is not installed locally.

Assumption to verify at execution: `project exec` runs its argument through a shell, so `&&` and `base64` work. If it does not, stop and record the limitation; do not switch to another allocation.

## 6. Teardown (mandatory)

Teardown runs after G6 and also after any abort, including a failed G3c, G4 or G5. It is done by the operator or agent by hand; the lease guard is only the backstop.

```
python3 -m colab_adapter.cli project status --session-id <SID> --json
python3 -m colab_adapter.cli project stop --session-id <SID> --json
python3 -m colab_adapter.cli sessions
python3 -m colab_adapter.cli usage
```

- `status` records the remaining lease and the guard state for the report (at most 3 status calls in the whole plan).
- `stop` targets only `<SID>` (and terminates that session's guard after a successful stop).
- After `stop`, `sessions` must not list `<SID>`. If it still does, retry `project stop --session-id <SID> --json` once and report the result; if it is still listed, tell the owner immediately so they can stop it from the Colab UI. Do not stop any other session.
- If `up` failed in a way that may have left a runtime (timeout, nonzero exit after `new` started), `sessions` is the source of truth: report any unexpected session id to the owner; do not stop sessions this plan did not create.
- If the guard is not alive and the session cannot be stopped, the operator runs `python3 -m colab_adapter.cli project reconcile --json` (this auto-stops expired *managed* sessions only) and then re-checks `sessions`.

### G7 (optional, owner decides): guard drill

To verify the guard against real Colab instead of trusting the offline fake, the owner may choose to skip the manual `stop` in teardown and let the guard stop the session at the deadline (the session stays allocated until at most the 900 s lease ends, costing a few extra minutes of T4 time). The operator watches `project status --session-id <SID> --json` (at most 3 calls) and, **if the session is still listed 120 s after the deadline**, performs the manual `stop` above. PASS only if `sessions` no longer lists `<SID>` and the registry shows `guard_outcome: STOPPED_VERIFIED`. This is the only step that validates the real `colab sessions` output format that the guard parses. If the owner does not choose G7, the guard's live behaviour stays **UNVERIFIED** and must be reported that way.

## 7. Usage before and after

1. Run `usage` in G2 and save the output as the before-snapshot.
2. Run `usage` at the end of teardown and save it as the after-snapshot.
3. Report the difference as the account-wide CU delta with the start and end timestamps. Do not describe it as per-job billing.

## 8. Expected verified results

| Check | Source | Pass condition |
|---|---|---|
| GPU is a real CUDA device | G4 | `GPU_PROBE_OK` line present with `device=cuda:0` |
| Training ran on GPU | G5 | `Device: cuda:0` and `DONE_OK ... device=cuda:0` |
| 60 telemetry steps | G5 | 60 `metric` events, steps 1 to 60, finite loss |
| Checkpoint written | G5 | `checkpoint_saved` event and `sha256=` in DONE_OK |
| Checkpoint integrity | G6 | local digest == remote `sha256sum` == sidecar == DONE_OK value |
| Clean teardown | Section 6 | `<SID>` absent from `sessions` after `stop` |

Any single failed check means the plan has not passed. Record the failure as a finding.

## 9. Abort criteria

Abort the plan, run teardown (section 6), and stop further steps when any of these happens:

- `project up` yields more than one session, or a session that was not created by G3c.
- `project up` fails after consuming the approval (do not request a second approval on your own; report to the owner).
- G3d shows the lease guard is not alive.
- `project up` is run again, or any step would allocate a second accelerator, or an approval is requested or used for a second allocation without a fresh owner instruction.
- G4 exits 2 or the `GPU_PROBE_OK` marker is missing.
- G5 exits with a non-zero code, or the `DONE_OK` marker is missing.
- The remaining lease from `status` is below 300 s before G5.
- A command outside this plan is needed to proceed.
- Any output shows a credential, token or account identifier. Stop and report it; do not paste it into the report.
- `sessions` lists sessions that the plan did not create and the owner has not approved touching them.

## 10. Prohibited actions

During this plan, do **not**:

- allocate more than one accelerator, or run `project up` more than once;
- run `colab new`, `colab run`, `colab ssh`, or any `colab` command directly;
- stop, attach to, or execute on any session not created by G3c (AGENTS.md Q08 invariant);
- run `project reconcile` except as the documented backstop in section 6 (it can stop expired managed sessions);
- forge, copy, or script the approval phrase, pipe input into `project approve`, set any bypass, or call `colab new` directly;
- start any keepalive, ping, or polling loop, or use `sleep` loops to hold a session open;
- download datasets, install packages with pip, or fetch models from the network;
- start the monitor server with any mutating endpoint, or expose it to a network;
- use `--no-reuse` (with a live session it always fails the global cap) or any flag not listed in section 5;
- publish or commit captured outputs that contain credentials or account details.

## 11. Limitations

- Nothing in this plan has been run against Colab. Provisioning time, lease behaviour during provisioning, exec output size, whether `project exec` runs through a shell, and the real `colab sessions`/`colab stop` output and exit codes are unverified.
- The lease guard is best effort: no billing cap; not robust to host reboot, `kill -9`, OAuth loss, or a changed `sessions` output format; its verification is only validated live if G7 is chosen. Operator checks of `sessions` and `usage` are mandatory regardless.
- The approval gate is an interlock, not a security boundary against a process running as the same OS user.
- CU cost is an account-wide delta, not per-job billing.
- The torch-dependent code paths were not executed locally (torch is not installed on the offline host); they are verified only by the remote run.
- It is unverified whether `-s` calls to `exec`/`upload` can auto-create a missing session (the installed upstream CLI exits with an error for a missing session in the paths inspected, but this is re-verified during G4 by checking `sessions` for unexpected ids).
