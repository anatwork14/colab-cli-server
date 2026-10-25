# Bounded live GPU test plan (live-smoke)

Status: **DRAFT, NOT APPROVED.** No Colab allocation may happen until the owner approves this plan in writing (see [Approval](#approval)). This document was prepared offline. Nothing in it has been executed against Colab.

Project fixtures: `examples/live-smoke/` (manifest `colab-project.json`, `gpu_probe.py`, `train_tiny.py`, `workload_manifest.json`). Offline tests: `tests/test_live_gpu_fixtures.py`.

## 1. Bounds

| Bound | Value |
|---|---|
| Accelerator | 1 x T4 |
| Allocations (`project up` / `colab new`) | at most 1, for the whole plan |
| Lease | `--lease-seconds 900` (hard cap 900 s) |
| Sessions touched | only the one session created by G3 |
| Remote workload | `gpu_probe.py` and `train_tiny.py` only; no dataset download, no pip install |
| Training size | 60 steps by default, hard maximum 200 (enforced by the script) |
| Keepalive / polling loops | none; status checks are manual and capped at 3 calls |

## 2. Expected runtime

Steps are deliberately tiny. These are estimates, not measurements; the first live run will replace them.

| Phase | Estimate |
|---|---|
| G1 offline checks (local) | under 1 min |
| G2 baseline usage and sessions | about 10 s |
| G3 provisioning (`project up`) | 1 to 5 min (unknown; this dominates) |
| G4 sync and GPU probe | about 1 min |
| G5 tiny training (60 steps) | under 1 min including torch import |
| G6 checkpoint retrieval and hash compare | about 1 min |
| Teardown and final usage | about 1 min |
| **Total wall clock** | **about 6 to 12 min**, within the 900 s lease only if provisioning is fast |

Abort if the remaining lease falls below 300 s before G5 (see abort criteria).

## 3. Cost measurement

CU consumption is measured as an **account-wide delta** from `usage` before G2 and after teardown. It is **not per-job billing**. Other activity on the account during the window is included in the delta, so record the window times and do not attribute the full delta to this test. Do not invent a per-job figure.

## 4. Approval

<APPROVAL-GATE-FIX PENDING>

`project up` currently provisions paid compute without the approval gate that Issue #25 will add. Until that fix lands, the owner must approve this plan explicitly and an operator must run G3 by hand from a terminal. Agents must not run G3 autonomously.

- Owner approval (name, date, GPU, lease, max allocations): ____________________
- Operator who runs G3-T: ____________________

## 5. Commands

All commands run from the repository root at commit `f4c9ee5` or later. Every flag below was checked against `python3 -m colab_adapter.cli project <sub> --help` at `f4c9ee5`. Do not add flags that are not listed here.

Placeholders:
- `<SID>`: the session id returned by G3 (read its key from the JSON output; do not guess the key name).
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

### G3: the single allocation <APPROVAL-GATE-FIX PENDING>

```
python3 -m colab_adapter.cli project up --manifest examples/live-smoke/colab-project.json --lease-seconds 900 --json
```

Expected: exactly one new managed session; its id is `<SID>`. If the output shows a session id that was not present in G2 and is not managed by this project, stop (see abort criteria). Do not run `project up` a second time for any reason.

### G4: sync and GPU probe

```
python3 -m colab_adapter.cli project sync --manifest examples/live-smoke/colab-project.json --session-id <SID> --json
python3 -m colab_adapter.cli project exec --session-id <SID> --working-dir /content/colab-workspaces/live-smoke --json "python3 gpu_probe.py --require-gpu"
```

Expected output markers (all must appear in the captured output):
- `CUDA available: True`
- `GPU name:` with a T4 device name
- `GPU_PROBE_OK device=cuda:0 checksum=<integer>`

The checksum is exact (small-integer inputs), so the same integer must be reported on every run. Record it. Exit code 2 means CUDA is missing: abort.

### G5: tiny training run

```
python3 -m colab_adapter.cli project exec --session-id <SID> --working-dir /content/colab-workspaces/live-smoke --json "python3 train_tiny.py --steps 60 --require-gpu"
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
python3 -m colab_adapter.cli project exec --session-id <SID> --working-dir /content/colab-workspaces/live-smoke --json "sha256sum artifacts/checkpoints/model.pt && cat artifacts/checkpoints/model.pt.sha256 && base64 -w0 artifacts/checkpoints/model.pt"
```

Procedure:
1. Save the JSON output to `<SCRATCH>/g6_exec.json`.
2. Extract the base64 payload and decode it to `<SCRATCH>/model.pt`. If the output is truncated or the payload is incomplete, mark G6 as failed and do not retry with a new allocation.
3. Compute the local SHA-256 of `<SCRATCH>/model.pt`.
4. Compare three values: the local digest, the remote `sha256sum` digest, and the digest in the `.sha256` sidecar. All three must be identical and equal the `sha256=` value printed by G5.
5. Optional: load the file with `torch.load(..., map_location="cpu")` on a machine that has torch, and confirm the keys `model_state_dict`, `step` (60) and `seed` (1234). Skip this if torch is not installed locally.

Assumption to verify at execution: `project exec` runs its argument through a shell, so `&&` and `base64` work. If it does not, stop and record the limitation; do not switch to another allocation.

## 6. Teardown (mandatory)

Teardown runs after G6 and also after any abort, including a failed G4 or G5.

```
python3 -m colab_adapter.cli project status --session-id <SID> --json
python3 -m colab_adapter.cli project stop --session-id <SID> --json
python3 -m colab_adapter.cli sessions
python3 -m colab_adapter.cli usage
```

- `status` records the remaining lease for the report (at most 3 status calls in the whole plan).
- `stop` targets only `<SID>`.
- After `stop`, `sessions` must not list `<SID>`. If it still does, retry `project stop --session-id <SID> --json` once and report the result. Do not stop any other session.

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

- `project up` yields more than one session, or a session that was not created by G3.
- `project up` is run again, or any step would allocate a second accelerator.
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
- stop, attach to, or execute on any session not created by G3 (AGENTS.md Q08 invariant);
- run `project reconcile` (it can stop expired sessions; this plan does not need it);
- start any keepalive, ping, or polling loop, or use `sleep` loops to hold a session open;
- download datasets, install packages with pip, or fetch models from the network;
- start the monitor server with any mutating endpoint, or expose it to a network;
- use `--no-reuse` or any flag not listed in section 5;
- publish or commit captured outputs that contain credentials or account details.

## 11. Limitations

- Nothing in this plan has been run against Colab. Provisioning time, lease behaviour during provisioning and exec output size are unverified.
- `project up` will change with the approval-gate fix (Issue #25); all G3 and related lines must be re-checked against the `--help` output of the fixed version before use.
- CU cost is an account-wide delta, not per-job billing.
- The torch-dependent code paths were not executed locally (torch is not installed on the offline host); they are verified only by the remote run.
