# Live Test Evidence Record

Status: **LIVE RESULTS RECORDED IN SECTION 7 (2026-10-26); the table in Section 3 is the original template and is superseded by Section 7 for the tests listed there.** Only the offline baseline row has any content. Every other row is `NOT RUN` until a test is actually executed and its artifacts are recorded.

## 1. Header

| Field | Value |
|---|---|
| Repository | colab-cli-server |
| Base commit (tested) | live Drive, T4, guard and second-device monitor tests ran on `3a62c9b291f755e06533756eba590dcd4d3ea094` (main after PR #28); the template was originally written against `f4c9ee52c90312bde744c07614620250e7a44bc4`. PR #32 (monitor fixes) merged afterwards and was **not** retested live. Assessment commit: `94e3fbebed07fcdf2563e99c180f2d3244fe2af0` (post-merge CI run 40256494607: 3.8, 3.12, 3.13 and browser-security success) |
| Commit containing this record | the merge commit of the PR that contains this file (see `git log -- docs/live/LIVE_TEST_EVIDENCE.md`) |
| Record date (initial) | 2026-10-25 |
| Live test date(s) | 2026-10-26 (UTC) |
| Host OS | Ubuntu 20.04.6 LTS, Linux 5.4 x86_64 (headless) |
| Python version | 3.8.10 (host); remote runtime 3.13.16 |
| Colab CLI version | 0.7.5 (observed; see Section 4) |
| Repo CLI version | `python3 -m colab_adapter.cli` at the commits above |
| Reviewer(s) | independent merge reviews on PRs #24 and #28 (GitHub reviews); Opus 5.5 independent evidence assessment, 2026-10-26 |
| Owner signoff | Signed by the owner on 2026-10-26, in chat: "Correct, I sign", on the verdict `LIVE_SUCCESS_PATH_VERIFIED_WITH_LIMITATIONS` and the limitations in Section 7.11. The owner's cleanup items (delete both leftover Drive roots after inspecting them, delete the throwaway monitor token files, confirm no active Colab runtime) were **not confirmed done** at signing and remain pending. |
| Opus signoff | Independent read-only assessment 2026-10-26: `LIVE_SUCCESS_PATH_VERIFIED_WITH_LIMITATIONS` (not production readiness; not failure-path verification). Opus verified the saved evidence offline and did not contact any cloud service. |

## 2. Rules

1. **No secrets.** No OAuth tokens, refresh tokens, API keys, cookies, session cookies, passwords, or `Authorization` header values appear anywhere in this record or its artifacts.
2. **Redacted account.** Account identity is recorded as a non-reversible label (for example `acct-<short-hash>`), never as an email address or account ID.
3. **Sanitized output only.** Command output and logs are pasted only after redaction. Lines containing tokens, paths under a home directory that reveal identity, or remote URLs with embedded credentials are removed or replaced with `<redacted>`.
4. **No raw OAuth responses.** Token endpoint responses, credential files, and auth debug output are never copied in, even when redacted.
5. **Allowed:** file sizes, SHA-256 / MD5 hashes, remote file IDs, session IDs, revision IDs, timestamps, and resource counts.
6. **Paths:** only non-sensitive repository-relative paths or scratch-directory names. No absolute home paths.
7. **Each row is evidence for one test ID only.** A row may not be marked PASS by reference to another row.

## 3. Results Table

Legend: `NOT RUN` = no execution has happened. `BLOCKED` = a prerequisite (authorization, second device, approval, resource) is missing. Entries marked `<TO BE FILLED ...>` require the orchestrator or executing engineer to supply content; they must not be guessed.

Columns: Test ID | Git commit | Environment | Test type | Authorization | Expected result | Observed result | Result | Artifacts | Resources | Cleanup | Defects

| Test ID | Git commit | Environment | Test type | Authorization | Expected result | Observed result | Result | Artifacts | Resources | Cleanup | Defects |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Offline baseline | f4c9ee5 | Local host (see Section 1); no network, no Colab, no Drive | Offline | Not required (no cloud resources) | `python -m compileall` succeeds on the repo; `bash -n` succeeds on every shell script; GitHub Actions CI run 40196101226 passes; local full test suite passes with zero failures | Local compileall and `bash -n`: OK. CI run 40196101226: passed. Local full-suite result: `<TO BE FILLED BY ORCHESTRATOR>` | NOT RUN (partial: compileall, `bash -n` and CI recorded as OK; full suite pending) | `<TO BE FILLED: hashes of compileall log and CI log, sanitized>` | None | N/A | None known |
| Gate tests (Issue #25) | f4c9ee5 | `<TO BE FILLED>` | Offline | Not required unless a live step is included | `<TO BE FILLED BY ORCHESTRATOR: gate criteria as written in Issue #25; not restated in the mission text>` | `<TO BE FILLED>` | NOT RUN | `<TO BE FILLED>` | None expected | N/A | Issue #25 |
| D1 | f4c9ee5 | `<TO BE FILLED>` | live Drive | `<TO BE FILLED: owner's verbatim authorization phrase and scope, no secrets>` | First publication: remote file ID exists; remote size equals local size; remote MD5 equals local MD5; downloaded bytes SHA-256 equals original SHA-256; manifest and journal entries are correct | `<TO BE FILLED>` | NOT RUN | `<TO BE FILLED: file ID, sizes, hashes, manifest/journal paths>` | `<TO BE FILLED: Drive API calls count, duration; account-wide usage delta (observed, not per-job billing)>` | `<TO BE FILLED: remote file removed or retained, with reason; verified>` | `<TO BE FILLED>` |
| D2 | f4c9ee5 | `<TO BE FILLED>` | live Drive | `<TO BE FILLED>` | Three revisions created: A (historical), B (historical), C (latest); each has an immutable file ID and a full-length content hash recorded | `<TO BE FILLED>` | NOT RUN | `<TO BE FILLED: file IDs and full hashes for A, B, C>` | `<TO BE FILLED>` | `<TO BE FILLED>` | `<TO BE FILLED>` |
| D3 | f4c9ee5 | `<TO BE FILLED>` | live Drive | `<TO BE FILLED>` | Replay of revision A: no duplicate upload created (remote file count unchanged for A); A is recorded as historical; C remains latest | `<TO BE FILLED>` | NOT RUN | `<TO BE FILLED: remote listing before/after, sanitized>` | `<TO BE FILLED>` | `<TO BE FILLED>` | `<TO BE FILLED>` |
| D4 | f4c9ee5 | `<TO BE FILLED>` | live Drive | `<TO BE FILLED>` | Exact recovery of A, B and C (hash match for each); adapter restart does not change results; after local manifest is deleted in an isolated directory, latest (C) is recovered from remote state alone | `<TO BE FILLED>` | NOT RUN | `<TO BE FILLED: recovered hashes for A, B, C>` | `<TO BE FILLED>` | `<TO BE FILLED: isolated directory removed>` | `<TO BE FILLED>` |
| D5 | f4c9ee5 | `<TO BE FILLED>` | live Drive | `<TO BE FILLED>` | Property limits enforced; remote PATCH semantics as documented; pagination returns complete listing; Unicode and long-path names round-trip; errors reported with actionable messages; no token leakage in output or logs; retries are idempotent | `<TO BE FILLED>` | NOT RUN | `<TO BE FILLED: sanitized log hashes>` | `<TO BE FILLED>` | `<TO BE FILLED>` | `<TO BE FILLED>` |
| G1 | f4c9ee5 | Colab (runtime type GPU, T4 expected) | live Colab | `<TO BE FILLED>` | Exactly one managed T4 allocation created; record includes session ID, start timestamp, actual GPU model reported, and lease deadline | `<TO BE FILLED>` | NOT RUN | `<TO BE FILLED: session ID, timestamps, GPU model>` | `<TO BE FILLED: sessions created (expected 1), duration, account-wide usage delta>` | `<TO BE FILLED>` | `<TO BE FILLED>` |
| G2 | f4c9ee5 | Colab managed session from G1 | live Colab | `<TO BE FILLED>` | Remote PyTorch probe returns real output: torch version, `cuda.is_available()` true, device name reported by the runtime | `<TO BE FILLED: sanitized probe output>` | NOT RUN | `<TO BE FILLED: output hash>` | `<TO BE FILLED>` | `<TO BE FILLED>` | `<TO BE FILLED>` |
| G3 | f4c9ee5 | Colab managed session from G1 | live Colab | `<TO BE FILLED>` | Deterministic tiny training run completes; checkpoint SHA-256 is verified after retrieval and matches the value computed remotely | `<TO BE FILLED>` | NOT RUN | `<TO BE FILLED: checkpoint SHA-256 (remote and retrieved)>` | `<TO BE FILLED>` | `<TO BE FILLED>` | `<TO BE FILLED>` |
| G4 | f4c9ee5 | Colab managed session; Drive as sync target | live Colab + live Drive | `<TO BE FILLED>` | Initial sync succeeds; after modifying exactly one file, incremental sync uploads only that file; unchanged files are skipped; no unrelated remote file is deleted | `<TO BE FILLED>` | NOT RUN | `<TO BE FILLED: file list and hashes before/after>` | `<TO BE FILLED>` | `<TO BE FILLED>` | `<TO BE FILLED>` |
| G5 | f4c9ee5 | Colab managed session from G1 | live Colab | `<TO BE FILLED>` | Status command, reconnect, managed-session recognition, and monitoring continuity all work within the same lease (no new allocation during the check) | `<TO BE FILLED>` | NOT RUN | `<TO BE FILLED: sanitized status output>` | `<TO BE FILLED: sessions count unchanged during check>` | `<TO BE FILLED>` | `<TO BE FILLED>` |
| G6 | f4c9ee5 | Colab (G1 session and any others on account) | live Colab | `<TO BE FILLED>` | Teardown always attempted; the G1 session is no longer active afterwards; unmanaged sessions are untouched (count and IDs unchanged) | `<TO BE FILLED>` | NOT RUN | `<TO BE FILLED: session state before/after>` | `<TO BE FILLED: managed/unmanaged counts before/after>` | `<TO BE FILLED: teardown verified (yes/no, evidence)>` | `<TO BE FILLED>` |
| X1 (cross-device) | f4c9ee5 | Two clients: primary device and second device (second device required) | live Colab + live Drive | `<TO BE FILLED>` | Authenticated dashboard returns OK; unauthenticated request returns 401; run and project identity correct; telemetry matches the managed session; latest and historical revisions correct; a second client creates no VM; details persist after stop; no credentials in responses or logs; clean stop | `<TO BE FILLED>` | NOT RUN (BLOCKED if second device is not available) | `<TO BE FILLED: sanitized response codes and identity fields>` | `<TO BE FILLED: VM count before/after second client>` | `<TO BE FILLED>` | `<TO BE FILLED>` |
| I1 (integrated workflow) | f4c9ee5 | `<TO BE FILLED>` | live Drive + live Colab | `<TO BE FILLED>` | End-to-end flow: publish, recover, allocate, probe, train, sync, monitor, teardown, each step's result matching the expectations of its component test (D, G, X rows) | `<TO BE FILLED>` | NOT RUN | `<TO BE FILLED>` | `<TO BE FILLED>` | `<TO BE FILLED>` | `<TO BE FILLED>` |

## 4. Pre-flight Facts

Observed read-only before any live test, on 2026-10-25. These facts describe the starting state only. They are not proof that the cached credential is valid.

- Colab CLI 0.7.5 is installed and already authenticated via a cached OAuth credential. This is **not** proof that the credential is valid; a live call is required to confirm it.
- Managed sessions: 0. Unmanaged sessions: 0.
- Account balance: 186.75 compute units. Usage: 0.00/hr with 0 active assignments.
- Host disk: `/mnt/ssd` is 98% full (about 42 GB free).

Companion preflight and setup documents: see `docs/live/README.md`.

## 5. Final Verdict

Exactly one of the following verdicts applies. Record it here only after all rows are complete.

| Verdict | Meaning |
|---|---|
| LIVE_INTEGRATION_VERIFIED | All PASS criteria below are met. |
| PARTIAL_LIVE_VALIDATION | Some live tests passed; at least one required test is NOT RUN or FAIL and the cause is not an authorization or resource block. |
| BLOCKED_BY_AUTHORIZATION | A required owner approval or scope is missing. |
| BLOCKED_BY_TECHNICAL_ISSUE | A technical problem (defect, environment, service error) prevents a required test from completing. |
| CLOUD_RESOURCE_CLEANUP_REQUIRED | Cloud resources (Colab sessions, Drive files) remain that were not intended, or teardown could not be verified. |

Current verdict: `LIVE_SUCCESS_PATH_VERIFIED_WITH_LIMITATIONS` (2026-10-26; see Section 7.11). This is **not** `LIVE_INTEGRATION_VERIFIED`: PASS criteria 3 and 5 are met only for the success path, and the failure paths, a single continuous GPU-to-Drive run and a post-#32 live monitor retest are still missing.

### PASS criteria (all required for LIVE_INTEGRATION_VERIFIED)

1. Real Drive upload and retrieval verified (D1).
2. Exact immutable revisions recovered (D2, D3, D4).
3. Real Colab GPU execution completed (G1, G2, G3).
4. Session teardown verified (G6).
5. No unintended allocation (G1, G4, G5, G6, X1 counts).
6. Compute-usage evidence recorded (observed account-wide deltas in each Resources column).
7. Authenticated monitor working (X1).
8. Cross-device test completed (X1, requires second device).
9. Blocking regressions resolved (Gate tests, Offline baseline full suite).
10. Owner and Opus signoff recorded (Section 1).

## 6. Honesty Rules

1. Never mark a test PASS from an exit code alone. The expected assertion in the table must be checked against observed output.
2. Mock-only or fake-backend runs are labelled **Offline**, never live.
3. If a prerequisite is missing (second device, owner approval, quota, credential validity), the row is **BLOCKED**, not PASS or FAIL.
4. Resource figures are observed account-wide deltas, not per-job billing.
5. Unfilled `<TO BE FILLED>` cells are not evidence. Do not infer them.


## 7. Live results, 2026-10-26 (main `3a62c9b291f755e06533756eba590dcd4d3ea094`)

Environment: Ubuntu 20.04.6, Python 3.8.10, official Colab CLI 0.7.5 (`--auth=oauth2`), uv 0.12.13. Remote runtime: Python 3.13.16, Torch 2.11.0+cu130, Tesla T4. Offline context only (not live results): local suite on this commit 847 tests, OK, 3 skipped (torch-dependent); post-merge CI run 40213697055 all four jobs success. Private raw evidence (not in the repo, directories 0700, files 0600): Drive `~/colab-live-evidence/drive-20261026/`; GPU raw command outputs (`up_out.json`, `g4_sync.json`, `g4_probe.json`, `g5_train.json`, `g6_exec.json`, `model_g6.pt`) `~/colab-live-evidence/gpu-20261026/`. Rows marked *operator-observed* below were read from live command output during the session but no raw copy was saved; treat them as less strongly evidenced than rows backed by a saved file. Identifiers below are the harness-created test objects only; no account address, token or secret appears here.

### 7.1 Authorization (owner, in chat)
- Drive: `APPROVE DRIVE SMOKE TEST` (scope `drive.file`, 3 revisions, cap 150 requests, no deletion by the harness).
- GPU: `APPROVE 1 T4 SESSION — 900 SECONDS — 5 CU LIMIT`; the owner personally ran `project approve apr-7d94d6d7…` in their own terminal and typed the phrase (`approved_at` recorded 00:40:06 UTC). One `project up`.
- Cleanup: "Allow all" (owner) for the follow-up items; only the Drive test tree was trashed (see 7.4).

### 7.2 Live Google Drive (real API, real `GoogleDriveArtifactStore`)
| Test | Result | Observed |
|---|---|---|
| D1 first publish | PASS | remote id, size, MD5, re-downloaded bytes' SHA256, manifest and journal verified (19.1 s) |
| D2 B then C | PASS | 3 distinct remote ids, each full hash verified, history kept (20.1 s) |
| D3 replay A | PASS | no upload, no PATCH, A historical, C still latest (1.6 s) |
| D4 recovery | PASS | exact A, B, C on a fresh store; after deleting only the isolated test manifest and backup, latest recovered (8.7 s) |
| D5 compatibility | PASS | property byte limits, real PATCH, `pageSize=1` listing, a Unicode filename (`モデル-chkpt-ü.pt`) and a 180-byte filename published and listed, error reporting (20.7 s) |

API requests: 88 of the 150 cap (42 list, 12 get, 9 download, 9 patch, 6 folder create, 5 upload init, 5 upload chunk). Secret scan: 0 leaks. Revision hashes (full values in the private evidence): A `08c4676f…` 65,536 B; B `be1d62f8…` 196,608 B; C `d7784ed9…` 524,288 B. Not exercised live: real 403/429 rate-limit and retry behaviour (cannot be provoked safely).

### 7.3 Live Colab T4 (one allocation)
| Step | Result | Observed |
|---|---|---|
| Baseline | PASS (operator-observed) | 186.75 CU, 0.00/hr, Active assignments 0, no runtime, no managed session |
| `project up` (single) | PASS | `up_out.json`: session `colab-v2-live-smoke-ca8ca3ab`, state READY, T4, lease 900 s, `assignments_before` 0; exit 0 and about 19 s are wall-clock readings from the shell (operator-observed) |
| G3d guard and account | PASS (operator-observed) | guard alive, deadline about 888 s ahead, `assignments_before` 0, Active assignments 1, session listed by name, usage rate 1.07/hr |
| G4 sync + probe | PASS | 3 files / 13,258 bytes synced; `CUDA available: True`, Tesla T4, `GPU_PROBE_OK device=cuda:0 checksum=-674`, final `EXIT_CODE 0` |
| G5 training | PASS | `Device: cuda:0`, 60 metric events for steps 1 to 60, loss 9.74 to 0.088, run_started / checkpoint_saved / run_finished present, `DONE_OK`, final `EXIT_CODE 0`; about 10 s wall clock, of which 5.3 s is the telemetry span. `FINAL_LOSS` 0.0735 differs from the step-60 metric loss 0.0878 (reported as emitted; not investigated) |
| G6 checkpoint | PASS | remote `sha256sum`, sidecar, `DONE_OK` value and the locally decoded 3,111-byte file all `277d08f9c90aba66d3ebf234553a9720ec86342dfcb8fae320cf3f0ea6fdec9f` |
| Teardown | PASS (operator-observed) | `project stop` exit 0 with `stopped: true`; real `sessions`: none; real `usage`: Active assignments 0; registry TERMINATED, `orphan_possible: false`, no lease-guard process left. `verification: strong` is defined in `project_sessions.py` as: the session is not listed by name AND the account-wide Active-assignments count is at or below the recorded baseline. It is an account-wide check, not proof about this specific runtime. |

Resources: one session, about 2 minutes. Account-wide balance 186.75 to 186.74 CU. **Not reconciled:** about 2 minutes at the observed 1.07 CU/hr predicts roughly 0.036 CU, while the display moved by 0.01 (rounding and/or usage lag). This is an account-wide display delta, not per-job billing, and no cost claim is made beyond 'well under the 5 CU limit'. No GPU run was started outside the plan.

### 7.4 Cleanup of the Drive test data
Authorized by the owner. The root folder could not be trashed with the narrow token: Drive answered 403 `appNotAuthorizedToChild` even after every item inside it was trashed. All 10 items under the root (5 files and the 5 nested folders `projects`, `colab-live-smoke`, `runs`, `run-001`, `checkpoints`) were moved to Trash (reversible); the root folder `colab-live-smoke-20261026T003700Z-ea2fb8` remains and must be removed by the owner in the Drive web UI. **Possible hidden child:** the refusal means a child outside this token's grant may exist (for example created from the web UI or by another app). The `drive.file` token cannot list such objects, so the owner should open the folder in the web UI and inspect it before emptying Trash. The harness cleanup plan lists the 5 files and the root but not the 5 nested folder ids; that is a documentation gap to fix. A broader-scope token was deliberately not used. Finding: with a `drive.file` token, trashing the harness root folder fails; the harness's cleanup plan should say so.

### 7.5 Not run / unverified
- Guard deadline drill (G7): see Section 7.8 (run later the same day and passed on the success path).
- Failure paths on real Colab (nonzero `new`, orphan runtime, `[?]` line, "not found" stop): not exercised; they remain offline-verified only.
- Cross-device monitoring: see Section 7.9 (run later the same day and passed).
- Integrated workflow: only the narrower step in Section 7.10 was run (publish from the saved copy after teardown); a single continuous GPU-to-Drive run was not.
- V1 `run`/`submit_job` and `colab_server.py run`: not covered and not safe-by-design for live use (known gap in PR #28).
- Provisioning under load, long-running exec output limits, and usage-count lag were not characterized.

### 7.6 Verdict at this point
`PARTIAL_LIVE_VALIDATION`. Verified live (success paths only): Drive D1-D5, one bounded T4 session with sync, GPU probe, training, checkpoint hash match and a verified teardown; usage accounting is not reconciled. Mandatory scenarios still missing for `LIVE_INTEGRATION_VERIFIED`: guard drill (G7), cross-device monitoring, integrated workflow, owner and Opus sign-off.

### 7.7 Real-provider compatibility: confirmed vs not
Confirmed live (success paths): `colab new`/`up`, `usage` parsing of Active assignments (0 to 1 to 0) and the rate, `sessions` showing the named session and then none, `stop` exit 0, `exec -f` with the final `EXIT_CODE` marker and a roughly 4 KB base64 transfer, workspace sync. Not confirmed: the `[?]` unnamed-runtime line (the regex has never matched a real line; `unnamed_before` 0 only means zero matches), the exit code and exact text of the empty `sessions` listing, upstream 'not found' on `stop`, ResourceExhausted, nonzero `new`, orphan handling, large exec output, Drive 403/429 retries, the guard's deadline stop.

### 7.8 G7 guard deadline drill, 2026-10-26 (second and last allocation)
Authorization: `APPROVE 1 T4 SESSION — 300 SECONDS — 1 CU LIMIT — GUARD DRILL` (owner); the owner ran `project approve apr-2c473fbb…` in their own terminal (`approved_at` 00:53:04 UTC). Baseline immediately before: 186.74 CU, Active assignments 0, no runtime listed, no managed session. One `project up` with `--lease-seconds 300`; session `colab-v2-live-smoke-9289d488` READY at about 00:53:33. No workload was run. No manual stop was issued.

| Check | Result | Observed |
|---|---|---|
| Guard alive before the deadline | PASS | `status`: `guard_alive` true (pid 469999), deadline 00:58:27, `assignments_before` 0; real `usage` Active assignments 1 (1.07 CU/hr); real `sessions` listed the session by name |
| Guard stops the session at the deadline | PASS | guard log: `stop attempt 1 verified` at 00:58:31 (4 s after the deadline); outcome `STOPPED_VERIFIED`, guard exit 0 |
| Independent verification | PASS | my own real `sessions` after the stop: "No active sessions found on server"; real `usage`: Active assignments 0, 0.00/hr; registry TERMINATED, `verification: strong` (active assignments 0 <= baseline 0), `orphan_possible: false`; no lease-guard process left |

Cost: 186.74 to 186.69 CU (about 0.05), within the 1 CU limit; about 5 minutes at 1.07 CU/hr predicts roughly 0.09, so accounting remains not reconciled (account-wide display, not per-job billing). Raw outputs, including the guard's own log: `~/colab-live-evidence/g7-20261026/` (0700 directory, 0600 files).

What this establishes: on real Colab, the host-side lease guard stopped exactly its own session at the lease deadline and the removal was confirmed by the provider's own `sessions` and `usage` output (success path, empty account otherwise). What it does not establish: behaviour when the guard is killed or the host reboots, repeated stop failures, a `not found` stop, orphan or `[?]` handling, or any guarantee that compute stops being billed; the lease remains a best-effort deadline, not a billing cap.

Updated verdict: `PARTIAL_LIVE_VALIDATION`. Still missing for `LIVE_INTEGRATION_VERIFIED`: cross-device monitoring (BLOCKED, needs a second device or SSH tunnel), the integrated workflow (checkpoint from the GPU run published to Drive), the failure paths on real Colab, and owner and Opus sign-off.

### 7.9 X1 cross-device monitoring, 2026-10-26
Setup: read-only monitor started on the host bound to `127.0.0.1:8080` only (a launcher calling the same `start_monitor_server`, with an isolated private telemetry ledger and artifacts directory; the host managed-session registry was read as usual). Data: the real G5 run (63 telemetry events from `g5_train.json`) as project `live-smoke`, run `run-g5-20261026`, plus the verified 3,111-byte checkpoint published to a local artifact store (sha256 `277d08f9…fdec9f`, one revision). Bearer token from a 0600 file via the environment (never on a command line or in chat). The owner's second device (a laptop) reached the monitor only through `ssh -L 8080:127.0.0.1:8080`; no public exposure, no TLS, no non-loopback binding.

| Check | Result | Observed |
|---|---|---|
| Second-device checker (`scripts/live/monitor_check.py`) | PASS (operator-reported) | `overall: PASS`, 13 of 13 checks (a1 a2 a3 b c1 c2 c3 d1 d2 d3 e1 e2 f), no failed IDs. The second device's evidence file stays on the owner's machine; no raw copy exists in the private directory |
| Host-side loopback run of the same checker (does not count as cross-device) | PASS | 13 of 13, `evidence_host_loopback.json` (private directory) |
| Unauthenticated request | PASS | HTTP 401 (observed on the host) |
| Dashboard in the second device's browser | PASS (owner-observed) | project `live-smoke`, run `run-g5-20261026`, 60 steps, loss chart visible in the Run Inspector, one checkpoint marked latest, no mutating controls |
| Second client created no VM | PASS | host-side real `sessions` (none) and `usage` (Active assignments 0, balance 186.69, unchanged) saved in `host_*_after.txt` / `host_*_final.txt`; no `colab new/run/ssh` or `project up` process seen |
| No secrets in responses or monitor log | PASS | checker e1/e2 on both runs; host grep of the monitor log for token/secret patterns and the token value: 0 matches |
| Monitor stopped | PASS | process gone, port 8080 closed |

Findings (issue #31; fixed afterwards by PR #32, merged and NOT retested live): the account-usage collector is never started (`poll_once` has no caller), so the panel shows DEGRADED / Samples 0 / Never / CU balance Unavailable; "Active Assignments: 0" is shown when the value is null; Drive status shows PENDING when no publication is recorded; the session table lists all 202 TERMINATED sessions without pagination; the run table shows the last of two step-60 loss events (0.0878 then `FINAL_LOSS` 0.0735).

Limits: this validates the SSH-tunnel path and the monitor's authentication and read-only behaviour from a second device. It does not validate public exposure, TLS or a reverse proxy, display of more than one checkpoint revision, or the live collector. The data was seeded from the recorded G5 output rather than streamed by a running job, so "telemetry matches the executed workload" is verified against a recorded run.

Updated verdict: `PARTIAL_LIVE_VALIDATION`. Verified live: Drive D1-D5, one bounded T4 session (sync, probe, training, checkpoint hash match, verified manual stop), the guard deadline stop (success path) and second-device monitoring. Still missing for `LIVE_INTEGRATION_VERIFIED`: the integrated workflow (GPU checkpoint published to Drive in one flow), failure paths on real Colab (nonzero `new`, orphan or `[?]`, 'not found' stop, Drive 403/429), and owner and Opus sign-off.

### 7.10 I1 integrated step: GPU checkpoint published to Drive, 2026-10-26
Scope (stated narrowly): this joins two separately verified live pieces. The 3,111-byte checkpoint produced on the real T4 in the G5 run and verified at four points in G6 (sha256 `277d08f9…fdec9f`) was published through the real `GoogleDriveArtifactStore` with the dedicated `drive.file` token into a new disposable root folder `colab-live-integrated-20261026T012445Z` (project `live-smoke`, run `run-g5-20261026`, path `checkpoints/model.pt`). It is NOT a single continuous GPU-to-Drive run: the GPU session had already been stopped and the file was published from its saved local copy in a later step. Authorization: the owner chose "Both, publish first" in reply to the question of which flow to run next (a Drive write; no GPU allocation).

| Check | Result | Observed |
|---|---|---|
| Source is the G6-verified file | PASS | script asserts sha256 equals `277d08f9…fdec9f` before publishing |
| Publish | PASS | remote file id recorded, size 3,111, sha256 equal, artifact id carries the full sha256, `is_latest` true |
| Round trip on a fresh adapter instance | PASS | downloaded bytes sha256 equals `277d08f9…fdec9f` (3,111 bytes) |
| Replay | PASS | second `publish` returned the same remote file id; the API counts show 1 upload init and 1 upload chunk in total |
| Secrets | PASS | no access, refresh or client-secret value appears in the output |

API requests: 32 (19 list, 6 folder create, 2 get, 2 download, 1 upload init, 1 upload chunk, 1 patch), cap 60 for this step. Evidence: `~/colab-live-evidence/integrated-20261026/evidence.json` (0600). No deletion was performed; the new root folder and file remain in Drive for the owner to remove in the web UI (the same `drive.file` limitation on trashing a root folder was seen earlier).

Updated verdict: `PARTIAL_LIVE_VALIDATION`. Verified live: Drive D1-D5, one bounded T4 session, the guard deadline stop (success path), second-device monitoring, and publication of the real GPU checkpoint to Drive with a verified round trip. Still missing for `LIVE_INTEGRATION_VERIFIED`: failure paths on real Colab (nonzero `new`, orphan or `[?]`, 'not found' stop, Drive 403/429), a single continuous GPU-to-Drive workflow, and owner and Opus sign-off.

### 7.11 Final independent assessment and sign-off, 2026-10-26
Assessor: Opus 5.5, read-only and offline, at `94e3fbe` (no cloud contact, no credential files read). Verdict: **`LIVE_SUCCESS_PATH_VERIFIED_WITH_LIMITATIONS`**. Nothing in the saved evidence contradicts this record. This verdict is not production readiness and not failure-path verification.

Evidence classes:
- *Independently saved and recomputed (high confidence):* Drive D1-D5 (47 checks all ok, 88 requests); T4 `up` output; G4 probe and sync; G5 training (63 telemetry events: 60 metric steps 1-60, plus start, checkpoint and finish events); G6 checkpoint (sha256 `277d08f9…fdec9f` recomputed from `model_g6.pt`, decoded payload byte-identical, matches remote `sha256sum`, sidecar, telemetry and `DONE_OK`); G7 guard stop (deadline 00:58:27.8, verified stop logged 00:58:31, outcome `STOPPED_VERIFIED`); X1 host-side loopback run (13/13); I1 publish and replay (one `upload_init`, one `upload_chunk`, same remote id).
- *Operator-observed or operator-reported (medium confidence, no raw copy):* the T4 baseline and G3d readings, the T4 teardown output, the second device's checker result (13/13) and its dashboard view, and "no manual stop was issued during G7" (inferred from files and code; cannot be fully excluded from the files alone).
- *Untested:* all real-provider failure paths.

Limitations that remain open (all non-production-ready conditions): (1) real-provider failure paths never exercised live - nonzero `colab new`, the `[?]` unnamed-runtime line (the regex has never matched a real line), upstream "not found" on `stop`, ResourceExhausted, orphan recovery, Drive 403/429 retries; (2) usage accounting does not reconcile (T4 run: display moved 0.01 CU against about 0.036 predicted; G7: 0.05 against about 0.09) - these are account-wide display deltas, not billing evidence; (3) the lease guard is host-side best effort (no survival of reboot, `kill -9` of the guard or OAuth loss; depends on the shared lock and the real `sessions` output format) and only one success-path drill on an otherwise empty account was run; (4) I1 is not one continuous run; (5) PR #32 changed the monitor (poller, Unavailable display, Drive status, session table) after the second-device test and has not been retested live; (6) the Drive hash values in D1-D5 are recorded but the synthetic source bytes were not saved, so they cannot be recomputed; the I1 cloud copy cannot be re-checked from the host without a new cloud read.

Blocking for any full or production-readiness claim: items (1) and (4), and the pending cleanup below. Non-blocking for the success-path verdict: items (2), (3) as disclosed, (5), the operator-only rows, and the permission-wording note (`publish_g6.py` and `launch_monitor.py` are mode 0644 and some seeded monitor subdirectories 0755 inside the 0700 evidence directory; harmless, but not all files in it are 0600).

Cleanup status at signing (from files and owner instruction only; cloud state not inspected): pending owner actions - inspect and delete the Drive roots `colab-live-smoke-20261026T003700Z-ea2fb8` (its 10 children are in Trash; the root could not be trashed with the narrow token, and a child outside the token's grant may exist) and `colab-live-integrated-20261026T012445Z`; delete the throwaway monitor token (`~/.config/colab-monitor/owner_token` on the host and the copy on the second device); remove the leftover temp directory `/tmp/g6-pub-jphfjk1r` on the host; confirm in the Colab UI that no runtime is active. No Colab session was left running according to the final host-side readings (Active assignments 0).

Owner sign-off checklist: (1) Drive roots inspected and deleted - pending; (2) monitor token files deleted - pending; (3) Colab UI shows no runtime and a balance near 186.69 - pending; (4) stale documentation corrected - done in this record, README and the GPU plan; (5) owner verdict signed - done (Section 1); (6) failure paths, a continuous run and a post-#32 monitor retest recorded as pending future work - done above.

### 7.12 Post-#32 read-only monitor retest, 2026-10-26 (host loopback only)
Scope: a read-only retest of the monitor after PR #32 (usage poller, *Unavailable* display, Drive status, session table). No GPU, no allocation, no Drive write, no OAuth change. The monitor ran on the host bound to `127.0.0.1:8080` with a throwaway bearer token (deleted afterwards), `poll_interval` 15 s, the same seeded G5 data as Section 7.9, and the real usage poller calling only the read-only `colab --auth=oauth2 usage`. This is a host-loopback test: it does **not** repeat the second-device check, and the browser rendering of the new UI was **not** looked at by a human or a real browser in this step.

| Check | Result | Observed |
|---|---|---|
| Clean baseline before start | PASS | real `usage`: 186.69 CU, 0.00/hr, Active assignments 0; real `sessions`: none |
| Before the first poll | PASS | `/api/v1/summary`: balance `null`, active_assignments `null` (not 0), `is_stale` true |
| Poller populates data | PASS | after one interval: balance 186.69, rate 0.0, active_assignments 0, `is_stale` false; `/compute/samples` held 4 samples (status OK, source `colab usage`) at roughly 17 s spacing; values equal the real `colab usage` output read at the same time |
| Full checker on loopback | PASS | `monitor_check.py` overall PASS (`evidence_host_loopback.json`); not cross-device evidence |
| Served page | PASS | HTTP 200; 0 `.innerHTML` uses and 0 inline `on*` handlers in the served HTML; strings *Unavailable* and *NOT RECORDED* and a terminated-sessions toggle present (string-level check only) |
| Run list | PASS | `drive_publication_status` is `NOT RECORDED` (was `PENDING`), latest checkpoint `model_g6.pt` state `known`, 1 checkpoint |
| Secrets | PASS | grep of the monitor log for token and secret patterns and the token value: 0 matches |
| Clean shutdown | PASS | process killed by PID; port 8080 closed; no `colab` or launcher process remained 20 s later (the poller stopped); no allocation command appeared; final real `sessions`: none; `usage`: Active assignments 0, 186.69 CU unchanged |

Raw files: `~/colab-live-evidence/monitor-retest-20261026/` (0700 directory; summaries, evidence JSON, served page, baseline and final snapshots). Not exercised: poller backoff after real `colab usage` failures (offline-tested only), behaviour with a live running session, the Unavailable path against a failing provider, the rendered session-table toggle in a real browser, and any second-device view of the new UI. The first poll comes one interval after start, so a freshly started monitor shows *Unavailable* until then (observed).

Effect on the earlier limitation list: limitation (5) in Section 7.11 (PR #32 not retested live) is **partly** resolved - the poller and the summary data path are now confirmed live on loopback; the rendered UI and the second-device view remain unverified live. The overall verdict is unchanged: `LIVE_SUCCESS_PATH_VERIFIED_WITH_LIMITATIONS`.

