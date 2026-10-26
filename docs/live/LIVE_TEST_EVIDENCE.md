# Live Test Evidence Record

Status: **LIVE RESULTS RECORDED IN SECTION 7 (2026-10-26); the table in Section 3 is the original template and is superseded by Section 7 for the tests listed there.** Only the offline baseline row has any content. Every other row is `NOT RUN` until a test is actually executed and its artifacts are recorded.

## 1. Header

| Field | Value |
|---|---|
| Repository | colab-cli-server |
| Base commit (tested) | `f4c9ee52c90312bde744c07614620250e7a44bc4` (main, "Merge PR #24: V2 storage, recovery, security and monitor hardening") |
| Commit containing this record | `<TO BE FILLED: SHA of the docs(live) commit that adds this file>` |
| Record date (initial) | 2026-10-25 |
| Live test date(s) | `<TO BE FILLED>` |
| Host OS | `<TO BE FILLED: uname -a output, sanitized>` |
| Python version | `<TO BE FILLED>` |
| Colab CLI version | 0.7.5 (observed; see Section 4) |
| Repo CLI version | `<TO BE FILLED>` |
| Reviewer(s) | `<TO BE FILLED: independent engineer>` |
| Owner signoff | `<TO BE FILLED>` |
| Opus signoff | `<TO BE FILLED>` |

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

Current verdict: `<TO BE FILLED>` (no live test has been executed at the time of this template).

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

Environment: Ubuntu 20.04.6, Python 3.8.10, official Colab CLI 0.7.5 (`--auth=oauth2`), uv 0.12.13. Remote runtime: Python 3.13.16, Torch 2.11.0+cu130, Tesla T4. Local suite on this commit: 847 tests, OK, 3 skipped (torch-dependent); post-merge CI run 40213697055: 3.8, 3.12, 3.13 and browser-security all success. Private raw evidence (not in the repo): `~/colab-live-evidence/drive-20261026/` (mode 0600). Identifiers below are the harness-created test objects only; no account address, token or secret appears here.

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
| D5 compatibility | PASS | property byte limits, real PATCH, `pageSize=1` listing, Unicode and long names, error reporting (20.7 s) |

API requests: 88 of the 150 cap (42 list, 12 get, 9 download, 9 patch, 6 folder create, 5 upload init, 5 upload chunk). Secret scan: 0 leaks. Revision hashes (full values in the private evidence): A `08c4676f…` 65,536 B; B `be1d62f8…` 196,608 B; C `d7784ed9…` 524,288 B. Not exercised live: real 403/429 rate-limit and retry behaviour (cannot be provoked safely).

### 7.3 Live Colab T4 (one allocation)
| Step | Result | Observed |
|---|---|---|
| Baseline | PASS | 186.75 CU, 0.00/hr, Active assignments 0, no runtime, no managed session |
| `project up` (single) | PASS | exit 0, session `colab-v2-live-smoke-ca8ca3ab` READY in about 19 s, T4, lease 900 s |
| G3d guard and account | PASS | guard alive, deadline about 888 s ahead, `assignments_before` 0, Active assignments 1, session listed by name, usage rate 1.07/hr |
| G4 sync + probe | PASS | 3 files / 13,258 bytes synced; `CUDA available: True`, Tesla T4, `GPU_PROBE_OK device=cuda:0 checksum=-674`, final `EXIT_CODE 0` |
| G5 training | PASS | `Device: cuda:0`, 60 metric events for steps 1 to 60, loss 9.74 to 0.088, run_started / checkpoint_saved / run_finished present, `DONE_OK`, final `EXIT_CODE 0` (about 10 s) |
| G6 checkpoint | PASS | remote `sha256sum`, sidecar, `DONE_OK` value and the locally decoded 3,111-byte file all `277d08f9c90aba66d3ebf234553a9720ec86342dfcb8fae320cf3f0ea6fdec9f` |
| Teardown | PASS | `project stop` exit 0; real `sessions`: none; `usage`: Active assignments 0; registry TERMINATED, `verification: strong`, `orphan_possible: false`; no lease-guard process left |

Resources: one session, about 2 minutes. Account-wide balance 186.75 to 186.74 CU (display rounds to 0.01; an account-wide delta, not per-job billing). No GPU run was started outside the plan.

### 7.4 Cleanup of the Drive test data
Authorized by the owner. The root folder could not be trashed with the narrow token: Drive answered 403 `appNotAuthorizedToChild` even after every item inside it was trashed. All 11 items under the root (5 files, 5 sub-folders, plus the `projects` tree) were moved to Trash (reversible); the empty root folder `colab-live-smoke-20261026T003700Z-ea2fb8` remains and must be removed by the owner in the Drive web UI. A broader-scope token was deliberately not used. Finding: with a `drive.file` token, trashing the harness root folder fails; the harness's cleanup plan should say so.

### 7.5 Not run / unverified
- Guard deadline drill (optional G7): not run; the guard's own deadline stop has never been observed on real Colab. Only the manual verified stop was exercised.
- Failure paths on real Colab (nonzero `new`, orphan runtime, `[?]` line, "not found" stop): not exercised; they remain offline-verified only.
- Cross-device monitoring: not run (needs a second device or SSH tunnel) - BLOCKED.
- Integrated workflow (checkpoint to Drive after GPU): not run.
- V1 `run`/`submit_job` and `colab_server.py run`: not covered and not safe-by-design for live use (known gap in PR #28).
- Provisioning under load, long-running exec output limits, and usage-count lag were not characterized.

### 7.6 Verdict at this point
`PARTIAL_LIVE_VALIDATION`. Verified live: Drive D1-D5, one bounded T4 session with sync, GPU probe, training, checkpoint hash match, verified teardown, and clean accounting. Mandatory scenarios still missing for `LIVE_INTEGRATION_VERIFIED`: guard drill (G7), cross-device monitoring, integrated workflow, owner and Opus sign-off.
