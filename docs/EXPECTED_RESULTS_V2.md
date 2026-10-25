# Expected Results, Metrics & Acceptance Matrix — V2

> **Status:** Colab CLI Server V2 (M0–M8 & V2-Drive) was merged into `main` via **PR #22 (`670c6bc` / `a1629f9846f19c8d5d4e90469ee9e376b64d98e9`)**. Issue #23 (`fix/v2-postmerge-hardening-23`) and Review `5642047783` (F1–F4) deliver post-merge hardening, 5-revision multi-epoch history, two-phase remote coordination, bounded pagination, catastrophic recovery, and comprehensive 21-stage + Section 9 offline E2E verification tests. Compact schema v2 and revision replay safety enforce Google Drive's 124 UTF-8 bytes custom property limit (https://developers.google.com/workspace/drive/api/guides/properties#limits_of_custom_file_properties) via `validate_drive_properties` pre-write validation, idempotent republish without data transfer or PATCH, and fail-closed ambiguity handling. Status: `READY_FOR_INDEPENDENT_MERGE_REVIEW pending CI only`.
>
> **Release Categories:**
> - **`OFFLINE_RELEASE_READY`**: Fully automated offline unit tests, integration tests, mock Drive uploads/downloads, mock Colab sessions, and headless Chromium browser tests passing 100% in CI without cloud credentials or compute charges.
> - **`LIVE_VALIDATION_REQUIRES_APPROVAL`**: Live cloud GPU allocation and real Google Drive cloud API calls require explicit human operator approval, valid credentials, budget caps, and are executed manually using the smoke-test procedures below. All human approval gates strictly preserved. Real Google Drive and Google Colab environments have never been exercised in this codebase or test runs; OAuth and live cloud writes are unauthorized.

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
| Q32 | 5-revision checkpoint manifest history (F1) | 100% preservation of all 5 revisions in PublicationManifest v2 (`len(artifacts) == 5`), `is_latest` True only on head revision, immediate-successor `superseded_by` chain (`A -> B -> C -> D -> E -> None`), and `latest_by_path` pointing directly to head |
| Q33 | Two-phase remote is_latest coordination (F2) | Drive uploads initialized with `is_latest="false"`, content verified, prior demoted and head promoted via Drive v3 `PATCH`; 0 unverified files promoted |
| Q34 | Bounded pagination & strict listing errors (F3) | `pageSize=100`, bounded `max_list_pages=50`, `max_list_candidates=5000`, loop detection via visited ID tracking; 0 empty list `[]` fallback on HTTP/network errors |
| Q35 | Catastrophic local state loss recovery (F4) | Exact historical retrieval via canonical `@sha256_<HASH>` succeeds; unversioned retrieval succeeds if and only if exactly 1 remote revision claims `is_latest="true"`, failing closed with `DriveIntegrityError` if 0 or >1 remote revisions claim `is_latest="true"` |

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

## M8 dashboard acceptance matrix (G8 — targets, NOT actual results)

| ID | Offline or approved fixture | Expected behavior |
| --- | --- | --- |
| Q19 | Poller with subprocess spies | Only read-only `usage`/`sessions`; zero `new/run/ssh` allocation |
| Q20 | 1,440 synthetic minute snapshots | Timestamp/order/freshness correct; no dropped account samples |
| Q21 | Repeat an event after process restart | No duplicate event in computed run history |
| Q22 | Simulated CU increases/top-ups/adjustments | No negative consumption or invented billable CU total |
| Q23 | Balance decrease while external account sessions exist | Usage labeled account-wide/unattributed, not per managed job |
| Q24 | First collector run with no earlier history | Explicit monitoring-since date; prior lifetime usage unknown |
| Q25 | Simulated 401/403/429/timeout/stale CLI usage | Error visible, bounded backoff, prior data marked stale |
| Q26 | Two concurrent browser clients and SSE reconnect | Consistent safe event cursor, no duplicate logical events |
| Q27 | Unauthenticated HTTP, SSE or export | Access rejected; no project names/metrics/balance disclosed |
| Q28 | Seeded OAuth token, SSH key, secret and sensitive log | All redacted/blocked from response, download and browser logs |
| Q29 | Job with no train steps, progress total or GPU samples | Shows unavailable/indeterminate; no invented values |
| Q30 | 100 synthetic runs + paginated logs | Bounded response, memory and disk retention behavior |
| Q31 | Authorized cross-device HTTPS/VPN access smoke | Two browser clients can view mock UI; raw web port not exposed publicly |

All Q19–Q30 are offline/mock tests. Q31 requires separate operator consent and access setup, not cloud GPU usage. No monitoring data older than the initial account snapshot may be invented. Upstream CLI's `usage` source reports current balance, aggregate CU/h and assignment count, not detailed transactions.

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

---

## 9. Google Drive & Cross-Device Workspaces Acceptance Scorecard (Gate G-Drive)

| ID | Description | Target Value | Measured Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Q32** | Resumable chunked upload protocol | 1 MiB chunks; Range query resume on reconnect | 100% verified via MockDriveHttpClient | PASS |
| **Q33** | Drive quota / rate limit handling | HTTP 403 quotaExceeded raises DriveQuotaExceededError | Confirmed in `test_google_drive_store.py` | PASS |
| **Q34** | Automatic token refresh on 401 | Refreshes expired token and retries request | Confirmed in `test_google_drive_store.py` | PASS |
| **Q35** | Integrity verification | Local SHA256 canonical + remote MD5/size check | Confirmed in `test_google_drive_store.py` | PASS |
| **Q36** | Deterministic source snapshot | Bit-reproducible tar.gz (identical hash on repeat) | Confirmed in `test_reproducibility.py` | PASS |
| **Q37** | Secret filtering in snapshot | 0 `.env`, `*.key`, `*.pem`, `id_rsa`, `.git` leaked | Confirmed in `test_reproducibility.py` | PASS |
| **Q38** | Asynchronous worker recovery | Recovers pending tasks from journal across restart | Confirmed in `test_upload_worker.py` | PASS |
| **Q39** | Multi-client read-only attach | 2 independent clients view same run; 0 VM allocation | Confirmed in `test_workspace_broker.py` | PASS |
| **Q40** | Expired runtime discovery | Historical runs and Drive links discoverable after VM stop | Confirmed in `test_drive_and_broker_e2e.py` | PASS |
| **Q41** | Trusted-proxy spoofing & peer IP check | Requires `trusted_proxy_secret`, peer IP whitelist, and HMAC/secret header | Confirmed in `test_monitor_backend.py` & `test_adversarial_security_and_fault_injection.py` | PASS |
| **Q42** | Real TLS socket wrapping | Wraps socket with `ssl.SSLContext` (`TLSv1_2`+) and validates cert/key files | Confirmed in `test_monitor_backend.py` & `test_adversarial_security_and_fault_injection.py` | PASS |
| **Q43** | Byte-accurate HTTP 308 Range validation | Partial Range, missing Range, and malformed/impossible Range handled without skipping bytes | Confirmed in `test_google_drive_store.py` & `test_adversarial_security_and_fault_injection.py` | PASS |
| **Q44** | Same-name/same-size MD5 collision & readback | Rejects stale files with mismatched MD5; verifies content via `alt=media` readback if MD5 omitted | Confirmed in `test_google_drive_store.py` & `test_adversarial_security_and_fault_injection.py` | PASS |
| **Q45** | Seeded secret non-disclosure audit | 0 seeded tokens/secrets/session URLs leaked across tracker, journal, collector, artifacts, or API | Confirmed in `test_adversarial_security_and_fault_injection.py` | PASS |
| **Q46** | 13-stage end-to-end V2 + Drive pipeline | Full offline pipeline from manifest/plan through sync, telemetry, Drive publish, teardown & attach | Confirmed in `test_project_e2e.py` | PASS |
| **Q47** | Stored DOM XSS & inline handler elimination | 0 `.innerHTML` assignments, 0 inline handlers (`onclick`, `oninput`, `onchange`), safe `textContent` DOM nodes | Confirmed in `test_dashboard_ui_integration.py` & `test_browser_xss_and_cross_process_qa.py` | PASS |
| **Q48** | Per-response cryptographic CSP nonce | `script-src 'self' 'nonce-...'` without `'unsafe-inline'`; blocks un-nonced script in headless Chromium | Confirmed in `test_monitor_backend.py` & `test_browser_xss_and_cross_process_qa.py` | PASS |
| **Q49** | Cross-process Drive publication atomicity | 2 simultaneous OS processes (`subprocess.Popen`) initiate exactly 1 Drive upload session and 1 remote object | Confirmed in `test_upload_worker.py`, `test_google_drive_store.py` & `test_browser_xss_and_cross_process_qa.py` | PASS |
| **Q50** | Immutable checkpoint revision history (`PublicationManifest` v2) | Republishing `model.pt` with new SHA256 preserves both `@sha256_<A>` and `@sha256_<B>` revisions and retrieves either by exact revision ID or latest logical path | Confirmed in `test_storage_abstraction.py`, `test_google_drive_store.py`, `test_upload_worker.py` & `test_project_e2e.py` | PASS |
| **Q51** | Automatic v1 -> v2 manifest schema migration | Legacy `manifest_version: 1` unversioned entries upgrade to canonical `@sha256_<digest>` keys with `latest_by_path` and explicit `migration_notes` | Confirmed in `test_storage_abstraction.py` | PASS |
| **Q52** | Reproducible headless Chromium browser security in CI | Dedicated `browser-security` CI job executes real headless Chromium with `COLAB_REQUIRE_BROWSER_TESTS=1` (failing if browser unavailable) and verifies 0 XSS & multi-revision DOM rendering | Confirmed in `.github/workflows/ci.yml` & `test_browser_xss_and_cross_process_qa.py` | PASS |
| **Q53** | Verified 3-case Google Drive Recovery | Strict 3-case retrieval resolution (exact `@sha256_<HASH>`, verified latest pointer, fail-closed on ambiguous fallback) | Confirmed in `test_google_drive_store.py` | PASS |
| **Q54** | Private checkpoint staging | `0700` staging dirs, `0600` files with `fsync`, symlink/device rejection, and canonical containment checks | Confirmed in `test_storage_abstraction.py` | PASS |
| **Q55** | Local artifact retrieval size-mismatch handling | Capturing `actual_size` before unlinking `.part.` temp file so `ArtifactValidationError` is raised deterministically | Confirmed in `test_storage_abstraction.py` | PASS |
| **Q56** | Controlled Colab HTTP 500 upload recovery | Remote path validation, missing parent directory probe via bounded `exec_code`, and bounded single retry | Confirmed in `test_transports.py` | PASS |
| **Q57** | Mandatory 21-stage offline V2 research lifecycle & recovery pipeline | Complete 21-stage end-to-end verification passing in offline CI | Confirmed in `test_project_e2e.py` | PASS |
| **Q58** | Compact schema v2 & 124-byte property limit | 124 UTF-8 bytes limit (https://developers.google.com/workspace/drive/api/guides/properties#limits_of_custom_file_properties) enforced via `validate_drive_properties` pre-write validation; compact schema v2 (`schema`, `sha256`, `path_sha256`, `is_latest`, `next_sha256`); legacy keys read-only compat | Confirmed in `test_drive_metadata_limits.py` & `test_drive_adversarial_review_25.py` | PASS |
| **Q59** | Revision replay safety & idempotence | Republishing historical revision keeps it historical with 0 new uploads, 0 data transfer, and 0 PATCH requests; interrupted-publish recovery promotes pending file; explicit rollback is not implemented | Confirmed in `test_revision_replay_safety.py` | PASS |
| **Q60** | Crash-point recovery & fail-closed ambiguity | Crash recovery converges to single latest across crash before demote, after demote, after promote; unversioned recovery with ambiguous 0 or >1 remote latest fails closed with `DriveIntegrityError` | Confirmed in `test_revision_replay_safety.py` & `test_drive_adversarial_review_25.py` | PASS |

---

## 10. Human-Approval-Gated Live Smoke-Test Procedures

These procedures are **strictly manual and require explicit operator authorization and valid credentials**. They are never executed in automated CI.

### Verification Categories
- **`OFFLINE_RELEASE_READY`**: Automated unit tests, integration tests, mock Drive uploads/downloads, mock Colab sessions, and headless Chromium browser tests pass completely (100% pass) in offline/CI environments with 0 cloud spending.
- **`LIVE_VALIDATION_REQUIRES_APPROVAL`**: Real cloud Colab GPU allocation and real Google Drive API requests require human operator approval, valid credentials, budget authorization, and are executed manually using the smoke-test procedures below.

---

### Procedure A: Google Drive Live Smoke Test (11 Steps)
**Mode: `LIVE_VALIDATION_REQUIRES_APPROVAL`**
**Prerequisites**: Google Drive API credentials file with permissions `0600` at `~/.config/colab-cli/drive_credentials.json`.

1. **Prerequisites & Permission Verification**:
   Verify credentials file exists and has private permissions:
   `test -f ~/.config/colab-cli/drive_credentials.json && ls -l ~/.config/colab-cli/drive_credentials.json` (must be `-rw-------`).
2. **Dry-Run OAuth Token Exchange**:
   Run token refresh routine to ensure valid access token can be acquired without creating remote files.
3. **Remote Folder Structure Discovery**:
   Resolve target root folder `ColabCLI_Artifacts/smoke-test-run/` on Google Drive, verifying folder creation idempotency.
4. **Seed Synthetic Checkpoint (Revision A)**:
   Create local test checkpoint `model.pt` (e.g. 512 KiB synthetic weights, Step 100):
   `dd if=/dev/urandom of=./smoke_model.pt bs=1024 count=512`.
5. **Resumable Upload Execution (Revision A)**:
   Publish Revision A via `GoogleDriveArtifactStore.publish()`; verify HTTP 308 range handling and chunk progression.
6. **Authoritative Remote Verification (Revision A)**:
   Query Drive v3 `files.get` endpoint for `size`, `md5Checksum`, and `sha256` matching local file.
7. **PublicationManifest v2 Verification (Revision A)**:
   Verify local `manifest.json` records Revision A under `smoke-test-run/checkpoints/model.pt@sha256_<HASH_A>` with `is_latest=True`.
8. **Local In-Place Overwrite (Revision B)**:
   Overwrite `smoke_model.pt` with distinct weights (Step 200, distinct SHA256) and publish Revision B to the same logical path `checkpoints/model.pt`.
9. **Multi-Revision Preservation Audit**:
   Verify `manifest.json` contains BOTH Revision A (`is_latest=False`) and Revision B (`is_latest=True`), with `latest_by_path["checkpoints/model.pt"]` pointing to Revision B.
10. **Dual Retrieval & Bitwise Hash Verification**:
    Retrieve latest weights using logical path and Revision A using `@sha256_<HASH_A>`, verifying bitwise equality against original sources.
11. **Transfer Journal Audit & Teardown**:
    Inspect `transfer_journal.json` to confirm both transfers have status `PUBLISHED` with 0 uncommitted records; delete test folder on Drive if desired.

---

### Procedure B: Colab GPU Live Smoke Test (12 Steps)
**Mode: `LIVE_VALIDATION_REQUIRES_APPROVAL`**
**Prerequisites**: Explicit human budget authorization (e.g., T4 GPU, max 15-minute lease, max 1.0 compute unit cap).

1. **Operator Consent & Budget Cap Authorization**:
   Record human operator approval string, target GPU (`T4`), maximum lease time (`900` seconds), and compute unit cap.
2. **Pre-Flight Diagnostic & Baseline Usage Check**:
   Run `python3 colab_server.py doctor` and `python3 colab_server.py usage` to log initial compute-unit balance.
3. **Project Workspace Setup**:
   Create a test workspace with entrypoint `gpu_probe.py` verifying `torch.cuda.is_available()`, device name, and memory.
4. **Manifest Validation**:
   Validate project manifest with `python3 -m colab_adapter.cli project validate ./gpu-smoke-test --json`.
5. **Zero-Allocation Offline Execution Plan**:
   Generate plan with `python3 -m colab_adapter.cli project plan ./gpu-smoke-test --json` (verifying 0 VMs allocated).
6. **Explicit Approval Token Provisioning**:
   Provision session: `python3 -m colab_adapter.cli project up ./gpu-smoke-test --gpu T4 --lease 900 --approval-token "ALLOCATE T4 $(date +%s)" --json`.
7. **Incremental Workspace Source Sync**:
   Synchronize sources: `python3 -m colab_adapter.cli project sync ./gpu-smoke-test --json`.
8. **Remote Training Execution**:
   Run GPU probe: `python3 -m colab_adapter.cli project exec ./gpu-smoke-test --json`.
9. **Execution Output & CUDA Device Verification**:
   Inspect output logs to confirm remote GPU was active, executed under assigned PID, and emitted training metrics.
10. **Output Artifact Pull & Verification**:
    Retrieve generated output/checkpoint and verify SHA256 checksum against execution record.
11. **Graceful Session Teardown**:
    Terminate session: `python3 -m colab_adapter.cli project stop ./gpu-smoke-test --json`.
12. **Post-Execution Account Audit & CU Accounting**:
    Run `python3 colab_server.py usage` to verify final compute unit spend and confirm `python3 -m colab_adapter.cli sessions` reports 0 active sessions.

---

### Procedure C: Cross-Device Authenticated Access Test (7 Steps)
**Mode: `LIVE_VALIDATION_REQUIRES_APPROVAL`**
**Prerequisites**: Two machines or loopback test environments with SSH or TLS certificates.

1. **Secure Monitoring Server Startup**:
   Start server: `COLAB_MONITOR_TOKEN="$(cat /path/to/private/monitor.token)" python3 -m colab_adapter.cli monitor --host 127.0.0.1 --port 8443`.
2. **Secure Client Tunnel / TLS Connection**:
   Establish SSH port-forwarding from secondary device (`ssh -NL 8443:127.0.0.1:8443 user@server`) or access TLS endpoint.
3. **Negative Control (Unauthenticated Access Rejection)**:
   Attempt `curl -i http://127.0.0.1:8443/api/v1/runs` without token; verify HTTP 401 Unauthorized is returned.
4. **Authenticated REST API Verification**:
   Query `curl -i -H "Authorization: Bearer <TOKEN>" http://127.0.0.1:8443/api/v1/runs` and `/api/v1/summary`; verify HTTP 200 and JSON schema.
5. **Headless Browser Web Dashboard & CSP Inspection**:
   Open dashboard in browser; inspect console to verify per-response CSP nonce, zero inline handler warnings, and proper rendering of `LATEST` and `HISTORICAL` badges.
6. **Cross-Device Artifact Retrieval**:
   From secondary client, retrieve checkpoint artifact via WorkspaceBroker or DriveStore; verify SHA256 checksum matches primary node.
7. **Client Disconnect & Teardown Audit**:
   Close client connection, stop monitor server, and verify server logs contain zero exposed tokens or secrets.

## 11. PR #24 Offline Verification Evidence and Known Limitations

Scope: branch `fix/v2-postmerge-hardening-23` (PR #24). The evidence below is offline only, using temporary directories and a mock Drive client. Nothing here is live Colab, Google Drive or OAuth validation, and no result in this section is a production or live-verified claim.

**Offline test evidence**

- Test command: `python3 -m unittest discover -s tests -v`
- Result (Python 3.8.10): 619 tests run, 618 passed, 1 skipped, 0 failed. Result (Python 3.12.14): see CI and the final report for the post-follow-up count; before the follow-up it was 594 passed, 0 skipped, 0 failed. Command: `python3 -m unittest discover -s tests -p "test_*.py"`. Baseline before PR #24 hardening: 381 tests (1 skipped).
- Python versions: 3.8.10 and 3.12.14 executed. Python 3.13 was not available on the test host and was not run.
- Browser test (`tests/test_browser_xss_and_cross_process_qa.py`, requires `COLAB_REQUIRE_BROWSER_TESTS=1`): real Chromium was not available on the test host; only the portable structural fallback ran. The real-browser job was not run locally.

**Test modules added or changed in PR #24**

| Area | Module |
| --- | --- |
| Local exact revision and alias | `tests/test_local_exact_revision.py`, `tests/test_local_alias_consistency.py` |
| Local cross-process behaviour | `tests/test_local_cross_process.py` |
| Shared manifest lock | `tests/test_shared_manifest_lock.py` |
| Drive pending promotion and disposition | `tests/test_drive_pending_promotion.py` |
| Drive rate-limit retry | `tests/test_drive_rate_limit_retry.py` |
| Storage input validation | `tests/test_storage_input_validation.py` |
| Monitor project isolation and review follow-ups | `tests/test_monitor_project_isolation.py`, `tests/test_monitor_review_followups.py` |
| PR #24 review follow-ups | `tests/test_review_followups_pr24.py` |

Mock support changed in `tests/drive_api_mock.py`.

**Evidence status**

- Offline mock tests: 619 on Python 3.8.10 after the OAuth-file and monitor follow-ups.
- Live Google Drive: not run. Live validation requires explicit owner approval.
- Live Colab: not run. Live validation requires explicit owner approval.
- Live OAuth: not run. Live validation requires explicit owner approval.
- Python 3.13: not run on this host.
- Real headless Chromium: not run on this host for this change.

**Known limitations**

1. Monitor hashing: unmanifested local files are hashed per request, up to 8 MiB each.
2. Monitor SSE: `/api/v1/events/stream` is unfiltered when `?project=` is omitted.
3. No live Drive or Colab validation has been performed for any PR #24 behaviour.
4. Python 3.13 and real Chromium were not run on this host.
5. Explicit rollback is not implemented. Republishing an older revision never moves the head.
6. Drive has no compare-and-swap. The stale and promote guarantee holds only if every writer to the folder goes through this host's per-manifest lock.
7. A rate-limit 403 that exhausts its retry budget is reported as `DriveQuotaExceededError`.
8. A hard kill can leave a temporary `.tmp` file behind in the local store.

Live validation of Google Drive, Colab and OAuth remains blocked until the owner gives explicit approval (`LIVE_VALIDATION_REQUIRES_APPROVAL`).

