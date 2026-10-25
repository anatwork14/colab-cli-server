# Google Drive Durable Artifact Storage (V2)

## Overview

In Google Colab, virtual machine disk storage is strictly ephemeral: runtimes can terminate unexpectedly due to timeouts, preemption, or user stops. High-frequency training writes directly to a mounted Google Drive folder are discouraged due to I/O throttling, rate limits, and metadata latency.

Colab V2 solves this by establishing a provider-independent artifact storage architecture (`ArtifactStore`). Complete checkpoints and reproducibility snapshots are verified locally on Ubuntu SSD storage before being asynchronously published to private Google Drive storage via chunked, resumable Google Drive API v3 transfers.

## Architecture

```text
Colab GPU Runtime
       |
       | (Training checkpoint generated on VM disk)
       v
Official CLI Download to Ubuntu Server
       |
       | (Atomic staging + multi-tier integrity checks: SHA256 + ZIP/PyTorch)
       v
Verified Local SSD Artifact Store (`./artifacts/`)
       |
       | (Asynchronous queue worker + persistent transfer journal)
       v
Google Drive Resumable Uploader (API v3)
       |
       | (Range query recovery, exponential backoff, remote MD5/size check)
       v
Private Google Drive Hierarchy (`Colab-Research/`)
```

## Google Drive Folder Hierarchy

Artifacts are organized deterministically under a dedicated private root folder:

```text
Colab-Research/
  projects/
    <project-id>/
      runs/
        <run-id>/
          manifest.json                # Publication manifest with all digests
          source_snapshot.tar.gz       # Bit-reproducible code archive
          environment.json             # Runtime platform and dependencies
          run_manifest.json            # Complete experiment metadata & git SHA
          checkpoints/
            step-001000.pt             # Verified immutable model weights
            step-002000.pt
          metrics.jsonl                # Time-series training loss and metrics
          training.log                 # Standard output & execution logs
          outputs/                     # Visualizations and exported tensors
```

All folder IDs are resolved and cached explicitly in memory and disk rather than relying solely on directory names.

## Resumable Upload Protocol & Lifecycle States

Transfers proceed through explicit, monotonic lifecycle states:

```text
LOCAL_STAGED
    ↓
LOCAL_VERIFIED
    ↓
UPLOAD_PENDING
    ↓
UPLOADING  <---- (Retries with backoff / Range query resume) ----+
    ↓                                                           |
DRIVE_VERIFIED                                           RETRY_PENDING
    ↓                                                           |
PUBLISHED                                                UPLOAD_FAILED
```

1. **LOCAL_STAGED**: File is downloaded to a temporary PID-tagged staging path.
2. **LOCAL_VERIFIED**: Multi-tier integrity verification passes (Tier 1 size/existence, Tier 2 SHA256 hash, Tier 3 ZIP/PyTorch deserialization check).
3. **UPLOAD_PENDING**: Transfer is registered in the persistent transfer journal (`journal.json`).
4. **UPLOADING**: Upload proceeds in 1 MiB chunks (multiples of 256 KiB) via Google Drive v3 resumable upload sessions.
5. **DRIVE_VERIFIED**: On HTTP 200/201, remote file ID is captured, remote size is verified, and remote `md5Checksum` is validated against local MD5.
6. **PUBLISHED**: Publication manifest is updated and committed atomically.

### Interrupted Upload Recovery & In-Process Retries

If a network interruption occurs during `UPLOADING`:
- The upload session URL and byte count are preserved in `journal.json` protected by cross-process file locks (`fcntl.flock`).
- The worker sends a status query (`Content-Range: bytes */<total_size>`).
- The Drive API responds with `308 Resume Incomplete` and the `Range: bytes=0-<N>` header. The store strictly validates `Range` syntax (`^bytes=0-(\d+)$`) and bounds (`0 < N + 1 <= start_byte + chunk_len <= total_size`), setting the next chunk offset to `<N> + 1`.
- If a `308` response omits the `Range` header, the store never assumes the chunk was committed; it reconciles the committed offset via `_query_upload_progress` and re-sends unacknowledged bytes.
- If an ambiguous network exception occurs (socket drop or timeout), the worker reconciles committed bytes with the server before backing off.
- The background `ArtifactUploadWorker` includes in-flight queue deduplication and an in-process retry scheduler that automatically retries `RETRY_PENDING` transfers using exponential backoff with jitter while the service is running, without requiring process restarts.
- If an upload session expires (`HTTP 404/410`), the store clears the stale session URL, checks if a file matching both `expected_size` and `expected_md5` was already committed in the target folder, or safely initiates a fresh upload session from byte `0`.
- During download (`retrieve`), files are streamed in 64 KiB chunks with bounded memory to a private temporary file (`0600`), incrementally hashing SHA256/MD5 and performing atomic rename only after full verification passes.

## Single-Host Cross-Process Artifact Publication Atomicity Guarantee

When multiple local processes (e.g., concurrent training jobs, background upload workers, or CLI publishing commands) interact with the artifact store simultaneously, the system provides strict single-host publication atomicity:

### 1. OS-Level File Locking (`fcntl.flock`)
- **Per-Artifact Locks**: Every transfer operation acquires an exclusive advisory OS-level lock on `.transfer_locks/<transfer_id>.lock` before reading or mutating transfer state.
- **Manifest Locks**: `PublicationManifest` operations acquire `.manifest.lock` ensuring concurrent workers do not interleave JSON writes or produce torn manifest states.

### 2. Durable Leases & Fencing
- Each `TransferRecord` tracked in `TransferJournal` carries durable lease attributes:
  - `owner_pid`: Process ID of the active worker.
  - `lease_token`: Unique cryptographic UUID identifying the lease instance.
  - `lease_expires_at`: Monotonic lease deadline preventing indefinite hangs.
  - `fencing_epoch`: Monotonically increasing counter incremented on every lease acquisition.
- **Dead-PID Lease Reclamation**: If a worker crashes or is terminated by `SIGKILL`, a surviving worker verifies PID liveness using `os.kill(owner_pid, 0)`. If the process no longer exists, the lease is safely reclaimed and the fencing epoch is bumped, preventing zombie workers from resuming stale transfers.

### 3. Content-Addressed Revision Safety
- **Deduplication**: If two processes attempt to publish identical content (`sha256` checksum and `size_bytes`), the second transfer immediately detects the existing published record, reuses the verified `remote_file_id`, and avoids duplicate Drive allocations or transfers.
- **Revision Safety**: If a file at the same filename is updated with new content (different `sha256`), the system invalidates stale session URLs and initiates a verified new remote upload. Stale hashes or obsolete remote file IDs are never returned.

### 4. Scope Boundary
- **Single-Host Guarantee**: Atomicity and lease guarantees are guaranteed on a single host via the local POSIX filesystem (`fcntl.flock`).
- **Distributed Multi-Host Writers Out of Scope**: Multi-host distributed writers sharing an uncoordinated NFS, SMB, or distributed mount without an external distributed lock service (e.g. etcd, ZooKeeper, or Consul) are explicitly out of scope for the local workspace journal.

---

## Canonical Revision Identity & PublicationManifest V2 Schema

In real-world deep learning workflows, multiple checkpoints frequently share the same logical path (e.g. `checkpoints/model.pt` overwritten each epoch). Google Drive Artifact Storage V2 manages multi-revision experiments transparently:

### 1. Canonical Revision Key Format & Immediate-Successor Chain
Each immutable revision is indexed by its content-addressed versioned ID:
```text
<project_id>/<run_id>/<logical_path>@sha256_<full_64_character_sha256>
```
Example:
```text
vision_v2/run-001/checkpoints/model.pt@sha256_3b4f5d133bb7c9c693999b2620703353cba10983fc4aa8162ac1e456fdffa766
```

#### Immediate-Successor `superseded_by` Chain
To track strict lineage without ambiguity, each historical revision records its immediate successor:
- `is_latest`: Boolean flag indicating whether this is the current active revision (`True` only for newest revision).
- `superseded_by`: Canonical artifact ID of the immediate next revision, or `None` if `is_latest=True`.

When 5 successive revisions (`A, B, C, D, E`) are published to the same logical path (`checkpoints/model.pt`):
```text
Revision A (step 100): is_latest=False, superseded_by -> Revision B
Revision B (step 200): is_latest=False, superseded_by -> Revision C
Revision C (step 300): is_latest=False, superseded_by -> Revision D
Revision D (step 400): is_latest=False, superseded_by -> Revision E
Revision E (step 500): is_latest=True,  superseded_by -> None
Chain: A -> B -> C -> D -> E -> None
```

#### Programmatic Accessors on `PublicationManifest`
`PublicationManifest` provides typed accessor methods for inspecting the latest revision of any logical path:
- `manifest.get_latest_artifact(logical_path: str) -> Optional[StoredArtifact]`: Returns the full `StoredArtifact` for the newest active revision of `logical_path`.
- `manifest.get_latest_revision_id(logical_path: str) -> Optional[str]`: Returns the revision string (e.g. `sha256_<digest>`) of the newest active revision.
- `manifest.get_latest_sha256(logical_path: str) -> Optional[str]`: Returns the 64-character lowercase SHA256 hex digest of the newest active revision.

### 2. PublicationManifest V2 Schema
The publication manifest on disk (`manifest.json`) adheres to version 2 schema:
```json
{
  "manifest_version": 2,
  "project_id": "vision_v2",
  "run_id": "run-001",
  "status": "PUBLISHED",
  "latest_by_path": {
    "checkpoints/model.pt": "vision_v2/run-001/checkpoints/model.pt@sha256_2222..."
  },
  "artifacts": {
    "vision_v2/run-001/checkpoints/model.pt@sha256_1111...": {
      "artifact_id": "vision_v2/run-001/checkpoints/model.pt@sha256_1111...",
      "storage_provider": "google_drive",
      "logical_path": "checkpoints/model.pt",
      "revision_id": "sha256_111111111111",
      "size_bytes": 1024,
      "sha256": "1111...",
      "remote_file_id": "drive-file-rev1",
      "created_at": "2026-10-24T10:00:00Z",
      "is_latest": false,
      "superseded_by": "vision_v2/run-001/checkpoints/model.pt@sha256_2222...",
      "metadata": {
        "step": 100,
        "epoch": 1,
        "superseded_by": "vision_v2/run-001/checkpoints/model.pt@sha256_2222..."
      }
    },
    "vision_v2/run-001/checkpoints/model.pt@sha256_2222...": {
      "artifact_id": "vision_v2/run-001/checkpoints/model.pt@sha256_2222...",
      "storage_provider": "google_drive",
      "logical_path": "checkpoints/model.pt",
      "revision_id": "sha256_222222222222",
      "size_bytes": 2048,
      "sha256": "2222...",
      "remote_file_id": "drive-file-rev2",
      "created_at": "2026-10-24T12:00:00Z",
      "is_latest": true,
      "superseded_by": null,
      "metadata": {
        "step": 200,
        "epoch": 2
      }
    }
  },
  "migration_notes": [
    "2026-10-24T12:00:00Z: Migrated manifest from version 1 to 2; indexed latest_by_path."
  ]
}
```

### 3. Automatic V1 to V2 Migration
When loading older `manifest_version: 1` files where `artifacts` keys lacked `@sha256_`, `PublicationManifest.load()` automatically:
- Preserves all historical records.
- Keys records by their canonical `@sha256_<full_sha256>` identity.
- Populates `latest_by_path` based on timestamps and step metadata.
- Updates `manifest_version` to `2` and appends an entry to `migration_notes`.

### 4. Retrieving Latest vs. Specific Historical Revisions
- **Retrieving Latest**:
  ```python
  # Resolves logical path "checkpoints/model.pt" via latest_by_path
  store.retrieve(
      artifact_id="vision_v2/run-001/checkpoints/model.pt",
      destination="./latest_model.pt",
  )
  ```
- **Retrieving Specific Historical Revision**:
  ```python
  # Retrieves exact historical revision from Epoch 1
  store.retrieve(
      artifact_id="vision_v2/run-001/checkpoints/model.pt@sha256_1111111111111111111111111111111111111111111111111111111111111111",
      destination="./epoch1_model.pt",
  )
  ```

### 5. Cross-Process Locking & Staging Protection
- All manifest modifications acquire `.manifest.lock` via `fcntl.flock`.
- Staged revisions are written to temporary files named `*.sha256_<hash[:12]>` or `*.tmp.*` before verification and atomic replacement.
- Disk scanners skip internal `*.sha256_*` staging files while preserving all historical records registered in `manifest.json`.

### 6. Real-Browser CI Security Testing
- Headless Chromium/Playwright test suites execute in CI against loopback `MonitorServer` instances.
- Verifies that `#artifacts-table` displays `LATEST` (green badge) for current revisions and `HISTORICAL` (neutral badge) for superseded revisions, with zero `.innerHTML` and strict CSP nonce enforcement.

### 7. Two-Phase Remote `is_latest` Coordination Protocol (Drive v3 `PATCH`)
To prevent race conditions, split-brain `is_latest` states, or premature promotion of unverified files, `GoogleDriveArtifactStore` coordinates remote revision state via a strict two-phase protocol:

```text
Phase 1: Resumable Upload with is_latest="false"
  Local File (model.pt) ---> Drive Resumable Upload Session
                            appProperties: {
                              is_latest: "false",
                              sha256: <HASH_NEW>,
                              revision_id: "sha256_<HASH_NEW>",
                              logical_path: "checkpoints/model.pt"
                            }
                            ↓
                     Content Verification (size + MD5 / SHA256 readback)

Phase 2: Atomic Remote Coordination via Drive v3 PATCH
  1. Demote Prior Revisions:
     For each remote file with matching logical_path claiming is_latest="true":
       PATCH /drive/v3/files/<prior_id>
       appProperties: { is_latest: "false", superseded_by: "sha256_<HASH_NEW>" }
  2. Promote New Revision:
     PATCH /drive/v3/files/<new_id>
     appProperties: { is_latest: "true", superseded_by: null }
```

- **Fail-Safe Invariant**: If an upload crashes, times out, or fails integrity verification during Phase 1, it remains marked `is_latest="false"`. It will never be picked up as the active latest revision.
- **Single Active Remote Latest**: Prior revisions are demoted before the new revision is promoted to `is_latest="true"`. At no point in steady-state operations can two remote files validly claim `is_latest="true"`.

### 8. Bounded Pagination & Strict Listing Error Classification
When querying Google Drive folders (e.g. during recovery, file discovery, or duplicate checks):

- **Bounded Pagination Controls**:
  - `pageSize`: Configured to `100` items per request.
  - `nextPageToken`: Evaluated sequentially until exhausted or bounds are reached.
  - `max_list_pages = 50`: Upper bound limiting pagination depth to prevent unbounded memory usage.
  - `max_list_candidates = 5000`: Hard cap on accumulated file candidates.
  - **Loop Detection & Deduplication**: All visited file IDs are registered in `seen_file_ids`. If a duplicate file ID or repeated page token is detected, pagination terminates safely or raises `DriveIntegrityError` to defend against cyclic or malicious folder graphs.
- **Strict Error Classification (Never Return `[]` on Faults)**:
  - If a listing request encounters an HTTP or network error, it **must never return an empty list `[]`**.
  - Returning `[]` on an error would lead the caller to conclude that the remote folder is empty, risking duplicate uploads, false overwrite decisions, or destructive state resets.
  - The listing engine strictly classifies and raises typed exceptions:
    - **HTTP 401 / 403**: Raises `DriveAuthenticationError`.
    - **HTTP 429 / Quota limits**: Raises `DriveQuotaExceededError`.
    - **HTTP 5xx / Network Drops / Socket Timeouts / Malformed JSON**: Raises `DriveTransferError`.

---

## Authoritative Verification Guarantees

1. **No Local Size Substitution & Readback Fallback**: The store never substitutes expected local file sizes when remote metadata is missing. Authoritative remote size and `md5Checksum` are queried from the Drive REST API (`GET /drive/v3/files/<id>?fields=id,name,size,md5Checksum`); if remote size is absent, a `DriveIntegrityError` is raised. If `md5Checksum` is omitted by the server, `publish()` performs a bounded streaming readback (`alt=media`) to verify the remote content's SHA256 and byte length before marking `DRIVE_VERIFIED`.
2. **Independent OAuth**: Drive credentials are authenticated separately from the Colab CLI identity. Colab CLI OAuth scopes are never used for unrestricted Drive writes.
3. **Local Protected Storage**: Drive tokens and journal files are stored server-side with strict `0600` file permissions and `0700` directory permissions.
4. **Zero Frontend / VM Exposure**: Drive OAuth credentials, refresh tokens, and upload session URLs are **never** passed to the Colab VM, exposed in web dashboard APIs, or included in client event streams.
5. **Private by Default**: No public link sharing or permissions are altered; files remain private to the authenticated researcher's Google Drive account.
6. **Fail-Safe Checkpoint Retention**: Local verified checkpoints are **never deleted** before Drive verification and publication succeed.
7. **Dashboard XSS & CSP Defense**: Remote Drive artifact metadata (file IDs, filenames, and SHA256 digests) displayed in the monitoring dashboard (`/api/v1/runs/<run_id>/drive` and `/api/v1/runs/<run_id>/artifacts`) are rendered strictly using `document.createElement()` and `.textContent` with zero unsafe `innerHTML` interpolation and zero inline handlers under strict CSP nonces (`script-src 'self' 'nonce-...'`, `'unsafe-inline'` removed).

---

## Verification Scope, PR Stack Status & Operational Rollback

### Three-Tier Verification Scope

1. **Tier 1 — Offline Unit, Integration & Resumable Fault-Injection Tests (Verified in CI & Locally)**:
   - 219 tests pass 100% offline without live Google cloud credentials (`test_google_drive_store.py`, `test_upload_worker.py`, `test_adversarial_security_and_fault_injection.py`, `test_v2_security_and_resumable_recovery.py`, `test_project_e2e.py`).
   - Simulates HTTP 308 range recovery, byte-accurate chunk offset reconciliation, missing Range headers, mid-upload network drops, expired sessions (HTTP 404/410), and corrupted journal recovery.
2. **Tier 2 — Local Headless Browser DOM & XSS Verification (Playwright / Chromium)**:
   - Headless browser verification testing artifact rendering in the M8 dashboard against live loopback `MonitorServer`.
   - Asserts copy-to-clipboard SHA-256 tooltips, Drive publication badges, and complete resistance to script injection from adversarial artifact names.
3. **Tier 3 — Owner-Gated Live Environment Checks (Pending Final Owner Signoff)**:
   - Live cloud verification requiring explicit human credentials:
     - Real Google OAuth credentials and live Google Drive cloud API (`Colab-Research/` folder structure).
     - Actual Colab GPU VM allocation.
     - Production reverse-proxy deployment with TLS termination.
   - Strictly excluded from automated CI runs pending owner authorization.

### Release & Merge Status

The Google Drive durable artifact storage engine was merged into `main` via **PR #22 (`670c6bc` / `a1629f9846f19c8d5d4e90469ee9e376b64d98e9`)**:
- **PR #3**: Initial Project Workspace Architecture Proposal.
- **PRs #13–#20**: Milestones M1–M8 implementations (including M7 Google Drive store and M8 telemetry visualization).
- **PR #22**: Multi-worker integration, fault-injection tests, and documentation reconciliation.

Post-merge hardening delivered under **Issue #23 (`fix/v2-postmerge-hardening-23`)** and **Review `5642047783` (F1–F4)** includes:
- **F1 Canonical Versioned Revision Identity & Accessors**: Canonical string IDs (`{project_id}/{run_id}/{logical_path}@sha256_{FULL_SHA256}`), immediate-successor `superseded_by` chain (`A -> B -> C -> D -> E -> None`), and typed accessors `get_latest_artifact()`, `get_latest_revision_id()`, `get_latest_sha256()`.
- **F2 Two-Phase Remote Coordination Protocol**: Drive upload with `is_latest="false"`, content verification, demote prior revisions via Drive v3 `PATCH`, promote new revision to `is_latest="true"` via `PATCH`.
- **F3 Bounded Pagination & Strict Error Classification**: `pageSize=100`, bounded `max_list_pages=50`, `max_list_candidates=5000`, loop detection via visited ID tracking, and strict error classification (never returning `[]` on HTTP 401/403/429/5xx or network errors).
- **F4 Catastrophic Loss Recovery Matrix**: Unversioned latest recovery via single verified remote `is_latest="true"` marker; fail-closed `DriveIntegrityError` when 0 or >1 remote revisions claim `is_latest="true"`; fail-closed `DriveTransferError` / `DriveAuthenticationError` / `DriveQuotaExceededError` on remote listing faults.
- **P1 Private Checkpoint Staging**: `0700` staging directories, `0600` temporary/staged files with `fsync`, symlink/special-file rejection, canonical staging-root containment validation.
- **P2 Local Artifact Retrieval Size-Mismatch Error Handling**: Capturing `actual_size` before unlinking `.part.` temp file so `ArtifactValidationError` is always raised instead of `FileNotFoundError`.
- **P2 Controlled Colab HTTP 500 Upload Recovery**: Validating remote paths, probing missing parent directories via bounded `exec_code`, and retrying at most once if newly created.
- **Mandatory Offline E2E Pipeline Verification**: Full lifecycle pass in `tests/test_project_e2e.py` including the 5-revision history & catastrophic manifest loss scenario (`test_five_revision_history_and_catastrophic_manifest_loss_e2e`).
- **Review Status**: `READY_FOR_INDEPENDENT_MERGE_REVIEW`.

> [!NOTE]
> All automated tests execute in strictly offline/mock environments (`OFFLINE_RELEASE_READY`). Any live cloud GPU allocation or live Google Drive API requests require explicit human operator approval (`LIVE_VALIDATION_REQUIRES_APPROVAL`).

## Human-Approval-Gated Live Smoke-Test Procedures

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
   Start server: `python3 -m colab_adapter.cli monitor --host 127.0.0.1 --port 8443 --auth-token "<MONITOR_AUTH_TOKEN>"`.
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

### Operational Rollback Instructions

If unexpected transfer issues occur in staging or production:

1. **Halt Upload Worker & Broker Daemons**:
   ```bash
   kill $(pgrep -f "colab_adapter.broker") 2>/dev/null || true
   kill $(pgrep -f "colab_adapter.monitor") 2>/dev/null || true
   ```
2. **Release Stale Locks**:
   ```bash
   rm -rf ./artifacts/.transfer_locks/*.lock
   ```
3. **Revert Git Branch**:
   ```bash
   git checkout main
   # Or revert the merge commit
   git revert -m 1 <MERGE_COMMIT_SHA>
   ```
4. **Preservation of Local Artifacts**:
   Rolling back the code never deletes local verified checkpoints (`./artifacts/projects/`) or uncommitted journal entries (`transfer_journal.json`), which remain available for manual recovery.
