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

### 7. Google Drive Custom Property Limit (124 UTF-8 Bytes) & Compact Schema v2

Google Drive imposes a hard limit on custom file properties (`appProperties`):
- **124 UTF-8 bytes limit**: Each property key and value combined must not exceed 124 UTF-8 bytes (see [Google Drive API limits of custom file properties](https://developers.google.com/workspace/drive/api/guides/properties#limits_of_custom_file_properties)).
- **Property count cap**: At most 30 custom properties per file.

Storing full canonical artifact IDs (e.g., `superseded_by: vision-research/experiment-001/checkpoints/model.pt@sha256_<64hex>`) consumed 136 UTF-8 bytes (13 bytes key + 123 bytes value), provoking HTTP 400 Bad Request rejection from the Drive API.

To guarantee compliance, Google Drive Artifact Storage V2 implements a strict separation:
- **Canonical local identity in `PublicationManifest`**: Full canonical artifact IDs (`{project_id}/{run_id}/{logical_path}@sha256_{full_sha256}`) live exclusively in the local `PublicationManifest` (persisted in `manifest.json` and `manifest.json.bak` with mode `0600`). Local `superseded_by` stores the full canonical successor ID.
- **Compact Schema v2 on Drive**: Remote Drive objects carry only compact, fixed-size properties:

| Property Key | Value Format | Size (key + value) | Description |
| :--- | :--- | :--- | :--- |
| `schema` | `"2"` | 7 bytes | Compact metadata schema version |
| `sha256` | `<64-hex>` | 70 bytes | Full content SHA-256 digest of this revision (never truncated, lowercased) |
| `path_sha256` | `<64-hex>` | 75 bytes | SHA-256 hex digest of the UTF-8 logical artifact ID (`{project_id}/{run_id}/{logical_path}`) |
| `is_latest` | `"true"` \| `"false"` | 14 bytes | Remote head revision marker |
| `next_sha256` | `<64-hex>` (optional) | 75 bytes | Content digest of immediate successor (present only on superseded historical revisions) |

- **Successor Canonical ID Reconstruction**: When reading remote properties, the full canonical successor ID is reconstructed deterministically from the known project, run, and logical path plus `next_sha256`:
  `canonical_successor_id = f"{project_id}/{run_id}/{logical_path}@sha256_{next_sha256}"`
- **Legacy Read-Only Compatibility**: Pre-v2 legacy properties (`logical_path`, `revision_id`, `project_id`, `run_id`, `superseded_by`) are parsed when inspecting older files (`successor_sha256()`, `props_match_path()`) but are **never written** for new revisions. When demoting or promoting files remotely, any existing legacy keys are explicitly deleted by setting their values to `None` in the Drive v3 `PATCH` body.
- **Pre-Flight Validation**: `validate_drive_properties` is executed before *every* remote write (resumable session initialization and metadata PATCH). Any property set exceeding limits is rejected locally with `DriveTransferError` before network dispatch. Error messages report only the offending key and byte lengths, strictly redacting property values.
- **Unicode & Long-Path Safety**: Long nested directory structures and multi-byte Unicode paths are bound collision-safely via `path_sha256` (64 hex characters), ensuring total property size never exceeds 75 bytes regardless of path length or character set.
- **Content Verification Over Metadata Hints**: Remote properties are treated strictly as metadata hints. Content integrity is always verified by downloading the bytes and hashing them independently (`alt=media`), never trusting property values blindly.

### 8. Two-Phase Remote `is_latest` Coordination Protocol & Replay Safety

To prevent race conditions, split-brain `is_latest` states, or premature promotion of unverified files, `GoogleDriveArtifactStore` coordinates remote revision state via a strict two-phase protocol:

```text
Phase 1: Resumable Upload with is_latest="false"
  Local File (model.pt) ---> Drive Resumable Upload Session
                            appProperties: {
                              schema: "2",
                              sha256: <HASH_NEW>,
                              path_sha256: <PATH_HASH>,
                              is_latest: "false"
                            }
                            ↓
                     Content Verification (size + MD5 / SHA256 readback)

Phase 2: Atomic Remote Coordination via Drive v3 PATCH
  1. Demote Prior Revisions:
     For each remote file matching path_sha256 claiming is_latest="true":
       PATCH /drive/v3/files/<prior_id>
       appProperties: {
         is_latest: "false",
         next_sha256: <HASH_NEW>,
         logical_path: null,
         revision_id: null,
         superseded_by: null
       }
  2. Promote New Revision:
     PATCH /drive/v3/files/<new_id>
     appProperties: {
       is_latest: "true",
       logical_path: null,
       revision_id: null,
       superseded_by: null
     }
```

#### Replay Semantics Matrix

The publish pipeline enforces strict replay safety across all invocation scenarios:

| Scenario | Trigger / Initial State | Publish Action | Remote Drive State | Local Manifest State |
| :--- | :--- | :--- | :--- | :--- |
| **New Revision** | New content SHA-256 published for logical path (e.g. Epoch B after A). | Uploads new file (`is_latest="false"`), verifies size/checksums, demotes prior revision (`next_sha256=sha_B`), promotes new revision (`is_latest="true"`). | Both remote files preserved; old file is historical (`is_latest="false"`), new file is latest (`is_latest="true"`). | Both revisions recorded; old has `superseded_by=canonical_id(B)`, new has `is_latest=True`, `latest_by_path` points to B. |
| **Idempotent Republish** | Existing historical revision re-published (e.g. republishing Revision A while B is latest). | Detected via manifest, journal, or remote file inspection (`_remote_revision_state` = `historical`). Historical revision stays historical. **0 new upload sessions, 0 bytes transferred, 0 Drive PATCH requests**. | Unchanged. B remains remote latest (`is_latest="true"`). A remains historical (`is_latest="false"`). | Manifest preserves B as latest (`latest_by_path` unchanged). A recorded as historical (`is_latest=False`). Returns stored artifact for A. |
| **Interrupted-Publish Recovery** | Previous publish crashed after upload/verify but before coordination (`_remote_revision_state` = `pending`). | Replay detects pending remote revision; coordinates latest pointer (`_coordinate_remote_latest_revision`) to demote prior revisions and promote pending file to `is_latest="true"`. | Pending remote file promoted to latest (`is_latest="true"`), prior demoted (`is_latest="false"`). Zero duplicate files created. | Manifest updated with published revision as latest. |
| **Explicit Rollback** | User desires to revert logical latest pointer to earlier revision A. | **NOT implemented**. Re-publishing A is strictly idempotent and does NOT demote B. (To make A the new latest, user must publish it under a new timestamp or new revision). | Historical revisions remain historical. | Head pointer is not moved backward by replay. |

#### Crash-Point Recovery Matrix

| Crash Point | Remote Drive State at Crash | Recovery Action on Publish Retry | Converged Final State |
| :--- | :--- | :--- | :--- |
| **Crash before demotion** (Upload succeeded with `is_latest="false"`, crash before PATCH demoting old) | Old file has `is_latest="true"`; new file has `is_latest="false"` (pending state). | Retry detects matching file in journal/remote; invokes `_coordinate_remote_latest_revision`, demotes old file, promotes new file. | Old file demoted (`is_latest="false"`), new file promoted (`is_latest="true"`), single latest. |
| **Crash after demotion before promotion** (Old file demoted to `is_latest="false"`, crash before PATCH promoting new file) | Old file has `is_latest="false"`; new file has `is_latest="false"` (0 files marked latest remotely). | Retry detects new file is `pending`; `publish()` calls `_coordinate_remote_latest_revision`, promoting new file to `is_latest="true"`. | Old file demoted, new file promoted (`is_latest="true"`), single latest. |
| **Crash after promotion before manifest save** (Both demote and promote PATCHes succeeded remotely, crash before saving `manifest.json`) | Remote has new file as `is_latest="true"`, old as `is_latest="false"`; local manifest missing new record. | Retry detects remote file already latest; registers new artifact in manifest, saves manifest atomically with backup `.json.bak`. | Single latest on Drive and in manifest; no duplicate upload. |

#### Fail-Closed Ambiguity Handling on Recovery

During unversioned recovery when the local manifest and journal are both unavailable:
- **0 latest remote files**: If no remote revision has `is_latest="true"`, retrieval fails closed with `DriveIntegrityError` (`"Ambiguous unversioned Drive recovery: no remote revision marked latest found..."`) unless `allow_unverified_recovery=True` on a single file with authoritative MD5 and size.
- **>1 latest remote files**: If multiple remote revisions claim `is_latest="true"`, retrieval fails closed with `DriveIntegrityError` (`"Ambiguous unversioned Drive recovery: multiple (...) remote revisions marked latest exist..."`).
- **Missing/Invalid SHA-256**: If the latest candidate lacks a valid 64-hex SHA-256 property, retrieval fails closed with `DriveIntegrityError`.

### 9. Bounded Pagination & Strict Listing Error Classification
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
- **F2 Two-Phase Remote Coordination Protocol**: Drive upload with `is_latest="false"`, content verification, demote prior revisions via Drive v3 `PATCH` (setting `next_sha256` and clearing legacy keys), promote new revision to `is_latest="true"` via `PATCH`. Compact schema v2 adheres strictly to the 124 UTF-8 bytes custom property limit (https://developers.google.com/workspace/drive/api/guides/properties#limits_of_custom_file_properties).
- **F3 Bounded Pagination & Strict Error Classification**: `pageSize=100`, bounded `max_list_pages=50`, `max_list_candidates=5000`, loop detection via visited ID tracking, and strict error classification (never returning `[]` on HTTP 401/403/429/5xx or network errors).
- **F4 Catastrophic Loss Recovery Matrix**: Unversioned latest recovery via single verified remote `is_latest="true"` marker; fail-closed `DriveIntegrityError` when 0 or >1 remote revisions claim `is_latest="true"`; fail-closed `DriveTransferError` / `DriveAuthenticationError` / `DriveQuotaExceededError` on remote listing faults.
- **P1 Private Checkpoint Staging**: `0700` staging directories, `0600` temporary/staged files with `fsync`, symlink/special-file rejection, canonical staging-root containment validation.
- **P2 Local Artifact Retrieval Size-Mismatch Error Handling**: Capturing `actual_size` before unlinking `.part.` temp file so `ArtifactValidationError` is always raised instead of `FileNotFoundError`.
- **P2 Controlled Colab HTTP 500 Upload Recovery**: Validating remote paths, probing missing parent directories via bounded `exec_code`, and retrying at most once if newly created.
- **Mandatory Offline E2E Pipeline Verification**: Full lifecycle pass in `tests/test_project_e2e.py` including the 5-revision history & catastrophic manifest loss scenario (`test_five_revision_history_and_catastrophic_manifest_loss_e2e`).
- **Review Status**: `READY_FOR_INDEPENDENT_MERGE_REVIEW pending CI only`.

> [!NOTE]
> All automated tests execute in strictly offline/mock environments (`OFFLINE_RELEASE_READY`). Real Google Drive and Google Colab environments have never been exercised in this codebase or test runs; live cloud GPU allocations, OAuth credentials, and live Drive writes are strictly unauthorized without explicit human operator approval (`LIVE_VALIDATION_REQUIRES_APPROVAL`).

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
   Validate project manifest with `python3 -m colab_adapter.cli project validate --manifest ./gpu-smoke-test/colab-project.json --json`.
5. **Zero-Allocation Offline Execution Plan**:
   Generate plan with `python3 -m colab_adapter.cli project plan --manifest ./gpu-smoke-test/colab-project.json --json` (verifying 0 VMs allocated).
6. **Explicit Approval Token Provisioning**:
   Provision session: `python3 -m colab_adapter.cli project approve-request --manifest ./gpu-smoke-test/colab-project.json --lease-seconds 900 --json` then `python3 -m colab_adapter.cli project approve <approval-id>          # operator, in their own terminal` then `python3 -m colab_adapter.cli project up --manifest ./gpu-smoke-test/colab-project.json --lease-seconds 900 --approval-id <approval-id> --json`.
7. **Incremental Workspace Source Sync**:
   Synchronize sources: `python3 -m colab_adapter.cli project sync --manifest ./gpu-smoke-test/colab-project.json --session-id <session-id> --json`.
8. **Remote Training Execution**:
   Run GPU probe: `python3 -m colab_adapter.cli project exec "python3 gpu_probe.py" --session-id <session-id> --json`.
9. **Execution Output & CUDA Device Verification**:
   Inspect output logs to confirm remote GPU was active, executed under assigned PID, and emitted training metrics.
10. **Output Artifact Pull & Verification**:
    Retrieve generated output/checkpoint and verify SHA256 checksum against execution record.
11. **Graceful Session Teardown**:
    Terminate session: `python3 -m colab_adapter.cli project stop --session-id <session-id> --json`.
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

## Drive Store Hardening: Pending Revisions, Locks and Rate Limits (PR #24, offline)

This section describes `colab_adapter/storage/google_drive.py`, `drive_metadata.py` and `worker.py` on branch `fix/v2-postmerge-hardening-23` (PR #24). Everything here is verified only by offline tests with a mock Drive client. No live Google Drive or OAuth validation has been performed, and live validation requires explicit owner approval.

### 1. Shared per-manifest lock

- The Drive store takes the same lock as the local store: `<run_dir>/.locks/manifest.lock`, where `<run_dir>` is the local run directory that holds the manifest. The two stores therefore serialise on one lock when they merge into one manifest. Details are in ARCHITECTURE_V2 section 14.
- The Drive store calls the lock with a 60-second timeout. A timeout raises `JournalLockError`.

### 2. Base marker captured at intent

- When a transfer is first recorded, the store reads the current remote head for the logical path and stores its full SHA-256 as the base marker. If there is no head, the marker is the literal `none`. The marker is kept in the transfer journal (`TransferRecord.base_sha256`) and written to the remote file as the property `base_sha256`. It is captured once per transfer, not on each attempt.
- A legacy journal record that has no marker is captured only if it has not started uploading (no remote file id, no resumable session, zero bytes sent). Any other legacy record keeps `None`, which the rules below treat as a legacy case.
- The marker is validated before any remote write. It must be `none` or a full 64-hex digest.

### 3. Pending revision disposition

An uploaded revision starts as `pending`. It is promoted to latest only after the decision below says so. `pending_disposition()` is a pure function of an observed snapshot:

| Disposition | When it applies | Effect |
| --- | --- | --- |
| `PROMOTE` | The pending revision is the genuine successor of the current head (the head's SHA equals the base marker). Or the head is missing, the base marker names the pending revision as its successor (interrupted coordination), or the base is `none` and nothing else exists for the path. | The current head is demoted (`is_latest=false`, `next_sha256` set to the pending digest), then the pending revision is promoted (`is_latest=true`). |
| `ALREADY_LATEST` | The head is already this content. | No write. |
| `STALE` | The head moved on since the upload was intended (base marker does not match the head). | Pending stays pending. It is not promoted and it is not demoted. It is still retrievable by its exact versioned id, with full-hash verification. |
| `FAIL_CLOSED` | The remote state cannot be explained, for example the latest names a different successor, or a legacy pending revision conflicts with the latest. | Raises `DriveIntegrityError`. Nothing is guessed. |

Legacy rules (records or remote files without a base marker):
- With a current head: the pending revision is promoted only if the head already names it as successor. Otherwise the result is `FAIL_CLOSED`.
- With no head: the pending revision is promoted only if no other non-pending revision exists for the path. Otherwise the result is `FAIL_CLOSED`.

The disposition is decided at the time of settlement, using the head re-read at that moment. It is not taken from the state at upload start.

### 4. Crash during coordination

- Demotion of the old head is written before the promotion. A crash between the two leaves no latest. The next recovery finds the base revision whose SHA equals the base marker and whose successor is the pending digest, and promotes the pending revision (`PROMOTE`).
- A crash after promotion but before the manifest is saved is repaired when the publish is replayed. The replay re-reads remote state and classifies it again.
- Recovery never trusts the remote properties as content proof. Content is verified by hashing downloaded bytes.

### 5. Limitation: no compare-and-swap on Drive

Google Drive offers no compare-and-swap for file properties. The stale and promote checks are therefore a read followed by a write. **The guarantee holds only if every writer to the Drive folder goes through this host's per-manifest lock.** A writer outside this host, or a writer that bypasses the lock, can still race between the read and the write. This is a documented limit, not a verified property of Drive.

### 6. Terminal integrity errors in the upload worker

- `DriveIntegrityError` is a subclass of `ArtifactValidationError`. In the upload worker, any `ArtifactValidationError` raised from a publish sets the transfer to `UPLOAD_FAILED` at once and is not retried.
- Other exceptions use bounded retries (`RETRY_PENDING`), up to the record's `max_retries`, and then `UPLOAD_FAILED`.

### 7. Drive rate-limit and quota behaviour

Classification of an HTTP 403 looks at `error.errors[].reason` in the JSON body. If the body is not JSON, a substring check is used. If both quota and rate reasons are present, quota wins.

| Class | Reasons | Behaviour |
| --- | --- | --- |
| Quota | `dailyLimitExceeded`, `quotaExceeded`, `storageQuotaExceeded` | `DriveQuotaExceededError`. Never retried, because retrying cannot help. |
| Rate | `rateLimitExceeded`, `userRateLimitExceeded`, `sharingRateLimitExceeded` | Bounded retry with backoff. When the retry budget is exhausted the store raises `DriveQuotaExceededError` with a message that says the rate-limit budget was exhausted. |
| Auth or other | Any other reason, or no recognised marker | `DriveAuthenticationError`. Never retried. |

Other status handling:
- HTTP 401 refreshes the access token and retries while the budget lasts. When the budget runs out it raises `DriveAuthenticationError`.
- HTTP 429 and 5xx (500, 502, 503, 504) are retried with bounded backoff. When the budget runs out they raise `DriveTransferError`.
- Retry-After is honoured in two forms, delta-seconds and HTTP-date. Negative, NaN, infinite and past values are ignored, and the store falls back to jittered exponential backoff. Every wait is capped at 60 seconds, whichever source it came from.

Note: a rate-limit 403 that exhausts its budget surfaces as `DriveQuotaExceededError`, not as a separate rate-limit error type. Callers that need to tell them apart must read the message text.

### 8. Drive read-path validation

- Drive publish validates `project_id`, `run_id`, the artifact type and the file name before any remote call.
- Drive `get_metadata()` and `retrieve()` validate the project and run segments of an artifact id before any manifest read, path construction or network call. A versioned retrieval requires a full 64-hex digest.
- Drive property values are checked by `validate_drive_properties()` before every remote write. Oversized or non-string values are rejected locally.
