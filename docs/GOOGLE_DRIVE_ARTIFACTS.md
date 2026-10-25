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

### 1. Canonical Revision Key Format
Each immutable revision is indexed by its content-addressed versioned ID:
```text
<project_id>/<run_id>/<logical_path>@sha256_<full_64_character_sha256>
```
Example:
```text
vision_v2/run-001/checkpoints/model.pt@sha256_3b4f5d133bb7c9c693999b2620703353cba10983fc4aa8162ac1e456fdffa766
```

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
      "metadata": {"step": 100, "epoch": 1}
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
      "metadata": {"step": 200, "epoch": 2}
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

### PR Review Stack Status

The Google Drive durable artifact storage engine is integrated within a reconciled PR review stack:
- **PR #3**: Initial Project Workspace Architecture Proposal.
- **PRs #13–#20**: Milestones M1–M8 implementations (including M7 Google Drive store and M8 telemetry visualization).
- **PR #22**: Multi-worker integration, fault-injection tests, and documentation reconciliation.

> [!IMPORTANT]
> PRs #3, #13–#20, and #22 form a reconciled review stack and **remain unmerged into `main`** pending final owner approval and completion of Tier 3 live environment checks.

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
