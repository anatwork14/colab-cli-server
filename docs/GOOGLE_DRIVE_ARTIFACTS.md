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
