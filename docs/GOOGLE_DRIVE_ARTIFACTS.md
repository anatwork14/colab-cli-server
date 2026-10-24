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
- The Drive API responds with `308 Resume Incomplete` and the `Range: bytes=0-<N>` header. The worker parses `<N>` and sets the next chunk offset to `<N> + 1`.
- If an ambiguous network exception occurs (socket drop or timeout), the worker reconciles committed bytes with the server before backing off.
- The background `ArtifactUploadWorker` includes an in-process retry scheduler that automatically retries `RETRY_PENDING` transfers using exponential backoff with jitter while the service is running, without requiring process restarts.
- If an upload session expires (`HTTP 404/410`), the store checks if the file was already committed in the target folder to prevent creating duplicate Drive objects, or safely re-initiates a fresh upload session.
- During download (`retrieve`), files are streamed in 64 KiB chunks with bounded memory to a private temporary file (`0600`), incrementally hashing SHA256/MD5 and performing atomic rename only after full verification passes.

## Authoritative Verification Guarantees

1. **No Local Size Substitution**: The store never substitutes expected local file sizes when remote metadata is missing. Authoritative remote size and `md5Checksum` are queried from the Drive REST API (`GET /drive/v3/files/<id>?fields=id,name,size,md5Checksum`); if remote size is absent, a `DriveIntegrityError` is raised and the artifact is not marked verified.
2. **Independent OAuth**: Drive credentials are authenticated separately from the Colab CLI identity. Colab CLI OAuth scopes are never used for unrestricted Drive writes.
3. **Local Protected Storage**: Drive tokens and journal files are stored server-side with strict `0600` file permissions and `0700` directory permissions.
4. **Zero Frontend / VM Exposure**: Drive OAuth credentials, refresh tokens, and upload session URLs are **never** passed to the Colab VM, exposed in web dashboard APIs, or included in client event streams.
5. **Private by Default**: No public link sharing or permissions are altered; files remain private to the authenticated researcher's Google Drive account.
6. **Fail-Safe Checkpoint Retention**: Local verified checkpoints are **never deleted** before Drive verification and publication succeed.
