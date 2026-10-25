# Artifact Recovery & Experiment Reproducibility (V2)

## Motivation

Deep learning training jobs running on ephemeral cloud infrastructure are vulnerable to unrecoverable data loss if the runtime terminates abruptly before checkpoints are transferred off the instance.

This guide details the end-to-end recovery workflow, code snapshot determinism, checkpoint retention policy, and disaster recovery procedures.

## Reproducibility Package

For each training run, Colab V2 generates an immutable reproducibility package:

```text
source_snapshot.tar.gz    # Deterministic tarball of project source code
source_manifest.json      # Per-file SHA256 hashes and file sizes
environment.json          # Python interpreter, OS platform, package versions
run_manifest.json         # Complete provenance: git commit SHA, arguments, timestamps
```

### Deterministic Source Snapshots

To guarantee bit-for-bit reproducibility:
1. All file paths are strictly sorted.
2. File mtime headers in tar entries and gzip streams are normalized to epoch `0.0`.
3. Ownership (`uid`, `gid`, `uname`, `gname`) is set to `0` / empty strings.
4. Sensitive files are strictly filtered out by default:
   - Git internals (`.git/`)
   - Environment secret files (`.env`, `.env.*`)
   - Cryptographic keys (`*.key`, `*.pem`, `id_rsa`)
   - Python bytecode caches (`__pycache__`, `*.pyc`)
   - Virtual environments (`.venv/`, `node_modules/`)

## Automatic Checkpoint Policy

Configurable via `colab.yaml` or programmatically via `CheckpointPolicyConfig`:

```yaml
artifacts:
  local_dir: ./artifacts

  cloud:
    provider: google_drive
    enabled: true
    project_folder: Colab-Research

  checkpoint:
    publish_on_complete: true
    publish_on_new_checkpoint: true
    minimum_interval_seconds: 600   # Rate limits uploads to at most once per 10 mins
    keep_last: 3                    # Retains latest 3 local checkpoints

  source_snapshot:
    enabled: true
```

### Safety Invariants

1. **At Least One Checkpoint Guaranteed**: Even if `keep_last` is set to `0` or disk cleanup is triggered, the system strictly preserves at least 1 verified local recoverable checkpoint on disk.
2. **Never Delete Cloud Objects**: Local pruning operations never delete or modify remote Google Drive objects without explicit human authorization.
3. **No Incomplete Checkpoint Uploads**: Multi-tier integrity verification must pass before an artifact is scheduled for cloud upload. If PyTorch weights cannot be verified or file size is truncated, publication is blocked.
4. **Interruption Recovery Boundary**: If training is interrupted *before* a completed checkpoint has been transferred off the remote VM disk, recovery cannot be guaranteed. Frequent, controlled checkpointing with interval throttling is strongly recommended.

## Disaster Recovery Procedure

If a Colab runtime terminates unexpectedly or a local host machine crashes:

### Step 1: Query Run State
From any authorized terminal or dashboard client:
```bash
python3 -m colab_adapter.cli project status
# Or query broker endpoint:
curl -H "Authorization: Bearer <TOKEN>" http://localhost:8080/api/v1/runs/<run_id>
```

### Step 2: Retrieve Latest Verified Checkpoint
If the checkpoint is stored in Google Drive:
```python
from colab_adapter.storage import GoogleDriveArtifactStore

store = GoogleDriveArtifactStore(credentials_file="~/.config/colab-cli-server/drive_token.json")
store.retrieve(
    artifact_id="<project-id>/<run-id>/checkpoints/step_1000.pt",
    destination="./resumed_step_1000.pt",
)
```

### Step 3: Plan Resumption Under New Approved Session
Resume training with explicit human approval for the new runtime:
```python
from colab_adapter.telemetry import plan_resume_from_checkpoint

resume_plan = plan_resume_from_checkpoint(
    checkpoint_path="./resumed_step_1000.pt",
    target_remote_dir="/content/project/checkpoints",
    extra_args={"resume": "True", "start_epoch": "10"},
)
print(resume_plan["cli_arguments"])
```
No automatic VM allocation is ever provoked; the user explicitly reviews and approves the new runtime.

---

## Single-Host Cross-Process Artifact Publication Atomicity Guarantee

During experiment execution, checkpoint recovery, and disaster recovery resumption, multiple host processes (e.g. background upload workers, training runs, and CLI recovery tools) coordinate safely:

### 1. OS-Level File Locking (`fcntl.flock`)
- **Per-Artifact Locks**: Every transfer operation acquires an exclusive advisory OS-level lock on `.transfer_locks/<transfer_id>.lock` before reading or updating transfer state.
- **Manifest Locks**: `PublicationManifest` operations acquire `.manifest.lock` ensuring concurrent processes do not interleave writes or corrupt JSON manifests.

### 2. Durable Leases & Fencing
- Each `TransferRecord` tracked in `TransferJournal` carries durable lease attributes:
  - `owner_pid`: Process ID of the active worker.
  - `lease_token`: Unique cryptographic UUID identifying the lease instance.
  - `lease_expires_at`: Monotonic lease deadline preventing indefinite hangs.
  - `fencing_epoch`: Monotonically increasing counter incremented on every lease acquisition.
- **Dead-PID Lease Reclamation**: If a worker crashes or is terminated by `SIGKILL`, surviving processes verify PID liveness using `os.kill(owner_pid, 0)`. If the process no longer exists, the lease is safely reclaimed and the fencing epoch is bumped.

### 3. Content-Addressed Revision Safety
- **Deduplication**: If two processes attempt to publish identical content (`sha256` digest and `size_bytes`), the second transfer immediately detects the existing published record, reuses the verified `remote_file_id`, and avoids duplicate Drive allocations.
- **Revision Safety**: If a file at the same filename is updated with new content (different `sha256`), the system invalidates stale session URLs and initiates a verified new remote upload. Stale hashes or obsolete remote file IDs are never returned.

### 4. Scope Boundary
- **Single-Host Guarantee**: Atomicity and lease guarantees are guaranteed on a single host via the local POSIX filesystem (`fcntl.flock`).
- **Distributed Multi-Host Writers Out of Scope**: Multi-host distributed writers sharing an uncoordinated network file system (NFS/SMB) without an external distributed lock service (e.g. etcd, ZooKeeper, or Consul) are explicitly out of scope for the local workspace journal.

---

## Dashboard XSS & CSP Security Architecture

When viewing recovered artifacts, logs, and experiment provenance through the monitoring web application:
- **Zero Inline Event Handlers**: HTML templates contain zero inline handlers (`onclick`, `oninput`, `onchange`, `onerror`).
- **Safe DOM Construction**: Zero unsafe `innerHTML` interpolation for untrusted run/log/artifact/session data; all dynamic DOM nodes are built via `document.createElement` and `.textContent`.
- **Per-Response Cryptographic CSP Nonces**: Enforces `script-src 'self' 'nonce-...'`, `style-src 'self' 'nonce-...'`, `object-src 'none'`, `base-uri 'none'`, and `frame-ancestors 'none'`, with `'unsafe-inline'` completely removed, alongside static `/monitor.js` and `/monitor.css` asset routes.

---

## Verification Scope, PR Stack Status & Operational Rollback

### Three-Tier Verification Scope

1. **Tier 1 — Offline Unit, Integration & Resumable Fault-Injection Tests (Verified in CI & Locally)**:
   - 219 tests pass 100% offline without live Google cloud credentials (`test_reproducibility.py`, `test_checkpoint_policy.py`, `test_artifacts_and_telemetry.py`, `test_google_drive_store.py`, `test_project_e2e.py`).
   - Simulates deterministic source snapshot tarball generation, mtime/ownership normalization, checkpoint interval throttling, and journal recovery from corrupt state.
2. **Tier 2 — Local Headless Browser DOM & XSS Verification (Playwright / Chromium)**:
   - Headless browser verification testing artifact inspection in the M8 dashboard against live loopback `MonitorServer`.
   - Asserts copy-to-clipboard SHA-256 tooltips, Drive publication badges, and complete resistance to script injection from adversarial artifact names.
3. **Tier 3 — Owner-Gated Live Environment Checks (Pending Final Owner Signoff)**:
   - Live cloud verification requiring explicit human credentials:
     - Real Google OAuth credentials and live Google Drive cloud API (`Colab-Research/` folder structure).
     - Actual Colab GPU VM allocation.
     - Production reverse-proxy deployment with TLS termination.
   - Strictly excluded from automated CI runs pending owner authorization.

### PR Review Stack Status

The artifact recovery and reproducibility subsystem is part of a reconciled review stack:
- **PR #3**: Initial Project Workspace Architecture Proposal.
- **PRs #13–#20**: Milestones M1–M8 implementations (including M6 reproducibility and M7 durable artifact store).
- **PR #22**: Multi-worker integration, fault-injection tests, and documentation reconciliation.

> [!IMPORTANT]
> PRs #3, #13–#20, and #22 form a reconciled review stack and **remain unmerged into `main`** pending final owner approval and completion of Tier 3 live environment checks.

### Operational Rollback Instructions

If an issue occurs during artifact recovery or migration:

1. **Stop Active Daemons**:
   ```bash
   kill $(pgrep -f "colab_adapter.broker") 2>/dev/null || true
   kill $(pgrep -f "colab_adapter.monitor") 2>/dev/null || true
   ```
2. **Clear Stale Lock State**:
   ```bash
   rm -rf ./artifacts/.transfer_locks/*.lock
   ```
3. **Revert Git Branch**:
   ```bash
   git checkout main
   # Or revert merge commit
   git revert -m 1 <MERGE_COMMIT_SHA>
   ```
4. **Data Safety**:
   Local checkpoints and source snapshots stored under `./artifacts/` are never purged or altered by rollback operations.
