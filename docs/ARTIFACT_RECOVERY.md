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
