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

### Step 2: Retrieve Latest or Historical Checkpoint
If the checkpoint is stored in Google Drive:

#### Option A: Retrieve the Latest Checkpoint
Querying by logical path automatically resolves the newest revision via `latest_by_path`:
```python
from colab_adapter.storage import GoogleDriveArtifactStore

store = GoogleDriveArtifactStore(credentials_file="~/.config/colab-cli-server/drive_token.json")
store.retrieve(
    artifact_id="<project-id>/<run-id>/checkpoints/model.pt",
    destination="./resumed_model_latest.pt",
)
```

#### Option B: Retrieve a Specific Historical Revision
To roll back to an earlier training epoch or perform ablation against historical weights, specify the canonical versioned ID:
```python
store.retrieve(
    artifact_id="<project-id>/<run-id>/checkpoints/model.pt@sha256_1111111111111111111111111111111111111111111111111111111111111111",
    destination="./resumed_model_epoch1.pt",
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

## Disaster Recovery Matrix & Failure Mode Taxonomy

The table below specifies exact system recovery behaviors across all local and remote failure modes, including catastrophic local manifest and journal loss:

| Failure Mode | Local State Available | Retrieval Request | Resolution Mechanism | Invariant / Expected Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **Normal Operation** | `manifest.json` intact, journal intact | Logical path (`checkpoints/model.pt`) | Resolved directly via `manifest.latest_by_path` pointer | Fetches newest revision (`is_latest=True`) with multi-tier hash verification. |
| **Normal Historical** | `manifest.json` intact, journal intact | Canonical ID (`...@sha256_<HASH>`) | Resolved directly from `manifest.artifacts[rev_id]` | Fetches exact historical revision matching hash. |
| **Primary Manifest Corruption / Deletion** | `manifest.json` missing/corrupt, `manifest.json.bak` exists | Any valid ID or logical path | `PublicationManifest.load()` automatically falls back to `.json.bak` | All revisions recovered; primary manifest rebuilt atomically with mode `0600`. |
| **Journal Loss Only** | `manifest.json` intact, `transfer_journal.json` lost | Any valid ID or logical path | Resolved via `manifest.json` metadata cache | Recovery succeeds without remote listing; journal re-created on next write. |
| **Catastrophic State Loss (Exact Historical)** | `manifest.json`, `.bak`, and journal all deleted | Canonical ID (`...@sha256_<HASH>`) | Drive folder queried; remote files filtered by `appProperties["sha256"]` | Matching remote file resolved, streamed to temporary file, verified, and safely restored. |
| **Catastrophic State Loss (Single Latest)** | `manifest.json`, `.bak`, and journal all deleted | Unversioned path (`checkpoints/model.pt`) | Remote Drive files queried for `appProperties["is_latest"] == "true"` and `path_sha256` | Exactly 1 file matches `is_latest="true"` with valid 64-hex `sha256`: successfully downloads newest verified revision. |
| **Catastrophic Loss: Unversioned Ambiguity (0 latest)** | `manifest.json`, `.bak`, and journal all deleted | Unversioned path (`checkpoints/model.pt`) | Remote Drive files queried for `appProperties["is_latest"] == "true"` | 0 files match `is_latest="true"`: **Fails closed** with `DriveIntegrityError`. No unverified file returned unless `allow_unverified_recovery=True` on single file with md5 and size. |
| **Catastrophic Loss: Remote Split-Brain (>1 latest)** | `manifest.json`, `.bak`, and journal all deleted | Unversioned path (`checkpoints/model.pt`) | Remote Drive files queried for `appProperties["is_latest"] == "true"` | >1 files match `is_latest="true"`: **Fails closed** with `DriveIntegrityError`. Ambiguous download blocked. |
| **Idempotent Historical Re-Publish** | Any state | Re-publish historical revision A | Checked against journal / remote revision state (`STATE_HISTORICAL`) | Historical stays historical (`is_latest="false"`). **0 new upload sessions, 0 bytes transferred, 0 Drive PATCH requests**. |
| **Explicit Rollback Request** | Any state | Request rollback of head pointer | User republishes earlier checkpoint A | **NOT implemented**. Re-publishing A is strictly idempotent and does NOT demote B. Head pointer is never reverted by replay. |
| **Remote Auth Failure During Recovery** | Any local state | Any retrieval or idempotency check | Drive v3 listing returns HTTP 401/403 | **Fails closed** with `DriveAuthenticationError`. Never returns empty `[]` or false 404. |
| **Remote Rate Limit / Quota Exceeded** | Any local state | Any retrieval or idempotency check | Drive v3 listing returns HTTP 429 / quota limit | **Fails closed** with `DriveQuotaExceededError`. Never returns empty `[]`. |
| **Network Timeout / Malformed Remote JSON** | Any local state | Any retrieval or idempotency check | Socket drop, timeout, 5xx, or invalid JSON | **Fails closed** with `DriveTransferError`. Never falls back to empty listing or silent failure. |
| **Corrupted Download Payload** | Any state | Any retrieval | Streamed content SHA256 does not match expected hash | Download rejected; temporary file unlinked; pre-existing local file **never overwritten**. |

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

## Multi-Revision Recovery & PublicationManifest V2 Schema

In distributed and multi-epoch research, models evolve across steps and epochs while reusing base filenames like `model.pt`. Colab V2 prevents historical overwrites and enables exact historical recovery:

### 1. Canonical Revision Keying
Every published checkpoint revision is assigned an immutable canonical ID:
`<project_id>/<run_id>/<logical_path>@sha256_<full_sha256>`

### 2. PublicationManifest V2 Structure
The run's `manifest.json` tracks all historical revisions under `manifest_version: 2`:
- `artifacts`: Maps versioned IDs to complete metadata (`size_bytes`, `sha256`, `remote_file_id`, `step`, `verified_at`).
- `latest_by_path`: Maps logical paths (`checkpoints/model.pt`) to the newest versioned artifact ID.
- `migration_notes`: Records automatic schema upgrade events.

### 3. Transparent V1 -> V2 Migration
When loading legacy V1 manifests, `PublicationManifest.load()` automatically generates versioned `@sha256_` keys, populates `latest_by_path`, updates `manifest_version` to `2`, and appends to `migration_notes`.

### 4. Cross-Process Locking & Staging Protection
- Concurrent writes acquire `.manifest.lock` via `fcntl.flock`.
- Staged revisions write to isolated `*.sha256_*` temporary files before verification, ensuring broken or incomplete uploads never corrupt the recoverable checkpoint history.

### 5. Browser-Verified Recovery UI
In the Monitoring Dashboard, the Checkpoints & Google Drive Artifacts table displays every revision with explicit `LATEST` vs `HISTORICAL` badges, revision IDs, steps, and SHA256 integrity verification. Real-browser headless tests guarantee secure rendering without XSS or inline script risks.

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

### Release & Merge Status

The artifact recovery and reproducibility subsystem was merged into `main` via **PR #22 (`670c6bc` / `a1629f9846f19c8d5d4e90469ee9e376b64d98e9`)**:
- **PR #3**: Initial Project Workspace Architecture Proposal.
- **PRs #13–#20**: Milestones M1–M8 implementations (including M6 reproducibility and M7 durable artifact store).
- **PR #22**: Multi-worker integration, fault-injection tests, and documentation reconciliation.

Post-merge hardening under **Issue #23 (`fix/v2-postmerge-hardening-23`)** and **Review `5642047783` (F1–F4)** delivers:
- **F1 Canonical Versioned Revision Identity & Accessors**: Canonical string IDs (`{project_id}/{run_id}/{logical_path}@sha256_{FULL_SHA256}`), immediate-successor `superseded_by` chain (`A -> B -> C -> D -> E -> None`), and typed accessors `get_latest_artifact()`, `get_latest_revision_id()`, `get_latest_sha256()`. Full canonical identities stay in `PublicationManifest`, while Drive custom properties strictly comply with Google Drive's 124 UTF-8 bytes limit (https://developers.google.com/workspace/drive/api/guides/properties#limits_of_custom_file_properties) using compact schema v2 (`schema: "2"`, `sha256`, `path_sha256`, `is_latest`, `next_sha256`) validated via `validate_drive_properties` before every remote write.
- **F2 Two-Phase Remote Coordination Protocol & Replay Safety**: Drive upload with `is_latest="false"`, content verification, demote prior revisions via Drive v3 `PATCH` (setting `next_sha256` and clearing legacy keys), promote new revision to `is_latest="true"` via `PATCH`. Historical revisions stay historical on replay (idempotent; 0 uploads, 0 PATCHes). Explicit rollback is not implemented.
- **F3 Bounded Pagination & Strict Error Classification**: `pageSize=100`, bounded `max_list_pages=50`, `max_list_candidates=5000`, loop detection via visited ID tracking, and strict error classification (never returning `[]` on HTTP 401/403/429/5xx or network errors).
- **F4 Catastrophic Loss Recovery Matrix**: Unversioned latest recovery via single verified remote `is_latest="true"` marker; fail-closed `DriveIntegrityError` when 0 or >1 remote revisions claim `is_latest="true"`; fail-closed `DriveTransferError` / `DriveAuthenticationError` / `DriveQuotaExceededError` on remote listing faults.
- **P1 Private Checkpoint Staging**: Staging directories created with `0700` and files with `0600` permissions and explicit `os.fsync()`, rejecting symlinks and special files.
- **P2 Local Artifact Size-Mismatch Error Handling**: Captures `actual_size` before unlinking partial `.part.` temp file, raising `ArtifactValidationError` deterministically.
- **P2 Controlled Colab HTTP 500 Upload Recovery**: Validates remote path and parent directory existence, retrying at most once if newly created.
- **Mandatory E2E Pipeline Verification**: Complete 21-stage offline lifecycle test and 5-revision catastrophic manifest loss scenario (`test_five_revision_history_and_catastrophic_manifest_loss_e2e`) passing in `tests/test_project_e2e.py`. Status: `READY_FOR_INDEPENDENT_MERGE_REVIEW pending CI only`.

> [!NOTE]
> All automated tests execute in offline/mock mode (`OFFLINE_RELEASE_READY`). Real Google Drive and Google Colab environments have never been exercised in this worktree or automated tests; live cloud GPU allocations, OAuth credentials, and live Drive writes are strictly unauthorized without explicit human operator approval (`LIVE_VALIDATION_REQUIRES_APPROVAL`).

## 20-Step Operator Guide for Artifact Recovery & Lifecycle

A comprehensive runbook using strictly real, supported CLI commands (`python3 colab_server.py ...` and `python3 -m colab_adapter.cli ...`):

1. **Host Python & Toolchain Verification**: `python3 --version && uv --version` (no sudo required).
2. **User-Space Upstream CLI Install**: `bash scripts/install.sh`.
3. **Headless OAuth Configuration**: `bash scripts/authenticate.sh`.
4. **Diagnostic Pre-Flight**: `python3 colab_server.py doctor` (or `python3 -m colab_adapter.cli doctor`).
5. **Account & Compute-Unit Audit**: `python3 colab_server.py usage` and `python3 -m colab_adapter.cli sessions`.
6. **Project Workspace Init**: `python3 -m colab_adapter.cli project init ./research-project --name research-project`.
7. **Local Code & Excludes Setup**: Place code in project directory; verify `.env` is excluded in `colab-project.json`.
8. **Manifest Validation**: `python3 -m colab_adapter.cli project validate ./research-project --json`.
9. **Offline Execution Plan**: `python3 -m colab_adapter.cli project plan ./research-project --json` (0 cloud allocations).
10. **Approval-Gated Session Allocation**:
    `python3 -m colab_adapter.cli project up ./research-project --gpu T4 --lease 3600 --approval-token "ALLOCATE T4 $(date +%s)" --json`.
11. **Initial Workspace Sync**: `python3 -m colab_adapter.cli project sync ./research-project --json`.
12. **Remote Training Execution**: `python3 -m colab_adapter.cli project exec ./research-project -- --epochs 10 --batch-size 32 --json`.
13. **Incremental Code Hot-Reload**: Modify local source code and run `python3 -m colab_adapter.cli project sync ./research-project --json`.
14. **Monitoring Dashboard Launch**: `python3 -m colab_adapter.cli monitor --host 127.0.0.1 --port 8080 --auth-token "<SECRET_TOKEN>"`.
15. **Drive Credential Security (0600)**: Configure `~/.config/colab-cli/drive_credentials.json` with permissions `chmod 0600 ~/.config/colab-cli/drive_credentials.json`.
16. **Multi-Revision Checkpoint Publishing**: Verify `LocalArtifactStore` and `GoogleDriveArtifactStore` preserve all revisions under immutable `@sha256_<HASH>` identities in `PublicationManifest` v2.
17. **Historical & Latest Checkpoint Retrieval**: Retrieve latest checkpoint via logical path `research-project/<run_id>/checkpoints/model.pt` or historical revision via canonical `@sha256_<HASH>` identity.
18. **Cross-Device Recovery Node Attachment**: Attach secondary workstation to inspect artifacts and download checkpoints with zero VM re-allocation.
19. **Managed Session Stop**: `python3 -m colab_adapter.cli project stop ./research-project --json`.
20. **Lease Reconciliation & Rollback**: `python3 -m colab_adapter.cli project reconcile --auto-stop --json`. If corruption or network failure occurs, restore from immutable local staging directory (`./artifacts/projects/<project>/runs/<run>/checkpoints/.revisions/<sha16>/model.pt`).

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

## Local Store: Exact-Revision and Latest-Alias Semantics (PR #24, offline)

This section describes `colab_adapter/storage/local.py` on branch `fix/v2-postmerge-hardening-23` (PR #24). All evidence is offline (temporary directories and mock clients). No live Colab or Google Drive run has been validated against this behaviour.

### Layout per run

```
projects/<project>/runs/<run>/
  manifest.json (+ manifest.json.bak)                 authoritative publication manifest
  .locks/manifest.lock                                per-manifest lock (see ARCHITECTURE_V2 section 14)
  checkpoints/<name>                                  mutable alias: cache of the manifest head only
  checkpoints/.revisions/<64-hex full sha256>/<name>  immutable revision file (all new writes)
  checkpoints/.revisions/<16-hex prefix>/<name>       legacy revision directory (read-only support)
```

### 1. Local exact-revision guarantee

- A versioned id (`<project>/<run>/<logical_path>@sha256_<digest>`) must carry a full 64-character lowercase hex digest. Any other length or case is rejected with `ArtifactValidationError` before the filesystem is touched.
- Historical lookup resolves the immutable file from the manifest entry or from `.revisions/<full sha>/`, and falls back to the legacy `.revisions/<sha[:16]>/` directory only for reads. The location check requires a regular, non-symlink file.
- Content is verified against the full SHA-256, not the 16-character prefix:
  - When there is no manifest entry, the file is accepted only if its full SHA-256 equals the requested digest (metadata `recovered_from: revisions_dir`).
  - When there is a manifest entry, the lookup checks existence and type only. `retrieve()` then compares the copied bytes against the recorded size and full SHA-256 before the atomic rename into the destination.
  - Inside a verified lookup, a file mismatch in the full-sha directory raises `ArtifactValidationError` as corruption. A mismatch in the legacy prefix directory is skipped, because it is a different revision that shares the prefix.
- A versioned lookup never falls back to the mutable alias `checkpoints/<name>`. If the exact revision cannot be found, the lookup returns nothing.
- New revisions are written only to `.revisions/<full sha>/`. Directories are created with mode 0700 and files with 0600. The content is staged in a private temporary file and hash-checked before it is linked into place. An existing file with different content is never overwritten. Legacy prefix directories are never written to.

### 2. Latest-alias semantics

- `manifest.json` is authoritative. Its `latest_by_path` map is the only source of the head.
- The alias `checkpoints/<name>` is a recoverable cache: a verified copy of the head revision. It is rewritten only while the manifest lock is held, by staging a verified copy and atomically replacing the alias.
- Unversioned reads resolve the head from the manifest, verify the head revision file, and repair the alias under the lock if its content does not match. A missing head revision file raises `ArtifactNotFoundError`; the alias is never used in its place. Exception: when the head was published into the shared manifest by another provider (for example Google Drive, `remote_file_id` set, no local bytes), the local store returns the manifest metadata with an empty `path`, `exists()` is False and `retrieve()` raises `ArtifactNotFoundError`; the alias is not touched.
- **Mixed providers on one run (scope limit).** The local and Drive stores share one manifest and one lock, so no entry is lost. They do not coordinate the *remote head*: a local publish of bytes that Drive holds as a stale pending revision is a local decision, and the local alias is reconciled only for heads that have local bytes. Consumers must use `get_metadata()`/`retrieve()` for the head, not the alias file.
- Publishing a new revision runs under one locked read-modify-write: install the immutable revision, add it to the manifest, save the manifest, then reconcile the alias. A crash between the manifest save and the alias write is repaired on the next publish or unversioned read.
- Replay of a historical revision (a publish whose digest is already in the manifest but is not the head) verifies the immutable file and returns the record with `is_latest=False`. It leaves the manifest and the alias untouched, so the head never moves.
- Republishing the current head is idempotent. It re-verifies the immutable file and repairs only the alias.
- Manifest loss fails closed. If `manifest.json` and `manifest.json.bak` are both missing while `.revisions/` exists under the run directory, `checkpoints/` or `outputs/`, reads and writes raise `StorageError`. The alias is not used to rebuild the manifest. The legacy plain-file lookup is used only when there is neither a manifest nor any `.revisions/` directory.
- Explicit rollback is not implemented. Republishing or retrieving an older revision does not change the head.

### 3. Crash recovery (local store)

| Crash point | State left on disk | Recovery |
| --- | --- | --- |
| During revision staging | Private temporary file; no final revision file | The temporary file is never linked into place. The next publish of the same content installs it again. |
| After revision install, before manifest save | Immutable revision file present, no manifest entry | A versioned lookup can still return it after full-hash verification. The next publish of the same content re-verifies and reuses it. |
| After manifest save, before alias write | Manifest head is new, alias is stale | The next publish of the same content, or the next unversioned read, repairs the alias under the lock. |
| During alias write | Alias is either the old or the new verified content (`os.replace` is atomic) | Same repair path as above. |
| Lock holder killed | OS releases the `flock` automatically | The next writer acquires the lock. |

A hard kill can leave a temporary `.tmp` file behind, because cleanup runs in a `finally` block. This was not checked against every read path in this review.

### 4. Related limits

- Rollback of the code never deletes local revision files; no code path in this section deletes revisions.
- Verification is offline only. The behaviour above is not evidence of live Colab or Drive behaviour.
