# Colab CLI Server

**Use Google Colab runtimes from a Linux/macOS server or terminal, with human-approved GPU job execution and reusable AI-agent instructions.**

[![CI](https://github.com/anatwork14/colab-cli-server/actions/workflows/ci.yml/badge.svg)](https://github.com/anatwork14/colab-cli-server/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Community project — not affiliated with, sponsored by, or endorsed by Google.** This repository integrates the official [googlecolab/google-colab-cli](https://github.com/googlecolab/google-colab-cli). It is **not** an alternative Colab API, a GPU reseller, an always-on GPU service, or a way to bypass Colab's limits.

## Why this project?

Google's official Colab CLI supports provisioning runtimes and executing local Python code remotely. This project makes those capabilities easier to adopt on **headless Ubuntu/Linux servers**, developer workstations, and **AI-agent-controlled research environments**, while keeping account authentication and accelerator provisioning under user control.

**What you get**
- No-sudo setup using [uv](https://docs.astral.sh/uv/) and the official `google-colab-cli` package (Python 3.12+ for upstream).
- Terminal-based OAuth for machines without a browser.
- A small **standard-library-only** `colab_server.py` entrypoint with read-only diagnostics and an interactive approval gate for one-shot GPU jobs.
- Official commands for sessions, packages, file upload/download, and compute-unit visibility.
- `AGENTS.md`, an agent skill, and a copy-paste agent setup prompt for Codex, Claude Code, and similar tools.
- Offline unit tests and CI that **never allocate Colab resources**.

## Architecture

```text
              Researcher / Developer / Agent
                            |
                       Local server
       +------------------------------------------+
       | colab_server.py (guard + diagnostics)    |
       |          or official colab CLI            |
       +------------------------------------------+
                   |                 |
          Plan / approval       OAuth identity
                   |                 |
                   +--------+--------+
                            |
                 Official google-colab-cli
                            |
                    Google Colab backend
                    |               |
                GPU runtime     CPU/TPU runtime
                    |
               Python scripts
                    |
            Logs / checkpoints / data
```

Colab runs on **another machine**. Its accelerator is not added to the local CUDA device list and its VRAM cannot be pooled with local GPUs. Runtime availability, duration, GPU type, and compute-unit consumption are controlled by Colab.

## Quick start (Ubuntu/Linux)

**Requirements:** Linux or macOS, shell, internet, `git`, and `uv`. You do not need root/sudo or a locally installed CUDA toolkit. The official CLI requires Python >=3.12; `uv` can install it for you. Windows users should run inside a supported Linux environment, subject to upstream compatibility.

```bash
git clone https://github.com/anatwork14/colab-cli-server.git
cd colab-cli-server

# Install official CLI using uv; does not create a GPU runtime
bash scripts/install.sh

# Start account authentication on your terminal (human interaction)
bash scripts/authenticate.sh

# No runtime allocation: check tooling, inspect your plan
python3 colab_server.py doctor
python3 colab_server.py plan --gpu T4 --script examples/gpu_probe.py

# Optional: read-only account balance, may prompt for initial auth
python3 colab_server.py usage

# GPU smoke test: interactively type the requested phrase to approve allocation
python3 colab_server.py run --gpu T4 --script examples/gpu_probe.py
```

`run` invokes upstream `colab run`, which provisions an ephemeral runtime, executes the script and normally releases it. **Every invocation is a new billable/compute-consuming attempt**; do not run this as an unattended retry loop. No GPU is used by `install.sh`, `doctor`, or `plan`.

### Install uv (if missing)

Follow the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/); inspect any remote installer before executing it. The installer here deliberately does **not** curl-and-execute arbitrary scripts on your behalf.

## Official CLI: stateful training + checkpoints

For workloads requiring packages, multiple files, persistent kernel state, or artifact retrieval, use the upstream commands directly:

```bash
# Provision only after explicit user approval
colab --auth=oauth2 new -s example-training --gpu T4

# Install remote dependencies (requirements file is local)
colab --auth=oauth2 install -s example-training -r requirements.txt

# Run local script remotely; use sufficient idle timeout for your workload
colab --auth=oauth2 exec -s example-training --timeout 300 -f train.py

# Retrieve generated checkpoint (must actually exist on remote VM)
colab --auth=oauth2 download -s example-training \
  /content/output/model.pt ./model.pt

# Always release the assigned runtime
colab --auth=oauth2 stop -s example-training
```

The commands above are a **manual workflow**, not an exception-safe wrapper. If any command fails, inspect `colab sessions` and explicitly stop the session. Prefer `colab run` for standalone scripts because it automatically attempts cleanup. **Important:** in the current upstream implementation, `colab exec` can print a Python traceback without necessarily returning a nonzero shell exit status. The adapter's `project exec` reports kernel tracebacks as failures, but it still does not prove the job's outputs are correct: verify remote completion and artifact correctness, and do not infer success solely from `exec`'s exit code.

## Agent-assisted setup

An agent may install tools, inspect the host, write job files, run **offline tests**, and present a plan. An agent **must not** authenticate on behalf of a user, allocate/retain GPUs, or consume compute units without explicit authorization for the specific operation.

Start here:
- [Agent policy](AGENTS.md)
- [Agent installation workflow](docs/AGENT_SETUP.md)
- [Copy-paste agent setup prompt](docs/AGENT_SETUP_PROMPT.md)
- [Agent skill](skills/colab-cli-server/SKILL.md)

**Approval gate limits (plain language).** The gate stops accidental or agent-initiated Colab allocation through this adapter. Each approval is bound to one exact plan (project, GPU, manifest and workspace contents), a maximum lease, a 15-minute expiry and single use, and it leaves an audit record. It does **not** stop a process that runs as the same OS user: such a process can read the record key, forge or edit records, drive a pseudo-terminal to answer the prompt, run `colab new` directly, or change the code. Direct use of `colab new`, `colab run` and `colab ssh` cannot be gated by this repository. The gate is a safety interlock, not a security boundary. The real boundary is the approver and Colab OAuth in a separate OS account or a protected executor. Until that exists, the operator should run `project up` personally for paid runtimes.

**What the approval fingerprint covers.** The workspace part of the fingerprint hashes every regular file under the project root except `artifacts_local_dir` (default `./artifacts`), the `.colab_experiments` state directory, `*.log` files, `__pycache__`, `*.pyc` and `.git`. Changing any covered file (for example a source file) requires a new approval. Pulling artifacts or writing logs does not, so an existing session can still be reused. `project plan` output is unchanged.

**Global cap and `--no-reuse`.** One managed session is allowed at a time. `project up --no-reuse` while any managed session is live always fails the cap (exit 4) and leaves the approval unconsumed: stop or reconcile the live session first. `colab_server.py run` follows the same cap: it registers a managed record before `colab run`, and the record stops counting toward the cap once the run finishes.
**Host-side lease guard (best effort; not a billing cap).** `project up` starts a detached `project lease-guard` process for that one session (and restarts it if it died, on `project exec` or `project sync`). At the lease deadline the guard calls `colab --auth=<mode> stop -s <that id>` only, then verifies the removal. Failed stops are retried with exponential backoff capped at 30 s and the guard does **not** give up on its own: the first failure flags the record `NEEDS_RECONCILIATION`, and retries continue until the stop is verified, the record becomes terminal, or the guard receives SIGTERM. A busy project lock is waited for and is not counted as a failed attempt. The guard never touches other sessions. **Failed allocations fail closed.** A nonzero `colab new` exit is *not* evidence that no VM was assigned (upstream assigns the VM before it records the local name). `project up` therefore records `FAILED` (which frees the one-session cap) **only** with positive evidence: a usage baseline read before `new`, and two consecutive clean readings afterwards (`colab sessions` readable and not listing the id, `colab usage` readable and exactly equal to the baseline, and no more unnamed `[?]` runtimes than before `new`; a different count means something else changed concurrently and is not proof). A missing baseline, unreadable `sessions`/`usage`, a count above the baseline, a lagging reading, a timeout, an interrupt or any unexpected exception leaves the record `NEEDS_RECONCILIATION` with `orphan_possible`, keeps counting toward the cap, spends the approval, prints a *possible orphan GPU runtime* warning, and never allocates again by itself. Only a failure to start the `colab` executable at all (`new` never ran) is recorded as `FAILED` without readings. The CLI exits **6** (JSON `orphan_possible: true`, with `session_id`) when an allocation failed or was interrupted and a runtime may still exist, and exits 1 only when absence was proven; Ctrl-C during `project up` prints the same warning and exits 6. Resolve an uncertain record with the operator procedure in `docs/live/GPU_TEST_PLAN.md` (check `usage` Active assignments and the Colab UI). **Scope:** this guarantee covers `project up` only. The older V1 paths (`colab_adapter.cli run` / `ColabGPUProvider.submit_job`, and `colab_server.py run`) can still record a run as finished or free the shared cap after `colab stop` exits 0 or after a failed `colab run` without server-side proof; they are not used by the live plan and must not be used for live tests until they get the same treatment. The proof is not absolute: a lagging provider-side count, or one unmanaged runtime ending while another is orphaned, can still fool it, so the live plan requires a baseline of 0 active assignments.

Verification of a stop, in order: `colab stop` must exit 0 without a "not found" message (upstream exits 0 and prints `[colab] Session '<id>' not found.` for an unknown name, so that case is recorded as `NEEDS_RECONCILIATION`, never `TERMINATED`); `colab sessions` must no longer list the id; and, when `up` recorded an `assignments_before` baseline from `colab usage` before `colab new`, the `Active assignments` count from `colab usage` must be back at or below that baseline. Without a baseline the record says `verification: weak` (sessions check only). A stop that exited 0 but failed the usage check is never re-issued; only verification is retried.

The guard does **not** survive a host reboot, a `kill -9` of the guard, or loss of OAuth, and it is not a guarantee: the lease clock starts at `up`, so provisioning time counts against it. After every test, check `project status` (exit code 5 means needs attention: `NEEDS_RECONCILIATION`, `STOPPING`, or a dead guard under an active lease), `sessions` and `usage` yourself. `project reconcile` (auto-stop is on by default; `--no-auto-stop` only audits) is the backstop: it handles expired leases, `NEEDS_RECONCILIATION`, `STOPPING` and `PROVISIONING` records older than 300 s, managed ids only, and prints a warning whenever a `NEEDS_RECONCILIATION` record exists.

**Do not Ctrl-C `project up`.** If `colab new` is interrupted (timeout, Ctrl-C) it may already have assigned a VM before the adapter recorded its local name. The record is kept as `NEEDS_RECONCILIATION` and still counts toward the cap. Upstream `colab sessions` shows such an unnamed VM only as a line starting `[?] <endpoint>`. Check `colab --auth=<mode> usage` (`Active assignments: N`) and the Colab UI, and do not assume the adapter has cleaned it up. The adapter does not clear `NEEDS_RECONCILIATION` automatically.

**`project exec` takes Python.** The argument is PYTHON code run by the remote kernel, because `colab exec -f` runs its file as kernel Python. `--working-dir` becomes an `os.chdir` prologue (absolute remote path, no `..`, no control characters). `--shell` runs the argument as a shell command on the managed VM only, with `cwd` set from `--working-dir`, and prints stdout, stderr and an `EXIT_CODE <n>` marker. The upstream exit status of `colab exec` is not the command's exit status, so success requires the marker to be present and equal to 0; a missing marker is reported as failure. Kernel tracebacks are reported as failure even though upstream exits 0. `project exec --timeout` (default 600 s) is passed to `colab exec --timeout` minus a 15 s margin and clamped to the remaining lease.

## Documentation

| Guide | What it covers |
| --- | --- |
| [Installation](docs/INSTALLATION.md) | Ubuntu/macOS, no sudo, prerequisites, official Git source |
| [Authentication](docs/AUTHENTICATION.md) | Headless OAuth, identity, ADC alternative, protecting tokens |
| [Architecture](docs/ARCHITECTURE.md) | Components, lifecycle, compute-unit and capability boundaries |
| [Jobs and artifacts](docs/JOBS.md) | One-shot vs. stateful execution, datasets, checkpoints |
| [Adapter & Provider](docs/ADAPTER.md) | Modular ColabGPUProvider, experiment tracking, CLI adapter |
| [Agent setup](docs/AGENT_SETUP.md) | Readiness gates and validation checklist |
| [Security](docs/SECURITY.md) | Public-repo safety, approvals, credentials, threat model |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Typical failures, allocation, runtime loss, cleanup |

## Limitations

- Paid Colab subscriptions do **not** guarantee a specific GPU, uptime, quota, or indefinite runtime.
- Remote execution and data transfer can incur latency; this is better for bounded research jobs than latency-critical production inference.
- Do not use this project to bypass idle detection, resource constraints, Colab policy, or authorization.
- Keep credentials and data out of Git; never execute untrusted remote research scripts on your connected account.

The official upstream source, not this wrapper, is the authority for supported flags:
[README](https://github.com/googlecolab/google-colab-cli#readme) · [Colab Terms](https://research.google.com/colaboratory/tos_v6.html).

## Development

```bash
python3 -m unittest discover -s tests -v
bash -n scripts/install.sh scripts/authenticate.sh
```

Tests mock subprocesses and never contact Google or allocate a runtime. Contributions are welcome; read [CONTRIBUTING.md](CONTRIBUTING.md).

## License and attribution

This wrapper and its documentation are licensed under [MIT](LICENSE). Google's `google-colab-cli` is a separate upstream project governed by its own Apache-2.0 license and Colab service terms. No upstream code is vendored here.

## V2 — VKong-Inspired Project Workspaces and Training Monitor

> **Status:** Colab CLI Server V2 (Milestones M1–M8, Google Drive Durable Storage & Cross-Device Workspaces) was merged into `main` via PR #22 (`670c6bc` / `a1629f9846f19c8d5d4e90469ee9e376b64d98e9`). Post-merge hardening and multi-revision catastrophic recovery (Review `5642047783` F1–F4) are delivered in **Issue #23 (`fix/v2-postmerge-hardening-23`)**. Status: merged. A first live validation was run on 2026-10-26; the success path is verified with limitations (verdict `LIVE_SUCCESS_PATH_VERIFIED_WITH_LIMITATIONS`, see [docs/live/LIVE_TEST_EVIDENCE.md](docs/live/LIVE_TEST_EVIDENCE.md)). This is not production readiness and not failure-path verification. All human approval gates remain in force.

The V2 implementation extends the original one-shot adapter into a comprehensive project-centric remote GPU workspace system and real-time training monitor:

- **Declarative Project Manifests (`colab-project.json`):** Declare source patterns, entrypoints, target accelerators, and artifact rules with offline validation and path escape rejection.
- **Content-Addressed Incremental Sync:** Two-phase workspace synchronization transfers only modified files via streaming chunked SHA-256 diffing, with atomic commit and opt-in remote file deletion.
- **Collision-Resistant Session Lifecycle:** Project runtime leases (`colab-v2-<name>-<uuid>`) prevent collisions, enforce state machine transitions, lock against concurrent allocations, and fail-closed on unmanaged user sessions.
- **Multi-Tier Checkpoint & Immutable Revision Recovery:** Staged atomic download with integrity checksums, ZIP header verification, safe PyTorch `weights_only=True` inspection, and immutable `@sha256_<digest>` revision history (`PublicationManifest` v2) preserving both latest and historical checkpoints across local and Google Drive storage.
- **Private Cross-Device Training & Compute Monitor:** Read-only HTTP/SSE dashboard displaying compute unit balance and burn rates when the usage poller has data (otherwise shown as *Unavailable*; the poller and display fixes in PR #32 have not been retested live), top-up detection, unattributed spend alerts, run loss/step charts, and checkpoint revision history (`LATEST` vs `HISTORICAL`), hardened with per-response cryptographic CSP nonces and verified in headless Chromium CI.

### Project Workspace CLI Commands

```bash
# Initialize a new project directory with template manifest and entrypoint
python3 -m colab_adapter.cli project init --root ./my-project --name my-project

# Validate manifest schema, path safety, and file existence
python3 -m colab_adapter.cli project validate --manifest ./my-project/colab-project.json --json

# Compute dry-run file sync diff and session plan without allocating compute
python3 -m colab_adapter.cli project plan --manifest ./my-project/colab-project.json --json

# Provision or attach to a remote project session under cross-process lock
python3 -m colab_adapter.cli project approve-request --manifest ./my-project/colab-project.json --lease-seconds 3600 --json
python3 -m colab_adapter.cli project approve <approval-id>          # operator, in their own terminal
python3 -m colab_adapter.cli project up --manifest ./my-project/colab-project.json --lease-seconds 3600 --approval-id <approval-id> --json

# Incrementally sync workspace source tree to the remote session
python3 -m colab_adapter.cli project sync --manifest ./my-project/colab-project.json --session-id <session-id> --json

# Run PYTHON code on the remote kernel (project exec takes Python, not a shell command)
python3 -m colab_adapter.cli project exec "import runpy; runpy.run_path('train.py', run_name='__main__')" --working-dir /content/colab-workspaces/my-project --session-id <session-id> --json

# Run a shell command on the managed VM (exit code is read from the EXIT_CODE marker)
python3 -m colab_adapter.cli project exec --shell "python3 train.py --epochs 10 --batch-size 32" --working-dir /content/colab-workspaces/my-project --session-id <session-id> --json

# Inspect remote project session status and lease TTL (exit code 5 = needs attention)
python3 -m colab_adapter.cli project status --session-id <session-id> --json

# Safely terminate project session
python3 -m colab_adapter.cli project stop --session-id <session-id> --json

# Reconcile expired project leases and release resources
python3 -m colab_adapter.cli project reconcile --json
```

All commands support `--json` for machine-readable output by autonomous agents and scripts.

### Training & Compute Monitor CLI Commands

```bash
# Start local authenticated monitoring server with web dashboard (loopback default)
COLAB_MONITOR_TOKEN="$(cat /path/to/private/monitor.token)" python3 -m colab_adapter.cli monitor --host 127.0.0.1 --port 8080

# Start monitoring server with TLS encryption or behind a verified identity proxy
COLAB_MONITOR_TOKEN="$(cat /path/to/private/monitor.token)" python3 -m colab_adapter.cli monitor --host 0.0.0.0 --port 8443 --allow-external \
  --require-tls \
  --tls-certfile /path/to/cert.pem --tls-keyfile /path/to/key.pem
```

## V2 Operator Experience Runbook (20-Step Guide)

A step-by-step operator guide using strictly real, supported CLI commands (`python3 colab_server.py ...` and `python3 -m colab_adapter.cli ...`):

1. **Host Environment Verification (No sudo)**:
   Verify host Python version and uv toolchain:
   ```bash
   python3 --version
   uv --version
   ```
2. **Automated Upstream CLI Installation (User Mode)**:
   Install the official upstream `google-colab-cli` in user space without root:
   ```bash
   bash scripts/install.sh
   ```
3. **Headless OAuth Account Authentication**:
   Perform browserless terminal-based Google authentication:
   ```bash
   bash scripts/authenticate.sh
   ```
4. **Environment & Diagnostic Health Inspection (Zero Compute Allocation)**:
   Verify local Python environment, CLI capability detection, and account readiness:
   ```bash
   python3 colab_server.py doctor
   # or: python3 -m colab_adapter.cli doctor
   ```
5. **Compute Units & Session Balance Audit**:
   Inspect current account compute-unit balance and running sessions without allocating:
   ```bash
   python3 colab_server.py usage
   python3 -m colab_adapter.cli sessions
   ```
6. **Project Workspace Initialization**:
   Create a structured research project directory with manifest template:
   ```bash
   python3 -m colab_adapter.cli project init --root ./research-project --name research-project
   ```
7. **Source & Dependency Preparation**:
   Populate local project code (`train.py`, `model.py`, `requirements.txt`). Excluded secrets (e.g. `.env`) remain strictly local.
8. **Project Manifest Validation**:
   Validate manifest schema, root path containment, and secret exclusion rules:
   ```bash
   python3 -m colab_adapter.cli project validate --manifest ./research-project/colab-project.json --json
   ```
9. **Zero-Allocation Offline Execution Plan**:
   Generate deterministic transfer diff, file count, and payload size without cloud allocation:
   ```bash
   python3 -m colab_adapter.cli project plan --manifest ./research-project/colab-project.json --json
   ```
10. **Human-Approval-Gated Session Provisioning**:
    Provision or attach to a remote Colab session with an explicit approval token:
    ```bash
    python3 -m colab_adapter.cli project approve-request --manifest ./research-project/colab-project.json --lease-seconds 3600 --json
python3 -m colab_adapter.cli project approve <approval-id>          # operator, in their own terminal
python3 -m colab_adapter.cli project up --manifest ./research-project/colab-project.json --lease-seconds 3600 --approval-id <approval-id> --json
    ```
11. **Initial Workspace Incremental Synchronization**:
    Sync workspace sources to the remote session with streaming SHA256 content indexing:
    ```bash
    python3 -m colab_adapter.cli project sync --manifest ./research-project/colab-project.json --session-id <session-id> --json
    ```
12. **Remote Training Execution**:
    Execute the remote training entrypoint with live streaming output:
    ```bash
    python3 -m colab_adapter.cli project exec --shell "python3 train.py --epochs 10 --batch-size 32" --working-dir /content/colab-workspaces/my-project --session-id <session-id> --json
    ```
13. **Incremental Code Synchronization (Hot Updates)**:
    Modify local code and re-synchronize; only modified files transfer:
    ```bash
    python3 -m colab_adapter.cli project sync --manifest ./research-project/colab-project.json --session-id <session-id> --json
    ```
14. **Cross-Device Monitoring Dashboard Startup**:
    Launch the read-only web monitoring server with per-response CSP nonces:
    ```bash
    COLAB_MONITOR_TOKEN="$(cat /path/to/private/monitor.token)" python3 -m colab_adapter.cli monitor --host 127.0.0.1 --port 8080
    ```
15. **Google Drive Credential Setup (0600 Permissions)**:
    Set up Google Drive API credentials with private filesystem permissions:
    ```bash
    # Stored at ~/.config/colab-cli/drive_credentials.json with mode 0600
    chmod 0600 ~/.config/colab-cli/drive_credentials.json
    ```
16. **Multi-Revision Checkpoint Publishing**:
    Publish successive training epochs (`model.pt`) to Google Drive; each revision is preserved under immutable `@sha256_<HASH>` identity in `PublicationManifest` v2 with immediate-successor `superseded_by` chain (`A -> B -> C -> D -> E -> None`). Remote Google Drive custom properties adhere to the strict 124 UTF-8 bytes limit (key + value, max 30 properties, citing [Google Drive API limits of custom file properties](https://developers.google.com/workspace/drive/api/guides/properties#limits_of_custom_file_properties)) enforced via `validate_drive_properties` before every remote write. Drive carries compact schema v2 (`schema: "2"`, `sha256`, `path_sha256`, `is_latest`, `next_sha256`), while legacy keys (`logical_path`, `revision_id`, `project_id`, `run_id`, `superseded_by`) are read-only for backward compatibility and cleaned up on PATCH. Republishing historical revisions is strictly idempotent (no data transfer, no PATCH); explicit rollback is not implemented.
17. **Historical & Latest Checkpoint Recovery**:
    Retrieve latest weights via logical path or specific historical checkpoint revision (`@sha256_<HASH>`) with bitwise verification; catastrophic local state loss safely falls back to remote Drive listing with fail-closed integrity checks (ambiguous 0 or >1 remote latest markers raise `DriveIntegrityError` fail-closed):
    ```bash
    python3 -c "from colab_adapter.storage import GoogleDriveArtifactStore, DriveCredentials; store = GoogleDriveArtifactStore(credentials_file='~/.config/colab-cli/drive_credentials.json'); store.retrieve('research-project/<run_id>/checkpoints/model.pt', './model_latest.pt')"
    ```
18. **Cross-Device Workspace Attachment**:
    Attach a secondary terminal or developer machine to observe training runs via the authenticated Workspace Broker without creating duplicate Colab VMs.
19. **Managed Session Teardown**:
    Safely stop the active project session and release cloud resources:
    ```bash
    python3 -m colab_adapter.cli project stop --session-id <session-id> --json
    ```
20. **Lease Reconciliation & Rollback / Troubleshooting**:
    Reconcile expired sessions, `NEEDS_RECONCILIATION`/`STOPPING` records and stale `PROVISIONING` records (managed ids only). Read every `WARNING:` line: a `NEEDS_RECONCILIATION` record may hide an unnamed orphan VM that only `colab usage` (active assignments) and the Colab UI can confirm:
    ```bash
    python3 -m colab_adapter.cli project reconcile --json
    ```
    Rollback: If network or recovery fails, inspect `transfer_journal.json`, verify hashes, and fall back to local staged checkpoints.

**Testing & Verification Demarcation:**
- All features and failure recovery modes are verified offline using deterministic mock clients (`OFFLINE_RELEASE_READY`).
- Live validation status: on 2026-10-26 the owner-approved success path was exercised against real Google Drive (D1-D5, `drive.file` scope) and real Colab (one bounded T4 session, a lease-guard deadline drill, a second-device monitor check and a Drive publication of the GPU checkpoint); see [docs/live/LIVE_TEST_EVIDENCE.md](docs/live/LIVE_TEST_EVIDENCE.md). Real-provider failure paths (nonzero `colab new`, orphan or `[?]` runtimes, Drive 403/429), a single continuous GPU-to-Drive run and a post-#32 live monitor retest are still unverified. Every live OAuth exchange, Drive write and GPU allocation still requires explicit, specific operator approval (`LIVE_VALIDATION_REQUIRES_APPROVAL`); the lease guard is a best-effort host-side deadline, not a billing cap.

**Security & Account Protection:**
- Poller executes strictly read-only official CLI queries (`colab usage`, `colab sessions`) and **never** provisions cloud VMs.
- Web dashboard is strictly read-only: no remote code execution, stop, or approval endpoints.
- Binds to localhost by default; access from other machines should use an SSH tunnel or VPN.

**Documentation:**
- [Progress ledger](progress.md) — Completed milestones, evidence, and change logs
- [Milestones](milestones.md) — M0–M8 and G-Drive release gates and evidence requirements
- [Google Drive Artifacts Guide](docs/GOOGLE_DRIVE_ARTIFACTS.md) — Resumable upload, verification, and private storage hierarchy
- [Cross-Device Workspaces Guide](docs/CROSS_DEVICE_WORKSPACES.md) — Multi-client read-only attach via Workspace Broker
- [Artifact Recovery & Reproducibility](docs/ARTIFACT_RECOVERY.md) — Bit-reproducible code snapshots and disaster recovery
- [V2 features](docs/FEATURES_V2.md) — Detailed feature specification
- [V2 architecture](docs/ARCHITECTURE_V2.md) — Security contracts and threat model
- [Expected results](docs/EXPECTED_RESULTS_V2.md) — Measurable acceptance criteria
- [Cross-device Monitor Guide](docs/MONITORING_DASHBOARD_V2.md) — Dashboard architecture and API specification

