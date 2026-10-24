# Feature Specification — Colab Project Workspaces V2

**Status:** PLANNED, not implemented. **Audience:** maintainers, research agents and human reviewers. **Purpose:** deliver a VKong-inspired GPU-rental developer experience backed by Google's **official** Colab CLI. This project is independent of VKong and does not integrate or resell its service.

## Product problem

Today ColabGPUProvider can send a local script to a remote VM and can upload explicitly selected inputs in a managed stateful job. A realistic research repository has Python modules, packages, configs, datasets, checkpoints and repeated edits. Users need to develop locally, send only changed files, execute on a leased GPU, inspect failures and download results, without manually creating or navigating Colab VMs.

**Target workflow:** Research agent edits a project on Ubuntu -> generates a safe plan -> human approves any new/retained runtime -> synchronize project workspace to the approved existing runtime -> run remotely with CUDA -> stream logs -> verify/download artifacts -> automatically release GPU unless session retention was separately approved.

**Not the goal:** Directly forwarding local CUDA calls, pooling VRAM with local GTX GPUs, guaranteed SSH availability, guaranteed A100/H100 access, container control/Docker images, 24/7 inference hosting, evading Colab policy, reproducing VKong's vendor billing/storage infrastructure.

## Personas and primary scenarios

- P1 **Researcher** trains/evaluates a multi-module PyTorch project from a local Ubuntu path with remote GPU.
- P2 **Assistant agent** proposes commands and diffs, runs offline checks, prepares approvals; may not self-authorize cloud spend.
- P3 **Operator** sees session/cost estimates, grants single-job approval, monitors and stops only owned sessions.
- P4 **Maintainer** debugs failure with reproducible local mocks and an evidence-backed history.

## Feature requirements and priority

| ID | Priority | Feature | Core behavior | Acceptance |
| --- | --- | --- | --- | --- |
| F01 | MUST | Versioned project manifest | Root, entrypoint, GPU, Python deps, input/output map, ignores, explicit retention; valid paths | Schema + negative fixtures, no VM |
| F02 | MUST | Plan/dry-run | Render selected files, hashes, transfer bytes estimate, runtime/cost uncertainty and required approvals | Zero allocating CLI calls |
| F03 | MUST | Existing-session transport | Managed session resolution, official upload/exec/download fallback | Missing/unowned session safely rejected |
| F04 | SHOULD | SSH-over-WebSocket | Probe official colab ssh --proxy-mode capability, optionally shell/rsync | 404/401/429 and missing SSH graceful fallback |
| F05 | MUST | Incremental source sync | Content hash diff, skip unchanged, atomic apply where feasible | One changed file -> one changed file |
| F06 | MUST | Safe managed workspace | Canonical remote root, secret excludes, symlink rules, opt-in deletion, protected remote manifest | No path escape/no unexpected deletion |
| F07 | MUST | Approval-aware runtime lifecycle | Up/attach/execute/stop semantics; explicit approval for provisioning/retention | No allocate from plan/attach/status/sync |
| F08 | SHOULD | Remote debug loop | Exec on approved runtime, logs, sync changes, read-only status; no implicit restart | Deterministic mock attach/retry |
| F09 | MUST | Artifact verification | Outputs/manifest hashes, safe checkpoint load, host storage outside ephemeral VM | Interrupted + resumed scenario |
| F10 | SHOULD | Observability | JSON history, machine-readable outcomes, bytes/files sync, time, cleanup state | Structured records; no secrets |
| F11 | SHOULD | Resource/cost controls | Max active owned sessions, lifespan, no unattended retries, read-only usage | Quota/cleanup tests; no price guarantees |
| F12 | MAY | Optional remote dependency cache | Reinstall only when manifest fingerprint changes; lifecycle scoped | Cache invalidates by hash |
| F13 | LATER | Provider abstraction | Local/Colab/dedicated rented GPU contract with real provider-specific semantics | Separate design, no placeholder calls |
| F14 | LATER | Watch mode | Debounced sync only while approved session active | Must not extend runtime implicitly |

## Proposed manifest contract (illustrative, not currently parsed)

~~~yaml
schema_version: 1
project:
  name: pytorch-demo
  root: .
runtime:
  provider: colab
  gpu: T4
  timeout_seconds: 900
  retain_session: false  # default is teardown; true requires separate approval
workspace:
  remote_root: /content/colab-workspaces/pytorch-demo
  include: ["src/**", "configs/**", "train.py", "requirements.txt"]
  exclude: [".git/**", ".venv/**", "**/__pycache__/**", ".env*", "**/*token*"]
  delete_remote: false
execution:
  entrypoint: train.py
  requirements: requirements.txt
  args: ["--config", "configs/default.yaml"]
artifacts:
  pull: ["outputs/status.json", "outputs/checkpoint.pt"]
  local_dir: ./artifacts
~~~

**Parsing decision:** support real YAML via an explicitly declared optional library with pinned/tested version, or use JSON for stdlib support. Never write a partial/surprising YAML parser. Exact field names/CLI flags can be refined in M1/M2 and then frozen with schema_version.

## Proposed operator interface (NOT implemented)

~~~bash
colab-adapter project init --root .
colab-adapter project validate --manifest colab.yaml
colab-adapter project plan --manifest colab.yaml --json  # offline, no VM
colab-adapter project up --manifest colab.yaml            # explicit human approval, new owned session
colab-adapter project sync --manifest colab.yaml --plan    # no allocation
colab-adapter project sync --manifest colab.yaml --apply   # existing approved session ONLY
colab-adapter project exec --manifest colab.yaml           # existing approved session ONLY
colab-adapter project status --manifest colab.yaml
colab-adapter project logs --manifest colab.yaml
colab-adapter project stop --manifest colab.yaml
~~~

The actual executable interface must enforce the approval contract **before** invoking any upstream CLI command capable of creating a session. The official ssh command may itself auto-create when the requested session is absent; command examples do not grant permission for this behavior.

## Functional semantics

### F01/F02 — plan the remote execution
- Resolve all selected file paths relative to a canonical project root. Paths must be confined to root, with explicit handling for symlinks; deny device/socket files and dangerous symlink targets.
- Produce manifest schema_version, project ID, file count, changed bytes, GPU selection, entrypoint, remote working directory, retained-session TTL, expected outputs and external-data references.
- Distinguish **estimates** (to transfer) from **observations** (bytes actually sent) and compute-unit balance snapshot from true billed cost. No invented prices.

### F03/F04 — transport strategy
- Preferred when safely available: SSH proxy transport to a previously authorized *managed* session; ordinary remote commands and file sync can use OpenSSH + rsync when binaries/protocol work.
- Baseline guaranteed by the wrapper: upstream managed new/exec/upload/download/stop operations, subject to upstream availability. If SSH is unsupported or cannot be guaranteed no-auto-provision, degrade to official CLI uploads/exec.
- No direct exposure of ssh ProxyCommand to untrusted same-UID agents. Do not assume preflight followed by ssh -s is race-free; design an enforceable reservation boundary or disable SSH in untrusted mode.
- Preserve integrity, error and cancellation semantics in both transports.

### F05/F06 — incremental synchronization
- Maintain immutable local hash manifest plus remote **last committed** manifest. Diff into ADD/MODIFY/UNCHANGED/DELETE.
- Default: upload ADD/MODIFY only. UNCHANGED = 0 transfer. DELETE is opt-in and may only operate under the adapter-owned remote_root, with an explicit preview and ownership proof.
- Two-phase remote staging + verification + manifest commit; interrupt leaves previous manifest and files recoverable. Avoid partial overwrite of entrypoint.
- Reject attempts to sync .git, .env, credential stores, SSH keys, sensitive path patterns, symlinks escaping project root. Never implement transparent home-directory sync.
- External large datasets are referenced/mounted/cached only with explicit capability and privacy approval. No automatic Google Drive OAuth or mounting.

### F07/F08 — session model
- States: PLANNED, AWAITING_APPROVAL, PROVISIONING, READY, SYNCING, RUNNING, COLLECTING, STOPPING, TERMINATED, FAILED, NEEDS_RECONCILIATION; transitions checked and persisted.
- Sessions retain project/job identity, owning PID/process start, created/last activity, requested accelerator, approved retention deadline, and operator disposition.
- Default session lifetime = the bounded job lifetime; retained session is separately approved and stopped by deadline. No ping/keepalive to bypass provider idling.
- Never stop unrelated user Colab assignments, never start a new VM because SSH alias or attach refers to a missing name.

### F09/F10/F11 — artifacts, diagnostics and guardrails
- Download requested output files with existence, size, optional content digest and expected status verification. Keep output provenance with project revision, source manifest digest, environment digest and session ID.
- Detect remote Python tracebacks even when the CLI process exits zero; use explicit structured status output for training; classify infrastructure errors separately.
- Record control actions and redacted summaries; default to private 0700 directories and 0600 files.
- Guard failed SSH transports, quota denial, runtime termination, transient network errors, timeout, disk full, partial downloads, missing checkpoint, session stop failure and corrupted registry.
- Tests must not allocate resources, perform OAuth, mutate real server state or use live external services.

## Compatibility and migration
- Preserve ColabGPUProvider and existing colab-adapter doctor/usage/plan/run/history/stop behavior. New project commands are additive.
- Host-side adapter must remain usable under Python 3.8; official google-colab-cli v0.7.x runs in an isolated supported Python >=3.12 tool environment.
- Avoid installing local GPU/CUDA dependencies or breaking GTX 1080 Ti workloads.
- Existing tracker data and registry must remain readable; migration is explicit, backed up and idempotent.

## External references
- Google upstream SSH design: https://github.com/googlecolab/google-colab-cli/blob/main/docs/06_ssh_access.md
- Official CLI operator skill: https://github.com/googlecolab/google-colab-cli/tree/main/skills/colab-operator
- Google Colab runtime restrictions: https://research.google.com/colaboratory/faq.html

**Important:** The sources prove *upstream documented capabilities*, not that SSH works on the user's particular paid Colab account/runtime. M3 tests that separately after human authorization.
