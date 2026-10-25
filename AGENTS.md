# Agent instructions — Colab CLI Server

These instructions apply to autonomous and assisted coding agents working **in this repository** or installing it on an end-user's Linux/macOS host.

## Mission

Make the official [googlecolab/google-colab-cli](https://github.com/googlecolab/google-colab-cli) usable from a terminal or headless server with clear setup, testability, responsible runtime consumption and vendor-neutral agent documentation. Do not replace or fork Google's API implementation when an official CLI command already exists.

## Non-negotiable operating rules

1. **Inspect before changing.** Check OS, available commands, `python3 --version`, `uv --version`, `colab --help`, and existing Colab sessions where authentication is already configured. Never print credentials.
2. **No cloud spending without permission.** `colab new`, `colab run`, `colab ssh` when it auto-provisions, and retaining live runtimes require explicit per-job user authorization. **Do not call these commands** during discovery, installation, CI, or tests.
3. **No bypassing approval.** Never script answers to the interactive prompt, forge approval through stdin/PTY, set a self-approved environment variable, or impersonate a user. If you are an agent in the same host account, the interactive prompt is not a strong policy boundary: request external approval from the human and use a separately protected executor for autonomy.
   **Plain limits of the approval gate:** a process running as the same OS account can read the HMAC key, forge records, drive `/dev/tty` through a pty, call `colab new`, `colab run` or `colab ssh` directly, and edit code or delete registry entries to free the cap. The gate is an interlock, not a security boundary. The operator should run `project up` (or `colab_server.py run`) personally for paid runtimes.
4. **Human-only sign-in.** Run the supported headless OAuth flow in a terminal when asked, then let the user complete the Google consent. Never solicit or store authorization codes, OAuth tokens or credentials in prompts, logs, Git, CI, tool outputs or agent memory.
5. **User account ownership.** Never use service-account impersonation to claim access to someone's paid Colab entitlement; authenticate as the actual authorized subscriber.
6. **Default cleanup.** Prefer `colab run` for a self-contained job; it normally tears down automatically. For `colab new` sessions, stop the specific session in `finally` on success, error and cancellation where possible, then verify active assignments. Never create keepalive/ping loops.
7. **Keep public code generic.** Never commit hostnames, SSH details, GPU inventory, private paths, subscription balance, raw experiment data, proprietary assets or private endpoints. Place examples in generic paths.
8. **Offline tests only by default.** Mock all Colab interactions. Live CPU/GPU probes need separate approval, and accelerator probes can consume credits.
9. **Do not claim a job succeeded solely because `colab exec` exited 0.** Verify expected output files/checkpoints and inspect kernel output; upstream `exec` may not propagate all remote exceptions as process failure. For `project exec`, the argument is PYTHON code run by the kernel; use `--shell` only for shell commands, and read the `EXIT_CODE <n>` marker (the upstream exit status is not the command's exit status). A missing marker is a failure.
10. **Never auto-merge or deploy to other hosts.** Use feature branches and PRs. Server installation requires environment access and the host owner's consent.

## Installation phases

- **Phase A — inspect (read-only):** Determine shell/OS, whether `uv` exists, verify Python compatibility for Google's package (>=3.12), check Git clone status and whether a local `colab` executable already exists.
- **Phase B — install (local changes only):** Review [docs/INSTALLATION.md](docs/INSTALLATION.md), execute `bash scripts/install.sh`, check `colab version` and `colab --help`. Installation is *not* permission to allocate cloud resources.
- **Phase C — authenticate (human handoff):** Explain `bash scripts/authenticate.sh`; the user completes login. Confirm only success/failure and scopes/identity if needed; redact everything credential-bearing.
- **Phase D — dry run (free):** `python3 -m unittest discover -s tests -v`, `python3 colab_server.py doctor`, and `python3 colab_server.py plan --gpu T4 --script examples/gpu_probe.py`. These must not allocate VMs.
- **Phase E — live probe (optional):** Ask which GPU/runtime and whether compute use is approved. A human runs `python3 colab_server.py run --gpu T4 --script examples/gpu_probe.py` and types the exact consent phrase. Check `colab --auth=oauth2 sessions` afterward and clean up any orphan.
- **Phase F — application integration:** Use the modular adapter in `colab_adapter` (documented in [docs/ADAPTER.md](docs/ADAPTER.md)). It enforces cross-process locking, input tampering checks, and session registry ownership. Place human approval in an independent protected control plane or separately protected executor before creating billable runtimes. Never automatically stop unrelated user sessions.

## Completion / acceptance criteria

- README is provider-neutral and complete; setup works without sudo if prerequisites are available.
- Upstream CLI is installed from an explicit, documented source, not vendored.
- `doctor` and `plan` make no network allocations.
- `run` refuses non-interactive execution and incorrect confirmation.
- No secrets or personal identifiers in tracked files.
- Tests and CI run offline; shell syntax checks pass.
- Docs explain OAuth vs VM-side `colab auth`, session expiration, quota and cleanup.
- Final report states what was verified locally, what was not tested against Colab, and whether a remote VM was used.

See [agent setup steps](docs/AGENT_SETUP.md) and the [copyable prompt](docs/AGENT_SETUP_PROMPT.md).

## V2 project workspace operations and agent guidelines

> **V2 Release Status:** Colab CLI Server V2 was merged into `main` via PR #22 (`670c6bc` / `a1629f9846f19c8d5d4e90469ee9e376b64d98e9`). Post-merge hardening and multi-revision catastrophic recovery (Review `5642047783` F1–F4) are delivered in **Issue #23 (`fix/v2-postmerge-hardening-23`)**. Status: `READY_FOR_INDEPENDENT_MERGE_REVIEW pending CI only`. All human approval gates strictly preserved (`LIVE_VALIDATION_REQUIRES_APPROVAL`).

V2 adds project workspace workflows (`python3 -m colab_adapter.cli project ...`) and compute monitoring (`python3 -m colab_adapter.cli monitor ...`). All autonomous agents operating in this repository or interacting with Colab must adhere to these rules:

1. **Always plan and validate offline first:**
   Use `python3 -m colab_adapter.cli project validate --manifest <file> --json` and `python3 -m colab_adapter.cli project plan --manifest <file> --json`. These perform dry-run file diffing and schema checks with zero cloud allocation.
2. **Never allocate compute without explicit human approval:**
   `python3 -m colab_adapter.cli project up --manifest <file> --lease-seconds N --approval-id <id>` provisions paid Colab compute and refuses without a granted, unconsumed approval (exit code 3). Agents may run `project approve-request` (pending only); only the operator runs `project approve <id>` in their own terminal. Never script, pipe or pseudo-terminal the approval phrase. Reusing an exact active session needs no new approval. Any change to a covered workspace file needs a new approval; artifacts, logs, `.colab_experiments` state, `__pycache__`, `*.pyc` and `.git` are not covered (see [docs/SECURITY.md](docs/SECURITY.md)). `--no-reuse` while any managed session is live always fails the global cap (exit 4) and does not consume the approval: stop or reconcile the live session first.
3. **Machine-readable JSON output:**
   Pass `--json` to the `python3 -m colab_adapter.cli project` subcommands (`plan`, `approve-request`, `approve`, `up`, `sync`, `exec`, `status`, `stop`, `reconcile`) for structured JSON output.
4. **Strict isolation from unmanaged sessions:**
   The adapter strictly manages only sessions prefixed with its managed namespace. Never attempt to attach to, execute on, or terminate external user sessions (Q08 fail-closed invariant).
5. **Read-only monitoring:**
   `python3 -m colab_adapter.cli usage` and `python3 -m colab_adapter.cli monitor` query account usage without provisioning resources. The monitor web dashboard is strictly read-only: do not construct or expose mutating endpoints.
6. **Backward compatibility & host safety:**
7. **Lease guard is best effort.** `project up` starts a host-side guard that stops only its own session at the lease deadline and retries until the stop is verified (`colab stop` exit 0 without "not found", `sessions` no longer lists the id, and `usage` active assignments back at or below the baseline recorded before `colab new` when known; otherwise the record says `verification: weak`). It is not a billing cap and not a guarantee (no survival across reboot, `kill -9` or OAuth loss). Operators must still check `project status` (exit 5 = needs attention), `sessions` and `usage` after every test; `project reconcile` is the backstop and covers `NEEDS_RECONCILIATION`, `STOPPING` and stale `PROVISIONING` records for managed ids only. Never Ctrl-C `project up`: an interrupted `colab new` can leave an unnamed `[?]` VM that only `usage` and the Colab UI reveal. `project exec --timeout` (default 600 s) bounds each remote run.
   Ensure `colab_adapter` remains compatible with standard Python 3.8+ on the host. Upstream `google-colab-cli` remains isolated in Python >=3.12. Do not touch host GPU drivers.

### V2 M8 monitor guardrails

- The initial monitor is strictly **READ ONLY**: no HTTP endpoints capable of remote Colab run/new/stop/SSH, arbitrary shell or bypassing approval. Agents must not publish a control interface.
- Account polling may invoke only approved **read-only** CLI usage/session inspection at a bounded interval with backoff; it must not create/retain Colab runtimes or request GPU metrics by SSH.
- Keep account-wide CU balance, rate and assignment counts separate from managed job metrics; neither per-job billing nor lifetime history prior to monitoring installation may be invented.
- Train loss/epoch/step/progress/checkpoint and optional GPU samples require actual instrumented run events. Missing values are unavailable, never guessed.
- Host monitoring on Ubuntu at loopback by default. Remote access from any computer requires explicit operator-approved authenticated TLS/identity proxy or VPN. Never expose tokens or project logs to unauthenticated browsers, raw public ports or client-side storage.
- Read [docs/MONITORING_DASHBOARD_V2.md](docs/MONITORING_DASHBOARD_V2.md) and meet milestone M8 / gate G8 / Q19–Q31. Offline mock development only without separate deployment consent.

### V2 Google Drive & Cross-Device Workspace guardrails (Issue #21)

- **Independent Drive OAuth**: Drive API credentials must remain completely separate from Colab CLI OAuth credentials. Never transmit Drive access tokens or refresh tokens to the remote Colab VM, client browsers, or logs.
- **Protected Server-Side Credential Storage**: All secret token files must be stored server-side with strict `0600` (user read/write only) filesystem permissions.
- **Zero Cloud Allocation on Attach**: Viewing an active or historical run via the Workspace Broker or web dashboard must **never** provoke a new Colab GPU allocation or auto-create a VM.
- **Google Drive Custom Property Limit (124 UTF-8 Bytes)**: Google Drive strictly limits each custom property (`appProperties`) to 124 UTF-8 bytes counting key and value together, with at most 30 properties per file (see [Google Drive API limits of custom file properties](https://developers.google.com/workspace/drive/api/guides/properties#limits_of_custom_file_properties)). `validate_drive_properties` is called before every remote write (resumable session initialization and metadata PATCH). Any property set exceeding limits is rejected with `DriveTransferError` locally *before* transmitting over the network, logging only the property key and byte lengths while redacting property values.
- **Compact Metadata Schema v2**: Remote Drive files carry only compact, fixed-size properties (`schema: "2"`, `sha256: <64-hex>`, `path_sha256: <64-hex>`, `is_latest: "true"|"false"`, `next_sha256: <64-hex>`). Full canonical artifact IDs (`project/run/path@sha256_<64hex>`) live exclusively in local `PublicationManifest` (persisted in `manifest.json` and `manifest.json.bak` with mode `0600`). Local `superseded_by` is the full canonical ID reconstructed from project/run/path + `next_sha256`. Legacy properties (`logical_path`, `revision_id`, `project_id`, `run_id`, `superseded_by`) are read-only for backward compatibility, never written for new revisions, and cleaned up via `None` on remote PATCH.
- **Unicode & Long-Path Safety**: Arbitrary long or Unicode logical paths are bound via collision-safe SHA-256 digests (`path_sha256`), eliminating any risk of Drive property size overflow regardless of path length or multi-byte characters.
- **Replay Safety & Idempotence**: Republishing an existing historical revision is strictly idempotent: historical revisions stay historical (`is_latest="false"`). Replay executes **NO new upload session, NO data transfer, and issues NO Drive PATCH request**. Interrupted publications (`pending` state) are safely promoted to latest on retry. Explicit rollback is NOT implemented (republishing an earlier revision does not revert the remote latest pointer).
- **Remote Coordination & Fail-Closed Integrity**: Remote latest revision coordination demotes prior revisions before promoting a new verified revision. If unversioned recovery encounters an ambiguous remote state (0 or >1 remote revisions marked `is_latest="true"`), it fails closed with `DriveIntegrityError`.
- **Durable Local Retention & Immutable Revision History**: The last valid local checkpoint must **never** be deleted before Google Drive upload and verification succeed. Publishing a new revision of the same checkpoint filename (`model.pt`) must preserve all prior verified revisions under canonical `@sha256_<full_sha256>` identities (`PublicationManifest` v2) and never overwrite historical revision metadata or Drive objects. Local retention pruning must never delete remote Drive objects without explicit human authorization.
- **Bit-Reproducible Snapshots**: Source snapshots must be deterministic (normalized headers) and strictly filter out `.env`, private keys, SSH keys, git internals, and cache directories.
- **Reproducible Real-Browser Security Verification**: Dashboard changes must maintain zero `.innerHTML` / inline handler usage, per-response cryptographic CSP nonces, and pass `COLAB_REQUIRE_BROWSER_TESTS=1 python3 -m unittest tests/test_browser_xss_and_cross_process_qa.py -v` in headless Chromium.
- **Testing Demarcation**: All functionality is verified offline with deterministic mock clients (`OFFLINE_RELEASE_READY`). Real Google Drive and Colab environments have never been exercised in this worktree or automated tests; live OAuth credentials and cloud writes are not authorized without explicit operator consent (`LIVE_VALIDATION_REQUIRES_APPROVAL`).

### V2 storage & monitor invariants (PR #24, offline)

See [docs/ARTIFACT_RECOVERY.md](docs/ARTIFACT_RECOVERY.md), [docs/GOOGLE_DRIVE_ARTIFACTS.md](docs/GOOGLE_DRIVE_ARTIFACTS.md), [docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md) sections 14–15, and [docs/MONITORING_DASHBOARD_V2.md](docs/MONITORING_DASHBOARD_V2.md) section 8.

- **Exact revisions**: a versioned id must carry a full 64-hex sha256. Resolve it only from the manifest or `.revisions/<full sha>/` (legacy `<sha16>/` is read-only). Never fall back to the mutable alias.
- **Manifest is authoritative**: the `checkpoints/<name>` alias is a cache of the head. Reconcile it only under the manifest lock. Replaying a historical revision never moves the head or the alias. If the manifest is lost while revisions exist, fail closed.
- **One lock per manifest**: `<run_dir>/.locks/manifest.lock` is shared by the local and Drive stores (thread RLock plus `fcntl.flock`, 0700 directory and 0600 file, bounded timeout raising `JournalLockError`). Do not add a second lock for the same manifest.
- **Drive pending revisions**: capture `base_sha256` once at intent and keep it in the journal. Promote only on `PROMOTE` or `ALREADY_LATEST`. A `STALE` revision stays pending and retrievable. Legacy records without a base follow the legacy rules and fail closed when ambiguous. `DriveIntegrityError` is terminal in the upload worker.
- **Drive has no compare-and-swap**: the stale and promote guarantee holds only if every writer to the folder goes through this host's locks. Do not describe it as holding for external writers.
- **Validation before side effects**: validate ids, file names, logical paths and artifact types (`storage/validation.py`) before any path, lock, staging directory or remote call. Ids are ASCII. File names are NFC with no separators or control/format characters. Staging names are hashed. Re-validate journal-read ids. Drive read paths validate too.
- **Drive 403 classification**: quota reasons are never retried. Rate reasons get bounded retries. Retry-After is capped at 60 s. Other 403s are auth errors and are never retried.
- **Monitor identity**: runs are keyed by `(project_id, run_id)`. Use `?project=`. A bare run id that matches several projects returns 409 `ambiguous_run`. Skip symlinks and exclude `.locks`. Label account-wide counters as account-wide. `is_runtime_active` is conservative.
- **Offline only**: do not describe mock or offline results as live Colab, Drive or OAuth validation, and never claim `PRODUCTION_READY` or `LIVE_VERIFIED`.

