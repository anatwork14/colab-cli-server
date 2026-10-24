# Agent instructions — Colab CLI Server

These instructions apply to autonomous and assisted coding agents working **in this repository** or installing it on an end-user's Linux/macOS host.

## Mission

Make the official [googlecolab/google-colab-cli](https://github.com/googlecolab/google-colab-cli) usable from a terminal or headless server with clear setup, testability, responsible runtime consumption and vendor-neutral agent documentation. Do not replace or fork Google's API implementation when an official CLI command already exists.

## Non-negotiable operating rules

1. **Inspect before changing.** Check OS, available commands, `python3 --version`, `uv --version`, `colab --help`, and existing Colab sessions where authentication is already configured. Never print credentials.
2. **No cloud spending without permission.** `colab new`, `colab run`, `colab ssh` when it auto-provisions, and retaining live runtimes require explicit per-job user authorization. **Do not call these commands** during discovery, installation, CI, or tests.
3. **No bypassing approval.** Never script answers to the interactive prompt, forge approval through stdin/PTY, set a self-approved environment variable, or impersonate a user. If you are an agent in the same host account, the interactive prompt is not a strong policy boundary: request external approval from the human and use a separately protected executor for autonomy.
4. **Human-only sign-in.** Run the supported headless OAuth flow in a terminal when asked, then let the user complete the Google consent. Never solicit or store authorization codes, OAuth tokens or credentials in prompts, logs, Git, CI, tool outputs or agent memory.
5. **User account ownership.** Never use service-account impersonation to claim access to someone's paid Colab entitlement; authenticate as the actual authorized subscriber.
6. **Default cleanup.** Prefer `colab run` for a self-contained job; it normally tears down automatically. For `colab new` sessions, stop the specific session in `finally` on success, error and cancellation where possible, then verify active assignments. Never create keepalive/ping loops.
7. **Keep public code generic.** Never commit hostnames, SSH details, GPU inventory, private paths, subscription balance, raw experiment data, proprietary assets or private endpoints. Place examples in generic paths.
8. **Offline tests only by default.** Mock all Colab interactions. Live CPU/GPU probes need separate approval, and accelerator probes can consume credits.
9. **Do not claim a job succeeded solely because `colab exec` exited 0.** Verify expected output files/checkpoints and inspect kernel output; upstream `exec` may not propagate all remote exceptions as process failure.
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

## V2 project workspace development (planned, not authorized deployment)

Before beginning V2, read [progress.md](progress.md), [milestones.md](milestones.md), [docs/FEATURES_V2.md](docs/FEATURES_V2.md), [docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md), [docs/EXPECTED_RESULTS_V2.md](docs/EXPECTED_RESULTS_V2.md) and [docs/AGENT_IMPLEMENTATION_V2.md](docs/AGENT_IMPLEMENTATION_V2.md). The documents define milestone gates; they do not authorize GPU allocation.

Mandatory V2 constraints:
- Treat official Colab SSH-over-WebSocket as **optional and potentially auto-provisioning**, not a guaranteed no-allocation connection. An already-owned session preflight alone is insufficient protection against races; until a hard authorization/no-create boundary is established, fail closed or use official CLI file operations instead.
- Planning, validation, local source hashing and dry-run diffs must be offline/no-allocation by construction. Runtime creation, retention or tests against live Colab always require explicit human approval. Never spoof a prompt or pass credentials through agents.
- Sync only declared project files in a confined remote root, reject secrets and path escapes, and make remote deletion opt-in. Never copy all of the user's home directory or rewrite GPU drivers.
- Keep the standard-library wrapper compatible with Python >=3.8 and Google's CLI separately installed with Python >=3.12. Do not modify unrelated local GPU workloads.
- Preserve V1 API, registry/session ownership, locking, tracking and tests. Work in small feature branches/PRs, update progress.md and milestones.md with evidence, do not auto-merge/deploy.
- Do not report feature completion from mocks as proof of live remote performance. Distinguish planned, offline proven and owner-approved live proven results.
