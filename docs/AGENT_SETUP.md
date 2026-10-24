# Agent-assisted installation and verification

This guide is intentionally generic. It applies to Codex, Claude Code, shell copilots, and other agents with **explicitly authorized** access to a user's Linux/macOS host.

## Operating principle

An agent can do substantial local setup. **The user retains control** over browser OAuth consent and paid/limited accelerator allocation. Local interactive prompts are a convenience guard only; robust autonomous approval must come from an independently controlled executor.

## Phase 1 — survey without cloud allocation

```bash
uname -s
python3 --version
git --version
command -v uv || true
command -v colab || true
python3 -m unittest discover -s tests -v
```

Check any existing Colab credentials **without printing their contents**. Don't reset or overwrite another user's session state. Identify platform/PATH problems before installing.

## Phase 2 — install official upstream

```bash
bash scripts/install.sh
colab version
colab --help
python3 colab_server.py doctor
```

If requested to install directly from the upstream Git repo rather than the published package, use `COLAB_INSTALL_SOURCE=git bash scripts/install.sh`. Never substitute a community fork for Google's source without explaining the difference.

## Phase 3 — human-only OAuth

Ask the human to run `bash scripts/authenticate.sh` in an interactive server session. The human opens Google's authorization URL on their own browser and pastes the code into their server terminal. The agent must never receive the authorization code or store OAuth token files. When the user confirms consent, safe read-only checks are `colab --auth=oauth2 sessions` and `colab --auth=oauth2 usage`.

`colab auth` is for VM-side GCP credentials, not CLI login.

## Phase 4 — local validation (no GPU spend)

```bash
python3 -m unittest discover -s tests -v
bash -n scripts/install.sh scripts/authenticate.sh
python3 colab_server.py plan --gpu T4 --script examples/gpu_probe.py
```

Run this phase even when account login is not yet possible. No cloud allocation is needed.

## Phase 5 — live verification, only if authorized

Show the human the requested accelerator, the exact code to execute, the possibility of compute-unit use, and the teardown plan. Ask for authorization for **this specific test**, not broad consent for future GPU use.

The human runs:

```bash
python3 colab_server.py run --gpu T4 --script examples/gpu_probe.py
```

Do not synthesize the approval phrase or operate a pseudo-terminal to fake a human response. If the GPU is unavailable, report the upstream error; do not silently escalate to an expensive tier.

Afterward, confirm `colab --auth=oauth2 sessions` shows no unexpected assignment. Stop remaining sessions by exact name after notifying the human.

## Phase 6 — integrate with a host scheduler (optional)

Consumers may implement a local/remote compute provider abstraction. Keep this general repository independent of Discord, Notion, GitHub workers, a particular home server, or any single agent vendor.

A safe scheduler should keep pending jobs in a queue until an independent user-approval system authorizes allocation. For each job, record intent, approved GPU, approval identity/reference, execution status, outcome, artifacts, usage observations and cleanup result. Protect the executor from same-account agents; do not rely on environment variables for approval.

## Acceptance checklist

- [ ] README and docs explain ephemeral Colab and subscription limits
- [ ] official CLI installs without sudo
- [ ] user's own OAuth authorization completes without exposing credentials
- [ ] `doctor` and `plan` require no GPU
- [ ] unauthenticated/noninteractive `run` refuses provisioning
- [ ] offline tests and shell syntax checks pass
- [ ] optional live smoke test used **specific approval**
- [ ] any runtime created has been released, or its current state disclosed
- [ ] agent reports exactly which steps were done and which require user action
