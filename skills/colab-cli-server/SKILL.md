---
name: colab-cli-server
description: Install and operate Google's official Colab CLI on Linux/macOS with headless OAuth, safe agent setup and human-approved GPU execution.
---

# Colab CLI Server — agent skill

Companion to Google's [official colab-operator skill](https://github.com/googlecolab/google-colab-cli/tree/main/skills/colab-operator).

## Mandatory rules

- Do not provision paid/limited resources (`colab new`, `colab run`, auto-provisioning `colab ssh`) without specific human approval.
- Do not complete OAuth consent or collect authorization codes, tokens or secret files. Human runs `bash scripts/authenticate.sh` in their own terminal.
- Never spoof the `ALLOCATE T4` confirmation, emulate a TTY, or bypass permission with agent-controlled environment variables.
- No keep-alive or quota-bypass loops. Never promise persistent Colab GPU uptime.
- Validate generated artifacts; upstream `colab exec` exit codes may not reflect every remote Python failure.
- Stop any named session once outputs are retrieved.

## Read-only/local setup

```bash
bash scripts/install.sh
python3 colab_server.py doctor
python3 -m unittest discover -s tests -v
python3 colab_server.py plan --gpu T4 --script examples/gpu_probe.py
```

Install from Google's Git repo only when requested: `COLAB_INSTALL_SOURCE=git bash scripts/install.sh`.

## Human authentication and testing

The user signs in through `bash scripts/authenticate.sh`. Read-only authenticated commands: `colab --auth=oauth2 sessions` and `colab --auth=oauth2 usage`. `colab auth` is *VM-side* Google Cloud auth, not CLI login.

A human can approve `python3 colab_server.py run --gpu T4 --script examples/gpu_probe.py` interactively. Agents must not type that confirmation. For stateful jobs, a protected executor performs `colab new` → `install/upload/exec` → `download` → `stop`.

## Completion

Report installed CLI version, offline checks, auth status, whether any VM was created, session cleanup, and remaining user steps. Read `AGENTS.md` and `docs/AGENT_SETUP.md` for full guidance.
