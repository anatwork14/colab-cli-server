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

The commands above are a **manual workflow**, not an exception-safe wrapper. If any command fails, inspect `colab sessions` and explicitly stop the session. Prefer `colab run` for standalone scripts because it automatically attempts cleanup. **Important:** in the current upstream implementation, `colab exec` can print a Python traceback without necessarily returning a nonzero shell exit status. Verify remote completion and artifact correctness; do not infer success solely from `exec`'s exit code.

## Agent-assisted setup

An agent may install tools, inspect the host, write job files, run **offline tests**, and present a plan. An agent **must not** authenticate on behalf of a user, allocate/retain GPUs, or consume compute units without explicit authorization for the specific operation.

Start here:
- [Agent policy](AGENTS.md)
- [Agent installation workflow](docs/AGENT_SETUP.md)
- [Copy-paste agent setup prompt](docs/AGENT_SETUP_PROMPT.md)
- [Agent skill](skills/colab-cli-server/SKILL.md)

**The interactive confirmation is a usability safeguard, not a strong security boundary against an agent that controls the same OS account.** If you provide autonomous shell access, enforce approval with an independently protected control plane or separate privileged executor.

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
