# Installation: Linux, macOS, headless servers

This community wrapper relies on the official [googlecolab/google-colab-cli](https://github.com/googlecolab/google-colab-cli). It does not copy the upstream API implementation. No local CUDA GPU or sudo permissions are required.

## Prerequisites

A shell, Git, Internet connectivity and `uv` from the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/). Upstream requires **Python >=3.12**; `uv` installs a compatible interpreter per user. Official CLI supports Linux/macOS; native Windows is not supported.

```bash
git clone https://github.com/anatwork14/colab-cli-server.git
cd colab-cli-server
bash scripts/install.sh
python3 colab_server.py doctor
```

This installs Google's published Python package without creating a VM. To explicitly install Google's Git main instead of a release:

```bash
COLAB_INSTALL_SOURCE=git bash scripts/install.sh
```

Git main may contain unreleased changes. For reproducible deployments, pin an upstream version and record the version used. If uv reports an existing installation, inspect `uv tool list` and intentionally upgrade or reinstall; do not silently replace existing tools.

If `colab` is not on PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
colab version
colab --help
```

## Authentication and verification

```bash
bash scripts/authenticate.sh
python3 -m unittest discover -s tests -v
python3 colab_server.py plan --gpu T4 --script examples/gpu_probe.py
```

The unit tests, `doctor`, and `plan` are offline or local-only; they allocate no Colab resources. `authenticate.sh` contacts Google's backend and initiates an interactive browser flow if needed, but does not allocate a VM.

For Docker, persist the user's HOME/`~/.config/colab-cli` securely outside source control. Never expose the account token directory to untrusted users or unrelated containers.

To request an actual GPU, read [Jobs](JOBS.md) and obtain explicit user consent first.
