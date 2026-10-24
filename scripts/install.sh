#!/usr/bin/env bash
set -euo pipefail
# This script installs only local tooling. It never provisions cloud resources.
if ! command -v uv >/dev/null 2>&1; then
  echo "Missing uv. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
case "$(uname -s)" in
  Linux|Darwin) ;;
  *) echo "Official Colab CLI supports Linux/macOS only." >&2; exit 1 ;;
esac
uv python install 3.12
case "${COLAB_INSTALL_SOURCE:-pypi}" in
  pypi)
    uv tool install --python 3.12 google-colab-cli ;;
  git)
    uv tool install --python 3.12 "git+https://github.com/googlecolab/google-colab-cli.git" ;;
  *) echo "COLAB_INSTALL_SOURCE must be pypi or git." >&2; exit 2 ;;
esac
export PATH="$HOME/.local/bin:$PATH"
colab version
echo "Installed official Colab CLI. No VM allocated. Next: bash scripts/authenticate.sh"
