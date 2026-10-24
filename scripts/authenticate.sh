#!/usr/bin/env bash
set -euo pipefail
# This script requires a human to finish Google's browser consent.
umask 077
export PATH="$HOME/.local/bin:$PATH"
if ! command -v colab >/dev/null 2>&1; then
  echo "Missing colab; first run bash scripts/install.sh." >&2
  exit 1
fi
echo "Authenticate with your subscription account. Never share OAuth codes with an AI agent."
colab --auth=oauth2 sessions
token_file="$HOME/.config/colab-cli/token.json"
if [[ -f "$token_file" ]]; then chmod 600 "$token_file"; fi
echo "Read-only session inspection completed."
