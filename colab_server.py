#!/usr/bin/env python3
"""Small, dependency-free safety/ergonomics wrapper around Google's official CLI.

The official CLI is the execution backend. This script does not implement
Colab APIs, store tokens or provision sessions on its own.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

GPUS = ("T4", "L4", "G4", "A100", "H100")
AUTH = ("oauth2", "adc")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colab_server.py",
        description="Human-reviewed entrypoint for the official Google Colab CLI.",
    )
    parser.add_argument(
        "--auth", choices=AUTH, default="oauth2",
        help="Explicit upstream CLI auth provider (default: oauth2).",
    )
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("doctor", help="Read-only local tool and environment checks.")
    subs.add_parser("sessions", help="List remote sessions (may require account login).")
    subs.add_parser("usage", help="Read-only compute-unit balance (may require login).")
    for name in ("plan", "run"):
        p = subs.add_parser(name, help="Inspect / interactively approve a one-shot job.")
        p.add_argument("--gpu", choices=GPUS, required=True)
        p.add_argument("--script", required=True, type=Path, help="Existing local Python file.")
        p.add_argument(
            "--timeout", type=float, default=120.0,
            help="Upstream kernel inactivity timeout in seconds; not a cost/time budget.",
        )
    return parser


def cli_command(provider: str, *args: str) -> list[str]:
    return ["colab", f"--auth={provider}", *args]


def doctor() -> int:
    print("Colab CLI Server doctor (read-only; does not allocate a VM)")
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"uv: {shutil.which('uv') or 'not found'}")
    cli = shutil.which("colab")
    print(f"colab: {cli or 'not found'}")
    token_path = Path.home() / ".config" / "colab-cli" / "token.json"
    print(f"OAuth cache present: {'yes (not proof of validity)' if token_path.exists() else 'no'}")
    if not cli:
        print("Install the official CLI with: bash scripts/install.sh", file=sys.stderr)
        return 1
    result = subprocess.run(["colab", "version"], check=False)
    return int(result.returncode)


def validate_job(args: argparse.Namespace) -> Path:
    script = args.script.expanduser().resolve()
    if not script.is_file() or script.suffix != ".py":
        raise ValueError("The --script path must be an existing .py file.")
    if not (0 < args.timeout <= 86400):
        raise ValueError("--timeout must be > 0 and <= 86400 seconds.")
    return script


def render_plan(provider: str, script: Path, gpu: str, timeout: float) -> list[str]:
    cmd = cli_command(
        provider, "run", "--gpu", gpu, "--timeout", str(timeout), str(script)
    )
    print(f"Job: {script}")
    print(f"Accelerator requested: {gpu} (availability NOT guaranteed)")
    print(f"Identity provider: {provider}")
    print("Impact: creates a new remote VM and may consume Colab compute units.")
    print("Script content runs remotely; local data and imports are not synced automatically.")
    print("Behavior: the upstream run command attempts automatic VM teardown.")
    print("Command: " + " ".join(subprocess.list2cmdline([x]) for x in cmd))
    return cmd


def human_confirm(gpu: str) -> bool:
    # Not a security boundary against malicious code sharing this OS identity.
    if not sys.stdin.isatty():
        print("REFUSED: provisioning needs a human at an interactive terminal.", file=sys.stderr)
        return False
    phrase = f"ALLOCATE {gpu}"
    print(f"To authorize this specific runtime, enter exactly: {phrase}")
    try:
        return input("> ").strip() == phrase
    except (EOFError, KeyboardInterrupt):
        return False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return doctor()
    if args.command in ("sessions", "usage"):
        if not shutil.which("colab"):
            print("colab not found; run bash scripts/install.sh", file=sys.stderr)
            return 1
        return subprocess.run(cli_command(args.auth, args.command), check=False).returncode
    try:
        script = validate_job(args)
    except ValueError as e:
        print(f"Invalid job: {e}", file=sys.stderr)
        return 2
    cmd = render_plan(args.auth, script, args.gpu, args.timeout)
    if args.command == "plan":
        print("PLAN ONLY: no allocation was requested.")
        return 0
    if not shutil.which("colab"):
        print("colab not found; run bash scripts/install.sh", file=sys.stderr)
        return 1
    if not human_confirm(args.gpu):
        print("NOT APPROVED: no Colab allocation requested.", file=sys.stderr)
        return 3
    try:
        return subprocess.run(cmd, check=False).returncode
    except KeyboardInterrupt:
        print(
            "Interrupted. Inspect 'colab sessions' for orphaned assignments "
            "and stop any remaining session manually.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    sys.exit(main())
