#!/usr/bin/env python3
"""Small, dependency-free safety/ergonomics wrapper around Google's official CLI.

The official CLI is the execution backend. This script does not implement
Colab APIs, store tokens or provision sessions on its own.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
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


RUN_RECORD_PREFIX = "colab-server-run-"
# Terminal registry states: not in ACTIVE_MANAGED_STATES, so a finished run does not count toward the cap.
STATE_TERMINATED = "TERMINATED"
STATE_FAILED = "FAILED"


def _state_registry():
    """Managed-session registry whose directory also holds the approval records."""
    from colab_adapter.provider import ManagedSessionRegistry

    return ManagedSessionRegistry()


def _state_lock(registry):
    """The allocation lock shared with the V1/V2 adapters, kept in the registry's directory."""
    from colab_adapter.provider import ColabProcessLock

    return ColabProcessLock(Path(registry.registry_path).parent / "colab_gpu.lock")


def _script_sha256(script: Path) -> str:
    import hashlib

    return hashlib.sha256(script.read_bytes()).hexdigest()


def build_run_plan(gpu: str, script: Path, timeout: float, auth: str) -> dict:
    from colab_adapter.approval import KIND_SERVER_RUN, SCHEMA

    return {
        "schema": SCHEMA,
        "kind": KIND_SERVER_RUN,
        "gpu": gpu,
        "auth": auth,
        "script_sha256": _script_sha256(script),
        "timeout": float(timeout),
    }


def run_gated(
    gpu: str,
    script: Path,
    timeout: float,
    auth: str,
    cmd: list,
    *,
    registry,
    lock,
    run=None,
) -> int:
    """Run `colab run` only inside the shared allocation gate.

    Order inside the ColabProcessLock: global cap (before any prompt), operator approval at
    /dev/tty with a per-approval nonce, verify-and-consume, script re-hash, managed registry record
    in PROVISIONING (counts toward the cap), `colab run`, then the record is finalized in `finally`
    (TERMINATED on exit 0, FAILED otherwise; terminal states do not count toward the cap).

    Not a security boundary against code running as this OS user; see colab_adapter/approval.py.
    """
    from colab_adapter.approval import (
        KIND_SERVER_RUN,
        ApprovalStore,
        approvals_dir_for_registry,
        digest_of,
        ensure_global_cap,
    )
    from colab_adapter.provider import ApprovalRequiredError, DuplicateAllocationError

    run = run or subprocess.run  # resolved at call time so tests can replace subprocess.run
    plan = build_run_plan(gpu, script, timeout, auth)
    store = ApprovalStore(approvals_dir_for_registry(registry))
    try:
        with lock:  # DuplicateAllocationError when another allocation holds the lock
            ensure_global_cap(registry)
            approval_id = _request_and_approve(store, gpu, script, plan)
            store.verify_and_consume(approval_id, kind=KIND_SERVER_RUN, plan_digest=digest_of(plan))
            # Re-hash right before the upstream CLI reads the script; the window shrinks but does not close.
            if _script_sha256(script) != plan["script_sha256"]:
                raise ApprovalRequiredError(
                    "The script changed after it was approved; request a new approval.",
                    reason="plan_mismatch",
                )
            record_id = f"{RUN_RECORD_PREFIX}{secrets.token_hex(4)}"
            registry.register(record_id, job_id=approval_id, gpu=gpu, project_name=script.stem)
            outcome = STATE_FAILED
            try:
                res = run(cmd, check=False)
                returncode = int(res.returncode)
                outcome = STATE_TERMINATED if returncode == 0 else STATE_FAILED
                return returncode
            except KeyboardInterrupt:
                print(
                    "Interrupted. Inspect 'colab sessions' for orphaned assignments "
                    "and stop any remaining session manually.",
                    file=sys.stderr,
                )
                return 130
            finally:
                registry.update_state(record_id, outcome)
    except DuplicateAllocationError as exc:  # includes GlobalCapExceededError
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 4
    except ApprovalRequiredError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        print("NOT APPROVED: no Colab allocation requested.", file=sys.stderr)
        return 3
    except (KeyboardInterrupt, EOFError):
        print(
            "REFUSED: interrupted at the approval prompt; no Colab allocation requested.",
            file=sys.stderr,
        )
        return 3


def _request_and_approve(store, gpu: str, script: Path, plan: dict) -> str:
    """Create a pending record with a per-approval nonce, have the operator approve it, return its id."""
    code = secrets.token_hex(4)
    record = store.request(
        kind=plan["kind"],
        plan=plan,
        gpu=gpu,
        summary=f"colab run --gpu {gpu} for {script.name} (auth {plan['auth']})",
        phrase_code=code,
    )
    store.approve(record["approval_id"])
    return record["approval_id"]


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
    registry = _state_registry()
    return run_gated(
        args.gpu, script, args.timeout, args.auth, cmd,
        registry=registry, lock=_state_lock(registry),
    )


if __name__ == "__main__":
    sys.exit(main())
