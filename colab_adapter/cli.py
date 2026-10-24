#!/usr/bin/env python3
"""CLI utility for researchers and agents to use Colab GPU Adapter."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support running directly or as a package module
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from colab_adapter.provider import (
        ApprovalRequiredError,
        ColabGPUProvider,
        ColabProviderError,
        JobSpec,
        SUPPORTED_GPUS,
    )
    from colab_adapter.tracker import ExperimentTracker
else:
    from .provider import (
        ApprovalRequiredError,
        ColabGPUProvider,
        ColabProviderError,
        JobSpec,
        SUPPORTED_GPUS,
    )
    from .tracker import ExperimentTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colab-adapter",
        description="Google Colab GPU Compute Provider Adapter CLI",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    subs.add_parser("doctor", help="Check local environment, CLI, local GPU and OAuth status.")
    subs.add_parser("sessions", help="List active remote Colab sessions.")
    subs.add_parser("usage", help="Show Colab compute-unit balance.")

    plan_parser = subs.add_parser("plan", help="Prepare an experiment plan without provisioning.")
    plan_parser.add_argument("--script", required=True, type=Path, help="Python training/probe script.")
    plan_parser.add_argument("--gpu", default="T4", choices=SUPPORTED_GPUS, help="Requested GPU type.")
    plan_parser.add_argument("--project", default="colab-research", help="Project name.")
    plan_parser.add_argument("--timeout", type=float, default=300.0, help="Execution timeout in seconds.")
    plan_parser.add_argument("--reqs", type=Path, default=None, help="Requirements file.")
    plan_parser.add_argument("--stateful", action="store_true", help="Stateful multi-step session.")

    run_parser = subs.add_parser("run", help="Run a GPU experiment (requires human terminal approval).")
    run_parser.add_argument("--script", required=True, type=Path, help="Python training/probe script.")
    run_parser.add_argument("--gpu", default="T4", choices=SUPPORTED_GPUS, help="Requested GPU type.")
    run_parser.add_argument("--project", default="colab-research", help="Project name.")
    run_parser.add_argument("--timeout", type=float, default=300.0, help="Execution timeout in seconds.")
    run_parser.add_argument("--reqs", type=Path, default=None, help="Requirements file.")
    run_parser.add_argument("--output-dir", type=Path, default=None, help="Local artifact download directory.")
    run_parser.add_argument("--stateful", action="store_true", help="Stateful multi-step session.")

    stop_parser = subs.add_parser("stop", help="Stop an active Colab session.")
    stop_parser.add_argument("session_id", help="Session ID to stop.")

    history_parser = subs.add_parser("history", help="Show recent experiment history.")
    history_parser.add_argument("--limit", type=int, default=10, help="Number of records to show.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    provider = ColabGPUProvider()

    if args.command == "doctor":
        status = provider.check_availability()
        print("=== Colab GPU Provider Doctor ===")
        print(f"Colab CLI installed:       {status['cli_installed']}")
        print(f"uv installed:              {status['uv_installed']}")
        print(f"Python version:            {status['python_version']}")
        print(f"Local GPU available:       {status['local_gpu_available']}")
        print(f"OAuth token present:       {status['oauth_token_present']}")
        print(f"Authenticated with Colab:  {status['authenticated']}")
        print(f"Active Colab sessions:     {len(status['active_sessions'])}")
        if status.get("error"):
            print(f"Status Note:               {status['error']}")
        return 0 if status["cli_installed"] else 1

    elif args.command == "sessions":
        avail = provider.check_availability()
        if not avail["authenticated"]:
            print(f"Error: {avail.get('error') or 'Not authenticated'}", file=sys.stderr)
            return 1
        print("Active Colab sessions:")
        for s in avail["active_sessions"]:
            print(f" - {s}")
        if not avail["active_sessions"]:
            print(" (None)")
        return 0

    elif args.command == "usage":
        u = provider.get_usage()
        if not u["success"]:
            print(f"Error checking usage: {u['error']}", file=sys.stderr)
            return 1
        print(u["raw_output"])
        return 0

    elif args.command == "plan":
        try:
            spec = JobSpec(
                project_name=args.project,
                script_path=args.script,
                gpu=args.gpu,
                requirements_path=args.reqs,
                timeout=args.timeout,
                stateful=args.stateful,
            )
            prep = provider.prepare_job(spec)
            print("=== Job Preparation Plan (NO VM ALLOCATED) ===")
            print(f"Job ID:          {prep.job_id}")
            print(f"Project:         {prep.spec.project_name}")
            print(f"Script:          {prep.spec.script_path}")
            print(f"GPU Requested:   {prep.spec.gpu}")
            print(f"Stateful:        {prep.spec.stateful}")
            print(f"Approval Nonce:  {prep.approval_nonce}")
            print(f"Required Phrase: ALLOCATE {prep.spec.gpu} {prep.approval_nonce}")
            print(f"Command Preview: {' '.join(prep.command_preview)}")
            return 0
        except ColabProviderError as e:
            print(f"Plan error: {e}", file=sys.stderr)
            return 2

    elif args.command == "run":
        try:
            spec = JobSpec(
                project_name=args.project,
                script_path=args.script,
                gpu=args.gpu,
                requirements_path=args.reqs,
                output_download_dir=args.output_dir,
                timeout=args.timeout,
                stateful=args.stateful,
            )
            prep = provider.prepare_job(spec)
            approved = provider.approve_job_interactive(prep)
            res = provider.submit_job(approved)
            print(f"Job finished: {res.state}")
            return 0 if res.success else 1
        except ApprovalRequiredError as e:
            print(f"Authorization Refused: {e}", file=sys.stderr)
            return 3
        except ColabProviderError as e:
            print(f"Execution Error: {e}", file=sys.stderr)
            return 4

    elif args.command == "stop":
        ok = provider.cancel_job(args.session_id)
        if ok:
            print(f"Session {args.session_id} stopped.")
            return 0
        else:
            print(f"Failed to stop session {args.session_id}.", file=sys.stderr)
            return 1

    elif args.command == "history":
        tracker = ExperimentTracker()
        records = tracker.list_jobs(limit=args.limit)
        print(f"=== Experiment History (last {len(records)}) ===")
        for r in records:
            status_symbol = "✓" if r.get("success") else "✗"
            print(
                f"[{status_symbol}] {r.get('job_id')} | {r.get('requested_accelerator')} | "
                f"{r.get('job_state')} | Start: {r.get('start_time')} | Project: {r.get('project_name')}"
            )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
