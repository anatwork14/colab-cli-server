#!/usr/bin/env python3
"""CLI utility for researchers and agents to use Colab GPU Adapter and Project Workspaces."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Support running directly or as a package module
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from colab_adapter.approval import GlobalCapExceededError
    from colab_adapter.artifacts import ArtifactManager
    from colab_adapter.errors import ColabProviderError, ColabV2Error
    from colab_adapter.monitor import start_monitor_server
    from colab_adapter.project_manifest import (
        ProjectSpec,
        init_project,
        load_manifest,
        plan_project,
        validate_manifest,
    )
    from colab_adapter.project_sessions import ProjectSessionManager
    from colab_adapter.provider import (
        ApprovalRequiredError,
        ColabGPUProvider,
        JobSpec,
        SUPPORTED_GPUS,
    )
    from colab_adapter.tracker import ExperimentTracker
    from colab_adapter.workspace import WorkspaceSyncEngine
else:
    from .approval import GlobalCapExceededError
    from .artifacts import ArtifactManager
    from .errors import ColabProviderError, ColabV2Error
    from .monitor import start_monitor_server
    from .project_manifest import (
        ProjectSpec,
        init_project,
        load_manifest,
        plan_project,
        validate_manifest,
    )
    from .project_sessions import ProjectSessionManager
    from .provider import (
        ApprovalRequiredError,
        ColabGPUProvider,
        JobSpec,
        SUPPORTED_GPUS,
    )
    from .tracker import ExperimentTracker
    from .workspace import WorkspaceSyncEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colab-adapter",
        description="Google Colab GPU Compute Provider & Project Workspaces CLI (V2)",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    # V1 Commands
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

    # V2 Project Workspace Commands
    proj_parser = subs.add_parser("project", help="Manage Colab GPU Project Workspaces (V2).")
    proj_subs = proj_parser.add_subparsers(dest="project_action", required=True)

    # project init
    init_p = proj_subs.add_parser("init", help="Initialize a project workspace manifest.")
    init_p.add_argument("--name", default="my-project", help="Project name.")
    init_p.add_argument("--root", type=Path, default=Path("."), help="Project root directory.")
    init_p.add_argument("--entrypoint", default="train.py", help="Entrypoint script.")
    init_p.add_argument("--gpu", default="T4", choices=SUPPORTED_GPUS, help="Target accelerator.")
    init_p.add_argument("--format", choices=["json", "yaml"], default="json", help="Manifest format.")

    # project validate
    val_p = proj_subs.add_parser("validate", help="Validate a project workspace manifest.")
    val_p.add_argument("--manifest", type=Path, default=None, help="Path to manifest file.")
    val_p.add_argument("--json", action="store_true", help="Output machine-readable JSON.")

    # project plan
    plan_p = proj_subs.add_parser("plan", help="Dry-run plan for workspace synchronization and resources (Q03).")
    plan_p.add_argument("--manifest", type=Path, default=None, help="Path to manifest file.")
    plan_p.add_argument("--json", action="store_true", help="Output machine-readable JSON.")

    # project approve-request (agent-runnable; writes a pending record, no cloud side effects)
    areq_p = proj_subs.add_parser(
        "approve-request",
        help="Write a pending allocation approval for the manifest's exact plan (no cloud side effects).",
    )
    areq_p.add_argument("--manifest", type=Path, default=None, help="Path to manifest file.")
    areq_p.add_argument("--lease-seconds", type=float, default=3600.0, help="Maximum lease the approval allows.")
    areq_p.add_argument("--json", action="store_true", help="Output machine-readable JSON.")

    # project approve (operator step; reads the phrase from /dev/tty)
    appr_p = proj_subs.add_parser(
        "approve",
        help="Operator step: type the approval phrase at the terminal (/dev/tty) to approve one request.",
    )
    appr_p.add_argument("approval_id", help="Approval id printed by 'project approve-request'.")
    appr_p.add_argument("--json", action="store_true", help="Output machine-readable JSON.")

    # project up
    up_p = proj_subs.add_parser(
        "up",
        help="Provision or attach a managed project session. New allocations need --approval-id.",
    )
    up_p.add_argument("--manifest", type=Path, default=None, help="Path to manifest file.")
    up_p.add_argument("--lease-seconds", type=float, default=3600.0, help="Lease duration in seconds.")
    up_p.add_argument("--approval-id", default=None, help="Operator-approved id from 'project approve'.")
    up_p.add_argument("--no-reuse", action="store_true", help="Force provisioning a new session instead of reusing.")
    up_p.add_argument("--json", action="store_true", help="Output machine-readable JSON.")

    # project sync
    sync_p = proj_subs.add_parser("sync", help="Synchronize local workspace changes to remote session.")
    sync_p.add_argument("--manifest", type=Path, default=None, help="Path to manifest file.")
    sync_p.add_argument("--session-id", default=None, help="Target session ID.")
    sync_p.add_argument("--delete-remote", action="store_true", help="Delete remote files deleted locally.")
    sync_p.add_argument("--json", action="store_true", help="Output machine-readable JSON.")

    # project exec
    exec_p = proj_subs.add_parser("exec", help="Execute command or script in remote project session.")
    exec_p.add_argument("exec_cmd", help="Command or code to execute remotely.")
    exec_p.add_argument("--session-id", required=True, help="Target session ID.")
    exec_p.add_argument("--working-dir", default=None, help="Remote working directory.")
    exec_p.add_argument("--json", action="store_true", help="Output machine-readable JSON.")

    # project status
    status_p = proj_subs.add_parser("status", help="Get session status and remaining lease duration.")
    status_p.add_argument("--session-id", required=True, help="Session ID to inspect.")
    status_p.add_argument("--json", action="store_true", help="Output machine-readable JSON.")

    # project stop
    stop_p = proj_subs.add_parser("stop", help="Safely stop a managed project session (Q08).")
    stop_p.add_argument("--session-id", required=True, help="Session ID to stop.")
    stop_p.add_argument("--json", action="store_true", help="Output machine-readable JSON.")

    # project reconcile
    rec_p = proj_subs.add_parser("reconcile", help="Reconcile and cleanup expired session leases.")
    rec_p.add_argument("--no-auto-stop", action="store_true", help="Only audit without stopping.")
    rec_p.add_argument("--json", action="store_true", help="Output machine-readable JSON.")

    # V2 Monitor Dashboard Command (M8)
    mon_parser = subs.add_parser("monitor", help="Start the cross-device training & compute monitor server.")
    mon_parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1 loopback).")
    mon_parser.add_argument("--port", type=int, default=8080, help="Listen port (default: 8080).")
    mon_parser.add_argument("--auth-token", default=None, help="Authentication bearer token.")
    mon_parser.add_argument("--allow-external", action="store_true", help="Allow binding to non-loopback interface.")
    mon_parser.add_argument("--trusted-proxy", action="store_true", help="Accept identity headers from a verified reverse proxy.")
    mon_parser.add_argument("--trusted-proxy-secret", default=None, help="Shared secret for verifying reverse-proxy headers.")
    mon_parser.add_argument("--require-tls", action="store_true", help="Enforce TLS encryption on the monitor socket.")
    mon_parser.add_argument("--tls-certfile", default=None, help="Path to TLS certificate PEM file.")
    mon_parser.add_argument("--tls-keyfile", default=None, help="Path to TLS private key PEM file.")

    return parser


def _find_default_manifest(root_dir: Optional[Path] = None) -> Path:
    base = Path(root_dir or ".").resolve()
    for name in ("colab-project.json", "colab-project.yaml", "colab-project.yml"):
        cand = base / name
        if cand.is_file():
            return cand
    return base / "colab-project.json"


def handle_project_command(args: argparse.Namespace) -> int:
    """Handle all 'colab-adapter project ...' subcommands."""
    action = args.project_action
    is_json = getattr(args, "json", False)

    try:
        if action == "init":
            target = init_project(
                target_dir=args.root,
                name=args.name,
            )
            print(f"Initialized workspace manifest at {target}")
            return 0

        manifest_path = getattr(args, "manifest", None) or _find_default_manifest()

        if action == "validate":
            spec = load_manifest(manifest_path)
            validate_manifest(spec)
            if is_json:
                print(json.dumps({"valid": True, "project": spec.name, "manifest": str(manifest_path)}, indent=2))
            else:
                print(f"Manifest valid: {spec.name} (root: {spec.root}, entrypoint: {spec.entrypoint})")
            return 0

        elif action == "plan":
            spec = load_manifest(manifest_path)
            plan = plan_project(spec)
            if is_json:
                print(json.dumps(plan, indent=2))
            else:
                print(f"=== Project Workspace Plan (0 Cloud Allocations) ===")
                print(f"Project Name:    {plan['project_name']}")
                print(f"Total Files:     {plan['total_files']}")
                print(f"Total Bytes:     {plan['total_bytes']} bytes")
                print(f"Manifest Digest: {plan['manifest_digest']}")
                print(f"GPU Requested:   {plan['gpu']}")
                print(f"Remote Root:     {plan['remote_root']}")
            return 0

        mgr = ProjectSessionManager()

        if action == "approve-request":
            spec = load_manifest(manifest_path)
            rec = mgr.request_approval(spec, lease_seconds=args.lease_seconds)
            view = mgr.approvals.public_view(rec)
            if is_json:
                print(json.dumps(view, indent=2))
            else:
                print("Approval requested (pending; no Colab resources were allocated).")
                print(f"Approval id:  {view['approval_id']}")
                print(f"Summary:      {view['summary']}")
                print(f"Expires at:   {view['expires_at']}")
                print(f"Operator runs: python3 -m colab_adapter.cli project approve {view['approval_id']}")
                print(f"Operator types: {view['phrase']}")
            return 0

        if action == "approve":
            # With --json only stdin must be a terminal: the phrase still comes from /dev/tty, and
            # stdout carries only the JSON object (prompts and summary go to /dev/tty or stderr).
            rec = mgr.approve(args.approval_id, require_stdout_tty=not is_json)
            view = mgr.approvals.public_view(rec)
            if is_json:
                print(json.dumps(view, indent=2))
            else:
                print(f"Approved {view['approval_id']} (single use; expires at {view['expires_at']}).")
                print(f"Next: python3 -m colab_adapter.cli project up --manifest <manifest> --approval-id {view['approval_id']}")
            return 0

        if action == "up":
            spec = load_manifest(manifest_path)
            reuse = not getattr(args, "no_reuse", False)
            session = mgr.up(
                spec,
                lease_seconds=args.lease_seconds,
                reuse_active=reuse,
                approval_id=getattr(args, "approval_id", None),
            )
            if is_json:
                print(json.dumps(session.to_dict(), indent=2))
            else:
                print(f"Session ready: {session.session_id} (state: {session.state}, lease remaining: {session.time_remaining():.0f}s)")
            return 0

        elif action == "sync":
            spec = load_manifest(manifest_path)
            s_id = args.session_id
            if not s_id:
                active = mgr.get_active_session_for_project(getattr(spec, "project_id", f"proj-{spec.name}"))
                if not active:
                    print(f"Error: No active session found for project '{spec.name}'. Run 'project up' first.", file=sys.stderr)
                    return 1
                s_id = active.session_id

            sync_engine = WorkspaceSyncEngine(transport=mgr.transport)
            res = sync_engine.apply_sync(spec, session_id=s_id)
            if is_json:
                print(json.dumps(res, indent=2))
            else:
                print(f"Sync complete for {s_id}: status={res.get('status', 'OK')}, transferred {res.get('files_transferred', 0)} files ({res.get('bytes_transferred', 0)} bytes).")
            return 0

        elif action == "exec":
            res = mgr.exec(
                session_id=args.session_id,
                command_or_code=args.exec_cmd,
                working_dir=args.working_dir,
            )
            if is_json:
                print(json.dumps(res, indent=2))
            else:
                print(res.get("output", ""))
            return 0 if res.get("success", False) else 1

        elif action == "status":
            stat = mgr.status(args.session_id)
            if is_json:
                print(json.dumps(stat, indent=2))
            else:
                print(f"Session: {stat.get('session_id')}")
                print(f"State:   {stat.get('state')}")
                print(f"Managed: {stat.get('managed')}")
                print(f"GPU:     {stat.get('accelerator')}")
                print(f"Lease:   {stat.get('time_remaining_seconds', 0):.0f}s remaining (active: {stat.get('lease_active')})")
            return 0

        elif action == "stop":
            ok = mgr.stop(args.session_id)
            if is_json:
                print(json.dumps({"session_id": args.session_id, "stopped": ok}, indent=2))
            else:
                print(f"Session '{args.session_id}' stopped successfully.")
            return 0

        elif action == "reconcile":
            auto_stop = not getattr(args, "no_auto_stop", False)
            rep = mgr.reconcile_leases(auto_stop=auto_stop)
            if is_json:
                print(json.dumps(rep, indent=2))
            else:
                print(f"Reconcile complete: stopped {len(rep['stopped'])}, failed {len(rep['failed'])}, active {len(rep['active'])}.")
            return 0

    except ApprovalRequiredError as e:
        print(json.dumps(e.to_dict(), indent=2), file=sys.stderr)
        return 3
    except GlobalCapExceededError as e:
        print(json.dumps({"error": str(e), "reason": e.reason}, indent=2), file=sys.stderr)
        return 4
    except ColabV2Error as e:
        if is_json:
            print(json.dumps(e.to_dict(), indent=2), file=sys.stderr)
        else:
            print(f"Project Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        if is_json:
            print(json.dumps({"error": str(e)}, indent=2), file=sys.stderr)
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "project":
        return handle_project_command(args)

    if args.command == "monitor":
        token = args.auth_token or os.environ.get("COLAB_MONITOR_TOKEN")
        proxy_secret = args.trusted_proxy_secret or os.environ.get("COLAB_TRUSTED_PROXY_SECRET")
        scheme = "https" if (args.require_tls or args.tls_certfile) else "http"
        print(f"Starting Colab Training & Compute Monitor on {scheme}://{args.host}:{args.port}")
        if token:
            print(f"Authentication active. Token: {token[:4]}****")
        elif args.trusted_proxy:
            print("Trusted reverse-proxy authentication active.")
        else:
            print("WARNING: Authentication disabled. Monitor is accessible to local loopback callers only.")
        server = start_monitor_server(
            host=args.host,
            port=args.port,
            auth_token=token,
            allow_external=args.allow_external,
            trusted_proxy=args.trusted_proxy,
            trusted_proxy_secret=proxy_secret,
            require_tls=args.require_tls,
            tls_certfile=args.tls_certfile,
            tls_keyfile=args.tls_keyfile,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down monitor server.")
            server.shutdown()
            server.server_close()
        return 0

    provider = ColabGPUProvider()

    if args.command == "doctor":
        status = provider.check_availability()
        print("=== Colab GPU Provider Doctor ===")
        print(f"Colab CLI installed:       {status['cli_installed']}")
        print(f"uv installed:              {status['uv_installed']}")
        print(f"Python version:            {status['python_version']}")
        print(f"Auth mode:                 {status['auth_mode']}")
        print(f"Credentials present:       {status['credentials_present']}")
        print(f"Authenticated with Colab:  {status['authenticated']}")
        print(f"Local GPU available:       {status['local_gpu_available']}")
        print(f"Active managed sessions:   {len(status['active_sessions'])}")
        print(f"Unmanaged user sessions:   {len(status['unmanaged_sessions'])}")
        if status.get("error"):
            print(f"Status Note:               {status['error']}")
        return 0 if status["cli_installed"] else 1

    elif args.command == "sessions":
        avail = provider.check_availability()
        if not avail["authenticated"]:
            print(f"Error: {avail.get('error') or 'Not authenticated'}", file=sys.stderr)
            return 1
        print("Active managed Colab sessions:")
        for s in avail["active_sessions"]:
            print(f" - {s}")
        if not avail["active_sessions"]:
            print(" (None)")
        if avail["unmanaged_sessions"]:
            print("Unmanaged user sessions (unaffected by adapter):")
            for s in avail["unmanaged_sessions"]:
                print(f" - {s}")
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
