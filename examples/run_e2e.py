#!/usr/bin/env python3
"""End-to-End PyTorch Training and Artifact Verification Orchestrator for Colab GPU."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root and colab_adapter are on python path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from colab_adapter.provider import (
    ApprovalRequiredError,
    ColabGPUProvider,
    ColabProviderError,
    JobSpec,
)
from examples.synthetic_dataset import generate_synthetic_dataset
from examples.verify_checkpoint import verify_checkpoint


def run_e2e_workflow(gpu: str = "T4", plan_only: bool = True) -> int:
    base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / "synthetic_data.csv"
    train_script = base_dir / "train_model.py"
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure synthetic dataset exists locally
    if not dataset_path.exists():
        generate_synthetic_dataset(dataset_path)

    provider = ColabGPUProvider()

    # 2. Check provider availability & environment
    status = provider.check_availability()
    print("=== Colab E2E Integration Status ===")
    print(f"Colab CLI installed:       {status['cli_installed']}")
    print(f"Authenticated:             {status['authenticated']}")
    print(f"Active Colab sessions:     {len(status['active_sessions'])}")
    if not status["authenticated"]:
        print(f"ERROR: {status.get('error')}", file=sys.stderr)
        return 1

    # 3. Configure stateful JobSpec
    spec = JobSpec(
        project_name="e2e-pytorch-validation",
        script_path=train_script,
        gpu=gpu,
        input_files={str(dataset_path): "/content/synthetic_data.csv"},
        expected_outputs=["/content/model_checkpoint.pt", "/content/status.json"],
        output_download_dir=artifacts_dir,
        timeout=300.0,
        stateful=True,
    )

    # 4. Prepare Job (Offline - NO VM ALLOCATED)
    prep = provider.prepare_job(spec)
    print("\n=== E2E Test Execution Plan (NO VM ALLOCATED) ===")
    print(f"Job ID:             {prep.job_id}")
    print(f"Requested GPU:      {spec.gpu}")
    print(f"Input Dataset:      {dataset_path} -> /content/synthetic_data.csv")
    print(f"Training Script:    {train_script}")
    print(f"Expected Artifacts: {spec.expected_outputs}")
    print(f"Local Download Dir: {artifacts_dir}")
    print(f"Approval Nonce:     {prep.approval_nonce}")
    print(f"Required Phrase:    ALLOCATE {spec.gpu} {prep.approval_nonce}")

    if plan_only:
        print("\nPLAN ONLY: No Colab runtime was provisioned or compute units consumed.")
        print("To execute this live integration test, run with --run and confirm interactively.")
        return 0

    # 5. Interactive Human Approval Gate
    try:
        approved = provider.approve_job_interactive(prep)
    except ApprovalRequiredError as e:
        print(f"\nAuthorization Refused: {e}", file=sys.stderr)
        return 3

    # 6. Execute Job on Colab Backend
    print("\nStarting Colab remote GPU job execution...")
    try:
        result = provider.submit_job(approved)
    except ColabProviderError as e:
        print(f"\nExecution Failed: {e}", file=sys.stderr)
        return 4

    # 7. Local Artifact Integrity & Weight Verification
    print("\nValidating downloaded artifacts...")
    ok = verify_checkpoint(artifacts_dir)
    if not ok:
        print("ERROR: Artifact verification failed!", file=sys.stderr)
        return 5

    # 8. Post-Execution Session Audit
    post_avail = provider.check_availability()
    active_now = post_avail.get("active_sessions", [])
    print(f"\nPost-Execution Audit: Active Colab sessions = {len(active_now)}")
    if active_now:
        print(f"WARNING: Unexpected active sessions remain: {active_now}", file=sys.stderr)
        return 6

    print("\n=== E2E INTEGRATION TEST SUCCEEDED ===")
    print(f"State:       {result.state}")
    print(f"Artifacts:   {result.output_files}")
    print(f"Cleanup:     {result.cleaned_up}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Colab E2E PyTorch Test Runner")
    parser.add_argument("--gpu", default="T4", help="Requested GPU type (default: T4)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", action="store_true", help="Prepare test plan (offline; no allocation)")
    group.add_argument("--run", action="store_true", help="Execute live test with interactive human approval")

    args = parser.parse_args()
    code = run_e2e_workflow(gpu=args.gpu, plan_only=args.plan)
    sys.exit(code)


if __name__ == "__main__":
    main()
