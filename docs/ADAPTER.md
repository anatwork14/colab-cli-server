# Colab GPU Provider Adapter

## Overview

The `colab_adapter` package provides a reusable, production-hardened Python adapter and CLI tool to integrate Google's official Colab CLI (`google-colab-cli` v0.7.5+) as an additional remote GPU compute backend for developer workstations, servers, and AI research agents.

It implements `ColabGPUProvider`, `ManagedSessionRegistry`, `ColabProcessLock`, and structured experiment logging (`ExperimentTracker`).

---

## Architecture & Control Flow

```text
               Research Agent / Researcher
                           │
       ┌───────────────────┴───────────────────┐
       │   ColabGPUProvider (Adapter Layer)    │
       │   - Cross-process file locking        │
       │   - Session ownership registry        │
       │   - Input tampering verification      │
       │   - Subprocess timeout guards         │
       │   - Error preservation & logging      │
       └───────────────────┬───────────────────┘
                           │
                 Subprocess Invocation
                           │
            Official google-colab-cli (OAuth2 / ADC)
                           │
                  Google Cloud Colab VM
               (Tesla T4, L4, A100, H100)
```

---

## Core Principles & Reliability Features

1. **Session Ownership & Protection of Unrelated Work:**
   The adapter maintains a local registry (`~/.colab_experiments/managed_sessions.json`) tracking only sessions provisioned by itself.
   - `cancel_job(session_id)` immediately refuses requests for sessions not registered as managed by this adapter, returning `False` without invoking remote termination.
   - One-shot jobs (`stateful=False`) are pre-registered with deterministic IDs passed explicitly via `colab run -s <session_id>`. If the local subprocess times out or is interrupted, the adapter explicitly attempts cleanup of the registered session.
   - `cleanup()` and `reconcile_stale_sessions()` strictly terminate only registered managed sessions.
   - Unmanaged user sessions on the account are detected, reported, and left completely untouched.

2. **Approval Architecture & Tamper Resistance:**
   - At planning time (`prepare_job`), the adapter computes a cryptographic fingerprint binding all immutable job assets and mutable execution parameters: project name, script content SHA256, requirements SHA256, input files (local path, remote transfer destination, content SHA256), expected outputs, output download directory, dataset identifier, requested GPU, timeout, and stateful flag.
   - At submission time (`submit_job`), all assets on disk and remote transfer destinations are re-verified. Any modification between approval and execution triggers `JobTamperedError`.
   - **Security Boundary:** The interactive terminal prompt (`ALLOCATE <GPU> <NONCE>`) is a human safeguard. In environments with autonomous agents sharing the same OS account, in-process nonces do not form a cryptographic sandbox. A **separately protected approval executor** running under an independent OS user or container with an out-of-band authorization channel must be used for untrusted autonomy.

3. **Cross-Process and Cross-Thread Concurrency Protection:**
   `ColabProcessLock` enforces synchronization using thread locks combined with non-blocking `fcntl.flock` on `~/.colab_experiments/colab_gpu.lock`.
   Concurrent submissions across processes or threads fail immediately with `DuplicateAllocationError`, preventing accidental duplicate billing.

4. **Deep Credential Sanitization:**
   `ExperimentTracker` recursively applies regular expression scrubbers across all logged fields, paths, and tracebacks, redacting:
   - Google OAuth authorization codes (`4/0A...`)
   - Google OAuth Bearer / access tokens (`ya29....`)
   - GitHub and generic API tokens (`ghp_...`, `ey...`)
   - PEM private keys
   - URL query secrets (`?code=...`, `&token=...`)
   All log files (`experiments.jsonl`, `managed_sessions.json`, `colab_gpu.lock`) are enforced with private `0600` permissions.

5. **Error Preservation & Independent Failure Tracking:**
   Primary execution errors (e.g. remote tracebacks, CUDA errors, missing files) are never masked by subsequent cleanup or usage-query failures. Cleanup and usage errors are captured as distinct fields in experiment metadata, and the primary exception is preserved.
   If execution succeeds but remote session teardown fails, the experiment is marked `CLEANUP_FAILED` with `success=False`, and a `CleanupError` is raised.

6. **Subprocess Timeout Guards:**
   Subprocess calls have explicit wall-clock timeouts (`spec.timeout + 180s`) to ensure local processes never hang indefinitely if the remote kernel or network deadlocks.

7. **Dual Authentication Support:**
   Supports both `auth_mode="oauth2"` (validating `~/.config/colab-cli/token.json`) and `auth_mode="adc"` (Application Default Credentials via `GOOGLE_APPLICATION_CREDENTIALS` or `gcloud`).

8. **Multi-Tier Checkpoint Verification with Fail-Closed Security:**
   - **Level 1 (Integrity):** SHA256 checksum and JSON manifest validation.
   - **Level 2 (Structure):** PyTorch ZIP format and file header inspection (`data.pkl`, format version).
   - **Level 3 (Inference):** Full weight loading via `torch.load(weights_only=True)` and forward-pass inference when PyTorch is installed locally.
   - **Fail-Closed Policy:** Automatic fallback to unrestricted `torch.load(..., weights_only=False)` is disabled to protect against arbitrary code execution via malicious pickle payloads.

9. **Interruption & Lifecycle State Machine:**
   Interrupting execution (e.g. `KeyboardInterrupt` / SIGINT) is explicitly caught and transitions the managed session and experiment record to `INTERRUPTED` with `success=False`, rather than silently succeeding.

---

## Python API Usage

```python
from colab_adapter import ColabGPUProvider, JobSpec

# Initialize provider (supports oauth2 or adc)
provider = ColabGPUProvider(auth_mode="oauth2")

# 1. Read-only environment inspection
status = provider.check_availability()
print(f"Authenticated: {status['authenticated']}")
print(f"Active managed sessions: {status['active_sessions']}")
print(f"Unmanaged user sessions: {status['unmanaged_sessions']}")

# 2. Prepare stateful multi-step training job (NO cloud compute allocated)
spec = JobSpec(
    project_name="arc-training",
    script_path="examples/train_model.py",
    gpu="T4",
    input_files={"examples/synthetic_data.csv": "/content/synthetic_data.csv"},
    expected_outputs=["/content/model_checkpoint.pt", "/content/status.json"],
    output_download_dir="./artifacts",
    timeout=300.0,
    stateful=True,
)
prepared = provider.prepare_job(spec)

# 3. Interactive human authorization (in terminal)
approved = provider.approve_job_interactive(prepared)

# 4. Execute job with cross-process locking, artifact verification, and guaranteed cleanup
result = provider.submit_job(approved)
if result.success:
    print(f"Downloaded checkpoints: {result.output_files}")
else:
    print(f"Execution failed: {result.error}")

# 5. Stale session reconciliation (stops only stale managed sessions)
reconciled = provider.reconcile_stale_sessions(auto_stop=True)
```

---

## CLI Usage

The package includes `colab-adapter` (or `python3 -m colab_adapter.cli`):

```bash
# Diagnostics and status (shows managed vs unmanaged sessions)
colab-adapter doctor
colab-adapter usage
colab-adapter sessions

# Plan job without allocating compute
colab-adapter plan --script examples/train_model.py --gpu T4

# Execute with terminal approval
colab-adapter run --script examples/train_model.py --gpu T4

# View experiment history (secrets redacted)
colab-adapter history --limit 10

# Safely terminate a managed session
colab-adapter stop <session_id>
```
