# Colab GPU Provider Adapter

## Overview

The `colab_adapter` package provides a reusable, modular Python adapter and CLI tool to integrate the official Google Colab CLI as an additional remote GPU compute backend for Linux/macOS servers and AI research agents.

It implements the `ColabGPUProvider` interface with structured experiment logging (`ExperimentTracker`), non-sensitive metadata tracking, and a human approval gate.

---

## Architecture

```text
               Research Agent / Researcher
                           │
       ┌───────────────────┴───────────────────┐
       │   ColabGPUProvider (Adapter Layer)    │
       │   - prepare_job() [No cloud cost]     │
       │   - approve_job_interactive()         │
       │   - submit_job() [Verified execution] │
       │   - reconcile_stale_sessions()        │
       └───────────────────┬───────────────────┘
                           │
                 Subprocess Invocation
                           │
            Official google-colab-cli (v0.7.5+)
                           │
                  Google Cloud Colab VM
               (Tesla T4, L4, A100, H100)
```

---

## Key Features

1. **Strict Separation of Planning and Provisioning:**
   `prepare_job(spec)` validates inputs, scripts, and GPU availability, producing a preview plan and unique approval nonce without allocating any cloud VM or consuming compute units.

2. **Concurrency & Resource Controls:**
   - Single active managed Colab session limit (`_max_active_sessions = 1`).
   - Duplicate submission prevention.
   - Guaranteed session teardown in `finally` blocks.

3. **Output & Artifact Verification:**
   Upstream `colab exec` may exit with status 0 even when remote Python tracebacks occur. `ColabGPUProvider` explicitly checks stdout/stderr for traceback patterns, verifies expected checkpoint files exist and are non-empty, and downloads them safely.

4. **Disk-Space Pre-Check:**
   Verifies that the target filesystem has sufficient free space (minimum 500 MB) before downloading remote checkpoints to avoid host disk exhaustion.

5. **Subprocess Timeout Guard:**
   Wraps all CLI operations in explicit timeouts to prevent local processes from hanging indefinitely on network or remote deadlocks.

6. **Non-Sensitive Structured Logging:**
   `ExperimentTracker` logs job lifecycle records into portable JSONL (`~/.colab_experiments/experiments.jsonl`), automatically stripping and redacting any potential credential keys.

---

## Security Model & Boundary Notes

> [!IMPORTANT]
> **Usability Safeguard vs. Security Boundary:**
> The interactive terminal prompt (`ALLOCATE <GPU> <NONCE>`) ensures human awareness and consent before any billable cloud compute is provisioned.
>
> However, if an autonomous AI agent executes under the **same operating-system account** as the user, it inherently has access to the user's files and shell. Therefore, an in-process confirmation check is a safeguard against unintended invocation—not a cryptographic security boundary against malicious or compromised code running in the same OS identity.
>
> For strict isolation of untrusted agents, run the agent in a restricted environment (container or separate OS user) communicating with a **separately protected executor** that manages Colab credentials and requires external out-of-band authorization.

---

## Python API Usage

```python
from colab_adapter import ColabGPUProvider, JobSpec

provider = ColabGPUProvider()

# 1. Read-only environment inspection
status = provider.check_availability()
print("Authenticated:", status["authenticated"])

# 2. Prepare stateful multi-step training job
spec = JobSpec(
    project_name="image-classification",
    script_path="examples/train_model.py",
    gpu="T4",
    input_files={"examples/synthetic_data.csv": "/content/synthetic_data.csv"},
    expected_outputs=["/content/model_checkpoint.pt", "/content/status.json"],
    output_download_dir="./artifacts",
    timeout=600.0,
    stateful=True,
)
prepared = provider.prepare_job(spec)

# 3. Interactive human authorization
approved = provider.approve_job_interactive(prepared)

# 4. Execute job with guaranteed cleanup
result = provider.submit_job(approved)
if result.success:
    print("Downloaded checkpoints:", result.output_files)
else:
    print("Execution failed:", result.error)
```

---

## CLI Usage

The package includes `colab-adapter` (or `python3 -m colab_adapter.cli`):

```bash
# Diagnostics and status
colab-adapter doctor
colab-adapter usage
colab-adapter sessions

# Plan job without allocating compute
colab-adapter plan --script examples/train_model.py --gpu T4

# Execute with terminal approval
colab-adapter run --script examples/train_model.py --gpu T4

# View experiment history
colab-adapter history --limit 10

# Terminate session
colab-adapter stop <session_id>
```
