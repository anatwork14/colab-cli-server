# Architecture and lifecycle

```mermaid
flowchart TB
  U[Developer or researcher] --> H[Linux/macOS host]
  A[AI coding agent] --> H
  H --> P[Plan / doctor: local only]
  H --> G{Human approved allocation?}
  G -->|No| X[No GPU allocation]
  G -->|Yes| C[Official Google colab CLI]
  C --> O[Google user OAuth / ADC]
  C --> R[Ephemeral Colab runtime]
  R --> J[Execute local Python remotely]
  J --> F[Checkpoint / logs / output]
  F --> H
  R --> Z[Release VM]
```

This project adds host-side guidance, an interactive one-shot approval gate, agent policy and offline tests; **Google's official CLI** owns provisioning, authentication, code execution and file transfer.

## Runtime lifecycle

**One-shot:** `colab run --gpu T4 script.py` allocates a fresh runtime, executes, and normally tears down. `colab_server.py run` asks the human to approve and delegates to that upstream command.

**Stateful:** `colab new -s trainer --gpu T4` → `install` / `upload` / `exec` → `download` → `stop`. Capture artifacts before stop. Treat remote loss and disconnects as expected failure modes. Never assume an `exec` process exit code alone proves success.

A Colab GPU is an independent remote accelerator, **not** a CUDA device on the host and **not** pooled VRAM with local GPUs. Colab subscription tiers, availability, quotas and resource lifetime govern what can be allocated.

## Optional external orchestrator

A consuming system may expose a structured GPU provider with states:

```text
DRAFT -> VALIDATED -> WAITING_APPROVAL -> APPROVED -> ALLOCATING
      -> RUNNING -> COLLECTING -> CLEANUP -> SUCCEEDED | FAILED | CANCELLED
```

Track job ID, code revision, requested GPU, artifacts, session ID, runtime and result. Never store tokens or access URLs in public logs. Human approval should live in a separately protected control plane; a prompt in the same OS account is not a security boundary against a malicious agent.

## Explicit design exclusions

No always-on GPU daemon, no keep-alive bypass, no subscription sharing, no vendor-specific server configuration, no live Colab calls in CI and no private infrastructure assumptions.
