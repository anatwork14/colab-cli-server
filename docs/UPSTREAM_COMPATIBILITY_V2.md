# Upstream Compatibility Audit & Security Contracts (V2)

**Document Status:** FROZEN (Milestone M1 / Release Gate G1)  
**Date:** 2026-10-24  
**Target Upstream CLI:** `google-colab-cli` version 0.7.5  
**Audience:** Maintainers, security reviewers, autonomous agents

---

## 1. Upstream Command Behavior Matrix

Every upstream command has been audited against both **official source code** (installed via uv tool at `/mnt/ssd/genos2/.local/share/uv/tools/google-colab-cli/lib/python3.12/site-packages/colab_cli`) and **live host execution** on the target Ubuntu system.

| Command | Subprocess Invocation | Source Verified | Host Verified | Allocates Compute? | Auto-Creates Session If Missing? | Notes / Exit Semantics |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `colab version` | `colab version` | YES | YES (`0.7.5`) | **NO** | NO | Returns `Version: 0.7.5` on stdout, exit code 0. |
| `colab --help` | `colab --help` | YES | YES | **NO** | NO | Lists all commands and global options, exit code 0. |
| `colab ssh --help` | `colab ssh --help` | YES | YES | **NO** | NO | Documents flags including `--proxy-mode`, `-s`, `--gpu`, `--tpu`, `--rm`. |
| `colab sessions` | `colab sessions` | YES | YES | **NO** | NO | Lists active sessions; queries Colab API. |
| `colab usage` | `colab usage` | YES | YES | **NO** | NO | Queries account compute-unit balance and consumption rate. |
| `colab upload` | `colab upload -s S LOCAL REMOTE` | YES | YES | **NO** | **NO** | Fails closed (`Exit(1)`) if session `S` is missing. Uses Jupyter Contents API. |
| `colab download` | `colab download -s S REMOTE LOCAL` | YES | YES | **NO** | **NO** | Fails closed (`Exit(1)`) if session `S` is missing. Uses Jupyter Contents API. |
| `colab ls` | `colab ls -s S REMOTE_PATH` | YES | YES | **NO** | **NO** | Fails closed (`Exit(1)`) if session `S` is missing. Uses Jupyter Contents API. |
| `colab rm` | `colab rm -s S REMOTE_PATH` | YES | YES | **NO** | **NO** | Fails closed (`Exit(1)`) if session `S` is missing. Uses Jupyter Contents API. |
| `colab exec` | `colab exec -s S -f SCRIPT` | YES | YES | **NO** | **NO** | Fails closed (`Exit(1)`) if session `S` is missing. Executes code via Jupyter kernel client. **Warning:** Returns exit code 0 even if kernel raises a Python exception (does not propagate traceback as CLI exit failure). |
| `colab new` | `colab new -s S [--gpu G]` | YES | YES | **YES** | **YES** | Explicitly allocates a new remote VM. May consume compute units. |
| `colab run` | `colab run [--gpu G] SCRIPT` | YES | YES | **YES** | **YES** | Allocates a fresh one-shot VM, executes script, and automatically stops VM upon exit. |
| `colab stop` | `colab stop -s S` | YES | YES | **NO** | NO | Terminates the specified session. |
| `colab ssh` (bare) | `colab ssh` | YES | YES | **YES (if no active session)** | **YES** | If local session store is empty, calls `_auto_create_session()`. |
| `colab ssh -s NAME` | `colab ssh -s NAME` | YES | YES | **YES (if NAME not in store)** | **YES** | If `NAME` is not in local store, calls `_auto_create_session(name=NAME)`. |
| `colab ssh --proxy-mode -s NAME` | `colab ssh --proxy-mode -s NAME` | YES | YES | **YES (if NAME not in store)** | **YES** | If `NAME` is not in local store, calls `_auto_create_session(name=NAME)`. |

---

## 2. The SSH Auto-Allocation Hazard & Safety Proof

### 2.1 The Vulnerability
In `colab_cli/commands/ssh.py`:
```python
def _select_proxy_session(
    session: Optional[str],
    gpu: Optional[str],
    tpu: Optional[str],
    *,
    high_mem: bool = False,
) -> tuple[SessionState, bool]:
    if session and not _session_exists(session):
        with contextlib.redirect_stdout(sys.stderr):
            return _auto_create_session(
                gpu, tpu, name=session, high_mem=high_mem
            ), True
    return _resolve_session(session), False
```
Where `_session_exists(name)` simply checks:
```python
def _session_exists(name: str) -> bool:
    from colab_cli.common import state
    return state.store.get(name) is not None
```

### 2.2 Consequence
If an autonomous agent or script attempts to use `colab ssh --proxy-mode -s <session>` for remote execution or file sync, and that session has expired, was pruned, or was not previously created, **the upstream CLI automatically provisions a brand new Colab VM without prompting the user**. This can incur unauthorized compute-unit charges.

Furthermore, a simple read preflight (e.g. `colab sessions`) followed by `colab ssh` has a Time-of-Check to Time-of-Use (TOCTOU) race condition: if the session terminates between the check and the SSH call, `colab ssh` will immediately allocate a replacement VM!

### 2.3 Binding Invariant for V2
To guarantee zero unexpected cloud allocations:
1. **Official CLI fallback is the primary transport:** `colab upload`, `colab exec`, and `colab download` NEVER auto-allocate. They fail closed if a session is missing.
2. **Guarded SSH Transport:** SSH transport is strictly disabled by default. If enabled by explicit configuration:
   - The session MUST be verified in `ManagedSessionRegistry` (local ownership check).
   - An active lease check must be performed.
   - An independent `SshSafetyGuard` checks `colab-cli` local state file directly to ensure the session exists in `~/.config/colab-cli/sessions.json` BEFORE dispatching any `colab ssh` process.
   - If the session is missing, `AutoAllocationBlockedError` is raised immediately. No `colab ssh` command is spawned.

---

## 3. Error Taxonomy

The V2 architecture establishes a typed, deterministic error taxonomy in `colab_adapter.errors`:

```
ColabV2Error (Base)
├── UpstreamCliError (Base for upstream CLI interaction errors)
│   ├── UpstreamCliNotFoundError
│   └── UpstreamCliExecutionError
├── AutoAllocationBlockedError (Raised when an operation would risk implicit VM allocation)
├── ManifestValidationError (Invalid manifest, schema violation, dangerous paths)
├── WorkspaceSyncError (File indexing, staging, transfer or remote manifest commit failure)
├── ArtifactValidationError (Missing output, truncated download, checksum mismatch, unsafe pickle)
└── TransportError (Base for transport abstraction)
    ├── TransportUnavailableError (SSH unsupported, proxy failure, missing binaries)
    ├── SessionNotFoundError (Requested session does not exist)
    ├── UnmanagedSessionError (Session exists but is not owned by this adapter instance)
    ├── SessionAllocationRequiredError (Operation requires an allocated runtime, but none exists)
    ├── SshHandshakeError (HTTP 400, 401, 403, 404, 429, 502 WebSocket upgrade failures)
    └── RemoteExecutionError (Non-zero kernel failure, timeout, unhandled traceback)
```

### HTTP & WebSocket Error Status Mapping for SSH
| Status Code | Meaning | Handling in V2 Transport |
| :---: | :--- | :--- |
| **400** | Bad request (missing pubkey header or unsupported RSA key) | Actionable configuration error; abort without retry. |
| **401** | Unauthorized (expired or invalid runtime-proxy token) | Mark session as expired/lost; fail closed. |
| **403** | Forbidden (permission denied or account restriction) | Actionable error; abort without retry. |
| **404** | Not Found (endpoint does not exist or SSH daemon not started on Colab VM) | Gracefully fall back to official CLI upload/exec transport. |
| **429** | Rate limited (too many concurrent connections) | Actionable error with backoff guidance; do not spawn new VM. |
| **502** | Bad gateway / upstream unavailable | Retry once with backoff or fall back to official CLI transport. |

---

## 4. Threat Model & Trust Boundaries for Autonomous Agents

1. **Same-UID Nonce Limitation:** In-process nonces or user confirmation prompts executed in the same operating-system account as an autonomous coding agent cannot serve as a cryptographic security boundary. A local agent can mock stdin or monkeypatch python functions.
2. **Defensive Invariant:** Unattended agent workflows (planning, dry-run diffs, manifest validation, capability probes, status queries) MUST be 100% offline and execute ZERO allocating commands.
3. **Privileged Actions:**
   - Any billable action (`colab new`, `colab run`, session provisioning) requires out-of-band operator authorization with an explicit `ApprovedJob` or external executor.
   - The adapter will never silently allocate or retain sessions.
4. **Data Privacy & Exclusions:**
   - Git metadata (`.git/**`), virtual environments (`.venv/**`), Python bytecode (`**/__pycache__/**`), environment secrets (`.env*`, `**/*token*`, `*.pem`, `id_*`) are strictly excluded by default and cannot be overridden by broad wildcards.
   - Symlinks resolving outside the project root are rejected (preventing arbitrary host file exfiltration).
