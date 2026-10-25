# Security and responsible operation

## Public repository rules

This is a **public** community project. Keep it free of personal OAuth JSON, refresh/access tokens, Colab session proxy URLs, account subscription details, SSH keys, private hostnames, private datasets, experimental research secrets and identifiable server inventories.

The CLI uses `~/.config/colab-cli/` outside this source tree. Keep it owned by the authorized UNIX user, with private file permissions, and out of CI artifacts. Do not paste authentication output into agent logs.

## Threat model: shell agents

A coding agent with shell permissions on the **same OS user** that owns the Google token can potentially read or misuse that token and invoke upstream `colab` directly. **The wrapper's interactive confirmation is not an actual security sandbox.** `isatty` is not proof of a human; an agent can fake a terminal. Treat the prompt as an intentional-use safeguard only.

For autonomous integration:
- Store authorization in an **independent approval service/executor** that the agent cannot write or impersonate.
- Use OS permission separation where possible: agent can submit job requests but cannot read account credentials or run a GPU provisioner.
- Restrict GPU types, rate, allowed workloads, and concurrency in that separate executor.
- Ensure every action is logged without exposing sensitive credential material.
- Allow a human to revoke future authorization and stop all existing assignments.
- Never equate `COLAB_APPROVED=1` or another agent-controlled variable with permission.

## Resource use and service restrictions

Paid Google Colab is a constrained interactive/cloud environment, not a guaranteed GPU infrastructure product. Availability, prices, limits, expiration and terms are controlled by Google. No scheduler should defeat idle pruning, simulate user activity, evade quotas, or auto-retry endlessly.

Use `colab usage` to inspect credit balance/rates (not a price quotation). Get specific authorization before every new GPU allocation, select a suitable accelerator, and call `colab stop` when no longer needed. An interrupted terminal command can leave an orphaned session; reconcile via `colab sessions`.

## Dependency/source hygiene

Only install the real `google-colab-cli` from the official package or Google GitHub repository. Review any change to a transitive package, verify Git source if using an unreleased branch, and do not run untrusted training scripts with access to the authenticated host.

## Responsible disclosure

If you discover a vulnerability involving Google's CLI or Colab backend, follow Google's official vulnerability reporting process. For this wrapper, open a private maintainer contact channel first rather than publishing exploit details or tokens in a public issue.

## Allocation approval gate (Issue #25)

Every new managed Colab allocation through `colab_adapter` needs an operator-granted, single-use approval record bound to one exact plan. The approval phrase is read from `/dev/tty` only. A rejected or missing approval makes zero `colab new` calls (exit code 3), and a shared cap allows one active managed session at a time (exit code 4).

**Approval gate limits (plain language).** The gate stops accidental or agent-initiated Colab allocation through this adapter. Each approval is bound to one exact plan (project, GPU, manifest and workspace contents), a maximum lease, a 15-minute expiry and single use, and it leaves an audit record. It does **not** stop a process that runs as the same OS user: such a process can read the record key, forge or edit records, drive a pseudo-terminal to answer the prompt, run `colab new` directly, or change the code. Direct use of `colab new`, `colab run` and `colab ssh` cannot be gated by this repository. The gate is a safety interlock, not a security boundary. The real boundary is the approver and Colab OAuth in a separate OS account or a protected executor. Until that exists, the operator should run `project up` personally for paid runtimes.

**Fingerprint coverage.** The workspace digest in a plan fingerprint covers every regular file under the project root, except: `artifacts_local_dir` (default `./artifacts`, resolved against the project root when inside it), the `.colab_experiments` approvals and registry state directory (when inside the root), `*.log` files, `__pycache__`, `*.pyc` and `.git`. Any change to a covered file, including a new source file, invalidates the plan and needs a new approval. Writing or pulling artifacts, logs and bytecode does not invalidate it. Reuse of a live exact session still works after artifacts are pulled. The fingerprint covers source and manifest content only: a process that changes an excluded file cannot alter a run through the fingerprint, so do not put executable or configuration content in excluded paths.

**Cap and `--no-reuse`.** A `--no-reuse` request while any managed session is active (V1 or V2, or a `colab_server.py run` record) always fails the global cap with exit 4, and the approval is not consumed. Stop or reconcile the live session first.
**Lease guard limits.** The per-session lease guard (`project lease-guard`, started by `project up`) is host-side, best-effort deadline enforcement. It acts only on its own managed session id, verifies removal with `colab sessions`, and writes a sanitized log (no tokens) to `<registry_dir>/lease_guards/<session_id>.log` (directory 0700, file 0600). It is **not** a billing cap and **not** a guarantee: it does not survive a host reboot, a `kill -9` of the guard, or loss of OAuth, and the lease clock includes provisioning time. The operator must still verify `sessions` and `usage` after every test. `project reconcile` is the backstop for a dead guard.

**Key handling.** The per-directory HMAC key must be a regular, non-symlink file owned by the current user. Group and other permission bits are removed. A key shorter than 32 bytes (for example an empty file left by a crash) is replaced atomically, since no record can have been signed with it. A symlink or a foreign-owned key is refused with reason `corrupt`.

**Terminal input.** The phrase is read from `/dev/tty` only, and stdin must be a terminal. With `project approve --json` stdout does not need to be a terminal, so the JSON can be piped. Ctrl-C or EOF at the prompt refuses the request (exit 3) without a traceback.

## Storage hardening controls (PR #24, offline)

These controls are implemented in `colab_adapter/storage/` and are verified only by offline tests. They do not replace the approval gates above.

- **Input validation before side effects.** Project, run, file-name, logical-path and artifact-type inputs are checked against the rules in `colab_adapter/storage/validation.py` before any path, lock, staging directory, journal record or remote request is derived from them. Ids are ASCII. File names are NFC Unicode with no separators, control or format characters. The upload worker re-validates ids read from the transfer journal, which is not trusted as input.
- **Lock files.** The per-manifest lock is `<run_dir>/.locks/manifest.lock`. The directory is mode 0700 and the file is mode 0600. Symlinked lock paths are refused.
- **Staging.** Staging directories use hashed names (`stg-<40 hex>`), so raw ids never form a path. Symlinked staging roots and directories are refused, and the resolved path must stay inside the staging root.
- **Errors.** Untrusted values in validation errors are escaped with `ascii()` and truncated. Drive property error messages show key names and sizes, never values.
- **Drive quota versus auth.** Drive 403 responses are classified by reason. Quota and auth errors are never retried. Rate-limit errors are retried with a 60-second cap on each wait.
- **Monitor.** The monitor remains read-only. Its selectors are validated and its listings skip symlinks and internal directories. See `docs/MONITORING_DASHBOARD_V2.md` section 8.

Limit: the Drive store has no compare-and-swap. Its guarantees hold only when every writer to the folder uses this host's locks (see `docs/GOOGLE_DRIVE_ARTIFACTS.md`, "Drive Store Hardening", section 5).


## OAuth token file writes (PR #24 follow-up, offline-verified)

- `DriveCredentials.save()` and the workspace broker's token file save never write secret bytes into an existing file. They create a fresh temp file in the same directory (`O_EXCL|O_NOFOLLOW`, mode `0600`, `fchmod(0600)` before the first byte), fsync it, and `os.replace` it over the target. A pre-existing `0644`/`0666`/`0777` token file therefore ends up `0600`, and a failed write leaves the previous file intact with no temp file left behind.
- A symlink at the Drive credentials path is *replaced* by a regular `0600` file (never written through). If you deliberately symlinked that file elsewhere, the link is replaced after the next token refresh. The broker resolves its operator-configured `tokens_file` once at construction, so a symlinked path means "use that target" for both load and save, and the target ends up `0600`.
- Parent directories are created `0700` only when the call creates them; existing directories are never re-chmodded.
- Limitation: a process killed (`SIGKILL`) between the temp write and the rename can leave a `0600` `.<name>.tmp.<pid>.<hex>` file holding tokens in the token directory. Nothing sweeps it; delete such files manually. A failed save after a successful token refresh fails closed (the refresh raises `DriveAuthenticationError`) rather than continuing with an unpersisted token.
- POSIX only (`fchmod`, `O_NOFOLLOW`). Parent-chain symlinks are still resolved; keep the token directory private to the service user.
