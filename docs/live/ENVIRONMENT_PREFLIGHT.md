# Environment preflight for live Colab validation

Scope: offline inspection only. Measured at repo HEAD `f4c9ee5` (merge of PR #24, `main`).
Allowed commands used: `colab --help`, `colab <sub> --help`, `colab version`, the offline
capability probe (`version`, `--help`, `ssh --help`), and the offline unit suite.
No `new`, `run`, `exec`, `ssh`, `upload`, `download`, `stop`, `sessions`, `usage` or `auth`
was executed. No network request was made and no credential or token file was read.

## 1. Host facts (measured)

| Item | Measured value | Note |
|---|---|---|
| OS | Ubuntu 20.04.6 LTS, kernel 5.4.0-216 | |
| Default `python3` | 3.8.10 | Adapter target is 3.8+; upstream CLI needs >=3.12 |
| `python3.11` | 3.11.16 under the user's `~/.local/bin` | Symlink targets a `/tmp/...` interpreter. Fragile: disappears if `/tmp` is cleaned |
| `python3.12` | 3.12.14 under `~/.local/bin` (uv-managed) | |
| uv | 0.12.13 | |
| colab CLI | 0.7.5 (`colab version` -> `Version: 0.7.5`) | `~/.local/bin/colab` -> uv tool env `google-colab-cli`, which runs Python 3.12.14 |
| CPUs | 8 | |
| RAM | 62 GiB total; 29 GiB used, 30 GiB buff/cache, 31 GiB available at measurement | |
| `/mnt/ssd` | 1.8T, 42G free, **98% used** | **Risk** for artifact and checkpoint storage. Free space before large downloads |
| Baseline suite | `python3 -m unittest discover -s tests` on Python 3.8.10: **Ran 619 tests, OK (skipped=1)**, 154.8 s | Measured locally at `f4c9ee5` |
| `main` CI | f4c9ee5 CI passed | Reported by the requester, **not verified locally** (no network) |

Offline capability probe (`colab_adapter.capability.probe_cli_capabilities`) returned
version `0.7.5`, 26 top-level commands, and `supports_ssh`, `supports_ssh_proxy_mode`,
`supports_upload`, `supports_download`, `supports_exec`, `supports_sessions`,
`supports_usage` all `True`.

## 2. Compatibility table: adapter `colab` invocations vs CLI 0.7.5 help

Global form used by the adapter: `colab --auth=<oauth2|adc> <subcommand> ...`.
Top-level help: `--auth <oauth2|adc>  Authentication strategy to use: 'oauth2' (public InstalledAppFlow), or 'adc' (Application Default Credentials).`
`colab --auth=oauth2 new --help` exits 0, so placing `--auth` before the subcommand is correct.

Status key: SUPPORTED = flag/command present in help; UNSUPPORTED = absent; UNVERIFIED = help does not state the behaviour the adapter relies on, and no live check was allowed.

| # | Adapter call site | Invocation | Upstream help evidence (quoted) | Status |
|---|---|---|---|---|
| 1 | `colab_adapter/provider.py:499`, `colab_adapter/transports/official_cli.py:73`, `colab_adapter/monitor/collector.py:148`, `colab_server.py:45` | `colab --auth=<mode> ...` | `--auth <oauth2\|adc>` "Authentication strategy to use" | SUPPORTED |
| 2 | `colab_server.py:60`, `colab_adapter/capability.py:119` | `colab version` | "Show the version of the Colab CLI"; prints `Version: 0.7.5` | SUPPORTED |
| 3 | `colab_adapter/capability.py:125-139` | `colab --help` (parses command names) | Commands section lists `new`, `run`, `exec`, `stop`, `sessions`, `usage`, `upload`, `download`, `ls`, `rm`, `install`, `ssh` | SUPPORTED |
| 4 | `colab_adapter/capability.py:143` | `colab ssh --help` contains `--proxy-mode` | "--proxy-mode  Act as an OpenSSH ProxyCommand-compatible WebSocket-stdio bridge" | SUPPORTED |
| 5 | `colab_adapter/capability.py:315-321` | `colab --config <file> ssh --proxy-mode` (no `-s`) | top-level `--config <str>` "Path to session state file"; ssh: "Bare ``colab ssh`` uses your only active session, or auto-creates one (like ``colab new``) if you have none" | SUPPORTED flags. **Paid-allocation behaviour UNVERIFIED**, see M5 |
| 6 | `colab_adapter/provider.py:548`, `colab_server.py:31`, `colab_server.py:104` | `colab sessions` | "List all active sessions" | SUPPORTED. Output format UNVERIFIED (parser `parse_session_lines`, `provider.py:393`), see M7 |
| 7 | `colab_adapter/provider.py:579`, `colab_adapter/monitor/collector.py:148`, `colab_server.py:32` | `colab usage` | "Show account compute-unit usage rate and balance" | SUPPORTED. Output format UNVERIFIED, see M7 |
| 8 | `colab_adapter/provider.py:655`, `provider.py:854`, `colab_adapter/project_sessions.py:293` | `colab new -s <id> --gpu <G>` | `-s, --session <str> Session name`; `--gpu <str> GPU accelerator variant. Supported: T4, L4, G4, H100, A100.` | SUPPORTED. GPU set matches `SUPPORTED_GPUS` (`provider.py:37`) and `GPUS` (`colab_server.py:16`) |
| 9 | `colab_adapter/provider.py:657`, `provider.py:814`, `colab_server.py:75` | `colab run -s <id> --gpu <G> --timeout <T> <script>` | `Usage: colab run [OPTIONS] {script} [script_args]...`; `--timeout <float> Timeout in seconds for code execution [default: 30.0]`; `--keep` "Do not stop the session after the script finishes" (not used) | SUPPORTED. Options precede the script path in all call sites (correct) |
| 10 | `colab_server.py:75` (`plan`/`run` path) | `colab run --gpu <G> --timeout <T> <script>` (no `-s`) | `-s, --session` "Name for the ephemeral session (auto-generated if omitted)" | SUPPORTED |
| 11 | `colab_adapter/provider.py:874` | `colab exec -s <id> --timeout <T> -f <script>` | `-s, --session`; `-f, --file <str> File to execute`; `--timeout <float> ... [default: 30.0]` | SUPPORTED |
| 12 | `colab_adapter/transports/official_cli.py:236` | `colab exec -s <id> -f <tmp.py> [--env K=V ...]` | `--env <str> Set an environment variable in the remote kernel as KEY=VALUE. Repeat for multiple variables.` | SUPPORTED. **No `--timeout` is passed**, so the upstream default 30.0 s applies, see M3 |
| 13 | `colab_adapter/provider.py:863` | `colab install -s <id> -r <requirements>` | `--requirement -r <str> Requirements file` | SUPPORTED |
| 14 | `colab_adapter/provider.py:869`, `official_cli.py:116` | `colab upload -s <id> <local> <remote>` | `Usage: colab upload [OPTIONS] {local_path} {remote_path}` | SUPPORTED (argument order matches) |
| 15 | `colab_adapter/provider.py:897`, `official_cli.py:193` | `colab download -s <id> <remote> <local>` | `Usage: colab download [OPTIONS] {remote_path} {local_path}` | SUPPORTED (argument order matches) |
| 16 | `colab_adapter/transports/official_cli.py:202` | `colab ls -s <id> <path>` | `Usage: colab ls [OPTIONS] [path]`; `path ... [default: content]` | SUPPORTED. Output parsing (skip `[colab]` lines) UNVERIFIED |
| 17 | `colab_adapter/transports/official_cli.py:218` | `colab rm -s <id> <path>` | `Usage: colab rm [OPTIONS] {path}`; "Remove a remote file" | SUPPORTED |
| 18 | `provider.py:838,905,1030,1053,1062,1142`, `colab_adapter/project_sessions.py:417` | `colab stop -s <id>` | "Stop a session"; `-s, --session <str> Session name` | SUPPORTED |
| 19 | `colab_adapter/project_sessions.py:417` | `colab stop -s <id>` (no `--auth`) | same | SUPPORTED, but see M1 (auth flag omitted) |
| 20 | `colab_adapter/project_sessions.py:293` | `colab new -s <id> --gpu <G>` (no `--auth`) | same as row 8 | SUPPORTED, but see M1 |

Error-text and output assumptions (not flags; cannot be checked offline):

| Adapter assumption | Location | Status |
|---|---|---|
| `"not found"` / `"no active sessions found"` in stderr means session missing | `official_cli.py` `_run_cmd` (CalledProcessError branch) | UNVERIFIED |
| `"ResourceExhausted"` / `"unavailable"` in logs means GPU allocation unavailable | `provider.py` one-shot `run` branch (after line 814) | UNVERIFIED |
| `sessions` stdout parsed by `parse_session_lines` | `provider.py:393`, `provider.py:548` | UNVERIFIED |
| `ls` stdout has `[colab]` status lines | `official_cli.py` `list_dir` | UNVERIFIED |
| `--config` file keys are session names (isolated store) | `capability.py` `_session_in_colab_cli_store` | UNVERIFIED |

## 3. Mismatches that could break or misfire a live run

| ID | Severity | Finding | Location | Effect |
|---|---|---|---|---|
| M1 | High | `project_sessions.py` calls `colab new` and `colab stop` **without** `--auth=<mode>`. Other paths pass `--auth=<auth_mode>`. The CLI default is `oauth2`. | `colab_adapter/project_sessions.py:293`, `:417` | With `adc` configured, allocation and stop use `oauth2` while exec, upload and download use `adc`. Identities split; a session may be created under one account and driven by another. Fix: add `f"--auth={...}"` before `new`/`stop`. |
| M2 | High (paid) | `project up` has no approval gate. `ProjectSessionManager.__init__` defaults `require_approval=False`, the CLI builds it with no argument, and `up()` is called without an approval token. | `colab_adapter/project_sessions.py:186`, `colab_adapter/cli.py:213`, `:218` | `colab_adapter project up` runs `colab new` (`project_sessions.py:293`) with no human gate. This conflicts with AGENTS.md rule 2. Not a flag mismatch, but it can allocate paid compute on a live run. |
| M3 | Medium | Transport `exec_code` does not pass `--timeout`. The upstream default is 30.0 s. The adapter's `timeout` argument only bounds the subprocess. | `official_cli.py:236` vs `provider.py:874` (which does pass `--timeout`) | Long scripts run through the transport can be cut at 30 s upstream. Fix: pass `--timeout` as the provider does. |
| M4 | Medium | `colab run` is passed `--timeout` in the kernel sense (help: "Timeout in seconds for code execution"). `colab_server.py` help calls it "Upstream kernel inactivity timeout". Both agree on the flag, but the semantics differ from the `plan` wording. | `colab_server.py:38-41`, `provider.py:814` | Documentation only. No functional break observed. |
| M5 | Medium (paid, guarded) | Bare `colab ssh` **auto-creates** a runtime when no session exists. The guard builds `ssh --proxy-mode` without `-s` and relies on the isolated `--config` store to prevent auto-creation. Upstream behaviour for a missing entry is not verified. | `colab_adapter/capability.py:315-321`; `transports/ssh.py:47-49` | Disabled by default (`transports/ssh.py:33`, `capability.py:173`). If enabled, an auto-allocation is possible. Keep disabled for live runs. |
| M6 | Medium | `exec`, `install`, `upload`, `download`, `ls`, `rm` with `-s <id>` are not documented as non-allocating. Help is silent on whether a missing session is auto-created. | `provider.py:863-897`, `official_cli.py:116-236` | UNVERIFIED. Treat as possible paid path until a supervised check. |
| M7 | Medium | Output formats and error strings for `sessions`, `usage`, `ls` and the "not found"/"ResourceExhausted" checks are assumptions, not verified against 0.7.5. | `provider.py:393`, `provider.py:548`, `official_cli.py` `_run_cmd`/`list_dir` | Wrong parsing can hide managed sessions from `cleanup()` or misclassify a failed allocation. Needs one approved read-only sample (`colab sessions`, `colab usage`) with output redacted. |
| M8 | Low | `colab_server.py` `plan`/`run` build `colab --auth=<p> run ...` correctly. `project_sessions.py` is the only allocation path with no `--auth`. | see M1 | Covered by M1. |
| M9 | Environment | `~/.local/bin/python3.11` is a symlink into `/tmp`. | host | Test or tool runs on 3.11 break if `/tmp` is cleared. Replace with a stable interpreter before depending on it. |
| M10 | Environment | `/mnt/ssd` has 42 GB free (98% used). | host | Risk for `colab_outputs` and checkpoint downloads. `check_disk_space` requires 500 MB only (`provider.py:116`). Free space before a live run. |

No flag in the adapter is UNSUPPORTED by CLI 0.7.5. All mismatches are about auth identity, approval gating, timeout propagation, auto-create semantics and unverified output formats.

## 4. Paid-runtime allocation points (file:line)

| # | Location | Trigger | Gate in place |
|---|---|---|---|
| A | `colab_adapter/provider.py:814` (`colab run -s ... --gpu ...`), inside `submit_job` (`provider.py:767`) | `colab_adapter/cli.py:415-417` (`run` subcommand): `prepare_job` -> `approve_job_interactive` -> `submit_job` | Interactive TTY approval (`provider.py:690`) |
| B | `colab_adapter/provider.py:854` (`colab new -s ... --gpu ...`), stateful branch of `submit_job` | same as A (`cli.py:415-417`, `--stateful`) | same as A; teardown `provider.py:905` (`stop`) in `finally` |
| C | `colab_adapter/project_sessions.py:293` (`colab new -s ... --gpu ...`), `ProjectSessionManager.up` (`project_sessions.py:241`) | `colab_adapter/cli.py:218` (`project up`) | **None by default**: `require_approval=False` (`project_sessions.py:186`); `cli.py:213` passes no approval token. See M2 |
| D | `colab_server.py:125` (`subprocess.run(cmd)` with `colab --auth=<p> run ... <script>`) | `colab_server.py` `run` subcommand, after `render_plan` (`:73`) | `human_confirm` (`:87`): TTY required, exact phrase `ALLOCATE <gpu>`. Non-TTY refused |
| E | `colab_adapter/capability.py:315-321` (`colab --config <f> ssh --proxy-mode`, no `-s`) | `SshSafetyGuard.build_safe_ssh_proxy_invocation` (`capability.py:277`), via `colab_adapter/transports/ssh.py:47-49` | `allow_ssh=False` by default (`transports/ssh.py:33`, `capability.py:173`). Not reached from the CLI. Auto-create possible if enabled (M5) |
| F | Implicit: `-s <id>` calls to `exec`, `install`, `upload`, `download`, `ls`, `rm` (`provider.py:863-897`, `official_cli.py:116-236`) | Only if upstream auto-creates on a missing session | UNVERIFIED (M6). Adapter assumes these fail on a missing session |
| G | Command preview only, no execution: `provider.py:655`, `provider.py:657`; `colab_adapter/cli.py:389` (`plan`) | `plan` paths | Not an allocation. Listed for completeness |

Read-only (no allocation by design): `sessions` (`provider.py:548`, `colab_server.py:104`), `usage` (`provider.py:579`, `colab_server.py:104`, `monitor/collector.py:148`), `version`, `--help`. Cleanup (`stop`) call sites: `provider.py:838, 905, 1030, 1053, 1062, 1142`; `project_sessions.py:417`.

## 5. Not verified in this preflight

- No live `colab` allocation, login, or session listing was run. Every row marked UNVERIFIED needs a supervised, approved probe.
- The CI status of `main` at f4c9ee5 was not checked locally.
- No Colab, Google Drive or OAuth endpoint was contacted.
