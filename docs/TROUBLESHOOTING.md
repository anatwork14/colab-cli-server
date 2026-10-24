# Troubleshooting

| Symptom | Cause to check | Action |
| --- | --- | --- |
| `uv: command not found` | uv not installed / PATH | Follow official uv install guide; reopen shell |
| `colab: command not found` | Tool binary not in PATH | `export PATH="$HOME/.local/bin:$PATH"`; `uv tool list` |
| Python version error | Upstream requires Python >=3.12 | `uv python install 3.12`, reinstall CLI with `--python 3.12` |
| HTTP 401/403 | Wrong Google identity or missing OAuth scopes | Use documented `--auth=oauth2`; inspect [auth guide](AUTHENTICATION.md) |
| Browser doesn't open on server | Headless host | Copy the printed Google authorization URL to your local browser |
| `colab auth` didn't fix CLI login | Wrong auth layer | CLI OAuth is separate from VM-side GCP auth |
| GPU allocation refused | Tier entitlement / capacity / quota / too many sessions | Inspect `colab usage` / `colab sessions`, try another approved GPU or later |
| Job times out | Kernel inactivity timeout, network failure, or reclaimed session | Increase `--timeout` reasonably, design checkpointing, inspect logs |
| `colab exec` returned 0 but traceback printed | Kernel error status isn't always reflected in process exit code | Validate output checkpoint, status marker and logs |
| Script imports fail | Other local source files weren't staged | Install dependencies, upload files, use self-contained Python |
| Checkpoint not found | Remote path differs from script output | `colab ls -s NAME /content/output`; verify script saved there |
| Unexpected active session | Interrupt/failed cleanup | `colab sessions`, then `colab stop -s NAME` |
| Agent cannot run `colab_server.py run` | Noninteractive refusal is expected | Have human authorize via terminal or independently protected executor |

## Diagnosis commands

```bash
colab version
python3 colab_server.py doctor
python3 colab_server.py plan --gpu T4 --script examples/gpu_probe.py
colab --auth=oauth2 sessions
colab --auth=oauth2 usage
```

The first three do not allocate cloud runtimes; the last two contact Colab but do not provision one. Do **not** paste raw OAuth credentials, server session tokens or proxy URLs into issue reports.

If a GPU call fails, do not loop retries automatically: capacity and policy limits may apply.
