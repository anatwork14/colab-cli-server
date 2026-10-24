# Copy-paste instruction for your coding agent

Paste the following instruction into your coding agent **after** it has access to a suitable Linux/macOS host. The repository is public; the machine's account and credentials are private.

---

You are a senior Linux/Python automation engineer. Set up and verify `https://github.com/anatwork14/colab-cli-server` on this machine, using Google's official `https://github.com/googlecolab/google-colab-cli` as the only Colab backend.

**Scope and constraints**

- Work on the machine you're already authorized to access. Do not assume sudo, Docker privileges, an NVIDIA GPU, or a browser is installed.
- Follow the repository's `AGENTS.md`, `docs/INSTALLATION.md`, `docs/AUTHENTICATION.md`, `docs/SECURITY.md` and `docs/AGENT_SETUP.md`.
- Do not create cloud VMs or consume compute units as part of unattended installation/testing.
- Do not complete browser OAuth on my behalf, request codes in chat, copy token JSON into your context, or commit secrets.
- Do not forge the interactive GPU approval phrase, use pseudo-terminals for approval, or bypass the guard via an environment variable. Runtime approval must be per job, from me.
- Do not bypass Colab quotas, session expiry or policy. No always-on keepalive loops.
- Keep the result generic, portable, documented and reproducible. Do not hard-code this host's name, username, GPU, account or private paths into the public project.

**Execution plan**

1. Inspect OS, PATH, Git, Python, uv, existing CLI and the current working tree. Report any pre-existing state. No cloud calls yet.
2. Clone the repository if absent. Review the docs and official CLI README. Use `bash scripts/install.sh` to install the published upstream CLI, or the documented official-Git source option when requested.
3. Verify local installation with `colab version`, `colab --help` and `python3 colab_server.py doctor`. Avoid redoing successful installations.
4. Run `python3 -m unittest discover -s tests -v`, shell syntax checks, and `python3 colab_server.py plan --gpu T4 --script examples/gpu_probe.py`. No Colab VM should be allocated.
5. Tell me how to run `bash scripts/authenticate.sh` in my own terminal. I will complete Google login. After confirmation, use safe read-only commands to check sessions and compute-unit usage, without disclosing secrets.
6. Prepare an exact live GPU smoke-test plan and teardown plan, but **do not run it without my explicit per-test approval**. I will type the consent phrase interactively.
7. If I later authorize integration into an agent scheduler, design a separate protected approval/executor boundary, per-job audit records, bounded retries, artifact retrieval and cleanup. Do not implement a bypass.
8. Provide a final report: files changed, exact commands verified, offline tests, authentication status, any GPU sessions created/stopped, remaining issues, and the next human-only step.

**Success means:** official Colab CLI installed, documented headless authorization, green offline tests, no personal secrets in code, no unauthorized compute-unit usage, and a clear path to user-approved research jobs.

---
