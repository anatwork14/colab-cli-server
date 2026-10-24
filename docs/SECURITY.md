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
