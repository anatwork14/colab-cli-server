# Cross-Device Test Plan (read-only monitor)

Status of this plan: **offline-built**. The checker and its tests run against a loopback
monitor with seeded fixtures. No second device, no Colab VM and no Google service was
exercised when this plan was written. A live run is a separate, owner-approved step.

## 1. Scope and what this does NOT prove

- Localhost-only testing does **not** validate remote access. A check that passes on
  `127.0.0.1` on the host proves only the local server. It says nothing about reachability
  from another network, TLS, proxies, firewalls or identity gateways.
- An SSH local-port-forward test from a second device proves that the tunnel and the
  monitor's auth work from that device. It still does not validate any public or
  non-loopback exposure. Public exposure is out of scope and is not described here.
- If no second device is available, the cross-device result is **BLOCKED**, not PASS.

## 2. Architecture for the test

```
second device                         host (Ubuntu, monitor owner)
-------------                         ----------------------------
browser / checker                     colab_adapter.cli monitor
  http://127.0.0.1:8080  --ssh -L-->  127.0.0.1:8080   (loopback only)
```

- The monitor stays bound to `127.0.0.1` on the host. Do not pass `--host 0.0.0.0`,
  `--allow-external` or a public interface for this test.
- The only path from the second device is the SSH tunnel.

## 3. Supplying the owner token safely

The token is never placed on a command line, in shell history, in a chat, or in a file
tracked by Git.

1. On the host, create a 0600 token file outside the repository (example path, generic):

   ```bash
   umask 077
   python3 -c 'import secrets,os; fd=os.open(os.path.expanduser("~/.config/colab-monitor/owner_token"), os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o600); os.write(fd, (secrets.token_urlsafe(32)+"\n").encode()); os.close(fd)'
   chmod 0600 ~/.config/colab-monitor/owner_token
   ```

2. Export it into the environment from the file, without echoing it:

   ```bash
   export COLAB_MONITOR_TOKEN="$(cat ~/.config/colab-monitor/owner_token)"
   ```

3. Start the monitor with the environment variable. `colab_adapter.cli monitor` reads
   `COLAB_MONITOR_TOKEN` when `--auth-token` is absent:

   ```bash
   python3 -m colab_adapter.cli monitor --host 127.0.0.1 --port 8080
   ```

   Do not use `--auth-token` for this test. Arguments are visible to other users through
   `ps` and may be saved in shell history. Note: the monitor's startup line prints the
   first four characters of the token; treat the monitor's stdout as sensitive and keep it
   out of shared logs.

4. On the second device, place the same token in a 0600 file via a channel you already
   trust (for example, copying the file over an SSH session you control). Do not paste it
   into chat or email. Export it the same way as in step 2.

`scripts/live/monitor_check.py` reads only the environment variable (default
`COLAB_MONITOR_TOKEN`). It refuses any `--token`-style flag or any argv element that
contains a token-like value, and it never prints the token. Use `--log-file` only for logs
you are allowed to scan.

## 4. Opening the tunnel from the second device

```bash
ssh -L 8080:127.0.0.1:8080 user@host
```

Replace `user@host` with your own account and host. Keep the SSH session open while testing.
Then, on the second device, the monitor is reached at `http://127.0.0.1:8080`.

## 5. Procedure

Run these from the repository checkout on the second device, with `COLAB_MONITOR_TOKEN`
exported (see section 3). Replace the placeholders with values from your own project.

```bash
# (a) snapshot managed sessions before any second-device activity
python3 scripts/live/monitor_check.py --base-url http://127.0.0.1:8080 \
  --expect-project <PROJECT_ID> --expect-run <RUN_ID> \
  --save-sessions-snapshot ~/monitor-test/sessions_before.json

# (b) full read-only validation from the second device
python3 scripts/live/monitor_check.py --base-url http://127.0.0.1:8080 \
  --expect-project <PROJECT_ID> --expect-run <RUN_ID> \
  --expect-latest-sha256 <FULL_64_HEX_HEAD_SHA256> --expect-revisions <N> \
  --log-file <PATH_TO_MONITOR_LOG> \
  --evidence-file ~/monitor-test/evidence_second_device.json

# (c) snapshot again after the validation
python3 scripts/live/monitor_check.py --base-url http://127.0.0.1:8080 \
  --expect-project <PROJECT_ID> --expect-run <RUN_ID> \
  --save-sessions-snapshot ~/monitor-test/sessions_after.json

# (d) assert the managed session set did not grow (no network access)
python3 scripts/live/monitor_check.py --assert-no-new-sessions \
  --sessions-before ~/monitor-test/sessions_before.json \
  --sessions-after ~/monitor-test/sessions_after.json \
  --evidence-file ~/monitor-test/evidence_no_new_sessions.json
```

Keep the evidence files out of Git. They contain counts and statuses, not tokens, but they
do contain your project and run identifiers.

## 6. Proving a second client does not create a VM

The second device must never cause a new Colab allocation. Evidence for this has two parts:

1. **Managed session set unchanged.** Step (d) above compares the `/api/v1/sessions`
   snapshots taken before and after the second-device session. It fails if any new managed
   session appeared. Optionally, on the host, run `python3 -m colab_adapter.cli sessions`
   before and after as a read-only listing.
2. **No allocation command was issued.** Zero `colab new`, `colab run`, or
   `colab_adapter.cli project up` invocations during the window. The checker cannot prove the
   absence of a command by itself. Record it from the host shell history and your own
   operation log for the test window, and keep that record with the evidence.

Attaching to a run through the monitor or workspace broker is read-only. A result that shows
a new managed session during step (a) to (d) is a FAIL.

## 7. What constitutes PASS

The second-device run is PASS only if **all** of the following hold:

- `scripts/live/monitor_check.py` exits 0 and its evidence file has `"overall": "PASS"`.
  The checks are:
  - a1, a2, a3: unauthenticated GET, `?token=` query auth and wrong bearer are rejected with 401.
  - b: owner GETs of health, summary, runs, sessions, run detail, artifacts and metrics return 200 with the expected shape.
  - c1, c2, c3: run list entries carry `project_id` and `run_id`; the expected `(project, run)` pair exists; only the manifest head is `is_latest: true`; the latest checkpoint state is `known` and matches the expected sha256.
  - d1, d2, d3: POST, PUT, DELETE and PATCH are refused with 405 or 404; no mutating method is advertised; the run list is unchanged by the probes.
  - e1, e2: no token or secret-shaped value appears in responses or in the scanned log.
  - f: CSP header with a per-response nonce; no `unsafe-inline`; every script and style tag carries the nonce; no inline event handlers.
- `--assert-no-new-sessions` exits 0.
- The owner's record shows zero allocation commands during the window (section 6).
- The same results are obtained from the host itself on `127.0.0.1`, so the tunnel is shown
  to add no differences in results.

PASS here means the read-only monitor behaved as specified on that run. It does not approve
remote exposure, and it does not validate Colab or Google behaviour.

## 8. Current known gap

- **d1 currently FAILS against the server as merged**: `PATCH` returns `501 Unsupported
  method` because `colab_adapter/monitor/server.py` has `do_POST`, `do_PUT` and `do_DELETE`
  but no `do_PATCH`. The checker requires 405 or 404, as the monitor's documentation states.
  The fix is a one-line `do_PATCH` that sends 405, after which
  `tests/test_live_monitor_check.py::test_full_pass_once_patch_returns_405` will pass and its
  `expectedFailure` marker must be removed.
- Until that is fixed, a full second-device PASS is not available. Report the run as FAIL on
  d1, not as PASS.

## 9. BLOCKED conditions

Mark the cross-device result **BLOCKED** (not PASS, not FAIL) when any of these hold:

- No second device is available for the test.
- The owner has not approved the tunnel or the test window.
- The token file cannot be created with mode 0600, or the token cannot be supplied through
  the environment.
- Only localhost results exist. Those are recorded as "local-only", which is a separate
  result from cross-device.

## 10. Commands that must not be run for this test

- `colab new`, `colab run`, `colab ssh` with auto-provisioning, and `colab_adapter.cli project up`.
- Any `--host 0.0.0.0`, `--allow-external`, public firewall or router change.
- Any step that prints, logs or commits the token.
