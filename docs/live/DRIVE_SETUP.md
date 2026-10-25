# Google Drive live smoke test: owner setup

Status: **not run**. The harness is verified only against the offline mock
(`tests/drive_api_mock.py`). Nothing in this document authorizes an agent to contact Google.
A live run happens only when you run the commands yourself, in your own terminal, with your
own Google account.

## 0. Important: Colab CLI sign-in does not cover Drive

The Colab CLI OAuth sign-in (`scripts/authenticate.sh`, `colab --auth=oauth2`) is a separate
credential for Colab. **It does not authorize the Google Drive REST client.** The Drive
harness needs its own OAuth client, its own consent, and its own token file. Do not reuse or
copy Colab credentials into the Drive token file.

## 1. Create an OAuth Desktop client in your own Google Cloud project

Use a project you control (a personal project is fine).

1. Open the Google Cloud console and select or create a project.
2. **APIs & Services > Library**: search for "Google Drive API" and click **Enable**.
3. **APIs & Services > OAuth consent screen**:
   - User type: **External**, unless you have a Workspace org and prefer Internal.
   - Publishing status: keep it in **Testing**. Do not publish the app.
   - Add **your own Google account** under **Test users**. Only test users can consent
     while the app is in Testing.
   - Scopes: you may leave the scope list empty. The script requests `drive.file` at runtime.
4. **APIs & Services > Credentials > Create credentials > OAuth client ID**:
   - Application type: **Desktop app**. Do not create a Web client.
   - Note the client ID and client secret. Store them in a password manager, not in the repo.
5. Optional: download the client JSON and save it outside the repository with mode 0600:

   ```bash
   install -m 600 /dev/null ~/.config/colab-drive/client_secret.json
   # then paste the downloaded JSON into that file with your editor
   ```

## 2. Why `drive.file` only

`https://www.googleapis.com/auth/drive.file` lets the app see and modify **only files it
created**. It cannot read the rest of your Drive. The helper refuses any token response that
grants a broader Drive scope and saves nothing in that case. Consequence: the smoke test can
only see the folder it creates in step 5 below, which is the point.

## 3. Where the token file lives

- The token is written only by `scripts/live/drive_authorize.py`, through
  `DriveCredentials.save()`: atomic write, mode **0600**.
- Choose a path **outside this repository**, for example `~/.config/colab-drive/token.json`.
  The helper refuses paths inside the repo and refuses to overwrite an existing file.
- Never commit it, paste it into a chat, or copy it to another machine.

## 4. Authorize (human-run, interactive terminal only)

Run this in your own terminal on the machine that will run the harness. It refuses to run
unless stdin and stdout are both TTYs.

```bash
export COLAB_DRIVE_OAUTH_CLIENT_ID='...'        # or pass --client-secret-file instead
export COLAB_DRIVE_OAUTH_CLIENT_SECRET='...'
python3 scripts/live/drive_authorize.py --token-file ~/.config/colab-drive/token.json
unset COLAB_DRIVE_OAUTH_CLIENT_ID COLAB_DRIVE_OAUTH_CLIENT_SECRET
```

What happens:

1. The script prints one consent URL. It also tries to open a browser.
2. Sign in as your test user and approve the `drive.file` permission.
3. The browser redirects to `http://127.0.0.1:<port>/callback`. The local listener accepts
   one redirect, checks the `state` value, and exchanges the code with PKCE (S256).
4. The only output is a success line with the file mode. Codes and tokens are never printed.

Expected final line: `Drive token file saved (scope: .../drive.file). File mode: 0600.`

### Headless server (no browser on the machine)

The redirect goes to `127.0.0.1` on the machine running the script. Forward that port over
SSH from the machine where your browser is:

```bash
# on your laptop: forward a fixed port to the server
ssh -L 8765:127.0.0.1:8765 <your-server>
# on the server, in that SSH session:
python3 scripts/live/drive_authorize.py --port 8765 --token-file ~/.config/colab-drive/token.json
```

Open the printed URL in your laptop browser. The redirect reaches the server through the
tunnel. No code is pasted by hand. Do not run this through an agent, CI, or a non-interactive
shell.

## 5. Dry run (free, no network)

```bash
python3 scripts/live/drive_smoke.py --revisions 3
```

Default mode is `--dry-run`. It prints the root folder name that would be created, the
checkpoint sizes, the operations, the API cap, and the cleanup plan. It makes **zero** network
requests and exits 0.

## 6. Execute (only after you approve the plan)

The harness needs both `--execute` and the exact approval phrase. Without them it refuses
and sends no request.

```bash
mkdir -m 700 -p ~/.local/state/colab-drive-evidence
python3 scripts/live/drive_smoke.py --execute \
  --approval-phrase "APPROVE DRIVE SMOKE TEST" \
  --revisions 3 \
  --token-file ~/.config/colab-drive/token.json \
  --evidence-dir ~/.local/state/colab-drive-evidence
```

Options: `--revisions 3` (default) or `5`. `--work-dir DIR` sets the local scratch directory
(outside the repo, created 0700; default is a new private temp directory). The token file must
exist with mode 0600.

### What the run does (D1 to D5)

| Test | Action | Pass condition |
|---|---|---|
| D1 | Publish checkpoint A (64 KiB) under `colab-live-smoke/run-001/checkpoints/model.pt` | Remote id, size and md5 match; re-download SHA256 matches; manifest and journal entries exist |
| D2 | Publish B, then C (and D, E for 5 revisions) | Every revision keeps its remote file; ids are distinct; each re-download SHA256 matches; exactly one `is_latest=true`, the last one |
| D3 | Replay A | No upload, no PATCH; Drive file count before and after is equal; A stays historical; latest unchanged |
| D4 | Fresh store: retrieve A/B/C exactly; delete only the test-directory `manifest.json` and `.json.bak`; fresh store recovers latest from Drive | Bytes match each time |
| D5 | Property limits (124 bytes accepted, 125 rejected locally); real PATCH of a probe key on the root folder, then removal; `pageSize=1` listing; a Unicode name and a 180-byte name published; missing-artifact error; secret-leak scan | All checks pass |

The harness stops at the first failed test. Later tests are recorded as `NOT_RUN`.

### API operations you should expect

Categories, as counted by the harness: `folder_create` (the root folder and its project/run/
checkpoints subfolders), `list` (folder and name lookups), `get` (metadata reads),
`upload_init` and `upload_chunk` (resumable uploads), `upload_query` (only if an upload is
interrupted), `download` (hash verification), `patch` (appProperties updates), and
`token_refresh` (if the access token expires). There is no delete.

Measured against the offline mock: 3 revisions = 88 requests, 5 revisions = 114.

### Evidence

The evidence JSON is written to `--evidence-dir` with mode 0600 (name
`drive-smoke-<root folder name>.json`). It contains ids, sizes, hashes, timings, PASS/FAIL per
check, and API counts. It contains no tokens, no client secret, and no account address. The
harness scans everything it prints and logs for the access token, refresh token and client
secret, and fails the run if one appears.

## 7. Cleanup (manual, separate step)

The harness **never deletes anything**. After the run, evidence lists:

- the root folder id (`colab-live-smoke-<UTC timestamp>-<random6>`)
- every revision file id created in the run

To clean up, delete that folder in the Drive web UI (Trash it, then empty Trash), or ask for a
separate, approved deletion step. Local scratch directories can be removed as well.

## 8. What is not covered

- Other Drive accounts, Workspace shared drives, and external writers to the same folder.
  Drive has no compare-and-swap; the stale-revision guarantee holds only for writers that go
  through this host's locks.
- Quota and rate-limit behaviour under real load, and Drive's real retry-after timing.
- Retention, rollback, and deletion of remote objects. Explicit rollback is not implemented.
- Colab VM-side access to Drive. The Colab runtime never receives the Drive token.
- Dashboard, broker and monitor paths that read Drive.

## 9. Known limitations found offline

- **Long names.** A 244-byte logical filename fails offline at the local transfer-lock file
  name (`ENAMETOOLONG`), before any Drive write. The lock name is built from the raw file
  name and should be hashed. The smoke test uses a 180-byte name, which the store accepts.
- **Call cap.** The brief targeted about 60 Drive requests. The D1 to D4 flow alone measures
  about 58 against the mock, so the cap is set to 150. This is a deliberate deviation for the
  owner to confirm.
- **Offline only.** Every test in `tests/test_live_drive_harness.py` runs against the mock or
  fake HTTP, browser and receiver objects. No test contacts Google. A passing offline run is
  not evidence of live Drive behaviour.
