"""Shared fake `colab` executable for offline tests (no real Colab is ever contacted).

The fake keeps its server-side state in a directory: one empty file per live assignment under
`<FAKE_COLAB_DIR>/alive/<session id>`. Its output mirrors the upstream google-colab-cli messages
verified in its source (commands/session.py, commands/usage.py, consumption.py):

- `sessions`: "[colab] No active sessions found on server." or one "[name] endpoint | ..." line each.
- `usage`: "Current balance: ...", "Usage rate: ...", "Active assignments: N".
- `stop -s <id>`: "[colab] Session '<id>' not found." with exit 0 when the id is unknown,
  otherwise "[colab] Session terminated." with exit 0.
- `exec -s <id> -f <file>`: runs the file with python3 locally (the upstream kernel stand-in); kernel
  errors go to stderr and the exit code stays 0, as upstream does.

- `new -s <id> --gpu <g>`: mode from env FAKE_COLAB_NEW_MODE (default ok): `ok` assigns <id> and exits 0;
  `fail_before` exits 1 with no assignment; `fail_after_orphan` assigns an UNNAMED orphan (upstream assigns the VM
  before it records the local name; `sessions` shows it as "[?] endpoint") and then exits 1;
  `fail_after_named` assigns <id> and exits 1; `ok_output_but_fail` assigns an orphan, prints "Session READY."
  and exits 1.
- `usage` switches: FAKE_COLAB_USAGE_FAIL (always exits 1), FAKE_COLAB_USAGE_FAIL_AFTER_NEW (exits 1 once `new`
  has been called), and the file `<FAKE_COLAB_DIR>/usage_lag_remaining` holding N, which hides unnamed orphans
  from the next N `usage` readings (a lagging server-side count).

`sessions` switch: FAKE_COLAB_SESSIONS_FAIL (exits 1). `new` switch: FAKE_COLAB_REMOVE_ON_NEW=<id> removes that
alive runtime when `new` is called.

Env switches: FAKE_COLAB_STOP_FAIL (every stop exits 1), FAKE_COLAB_STOP_LIE (stop exits 0 but keeps
the id alive), and the file `<FAKE_COLAB_DIR>/stop_fail_remaining` holding N, which makes the next N
stops exit 1 and then decrements.
"""
import os
import sys
from pathlib import Path

FAKE_COLAB_SOURCE = '''#!{python}
import json
import os
import subprocess
import sys

state_dir = os.environ["FAKE_COLAB_DIR"]
calls_file = os.environ["FAKE_COLAB_CALLS"]
args = sys.argv[1:]
with open(calls_file, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(args) + "\\n")
alive_dir = os.path.join(state_dir, "alive")
os.makedirs(alive_dir, exist_ok=True)
rest = [a for a in args if not a.startswith("--auth=")]
command = rest[0] if rest else ""


def alive_ids():
    return sorted(os.listdir(alive_dir))


if command == "sessions":
    if os.environ.get("FAKE_COLAB_SESSIONS_FAIL"):
        print("sessions unavailable", file=sys.stderr)
        sys.exit(1)
    ids = alive_ids()
    if not ids:
        print("[colab] No active sessions found on server.")
    for sid in ids:
        name = "?" if sid.startswith("orphan-") else sid
        print("[" + name + "] ep-" + sid + " | Hardware: T4 | Shape: Standard | Variant: GPU")
    sys.exit(0)
if command == "usage":
    if os.environ.get("FAKE_COLAB_USAGE_FAIL"):
        print("usage unavailable", file=sys.stderr)
        sys.exit(1)
    if os.environ.get("FAKE_COLAB_USAGE_FAIL_AFTER_NEW") and os.path.exists(os.path.join(state_dir, "new_called")):
        print("usage unavailable", file=sys.stderr)
        sys.exit(1)
    ids = alive_ids()
    lag = os.path.join(state_dir, "usage_lag_remaining")
    if os.path.exists(lag):
        remaining = int(open(lag, encoding="utf-8").read().strip() or "0")
        if remaining > 0:
            with open(lag, "w", encoding="utf-8") as fh:
                fh.write(str(remaining - 1))
            ids = [i for i in ids if not i.startswith("orphan-")]
    print("Current balance: 10.00 compute units")
    print("Usage rate: 1.00/hr")
    print("Active assignments: %d" % len(ids))
    sys.exit(0)
if command == "new":
    sid = rest[rest.index("-s") + 1]
    mode = os.environ.get("FAKE_COLAB_NEW_MODE", "ok")
    open(os.path.join(state_dir, "new_called"), "w", encoding="utf-8").close()
    gone = os.environ.get("FAKE_COLAB_REMOVE_ON_NEW")
    if gone and os.path.exists(os.path.join(alive_dir, gone)):
        os.remove(os.path.join(alive_dir, gone))  # another runtime ends while `new` runs (concurrent change)
    if mode == "fail_before":
        print("[colab] Error: no GPU quota available.", file=sys.stderr)
        sys.exit(1)
    if mode in ("fail_after_orphan", "ok_output_but_fail"):
        open(os.path.join(alive_dir, "orphan-" + sid), "w", encoding="utf-8").close()
        if mode == "ok_output_but_fail":
            print("[colab] Session READY.")
        print("[colab] Error: connection reset while recording the session.", file=sys.stderr)
        sys.exit(1)
    open(os.path.join(alive_dir, sid), "w", encoding="utf-8").close()
    if mode == "fail_after_named":
        print("[colab] Error: could not finish setup.", file=sys.stderr)
        sys.exit(1)
    print("[colab] Session READY.")
    sys.exit(0)
if command == "stop":
    sid = rest[rest.index("-s") + 1]
    counter = os.path.join(state_dir, "stop_fail_remaining")
    if os.path.exists(counter):
        remaining = int(open(counter, encoding="utf-8").read().strip() or "0")
        if remaining > 0:
            with open(counter, "w", encoding="utf-8") as fh:
                fh.write(str(remaining - 1))
            print("stop failed for " + sid + " token ya29.FAKETOKEN0123456789abcdefXYZ", file=sys.stderr)
            sys.exit(1)
    if os.environ.get("FAKE_COLAB_STOP_FAIL"):
        print("stop failed for " + sid + " token ya29.FAKETOKEN0123456789abcdefXYZ", file=sys.stderr)
        sys.exit(1)
    path = os.path.join(alive_dir, sid)
    if not os.path.exists(path):
        print("[colab] Session '" + sid + "' not found.")
        sys.exit(0)
    if not os.environ.get("FAKE_COLAB_STOP_LIE"):
        os.remove(path)
    print("[colab] Session terminated.")
    sys.exit(0)
if command == "exec":
    script = rest[rest.index("-f") + 1]
    res = subprocess.run([sys.executable, script], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr)
    sys.exit(0)
sys.exit(0)
'''


def write_fake_colab(path: Path) -> Path:
    """Write the fake executable to `path` (mode 0700) and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FAKE_COLAB_SOURCE.format(python=sys.executable), encoding="utf-8")
    path.chmod(0o700)
    return path


def set_alive(state_dir: Path, session_id: str) -> None:
    alive = Path(state_dir) / "alive"
    alive.mkdir(parents=True, exist_ok=True)
    (alive / session_id).write_text("", encoding="utf-8")


def is_alive(state_dir: Path, session_id: str) -> bool:
    return (Path(state_dir) / "alive" / session_id).exists()


def fake_env(state_dir: Path, calls_file: Path, home: Path) -> dict:
    return {
        "HOME": str(home),
        "FAKE_COLAB_DIR": str(state_dir),
        "FAKE_COLAB_CALLS": str(calls_file),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": os.environ.get("PATH", ""),
    }
