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
    ids = alive_ids()
    if not ids:
        print("[colab] No active sessions found on server.")
    for sid in ids:
        print("[" + sid + "] ep-" + sid + " | Hardware: T4 | Shape: Standard | Variant: GPU")
    sys.exit(0)
if command == "usage":
    print("Current balance: 10.00 compute units")
    print("Usage rate: 1.00/hr")
    print("Active assignments: %d" % len(alive_ids()))
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
