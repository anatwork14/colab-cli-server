"""Two environment snapshots per window and one diagnostic dump per cell."""

import json
import os
import subprocess
import time
from pathlib import Path

FIELDS = ("index", "clocks.sm", "clocks.mem", "temperature.gpu", "clocks_event_reasons.active")


def _smi(*args):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible == "":
        return ""
    command = ["nvidia-smi", *args]
    if visible:
        command += ["-i", visible]
    return subprocess.check_output(command, text=True, timeout=10)


def snapshot(path, edge):
    row = {"edge": edge, "t_unix": time.time()}
    try:
        raw = _smi("--query-gpu=" + ",".join(FIELDS), "--format=csv,noheader,nounits")
        row["gpus"] = [
            dict(zip(FIELDS, [x.strip() for x in line.split(",")], strict=True))
            for line in raw.splitlines()
            if line.strip()
        ]
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        row["error"] = str(exc)
    with Path(path).open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def dump_cell(path):
    try:
        text = _smi("-q")
    except (OSError, subprocess.SubprocessError) as exc:
        text = str(exc)
    Path(path).write_text(text)
