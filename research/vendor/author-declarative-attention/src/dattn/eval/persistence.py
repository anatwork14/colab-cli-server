"""POSIX process locks and durable, unique-key JSONL persistence.

Every cooperating writer takes the same sidecar flock, which survives inode
replacement of the data file. A committed row is a complete newline-terminated
JSON record followed by fsync. Recovery truncates an interrupted final line only.
Malformed complete lines and duplicate successful keys are errors, never hidden.
A row in flight at death has no committed record and is generated again. A complete
record that reached disk before death is retained even if progress was not updated.
Locks are released by the kernel at process death. This requires a filesystem with
working POSIX flock and atomic same-directory rename (the campaign's local mount).
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def lock(path: Path, *, blocking: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def atomic_bytes(path: Path, data: bytes):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            temporary.replace(path)
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            temporary.unlink(missing_ok=True)


def key(row):
    return row["subset"], row["row_index"], row.get("response_index", 0)


def failed(row):
    return bool(row.get("transport_error", bool(row.get("error"))))


def _read(path):
    if not path.exists():
        return []
    with path.open("r+b") as handle:
        data = handle.read()
        if data and not data.endswith(b"\n"):
            end = data.rfind(b"\n") + 1
            handle.truncate(end)
            handle.flush()
            os.fsync(handle.fileno())
            data = data[:end]
    return [json.loads(line) for line in data.splitlines()]


def read_records(path: Path):
    with lock(path.with_suffix(".lock")):
        return _read(path)


def append_records(path: Path, records):
    """Persist unique keys and return the number actually added or replaced."""
    with lock(path.with_suffix(".lock")):
        old = _read(path)
        known = {}
        for row in old:
            k = key(row)
            if k in known and not failed(known[k]) and not failed(row):
                raise ValueError(f"duplicate successful record: {k}")
            if k not in known or failed(known[k]):
                known[k] = row
        added = []
        rewrite = len(known) != len(old)
        for row in records:
            k = key(row)
            if k in known:
                if not failed(known[k]):
                    continue
                rewrite = True
            known[k] = row
            added.append(row)
        if rewrite:
            atomic_bytes(path, b"".join((json.dumps(r) + "\n").encode() for r in known.values()))
        elif added:
            with path.open("ab") as handle:
                for row in added:
                    handle.write((json.dumps(row) + "\n").encode())
                handle.flush()
                os.fsync(handle.fileno())
        return len({key(row) for row in added})
