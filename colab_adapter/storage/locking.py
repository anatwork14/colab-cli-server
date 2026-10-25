"""Reusable exclusive lock guarding one publication manifest (cross-thread and cross-process).

Lock identity and location
--------------------------
The lock key is the manifest file path. For a manifest stored at ``<run_dir>/manifest.json``
the lock file is always::

    <run_dir>/.locks/manifest.lock

Both the lock file location and its key are derived only from the manifest path, so every
writer that resolves the same manifest (for example the local store and the Google Drive
store, which merges into the same manifest) contends on the same lock. The ``.locks``
directory is created with mode 0700 and the lock file with mode 0600. Symlinked lock
directories or lock files are refused (``O_NOFOLLOW`` plus ``lstat`` checks).

Semantics
---------
* Thread level: one ``threading.RLock`` per lock key, so threads of one process serialise.
* Re-entrancy: a reference count per key lets the owning thread nest the context manager
  without deadlocking. The OS-level ``flock`` is taken once, when the count goes 0 -> 1, and
  released when it returns to 0. This is required because ``flock`` locks are per open file
  description, so a second descriptor in the same process would block on itself.
* Process level: ``fcntl.flock(LOCK_EX)`` polled with ``LOCK_NB`` until the deadline. The
  kernel releases the lock automatically if the holder dies (for example on SIGKILL).
* Bounded wait: every acquisition waits at most ``timeout`` seconds and then raises
  ``JournalLockError`` (a ``StorageError`` from ``colab_adapter.errors``).
* ``fcntl`` is imported behind a guard. Where it is unavailable (non-POSIX hosts) only the
  thread-level lock is enforced; ``CROSS_PROCESS_SUPPORTED`` reports which mode applies.
"""
from __future__ import annotations

import contextlib
import errno
import os
import stat
import threading
import time
from pathlib import Path
from typing import Dict, Iterator, Optional, Union

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX hosts
    fcntl = None

from ..errors import JournalLockError, StorageError

LOCKS_DIR_NAME = ".locks"
MANIFEST_LOCK_FILENAME = "manifest.lock"
DEFAULT_LOCK_TIMEOUT_S = 60.0
CROSS_PROCESS_SUPPORTED = fcntl is not None

_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_FLOCK_RETRY_SLEEP_S = 0.01
_BUSY_ERRNOS = frozenset({errno.EAGAIN, errno.EACCES, getattr(errno, "EWOULDBLOCK", errno.EAGAIN)})


class _LockState:
    """Per-key bookkeeping. Mutated only while ``rlock`` is held by the calling thread."""

    __slots__ = ("rlock", "refcount", "fd")

    def __init__(self) -> None:
        self.rlock = threading.RLock()
        self.refcount = 0
        self.fd: Optional[int] = None


_STATES: Dict[str, _LockState] = {}
_STATES_MUTEX = threading.Lock()


def manifest_lock_path(manifest_path: Union[str, Path]) -> Path:
    """Return the deterministic lock file path for a manifest path (``<run_dir>/.locks/manifest.lock``)."""
    run_dir = Path(manifest_path).resolve().parent
    return run_dir / LOCKS_DIR_NAME / MANIFEST_LOCK_FILENAME


def _state_for(key: str) -> _LockState:
    with _STATES_MUTEX:
        state = _STATES.get(key)
        if state is None:
            state = _LockState()
            _STATES[key] = state
        return state


def _ensure_lock_dir(lock_dir: Path) -> None:
    """Create ``lock_dir`` (mode 0700) if needed; refuse symlinks and non-directories."""
    try:
        st = os.lstat(str(lock_dir))
    except FileNotFoundError:
        try:
            # The run directory may not exist yet (first publish through the Drive store).
            os.makedirs(str(lock_dir.parent), mode=0o700, exist_ok=True)
            os.mkdir(str(lock_dir), 0o700)
        except FileExistsError:
            pass
        except OSError as exc:
            raise StorageError(
                f"Cannot create manifest lock directory {lock_dir}: {exc}",
                details={"path": str(lock_dir)},
            ) from exc
        st = os.lstat(str(lock_dir))
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise StorageError(
            f"Manifest lock directory is a symlink or not a directory: {lock_dir}",
            details={"path": str(lock_dir)},
        )
    try:
        os.chmod(str(lock_dir), 0o700)
    except OSError:
        pass


def _open_lock_file(lock_file: Path) -> int:
    _ensure_lock_dir(lock_file.parent)
    try:
        fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT | _O_NOFOLLOW, 0o600)
    except OSError as exc:
        raise StorageError(
            f"Cannot open manifest lock file {lock_file}: {exc}",
            details={"path": str(lock_file)},
        ) from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise StorageError(
                f"Manifest lock path is not a regular file: {lock_file}",
                details={"path": str(lock_file)},
            )
        try:
            os.chmod(fd, 0o600)
        except OSError:
            pass
    except BaseException:
        os.close(fd)
        raise
    return fd


def _flock_until(fd: int, deadline: float, lock_name: str, timeout: float) -> None:
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError as exc:
            if exc.errno not in _BUSY_ERRNOS:
                raise StorageError(f"flock failed for {lock_name}: {exc}") from exc
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise JournalLockError(
                f"Cross-process manifest lock acquisition timed out after {timeout}s for {lock_name}",
                details={"lock": lock_name, "timeout_s": timeout},
            )
        time.sleep(min(_FLOCK_RETRY_SLEEP_S, remaining))


@contextlib.contextmanager
def manifest_lock(
    manifest_path: Union[str, Path],
    timeout: float = DEFAULT_LOCK_TIMEOUT_S,
) -> Iterator[Path]:
    """Hold the exclusive lock for ``manifest_path`` for the duration of the ``with`` block.

    Yields the lock file path. Raises ``JournalLockError`` if the lock cannot be obtained
    within ``timeout`` seconds. Re-entrant for the owning thread.
    """
    timeout = max(0.0, float(timeout))
    lock_file = manifest_lock_path(manifest_path)
    key = str(lock_file)
    state = _state_for(key)
    deadline = time.monotonic() + timeout

    if not state.rlock.acquire(timeout=timeout):
        raise JournalLockError(
            f"Thread manifest lock acquisition timed out after {timeout}s for {lock_file}",
            details={"lock": str(lock_file), "timeout_s": timeout},
        )
    try:
        if state.refcount == 0 and CROSS_PROCESS_SUPPORTED:
            fd = _open_lock_file(lock_file)
            try:
                _flock_until(fd, deadline, str(lock_file), timeout)
            except BaseException:
                os.close(fd)
                raise
            state.fd = fd
        state.refcount += 1
        try:
            yield lock_file
        finally:
            state.refcount -= 1
            if state.refcount == 0 and state.fd is not None:
                fd = state.fd
                state.fd = None
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                finally:
                    os.close(fd)
    finally:
        state.rlock.release()
