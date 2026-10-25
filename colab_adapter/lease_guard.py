"""Host-side lease deadline guard for ONE managed Colab session (internal).

What it is: a best-effort, session-scoped deadline enforcer. A detached process sleeps until the
session's lease deadline, then stops exactly that session through the project stop path, which
verifies the removal (`colab sessions` plus, when a baseline was recorded before `colab new`,
`colab usage` active assignments). Failures are retried with capped backoff, indefinitely by default.

What it is NOT: a billing cap, a hard guarantee, or a substitute for `project status`, `sessions`
and `usage` checks by the operator. It does not survive a host reboot, `kill -9` of the guard, or
loss of OAuth. `project reconcile` (auto-stop by default) is the backstop for dead guards and for
records left in NEEDS_RECONCILIATION, STOPPING or stale PROVISIONING.

Invariants: the guard only ever acts on its own session id, never loops over sessions, never stops
unmanaged sessions, never writes tokens or raw CLI output to its log, and never flips a TERMINATED
record back to NEEDS_RECONCILIATION.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

from .provider import ColabProcessLock, ManagedSessionRegistry, is_process_running

GUARD_DIR_NAME = "lease_guards"
GUARD_SESSION_ID_RE = re.compile(r"colab-v2-[A-Za-z0-9_\-]{1,40}")
TERMINAL_STATES = frozenset({"TERMINATED", "FAILED"})
# Upper bound on the time the guard waits for the project lock per stop attempt. A busy lock is retried
# and is never counted as a failed stop attempt.
LOCK_WAIT_S = 30.0
STOP_BACKOFF_BASE_S = 1.0
STOP_BACKOFF_CAP_S = 30.0
GUARD_MARKER = "lease-guard"

OUTCOME_STOPPED_VERIFIED = "STOPPED_VERIFIED"
OUTCOME_ENDED_EXTERNALLY = "ENDED_EXTERNALLY"
OUTCOME_SIGNALLED = "SIGNALLED_NO_STOP"
OUTCOME_NEEDS_RECONCILIATION = "NEEDS_RECONCILIATION"
OUTCOME_GUARD_ERROR = "GUARD_ERROR"

_REDACTIONS = (
    (re.compile(r"ya29\.[A-Za-z0-9._\-]+"), "[REDACTED]"),
    (re.compile(r"1//[A-Za-z0-9._\-]{8,}"), "[REDACTED]"),
    (re.compile(r"(?i)\b(bearer|token|secret|password|authorization)\b[\s:=]+\S+"), r"\1=[REDACTED]"),
    (re.compile(r"[A-Za-z0-9_\-/+=.]{32,}"), "[REDACTED]"),
)


def redact(text: str) -> str:
    """Remove token-looking strings from log text."""
    out = str(text)
    for pattern, replacement in _REDACTIONS:
        out = pattern.sub(replacement, out)
    return out


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def epoch_from_iso(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def default_backoff(attempt: int) -> float:
    """Exponential backoff for stop retries, capped at 30 s. attempt is 0-based."""
    return min(STOP_BACKOFF_CAP_S, STOP_BACKOFF_BASE_S * (2 ** max(0, attempt)))


def validate_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not GUARD_SESSION_ID_RE.fullmatch(session_id):
        raise ValueError("Invalid managed session id for lease guard.")
    return session_id


def guard_dir(registry_path: Path) -> Path:
    return Path(registry_path).parent / GUARD_DIR_NAME


def ensure_guard_dir(registry_path: Path) -> Path:
    d = guard_dir(registry_path)
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    return d


def guard_log_path(registry_path: Path, session_id: str) -> Path:
    return guard_dir(registry_path) / f"{validate_session_id(session_id)}.log"


def open_private_append(path: Path) -> int:
    """Open or create a 0600 file for appending and return its descriptor."""
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.fchmod(fd, 0o600)
    return fd


@contextlib.contextmanager
def guard_spawn_lock(registry_path: Path, session_id: str) -> Iterator[None]:
    """Serialize guard (re)spawns for one session so only one guard is started at a time."""
    ensure_guard_dir(registry_path)
    path = guard_dir(registry_path) / f"{validate_session_id(session_id)}.spawn.lock"
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def pid_argv(pid: int) -> Optional[List[str]]:
    """Return the argv of a live process, or None if it is gone, a zombie or unreadable."""
    if not pid or pid <= 0:
        return None
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        raw = None
    if raw is not None:
        parts = [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]
        return parts or None
    try:
        res = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = (res.stdout or "").strip()
    return out.split() if out else None


def guard_process_matches(pid: Optional[int], session_id: str) -> bool:
    """True only if pid is a live lease guard for exactly this session id."""
    try:
        pid_int = int(pid) if pid is not None else 0
    except (TypeError, ValueError):
        return False
    if pid_int <= 0 or not is_process_running(pid_int):
        return False
    argv = pid_argv(pid_int)
    return bool(argv) and GUARD_MARKER in argv and session_id in argv


def _read_record(registry: ManagedSessionRegistry, session_id: str) -> Optional[dict]:
    return registry.get_all_managed().get(session_id)


def _update_record(registry: ManagedSessionRegistry, session_id: str, mutate: Callable[[dict], object]) -> bool:
    """Read-modify-write one record under the registry lock. A mutate that returns False skips the write."""
    with registry._lock():
        data = registry._read_locked()
        if session_id not in data:
            return False
        if mutate(data[session_id]) is False:
            return False
        registry._atomic_write_locked(data)
        return True


def merge_guard_metadata(registry: ManagedSessionRegistry, session_id: str, updates: Dict[str, object]) -> None:
    def mutate(rec: dict) -> None:
        rec.setdefault("metadata", {}).update(updates)

    _update_record(registry, session_id, mutate)


def _mark_needs_reconciliation(registry: ManagedSessionRegistry, session_id: str, reason: str) -> None:
    """Flag the session for operator/reconcile attention. Re-reads under the registry lock.

    Never flips a TERMINATED or FAILED record back: a newer terminal state always wins.
    """

    def mutate(rec: dict) -> Optional[bool]:
        if rec.get("state") in TERMINAL_STATES:
            return False
        rec["state"] = "NEEDS_RECONCILIATION"
        rec["updated_at"] = iso_now()
        rec.setdefault("metadata", {})["last_transition_reason"] = reason
        return None

    _update_record(registry, session_id, mutate)


class _GuardLog:
    def __init__(self, path: Path):
        self.path = path

    def write(self, message: str) -> None:
        line = f"{iso_now()} {redact(message)}\n".encode("utf-8")
        fd = open_private_append(self.path)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)


class _SignalState:
    def __init__(self) -> None:
        self.requested = False


def _sleep_interruptible(seconds: float, state: _SignalState) -> None:
    end = time.monotonic() + max(0.0, seconds)
    while not state.requested:
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.2, remaining))


def run_guard(
    session_id: str,
    registry_path: Path,
    colab_binary: str,
    auth_mode: str,
    deadline_ts: float,
    poll_interval: float = 2.0,
    max_stop_attempts: Optional[int] = None,
    backoff: Optional[Callable[[int], float]] = None,
    lock: Optional[ColabProcessLock] = None,
    lock_wait_s: float = LOCK_WAIT_S,
) -> int:
    """Wait for the lease deadline, then stop exactly this session and verify its removal.

    Returns 0 when the session ended (stopped and verified, or ended by someone else before the
    deadline), 1 when the guard gave up (only possible when max_stop_attempts is set; production
    passes None, so the guard retries until verified, the record is terminal, or SIGTERM arrives).
    """
    from .project_sessions import (
        ManagerLockBusyError,
        ProjectSessionManager,
        StopNotFoundError,
        StopUnverifiedError,
    )

    validate_session_id(session_id)
    registry_path = Path(registry_path)
    ensure_guard_dir(registry_path)
    log = _GuardLog(guard_log_path(registry_path, session_id))
    registry = ManagedSessionRegistry(registry_path)
    backoff_fn = backoff or default_backoff
    state = _SignalState()

    def _on_sigterm(signum, frame):  # noqa: ARG001 - signature required by signal
        state.requested = True

    previous = None
    if hasattr(signal, "SIGTERM") and _is_main_thread():
        previous = signal.signal(signal.SIGTERM, _on_sigterm)

    def finish(outcome: str, code: int) -> int:
        merge_guard_metadata(registry, session_id, {"guard_outcome": outcome, "guard_finished_at": iso_now()})
        log.write(f"guard finished: outcome={outcome} exit={code}")
        return code

    try:
        merge_guard_metadata(registry, session_id, {
            "guard_started_at": iso_now(),
            "guard_pid": os.getpid(),
            "guard_deadline": datetime.fromtimestamp(deadline_ts, timezone.utc).isoformat(),
            "guard_outcome": None,
        })
        log.write(f"guard started for {session_id}")
        manager = ProjectSessionManager(registry=registry, lock=lock, colab_binary=colab_binary,
                                        auth_mode=auth_mode)

        # Phase 1: wait for the deadline; exit silently if the session ends first.
        while True:
            if state.requested:
                log.write("SIGTERM received before deadline; exiting without stopping")
                return finish(OUTCOME_SIGNALLED, 0)
            rec = _read_record(registry, session_id)
            if rec is None or rec.get("state") in TERMINAL_STATES:
                log.write("session record missing or terminal before deadline; nothing to do")
                return finish(OUTCOME_ENDED_EXTERNALLY, 0)
            remaining = deadline_ts - time.time()
            if remaining <= 0:
                break
            _sleep_interruptible(min(poll_interval, remaining), state)

        # Phase 2: deadline reached. Retry until verified (unbounded unless max_stop_attempts is set).
        reason = "deadline reached"
        failures = 0
        # True after a stop that exited 0 while `colab sessions` no longer listed the id but the usage
        # baseline check failed. Re-stopping would only report "not found", so only verification is retried.
        # The flag is persisted by the stop path, so a guard started later behaves the same way.
        start_rec = _read_record(registry, session_id) or {}
        verify_only = (start_rec.get("metadata") or {}).get("stop_unverified") == "usage"
        while True:
            if state.requested:
                log.write("SIGTERM received at deadline; exiting without stopping")
                return finish(OUTCOME_SIGNALLED, 0)
            rec = _read_record(registry, session_id)
            if rec is None or rec.get("state") == "TERMINATED":
                log.write("session ended before guard stop; nothing to do")
                return finish(OUTCOME_ENDED_EXTERNALLY, 0)
            if max_stop_attempts is not None and failures >= max_stop_attempts:
                break
            attempt_no = failures + 1
            try:
                if rec.get("state") == "FAILED" and manager.verify_stopped(session_id, lock_timeout=lock_wait_s):
                    log.write("record is FAILED and verified gone; nothing to do")
                    return finish(OUTCOME_ENDED_EXTERNALLY, 0)
                if verify_only:
                    if manager.verify_stopped(session_id, lock_timeout=lock_wait_s):
                        log.write(f"verification attempt {attempt_no} passed after an unverified stop")
                        return finish(OUTCOME_STOPPED_VERIFIED, 0)
                    failure = f"verification attempt {attempt_no} still failing after an unverified stop"
                else:
                    manager.stop(session_id, terminate_guard=False, lock_timeout=lock_wait_s)
                    log.write(f"stop attempt {attempt_no} verified")
                    return finish(OUTCOME_STOPPED_VERIFIED, 0)
            except ManagerLockBusyError:
                # Waiting on a busy project lock is not a failed attempt.
                log.write("project lock busy; waiting (not counted as a failed attempt)")
                continue
            except StopUnverifiedError as exc:
                verify_only = not exc.listed
                failure = f"stop attempt {attempt_no} returned success but was not verified: {exc}"
            except Exception as exc:  # noqa: BLE001 - classify and retry
                if verify_only:
                    failure = f"verification attempt {attempt_no} failed: {type(exc).__name__}"
                elif isinstance(exc, StopNotFoundError):
                    # Fixed, secret-free text owned by the adapter; keeps the orphan reason on the record.
                    failure = f"stop attempt {attempt_no} failed: {exc.reason}"
                else:
                    failure = f"stop attempt {attempt_no} failed: {type(exc).__name__}"
            failures += 1
            reason = failure
            _mark_needs_reconciliation(registry, session_id, reason)
            log.write(reason)
            if max_stop_attempts is not None and failures >= max_stop_attempts:
                break
            _sleep_interruptible(backoff_fn(failures - 1), state)

        _mark_needs_reconciliation(registry, session_id, f"guard gave up after {failures} attempts: {reason}")
        log.write(f"giving up after {failures} stop attempts (bounded by max_stop_attempts); flagged NEEDS_RECONCILIATION")
        return finish(OUTCOME_NEEDS_RECONCILIATION, 1)
    except Exception as exc:  # noqa: BLE001 - never leak messages that may contain secrets
        try:
            log.write(f"guard error: {type(exc).__name__}")
            merge_guard_metadata(registry, session_id, {"guard_outcome": OUTCOME_GUARD_ERROR})
        except Exception:  # noqa: BLE001
            pass
        return 1
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


def _is_main_thread() -> bool:
    import threading

    return threading.current_thread() is threading.main_thread()


def child_env() -> Dict[str, str]:
    """Environment for the detached guard: inherit, but make sure the package is importable."""
    env = dict(os.environ)
    package_parent = str(Path(__file__).resolve().parent.parent)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = package_parent + (os.pathsep + existing if existing else "")
    return env


def guard_command(session_id: str, registry_path: Path, deadline_ts: float,
                  colab_binary: str, auth_mode: str) -> List[str]:
    return [
        sys.executable, "-m", "colab_adapter.cli", "project", "lease-guard",
        "--session-id", session_id,
        "--registry-path", str(registry_path),
        "--deadline-ts", repr(float(deadline_ts)),
        "--colab-binary", colab_binary,
        "--auth-mode", auth_mode,
    ]
