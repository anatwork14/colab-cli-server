"""Human approval gate for Colab allocations (Issue #25).

Every billable allocation made through this adapter needs a one-time approval record:

1. ``request`` (agent-runnable, no cloud side effects) writes a ``pending`` record that
   binds one exact plan digest, GPU, maximal lease and a short phrase.
2. ``approve`` (operator step) reads the phrase from ``/dev/tty`` and marks the record ``approved``.
3. ``verify_and_consume`` runs inside the allocation critical section, checks status, expiry,
   plan digest and lease, and marks the record ``consumed`` *before* the CLI is invoked.

Each record carries an HMAC over its fields using a per-directory key. The key lives in the
same OS account, so this is tamper-evidence and an audit aid, not an authority boundary.
An agent running as the same OS user can read the key, forge records, or drive a pty.
The real boundary is the operator running ``approve`` from a separate account or shell.
"""
from __future__ import annotations

import dataclasses
import errno
import fcntl
import glob
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import sys
import termios
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .project_manifest import ProjectSpec, discover_project_files, validate_manifest
from .provider import (
    ACTIVE_MANAGED_STATES,
    ApprovalRequiredError,
    DuplicateAllocationError,
    ManagedSessionRegistry,
)

SCHEMA = 1
APPROVAL_TTL_SECONDS = 900.0          # a pending or approved record is usable for 15 minutes
MAX_LEASE_SECONDS = 86400.0
MAX_ACTIVE_MANAGED_SESSIONS = 1       # shared cap across V1 and V2 allocations
KIND_V2_UP = "v2_up"
KIND_V1_JOB = "v1_job"
KIND_SERVER_RUN = "colab_server_run"
_ID_RE = re.compile(r"apr-[0-9a-f]{32}")  # always used with fullmatch(): no trailing-newline acceptance
_KEY_BYTES = 32
# Covered-file rules for plan_fingerprint(), on top of MANDATORY_EXCLUDE_PATTERNS (which already
# drop .git, __pycache__, *.pyc). Logs and the local state directory never change what a run uses.
FINGERPRINT_EXTRA_EXCLUDES = ("*.log", "**/.colab_experiments", "**/.colab_experiments/**")


class GlobalCapExceededError(DuplicateAllocationError):
    """Raised when a new allocation would exceed the shared managed-session cap."""

    reason = "global_cap_exceeded"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def digest_of(plan: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(plan)).hexdigest()


def _relative_inside_root(path: Path, root: Path) -> Optional[str]:
    """POSIX path of ``path`` relative to ``root`` when it lies strictly inside it, else None."""
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    rel_posix = rel.as_posix()
    return None if rel_posix in ("", ".") else rel_posix


def _fingerprint_excludes(spec: ProjectSpec) -> List[str]:
    """Extra exclude patterns for the workspace digest, including artifacts_local_dir when inside the root."""
    patterns = list(FINGERPRINT_EXTRA_EXCLUDES)
    artifacts = Path(spec.artifacts_local_dir)
    if not artifacts.is_absolute():
        artifacts = Path(spec.root) / artifacts
    rel = _relative_inside_root(artifacts, Path(spec.root))
    if rel is not None:
        escaped = glob.escape(rel)  # literal directory name even if it holds glob characters
        patterns.extend([escaped, escaped + "/**"])
    return patterns


def workspace_digest(spec: ProjectSpec) -> str:
    """Digest of the covered source files under the project root (see docs/SECURITY.md).

    Same hashing format as ``plan_project``'s ``manifest_digest``, but over a filtered file set.
    Not covered: artifacts_local_dir, the `.colab_experiments` state directory, `*.log`, `__pycache__`,
    `*.pyc` and `.git`. ``plan_project`` itself is unchanged, so `project plan` output is unaffected.
    """
    validate_manifest(spec)
    scoped = dataclasses.replace(spec, exclude=list(spec.exclude) + _fingerprint_excludes(spec))
    files = discover_project_files(scoped)
    return hashlib.sha256(
        json.dumps([(f[0], f[1], f[2]) for f in files], sort_keys=True).encode("utf-8")
    ).hexdigest()


def plan_fingerprint(spec: ProjectSpec) -> str:
    """Digest binding a project's identity, GPU, remote root, manifest and covered source files."""
    project_id = getattr(spec, "project_id", None) or f"proj-{spec.name}"
    return digest_of({
        "schema": SCHEMA,
        "project_id": project_id,
        "name": spec.name,
        "gpu": spec.gpu,
        "remote_root": spec.remote_root,
        "manifest": spec.to_dict(),
        "workspace_digest": workspace_digest(spec),
    })


def v2_plan(spec: ProjectSpec, colab_binary: str) -> Dict[str, Any]:
    project_id = getattr(spec, "project_id", None) or f"proj-{spec.name}"
    return {
        "schema": SCHEMA,
        "kind": KIND_V2_UP,
        "project_id": project_id,
        "name": spec.name,
        "gpu": spec.gpu,
        "remote_root": spec.remote_root,
        "fingerprint": plan_fingerprint(spec),
        "colab_binary": colab_binary,
    }


def v1_job_plan(prep: Any) -> Dict[str, Any]:
    spec = prep.spec
    return {
        "schema": SCHEMA,
        "kind": KIND_V1_JOB,
        "job_id": prep.job_id,
        "gpu": spec.gpu.upper(),
        "fingerprint": prep.fingerprint,
        "script_sha256": prep.script_sha256,
        "reqs_sha256": prep.reqs_sha256,
        "inputs_sha256": prep.inputs_sha256,
        "stateful": bool(spec.stateful),
        "timeout": float(spec.timeout),
    }


def default_approvals_dir() -> Path:
    return Path.home() / ".colab_experiments" / "approvals"


def approvals_dir_for_registry(registry: ManagedSessionRegistry) -> Path:
    return Path(registry.registry_path).parent / "approvals"


def count_active_managed(registry: ManagedSessionRegistry) -> int:
    return sum(
        1 for info in registry.get_all_managed().values()
        if info.get("state") in ACTIVE_MANAGED_STATES
    )


def ensure_global_cap(registry: ManagedSessionRegistry, cap: int = MAX_ACTIVE_MANAGED_SESSIONS) -> None:
    """Refuse a new allocation when the shared active-session cap is already reached."""
    active = count_active_managed(registry)
    if active >= cap:
        raise GlobalCapExceededError(
            f"Cannot allocate: {active} managed Colab session(s) already active; the global cap is {cap}. "
            "Stop or reconcile the existing managed session first."
        )


def _stdio_is_tty(require_stdout: bool = True) -> bool:
    """stdin must be a terminal. stdout must be one too unless the caller prints JSON (`--json`)."""
    try:
        if not sys.stdin.isatty():
            return False
        return bool(sys.stdout.isatty()) if require_stdout else True
    except (AttributeError, ValueError):
        return False


def _open_tty() -> int:
    return os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)


def read_tty_phrase(prompt: str, require_stdout_tty: bool = True) -> str:
    """Read one line from the controlling terminal (/dev/tty), never from piped stdin.

    The phrase is always read from /dev/tty. A caller that prints JSON to stdout passes
    ``require_stdout_tty=False``; stdin must still be a terminal.
    """
    if not _stdio_is_tty(require_stdout=require_stdout_tty):
        raise ApprovalRequiredError(
            "REFUSED: Colab GPU provisioning requires human authorization at an interactive terminal "
            "(stdin is not a tty).",
            reason="non_tty",
        )
    try:
        fd = _open_tty()
    except OSError as exc:
        name = errno.errorcode.get(exc.errno or 0, "error")
        raise ApprovalRequiredError(
            f"REFUSED: no controlling terminal available (/dev/tty: {name}).",
            reason="no_controlling_tty",
        ) from exc
    try:
        try:
            termios.tcflush(fd, termios.TCIFLUSH)  # drop keystrokes typed before the prompt
        except termios.error:
            pass
        os.write(fd, prompt.encode("utf-8"))
        buf = bytearray()
        while len(buf) < 256:
            chunk = os.read(fd, 1)
            if not chunk or chunk == b"\n":
                break
            buf += chunk
        return buf.decode("utf-8", "replace").strip()
    except OSError as exc:
        raise ApprovalRequiredError("REFUSED: failed to read from the controlling terminal.", reason="tty_error") from exc
    finally:
        os.close(fd)


class ApprovalStore:
    """File-backed, HMAC-tagged, single-use approval records under one directory."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    # ----- storage primitives -------------------------------------------------
    def _ensure_dir(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        os.chmod(self.directory, 0o700)

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self._ensure_dir()
        fd = os.open(str(self.directory / ".lock"), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _corrupt_key(self, detail: str) -> ApprovalRequiredError:
        return ApprovalRequiredError(f"Approval key is unusable ({detail}); refusing to use approvals.", reason="corrupt")

    def _read_key_file(self, path: Path) -> bytes:
        """Read the key without following symlinks; the opened descriptor must be a regular file."""
        try:
            fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            raise self._corrupt_key("cannot open") from exc
        with os.fdopen(fd, "rb") as fh:
            if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
                raise self._corrupt_key("not a regular file")
            return fh.read(4096)

    def _install_key(self, path: Path, replace: bool) -> bytes:
        """Write a new key via temp file + fsync. ``replace=False`` creates it with no clobber (os.link)."""
        key = secrets.token_bytes(_KEY_BYTES)
        tmp = self.directory / f".hmac.key.{os.getpid()}.{secrets.token_hex(4)}.tmp"
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(key)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            if replace:
                os.replace(str(tmp), str(path))
            else:
                os.link(str(tmp), str(path))  # EEXIST when another creator won the race
        except FileExistsError:
            return self._read_key_file(path)
        finally:
            try:
                os.unlink(str(tmp))
            except FileNotFoundError:
                pass
        dfd = os.open(str(self.directory), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        return key

    def _key(self) -> bytes:
        """Return the per-directory HMAC key, creating or repairing it. Callers hold ``_lock``.

        A key shorter than 32 bytes is a crash leftover: no record can have been signed with it, so it
        is replaced atomically. A symlink, a non-regular file or a key owned by another user is refused
        as ``corrupt``. Group and other permission bits are removed from our own key file.
        """
        path = self.directory / ".hmac.key"
        try:
            st = os.lstat(str(path))
        except FileNotFoundError:
            return self._install_key(path, replace=False)
        if not stat.S_ISREG(st.st_mode):
            raise self._corrupt_key("symlink or not a regular file")
        if hasattr(os, "getuid") and st.st_uid != os.getuid():
            raise self._corrupt_key("not owned by this user")
        if stat.S_IMODE(st.st_mode) & 0o077:
            os.chmod(str(path), 0o600)
        key = self._read_key_file(path)
        if len(key) >= _KEY_BYTES:
            return key
        return self._install_key(path, replace=True)

    def _mac(self, rec: Dict[str, Any]) -> str:
        body = {k: v for k, v in rec.items() if k != "mac"}
        return hmac.new(self._key(), canonical_json(body), hashlib.sha256).hexdigest()

    def _path(self, approval_id: str) -> Path:
        if not isinstance(approval_id, str) or not _ID_RE.fullmatch(approval_id):
            raise ApprovalRequiredError("Unknown approval id.", reason="not_found")
        return self.directory / f"{approval_id}.json"

    def _write(self, rec: Dict[str, Any]) -> None:
        rec = dict(rec)
        rec["mac"] = self._mac(rec)
        final = self._path(rec["approval_id"])
        tmp = self.directory / f".{rec['approval_id']}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True, indent=2) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(str(tmp), str(final))
        dfd = os.open(str(self.directory), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)

    def _read(self, approval_id: str) -> Dict[str, Any]:
        path = self._path(approval_id)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                rec = json.load(fh)
        except FileNotFoundError as exc:
            raise ApprovalRequiredError("No approval record with that id.", reason="not_found") from exc
        except (OSError, ValueError) as exc:
            raise ApprovalRequiredError("Approval record is unreadable.", reason="corrupt") from exc
        if not isinstance(rec, dict) or not isinstance(rec.get("mac"), str):
            raise ApprovalRequiredError("Approval record is malformed.", reason="invalid_mac")
        if not hmac.compare_digest(rec["mac"], self._mac(rec)):
            raise ApprovalRequiredError("Approval record failed integrity check (HMAC).", reason="invalid_mac")
        return rec

    @staticmethod
    def _check_live(rec: Dict[str, Any], now: datetime) -> None:
        expires = datetime.fromisoformat(rec["expires_at"])
        if now >= expires:
            raise ApprovalRequiredError("Approval has expired; request a new one.", reason="expired")

    # ----- public API ---------------------------------------------------------
    def request(
        self,
        kind: str,
        plan: Dict[str, Any],
        gpu: str,
        summary: str,
        phrase_code: str,
        max_lease_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Write a pending record. Performs no cloud side effects."""
        if max_lease_seconds is not None and not (0 < max_lease_seconds <= MAX_LEASE_SECONDS):
            raise ApprovalRequiredError("Approval lease must be > 0 and <= 86400 seconds.", reason="invalid_lease")
        now = _utcnow()
        phrase = f"ALLOCATE {gpu}" + (f" {phrase_code}" if phrase_code else "")
        rec = {
            "schema": SCHEMA,
            "approval_id": f"apr-{secrets.token_hex(16)}",  # 128-bit random id
            "kind": kind,
            "gpu": gpu,
            "summary": summary,
            "plan": plan,
            "plan_digest": digest_of(plan),
            "phrase": phrase,
            "max_lease_seconds": max_lease_seconds,
            "status": "pending",
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=APPROVAL_TTL_SECONDS)).isoformat(),
            "approved_at": None,
            "consumed_at": None,
        }
        with self._lock():
            self._write(rec)
        return rec

    def approve(self, approval_id: str, require_stdout_tty: bool = True) -> Dict[str, Any]:
        """Operator step: show the summary, read the exact phrase from /dev/tty, mark approved.

        ``require_stdout_tty=False`` serves callers that print JSON to stdout; stdin must still be a tty.
        """
        with self._lock():
            rec = self._read(approval_id)
        if rec["status"] != "pending":
            raise ApprovalRequiredError(f"Approval is {rec['status']}, not pending.", reason=rec["status"])
        self._check_live(rec, _utcnow())

        print(f"Approval {rec['approval_id']} ({rec['kind']}): {rec['summary']}", file=sys.stderr)
        print(f"Plan digest: {rec['plan_digest']}", file=sys.stderr)
        entered = read_tty_phrase(f"To approve, type exactly: {rec['phrase']}\n> ",
                                  require_stdout_tty=require_stdout_tty)

        with self._lock():
            rec = self._read(approval_id)
            if rec["status"] != "pending":
                raise ApprovalRequiredError(f"Approval is {rec['status']}, not pending.", reason=rec["status"])
            now = _utcnow()
            self._check_live(rec, now)
            if not hmac.compare_digest(entered.encode("utf-8"), rec["phrase"].encode("utf-8")):
                raise ApprovalRequiredError("Approval rejected: the entered phrase did not match.", reason="phrase_mismatch")
            rec["status"] = "approved"
            rec["approved_at"] = now.isoformat()
            self._write(rec)
            return rec

    def verify_and_consume(
        self,
        approval_id: Optional[str],
        kind: str,
        plan_digest: str,
        lease_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Atomically check an approval and mark it consumed. Call inside the allocation lock."""
        if not approval_id:
            raise ApprovalRequiredError("An approval id is required to allocate compute.", reason="missing")
        with self._lock():
            rec = self._read(approval_id)
            if rec["status"] == "consumed":
                raise ApprovalRequiredError("Approval was already consumed; request a new one.", reason="consumed")
            if rec["status"] != "approved":
                raise ApprovalRequiredError("Approval has not been granted by the operator.", reason="not_approved")
            self._check_live(rec, _utcnow())
            if rec["kind"] != kind:
                raise ApprovalRequiredError("Approval is for a different kind of allocation.", reason="wrong_kind")
            if not hmac.compare_digest(rec["plan_digest"], plan_digest):
                raise ApprovalRequiredError(
                    "Approval does not match this plan (project, GPU, fingerprint or inputs changed).",
                    reason="plan_mismatch",
                )
            max_lease = rec.get("max_lease_seconds")
            if lease_seconds is not None and max_lease is not None and float(lease_seconds) > float(max_lease):
                raise ApprovalRequiredError(
                    f"Requested lease {lease_seconds}s exceeds the approved maximum {max_lease}s.",
                    reason="lease_exceeded",
                )
            rec["status"] = "consumed"
            rec["consumed_at"] = _utcnow().isoformat()
            self._write(rec)
            return rec

    def public_view(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        keys = ("approval_id", "kind", "gpu", "summary", "phrase", "plan_digest", "max_lease_seconds",
                "status", "created_at", "expires_at", "approved_at", "consumed_at")
        return {k: rec.get(k) for k in keys}
