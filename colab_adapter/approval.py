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

import errno
import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import termios
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .project_manifest import ProjectSpec, plan_project
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
_ID_RE = re.compile(r"^apr-[0-9a-f]{32}$")


class GlobalCapExceededError(DuplicateAllocationError):
    """Raised when a new allocation would exceed the shared managed-session cap."""

    reason = "global_cap_exceeded"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def digest_of(plan: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(plan)).hexdigest()


def plan_fingerprint(spec: ProjectSpec) -> str:
    """Digest binding a project's identity, GPU, remote root, manifest and workspace contents."""
    plan = plan_project(spec)
    project_id = getattr(spec, "project_id", None) or f"proj-{spec.name}"
    return digest_of({
        "schema": SCHEMA,
        "project_id": project_id,
        "name": spec.name,
        "gpu": spec.gpu,
        "remote_root": spec.remote_root,
        "manifest": spec.to_dict(),
        "workspace_digest": plan["manifest_digest"],
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


def _stdio_is_tty() -> bool:
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def _open_tty() -> int:
    return os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)


def read_tty_phrase(prompt: str) -> str:
    """Read one line from the controlling terminal (/dev/tty), never from piped stdin."""
    if not _stdio_is_tty():
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

    def _key(self) -> bytes:
        path = self.directory / ".hmac.key"
        for _ in range(2):
            try:
                with open(path, "rb") as fh:
                    key = fh.read()
                if len(key) >= 32:
                    return key
            except FileNotFoundError:
                pass
            try:
                fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                continue
            with os.fdopen(fd, "wb") as fh:
                fh.write(secrets.token_bytes(32))
                fh.flush()
                os.fsync(fh.fileno())
        raise ApprovalRequiredError("Approval key is unreadable; refusing to use approvals.", reason="corrupt")

    def _mac(self, rec: Dict[str, Any]) -> str:
        body = {k: v for k, v in rec.items() if k != "mac"}
        return hmac.new(self._key(), canonical_json(body), hashlib.sha256).hexdigest()

    def _path(self, approval_id: str) -> Path:
        if not isinstance(approval_id, str) or not _ID_RE.match(approval_id):
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

    def approve(self, approval_id: str) -> Dict[str, Any]:
        """Operator step: show the summary, read the exact phrase from /dev/tty, mark approved."""
        with self._lock():
            rec = self._read(approval_id)
        if rec["status"] != "pending":
            raise ApprovalRequiredError(f"Approval is {rec['status']}, not pending.", reason=rec["status"])
        self._check_live(rec, _utcnow())

        print(f"Approval {rec['approval_id']} ({rec['kind']}): {rec['summary']}", file=sys.stderr)
        print(f"Plan digest: {rec['plan_digest']}", file=sys.stderr)
        entered = read_tty_phrase(f"To approve, type exactly: {rec['phrase']}\n> ")

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
