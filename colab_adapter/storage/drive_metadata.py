"""Compact, API-compliant Google Drive ``appProperties`` contract.

Google Drive limits each custom property to 124 UTF-8 bytes counting the key
and the value together (and caps the number of properties per file). Full
canonical artifact IDs (``project/run/path@sha256_<64hex>``) therefore MUST NOT
be stored in a single Drive property. They live in ``PublicationManifest``;
Drive only carries compact, fixed-size, verifiable-by-content identifiers.

Schema v2 (all values are strings):

``schema``       ``"2"``
``sha256``       full 64-hex content digest of this revision (never truncated)
``path_sha256``  sha256 of the UTF-8 *logical artifact ID*
                 ``{project_id}/{run_id}/{logical_path}`` (collision-safe for
                 long / Unicode paths, binds the file to its project/run/path)
``is_latest``    ``"true"`` | ``"false"``
``next_sha256``  digest of the *immediate successor* revision; present only on
                 superseded (historical) revisions. The full canonical
                 successor ID is reconstructed as
                 ``{project_id}/{run_id}/{logical_path}@sha256_{next_sha256}``.

Legacy (pre-v2) properties ``logical_path``, ``revision_id``, ``project_id``,
``run_id``, ``superseded_by`` are READ for backward compatibility but never
written. Remote properties are metadata hints; content is always verified by
hashing the downloaded bytes, never trusted from a property.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from ..errors import DriveTransferError

DRIVE_PROPERTY_MAX_BYTES = 124  # key + value, UTF-8 bytes
DRIVE_MAX_APP_PROPERTIES = 30  # conservative per-file private property cap
SCHEMA_VERSION = "2"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")

STATE_LATEST = "latest"
STATE_HISTORICAL = "historical"
STATE_PENDING = "pending"

# Compact remote property recording the revision that was the authoritative
# latest of the same logical path when this pending upload was intended.
# Value is a full 64-hex sha256, or the literal BASE_NONE when no latest existed.
PROP_BASE_SHA256 = "base_sha256"
BASE_NONE = "none"

# pending_disposition() outcomes. No caller may promote a pending revision
# except on PROMOTE (or ALREADY_LATEST, which needs no write).
DISPOSITION_PROMOTE = "PROMOTE"
DISPOSITION_ALREADY_LATEST = "ALREADY_LATEST"
DISPOSITION_STALE = "STALE"
DISPOSITION_FAIL_CLOSED = "FAIL_CLOSED"


def is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX64.match(value))


def validate_drive_properties(properties: Optional[Mapping[str, Any]]) -> None:
    """Reject any property set the real Drive API would reject.

    Raises ``DriveTransferError`` *before* any remote write. Error text names
    only the key and sizes, never the value (values may contain paths).
    """
    if not properties:
        return
    if len(properties) > DRIVE_MAX_APP_PROPERTIES:
        raise DriveTransferError(
            f"Drive custom properties exceed maximum count ({len(properties)} > {DRIVE_MAX_APP_PROPERTIES})"
        )
    for key, value in properties.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise DriveTransferError("Drive custom property keys and values must be strings")
        if not key:
            raise DriveTransferError("Drive custom property key must not be empty")
        size = len(key.encode("utf-8")) + len(value.encode("utf-8"))
        if size > DRIVE_PROPERTY_MAX_BYTES:
            raise DriveTransferError(
                f"Drive custom property '{key}' exceeds {DRIVE_PROPERTY_MAX_BYTES} UTF-8 bytes ({size} bytes)"
            )


def logical_artifact_id(project_id: str, run_id: str, logical_path: str) -> str:
    return f"{project_id}/{run_id}/{logical_path.lstrip('/')}"


def path_digest(project_id: str, run_id: str, logical_path: str) -> str:
    """Deterministic full SHA256 of the logical artifact ID (never truncated)."""
    return hashlib.sha256(logical_artifact_id(project_id, run_id, logical_path).encode("utf-8")).hexdigest()


def normalize_base_sha256(value: Optional[str]) -> Optional[str]:
    """Validate a base marker: None (legacy/unknown), ``"none"`` or a full 64-hex sha256."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise DriveTransferError("Drive base_sha256 must be a string")
    v = value.strip().lower()
    if v == BASE_NONE:
        return BASE_NONE
    if not is_sha256_hex(v):
        raise DriveTransferError("Drive base_sha256 must be 'none' or a full 64-hex sha256")
    return v


def base_from_props(props: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Base marker persisted on a remote revision, or None for legacy files without one."""
    raw = (props or {}).get(PROP_BASE_SHA256)
    if not isinstance(raw, str) or not raw.strip():
        return None
    return normalize_base_sha256(raw)


def build_revision_properties(
    sha256: str,
    project_id: str,
    run_id: str,
    logical_path: str,
    is_latest: bool,
    next_sha256: Optional[str] = None,
    base_sha256: Optional[str] = None,
) -> Dict[str, str]:
    """Build and validate compact v2 properties for one revision."""
    sha = sha256.strip().lower()
    if not is_sha256_hex(sha):
        raise DriveTransferError("Drive revision metadata requires a full 64-hex sha256")
    props: Dict[str, str] = {
        "schema": SCHEMA_VERSION,
        "sha256": sha,
        "path_sha256": path_digest(project_id, run_id, logical_path),
        "is_latest": "true" if is_latest else "false",
    }
    if next_sha256:
        nxt = next_sha256.strip().lower()
        if not is_sha256_hex(nxt):
            raise DriveTransferError("Drive next_sha256 must be a full 64-hex sha256")
        props["next_sha256"] = nxt
    base = normalize_base_sha256(base_sha256)
    if base is not None:
        props[PROP_BASE_SHA256] = base
    validate_drive_properties(props)
    return props


def successor_sha256(props: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Digest of the immediate successor, from v2 ``next_sha256`` or legacy ``superseded_by``."""
    p = props or {}
    nxt = p.get("next_sha256")
    if isinstance(nxt, str) and nxt.strip():
        return nxt.strip().lower()
    legacy = p.get("superseded_by")
    if isinstance(legacy, str) and legacy.strip():
        m = re.search(r"sha256_([0-9a-fA-F]{64})$", legacy.strip())
        return m.group(1).lower() if m else legacy.strip().lower()
    return None


def revision_state(props: Optional[Mapping[str, Any]]) -> str:
    """Classify a remote revision from its properties.

    * ``latest``     - ``is_latest == "true"``
    * ``historical`` - superseded (has a successor marker); replaying it must
                       never change the latest pointer
    * ``pending``    - uploaded/verified but never promoted (interrupted publish)
    """
    p = props or {}
    if p.get("is_latest") == "true":
        return STATE_LATEST
    if successor_sha256(p):
        return STATE_HISTORICAL
    return STATE_PENDING


def props_match_path(props: Optional[Mapping[str, Any]], project_id: str, run_id: str, logical_path: str) -> Optional[bool]:
    """True/False if the properties identify a path, ``None`` if they carry no path info."""
    p = props or {}
    pd = p.get("path_sha256")
    if isinstance(pd, str) and pd:
        return pd.lower() == path_digest(project_id, run_id, logical_path)
    legacy = p.get("logical_path")
    if isinstance(legacy, str) and legacy:
        return legacy.lstrip("/") == logical_path.lstrip("/")
    return None


def canonical_successor_id(project_id: str, run_id: str, logical_path: str, next_sha256: str) -> str:
    return f"{logical_artifact_id(project_id, run_id, logical_path)}@sha256_{next_sha256.lower()}"


@dataclass(frozen=True)
class RemoteRevisionRef:
    """Observed remote revision: file id, content sha256 and immediate successor (if any)."""

    file_id: str
    sha256: str
    successor_sha256: Optional[str] = None


@dataclass(frozen=True)
class PendingDecision:
    disposition: str
    reason: str


def pending_disposition(
    *,
    pending_id: str,
    pending_sha256: str,
    pending_base_sha256: Optional[str],
    head: Optional[RemoteRevisionRef],
    base_revision: Optional[RemoteRevisionRef] = None,
    path_has_other_non_pending: bool = False,
) -> PendingDecision:
    """Decide what may happen to a pending (uploaded, never promoted) revision.

    Inputs are an observed snapshot: ``head`` is the unique remote revision marked
    latest for the path (``None`` if none), ``pending_base_sha256`` the base marker
    recorded at intent time (``"none"``, a sha, or ``None`` for legacy pending
    files), ``base_revision`` the remote file whose sha equals the base (looked up
    only when there is no head), and ``path_has_other_non_pending`` whether any
    non-pending revision of the path exists.

    * PROMOTE        - pending is the genuine successor of the current head (or the
                       interrupted coordination left no head and the base names the
                       pending revision as its successor).
    * ALREADY_LATEST - the head already is this content; idempotent no-op.
    * STALE          - the head moved on since the pending upload was intended; the
                       pending revision must stay pending (never promoted).
    * FAIL_CLOSED    - the remote state cannot be explained; never guess.
    """
    pending_sha = pending_sha256.strip().lower()
    base = normalize_base_sha256(pending_base_sha256)

    if head is None:
        if base is None or base == BASE_NONE:
            # No latest now. Legacy files without a marker are treated like "none": nothing can be
            # superseded unless some other non-pending revision of the path exists.
            if path_has_other_non_pending:
                return PendingDecision(DISPOSITION_FAIL_CLOSED, "no latest revision but other non-pending revisions exist")
            if base is None:
                return PendingDecision(DISPOSITION_PROMOTE, "legacy pending with no latest and no other revisions")
            return PendingDecision(DISPOSITION_PROMOTE, "no latest existed at intent time and none exists now")
        if (
            base_revision is not None
            and base_revision.sha256 == base
            and base_revision.successor_sha256 == pending_sha
        ):
            return PendingDecision(DISPOSITION_PROMOTE, "interrupted coordination: base names pending as successor")
        return PendingDecision(DISPOSITION_FAIL_CLOSED, "no latest revision and base does not name pending as successor")

    head_sha = head.sha256.strip().lower()
    if head.file_id == pending_id or head_sha == pending_sha:
        return PendingDecision(DISPOSITION_ALREADY_LATEST, "pending content is already the latest revision")

    if base is None:
        if head.successor_sha256 == pending_sha:
            return PendingDecision(DISPOSITION_PROMOTE, "legacy pending named as successor by current latest")
        return PendingDecision(DISPOSITION_FAIL_CLOSED, "legacy pending revision without base marker conflicts with latest")

    if base == head_sha:
        if head.successor_sha256 not in (None, pending_sha):
            return PendingDecision(DISPOSITION_FAIL_CLOSED, "latest revision names a different successor")
        return PendingDecision(DISPOSITION_PROMOTE, "pending is the genuine successor of the current latest")

    return PendingDecision(DISPOSITION_STALE, "latest moved on since the pending upload was intended")
