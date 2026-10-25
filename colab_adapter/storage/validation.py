"""Shared validators for untrusted storage inputs (project/run IDs, paths, filenames, types).

Pure functions: they never touch the filesystem, so callers can run them before any side
effect. Contract C3 / C9:

* project and run IDs are ASCII only, ``[A-Za-z0-9][A-Za-z0-9_.-]{0,127}``, checked with
  ``re.fullmatch`` (never ``$``, which accepts a trailing newline), and must not contain ``..``.
* file names and logical path segments may be Unicode, but must already be NFC-normalized and
  must not contain control, format, or line/paragraph separator characters, separators, or
  the names ``.`` and ``..``.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from ..errors import ArtifactValidationError

MAX_ID_LENGTH = 128
MAX_FILENAME_BYTES = 255
MAX_LOGICAL_PATH_LENGTH = 512

_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,%d}" % (MAX_ID_LENGTH - 1))

# Categories never allowed in file names or logical path segments.
_FORBIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})

_ARTIFACT_TYPES = frozenset(
    {"checkpoint", "log", "metrics", "source_snapshot", "manifest", "environment", "output"}
)


def _describe(value: Any) -> str:
    """Render an untrusted value safely for an error message (escapes control characters)."""
    return ascii(value)[:200]


def _reject(field: str, value: Any, reason: str) -> ArtifactValidationError:
    return ArtifactValidationError(
        f"Invalid {field} {_describe(value)}: {reason}.",
        details={field: _describe(value)},
    )


def validate_identifier(value: Any, field: str = "identifier") -> str:
    """Validate an ASCII path-safe identifier (project_id, run_id)."""
    if not isinstance(value, str):
        raise _reject(field, value, "must be a string")
    if not _ID_RE.fullmatch(value):
        raise _reject(
            field,
            value,
            "must be 1-%d ASCII characters of [A-Za-z0-9_.-] starting with a letter or digit" % MAX_ID_LENGTH,
        )
    if ".." in value:
        raise _reject(field, value, "'..' sequences are not allowed")
    return value


def validate_project_id(value: Any) -> str:
    """Validate a project identifier. Raises ArtifactValidationError."""
    return validate_identifier(value, "project_id")


def validate_run_id(value: Any) -> str:
    """Validate a run identifier. Raises ArtifactValidationError."""
    return validate_identifier(value, "run_id")


def validate_filename(value: Any) -> str:
    """Validate a bare file name: NFC Unicode allowed, no separators, no '.'/'..', no control chars."""
    field = "filename"
    if not isinstance(value, str):
        raise _reject(field, value, "must be a string")
    if not value or len(value.encode("utf-8", "strict")) > MAX_FILENAME_BYTES:
        raise _reject(field, value, "must be 1-%d bytes" % MAX_FILENAME_BYTES)
    if value in (".", ".."):
        raise _reject(field, value, "dot segments are not allowed")
    if value.startswith("."):
        raise _reject(field, value, "hidden (leading '.') names are reserved for store metadata")
    if "/" in value or "\\" in value:
        raise _reject(field, value, "path separators are not allowed")
    if any(unicodedata.category(ch) in _FORBIDDEN_CATEGORIES for ch in value):
        raise _reject(field, value, "control, format or line/paragraph separator characters are not allowed")
    if not unicodedata.is_normalized("NFC", value):
        raise _reject(field, value, "must be Unicode NFC-normalized")
    return value


def validate_logical_path(value: Any) -> str:
    """Validate a relative logical artifact path such as 'checkpoints/model.pt'.

    Every segment must satisfy the file name rule; absolute paths, empty segments and
    '.'/'..' segments are rejected.
    """
    field = "logical_path"
    if not isinstance(value, str):
        raise _reject(field, value, "must be a string")
    if not value or len(value) > MAX_LOGICAL_PATH_LENGTH:
        raise _reject(field, value, "must be between 1 and %d characters" % MAX_LOGICAL_PATH_LENGTH)
    if value.startswith("/"):
        raise _reject(field, value, "absolute paths are not allowed")
    for segment in value.split("/"):
        if not segment:
            raise _reject(field, value, "empty path segments are not allowed")
        try:
            validate_filename(segment)
        except ArtifactValidationError:
            raise _reject(field, value, "contains an invalid path segment")
    return value


def validate_artifact_type(value: Any) -> str:
    """Validate an artifact type against the known ArtifactType values."""
    if not isinstance(value, str) or value not in _ARTIFACT_TYPES:
        raise _reject("artifact_type", value, "unknown artifact type")
    return value
