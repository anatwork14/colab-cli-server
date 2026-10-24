"""Structured experiment logger for Colab GPU jobs.

Maintains non-sensitive metadata in JSON/JSONL format.
Ensures secrets, tokens, authorization credentials, and sensitive output
are deeply sanitized and redacted from persisted logs.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Sensitive dictionary key substrings to scrub
_SENSITIVE_KEYS = (
    "token",
    "auth",
    "secret",
    "password",
    "key",
    "credential",
    "resumable_session_url",
    "session_url",
    "upload_url",
    "proxy_secret",
    "trusted_proxy_secret",
    "x_proxy_secret",
    "x_api_token",
    "id_token",
)

# Regex patterns for detecting and redacting credentials in any string
SENSITIVE_PATTERNS = [
    # Google Drive resumable upload URLs
    (re.compile(r"https://www\.googleapis\.com/upload/drive/v3/files\?[^\s\"'<>]+"), "[REDACTED_UPLOAD_SESSION_URL]"),
    # Upload session query params
    (re.compile(r"(session_id|upload_id)=[A-Za-z0-9._\-]+"), r"\1=[REDACTED_SESSION_ID]"),
    # URL-encoded form parameters
    (re.compile(r"(client_secret|refresh_token|access_token|id_token)=[^&\s\"']+"), r"\1=[REDACTED_SECRET]"),
    # JSON string values for sensitive keys in raw text snippets
    (re.compile(r'(?i)((\\?["\'])(?:client_secret|refresh_token|access_token|resumable_session_url)\2\s*:\s*(\\?["\']))[^\r\n]*?(\3)'), r"\g<1>[REDACTED]\g<4>"),
    # Google OAuth authorization code (e.g. 4/0AXl...)
    (re.compile(r"4/[0-9a-zA-Z_\-]{20,}"), "[REDACTED_OAUTH_CODE]"),
    # Google OAuth Bearer / access token (ya29....)
    (re.compile(r"ya29\.[0-9a-zA-Z_\-]+"), "[REDACTED_GOOGLE_TOKEN]"),
    # GitHub / generic API tokens
    (re.compile(r"gh[pousr]_[0-9a-zA-Z]{20,}"), "[REDACTED_GITHUB_TOKEN]"),
    # JWT tokens
    (re.compile(r"ey[A-Za-z0-9_\-]{15,}\.ey[A-Za-z0-9_\-]{15,}\.[A-Za-z0-9_\-]+"), "[REDACTED_JWT]"),
    # PEM Private Keys
    (re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]+?-----END [A-Z ]+ PRIVATE KEY-----"), "[REDACTED_PRIVATE_KEY]"),
    # URL query parameters containing credentials
    (re.compile(r"([?&](?:code|token|key|secret|password|access_token)=)[^&\s]+"), r"\1[REDACTED_PARAM]"),
    # Key-value secret assignments in text or tracebacks
    (re.compile(r"(?i)(?:token|secret|password|api_key|access_token|auth)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{8,})['\"]?"), "[REDACTED_SECRET]"),
]


def sanitize_text(text: Optional[str]) -> Optional[str]:
    """Recursively scrub known sensitive patterns from text."""
    if not text:
        return text
    sanitized = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def sanitize_data(data: Any) -> Any:
    """Recursively sanitize strings, dictionaries, and lists."""
    if isinstance(data, str):
        return sanitize_text(data)
    elif isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            # If the key itself looks like a secret field, redact its value entirely
            if any(s in str(k).lower() for s in _SENSITIVE_KEYS):
                cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = sanitize_data(v)
        return cleaned
    elif isinstance(data, list):
        return [sanitize_data(item) for item in data]
    return data


@dataclass
class ExperimentMetadata:
    job_id: str
    project_name: str
    requested_accelerator: str
    start_time: str
    completion_time: Optional[str]
    job_state: str  # PREPARED, APPROVED, RUNNING, COMPLETED, FAILED, CANCELLED
    script_path: str
    script_revision: Optional[str]
    dependencies: List[str]
    dataset_identifier: Optional[str]
    output_artifact_paths: List[str]
    success: bool
    error_summary: Optional[str]
    cleanup_status: str  # CLEANED, RUNTIME_STOPPED, FAILED, ORPHANED, NOT_APPLICABLE
    cleanup_error: Optional[str] = None
    usage_error: Optional[str] = None
    compute_units_before: Optional[str] = None
    compute_units_after: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Deeply sanitize all fields
        return sanitize_data(d)


class ExperimentTracker:
    def __init__(self, log_dir: Optional[Path] = None):
        if log_dir is None:
            log_dir = Path.home() / ".colab_experiments"
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Ensure private directory permissions (0700)
        try:
            os.chmod(self.log_dir, 0o700)
        except OSError:
            pass
        self.jsonl_path = self.log_dir / "experiments.jsonl"

    def record_job(self, meta: ExperimentMetadata) -> None:
        data = meta.to_dict()
        line = json.dumps(data, ensure_ascii=False) + "\n"

        # Write with strict 0600 permissions
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        fd = os.open(str(self.jsonl_path), flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.jsonl_path.exists():
            return []
        jobs: List[Dict[str, Any]] = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        jobs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return jobs[-limit:]

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        for job in reversed(self.list_jobs(limit=1000)):
            if job.get("job_id") == job_id:
                return job
        return None
