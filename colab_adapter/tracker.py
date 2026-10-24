"""Structured experiment logger for Colab GPU jobs.

Maintains non-sensitive metadata in JSON/JSONL format.
Ensures secrets, tokens, and authorization credentials are never logged.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    compute_units_before: Optional[str]
    compute_units_after: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Extra safety check: redact any sensitive key if accidentally present
        sensitive_keys = {"token", "auth", "secret", "password", "key", "credential"}
        for k in list(d.keys()):
            if any(s in k.lower() for s in sensitive_keys):
                d[k] = "[REDACTED]"
        return d


class ExperimentTracker:
    def __init__(self, log_dir: Optional[Path] = None):
        if log_dir is None:
            log_dir = Path.home() / ".colab_experiments"
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Ensure private permissions
        try:
            os.chmod(self.log_dir, 0o700)
        except OSError:
            pass
        self.jsonl_path = self.log_dir / "experiments.jsonl"

    def record_job(self, meta: ExperimentMetadata) -> None:
        data = meta.to_dict()
        line = json.dumps(data, ensure_ascii=False)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

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
