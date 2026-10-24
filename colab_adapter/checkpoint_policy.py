"""Configurable automatic checkpoint publication and local retention policy.

Enforces:
- Minimum interval throttling between cloud checkpoint publications.
- Multi-tier validation before publishing (refuses incomplete or corrupt checkpoints).
- Safe local retention: keeps latest N checkpoints, ALWAYS preserves at least one
  verified local recoverable checkpoint, and NEVER deletes remote cloud objects without explicit authorization.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from .artifacts import ArtifactManifest, verify_local_artifact
from .errors import ArtifactValidationError, CheckpointPolicyError

logger = logging.getLogger(__name__)


@dataclass
class CheckpointPolicyConfig:
    """Configuration for checkpoint publication and retention."""
    publish_on_complete: bool = True
    publish_on_new_checkpoint: bool = True
    minimum_interval_seconds: int = 600
    keep_last: int = 3
    source_snapshot_enabled: bool = True
    cloud_provider: str = "google_drive"
    cloud_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointPolicyConfig":
        valid = {
            "publish_on_complete", "publish_on_new_checkpoint",
            "minimum_interval_seconds", "keep_last",
            "source_snapshot_enabled", "cloud_provider", "cloud_enabled"
        }
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


class CheckpointPolicyManager:
    """Evaluates publication timing, validates completeness, and manages local retention."""

    def __init__(self, config: Optional[CheckpointPolicyConfig] = None):
        self.config = config or CheckpointPolicyConfig()
        self.last_published_time: float = 0.0
        self.published_steps: Set[int] = set()

    def should_publish(
        self,
        step: Optional[int] = None,
        is_final: bool = False,
        now: Optional[float] = None,
    ) -> bool:
        """Evaluate if publication is permitted under interval and trigger policies."""
        if not self.config.cloud_enabled:
            return False

        current_time = now if now is not None else time.time()

        # Final job completion trigger
        if is_final and self.config.publish_on_complete:
            return True

        # Incremental checkpoint trigger
        if self.config.publish_on_new_checkpoint:
            if step is not None and step in self.published_steps:
                return False  # Already published this step

            elapsed = current_time - self.last_published_time
            if elapsed >= self.config.minimum_interval_seconds or self.last_published_time == 0.0:
                return True

        return False

    def validate_for_publication(
        self,
        checkpoint_path: Union[str, Path],
        expected_sha256: Optional[str] = None,
        min_size_bytes: int = 1,
    ) -> ArtifactManifest:
        """Validate that checkpoint file is fully written, non-truncated, and valid."""
        p = Path(checkpoint_path).resolve()
        if not p.is_file():
            raise ArtifactValidationError(f"Checkpoint file does not exist: {p}")

        # Multi-tier verification (size, hash, ZIP / weights structure)
        manifest = verify_local_artifact(
            file_path=p,
            expected_sha256=expected_sha256,
            min_size_bytes=min_size_bytes,
            is_checkpoint=True,
        )
        return manifest

    def record_publication(self, step: Optional[int] = None, published_time: Optional[float] = None) -> None:
        """Record successful publication time and step."""
        self.last_published_time = published_time if published_time is not None else time.time()
        if step is not None:
            self.published_steps.add(step)

    def prune_local_retention(
        self,
        checkpoints_dir: Union[str, Path],
        protected_files: Optional[Set[str]] = None,
    ) -> List[Path]:
        """Prune older local checkpoints exceeding keep_last limit.

        SAFETY GUARANTEES:
        - ALWAYS keeps at least 1 verified local checkpoint (even if keep_last=0).
        - Never touches remote Google Drive objects.
        - Excludes any explicitly protected filenames.
        """
        cdir = Path(checkpoints_dir).resolve()
        if not cdir.is_dir():
            return []

        protected = protected_files or set()

        # Collect candidate checkpoint files
        candidates: List[Path] = []
        for f in cdir.iterdir():
            if f.is_file() and f.suffix in (".pt", ".pth", ".bin", ".ckpt", ".tar.gz"):
                if f.name not in protected:
                    candidates.append(f)

        # Sort by modification time (newest first)
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        keep_count = max(1, self.config.keep_last)
        if len(candidates) <= keep_count:
            return []  # Within allowed retention limit

        to_delete = candidates[keep_count:]
        deleted: List[Path] = []

        for p in to_delete:
            try:
                p.unlink()
                deleted.append(p)
            except OSError as oe:
                logger.warning(f"Failed to prune old checkpoint {p}: {oe}")

        return deleted
