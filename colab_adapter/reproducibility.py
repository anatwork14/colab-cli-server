"""Code snapshot generation and experiment reproducibility manifests for Colab V2.

Produces deterministic source snapshots (source_snapshot.tar.gz), source manifests,
environment descriptions, and comprehensive run manifests with secret filtering.
"""
from __future__ import annotations

import io
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from .storage.base import _utc_now_iso
from .storage.manifest import compute_file_sha256

# File patterns strictly excluded from source snapshots to prevent secret/data leaks
EXCLUDED_PATTERNS = [
    re.compile(r"^\.git(/|$)"),
    re.compile(r"(^|/)\.env($|\..*)"),
    re.compile(r"\.(key|pem|p12|pfx|pkcs12)$", re.IGNORECASE),
    re.compile(r"(^|/)id_(rsa|dsa|ecdsa|ed25519)($|\.)"),
    re.compile(r"(^|/)credentials(\.json)?$", re.IGNORECASE),
    re.compile(r"(^|/)secrets(\.json)?$", re.IGNORECASE),
    re.compile(r"(^|/)(__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache)(/|$)"),
    re.compile(r"\.pyc$"),
    re.compile(r"(^|/)\.venv(/|$)"),
    re.compile(r"(^|/)node_modules(/|$)"),
    re.compile(r"(^|/)\.DS_Store$"),
]


def is_excluded_path(rel_path: str) -> bool:
    """Return True if rel_path matches any sensitive or cache pattern."""
    normalized = rel_path.replace("\\", "/")
    for pattern in EXCLUDED_PATTERNS:
        if pattern.search(normalized):
            return True
    return False


def get_git_provenance(project_dir: Path) -> Dict[str, Any]:
    """Capture git commit, branch, and dirty status if inside a git repository."""
    info: Dict[str, Any] = {
        "is_git": False,
        "commit": None,
        "branch": None,
        "dirty": False,
    }
    git_dir = project_dir / ".git"
    if not git_dir.exists():
        return info

    try:
        # Commit hash
        c = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        info["commit"] = c.stdout.strip()
        info["is_git"] = True

        # Branch
        b = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        if b.returncode == 0:
            info["branch"] = b.stdout.strip()

        # Dirty status
        d = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        if d.returncode == 0:
            info["dirty"] = len(d.stdout.strip()) > 0
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return info


def get_environment_info() -> Dict[str, Any]:
    """Capture deterministic runtime environment information without secrets."""
    packages: Dict[str, str] = {}
    try:
        # Query installed packages via pip list if available
        p = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
        )
        if p.returncode == 0:
            data = json.loads(p.stdout)
            packages = {item["name"]: item["version"] for item in data}
    except Exception:
        pass

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "installed_packages": packages,
        "captured_at": _utc_now_iso(),
    }


def create_deterministic_source_snapshot(
    project_dir: Union[str, Path],
    dest_tar_gz: Union[str, Path],
    extra_excludes: Optional[List[str]] = None,
) -> Tuple[Path, Dict[str, Dict[str, Any]]]:
    """Create a deterministic, bit-reproducible source_snapshot.tar.gz.

    All tar entry timestamps are clamped to epoch 0, ownership is zeroed,
    and files are added in strictly sorted order.
    Returns (dest_path, source_manifest_dict).
    """
    src_dir = Path(project_dir).resolve()
    dest = Path(dest_tar_gz).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    extra_patterns = [re.compile(re.escape(e)) for e in (extra_excludes or [])]
    source_manifest: Dict[str, Dict[str, Any]] = {}

    all_files: List[Tuple[str, Path]] = []
    for root, dirs, files in os.walk(src_dir):
        # Sort in-place for determinism
        dirs.sort()
        files.sort()
        for f in files:
            full_path = Path(root) / f
            rel_path = str(full_path.relative_to(src_dir)).replace("\\", "/")
            if is_excluded_path(rel_path):
                continue
            if any(p.search(rel_path) for p in extra_patterns):
                continue
            all_files.append((rel_path, full_path))

    all_files.sort(key=lambda x: x[0])

    # Write tar.gz with deterministic gzip and tar headers (mtime=0)
    import gzip

    with open(dest, "wb") as f_out:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f_out, mtime=0.0) as gz_out:
            with tarfile.open(mode="w", fileobj=gz_out) as tar:
                for rel_path, full_path in all_files:
                    sha256 = compute_file_sha256(full_path)
                    stat_res = full_path.stat()
                    source_manifest[rel_path] = {
                        "sha256": sha256,
                        "size_bytes": stat_res.st_size,
                    }

                    ti = tar.gettarinfo(str(full_path), arcname=rel_path)
                    ti.mtime = 0
                    ti.uid = 0
                    ti.gid = 0
                    ti.uname = ""
                    ti.gname = ""
                    if ti.isreg():
                        with open(full_path, "rb") as f_in:
                            tar.addfile(ti, f_in)
                    else:
                        tar.addfile(ti)

    return dest, source_manifest



@dataclass
class RunManifest:
    """Complete experiment reproducibility manifest."""
    project_id: str
    run_id: str
    entrypoint: str
    arguments: List[str] = field(default_factory=list)
    git_provenance: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    source_snapshot_sha256: Optional[str] = None
    source_files: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    requested_gpu: Optional[str] = None
    started_at: str = field(default_factory=_utc_now_iso)
    ended_at: Optional[str] = None
    status: str = "INITIALIZED"  # INITIALIZED, RUNNING, COMPLETED, FAILED
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunManifest":
        valid = {
            "project_id", "run_id", "entrypoint", "arguments",
            "git_provenance", "environment", "source_snapshot_sha256",
            "source_files", "requested_gpu", "started_at", "ended_at",
            "status", "artifacts", "metadata"
        }
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)

    def save(self, dest_path: Union[str, Path]) -> Path:
        dest = Path(dest_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / f"{dest.name}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(dest))
        return dest
