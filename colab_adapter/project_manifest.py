"""Project Manifest Schema, Validation, and Safe Offline Planning (V2).

Provides:
- ProjectSpec: Typed specification for multi-module research workspaces.
- load_manifest: Reads JSON (stdlib) or YAML (optional PyYAML) manifests.
- validate_manifest: Security checks for path traversal, symlink escapes, and secrets.
- plan_project: 100% offline file discovery, SHA256 hashing, and transfer byte estimation.
"""
from dataclasses import asdict, dataclass, field
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from .errors import ManifestValidationError

SUPPORTED_SCHEMA_VERSIONS = {1}
SUPPORTED_ACCELERATORS = {"T4", "L4", "A100", "H100", "V100", "CPU"}

# Mandatory deny-list patterns that CANNOT be overridden by include rules
MANDATORY_EXCLUDE_PATTERNS = [
    ".git/**",
    ".git",
    ".venv/**",
    ".venv",
    "venv/**",
    "venv",
    "**/__pycache__/**",
    "**/__pycache__",
    "*.pyc",
    "*.pyo",
    ".env*",
    "**/.env*",
    "*token*",
    "**/*token*",
    "*.pem",
    "*.key",
    "*.id_rsa*",
    "*.id_ed25519*",
    "**/.config/colab-cli/**",
    "**/.colab-cli-oauth*",
]


@dataclass
class ProjectSpec:
    """Specification of a Colab GPU project workspace."""

    name: str
    root: Path
    entrypoint: str
    schema_version: int = 1
    gpu: str = "T4"
    timeout_seconds: float = 900.0
    retain_session: bool = False
    remote_root: Optional[str] = None
    requirements: Optional[str] = None
    args: List[str] = field(default_factory=list)
    include: List[str] = field(default_factory=lambda: ["**/*"])
    exclude: List[str] = field(default_factory=list)
    delete_remote: bool = False
    artifacts_pull: List[str] = field(default_factory=list)
    artifacts_local_dir: Path = field(default_factory=lambda: Path("./artifacts"))

    def __post_init__(self):
        self.root = Path(self.root).resolve()
        if not self.remote_root:
            clean_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", self.name)
            self.remote_root = f"/content/colab-workspaces/{clean_name}"
        self.artifacts_local_dir = Path(self.artifacts_local_dir)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["root"] = str(self.root)
        d["artifacts_local_dir"] = str(self.artifacts_local_dir)
        return d


def load_manifest(path: Union[str, Path]) -> ProjectSpec:
    """Loads a project manifest from a JSON or YAML file."""
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise ManifestValidationError(f"Manifest file not found: {manifest_path}")

    content = manifest_path.read_text(encoding="utf-8")
    ext = manifest_path.suffix.lower()

    if ext in (".yaml", ".yml"):
        try:
            import yaml
            raw = yaml.safe_load(content)
        except ImportError:
            raise ManifestValidationError(
                "PyYAML is not installed. To parse YAML manifests, install PyYAML "
                "or convert your manifest to JSON (e.g. colab-project.json)."
            )
        except Exception as e:
            raise ManifestValidationError(f"Invalid YAML syntax in {manifest_path}: {e}")
    elif ext == ".json":
        try:
            raw = json.loads(content)
        except Exception as e:
            raise ManifestValidationError(f"Invalid JSON syntax in {manifest_path}: {e}")
    else:
        # Try JSON first, then YAML if available
        try:
            raw = json.loads(content)
        except Exception:
            try:
                import yaml
                raw = yaml.safe_load(content)
            except Exception as e:
                raise ManifestValidationError(
                    f"Unsupported or invalid manifest format for {manifest_path}: {e}"
                )

    if not isinstance(raw, dict):
        raise ManifestValidationError("Manifest root must be a dictionary.")

    return _parse_raw_manifest(raw, base_dir=manifest_path.parent)


def _parse_raw_manifest(raw: Dict[str, Any], base_dir: Path) -> ProjectSpec:
    """Parses a raw dictionary into a validated ProjectSpec."""
    schema_version = raw.get("schema_version", 1)
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ManifestValidationError(
            f"Unsupported schema_version: {schema_version}. Supported versions: {SUPPORTED_SCHEMA_VERSIONS}"
        )

    # Support nested sections or flat dicts
    proj_sec = raw.get("project", {})
    runtime_sec = raw.get("runtime", {})
    workspace_sec = raw.get("workspace", {})
    exec_sec = raw.get("execution", {})
    artifacts_sec = raw.get("artifacts", {})

    name = proj_sec.get("name") or raw.get("name")
    if not name or not isinstance(name, str) or not name.strip():
        raise ManifestValidationError("Project 'name' is required.")

    raw_root = proj_sec.get("root") or raw.get("root") or "."
    resolved_root = (base_dir / raw_root).resolve()

    gpu = (runtime_sec.get("gpu") or raw.get("gpu") or "T4").upper()
    timeout = float(runtime_sec.get("timeout_seconds") or raw.get("timeout_seconds") or 900.0)
    retain = bool(runtime_sec.get("retain_session") or raw.get("retain_session") or False)

    entrypoint = exec_sec.get("entrypoint") or raw.get("entrypoint")
    if not entrypoint or not isinstance(entrypoint, str):
        raise ManifestValidationError("Execution 'entrypoint' is required.")

    reqs = exec_sec.get("requirements") or raw.get("requirements")
    args = exec_sec.get("args") or raw.get("args") or []
    if not isinstance(args, list):
        raise ManifestValidationError("'args' must be a list of strings.")

    remote_root = workspace_sec.get("remote_root") or raw.get("remote_root")
    includes = workspace_sec.get("include") or raw.get("include") or ["**/*"]
    excludes = workspace_sec.get("exclude") or raw.get("exclude") or []
    delete_remote = bool(workspace_sec.get("delete_remote") or raw.get("delete_remote") or False)

    pull = artifacts_sec.get("pull") or raw.get("artifacts_pull") or []
    local_art_dir = artifacts_sec.get("local_dir") or raw.get("artifacts_local_dir") or "./artifacts"

    spec = ProjectSpec(
        schema_version=schema_version,
        name=name.strip(),
        root=resolved_root,
        entrypoint=entrypoint.strip(),
        gpu=gpu,
        timeout_seconds=timeout,
        retain_session=retain,
        remote_root=remote_root,
        requirements=reqs.strip() if reqs else None,
        args=[str(a) for a in args],
        include=list(includes) if isinstance(includes, list) else [str(includes)],
        exclude=list(excludes) if isinstance(excludes, list) else [str(excludes)],
        delete_remote=delete_remote,
        artifacts_pull=list(pull) if isinstance(pull, list) else [str(pull)],
        artifacts_local_dir=(resolved_root / local_art_dir).resolve(),
    )

    validate_manifest(spec)
    return spec


def validate_manifest(spec: ProjectSpec) -> None:
    """Validates paths, security rules, and parameter boundaries for a ProjectSpec."""
    if not spec.root.is_dir():
        raise ManifestValidationError(f"Project root directory does not exist: {spec.root}")

    # Check GPU accelerator
    if spec.gpu not in SUPPORTED_ACCELERATORS:
        raise ManifestValidationError(
            f"Unsupported accelerator '{spec.gpu}'. Supported: {sorted(SUPPORTED_ACCELERATORS)}"
        )

    # Check timeouts
    if spec.timeout_seconds < 10.0 or spec.timeout_seconds > 86400.0:
        raise ManifestValidationError(
            f"timeout_seconds ({spec.timeout_seconds}) must be between 10.0 and 86400.0 (24h)."
        )

    # Validate entrypoint: must be relative and reside under root
    _validate_safe_relative_path(spec.entrypoint, spec.root, "entrypoint")
    entrypoint_path = (spec.root / spec.entrypoint).resolve()
    if not entrypoint_path.is_file():
        raise ManifestValidationError(f"Entrypoint file not found: {entrypoint_path}")

    # Validate requirements if specified
    if spec.requirements:
        _validate_safe_relative_path(spec.requirements, spec.root, "requirements")
        reqs_path = (spec.root / spec.requirements).resolve()
        if not reqs_path.is_file():
            raise ManifestValidationError(f"Requirements file not found: {reqs_path}")

    # Validate artifacts_pull
    for art in spec.artifacts_pull:
        if os.path.isabs(art) or ".." in art.split("/"):
            raise ManifestValidationError(f"Invalid artifact pull path '{art}': must be relative.")


def _validate_safe_relative_path(rel_path: str, root_dir: Path, field_name: str) -> None:
    """Rejects absolute paths, '..' escapes, and path traversal."""
    if not rel_path or not rel_path.strip():
        raise ManifestValidationError(f"{field_name} cannot be empty.")
    p = Path(rel_path)
    if p.is_absolute():
        raise ManifestValidationError(f"{field_name} must be a relative path, got absolute: {rel_path}")
    parts = p.parts
    if ".." in parts:
        raise ManifestValidationError(
            f"Path traversal detected in {field_name} ('{rel_path}'): '..' is not allowed."
        )
    # Target resolution check
    resolved = (root_dir / p).resolve()
    try:
        common = os.path.commonpath([str(root_dir), str(resolved)])
        if common != str(root_dir):
            raise ManifestValidationError(
                f"{field_name} ('{rel_path}') resolves outside project root: {resolved}"
            )
    except ValueError:
        raise ManifestValidationError(f"{field_name} resolves to a different drive/mount.")


def is_path_excluded(rel_posix: str, custom_excludes: Sequence[str]) -> bool:
    """Checks whether a normalized relative POSIX path matches any exclude pattern."""
    all_excludes = list(MANDATORY_EXCLUDE_PATTERNS) + list(custom_excludes)
    for pattern in all_excludes:
        if fnmatch.fnmatch(rel_posix, pattern) or fnmatch.fnmatch(f"/{rel_posix}", pattern):
            return True
        # Match directory prefixes (e.g. .git/** matches .git/config)
        clean_pat = pattern.rstrip("/*")
        if rel_posix == clean_pat or rel_posix.startswith(f"{clean_pat}/"):
            return True
    return False


def is_path_included(rel_posix: str, includes: Sequence[str]) -> bool:
    """Checks whether a normalized relative POSIX path matches any include pattern."""
    if not includes or "**/*" in includes or "*" in includes:
        return True
    for pattern in includes:
        if fnmatch.fnmatch(rel_posix, pattern) or fnmatch.fnmatch(f"/{rel_posix}", pattern):
            return True
        clean_pat = pattern.rstrip("/*")
        if rel_posix == clean_pat or rel_posix.startswith(f"{clean_pat}/"):
            return True
    return False


def discover_project_files(spec: ProjectSpec) -> List[Tuple[str, int, str]]:
    """Traverses spec.root, applies include/exclude filters, checks symlinks,

    and returns a sorted list of (rel_path_posix, size_bytes, sha256_hash).
    """
    results: List[Tuple[str, int, str]] = []
    root = spec.root

    for current_dir, dirnames, filenames in os.walk(str(root), followlinks=False):
        current_path = Path(current_dir)
        rel_dir = current_path.relative_to(root).as_posix()
        if rel_dir == ".":
            rel_dir = ""

        # Filter directories in-place to avoid traversing into .git, .venv, etc.
        filtered_dirs = []
        for d in dirnames:
            dir_rel = f"{rel_dir}/{d}" if rel_dir else d
            if not is_path_excluded(dir_rel, spec.exclude):
                filtered_dirs.append(d)
        dirnames[:] = filtered_dirs

        for fname in filenames:
            file_rel = f"{rel_dir}/{fname}" if rel_dir else fname
            full_path = current_path / fname

            # 1. Check exclusions first (excluded files/symlinks do not trigger errors)
            if is_path_excluded(file_rel, spec.exclude):
                continue

            # 2. Symlink escape check for included files
            if full_path.is_symlink():
                target = full_path.resolve()
                try:
                    common = os.path.commonpath([str(root), str(target)])
                    if common != str(root):
                        raise ManifestValidationError(
                            f"Symlink security escape detected: '{file_rel}' points outside project root ({target})"
                        )
                except ValueError:
                    raise ManifestValidationError(
                        f"Symlink points across drive boundary: '{file_rel}' -> {target}"
                    )
                if not target.is_file():
                    # Dangling symlink
                    continue

            # 3. Check regular file
            if not full_path.is_file():
                continue

            # 4. Check inclusions
            if not is_path_included(file_rel, spec.include):
                continue

            # 4. Compute size and sha256
            stat = full_path.stat()
            size = stat.st_size
            sha = _hash_file(full_path)
            results.append((file_rel, size, sha))

    results.sort(key=lambda x: x[0])
    return results


def _hash_file(path: Path) -> str:
    """Computes SHA256 digest of a file in 64KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def plan_project(spec: ProjectSpec) -> Dict[str, Any]:
    """Generates a complete, deterministic offline plan for project workspace execution.
    
    Zero VM allocation, zero subprocess execution.
    """
    validate_manifest(spec)
    files = discover_project_files(spec)
    total_bytes = sum(f[1] for f in files)

    return {
        "schema_version": spec.schema_version,
        "project_name": spec.name,
        "local_root": str(spec.root),
        "remote_root": spec.remote_root,
        "gpu": spec.gpu,
        "timeout_seconds": spec.timeout_seconds,
        "retain_session": spec.retain_session,
        "entrypoint": spec.entrypoint,
        "requirements": spec.requirements,
        "args": spec.args,
        "artifacts_pull": spec.artifacts_pull,
        "artifacts_local_dir": str(spec.artifacts_local_dir),
        "delete_remote": spec.delete_remote,
        "files_count": len(files),
        "total_bytes": total_bytes,
        "files": [
            {"path": f[0], "size": f[1], "sha256": f[2]}
            for f in files
        ],
        "mandatory_excludes": MANDATORY_EXCLUDE_PATTERNS,
    }


def init_project(target_dir: Union[str, Path], name: Optional[str] = None) -> Path:
    """Initializes a starter colab-project.json manifest in target_dir."""
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    proj_name = name or target_path.name or "colab-project"

    manifest_file = target_path / "colab-project.json"
    if manifest_file.exists():
        raise ManifestValidationError(f"Manifest already exists: {manifest_file}")

    template = {
        "schema_version": 1,
        "project": {
            "name": proj_name,
            "root": ".",
        },
        "runtime": {
            "gpu": "T4",
            "timeout_seconds": 900.0,
            "retain_session": False,
        },
        "workspace": {
            "remote_root": f"/content/colab-workspaces/{proj_name}",
            "include": ["**/*"],
            "exclude": ["datasets/**", "*.log"],
            "delete_remote": False,
        },
        "execution": {
            "entrypoint": "train.py",
            "requirements": "requirements.txt",
            "args": [],
        },
        "artifacts": {
            "pull": ["outputs/checkpoint.pt", "outputs/metrics.json"],
            "local_dir": "./artifacts",
        },
    }

    manifest_file.write_text(json.dumps(template, indent=2), encoding="utf-8")
    return manifest_file


load_project_manifest = load_manifest
create_execution_plan = plan_project
