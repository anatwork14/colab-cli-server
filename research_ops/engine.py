"""Reproducibility guard: distinguish stock vLLM wheels from real modified source."""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

PINNED_VLLM_VERSION = "0.29.0"
MODES = ("stock", "source_python", "source_compiled")
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def git_value(root, *argv):
    return subprocess.run(["git", "-C", str(root)] + list(argv),
                          check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True,
                          timeout=20).stdout.strip()


def check_source(source_root, revision):
    if not source_root or not Path(source_root).is_absolute():
        raise ValueError("source_root must be an absolute checkout path")
    if not isinstance(revision, str) or not COMMIT.fullmatch(revision):
        raise ValueError("source_revision must be a full 40-hex Git commit SHA")
    root = Path(source_root).resolve()
    if not (root / "vllm" / "__init__.py").is_file():
        raise ValueError("source_root must contain vllm/__init__.py")
    current = git_value(root, "rev-parse", "HEAD")
    if current != revision:
        raise ValueError("vLLM checkout SHA differs from frozen engine revision")
    dirty = git_value(root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError("Tracked engine source modified; commit changes before comparison")
    return {"engine_source_path": str(root), "engine_git_sha": current,
            "engine_clean_tracked": True}


def plan(mode, source_root=None, source_revision=None):
    if mode not in MODES:
        raise ValueError("Unsupported vLLM mode")
    out = {"engine": "vllm", "vllm_version": PINNED_VLLM_VERSION,
           "mode": mode, "executes_commands": False,
           "source_changes_reach_running_engine": mode != "stock"}
    if mode == "stock":
        if source_root or source_revision:
            raise ValueError("Stock wheel cannot declare modified engine source")
        out["install"] = ["python -m pip install 'vllm==0.29.0'"]
    else:
        out.update(check_source(source_root, source_revision))
        if mode == "source_python":
            out["install"] = ["cd " + str(Path(source_root).resolve()),
                              "VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto"]
            out["limitations"] = "Python edits only; CUDA/C++ changes require full compilation."
        else:
            out["install"] = ["cd " + str(Path(source_root).resolve()),
                              "uv pip install --editable . --torch-backend=auto"]
            out["limitations"] = "Full native build: requires compatible CUDA, compiler, disk and time."
    return out


def verify(mode, source_root=None, source_revision=None, require_gpu=False):
    setup = plan(mode, source_root, source_revision)
    ver = importlib.metadata.version("vllm")
    if not (ver == PINNED_VLLM_VERSION or
            (mode != "stock" and ver.startswith(PINNED_VLLM_VERSION + "."))):
        raise ValueError("Wrong vLLM distribution version: " + ver)
    spec = importlib.util.find_spec("vllm")
    origin = Path(spec.origin).resolve() if spec is not None and spec.origin else None
    if origin is None:
        raise RuntimeError("Installed vllm has no inspectable import origin")
    if mode != "stock":
        root = Path(source_root).resolve()
        try:
            origin.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("Installed engine still imports stock wheel, NOT modified source: " +
                               str(origin)) from exc
    if require_gpu:
        if not shutil.which("nvidia-smi"):
            raise RuntimeError("nvidia-smi missing: cannot label this a GPU experiment")
        subprocess.run(["nvidia-smi", "-L"], check=True, stdout=subprocess.PIPE, timeout=20)
    return {"status": "ENGINE_IDENTITY_VERIFIED_NOT_BENCHMARK_RUN",
            "installed_version": ver, "import_origin": str(origin),
            "gpu_preflight": require_gpu, "engine": setup}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "verify"))
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--source-root")
    parser.add_argument("--source-revision")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if args.action == "plan":
        result = plan(args.mode, args.source_root, args.source_revision)
    else:
        result = verify(args.mode, args.source_root, args.source_revision, args.require_gpu)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
