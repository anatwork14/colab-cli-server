"""Immutable asset staging for inference experiments. Stdlib-only except optional HF downloads.

This module is deliberately provider-independent: run on the dedicated host to
prepare/publish data or inside an authorized Colab GPU to stage assets under
local ephemeral disk. It never allocates cloud resources or trains models.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import tempfile
from datetime import datetime, timezone
import zipfile

ID = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")
SHA = re.compile(r"^[0-9a-f]{64}$")
HF_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
RECEIPT = ".research_asset_receipt.json"
# Avoid arbitrary untrusted archive decompression and path explosion.
MAX_ARCHIVE_FILES = 200000
MAX_UNPACK_BYTES = 150 * (1024 ** 3)
CHUNK = 4 * 1024 * 1024


def _safe_relative(raw):
    if not isinstance(raw, str) or not raw or "\\" in raw or "\x00" in raw:
        raise ValueError("Unsafe relative path")
    p = PurePosixPath(raw)
    if p.is_absolute() or any(x in (".", "..", "") for x in raw.split("/")):
        raise ValueError("Relative paths must not contain . or .. or empty sections")
    if any(ord(ch) < 32 for ch in raw) or raw.startswith("~"):
        raise ValueError("Unsafe relative path")
    return p


def _hex64(value, label):
    if not isinstance(value, str) or SHA.fullmatch(value) is None:
        raise ValueError(label + " requires a lowercase SHA256")
    return value


def read_manifest(file):
    obj = json.loads(Path(file).read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or obj.get("schema_version") != 1:
        raise ValueError("Expected asset manifest schema_version=1")
    assets = obj.get("assets")
    if not isinstance(assets, list):
        raise ValueError("assets must be a list")
    seen = set()
    for a in assets:
        if not isinstance(a, dict) or not ID.fullmatch(str(a.get("id", ""))):
            raise ValueError("Invalid asset ID")
        if a["id"] in seen:
            raise ValueError("Duplicate asset ID")
        seen.add(a["id"])
        if a.get("role") not in ("model", "dataset"):
            raise ValueError("Asset role must be model or dataset")
        expected = a.get("expected_files")
        if not isinstance(expected, list) or not expected or len(set(expected)) != len(expected):
            raise ValueError("Nonempty unique expected_files needed")
        for f in expected:
            _safe_relative(f)
            if f == RECEIPT or f.startswith(".cache/"):
                raise ValueError("Expected file cannot be metadata")
        sources = a.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError("At least one pinned source required")
        for s in sources:
            if not isinstance(s, dict) or s.get("type") not in ("huggingface", "archive", "directory"):
                raise ValueError("Unknown source type")
            t = s["type"]
            if t == "huggingface":
                if not HF_ID.fullmatch(str(s.get("repo_id", ""))):
                    raise ValueError("Hugging Face namespace/repo_id required")
                rev = s.get("revision")
                if not isinstance(rev, str) or re.fullmatch(r"[0-9a-f]{40}", rev) is None:
                    raise ValueError("Hugging Face source must pin full 40-hex commit SHA")
                if s.get("repo_type", a["role"]) != a["role"]:
                    raise ValueError("HF repo_type must match asset role")
            else:
                path = s.get("path")
                if not isinstance(path, str) or not Path(path).is_absolute():
                    raise ValueError("Source directory/archive path must be absolute")
                if t == "archive":
                    _hex64(s.get("sha256"), "Archive")
                    if not path.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz")):
                        raise ValueError("Supported archives: ZIP, TAR, TAR.GZ")
                else:
                    _hex64(s.get("tree_sha256"), "Directory tree")
        if a.get("license_status", "NOT_REVIEWED") not in ("NOT_REVIEWED", "REVIEWED"):
            raise ValueError("license_status must be NOT_REVIEWED or REVIEWED")
    return obj


def hash_file(file):
    digest = hashlib.sha256()
    with Path(file).open("rb") as src:
        while True:
            chunk = src.read(CHUNK)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _tree_files(root):
    root = Path(root)
    for dirpath, dirs, files in os.walk(str(root), followlinks=False):
        dirs.sort()
        files.sort()
        for name in dirs:
            p = Path(dirpath) / name
            if p.is_symlink():
                raise ValueError("Symlink directory is not allowed: " + str(p))
        for name in files:
            p = Path(dirpath) / name
            if p.is_symlink() or not p.is_file() or not stat.S_ISREG(p.stat().st_mode):
                raise ValueError("Only regular files accepted: " + str(p))
            relative = p.relative_to(root).as_posix()
            if relative == RECEIPT or relative.startswith(".cache/"):
                continue
            _safe_relative(relative)
            yield p, relative


def tree_info(root):
    """Deterministic tree hash: relative path + bytes length + each file SHA256."""
    digest = hashlib.sha256()
    count = 0
    total = 0
    for path, name in _tree_files(root):
        size = path.stat().st_size
        h = hash_file(path)
        digest.update(json.dumps([name, size, h], separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n")
        count += 1
        total += size
    return {"tree_sha256": digest.hexdigest(), "file_count": count, "total_bytes": total}


def _find_asset(manifest, asset_id):
    for asset in manifest["assets"]:
        if asset["id"] == asset_id:
            return asset
    raise ValueError("Asset not listed in frozen manifest: " + asset_id)


def select_source(asset, index=None):
    """Select first *available* local source, else explicit HF source; never verify silently."""
    if index is not None:
        if index < 0 or index >= len(asset["sources"]):
            raise ValueError("Source index outside manifest")
        return index, asset["sources"][index]
    for i, src in enumerate(asset["sources"]):
        if src["type"] != "huggingface" and Path(src["path"]).exists():
            return i, src
    for i, src in enumerate(asset["sources"]):
        if src["type"] == "huggingface":
            return i, src
    raise FileNotFoundError("No available pinned local asset; no HF fallback configured")


def _safe_archive_member(name):
    # ZIP may contain backslashes, which on Windows are separators; reject for all platforms.
    trimmed = name[:-1] if name.endswith("/") else name
    return _safe_relative(trimmed)


def _unpack_archive(archive, destination):
    entries = set()
    used = 0
    count = 0

    def check(name, size):
        nonlocal count, used
        p = _safe_archive_member(name)
        normalized = p.as_posix()
        if normalized in entries:
            raise ValueError("Duplicate archive member: " + normalized)
        entries.add(normalized)
        count += 1
        used += size
        if count > MAX_ARCHIVE_FILES or used > MAX_UNPACK_BYTES or size < 0:
            raise ValueError("Archive limits exceeded")
        return destination / normalized

    def extract(stream, target, size):
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create: no overwrite even within the same archive.
        with target.open("xb") as f:
            remaining = size
            while remaining:
                part = stream.read(min(CHUNK, remaining))
                if not part:
                    raise ValueError("Archive truncated")
                f.write(part)
                remaining -= len(part)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if member.is_dir():
                    continue
                mode = (member.external_attr >> 16) & 0o170000
                if mode not in (0, stat.S_IFREG):
                    raise ValueError("Nonregular ZIP entry rejected")
                dest = check(member.filename, member.file_size)
                with z.open(member) as stream:
                    extract(stream, dest, member.file_size)
    else:
        with tarfile.open(archive, "r:*") as t:
            for member in t:
                if member.isdir():
                    continue
                if not member.isfile() or member.issym() or member.islnk():
                    raise ValueError("Nonregular TAR entry rejected")
                dest = check(member.name, member.size)
                with t.extractfile(member) as stream:
                    extract(stream, dest, member.size)
    return {"unpacked_files": count, "unpacked_bytes": used}


def _copy_directory(src, dest):
    # Fail on symlinks even when the root is a symlink. Existing directories
    # can be a persistent host cache, but their exact content must be pinned.
    if src.is_symlink() or not src.is_dir():
        raise ValueError("Local source must be an ordinary directory")
    for p, relative in _tree_files(src):
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(p), str(target))


def _check_expected(dest, asset):
    for rel in asset["expected_files"]:
        target = dest.joinpath(*_safe_relative(rel).parts)
        if not target.is_file() or target.is_symlink():
            raise ValueError("Expected file missing from staged asset: " + rel)


def _stage_source(src, tmp, asset):
    typ = src["type"]
    if typ == "archive":
        path = Path(src["path"])
        actual = hash_file(path)
        if actual != src["sha256"]:
            raise ValueError("Archive SHA256 mismatch; refusing silent fallback")
        _unpack_archive(path, tmp)
    elif typ == "directory":
        path = Path(src["path"])
        fingerprint = tree_info(path)["tree_sha256"]
        if fingerprint != src["tree_sha256"]:
            raise ValueError("Persistent directory changed; refusing staged copy")
        _copy_directory(path, tmp)
    else:
        # This is intentionally a lazy dependency on the GPU worker.
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("Install huggingface_hub in the GPU worker to fetch pinned assets") from exc
        snapshot_download(repo_id=src["repo_id"], revision=src["revision"],
                          repo_type=src.get("repo_type", asset["role"]),
                          local_dir=str(tmp))
        # The hub writes .cache/huggingface metadata in local_dir.
        # It is skipped in content fingerprints but retained to enable resumption.


def stage(manifest_file, asset_id, scratch, source_index=None):
    manifest = read_manifest(manifest_file)
    asset = _find_asset(manifest, asset_id)
    index, src = select_source(asset, source_index)
    root = Path(scratch).expanduser().resolve()
    # Never allow experiment data to be written into a git checkout.
    if (root / ".git").exists():
        raise ValueError("Refusing to stage into a Git repository")
    root.mkdir(parents=True, exist_ok=True)
    target = root / asset_id
    if target.exists():
        receipt = verify(manifest_file, asset_id, scratch)
        return dict(receipt, reused=True)
    with tempfile.TemporaryDirectory(prefix=".stage-" + asset_id + "-", dir=str(root)) as name:
        tmp = Path(name) / "payload"
        tmp.mkdir()
        _stage_source(src, tmp, asset)
        _check_expected(tmp, asset)
        info = tree_info(tmp)
        if not info["file_count"]:
            raise ValueError("Empty staged asset")
        if src["type"] == "directory" and info["tree_sha256"] != src["tree_sha256"]:
            raise ValueError("Source changed during staging")
        receipt = dict(info)
        receipt.update({"schema_version": 1, "asset_id": asset_id,
                        "role": asset["role"], "source_index": index,
                        "source": src, "source_manifest_sha256": hash_file(manifest_file),
                        "staged_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "VERIFIED_STAGED_ASSET"})
        (tmp / RECEIPT).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
        # A single atomic directory rename makes a completed asset visible.
        tmp.rename(target)
        return dict(receipt, reused=False, staged_path=str(target))


def verify(manifest_file, asset_id, scratch):
    manifest = read_manifest(manifest_file)
    asset = _find_asset(manifest, asset_id)
    target = Path(scratch).expanduser().resolve() / asset_id
    path = target / RECEIPT
    if not path.is_file() or path.is_symlink():
        raise ValueError("Missing verified asset receipt")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("asset_id") != asset_id or receipt.get("status") != "VERIFIED_STAGED_ASSET":
        raise ValueError("Invalid staged asset receipt")
    if receipt.get("source_manifest_sha256") != hash_file(manifest_file):
        raise ValueError("Asset manifest changed: must create new run, not reuse")
    idx = receipt.get("source_index")
    if type(idx) is not int or idx >= len(asset["sources"]) or idx < 0:
        raise ValueError("Invalid source index")
    if receipt.get("source") != asset["sources"][idx]:
        raise ValueError("Source provenance changed")
    _check_expected(target, asset)
    info = tree_info(target)
    if any(info[k] != receipt.get(k) for k in ("tree_sha256", "file_count", "total_bytes")):
        raise ValueError("Asset content changed or is incomplete")
    return dict(receipt, staged_path=str(target), verified=True)


def plan(manifest_file, asset_id, scratch, source_index=None):
    manifest = read_manifest(manifest_file)
    asset = _find_asset(manifest, asset_id)
    index, src = select_source(asset, source_index)
    root = Path(scratch).expanduser()
    target = root / asset_id
    return {"action": "PLAN_ONLY_NO_NETWORK", "asset_id": asset_id,
            "source_index": index, "source": src, "target": str(target),
            "target_exists": target.exists(), "role": asset["role"],
            "does_not_upload_to_drive": True,
            "require_execute_for_download_or_copy": True}


def pack(manifest_file, asset_id, scratch, output):
    """Make optional immutable TAR mirror from a verified staged asset.

    Pack locally first. Only put it in Google Drive when operator explicitly
    chooses that destination and has reviewed licensing/privacy/storage quota.
    """
    verified = verify(manifest_file, asset_id, scratch)
    source = Path(scratch).expanduser().resolve() / asset_id
    outfile = Path(output).expanduser()
    if not outfile.is_absolute() or outfile.suffix != ".tar":
        raise ValueError("Mirror output must be an absolute .tar path")
    outfile = outfile.resolve()
    if outfile.exists() or (Path(str(outfile) + ".sha256.json")).exists():
        raise FileExistsError("Refusing to overwrite existing cached archive/sidecar")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix="." + outfile.name + ".", dir=str(outfile.parent))
    os.close(fd)
    try:
        with tarfile.open(filename, "w") as tf:
            for path, relative in _tree_files(source):
                info = tf.gettarinfo(str(path), arcname=relative)
                info.uid, info.gid = 0, 0
                info.uname, info.gname = "", ""
                info.mtime = 0
                with path.open("rb") as reader:
                    tf.addfile(info, reader)
        sha256 = hash_file(filename)
        os.replace(filename, str(outfile))
        result = {"status": "VERIFIED_OPTIONAL_ARCHIVE_MIRROR",
                  "asset_id": asset_id, "archive": str(outfile),
                  "archive_sha256": sha256, "source_tree_sha256": verified["tree_sha256"],
                  "bytes": outfile.stat().st_size,
                  "source_template": {"type": "archive", "path": str(outfile),
                                      "sha256": sha256},
                  "warning": "Drive mirror is OPTIONAL. Remote upload/replication verification is not implied."}
        Path(str(outfile) + ".sha256.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    finally:
        if os.path.exists(filename):
            os.unlink(filename)


def main():
    p = argparse.ArgumentParser(description="Pinned real model/dataset staging; dry-run first.")
    p.add_argument("action", choices=("plan", "stage", "verify", "validate", "pack"))
    p.add_argument("--manifest", required=True)
    p.add_argument("--asset")
    p.add_argument("--scratch", default="/content/research-assets")
    p.add_argument("--source-index", type=int, default=None)
    p.add_argument("--output", help="Absolute .tar destination for optional verified mirror")
    p.add_argument("--execute", action="store_true",
                   help="Explicitly allow file copying/downloading into scratch")
    a = p.parse_args()
    if a.action == "validate":
        result = {"schema": 1, "asset_ids": [x["id"] for x in read_manifest(a.manifest)["assets"]],
                  "cloud_allocation": False}
    elif not a.asset:
        p.error("--asset is required except for validate")
    elif a.action == "plan":
        result = plan(a.manifest, a.asset, a.scratch, a.source_index)
    elif a.action == "verify":
        result = verify(a.manifest, a.asset, a.scratch)
    elif not a.execute:
        raise SystemExit("stage/pack require --execute (large file I/O); run plan first")
    elif a.action == "pack":
        if not a.output:
            p.error("pack requires --output /absolute/path/asset.tar")
        result = pack(a.manifest, a.asset, a.scratch, a.output)
    else:
        result = stage(a.manifest, a.asset, a.scratch, a.source_index)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
