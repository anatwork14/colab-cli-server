"""Offline tests only. Bytes are unit-test payloads, never benchmark datasets."""
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from research_ops import assets, engine, experiments


def _write(root, name, content):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def _asset_manifest(root, *asset_defs):
    p = root / "asset-manifest.json"
    p.write_text(json.dumps({"schema_version": 1, "assets": list(asset_defs)}, sort_keys=True))
    return p


def _dir_asset(id, role, src, files):
    return {"id": id, "role": role, "expected_files": files,
            "sources": [{"type": "directory", "path": str(src.resolve()),
                         "tree_sha256": assets.tree_info(src)["tree_sha256"]}]}


class AssetPolicyTests(unittest.TestCase):
    def test_stage_verify_reuse_and_detect_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            src = root / "source"
            _write(src, "config.json", b'{"model_type":"test"}')
            _write(src, "weights.safetensors", b"unit-test-only")
            spec = _asset_manifest(root, _dir_asset("model-01", "model", src, ["config.json"]))
            scratch = root / "gpu-scratch"
            p = assets.plan(spec, "model-01", scratch)
            self.assertEqual(p["source"]["type"], "directory")
            self.assertFalse(scratch.exists())  # Plan cannot allocate or write
            first = assets.stage(spec, "model-01", scratch)
            self.assertFalse(first["reused"])
            self.assertTrue(assets.verify(spec, "model-01", scratch)["verified"])
            self.assertTrue(assets.stage(spec, "model-01", scratch)["reused"])
            (scratch / "model-01" / "weights.safetensors").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                assets.verify(spec, "model-01", scratch)

    def test_archive_sha_safe_unpack_optional_mirror(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "model.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("config.json", '{"model_type":"test"}')
                z.writestr("model.safetensors", "test fixture only")
            row = {"id": "model-01", "role": "model",
                   "expected_files": ["config.json"],
                   "sources": [{"type": "archive", "path": str(archive),
                                "sha256": assets.hash_file(archive)}]}
            spec = _asset_manifest(root, row)
            original_spec = root / "asset-original.json"
            original_spec.write_bytes(spec.read_bytes())
            scratch = root / "gpu"
            assets.stage(spec, "model-01", scratch)
            mirrored = assets.pack(spec, "model-01", scratch, root / "optional-mirror.tar")
            self.assertEqual(mirrored["source_template"]["type"], "archive")
            self.assertEqual(assets.hash_file(root / "optional-mirror.tar"),
                             mirrored["archive_sha256"])
            with self.assertRaises(FileExistsError):
                assets.pack(spec, "model-01", scratch, root / "optional-mirror.tar")
            row["sources"] = [mirrored["source_template"]]
            other = _asset_manifest(root, row)
            assets.stage(other, "model-01", root / "new-gpu")
            self.assertTrue(assets.verify(other, "model-01", root / "new-gpu")["verified"])
            archive.write_bytes(b"tampered-archive")
            with self.assertRaises(ValueError):
                assets.stage(original_spec, "model-01", root / "third-gpu")

    def test_reject_archive_traversal_and_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            traversal = root / "traversal.tar"
            with tarfile.open(traversal, "w") as tf:
                info = tarfile.TarInfo("../outside")
                info.size = 1
                tf.addfile(info, io.BytesIO(b"X"))
            with self.assertRaises(ValueError):
                assets._unpack_archive(traversal, root / "extract")
            self.assertFalse((root / "outside").exists())
            link = root / "link.tar"
            with tarfile.open(link, "w") as tf:
                info = tarfile.TarInfo("symlink")
                info.type = tarfile.SYMTYPE
                info.linkname = "/etc/passwd"
                tf.addfile(info)
            with self.assertRaises(ValueError):
                assets._unpack_archive(link, root / "extract2")

    def test_bad_pins_and_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row = {"id": "model-01", "role": "model",
                   "expected_files": ["../secret"],
                   "sources": [{"type": "huggingface", "repo_id": "Org/Repo",
                                "revision": "main"}]}
            spec = _asset_manifest(root, row)
            with self.assertRaises(ValueError):
                assets.read_manifest(spec)
            row["expected_files"] = ["config.json"]
            _asset_manifest(root, row)
            with self.assertRaises(ValueError):
                assets.read_manifest(spec)
            row["sources"][0]["revision"] = "a" * 40
            _asset_manifest(root, row)
            self.assertEqual(assets.read_manifest(spec)["assets"][0]["id"], "model-01")

    def test_no_silent_fallback_on_corrupt_local_mirror(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = _write(root, "model.tar", b"bad archive")
            row = {"id":"model-01", "role":"model", "expected_files":["config.json"],
                   "sources":[
                       {"type":"archive", "path":str(archive), "sha256":"0"*64},
                       {"type":"huggingface", "repo_id":"Org/Repo", "revision":"a"*40}
                   ]}
            spec = _asset_manifest(root, row)
            self.assertEqual(assets.plan(spec, "model-01", root / "scratch")["source_index"], 0)
            with self.assertRaises(ValueError):
                assets.stage(spec, "model-01", root / "scratch")
            self.assertFalse((root / "scratch" / "model-01").exists())


class EngineIdentityTests(unittest.TestCase):
    def test_stock_does_not_claim_modified_checkout(self):
        self.assertFalse(engine.plan("stock")["source_changes_reach_running_engine"])
        with self.assertRaises(ValueError):
            engine.plan("stock", "/tmp/checkout", "a" * 40)

    def test_modified_engine_requires_committed_clean_checkout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write(root, "vllm/__init__.py", b"# offline source fixture\n")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email",
                            "noreply@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name",
                            "OfflineTest"], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "source"], check=True)
            rev = engine.git_value(root, "rev-parse", "HEAD")
            self.assertEqual(engine.plan("source_python", str(root), rev)["engine_git_sha"], rev)
            (root / "vllm/__init__.py").write_text("# uncommitted modification\n")
            with self.assertRaises(ValueError):
                engine.plan("source_compiled", str(root), rev)


class ResearchFreezeTests(unittest.TestCase):
    def test_freeze_keeps_input_status_unexecuted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "cache-model"
            dataset = root / "cache-dataset"
            _write(model, "config.json", b"fixture")
            _write(dataset, "manifest.json", b"fixture")
            specs = _asset_manifest(root,
                _dir_asset("model-01", "model", model, ["config.json"]),
                _dir_asset("data-01", "dataset", dataset, ["manifest.json"]))
            scratch = root / "scratch"
            assets.stage(specs, "model-01", scratch)
            assets.stage(specs, "data-01", scratch)
            ex = root / "experiment.json"
            ex.write_text(json.dumps({
                "schema_version":1, "kind":"inference_engine_benchmark",
                "study_id":"research-001", "experiment_id":"engine-001",
                "research_question":"Does candidate improve p95?",
                "hypothesis":"Unverified", "training":False, "synthetic_data":False,
                "cells":[{"id":"baseline", "engine":{"mode":"stock"},
                          "model_asset_id":"model-01", "dataset_asset_id":"data-01",
                          "parameters":{"concurrency":1}, "metrics":["ttft_p95_ms"]}]
            }))
            self.assertEqual(experiments.plan_experiment(ex, specs, scratch)["status"],
                             "OFFLINE_PLAN_NO_ALLOCATION")
            with patch.object(experiments, "engine_verify",
                              return_value={"status":"OFFLINE_MOCK_ENGINE_FOR_TEST_ONLY"}):
                receipt = experiments.freeze(ex, specs, scratch, root / "output", "run-001")
            run = json.loads(Path(receipt["path"]).read_text())
            self.assertEqual(run["status"], "FROZEN_NOT_EXECUTED")
            self.assertIsNone(run["results"])
            self.assertFalse(run["official_score"])
            with patch.object(experiments, "engine_verify", return_value={}):
                with self.assertRaises(FileExistsError):
                    experiments.freeze(ex, specs, scratch, root / "output", "run-001")

if __name__ == "__main__":
    unittest.main()
