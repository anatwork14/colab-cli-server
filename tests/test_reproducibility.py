"""Unit tests for code snapshot determinism, secret filtering, and experiment manifests.

Part of Colab V2 Cross-Device Workspaces & Google Drive (Issue #21).
"""
import json
import os
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

from colab_adapter.reproducibility import (
    RunManifest,
    create_deterministic_source_snapshot,
    get_environment_info,
    get_git_provenance,
    is_excluded_path,
)
from colab_adapter.storage import compute_file_sha256


class TestReproducibility(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="colab_repro_test_")
        self.addCleanup(lambda: shutil.rmtree(self.test_dir, ignore_errors=True))
        self.project_dir = Path(self.test_dir) / "project"
        self.project_dir.mkdir(parents=True, exist_ok=True)

        # Create sample legitimate project files
        (self.project_dir / "train.py").write_text("print('training...')", encoding="utf-8")
        (self.project_dir / "model.py").write_text("class Net: pass", encoding="utf-8")
        (self.project_dir / "utils").mkdir(parents=True, exist_ok=True)
        (self.project_dir / "utils" / "helper.py").write_text("def helper(): pass", encoding="utf-8")

        # Create sensitive files that must be excluded
        (self.project_dir / ".env").write_text("SECRET_KEY=supersecret123", encoding="utf-8")
        (self.project_dir / ".env.local").write_text("API_KEY=xyz", encoding="utf-8")
        (self.project_dir / "server.key").write_text("PRIVATE KEY DATA", encoding="utf-8")
        (self.project_dir / "cert.pem").write_text("CERT DATA", encoding="utf-8")
        (self.project_dir / "id_rsa").write_text("SSH KEY", encoding="utf-8")
        (self.project_dir / "__pycache__").mkdir(parents=True, exist_ok=True)
        (self.project_dir / "__pycache__" / "train.cpython-312.pyc").write_bytes(b"PYC_BYTECODE")
        (self.project_dir / ".git").mkdir(parents=True, exist_ok=True)
        (self.project_dir / ".git" / "config").write_text("git config", encoding="utf-8")

    def test_secret_filtering_rules(self):
        """Sensitive paths must match exclusion filter."""
        self.assertTrue(is_excluded_path(".env"))
        self.assertTrue(is_excluded_path("nested/.env"))
        self.assertTrue(is_excluded_path("secrets/private.key"))
        self.assertTrue(is_excluded_path("certs/server.pem"))
        self.assertTrue(is_excluded_path("id_rsa"))
        self.assertTrue(is_excluded_path(".git/config"))
        self.assertTrue(is_excluded_path("__pycache__/foo.pyc"))
        self.assertTrue(is_excluded_path(".venv/bin/python"))

        # Legitimate code files must not be excluded
        self.assertFalse(is_excluded_path("train.py"))
        self.assertFalse(is_excluded_path("models/transformer.py"))
        self.assertFalse(is_excluded_path("requirements.txt"))

    def test_deterministic_snapshot_bit_reproducibility(self):
        """Two snapshot builds from the same directory produce identical SHA256."""
        tar1 = Path(self.test_dir) / "snap1.tar.gz"
        tar2 = Path(self.test_dir) / "snap2.tar.gz"

        create_deterministic_source_snapshot(self.project_dir, tar1)
        create_deterministic_source_snapshot(self.project_dir, tar2)

        hash1 = compute_file_sha256(tar1)
        hash2 = compute_file_sha256(tar2)
        self.assertEqual(hash1, hash2, "Snapshot hashes must be bit-for-bit identical")

    def test_snapshot_strictly_excludes_sensitive_files(self):
        """Inspect generated tar archive to ensure no secrets or git files leaked."""
        tar_path = Path(self.test_dir) / "snapshot.tar.gz"
        _, manifest = create_deterministic_source_snapshot(self.project_dir, tar_path)

        with tarfile.open(tar_path, "r:gz") as tar:
            names = tar.getnames()

        # Legitimate files present
        self.assertIn("train.py", names)
        self.assertIn("model.py", names)
        self.assertIn("utils/helper.py", names)

        # Sensitive files absent
        for name in names:
            self.assertFalse(".env" in name)
            self.assertFalse(".git" in name)
            self.assertFalse(name.endswith(".key"))
            self.assertFalse(name.endswith(".pem"))
            self.assertFalse("id_rsa" in name)
            self.assertFalse("__pycache__" in name)

    def test_environment_info_capture(self):
        """Environment info capture returns platform and python details without secrets."""
        env = get_environment_info()
        self.assertIn("python_version", env)
        self.assertIn("platform", env)
        self.assertIn("os", env)

    def test_run_manifest_roundtrip(self):
        """RunManifest saves and loads faithfully with all metadata."""
        rm = RunManifest(
            project_id="cifar",
            run_id="run_42",
            entrypoint="train.py",
            arguments=["--batch-size", "64", "--lr", "0.001"],
            git_provenance={"commit": "deadbeef123", "branch": "main", "dirty": False},
            environment={"python_version": "3.12.0"},
            source_snapshot_sha256="abc12345",
            requested_gpu="T4",
        )
        dest = Path(self.test_dir) / "run_manifest.json"
        rm.save(dest)
        self.assertTrue(dest.exists())

        loaded = RunManifest.from_dict(json.loads(dest.read_text(encoding="utf-8")))
        self.assertEqual(loaded.project_id, "cifar")
        self.assertEqual(loaded.run_id, "run_42")
        self.assertEqual(loaded.arguments, ["--batch-size", "64", "--lr", "0.001"])
        self.assertEqual(loaded.requested_gpu, "T4")


if __name__ == "__main__":
    unittest.main()
