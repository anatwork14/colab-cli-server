"""Offline unit and security tests for V2 Project Manifest parser and planner (M2 / G2)."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from colab_adapter.errors import ManifestValidationError
from colab_adapter.project_manifest import (
    ProjectSpec,
    discover_project_files,
    init_project,
    load_manifest,
    plan_project,
    validate_manifest,
)


class TestProjectManifest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="colab_test_m2_")
        self.proj_dir = Path(self.tmp_dir) / "pytorch_research"
        self.proj_dir.mkdir()

        # Multi-module project structure
        (self.proj_dir / "src").mkdir()
        (self.proj_dir / "configs").mkdir()
        (self.proj_dir / "train.py").write_text("print('train')", encoding="utf-8")
        (self.proj_dir / "requirements.txt").write_text("torch>=2.0\n", encoding="utf-8")
        (self.proj_dir / "src" / "__init__.py").write_text("", encoding="utf-8")
        (self.proj_dir / "src" / "model.py").write_text("class Net: pass\n", encoding="utf-8")
        (self.proj_dir / "src" / "utils.py").write_text("def helper(): pass\n", encoding="utf-8")
        (self.proj_dir / "configs" / "baseline.json").write_text('{"lr": 0.001}', encoding="utf-8")

        # Secret / non-transferable files that MUST be ignored
        (self.proj_dir / ".git").mkdir()
        (self.proj_dir / ".git" / "config").write_text("git config secret", encoding="utf-8")
        (self.proj_dir / ".venv").mkdir()
        (self.proj_dir / ".venv" / "pyvenv.cfg").write_text("home = /usr/bin", encoding="utf-8")
        (self.proj_dir / ".env").write_text("OAUTH_SECRET=xyz", encoding="utf-8")
        (self.proj_dir / "service_token.json").write_text('{"key": "private"}', encoding="utf-8")
        (self.proj_dir / "src" / "__pycache__").mkdir()
        (self.proj_dir / "src" / "__pycache__" / "model.cpython-312.pyc").write_bytes(b"\x00\x00")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_json_manifest_load_and_offline_plan(self):
        manifest_data = {
            "schema_version": 1,
            "project": {
                "name": "pytorch-research",
                "root": ".",
            },
            "runtime": {
                "gpu": "T4",
                "timeout_seconds": 600.0,
                "retain_session": False,
            },
            "workspace": {
                "include": ["**/*"],
                "exclude": ["configs/dev_*"],
                "delete_remote": False,
            },
            "execution": {
                "entrypoint": "train.py",
                "requirements": "requirements.txt",
                "args": ["--epochs", "5"],
            },
            "artifacts": {
                "pull": ["outputs/model.pt"],
                "local_dir": "./artifacts",
            },
        }
        manifest_path = self.proj_dir / "colab-project.json"
        manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

        # Subprocess spy: ensure ZERO subprocess calls are made during planning
        with patch("subprocess.run") as mock_run:
            spec = load_manifest(manifest_path)
            self.assertEqual(spec.name, "pytorch-research")
            self.assertEqual(spec.gpu, "T4")
            self.assertEqual(spec.entrypoint, "train.py")
            self.assertEqual(spec.requirements, "requirements.txt")
            self.assertEqual(spec.args, ["--epochs", "5"])

            plan = plan_project(spec)
            mock_run.assert_not_called()

            # Verify files discovered
            discovered_paths = [f["path"] for f in plan["files"]]
            self.assertIn("train.py", discovered_paths)
            self.assertIn("requirements.txt", discovered_paths)
            self.assertIn("src/__init__.py", discovered_paths)
            self.assertIn("src/model.py", discovered_paths)
            self.assertIn("src/utils.py", discovered_paths)
            self.assertIn("configs/baseline.json", discovered_paths)

            # Verify MANDATORY SECRETS are completely excluded!
            self.assertNotIn(".git/config", discovered_paths)
            self.assertNotIn(".venv/pyvenv.cfg", discovered_paths)
            self.assertNotIn(".env", discovered_paths)
            self.assertNotIn("service_token.json", discovered_paths)
            self.assertNotIn("src/__pycache__/model.cpython-312.pyc", discovered_paths)

            # Check plan metadata
            self.assertEqual(plan["project_name"], "pytorch-research")
            self.assertEqual(plan["files_count"], len(discovered_paths))
            self.assertGreater(plan["total_bytes"], 0)

    def test_path_traversal_in_entrypoint_rejected(self):
        spec = ProjectSpec(
            name="test-traversal",
            root=self.proj_dir,
            entrypoint="../train.py",
        )
        with self.assertRaises(ManifestValidationError) as ctx:
            validate_manifest(spec)
        self.assertIn("Path traversal detected", str(ctx.exception))

    def test_absolute_path_in_entrypoint_rejected(self):
        spec = ProjectSpec(
            name="test-abs",
            root=self.proj_dir,
            entrypoint="/etc/passwd",
        )
        with self.assertRaises(ManifestValidationError) as ctx:
            validate_manifest(spec)
        self.assertIn("must be a relative path", str(ctx.exception))

    def test_symlink_escape_rejected_in_discovery(self):
        # Create a file outside the project root
        outside_file = Path(self.tmp_dir) / "outside_secret.txt"
        outside_file.write_text("secret outside", encoding="utf-8")

        # Symlink inside pointing to outside file
        bad_link = self.proj_dir / "link_to_outside.txt"
        bad_link.symlink_to(outside_file)

        spec = ProjectSpec(
            name="test-symlink",
            root=self.proj_dir,
            entrypoint="train.py",
        )
        with self.assertRaises(ManifestValidationError) as ctx:
            discover_project_files(spec)
        self.assertIn("Symlink security escape detected", str(ctx.exception))

    def test_invalid_accelerator_rejected(self):
        spec = ProjectSpec(
            name="test-gpu",
            root=self.proj_dir,
            entrypoint="train.py",
            gpu="QUANTUM_GPU_9000",
        )
        with self.assertRaises(ManifestValidationError) as ctx:
            validate_manifest(spec)
        self.assertIn("Unsupported accelerator", str(ctx.exception))

    def test_invalid_timeout_rejected(self):
        spec = ProjectSpec(
            name="test-timeout",
            root=self.proj_dir,
            entrypoint="train.py",
            timeout_seconds=2.0,  # below minimum
        )
        with self.assertRaises(ManifestValidationError) as ctx:
            validate_manifest(spec)
        self.assertIn("timeout_seconds", str(ctx.exception))

    def test_missing_entrypoint_file_rejected(self):
        spec = ProjectSpec(
            name="test-missing",
            root=self.proj_dir,
            entrypoint="nonexistent_train.py",
        )
        with self.assertRaises(ManifestValidationError) as ctx:
            validate_manifest(spec)
        self.assertIn("Entrypoint file not found", str(ctx.exception))

    def test_init_project_template_generation(self):
        target = Path(self.tmp_dir) / "new_project"
        manifest_file = init_project(target, name="starter-ml")
        self.assertTrue(manifest_file.is_file())
        content = json.loads(manifest_file.read_text(encoding="utf-8"))
        self.assertEqual(content["schema_version"], 1)
        self.assertEqual(content["project"]["name"], "starter-ml")

        # Re-initialization should fail closed
        with self.assertRaises(ManifestValidationError):
            init_project(target, name="starter-ml")


if __name__ == "__main__":
    unittest.main()
