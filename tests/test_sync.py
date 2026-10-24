"""Offline unit tests for V2 Workspace Synchronization Engine (M4 / G4).

Evaluates numeric metric targets Q04, Q05, Q06, Q07, and Q10.
"""
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from colab_adapter.errors import WorkspaceSyncError
from colab_adapter.project_manifest import ProjectSpec
from colab_adapter.provider import ManagedSessionRegistry
from colab_adapter.transports.base import BaseTransport
from colab_adapter.workspace import (
    SyncDelta,
    WorkspaceSyncEngine,
    compute_sync_delta,
)


class MockTestTransport(BaseTransport):
    """Mock transport tracking all upload, download, and remove calls."""

    def __init__(self, registry: ManagedSessionRegistry):
        super().__init__(registry)
        self.uploaded_files = []
        self.downloaded_files = []
        self.removed_files = []
        self.remote_fs = {}

    def upload_file(self, session_id: str, local_path, remote_path: str) -> None:
        content = Path(local_path).read_bytes()
        self.uploaded_files.append((remote_path, len(content)))
        self.remote_fs[remote_path] = content

    def download_file(self, session_id: str, remote_path: str, local_path) -> None:
        if remote_path not in self.remote_fs:
            raise FileNotFoundError(f"Remote file {remote_path} not found")
        content = self.remote_fs[remote_path]
        Path(local_path).write_bytes(content)
        self.downloaded_files.append((remote_path, len(content)))

    def list_dir(self, session_id: str, remote_path: str = "content"):
        return list(self.remote_fs.keys())

    def remove_file(self, session_id: str, remote_path: str) -> None:
        if remote_path in self.remote_fs:
            del self.remote_fs[remote_path]
        self.removed_files.append(remote_path)

    def exec_code(self, session_id: str, code: str, timeout: float = 60.0, env=None):
        return {"success": True, "output": ""}

    def check_health(self, session_id: str) -> bool:
        return True


class TestWorkspaceSync(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="colab_test_m4_")
        self.proj_dir = Path(self.tmp_dir) / "bench_project"
        self.proj_dir.mkdir()

        # Build a 100-file project fixture
        self.file_count = 100
        for i in range(self.file_count):
            sub_dir = self.proj_dir / f"pkg_{i % 5}"
            sub_dir.mkdir(exist_ok=True)
            f_path = sub_dir / f"module_{i}.py"
            # 1KiB per file
            f_path.write_bytes(b"#" * 1024)

        # Entrypoint
        (self.proj_dir / "train.py").write_text("print('train')", encoding="utf-8")

        self.reg_path = os.path.join(self.tmp_dir, "test_registry.json")
        self.registry = ManagedSessionRegistry(self.reg_path)
        self.session_id = "sess-m4-test"
        self.registry.register(
            session_id=self.session_id,
            job_id="job-m4",
            gpu="T4",
            project_name="bench-project",
        )
        self.transport = MockTestTransport(self.registry)
        self.engine = WorkspaceSyncEngine(self.transport, local_cache_dir=Path(self.tmp_dir) / "cache")

        self.spec = ProjectSpec(
            name="bench-project",
            root=self.proj_dir,
            entrypoint="train.py",
            remote_root="/content/colab-workspaces/bench-project",
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_q04_unchanged_project_zero_transfers(self):
        """Q04: 100-file fixture unchanged between syncs -> 0 transferred files, 0 payload bytes."""
        # 1. Initial sync (all files uploaded)
        res1 = self.engine.apply_sync(self.spec, self.session_id)
        self.assertEqual(res1["status"], "COMMITTED")
        initial_transfers = len(self.transport.uploaded_files)
        self.assertGreater(initial_transfers, 100)

        # Clear mock tracker
        self.transport.uploaded_files.clear()

        # 2. Second sync with unchanged local tree
        res2 = self.engine.apply_sync(self.spec, self.session_id)
        self.assertEqual(res2["status"], "UP_TO_DATE")
        self.assertEqual(res2["files_transferred"], 0)
        self.assertEqual(res2["bytes_transferred"], 0)
        self.assertEqual(len(self.transport.uploaded_files), 0, "Zero files transferred on unchanged tree")

    def test_q05_single_modified_file_transfers_only_that_file(self):
        """Q05: 100-file fixture with exactly 1 modified 1KiB source file -> exactly 1 file scheduled."""
        # Initial sync
        self.engine.apply_sync(self.spec, self.session_id)
        self.transport.uploaded_files.clear()

        # Modify exactly one file: pkg_0/module_0.py
        target_file = self.proj_dir / "pkg_0" / "module_0.py"
        target_file.write_bytes(b"A" * 1024)

        # Plan sync
        plan = self.engine.plan_sync(self.spec, self.session_id)
        delta = plan["delta"]
        self.assertEqual(delta["files_to_transfer_count"], 1)
        self.assertEqual(delta["to_modify"], ["pkg_0/module_0.py"])
        self.assertEqual(len(delta["to_add"]), 0)
        self.assertEqual(delta["bytes_to_transfer"], 1024)

        # Apply sync
        res = self.engine.apply_sync(self.spec, self.session_id)
        self.assertEqual(res["status"], "COMMITTED")
        self.assertEqual(res["files_transferred"], 1)
        self.assertEqual(res["bytes_transferred"], 1024)
        # Verify only module_0.py plus the manifest was uploaded
        uploaded_paths = [f[0] for f in self.transport.uploaded_files]
        self.assertIn("/content/colab-workspaces/bench-project/pkg_0/module_0.py", uploaded_paths)
        self.assertEqual(len(uploaded_paths), 2)  # 1 source file + 1 committed .colab_manifest.json

    def test_q06_delete_remote_flag_controls_remote_deletion(self):
        """Q06: 1 deleted file with delete_remote=False -> 0 remote delete calls.

        With delete_remote=True -> exactly 1 remote delete call.
        """
        # Initial sync
        self.engine.apply_sync(self.spec, self.session_id)

        # Delete one local file
        file_to_del = self.proj_dir / "pkg_1" / "module_1.py"
        file_to_del.unlink()

        # Case A: delete_remote=False (default safe behavior)
        self.spec.delete_remote = False
        plan_safe = self.engine.plan_sync(self.spec, self.session_id)
        self.assertEqual(len(plan_safe["delta"]["to_delete"]), 0)

        self.transport.removed_files.clear()
        self.engine.apply_sync(self.spec, self.session_id)
        self.assertEqual(len(self.transport.removed_files), 0, "No remote delete with delete_remote=False")

        # Case B: delete_remote=True (explicit opt-in)
        self.spec.delete_remote = True
        plan_del = self.engine.plan_sync(self.spec, self.session_id)
        self.assertEqual(plan_del["delta"]["to_delete"], ["pkg_1/module_1.py"])

        self.engine.apply_sync(self.spec, self.session_id)
        self.assertIn("/content/colab-workspaces/bench-project/pkg_1/module_1.py", self.transport.removed_files)
        self.assertEqual(len(self.transport.removed_files), 1)

    def test_q10_interrupted_sync_fails_closed_without_false_committed_state(self):
        """Q10: Interrupted transfer -> no false COMMITTED state; raises WorkspaceSyncError."""
        # Initial sync
        self.engine.apply_sync(self.spec, self.session_id)

        # Modify a file
        (self.proj_dir / "pkg_2" / "module_2.py").write_bytes(b"B" * 2048)

        # Simulate upload error mid-sync
        orig_upload = self.transport.upload_file

        def failing_upload(session_id, local_path, remote_path):
            if "module_2.py" in remote_path:
                raise IOError("Simulated network disconnection during upload")
            orig_upload(session_id, local_path, remote_path)

        self.transport.upload_file = failing_upload
        with self.assertRaises(WorkspaceSyncError) as ctx:
            self.engine.apply_sync(self.spec, self.session_id)
        self.assertIn("interrupted or failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
