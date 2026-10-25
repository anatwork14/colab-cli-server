"""Opus checkpoint-3 findings: large local checkpoints listed once; remote-published head has no local path."""
import os
import tempfile
import unittest
from pathlib import Path

from colab_adapter.errors import ArtifactNotFoundError
from colab_adapter.storage.local import LocalArtifactStore
from colab_adapter.storage.manifest import PublicationManifest, compute_file_sha256
from colab_adapter.storage.base import StoredArtifact, format_revision_artifact_id
from tests.test_monitor_review_followups import MIB, TestMonitorReviewFollowups


class LargeLocalCheckpointListedOnce(TestMonitorReviewFollowups):
    def test_checkpoint_over_hash_cap_is_not_listed_twice(self):
        src = self.tmp_dir / "model.pt"
        with open(src, "wb") as fh:
            fh.write(b"\x01" * 16)
            fh.truncate(9 * MIB)
        LocalArtifactStore(self.artifacts_dir).publish(src, "alpha", "run_big", "checkpoint")
        arts = self._local_artifacts("alpha", "run_big")
        models = [a for a in arts if str(a.get("logical_path", "")).endswith("model.pt")
                  or str(a.get("artifact_id", "")).endswith("model.pt")]
        self.assertEqual(len(models), 1, models)
        self.assertNotEqual(models[0].get("revision_id"), "unversioned")


class RemotePublishedHeadHasNoLocalPath(unittest.TestCase):
    def test_local_store_reports_no_local_bytes_for_drive_head(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sha = "c" * 64
            run_dir = root / "projects" / "p" / "runs" / "r"
            run_dir.mkdir(parents=True)
            aid = format_revision_artifact_id("p", "r", "checkpoints/model.pt", sha)
            m = PublicationManifest(project_id="p", run_id="r")
            m.add_artifact(StoredArtifact(
                artifact_id=aid, storage_provider="google_drive", path="checkpoints/model.pt",
                size_bytes=3, sha256=sha, project_id="p", run_id="r", artifact_type="checkpoint",
                remote_file_id="file-1", logical_path="checkpoints/model.pt",
                revision_id="sha256_" + sha, is_latest=True))
            m.save(run_dir / "manifest.json")
            store = LocalArtifactStore(root)
            # A same-named file in the current directory must not make the head look local.
            cwd = os.getcwd()
            os.chdir(td)
            try:
                (Path(td) / "checkpoints").mkdir(exist_ok=True)
                (Path(td) / "checkpoints" / "model.pt").write_bytes(b"xyz")
                meta = store.get_metadata("p/r/checkpoints/model.pt")
                self.assertEqual(meta.sha256, sha)
                self.assertEqual(meta.path, "")
                self.assertFalse(store.exists("p/r/checkpoints/model.pt"))
                with self.assertRaises(ArtifactNotFoundError):
                    store.retrieve("p/r/checkpoints/model.pt", Path(td) / "out.bin")
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
