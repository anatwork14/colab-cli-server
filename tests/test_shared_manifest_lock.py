"""Drive and local stores must contend on the same per-manifest lock (Opus A5)."""
import tempfile
import threading
import unittest
from pathlib import Path

from colab_adapter.errors import JournalLockError
from colab_adapter.storage.google_drive import GoogleDriveArtifactStore
from colab_adapter.storage.locking import manifest_lock, manifest_lock_path


class SharedManifestLockTest(unittest.TestCase):
    def test_drive_store_uses_same_lock_file_as_local_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = GoogleDriveArtifactStore.__new__(GoogleDriveArtifactStore)
            store.staging_dir = root
            manifest = root / "projects" / "p" / "runs" / "r" / "manifest.json"
            held = threading.Event()
            release = threading.Event()

            def holder():
                with manifest_lock(manifest):
                    held.set()
                    release.wait(10)

            t = threading.Thread(target=holder)
            t.start()
            try:
                self.assertTrue(held.wait(5))
                with self.assertRaises(JournalLockError):
                    with store._manifest_lock("p", "r", timeout=0.2):
                        pass
            finally:
                release.set()
                t.join(10)
            with store._manifest_lock("p", "r", timeout=2):
                pass
            self.assertTrue(manifest_lock_path(manifest).exists())


if __name__ == "__main__":
    unittest.main()
