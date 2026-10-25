"""Mutable alias consistency and fail-closed head handling for LocalArtifactStore (contract C2/B6)."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from colab_adapter.errors import ArtifactNotFoundError, ArtifactValidationError, StorageError
from colab_adapter.storage.local import LocalArtifactStore
from colab_adapter.storage.manifest import PublicationManifest

from tests.test_local_exact_revision import LocalStoreTestBase, LP, PROJ, RUN, _sha


class AliasConsistencyTest(LocalStoreTestBase):
    A = b"alias-A" * 800
    B = b"alias-B" * 900

    def test_crash_between_manifest_save_and_alias_write_is_repaired_on_next_access(self):
        self.publish(self.A)
        with mock.patch.object(LocalArtifactStore, "_write_alias", side_effect=RuntimeError("injected crash")):
            with self.assertRaises(RuntimeError):
                self.publish(self.B)

        # Manifest committed B as head, but the alias still points at A (stale cache).
        self.assertEqual(PublicationManifest.load(self.manifest_path).latest_by_path[LP], self.artifact_id(self.B))
        self.assertEqual(self.alias.read_bytes(), self.A)

        fresh = LocalArtifactStore(root_dir=self.root)
        meta = fresh.get_metadata(self.unversioned())
        self.assertEqual(meta.sha256, _sha(self.B))
        self.assertTrue(meta.is_latest)
        self.assertEqual(self.alias.read_bytes(), self.B)

    def test_next_publish_of_head_repairs_stale_alias(self):
        self.publish(self.A)
        with mock.patch.object(LocalArtifactStore, "_write_alias", side_effect=RuntimeError("injected crash")):
            with self.assertRaises(RuntimeError):
                self.publish(self.B)
        again = self.publish(self.B)
        self.assertTrue(again.is_latest)
        self.assertEqual(self.alias.read_bytes(), self.B)

    def test_tampered_alias_is_repaired_by_unversioned_read(self):
        self.publish(self.A)
        self.publish(self.B)
        self.alias.write_bytes(b"tampered")
        meta = LocalArtifactStore(root_dir=self.root).get_metadata(self.unversioned())
        self.assertEqual(meta.sha256, _sha(self.B))
        self.assertEqual(self.alias.read_bytes(), self.B)

    def test_historical_replay_does_not_touch_tampered_alias(self):
        self.publish(self.A)
        self.publish(self.B)
        self.alias.write_bytes(b"left-alone")
        self.publish(self.A)
        self.assertEqual(self.alias.read_bytes(), b"left-alone")

    def test_symlinked_alias_is_replaced_and_target_not_written(self):
        outside = os.path.join(self.tmp, "outside_target.bin")
        with open(outside, "wb") as fh:
            fh.write(b"outside-original")
        self.publish(self.A)
        self.alias.unlink()
        os.symlink(outside, str(self.alias))
        self.publish(self.A)
        self.assertFalse(os.path.islink(str(self.alias)))
        self.assertEqual(self.alias.read_bytes(), self.A)
        self.assertEqual(Path(outside).read_bytes(), b"outside-original")

    def test_missing_head_revision_file_fails_closed(self):
        self.publish(self.A)
        self.publish(self.B)
        self.rev_path_b = self.run_dir / "checkpoints" / ".revisions" / _sha(self.B) / "model.pt"
        self.rev_path_b.unlink()
        fresh = LocalArtifactStore(root_dir=self.root)
        with self.assertRaises(ArtifactNotFoundError):
            fresh.get_metadata(self.unversioned())
        with self.assertRaises(ArtifactNotFoundError):
            fresh.retrieve(self.unversioned(), os.path.join(self.tmp, "head_out.pt"))

    def test_head_revision_with_wrong_hash_fails_closed_and_alias_not_replaced_by_it(self):
        self.publish(self.A)
        self.publish(self.B)
        self.alias.write_bytes(b"A-cache-like-bytes")
        rev_b = self.run_dir / "checkpoints" / ".revisions" / _sha(self.B) / "model.pt"
        rev_b.write_bytes(b"corrupt" * 10)
        fresh = LocalArtifactStore(root_dir=self.root)
        with self.assertRaises(ArtifactValidationError):
            fresh.get_metadata(self.unversioned())
        self.assertNotEqual(self.alias.read_bytes(), rev_b.read_bytes())

    def test_manifest_head_is_authoritative_over_alias_content(self):
        self.publish(self.A)
        self.publish(self.B)
        # Alias content is set to A's bytes: the manifest head (B) must still win.
        self.alias.write_bytes(self.A)
        meta = LocalArtifactStore(root_dir=self.root).get_metadata(self.unversioned())
        self.assertEqual(meta.sha256, _sha(self.B))


class ManifestLossRecoveryTest(LocalStoreTestBase):
    A = b"loss-A" * 300
    B = b"loss-B" * 400

    def test_bak_only_recovery_still_serves_history(self):
        self.publish(self.A)
        self.publish(self.B)
        self.manifest_path.unlink()
        fresh = LocalArtifactStore(root_dir=self.root)
        self.assertEqual(fresh.get_metadata(self.unversioned()).sha256, _sha(self.B))
        self.assertEqual(fresh.get_metadata(self.artifact_id(self.A)).sha256, _sha(self.A))

    def test_repeated_head_publish_after_manifest_loss_fails_closed(self):
        self.publish(self.A)
        self.manifest_path.unlink()
        Path(str(self.manifest_path) + ".bak").unlink()
        with self.assertRaises(StorageError):
            self.publish(self.A)
        self.assertFalse(self.manifest_path.exists())


if __name__ == "__main__":
    unittest.main()
