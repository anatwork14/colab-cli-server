"""Exact historical retrieval and manifest-authority rules for LocalArtifactStore (contract C2/B6-B8)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from colab_adapter.errors import ArtifactNotFoundError, ArtifactValidationError, StorageError
from colab_adapter.storage.local import LocalArtifactStore
from colab_adapter.storage.manifest import PublicationManifest

PROJ = "proj1"
RUN = "run1"
LP = "checkpoints/model.pt"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class LocalStoreTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="local_exact_")
        self.root = os.path.join(self.tmp, "store")
        self.work = os.path.join(self.tmp, "work")
        os.makedirs(self.work)
        self.store = LocalArtifactStore(root_dir=self.root)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # helpers -------------------------------------------------------------
    @property
    def run_dir(self) -> Path:
        return Path(self.root) / "projects" / PROJ / "runs" / RUN

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def alias(self) -> Path:
        return self.run_dir / LP

    def rev_file(self, sha: str) -> Path:
        return self.run_dir / "checkpoints" / ".revisions" / sha / "model.pt"

    def legacy_rev_file(self, sha: str) -> Path:
        return self.run_dir / "checkpoints" / ".revisions" / sha[:16] / "model.pt"

    def src(self, payload: bytes, name: str = "model.pt") -> str:
        path = os.path.join(self.work, name)
        with open(path, "wb") as fh:
            fh.write(payload)
        return path

    def publish(self, payload: bytes, store=None):
        return (store or self.store).publish(
            source=self.src(payload), project_id=PROJ, run_id=RUN, artifact_type="checkpoint"
        )

    def artifact_id(self, payload: bytes) -> str:
        return f"{PROJ}/{RUN}/{LP}@sha256_{_sha(payload)}"

    def unversioned(self) -> str:
        return f"{PROJ}/{RUN}/{LP}"

    def read_manifest_json(self):
        with open(self.manifest_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def write_manifest_json(self, data):
        for path in (self.manifest_path, Path(str(self.manifest_path) + ".bak")):
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.chmod(path, 0o600)


class ExactRevisionReplayTest(LocalStoreTestBase):
    A = b"checkpoint-A" * 1000
    B = b"checkpoint-B" * 1500

    def test_publish_a_b_then_replay_a_keeps_head_alias_and_history(self):
        a = self.publish(self.A)
        b = self.publish(self.B)
        self.assertTrue(a.is_latest)
        self.assertTrue(b.is_latest)

        manifest_before = self.manifest_path.read_bytes()
        alias_before = self.alias.read_bytes()

        replay = self.publish(self.A)
        self.assertEqual(replay.artifact_id, a.artifact_id)
        self.assertFalse(replay.is_latest, "replay of a historical revision must not report latest")

        # Replay touches neither the manifest head nor the alias.
        self.assertEqual(self.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(self.alias.read_bytes(), alias_before)

        manifest = PublicationManifest.load(self.manifest_path)
        self.assertEqual(manifest.latest_by_path[LP], b.artifact_id)
        self.assertEqual(len(manifest.artifacts), 2)

        # Immutable files preserved byte for byte.
        self.assertEqual(self.rev_file(_sha(self.A)).read_bytes(), self.A)
        self.assertEqual(self.rev_file(_sha(self.B)).read_bytes(), self.B)

        self.assertEqual(self.alias.read_bytes(), self.B)
        self._assert_lookups(self.store)

    def test_fresh_store_instance_gives_same_answers(self):
        self.publish(self.A)
        self.publish(self.B)
        self.publish(self.A)
        self._assert_lookups(LocalArtifactStore(root_dir=self.root))

    def _assert_lookups(self, store):
        unv = store.get_metadata(self.unversioned())
        self.assertIsNotNone(unv)
        self.assertEqual(unv.sha256, _sha(self.B))
        self.assertTrue(unv.is_latest)

        exact_a = store.get_metadata(self.artifact_id(self.A))
        self.assertEqual(exact_a.sha256, _sha(self.A))
        self.assertFalse(exact_a.is_latest)

        exact_b = store.get_metadata(self.artifact_id(self.B))
        self.assertEqual(exact_b.sha256, _sha(self.B))
        self.assertTrue(exact_b.is_latest)

        out_dir = os.path.join(self.tmp, "out_" + _sha(self.A)[:8])
        os.makedirs(out_dir, exist_ok=True)
        retrieved = store.retrieve(self.artifact_id(self.A), os.path.join(out_dir, "a.pt"))
        self.assertEqual(Path(retrieved).read_bytes(), self.A)
        retrieved_unv = store.retrieve(self.unversioned(), os.path.join(out_dir, "latest.pt"))
        self.assertEqual(Path(retrieved_unv).read_bytes(), self.B)

    def test_repeating_head_is_idempotent(self):
        self.publish(self.A)
        b1 = self.publish(self.B)
        manifest_before = self.manifest_path.read_bytes()
        b2 = self.publish(self.B)
        self.assertEqual(b2.artifact_id, b1.artifact_id)
        self.assertTrue(b2.is_latest)
        self.assertEqual(self.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(len(PublicationManifest.load(self.manifest_path).artifacts), 2)
        self.assertEqual(self.alias.read_bytes(), self.B)

    def test_get_artifact_rejects_ambiguous_name_match(self):
        manifest = PublicationManifest(project_id=PROJ, run_id=RUN)
        from colab_adapter.storage.base import StoredArtifact

        for lp, payload in (("checkpoints/model.pt", self.A), ("outputs/model.pt", self.B)):
            manifest.add_artifact(
                StoredArtifact(
                    artifact_id=f"{PROJ}/{RUN}/{lp}@sha256_{_sha(payload)}",
                    storage_provider="local",
                    path=f"/x/{lp}",
                    size_bytes=len(payload),
                    sha256=_sha(payload),
                    project_id=PROJ,
                    run_id=RUN,
                    logical_path=lp,
                )
            )
        self.assertIsNone(manifest.get_artifact("model.pt"))
        self.assertEqual(manifest.get_artifact("checkpoints/model.pt").sha256, _sha(self.A))
        self.assertEqual(manifest.get_artifact("outputs/model.pt").sha256, _sha(self.B))


class ExactRevisionLookupTest(LocalStoreTestBase):
    A = b"exact-A" * 500
    B = b"exact-B" * 700

    def setUp(self):
        super().setUp()
        self.publish(self.A)
        self.publish(self.B)

    def test_exact_a_served_from_revisions_when_manifest_entry_deleted(self):
        data = self.read_manifest_json()
        key = self.artifact_id(self.A)
        del data["artifacts"][key]
        self.write_manifest_json(data)

        meta = self.store.get_metadata(key)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.sha256, _sha(self.A))
        self.assertFalse(meta.is_latest)
        out = self.store.retrieve(key, os.path.join(self.tmp, "a_out.pt"))
        self.assertEqual(Path(out).read_bytes(), self.A)

    def test_exact_revision_missing_everywhere_raises_not_found(self):
        key = self.artifact_id(self.A)
        self.rev_file(_sha(self.A)).unlink()
        self.assertIsNone(self.store.get_metadata(key))
        self.assertFalse(self.store.exists(key))
        dest = os.path.join(self.tmp, "missing_out.pt")
        with self.assertRaises(ArtifactNotFoundError):
            self.store.retrieve(key, dest)
        self.assertFalse(os.path.exists(dest))

    def test_exact_lookup_never_falls_back_to_alias(self):
        # Remove A's immutable file and manifest entry; the alias holds B, which must not be served for A.
        key = self.artifact_id(self.A)
        self.rev_file(_sha(self.A)).unlink()
        data = self.read_manifest_json()
        del data["artifacts"][key]
        self.write_manifest_json(data)
        self.assertIsNone(self.store.get_metadata(key))

    def test_corrupted_immutable_revision_fails_typed_and_keeps_destination(self):
        key = self.artifact_id(self.A)
        self.rev_file(_sha(self.A)).write_bytes(b"X" * len(self.A))
        dest = os.path.join(self.tmp, "dest.pt")
        with open(dest, "wb") as fh:
            fh.write(b"PRECIOUS")
        with self.assertRaises(ArtifactValidationError):
            self.store.retrieve(key, dest)
        self.assertEqual(Path(dest).read_bytes(), b"PRECIOUS")

    def test_corrupted_revision_without_manifest_entry_is_rejected_by_full_hash(self):
        key = self.artifact_id(self.A)
        data = self.read_manifest_json()
        del data["artifacts"][key]
        self.write_manifest_json(data)
        self.rev_file(_sha(self.A)).write_bytes(b"Y" * len(self.A))
        with self.assertRaises(ArtifactValidationError):
            self.store.get_metadata(key)

    def test_malformed_sha_suffixes_are_rejected(self):
        bad_suffixes = ["a" * 63, "a" * 65, "g" * 64, _sha(self.A)[:16], "", "../" + "a" * 61]
        for suffix in bad_suffixes:
            artifact_id = f"{PROJ}/{RUN}/{LP}@sha256_{suffix}"
            with self.subTest(suffix=suffix):
                with self.assertRaises(ArtifactValidationError):
                    self.store.get_metadata(artifact_id)
                with self.assertRaises(ArtifactValidationError):
                    self.store.retrieve(artifact_id, os.path.join(self.tmp, "bad.pt"))

    def test_legacy_sha16_revision_directory_is_still_readable(self):
        key = self.artifact_id(self.A)
        legacy_dir = self.legacy_rev_file(_sha(self.A)).parent
        legacy_dir.mkdir(parents=True, exist_ok=True)
        os.replace(str(self.rev_file(_sha(self.A))), str(self.legacy_rev_file(_sha(self.A))))
        os.rmdir(str(self.rev_file(_sha(self.A)).parent))

        self.assertEqual(self.store.get_metadata(key).sha256, _sha(self.A))
        data = self.read_manifest_json()
        del data["artifacts"][key]
        self.write_manifest_json(data)
        meta = self.store.get_metadata(key)
        self.assertEqual(meta.sha256, _sha(self.A))
        out = self.store.retrieve(key, os.path.join(self.tmp, "legacy_out.pt"))
        self.assertEqual(Path(out).read_bytes(), self.A)

    def test_existing_different_content_revision_is_never_overwritten(self):
        c = b"C-content" * 300
        target = self.rev_file(_sha(c))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"squatter")
        with self.assertRaises(ArtifactValidationError):
            self.publish(c)
        self.assertEqual(target.read_bytes(), b"squatter")
        self.assertEqual(self.alias.read_bytes(), self.B)

    def test_republishing_head_with_corrupted_revision_fails_closed(self):
        self.rev_file(_sha(self.B)).write_bytes(b"Z" * len(self.B))
        with self.assertRaises(ArtifactValidationError):
            self.publish(self.B)

    def test_symlinked_revisions_directory_is_refused(self):
        revs = self.run_dir / "checkpoints" / ".revisions"
        shutil.move(str(revs), os.path.join(self.tmp, "moved_revs"))
        os.symlink(os.path.join(self.tmp, "moved_revs"), str(revs))
        with self.assertRaises(ArtifactValidationError):
            self.store.get_metadata(self.artifact_id(self.A))


class ReadPathsDoNotCreateDirectoriesTest(LocalStoreTestBase):
    def test_read_paths_do_not_mkdir_run_dirs(self):
        store = LocalArtifactStore(root_dir=self.root)
        missing_id = f"{PROJ}/never_run/{LP}"
        self.assertIsNone(store.get_metadata(missing_id))
        self.assertFalse(store.exists(f"{PROJ}/never_run/{LP}@sha256_{'a' * 64}"))
        self.assertEqual(store.list_artifacts(PROJ, "never_run"), [])
        self.assertEqual(store.list_artifacts(PROJ), [])
        self.assertFalse(os.path.exists(os.path.join(self.root, "projects")))


class ManifestAuthorityTest(LocalStoreTestBase):
    A = b"auth-A" * 100
    B = b"auth-B" * 200

    def test_manifest_loss_fails_closed(self):
        self.publish(self.A)
        self.publish(self.B)
        revs_before = sorted(str(p) for p in (self.run_dir / "checkpoints" / ".revisions").rglob("*"))
        self.manifest_path.unlink()
        Path(str(self.manifest_path) + ".bak").unlink()

        fresh = LocalArtifactStore(root_dir=self.root)
        with self.assertRaises(StorageError):
            fresh.get_metadata(f"{PROJ}/{RUN}/{LP}")
        with self.assertRaises(StorageError):
            fresh.get_metadata(self.artifact_id(self.A))
        with self.assertRaises(StorageError):
            fresh.retrieve(self.unversioned(), os.path.join(self.tmp, "loss_out.pt"))
        with self.assertRaises(StorageError):
            self.publish(b"new-after-loss", store=fresh)
        revs_after = sorted(str(p) for p in (self.run_dir / "checkpoints" / ".revisions").rglob("*"))
        self.assertEqual(revs_before, revs_after, "publish must not create revisions after manifest loss")
        self.assertFalse(self.manifest_path.exists())

    def test_legacy_layout_without_manifest_or_revisions_still_serves_plain_file(self):
        legacy_dir = self.run_dir / "checkpoints"
        legacy_dir.mkdir(parents=True)
        (legacy_dir / "model.pt").write_bytes(self.A)
        meta = LocalArtifactStore(root_dir=self.root).get_metadata(self.unversioned())
        self.assertIsNotNone(meta)
        self.assertEqual(meta.sha256, _sha(self.A))

    def test_versioned_id_for_unknown_run_returns_none_without_alias_fallback(self):
        self.publish(self.A)
        other = f"{PROJ}/other_run/{LP}@sha256_{_sha(self.A)}"
        self.assertIsNone(self.store.get_metadata(other))


if __name__ == "__main__":
    unittest.main()
