"""Input validation for storage identifiers (project_id, run_id, logical paths, filenames, types).

Every malicious input must be rejected with ArtifactValidationError BEFORE any filesystem
mutation, journal write, lock-path construction or remote (Drive) request. Each test runs
inside an isolated sandbox directory and compares a full tree snapshot before and after.
"""
import hashlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from colab_adapter.errors import ArtifactValidationError
from colab_adapter.storage import (
    ArtifactType,
    DriveCredentials,
    GoogleDriveArtifactStore,
    LocalArtifactStore,
)
from colab_adapter.storage.base import ArtifactStore
from colab_adapter.storage.manifest import compute_file_sha256
from colab_adapter.storage.transfer import TransferJournal
from colab_adapter.storage.validation import (
    validate_artifact_type,
    validate_filename,
    validate_logical_path,
    validate_project_id,
    validate_run_id,
)
from colab_adapter.storage.worker import ArtifactUploadWorker

MALICIOUS_IDS = [
    "../../outside",
    "../project",
    "project/../../outside",
    "project\\..\\outside",
    "/absolute/path",
    ".",
    "..",
    "",
    "a\x00b",
    "\x00",
    "proj\nx",
    "trailing-newline\n",
    "tab\tx",
    "esc\x1b[0m",
    "del\x7f",
    "pröject",
    "proj‮x",
    "日本語",
    "a..b",
    "-leading-dash",
    "_leading-underscore",
    "has space",
    "x" * 129,
]

NON_STR_IDS = [None, 0, 1.5, b"p1", ["p1"], Path("p1")]


def _tree_snapshot(root):
    """Return a sorted, comparable description of everything under root (no symlink following)."""
    entries = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames:
            full = os.path.join(dirpath, name)
            entries.append(("dir", os.path.relpath(full, root), os.path.islink(full)))
        for name in filenames:
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                entries.append(("link", os.path.relpath(full, root), os.readlink(full)))
                continue
            with open(full, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
            entries.append(("file", os.path.relpath(full, root), digest))
    return sorted(entries)


class _CountingStore:
    """Minimal store stub that counts publish() calls and never touches the filesystem."""

    provider_name = "counting"

    def __init__(self):
        self.publish_calls = 0

    def publish(self, *args, **kwargs):
        self.publish_calls += 1
        raise AssertionError("publish must not be reached for invalid input")


class _CountingDriveClient:
    """DriveHttpClient stub that records every request it is asked to send."""

    def __init__(self):
        self.requests = []

    def request(self, method, url, *args, **kwargs):
        self.requests.append((method, url))
        return 500, {}, b'{"error": "unexpected request"}'


class _SandboxTestCase(unittest.TestCase):
    def setUp(self):
        self.sandbox = tempfile.mkdtemp(prefix="storage_validation_")
        self.addCleanup(shutil.rmtree, self.sandbox, ignore_errors=True)
        # Everything the code under test may write lives under iso/; sandbox/ is the "outside".
        self.iso = os.path.join(self.sandbox, "iso")
        self.store_root = os.path.join(self.iso, "store")
        self.journal_path = os.path.join(self.iso, "journal", "transfers.json")
        self.elsewhere = os.path.join(self.sandbox, "elsewhere")
        os.makedirs(self.elsewhere)
        os.makedirs(self.iso)
        self.src = Path(self.iso, "src", "model.pt")
        self.src.parent.mkdir(parents=True)
        self.src.write_bytes(b"checkpoint-bytes" * 64)

    def assertNothingCreated(self, before):
        after = _tree_snapshot(self.sandbox)
        self.assertEqual(before, after, "filesystem changed after rejected input")
        self.assertEqual(os.listdir(self.elsewhere), [])


class ValidationRuleTest(unittest.TestCase):
    """Direct tests of the pure validators in colab_adapter.storage.validation."""

    def test_accepts_compatible_ids(self):
        for ok in ["p1", "proj-adv", "run_102", "run-historic-99", "a.b-c_d", "x", "x" * 128,
                   "unmanaged-colab-session-123", "9lives"]:
            self.assertEqual(validate_project_id(ok), ok)
            self.assertEqual(validate_run_id(ok), ok)

    def test_rejects_malicious_ids(self):
        for bad in MALICIOUS_IDS + NON_STR_IDS:
            with self.subTest(value=bad):
                with self.assertRaises(ArtifactValidationError):
                    validate_project_id(bad)
                with self.assertRaises(ArtifactValidationError):
                    validate_run_id(bad)

    def test_filename_accepts_nfc_unicode_but_rejects_traversal_and_format_chars(self):
        self.assertEqual(validate_filename("модель.pt"), "модель.pt")
        self.assertEqual(validate_filename("a..b.pt"), "a..b.pt")
        for bad in [".", "..", ".hidden", "a/b", "a\\b", "é.pt",  # NFD, not NFC
                    "rlo‮gnp.pt", "ls x", "zero​width.pt", "nul\x00.pt"]:
            with self.subTest(value=bad):
                with self.assertRaises(ArtifactValidationError):
                    validate_filename(bad)

    def test_error_message_escapes_control_characters(self):
        with self.assertRaises(ArtifactValidationError) as ctx:
            validate_project_id("evil\x00\x1b[2J\nname")
        self.assertNotIn("\x00", str(ctx.exception))
        self.assertNotIn("\x1b", str(ctx.exception))
        self.assertNotIn("\n", str(ctx.exception))

    def test_logical_path_rules(self):
        self.assertEqual(validate_logical_path("checkpoints/model.pt"), "checkpoints/model.pt")
        for bad in ["/abs/model.pt", "a/../b", "a\\b", "a//b", "", "a/./b", "a/..", "a/\x00b", "a/" + "x" * 600]:
            with self.subTest(value=bad):
                with self.assertRaises(ArtifactValidationError):
                    validate_logical_path(bad)

    def test_filename_and_artifact_type_rules(self):
        self.assertEqual(validate_filename("model.pt"), "model.pt")
        for bad in ["a/b.pt", "..", ".", "", "model\x00.pt", "x" * 256]:
            with self.subTest(value=bad):
                with self.assertRaises(ArtifactValidationError):
                    validate_filename(bad)
        for kind in ArtifactType:
            self.assertEqual(validate_artifact_type(kind.value), kind.value)
        for bad in ["../evil", "unknown", "", None, 3]:
            with self.subTest(value=bad):
                with self.assertRaises(ArtifactValidationError):
                    validate_artifact_type(bad)


class LocalArtifactStoreInputValidationTest(_SandboxTestCase):
    def setUp(self):
        super().setUp()
        self.store = LocalArtifactStore(root_dir=self.store_root)

    def test_publish_rejects_malicious_project_and_run_ids_without_side_effects(self):
        before = _tree_snapshot(self.sandbox)
        for bad in MALICIOUS_IDS + NON_STR_IDS:
            with self.subTest(value=bad):
                with self.assertRaises(ArtifactValidationError):
                    self.store.publish(self.src, project_id=bad, run_id="r1")
                with self.assertRaises(ArtifactValidationError):
                    self.store.publish(self.src, project_id="p1", run_id=bad)
        self.assertNothingCreated(before)

    def test_get_metadata_rejects_traversal_in_versioned_ids(self):
        before = _tree_snapshot(self.sandbox)
        sha = "a" * 64
        for artifact_id in [
            f"../../outside/r1/checkpoints/model.pt@sha256_{sha}",
            f"p1/../../outside/checkpoints/model.pt@sha256_{sha}",
            f"p1/..\\x/checkpoints/model.pt@sha256_{sha}",
            f"p1/r\x00x/checkpoints/model.pt@sha256_{sha}",
        ]:
            with self.subTest(artifact_id=artifact_id):
                with self.assertRaises(ArtifactValidationError):
                    self.store.get_metadata(artifact_id)
        self.assertNothingCreated(before)

    def test_valid_publish_still_works(self):
        stored = self.store.publish(self.src, project_id="p1", run_id="run_1")
        self.assertEqual(stored.project_id, "p1")
        self.assertTrue(os.path.isfile(stored.path))


class UploadWorkerEnqueueInputValidationTest(_SandboxTestCase):
    def setUp(self):
        super().setUp()
        self.store = _CountingStore()
        self.journal = TransferJournal(self.journal_path)
        self.worker = ArtifactUploadWorker(store=self.store, journal=self.journal, max_workers=1)

    def _journal_bytes(self):
        if not os.path.exists(self.journal_path):
            return None
        with open(self.journal_path, "rb") as fh:
            return fh.read()

    def assertRejectedWithoutSideEffects(self, **kwargs):
        before = _tree_snapshot(self.sandbox)
        journal_before = self._journal_bytes()
        with self.assertRaises(ArtifactValidationError):
            self.worker.enqueue(local_path=self.src, **kwargs)
        self.assertNothingCreated(before)
        self.assertEqual(self._journal_bytes(), journal_before, "journal changed after rejected input")
        self.assertEqual(self.journal.get_pending_transfers(), [])
        self.assertEqual(self.store.publish_calls, 0)

    def test_rejects_malicious_project_ids(self):
        for bad in MALICIOUS_IDS + NON_STR_IDS:
            with self.subTest(project_id=bad):
                self.assertRejectedWithoutSideEffects(project_id=bad, run_id="r1")

    def test_rejects_malicious_run_ids(self):
        for bad in MALICIOUS_IDS + NON_STR_IDS:
            with self.subTest(run_id=bad):
                self.assertRejectedWithoutSideEffects(project_id="p1", run_id=bad)

    def test_rejects_malicious_category_and_artifact_type(self):
        self.assertRejectedWithoutSideEffects(project_id="p1", run_id="r1", category="../../outside")
        self.assertRejectedWithoutSideEffects(project_id="p1", run_id="r1", category="a\x00b")
        self.assertRejectedWithoutSideEffects(project_id="p1", run_id="r1", artifact_type="../evil")

    def test_rejects_malicious_ids_passed_positionally(self):
        before = _tree_snapshot(self.sandbox)
        with self.assertRaises(ArtifactValidationError):
            self.worker.enqueue(self.src, "../../outside", "r1")
        self.assertNothingCreated(before)
        self.assertEqual(self.store.publish_calls, 0)

    def test_rejects_symlinked_staging_root_without_writing_through_it(self):
        os.makedirs(os.path.dirname(self.journal_path), exist_ok=True)
        staging_root = os.path.join(os.path.dirname(self.journal_path), ".staged_uploads")
        os.symlink(self.elsewhere, staging_root)
        before = _tree_snapshot(self.sandbox)
        journal_before = self._journal_bytes()
        with self.assertRaises(ArtifactValidationError):
            self.worker.enqueue(local_path=self.src, project_id="p1", run_id="r1")
        self.assertNothingCreated(before)
        self.assertEqual(self._journal_bytes(), journal_before)
        self.assertEqual(self.store.publish_calls, 0)

    def test_rejects_symlinked_staging_directory_pointing_outside(self):
        # Ask the worker where the staging directory for this revision lives (no side effects), then
        # replace it with a symlink pointing outside the staging root.
        sha = compute_file_sha256(self.src)
        staging_dir = str(self.worker._resolve_staging_dir("p1", "r1", sha))
        os.makedirs(os.path.dirname(staging_dir))
        os.symlink(self.elsewhere, staging_dir)
        before = _tree_snapshot(self.sandbox)
        with self.assertRaises(ArtifactValidationError):
            self.worker.enqueue(local_path=self.src, project_id="p1", run_id="r1")
        self.assertNothingCreated(before)
        self.assertEqual(self.store.publish_calls, 0)

    def test_valid_enqueue_stages_inside_journal_directory(self):
        record = self.worker.enqueue(local_path=self.src, project_id="p1", run_id="r1")
        self.assertEqual(record.project_id, "p1")
        staged = Path(record.local_path).resolve()
        staging_root = (Path(self.journal_path).parent / ".staged_uploads").resolve()
        staged.relative_to(staging_root)  # raises if not contained
        self.assertTrue(staged.is_file())


class DriveStorePublishInputValidationTest(_SandboxTestCase):
    """Drive publish() must reject bad ids before any Drive API request.

    This depends on the integrator hook in GoogleDriveArtifactStore.publish() that calls
    validate_project_id / validate_run_id first. Until that hook lands these tests are
    expected to FAIL (the store currently sends HTTP requests for these inputs).
    """

    def setUp(self):
        super().setUp()
        self.client = _CountingDriveClient()
        self.creds = DriveCredentials(
            client_id="test-client-id",
            client_secret="test-client-secret",
            refresh_token="test-refresh-token",
            access_token="initial-access-token",
            token_expiry=9999999999.0,
        )
        self.drive = GoogleDriveArtifactStore(
            credentials=self.creds,
            local_staging_dir=os.path.join(self.iso, "drive-staging"),
            journal_path=os.path.join(self.iso, "drive-journal", "transfers.json"),
            http_client=self.client,
            initial_backoff_sec=0.0,
        )

    def test_publish_rejects_malicious_ids_before_any_drive_request(self):
        before = _tree_snapshot(self.sandbox)
        for bad in MALICIOUS_IDS + NON_STR_IDS:
            with self.subTest(value=bad):
                with self.assertRaises(ArtifactValidationError):
                    self.drive.publish(self.src, project_id=bad, run_id="r1",
                                       artifact_type=ArtifactType.CHECKPOINT.value)
                with self.assertRaises(ArtifactValidationError):
                    self.drive.publish(self.src, project_id="p1", run_id=bad,
                                       artifact_type=ArtifactType.CHECKPOINT.value)
        self.assertEqual(self.client.requests, [])
        self.assertNothingCreated(before)


if __name__ == "__main__":
    unittest.main()
