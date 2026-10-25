"""Independent adversarial QA for PR #24 hardening (Worker E).

Attacks, all offline (local filesystem, StrictDriveMock / FileBackedDriveMock, loopback monitor):
  * artifact identity: manifest/backup loss, corrupt manifest, wrong sha, same filename in
    checkpoints/ and outputs/, replayed historical revision, tampered immutable revision,
    head whose file is missing, stray partial files, versioned ids with short shas
  * concurrency: thread storms, real OS processes ('spawn'), mixed Drive + local publishers
    sharing one manifest lock
  * security: unicode / control-character identifiers and filenames, symlinked run dirs,
    .revisions and projects roots, symlinked staging root, source mutated during staging,
    permissions of locks, manifests, revisions
  * monitor isolation: identical run ids across projects, ?project= traversal, ambiguous legacy
    endpoint, SSE project filter, symlinked project directories
  * network: Drive 403 (rate limit vs daily quota vs forbidden), 429, 503 budget exhaustion,
    lost PATCH response, expired resumable session, corrupted and truncated downloads

Failures in this file are real findings and are intentionally NOT skipped or marked expectedFailure.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import stat
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from colab_adapter.errors import (
    ArtifactNotFoundError,
    ArtifactValidationError,
    ColabV2Error,
    DriveAuthenticationError,
    DriveQuotaExceededError,
    StorageError,
)
from colab_adapter.monitor.collector import AccountUsageCollector
from colab_adapter.monitor.server import start_monitor_server
from colab_adapter.provider import ManagedSessionRegistry
from colab_adapter.storage import ArtifactType, DriveCredentials, GoogleDriveArtifactStore
from colab_adapter.storage.base import format_logical_artifact_id, format_revision_artifact_id
from colab_adapter.storage.local import LocalArtifactStore
from colab_adapter.storage.validation import (
    validate_artifact_type,
    validate_filename,
    validate_logical_path,
    validate_project_id,
    validate_run_id,
)
from colab_adapter.telemetry import TelemetryEmitter
from tests.drive_api_mock import FileBackedDriveMock, StrictDriveMock

PROJECT = "proj-qa"
RUN = "run_001"
TYPED = (ColabV2Error,)
REFUSAL = (StorageError, ArtifactValidationError, ArtifactNotFoundError)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(str(path)).st_mode)


def _drive_creds(path: Path) -> DriveCredentials:
    creds = DriveCredentials(
        client_id="cid", client_secret="csecret", refresh_token="rtoken",
        access_token="atoken", token_expiry=9999999999.0,
    )
    creds.save(path)
    return creds


def _drive_store(root: Path, cred_path: Path, client, **kwargs) -> GoogleDriveArtifactStore:
    creds = DriveCredentials.load(cred_path)
    params = dict(
        credentials=creds,
        credentials_file=cred_path,
        local_staging_dir=root,
        http_client=client,
        chunk_size=1024 * 1024,
        initial_backoff_sec=0.001,
    )
    params.update(kwargs)
    return GoogleDriveArtifactStore(**params)


# --- module-level workers for 'spawn' child processes (must be importable) -----------------

def _mp_local_worker(root: str, src_dir: str, count: int, tag: str, project: str, run: str) -> None:
    store = LocalArtifactStore(root)
    for i in range(count):
        src = Path(src_dir) / f"{tag}_{i}.pt"
        src.write_bytes(f"{tag}-payload-{i}".encode("utf-8") * 64)
        store.publish(src, project, run, ArtifactType.CHECKPOINT.value)


def _mp_drive_worker(root: str, cred_path: str, state_path: str, src_dir: str, count: int, tag: str,
                     project: str, run: str) -> None:
    client = FileBackedDriveMock(state_path)
    store = _drive_store(Path(root), Path(cred_path), client)
    for i in range(count):
        src = Path(src_dir) / f"{tag}_{i}.pt"
        src.write_bytes(f"{tag}-drive-{i}".encode("utf-8") * 64)
        store.publish(src, project, run, ArtifactType.CHECKPOINT.value)


class _TmpCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="qa_pr24_"))
        self.addCleanup(lambda: shutil.rmtree(str(self.tmp), ignore_errors=True))
        self.root = self.tmp / "store"
        self.src_dir = self.tmp / "src"
        self.src_dir.mkdir()
        self._n = 0

    def _src(self, data: bytes, name: str = "model.pt") -> Path:
        self._n += 1
        d = self.src_dir / f"s{self._n}"
        d.mkdir()
        p = d / name
        p.write_bytes(data)
        return p

    def _run_dir(self, project: str = PROJECT, run: str = RUN) -> Path:
        return self.root / "projects" / project / "runs" / run

    def _manifest(self, project: str = PROJECT, run: str = RUN) -> Dict[str, Any]:
        return json.loads((self._run_dir(project, run) / "manifest.json").read_text(encoding="utf-8"))

    def _rev_file(self, sha: str, name: str = "model.pt", rel_dir: str = "checkpoints") -> Path:
        return self._run_dir() / rel_dir / ".revisions" / sha / name


class LocalIdentityTests(_TmpCase):
    def setUp(self) -> None:
        super().setUp()
        self.store = LocalArtifactStore(self.root)
        self.blobs = {k: f"REVISION-{k}-".encode("utf-8") * 128 for k in "ABCDE"}
        self.shas = {k: _sha(v) for k, v in self.blobs.items()}
        self.published = {}
        for k in "ABCDE":
            self.published[k] = self.store.publish(self._src(self.blobs[k]), PROJECT, RUN, "checkpoint")

    def _id(self, k: str) -> str:
        return format_revision_artifact_id(PROJECT, RUN, "checkpoints/model.pt", self.shas[k])

    def _logical(self) -> str:
        return format_logical_artifact_id(PROJECT, RUN, "checkpoints/model.pt")

    def test_missing_manifest_and_backup_fails_closed(self):
        run_dir = self._run_dir()
        (run_dir / "manifest.json").unlink()
        (run_dir / "manifest.json.bak").unlink()
        meta = None
        try:
            meta = self.store.get_metadata(self._logical())
        except REFUSAL:
            meta = None
        self.assertIsNone(meta, "head served without any manifest (alias fallback)")
        with self.assertRaises(REFUSAL):
            self.store.publish(self._src(b"NEW-AFTER-LOSS"), PROJECT, RUN, "checkpoint")
        self.assertFalse((run_dir / "manifest.json").exists(), "publish bootstrapped an empty manifest over lost history")

    def test_corrupted_manifest_is_not_overwritten_by_publish(self):
        manifest_path = self._run_dir() / "manifest.json"
        manifest_path.write_bytes(b"{not json")
        (self._run_dir() / "manifest.json.bak").unlink()
        with self.assertRaises(Exception):
            self.store.publish(self._src(b"NEW-CORRUPT"), PROJECT, RUN, "checkpoint")
        self.assertEqual(manifest_path.read_bytes(), b"{not json")

    def test_corrupted_primary_manifest_recovers_from_backup(self):
        (self._run_dir() / "manifest.json").write_bytes(b"\x00garbage")
        meta = self.store.get_metadata(self._logical())
        self.assertIsNotNone(meta)
        self.assertEqual(meta.sha256, self.shas["E"])

    def test_manifest_deleted_only_recovers_from_backup_for_history(self):
        (self._run_dir() / "manifest.json").unlink()
        for k in "ABCDE":
            meta = self.store.get_metadata(self._id(k))
            self.assertIsNotNone(meta, k)
            self.assertEqual(meta.sha256, self.shas[k], k)

    def test_wrong_expected_sha_has_no_side_effects(self):
        before = (self._run_dir() / "manifest.json").read_bytes()
        with self.assertRaises(ArtifactValidationError):
            self.store.publish(self._src(b"WRONG-SHA"), PROJECT, RUN, "checkpoint", expected_sha256=_sha(b"other"))
        self.assertEqual((self._run_dir() / "manifest.json").read_bytes(), before)

    def test_same_filename_in_checkpoints_and_outputs_stays_distinct(self):
        self.store.publish(self._src(b"OUTPUT-BYTES", "model.pt"), PROJECT, RUN, "output")
        ck = self.store.get_metadata(self._logical())
        out = self.store.get_metadata(format_logical_artifact_id(PROJECT, RUN, "outputs/model.pt"))
        self.assertEqual(ck.sha256, self.shas["E"])
        self.assertEqual(out.sha256, _sha(b"OUTPUT-BYTES"))
        self.assertNotEqual(ck.logical_path, out.logical_path)

    def test_replayed_historical_revision_keeps_head_and_alias(self):
        replay = self.store.publish(self._src(self.blobs["A"]), PROJECT, RUN, "checkpoint")
        self.assertFalse(replay.is_latest)
        self.assertEqual(replay.sha256, self.shas["A"])
        self.assertEqual(self.store.get_metadata(self._logical()).sha256, self.shas["E"])
        alias = self._run_dir() / "checkpoints" / "model.pt"
        self.assertEqual(alias.read_bytes(), self.blobs["E"])

    def test_all_five_historical_revisions_retrieve_exactly(self):
        for k in "ABCDE":
            dest = self.tmp / f"out_{k}.pt"
            self.store.retrieve(self._id(k), dest)
            self.assertEqual(dest.read_bytes(), self.blobs[k], k)

    def test_tampered_immutable_revision_is_refused(self):
        rev = self._rev_file(self.shas["A"])
        os.chmod(str(rev), 0o600)
        rev.write_bytes(b"X" * len(self.blobs["A"]))
        dest = self.tmp / "tampered.pt"
        with self.assertRaises(TYPED):
            self.store.retrieve(self._id("A"), dest)
        self.assertFalse(dest.exists())
        with self.assertRaises(TYPED):
            self.store.publish(self._src(self.blobs["A"]), PROJECT, RUN, "checkpoint")

    def test_head_file_missing_fails_closed(self):
        os.chmod(str(self._rev_file(self.shas["E"])), 0o600)
        self._rev_file(self.shas["E"]).unlink()
        dest = self.tmp / "head.pt"
        with self.assertRaises(TYPED):
            self.store.retrieve(self._logical(), dest)
        self.assertFalse(dest.exists(), "unversioned retrieve served the mutable alias after head file loss")

    def test_alias_tampering_is_repaired_from_manifest_head(self):
        alias = self._run_dir() / "checkpoints" / "model.pt"
        os.chmod(str(alias), 0o600)
        alias.write_bytes(b"TAMPERED ALIAS")
        self.store.publish(self._src(self.blobs["E"]), PROJECT, RUN, "checkpoint")
        self.assertEqual(alias.read_bytes(), self.blobs["E"])

    def test_stray_partial_files_are_ignored(self):
        rev_dir = self._run_dir() / "checkpoints" / ".revisions" / self.shas["A"]
        (rev_dir / "model.pt.part").write_bytes(b"partial")
        (self._run_dir() / "checkpoints" / ".tmp-leftover").write_bytes(b"junk")
        arts = self.store.list_artifacts(PROJECT, RUN, include_history=True)
        self.assertEqual(len(arts), 5)
        dest = self.tmp / "exact_a.pt"
        self.store.retrieve(self._id("A"), dest)
        self.assertEqual(dest.read_bytes(), self.blobs["A"])

    def test_versioned_id_with_short_sha_is_not_resolved(self):
        short = f"{PROJECT}/{RUN}/checkpoints/model.pt@sha256_{self.shas['A'][:16]}"
        meta = None
        try:
            meta = self.store.get_metadata(short)
        except REFUSAL:
            meta = None
        self.assertIsNone(meta, "16-hex prefix resolved to a revision")
        with self.assertRaises(REFUSAL):
            self.store.retrieve(short, self.tmp / "short.pt")

    def test_read_paths_never_create_directories(self):
        self.assertFalse(self.store.exists(format_logical_artifact_id("ghost-proj", "run_x", "checkpoints/model.pt")))
        self.store.get_metadata(format_logical_artifact_id("ghost-proj", "run_x", "checkpoints/model.pt"))
        self.store.list_artifacts("ghost-proj")
        self.assertFalse((self.root / "projects" / "ghost-proj").exists())

    def test_nfc_name_accepted_and_nfd_name_rejected(self):
        nfc = "modèle.pt"
        nfd = "modèle.pt"
        meta = self.store.publish(self._src(b"NFC", nfc), PROJECT, RUN, "checkpoint")
        self.assertEqual(meta.sha256, _sha(b"NFC"))
        with self.assertRaises(ArtifactValidationError):
            self.store.publish(self._src(b"NFD", nfd), PROJECT, RUN, "checkpoint")

    def test_bidi_zero_width_and_newline_filenames_rejected(self):
        for bad in ("model‮.pt", "mod​el.pt", "model\n.pt", "model\x00.pt", "mo del.pt"):
            with self.subTest(name=ascii(bad)):
                with self.assertRaises(ArtifactValidationError):
                    validate_filename(bad)
                bad_src = self._src(b"X", "placeholder.pt").parent / bad
                try:
                    bad_src.write_bytes(b"X")
                except (OSError, ValueError):
                    continue  # the OS refuses the name; nothing to publish
                with self.assertRaises(ArtifactValidationError):
                    self.store.publish(bad_src, PROJECT, RUN, "checkpoint")

    def test_identifier_unicode_and_control_tricks_rejected(self):
        bad_ids = [
            "prоj",                  # Cyrillic o
            "ｐｒｏｊ",   # fullwidth
            "run\n", "run\x00", "run​", "run\r",
            "a..b", "..", ".hidden", "-lead", " spaced", "a/b", "a\\b", "x" * 129, "",
        ]
        for bad in bad_ids:
            with self.subTest(value=ascii(bad)):
                with self.assertRaises(ArtifactValidationError):
                    validate_project_id(bad)
                with self.assertRaises(ArtifactValidationError):
                    validate_run_id(bad)

    def test_run_newline_id_publish_creates_nothing(self):
        with self.assertRaises(ArtifactValidationError):
            self.store.publish(self._src(b"NL"), PROJECT, "run_001\n", "checkpoint")
        with self.assertRaises(ArtifactValidationError):
            self.store.publish(self._src(b"TRAV"), "..", RUN, "checkpoint")
        self.assertFalse((self.root / "projects" / PROJECT / "runs" / "run_001\n").exists())

    def test_logical_path_traversal_rejected(self):
        for bad in ("../outside.pt", "checkpoints/../../x.pt", "/abs.pt", "checkpoints//x.pt",
                    "checkpoints/.revisions/x.pt"):
            with self.subTest(path=bad):
                with self.assertRaises(ArtifactValidationError):
                    validate_logical_path(bad)
        with self.assertRaises(ArtifactValidationError):
            validate_artifact_type("checkpoint/../../evil")

    def test_source_mutated_after_hashing_is_refused(self):
        from colab_adapter.storage import local as local_mod

        real = local_mod.compute_file_sha256
        src = self._src(b"ORIGINAL-CONTENT-" * 64)

        def mutating_hash(path):
            digest = real(path)
            Path(path).write_bytes(b"MUTATED-AFTER-HASH-" * 64)
            return digest

        before = self._manifest()
        local_mod.compute_file_sha256 = mutating_hash
        try:
            with self.assertRaises(TYPED):
                self.store.publish(src, PROJECT, RUN, "checkpoint")
        finally:
            local_mod.compute_file_sha256 = real
        self.assertEqual(self._manifest(), before, "manifest changed for a mutated source")

    def test_source_symlink_refused(self):
        link = self.src_dir / "link.pt"
        os.symlink(str(self._src(b"TARGET")), str(link))
        with self.assertRaises(ArtifactValidationError):
            self.store.publish(link, PROJECT, RUN, "checkpoint")

    def test_permissions_of_locks_manifest_and_revisions(self):
        self.assertEqual(_mode(self._run_dir() / ".locks"), 0o700)
        self.assertEqual(_mode(self._run_dir() / ".locks" / "manifest.lock"), 0o600)
        self.assertEqual(_mode(self._run_dir() / "manifest.json"), 0o600)
        self.assertEqual(_mode(self._run_dir() / "manifest.json.bak"), 0o600)
        self.assertEqual(_mode(self._run_dir() / "checkpoints" / ".revisions"), 0o700)
        self.assertEqual(_mode(self._run_dir() / "checkpoints" / ".revisions" / self.shas["A"]), 0o700)
        self.assertEqual(_mode(self._rev_file(self.shas["A"])), 0o600)


class LocalSymlinkAndRootTests(_TmpCase):
    def test_symlinked_run_dir_refused(self):
        outside = self.tmp / "outside_run"
        outside.mkdir()
        (self.root / "projects" / PROJECT / "runs").mkdir(parents=True)
        os.symlink(str(outside), str(self.root / "projects" / PROJECT / "runs" / RUN))
        store = LocalArtifactStore(self.root)
        with self.assertRaises(TYPED):
            store.publish(self._src(b"ESCAPE"), PROJECT, RUN, "checkpoint")
        self.assertEqual(list(outside.iterdir()), [], "publish wrote through a symlinked run dir")

    def test_symlinked_projects_root_refused(self):
        outside = self.tmp / "outside_projects"
        outside.mkdir()
        self.root.mkdir(parents=True)
        os.symlink(str(outside), str(self.root / "projects"))
        store = LocalArtifactStore(self.root)
        with self.assertRaises(TYPED):
            store.publish(self._src(b"ESCAPE"), PROJECT, RUN, "checkpoint")
        self.assertEqual(list(outside.iterdir()), [], "publish wrote through a symlinked projects root")

    def test_symlinked_revisions_dir_refused_after_history_exists(self):
        store = LocalArtifactStore(self.root)
        store.publish(self._src(b"FIRST"), PROJECT, RUN, "checkpoint")
        ck = self._run_dir() / "checkpoints"
        outside = self.tmp / "outside_rev"
        outside.mkdir()
        shutil.rmtree(str(ck / ".revisions"))
        os.symlink(str(outside), str(ck / ".revisions"))
        with self.assertRaises(TYPED):
            store.publish(self._src(b"SECOND"), PROJECT, RUN, "checkpoint")
        self.assertEqual(list(outside.iterdir()), [], "revision written through symlinked .revisions")

    def test_symlinked_store_root_keeps_all_writes_under_resolved_root(self):
        # INTEGRATOR NOTE: the operator-chosen store root may itself be a symlink (e.g. a mounted volume);
        # attacker-reachable subdirectories (.revisions, run dirs, projects) are rejected when symlinked
        # (see the tests above). Writes must land under the resolved real root and nowhere else.
        real = self.tmp / "real_root"
        real.mkdir()
        link = self.tmp / "link_root"
        os.symlink(str(real), str(link))
        store = LocalArtifactStore(link)
        stored = store.publish(self._src(b"VIA-LINK"), PROJECT, RUN, "checkpoint")
        self.assertTrue(Path(stored.path).resolve().is_relative_to(real.resolve())
                        if hasattr(Path, "is_relative_to") else str(Path(stored.path).resolve()).startswith(str(real.resolve())))


class LocalConcurrencyTests(_TmpCase):
    def test_thread_storm_single_store_instance(self):
        store = LocalArtifactStore(self.root)
        errors: List[BaseException] = []
        blobs: Dict[str, bytes] = {}

        def worker(t: int) -> None:
            try:
                for i in range(3):
                    data = f"thread-{t}-rev-{i}".encode("utf-8") * 50
                    blobs[_sha(data)] = data
                    store.publish(self._src(data), PROJECT, RUN, "checkpoint")
            except BaseException as exc:  # noqa: BLE001 - recorded for assertion
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(120)
        self.assertEqual(errors, [])
        manifest = self._manifest()
        self.assertEqual(len(manifest["artifacts"]), 24)
        head_id = manifest["latest_by_path"]["checkpoints/model.pt"]
        self.assertIn(head_id.split("@sha256_")[1], blobs, "head is not one of the published revisions")
        for sha, data in blobs.items():
            dest = self.tmp / f"t_{sha[:8]}.pt"
            store.retrieve(format_revision_artifact_id(PROJECT, RUN, "checkpoints/model.pt", sha), dest)
            self.assertEqual(dest.read_bytes(), data)

    def test_thread_storm_separate_instances_same_root(self):
        errors: List[BaseException] = []

        def worker(t: int) -> None:
            try:
                store = LocalArtifactStore(self.root)
                store.publish(self._src(f"inst-{t}".encode("utf-8") * 40), PROJECT, RUN, "checkpoint")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(6)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(120)
        self.assertEqual(errors, [])
        self.assertEqual(len(self._manifest()["artifacts"]), 6)

    def test_multiprocess_spawn_publishers_share_manifest(self):
        ctx = multiprocessing.get_context("spawn")
        procs = [
            ctx.Process(target=_mp_local_worker, args=(str(self.root), str(self.src_dir), 4, f"p{i}", PROJECT, RUN))
            for i in range(3)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(240)
        self.assertEqual([p.exitcode for p in procs], [0, 0, 0])
        manifest = self._manifest()
        self.assertEqual(len(manifest["artifacts"]), 12)
        leftovers = [p.name for p in self._run_dir().rglob("*.tmp*")]
        self.assertEqual(leftovers, [], "temp files left behind by concurrent publishers")
        # Each child publishes distinctly named files, so every logical path is its own head.
        self.assertEqual(len(manifest["latest_by_path"]), 12)
        store = LocalArtifactStore(self.root)
        for i in range(3):
            for j in range(4):
                name = f"checkpoints/p{i}_{j}.pt"
                data = f"p{i}-payload-{j}".encode("utf-8") * 64
                meta = store.get_metadata(format_logical_artifact_id(PROJECT, RUN, name))
                self.assertIsNotNone(meta, name)
                self.assertEqual(meta.sha256, _sha(data), name)

    def test_mixed_drive_and_local_threads_share_one_manifest(self):
        cred = self.tmp / "creds.json"
        _drive_creds(cred)
        mock = StrictDriveMock()
        local = LocalArtifactStore(self.root)
        drive = _drive_store(self.root, cred, mock)
        errors: List[BaseException] = []

        def run_local() -> None:
            try:
                for i in range(4):
                    local.publish(self._src(f"local-{i}".encode("utf-8") * 30), PROJECT, RUN, "checkpoint")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def run_drive() -> None:
            try:
                for i in range(4):
                    drive.publish(self._src(f"drive-{i}".encode("utf-8") * 30), PROJECT, RUN, "checkpoint")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        ts = [threading.Thread(target=run_local), threading.Thread(target=run_drive)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(240)
        self.assertEqual(errors, [])
        self.assertEqual(len(self._manifest()["artifacts"]), 8,
                         "a Drive or local publication was lost from the shared manifest")

    def test_multiprocess_mixed_local_and_drive_publishers(self):
        cred = self.tmp / "creds.json"
        _drive_creds(cred)
        state = self.tmp / "drive_state.pkl"
        ctx = multiprocessing.get_context("spawn")
        procs = [
            ctx.Process(target=_mp_local_worker, args=(str(self.root), str(self.src_dir), 3, "lp", PROJECT, RUN)),
            ctx.Process(target=_mp_drive_worker,
                        args=(str(self.root), str(cred), str(state), str(self.src_dir), 3, "dp", PROJECT, RUN)),
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(240)
        self.assertEqual([p.exitcode for p in procs], [0, 0])
        manifest = self._manifest()
        self.assertEqual(len(manifest["artifacts"]), 6,
                         "cross-process local/Drive publications collided on the manifest")
        providers = {a.get("storage_provider") for a in manifest["artifacts"].values()}
        self.assertEqual(providers, {"local", "google_drive"})


class DriveSecurityAndFaultTests(_TmpCase):
    def setUp(self) -> None:
        super().setUp()
        self.cred = self.tmp / "creds.json"
        _drive_creds(self.cred)
        self.mock = StrictDriveMock()
        self.store = _drive_store(self.root, self.cred, self.mock)

    def _remote_by_sha(self, sha: str) -> List[Dict[str, Any]]:
        return [f for f in self.mock.mock_files if f.get("appProperties", {}).get("sha256") == sha]

    def test_invalid_run_id_rejected_before_remote_or_staging(self):
        data_src = self._src(b"DRIVE-NL")
        with self.assertRaises(ArtifactValidationError):
            self.store.publish(data_src, PROJECT, "run_001\n", ArtifactType.CHECKPOINT.value)
        with self.assertRaises(ArtifactValidationError):
            self.store.publish(data_src, "ｐroj", RUN, ArtifactType.CHECKPOINT.value)
        self.assertEqual(self.mock.mock_files, [], "remote side effect before validation")
        self.assertEqual(self.mock.recorded_requests, [], "HTTP request before validation")
        self.assertFalse((self.root / "projects").exists())

    def test_403_user_rate_limit_is_retried_and_succeeds(self):
        body = json.dumps({"error": {"code": 403, "errors": [{"reason": "userRateLimitExceeded"}]}}).encode()
        self.mock.inject_fault("init", status_code=403, count=1, body=body)
        meta = self.store.publish(self._src(b"RATE-LIMITED"), PROJECT, RUN, ArtifactType.CHECKPOINT.value)
        self.assertEqual(meta.sha256, _sha(b"RATE-LIMITED"))
        self.assertEqual(len(self._remote_by_sha(_sha(b"RATE-LIMITED"))), 1)

    def test_403_daily_limit_is_not_retried(self):
        body = json.dumps({"error": {"code": 403, "errors": [{"reason": "dailyLimitExceeded"}]}}).encode()
        self.mock.inject_fault("init", status_code=403, count=5, body=body)
        with self.assertRaises(DriveQuotaExceededError):
            self.store.publish(self._src(b"DAILY"), PROJECT, RUN, ArtifactType.CHECKPOINT.value)
        self.assertEqual(len(self.mock.fault_queues["init"]), 4, "daily quota 403 was retried")

    def test_403_forbidden_is_auth_error_and_not_retried(self):
        body = json.dumps({"error": {"code": 403, "errors": [{"reason": "forbidden"}]}}).encode()
        self.mock.inject_fault("init", status_code=403, count=5, body=body)
        with self.assertRaises(DriveAuthenticationError):
            self.store.publish(self._src(b"FORBIDDEN"), PROJECT, RUN, ArtifactType.CHECKPOINT.value)
        self.assertEqual(len(self.mock.fault_queues["init"]), 4, "plain 403 was retried")

    def test_malformed_403_body_is_not_retried(self):
        self.mock.inject_fault("init", status_code=403, count=5, body=b"<html>forbidden</html>")
        with self.assertRaises(TYPED):
            self.store.publish(self._src(b"MALFORMED-403"), PROJECT, RUN, ArtifactType.CHECKPOINT.value)
        self.assertGreaterEqual(len(self.mock.fault_queues["init"]), 3,
                                "malformed 403 retried without a rate-limit reason")

    def test_429_with_retry_recovers(self):
        self.mock.inject_fault("init", status_code=429, count=2)
        meta = self.store.publish(self._src(b"TOO-MANY"), PROJECT, RUN, ArtifactType.CHECKPOINT.value)
        self.assertEqual(meta.sha256, _sha(b"TOO-MANY"))

    def test_transient_503_on_resumable_init_recovers(self):
        self.mock.inject_fault("init", status_code=503, count=1)
        data = b"TRANSIENT-503-INIT" * 20
        meta = self.store.publish(self._src(data), PROJECT, RUN, ArtifactType.CHECKPOINT.value)
        self.assertEqual(meta.sha256, _sha(data))

    def test_503_budget_exhaustion_is_typed_and_bounded(self):
        self.store.max_retries = 2
        self.mock.inject_fault("init", status_code=503, count=50)
        with self.assertRaises(TYPED):
            self.store.publish(self._src(b"DOWN"), PROJECT, RUN, ArtifactType.CHECKPOINT.value)
        consumed = 50 - len(self.mock.fault_queues["init"])
        self.assertLessEqual(consumed, 3 + 1, "retry budget not bounded by max_retries")

    def test_lost_patch_response_does_not_duplicate_or_lose_head(self):
        self.mock.lost_final_response_on_patch = True
        self.mock.lost_final_response_count = 1
        data = b"LOST-PATCH-REVISION" * 20
        self.store.publish(self._src(data), PROJECT, RUN, ArtifactType.CHECKPOINT.value)
        sha = _sha(data)
        self.assertEqual(len(self._remote_by_sha(sha)), 1, "lost PATCH response duplicated the upload")
        head = self.store.get_metadata(format_logical_artifact_id(PROJECT, RUN, "checkpoints/model.pt"))
        self.assertIsNotNone(head)
        self.assertEqual(head.sha256, sha)

    def test_expired_resumable_session_recovers_without_head_corruption(self):
        data = b"SESSION-EXPIRES" * 40
        self.mock.inject_fault("chunk", status_code=404, count=1)
        try:
            self.store.publish(self._src(data), PROJECT, RUN, ArtifactType.CHECKPOINT.value)
        except TYPED:
            pass
        meta = self.store.publish(self._src(data), PROJECT, RUN, ArtifactType.CHECKPOINT.value)
        self.assertEqual(meta.sha256, _sha(data))
        dest = self.tmp / "session.pt"
        self.store.retrieve(meta.artifact_id, dest)
        self.assertEqual(dest.read_bytes(), data)

    def test_corrupted_download_is_detected_and_destination_untouched(self):
        data = b"DOWNLOAD-ME" * 30
        meta = self.store.publish(self._src(data), PROJECT, RUN, ArtifactType.CHECKPOINT.value)
        dest = self.tmp / "dl.pt"
        dest.write_bytes(b"PRE-EXISTING")
        self.mock.corrupt_download_bytes = b"C" * len(data)
        with self.assertRaises(TYPED):
            self.store.retrieve(meta.artifact_id, dest)
        self.assertEqual(dest.read_bytes(), b"PRE-EXISTING")

    def test_truncated_download_is_detected(self):
        data = b"TRUNCATE-ME" * 30
        meta = self.store.publish(self._src(data), PROJECT, RUN, ArtifactType.CHECKPOINT.value)
        self.mock.truncate_download_bytes_to = 11
        dest = self.tmp / "short.pt"
        with self.assertRaises(TYPED):
            self.store.retrieve(meta.artifact_id, dest)
        self.assertFalse(dest.exists())

    def test_drive_versioned_retrieve_requires_full_sha(self):
        meta = self.store.publish(self._src(b"FULL-SHA"), PROJECT, RUN, ArtifactType.CHECKPOINT.value)
        short = f"{PROJECT}/{RUN}/checkpoints/model.pt@sha256_{meta.sha256[:20]}"
        with self.assertRaises(TYPED):
            self.store.retrieve(short, self.tmp / "short_drive.pt")

    def test_403_rate_limit_on_upload_chunk_recovers(self):
        body = json.dumps({"error": {"code": 403, "errors": [{"reason": "sharingRateLimitExceeded"}]}}).encode()
        self.mock.inject_fault("chunk", status_code=403, count=1, body=body)
        data = b"CHUNK-RATE-LIMIT" * 30
        meta = self.store.publish(self._src(data), PROJECT, RUN, ArtifactType.CHECKPOINT.value)
        self.assertEqual(meta.sha256, _sha(data))
        self.assertEqual(len(self._remote_by_sha(_sha(data))), 1)

    def test_corrupted_journal_preserves_corrupt_copy_and_recovers_from_backup(self):
        # INTEGRATOR NOTE: the journal's designed contract (see test_adversarial_security_and_fault_injection) is to
        # recover from the validated .bak. The corrupt primary must be preserved for forensics, never silently lost.
        journal = self.root / "drive_transfers.json"
        journal.write_bytes(b"{truncated-journal")
        self.store.publish(self._src(b"JOURNAL-CORRUPT"), PROJECT, RUN, ArtifactType.CHECKPOINT.value)
        preserved = list(self.root.glob("drive_transfers.json.corrupt.*"))
        self.assertTrue(preserved, "corrupt journal copy was not preserved")
        self.assertTrue(any(p.read_bytes() == b"{truncated-journal" for p in preserved))

    def test_drive_publish_keeps_local_entries_in_manifest(self):
        local = LocalArtifactStore(self.root)
        local.publish(self._src(b"LOCAL-ONE"), PROJECT, RUN, "checkpoint")
        self.store.publish(self._src(b"DRIVE-TWO"), PROJECT, RUN, ArtifactType.CHECKPOINT.value)
        providers = {a.get("storage_provider") for a in self._manifest()["artifacts"].values()}
        self.assertEqual(providers, {"local", "google_drive"})


class MonitorIsolationTests(_TmpCase):
    def setUp(self) -> None:
        super().setUp()
        self.ledger = self.tmp / "telemetry.jsonl"
        self.acct = self.tmp / "account.jsonl"
        self.registry = ManagedSessionRegistry(self.tmp / "sessions.json")
        self.token = "qa-monitor-token-0123456789"
        self.store = LocalArtifactStore(self.root)
        # Same run id, same checkpoint filename, different content, two projects.
        self.a_bytes = b"VISION-CHECKPOINT" * 40
        self.b_bytes = b"FEDERATED-CHECKPOINT" * 40
        self.store.publish(self._src(self.a_bytes), "proj-vision", RUN, "checkpoint")
        self.store.publish(self._src(self.b_bytes), "proj-federated", RUN, "checkpoint")
        tel_a = TelemetryEmitter(ledger_path=self.ledger, run_id=RUN, project_id="proj-vision")
        tel_b = TelemetryEmitter(ledger_path=self.ledger, run_id=RUN, project_id="proj-federated")
        tel_a.emit(event_type="train_step", epoch=1, step=10, metrics={"loss": 0.11})
        tel_b.emit(event_type="train_step", epoch=9, step=90, metrics={"loss": 9.99})
        # Legacy record without project_id must not be attributed to a named project.
        with open(self.ledger, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event_id": "evt-legacy", "run_id": RUN, "event_type": "train_step",
                                "epoch": 5, "step": 50, "metrics": {"loss": 5.5}}) + "\n")
        self.server = start_monitor_server(
            host="127.0.0.1", port=0,
            collector=AccountUsageCollector(ledger_path=self.acct),
            registry=self.registry,
            telemetry=TelemetryEmitter(ledger_path=self.ledger),
            auth_token=self.token,
            allow_external=False,
            artifacts_dir=self.root,
        )
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        th = threading.Thread(target=self.server.serve_forever, daemon=True)
        th.start()
        self.addCleanup(self._stop)

    def _stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _get(self, path: str):
        req = urllib.request.Request(self.base + path, headers={"Authorization": f"Bearer {self.token}"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8") or "null")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            try:
                return e.code, json.loads(body)
            except ValueError:
                return e.code, body

    def test_runs_list_has_one_entry_per_project_for_same_run_id(self):
        status, runs = self._get("/api/v1/runs")
        self.assertEqual(status, 200)
        keys = sorted((r["project_id"], r["run_id"]) for r in runs if r["run_id"] == RUN)
        self.assertEqual(keys, [("proj-federated", RUN), ("proj-vision", RUN)])

    def test_unqualified_legacy_run_url_is_ambiguous_409(self):
        status, body = self._get(f"/api/v1/runs/{RUN}")
        self.assertEqual(status, 409)
        self.assertEqual(body.get("error"), "ambiguous_run")
        self.assertEqual(sorted(body.get("projects", [])), ["proj-federated", "proj-vision"])

    def test_project_qualified_detail_never_mixes_projects(self):
        status, body = self._get(f"/api/v1/runs/{RUN}?project=proj-vision")
        self.assertEqual(status, 200)
        self.assertEqual(body["project_id"], "proj-vision")
        for art in body["artifacts"]:
            self.assertTrue(str(art.get("artifact_id", "")).startswith("proj-vision/"), art.get("artifact_id"))
        self.assertNotIn("proj-federated", json.dumps(body))
        self.assertTrue(all(e.get("project_id") == "proj-vision" for e in body["events"]))
        self.assertEqual(len(body["events"]), 1)

    def test_legacy_event_without_project_not_attributed(self):
        status, body = self._get(f"/api/v1/runs/{RUN}/metrics?project=proj-vision")
        self.assertEqual(status, 200)
        losses = [p["metrics"].get("loss") for p in body]
        self.assertNotIn(5.5, losses)
        self.assertNotIn(9.99, losses)
        self.assertIn(0.11, losses)

    def test_metrics_isolated_per_project(self):
        status, body = self._get(f"/api/v1/runs/{RUN}/metrics?project=proj-federated")
        self.assertEqual(status, 200)
        self.assertEqual([p["metrics"]["loss"] for p in body], [9.99])

    def test_project_traversal_selectors_rejected(self):
        for bad in ("../proj-vision", "proj-vision/../proj-federated", "..", ".", "%2e%2e", "proj%00x"):
            with self.subTest(project=bad):
                status, _ = self._get(f"/api/v1/runs/{RUN}?project={bad}")
                self.assertEqual(status, 400)
                status, _ = self._get(f"/api/v1/runs?project={bad}")
                self.assertEqual(status, 400)

    def test_project_selector_trailing_newline_rejected(self):
        # C9: selectors must use fullmatch semantics; a '$' anchored regex accepts "x\n".
        for bad in ("proj-vision%0A", "proj-vision%00"):
            with self.subTest(project=bad):
                status, _ = self._get(f"/api/v1/runs/{RUN}?project={bad}")
                self.assertEqual(status, 400)
                status, _ = self._get(f"/api/v1/runs?project={bad}")
                self.assertEqual(status, 400)

    def test_run_id_traversal_rejected(self):
        for bad in ("..%2F..%2Fetc", "..", "run_001%2F..%2F..%2Fproj-vision"):
            with self.subTest(run=bad):
                status, _ = self._get(f"/api/v1/runs/{bad}?project=proj-vision")
                self.assertIn(status, (400, 404))

    def test_unknown_project_is_404_not_fallback(self):
        status, _ = self._get(f"/api/v1/runs/{RUN}?project=proj-missing")
        self.assertEqual(status, 404)

    def test_sse_project_filter_excludes_other_project(self):
        req = urllib.request.Request(self.base + "/api/v1/events/stream?project=proj-federated",
                                     headers={"Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            chunk = resp.read().decode("utf-8")
        self.assertIn("event: train_step", chunk)
        self.assertNotIn('"loss": 0.11', chunk)
        self.assertNotIn("evt-legacy", chunk)

    def test_summary_is_not_project_mixed(self):
        status, body = self._get("/api/v1/summary")
        self.assertEqual(status, 200)
        self.assertNotIn("VISION-CHECKPOINT", json.dumps(body))

    def test_symlinked_project_dir_does_not_duplicate_runs(self):
        os.symlink(str(self.root / "projects" / "proj-vision"), str(self.root / "projects" / "evil-alias"))
        status, runs = self._get("/api/v1/runs")
        self.assertEqual(status, 200)
        self.assertNotIn("evil-alias", {r["project_id"] for r in runs})
        status, body = self._get(f"/api/v1/runs/{RUN}")
        self.assertEqual(status, 409)
        self.assertNotIn("evil-alias", body.get("projects", []))

    def test_write_methods_rejected_on_run_endpoints(self):
        req = urllib.request.Request(self.base + f"/api/v1/runs/{RUN}?project=proj-vision",
                                     data=b"{}", method="DELETE",
                                     headers={"Authorization": f"Bearer {self.token}"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(cm.exception.code, 405)

    def test_unauthenticated_run_detail_rejected(self):
        req = urllib.request.Request(self.base + f"/api/v1/runs/{RUN}?project=proj-vision")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(cm.exception.code, 401)


if __name__ == "__main__":
    unittest.main()
