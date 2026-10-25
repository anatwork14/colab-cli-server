"""Offline tests for Drive pending-revision promotion (contract C1 / ADDENDUM A1-A4).

F1: revision A uploaded but promotion interrupted (PENDING); revision B published
fully and becomes latest; retrying A must NOT promote A over B.

T1-T9 (contract): stale pending, with/without manifest, manifest and backup lost,
journal lost (Drive-only state), lost PATCH response, two OS processes, no later
revision, genuine successor, corrupted remote metadata.
"""
from __future__ import annotations

import hashlib
import multiprocessing
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from colab_adapter.errors import ArtifactValidationError, DriveIntegrityError
from colab_adapter.storage import (
    ArtifactType,
    DriveCredentials,
    GoogleDriveArtifactStore,
)
from colab_adapter.storage.drive_metadata import (
    BASE_NONE,
    DISPOSITION_ALREADY_LATEST,
    DISPOSITION_FAIL_CLOSED,
    DISPOSITION_PROMOTE,
    DISPOSITION_STALE,
    RemoteRevisionRef,
    build_revision_properties,
    pending_disposition,
)
from tests.drive_api_mock import FileBackedDriveMock, StrictDriveMock

H_A = hashlib.sha256(b"REV_A").hexdigest()
H_B = hashlib.sha256(b"REV_B").hexdigest()
H_C = hashlib.sha256(b"REV_C").hexdigest()
H_Z = hashlib.sha256(b"OTHER").hexdigest()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_credentials(path: Path) -> DriveCredentials:
    creds = DriveCredentials(
        client_id="cid",
        client_secret="csecret",
        refresh_token="rtoken",
        access_token="atoken",
        token_expiry=9999999999.0,
    )
    creds.save(path)
    return creds


def _make_store(creds: DriveCredentials, cred_path: Path, staging: Path, client) -> GoogleDriveArtifactStore:
    return GoogleDriveArtifactStore(
        credentials=creds,
        credentials_file=cred_path,
        local_staging_dir=staging,
        http_client=client,
        chunk_size=1024 * 1024,
        initial_backoff_sec=0.01,
    )


class DrivePendingBase(unittest.TestCase):
    PROJECT = "proj_pp"
    RUN = "run_pp_01"

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="pending_promo_")
        self.addCleanup(lambda: shutil.rmtree(self.test_dir, ignore_errors=True))
        self.staging = Path(self.test_dir) / "staging"
        self.cred_path = Path(self.test_dir) / "credentials.json"
        self.creds = _make_credentials(self.cred_path)
        self.mock = StrictDriveMock()
        self.store = _make_store(self.creds, self.cred_path, self.staging, self.mock)
        self.model = Path(self.test_dir) / "model.pt"

    # -- helpers -----------------------------------------------------------------
    def _publish(self, data: bytes, store: Optional[GoogleDriveArtifactStore] = None):
        self.model.write_bytes(data)
        return (store or self.store).publish(self.model, self.PROJECT, self.RUN, ArtifactType.CHECKPOINT.value)

    def _folder(self) -> str:
        return self.store.resolve_target_folder_id(self.PROJECT, self.RUN, "checkpoints")

    def _remote_files(self) -> Dict[str, Dict[str, Any]]:
        folder = self._folder()
        return {
            f["id"]: f for f in self.mock.mock_files
            if f.get("name") == "model.pt" and f.get("parent") == folder
        }

    def _file_for_sha(self, sha: str) -> Dict[str, Any]:
        matches = [f for f in self._remote_files().values() if f.get("appProperties", {}).get("sha256") == sha]
        self.assertEqual(len(matches), 1, f"expected exactly one remote file for sha {sha[:8]}")
        return matches[0]

    def _make_pending(self, data: bytes) -> str:
        """Upload data but make the promotion PATCH fail for good -> revision stays PENDING."""
        self.mock.fail_next_n_patches = 100
        try:
            with self.assertRaises(Exception):
                self._publish(data)
        finally:
            self.mock.fail_next_n_patches = 0
        return self._file_for_sha(hashlib.sha256(data).hexdigest())["id"]

    def _remote_props(self, file_id: str) -> Dict[str, Any]:
        for f in self.mock.mock_files:
            if f.get("id") == file_id:
                return dict(f.get("appProperties") or {})
        raise AssertionError(f"no remote file {file_id}")

    def _set_remote_props(self, file_id: str, props: Dict[str, Any]) -> None:
        for f in self.mock.mock_files:
            if f.get("id") == file_id:
                f["appProperties"] = dict(props)
                return
        raise AssertionError(f"no remote file {file_id}")

    def _latest_remote_ids(self) -> List[str]:
        return [fid for fid, f in self._remote_files().items() if f.get("appProperties", {}).get("is_latest") == "true"]

    def _retrieve_bytes(self, artifact_id: str, store: Optional[GoogleDriveArtifactStore] = None) -> bytes:
        out = Path(self.test_dir) / f"out_{hashlib.md5(artifact_id.encode()).hexdigest()}.bin"
        (store or self.store).retrieve(artifact_id, out)
        return out.read_bytes()

    def _manifest_paths(self):
        mdir = self.staging / "projects" / self.PROJECT / "runs" / self.RUN
        return mdir / "manifest.json", mdir / "manifest.json.bak"


class TestPendingPromotion(DrivePendingBase):
    DATA_A = b"REV_A_" + b"a" * 64
    DATA_B = b"REV_B_" + b"b" * 64
    DATA_C = b"REV_C_" + b"c" * 64

    def test_f1_stale_pending_retry_does_not_promote_over_newer_latest(self):
        """F1 (red on original code): A pending, B published fully, retry A must not promote A."""
        a_id = self._make_pending(self.DATA_A)
        self.assertNotEqual(self._remote_props(a_id).get("is_latest"), "true")

        art_b = self._publish(self.DATA_B)
        self.assertTrue(art_b.is_latest)

        art_a = self._publish(self.DATA_A)
        self.assertEqual(art_a.remote_file_id, a_id)
        self.assertFalse(art_a.is_latest)
        self.assertEqual(self._latest_remote_ids(), [art_b.remote_file_id])
        manifest = self.store._load_or_recover_manifest(self.PROJECT, self.RUN)
        self.assertEqual(manifest.latest_by_path["checkpoints/model.pt"], art_b.artifact_id)

    def test_t1_stale_pending_stays_pending_no_write_no_upload(self):
        a_id = self._make_pending(self.DATA_A)
        art_b = self._publish(self.DATA_B)
        sessions_before = self.mock.upload_sessions_initiated
        patches_before = len(self.mock.recorded_patches)

        art_a = self._publish(self.DATA_A)

        self.assertFalse(art_a.is_latest)
        self.assertEqual(self.mock.upload_sessions_initiated, sessions_before, "no duplicate upload")
        self.assertEqual(len(self.mock.recorded_patches), patches_before, "stale pending must not be PATCHed")
        props_a = self._remote_props(a_id)
        self.assertNotEqual(props_a.get("is_latest"), "true")
        self.assertNotIn("next_sha256", props_a, "stale pending must never be marked with a successor")
        self.assertEqual(self._latest_remote_ids(), [art_b.remote_file_id])
        self.assertEqual(self._retrieve_bytes(art_a.artifact_id), self.DATA_A)

    def test_t2_stale_pending_with_manifest_keeps_head(self):
        self._make_pending(self.DATA_A)
        art_b = self._publish(self.DATA_B)
        manifest_path, _ = self._manifest_paths()
        self.assertTrue(manifest_path.exists())

        art_a = self._publish(self.DATA_A)

        self.assertFalse(art_a.is_latest)
        manifest = self.store._load_or_recover_manifest(self.PROJECT, self.RUN)
        self.assertEqual(manifest.latest_by_path["checkpoints/model.pt"], art_b.artifact_id)
        self.assertFalse(manifest.get_artifact(art_a.artifact_id).is_latest)
        self.assertEqual(self._retrieve_bytes(art_a.artifact_id), self.DATA_A)
        self.assertEqual(self._retrieve_bytes(art_b.artifact_id), self.DATA_B)

    def test_t3_manifest_and_backup_deleted_stale_still_not_promoted(self):
        a_id = self._make_pending(self.DATA_A)
        art_b = self._publish(self.DATA_B)
        for p in self._manifest_paths():
            if p.exists():
                p.unlink()

        art_a = self._publish(self.DATA_A)

        self.assertEqual(art_a.remote_file_id, a_id)
        self.assertFalse(art_a.is_latest)
        self.assertEqual(self._latest_remote_ids(), [art_b.remote_file_id])
        self.assertEqual(self._retrieve_bytes(art_a.artifact_id), self.DATA_A)

    def test_t4_journal_kept_manifest_lost_no_upload_no_promotion(self):
        self._make_pending(self.DATA_A)
        art_b = self._publish(self.DATA_B)
        manifest_path, bak_path = self._manifest_paths()
        for p in (manifest_path, bak_path):
            if p.exists():
                p.unlink()
        sessions_before = self.mock.upload_sessions_initiated

        art_a = self._publish(self.DATA_A)

        self.assertFalse(art_a.is_latest)
        self.assertEqual(self.mock.upload_sessions_initiated, sessions_before)
        self.assertEqual(self._latest_remote_ids(), [art_b.remote_file_id])

    def test_t5a_journal_lost_fresh_adapter_stale_with_base_not_promoted(self):
        self._make_pending(self.DATA_A)
        art_b = self._publish(self.DATA_B)
        journal = self.staging / "drive_transfers.json"
        for p in (journal, *self._manifest_paths()):
            if p.exists():
                p.unlink()
        fresh = _make_store(self.creds, self.cred_path, self.staging, self.mock)

        art_a = self._publish(self.DATA_A, store=fresh)

        self.assertFalse(art_a.is_latest)
        self.assertEqual(self._latest_remote_ids(), [art_b.remote_file_id])

    def test_t5b_journal_lost_legacy_pending_without_base_is_ambiguous_fail_closed(self):
        a_id = self._make_pending(self.DATA_A)
        legacy = {k: v for k, v in self._remote_props(a_id).items() if k != "base_sha256"}
        self._set_remote_props(a_id, legacy)  # simulate a pre-C1 pending file
        art_b = self._publish(self.DATA_B)
        self.assertEqual(self._latest_remote_ids(), [art_b.remote_file_id])
        for p in (self.staging / "drive_transfers.json", *self._manifest_paths()):
            if p.exists():
                p.unlink()
        fresh = _make_store(self.creds, self.cred_path, self.staging, self.mock)

        with self.assertRaises(DriveIntegrityError):
            self._publish(self.DATA_A, store=fresh)
        self.assertEqual(self._latest_remote_ids(), [art_b.remote_file_id])

    def test_t6_lost_promotion_patch_response_retried_without_duplicate(self):
        self.mock.lost_final_response_on_patch = True
        self.mock.lost_final_response_count = 1
        art = self._publish(self.DATA_A)

        self.assertTrue(art.is_latest)
        self.assertEqual(self.mock.upload_sessions_initiated, 1)
        self.assertEqual(len(self._remote_files()), 1, "no duplicate remote file after ambiguous PATCH")
        self.assertEqual(self._latest_remote_ids(), [art.remote_file_id])

    def test_t8a_no_later_revision_pending_promotes_on_retry(self):
        a_id = self._make_pending(self.DATA_A)
        art_a = self._publish(self.DATA_A)
        self.assertEqual(art_a.remote_file_id, a_id)
        self.assertTrue(art_a.is_latest)
        self.assertEqual(self._latest_remote_ids(), [a_id])

    def test_t8b_genuine_successor_c_after_b_promotes_and_chains(self):
        art_a = self._publish(self.DATA_A)
        art_b = self._publish(self.DATA_B)
        art_c = self._publish(self.DATA_C)
        self.assertTrue(art_c.is_latest)
        self.assertEqual(self._latest_remote_ids(), [art_c.remote_file_id])
        self.assertEqual(self._remote_props(art_b.remote_file_id).get("next_sha256"), _sha(self.DATA_C))
        self.assertEqual(self._remote_props(art_a.remote_file_id).get("next_sha256"), _sha(self.DATA_B))

    def test_crash_mid_coordination_no_head_base_historical_next_matches_promotes(self):
        """Demotion of X committed, promotion of P crashed: no head, X.next == P -> promote P."""
        art_x = self._publish(self.DATA_A)
        self.mock.abort_before_promote_patch = True
        try:
            with self.assertRaises(Exception):
                self._publish(self.DATA_B)
        finally:
            self.mock.abort_before_promote_patch = False
        self.assertEqual(self._remote_props(art_x.remote_file_id).get("next_sha256"), _sha(self.DATA_B))
        self.assertEqual(self._latest_remote_ids(), [])

        art_b = self._publish(self.DATA_B)

        self.assertTrue(art_b.is_latest)
        self.assertEqual(self._latest_remote_ids(), [art_b.remote_file_id])
        self.assertNotEqual(self._remote_props(art_x.remote_file_id).get("is_latest"), "true")

    def test_head_with_conflicting_successor_fails_closed(self):
        art_h = self._publish(self.DATA_A)
        props = self._remote_props(art_h.remote_file_id)
        props["next_sha256"] = H_Z  # latest that already names a different successor
        self._set_remote_props(art_h.remote_file_id, props)

        with self.assertRaises(DriveIntegrityError):
            self._publish(self.DATA_B)
        self.assertEqual(self._latest_remote_ids(), [art_h.remote_file_id])

    def test_t9_corrupted_remote_metadata_is_typed_failure(self):
        art_h = self._publish(self.DATA_A)
        props = self._remote_props(art_h.remote_file_id)
        props["sha256"] = "not-a-sha"
        self._set_remote_props(art_h.remote_file_id, props)

        with self.assertRaises(DriveIntegrityError):
            self._publish(self.DATA_B)

    def test_pending_integrity_error_is_terminal_for_upload_worker(self):
        # ArtifactUploadWorker marks ArtifactValidationError as terminal (no retry); DriveIntegrityError
        # is a subclass, so ambiguous pending recovery is never retried forever.
        self.assertTrue(issubclass(DriveIntegrityError, ArtifactValidationError))


class TestPendingDispositionHelper(unittest.TestCase):
    """Pure decision table of pending_disposition()."""

    def _ref(self, sha: str, successor: Optional[str] = None, fid: str = "head-id") -> RemoteRevisionRef:
        return RemoteRevisionRef(file_id=fid, sha256=sha, successor_sha256=successor)

    def test_no_head_base_none_promotes_when_nothing_else(self):
        d = pending_disposition(pending_id="p", pending_sha256=H_A, pending_base_sha256=BASE_NONE, head=None)
        self.assertEqual(d.disposition, DISPOSITION_PROMOTE)

    def test_no_head_base_none_fails_closed_when_other_revisions_exist(self):
        d = pending_disposition(
            pending_id="p", pending_sha256=H_A, pending_base_sha256=BASE_NONE, head=None,
            path_has_other_non_pending=True,
        )
        self.assertEqual(d.disposition, DISPOSITION_FAIL_CLOSED)

    def test_no_head_base_historical_named_successor_promotes(self):
        base = self._ref(H_B, successor=H_A, fid="x")
        d = pending_disposition(pending_id="p", pending_sha256=H_A, pending_base_sha256=H_B, head=None, base_revision=base)
        self.assertEqual(d.disposition, DISPOSITION_PROMOTE)

    def test_no_head_base_sha_without_matching_successor_fails_closed(self):
        base = self._ref(H_B, successor=H_Z, fid="x")
        d = pending_disposition(pending_id="p", pending_sha256=H_A, pending_base_sha256=H_B, head=None, base_revision=base)
        self.assertEqual(d.disposition, DISPOSITION_FAIL_CLOSED)

    def test_head_equal_to_base_promotes(self):
        d = pending_disposition(pending_id="p", pending_sha256=H_A, pending_base_sha256=H_B, head=self._ref(H_B))
        self.assertEqual(d.disposition, DISPOSITION_PROMOTE)

    def test_head_moved_on_is_stale(self):
        d = pending_disposition(pending_id="p", pending_sha256=H_A, pending_base_sha256=H_B, head=self._ref(H_C))
        self.assertEqual(d.disposition, DISPOSITION_STALE)

    def test_base_none_with_head_is_stale(self):
        d = pending_disposition(pending_id="p", pending_sha256=H_A, pending_base_sha256=BASE_NONE, head=self._ref(H_C))
        self.assertEqual(d.disposition, DISPOSITION_STALE)

    def test_head_is_pending_itself_is_already_latest(self):
        d = pending_disposition(pending_id="p", pending_sha256=H_A, pending_base_sha256=BASE_NONE, head=self._ref(H_A, fid="p"))
        self.assertEqual(d.disposition, DISPOSITION_ALREADY_LATEST)

    def test_legacy_without_base_promotes_only_if_head_names_it(self):
        self.assertEqual(
            pending_disposition(pending_id="p", pending_sha256=H_A, pending_base_sha256=None,
                                head=self._ref(H_C, successor=H_A)).disposition,
            DISPOSITION_PROMOTE,
        )
        self.assertEqual(
            pending_disposition(pending_id="p", pending_sha256=H_A, pending_base_sha256=None,
                                head=self._ref(H_C)).disposition,
            DISPOSITION_FAIL_CLOSED,
        )

    def test_base_property_is_compact_and_validated(self):
        props = build_revision_properties(
            sha256=H_A, project_id="p", run_id="r", logical_path="checkpoints/m.pt",
            is_latest=False, base_sha256=H_B,
        )
        self.assertEqual(props["base_sha256"], H_B)
        for k, v in props.items():
            self.assertLessEqual(len(k.encode()) + len(v.encode()), 124)
        with self.assertRaises(Exception):
            build_revision_properties(
                sha256=H_A, project_id="p", run_id="r", logical_path="m.pt", is_latest=False, base_sha256="xyz",
            )


def _two_process_publish(state_path: str, cred_path: str, staging: str, src: str, attempts: int, out_path: str) -> None:
    """Child process: retry publishing one revision against the shared Drive state."""
    import json as _json
    from colab_adapter.storage import ArtifactType as _AT, DriveCredentials as _DC, GoogleDriveArtifactStore as _Store
    from tests.drive_api_mock import FileBackedDriveMock as _FBM

    creds = _DC.load(cred_path)
    client = _FBM(state_path)
    store = _Store(
        credentials=creds,
        credentials_file=cred_path,
        local_staging_dir=staging,
        http_client=client,
        chunk_size=1024 * 1024,
        initial_backoff_sec=0.01,
    )
    results = []
    for _ in range(attempts):
        try:
            art = store.publish(src, "proj_mp", "run_mp", _AT.CHECKPOINT.value)
            results.append({"sha": art.sha256, "is_latest": bool(art.is_latest), "remote": art.remote_file_id})
        except Exception as e:  # recorded for the parent; contention must not corrupt state
            results.append({"error": type(e).__name__, "msg": str(e)[:200]})
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(results, f)


class TestTwoIndependentProcesses(unittest.TestCase):
    """T7: two OS processes retry publishing different revisions of the same checkpoint name."""

    def test_two_processes_retrying_different_revisions_keep_single_head(self):
        tmp = tempfile.mkdtemp(prefix="pending_mp_")
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        state = Path(tmp) / "drive_state.pkl"
        staging = Path(tmp) / "staging"
        cred = Path(tmp) / "creds.json"
        _make_credentials(cred)
        src_a = Path(tmp) / "a" / "model.pt"
        src_b = Path(tmp) / "b" / "model.pt"
        src_a.parent.mkdir()
        src_b.parent.mkdir()
        src_a.write_bytes(b"MP_REV_A" + b"a" * 64)
        src_b.write_bytes(b"MP_REV_B" + b"b" * 64)

        ctx = multiprocessing.get_context("spawn")
        outs = [Path(tmp) / "out_a.json", Path(tmp) / "out_b.json"]
        procs = [
            ctx.Process(target=_two_process_publish, args=(str(state), str(cred), str(staging), str(src_a), 2, str(outs[0]))),
            ctx.Process(target=_two_process_publish, args=(str(state), str(cred), str(staging), str(src_b), 2, str(outs[1]))),
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=240)
            self.assertEqual(p.exitcode, 0, "child process failed")

        import json as _json
        results = [_json.loads(o.read_text()) for o in outs]
        for res in results:
            self.assertTrue(all("error" not in r for r in res), res)

        final = FileBackedDriveMock(state)
        final._load_state()
        files = [f for f in final.mock_files if f.get("name") == "model.pt"]
        shas = sorted(f["appProperties"]["sha256"] for f in files)
        expected = sorted([hashlib.sha256(src_a.read_bytes()).hexdigest(), hashlib.sha256(src_b.read_bytes()).hexdigest()])
        self.assertEqual(shas, expected, "exactly one remote file per revision (no duplicate upload)")
        heads = [f for f in files if f["appProperties"].get("is_latest") == "true"]
        self.assertEqual(len(heads), 1, "exactly one remote latest")
        for f in files:
            if f is not heads[0]:
                self.assertNotEqual(f["appProperties"].get("is_latest"), "true")


if __name__ == "__main__":
    unittest.main()
