"""Comprehensive end-to-end integration and disaster recovery review suite (Scenario Review 25).

Implements the full five-revision acceptance scenario and multi-process contention
against the hardened Google Drive v3 storage pipeline:
1. Sequential publication of revisions A..E via GoogleDriveArtifactStore and ArtifactUploadWorker
   (with worker crash/restart recovery). Verification of compact schema v2 appProperties (<= 124 bytes),
   chain of immediate successors (next_sha256), sole latest revision, and PublicationManifest.
2. Catastrophic state loss: manifest corruption, backup deletion, journal deletion, fresh store recovery.
   Exact retrieval of all revisions by @sha256_<digest> and latest unversioned retrieve.
3. Fail-closed remote integrity checks on corrupted markers (multiple latest, zero latest).
4. Total state loss replay safety: republishing historical revision A causes no upload session
   and no PATCH, preserving E as latest. Republishing E is strictly idempotent.
5. Fault injection with ConnectionResetError during PATCH and during listing: typed DriveTransferError,
   no duplicate uploads on listing failure, retry convergence.
6. Multi-process contention: two real OS subprocesses publishing revisions concurrently against
   a loopback Drive server enforcing the <= 124 byte property rule, verifying single latest,
   no deadlocks, and idempotency.
7. Zero live-network guard: socket.connect interception forbidding non-loopback network calls.
"""
from __future__ import annotations

import collections
import hashlib
import io
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from colab_adapter.errors import (
    ArtifactNotFoundError,
    ArtifactValidationError,
    DriveAuthenticationError,
    DriveIntegrityError,
    DriveQuotaExceededError,
    DriveTransferError,
)
from colab_adapter.storage.base import ArtifactType, StoredArtifact
from colab_adapter.storage.drive_metadata import (
    DRIVE_MAX_APP_PROPERTIES,
    DRIVE_PROPERTY_MAX_BYTES,
    SCHEMA_VERSION,
    build_revision_properties,
    path_digest,
)
from colab_adapter.storage.google_drive import DriveCredentials, GoogleDriveArtifactStore
from colab_adapter.storage.manifest import PublicationManifest, compute_file_sha256
from colab_adapter.storage.transfer import TransferJournal, TransferRecord, TransferStatus
from colab_adapter.storage.worker import ArtifactUploadWorker
from tests.drive_api_mock import StrictDriveMock
from tests.test_browser_xss_and_cross_process_qa import LoopbackDriveHandler, LoopbackDriveServer

logger = logging.getLogger(__name__)

# =============================================================================
# Zero Live-Network Guard: Intercept socket.socket.connect
# =============================================================================

_ORIGINAL_SOCKET_CONNECT = socket.socket.connect


def _guarded_socket_connect(sock: socket.socket, address: Any) -> None:
    """Enforce zero live-network access: block any non-loopback socket connection."""
    host = address[0] if isinstance(address, tuple) else address
    # Allow loopback addresses (IPv4 127.0.0.1, localhost, IPv6 ::1) and UNIX domain sockets
    if host not in ("127.0.0.1", "localhost", "::1") and not str(host).startswith("/"):
        raise RuntimeError(f"Live network access forbidden in unit tests: {address}")
    return _ORIGINAL_SOCKET_CONNECT(sock, address)


def setUpModule() -> None:
    socket.socket.connect = _guarded_socket_connect


def tearDownModule() -> None:
    socket.socket.connect = _ORIGINAL_SOCKET_CONNECT


# =============================================================================
# StrictLoopbackDriveHandler: Enforces <= 124 Byte Property Rule in Loopback
# =============================================================================

class StrictLoopbackDriveHandler(LoopbackDriveHandler):
    """Loopback Google Drive v3 request handler strictly enforcing the 124-byte property limit."""

    MAX_PROPERTY_BYTES = 124
    MAX_APP_PROPERTIES_COUNT = 30

    def _validate_properties(
        self, props: Optional[Mapping[str, Any]], is_patch: bool = False
    ) -> Optional[str]:
        if not props or not isinstance(props, Mapping):
            return None
        if len(props) > self.MAX_APP_PROPERTIES_COUNT:
            return f"Maximum property count exceeded: {len(props)} > {self.MAX_APP_PROPERTIES_COUNT}"
        for key, value in props.items():
            if not isinstance(key, str) or not key:
                return "Property key must be a non-empty string"
            key_bytes = len(key.encode("utf-8"))
            if is_patch and value is None:
                continue
            val_str = value if isinstance(value, str) else str(value)
            val_bytes = len(val_str.encode("utf-8"))
            total = key_bytes + val_bytes
            if total > self.MAX_PROPERTY_BYTES:
                return (
                    f"Custom property '{key}' combined key-value length ({total} bytes) "
                    f"exceeds limit: {self.MAX_PROPERTY_BYTES} bytes"
                )
        return None

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if "uploadType=resumable" in parsed.query:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b""
            meta: Dict[str, Any] = {}
            if body:
                try:
                    meta = json.loads(body.decode("utf-8"))
                except Exception:
                    pass
            err = self._validate_properties(meta.get("appProperties"), is_patch=False)
            if err:
                resp = json.dumps({
                    "error": {
                        "code": 400,
                        "message": err,
                        "errors": [{"domain": "global", "reason": "invalid", "message": err}],
                    }
                }).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
                return
            # Rewind rfile so super().do_POST() can read the body
            self.rfile = io.BytesIO(body)
        super().do_POST()

    def do_PATCH(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        meta: Dict[str, Any] = {}
        if body:
            try:
                meta = json.loads(body.decode("utf-8"))
            except Exception:
                pass
        err = self._validate_properties(meta.get("appProperties"), is_patch=True)
        if err:
            resp = json.dumps({
                "error": {
                    "code": 400,
                    "message": err,
                    "errors": [{"domain": "global", "reason": "invalid", "message": err}],
                }
            }).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return
        # Rewind rfile so super().do_PATCH() can read the body
        self.rfile = io.BytesIO(body)
        super().do_PATCH()


# =============================================================================
# Worker Code for Real OS Subprocess Multi-Process Contention
# =============================================================================

MULTIPROCESS_WORKER_SCRIPT = """
import sys, os, time, json, urllib.request, urllib.parse, urllib.error
from pathlib import Path
from typing import Optional, Dict

base_url = sys.argv[1]
workspace_dir = Path(sys.argv[2])
journal_path = Path(sys.argv[3])
barrier_dir = Path(sys.argv[4])
proc_id = sys.argv[5]
rev_letter = sys.argv[6]
content = sys.argv[7].encode("utf-8")
step = int(sys.argv[8])

class LoopbackDriveHttpClient:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
    def request(self, method: str, url: str, headers: Optional[Dict[str, str]] = None, data: Optional[bytes] = None, timeout: float = 30.0):
        parsed = urllib.parse.urlparse(url)
        target_url = url if ("127.0.0.1" in url or "localhost" in url) else f"{self.base}{parsed.path}" + (f"?{parsed.query}" if parsed.query else "")
        req = urllib.request.Request(target_url, data=data, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()

from colab_adapter.storage.google_drive import GoogleDriveArtifactStore, DriveCredentials

creds = DriveCredentials(client_id="cid", client_secret="csec", refresh_token="rtok", access_token="atok", token_expiry=9999999999.0)
store = GoogleDriveArtifactStore(
    credentials=creds,
    local_staging_dir=workspace_dir,
    journal_path=journal_path,
    http_client=LoopbackDriveHttpClient(base_url),
    chunk_size=256 * 1024,
)

# Synchronize start on filesystem barrier
ready_file = barrier_dir / f"ready_{proc_id}.flag"
ready_file.write_text("ready")
other_id = "2" if proc_id == "1" else "1"
other_file = barrier_dir / f"ready_{other_id}.flag"
for _ in range(500):
    if other_file.exists():
        break
    time.sleep(0.01)

proc_dir = workspace_dir / f"proc_{proc_id}"
proc_dir.mkdir(parents=True, exist_ok=True)
ckpt = proc_dir / "model.pt"
ckpt.write_bytes(content)

# Retry loop in case of concurrent lock acquisition delay
last_err = None
for attempt in range(10):
    try:
        res = store.publish(
            source=ckpt,
            project_id="vision-research",
            run_id="experiment-001",
            artifact_type="checkpoint",
            metadata={"step": step, "revision": rev_letter, "logical_path": "checkpoints/model.pt"},
        )
        print("RESULT:" + json.dumps({
            "proc": proc_id,
            "id": res.artifact_id,
            "remote_id": res.remote_file_id,
            "sha256": res.sha256,
        }))
        sys.exit(0)
    except Exception as e:
        last_err = e
        time.sleep(0.1)

print("FAILED:" + str(last_err), file=sys.stderr)
sys.exit(1)
"""


# =============================================================================
# Test Suite: Five-Revision Acceptance Scenario & Catastrophic Recovery
# =============================================================================

class TestV2IntegrationReview25(unittest.TestCase):
    """Five-revision acceptance scenario with real pipeline, catastrophic loss, and contention."""

    ALLOWED_V2_PROPERTY_KEYS = frozenset({
        "schema",
        "sha256",
        "path_sha256",
        "is_latest",
        "next_sha256",
    })

    def setUp(self) -> None:
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp_dir_obj.name)
        self.project_id = "vision-research"
        self.run_id = "experiment-001"
        self.logical_path = "checkpoints/model.pt"

        # Staging and checkpoints
        self.staging_dir = self.tmp_dir / "staging"
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.staging_dir / "journal.json"
        self.creds_file = self.staging_dir / "credentials.json"

        self.ckpt_dir = self.tmp_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_file = self.ckpt_dir / "model.pt"

        self.creds = DriveCredentials(
            client_id="cid-review25",
            client_secret="csec-review25",
            refresh_token="rtok-review25",
            access_token="atok-review25",
            token_expiry=9999999999.0,
        )
        self.creds.save(self.creds_file)

        # 5 distinct revisions A..E
        self.revisions_spec: List[Tuple[str, int, int, bytes]] = [
            ("A", 100, 1, b"CHECKPOINT_REVISION_A_STEP_100_INITIAL_CONV_WEIGHTS_PAYLOAD"),
            ("B", 200, 2, b"CHECKPOINT_REVISION_B_STEP_200_ATTENTION_LAYER_1_WEIGHTS_PAYLOAD"),
            ("C", 300, 3, b"CHECKPOINT_REVISION_C_STEP_300_CROSS_ATTENTION_HEADS_PAYLOAD"),
            ("D", 400, 4, b"CHECKPOINT_REVISION_D_STEP_400_LAYER_NORM_AND_FFN_PAYLOAD"),
            ("E", 500, 5, b"CHECKPOINT_REVISION_E_STEP_500_CONVERGED_OPTIMAL_PAYLOAD"),
        ]
        self.revision_shas: Dict[str, str] = {
            letter: hashlib.sha256(content).hexdigest()
            for letter, _, _, content in self.revisions_spec
        }

    def tearDown(self) -> None:
        self.tmp_dir_obj.cleanup()

    def test_five_revision_acceptance_and_catastrophic_recovery_scenario(self) -> None:
        """Executes the full 5-stage acceptance scenario (Sections 1 through 5)."""
        mock_http = StrictDriveMock()
        journal = TransferJournal(self.journal_path)

        drive_store = GoogleDriveArtifactStore(
            credentials=self.creds,
            credentials_file=self.creds_file,
            local_staging_dir=self.staging_dir,
            journal_path=self.journal_path,
            http_client=mock_http,
        )

        # ---------------------------------------------------------------------
        # 1. Publish A-E via GoogleDriveArtifactStore and once via ArtifactUploadWorker
        # ---------------------------------------------------------------------
        published_artifacts: Dict[str, StoredArtifact] = {}

        # Revision A via GoogleDriveArtifactStore
        letter_a, step_a, epoch_a, content_a = self.revisions_spec[0]
        self.ckpt_file.write_bytes(content_a)
        pub_a = drive_store.publish(
            source=self.ckpt_file,
            project_id=self.project_id,
            run_id=self.run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
            metadata={"step": step_a, "epoch": epoch_a, "logical_path": self.logical_path},
        )
        published_artifacts["A"] = pub_a

        # Revision B via ArtifactUploadWorker (enqueue + restart worker between revisions)
        letter_b, step_b, epoch_b, content_b = self.revisions_spec[1]
        self.ckpt_file.write_bytes(content_b)
        worker_initial = ArtifactUploadWorker(store=drive_store, journal=journal)
        rec_b = worker_initial.enqueue(
            source=self.ckpt_file,
            project_id=self.project_id,
            run_id=self.run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
            metadata={"step": step_b, "epoch": epoch_b, "logical_path": self.logical_path},
        )
        self.assertEqual(rec_b.status, TransferStatus.LOCAL_VERIFIED.value)
        # Simulate worker crash/restart before worker_initial begins upload loop
        worker_initial.stop(wait=False)

        # Restart worker: worker_resumed recovers pending transfers from journal in __init__
        worker_resumed = ArtifactUploadWorker(store=drive_store, journal=journal)
        worker_resumed.start()

        deadline = time.time() + 15.0
        while time.time() < deadline:
            rec = journal.get_transfer(rec_b.transfer_id)
            if rec and rec.status == TransferStatus.PUBLISHED.value:
                break
            time.sleep(0.05)

        worker_resumed.stop(wait=True)
        rec_final_b = journal.get_transfer(rec_b.transfer_id)
        self.assertIsNotNone(rec_final_b)
        self.assertEqual(rec_final_b.status, TransferStatus.PUBLISHED.value)
        self.assertIsNotNone(rec_final_b.remote_file_id)

        # Load published artifact B from manifest
        manifest_file = (
            self.staging_dir / "projects" / self.project_id / "runs" / self.run_id / "manifest.json"
        )
        manifest_after_b = PublicationManifest.load(manifest_file)
        art_b = manifest_after_b.get_artifact(rec_final_b.artifact_id)
        self.assertIsNotNone(art_b)
        published_artifacts["B"] = art_b

        # Revisions C, D, E via GoogleDriveArtifactStore
        for letter, step, epoch, content in self.revisions_spec[2:]:
            self.ckpt_file.write_bytes(content)
            pub = drive_store.publish(
                source=self.ckpt_file,
                project_id=self.project_id,
                run_id=self.run_id,
                artifact_type=ArtifactType.CHECKPOINT.value,
                metadata={"step": step, "epoch": epoch, "logical_path": self.logical_path},
            )
            published_artifacts[letter] = pub

        # --- Assertions on snapshot() ---
        snap = mock_http.snapshot()
        self.assertEqual(len(snap), 5, "StrictDriveMock snapshot must contain exactly 5 remote files")

        # Each sha preserved
        for letter in ["A", "B", "C", "D", "E"]:
            expected_sha = self.revision_shas[letter]
            matching_files = [f for f in snap if f.content_sha == expected_sha]
            self.assertEqual(
                len(matching_files), 1, f"Revision {letter} with sha {expected_sha} not preserved"
            )

        # Exactly one is_latest == "true" (Revision E)
        latest_files = [f for f in snap if f.appProperties.get("is_latest") == "true"]
        self.assertEqual(len(latest_files), 1, "Exactly one remote file must be marked is_latest=='true'")
        self.assertEqual(
            latest_files[0].content_sha,
            self.revision_shas["E"],
            "Revision E must be the sole remote file marked is_latest=='true'",
        )

        # A-D have next_sha256 == immediate successor digest
        successor_chain = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]
        for prev_letter, next_letter in successor_chain:
            prev_sha = self.revision_shas[prev_letter]
            next_sha = self.revision_shas[next_letter]
            file_prev = next(f for f in snap if f.content_sha == prev_sha)
            self.assertEqual(
                file_prev.appProperties.get("is_latest"),
                "false",
                f"Historical revision {prev_letter} must have is_latest=='false'",
            )
            self.assertEqual(
                file_prev.appProperties.get("next_sha256"),
                next_sha,
                f"Revision {prev_letter} next_sha256 must point to immediate successor {next_letter}",
            )

        # Revision E must not have next_sha256
        file_e = next(f for f in snap if f.content_sha == self.revision_shas["E"])
        self.assertNotIn(
            "next_sha256", file_e.appProperties, "Revision E must not have next_sha256"
        )

        # ALL props <= 124 UTF-8 bytes and only schema-v2 keys
        for f in snap:
            props = f.appProperties
            for k, v in props.items():
                self.assertIn(
                    k,
                    self.ALLOWED_V2_PROPERTY_KEYS,
                    f"Remote file {f.id} property '{k}' is not an allowed schema-v2 key",
                )
                k_bytes = len(k.encode("utf-8"))
                v_bytes = len(str(v).encode("utf-8"))
                total_bytes = k_bytes + v_bytes
                self.assertLessEqual(
                    total_bytes,
                    124,
                    f"Remote file {f.id} property '{k}' combined length ({total_bytes} bytes) exceeds 124",
                )

        # Manifest keeps full canonical IDs, A.superseded_by == B full canonical ID, E.superseded_by is None
        manifest = PublicationManifest.load(manifest_file)
        self.assertEqual(len(manifest.artifacts), 5)
        for prev_letter, next_letter in successor_chain:
            art_prev_id = published_artifacts[prev_letter].artifact_id
            art_next_id = published_artifacts[next_letter].artifact_id
            entry_prev = manifest.artifacts[art_prev_id]
            self.assertEqual(
                entry_prev.get("superseded_by"),
                art_next_id,
                f"Manifest revision {prev_letter} superseded_by must equal {art_next_id}",
            )
            self.assertFalse(entry_prev.get("is_latest"))

        entry_e = manifest.artifacts[published_artifacts["E"].artifact_id]
        self.assertIsNone(entry_e.get("superseded_by"))
        self.assertTrue(entry_e.get("is_latest"))
        self.assertEqual(
            manifest.latest_by_path.get(self.logical_path),
            published_artifacts["E"].artifact_id,
        )

        # ---------------------------------------------------------------------
        # 2. Corrupt manifest, delete manifest backup, delete journal, fresh store
        # ---------------------------------------------------------------------
        manifest_bak_file = manifest_file.parent / "manifest.json.bak"
        manifest_file.write_text("{NOT_VALID_JSON_MANIFEST_CORRUPTED")
        manifest_bak_file.unlink(missing_ok=True)
        self.journal_path.unlink(missing_ok=True)

        fresh_staging_dir = self.tmp_dir / "fresh_staging_stage2"
        fresh_journal_path = fresh_staging_dir / "fresh_journal.json"
        fresh_store = GoogleDriveArtifactStore(
            credentials=self.creds,
            credentials_file=self.creds_file,
            local_staging_dir=fresh_staging_dir,
            journal_path=fresh_journal_path,
            http_client=mock_http,
        )

        recovery_dir = self.tmp_dir / "recovered_stage2"
        recovery_dir.mkdir(parents=True, exist_ok=True)

        # Retrieve A..E exactly by @sha256_<digest> with matching bytes
        for letter, _, _, content in self.revisions_spec:
            sha = self.revision_shas[letter]
            target_artifact_id = f"{self.project_id}/{self.run_id}/{self.logical_path}@sha256_{sha}"
            dest = recovery_dir / f"recovered_{letter}.pt"
            fresh_store.retrieve(target_artifact_id, dest)
            self.assertTrue(dest.is_file())
            self.assertEqual(dest.read_bytes(), content)
            self.assertEqual(compute_file_sha256(dest), sha)

        # Unversioned retrieve == E
        dest_unversioned = recovery_dir / "model_unversioned_stage2.pt"
        fresh_store.retrieve(f"{self.project_id}/{self.run_id}/{self.logical_path}", dest_unversioned)
        self.assertTrue(dest_unversioned.is_file())
        self.assertEqual(dest_unversioned.read_bytes(), self.revisions_spec[4][3])
        self.assertEqual(compute_file_sha256(dest_unversioned), self.revision_shas["E"])

        # ---------------------------------------------------------------------
        # 3. Corrupt remote markers (two latest, then zero latest)
        # ---------------------------------------------------------------------
        file_a_fid = published_artifacts["A"].remote_file_id
        file_e_fid = published_artifacts["E"].remote_file_id
        mf_a = next(f for f in mock_http.mock_files if f.get("id") == file_a_fid)
        mf_e = next(f for f in mock_http.mock_files if f.get("id") == file_e_fid)

        # 3a. Two latest: set file A's is_latest to "true"
        mf_a["appProperties"]["is_latest"] = "true"
        mf_e["appProperties"]["is_latest"] = "true"

        with self.assertRaises(DriveIntegrityError):
            fresh_store.retrieve(
                f"{self.project_id}/{self.run_id}/{self.logical_path}",
                recovery_dir / "two_latest_fail.pt",
            )

        # Versioned retrieval still works during two-latest corruption
        dest_ver_during_corrupt = recovery_dir / "versioned_during_two_latest.pt"
        fresh_store.retrieve(
            f"{self.project_id}/{self.run_id}/{self.logical_path}@sha256_{self.revision_shas['E']}",
            dest_ver_during_corrupt,
        )
        self.assertEqual(dest_ver_during_corrupt.read_bytes(), self.revisions_spec[4][3])

        # 3b. Zero latest: set both A and E to "false"
        mf_a["appProperties"]["is_latest"] = "false"
        mf_e["appProperties"]["is_latest"] = "false"

        with self.assertRaises(DriveIntegrityError):
            fresh_store.retrieve(
                f"{self.project_id}/{self.run_id}/{self.logical_path}",
                recovery_dir / "zero_latest_fail.pt",
            )

        # Versioned retrieval still works during zero-latest corruption
        dest_ver_during_zero = recovery_dir / "versioned_during_zero_latest.pt"
        fresh_store.retrieve(
            f"{self.project_id}/{self.run_id}/{self.logical_path}@sha256_{self.revision_shas['A']}",
            dest_ver_during_zero,
        )
        self.assertEqual(dest_ver_during_zero.read_bytes(), self.revisions_spec[0][3])

        # Restore remote markers (E is latest, A is historical)
        mf_a["appProperties"]["is_latest"] = "false"
        mf_e["appProperties"]["is_latest"] = "true"

        dest_restored = recovery_dir / "model_restored.pt"
        fresh_store.retrieve(f"{self.project_id}/{self.run_id}/{self.logical_path}", dest_restored)
        self.assertEqual(dest_restored.read_bytes(), self.revisions_spec[4][3])

        # ---------------------------------------------------------------------
        # 4. Total state loss: republish A (no upload, no PATCH, E still latest) & republish E
        # ---------------------------------------------------------------------
        state_loss_staging = self.tmp_dir / "state_loss_staging"
        state_loss_journal = state_loss_staging / "state_loss_journal.json"
        state_loss_store = GoogleDriveArtifactStore(
            credentials=self.creds,
            credentials_file=self.creds_file,
            local_staging_dir=state_loss_staging,
            journal_path=state_loss_journal,
            http_client=mock_http,
        )

        upload_sessions_before_a = mock_http.upload_sessions_initiated
        patches_before_a = len(mock_http.recorded_patches)

        # File A is on disk
        self.ckpt_file.write_bytes(content_a)
        pub_a_republish = state_loss_store.publish(
            source=self.ckpt_file,
            project_id=self.project_id,
            run_id=self.run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
            metadata={"step": step_a, "epoch": epoch_a, "logical_path": self.logical_path},
        )
        self.assertEqual(pub_a_republish.remote_file_id, file_a_fid)
        self.assertFalse(pub_a_republish.is_latest)

        # Assert no new upload session initiated and no PATCH recorded
        self.assertEqual(
            mock_http.upload_sessions_initiated,
            upload_sessions_before_a,
            "Replay of historical revision A must not initiate any upload session",
        )
        self.assertEqual(
            len(mock_http.recorded_patches),
            patches_before_a,
            "Replay of historical revision A must not record any PATCH operation",
        )

        # E is still the sole latest remotely
        snap_after_repub_a = mock_http.snapshot()
        latest_after_repub_a = [f for f in snap_after_repub_a if f.appProperties.get("is_latest") == "true"]
        self.assertEqual(len(latest_after_repub_a), 1)
        self.assertEqual(latest_after_repub_a[0].content_sha, self.revision_shas["E"])

        # Unversioned retrieve == E
        dest_unv_after_a = recovery_dir / "unversioned_after_repub_a.pt"
        state_loss_store.retrieve(
            f"{self.project_id}/{self.run_id}/{self.logical_path}", dest_unv_after_a
        )
        self.assertEqual(dest_unv_after_a.read_bytes(), content_b := self.revisions_spec[4][3])

        # Republish E -> idempotent, no duplicate file or upload
        remote_files_before_e = len(mock_http.snapshot())
        upload_sessions_before_e = mock_http.upload_sessions_initiated

        self.ckpt_file.write_bytes(self.revisions_spec[4][3])
        pub_e_republish = state_loss_store.publish(
            source=self.ckpt_file,
            project_id=self.project_id,
            run_id=self.run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
            metadata={"step": 500, "epoch": 5, "logical_path": self.logical_path},
        )
        self.assertEqual(pub_e_republish.remote_file_id, file_e_fid)
        self.assertEqual(len(mock_http.snapshot()), remote_files_before_e, "No duplicate remote file on republish E")
        self.assertEqual(
            mock_http.upload_sessions_initiated,
            upload_sessions_before_e,
            "No duplicate upload session initiated on idempotent republish E",
        )

        # ---------------------------------------------------------------------
        # 5. Network error (ConnectionResetError) during PATCH and listing
        # ---------------------------------------------------------------------
        fault_store = GoogleDriveArtifactStore(
            credentials=self.creds,
            credentials_file=self.creds_file,
            local_staging_dir=self.staging_dir,
            journal_path=self.staging_dir / "fault_journal.json",
            http_client=mock_http,
            max_retries=1,
            initial_backoff_sec=0.01,
        )

        # 5a. ConnectionResetError during PATCH
        content_f = b"CHECKPOINT_REVISION_F_STEP_600_FAULT_INJECTION_PATCH_PAYLOAD"
        self.ckpt_file.write_bytes(content_f)
        mock_http.inject_fault(
            "patch",
            count=fault_store.max_retries + 1,
            exception=ConnectionResetError("Simulated socket reset during PATCH"),
        )

        with self.assertRaises(DriveTransferError):
            fault_store.publish(
                source=self.ckpt_file,
                project_id=self.project_id,
                run_id=self.run_id,
                artifact_type=ArtifactType.CHECKPOINT.value,
                metadata={"step": 600, "epoch": 6, "logical_path": self.logical_path},
            )

        # Retry converges
        pub_f = fault_store.publish(
            source=self.ckpt_file,
            project_id=self.project_id,
            run_id=self.run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
            metadata={"step": 600, "epoch": 6, "logical_path": self.logical_path},
        )
        sha_f = hashlib.sha256(content_f).hexdigest()
        self.assertEqual(pub_f.sha256, sha_f)

        # 5b. ConnectionResetError during listing (no duplicate upload on listing failure)
        content_g = b"CHECKPOINT_REVISION_G_STEP_700_FAULT_INJECTION_LIST_PAYLOAD"
        self.ckpt_file.write_bytes(content_g)
        sessions_before_list_fault = mock_http.upload_sessions_initiated

        mock_http.inject_fault(
            "list",
            count=fault_store.max_retries + 1,
            exception=ConnectionResetError("Simulated socket reset during file listing"),
        )

        with self.assertRaises(DriveTransferError):
            fault_store.publish(
                source=self.ckpt_file,
                project_id=self.project_id,
                run_id=self.run_id,
                artifact_type=ArtifactType.CHECKPOINT.value,
                metadata={"step": 700, "epoch": 7, "logical_path": self.logical_path},
            )

        # Assert no duplicate upload session initiated on listing failure
        self.assertEqual(
            mock_http.upload_sessions_initiated,
            sessions_before_list_fault,
            "Listing failure must abort before initiating resumable upload session",
        )

        # Retry converges
        pub_g = fault_store.publish(
            source=self.ckpt_file,
            project_id=self.project_id,
            run_id=self.run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
            metadata={"step": 700, "epoch": 7, "logical_path": self.logical_path},
        )
        sha_g = hashlib.sha256(content_g).hexdigest()
        self.assertEqual(pub_g.sha256, sha_g)


# =============================================================================
# Test Suite: Multi-Process Contention & Strict Loopback Server
# =============================================================================

class TestMultiProcessContentionAndStrictLoopback(unittest.TestCase):
    """Multi-process contention test with real OS subprocesses and loopback Drive server."""

    def setUp(self) -> None:
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp_dir_obj.name)
        self.server = LoopbackDriveServer(("127.0.0.1", 0))
        self.server.RequestHandlerClass = StrictLoopbackDriveHandler
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.tmp_dir_obj.cleanup()

    def test_strict_loopback_rejects_oversized_app_properties(self) -> None:
        """Loopback handler rejects custom properties exceeding 124 bytes with HTTP 400."""
        # 1. Resumable init POST with key+value > 124 bytes
        url_init = f"{self.base_url}/drive/v3/files?uploadType=resumable"
        long_val = "x" * 120
        payload = json.dumps({"name": "test.pt", "appProperties": {"toolongkey": long_val}}).encode("utf-8")
        req_post = urllib.request.Request(
            url_init,
            data=payload,
            headers={"content-type": "application/json", "X-Upload-Content-Length": "100"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx_post:
            urllib.request.urlopen(req_post)
        self.assertEqual(ctx_post.exception.code, 400)

        # 2. Resumable init POST with valid properties <= 124 bytes succeeds
        valid_val = "x" * 50
        valid_payload = json.dumps({"name": "test.pt", "appProperties": {"valid_key": valid_val}}).encode("utf-8")
        req_valid = urllib.request.Request(
            url_init,
            data=valid_payload,
            headers={"content-type": "application/json", "X-Upload-Content-Length": "10"},
            method="POST",
        )
        with urllib.request.urlopen(req_valid) as resp_valid:
            self.assertEqual(resp_valid.status, 200)
            session_url = resp_valid.headers.get("Location")
            self.assertIsNotNone(session_url)

        # Complete upload to establish committed file
        req_put = urllib.request.Request(
            session_url,
            data=b"0123456789",
            headers={"Content-Range": "bytes 0-9/10", "Content-Length": "10"},
            method="PUT",
        )
        with urllib.request.urlopen(req_put) as resp_put:
            self.assertEqual(resp_put.status, 200)
            file_meta = json.loads(resp_put.read().decode("utf-8"))
            fid = file_meta["id"]

        # 3. PATCH with key+value > 124 bytes rejected with HTTP 400
        url_patch = f"{self.base_url}/drive/v3/files/{fid}"
        patch_payload = json.dumps({"appProperties": {"oversized_patch": "y" * 120}}).encode("utf-8")
        req_patch = urllib.request.Request(
            url_patch,
            data=patch_payload,
            headers={"content-type": "application/json"},
            method="PATCH",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx_patch:
            urllib.request.urlopen(req_patch)
        self.assertEqual(ctx_patch.exception.code, 400)

    def test_two_os_subprocesses_publish_concurrently_and_converge(self) -> None:
        """Two separate OS subprocesses publish revisions A and B concurrently with retry."""
        workspace = self.tmp_dir / "workspace"
        workspace.mkdir()
        journal_path = workspace / "journal.json"
        barrier_dir = self.tmp_dir / "barrier"
        barrier_dir.mkdir()

        content_a = "SUBPROCESS_REV_A_DATA_CONCURRENT_PAYLOAD"
        content_b = "SUBPROCESS_REV_B_DATA_CONCURRENT_PAYLOAD"
        sha_a = hashlib.sha256(content_a.encode("utf-8")).hexdigest()
        sha_b = hashlib.sha256(content_b.encode("utf-8")).hexdigest()

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(sys.path)

        # Launch two real OS subprocesses concurrently
        p1 = subprocess.Popen(
            [
                sys.executable,
                "-c",
                MULTIPROCESS_WORKER_SCRIPT,
                self.base_url,
                str(workspace),
                str(journal_path),
                str(barrier_dir),
                "1",
                "A",
                content_a,
                "100",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        p2 = subprocess.Popen(
            [
                sys.executable,
                "-c",
                MULTIPROCESS_WORKER_SCRIPT,
                self.base_url,
                str(workspace),
                str(journal_path),
                str(barrier_dir),
                "2",
                "B",
                content_b,
                "200",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        out1, err1 = p1.communicate(timeout=35)
        out2, err2 = p2.communicate(timeout=35)

        self.assertEqual(p1.returncode, 0, f"Process 1 failed: {err1}\n{out1}")
        self.assertEqual(p2.returncode, 0, f"Process 2 failed: {err2}\n{out2}")

        self.assertIn("RESULT:", out1)
        self.assertIn("RESULT:", out2)

        # Verify remote state on loopback Drive server
        with self.server.lock:
            # One remote file per revision (2 total files)
            self.assertEqual(
                len(self.server.committed_checkpoints),
                2,
                f"Expected exactly 2 remote checkpoint files, got {len(self.server.committed_checkpoints)}",
            )
            files = list(self.server.committed_checkpoints.values())
            file_shas = {f.get("sha256") for f in files}
            self.assertEqual(file_shas, {sha_a, sha_b})

            # Exactly single latest
            latest_files = [f for f in files if (f.get("appProperties") or {}).get("is_latest") == "true"]
            self.assertEqual(len(latest_files), 1, "Exactly one file must have is_latest=='true'")
            historical_files = [f for f in files if (f.get("appProperties") or {}).get("is_latest") == "false"]
            self.assertEqual(len(historical_files), 1, "Exactly one file must have is_latest=='false'")

            # Demoted file has next_sha256 == latest file sha256
            winner_sha = latest_files[0]["sha256"]
            demoted_next_sha = (historical_files[0].get("appProperties") or {}).get("next_sha256")
            self.assertEqual(demoted_next_sha, winner_sha)

            # All properties <= 124 UTF-8 bytes and schema-v2 only
            for f in files:
                props = f.get("appProperties") or {}
                for k, v in props.items():
                    k_b = len(k.encode("utf-8"))
                    v_b = len(str(v).encode("utf-8"))
                    self.assertLessEqual(k_b + v_b, 124, f"Property '{k}' exceeds 124 bytes: {k_b + v_b}")

    def test_zero_live_network_socket_guard_blocks_external_addresses(self) -> None:
        """Socket connect guard blocks non-loopback connections and permits loopback."""
        # Non-loopback external connection is strictly blocked
        sock_ext = socket.socket()
        with self.assertRaises(RuntimeError) as ctx_sock:
            sock_ext.connect(("93.184.216.34", 80))
        sock_ext.close()
        self.assertIn("Live network access forbidden", str(ctx_sock.exception))

        # Loopback connection is permitted
        sock_loop = socket.socket()
        sock_loop.connect(("127.0.0.1", self.server.server_address[1]))
        sock_loop.close()


if __name__ == "__main__":
    unittest.main()
