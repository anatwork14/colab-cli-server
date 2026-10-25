"""Offline end-to-end acceptance scenario for the PR #24 v2 hardening (Worker E).

World: two projects, 'vision-research' and 'federated-ids', both with run 'run_001'.
  * vision-research/model.pt revisions A..E. A is pushed to the Drive mock with its promotion
    PATCH failing (left PENDING, not in the shared manifest), then B (local), C (Drive),
    D (local), E (Drive) are published.
  * federated-ids/model.pt: F (local) and G (Drive), a different checkpoint.
  * Telemetry for both projects uses the same run id and the same checkpoint filename.

Revisions are retrieved through the store that owns them: manifest entries with
storage_provider 'local' through LocalArtifactStore, everything else (Drive-published or
pending) through a fresh GoogleDriveArtifactStore on the same Drive mock state.

The main scenario test runs in order: monitor queries (bearer auth), telemetry updates,
dashboards per project, fresh store instances (process restart), manifest.json-only loss with
backup recovery, exact retrieval of all five, stale pending A replayed after E (E must stay
head), historical replay of a local revision, cross-process concurrent publication into
federated-ids, path traversal attempts and final SHA-256 checks.

Focused tests at the end pin cross-store interop expectations. They are NOT weakened: a failure
there is a real finding in the current code.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import re
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from colab_adapter.errors import ArtifactNotFoundError, ArtifactValidationError, ColabV2Error, StorageError
from colab_adapter.monitor.collector import AccountUsageCollector
from colab_adapter.monitor.server import start_monitor_server
from colab_adapter.provider import ManagedSessionRegistry
from colab_adapter.storage import ArtifactType, DriveCredentials, GoogleDriveArtifactStore
from colab_adapter.storage.base import format_logical_artifact_id, format_revision_artifact_id
from colab_adapter.storage.local import LocalArtifactStore
from colab_adapter.telemetry import TelemetryEmitter
from tests.drive_api_mock import FileBackedDriveMock
from tests.test_adversarial_qa_pr24 import _mp_drive_worker, _mp_local_worker

VISION = "vision-research"
FEDERATED = "federated-ids"
RUN = "run_001"
LOGICAL = "checkpoints/model.pt"
TOKEN = "acceptance-token-pr24-0123456789"
HEX64 = re.compile(r"[0-9a-f]{64}")
REFUSAL = (StorageError, ArtifactValidationError, ArtifactNotFoundError, ColabV2Error)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class AcceptanceScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="accept_pr24_"))
        self.addCleanup(lambda: shutil.rmtree(str(self.tmp), ignore_errors=True))
        self.root = self.tmp / "store"
        self.src_dir = self.tmp / "src"
        self.src_dir.mkdir()
        self.state = self.tmp / "drive_state.pkl"
        self.cred = self.tmp / "creds.json"
        DriveCredentials(
            client_id="cid", client_secret="csecret", refresh_token="rtoken",
            access_token="atoken", token_expiry=9999999999.0,
        ).save(self.cred)
        self._n = 0

        self.local = LocalArtifactStore(self.root)
        self.drive = self._drive_store()

        self.vision_bytes = {k: f"VISION-REVISION-{k}-".encode("utf-8") * 300 for k in "ABCDE"}
        self.vision_sha = {k: _sha(v) for k, v in self.vision_bytes.items()}
        self.fed_bytes = {
            "F": b"FEDERATED-CHECKPOINT-F-" * 300,
            "G": b"FEDERATED-CHECKPOINT-G-" * 300,
        }
        self.fed_sha = {k: _sha(v) for k, v in self.fed_bytes.items()}

        # A: Drive upload whose promotion keeps failing -> A stays PENDING on Drive.
        self._set_drive_fail_patches(100)
        try:
            with self.assertRaises(Exception):
                self.drive.publish(self._src("A", self.vision_bytes["A"]), VISION, RUN, ArtifactType.CHECKPOINT.value)
        finally:
            self._set_drive_fail_patches(0)
        self.local.publish(self._src("B", self.vision_bytes["B"]), VISION, RUN, "checkpoint")
        self.drive.publish(self._src("C", self.vision_bytes["C"]), VISION, RUN, ArtifactType.CHECKPOINT.value)
        self.local.publish(self._src("D", self.vision_bytes["D"]), VISION, RUN, "checkpoint")
        self.drive.publish(self._src("E", self.vision_bytes["E"]), VISION, RUN, ArtifactType.CHECKPOINT.value)

        self.local.publish(self._src("F", self.fed_bytes["F"]), FEDERATED, RUN, "checkpoint")
        self.drive.publish(self._src("G", self.fed_bytes["G"]), FEDERATED, RUN, ArtifactType.CHECKPOINT.value)

        self.ledger = self.tmp / "telemetry.jsonl"
        self.acct = self.tmp / "account.jsonl"
        self.tel_vision = TelemetryEmitter(ledger_path=self.ledger, run_id=RUN, project_id=VISION)
        self.tel_fed = TelemetryEmitter(ledger_path=self.ledger, run_id=RUN, project_id=FEDERATED)
        for step in range(1, 4):
            self.tel_vision.emit(event_type="train_step", epoch=step, step=step * 100, metrics={"loss": 1.0 / step})
            self.tel_fed.emit(event_type="train_step", epoch=step, step=step * 7, metrics={"loss": 50.0 + step})
        self.tel_vision.emit(event_type="checkpoint_saved", epoch=3, step=300)
        self.tel_fed.emit(event_type="checkpoint_saved", epoch=3, step=21)

        self.server = start_monitor_server(
            host="127.0.0.1", port=0,
            collector=AccountUsageCollector(ledger_path=self.acct),
            registry=ManagedSessionRegistry(self.tmp / "sessions.json"),
            telemetry=TelemetryEmitter(ledger_path=self.ledger),
            auth_token=TOKEN,
            allow_external=False,
            artifacts_dir=self.root,
        )
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self._stop_server)

    # ------------------------------------------------------------------ helpers
    def _stop_server(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _drive_store(self) -> GoogleDriveArtifactStore:
        return GoogleDriveArtifactStore(
            credentials=DriveCredentials.load(self.cred),
            credentials_file=self.cred,
            local_staging_dir=self.root,
            http_client=FileBackedDriveMock(self.state),
            chunk_size=1024 * 1024,
            initial_backoff_sec=0.001,
        )

    def _set_drive_fail_patches(self, n: int) -> None:
        mock = FileBackedDriveMock(self.state)
        mock._load_state()
        mock.fail_next_n_patches = n
        mock._save_state()

    def _drive_files(self) -> List[Dict[str, Any]]:
        mock = FileBackedDriveMock(self.state)
        mock._load_state()
        return list(mock.mock_files)

    def _src(self, tag: str, data: bytes, name: str = "model.pt") -> Path:
        self._n += 1
        d = self.src_dir / f"{tag}_{self._n}"
        d.mkdir()
        p = d / name
        p.write_bytes(data)
        return p

    def _id(self, project: str, sha: str) -> str:
        return format_revision_artifact_id(project, RUN, LOGICAL, sha)

    def _head_id(self, project: str) -> str:
        return format_logical_artifact_id(project, RUN, LOGICAL)

    def _manifest_json(self, project: str) -> Dict[str, Any]:
        run_dir = self.root / "projects" / project / "runs" / RUN
        for name in ("manifest.json", "manifest.json.bak"):
            p = run_dir / name
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        return {"artifacts": {}, "latest_by_path": {}}

    def _entry(self, project: str, sha: str) -> Optional[Dict[str, Any]]:
        for entry in self._manifest_json(project)["artifacts"].values():
            if entry.get("sha256") == sha and entry.get("project_id", project) == project:
                return entry
        return None

    def _retrieve_exact(self, project: str, sha: str, dest: Path) -> Path:
        entry = self._entry(project, sha)
        if entry is not None and entry.get("storage_provider") == "local":
            return LocalArtifactStore(self.root).retrieve(self._id(project, sha), dest)
        return self._drive_store().retrieve(self._id(project, sha), dest)

    def _get(self, path: str):
        req = urllib.request.Request(self.base + path, headers={"Authorization": f"Bearer {TOKEN}"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read().decode("utf-8")
                return resp.status, json.loads(raw)
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                return e.code, json.loads(raw)
            except ValueError:
                return e.code, raw

    def _sse(self, project: str) -> List[Dict[str, Any]]:
        req = urllib.request.Request(self.base + f"/api/v1/events/stream?project={project}",
                                     headers={"Authorization": f"Bearer {TOKEN}"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode("utf-8")
        return [json.loads(line[len("data: "):]) for line in text.splitlines() if line.startswith("data: ")]

    def _dashboard_payloads(self, project: str) -> Dict[str, str]:
        payloads = {}
        status, body = self._get(f"/api/v1/runs/{RUN}?project={project}")
        self.assertEqual(status, 200, f"detail for {project}")
        payloads["detail"] = json.dumps(body)
        status, body = self._get(f"/api/v1/runs/{RUN}/metrics?project={project}")
        self.assertEqual(status, 200)
        payloads["metrics"] = json.dumps(body)
        status, body = self._get(f"/api/v1/runs/{RUN}/drive?project={project}")
        self.assertEqual(status, 200)
        payloads["drive"] = json.dumps(body)
        status, body = self._get(f"/api/v1/runs/{RUN}/manifest?project={project}")
        self.assertEqual(status, 200)
        payloads["manifest"] = json.dumps(body)
        payloads["sse"] = json.dumps(self._sse(project))
        return payloads

    def _assert_isolated(self, project: str, foreign_shas: Set[str], foreign_project: str) -> None:
        for name, payload in self._dashboard_payloads(project).items():
            with self.subTest(project=project, view=name):
                leaked = set(HEX64.findall(payload)) & foreign_shas
                self.assertFalse(leaked, f"{project} {name} view leaks foreign revisions")
                self.assertNotIn(f'"project_id": "{foreign_project}"', payload)
                self.assertNotIn(f'"{foreign_project}/', payload)

    # --------------------------------------------------------------- main flow
    def test_full_acceptance_scenario(self) -> None:
        vision_all = set(self.vision_sha.values())
        fed_all = set(self.fed_sha.values())

        # 1. Monitor: both projects appear for the same run id; unqualified legacy URL is ambiguous.
        status, runs = self._get("/api/v1/runs")
        self.assertEqual(status, 200)
        keys = sorted((r["project_id"], r["run_id"]) for r in runs if r["run_id"] == RUN)
        self.assertEqual(keys, [(FEDERATED, RUN), (VISION, RUN)])
        status, body = self._get(f"/api/v1/runs/{RUN}")
        self.assertEqual(status, 409)
        self.assertEqual(sorted(body["projects"]), [FEDERATED, VISION])

        # 2. Dashboards before any restart: isolated; traversal selectors are rejected.
        self._assert_isolated(VISION, fed_all, FEDERATED)
        self._assert_isolated(FEDERATED, vision_all, VISION)
        for bad in ("../federated-ids", "vision-research/../federated-ids", "..", "%2e%2e"):
            status, _ = self._get(f"/api/v1/runs/{RUN}?project={bad}")
            self.assertEqual(status, 400, bad)
        status, _ = self._get(f"/api/v1/runs/..%2F..%2F{FEDERATED}?project={VISION}")
        self.assertIn(status, (400, 404))

        # 3. Telemetry update: new events reach only their project's dashboard.
        self.tel_vision.emit(event_type="train_step", epoch=4, step=400, metrics={"loss": 0.01})
        status, runs = self._get("/api/v1/runs")
        by_project = {r["project_id"]: r for r in runs if r["run_id"] == RUN}
        self.assertEqual(by_project[VISION]["latest_step"], 400)
        self.assertEqual(by_project[FEDERATED]["latest_step"], 21)
        self.assertEqual(by_project[VISION]["latest_metrics"]["loss"], 0.01)
        self.assertEqual(by_project[FEDERATED]["latest_metrics"]["loss"], 53.0)

        # 4. Stale pending A is on Drive as non-head; the Drive head must be E.
        remote_a = [f for f in self._drive_files() if f.get("appProperties", {}).get("sha256") == self.vision_sha["A"]]
        self.assertEqual(len(remote_a), 1)
        self.assertNotEqual(remote_a[0]["appProperties"].get("is_latest"), "true")
        drive_head = self._drive_store().get_metadata(self._head_id(VISION))
        self.assertEqual(drive_head.sha256, self.vision_sha["E"])

        # 5. Process restart: fresh store instances see the same history.
        local2 = LocalArtifactStore(self.root)
        drive2 = self._drive_store()
        for k in "BD":
            meta = local2.get_metadata(self._id(VISION, self.vision_sha[k]))
            self.assertIsNotNone(meta, k)
            self.assertEqual(meta.sha256, self.vision_sha[k], k)
        for k in "CE":
            meta = drive2.get_metadata(self._id(VISION, self.vision_sha[k]))
            self.assertIsNotNone(meta, k)
            self.assertEqual(meta.sha256, self.vision_sha[k], k)
        self.assertEqual(drive2.get_metadata(self._head_id(VISION)).sha256, self.vision_sha["E"])
        self.assertEqual(drive2.get_metadata(self._head_id(FEDERATED)).sha256, self.fed_sha["G"])

        # 6. Manifest backup recovery: lose manifest.json only, history must survive.
        (self.root / "projects" / VISION / "runs" / RUN / "manifest.json").unlink()
        self.assertEqual(self._drive_store().get_metadata(self._head_id(VISION)).sha256, self.vision_sha["E"])
        for k in "BCDE":
            dest = self.tmp / f"restored_{k}.pt"
            self._retrieve_exact(VISION, self.vision_sha[k], dest)
            self.assertEqual(_file_sha(dest), self.vision_sha[k], k)

        # 7. Exact retrieval of all five, including the pending A (Drive only).
        for k in "ABCDE":
            dest = self.tmp / f"exact_{k}.pt"
            self._retrieve_exact(VISION, self.vision_sha[k], dest)
            self.assertEqual(_file_sha(dest), self.vision_sha[k], f"exact {k}")

        # 8. Stale pending A replayed after E: must not demote E, no duplicate upload.
        drive3 = self._drive_store()
        replay_a = drive3.publish(self._src("A2", self.vision_bytes["A"]), VISION, RUN, ArtifactType.CHECKPOINT.value)
        self.assertEqual(replay_a.sha256, self.vision_sha["A"])
        self.assertFalse(replay_a.is_latest, "stale pending A reported as latest after E")
        self.assertEqual(drive3.get_metadata(self._head_id(VISION)).sha256, self.vision_sha["E"])
        remote = self._drive_files()
        heads = [f for f in remote if f.get("appProperties", {}).get("is_latest") == "true"
                 and f.get("appProperties", {}).get("sha256") in vision_all]
        self.assertEqual([h["appProperties"]["sha256"] for h in heads], [self.vision_sha["E"]])
        self.assertEqual(
            len([f for f in remote if f.get("appProperties", {}).get("sha256") == self.vision_sha["A"]]), 1,
            "replay uploaded A again",
        )

        # 9. Historical replay of a local revision (B) on the local store: head and alias stay E.
        head_before = self._manifest_json(VISION)["latest_by_path"][LOGICAL]
        replay_b = LocalArtifactStore(self.root).publish(
            self._src("B2", self.vision_bytes["B"]), VISION, RUN, "checkpoint"
        )
        self.assertFalse(replay_b.is_latest)
        self.assertEqual(replay_b.sha256, self.vision_sha["B"])
        alias = self.root / "projects" / VISION / "runs" / RUN / "checkpoints" / "model.pt"
        with self.subTest("mutable alias must track the manifest head (E, written by Drive)"):
            self.assertEqual(_file_sha(alias), self.vision_sha["E"])
        self.assertEqual(self._manifest_json(VISION)["latest_by_path"][LOGICAL], head_before)

        # 10. Cross-process concurrent publication into federated-ids (local + Drive, OS processes).
        ctx = multiprocessing.get_context("spawn")
        procs = [
            ctx.Process(target=_mp_local_worker, args=(str(self.root), str(self.src_dir), 3, "xl", FEDERATED, RUN)),
            ctx.Process(target=_mp_drive_worker,
                        args=(str(self.root), str(self.cred), str(self.state), str(self.src_dir), 3, "xd",
                              FEDERATED, RUN)),
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(300)
        self.assertEqual([p.exitcode for p in procs], [0, 0])
        fed_manifest = self._manifest_json(FEDERATED)
        self.assertEqual(len(fed_manifest["artifacts"]), 8, "concurrent federated publications lost or duplicated")
        self.assertEqual({a["sha256"] for a in self._manifest_json(VISION)["artifacts"].values()
                          if a.get("sha256") in vision_all}, {a["sha256"] for a in self._manifest_json(VISION)["artifacts"].values()
                                                             if a.get("sha256") in vision_all})
        self.assertEqual(self._manifest_json(VISION)["latest_by_path"][LOGICAL], head_before,
                         "federated work moved the vision head")
        self.assertEqual(self._drive_store().get_metadata(self._head_id(FEDERATED)).sha256, self.fed_sha["G"])

        # 11. Dashboards after restart, backup recovery, replay and concurrency.
        fed_all_now = fed_all | {_sha(f"xl-payload-{i}".encode("utf-8") * 64) for i in range(3)} \
            | {_sha(f"xd-drive-{i}".encode("utf-8") * 64) for i in range(3)}
        self._assert_isolated(VISION, fed_all_now, FEDERATED)
        self._assert_isolated(FEDERATED, vision_all, VISION)

        # 12. Store-level traversal attempts never succeed.
        traversal_ids = [
            f"{VISION}/{RUN}/../../{FEDERATED}/{RUN}/checkpoints/model.pt",
            f"{VISION}/{RUN}/checkpoints/../../../{FEDERATED}/{RUN}/checkpoints/model.pt",
            f"../{VISION}/{RUN}/checkpoints/model.pt",
            f"{VISION}/{RUN}/checkpoints/model.pt@sha256_../../../../etc",
        ]
        for bad in traversal_ids:
            with self.subTest(artifact_id=bad):
                leaked = None
                try:
                    leaked = LocalArtifactStore(self.root).get_metadata(bad)
                except REFUSAL:
                    leaked = None
                self.assertIsNone(leaked)
                with self.assertRaises(REFUSAL):
                    LocalArtifactStore(self.root).retrieve(bad, self.tmp / "escape.pt")
        self.assertFalse((self.tmp / "escape.pt").exists())
        with self.assertRaises(REFUSAL):
            LocalArtifactStore(self.root).publish(self._src("X", b"X"), "../" + VISION, RUN, "checkpoint")

        # 13. Final SHA-256 checks on every revision and both heads.
        for k in "ABCDE":
            dest = self.tmp / f"final_{k}.pt"
            self._retrieve_exact(VISION, self.vision_sha[k], dest)
            self.assertEqual(_file_sha(dest), self.vision_sha[k], f"vision {k}")
        head_dest = self.tmp / "final_head.pt"
        self._drive_store().retrieve(self._id(VISION, self.vision_sha["E"]), head_dest)
        self.assertEqual(_file_sha(head_dest), self.vision_sha["E"])
        fed_dest = self.tmp / "final_fed.pt"
        self._retrieve_exact(FEDERATED, self.fed_sha["G"], fed_dest)
        self.assertEqual(_file_sha(fed_dest), self.fed_sha["G"])
        fed_f = self.tmp / "final_fed_F.pt"
        self._retrieve_exact(FEDERATED, self.fed_sha["F"], fed_f)
        self.assertEqual(_file_sha(fed_f), self.fed_sha["F"])

    # ---------------------------------------------------------- focused checks
    def test_monitor_rejects_unauthenticated_and_query_token(self) -> None:
        for path in (f"/api/v1/runs/{RUN}?project={VISION}", "/api/v1/runs"):
            req = urllib.request.Request(self.base + path)
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(req, timeout=10)
            self.assertEqual(cm.exception.code, 401)
        req = urllib.request.Request(self.base + f"/api/v1/runs/{RUN}?project={VISION}&token={TOKEN}")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(cm.exception.code, 401)

    def test_stale_pending_replay_alone_keeps_head(self) -> None:
        drive = self._drive_store()
        drive.publish(self._src("A4", self.vision_bytes["A"]), VISION, RUN, ArtifactType.CHECKPOINT.value)
        self.assertEqual(drive.get_metadata(self._head_id(VISION)).sha256, self.vision_sha["E"])

    def test_dashboards_isolated_with_identical_checkpoint_filenames(self) -> None:
        self._assert_isolated(VISION, set(self.fed_sha.values()), FEDERATED)
        self._assert_isolated(FEDERATED, set(self.vision_sha.values()), VISION)

    # ------------------------------------------- findings: cross-store interop
    def test_dashboard_does_not_list_internal_lock_file_as_artifact(self) -> None:
        status, body = self._get(f"/api/v1/runs/{RUN}?project={VISION}")
        self.assertEqual(status, 200)
        names = [str(a.get("logical_path") or a.get("path") or "") for a in body["artifacts"]]
        self.assertFalse(any(".locks" in n or "manifest.lock" in n for n in names), names)

    def test_local_store_serves_metadata_for_drive_published_head(self) -> None:
        meta = LocalArtifactStore(self.root).get_metadata(self._head_id(VISION))
        self.assertIsNotNone(meta)
        self.assertEqual(meta.sha256, self.vision_sha["E"])

    def test_local_publish_of_pending_drive_revision_does_not_steal_head(self) -> None:
        LocalArtifactStore(self.root).publish(self._src("A5", self.vision_bytes["A"]), VISION, RUN, "checkpoint")
        self.assertEqual(self._manifest_json(VISION)["latest_by_path"][LOGICAL], self._id(VISION, self.vision_sha["E"]),
                         "local publish of stale pending A took the head from Drive's E")
        self.assertEqual(self._drive_store().get_metadata(self._head_id(VISION)).sha256, self.vision_sha["E"])


if __name__ == "__main__":
    unittest.main()
