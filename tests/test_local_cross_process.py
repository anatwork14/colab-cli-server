"""Real OS-process tests for LocalArtifactStore and the manifest lock (contract C2/A5, C5).

All child processes use the 'spawn' start method, so no state is inherited from the parent.
Every join has a timeout and stuck children are killed, so a deadlock fails the test instead
of hanging the suite.
"""
from __future__ import annotations

import glob
import hashlib
import multiprocessing as mp
import os
import shutil
import signal
import tempfile
import threading
import time
import unittest
from pathlib import Path

from colab_adapter.errors import ArtifactValidationError, JournalLockError
from colab_adapter.storage.local import LocalArtifactStore
from colab_adapter.storage.manifest import PublicationManifest
from tests.test_local_exact_revision import LP, PROJ, RUN, _sha

JOIN_TIMEOUT = 90.0
_CTX = mp.get_context("spawn")


# --------------------------------------------------------------------------- #
# Child entry points (module level so the spawn start method can import them)
# --------------------------------------------------------------------------- #

def _child_publish(root, src, project, run, lock_timeout, start_barrier, result_q, crash_before_commit):
    try:
        if crash_before_commit:
            # Die after the immutable revision file is written but before the manifest is committed.
            def _die_before_commit(self, dest_path):
                os._exit(17)

            PublicationManifest.save = _die_before_commit
        store = LocalArtifactStore(root_dir=root, lock_timeout=lock_timeout)
        if start_barrier is not None:
            start_barrier.wait(timeout=JOIN_TIMEOUT)
        stored = store.publish(source=src, project_id=project, run_id=run, artifact_type="checkpoint")
        result_q.put(("ok", stored.artifact_id, bool(stored.is_latest)))
    except BaseException as exc:  # report every failure; never leave the parent waiting
        result_q.put(("err", type(exc).__name__, str(exc)[:300]))


def _child_hold_lock(manifest_path, ready_evt):
    from colab_adapter.storage.locking import manifest_lock

    with manifest_lock(manifest_path, timeout=JOIN_TIMEOUT):
        ready_evt.set()
        time.sleep(3600)


def _child_slow_publish(root, src, project, run, progress_path):
    """Publish with a deliberately slow copy so the parent can SIGKILL it while holding the lock."""
    import shutil as _shutil

    real_copy = _shutil.copyfileobj

    def slow_copy(fin, fout, length=0):
        while True:
            chunk = fin.read(4096)
            if not chunk:
                break
            fout.write(chunk)
            fout.flush()
            with open(progress_path, "ab") as fh:
                fh.write(b".")
            time.sleep(0.005)

    _shutil.copyfileobj = slow_copy
    try:
        store = LocalArtifactStore(root_dir=root, lock_timeout=JOIN_TIMEOUT)
        store.publish(source=src, project_id=project, run_id=run, artifact_type="checkpoint")
    finally:
        _shutil.copyfileobj = real_copy


def _child_reader(root, exact_id, unversioned_id, scratch_dir, stop_evt, result_q):
    store = LocalArtifactStore(root_dir=root, lock_timeout=JOIN_TIMEOUT)
    errors = []
    ok = 0
    attempts = 0
    i = 0
    while not stop_evt.is_set():
        for artifact_id in (exact_id, unversioned_id):
            attempts += 1
            i += 1
            dest = os.path.join(scratch_dir, f"read_{i}.bin")
            try:
                meta = store.get_metadata(artifact_id)
                if meta is None:
                    errors.append(("missing", artifact_id))
                    continue
                store.retrieve(artifact_id, dest)
                os.unlink(dest)
                ok += 1
            except BaseException as exc:
                errors.append((type(exc).__name__, str(exc)[:200]))
    result_q.put((ok, attempts, errors[:20], len(errors)))


# --------------------------------------------------------------------------- #
# Test base
# --------------------------------------------------------------------------- #

class CrossProcessTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="local_xproc_")
        self.root = os.path.join(self.tmp, "store")
        self.work = os.path.join(self.tmp, "work")
        os.makedirs(self.work)
        self.procs = []
        self.store = LocalArtifactStore(root_dir=self.root, lock_timeout=JOIN_TIMEOUT)

    def tearDown(self):
        for p in self.procs:
            if p.is_alive():
                p.kill()
            p.join(timeout=10)
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

    def make_src(self, tag: str, size: int = 4096, slot: str = "") -> str:
        d = os.path.join(self.work, tag + slot)
        os.makedirs(d, exist_ok=True)
        payload = (f"{tag}|".encode() * (size // (len(tag) + 1) + 1))[:size]
        path = os.path.join(d, "model.pt")
        with open(path, "wb") as fh:
            fh.write(payload)
        return path

    def payload_of(self, src: str) -> bytes:
        return Path(src).read_bytes()

    def spawn(self, target, *args):
        p = _CTX.Process(target=target, args=args)
        p.daemon = True
        p.start()
        self.procs.append(p)
        return p

    def join_all(self, procs, timeout=JOIN_TIMEOUT):
        for p in procs:
            p.join(timeout=timeout)
            self.assertFalse(p.is_alive(), f"child {p.pid} did not finish (possible deadlock)")

    def collect(self, q, n, timeout=JOIN_TIMEOUT):
        return [q.get(timeout=timeout) for _ in range(n)]

    def publish_in_children(self, srcs, lock_timeout=JOIN_TIMEOUT):
        ctx = _CTX
        barrier = ctx.Barrier(len(srcs))
        q = ctx.Queue()
        procs = [
            self.spawn(_child_publish, self.root, src, PROJ, RUN, lock_timeout, barrier, q, False)
            for src in srcs
        ]
        results = self.collect(q, len(srcs))
        self.join_all(procs)
        return results

    def assert_consistent(self, payloads, expect_head=None):
        """Manifest parses; every revision exists once; lineage is a single chain; all exact revisions read back."""
        manifest = PublicationManifest.load(self.manifest_path, allow_backup_recovery=False)
        shas = {_sha(p) for p in payloads}
        revs = manifest.list_revisions(LP)
        self.assertEqual(len(revs), len(shas), "lost or duplicated revisions in manifest")
        self.assertEqual({r.sha256 for r in revs}, shas)
        self.assertEqual(len(manifest.artifacts), len(shas))

        head_id = manifest.latest_by_path[LP]
        head_sha = manifest.artifacts[head_id]["sha256"]
        self.assertIn(head_sha, shas)
        if expect_head is not None:
            self.assertEqual(head_sha, _sha(expect_head))

        # Lineage: every non-head revision reaches the head through superseded_by, without cycles.
        for rev in revs:
            cur = rev.artifact_id
            seen = set()
            while cur != head_id:
                self.assertNotIn(cur, seen, "superseded_by cycle")
                seen.add(cur)
                nxt = manifest.artifacts[cur].get("superseded_by")
                self.assertIsNotNone(nxt, f"{cur} is not superseded yet is not the head")
                self.assertIn(nxt, manifest.artifacts)
                cur = nxt

        for payload in payloads:
            sha = _sha(payload)
            rev_file = self.run_dir / "checkpoints" / ".revisions" / sha / "model.pt"
            self.assertEqual(_sha_file(rev_file), sha)
            store = LocalArtifactStore(root_dir=self.root)
            out = os.path.join(self.tmp, f"exact_{sha[:8]}.bin")
            store.retrieve(f"{PROJ}/{RUN}/{LP}@sha256_{sha}", out)
            self.assertEqual(Path(out).read_bytes(), payload)

        self.assertEqual(_sha_file(self.alias), head_sha)


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Concurrent publishing
# --------------------------------------------------------------------------- #

class ConcurrentPublishTest(CrossProcessTestBase):
    def test_concurrent_distinct_revisions_A_and_B(self):
        srcs = [self.make_src("A", 200_000), self.make_src("B", 300_000)]
        results = self.publish_in_children(srcs)
        self.assertEqual([r[0] for r in results], ["ok", "ok"], results)
        self.assert_consistent([self.payload_of(s) for s in srcs])

    def test_concurrent_repeated_revision_is_idempotent(self):
        src = self.make_src("A", 150_000)
        results = self.publish_in_children([src] * 4)
        self.assertEqual([r[0] for r in results], ["ok"] * 4, results)
        ids = {r[1] for r in results}
        self.assertEqual(len(ids), 1, "repeated publication must yield one revision id")
        self.assert_consistent([self.payload_of(src)], expect_head=self.payload_of(src))
        manifest = PublicationManifest.load(self.manifest_path)
        self.assertEqual(len(manifest.artifacts), 1)

    def test_concurrent_A_B_C_chain_has_single_consistent_lineage(self):
        srcs = [self.make_src("A", 100_000, "_s"), self.make_src("B", 120_000, "_s"), self.make_src("C", 140_000, "_s")]
        results = self.publish_in_children(srcs)
        self.assertEqual([r[0] for r in results], ["ok"] * 3, results)
        self.assert_consistent([self.payload_of(s) for s in srcs])

    def test_sequential_then_concurrent_repeat_of_history_keeps_head(self):
        a, b = self.make_src("A", 90_000, "_q"), self.make_src("B", 95_000, "_q")
        self.store.publish(source=a, project_id=PROJ, run_id=RUN, artifact_type="checkpoint")
        self.store.publish(source=b, project_id=PROJ, run_id=RUN, artifact_type="checkpoint")
        results = self.publish_in_children([a, a, a])
        self.assertEqual([r[0] for r in results], ["ok"] * 3, results)
        self.assert_consistent([self.payload_of(a), self.payload_of(b)], expect_head=self.payload_of(b))


class ReaderDuringPublishTest(CrossProcessTestBase):
    def test_reader_never_sees_errors_while_publishing_successors(self):
        a = self.make_src("A", 60_000, "_r")
        self.store.publish(source=a, project_id=PROJ, run_id=RUN, artifact_type="checkpoint")
        exact_a = f"{PROJ}/{RUN}/{LP}@sha256_{_sha(self.payload_of(a))}"
        unversioned = f"{PROJ}/{RUN}/{LP}"

        stop = _CTX.Event()
        q = _CTX.Queue()
        scratch = os.path.join(self.tmp, "reader_scratch")
        os.makedirs(scratch)
        reader = self.spawn(_child_reader, self.root, exact_a, unversioned, scratch, stop, q)

        payloads = [self.payload_of(a)]
        for tag in ("B", "C", "D"):
            src = self.make_src(tag, 70_000, "_r")
            payloads.append(self.payload_of(src))
            self.store.publish(source=src, project_id=PROJ, run_id=RUN, artifact_type="checkpoint")
        time.sleep(0.5)
        stop.set()
        ok, attempts, errors, n_errors = q.get(timeout=JOIN_TIMEOUT)
        self.join_all([reader])
        self.assertEqual(errors, [], f"reader observed errors during publish: {errors}")
        self.assertGreater(ok, 0)
        self.assert_consistent(payloads, expect_head=payloads[-1])


# --------------------------------------------------------------------------- #
# Lock behaviour across processes and crashes
# --------------------------------------------------------------------------- #

class ManifestLockProcessTest(CrossProcessTestBase):
    def _hold_lock_in_child(self):
        ready = _CTX.Event()
        holder = self.spawn(_child_hold_lock, str(self.manifest_path), ready)
        self.assertTrue(ready.wait(timeout=JOIN_TIMEOUT), "lock holder did not start")
        return holder

    def test_publish_times_out_while_another_process_holds_lock(self):
        # The manifest directory must exist before the lock can be taken.
        self.store.publish(source=self.make_src("A", 10_000, "_t"), project_id=PROJ, run_id=RUN, artifact_type="checkpoint")
        holder = self._hold_lock_in_child()
        try:
            impatient = LocalArtifactStore(root_dir=self.root, lock_timeout=0.5)
            src = self.make_src("B", 10_000, "_t")
            started = time.monotonic()
            with self.assertRaises(JournalLockError):
                impatient.publish(source=src, project_id=PROJ, run_id=RUN, artifact_type="checkpoint")
            self.assertLess(time.monotonic() - started, 15.0, "lock wait was not bounded")
        finally:
            holder.kill()
            holder.join(timeout=JOIN_TIMEOUT)
        # Once released, the same publish succeeds.
        stored = self.store.publish(source=self.make_src("B", 10_000, "_t"), project_id=PROJ, run_id=RUN, artifact_type="checkpoint")
        self.assertTrue(stored.is_latest)

    def test_sigkill_of_lock_holder_releases_lock_for_next_process(self):
        self.store.publish(source=self.make_src("A", 10_000, "_k"), project_id=PROJ, run_id=RUN, artifact_type="checkpoint")
        holder = self._hold_lock_in_child()
        os.kill(holder.pid, signal.SIGKILL)
        holder.join(timeout=JOIN_TIMEOUT)
        self.assertFalse(holder.is_alive())
        self.assertEqual(holder.exitcode, -signal.SIGKILL)

        q = _CTX.Queue()
        src = self.make_src("B", 10_000, "_k")
        proc = self.spawn(_child_publish, self.root, src, PROJ, RUN, 20.0, None, q, False)
        status = self.collect(q, 1)[0]
        self.join_all([proc])
        self.assertEqual(status[0], "ok", status)
        self.assert_consistent([self.payload_of(self.make_src("A", 10_000, "_k")), self.payload_of(src)], expect_head=self.payload_of(src))

    def test_sigkill_mid_publish_leaves_manifest_parseable_and_recoverable(self):
        self.store.publish(source=self.make_src("A", 10_000, "_m"), project_id=PROJ, run_id=RUN, artifact_type="checkpoint")
        big = self.make_src("B", 1_000_000, "_m")
        progress = os.path.join(self.tmp, "progress.bin")
        victim = self.spawn(_child_slow_publish, self.root, big, PROJ, RUN, progress)

        deadline = time.monotonic() + JOIN_TIMEOUT
        while time.monotonic() < deadline:
            if os.path.exists(progress) and os.path.getsize(progress) >= 20:
                break
            time.sleep(0.01)
        self.assertTrue(os.path.exists(progress), "victim never started copying")
        os.kill(victim.pid, signal.SIGKILL)
        victim.join(timeout=JOIN_TIMEOUT)
        self.assertEqual(victim.exitcode, -signal.SIGKILL)

        # Manifest is still parseable with no backup recovery and only holds A.
        manifest = PublicationManifest.load(self.manifest_path, allow_backup_recovery=False)
        self.assertEqual(len(manifest.artifacts), 1)
        self.assertEqual(manifest.latest_by_path[LP], f"{PROJ}/{RUN}/{LP}@sha256_{_sha(self.payload_of(self.make_src('A', 10_000, '_m')))}")

        # The killed publish left nothing committed; the lock is free and a retry completes.
        stored = self.store.publish(source=big, project_id=PROJ, run_id=RUN, artifact_type="checkpoint")
        self.assertTrue(stored.is_latest)
        self.assert_consistent([self.payload_of(self.make_src("A", 10_000, "_m")), self.payload_of(big)], expect_head=self.payload_of(big))

    def test_process_exit_after_immutable_write_before_manifest_commit(self):
        a = self.make_src("A", 30_000, "_x")
        b = self.make_src("B", 40_000, "_x")
        self.store.publish(source=a, project_id=PROJ, run_id=RUN, artifact_type="checkpoint")

        q = _CTX.Queue()
        crasher = self.spawn(_child_publish, self.root, b, PROJ, RUN, JOIN_TIMEOUT, None, q, True)
        self.join_all([crasher])
        self.assertEqual(crasher.exitcode, 17)

        # The immutable B file exists, but the manifest never recorded B and still points at A.
        b_rev = self.run_dir / "checkpoints" / ".revisions" / _sha(self.payload_of(b)) / "model.pt"
        self.assertTrue(b_rev.is_file())
        manifest = PublicationManifest.load(self.manifest_path, allow_backup_recovery=False)
        self.assertEqual(len(manifest.artifacts), 1)
        self.assertNotIn(f"{PROJ}/{RUN}/{LP}@sha256_{_sha(self.payload_of(b))}", manifest.artifacts)

        # A retry adopts the existing immutable file, commits B as head and keeps A.
        stored = self.store.publish(source=b, project_id=PROJ, run_id=RUN, artifact_type="checkpoint")
        self.assertTrue(stored.is_latest)
        self.assert_consistent([self.payload_of(a), self.payload_of(b)], expect_head=self.payload_of(b))


class LockPrimitiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="local_lockprim_")
        self.run_dir = Path(self.tmp) / "run"
        self.run_dir.mkdir()
        self.manifest = self.run_dir / "manifest.json"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_lock_file_location_and_modes(self):
        from colab_adapter.storage.locking import manifest_lock, manifest_lock_path

        self.assertEqual(manifest_lock_path(self.manifest), self.run_dir.resolve() / ".locks" / "manifest.lock")
        with manifest_lock(self.manifest, timeout=5) as lock_file:
            self.assertEqual(Path(lock_file), self.run_dir.resolve() / ".locks" / "manifest.lock")
            self.assertEqual(oct(os.stat(lock_file.parent).st_mode & 0o777), oct(0o700))
            self.assertEqual(oct(os.stat(lock_file).st_mode & 0o777), oct(0o600))

    def test_reentrant_nested_acquisition_in_same_thread(self):
        from colab_adapter.storage.locking import manifest_lock

        done = threading.Event()

        def nested():
            with manifest_lock(self.manifest, timeout=5):
                with manifest_lock(self.manifest, timeout=5):
                    with manifest_lock(self.manifest, timeout=5):
                        pass
            done.set()

        t = threading.Thread(target=nested, daemon=True)
        t.start()
        t.join(timeout=20)
        self.assertTrue(done.is_set(), "re-entrant acquisition deadlocked")

    def test_other_thread_times_out_instead_of_deadlocking(self):
        from colab_adapter.storage.locking import manifest_lock

        held = threading.Event()
        release = threading.Event()

        def holder():
            with manifest_lock(self.manifest, timeout=5):
                held.set()
                release.wait(timeout=30)

        t = threading.Thread(target=holder, daemon=True)
        t.start()
        self.assertTrue(held.wait(timeout=10))
        try:
            started = time.monotonic()
            with self.assertRaises(JournalLockError):
                with manifest_lock(self.manifest, timeout=0.3):
                    pass
            self.assertLess(time.monotonic() - started, 10.0)
        finally:
            release.set()
            t.join(timeout=20)

    def test_symlinked_lock_directory_is_refused(self):
        from colab_adapter.errors import StorageError
        from colab_adapter.storage.locking import manifest_lock

        outside = Path(self.tmp) / "outside_locks"
        outside.mkdir()
        os.symlink(str(outside), str(self.run_dir / ".locks"))
        with self.assertRaises(StorageError):
            with manifest_lock(self.manifest, timeout=1):
                pass


if __name__ == "__main__":
    unittest.main()
