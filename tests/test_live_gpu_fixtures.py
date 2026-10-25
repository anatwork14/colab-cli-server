"""Offline validation of the live-smoke GPU fixtures (examples/live-smoke).

Everything here runs locally without Colab, network, or torch. Tests that need
PyTorch are skipped when it is not importable.
"""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

REPO = Path(__file__).resolve().parents[1]
LIVE = REPO / "examples" / "live-smoke"
MANIFEST = LIVE / "colab-project.json"
WORKLOAD = LIVE / "workload_manifest.json"
PROBE = LIVE / "gpu_probe.py"
TRAIN = LIVE / "train_tiny.py"
PYTHON = sys.executable

# Runs a script with torch made unimportable (sys.modules entry set to None).
HIDE_TORCH = (
    "import runpy, sys; sys.modules['torch'] = None; "
    "script = sys.argv[1]; sys.argv = [script] + sys.argv[2:]; "
    "runpy.run_path(script, run_name='__main__')"
)

EXPECTED_FILES = {"gpu_probe.py", "train_tiny.py", "workload_manifest.json"}
TELEMETRY_KEYS = {
    "schema_version", "event_id", "run_id", "project_id", "emitted_at", "event_type",
    "epoch", "step", "total_steps", "metrics", "gpu_info", "source", "details",
}


def _torch_available():
    return importlib.util.find_spec("torch") is not None


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _env_without_colab_ledger():
    env = dict(os.environ)
    env.pop("COLAB_TELEMETRY_LEDGER", None)
    env.pop("COLAB_RUN_ID", None)
    return env


def _run_hidden_torch(script, args, cwd=None, env=None):
    return subprocess.run(
        [PYTHON, "-c", HIDE_TORCH, str(script)] + list(args),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
        cwd=cwd or str(REPO), env=env or _env_without_colab_ledger(), timeout=120,
    )


class TestLiveSmokeCompile(unittest.TestCase):
    def test_scripts_py_compile(self):
        tmp = tempfile.mkdtemp(prefix="live_smoke_compile_")
        try:
            for index, script in enumerate([PROBE, TRAIN]):
                py_compile.compile(str(script), cfile=os.path.join(tmp, "out%d.pyc" % index),
                                   doraise=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_no_pip_or_download_calls_in_scripts(self):
        for script in (PROBE, TRAIN):
            text = script.read_text(encoding="utf-8")
            for forbidden in ("subprocess", "os.system(", "urlopen", "import urllib", "import requests",
                              "wget ", "curl ", "pip._internal", "ensurepip"):
                self.assertNotIn(forbidden, text, "%s contains %r" % (script.name, forbidden))


class TestLiveSmokeManifest(unittest.TestCase):
    def test_load_manifest_validates(self):
        from colab_adapter.project_manifest import load_manifest

        spec = load_manifest(MANIFEST)
        self.assertEqual(spec.name, "live-smoke")
        self.assertEqual(spec.gpu, "T4")
        self.assertEqual(spec.entrypoint, "train_tiny.py")
        self.assertEqual(spec.remote_root, "/content/colab-workspaces/live-smoke")
        self.assertLessEqual(spec.timeout_seconds, 900.0)
        self.assertFalse(spec.retain_session)
        self.assertEqual(spec.root, LIVE.resolve())

    def test_plan_dry_run_reports_t4_and_zero_allocation(self):
        fake_bin = tempfile.mkdtemp(prefix="live_smoke_fake_colab_")
        marker = os.path.join(fake_bin, "colab_invoked")
        try:
            # A fake `colab` executable that records any invocation. The plan must never call it.
            fake = os.path.join(fake_bin, "colab")
            with open(fake, "w", encoding="utf-8") as handle:
                handle.write("#!/bin/sh\necho \"$@\" >> '%s'\nexit 99\n" % marker)
            os.chmod(fake, 0o755)
            env = _env_without_colab_ledger()
            env["PATH"] = fake_bin + os.pathsep + env.get("PATH", "")
            result = subprocess.run(
                [PYTHON, "-m", "colab_adapter.cli", "project", "plan",
                 "--manifest", str(MANIFEST), "--json"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                cwd=str(REPO), env=env, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["gpu"], "T4")
            self.assertEqual(plan["project_name"], "live-smoke")
            self.assertEqual(plan["entrypoint"], "train_tiny.py")
            self.assertEqual({f["path"] for f in plan["files"]}, EXPECTED_FILES)
            self.assertFalse(os.path.exists(marker), "plan invoked the colab CLI")
        finally:
            shutil.rmtree(fake_bin, ignore_errors=True)

    def test_validate_dry_run(self):
        result = subprocess.run(
            [PYTHON, "-m", "colab_adapter.cli", "project", "validate",
             "--manifest", str(MANIFEST), "--json"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
            cwd=str(REPO), env=_env_without_colab_ledger(), timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["valid"])


class TestWorkloadManifestHashes(unittest.TestCase):
    def test_hashes_and_sizes_match_files(self):
        doc = json.loads(WORKLOAD.read_text(encoding="utf-8"))
        self.assertEqual({entry["path"] for entry in doc["files"]}, EXPECTED_FILES - {"workload_manifest.json"})
        for entry in doc["files"]:
            path = LIVE / entry["path"]
            self.assertEqual(_sha256(path), entry["sha256"], entry["path"])
            self.assertEqual(path.stat().st_size, entry["size"], entry["path"])


class TestDegradeWithoutTorch(unittest.TestCase):
    def test_probe_without_torch_exits_3_with_marker(self):
        result = _run_hidden_torch(PROBE, [])
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertIn("CUDA available: False", result.stdout)
        self.assertIn("GPU_PROBE_NO_TORCH", result.stdout)
        self.assertNotIn("GPU_PROBE_OK", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_probe_require_gpu_without_torch_exits_2(self):
        result = _run_hidden_torch(PROBE, ["--require-gpu"])
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("GPU_PROBE_REQUIRE_GPU_FAILED", result.stdout)
        self.assertNotIn("GPU_PROBE_OK", result.stdout)

    def test_train_without_torch_exits_3_and_writes_nothing(self):
        tmp = tempfile.mkdtemp(prefix="live_smoke_notorch_")
        try:
            result = _run_hidden_torch(TRAIN, [], cwd=tmp)
            self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
            self.assertIn("TRAIN_NO_TORCH", result.stdout)
            self.assertNotIn("DONE_OK", result.stdout)
            self.assertFalse(os.path.exists(os.path.join(tmp, "artifacts")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_train_require_gpu_without_torch_exits_2(self):
        tmp = tempfile.mkdtemp(prefix="live_smoke_notorch_")
        try:
            result = _run_hidden_torch(TRAIN, ["--require-gpu"], cwd=tmp)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("TRAIN_REQUIRE_GPU_FAILED", result.stdout)
            self.assertNotIn("DONE_OK", result.stdout)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_steps_bound_rejected_before_torch_import(self):
        for bad in ("0", "201", "500"):
            result = _run_hidden_torch(TRAIN, ["--steps", bad])
            self.assertEqual(result.returncode, 1, bad)
            self.assertIn("TRAIN_CONFIG_ERROR", result.stdout)


class TestTelemetrySchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = _load_module(TRAIN, "live_smoke_train_tiny")

    def test_make_event_keys_match_telemetry_emitter(self):
        from colab_adapter.telemetry import TelemetryEmitter

        tmp = tempfile.mkdtemp(prefix="live_smoke_schema_")
        try:
            emitter = TelemetryEmitter(ledger_path=Path(tmp) / "ledger.jsonl",
                                       run_id="run-x", project_id="live-smoke")
            reference = emitter.emit("metric", epoch=1, step=1, total_steps=5,
                                     metrics={"loss": 0.5})
            event = self.train.make_event("metric", "run-x", "live-smoke", epoch=1, step=1,
                                          total_steps=5, metrics={"loss": 0.5})
            self.assertEqual(set(event.keys()), set(reference.keys()))
            self.assertEqual(set(event.keys()), TELEMETRY_KEYS)
            self.assertEqual(event["source"], reference["source"])
            self.assertEqual(event["schema_version"], reference["schema_version"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_emit_writes_parseable_ledger_lines_with_monitor_keys(self):
        tmp = tempfile.mkdtemp(prefix="live_smoke_ledger_")
        try:
            ledger = os.path.join(tmp, "telemetry_events.jsonl")
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.train.emit(self.train.make_event("run_started", "run-y", "live-smoke",
                                                      total_steps=3), ledger)
                self.train.emit(self.train.make_event("metric", "run-y", "live-smoke", epoch=1,
                                                      step=1, total_steps=3,
                                                      metrics={"loss": 0.25}), ledger)
            stdout_lines = [l for l in buf.getvalue().splitlines() if l.startswith("TELEMETRY ")]
            self.assertEqual(len(stdout_lines), 2)
            with open(ledger, "r", encoding="utf-8") as handle:
                ledger_lines = [l for l in handle.read().splitlines() if l.strip()]
            self.assertEqual(len(ledger_lines), 2)
            for line in ledger_lines:
                record = json.loads(line)
                self.assertTrue(TELEMETRY_KEYS.issubset(record.keys()))
                self.assertIn(record["event_type"], {"run_started", "metric"})
            metric = json.loads(ledger_lines[1])
            self.assertEqual(metric["step"], 1)
            self.assertEqual(metric["epoch"], 1)
            self.assertIn("loss", metric["metrics"])
            self.assertEqual(os.stat(ledger).st_mode & 0o777, 0o600)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_sidecar_parser_accepts_sha256sum_format(self):
        tmp = tempfile.mkdtemp(prefix="live_smoke_sidecar_")
        try:
            sidecar = os.path.join(tmp, "model.pt.sha256")
            digest = "ab" * 32
            Path(sidecar).write_text("%s  model.pt\n" % digest, encoding="utf-8")
            self.assertEqual(self.train.read_sidecar(sidecar), (digest, "model.pt"))
            Path(sidecar).write_text("short  model.pt\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.train.read_sidecar(sidecar)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(_torch_available(), "PyTorch not installed locally; remote-only check")
class TestLocalTorchRun(unittest.TestCase):
    """Runs the scripts on CPU when torch is available locally. Never uses CUDA by requirement."""

    def test_probe_cpu_reports_no_cuda_marker(self):
        result = subprocess.run([PYTHON, str(PROBE)], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True,
                                env=_env_without_colab_ledger(), timeout=300)
        if "CUDA available: True" in result.stdout:
            self.skipTest("CUDA visible locally; CPU-only path not exercised")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("GPU_PROBE_NO_CUDA", result.stdout)

    def test_train_cpu_short_run_emits_markers_telemetry_and_checkpoint(self):
        tmp = tempfile.mkdtemp(prefix="live_smoke_cpu_run_")
        try:
            ledger = os.path.join(tmp, "ledger.jsonl")
            env = _env_without_colab_ledger()
            env["COLAB_TELEMETRY_LEDGER"] = ledger
            out_dir = os.path.join(tmp, "artifacts", "checkpoints")
            result = subprocess.run(
                [PYTHON, str(TRAIN), "--steps", "10", "--out-dir", out_dir],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                cwd=tmp, env=env, timeout=300,
            )
            if "CUDA available: True" in result.stdout:
                self.skipTest("CUDA visible locally; CPU-only path not exercised")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("DONE_OK", result.stdout)
            telemetry = [json.loads(l[len("TELEMETRY "):]) for l in result.stdout.splitlines()
                         if l.startswith("TELEMETRY ")]
            metric_steps = [e["step"] for e in telemetry if e["event_type"] == "metric"]
            self.assertEqual(metric_steps, list(range(1, 11)))
            self.assertTrue(all(TELEMETRY_KEYS.issubset(e.keys()) for e in telemetry))
            with open(ledger, "r", encoding="utf-8") as handle:
                self.assertEqual(len([l for l in handle if l.strip()]), len(telemetry))
            ckpt = os.path.join(out_dir, "model.pt")
            sidecar = ckpt + ".sha256"
            digest, name = self.load_sidecar(sidecar)
            self.assertEqual(name, "model.pt")
            self.assertEqual(_sha256(ckpt), digest)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @staticmethod
    def load_sidecar(path):
        parts = Path(path).read_text(encoding="utf-8").split()
        return parts[0], parts[1]


if __name__ == "__main__":
    unittest.main()
