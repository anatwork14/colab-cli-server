"""Offline mock test for E2E stateful training and verification."""
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from examples.verify_checkpoint import verify_checkpoint
from colab_adapter.provider import ApprovedJob, ColabGPUProvider, JobSpec


class TestE2EMockWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.base_dir = Path(self.temp_dir.name)

        # Create dummy script & data
        self.script_path = self.base_dir / "train.py"
        self.script_path.write_text("print('training')", encoding="utf-8")
        self.data_path = self.base_dir / "data.csv"
        self.data_path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

        self.artifacts_dir = self.base_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.provider = ColabGPUProvider(cli_executable="/usr/local/bin/colab")

    def test_mock_stateful_e2e_lifecycle_and_verification(self):
        spec = JobSpec(
            project_name="test-e2e",
            script_path=self.script_path,
            gpu="T4",
            input_files={str(self.data_path): "/content/data.csv"},
            expected_outputs=["/content/model_checkpoint.pt", "/content/status.json"],
            output_download_dir=self.artifacts_dir,
            timeout=300.0,
            stateful=True,
        )
        prep = self.provider.prepare_job(spec)
        approved = ApprovedJob(
            prepared_job=prep,
            approval_code=prep.approval_nonce,
            approved_by="tester",
            fingerprint=prep.fingerprint,
            approved_at="now",
        )

        # Mock download action by writing realistic checkpoint and status.json
        checkpoint_path = self.artifacts_dir / "model_checkpoint.pt"
        status_path = self.artifacts_dir / "status.json"

        def create_dummy_checkpoint():
            with zipfile.ZipFile(str(checkpoint_path), "w") as zf:
                zf.writestr("archive/data.pkl", b"dummy_pickle_data")
                zf.writestr("archive/version", b"3")
            import hashlib
            h = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            manifest = {
                "status": "SUCCESS",
                "model_file": "model_checkpoint.pt",
                "sha256": h,
                "final_loss": 0.042,
                "device": "cuda:0",
                "gpu_name": "Tesla T4",
            }
            status_path.write_text(json.dumps(manifest), encoding="utf-8")

        with mock.patch.object(self.provider, "check_availability", return_value={"authenticated": True, "active_sessions": []}):
            with mock.patch.object(self.provider, "_run_cli") as mock_cli:
                def side_effect(cmd_args, *args, **kwargs):
                    cmd = cmd_args[0]
                    if cmd == "new":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="Session created", stderr="")
                    elif cmd == "upload":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="Uploaded", stderr="")
                    elif cmd == "exec":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="Training ok\nLoss: 0.042", stderr="")
                    elif cmd == "download":
                        create_dummy_checkpoint()
                        return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="Downloaded", stderr="")
                    elif cmd == "stop":
                        return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="Stopped", stderr="")
                    return subprocess.CompletedProcess(args=cmd_args, returncode=0, stdout="", stderr="")

                mock_cli.side_effect = side_effect
                result = self.provider.submit_job(approved)

        self.assertTrue(result.success)
        self.assertEqual(len(result.output_files), 2)
        # Verify checkpoint
        ok = verify_checkpoint(self.artifacts_dir)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
