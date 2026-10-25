"""Unit tests for automatic checkpoint publication and local retention policies.

Part of Colab V2 Cross-Device Workspaces & Google Drive (Issue #21).
"""
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from colab_adapter.checkpoint_policy import CheckpointPolicyConfig, CheckpointPolicyManager
from colab_adapter.errors import ArtifactValidationError


class TestCheckpointPolicy(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="colab_policy_test_")
        self.addCleanup(lambda: shutil.rmtree(self.test_dir, ignore_errors=True))
        self.checkpoints_dir = Path(self.test_dir) / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.config = CheckpointPolicyConfig(
            publish_on_complete=True,
            publish_on_new_checkpoint=True,
            minimum_interval_seconds=600,  # 10 minutes
            keep_last=2,
            cloud_enabled=True,
        )
        self.manager = CheckpointPolicyManager(self.config)

    def test_interval_throttling_rejects_premature_checkpoint(self):
        """Checkpoints arriving before minimum_interval_seconds are throttled."""
        t0 = 1000.0
        # First checkpoint at t0 is accepted
        self.assertTrue(self.manager.should_publish(step=100, is_final=False, now=t0))
        self.manager.record_publication(step=100, published_time=t0)

        # Checkpoint arriving 300 seconds later (interval is 600) must be throttled
        t1 = t0 + 300.0
        self.assertFalse(self.manager.should_publish(step=200, is_final=False, now=t1))

        # Checkpoint arriving 601 seconds later is permitted
        t2 = t0 + 601.0
        self.assertTrue(self.manager.should_publish(step=300, is_final=False, now=t2))

    def test_publish_on_complete_always_permitted(self):
        """Final checkpoint on complete is always published even within interval."""
        t0 = 1000.0
        self.manager.record_publication(step=100, published_time=t0)

        # 5 seconds later, final checkpoint arrives -> must be allowed
        self.assertTrue(self.manager.should_publish(step=105, is_final=True, now=t0 + 5.0))

    def test_prune_retention_keeps_last_n_and_never_deletes_last_checkpoint(self):
        """Pruning deletes older checkpoints beyond keep_last, but always preserves at least one."""
        # Create 4 fake checkpoint files with distinct mtimes
        f1 = self.checkpoints_dir / "step_100.pt"
        f2 = self.checkpoints_dir / "step_200.pt"
        f3 = self.checkpoints_dir / "step_300.pt"
        f4 = self.checkpoints_dir / "step_400.pt"

        f1.write_bytes(b"c1")
        f2.write_bytes(b"c2")
        f3.write_bytes(b"c3")
        f4.write_bytes(b"c4")

        # Set increasing mtime
        now = time.time()
        import os
        os.utime(f1, (now - 400, now - 400))
        os.utime(f2, (now - 300, now - 300))
        os.utime(f3, (now - 200, now - 200))
        os.utime(f4, (now - 100, now - 100))

        # keep_last is 2 -> f4 and f3 are kept, f1 and f2 pruned
        pruned = self.manager.prune_local_retention(self.checkpoints_dir)
        self.assertEqual(len(pruned), 2)
        self.assertIn(f1, pruned)
        self.assertIn(f2, pruned)
        self.assertTrue(f3.exists())
        self.assertTrue(f4.exists())
        self.assertFalse(f1.exists())
        self.assertFalse(f2.exists())

    def test_pruning_with_keep_last_zero_still_preserves_one_checkpoint(self):
        """Safety contract: keep_last=0 must still preserve at least 1 recoverable checkpoint."""
        self.manager.config.keep_last = 0
        f1 = self.checkpoints_dir / "only_checkpoint.pt"
        f1.write_bytes(b"RECOVERABLE_CHECKPOINT")

        pruned = self.manager.prune_local_retention(self.checkpoints_dir)
        self.assertEqual(len(pruned), 0)
        self.assertTrue(f1.exists(), "Last valid checkpoint must never be deleted!")

    def test_validation_rejects_missing_or_corrupt_files(self):
        """Nonexistent checkpoint raises ArtifactValidationError."""
        with self.assertRaises(ArtifactValidationError):
            self.manager.validate_for_publication(self.checkpoints_dir / "nonexistent.pt")


if __name__ == "__main__":
    unittest.main()
