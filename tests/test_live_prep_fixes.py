"""Fixes found while preparing the live integration campaign (offline)."""
import tempfile
import unittest
from pathlib import Path

from colab_adapter.storage.transfer import TransferJournal, safe_lock_filename


class LongLockNameTest(unittest.TestCase):
    def test_long_names_fit_filesystem_limits_and_stay_distinct(self):
        a = "artifact-google_drive-proj-run-" + "a" * 244
        b = "artifact-google_drive-proj-run-" + "a" * 243 + "b"
        sa, sb = safe_lock_filename(a), safe_lock_filename(b)
        self.assertLessEqual(len(sa) + len(".lock"), 255)
        self.assertNotEqual(sa, sb)
        self.assertEqual(sa, safe_lock_filename(a))  # stable across calls/processes
        self.assertEqual(safe_lock_filename("artifact-x-y"), "artifact-x-y")  # short names unchanged

    def test_journal_named_lock_works_for_a_244_byte_name(self):
        with tempfile.TemporaryDirectory() as td:
            j = TransferJournal(Path(td) / "journal" / "transfers.json")
            with j.acquire_named_lock("artifact-p-r-" + "n" * 244, timeout=2) as lock_path:
                self.assertTrue(Path(lock_path).exists())
                self.assertLessEqual(len(Path(lock_path).name), 255)


if __name__ == "__main__":
    unittest.main()
