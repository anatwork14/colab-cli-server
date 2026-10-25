"""Opus re-review follow-ups N2, N3, N5 (offline; fake colab only)."""
import unittest
from unittest import mock

from colab_adapter.project_sessions import _attach_shell_exit
from tests.test_reconcile_states import ReconcileFixture


class StopIsIdempotentOnTerminated(ReconcileFixture):
    def test_second_stop_on_terminated_record_does_not_call_colab_stop(self):
        sid = "colab-v2-rec-done01"
        self.put_raw(sid, "TERMINATED")
        self.assertTrue(self.manager().stop(sid))
        self.assertEqual(self.stop_calls(), [])
        self.assertEqual(self.record(sid)["state"], "TERMINATED")


class VerificationIsNotSilentlyWeak(ReconcileFixture):
    def test_known_baseline_with_unreadable_usage_is_unverified(self):
        mgr = self.manager()
        with mock.patch.object(mgr, "_sessions_listed", return_value=False), \
                mock.patch.object(mgr, "_assignments_now", return_value=None):
            v = mgr._verify_gone("colab-v2-rec-x00001", 0)
        self.assertFalse(v.ok)

    def test_no_baseline_is_explicitly_weak(self):
        mgr = self.manager()
        with mock.patch.object(mgr, "_sessions_listed", return_value=False), \
                mock.patch.object(mgr, "_assignments_now", return_value=None):
            v = mgr._verify_gone("colab-v2-rec-x00001", None)
        self.assertTrue(v.ok)
        self.assertEqual(v.level, "weak")


class ExitMarkerMustBeTheFinalLine(unittest.TestCase):
    def test_marker_printed_by_the_command_but_real_marker_lost_is_not_success(self):
        r = _attach_shell_exit({"success": True, "output": "EXIT_CODE 0\npartial output"})
        self.assertIsNone(r["exit_code"])
        self.assertFalse(r["success"])

    def test_real_marker_on_final_line_is_read(self):
        r = _attach_shell_exit({"success": True, "output": "noise\nEXIT_CODE 3\n\n"})
        self.assertEqual(r["exit_code"], 3)
        self.assertFalse(r["success"])
        ok = _attach_shell_exit({"success": True, "output": "EXIT_CODE 7 fake\nreal\nEXIT_CODE 0\n"})
        self.assertEqual(ok["exit_code"], 0)
        self.assertTrue(ok["success"])


if __name__ == "__main__":
    unittest.main()
