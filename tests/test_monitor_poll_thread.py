"""Background account-usage polling owned by the monitor server (M8 defect: collector never started).

All collectors here are fakes: either a fake object recording poll_once calls, or an
AccountUsageCollector with an injected run_cmd_fn. The real `colab` binary is never executed.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from colab_adapter.monitor import poller as poller_mod
from colab_adapter.monitor.collector import AccountSample, AccountUsageCollector
from colab_adapter.monitor.poller import (
    POLL_MAX_BACKOFF_SECONDS,
    POLL_MIN_INTERVAL_SECONDS,
    MonitorPoller,
    poll_delay_seconds,
)
from colab_adapter.monitor.server import start_monitor_server
from colab_adapter.provider import ManagedSessionRegistry
from colab_adapter.telemetry import TelemetryEmitter

POLLER_THREAD_NAME = "colab-monitor-poller"


def _live_poller_threads():
    return [t for t in threading.enumerate() if t.name == POLLER_THREAD_NAME and t.is_alive()]


class FakeCollector:
    """Records poll_once calls; never touches a CLI."""

    def __init__(self, status="OK", raise_exc=None):
        self.calls = 0
        self.status = status
        self.raise_exc = raise_exc
        self.lock = threading.Lock()

    def poll_once(self, unmanaged_sessions_count=0):
        with self.lock:
            self.calls += 1
        if self.raise_exc is not None:
            raise self.raise_exc
        return AccountSample(sample_id="s", observed_at="2026-10-26T00:00:00+00:00", status=self.status)


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class PollDelayTests(unittest.TestCase):
    def test_success_uses_base_interval(self):
        self.assertEqual(poll_delay_seconds(60.0, 0), 60.0)

    def test_backoff_doubles_per_consecutive_failure(self):
        self.assertEqual(poll_delay_seconds(60.0, 1), 120.0)
        self.assertEqual(poll_delay_seconds(60.0, 2), 240.0)
        self.assertEqual(poll_delay_seconds(60.0, 3), 480.0)

    def test_backoff_is_capped_at_600_seconds(self):
        self.assertEqual(POLL_MAX_BACKOFF_SECONDS, 600.0)
        self.assertEqual(poll_delay_seconds(60.0, 4), 600.0)
        self.assertEqual(poll_delay_seconds(60.0, 50), 600.0)

    def test_cap_never_shrinks_an_interval_longer_than_600(self):
        self.assertEqual(poll_delay_seconds(900.0, 0), 900.0)
        self.assertEqual(poll_delay_seconds(900.0, 1), 900.0)

    def test_minimum_interval_is_15_seconds(self):
        self.assertEqual(POLL_MIN_INTERVAL_SECONDS, 15.0)


class MonitorPollerThreadTests(unittest.TestCase):
    def tearDown(self):
        # No thread leaks: every poller started by a test must be gone once the test ends.
        self.assertEqual(_live_poller_threads(), [])

    def test_polls_repeatedly_with_tiny_interval(self):
        fake = FakeCollector()
        p = MonitorPoller(fake, interval=0.01, min_interval=0.001).start()
        try:
            self.assertTrue(_wait_for(lambda: fake.calls >= 3))
        finally:
            p.stop()
        self.assertFalse(p.is_alive())

    def test_consecutive_failures_back_off_then_reset_on_success(self):
        fake = FakeCollector(status="ERROR")
        p = MonitorPoller(fake, interval=0.001, min_interval=0.001).start()
        try:
            self.assertTrue(_wait_for(lambda: p.consecutive_failures >= 3))
            fake.status = "OK"
            self.assertTrue(_wait_for(lambda: p.consecutive_failures == 0))
        finally:
            p.stop()

    def test_raising_collector_counts_as_failure_and_thread_survives(self):
        fake = FakeCollector(raise_exc=RuntimeError("boom"))
        p = MonitorPoller(fake, interval=0.001, min_interval=0.001).start()
        try:
            self.assertTrue(_wait_for(lambda: fake.calls >= 3))
            self.assertTrue(p.is_alive())
            self.assertGreaterEqual(p.consecutive_failures, 3)
        finally:
            p.stop()

    def test_stop_returns_within_one_second_for_long_interval(self):
        fake = FakeCollector()
        p = MonitorPoller(fake, interval=3600.0).start()
        started = time.monotonic()
        p.stop()
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertFalse(p.is_alive())
        self.assertEqual(fake.calls, 0)  # first poll waits one interval, so no CLI call at start

    def test_poller_only_dispatches_read_only_usage_command(self):
        calls = []

        def run_cmd(cmd, **kwargs):
            calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="Current balance: 10\n", stderr="")

        with tempfile.TemporaryDirectory(prefix="colab_poll_ro_") as tmp:
            collector = AccountUsageCollector(ledger_path=Path(tmp) / "ledger.jsonl", run_cmd_fn=run_cmd)
            p = MonitorPoller(collector, interval=0.01, min_interval=0.001).start()
            try:
                self.assertTrue(_wait_for(lambda: len(calls) >= 2))
            finally:
                p.stop()
        for cmd in calls:
            self.assertEqual(cmd[-1], "usage")
            self.assertEqual(len(cmd), 3)


class MonitorServerPollingTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="colab_poll_srv_"))
        self.token = "poll-test-token-0123456789"
        self.registry = ManagedSessionRegistry(self.tmp_dir / "sessions.json")
        self.telemetry = TelemetryEmitter(ledger_path=self.tmp_dir / "telemetry.jsonl", run_id="r-poll")
        self.server = None

    def tearDown(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        self.assertTrue(_wait_for(lambda: not _live_poller_threads(), timeout=2.0), "poller thread leaked")

    def _start(self, collector, **kwargs):
        self.server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            collector=collector,
            registry=self.registry,
            telemetry=self.telemetry,
            auth_token=self.token,
            **kwargs,
        )
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self.server

    def _get_json(self, path):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.server.server_port}{path}",
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_default_interval_starts_a_poller_clamped_to_fifteen_seconds(self):
        self._start(FakeCollector(), poll_interval=0.5)
        self.assertIsNotNone(self.server._poller)
        self.assertEqual(self.server._poller.interval, 15.0)
        self.assertEqual(len(_live_poller_threads()), 1)

    def test_default_poll_interval_is_sixty_seconds(self):
        self._start(FakeCollector())
        self.assertEqual(self.server._poller.interval, 60.0)

    def test_zero_and_none_disable_polling(self):
        for value in (0, 0.0, None):
            with self.subTest(poll_interval=value):
                self._start(FakeCollector(), poll_interval=value)
                self.assertIsNone(self.server._poller)
                self.assertEqual(_live_poller_threads(), [])
                self.server.shutdown()
                self.server.server_close()
                self.server = None

    def test_shutdown_stops_poller_within_one_second(self):
        self._start(FakeCollector(), poll_interval=3600.0)
        self.assertEqual(len(_live_poller_threads()), 1)
        started = time.monotonic()
        self.server.shutdown()
        self.server.server_close()
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(_live_poller_threads(), [])
        self.server = None

    def test_server_close_alone_stops_poller(self):
        self._start(FakeCollector(), poll_interval=3600.0)
        self.server.server_close()
        self.assertTrue(_wait_for(lambda: not _live_poller_threads(), timeout=1.0))
        self.server = None

    def test_poll_errors_do_not_kill_the_server(self):
        def run_cmd(cmd, **kwargs):
            raise OSError("cli unavailable")

        collector = AccountUsageCollector(ledger_path=self.tmp_dir / "ledger.jsonl", run_cmd_fn=run_cmd)
        with mock.patch.object(poller_mod, "POLL_MIN_INTERVAL_SECONDS", 0.001):
            self._start(collector, poll_interval=0.001)
            self.assertTrue(_wait_for(lambda: collector.backoff_level >= 2))
            health = self._get_json("/api/v1/health")
            self.assertTrue(self.server._poller.is_alive())
        self.assertEqual(health["status"], "ERROR")
        self.assertGreaterEqual(health["collector_backoff_level"], 2)

    def test_summary_reflects_successful_poll_with_fake_cli(self):
        def run_cmd(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd, 0, stdout="Current balance: 84.5\nUsage rate: 1.9\nActive assignments: 2\n", stderr=""
            )

        collector = AccountUsageCollector(ledger_path=self.tmp_dir / "ledger.jsonl", run_cmd_fn=run_cmd)
        with mock.patch.object(poller_mod, "POLL_MIN_INTERVAL_SECONDS", 0.001):
            self._start(collector, poll_interval=0.01)

            def summary_ready():
                return self._get_json("/api/v1/summary")["current_balance_cu"] is not None

            self.assertTrue(_wait_for(summary_ready, timeout=5.0))
        summary = self._get_json("/api/v1/summary")
        self.assertEqual(summary["current_balance_cu"], 84.5)
        self.assertEqual(summary["current_rate_cu_per_hour"], 1.9)
        self.assertEqual(summary["active_assignments"], 2)
        self.assertFalse(summary["is_stale"])
        self.assertIsNotNone(summary["last_sample_time"])


if __name__ == "__main__":
    unittest.main()
