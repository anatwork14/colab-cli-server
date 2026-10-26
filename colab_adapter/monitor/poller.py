"""Bounded background poller for the account usage collector (M8).

The monitor server owns one daemon thread that calls ``AccountUsageCollector.poll_once()``.
``poll_once`` dispatches only the read-only ``colab --auth=oauth2 usage`` command. The thread:

- waits one interval before its first poll (no CLI call at server start);
- backs off exponentially on consecutive failures (ERROR samples or exceptions), capped at 600 s;
- waits on a ``threading.Event`` so ``stop()`` (called by ``MonitorServer.shutdown``/``server_close``)
  wakes it at once; no lock is held while waiting;
- never lets an exception escape into the HTTP server.
"""
from __future__ import annotations

import threading
from typing import Any, Optional

POLL_MIN_INTERVAL_SECONDS = 15.0
POLL_MAX_BACKOFF_SECONDS = 600.0
POLLER_THREAD_NAME = "colab-monitor-poller"


def clamp_poll_interval(poll_interval: Optional[float]) -> Optional[float]:
    """Return the effective interval, or None when polling is disabled (None or <= 0)."""
    if poll_interval is None:
        return None
    value = float(poll_interval)
    if value <= 0:
        return None
    return max(value, POLL_MIN_INTERVAL_SECONDS)


def poll_delay_seconds(interval: float, consecutive_failures: int) -> float:
    """Delay before the next poll: the interval after success, doubled per failure up to the cap.

    The cap is 600 s, or the interval itself when the operator asked for something longer.
    """
    if consecutive_failures <= 0:
        return float(interval)
    cap = max(float(interval), POLL_MAX_BACKOFF_SECONDS)
    exponent = min(consecutive_failures, 32)
    return min(float(interval) * (2 ** exponent), cap)


class MonitorPoller:
    """Daemon thread that polls an account usage collector at a bounded, backed-off interval."""

    def __init__(
        self,
        collector: Any,
        interval: float,
        min_interval: Optional[float] = None,
    ):
        floor = POLL_MIN_INTERVAL_SECONDS if min_interval is None else float(min_interval)
        self.collector = collector
        self.interval = max(float(interval), floor)
        self.consecutive_failures = 0
        self.poll_count = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=POLLER_THREAD_NAME, daemon=True)

    def start(self) -> "MonitorPoller":
        self._thread.start()
        return self

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self, timeout: float = 1.0) -> None:
        """Signal the thread to exit and wait briefly. Idempotent; never raises."""
        self._stop_event.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout)

    def _poll_succeeded(self) -> bool:
        try:
            sample = self.collector.poll_once()
        except Exception:
            return False
        status = getattr(sample, "status", "OK")
        return status == "OK"

    def _run(self) -> None:
        while True:
            delay = poll_delay_seconds(self.interval, self.consecutive_failures)
            if self._stop_event.wait(delay):
                return
            try:
                ok = self._poll_succeeded()
            except Exception:
                ok = False
            self.poll_count += 1
            self.consecutive_failures = 0 if ok else self.consecutive_failures + 1
            if self._stop_event.is_set():
                return
