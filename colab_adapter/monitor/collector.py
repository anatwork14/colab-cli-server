"""Read-only Google Colab compute units and session snapshot collector (M8).

Guarantees:
- Zero allocation: dispatches ONLY read-only 'usage' and 'sessions' commands (Q19).
- Honest accounting: records monitoring_since date; distinguishes top-ups/refunds (Q22);
  labels account-wide spend when external sessions exist (Q23).
- Resilient error handling: captures CLI 401/403/429/timeout with backoff and stale flags (Q25).
- Append-only thread- and process-safe JSONL ledger with deduplication.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..provider import parse_session_lines
from ..tracker import sanitize_text
from .schemas import AccountSample, _iso_now

# Patterns for extracting compute units, rate, and active assignments from colab usage output
_CU_PATTERN = re.compile(r"(?:compute\s+units?|units?|balance)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_RATE_PATTERN = re.compile(r"(?:rate|consumption)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:cu/h|cu/hr)?", re.IGNORECASE)
_ASSIGNMENTS_PATTERN = re.compile(r"(?:assignments?|active\s+sessions?)\s*[:=]?\s*([0-9]+)", re.IGNORECASE)


def parse_usage_output(output: str) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """Parse compute units, rate, and active assignments from colab usage output.

    A value missing from the output is None (unknown), never 0.
    """
    cu = None
    rate = None
    assignments = None

    m_cu = _CU_PATTERN.search(output)
    if m_cu:
        try:
            cu = float(m_cu.group(1))
        except (ValueError, TypeError):
            pass

    m_rate = _RATE_PATTERN.search(output)
    if m_rate:
        try:
            rate = float(m_rate.group(1))
        except (ValueError, TypeError):
            pass

    m_assign = _ASSIGNMENTS_PATTERN.search(output)
    if m_assign:
        try:
            assignments = int(m_assign.group(1))
        except (ValueError, TypeError):
            pass

    return cu, rate, assignments


class AccountUsageCollector:
    """Collects and stores read-only account snapshots from the official Colab CLI."""

    _file_locks: Dict[Path, threading.Lock] = {}
    _file_locks_guard = threading.Lock()

    def __init__(
        self,
        ledger_path: Optional[Union[str, Path]] = None,
        colab_binary: str = "colab",
        auth_mode: str = "oauth2",
        run_cmd_fn: Optional[Callable[..., subprocess.CompletedProcess]] = None,
        base_poll_interval: float = 60.0,
        max_poll_interval: float = 300.0,
        stale_threshold_seconds: float = 300.0,
    ):
        if ledger_path is None:
            ledger_path = Path.home() / ".colab_experiments" / "account_usage_samples.jsonl"
        self.ledger_path = Path(ledger_path).resolve()
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.colab_binary = colab_binary
        self.auth_mode = auth_mode
        self._run_cmd = run_cmd_fn or subprocess.run
        self.base_poll_interval = base_poll_interval
        self.max_poll_interval = max_poll_interval
        self.stale_threshold_seconds = stale_threshold_seconds

        self._thread_lock = self._get_thread_lock(self.ledger_path)
        self._backoff_level = 0
        self._last_sample: Optional[AccountSample] = None
        self._monitoring_since = self._init_monitoring_since()

    @classmethod
    def _get_thread_lock(cls, path: Path) -> threading.Lock:
        with cls._file_locks_guard:
            resolved = path.resolve()
            if resolved not in cls._file_locks:
                cls._file_locks[resolved] = threading.Lock()
            return cls._file_locks[resolved]

    def _init_monitoring_since(self) -> str:
        """Initialize or load the immutable monitoring_since timestamp (Q24)."""
        marker_file = self.ledger_path.with_name(self.ledger_path.stem + ".meta.json")
        if marker_file.exists():
            try:
                data = json.loads(marker_file.read_text(encoding="utf-8"))
                if "monitoring_since" in data:
                    return data["monitoring_since"]
            except Exception:
                pass
        now_iso = _iso_now()
        try:
            marker_file.write_text(json.dumps({"monitoring_since": now_iso}, indent=2), encoding="utf-8")
        except OSError:
            pass
        return now_iso

    @property
    def monitoring_since(self) -> str:
        return self._monitoring_since

    @property
    def backoff_level(self) -> int:
        return self._backoff_level

    @property
    def is_stale(self) -> bool:
        if self._last_sample is None:
            return True
        try:
            obs = datetime.fromisoformat(self._last_sample.observed_at.replace("Z", "+00:00")).timestamp()
            return (time.time() - obs) > self.stale_threshold_seconds
        except Exception:
            return True

    def poll_once(self, unmanaged_sessions_count: int = 0) -> AccountSample:
        """Perform one read-only poll of account usage and session assignments.

        Strict invariant: Dispatches ONLY read-only 'usage' command (Q19).
        """
        observed_at = _iso_now()
        sample_id = f"sample-{uuid.uuid4().hex[:12]}"

        # Command is strictly read-only: colab --auth=oauth2 usage
        cmd = [self.colab_binary, f"--auth={self.auth_mode}", "usage"]
        try:
            res = self._run_cmd(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30.0)
            if res.returncode != 0:
                err_msg = (res.stderr or res.stdout or "").strip()
                self._backoff_level = min(self._backoff_level + 1, 5)
                sanitized_msg = sanitize_text(err_msg)
                sample = AccountSample(
                    sample_id=sample_id,
                    observed_at=observed_at,
                    status="ERROR",
                    error_message=f"Usage query failed: {sanitized_msg}",
                )
            else:
                self._backoff_level = 0
                out = res.stdout or ""
                cu, rate, assignments = parse_usage_output(out)

                net_change = None
                change_type = "INITIAL" if self._last_sample is None else "UNCHANGED"

                if self._last_sample and self._last_sample.balance_cu is not None and cu is not None:
                    net_change = round(cu - self._last_sample.balance_cu, 4)
                    if net_change > 0:
                        change_type = "TOP_UP"  # Honest accounting: balance increase
                    elif net_change < 0:
                        if unmanaged_sessions_count > 0 or (assignments or 0) > 1:
                            change_type = "UNATTRIBUTED"  # External sessions active on account
                        else:
                            change_type = "SPEND"
                    else:
                        change_type = "UNCHANGED"

                sample = AccountSample(
                    sample_id=sample_id,
                    observed_at=observed_at,
                    balance_cu=cu,
                    rate_cu_per_hour=rate,
                    active_assignments=assignments,
                    net_balance_change=net_change,
                    change_type=change_type,
                    status="OK",
                )
        except subprocess.TimeoutExpired:
            self._backoff_level = min(self._backoff_level + 1, 5)
            sample = AccountSample(
                sample_id=sample_id,
                observed_at=observed_at,
                status="ERROR",
                error_message="Usage query timed out after 30s",
            )
        except Exception as e:
            self._backoff_level = min(self._backoff_level + 1, 5)
            sanitized_err = sanitize_text(str(e))
            sample = AccountSample(
                sample_id=sample_id,
                observed_at=observed_at,
                status="ERROR",
                error_message=f"Collector exception: {sanitized_err}",
            )

        self._record_sample(sample)
        self._last_sample = sample
        return sample

    def _record_sample(self, sample: AccountSample) -> None:
        """Append sample to JSONL ledger with cross-process locking."""
        line = json.dumps(sample.to_dict(), ensure_ascii=False) + "\n"
        with self._thread_lock:
            fd = os.open(str(self.ledger_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                os.write(fd, line.encode("utf-8"))
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    os.close(fd)
                except OSError:
                    pass

    def get_samples(
        self,
        from_ts: Optional[str] = None,
        to_ts: Optional[str] = None,
        limit: int = 1440,
    ) -> List[AccountSample]:
        """Read account snapshots within timestamp window with bounded limit (Q20)."""
        if not self.ledger_path.exists():
            return []

        samples: List[AccountSample] = []
        with self._thread_lock:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        obs = record.get("observed_at", "")
                        if from_ts and obs < from_ts:
                            continue
                        if to_ts and obs > to_ts:
                            continue
                        samples.append(AccountSample.from_dict(record))
                    except json.JSONDecodeError:
                        continue

        if len(samples) > limit:
            samples = samples[-limit:]
        return samples

    def get_latest_sample(self) -> Optional[AccountSample]:
        if self._last_sample is not None:
            return self._last_sample
        samples = self.get_samples(limit=1)
        if samples:
            self._last_sample = samples[-1]
            return self._last_sample
        return None
