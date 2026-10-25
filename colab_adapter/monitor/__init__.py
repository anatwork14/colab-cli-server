"""Colab Project Workspaces Monitoring Dashboard and Collector (Milestone M8).

Provides:
- Account compute-unit snapshot collector (`AccountUsageCollector`).
- Append-only JSONL ledgers for usage snapshots and training telemetry.
- Honest accounting: distinct top-up detection, unattributed usage labeling.
- Authenticated, read-only loopback HTTP server with REST, SSE streams, and CSV export.
- Embedded responsive monitoring dashboard UI.
"""
from .collector import (
    AccountSample,
    AccountUsageCollector,
    parse_usage_output,
)
from .schemas import (
    HealthStatus,
    MonitoringSummary,
    RunSummary,
)
from .server import (
    MonitorServer,
    start_monitor_server,
)

__all__ = [
    "AccountSample",
    "AccountUsageCollector",
    "parse_usage_output",
    "HealthStatus",
    "MonitoringSummary",
    "RunSummary",
    "MonitorServer",
    "start_monitor_server",
]
