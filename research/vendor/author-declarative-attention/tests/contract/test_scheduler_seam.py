"""The bench observer wraps Scheduler.schedule, so its signature is a contract.

vLLM 0.29.0 calls ``self.scheduler.schedule(self._should_throttle_prefills())`` from
EngineCore, one positional argument. The wrapper installed by
``VllmEngineAdapter.run_arm`` must pass any arguments through unchanged, and the unit
test double must present the same signature as the real scheduler.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.contract


def test_schedule_takes_throttle_prefills() -> None:
    vllm = pytest.importorskip("vllm")
    from vllm.v1.core.sched.scheduler import Scheduler

    params = list(inspect.signature(Scheduler.schedule).parameters)
    assert params == ["self", "throttle_prefills"], params
    del vllm


def test_engine_core_calls_schedule_with_one_argument() -> None:
    pytest.importorskip("vllm")
    from vllm.v1.engine import core

    src = inspect.getsource(core)
    assert "self.scheduler.schedule(self._should_throttle_prefills())" in src
