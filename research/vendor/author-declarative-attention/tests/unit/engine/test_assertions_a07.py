"""A07's second conjunct: no compacted layer runs with an active sliding window.

The first conjunct reads the KV-cache spec. This one reads `layer.impl.sliding_window`,
which is what the kernel actually masks on. `docs/DESIGN.md` section 6 row A07 promises
both. These tests pin the two properties that matter for a BOOT assertion: it fails on a
real window, and it never fails on the shapes the pinned vLLM 0.29.0 backends produce for
an unwindowed layer, nor on anything it cannot resolve.
"""

from __future__ import annotations

from typing import Any

import pytest

from dattn.engine.assertions import _a07_layer_windows, _window_is_off


class _Impl:
    def __init__(self, sliding_window: Any) -> None:
        self.sliding_window = sliding_window


class _Layer:
    def __init__(self, sliding_window: Any) -> None:
        self.impl = _Impl(sliding_window)


class _Compilation:
    def __init__(self, ctx: Any) -> None:
        self.static_forward_context = ctx


class _Cfg:
    def __init__(self, ctx: Any) -> None:
        self.compilation_config = _Compilation(ctx)


class _Rec:
    """Enough of `_Ctx` to drive one assertion without importing a vLLM config."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.current = "A07"
        self.resolved: dict[str, Any] = {}
        self.failures: list[str] = []

    def record(self, key: str, value: Any) -> Any:
        self.resolved[key] = value
        return value

    def want(self, ok: bool, message: str) -> None:
        if not ok:
            self.failures.append(f"{self.current}: {message}")


def _groups(names: list[str], *, is_dattn: bool = True) -> list[dict[str, Any]]:
    return [{"kv_cache_group_id": 0, "is_dattn": is_dattn, "layer_names": names}]


# `triton_attn.py:472`, `flashinfer.py:1758` and `flash_attn.py:858` all build this pair
# for a layer with no window in the pinned 0.29.0.
@pytest.mark.parametrize("window", [None, (-1, -1), [-1, -1], -1])
def test_unwindowed_encodings_never_fail(window: Any) -> None:
    c = _Rec(_Cfg({"layer.0": _Layer(window)}))
    _a07_layer_windows(c, _groups(["layer.0"]))
    assert c.failures == []
    assert _window_is_off(window)


@pytest.mark.parametrize("window", [(511, 0), (1023, 1023), 512, [4095, 0]])
def test_an_active_window_on_a_compacted_layer_fails(window: Any) -> None:
    c = _Rec(_Cfg({"layer.0": _Layer(window)}))
    _a07_layer_windows(c, _groups(["layer.0"]))
    assert len(c.failures) == 1
    assert "the PACKED key index" in c.failures[0]
    assert "layer.0" in c.failures[0]
    assert not _window_is_off(window)


def test_a_windowed_layer_outside_the_da_binding_is_ignored() -> None:
    c = _Rec(_Cfg({"layer.0": _Layer((511, 0))}))
    _a07_layer_windows(c, _groups(["layer.0"], is_dattn=False))
    assert c.failures == []


@pytest.mark.parametrize(
    "ctx",
    [None, {}, {"layer.0": None}, {"layer.0": object()}, {"other": _Layer((511, 0))}],
)
def test_unresolvable_layers_are_recorded_and_never_refused(ctx: Any) -> None:
    """A boot assertion that wrongly refuses costs a whole run. Positive evidence only."""
    c = _Rec(_Cfg(ctx))
    _a07_layer_windows(c, _groups(["layer.0"]))
    assert c.failures == []
    assert "a07_layer_windows" in c.resolved


def test_resolved_windows_reach_the_report() -> None:
    c = _Rec(_Cfg({"a": _Layer((-1, -1)), "b": _Layer(None)}))
    _a07_layer_windows(c, _groups(["a", "b"]))
    assert c.resolved["a07_layer_windows"] == {"a": [-1, -1], "b": None}
    assert c.resolved["a07_layers_unresolved"] == []


def test_an_unrecognised_encoding_is_not_silently_treated_as_off() -> None:
    assert not _window_is_off("512")
    assert not _window_is_off(())
    assert not _window_is_off(object())
