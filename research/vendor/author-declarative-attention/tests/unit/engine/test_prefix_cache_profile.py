"""Prefix caching must match the selected profile in either direction."""

from types import SimpleNamespace as NS

import pytest

from dattn.engine.assertions import _a11, _Ctx


@pytest.mark.parametrize("expected", [False, True])
@pytest.mark.parametrize("actual", [False, True])
def test_prefix_cache_binding_matches_profile(expected, actual):
    c = _Ctx(
        NS(cache_config=NS(enable_prefix_caching=actual)), None, NS(engine_args={"enable_prefix_caching": expected})
    )
    c.current = "A11"
    _a11(c)
    assert bool(c.report.failures) is (actual != expected)
    assert c.report.resolved["enable_prefix_caching"] is actual


def test_placeholder_refuses_boot_but_explicit_probe_override_can_resolve():
    from dataclasses import replace

    from dattn.contracts import DattnError
    from dattn.engine.launch import engine_kwargs
    from dattn.engine.profile import load_profile

    profile = load_profile("google/gemma-4-31B-it", use="walltime")
    profile = replace(profile, engine_args=dict(profile.engine_args, kv_cache_memory_bytes="TO BE PINNED FROM PROBE"))
    with pytest.raises(DattnError, match="TO BE PINNED FROM PROBE"):
        engine_kwargs(profile)
    assert engine_kwargs(profile, kv_cache_memory_bytes=None)["kv_cache_memory_bytes"] is None
