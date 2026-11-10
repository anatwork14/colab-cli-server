"""The tokenizer factory the FSM decodes with, pinned to vLLM 0.29.0.

This file exists because its absence cost a live server. `DattnLogitsProcessor._text`
needs the decoded text of every sampled token: the FSM scans the tail of the response for
`<focus>`, `<local>` and `</...>` (DESIGN.md 1.1), and it cannot do that from token ids.
The import it used, `vllm.transformers_utils.tokenizer`, is where that factory lived up
to 0.2x. In 0.29.0 it is `vllm.tokenizers.registry`, and the module is gone.

Nothing caught it. Every GPU test drives the mask directly with `forced_mask`, which never
enters the FSM, so the first code path to touch a tokenizer was the first real request to
a real server, where the `ModuleNotFoundError` came out of `refresh_metadata` inside
`execute_model` and took the whole EngineCore down with `EngineDeadError`. Eight rollouts
came back as HTTP 500.

CHARTER.md's rule is "every vLLM internal touched gets a contract test pinned to the vLLM
version". This is that test for this internal.
"""

from __future__ import annotations

import importlib

import pytest

from dattn.engine.processor import TOKENIZER_FACTORY, _tokenizer

pytestmark = pytest.mark.contract


def test_the_factory_dattn_names_exists_where_it_says() -> None:
    module_name, attr = TOKENIZER_FACTORY
    module = importlib.import_module(module_name)
    assert callable(getattr(module, attr))


def test_the_pre_0_29_location_is_gone_so_the_old_import_cannot_come_back() -> None:
    """A regression guard with a memory: naming the dead module is the whole point."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("vllm.transformers_utils.tokenizer")


def test_the_factory_takes_a_model_config_and_honours_skip_tokenizer_init() -> None:
    """The two facts `_tokenizer` depends on, read off the installed signature."""
    module_name, attr = TOKENIZER_FACTORY
    fn = getattr(importlib.import_module(module_name), attr)
    params = list(__import__("inspect").signature(fn).parameters)
    assert params[0] == "model_config"
    src = __import__("inspect").getsource(fn)
    assert "skip_tokenizer_init" in src and "return None" in src


def test_a_tokenizerless_engine_is_refused_by_name() -> None:
    """`skip_tokenizer_init` returns None, and a None tokenizer must not reach the FSM."""
    from dattn.contracts import DattnInvariantError

    class _ModelConfig:
        skip_tokenizer_init = True
        hf_config = None

    class _Config:
        model_config = _ModelConfig()

    with pytest.raises(DattnInvariantError, match="skip_tokenizer_init"):
        _tokenizer(_Config())


def test_no_config_means_no_tokenizer_rather_than_a_crash() -> None:
    assert _tokenizer(None) is None
