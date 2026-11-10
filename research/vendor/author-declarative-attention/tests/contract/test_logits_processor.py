"""The V1 logits-processor surface, pinned to vLLM 0.29.0.

`DattnLogitsProcessor` is where every vLLM-facing assumption lives (DESIGN.md section
7, WP5), so every one of those assumptions is pinned here: the ABC, the shape of
`AddedRequest`, the removed/added/moved contract, and the two reasons the FQCN route is
the only acceptable registration.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from dattn.contracts import INF, DattnRequest, Mask, Mode, PromptLayout
from dattn.engine import processor as proc
from dattn.engine.state import DattnState

pytestmark = pytest.mark.contract

LAYOUT = PromptLayout(n_prompt_tokens=64, question_start=48, segments=((16, 32),), sink=16)


def test_the_abc_has_the_three_methods_dattn_implements() -> None:
    from vllm.v1.sample.logits_processor import LogitsProcessor

    assert issubclass(proc.DattnLogitsProcessor, LogitsProcessor)
    for name in ("apply", "is_argmax_invariant", "update_state"):
        assert getattr(LogitsProcessor, name).__isabstractmethod__ is True
    init = inspect.signature(LogitsProcessor.__init__)
    assert list(init.parameters) == ["self", "vllm_config", "device", "is_pin_memory"]


def test_dattns_init_matches_the_abcs_positional_order() -> None:
    """vLLM constructs it as `cls(vllm_config, device, is_pin_memory)`."""
    sig = inspect.signature(proc.DattnLogitsProcessor.__init__)
    assert list(sig.parameters)[:4] == ["self", "vllm_config", "device", "is_pin_memory"]


def test_added_request_is_a_four_tuple_of_index_params_prompt_output() -> None:
    """`N:v1/sample/logits_processor/interface.py:29`. There is no request id in it."""
    import typing

    from vllm.v1.sample.logits_processor.interface import AddedRequest

    args = typing.get_args(AddedRequest)
    assert len(args) == 4
    assert args[0] is int
    assert args[3] == list[int]


def test_batch_update_carries_removed_added_moved_and_a_batch_size() -> None:
    import dataclasses

    from vllm.v1.sample.logits_processor.interface import BatchUpdate

    names = [f.name for f in dataclasses.fields(BatchUpdate)]
    assert names == ["batch_size", "removed", "added", "moved"]
    assert BatchUpdate.__dataclass_params__.frozen is True


def test_move_directionality_has_exactly_the_two_cases_dattn_handles() -> None:
    from vllm.v1.sample.logits_processor.interface import MoveDirectionality

    assert {m.name for m in MoveDirectionality} == {"UNIDIRECTIONAL", "SWAP"}


def test_the_documented_order_is_removed_added_moved() -> None:
    from vllm.v1.sample.logits_processor import interface

    doc = inspect.getsource(interface.BatchUpdate)
    assert "removed, added, moved" in doc


def test_a_real_batch_update_replays_through_the_real_processor() -> None:
    """Behaviour, not shape: add two, swap them, remove one, with vLLM's own types."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.sample.logits_processor.interface import BatchUpdate, MoveDirectionality

    lp = proc.DattnLogitsProcessor()
    lp.state = DattnState()
    lp.state.configure(4, torch.device("cpu"), pin_memory=False)

    def params(mask: Mask, req_id: str) -> SamplingParams:
        payload = DattnRequest(layout=LAYOUT, forced_mask=mask).to_extra_args()
        payload["req_id"] = req_id
        return SamplingParams(extra_args={"dattn": payload})

    mask_a = Mask.of((0, 16), (48, INF))
    mask_b = Mask.of((0, 8), (56, INF))
    lp.update_state(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=((0, params(mask_a, "a"), [0] * 64, []), (1, params(mask_b, "b"), [0] * 64, [])),
            moved=(),
        )
    )
    assert lp.state._intervals_host[0, :2].tolist() == [[0, 16], [48, INF]]
    assert lp.state._intervals_host[1, :2].tolist() == [[0, 8], [56, INF]]

    lp.update_state(BatchUpdate(batch_size=2, removed=(), added=(), moved=((0, 1, MoveDirectionality.SWAP),)))
    assert lp.state.rows[0].req_id == "b" and lp.state.rows[1].req_id == "a"
    assert lp.state._intervals_host[0, :2].tolist() == [[0, 8], [56, INF]]
    assert lp.state._intervals_host[1, :2].tolist() == [[0, 16], [48, INF]]

    lp.update_state(BatchUpdate(batch_size=1, removed=(1,), added=(), moved=()))
    assert lp.state.rows[1] is None
    assert lp.state._intervals_host[1, 0].tolist() == [0, INF]
    assert int(lp.state._counts_host[1]) == 1


def test_sampling_params_carries_extra_args_through_untouched() -> None:
    from vllm.sampling_params import SamplingParams

    payload = DattnRequest(layout=LAYOUT, forced_mask=Mask.of((0, 16), (48, INF))).to_extra_args()
    sp = SamplingParams(extra_args={"dattn": payload})
    back = DattnRequest.from_extra_args(sp.extra_args["dattn"])
    assert back.layout == LAYOUT
    assert back.forced_mask == Mask.of((0, 16), (48, INF))
    assert Mode.GLOBAL == 0


def test_the_fqcn_route_is_the_only_one_that_sets_need_output_token_ids() -> None:
    """`N:v1/worker/gpu_model_runner.py:739-742,781`: computed from model_config alone."""
    from vllm.v1.worker import gpu_model_runner

    src = inspect.getsource(gpu_model_runner.GPUModelRunner.__init__)
    assert "logits_processors = model_config.logits_processors" in src
    assert "logitsprocs_need_output_token_ids=bool(custom_logitsprocs)" in src
    # The flag is set from `custom_logitsprocs`, which is `model_config.logits_processors`
    # and nothing else. Installed entry points do not reach it.
    head = src.split("custom_logitsprocs: Sequence")[1].split("\n\n")[0]
    assert "entry_points" not in head


def test_the_mrv2_demotion_counts_installed_entry_points() -> None:
    """RC C5: shipping a vllm.logits_processors entry point would pick the runner for us."""
    from vllm.config import vllm as vllm_config_mod

    src = inspect.getsource(vllm_config_mod)
    assert 'entry_points(group="vllm.logits_processors")' in src
    assert "custom logits processors" in src
    assert "VLLM_USE_V2_MODEL_RUNNER" in src


def test_dattn_ships_no_logits_processor_entry_point() -> None:
    from importlib.metadata import entry_points

    eps = list(entry_points(group="vllm.logits_processors"))
    assert eps == [], f"a vllm.logits_processors entry point is installed: {eps}"


def test_dattn_does_ship_the_general_plugin_entry_point() -> None:
    from importlib.metadata import entry_points

    values = {ep.value for ep in entry_points(group="vllm.general_plugins")}
    assert "dattn.engine.plugin:install_plugin" in values


def test_the_logits_processor_fqcn_resolves_the_way_vllm_resolves_it() -> None:
    """`_load_logitsprocs_by_fqcns` splits on ':' and requires a LogitsProcessor subclass."""
    import importlib

    from vllm.v1.sample.logits_processor import LogitsProcessor

    from dattn.engine.plugin import LOGITS_PROCESSOR_FQCN

    module_path, qualname = LOGITS_PROCESSOR_FQCN.split(":")
    obj = importlib.import_module(module_path)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    assert isinstance(obj, type) and issubclass(obj, LogitsProcessor)
    assert obj is proc.DattnLogitsProcessor


def test_the_worker_extension_names_do_not_collide_with_the_0_29_0_worker() -> None:
    """`worker_extension_cls` is injected as a BASE of the worker class, and vLLM asserts
    that no attribute of it already exists on the worker
    (`N:v1/worker/worker_base.py:290-299`). A collision is a hard boot failure, so the
    `dattn_` prefix is a contract, not a style choice."""
    from vllm.v1.worker.gpu_worker import Worker

    from dattn.engine.worker_ext import DattnWorkerExtension

    for attr in dir(DattnWorkerExtension):
        if attr.startswith("__"):
            continue
        assert not hasattr(Worker, attr), f"{attr} already exists on vLLM's Worker"
