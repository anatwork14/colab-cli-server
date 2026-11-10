"""CPU safety boundaries for the optional prefix profile."""

from types import SimpleNamespace

import pytest
import torch

from dattn.contracts import DattnInvariantError, DattnRequest
from dattn.engine import processor
from dattn.engine.prefix import prompt_only, refuse_preemption
from dattn.engine.profile import profile_row
from dattn.engine.state import DattnState


@pytest.mark.parametrize("model", ["google/gemma-4-31B-it", "Qwen/Qwen3.6-27B"])
def test_profile_diff_is_only_caching_and_name(model):
    plain, prefix = profile_row(model), profile_row(model, use="parity_prefix")
    prefix["use"] = "parity"
    prefix["engine_args"]["enable_prefix_caching"] = False
    assert prefix == plain


def test_generated_kv_cannot_be_published_as_prompt_kv():
    calls = []
    guarded = prompt_only(lambda obj, req, count: calls.append(count))
    req = SimpleNamespace(num_prompt_tokens=31)
    for count in (16, 31, 32, 100):
        guarded(None, req, count)
    assert calls == [16, 31]


def test_preemption_refuses_before_mutating_request():
    req = SimpleNamespace(request_id="child", num_computed_tokens=900)
    with pytest.raises(DattnInvariantError, match="refuses recompute"):
        refuse_preemption(None, req, 0)
    assert req.num_computed_tokens == 900


def test_children_have_independent_state_and_recorder_ids(layout):
    lp = processor.DattnLogitsProcessor()
    lp.state = DattnState()
    lp.state.configure(4, torch.device("cpu"), pin_memory=False)
    lp.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=True))
    payload = DattnRequest(layout=layout).to_extra_args()
    payload["req_id"] = "parent"
    params = SimpleNamespace(extra_args={"dattn": payload})
    outputs = [[], [], [], []]
    for i, out in enumerate(outputs):
        lp._add((i, params, [0] * layout.n_prompt_tokens, out))
    states = list(lp.state.rows)
    assert len(lp.state.by_key) == 4
    assert len({st.req_id for st in states}) == 4
    assert len({id(st.fsm) for st in states}) == 4
    lp.state.remove_row(2)
    assert len(lp.state.by_key) == 3
    assert lp.state.rows[0] is states[0]
    lp._add((2, params, [0] * layout.n_prompt_tokens, outputs[2]))
    assert lp.state.rows[2] is states[2]
    assert lp._anon == 4
    label = states[2].req_id
    lp.state.remove_row(2)
    lp.state.detached.clear()
    # Even reuse of the identical list after retirement gets a fresh trace label.
    lp._add((2, params, [0] * layout.n_prompt_tokens, outputs[2]))
    assert lp.state.rows[2].req_id != label
    assert lp._anon == 5
