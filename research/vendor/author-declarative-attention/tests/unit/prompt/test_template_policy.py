"""Campaign template pins, static render fixtures, and one policy for rollout/replay."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from fakes import FakeClient

from dattn.eval import runner
from dattn.eval.datasets import EvalRow
from dattn.eval.prompts import build_prompt
from dattn.eval.replay import replay_row
from dattn.eval.runner import RunPaths, run_rollouts
from dattn.eval.sampling import RolloutConfig
from dattn.eval.schema import iter_jsonl, json_safe
from dattn.prompt.template_policy import TEMPLATE_SHA256, resolve_template, template_metadata

CONTEXT = "Paris is in France."
QUESTION = "Where is Paris?"
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("model,family", [("google/gemma-4-31B-it", "gemma4"), ("Qwen/Qwen3.6-27B", "qwen3_5")])
def test_vendored_bytes_match_pin(model, family):
    template = resolve_template(None, model, "dac_bundled")
    assert hashlib.sha256(template.encode()).hexdigest() == TEMPLATE_SHA256[family]


def test_unknown_policy_and_model_fail_closed():
    with pytest.raises(ValueError, match="unknown template policy"):
        resolve_template(None, "Qwen/Qwen3.6-27B", "typo")
    with pytest.raises(ValueError, match="no dac_bundled template"):
        resolve_template(None, "unknown", "dac_bundled")


def test_bundled_render_matches_static_fixture_and_stock_differs(tokenizer):
    model = tokenizer.name_or_path
    family = "gemma4" if "gemma" in model else "qwen3_5"
    original = tokenizer.chat_template
    rendered = {}
    for policy in ("dac_bundled", "stock"):
        template = resolve_template(tokenizer, model, policy)
        rendered[policy] = build_prompt(
            tokenizer, context=CONTEXT, question=QUESTION, fmt="chunk_multi_turn_tool", chat_template=template
        )
    assert rendered["dac_bundled"].text == (FIXTURES / f"{family}_dac_bundled.txt").read_text()
    assert rendered["dac_bundled"].text != rendered["stock"].text
    assert tokenizer.chat_template == original


@pytest.mark.parametrize("policy", ["dac_bundled", "stock"])
@pytest.mark.parametrize(
    "arm,fmt,enabled",
    [("da", "chunk_multi_turn_tool", True), ("da_nm", "chunk_multi_turn_tool", False), ("vanilla", "vanilla", False)],
)
def test_rollout_and_replay_share_metadata_and_refuse_drift(
    tokenizer, policy, arm, fmt, enabled, tmp_path, monkeypatch
):
    config = RolloutConfig(
        model_id=tokenizer.name_or_path, arm=arm, prompt_format=fmt, dattn_enabled=enabled, template_policy=policy
    )
    paths = RunPaths(tmp_path / "cell")
    row = EvalRow(
        subset="s", row_index=0, split="test", context=CONTEXT, question=QUESTION, answer="France", rubric="France"
    )
    monkeypatch.setattr(runner, "ChatClient", lambda server: FakeClient(server, response="<answer>France</answer>"))
    assert run_rollouts([row], tokenizer=tokenizer, config=config, server=object(), paths=paths) == 1
    meta = json.loads(paths.meta.read_text())
    assert meta["rendering"] == template_metadata(resolve_template(tokenizer, config.model_id, policy), policy)
    recorded = next(iter_jsonl(paths.responses))
    joined = {**recorded, "context": CONTEXT}
    metrics = replay_row(joined, tokenizer=tokenizer, run_meta=meta)
    assert json_safe(metrics) == recorded["dac_metrics"]
    with pytest.raises(ValueError, match="requires the rollout meta"):
        replay_row(joined, tokenizer=tokenizer)
    with pytest.raises(ValueError, match="prompt hash"):
        replay_row({**joined, "prompt_sha256": "bad"}, tokenizer=tokenizer, run_meta=meta)
    with pytest.raises(ValueError, match="token count"):
        replay_row({**joined, "rendered_prompt_tokens": 0}, tokenizer=tokenizer, run_meta=meta)

    other = "stock" if policy == "dac_bundled" else "dac_bundled"
    meta["rollout"]["template_policy"] = other
    with pytest.raises(ValueError, match="disagrees with rollout metadata"):
        replay_row(joined, tokenizer=tokenizer, run_meta=meta)
    original = paths.meta.read_bytes()
    with pytest.raises(ValueError, match="different or unrecorded template"):
        run_rollouts(
            [row], tokenizer=tokenizer, config=replace(config, template_policy=other), server=object(), paths=paths
        )
    assert paths.meta.read_bytes() == original


def test_campaign_vanilla_omits_system_and_pins_rollout_and_replay(tokenizer, tmp_path, monkeypatch):
    from dattn.campaign import driver
    from dattn.prompt.formats.base import DEFAULT_SYSTEM
    from dattn.prompt.render import render

    context = CONTEXT * 10
    model = "gemma" if "gemma" in tokenizer.name_or_path else "qwen"
    config = driver.config(model, "vanilla")
    assert config.vanilla_system == ""
    assert driver.provenance(model, "vanilla", tokenizer)["rollout"]["vanilla_system"] == ""
    template = resolve_template(tokenizer, config.model_id, config.template_policy)
    kwargs = {"context": context, "question": QUESTION, "fmt": "vanilla", "chat_template": template}
    historical = render(tokenizer, **kwargs, vanilla_system=config.vanilla_system)
    default = render(tokenizer, **kwargs)
    assert [m["role"] for m in historical.messages] == ["user"]
    assert default.messages[0] == {"role": "system", "content": DEFAULT_SYSTEM}
    assert len(default.ids) - len(historical.ids) == 11
    assert RolloutConfig(model_id=config.model_id, arm="vanilla").vanilla_system is None

    paths = RunPaths(tmp_path / "cell")
    row = EvalRow(
        subset="s", row_index=0, split="test", context=context, question=QUESTION, answer="France", rubric="France"
    )
    monkeypatch.setattr(runner, "ChatClient", lambda server: FakeClient(server, response="<answer>France</answer>"))
    assert run_rollouts([row], tokenizer=tokenizer, config=config, server=object(), paths=paths) == 1
    meta = json.loads(paths.meta.read_text())
    recorded = next(iter_jsonl(paths.responses))
    assert meta["rollout"]["vanilla_system"] == ""
    assert recorded["prompt_sha256"] == build_prompt(tokenizer, **kwargs, vanilla_system="").sha256
    assert (
        json_safe(replay_row({**recorded, "context": context}, tokenizer=tokenizer, run_meta=meta))
        == recorded["dac_metrics"]
    )
    with pytest.raises(ValueError, match="different vanilla system turn"):
        run_rollouts(
            [row], tokenizer=tokenizer, config=replace(config, vanilla_system=None), server=object(), paths=paths
        )
