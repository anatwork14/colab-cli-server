"""The persisted schema, the sampling recipes and the Lane A adapter."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from dattn.contracts import PromptLayout
from dattn.eval.prompts import PROMPT_FORMATS, BuiltPrompt, build_prompt, dattn_extra_args, dattn_xargs, prompt_sha256
from dattn.eval.sampling import GEMMA_4, QWEN_NON_THINKING_GENERAL, ROLLOUT_MAX_TOKENS, RolloutConfig, lookup_sampling
from dattn.eval.schema import ResponseRecord, RunMeta, iter_jsonl, json_safe, write_jsonl

# --------------------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------------------


def test_a_nan_metric_is_written_as_null_not_nan() -> None:
    """Major 19. Every vanilla row carries five NaNs and must still serialise."""
    from dattn.fsm.metrics import Ledger, full_metrics

    metrics = full_metrics(Ledger(0))
    assert math.isnan(metrics["focus_success_rate"])  # the ledger really is empty
    rec = ResponseRecord(subset="s", row_index=0, dac_metrics=dict(metrics))
    out = rec.to_json()
    assert "NaN" not in out and "Infinity" not in out
    back = json.loads(out)["dac_metrics"]
    assert back["focus_success_rate"] is None
    assert back["mean_focus_area"] is None
    assert back["local_ratio"] == 0.0  # dac's asymmetry survives sanitisation
    assert set(back) == set(metrics)  # nothing is dropped, only NaN is rewritten


def test_write_jsonl_sanitises_nested_nan(tmp_path: Path) -> None:
    row = {"a": {"b": [{"c": math.nan}]}, "d": 1.5}
    write_jsonl(tmp_path / "r.jsonl", [row])
    (back,) = list(iter_jsonl(tmp_path / "r.jsonl"))
    assert back == {"a": {"b": [{"c": None}]}, "d": 1.5}


def test_an_infinity_is_also_nulled() -> None:
    assert json_safe([math.inf, -math.inf, math.nan, 0.0]) == [None, None, None, 0.0]
    assert json_safe({"k": math.inf}) == {"k": None}
    assert json_safe((1, math.nan)) == [1, None]  # dac's helper lists a tuple
    assert json_safe("nan") == "nan" and json_safe(True) is True and json_safe(None) is None


def test_a_record_round_trips_through_json() -> None:
    rec = ResponseRecord(subset="s", row_index=3, response="<answer>x</answer>", finish_reason="length")
    assert ResponseRecord.from_dict(json.loads(rec.to_json())) == rec
    assert rec.id == "s#3"
    assert rec.key == ("s", 3, 0)


def test_truncated_is_derived_from_the_real_finish_reason() -> None:
    """dac writes the string "stop" in every path, so a dac row is never `truncated`."""
    assert ResponseRecord(subset="s", row_index=0, finish_reason="length").truncated
    assert not ResponseRecord(subset="s", row_index=0, finish_reason="stop").truncated
    assert not ResponseRecord(subset="s", row_index=0).truncated


def test_raw_row_coercions_and_unknown_fields_are_refused():
    with pytest.raises(TypeError):
        ResponseRecord.from_dict({"subset": "s", "row_index": 0, "something_dac_only": 1})
    with pytest.raises(ValueError, match="strings"):
        ResponseRecord.from_dict({"subset": "s", "row_index": 0, "response": None})
    with pytest.raises(TypeError):
        ResponseRecord.from_dict({"response": "no identity"})


def test_the_schema_carries_every_dac_field() -> None:
    dac_fields = {
        "subset",
        "row_index",
        "response_index",
        "question",
        "answer",
        "rubric",
        "raw_context_tokens",
        "rendered_prompt_tokens",
        "n_bookmarks",
        "response",
        "parsed_answer",
        "format_correct",
        "dac_metrics",
    }
    assert dac_fields <= set(ResponseRecord.__slots__)


def test_a_malformed_jsonl_line_raises_rather_than_vanishing(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n\nnot json\n')
    with pytest.raises(ValueError, match="not valid JSON"):
        list(iter_jsonl(path))


def test_write_jsonl_appends(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    write_jsonl(path, [{"a": 1}])
    write_jsonl(path, [{"a": 2}], append=True)
    assert [d["a"] for d in iter_jsonl(path)] == [1, 2]


def test_run_meta_writes_and_reloads(tmp_path: Path) -> None:
    meta = RunMeta(model_id="m", arm="da", notes=["n"])
    meta.write(tmp_path / "meta.json")
    assert RunMeta.from_dict(json.loads((tmp_path / "meta.json").read_text())) == meta


# --------------------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------------------


def test_the_recorded_recipes_are_the_model_card_values() -> None:
    assert lookup_sampling("Qwen/Qwen3.6-27B", thinking=False) == QWEN_NON_THINKING_GENERAL
    assert lookup_sampling("google/gemma-4-31B-it", thinking=False) == GEMMA_4
    # Qwen3.6-27B's thinking recipe differs from Qwen3.5's by presence_penalty.
    assert lookup_sampling("Qwen/Qwen3.6-27B", thinking=True)["presence_penalty"] == 0.0
    assert lookup_sampling("Qwen/Qwen3.5-4B", thinking=True)["presence_penalty"] == 1.5


def test_lookup_returns_a_copy_not_the_shared_dict() -> None:
    """dac returns the module-level dict (`B` E35). Mutating it there is a live hazard."""
    got = lookup_sampling("Qwen/Qwen3.6-27B", thinking=False)
    got["temperature"] = 99.0
    assert QWEN_NON_THINKING_GENERAL["temperature"] == 0.7


def test_an_unknown_model_names_what_to_do() -> None:
    with pytest.raises(KeyError, match="SAMPLING_PARAMS"):
        lookup_sampling("nobody/nothing", thinking=False)


def test_the_rollout_default_has_no_seed_which_is_what_dac_did() -> None:
    cfg = RolloutConfig(model_id="Qwen/Qwen3.6-27B", arm="da").resolved()
    assert cfg.seed is None
    assert cfg.max_tokens == ROLLOUT_MAX_TOKENS == 8192
    body = cfg.request_body()
    assert "seed" not in body
    assert body["temperature"] == 0.7
    # vLLM extensions travel under extra_body, not as top-level OpenAI fields.
    assert body["extra_body"] == {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}


def test_gemma_leaves_the_unspecified_knobs_to_the_engine() -> None:
    body = RolloutConfig(model_id="google/gemma-4-31B-it", arm="da").request_body()
    assert set(body["extra_body"]) == {"top_k"}
    assert "presence_penalty" not in body


# --------------------------------------------------------------------------------------
# the Lane A adapter
# --------------------------------------------------------------------------------------


def test_prompt_sha256_is_over_the_text() -> None:
    import hashlib

    assert prompt_sha256("abc") == hashlib.sha256(b"abc").hexdigest()


def test_build_prompt_uses_the_shared_renderer(monkeypatch) -> None:
    from dattn.contracts import Rendered

    layout = PromptLayout(n_prompt_tokens=50, question_start=40, segments=((16, 30),))

    def fake(tokenizer, **kw):
        assert kw["fmt"] == "chunk_multi_turn_tool"
        return Rendered(
            text="PROMPT",
            ids=tuple(range(50)),
            layout=layout,
            messages=(),
            tools=None,
            segments=(object(),),  # type: ignore[arg-type]
            format_error="",
        )

    import importlib

    monkeypatch.setattr(importlib.import_module("dattn.prompt.render"), "render", fake)
    built = build_prompt(None, context="c", question="q", fmt="chunk_multi_turn_tool")
    assert isinstance(built, BuiltPrompt)
    assert built.text == "PROMPT"
    assert built.n_prompt_tokens == 50
    assert built.n_segments == 1
    assert built.sha256 == prompt_sha256("PROMPT")


def test_an_unknown_format_is_rejected_before_the_renderer_runs() -> None:
    with pytest.raises(ValueError, match="unknown prompt format"):
        build_prompt(None, context="c", question="q", fmt="bookmark")
    assert PROMPT_FORMATS == ("chunk_multi_turn_tool", "chunk_single_turn_xml", "vanilla")


def test_the_extra_args_payload_is_json_and_round_trips() -> None:
    from dattn.contracts import DattnRequest

    layout = PromptLayout(n_prompt_tokens=50, question_start=40, segments=((16, 30),))
    payload = dattn_extra_args(layout)
    assert DattnRequest.from_extra_args(payload).layout == layout

    xargs = dattn_xargs(layout, enabled=False)
    assert set(xargs) == {"dattn"}
    assert isinstance(xargs["dattn"], str)  # vllm_xargs cannot carry a nested dict
    decoded = json.loads(xargs["dattn"])
    assert decoded["enabled"] is False
    assert DattnRequest.from_extra_args(decoded).layout == layout
