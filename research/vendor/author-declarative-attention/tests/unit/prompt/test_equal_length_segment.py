"""Lossless token-index cuts and the same segmenter throughout eval and replay."""

import json
from dataclasses import replace

import pytest
import synth_contexts
from fakes import FakeClient
from test_prompt_segment import FakeFastTokenizer

from dattn import cli
from dattn.eval import datasets, runner
from dattn.eval.datasets import EvalRow
from dattn.eval.replay import replay_row
from dattn.eval.runner import RunPaths, run_rollouts
from dattn.eval.sampling import RolloutConfig
from dattn.eval.schema import iter_jsonl, json_safe
from dattn.prompt.render import render
from dattn.prompt.segment import EMPTY_CONTEXT, normalize_context, segment_context, segment_equal_length
from dattn.prompt.template_policy import resolve_template


def assert_partition(context, segments):
    assert "".join(s.text for s in segments) == context
    assert [s.id for s in segments] == list(range(1, len(segments) + 1))
    cursor = 0
    for s in segments:
        assert s.char_start == cursor
        assert s.char_end > s.char_start
        assert s.text == context[s.char_start : s.char_end]
        cursor = s.char_end
    assert cursor == len(context)


@pytest.mark.parametrize("n_tokens,n", [(12, 3), (11, 3), (1, 1), (17, 1), (2, 5), (0, 3)])
def test_exact_indices_equal_lengths_and_last_remainder(n_tokens, n):
    tok = FakeFastTokenizer()
    context = "abcd" * n_tokens
    segments = segment_equal_length(tok, context, n)
    assert_partition(context, segments)
    if not context:
        assert segments == []
        return
    realized = min(n, n_tokens)
    width, remainder = divmod(n_tokens, realized)
    assert len(segments) == realized
    assert [s.token_count for s in segments] == [width] * (realized - 1) + [width + remainder]
    starts = [a for a, _ in tok(context, return_offsets_mapping=True)["offset_mapping"]]
    assert [s.char_start for s in segments] == [0] + [starts[k * width] for k in range(1, realized)]


@pytest.mark.parametrize("context", ["", " \n\t", EMPTY_CONTEXT, "hi", "café 日本語 🧪 text"])
def test_normalized_degenerate_and_unicode_contexts(tokenizer, context):
    context = normalize_context(context)
    n = len(segment_context(tokenizer, context))
    segments = segment_equal_length(tokenizer, context, n)
    assert_partition(context, segments)
    assert len(segments) == n == 1
    assert "\ufffd" not in segments[0].text


def test_same_count_as_semantic_and_exact_cuts_with_cached_tokenizer(tokenizer):
    context = synth_contexts.context_for("five_segments")
    semantic = segment_context(tokenizer, context)
    segments = segment_equal_length(tokenizer, context, len(semantic))
    assert_partition(context, segments)
    assert len(segments) == len(semantic) == 5
    offsets = tokenizer(context, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    width, remainder = divmod(len(offsets), len(semantic))
    assert [s.char_start for s in segments] == [0] + [offsets[k * width][0] for k in range(1, len(semantic))]
    assert [s.token_count for s in segments] == [width] * (len(semantic) - 1) + [width + remainder]
    assert [s.char_end for s in segments] != [s.char_end for s in semantic]


@pytest.mark.parametrize(
    "context,offsets,n,counts",
    [
        ("🧪x", [(0, 1), (0, 1), (0, 1), (1, 2)], 4, [3, 1]),
        ("a🧪b", [(0, 1), (1, 2), (1, 2), (2, 3)], 2, [1, 3]),
        ("  a b  ", [(2, 3), (4, 5)], 2, [1, 1]),
        ("text", [], 0, [0]),
    ],
)
def test_shared_offsets_gaps_and_tokenless_text_are_lossless(context, offsets, n, counts):
    class OffsetTokenizer:
        is_fast = True
        calls = 0

        def __call__(self, text, *, return_offsets_mapping, add_special_tokens):
            self.calls += 1
            assert text == context and return_offsets_mapping and not add_special_tokens
            return {"offset_mapping": offsets}

        def decode(self, *args, **kwargs):
            pytest.fail("segment_equal_length must never decode")

    tok = OffsetTokenizer()
    segments = segment_equal_length(tok, context, n)
    assert_partition(context, segments)
    assert [s.token_count for s in segments] == counts
    assert tok.calls == 1


def test_unknown_segmenter_and_slow_tokenizer_fail_closed():
    with pytest.raises(ValueError, match="fast tokenizer"):
        segment_equal_length(None, "text", 1)
    with pytest.raises(ValueError, match="unknown segmenter"):
        render(None, context="c", question="q", fmt="chunk_multi_turn_tool", segmenter="typo")
    with pytest.raises(ValueError, match="unknown segmenter"):
        RolloutConfig(model_id="model", arm="da", segmenter="typo")
    with pytest.raises(ValueError, match="da_eqseg requires"):
        RolloutConfig(model_id="model", arm="da_eqseg")


@pytest.mark.parametrize("arm,segmenter", [("da", "semantic"), ("da_eqseg", "equal_length")])
def test_cli_rollout_replay_and_resume_use_one_segmenter(tokenizer, arm, segmenter, tmp_path, monkeypatch):
    context = synth_contexts.context_for("five_segments")
    row = EvalRow(subset="s", row_index=0, split="test", context=context, question="q", answer="x", rubric="x")
    monkeypatch.setattr(cli, "_tokenizer", lambda model: tokenizer)
    monkeypatch.setattr(datasets, "select_rows", lambda **kw: [row])
    sent = []

    class Client(FakeClient):
        def complete_many(self, prompts, **kw):
            sent.extend(prompts)
            assert json.loads(kw["xargs"][0]["dattn"])["enabled"] is True
            return super().complete_many(prompts, **kw)

    monkeypatch.setattr(runner, "ChatClient", lambda server: Client(server, response="<answer>x</answer>"))
    assert cli.main(["eval", "--model", tokenizer.name_or_path, "--arm", arm, "--out", str(tmp_path)]) == 0
    paths = RunPaths.for_cell(tmp_path, tokenizer.name_or_path, arm)
    meta = json.loads(paths.meta.read_text())
    config = RolloutConfig(**meta["rollout"])
    assert config.segmenter == segmenter
    assert config.prompt_format == "chunk_multi_turn_tool" and config.dattn_enabled
    recorded = next(iter_jsonl(paths.responses))
    assert recorded["segmenter"] == segmenter
    assert recorded["n_semantic_segments"] == recorded["n_bookmarks"] == 5
    expected_segments = segment_context(tokenizer, context)
    if segmenter == "equal_length":
        expected_segments = segment_equal_length(tokenizer, context, len(expected_segments))
    rendered = render(
        tokenizer,
        context=context,
        question="q",
        fmt=config.prompt_format,
        segmenter=segmenter,
        chat_template=resolve_template(tokenizer, config.model_id, config.template_policy),
    )
    assert list(rendered.segments) == expected_segments
    assert sent == [rendered.ids]
    joined = {**recorded, "context": context}
    assert json_safe(replay_row(joined, tokenizer=tokenizer, run_meta=meta)) == recorded["dac_metrics"]
    other = "semantic" if segmenter == "equal_length" else "equal_length"
    with pytest.raises(ValueError, match="segmenter disagrees"):
        replay_row({**joined, "segmenter": other}, tokenizer=tokenizer, run_meta=meta)
    changed_meta = {**meta, "rollout": {**meta["rollout"], "arm": "custom", "segmenter": other}}
    with pytest.raises(ValueError, match="segmenter disagrees"):
        replay_row(joined, tokenizer=tokenizer, run_meta=changed_meta)
    # Even a record without the added segmenter field still has its original prompt hash.
    del joined["segmenter"]
    with pytest.raises(ValueError, match="prompt hash"):
        replay_row(joined, tokenizer=tokenizer, run_meta=changed_meta)
    before = paths.meta.read_bytes(), paths.responses.read_bytes()
    assert run_rollouts([row], tokenizer=tokenizer, config=config, server=object(), paths=paths) == 0
    with pytest.raises(ValueError, match="different segmenter"):
        run_rollouts(
            [row],
            tokenizer=tokenizer,
            config=replace(config, arm="custom", segmenter=other),
            server=object(),
            paths=paths,
        )
    assert (paths.meta.read_bytes(), paths.responses.read_bytes()) == before


def test_render_cli_exposes_segmenter_and_both_counts(tokenizer, tmp_path, monkeypatch, capsys):
    context_file = tmp_path / "context.txt"
    context_file.write_text(synth_contexts.context_for("five_segments"))
    monkeypatch.setattr(cli, "_tokenizer", lambda model: tokenizer)
    assert (
        cli.main(
            [
                "render",
                "--model",
                tokenizer.name_or_path,
                "--context-file",
                str(context_file),
                "--question",
                "q",
                "--segmenter",
                "equal_length",
            ]
        )
        == 0
    )
    got = json.loads(capsys.readouterr().out)
    assert got["segmenter"] == "equal_length"
    assert got["n_semantic_segments"] == got["n_segments"] == 5
