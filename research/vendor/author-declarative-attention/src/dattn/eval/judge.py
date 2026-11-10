"""The judge step: Qwen3.5-4B, thinking on, dac's rubric prompt byte for byte.

The judge is a **measurement instrument**, so everything about it that dac fixed stays
fixed: the model (`Qwen/Qwen3.5-4B`), thinking on, the `thinking-general` sampling
recipe, `max_tokens=32768`, the three fields it sees (`question`, `rubric`,
`parsed_answer`) and never the context, and the rubric template byte for byte.
Reference runs used these settings and measured the judge's own
self-inconsistency at 1.4-2.0%, which is the floor on any comparison.

`templates/judge_rubric_v3.jinja` is `evaluation_with_rubric_v3.jinja`, sha256
`2c51b53d...`, md5 `2deed1ad...`. It is data, not code: it is the prompt the published
numbers were produced under, and reproducing it is the whole point. `JUDGE_TEMPLATE_SHA256`
pins it and `tests/unit/eval/test_judge.py` fails if the bytes ever drift.

**How the request goes out.** dac posted CHAT MESSAGES with thinking enabled and let the
server apply Qwen's own template, so dattn posts `/v1/chat/completions` with one user
turn and `chat_template_kwargs={"enable_thinking": true}` (`JudgeConfig.request_body`),
never the rendered rubric as a raw `/v1/completions` prompt. The judge server must be
booted without a reasoning parser so the response retains its original think tags.

**What dattn changes, deliberately:** the verdict parser. dac's
(`run_eval.py:89-102`) returns `None` on

- a truncated `<think>` with no closing tag: 141 of 142 null verdicts on the
  legacy data were exactly this, and 472 exist repo-wide;
- `{"correct": "true"}`, `{"correct": 1}`, `{'correct': True}`, or JSON with prose around
  it;

and takes the FIRST of two verdicts, so a self-correcting judge is scored on its first
guess. Every one of those becomes `None`, and dac scores `None` as **wrong**
while keeping the row in the denominator.

**What dattn does NOT change:** where the verdict is read from. dac splits on the FIRST
`</think>` and parses only the tail, and so does `_verdict_regions`. dattn used to split
on the LAST close tag, which recovered a verdict stated before a reopened scratchpad
(`B` E12); that was dropped, deliberately, because this parser exists to be comparable
with dac's recorded judgments and a silent improvement is still a divergence. See
`_verdict_regions` for the full reasoning and the measurement behind it.

dattn parses in explicit, recorded stages within that tail, prefers the LAST verdict in
it, and never converts an unparseable verdict into a wrong answer silently:
`Verdict.correct` is `None` and `Verdict.stage` says why. `Verdict.dac_correct` is what
dac's parser would have said, on the same text, so historical parser differences remain inspectable.
The aggregator counts `None` as wrong, because that is what the
published numbers did, but it always reports `n_unparsable` alongside.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from dattn.eval.sampling import JUDGE_MAX_TOKENS, lookup_sampling

__all__ = [
    "JUDGE_MAX_TOKENS",
    "JUDGE_MODEL",
    "JUDGE_TEMPLATE_SHA256",
    "PARSE_STAGES",
    "JudgeConfig",
    "Judgment",
    "Verdict",
    "dac_parse_verdict",
    "judge_template_path",
    "judge_template_text",
    "parse_verdict",
    "render_judge_prompt",
]

#: dac `full_eval.sh:JUDGE_MODEL`, and the exact snapshot `851bf6e806...` (`E 4`).
JUDGE_MODEL: Final[str] = "Qwen/Qwen3.5-4B"

#: sha256 of `templates/judge_rubric_v3.jinja`, which must equal dac's
#: `templates/evaluation_with_rubric_v3.jinja`.
JUDGE_TEMPLATE_SHA256: Final[str] = "2c51b53d3315f3b8d143f12a38485d60a7be11eaedd0d767f4a2f532bb867e82"

#: In the order the parser tries them. `unparsable` is a real outcome, not a failure.
PARSE_STAGES: Final[tuple[str, ...]] = (
    "json",  # the whole post-think text is a JSON object with a bool `correct`
    "json_block",  # a JSON object somewhere in the text has a bool-like `correct`
    "regex",  # the last `"correct": true|false`
    "regex_lenient",  # the last `correct` key with a bool-like value, quotes optional
    "unparsable",
)

_THINK_CLOSE: Final[str] = "</think>"
_THINK_OPEN: Final[str] = "<think>"
_CORRECT_STRICT: Final[re.Pattern[str]] = re.compile(r'"correct":\s*(true|false)')
_CORRECT_LENIENT: Final[re.Pattern[str]] = re.compile(
    r"""['"]?correct['"]?\s*[:=]\s*['"]?(true|false|yes|no|1|0)['"]?""", re.IGNORECASE
)
_JSON_BLOCK: Final[re.Pattern[str]] = re.compile(r"\{[^{}]*\}", re.DOTALL)
_TRUE_WORDS: Final[frozenset[str]] = frozenset({"true", "yes", "1"})


# --------------------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------------------


def judge_template_path() -> Path:
    return Path(__file__).resolve().parent / "templates" / "judge_rubric_v3.jinja"


def judge_template_text() -> str:
    return judge_template_path().read_text()


def render_judge_prompt(question: str, rubric: str, response: str, *, template: str | None = None) -> str:
    """Render the rubric prompt byte-identically to dac's `Template(...).render(...)`.

    The template has three `{{ }}` expressions and no control structures, no filters and
    no comments, so substitution is exact. The one non-obvious rule reproduced here is
    Jinja's `keep_trailing_newline=False` default, which drops exactly one trailing
    newline; `tests/unit/eval/test_judge.py` checks this against real Jinja when it is
    importable.

    dac passes `response=r["parsed_answer"] or ""` (`run_eval.py:670-675`), so a
    format-correct-but-empty `<answer></answer>` reaches the judge as an empty
    `<model_response>`. That is reproduced, not fixed, here.
    """
    text = judge_template_text() if template is None else template
    for name, value in (("question", question), ("rubric", rubric), ("response", response)):
        text = text.replace("{{ " + name + " }}", value)
    return text[:-1] if text.endswith("\n") else text


# --------------------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Verdict:
    """A parsed judge verdict, with the stage that produced it."""

    correct: bool | None
    stage: str
    #: What dac's parser would have returned on the same text.
    dac_correct: bool | None = None
    #: A `<think>` block that never closed.
    think_truncated: bool = False
    #: More than one verdict in the text. dac takes the first, dattn the last.
    n_candidates: int = 0

    @property
    def parsed(self) -> bool:
        return self.correct is not None

    @property
    def differs_from_dac(self) -> bool:
        return self.correct != self.dac_correct

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def dac_parse_verdict(text: str) -> bool | None:
    """dac's parser, reproduced exactly (`run_eval.py:89-102`). For parity only."""
    if _THINK_CLOSE in text:
        text = text.split(_THINK_CLOSE, 1)[1].strip()
    try:
        v = json.loads(text)["correct"]
        if isinstance(v, bool):
            return v
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    m = _CORRECT_STRICT.search(text)
    return (m.group(1) == "true") if m else None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value) if value in (0, 1) else None
    if isinstance(value, str):
        low = value.strip().lower()
        if low in _TRUE_WORDS:
            return True
        if low in {"false", "no", "0"}:
            return False
    return None


def _scan(cand: str) -> tuple[bool | None, str]:
    """Read a verdict out of ONE region of text, trying the stages in order."""
    try:
        obj = json.loads(cand)
    except (json.JSONDecodeError, TypeError):
        obj = None
    if isinstance(obj, Mapping) and isinstance(obj.get("correct"), bool):
        return obj["correct"], "json"

    for block in reversed(_JSON_BLOCK.findall(cand)):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, Mapping) and "correct" in obj:
            val = _as_bool(obj["correct"])
            if val is not None:
                return val, "json_block"

    hits = _CORRECT_STRICT.findall(cand)
    if hits:
        return hits[-1] == "true", "regex"

    hits = _CORRECT_LENIENT.findall(cand)
    if hits:
        return hits[-1].lower() in _TRUE_WORDS, "regex_lenient"

    return None, "unparsable"


def _verdict_regions(text: str) -> list[str]:
    """The one region of a judge response that may hold a verdict.

    Everything through the FIRST `</think>` is dropped and only the tail is parsed. That
    is dac's rule verbatim (`run_eval.py:93-102`), and it is the rule every published
    accuracy number was produced under. The chain of thought is never a candidate.

    `_THINK_CLOSE in text` is the whole test, with no `<think>` required: Qwen's chat
    template injects the opener into the PROMPT, so a real judge response carries the
    close tag alone. All 500 sampled rows of dac's own recorded judgments for
    gemma_4_31b_it/dac_chunk_multi_turn_tool_v1 have `</think>` and none has `<think>`.
    A response with no `</think>` at all is a truncated think or a judge that
    never thought, and the whole of it is the region.

    **First, not last, and this is a change of mind.** dattn used to split on the last
    close tag so that a judge which reopened its scratchpad after answering did not erase
    its own verdict, and used to fall back to the text before an explicit
    opener. Both are gone. The parser exists to be comparable with dac's recorded
    judgments, so a rule that is better than dac's is still a divergence we would have to
    disclose and could not evidence. The two rules differ only on a response carrying more
    than one close tag: 0 of the 500 real rows do, so nothing recorded moves, and the
    difference would only ever appear in the population the new chat-completions judge
    generates, which is exactly where an undisclosed divergence would be invisible and
    would move a headline number. Comparability takes precedence. This is deliberately dac's
    behaviour: `'{"correct": true}<think>no</think>'` parses the empty tail and is
    `unparsable`, which `dac_parse_verdict` also reports, and the aggregator scores `None`
    as wrong exactly as dac did.
    """
    body = text.split(_THINK_CLOSE, 1)[1].strip() if _THINK_CLOSE in text else text.strip()
    return [body] if body else []


def parse_verdict(text: str) -> Verdict:
    """Parse a judge response into a verdict, recording how it was reached."""
    dac = dac_parse_verdict(text)
    think_truncated = _THINK_OPEN in text and _THINK_CLOSE not in text
    n_candidates = len(_CORRECT_LENIENT.findall(text))

    # Region major, stage minor. The other way round a weak stage on a later region beat
    # a strong stage on the body, which is how a verdict from inside the scratchpad could
    # be returned in preference to the final one.
    for region in _verdict_regions(text):
        val, stage = _scan(region)
        if stage != "unparsable":
            return Verdict(val, stage, dac, think_truncated, n_candidates)

    return Verdict(None, "unparsable", dac, think_truncated, n_candidates)


# --------------------------------------------------------------------------------------
# Records and configuration
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class Judgment:
    """One judged rollout. dac's four fields, plus how the verdict was reached."""

    subset: str
    row_index: int
    response_index: int
    correct: bool | None
    evaluation_text: str = ""
    stage: str = "unparsable"
    dac_correct: bool | None = None
    think_truncated: bool = False
    n_candidates: int = 0
    finish_reason: str = ""
    error: str = ""
    transport_error: bool = False

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.subset, self.row_index, self.response_index)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Judgment:
        kw = {k: v for k, v in d.items() if k in set(cls.__slots__)}
        kw.setdefault("response_index", 0)
        kw.setdefault("correct", None)
        # Old records only had error, which the HTTP client used for transport failures.
        kw.setdefault("transport_error", bool(d.get("error")))
        return cls(**kw)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    """Everything the judge phase pins. Persisted into `meta.json`."""

    model_id: str = JUDGE_MODEL
    thinking: bool = True
    max_tokens: int = JUDGE_MAX_TOKENS
    template_sha256: str = JUDGE_TEMPLATE_SHA256
    seed: int | None = None
    chunk_size: int = 256

    def sampling(self) -> dict[str, float | int]:
        return lookup_sampling(self.model_id, thinking=self.thinking)

    def request_body(self) -> dict[str, Any]:
        """The judge half of an OpenAI chat-completions body, dac's recipe verbatim.

        Post it to `{base_url}/v1/chat/completions` with one user turn holding
        `render_judge_prompt(...)`, and read the verdict from
        `choices[0].message.content`. The wire shape, verified field by field against
        dac, is

            {"model": "Qwen/Qwen3.5-4B",
             "messages": [{"role": "user", "content": <rendered rubric prompt>}],
             "chat_template_kwargs": {"enable_thinking": true},
             "max_tokens": 32768, "n": 1,
             "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
             "presence_penalty": 1.5, "repetition_penalty": 1.0}

        No `stop`, no `seed`, no `tools`, no system turn. `top_k`, `min_p` and
        `repetition_penalty` are vLLM extensions rather than OpenAI fields, so they
        travel in `extra_body` and `ChatClient.chat` flattens them, exactly as
        `RolloutConfig.request_body` does.

        Provenance. The six sampling values are `model_specs.QWEN3_5_THINKING_GENERAL_SAMPLING_PARAMS`,
        selected by `lookup_sampling_params(judge_model, thinking=True)`
        (`run_eval.py:1197-1199`). `max_tokens=32768` is `full_eval.sh:79`
        (`--judge-max-tokens 32768`). `enable_thinking` is `full_eval.sh:79`
        (`--judge-enable-thinking`), which dac shipped as a per-item `enable_thinking`
        field its own server fed to `apply_chat_template`; `chat_template_kwargs` is how a
        stock vLLM OpenAI endpoint forwards the same thing. `n` never appears in dac's
        client payload because dac's server forced `n=eval_repeat` and ignored the
        request's `n` (`serve.py:553`); the judge server was launched with no
        `--eval-repeat`, whose default is 1, so `n: 1` is that forcing written down.

        **The judge server must be booted WITHOUT `--reasoning-parser`.** dac's judge
        server had none, so its `evaluation_text` is one string containing the chain of
        thought, the `</think>` close and the verdict, and every downstream statistic
        (`stage`, `think_truncated`, `n_candidates`, `dac_correct`) is defined on that
        string. With a reasoning parser vLLM moves the think block into
        `message.reasoning_content` and leaves only the trailing JSON in
        `message.content`. The client reads the response as supplied and does not
        reassemble reasoning-parser output.
        """
        body: dict[str, Any] = {
            **self.sampling(),
            "max_tokens": self.max_tokens,
            "n": 1,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        if self.seed is not None:
            body["seed"] = self.seed
        extra = {k: body.pop(k) for k in ("top_k", "min_p", "repetition_penalty") if k in body}
        if extra:
            body["extra_body"] = extra
        return body

    def messages(self, question: str, rubric: str, response: str) -> list[dict[str, str]]:
        """dac's one user turn, no system turn (`run_eval.py:668-675`)."""
        return [{"role": "user", "content": render_judge_prompt(question, rubric, response)}]

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "sampling": self.sampling()}
