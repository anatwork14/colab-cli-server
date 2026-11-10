"""Fixed model-card recipes and the recorded rollout and judge token budgets.

Generation is unseeded unless RolloutConfig.seed is supplied. Independent runs
therefore include sampling noise. The 8192-token rollout cap affects truncation
and format-failure counts, so finish_reason stays in every response record and
truncated answers stay in the accuracy denominator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Final

__all__ = [
    "GEMMA_4",
    "JUDGE_MAX_TOKENS",
    "QWEN3_5_THINKING_GENERAL",
    "QWEN3_6_27B_THINKING_GENERAL",
    "QWEN_NON_THINKING_GENERAL",
    "ROLLOUT_MAX_TOKENS",
    "SAMPLING_PARAMS",
    "RolloutConfig",
    "lookup_sampling",
]

#: Uniform across every Qwen 3.5 / 3.6 card dac checked.
QWEN_NON_THINKING_GENERAL: Final[dict[str, float | int]] = {
    "temperature": 0.7,
    "top_p": 0.80,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}

QWEN3_5_THINKING_GENERAL: Final[dict[str, float | int]] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}

QWEN3_6_27B_THINKING_GENERAL: Final[dict[str, float | int]] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}

#: Gemma 4's card publishes one recipe and leaves min_p / penalties unspecified, so they
#: are omitted and the engine defaults (0.0, 0.0, 1.0) apply. Keep it that way.
GEMMA_4: Final[dict[str, float | int]] = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}

#: `{model_id: {"non_thinking": ..., "thinking": ...}}`.
SAMPLING_PARAMS: Final[dict[str, dict[str, dict[str, float | int]]]] = {
    "Qwen/Qwen3.5-4B": {"non_thinking": QWEN_NON_THINKING_GENERAL, "thinking": QWEN3_5_THINKING_GENERAL},
    "Qwen/Qwen3.5-9B": {"non_thinking": QWEN_NON_THINKING_GENERAL, "thinking": QWEN3_5_THINKING_GENERAL},
    "Qwen/Qwen3.6-27B": {"non_thinking": QWEN_NON_THINKING_GENERAL, "thinking": QWEN3_6_27B_THINKING_GENERAL},
    "Qwen/Qwen3.6-27B-FP8": {
        "non_thinking": QWEN_NON_THINKING_GENERAL,
        "thinking": QWEN3_6_27B_THINKING_GENERAL,
    },
    "google/gemma-4-31B-it": {"non_thinking": GEMMA_4, "thinking": GEMMA_4},
    "google/gemma-4-12B-it": {"non_thinking": GEMMA_4, "thinking": GEMMA_4},
    "google/gemma-4-E4B-it": {"non_thinking": GEMMA_4, "thinking": GEMMA_4},
    "google/gemma-4-26B-A4B-it": {"non_thinking": GEMMA_4, "thinking": GEMMA_4},
}

#: dac `run_eval.py:1106` and `full_eval.sh` `--max-tokens 8192`.
ROLLOUT_MAX_TOKENS: Final[int] = 8192
#: dac `--judge-max-tokens 32768`.
JUDGE_MAX_TOKENS: Final[int] = 32768


def lookup_sampling(model_id: str, *, thinking: bool) -> dict[str, float | int]:
    """A COPY of the recipe. dac returns the shared module dict; this does not."""
    try:
        entry = SAMPLING_PARAMS[model_id]
    except KeyError:
        raise KeyError(
            f"no recorded sampling recipe for {model_id!r}. Add it to "
            "dattn.eval.sampling.SAMPLING_PARAMS with a citation to the model card."
        ) from None
    return dict(entry["thinking" if thinking else "non_thinking"])


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    """One arm's generation configuration, persisted verbatim with the outputs."""

    model_id: str
    arm: str
    prompt_format: str = "chunk_multi_turn_tool"
    dattn_enabled: bool = True
    thinking: bool = False
    max_tokens: int = ROLLOUT_MAX_TOKENS
    n: int = 1
    #: `None` reproduces dac, which has no generation seed at all.
    seed: int | None = None
    template_policy: str = "dac_bundled"
    segmenter: str = "semantic"
    tool_schema: bool = True
    vanilla_system: str | None = None
    sampling: dict[str, float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from dattn.prompt.segment import SEGMENTERS

        if self.segmenter not in SEGMENTERS:
            raise ValueError(f"unknown segmenter {self.segmenter!r}")
        if self.arm == "da_eqseg" and (
            self.segmenter != "equal_length" or self.prompt_format != "chunk_multi_turn_tool" or not self.dattn_enabled
        ):
            raise ValueError("da_eqseg requires equal_length segmentation, tool-call turns and a live mask")

    def resolved(self) -> RolloutConfig:
        """Fill `sampling` from the model card recipe when it was left empty."""
        if self.sampling:
            return self
        return RolloutConfig(**{**asdict(self), "sampling": lookup_sampling(self.model_id, thinking=self.thinking)})

    def request_body(self) -> dict[str, Any]:
        """The sampling half of an OpenAI chat-completions body."""
        body: dict[str, Any] = {**self.resolved().sampling, "max_tokens": self.max_tokens, "n": self.n}
        if self.seed is not None:
            body["seed"] = self.seed
        # top_k / min_p / repetition_penalty are vLLM extensions, not OpenAI fields.
        extra = {k: body.pop(k) for k in ("top_k", "min_p", "repetition_penalty") if k in body}
        if extra:
            body["extra_body"] = extra
        return body

    def to_dict(self) -> dict[str, object]:
        return asdict(self.resolved())
