"""The `<answer>` format gate, and the one place dattn can deliberately differ from dac.

dac (`run_eval.py:83-86`) does:

    m = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)

`re.search` plus a lazy group takes the **first** `<answer>` block. A model that answers,
notices its mistake and answers again is scored on its first guess, and
`<answer>WRONG</answer> ... <answer>RIGHT</answer>` is graded WRONG.

**The first block is dattn's default**, because it is the measurement the published
numbers were produced under and the gate is an instrument. dattn defaulted to the last
block until the owner ruled otherwise; the argument for the last block is a good one and
it survives as `--last-answer`, not as a silent change to every number.

`--last-answer` grades the last block instead: the instruction templates ask for one
answer at the end of the response, the last block is the model's final answer under every
reading of the protocol, and taking the first one measures a transcription artefact rather
than the model. On the recorded dac data the two policies differ on 11 of 1611 Qwen
vanilla rows, 9 of 1611 Qwen DA rows and 0 Gemma rows. Every `Extraction` carries
`first_text` under both policies, so a parity run can quantify the deviation without a
second pass.

Everything else about the gate is dac's on purpose, because the gate is the measurement
instrument and changing it changes the published numbers:

- case sensitive, so `<Answer>` is a format failure;
- no attribute tolerance, so `<answer type="final">` is a format failure;
- `.strip()` on the captured text;
- an empty `<answer></answer>` is format-CORRECT with an empty body, which is then sent
  to the judge as an empty `<model_response>`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

__all__ = ["ANSWER_RE", "Extraction", "extract_answer"]

#: dac's regex, byte for byte (`run_eval.py:85`).
ANSWER_RE: Final[re.Pattern[str]] = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

#: What dac writes into `parsed_answer` when the gate fails.
INVALID_SENTINEL: Final[str] = "<invalid_response>"


@dataclass(frozen=True, slots=True)
class Extraction:
    """The result of the format gate, carrying both policies."""

    text: str | None
    format_correct: bool
    n_blocks: int
    policy: str  # "first" (the default, dac's) or "last"
    #: What dac's `re.search` would have produced, always, for the parity report.
    first_text: str | None = None
    #: The other policy's text, always, so one run prices `--last-answer` without a second
    #: pass. `first_text` did that job while the last block was the default; now that the
    #: first block is, both are carried and `text` is one of them.
    last_text: str | None = None

    @property
    def parsed_answer(self) -> str:
        """dac's `parsed_answer` field: the text, or the invalid sentinel."""
        return self.text if self.format_correct and self.text is not None else INVALID_SENTINEL

    @property
    def differs_from_dac(self) -> bool:
        """False on the default policy, by construction: the default IS dac's."""
        return self.text != self.first_text

    @property
    def policies_differ(self) -> bool:
        """The two policies pick different text on this row, whichever one is active.

        On the recorded dac 128-cap cells this is True on 11 of 1611 Qwen vanilla rows,
        9 of 1611 Qwen DA rows and 0 of 1611 rows for either Gemma arm.
        """
        return self.first_text != self.last_text


def extract_answer(response_text: str, *, last: bool = False, first: bool | None = None) -> Extraction:
    """Apply the format gate. The default is the FIRST block, which is dac's and the paper's.

    `last=True` grades the last block instead, which is `dattn eval --last-answer`.

    `first` is a transition shim for callers still written against the old signature,
    where the default was the last block and `first=True` opted into dac's. A truthy
    `first` selects the first block, which is now also the default. A falsy or absent
    `first` means the caller expressed no preference, so `last` decides. Pass `last=`
    in new code. The runner and CLI already pass `last_answer`.
    """
    if first:
        last = False
    policy = "last" if last else "first"
    blocks = ANSWER_RE.findall(response_text)
    if not blocks:
        return Extraction(text=None, format_correct=False, n_blocks=0, policy=policy)
    first_text = blocks[0].strip()
    last_text = blocks[-1].strip()
    return Extraction(
        text=last_text if last else first_text,
        format_correct=True,
        n_blocks=len(blocks),
        policy=policy,
        first_text=first_text,
        last_text=last_text,
    )
