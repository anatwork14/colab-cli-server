"""DeclarativeFSM.

One instance per request, driven by the logits processor. It owns three things: the
rolling 500-character scan window, the current mode, and the mask that applies to the next
forward. Everything else is bookkeeping in `dattn.fsm.metrics.Ledger`.

State table:

| mode   | event                        | next   | mask                            |
|--------|------------------------------|--------|---------------------------------|
| GLOBAL | `<focus ...>`, gates pass    | FOCUS  | sink, spans, open tail          |
| GLOBAL | `<focus ...>`, a gate fails  | GLOBAL | unchanged, the attempt counts   |
| GLOBAL | `<local>` or `<answer>`      | LOCAL  | sink, open tail                 |
| FOCUS  | `</focus>`                   | GLOBAL | dense, and a success is counted |
| LOCAL  | `</local>` or `</answer>`    | GLOBAL | dense                           |

Everything else is inert. A `<focus ...>` inside `<local>` is not even counted as an
attempt, because LOCAL only ever looks for its own close tag. Nesting collapses on the
inner close, and the outer close is then a stray in GLOBAL.
"""

from dattn.contracts import DattnInvariantError, Mask, Mode, PromptLayout
from dattn.fsm.mask import DENSE, base_mask, focus_mask
from dattn.fsm.metrics import Ledger, full_metrics
from dattn.fsm.tags import (
    RE_FOCUS_CLOSE,
    RE_FOCUS_OPEN,
    RE_LOCAL_CLOSE,
    RE_LOCAL_OPEN,
    SCAN_CHARS,
    parse_focus_ids,
    scan_tail,
)

__all__ = ["DeclarativeFSM"]


class DeclarativeFSM:
    """The per-request state machine.

    `mask` is the mask that applies to the NEXT forward. `step()` records the ledger at
    `n_seen + 1` BEFORE applying the new mask, which is dac's convention and the one the
    published `attention_ratio` numbers use (DESIGN.md section 1.1).

    The scan window is seeded with `layout.prompt_tail`, so a prompt whose decoded tail
    already ends in a tag fires on the first generated token that carries a `>`. Both
    references do this, and it is a real prompt-injection surface for any template that
    ends near user text (`F 5.1`).
    """

    __slots__ = ("_base", "layout", "ledger", "mask", "mode", "n_seen", "text")

    mode: Mode
    mask: Mask

    def __init__(self, layout: PromptLayout) -> None:
        self.layout = layout
        self.mode = Mode.GLOBAL
        self.mask = DENSE
        self.n_seen = layout.n_prompt_tokens
        self.text = layout.prompt_tail[-SCAN_CHARS:]
        self._base = base_mask(layout)
        self.ledger = Ledger(len(layout.segments))
        self.ledger.format_error = layout.format_error

    # -- driving -----------------------------------------------------------------------

    def step(self, token_id: int, token_text: str) -> Mask | None:
        """Feed one sampled token. Returns the new mask when it changed, else `None`.

        Raises `DattnInvariantError` on a negative `token_id`, unconditionally and not
        behind a debug flag: that converts a silent-wrong-answer bug into a crash
        (DESIGN.md section 2.7). vLLM's async-scheduling placeholder is `-1`, and a caller
        that reaches here with one has skipped the "stop at the first negative" gate.
        """
        if token_id < 0:
            raise DattnInvariantError(f"negative token id {token_id}: the caller must stop at the first placeholder")
        self.ledger.record(self.ledger.total_steps + 1, self.mode, self.mask, self.n_seen + 1)
        self.text = (self.text + token_text)[-SCAN_CHARS:]
        # The scan runs only on a token whose decoded text contains `>`, and the
        # transition takes effect on the next forward (`main.tex:501`).
        changed = self._check_tags() if ">" in token_text else None
        self.n_seen += 1
        return changed

    def metrics(self) -> dict[str, float | int | str]:
        """dac's key set, so parity is a dict comparison. See `dattn.fsm.metrics`."""
        return full_metrics(self.ledger)

    # -- transitions -------------------------------------------------------------------

    def _check_tags(self) -> Mask | None:
        window = scan_tail(self.text)
        if self.mode is Mode.GLOBAL:
            match = RE_FOCUS_OPEN.search(window)
            if match is not None:
                return self._focus_open(match.group(0))
            if RE_LOCAL_OPEN.search(window) is not None:
                return self._local_open()
            return None
        if self.mode is Mode.FOCUS:
            if RE_FOCUS_CLOSE.search(window) is not None:
                self.ledger.note_focus_close()
                return self._close()
            return None
        if RE_LOCAL_CLOSE.search(window) is not None:
            return self._close()
        return None

    def _focus_open(self, raw_tag: str) -> Mask | None:
        """A loose `<focus...>` detection. The attempt is counted before the gates run."""
        self.ledger.note_focus_attempt()
        ids, fail = parse_focus_ids(
            raw_tag,
            self.layout.tag_keyword,
            len(self.layout.segments),
            self.layout.max_ids_per_focus,
        )
        if fail:
            self.ledger.note_focus_fail(fail)
            return None
        self.mode = Mode.FOCUS
        self.mask = focus_mask(self.layout, ids)
        spans = Mask.of(*(self.layout.segments[i - 1] for i in ids))
        self.ledger.note_focus_open(ids, self.mask.area(self.n_seen), spans.area())
        return self.mask

    def _local_open(self) -> Mask:
        self.mode = Mode.LOCAL
        self.mask = self._base
        self.ledger.note_local_open(self.mask.area(self.n_seen))
        return self.mask

    def _close(self) -> Mask:
        self.mode = Mode.GLOBAL
        self.mask = DENSE
        return self.mask
