"""dattn: Declarative Attention on vLLM.

Public re-exports only. Nothing here imports vllm or torch at module scope, so
`import dattn` is cheap and works on a CPU box (DESIGN.md section 1).
"""

from dattn.contracts import (
    INF,
    MAX_IDS_PER_FOCUS,
    MAX_INTERVALS,
    NULL_BLOCK_ID,
    SCAN_CHARS,
    SINK_TOKENS,
    BackendProfile,
    DattnBootError,
    DattnError,
    DattnInvariantError,
    DattnRequest,
    LayoutError,
    Mask,
    Mode,
    PromptLayout,
    Rendered,
    RenderError,
    Segment,
    StepRecord,
    StepRows,
)

__version__ = "0.1.0"

__all__ = [
    "INF",
    "MAX_IDS_PER_FOCUS",
    "MAX_INTERVALS",
    "NULL_BLOCK_ID",
    "SCAN_CHARS",
    "SINK_TOKENS",
    "BackendProfile",
    "DattnBootError",
    "DattnError",
    "DattnInvariantError",
    "DattnRequest",
    "LayoutError",
    "Mask",
    "Mode",
    "PromptLayout",
    "RenderError",
    "Rendered",
    "Segment",
    "StepRecord",
    "StepRows",
    "__version__",
]
