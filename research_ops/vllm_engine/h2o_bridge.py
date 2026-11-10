"""Fail-closed integration seam for independent DA and H2O research methods.

The external anatwork14/h2o-vllm package currently implements a CPU policy
and FlashAttention registration canary, NOT real H2O physical KV eviction.
This module only performs conservative candidate filtering for a future
scheduler-owned allocator. It NEVER changes a vLLM block table, memory
allocation, logits, or performance metric.

Qwen3.5 Gated DeltaNet/state-space groups are ineligible. Original DA
controls which KV to READ; protected tokens are a separate stronger signal
for H2O's eventual resident-KV decision.
"""
from dataclasses import dataclass
import json
from pathlib import Path

PINNED_ENGINE = "0.29.0"
PINNED_H2O = "ac75c2a8a9e76832b2a4139b9363373b56336bfb"
PINNED_VLLM = "98dff2a81d747d1dba01a47f939f48c3526d4206"
SCHEMA = "h2o-vllm/1"


class H2OIntegrationBlocked(RuntimeError):
    pass


def validate_h2o_policy_receipt(receipt):
    """Reject claims of a GPU-qualified eviction engine from the current port."""
    if isinstance(receipt, (str, Path)):
        obj = json.loads(Path(receipt).read_text(encoding="utf-8"))
    else:
        obj = receipt
    if not isinstance(obj, dict):
        raise H2OIntegrationBlocked("H2O manifest must be a JSON object")
    source = obj.get("source")
    if not isinstance(source, dict) or (
        source.get("commit") != PINNED_H2O
        or source.get("repository") != "https://github.com/FMInference/H2O"
    ):
        raise H2OIntegrationBlocked("Original H2O code provenance is missing/mismatched")
    if (obj.get("schema_version") != SCHEMA
        or obj.get("engine_version") != PINNED_ENGINE
        or obj.get("engine_tag_sha") != PINNED_VLLM
        or obj.get("engine") != "vllm"):
        raise H2OIntegrationBlocked("Wrong H2O/engine integration contract version")
    if obj.get("mode") != "policy-only" or obj.get("method") != "h2o-policy-simulation":
        raise H2OIntegrationBlocked("Only CPU H2O policy receipts are supported for planning")
    # A declaration does not confer runtime authority, even if future code
    # accidentally marks the canary as qualified.
    if any(obj.get(key) is not False for key in (
        "runtime_verified", "attention_probabilities_collected",
        "physical_kv_eviction", "real_h2o_eviction", "gpu_qualified",
        "performance_claims_allowed")):
        raise H2OIntegrationBlocked("Unexpected H2O GPU/eviction claim in policy-only receipt")
    policy = obj.get("policy")
    if not isinstance(policy, dict) or not all(
        isinstance(policy.get(name), int) and not isinstance(policy.get(name), bool)
        for name in ("heavy_hitter_tokens", "recent_tokens", "sink_tokens", "block_size")
    ):
        raise H2OIntegrationBlocked("Malformed CPU H2O policy configuration")
    if policy["block_size"] <= 0 or any(policy[k] < 0 for k in (
        "heavy_hitter_tokens", "recent_tokens", "sink_tokens")):
        raise H2OIntegrationBlocked("Invalid H2O budgets")
    return obj


def _positions(values, n, label):
    if not isinstance(values, (list, tuple, set, frozenset)):
        raise ValueError(label + " must be an explicit sequence or set of token positions")
    seen = set()
    for p in values:
        if type(p) is not int or p < 0 or p >= n:
            raise ValueError(label + " contains invalid absolute token positions")
        seen.add(p)
    return seen


@dataclass(frozen=True)
class CandidatePlan:
    """A *non-executable* projection, not physical KV eviction permission."""
    status: str
    candidate_full_blocks: tuple
    protected_positions_count: int
    h2o_common_candidate_count: int
    filtered_candidate_count: int
    block_size: int
    attention_group: str
    physical_eviction_performed: bool = False
    physical_kv_bytes_freed: object = None
    gpu_qualified: bool = False
    da_global_reentry_hazard: bool = True
    historic_kv_reconstruction_verified: bool = False
    physical_reclaim_authorized: bool = False

    def to_dict(self):
        return dict(self.__dict__)


def plan_protected_eviction_candidates(
    *,
    manifest,
    sequence_length,
    per_head_h2o_evict_positions,
    da_important_positions,
    grounded_evidence_positions,
    current_response_positions,
    sink_positions,
    block_size,
    attention_group="full_attention",
    stage="decode",
    speculative_method=None,
):
    """Find block-aligned H2O candidates that protect all internal signals.

    All KV/query heads must independently nominate every token in a block.
    Partial final blocks remain resident. This is intentionally conservative:
    vLLM's actual block size may differ across hybrid-attention cache groups.

    DA can move from FOCUS/LOCAL back to GLOBAL and read all historic KV.
    Hence even filtered candidate blocks remain NONEXECUTABLE until a
    proven no-reentry or exact KV-reconstruction protocol exists.
    """
    validate_h2o_policy_receipt(manifest)
    if attention_group != "full_attention" or stage != "decode":
        raise H2OIntegrationBlocked(
            "H2O/DA cannot operate on GDN/other attention or during prefill")
    if speculative_method is not None:
        raise H2OIntegrationBlocked(
            "MTP, DFlash and EAGLE3 H2O/DA token acceptance is not implemented")
    if type(sequence_length) is not int or sequence_length <= 0:
        raise ValueError("Real, positive post-processor token length required")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("Use the native full-attention KV cache kernel block size")
    if block_size != manifest["policy"]["block_size"]:
        raise H2OIntegrationBlocked(
            "Selected H2O block size differs from frozen policy configuration")
    if not isinstance(per_head_h2o_evict_positions, (list, tuple)) or not per_head_h2o_evict_positions:
        raise ValueError("Every head must provide an explicit H2O candidate set")
    head_sets = [_positions(v,sequence_length,"H2O head") for v in per_head_h2o_evict_positions]
    common = set.intersection(*head_sets)
    protected = set()
    for label, values in (
        ("Declarative Attention", da_important_positions),
        ("Grounded evidence", grounded_evidence_positions),
        ("Response tail", current_response_positions),
        ("Sink tokens", sink_positions),
    ):
        protected.update(_positions(values, sequence_length, label))
    eligible = common - protected
    candidate_blocks = []
    for block_id in range(sequence_length // block_size):
        tokens = set(range(block_id*block_size,(block_id+1)*block_size))
        if tokens <= eligible:
            candidate_blocks.append(block_id)
    return CandidatePlan(
        status="PROTECTION_POLICY_ONLY_PHYSICAL_EVICTION_BLOCKED",
        candidate_full_blocks=tuple(candidate_blocks),
        protected_positions_count=len(protected),
        h2o_common_candidate_count=len(common),
        filtered_candidate_count=len(eligible),
        block_size=block_size,
        attention_group=attention_group,
    )


def reject_h2o_gpu_execution(*args, **kwargs):
    """Single hard guard used by launchers until real H2O eviction is qualified."""
    raise H2OIntegrationBlocked(
        "h2o-vllm currently provides policy simulation and CUSTOM pass-through only. "
        "There is no qualified physical H2O KV eviction to benchmark.")
