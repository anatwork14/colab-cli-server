"""Guards for the opt-in parity_prefix profile on the installed vLLM 0.29.

Only dense prompt KV may enter the prefix cache. Generated KV depends on the
DA mask history, which vLLM's token hashes do not describe. Recompute preemption
is refused before freeing any blocks because DA does not replay masks in prefill.
"""

from __future__ import annotations

from functools import wraps

from dattn.contracts import DattnInvariantError


def prompt_only(original):
    @wraps(original)
    def cache_blocks(self, request, num_computed_tokens):
        # Do not relabel a later recurrent state as an earlier prompt checkpoint.
        # Prompt publication already happens during the original prefill steps.
        if num_computed_tokens > request.num_prompt_tokens:
            return None
        return original(self, request, num_computed_tokens)

    return cache_blocks


def refuse_preemption(self, request, timestamp, drop_stale_output=False):
    raise DattnInvariantError(
        f"parity_prefix refuses recompute preemption for {request.request_id}. "
        "DA mask history cannot be replayed by dense prefill. No sample is certified."
    )


def install_guards():
    """Install only inside explicitly selected parity_prefix engine processes."""
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator, KVCacheCoordinator
    from vllm.v1.core.sched.scheduler import Scheduler

    for cls in (KVCacheCoordinator, HybridKVCacheCoordinator):
        if not getattr(cls.cache_blocks, "_dattn_prompt_only", False):
            wrapped = prompt_only(cls.cache_blocks)
            wrapped._dattn_prompt_only = True
            cls.cache_blocks = wrapped
    Scheduler._preempt_request = refuse_preemption
