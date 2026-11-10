"""Declarative Attention prototype: per-request, per-role, rollback-safe.

The vendor code lives in research/vendor/declarative-attention. This is a
control-plane prototype, NOT an installed vLLM attention-metadata patch.
Actual inference must fail closed until source and GPU proof exist.
"""
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import re
from typing import Callable, Dict, Optional, Tuple

VENDOR = Path(__file__).resolve().parents[2] / "research" / "vendor" / "declarative-attention"
GATES = ("vllm_029_source_hook_loaded", "target_hybrid_gdn_state_preserved",
         "processor_token_offsets_verified", "image_patch_offsets_verified",
         "absolute_kv_positions_preserved", "block_table_kernel_verified",
         "prefill_decode_continuity_verified", "kv_cache_not_evicted")
SPEC_GATE = "speculative_acceptance_rejection_replay_verified"
ROLES = ("target", "draft")


class DAQualificationError(RuntimeError):
    pass


def check_engine_proof(proof, speculative=False):
    if not isinstance(proof, dict):
        raise DAQualificationError("Missing exact DA vLLM engine proof")
    required = GATES + ((SPEC_GATE,) if speculative else ())
    missing = [key for key in required if proof.get(key) is not True]
    if missing:
        raise DAQualificationError("DA runtime not qualified: " + ", ".join(missing))
    sha = proof.get("vllm_engine_commit")
    if not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise DAQualificationError("DA proof requires exact 40-character engine Git SHA")
    return "DA_ENGINE_PROOF_PRESENT_GPU_EVIDENCE_STILL_REQUIRED"


def token_spans(spans, prompt_length):
    if type(prompt_length) is not int or prompt_length < 1:
        raise ValueError("Actual post-processor prompt token length required")
    if not isinstance(spans, dict) or not spans:
        raise ValueError("Explicit multimodal token-span mapping required")
    parsed = []
    for key, span in spans.items():
        if type(key) is not int or key < 1:
            raise ValueError("Chunk IDs must be positive integers")
        if not isinstance(span, (tuple, list)) or len(span) != 2:
            raise ValueError("Token span requires two positions")
        a, b = span
        if type(a) is not int or type(b) is not int or not (0 <= a < b <= prompt_length):
            raise ValueError("Chunk span must refer to actual processed token offsets")
        parsed.append((a, b, key))
    parsed.sort()
    if any(parsed[i][1] > parsed[i + 1][0] for i in range(len(parsed) - 1)):
        raise ValueError("Overlapping source chunk spans")
    return {key: (a, b) for a, b, key in parsed}


def load_vendored_machine():
    """Load only original vendored DA core without optional training dependencies.

    The upstream package initializer imports x_transformers, which is
    irrelevant to training-free inference. We directly load the exact
    checked-in core module; torch/einops/jinja2 remain runtime dependencies.
    """
    core = VENDOR / "declarative_attention" / "declarative_attention.py"
    if not core.is_file() or not (VENDOR / "LICENSE").is_file():
        raise DAQualificationError("Pinned upstream DA source/license missing")
    spec = importlib.util.spec_from_file_location(
        "_colab_cli_vendored_declarative_attention", str(core))
    if spec is None or spec.loader is None:
        raise DAQualificationError("Cannot load pinned upstream DA source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DeclarativeAttention


@dataclass
class RequestDAState:
    request_id: str
    spans: Dict[int, Tuple[int, int]]
    prompt_tokens: int
    da_roles: Tuple[str, ...]
    machine_factory: Callable
    tokenizer_decode: Optional[Callable] = None

    def __post_init__(self):
        if not self.request_id:
            raise ValueError("Missing per-request identity")
        self.spans = token_spans(self.spans, self.prompt_tokens)
        if not self.da_roles or len(set(self.da_roles)) != len(self.da_roles):
            raise ValueError("Each DA role must appear once")
        if any(role not in ROLES for role in self.da_roles):
            raise ValueError("DA role must be draft or target")
        self.accepted_tokens = []
        self.proposed_tokens = None
        self.machines = {r: self._replay([]) for r in self.da_roles}

    def _replay(self, tokens):
        machine = self.machine_factory(self.spans,
                                       tokenizer_decode=self.tokenizer_decode,
                                       strict=True, block_size=16)
        for token in tokens:
            machine.step(token)
        return machine

    def preview_draft(self, proposal):
        """Inspect temporary candidate state; commit nothing until verification."""
        if "draft" not in self.machines:
            raise DAQualificationError("Cannot draft-mask a target-only DA mode")
        if self.proposed_tokens is not None:
            raise DAQualificationError("Pending speculative proposal unresolved")
        if not isinstance(proposal, (list, tuple)) or not proposal:
            raise ValueError("Nonempty proposed token list required")
        if any(type(t) is not int or t < 0 for t in proposal):
            raise ValueError("Draft proposal must be token IDs")
        self.proposed_tokens = list(proposal)
        return self._replay(self.accepted_tokens + self.proposed_tokens)

    def commit(self, accepted, accepted_draft_count=0):
        if not isinstance(accepted, (list, tuple)) or not accepted:
            raise ValueError("Only actual target-verified tokens can be committed")
        if any(type(t) is not int or t < 0 for t in accepted):
            raise ValueError("Accepted stream requires valid token IDs")
        if type(accepted_draft_count) is not int or accepted_draft_count < 0:
            raise ValueError("Bad accepted proposal length")
        if self.proposed_tokens is not None:
            p = self.proposed_tokens
            if accepted_draft_count > len(p) or accepted[:accepted_draft_count] != p[:accepted_draft_count]:
                raise DAQualificationError("Accepted prefix does not match proposal")
            if len(accepted) > accepted_draft_count + 1:
                raise DAQualificationError("At most one target correction token allowed")
        elif accepted_draft_count:
            raise DAQualificationError("No draft proposal exists")
        # Rebuild parser from committed stream to prevent any rejected tag
        # or draft branch from permanently mutating KV visibility.
        next_tokens = self.accepted_tokens + list(accepted)
        machines = {r: self._replay(next_tokens) for r in self.da_roles}
        self.accepted_tokens = next_tokens
        self.machines = machines
        self.proposed_tokens = None

    def discard_draft(self):
        self.proposed_tokens = None

    def kept_blocks(self, role, sequence_length):
        if role not in self.machines:
            raise DAQualificationError("Role has no DA masking")
        if sequence_length < self.prompt_tokens + len(self.accepted_tokens):
            raise ValueError("Sequence shorter than committed prefix")
        return self.machines[role].get_kept_blocks(sequence_length)


class DAController:
    def __init__(self, machine_factory=None):
        self.factory = machine_factory or load_vendored_machine()
        self.requests = {}

    def register(self, request_id, spans, prompt_tokens, roles, tokenizer_decode=None):
        if request_id in self.requests:
            raise DAQualificationError("Duplicate request ID")
        state = RequestDAState(request_id, spans, prompt_tokens,
                               tuple(roles), self.factory, tokenizer_decode)
        self.requests[request_id] = state
        return state

    def unregister(self, request_id):
        self.requests.pop(request_id, None)
