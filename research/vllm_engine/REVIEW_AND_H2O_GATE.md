# Technical-lead audit: DA, H2O and speculative vLLM 0.29.0

Date: November 10, 2026. Scope: colab-cli-server PR #38 and
anatwork14/h2o-vllm feature branch feat/h2o-vllm-029-port.
**Verdict: offline policy integration only; GPU H2O × DA BLOCKED.**

## Verified source identities

| Layer | Commit / dependency | Current qualification |
| --- | --- | --- |
| vLLM | v0.29.0, 98dff2a81d747d1dba01a47f939f48c3526d4206 | Pinned target API; new combinations untested on GPU |
| Original DA | itsnamgyu/declarative-attention, 24110b7b441ad9423f55952a2c02d23eac80ec52 | 178 exact source blobs, vLLM CUSTOM backend, metadata compaction |
| Original H2O | FMInference/H2O, ac75c2a8a9e76832b2a4139b9363373b56336bfb | HF/FlexGen original, not a vLLM 0.29.0 eviction port |
| h2o-vllm | anatwork14/h2o-vllm, feat/h2o-vllm-029-port | CPU score-selection policy and FlashAttention passthrough canary only |
| Qwen target | Qwen/Qwen3.5-4B | Real multimodal/GDN GPU parity pending |
| Draft methods | MTP, DFlash, EAGLE3 | Each needs real vLLM and verifier-acceptance tests |

Original DA changes which physical KV blocks are *read* by manipulating
attention metadata; it does not evict those pages or free allocator memory.
H2O ultimately seeks to physically reclaim cache using true heavy-hitter
attention statistics. Those must remain separate control and ownership planes.

## Findings and applied fixes

| ID | Severity | Confirmed problem | Change |
| --- | --- | --- | --- |
| R01 | P0 | Both DA and H2O pass-through register vLLM CUSTOM backend; last writer may win | H2O registration rejects active DA environment and already-owned CUSTOM backend; processes isolated |
| R02 | High | Server readiness command could contact a different healthy localhost port | Verify frozen port and exact loopback bind before health probe |
| R03 | High | Stop might race a wrong process group | Check frozen PID start ticks and correct owned PGID before SIGTERM |
| R04 | High | Benchmark results could be mislabeled as MTP/DFlash/EAGLE3 | Require explicit experiment-arm match, draft byte hash and frozen server receipt |
| R05 | High | Cached archive model could be labeled Qwen without original provenance | Require matching upstream HF repo and full 40-hex revision even for local archives |
| R06 | P0 scientific | H2O CPU selector/pass-through could be mistaken for real GPU eviction | Policy-only manifest validation; any H2O GPU serving request remains blocked |
| R07 | P0 scientific | DA FOCUS or LOCAL may return to GLOBAL and need KV evicted by H2O | Mark all physical reclaim blocked until proven no-reentry or exact history KV reconstruction |
| R08 | High | GQA head disagreements, partially protected blocks and GDN state complicate reclaim | CPU bridge intersects all head candidate sets, requires entire blocks and protects DA/evidence/sink/tail; rejects GDN, prefill and speculative cases |
| R09 | Medium | H2O documentation claimed benchmark repo contains only README | Corrected to reflect current research_bench implementation |

## Combined architecture (future, NOT running)

    Original VLM input + post-processor image/patch positions
        ↓
    Internal signals: DA chunks, grounded evidence, sink, response tail
        ├── DA read-visibility mask (custom metadata builder)
        └── Physical-KV protection mask ──────────────────┐
    Genuine per-layer/head attention probabilities       │
        ↓                                                │
    H2O heavy-hitter / recent-token policy               │
        ↓                                                │
    Full-block head intersection ── subtract protected KV┘
        ↓
    Physical reclamation gate [BLOCKED]
        ↓
    Future single-owner vLLM scheduler and cache manager

Never confuse the DA scratch compacted read-table with the original
physical KV block table. Reclamation must preserve scheduler refcounts,
prefix sharing, hybrid GDN state, RoPE absolute positions, chunked prefill,
graph capture, multi-request concurrency, cancellation and draft rejection.

**GLOBAL reentry is a fundamental conflict.** DA's state machine can
return from a narrow FOCUS mask to all historical KV. Eviction based only on
today's focus may make tomorrow's GLOBAL inference incorrect. Valid choices
require a justified irreversible-drop policy with clearly changed semantics
or exact re-materialization/recomputation, with its cost benchmarked.
Merely unioning importance masks does not prove this.

## Available code contract

The local research_ops/vllm_engine/h2o_bridge.py checks the exact versioned
h2o-vllm/1 policy receipt, original H2O commit and vLLM tag. It builds
non-executable conservative candidate block IDs after protecting DA
importance, grounded evidence, sink and active output tokens. It returns
physical_eviction_performed=false, physical_reclaim_authorized=false,
gpu_qualified=false, physical_kv_bytes_freed=null and
da_global_reentry_hazard=true.

A caller may not pass this CPU policy plan off as H2O runtime. H2O's
passthrough CUSTOM backend is incompatible with author DA in the same
process; keep each in an isolated GPU server until one canonical backend
implements both.

## Next engineering gates

G1: Reproduce author's DA source/contract suite under Python 3.13,
vLLM 0.29.0; boot a Qwen3.5-4B full-attention-only profile on an
explicitly approved A100/L4; preserve native GDN.

G2: Real MemLens/MMLongBench-Doc VLM post-processor image token mapping,
DA versus dense-tagged DA_nm versus vanilla parity, correct KV mask effect.

G3: MTP/DFlash/EAGLE3 each independently; verifier-accepted token callbacks
for target/draft/both, with rejection rollback and exact position accounting.

G4: Real H2O per-layer/head attention score capture; reference H2O selection
parity including grouped-query heads and score-collection overhead.

G5: Correct physical full-block eviction and reuse: refcounts, RoPE,
physical freed KV/VRAM, cache-prefix sharing, scheduling and CUDA graphs.

G6: One composable DA+H2O engine with dynamic GLOBAL reentry correctness,
correct multimodal and speculative behavior, then paired real benchmark
quality/latency/throughput with independent official dataset scoring.

**Do not merge or claim publication-quality GPU improvements before G1–G6.**
