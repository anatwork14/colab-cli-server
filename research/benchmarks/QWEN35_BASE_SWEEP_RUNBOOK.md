# Qwen3.5 0.8B and 4B base throughput: GPU agent runbook

Version: 2026-11-10. Frozen configuration:
research/benchmarks/qwen35_inference_sweep.json.
Entry points: research_ops/throughput/{plan,workload,runner,job,collect}.py.

**Offline code readiness is not a completed GPU benchmark.**
Only the Base arm can run after operator-approved allocation; H2O and
Declarative Attention (DA) remain blocked by their genuine GPU integration
gates.

## What “Base model” means

Use post-trained chat/VLM-capable models Qwen/Qwen3.5-0.8B and
Qwen/Qwen3.5-4B, NOT the distinct pretraining-only checkpoints with the
-Base filename suffix. Base here means NORMAL unmodified vLLM inference,
without H2O, Declarative Attention or speculative decoding. The workload
is TEXT ONLY on models capable of multimodal inference; it proves no
image evidence reasoning or accuracy on MemLens/MMLongBench-Doc.

## Protocol matrix

| Dimension | Frozen value |
| --- | --- |
| Models | Qwen3.5-0.8B, Qwen3.5-4B |
| Arms | Base, H2O+Base, DA+Base |
| Input context after chat template | 128 / 512 / 1024 / 2048 / 4096 / 8192 / 16384 |
| CCU | 1 / 2 / 4 / 8 / 16 / 32 |
| Measured requests per cell | max(16, 4 x CCU) |
| Warmup requests per cell | CCU; each must have distinct prompt |
| Output | exactly 256 tokens per request |
| Sampling | temp=0, min_tokens=256, max_completion_tokens=256, ignore_eos=true |
| Endpoint | /v1/chat/completions with SSE streaming |
| Thinking | disabled in server default and each request |
| Scheduler cap | --max-num-seqs 32 |
| Prefill context | --max-model-len 18432; 16K + 256 + chat overhead |
| Prefix reuse | --no-enable-prefix-caching |
| Input seeds | fresh random 63-bit seed per cell; unique per-request children |
| Arrival model | closed loop, at most CCU active HTTP calls; others queue |
| Official dataset scoring | none; this is randomized stress traffic only |

42 cells per model/arm. Six model-arm slices total 252 cells.
11,424 measured requests and 2,646 warmups = 14,070 requests for
the full grid. 84 BASE cells can be legitimately measured after GPU
preflight; 168 H2O/DA cells remain BLOCKED. The ~2h15 duration mentioned
by the owner is NOT measured, and can only be hypothesized for ONE
42-cell slice. Even a hypothetical constant 2h15 per slice would imply
~13h30 for all six, excluding startup, staging and failure handling.
Do not promise completion time or Colab continuity.

### Important input representation distinction

In vLLM 0.29.0 the chat-completions request contains messages, not an
arbitrary raw prompt_token_ids list. Our runner chooses original random
tokenizer IDs (excluding special IDs), decodes them into unique text,
then re-encodes using the real thinking-off chat template. It enforces
the requested EXACT EFFECTIVE CHAT PROMPT token count on every request
using server-returned usage. The sampled original ID list may differ
from actual post-template IDs. This benchmark is therefore
random-token-derived chat stress traffic, NOT raw-ID injection.
A strict raw-token-ID test would need a separate completions endpoint.

Our runner measures first GENERATED token, not the role-only SSE event
which some generic benchmark clients count as TTFT. It rejects any
request that does not deliver all 256 tokens, and reports TPOT as an
SSE streaming estimate rather than exact per-token GPU timing.

No prefix caching means no repeated-prefix reuse; normal in-flight
per-request KV caching is still required for standard inference.

## G0 — Offline preflight on dedicated Ubuntu server, no GPU

    python3 -m research_ops.throughput.plan
    python3 -m research_ops.throughput.plan --model qwen35_08b --arm base --context 128 --ccu 1
    python3 -m unittest tests/test_throughput_sweep.py -v
    python3 colab_server.py doctor
    python3 -m colab_adapter.cli project validate --manifest research/benchmarks/colab-project.json --json
    python3 -m colab_adapter.cli project plan --manifest research/benchmarks/colab-project.json --json

Resolve exact HF publisher model commits on dedicated host, metadata
only (not weights):

    python3 -m research_ops.vllm_engine.checkpoints resolve --model qwen35_08b --methods ""
    python3 -m research_ops.vllm_engine.checkpoints resolve --model qwen35_4b --methods ""

Freeze full model SHAs, license and actual downloaded source tree checksums
using existing research_ops.assets. Stage weights once per GPU VM to
/content/research-assets before timing, or restore verified content-addressed
host/Drive caches. Never read weights from Drive FUSE while measuring.
Do not silently substitute a different model, GPU, dtype or tokenizer.

The Colab project manifest syncs Python research_ops plus checkpoint
catalog and throughput protocol, not model weights, benchmark PDFs,
images or other large files.

## G1 — Explicit Colab GPU authorization

Reuse the EXISTING project approve-request / approve / up / sync
workflow from research/benchmarks/AGENT_RUNBOOK.md. An operator must
approve the requested GPU and time/CU budget. Prefer the same A100 40GB
type for both model sizes and all future treatment arms for comparable
hardware conditions. Availability and capacity are not guaranteed.

A Colab VM may disappear unpredictably; the dedicated server stays the
durable control plane. The existing project exec default timeout is 600s.
A GPU job can outlive an individual short exec only if detached process
continuity is proven on the real VM by separate subsequent CLI calls.
Never use keepalive or attempt to bypass Colab session limits.

## G2 — Start baseline server on the ALREADY approved GPU VM

From /content/colab-workspaces/vlm-two-dataset, after installing and
verifying vLLM==0.29.0 and staging model weights:

    python3 -m research_ops.benchmarks.serve start \
      --assets /content/private-assets.json \
      --model-asset REAL_QWEN08_ASSET_ID \
      --model Qwen/Qwen3.5-0.8B \
      --model-profile qwen35_08b --experiment-mode normal \
      --engine-mode stock --max-model-len 18432 \
      --max-images 1 --throughput-sweep \
      --run-id qwen08-base-server --execute

    python3 -m research_ops.benchmarks.serve status \
      --run-id qwen08-base-server --port 8000

Require READY_VERSION_VERIFIED and an authentic server_process.json:
    
    /content/research-vllm-servers/qwen08-base-server/server_process.json

The server source must be the exact pinned original Qwen checkpoint, not
a random local folder. The process receipt must show a live owned PID,
vLLM 0.29.0, max-num-seqs 32, max-model-len >= 18432,
--no-enable-prefix-caching, thinking off and zero speculative flags.
The 4B server uses Qwen/Qwen3.5-4B and qwen35_4b with a NEW run ID
after the 0.8B server has stopped; do not run both concurrently.

A configured CCU of 32 does NOT guarantee 32 simultaneous 16K resident
KV contexts. Inadequate KV blocks cause queueing, preemption or errors.
Never silently reduce CCU or invent the number of actual resident requests.

## G3 — One original stress pilot, exact measured protocol

Run first on the approved GPU VM:

    python3 -m research_ops.throughput.runner \
      --model qwen35_08b --arm base \
      --context 128 --ccu 1 \
      --server-receipt /content/research-vllm-servers/qwen08-base-server/server_process.json \
      --run-id qwen08-i128-c1-attempt01 --execute

Expected structural evidence (NOT invented performance metrics):
16 measured request receipts and 1 unique warmup; zero HTTP errors;
server prompt_tokens=128 and completion_tokens=256 for EVERY measured
request; no repeated input hash; honest TTFT and streamed timing.
Any missing/short response makes the cell INCOMPLETE_NOT_COMPARABLE.

To run a longer cell through short Colab CLI commands, use a bounded,
single-cell detached job in the already-running VM:

    python3 -m research_ops.throughput.job start \
      --model qwen35_08b --arm base --context 128 --ccu 1 \
      --server-receipt /content/research-vllm-servers/qwen08-base-server/server_process.json \
      --run-id qwen08-i128-c1-attempt02 --execute

    python3 -m research_ops.throughput.job status \
      --run-id qwen08-i128-c1-attempt02

This job DOES NOT allocate a GPU. It writes an owned process/PID
receipt, a runner log, and one write-once receipt per request.
First prove the detached job persists across TWO separate colab CLI
project exec calls before any multi-hour unattended work. GPU job
continuity is currently UNVERIFIED.

Next pilot 16384/CCU1, 128/CCU32, and finally 16384/CCU32, only
with explicit capacity and lease checks. Also smoke the 4B model
on the same GPU type before starting its 42-cell full grid.
Only ONE throughput cell may be active per managed GPU process,
protected by a fail-fast exclusive file lock.

## G4 — Durable evidence and collection

Cell output is staged on Colab:

    /content/qwen35-throughput-runs/<attempt-id>/
       attempt_manifest.json
       prompt_inventory.json
       requests/warmup/*.json
       requests/measured/*.json
       summary.json

Detached process receipts go under:

    /content/qwen35-throughput-jobs/<attempt-id>/

Copy finished cell artifacts to dedicated-host persistent storage
using the official Colab CLI download/sync procedure and rehash files.
Never assume Colab /content survives runtime destruction. Every
retry requires a NEW attempt ID and fresh random seed.

After the required cells have been downloaded to the host, run:

    python3 -m research_ops.throughput.collect \
      --root /mnt/qwen35-results \
      --csv /mnt/qwen35-results/verified-cells.csv \
      --report /mnt/qwen35-results/completeness.json

The aggregator refuses missing/failed requests and duplicate completed
attempts, forbids fabricated H2O/DA results and never reports MemLens
accuracy or hallucination from random stress prompts.

## G5 — The two unqualified treatment arms

H2O+Base: BLOCKED. anatwork14/h2o-vllm currently only implements
CPU heavy-hitter selection and FlashAttention pass-through, not
real GPU attention statistics or scheduler-owned physical KV eviction.
A pass-through is not a valid H2O runtime.

DA+Base: BLOCKED. Original authors' DA vLLM plugin is vendored, but
Qwen3.5-0.8B/4B visual/GDN token-to-KV offset mapping, attention mask
behavior and GPU correctness parity are not established. Do not label
a normal model as DA.

Review research/vllm_engine/REVIEW_AND_H2O_GATE.md. Real H2O eviction
also conflicts with DA future GLOBAL reentry unless exact historic
KV reconstruction/no-reentry semantics are proven.

The user can authorize a pilot AFTER review. Agents must not silently
merge, provision new paid Colab sessions, train, fine-tune, synthesize
official evaluation records, or report unmeasured results.
