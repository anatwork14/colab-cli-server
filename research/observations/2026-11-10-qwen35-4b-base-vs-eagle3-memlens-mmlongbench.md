# Living observation plan — Qwen3.5-4B Base vs EAGLE3

**Plan revision:** 0.1 | **Date:** 2026-11-10 | **Status:** PLANNED_NOT_EXECUTED / UNOBSERVED
**Location:** research/observations, by owner's request. This is a living plan, NOT a measurement.
**Companion:** [Machine-readable plan](qwen35-4b-base-eagle3-plan.v1.json)

The scope may change as research advances. Append every decision to the change log and increment the plan revision. Once a GPU run begins, its actual input, model, draft, engine, data, seed and evaluator commits, request-trace hash, and results remain **immutable** even if this planning document changes. No numerical performance or quality result has been observed.

## 1. Study and arms

**Research question:** How does EAGLE3 speculative decoding affect Qwen3.5-4B VLM serving performance on authentic MemLens and MMLongBench-Doc, while preserving original-task response quality?

| Arm | Serving method | Target | Draft | Qualification |
| --- | --- | --- | --- | --- |
| A0 | Base = ordinary decoding | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | None | GPU baseline preflight PENDING |
| A1 | EAGLE3 speculative decoding | Same exact Qwen/Qwen3.5-4B target | [yuyijiong/Qwen3.5-4B-Eagle3](https://huggingface.co/yuyijiong/Qwen3.5-4B-Eagle3) | vLLM 0.29.0 **multimodal** draft + target verification BLOCKED/UNQUALIFIED |

Base means **normal non-speculative inference** of the **post-trained** Qwen/Qwen3.5-4B, NOT the separately published Qwen3.5-4B-Base pretraining checkpoint.

The EAGLE3 checkpoint card documents a 50K-vocabulary small draft, target hidden-state features, and SGLang experiments on text tasks. None of those establish functioning vLLM 0.29.0 multimodal inference. The repository checkpoint catalog currently proposes EAGLE3 with 3 speculative tokens; do not treat this as a proven optimum or force an incompatible tokenizer/aux-state mapping. Record real parsed engine configuration, accepted/rejected tokens and any disabled-speculation fallback.

**Engine:** vLLM 0.29.0, upstream tag commit 98dff2a81d747d1dba01a47f939f48c3526d4206. Model and draft exact HF commits are PENDING. Pin actual executable import SHA, runtime version, kernel build, GPU, driver, CUDA, torch, precision and prompt template at execution.

**Deferred:** 0.8B, 2B, 9B, H2O, Declarative Attention, MTP, DFlash, DA×H2O and speculative DA. No training/fine-tuning. Do not integrate other research repositories into this initial experiment.

## 2. The only two official benchmark populations

| Dataset | Primary original source | Full quality population per arm | Official evaluation |
| --- | --- | --- | --- |
| **MemLens 32K** | [Author Hugging Face release](https://huggingface.co/datasets/xiyuRenBill/MEMLENS), [original evaluation code](https://github.com/xrenaf/MEMLENS) | 789 original questions, complete conversation roles/sessions/chronology/images and question types | Original LLM-as-judge accuracy and by-question-type accuracy; independent supported-answer / hallucination rubric only if qualified |
| **MMLongBench-Doc** | [Author Hugging Face release](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc), [original evaluation code](https://github.com/mayubo2333/MMLongBench-Doc) | 1,091 questions, 135 original PDFs, every page, figures, tables, evidence and unanswerable flags | Original answer extraction plus scoring, accuracy/F1 and cross-page, unanswerable, document/answer/evidence type analyses |

Initial MemLens scope is only the **32K release**. 64K, 128K and 256K are future separately frozen studies; do not silently mix context releases.

**Input integrity:** resolve publisher HF 40-char SHAs; verify every original JSON/Parquet question and source image/PDF with complete SHA256/tree receipts. Do not mix older GitHub questions with corrected HF evidence-page metadata. No silently skipped pages, missing images, altered questions, paraphrases, synthetic answers, or substitute PDFs. Failures and overlength examples stay in denominator with explicit labels.

## 3. Crucial distinction: quality versus performance

**Q — Official-quality evaluation** must use untouched real author inputs and full multimodal context. Run all 789 MemLens and 1,091 MMLongBench-Doc source QIDs **once per qualified arm** under matched temperature=0, verified thinking-disabled chat template and a declared maximum initial generation budget of 256 tokens. **Do not force ignore-EOS on this quality track**; stopping behavior and official author scoring must remain meaningful. Record truncations; if 256 is insufficient, preregister a new quality protocol before reruns. Run only the authors' qualified scorer: existing proxy exact match is not official accuracy. Compare paired QID predictions to identify unexpected EAGLE3 verification changes. Theoretical full population: 1,880 unique questions × 2 arms = **3,760 scored arm-question evaluations**; not a measured result.

**P — Real-data serving-performance evaluation** is a **separate fixed-output load test** using only original, unchanged dataset content. Run the following candidate grid:

| Input target (actual processed tokens) | CCU 1 | CCU 2 | CCU 4 | CCU 8 | CCU 16 | CCU 32 |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 16 | 16 | 16 | 32 | 64 | 128 |
| 512 | 16 | 16 | 16 | 32 | 64 | 128 |
| 1,024 | 16 | 16 | 16 | 32 | 64 | 128 |
| 2,048 | 16 | 16 | 16 | 32 | 64 | 128 |
| 4,096 | 16 | 16 | 16 | 32 | 64 | 128 |
| 8,192 | 16 | 16 | 16 | 32 | 64 | 128 |
| 16,384 | 16 | 16 | 16 | 32 | 64 | 128 |

Each table cell is measured request **attempts**, excluding **warmups = CCU**. Counts are max(16,4×CCU). Set max-num-seqs=32, disable prefix caching, serve via streaming chat, temperature 0 and verified thinking-off. Fixed output **256 tokens** with ignore_eos=true **only after proving that the actual vLLM chat endpoint honors it**. If it does not, BLOCK strict output-token comparability instead of misreporting early EOS.

**Counting:** 42 candidate cells per arm per dataset = **168 candidate cells** across 2×2 combinations; **1,904 measured attempts + 441 warmups** per arm/dataset; **7,616 measured attempts + 1,764 warmups** for all four arm/dataset pairs. These are **theoretical planning totals**, not proof that enough real inputs exist. If preregistering 3 paired independent repetitions, theoretical totals triple to 504 cells, 22,848 attempts and 5,292 warmups.

**Natural-length coverage:** input targets are *actual post-chat-template, post-multimodal-processor tokens*, not PDF page counts or text characters. Predeclare ±10% token bucket tolerance (proposal; freeze before first GPU execution), inventory genuine source inputs in each bucket and retain the full question/PDF/session context. Never truncate PDF pages, drop conversation turns, pad with random tokens, or fabricate prompt content under a dataset name. If original natural lengths cannot fill a bucket, label it **INSUFFICIENT_AUTHENTIC_INPUTS / INAPPLICABLE**, report real coverage and leave the cell unexecuted. MemLens 32K may have no natural examples in several 128–16K buckets. Repeated QIDs between separate load cells are tracked as repetitions, never counted as new official-quality samples.

**Pairing:** choose a deterministic sample-ID/order/seed trace per dataset × length × CCU × repetition. A0 and A1 consume **the same frozen requests**, with identical warmup policy and timestamp boundaries. Change seeds only between independent repetitions; do not vary content between methods. Prefix-cache-disabled is not equivalent to disabling normal GPU page reuse. CCU is offered concurrency; actual scheduler residency can be lower and requests must be counted while queued.

**Existing implementation gap:** current research_ops.benchmarks.runner caps warmups at 10; this plan needs 16 and 32. It also lacks a complete 42-cell sweep orchestrator, strict chat ignore-EOS verification, p99/true per-token timestamps, full GPU/KV telemetry and qualified official scoring. **These are blocking engineering tasks, not existing features.**

## 4. Metrics — no fabricated results

| Area | Required measurements or interpretations |
| --- | --- |
| **Quality** | Official MemLens judged accuracy + by-question-type; original MMLongBench-Doc accuracy/F1 + cross-page, unanswerable, answer format, PDF/document and evidence type; QID coverage and score differences |
| **Hallucination** | False-support / unsupported-claim rate only with an explicit, validated judge rubric; unanswerable false assertion rate; judge agreement/coverage and failures |
| **TTFT** | Client first generated token after request submit, **including queue wait**, p50/p95/p99 milliseconds; differentiate non-token SSE events |
| **TPOT / ITL** | True token interval timing p50/p95/p99 if instrumented; otherwise clearly name client-derived (last−first)/(completion_tokens−1) **TPOT estimate**; never mislabel SSE chunks as tokens |
| **Latency** | End-to-end p50/p95/p99; breakdown admission, queue, prefill, decode and serialization wherever measured |
| **Throughput** | Completed requests/s, output tokens/s, input+output tokens/s; offered CCU versus achieved CCU and scaling efficiency |
| **GPU / KV** | Process and device VRAM peak, torch allocated/reserved, KV-cache physical occupancy/high-water, preemptions/recomputations, OOM, telemetry source and sample interval |
| **EAGLE3** | Draft proposed tokens, verifier-accepted tokens, acceptance ratio/accepted length, rejected positions, target verification steps, draft compute/VRAM overhead and any fallback or disabled speculative steps |
| **Reliability** | Every attempt, success, failure, timeout, invalid-input, cancellation, HTTP error, OOM, unsupported input, output-token-count compliance |
| **Paired comparison** | Base/EAGLE3 TTFT/TPOT/E2E speedup, EAGLE3/Base tokens-per-second throughput speedup, memory delta, official-score delta, confidence intervals across frozen independent paired runs |

Primary scientific condition: no material official quality regression **under a preregistered non-inferiority tolerance** while obtaining a real measured serving benefit. Do not invent a numeric non-inferiority margin or claim a speedup in advance.

## 5. Execution gates and status

| Gate | Requirement | Status |
| --- | --- | --- |
| G0 | Publisher dataset and target/draft HF full SHA + original byte receipts; all source QIDs/PDFs/images | PENDING |
| G1 | Verified real vLLM 0.29.0 and target multimodal model import/source, CUDA, GPU, no accidental DA/H2O plugin | PENDING |
| G2 | Qwen Base real MemLens image and MMLongBench PDF-page pilot on correct chat template | PENDING |
| G3 | EAGLE3 draft full/vocab/hidden-state compatibility in vLLM 0.29.0 **with actual vision inputs** | **BLOCKED_UNQUALIFIED** |
| G4 | Lossless target verification and rejection/acceptance trace; no hidden fallback to normal | PENDING_G3 |
| G5 | Full official dataset scoring workflows + calibrated hallucination evaluation | PENDING |
| G6 | Real-data token length inventory and full streaming/load/ignore-EOS/cache/telemetry runner | PENDING |
| G7 | Operator-approved GPU, paired short pilot, immutable server/asset/workload receipts, then full sweep | NOT_RUN |
| G8 | Full durable results transfer to Ubuntu server, independent audit, uncertainty and failure analysis | NOT_RUN |

Use the dedicated Ubuntu server as persistent control plane and Colab GPU only after explicit owner approval; no GPU is provisioned by this plan. A0 may proceed independently if A1 is blocked, but A0-only data do **not** establish an A0-versus-A1 performance comparison. Never convert EAGLE3's published SGLang text numbers into a result of our vLLM 0.29.0 multimodal setup.

## 6. Reproducibility and run artifact contract

- Same model revision/weights, tokenizer, BF16 if supported, GPU, driver, vLLM source SHA and engine flags for A0/A1. Same temperature, thinking setting, input and output control per track.
- max_model_len must fit actual processed prompt tokens **plus** output 256 and chat/visual overhead; a nominal 16,384 input token request cannot run in a model configured for exactly 16,384 total sequence tokens.
- Record vLLM scheduling/max-batched-tokens, max-num-seqs=32, GPU memory utilization, prefix cache disabled flag, cache type, speculative config, full local model and draft tree SHA.
- One write-once run ID per dataset, arm, track, seed, input bucket and CCU. Save immutable source version snapshot, assigned QID list, warmup list, per-request streaming times/usage/output/status, GPU telemetry, error receipts, official judge raw responses and aggregation.
- Aggregate completed request metrics with elapsed timing windows declared; **never remove failures from the attempted denominator**. Mark missing engine metrics UNKNOWN instead of 0. On resume with disjoint timing windows, do not fabricate aggregate tokens/sec from combined records.
- Initial one-seed engineering run is a pilot; three independent, matched seeds are a proposal for later reportable comparisons, not an observed repetition count.

## 7. Evolving observation / decisions

| Date | Revision | Decision and impact | Evidence |
| --- | --- | --- | --- |
| 2026-11-10 | 0.1 | Narrowed prior 0.8B+4B, Base/H2O/DA scan to **Qwen3.5-4B A0 Base vs A1 EAGLE3** on **MemLens + MMLongBench-Doc only**; kept context/CCU/latency/throughput/VRAM metrics; separated authentic quality from fixed-output load; marked EAGLE3 VLM and runner gaps explicitly | Repository source and publisher links; **NO GPU RUN** |

**Next revision:** append observed evidence and new decisions rather than rewriting old rows; increment 0.x; never modify frozen run results to fit a changed plan. Actual experiments will get separate evidence-backed observation entries following [TEMPLATE.md](TEMPLATE.md).

**Current verdict: PLANNED_NOT_EXECUTED.** All quality/latency/throughput/VRAM/acceptance results = NOT_MEASURED.
