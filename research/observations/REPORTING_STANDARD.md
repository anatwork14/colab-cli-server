# Observation reporting standard — VLM inference

**Audience:** manager / technical lead. **Deliverable:** one short Markdown decision brief per executed observation, named `<observation-id>-report.md`, using [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md). Target **one screen / ~250–400 words**; never sacrifice a material caveat to meet the target. The living observation plan is mutable; every executed run manifest, receipt and raw output is immutable.

## Publish rule (in priority order)

1. **Answer first:** say what changed, compared with which baseline, under which workload, and what decision follows. No lengthy background, model-selection survey, launch commands, or decorative matrices in the reader's path.
2. **Show only decisive evidence:** up to ~5 metrics with units, aggregation, denominators, quality status and measured coverage. A speedup is `baseline time / candidate time` **for matched requests**; distinguish mean of paired ratios, ratio of aggregate throughput, and ratio of medians. For accuracy, show absolute percentage-point differences, not relative percent.
3. **State scope next to claims:** distinguish `OFFICIAL_FULL_CONTEXT`, `ORACLE_EVIDENCE_PAGE`, `REDUCED_CONTEXT`, and `PERFORMANCE_STRESS`. A selected PDF page or image-only conversation is not an official full-context dataset evaluation. A correctness proxy is not an official score.
4. **Surface uncertainty:** report paired N, run repetitions, spread/CI when defensible, error counts and the largest limitation. If only one example has been tested, say *single-case pilot*; do not generalize. Failed, truncated, missing or excluded attempts remain visible in coverage.
5. **Link evidence instead of pasting it:** attach immutable run/config/data/model/draft digests; original question IDs and preprocessing policy; raw per-request traces; scorer version; GPU/driver/CUDA/vLLM; warmup, scheduler, cache, arrival and CCU definitions; accepted speculative tokens; independent audit and exact reproducible commands. The report can link to a technical appendix or logs without duplicating them.

## Evidence and claim labels

| Label | Meaning |
| --- | --- |
| `REPORTED` | Numbers supplied in an agent summary; primary artifacts not checked by this author. |
| `VERIFIED` | Primary result files, provenance, and arithmetic inspected and reconciled. |
| `INDEPENDENTLY_AUDITED` | A separate reviewer checked named immutable artifacts and documented a verdict. |
| `PLANNED` / `PILOT` / `REPLICATED` / `QUALIFIED` / `BLOCKED` | Execution stage, independent of the evidence label; a reviewed pilot is not full-dataset qualification. |

## Metric guardrails

- **Fairness:** compare paired original inputs, model/weights/processor and output policies, same GPU and controlled cache; state warmup, repetitions, errors and offered **versus achieved** concurrency. Record any change in output length or thinking mode.
- **Timing:** identify measurement endpoints for TTFT, E2E, TPOT/ITL and throughput. Specify p50/p95 if reported; do not call a usage-derived TPOT estimate a true inter-token measurement. Separate prefill-bound from decode-bound workloads.
- **Memory:** `torch.cuda.mem_get_info()` gives a device free-memory snapshot. Do **not** call it peak VRAM unless memory was sampled or otherwise instrumented over the relevant measurement window. Distinguish device-used, process-allocated, KV occupancy and model-only usage.
- **Speculation:** report *actual* accepted/proposed draft counts, histogram and fallback status; never infer acceptance rate from latency or claim image-aware drafting when the draft does not receive vision embeddings.
- **Quality:** preserve publisher scoring and full original inputs for official claims; mark local normalization, substring matching, answer extraction and LLM judging as separate protocols. Report parity separately from task correctness; greedy output mismatch is not automatically wrong.
- **Do not manufacture cells:** the example CCU×length matrix in the initial guide is illustration, **not measured data**. Report `NOT_MEASURED`, `INAPPLICABLE` or `BLOCKED` with the reason and real population coverage. Do not force all 42 proposed cells into a single management report.

## Workflow

For each living observation: preregister the experiment and comparison criteria → execute and preserve raw receipts → verify calculations/scoring/coverage → seek independent audit → publish the compact report and link evidence → append any new plan revision separately. Never rewrite earlier run results after a plan change. If not all gates pass, publish the partial finding explicitly as a pilot or blocker, not a full benchmark.

**Methodological references** (principles, **not** claims of compliance): [NeurIPS reproducibility checklist](https://nips.cc/public/guides/PaperChecklist), [MLPerf Inference rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc), [vLLM serving-benchmark metric semantics](https://docs.vllm.ai/en/latest/benchmarking/cli/).
