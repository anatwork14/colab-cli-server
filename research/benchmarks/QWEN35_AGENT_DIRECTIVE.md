# Agent handoff: Qwen3.5 two-model, three-arm throughput setup

You are the designated GPU-execution agent working on
anatwork14/colab-cli-server on the dedicated Ubuntu server. Work from
the exact current feature/review branch, not guessed earlier summaries.
Read research/benchmarks/QWEN35_BASE_SWEEP_RUNBOOK.md and the frozen
qwen35_inference_sweep.json before acting.

Implement and evidence-gate 2 models (Qwen3.5-0.8B and Qwen3.5-4B
post-trained chat variants), 3 treatment arms (Base, H2O+Base,
DA+Base), 7 context sizes, CCU 1/2/4/8/16/32. Input must contain exact
post-chat-template tokens, output exactly 256 tokens, streaming chat,
ignore_eos=true, thinking off, no prefix cache, server
max-num-seqs=32. Each cell measures max(16,4*CCU) independent
prompts plus CCU independent warmups, with fresh recorded seeds.
This is a randomized stress workload, not an official accuracy dataset.

G0 — run all CPU tests and offline planner, verify 252 cells
(84 Base currently executable only after GPU preflight). Resolve real
publisher HF model SHAs, verify private cache source receipts, Colab CLI
doctor, synced files, requirements, available disk and chosen GPU.

G1 — do not allocate Colab without owner/operator approval of exact
GPU and lease. If absent, return OFFLINE_READY_ONLY.

G2 — stage Qwen3.5-0.8B on authorized GPU local SSD, verify
vLLM==0.29.0 and tokenizer. Launch Base-only throughput server.
Do NOT enable speculative decoding, H2O or DA plugins.
Prove live port, owned process, cache-disabled and full model source SHA.

G3 — run 0.8B input128/CCU1 test (16 successful measured and one
distinct warmup). Download and verify all receipts on dedicated host.
Prove detached job continuation across two independently executed
Colab CLI commands before long runs.

G4 — run 16K/CCU1, 128/CCU32 then 16K/CCU32 only if hardware
and approved runtime support it. Repeat initial smoke with 4B on
the same GPU type, sequentially after stopping the 0.8B engine.

G5 — when all gates pass and the operator has authorized resources,
execute the 42 Base cells per model, one at a time with unique attempt
IDs, fresh seeds, durable host checkpoint transfers after every cell.
No overlaid process, no silent retries or altered settings.
Aggregate with research_ops.throughput.collect and publish only
measured, fully verified metrics.

H2O+Base and DA+Base MUST stay BLOCKED until real, independent
vLLM GPU kernel/model correctness is established; never claim
H2O with a pass-through or DA with a stock engine.
Report ACTUAL_PASS, ACTUAL_FAIL and NOT_EXECUTED gates explicitly.
Do not train, fine-tune, manufacture quality labels, link unrelated
repositories or merge to main without review.
