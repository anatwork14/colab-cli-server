# Agent Colab CLI research pilot: MemLens + MMLongBench-Doc

**Standalone** inference-engine research in colab-cli-server only. Do NOT
connect other experiment repositories. Do NOT train, fine-tune, create
synthetic benchmark records, or fabricate quality numbers.

## Readiness and scientific gates

- **MemLens**: original 789 question IDs per 32k/64k/128k/256k
  release; preserve conversation roles, dates, images and full context.
  Its official accuracy uses a separate LLM judge. Local string matching
  is a diagnostic proxy only.
- **MMLongBench-Doc**: upstream GitHub README currently lists 1091
  questions/135 PDF documents; NeurIPS paper and leaderboard versions
  have different question totals. Pin exact release and question-file
  SHA256. Official accuracy/F1 uses a separate answer extraction and
  three-stage scoring protocol, not a raw answer equality comparison.
- Audit all original question IDs and all referenced images/PDFs first.
  No failed/corrupt PDFs are dropped. A prior collection had two PDF
  integrity mismatches; those must be repaired/independently verified.
- Serve a real pinned model with the intended actual vLLM engine version.
  Change only one independent variable between baseline and candidate.
- Time TTFT/E2E in the streaming client. Its TPOT is an estimate from
  total completion tokens and the streaming window, NOT real per-token
  ITL. Client throughput includes request preparation. Official accuracy,
  F1 and hallucination remain NOT_MEASURED until original upstream
  evaluators are independently qualified and run.

Original evidence for protocols (URLs only, no imported repo dependency):
https://github.com/xrenaf/MEMLENS
https://huggingface.co/datasets/xiyuRenBill/MEMLENS
https://github.com/mayubo2333/MMLongBench-Doc

## Modules and configurations

- research_ops/benchmarks/datasets.py — data audit (reads real data only)
- research_ops/benchmarks/messages.py — faithful conversations/full PDFs
- research_ops/benchmarks/agent.py — offline two-dataset readiness plan
- research_ops/benchmarks/serve.py — launch local vLLM inside approved VM
- research_ops/benchmarks/runner.py — streaming requests and receipts
- research_ops/benchmarks/export.py — real predictions for separate judges
- research/benchmarks/colab-project.json — code-only Colab project sync
- research/benchmarks/campaign.example.json — INVALID until real assets set

## Phase 0: read-only planning on the Ubuntu dedicated server

    python3 colab_server.py doctor
    python3 -m colab_adapter.cli project validate --manifest research/benchmarks/colab-project.json --json
    python3 -m colab_adapter.cli project plan --manifest research/benchmarks/colab-project.json --json
    python3 -m research_ops.benchmarks.agent plan --campaign research/benchmarks/campaign.example.json

The final command is EXPECTED TO BE BLOCKED until real assets and model
weights are staged, checksum verified and audited. No cloud use in Phase 0.
Copy campaign.example.json to a private file and replace asset paths/IDs
with real pinned sources; never commit host-specific paths or tokens.

MemLens audit example (adjust actual staged root):

    python3 -m research_ops.benchmarks.datasets --kind memlens --root /content/research-assets/memlens-v1 --samples /content/research-assets/memlens-v1/dataset_32k.json --expected-count 789

MMLongBench-Doc audit example (PDF deep check required):

    python3 -m research_ops.benchmarks.datasets --kind mmlongbench_doc --root /content/research-assets/docs-v1 --samples /content/research-assets/docs-v1/samples.json --expected-count 1091 --deep-pdf

Use the exact expected count from the pinned input release (not a count
chosen to make an incomplete mirror look complete). PyMuPDF is a separate
optional Python dependency for PDF rendering and deep PDF validation.

## Phase 1: Colab GPU lifecycle — user-controlled allocation

Only after the owner approves the requested GPU, CU budget and lease:

    python3 -m colab_adapter.cli project approve-request --manifest research/benchmarks/colab-project.json --lease-seconds 3600 --json

The human operator then runs, with the actual returned approval ID:

    python3 -m colab_adapter.cli project approve APPROVAL_ID
    python3 -m colab_adapter.cli project up --manifest research/benchmarks/colab-project.json --lease-seconds 3600 --approval-id APPROVAL_ID --json

Only with the real session ID returned above:

    python3 -m colab_adapter.cli project sync --manifest research/benchmarks/colab-project.json --session-id SESSION_ID --json

The manifest requests A100 and syncs only local research_ops Python
files, NOT datasets/models. A100 availability and suitability are NOT
guaranteed. The lease guard is best-effort, not a billing guarantee.
Do not auto-provision via SSH, loop retries or bypass approval. Existing
project exec supports --shell and --working-dir, has default 600-second
timeout and is bounded by remaining lease.

## Phase 2: stage approved REAL assets on Colab local disk

Inside the authorized GPU VM, use preauthorized Drive mounts/verified
archives or pinned Hugging Face downloads. Always use exact checksums.

    python3 -m pip install huggingface_hub pymupdf
    python3 -m research_ops.assets stage --manifest /content/assets.json --asset REAL_MODEL_ID --scratch /content/research-assets --execute
    python3 -m research_ops.assets stage --manifest /content/assets.json --asset REAL_DATASET_ID --scratch /content/research-assets --execute

Invoke via project exec --shell with working directory
/content/colab-workspaces/vlm-two-dataset. Copy/extract on /content SSD
before timing. DO NOT stream model weights or PDF/image media from a Drive
mount during the measured part of the benchmark. A model and dataset
already staged and verified in this VM are reused across experiment cells.

## Phase 3: verified model serving (inside approved VM)

Install vllm==0.29.0 for baseline or a clean editable/compiled source
build for actual inference-engine modifications. Verify actual import
origin and full committed Git revision using research_ops.engine.

    python3 -m research_ops.benchmarks.serve start --assets /content/assets.json --model-asset REAL_MODEL_ID --model REAL_MODEL_SERVING_NAME --engine-mode stock --max-model-len 32768 --max-images 160 --run-id serve-001 --execute

Check readiness (not inferred from process launch):

    python3 -m research_ops.benchmarks.serve status --run-id serve-001

Only READY_VERSION_VERIFIED permits inference. Serving binds to localhost
(127.0.0.1), not public internet. For source-edited methods supply
--engine-mode source_python/source_compiled with --source-root and
--source-revision, and install the built engine before serving. MTP or
speculative settings can be passed using a strictly validated optional
--speculative-config-json. vLLM model support has NOT yet been GPU tested.

## Phase 4: two genuine question pilots; NO synthetic dataset

MemLens on ORIGINAL 32k release questions, a pilot of first 2 questions:

    python3 -m research_ops.benchmarks.runner --kind memlens --assets /content/assets.json --model-asset REAL_MODEL_ID --dataset-asset MEMLENS_ID --samples-rel dataset_32k.json --expected-count 789 --model REAL_MODEL_SERVING_NAME --max-images 160 --max-tokens 256 --concurrency 1 --limit 2 --run-id memlens-pilot-001 --execute

MMLongBench-Doc on actual full PDF page images (not evidence-only):

    python3 -m research_ops.benchmarks.runner --kind mmlongbench_doc --assets /content/assets.json --model-asset REAL_MODEL_ID --dataset-asset MMLONG_ID --samples-rel samples.json --expected-count 1091 --deep-pdf --max-pages 128 --dpi 72 --model REAL_MODEL_SERVING_NAME --max-tokens 256 --concurrency 1 --limit 2 --run-id doc-pilot-001 --execute

These are ENGINEERING PILOTS using real questions, not official score runs.
Remove --limit for full population after success, with an explicit new run
ID. More than max_pages, unsupported visual tokens, GPU OOM, PDF corruption
or missing images must produce failures, NOT quiet page/image truncation.
Do not compare dpi=72 engineering results against paper numbers where
different page-rendering settings were used. The source benchmark may need
more than 128 pages or max_model_len 32768; fail closed and inspect.

The runner creates one immutable JSON receipt per original sample ID,
writes engineering_summary.json and never labels proxy exact match as an
official score. A repeated command with --resume reads unchanged frozen
receipts. On resume, throughput for mixed time windows is NOT_MEASURED.

## Phase 5: official evaluation and artifact recovery

    python3 -m research_ops.benchmarks.export --kind memlens --run-root /content/research-benchmark-runs/memlens-pilot-001 --samples /content/research-assets/MEMLENS_ID/dataset_32k.json --output /content/memlens-export.json

    python3 -m research_ops.benchmarks.export --kind mmlongbench_doc --run-root /content/research-benchmark-runs/doc-pilot-001 --samples /content/research-assets/MMLONG_ID/samples.json --output /content/doc-export.json

Exports include complete question IDs, failed/missing items, reference
answers and raw predictions, marked AWAITING_OFFICIAL_EVALUATOR. The
MMLongBench exporter writes pred=null, score=null until the real answer
extractor runs. Run official scoring on a separately approved, qualified
judge setup; preserve rubric, prompt, model revision and judge receipts.

Before stopping the Colab VM, archive its run folder and copy it to the
dedicated server with the existing project exec and official Colab CLI
download command. Verify local hash against the remote output and only
then stop the owned session:

    python3 -m colab_adapter.cli project stop --session-id SESSION_ID --json

Never treat ephemeral /content as durable across Colab runtime releases.

## Agent safety and remaining blockers

1. Agents can inspect the repository, plan projects, audit real inputs,
   run offline tests and prepare pending approval requests.
2. An operator must approve actual Colab allocation, data transfer
   requiring quota, gated models, and any paid judge call.
3. No host from ChatGPT is connected to the dedicated Ubuntu server here:
   real server OAuth, GPU, free disk, Drive path and credentials are NOT
   verified by GitHub CI.
4. Per-question receipts survive within the current VM but multi-hour
   unattended GPU sessions need verified periodic cross-VM publication.
5. ITL/physical peak VRAM/AIPerf are not qualified here; true server-side
   token timing requires specialized telemetry and an explicit benchmark.
6. NO FULL_BENCHMARK_READY verdict until both dataset audits, real
   GPU serving, full population and official judge parity have receipts.
