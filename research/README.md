# Standalone inference-engine research workspace

Scope: training-free evaluation and engineering of inference engines, including
vLLM Python and native-kernel modifications. **No training, fine-tuning, synthetic
benchmark-data generation, or linking to other GitHub research repositories.**

## Structure

    colab_adapter/           Existing Colab infrastructure (unchanged)
    research_ops/            Asset staging, source verification, run freezing
    research/
      questions/            Falsifiable questions and objectives
      literature/           Evidence and limitations of cited works
      decisions/            ADRs, including cache policy and engine provenance
      protocols/            Frozen asset and experiment definitions
      studies/              Study-level hypotheses and ablation matrices
      experiments/          Declared experimental cells, before GPU runs
      observations/         Evidence notes plus explicitly UNOBSERVED living plans
      results/              Published summaries only; never raw/large blobs
      agents/               Agent operating requirements
    tests/test_research_*.py  Offline CPU tests, no GPU/Drive API
    examples/                Older general CLI/training examples, NOT active research

## Responsibility separation

| Component | Source-of-truth | Is it proof of benchmark success? |
|---|---|---|
| Question/hypothesis | Version-controlled study document | No |
| Experiment protocol | Frozen config, exact IDs, seeds, scoring method | No |
| Model weights | Hub commit SHA or verified local tree/archive SHA256 | No |
| Dataset | Original dataset manifest, sample IDs and checksums | No |
| Engine source | Git SHA; local executable import provenance | No |
| Freeze manifest | Run-specific copy of verified inputs | No |
| Raw inference metrics and GPU logs | Durable host artifacts, optional Drive backup | Maybe, after audit |
| Observation/conclusion | Evidence-based result summary | Only when independently verified |

### Typical workflow

1. Define hypothesis and baseline/candidate comparison, metrics, warmup, token
   limit, concurrency, seeds, stopping rules, official scoring contract.
2. Create a private REAL assets JSON based on protocols/assets.example.json;
   fill model/dataset IDs, a full pinned Hugging Face commit or local file hash,
   and expected files. No placeholder SHA or synthetic data is treated as real.
3. Plan and stage on Colab local disk, for example /content/research-assets.
   The dedicated host retains authoritative copies; Drive is optional for
   durable archives. Streaming from a Drive mount while measuring inference
   is forbidden because of I/O variability.
4. Build the actually intended engine implementation (wheel baseline OR clean
   source checkout). Source-only Python changes require editable installation;
   C++/CUDA changes require recompilation. Verify that imports resolve to
   that very checkout and SHA.
5. Freeze run inputs using research_ops.experiments. This deliberately writes
   FROZEN_NOT_EXECUTED, not a fake PASS or inferred accuracy.
6. Use the existing colab_adapter for approved GPU session lifecycle and
   real serving. Keep heavyweight model and dataset files out of Git.
7. Recover GPU logs/result files and checksums to persistent server storage.
   Publish observations only after manual or independent evidence review.

### Standalone commands

    python3 -m research_ops.assets validate --manifest research/protocols/assets.example.json
    python3 -m research_ops.engine plan --mode stock

After replacing placeholders with a real pinned manifest:

    python3 -m research_ops.assets plan --manifest /abs/assets.json --asset real-model --scratch /content/research-assets
    python3 -m research_ops.assets stage --manifest /abs/assets.json --asset real-model --scratch /content/research-assets --execute
    python3 -m research_ops.assets verify --manifest /abs/assets.json --asset real-model --scratch /content/research-assets
    python3 -m research_ops.experiments plan --assets /abs/assets.json --experiment /abs/experiment.json
    python3 -m research_ops.experiments freeze --assets /abs/assets.json --experiment /abs/experiment.json --scratch /content/research-assets --output /content/research-runs --run-id test-001 --require-gpu --execute

The templates are intentionally NOT runnable until REAL pinned asset
definitions are added. All commands here are host/GPU-local and do not
allocate Colab sessions or trigger Google Drive API operations.

## Strong safety/provenance contract

- Stock wheel tests may not claim to contain modified engine patches.
- Modified engine tests must have clean, committed Git source AND verified
  editable/native build provenance, including actual imported vllm location.
- Receipt and source-manifest hashes prevent silent model/dataset swaps.
- Per-job result directories are write-once and are not in the Git checkout.
- Real GPU execution requires the existing adapter's scoped operator approval;
  neither research_ops nor an agent can allocate via its API.
- The older examples/train_model.py and examples/synthetic_dataset.py remain
  historical generic examples, not active data-science research methods.

Owner-requested evolving first benchmark: [Qwen3.5-4B Base versus EAGLE3 on MemLens and MMLongBench-Doc](observations/2026-11-10-qwen35-4b-base-vs-eagle3-memlens-mmlongbench.md). A plan is not an observed result; frozen executed run evidence stays immutable.
