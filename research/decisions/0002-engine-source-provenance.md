# ADR-0002 — Run the actual modified inference engine

Status: ACCEPTED; on-GPU runtime verification still required.

Independent variable can be a changed vLLM inference engine rather than
different model weights. Installing a stock vllm==0.29.0 wheel next to edited
source is NOT evidence that modified code ran.

Supported modes:
- stock: exact version-pinned PyPI wheel, control arm.
- source_python: clean full source checkout at a committed SHA, editable
  install with compatible precompiled native libraries. Only Python changes.
- source_compiled: build full checkout for modified CUDA/C++ kernels.

research_ops.engine plan prints commands, never builds automatically.
research_ops.engine verify checks distribution version, actual Python import
origin, clean Git checkout and exact full commit SHA. Use --require-gpu to
also check visible NVIDIA GPU. Build and data-transfer timings remain outside
the timed model serving measurements.

No source checkout/submodule or integration with another project is created.
The operator can place its chosen full engine checkout in a separate runtime
working directory once authorized.

Source: https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
