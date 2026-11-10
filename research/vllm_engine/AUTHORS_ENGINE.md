# Canonical authors' Declarative Attention engine integration

Canonical primary source: https://github.com/itsnamgyu/declarative-attention
at commit 24110b7b441ad9423f55952a2c02d23eac80ec52 (MIT).

The original 178 files are copied verbatim to
research/vendor/author-declarative-attention. UPSTREAM_PROVENANCE.json lists
each source Git blob ID. Verified 178/178 against the original repository.
The earlier research/vendor/declarative-attention contains the independent
lucidrains prototype: SECONDARY REFERENCE ONLY, never our validated engine.

## What the authors actually provide

The canonical authors' dattn implementation pins vllm==0.29.0 and Python
3.13. It registers a CUSTOM attention backend through the
vllm.general_plugins entry point, maps only full_attention KV groups,
wraps the original metadata builder, leaves the full physical KV cache
resident, uses scratch block tables, and advances a request-local FSM
through a logits processor. It provides worker assertions, audit/record
RPC and actual GPU kernel tests.

Primary code locations under the copied source:
- src/dattn/engine/plugin.py: custom backend registration
- src/dattn/engine/backend.py: preserve the native attention backend
- src/dattn/engine/builder.py: metadata-builder compaction
- src/dattn/engine/compact.py and kernel.py: compacted block reads
- src/dattn/engine/processor.py: request-local output-driven FSM
- src/dattn/engine/state.py: request lifecycle, compacted state
- src/dattn/engine/worker_ext.py: RPC instrumentation
- tests/contract and tests/gpu: original proof and parity suite
- docs/DESIGN.md and docs/PROVENANCE.md: scientific protocol and limits

Upstream originally profiles Qwen3.6-27B and Gemma4-31B on SM100,
not Qwen3.5-4B on Colab A100 SM80 or L4 SM89.
Its published source explicitly treats speculative decoding as unsupported.
Neither upstream runtime success nor our CPU tests qualifies new GPU
combinations automatically.

## Qwen3.5-4B proposal, without false readiness

Research target: post-trained Qwen/Qwen3.5-4B. Native MTP is part of the
target; external draft candidates are z-lab/Qwen3.5-4B-DFlash and
yuyijiong/Qwen3.5-4B-Eagle3 (the latter documented for SGLang).
The source checkpoints need exact 40-char Hugging Face revisions, complete
weights and direct GPU validation under vLLM 0.29.0.

First verify the author snapshot on the dedicated server:

    python3 -m research_ops.vllm_engine.author_da verify-source

Plan a candidate A100 full-attention profile (not a GPU result):

    python3 -m research_ops.vllm_engine.author_da candidate-profile \
      --hardware a100 --backend TRITON_ATTN \
      --local-model /content/research-assets/qwen35-4b \
      --max-model-len 32768 \
      --output /content/author-qwen35-a100.toml

Alternative: --hardware l4 with explicitly chosen FLASHINFER or TRITON_ATTN.
These are *unqualified candidates*, not proven backend selections.

Only inside a user-approved Colab GPU running Python3.13 install the
original authors' package as editable from the included local path:

    python3.13 -m pip install -e research/vendor/author-declarative-attention

Then use its original dattn CLI with DATTN_BACKEND_PROFILES set to
the candidate TOML. First run probe --no-boot --json, then a real
probe that boots the engine and executes backend assertions. Do not
bypass A17 chat-template checks with --no-check or silently disable the
general plugin. Record the actual GPU, native attention backend, GDN KV
groups and resolved metadata-builder classes.

## Critical VLM and speculative blockers

1. MemLens and MMLongBench-Doc send real original images and PDF pages.
   Author dattn requires the rendered segment layout in vllm_xargs,
   matching real post-processor token positions. Our current OpenAI
   client does NOT yet provide verified image-patch token spans. Without
   this, a DA-registered server can serve images densely and falsely
   appear as a DA benchmark. Such a run is NOT qualified.
2. Preserve original physical KV, absolute positions, Gated DeltaNet
   recurrent state and all full-attention vs GDN backend distinctions.
3. For MTP, DFlash, EAGLE3, draft/target/both DA states may commit ONLY
   tokens accepted by the target verifier. Rejected proposals and
   aborted tags must roll back. Authors' default backend does not
   implement this speculative integration.
4. Actual model compatibility, prefill/decode parity, chunked context,
   GPU graph capture and per-request lifecycle require tests.
5. Compare author DA vs DA_nm (identical tagged prompts and dense
   attention), plus vanilla text baseline, before claiming attention
   computation benefits. No model training or synthetic benchmark data.

## Qualification status

- AUTHOR_SOURCE: PASS, original 178/178 Git blobs exact.
- PINNED VLLM: authors specify 0.29.0; installation not GPU-tested here.
- QWEN3.5_4B_SM80_SM89: PROPOSED, not boot-qualified.
- ORIGINAL MEMLENS/MM-LONG MULTIMODAL DA LAYOUT: NOT IMPLEMENTED.
- DA_NORMAL QWEN VLM MASK LOAD-BEARING: NOT QUALIFIED.
- DA + MTP (DRAFT/TARGET/BOTH): NOT IMPLEMENTED.
- DA + DFLASH (DRAFT/TARGET/BOTH): NOT IMPLEMENTED.
- DA + EAGLE3 (DRAFT/TARGET/BOTH): NOT IMPLEMENTED.
- OFFICIAL DATASET ACCURACY, F1, HALLUCINATION: NOT MEASURED.

Normal speculative profiles remain independently testable using the
existing plan/serve integration; a working draft model or source mask
alone is NOT evidence that the combined DA arm has been implemented.

## Technical-lead integration review

See [REVIEW_AND_H2O_GATE.md](REVIEW_AND_H2O_GATE.md). The H2O adapter is CPU-policy only; DA and H2O both register the singleton CUSTOM backend. Physical eviction must address DA GLOBAL history reentry, scheduler ownership and grouped-query KV blocks before any combined performance claim.
