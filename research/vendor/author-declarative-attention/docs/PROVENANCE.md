# Provenance and attribution

`dattn` is a clean public implementation of Declarative Attention on vLLM 0.29.x. It is written
from scratch against a written specification (`docs/DESIGN.md`). It is not a fork of, and contains
no copied source from, either prior implementation. This document records what was learned from
each, what is an independent reimplementation, and the attribution text to ship.

Both prior implementations are the work of the same author team as this one, and `dattn` is the
team's shared public artifact rather than a third-party derivative of anyone's private code.

## 1. The two prior implementations

| | `dac` | `SemanticSparser` / package `semsparse` |
|---|---|---|
| author | the paper's authors | the paper's authors |
| version read | private repository, HEAD `fb1ef99` | private repository, `main` and a development worktree at HEAD `225f7b5` |
| role | the reference implementation the paper's numbers were produced with | an independent clean-room rewrite of the same paper |
| vLLM | `0.20.*`, resolves to 0.20.2 | `vllm==0.20.2` exact |
| licence file present | no | no |

Both were read **read-only**. Nothing was written to, built in, or executed inside either tree
during the design work. Where code had to be run for verification, it was copied to a scratchpad
first.

## 2. What was learned from `dac`

`dac` is the source of the *semantics*. Everything below is a behavioural specification that `dattn`
reimplements, not code that `dattn` reuses.

- **The tag grammar.** Four end-anchored regexes with an eight-character tail,
  `<focus ...>` / `</focus>` / `<(?:local|answer)>` / `</(?:local|answer)>`, case sensitive, scanned
  only on a token whose decoded text contains `>`, over a 500-character rolling window, with the
  transition taking effect on the next forward.
- **The state machine.** Three modes, FOCUS opening only from GLOBAL, `<answer>` as an alias of
  `<local>`, `<global>` inert, nesting collapsing on the inner close, a focus tag inside `<local>`
  invisible even to the counters, and every failure leaving the request in GLOBAL.
- **The decline rules and their gate order**, and the five index-path failure reasons
  (`segment_format_error`, `syntax_error`, `empty_index_list`, `more_than_three_indices`,
  `out_of_range_index`).
- **The mask composition**: a 16-token sink, a local window from the question through the end of the
  prompt, the entire response so far, plus the named segment spans in FOCUS.
  `DAC_ATTENTION_SINK_SIZE = 16`, `MAX_SEGMENTS_PER_FOCUS = 3`, success counted on `</focus>`.
- **The segmenter**: target 2048 tokens, hard cap 2560, a five-level delimiter ladder, greedy
  packing, contiguous ids from 1, an `<empty_context>` placeholder for an empty context.
- **The prompt formats**: `chunk_multi_turn_tool` (the paper's headline `get_magic_chunk` tool
  transcript with `Magic Chunk N` bodies and the `magic_chunks` tag keyword) and
  `chunk_single_turn_xml` (the inline `<magic_chunk index="N">` ablation).
- **The metric key set**, which `dattn` emits verbatim so that parity is a dict comparison.
- **The evaluation pipeline shape**: the 15 subsets, the 128-row cap, the seed, the context filter,
  the `<answer>` extraction gate, the judge prompt and the macro-15 aggregation.
- **Recorded outputs used as test data.** `dattn`'s FSM fixtures were regenerated as documented in
  `tests/fixtures/fsm/README.md` from dac's `responses.jsonl` files, which are *data*: recorded model
  responses plus dac's own computed metrics. They are the oracle for the parity harness. No dac Python
  source file and no dac test file is copied.

**Design elements of `dac` that `dattn` deliberately does not adopt**, each for a reason recorded in
`docs/DESIGN.md`:

- the dense `(max_num_seqs, max_model_len)` bool mask buffer, 256 MB device plus 256 MB pinned per
  TP rank,
- in-place mutation of vLLM's own block table,
- the `sitecustomize` PYTHONPATH hook,
- the Triton `num_stages` launch proxy,
- the per-family span-recovery regexes keyed on the tokenizer name, which silently truncate a
  segment span when the context quotes ChatML tool markup,
- the serve-time re-derivation of the layout from token ids, which costs seconds of engine-core CPU
  per request and repeats on every preemption.

The step 3 evaluation campaign vendors two byte-identical dac chat templates as data.
Their provenance and sha256 pins live in `src/dattn/prompt/template_policy.py`.
The bundled Qwen template drops the tool declaration, deliberately reproduced for score parity.
The stock rendering policy remains available.

## 3. What was learned from `SemanticSparser`

`semsparse` is the source of the *engine integration shape*. Again, specification not code.

- **Block-table compaction as the mechanism**, applied inside `builder.build()` on a shallow copy of
  `CommonAttentionMetadata` re-pointed at private per-builder scratch, so no vLLM-owned tensor is
  ever mutated.
- **The interval mask encoding**: at most five sorted, disjoint, non-touching half-open intervals per
  row, with an open end sentinel, carried in small int32 device buffers with pinned host mirrors.
- **The compaction plan**: outward block rounding, a running-maximum dedup that collapses ranges
  which touch only after rounding, prefix sums to place each kept run, and a fused Triton gather
  that replays the plan per row in a `static_range` loop with int64 arithmetic throughout.
- **The activity gates**: decode-only, not-dense, non-empty, and a separate `is_prefilling` gate that
  closes the one-token-final-prefill-chunk hole.
- **The `tail_kept` latent defect**, which `dattn` fixes rather than inherits: computing "is the last
  block kept?" from the rounded end of any interval rather than from the last positive-length piece
  is wrong exactly when an interval starts at or beyond `seq_len`. `dattn` computes it from the last
  positive-length piece and asserts the invariant.
- **Disabling the `update_block_table` fast path** on the builder, without which the compaction
  reaches only the first of several same-spec KV-cache groups while the shared shrunken `seq_lens`
  reaches all of them.
- **Refusing unsupported configurations loudly at boot** rather than degrading silently: pipeline
  parallelism, context parallelism, micro-batching, speculative decoding, cascade attention, the V2
  model runner, windowed or subclassed KV-cache specs.
- **Cudagraph safety**: allocate the scratch once, never move it, always slice from index 0, and
  stage an identity block table during capture.
- **The layout travelling in `SamplingParams.extra_args`** instead of being re-derived in the engine.
- **Template-agnostic span recovery by prefix render plus binary search**, validated by body
  containment, which is structurally immune to context content.
- **Test discipline**: a large CPU suite plus a small opt-in GPU suite, a contract test that pins the
  vLLM internals the integration depends on, and a needle test that proves the mask is load-bearing.
- **CUDA-event step timing inside the worker**, with the stop event closable in either
  `execute_model` or the `sample_tokens` wrapper because vLLM splits the two calls.

**Elements of `semsparse` that `dattn` deliberately does not adopt**:

- the `BarrierScheduler`, which breaks loudly on vLLM 0.29.0 and is unnecessary for the merged
  protocol,
- monkeypatching `AttentionGroup.create_metadata_builders` and rebinding builder methods, both
  replaced by the official `register_backend` plus `backend_per_kind` path that 0.29.0 added,
- the batch-sum-only device record of attended tokens, replaced by a per-row ring buffer,
- `tool_schema: false` on the Qwen profiles,
- the `magic_chunk` style name, which maps to the inline xml ablation rather than to the paper's
  headline format.

## 4. What is an independent reimplementation

Everything in `src/dattn` and `tests/`. Specifically:

- `dattn.fsm`: written against an internal behavioural table (FSM semantics diff),
  which established by execution that the two prior state machines agree on 56 adversarial cases
  across five tokenizers and three layouts plus 3800 fuzz cases. That table, not either codebase, is
  the specification.
- `dattn.prompt`: written against `main.tex` and the same table. Stock rendering asserts a
  tool declaration for both model families. Step 3 evaluation explicitly selects the vendored
  dac templates, including Qwen's omission, to match the prompts dac served.
- `dattn.engine`: written against vLLM 0.29.0's own extension points. The registration path
  (`register_backend` with `AttentionBackendEnum.CUSTOM`, bound per KV-cache kind through
  `AttentionConfig.backend_per_kind`) does not exist in 0.20.2 and therefore has no counterpart in
  either reference.
- `dattn.metrics`, `dattn.eval`, `dattn.bench`: written against the merged protocol in
  an internal wall-time specification derived from the paper (section 6) and an internal hook
  analysis (sections 3 and 6). Neither prior implementation implements
  the paper's protocol.
- FSM fixtures were regenerated from recorded dac outputs as documented in
  `tests/fixtures/fsm/README.md`. No fixture file is copied from either repository.

Where a constant is a specification (16-token sink, 2048/2560 segmentation, at most three ids per
focus, five mask intervals, a 500-character scan window, an eight-character tag tail), it is
reproduced because it defines the behaviour, and each one is cited to `main.tex` or to the
behavioural table in the source comment next to it.

## 5. Attribution and licensing

`dattn` is written by the same team that wrote the paper and both prior implementations. The method,
the protocol and the evaluation design were developed for the paper. The engine integration shape
that `dattn` follows was developed in `SemanticSparser`. Both prior implementations come from the
paper's authors, publishing this as one project, so `dattn` records both as a matter of provenance
and not of permission. The team decides together what `dattn` ships and under what terms.

`dattn` is a reimplementation rather than a port because the design targets vLLM 0.29.x extension
points that did not exist in 0.20.2, and because a specification-driven rewrite is the only way to
get a parity harness that can adjudicate between the two references. It is not a reimplementation
for legal reasons.

**Repository state as of 2026-09-16.**

| tree | LICENSE / COPYING / NOTICE | licence headers in source | `license` key in `pyproject.toml` |
|---|---|---|---|
| `dac` | none | none | none |
| `SemanticSparser` | none | none | none |
| `SemanticSparser` development worktree | none | none | none |
| `dattn` | `LICENSE` (MIT) | none | `MIT` |
| `vllm` 0.29.0 | Apache-2.0, with `SPDX-License-Identifier: Apache-2.0` headers throughout | yes | yes |

Actions before the first public commit:

1. **Resolved: `dattn` is released under the MIT licence (`LICENSE` at the repository root).**
   The original note follows. **`dattn` needs its own licence file, and the team chooses it.** No repository in the table above
   carries one today, so publishing `dattn` means picking terms for the first time rather than
   inheriting any. Add a `LICENSE` at the repository root and an SPDX header in each source file
   once the choice is made. Apache-2.0 is the natural default because it matches vLLM and imposes
   nothing on downstream users, but that is a recommendation, not a decision. This is open decision
   O11 in `docs/DESIGN.md`.
2. **Whatever `dattn` ships under, the two prior trees should get the same treatment or an explicit
   note saying they stay private.** Leaving them with no licence file while `dattn` is public is a
   loose end for anyone who later wants to reproduce the paper's numbers from `dac`.
3. **The recorded dac outputs contain third-party dataset content** (questions, rubrics and model
   responses derived from LooGLE, ZeroSCROLLS, RULER, LongBench and the other subsets of the
   benchmark table). Before any fixture derived from them is committed, check each source dataset's
   licence and redistribution terms. Prefer fixtures that store only token ids, spans, modes and
   metrics, with no verbatim context text, which is sufficient for the parity harness and avoids the
   question entirely. Any fixture that does carry source text must name its dataset and licence in
   the fixture header.
4. **vLLM is a dependency, not vendored.** No vLLM source is copied into `dattn`. Where a docstring
   or a contract test quotes a vLLM line, it is a short citation for identification, with the file
   and version named.
5. **Model weights and chat templates** are pulled from the Hugging Face Hub at run time and are
   subject to their own licences (Gemma's terms of use, Qwen's licence). `dattn` ships neither, and
   the golden files record only sha256 digests of the rendered templates, not their contents.

## 6. Attribution text

To ship in `README.md`, and in the paper's artifact statement.

> `dattn` is the reference public implementation of Declarative Attention on vLLM, from the authors
> of the paper. It was written from a written specification and contains no source code from the
> two earlier implementations the authors built while doing the research.
>
> The declarative-attention protocol, the tag grammar, the state-machine semantics, the segmenter,
> the prompt formats and the evaluation protocol are from the paper "Declarative
> Attention", and are reimplemented here from that paper and from a behavioural specification
> derived from the `dac` reference implementation.
>
> The engine-integration design, in particular block-table compaction with an interval mask
> encoding, the fused compaction kernel, disabling vLLM's `update_block_table` fast path, the
> refuse-loudly configuration policy and the contract-test-plus-needle-test discipline, follows the
> approach taken in `SemanticSparser`, reimplemented here against vLLM 0.29.x's
> official extension points.
>
> Built on vLLM (Apache-2.0). This project is not affiliated with or endorsed by the vLLM project.

A short form, for the head of each engine source file:

> Reimplementation. Semantics per docs/PROVENANCE.md. No code copied from prior implementations.

## 7. Changes made deliberately against both references

Listed here so that reviewers of all three trees can reconcile them, and so the parity harness's
known-delta list has a single home.

| # | change | reason |
|---|---|---|
| P1 | span recovery by prefix render, never by per-family regex | dac silently truncates a segment span when the context quotes ChatML tool markup |
| P2 | `len(spans) == n_segments` asserted at render time and the result shipped in the layout | closes the same failure at serve time, which neither reference does |
| P3 | `max_ids_per_focus <= MAX_INTERVALS - 2` validated on the request | a request-controlled value of 4 crashes the engine core inside `update_state` in `semsparse` |
| P4 | `tail_kept` computed from the last positive-length piece, plus an asserted invariant | the **interval-based** reference `semsparse` uses `run_K == nblk` (`semsparse/engine/compact.py:84`, `semsparse/engine/kernel.py:89`) and is wrong when an interval starts at or beyond `seq_len`. `dac` has no interval encoding and is **not** affected: it tests the last block directly at `dac/src/dac/vllm_adapter/patching.py:313` |
| P5 | pad tokens counted in `total_attendable` | matches the paper's definition, dac skips them |
| P6 | stock rendering asserts the tool declaration for both families, campaign evaluation defaults to dac_bundled | the bundled Qwen template omits the schema. Step 3 reproduces this omission and records the template hash |
| P7 | xml segment ids must be contiguous **and** ascending | dac compares `sorted(ids)` to the range (`dac/src/dac/focus/base.py:80`), so `2, 1` is accepted there. dac still **resolves by id**, through a dict keyed by `segment_id` (`base.py:82`, lookup at `:177`), so its mapping stays correct. dattn refuses the input instead, because out-of-order ids mean the detector drifted |
| P8 | per-row, block-rounded, post-compaction attended counts read from the device without a per-step sync | neither reference has this, dac's cheap number is token-exact and `semsparse`'s device number is a batch sum |
| P9 | format names follow dac (`chunk_multi_turn_tool`, `chunk_single_turn_xml`) | `semsparse`'s `magic_chunk` maps to the xml ablation, not the paper's headline format |
| P10 | no scheduler subclass and no monkeypatching, registration through `register_backend` and `backend_per_kind` | both references predate those extension points |
| P11 | the vanilla baseline prepends `DEFAULT_SYSTEM` ("You are a helpful assistant."), matching current dac at `fb1ef99` (`src/dac/templates/utils.py:13`, added by `c995167` "Prepend a default system instruction so context cannot leak into the attention sink", 2026-06-12 16:12) | the paper's Table 2 and Table 6 vanilla rollout is OLDER than `c995167` and carries NO system turn (internal note `Y_format_ablation`, sections 1.2-1.4). The owner ruled dattn tracks current dac rather than the frozen baseline, so the dattn vanilla prompt is 11 tokens longer than the one behind Table 2 on 1611 of 1611 rows for both models, and the vanilla arm is NOT byte-comparable to that table. The DA and DA-nm arms are unaffected: 0-token delta on 1611 of 1611 rows, because every DA formatter carried the system turn from its first commit (`1e394b4`) |

Phase 5 clarification of P11: the system turn itself adds 11 tokens on all 1611
rows per model. Against archived counts, current vanilla has Gemma `{11: 1611}`
and Qwen `{11: 1610, 29: 1}`. The extra 18 on Qwen multidoc_qa#108 remains in
the no-system control and is the same tokenizer residual seen in the DA audit.
The zero DA delta refers to current dattn versus current dac rendering, not every
archived token count.
