# Design

## 1. Scope and layers

Declarative Attention changes which historical KV blocks a decode request reads.
The model selects document segments by emitting tags in its response.
The implementation preserves the original KV cache and changes attention metadata.
Its public surface supports serving, evaluation, replay and paired timing studies.

The supported profiles target Gemma-4-31B-it and Qwen3.6-27B on SM100.
The dependency contract is vLLM 0.29.0 and Python 3.13.
A different backend or upstream release needs fresh contract and hardware validation.

| Package | Responsibility |
| --- | --- |
| contracts | Shared layouts, interval masks, request payloads and step records |
| fsm | Tag parsing, state transitions, masks and token-exact accounting |
| prompt | Segmenting, rendering and recovering token positions |
| engine | vLLM registration, request state, compaction and capture staging |
| metrics | Worker timing, record rings, audits and environment observations |
| eval | Dataset selection, HTTP rollouts, judging, replay and accuracy |
| bench | Pilot sizing, matched arm captures, pairing and batch statistics |
| cli | Connect the public commands to these packages |

The FSM depends on contracts and the standard library.
Prompt rendering depends on contracts and the tokenizer interface.
Engine code may use contracts, the FSM and prompt layouts.
Engine code must not import metrics.
Metrics reaches the engine only through engine.api.
Eval does not import torch or vLLM.
The CLI loads engine dependencies only for commands that need them.
The import-discipline test enforces these boundaries.

Model generation and judge calls use an OpenAI-compatible HTTP server.
Eval sends already-rendered token IDs to the completions endpoint.
This preserves the token positions used by the request layout.
Judge calls send chat messages so the server applies the judge model's template.

## 2. Prompt layout

A PromptLayout records the prompt length, question boundary and segment spans.
Spans use half-open token intervals.
Segment IDs are one-based, contiguous and ascending.
The sink contains the first 16 tokens.
The question and generated response belong to the retained open tail.

The default segment target is 2048 tokens and the usual cap is 2560.
The segmenter uses progressively finer delimiters, leaving an atomic word intact even above the cap.
An empty context has an explicit placeholder.
The layout also stores the decoded prompt tail used to seed the tag scanner.

The public prompt formats are:

- chunk_multi_turn_tool, the headline tool transcript.
- chunk_single_turn_xml, the inline XML format.
- vanilla, the unsegmented baseline format.

The tool transcript declares get_magic_chunk through the tokenizer's tools argument.
Each segment has a tool-call turn and a matching tool-response turn.
Each response body names its Magic Chunk ID.
The final question follows the document transcript.

Evaluation defaults to the vendored dac templates through `--template-policy dac_bundled`.
The `stock` policy uses the tokenizer's native chat template.
Both pass the same tools object and disable thinking.
The pinned bundled Qwen template omits the tool declaration, matching dac's served prompts.
Only that exact template is exempt from the tool-declaration assertion.
Run metadata records the policy and template sha256.
Replay consumes that metadata, and rollout resume refuses mixing templates within a cell.
The lower-level renderer and timing benchmark retain the tokenizer's native template by default.
Historical prompt differences and their evidence are in research/NOTES.md.

Token spans are recovered by prefix rendering and body containment.
They are not inferred from model-family regular expressions.
The renderer verifies that the layout and rendered token IDs agree.
A formatting failure is recorded explicitly.

The layout travels with the rendered token IDs.
The serving worker does not rediscover document spans from the whole prompt.
The JSON request payload is carried through SamplingParams.extra_args.
HTTP evaluation encodes that payload in vllm_xargs.

## 3. FSM semantics

Each request owns a DeclarativeFSM.
Its modes are GLOBAL, FOCUS and LOCAL.
GLOBAL reads the dense history.
FOCUS retains selected segments plus the sink and open tail.
LOCAL retains the sink and open tail.

The scanner keeps the last 500 decoded characters.
It scans only when the current decoded token contains a greater-than character.
Tags are case sensitive and matched near the end of the scan window.
The scan window starts with the prompt's decoded tail.

The transition table is:

| Current mode | Recognized event | Next mode |
| --- | --- | --- |
| GLOBAL | Valid focus opening tag | FOCUS |
| GLOBAL | Invalid focus opening tag | GLOBAL |
| GLOBAL | Local or answer opening tag | LOCAL |
| FOCUS | Focus closing tag | GLOBAL |
| LOCAL | Local or answer closing tag | GLOBAL |

A focus opening names segment IDs using the prompt format's keyword.
The default tool format uses magic_chunks.
At most three segment IDs may be selected.
Empty selections, invalid syntax and out-of-range IDs are refused.
Layouts are validated before use.

A focus attempt is counted when its opening tag is recognized in GLOBAL.
A successful focus is counted when its closing tag is recognized in FOCUS.
Opening focus inside LOCAL does not count as an attempt.
Other tags that are not recognized in the current mode are inert.
A global tag does not introduce a fourth transition rule.

Nested tags do not create a stack.
The recognized inner close returns the current state to GLOBAL.
A later outer close is then inert.
These semantics are intentional and covered by recorded fixtures.

Masks are normalized unions of nonoverlapping half-open intervals.
FOCUS uses the sink, selected document spans and the open tail.
LOCAL uses the sink and open tail.
GLOBAL uses the dense mask.
The open tail includes the question and all generated tokens so far.

A recognized tag changes the mask for the next forward.
The ledger records the current step before applying that transition.
This one-step lag must agree between online processing and CPU replay.

Negative token IDs are asynchronous placeholders.
The processor stops before them.
Passing a negative ID to the FSM raises an invariant error.
Special token text is excluded from the tag scanner.

The ledger records token-exact attended area and the full denominator.
A zero denominator produces an undefined metric in memory.
JSON serialization converts nonfinite numbers to null.
It never writes nonstandard NaN tokens.

## 4. vLLM extension points

The integration uses four public-facing registration surfaces:

1. A vllm.general_plugins entry point installs the custom backend.
2. register_backend associates CUSTOM with the DA backend.
3. logits_processors installs DattnLogitsProcessor by its qualified name.
4. worker_extension_cls provides record, drain, audit and probe RPC methods.

There is no vllm.logits_processors package entry point.
The explicit logits_processors argument is needed to retain output token IDs.
DattnLogitsProcessor is not argmax invariant.
Greedy generation must still execute its bookkeeping.

The attention configuration binds CUSTOM only to full-attention groups.
Other attention groups retain their native backend and builder.
The backend wrapper preserves the native attention implementation.
The replacement builder changes metadata for eligible decode rows.

The worker boot hook runs after cache initialization and capture.
It checks the resolved configuration and attaches the optional recorder.
It does not need an HTTP request to initialize the worker.
The worker RPC bridge works for both offline engines and serving.

Profiles are flat model/use tables in research/backend_profiles.toml.
Each contains engine arguments, environment settings, backend and DA block size.
No profile inheritance, schema version or status registry is involved.
DATTN_BACKEND_PROFILES can select another file.

Gemma uses its Triton attention backend.
Qwen parity permits the TRT-LLM attention path.
Qwen walltime disables that path and uses the native planner.
That native planner needs the compacted sequence lengths mirrored on the CPU.
This distinction is explicit in the profile.

A03 checks actual DA binding and native-backend equality.
It also refuses a DA builder that supports update_block_table.
That fast path can bypass fresh scratch staging.
The refusal names the affected group and builder.

The block-size check compares live DA groups with the profile.
A07 checks both the cache specification and each resolved layer implementation.
A compacted group must not apply a sliding window to packed key positions.
A08 requires cascade attention to be disabled.
A11 requires prefix caching to be disabled.
A14 checks the DA logits processor and its output-token-ID bookkeeping.

Vanilla boot requires both DA installation booleans to be false.
DA boot requires both to be true.
Template hashes remain recorded provenance.
The profile does not hard-fail on a template hash registry.

## 5. Compaction and graph capture

The engine owns request state independently of vLLM's live block tables.
Request additions, removals and row moves update the state mapping.
The logits processor advances each request's FSM from newly available output IDs.
Masks are staged only after that bookkeeping.

A block-rounded mask rounds each interval outward to the kernel block size.
Overlapping rounded ranges are merged.
Each request is rounded independently.
The retained partial final page uses its true remaining token count.

The fused Triton kernel gathers retained block IDs into private scratch.
It writes compacted lengths and retained-block counts alongside them.
The native KV cache is not copied or rewritten.
The original block table remains intact.

Dense rows remain dense.
Prefill and ineligible rows use identity metadata.
Decode eligibility excludes rows still in prefill, including a final one-token chunk.
CPU mirrors support the native planner where the profile requires them.

Scratch allocation is stable across graph capture and replay.
Capture stages identity rows into those same buffers.
Runtime decode updates their contents without moving their storage.
A cheap pointer-identity check guards the DA path.

Unused scratch columns must not affect logits.
Corrupting the staged lengths must affect logits.
The GPU poison test checks both halves.
The replay test separately checks sensitivity to shrunken lengths.

Triton is required.
There is no alternative Torch production planner.
compact.oracle remains an independent set-union reference for kernel tests.
CPU state tests use it as a test double.

## 6. Recorder and clock

The common timestamp t_sched_ns is taken at worker execute_model entry.
Both DA and builder-free vanilla use that hook.
Vanilla requires no DA logits processor to obtain a timestamp.

The wall period assigned to step k is:

    wall_ms[k] = (t_sched_ns[k+1] - t_sched_ns[k]) / 1e6

It is a successor period on the host clock.
It includes the execution and intervening work between those worker entries.
It is not the forward CUDA-event duration.
The last record has no successor and cannot define a period.

The phase 2 implementation stamped InputBatch.refresh_metadata.
The current boundary is execute_model entry.
Timing acceptance must explicitly account for that boundary change.

Step records include engine and recorder IDs, request counts and execution kind.
They include prefill-token counts and actual CUDA-graph mode.
Per-request rows include sequence length and attended-token counts.
Block-rounded and token-exact attended counts remain distinct fields.

The recorder uses bounded rings and drains rather than silently dropping data.
Device-versus-mirror audits check what the GPU actually read.
The serve command retains both --record and a shutdown drain.
The worker bridge exposes recording, draining, auditing and probing together.

Each benchmark window has two environment snapshots, before and after.
They record SM clock, memory clock, temperature and throttle reasons.
Each cell also writes one nvidia-smi -q dump.
These are observations, not acceptance gates.

## 7. Measured pool and block-accounted batch draws

Probe both arm processes at gpu_memory_utilization=0.92 for each model.
The pin is the smaller reported KV budget rounded down to a whole GiB.
The shipped walltime rows contain a placeholder and refuse normal boots.
Only the explicit pool-resolution probe bypasses that placeholder.
Publish both probes and the resulting pin with the measurements.
A boot failure at that pin is a failure, never a reason to reduce the pin.

KV groups alias one physical allocation.
A physical block ID is owned by one group at a time.
Physical bytes per block are the maximum layer-inclusive page cost of a group.
vLLM floors the budget to a whole number of these blocks.
The probe reconciles that allocation with vLLM's backing tensor metadata.
It prints the budget, allocation, null reservation and rounding remainder.
Each group's logical full-capacity product is printed separately.
Those aliased logical products must not be added to recover physical bytes.
Usage excludes the reserved null block from its denominator.

A pinned boot writes the physical block size and resolved group table to pool.json.
Every benchmark boot compares its geometry with that record.
Both arms use the same pin and geometry.
The capture-size list includes every integer through max_num_seqs.
The resolved list must contain the actual drawn C.

For each DA prompt use L = prompt length + 512.
A full-attention group holds ceil(L / block_size) blocks.
A sliding group holds ceil(min(L, sliding_window) / block_size) blocks.
A GDN or mamba group holds one block.
Add the group block counts and multiply by physical bytes per block.
The 0.80 target normalizes per-sequence logical KV read load against the pool.
Cascade attention is disabled, so shared blocks still participate in each sequence's reads.
Sharing can lower physical usage. We do not fill that slack with more sequences,
which would exceed the intended load of a pool holding unique prompts.
This is a logical normalization, not equal HBM traffic or a prediction of DA savings.
Hardware caches can serve repeated reads of shared data.

For each task and batch, shuffle selected rows without replacement.
The deterministic seed includes protocol seed, task name and batch index.
Stop on reaching [0.77, 0.83] of physical allocation.
Skip a row that would overshoot the upper edge.
Stop at exhaustion or max_num_seqs and record any shortfall.
An empty draw or a stored C above max_num_seqs refuses.
Persist all five ordered draws, seeds, C values and predicted occupancies.
Both arms use precisely those rows, even if C differs between batches.
The pin identity includes prompts, protocol, profile, prefix caching and prefill budget.

Validate once per model and run, on the first batch of the first task.
At the largest observed warm-pass running count, prefer the first all-decode step.
Before scheduling, every running request must have completed prefill with no
in-flight tokens. The step must schedule one decode token for each request, so
sliding-window reclamation has run before measuring usage. If no such step exists
at that count, use the first maximum-running snapshot. Record snapshot_kind as
all_decode or max_running. All C need not have been resident together.
Record n_running, C, request IDs,
computed and prompt token counts, usage, step index and queue sizes.
Evaluate the same block model on those actual resident token counts.
Print predicted bytes / physical allocation and measured scheduler usage.
With prefix caching, refuse measured usage above predicted plus 0.03.
Sharing can lower physical usage, so there is no lower-side refusal.
An uncached accounting-control boot retains the two-sided absolute 0.03 check.
Record measured minus predicted, the caching flag and the gate form.
The null-block denominator difference remains explicit in the pool record.
Later cells retain their snapshot and reference the first validation without gating again.
This observation precedes freeing and is separate from the zero-usage hygiene check.

Realised occupancy is recorded at every measured step, with its mean and whole-record peak.
It need not equal the admission target and is not a rectangle gate.
DA compaction changes attention reads without freeing the original KV allocation.
Any preemption anywhere in the timed record refuses the capture.

## 8. Two-process measurement

DA and vanilla run in separate engine processes, one batch per process.
There is no engine-level runtime switch.
Vanilla uses builder_free_kwargs and removes the DA builder and LP.
Both arms retain the same explicit native backend and recorder bridge.

The walltime decode cap is 512 generated tokens per request.
Vanilla uses max_tokens=min_tokens=512 and ignore_eos.
DA uses max_tokens=512 and may stop at EOS.
The cap is a timing setting, not an accuracy setting.
Answers requiring more than 512 tokens are truncated in this timing experiment.
Each repetition is paired and trimmed to its own DA rectangle.

The batch and prompt identities must agree across arms.
Resolved engine arguments, prefix-cache setting and geometry are compared.
Only the DA binding fields may differ.
Actual installation booleans must match each arm's role.
The boot identity includes process id, start time and resolved configuration hash.

Each arm boot first runs a 64-token untimed warm pass on the selected prompts.
It uses the same layouts and initial FSM state as the measured requests.
The warm pass establishes the intended prefix-cache state.
This supersedes the former requirement that no reusable KV survive warmup.
The scheduler must have no running or waiting requests and zero live KV usage
before every timed repetition. Free cached blocks can still hold reusable KV.
Physical cache counts are recorded as an informational witness.
Actual coverage is checked through num_cached_tokens for every request in every rep.
Read the hybrid coordinator's resolved cache-lookup alignment, or the resolved
scheduler alignment for a single cache type. The expected reusable prefix is
floor((prompt_len - 1) / alignment) times alignment. vLLM recomputes the last
prompt token for logits, including a whole block at an aligned endpoint.
Record expected_aligned_hit, observed cached tokens and their nonnegative deficit.
Any positive deficit, missing count or invalid count refuses the capture and names
the request. A prompt shorter than the alignment has no reusable aligned block.
Only the prompt prefix may be reused, not the warmup's generated continuation.
Prefill steps before the rectangle are recorded, including for refused captures.
Also record recompute_tokens, resolved alignment and
ceil(recompute_tokens / max_num_batched_tokens) as a prefill-step lower bound.
G2 is diagnostic, never a refusal or a two-step cache-miss flag.
Qwen can recompute up to one 784-token aligned block per request, so the rectangle
opens after a few mixed steps at high C. Gemma's tail is at most one small block.
Use the boot-resolved alignment if it changes. At C = 128, 784-token tails need
at least seven steps under the 16384-token budget, before scheduler overhead.

Read-load sizing alone does not guarantee retention between repetitions.
A CPU allocator counterexample (an unpublished probe)
evicts cached GDN prompt states during old/new state overlap at modelled load 0.776.
It is an allocator construction, not a captured GPU failure. Therefore each rep's
aligned-hit witness establishes reuse. Neither spare memory nor aggregate cache
counts substitute for that observation. The warm pass need not hold all C at once.

The boot then runs eight timed decodes of the same batch.
Each uses the model's recommended sampled recipe from eval/sampling.py.
Gemma uses temperature 1.0, top_p 0.95 and top_k 64.
Qwen uses its own model row.
The request seed depends on the batch seed, request occurrence and repetition index.
The same batch and repetition use the same random stream across arms and boots.
This does not guarantee identical token sequences for different decoding policies.

Each DA repetition has its own masks, EOS position and retained length.
The closed rectangle starts after the first full decode step.
It closes on the first shrink, loss of pure decode or record exhaustion.
It does not close at an occupancy threshold.
The per-repetition raw and kept step counts are reported.
This estimates timing over slightly different step counts and mask trajectories.
It is not a full-response latency or independently optimized throughput estimate.

Across batches, DA runs first when task_index + batch_index is even.
Vanilla runs first when that sum is odd.
The task index is the position in the recorded protocol task list.
All eight repetitions in an arm boot run in order, 0 through 7.
Both arm processes finish before moving to the next batch.
Five batches therefore have a 3/2 arm-order imbalance for each task.
Pair output records the imbalance and the bin-0 saving by process order.

The estimator uses within-batch pooled steps over repetitions.
All kept steps in a batch and bin contribute to each arm's mean.
A repetition with more kept steps therefore carries more timing evidence.
The task/bin saving is a ratio of means, never a mean of repetition percentages.
Per-bin repetition support is reported alongside kept steps and contributing batches.

Repetitions provide timing replication within a batch.
Batches and their separate boots provide statistical replication.
Repetitions never become additional bootstrap batches.
Vanilla repetition-pair drift uses common kept positions after each repetition's
own DA trim, rather than the longer untrimmed vanilla capture.
Acceptance also reports a predeclared fresh-boot A/A pair for each arm and model.
The within-boot drift diagnostic alone does not establish fresh-boot stability.

## 9. Pairing and statistics

Pairing keys identify task, batch, repetition and step index.
Arm names label the two sides.
The same matching mechanism handles DA/vanilla and vanilla/vanilla.

Engine and recorder step contiguity are computed before filtering.
A missing step cannot be bridged by dropping an intervening record.
Zero or negative periods raise a fault naming the cell and step.

A kept pair requires decode on both sides and a contiguous successor.
Neither side may contain prefill tokens or preemption.
num_reqs and num_reqs_padded must both equal C.
Kept steps must use FULL CUDA graphs.
Leading trimming is applied within each repetition.

The mask ratio is one minus attended blocks divided by sequence blocks.
Each request is rounded before the batch sums are formed.
Ten-percent bins group paired steps by that ratio.

Within each task and bin, saving is one minus the ratio of mean periods.
Across tasks, the macro gives each contributing task equal weight.
The headline population is the 15 registered tasks.
Partial reports are explicitly labelled.

Bootstrap samples whole batches within each fixed task.
It never samples individual steps, repetitions or tasks as independent units.
A bin gets no interval if a contributing task has fewer than two batches.
Reports expose n_tasks, n_batches and n_batches_min.

Bin-0 saving is also split by actual pass order.
That diagnostic uses the same ratio-of-means then equal-task macro.
It includes contributing task, batch and kept-step counts.
One saving-versus-bin plot is retained.

Vanilla under-fills the batch relative to DA, so the measured saving is a lower bound.

## 10. Evaluation and validation

Dataset selection applies rubric filtering, context-token filtering, shuffle and prefix.
The default selection seed is 42 and the cap is 128 rows per task.
Selection provenance is checked before resuming existing responses.
Changing it refuses the append rather than silently mixing populations.

Rollout identity is subset, row_index and response_index.
Responses are append-only.
A transport failure is a flagged event eligible for retry.
An empty model answer is a terminal answer.
Completed keys are skipped, including partial multi-response resumes.

The HTTP client allows at most one bounded retry.
Rollouts use token completions and judging uses chat.
Judge servers run without reasoning-parser reassembly.
Judgments and malformed model answers remain visible.

CPU replay runs the same FSM over recorded response text.
Dense arms receive an arithmetic dense-attention trace.
Replay does not claim to reconstruct GPU block-rounded counts.

Accuracy reports macro task accuracy and pooled micro accuracy from the same rows.
Format failures, unjudged answers and unparsable verdicts stay in the denominator.
Transport failures are counted separately from model answers.
The report includes n_format_fail and n_unparsable.
No policy drops truncations or nonterminating answers.

Mask parity checks full rollout keys, row identity and nonempty prompt hashes.
It does not certify sampled-text equality or mask correctness.
Kernel and engine tests establish those separate properties.

The CPU suite includes protected FSM/prompt tests and upstream contracts.
Hypothesis remains in the FSM tests.
GPU tests cover kernel equivalence, boot, exclusion, dense parity and scratch replay.
A short end-to-end test covers rendering, generation, recording and replay.
Hardware tests require an explicitly assigned GPU and are not part of CPU validation.
