# Declarative Attention

Declarative Attention lets a model select document segments by emitting tags while
it generates. `dattn` implements the tag state machine, prompt layouts and compacted
attention metadata on vLLM. It also provides evaluation and paired timing tools.
The original KV cache is preserved.

Vanilla under-fills the batch relative to DA, so the measured saving is a lower bound.

## Install

Use Python 3.13 and run these commands from a source checkout:

```sh
uv sync --extra eval
uv run dattn --help
```

vLLM is pinned to 0.29.0. The supplied engine profiles target SM100 with
`google/gemma-4-31B-it` and `Qwen/Qwen3.6-27B`. Model access and sufficient device
memory are required for generation. Other hardware needs a validated profile.
Dataset loading uses the `eval` extra. CPU analysis consumes recorded outputs.

Profiles live in [research/backend_profiles.toml](research/backend_profiles.toml).
Set `DATTN_BACKEND_PROFILES` to use a different file. Profiles explicitly specify
the native backend, DA block size, engine arguments and benchmark settings.

## Serve and inspect

Set `CUDA_VISIBLE_DEVICES` to a device assigned to your run. Preview the server
command, then add `--exec` to start it:

```sh
uv run dattn serve --model google/gemma-4-31B-it --profile parity
uv run dattn serve --model google/gemma-4-31B-it --profile parity --exec
```

`--record --record-dir runs/records` enables worker step records and the shutdown
drain. The default server address is `http://localhost:8000`. DA requests carry a
rendered layout with their token IDs, which `dattn eval` supplies automatically.

Resolve a profile without loading weights, or omit `--no-boot` to boot the selected profile
with backend assertions:

```sh
uv run dattn probe --model google/gemma-4-31B-it --no-boot --json
```

## Evaluate and judge

With the matching server running:

```sh
uv run dattn eval --model google/gemma-4-31B-it --arm da --out outputs
```

The default population is the 15 headline tasks, capped at 128 selected rows per
task. `--subsets` and `--n-per-subset` select a smaller diagnostic run.
`--select-only` writes selected rows without contacting the server.
`--arm da_nm` keeps the same prompt with a dense mask. `--arm vanilla` uses the
unsegmented prompt. XML formats are available as `xml` and `xml_nm`.

Each cell contains `rows.jsonl`, `responses.jsonl`, `judgments.jsonl` when judged,
and `meta.json`. Re-running the same command appends missing rollout keys.
Selection drift refuses the resume. Transport failures remain flagged in the log
and may be retried. Empty or malformed model answers count as answers.

Start an OpenAI-compatible judge server separately, using `Qwen/Qwen3.5-4B`
without a reasoning parser. Point the judge client at that server:

```sh
uv run dattn judge --dir outputs/gemma_4_31b_it/da --base-url http://localhost:8001
uv run dattn analyse --dirs outputs/gemma_4_31b_it/da --json-out accuracy.json
```

Judging uses chat messages, thinking enabled and the fixed rubric template.
Accuracy reports macro task accuracy and pooled micro accuracy from the same
rows, plus format-failure and unparsable-verdict counts. Truncated answers stay
in the denominator. Partial task coverage is labelled.

Compare two same-prompt cells with the public identity check:

```sh
uv run dattn parity --left outputs/gemma_4_31b_it/da --right outputs/gemma_4_31b_it/da_nm
```

This checks rollout IDs, row identity and prompt hashes. It does not compare
sampled answers or require private reference outputs.

## Benchmark and analyse

Benchmark arms run in separate processes, with eight sampled decodes of one batch
inside each process. Both arms have a 512-token timing cap. Vanilla generates all
512 tokens and DA can stop at EOS. This is not an accuracy-generation setting.
Each repetition is trimmed to its own DA pre-first-EOS rectangle.

First probe both arms at gpu_memory_utilization=0.92. The shipped walltime pool
values are placeholders and normal boots refuse them. For each model, copy the
smaller printed resolved KV budget, rounded down to a whole GiB, into that model's
walltime kv_cache_memory_bytes row. No numerical pin is claimed before this probe.
Publish the two probe outputs and that pin with the results.

The probe reports physical allocation, null-block reservation, rounding remainder
and logical capacities of the aliased KV groups. Group capacities cannot be added
to recover the pool. The old 53/23 GiB pool derivation is withdrawn.
Never reduce a pin to get past an OOM or failed capture.

For example, with an assigned GPU and a separate output root per model:

```sh
model=google/gemma-4-31B-it
out=runs/gemma-timing
mkdir -p "$out/pins"

for arm in da vanilla
do
  uv run dattn probe --model "$model" --profile walltime --arm "$arm" \
    --resolve-pool --pool-out "$out/probe-$arm.json" --json
done
# Set this model's TOML pin to floor(min(resolved budgets) / 2**30) * 2**30.
# Then establish the pinned geometry and verify it in the other arm.
uv run dattn probe --model "$model" --profile walltime --arm da \
  --pool-out "$out/pins/pool.json" --json
uv run dattn probe --model "$model" --profile walltime --arm vanilla \
  --pool-record "$out/pins/pool.json" --json
```

Sizing uses the boot-resolved group block counts for each DA prompt plus 512
tokens, multiplied by the shared physical bytes per block. Rows are drawn without
replacement in seeded task/batch order until the total enters 80% of the pool
plus or minus three percentage points. This normalizes per-sequence logical KV
read load. Shared prefixes can reduce physical usage, but we do not fill that
slack with extra sequences and increase the defined load. Hardware cache reuse
means this normalization does not guarantee equal HBM traffic for unique prompts.
Overshoots are skipped and exhaustion or
max_num_seqs shortfalls are recorded. All five draws are persisted. Both arms use
the same draw, with C allowed to differ across batches.

The 64-step warm pass on those prompts establishes the intended prefix-cache
state. At maximum warm-pass concurrency, the first batch prefers the first step
that schedules every running request for decode after prefill is complete.
If none exists, it retains the first maximum-running snapshot. The recorded
snapshot_kind distinguishes all_decode from max_running.
With caching, measured resident usage must not exceed its modelled value by more
than three percentage points. A separate uncached accounting control checks both
sides. Before every timed repetition the scheduler must be idle
with zero live usage. Cached-block counts are informational. Each request's
recorded num_cached_tokens must cover floor((prompt_len - 1) / resolved_alignment)
aligned blocks. The capture records the expected hit and deficit and refuses any
deficit. Qwen can recompute a tail up to one 784-token block, and Gemma one small
block. Prefill steps and their token-budget lower bound are recorded, never gated.
At high concurrency a few mixed steps can precede the rectangle. Any preemption refuses.

Run five batches of a diagnostic task in alternating process order:

```sh
task=longbench_synthetic_v4_1_ruler_niah_single_1
for batch in 0 1 2 3 4
do
  arms="da vanilla"
  if [ $((batch % 2)) -eq 1 ]
  then
    arms="vanilla da"
  fi
  for arm in $arms
  do
    uv run dattn bench --model "$model" --arm "$arm" \
      --subsets "$task" --task-index 0 --batch-index "$batch" \
      --pins "$out/pins" --out "$out/$arm"
  done
done
uv run dattn analyse --bench --pair --dirs "$out/da" "$out/vanilla" \
  --pair-out "$out/paired" --allow-partial \
  --json-out "$out/summary.json" --plot "$out/saving.png"
```

For the full matrix, use --subsets headline and --task-index 0 through 14 with a
fresh draw directory containing the same pool.json. Arm order uses task index
plus batch index parity. Each command runs all eight repetitions inside its boot,
with distinct request seeds and per-repetition lengths recorded.

Analysis uses within-batch pooled steps over repetitions, then the existing
ratio of means per task and mask bin and an equal-task macro. Confidence intervals
resample whole batches within each task. Reports expose task, batch and repetition
support, thin bins, pass-order diagnostics and burst sensitivity. Fewer than two
batches in a contributing task yields no interval. The unexcluded macro is primary.
Wall periods run from one worker execute_model entry to the next.

## Tests and design

```sh
CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/unit tests/contract
uv run ruff check src tests
uv run ruff format --check src tests
```

GPU tests run separately with an explicitly assigned device and `-m gpu`.
`tests/gpu/run_gpu_suite.sh gemma` and `tests/gpu/run_gpu_suite.sh Qwen` run the
kernel tests, five engine acceptance checks and a short end-to-end test.

See [DESIGN.md](docs/DESIGN.md) for the layering rule, FSM semantics and timing
protocol, [PROVENANCE.md](docs/PROVENANCE.md) for attribution, and
[research/NOTES.md](research/NOTES.md) for historical backend and prompt findings.

## License

MIT. See [LICENSE](LICENSE).
