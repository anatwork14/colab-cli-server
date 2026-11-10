#!/usr/bin/env bash
# One pytest process per GPU test, because vLLM 0.29 does not release the model weights on
# an in-process teardown (see `conftest.get_engine`). Without this a whole-file run skips
# every test past the third engine boot.
# Assign an idle GPU explicitly through CUDA_VISIBLE_DEVICES.
set -u
: "${CUDA_VISIBLE_DEVICES:?assign an idle GPU explicitly}"
cd "$(dirname "$0")/../.." || exit 1
FILTER="${1:-}"
TESTS=(
  "tests/gpu/test_kernel_equivalence.py"
  "tests/gpu/test_engine_dattn.py::test_the_boot_assertions_pass_on_the_real_backend"
  "tests/gpu/test_engine_dattn.py::test_mask_hides_excluded_context"
  "tests/gpu/test_engine_dattn.py::test_forced_global_reproduces_vanilla_greedy_output"
  "tests/gpu/test_engine_dattn.py::test_da_capture_uses_live_scratch_and_ignores_unstaged_tail"
  "tests/gpu/test_engine_dattn.py::test_a_replayed_cudagraph_honours_the_shrunken_seq_lens"
  "tests/gpu/test_end_to_end.py::test_prompt_generation_record_and_replay"
)
rc=0
for t in "${TESTS[@]}"; do
  args=(-m gpu -q -p no:randomly --no-header -s "$t")
  # The kernel tests carry no model parameter, so a model filter would deselect them all.
  case "$t" in *test_engine_dattn.py*|*test_end_to_end.py*) [ -n "$FILTER" ] && args+=(-k "$FILTER");; esac
  echo "=== $t ${FILTER:+[$FILTER]}"
  uv run pytest "${args[@]}" || rc=1
done
exit $rc
