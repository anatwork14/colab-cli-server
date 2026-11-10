# dattn developer entry points. Everything runs through uv.
#
#   make lint test
#
# GPU tests require an explicitly assigned GPU.

UV ?= uv
GPU ?=
MODEL ?= gemma

.DEFAULT_GOAL := help
.PHONY: help lint fmt typecheck test test-unit test-contract test-gpu test-gpu-inproc test-all check clean

help:  ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-14s %s\n", $$1, $$2}'

lint:  ## ruff check everywhere + format check on src and tests
	$(UV) run ruff check .
	$(UV) run ruff format --check src tests

fmt:  ## ruff autofix + format src and tests
	$(UV) run ruff check --fix .
	$(UV) run ruff format src tests

test: test-unit  ## the default suite: unit only, no GPU, CPU only

test-unit:  ## tests/unit
	$(UV) run pytest tests/unit

test-contract:  ## tests/contract, pinned to vllm 0.29.0, CPU only
	$(UV) run pytest tests/contract -m contract

test-gpu:  ## tests/gpu on GPU $(GPU) for MODEL=$(MODEL), one test per process
	@test -n "$(GPU)" || { echo "Set GPU to an explicitly assigned device" >&2; exit 2; }
	CUDA_VISIBLE_DEVICES=$(GPU) tests/gpu/run_gpu_suite.sh $(MODEL)

test-gpu-inproc:  ## tests/gpu in ONE process: fails on the fourth boot (vLLM 0.29 frees no weights)
	@test -n "$(GPU)" || { echo "Set GPU to an explicitly assigned device" >&2; exit 2; }
	CUDA_VISIBLE_DEVICES=$(GPU) $(UV) run pytest tests/gpu -m gpu

test-all: test-unit test-contract  ## everything that does not need a GPU

check: lint test-all  ## CPU checks described in docs/DESIGN.md

clean:  ## drop caches
	rm -rf .pytest_cache .ruff_cache
	find src tests -name __pycache__ -type d -prune -exec rm -rf {} +
