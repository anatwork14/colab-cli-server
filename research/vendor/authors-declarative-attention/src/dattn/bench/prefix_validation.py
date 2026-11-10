"""GPU worker for campaign prefix validation, reached through the CLI boundary."""

from __future__ import annotations

import json
from collections import defaultdict

from dattn.campaign.population import ARMS, MODELS, identity
from dattn.campaign.validate_prefix import DECODE_TOKENS, first_difference
from dattn.eval.persistence import atomic_json


class Capture:
    """Record host mask metadata used by the actual forwards, without GPU timers."""

    def __init__(self):
        self.rows = []
        self.traces = defaultdict(dict)

    def note_lp(self, elapsed, rows):
        self.rows = rows
        for name, mode, _length, attended, pos, _age in rows:
            if pos > 0:
                self.traces[name][pos] = [pos, mode, attended, {}]

    def note_compaction(self, group, device, lengths, elapsed):
        for row, length in zip(self.rows, lengths, strict=True):
            name, mode, seq_len, attended, pos, age = row
            if pos > 0:
                self.traces[name][pos][3][str(group)] = length


def start_capture(worker):
    from dattn.engine.api import set_step_recorder

    set_step_recorder(Capture())


def finish_capture(worker):
    from dattn.engine.api import get_step_recorder, set_step_recorder

    capture = get_step_recorder()
    set_step_recorder(None)
    return [list(values.values()) for name, values in sorted(capture.traces.items())]


def greedy_params(cls, extra, n):
    # vLLM rejects constructing greedy n>1. Set n after constructing verified
    # greedy params. LLMEngine.add_request still fans this actual parent out via
    # ParentRequest into n separate requests with n=1. No sampler is patched.
    params = cls(temperature=0.0, n=1, max_tokens=DECODE_TOKENS, extra_args=extra)
    params.n = n
    return params


def worker(args):
    from dattn.campaign import driver
    from dattn.engine.launch import apply_env, engine_kwargs
    from dattn.engine.profile import load_profile
    from dattn.eval.prompts import build_prompt, dattn_extra_args
    from dattn.eval.replay import dense_trace, replay_record
    from dattn.prompt.template_policy import resolve_template

    profile = load_profile(MODELS[args.model], use=args.profile)
    apply_env(profile)
    from vllm import LLM, SamplingParams

    plan = json.loads(args.plan.read_text())
    tok = driver.tokenizer(MODELS[args.model])
    llm = LLM(**engine_kwargs(profile, seed=0))
    result = {"profile": args.profile, "arms": {}}
    for arm in ARMS:
        cfg = driver.config(args.model, arm)
        template = resolve_template(tok, cfg.model_id, cfg.template_policy)
        built = {}
        for row in plan["rows"] + plan["pair"]:
            key = repr(identity(row))
            built[key] = build_prompt(
                tok,
                context=row["context"],
                question=row["question"],
                fmt=cfg.prompt_format,
                tool_schema=cfg.tool_schema,
                chat_template=template,
                segmenter=cfg.segmenter,
                vanilla_system=cfg.vanilla_system,
            )
        cases = {}

        def generate(row, n, built=built, cfg=cfg):
            prompt = built[repr(identity(row))]
            payload = dattn_extra_args(prompt.layout, enabled=cfg.dattn_enabled)
            payload["req_id"] = "validation"
            llm.collective_rpc(start_capture)
            outputs = llm.generate(
                {"prompt_token_ids": list(prompt.ids)},
                greedy_params(SamplingParams, {"dattn": payload}, n),
                use_tqdm=False,
            )
            live = llm.collective_rpc(finish_capture)
            if len(outputs) != 1 or len(outputs[0].outputs) != n or len(live) != 1 or len(live[0]) != n:
                raise ValueError("missing parent, child outputs or live child traces")
            live = live[0]
            items = []
            for completion, child_trace in zip(sorted(outputs[0].outputs, key=lambda x: x.index), live, strict=True):
                trace = (
                    replay_record(completion.text, prompt.layout, tok)
                    if cfg.dattn_enabled
                    else dense_trace(len(completion.token_ids), prompt.layout)
                )
                if not child_trace or any(not step[3] for step in child_trace):
                    raise ValueError("live trace missing compaction evidence")
                items.append(
                    {
                        "token_ids": list(completion.token_ids),
                        "text": completion.text,
                        "finish_reason": completion.finish_reason,
                        "mode_trace": trace["mode_trace"],
                        "attended_tokens_exact": trace["attended_tokens_exact"],
                        "live_trace": child_trace,
                        "cached_tokens": outputs[0].num_cached_tokens,
                    }
                )
            return items

        for row in plan["rows"] + plan["pair"]:
            key = repr(identity(row))
            if key in cases:
                continue
            if args.profile == "parity_prefix" and not llm.reset_prefix_cache():
                raise ValueError("prefix reset failed")
            cases[key] = {"cold": generate(row, 1)}
            if args.profile == "parity_prefix":
                cases[key]["warm"] = generate(row, 1)
                cases[key]["children"] = generate(row, 4)
                if not cases[key]["warm"][0]["cached_tokens"] or not cases[key]["children"][0]["cached_tokens"]:
                    raise ValueError("identity check had no warm prefix hits")
        if args.profile == "parity_prefix":
            if not llm.reset_prefix_cache():
                raise ValueError("prefix reset failed")
            a, b = plan["pair"]
            a_ids, b_ids = built[repr(identity(a))].ids, built[repr(identity(b))].ids
            shared = first_difference(a_ids, b_ids)
            if shared is None or shared < profile.dattn_block_size:
                raise ValueError("document pair lacks a reusable rendered token prefix")
            generate(a, 1)
            pair_output = generate(b, 1)
            if not pair_output[0]["cached_tokens"]:
                raise ValueError("different-question validation had no prefix hits")
            result.setdefault("pair", {})[arm] = pair_output[0]
        result["arms"][arm] = cases
    atomic_json(args.artifact, result)
    llm.llm_engine.engine_core.shutdown()
    return 0
