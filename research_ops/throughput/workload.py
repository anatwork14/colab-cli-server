"""Random-token-derived, exact-length chat prompts and streaming measurement.

This benchmark deliberately generates TEXT-ONLY synthetic stress prompts,
NOT synthetic evaluation records for MemLens or MMLongBench-Doc.

vLLM 0.29.0 chat API accepts messages, NOT raw prompt_token_ids. Therefore
random original tokenizer IDs are decoded to text, re-rendered with the exact
non-thinking chat template and checked for the *effective* prompt token count.
The original random IDs need not equal the final chat-template token sequence.
For raw-ID injection use a separately labelled /v1/completions experiment.
"""
import hashlib
from functools import lru_cache
import json
import random
import re
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

BAD_MARKERS=("<|im_start|>","<|im_end|>","<|im_sep|>","<|fim_prefix|>",
             "<|fim_middle|>","<|endoftext|>")

class PromptConstructionError(ValueError):
    pass


def child_seed(seed,index):
    raw="{}:{}".format(seed,index).encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],"big") & ((1<<63)-1)


@lru_cache(maxsize=8)
def _allowed_token_ids(vocab_size,special):
    """Build the approved ID pool ONCE per tokenizer vocabulary, not per request."""
    if type(vocab_size) is not int or vocab_size<100:
        raise PromptConstructionError("Tokenizer vocabulary invalid")
    special=set(special)
    return tuple(i for i in range(vocab_size) if i not in special)


def render_chat(tokenizer,text):
    messages=[{"role":"user","content":text}]
    try:
        tokens=tokenizer.apply_chat_template(
            messages,tokenize=True,add_generation_prompt=True,
            enable_thinking=False)
    except (AttributeError,KeyError,TypeError) as exc:
        raise PromptConstructionError("Model tokenizer lacks an approved non-thinking chat template") from exc
    if not isinstance(tokens,list) or not all(type(x) is int for x in tokens):
        raise PromptConstructionError("Tokenizer returned unexpected multimodal/structured token representation")
    return tokens


def build_prompt(tokenizer,target_tokens,seed,*,max_attempts=30):
    if type(target_tokens) is not int or target_tokens<16:
        raise PromptConstructionError("Input context must contain at least 16 post-template tokens")
    if type(seed) is not int or seed<0:
        raise PromptConstructionError("Seed must be a nonnegative integer")
    if type(max_attempts) is not int or max_attempts<1:
        raise PromptConstructionError("Invalid adjustment budget")
    special=tuple(sorted(set(getattr(tokenizer,"all_special_ids",[]))))
    allowed=_allowed_token_ids(int(tokenizer.vocab_size),special)
    rng=random.Random(seed)
    shell=len(render_chat(tokenizer,""))
    if shell>=target_tokens:
        raise PromptConstructionError("Chat template overhead already exceeds context size")
    guess=max(1,target_tokens-shell)
    ids=[allowed[rng.randrange(len(allowed))] for _ in range(guess)]
    for turn in range(max_attempts):
        text=tokenizer.decode(ids,skip_special_tokens=True,
                              clean_up_tokenization_spaces=False)
        if not isinstance(text,str) or not text or any(x in text for x in BAD_MARKERS):
            # A sampled text fragment may contain chat-role syntax. Reset to a
            # fresh independent child sequence; never allow role injection.
            ids=[allowed[rng.randrange(len(allowed))] for _ in range(guess)]
            continue
        rendered=render_chat(tokenizer,text)
        count=len(rendered)
        if count==target_tokens:
            return {
                "messages":[{"role":"user","content":text}],
                "prompt_tokens":count,
                "prompt_sha256":hashlib.sha256(text.encode("utf8")).hexdigest(),
                "source_id_count":len(ids),
                "origin":"random_token_ids_decoded_then_chat_rendered",
                "seed":seed,"adjustments":turn,
            }
        delta=target_tokens-count
        if delta>0:
            ids.extend(allowed[rng.randrange(len(allowed))] for _ in range(delta))
        elif len(ids)+delta>0:
            ids=ids[:delta]
        else:
            ids=[allowed[rng.randrange(len(allowed))] for _ in range(max(1,guess))]
    raise PromptConstructionError(
        "Unable to generate exact effective chat context length after {} attempts".format(max_attempts))


def payload(model,prompt,seed,output_tokens=256):
    if prompt["prompt_tokens"]<1:
        raise ValueError("Prompt count was not verified by actual tokenizer")
    if output_tokens!=256:
        raise ValueError("Fixed-output research protocol requires 256 tokens")
    return {
        "model":model,"messages":prompt["messages"],
        "stream":True,"stream_options":{"include_usage":True},
        "return_token_ids":True,"temperature":0,"seed":seed,
        "max_completion_tokens":256,"min_tokens":256,"ignore_eos":True,
        "stop":[],"stop_token_ids":[],
        "chat_template_kwargs":{"enable_thinking":False},
    }


def _get_token_ids(choice):
    ids=choice.get("token_ids")
    if ids is None:
        ids=(choice.get("delta") or {}).get("token_ids")
    return ids if isinstance(ids,list) and all(type(x) is int for x in ids) else []


def request_stream(url,model,prompt,seed,timeout=900):
    """Measure first *generated token*, not the role-only initial SSE frame."""
    body=json.dumps(payload(model,prompt,seed),ensure_ascii=True,
                    separators=(",",":")).encode("utf8")
    req=Request(url,method="POST",data=body,
                headers={"Content-Type":"application/json","Accept":"text/event-stream"})
    start=time.perf_counter()
    token_first=None
    token_last=None
    stream_token_count=0
    text_parts=[]
    usage=None
    finish=None
    done=False
    chunks=0
    try:
        with urlopen(req,timeout=timeout) as response:
            if response.status!=200:
                raise RuntimeError("Non-200 HTTP streaming response")
            for line in response:
                stripped=line.strip()
                if not stripped.startswith(b"data:"):
                    continue
                raw=stripped[5:].strip()
                if raw==b"[DONE]":
                    done=True
                    break
                event=json.loads(raw)
                if event.get("error"):
                    raise RuntimeError("Remote vLLM streamed an error")
                if isinstance(event.get("usage"),dict):
                    usage=event["usage"]
                for choice in event.get("choices",[]):
                    delta=choice.get("delta") or {}
                    content=delta.get("content")
                    ids=_get_token_ids(choice)
                    # Ignore role-only events, which vLLM's built-in benchmark
                    # may otherwise count as first-token latency.
                    if ids or (isinstance(content,str) and bool(content)):
                        t=time.perf_counter()
                        if token_first is None:token_first=t
                        token_last=t
                        chunks+=1
                        stream_token_count+=len(ids)
                        if isinstance(content,str):text_parts.append(content)
                    if choice.get("finish_reason") is not None:
                        finish=choice["finish_reason"]
    except HTTPError as exc:
        # Keep sensitive prompt contents out of failure receipts.
        raise RuntimeError("vLLM HTTP error {}".format(exc.code)) from exc
    end=time.perf_counter()
    if not done or token_first is None or not isinstance(usage,dict):
        raise RuntimeError("Incomplete SSE stream/first-token/usage metadata")
    actual_prompt=usage.get("prompt_tokens")
    actual_output=usage.get("completion_tokens")
    if type(actual_prompt) is not int or actual_prompt!=prompt["prompt_tokens"]:
        raise RuntimeError("Server chat prompt count disagrees with locally rendered token count")
    if actual_output!=256:
        raise RuntimeError("Truncated/early-finished output: {} tokens instead of 256".format(actual_output))
    if stream_token_count and stream_token_count!=actual_output:
        raise RuntimeError("Stream token IDs do not sum to final completion usage")
    text_sha=hashlib.sha256("".join(text_parts).encode("utf8")).hexdigest()
    return {
        "status":"SUCCESS","prompt_tokens":actual_prompt,
        "completion_tokens":actual_output,"finish_reason":finish,
        "ttft_ms":round((token_first-start)*1000,4),
        "e2e_ms":round((end-start)*1000,4),
        "tpot_ms_estimate":round((token_last-token_first)*1000/(actual_output-1),4),
        "tpot_note":"client_first_last_stream_token_interval_divided_by_255_NOT_exact_server_ITL",
        "content_chunks":chunks,"content_sha256":text_sha,
        "stream_token_ids_seen":stream_token_count,
        "request_bytes":len(body),
    }
