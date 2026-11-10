"""Canonical authors' dattn vLLM plugin: immutable source proof and GPU profile proposals.

This does NOT implement speculative + DA or multimodal processor-span mapping.
The real dattn source is vendored in research/vendor/author-declarative-attention.
Profile generation is an OFFLINE engineering proposal, not GPU qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
AUTHOR = ROOT / "research" / "vendor" / "author-declarative-attention"
META = AUTHOR / "UPSTREAM_PROVENANCE.json"
MODEL = "Qwen/Qwen3.5-4B"
VLLM_RELEASE = "0.29.0"
SOURCE_SHA = "24110b7b441ad9423f55952a2c02d23eac80ec52"
CAPABILITIES = {"a100": "sm80", "l4": "sm89"}
BACKENDS = ("TRITON_ATTN", "FLASHINFER")
TYPES = ("da_normal", "da_draft", "da_target", "da_both")


class AuthorDAError(ValueError):
    pass


def _blob_sha(blob):
    return hashlib.sha1(b"blob " + str(len(blob)).encode("ascii") + b"\0" + blob).hexdigest()


def validate_vendor():
    """Rehash each original Git blob: vendoring must be identical to authors' SHA."""
    if not META.is_file():
        raise AuthorDAError("Missing authors' provenance manifest")
    manifest = json.loads(META.read_text(encoding="utf8"))
    if manifest.get("full_commit_sha") != SOURCE_SHA or not manifest.get("git_blob_sha_verified"):
        raise AuthorDAError("Wrong or unqualified authors' source SHA")
    blobs = manifest.get("blob_shas")
    if not isinstance(blobs, dict) or len(blobs) != 178:
        raise AuthorDAError("Full 178-file authors' source inventory required")
    mismatch = []
    for relative, original_sha in blobs.items():
        if (not isinstance(relative,str) or relative.startswith("/") or
                ".." in Path(relative).parts or not re.fullmatch(r"[a-f0-9]{40}",original_sha)):
            raise AuthorDAError("Unsafe provenance inventory")
        file = AUTHOR / relative
        if not file.is_file() or _blob_sha(file.read_bytes()) != original_sha:
            mismatch.append(relative)
    if mismatch:
        raise AuthorDAError("Authors' vendored code changed: "+",".join(mismatch[:8]))
    return {"source":manifest["canonical_source_url"],"commit":SOURCE_SHA,
            "file_count":len(blobs),"all_original_git_blobs_exact":True,
            "license":"MIT","upstream_engine":VLLM_RELEASE,
            "original_source_modified":False}


def _quote(value):
    return json.dumps(str(value), ensure_ascii=True)


def candidate_profile(*, hardware, backend, local_model, max_model_len,
                      gpu_memory_utilization=0.85):
    if hardware not in CAPABILITIES or backend not in BACKENDS:
        raise AuthorDAError("Choose explicit A100/L4 and FLASHINFER/TRITON_ATTN backend")
    path = Path(local_model)
    if not path.is_absolute() or ".." in path.parts:
        raise AuthorDAError("Require absolute staged, checksum-verified Qwen3.5 model directory")
    if not 128 <= max_model_len <= 262144:
        raise AuthorDAError("Invalid requested context length")
    if not 0.1 <= gpu_memory_utilization <= 0.95:
        raise AuthorDAError("Invalid GPU memory reservation")
    capability = CAPABILITIES[hardware]
    use = "candidate_" + capability
    key = MODEL + "/" + use
    header = "[" + _quote(key) + "]\n"
    profile = header + "\n".join((
        "model_id = " + _quote(MODEL),
        "capability = " + _quote(capability),
        "use = " + _quote(use),
        "base_backend = " + _quote(backend),
        "dattn_block_size = 16",
        "needs_compacted_seq_lens_cpu = " + ("true" if backend=="FLASHINFER" else "false"),
    )) + "\n\n"
    profile += "[" + _quote(key) + ".engine_args]\n"
    values = {
        "async_scheduling":False,"decode_context_parallel_size":1,
        "disable_cascade_attn":True,"disable_log_stats":False,
        "enable_prefix_caching":False,"enforce_eager":True,
        "kv_cache_dtype":"auto","max_num_batched_tokens":8192,
        "pipeline_parallel_size":1,"prefill_context_parallel_size":1,
        "tensor_parallel_size":1,"model":str(path),
        "attention_backend":backend,
        "attention_config":json.dumps({"backend_per_kind":{"full_attention":"CUSTOM"}}),
        "max_model_len":max_model_len,"gpu_memory_utilization":gpu_memory_utilization,
        "max_num_seqs":4,
    }
    for k,value in values.items():
        profile += k+" = "+(
            "true" if value is True else "false" if value is False else
            str(value) if isinstance(value,(int,float)) else _quote(value))+"\n"
    profile += "logits_processors = ["+_quote(
        "dattn.engine.processor:DattnLogitsProcessor")+"]\n"
    profile += "worker_extension_cls = "+_quote(
        "dattn.engine.worker_ext.DattnWorkerExtension")+"\n\n"
    profile += "["+_quote(key)+".env]\n"
    for k,v in {
        "VLLM_USE_V2_MODEL_RUNNER":"0","VLLM_USE_FLASHINFER_SAMPLER":"0",
        "VLLM_ENABLE_V1_MULTIPROCESSING":"0",
        "VLLM_ATTENTION_BACKEND":"__must_be_unset__"
    }.items():
        profile += k+" = "+_quote(v)+"\n"
    profile += "\n["+_quote(key)+".bench]\n"
    profile += ("max_tokens = 256\nrepetitions = 2\nbatches_per_task = 1\n"+
                "subset_size = 2\nseed = 0\nleading_trim = 0\n"+
                "n_resamples = 2000\nci = 0.95\n")
    return {
        "status":"PROPOSED_QWEN35_VLM_PLUGIN_PROFILE_NOT_GPU_QUALIFIED",
        "model":MODEL,"hardware":hardware,"capability":capability,
        "backend":backend,"use":use,"profile_toml":profile,
        "tested_with_vision_inputs":False,
        "tested_with_speculative_decoding":False,
        "requires_actual_processor_offsets":True,
        "requires_backend_boot_assertions":True,
        "requires_real_vision_KV_mask_effect":True,
        "requires_user_authorized_GPU":True
    }


def installation_plan(profile_file, use):
    if not Path(profile_file).is_absolute():
        raise AuthorDAError("Choose an absolute candidate profile path")
    if not isinstance(use,str) or not re.fullmatch(r"candidate_sm8[09]",use):
        raise AuthorDAError("Invalid provisional profile use")
    base = "DATTN_BACKEND_PROFILES="+str(Path(profile_file))
    return {
        "status":"DRY_RUN_GPU_INSTALL_AND_PROBE_NOT_EXECUTED",
        "python_required":"3.13",
        "install":["python3.13 -m pip install -e "+str(AUTHOR)],
        "profile_environment":{ "DATTN_BACKEND_PROFILES":str(Path(profile_file)),
                               "VLLM_ATTENTION_BACKEND":"MUST_BE_UNSET",
                               "VLLM_PLUGINS":"MUST_BE_UNSET_OR_INCLUDE_DATTN"},
        "inspection":[base+" dattn probe --model "+MODEL+" --capability "+
                      ("sm80" if use.endswith("80") else "sm89")+
                      " --profile "+use+" --no-boot --json"],
        "hardware_tests":[
            "Author source tests/contract must pass under vllm==0.29.0",
            "dattn probe boot GPU must report CUSTOM full_attention and native GDN",
            "Long-context Qwen3.5 real VLM image layout must map to processor tokens",
            "Token-level causal-mask intervention/DA-vs-DA_nm parity",
            "Speculative DA remains blocked until accepted/rejected-token extension"
        ],
        "do_not_claim":"Author DA working with Qwen3.5 multimodal or speculative until evidence"
    }


def main():
    p=argparse.ArgumentParser(description="Use canonical authors' real dattn, not lucidrains prototype")
    p.add_argument("action",choices=("verify-source","candidate-profile"))
    p.add_argument("--hardware",choices=tuple(CAPABILITIES),default="a100")
    p.add_argument("--backend",choices=BACKENDS)
    p.add_argument("--local-model")
    p.add_argument("--max-model-len",type=int,default=32768)
    p.add_argument("--output",help="Write exactly one new experimental TOML file")
    a=p.parse_args()
    if a.action=="verify-source":
        result=validate_vendor()
    else:
        if not a.backend or not a.local_model:
            p.error("candidate-profile requires --backend and --local-model")
        result=candidate_profile(hardware=a.hardware,backend=a.backend,
                                 local_model=a.local_model,max_model_len=a.max_model_len)
        if a.output:
            output=Path(a.output)
            if not output.is_absolute():
                p.error("Output must be absolute")
            output.parent.mkdir(parents=True,exist_ok=True)
            with output.open("x",encoding="utf8") as f:
                f.write(result["profile_toml"])
            result["written_profile"]=str(output)
            result["install_and_probe_plan"]=installation_plan(str(output),result["use"])
            result.pop("profile_toml")
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
