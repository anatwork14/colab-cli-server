"""Metadata-only HF checkpoint SHA resolution for target and existing draft weights.

No weights, training data, fine-tuning, cloud sessions, or inference invoked.
"""
import argparse
import json
from .matrix import METHODS, load_catalog


def select(model, methods):
    catalog=load_catalog()
    if model not in catalog["models"]:
        raise ValueError("Unknown model profile")
    target=catalog["models"][model]
    mapping={"target":target["target_id"]}
    if len(set(methods))!=len(methods):
        raise ValueError("Duplicate speculative methods")
    for method in methods:
        if method not in METHODS:
            raise ValueError("Unknown speculative method")
        if method=="mtp":
            if not target["native_mtp"]["available_in_model_family"]:
                raise ValueError("Native MTP not declared for target")
            continue
        source=target.get(method)
        if not source:
            raise ValueError("This target lacks the requested external draft model")
        mapping[method]=source["draft_id"]
    return mapping


def resolve(model, methods, api=None):
    """Resolve exact 40-character HF revisions by contacting official model metadata."""
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError("Install huggingface_hub before metadata resolution") from exc
        api=HfApi()
    mapping=select(model,methods)
    import re
    out={}
    for label,repo_id in mapping.items():
        info=api.model_info(repo_id=repo_id)
        sha=getattr(info,"sha",None)
        if not isinstance(sha,str) or not re.fullmatch(r"[a-f0-9]{40}",sha):
            raise ValueError("HF did not return a full checkpoint commit for "+repo_id)
        out[label]={"repo_id":repo_id,"hf_commit_sha":sha,
                    "validation":"METADATA_ONLY_WEIGHTS_NOT_DOWNLOADED",
                    "url":"https://huggingface.co/"+repo_id}
    return {"model_profile":model,"models":out,
            "source_pins_resolved":True,"gpu_verified":False,
            "draft_target_compatibility_verified":False,
            "official_vision_speculative_quality_verified":False,
            "action_required":"Review model card/license, stage pinned bytes, then GPU startup and multimodal parity"}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("plan","resolve"))
    p.add_argument("--model",default="qwen35_4b",choices=tuple(load_catalog()["models"]))
    p.add_argument("--methods",default="mtp,dflash,eagle3")
    a=p.parse_args()
    methods=tuple(x for x in a.methods.split(",") if x)
    if a.action=="plan":
        result={"status":"OFFLINE_NO_NETWORK","models":select(a.model,methods)}
    else:
        result=resolve(a.model,methods)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
