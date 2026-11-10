"""Strict, immutable experiment-arm planning for a Qwen3.5 VLM vLLM engine.

Six user-visible mode categories expand into fourteen actual arms when all
three speculative decoders are selected. No GPU is used and no claim of
Declarative Attention execution is made merely from a CLI argument.
"""
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "research" / "vllm_engine" / "checkpoint_catalog.json"
MODES = ("normal", "speculative", "da_normal", "da_draft", "da_target", "da_both")
METHODS = ("mtp", "dflash", "eagle3")
FULL_SHA = re.compile(r"^[a-f0-9]{40}$")
DA_ROLES = {"normal": (), "speculative": (), "da_normal": ("target",),
            "da_draft": ("draft",), "da_target": ("target",),
            "da_both": ("target", "draft")}


@dataclass(frozen=True)
class Arm:
    model_profile: str
    target_id: str
    mode: str
    method: Optional[str]
    da_roles: Tuple[str, ...]
    draft_id: Optional[str]
    num_speculative_tokens: Optional[int]
    status: str

    @property
    def slug(self):
        return self.model_profile + "-" + self.mode + (
            "-" + self.method if self.method else "")

    def to_dict(self):
        return {"id": self.slug, "model_profile": self.model_profile,
                "target_id": self.target_id, "mode": self.mode,
                "speculative_method": self.method,
                "declarative_attention_roles": list(self.da_roles),
                "draft_id": self.draft_id,
                "num_speculative_tokens": self.num_speculative_tokens,
                "engineering_status": self.status,
                "vllm_version": "0.29.0",
                "offline_plan_only": True}


def load_catalog():
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("guards", {}).get("no_training") is not True:
        raise ValueError("Invalid immutable Qwen VLM checkpoint catalog")
    return data


def make_arm(model, mode, method=None, *, catalog=None):
    c = catalog or load_catalog()
    if model not in c["models"]:
        raise ValueError("Unknown model profile")
    if mode not in MODES:
        raise ValueError("Unknown inference mode")
    spec = mode in ("speculative", "da_draft", "da_target", "da_both")
    if spec != (method in METHODS):
        raise ValueError("Speculative mode requires exactly one of mtp/dflash/eagle3")
    target = c["models"][model]
    if float(target["parameters_billion"]) >= 10 or not target["vision_capable"]:
        raise ValueError("Select a vision-capable target smaller than 10B")
    draft = None
    tokens = None
    if spec:
        if method == "mtp":
            mtp = target.get("native_mtp")
            if not mtp or not mtp.get("available_in_model_family"):
                raise ValueError("Target does not declare native MTP")
            tokens = mtp["num_speculative_tokens"]
        else:
            details = target.get(method)
            if not details or not details.get("draft_id"):
                raise ValueError("No checkpoint published for this target+method")
            draft = details["draft_id"]
            tokens = details["num_speculative_tokens"]
    da_roles = DA_ROLES[mode]
    status = ("DA_ENGINE_HOOK_NOT_INTEGRATED" if da_roles else
              "DRAFT_RUNTIME_UNQUALIFIED" if method in ("dflash", "eagle3") else
              "MTP_GPU_UNQUALIFIED" if method == "mtp" else
              "STOCK_GPU_BASELINE_PENDING")
    return Arm(model,target["target_id"],mode,method,da_roles,draft,tokens,status)


def matrix(model="qwen35_4b"):
    c = load_catalog()
    return ([make_arm(model,"normal",catalog=c)] +
            [make_arm(model,"speculative",m,catalog=c) for m in METHODS] +
            [make_arm(model,"da_normal",catalog=c)] +
            [make_arm(model,mode,m,catalog=c)
             for m in METHODS for mode in ("da_draft","da_target","da_both")])


def command_for_arm(arm, target_path, *, draft_path=None, served_name=None,
                    max_model_len=32768, gpu_memory_utilization=0.85,
                    max_images=160):
    """Build a dry-run actual vLLM argv ONLY for arms with no DA.

    It cannot claim source-level DA integration, regardless of flags.
    All binary assets must be separately pinned/verified at launch time.
    """
    if arm.da_roles:
        raise RuntimeError("DA requires verified vLLM 0.29.0 engine hook; not an argv flag")
    if not isinstance(target_path,str) or not Path(target_path).is_absolute():
        raise ValueError("Target must be verified absolute local checkpoint path")
    if not 0.1<=gpu_memory_utilization<=0.96 or max_model_len<128 or max_images<1:
        raise ValueError("Invalid serving resources")
    args=["vllm","serve",target_path,"--served-model-name",
          served_name or arm.target_id,"--host","127.0.0.1",
          "--max-model-len",str(max_model_len),
          "--gpu-memory-utilization",str(gpu_memory_utilization),
          "--limit-mm-per-prompt",json.dumps({"image":max_images})]
    if arm.method:
        spec={"method":arm.method,"num_speculative_tokens":arm.num_speculative_tokens}
        if arm.method in ("dflash","eagle3"):
            if not isinstance(draft_path,str) or not Path(draft_path).is_absolute():
                raise ValueError("Separate draft must be staged locally and SHA-verified")
            if Path(draft_path).resolve()==Path(target_path).resolve():
                raise ValueError("Draft and target must not reuse same checkpoint")
            spec["model"]=draft_path
        elif draft_path:
            raise ValueError("Native MTP is embedded: no external draft path")
        args.extend(["--speculative-config",json.dumps(spec,sort_keys=True,
                                                        separators=(",",":"))])
    return args


def main():
    p=argparse.ArgumentParser(description="Plan 6 modes / 14 benchmark arms; never allocate GPU")
    p.add_argument("--model",default="qwen35_4b",choices=tuple(load_catalog()["models"]))
    p.add_argument("--mode",choices=MODES)
    p.add_argument("--method",choices=METHODS)
    p.add_argument("--target-path")
    p.add_argument("--draft-path")
    p.add_argument("--served-name")
    p.add_argument("--format",choices=("json","table"),default="json")
    a=p.parse_args()
    plans=[make_arm(a.model,a.mode,a.method)] if a.mode else matrix(a.model)
    if not a.mode and a.method:
        p.error("--method is meaningful only with --mode")
    result={"model":a.model,"arm_count":len(plans),"no_training":True,
            "gpu_executed":False,"arms":[arm.to_dict() for arm in plans]}
    if a.target_path:
        if len(plans)!=1:
            p.error("--target-path requires exactly one --mode")
        result["argv"]=command_for_arm(plans[0],a.target_path,
             draft_path=a.draft_path,served_name=a.served_name)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
