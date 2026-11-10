"""Pin and prepare publisher-authentic real MemLens/MMLongBench-Doc releases.

The only source of dataset bytes is an explicitly selected publisher-controlled
Hugging Face DATASET revision. This is not a training/data-generation script.
No cloud GPU, Google Drive API, repository submodule or judge is invoked.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

from research_ops.assets import hash_file, tree_info
from .datasets import audit, DatasetIntegrityError

_SHA40 = re.compile(r"^[a-f0-9]{40}$")
_VARIANTS = {
    "memlens": ("32k", "64k", "128k", "256k"),
    "mmlongbench_doc": ("default",),
}
_REPOS = {
    "memlens": "xiyuRenBill/MEMLENS",
    "mmlongbench_doc": "yubo2333/MMLongBench-Doc",
}
_COUNTS = {"memlens":789, "mmlongbench_doc":1091}
_SIDE = ".official-dataset-source-receipt.json"


def _select(kind, variant):
    if kind not in _VARIANTS or variant not in _VARIANTS[kind]:
        raise ValueError("Unsupported benchmark or release variant")
    return {
        "kind":kind,"variant":variant,"repo_id":_REPOS[kind],
        "expected_questions":_COUNTS[kind],
        "questions":("dataset_"+variant+".json" if kind=="memlens"
                     else "samples.json"),
        "files":(["dataset_"+variant+".json", "release_images/**",
                  "README.md", "LICENSE-DATA"] if kind=="memlens" else
                 ["data/train-*.parquet","documents/*.pdf","README.md"]),
        "official_repo_url":"https://huggingface.co/datasets/"+_REPOS[kind],
        "publisher_controlled":True,
    }


def _pin(revision):
    if not isinstance(revision,str) or not _SHA40.fullmatch(revision):
        raise ValueError("Benchmark sources require full 40-character immutable HF commit SHA")
    return revision


def _hf_api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub on the dedicated server") from exc
    return HfApi()


def resolve(kind, variant):
    """Network read-only: report actual full SHA without downloading dataset bytes."""
    spec=_select(kind,variant)
    info=_hf_api().dataset_info(repo_id=spec["repo_id"])
    sha=_pin(info.sha)
    files={f.rfilename for f in info.siblings}
    required=(
        {spec["questions"]} if kind=="memlens" else
        {"data/train-00000-of-00001.parquet"}
    )
    if not required.issubset(files):
        raise RuntimeError("Publisher dataset revision does not expose expected question file")
    if kind=="memlens":
        image_files=[n for n in files if n.startswith("release_images/") and
                     n.lower().endswith((".jpg",".jpeg",".png",".webp"))]
        if len(image_files)!=4695:
            raise RuntimeError("Publisher release image inventory changed: "+
                               str(len(image_files))+" != 4695; review before pinning")
    else:
        pdf_files=[n for n in files if n.startswith("documents/") and n.lower().endswith(".pdf")]
        if len(pdf_files)!=135:
            raise RuntimeError("Publisher release document inventory changed: "+
                               str(len(pdf_files))+" != 135; review before pinning")
    return {
        "status":"FULL_PUBLISHER_REVISION_RESOLVED_NO_DOWNLOAD",
        "catalog":spec,"hf_commit_sha":sha,
        "file_inventory_checked":True,
        "action_required":"Record full SHA, then explicitly run fetch --execute",
        "has_downloaded_bytes":False,
    }


def _parquet_to_original_json(root):
    """Format conversion ONLY: no new annotations/questions, no reordering."""
    import glob
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install pyarrow to read publisher's original Parquet file") from exc
    input_files=sorted(Path(root).glob("data/train-*.parquet"))
    if len(input_files)!=1:
        raise DatasetIntegrityError("Expected exactly one publisher train Parquet shard")
    output=Path(root)/"samples.json"
    if output.exists():
        raise FileExistsError("Derived question JSON already exists, refusing overwrite")
    table=pq.read_table(str(input_files[0]))
    rows=table.to_pylist()
    if len(rows)!=1091:
        raise DatasetIntegrityError("Publisher Parquet question count != 1091")
    required={"doc_id","doc_type","question","answer","evidence_pages",
              "evidence_sources","answer_format"}
    if any(not required.issubset(row) for row in rows):
        raise DatasetIntegrityError("Publisher question fields missing")
    # Do not synthesize sample IDs, alter answers or coerce evidence labels.
    with output.open("x",encoding="utf-8") as f:
        json.dump(rows,f,ensure_ascii=False,indent=2)
        f.write("\n")
    return {"original_parquet":input_files[0].relative_to(root).as_posix(),
            "original_parquet_sha256":hash_file(input_files[0]),
            "json_conversion":"LOSSLESS_ROW_PRESERVING_FORMAT_CONVERSION",
            "questions_json_sha256":hash_file(output),
            "questions_original":len(rows),"questions_converted":len(rows)}


def _inventory(spec, root, deep_pdf):
    root=Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError("Real publisher data directory absent")
    kind=spec["kind"]
    extra={}
    if kind=="memlens":
        file=root/spec["questions"]
        if not file.is_file():
            raise FileNotFoundError("Original publisher JSON absent: "+str(file))
        images=list((root/"release_images").rglob("*")) if (root/"release_images").exists() else []
        image_count=sum(1 for p in images if p.is_file() and
                p.suffix.lower() in (".jpg",".jpeg",".png",".webp"))
        if image_count!=4695:
            raise DatasetIntegrityError("MemLens full image inventory incomplete: "+str(image_count))
        extra["total_author_images"]=image_count
    else:
        pdfs=list((root/"documents").glob("*.pdf"))
        if len(pdfs)!=135 or any(p.is_symlink() for p in pdfs):
            raise DatasetIntegrityError("MMLongBench official PDF inventory must contain 135 files")
        if not (root/"samples.json").is_file():
            raise FileNotFoundError("Call fetch or prepare to convert publisher Parquet to samples.json")
        extra["total_author_pdfs"]=len(pdfs)
    report=audit(kind,root/spec["questions"],root,
                 expected_count=spec["expected_questions"],deep_pdf=deep_pdf)
    if report["status"]!="COMPLETE_FOR_ENGINEERING_PILOT":
        raise DatasetIntegrityError("Original source failed full integrity audit: "+
                                    json.dumps(report["blocking_reasons"],sort_keys=True))
    return {"audit":report,"additional_inventory":extra,
            "tree":tree_info(root),
            "question_json_sha256":hash_file(root/spec["questions"])}


def _receipt_path(root):
    root=Path(root).resolve()
    return root.parent/(root.name+_SIDE)


def audit_existing(kind,variant,root,*,deep_pdf=False):
    spec=_select(kind,variant)
    root=Path(root).resolve()
    info=_inventory(spec,root,deep_pdf=deep_pdf or kind=="mmlongbench_doc")
    previous=_receipt_path(root)
    status="AUDITED_LOCAL_DATA_WITHOUT_CONFIRMED_HF_PROVENANCE"
    if previous.exists():
        rec=json.loads(previous.read_text(encoding="utf-8"))
        if (rec.get("repo_id")==spec["repo_id"] and
                rec.get("hf_commit_sha") and
                rec.get("tree_sha256")==info["tree"]["tree_sha256"] and
                rec.get("question_sha256")==info["question_json_sha256"]):
            status="PUBLISHER_PINNED_AND_CONTENT_REVERIFIED"
    return dict(status=status,repo_id=spec["repo_id"],
                release_variant=variant,**info)


def fetch(kind,variant,revision,target,execute=False):
    spec=_select(kind,variant)
    sha=_pin(revision)
    target=Path(target).expanduser()
    if not target.is_absolute():
        raise ValueError("--target must be absolute; never write into git working tree implicitly")
    target=target.resolve()
    side=_receipt_path(target)
    if target.exists() or side.exists():
        raise FileExistsError("Target/receipt already exists; use audit for prior data; do not overwrite")
    if not execute:
        return dict(status="PLAN_ONLY_NO_DOWNLOAD",repo_id=spec["repo_id"],
                    hf_commit_sha=sha,target=str(target),selected_patterns=spec["files"])
    info=_hf_api().dataset_info(repo_id=spec["repo_id"],revision=sha)
    if _pin(info.sha)!=sha:
        raise RuntimeError("Requested immutable SHA did not resolve to same revision")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub to download the original release") from exc
    target.parent.mkdir(parents=True,exist_ok=True)
    # Download only the real dataset questions and associated original images/PDFs;
    # never mirror extra paraphrased or third-party files.
    actual=snapshot_download(repo_id=spec["repo_id"],repo_type="dataset",
             revision=sha,local_dir=str(target),allow_patterns=spec["files"])
    if Path(actual).resolve()!=target:
        raise RuntimeError("Hugging Face snapshot unexpectedly staged outside explicit target")
    conversion=None
    if kind=="mmlongbench_doc":
        conversion=_parquet_to_original_json(target)
    report=_inventory(spec,target,deep_pdf=kind=="mmlongbench_doc")
    result={
        "status":"PUBLISHER_REVISION_FETCHED_DATA_INTEGRITY_VERIFIED",
        "repo_id":spec["repo_id"],"dataset_url":spec["official_repo_url"],
        "hf_commit_sha":sha,"variant":variant,"downloaded_utc":datetime.now(
            timezone.utc).isoformat(),
        "input_format":"PUBLISHER_JSON" if kind=="memlens" else "PUBLISHER_PARQUET",
        "conversion":conversion,"question_file":spec["questions"],
        "question_sha256":report["question_json_sha256"],
        "tree_sha256":report["tree"]["tree_sha256"],
        "files":report["tree"]["file_count"],
        "bytes":report["tree"]["total_bytes"],
        "audit":report["audit"],
        "official_accuracy_computed":False,
        "official_judge_run":False,
        "no_generated_questions":True,
    }
    # Sidecar is outside the hashed dataset tree. Never confuse source
    # pin verification with official benchmark scoring qualification.
    with side.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,sort_keys=True)
        f.write("\n")
    return {"status":result["status"],"source_receipt":str(side),
            "question_sha256":result["question_sha256"],"tree_sha256":result["tree_sha256"],
            "hf_commit_sha":sha,"audit_status":report["audit"]["status"]}


def prepare_existing(kind,variant,target,execute=False):
    spec=_select(kind,variant)
    if kind!="mmlongbench_doc":
        return audit_existing(kind,variant,target)
    path=Path(target).resolve()
    if not path.is_dir():
        raise FileNotFoundError("Original publisher directory missing")
    if (path/"samples.json").exists():
        return audit_existing(kind,variant,path)
    if not execute:
        return {"status":"PLAN_ONLY_PARQUET_CONVERSION_NOT_PERFORMED",
                "source":str(path),"no_questions_synthesized":True}
    converted=_parquet_to_original_json(path)
    report=_inventory(spec,path,deep_pdf=True)
    return {"status":"LOCAL_PUBLISHER_FORMAT_CONVERTED_ORIGIN_NOT_YET_CONFIRMED",
            "conversion":converted,"audit":report["audit"],
            "note":"Existing local files alone cannot establish authentic HF commit provenance."}


def main():
    p=argparse.ArgumentParser(description="Fetch publisher sources, pin a full SHA; no training.")
    p.add_argument("action",choices=("resolve","plan","fetch","prepare","audit"))
    p.add_argument("--kind",choices=sorted(_REPOS),required=True)
    p.add_argument("--variant",help="MemLens: 32k/64k/128k/256k; MMLong: default")
    p.add_argument("--revision",help="Full 40-hex commit, required for plan/fetch")
    p.add_argument("--target",help="Absolute dataset root")
    p.add_argument("--execute",action="store_true",help="Required for downloads or JSON conversion")
    a=p.parse_args()
    variant=a.variant or ("32k" if a.kind=="memlens" else "default")
    if a.action=="resolve":
        result=resolve(a.kind,variant)
    elif a.action in ("plan","fetch"):
        if not a.target or not a.revision:
            p.error("plan/fetch require --target and full --revision from resolve")
        result=fetch(a.kind,variant,a.revision,a.target,
                     execute=a.action=="fetch" and a.execute)
        if a.action=="fetch" and not a.execute:
            p.error("fetch requires --execute; use plan to inspect first")
    elif not a.target:
        p.error("audit/prepare require --target existing original publisher directory")
    elif a.action=="prepare":
        result=prepare_existing(a.kind,variant,a.target,execute=a.execute)
    else:
        result=audit_existing(a.kind,variant,a.target)
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
