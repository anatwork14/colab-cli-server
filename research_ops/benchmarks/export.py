"""Lossless export of original questions + generated predictions for external official graders.

Does NOT import an external repository or claim official scoring parity.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .datasets import load_records, identity, is_unanswerable, evidence_pages
from .runner import _slug_receipt, _atomic_json


def prepare_export(run_root, samples, kind, output):
    root=Path(run_root).resolve()
    meta=json.loads((root/"run_config.json").read_text(encoding="utf8"))
    if meta["kind"]!=kind:
        raise ValueError("Wrong dataset kind for frozen run")
    from research_ops.assets import hash_file
    if meta["question_file_sha256"]!=hash_file(samples):
        raise ValueError("Wrong question file revision")
    source=load_records(kind,samples)
    selected=source[:meta["limit"]] if meta["limit"] else source
    if not selected:
        raise ValueError("No frozen original questions")
    data=[]
    failures=0
    for i,row in enumerate(selected):
        qid=identity(kind,row,i)
        file=_slug_receipt(root,qid)
        if not file.is_file():
            failures+=1
            receipt={"sample_id":qid,"status":"MISSING","prediction":""}
        else:
            receipt=json.loads(file.read_text(encoding="utf8"))
            if receipt.get("sample_id")!=qid:
                raise ValueError("Receipt identity mismatch")
            if receipt.get("status")!="SUCCESS":
                failures+=1
        answer=receipt.get("prediction","")
        if kind=="memlens":
            item={
                "question_id":qid,
                "question":row["question"],
                "question_type":row.get("question_type"),
                "question_subtype":row.get("question_subtype"),
                "old_answer":row.get("old_answer"),
                "reference_answer":row.get("answer"),
                "prediction":answer,
                "raw_prediction":answer,
                "inference_status":receipt.get("status"),
                "original_index":i,
            }
        else:
            item={
                "id":qid,"doc_id":row.get("doc_id"),
                "doc_type":row.get("doc_type"),
                "question":row.get("question"),
                "answer":row.get("answer"),
                "answer_format":row.get("answer_format"),
                "evidence_pages":row.get("evidence_pages"),
                "evidence_sources":row.get("evidence_sources"),
                "raw_response":answer,
                # MMLongBench official pipeline requires a separately run
                # answer extractor before valid pred + eval_score.
                "pred":None,
                "score":None,
                "requires_answer_extraction":True,
                "inference_status":receipt.get("status"),
                "original_index":i,
            }
        data.append(item)
    outfile=Path(output)
    outfile.parent.mkdir(parents=True,exist_ok=True)
    if outfile.exists():
        raise FileExistsError("Refusing to overwrite published evaluation export")
    payload={
        "schema_version":1,"dataset":kind,"source_run_id":meta["run_id"],
        "question_file_sha256":meta["question_file_sha256"],
        "status":"AWAITING_OFFICIAL_EVALUATOR",
        "official_accuracy":None,"official_f1":None,
        "completeness":len(selected)-failures,
        "population":len(selected),"failed_or_missing":failures,
        "subset":bool(meta.get("limit")),"data":data,
    }
    _atomic_json(outfile,payload)
    return {"status":payload["status"],"output":str(outfile),
            "entries":len(data),"failed_or_missing":failures,
            "official_score_qualified":False}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--kind",choices=("memlens","mmlongbench_doc"),required=True)
    p.add_argument("--run-root",required=True)
    p.add_argument("--samples",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    print(json.dumps(prepare_export(a.run_root,a.samples,a.kind,a.output),indent=2))


if __name__=="__main__":
    main()
