"""Auditable readers for real MemLens and MMLongBench-Doc releases.

Never synthesize benchmark rows or silently remove missing-document questions.
External Hugging Face snapshots and Drive contents must be staged separately.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

KIND = ("memlens", "mmlongbench_doc")


class DatasetIntegrityError(ValueError):
    pass


def safe_relative(path):
    if not isinstance(path, str) or not path or any(ord(ch) < 32 for ch in path):
        raise DatasetIntegrityError("Invalid or empty dataset-relative path")
    path = path[2:] if path.startswith("./") else path
    if ("://" in path or ":" in path.split("/",1)[0] or "\\" in path or
            path.startswith("/") or any(x in ("", ".", "..") for x in path.split("/")):
        raise DatasetIntegrityError("Unsafe dataset-relative path")
    return path


def safe_file(root, relative):
    name = safe_relative(relative)
    base = Path(root).resolve()
    result = (base / name).resolve()
    if not _contained(base, result):
        raise DatasetIntegrityError("Asset escapes frozen dataset root")
    return result


def _contained(base, path):
    try:
        Path(path).resolve().relative_to(Path(base).resolve())
        return True
    except ValueError:
        return False


def locate_file(root, raw_path, *, media=False):
    root = Path(root).resolve()
    name = safe_relative(raw_path)
    locations = ([root / "release_images" / name, root / name] if media
                 else [root / name])
    candidates = [p.resolve() for p in locations if p.is_file() and not p.is_symlink()]
    candidates = [p for p in candidates if _contained(root, p)]
    if not candidates:
        raise DatasetIntegrityError("Referenced local asset not staged: " + name)
    if len(set(candidates)) != 1:
        # Two copies with same path need explicit resolution, not silent preference.
        raise DatasetIntegrityError("Ambiguous asset path: " + name)
    return candidates[0]


def image_reference(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        raw = next((value[k] for k in ("file", "path", "file_path", "img_file")
                    if value.get(k)), None)
        if isinstance(raw, list):
            if len(raw) != 1:
                raise DatasetIntegrityError("Image path list must have exactly one entry")
            raw = raw[0]
        return raw
    raise DatasetIntegrityError("Unsupported MemLens image metadata")


def load_records(kind, samples):
    if kind not in KIND:
        raise ValueError("Unknown dataset kind")
    file = Path(samples)
    if not file.is_file() or file.is_symlink():
        raise FileNotFoundError("Real dataset JSON file missing: " + str(file))
    with file.open("r", encoding="utf-8") as h:
        raw = json.load(h)
    rows = raw.get("data") if isinstance(raw, dict) and kind == "memlens" else raw
    if not isinstance(rows, list) or not rows or not all(isinstance(x, dict) for x in rows):
        raise DatasetIntegrityError("Expected nonempty list of REAL question records")
    return rows


def identity(kind, row, position):
    if kind == "memlens":
        rid = row.get("question_id")
        if rid is None or not str(rid).strip():
            raise DatasetIntegrityError("MemLens question_id absent")
        return str(rid)
    # Original MMLongBench-Doc JSON has no universal question_id. Freeze its
    # original index and file checksum; never renumber after filtering.
    rid = row.get("question_id", row.get("id"))
    return str(rid) if rid is not None and str(rid).strip() else "original-index-" + str(position)


def evidence_pages(row):
    value = row.get("evidence_pages", [])
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise DatasetIntegrityError("Malformed evidence_pages") from exc
    if not isinstance(value, list):
        raise DatasetIntegrityError("evidence_pages must be a list")
    if not all(type(x) is int and x >= 0 for x in value):
        raise DatasetIntegrityError("Invalid evidence page index")
    return value


def is_unanswerable(row):
    return str(row.get("answer", "")).strip().lower() in (
        "not answerable", "unanswerable", "insufficient information")


def audit(kind, samples, root, *, expected_count=None, deep_pdf=False):
    """Return full-denominator audit. No filtered or reduced evaluation set."""
    rows = load_records(kind, samples)
    base = Path(root).resolve()
    sf = Path(samples).resolve()
    if not _contained(base, sf):
        raise DatasetIntegrityError("JSON questions must be inside staged dataset root")
    if expected_count is None or type(expected_count) is not int or expected_count <= 0:
        raise DatasetIntegrityError("expected_count REQUIRED from a pinned release; no guessed denominators")
    reasons = Counter()
    categories = Counter()
    missing = []
    seen = set()
    referenced_pdfs = set()
    referenced_images = set()
    counts = {"questions_total":len(rows), "unanswerable_total":0,
              "cross_page_total":0, "source_file_sha256": _hash_file(sf),
              "expected_questions":expected_count}
    if len(rows) != expected_count:
        reasons["unexpected_question_count"] += abs(len(rows) - expected_count)
    fitz = None
    if kind == "mmlongbench_doc":
        if deep_pdf:
            try:
                import fitz as fitz_module
            except ImportError as exc:
                raise DatasetIntegrityError("Install PyMuPDF for deep PDF validation") from exc
            fitz = fitz_module
        else:
            reasons["pdf_deep_validation_not_run"] += 1
    for i, row in enumerate(rows):
        try:
            rid = identity(kind, row, i)
            if rid in seen:
                raise DatasetIntegrityError("Duplicate question ID")
            seen.add(rid)
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise DatasetIntegrityError("Missing question text")
            if row.get("answer") is None:
                raise DatasetIntegrityError("Missing gold reference answer")
            if kind == "memlens":
                kind_key = str(row.get("question_type", "UNLABELLED"))
                categories[kind_key] += 1
                sessions = row.get("haystack_sessions")
                if not isinstance(sessions, list) or not sessions:
                    raise DatasetIntegrityError("No conversation history")
                for session in sessions:
                    turns = session.get("session") if isinstance(session, dict) else session
                    if not isinstance(turns, list):
                        raise DatasetIntegrityError("Malformed session turns")
                    for turn in turns:
                        if not isinstance(turn, dict) or turn.get("role") not in ("user", "assistant"):
                            raise DatasetIntegrityError("Invalid turn or conversation role")
                        if not isinstance(turn.get("content", ""), str):
                            raise DatasetIntegrityError("Invalid conversation turn text")
                        images = turn.get("images", [])
                        if not isinstance(images, list):
                            raise DatasetIntegrityError("images is not a list")
                        for img in images:
                            path = image_reference(img)
                            if not isinstance(path, str) or path.startswith(("http:", "https:")):
                                raise DatasetIntegrityError("Local image absent; URL streaming forbidden")
                            local = locate_file(base, path, media=True)
                            referenced_images.add(str(local))
            else:
                pdf = row.get("doc_id")
                if not isinstance(pdf, str) or not pdf.lower().endswith(".pdf"):
                    raise DatasetIntegrityError("Missing PDF doc_id")
                path = locate_file(base, "documents/" + safe_relative(pdf))
                referenced_pdfs.add(str(path))
                with path.open("rb") as stream:
                    if stream.read(5) != b"%PDF-":
                        raise DatasetIntegrityError("Invalid PDF header")
                pages = evidence_pages(row)
                if is_unanswerable(row):
                    counts["unanswerable_total"] += 1
                elif len(set(pages)) > 1:
                    counts["cross_page_total"] += 1
                categories[str(row.get("doc_type", "UNLABELLED"))] += 1
                if deep_pdf:
                    with fitz.open(str(path)) as doc:
                        if len(doc) < 1:
                            raise DatasetIntegrityError("PDF has zero pages")
                        if pages and max(pages) > len(doc):
                            # Avoid assuming zero/one-based indices; both agree if <= len.
                            raise DatasetIntegrityError("Evidence page index exceeds PDF size")
        except (DatasetIntegrityError, OSError, ValueError) as exc:
            reason = str(exc).split(":",1)[0][:100]
            reasons[reason] += 1
            if len(missing) < 30:
                missing.append({"original_index":i,
                                "id":str(row.get("question_id",row.get("id",i)))[:90],
                                "reason":reason})
    status = "COMPLETE_FOR_ENGINEERING_PILOT" if not reasons else "BLOCKED_INTEGRITY_OR_COVERAGE"
    return {"dataset":kind,"status":status,"source_json":str(sf),
            "source_sha256":counts["source_file_sha256"],
            "expected_questions":expected_count,"questions_total":len(rows),
            "validated_question_ids":len(seen), "unique_image_files":len(referenced_images),
            "unique_pdf_files":len(referenced_pdfs),
            "unanswerable_questions":counts["unanswerable_total"],
            "cross_page_questions":counts["cross_page_total"],
            "categories":dict(categories),"blocking_reasons":dict(reasons),
            "first_30_issues":missing,
            "data_generation":False,"allow_silent_drop":False,
            "official_scoring_qualified":False}


def _hash_file(path):
    from research_ops.assets import hash_file
    return hash_file(path)


def main():
    p = argparse.ArgumentParser(description="Offline audit of REAL multimodal benchmark data.")
    p.add_argument("--kind", choices=KIND, required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--samples", required=True)
    p.add_argument("--expected-count", type=int, required=True)
    p.add_argument("--deep-pdf", action="store_true")
    p.add_argument("--output", help="Write audit JSON once; no overwrite")
    a = p.parse_args()
    result = audit(a.kind,a.samples,a.root,expected_count=a.expected_count,deep_pdf=a.deep_pdf)
    if a.output:
        target = Path(a.output)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("x",encoding="utf-8") as f:
            json.dump(result,f,indent=2)
            f.write("\n")
    print(json.dumps(result,indent=2))
    if result["status"] != "COMPLETE_FOR_ENGINEERING_PILOT":
        raise SystemExit(2)

if __name__ == "__main__":
    main()
