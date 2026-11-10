"""Build faithful multimodal requests from staged REAL benchmark entries.

Derived PDF page rasterizations are an input representation, not synthetic data.
No prompts, images or PDF pages are written into Git or metric receipts.
"""
import base64
import io
from pathlib import Path
import re
import time

from .datasets import (
    DatasetIntegrityError, evidence_pages, image_reference, locate_file,
    safe_relative, is_unanswerable,
)

SYSTEM_MEMLENS = ("Provide answers based on the given conversation history. "
                  "If the question cannot be answered based on the given conversation, "
                  'respond with "Insufficient information".')
INSTRUCTION_MEMLENS = "Directly output the answer with no extra output."
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".webp": "image/webp"}
HEADER = "=== Session {index} (Date: {date}) ==="


def text_block(value):
    return {"type": "text", "text": value}


def _image_data(path, remaining_bytes):
    path = Path(path)
    mime = MIME.get(path.suffix.lower())
    if mime is None:
        raise DatasetIntegrityError("Unsupported image codec: " + path.suffix)
    size = path.stat().st_size
    if size <= 0 or size > remaining_bytes:
        raise DatasetIntegrityError("Image or aggregate payload exceeds limit")
    with path.open("rb") as handle:
        data = handle.read()
    if len(data) != size:
        raise DatasetIntegrityError("Image changed while reading")
    return {"type": "image_url", "image_url": {
        "url": "data:" + mime + ";base64," + base64.b64encode(data).decode("ascii")
    }}, size


def _images_from_turn(text, image_infos, image_root, budget):
    blocks = []
    used = 0
    n_images = 0
    parts = re.split(r"(<image>)", text)
    refs = []
    for raw in image_infos:
        name = image_reference(raw)
        if not isinstance(name, str) or name.startswith(("http:", "https:")):
            raise DatasetIntegrityError("Remote image URL prohibited during benchmark")
        refs.append(locate_file(image_root, name, media=True))
    cursor = 0
    for part in parts:
        if part == "<image>":
            if cursor >= len(refs):
                raise DatasetIntegrityError("Placeholder has no matching local image")
            block, size = _image_data(refs[cursor], budget - used)
            blocks.append(block)
            used += size
            n_images += 1
            cursor += 1
        elif part:
            blocks.append(text_block(part))
    # Match upstream build_messages() semantics: images exceeding placeholder
    # count are attached after text, never dropped.
    while cursor < len(refs):
        block, size = _image_data(refs[cursor], budget - used)
        blocks.append(block)
        used += size
        n_images += 1
        cursor += 1
    return blocks, n_images, used


def memlens_messages(row, image_root, *, max_images=160, max_image_bytes=128 * 1024 * 1024):
    messages = [{"role": "system", "content": [text_block(SYSTEM_MEMLENS)]}]
    sessions = row.get("haystack_sessions", [])
    dates = row.get("haystack_dates", [])
    image_count, byte_count, turn_count = 0, 0, 0
    if not isinstance(sessions, list) or not sessions:
        raise DatasetIntegrityError("Empty MemLens conversation")
    for index, session in enumerate(sessions, 1):
        if isinstance(session, dict):
            turns = session.get("session", [])
            date = session.get("date", "unknown")
        else:
            turns = session
            date = dates[index - 1] if index - 1 < len(dates) else "unknown"
        if not isinstance(turns, list):
            raise DatasetIntegrityError("Malformed MemLens session")
        first = True
        for turn in turns:
            if turn.get("role") not in ("user", "assistant"):
                raise DatasetIntegrityError("Unsupported MemLens turn role")
            blocks, count, consumed = _images_from_turn(
                turn.get("content", ""), turn.get("images", []), image_root,
                max_image_bytes - byte_count)
            image_count += count
            byte_count += consumed
            if image_count > max_images:
                raise DatasetIntegrityError("Multimodal image count exceeds configured limit")
            if first:
                header = HEADER.format(index=index,date=date)
                if blocks and blocks[0]["type"] == "text":
                    blocks[0]["text"] = header + "\n" + blocks[0]["text"]
                else:
                    blocks.insert(0,text_block(header))
                first = False
            if blocks:
                messages.append({"role": turn["role"], "content": blocks})
                turn_count += 1
    messages.append({"role": "user", "content": [text_block(
        INSTRUCTION_MEMLENS + "\nQuestion Date: " +
        str(row.get("question_date", "unknown")) + "\nQuestion: " +
        row["question"] + "\n")]})
    return messages, {
        "images_sent":image_count, "image_bytes":byte_count,
        "conversation_sessions":len(sessions), "conversation_turns":turn_count,
        "input_kind":"original_memlens_multimodal_chat",
        "prompt_truncation":False,
    }


def doc_messages(row, dataset_root, *, max_pages=128,
                 dpi=72, max_image_bytes=192 * 1024 * 1024):
    if not 40 <= dpi <= 180:
        raise ValueError("dpi outside safe 40..180 range")
    if max_pages < 1:
        raise ValueError("max_pages must be positive")
    name = safe_relative(row["doc_id"])
    path = locate_file(dataset_root, "documents/" + name)
    try:
        import fitz
    except ImportError as exc:
        raise DatasetIntegrityError("PyMuPDF required for genuine PDF page rendering") from exc
    used = 0
    with fitz.open(str(path)) as doc:
        pages = len(doc)
        if pages < 1:
            raise DatasetIntegrityError("PDF contains no pages")
        if pages > max_pages:
            raise DatasetIntegrityError("Full PDF has more pages than configured; no silent truncation")
        blocks = [text_block(
            "Answer the question using the full document pages in order. "
            "If the document does not provide the answer, reply 'Not answerable'. "
            "Return only the answer.\n")]
        for i in range(pages):
            pix = doc[i].get_pixmap(dpi=dpi, alpha=False)
            data = pix.tobytes("png")
            used += len(data)
            if used > max_image_bytes:
                raise DatasetIntegrityError("Rendered full document exceeds image-byte budget")
            blocks.append(text_block("Page " + str(i+1) + " of " + str(pages) + ":"))
            blocks.append({"type":"image_url","image_url":{
                "url":"data:image/png;base64,"+base64.b64encode(data).decode("ascii")}})
    blocks.append(text_block("Question: " + row["question"] + "\nAnswer:"))
    return [{"role":"user","content":blocks}], {
        "images_sent": pages, "image_bytes":used, "pdf_pages":pages,
        "pdf_render_dpi":dpi, "input_kind":"full_doc_pdf_pages",
        "prompt_truncation":False,
        "is_unanswerable_reference":is_unanswerable(row),
        "evidence_page_count":len(evidence_pages(row)),
    }


def build(kind, row, root, *, max_images=160, max_pages=128,
          max_image_bytes=192*1024*1024, dpi=72):
    start = time.perf_counter()
    if kind == "memlens":
        messages, info = memlens_messages(
            row, root, max_images=max_images, max_image_bytes=max_image_bytes)
    elif kind == "mmlongbench_doc":
        messages, info = doc_messages(row, root, max_pages=max_pages,
            dpi=dpi, max_image_bytes=max_image_bytes)
    else:
        raise ValueError("Unrecognized benchmark kind")
    info["preparation_ms"] = (time.perf_counter()-start)*1000
    return messages, info
