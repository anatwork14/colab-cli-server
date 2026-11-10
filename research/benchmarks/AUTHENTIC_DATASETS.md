# Author-controlled dataset sources for benchmarks

**Decision date: 2026-11-10.** Only author-maintained Hugging Face datasets are
primary data sources; GitHub author repositories are used for official
evaluation code reference, not as runtime dependencies.

| Dataset | Primary bytes | Evaluation code | Expected |
| --- | --- | --- | --- |
| MemLens | https://huggingface.co/datasets/xiyuRenBill/MEMLENS | https://github.com/xrenaf/MEMLENS | 789 question IDs per 32K,64K,128K,256K configuration; 4,695 original release images |
| MMLongBench-Doc | https://huggingface.co/datasets/yubo2333/MMLongBench-Doc | https://github.com/mayubo2333/MMLongBench-Doc | 1,091 questions; 135 source PDFs |

The selected source catalog at OFFICIAL_SOURCES.json records author links,
license boundaries and checked code revisions. MemLens question/session
annotations are CC-BY-4.0; its sourced images may have different original
copyright/license. MMLongBench-Doc is CC-BY-NC-4.0 for research use. Honor
all distribution and access restrictions.

## Important: do not use moving main or abbreviated hashes

On the Ubuntu dedicated server with an authorized network connection:

    python3 -m pip install "huggingface_hub>=0.34,<2" "pyarrow>=15,<25" "pymupdf>=1.24,<2"

    python3 -m research_ops.benchmarks.sources resolve --kind memlens --variant 32k
    python3 -m research_ops.benchmarks.sources resolve --kind mmlongbench_doc

These are read-only metadata operations. They print FULL Hugging Face commit
SHAs after verifying that the publisher's file inventory contains the
required questions and 4,695 images or 135 PDFs. Do not guess a full SHA
from web UI short abbreviations or silently substitute a mirror.

## Plan downloads before allocating storage/bandwidth

Replace the symbolic placeholders with actual full 40-hex commit IDs:

    python3 -m research_ops.benchmarks.sources plan --kind memlens --variant 32k --revision MEMLENS_FULL_40_HEX_SHA --target /mnt/benchmark-assets/memlens-32k

    python3 -m research_ops.benchmarks.sources plan --kind mmlongbench_doc --revision MMLONG_FULL_40_HEX_SHA --target /mnt/benchmark-assets/mmlongbench-doc

These plan commands are offline, do not allocate GPUs and create no files.

## Fetch original publisher files after approval

    python3 -m research_ops.benchmarks.sources fetch --kind memlens --variant 32k --revision MEMLENS_FULL_40_HEX_SHA --target /mnt/benchmark-assets/memlens-32k --execute

    python3 -m research_ops.benchmarks.sources fetch --kind mmlongbench_doc --revision MMLONG_FULL_40_HEX_SHA --target /mnt/benchmark-assets/mmlongbench-doc --execute

MemLens downloads original dataset_32k.json and release_images/; for 64K,
128K and 256K use separately pinned variants. Only one original JSON
variant is transferred per call, not redundant Parquet conversions.

MMLongBench-Doc downloads publisher data/train-00000-of-00001.parquet and
documents/*.pdf. The script losslessly converts the 1,091 original
Parquet rows into samples.json for the existing runner, without creating,
altering, filtering or reordering questions. It records the raw Parquet
and converted JSON SHA256. It never substitutes third-party PDFs.

Both complete inventory and every referenced image/PDF undergo validation.
PyMuPDF checks original PDF page structure. A successful fetch emits a
source-receipt JSON file adjacent to the downloaded directory, containing
the immutable HF SHA, original question SHA, entire directory tree hash
and integrity audit. Incomplete/corrupted snapshots are BLOCKED; an
existing target is never silently overwritten. Partial transfers may leave
a diagnostics directory; recover deliberately.

## Audit existing data without falsely claiming HF origin

    python3 -m research_ops.benchmarks.sources audit --kind memlens --variant 32k --target /mnt/benchmark-assets/memlens-32k

    python3 -m research_ops.benchmarks.sources audit --kind mmlongbench_doc --target /mnt/benchmark-assets/mmlongbench-doc

Old mirrors can be validated for content coverage but cannot automatically
claim publisher provenance without matching pinned source receipts.

## Colab transfer/cache decision

Persist publisher originals and receipts on the dedicated server. Optionally
archive and mirror to Google Drive only when warranted by repeated download
cost/time and compatible with license and storage quota. Use
research_ops.assets for verification and controlled transfer to a running
Colab GPU. Rehash after transfer. ALL timing/inference uses local /content
disk, never mounted Drive FUSE or a remote PDF URL.

## Release/scoring distinction

MMLongBench HF publisher data has a November 6, 2025 evidence-page
correction; older GitHub samples may not match. Do not mix old GitHub
samples.json with the newer HF Parquet rows. Pin the exact data SHA and
qualify answer extraction + official scoring against this release.

MemLens official results require its separate LLM judge. MMLongBench-Doc
requires official answer extraction and eval_score.py. Full authentic
data does NOT equal qualified official accuracy, F1 or hallucination.

No dataset bytes or model weights are committed. No GPU allocation,
model training, fine-tuning, synthetic benchmark data, external repository
integration or paid LLM judge is triggered by the source tools.
