# ADR-0001 — Model and dataset cache policy

Status: ACCEPTED as engineering policy; GPU transfer timings not yet measured.

## Decision

- Models: default to full-commit pinned Hugging Face snapshot_download into
  Colab ephemeral local disk, once per GPU session, then reuse across baseline
  and candidate arms. An independent host-side Hugging Face cache is useful,
  but a dedicated host filesystem is NOT automatically visible to Colab.
- Datasets: retain original immutable dataset and checksum manifests on the
  dedicated persistent server. Optionally keep approved, versioned archives
  on Google Drive. Before inference, copy/extract and verify into Colab local
  disk. Real dataset selection, identity and splits remain frozen.
- Drive weights mirror: optional only when empirical download instability,
  gated access, or repeated large transfers justify the extra quota/storage
  and copying overhead. Never treat a Drive mount as live model filesystem for
  the timed benchmark.
- Results: publish raw logs, model/profile metadata, GPU identity, metric
  files and checksums to persistent storage after each cell.

## Mechanisms implemented

research_ops.assets actions:

- validate: asset manifest schema and pin checks, offline.
- plan: first available local directory/archive or pinned HF fallback, offline.
- stage --execute: exact SHA256 verify, safe unpack or pinned HF download,
  frozen expected-file verification, full tree SHA256, atomic staged receipt.
- verify: rehash staged tree and verify immutable manifest identity.
- pack --execute: optional TAR mirror plus SHA256 sidecar; output path may be
  a Google Drive mounted directory when explicitly authorized. This action
  does NOT invoke Google Drive OAuth, or guarantee remote replication.

Archives are TAR/TAR.GZ/ZIP, with file-count/expanded-byte limits and no
symlinks, hardlinks or path traversal. A corrupt cache fails closed instead
of silently falling back to a different source. Explicit source selection
uses --source-index. Check local GPU disk size and Drive storage quota.

## Evidence

Google Colab FAQ documents mounted Drive slow/random IO, quotas and per-file
errors, and recommends copying archives into local VM storage for intensive
use: https://research.google.com/colaboratory/faq.html

Hugging Face documentation describes content-addressed snapshots, revisions
and reuse: https://huggingface.co/docs/huggingface_hub/guides/manage-cache

This repo never commits user model weights, media bytes, tokens or private paths.
