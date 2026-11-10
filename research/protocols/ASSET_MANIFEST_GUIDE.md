# Asset manifest and cache staging guide

The machine used for orchestration (dedicated server) and the execution
environment (Colab GPU) do not automatically share a filesystem. GPU runs
always load weights/dataset images from the GPU worker's local disk.

## Preparing a private, executable assets.json

Use research/protocols/assets.example.json as a starting point, then fill
the following structure with REAL revision and checksum values.

    {
      "schema_version": 1,
      "assets": [
        {
          "id": "qwen-model",
          "role": "model",
          "expected_files": ["config.json"],
          "license_status": "REVIEWED",
          "sources": [
            {
              "type": "archive",
              "path": "/content/drive/MyDrive/inference-assets/qwen-model.tar",
              "sha256": "REPLACE_WITH_REAL_SHA256"
            },
            {
              "type": "huggingface",
              "repo_id": "Qwen/Qwen3.5-4B",
              "revision": "REPLACE_WITH_ACTUAL_FULL_40_HEX_COMMIT"
            }
          ]
        },
        {
          "id": "real-dataset",
          "role": "dataset",
          "expected_files": ["manifest.json", "samples.jsonl"],
          "license_status": "REVIEWED",
          "sources": [
            {
              "type": "archive",
              "path": "/content/drive/MyDrive/inference-assets/real-dataset.tar",
              "sha256": "REPLACE_WITH_REAL_SHA256"
            }
          ]
        }
      ]
    }

The example above is **illustrative only** and intentionally invalid until
real checksums/revisions and access permissions have been verified. It is
not an instruction to download a particular model or dataset. All actual
assets are selected by the operator and must use approved licensing and
official source revisions.

If Drive is not mounted, the missing Drive archive is skipped during source
selection and an explicitly configured pinned Hub source is selected for
the model. If a Drive archive is present but corrupt, execution FAILS rather
than silently downloading the alternate. To override intentionally:

    python3 -m research_ops.assets plan --manifest /abs/assets.json \
      --asset qwen-model --scratch /content/research-assets --source-index 1

A directory already cached on the persistent host is also supported:

    {
      "type": "directory",
      "path": "/abs/host/cache/my-model",
      "tree_sha256": "REPLACE_WITH_ACTUAL_SHA256_OF_DIRECTORY_TREE"
    }

This directory path is only usable in the process that can access it. The
server-side tree cannot magically be opened from the Colab VM. Use a verified
archive transfer or Hub redownload from Colab for such cases.

## Case A: default model-download route

On a *running, authorized GPU session*:

    python3 -m pip install huggingface_hub
    python3 -m research_ops.assets plan \
      --manifest /content/assets.json --asset qwen-model \
      --scratch /content/research-assets
    python3 -m research_ops.assets stage \
      --manifest /content/assets.json --asset qwen-model \
      --scratch /content/research-assets --source-index 1 --execute

This performs a real Hugging Face snapshot_download to /content local
storage using a full 40-hex revision. There is no automatic synthetic data
creation or model training. Subsequent calls reuse and reverify staged files
within the same Colab VM. Different sessions still have fresh ephemeral
disks. For gated repos, authenticate via protected user/Colab runtime
credentials, never via the asset manifest or Git.

## Case B: persistent Drive archive for frequently reused inputs

Prepare a verified staged asset on a persistent host, or on an authorized
worker with a mounted Drive:

    python3 -m research_ops.assets pack --manifest /abs/assets.json \
      --asset real-dataset --scratch /abs/staged-assets \
      --output /abs/persistent-cache/real-dataset.tar --execute

This creates a deterministic TAR and corresponding .sha256.json sidecar.
The archive can be moved into Google Drive using the existing authorized
Google Drive artifact system, an approved file copy, or the Google Drive UI.
When using a live mounted /content/drive destination, the packer refuses
to pretend that the path is persistent if the mount is absent.

On a GPU VM with the Drive mount already authorized, stage this pinned
archive (after updating its archive SHA256 in the manifest) into /content:

    python3 -m research_ops.assets stage \
      --manifest /content/assets.json --asset real-dataset \
      --scratch /content/research-assets --execute

Do NOT benchmark using /content/drive/... paths as model/dataset inputs.
Google Drive quota and FUSE I/O introduce confounding latency and failures.
When the archive is large, provision sufficient scratch space for the TAR,
expanded data, engine source, compile cache and result traces.

## Which approach should be used?

| Situation | Default |
| --- | --- |
| Public, stable model checkpoint accessible from Hugging Face | Pinned Hub download on each new GPU VM |
| Model used across many cells in one VM | Reuse the same verified /content staging |
| Gated model or repeated Hub transfer failure | Versioned Drive mirror if access/license permits |
| Dataset available on dedicated persistent host | Keep original there; transfer verified archive |
| Dataset already in Drive | Verified Drive archive, copied/extracted into GPU scratch |
| Model checkpoint updated | New commit, new asset receipt, new study cell |
| Code changes only | Keep weights/dataset fixed; rebuild/reinstall only the engine |

**Performance methodology**: record download/cache-fill and engine
build/warmup times separately from serving TTFT, TPOT, ITL, throughput and
GPU peak memory. Comparing cache-cold vs cache-warm startup is a different
experiment and should use different cells/protocols.

The repository does not automatically upload or download all model weights
because doing so could consume significant quota, bandwidth and storage.
Every staging or packing operation requires --execute.

## vLLM engine identity

For a baseline:

    python3 -m research_ops.engine plan --mode stock
    python3 -m research_ops.engine verify --mode stock --require-gpu

For Python-level engine modifications, use an existing local clone of full
vLLM source from the authorized revision:

    python3 -m research_ops.engine plan --mode source_python \
      --source-root /abs/full-vllm-checkout --source-revision FULL_40_HEX_SHA

Follow the emitted editable-install instructions on the worker, then run
engine verify with the same arguments and --require-gpu. For CUDA/C++ kernel
edits use --mode source_compiled instead, and rebuild native libraries.
A wheel still importing from site-packages while a modified checkout sits
elsewhere is a hard failure.

No other GitHub project is cloned, linked or imported by this standalone
research workspace.
