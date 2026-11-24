# Recipes

A recipe is one model on one card: `recipes/<org>/<repo>/<card>.toml`, where `<org>/<repo>` is the model's Hugging Face repository. Users name the model by that repo id: `tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF`. It names the exact engine image, the exact model files, and the settings that were measured. The launcher only runs what a recipe describes. Changing `[engine]` or `[run]` needs a new receipt.

## Schema 1

```toml
schema = 1                 # required, integer
status = "draft"           # "draft" or "released"; the launcher refuses drafts unless --allow-draft

[model]
name       = "pn3"      # optional short alias; the repo id below is the model's name
title      = "..."         # one-line human name
hf_repo    = "org/repo"    # Hugging Face model repository: what users type, matched case-insensitively
revision   = "<40-hex>"    # the repository commit, never a branch name
file       = "x.gguf"      # path of the weights file inside the repository
sha256     = "<64-hex>"    # sha256 of that file; the launcher verifies it after download
size_bytes = 0             # exact size, for the disk-space check and progress
license    = "..."         # the model's license

[card]
id                 = "p100"   # matches the file name
title              = "NVIDIA Tesla P100 16 GB"
compute_capability = "6.0"    # must equal nvidia-smi's compute_cap
min_memory_mib     = 16000    # nvidia-smi memory.total must be at least this
min_driver         = "525.60.13"

[engine]
release = "engine-v0.1.0"     # the engine release this recipe was measured on
arch    = "sm60"
image   = "ghcr.io/iammrduncan/tensorless-engine@sha256:<64-hex>"   # always by digest

[run]
alias       = "org/repo"      # the served model name (/v1/models); the repo id
context     = 65536           # passed as -c
threads_max = 8               # -t and -tb are min(host CPUs, threads_max)
args        = ["-ngl", "99"]  # every other llama-server argument, in order
env         = { GGML_CUDA_FATTN_SM60 = "1" }   # engine switches, exactly as measured

[receipt]                     # what was measured, and where; all optional
source  = "..."               # who measured it and on what build
summary = "..."
decode_tps  = { short = 0.0 } # keys: short, ctx_32k, ctx_65k
prefill_tps = { ctx_4k = 0.0 }
notes   = ["..."]
```

The launcher adds `-m`, `--host`, `--port`, `-c`, `--alias`, `-t` and `-tb` itself, so recipes leave them out. Overriding the context or adding arguments at the command line is allowed, and the run is then reported as untuned.

## A pack instead of one file

Some models are not one GGUF file but a set of files that the engine reads in place, after a short prepare step writes a small GGUF that names them. Such a recipe leaves out `file`, `sha256` and `size_bytes`, and has a `[model.pack]` table instead:

```toml
[model]
title    = "..."
hf_repo  = "org/repo"
revision = "<40-hex>"
license  = "..."

[model.pack]
prepare = ["python3", "-E", "-s", "/opt/tensorless/inkling-prepare/prepare.py"]   # run in the engine image
files = [                                                                           # every file the pack needs
  { name = "config.json",      sha256 = "<64-hex>", size_bytes = 2699 },
  { name = "embed_packed.npz", sha256 = "<64-hex>", size_bytes = 214346212 },
]
```

- `files`: each file's path inside the repository (relative, forward slashes, no `..`, no `:`; each name once, compared without case), its sha256 and its exact size. The launcher downloads each into the Hugging Face cache and verifies it, as it does one file; the disk-space check counts their sum. One all-zero sha256 or one `size_bytes = 0` marks the recipe unpublished.
- `prepare`: a command, as an argv, that the launcher runs in the recipe's engine image before each serve, with the image's entrypoint replaced by `prepare[0]` and two more arguments: the pack's folder and an empty output folder. No GPU and no network. The last line it prints must be a path in the output folder; llama-server gets it as `-m`.

In both runs the repository's cache folder (`models--<org>--<repo>`, so that the snapshot's links into `blobs/` resolve) is mounted read-only at `/models/repo`, and the pack's folder is `/models/repo/snapshots/<revision>`. The output folder is `/models/prepared`: writable for the prepare step, read-only for the server. On the host it is `$XDG_STATE_HOME/tensorless/prepared/<org>--<repo>/<card>/<revision>-<image digest>/` (12 characters of each; `~/.local/state` without `XDG_STATE_HOME`). The prepared file may record the pack's folder, so the server sees the pack at the same path the prepare step saw.
