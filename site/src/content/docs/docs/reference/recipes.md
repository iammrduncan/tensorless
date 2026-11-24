---
title: "Recipe file"
description: "The recipe file, schema 1: every table and key, one model file or a pack of files, the checks the launcher applies, and what makes a recipe a draft or unpublished."
type: reference
sidebar:
  order: 2
---

A recipe is one model on one card. It names the exact engine image, the exact model files, the settings that were measured, and the numbers they gave. The launcher only runs what a recipe describes.

## Location

```text
recipes/<org>/<repo>/<card>.toml
```

- `<org>/<repo>` is the model's Hugging Face repository, and must equal `model.hf_repo`.
- `<card>` is the card id, and must equal `card.id`.
- The built-in recipes are the repository's [`recipes/`](https://github.com/iammrduncan/tensorless/tree/main/recipes) directory, embedded into the launcher when it is built. `--recipes DIR` reads `DIR/<org>/<repo>/<card>.toml` instead.
- Other files, and files and directories whose name starts with `.`, are ignored.
- One invalid recipe fails the whole load.

## Example

```toml title="recipes/iamMrDuncan/Qwen3.8-27B-PN3-GGUF/p100.toml (part)"
schema = 1
status = "released"

[model]
name       = "pn3"
title      = "Qwen3.8-27B-PN3 (P100-native 3-bit, mix P0k+19)"
hf_repo    = "iamMrDuncan/Qwen3.8-27B-PN3-GGUF"
revision   = "c701b6690ec360b16d40f85df1cb76062e30cd38"
file       = "Qwen3.8-27B-PN3-P0k19.gguf"
sha256     = "15ce2d4469673f1f1753c55de3780cfe04311dd6f05cb41633d73df128e91476"
size_bytes = 12954114496
license    = "Apache-2.0 (from Qwen/Qwen3.8-27B)"

[card]
id                 = "p100"
title              = "NVIDIA Tesla P100 16 GB"
compute_capability = "6.0"
min_memory_mib     = 16000
min_driver         = "525.60.13"

[engine]
release = "engine-v0.3.0"
arch    = "sm60"
image   = "ghcr.io/iammrduncan/tensorless-engine@sha256:3423499a1676c557341bc7e959b65845b59cd791cc7b30e36f862c4c5528bd5e"

[run]
alias       = "iamMrDuncan/Qwen3.8-27B-PN3-GGUF"
context     = 65536
threads_max = 8
args = ["-np", "1", "-ngl", "99", "-ot", "token_embd.weight=CUDA0", "-ctk", "q4_0", "-ctv", "q4_0",
        "-sm", "none", "-lm", "none", "-fit", "off", "--threads-http", "4", "-b", "512", "-ub", "512",
        "-fa", "on", "--jinja"]

[run.env]
GGML_CUDA_FATTN_SM60 = "1"
# ... the other engine switches

[receipt]
summary     = "Prefill about 200 tok/s on a cold prompt of about 4,100 tokens, 1.65x engine-v0.1.0. Decode about 30 tok/s at short context, 1.14x the Q3 baseline model on the same card. Not yet measured at 32K or 65K."
decode_tps  = { short = 30.43 }
prefill_tps = { ctx_4k = 200.1 }
```

The full files are in [`recipes/`](https://github.com/iammrduncan/tensorless/tree/main/recipes). The schema's own description is [recipes/README.md](https://github.com/iammrduncan/tensorless/blob/main/recipes/README.md).

## Top level

| Key | Type | Required | Meaning |
|---|---|---|---|
| `schema` | integer | yes | The recipe schema. Must be `1`; any other value fails with a request to update tensorless. |
| `status` | string | yes | `"draft"` or `"released"`. See [Draft, released and unpublished](#draft-released-and-unpublished). |

## [model]

| Key | Type | Required | Meaning |
|---|---|---|---|
| `name` | string | no | A short alias, such as `pn3`, that works anywhere a repo id does. Lowercase letters, digits, `.`, `_` and `-`. It must not equal any repo id, nor another model's short name. |
| `title` | string | no | A one-line human name, shown by `info` and `models --json`. |
| `hf_repo` | string | yes | The Hugging Face model repository, `org/name`. This is what users type, matched without case. Two repo ids that differ only in case are refused. |
| `revision` | string | yes | The repository commit: 40 lowercase hex characters. Never a branch or tag name. |
| `file` | string | yes, without `[model.pack]` | The weights file's path inside the repository: relative, with forward slashes, clean (no `..`), without `:` or newlines. |
| `sha256` | string | yes, without `[model.pack]` | The file's sha256, 64 lowercase hex characters. The launcher checks the download against it. |
| `size_bytes` | integer | yes, without `[model.pack]` | The file's exact size, for the disk-space check and progress. Must not be negative. `0` marks the recipe unpublished. |
| `license` | string | yes | The model's license. |

A recipe names one file (`file`, `sha256`, `size_bytes`) or a pack (`[model.pack]`), never both: a recipe with both is refused.

## [model.pack]

A pack is a model that is several files of its repository, which the engine reads in place after a prepare step. All the files come from `hf_repo` at `revision`.

```toml title="A pack (part)"
[model.pack]
prepare = ["python3", "-E", "-s", "/opt/tensorless/inkling-prepare/prepare.py"]
files = [
  { name = "config.json",      sha256 = "8216dd43157f5474cfff88f486a1e433fef703c674f694e485a86a761c412b1e", size_bytes = 2699 },
  { name = "embed_packed.npz", sha256 = "41e80f05815383d5ddf485d2ace7e00f3f50b522ceda84fd5bd4acfc6337b428", size_bytes = 214346212 },
]
```

| Key | Type | Required | Meaning |
|---|---|---|---|
| `prepare` | array of strings | yes | The command that prepares the pack, as an argv, run in the recipe's engine image. Not empty; no empty argument and no newline. |
| `files` | array of tables | yes | Every file the pack needs, at least one and at most 4,096. |
| `files[].name` | string | yes | The file's path inside the repository, with the same rules as `model.file`. Each name once, compared without case; a name cannot also be a folder of another. |
| `files[].sha256` | string | yes | The file's sha256, 64 lowercase hex characters. |
| `files[].size_bytes` | integer | yes | The file's exact size. Not negative. The disk-space check counts the sum. |

How the launcher runs a pack:

1. It downloads and verifies each file into the [model cache](../how-to/run/model-cache.md), as it does one file.
2. Before each `serve` or `run`, it runs `prepare` in the engine image, with `prepare[0]` as the entrypoint, no GPU, no network, and two more arguments: the pack's folder, `/models/repo/snapshots/<revision>`, and an empty output folder, `/models/prepared`. `/models/repo` is the repository's cache folder, mounted read-only, so that the snapshot's links into `blobs/` resolve.
3. The last line the command prints must be a path in `/models/prepared`. The server starts with `-m` set to it, the same cache folder at `/models/repo`, and the output folder read-only at `/models/prepared`.

The output folder is in the launcher's [state directory](files-and-environment.md), one per recipe, revision and image, and is emptied before each prepare step.

## [card]

| Key | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | The card id, such as `p100`. Lowercase letters, digits, `.`, `_` and `-`. Must equal the file name without `.toml`. |
| `title` | string | no | The card's human name, such as `NVIDIA Tesla P100 16 GB`. |
| `compute_capability` | string | yes | Must equal the `compute_cap` that `nvidia-smi` reports, such as `"6.0"`. |
| `min_memory_mib` | integer | yes | The GPU's `memory.total`, in MiB, must be at least this. Positive. |
| `min_driver` | string | yes | The oldest NVIDIA driver the recipe runs on, a dotted version such as `"525.60.13"`. |

A GPU matches a card when its compute capability equals `compute_capability` and its memory is at least `min_memory_mib`. A matching GPU with an older driver is refused, with the driver to install.

## [engine]

| Key | Type | Required | Meaning |
|---|---|---|---|
| `release` | string | yes | The engine release the recipe was measured on, such as `engine-v0.3.0`. |
| `arch` | string | yes | The GPU architecture of the image, such as `sm60`. |
| `image` | string | yes | The engine image, always by digest: `name@sha256:<64 hex>`. A tag is refused. |

## [run]

| Key | Type | Required | Meaning |
|---|---|---|---|
| `alias` | string | yes | The served model name, as `/v1/models` lists it. Letters, digits, `.`, `_`, `-` and `/`; the recipes use the repo id. Passed as `--alias`. |
| `context` | integer | yes | The context length, passed as `-c`. Positive. `--context` overrides it and makes the run untuned. |
| `threads_max` | integer | yes | The most CPU threads to use. `-t`, `-tb` and `OMP_NUM_THREADS` are the smaller of this and the host's CPU count. Positive. |
| `args` | array of strings | no | Every other `llama-server` argument, in order. |
| `env` | table of strings | no | The engine switches, exactly as measured. Each key must be a valid variable name. See [Engine switches](engine-switches.md). |

The launcher sets these itself, so `args` must not contain them: `-m`, `--model`, `--host`, `--port`, `-c`, `--ctx-size`, `-a`, `--alias`, `-t`, `--threads`, `-tb`, `--threads-batch`. `env` must not set `OMP_NUM_THREADS`.

## [receipt]

What was measured, and where. Every key is optional. `info` shows them, and `bench` compares its result with `decode_tps.short`.

| Key | Type | Meaning |
|---|---|---|
| `source` | string | Who measured the numbers, and on what build. |
| `summary` | string | One or two sentences on the result. |
| `decode_tps` | table of numbers | Decode speed in tokens per second. Keys: `short`, `ctx_32k`, `ctx_65k`. Not negative. |
| `prefill_tps` | table of numbers | Prefill speed in tokens per second, for example `ctx_4k`. Not negative. |
| `notes` | array of strings | The scope of each number, and anything else a reader needs. |

`info` shows measurement keys in the order `short`, `ctx_4k`, `ctx_32k`, `ctx_65k`, then any other keys alphabetically. The full measurements of a release are in its [receipts](receipts.md).

## Checks

The launcher refuses a recipe file that:

- has a key the schema does not have (`unknown keys`);
- names both a file and a pack;
- has a value that breaks a rule in the tables above;
- sits at a path that does not match its `model.hf_repo` and `card.id`.

## Draft, released and unpublished

| State | How it is set | `pull`, `run`, `serve` | `info` | `--dry-run` | `models` shows |
|---|---|---|---|---|---|
| Released | `status = "released"` | Allowed. | Shown. | Shown. | `released` |
| Draft | `status = "draft"` | Refused unless `--allow-draft`. | Shown. | Refused unless `--allow-draft`. | `draft` |
| Unpublished | An all-zero `revision`, `sha256` or image digest, or `size_bytes = 0` (in a pack, of any file) | Refused, even with `--allow-draft`. | Shown, with `NOT PUBLISHED`. | Shown. | `draft, unpublished` or `released, unpublished` |

A release moves a recipe to `released` only after the release gate has measured its image on its card. A pull request that changes `[run]` or `[engine]` needs a new receipt.
