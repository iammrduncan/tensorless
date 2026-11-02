---
title: "Recipe file"
description: "The recipe file, schema 1: every table and key, the checks the launcher applies, and what makes a recipe a draft or unpublished."
type: reference
sidebar:
  order: 2
---

A recipe is one model on one card. It names the exact engine image, the exact model file, the settings that were measured, and the numbers they gave. The launcher only runs what a recipe describes.

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
release = "engine-v0.1.0"
arch    = "sm60"
image   = "ghcr.io/iammrduncan/tensorless-engine@sha256:35409ff97c2f518e468790c28bd19206ae777a945fcffe5775585e28fd9939a7"

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
summary     = "Decode about 30 tok/s at short context, 1.14x the Q3 baseline model on the same card. Not yet measured at 32K or 65K."
decode_tps  = { short = 30.06 }
prefill_tps = { ctx_4k = 121.4 }
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
| `file` | string | yes | The weights file's path inside the repository: relative, with forward slashes, clean (no `..`), without `:` or newlines. |
| `sha256` | string | yes | The file's sha256, 64 lowercase hex characters. The launcher checks the download against it. |
| `size_bytes` | integer | yes | The file's exact size, for the disk-space check and progress. Must not be negative. `0` marks the recipe unpublished. |
| `license` | string | yes | The model's license. |

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
| `release` | string | yes | The engine release the recipe was measured on, such as `engine-v0.1.0`. |
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
- has a value that breaks a rule in the tables above;
- sits at a path that does not match its `model.hf_repo` and `card.id`.

## Draft, released and unpublished

| State | How it is set | `pull`, `run`, `serve` | `info` | `--dry-run` | `models` shows |
|---|---|---|---|---|---|
| Released | `status = "released"` | Allowed. | Shown. | Shown. | `released` |
| Draft | `status = "draft"` | Refused unless `--allow-draft`. | Shown. | Refused unless `--allow-draft`. | `draft` |
| Unpublished | An all-zero `revision`, `sha256` or image digest, or `size_bytes = 0` | Refused, even with `--allow-draft`. | Shown, with `NOT PUBLISHED`. | Shown. | `draft, unpublished` or `released, unpublished` |

A release moves a recipe to `released` only after the release gate has measured its image on its card. A pull request that changes `[run]` or `[engine]` needs a new receipt.
