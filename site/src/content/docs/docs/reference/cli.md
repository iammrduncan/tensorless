---
title: "Command line"
description: "Every tensorless command, its flags and defaults, the MODEL argument, untuned runs and exit codes."
type: reference
sidebar:
  order: 1
---

`tensorless` is one static binary. It runs a model's [recipe](recipes.md) on your GPU: it pulls the engine image by digest, downloads the model into the Hugging Face cache, checks its sha256, and starts `llama-server` in podman or docker. It holds no inference code.

```text
tensorless COMMAND [MODEL] [flags] [-- llama-server args]
```

Run `tensorless help` for the list of commands, and `tensorless COMMAND -h` for a command's flags.

## Commands

| Command | Usage | Summary |
|---|---|---|
| [`run`](#run) | `tensorless run MODEL [flags] [-- llama-server args]` | Pull, then serve. |
| [`pull`](#pull) | `tensorless pull MODEL [flags]` | Pull the engine image by digest and download the model into the Hugging Face cache. |
| [`serve`](#serve) | `tensorless serve MODEL [flags] [-- llama-server args]` | Start the server, wait until it is healthy, and print the endpoint. |
| [`stop`](#stop) | `tensorless stop MODEL [flags]` | Stop the model's server. |
| [`bench`](#bench) | `tensorless bench MODEL [flags]` | Measure decode speed against the running server and compare it with the receipt. |
| [`models`](#models) | `tensorless models [flags]` | List the recipes, whether they fit this GPU, and whether the model is cached. |
| [`info`](#info) | `tensorless info MODEL [flags]` | Show a recipe exactly as it would run. Needs no GPU. |
| [`doctor`](#doctor) | `tensorless doctor [flags]` | Check the GPU, driver, container runtime, NVIDIA container support and disk space. |
| [`capabilities`](#capabilities) | `tensorless capabilities [flags]` | Describe this launcher: version, recipes, runtimes. Needs no GPU, runtime or network. |
| [`version`](#version) | `tensorless version` | Print the version. |
| [`help`](#help) | `tensorless help` | Print the list of commands. |

## The MODEL argument

`run`, `pull`, `serve`, `stop`, `bench` and `info` take exactly one MODEL. `models`, `doctor`, `capabilities` and `version` take none.

| Form | Example | Meaning |
|---|---|---|
| Repo id | `iamMrDuncan/Qwen3.8-27B-PN3-GGUF` | The model's Hugging Face repository. Case does not matter, as on the Hub. |
| Short name | `pn3` | A recipe's optional `model.name`. It works anywhere a repo id does. |
| Repo id with a revision | `iamMrDuncan/Qwen3.8-27B-PN3-GGUF@<40-hex commit>` | Another commit of the repository. The revision must be a 40-character commit, not a branch or a tag. A revision other than the recipe's needs `--allow-draft`, makes the run [untuned](#untuned-runs), and the file must still match the recipe's sha256. |

When a model has recipes for several cards, the launcher picks the one whose card matches a GPU that `nvidia-smi` reports, or the one `--card` names. [Launch targets](launch-targets.md) lists the models of this release, and `tensorless models` lists the ones your launcher knows.

## Flags

Every command except `version` accepts the common flags. Flags may come before or after MODEL. A flag is written with one dash or two (`-port` or `--port`), and its value after a space or `=`.

### Common flags

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--recipes DIR` | path | built-in recipes | Read recipes from `DIR/<org>/<repo>/<card>.toml` instead of the ones built into the binary. |
| `--card ID` | string | detected | Use this card's recipe, for example `p100`, and skip GPU detection, with its memory and driver checks. |
| `--gpu N` | integer ≥ 0 | the first GPU that matches the recipe | Use GPU index `N`, as `nvidia-smi` numbers them. The container gets only that GPU. |
| `--runtime NAME` | `auto`, `podman` or `docker` | `auto` | The container runtime. `auto` prefers podman, then docker. |
| `--cache DIR` | path | `$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub` | The Hugging Face cache directory for model files. |
| `--allow-draft` | switch | off | Allow a draft recipe, or a `MODEL@REVISION` other than the recipe's. The run is then untuned. It never allows a recipe that is [not published](recipes.md#draft-released-and-unpublished). |
| `--dry-run` | switch | off | Print every command and download URL instead of running it. It still reads the GPU list with `nvidia-smi`, unless `--card` is given. |
| `--host ADDR` | string | `127.0.0.1` | The host address the server is published on. Must not be empty. |
| `--port N` | integer 1–65535 | `8080` | The host port the server is published on. |
| `--context N` | integer > 0 | the recipe's `run.context` | The context length. Any other value than the recipe's makes the run untuned. |

### Flags of some commands

| Flag | Commands | Default | Meaning |
|---|---|---|---|
| `--detach` | `run`, `serve` | off | Return once the server is healthy instead of following its logs. |
| `--json` | `bench`, `models`, `info`, `capabilities` | off | Print JSON. |
| `--runs N` | `bench` | `3` | The number of measured runs. At least 1. |
| `--tokens N` | `bench` | `256` | Tokens to generate per run. At least 2. |

### Extra llama-server arguments

`run`, `serve` and `info` accept arguments after `--`. They are passed to `llama-server` after the recipe's own arguments, and they make the run untuned. Other commands refuse them.

The launcher sets these `llama-server` flags itself, so they are refused after `--` (and in a recipe's `run.args`): `-m`, `--model`, `--host`, `--port`, `-c`, `--ctx-size`, `-a`, `--alias`, `-t`, `--threads`, `-tb`, `--threads-batch`. Use `--host`, `--port` and `--context` instead.

## Untuned runs

A run is untuned when the command line changes the configuration a recipe measured:

- `--context` with another value than the recipe's;
- extra arguments after `--`;
- a `MODEL@REVISION` other than the recipe's.

`run` and `serve` print `UNTUNED:` and the reason before they start, and say that the receipt's numbers do not apply. `info` shows the reason in an `UNTUNED` line, and `--dry-run` in its header. `--host`, `--port`, `--gpu`, `--card`, `--cache` and `--runtime` do not make a run untuned.

## Refused recipes

| Recipe | `pull`, `run`, `serve` | `info` | `--dry-run` |
|---|---|---|---|
| Draft (`status = "draft"`) | Refused unless `--allow-draft`. | Shown. | Refused unless `--allow-draft`. |
| Not published: an all-zero revision, sha256 or image digest, or `size_bytes = 0` | Refused, even with `--allow-draft`. | Shown, marked `NOT PUBLISHED`. | Shown. |

## run

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF --detach --port 9000
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF -- --temp 0.7
```

Runs [`pull`](#pull), then [`serve`](#serve), in one command. Before it pulls, it checks that the port is free and that the model's container is not already running.

## pull

```sh
tensorless pull iamMrDuncan/Qwen3.8-27B-PN3-GGUF
```

1. Pulls the engine image by digest, unless `podman image inspect` (or `docker image inspect`) already finds it.
2. Downloads the model file at the recipe's revision into the Hugging Face cache, as `<cache>/models--<org>--<repo>/blobs/<sha256>`, and checks its sha256.
3. Links it into the snapshot, `<cache>/models--<org>--<repo>/snapshots/<revision>/<file>`, so other Hugging Face tools find it.

| Behavior | Detail |
|---|---|
| Resume | An interrupted download is kept as `<blob>.incomplete`. The next `pull` hashes what is there and resumes with an HTTP range request. |
| Disk space | Checked before the download. Too little space fails with the space needed and the space free. |
| Stall | A download that receives nothing for 2 minutes fails; run the command again to resume. |
| Already cached | A cached file is hashed once. A marker in the state directory then skips the hash while the file's size and modification time stay the same. |
| sha256 mismatch | The download is deleted and the command fails. |

## serve

```sh
tensorless serve iamMrDuncan/Qwen3.8-27B-PN3-GGUF
tensorless serve iamMrDuncan/Qwen3.8-27B-PN3-GGUF --detach
```

Starts the model's container from an image and model that `pull` already fetched and verified. It fails if the image is not pulled or the model is not downloaded, and names the `pull` command to run.

| Behavior | Detail |
|---|---|
| Container name | `tensorless-` and the repo id, lowercased, with every character other than `a`–`z`, `0`–`9`, `_`, `.` and `-` replaced by `-`. For example `tensorless-iammrduncan-qwen3.8-27b-pn3-gguf`. |
| Already running | Fails with `is already running` and points to `tensorless stop`. A stopped container of the same name is removed first. |
| Port in use | Fails with `cannot use port` before anything starts. |
| Waiting | Polls `http://<host>:<port>/health` every second, for up to 20 minutes, while the model loads. |
| Ready | Prints the endpoint (the OpenAI-compatible API is under `/v1`) and a `curl` example. The served model name is the recipe's `run.alias`, the repo id. |
| Foreground | Follows the container's logs. Ctrl-C or SIGTERM stops the container. |
| `--detach` | Returns once the server is healthy, and prints the `tensorless stop` command. |
| Exits early | If the container exits before it is healthy, prints its last log lines (up to 60) and fails with `the server exited before it was ready`. |
| Timeout | If the server is not healthy after 20 minutes, stops it and fails. |

The container command (podman shown):

```text
podman run -d --rm --name <container name> \
    --device nvidia.com/gpu=<all or N> --security-opt label=disable \
    -p <host>:<port>:8080 -v <blob>:/models/<file>:ro \
    -e OMP_NUM_THREADS=<T> -e <each run.env variable, sorted> \
    <engine.image> \
    -m /models/<file> --host 0.0.0.0 --port 8080 -c <context> \
    --alias <run.alias> -t <T> -tb <T> <run.args> <extra args>
```

- With docker, the GPU is `--gpus all` or `--gpus device=N`, and there is no `--security-opt`.
- `T` is the smaller of the host's CPU count and the recipe's `run.threads_max`.
- The GPU is `all` when every GPU on the host matches the recipe's card. When some do not, the container gets only the matched GPU, so `llama-server` cannot land on a card the image was not built for. `--gpu N` always gives only GPU `N`.
- `--host` with an IPv6 address is published as `[ADDR]:<port>:8080`.

## stop

```sh
tensorless stop iamMrDuncan/Qwen3.8-27B-PN3-GGUF
```

Stops the model's container. With `--runtime auto`, it looks in every installed runtime, podman and docker. It prints `stopped <name>`, or `<name> is not running`. With `--dry-run`, it prints the stop commands.

## bench

```sh
tensorless bench iamMrDuncan/Qwen3.8-27B-PN3-GGUF
tensorless bench iamMrDuncan/Qwen3.8-27B-PN3-GGUF --runs 5 --tokens 256 --json
```

Measures decode speed against a server that is already running on `--host` and `--port`. It fails if nothing answers `/health` there. It warns if the server's `/v1/models` does not list the recipe's alias.

| Step | Detail |
|---|---|
| Prompt | The first 2,000 characters of the Apache-2.0 license text, then a blank line and `Continue the text above.` |
| Warm-up | One request for 8 tokens, not timed. |
| Measured runs | `--runs` streamed `POST /completion` requests, each for `--tokens` tokens: temperature 0, `top_k` 1, seed 424242, `ignore_eos`, no prompt cache. Each request uses a new connection. |
| Decode speed | (tokens − 1) / (time of the last streamed token − time of the first). |
| TTFT | The time of the first streamed token minus the start of the request. |
| Result | The median decode speed and TTFT of the runs, the prompt tokens, and the comparison with the recipe's `receipt.decode_tps.short`. |

The JSON output has `model`, `card`, `endpoint`, `runs` (each with `decode_tps`, `ttft_ms`, `prompt_tokens` and `tokens`), `median_decode_tps`, `median_ttft_ms`, `prompt_tokens`, and, when the recipe has a short-context receipt, `receipt_decode_tps` and `vs_receipt_percent`.

## models

```sh
tensorless models
tensorless models --json
```

Lists every recipe the launcher knows, one row each:

| Column | Values |
|---|---|
| `MODEL` | The repo id. |
| `CARD` | The card id. |
| `STATUS` | `draft` or `released`, with `, unpublished` while a pinned value is a placeholder. |
| `ENGINE` | The engine release. |
| `FITS` | `yes`, `no`, or `unknown` when no GPU can be read. `yes` needs a matching GPU and a new enough driver. |
| `CACHED` | `yes` (the file is there at its full size), `partial` (an interrupted download) or `no`. |

The JSON rows have `model`, `name`, `card`, `title`, `status`, `published`, `release`, `arch`, `fits` and `cached`.

## info

```sh
tensorless info iamMrDuncan/Qwen3.8-27B-PN3-GGUF
tensorless info pn3 --card p100 --json
```

Shows a recipe exactly as it would run. It needs no GPU, downloads nothing and starts nothing. It shows:

- the model's repository, revision, file, sha256, size, license and download URL, and its place in the cache;
- the card requirements and the engine image;
- the run settings, the thread count, and the full container command with the endpoint;
- whether the run is untuned, and why;
- the receipt: summary, source, decode and prefill numbers, and notes.

`--context` and arguments after `--` show in the command and the `UNTUNED` line. The JSON output has `recipe`, `published`, `unpublished`, `fits`, `untuned`, `download` (`url`, `blob`, `snapshot`, `size_bytes`, `cached`), `runtime`, `gpu`, `threads`, `command` and `endpoint`.

## doctor

```sh
tensorless doctor
```

Checks the host. Each line starts with `PASS`, `WARN` or `FAIL`; a failed check is followed by a `fix:` line with the command or step that fixes it. It ends with `all checks passed.` or `N checks failed.`, and exits 1 if any check failed.

| Check | Passes when |
|---|---|
| NVIDIA driver | `nvidia-smi` is installed and lists the GPUs. |
| GPU | A GPU matches a recipe's compute capability and memory. A GPU that fits no recipe is a `WARN`; no fitting GPU at all is a `FAIL`. |
| Driver version | The driver is at least the highest `min_driver` of the recipes that fit. |
| Container runtime | podman or docker is installed (`--runtime` picks one). |
| NVIDIA container support | podman: a CDI spec at `/etc/cdi/nvidia.yaml` or `/var/run/cdi/nvidia.yaml`. docker: the nvidia runtime in `docker info`, a CDI spec, or the NVIDIA container toolkit's hook. |
| Disk space | The model cache has room for the largest model that is not cached yet, among the recipes that fit (or all recipes, when no GPU can be read). |

## capabilities

```sh
tensorless capabilities --json
```

Describes the launcher. It runs no external command and needs no GPU, driver, runtime or network.

| JSON field | Meaning |
|---|---|
| `name` | `tensorless` |
| `version`, `commit` | The launcher's version and the commit it was built from. |
| `platform` | Operating system and architecture, such as `linux/amd64`. |
| `recipe_schema` | The recipe schema this launcher reads: `1`. |
| `runtimes` | `podman`, `docker`. |
| `cards` | Each known card: `id`, `title`, `compute_capability`, `min_memory_mib`, `min_driver`, `arch`. |
| `recipes` | Each recipe: `model`, `name`, `card`, `status`, `published`, `release`, `arch`. |

## version

```sh
tensorless version
```

Prints the version, the commit and the platform, for example `tensorless 0.2.0 (commit bc8d564127a625363917c84491786dd41f91d1e0), linux/amd64`. `tensorless --version` does the same. It takes no flags.

## help

`tensorless help`, `tensorless -h` and `tensorless --help` print the list of commands. `tensorless` with no command prints it too, and exits 2.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success, including `help` and `-h`. |
| `1` | The command failed, or `doctor` found a failure. |
| `2` | Bad usage: an unknown command or flag, a missing or extra MODEL, a refused value. |
| `130` | Interrupted by Ctrl-C or SIGTERM. |

## Environment

The launcher reads `HF_HUB_CACHE`, `HF_HOME`, `HF_ENDPOINT`, `HF_TOKEN` and `XDG_STATE_HOME`. See [Files and environment variables](files-and-environment.md).
