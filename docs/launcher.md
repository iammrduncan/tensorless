# The tensorless launcher

`tensorless` runs a model that was tuned for your GPU, on the engine it was tuned with, in one command:

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF
```

It detects the card, picks the model's recipe for that card, pulls the engine image by digest, downloads the model at its pinned revision into the Hugging Face cache and checks its sha256, then starts an OpenAI-compatible `llama-server` in a podman or docker container. The launcher holds no inference code. How the launcher, recipes and images fit together is in [design.md](design.md).

**Status:** v0.2.0. Both built-in recipes are released: `tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF` and `tensorless run iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF` need no extra flags. A draft recipe needs `--allow-draft`, and a recipe that still has placeholder values is refused even with it; `tensorless models` marks those "unpublished".

## Install

Build it from source with Go 1.23 or newer:

```sh
go install github.com/iammrduncan/tensorless/cmd/tensorless@latest
```

Or from a checkout, with the version stamped in:

```sh
go build -ldflags "-X main.version=$(git describe --tags --always) -X main.commit=$(git rev-parse --short HEAD)" ./cmd/tensorless
```

The binary is static and has no runtime dependencies. To serve a model, the host also needs:

- an NVIDIA driver at least as new as the recipe's `min_driver` (the P100 recipes need 525.60.13; R580 is the last branch with Pascal support);
- podman, or docker;
- NVIDIA container support: for podman, a CDI spec from the NVIDIA Container Toolkit; for docker, the toolkit's nvidia runtime or a CDI spec.

`tensorless doctor` checks all of this and prints the fix for anything missing.

## The MODEL argument

MODEL is the model's Hugging Face repo id, for example `iamMrDuncan/Qwen3.8-27B-PN3-GGUF`.

- **Case does not matter**, as on the Hub. The cache directory always keeps the recipe's spelling: `models--iamMrDuncan--Qwen3.8-27B-PN3-GGUF`.
- **`ORG/REPO@REVISION`** picks another commit (40 hex characters) of the repository. That run differs from what the recipe measured, so it needs `--allow-draft` and is reported as untuned. The downloaded file must still match the recipe's sha256.
- **A recipe can also define a short name**, such as `pn3`, which works anywhere a repo id does.

`tensorless models` lists the repo ids it knows.

## Commands

### run

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF --detach --port 9000
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF -- --temp 0.7
```

Runs `pull`, then `serve`. Arguments after `--` go to `llama-server` after the recipe's own arguments.

### pull

```sh
tensorless pull iamMrDuncan/Qwen3.8-27B-PN3-GGUF
```

Pulls the engine image by digest (skipped if `podman image inspect` finds it). Downloads the model into the Hugging Face cache, verifies its sha256 and links it into the snapshot.

- **Resuming:** an interrupted download resumes where it stopped. Run the same command again.
- **Disk space:** it checks the free space first.
- **Already cached:** a file that is already in the cache is hashed once. After that, a marker in the state directory skips the hash while the file's size and modification time stay the same.

### serve

```sh
tensorless serve iamMrDuncan/Qwen3.8-27B-PN3-GGUF
tensorless serve iamMrDuncan/Qwen3.8-27B-PN3-GGUF --detach
```

Starts the container `tensorless-iammrduncan-qwen3.8-27b-pn3-gguf` from an image and model that `pull` already fetched and verified.

- **Waiting:** it polls `http://127.0.0.1:8080/health` every second, for up to 20 minutes, until the model has loaded.
- **When it is ready:** it prints the endpoint and a `curl` example. The served model name, in `/v1/models`, is the repo id.
- **In the foreground:** it follows the container's logs. Ctrl-C or SIGTERM stops the container.
- **With `--detach`:** it returns once the server is healthy.
- **If the container exits first:** it prints the last 60 log lines and fails.
- **If it is already running:** it says so and points to `tensorless stop`.

The server binds to `127.0.0.1` unless you pass `--host`.

A run with `--context` set to another value, extra arguments after `--`, or another `@REVISION` is **untuned**: the launcher says so before it starts, and the receipt's numbers do not apply to it.

The container runs as follows (podman shown; docker uses `--gpus all` or `--gpus device=N` and no `--security-opt`). `T` is the smaller of the host's CPU count and the recipe's `threads_max`.

```sh
podman run -d --rm --name tensorless-iammrduncan-qwen3.8-27b-pn3-gguf \
    --device nvidia.com/gpu=all --security-opt label=disable \
    -p 127.0.0.1:8080:8080 -v BLOB:/models/FILE.gguf:ro \
    -e OMP_NUM_THREADS=T -e GGML_CUDA_...=1 ... IMAGE@sha256:... \
    -m /models/FILE.gguf --host 0.0.0.0 --port 8080 -c CONTEXT \
    --alias iamMrDuncan/Qwen3.8-27B-PN3-GGUF -t T -tb T <recipe args> <extra args>
```

When the host has GPUs that do not match the recipe, the container gets only the matched one (`nvidia.com/gpu=N`), so `llama-server` cannot land on a card the image was not built for. `tensorless info` and `--dry-run` show the exact command.

### stop

```sh
tensorless stop iamMrDuncan/Qwen3.8-27B-PN3-GGUF
```

Stops the model's container. With `--runtime auto`, it looks in both podman and docker.

### bench

```sh
tensorless bench iamMrDuncan/Qwen3.8-27B-PN3-GGUF
tensorless bench iamMrDuncan/Qwen3.8-27B-PN3-GGUF --runs 5 --tokens 256 --json
```

Measures decode speed against a server that is already running on `--host`/`--port`. Start one with `tensorless serve MODEL --detach`. It mirrors the hill-climb judge:

1. **Warm-up:** one 8-token request.
2. **Measured runs:** `--runs` greedy `/completion` requests (temperature 0, top_k 1, seed 424242, `ignore_eos`, no prompt cache), each generating `--tokens` tokens.
3. **Timing:** each streamed event is stamped. Decode speed is (tokens − 1) / (last event − first event). TTFT is the first event minus the request start.

The prompt is built in: the first 2,000 characters of the Apache-2.0 license text, followed by "Continue the text above." It reports the median decode speed, TTFT and prompt tokens, and compares them with the recipe's receipt (`decode_tps.short`). The receipt was measured with a different prompt, so expect a few percent of difference.

### models

```sh
tensorless models
tensorless models --json
```

Lists every recipe: model, card, status (draft or released, and "unpublished" while it has placeholders), engine release, whether it fits the detected GPU (`yes`, `no`, or `unknown` without a GPU) and whether the model file is in the cache (`yes`, `partial` or `no`).

### info

```sh
tensorless info iamMrDuncan/Qwen3.8-27B-PN3-GGUF
tensorless info pn3 --card p100 --json
```

Shows a recipe exactly as it would run, with no GPU needed:

- the model's repository, revision, file, sha256, size, license and download URL;
- where it goes in the cache;
- the card requirements and the engine image;
- the run settings, with the full container command;
- the receipt.

### doctor

```sh
tensorless doctor
```

Checks the host and prints PASS, WARN or FAIL for each item, with the command that fixes it. It exits 1 if anything fails. See [Troubleshooting](#troubleshooting).

### capabilities

```sh
tensorless capabilities --json
```

Prints the version, commit, platform, recipe schema, supported runtimes, the known cards and the known recipes (model, card, status, engine release, arch). It runs no external command and needs no GPU, driver, runtime or network.

### version

```sh
tensorless version
```

## Flags

The commands that take a MODEL accept these flags before or after it.

| Flag | Default | Meaning |
|---|---|---|
| `--recipes DIR` | built-in | Read recipes from `DIR/<org>/<repo>/<card>.toml` instead of the embedded ones. |
| `--card ID` | detected | Use this card's recipe, for example `p100`, and skip GPU detection, including its memory and driver checks. |
| `--gpu N` | first match | Use GPU index `N`, as nvidia-smi numbers them. |
| `--runtime` | `auto` | `auto`, `podman` or `docker`. `auto` prefers podman. |
| `--cache DIR` | see below | The Hugging Face cache directory. |
| `--allow-draft` | off | Allow a draft recipe, or a `MODEL@REVISION` other than the recipe's. It never allows a recipe whose fields are still placeholders. |
| `--dry-run` | off | Print every command and download URL instead of running them. It still reads the GPU list with nvidia-smi. |
| `--host` | `127.0.0.1` | The host address the server is published on. |
| `--port` | `8080` | The host port. |
| `--context N` | recipe | The context length. Any other value than the recipe's makes the run untuned. |
| `--detach` | off | `run` and `serve`: return once the server is healthy. |
| `--json` | off | `models`, `info`, `bench`, `capabilities`: print JSON. |
| `--runs N`, `--tokens N` | 3, 256 | `bench` only. |

Two kinds of recipe are refused:

- **A draft** (status `draft`) is refused unless you pass `--allow-draft`.
- **A recipe that is not published yet**, whose revision or image digest is all zeros or whose `size_bytes` is 0, is refused for `pull`, `run` and `serve` even with `--allow-draft`. `info` and `--dry-run` still show it.

## Where files go

| What | Where |
|---|---|
| Model files | The Hugging Face cache: `--cache`, else `$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`. The file is at `models--ORG--REPO/blobs/<sha256>`, with the symlink `models--ORG--REPO/snapshots/<revision>/<file>` pointing to it, so other Hugging Face tools see it. A partial download is `<blob>.incomplete`. |
| Verification markers | `$XDG_STATE_HOME/tensorless/verified/<sha256>.json`, else `~/.local/state/tensorless/verified/`. |
| Engine images | The container runtime's own storage. |
| Containers | `tensorless-<repo id, lowercased, with / replaced by ->`, started with `--rm`. |

Environment variables:

- `HF_ENDPOINT` changes the Hub address (default `https://huggingface.co`).
- `HF_TOKEN` is sent as `Authorization: Bearer` for private or gated repositories. It is never printed.

The launcher sends no telemetry. It uses the network only to download from the Hub; the container runtime pulls the image from the registry.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. |
| 1 | The command failed, or `doctor` found a failure. |
| 2 | Bad usage. |
| 130 | Interrupted. |

## Troubleshooting

`tensorless doctor` checks these, in order.

| Check | If it fails |
|---|---|
| nvidia-smi is present and lists the GPUs | Install the NVIDIA driver, at least the recipes' `min_driver`, and reboot. If nvidia-smi is installed but fails, the driver is not loaded: reboot, then run `nvidia-smi`. |
| A GPU matches a recipe (compute capability and memory) | There is no recipe for this card yet. A 12 GB P100 does not fit the 16 GB recipes. |
| The driver version is at least the recipe's `min_driver` | Install a newer driver. For Pascal cards, R580 is the last branch with support. |
| podman or docker is installed | `sudo apt install podman` or `sudo dnf install podman` (or install docker). |
| podman: a CDI spec at `/etc/cdi/nvidia.yaml` or `/var/run/cdi/nvidia.yaml` | Install nvidia-container-toolkit, then `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`. Generate it again after each driver upgrade. |
| docker: the nvidia runtime in `docker info`, a CDI spec, or the toolkit's hook | Install nvidia-container-toolkit, then `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`. |
| Free disk space in the model cache | Free some space, or pass `--cache DIR` (or set `HF_HUB_CACHE`) on a larger disk. |

Other messages, and what to do:

| Message | What to do |
|---|---|
| "is not published yet" | The recipe's model upload or engine image does not exist yet. Wait for a release. |
| "is a draft" | The recipe has not passed the release gate. Pass `--allow-draft` to try it anyway. |
| "is already running" | The model's container already exists. Run `tensorless stop MODEL` first. |
| "cannot use port" | Something else listens there. Pass `--port`. |
| "sha256 mismatch" | The download was deleted. Run the command again. If it repeats, the upload or the recipe is wrong; please report it. |
| "the server exited before it was ready" | Read the log lines above it. `no kernel image is available` means the image does not match the card. An out-of-memory error means another process is using the GPU. |
| "HTTP 401" or "HTTP 403" from the Hub | The repository is private or gated. Accept its terms on the Hub and set `HF_TOKEN`. |
