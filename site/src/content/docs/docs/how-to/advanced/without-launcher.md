---
title: "Run the engine image without the launcher"
description: "Pull the engine image, download and check the model yourself, and start llama-server with podman or docker exactly as a recipe does."
type: how-to
sidebar:
  order: 1
---

The launcher only runs a container. You can run the same container yourself, for example from your own scripts, a compose file or a service manager. This guide starts Qwen3.8-27B-PN3 exactly as its recipe does.

## Before you start

You need:

- a Linux x86_64 host (the engine image is linux/amd64) with a Tesla P100 16 GB that no other process is using;
- an x86-64 CPU with AVX2, FMA and F16C;
- an NVIDIA driver from R525 to R580;
- podman with an NVIDIA CDI spec, or docker with the NVIDIA container toolkit (see [Prepare a Linux host for a P100](../setup/prepare-host.md));
- about 13 GB of free disk space for the model, and a few GB for the image;
- `curl` and `sha256sum`.

:::tip
`tensorless info pn3 --card p100` prints the exact container command for your machine, on any computer, with no GPU: `tensorless info` works on macOS too. Use it to check the command below, or to get the one for another recipe.
:::

## 1. Pull the engine image by digest

```sh
podman pull ghcr.io/iammrduncan/tensorless-engine@sha256:b95f77cd01cf0f785ee8cc853fc09454bdf4851f7214e8253c6529759c1eb7a7
# or: docker pull ghcr.io/iammrduncan/tensorless-engine@sha256:b95f77cd01cf0f785ee8cc853fc09454bdf4851f7214e8253c6529759c1eb7a7
```

Pull by digest, not by tag: the digest is what the receipt measured.

## 2. Download the model and check it

Download the file at the revision the recipe pins, then check its sha256:

```sh
mkdir -p ~/models && cd ~/models
curl -fL -C - -o Qwen3.8-27B-PN3-P0k19.gguf \
  https://huggingface.co/iamMrDuncan/Qwen3.8-27B-PN3-GGUF/resolve/c701b6690ec360b16d40f85df1cb76062e30cd38/Qwen3.8-27B-PN3-P0k19.gguf
echo "15ce2d4469673f1f1753c55de3780cfe04311dd6f05cb41633d73df128e91476  Qwen3.8-27B-PN3-P0k19.gguf" | sha256sum -c
```

`sha256sum` prints `Qwen3.8-27B-PN3-P0k19.gguf: OK`. If the download stops, run the `curl` line again: `-C -` resumes it. The values for each model are in [Launch targets](../../reference/launch-targets.md).

## 3. Start the server

With podman:

```sh
podman run -d --rm --name tensorless-pn3 \
  --device nvidia.com/gpu=all --security-opt label=disable \
  -p 127.0.0.1:8080:8080 \
  -v ~/models/Qwen3.8-27B-PN3-P0k19.gguf:/models/Qwen3.8-27B-PN3-P0k19.gguf:ro \
  -e OMP_NUM_THREADS=8 \
  -e GGML_CUDA_FATTN_SM60=1 -e GGML_CUDA_FATTN_SM60_CTAS=4 -e GGML_CUDA_FATTN_TILE_OCC1=1 \
  -e GGML_CUDA_FUSE_PRE_ADD=1 -e GGML_CUDA_MMV_SM60=1 \
  -e GGML_CUDA_MMV_SM60_INPLACE_IQ3S=1 -e GGML_CUDA_MMV_SM60_INPLACE_XXS=1 \
  -e GGML_CUDA_MMV_SM60_IQ3S_STOCK=1 -e GGML_CUDA_MMV_SM60_Q3K=1 \
  -e GGML_CUDA_MMV_SM60_Q4K_DHEAD=1 -e GGML_CUDA_MMV_SM60_Q4K_HEAD=1 \
  -e GGML_CUDA_MMV_SM60_Q6K=1 -e GGML_CUDA_MMV_SM60_XXS_KB_WIDE=1 \
  -e GGML_CUDA_MMV_SM60_XXS_STOCK=1 -e GGML_CUDA_NORM_EMIT_Q8=1 \
  -e GGML_CUDA_PREFILL_CHUNK_GEMM=1 -e GGML_CUDA_PREFILL_CHUNK_STREAMS=3 \
  -e GGML_CUDA_PREFILL_DEQ_F4=1 -e GGML_CUDA_PREFILL_REP_DIRECT=1 \
  -e GGML_CUDA_SM86_GDN_COLS=8 -e GGML_CUDA_SM86_IQ3_SMEM_GRID=1 \
  ghcr.io/iammrduncan/tensorless-engine@sha256:b95f77cd01cf0f785ee8cc853fc09454bdf4851f7214e8253c6529759c1eb7a7 \
  -m /models/Qwen3.8-27B-PN3-P0k19.gguf --host 0.0.0.0 --port 8080 -c 65536 \
  --alias iamMrDuncan/Qwen3.8-27B-PN3-GGUF -t 8 -tb 8 \
  -np 1 -ngl 99 -ot token_embd.weight=CUDA0 -ctk q4_0 -ctv q4_0 -sm none -lm none -fit off \
  --threads-http 4 -b 512 -ub 512 -fa on --jinja
```

With docker, replace the second line with `--gpus all`, as the launcher does.

What the parts are:

| Part | From the recipe |
|---|---|
| `-e GGML_CUDA_...` | `[run.env]`: the 21 [engine switches](../../reference/engine-switches.md), exactly as measured. |
| `-c 65536` | `run.context`. |
| `--alias` | `run.alias`, the served model name. |
| `-t 8 -tb 8`, `OMP_NUM_THREADS=8` | `run.threads_max`. The launcher uses the smaller of this and your CPU count. |
| Everything after `-tb 8` | `run.args`, in order. |

The image's entrypoint is `llama-server`, so the arguments after the image name go to it. To give the container only one GPU, use `--device nvidia.com/gpu=N` (podman) or `--gpus device=N` (docker).

## 4. Check that it works

The model takes a little while to load. Follow the logs until the server listens:

```sh
podman logs -f tensorless-pn3
```

Then, from another terminal:

```sh
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models
```

`/health` answers `{"status":"ok"}` once the model is loaded, and `/v1/models` lists `iamMrDuncan/Qwen3.8-27B-PN3-GGUF`. To measure it the way the receipts did, run `tensorless bench pn3 --card p100`: `bench` needs only the server.

## 5. Stop it

```sh
podman stop tensorless-pn3
```

The container was started with `--rm`, so stopping it also removes it.

## If it fails

- `no kernel image is available` in the logs: the image does not match the card. The `sm60` image runs only on compute capability 6.0.
- An out-of-memory error: another process is using the GPU. The model and its context take about 14 GB of the card's 16 GB.
- An error about the GPU device: podman has no CDI spec, or docker has no NVIDIA support. `tensorless doctor` checks both and prints the fix.
