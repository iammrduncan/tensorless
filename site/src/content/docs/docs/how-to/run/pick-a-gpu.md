---
title: "Pick a GPU on a host with several"
description: "Choose which GPU a model runs on with --gpu, run one model per card, and know what --card skips."
type: how-to
sidebar:
  order: 6
---

On a host with more than one GPU, the launcher picks the first one that matches the recipe. This page chooses the GPU yourself, and runs a different model on each card.

You need the launcher installed and a host that passes `tensorless doctor`. See [Prepare a Linux host for a P100](../setup/prepare-host.md).

## 1. Find the GPU's index

The launcher numbers GPUs the way `nvidia-smi` does. List them:

```sh
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

`tensorless doctor` lists them too, with the recipes each one fits.

## 2. Run the model on that GPU

Pass the index with `--gpu`:

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF --gpu 1
```

The launcher checks that GPU 1 exists and matches the recipe (its compute capability, its memory and the driver), and the container then gets only that GPU: `--device nvidia.com/gpu=1` with podman, `--gpus device=1` with docker. If there is no GPU 1, it says so and lists the ones `nvidia-smi` sees. If GPU 1 does not fit the recipe, it says what the recipe needs.

## How the launcher chooses without --gpu

- It takes the first GPU, by index, that matches the recipe.
- If every GPU on the host matches, the container gets all of them. The recipes run `llama-server` with `-sm none`, which keeps the model on one card.
- If some GPUs don't match, the container gets only the matched one, so `llama-server` cannot land on a card the engine image was not built for.

## Run one model per card

Each model gets its own container, named after its repo id, so two different models can run side by side on two cards. Give each its own GPU and port:

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF --detach --gpu 0
tensorless run iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF --detach --gpu 1 --port 8081
```

The same model cannot run twice: its container name is taken, and the second `run` says it `is already running`.

## Skip detection with --card

`--card p100` chooses the P100 recipe without asking `nvidia-smi`. It is meant for showing a recipe (`info`, `--dry-run`) on a machine without the card. When you start a server with it:

- the launcher checks nothing about the GPU: not its compute capability, its memory or the driver version;
- the container gets every GPU, unless you also pass `--gpu`.

So on a real P100 host, leave `--card` out and let the launcher check the card.

## Check that it worked

While the model runs, `nvidia-smi` shows its memory in use on the GPU you chose (about 14,000 MiB for Qwen3.8-27B-PN3), and the other cards stay free for other work:

```sh
nvidia-smi --query-gpu=index,memory.used --format=csv
```
