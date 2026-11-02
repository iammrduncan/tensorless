---
title: "See exactly what will run"
description: "Show a recipe, its download and its full container command with tensorless info and --dry-run, before anything is downloaded or started."
type: how-to
sidebar:
  order: 7
---

Before you download 13 GB or start a server, you can see exactly what the launcher would do: which file it downloads from where, which image it pulls by digest, and the full `podman run` or `docker run` command. `tensorless info` shows one recipe; `--dry-run` shows the plan of a command.

You need the launcher installed. See [Install the launcher](../setup/install-launcher.md). Neither needs a GPU, so you can do this on a laptop.

## Show a recipe

```sh
tensorless info iamMrDuncan/Qwen3.8-27B-PN3-GGUF
```

```text
iamMrDuncan/Qwen3.8-27B-PN3-GGUF on p100: Qwen3.8-27B-PN3 (P100-native 3-bit, mix P0k+19)
  short name      pn3
  status          released
  recipe          recipes/iamMrDuncan/Qwen3.8-27B-PN3-GGUF/p100.toml
  fits this host  unknown (nvidia-smi not found)

Model
  repository  iamMrDuncan/Qwen3.8-27B-PN3-GGUF
  revision    c701b6690ec360b16d40f85df1cb76062e30cd38
  file        Qwen3.8-27B-PN3-P0k19.gguf
  sha256      15ce2d4469673f1f1753c55de3780cfe04311dd6f05cb41633d73df128e91476
  size        12.95 GB (12954114496 bytes)
  license     Apache-2.0 (from Qwen/Qwen3.8-27B)
  url         https://huggingface.co/iamMrDuncan/Qwen3.8-27B-PN3-GGUF/resolve/c701b6690ec360b16d40f85df1cb76062e30cd38/Qwen3.8-27B-PN3-P0k19.gguf
  cache       /home/you/.cache/huggingface/hub/models--iamMrDuncan--Qwen3.8-27B-PN3-GGUF/blobs/15ce2d4469673f1f1753c55de3780cfe04311dd6f05cb41633d73df128e91476
  cached      no

Card
  title    NVIDIA Tesla P100 16 GB
  compute  6.0
  memory   at least 16000 MiB
  driver   525.60.13 or newer

Engine
  release  engine-v0.1.0
  arch     sm60
  image    ghcr.io/iammrduncan/tensorless-engine@sha256:35409ff97c2f518e468790c28bd19206ae777a945fcffe5775585e28fd9939a7
```

The output goes on with **Run** (the context, the arguments, the engine switches and the full container command) and **Receipt** (what was measured). Each part:

- **The header** says which recipe file this is, whether it is released, and whether a GPU on this host fits it.
- **Model** is the exact file: its repository and pinned revision, its sha256 and size, the download address, and where it goes in your cache.
- **Card** is what the GPU needs: compute capability, memory and driver.
- **Engine** is the image, by digest, and the engine release it was measured on.
- **Run** is how the server is started, with the command your host would run.
- **Receipt** is what was measured with exactly these settings. The [Receipts](../../reference/receipts.md) page has them all.

`info` takes most of the flags `run` takes. Add `--context`, `--port`, `--host`, `--gpu` or arguments after `--`, and it shows the command with them, with an `UNTUNED` line in the header when they change the measured configuration. Add `--runtime docker` to see the docker command.

## Get it as JSON

```sh
tensorless info pn3 --json
```

The JSON holds the whole recipe under `recipe`, and what the launcher worked out for this host: `published`, `fits`, `download` (the URL, the blob and snapshot paths, the size and whether it is cached), `runtime`, `gpu`, `threads`, `command` (as a list of arguments) and `endpoint`.

`tensorless models --json` and `tensorless capabilities --json` list every recipe in short form: model, card, status and engine release.

## Show the plan of a command

Add `--dry-run` to `run`, `pull`, `serve` or `stop`. Nothing is pulled, downloaded or started:

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF --card p100 --dry-run
```

```text
dry run of 'tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF': nothing is pulled, downloaded or started.
recipe:   iamMrDuncan/Qwen3.8-27B-PN3-GGUF on p100 (released), from recipes/iamMrDuncan/Qwen3.8-27B-PN3-GGUF/p100.toml
gpu:      chosen with --card; GPU detection skipped; the container gets GPU all
runtime:  podman

# 1. pull the engine image, unless 'podman image inspect' finds it
podman pull \
    ghcr.io/iammrduncan/tensorless-engine@sha256:35409ff97c2f518e468790c28bd19206ae777a945fcffe5775585e28fd9939a7

# 2. download the model (12.95 GB), resuming a partial download, and check its sha256 15ce2d4469673f1f1753c55de3780cfe04311dd6f05cb41633d73df128e91476
GET https://huggingface.co/iamMrDuncan/Qwen3.8-27B-PN3-GGUF/resolve/c701b6690ec360b16d40f85df1cb76062e30cd38/Qwen3.8-27B-PN3-P0k19.gguf
    into /home/you/.cache/huggingface/hub/models--iamMrDuncan--Qwen3.8-27B-PN3-GGUF/blobs/15ce2d4469673f1f1753c55de3780cfe04311dd6f05cb41633d73df128e91476
```

It goes on with the snapshot link, the full `podman run` command, and how it waits for the server. This example ran on a machine without the card, so `--card p100` chose the recipe. On a P100 host, leave `--card` out: the `gpu:` line then names the GPU that matched.

A dry run still asks `nvidia-smi` for the GPU list, so it shows what would happen on this host. Without a GPU, `--dry-run` and `info` fall back to the model's only recipe; pass `--card p100` to choose it yourself.

## Check that it worked

Compare the command with the recipe file named on the `recipe` line, in [recipes/](https://github.com/iammrduncan/tensorless/tree/main/recipes): the image digest, the model's revision and sha256, the arguments and the switches are the same. When you then run the command without `--dry-run`, the launcher downloads and starts exactly that.
