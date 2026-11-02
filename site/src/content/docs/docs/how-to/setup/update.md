---
title: "Update tensorless"
description: "Move to a newer launcher release, and know what an update changes: the recipes the launcher knows, and never a model's engine without a new receipt."
type: how-to
sidebar:
  order: 4
---

Update the launcher to get new launch targets and recipes that moved to a newer engine release. An update replaces one binary. Running servers, downloaded models and pulled images are not touched.

## What you need

- The version you have: `tensorless version`.
- The version you want, from the [releases page](https://github.com/iammrduncan/tensorless/releases).

## 1. Install the new launcher

Run the install script again, with the release you want:

```sh
curl -fsSL https://tensorless.dev/install.sh | TENSORLESS_VERSION=0.1.0 sh
```

Replace `0.1.0` with the version you want. Without `TENSORLESS_VERSION`, the script installs the release it names as its default. If you installed somewhere other than `~/.local/bin`, set `TENSORLESS_INSTALL_DIR` to the same place, so that the new binary replaces the old one.

If you installed from an archive, download and check the new archive the same way, and copy its binary over the old one. See [Install the launcher](install-launcher.md#install-from-a-release-archive).

To go back to an older release, do the same with the older version.

## 2. See what the new launcher knows

```sh
tensorless models
```

The list shows every launch target in this launcher, with its engine release.

## 3. Restart a running model

A server that was running before the update keeps running on what it started with. To move it to the new launcher's recipe, stop it and run it again:

```sh
tensorless stop iamMrDuncan/Qwen3.8-27B-PN3-GGUF
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF
```

## What an update changes

- **The recipes.** Recipes are built into the launcher, so a new launch target, or a recipe that moved to a new engine release, comes with a new launcher.
- **The engine, only when a recipe moves.** A recipe pins its engine image by digest. A model's engine changes only when its recipe moves to a new engine release, and a recipe moves only with a new receipt for that model on that card. If it moved, the next `run` or `pull` pulls the new image. Older images stay in the container runtime's storage until you remove them, with `podman rmi` or `docker rmi`.
- **The model, only when a recipe pins a new file.** Models stay in the Hugging Face cache. If the recipe's revision and sha256 did not change, nothing is downloaded again. If they changed, the new file is downloaded next to the old one, which stays until you delete it.

[How a release is made](../../explanation/releases-and-receipts.md) explains why recipes only move with a receipt.

## How to tell it worked

`tensorless version` prints the new version.
