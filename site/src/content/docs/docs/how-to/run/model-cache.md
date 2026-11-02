---
title: "Keep models on another disk"
description: "Point the launcher's model downloads at a larger disk with --cache or the Hugging Face cache variables, and move models you already have."
type: how-to
sidebar:
  order: 5
---

Each launch target is one file of 11.8 to 13 GB. The launcher keeps it in the Hugging Face cache, by default in your home directory. This page puts the cache on another disk, and moves models you have already downloaded.

You need the launcher installed. See [Install the launcher](../setup/install-launcher.md).

## Where the cache is

The launcher uses the first of these that is set:

1. `--cache DIR` on the command line;
2. the `HF_HUB_CACHE` environment variable;
3. `HF_HOME`, with `/hub` added;
4. `~/.cache/huggingface/hub`.

These are the same variables the Hugging Face tools read, so the launcher and those tools share one cache.

## 1. Choose the new place

Pick a directory on a disk with room for every model you want, about 13 GB each. Its path must not contain `:`, because the container mount cannot express one; the launcher refuses such a path.

## 2. Point the launcher at it

For every command, set the variable in your shell's profile:

```sh
export HF_HUB_CACHE=/data/hf-cache
```

Or for one command, pass `--cache`:

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF --cache /data/hf-cache
```

Use the same cache for `pull`, `run`, `serve` and `models`: a command that looks in another cache does not find the file there.

:::caution
`sudo` and service managers usually change `$HOME` and the environment, so a command run that way looks in another cache. The launcher itself does not need root: run it as the user who owns the cache.
:::

## 3. Move models you already have (optional)

Copy the model's whole folder, keeping its symlinks, then remove the old one:

```sh
rsync -a ~/.cache/huggingface/hub/models--iamMrDuncan--Qwen3.8-27B-PN3-GGUF /data/hf-cache/
rm -rf ~/.cache/huggingface/hub/models--iamMrDuncan--Qwen3.8-27B-PN3-GGUF
```

The next command that uses the file checks its sha256 once more, because the launcher remembers a verified file by its path. That reads the whole file, so it takes a while; after it, the check is skipped again.

## What is in the cache

For each model, the launcher writes the layout the Hugging Face tools use:

```text
models--iamMrDuncan--Qwen3.8-27B-PN3-GGUF/
  blobs/15ce2d4469673f1f1753c55de3780cfe04311dd6f05cb41633d73df128e91476
  snapshots/c701b6690ec360b16d40f85df1cb76062e30cd38/Qwen3.8-27B-PN3-P0k19.gguf -> ../../blobs/15ce2d44...
```

- **The blob** is the file itself, named by its sha256. The server's container mounts it read-only.
- **The snapshot** is a symlink named after the recipe's revision and file, so other Hugging Face tools that ask for that revision find the file.
- **A partial download** is the blob's name with `.incomplete` added. Run the same command again to resume it where it stopped. `tensorless models` shows such a model as `partial`.

Before a download, the launcher checks that the disk has room for it and refuses with `not enough disk space` if not.

The launcher remembers which files it has verified in `$XDG_STATE_HOME/tensorless/verified/`, or `~/.local/state/tensorless/verified/` when `XDG_STATE_HOME` is not set: one small file per model, named by its sha256. While the model file keeps the same path, size and modification time, the launcher does not hash it again. Deleting these markers is safe; the next run verifies again.

## Check that it worked

```sh
tensorless models
```

The **CACHED** column says `yes` for a model in the new cache. `tensorless info pn3` shows the full path under **Model**, on the `cache` line, and `tensorless doctor` reports the free space where the cache now is.
