---
title: "Try a recipe of your own"
description: "Write a draft recipe in a directory of your own, check it with info and --dry-run, and run it with --recipes and --allow-draft."
type: how-to
sidebar:
  order: 3
---

A recipe is a small TOML file: one model on one card. To try other settings, another model or your own engine build, write a draft recipe in a directory of your own and point the launcher at it. Nothing in the launcher needs to change.

## Before you start

You need:

- the `tensorless` launcher (see [Install the launcher](../setup/install-launcher.md));
- for a new model: its Hugging Face repository, the commit to pin, the GGUF file's name, its sha256 and its exact size in bytes;
- the [Recipe file](../../reference/recipes.md) reference open beside you.

## 1. Make a recipes directory

A recipes directory holds `<org>/<repo>/<card>.toml`, where `<org>/<repo>` is the model's Hugging Face repository. Start from a built-in recipe:

```sh
mkdir -p ~/my-recipes/iamMrDuncan/Qwen3.8-27B-PN3-GGUF
curl -fsSL -o ~/my-recipes/iamMrDuncan/Qwen3.8-27B-PN3-GGUF/p100.toml \
  https://raw.githubusercontent.com/iammrduncan/tensorless/main/recipes/iamMrDuncan/Qwen3.8-27B-PN3-GGUF/p100.toml
```

With `--recipes DIR`, the launcher reads only that directory, not its built-in recipes. Copy every recipe you want to use with it.

## 2. Make it a draft and change it

Open the file and set its status to draft:

```toml
status = "draft"
```

Then make your change. For example:

- **Other settings:** edit `[run]` (`context`, `args`) or `[run.env]` (the [engine switches](../../reference/engine-switches.md)).
- **Another model:** put the file under that model's `<org>/<repo>/` directory, and fill in `[model]`: `hf_repo` (the same as the directory), `revision` (a 40-character commit, not a branch), `file`, `sha256`, `size_bytes` and `license`. Change `run.alias` to the new repo id, and `model.name` to a short name of its own, or remove it.
- **Your own engine build:** push it to a registry you control, and set `[engine].image` to `<registry>/<name>@sha256:<digest>`. A tag is refused.

To get a file's sha256 and size from a copy you downloaded:

```sh
sha256sum MODEL.gguf
stat -c %s MODEL.gguf
```

Leave out the `llama-server` flags the launcher sets itself: `-m`, `--host`, `--port`, `-c`, `--alias`, `-t` and `-tb` (and their long forms), and `OMP_NUM_THREADS` in `[run.env]`.

## 3. Check it

`info` reads the recipe and shows it exactly as it would run. It needs no GPU:

```sh
tensorless info iamMrDuncan/Qwen3.8-27B-PN3-GGUF --recipes ~/my-recipes --card p100
```

A recipe with a mistake fails with every problem listed, for example:

```text
tensorless: /home/user/my-recipes/iamMrDuncan/Qwen3.8-27B-PN3-GGUF/p100.toml: invalid recipe:
  - engine.image must be pinned by digest (name@sha256:<64 hex>), not "ghcr.io/iammrduncan/tensorless-engine:0.2.0-sm60"
  - run.args must not set -c; the launcher sets it
```

A misspelled key fails with `unknown keys:` and its name. Fix the file and run `info` again until it shows the recipe.

Then see the whole plan, with nothing pulled or started:

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF --recipes ~/my-recipes --allow-draft --dry-run
```

`tensorless models --recipes ~/my-recipes` lists it with the status `draft`.

## 4. Run it

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF --recipes ~/my-recipes --allow-draft
```

Without `--allow-draft`, the launcher refuses a draft:

```text
tensorless: the recipe for iamMrDuncan/Qwen3.8-27B-PN3-GGUF on p100 is a draft: it has not passed the release gate. Pass --allow-draft to use it anyway; 'tensorless info iamMrDuncan/Qwen3.8-27B-PN3-GGUF' shows it
```

With it, the launcher prints `note: using the draft recipe for iamMrDuncan/Qwen3.8-27B-PN3-GGUF on p100 (--allow-draft)`, then pulls, verifies and serves as usual. It has worked when it prints that the model `is ready at http://127.0.0.1:8080`.

A recipe whose revision, sha256 or image digest is still all zeros, or whose `size_bytes` is 0, is refused even with `--allow-draft`: those are placeholders, and nothing can be pulled for them.

## 5. Measure it

With the server running, measure decode speed:

```sh
tensorless bench iamMrDuncan/Qwen3.8-27B-PN3-GGUF --recipes ~/my-recipes
```

`bench` compares the result with the recipe's `[receipt]`. If you changed the settings, that receipt no longer describes them: remove or replace its numbers. See [Benchmark a running server](../use/benchmark.md).

## Make it a released recipe

A recipe becomes `released` in the repository, through a pull request, only with a receipt: the release gate's measurements of that model on that card, with that image. A pull request that changes `[run]` or `[engine]` of a released recipe needs a new receipt too. How to propose one is in [Contributing](../../project/contributing.md).
