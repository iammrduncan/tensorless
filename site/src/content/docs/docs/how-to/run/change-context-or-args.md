---
title: "Change the context length or server arguments"
description: "Run a model with another context length, or pass extra llama-server arguments, and know what that does to its measured numbers."
type: how-to
sidebar:
  order: 4
---

Each recipe runs `llama-server` with exactly the settings that were measured, including a context of 65,536 tokens. You can change the context length, or add `llama-server` arguments of your own. The launcher then calls the run **untuned**: it is no longer the configuration the receipt measured, so the receipt's numbers don't apply to it.

You need the launcher installed and a host that passes `tensorless doctor`. See [Prepare a Linux host for a P100](../setup/prepare-host.md). Words such as *recipe* and *receipt* are in the [glossary](../../reference/glossary.md).

## See the measured settings first

```sh
tensorless info pn3
```

Under **Run**, `context` is the context length and `args` lists every other argument the recipe passes, in order. [See exactly what will run](preview-a-run.md) explains the rest of the output.

## Change the context length

Pass `--context` to `run` or `serve`:

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF --context 32768
```

Before it starts, the launcher says what differs from the recipe:

```text
UNTUNED: --context 32768 instead of 65536. This is not the configuration the receipt measured; its numbers do not apply.
```

The value goes to `llama-server` as `-c`. With `-np 1` in the recipes, the server has one slot, and that slot gets the whole context.

:::caution
The recipes' 65,536-token context already takes nearly all of the P100's 16 GB: with Qwen3.8-27B-PN3, the published image used 13,997 MiB once loaded and 14,577 MiB while generating ([the receipt](../../reference/receipts.md)). A larger context needs more memory than the card has, and the server fails while it loads. A smaller one frees memory.
:::

## Pass extra llama-server arguments

Everything after `--` goes to `llama-server`, after the recipe's own arguments:

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF -- --temp 0.7
```

```text
UNTUNED: extra arguments --temp 0.7. This is not the configuration the receipt measured; its numbers do not apply.
```

Because they come last, your arguments can also override one of the recipe's. The launcher sets the model, address, context, alias and threads itself, so it refuses `-m`, `--host`, `--port`, `-c`, `--alias`, `-t` and `-tb` (and their long forms) after `--`:

```text
tensorless: do not pass -c after --: the launcher sets it. Use --host, --port or --context instead; the model, alias and threads come from the recipe
Run 'tensorless help' for usage.
```

Only `run`, `serve` and `info` take arguments after `--`. The others refuse them.

:::tip
Sampling settings such as temperature can also be sent with each request, which leaves the run tuned. See [Connect an OpenAI-compatible client](../use/openai-clients.md).
:::

## Combine both

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF --context 32768 -- --temp 0.7
```

```text
UNTUNED: --context 32768 instead of 65536 and extra arguments --temp 0.7. This is not the configuration the receipt measured; its numbers do not apply.
```

## Check that it worked

Preview the command before you run it, with the same flags and `--dry-run`. The plan's header includes the `UNTUNED` line, and the `podman run` (or `docker run`) command shows your `-c` value and, at the end, your extra arguments:

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF --context 32768 --dry-run -- --temp 0.7
```

To measure an untuned run, start the server, then run `tensorless bench` with the same `--context`. The result notes that the run is untuned, and its comparison with the receipt is only a rough guide. [Tuned and untuned runs](../../explanation/tuned-and-untuned.md) explains why.
