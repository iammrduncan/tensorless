---
title: "Tuned and untuned runs"
description: "What it means for a run to match its recipe, what makes a run untuned, and why the launcher labels it instead of forbidding it."
type: explanation
sidebar:
  order: 7
---

A recipe's numbers are true for one exact configuration. This page explains what that configuration is, which changes take a run outside it, and why tensorless lets you make those changes but tells you when you have.

## What "tuned" means

Each recipe has a `[run]` section: the context length, every `llama-server` flag, and the engine switches, exactly as the release gate measured them. For Qwen3.8-27B-PN3 on the P100 that is a 65,536-token context, a q4_0 KV cache, flash attention, batches of 512, and 21 `GGML_CUDA_*` switches that turn on the P100 kernels.

A **tuned** run is that configuration, on the image the recipe pins, with the model file whose sha256 the recipe names. The receipt's numbers apply to it. `tensorless run MODEL`, with no extra settings, is always a tuned run.

## What makes a run untuned

Three things change the measured configuration:

| You pass | Why it matters |
|---|---|
| `--context N`, with N other than the recipe's | The KV cache changes size, and with it the memory left on the card and the speed of attention. |
| Extra `llama-server` arguments after `--` | Any flag can change what the server does: sampling, batching, offload, cache types. |
| `MODEL@REVISION`, a commit other than the recipe's | Another revision of the repository may hold other files. The downloaded file must still match the recipe's sha256. |

When you do any of these, the launcher says so before it starts:

```text
UNTUNED: --context 32768 instead of 65536. This is not the configuration the receipt measured; its numbers do not apply.
```

`tensorless info --json` reports the same thing in its `untuned` field, `--dry-run` prints it in the plan, and `tensorless bench` adds a note under its results.

Some options do not make a run untuned: `--port`, `--host`, `--gpu`, `--runtime` and `--cache`. They change where the server listens, which card it uses, how the container starts and where the model is stored, but not what the engine computes.

## Why label, and not forbid

The launcher's rule is to [refuse rather than guess](how-tensorless-works.md#refuse-rather-than-guess). An untuned run is not a guess, though. It is you choosing something on purpose: a shorter context to leave room for another process, a sampling flag your client needs, a newer revision to test. Refusing those would only push people to run the engine by hand, where nothing checks anything.

So the line is drawn by what the launcher can know:

- **Things it can check, it checks,** every time: the card and the driver (unless you pass `--card`, which skips GPU detection), the image digest, the model's hash. Those are refused when wrong.
- **Things that are your choice, it allows,** and labels. The label is there so that nobody, you included, compares an untuned run with the receipt and draws the wrong conclusion.

A run that is slower than its receipt is worth knowing about only if it is tuned. That is why the label exists.

## Drafts and unpublished recipes

Two more kinds of recipe sit outside the tuned path:

- **A draft** is a recipe that has not passed the release gate. Its settings may be right, but nothing has confirmed them on the image it names. The launcher refuses it unless you pass `--allow-draft`, and says it is using a draft when you do. `MODEL@REVISION` needs the same flag, for the same reason: no gate measured it.
- **An unpublished recipe** still has placeholder values: an all-zero revision or image digest, or a model size of 0. It cannot run at all, even with `--allow-draft`, because there is nothing real to pull. `tensorless info` and `--dry-run` still show it, so its plan can be reviewed before the upload exists.

Both released recipes in tensorless v0.2.0 need no flags. `tensorless models` shows each recipe's status.
