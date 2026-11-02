---
title: "Share your results"
description: "Collect a benchmark, the launcher's version and your card's details, and post them on Discord or in a GitHub issue."
type: how-to
sidebar:
  order: 3
---

Numbers from other people's P100s show whether a recipe holds up on cards, drivers and hosts it was not measured on. This page collects what makes a result useful, and says where to post it.

You need a model running and benchmarked: see [Benchmark a running server](benchmark.md).

## 1. Benchmark as JSON

Run the benchmark with its defaults, so your numbers compare with everyone else's, and save the JSON:

```sh
tensorless bench iamMrDuncan/Qwen3.8-27B-PN3-GGUF --json > bench.json
```

It holds every run, the medians, and the difference from the recipe's receipt.

## 2. Record the launcher and the card

```sh
tensorless version
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv
```

`tensorless version` names the release and commit. The `nvidia-smi` line gives the card's exact model, its memory, the driver version and its power limit, which can all change the speed. `tensorless capabilities --json` adds the recipes your launcher knows, if someone asks.

## 3. Note how you ran it

Say, in a sentence or two:

- whether you used podman or docker;
- whether you passed `--context` or arguments after `--` (an untuned run measures something else);
- whether anything else used the GPU while it ran;
- anything unusual about the host: a PCIe riser, a card in an x4 slot, a cooling mod.

The JSON's `endpoint` shows the address you benchmarked; check that you are happy to share it.

## 4. Post it

- **Discord:** [join the tensorless server](https://discord.gg/3Qs2uejUf9) and post your results with the files above. It is the place for questions, results and ideas.
- **GitHub:** if your numbers are far from the receipt, or something failed, [open an issue](https://github.com/iammrduncan/tensorless/issues) with the same details, so it can be traced.

## Check that it worked

Your post includes `bench.json` (or its median and `vs_receipt_percent`), the output of `tensorless version`, the `nvidia-smi` line, and how you ran it. That is enough for someone else to repeat your measurement.
