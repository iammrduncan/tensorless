---
title: "Known limits"
description: "What tensorless v0.1.0 does not do yet, what has not been measured, and what to avoid."
type: project
sidebar:
  order: 5
---

tensorless v0.1.0 is a first release. This page lists what it does not do yet, what has not been measured, and the one thing to avoid. Each item says what is true today; none of them is a promise of when it will change.

## Long context is not measured on this engine

The receipts for engine-v0.1.0 measure decode at short context: 256 tokens after a prompt of a few hundred. Decode at 32K and 65K context has not been measured on this engine yet.

- **Qwen3.8-27B-PN3:** only the hill climb's starting point was measured at long context, at 7.55 and 7.39 tok/s for 32K and 65K. The climb then made short-context decode almost four times faster. Attention takes a larger share of each token at long context, so expect this engine's long-context decode to be below its 30 tok/s, by an amount nobody has measured yet.
- **Qwen3.8-27B-UD-IQ3_XXS-Q3Floor:** 24.5 and 22.9 tok/s at 32K and 65K, measured on the climb's starting commit. This model's code path changed little after that: its text stayed bit-identical, and its speed was guarded within 3%.

Both recipes still allocate the full 65,536-token context.

## Don't run PN models on the CPU

The engine's CPU backend decodes PN weights wrong in its dot product: it shifts by element index instead of pair index, and most PN weights come out wrong. The GPU paths are correct, and every recipe runs the whole model on the GPU, so `tensorless run` is not affected.

It matters only if you run the engine by hand. Don't run Qwen3.8-27B-PN3 on the CPU backend, or with only some layers on the GPU, until this is fixed. [`engine/PROVENANCE.md`](https://github.com/iammrduncan/tensorless/blob/main/engine/PROVENANCE.md) records it.

## No speculative decoding yet

Neither recipe uses speculative decoding. An earlier kernel campaign ran Q3Floor with multi-token prediction (MTP) at about 28 tok/s at 32K and 65K context, on its own engine version, but that mode is not in any recipe.

## One request at a time

Both recipes run `llama-server` with one slot (`-np 1`), the configuration that was measured. It serves one request at a time; others wait their turn. Batched or concurrent throughput has not been measured.

## One card family, on x86_64 Linux

- **Only the Tesla P100 16 GB** (`sm_60`) has recipes. A 12 GB P100 does not fit them, and other cards have no recipe.
- **The engine image is built for Linux on x86_64 only.** The launcher also comes for Linux aarch64 and macOS on Apple silicon, where it can list and show recipes and benchmark a server running on another machine, but cannot serve a model.
- **The driver must be from R525 to R580.** R580 is the last driver branch that supports Pascal.

## Not yet confirmed

The project's design notes list these as still to confirm:

- the image's supported driver range, on a real P100 host;
- the launcher's container support, on rootless podman with CDI and on docker with the NVIDIA container toolkit.

If you run tensorless on either setup, your result helps close these. Tell us on [Discord](https://discord.gg/3Qs2uejUf9) or in an [issue](https://github.com/iammrduncan/tensorless/issues).

## Quality on long prose

Qwen3.8-27B-PN3 matches Q3Floor on code, short text, long-context recall and a task screen, but has 7.5% higher perplexity on long prose at 32K. The cause is the 3-bit quantization, not the kernels. [Why a P100-native weight format](../explanation/pn-format.md#what-it-costs) has the measurements.
