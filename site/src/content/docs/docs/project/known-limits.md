---
title: "Known limits"
description: "What tensorless v0.2.0 does not do yet, what has not been measured, and what to avoid."
type: project
sidebar:
  order: 5
---

tensorless v0.2.0 is an early release. This page lists what it does not do yet, what has not been measured, and the one thing to avoid. Each item says what is true today; none of them is a promise of when it will change.

## Long context is not measured on this engine

The receipts for engine-v0.2.0 measure decode at short context, 256 tokens after a prompt of a few hundred, and prefill on prompts of about 4,000 tokens. Decode and prefill at 32K and 65K context have not been measured on this engine yet.

- **Qwen3.8-27B-PN3:** only the decode hill climb's starting point was measured at long context, at 7.55 and 7.39 tok/s for 32K and 65K. That climb then made short-context decode almost four times faster. Attention takes a larger share of each token at long context, so expect this engine's long-context decode to be below its 30 tok/s, and its long-prompt prefill to be below its 200 tok/s, by amounts nobody has measured yet.
- **Qwen3.8-27B-UD-IQ3_XXS-Q3Floor:** 24.5 and 22.9 tok/s at 32K and 65K, measured on the decode climb's starting commit, two engine releases back. This model's code path changed little after that: its text stayed bit-identical through both hill climbs, and its speed was guarded within 3%.

Both recipes still allocate the full 65,536-token context.

## The quality checks were not repeated for this engine

Qwen3.8-27B-PN3's perplexity, needle recall and task screen were measured on the decode hill climb's starting point. Each hill climb since has held the model's outputs close to its own starting point: KL divergence within 2.593e-3, the same top token at 95% or more of positions, perplexity within 0.25%. engine-v0.2.0 also changed the arithmetic of prompt processing: from block 27 up, a prompt's batched products run on fp16 operands. So this engine's answers are close to engine-v0.1.0's, not identical: on two test prompts, 256 greedy tokens came out as different text. The full quality checks have not been run again on this engine.

## Little memory to spare

With Qwen3.8-27B-PN3, the engine uses 15,383 MiB of the card's 16,269 MiB once it has read its first prompt, about 800 MiB more than engine-v0.1.0. Anything else on the card may not fit.

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
