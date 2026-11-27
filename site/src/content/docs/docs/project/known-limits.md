---
title: "Known limits"
description: "What tensorless v0.4.1 does not do yet, what has not been measured, and what to avoid."
type: project
sidebar:
  order: 5
---

tensorless v0.4.1 is an early release. This page lists what it does not do yet, what has not been measured, and the one thing to avoid. Each item says what is true today; none of them is a promise of when it will change.

## Long context is not measured for the Qwen3.8-27B models

For the two Qwen3.8-27B models, the receipts measure decode at short context, 256 tokens after a prompt of a few hundred, and prefill on prompts of about 4,000 tokens. Decode and prefill at 32K and 65K context have not been measured on this engine yet.

- **Qwen3.8-27B-PN3:** only the decode hill climb's starting point was measured at long context, at 7.55 and 7.39 tok/s for 32K and 65K. That climb then made short-context decode almost four times faster. Attention takes a larger share of each token at long context, so expect this engine's long-context decode to be below its 30 tok/s, and its long-prompt prefill to be below its 200 tok/s, by amounts nobody has measured yet.
- **Qwen3.8-27B-UD-IQ3_XXS-Q3Floor:** 24.5 and 22.9 tok/s at 32K and 65K, measured on the decode climb's starting commit, five engine releases back. This model's code path changed little after that: its text stayed bit-identical through both Qwen hill climbs, and its speed was guarded within 3%.

Both Qwen3.8-27B recipes still allocate the full 65,536-token context. North-Mini-Code-1.0 was measured after a 32K prompt, at 66 tok/s; its recipe allocates 36,864 tokens, and nothing longer has been measured.

## Inkling-2-Mini is a draft

The Inkling-2-Mini recipe stays on engine-v0.4.0 and pins its image, `0.4.0-sm60`, which has not been measured with the model on a P100: its numbers are the research build's. `pull`, `run` and `serve` refuse it unless you pass `--allow-draft`, and with it they run a configuration the release gate has not seen. It becomes a launch target in a later patch release, once its gate has run. Also not measured for it: `/v1/chat/completions` (its chat template has a thinking channel, and how the server shows it was not checked), the prompt cache across turns, contexts above 32,768 tokens and a load with no GPU. See its [receipt draft](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.4.0/Inkling-2-Mini-LookAtMyQuant-12.8GB-p100.md).

## The quality checks were not repeated for this engine

No quality checks were run for North-Mini-Code-1.0: its hill climb held the model's output close to the climb's starting point (mean KL at most 0.003 nat, the same top token at 98% or more of positions), not to a quality bar. Above about 8,192 tokens of context, its output differs from the starting point's, inside those limits.

Qwen3.8-27B-PN3's perplexity, needle recall and task screen were measured on the decode hill climb's starting point. Each hill climb since has held the model's outputs close to its own starting point: KL divergence within 2.593e-3, the same top token at 95% or more of positions, perplexity within 0.25%. engine-v0.2.0 also changed the arithmetic of prompt processing: from block 27 up, a prompt's batched products run on fp16 operands. So this engine's answers are close to engine-v0.1.0's, not identical: on two test prompts, 256 greedy tokens came out as different text. The full quality checks have not been run again on this engine.

## Little memory to spare

With Qwen3.8-27B-PN3, the engine uses 15,381 MiB of the card's 16,269 MiB once it has read its first prompt, about 800 MiB more than engine-v0.1.0. Anything else on the card may not fit. North-Mini-Code-1.0 usually leaves more: 1,840 MiB free from its first prompt on. But at a chat turn that reuses the prompt cache, its used memory can rise by 864 MiB and stay there: in engine-v0.4.1's release gate it rose to 15,295 MiB, with 976 MiB free. The server did not fail. engine-v0.4.0's image, run in the same hour on the same card, did the same, to 15,273 MiB, as it had in engine-v0.4.0's own gate, so the step has been there since engine-v0.4.0; engine-v0.3.0's gate ran the same chat without it. Its cause is not known ([receipt](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.4.1/North-Mini-Code-1.0-GGUF-p100.md#gpu-memory-at-the-chats-second-pass)).

## North-Mini-Code-1.0: the prompt cache can change its answers

With the prompt cache on, a request that reuses a cached prompt can get a different greedy answer from the same request sent cold. In the release gate, a three-turn chat sent twice gave different answers on the two passes, and the same request sent twice with `cache_prompt` false gave the same text. The hill climb's judge found it too, for the champion and the candidates alike, so it is not new in engine-v0.3.0 or later releases. In engine-v0.4.1's gate, the chat's two passes gave the same token counts on engine-v0.4.1's image as on engine-v0.4.0's. If you need the same answer for the same request, send it with `cache_prompt` false. The Qwen3.8-27B models gave the same answers both times.

## North-Mini-Code-1.0 reasons before it answers

`/v1/chat/completions` returns its reasoning in `reasoning_content` and the answer in `content`. The reasoning counts against `max_tokens`: in the release gate, a limit of 200 tokens left every answer empty. Give it room.

## Don't run PN models on the CPU

The engine's CPU backend decodes PN weights wrong in its dot product: it shifts by element index instead of pair index, and most PN weights come out wrong. The GPU paths are correct, and every recipe runs the whole model on the GPU, so `tensorless run` is not affected.

It matters only if you run the engine by hand. Don't run Qwen3.8-27B-PN3 on the CPU backend, or with only some layers on the GPU, until this is fixed. [`engine/PROVENANCE.md`](https://github.com/iammrduncan/tensorless/blob/main/engine/PROVENANCE.md) records it.

## No speculative decoding yet

No recipe uses speculative decoding. An earlier kernel campaign ran Q3Floor with multi-token prediction (MTP) at about 28 tok/s at 32K and 65K context, on its own engine version, but that mode is not in any recipe.

## One request at a time

Every recipe runs `llama-server` with one slot (`-np 1`), the configuration that was measured. It serves one request at a time; others wait their turn. Batched or concurrent throughput has not been measured.

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
