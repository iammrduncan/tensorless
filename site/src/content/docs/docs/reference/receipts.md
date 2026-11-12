---
title: "Receipts"
description: "What was measured for each engine-v0.2.0 recipe, by whom and on what: speed, the checks every step had to pass, and the published image's own run."
type: reference
sidebar:
  order: 7
---

A receipt is what the release gate measured for a recipe: speed at stated contexts, quality checks, and the image digest and model hash they were taken on. Each number below comes with its scope. The source files are in [`docs/receipts/engine-v0.2.0/`](https://github.com/iammrduncan/tensorless/tree/main/docs/receipts/engine-v0.2.0). The receipts of the first release are in [`docs/receipts/engine-v0.1.0/`](https://github.com/iammrduncan/tensorless/tree/main/docs/receipts/engine-v0.1.0).

| Recipe | Receipt file |
|---|---|
| `iamMrDuncan/Qwen3.8-27B-PN3-GGUF` on `p100` | [Qwen3.8-27B-PN3-GGUF-p100.md](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.2.0/Qwen3.8-27B-PN3-GGUF-p100.md) |
| `iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF` on `p100` | [Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF-p100.md](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.2.0/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF-p100.md) |

## Two kinds of measurement

| Kind | Who measured | On what | How |
|---|---|---|---|
| Research receipt | A hill climb's judge | A source build of the engine's research commit, with the same CUDA configuration as the image, on one P100 | Timed from outside the engine: the request and each streamed token are stamped on the judge's own clock. Greedy decoding. |
| Published image | `tensorless bench`, and a prefill timing by the same method | The released image digest, with exactly the recipe's switches and flags, on one Tesla P100-PCIE-16GB | Decode: the median of 3 runs of 256 greedy tokens after a 429-token prompt. See [Command line](cli.md#bench). Prefill: one cold prompt, its tokens over the time from the request to the first token, the median of 3. |

The two use different prompts, so a few percent of difference between them is expected.

## Qwen3.8-27B-PN3

The engine source of v0.2.0 is the champion of an automated prefill hill climb that started from engine-v0.1.0: research commit `df673571`, exported unchanged except for six comment lines, plus one fix made afterwards (see [`engine/PROVENANCE.md`](https://github.com/iammrduncan/tensorless/blob/main/engine/PROVENANCE.md)). The recipe's flags and its 21 engine switches are the ones engine-v0.1.0 ran with.

### Speed, research receipt

| Measure | engine-v0.2.0 | engine-v0.1.0 | Scope |
|---|---:|---:|---|
| Prefill, code prompt | **200.1 tok/s** | 121.4 | one cold prompt of about 4,100 tokens in micro-batches of 512; the median of 3 confirmation blocks |
| Prefill, prose prompt | **198.9 tok/s** | 119.6 | the same, prose prompt |
| Decode, code / prose | 30.43 / 30.31 tok/s | 29.95 / 29.96 | 256 greedy tokens after a cold prompt of about 600 tokens |
| First token of the decode prompt, code / prose | 4.35 / 4.81 s | 7.12 / 7.83 | the same request |
| Q3Floor decode, same card, same cells | 26.53 / 26.58 tok/s | 26.26 / 26.27 | the comparison model, decoded in every block |

The climb ran 369 attempts and promoted 26. Prefill is 1.65× engine-v0.1.0. The decode was a guard: no step could leave it more than 1% below the starting point.

### What every step had to keep

| Check | Bar held |
|---|---|
| KL divergence to engine-v0.1.0's outputs: a prose prompt of about 2,350 tokens, then 128 forced tokens | within 2.593e-3 |
| Same top token as engine-v0.1.0 | at 95% or more of positions |
| Decode-path perplexity | within 0.25% |
| Q3Floor's greedy text | bit-identical |
| PN golden blocks, and one real tensor of each PN class | decoded bit-exactly on CPU and GPU |
| GPU memory free with the 65,536-token context allocated | at least 512 MiB |
| A hidden prompt of about 2,400 tokens | no more than 1% slower |

The first ten promotions left the arithmetic of a full 512-token batch as it was. The later ones run the deeper blocks' batched products on fp16 operands, inside these bars. So this engine's outputs are close to engine-v0.1.0's, not identical: on two decode prompts, the 256 greedy tokens came out as different text.

### The published image (2026-11-12)

`ghcr.io/iammrduncan/tensorless-engine@sha256:b95f77cd01cf0f785ee8cc853fc09454bdf4851f7214e8253c6529759c1eb7a7`, measured as `0.2.0-rc1`, on one Tesla P100-PCIE-16GB that reported compute capability 6.0 and 16,269 MiB, with the model file checked against sha256 `15ce2d44…`:

| Measure | Value |
|---|---:|
| `tensorless bench` decode, median of 3 runs (256 tokens after a 429-token prompt) | **30.48 tok/s** (30.44 / 30.48 / 30.49) |
| Against the research receipt (30.43) | +0.2% |
| `llama-server`'s own decode timer, 3 runs of 256 tokens after an 18-token prompt | 30.49 / 30.50 / 30.48 tok/s |
| Time to first token, 429-token prompt | 2.60 s (median); engine-v0.1.0's image: 4.46 s |
| Prefill, a cold prompt of 3,744 tokens (code) / 4,657 tokens (prose) | **198.9 / 201.5 tok/s** |
| The same prefill on engine-v0.1.0's image, same card, same day | 118.9 / 121.0 tok/s |
| A chat of three turns with the prompt cache on, sent twice | answers, the same both times |
| GPU memory, idle / after the first prompt | 14,139 / 15,383 MiB |
| Load to listening, page cache partly warm | 26.0 s |

`bench` ran on another computer, through Wi-Fi, ssh and a port-forward to the server. The network adds latency to the time to first token, but not to the decode rate, which is timed between streamed tokens.

### A fault found before the release

Before the release candidate was built, an image built from the champion's source was run on a P100. It was as fast as the judge had measured, and it aborted on the second turn of any chat that reused the prompt cache. The released engine-v0.1.0 image answered the same chats. The hill climb's judge had sent every prompt with the prompt cache off, so no judged run reached the fault. The cause was in how the engine reported a position to the server while it held a prompt batch. The fix is 12 lines and changes no arithmetic. The published image has it, and the chat test above is the one that found the fault.

### Not measured yet

- **Qwen3.8-27B-PN3's quality checks on this engine.** Perplexity, needle recall and the task screen were last run on the decode climb's starting point ([the engine-v0.1.0 receipt](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.1.0/Qwen3.8-27B-PN3-GGUF-p100.md)).
- **32K and 65K context on this engine**, for decode and for prefill. At those contexts only the decode climb's starting point was measured: 7.55 / 7.39 tok/s for Qwen3.8-27B-PN3, against 24.52 / 22.92 for Q3Floor.
- **Saving and restoring a slot, a context shift on a full context, and a request cancelled in the middle of a prompt.**
- **Speculative decoding** for Qwen3.8-27B-PN3.

## Qwen3.8-27B-UD-IQ3_XXS-Q3Floor

### The published image (2026-11-12)

The same image digest, measured as `0.2.0-rc1` and released as `0.2.0-sm60`, on one Tesla P100-PCIE-16GB, with exactly the recipe's switches and flags. The model file was checked against sha256 `4b4d6a39…`, the same bytes as the Hugging Face upload at revision `11f76ec7`.

| Measure | Value |
|---|---:|
| `tensorless bench` decode, median of 3 runs (256 tokens after a 429-token prompt) | **26.69 tok/s** (26.69 / 26.77 / 26.66) |
| Against the research receipt (26.53) | +0.6% |
| `llama-server`'s own decode timer, 3 runs of 256 tokens after an 18-token prompt | 26.70 / 26.71 / 26.71 tok/s |
| Time to first token, 429-token prompt | 3.86 s (median) |
| Prefill, a cold prompt of 3,744 tokens (code) / 4,657 tokens (prose) | 144.7 / 148.4 tok/s |
| A chat of three turns with the prompt cache on, sent twice | answers, the same both times |
| GPU memory, idle / generating | 13,289 / 13,869 MiB |
| Load to listening, page cache partly warm | 29.8 s |

### Research receipt

The hill climbs' judge decoded this model in every block, as the comparison model for Qwen3.8-27B-PN3.

| Measure | Code | Prose | Scope |
|---|---:|---:|---|
| Decode, short context | 26.53 tok/s | 26.58 tok/s | 256 greedy tokens after a cold prompt of about 600 tokens, on this engine's research commit |
| Decode at 32K | 24.52 tok/s | 24.54 tok/s | 1,024 greedy tokens after a prompt filling the context, on the decode climb's starting commit |
| Decode at 65K | 22.92 tok/s | 22.97 tok/s | the same |

Its greedy text stayed bit-identical through both climbs, and its speed was guarded within 3%.

### Not measured yet

- 32K and 65K context on this engine.
- This model's prefill on engine-v0.1.0, for comparison.
- Speculative decoding (MTP) on this engine. The earlier kernel campaign ran it at about 28 tok/s on its own engine version; that mode is not in this recipe.
