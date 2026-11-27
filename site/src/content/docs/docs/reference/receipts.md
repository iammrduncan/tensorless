---
title: "Receipts"
description: "What was measured for each engine-v0.4.1 recipe, by whom and on what: speed, the checks every step had to pass, and the published image's own run."
type: reference
sidebar:
  order: 7
---

A receipt is what the release gate measured for a recipe: speed at stated contexts, quality checks, and the image digest and model hash they were taken on. Each number below comes with its scope. The source files are in [`docs/receipts/engine-v0.4.1/`](https://github.com/iammrduncan/tensorless/tree/main/docs/receipts/engine-v0.4.1). The receipts of the earlier releases are in [`docs/receipts/engine-v0.4.0/`](https://github.com/iammrduncan/tensorless/tree/main/docs/receipts/engine-v0.4.0), [`docs/receipts/engine-v0.3.0/`](https://github.com/iammrduncan/tensorless/tree/main/docs/receipts/engine-v0.3.0), [`docs/receipts/engine-v0.2.0/`](https://github.com/iammrduncan/tensorless/tree/main/docs/receipts/engine-v0.2.0) and [`docs/receipts/engine-v0.1.0/`](https://github.com/iammrduncan/tensorless/tree/main/docs/receipts/engine-v0.1.0).

| Recipe | Receipt file |
|---|---|
| `iamMrDuncan/Qwen3.8-27B-PN3-GGUF` on `p100` | [Qwen3.8-27B-PN3-GGUF-p100.md](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.4.1/Qwen3.8-27B-PN3-GGUF-p100.md), with its research receipt in [engine-v0.2.0's](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.2.0/Qwen3.8-27B-PN3-GGUF-p100.md) |
| `iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF` on `p100` | [Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF-p100.md](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.4.1/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF-p100.md) |
| `unsloth/North-Mini-Code-1.0-GGUF` on `p100` | [North-Mini-Code-1.0-GGUF-p100.md](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.4.1/North-Mini-Code-1.0-GGUF-p100.md), with its decode research receipt in [engine-v0.3.0's](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.3.0/North-Mini-Code-1.0-GGUF-p100.md) |

The draft recipe, `shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-12.8GB` on `p100`, stays on engine-v0.4.0 and has a receipt draft, [Inkling-2-Mini-LookAtMyQuant-12.8GB-p100.md](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.4.0/Inkling-2-Mini-LookAtMyQuant-12.8GB-p100.md): the research build's numbers only, since engine-v0.4.0's published image has not been measured with it yet. See [Launch targets](launch-targets.md#draft-in-v041-inkling-2-mini).

## Two kinds of measurement

| Kind | Who measured | On what | How |
|---|---|---|---|
| Research receipt | A hill climb's judge | A source build of the engine's research commit, with the same CUDA configuration as the image, on one P100 | Timed from outside the engine: the request and each streamed token are stamped on the judge's own clock. Greedy decoding. |
| Published image | `tensorless bench`, and prefill and decode timings by the same method | The released image digest, with exactly the recipe's switches and flags, on one Tesla P100-PCIE-16GB | Decode: the median of 3 runs of 256 greedy tokens after a 429-token prompt (411 for North-Mini-Code-1.0). See [Command line](cli.md#bench). Prefill: one cold prompt, its tokens over the time from the request to the first token, the median of 3. |

The two use different prompts, so a few percent of difference between them is expected.

## The host sync schedule

engine-v0.4.0 added a setting for how the host thread waits for the GPU in a CUDA synchronize, `GGML_CUDA_SYNC_SCHED` (see [Engine switches](engine-switches.md#the-host-sync-schedule)). Its default is `blocking`: the thread sleeps until the GPU signals. engine-v0.3.0 and earlier always used the CUDA driver's default, `auto`. engine-v0.4.0's release gate measured each launch target both ways:

| Launch target | `auto` | `blocking` | engine-v0.3.0 |
|---|---:|---:|---:|
| Qwen3.8-27B-PN3, `tensorless bench`, two benches | **30.46 / 30.46 tok/s** | 30.20 / 30.13 | 30.45 |
| Qwen3.8-27B-UD-IQ3_XXS-Q3Floor, the same | **26.63 / 26.61 tok/s** | 26.56 / 26.61 | 26.53 |
| North-Mini-Code-1.0, `tensorless bench` | **94.82 tok/s** | 93.57 | 93.70 |
| North-Mini-Code-1.0, decode after a 3,911-token prompt | **79.31 tok/s** | 79.51 | 79.94 |

`blocking` cost Qwen3.8-27B-PN3 about 1%, and made no difference beyond the runs' spread for the other two. So the three released recipes set `GGML_CUDA_SYNC_SCHED = "auto"`, which is how engine-v0.3.0 ran them, and engine-v0.4.1's gate ran every recipe with it. During the Qwen models' `blocking` runs, the host's two other P100s were busy at 100%; the gate's notes do not say whether they were during the `auto` runs.

## Qwen3.8-27B-PN3

This model's research numbers come from engine-v0.2.0, the champion of an automated prefill hill climb that started from engine-v0.1.0: research commit `df673571`, exported unchanged except for six comment lines, plus one fix made afterwards. engine-v0.3.0 added a decode climb on [another model](#north-mini-code-10), engine-v0.4.0 added Inkling-2-Mini, and engine-v0.4.1 adds a prefill climb on North-Mini-Code-1.0. This recipe moved to each after the release gate measured it, unchanged but for engine-v0.4.0's `GGML_CUDA_SYNC_SCHED = "auto"` ([below](#the-published-image-2026-11-27)). The recipe's flags and its 21 engine switches are the ones engine-v0.1.0 ran with.

### Speed, research receipt (engine-v0.2.0)

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

### The published image (2026-11-27)

`ghcr.io/iammrduncan/tensorless-engine@sha256:c89e60b34a0f6098db20f2f2b43c78a93bb70192833cf7656b4a84ae8974346a`, measured as `0.4.1-rc1` and released as `0.4.1-sm60`, on one Tesla P100-PCIE-16GB, with exactly the recipe's switches and flags. engine-v0.4.0's image, `0.4.0-sm60`, ran straight after it on the same card with the same tests.

| Measure | engine-v0.4.1 | engine-v0.4.0, the same hour | engine-v0.4.0's gate (2026-11-24) |
|---|---:|---:|---:|
| Greedy replies, three prompts, 256 tokens each | 3 replies | **the same bytes** | |
| `tensorless bench` decode, median of 3 runs (256 tokens after a 429-token prompt) | **30.30 tok/s** (30.29 / 30.30 / 30.42) | 30.17 (30.14 / 30.17 / 30.18) | 30.46 |
| Against the research receipt (30.43) | −0.4% | −0.9% | +0.1% |
| Time to first token, 429-token prompt | 2.89 s (median) | 2.88 s | 2.81 s; engine-v0.1.0's image: 4.46 s |
| Prefill, a cold prompt of code / prose (3,651 / 4,208 tokens) | **192.8 / 197.5 tok/s** | 192.4 / 197.4 | not measured; engine-v0.3.0's image: 194.6 / 199.2 |
| GPU memory, idle / after the first prompt | 14,139 / 15,381 MiB (15,383 at most) | the same | 14,139 / 15,381 |
| Load to listening | 54.4 s, page cache cold | 3.8 s, warm | 8.4 s, warm |

Against engine-v0.4.0's image, nothing changed. The bench reads about 1% below engine-v0.4.0's own gate, but engine-v0.4.0's image read the same in this hour on this card, so the difference comes from the card or the hour, not from engine-v0.4.1. Prefill is timed from the client through a port-forward; `llama-server`'s own timer read 196.3 / 200.6 tok/s. The chat was not run again: on engine-v0.3.0's image, a chat of three turns with the prompt cache on, sent twice, gave the same answers both times. On engine-v0.2.0's own gate, engine-v0.1.0's image read 118.9 / 121.0 tok/s on that gate's prompts, the same day.

### A fault found before engine-v0.2.0's release

Before engine-v0.2.0's release candidate was built, an image built from the champion's source was run on a P100. It was as fast as the judge had measured, and it aborted on the second turn of any chat that reused the prompt cache. The released engine-v0.1.0 image answered the same chats. The hill climb's judge had sent every prompt with the prompt cache off, so no judged run reached the fault. The cause was in how the engine reported a position to the server while it held a prompt batch. The fix is 12 lines and changes no arithmetic. Every published image since has it, and the chat test above is the one that found the fault.

### Not measured yet

- **Whether this model's greedy text on engine-v0.3.0 or later is engine-v0.2.0's.** engine-v0.4.1's gate compared it with engine-v0.4.0's only: the same.
- **The prompt cache (a chat) on engine-v0.4.0 and engine-v0.4.1.**
- **Qwen3.8-27B-PN3's quality checks on this engine.** Perplexity, needle recall and the task screen were last run on the decode climb's starting point ([the engine-v0.1.0 receipt](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.1.0/Qwen3.8-27B-PN3-GGUF-p100.md)).
- **32K and 65K context on this engine**, for decode and for prefill. At those contexts only the decode climb's starting point was measured: 7.55 / 7.39 tok/s for Qwen3.8-27B-PN3, against 24.52 / 22.92 for Q3Floor.
- **Saving and restoring a slot, a context shift on a full context, and a request cancelled in the middle of a prompt.**
- **Speculative decoding** for Qwen3.8-27B-PN3.

## Qwen3.8-27B-UD-IQ3_XXS-Q3Floor

### The published image (2026-11-27)

The same image digest, measured as `0.4.1-rc1` and released as `0.4.1-sm60`, on one Tesla P100-PCIE-16GB, with exactly the recipe's switches and flags, and engine-v0.4.0's image straight after it, as for Qwen3.8-27B-PN3.

| Measure | engine-v0.4.1 | engine-v0.4.0, the same hour | engine-v0.4.0's gate (2026-11-24) |
|---|---:|---:|---:|
| Greedy replies, three prompts, 256 tokens each | 3 replies | **the same bytes** | |
| `tensorless bench` decode, median of 3 runs (256 tokens after a 429-token prompt) | **26.62 tok/s** (26.58 / 26.62 / 26.65) | 26.65 (26.65 / 26.63 / 26.66) | 26.63 |
| Against the research receipt (26.53) | +0.3% | +0.5% | +0.4% |
| Time to first token, 429-token prompt | 4.12 s (median) | 4.13 s | 4.04 s |
| Prefill, a cold prompt of code / prose (3,651 / 4,208 tokens) | **142.0 / 147.0 tok/s** | 141.9 / 146.8 | not measured; engine-v0.3.0's image: 141.9 / 147.2 |
| GPU memory, idle / generating | 13,289 / 13,869 MiB | the same | 13,289 / 13,869 |
| Load to listening | 54.4 s, page cache cold | 5.7 s, warm | 8.9 s, warm |

Against engine-v0.4.0's image, nothing changed. The chat was not run again: on engine-v0.3.0's image, the cached chat gave the same answers both times.

### Research receipt

The hill climbs' judge decoded this model in every block, as the comparison model for Qwen3.8-27B-PN3.

| Measure | Code | Prose | Scope |
|---|---:|---:|---|
| Decode, short context | 26.53 tok/s | 26.58 tok/s | 256 greedy tokens after a cold prompt of about 600 tokens, on engine-v0.2.0's research commit |
| Decode at 32K | 24.52 tok/s | 24.54 tok/s | 1,024 greedy tokens after a prompt filling the context, on the decode climb's starting commit |
| Decode at 65K | 22.92 tok/s | 22.97 tok/s | the same |

Its greedy text stayed bit-identical through both Qwen climbs, and its speed was guarded within 3%.

### Not measured yet

- Whether its greedy text on engine-v0.3.0 or later is engine-v0.2.0's. engine-v0.4.1's gate compared it with engine-v0.4.0's only: the same.
- The prompt cache (a chat) on engine-v0.4.0 and engine-v0.4.1.
- 32K and 65K context on this engine.
- This model's prefill on engine-v0.1.0, for comparison.
- Speculative decoding (MTP) on this engine. The earlier kernel campaign ran it at about 28 tok/s on its own engine version; that mode is not in this recipe.

## North-Mini-Code-1.0

CohereLabs' North-Mini-Code-1.0 in Unsloth's UD-IQ3_XXS file, unmodified. engine-v0.3.0 is the champion of an automated decode hill climb on this model, on one P100: hill-climb commit `7d8be264`, champion 23 of the run, which started from engine-v0.2.0 plus a fix to the sliding-window KV cache. Two changes were made after the import: a guard that is never taken, and a fix that lets the model load with no GPU (see [`engine/PROVENANCE.md`](https://github.com/iammrduncan/tensorless/blob/main/engine/PROVENANCE.md)). The recipe runs what the judge ran: a 36,864-token context with a q8_0 KV cache, and the same 21 engine switches as the Qwen3.8-27B recipes. It moved to engine-v0.4.0 unchanged but for `GGML_CUDA_SYNC_SCHED = "auto"`, after the release gate measured it. engine-v0.4.1 adds the champion of a prefill hill climb on this model ([below](#prefill-research-receipt-engine-v041)), and the recipe moved to it unchanged but for its `prefill_tps`, now that climb's, after the release gate measured it next to engine-v0.4.0's image ([below](#the-published-image-2026-11-27-2)).

### Speed, research receipt (engine-v0.3.0)

| Measure | engine-v0.3.0's research commit | The starting point | Scope |
|---|---:|---:|---|
| Decode after a 4,096-token prompt | **79.77 tok/s** | 49.28 | 256 greedy tokens after a cold code prompt; champion 23's confirmation on 5 fresh blocks |
| Decode after a 32,768-token prompt | **66.06 tok/s** | 35.83 | the same |
| Prefill, 4,096 / 32,768-token prompt | 446.4 / 334.7 tok/s | | the climb did not change prefill |

The climb judged 127 submissions and confirmed 22. Its decode metric, over both prompts, went from 42.03 to 72.62 tok/s: +74.25%, the product of the confirmed gains.

### What every step had to keep

| Check | Bar held |
|---|---|
| Teacher-forced output against the starting point's own: mean KL divergence | at most 0.003 nat |
| Same top token as the starting point | at 98% or more of positions |
| The judge's greedy replies | 0 changed |

Below about 8,192 tokens of context, the output is bit-identical to the starting point's. Above it, kernels chosen by context length add in another order, so the output differs, inside these bars.

The speed came from one-token attention over the q8_0 KV cache, the Q6_K attention projections (Q, K and V merged at load into one GEMV, with the router's GEMV in the same launch), the experts' IQ2_S and IQ3_S GEMVs, a fused mixture-of-experts tail, and RoPE fused with the KV cache write. All of it is on by default for this model; [Engine switches](engine-switches.md#the-north-mini-code-10-decode-path) lists what turns parts of it off.

### Prefill, research receipt (engine-v0.4.1)

engine-v0.4.1 adds the champion of an automated prefill hill climb on this model, on one P100: hill-climb commit `1bfd3d6b`, champion 7 of the run, which started from engine-v0.3.0 with the fix for a load with no GPU. Its change was then applied onto engine-v0.4.0 ([`engine/PROVENANCE.md`](https://github.com/iammrduncan/tensorless/blob/main/engine/PROVENANCE.md)). The judge timed the first token of a cold code prompt, on engine-v0.3.0's base:

| Measure | The champion, last confirmation | The starting point | Scope |
|---|---:|---:|---|
| Prefill, a 4,096-token prompt | **642.1 tok/s** | 446.5 | prompt tokens over the time to the first token |
| Prefill, a 32,768-token prompt | **434.2 tok/s** | 334.8 | the same |
| The climb's decode metric, both prompts | 72.70 tok/s | 72.84 | |

The climb judged 62 submissions and confirmed 6. Its prefill metric, the geometric mean of the two prompts, went from 386.67 to 527.96 tok/s: +36.6%. The sixth gain, a skip of fully masked key chunks in the flash-attention tile (+1.3%), is not in the last champion, so it is not in engine-v0.4.1.

| Check | Bar held |
|---|---|
| Decode, against the starting point | no more than 1% slower |
| Sampled decode, against the starting point | no more than 3% slower |
| Teacher-forced output against the starting point's own: mean KL divergence | at most 0.005 nat |
| Same top token as the starting point | at 97.5% or more of positions |
| The card's free memory | a guard |

The speed came from the MMQ path on `sm_60`, which a prompt's batched products take: a smaller tile for IQ2_S and IQ3_S, the experts' launches only for the tiles that hold tokens, and dot kernels with no branch on the column, built from 16-bit multiply-adds because the P100 has no DP4A instruction. And the many-query flash-attention tile at head size 128 may now use more registers. Each keeps the order of the arithmetic. All of it is on by default, with no switch.

### The published image (2026-11-27)

`sha256:c89e60b34a0f…`, measured as `0.4.1-rc1` and released as `0.4.1-sm60`, on one Tesla P100-PCIE-16GB, with the recipe's switches and flags, `GGML_CUDA_SYNC_SCHED=auto` included. engine-v0.4.0's image, `0.4.0-sm60`, ran straight after it on the same card with the same tests:

| Measure | engine-v0.4.1 | engine-v0.4.0, the same hour | engine-v0.4.0's gate (2026-11-24) |
|---|---:|---:|---:|
| Greedy replies, three prompts of 3,122 to 15,837 tokens | 3 replies | **the same bytes** | |
| `tensorless bench` decode, median of 3 runs (256 tokens after a 411-token prompt) | **94.87 tok/s** (94.87 / 94.88 / 94.74) | 95.03 | 94.82 |
| Time to first token, 411-token prompt | 0.972 s (median) | 1.456 s | 1.382 s |
| Decode, 256 greedy tokens after a 3,911-token code prompt, median of 3 | **80.39 tok/s** | 80.76 | 79.31 |
| Decode, 256 greedy tokens after a 31,832-token code prompt, median of 3 | **66.80 tok/s** | 67.04 | 66.23, with `blocking` |
| Prefill, a cold prompt of 3,122 tokens (code) / 3,827 tokens (prose), median of 3 | **600.3 / 598.6 tok/s** | 434.2 / 435.1 | 438.5 / 438.3, with `blocking` |
| Prefill of the 3,911- / 31,832-token prompts, `llama-server`'s own timer | **624.5 / 406.9 tok/s** | 444.8 / 333.7 | 445.7 / 333.6 |
| A chat of three turns with the prompt cache on, sent twice | answers, no abort; the two passes' answers differ | the same token counts, turn by turn | the same |
| GPU memory, idle / from the first prompt on | 13,877 / 14,431, then 15,295 MiB | 13,877 / 14,409, then 15,273 | 13,877 / 14,409, then 15,273 |
| Load to listening | 77.2 s, page cache cold | 25.8 s | 3.5 s, warm |

**Prefill is about 40% faster.** Against engine-v0.4.0's image, prefill read +38.3% / +37.6% on the client's clock for the code and prose prompts, and on the server's timer +40.4% after 3,911 tokens and +21.9% after 31,832. The climb's judge read +36.6% over engine-v0.3.0, on its own prompts. The greedy replies were byte for byte the same, so the faster path gave the same output on these prompts.

**Decode did not change:** within 0.5% of engine-v0.4.0's image in all three measures, inside the spread between repeats. Against the judge, it read +0.8% after about 4,000 tokens and +1.1% after about 32,000. `tensorless bench` reads 18.9% above the recipe's `decode_tps.short` (79.77), because its prompt is 411 tokens and the receipt's is 4,096: compare `bench` on your own card with 94.87, not with the receipt.

**GPU memory rose once in the chat, on both images.** At the first turn of the chat's second pass, used memory stepped up by 864 MiB and stayed there: to 15,295 MiB on engine-v0.4.1's image and to 15,273 MiB on engine-v0.4.0's, the figure engine-v0.4.0's gate recorded. So the step is not new in engine-v0.4.1; engine-v0.3.0's gate ran the same chat and never went above 14,409 MiB. The cause is not known, and the server did not fail: 976 MiB stayed free. engine-v0.4.1's image uses 22 MiB more than engine-v0.4.0's throughout ([receipt](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.4.1/North-Mini-Code-1.0-GGUF-p100.md#gpu-memory-at-the-chats-second-pass)).

**The prompt cache can change the answer.** In the chat above, the two passes' answers differed, as on engine-v0.3.0 and engine-v0.4.0, where a reply of the first turn was 670 tokens on one pass and 729 on the other. The same request sent twice with `cache_prompt` false gave the same text on engine-v0.3.0. The hill climb's judge found the same difference between a cached second turn and the same turn sent cold, for the champion and the candidates alike. It is a known limit of this model on this engine.

**A load with no GPU works.** On engine-v0.3.0's image, in a container with no GPU device, the model loaded on the CPU and answered a 32-token completion correctly, at 9.74 tok/s on 8 threads. Not checked again on engine-v0.4.0 or engine-v0.4.1.

### Not measured yet

- The model's quality checks: no perplexity, code-task or recall check ran in the release gate.
- The cause of the memory step above.
- The prefill climb's sixth gain together with the other five.
- The two changes after the decode climb's import, under the judge. The published image has them and reads the judge's speed.
- Contexts above 36,864 tokens, speculative decoding, and more than one request at a time.
