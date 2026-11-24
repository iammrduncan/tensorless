---
title: "Receipts"
description: "What was measured for each engine-v0.4.0 recipe, by whom and on what: speed, the checks every step had to pass, and the published image's own run."
type: reference
sidebar:
  order: 7
---

A receipt is what the release gate measured for a recipe: speed at stated contexts, quality checks, and the image digest and model hash they were taken on. Each number below comes with its scope. The source files are in [`docs/receipts/engine-v0.4.0/`](https://github.com/iammrduncan/tensorless/tree/main/docs/receipts/engine-v0.4.0). The receipts of the earlier releases are in [`docs/receipts/engine-v0.3.0/`](https://github.com/iammrduncan/tensorless/tree/main/docs/receipts/engine-v0.3.0), [`docs/receipts/engine-v0.2.0/`](https://github.com/iammrduncan/tensorless/tree/main/docs/receipts/engine-v0.2.0) and [`docs/receipts/engine-v0.1.0/`](https://github.com/iammrduncan/tensorless/tree/main/docs/receipts/engine-v0.1.0).

| Recipe | Receipt file |
|---|---|
| `iamMrDuncan/Qwen3.8-27B-PN3-GGUF` on `p100` | [Qwen3.8-27B-PN3-GGUF-p100.md](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.4.0/Qwen3.8-27B-PN3-GGUF-p100.md), with its research receipt in [engine-v0.2.0's](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.2.0/Qwen3.8-27B-PN3-GGUF-p100.md) |
| `iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF` on `p100` | [Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF-p100.md](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.4.0/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF-p100.md) |
| `unsloth/North-Mini-Code-1.0-GGUF` on `p100` | [North-Mini-Code-1.0-GGUF-p100.md](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.4.0/North-Mini-Code-1.0-GGUF-p100.md), with its research receipt in [engine-v0.3.0's](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.3.0/North-Mini-Code-1.0-GGUF-p100.md) |

The draft recipe of this release, `shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-12.8GB` on `p100`, has a receipt draft, [Inkling-2-Mini-LookAtMyQuant-12.8GB-p100.md](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.4.0/Inkling-2-Mini-LookAtMyQuant-12.8GB-p100.md): the research build's numbers only, since the published image has not been measured with it yet. See [Launch targets](launch-targets.md#draft-in-v040-inkling-2-mini).

## Two kinds of measurement

| Kind | Who measured | On what | How |
|---|---|---|---|
| Research receipt | A hill climb's judge | A source build of the engine's research commit, with the same CUDA configuration as the image, on one P100 | Timed from outside the engine: the request and each streamed token are stamped on the judge's own clock. Greedy decoding. |
| Published image | `tensorless bench`, and prefill and decode timings by the same method | The released image digest, with exactly the recipe's switches and flags, on one Tesla P100-PCIE-16GB | Decode: the median of 3 runs of 256 greedy tokens after a 429-token prompt (411 for North-Mini-Code-1.0). See [Command line](cli.md#bench). Prefill: one cold prompt, its tokens over the time from the request to the first token, the median of 3. |

The two use different prompts, so a few percent of difference between them is expected.

## The host sync schedule

engine-v0.4.0 adds a setting for how the host thread waits for the GPU in a CUDA synchronize, `GGML_CUDA_SYNC_SCHED` (see [Engine switches](engine-switches.md#the-host-sync-schedule)). Its default is `blocking`: the thread sleeps until the GPU signals. engine-v0.3.0 and earlier always used the CUDA driver's default, `auto`. The release gate measured each launch target both ways:

| Launch target | `auto` | `blocking` | engine-v0.3.0 |
|---|---:|---:|---:|
| Qwen3.8-27B-PN3, `tensorless bench`, two benches | **30.46 / 30.46 tok/s** | 30.20 / 30.13 | 30.45 |
| Qwen3.8-27B-UD-IQ3_XXS-Q3Floor, the same | **26.63 / 26.61 tok/s** | 26.56 / 26.61 | 26.53 |
| North-Mini-Code-1.0, `tensorless bench` | **94.82 tok/s** | 93.57 | 93.70 |
| North-Mini-Code-1.0, decode after a 3,911-token prompt | **79.31 tok/s** | 79.51 | 79.94 |

`blocking` cost Qwen3.8-27B-PN3 about 1%, and made no difference beyond the runs' spread for the other two. So the three released recipes set `GGML_CUDA_SYNC_SCHED = "auto"`, which is how engine-v0.3.0 ran them, and the numbers below are the `auto` runs' unless they say otherwise. During the Qwen models' `blocking` runs, the host's two other P100s were busy at 100%; the gate's notes do not say whether they were during the `auto` runs.

## Qwen3.8-27B-PN3

This model's research numbers come from engine-v0.2.0, the champion of an automated prefill hill climb that started from engine-v0.1.0: research commit `df673571`, exported unchanged except for six comment lines, plus one fix made afterwards. engine-v0.3.0 added a decode climb on [another model](#north-mini-code-10) and engine-v0.4.0 adds Inkling-2-Mini, and this recipe moved to each after the release gate measured it, unchanged but for engine-v0.4.0's `GGML_CUDA_SYNC_SCHED = "auto"` ([below](#the-published-image-2026-11-24)). The recipe's flags and its 21 engine switches are the ones engine-v0.1.0 ran with.

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

### The published image (2026-11-24)

`ghcr.io/iammrduncan/tensorless-engine@sha256:4a0153230be9e2b5957a5aaf58cbb0ba1f79bfeb94eacd4617189f16ca35e997`, measured as `0.4.0-rc1` and released as `0.4.0-sm60`, on one Tesla P100-PCIE-16GB, with exactly the recipe's switches and flags. This gate ran `tensorless bench` only, twice with `auto` and twice with `blocking`.

| Measure | engine-v0.4.0, `auto` | engine-v0.3.0's gate (2026-11-23) |
|---|---:|---:|
| `tensorless bench` decode, median of 3 runs (256 tokens after a 429-token prompt) | **30.46 tok/s** (30.46 / 30.51 / 30.44), and 30.46 again | 30.45 (30.32 / 30.45 / 30.46) |
| Against the research receipt (30.43) | +0.1% | +0.1% |
| Time to first token, 429-token prompt | 2.81 s (median) | 2.81 s; engine-v0.1.0's image: 4.46 s |
| GPU memory, idle / after the first prompt | 14,139 / 15,381 MiB | 14,139 / 15,381 |
| Load to listening | 8.4 s, page cache warm (53.7 s cold, in the `blocking` run) | 60.3 s, page cache cold |

Prefill and the chat were not measured again. On engine-v0.3.0's image, a cold prompt of code / prose read **194.6 / 199.2 tok/s** (3,651 / 4,208 tokens, timed from the client through a port-forward; `llama-server`'s own timer read 197.8 / 202.1), and a chat of three turns with the prompt cache on, sent twice, gave the same answers both times. On engine-v0.2.0's own gate, engine-v0.1.0's image read 118.9 / 121.0 tok/s on that gate's prompts, the same day.

### A fault found before engine-v0.2.0's release

Before engine-v0.2.0's release candidate was built, an image built from the champion's source was run on a P100. It was as fast as the judge had measured, and it aborted on the second turn of any chat that reused the prompt cache. The released engine-v0.1.0 image answered the same chats. The hill climb's judge had sent every prompt with the prompt cache off, so no judged run reached the fault. The cause was in how the engine reported a position to the server while it held a prompt batch. The fix is 12 lines and changes no arithmetic. Every published image since has it, and the chat test above is the one that found the fault.

### Not measured yet

- **Whether this model's greedy text on engine-v0.4.0 or engine-v0.3.0 is engine-v0.2.0's.** Neither gate compared it.
- **Prefill and the prompt cache on engine-v0.4.0.** Its gate ran `tensorless bench` only.
- **Qwen3.8-27B-PN3's quality checks on this engine.** Perplexity, needle recall and the task screen were last run on the decode climb's starting point ([the engine-v0.1.0 receipt](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.1.0/Qwen3.8-27B-PN3-GGUF-p100.md)).
- **32K and 65K context on this engine**, for decode and for prefill. At those contexts only the decode climb's starting point was measured: 7.55 / 7.39 tok/s for Qwen3.8-27B-PN3, against 24.52 / 22.92 for Q3Floor.
- **Saving and restoring a slot, a context shift on a full context, and a request cancelled in the middle of a prompt.**
- **Speculative decoding** for Qwen3.8-27B-PN3.

## Qwen3.8-27B-UD-IQ3_XXS-Q3Floor

### The published image (2026-11-24)

The same image digest, measured as `0.4.0-rc1` and released as `0.4.0-sm60`, on one Tesla P100-PCIE-16GB, with exactly the recipe's switches and flags. `tensorless bench` only, as for Qwen3.8-27B-PN3.

| Measure | engine-v0.4.0, `auto` | engine-v0.3.0's gate (2026-11-23) |
|---|---:|---:|
| `tensorless bench` decode, median of 3 runs (256 tokens after a 429-token prompt) | **26.63 tok/s** (26.62 / 26.63 / 26.63), and 26.61 | 26.53 (26.53 / 26.61 / 26.39) |
| Against the research receipt (26.53) | +0.4% | −0.02% |
| Time to first token, 429-token prompt | 4.04 s (median) | 4.13 s |
| GPU memory, idle / generating | 13,289 / 13,869 MiB | 13,289 / 13,869 |
| Load to listening | 8.9 s, page cache warm (49.4 s cold, in the `blocking` run) | 57.4 s, page cache cold |

On engine-v0.3.0's image, prefill read 141.9 / 147.2 tok/s on a cold prompt of code / prose (3,651 / 4,208 tokens), and the cached chat gave the same answers both times.

### Research receipt

The hill climbs' judge decoded this model in every block, as the comparison model for Qwen3.8-27B-PN3.

| Measure | Code | Prose | Scope |
|---|---:|---:|---|
| Decode, short context | 26.53 tok/s | 26.58 tok/s | 256 greedy tokens after a cold prompt of about 600 tokens, on engine-v0.2.0's research commit |
| Decode at 32K | 24.52 tok/s | 24.54 tok/s | 1,024 greedy tokens after a prompt filling the context, on the decode climb's starting commit |
| Decode at 65K | 22.92 tok/s | 22.97 tok/s | the same |

Its greedy text stayed bit-identical through both Qwen climbs, and its speed was guarded within 3%.

### Not measured yet

- Whether its greedy text on engine-v0.4.0 or engine-v0.3.0 is engine-v0.2.0's.
- Prefill and the prompt cache on engine-v0.4.0.
- 32K and 65K context on this engine.
- This model's prefill on engine-v0.1.0, for comparison.
- Speculative decoding (MTP) on this engine. The earlier kernel campaign ran it at about 28 tok/s on its own engine version; that mode is not in this recipe.

## North-Mini-Code-1.0

CohereLabs' North-Mini-Code-1.0 in Unsloth's UD-IQ3_XXS file, unmodified. engine-v0.3.0 is the champion of an automated decode hill climb on this model, on one P100: hill-climb commit `7d8be264`, champion 23 of the run, which started from engine-v0.2.0 plus a fix to the sliding-window KV cache. Two changes were made after the import: a guard that is never taken, and a fix that lets the model load with no GPU (see [`engine/PROVENANCE.md`](https://github.com/iammrduncan/tensorless/blob/main/engine/PROVENANCE.md)). The recipe runs what the judge ran: a 36,864-token context with a q8_0 KV cache, and the same 21 engine switches as the Qwen3.8-27B recipes. It moved to engine-v0.4.0 unchanged but for `GGML_CUDA_SYNC_SCHED = "auto"`, after the release gate measured it ([below](#the-published-image-2026-11-24-2)).

### Speed, research receipt

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

### The published image (2026-11-24)

`sha256:4a0153230be9…`, measured as `0.4.0-rc1` and released as `0.4.0-sm60`, on one Tesla P100-PCIE-16GB, with the recipe's switches and flags. Run 1 was the full gate with the engine's default sync schedule, `blocking`; run 1b, with the recipe's `auto`, measured decode after 3,911 tokens and `tensorless bench`:

| Measure | Run 1b, `auto` | Run 1, `blocking` | engine-v0.3.0's gate |
|---|---:|---:|---:|
| `tensorless bench` decode, median of 3 runs (256 tokens after a 411-token prompt) | **94.82 tok/s** (95.15 / 93.05 / 94.82) | 93.57 | 93.70 |
| Time to first token, 411-token prompt | 1.382 s (median) | 1.407 s | 1.415 s |
| Decode, 256 greedy tokens after a 3,911-token code prompt, median of 3 | **79.31 tok/s** | 79.51 | 79.94 |
| Decode, 256 greedy tokens after a 31,832-token code prompt, median of 3 | | **66.23 tok/s** | 66.63 |
| Prefill, a cold prompt of 3,122 tokens (code) / 3,827 tokens (prose), median of 3 | | **438.5 / 438.3 tok/s** | 441.6 / 439.3 |
| A chat of three turns with the prompt cache on, sent twice | | answers, no abort; the two passes' answers differ | the same |
| GPU memory, idle / from the first prompt on | 13,877 / 14,409 MiB | 13,877 / 14,409, then 15,273 MiB | 13,877 / 14,409 |
| Load to listening | 3.5 s, page cache warm | 58.1 s, page cache cold | 3.79 s, warm |

Against the judge, decode read −0.6% after about 4,000 tokens (run 1b) and +0.3% after about 32,000 (run 1). `tensorless bench` reads 18.9% above the recipe's `decode_tps.short` (79.77), because its prompt is 411 tokens and the receipt's is 4,096: compare `bench` on your own card with 94.82, not with the receipt.

**GPU memory rose once, for a reason not yet known.** In run 1, used memory stepped from 14,409 to 15,273 MiB at the first turn of the chat's second pass, where the server took a cached prompt 0.992 similar to the request and evaluated 6 new tokens, and stayed there. 998 MiB stayed free, and the server did not fail. engine-v0.3.0's gate ran the same chat and never went above 14,409 MiB. Run 1b had no chat, so the step was not reproduced. Apart from the host sync schedule, the engine's changes since engine-v0.3.0 outside Inkling-2-Mini's own code take effect only for its architecture or with the indexed bias.

**The prompt cache can change the answer.** In the chat above, the two passes' answers differed, as on engine-v0.3.0, where a reply of the first turn was 670 tokens on one pass and 729 on the other. The same request sent twice with `cache_prompt` false gave the same text on engine-v0.3.0. The hill climb's judge found the same difference between a cached second turn and the same turn sent cold, for the champion and the candidates alike. It is a known limit of this model on this engine.

**A load with no GPU works.** On engine-v0.3.0's image, in a container with no GPU device, the model loaded on the CPU and answered a 32-token completion correctly, at 9.74 tok/s on 8 threads. Not checked again on engine-v0.4.0.

### Not measured yet

- The model's quality checks: no perplexity, code-task or recall check ran in the release gate.
- With `auto`, decode after 32K, prefill and the chat; and the memory step above, a second time.
- The two changes after the import, under the judge. The published image has them and reads the judge's speed.
- Contexts above 36,864 tokens, speculative decoding, and more than one request at a time.
