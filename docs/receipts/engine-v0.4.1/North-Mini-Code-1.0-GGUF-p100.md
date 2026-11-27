# Receipt: unsloth/North-Mini-Code-1.0-GGUF on one Tesla P100, engine-v0.4.1

What has been measured for `recipes/unsloth/North-Mini-Code-1.0-GGUF/p100.toml`, by whom, and on what. Numbers go here only with their scope.

**Status: released.** The recipe moved to engine-v0.4.1 after the published image was measured with this model on a P100, next to engine-v0.4.0's image ([below](#the-published-image-on-a-p100-2026-11-27)). Its `[run]` section did not change. Its `prefill_tps` is now the prefill hill climb's.

## What changed in the engine

engine-v0.4.1 is engine-v0.4.0 plus the champion of a prefill hill climb on this model: seven files of the sm_60 CUDA code for prompt processing, on by default, with no new switch (`engine/PROVENANCE.md`):
- the MMQ tile for IQ2_S and IQ3_S on sm_60;
- the experts' MMQ launched only for the (expert, column tile) pairs that hold tokens;
- MMQ dot kernels specialised by a warp's number of active columns, and built from 16-bit multiply-adds, since sm_60 has no DP4A instruction;
- more registers for the many-query flash-attention tile at head size 128.

Each change was made to keep the order of the arithmetic. Decode, a single token, does not take these paths.

## What the research measured

The climb started from engine-v0.3.0 with the fix for a load with no GPU. Its judge timed the first token of a cold 4,096- and 32,768-token code prompt, on a source build, and held decode within 1% of the starting point, sampled decode within 3%, the teacher-forced output against the starting point's own (mean KL at most 0.005 nat, top-1 agreement at least 97.5%), and the card's free memory.

| Measure | Champion 7, last confirmation | The starting point |
|---|---:|---:|
| Prefill, a 4,096-token prompt | **642.1 tok/s** | 446.5 |
| Prefill, a 32,768-token prompt | **434.2 tok/s** | 334.8 |
| The climb's prefill metric (geometric mean of the two) | 527.96 tok/s | 386.67 |
| The climb's decode metric | 72.70 tok/s | 72.84 |

62 submissions, 6 confirmed improvements. The sixth, a skip of fully masked key chunks in the flash-attention tile (+1.3%), is not in the last champion, so it is not in engine-v0.4.1.

The recipe's `prefill_tps.ctx_4k` is now 642.1, from this climb. Its `decode_tps` stays the decode hill climb's (79.77 after 4,096 tokens, 66.06 after 32,768), on engine-v0.3.0's hill-climb commit; [the engine-v0.3.0 receipt](../engine-v0.3.0/North-Mini-Code-1.0-GGUF-p100.md) has them, with what that climb held.

## The published image on a P100 (2026-11-27)

The image `ghcr.io/iammrduncan/tensorless-engine@sha256:c89e60b34a0f6098db20f2f2b43c78a93bb70192833cf7656b4a84ae8974346a` was built once by CI from commit `8b95e6bc`, measured as `0.4.1-rc1-sm60`, and released as `0.4.1-sm60`, the same digest.
- It ran as a cluster workload on one Tesla P100-PCIE-16GB, pulled by that digest, with the recipe's arguments and its 22 engine switches (`GGML_CUDA_SYNC_SCHED=auto` included), `-t 8 -tb 8`, `OMP_NUM_THREADS=8` and 8 CPUs. 0 restarts.
- **B** is this image, from 23:43 to 23:52 UTC on 2026-11-26. **A** is engine-v0.4.0's released image, `0.4.0-sm60` (`sha256:4a015323…`), run straight after it on the same card, from 23:53 to 00:03, with the same recipe and the same tests.
- The model file was a copy of the file the recipe pins, sha256 `029b51f9…`. The client and the prompts were engine-v0.4.0's gate's, plus a greedy check: three fixed prompts, 256 greedy tokens each, and the sha256 of each reply. `tensorless bench` was the same binary for A and B.
- Two of the host's other P100s were at 100% with another workload throughout. Nothing else ran on the gate's card.

| Measure | B: engine-v0.4.1 | A: engine-v0.4.0, the same hour | engine-v0.4.0's gate (2026-11-24) |
|---|---:|---:|---:|
| Greedy replies, 3,122- / 3,827- / 15,837-token prompts | 3 replies (256, 256 and 254 tokens) | **the same bytes**, all three | |
| `tensorless bench` decode, median of 3 runs (256 tokens after a 411-token prompt) | **94.87 tok/s** (94.87 / 94.88 / 94.74) | 95.03 (95.23 / 94.90 / 95.03) | 94.82 |
| Time to first token, 411-token prompt, median | 0.972 s | 1.456 s | 1.382 s |
| Decode, 256 greedy tokens after a 3,911-token code prompt, median of 3 | **80.39 tok/s** (80.78 / 80.39 / 80.29) | 80.76 (80.76 / 80.28 / 80.89) | 79.31 |
| Decode, 256 greedy tokens after a 31,832-token code prompt, median of 3 | **66.80 tok/s** (66.80 / 67.12 / 66.08) | 67.04 (67.07 / 67.04 / 67.03) | 66.23, with `blocking` |
| Prefill, a cold prompt of 3,122 tokens (code) / 3,827 tokens (prose), median of 3 | **600.3 / 598.6 tok/s** | 434.2 / 435.1 | 438.5 / 438.3, with `blocking` |
| The same, llama-server's own timer | 644.6 / 631.2 | 456.9 / 452.4 | 456.4 / 452.3 |
| Prefill of the 3,911- / 31,832-token prompts, llama-server's own timer, median | **624.5 / 406.9 tok/s** | 444.8 / 333.7 | 445.7 / 333.6 (the second with `blocking`) |
| A chat of three turns with the prompt cache on, sent twice | every request returned, no abort; the two passes' answers differ | the same token counts and lengths, turn by turn | the same |
| GPU memory, idle / from the first prompt on | 13,877 / 14,431 MiB, then 15,295 at the chat's second pass | 13,877 / 14,409, then 15,273 | 13,877 / 14,409, then 15,273 |
| Least memory free | 976 MiB | 998 MiB | 998 MiB |
| Load to listening | 77.2 s, run first, page cache cold | 25.8 s, after B | 3.5 s, warm |

The model takes the same buffers on the GPU on both images: 11,228.03 MiB of weights, KV caches of 497.25 MiB (13 full-attention layers) and 191.25 MiB (36 sliding-window layers), and 342.54 MiB of compute buffers.

**Prefill.** Against engine-v0.4.0's image, prefill read +38.3% / +37.6% on the client's clock for the code and prose prompts, and +41.1% / +39.5% on the server's. On the server's timer, the 3,911-token prompt read +40.4% and the 31,832-token prompt +21.9%: a geometric mean of +30.8%. The climb's judge read +36.6% on its own prompts, over engine-v0.3.0. The time to first token of `tensorless bench`'s 411-token prompt fell from 1.456 to 0.972 s.

**Decode** read within 0.5% of engine-v0.4.0's image in all three measures (−0.46% after 3,911 tokens, −0.36% after 31,832, −0.17% on the bench), inside the spread between repeats. Against the judge, it read +0.8% after about 4,000 tokens (79.77) and +1.1% after about 32,000 (66.06). `tensorless bench` reads 18.9% above the recipe's `decode_tps.short` (79.77), because its prompt is 411 tokens, not 4,096: compare `bench` on your own card with 94.87, not with the receipt.

**How it was timed** is engine-v0.4.0's gate's way ([its receipt](../engine-v0.4.0/North-Mini-Code-1.0-GGUF-p100.md)): `/completion` through a port-forward, temperature 0, `top_k` 1, `ignore_eos`, `cache_prompt` false, and the same prompts. The cold load of B, which ran first, was the file being read from disk, not the image.

### The same output

The three greedy replies were byte for byte the same on both images, and the cached chat gave the same token counts and answer lengths on both, turn by turn. So, on these prompts, engine-v0.4.1's changes to prompt processing did not change this model's output.

### GPU memory at the chat's second pass

At the first turn of the chat's second pass, as in engine-v0.4.0's gate, used memory stepped up by 864 MiB and held there to the end of the run: from 14,431 to 15,295 MiB on B, and from 14,409 to 15,273 MiB on A, engine-v0.4.0's own figure. So the step is not new in engine-v0.4.1: engine-v0.4.0's image does it too, now with `auto`. engine-v0.3.0's gate ran the same chat and never went above 14,409 MiB. The cause is not known. The server did not fail: 976 MiB stayed free on B.

B uses 22 MiB more than A throughout: 14,431 against 14,409 MiB from the first prompt on.

### The prompt cache

As on engine-v0.3.0 and engine-v0.4.0, the chat's two passes gave different answers; the [engine-v0.3.0 receipt](../engine-v0.3.0/North-Mini-Code-1.0-GGUF-p100.md#the-prompt-cache) explains it. Reusing the prompt cache can change this model's greedy output on this engine. If you need the same answer for the same request, send it with `cache_prompt` false.

## Not measured yet

- **The cause of the memory step** at the chat's second pass.
- **The climb's sixth gain** together with the other five.
- **A load with no GPU** on this engine. engine-v0.3.0's image loaded and answered on the CPU; it was not checked again on engine-v0.4.0 or engine-v0.4.1.
- **Whether this model's greedy text is engine-v0.3.0's.** This gate compared engine-v0.4.1 with engine-v0.4.0 only.
- Everything the [engine-v0.3.0 receipt](../engine-v0.3.0/North-Mini-Code-1.0-GGUF-p100.md#not-measured-yet) lists: the model's quality checks, contexts above 36,864 tokens, speculative decoding, and more than one request at a time.
