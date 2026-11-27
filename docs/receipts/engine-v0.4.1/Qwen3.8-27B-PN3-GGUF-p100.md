# Receipt: iamMrDuncan/Qwen3.8-27B-PN3-GGUF on one Tesla P100, engine-v0.4.1

What has been measured for `recipes/iamMrDuncan/Qwen3.8-27B-PN3-GGUF/p100.toml`, by whom, and on what. Numbers go here only with their scope.

**Status: released.** The recipe moved to engine-v0.4.1 after the published image was measured with this model on a P100, next to engine-v0.4.0's image ([below](#the-published-image-on-a-p100-2026-11-27)). Its `[run]` section did not change.

## What changed in the engine

engine-v0.4.1 adds the champion of a prefill hill climb on North-Mini-Code-1.0 ([its receipt](North-Mini-Code-1.0-GGUF-p100.md#what-changed-in-the-engine)): changes to the sm_60 CUDA code for prompt processing, on by default. Some of them, in the MMQ path, can run for any model whose weights take that path, so the gate checked this model too: its output, its prefill, its bench and its memory, against engine-v0.4.0's image.

The research numbers in the recipe are still engine-v0.2.0's: the prefill hill climb's judge, on a source build of its research commit (decode 30.43 tok/s, prefill 200.1 tok/s). [The engine-v0.2.0 receipt](../engine-v0.2.0/Qwen3.8-27B-PN3-GGUF-p100.md) has them, with what that climb held.

## The published image on a P100 (2026-11-27)

The image `ghcr.io/iammrduncan/tensorless-engine@sha256:c89e60b34a0f6098db20f2f2b43c78a93bb70192833cf7656b4a84ae8974346a` was built once by CI from commit `8b95e6bc`, measured as `0.4.1-rc1-sm60`, and released as `0.4.1-sm60`, the same digest.
- It ran as a cluster workload on one Tesla P100-PCIE-16GB, pulled by that digest, with the recipe's arguments and its 22 engine switches (`GGML_CUDA_SYNC_SCHED=auto` included), `-t 8 -tb 8`, `OMP_NUM_THREADS=8` and 8 CPUs. 0 restarts.
- **B** is this image, from 00:05 to 00:11 UTC on 2026-11-27. **A** is engine-v0.4.0's released image, `0.4.0-sm60` (`sha256:4a015323…`), run straight after it on the same card, from 00:12 to 00:17, with the same tests. `tensorless bench` was the same binary for both.
- The model file was a copy of the file the recipe pins, sha256 `15ce2d44…`.
- Two of the host's other P100s were at 100% with another workload throughout. Nothing else ran on the gate's card.

| Measure | B: engine-v0.4.1 | A: engine-v0.4.0, the same hour | engine-v0.4.0's gate (2026-11-24) |
|---|---:|---:|---:|
| Greedy replies, 3,651- / 4,208- / 15,382-token prompts, 256 tokens each | 3 replies | **the same bytes**, all three | |
| `tensorless bench` decode, median of 3 runs (256 tokens after a 429-token prompt) | **30.30 tok/s** (30.29 / 30.30 / 30.42) | 30.17 (30.14 / 30.17 / 30.18) | 30.46 |
| Against the research receipt (30.43) | −0.4% | −0.9% | +0.1% |
| Time to first token, 429-token prompt, median | 2.89 s | 2.88 s | 2.81 s |
| Prefill, a cold prompt of 3,651 tokens (code) / 4,208 tokens (prose), median of 3 | **192.8 / 197.5 tok/s** | 192.4 / 197.4 | not measured; engine-v0.3.0's image: 194.6 / 199.2 |
| The same, llama-server's own timer | 196.3 / 200.6 | 196.0 / 200.5 | engine-v0.3.0's image: 197.8 / 202.1 |
| GPU memory, idle / after the first prompt | 14,139 / 15,381 MiB, 15,383 at most (888 MiB free) | the same | 14,139 / 15,381 |
| Load to listening | 54.4 s, run first, page cache cold | 3.8 s, after B | 8.4 s, warm |

The model takes the same buffers on both images: 12,008.66 MiB of weights, a KV cache of 1,152.00 MiB and 540.04 MiB of compute buffers.

**Against engine-v0.4.0's image, nothing changed:** the same greedy text, prefill within 0.2%, the bench +0.4%, and the same memory.

**The bench reads about 1% below engine-v0.4.0's gate,** 30.30 against 30.46. engine-v0.4.0's own image read 30.17 in the same hour on the same card, so the difference comes from the card or the hour, not from engine-v0.4.1. engine-v0.4.0's gate ran this model on another of the host's P100s.

**How it was timed** is engine-v0.4.0's gate's way ([its receipt](../engine-v0.4.0/Qwen3.8-27B-PN3-GGUF-p100.md)), and prefill is engine-v0.3.0's gate's ([its receipt](../engine-v0.3.0/Qwen3.8-27B-PN3-GGUF-p100.md)): `/completion` through a port-forward, temperature 0, `cache_prompt` false. The cold load of B, which ran first, was the file being read from disk, not the image.

## Not measured yet

- **The chat with the prompt cache, with `auto`.** The last chat on a published image was engine-v0.3.0's, which gave the same answers both times.
- **Decode after longer prompts:** 32K and 65K context on this engine, for decode and for prefill.
- **Whether this model's greedy text is engine-v0.2.0's.** This gate compared engine-v0.4.1 with engine-v0.4.0 only.
- Everything the [engine-v0.2.0 receipt](../engine-v0.2.0/Qwen3.8-27B-PN3-GGUF-p100.md#not-measured-yet) lists: the model's quality checks on this engine, and speculative decoding (MTP).
