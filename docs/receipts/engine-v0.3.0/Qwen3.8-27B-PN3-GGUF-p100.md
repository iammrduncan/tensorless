# Receipt: iamMrDuncan/Qwen3.8-27B-PN3-GGUF on one Tesla P100, engine-v0.3.0

What has been measured for `recipes/iamMrDuncan/Qwen3.8-27B-PN3-GGUF/p100.toml`, by whom, and on what. Numbers go here only with their scope.

**Status: released.** The recipe moved to engine-v0.3.0 after the published image was measured with this model on a P100 ([below](#the-published-image-on-a-p100-2026-11-23)). Its `[run]` section did not change.

## What changed in the engine

engine-v0.3.0 adds the champion of a decode hill climb on another model, North-Mini-Code-1.0 ([its receipt](North-Mini-Code-1.0-GGUF-p100.md)). Most of that work is on paths this model does not take: attention over a q8_0 KV cache at head size 128, mixture-of-experts kernels, and North-Mini-Code's own graph. The climb's judge measured North-Mini-Code-1.0, so for this model the numbers below are the release gate's.

The research numbers in the recipe are still engine-v0.2.0's: the prefill hill climb's judge, on a source build of its research commit (decode 30.43 tok/s, prefill 200.1 tok/s). [The engine-v0.2.0 receipt](../engine-v0.2.0/Qwen3.8-27B-PN3-GGUF-p100.md) has them, with what that climb held.

## The published image on a P100 (2026-11-23)

The image `ghcr.io/iammrduncan/tensorless-engine@sha256:3423499a1676c557341bc7e959b65845b59cd791cc7b30e36f862c4c5528bd5e` was measured as `0.3.0-rc1` and released as `0.3.0-sm60`, the same digest.
- It ran as a cluster workload on the lab's P100 node, pulled by that digest, with exactly the recipe's arguments and switches and `OMP_NUM_THREADS=8`.
- The model file was the lab's copy of the file the recipe pins, sha256 `15ce2d44…`.
- The card was a Tesla P100-PCIE-16GB. It reported compute capability 6.0 and 16,269 MiB.

| Measure | engine-v0.3.0 | engine-v0.2.0's gate (2026-11-12) |
|---|---:|---:|
| `tensorless bench` decode, median of 3 runs (256 tokens after a 429-token prompt) | **30.45 tok/s** (30.32 / 30.45 / 30.46) | 30.48 |
| Against the research receipt (30.43) | +0.1% | +0.2% |
| Time to first token, 429-token prompt | 2.81 s (median) | 2.60 s |
| Prefill, a cold prompt of code / prose, median of 3 | **194.6 / 199.2 tok/s** (3,651 / 4,208 tokens) | 198.9 / 201.5 (3,744 / 4,657 tokens) |
| The same prefill, llama-server's own timer | 197.8 / 202.1 tok/s | |
| Decode, 256 greedy tokens after a 678- / 743-token prompt, one run each | 30.38 / 30.41 tok/s | 30.52 / 30.75, after 555 / 700 tokens |
| A chat of three turns with the prompt cache on, sent twice | answers, the same both times | the same |
| GPU memory, idle / after the first prompt | 14,139 / 15,381 MiB (2,132 / 890 MiB free) | 14,139 / 15,383 |
| Load to listening, page cache cold | 60.3 s | 26.0 s, partly warm |

The two gates used different prompts and different clients, so the prefill and first-token columns are not a like-for-like comparison. Prefill and the time to first token are timed from the client, through a port-forward, and include the network; llama-server's own timer on the same prompts read about 1.5% higher. Decode is timed between streamed tokens, which the network does not change. This gate's prompts are the start of two texts in this repository, plus an instruction: `engine/src/llama-model.cpp` at `6ca26460` and `engine/docs/llamampere-v0.3/ARTICLE.md`, the first 13,500 characters of each for prefill and the first 2,400 for decode.

## Not measured yet

- **Whether this model's greedy text is engine-v0.2.0's.** The gate did not compare the text of the two engines.
- Everything the [engine-v0.2.0 receipt](../engine-v0.2.0/Qwen3.8-27B-PN3-GGUF-p100.md#not-measured-yet) lists: the model's quality checks on this engine, prefill and decode at 32K and 65K, and speculative decoding.
