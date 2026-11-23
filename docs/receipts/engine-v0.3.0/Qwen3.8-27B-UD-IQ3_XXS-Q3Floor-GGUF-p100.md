# Receipt: iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF on one Tesla P100, engine-v0.3.0

What has been measured for `recipes/iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF/p100.toml`, by whom, and on what.

**Status: released.** The recipe moved to engine-v0.3.0 after the published image was measured with this model on a P100. Its `[run]` section did not change. engine-v0.3.0's new work is a decode hill climb on North-Mini-Code-1.0, so for this model the numbers below are the release gate's; the [Qwen3.8-27B-PN3 receipt](Qwen3.8-27B-PN3-GGUF-p100.md#what-changed-in-the-engine) of this release says more.

## The published image on a P100 (2026-11-23)

The image `ghcr.io/iammrduncan/tensorless-engine@sha256:3423499a1676c557341bc7e959b65845b59cd791cc7b30e36f862c4c5528bd5e` was measured as `0.3.0-rc1` and released as `0.3.0-sm60`, the same digest.
- It ran on one Tesla P100-PCIE-16GB, with exactly the recipe's arguments and switches and `OMP_NUM_THREADS=8`.
- The model file was the lab's copy of the file the recipe pins, sha256 `4b4d6a39…`.

| Measure | engine-v0.3.0 | engine-v0.2.0's gate (2026-11-12) |
|---|---:|---:|
| `tensorless bench` decode, median of 3 runs (256 tokens after a 429-token prompt) | **26.53 tok/s** (26.53 / 26.61 / 26.39) | 26.69 |
| Against the research receipt (26.53) | −0.02% | +0.6% |
| Time to first token, 429-token prompt | 4.13 s (median) | 3.86 s |
| Prefill, a cold prompt of code / prose, median of 3 | 141.9 / 147.2 tok/s (3,651 / 4,208 tokens) | 144.7 / 148.4 (3,744 / 4,657 tokens) |
| The same prefill, llama-server's own timer | 143.8 / 148.8 tok/s | |
| Decode, 256 greedy tokens after a 678- / 743-token prompt, one run each | 26.61 / 26.61 tok/s | 26.61 / 26.78, after 555 / 700 tokens |
| A chat of three turns with the prompt cache on, sent twice | answers, the same both times | the same |
| GPU memory, idle / generating | 13,289 / 13,869 MiB (2,982 / 2,402 MiB free) | 13,289 / 13,869 |
| Load to listening, page cache cold | 57.4 s | 29.8 s, partly warm |

The prompts and the client were not engine-v0.2.0's gate's, so the prefill and first-token columns are not a like-for-like comparison: the [Qwen3.8-27B-PN3 receipt](Qwen3.8-27B-PN3-GGUF-p100.md#the-published-image-on-a-p100-2026-11-23) of this release describes them.

## What the research measured

The recipe's numbers are unchanged: 26.53 tok/s at short context on engine-v0.2.0's research commit, and 24.52 / 22.92 tok/s at 32K / 65K on the decode climb's starting commit, three engine releases back. [The engine-v0.2.0 receipt](../engine-v0.2.0/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF-p100.md) has them in full.

## Not measured yet

- Whether this model's greedy text is engine-v0.2.0's. It stayed bit-identical through both Qwen hill climbs; this gate did not compare the text of the two engines.
- 32K / 65K context on this engine.
- Speculative decoding (MTP) on this engine.
