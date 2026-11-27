# Receipt: iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF on one Tesla P100, engine-v0.4.1

What has been measured for `recipes/iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF/p100.toml`, by whom, and on what.

**Status: released.** The recipe moved to engine-v0.4.1 after the published image was measured with this model on a P100, next to engine-v0.4.0's image. Its `[run]` section did not change. engine-v0.4.1's new work is the North-Mini-Code prefill climb's, in the sm_60 code for prompt processing. This model's weights are standard llama.cpp IQ types, which can take the same MMQ path, so the gate checked its output, prefill, bench and memory; the [North-Mini-Code-1.0 receipt](North-Mini-Code-1.0-GGUF-p100.md#what-changed-in-the-engine) of this release says what changed.

## The published image on a P100 (2026-11-27)

The image `ghcr.io/iammrduncan/tensorless-engine@sha256:c89e60b34a0f6098db20f2f2b43c78a93bb70192833cf7656b4a84ae8974346a` was measured as `0.4.1-rc1-sm60` and released as `0.4.1-sm60`, the same digest.
- It ran on one Tesla P100-PCIE-16GB, with the recipe's arguments and its 22 engine switches (`GGML_CUDA_SYNC_SCHED=auto` included), `-t 8 -tb 8` and `OMP_NUM_THREADS=8`. 0 restarts.
- **B** is this image, from 00:19 to 00:26 UTC on 2026-11-27. **A** is engine-v0.4.0's released image, `0.4.0-sm60` (`sha256:4a015323…`), run straight after it on the same card, from 00:28 to 00:35, with the same tests.
- The model file was a copy of the file the recipe pins, sha256 `4b4d6a39…`.
- Two of the host's other P100s were at 100% with another workload throughout. Nothing else ran on the gate's card.

| Measure | B: engine-v0.4.1 | A: engine-v0.4.0, the same hour | engine-v0.4.0's gate (2026-11-24) |
|---|---:|---:|---:|
| Greedy replies, 3,651- / 4,208- / 15,382-token prompts, 256 tokens each | 3 replies | **the same bytes**, all three | |
| `tensorless bench` decode, median of 3 runs (256 tokens after a 429-token prompt) | **26.62 tok/s** (26.58 / 26.62 / 26.65) | 26.65 (26.65 / 26.63 / 26.66) | 26.63 |
| Against the research receipt (26.53) | +0.3% | +0.5% | +0.4% |
| Time to first token, 429-token prompt, median | 4.12 s | 4.13 s | 4.04 s |
| Prefill, a cold prompt of 3,651 tokens (code) / 4,208 tokens (prose), median of 3 | **142.0 / 147.0 tok/s** | 141.9 / 146.8 | not measured; engine-v0.3.0's image: 141.9 / 147.2 |
| The same, llama-server's own timer | 143.9 / 148.7 | 143.7 / 148.7 | |
| GPU memory, idle / generating | 13,289 / 13,869 MiB (2,402 MiB free) | the same | 13,289 / 13,869 |
| Load to listening | 54.4 s, run first, page cache cold | 5.7 s, after B | 8.9 s, warm |

The model takes the same buffers on both images: 11,023.55 MiB of weights, a KV cache of 1,152.00 MiB and 520.04 MiB of compute buffers.

Against engine-v0.4.0's image, nothing changed: the same greedy text, prefill within 0.2%, the bench −0.1%, and the same memory.

## What the research measured

The recipe's numbers are unchanged: 26.53 tok/s at short context on engine-v0.2.0's research commit, and 24.52 / 22.92 tok/s at 32K / 65K on the decode climb's starting commit, five engine releases back. [The engine-v0.2.0 receipt](../engine-v0.2.0/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF-p100.md) has them in full.

## Not measured yet

- The chat with the prompt cache, with `auto`. The last chat on a published image was engine-v0.3.0's, which gave the same answers both times.
- 32K / 65K context on this engine, for decode and for prefill.
- Whether this model's greedy text is engine-v0.2.0's. It stayed bit-identical through both Qwen hill climbs; this gate compared engine-v0.4.1 with engine-v0.4.0 only.
- The model's quality checks on this engine.
- Speculative decoding (MTP) on this engine.
