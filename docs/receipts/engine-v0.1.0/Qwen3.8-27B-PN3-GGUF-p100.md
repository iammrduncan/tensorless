# Receipt: iamMrDuncan/Qwen3.8-27B-PN3-GGUF on one Tesla P100, engine-v0.1.0

What has been measured for `recipes/iamMrDuncan/Qwen3.8-27B-PN3-GGUF/p100.toml`, by whom, and on what. Numbers go here only with their scope.

## What was measured

The engine source of v0.1.0 is the champion of an automated decode hill climb: research commit `cfcab96e`, exported unchanged except for six comment lines (see `engine/PROVENANCE.md`). The climb's judge built that source with the same CUDA configuration as the image and timed it on one P100, from outside the engine (the driver stamps each streamed token on its own clock):

| Measure | Qwen3.8-27B-PN3 | Qwen3.8-27B-UD-IQ3_XXS-Q3Floor, same card, same cells | Scope |
|---|---:|---:|---|
| Decode, code prompt | **30.06 tok/s** | 26.30 | 256 greedy tokens after a cold prompt of about 600 tokens; the median of 3 confirmation blocks |
| Decode, prose prompt | **30.18 tok/s** | 26.31 | the same, prose prompt |
| Decode, four days later on another card | 29.95 / 29.96 tok/s | 26.26 / 26.27 | the same cells, the baseline of the next campaign |
| Prefill, code / prose | 121.4 / 119.6 tok/s | — | a cold prompt of about 4,100 tokens, 8 ubatches of 512 |
| Decode at the start of the climb | 7.74 / 7.73 tok/s | 26.30 / 26.31 | the same cells: 3.89× over the climb |

Every step of the climb kept Qwen3.8-27B-PN3's numerics where they started: KL to the starting point's outputs within 2.593e-3 on 128 forced prose tokens, the same top token at 95% or more of positions, and decode-path perplexity within 0.25%. Qwen3.8-27B-UD-IQ3_XXS-Q3Floor's greedy text stayed bit-identical, and the PN golden blocks and one real tensor of each PN class decoded bit-exactly on CPU and GPU.

## Qwen3.8-27B-PN3's quality, measured on the climb's starting point

The decode climb held quality at its starting point. That starting point was gated against Qwen3.8-27B-UD-IQ3_XXS-Q3Floor on one P100, with each model's file and these bars:

| Check | Qwen3.8-27B-PN3 | Q3Floor | Bar |
|---|---:|---:|---|
| Perplexity, ctx 2,048 (6 chunks) | 2.5494 | 2.5682 | ≤ 1.005 × Q3Floor: passes (0.993×) |
| Decode-path perplexity, code at 32K (512 forced tokens) | 1.38147 | 1.37773 | ≤ 1.005 × Q3Floor: passes (1.003×) |
| Decode-path perplexity, prose at 32K | 6.07793 | 5.65357 | ≤ 1.005 × Q3Floor: **misses (1.075×)**, accepted |
| Needle recall at 1K–31K (32K context) and 1K–63K (65K context) | 8 of 8 | exact | exact |
| Task screen (26 cases) | 23 | 23 | at most 1 fewer than Q3Floor |

The prose miss belongs to the quantization: the format's own fp32 reference path scores 6.07572, so the fast kernels account for 0.5% of the gap. It was accepted.

## Not measured yet

- **The published image itself.** CI builds `ghcr.io/iammrduncan/tensorless-engine:0.1.0-sm60` from the same source and CUDA flags, with a portable CPU baseline instead of the builder's CPU. Its first run on a P100 should land near the decode numbers above. Run `tensorless bench iamMrDuncan/Qwen3.8-27B-PN3-GGUF` and record the result here.
- **32K and 65K context for this engine.** At those contexts only the climb's starting point was measured: 7.55 / 7.39 tok/s for Qwen3.8-27B-PN3 against 24.52 / 22.92 for Qwen3.8-27B-UD-IQ3_XXS-Q3Floor. Attention costs a larger share of a 30 tok/s token than of a 7.5 tok/s one, so expect this engine's long-context decode to be below 30.
- **Speculative decoding** for Qwen3.8-27B-PN3.
