---
title: "Receipts"
description: "What was measured for each engine-v0.1.0 recipe, by whom and on what: speed, quality checks and the published image's own run."
type: reference
sidebar:
  order: 7
---

A receipt is what the release gate measured for a recipe: speed at stated contexts, quality checks, and the image digest and model hash they were taken on. Each number below comes with its scope. The source files are in [`docs/receipts/engine-v0.1.0/`](https://github.com/iammrduncan/tensorless/tree/main/docs/receipts/engine-v0.1.0).

| Recipe | Receipt file |
|---|---|
| `iamMrDuncan/Qwen3.8-27B-PN3-GGUF` on `p100` | [Qwen3.8-27B-PN3-GGUF-p100.md](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.1.0/Qwen3.8-27B-PN3-GGUF-p100.md) |
| `iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF` on `p100` | [Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF-p100.md](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.1.0/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF-p100.md) |

## Two kinds of measurement

| Kind | Who measured | On what | How |
|---|---|---|---|
| Research receipt | The decode hill climb's judge | A source build of the engine commit, with the same CUDA configuration as the image, on one P100 | Timed from outside the engine: each streamed token is stamped on the judge's own clock. Greedy decoding. |
| Published image | `tensorless bench` | The released image digest, with exactly the recipe's switches and flags, on one Tesla P100-PCIE-16GB | The median of 3 runs of 256 greedy tokens after a 429-token prompt. See [Command line](cli.md#bench). |

The two use different prompts, so a few percent of difference between them is expected.

## Qwen3.8-27B-PN3

The engine source of v0.1.0 is the champion of an automated decode hill climb: research commit `cfcab96e`, exported unchanged except for six comment lines (see [`engine/PROVENANCE.md`](https://github.com/iammrduncan/tensorless/blob/main/engine/PROVENANCE.md)).

### Speed, research receipt

| Measure | Qwen3.8-27B-PN3 | Q3Floor, same card, same cells | Scope |
|---|---:|---:|---|
| Decode, code prompt | **30.06 tok/s** | 26.30 | 256 greedy tokens after a cold prompt of about 600 tokens; the median of 3 confirmation blocks |
| Decode, prose prompt | **30.18 tok/s** | 26.31 | the same, prose prompt |
| Decode, four days later on another card | 29.95 / 29.96 tok/s | 26.26 / 26.27 | the same cells, the baseline of the next campaign |
| Prefill, code / prose | 121.4 / 119.6 tok/s | — | a cold prompt of about 4,100 tokens, 8 micro-batches of 512 |
| Decode at the start of the climb | 7.74 / 7.73 tok/s | 26.30 / 26.31 | the same cells: 3.89× over the climb |

### Numerics held during the climb

Every step of the climb kept Qwen3.8-27B-PN3's numerics where they started:

| Check | Bar held |
|---|---|
| KL divergence to the starting point's outputs, 128 forced prose tokens | within 2.593e-3 |
| Same top token as the starting point | at 95% or more of positions |
| Decode-path perplexity | within 0.25% |
| Q3Floor's greedy text | bit-identical |
| PN golden blocks, and one real tensor of each PN class | decoded bit-exactly on CPU and GPU |

### Quality, measured on the climb's starting point

The climb held quality at its starting point. That starting point was gated against Q3Floor on one P100, each model with its own file:

| Check | Qwen3.8-27B-PN3 | Q3Floor | Bar |
|---|---:|---:|---|
| Perplexity, context 2,048 (6 chunks) | 2.5494 | 2.5682 | ≤ 1.005 × Q3Floor: passes (0.993×) |
| Decode-path perplexity, code at 32K (512 forced tokens) | 1.38147 | 1.37773 | ≤ 1.005 × Q3Floor: passes (1.003×) |
| Decode-path perplexity, prose at 32K | 6.07793 | 5.65357 | ≤ 1.005 × Q3Floor: **misses (1.075×)**, accepted |
| Needle recall at 1K–31K (32K context) and 1K–63K (65K context) | 8 of 8 | exact | exact |
| Task screen (26 cases) | 23 | 23 | at most 1 fewer than Q3Floor |

The prose miss belongs to the quantization, not the kernels: the format's own fp32 reference path scores 6.07572, so the fast kernels account for 0.5% of the gap. It was accepted.

### The published image (2026-11-02)

`ghcr.io/iammrduncan/tensorless-engine@sha256:35409ff97c2f518e468790c28bd19206ae777a945fcffe5775585e28fd9939a7`, measured as `0.1.0-rc1`, on one Tesla P100-PCIE-16GB that reported compute capability 6.0 and 16,269 MiB, with the model file checked against sha256 `15ce2d44…`:

| Measure | Value |
|---|---:|
| `tensorless bench` decode, median of 3 runs (256 tokens after a 429-token prompt) | **30.03 tok/s** (30.03 / 29.83 / 30.13) |
| Against the research receipt (30.06) | −0.1% |
| `llama-server`'s own decode timer, 3 runs of 256 tokens after a 45-token prompt | 29.97 / 29.94 / 29.94 tok/s |
| Time to first token, 429-token prompt | 4.46 s (median) |
| GPU memory, idle / generating | 13,997 / 14,577 MiB |
| Load to listening, page cache warm | 3.6 s |

`bench` ran on another computer, through Wi-Fi, ssh and a port-forward to the server. The network adds latency to the time to first token, but not to the decode rate, which is timed between streamed tokens.

### Not measured yet

- **32K and 65K context on this engine.** At those contexts only the climb's starting point was measured: 7.55 / 7.39 tok/s for Qwen3.8-27B-PN3, against 24.52 / 22.92 for Q3Floor. Attention costs a larger share of a 30 tok/s token than of a 7.5 tok/s one, so expect this engine's long-context decode to be below 30.
- **Speculative decoding** for Qwen3.8-27B-PN3.

## Qwen3.8-27B-UD-IQ3_XXS-Q3Floor

### The published image (2026-11-02)

The same image digest, measured as `0.1.0-rc1` and released as `0.1.0-sm60`, on one Tesla P100-PCIE-16GB, with exactly the recipe's switches and flags. The model file was checked against sha256 `4b4d6a39…`, the same bytes as the Hugging Face upload at revision `11f76ec7`.

| Measure | Value |
|---|---:|
| `tensorless bench` decode, median of 3 runs (256 tokens after a 429-token prompt) | **26.18 tok/s** (26.16 / 26.18 / 26.19) |
| Against the research receipt (26.26) | −0.3% |
| `llama-server`'s own decode timer, 3 runs of 256 tokens after a 45-token prompt | 26.16 / 26.19 / 26.21 tok/s |
| Time to first token, 429-token prompt | 3.85 s (median) |
| GPU memory, idle / generating | 13,157 / 13,737 MiB |
| Load to listening, page cache warm | 3.5 s |

### Research receipt

The hill climb's judge decoded this model in every block, as the comparison model for Qwen3.8-27B-PN3.

| Measure | Code | Prose | Scope |
|---|---:|---:|---|
| Decode, short context | 26.26 tok/s | 26.27 tok/s | 256 greedy tokens after a cold prompt of about 600 tokens, on this engine commit |
| Decode at 32K | 24.52 tok/s | 24.54 tok/s | 1,024 greedy tokens after a prompt filling the context, on the climb's starting commit |
| Decode at 65K | 22.92 tok/s | 22.97 tok/s | the same |

Its greedy text stayed bit-identical through the whole climb, and its speed was guarded within 3%.

### Not measured yet

- 32K and 65K context on this engine commit.
- Speculative decoding (MTP) on this engine. The earlier kernel campaign ran it at about 28 tok/s on its own engine version; that mode is not in this recipe.
