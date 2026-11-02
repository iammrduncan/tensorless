---
title: "Launch targets"
description: "Every model tensorless can launch in this release: its files, hashes, image, settings and measured speed."
type: reference
sidebar:
  order: 3
---

A launch target is one recipe: one model on one card, released after its image was measured on that card. tensorless v0.1.0 has two, both on the NVIDIA Tesla P100 16 GB and both on `engine-v0.1.0`.

| Launch command | Short name | Decode, published image | Decode, research receipt |
|---|---|---:|---:|
| `tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF` | `pn3` | **30.03 tok/s** | 30.06 tok/s |
| `tensorless run iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF` | `q3floor` | **26.18 tok/s** | 26.26 tok/s |

Both decode numbers are greedy, single-stream, at short context. "Published image" is `tensorless bench` on the released image: the median of 3 runs of 256 tokens after a 429-token prompt, on one Tesla P100-PCIE-16GB. "Research receipt" is the decode hill climb's judge on a source build of the same engine commit: 256 tokens after a cold prompt of about 600 tokens. The full measurements are in [Receipts](receipts.md).

`tensorless models` lists the recipes your launcher knows, and `tensorless info MODEL` shows any of them in full.

## Qwen3.8-27B-PN3

Qwen3.8-27B in PN, the P100-native 3-bit weight format of this engine (mix P0k+19). Stock llama.cpp cannot load it. See [PN weight format](pn-format.md).

| Field | Value |
|---|---|
| Repository | [`iamMrDuncan/Qwen3.8-27B-PN3-GGUF`](https://huggingface.co/iamMrDuncan/Qwen3.8-27B-PN3-GGUF) |
| Short name | `pn3` |
| Revision | `c701b6690ec360b16d40f85df1cb76062e30cd38` |
| File | `Qwen3.8-27B-PN3-P0k19.gguf` |
| Size | 12,954,114,496 bytes (12.95 GB, 12.1 GiB) |
| sha256 | `15ce2d4469673f1f1753c55de3780cfe04311dd6f05cb41633d73df128e91476` |
| License | Apache-2.0 (from Qwen/Qwen3.8-27B) |
| Card | `p100`: compute capability 6.0, at least 16,000 MiB, driver 525.60.13 or newer |
| Engine release | `engine-v0.1.0`, `sm60` |
| Image | `ghcr.io/iammrduncan/tensorless-engine@sha256:35409ff97c2f518e468790c28bd19206ae777a945fcffe5775585e28fd9939a7` (tag `0.1.0-sm60`) |
| Context | 65,536 tokens, q4_0 KV cache |
| Recipe | [`recipes/iamMrDuncan/Qwen3.8-27B-PN3-GGUF/p100.toml`](https://github.com/iammrduncan/tensorless/blob/main/recipes/iamMrDuncan/Qwen3.8-27B-PN3-GGUF/p100.toml) |

| Measure | Value | Scope |
|---|---:|---|
| Decode, published image | 30.03 tok/s | `tensorless bench`, median of 3 runs, 256 tokens after a 429-token prompt |
| Decode, short context | 30.06 tok/s | research receipt; code prompt (prose: 30.18) |
| Prefill, about 4,100 tokens | 121.4 tok/s | research receipt; code prompt (prose: 119.6) |
| Decode at 32K and 65K | not measured on this engine | |

## Qwen3.8-27B-UD-IQ3_XXS-Q3Floor

Unsloth's Qwen3.8-27B UD-IQ3_XXS with a 3-bit floor: its 91 sub-3-bit tensors re-quantized to IQ3_XXS, and `token_embd` taken from Unsloth's UD-IQ4_XS. It uses standard llama.cpp types, and everything else in the file is Unsloth's, byte for byte. It is the model the P100 kernels were first tuned on, and the comparison model of every Qwen3.8-27B-PN3 measurement. See [Models built on someone else's file](../explanation/models-built-on-others.md).

| Field | Value |
|---|---|
| Repository | [`iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF`](https://huggingface.co/iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF) |
| Built from | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) at `4ca720788d1e01f1bff70c033e0d0028fd02e502`, `Qwen3.8-27B-UD-IQ3_XXS.gguf` |
| Short name | `q3floor` |
| Revision | `11f76ec7e1f7be8e76563ee2b9a912a8519e9801` |
| File | `Qwen3.8-27B-UD-IQ3_XXS-Q3Floor.gguf` |
| Size | 11,758,422,944 bytes (11.76 GB, 11.0 GiB) |
| sha256 | `4b4d6a39dca07c915472b3fdf70267385f6e6f3f38befd5223c9a1ac272cecd0` |
| License | Apache-2.0 (from Qwen/Qwen3.8-27B and unsloth/Qwen3.8-27B-GGUF) |
| Card | `p100`: compute capability 6.0, at least 16,000 MiB, driver 525.60.13 or newer |
| Engine release | `engine-v0.1.0`, `sm60` |
| Image | `ghcr.io/iammrduncan/tensorless-engine@sha256:35409ff97c2f518e468790c28bd19206ae777a945fcffe5775585e28fd9939a7` (tag `0.1.0-sm60`) |
| Context | 65,536 tokens, q4_0 KV cache |
| Recipe | [`recipes/iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF/p100.toml`](https://github.com/iammrduncan/tensorless/blob/main/recipes/iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF/p100.toml) |

| Measure | Value | Scope |
|---|---:|---|
| Decode, published image | 26.18 tok/s | `tensorless bench`, median of 3 runs, 256 tokens after a 429-token prompt |
| Decode, short context | 26.26 tok/s | research receipt on this engine commit; code prompt (prose: 26.27) |
| Decode at 32K / 65K | 24.52 / 22.92 tok/s | research receipt on the climb's starting commit; 1,024 tokens after a 31.7K / 64.5K-token prompt, code (prose: 24.54 / 22.97) |

## Settings both recipes share

Both recipes run the same image with the same arguments and the same 21 [engine switches](engine-switches.md). The launcher adds the model path, host, port, context, alias and thread counts.

| `llama-server` argument | Meaning |
|---|---|
| `-np 1` | One request slot. |
| `-ngl 99` | Every layer on the GPU. |
| `-ot token_embd.weight=CUDA0` | The token embedding on the GPU too. |
| `-ctk q4_0 -ctv q4_0` | A q4_0 KV cache. |
| `-sm none` | One GPU, no split. |
| `-lm none` | Load mode `none`: no memory-mapping or locking of the model file. |
| `-fit off` | Leaves unset arguments as they are, instead of adjusting them to fit in device memory. |
| `--threads-http 4` | Four HTTP threads. |
| `-b 512 -ub 512` | Batch and micro-batch of 512 tokens. |
| `-fa on` | Flash attention on. |
| `--jinja` | Applies the model's chat template for `/v1/chat/completions`. Not part of the measured configuration. |

| Recipe setting | Value |
|---|---|
| `context` | 65536 |
| `threads_max` | 8 |
| `min_driver` | 525.60.13 |

The model and its 65,536-token context take most of the card's 16 GB: the published image used 13,997 MiB idle and 14,577 MiB generating with Qwen3.8-27B-PN3, and 13,157 / 13,737 MiB with Q3Floor.
