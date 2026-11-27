---
title: "Launch targets"
description: "Every model tensorless can launch in this release: its files, hashes, image, settings and measured speed."
type: reference
sidebar:
  order: 3
---

A launch target is one recipe: one model on one card, released after its image was measured on that card. tensorless v0.4.1 has three, all on the NVIDIA Tesla P100 16 GB and all on `engine-v0.4.1`. A fourth recipe, Inkling-2-Mini, is a [draft](#draft-in-v041-inkling-2-mini), on `engine-v0.4.0`.

| Command | Short name | Card | Decode | Prefill | Decode at 32K |
|---|---|---|---:|---:|---:|
| `tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF` | `pn3` | 1× Tesla P100 16 GB | 30.4 | 200 | – |
| `tensorless run iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF` | `q3floor` | 1× Tesla P100 16 GB | 26.5 | – | 24.5 |
| `tensorless run unsloth/North-Mini-Code-1.0-GGUF` | `nmc` | 1× Tesla P100 16 GB | 79.8 | 642 | 66.1 |
| `tensorless run shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-12.8GB --allow-draft` (draft) | `inkling2-mini` | 1× Tesla P100 16 GB | 28.5 | 240 | 25.5 |

Card is how many cards and which: every launch target today runs on one. Tokens a second, greedy. These are the numbers in each recipe's `[receipt]`:
- decode after a short prompt (North-Mini-Code-1.0: 4,096 tokens);
- prefill on a cold prompt of about 4,000 tokens;
- decode after a 32K prompt.

– means not measured yet. Most of these come from a hill climb's or goal campaign's judge, on a source build of a research commit. Each target's section below gives the published image's own measurement beside them; the published image has decode 30.30, 26.62 and 94.87 tok/s with `tensorless bench`, on a 429- or 411-token prompt, with the recipes' `GGML_CUDA_SYNC_SCHED=auto`. The full measurements are in [Receipts](receipts.md).

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
| Engine release | `engine-v0.4.1`, `sm60` |
| Image | `ghcr.io/iammrduncan/tensorless-engine@sha256:c89e60b34a0f6098db20f2f2b43c78a93bb70192833cf7656b4a84ae8974346a` (tag `0.4.1-sm60`) |
| Context | 65,536 tokens, q4_0 KV cache |
| Recipe | [`recipes/iamMrDuncan/Qwen3.8-27B-PN3-GGUF/p100.toml`](https://github.com/iammrduncan/tensorless/blob/main/recipes/iamMrDuncan/Qwen3.8-27B-PN3-GGUF/p100.toml) |

| Measure | Value | Scope |
|---|---:|---|
| Decode, published image | 30.30 tok/s | `tensorless bench`, median of 3 runs, 256 tokens after a 429-token prompt; engine-v0.4.0's image read 30.17 on the same card in the same hour |
| Time to first token, published image | 2.89 s | the same runs, 429-token prompt (engine-v0.1.0: 4.46 s) |
| Prefill, published image | 192.8 tok/s | one cold prompt of 3,651 tokens, code (prose, 4,208 tokens: 197.5) |
| Decode, short context | 30.43 tok/s | research receipt on engine-v0.2.0's research commit; code prompt (prose: 30.31) |
| Prefill, about 4,100 tokens | 200.1 tok/s | research receipt on engine-v0.2.0's research commit; code prompt (prose: 198.9). engine-v0.1.0: 121.4 |
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
| Engine release | `engine-v0.4.1`, `sm60` |
| Image | `ghcr.io/iammrduncan/tensorless-engine@sha256:c89e60b34a0f6098db20f2f2b43c78a93bb70192833cf7656b4a84ae8974346a` (tag `0.4.1-sm60`) |
| Context | 65,536 tokens, q4_0 KV cache |
| Recipe | [`recipes/iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF/p100.toml`](https://github.com/iammrduncan/tensorless/blob/main/recipes/iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF/p100.toml) |

| Measure | Value | Scope |
|---|---:|---|
| Decode, published image | 26.62 tok/s | `tensorless bench`, median of 3 runs, 256 tokens after a 429-token prompt; engine-v0.4.0's image read 26.65 on the same card in the same hour |
| Time to first token, published image | 4.12 s | the same runs, 429-token prompt |
| Prefill, published image | 142.0 tok/s | one cold prompt of 3,651 tokens, code (prose, 4,208 tokens: 147.0) |
| Decode, short context | 26.53 tok/s | research receipt on engine-v0.2.0's research commit; code prompt (prose: 26.58) |
| Decode at 32K / 65K | 24.52 / 22.92 tok/s | research receipt on the decode climb's starting commit, five engine releases back; 1,024 tokens after a 31.7K / 64.5K-token prompt, code (prose: 24.54 / 22.97) |

## North-Mini-Code-1.0 UD-IQ3_XXS

CohereLabs' North-Mini-Code-1.0, a 30B mixture-of-experts model for code (49 layers, 8 of 128 experts per token, a 4,096-token sliding window on 36 layers), in Unsloth's UD-IQ3_XXS file, unmodified. Its experts are IQ2_S and IQ3_S, its attention and head Q6_K: standard llama.cpp types. engine-v0.3.0's decode hill climb and engine-v0.4.1's prefill hill climb were run on it. It reasons before it answers: `/v1/chat/completions` returns the reasoning in `reasoning_content`, so give it room in `max_tokens`.

| Field | Value |
|---|---|
| Repository | [`unsloth/North-Mini-Code-1.0-GGUF`](https://huggingface.co/unsloth/North-Mini-Code-1.0-GGUF) |
| Short name | `nmc` |
| Revision | `e306bb4bf0df610f5471d97a01de2b6e0b24d356` |
| File | `North-Mini-Code-1.0-UD-IQ3_XXS.gguf` |
| Size | 11,708,375,136 bytes (11.71 GB, 10.9 GiB) |
| sha256 | `029b51f93570c4295680bf6fb138bbc51bba7c377d613f75ac4eaa6913597140` |
| License | Apache-2.0 (from CohereLabs/North-Mini-Code-1.0 and unsloth/North-Mini-Code-1.0-GGUF) |
| Card | `p100`: compute capability 6.0, at least 16,000 MiB, driver 525.60.13 or newer |
| Engine release | `engine-v0.4.1`, `sm60` |
| Image | `ghcr.io/iammrduncan/tensorless-engine@sha256:c89e60b34a0f6098db20f2f2b43c78a93bb70192833cf7656b4a84ae8974346a` (tag `0.4.1-sm60`) |
| Context | 36,864 tokens, q8_0 KV cache |
| Recipe | [`recipes/unsloth/North-Mini-Code-1.0-GGUF/p100.toml`](https://github.com/iammrduncan/tensorless/blob/main/recipes/unsloth/North-Mini-Code-1.0-GGUF/p100.toml) |

| Measure | Value | Scope |
|---|---:|---|
| Decode, published image | 94.87 tok/s | `tensorless bench`, median of 3 runs, 256 tokens after a 411-token prompt |
| Time to first token, published image | 0.972 s | the same runs, 411-token prompt (engine-v0.4.0's image, the same hour: 1.456 s) |
| Decode after about 4,000 tokens, published image | 80.39 tok/s | 256 greedy tokens after a cold 3,911-token code prompt, median of 3 |
| Decode after about 32,000 tokens, published image | 66.80 tok/s | the same after a cold 31,832-token code prompt |
| Prefill, published image | 600.3 tok/s | one cold prompt of 3,122 tokens, code (prose, 3,827 tokens: 598.6); engine-v0.4.0's image, the same hour: 434.2 |
| Prefill after about 32,000 tokens, published image | 406.9 tok/s | `llama-server`'s own timer, the 31,832-token prompt above; engine-v0.4.0's image: 333.7 |
| Decode after 4,096 tokens | 79.77 tok/s | research receipt on engine-v0.3.0's research commit; the starting point read 49.28 |
| Decode after 32,768 tokens | 66.06 tok/s | research receipt; the starting point read 35.83 |
| Prefill, 4,096 / 32,768 tokens | 642.1 / 434.2 tok/s | research receipt: engine-v0.4.1's prefill hill climb's judge, last confirmation, on engine-v0.3.0's base; its starting point read 446.5 / 334.8 |

## Settings the Qwen3.8-27B recipes share

The three launch targets run the same image with the same 21 [engine switches](engine-switches.md), and `GGML_CUDA_SYNC_SCHED=auto`, which keeps engine-v0.3.0's host sync schedule. The two Qwen3.8-27B recipes also share their arguments. The launcher adds the model path, host, port, context, alias and thread counts.

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

The model and its 65,536-token context take most of the card's 16 GB: the published image used 14,139 MiB idle and 15,381 MiB after its first prompt with Qwen3.8-27B-PN3 (15,383 at most), and 13,289 / 13,869 MiB with Q3Floor, the same as engine-v0.4.0's image.

North-Mini-Code-1.0's recipe differs in four places: a context of 36,864 tokens, a q8_0 KV cache (`-ctk q8_0 -ctv q8_0`), and no `-ot token_embd.weight=CUDA0` or `-sm none`. Those are the settings its hill climb's judge ran. The published image used 13,877 MiB idle and 14,431 MiB from its first prompt on, 32K prompts included: 22 MiB more than engine-v0.4.0's image. At a chat turn that reused the prompt cache it rose to 15,295 MiB, with 976 MiB still free. engine-v0.4.0's image did the same in the same hour, to 15,273 MiB, so the step is not new in this release; its cause is not known ([receipt](https://github.com/iammrduncan/tensorless/blob/main/docs/receipts/engine-v0.4.1/North-Mini-Code-1.0-GGUF-p100.md#gpu-memory-at-the-chats-second-pass)).

## Draft in v0.4.1: Inkling-2-Mini

Not a launch target yet. Its recipe stays on `engine-v0.4.0` and pins that release's published image, `0.4.0-sm60`, which has not been measured with this model on a P100, so the recipe is a draft: `pull`, `run` and `serve` refuse it unless you pass `--allow-draft`, and then run a configuration nobody has measured on the published image. It becomes a launch target in a later patch release, once it has been. `tensorless info inkling2-mini` and `--dry-run` show it.

Thinking Machines' Inkling-2-Mini (architecture `inkling2_loop`: 24 layers, 8 of 128 experts per token, a core of middle layers that runs three times), in LookAtMyQuant's 12.8 GB pack, unmodified. The pack is not one GGUF file: it is 63 files that the engine reads in place, after a prepare step writes a small GGUF that names the pack's arrays. See [Recipe file](recipes.md#modelpack).

| Field | Value |
|---|---|
| Repository | [`shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-12.8GB`](https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-12.8GB) |
| Short name | `inkling2-mini` |
| Revision | `758a177fdaba30e53cce1266504b5428290aa0c9` |
| Files | 63, each pinned by name, sha256 and size in the recipe |
| Size | 12,760,459,456 bytes in all (12.76 GB, 11.9 GiB) |
| Prepare step | `python3 -E -s /opt/tensorless/inkling-prepare/prepare.py`, in the engine image, before each serve |
| License | Apache-2.0 (from thinkingmachines/Inkling-2-Mini and shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-12.8GB) |
| Card | `p100`: compute capability 6.0, at least 16,000 MiB, driver 525.60.13 or newer |
| Engine release | `engine-v0.4.0`, `sm60` |
| Image | `ghcr.io/iammrduncan/tensorless-engine@sha256:4a0153230be9e2b5957a5aaf58cbb0ba1f79bfeb94eacd4617189f16ca35e997` (tag `0.4.0-sm60`), not yet measured with this model |
| Context | 32,768 tokens, f16 KV cache |
| Recipe | [`recipes/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-12.8GB/p100.toml`](https://github.com/iammrduncan/tensorless/blob/main/recipes/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-12.8GB/p100.toml) |

| Measure | Value | Scope |
|---|---:|---|
| Decode after a 512-token prompt | 28.52 tok/s | research build, the goal campaign's judge: median of 3 greedy replies of 128 tokens |
| Decode after a 32,000-token prompt | 25.45 tok/s | the same, median of 2 |
| Prefill, 4,096 tokens | 239.89 tok/s | the same, a cold prompt |
| Peak card memory | 14,151 MiB | the same, at a 32,768-token context |

Its arguments are `-np 1 -ngl 99 -ctk f16 -ctv f16 -lm none -fit off --threads-http 4 -b 512 -ub 512 -fa auto --jinja`, with the same 21 engine switches as the other recipes and no `GGML_CUDA_SYNC_SCHED`, so the engine's default, `blocking`: what the campaign's judge ran, apart from `--jinja`.
