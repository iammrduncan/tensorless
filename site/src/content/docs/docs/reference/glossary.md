---
title: "Glossary"
description: "The words tensorless uses, from alias to untuned, each in a sentence or two."
type: reference
sidebar:
  order: 9
---

| Term | Meaning |
|---|---|
| **Alias** | The served model name, which `/v1/models` lists and clients send as `model`. A recipe's `run.alias`; the recipes use the repo id. |
| **Card** | A GPU model a recipe is written for, with an id such as `p100`. A recipe's `[card]` table says which GPUs match: compute capability, memory and driver. See [Recipe file](recipes.md#card). |
| **Champion** | A change that won a hill-climb campaign: it was faster than the campaign's best so far, under that campaign's guards. A champion only has to beat the previous one, so it may not contain every earlier winning change. |
| **Compute capability** | NVIDIA's version number for a GPU architecture. The Tesla P100 is 6.0 (`sm_60`). `nvidia-smi` reports it as `compute_cap`. |
| **Context** | The number of tokens the server keeps for one conversation: prompt and output together. The recipes allocate 65,536 tokens. `--context` changes it and makes the run untuned. |
| **Decode** | Generating output tokens one at a time, after the prompt has been read. Decode speed is in tokens per second (tok/s). On a P100 it is limited mostly by how fast the weights can be read from memory. |
| **Digest** | The sha256 of a container image's manifest, such as `sha256:35409ff9…`. A digest names exactly one image; a tag can be moved. Recipes pin images by digest. |
| **Draft** | A recipe with `status = "draft"`: it has not passed the release gate. `pull`, `run` and `serve` refuse it unless `--allow-draft`. |
| **Engine** | The inference engine: a fork of llama.cpp tuned for the P100, in [`engine/`](https://github.com/iammrduncan/tensorless/tree/main/engine). It runs as `llama-server` inside the engine image. |
| **Engine release** | A tagged engine version, `engine-vX.Y.Z`, such as `engine-v0.1.0`: the exact source in `engine/` at that tag, and its build configuration. Versioned separately from the launcher. |
| **Engine switch** | A `GGML_CUDA_*` environment variable that turns on one of the engine's P100 kernels or changes its setting. A recipe sets them in `[run.env]`, exactly as measured. See [Engine switches](engine-switches.md). |
| **Hill climb** | An automated campaign in which AI agents propose changes to the engine, and a judge they cannot change measures each one and keeps only the ones that are faster and pass the guards. The engine-v0.1.0 decode climb made 131 judged attempts and kept 28 improvements. |
| **Image** | One build of an engine release for one GPU architecture, published as `ghcr.io/iammrduncan/tensorless-engine:<X.Y.Z>-<arch>`. See [Engine image](engine-image.md). |
| **Judge** | The measuring program of a hill climb. It builds the candidate, runs it on the card, times it from outside the engine and checks its numerics. Agents cannot see or change it. |
| **Launch target** | A model you can launch: one released recipe. See [Launch targets](launch-targets.md). |
| **Launcher** | The `tensorless` command. It holds no inference code; it reads recipes, fetches and checks the image and the model, and runs the container. |
| **MTP** | Multi-token prediction: a form of speculative decoding in which a small draft head proposes tokens that the model then checks. Not in the v0.1.0 recipes. |
| **P0k+19** | The tensor mix of Qwen3.8-27B-PN3: which tensors use `pn3t`, which use `pn4t`, and which keep standard types. See [PN weight format](pn-format.md#files-that-use-pn). |
| **Pascal** | NVIDIA's GPU architecture of 2016, which includes the Tesla P100. CUDA 13 dropped it, and the R580 driver branch is its last. |
| **PN** | The P100-native block weight format of this engine, PNX-FORMAT-v0. Every block is 34 bytes and decodes exactly in fp32. See [PN weight format](pn-format.md). |
| **pn3t** | The 3-bit PN type, ggml type 124: 80 weights in 34 bytes, 3.40 bits per weight. |
| **pn4t** | The 4-bit PN type, ggml type 125: 64 weights in 34 bytes, 4.25 bits per weight. |
| **Prefill** | Reading the prompt before the first output token. Prefill speed is in prompt tokens per second. |
| **Q3Floor** | Short for Qwen3.8-27B-UD-IQ3_XXS-Q3Floor: Unsloth's UD-IQ3_XXS quantization of Qwen3.8-27B with its 91 sub-3-bit tensors raised to IQ3_XXS, a 3-bit floor, and its token embedding replaced by the Q3_K one from Unsloth's UD-IQ4_XS. |
| **Receipt** | What the release gate measured for a recipe, with the scope of each number and the image digest and model hash it was measured on. See [Receipts](receipts.md). |
| **Recipe** | One model on one card, `recipes/<org>/<repo>/<card>.toml`: the image digest, the model's repository, revision, file and sha256, the engine switches and server flags, the hardware requirements and the measured numbers. See [Recipe file](recipes.md). |
| **Release candidate** | An engine image built for measuring, tagged `X.Y.Z-rcN-<arch>`. When it passes the release gate, its digest gets the release tag without a rebuild. |
| **Release gate** | The checks a candidate image must pass for every recipe on every card before it is released: build checks, numerics against the previous release, speed, long-context decode and prefill, and quality checks. |
| **Released** | A recipe with `status = "released"`, whose image was measured on its card. `pull`, `run` and `serve` accept it with no extra flag. |
| **Repo id** | A Hugging Face repository name, `org/repo`, such as `iamMrDuncan/Qwen3.8-27B-PN3-GGUF`. tensorless names models by repo id, and matches it without case. |
| **Short context** | The decode measurement at a short prompt: 256 greedy tokens after a prompt of about 600 tokens in the research receipts. Its receipt key is `short`. |
| **Short name** | A recipe's optional `model.name`, such as `pn3` or `q3floor`, which works anywhere a repo id does. |
| **sm_60** | The CUDA target for compute capability 6.0, the Tesla P100. Written `sm60` in image tags and recipes. |
| **TTFT** | Time to first token: from sending a request to receiving the first generated token. It includes prefill. |
| **UD** | Unsloth Dynamic, Unsloth's name for its mixed-precision quantizations, such as UD-IQ3_XXS. |
| **Unpublished** | A recipe whose revision, sha256 or image digest is all zeros, or whose `size_bytes` is 0. It cannot be pulled or run, even with `--allow-draft`. |
| **Untuned** | A run that differs from what the recipe measured: another `--context`, extra arguments after `--`, or another `@REVISION`. The launcher labels it, and the receipt's numbers do not apply. |
