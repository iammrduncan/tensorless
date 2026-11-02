---
title: "License"
description: "tensorless is Apache-2.0; its engine keeps the MIT notices of the projects it derives from; model weights keep their own licenses."
type: project
sidebar:
  order: 4
---

tensorless is open source under the **Apache License, Version 2.0**. This page says what that covers, what keeps another license, and where each notice is.

## What is Apache-2.0

The launcher, the recipes, the container files, these docs and every change made for tensorless are licensed under [Apache-2.0](https://github.com/iammrduncan/tensorless/blob/main/LICENSE). The project's [NOTICE](https://github.com/iammrduncan/tensorless/blob/main/NOTICE) file carries the attribution that Apache-2.0 asks you to keep.

## What keeps its own license

**The engine's upstreams.** The engine in `engine/` is derived from MIT-licensed projects, and their copyright and permission notices stay with it:

| Project | License |
|---|---|
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp), the ggml authors | MIT |
| [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant), TheTom | MIT, the same notice |
| [llamAmpere](https://github.com/JakeATX/llamAmpere), JakeATX | MIT, the same notice |
| [exllamav3](https://github.com/turboderp-org/exllamav3), turboderp (portions, for the EXL3 weight types) | MIT |

**Libraries vendored into the engine,** such as cpp-httplib, nlohmann/json and xxHash, each under its own license (MIT, BSD-2-Clause, public domain and others). **The launcher's one dependency,** [BurntSushi/toml](https://github.com/BurntSushi/toml), is MIT. **The CUDA libraries** in the engine images come from NVIDIA's CUDA base images, under the NVIDIA CUDA Toolkit license.

[THIRD_PARTY_NOTICES.md](https://github.com/iammrduncan/tensorless/blob/main/THIRD_PARTY_NOTICES.md) lists every one of them and where its notice is. The license texts are in [LICENSES/](https://github.com/iammrduncan/tensorless/tree/main/LICENSES) and in the engine's own license files. The release archives of the launcher carry `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md` and `LICENSES/`, and each engine image carries the same files under `/opt/tensorless/`.

## Model weights

Model weights are not part of tensorless. They are downloaded from Hugging Face and keep their own licenses. Each recipe names its model's license, and `tensorless info MODEL` shows it before anything is downloaded.

| Model | License |
|---|---|
| Qwen3.8-27B-PN3 | Apache-2.0, from Qwen/Qwen3.8-27B |
| Qwen3.8-27B-UD-IQ3_XXS-Q3Floor | Apache-2.0, from Qwen/Qwen3.8-27B and unsloth/Qwen3.8-27B-GGUF |

Read a model's card on Hugging Face for its full terms.
