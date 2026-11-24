<p align="center"><img src="docs/assets/banner.png" alt="Bitsy, the tensorless mascot, beside the name tensorless, the line &quot;A $100 GPU card. We made it fly.&quot; and the command tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF, with the model drawn as a glowing tower of layers" width="100%"></p>

# tensorless

<p align="center"><a href="https://huggingface.co/collections/iamMrDuncan/tensorless"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md-dark.svg" alt="Models on Hugging Face" height="28"></a></p>

**The models are on [Hugging Face](https://huggingface.co/collections/iamMrDuncan/tensorless)**: every launch target, in one collection.

**Join us on [Discord](https://discord.gg/3Qs2uejUf9)**: questions, results from your own P100, and ideas.

**[Credits](CREDITS.md)**: the projects, papers, models, datasets and people tensorless is built on.

Custom engine + quant inference for P100s. A ~$100 GPU card, and we will make it fly.

tensorless runs a model that was tuned for your GPU, on the engine it was tuned with, in one command:

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF
```

The first target is the NVIDIA Tesla P100 (16 GB, sm_60), running Qwen3.8-27B-PN3: a 3-bit quantization of Qwen3.8-27B made for this card, served by a llama.cpp-based engine with kernels written for Pascal.

**Status: v0.4.0 released.** The launcher, the engine image `0.4.0-sm60` and the models of all three launch targets are published, and the published image has been measured on a real P100 for each launch target. How the pieces fit together, and how engine releases are made, is in [docs/design.md](docs/design.md).

**New in v0.4.0: models that are more than one file.** A recipe can now name a pack: several files of a Hugging Face repository that the engine reads in place, after a short prepare step in the engine image writes a small GGUF that names them. The launcher downloads and checks each file into the Hugging Face cache, as it does one ([recipe file](recipes/README.md#a-pack-instead-of-one-file)). The first such model, Inkling-2-Mini, is in this release as a draft recipe ([below](#launch-targets)). The three launch targets moved to engine-v0.4.0 with one setting added, `GGML_CUDA_SYNC_SCHED = "auto"`, which keeps engine-v0.3.0's way of waiting for the GPU; the published image was measured with each of them ([receipts](docs/receipts/engine-v0.4.0/)).

v0.3.0 added North-Mini-Code-1.0, whose decode hill climb took it from 49 to 80 tok/s after a 4,096-token prompt ([receipt](docs/receipts/engine-v0.3.0/North-Mini-Code-1.0-GGUF-p100.md)). v0.2.0 made Qwen3.8-27B-PN3's prompts faster: from 121 to 200 tok/s on a 4,100-token prompt ([receipt](docs/receipts/engine-v0.2.0/Qwen3.8-27B-PN3-GGUF-p100.md)).

## Launch targets

Every model tensorless can launch today. Each row is a recipe: the exact engine image, model file and settings that were measured on that card.

| Launch command | Model | Card | Engine | Decode, short context | Prefill, about 4,000 tokens | Decode, 32K / 65K | Status |
|---|---|---|---|---:|---:|---:|---|
| `tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF` | **Qwen3.8-27B-PN3**: Qwen3.8-27B in our own P100-native 3-bit weight format ([PN](docs/pn-format.md)), 12.1 GiB | Tesla P100 16 GB | engine-v0.4.0 | **30.4 tok/s** | **200 tok/s** | not yet measured | **released**; published image measured at **30.46 tok/s** decode |
| `tensorless run iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF` | **Qwen3.8-27B-UD-IQ3_XXS-Q3Floor**: Unsloth's Qwen3.8-27B UD-IQ3_XXS with its 91 sub-3-bit tensors re-quantized to IQ3_XXS, a 3-bit floor (standard llama.cpp types; the model the P100 kernels were first tuned on), 11.0 GiB | Tesla P100 16 GB | engine-v0.4.0 | 26.5 tok/s | 142 tok/s, on engine-v0.3.0's published image | 24.5 / 22.9 tok/s, on the decode climb's starting commit | **released**; published image measured at **26.63 tok/s** decode |
| `tensorless run unsloth/North-Mini-Code-1.0-GGUF` | **North-Mini-Code-1.0 UD-IQ3_XXS**: CohereLabs' 30B mixture-of-experts model for code (8 of 128 experts per token) in Unsloth's UD-IQ3_XXS file, unmodified (standard llama.cpp types), 10.9 GiB | Tesla P100 16 GB | engine-v0.4.0 | **80 tok/s**, after a 4,096-token prompt | **446 tok/s** | **66 tok/s** at 32K; its context is 36,864 tokens | **released**; published image measured at **79.31 tok/s** decode after a 3,911-token prompt and **438 tok/s** prefill |

Decode is greedy single-stream generation: "short context" is 256 tokens after a prompt of about 600 tokens, and 32K/65K is 1,024 tokens after a prompt that fills the context. For North-Mini-Code-1.0 both are 256 tokens, after a cold 4,096- or 32,768-token code prompt. Prefill is one cold prompt, timed from the request to the first token. The decode columns, and the prefill of Qwen3.8-27B-PN3 and North-Mini-Code-1.0, come from the research hill climbs' judge, measured on one P100 with a source build of the engine. The status column has the published image's own measurement, taken before each recipe was marked released: decode with `tensorless bench` for the Qwen models (256 tokens after a 429-token prompt), and for North-Mini-Code-1.0 256 tokens after a 3,911-token prompt, timed the same way; North-Mini-Code-1.0's prefill on a 3,122-token prompt, with the engine's default sync schedule (`blocking`, not the recipe's `auto`). engine-v0.4.0's gate measured the Qwen models' decode only; their prefill on a published image is engine-v0.3.0's. The receipts are in [docs/receipts/](docs/receipts/), and `tensorless models` lists the recipes your launcher knows.

**A draft in v0.4.0: Inkling-2-Mini.** `shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-12.8GB` (short name `inkling2-mini`) is Thinking Machines' Inkling-2-Mini, a mixture-of-experts model that runs a core of its layers three times, in LookAtMyQuant's 12.8 GB pack, unmodified. The pack is 63 files, not one GGUF: the launcher downloads and checks each file, a short prepare step in the engine image writes a small GGUF that names the pack's arrays, and the engine reads the pack in place. A goal campaign on one P100 made the engine run it: on a source build, decode 28.5 tok/s after a 512-token prompt and 25.5 after 32,000, prefill 240 tok/s on a cold 4,096-token prompt, a 32K context in 14,151 MiB, and output that matches the pack's own float32 reference ([receipt draft](docs/receipts/engine-v0.4.0/Inkling-2-Mini-LookAtMyQuant-12.8GB-p100.md)). Its recipe pins the published `0.4.0-sm60` image, but that image has not been measured with this model yet, so the recipe stays a draft: `pull` and `run` refuse it unless you pass `--allow-draft`, and then run it untried. It becomes a launch target in a later patch release, once the image is measured with it on a P100. `tensorless info inkling2-mini` shows it.

## Quick start (Linux, Tesla P100)

You need:
- an NVIDIA driver from R525 to R580 (R580 is the last branch that supports Pascal);
- podman or docker;
- the NVIDIA container toolkit;
- about 13 GB of free disk space for the model, plus a few GB for the image.

```sh
# NVIDIA container support (once): podman uses CDI
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# the launcher: a single static binary
V=0.4.0
curl -fLO https://github.com/iammrduncan/tensorless/releases/download/v$V/tensorless-$V-linux-x86_64.tar.gz
curl -fLO https://github.com/iammrduncan/tensorless/releases/download/v$V/tensorless-$V-linux-x86_64.tar.gz.sha256
sha256sum -c tensorless-$V-linux-x86_64.tar.gz.sha256 && tar -xzf tensorless-$V-linux-x86_64.tar.gz
cd tensorless-$V-linux-x86_64

./tensorless doctor                                                # checks GPU, driver, runtime, GPU passthrough, disk
./tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF                  # pulls, verifies, serves on 127.0.0.1:8080
```

Then, from another terminal:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"iamMrDuncan/Qwen3.8-27B-PN3-GGUF","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}'
./tensorless bench iamMrDuncan/Qwen3.8-27B-PN3-GGUF                # decode tok/s, compared with the recipe's receipt
```

Stop the GPU's other users first: the model and its 65,536-token context take nearly all of the card's 16 GB. Every command is described in [docs/launcher.md](docs/launcher.md).

## How it fits together

- **Engine** (`engine/`): the inference engine source, a llama.cpp fork, published at each release.
- **Images**: one container image per engine release and GPU architecture, pinned by digest.
- **Recipes** (`recipes/<org>/<repo>/<card>.toml`): one model, named by its Hugging Face repository, on one card. A recipe names the exact image, model file, settings and measured numbers.
- **Launcher** (`tensorless`): detects your GPU, picks the recipe, pulls and verifies the image and the model, and starts an OpenAI-compatible server.

## License

Apache-2.0. The engine is derived from MIT-licensed projects (llama.cpp, llama-cpp-turboquant, llamAmpere); see [NOTICE](NOTICE) and [LICENSES/](LICENSES/). Model weights keep their own licenses.
