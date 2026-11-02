---
title: "Why the P100"
description: "Why tensorless starts with a 2016 data-center card, what limits its speed, and what the engine does about it."
type: explanation
sidebar:
  order: 2
---

The first card tensorless supports is the NVIDIA Tesla P100 16 GB. This page explains why that card is worth the work, what limits how fast it can run a 27-billion-parameter model, and why it needed its own engine and its own weight format.

## A lot of fast memory for little money

The P100 is a 2016 data-center card from NVIDIA's Pascal generation. Retired from servers, it sells used for around $100. For that price you get:

- **16 GB of HBM2 memory,** stacked next to the chip, at **732 GB/s** (the PCIe 16 GB model);
- **compute capability 6.0** (`sm_60`), the data-center member of the Pascal family.

16 GB is enough to hold a 27B model at about 3 bits per weight, with a long context beside it. 732 GB/s is more memory bandwidth than many much newer consumer cards have. For running a language model one token at a time, that bandwidth is what matters most.

## Why memory bandwidth sets the pace

Generating text (decode) produces one token at a time. For each token, the model's weights are read from memory once and each is used for only a little arithmetic. So decode is **memory-bound**: the card spends its time waiting for weights to arrive, not computing with them.

That gives a simple ceiling. Here is an estimate for Qwen3.8-27B-PN3:

```text
model file                  12,954,114,496 bytes   (12.1 GiB)
rated memory bandwidth      732 GB/s
time to read every byte     12.95 GB / 732 GB/s  ≈ 17.7 ms
ceiling                     ≈ 56 tokens per second
```

This is a rough upper bound, not a prediction. It assumes every byte of the file is read once per token at the full rated bandwidth. In practice the embedding table is only looked up, not read whole; the KV cache adds reads that grow with the context; and no real kernel reaches the rated bandwidth. The released engine decodes this model at about 30 tokens per second at short context, a bit over half the estimate.

The same arithmetic says why low-bit formats exist at all: fewer bytes per weight means fewer bytes to read per token.

## Why bytes are not the whole story on sm_60

On newer cards, fewer bits per weight almost always means faster decode. On the P100 it does not, and the reason is what `sm_60` lacks:

- **No DP4A.** The other Pascal chips (`sm_61`, such as the GeForce GTX 10 series and the Tesla P40) and every generation since have an instruction that multiplies and adds four 8-bit integers at once. llama.cpp's quantized kernels lean on it. `sm_60` does not have it, so those kernels fall back to slower paths.
- **No tensor cores.** They arrived with Volta, the next generation.

Without those, unpacking each weight costs instructions, and the lowest-bit formats cost the most instructions to unpack. A GEMV (the matrix-vector product at the heart of decode) on the P100 can become limited by **instructions per weight** rather than bits per weight.

The evidence is in the two launch targets:

| Model | File size | Decode, short context |
|---|---:|---:|
| Qwen3.8-27B-UD-IQ3_XXS-Q3Floor | 11.0 GiB | 26.3 tok/s |
| Qwen3.8-27B-PN3 | 12.1 GiB | 30.1 tok/s |

The PN3 file is about 10% larger and decodes about 14% faster. Its weights are laid out so the P100 can unpack them cheaply. The Q3Floor model shows the same effect from the other side: raising its 91 sub-3-bit tensors to 3 bits made it both better and faster on this card ([its model card](https://github.com/iammrduncan/tensorless/blob/main/docs/model-cards/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF.md) has the numbers). [Why a P100-native weight format](pn-format.md) explains the layout.

## What it takes to keep running Pascal

The P100 is an old card, and the software around it is moving on:

- **CUDA 13 dropped Pascal.** CUDA 12 is the last major version that compiles for `sm_60`, and the engine image is built with CUDA 12.6. The CUDA runtime is inside the image, so your host does not need CUDA at all.
- **The R580 driver branch is Pascal's last.** The image works with drivers from R525 to R580. Newer branches will not support the card.

So the P100 engine is a fixed target: one architecture, one CUDA line, one driver range. That is part of why the work pays off. A kernel tuned for `sm_60` today will still be the right kernel for every P100 next year.

## Why a whole engine, and not settings for llama.cpp

Stock llama.cpp runs on the P100, but its fast paths assume features the card does not have. Getting from "runs" to "fast" took new kernels: decode attention, GEMVs for each weight type the models use, an output-head kernel, and prefill paths, each written for `sm_60` and switched on by the recipe. [Where the engine comes from](the-engine.md) tells that story, and the [engine switches reference](../reference/engine-switches.md) lists the switches.
