---
title: "Models built on someone else's file"
description: "Why the Q3Floor model is hosted whole under tensorless's name, how it is named, and how its original publisher is credited."
type: explanation
sidebar:
  order: 6
---

Some tuned models are a small change to a file someone else published. Qwen3.8-27B-UD-IQ3_XXS-Q3Floor is one: most of its bytes are Unsloth's. This page explains how such a model is built, why tensorless hosts the whole file instead of patching the original on your machine, and how the name and the credit work.

## What Q3Floor is

Unsloth publishes a ladder of quantizations of Qwen3.8-27B. Their **UD-IQ3_XXS** file ("UD" is Unsloth Dynamic) keeps most tensors at about 3 bits, but stores 91 of them below 3 bits, in types such as IQ2_S, IQ2_XXS and IQ1_S. That saves space. On the P100 it costs speed, because the lowest-bit types are the most expensive to unpack on `sm_60` ([Why the P100](why-the-p100.md)).

Qwen3.8-27B-UD-IQ3_XXS-Q3Floor puts a **3-bit floor** under that file:

| Part | In Unsloth's UD-IQ3_XXS | In Q3Floor |
|---|---|---|
| 91 tensors below 3 bits | IQ2_S, IQ2_XXS, IQ2_XS, Q2_K, IQ1_S, IQ1_M | IQ3_XXS, re-quantized with stock `llama-quantize` from Unsloth's BF16 file and their importance matrix |
| The token embedding | Q2_K | Q3_K, copied byte for byte from Unsloth's UD-IQ4_XS |
| Everything else, metadata included | | unchanged, byte for byte |

About 73% of the file's bytes are Unsloth's, untouched. Even the re-quantized tensors are made from their BF16 conversion with their importance matrix. It uses only standard llama.cpp types, so it also loads in stock llama.cpp.

On the P100 the floor made the model both faster and better: about 4.5% faster at 32K and 65K context than the original file on the same engine, and lower perplexity on a short check. The [model card](https://github.com/iammrduncan/tensorless/blob/main/docs/model-cards/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF.md) has the numbers.

## Why host the whole file

Another way would be to point the recipe at Unsloth's repository and apply the change on your machine, at download time. tensorless does not do that, for three reasons.

- **The receipts measured one exact file.** Its sha256 is `4b4d6a39…`. A remix on your machine would have to produce those exact bytes every time, with your copy of `llama-quantize`, from Unsloth's files at the same revision. Any difference, and the numbers no longer apply. Hosting the measured file means the launcher can check the hash and know.
- **Pinning works the same way for every model.** A recipe names a repository, a revision, a file and a sha256. A remix step would be a second kind of recipe, with its own failure modes.
- **It is less to download and less to run.** One 11 GB file, checked once. No BF16 source file, no quantization run.

So the file lives on Hugging Face beside Qwen3.8-27B-PN3, as `iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF`: renamed, and nothing else.

## How it is named

The name says what the file is:

- **It keeps the original label.** "UD-IQ3_XXS" is Unsloth's own name for the quantization, so their users recognize what it is built on.
- **It adds what changed.** "Q3Floor": no tensor below 3 bits.

The file's internal metadata is still Unsloth's, `general.name` included. Changing it would change the sha256 the receipts were measured on, so the file was renamed, not rewritten.

## How the original publisher is credited

A model built on someone else's file says so where people look first:

- **the model card** names Unsloth, links their repository at the exact revision used (`4ca72078…`), lists every change, and carries their license (Apache-2.0, from Qwen and Unsloth);
- **the recipe** says the same in its comments, and names the license;
- **the [credits](../project/credits.md)** say that the Q3Floor model is mostly Unsloth's work.

This is the rule for any future model of this kind: host the measured bytes whole, keep the original publisher's name in the model's name, and credit them by revision.
