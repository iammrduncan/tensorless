---
title: "Why a P100-native weight format"
description: "The design choices behind PN, the weight format of Qwen3.8-27B-PN3, and the quality it trades for speed."
type: explanation
sidebar:
  order: 4
---

Qwen3.8-27B-PN3 stores most of its weights in **PN**, a weight format made for the Tesla P100. This page explains why a new format was worth making, what each of its design choices buys, and what it costs in quality. The exact layout, bit by bit, is in the [PN format reference](../reference/pn-format.md).

## The problem PN solves

On the P100, the cost of a decode GEMV is set by how much work it takes to unpack each weight, not only by how many bits each weight has. [Why the P100](why-the-p100.md) explains why: `sm_60` lacks the integer dot-product instruction that llama.cpp's low-bit kernels rely on. The existing 3-bit formats were designed for other hardware. Their blocks hold scales and codes in layouts that cost the P100 extra instructions to take apart.

PN starts from the other end: what is the cheapest thing for an `sm_60` lane to decode? The answer turned into these choices.

## The choices

### Every block is 34 bytes

Both PN types use the same block: a 2-byte scale and eight 32-bit words of codes.

| Type | Weights per block | Bits per weight | Codes per word |
|---|---:|---:|---:|
| `pn3t` | 80 | 3.40 | 10 three-bit codes |
| `pn4t` | 64 | 4.25 | 8 four-bit codes |

One block size for both types means the kernels share their addressing. A row must hold a whole number of blocks: 5120, the input width of most of Qwen3.8-27B's weight matrices, is exactly 64 `pn3t` blocks. 6144 and 17408 are not multiples of 80, so those tensors use `pn4t`, whose 64 divides all three.

### One lane, one word

In the decode GEMV, each GPU lane owns one packed 32-bit word, ten `pn3t` codes or eight `pn4t` codes, and the activations they multiply. The word is loaded once and decoded in registers. There is no per-weight division to find a block, and no scale stored per small group to look up.

### Codes never straddle the two halves of a word

Codes are packed in pairs: even codes fill the low 16 bits of a word, odd codes the high 16 bits, at the same offsets. No code crosses the middle.

That matters because of alignment. A block is 34 bytes, so a 32-bit word inside it is 4-byte aligned only in every other block. Each 16-bit half, though, is always 2-byte aligned. Because no code crosses the middle, a kernel can load the halves separately and decode each code from its own half, safely, in every block.

### Exact decode in fp32

A weight is `w = d * (n - z)`: the block's scale times the code minus a fixed zero point. The scale is an fp16 number (11 significant bits) and `n - z` is a small integer, so the product always fits exactly in an fp32 number. There is no rounding in decode.

That makes PN easy to check. The CPU reference decoder and every GPU kernel produce bit-identical weights, and the format's golden tests compare them byte for byte. During the engine's hill climb, any kernel change that altered a single decoded weight could be caught.

### Zero is a code, and the scale has a sign

The codes are **mid-tread**: one code means exactly zero, so small weights that round to zero stay zero. That leaves an uneven range, for example −4 to +3 steps for `pn3t`. The scale is **signed**, so the quantizer can point the extra step at whichever side of the block has the larger weight. The largest weight in each block gets the full range.

## How Qwen3.8-27B-PN3 uses it

A PN model is still a GGUF file. PN types sit next to standard ggml types, and each tensor gets the type that suits it. Qwen3.8-27B-PN3 uses the mix called **P0k+19**:

- `pn3t` on every decode GEMV tensor 5120 wide, except `attn_k` and `attn_v`;
- `pn4t` on `attn_k`, `attn_v` and the tensors 6144 and 17408 wide;
- 19 more tensors promoted from `pn3t` to `pn4t`;
- standard types for the rest: the token embedding Q3_K, the output head Q4_K, `ssm_alpha` and `ssm_beta` Q8_0, and the norms F32.

The model also applies a **Hadamard rotation** to the activations that reach the PN weights, and the engine performs it at run time. A rotation spreads a few very large values across many, so a block's single scale fits its weights better. This follows the line of work in QuaRot and SpinQuant, and the weights were quantized with a GPTQ-style method; the [credits](../project/credits.md) list the papers.

The result is a 12.1 GiB file that decodes at about 30 tokens per second on one P100, against 26.3 for the 11.0 GiB Qwen3.8-27B-UD-IQ3_XXS-Q3Floor on the same card and engine.

## What it costs

PN gives up some quality for its speed, and the receipts say how much. Before the decode hill climb started, Qwen3.8-27B-PN3 was checked against Q3Floor on one P100, with these bars:

| Check | Qwen3.8-27B-PN3 | Q3Floor | Bar |
|---|---:|---:|---|
| Perplexity, context 2,048 (6 chunks) | 2.5494 | 2.5682 | at most 1.005 × Q3Floor: passes (0.993×) |
| Decode-path perplexity, code at 32K | 1.38147 | 1.37773 | at most 1.005 × Q3Floor: passes (1.003×) |
| Decode-path perplexity, prose at 32K | 6.07793 | 5.65357 | at most 1.005 × Q3Floor: **misses (1.075×)** |
| Needle recall, up to 31K and 63K | 8 of 8 | exact | exact |
| Task screen (26 cases) | 23 | 23 | at most 1 fewer than Q3Floor |

On code, short text, long-context recall and the task screen, PN3 matches Q3Floor. On long prose it is measurably worse: 7.5% higher perplexity.

That miss was accepted, for a reason the measurements give. The format's own fp32 reference path scores 6.07572 on the same check, so the fast kernels account for only 0.5% of the gap. The rest belongs to the quantization itself, three bits per weight on most tensors. The hill climb then held every later step to the starting point's outputs, within tight bars, so the speed-up from 7.74 to 30 tokens per second did not trade away more quality.

If long prose quality matters more to you than speed, Q3Floor is the safer choice on this card today. The [launch targets](../reference/launch-targets.md) list both.

## Why the format is specific to this engine

PN's GGUF type ids, 124 and 125, are only known to the tensorless engine's fork of ggml. Stock llama.cpp cannot load a PN file, and because the ids are not reserved upstream, another build could one day misread it. So PN files need the tensorless engine, and the format is written down in full, with a reference decoder and test vectors, for anyone who wants to implement it elsewhere.
