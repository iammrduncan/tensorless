---
title: "PN weight format"
description: "PNX-FORMAT-v0: the pn3t and pn4t block layouts, how a value decodes, the reference quantizer, validation, and the files that use them."
type: reference
sidebar:
  order: 6
---

PN is a family of block-quantized weight types designed for the NVIDIA Tesla P100 (`sm_60`). This page describes version 0 of the format, PNX-FORMAT-v0, and its two types, which Qwen3.8-27B-PN3 uses. Why the format looks like this is in [Why a P100-native weight format](../explanation/pn-format.md).

The implementation in the engine is normative:

- the layout: [`engine/ggml/src/ggml-common.h`](https://github.com/iammrduncan/tensorless/blob/main/engine/ggml/src/ggml-common.h);
- decoding, the reference quantizer and validation: [`engine/ggml/src/ggml-quants.c`](https://github.com/iammrduncan/tensorless/blob/main/engine/ggml/src/ggml-quants.c) (`dequantize_row_pn3t`, `dequantize_row_pn4t`, `quantize_row_pnt_ref`);
- the golden tests: [`engine/tests/test-pn-format.cpp`](https://github.com/iammrduncan/tensorless/blob/main/engine/tests/test-pn-format.cpp).

## Types

| ggml type id | Name | Weights per block | Bytes per block | Bits per weight | Codes per u32 word | Zero point `z` |
|---:|---|---:|---:|---:|---|---:|
| 124 | `pn3t` | 80 | 34 | 3.40 | 10 (5 pairs of 3-bit codes) | 4 |
| 125 | `pn4t` | 64 | 34 | 4.25 | 8 (4 pairs of 4-bit codes) | 8 |

The type ids are specific to this engine's ggml fork. Stock llama.cpp does not know them and cannot load a PN file.

## Block layout

Every block is 34 bytes:

```text
byte 0..1    d       fp16, little-endian, SIGNED scale
byte 2..33   qs[32]  eight little-endian u32 words, w0..w7
```

Element `e` of a block (counting from 0) is in word `e / 10` for `pn3t` and `e / 8` for `pn4t`. Inside the word, its local index is `t = e mod 10` (`pn3t`) or `t = e mod 8` (`pn4t`):

| Type | Even `t = 2j` | Odd `t = 2j + 1` | Unused bits |
|---|---|---|---|
| `pn3t` | bits `3j .. 3j+2` | bits `16+3j .. 16+3j+2` | 15 and 31 (reserved, must be 0) |
| `pn4t` | bits `4j .. 4j+3` | bits `16+4j .. 16+4j+3` | none |

So the shift of code `t` is:

```text
shift = BITS * (t >> 1) + 16 * (t & 1)
```

Even codes fill the low 16-bit half of the word and odd codes the high half. No code crosses the halves.

Block `b` starts at byte `34 * b`. The scale is always 2-byte aligned, and a word is 4-byte aligned only in even blocks. Readers must not assume that a word inside a block is 4-byte aligned.

## Value

```text
w = d * (n - z)        n = the code (0..7 for pn3t, 0..15 for pn4t), z = 2^(bits-1)
```

| Type | Codes decode to |
|---|---|
| `pn3t` | `-4d .. +3d` |
| `pn4t` | `-8d .. +7d` |

- Zero is a code (`n = z`), so the types are mid-tread.
- Both steps are exact in fp32: `d` has 11 significant bits and `n - z` at most 4.
- The sign of a zero result is the sign of `d`: code `z` with a negative `d` decodes to `-0.0`.

## Reference quantizer

For each group of 80 (`pn3t`) or 64 (`pn4t`) weights:

1. Compute `amax`, `max` and `min` in double precision. A group with `amax == 0` is the canonical zero group: `d = +0.0`, and every code is `z`.
2. Let `sigma = +1` if `|min| > max`, else `-1`. The largest-magnitude weight then maps to code 0, at the end with the larger magnitude.
3. For each `m` in `{0.50, 0.52, ..., 1.20}` (36 values), take `s = fp16(sigma * m * amax / z)`, rounded to fp16 before it is scored, and skip `s == 0`. The codes are `n = clamp(rint(w / s) + z, 0, 2z - 1)`, and the error is `sum (w - s (n - z))^2`, in double precision.
4. Keep the `s` with the smallest error. A tie goes to the larger `m`.
5. Write `d = s` and pack the codes as above, with the reserved bits 0.

## Validation

A row is invalid when:

- any block's `d` is not finite (an fp16 NaN or infinity); or
- for `pn3t`, bit 15 or bit 31 of any word is set.

## Files that use PN

A PN model is a GGUF file whose weight tensors use types 124 and 125 next to standard ggml types.

Qwen3.8-27B-PN3 is Qwen3.8-27B with mix **P0k+19**:

| Tensors | Type |
|---|---|
| Every matrix-vector tensor with K = 5120, except `attn_k` and `attn_v` | `pn3t` |
| `attn_k`, `attn_v`, and the tensors with K = 6144 and K = 17408 | `pn4t` |
| 19 more tensors, promoted | `pn4t` |
| `token_embd` | Q3_K |
| `output` | Q4_K |
| `ssm_alpha`, `ssm_beta` | Q8_0 |
| Norms | F32 |

Qwen3.8-27B-PN3 also applies a Hadamard rotation to the activations that reach the PN weights. The engine performs it.

## Where PN runs

| Path | State in `engine-v0.2.0` |
|---|---|
| CUDA, `sm_60` | Correct. PN tensors use their own GEMV kernels (`ggml-cuda/pn-sm60.cu`) and, for a prompt's batches, their own GEMM kernels (`ggml-cuda/pn-gemm-sm60.cu`), with no switch needed. |
| CPU reference decode (`dequantize_row_pn3t`, `dequantize_row_pn4t`) | Correct. |
| CPU dot product (`ggml_vec_dot_pn_t_q8_0`) | **Wrong**: it shifts by element index instead of pair index and decodes most PN weights wrong. Do not run PN models on the CPU backend or with partial offload. Every recipe runs the whole model on the GPU. |

The defect is recorded in [`engine/PROVENANCE.md`](https://github.com/iammrduncan/tensorless/blob/main/engine/PROVENANCE.md).
