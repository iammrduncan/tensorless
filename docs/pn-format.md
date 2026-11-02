# The PN weight format (PNX-FORMAT-v0)

PN is a family of block-quantized weight types designed for the NVIDIA Tesla P100 (sm_60): every block is the same 34 bytes, values decode exactly in fp32, and a packed word holds whole pairs of codes, so a GPU lane can decode a word without crossing a 16-bit boundary. The tensorless engine implements two of its types, which PN3M19 uses.

The implementation in this repository is normative: `engine/ggml/src/ggml-common.h` (layout), `engine/ggml/src/ggml-quants.c` (`dequantize_row_pn3t`, `dequantize_row_pn4t`, `quantize_row_pnt_ref`, validation) and the golden tests in `engine/tests/test-pn-format.cpp`.

## Types

| ggml type | Name | Weights per block | Bytes per block | Bits per weight | Codes per u32 word | Zero point z |
|---|---|---:|---:|---:|---:|---:|
| 124 | `pn3t` | 80 | 34 | 3.40 | 10 (5 pairs of 3-bit codes) | 4 |
| 125 | `pn4t` | 64 | 34 | 4.25 | 8 (4 pairs of 4-bit codes) | 8 |

The type ids are specific to this engine's ggml fork. Stock llama.cpp does not know them and cannot load a PN file.

## Block layout

```text
byte 0..1    d       fp16, little-endian, SIGNED scale
byte 2..33   qs[32]  eight little-endian u32 words, w0..w7
```

Element `e` of a block (0-based) is in word `e / 10` for `pn3t`, `e / 8` for `pn4t`. Inside a word, with local index `t` = `e mod 10` or `e mod 8`:

| Type | Even `t = 2j` | Odd `t = 2j + 1` | Unused bits |
|---|---|---|---|
| `pn3t` | bits `3j .. 3j+2` | bits `16+3j .. 16+3j+2` | 15 and 31 (reserved, must be 0) |
| `pn4t` | bits `4j .. 4j+3` | bits `16+4j .. 16+4j+3` | none |

So the shift for code `t` is `BITS * (t >> 1) + 16 * (t & 1)`: even codes fill the low 16-bit half and odd codes the high half, and no code straddles the halves.

A block starts at byte `34 * b`, so the scale is 2-byte aligned and a word is 4-byte aligned only in even blocks. Readers must not assume 4-byte alignment of a word inside a block.

## Value

```text
w = d * (n - z)        n = the code (0..7 for pn3t, 0..15 for pn4t), z = 2^(bits-1)
```

Codes map to `-4d .. +3d` (`pn3t`) and `-8d .. +7d` (`pn4t`); zero is a code (`n = z`), so the types are mid-tread. Both steps are exact in fp32: `d` has 11 significant bits and `n - z` at most 4. The sign of a zero result is the sign of `d`, so code `z` with a negative `d` decodes to `-0.0`.

## Quantizer (reference)

Per group of 80 (`pn3t`) or 64 (`pn4t`) weights:

1. Compute `amax`, `max` and `min` in double precision. A group with `amax == 0` is the canonical zero group: `d = +0.0`, every code `z`.
2. `sigma = +1` if `|min| > max`, else `-1`, so the largest-magnitude weight maps to code 0, the end with the larger magnitude.
3. For `m` in `{0.50, 0.52, ..., 1.20}` (36 values), take `s = fp16(sigma * m * amax / z)`, rounded to fp16 before it is scored, and skip `s == 0`. Codes are `n = clamp(rint(w / s) + z, 0, 2z - 1)`, and the error is `sum (w - s (n - z))^2` in double precision.
4. Keep the `s` with the smallest error. Ties go to the larger `m`.
5. Write `d = s` and pack the codes as above, with reserved bits 0.

## Validation

A row is invalid when any block's `d` is not finite (fp16 NaN or infinity), or, for `pn3t`, when bit 15 or bit 31 of any word is set.

## Files that use PN

A PN model is a GGUF file whose weight tensors use types 124 and 125 next to standard ggml types. PN3M19 is Qwen3.8-27B with mix **P0k+19**:
- `pn3t` on every GEMV tensor with K = 5120, except `attn_k` and `attn_v`;
- `pn4t` on `attn_k`, `attn_v` and the tensors with K = 6144 and 17408;
- 19 more tensors promoted to `pn4t`;
- the remaining tensors keep their standard types: `token_embd` Q3_K, `output` Q4_K, `ssm_alpha`/`ssm_beta` Q8_0, norms F32.

PN3M19 also applies a Hadamard rotation to the activations that reach the PN weights, which the engine performs.
