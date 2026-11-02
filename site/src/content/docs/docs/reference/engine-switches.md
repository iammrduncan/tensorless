---
title: "Engine switches"
description: "The GGML_CUDA_* environment variables the P100 recipes set, what each one selects in the engine, and its default."
type: reference
sidebar:
  order: 5
---

The engine's P100 kernels are opt-in. Each is turned on by a `GGML_CUDA_*` environment variable, which a recipe sets in its `[run.env]` table, exactly as the release gate measured it. The launcher passes them to the container with `-e`, in alphabetical order. Leaving one out, or changing its value, runs a configuration that was not measured.

Both `engine-v0.1.0` recipes set the same 21 switches with the same values. The engine reads each one once, when it is first needed. The source files named below are under [`engine/ggml/src/ggml-cuda/`](https://github.com/iammrduncan/tensorless/tree/main/engine/ggml/src/ggml-cuda).

The PN weight types need no switch: their kernels run whenever a tensor is `pn3t` or `pn4t`.

## Attention

| Variable | Recipe value | Unset | What it selects |
|---|---|---|---|
| `GGML_CUDA_FATTN_SM60` | `1` | off | A split-KV flash-decoding attention kernel for `sm_60`, for a q4_0 K and V cache with head size 256, six query heads per KV head, and 1 to 4 query tokens. It replaces the stock vector and tile kernels in that case (`fattn-sm60.cu`). |
| `GGML_CUDA_FATTN_SM60_CTAS` | `4` | `8` | How many thread blocks per streaming multiprocessor the split-KV kernel aims for when it splits the KV cache. `4` aims for one full wave of blocks. Used only when `GGML_CUDA_FATTN_SM60` applies. |
| `GGML_CUDA_FATTN_TILE_OCC1` | `1` | off | For the tile attention kernel at head size 256 with more than a few query columns (prefill): launch bounds of one block per SM instead of two, which frees registers (`fattn-tile.cuh`). The engine checks only that the variable is set. |

## Matrix-vector products (decode)

These replace stock `mul_mat_vec_q` paths on compute capability 6.0. All but `GGML_CUDA_SM86_IQ3_SMEM_GRID` live in `mmvh-sm60.cu`, and they take effect only when `GGML_CUDA_MMV_SM60` is on.

| Variable | Recipe value | Unset | What it selects |
|---|---|---|---|
| `GGML_CUDA_MMV_SM60` | `1` | off | The `sm_60` GEMV family. Its base kernel multiplies IQ3_XXS and IQ3_S weights by fp16 activations with codebook tables in shared memory, for 1 to 4 columns, in place of the stock path's q8_1 activations and emulated `dp4a`. |
| `GGML_CUDA_MMV_SM60_XXS_STOCK` | `1` | off | Leaves IQ3_XXS products on the stock path, which is faster for them. Products of 2 or more columns get the IQ3_XXS repack first, which the stock path then reads. |
| `GGML_CUDA_MMV_SM60_IQ3S_STOCK` | `1` | off | Leaves single-column (one-token decode) IQ3_S products on the stock path, which is faster there. Wider IQ3_S products stay on the `sm_60` kernel. |
| `GGML_CUDA_MMV_SM60_XXS_KB_WIDE` | `1` | off | With `GGML_CUDA_MMV_SM60_XXS_STOCK`, an exception: IQ3_XXS weights that were not repacked go to the `sm_60` kernel at 2 or 3 columns (`mmvh-sm60.cuh`). |
| `GGML_CUDA_MMV_SM60_INPLACE_XXS` | `1` | off | Writes the IQ3_XXS repack (wide, aligned records the GEMV reads) in place, over the tensor's own enlarged storage, instead of in a separate buffer of about 1 GB. Enlarges those tensors' allocations (`ggml-cuda.cu`). |
| `GGML_CUDA_MMV_SM60_INPLACE_IQ3S` | `1` | off | The same for IQ3_S, instead of a separate buffer of about 2.6 GB. |
| `GGML_CUDA_MMV_SM60_Q4K_HEAD` | `1` | off | An `sm_60` kernel for Q4_K weights with at least 65,536 rows, at 1 to 4 columns: the output head. Weights decode as fp16 subnormals against power-of-two prescaled activations. |
| `GGML_CUDA_MMV_SM60_Q3K` | `1` | off | The same kind of kernel for Q3_K weights at 1 to 4 columns. The weights are repacked once into 112-byte records (the 110-byte blocks plus 2 pad bytes) in a side buffer. |
| `GGML_CUDA_MMV_SM60_Q6K` | `1` | off | The same kind of kernel for Q6_K weights at 1 or 2 columns, repacked once into 224-byte records. |
| `GGML_CUDA_MMV_SM60_Q4K_DHEAD` | `1` | off | For the draft head of MTP speculative decoding: gathers the fixed subset of output-head rows it scores into one contiguous buffer, and runs the head kernel on it. The current recipes use no speculative decoding. |
| `GGML_CUDA_SM86_IQ3_SMEM_GRID` | `1` | off | Copies the IQ3_XXS and IQ3_S codebook grid into shared memory once per block in the stock `mul_mat_vec_q` path, on compute capability 8.6 and 6.0. Same values and order of operations, so the results are bit-identical (`mmvq.cu`). The variable must start with `1`. |

## Fusion and normalization

| Variable | Recipe value | Unset | What it selects |
|---|---|---|---|
| `GGML_CUDA_FUSE_PRE_ADD` | `1` | off | Fuses a residual ADD, the RMS_NORM after it and the MUL after that into one launch, with a check that the fused write cannot overlap a read (`ggml-cuda.cu`). |
| `GGML_CUDA_NORM_EMIT_Q8` | `1` | off | Lets that fused norm also write its output in q8_1, ready for the matrix-vector products that read it, when the shape and every reader allow it (`ggml-cuda.cu`). |

## Prefill

| Variable | Recipe value | Unset | What it selects |
|---|---|---|---|
| `GGML_CUDA_PREFILL_DEQ_F4` | `1` | off | Dequantizes weights to fp32 for prefill with one thread per group of 4 weights and one `float4` store: IQ4_XS weights, and, with `GGML_CUDA_PREFILL_REP_DIRECT`, the repacked IQ3_XXS and IQ3_S records (`convert.cu`). |
| `GGML_CUDA_PREFILL_REP_DIRECT` | `1` | off | Dequantizes repacked IQ3_XXS and IQ3_S records directly, without rebuilding the original block in shared memory first. Same integers and order of operations, so bit-identical (`convert.cu`). |
| `GGML_CUDA_PREFILL_CHUNK_GEMM` | `1` | off | For an fp32 matrix product with more than 1,024 columns, issues it as chunks of 1,024 columns plus a remainder (`ggml-cuda.cu`). The engine checks only that the variable is set. |
| `GGML_CUDA_PREFILL_CHUNK_STREAMS` | `3` | `0` | With chunked GEMM, spreads one weight's chunks over up to this many worker streams, for shapes with at least 5,120 rows. Each chunk is the same call, so the result is the same as in sequence. `0` turns it off. |
| `GGML_CUDA_SM86_GDN_COLS` | `8` | `4` | Columns of state per block in the gated delta net kernel, the recurrent layers of the model. One of `1`, `2`, `4` or `8`; another value means `4`. `8` applies only to batches of 64 tokens or more; shorter ones use `4` (`gated_delta_net.cu`). |

## Other switches

The engine reads more `GGML_CUDA_*` variables than these: alternatives that lost, debug output and statistics. They are not part of any recipe, and the release gate did not measure them. The list of a recipe's switches is in its file, and `tensorless info MODEL` prints them.
