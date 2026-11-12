# Engine provenance: engine-v0.2.0

This directory is the tensorless engine at release **engine-v0.2.0**: a fork of llama.cpp for the NVIDIA Tesla P100 (sm_60), exported from the research repository where its kernels were developed and hill-climbed.

| Item | Value |
|---|---|
| Upstream base | [JakeATX/llamAmpere](https://github.com/JakeATX/llamAmpere) `2cb16936b5d081a92f1d0369954561efb6d1e2c7` (MIT), itself a fork of [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) and [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) |
| Research commit | `df673571ed7238d2fe30d4b8355d7707d6672054` (git tree `ee5aa3a361b47887dcaeef60316b6bf29a80a947`) |
| Previous release | engine-v0.1.0, research commit `cfcab96e55d0827055c4b6cb2e8957dc0ebba712` |
| Files changed from the upstream base | 64 |
| Files changed from engine-v0.1.0 | 17 (15 changed, 2 new: `ggml/src/ggml-cuda/pn-gemm-sm60.cu` and `.cuh`) |
| Tree digest | `b44cc91baf8c6c03aebfd53c9757cb06b42d633701810f5f1eefa5d721f6e7cb` (sha256 over sorted relative paths and contents) |

## What this fork adds

- **P100 decode kernels** (opt-in through `GGML_CUDA_*` switches): split-KV q4_0 flash-decoding attention (`fattn-sm60`), sm_60 quantized GEMVs and an IQ3_XXS fp16-table GEMV (`mmvh-sm60`, `mmvq` paths), Q3_K/Q4_K/Q6_K and output-head kernels, prefill dequantization and chunked GEMM, norm fusions, and the Qwen3.5-architecture graph changes they need.
- **The PN weight types** `pn3t` (124) and `pn4t` (125), with their CPU reference, CUDA dequantization and sm_60 GEMV kernels (`ggml-cuda/pn-sm60.cu`). The format is specified in tensorless `docs/pn-format.md`.
- **The PN prefill path** (new in this release, on by default; `ggml-cuda/pn-gemm-sm60.cu`): what a batched PN `MUL_MAT` (more than 4 tokens, so every prompt ubatch) runs instead of dequantize-to-fp32 plus cuBLAS SGEMM.
  - An exact-order fp32 GEMM for the m = 5120 and 12288 shapes, and a cuBLASLt route for the other large shapes. Both give the same bits as the cuBLAS call they replace.
  - A half2 (fp16) GEMM with fp32 accumulation across windows for the weights of blocks 31 and up, except `attn_k` and `attn_v`, and a shorter-chain precision tier for blocks 27 to 30 in the ubatches that are neither a prompt's first 512 tokens nor its final ubatch.
  - Vectorized PN dequantization, to fp32 and to fp16, prefetched on a side stream while the previous GEMM runs.
  - A fused SwiGLU epilogue for the FFN gate/up pair, and smaller changes to concat, RMS norm, the gated delta net and the flash-attention tile on the batched path.
  - In `llama_context::decode`, two consecutive prompt batches with no outputs run as one ubatch of up to twice `n_ubatch` tokens.
  - For T = 1 decode, `ssm_beta` and `ssm_alpha` load as views of one tensor and run as one GEMV.
- `tools/tf-logits` (teacher-forced logits) and `tests/test-pn-format.cpp`.

The PN kernels are the result of two automated hill climbs, each holding the model's numerics against its starting point:
- **Decode** (engine-v0.1.0): 131 judged attempts, 28 confirmed improvements, PN3M19 decode on one P100 from 7.74 to about 30 tok/s at short context.
- **Prefill** (this release): 369 attempts, 26 promoted improvements, PN3M19 prefill on one P100 from 121.4 to 200.1 tok/s on a cold prompt of about 4,100 tokens, with the decode held within 1% of its starting point.

## What the export changed

Research-only files were left out: `.auto/`, `v2-lane.env`, `verdict.txt`, `HISTORY-MAP.tsv` (9 files: autoresearch state, the research lane's switch file, a status note, a commit map).

Comment lines that pointed into the private research repository were rewritten, one line for one line, so line numbers and compiled code are unchanged:
- `ggml/src/ggml-cuda/fattn-sm60.cu:2`
- `ggml/src/ggml-cuda/mmvh-sm60.cu:2`
- `ggml/src/ggml-cuda/mmvh-sm60.cuh:1`
- `ggml/src/ggml-cuda/pn-sm60.cu:3`
- `ggml/src/ggml-quants.c:169`
- `tests/test-pn-format.cpp:3`

The upstream license is in `LICENSE`; tensorless's own changes are Apache-2.0 (see the repository's `NOTICE`).

## Known issue

The CPU backend's PN dot product (`ggml_vec_dot_pn_t_q8_0` in `ggml-cpu/quants.c`) shifts by element index instead of pair index and decodes most PN weights wrong. GPU paths are correct and every tensorless recipe runs the whole model on the GPU; do not run PN models on the CPU backend or with partial offload until it is fixed.
