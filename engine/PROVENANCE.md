# Engine provenance: engine-v0.1.0

This directory is the tensorless engine at release **engine-v0.1.0**: a fork of llama.cpp for the NVIDIA Tesla P100 (sm_60), exported from the research repository where its kernels were developed and hill-climbed.

| Item | Value |
|---|---|
| Upstream base | [JakeATX/llamAmpere](https://github.com/JakeATX/llamAmpere) `2cb16936b5d081a92f1d0369954561efb6d1e2c7` (MIT), itself a fork of [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) and [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) |
| Research commit | `cfcab96e55d0827055c4b6cb2e8957dc0ebba712` |
| Files changed from the upstream base | 55 |
| Tree digest | `d6a8a4403918fe7abd1e00374245c7114107c23d808ca6bd7a99a36025e52176` (sha256 over sorted relative paths and contents) |

## What this fork adds

- **P100 decode kernels** (opt-in through `GGML_CUDA_*` switches): split-KV q4_0 flash-decoding attention (`fattn-sm60`), sm_60 quantized GEMVs and an IQ3_XXS fp16-table GEMV (`mmvh-sm60`, `mmvq` paths), Q3_K/Q4_K/Q6_K and output-head kernels, prefill dequantization and chunked GEMM, norm fusions, and the Qwen3.5-architecture graph changes they need.
- **The PN weight types** `pn3t` (124) and `pn4t` (125), with their CPU reference, CUDA dequantization and sm_60 GEMV kernels (`ggml-cuda/pn-sm60.cu`). The format is specified in tensorless `docs/pn-format.md`.
- `tools/tf-logits` (teacher-forced logits) and `tests/test-pn-format.cpp`.

The PN kernels at this release are the result of an automated decode hill climb: 131 judged attempts, 28 confirmed improvements, PN3M19 decode on one P100 from 7.74 to about 30 tok/s at short context, with every step's numerics held against its starting point.

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
