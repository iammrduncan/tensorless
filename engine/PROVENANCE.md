# Engine provenance: engine-v0.3.0

This directory is the tensorless engine at release **engine-v0.3.0**: a fork of llama.cpp for the NVIDIA Tesla P100 (sm_60). Its kernels were developed and hill-climbed in automated campaigns; this release adds the decode work of a hill climb on North-Mini-Code-1.0, run on this repository's engine.

| Item | Value |
|---|---|
| Upstream base | [JakeATX/llamAmpere](https://github.com/JakeATX/llamAmpere) `2cb16936b5d081a92f1d0369954561efb6d1e2c7` (MIT), itself a fork of [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) and [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) |
| Hill-climb commit | `7d8be264fcadf1ec7ea9019f466c01e1253d27c0` (engine tree `bfbcbf1cb0f28cf4c4b9d432e07913c1e45ce628`), the champion of the North-Mini-Code decode hill climb |
| Previous release | engine-v0.2.0 (research commit `df673571ed7238d2fe30d4b8355d7707d6672054`) |
| Files changed from engine-v0.2.0 | 18, none new: `src/llama-kv-cache-iswa.cpp` (the climb's starting point, below) and 17 from the climb |

## What this fork adds

- **P100 decode kernels** (opt-in through `GGML_CUDA_*` switches): split-KV q4_0 flash-decoding attention (`fattn-sm60`), sm_60 quantized GEMVs and an IQ3_XXS fp16-table GEMV (`mmvh-sm60`, `mmvq` paths), Q3_K/Q4_K/Q6_K and output-head kernels, prefill dequantization and chunked GEMM, norm fusions, and the Qwen3.5-architecture graph changes they need.
- **The PN weight types** `pn3t` (124) and `pn4t` (125), with their CPU reference, CUDA dequantization and sm_60 GEMV kernels (`ggml-cuda/pn-sm60.cu`). The format is specified in tensorless `docs/pn-format.md`.
- **The PN prefill path** (from engine-v0.2.0, on by default; `ggml-cuda/pn-gemm-sm60.cu`): what a batched PN `MUL_MAT` (more than 4 tokens, so every prompt ubatch) runs instead of dequantize-to-fp32 plus cuBLAS SGEMM.
  - An exact-order fp32 GEMM for the m = 5120 and 12288 shapes, and a cuBLASLt route for the other large shapes. Both give the same bits as the cuBLAS call they replace.
  - A half2 (fp16) GEMM with fp32 accumulation across windows for the weights of blocks 31 and up, except `attn_k` and `attn_v`, and a shorter-chain precision tier for blocks 27 to 30 in the ubatches that are neither a prompt's first 512 tokens nor its final ubatch.
  - Vectorized PN dequantization, to fp32 and to fp16, prefetched on a side stream while the previous GEMM runs.
  - A fused SwiGLU epilogue for the FFN gate/up pair, and smaller changes to concat, RMS norm, the gated delta net and the flash-attention tile on the batched path.
  - In `llama_context::decode`, two consecutive prompt batches with no outputs run as one ubatch of up to twice `n_ubatch` tokens.
  - For T = 1 decode, `ssm_beta` and `ssm_alpha` load as views of one tensor and run as one GEMV.
- `tools/tf-logits` (teacher-forced logits) and `tests/test-pn-format.cpp`.

The PN kernels are the result of two automated hill climbs, each holding the model's numerics against its starting point:
- **Decode** (engine-v0.1.0): 131 judged attempts, 28 confirmed improvements, PN3M19 decode on one P100 from 7.74 to about 30 tok/s at short context.
- **Prefill** (engine-v0.2.0): 369 attempts, 26 promoted improvements, PN3M19 prefill on one P100 from 121.4 to 200.1 tok/s on a cold prompt of about 4,100 tokens, with the decode held within 1% of its starting point.

## New in engine-v0.3.0: North-Mini-Code-1.0 decode

The result of an automated decode hill climb on CohereLabs' North-Mini-Code-1.0 (cohere2moe: 128 experts, 8 active, a 4,096-token sliding window on 36 of 49 layers) in Unsloth's UD-IQ3_XXS file, on one P100. Its judge measured greedy decode after a cold 4,096- and 32,768-token code prompt, and held every change's teacher-forced output against the starting point's own (mean KL at most 0.003 nat, top-1 agreement at least 98%).

- **The starting point** (engine-v0.2.0 plus one fix): the sliding-window KV cache keeps room for two ubatches, so a prompt longer than the window no longer streams nothing (`src/llama-kv-cache-iswa.cpp`).
- **The climb:** 127 judged submissions, 22 confirmed improvements, decode from 42.03 to 72.62 tok/s (+74.25%, the product of the confirmed gains; 49.28 to 79.77 after a 4K prompt, 35.83 to 66.06 after 32K), sampled decode +61.7%, prefill unchanged. 0 of the judge's greedy replies changed.
- **What changed,** all on the CUDA path and on by default for this model (switches only turn them off):
  - flash attention for one decoded token with a q8_0 KV cache at head size 128: a 16-lane KQ mapping, one K load shared by the two query heads of a group (and a four-head path for long contexts), a transposed reduction, whole-row KQ, and the KV split count chosen by context length (`fattn-vec.cuh`, `fattn-common.cuh`);
  - the Q6_K attention projections: rows per warp from the device fill, a fused activation prep, a prep cache shared by Q, K and V, a load-time merge of Q, K and V into one GEMV, and the router GEMV as tail CTAs of that launch (`mmvh-sm60.cu`; `GGML_CUDA_MMV_SM60_QKV_ROUTER=0` turns the last off);
  - the experts: IQ3_S rows owned by a warp for the down projection, a leaner IQ2_S dot product for gate and up (`GGML_CUDA_IQ2S_NO_DIET` at build time turns it off), one q8_1 quantization shared by gate and up, and SwiGLU writing q8_1 for the down projection (`mmvq.cu`, `vecdotq.cuh`, `unary.cu`; `GGML_CUDA_SWIGLU_Q8_OFF=1` turns the last off);
  - the MoE tail (router weights, sum and the residual add) as one kernel (`moe-weighted-reduction.cu`), and the Q and K RoPE fused with the q8_0 KV cache write (`rope.cu`);
  - graph order in `src/models/cohere2moe.cpp` and contiguous one-token Q/K/V views in `src/llama-graph.cpp`, so the fusions above can match.
- **Output over long contexts:** below about 8,192 tokens of context the output is bit-identical to the starting point's. Above it, kernels chosen by context length change the summation order, and the output differs from the starting point's inside the climb's limits.
- **Rebuilt from the run's export:** the 17-patch attempt chain from the starting point to the champion, checked blob for blob against every patch's index lines, then scanned for lab-only and private terms (0 hits).

## From engine-v0.2.0

### What the export changed

Research-only files were left out: `.auto/`, `v2-lane.env`, `verdict.txt`, `HISTORY-MAP.tsv` (9 files: autoresearch state, the research lane's switch file, a status note, a commit map).

Comment lines that pointed into the private research repository were rewritten, one line for one line, so line numbers and compiled code are unchanged:
- `ggml/src/ggml-cuda/fattn-sm60.cu:2`
- `ggml/src/ggml-cuda/mmvh-sm60.cu:2`
- `ggml/src/ggml-cuda/mmvh-sm60.cuh:1`
- `ggml/src/ggml-cuda/pn-sm60.cu:3`
- `ggml/src/ggml-quants.c:169`
- `tests/test-pn-format.cpp:3`

### Changed after the export

One fix was made in this repository, on top of the exported research commit. It is not part of what the hill climb's judge measured:

- **A held prompt batch reported the wrong minimum position** (`src/llama-context.cpp`, with `seq_pos_min_is_last()` in `src/llama-memory.h`, `llama-memory-recurrent.h`, `llama-memory-hybrid.h` and `llama-memory-hybrid-iswa.h`). While `llama_context::decode` holds a prompt batch, `llama_memory_seq_pos_min` has to answer for it. It answered with the batch's first position. A memory with a recurrent state reports its last position there, and `llama-server` records its context checkpoints with that value, so a checkpoint came out one token short. The next request that reused the prompt cache then asked for a rollback the recurrent state cannot do, and the server aborted with `failed to remove sequence`. The judge sent every prompt with `cache_prompt` false, so no judged run reached it. The fix reports the position the memory will report once the batch is applied.

## Changed after the import (engine-v0.3.0)

- **A load with no GPU aborted** (`src/models/cohere2moe.cpp`): the load-time merge of a layer's Q, K and V weights into one tensor made them views of it, and on the CPU the weights are mapped from the file tensor by tensor, where a view can't be mapped (`GGML_ASSERT(tensor->view_src == NULL)` in `ggml_backend_tensor_alloc`). The merge now happens only for a layer on a GPU, so a layer on the CPU loads its three weights as before. On the P100 every layer is on the GPU, so what the judge measured is unchanged. Checked with the release's image and no GPU: the model loads and answers.
- **A guard for the router CTAs** (`ggml/src/ggml-cuda/mmvh-sm60.cu`, `try_q6k`): the QKV-with-router launch now refuses, and its caller runs both nodes alone, when the activation pointer it would give the router is null. The checks before it already hold the router to the one-column case, so the branch is never taken and the output is unchanged; it keeps a later change to the fused-prep condition from handing the router a null pointer.

The upstream license is in `LICENSE`; tensorless's own changes are Apache-2.0 (see the repository's `NOTICE`).

## Known issue

The CPU backend's PN dot product (`ggml_vec_dot_pn_t_q8_0` in `ggml-cpu/quants.c`) shifts by element index instead of pair index and decodes most PN weights wrong. GPU paths are correct and every tensorless recipe runs the whole model on the GPU; do not run PN models on the CPU backend or with partial offload until it is fixed.

## Inkling-2-Mini (inkling2_loop) from LookAtMyQuant's packed format

- `ggml/include/ggml-inkling2.h`, `tools/inkling-dump/check_rows.py`: ports of LookAtMyQuant's
  `runtime_cuda/inkling/formats.py` and `format_metadata.py` decoders (github.com/ShinMegamiBoson/lookatmyquant, MIT).
- `src/models/inkling2-loop.cpp`: written for this tree after transformers' `modeling_inkling2_loop.py` (Apache-2.0) and
  the design of upstream llama.cpp's `src/models/inkling2.cpp` / `inkling2-loop.cpp` (MIT): slot numbering, depth mix,
  message gates, convolutions with recurrent state, the sink folded into the bias table.
- `tests/inkling2/reference_logits.py` wraps LookAtMyQuant's `reference.py` with the embedding kept in float32.
- `ggml_flash_attn_ext_add_indexed_bias` (`ggml.c`, CPU `ops.cpp`, CUDA TILE kernel): the contract of upstream llama.cpp
  PR 32740 (MIT), F32 table [n_bins, n_head, T] and I32 index [n_kv, T]; inkling2_loop feeds it its sink-folded relative
  bias table and the relative-position index, so attention runs as flash attention without a dense KQ bias.
