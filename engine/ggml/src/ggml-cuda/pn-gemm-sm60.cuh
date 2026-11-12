#pragma once

#include "common.cuh"

// P100 (sm_60) exact-order fp32 SGEMM for the batched PN MUL_MAT (pn-gemm-sm60.cu). Applies only to PN src0
// (fp32-dequantized into the scratch buffer), m = 5120, n = 512, and k a multiple of 16 with contiguous
// rows; the caller keeps cuBLAS for everything else.
bool ggml_cuda_pn_gemm_sm60_applies(ggml_type src0_type, int64_t m, int64_t n, int64_t k, int64_t s01, int64_t s11, int64_t ldc);

// C (m x n, column-major, ldc = m) = A (m x k, row-major, row stride k) * B^T (B is n x k, row stride k).
void ggml_cuda_pn_gemm_sm60(cudaStream_t stream, const float * A, const float * B, float * C, int64_t m, int64_t n, int64_t k);

// Depth-gated half2 GEMM (fp16 operands, half2 partial sums over 16 products, fp32 across chunks) for the batched PN
// MUL_MAT of the last blocks, excluding attn_k and attn_v; name is the weight's tensor name ("blk.N.<tensor>.weight").
bool ggml_cuda_pn_h2_sm60_applies(ggml_type src0_type, const char * name, int64_t m, int64_t n, int64_t k,
                                  int64_t s01, int64_t s11, int64_t ldc);
// seg: the weight's chain depth from ggml_cuda_pn_h2_sm60_seg(name) (0 = the 32-deep chains).
void ggml_cuda_pn_h2_sm60(cudaStream_t stream, const float * A, const half * B, float * C, int64_t m, int64_t n, int64_t k, int seg);
// The same GEMM on fp16 weights: A holds the values ggml_cuda_pn_h2_sm60 rounds to half when it stages fp32 A, so the
// result is bit-identical. The side stream writes them with ggml_cuda_pn_to_fp16 (convert.cu).
void ggml_cuda_pn_h2_sm60_f16(cudaStream_t stream, const half * A, const half * B, float * C, int64_t m, int64_t n, int64_t k, int seg);
// The h2 GEMM as one side of a SWIGLU pair: C = silu(gate) * up, where this GEMM's result is the gate (mine_is_gate) or
// the up, and the other side's fp32 result is `other` (same layout as C). Same GEMM bits; the epilogue replaces the store.
void ggml_cuda_pn_h2_sm60_glu(cudaStream_t stream, const float * A, const half * B, const float * other, float * C,
                              bool mine_is_gate, int64_t m, int64_t n, int64_t k, int seg);
void ggml_cuda_pn_h2_sm60_glu_f16(cudaStream_t stream, const half * A, const half * B, const float * other, float * C,
                                  bool mine_is_gate, int64_t m, int64_t n, int64_t k, int seg);
// Chain depth of the h2 GEMM for the weight `name` ("blk.N.<tensor>.weight"): 0, or the tier's 8 or 16 (GGML_PN_TIER_*).
int ggml_cuda_pn_h2_sm60_seg(const char * name);
// The ubatch the next graph computes: pos0 is its first token's position, is_last says it is the prompt's final ubatch.
// The tier's position rule reads it (pn-gemm-sm60.cu); llama_context calls it through the backend's proc address.
void ggml_cuda_pn_tier_set_ubatch(int64_t pos0, bool is_last);
// The sink split (the tier's columns 512..1023 of the paired sink ubatch, pos0 0): the columns of the chain route, and the
// chain depth of the tier's GEMM over the rest. Returns 0 when the split does not apply to this batched MUL_MAT.
#define PN_TIER_SINK_COLS 512
int ggml_cuda_pn_sink_split_seg(ggml_type src0_type, const char * name, int64_t m, int64_t n, int64_t k, int64_t s01, int64_t s11, int64_t ldc);
// The B layout depends on the tile the GEMM of (m, n) runs with, so the allocation, the conversion and the launch take m.
int64_t ggml_cuda_pn_h2_b_elems(int64_t m, int64_t n, int64_t k);
void ggml_cuda_pn_b_to_fp16(cudaStream_t stream, const float * B, half * Bt, int64_t m, int64_t n, int64_t k);
void ggml_cuda_pn_to_fp16(ggml_type type, const void * x, half * y, int64_t k, cudaStream_t stream);

// cuBLASLt route for the batched PN MUL_MAT at m >= 6144 and n >= 256 (algorithm 0, 128x128 tile, CTA swizzle 1, split-K 1).
// Returns false without computing when it does not apply or cuBLASLt refuses the algorithm; the caller then runs cublasSgemm.
bool ggml_cuda_pn_cublaslt_sm60(cudaStream_t stream, ggml_type src0_type, const float * A, const float * B, float * C,
                                int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc);
