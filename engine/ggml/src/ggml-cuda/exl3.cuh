#pragma once
// EXL3 format and trellis GEMV design: Turboderp, exllamav3 (https://github.com/turboderp-org/exllamav3), MIT License,
// Copyright (c) 2025 Turboderp; see licenses/LICENSE-exllamav3. Independent ggml/CUDA reimplementation.
#include "common.cuh"

static __device__ __forceinline__ uint32_t exl3_mul1_sum(uint32_t code) {
    const uint32_t x = code * 0x83DCD12Du;
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA) || __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
    return __dp4a(x, 0x01010101u, 0x6400u);
#else
    // SM60 has no dp4a, but its packed-video SAD instruction sums four
    // unsigned byte differences in one operation.  Against zero this is the
    // same byte sum as dp4a(x, 0x01010101), without the scalar mask/shift/add
    // chain in every trellis-code decode.
    return __vsadu4(x, 0u) + 0x6400u;
#endif
}

// Parts that lack a byte-dot-product accumulator (i.e. everything before dp4a) want the *un*biased byte sum:
// the trellis GEMV packs two sums into one half2 and sets the 0x6400 codebook bias for both halves at once.
// A raw byte sum is at most 4*255 = 0x3FC, so bits 10..15 are clear and OR-ing is exact addition.
#define EXL3_MUL1_BIAS_IS_PACKED (!defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ < GGML_CUDA_CC_DP4A)

static __device__ __forceinline__ uint32_t exl3_mul1_rawsum(uint32_t code) {
    const uint32_t x = code * 0x83DCD12Du;
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA) || __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
    return __dp4a(x, 0x01010101u, 0u);
#else
    return __vsadu4(x, 0u);
#endif
}

// Pack two raw mul1 byte sums into the fp16 pair (1024 + sum) that the codebook affine transform consumes.
static __device__ __forceinline__ half2 exl3_mul1_pair_from_rawsum(const uint32_t (&sum)[2]) {
    // Both branches feed the codebook affine the fp16 *bit pattern* 0x6400 + sum, never the integer converted to
    // half; the sums are only reinterpreted here.
    const half2 raw = __halves2half2(__ushort_as_half((unsigned short) sum[0]), __ushort_as_half((unsigned short) sum[1]));
#if EXL3_MUL1_BIAS_IS_PACKED
    const uint32_t bits = *reinterpret_cast<const uint32_t *>(&raw) | 0x64006400u;
    return *reinterpret_cast<const half2 *>(&bits);
#else
    return raw;
#endif
}

// 128-point Walsh-Hadamard in Sylvester order (== the file's exl3_had128.weight up to the 1/sqrt(128) scale),
// one warp per 128-chunk, lane holds indices j*32 + lane (j = 0..3): bits 5,6 in registers, bits 0..4 via shfl.
static __device__ __forceinline__ void exl3_wht128(float (&v)[4]) {
    const float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
    v[0] = a0 + a2; v[1] = a1 + a3; v[2] = a0 - a2; v[3] = a1 - a3;
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? (o - v[j]) : (v[j] + o);
        }
    }
}
#define EXL3_WHT128_SCALE 0.08838834764831845f   // 1/sqrt(128)


// EXL3 (exllamav3 trellis) weights. M1 reference path: reconstruct the whole [K, N] weight to f16 in pool memory
// (codebook values only) and run a cuBLAS GEMM. The .suh/.svh vectors and the 128-block Hadamard are graph ops
// around the mul_mat (llama build_lora_mm). Fused decode GEMV/GEMM come in later milestones.

// dst: compact f16 [output_count][K] row-major, codebook values, no suh/svh/Hadamard
void ggml_cuda_exl3_reconstruct_f16(const ggml_tensor * src0, half * dst, int64_t output_start, int64_t output_count, cudaStream_t stream);
void ggml_cuda_exl3_reconstruct_f32(const ggml_tensor * src0, float * dst, int64_t output_start, int64_t output_count, cudaStream_t stream);

// M2: trellis-direct GEMV/GEMM for T <= 8 activation columns (env GGML_CUDA_EXL3_GEMV=0 forces the M1 path)
bool ggml_cuda_exl3_gemv_supported(const int64_t T);
int  ggml_cuda_exl3_gemv_ksplit(const int kt, const int nt, const int T);
// suh/svh may be null (no Hadamard/scale glue)
void ggml_cuda_exl3_gemv(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const int64_t T,
                         const float * suh, const float * svh);

// out[T][K] (f16) = scale * H128(suh * x)  (suh null: scale * x)
void ggml_cuda_exl3_glue_in_f16(const float * x, const float * suh, half * out, const int K, const int64_t T, const float scale, cudaStream_t stream);
void ggml_cuda_exl3_glue_in_f16_checked(const float * x, const float * suh, half * out, int * f16_error,
                                        const int K, const int64_t T, const float scale, cudaStream_t stream);
void ggml_cuda_exl3_glue_in_f32(const float * x, const float * suh, float * out, const int K, const int64_t T, const float scale, cudaStream_t stream);
// y[T][N] = scale * svh * H128( sum_ks part[ks][T][N] )  (svh null: scale * sum)
void ggml_cuda_exl3_glue_out(const float * part, const float * svh, float * y, const int N, const int64_t T, const int ksplit, const float scale, cudaStream_t stream);
void ggml_cuda_exl3_glue_out_slice(const float * part, const float * svh, float * y, const int N, const int output_start, const int output_count,
                                   const int64_t T, const int ksplit, const float scale, cudaStream_t stream);

void ggml_cuda_mul_mat_exl3(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
