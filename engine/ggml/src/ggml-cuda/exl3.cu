// EXL3 format, "mul1" codebook arithmetic, tile layout and trellis GEMV design: Turboderp, exllamav3
// (https://github.com/turboderp-org/exllamav3), MIT License, Copyright (c) 2025 Turboderp; see
// licenses/LICENSE-exllamav3. EXL3 is Turboderp's streamlined variant of QTIP (Tseng, Sun, Hou, De Sa,
// "QTIP: Quantization with Trellises and Incoherence Processing", NeurIPS 2024, arXiv:2406.11235).
// This file is an independent ggml/CUDA reimplementation written with the exllamav3 source as reference;
// no QTIP code was used.
#include "exl3.cuh"

#include <type_traits>
#include "convert.cuh"

// exllamav3 "mul1" codebook (cb2, exl3_dq.cuh): x = code * 0x83DCD12D; s = bytesum(x) + 0x6400 as an fp16 bit
// pattern; value = hfma(s, fp16(0x1eee), fp16(0xc931)) -- one rounding.
static __device__ __forceinline__ half exl3_mul1_decode(uint32_t code) {
    const uint32_t s = exl3_mul1_sum(code);
    const half h = __ushort_as_half((unsigned short) s);
    return __hfma(h, __ushort_as_half((unsigned short) 0x1eee), __ushort_as_half((unsigned short) 0xc931));
}

// one block (256 threads) per 16x16 tile, one thread per weight; tile order is the native trellis order
// [K/16][N/16][8*bits words] (k-tile major).
template <typename OUT>
static __global__ void k_exl3_reconstruct(const uint32_t * __restrict__ trellis, OUT * __restrict__ W,
                                          const int kt, const int nt, const int output_tile_start, const int output_tile_count,
                                          const int bits) {
    __shared__ uint32_t words[64];
    const int tile = blockIdx.x;
    const int ki = tile / output_tile_count;
    const int ni_local = tile - ki * output_tile_count;
    const int ni = output_tile_start + ni_local;
    const int nw = 8 * bits;
    const uint32_t * src = trellis + ((size_t) ki * nt + ni) * nw;
    if (threadIdx.x < (unsigned) nw) {
        words[threadIdx.x] = src[threadIdx.x];
    }
    __syncthreads();
    const int t  = threadIdx.x;
    const int b0 = ((t + 257) * bits - 16) % (256 * bits);
    const int i0 = b0 >> 5;
    const int sh = b0 & 31;
    const uint32_t a = words[i0];
    const uint32_t b = words[(i0 + 1) % nw];
    const uint32_t code = (uint32_t) (((((uint64_t) a) << 32) | (uint64_t) b) >> (48 - sh)) & 0xffffu;
    const int lane = t >> 3;
    const int j    = t & 7;
    const int row  = 2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1);      // k within tile
    const int col  = 2 * (lane >> 3) + ((lane >> 2) & 1) + 8 * (j >> 2);  // n within tile
    const size_t K = (size_t) kt * 16;
    const half value = exl3_mul1_decode(code);
    if constexpr (std::is_same_v<OUT, half>) {
        W[((size_t) ni_local * 16 + col) * K + (size_t) ki * 16 + row] = value;
    } else {
        W[((size_t) ni_local * 16 + col) * K + (size_t) ki * 16 + row] = __half2float(value);
    }
}

template <typename OUT>
static void exl3_reconstruct(const ggml_tensor * src0, OUT * dst, int64_t output_start, int64_t output_count, cudaStream_t stream) {
    const int bits = ggml_exl3_bits(src0->type);
    GGML_ASSERT(bits != 0);
    GGML_ASSERT(src0->ne[0] % 16 == 0 && src0->ne[1] % 16 == 0 && src0->ne[2] == 1 && src0->ne[3] == 1);
    GGML_ASSERT(output_start >= 0 && output_count > 0 && output_start + output_count <= src0->ne[1]);
    GGML_ASSERT(output_start % 16 == 0 && output_count % 16 == 0);
    const int kt = (int) (src0->ne[0] / 16);
    const int nt = (int) (src0->ne[1] / 16);
    const int output_tile_start = (int) (output_start / 16);
    const int output_tile_count = (int) (output_count / 16);
    k_exl3_reconstruct<OUT><<<kt * output_tile_count, 256, 0, stream>>>(
        (const uint32_t *) src0->data, dst, kt, nt, output_tile_start, output_tile_count, bits);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_exl3_reconstruct_f16(const ggml_tensor * src0, half * dst, int64_t output_start, int64_t output_count, cudaStream_t stream) {
    exl3_reconstruct(src0, dst, output_start, output_count, stream);
}

void ggml_cuda_exl3_reconstruct_f32(const ggml_tensor * src0, float * dst, int64_t output_start, int64_t output_count, cudaStream_t stream) {
    exl3_reconstruct(src0, dst, output_start, output_count, stream);
}

// ---- fused glue: y = svh * H128( W . H128( suh * x ) )  (exllamav3 exl3.py forward, Hadamard block 128) ----
static __device__ __forceinline__ void exl3_store(float * p, float v, int *) { *p = v; }
static __device__ __forceinline__ void exl3_store(half * p, float v, int * f16_error) {
    const half h = __float2half_rn(v);
    if (f16_error != nullptr) {
        const float restored = __half2float(h);
        const float allowed = fmaxf(1e-3f, fabsf(v) * 5e-4f);
        if (!isfinite(v) || !isfinite(restored) || fabsf(v - restored) > allowed) {
            atomicExch(f16_error, 1);
        }
    }
    *p = h;
}

// out[T][K] = scale * H128(suh * x)   (HAD)  |  scale * x   (!HAD);  x, out contiguous [T][K], K % 128 == 0
template <typename OUT, bool HAD>
static __global__ void k_exl3_glue_in(const float * __restrict__ x, const float * __restrict__ suh, OUT * __restrict__ out,
                                      int * __restrict__ f16_error, const int K, const size_t n_chunks, const float scale) {
    const size_t c = (size_t) blockIdx.x * 4 + (threadIdx.x >> 5);
    if (c >= n_chunks) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const size_t base = c * 128;
    const int kc = (int) (base % (size_t) K);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = x[base + j * 32 + lane];
        if (HAD) {
            v[j] *= suh[kc + j * 32 + lane];
        }
    }
    float s = scale;
    if (HAD) {
        exl3_wht128(v);
        s *= EXL3_WHT128_SCALE;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        exl3_store(out + base + j * 32 + lane, v[j] * s, f16_error);
    }
}

// y[T][N] = scale * svh * H128( sum_ks part[ks][T][N] )   (HAD)  |  scale * sum_ks part   (!HAD);  N % 128 == 0
template <bool HAD>
static __global__ void k_exl3_glue_out(const float * __restrict__ part, const float * __restrict__ svh, float * __restrict__ y,
                                       const int N, const int output_start, const int output_count, const size_t n_chunks,
                                       const int ksplit, const size_t ks_stride, const float scale) {
    const size_t c = (size_t) blockIdx.x * 4 + (threadIdx.x >> 5);
    if (c >= n_chunks) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const size_t base = c * 128;
    const int nc = (int) (base % (size_t) output_count);
    const int t = (int) (base / (size_t) output_count);
    // sum the split-K partials with four rows in flight (the serial chain was latency-bound at width > 1);
    // summation order is fixed => deterministic
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int ks = 0;
    for (; ks + 4 <= ksplit; ks += 4) {
        const float * p = part + (size_t) ks * ks_stride + base + lane;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float a0 = p[j * 32];
            const float a1 = p[ks_stride + j * 32];
            const float a2 = p[2 * ks_stride + j * 32];
            const float a3 = p[3 * ks_stride + j * 32];
            v[j] += (a0 + a1) + (a2 + a3);
        }
    }
    for (; ks < ksplit; ++ks) {
        const float * p = part + (size_t) ks * ks_stride + base + lane;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j] += p[j * 32];
        }
    }
    float s = scale;
    if (HAD) {
        exl3_wht128(v);
        s *= EXL3_WHT128_SCALE;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int n = nc + j * 32 + lane;
        const float m = HAD ? svh[output_start + n] : 1.0f;
        y[(size_t) t * N + output_start + n] = v[j] * s * m;
    }
}

template <typename OUT>
static void exl3_glue_in(const float * x, const float * suh, OUT * out, int * f16_error,
                         const int K, const int64_t T, const float scale, cudaStream_t stream) {
    GGML_ASSERT(K % 128 == 0);
    const size_t n_chunks = (size_t) T * K / 128;
    const int nb = (int) ((n_chunks + 3) / 4);
    if (suh) {
        k_exl3_glue_in<OUT, true ><<<nb, 128, 0, stream>>>(x, suh, out, f16_error, K, n_chunks, scale);
    } else {
        k_exl3_glue_in<OUT, false><<<nb, 128, 0, stream>>>(x, suh, out, f16_error, K, n_chunks, scale);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_exl3_glue_in_f16(const float * x, const float * suh, half * out, const int K, const int64_t T, const float scale, cudaStream_t stream) {
    exl3_glue_in<half>(x, suh, out, nullptr, K, T, scale, stream);
}

void ggml_cuda_exl3_glue_in_f16_checked(const float * x, const float * suh, half * out, int * f16_error,
                                        const int K, const int64_t T, const float scale, cudaStream_t stream) {
    exl3_glue_in<half>(x, suh, out, f16_error, K, T, scale, stream);
}

void ggml_cuda_exl3_glue_in_f32(const float * x, const float * suh, float * out, const int K, const int64_t T, const float scale, cudaStream_t stream) {
    exl3_glue_in<float>(x, suh, out, nullptr, K, T, scale, stream);
}

void ggml_cuda_exl3_glue_out(const float * part, const float * svh, float * y, const int N, const int64_t T, const int ksplit, const float scale, cudaStream_t stream) {
    ggml_cuda_exl3_glue_out_slice(part, svh, y, N, 0, N, T, ksplit, scale, stream);
}

void ggml_cuda_exl3_glue_out_slice(const float * part, const float * svh, float * y, const int N, const int output_start, const int output_count,
                                   const int64_t T, const int ksplit, const float scale, cudaStream_t stream) {
    GGML_ASSERT(N % 128 == 0);
    GGML_ASSERT(output_start >= 0 && output_count > 0 && output_start + output_count <= N);
    GGML_ASSERT(output_start % 128 == 0 && output_count % 128 == 0);
    const size_t n_chunks = (size_t) T * output_count / 128;
    const size_t ks_stride = (size_t) T * output_count;
    const int nb = (int) ((n_chunks + 3) / 4);
    if (svh) {
        k_exl3_glue_out<true ><<<nb, 128, 0, stream>>>(part, svh, y, N, output_start, output_count, n_chunks, ksplit, ks_stride, scale);
    } else {
        k_exl3_glue_out<false><<<nb, 128, 0, stream>>>(part, svh, y, N, output_start, output_count, n_chunks, ksplit, ks_stride, scale);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_exl3(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_exl3_bits(src0->type) != 0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src1) && ggml_is_contiguous(dst));
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    GGML_ASSERT(src1->ne[0] == K && dst->ne[0] == N);
    const int64_t T = src1->ne[1] * src1->ne[2] * src1->ne[3];   // all activation columns (src0 is 2-D, broadcast)

    // fused glue (llama build_lora_mm): src[2] = suh [K], src[3] = svh [N]; absent -> codebook values only
    const ggml_tensor * suh_t = dst->src[2];
    const ggml_tensor * svh_t = dst->src[3];
    const float * suh = nullptr;
    const float * svh = nullptr;
    if (suh_t != nullptr) {
        GGML_ASSERT(svh_t != nullptr && suh_t->type == GGML_TYPE_F32 && svh_t->type == GGML_TYPE_F32);
        GGML_ASSERT(suh_t->ne[0] == K && svh_t->ne[0] == N && K % 128 == 0 && N % 128 == 0);
        GGML_ASSERT(ggml_is_contiguous(suh_t) && ggml_is_contiguous(svh_t));
        suh = (const float *) suh_t->data;
        svh = (const float *) svh_t->data;
    }

    if (ggml_cuda_exl3_gemv_supported(T)) {
        ggml_cuda_exl3_gemv(ctx, src0, src1, dst, T, suh, svh);
        return;
    }

    const int id = ggml_cuda_get_device();
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<half> x_f16(ctx.pool(id), K * T);
    ggml_cuda_pool_alloc<int> f16_error(ctx.pool(id), 1);
    CUDA_CHECK(cudaMemsetAsync(f16_error.get(), 0, sizeof(int), stream));
    ggml_cuda_exl3_glue_in_f16_checked((const float *) src1->data, suh, x_f16.get(), f16_error.get(), (int) K, T, 1.0f, stream);
    int f16_error_host = 0;
    CUDA_CHECK(cudaMemcpyAsync(&f16_error_host, f16_error.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const char * force_f32_env = getenv("GGML_CUDA_EXL3_FORCE_F32");
    const bool use_f32 = f16_error_host != 0 || (force_f32_env != nullptr && strcmp(force_f32_env, "1") == 0);

    ggml_cuda_pool_alloc<float> x_f32(ctx.pool(id));
    if (use_f32) {
        x_f32.alloc(K * T);
        ggml_cuda_exl3_glue_in_f32((const float *) src1->data, suh, x_f32.get(), (int) K, T, 1.0f, stream);
        static bool reported = false;
        if (!reported) {
            GGML_LOG_WARN("EXL3: using bounded F32 reconstruction fallback for transformed activations\n");
            reported = true;
        }
    }

    static constexpr size_t scratch_budget = 64 * 1024 * 1024;
    const size_t weight_element_size = use_f32 ? sizeof(float) : sizeof(half);
    const size_t bytes_per_output = (size_t) K * weight_element_size + (size_t) T * sizeof(float);
    int64_t output_chunk = (int64_t) (scratch_budget / bytes_per_output);
    output_chunk = std::min(output_chunk, N);
    output_chunk = output_chunk / 128 * 128;
    if (output_chunk < 128) {
        GGML_ABORT("EXL3: 64 MiB scratch budget cannot hold one 128-feature output block");
    }
    GGML_ASSERT((size_t) output_chunk * bytes_per_output <= scratch_budget);

    ggml_cuda_pool_alloc<half> w_f16(ctx.pool(id));
    ggml_cuda_pool_alloc<float> w_f32(ctx.pool(id));
    if (use_f32) {
        w_f32.alloc((size_t) K * output_chunk);
    } else {
        w_f16.alloc((size_t) K * output_chunk);
    }
    ggml_cuda_pool_alloc<float> y_slice(ctx.pool(id), (size_t) T * output_chunk);

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
    const int cc = ggml_cuda_info().devices[id].cc;
    const auto algorithm = cc <= GGML_CUDA_CC_PASCAL ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    const void * weight = use_f32 ? (const void *) w_f32.get() : (const void *) w_f16.get();
    const void * input = use_f32 ? (const void *) x_f32.get() : (const void *) x_f16.get();
    const cudaDataType_t input_type = use_f32 ? CUDA_R_32F : CUDA_R_16F;
    for (int64_t output_start = 0; output_start < N; output_start += output_chunk) {
        const int64_t output_count = std::min(output_chunk, N - output_start);
        if (use_f32) {
            ggml_cuda_exl3_reconstruct_f32(src0, w_f32.get(), output_start, output_count, stream);
        } else {
            ggml_cuda_exl3_reconstruct_f16(src0, w_f16.get(), output_start, output_count, stream);
        }

        // y[output_count x T] = W[output_count x K] . X[K x T], with F32 accumulation.
        CUBLAS_CHECK(
            cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                    (int) output_count, (int) T, (int) K,
                    &alpha, weight, input_type, (int) K,
                            input,  input_type, (int) K,
                    &beta,  y_slice.get(), CUDA_R_32F, (int) output_count,
                    CUBLAS_COMPUTE_32F, algorithm));
        ggml_cuda_exl3_glue_out_slice(y_slice.get(), svh, (float *) dst->data, (int) N, (int) output_start, (int) output_count,
                                      T, 1, 1.0f, stream);
    }
}
