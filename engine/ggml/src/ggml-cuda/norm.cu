#include "norm.cuh"
#include "scale.cuh"
#include "unary.cuh"
#include <cstdint>

template <int block_size>
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float2 mean_var = make_float2(0.0f, 0.0f);

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = (x[col] - mean) * inv_std;
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    const int start =     blockIdx.x*group_size + threadIdx.x;
    const int end   = min(blockIdx.x*group_size + group_size,  ne_elements);

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        const float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum + 32);

    const float variance = tmp / group_size;
    const float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

template <int block_size, bool do_multiply = false, bool do_add = false, bool do_pre_add = false>
static __global__ void rms_norm_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                    // SBB-19: fold the residual ADD into the prologue: x = pre_a + pre_b written to pre_dst
                                    // (the ADD's dst). One row maps to one block, so no cross-block sync is needed.
                                    const float * pre_a                = nullptr,
                                    const float * pre_b                = nullptr,
                                    float *       pre_dst              = nullptr) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_pre_add) {
        const int64_t pre_off = sample*stride_sample + channel*stride_channel + row*stride_row;
        pre_a   += pre_off;
        pre_b   += pre_off;
        pre_dst += pre_off;
    }

    if constexpr (do_add) {
        const int add_row     = fastmodulo(row, add_nrows_packed);
        const int add_channel = fastmodulo(channel, add_nchannels_packed);
        const int add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    // x is read twice (sum of squares, then scaling). For a single row the grid is 1 block, so keep the row in
    // registers when it fits and drop the second read. Only the <1024> instantiation sees ncols > block_size.
    constexpr int  max_cache = 8;
    constexpr bool do_cache  = block_size >= 1024;
    const bool cached = do_cache && ncols > block_size && ncols <= max_cache*block_size;
    float xv[max_cache];
    // Prefetch the norm weights in the same pass as x: the reduction between the two passes otherwise
    // serialises the two streams, and a single row is one block, so there is little else in flight.
    // Same values and same expression order, so the result is unchanged.
    float mulv[max_cache];

    ggml_cuda_pdl_sync();
    if (cached) {
#pragma unroll
        for (int k = 0; k < max_cache; ++k) {
            const int col = tid + k*block_size;
            float     xi  = 0.0f;
            if (col < ncols) {
                if constexpr (do_pre_add) {
                    xi           = pre_a[col] + pre_b[col];
                    pre_dst[col] = xi;   // always written: the next layer reads it as its residual
                } else {
                    xi = x[col];
                }
            }
            xv[k] = xi;
            tmp  += xi * xi;
            if constexpr (do_multiply) {
                mulv[k] = col < ncols ? mul[fastmodulo(col, mul_ncols_packed)] : 0.0f;
            }
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            float xi;
            if constexpr (do_pre_add) {
                xi           = pre_a[col] + pre_b[col];
                pre_dst[col] = xi;
            } else {
                xi = x[col];
            }
            tmp += xi * xi;
        }
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    auto store = [&](const int col, const float xi, const float mulf) {
        if constexpr (do_multiply && do_add) {
            const int add_col = fastmodulo(col, add_ncols_packed);
            dst[col]          = scale * xi * mulf + add[add_col];
        } else if constexpr (do_multiply) {
            dst[col]          = scale * xi * mulf;
        } else {
            dst[col] = scale * xi;
        }
    };

    if (cached) {
#pragma unroll
        for (int k = 0; k < max_cache; ++k) {
            const int col = tid + k*block_size;
            if (col < ncols) {
                if constexpr (do_multiply) {
                    store(col, xv[k], mulv[k]);
                } else {
                    store(col, xv[k], 0.0f);
                }
            }
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            if constexpr (do_multiply) {
                store(col, x[col], mul[fastmodulo(col, mul_ncols_packed)]);
            } else {
                store(col, x[col], 0.0f);
            }
        }
    }
}

// Width-specialized pre-add RMS for a plain norm weight: one weight vector of ncols that matches the row, so
// the broadcast index is the column itself. NS = ceil(ncols / block_size) is a compile-time slot count, so no
// unused slots, predicated-off loads or modulo address math are issued. The per-thread accumulation order,
// the block reduction and the output expression are those of rms_norm_f32 (cached path), so bits are unchanged.
template <int block_size, int NS>
static __global__ void rms_norm_f32_plain(const float * pre_a, const float * pre_b, float * pre_dst,
                                          const float * mul, float * dst, const int ncols,
                                          const int64_t stride_row, const int64_t stride_channel,
                                          const int64_t stride_sample, const float eps) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    const int64_t pre_off = sample*stride_sample + channel*stride_channel + row*stride_row;
    pre_a   += pre_off;
    pre_b   += pre_off;
    pre_dst += pre_off;

    float tmp = 0.0f; // partial sum for thread in warp
    float xv[NS];
    float mulv[NS];

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int k = 0; k < NS; ++k) {
        const int col = tid + k*block_size;
        float     xi  = 0.0f;
        if (col < ncols) {
            xi           = pre_a[col] + pre_b[col];
            pre_dst[col] = xi;   // always written: the next layer reads it as its residual
            mulv[k]      = mul[col];
        } else {
            mulv[k] = 0.0f;
        }
        xv[k] = xi;
        tmp  += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

#pragma unroll
    for (int k = 0; k < NS; ++k) {
        const int col = tid + k*block_size;
        if (col < ncols) {
            dst[col] = scale * xv[k] * mulv[k];
        }
    }
}

template <int block_size>
static __global__ void rms_norm_back_f32(
        const float * grad, const float * xf, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    grad += int64_t(row)*ncols;
    xf   += int64_t(row)*ncols;
    dst  += int64_t(row)*ncols;

    float sum_xx = 0.0f; // sum for squares of x, equivalent to forward pass
    float sum_xg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xfi = xf[col];
        sum_xx += xfi * xfi;
        sum_xg += xfi * grad[col];
    }

    // sum up partial sums
    sum_xx = warp_reduce_sum(sum_xx);
    sum_xg = warp_reduce_sum(sum_xg);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum_xx[32];
        __shared__ float s_sum_xg[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum_xx[warp_id] = sum_xx;
            s_sum_xg[warp_id] = sum_xg;
        }
        __syncthreads();

        sum_xx = s_sum_xx[lane_id];
        sum_xx = warp_reduce_sum(sum_xx);

        sum_xg = s_sum_xg[lane_id];
        sum_xg = warp_reduce_sum(sum_xg);
    }

    const float mean_eps = sum_xx / ncols + eps;
    const float sum_eps  = sum_xx + ncols*eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x    = -scale_grad * sum_xg/sum_eps;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale_grad*grad[col] + scale_x*xf[col];
    }
}

// template <int block_size>
// static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
//     const int row = blockIdx.x*blockDim.y + threadIdx.y;
//     const int tid = threadIdx.x;

//     float tmp = 0.0f; // partial sum for thread in warp

//     for (int col = tid; col < ncols; col += block_size) {
//         const float xi = x[row*ncols + col];
//         tmp += xi * xi;
//     }

//     // sum up partial sums
//     tmp = warp_reduce_sum(tmp);
//     if (block_size > WARP_SIZE) {
//         __shared__ float s_sum[32];
//         int warp_id = threadIdx.x / WARP_SIZE;
//         int lane_id = threadIdx.x % WARP_SIZE;
//         if (lane_id == 0) {
//             s_sum[warp_id] = tmp;
//         }
//         __syncthreads();
//         tmp = s_sum[lane_id];
//         tmp = warp_reduce_sum(tmp);
//     }

//     // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//     const float scale = rsqrtf(fmaxf(tmp, eps * eps));

//     for (int col = tid; col < ncols; col += block_size) {
//         dst[row*ncols + col] = scale * x[row*ncols + col];
//     }
// }

template <int block_size>
static __global__ void l2_norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    ggml_cuda_pdl_lc();

    // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
    const float scale = rsqrtf(fmaxf(tmp, eps * eps));

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col];
    }
}

static void norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

static void group_norm_f32_cuda(
        const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, block_dims.x > WARP_SIZE ? 2 * 32 * sizeof(float): 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

// Grid-wide barrier (NORM-EMITS-BARRIER): sense-reversing generation counter, self-resetting,
// no extra launch. Safe only when every CTA is resident (all A1 norm grids are 20-48 CTAs on
// 56 SMs; the detector gates nrows*(ncols/256)*nch*ns <= 48). Bitwise proven in
// kernels/norm-blockshape/norm_barrier.cu at 5120/6144 x nrows 1/2 with pre_dst aliased.
__device__ unsigned int g_emit_bar_count = 0;
__device__ volatile unsigned int g_emit_bar_gen = 0;
static __device__ __forceinline__ void emit_grid_barrier(unsigned int total) {
    __syncthreads();  // all load-phase work of this CTA done
    if (threadIdx.x == 0) {
        // read the generation BEFORE arriving: if the last CTA bumps gen in the gap between our
        // arrival and a later gen read, we would spin on the new generation forever (cg grid.sync)
        const unsigned int gen = g_emit_bar_gen;
        __threadfence();
        const unsigned int arrived = atomicAdd(&g_emit_bar_count, 1u);
        if (arrived == total - 1) {
            g_emit_bar_count = 0;
            __threadfence();
            atomicAdd((unsigned int *) &g_emit_bar_gen, 1u);
        } else {
            while (g_emit_bar_gen == gen) { }
            __threadfence();
        }
    }
    __syncthreads();
}

// NORM-EMITS-STEP2 v4: the pre_add norm with an in-producer q8_1 emit (wide256 geometry).
// NV*1024 == ncols cached lanes (NV <= 8), N = ncols/256 CTAs per row; each recomputes the
// full-row ladder (identical block_reduce_sum, so the scale is bit-identical), writes its own
// 256-col slice of pre_dst/dst, and emits its slice's q8_1 groups with the engine's exact
// quantize op sequence (warp holds one contiguous 32-aligned group). Bitwise proven in
// kernels/norm-blockshape/norm_emits2.cu at nrows 1-4, both byte orders. `barrier` defers every
// pre_dst/dst write behind a grid-wide barrier: the wide CTAs race if anything the kernel writes
// aliases anything a sibling still reads (the add in place, or the allocator placing the MUL dst
// over a dead add input) - the A1 lane reality at every site (NORM-EMITS-BARRIER journal).
template <int NV, bool SWIZZLE>
static __global__ void rms_norm_f32_emit(const float * x,
                                         float *       dst,
                                         const int     ncols,
                                         const int64_t stride_row,
                                         const int64_t stride_channel,
                                         const int64_t stride_sample,
                                         const float   eps,
                                         const float * mul,
                                         const int64_t mul_stride_row,
                                         const int64_t mul_stride_channel,
                                         const int64_t mul_stride_sample,
                                         const uint3   mul_ncols_packed,
                                         const uint3   mul_nrows_packed,
                                         const uint3   mul_nchannels_packed,
                                         const uint3   mul_nsamples_packed,
                                         const float * pre_a,
                                         const float * pre_b,
                                         float *       pre_dst,
                                         block_q8_1 * q8,
                                         const int     npad,
                                         const int     barrier) {
    constexpr int SLICE = 256;
    constexpr int N     = NV * 4;
    const int part = blockIdx.x % N;
    const int row  = blockIdx.x / N;
    const int nrows_total = gridDim.x / N;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    const int tid = threadIdx.x;
    constexpr int LANESP  = 1024 / SLICE;
    const int     p0      = part * SLICE;
    const int     lane    = p0 / 1024;
    const int     tstart  = p0 % 1024;

    const int64_t xoff = sample*stride_sample + channel*stride_channel + row*stride_row;
    x       += xoff;
    pre_a   += xoff;
    pre_b   += xoff;
    pre_dst += xoff;
    dst     += ((int64_t)(sample*gridDim.y + channel)*nrows_total + row)*ncols;
    // broadcast dimensions are handled with the same fastmodulo pattern as the live rms_norm_f32
    const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
    const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
    const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
    mul     += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    block_q8_1 * yrow = q8 + ((int64_t)(sample*gridDim.y + channel)*nrows_total + row) * (npad/32);

    ggml_cuda_pdl_sync();
    float tmp = 0.0f;
    float xv[NV];
    float mulv[NV];
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        const int col = tid + k*1024;
        const float xi = pre_a[col] + pre_b[col];
        xv[k] = xi;
        tmp  += xi * xi;
        mulv[k] = mul[fastmodulo(col, mul_ncols_packed)];
    }

    if (barrier) {
        emit_grid_barrier(gridDim.x * gridDim.y * gridDim.z);
    }
    if (tid >= tstart && tid < tstart + SLICE) {
        pre_dst[p0 + (tid - tstart)] = xv[lane];
    }

    extern __shared__ float s_sum_emit[];
    tmp = block_reduce<block_reduce_method::SUM, 1024>(tmp, s_sum_emit);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    if (tid >= tstart && tid < tstart + SLICE) {
        const int col = p0 + (tid - tstart);
        const float dv = scale * xv[lane] * mulv[lane];
        dst[col] = dv;
        const int g_local = (tid - tstart) / 32;
        const int iqs     = (tid - tstart) % 32;
        float amax = fabsf(dv);
        amax = warp_reduce_max(amax);
        const float  d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : roundf(dv / d);
        const int g = part * (SLICE/32) + g_local;
        if constexpr (SWIZZLE) {
            yrow[g].qs[2*(iqs % 16) + iqs/16] = q;
        } else {
            yrow[g].qs[iqs] = q;
        }
        const float sum = warp_reduce_sum((float) q);
        if (tid % 32 == 0) {
            yrow[g].ds = make_half2(d, d * sum);
        }
    }
}

// V3-19: one warp per row (8 rows per 256-thread block) for the per-head prefill norms
// (ncols 128/256), which run the stock 256-thread block at 94-117 GB/s with half the block idle.
// Reduction is bitwise stock: the per-warp xor butterfly equals the stock warp-j partial and the
// combine replays the stock block_reduce second stage; rows beyond nrows contribute +0.0f.
template <bool do_multiply>
static __global__ void rms_norm_f32_warp(const float * x,
                                         float *       dst,
                                         const int     ncols,
                                         const int     nrows,
                                         const int64_t stride_row,
                                         const int64_t stride_channel,
                                         const int64_t stride_sample,
                                         const float   eps,
                                         const float * mul                  = nullptr,
                                         const int64_t mul_stride_row       = 0,
                                         const int64_t mul_stride_channel   = 0,
                                         const int64_t mul_stride_sample    = 0,
                                         const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                         const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                         const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                         const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0)) {
    ggml_cuda_pdl_lc();
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    const int row     = blockIdx.x * warps_per_block + warp_id;
    const bool active = row < nrows;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;

    const float * xr = x + sample*stride_sample + channel*stride_channel + row*stride_row;
    float *       dr = dst + ((int64_t)((int64_t)sample*gridDim.y + channel)*nrows + row)*ncols;
    if constexpr (do_multiply) {
        if (active) {
            const uint32_t mrow     = fastmodulo(row, mul_nrows_packed);
            const uint32_t mchannel = fastmodulo(channel, mul_nchannels_packed);
            const uint32_t msample  = fastmodulo(sample, mul_nsamples_packed);
            mul += msample * mul_stride_sample + mchannel * mul_stride_channel + mrow * mul_stride_row;
        }
    }

    float s_row[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        s_row[j] = 0.0f;
    }
    if (active) {
#pragma unroll
        for (int j = 0; j < ncols/WARP_SIZE; ++j) {
            const float xi = xr[lane + j*WARP_SIZE];
            s_row[j] = warp_reduce_sum(0.0f + xi*xi);
        }
    }
    const float total = ((s_row[0] + s_row[4]) + (s_row[2] + s_row[6])) +
                        ((s_row[1] + s_row[5]) + (s_row[3] + s_row[7]));

    const float mean  = total / ncols;
    const float scale = rsqrtf(mean + eps);
    if (!active) {
        return;
    }
    for (int j = 0; j < ncols/WARP_SIZE; ++j) {
        const int col = lane + j*WARP_SIZE;
        const float xi = xr[col];
        if constexpr (do_multiply) {
            dr[col] = scale * xi * mul[fastmodulo(col, mul_ncols_packed)];
        } else {
            dr[col] = scale * xi;
        }
    }
}

// V3-19 dispatch: per-head shapes only, and only for n_tokens >= 64 (channels x samples) so
// decode and the MTP verify keep the stock kernel.
static bool rms_norm_warp_ok(const int ncols, const int nchannels, const int nsamples) {
    static const bool on = [] {
        const char * s = getenv("GGML_CUDA_NORM_WARP");
        return s != nullptr && atoi(s) != 0;
    }();
    return on && ncols >= 32 && ncols <= 256 && ncols % 32 == 0 && (int64_t)nchannels * nsamples >= 64;
}

// RMS_NORM followed by SCALE (bias 0) in one launch: the scale is applied in the store. The reduction is the
// one of rms_norm_f32<256, false>, and the store computes (scale*x)*post_scale + post_bias, the same two
// roundings as rms_norm_f32 followed by scale_f32, so the bits are unchanged.
template <int block_size>
static __global__ void rms_norm_f32_scale(const float * x, float * dst, const int ncols,
                                          const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
                                          const float eps, const float post_scale, const float post_bias) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f;

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col] * post_scale + post_bias;
    }
}

// RMS_NORM -> MUL (weight) -> SILU (gate) -> MUL (the gated norm of the GDN layers) in one launch. The norm's
// store is rms_norm_f32<256, true>'s, (scale*x)*w, and the gate multiplies it last as in the elementwise chain
// (silu(gate) * normed), so the bits are unchanged. The intermediate rows are never written.
template <int block_size>
static __global__ void rms_norm_f32_gate(const float * x, const float * w, const float * gate, float * dst,
                                         const int ncols, const int64_t stride_row, const int64_t stride_channel,
                                         const int64_t stride_sample, const float eps) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x    += sample*stride_sample + channel*stride_channel + row*stride_row;
    const int64_t out_off = ((int64_t) (sample*nchannels + channel)*nrows + row)*ncols;
    dst  += out_off;
    gate += out_off;

    float tmp = 0.0f;

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const float normed = scale * x[col] * w[col];
        dst[col] = ggml_cuda_op_silu_single(gate[col]) * normed;
    }
}

static void rms_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        if (rms_norm_warp_ok(ncols, nchannels, nsamples)) {
            const int  wpb  = 256 / WARP_SIZE;
            const dim3 grid((nrows + wpb - 1) / wpb, nchannels, nsamples);
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = {grid, block_dims, 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32_warp<false>, launch_params,
                x, dst, ncols, nrows, stride_row, stride_channel, stride_sample, eps,
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0));
            return;
        }
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<256, false>, launch_params,
            x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
            nullptr, nullptr, nullptr);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<1024, false>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
            nullptr, nullptr, nullptr);
    }
}

// NORM-EMITS-STEP2 launcher: wide256 q8_1 emit form of the pre_add norm. Returns false when the
// shape is out of the fused envelope (caller falls back to the unfused path).
static bool rms_norm_f32_emit_cuda(const float * x, float * dst, const float * mul,
        const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
        const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const uint32_t mul_ncols, const uint32_t mul_ne1, const uint32_t mul_ne2, const uint32_t mul_ne3,
        const float eps, const float * pre_a, const float * pre_b, float * pre_dst,
        char * q8, const int64_t npad, const bool swizzle, const bool barrier, cudaStream_t stream) {
    if (ncols % 1024 != 0 || ncols < 1024 || ncols > 8192) {
        return false;
    }
    const int nv = ncols / 1024;
    // broadcast divisors are the MUL's dims (the stock launchers feed mul_src->ne[1..3]); the
    // source's dims make row 1 of a rows=2 verify read the adjacent weight row (in bounds, silent)
    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_ne1);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_ne2);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_ne3);
    const dim3 blocks_num(nrows * (ncols/256), nchannels, nsamples);
    const dim3 block_dims(1024, 1, 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 32 * sizeof(float), stream};
    const int64_t np = npad;
    switch ((nv << 1) | (swizzle ? 1 : 0)) {
        case (1<<1)|0: ggml_cuda_kernel_launch(rms_norm_f32_emit<1, false>, lp, x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, pre_a, pre_b, pre_dst, (block_q8_1 *) q8, (int) np, barrier ? 1 : 0); return true;
        case (1<<1)|1: ggml_cuda_kernel_launch(rms_norm_f32_emit<1, true>,  lp, x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, pre_a, pre_b, pre_dst, (block_q8_1 *) q8, (int) np, barrier ? 1 : 0); return true;
        case (2<<1)|0: ggml_cuda_kernel_launch(rms_norm_f32_emit<2, false>, lp, x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, pre_a, pre_b, pre_dst, (block_q8_1 *) q8, (int) np, barrier ? 1 : 0); return true;
        case (2<<1)|1: ggml_cuda_kernel_launch(rms_norm_f32_emit<2, true>,  lp, x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, pre_a, pre_b, pre_dst, (block_q8_1 *) q8, (int) np, barrier ? 1 : 0); return true;
        case (3<<1)|0: ggml_cuda_kernel_launch(rms_norm_f32_emit<3, false>, lp, x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, pre_a, pre_b, pre_dst, (block_q8_1 *) q8, (int) np, barrier ? 1 : 0); return true;
        case (3<<1)|1: ggml_cuda_kernel_launch(rms_norm_f32_emit<3, true>,  lp, x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, pre_a, pre_b, pre_dst, (block_q8_1 *) q8, (int) np, barrier ? 1 : 0); return true;
        case (4<<1)|0: ggml_cuda_kernel_launch(rms_norm_f32_emit<4, false>, lp, x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, pre_a, pre_b, pre_dst, (block_q8_1 *) q8, (int) np, barrier ? 1 : 0); return true;
        case (4<<1)|1: ggml_cuda_kernel_launch(rms_norm_f32_emit<4, true>,  lp, x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, pre_a, pre_b, pre_dst, (block_q8_1 *) q8, (int) np, barrier ? 1 : 0); return true;
        case (5<<1)|0: ggml_cuda_kernel_launch(rms_norm_f32_emit<5, false>, lp, x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, pre_a, pre_b, pre_dst, (block_q8_1 *) q8, (int) np, barrier ? 1 : 0); return true;
        case (5<<1)|1: ggml_cuda_kernel_launch(rms_norm_f32_emit<5, true>,  lp, x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, pre_a, pre_b, pre_dst, (block_q8_1 *) q8, (int) np, barrier ? 1 : 0); return true;
        case (6<<1)|0: ggml_cuda_kernel_launch(rms_norm_f32_emit<6, false>, lp, x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, pre_a, pre_b, pre_dst, (block_q8_1 *) q8, (int) np, barrier ? 1 : 0); return true;
        case (6<<1)|1: ggml_cuda_kernel_launch(rms_norm_f32_emit<6, true>,  lp, x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, pre_a, pre_b, pre_dst, (block_q8_1 *) q8, (int) np, barrier ? 1 : 0); return true;
        case (7<<1)|0: ggml_cuda_kernel_launch(rms_norm_f32_emit<7, false>, lp, x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, pre_a, pre_b, pre_dst, (block_q8_1 *) q8, (int) np, barrier ? 1 : 0); return true;
        case (7<<1)|1: ggml_cuda_kernel_launch(rms_norm_f32_emit<7, true>,  lp, x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, pre_a, pre_b, pre_dst, (block_q8_1 *) q8, (int) np, barrier ? 1 : 0); return true;
        case (8<<1)|0: ggml_cuda_kernel_launch(rms_norm_f32_emit<8, false>, lp, x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, pre_a, pre_b, pre_dst, (block_q8_1 *) q8, (int) np, barrier ? 1 : 0); return true;
        default:       ggml_cuda_kernel_launch(rms_norm_f32_emit<8, true>,  lp, x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, pre_a, pre_b, pre_dst, (block_q8_1 *) q8, (int) np, barrier ? 1 : 0); return true;
    }
}

// Returns false outside the envelope (1024 < ncols <= 8192 is the cached one-CTA-per-row geometry); the caller
// then takes the generic kernel.
static bool rms_norm_f32_plain_cuda(const float * pre_a, const float * pre_b, float * pre_dst, const float * mul,
        float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps,
        cudaStream_t stream) {
    if (ncols <= 1024 || ncols > 8192) {
        return false;
    }
    const dim3 blocks_num(nrows, nchannels, nsamples);
    const dim3 block_dims(1024, 1, 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 32 * sizeof(float), stream};
    switch ((ncols + 1023) / 1024) {
        case 2: ggml_cuda_kernel_launch(rms_norm_f32_plain<1024, 2>, lp, pre_a, pre_b, pre_dst, mul, dst, ncols, stride_row, stride_channel, stride_sample, eps); return true;
        case 3: ggml_cuda_kernel_launch(rms_norm_f32_plain<1024, 3>, lp, pre_a, pre_b, pre_dst, mul, dst, ncols, stride_row, stride_channel, stride_sample, eps); return true;
        case 4: ggml_cuda_kernel_launch(rms_norm_f32_plain<1024, 4>, lp, pre_a, pre_b, pre_dst, mul, dst, ncols, stride_row, stride_channel, stride_sample, eps); return true;
        case 5: ggml_cuda_kernel_launch(rms_norm_f32_plain<1024, 5>, lp, pre_a, pre_b, pre_dst, mul, dst, ncols, stride_row, stride_channel, stride_sample, eps); return true;
        case 6: ggml_cuda_kernel_launch(rms_norm_f32_plain<1024, 6>, lp, pre_a, pre_b, pre_dst, mul, dst, ncols, stride_row, stride_channel, stride_sample, eps); return true;
        case 7: ggml_cuda_kernel_launch(rms_norm_f32_plain<1024, 7>, lp, pre_a, pre_b, pre_dst, mul, dst, ncols, stride_row, stride_channel, stride_sample, eps); return true;
        case 8: ggml_cuda_kernel_launch(rms_norm_f32_plain<1024, 8>, lp, pre_a, pre_b, pre_dst, mul, dst, ncols, stride_row, stride_channel, stride_sample, eps); return true;
        default: return false;
    }
}

static void rms_norm_mul_f32_cuda(const float *  x,
                                  const float *  mul,
                                  const float *  add,
                                  float *        dst,
                                  const int      ncols,
                                  const int      nrows,
                                  const int      nchannels,
                                  const int      nsamples,
                                  const int64_t  stride_row,
                                  const int64_t  stride_channel,
                                  const int64_t  stride_sample,
                                  const int64_t  mul_stride_row,
                                  const int64_t  mul_stride_channel,
                                  const int64_t  mul_stride_sample,
                                  const uint32_t mul_ncols,
                                  const uint32_t mul_nrows,
                                  const uint32_t mul_nchannels,
                                  const uint32_t mul_nsamples,
                                  const int64_t  add_stride_row,
                                  const int64_t  add_stride_channel,
                                  const int64_t  add_stride_sample,
                                  const uint32_t add_ncols,
                                  const uint32_t add_nrows,
                                  const uint32_t add_nchannels,
                                  const uint32_t add_nsamples,
                                  const float    eps,
                                  cudaStream_t   stream,
                                  // SBB-19: non-null folds the residual add into the prologue
                                  const float *  pre_a   = nullptr,
                                  const float *  pre_b   = nullptr,
                                  float *        pre_dst = nullptr) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (mul != nullptr && add == nullptr && pre_a != nullptr) {
        // single-row call with a plain 1-D norm weight: width-specialized kernel, same bits as the generic one below.
        // Multi-row calls stay generic: with this kernel the prose prompt's output differed run to run.
        if (nrows == 1 && nchannels == 1 && nsamples == 1 &&
            mul_ncols == (uint32_t) ncols && mul_nrows == 1 && mul_nchannels == 1 && mul_nsamples == 1 &&
            rms_norm_f32_plain_cuda(pre_a, pre_b, pre_dst, mul, dst, ncols, nrows, nchannels, nsamples,
                                    stride_row, stride_channel, stride_sample, eps, stream)) {
            return;
        }
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<256, true, false, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
                pre_a, pre_b, pre_dst);
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<1024, true, false, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
                pre_a, pre_b, pre_dst);
        }
        return;
    }
    if (mul == nullptr) {
        rms_norm_f32_cuda(x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    if (add == nullptr) {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
        if (ncols < 1024) {
            if (rms_norm_warp_ok(ncols, nchannels, nsamples)) {
                const int  wpb  = 256 / WARP_SIZE;
                const dim3 grid((nrows + wpb - 1) / wpb, nchannels, nsamples);
                const dim3 block_dims(256, 1, 1);
                const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{grid, block_dims, 0, stream};
                ggml_cuda_kernel_launch(rms_norm_f32_warp<true>, launch_params,
                    x, dst, ncols, nrows, stride_row, stride_channel, stride_sample, eps,
                    mul, mul_stride_row, mul_stride_channel, mul_stride_sample,
                    mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
                return;
            }
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<256, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
            nullptr, nullptr, nullptr);
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<1024, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
            nullptr, nullptr, nullptr);
        }
    } else {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

        const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims,block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<256, true, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed,
                nullptr, nullptr, nullptr);
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<1024, true, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed,
                nullptr, nullptr, nullptr);
        }
    }
}

static void rms_norm_back_f32_cuda(const float * grad, const float * xf, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32<1024><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    }
}

static void l2_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<WARP_SIZE>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<1024>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

// RMS_NORM -> SCALE (bias 0) as one launch that writes the scale's dst; the rms output is never written.
// Shapes outside the <256> form (ncols >= 1024, or the warp variant) run the two ops unchanged.
void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms, ggml_tensor * scale) {
    const ggml_tensor * src0 = rms->src[0];
    const int64_t ncols = src0->ne[0];

    if (ncols >= 1024 || rms_norm_warp_ok((int) ncols, (int) src0->ne[2], (int) src0->ne[3])) {
        ggml_cuda_op_rms_norm(ctx, rms);
        ggml_cuda_op_scale(ctx, scale);
        return;
    }

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && rms->type == GGML_TYPE_F32 && scale->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_is_contiguous(rms) && ggml_is_contiguous(scale));

    float eps;
    memcpy(&eps, rms->op_params, sizeof(float));
    float post_scale;
    float post_bias;
    memcpy(&post_scale, (const float *) scale->op_params + 0, sizeof(float));
    memcpy(&post_bias,  (const float *) scale->op_params + 1, sizeof(float));

    const int64_t s01 = src0->nb[1] / sizeof(float);
    const int64_t s02 = src0->nb[2] / sizeof(float);
    const int64_t s03 = src0->nb[3] / sizeof(float);

    const dim3 blocks_num(src0->ne[1], src0->ne[2], src0->ne[3]);
    const dim3 block_dims(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32 * sizeof(float), ctx.stream()};
    ggml_cuda_kernel_launch(rms_norm_f32_scale<256>, launch_params, (const float *) src0->data, (float *) scale->data,
        (int) ncols, s01, s02, s03, eps, post_scale, post_bias);
}

bool ggml_cuda_rms_norm_gate_ok(const ggml_tensor * rms) {
    const ggml_tensor * src0 = rms->src[0];
    return src0->ne[0] < 1024 && !rms_norm_warp_ok((int) src0->ne[0], (int) src0->ne[2], (int) src0->ne[3]);
}

// RMS_NORM -> MUL (weight) -> SILU (gate) -> MUL as one launch that writes the last MUL's dst. The caller checks the
// shapes with ggml_cuda_rms_norm_gate_ok() and the layouts (f32, contiguous weight and gate, gate shaped like rms).
void ggml_cuda_op_rms_norm_gate_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms, ggml_tensor * weight_mul,
                                      ggml_tensor * gate_silu, ggml_tensor * out) {
    const ggml_tensor * src0 = rms->src[0];
    const ggml_tensor * w    = weight_mul->src[1];
    const ggml_tensor * gate = gate_silu->src[0];

    GGML_ASSERT(ggml_cuda_rms_norm_gate_ok(rms));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && gate->type == GGML_TYPE_F32);
    GGML_ASSERT(rms->type == GGML_TYPE_F32 && out->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == sizeof(float) && w->nb[0] == sizeof(float) && gate->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_is_contiguous(w) && ggml_nelements(w) == src0->ne[0]);
    GGML_ASSERT(ggml_is_contiguous(gate) && ggml_is_contiguous(out) && ggml_are_same_shape(gate, rms));

    float eps;
    memcpy(&eps, rms->op_params, sizeof(float));

    const int64_t ncols = src0->ne[0];
    const int64_t s01 = src0->nb[1] / sizeof(float);
    const int64_t s02 = src0->nb[2] / sizeof(float);
    const int64_t s03 = src0->nb[3] / sizeof(float);

    const dim3 blocks_num(src0->ne[1], src0->ne[2], src0->ne[3]);
    const dim3 block_dims(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32 * sizeof(float), ctx.stream()};
    ggml_cuda_kernel_launch(rms_norm_f32_gate<256>, launch_params, (const float *) src0->data, (const float *) w->data,
        (const float *) gate->data, (float *) out->data, (int) ncols, s01, s02, s03, eps);
}

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float * src0_d = (const float *) rms_norm_src->data;
    const float * mul_d = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == dst) {
        mul_d = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_d = (float *) mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_cuda(src0_d, mul_d, nullptr, dst_d,
                          ne00, ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ 0, 0, 0,
                          0, 0, 0, 0,
                          eps, stream);
}

// SBB-19: fuse ADD -> RMS_NORM -> MUL into one launch. The ADD's output (the next layer's residual) is still
// written, so no node is eliminated; the only requirement is that rms_norm's input is exactly the preceding ADD.
void ggml_cuda_op_rms_norm_fused_pre_add(ggml_backend_cuda_context & ctx,
                                         ggml_tensor *               add_tensor,
                                         ggml_tensor *               dst,
                                         ggml_tensor *               mul_tensor,
                                         char *                      q8_emit,
                                         int64_t                     q8_npad,
                                         bool                        q8_swizzle,
                                         bool                        q8_barrier) {
    const ggml_tensor * rms_norm_src = (const ggml_tensor *) dst->src[0];
    GGML_ASSERT(rms_norm_src == add_tensor);

    float eps = 0.0f;
    memcpy(&eps, dst->op_params, sizeof(float));

    const float * mul_d   = nullptr;
    const ggml_tensor * mul_src = nullptr;
    if (mul_tensor->src[0] == dst) {
        mul_d   = (const float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_d   = (const float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    if (q8_emit != nullptr) {
        // NORM-EMITS-STEP2: wide256 emit form; on success the q8_cache record makes the consumer
        // GEMVs skip their own quantize launch. Same stream, so the record is ordered after the write.
        if (rms_norm_f32_emit_cuda((const float *) rms_norm_src->data, (float *) mul_tensor->data, mul_d,
                (int) ne00, (int) ne01, (int) ne02, (int) ne03,
                s01, s02, s03, mul_s01, mul_s02, mul_s03, (uint32_t) mul_src->ne[0],
                (uint32_t) mul_src->ne[1], (uint32_t) mul_src->ne[2], (uint32_t) mul_src->ne[3], eps,
                (const float *) add_tensor->src[0]->data, (const float *) add_tensor->src[1]->data,
                (float *) add_tensor->data, q8_emit, q8_npad, q8_swizzle, q8_barrier, ctx.stream())) {
            // BISECT (supervisor 05:10Z): skip the qc stamp so the GEMVs requantize from the f32
            // dst; identical output then means the q8 bytes/consumer contract is at fault.
            static const bool no_stamp = getenv("GGML_CUDA_NORM_EMIT_NO_STAMP") != nullptr &&
                                         atoi(getenv("GGML_CUDA_NORM_EMIT_NO_STAMP")) != 0;
            if (!no_stamp) {
            auto & qc = ctx.q8_cache;
            const int64_t ne10_padded = GGML_PAD(mul_tensor->ne[0], MATRIX_ROW_PADDING);
            qc.src1        = mul_tensor;
            qc.data        = mul_tensor->data;
            qc.epoch       = ctx.graph_epoch;
            qc.size        = (size_t) mul_tensor->ne[2]*mul_tensor->ne[3] * mul_tensor->ne[1]*ne10_padded * sizeof(block_q8_1)/QK8_1;
            qc.ne10_padded = ne10_padded;
            qc.type        = q8_swizzle ? GGML_TYPE_IQ4_XS : GGML_TYPE_Q8_0;
            qc.dev         = ctx.device;
            }
            return;
        }
    }

    rms_norm_mul_f32_cuda((const float *) rms_norm_src->data, mul_d, nullptr, (float *) mul_tensor->data,
                          ne00, ne01, ne02, ne03,
                          s01, s02, s03,
                          mul_s01, mul_s02, mul_s03,
                          (int) mul_src->ne[0], (int) mul_src->ne[1], (int) mul_src->ne[2], (int) mul_src->ne[3],
                          0, 0, 0,
                          0, 0, 0, 0,
                          eps, ctx.stream(),
                          (const float *) add_tensor->src[0]->data,
                          (const float *) add_tensor->src[1]->data,
                          (float *)       add_tensor->data);
}

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float               eps          = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float *       src0_d  = (const float *) rms_norm_src->data;
    const float *       mul_d   = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d   = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_d   = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const float *       add_d   = nullptr;
    const ggml_tensor * add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        add_d   = (float *) add_tensor->src[1]->data;
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_d   = (float *) add_tensor->src[0]->data;
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float *      dst_d  = (float *) add_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const int add_ncols     = add_src->ne[0];
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    rms_norm_mul_f32_cuda(src0_d, mul_d,add_d,dst_d,
                          ne00,ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ add_s01, add_s02, add_s03,
                          add_ncols, add_nrows, add_nchannels, add_nsamples,
                          eps, stream);
}

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0]; // gradients
    const ggml_tensor * src0f = dst->src[1]; // src0 from forward pass

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(grad));

    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    rms_norm_back_f32_cuda(grad_d, src0f_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    l2_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}
