// Inkling-2 packed matrix products (GGML_OP_INK2_MUL_MAT): each block decodes a few rows of one matrix to float32
// in shared memory, exactly as ggml-inkling2.h (formats.py) defines them, and dots them with every activation
// column routed to that matrix. Correctness first: no tensor cores, no FP16 math.

#include "ink2.cuh"
#include "ggml-inkling2.h"

#define INK2_THREADS 256
#define INK2_RMAX    8

struct ink2_args {
    const uint8_t * codes;
    const uint8_t * scales;
    const void    * table;
    const float   * e4m3;
    const float   * x;
    const int32_t * ids;
    const int16_t * map;
    float         * y;
    const int32_t * row_map;
    int M_out;
    int64_t codes_nb_e, scales_nb_e; // byte strides between the experts of a slab
    int M, K, R;
    int N, n_used, n_xs, n_exp;
};

template <int FMT>
static __device__ void ink2_decode_row_par(const ink2_args & a, const uint8_t * codes, const uint8_t * scales, float outer,
                                           int r, float * out) {
    const int K = a.K;
    if constexpr (FMT <= INK2_RTR4) {
        const ink2_trellis_params p = ink2_trellis(FMT);
        const int64_t wpr = (int64_t) K * p.k / 32;
        const uint32_t * words = (const uint32_t *) codes + r * wpr;
        const float s = ink2_f16_to_f32(((const uint16_t *) scales)[r]);
        const uint16_t * lut = (const uint16_t *) a.table;
        for (int j = threadIdx.x; j < K / p.g; j += blockDim.x) {
            const uint32_t idx = ink2_trellis_index(words, K, j * p.g + p.g - 1, p);
#pragma unroll
            for (int t = 0; t < p.g; ++t) {
                out[j * p.g + t] = ink2_f16_to_f32(lut[idx * p.g + t]) * s;
            }
        }
    } else if constexpr (FMT == INK2_LM6 || FMT == INK2_LM8) {
        constexpr int k = FMT == INK2_LM6 ? 6 : 8;
        const uint8_t * row = codes + ((int64_t) r * K * k) / 8;
        const float s = ink2_f16_to_f32(((const uint16_t *) scales)[r]);
        const float * levels = (const float *) a.table;
        for (int j = threadIdx.x; j < K; j += blockDim.x) {
            out[j] = levels[ink2_lm_index(row, k, j)] * s;
        }
    } else {
        const float e2m1[16] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                 -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };
        const uint8_t * crow = codes + (int64_t) r * (K / 2);
        const uint8_t * srow = scales + (int64_t) r * (K / 16);
        for (int j = threadIdx.x; j < K; j += blockDim.x) {
            const uint8_t c = crow[j >> 1];
            const float v = e2m1[(j & 1) ? (c >> 4) : (c & 0xF)];
            out[j] = (v * a.e4m3[srow[j >> 4]]) * outer;
        }
    }
}

template <int FMT>
static __global__ void ink2_mul_mat_kernel(const ink2_args a) {
    extern __shared__ float smem[];
    float * rows = smem;
    int   * list = (int *) (smem + a.R * a.K);
    __shared__ int s_e, s_cnt;

    const bool experts = a.ids != nullptr;
    const uint8_t * codes  = a.codes;
    const uint8_t * scales = a.scales;
    float outer = 1.0f;
    int cnt = a.N;

    if (experts) {
        const int slab = blockIdx.y;
        if (threadIdx.x == 0) {
            s_e = -1;
            s_cnt = 0;
        }
        __syncthreads();
        for (int e = threadIdx.x; e < a.n_exp; e += blockDim.x) {
            if (a.map[2 * e] == FMT && a.map[2 * e + 1] == slab) {
                s_e = e;
            }
        }
        __syncthreads();
        const int e = s_e;
        if (e < 0) {
            return;
        }
        for (int i = threadIdx.x; i < a.n_used * a.N; i += blockDim.x) {
            if (a.ids[i] == e) {
                list[atomicAdd(&s_cnt, 1)] = i;
            }
        }
        __syncthreads();
        cnt = s_cnt;
        if (cnt == 0) {
            return;
        }
        codes  += slab * a.codes_nb_e;
        scales += slab * a.scales_nb_e;
        if constexpr (FMT == INK2_NVFP4) {
            outer = ((const float *) a.table)[slab];
        }
    }

    const int r0 = blockIdx.x * a.R;
    const int nr = min(a.R, a.M - r0);
    for (int rr = 0; rr < nr; ++rr) {
        ink2_decode_row_par<FMT>(a, codes, scales, outer, r0 + rr, rows + rr * a.K);
    }
    __syncthreads();

    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int n_warps = blockDim.x / WARP_SIZE;
    for (int j = warp; j < cnt; j += n_warps) {
        int64_t out_col, x_col;
        if (experts) {
            const int i = list[j];
            const int t = i / a.n_used, s = i % a.n_used;
            out_col = i;
            x_col = (int64_t) t * a.n_xs + (a.n_xs > 1 ? s : 0);
        } else {
            out_col = j;
            x_col = j;
        }
        const float * xt = a.x + x_col * a.K;
        float acc[INK2_RMAX];
#pragma unroll
        for (int rr = 0; rr < INK2_RMAX; ++rr) {
            acc[rr] = 0.0f;
        }
        for (int k = lane; k < a.K; k += WARP_SIZE) {
            const float xv = xt[k];
#pragma unroll
            for (int rr = 0; rr < INK2_RMAX; ++rr) {
                if (rr < nr) {
                    acc[rr] += rows[rr * a.K + k] * xv;
                }
            }
        }
#pragma unroll
        for (int rr = 0; rr < INK2_RMAX; ++rr) {
            if (rr < nr) {
                const float v = warp_reduce_sum(acc[rr]);
                if (lane == 0) {
                    const int row = a.row_map ? a.row_map[r0 + rr] : r0 + rr;
                    a.y[out_col * a.M_out + row] = v;
                }
            }
        }
    }
}

// Small-N path (decode): one warp per row. The warp stages the row's code bytes (and NVFP4 block scales) in shared
// memory with coalesced 16-byte loads, prefetching its next row into registers while it computes the current one;
// the code table sits in shared memory as f32. Every column's sum runs in the same order as ink2_mul_mat_kernel's
// (lane k = lane + 32 t ascending, then the warp butterfly) and each weight is the same f32 value, so the two kernels
// agree bitwise. Experts: blockIdx.y is a routed (token, slot) pair; blocks whose expert is in another format leave
// at once.

#define INK2_VEC_NC    8  // most columns of the dense small-N path
#define INK2_VEC_WARPS 8
#define INK2_VEC_V4    6  // most 16-byte code loads per lane and row (lm8, 3072 columns)
#define INK2_VEC_S4    3  // most 4-byte NVFP4 scale loads per lane and row (2304 columns: 36 words)

template <int FMT>
static __device__ __forceinline__ float ink2_vec_weight(const uint8_t * buf, const uint8_t * sbuf, const float * tab,
                                                        float s, float outer, int K, int k) {
    if constexpr (FMT <= INK2_RTR4) {
        constexpr int kb = FMT == INK2_HT1 || FMT == INK2_RTR1 ? 1 : FMT == INK2_HT2 || FMT == INK2_RTR2 ? 2 :
                           FMT == INK2_RTR4 ? 4 : 3;
        constexpr int g  = FMT == INK2_HT1 || FMT == INK2_HT2 ? 4 : FMT == INK2_HT3 ? 2 : 1;
        constexpr int l2 = FMT == INK2_HT1 || FMT == INK2_HT2 ? 9 : FMT == INK2_HT3 ? 10 : 12;
        const uint32_t * words = (const uint32_t *) buf; // one zero word past the row: bits past the end read as 0
        const int last = (k / g) * g + g - 1;
        const int bp = (K - 1 - last) * kb;
        const int wi = bp >> 5, off = bp & 31;
        const uint64_t lo = words[wi];
        const uint64_t hi = words[wi + 1];
        uint32_t idx = (uint32_t) (((lo | (hi << 32)) >> off) & 4095u);
        if constexpr (FMT <= INK2_HT3) {
            idx = (uint32_t) (idx * 0x9E3779B1u) >> (32 - l2);
        }
        return tab[idx * g + (k % g)] * s;
    } else if constexpr (FMT == INK2_LM8) {
        return tab[buf[k]] * s;
    } else if constexpr (FMT == INK2_LM6) {
        const uint8_t * t = buf + (k >> 2) * 3;
        const uint32_t v = (uint32_t) t[0] | ((uint32_t) t[1] << 8) | ((uint32_t) t[2] << 16);
        return tab[(v >> (6 * (k & 3))) & 63u] * s;
    } else {
        // e2m1: twice the magnitudes 0 1 2 3 4 6 8 12 as nibbles of one word, the sign bit OR-ed in (-0.0 for 8)
        const uint32_t nib = ((uint32_t) buf[k >> 1] >> ((k & 1) * 4)) & 0xF;
        const float mag = (float) ((0xC8643210u >> ((nib & 7) * 4)) & 0xF) * 0.5f;
        const float v = __int_as_float(__float_as_int(mag) | (int) ((nib & 8) << 28));
        return (v * tab[sbuf[k >> 4]]) * outer;
    }
}

template <int FMT, int NC>
static __global__ void __launch_bounds__(INK2_VEC_WARPS * WARP_SIZE)
ink2_mul_mat_vec_kernel(const ink2_args a, const int rpb, const int n_tab, const int x_floats, const int row_bytes,
                        const int sc_bytes) {
    extern __shared__ float smem[];
    float * tab = smem;

    const bool experts = a.ids != nullptr;
    const uint8_t * codes  = a.codes;
    const uint8_t * scales = a.scales;
    float outer = 1.0f;
    int ncols = a.N;
    int64_t x_col0 = 0, out_col0 = 0;

    if (experts) {
        const int i = blockIdx.y;
        const int e = a.ids[i];
        if (a.map[2 * e] != FMT) {
            return;
        }
        const int slab = a.map[2 * e + 1];
        codes  += slab * a.codes_nb_e;
        scales += slab * a.scales_nb_e;
        if constexpr (FMT == INK2_NVFP4) {
            outer = ((const float *) a.table)[slab];
        }
        const int t = i / a.n_used, s = i % a.n_used;
        ncols = 1;
        out_col0 = i;
        x_col0 = (int64_t) t * a.n_xs + (a.n_xs > 1 ? s : 0);
    }

    // the code table as f32 (the same values ink2_decode_row_par makes per weight)
    for (int j = threadIdx.x; j < n_tab; j += blockDim.x) {
        if constexpr (FMT <= INK2_RTR4) {
            tab[j] = ink2_f16_to_f32(((const uint16_t *) a.table)[j]);
        } else if constexpr (FMT == INK2_NVFP4) {
            tab[j] = a.e4m3[j];
        } else {
            tab[j] = ((const float *) a.table)[j];
        }
    }

    // the block's activation columns
    const int K = a.K;
    float * xs = smem + n_tab;
    {
        const float4 * src = (const float4 *) (a.x + x_col0 * K);
        for (int j = threadIdx.x; j < ncols * K / 4; j += blockDim.x) {
            ((float4 *) xs)[j] = __ldg(src + j);
        }
    }

    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int buf_bytes = (row_bytes + 16 + sc_bytes + 15) / 16 * 16; // 16-byte aligned per warp
    uint8_t * buf  = (uint8_t *) (xs + x_floats) + warp * buf_bytes;
    uint8_t * sbuf = buf + row_bytes + 16;
    if (lane < 4) {
        ((uint32_t *) (buf + row_bytes))[lane] = 0;
    }
    __syncthreads();

    const int nv = row_bytes / 16, ns = sc_bytes / 4;
    const int r_end = min(a.M, (int) (blockIdx.x + 1) * rpb);
    int r = blockIdx.x * rpb + warp;

    uint4 pre[INK2_VEC_V4];
    uint32_t spre[INK2_VEC_S4];
    uint16_t s_pre = 0;
#define INK2_VEC_LOAD(rr)                                                                                 \
    do {                                                                                                  \
        const uint4 * src = (const uint4 *) (codes + (int64_t) (rr) * row_bytes);                          \
        _Pragma("unroll") for (int q = 0; q < INK2_VEC_V4; ++q) {                                         \
            if (lane + 32 * q < nv) pre[q] = __ldg(src + lane + 32 * q);                                  \
        }                                                                                                 \
        if constexpr (FMT == INK2_NVFP4) {                                                                \
            const uint32_t * ssrc = (const uint32_t *) (scales + (int64_t) (rr) * sc_bytes);              \
            _Pragma("unroll") for (int q = 0; q < INK2_VEC_S4; ++q) {                                     \
                if (lane + 32 * q < ns) spre[q] = __ldg(ssrc + lane + 32 * q);                            \
            }                                                                                             \
        } else {                                                                                          \
            s_pre = __ldg((const unsigned short *) scales + (rr));                                        \
        }                                                                                                 \
    } while (0)

    if (r < r_end) {
        INK2_VEC_LOAD(r);
    }
    for (; r < r_end; r += INK2_VEC_WARPS) {
        __syncwarp();
#pragma unroll
        for (int q = 0; q < INK2_VEC_V4; ++q) {
            if (lane + 32 * q < nv) ((uint4 *) buf)[lane + 32 * q] = pre[q];
        }
        if constexpr (FMT == INK2_NVFP4) {
#pragma unroll
            for (int q = 0; q < INK2_VEC_S4; ++q) {
                if (lane + 32 * q < ns) ((uint32_t *) sbuf)[lane + 32 * q] = spre[q];
            }
        }
        const float s = FMT == INK2_NVFP4 ? 1.0f : ink2_f16_to_f32(s_pre);
        __syncwarp();
        if (r + INK2_VEC_WARPS < r_end) {
            INK2_VEC_LOAD(r + INK2_VEC_WARPS);
        }

        float acc[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            acc[c] = 0.0f;
        }
#pragma unroll 4
        for (int k = lane; k < K; k += WARP_SIZE) {
            const float wv = ink2_vec_weight<FMT>(buf, sbuf, tab, s, outer, K, k);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                if (NC == 1 || c < ncols) {
                    acc[c] += wv * xs[c * K + k];
                }
            }
        }
        const int row = a.row_map ? a.row_map[r] : r;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (NC == 1 || c < ncols) {
                const float v = warp_reduce_sum(acc[c]);
                if (lane == 0) {
                    a.y[(out_col0 + c) * a.M_out + row] = v;
                }
            }
        }
    }
#undef INK2_VEC_LOAD
}

// false when the matrix does not fit the small-N kernel's staging (alignment, row size); the caller falls back
template <int FMT>
static bool ink2_launch_vec(const ink2_args & a, int n_pairs, cudaStream_t stream) {
    int n_tab, row_bytes, sc_bytes = 0;
    if constexpr (FMT <= INK2_RTR4) {
        const ink2_trellis_params p = ink2_trellis(FMT);
        n_tab = (1 << p.log2K) * p.g;
        row_bytes = a.K * p.k / 8;
    } else if constexpr (FMT == INK2_LM6) {
        n_tab = 64;
        row_bytes = a.K * 6 / 8;
    } else if constexpr (FMT == INK2_LM8) {
        n_tab = 256;
        row_bytes = a.K;
    } else {
        n_tab = 256;
        row_bytes = a.K / 2;
        sc_bytes = a.K / 16;
    }
    const bool fits = a.K % 32 == 0 && row_bytes % 16 == 0 && row_bytes <= INK2_VEC_V4 * 32 * 16 &&
                      sc_bytes % 4 == 0 && sc_bytes <= INK2_VEC_S4 * 32 * 4 &&
                      (uintptr_t) a.codes % 16 == 0 && (uintptr_t) a.x % 16 == 0 &&
                      (FMT != INK2_NVFP4 || (uintptr_t) a.scales % 4 == 0) &&
                      (!a.ids || (a.codes_nb_e % 16 == 0 && (FMT != INK2_NVFP4 || a.scales_nb_e % 4 == 0)));
    if (!fits) {
        return false;
    }
    // aim at about four blocks per SM of useful work; experts: about half the routed pairs land in one format
    const int n_eff = a.ids ? std::max(1, n_pairs / 2) : 1;
    int rpb = (int) (((int64_t) a.M * n_eff + 223) / 224);
    rpb = std::max(INK2_VEC_WARPS, std::min(64, (rpb + INK2_VEC_WARPS - 1) / INK2_VEC_WARPS * INK2_VEC_WARPS));
    const int x_floats = (a.ids ? 1 : a.N) * a.K;
    const size_t smem = n_tab * sizeof(float) + x_floats * sizeof(float) +
                        (size_t) INK2_VEC_WARPS * ((row_bytes + 16 + sc_bytes + 15) / 16 * 16);
    if (smem > 48 * 1024) {
        return false;
    }
    const dim3 grid((a.M + rpb - 1) / rpb, n_pairs, 1);
    if (x_floats == a.K) {
        ink2_mul_mat_vec_kernel<FMT, 1><<<grid, INK2_VEC_WARPS * WARP_SIZE, smem, stream>>>(
            a, rpb, n_tab, x_floats, row_bytes, sc_bytes);
    } else {
        ink2_mul_mat_vec_kernel<FMT, INK2_VEC_NC><<<grid, INK2_VEC_WARPS * WARP_SIZE, smem, stream>>>(
            a, rpb, n_tab, x_floats, row_bytes, sc_bytes);
    }
    return true;
}

// One-column decode path (dense N == 1, and every routed expert pair): one warp per row, each lane owning whole
// 16-weight chunks (k = 16 c .. 16 c + 15, c = lane, lane + 32, ...) read straight from global memory (trellis: the three
// words holding the chunk's windows; NVFP4: one uint2 of nibbles and the chunk's E4M3 byte; Lloyd-Max: 16 or 12 bytes).
// Each chunk's dot runs in f32 on the code table's value; NVFP4 multiplies the chunk sum by its E4M3 block scale
// (LookAtMyQuant's gemv.py order); the row's f16 scale, or the expert's outer scale, multiplies the warp's sum.

#define INK2_CHUNK_MAXC 6 // most chunks per lane and row (3072 columns)

// the chunk's raw codes: trellis the three words from the lowest bit position (that of column kq + 15), NVFP4 the
// nibbles in x, y and the E4M3 byte in z, lm8 16 bytes, lm6 12 bytes
template <int FMT>
static __device__ __forceinline__ uint4 ink2_chunk_load(const uint8_t * crow, const uint8_t * srow, int K, int row_bytes,
                                                        int kq) {
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if constexpr (FMT <= INK2_RTR4) {
        constexpr int kb = FMT == INK2_HT1 || FMT == INK2_RTR1 ? 1 : FMT == INK2_HT2 || FMT == INK2_RTR2 ? 2 :
                           FMT == INK2_RTR4 ? 4 : 3;
        const uint32_t * words = (const uint32_t *) crow;
        const int wpr = row_bytes / 4;
        const int w0i = ((K - 1 - (kq + 15)) * kb) >> 5;
        v.x = __ldg(words + w0i);
        v.y = w0i + 1 < wpr ? __ldg(words + w0i + 1) : 0u; // bits past the end read as zero
        v.z = w0i + 2 < wpr ? __ldg(words + w0i + 2) : 0u;
    } else if constexpr (FMT == INK2_NVFP4) {
        const uint2 c = __ldg((const uint2 *) (crow + kq / 2));
        v.x = c.x;
        v.y = c.y;
        v.z = __ldg(srow + kq / 16);
    } else if constexpr (FMT == INK2_LM8) {
        v = __ldg((const uint4 *) (crow + kq));
    } else {
        const uint32_t * cp = (const uint32_t *) (crow + kq * 6 / 8);
        v.x = __ldg(cp);
        v.y = __ldg(cp + 1);
        v.z = __ldg(cp + 2);
    }
    return v;
}

// the chunk's 16 code-table values (NVFP4: the e2m1 values; the block scale is applied to the chunk's sum)
template <int FMT>
static __device__ __forceinline__ void ink2_chunk_decode(const uint4 raw, const float * tab, const float * e2m1, int K,
                                                         int kq, float * w) {
    if constexpr (FMT <= INK2_RTR4) {
        constexpr int kb = FMT == INK2_HT1 || FMT == INK2_RTR1 ? 1 : FMT == INK2_HT2 || FMT == INK2_RTR2 ? 2 :
                           FMT == INK2_RTR4 ? 4 : 3;
        constexpr int g  = FMT == INK2_HT1 || FMT == INK2_HT2 ? 4 : FMT == INK2_HT3 ? 2 : 1;
        constexpr int l2 = FMT == INK2_HT1 || FMT == INK2_HT2 ? 9 : FMT == INK2_HT3 ? 10 : 12;
        // the windows start at bit (15 - j - g + 1) kb past the chunk's lowest position, which lies at bit 0 or 16 of
        // the first word (16 kb divides it): shift that out so every window offset below is a constant
        uint32_t w0 = raw.x, w1 = raw.y, w2 = raw.z;
        if constexpr (kb % 2 == 1) {
            if (((K - 1 - (kq + 15)) * kb) & 16) {
                w0 = __funnelshift_r(w0, w1, 16);
                w1 = __funnelshift_r(w1, w2, 16);
                w2 = w2 >> 16;
            }
        }
#pragma unroll
        for (int j = 0; j < 16; j += g) {
            const int o = (15 - (j + g - 1)) * kb;
            const uint32_t wl = o < 32 ? w0 : o < 64 ? w1 : w2;
            const uint32_t wh = o < 32 ? w1 : w2;
            const int sh = o & 31;
            uint32_t idx = (sh + 12 <= 32 ? wl >> sh : __funnelshift_r(wl, wh, sh)) & 4095u;
            if constexpr (FMT <= INK2_HT3) {
                idx = (uint32_t) (idx * 0x9E3779B1u) >> (32 - l2);
            }
#pragma unroll
            for (int t = 0; t < g; ++t) {
                w[j + t] = tab[idx * g + t];
            }
        }
    } else if constexpr (FMT == INK2_NVFP4) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            w[j] = e2m1[((j < 8 ? raw.x : raw.y) >> ((j & 7) * 4)) & 0xF];
        }
    } else if constexpr (FMT == INK2_LM8) {
        const uint32_t cw[4] = { raw.x, raw.y, raw.z, raw.w };
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            w[j] = tab[(cw[j / 4] >> ((j % 4) * 8)) & 0xFF];
        }
    } else {
        const uint64_t lo = (uint64_t) raw.x | ((uint64_t) raw.y << 32);
        const uint64_t hi = (uint64_t) raw.y | ((uint64_t) raw.z << 32);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int b = 6 * j;
            w[j] = tab[(b < 32 ? lo >> b : hi >> (b - 32)) & 63u];
        }
    }
}

template <int FMT, int NR>
static __global__ void __launch_bounds__(INK2_VEC_WARPS * WARP_SIZE)
ink2_mul_mat_chunk_kernel(const ink2_args a, const int rpb, const int n_tab, const int row_bytes) {
    extern __shared__ float smem[];
    __shared__ float e2m1[16];
    float * tab = smem;
    float * xs  = smem + n_tab;

    const uint8_t * codes  = a.codes;
    const uint8_t * scales = a.scales;
    float outer = 1.0f;
    int64_t x_col = 0, out_col = 0;
    if (a.ids) {
        const int i = blockIdx.y;
        const int e = a.ids[i];
        if (a.map[2 * e] != FMT) {
            return;
        }
        const int slab = a.map[2 * e + 1];
        codes  += slab * a.codes_nb_e;
        scales += slab * a.scales_nb_e;
        if constexpr (FMT == INK2_NVFP4) {
            outer = ((const float *) a.table)[slab];
        }
        const int t = i / a.n_used, s = i % a.n_used;
        out_col = i;
        x_col = (int64_t) t * a.n_xs + (a.n_xs > 1 ? s : 0);
    }
    const int K = a.K;
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int r_end = min(a.M, (int) (blockIdx.x + 1) * rpb);
    const int n_chunks = K / 16;
    for (int j = threadIdx.x; j < n_tab; j += blockDim.x) {
        if constexpr (FMT <= INK2_RTR4) {
            tab[j] = ink2_f16_to_f32(((const uint16_t *) a.table)[j]);
        } else if constexpr (FMT == INK2_NVFP4) {
            tab[j] = a.e4m3[j];
        } else {
            tab[j] = ((const float *) a.table)[j];
        }
    }
    if (FMT == INK2_NVFP4 && threadIdx.x < 16) {
        const int nib = threadIdx.x;
        const float mag = (float) ((0xC8643210u >> ((nib & 7) * 4)) & 0xF) * 0.5f;
        e2m1[nib] = __int_as_float(__float_as_int(mag) | (int) ((nib & 8) << 28));
    }
    {
        const float4 * src = (const float4 *) (a.x + x_col * K);
        for (int j = threadIdx.x; j < K / 4; j += blockDim.x) {
            ((float4 *) xs)[j] = __ldg(src + j);
        }
    }
    __syncthreads();

    // NR rows per pass (r + i INK2_VEC_WARPS) share each activation load; each row's sums run in the same order
    // whatever the pass width
    for (int r = blockIdx.x * rpb + warp; r < r_end; r += NR * INK2_VEC_WARPS) {
        int rr[NR];
#pragma unroll
        for (int i = 0; i < NR; ++i) {
            rr[i] = r + i * INK2_VEC_WARPS < r_end ? r + i * INK2_VEC_WARPS : r;
        }
        float acc[NR];
#pragma unroll
        for (int i = 0; i < NR; ++i) {
            acc[i] = 0.0f;
        }
        for (int c = lane; c < n_chunks; c += WARP_SIZE) {
            const int kq = c * 16;
            uint4 raw[NR];
#pragma unroll
            for (int i = 0; i < NR; ++i) {
                raw[i] = ink2_chunk_load<FMT>(codes + (int64_t) rr[i] * row_bytes,
                                              FMT == INK2_NVFP4 ? scales + (int64_t) rr[i] * (K / 16) : nullptr,
                                              K, row_bytes, kq);
            }
            float w[NR][16];
#pragma unroll
            for (int i = 0; i < NR; ++i) {
                ink2_chunk_decode<FMT>(raw[i], tab, e2m1, K, kq, w[i]);
            }
            const float4 * x4 = (const float4 *) (xs + kq);
            float sm[NR];
#pragma unroll
            for (int i = 0; i < NR; ++i) {
                sm[i] = 0.0f;
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float4 xv = x4[q];
#pragma unroll
                for (int i = 0; i < NR; ++i) {
                    sm[i] += w[i][4 * q + 0] * xv.x;
                    sm[i] += w[i][4 * q + 1] * xv.y;
                    sm[i] += w[i][4 * q + 2] * xv.z;
                    sm[i] += w[i][4 * q + 3] * xv.w;
                }
            }
#pragma unroll
            for (int i = 0; i < NR; ++i) {
                if constexpr (FMT == INK2_NVFP4) {
                    sm[i] *= tab[raw[i].z];
                }
                acc[i] += sm[i];
            }
        }
#pragma unroll
        for (int i = 0; i < NR; ++i) {
            acc[i] = warp_reduce_sum(acc[i]);
        }
        if (lane == 0) {
#pragma unroll
            for (int i = 0; i < NR; ++i) {
                if (i == 0 || rr[i] != r) {
                    float sc = outer;
                    if constexpr (FMT != INK2_NVFP4) {
                        sc = ink2_f16_to_f32(__ldg((const unsigned short *) scales + rr[i]));
                    }
                    a.y[out_col * a.M_out + (a.row_map ? a.row_map[rr[i]] : rr[i])] = acc[i] * sc;
                }
            }
        }
    }
}

template <int FMT>
static bool ink2_launch_chunk(const ink2_args & a, int n_pairs, cudaStream_t stream) {
    int n_tab, row_bytes;
    if constexpr (FMT <= INK2_RTR4) {
        const ink2_trellis_params p = ink2_trellis(FMT);
        n_tab = (1 << p.log2K) * p.g;
        row_bytes = a.K * p.k / 8;
    } else if constexpr (FMT == INK2_LM6) {
        n_tab = 64;
        row_bytes = a.K * 6 / 8;
    } else if constexpr (FMT == INK2_LM8) {
        n_tab = 256;
        row_bytes = a.K;
    } else {
        n_tab = 256;
        row_bytes = a.K / 2;
    }
    const int64_t nb_e = a.ids ? a.codes_nb_e : 0;
    if (a.K % 16 != 0 || a.K > INK2_CHUNK_MAXC * WARP_SIZE * 16 || (uintptr_t) a.x % 16 != 0 || row_bytes % 4 != 0 ||
        (uintptr_t) a.codes % 16 != 0 || nb_e % 16 != 0 || (FMT == INK2_LM8 && row_bytes % 16 != 0) ||
        (FMT == INK2_NVFP4 && row_bytes % 8 != 0)) {
        return false;
    }
    const size_t smem = (size_t) (n_tab + a.K) * sizeof(float);
    if (smem > 48 * 1024) {
        return false;
    }
    const int n_eff = a.ids ? std::max(1, n_pairs / 2) : 1;
    static const int rows_div = getenv("GGML_INK2_CHUNK_DIV") ? atoi(getenv("GGML_INK2_CHUNK_DIV")) : 448;
    int rpb = (int) (((int64_t) a.M * n_eff + rows_div - 1) / rows_div);
    // experts: four rows per warp pass; dense matrices (one launch, few rows): two, for more blocks
    static const int nr_dense = getenv("GGML_INK2_CHUNK_NR_DENSE") ? atoi(getenv("GGML_INK2_CHUNK_NR_DENSE")) : 2;
    const int nr = a.ids ? 4 : nr_dense;
    const int pass = nr * INK2_VEC_WARPS;
    rpb = std::max(pass, std::min(std::max(64, pass), (rpb + pass - 1) / pass * pass));
    const dim3 grid((a.M + rpb - 1) / rpb, n_pairs, 1);
    const int nt = INK2_VEC_WARPS * WARP_SIZE;
    switch (nr) {
        case 1:  ink2_mul_mat_chunk_kernel<FMT, 1><<<grid, nt, smem, stream>>>(a, rpb, n_tab, row_bytes); break;
        case 2:  ink2_mul_mat_chunk_kernel<FMT, 2><<<grid, nt, smem, stream>>>(a, rpb, n_tab, row_bytes); break;
        default: ink2_mul_mat_chunk_kernel<FMT, 4><<<grid, nt, smem, stream>>>(a, rpb, n_tab, row_bytes); break;
    }
    return true;
}

// Prefill path (N > INK2_TILE_MIN_N columns): a tiled GEMM. Each block owns BM rows x BN routed columns (experts: the
// (token, slot) pairs of one slab, listed once per op by ink2_route_kernel) and walks K in steps of BK: the block
// decodes the BM x BK weight tile into shared memory as the code table's f32 value (NVFP4: e2m1 x E4M3 block scale),
// stages the BK x BN activation tile, and each thread accumulates a 4 x 4 register tile in f32. The row's f16 scale
// (NVFP4: the expert's outer scale) multiplies the sum after the K loop. Not bitwise equal to ink2_mul_mat_kernel
// (which scales each weight before the dot): rounding-level differences only.

#define INK2_TILE_BM      128
#define INK2_TILE_BN      32
#define INK2_TILE_BK      32
#define INK2_TILE_THREADS 256
#define INK2_TILE_MIN_N   32

static __global__ void ink2_route_kernel(const int32_t * ids, const int16_t * map, int fmt, int n_exp, int n_pairs,
                                         int cap, int * list, int * counts) {
    __shared__ int s_e, s_cnt;
    const int slab = blockIdx.x;
    if (threadIdx.x == 0) {
        s_e = -1;
        s_cnt = 0;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_exp; e += blockDim.x) {
        if (map[2 * e] == fmt && map[2 * e + 1] == slab) {
            s_e = e;
        }
    }
    __syncthreads();
    const int e = s_e;
    if (e >= 0) {
        for (int i = threadIdx.x; i < n_pairs; i += blockDim.x) {
            if (ids[i] == e) {
                const int j = atomicAdd(&s_cnt, 1);
                if (j < cap) {
                    list[(int64_t) slab * cap + j] = i;
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        counts[slab] = min(s_cnt, cap);
    }
}

template <int FMT>
static __global__ void __launch_bounds__(INK2_TILE_THREADS)
ink2_mul_mat_tile_kernel(const ink2_args a, const int * list, const int * counts, const int cap, const int n_tab,
                         const int row_bytes) {
    extern __shared__ float smem[];
    float * tab = smem;
    float * Ws  = tab + n_tab;                                  // [BK][BM + 4]
    float * Xs  = Ws + INK2_TILE_BK * (INK2_TILE_BM + 4);      // [BK][BN + 4]
    __shared__ int s_x[INK2_TILE_BN], s_o[INK2_TILE_BN];

    const bool experts = a.ids != nullptr;
    const uint8_t * codes  = a.codes;
    const uint8_t * scales = a.scales;
    float outer = 1.0f;
    const int c0 = blockIdx.y * INK2_TILE_BN;
    int cnt = a.N;
    const int * lst = nullptr;
    if (experts) {
        const int slab = blockIdx.z;
        cnt = counts[slab];
        if (c0 >= cnt) {
            return;
        }
        lst = list + (int64_t) slab * cap;
        codes  += slab * a.codes_nb_e;
        scales += slab * a.scales_nb_e;
        if constexpr (FMT == INK2_NVFP4) {
            outer = ((const float *) a.table)[slab];
        }
    } else if (c0 >= cnt) {
        return;
    }
    const int K = a.K;
    const int tid = threadIdx.x;
    if (tid < INK2_TILE_BN) {
        const int c = c0 + tid;
        if (c < cnt) {
            if (experts) {
                const int i = lst[c];
                const int t = i / a.n_used, s = i % a.n_used;
                s_o[tid] = i;
                s_x[tid] = t * a.n_xs + (a.n_xs > 1 ? s : 0);
            } else {
                s_o[tid] = c;
                s_x[tid] = c;
            }
        } else {
            s_o[tid] = -1;
            s_x[tid] = -1;
        }
    }
    for (int j = tid; j < n_tab; j += blockDim.x) {
        if constexpr (FMT <= INK2_RTR4) {
            tab[j] = ink2_f16_to_f32(((const uint16_t *) a.table)[j]);
        } else if constexpr (FMT == INK2_NVFP4) {
            tab[j] = a.e4m3[j];
        } else {
            tab[j] = ((const float *) a.table)[j];
        }
    }
    __syncthreads();

    const int r0 = blockIdx.x * INK2_TILE_BM;
    // decode role: one row, 16 consecutive k of the BK step
    const int dr = tid % INK2_TILE_BM, dk = (tid / INK2_TILE_BM) * 16;
    const bool row_ok = r0 + dr < a.M;
    const uint8_t * crow = codes + (int64_t) (r0 + dr) * row_bytes;
    // activation role: one column, 4 consecutive k
    const int xc = tid / 8, xk = (tid % 8) * 4;
    const int xcol = s_x[xc];
    const float * xsrc = xcol >= 0 ? a.x + (int64_t) xcol * K : nullptr;
    // compute role: 4 rows x 4 columns
    const int tx = tid % 8, ty = tid / 8;

    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < K; k0 += INK2_TILE_BK) {
        // weights
        if (row_ok) {
            float * wdst = Ws + dk * (INK2_TILE_BM + 4) + dr;
            const int kq = k0 + dk; // a multiple of 16
            if constexpr (FMT <= INK2_RTR4) {
                constexpr int kb = FMT == INK2_HT1 || FMT == INK2_RTR1 ? 1 : FMT == INK2_HT2 || FMT == INK2_RTR2 ? 2 :
                                   FMT == INK2_RTR4 ? 4 : 3;
                constexpr int g  = FMT == INK2_HT1 || FMT == INK2_HT2 ? 4 : FMT == INK2_HT3 ? 2 : 1;
                constexpr int l2 = FMT == INK2_HT1 || FMT == INK2_HT2 ? 9 : FMT == INK2_HT3 ? 10 : 12;
                const uint32_t * words = (const uint32_t *) crow;
                const int wpr = row_bytes / 4;
                // the 16 weights' windows lie in 3 words from the lowest bit position (that of column kq + 15)
                const int bp0 = (K - 1 - (kq + 15)) * kb;
                const int w0i = bp0 >> 5;
                const uint32_t w0 = __ldg(words + w0i);
                const uint32_t w1 = w0i + 1 < wpr ? __ldg(words + w0i + 1) : 0u; // bits past the end read as zero
                const uint32_t w2 = w0i + 2 < wpr ? __ldg(words + w0i + 2) : 0u;
                const uint64_t lo = (uint64_t) w0 | ((uint64_t) w1 << 32);
                const uint64_t hi = (uint64_t) w1 | ((uint64_t) w2 << 32);
#pragma unroll
                for (int j = 0; j < 16; j += g) {
                    const int rel = (K - 1 - (kq + j + g - 1)) * kb - w0i * 32;
                    uint32_t idx = (uint32_t) ((rel < 32 ? lo >> rel : hi >> (rel - 32)) & 4095u);
                    if constexpr (FMT <= INK2_HT3) {
                        idx = (uint32_t) (idx * 0x9E3779B1u) >> (32 - l2);
                    }
#pragma unroll
                    for (int t = 0; t < g; ++t) {
                        wdst[(j + t) * (INK2_TILE_BM + 4)] = tab[idx * g + t];
                    }
                }
            } else if constexpr (FMT == INK2_NVFP4) {
                const uint2 c = __ldg((const uint2 *) (crow + kq / 2));
                const float bs = tab[__ldg(scales + (int64_t) (r0 + dr) * (K / 16) + kq / 16)];
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const uint32_t nib = ((j < 8 ? c.x : c.y) >> ((j & 7) * 4)) & 0xF;
                    const float mag = (float) ((0xC8643210u >> ((nib & 7) * 4)) & 0xF) * 0.5f;
                    const float v = __int_as_float(__float_as_int(mag) | (int) ((nib & 8) << 28));
                    wdst[j * (INK2_TILE_BM + 4)] = v * bs;
                }
            } else if constexpr (FMT == INK2_LM8) {
                const uint4 c = __ldg((const uint4 *) (crow + kq));
                const uint32_t cw[4] = { c.x, c.y, c.z, c.w };
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    wdst[j * (INK2_TILE_BM + 4)] = tab[(cw[j / 4] >> ((j % 4) * 8)) & 0xFF];
                }
            } else {
                const uint32_t * cp = (const uint32_t *) (crow + kq * 6 / 8);
                const uint64_t lo = (uint64_t) __ldg(cp) | ((uint64_t) __ldg(cp + 1) << 32);
                const uint64_t hi = (uint64_t) __ldg(cp + 1) | ((uint64_t) __ldg(cp + 2) << 32);
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const int b = 6 * j;
                    wdst[j * (INK2_TILE_BM + 4)] = tab[(b < 32 ? lo >> b : hi >> (b - 32)) & 63u];
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                Ws[(dk + j) * (INK2_TILE_BM + 4) + dr] = 0.0f;
            }
        }
        // activations
        {
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (xsrc) {
                v = __ldg((const float4 *) (xsrc + k0 + xk));
            }
            Xs[(xk + 0) * (INK2_TILE_BN + 4) + xc] = v.x;
            Xs[(xk + 1) * (INK2_TILE_BN + 4) + xc] = v.y;
            Xs[(xk + 2) * (INK2_TILE_BN + 4) + xc] = v.z;
            Xs[(xk + 3) * (INK2_TILE_BN + 4) + xc] = v.w;
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < INK2_TILE_BK; ++kk) {
            const float4 w = *(const float4 *) (Ws + kk * (INK2_TILE_BM + 4) + ty * 4);
            const float4 x = *(const float4 *) (Xs + kk * (INK2_TILE_BN + 4) + tx * 4);
            const float wv[4] = { w.x, w.y, w.z, w.w };
            const float xv[4] = { x.x, x.y, x.z, x.w };
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += wv[i] * xv[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = r0 + ty * 4 + i;
        if (r >= a.M) {
            continue;
        }
        float sc = outer;
        if constexpr (FMT != INK2_NVFP4) {
            sc = ink2_f16_to_f32(((const uint16_t *) scales)[r]);
        }
        const int row = a.row_map ? a.row_map[r] : r;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int o = s_o[tx * 4 + j];
            if (o >= 0) {
                a.y[(int64_t) o * a.M_out + row] = acc[i][j] * sc;
            }
        }
    }
}

template <int FMT>
static bool ink2_launch_tile(ggml_backend_cuda_context & ctx, const ink2_args & a, int n_slab, cudaStream_t stream) {
    int n_tab, row_bytes;
    if constexpr (FMT <= INK2_RTR4) {
        const ink2_trellis_params p = ink2_trellis(FMT);
        n_tab = (1 << p.log2K) * p.g;
        row_bytes = a.K * p.k / 8;
    } else if constexpr (FMT == INK2_LM6) {
        n_tab = 64;
        row_bytes = a.K * 6 / 8;
    } else if constexpr (FMT == INK2_LM8) {
        n_tab = 256;
        row_bytes = a.K;
    } else {
        n_tab = 256;
        row_bytes = a.K / 2;
    }
    const int64_t nb_e = a.ids ? a.codes_nb_e : 0;
    if (a.K % INK2_TILE_BK != 0 || (uintptr_t) a.x % 16 != 0 || row_bytes % 4 != 0 || (uintptr_t) a.codes % 16 != 0 ||
        nb_e % 16 != 0 || (FMT == INK2_LM8 && row_bytes % 16 != 0) || (FMT == INK2_NVFP4 && row_bytes % 8 != 0)) {
        return false;
    }
    const size_t smem = (size_t) (n_tab + INK2_TILE_BK * (INK2_TILE_BM + 4) + INK2_TILE_BK * (INK2_TILE_BN + 4)) *
                        sizeof(float);
    if (smem > 48 * 1024) {
        return false;
    }
    ggml_cuda_pool_alloc<int> list(ctx.pool());
    ggml_cuda_pool_alloc<int> counts(ctx.pool());
    int cap = 0;
    if (a.ids) {
        cap = a.N; // an expert is chosen at most once per token
        list.alloc((size_t) n_slab * cap);
        counts.alloc(n_slab);
        ink2_route_kernel<<<n_slab, 256, 0, stream>>>(a.ids, a.map, FMT, a.n_exp, a.n_used * a.N, cap, list.get(),
                                                      counts.get());
    }
    const dim3 grid((a.M + INK2_TILE_BM - 1) / INK2_TILE_BM, (a.N + INK2_TILE_BN - 1) / INK2_TILE_BN,
                    a.ids ? n_slab : 1);
    ink2_mul_mat_tile_kernel<FMT><<<grid, INK2_TILE_THREADS, smem, stream>>>(
        a, a.ids ? list.get() : nullptr, a.ids ? counts.get() : nullptr, cap, n_tab, row_bytes);
    return true;
}

void ggml_cuda_op_ink2_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * codes  = dst->src[0];
    const ggml_tensor * scales = dst->src[1];
    const ggml_tensor * table  = dst->src[2];
    const ggml_tensor * aux    = dst->src[3];
    const ggml_tensor * x      = dst->src[4];
    const ggml_tensor * ids    = dst->src[5];
    const ggml_tensor * map    = dst->src[6];
    cudaStream_t stream = ctx.stream();

    ink2_args a = {};
    const int fmt = ggml_get_op_params_i32(dst, 0);
    a.M = ggml_get_op_params_i32(dst, 1);
    a.K = ggml_get_op_params_i32(dst, 2);
    a.R = std::max(1, std::min(INK2_RMAX, 10240 / a.K));
    a.codes  = (const uint8_t *) codes->data;
    a.scales = (const uint8_t *) scales->data;
    a.table  = table->data;
    a.e4m3   = aux ? (const float *) aux->data : nullptr;
    a.x      = (const float *) x->data;
    a.y      = (float *) dst->data;
    a.row_map = dst->src[7] ? (const int32_t *) dst->src[7]->data : nullptr;
    a.M_out  = dst->ne[0];

    int n_slab = 1;
    if (ids) {
        a.ids    = (const int32_t *) ids->data;
        a.map    = (const int16_t *) map->data;
        a.n_used = ids->ne[0];
        a.N      = ids->ne[1];
        a.n_xs   = x->ne[1];
        a.n_exp  = map->ne[1];
        a.codes_nb_e  = codes->nb[2];
        a.scales_nb_e = fmt == INK2_NVFP4 ? scales->nb[2] : scales->nb[1];
        n_slab = fmt == INK2_NVFP4 ? codes->ne[2] : codes->ne[2];
        if (!ggml_get_op_params_i32(dst, 3)) {
            CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));
        }
    } else {
        a.N = dst->ne[1];
        a.n_used = 1;
        a.n_xs = 1;
    }

    static const bool no_chunk = getenv("GGML_INK2_NO_CHUNK") != nullptr;
    if (!no_chunk && (ids ? a.N <= INK2_VEC_NC : a.N == 1)) {
        const int n_pairs = ids ? a.n_used * a.N : 1;
        bool done = false;
        switch (fmt) {
            case INK2_HT1:   done = ink2_launch_chunk<INK2_HT1>  (a, n_pairs, stream); break;
            case INK2_HT2:   done = ink2_launch_chunk<INK2_HT2>  (a, n_pairs, stream); break;
            case INK2_HT3:   done = ink2_launch_chunk<INK2_HT3>  (a, n_pairs, stream); break;
            case INK2_RTR1:  done = ink2_launch_chunk<INK2_RTR1> (a, n_pairs, stream); break;
            case INK2_RTR2:  done = ink2_launch_chunk<INK2_RTR2> (a, n_pairs, stream); break;
            case INK2_RTR3:  done = ink2_launch_chunk<INK2_RTR3> (a, n_pairs, stream); break;
            case INK2_RTR4:  done = ink2_launch_chunk<INK2_RTR4> (a, n_pairs, stream); break;
            case INK2_NVFP4: done = ink2_launch_chunk<INK2_NVFP4>(a, n_pairs, stream); break;
            case INK2_LM6:   done = ink2_launch_chunk<INK2_LM6>  (a, n_pairs, stream); break;
            case INK2_LM8:   done = ink2_launch_chunk<INK2_LM8>  (a, n_pairs, stream); break;
            default: GGML_ABORT("ink2: unknown format %d", fmt);
        }
        if (done) {
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }

    static const bool no_vec = getenv("GGML_INK2_NO_VEC") != nullptr;
    if (!no_vec && a.N <= INK2_VEC_NC) {
        const int n_pairs = ids ? a.n_used * a.N : 1;
        bool done = false;
        switch (fmt) {
            case INK2_HT1:   done = ink2_launch_vec<INK2_HT1>  (a, n_pairs, stream); break;
            case INK2_HT2:   done = ink2_launch_vec<INK2_HT2>  (a, n_pairs, stream); break;
            case INK2_HT3:   done = ink2_launch_vec<INK2_HT3>  (a, n_pairs, stream); break;
            case INK2_RTR1:  done = ink2_launch_vec<INK2_RTR1> (a, n_pairs, stream); break;
            case INK2_RTR2:  done = ink2_launch_vec<INK2_RTR2> (a, n_pairs, stream); break;
            case INK2_RTR3:  done = ink2_launch_vec<INK2_RTR3> (a, n_pairs, stream); break;
            case INK2_RTR4:  done = ink2_launch_vec<INK2_RTR4> (a, n_pairs, stream); break;
            case INK2_NVFP4: done = ink2_launch_vec<INK2_NVFP4>(a, n_pairs, stream); break;
            case INK2_LM6:   done = ink2_launch_vec<INK2_LM6>  (a, n_pairs, stream); break;
            case INK2_LM8:   done = ink2_launch_vec<INK2_LM8>  (a, n_pairs, stream); break;
            default: GGML_ABORT("ink2: unknown format %d", fmt);
        }
        if (done) {
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }

    static const bool no_tile = getenv("GGML_INK2_NO_TILE") != nullptr;
    if (!no_tile && a.N > INK2_TILE_MIN_N) {
        bool done = false;
        switch (fmt) {
            case INK2_HT1:   done = ink2_launch_tile<INK2_HT1>  (ctx, a, n_slab, stream); break;
            case INK2_HT2:   done = ink2_launch_tile<INK2_HT2>  (ctx, a, n_slab, stream); break;
            case INK2_HT3:   done = ink2_launch_tile<INK2_HT3>  (ctx, a, n_slab, stream); break;
            case INK2_RTR1:  done = ink2_launch_tile<INK2_RTR1> (ctx, a, n_slab, stream); break;
            case INK2_RTR2:  done = ink2_launch_tile<INK2_RTR2> (ctx, a, n_slab, stream); break;
            case INK2_RTR3:  done = ink2_launch_tile<INK2_RTR3> (ctx, a, n_slab, stream); break;
            case INK2_RTR4:  done = ink2_launch_tile<INK2_RTR4> (ctx, a, n_slab, stream); break;
            case INK2_NVFP4: done = ink2_launch_tile<INK2_NVFP4>(ctx, a, n_slab, stream); break;
            case INK2_LM6:   done = ink2_launch_tile<INK2_LM6>  (ctx, a, n_slab, stream); break;
            case INK2_LM8:   done = ink2_launch_tile<INK2_LM8>  (ctx, a, n_slab, stream); break;
            default: GGML_ABORT("ink2: unknown format %d", fmt);
        }
        if (done) {
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }

    const dim3 grid((a.M + a.R - 1) / a.R, n_slab, 1);
    const size_t smem = (size_t) a.R * a.K * sizeof(float) + (ids ? (size_t) a.N * sizeof(int) : 0);
    GGML_ASSERT(smem <= 48 * 1024);

    switch (fmt) {
        case INK2_HT1:   ink2_mul_mat_kernel<INK2_HT1>  <<<grid, INK2_THREADS, smem, stream>>>(a); break;
        case INK2_HT2:   ink2_mul_mat_kernel<INK2_HT2>  <<<grid, INK2_THREADS, smem, stream>>>(a); break;
        case INK2_HT3:   ink2_mul_mat_kernel<INK2_HT3>  <<<grid, INK2_THREADS, smem, stream>>>(a); break;
        case INK2_RTR1:  ink2_mul_mat_kernel<INK2_RTR1> <<<grid, INK2_THREADS, smem, stream>>>(a); break;
        case INK2_RTR2:  ink2_mul_mat_kernel<INK2_RTR2> <<<grid, INK2_THREADS, smem, stream>>>(a); break;
        case INK2_RTR3:  ink2_mul_mat_kernel<INK2_RTR3> <<<grid, INK2_THREADS, smem, stream>>>(a); break;
        case INK2_RTR4:  ink2_mul_mat_kernel<INK2_RTR4> <<<grid, INK2_THREADS, smem, stream>>>(a); break;
        case INK2_NVFP4: ink2_mul_mat_kernel<INK2_NVFP4><<<grid, INK2_THREADS, smem, stream>>>(a); break;
        case INK2_LM6:   ink2_mul_mat_kernel<INK2_LM6>  <<<grid, INK2_THREADS, smem, stream>>>(a); break;
        case INK2_LM8:   ink2_mul_mat_kernel<INK2_LM8>  <<<grid, INK2_THREADS, smem, stream>>>(a); break;
        default: GGML_ABORT("ink2: unknown format %d", fmt);
    }
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void ink2_rel_bias_kernel(float * y, const char * table, const char * idx, int64_t n_kv, int64_t n_q,
                                            int64_t n_head, int64_t tnb1, int64_t tnb2, int64_t inb1) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_kv * n_q * n_head) {
        return;
    }
    const int64_t j = i % n_kv;
    const int64_t q = (i / n_kv) % n_q;
    const int64_t h = i / (n_kv * n_q);
    const int32_t b = ((const int32_t *) (idx + q * inb1))[j];
    y[i] += *(const float *) (table + q * tnb2 + b * tnb1 + h * sizeof(float));
}

void ggml_cuda_op_ink2_rel_bias(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * table = dst->src[1];
    const ggml_tensor * idx   = dst->src[2];
    const int64_t n = ggml_nelements(dst);
    const int threads = 256;
    ink2_rel_bias_kernel<<<(n + threads - 1) / threads, threads, 0, ctx.stream()>>>(
        (float *) dst->data, (const char *) table->data, (const char *) idx->data, dst->ne[0], dst->ne[1], dst->ne[2],
        table->nb[1], table->nb[2], idx->nb[1]);
    CUDA_CHECK(cudaGetLastError());
}

// GGML_OP_INK2_RHT: one block per column. x * signs_in in shared memory with the B factor (given transposed, so a
// straight copy), each b-long slice dotted with B four outputs a thread (one x read for four products); then the
// Sylvester factor as butterflies over the a index, one warp per column of the [a, b] tile (lanes hold the a index,
// shuffles for the strides under 32, registers above), and the scale and signs_out on the way out.
#define INK2_RHT_THREADS 1024
#define INK2_RHT_J       8  // most a/32 (a = 256)

static __global__ void __launch_bounds__(INK2_RHT_THREADS)
ink2_rht_kernel(const float * x, const float * Bt, const float * si, const float * so, float * y,
                int n, int a, int b, float scale, bool aligned) {
    extern __shared__ float smem[];
    float * xs = smem, * z = smem + n, * Bs = smem + 2 * n;
    const int64_t c = blockIdx.x;
    x += c * n;
    y += c * n;
    if (aligned) {
        for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
            float4 v = ((const float4 *) x)[i];
            if (si) {
                const float4 s4 = ((const float4 *) si)[i];
                v.x *= s4.x; v.y *= s4.y; v.z *= s4.z; v.w *= s4.w;
            }
            ((float4 *) xs)[i] = v;
        }
        for (int i = threadIdx.x; i < b * b / 4; i += blockDim.x) {
            ((float4 *) Bs)[i] = ((const float4 *) Bt)[i];
        }
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            xs[i] = si ? x[i] * si[i] : x[i];
        }
        for (int i = threadIdx.x; i < b * b; i += blockDim.x) {
            Bs[i] = Bt[i];
        }
    }
    __syncthreads();
    const int b4 = b / 4;
    for (int t = threadIdx.x; t < a * b4; t += blockDim.x) {
        const int ak = t / b4, bi = (t - ak * b4) * 4;
        const float * xr = xs + ak * b;
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int bk = 0; bk < b; ++bk) {
            const float xv = xr[bk];
            const float4 bv = *(const float4 *) (Bs + bk * b + bi);
            acc.x += bv.x * xv;
            acc.y += bv.y * xv;
            acc.z += bv.z * xv;
            acc.w += bv.w * xv;
        }
        *(float4 *) (z + ak * b + bi) = acc;
    }
    __syncthreads();
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int n_warps = blockDim.x / WARP_SIZE;
    const int aw = min(a, WARP_SIZE);           // lanes per column
    const int cpw = WARP_SIZE / aw;             // columns per warp (a = 16: two)
    const int J = a / aw;                       // values per lane
    const int sub = lane / aw, l = lane % aw;
    for (int bi0 = warp * cpw; bi0 < b; bi0 += n_warps * cpw) {
        const int bi = bi0 + sub;
        const bool ok = bi < b;
        float v[INK2_RHT_J];
#pragma unroll
        for (int j = 0; j < INK2_RHT_J; ++j) {
            v[j] = ok && j < J ? z[(l + aw * j) * b + bi] : 0.0f;
        }
        for (int h = 1; h < aw; h *= 2) {
#pragma unroll
            for (int j = 0; j < INK2_RHT_J; ++j) {
                const float o = __shfl_xor_sync(0xffffffff, v[j], h);
                v[j] = (l & h) ? o - v[j] : v[j] + o;
            }
        }
#pragma unroll
        for (int h = 1; h < INK2_RHT_J; h *= 2) {
#pragma unroll
            for (int j = 0; j < INK2_RHT_J; ++j) {
                if ((j & h) == 0 && j + h < J) {
                    const float u = v[j], w = v[j + h];
                    v[j] = u + w;
                    v[j + h] = u - w;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < INK2_RHT_J; ++j) {
            if (ok && j < J) {
                z[(l + aw * j) * b + bi] = v[j];
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        y[i] = so ? (z[i] * scale) * so[i] : z[i] * scale;
    }
}

// The same rotation split over the B index: block (c, g) owns the columns bi = 4 g .. 4 g + 3 of z = X B^T (X the
// column as [a, b]), so its A transform (over ak, per bi) needs no other block. Every output is computed with the same
// operations in the same order as ink2_rht_kernel's, so the two agree bitwise; one decode column now spreads over b/4
// SMs instead of one.
#define INK2_RHT_SPLIT_THREADS 128
#define INK2_RHT_BW            4

struct ink2_rht_sets {
    const float * si[5]; // ggml_ink2_rht_multi: set blockIdx.z's input signs
};

static __global__ void __launch_bounds__(INK2_RHT_SPLIT_THREADS)
ink2_rht_split_kernel(const float * x, const float * Bt, const float * si, const float * so, float * y,
                      int n, int a, int b, float scale, const ink2_rht_sets sets, int64_t cnt) {
    extern __shared__ float smem[];
    float * xs = smem, * z = smem + n, * Bs = z + a * INK2_RHT_BW;
    const int64_t c = blockIdx.x;
    const int bi0 = blockIdx.y * INK2_RHT_BW;
    if (gridDim.z > 1) {
        si = sets.si[blockIdx.z];
        y += blockIdx.z * cnt * n;
    }
    x += c * n;
    y += c * n;
    for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
        float4 v = ((const float4 *) x)[i];
        if (si) {
            const float4 s4 = ((const float4 *) si)[i];
            v.x *= s4.x; v.y *= s4.y; v.z *= s4.z; v.w *= s4.w;
        }
        ((float4 *) xs)[i] = v;
    }
    for (int bk = threadIdx.x; bk < b; bk += blockDim.x) {
        ((float4 *) Bs)[bk] = *(const float4 *) (Bt + bk * b + bi0);
    }
    __syncthreads();
    for (int ak = threadIdx.x; ak < a; ak += blockDim.x) {
        const float * xr = xs + ak * b;
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int bk = 0; bk < b; ++bk) {
            const float xv = xr[bk];
            const float4 bv = ((const float4 *) Bs)[bk];
            acc.x += bv.x * xv;
            acc.y += bv.y * xv;
            acc.z += bv.z * xv;
            acc.w += bv.w * xv;
        }
        ((float4 *) z)[ak] = acc; // z [a, BW]
    }
    __syncthreads();
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int n_warps = blockDim.x / WARP_SIZE;
    const int aw = min(a, WARP_SIZE);
    const int cpw = WARP_SIZE / aw;
    const int J = a / aw;
    const int sub = lane / aw, l = lane % aw;
    for (int bl0 = warp * cpw; bl0 < INK2_RHT_BW; bl0 += n_warps * cpw) {
        const int bl = bl0 + sub;
        const bool ok = bl < INK2_RHT_BW;
        float v[INK2_RHT_J];
#pragma unroll
        for (int j = 0; j < INK2_RHT_J; ++j) {
            v[j] = ok && j < J ? z[(l + aw * j) * INK2_RHT_BW + bl] : 0.0f;
        }
        for (int h = 1; h < aw; h *= 2) {
#pragma unroll
            for (int j = 0; j < INK2_RHT_J; ++j) {
                const float o = __shfl_xor_sync(0xffffffff, v[j], h);
                v[j] = (l & h) ? o - v[j] : v[j] + o;
            }
        }
#pragma unroll
        for (int h = 1; h < INK2_RHT_J; h *= 2) {
#pragma unroll
            for (int j = 0; j < INK2_RHT_J; ++j) {
                if ((j & h) == 0 && j + h < J) {
                    const float u = v[j], w = v[j + h];
                    v[j] = u + w;
                    v[j + h] = u - w;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < INK2_RHT_J; ++j) {
            if (ok && j < J) {
                const int i = (l + aw * j) * b + bi0 + bl;
                y[i] = so ? (v[j] * scale) * so[i] : v[j] * scale;
            }
        }
    }
}

void ggml_cuda_op_ink2_rht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * B = dst->src[1];
    const int n = (int) x->ne[0], b = (int) B->ne[0];
    const int a = ggml_get_op_params_i32(dst, 0);
    const float scale = ggml_get_op_params_f32(dst, 1);
    const int n_sets = std::max(1, ggml_get_op_params_i32(dst, 2));
    const int64_t cnt = ggml_nelements(x) / n;
    const size_t smem = (size_t) (2 * n + b * b) * sizeof(float);
    GGML_ASSERT(smem <= 48 * 1024 && cnt <= INT_MAX && b % 4 == 0 && n % 4 == 0 && a <= 32 * INK2_RHT_J);
    const float * so = dst->src[3] ? (const float *) dst->src[3]->data : nullptr;
    ink2_rht_sets sets = {};
    bool aligned = (uintptr_t) x->data % 16 == 0 && (uintptr_t) B->data % 16 == 0;
    for (int z = 0; z < n_sets; ++z) {
        const ggml_tensor * st = dst->src[z == 0 ? 2 : 3 + z];
        sets.si[z] = st ? (const float *) st->data : nullptr;
        aligned = aligned && (uintptr_t) sets.si[z] % 16 == 0;
    }
    const float * si = sets.si[0];
    static const bool no_split = getenv("GGML_INK2_RHT_NO_SPLIT") != nullptr;
    const size_t smem_s = (size_t) (n + a * INK2_RHT_BW + b * INK2_RHT_BW) * sizeof(float);
    if (!no_split && aligned && b % INK2_RHT_BW == 0 && smem_s <= 48 * 1024) {
        ink2_rht_split_kernel<<<dim3((unsigned) cnt, b / INK2_RHT_BW, n_sets), INK2_RHT_SPLIT_THREADS, smem_s,
                                ctx.stream()>>>(
            (const float *) x->data, (const float *) B->data, si, so, (float *) dst->data, n, a, b, scale, sets, cnt);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    for (int z = 0; z < n_sets; ++z) {
        const bool al = (uintptr_t) x->data % 16 == 0 && (uintptr_t) B->data % 16 == 0 && (uintptr_t) sets.si[z] % 16 == 0;
        ink2_rht_kernel<<<(unsigned) cnt, INK2_RHT_THREADS, smem, ctx.stream()>>>(
            (const float *) x->data, (const float *) B->data, sets.si[z], so, (float *) dst->data + z * cnt * n, n, a, b,
            scale, al);
    }
    CUDA_CHECK(cudaGetLastError());
}
