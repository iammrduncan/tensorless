// sm_60 IQ3_XXS GEMV (P100 v2 K-B v0.1). Self-contained device code; its standalone harness is in the research repository.
//
// The activation is converted to fp16 in 32-element blocks with an exact power-of-two scale; the IQ3_XXS grid and the
// 7-bit sign fields are expanded to shared memory, so dequantize + dot runs in HFMA2 with no global table traffic.
// R rows per warp share each activation load. T = 1..4 columns reuse one dequant (verify widths).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace sm60mmv {

constexpr int QK       = 256;  // weights per super-block
constexpr int SB_BYTES = 98;   // sizeof(block_iq3_xxs)
constexpr int SB_BYTES_S = 110;   // sizeof(block_iq3_s)
constexpr int SB_BYTES_S_REP = 112;  // padded so a repacked super-block starts 4-aligned for every block index
constexpr int NW       = 4;    // warps per CTA
constexpr int NTHREADS = NW*32;

struct Tables {
    uint2 grid[256];  // iq3xxs_grid entry -> 4 fp16 magnitudes (w0,w1 | w2,w3)
    uint4 sign[128];  // 7-bit sign field -> four half2 XOR masks
};

__device__ __forceinline__ uint32_t h2u(const half2 h) { return *reinterpret_cast<const uint32_t *>(&h); }
__device__ __forceinline__ half2    u2h(const uint32_t u) { return *reinterpret_cast<const half2 *>(&u); }

__device__ __forceinline__ void build_tables(Tables & tab) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        const uint32_t w = __ldg(iq3xxs_grid + i);
        const half2 lo = __halves2half2(__ushort2half_rn(w & 0xFF), __ushort2half_rn((w >> 8) & 0xFF));
        const half2 hi = __halves2half2(__ushort2half_rn((w >> 16) & 0xFF), __ushort2half_rn(w >> 24));
        tab.grid[i] = make_uint2(h2u(lo), h2u(hi));
    }
    for (int i = threadIdx.x; i < 128; i += blockDim.x) {
        const uint32_t s = __ldg(ksigns_iq2xs + i);
        uint32_t m[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            m[j] = (((s >> (2*j)) & 1u) << 15) | (((s >> (2*j + 1)) & 1u) << 31);
        }
        tab.sign[i] = make_uint4(m[0], m[1], m[2], m[3]);
    }
}

// x (f32, T contiguous columns of k) -> xh (fp16, |xh| < 1) + one power-of-two scale per 32 elements. One warp per block.
__global__ void convert_x(const float * __restrict__ x, half * __restrict__ xh, float * __restrict__ xs, int nblk_total) {
    const int blk  = blockIdx.x*(blockDim.x/32) + threadIdx.x/32;
    const int lane = threadIdx.x % 32;
    if (blk >= nblk_total) {
        return;
    }
    const float v = x[blk*32 + lane];
    float a = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFF, a, off));
    }
    int e = 0;
    frexpf(a, &e);
    const float s = a > 0.0f ? ldexpf(1.0f, e) : 1.0f;
    xh[blk*32 + lane] = __float2half_rn(v / s);
    if (lane == 0) {
        xs[blk] = s;
    }
}

// default off: allow K-B to handle non-repacked IQ3_XXS at ncols 2..3 (see mmvh-sm60.cu)
inline bool kb_wide_xxs() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_MMV_SM60_XXS_KB_WIDE");
        return e != nullptr && atoi(e) != 0;
    }();
    return on;
}

// y[t*m + row] = sum_j W[row, j] * x[t*k + j]. W rows are row_bytes apart (ggml nb[1]); k % 256 == 0.
template <int T, int R>
__global__ void __launch_bounds__(NTHREADS)
gemv_iq3xxs_rows(const uint8_t * __restrict__ W, const int64_t row_bytes, const half * __restrict__ xh,
                 const float * __restrict__ xs, float * __restrict__ y, const int m, const int k) {
    __shared__ Tables tab;
    build_tables(tab);
    __syncthreads();

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int nsub = k / 32;

    for (int row0 = (blockIdx.x*NW + warp)*R; row0 < m; row0 += gridDim.x*NW*R) {
        float acc[R][T];
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int t = 0; t < T; ++t) {
                acc[r][t] = 0.0f;
            }
        }
        for (int s = lane; s < nsub; s += 32) {
            uint4 xv[T][4];
            float xsc[T];
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(t)*k + 32*s);
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    xv[t][l] = __ldg(xp + l);
                }
                xsc[t] = __ldg(xs + t*nsub + s);
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (row0 + r >= m) {
                    break;
                }
                const uint8_t  * blk = W + int64_t(row0 + r)*row_bytes + (s >> 3)*SB_BYTES;
                const int        ib  = s & 7;
                const uint16_t * q16 = reinterpret_cast<const uint16_t *>(blk + 2 + 8*ib);
                const uint16_t * a16 = reinterpret_cast<const uint16_t *>(blk + 2 + 64 + 4*ib);
                const uint32_t q0  = __ldg(q16 + 0) | (uint32_t(__ldg(q16 + 1)) << 16);
                const uint32_t q1  = __ldg(q16 + 2) | (uint32_t(__ldg(q16 + 3)) << 16);
                const uint32_t aux = __ldg(a16 + 0) | (uint32_t(__ldg(a16 + 1)) << 16);
                const float    d   = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t *>(blk))));
                const float    db  = d*(0.5f + float(aux >> 28))*0.5f;
                half2 h[T];
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    h[t] = __float2half2_rn(0.0f);
                }
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const uint32_t qw = l < 2 ? q0 : q1;
                    const uint2 g0 = tab.grid[(qw >> (16*(l & 1)))     & 0xFF];
                    const uint2 g1 = tab.grid[(qw >> (16*(l & 1) + 8)) & 0xFF];
                    const uint4 sm = tab.sign[(aux >> (7*l)) & 127];
                    const half2 w0 = u2h(g0.x ^ sm.x);
                    const half2 w1 = u2h(g0.y ^ sm.y);
                    const half2 w2 = u2h(g1.x ^ sm.z);
                    const half2 w3 = u2h(g1.y ^ sm.w);
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        h[t] = __hfma2(w0, u2h(xv[t][l].x), h[t]);
                        h[t] = __hfma2(w1, u2h(xv[t][l].y), h[t]);
                        h[t] = __hfma2(w2, u2h(xv[t][l].z), h[t]);
                        h[t] = __hfma2(w3, u2h(xv[t][l].w), h[t]);
                    }
                }
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    const float2 f = __half22float2(h[t]);
                    acc[r][t] += (f.x + f.y)*db*xsc[t];
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int t = 0; t < T; ++t) {
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    acc[r][t] += __shfl_xor_sync(0xFFFFFFFF, acc[r][t], off);
                }
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (row0 + r < m) {
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        y[int64_t(t)*m + row0 + r] = acc[r][t];
                    }
                }
            }
        }
    }
}

// IQ3_S: 512-entry grid + explicit sign bytes; 16 bytes of block metadata per 32 weights.
struct TablesS {
    uint2 grid[512];  // iq3s_grid entry -> 4 fp16 magnitudes
    uint4 sign[256];  // sign byte -> four half2 XOR masks
};

__device__ __forceinline__ void build_tables_s(TablesS & tab) {
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        const uint32_t w = __ldg(iq3s_grid + i);
        const half2 lo = __halves2half2(__ushort2half_rn(w & 0xFF), __ushort2half_rn((w >> 8) & 0xFF));
        const half2 hi = __halves2half2(__ushort2half_rn((w >> 16) & 0xFF), __ushort2half_rn(w >> 24));
        tab.grid[i] = make_uint2(h2u(lo), h2u(hi));
    }
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        uint32_t m[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            m[j] = (((i >> (2*j)) & 1u) << 15) | (((i >> (2*j + 1)) & 1u) << 31);
        }
        tab.sign[i] = make_uint4(m[0], m[1], m[2], m[3]);
    }
}

// IQ3_S GEMV: same HFMA2 inner loop as IQ3_XXS, different index/scale/sign packing.
template <int T, int R>
__global__ void __launch_bounds__(NTHREADS)
gemv_iq3s_rows(const uint8_t * __restrict__ W, const int64_t row_bytes, const half * __restrict__ xh,
               const float * __restrict__ xs, float * __restrict__ y, const int m, const int k) {
    __shared__ TablesS tab;
    build_tables_s(tab);
    __syncthreads();

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int nsub = k / 32;

    for (int row0 = (blockIdx.x*NW + warp)*R; row0 < m; row0 += gridDim.x*NW*R) {
        float acc[R][T];
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int t = 0; t < T; ++t) {
                acc[r][t] = 0.0f;
            }
        }
        for (int s = lane; s < nsub; s += 32) {
            uint4 xv[T][4];
            float xsc[T];
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(t)*k + 32*s);
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    xv[t][l] = __ldg(xp + l);
                }
                xsc[t] = __ldg(xs + t*nsub + s);
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (row0 + r >= m) {
                    break;
                }
                const uint8_t * blk = W + int64_t(row0 + r)*row_bytes + (s >> 3)*SB_BYTES_S;
                const int       ib  = s & 7;
                const uint8_t * qsb = blk + 2 + 8*ib;
                const uint8_t * sgn = blk + 2 + 64 + 8 + 4*ib;
                const uint32_t  qh  = blk[2 + 64 + ib];
                const uint32_t  sc  = blk[2 + 64 + 8 + 32 + (ib >> 1)];
                const float     d   = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t *>(blk))));
                const float     db  = d*(1.0f + 2.0f*float((sc >> (4*(ib & 1))) & 0xf));
                half2 h[T];
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    h[t] = __float2half2_rn(0.0f);
                }
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const uint32_t g1i = uint32_t(__ldg(qsb + 2*l + 0)) | ((qh << (8 - 2*l)) & 256u);
                    const uint32_t g2i = uint32_t(__ldg(qsb + 2*l + 1)) | ((qh << (7 - 2*l)) & 256u);
                    const uint2 g0 = tab.grid[g1i];
                    const uint2 g1 = tab.grid[g2i];
                    const uint4 sm = tab.sign[__ldg(sgn + l)];
                    const half2 w0 = u2h(g0.x ^ sm.x);
                    const half2 w1 = u2h(g0.y ^ sm.y);
                    const half2 w2 = u2h(g1.x ^ sm.z);
                    const half2 w3 = u2h(g1.y ^ sm.w);
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        h[t] = __hfma2(w0, u2h(xv[t][l].x), h[t]);
                        h[t] = __hfma2(w1, u2h(xv[t][l].y), h[t]);
                        h[t] = __hfma2(w2, u2h(xv[t][l].z), h[t]);
                        h[t] = __hfma2(w3, u2h(xv[t][l].w), h[t]);
                    }
                }
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    const float2 f = __half22float2(h[t]);
                    acc[r][t] += (f.x + f.y)*db*xsc[t];
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int t = 0; t < T; ++t) {
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    acc[r][t] += __shfl_xor_sync(0xFFFFFFFF, acc[r][t], off);
                }
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (row0 + r < m) {
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        y[int64_t(t)*m + row0 + r] = acc[r][t];
                    }
                }
            }
        }
    }
}

// T = 1..4 columns; R = 4 rows/warp for T <= 2, 2 for T = 3..4.
inline cudaError_t launch(const uint8_t * W, int64_t row_bytes, const float * x, half * xh, float * xs, float * y,
                          int m, int k, int T, cudaStream_t stream, bool convert = true) {
    const int nblk_total = T*(k/32);
    if (convert) { convert_x<<<(nblk_total + 7)/8, 256, 0, stream>>>(x, xh, xs, nblk_total); }
    const int R = T <= 2 ? 4 : 2;
    const int blocks = (m + NW*R - 1)/(NW*R);
    switch (T) {
        case 1: gemv_iq3xxs_rows<1, 4><<<blocks, NTHREADS, 0, stream>>>(W, row_bytes, xh, xs, y, m, k); break;
        case 2: gemv_iq3xxs_rows<2, 4><<<blocks, NTHREADS, 0, stream>>>(W, row_bytes, xh, xs, y, m, k); break;
        case 3: gemv_iq3xxs_rows<3, 2><<<blocks, NTHREADS, 0, stream>>>(W, row_bytes, xh, xs, y, m, k); break;
        case 4: gemv_iq3xxs_rows<4, 2><<<blocks, NTHREADS, 0, stream>>>(W, row_bytes, xh, xs, y, m, k); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

// IQ3_S launcher: same T/R policy as the IQ3_XXS path.
inline cudaError_t launch_iq3s(const uint8_t * W, int64_t row_bytes, const float * x, half * xh, float * xs, float * y,
                               int m, int k, int T, cudaStream_t stream, bool convert = true) {
    const int nblk_total = T*(k/32);
    if (convert) { convert_x<<<(nblk_total + 7)/8, 256, 0, stream>>>(x, xh, xs, nblk_total); }
    const int R = T <= 2 ? 4 : 2;
    const int blocks = (m + NW*R - 1)/(NW*R);
    switch (T) {
        case 1: gemv_iq3s_rows<1, 4><<<blocks, NTHREADS, 0, stream>>>(W, row_bytes, xh, xs, y, m, k); break;
        case 2: gemv_iq3s_rows<2, 4><<<blocks, NTHREADS, 0, stream>>>(W, row_bytes, xh, xs, y, m, k); break;
        case 3: gemv_iq3s_rows<3, 2><<<blocks, NTHREADS, 0, stream>>>(W, row_bytes, xh, xs, y, m, k); break;
        case 4: gemv_iq3s_rows<4, 2><<<blocks, NTHREADS, 0, stream>>>(W, row_bytes, xh, xs, y, m, k); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

// IQ3_XXS repack: original 98-byte block -> 100-byte block with the 8 sub-blocks' metadata as 12-byte
// 4-aligned records, followed by d: [8 x 12][d(2)][pad(2)]. The record is {qs[8], gas[4]}; row stride is
// (k/256)*100, so every record is 4-aligned and a lane loads 3 uint32 instead of 7 two-byte pieces.
__global__ void repack_iq3xxs(const uint8_t * __restrict__ src, uint8_t * __restrict__ dst, int nblocks) {
    const int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b >= nblocks) {
        return;
    }
    const uint8_t * s = src + (int64_t) b * SB_BYTES;
    uint8_t       * d = dst + (int64_t) b * 100;
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        const uint16_t * q = reinterpret_cast<const uint16_t *>(s + 2 + 8*ib);
        const uint16_t * a = reinterpret_cast<const uint16_t *>(s + 2 + 64 + 4*ib);
        uint32_t *       r = reinterpret_cast<uint32_t *>(d + 12*ib);
        r[0] = __ldg(q + 0) | (uint32_t(__ldg(q + 1)) << 16);
        r[1] = __ldg(q + 2) | (uint32_t(__ldg(q + 3)) << 16);
        r[2] = __ldg(a + 0) | (uint32_t(__ldg(a + 1)) << 16);
    }
    *reinterpret_cast<uint16_t *>(d + 96) = __ldg(reinterpret_cast<const uint16_t *>(s));
}

// Same as gemv_iq3xxs_rows but over the repacked 100-byte blocks.
template <int T, int R>
__global__ void __launch_bounds__(NTHREADS)
gemv_iq3xxs_rows_rep(const uint8_t * __restrict__ W, const int64_t row_bytes, const half * __restrict__ xh,
                     const float * __restrict__ xs, float * __restrict__ y, const int m, const int k) {
    __shared__ Tables tab;
    build_tables(tab);
    __syncthreads();

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int nsub = k / 32;

    for (int row0 = (blockIdx.x*NW + warp)*R; row0 < m; row0 += gridDim.x*NW*R) {
        float acc[R][T];
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int t = 0; t < T; ++t) {
                acc[r][t] = 0.0f;
            }
        }
        for (int s = lane; s < nsub; s += 32) {
            uint4 xv[T][4];
            float xsc[T];
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(t)*k + 32*s);
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    xv[t][l] = __ldg(xp + l);
                }
                xsc[t] = __ldg(xs + t*nsub + s);
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (row0 + r >= m) {
                    break;
                }
                const uint8_t  * blk = W + int64_t(row0 + r)*row_bytes + (s >> 3)*100;
                const int        ib  = s & 7;
                const uint32_t * rec = reinterpret_cast<const uint32_t *>(blk + 12*ib);
                const uint32_t q0  = __ldg(rec + 0);
                const uint32_t q1  = __ldg(rec + 1);
                const uint32_t aux = __ldg(rec + 2);
                const float    d   = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t *>(blk + 96))));
                const float    db  = d*(0.5f + float(aux >> 28))*0.5f;
                half2 h[T];
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    h[t] = __float2half2_rn(0.0f);
                }
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const uint32_t qw = l < 2 ? q0 : q1;
                    const uint2 g0 = tab.grid[(qw >> (16*(l & 1)))     & 0xFF];
                    const uint2 g1 = tab.grid[(qw >> (16*(l & 1) + 8)) & 0xFF];
                    const uint4 sm = tab.sign[(aux >> (7*l)) & 127];
                    const half2 w0 = u2h(g0.x ^ sm.x);
                    const half2 w1 = u2h(g0.y ^ sm.y);
                    const half2 w2 = u2h(g1.x ^ sm.z);
                    const half2 w3 = u2h(g1.y ^ sm.w);
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        h[t] = __hfma2(w0, u2h(xv[t][l].x), h[t]);
                        h[t] = __hfma2(w1, u2h(xv[t][l].y), h[t]);
                        h[t] = __hfma2(w2, u2h(xv[t][l].z), h[t]);
                        h[t] = __hfma2(w3, u2h(xv[t][l].w), h[t]);
                    }
                }
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    const float2 f = __half22float2(h[t]);
                    acc[r][t] += (f.x + f.y)*db*xsc[t];
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int t = 0; t < T; ++t) {
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    acc[r][t] += __shfl_xor_sync(0xFFFFFFFF, acc[r][t], off);
                }
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (row0 + r < m) {
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        y[int64_t(t)*m + row0 + r] = acc[r][t];
                    }
                }
            }
        }
    }
}

inline cudaError_t launch_iq3xxs_rep(const uint8_t * W, int64_t row_bytes, const float * x, half * xh, float * xs, float * y,
                                     int m, int k, int T, cudaStream_t stream, bool convert = true) {
    // R-08 diagnostic: for T=2 run the T=1 body once per column instead. If the WEIGHTS case then
    // passes, the T=2 template body is wrong; if it still fails, the caller's operands are.
    static const bool split_t = getenv("GGML_CUDA_MMV_SM60_SPLIT_T") != nullptr;
    if (split_t && T == 2) {
        const int nblk_total = 2*(k/32);
        if (convert) { convert_x<<<(nblk_total + 7)/8, 256, 0, stream>>>(x, xh, xs, nblk_total); }
        const int nsub = k/32;
        const int blocks = (m + NW*4 - 1)/(NW*4);
        for (int t = 0; t < 2; ++t) {
            gemv_iq3xxs_rows_rep<1, 4><<<blocks, NTHREADS, 0, stream>>>(W, row_bytes, xh + (int64_t) t*k, xs + t*nsub, y + (int64_t) t*m, m, k);
        }
        return cudaGetLastError();
    }
    const int nblk_total = T*(k/32);
    if (convert) { convert_x<<<(nblk_total + 7)/8, 256, 0, stream>>>(x, xh, xs, nblk_total); }
    const int R = T <= 2 ? 4 : 2;
    const int blocks = (m + NW*R - 1)/(NW*R);
    switch (T) {
        case 1: gemv_iq3xxs_rows_rep<1, 4><<<blocks, NTHREADS, 0, stream>>>(W, row_bytes, xh, xs, y, m, k); break;
        case 2: gemv_iq3xxs_rows_rep<2, 4><<<blocks, NTHREADS, 0, stream>>>(W, row_bytes, xh, xs, y, m, k); break;
        case 3: gemv_iq3xxs_rows_rep<3, 2><<<blocks, NTHREADS, 0, stream>>>(W, row_bytes, xh, xs, y, m, k); break;
        case 4: gemv_iq3xxs_rows_rep<4, 2><<<blocks, NTHREADS, 0, stream>>>(W, row_bytes, xh, xs, y, m, k); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

// IQ3_S repack, same total size (110 B/block, so no extra bytes): 8 records of [qs[8]][signs[4]] at 12*ib
// (4-aligned because 110*nb is 4-divisible for every target k), then qh[8] at 96, scales[4] at 104, d at 108.
// The lane reads 3 uint32 (qs+signs) + qh + scale + d instead of ~15 byte loads.
__global__ void repack_iq3s(const uint8_t * __restrict__ src, uint8_t * __restrict__ dst, int nblocks) {
    const int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b >= nblocks) {
        return;
    }
    const uint8_t * s = src + (int64_t) b * SB_BYTES_S;
    uint8_t       * d = dst + (int64_t) b * SB_BYTES_S_REP;
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        const uint16_t * q  = reinterpret_cast<const uint16_t *>(s + 2 + 8*ib);
        uint16_t *       rq = reinterpret_cast<uint16_t *>(d + 12*ib);
        rq[0] = __ldg(q + 0);
        rq[1] = __ldg(q + 1);
        rq[2] = __ldg(q + 2);
        rq[3] = __ldg(q + 3);
        const uint16_t * sg = reinterpret_cast<const uint16_t *>(s + 2 + 64 + 8 + 4*ib);
        uint16_t *       rs = reinterpret_cast<uint16_t *>(d + 12*ib + 8);
        rs[0] = __ldg(sg + 0);
        rs[1] = __ldg(sg + 1);
        d[96 + ib] = s[2 + 64 + ib];
    }
    *reinterpret_cast<uint16_t *>(d + 104) = __ldg(reinterpret_cast<const uint16_t *>(s + 106));
    *reinterpret_cast<uint16_t *>(d + 106) = __ldg(reinterpret_cast<const uint16_t *>(s + 108));
    *reinterpret_cast<uint16_t *>(d + 108) = __ldg(reinterpret_cast<const uint16_t *>(s));
}

template <int T, int R>
__global__ void __launch_bounds__(NTHREADS, 4)
gemv_iq3s_rows_rep(const uint8_t * __restrict__ W, const int64_t row_bytes, const half * __restrict__ xh,
                   const float * __restrict__ xs, float * __restrict__ y, const int m, const int k) {
    __shared__ TablesS tab;
    build_tables_s(tab);
    __syncthreads();

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int nsub = k / 32;

    for (int row0 = (blockIdx.x*NW + warp)*R; row0 < m; row0 += gridDim.x*NW*R) {
        float acc[R][T];
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int t = 0; t < T; ++t) {
                acc[r][t] = 0.0f;
            }
        }
        for (int s = lane; s < nsub; s += 32) {
            uint4 xv[T][4];
            float xsc[T];
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(t)*k + 32*s);
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    xv[t][l] = __ldg(xp + l);
                }
                xsc[t] = __ldg(xs + t*nsub + s);
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (row0 + r >= m) {
                    break;
                }
                const uint8_t  * blk = W + int64_t(row0 + r)*row_bytes + (s >> 3)*SB_BYTES_S_REP;
                const int        ib  = s & 7;
                const uint32_t * rec = reinterpret_cast<const uint32_t *>(blk + 12*ib);
                const uint32_t q0 = __ldg(rec + 0);
                const uint32_t q1 = __ldg(rec + 1);
                const uint32_t sg = __ldg(rec + 2);
                const uint32_t qh = __ldg(blk + 96 + ib);
                const uint32_t sc = __ldg(blk + 104 + (ib >> 1));
                const float    d  = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t *>(blk + 108))));
                const float    db = d*(1.0f + 2.0f*float((sc >> (4*(ib & 1))) & 0xf));
                half2 h[T];
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    h[t] = __float2half2_rn(0.0f);
                }
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const uint32_t qw  = l < 2 ? q0 : q1;
                    const uint32_t g1i = ((qw >> (16*(l & 1)))     & 0xFFu) | ((qh << (8 - 2*l)) & 256u);
                    const uint32_t g2i = ((qw >> (16*(l & 1) + 8)) & 0xFFu) | ((qh << (7 - 2*l)) & 256u);
                    const uint32_t sb  = (sg >> (8*l)) & 0xFFu;
                    const uint2 g0 = tab.grid[g1i];
                    const uint2 g1 = tab.grid[g2i];
                    const uint4 sm = tab.sign[sb];
                    const half2 w0 = u2h(g0.x ^ sm.x);
                    const half2 w1 = u2h(g0.y ^ sm.y);
                    const half2 w2 = u2h(g1.x ^ sm.z);
                    const half2 w3 = u2h(g1.y ^ sm.w);
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        h[t] = __hfma2(w0, u2h(xv[t][l].x), h[t]);
                        h[t] = __hfma2(w1, u2h(xv[t][l].y), h[t]);
                        h[t] = __hfma2(w2, u2h(xv[t][l].z), h[t]);
                        h[t] = __hfma2(w3, u2h(xv[t][l].w), h[t]);
                    }
                }
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    const float2 f = __half22float2(h[t]);
                    acc[r][t] += (f.x + f.y)*db*xsc[t];
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int t = 0; t < T; ++t) {
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    acc[r][t] += __shfl_xor_sync(0xFFFFFFFF, acc[r][t], off);
                }
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (row0 + r < m) {
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        y[int64_t(t)*m + row0 + r] = acc[r][t];
                    }
                }
            }
        }
    }
}

inline cudaError_t launch_iq3s_rep(const uint8_t * W, int64_t row_bytes, const float * x, half * xh, float * xs, float * y,
                                   int m, int k, int T, cudaStream_t stream, bool convert = true) {
    const int nblk_total = T*(k/32);
    if (convert) { convert_x<<<(nblk_total + 7)/8, 256, 0, stream>>>(x, xh, xs, nblk_total); }
    const int R = T == 1 ? 4 : (T == 2 ? 8 : 2);
    const int blocks = (m + NW*R - 1)/(NW*R);
    switch (T) {
        case 1: gemv_iq3s_rows_rep<1, 4><<<blocks, NTHREADS, 0, stream>>>(W, row_bytes, xh, xs, y, m, k); break;
        case 2: gemv_iq3s_rows_rep<2, 8><<<blocks, NTHREADS, 0, stream>>>(W, row_bytes, xh, xs, y, m, k); break;
        case 3: gemv_iq3s_rows_rep<3, 2><<<blocks, NTHREADS, 0, stream>>>(W, row_bytes, xh, xs, y, m, k); break;
        case 4: gemv_iq3s_rows_rep<4, 2><<<blocks, NTHREADS, 0, stream>>>(W, row_bytes, xh, xs, y, m, k); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

}  // namespace sm60mmv