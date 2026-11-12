// P100 (sm_60) fp32 GEMM for the batched PN MUL_MAT at N = 512 (the m = 5120 and m = 12288 shapes: ffn_down, ssm_out,
// attn_output, attn_q).
//
// cuBLAS's sgemm at these shapes accumulates each output in one fp32 register, FMA by FMA, k ascending from 0
// (no split-K). This kernel does the same chain per output, so its result is bit-identical to cuBLAS's. Its
// 64x128 tile gives 320 CTAs over the card's slots (768 at m = 12288), which fills the waves better than cuBLAS's
// 128x128 tiles. Only the shapes in ggml_cuda_pn_gemm_sm60_applies are routed here; every other shape stays on cuBLAS.

#include "pn-gemm-sm60.cuh"
#include "common.cuh"
#include "unary.cuh"

#include <cublasLt.h>

#include <type_traits>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define PN_GEMM_BM  64
#define PN_GEMM_BN  128
#define PN_GEMM_BK  16
#define PN_GEMM_TM  4
#define PN_GEMM_TN  8
#define PN_GEMM_NT  ((PN_GEMM_BM / PN_GEMM_TM) * (PN_GEMM_BN / PN_GEMM_TN))  // 256 threads
#define PN_GEMM_NA  (PN_GEMM_BM * PN_GEMM_BK / 4 / PN_GEMM_NT)                // float4 per thread, A tile
#define PN_GEMM_NB  (PN_GEMM_BN * PN_GEMM_BK / 4 / PN_GEMM_NT)                // float4 per thread, B tile
#define PN_GEMM_LDA (PN_GEMM_BM + 4)  // padded smem row stride: keeps float4 reads aligned, spreads the transposed stores
#define PN_GEMM_LDB (PN_GEMM_BN + 4)

// C[j*m + r] = sum_i A[r*k + i] * B[j*k + i] with one accumulator per output and k ascending.
// Tiles are staged through shared memory: two buffers, and two register sets, so the global loads run two k-tiles
// ahead of the math. Tile t is read from buffer t & 1 and was loaded into register set t & 1, so each k-tile needs
// one __syncthreads. The grid is row-block major in blockIdx.y: the column-block CTAs of one row block (one A tile)
// launch together and share it through L2.
// The staging is issued inside the math, one piece per k-step: a global load of the tile after next and a shared
// store of the next tile, beside the FFMAs, not in a burst before the barrier. The FFMA order of each output is
// unchanged, so the bits are the same.
// PARTIAL: n is not a multiple of PN_GEMM_BN (the last ubatch of a prompt). Columns past n load zeros and store nothing;
// each valid output's chain is unchanged, so the bits are the same as the full-width instantiation's.
template <bool PARTIAL>
static __global__ void __launch_bounds__(PN_GEMM_NT, 2) pn_gemm_sm60_kernel(
        const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int m, int n, int k) {
    constexpr int BK = PN_GEMM_BK;
    constexpr int TX = PN_GEMM_BM / PN_GEMM_TM;
    constexpr int NLD = PN_GEMM_NA + PN_GEMM_NB;  // global loads per tile and thread (float4 each)
    constexpr int NST = 4 * NLD;                   // shared stores per tile and thread (one float each)

    __shared__ __align__(16) float As[2][BK][PN_GEMM_LDA];
    __shared__ __align__(16) float Bs[2][BK][PN_GEMM_LDB];

    const int tid = threadIdx.x;
    const int tx  = tid % TX;
    const int ty  = tid / TX;
    const int r0  = blockIdx.y * PN_GEMM_BM;
    const int j0  = blockIdx.x * PN_GEMM_BN;

    float4 ra0[PN_GEMM_NA], rb0[PN_GEMM_NB], ra1[PN_GEMM_NA], rb1[PN_GEMM_NB];

    // Piece p of tile k0: the A float4s first, then the B float4s.
    auto load_piece = [&](float4 (&ra)[PN_GEMM_NA], float4 (&rb)[PN_GEMM_NB], int k0, int p) {
        if (p < PN_GEMM_NA) {
            const int idx = tid + PN_GEMM_NT * p;
            const int row = idx / (BK / 4), kq = idx % (BK / 4);
            ra[p] = *reinterpret_cast<const float4 *>(A + (size_t) (r0 + row) * k + k0 + kq * 4);
        } else {
            const int q = p - PN_GEMM_NA;
            const int idx = tid + PN_GEMM_NT * q;
            const int row = idx / (BK / 4), kq = idx % (BK / 4);
            // A padding row loads row j0 (a real row: j0 < n) and selects zero, so the load issues like the full-width one.
            const bool live = !PARTIAL || j0 + row < n;
            const float4 v = *reinterpret_cast<const float4 *>(B + (size_t) (live ? j0 + row : j0) * k + k0 + kq * 4);
            rb[q] = live ? v : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };
    // Scalar s of the staged registers (four per float4, A first) into buffer buf, transposed to k-major rows.
    auto store_piece = [&](const float4 (&ra)[PN_GEMM_NA], const float4 (&rb)[PN_GEMM_NB], int buf, int s) {
        if (s < 4 * PN_GEMM_NA) {
            const int q = s / 4, c = s % 4;
            const int idx = tid + PN_GEMM_NT * q;
            const int row = idx / (BK / 4), kq = idx % (BK / 4);
            As[buf][kq * 4 + c][row] = c == 0 ? ra[q].x : c == 1 ? ra[q].y : c == 2 ? ra[q].z : ra[q].w;
        } else {
            const int e = s - 4 * PN_GEMM_NA;
            const int q = e / 4, c = e % 4;
            const int idx = tid + PN_GEMM_NT * q;
            const int row = idx / (BK / 4), kq = idx % (BK / 4);
            Bs[buf][kq * 4 + c][row] = c == 0 ? rb[q].x : c == 1 ? rb[q].y : c == 2 ? rb[q].z : rb[q].w;
        }
    };

    float acc[PN_GEMM_TM][PN_GEMM_TN];
#pragma unroll
    for (int i = 0; i < PN_GEMM_TM; ++i)
#pragma unroll
        for (int j = 0; j < PN_GEMM_TN; ++j) acc[i][j] = 0.0f;

    // Math of the tile in buffer cur. Beside its k-steps: the stores of the staged registers (sa, sb) into buffer nbuf
    // when st, and the loads of tile k0 into registers (la, lb) when ld.
    auto compute_tile = [&](int cur, bool st, const float4 (&sa)[PN_GEMM_NA], const float4 (&sb)[PN_GEMM_NB], int nbuf,
                            bool ld, float4 (&la)[PN_GEMM_NA], float4 (&lb)[PN_GEMM_NB], int k0) {
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[PN_GEMM_TM], b[PN_GEMM_TN];
            {
                const float4 v = *reinterpret_cast<const float4 *>(&As[cur][kk][tx * PN_GEMM_TM]);
                a[0] = v.x; a[1] = v.y; a[2] = v.z; a[3] = v.w;
            }
#pragma unroll
            for (int j = 0; j < PN_GEMM_TN; j += 4) {
                const float4 v = *reinterpret_cast<const float4 *>(&Bs[cur][kk][ty * PN_GEMM_TN + j]);
                b[j] = v.x; b[j + 1] = v.y; b[j + 2] = v.z; b[j + 3] = v.w;
            }
#pragma unroll
            for (int i = 0; i < PN_GEMM_TM; ++i)
#pragma unroll
                for (int j = 0; j < PN_GEMM_TN; ++j) acc[i][j] = __fmaf_rn(a[i], b[j], acc[i][j]);
            if (st && kk < NST) {
                store_piece(sa, sb, nbuf, kk);
            }
            if (ld && kk < NLD) {
                load_piece(la, lb, k0, kk);
            }
        }
    };

    // Unrolled by two so each register set has a fixed name: even tiles use buffer and set 0, odd tiles 1.
    const int T = k / BK;
#pragma unroll
    for (int p = 0; p < NLD; ++p) {
        load_piece(ra0, rb0, 0, p);
    }
#pragma unroll
    for (int s = 0; s < NST; ++s) {
        store_piece(ra0, rb0, 0, s);
    }
    if (T > 1) {
#pragma unroll
        for (int p = 0; p < NLD; ++p) {
            load_piece(ra1, rb1, BK, p);
        }
    }
    __syncthreads();
    for (int t = 0; t < T; t += 2) {
        // tile t (buffer 0): load tile t+2 into set 0, math, store tile t+1 (set 1) into buffer 1
        compute_tile(0, t + 1 < T, ra1, rb1, 1, t + 2 < T, ra0, rb0, (t + 2) * BK);
        __syncthreads();
        if (t + 1 >= T) {
            break;
        }
        // tile t+1 (buffer 1): load tile t+3 into set 1, math, store tile t+2 (set 0) into buffer 0
        compute_tile(1, t + 2 < T, ra0, rb0, 0, t + 3 < T, ra1, rb1, (t + 3) * BK);
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < PN_GEMM_TM; ++i) {
        const int r = r0 + tx * PN_GEMM_TM + i;
#pragma unroll
        for (int j = 0; j < PN_GEMM_TN; ++j) {
            const int col = j0 + ty * PN_GEMM_TN + j;
            if (!PARTIAL || col < n) {
                C[(size_t) col * m + r] = acc[i][j];
            }
        }
    }
}

bool ggml_cuda_pn_gemm_sm60_applies(ggml_type src0_type, int64_t m, int64_t n, int64_t k, int64_t s01, int64_t s11, int64_t ldc) {
    // m = 5120 (ffn_down, ssm_out, attn_output) and m = 12288 (attn_q): this kernel beats cuBLAS there. m = 6144 (full
    // ubatches only, n = 512, see n_ok) also takes it. The other PN shapes stay on cuBLAS: m = 10240 is level with it,
    // m = 17408 (ffn_gate/ffn_up) is about 3% slower.
    const bool shape_ok = m == 5120 || m == 12288 || m == 6144;
    // Every multiple of 512 (a full ubatch, or two of them coalesced into one call): each column's bits are those of its
    // n = 512 call. The partial last ubatch (n = 390..511) only at m = 5120: the cuBLASLt route serves m = 12288 partial batches faster.
    const bool n_ok = (n > 0 && n % 512 == 0) || (m == 5120 && n >= 390 && n < 512);
    return (src0_type == GGML_TYPE_PN3T || src0_type == GGML_TYPE_PN4T) &&
           shape_ok && n_ok && k % PN_GEMM_BK == 0 &&
           s01 == k && s11 == k && ldc == m;
}

void ggml_cuda_pn_gemm_sm60(cudaStream_t stream, const float * A, const float * B, float * C, int64_t m, int64_t n, int64_t k) {
    const dim3 grid((unsigned) ((n + PN_GEMM_BN - 1) / PN_GEMM_BN), (unsigned) (m / PN_GEMM_BM));
    if (n % PN_GEMM_BN == 0) {
        pn_gemm_sm60_kernel<false><<<grid, PN_GEMM_NT, 0, stream>>>(A, B, C, (int) m, (int) n, (int) k);
    } else {
        pn_gemm_sm60_kernel<true><<<grid, PN_GEMM_NT, 0, stream>>>(A, B, C, (int) m, (int) n, (int) k);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Half2 accumulation over each 64-k window (two 32-deep half2 chains, joined by one HADD2), flushed to fp32 once per
// window. Operands are fp16 (A and B converted when staged), the 64x128 tile and 8x4 thread micro-tile as above.
// Each thread keeps 4 x 4 half2 accumulators (rows i, column pairs j2); a pair holds columns 2*j2 and 2*j2+1.
// Staging stays 32 k a tile: a window is two tiles, chain 0 takes the even tile's 32 k, chain 1 the odd tile's 32 k.
// Two tiles share the kernel: the wide 64x128 tile, and the narrow 128x64 tile for a batch whose padded width shrinks
// (pn_h2_narrow). Both keep each output's k order (the chains split by tile parity), so the output bytes do not change.
#define PN_H2_WIDE_BM   64
#define PN_H2_WIDE_BN   128
#define PN_H2_NARROW_BM 128
#define PN_H2_NARROW_BN 64
#define PN_H2_BK  32
#define PN_H2_TM  4
#define PN_H2_TN  8
#define PN_H2_NT  256

// The narrow tile has ceil(m/128) * ceil(n/64) CTAs against the wide tile's ceil(m/64) * ceil(n/128) = 2 * ceil(m/128) *
// ceil(n/128) (m a multiple of 128). It has fewer only when ceil(n/64) < 2 * ceil(n/128): n = 39 halves the count
// (1 < 2), n = 100 keeps it (2 = 2). A multiple of 128 is never narrow. GGML_PN_H2_WIDE=1 is a dev switch for the wide tile.
static bool pn_h2_narrow(int64_t m, int64_t n) {
    static const bool wide_only = getenv("GGML_PN_H2_WIDE") != nullptr;
    if (wide_only || m % PN_H2_NARROW_BM != 0) {
        return false;
    }
    return (n + PN_H2_NARROW_BN - 1) / PN_H2_NARROW_BN < 2 * ((n + PN_H2_WIDE_BN - 1) / PN_H2_WIDE_BN);
}

static int64_t pn_h2_bcols(int64_t m, int64_t n) {
    const int64_t bn = pn_h2_narrow(m, n) ? PN_H2_NARROW_BN : PN_H2_WIDE_BN;
    return (n + bn - 1) / bn * bn;
}

static const bool pn_h2_b_stats = getenv("GGML_CUDA_PN_PREFETCH_STATS") != nullptr;
static uint64_t pn_h2_b_conversions = 0;

// EPI 0: C = the GEMM. EPI 1: C = silu(E) * acc (the register result is the SWIGLU's up). EPI 2: C = silu(acc) * E (the
// register result is its gate). E is the other GEMM's fp32 result in C's layout; the expression is the one unary_gated_op_kernel
// evaluates for SWIGLU, so the values are the GEMM's and the epilogue's own, rounded the same way.
// SEG 0: the half2 chains of a 32-k tile, as above. SEG 8 or 16 (the tier switch, pn_h2_tier_seg): each chain restarts every
// SEG k and its sum is added to the fp32 accumulator at each segment's end, in k order (OUTER_H2 is false then).
template <bool PARTIAL, bool A_HALF, int BM, int BN, int EPI, bool OUTER_H2, int SEG>
static __global__ void __launch_bounds__(PN_H2_NT, 2) pn_h2_sm60_kernel(
        const void * __restrict__ Av, const __half * __restrict__ Bh, float * __restrict__ C, int m, int n, int k, int ldb,
        const float * __restrict__ E, bool vec) {
    constexpr int BK  = PN_H2_BK;
    constexpr int CHAIN = SEG != 0 ? SEG : BK;   // k steps per half2 chain
    constexpr int LDA = BM + 2;                   // A row stride in half2 (a k pair per entry): odd k pairs sit 2 rows on (no bank conflicts)
    constexpr int LDB = BN + 8;
    constexpr int TX  = BM / PN_H2_TM;
    constexpr int BCG = BN / 8;                   // 16-byte column groups of a B row
    constexpr int BRS = PN_H2_NT / BCG;           // B rows per pass: a thread stages BK / BRS rows of B per tile
    constexpr int KQ  = BK / 4;                   // float4 (or uint2) slots per tile row
    constexpr int QA  = BM * KQ / PN_H2_NT;       // A slots per thread and tile
    constexpr int QB  = BK / BRS;                 // B slots per thread and tile
    constexpr int RS  = PN_H2_NT / KQ;            // rows between one thread's slots: slot q is row (row0 + q * RS)

    // A is fp32 weights (converted to half when staged) or, with A_HALF, half weights already (staged as they are)
    using a_slot_t = typename std::conditional<A_HALF, uint2, float4>::type;
    const float  * A  = reinterpret_cast<const float  *>(Av);
    const __half * Ah = reinterpret_cast<const __half *>(Av);

    // A is staged as k pairs: Ap[kp][row] holds k = 2 kp and 2 kp + 1 of a row, so one 16-byte load gives the four rows
    // of a thread's micro-tile for two k-steps.
    __shared__ __align__(16) __half2 Ap[2][BK / 2][LDA];
    __shared__ __align__(16) __half  Bs[2][BK][LDB];

    const int tid  = threadIdx.x;
    const int tx   = tid % TX;
    const int ty   = tid / TX;
    const int r0   = blockIdx.y * BM;
    const int j0   = blockIdx.x * BN;
    const int kq   = tid % KQ;
    const int row0 = tid / KQ;
    const int brow = tid / BCG;
    const int bcol = (tid % BCG) * 8;

    // Element offsets of this thread's slots, computed once: a tile adds only its k0.
    const size_t offA  = (size_t) (r0 + row0) * k + kq * 4;
    const size_t stepR = (size_t) RS * k;
    const size_t offB  = (size_t) brow * ldb + j0 + bcol;
    const size_t stepB = (size_t) BRS * ldb;

    // One register set: a tile's global loads are issued one tile ahead of their shared stores, so each load has the
    // math of a whole 32-k tile to hide behind (the same k distance the two 16-k register sets gave).
    a_slot_t ra[QA];
    uint4 rb[QB];

    auto load_tile = [&](int k0) {
#pragma unroll
        for (int q = 0; q < QA; ++q) {
            if constexpr (A_HALF) {
                ra[q] = *reinterpret_cast<const uint2 *>(Ah + offA + q * stepR + k0);
            } else {
                ra[q] = *reinterpret_cast<const float4 *>(A + offA + q * stepR + k0);
            }
        }
#pragma unroll
        for (int q = 0; q < QB; ++q) {
            rb[q] = *reinterpret_cast<const uint4 *>(Bh + offB + q * stepB + (size_t) k0 * ldb);
        }
    };
    auto store_tile = [&](int buf) {
#pragma unroll
        for (int q = 0; q < QA; ++q) {
            const int row = row0 + q * RS;
            __half2 k01, k23;  // the slot's k = 4 kq, 4 kq + 1 and 4 kq + 2, 4 kq + 3 pairs
            if constexpr (A_HALF) {
                k01 = *reinterpret_cast<const __half2 *>(&ra[q].x);
                k23 = *reinterpret_cast<const __half2 *>(&ra[q].y);
            } else {
                k01 = __floats2half2_rn(ra[q].x, ra[q].y);
                k23 = __floats2half2_rn(ra[q].z, ra[q].w);
            }
            Ap[buf][kq * 2 + 0][row] = k01;
            Ap[buf][kq * 2 + 1][row + 2] = k23;
        }
#pragma unroll
        for (int q = 0; q < QB; ++q) {
            *reinterpret_cast<uint4 *>(&Bs[buf][brow + BRS * q][bcol]) = rb[q];
        }
    };

    // Cross-window sum: fp32 (OUTER_H2 false), or fp16 with one HADD2 per half2 per window (OUTER_H2 true, the k <= 5120
    // projections). Only the one this instance uses is live; the other is dead code the compiler drops.
    float   acc[PN_H2_TM][PN_H2_TN];
    __half2 acch[PN_H2_TM][PN_H2_TN / 2];
#pragma unroll
    for (int i = 0; i < PN_H2_TM; ++i) {
#pragma unroll
        for (int j = 0; j < PN_H2_TN; ++j) acc[i][j] = 0.0f;
#pragma unroll
        for (int j = 0; j < PN_H2_TN / 2; ++j) acch[i][j] = __floats2half2_rn(0.0f, 0.0f);
    }
    // Two half2 chains per window, one per tile: chain C sums the 32 k of tile 2w + C. A chain's first product starts
    // it (__hmul2 rounds as fma with a zero addend does), so the chains need no reset after a flush.
    __half2 acc16[2][PN_H2_TM][PN_H2_TN / 2];

    // One staged tile (32 k) from buffer buf into chain C, a compile-time index: the tile's parity.
    auto compute_tile = [&](auto chain, int buf) {
        constexpr int C = decltype(chain)::value;
#pragma unroll
        for (int kp = 0; kp < BK / 2; ++kp) {
            // Four rows of k = 2 kp, 2 kp + 1 in one load; the k-step picks the low or the high half of each pair.
            const uint4 av = *reinterpret_cast<const uint4 *>(&Ap[buf][kp][tx * PN_H2_TM + 2 * (kp & 1)]);
            const __half2 * ap = reinterpret_cast<const __half2 *>(&av);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int kk = 2 * kp + h;
                __half2 a2[PN_H2_TM];
#pragma unroll
                for (int i = 0; i < PN_H2_TM; ++i) {
                    a2[i] = h == 0 ? __low2half2(ap[i]) : __high2half2(ap[i]);
                }
                const __half2 * bp = reinterpret_cast<const __half2 *>(&Bs[buf][kk][ty * PN_H2_TN]);
#pragma unroll
                for (int j = 0; j < PN_H2_TN / 2; ++j) {
                    const __half2 b2 = bp[j];
#pragma unroll
                    for (int i = 0; i < PN_H2_TM; ++i) {
                        acc16[C][i][j] = (kk % CHAIN == 0) ? __hmul2(a2[i], b2) : __hfma2(a2[i], b2, acc16[C][i][j]);
                    }
                }
                if constexpr (SEG != 0) {
                    if (kk % SEG == SEG - 1) {
                        // the segment's half2 sum joins the fp32 accumulator, lane by lane
#pragma unroll
                        for (int i = 0; i < PN_H2_TM; ++i) {
#pragma unroll
                            for (int j = 0; j < PN_H2_TN / 2; ++j) {
                                acc[i][2 * j]     += __low2float(acc16[C][i][j]);
                                acc[i][2 * j + 1] += __high2float(acc16[C][i][j]);
                            }
                        }
                    }
                }
            }
        }
    };

    // Once per 64-k window: add both chains to the cross-window sum. In fp32 the join is in fp32 (the seed's); in fp16
    // (OUTER_H2) both chains join in fp16 first, and that sum is added to the half2 accumulator.
    auto flush_window = [&]() {
        if constexpr (SEG != 0) {
            return;  // the tier's chains joined at their segment ends (compute_tile)
        }
#pragma unroll
        for (int i = 0; i < PN_H2_TM; ++i) {
#pragma unroll
            for (int j = 0; j < PN_H2_TN / 2; ++j) {
                if constexpr (OUTER_H2) {
                    acch[i][j] = __hadd2(acch[i][j], __hadd2(acc16[0][i][j], acc16[1][i][j]));
                } else {
                    const __half2 s0 = acc16[0][i][j];
                    const __half2 s1 = acc16[1][i][j];
                    acc[i][2 * j]     += __low2float(s0) + __low2float(s1);
                    acc[i][2 * j + 1] += __high2float(s0) + __high2float(s1);
                }
            }
        }
    };

    // Window w is tiles 2w (buffer 0, chain 0) and 2w + 1 (buffer 1, chain 1). Each tile's step issues the loads of the
    // next tile, computes its own, stores the next tile into the other buffer (its last reader passed the previous
    // barrier), then one barrier. k is a multiple of 64, so the windows are whole.
    const int W = k / (2 * BK);
    load_tile(0);
    store_tile(0);
    __syncthreads();
    for (int w = 0; w < W; ++w) {
        load_tile((2 * w + 1) * BK);
        compute_tile(std::integral_constant<int, 0>(), 0);
        store_tile(1);
        __syncthreads();

        if (w + 1 < W) {
            load_tile((2 * w + 2) * BK);
        }
        compute_tile(std::integral_constant<int, 1>(), 1);
        flush_window();
        if (w + 1 < W) {
            store_tile(0);
        }
        __syncthreads();
    }

    // A column's PN_H2_TM rows are consecutive in C (and in E), so one float4 store (and one float4 load of E) per column
    // when vec: the host sets vec when m and both pointers are 16-byte aligned. The values are the scalar path's.
#pragma unroll
    for (int j = 0; j < PN_H2_TN; ++j) {
        const int col = j0 + ty * PN_H2_TN + j;
        if (!PARTIAL || col < n) {
            float v[PN_H2_TM];
#pragma unroll
            for (int i = 0; i < PN_H2_TM; ++i) {
                if constexpr (OUTER_H2) {
                    const __half2 o = acch[i][j / 2];
                    v[i] = (j & 1) ? __high2float(o) : __low2float(o);
                } else {
                    v[i] = acc[i][j];
                }
            }
            const size_t idx = (size_t) col * m + r0 + tx * PN_H2_TM;
            float e[PN_H2_TM] = {};
            if constexpr (EPI != 0) {
                if (vec) {
                    const float4 e4 = *reinterpret_cast<const float4 *>(E + idx);
                    e[0] = e4.x;
                    e[1] = e4.y;
                    e[2] = e4.z;
                    e[3] = e4.w;
                } else {
#pragma unroll
                    for (int i = 0; i < PN_H2_TM; ++i) {
                        e[i] = E[idx + i];
                    }
                }
            }
            float o[PN_H2_TM];
#pragma unroll
            for (int i = 0; i < PN_H2_TM; ++i) {
                if constexpr (EPI == 0) {
                    o[i] = v[i];
                } else if constexpr (EPI == 1) {
                    o[i] = ggml_cuda_op_silu_single(e[i]) * v[i];
                } else {
                    o[i] = ggml_cuda_op_silu_single(v[i]) * e[i];
                }
            }
            if (vec) {
                *reinterpret_cast<float4 *>(C + idx) = make_float4(o[0], o[1], o[2], o[3]);
            } else {
#pragma unroll
                for (int i = 0; i < PN_H2_TM; ++i) {
                    C[idx + i] = o[i];
                }
            }
        }
    }
}

// EPI runs on both tiles: the narrow and the wide tile take the same epilogue (pn_h2_sm60_kernel's EPI).
template <bool A_HALF, int BM, int BN, int EPI, bool OUTER_H2, int SEG>
static void pn_h2_sm60_launch_tile(cudaStream_t stream, const void * A, const __half * B, float * C, int64_t m, int64_t n, int64_t k,
                                   const float * E) {
    const dim3 grid((unsigned) ((n + BN - 1) / BN), (unsigned) (m / BM));
    const int ldb = (int) ((n + BN - 1) / BN * BN);
    const bool vec = m % 4 == 0 && (((uintptr_t) C | (uintptr_t) E) & 15) == 0;
    if (n % BN == 0) {
        pn_h2_sm60_kernel<false, A_HALF, BM, BN, EPI, OUTER_H2, SEG><<<grid, PN_H2_NT, 0, stream>>>(A, B, C, (int) m, (int) n, (int) k, ldb, E, vec);
    } else {
        pn_h2_sm60_kernel<true, A_HALF, BM, BN, EPI, OUTER_H2, SEG><<<grid, PN_H2_NT, 0, stream>>>(A, B, C, (int) m, (int) n, (int) k, ldb, E, vec);
    }
}

// The cross-window sum's precision follows the window count (k / 64). Every window adds its rounding once per output, so
// the error grows with the count: fp16 up to 80 windows (the k = 5120 projections), fp32 above (ffn_down, k = 17408, 272).
#define PN_H2_OUTER_H2_MAX_K 5120

template <bool A_HALF, int EPI, int SEG>
static void pn_h2_sm60_launch_seg(cudaStream_t stream, const void * A, const __half * B, float * C, int64_t m, int64_t n, int64_t k,
                                  const float * E) {
    if constexpr (SEG != 0) {
        // the tier's chains always join in fp32
        if (pn_h2_narrow(m, n)) {
            pn_h2_sm60_launch_tile<A_HALF, PN_H2_NARROW_BM, PN_H2_NARROW_BN, EPI, false, SEG>(stream, A, B, C, m, n, k, E);
        } else {
            pn_h2_sm60_launch_tile<A_HALF, PN_H2_WIDE_BM, PN_H2_WIDE_BN, EPI, false, SEG>(stream, A, B, C, m, n, k, E);
        }
    } else {
        const bool outer_h2 = k <= PN_H2_OUTER_H2_MAX_K;
        if (pn_h2_narrow(m, n)) {
            if (outer_h2) {
                pn_h2_sm60_launch_tile<A_HALF, PN_H2_NARROW_BM, PN_H2_NARROW_BN, EPI, true, 0>(stream, A, B, C, m, n, k, E);
            } else {
                pn_h2_sm60_launch_tile<A_HALF, PN_H2_NARROW_BM, PN_H2_NARROW_BN, EPI, false, 0>(stream, A, B, C, m, n, k, E);
            }
        } else {
            if (outer_h2) {
                pn_h2_sm60_launch_tile<A_HALF, PN_H2_WIDE_BM, PN_H2_WIDE_BN, EPI, true, 0>(stream, A, B, C, m, n, k, E);
            } else {
                pn_h2_sm60_launch_tile<A_HALF, PN_H2_WIDE_BM, PN_H2_WIDE_BN, EPI, false, 0>(stream, A, B, C, m, n, k, E);
            }
        }
    }
}

// seg is the chain depth of the weight (pn_h2_tier_seg): 0 is the champion's arithmetic.
template <bool A_HALF, int EPI = 0>
static void pn_h2_sm60_launch(cudaStream_t stream, const void * A, const __half * B, float * C, int64_t m, int64_t n, int64_t k,
                              const float * E, int seg) {
    switch (seg) {
        case 8:
            pn_h2_sm60_launch_seg<A_HALF, EPI, 8>(stream, A, B, C, m, n, k, E);
            break;
        case 16:
            pn_h2_sm60_launch_seg<A_HALF, EPI, 16>(stream, A, B, C, m, n, k, E);
            break;
        default:
            pn_h2_sm60_launch_seg<A_HALF, EPI, 0>(stream, A, B, C, m, n, k, E);
            break;
    }
}

int64_t ggml_cuda_pn_h2_b_elems(int64_t m, int64_t n, int64_t k) { return pn_h2_bcols(m, n) * k; }

static __global__ void pn_b_to_fp16_t_kernel(const float * __restrict__ B, half * __restrict__ Bt, int n, int ldb, int k) {
    __shared__ float tile[32][33];
    const int kk0 = blockIdx.x * 32;
    const int j0 = blockIdx.y * 32;
#pragma unroll
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int j = j0 + r;
        tile[r][threadIdx.x] = j < n ? B[(size_t) j * k + kk0 + threadIdx.x] : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int r = threadIdx.y; r < 32; r += 8) {
        Bt[(size_t) (kk0 + r) * ldb + j0 + threadIdx.x] = __float2half(tile[threadIdx.x][r]);
    }
}

void ggml_cuda_pn_b_to_fp16(cudaStream_t stream, const float * B, half * Bt, int64_t m, int64_t n, int64_t k) {
    const int ldb = (int) pn_h2_bcols(m, n);
    if (pn_h2_b_stats) {
        fprintf(stderr, "pn-h2-b conversions=%llu n=%lld k=%lld\n",
                (unsigned long long) ++pn_h2_b_conversions, (long long) n, (long long) k);
    }
    pn_b_to_fp16_t_kernel<<<dim3((unsigned) (k / 32), (unsigned) (ldb / 32)), dim3(32, 8), 0, stream>>>(B, Bt, (int)n, ldb, (int)k);
    CUDA_CHECK(cudaGetLastError());
}

// Depth-gated routing. Batched PN MUL_MAT whose weight belongs to block >= PN_H2_MIN_BLK (31 of PN3M19's 64 blocks: the
// deepest 33 take it, which is as far as the numerics go: mean KL 1.7e-3 on the numerics text at 31, 3.0e-3 at 30 and 28,
// against 2.593e-3) takes this kernel, except attn_k and attn_v, which keep the chain's original fp32 algorithm.
// Only batched MUL_MAT (more than 4 columns, every prefill ubatch) takes it; ncols 1..4 stay on the PN GEMVs.
// GGML_PN_H2_OFF=1 turns the route off (the seed's routing); GGML_PN_H2_MIN_BLK and GGML_PN_H2_KV=1 are dev switches.
static int pn_h2_env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v != nullptr ? atoi(v) : def;
}

// The ubatch the next graph computes, set by llama_context before each graph (ggml_cuda_pn_tier_set_ubatch). The tier
// covers only the ubatches that start at position PN_TIER_POS0 or later and are not the prompt's final ubatch: the sink
// band and the ubatch that makes the output keep the champion route. Before any set, the tier is off.
#define PN_TIER_POS0 512
static bool    pn_tier_set     = false;
static int64_t pn_tier_pos0    = 0;
static bool    pn_tier_is_last = false;

// Dev counter (GGML_PN_TIER_COUNT=1, stderr): the tier's batched MUL_MATs per block in each ubatch, and the sink split's
// (sink=N, the ubatch's batched MUL_MATs that ran their columns 512..1023 on the tier). The next set prints them.
static int pn_tier_fired[256];
static int pn_tier_sink_fired = 0;
static int pn_tier_seg16_fired = 0;

void ggml_cuda_pn_tier_set_ubatch(int64_t pos0, bool is_last) {
    static const bool dev_count = pn_h2_env_int("GGML_PN_TIER_COUNT", 0) != 0;
    if (dev_count && pn_tier_set) {
        fprintf(stderr, "pn-tier: ubatch pos0=%lld last=%d covered=%d sink=%d seg16=%d fired:", (long long) pn_tier_pos0,
                (int) pn_tier_is_last, (int) (pn_tier_pos0 >= PN_TIER_POS0 && !pn_tier_is_last), pn_tier_sink_fired,
                pn_tier_seg16_fired);
        for (int b = 0; b < 256; ++b) {
            if (pn_tier_fired[b] != 0) {
                fprintf(stderr, " blk%d=%d", b, pn_tier_fired[b]);
            }
        }
        fprintf(stderr, "\n");
    }
    for (int & c : pn_tier_fired) {
        c = 0;
    }
    pn_tier_sink_fired = 0;
    pn_tier_seg16_fired = 0;
    pn_tier_set     = true;
    pn_tier_pos0    = pos0;
    pn_tier_is_last = is_last;
}

static bool pn_tier_covers_ubatch() {
    return pn_tier_set && pn_tier_pos0 >= PN_TIER_POS0 && !pn_tier_is_last;
}

// Precision tier: the batched PN MULs of blocks GGML_PN_TIER_LO..GGML_PN_TIER_HI (inclusive, default 27..30) take this
// kernel with GGML_PN_TIER_SEG-deep half2 chains (8 or 16, fp32 join at each segment end), below the MIN_BLK gate too,
// in the ubatches pn_tier_covers_ubatch admits. Block 27 takes it for its non-K/V sites only (attn_k/attn_v are kept by
// ggml_cuda_pn_h2_sm60_applies). GGML_PN_TIER_LO=-1 turns the tier off (the 31-gate routing alone); GGML_PN_TIER_LO=28
// restores the tier of block 28 up. GGML_PN_TIER_FFN_ONLY=1 keeps the tier to the ffn_* weights.
// Returns the chain depth of a weight: 0 is the 32-deep chain.
// The mixed arm: the tier's FFN sites (ffn_gate, ffn_up, ffn_down) take 16-deep chains (GGML_PN_TIER_FFN_SEG, default 16;
// 8 gives them the 8-deep chain too), and the other tier weights keep GGML_PN_TIER_SEG (default 8).
static int pn_tier_chain_depth(const char * name, int seg) {
    static const int ffn_seg = pn_h2_env_int("GGML_PN_TIER_FFN_SEG", 16) == 8 ? 8 : 16;
    if (strstr(name, ".ffn_gate.") != nullptr || strstr(name, ".ffn_up.") != nullptr || strstr(name, ".ffn_down.") != nullptr) {
        return ffn_seg;
    }
    return seg;
}

static int pn_h2_tier_seg(const char * name, int blk) {
    static const int  lo       = pn_h2_env_int("GGML_PN_TIER_LO", 27);
    static const int  hi       = pn_h2_env_int("GGML_PN_TIER_HI", 30);
    static const int  seg      = pn_h2_env_int("GGML_PN_TIER_SEG", 8) == 16 ? 16 : 8;
    static const bool ffn_only = pn_h2_env_int("GGML_PN_TIER_FFN_ONLY", 0) != 0;
    if (name == nullptr || lo < 0 || blk < lo || blk > hi || !pn_tier_covers_ubatch()) {
        return 0;
    }
    if (ffn_only && strstr(name, ".ffn_") == nullptr) {
        return 0;
    }
    return pn_tier_chain_depth(name, seg);
}

int ggml_cuda_pn_h2_sm60_seg(const char * name) {
    int blk = -1;
    if (name == nullptr || sscanf(name, "blk.%d.", &blk) != 1) {
        return 0;
    }
    const int seg = pn_h2_tier_seg(name, blk);
    if (seg != 0 && blk >= 0 && blk < 256) {
        pn_tier_fired[blk]++;
    }
    if (seg == 16) {
        pn_tier_seg16_fired++;
    }
    return seg;
}

// Sink split (part 1 of the tier): the paired sink ubatch (pos0 0, 1024 columns, the first two 512-column ubatches) runs its
// columns 0..511 on the champion route, and its columns 512..1023 on the tier, for the tier's weights of blocks 28..30 (the
// same set as the tier of the later ubatches; attn_k/attn_v keep fp32 when GGML_PN_H2_KV is 0). The tier's column half has
// its own fp16 B transpose and is launched with the chain depth of its weight (pn_tier_chain_depth). GGML_PN_TIER_SINK=0 turns it off.
// Returns the seg of the tier's column half, or 0; the caller runs the chain route on the first PN_TIER_SINK_COLS columns.
int ggml_cuda_pn_sink_split_seg(ggml_type src0_type, const char * name, int64_t m, int64_t n, int64_t k, int64_t s01,
                                int64_t s11, int64_t ldc) {
    static const bool on       = pn_h2_env_int("GGML_PN_TIER_SINK", 1) != 0 && pn_h2_env_int("GGML_PN_TIER_LO", 27) >= 0;
    static const int  hi       = pn_h2_env_int("GGML_PN_TIER_HI", 30);
    static const int  seg      = pn_h2_env_int("GGML_PN_TIER_SEG", 8) == 16 ? 16 : 8;
    static const bool kv_keep  = pn_h2_env_int("GGML_PN_H2_KV", 0) == 0;
    static const bool ffn_only = pn_h2_env_int("GGML_PN_TIER_FFN_ONLY", 0) != 0;
    int blk = -1;
    if (!on || name == nullptr || !pn_tier_set || pn_tier_pos0 != 0 || pn_tier_is_last || n != 2 * PN_TIER_SINK_COLS ||
        sscanf(name, "blk.%d.", &blk) != 1 || blk < 28 || blk > hi) {
        return 0;
    }
    if (kv_keep && (strstr(name, ".attn_k.") != nullptr || strstr(name, ".attn_v.") != nullptr)) {
        return 0;
    }
    if (ffn_only && strstr(name, ".ffn_") == nullptr) {
        return 0;
    }
    if (!((src0_type == GGML_TYPE_PN3T || src0_type == GGML_TYPE_PN4T) && m % PN_H2_WIDE_BM == 0 && k % (2 * PN_H2_BK) == 0 &&
          s01 == k && s11 == k && ldc == m)) {
        return 0;
    }
    pn_tier_fired[blk]++;
    pn_tier_sink_fired++;
    const int depth = pn_tier_chain_depth(name, seg);
    if (depth == 16) {
        pn_tier_seg16_fired++;
    }
    return depth;
}

bool ggml_cuda_pn_h2_sm60_applies(ggml_type src0_type, const char * name, int64_t m, int64_t n, int64_t k,
                                  int64_t s01, int64_t s11, int64_t ldc) {
    static const bool off      = pn_h2_env_int("GGML_PN_H2_OFF", 0) != 0;
    static const int  min_blk  = pn_h2_env_int("GGML_PN_H2_MIN_BLK", 31);
    static const bool kv_keep  = pn_h2_env_int("GGML_PN_H2_KV", 0) == 0;
    if (off || name == nullptr) {
        return false;
    }
    int blk = -1;
    if (sscanf(name, "blk.%d.", &blk) != 1 || (blk < min_blk && pn_h2_tier_seg(name, blk) == 0)) {
        return false;
    }
    if (kv_keep && (strstr(name, ".attn_k.") != nullptr || strstr(name, ".attn_v.") != nullptr)) {
        return false;
    }
    return (src0_type == GGML_TYPE_PN3T || src0_type == GGML_TYPE_PN4T) &&
           m % PN_H2_WIDE_BM == 0 && n > 4 && k % (2 * PN_H2_BK) == 0 && s01 == k && s11 == k && ldc == m;
}

void ggml_cuda_pn_h2_sm60(cudaStream_t stream, const float * A, const half * B, float * C, int64_t m, int64_t n, int64_t k, int seg) {
    pn_h2_sm60_launch<false>(stream, A, B, C, m, n, k, nullptr, seg);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_pn_h2_sm60_f16(cudaStream_t stream, const half * A, const half * B, float * C, int64_t m, int64_t n, int64_t k, int seg) {
    pn_h2_sm60_launch<true>(stream, A, B, C, m, n, k, nullptr, seg);
    CUDA_CHECK(cudaGetLastError());
}

// The SWIGLU pair epilogue: the GEMM above with its final store replaced (see pn_h2_sm60_kernel's EPI). mine_is_gate says
// whether this GEMM is the SWIGLU's gate (its register result is silu'd) or its up (the other GEMM's result is silu'd).
void ggml_cuda_pn_h2_sm60_glu(cudaStream_t stream, const float * A, const half * B, const float * other, float * C,
                              bool mine_is_gate, int64_t m, int64_t n, int64_t k, int seg) {
    if (mine_is_gate) {
        pn_h2_sm60_launch<false, 2>(stream, A, B, C, m, n, k, other, seg);
    } else {
        pn_h2_sm60_launch<false, 1>(stream, A, B, C, m, n, k, other, seg);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_pn_h2_sm60_glu_f16(cudaStream_t stream, const half * A, const half * B, const float * other, float * C,
                                  bool mine_is_gate, int64_t m, int64_t n, int64_t k, int seg) {
    if (mine_is_gate) {
        pn_h2_sm60_launch<true, 2>(stream, A, B, C, m, n, k, other, seg);
    } else {
        pn_h2_sm60_launch<true, 1>(stream, A, B, C, m, n, k, other, seg);
    }
    CUDA_CHECK(cudaGetLastError());
}

// cuBLASLt route for the other batched PN shapes (m >= 6144, n >= 256). The chain kernel above is checked first in
// ggml-cuda.cu, so m = 5120 and 12288 never reach here. Algorithm 0 with the 128x128 tile, CTA swizzling 1 and split-K 1
// runs the same no-split-K fp32 chain as the cublasSgemm call it replaces. Returns false, with nothing computed, when
// cuBLASLt does not apply or refuses that algorithm without a workspace; the caller then runs cublasSgemm.
bool ggml_cuda_pn_cublaslt_sm60(cudaStream_t stream, ggml_type src0_type, const float * A, const float * B, float * C,
                                int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc) {
    if (!(src0_type == GGML_TYPE_PN3T || src0_type == GGML_TYPE_PN4T) || m < 6144 || n < 256) {
        return false;
    }
    static cublasLtHandle_t lt = nullptr;
    if (lt == nullptr && cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) {
        lt = nullptr;
        return false;
    }

    const cublasOperation_t transa = CUBLAS_OP_T;   // as the cublasSgemm call: A is k x m, transposed
    const cublasOperation_t transb = CUBLAS_OP_N;
    const uint32_t tile    = CUBLASLT_MATMUL_TILE_128x128;
    const uint32_t swizzle = 1;
    const uint32_t splitk  = 1;
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasLtMatmulDesc_t   op = nullptr;
    cublasLtMatrixLayout_t Ad = nullptr;
    cublasLtMatrixLayout_t Bd = nullptr;
    cublasLtMatrixLayout_t Cd = nullptr;
    cublasLtMatmulAlgo_t   algo;
    cublasLtMatmulHeuristicResult_t heur = {};
    bool ok = cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS &&
              cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)) == CUBLAS_STATUS_SUCCESS &&
              cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)) == CUBLAS_STATUS_SUCCESS &&
              cublasLtMatrixLayoutCreate(&Ad, CUDA_R_32F, k, m, lda) == CUBLAS_STATUS_SUCCESS &&
              cublasLtMatrixLayoutCreate(&Bd, CUDA_R_32F, k, n, ldb) == CUBLAS_STATUS_SUCCESS &&
              cublasLtMatrixLayoutCreate(&Cd, CUDA_R_32F, m, n, ldc) == CUBLAS_STATUS_SUCCESS &&
              cublasLtMatmulAlgoInit(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo) == CUBLAS_STATUS_SUCCESS &&
              cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile)) == CUBLAS_STATUS_SUCCESS &&
              cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle)) == CUBLAS_STATUS_SUCCESS &&
              cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk)) == CUBLAS_STATUS_SUCCESS &&
              cublasLtMatmulAlgoCheck(lt, op, Ad, Bd, Cd, Cd, &algo, &heur) == CUBLAS_STATUS_SUCCESS &&
              heur.workspaceSize == 0;
    if (ok) {
        ok = cublasLtMatmul(lt, op, &alpha, A, Ad, B, Bd, &beta, C, Cd, C, Cd, &algo, nullptr, 0, stream) == CUBLAS_STATUS_SUCCESS;
    }

    if (Cd) { cublasLtMatrixLayoutDestroy(Cd); }
    if (Bd) { cublasLtMatrixLayoutDestroy(Bd); }
    if (Ad) { cublasLtMatrixLayoutDestroy(Ad); }
    if (op) { cublasLtMatmulDescDestroy(op); }
    return ok;
}
