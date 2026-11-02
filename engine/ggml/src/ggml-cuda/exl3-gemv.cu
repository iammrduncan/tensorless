// EXL3 format, "mul1" codebook arithmetic, tile layout and trellis GEMV design: Turboderp, exllamav3
// (https://github.com/turboderp-org/exllamav3), MIT License, Copyright (c) 2025 Turboderp; see
// licenses/LICENSE-exllamav3. EXL3 is Turboderp's streamlined variant of QTIP (Tseng, Sun, Hou, De Sa,
// "QTIP: Quantization with Trellises and Incoherence Processing", NeurIPS 2024, arXiv:2406.11235).
// This file is an independent ggml/CUDA reimplementation written with the exllamav3 source as reference;
// no QTIP code was used.
#include "exl3.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define EXL3_GEMV_HAVE_COOP 1
#else
#define EXL3_GEMV_HAVE_COOP 0
#endif

// EXL3 M2/M3: decode-width GEMV/GEMM (T <= EXL3_GEMV_MAX_T activation columns) straight from the trellis stream.
//
// Data format (exllamav3, MIT; see exl3.cu for the codebook): the weight [K, N] is stored as 16x16 tiles in
// k-tile-major order [K/16][N/16][8*bits words]. Inside a tile the 256 weights follow the tensor-core lane order:
// lane t (0..31), slot j (0..7) -> row k = 2*(t&3) + (j&1) + 8*((j>>1)&1), col n = (t>>2) + 8*(j>>2).
// Weight i of the tile is the 16-bit window ending at bit (i+1)*bits of the tail-biting bit stream (MSB-first
// inside 32-bit words), decoded through the "mul1" codebook. That lane order is exactly the B operand layout of
// mma.m16n8k16 (two n8 halves), which the T >= EXL3_GEMV_MMA_MIN_T path uses with x as the A operand.
//
// Kernel: persistent blocks of 4 warps walk work items (n-tile, k-split). A warp walks k-tiles of one n-tile.
// Loads: at <= 4 bits a tile is <= 32 words, so every lane loads one word (one coalesced 128 B request per warp
// per tile) and the two words covering its eight windows come from lane shuffles; at 5..8 bits each lane loads
// its three words directly. The eight codes are decoded (dp4a codebook) into four half2 row pairs and either
// FMAed against x in half2 (T = 1: 4 HFMA2 per tile) or fed as the B fragments of two mma.m16n8k16 with fp16
// accumulation folded to fp32 every U tiles (T >= 2: cost independent of T up to 8). Warps reduce through shared
// memory; k-splits reduce in the glue_out kernel. Summation order is fixed => deterministic.
//
// M4 (FUSED): the same kernel launched cooperatively does the glue in-kernel -- phase 1 writes the fp16 staging
// buffer xh = scale * H128(suh * x) (grid-strided over 128-chunks), grid barrier, phase 2 = the tile loop into the
// split-K partials, grid barrier, phase 3 reduces the partials, applies H128 and svh and writes y. Two ~1 us grid
// barriers replace two glue kernels and two graph nodes per matmul. Measured 2026-09-17: the main GEMV pays ~10.6 us
// per call for the barriers on the 1008-block persistent grid, more than the glue costs, so the split-glue path stays
// the default; GGML_CUDA_EXL3_FUSED=1 opts in, EXL3_GEMV_BPS caps the cooperative grid at n blocks per SM.

#define EXL3_GEMV_NWARPS 4
#define EXL3_GEMV_MAX_T  16
#ifndef EXL3_GEMV_MMA_MIN_T
#define EXL3_GEMV_MMA_MIN_T 2
#endif

#ifdef __CUDA_ARCH_LIST__
static constexpr bool EXL3_GEMV_MMA_COMPILED = ggml_cuda_highest_compiled_arch(9999) >= GGML_CUDA_CC_VOLTA;
#else
static constexpr bool EXL3_GEMV_MMA_COMPILED = true;
#endif

// x is staged in shared memory as fp16 (scaled by 1/16 so that fp16 partial sums cannot overflow: |w| <= 3.5,
// |x/16| <= 4096, at most 64 products per fp16 partial)
#define EXL3_GEMV_X_SCALE    0.0625f
template <int T> struct exl3_gemv_cfg {
    static constexpr bool MMA        = T >= EXL3_GEMV_MMA_MIN_T;
    static constexpr int  XROWS      = MMA ? (T <= 8 ? 8 : 16) : T;  // staged x rows (mma: rows T..XROWS-1 stay zero)
    // k-tiles per item = the per-block x staging buffer (XROWS * KTILES_MAX * 32 bytes). T == 1 gets 7.7 KB and is
    // still register-limited to 8 resident blocks - 8 * 7936 B just fits SM60's 64 KB/SM, so if this ever grows,
    // occupancy silently drops to 7 blocks and the persistent grid shrinks with it. Bigger items amortise the per-item
    // cost (stage x, two barriers, write the split-K partial): 128 -> 192 measured +2.6% (run #23) and 192 -> 240
    // +0.4-0.7% on the clean constituents (a 240-tile item also lands kt = 1088 on 5 splits with every warp ending on
    // a whole group step). 256 is clearly worse: 8.4 KB/block crosses the 8-block line and the replica loses ~2%.
    static constexpr int  KTILES_MAX = XROWS <= 2 ? 240 : (XROWS <= 8 ? 64 : 32);
};

static __device__ __forceinline__ uint32_t exl3_h2u(const half2 v) { return *reinterpret_cast<const uint32_t *>(&v); }
static __device__ __forceinline__ half2    exl3_u2h(const uint32_t v) { return *reinterpret_cast<const half2 *>(&v); }

// d[16x8] += a[16x16] . b[16x8], all fp16 (row-major a, "col" b: the trellis lane order)
static __device__ __forceinline__ void exl3_mma_f16(uint32_t (&d)[2], const uint32_t (&a)[4], const uint32_t b0, const uint32_t b1) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    NO_DEVICE_CODE;
#else
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// At T = 1 the persistent grid is capped by shared memory (4.4 KB/block => 15 blocks/SM on SM60), so ptxas'
// default maximum-occupancy register target is needlessly tight and the decode inner loop schedules poorly.
// Promising EXL3_GEMV_MIN_BLOCKS blocks/SM lets ptxas keep ~51 registers/thread instead. Measured 2026-09-21 on a
// standalone replica of this kernel over the real Qwen3.8-27B shape mix: 2.50 bpw 48.7 -> 47.0 ms/token,
// 4.00 bpw 48.5 -> 46.0 ms/token, single-shape 2-bit probe 0.159 -> 0.145 ms, bit-identical output.  T >= 1 keeps
// the previous (unconstrained) allocation so the mma and cooperative FUSED paths are untouched.
#define EXL3_GEMV_MIN_BLOCKS 12
// Register budget for the T == 1 instantiations: the launch-bounds block target is what sets ptxas' register ceiling
// (8 blocks/SM -> 64 registers/thread, 7 -> ~73, 12 -> ~40). Relaxing from 12 to 8 measured +1.6% on the real
// benchmark (run #14) and occupancy then falls to 8 resident blocks - registers, not shared memory (6 KB/block with
// the 192-tile staging buffer allows 10), are the limit - which the latency-bound decode prefers.
#define EXL3_GEMV_MINB_REGS 8
#define EXL3_GEMV_LAUNCH_BOUNDS(T) __launch_bounds__(32 * EXL3_GEMV_NWARPS, (T) == 1 ? EXL3_GEMV_MINB_REGS : 1)

// k-tiles of weight words in flight per warp (= fp16 fold period) in the decode group loop. Swept on the real
// benchmark at the 64-register T==1 budget: for 2/3-bit 5/6/7/8/10 were measured, 8 best (7 = -2.1%, 10 = -1.7%); for
// 4..8-bit 4 = -1.55%, 6 = best before the stride fix, 8 best after it, 10 = -0.7% (run #27: the replica over-predicts
// high U because it has no KV/attention traffic, while extra k-tiles in flight mean extra outstanding weight loads, and
// the 2048-deep cache case is exactly where that hurts). Also used host-side to align work items to whole group steps.
#define EXL3_GEMV_U 8

template <int BITS, int T, bool FUSED>
static __global__ void EXL3_GEMV_LAUNCH_BOUNDS(T)
k_exl3_gemv(const uint32_t * __restrict__ trellis, half2 * __restrict__ x, float * __restrict__ y,
            const int kt, const int nt, const int K, const int N, const int ksplit, const int kpi,
            const float * __restrict__ xf, const float * __restrict__ suh, const float * __restrict__ svh,
            float * __restrict__ yf, const int post, const long long stepW) {
    using cfg = exl3_gemv_cfg<T>;
    constexpr bool MMA        = cfg::MMA;
    constexpr int  XROWS      = cfg::XROWS;
    constexpr int  KTILES_MAX = cfg::KTILES_MAX;
    constexpr int  NW    = 8 * BITS;       // words per tile
    constexpr int  NBITS = 256 * BITS;     // bits per tile
    // a lane's eight windows span rel0 + 7*BITS + 16 bits with rel0 = ((8*lane+1)*BITS - 16) & 31; max over the
    // lanes: 48 (2b), 64 (3b, 4b), 80 (5b, 6b), 96 (7b, 8b) => 2 words up to 4 bits, 3 words above
    constexpr int  NWORDS = BITS <= 4 ? 2 : 3;
    constexpr bool SHFL   = BITS <= 4;     // one word per lane per tile, windows resolved by shuffles
    // k-tiles of weight words in flight per warp (= fp16 fold period); see EXL3_GEMV_U for the sweep that picked 8.
    constexpr int  U      = EXL3_GEMV_U;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int q    = lane & 3;             // row pair selector
    const int g    = lane >> 2;            // column (and column + 8); mma: A/D row

    // The lane's eight windows start at bit p0 + j*BITS (MSB-first, tail-biting) of the tile stream. Take the
    // NWORDS words covering [ws*32, ...) and shift them left by rel0 = p0 - ws*32 so that window j starts at the
    // compile-time bit offset j*BITS of the shifted stream.
    const int p0   = ((lane * 8 + 1) * BITS - 16 + NBITS) % NBITS;
    const int ws   = p0 >> 5;
    const int rel0 = p0 & 31;
    int woff[NWORDS];                      // word index inside the tile == source lane in the SHFL path
#pragma unroll
    for (int m = 0; m < NWORDS; ++m) {
        woff[m] = (ws + m) % NW;
    }
    const int    own_w  = SHFL ? (lane % NW) : 0;                              // word this lane loads (SHFL)
    // words per k-tile row == stepW / EXL3_GEMV_NWARPS; the stride arrives as a kernel argument on purpose: it is the
    // product nt*NW, and with a 64-bit value derived inside the kernel ptxas re-materialises a full 32x32 multiply by
    // NWARPS*NW*4 for every grouped weight load (SM60 needs 3 XMAD for that) instead of keeping it in a register pair.

    const half2 cb_mul = __halves2half2(__ushort_as_half((unsigned short) 0x1eee), __ushort_as_half((unsigned short) 0x1eee));
    const half2 cb_add = __halves2half2(__ushort_as_half((unsigned short) 0xc931), __ushort_as_half((unsigned short) 0xc931));

    __shared__ float part[EXL3_GEMV_NWARPS][T][16];
    __shared__ __align__(16) half2 xs[XROWS][KTILES_MAX * 8];   // fp16 x of the item's k-range, as row pairs

    if (MMA && T < XROWS) {   // A rows T..7 are zero for every item (the barrier of the first staging orders this)
        half2 * z = &xs[T][0];
        for (int p = threadIdx.x; p < (XROWS - T) * KTILES_MAX * 8; p += 32 * EXL3_GEMV_NWARPS) {
            z[p] = __float2half2_rn(0.0f);
        }
    }

    // eight codes of a tile -> four half2 row pairs: wv[jp] = (slot 2jp, slot 2jp+1) = rows (2q, 2q+1) [jp even]
    // or (2q+8, 2q+9) [jp odd] of column g + 8*(jp>>1)
    auto decode = [&](const uint32_t (&w)[NWORDS], half2 (&wv)[4]) {
        uint32_t a[NWORDS];
#pragma unroll
        for (int m = 0; m < NWORDS; ++m) {
            a[m] = __funnelshift_l(m + 1 < NWORDS ? w[m + 1] : 0u, w[m], rel0);
        }
        // One more window per adjacent word pair, positioned at bit 16 of that pair's stream. Every code whose
        // 16-bit window straddles the word boundary (o & 31 > 16: 2 of 8 at 3 bits, 3 at 4..6 bits) lies wholly
        // inside it at a compile-time offset, so it costs one shift instead of a funnel shift plus a mask, and the
        // pair's second shifted word is no longer needed anywhere else. Bit-identical by construction and verified:
        // 0 mismatches over 41.9B extractions for every bit width and every rel0 (midcheck.cu).
        uint32_t mid[NWORDS > 1 ? NWORDS - 1 : 1];
#pragma unroll
        for (int u = 0; u + 1 < NWORDS; ++u) {
            mid[u] = __funnelshift_r(a[u + 1], a[u], 16);
        }
#pragma unroll
        for (int jp = 0; jp < 4; ++jp) {
            uint32_t sum[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int o  = (2 * jp + e) * BITS;           // compile-time after unrolling
                const int wi = o >> 5;
                const int sh = o & 31;
                const uint32_t hi = a[wi];
                uint32_t code;
                if (sh == 0) {
                    code = hi >> 16;                                  // already zero extended to 16 bits
                } else if (sh <= 16) {
                    code = (hi >> (16 - sh)) & 0xffffu;              // one shift + one free LOP3 mask
                } else {
                    code = (mid[wi] >> (32 - sh)) & 0xffffu;         // straddler, single shift out of mid[]
                }
                // A code word is a 16-bit window of the lane's bit stream, so the straightforward form
                // (__funnelshift_l(lo, hi, sh) >> 16, plus byte_perm for sh == 8) spends two instructions on the
                // ~1 warp-instr/SM/cycle shift pipe per code. Positioning the window at the bottom of the funnel
                // result instead needs one shift pipe op plus a mask on the fast logic pipe, which is bit-identical
                // (verified over ~34M random (rel0, word) tuples for every bit width, extract_check.cu) and cuts
                // ~7 of the ~15 shift-pipe ops per tile: shape-mix harness 42.5 -> 39.7 ms/token at 2.50 bpw and
                // 43.0 -> 39.0 at 4.00 bpw.
#if EXL3_MUL1_BIAS_IS_PACKED
                sum[e] = exl3_mul1_rawsum(code);
#else
                sum[e] = exl3_mul1_sum(code);
#endif
            }
            wv[jp] = __hfma2(exl3_mul1_pair_from_rawsum(sum), cb_mul, cb_add);
        }
    };

#if EXL3_GEMV_HAVE_COOP
    const int gw  = blockIdx.x * EXL3_GEMV_NWARPS + warp;   // grid-wide warp index (fused glue phases)
    const int gws = gridDim.x * EXL3_GEMV_NWARPS;
    if (FUSED) {
        // phase 1: xh[T][K] = X_SCALE * H128(suh * x)   (suh null: X_SCALE * x); one warp per 128-chunk
        half * xh = reinterpret_cast<half *>(x);
        const int n_chunks = T * K / 128;
        for (int c = gw; c < n_chunks; c += gws) {
            const size_t base = (size_t) c * 128;
            const int kc = (int) (base % (size_t) K);
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v[j] = xf[base + j * 32 + lane];
                if (suh != nullptr) {
                    v[j] *= suh[kc + j * 32 + lane];
                }
            }
            float sc = EXL3_GEMV_X_SCALE;
            if (suh != nullptr) {
                exl3_wht128(v);
                sc *= EXL3_WHT128_SCALE;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                xh[base + j * 32 + lane] = __float2half_rn(v[j] * sc);
            }
        }
        cg::this_grid().sync();
    }
#endif

    // persistent: each block walks a contiguous, balanced run of work items (n-tile, k-split) -- every block gets
    // floor(n_items/gridDim.x) items, the first n_items%gridDim.x blocks one more. Runs are contiguous on purpose:
    // a run nearly always lies inside a single k-split, so x only has to be staged into shared memory once per run
    // instead of once per item. That staging plus its barrier cost roughly 4 us per item, 5-7% of the whole GEMV, and
    // the tail balance is unchanged (a run is exactly the same length as a stride pass used to be).
    const int n_items = nt * ksplit;
    const int run_len = n_items / gridDim.x, ragged = n_items % gridDim.x;
    const int item_begin = (int) blockIdx.x * run_len + min((int) blockIdx.x, ragged);
    const int item_end   = item_begin + run_len + ((int) blockIdx.x < ragged ? 1 : 0);
    int staged_ks = -1;
    for (int item = item_begin; item < item_end; ++item) {
        const int ks = item / nt;                                               // 32-bit: item < 2^31
        const int ni = item - ks * nt;
        const int kb = ks * kpi;
        const int nk = min(kpi, kt - kb);                                       // <= KTILES_MAX

        if (ks != staged_ks) {
            // stage x[t][kb*16 .. (kb+nk)*16) (fp16 pairs, pre-scaled by EXL3_GEMV_X_SCALE); the previous item's
            // trailing barrier already ordered every read of the old contents, and the first item of a block always
            // stages, which is what publishes the MMA zero fill below.
            staged_ks = ks;
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const half2 * xt = x + (size_t) t * (K / 2) + (size_t) kb * 8;
                for (int p = threadIdx.x; p < nk * 8; p += 32 * EXL3_GEMV_NWARPS) {
                    xs[t][p] = FUSED ? __ldcg(xt + p) : xt[p];
                }
            }
            __syncthreads();
        }

        const uint32_t * wp = trellis + ((size_t) (kb + warp) * nt + ni) * NW;

        float    acc[2][T];      // T = 1..: fp32 per column slot (HFMA2 path)
        uint32_t d[2][2];        // mma: fp16 D fragments per n8 half
        float2   accf[2];        // mma: fp32 fold of D row g per n8 half
        float2   accg[2];        // mma, XROWS == 16: fp32 fold of D row g + 8 per n8 half
#pragma unroll
        for (int s = 0; s < 2; ++s) {
#pragma unroll
            for (int t = 0; t < T; ++t) {
                acc[s][t] = 0.0f;
            }
            d[s][0] = 0u; d[s][1] = 0u;
            accf[s] = make_float2(0.0f, 0.0f);
            accg[s] = make_float2(0.0f, 0.0f);
        }

        auto tile_fma = [&](const uint32_t (&w)[NWORDS], const int kl) {
            half2 wv[4];
            decode(w, wv);
            if (MMA) {
                uint32_t a[4];
                a[0] = exl3_h2u(xs[g][kl * 8 + q]);          // A[g][2q, 2q+1]
                a[1] = XROWS > 8 ? exl3_h2u(xs[(XROWS > 8 ? g + 8 : 0)][kl * 8 + q]) : 0u;       // A[g+8][2q, 2q+1]
                a[2] = exl3_h2u(xs[g][kl * 8 + q + 4]);      // A[g][2q+8, 2q+9]
                a[3] = XROWS > 8 ? exl3_h2u(xs[(XROWS > 8 ? g + 8 : 0)][kl * 8 + q + 4]) : 0u;   // A[g+8][2q+8, 2q+9]
                exl3_mma_f16(d[0], a, exl3_h2u(wv[0]), exl3_h2u(wv[1]));
                exl3_mma_f16(d[1], a, exl3_h2u(wv[2]), exl3_h2u(wv[3]));
            } else {
                // x row pairs of this tile for this lane: (2q, 2q+1) and (2q+8, 2q+9)
                half2 xa[T], xb[T];
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    xa[t] = xs[t][kl * 8 + q];
                    xb[t] = xs[t][kl * 8 + q + 4];
                }
                half2 acc2[2][T];
#pragma unroll
                for (int s = 0; s < 2; ++s) {
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        acc2[s][t] = __float2half2_rn(0.0f);
                    }
                }
#pragma unroll
                for (int jp = 0; jp < 4; ++jp) {
                    const int s = jp >> 1;
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        acc2[s][t] = __hfma2(wv[jp], (jp & 1) ? xb[t] : xa[t], acc2[s][t]);
                    }
                }
#pragma unroll
                for (int s = 0; s < 2; ++s) {
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        acc[s][t] += __half2float(__hadd(__low2half(acc2[s][t]), __high2half(acc2[s][t])));
                    }
                }
            }
        };
        auto fold = [&]() {   // mma: fp16 D (row g = d[.][0]) -> fp32, restart the fp16 accumulation
            if (MMA) {
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    const float2 v = __half22float2(exl3_u2h(d[s][0]));
                    accf[s].x += v.x;
                    accf[s].y += v.y;
                    if (XROWS > 8) {
                        const float2 u = __half22float2(exl3_u2h(d[s][1]));
                        accg[s].x += u.x;
                        accg[s].y += u.y;
                    }
                    d[s][0] = 0u; d[s][1] = 0u;
                }
            }
        };
        // weight words of tile u of a group -> the lane's NWORDS window words
        auto gather = [&](const uint32_t (&own)[U], uint32_t (&w)[U][NWORDS]) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
#pragma unroll
                for (int m = 0; m < NWORDS; ++m) {
                    if (BITS == 4 && m == 1) {
                        w[u][m] = own[u];                    // 4 bits: ws == lane - 1, word 1 is the lane's own
                    } else {
                        w[u][m] = __shfl_sync(0xffffffffu, own[u], woff[m]);
                    }
                }
            }
        };

        const int n_my = (nk - warp + EXL3_GEMV_NWARPS - 1) / EXL3_GEMV_NWARPS;   // tiles this warp owns
        const size_t sw = (size_t) stepW;                                         // words between this warp's tiles
        const uint32_t * wl[NWORDS];
#pragma unroll
        for (int m = 0; m < NWORDS; ++m) wl[m] = wp + (SHFL ? own_w : woff[m]);
        int step = 0;
        for (; step + U <= n_my; step += U) {
            uint32_t w[U][NWORDS];
            // Walk a running pointer instead of writing wl[m] + u * sw: sw is a runtime value (it contains nt), so the
            // scaled form made ptxas re-materialise a full 32-bit multiply by NWARPS*NW*4 for every grouped weight
            // load (SM60 has no single-instruction 32x32 multiply). Visible in the T==1 SASS as 2-3 XMAD chains per
            // load at 3 bits (NW=24 => x384), where the 2-bit kernel only paid shifts (NW=16 => x256).
            if (SHFL) {
                uint32_t own[U];
                const uint32_t * p = wl[0];
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    own[u] = __ldcs(p);
                    p += sw;
                }
                gather(own, w);
            } else {
                const uint32_t * p[NWORDS];
#pragma unroll
                for (int m = 0; m < NWORDS; ++m) p[m] = wl[m];
#pragma unroll
                for (int u = 0; u < U; ++u) {
#pragma unroll
                    for (int m = 0; m < NWORDS; ++m) {
                        w[u][m] = __ldcs(p[m]);
                        p[m] += sw;
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                tile_fma(w[u], (step + u) * EXL3_GEMV_NWARPS + warp);
            }
            fold();
#pragma unroll
            for (int m = 0; m < NWORDS; ++m) wl[m] += (size_t) U * sw;
        }
        for (; step < n_my; ++step) {
            uint32_t w[NWORDS];
            if (SHFL) {
                const uint32_t own = __ldcs(wl[0]);
#pragma unroll
                for (int m = 0; m < NWORDS; ++m) {
                    w[m] = (BITS == 4 && m == 1) ? own : __shfl_sync(0xffffffffu, own, woff[m]);
                }
            } else {
#pragma unroll
                for (int m = 0; m < NWORDS; ++m) {
                    w[m] = __ldcs(wl[m]);
                }
            }
            tile_fma(w, step * EXL3_GEMV_NWARPS + warp);
            fold();
#pragma unroll
            for (int m = 0; m < NWORDS; ++m) wl[m] += sw;
        }

        if (MMA) {
            // lane holds D[row g][cols 2q, 2q+1] of both n8 halves
            if (g < T) {
                part[warp][g][2 * q]         = accf[0].x;
                part[warp][g][2 * q + 1]     = accf[0].y;
                part[warp][g][8 + 2 * q]     = accf[1].x;
                part[warp][g][8 + 2 * q + 1] = accf[1].y;
            }
            if (XROWS > 8 && g + 8 < T) {   // D rows 8..15 (d[.][1]) hold x rows g + 8
                const int r = (XROWS > 8) ? g + 8 : 0;
                part[warp][r][2 * q]         = accg[0].x;
                part[warp][r][2 * q + 1]     = accg[0].y;
                part[warp][r][8 + 2 * q]     = accg[1].x;
                part[warp][r][8 + 2 * q + 1] = accg[1].y;
            }
        } else {
            // reduce the four lanes sharing a column
#pragma unroll
            for (int s = 0; s < 2; ++s) {
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    float v = acc[s][t];
                    v += __shfl_xor_sync(0xffffffffu, v, 1);
                    v += __shfl_xor_sync(0xffffffffu, v, 2);
                    acc[s][t] = v;
                }
            }
            if (q == 0) {
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    part[warp][t][g]     = acc[0][t];
                    part[warp][t][g + 8] = acc[1][t];
                }
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < 16 * T; i += 32 * EXL3_GEMV_NWARPS) {
            const int t = i >> 4;
            const int n = i & 15;
            float v = 0.0f;
#pragma unroll
            for (int wv = 0; wv < EXL3_GEMV_NWARPS; ++wv) {
                v += part[wv][t][n];
            }
            y[((size_t) ks * T + t) * N + (size_t) ni * 16 + n] = v * (1.0f / EXL3_GEMV_X_SCALE);
        }
        __syncthreads();   // part / xs are reused by the next item
    }

#if EXL3_GEMV_HAVE_COOP
    if (FUSED && post) {
        // phase 3: yf[T][N] = svh * H128( sum_ks part[ks][T][N] )   (svh null: the sum); one warp per 128-chunk
        cg::this_grid().sync();
        const int n_chunks = T * N / 128;
        const size_t ks_stride = (size_t) T * N;
        for (int c = gw; c < n_chunks; c += gws) {
            const size_t base = (size_t) c * 128;
            const int nc = (int) (base % (size_t) N);
            float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            int ks = 0;
            for (; ks + 4 <= ksplit; ks += 4) {          // four partial rows in flight, fixed summation order
                const float * p = y + (size_t) ks * ks_stride + base + lane;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float a0 = __ldcg(p + j * 32);
                    const float a1 = __ldcg(p + ks_stride + j * 32);
                    const float a2 = __ldcg(p + 2 * ks_stride + j * 32);
                    const float a3 = __ldcg(p + 3 * ks_stride + j * 32);
                    v[j] += (a0 + a1) + (a2 + a3);
                }
            }
            for (; ks < ksplit; ++ks) {
                const float * p = y + (size_t) ks * ks_stride + base + lane;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    v[j] += __ldcg(p + j * 32);
                }
            }
            float sc = 1.0f;
            if (svh != nullptr) {
                exl3_wht128(v);
                sc = EXL3_WHT128_SCALE;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float m = (svh != nullptr) ? svh[nc + j * 32 + lane] : 1.0f;
                yf[base + j * 32 + lane] = v[j] * sc * m;
            }
        }
    }
#endif
}

static int exl3_gemv_bps_cap() {   // EXL3_GEMV_BPS: cap the cooperative grid at this many blocks per SM (0 = none)
    static const int cap = [] {
        const char * env = getenv("EXL3_GEMV_BPS");
        return env ? atoi(env) : 0;
    }();
    return cap;
}

static int exl3_gemv_ktiles_max(const int T);
static int exl3_gemv_kpi_of(const int kt, const int ksplit, const int T);

struct exl3_gemv_args {
    const uint32_t * trellis; half2 * x; float * out; int kt, nt, K, N, ksplit;
    const float * xf; const float * suh; const float * svh; float * yf; bool post;
};

template <int BITS, int T, bool FUSED>
static void exl3_gemv_launch_bt(const exl3_gemv_args & a, cudaStream_t stream) {
    static int grid_max = 0;   // resident blocks on the device (per instantiation), measured once
    if (grid_max == 0) {
        int nb = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_exl3_gemv<BITS, T, FUSED>, 32 * EXL3_GEMV_NWARPS, 0));
        const int id = ggml_cuda_get_device();
        nb = std::max(1, nb);
        if (FUSED && exl3_gemv_bps_cap() > 0) {
            nb = std::min(nb, exl3_gemv_bps_cap());
        }
        grid_max = nb * ggml_cuda_info().devices[id].nsm;
    }
    const int kpi = exl3_gemv_kpi_of(a.kt, a.ksplit, T);                        // k-tiles per item (group-aligned at T==1)
    GGML_ASSERT(kpi <= exl3_gemv_cfg<T>::KTILES_MAX);
    const int n_items = a.nt * a.ksplit;
    const dim3 grid(std::min(n_items, grid_max));
    const dim3 block(32 * EXL3_GEMV_NWARPS);
    const uint32_t * trellis = a.trellis; half2 * x = a.x; float * out = a.out;
    int kt = a.kt, nt = a.nt, K = a.K, N = a.N, ksplit = a.ksplit, kpi_ = kpi;
    const float * xf = a.xf; const float * suh = a.suh; const float * svh = a.svh; float * yf = a.yf;
    int post = a.post ? 1 : 0;
    long long stepW = (long long) EXL3_GEMV_NWARPS * a.nt * 8 * BITS;   // words between one warp's consecutive k-tiles
#if EXL3_GEMV_HAVE_COOP
    if (FUSED) {
        void * args[] = {&trellis, &x, &out, &kt, &nt, &K, &N, &ksplit, &kpi_, &xf, &suh, &svh, &yf, &post, &stepW};
        CUDA_CHECK(cudaLaunchCooperativeKernel((const void *) k_exl3_gemv<BITS, T, FUSED>, grid, block, args, 0, stream));
        return;
    }
#endif
    k_exl3_gemv<BITS, T, FUSED><<<grid, block, 0, stream>>>(trellis, x, out, kt, nt, K, N, ksplit, kpi_, xf, suh, svh, yf, post, stepW);
}

template <int T, bool FUSED>
static void exl3_gemv_launch(const exl3_gemv_args & a, const int bits, cudaStream_t stream) {
    switch (bits) {
        case 2: exl3_gemv_launch_bt<2, T, FUSED>(a, stream); break;
        case 3: exl3_gemv_launch_bt<3, T, FUSED>(a, stream); break;
        case 4: exl3_gemv_launch_bt<4, T, FUSED>(a, stream); break;
        case 5: exl3_gemv_launch_bt<5, T, FUSED>(a, stream); break;
        case 6: exl3_gemv_launch_bt<6, T, FUSED>(a, stream); break;
        case 7: exl3_gemv_launch_bt<7, T, FUSED>(a, stream); break;
        case 8: exl3_gemv_launch_bt<8, T, FUSED>(a, stream); break;
        default: GGML_ABORT("exl3 gemv: unsupported bits");
    }
}

template <bool FUSED>
static void exl3_gemv_launch_t(const exl3_gemv_args & a, const int bits, const int T, cudaStream_t stream) {
    if constexpr (!EXL3_GEMV_MMA_COMPILED) {
        GGML_ASSERT(T == 1);
        exl3_gemv_launch<1, FUSED>(a, bits, stream);
    } else {
        switch (T) {
            case 1: exl3_gemv_launch<1, FUSED>(a, bits, stream); break;
            case 2: exl3_gemv_launch<2, FUSED>(a, bits, stream); break;
            case 3: exl3_gemv_launch<3, FUSED>(a, bits, stream); break;
            case 4: exl3_gemv_launch<4, FUSED>(a, bits, stream); break;
            case 5: exl3_gemv_launch<5, FUSED>(a, bits, stream); break;
            case 6: exl3_gemv_launch<6, FUSED>(a, bits, stream); break;
            case 7: exl3_gemv_launch<7, FUSED>(a, bits, stream); break;
            case 8: exl3_gemv_launch<8, FUSED>(a, bits, stream); break;
            case 9: exl3_gemv_launch<9, FUSED>(a, bits, stream); break;
            case 10: exl3_gemv_launch<10, FUSED>(a, bits, stream); break;
            case 11: exl3_gemv_launch<11, FUSED>(a, bits, stream); break;
            case 12: exl3_gemv_launch<12, FUSED>(a, bits, stream); break;
            case 13: exl3_gemv_launch<13, FUSED>(a, bits, stream); break;
            case 14: exl3_gemv_launch<14, FUSED>(a, bits, stream); break;
            case 15: exl3_gemv_launch<15, FUSED>(a, bits, stream); break;
            case 16: exl3_gemv_launch<16, FUSED>(a, bits, stream); break;
            default: GGML_ABORT("exl3 gemv: unsupported T");
        }
    }
}

static bool exl3_gemv_fused_enabled() {
#if EXL3_GEMV_HAVE_COOP
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_EXL3_FUSED");   // M4 measured slower (barriers on a 1008-block grid): opt-in
        return env != nullptr && strcmp(env, "1") == 0;
    }();
    return enabled;
#else
    return false;
#endif
}

bool ggml_cuda_exl3_gemv_supported(const int64_t T) {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_EXL3_GEMV");
        return env == nullptr || strcmp(env, "0") != 0;
    }();
    if (!enabled || T < 1 || T > EXL3_GEMV_MAX_T) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    return cc >= GGML_CUDA_CC_VOLTA || T == 1;
}

static int exl3_gemv_ktiles_max(const int T) {
    static const int tab[EXL3_GEMV_MAX_T] = {
        exl3_gemv_cfg<1>::KTILES_MAX, exl3_gemv_cfg<2>::KTILES_MAX, exl3_gemv_cfg<3>::KTILES_MAX, exl3_gemv_cfg<4>::KTILES_MAX,
        exl3_gemv_cfg<5>::KTILES_MAX, exl3_gemv_cfg<6>::KTILES_MAX, exl3_gemv_cfg<7>::KTILES_MAX, exl3_gemv_cfg<8>::KTILES_MAX,
        exl3_gemv_cfg<9>::KTILES_MAX, exl3_gemv_cfg<10>::KTILES_MAX, exl3_gemv_cfg<11>::KTILES_MAX, exl3_gemv_cfg<12>::KTILES_MAX,
        exl3_gemv_cfg<13>::KTILES_MAX, exl3_gemv_cfg<14>::KTILES_MAX, exl3_gemv_cfg<15>::KTILES_MAX, exl3_gemv_cfg<16>::KTILES_MAX,
    };
    return tab[T - 1];
}

static int exl3_gemv_kpi_of(const int kt, const int ksplit, const int T) {
    const int kpi = (kt + ksplit - 1) / ksplit;
    if (T != 1) {
        return kpi;
    }
    // Align the item to a whole number of block group steps (NWARPS * U k-tiles): a warp whose item ends mid-group
    // drops into the scalar per-tile tail loop (no grouped loads, no grouped shuffle gather), which is much slower.
    const int grp = EXL3_GEMV_NWARPS * EXL3_GEMV_U;
    const int r   = (kpi + grp - 1) / grp * grp;
    return r > exl3_gemv_ktiles_max(T) ? exl3_gemv_ktiles_max(T) : r;
}

int ggml_cuda_exl3_gemv_ksplit(const int kt, const int nt, const int T) {
    const int ktiles_max = exl3_gemv_ktiles_max(T);
    const int ksplit_max = kt / (2 * EXL3_GEMV_NWARPS);   // >= 2 k-tiles per warp
    int ksplit;
    if (T == 1) {
        // Single token: every work item costs a fixed amount of overhead (stage x into shared memory, two barriers,
        // write the split-K partial) that is worth roughly 15-20 k-tiles of decode, and the kernel is latency rather
        // than bandwidth bound. Splitting k beyond what the staging buffer forces therefore costs more than the
        // smoother load balance buys: sweeping ksplit over the real Qwen3.8-27B shape mix (2.50 bpw 46.1 -> 45.4
        // ms/token, 4.00 bpw 44.9 -> 44.3) picked the *coarsest* legal split on every shape with a wide output, and
        // the old ceil(1536/nt) target was 10-30% off on attn_gate/ssm_out/attn_k/attn_v. Narrow outputs are the one
        // exception: one item per block would leave most of the machine with no work at all, so keep half of the
        // resident grid busy there.
        ksplit = (kt + ktiles_max - 1) / ktiles_max;
        if (ksplit < 1) ksplit = 1;
        const int grid_half = EXL3_GEMV_MIN_BLOCKS * ggml_cuda_info().devices[ggml_cuda_get_device()].nsm / 2;
        while (nt * ksplit < grid_half && ksplit < ksplit_max) ksplit++;
        // Retry with an aligned item length if that still keeps enough work items alive: kt = 1088 (ffn_down, ~21% of
        // a 2.50 bpw token) at the coarsest legal ksplit = 6 has kpi = 182 = 5.69 groups, so 6 of every 46 tiles per
        // warp ran in the scalar tail; rounding kpi up to 192 leaves ksplit at 6 and every item a whole group count.
        const int ks_al = (kt + exl3_gemv_kpi_of(kt, ksplit, T) - 1) / exl3_gemv_kpi_of(kt, ksplit, T);
        if (nt * ks_al >= grid_half) ksplit = ks_al;
    } else {
        // enough work items to cover the SMs several times over, at most KTILES_MAX k-tiles per item (x is staged
        // per item in shared memory), and >= 2 k-tiles per warp
        ksplit = (1536 + nt - 1) / nt;
        if (ksplit > ksplit_max) ksplit = ksplit_max;
        if (ksplit < 1) ksplit = 1;
    }
    // kpi = ceil(kt/ksplit) must fit the staging buffer
    while ((kt + ksplit - 1) / ksplit > ktiles_max) ksplit++;
    return ksplit;
}

void ggml_cuda_exl3_gemv(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const int64_t T,
                         const float * suh, const float * svh) {
    const int bits = ggml_exl3_bits(src0->type);
    GGML_ASSERT(bits != 0 && T >= 1 && T <= EXL3_GEMV_MAX_T);
    const int K  = (int) src0->ne[0];
    const int N  = (int) src0->ne[1];
    const int kt = K / 16;
    const int nt = N / 16;
    const int ksplit = ggml_cuda_exl3_gemv_ksplit(kt, nt, (int) T);

    const int id = ggml_cuda_get_device();
    cudaStream_t stream = ctx.stream();

    // x -> fp16 [T][K], pre-scaled (and suh * / Hadamard when fused)
    ggml_cuda_pool_alloc<half> xh(ctx.pool(id), (size_t) T * K);
    const bool fused = exl3_gemv_fused_enabled();
    if (!fused) {
        ggml_cuda_exl3_glue_in_f16((const float *) src1->data, suh, xh.get(), K, T, EXL3_GEMV_X_SCALE, stream);
    }

    const bool post = ksplit > 1 || svh != nullptr;
    ggml_cuda_pool_alloc<float> part(ctx.pool(id));
    float * y   = (float *) dst->data;
    float * out = post ? part.alloc((size_t) ksplit * T * N) : y;
    exl3_gemv_args a;
    a.trellis = (const uint32_t *) src0->data; a.x = (half2 *) xh.get(); a.out = out;
    a.kt = kt; a.nt = nt; a.K = K; a.N = N; a.ksplit = ksplit;
    a.xf = (const float *) src1->data; a.suh = suh; a.svh = svh; a.yf = y; a.post = post;
    if (fused) {
        exl3_gemv_launch_t<true>(a, bits, (int) T, stream);
        return;
    }
    exl3_gemv_launch_t<false>(a, bits, (int) T, stream);
    if (post) {
        ggml_cuda_exl3_glue_out(out, svh, y, N, T, ksplit, 1.0f, stream);
    }
}
