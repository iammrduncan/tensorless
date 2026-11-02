// fattn-sm60.cuh -- split-KV flash-decoding for Tesla P100 (sm_60) with q4_0 K/V cache, head dim 256.
//
// Custom kernel "K-A" of the P100 v2 plan (IMPLEMENTATION-PLAN.md). Self-contained (CUDA runtime + cuda_fp16 only)
// so the same code runs in the standalone tests (kernels/fattn-sm60/test_fattn.cu) and inside ggml-cuda.
//
// Why a new kernel: the fork's flash_attn_ext_vec<256,...,q4_0,q4_0> reads the cache at ~38 GB/s on sm_60
// (v1 R06: 12.5 ms of depth term at 25K occupied vs 4.95 ms for f16 K/V moving 3.6x the bytes). It is
// latency-bound: 255 registers, 2 blocks/SM, one warp per K row with a 32-lane reduction, and each K row loaded
// in 2-byte pieces. On Pascal with >2 query columns the dispatcher falls back to the TILE kernel, which
// dequantizes the WHOLE cache to f16 on every call. This kernel serves 1..8 query tokens x all GQA query heads of
// a KV head from one read of each K/V byte.
//
// Design (one CTA = one KV head x one chunk of the KV sequence, 128 threads):
//   * Q for the CTA's GQA heads x T tokens is scaled, converted to fp16 and stored in shared memory in the
//     *permuted pair order* produced by the nibble dequantizer below, so K never needs re-ordering.
//   * QK: thread-per-position. Each thread loads its whole q4_0 K row (144 B = 9 x 16-byte loads, all in flight
//     at once), dequantizes nibbles with the 0x6400 magic (LOP3 + HSUB2, exact integers q-8 in fp16) and runs
//     HFMA2 against Q from shared memory (LDS.128 broadcasts). Per 32-element block the fp16 partial (two lanes of
//     8 products each) is folded into an fp32 logit with the block scale.
//   * Online softmax in fp32 per query column; masked / out-of-range positions contribute exactly 0.
//   * PV: the V tile is staged in shared memory, re-packed so every quant word is 4-byte aligned; each thread owns
//     2 output dims and accumulates p * v in fp32 over all positions of the tile.
//   * A second kernel merges the per-chunk (max, sum, acc) partials with the log-sum-exp rule.
//
// Numerics: logits carry an fp16 partial-sum error of ~1e-4 relative; everything after the logit is fp32.
// This is NOT bit-identical to the fork's q8_1 integer path; promotion goes through the v2 distributional gate.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace sm60fa {

constexpr int D          = 256;           // head dim (K and V)
constexpr int QK4        = 32;            // q4_0 block size
constexpr int NBLK       = D / QK4;       // 8 blocks per row
constexpr int BLK_BYTES  = 18;            // q4_0 block: half d + 16 bytes of nibbles
constexpr int ROW_BYTES  = NBLK * BLK_BYTES;  // 144
constexpr int ROW_WORDS  = ROW_BYTES / 4;     // 36
constexpr int NTHREADS   = 128;
constexpr int TILE       = NTHREADS;      // KV positions per tile (thread-per-position for QK)
constexpr int MAX_T      = 8;             // max query tokens per call (verify width + 1)

// Element index (within a 32-element block) of the first member of permuted pair p (0..15); second = first + 2.
//   p = 4*i + r : r=0 -> (4i, 4i+2), r=1 -> (4i+16, 4i+18), r=2 -> (4i+1, 4i+3), r=3 -> (4i+17, 4i+19)
__host__ __device__ constexpr int pair_elem0(int p) {
    return (p / 4) * 4 + ((p % 4) == 0 ? 0 : (p % 4) == 1 ? 16 : (p % 4) == 2 ? 1 : 17);
}

__device__ __forceinline__ half2 u32_as_half2(uint32_t u) { return *reinterpret_cast<half2 *>(&u); }
__device__ __forceinline__ uint32_t half2_as_u32(half2 h) { return *reinterpret_cast<uint32_t *>(&h); }

// q4_0 word (4 bytes of qs = elements 4i..4i+3 low nibbles, 4i+16..4i+19 high nibbles) -> 4 half2 of (q - 8),
// in permuted pair order r = 0..3.
__device__ __forceinline__ void dq4_word(const uint32_t w, half2 (&h)[4]) {
    const uint32_t m  = 0x000F000Fu;
    const uint32_t mg = 0x64006400u;                       // fp16 1024.0 in both halves
    const half2 off   = u32_as_half2(0x64086408u);          // 1032.0 = 1024 + 8 (q4_0 zero point)
    h[0] = __hsub2(u32_as_half2(( w        & m) | mg), off);
    h[1] = __hsub2(u32_as_half2(((w >>  4) & m) | mg), off);
    h[2] = __hsub2(u32_as_half2(((w >>  8) & m) | mg), off);
    h[3] = __hsub2(u32_as_half2(((w >> 12) & m) | mg), off);
}

// Word k (0..3) of the qs of block b, and the scale bits of block b, from a row held as 36 u32 (144 B).
template <int b, int k>
__device__ __forceinline__ uint32_t row_qword(const uint32_t (&r)[ROW_WORDS]) {
    constexpr int off = b * BLK_BYTES + 2 + 4 * k;   // byte offset of the word
    if constexpr ((off & 3) == 0) {
        return r[off >> 2];
    } else {                                         // off % 4 == 2
        return __funnelshift_r(r[off >> 2], r[(off >> 2) + 1], 16);
    }
}
template <int b>
__device__ __forceinline__ uint32_t row_dbits(const uint32_t (&r)[ROW_WORDS]) {
    constexpr int off = b * BLK_BYTES;
    if constexpr ((off & 3) == 0) {
        return r[off >> 2] & 0xFFFFu;
    } else {
        return r[off >> 2] >> 16;
    }
}

__device__ __forceinline__ void load_row(const char * __restrict__ p, uint32_t (&r)[ROW_WORDS]) {
    const uint4 * p4 = reinterpret_cast<const uint4 *>(p);
#pragma unroll
    for (int i = 0; i < ROW_WORDS / 4; ++i) {
        const uint4 v = __ldg(p4 + i);
        r[4 * i + 0] = v.x; r[4 * i + 1] = v.y; r[4 * i + 2] = v.z; r[4 * i + 3] = v.w;
    }
}

struct Params {
    const float * Q;        // [D, T, n_head] f32, strides in floats
    const char  * K;        // q4_0 rows
    const char  * V;        // q4_0 rows
    const half  * mask;     // [n_kv, >=T] f16 rows (0 / -inf), or nullptr
    float       * dst;      // [D, n_head, T] f32 contiguous
    float       * part_acc; // [n_head_kv][n_splits][C][D]
    float2      * part_ml;  // [n_head_kv][n_splits][C]  (max, sum)
    int   n_kv;             // KV positions (padded length of the view)
    int   n_head;           // query heads
    int   n_head_kv;
    int   T;                // query tokens (1..8)
    int   kv_per_split;     // multiple of TILE
    int   n_splits;
    long  q_tok, q_head;    // Q strides (floats)
    long  k_pos, k_head;    // K strides (bytes)
    long  v_pos, v_head;    // V strides (bytes)
    long  m_row;            // mask row stride (halfs)
    float scale;
};

// Shared memory layout (bytes), C = GQA * T columns:
//   Qs   : C * NBLK * 16 half2        (C * 512 B)
//   Vt   : TILE * ROW_WORDS u32        (18,432 B) V tile, repacked: [8 scales as 4 words][8 blocks x 4 qs words]
//   Pt   : TILE * C float              (logits, then probabilities), layout [pos][c]
//   stat : C * 4 floats                (running max, running sum, tile alpha, spare)
template <int GQA, int T>
struct Smem {
    static constexpr int C       = GQA * T;
    static constexpr int Q_BYTES = C * NBLK * 16 * 4;
    // V is staged in VSPLIT chunks of TILE/VSPLIT positions. At T == 1 the kernel is REGISTER-capped
    // at 4 CTAs/SM, so a deeper split would only add staging rounds; at T > 1 the 2-way split's
    // 21696 B is what caps <6,2> at 3 CTAs/SM (its 124 regs already allow 4), and 8-way brings the
    // total to 14784 B <= 16384, i.e. 4 CTAs/SM for one extra pair of staging rounds per tile.
    static constexpr int VSPLIT  = (T > 1) ? 8 : 2;
    static constexpr int V_BYTES = (TILE/VSPLIT) * ROW_WORDS * 4;
    static constexpr int P_BYTES = TILE * C * 4;
    static constexpr int S_BYTES = C * 4 * 4;
    static constexpr int TOTAL   = Q_BYTES + V_BYTES + P_BYTES + S_BYTES;
};

// T=1 is register-capped at 3 CTAs/SM (152 regs); __launch_bounds__(128,4) was tried and spills (140 B) then loses
// ~17% standalone. Keep the unconstrained schedule.
template <int GQA, int T, bool PVHFMA2 = false>
__global__ void __launch_bounds__(NTHREADS)
fattn_q4_sm60_chunk(const Params p) {
    constexpr int C = GQA * T;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    uint4    * Qs4  = reinterpret_cast<uint4 *>(smem_raw);                                   // [C][NBLK][4] uint4
    uint32_t * Vt   = reinterpret_cast<uint32_t *>(smem_raw + Smem<GQA, T>::Q_BYTES);          // [TILE][36]
    float    * Pt   = reinterpret_cast<float *>(smem_raw + Smem<GQA, T>::Q_BYTES + Smem<GQA, T>::V_BYTES);
    float    * stat = Pt + TILE * C;                                                          // [C][4]

    const int tid   = threadIdx.x;
    const int split = blockIdx.x;
    const int hkv   = blockIdx.y;
    const int kv0   = split * p.kv_per_split;
    const int kv1   = min(kv0 + p.kv_per_split, p.n_kv);

    // ---- stage Q (scaled, fp16, permuted pair order) -------------------------------------------------------------
    // column c = t * GQA + g  ->  query head h = hkv * GQA + g, token t
    half2 * Qs2 = reinterpret_cast<half2 *>(smem_raw);
    for (int idx = tid; idx < C * NBLK * 16; idx += NTHREADS) {
        const int c  = idx / (NBLK * 16);
        const int b  = (idx / 16) % NBLK;
        const int pr = idx % 16;
        const int t  = c / GQA;
        const int h  = hkv * GQA + (c % GQA);
        const float * q = p.Q + (long) t * p.q_tok + (long) h * p.q_head + b * QK4;
        const int e0 = pair_elem0(pr);
        Qs2[idx] = __floats2half2_rn(q[e0] * p.scale, q[e0 + 2] * p.scale);
    }
    if (tid < C) {
        stat[tid * 4 + 0] = -INFINITY;  // running max
        stat[tid * 4 + 1] = 0.0f;       // running sum
        stat[tid * 4 + 2] = 1.0f;       // tile alpha
    }

    // PV ownership: thread owns permuted pair pr of block b  ->  word k = pr / 4, nibble shift r = pr % 4
    const int pv_b  = tid / 16;
    const int pv_pr = tid % 16;
    const int pv_k  = pv_pr / 4;
    const int pv_sh = 4 * (pv_pr % 4);
    float acc[C][2];
#pragma unroll
    for (int c = 0; c < C; ++c) { acc[c][0] = 0.0f; acc[c][1] = 0.0f; }

    __syncthreads();

    for (int tile0 = kv0; tile0 < kv1; tile0 += TILE) {
        const int  pos   = tile0 + tid;
        const bool valid = pos < kv1;
        const int  n_in  = min(TILE, kv1 - tile0);

        // ---- load this thread's K row (the V row is loaded in the PV phase so kr/vr never share a live range) ------
        uint32_t kr[ROW_WORDS];
        if (valid) {
            load_row(p.K + (long) pos * p.k_pos + (long) hkv * p.k_head, kr);
        } else {
#pragma unroll
            for (int i = 0; i < ROW_WORDS; ++i) { kr[i] = 0u; }
        }
        uint32_t vr[ROW_WORDS];

        // ---- V row staging is deferred into the PV phase, in two TILE/2-position halves, so Vt is half the size.
        auto stage_v = [&](const int slot) {
            uint32_t * dstv = Vt + slot * ROW_WORDS;
            dstv[0] = row_dbits<0>(vr) | (row_dbits<1>(vr) << 16);
            dstv[1] = row_dbits<2>(vr) | (row_dbits<3>(vr) << 16);
            dstv[2] = row_dbits<4>(vr) | (row_dbits<5>(vr) << 16);
            dstv[3] = row_dbits<6>(vr) | (row_dbits<7>(vr) << 16);
#define SM60FA_VQ(b) dstv[4 + 4 * b + 0] = row_qword<b, 0>(vr); dstv[4 + 4 * b + 1] = row_qword<b, 1>(vr); \
                     dstv[4 + 4 * b + 2] = row_qword<b, 2>(vr); dstv[4 + 4 * b + 3] = row_qword<b, 3>(vr);
            SM60FA_VQ(0) SM60FA_VQ(1) SM60FA_VQ(2) SM60FA_VQ(3) SM60FA_VQ(4) SM60FA_VQ(5) SM60FA_VQ(6) SM60FA_VQ(7)
#undef SM60FA_VQ
        };

        // ---- QK: logits for all C columns at this thread's position -------------------------------------------
        float s[C];
#pragma unroll
        for (int c = 0; c < C; ++c) { s[c] = 0.0f; }
#define SM60FA_QK_BLOCK(b)                                                                                       \
        {                                                                                                        \
            half2 hk[16];                                                                                        \
            { half2 t4[4]; dq4_word(row_qword<b, 0>(kr), t4); hk[0] = t4[0]; hk[1] = t4[1]; hk[2] = t4[2]; hk[3] = t4[3]; }   \
            { half2 t4[4]; dq4_word(row_qword<b, 1>(kr), t4); hk[4] = t4[0]; hk[5] = t4[1]; hk[6] = t4[2]; hk[7] = t4[3]; }   \
            { half2 t4[4]; dq4_word(row_qword<b, 2>(kr), t4); hk[8] = t4[0]; hk[9] = t4[1]; hk[10] = t4[2]; hk[11] = t4[3]; } \
            { half2 t4[4]; dq4_word(row_qword<b, 3>(kr), t4); hk[12] = t4[0]; hk[13] = t4[1]; hk[14] = t4[2]; hk[15] = t4[3]; }\
            const float db = __half2float(__ushort_as_half((unsigned short) row_dbits<b>(kr)));                 \
            _Pragma("unroll")                                                                                    \
            for (int c = 0; c < C; ++c) {                                                                        \
                const uint4 * q4 = Qs4 + (c * NBLK + b) * 4;                                                     \
                half2 a0 = __float2half2_rn(0.0f), a1 = __float2half2_rn(0.0f);                                   \
                _Pragma("unroll")                                                                                \
                for (int j = 0; j < 4; ++j) {                                                                    \
                    const uint4 qv = q4[j];                                                                      \
                    if (j < 2) {                                                                                 \
                        a0 = __hfma2(hk[4 * j + 0], u32_as_half2(qv.x), a0);                                     \
                        a0 = __hfma2(hk[4 * j + 1], u32_as_half2(qv.y), a0);                                     \
                        a0 = __hfma2(hk[4 * j + 2], u32_as_half2(qv.z), a0);                                     \
                        a0 = __hfma2(hk[4 * j + 3], u32_as_half2(qv.w), a0);                                     \
                    } else {                                                                                     \
                        a1 = __hfma2(hk[4 * j + 0], u32_as_half2(qv.x), a1);                                     \
                        a1 = __hfma2(hk[4 * j + 1], u32_as_half2(qv.y), a1);                                     \
                        a1 = __hfma2(hk[4 * j + 2], u32_as_half2(qv.z), a1);                                     \
                        a1 = __hfma2(hk[4 * j + 3], u32_as_half2(qv.w), a1);                                     \
                    }                                                                                            \
                }                                                                                                \
                const float2 f0 = __half22float2(a0), f1 = __half22float2(a1);                                   \
                s[c] = fmaf(db, (f0.x + f0.y) + (f1.x + f1.y), s[c]);                                            \
            }                                                                                                    \
        }
        SM60FA_QK_BLOCK(0) SM60FA_QK_BLOCK(1) SM60FA_QK_BLOCK(2) SM60FA_QK_BLOCK(3)
        SM60FA_QK_BLOCK(4) SM60FA_QK_BLOCK(5) SM60FA_QK_BLOCK(6) SM60FA_QK_BLOCK(7)
#undef SM60FA_QK_BLOCK

        // mask / range -> logits into Pt[pos][c]
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float v = -INFINITY;
            if (valid) {
                const int t = c / GQA;
                const float mk = p.mask ? __half2float(p.mask[(long) t * p.m_row + pos]) : 0.0f;
                v = s[c] + mk;
            }
            Pt[tid * C + c] = v;
        }
        __syncthreads();

        // ---- online softmax per column (columns spread over the 4 warps) -----------------------------------------
        {
            const int warp = tid / 32, lane = tid % 32;
            for (int c = warp; c < C; c += NTHREADS / 32) {
                float l0 = Pt[(lane +  0) * C + c], l1 = Pt[(lane + 32) * C + c];
                float l2 = Pt[(lane + 64) * C + c], l3 = Pt[(lane + 96) * C + c];
                float mx = fmaxf(fmaxf(l0, l1), fmaxf(l2, l3));
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) { mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o)); }
                const float m_old = stat[c * 4 + 0];
                const float m_new = fmaxf(m_old, mx);
                // all-masked so far: keep probabilities exactly 0 and avoid (-inf) - (-inf)
                const bool  dead  = (m_new == -INFINITY);
                const float e0 = dead ? 0.0f : __expf(l0 - m_new), e1 = dead ? 0.0f : __expf(l1 - m_new);
                const float e2 = dead ? 0.0f : __expf(l2 - m_new), e3 = dead ? 0.0f : __expf(l3 - m_new);
                if (PVHFMA2) {
                    // pre-broadcast (p,p) half2: still 4 B per (position, column)
                    reinterpret_cast<half2 *>(Pt)[(lane +  0) * C + c] = __float2half2_rn(e0);
                    reinterpret_cast<half2 *>(Pt)[(lane + 32) * C + c] = __float2half2_rn(e1);
                    reinterpret_cast<half2 *>(Pt)[(lane + 64) * C + c] = __float2half2_rn(e2);
                    reinterpret_cast<half2 *>(Pt)[(lane + 96) * C + c] = __float2half2_rn(e3);
                } else {
                    Pt[(lane +  0) * C + c] = e0; Pt[(lane + 32) * C + c] = e1;
                    Pt[(lane + 64) * C + c] = e2; Pt[(lane + 96) * C + c] = e3;
                }
                float sm = (e0 + e1) + (e2 + e3);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) { sm += __shfl_xor_sync(0xffffffffu, sm, o); }
                if (lane == 0) {
                    const float alpha = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
                    stat[c * 4 + 0] = m_new;
                    stat[c * 4 + 1] = stat[c * 4 + 1] * alpha + sm;
                    stat[c * 4 + 2] = alpha;
                }
            }
        }
        __syncthreads();

        // ---- PV: this thread's 2 output dims for all C columns over the tile ---------------------------------------
#pragma unroll
        for (int c = 0; c < C; ++c) {
            const float alpha = stat[c * 4 + 2];
            acc[c][0] *= alpha;
            acc[c][1] *= alpha;
        }
        {
            if (valid) {
                load_row(p.V + (long) pos * p.v_pos + (long) hkv * p.v_head, vr);
            } else {
#pragma unroll
                for (int i = 0; i < ROW_WORDS; ++i) { vr[i] = 0u; }
            }
            const half2 off = u32_as_half2(0x64086408u);
            half2 hacc[C];
            if (PVHFMA2) {
#pragma unroll
                for (int c = 0; c < C; ++c) { hacc[c] = __float2half2_rn(0.0f); }
            }
            // flush the half2 partials into the fp32 accumulators (which carry the alpha rescale)
            auto flush_hacc = [&]() {
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    acc[c][0] += __low2float(hacc[c]);
                    acc[c][1] += __high2float(hacc[c]);
                    hacc[c] = __float2half2_rn(0.0f);
                }
            };
#pragma unroll
            for (int vh = 0; vh < Smem<GQA, T>::VSPLIT; ++vh) {
                const int p0 = vh * (TILE/Smem<GQA, T>::VSPLIT);
                const int hc = min(TILE/Smem<GQA, T>::VSPLIT, n_in - p0);
                if (hc <= 0) {
                    break;
                }
                if (tid >= vh*(TILE/Smem<GQA, T>::VSPLIT) && tid < (vh + 1)*(TILE/Smem<GQA, T>::VSPLIT)) {
                    stage_v(tid - vh*(TILE/Smem<GQA, T>::VSPLIT));
                }
                __syncthreads();
                if (PVHFMA2 && hc == TILE/Smem<GQA, T>::VSPLIT) {
                    // Guard-free fast path: 16-position blocks off ONE per-block base, so the
                    // per-position Vt/Pt offsets fold into immediates like the OFF loop's LDS.128.
                    for (int jb = 0; jb < hc; jb += 8) {
                        const uint32_t * rowb = Vt + jb * ROW_WORDS;
                        const half2    * pb   = reinterpret_cast<const half2 *>(Pt) + (p0 + jb) * C;
#pragma unroll 4
                        for (int jj = 0; jj < 8; ++jj) {
                            const uint32_t sc = rowb[jj*ROW_WORDS + (pv_b >> 1)];
                            const half     d  = __ushort_as_half((unsigned short) ((pv_b & 1) ? (sc >> 16) : (sc & 0xFFFFu)));
                            const uint32_t w  = rowb[jj*ROW_WORDS + 4 + 4*pv_b + pv_k];
                            const half2    q  = __hsub2(u32_as_half2(((w >> pv_sh) & 0x000F000Fu) | 0x64006400u), off);
                            const half2    v  = __hmul2(q, __half2half2(d));
#pragma unroll
                            for (int c = 0; c < C; ++c) { hacc[c] = __hfma2(pb[jj*C + c], v, hacc[c]); }
                        }
                        flush_hacc();
                    }
                    __syncthreads();
                    continue;
                }
                for (int j = 0; j < hc; ++j) {
                    const uint32_t * row = Vt + j * ROW_WORDS;
                    const uint32_t   sc  = row[pv_b >> 1];
                    const half       d   = __ushort_as_half((unsigned short) ((pv_b & 1) ? (sc >> 16) : (sc & 0xFFFFu)));
                    const uint32_t   w   = row[4 + 4 * pv_b + pv_k];
                    const half2      q   = __hsub2(u32_as_half2(((w >> pv_sh) & 0x000F000Fu) | 0x64006400u), off);
                    if (PVHFMA2) {
                        const half2  v   = __hmul2(q, __half2half2(d));
                        const half2 *pj  = reinterpret_cast<const half2 *>(Pt) + (p0 + j) * C;
#pragma unroll
                        for (int c = 0; c < C; ++c) { hacc[c] = __hfma2(pj[c], v, hacc[c]); }
                        if ((j & 7) == 7) { flush_hacc(); }
                    } else {
                        const float2     v   = __half22float2(__hmul2(q, __half2half2(d)));
                        const float    * pj  = Pt + (p0 + j) * C;
#pragma unroll
                        for (int c = 0; c < C; ++c) {
                            const float pr = pj[c];
                            acc[c][0] = fmaf(pr, v.x, acc[c][0]);
                            acc[c][1] = fmaf(pr, v.y, acc[c][1]);
                        }
                    }
                }
                if (PVHFMA2) { flush_hacc(); }
                __syncthreads();  // Vt half reused by the next half / tile
            }
        }
    }

    // ---- write partials (un-permute dims) -------------------------------------------------------------------------
    const long base = (long) (hkv * p.n_splits + split);
    const int  e0   = pv_b * QK4 + pair_elem0(pv_pr);
#pragma unroll
    for (int c = 0; c < C; ++c) {
        float * a = p.part_acc + (base * C + c) * D;
        a[e0]     = acc[c][0];
        a[e0 + 2] = acc[c][1];
    }
    if (tid < C) {
        p.part_ml[base * C + tid] = make_float2(stat[tid * 4 + 0], stat[tid * 4 + 1]);
    }
}

// Merge split-KV partials: grid (n_head, T), block D threads.
template <int GQA>
__global__ void __launch_bounds__(D)
fattn_q4_sm60_combine(const Params p) {
    const int h   = blockIdx.x;
    const int t   = blockIdx.y;
    const int hkv = h / GQA;
    const int C   = GQA * p.T;
    const int c   = t * GQA + (h % GQA);
    const int d   = threadIdx.x;

    extern __shared__ float2 ml[];  // n_splits entries
    for (int i = d; i < p.n_splits; i += D) {
        ml[i] = p.part_ml[(long) (hkv * p.n_splits + i) * C + c];
    }
    __syncthreads();
    float M = -INFINITY;
    for (int i = 0; i < p.n_splits; ++i) { M = fmaxf(M, ml[i].x); }
    float num = 0.0f, den = 0.0f;
    // The part_acc reads are strided by C*D, so with only ~32 blocks resident each one costs a full
    // memory latency and the serial loop stalls on them. Hoisting the base/stride and unrolling lets
    // several loads be in flight. Same accumulation order, so the result is unchanged.
    const float * acc = p.part_acc + (((long) hkv * p.n_splits) * C + c) * D + d;
    const long    acc_stride = (long) C * D;
#pragma unroll 4
    for (int i = 0; i < p.n_splits; ++i, acc += acc_stride) {
        if (ml[i].x == -INFINITY) { continue; }
        const float w = __expf(ml[i].x - M);
        den = fmaf(w, ml[i].y, den);
        num = fmaf(w, *acc, num);
    }
    p.dst[((long) t * p.n_head + h) * D + d] = den > 0.0f ? num / den : 0.0f;
}

// Workspace needed for a call (bytes).
inline size_t workspace_bytes(int n_head_kv, int n_splits, int C) {
    return (size_t) n_head_kv * n_splits * C * (D * sizeof(float) + sizeof(float2));
}

// Choose the KV chunk so that there are ~`target_ctas` CTAs (>= ~8 per SM on a 56-SM P100).
inline void choose_split(int n_kv, int n_head_kv, int target_ctas, int & kv_per_split, int & n_splits) {
    int want = target_ctas / (n_head_kv > 0 ? n_head_kv : 1);
    if (want < 1) { want = 1; }
    kv_per_split = (n_kv + want - 1) / want;
    kv_per_split = ((kv_per_split + TILE - 1) / TILE) * TILE;
    if (kv_per_split < TILE) { kv_per_split = TILE; }
    n_splits = (n_kv + kv_per_split - 1) / kv_per_split;
}

// R-28 KA-PIN: derive the split from the ALLOCATED kv length, not from this op's padded n_kv. With n_kv the
// split steps 1152 -> 1280 once, partway through a 65K cell, so a verify row and its serial step can split
// differently - a prerequisite for exact speculation. The allocated length is fixed per cell, so the split is
// too. n_splits still covers the length actually attended. At 32K both derivations give 640, so the pin is a
// no-op there; it only bites above 64512.
inline void choose_split_pinned(int n_kv_actual, int n_kv_alloc, int n_head_kv, int target_ctas,
                                int & kv_per_split, int & n_splits) {
    if (n_kv_alloc < n_kv_actual) {          // no usable allocation hint: keep the old derivation
        choose_split(n_kv_actual, n_head_kv, target_ctas, kv_per_split, n_splits);
        return;
    }
    int want = target_ctas / (n_head_kv > 0 ? n_head_kv : 1);
    if (want < 1) { want = 1; }
    kv_per_split = (n_kv_alloc + want - 1) / want;
    kv_per_split = ((kv_per_split + TILE - 1) / TILE) * TILE;
    if (kv_per_split < TILE) { kv_per_split = TILE; }
    n_splits = (n_kv_actual + kv_per_split - 1) / kv_per_split;
    if (n_splits < 1) { n_splits = 1; }
}

template <int GQA, int T>
inline cudaError_t launch_t(Params p, cudaStream_t stream) {
    constexpr int C = GQA * T;
    const int smem = Smem<GQA, T>::TOTAL;
    static bool attr_set = false;  // per template instance
    if (!attr_set) {
        cudaError_t e = cudaFuncSetAttribute(fattn_q4_sm60_chunk<GQA, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) { return e; }
        attr_set = true;
    }
    static const bool pvh = getenv("GGML_CUDA_FATTN_SM60_PVHFMA2") != nullptr && atoi(getenv("GGML_CUDA_FATTN_SM60_PVHFMA2")) != 0;
    if (pvh && T == 1) {
        static bool attr_set_pvh = false;
        if (!attr_set_pvh) {
            cudaError_t e = cudaFuncSetAttribute(fattn_q4_sm60_chunk<GQA, T, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            if (e != cudaSuccess) { return e; }
            attr_set_pvh = true;
        }
        fattn_q4_sm60_chunk<GQA, T, true><<<dim3(p.n_splits, p.n_head_kv), NTHREADS, smem, stream>>>(p);
        fattn_q4_sm60_combine<GQA><<<dim3(p.n_head, p.T), D, p.n_splits * sizeof(float2), stream>>>(p);
        return cudaGetLastError();
    }
    fattn_q4_sm60_chunk<GQA, T><<<dim3(p.n_splits, p.n_head_kv), NTHREADS, smem, stream>>>(p);
    fattn_q4_sm60_combine<GQA><<<dim3(p.n_head, p.T), D, p.n_splits * sizeof(float2), stream>>>(p);
    (void) C;
    return cudaGetLastError();
}

// Largest T this build supports: GP100 allows at most 48 KB of shared memory per block (no opt-in above 48 KB
// before Volta). With 128-position tiles Q + P need 1 KB per column, so C = 6 * T <= 24 -> T <= 4.
constexpr int T_SUPPORTED = 4;
static_assert(Smem<6, T_SUPPORTED>::TOTAL <= 48 * 1024, "shared memory exceeds the sm_60 per-block limit");

// Entry point: GQA must be 6 (Qwen3.8-27B: 24 query / 4 KV heads); T in 1..T_SUPPORTED, each with its own
// instance (padding T up would read beyond Q). Returns cudaErrorInvalidValue for unsupported shapes so callers can
// fall back to the generic kernels.
inline cudaError_t launch(Params p, cudaStream_t stream) {
    if (p.n_head_kv <= 0 || p.n_head != 6 * p.n_head_kv) { return cudaErrorInvalidValue; }
    switch (p.T) {
        case 1: return launch_t<6, 1>(p, stream);
        case 2: return launch_t<6, 2>(p, stream);
        case 3: return launch_t<6, 3>(p, stream);
        case 4: return launch_t<6, 4>(p, stream);
        default: return cudaErrorInvalidValue;
    }
}

}  // namespace sm60fa
