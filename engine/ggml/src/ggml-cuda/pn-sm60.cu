// P100 (sm_60) GEMV for the PN signed mid-tread weight types 124 (pn3t) / 125 (pn4t).
//
// Format (tensorless docs/pn-format.md; reference decoder dequantize_row_pn3t/pn4t in ggml-quants.c):
//   one block = 34 B = fp16 signed scale d + 8 little-endian u32 words;
//   pn3t: 10 codes per word (even local index in the low half at bits 3j, odd in the high half at 16+3j,
//         bits 15/31 reserved), 80 weights per block;  pn4t: 8 codes per word (nibbles at 4j / 16+4j), 64 weights.
//   value: w = d * (n - z), z = 4 (pn3t) / 8 (pn4t); both steps are exact in fp32.
//
// Decode GEMV: each lane owns one packed u32 word (10 pn3t / 8 pn4t codes) and its run of
// activation values, so there is no per-element block division and the word is loaded once.
//
// GGML_CUDA_PN_REF=1  -> the fp32 reference path: dequantize the whole weight once per MUL_MAT
//                        into a pooled fp32 scratch buffer, then a coalesced fp32 GEMV over it.
// GGML_CUDA_PN_STATS=1 -> print one census line per distinct (type, kernel, m, k, ncols) dispatch.

#include "mmvq.cuh"
#include "convert.cuh"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_set>

static __device__ __forceinline__ uint32_t pn_ld_word(const uint8_t * GGML_RESTRICT p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

// Same packed word as pn_ld_word, fetched as two 2-byte halves. A block is 34 B, so 34*b + 2 + 4*wi is
// always even: both halves are 2-byte aligned and the bytes are identical, hence bit-identical weights.
static __device__ __forceinline__ uint32_t pn_ld_word16(const uint8_t * GGML_RESTRICT p) {
    const uint16_t lo = *reinterpret_cast<const uint16_t *>(p);
    const uint16_t hi = *reinterpret_cast<const uint16_t *>(p + 2);
    return (uint32_t) lo | ((uint32_t) hi << 16);
}

// The 16-bit half of a packed word that starts at p (2-byte aligned, like pn_ld_word16's halves). Codes never
// straddle the halves, so a consumer decodes each code from its own half and never assembles the word.
static __device__ __forceinline__ uint32_t pn_ld_half(const uint8_t * GGML_RESTRICT p) {
    return (uint32_t) *reinterpret_cast<const uint16_t *>(p);
}

// Cooperative staging copy of n floats (n a multiple of 4) from global to shared memory by the nthr threads of
// the CTA. When both pointers are 16-byte aligned, each thread moves float4s (one 128-bit load and one 128-bit
// store per slot) instead of one scalar per iteration; otherwise it falls back to the scalar loop. Either way
// the values are copied bit for bit. The alignment test depends only on the CTA-uniform base pointers.
static __device__ __forceinline__ void pn_stage_x(
        float * GGML_RESTRICT dst, const float * GGML_RESTRICT src, const int n, const int tid, const int nthr) {
    if (((((uintptr_t) src) | ((uintptr_t) dst)) & 15) != 0) {
        for (int idx = tid; idx < n; idx += nthr) {
            dst[idx] = src[idx];
        }
        return;
    }
    const int nv = n >> 2;
    for (int i = tid; i < nv; i += nthr) {
        reinterpret_cast<float4 *>(dst)[i] = reinterpret_cast<const float4 *>(src)[i];
    }
    for (int idx = (nv << 2) + tid; idx < n; idx += nthr) {
        dst[idx] = src[idx];
    }
}

// one warp per output row, NC activation columns (1..4)
template <int QK, int BITS, int NC>
static __global__ void pn_gemv_kernel(
        const uint8_t * GGML_RESTRICT w, const float * GGML_RESTRICT x, float * GGML_RESTRICT dst,
        const int64_t k, const int64_t nrows) {
    const int64_t row = (int64_t) blockIdx.x*blockDim.y + threadIdx.y;
    if (row >= nrows) {
        return;
    }

    const int     lane = threadIdx.x;
    const int64_t nblk = k / QK;
    const uint8_t * GGML_RESTRICT wr = w + row*nblk*PN_BLOCK_BYTES;

    constexpr int Z        = 1 << (BITS - 1);
    constexpr int MASK     = (1 << BITS) - 1;
    constexpr int PER_WORD = (BITS == 3) ? 10 : 8;   // codes in one u32 word
    constexpr int WORDS    = 8;

    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        acc[c] = 0.0f;
    }

    // Word g holds the strided codes 0..PER_WORD-1 at local positions g%8*PER_WORD + t.
    const int64_t nwords = nblk*WORDS;
    for (int64_t g = lane; g < nwords; g += 32) {
        const int64_t     b   = g >> 3;
        const int         wi  = (int) (g & (WORDS - 1));
        const uint8_t   * blk = wr + b*PN_BLOCK_BYTES;
        const float       d   = __half2float(*(const half *) blk);   // 34*b is even -> 2-byte aligned
        const uint32_t    v   = pn_ld_word(blk + 2 + 4*wi);
        const int64_t     p0  = b*QK + (int64_t) wi*PER_WORD;
#pragma unroll
        for (int t = 0; t < PER_WORD; ++t) {
            const int sh = (BITS == 3) ? 3*(t >> 1) : 4*(t >> 1);
            const int n  = (v >> (sh + ((t & 1) ? 16 : 0))) & MASK;
            const float wv = d * ((float) n - (float) Z);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                acc[c] += wv * x[c*k + p0 + t];
            }
        }
    }

#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc[c] += __shfl_down_sync(0xffffffffu, acc[c], off);
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            dst[c*nrows + row] = acc[c];
        }
    }
}

// T = 1 variant: the activation vector is staged once per CTA into shared memory with coalesced
// warp-linear loads; each row-warp then reads it from smem instead of re-issuing the strided global
// loads of pn_gemv_kernel (lane l reads x at stride PER_WORD). Weights, the fp32 w = d*(n - z)
// arithmetic and the per-lane accumulation order are unchanged, so the result is bit-identical.
// Chunks of CH packed blocks are staged with a double buffer; every warp takes the barrier even when
// its row is out of range, so a partial last CTA cannot desynchronise.
template <int QK, int BITS, int CH, int ROWS, int NWARP>
static __global__ void pn_gemv_smem_kernel(
        const uint8_t * GGML_RESTRICT w, const float * GGML_RESTRICT x, float * GGML_RESTRICT dst,
        const int64_t k, const int64_t nrows) {
    constexpr int Z        = 1 << (BITS - 1);
    constexpr int MASK     = (1 << BITS) - 1;
    constexpr int PER_WORD = (BITS == 3) ? 10 : 8;
    constexpr int WORDS    = 8;

    __shared__ __align__(16) float sx[2][CH*QK];

    const int     lane = threadIdx.x;
    const int     tid  = threadIdx.y*32 + lane;
    const int     nthr = 32*blockDim.y;
    const int64_t row0 = (int64_t) (blockIdx.x*blockDim.y + threadIdx.y)*ROWS;

    const int64_t nblk   = k / QK;
    const int     nchunk = (int) ((nblk + CH - 1)/CH);

    {
        const int lim = (int) (nblk < CH ? nblk : CH);
        pn_stage_x(sx[0], x, lim*QK, tid, nthr);
    }
    __syncthreads();

    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        acc[r] = 0.0f;
    }
    bool active[ROWS];
    const uint8_t * wr[ROWS];
    // Each warp owns a contiguous row triplet (row0 .. row0+ROWS-1) instead of the champion's
    // NWARP-strided interleave. The ROWS rows' packed blocks then sit a uniform stride apart in the
    // lane's own address stream, so the triplet's weight loads at one word g are adjacent in L2/TLB
    // space and issue back-to-back; measured 1.10-1.16x on the pn3t shapes, neutral on pn4t.
    // Every row still accumulates over g ascending then t ascending, so outputs are bit-identical.
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int64_t row = row0 + (int64_t) r;
        active[r] = row < nrows;
        wr[r]     = w + row*nblk*PN_BLOCK_BYTES;
    }

    for (int c = 0; c < nchunk; ++c) {
        if (c + 1 < nchunk) {
            const int64_t rem = nblk - (int64_t) (c + 1)*CH;
            const int     lim = (int) (rem < CH ? rem : CH);
            const float * src = x + (int64_t) (c + 1)*CH*QK;
            float       * buf = sx[(c + 1) & 1];
            pn_stage_x(buf, src, lim*QK, tid, nthr);
        }
        const float * cur = sx[c & 1];
        const int64_t b0  = (int64_t) c*CH;
        int64_t       gend = (b0 + CH)*WORDS;
        if (gend > nblk*WORDS) {
            gend = nblk*WORDS;
        }
        // Rows are interleaved at a fixed word g: every row's own accumulation order (g ascending, then
        // t ascending) is unchanged, so the result is bit-identical to the row-outer loop, but the ROWS
        // weight loads of one g are in flight together instead of one row's stream being serialised.
        for (int64_t g = b0*WORDS + lane; g < gend; g += 32) {
            const int64_t   b   = g >> 3;
            const int       wi  = (int) (g & (WORDS - 1));
            const float   * xp  = cur + (b - b0)*QK + wi*PER_WORD;
            float     d[ROWS];
            uint32_t  v[ROWS];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (active[r]) {
                    const uint8_t * blk = wr[r] + b*PN_BLOCK_BYTES;
                    d[r] = __half2float(*(const half *) blk);
                    v[r] = pn_ld_word16(blk + 2 + 4*wi);
                }
            }
#pragma unroll
            for (int t = 0; t < PER_WORD; ++t) {
                const int sh = (BITS == 3) ? 3*(t >> 1) : 4*(t >> 1);
                const float xt = xp[t];
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    if (active[r]) {
                        const int n = (v[r] >> (sh + ((t & 1) ? 16 : 0))) & MASK;
                        acc[r] += (d[r] * ((float) n - (float) Z)) * xt;
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (active[r]) {
                dst[row0 + (int64_t) r] = acc[r];
            }
        }
    }
}

// Software-pipelined variant of pn_gemv_smem_kernel for the complete-chunk case. With CH=32 every
// lane takes exactly eight word positions per chunk: g = b0*8 + lane + 32*j (j = 0..7). The word index
// wi = lane&7 and the block-in-chunk bl = lane>>3 are lane constants, so the block pointers advance by
// a fixed 4*PN_BLOCK_BYTES and the activation pointer by 4*QK floats; and two positions ahead are
// fetched into a two-slot register buffer before the current position's arithmetic runs, so each lane
// keeps two weight loads in flight instead of stalling on one. Row mapping, staging, arithmetic and the
// accumulation order (g ascending, then t ascending) are unchanged, so the result is bit-identical.
// The possibly-partial last chunk keeps the runtime-clamped traversal of pn_gemv_smem_kernel.
template <int QK, int BITS, int CH, int ROWS, int NWARP>
static __global__ void pn_gemv_smem_pf_kernel(
        const uint8_t * GGML_RESTRICT w, const float * GGML_RESTRICT x, float * GGML_RESTRICT dst,
        const int64_t k, const int64_t nrows) {
    constexpr int Z        = 1 << (BITS - 1);
    constexpr int MASK     = (1 << BITS) - 1;
    constexpr int PER_WORD = (BITS == 3) ? 10 : 8;
    constexpr int WORDS    = 8;
    constexpr int NIT      = CH*WORDS/32;   // word positions per lane in a complete chunk
    constexpr int DEPTH    = 2;             // positions fetched ahead

    __shared__ __align__(16) float sx[2][CH*QK];

    const int     lane = threadIdx.x;
    const int     tid  = threadIdx.y*32 + lane;
    const int     nthr = 32*blockDim.y;
    const int64_t row0 = (int64_t) (blockIdx.x*blockDim.y + threadIdx.y)*ROWS;

    const int64_t nblk   = k / QK;
    const int     nchunk = (int) ((nblk + CH - 1)/CH);

    {
        const int lim = (int) (nblk < CH ? nblk : CH);
        pn_stage_x(sx[0], x, lim*QK, tid, nthr);
    }
    __syncthreads();

    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        acc[r] = 0.0f;
    }
    bool active[ROWS];
    const uint8_t * wr[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int64_t row = row0 + (int64_t) r;
        active[r] = row < nrows;
        wr[r]     = w + row*nblk*PN_BLOCK_BYTES;
    }

    const int wi = lane & (WORDS - 1);
    const int bl = lane >> 3;

    for (int c = 0; c < nchunk; ++c) {
        if (c + 1 < nchunk) {
            const int64_t rem = nblk - (int64_t) (c + 1)*CH;
            const int     lim = (int) (rem < CH ? rem : CH);
            const float * src = x + (int64_t) (c + 1)*CH*QK;
            float       * buf = sx[(c + 1) & 1];
            pn_stage_x(buf, src, lim*QK, tid, nthr);
        }
        const float * cur = sx[c & 1];
        const int64_t b0  = (int64_t) c*CH;
        if ((c + 1)*CH <= nblk) {
            const float * xp = cur + bl*QK + wi*PER_WORD;
            const uint8_t * bp[ROWS];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                bp[r] = wr[r] + (b0 + bl)*PN_BLOCK_BYTES;
            }
            float    d[ROWS][DEPTH];
            uint32_t v[ROWS][DEPTH];
#pragma unroll
            for (int q = 0; q < DEPTH; ++q) {
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    if (active[r]) {
                        const uint8_t * p = bp[r] + q*4*PN_BLOCK_BYTES;
                        d[r][q] = __half2float(*(const half *) p);
                        v[r][q] = pn_ld_word16(p + 2 + 4*wi);
                    }
                }
            }
#pragma unroll
            for (int it = 0; it < NIT; ++it) {
                const float * xts = xp;
                const int slot = it & (DEPTH - 1);
                float    dn[ROWS];
                uint32_t vn[ROWS];
                if (it + DEPTH < NIT) {
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        if (active[r]) {
                            const uint8_t * p = bp[r] + (it + DEPTH)*4*PN_BLOCK_BYTES;
                            dn[r] = __half2float(*(const half *) p);
                            vn[r] = pn_ld_word16(p + 2 + 4*wi);
                        }
                    }
                }
#pragma unroll
                for (int t = 0; t < PER_WORD; ++t) {
                    const int   sh = (BITS == 3) ? 3*(t >> 1) : 4*(t >> 1);
                    const float xt = xts[t];
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        if (active[r]) {
                            const int n = (v[r][slot] >> (sh + ((t & 1) ? 16 : 0))) & MASK;
                            acc[r] += (d[r][slot] * ((float) n - (float) Z)) * xt;
                        }
                    }
                }
                if (it + DEPTH < NIT) {
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        d[r][slot] = dn[r];
                        v[r][slot] = vn[r];
                    }
                }
                xp += 4*QK;
            }
        } else {
            int64_t gend = (b0 + CH)*WORDS;
            if (gend > nblk*WORDS) {
                gend = nblk*WORDS;
            }
            for (int64_t g = b0*WORDS + lane; g < gend; g += 32) {
                const int64_t   b   = g >> 3;
                const int       wj  = (int) (g & (WORDS - 1));
                const float   * xq  = cur + (b - b0)*QK + wj*PER_WORD;
                float     d2[ROWS];
                uint32_t  v2[ROWS];
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    if (active[r]) {
                        const uint8_t * blk = wr[r] + b*PN_BLOCK_BYTES;
                        d2[r] = __half2float(*(const half *) blk);
                        v2[r] = pn_ld_word16(blk + 2 + 4*wj);
                    }
                }
#pragma unroll
                for (int t = 0; t < PER_WORD; ++t) {
                    const int   sh = (BITS == 3) ? 3*(t >> 1) : 4*(t >> 1);
                    const float xt = xq[t];
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        if (active[r]) {
                            const int n = (v2[r] >> (sh + ((t & 1) ? 16 : 0))) & MASK;
                            acc[r] += (d2[r] * ((float) n - (float) Z)) * xt;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (active[r]) {
                dst[row0 + (int64_t) r] = acc[r];
            }
        }
    }
}

// Field placement for the weight-staged T=1 GEMV, read from the two 16-bit halves of the packed word. Code t
// lies in half t&1 (even codes low, odd codes high) at bit p = BITS*(t>>1) of that half, so p <= 23-BITS for
// every code: (half | 0x3F800000) & ((mask << p) | 0x3F800000) is the float 1 + n*2^(p-23) from one LOP, and no
// word is assembled and no field needs a shift.
template <int BITS>
static __device__ __forceinline__ int pn_field_pos(const int t) {
    return BITS*(t >> 1);
}

// 2^(23-p): the activation of code t is stored times this, x' = x * 2^(23-p), so that
// (1 + n*2^(p-23)) * x' - x' = n * x exactly (one fma, no cancellation: the product is exact inside the fma).
template <int BITS>
static __device__ __forceinline__ float pn_xscale(const int t) {
    return __uint_as_float((uint32_t) (150 - pn_field_pos<BITS>(t)) << 23);
}

// 2^(p-23): undoes pn_xscale for the activation sum of the word.
template <int BITS>
static __device__ __forceinline__ float pn_inv_xscale(const int t) {
    return __uint_as_float((uint32_t) (104 + pn_field_pos<BITS>(t)) << 23);
}

// Warp-private weight staging for the T=1 GEMV. Each warp moves the packed bytes of its own ROWS rows through its
// own shared-memory slice and fences with __syncwarp, so the weights need no block barrier. The activation chunks are
// shared by every warp, so they are staged two chunks at a time (one group), double-buffered: one block barrier per
// group. The lane-to-(block, word) mapping, the per-lane accumulation order (block ascending, then word position, then
// code) and the fp32 arithmetic are those of pn_gemv_wstage_kernel, so the output is bit-identical to it. Requires
// nblk % CB == 0 and 16-byte aligned base pointers (checked by the caller).
template <int QK, int BITS, int CB, int ROWS, int NWARP>
static __global__ void pn_gemv_wpriv_kernel(
        const uint8_t * GGML_RESTRICT w, const float * GGML_RESTRICT x, float * GGML_RESTRICT dst,
        const int64_t k, const int64_t nrows) {
    constexpr int Z        = 1 << (BITS - 1);
    constexpr int MASK     = (1 << BITS) - 1;
    constexpr int PER_WORD = (BITS == 3) ? 10 : 8;
    constexpr int WORDS    = 8;
    constexpr int XG       = 2;                      // chunks per activation group: one block barrier per group
    constexpr int NIT      = CB*WORDS/32;            // word positions per lane in a chunk
    constexpr int NROWS    = NWARP*ROWS;             // rows per CTA
    constexpr int NTHR     = 32*NWARP;
    constexpr int CBYTES   = CB*PN_BLOCK_BYTES;      // staged bytes of one row in one chunk
    constexpr int CPIECE   = CBYTES/16;              // 16-byte pieces per row per chunk
    constexpr int NPIECE   = ROWS*CPIECE;            // pieces per warp per chunk
    constexpr int NPC      = (NPIECE + 31)/32;       // pieces per lane per chunk
    constexpr int GCOLS    = XG*CB*QK;               // activations per group
    constexpr int NXV      = GCOLS/4;                // float4 pieces of x per group
    constexpr int NXPC     = (NXV + NTHR - 1)/NTHR;  // float4 pieces per thread
    static_assert(CBYTES % 16 == 0, "a staged row chunk must be a whole number of 16-byte pieces");
    static_assert(CB*QK % PER_WORD == 0, "the activation pattern must repeat every chunk");

    __shared__ __align__(16) float   sx[2][GCOLS];
    __shared__ __align__(16) uint8_t sw[NWARP][ROWS][CBYTES];   // one slice per warp: no block barrier for weights

    const int     lane   = threadIdx.x;
    const int     warp   = threadIdx.y;
    const int     tid    = warp*32 + lane;
    const int64_t cta_r0 = (int64_t) blockIdx.x*NROWS;
    const int64_t nblk   = k / QK;
    const int     nchunk = (int) (nblk / CB);

    // Two register slots for the weights. At step c, slot c&1 is free (chunk c is already in this warp's slice) and
    // takes chunk c+2, while slot (c+1)&1 holds chunk c+1 until the step stores it. So two chunks are in flight.
    uint4  wA[NPC], wB[NPC];
    float4 xr[NXPC];
    // x' scale of the activations of each piece this thread stages, as in pn_gemv_wstage_kernel: the pattern repeats
    // every chunk, so these are fixed for the whole kernel.
    float  xsc[NXPC][2];
#pragma unroll
    for (int j = 0; j < NXPC; ++j) {
        const int p = tid + j*NTHR;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            xsc[j][h] = (p < NXV) ? pn_xscale<BITS>((4*p + 2*h) % PER_WORD) : 0.0f;
        }
    }

    // Staging map, fixed per lane. Piece p = lane + j*32 of the warp is the 16-byte unit p%CPIECE of row p/CPIECE of the
    // warp's rows, and it lands at byte 16*p of the warp's slice. goff[j] is the offset of chunk 0; chunk c adds c*CBYTES.
    // A row past nrows reads the last row instead (its results are never stored). The offsets fit in 32 bits (the caller
    // checks m*nblk*34 < 2^32).
    uint32_t goff[NPC];
    bool     wok[NPC];
#pragma unroll
    for (int j = 0; j < NPC; ++j) {
        const int     p   = lane + j*32;
        const int64_t rr  = cta_r0 + warp*ROWS + p / CPIECE;
        const int64_t row = rr < nrows ? rr : nrows - 1;
        goff[j] = (uint32_t) (row*nblk*PN_BLOCK_BYTES + 16*(p % CPIECE));
        wok[j]  = p < NPIECE;
    }

    // Chunks are loaded in order, so the chunk base is a running pointer rather than a 64-bit multiply of the index.
    const uint8_t * wnext = w;
    // global -> registers for the next chunk of weights
    auto load_w = [&](uint4 (&wr)[NPC]) {
        const uint8_t * wc = wnext;
        wnext += CBYTES;
#pragma unroll
        for (int j = 0; j < NPC; ++j) {
            if (wok[j]) {
                wr[j] = *reinterpret_cast<const uint4 *>(wc + goff[j]);
            }
        }
    };
    // registers -> this warp's slice (the caller fences it with __syncwarp)
    auto store_w = [&](uint4 (&wr)[NPC]) {
#pragma unroll
        for (int j = 0; j < NPC; ++j) {
            if (wok[j]) {
                *reinterpret_cast<uint4 *>(&sw[warp][0][0] + 16*(lane + j*32)) = wr[j];
            }
        }
    };
    // global -> registers for activation group g (a piece past the end of x loads zeros; it is never read)
    auto load_x = [&](const int g) {
        const float * xg = x + (int64_t) g*GCOLS;
#pragma unroll
        for (int j = 0; j < NXPC; ++j) {
            const int p = tid + j*NTHR;
            xr[j] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (p < NXV && (int64_t) g*GCOLS + 4*p < k) {
                xr[j] = reinterpret_cast<const float4 *>(xg)[p];
            }
        }
    };
    // registers -> activation buffer buf, scaled
    auto store_x = [&](const int buf) {
#pragma unroll
        for (int j = 0; j < NXPC; ++j) {
            const int p = tid + j*NTHR;
            if (p < NXV) {
                float4 xv = xr[j];
                xv.x *= xsc[j][0];
                xv.y *= xsc[j][0];
                xv.z *= xsc[j][1];
                xv.w *= xsc[j][1];
                reinterpret_cast<float4 *>(sx[buf])[p] = xv;
            }
        }
    };

    load_w(wA);                                  // chunk 0
    load_x(0);
    if (nchunk > 1) {
        load_w(wB);                              // chunk 1
    }
    store_x(0);
    store_w(wA);
    __syncwarp();
    __syncthreads();

    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        acc[r] = 0.0f;
    }
    const int wi = lane & (WORDS - 1);
    const int bl = lane >> 3;
    // Lane-fixed addressing, as in pn_gemv_wstage_kernel, within this warp's slice.
    const uint8_t * const sw_lane = &sw[warp][0][bl*PN_BLOCK_BYTES];
    const int             wofs    = 2 + 4*wi;         // byte offset of the word inside its block

    // Step c: chunk c is decoded from this warp's slice and from half c&1 of activation buffer buf (the group's),
    // chunk c+2 goes into slot wl (= slot c&1), and chunk c+1 is stored from slot ws (= slot (c+1)&1) once every lane
    // is done with chunk c.
    auto step = [&](const int c, const int buf, uint4 (&wl)[NPC], uint4 (&ws)[NPC]) {
        if (c + 2 < nchunk) {
            load_w(wl);
        }
        const float * xbase = sx[buf] + (c & 1)*CB*QK + bl*QK + wi*PER_WORD;
#pragma unroll
        for (int it = 0; it < NIT; ++it) {
            const float * xp = xbase + it*4*QK;         // block it*4+bl of the chunk; lane = 8*bl + wi as in the pf kernel
            float xv[PER_WORD];                      // prescaled activations of this word, shared by the rows
            float xsum = 0.0f;                       // sum of the unscaled activations of this word
#pragma unroll
            for (int t = 0; t < PER_WORD; t += 2) {  // PER_WORD is even and the word's start is 8-byte aligned
                const float2 x2 = *reinterpret_cast<const float2 *>(xp + t);
                xv[t]     = x2.x;
                xv[t + 1] = x2.y;
            }
#pragma unroll
            for (int t = 0; t < PER_WORD; ++t) {
                xsum = fmaf(xv[t], pn_inv_xscale<BITS>(t), xsum);
            }
            const float zoff = -(float) Z * xsum;    // exact: Z is a power of two
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const uint8_t * blk = sw_lane + r*CBYTES + it*4*PN_BLOCK_BYTES;
                const float    d = __half2float(*reinterpret_cast<const half *>(blk));
                // the two halves of the word are read on their own, as in pn_gemv_wstage_kernel
                uint32_t wl16 = pn_ld_half(blk + wofs) | 0x3F800000u;
                uint32_t wh16 = pn_ld_half(blk + wofs + 2) | 0x3F800000u;
                asm("" : "+r"(wl16), "+r"(wh16));
                float bs = zoff;
#pragma unroll
                for (int t = 0; t < PER_WORD; ++t) {
                    const uint32_t src = (t & 1) ? wh16 : wl16;
                    const uint32_t fm  = ((uint32_t) MASK << pn_field_pos<BITS>(t)) | 0x3F800000u;
                    const float    f   = __uint_as_float(src & fm);
                    bs += fmaf(f, xv[t], -xv[t]);    // (1 + n*2^(p-23)) * x' - x' = n * x
                }
                acc[r] = fmaf(d, bs, acc[r]);
            }
        }
        __syncwarp();                                // every lane is done reading chunk c from the slice
        if (c + 1 < nchunk) {
            store_w(ws);
        }
        __syncwarp();                                // chunk c+1 is visible to the warp
    };

    // one group per iteration: chunks c and c+1 read activation buffer g&1; group g+1 is staged into the other buffer
    // while this group decodes, and the block barrier after it publishes that buffer
    for (int g = 0, c = 0; c < nchunk; ++g, c += XG) {
        const bool next = c + XG < nchunk;       // group g+1 exists
        if (next) {
            load_x(g + 1);
        }
        step(c, g & 1, wA, wB);
        if (c + 1 < nchunk) {
            step(c + 1, g & 1, wB, wA);
        }
        if (next) {
            store_x((g + 1) & 1);
            __syncthreads();
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int64_t row = cta_r0 + warp*ROWS + r;
            if (row < nrows) {
                dst[row] = acc[r];
            }
        }
    }
}

// 16x16-bit multiply-add into a 32-bit accumulator: one XMAD on sm_60, where a 32-bit IMAD is three.
// Both operands are in [-32768, 32767] (the decoded codes and the int16 activations), so the product is exact.
static __device__ __forceinline__ int pn_mad16(const int a, const int b, const int c) {
    int d;
    asm("mad.wide.s16 %0, %1, %2, %3;" : "=r"(d) : "h"((int16_t) a), "h"((int16_t) b), "r"(c));
    return d;
}

// Integer-domain variant of pn_gemv_wpriv_kernel for pn3t at ncols=1 (BITS=3, CB=8, ROWS=3, NWARP=8). The whole
// activation vector is quantized once per CTA to int16 with one scale per vector: xq = round(x * 32767/amax), amax the
// largest |x| of the vector (a prologue reduces it across the CTA). A row-word's codes are decoded as integers:
// bs = sum_t (n_t - 4)*xq_t, exact in int32 and below 2^24 in magnitude, so d*bs is the only rounding of the row sum.
// The vector scale amax/32767 is applied once per row at the end. The weight staging, the lane-to-(block, word)
// mapping and the warp reduction are those of pn_gemv_wpriv_kernel. Dynamic shared memory: the per-warp weight slices
// (as in the fp32 kernel), then one float per warp for the amax reduction, then the k int16 activations.
// The launch bound asks for three CTAs (24 warps) per SM. sm_60 allocates registers per warp in units of 256, so that
// needs at most 80 registers per thread; the five-row build uses exactly 80, with no spills.
// XQ (float4 slots per thread of the activation prologue) is ceil(k/(4*NWARP*32)), so the template is instantiated per
// slot count. At k=5120 that is five slots, not eight: the SASS before the first barrier shrinks from 212 to 119 instructions.
template <int CB, int ROWS, int NWARP, int XQ>
static __global__ void __launch_bounds__(32*NWARP, 3) pn3t_gemv_wint_kernel(
        const uint8_t * GGML_RESTRICT w, const float * GGML_RESTRICT x, float * GGML_RESTRICT dst,
        const int k, const int nrows) {
    constexpr int Z        = 4;                      // 1 << (3 - 1)
    constexpr int PER_WORD = 10;
    constexpr int WORDS    = 8;
    constexpr int NIT      = CB*WORDS/32;            // word positions per lane in a chunk
    constexpr int NROWS    = NWARP*ROWS;             // rows per CTA
    constexpr int NTHR     = 32*NWARP;
    constexpr int CBYTES   = CB*PN_BLOCK_BYTES;      // staged bytes of one row in one chunk
    constexpr int CPIECE   = CBYTES/16;              // 16-byte pieces per row per chunk
    constexpr int NPIECE   = ROWS*CPIECE;            // pieces per warp per chunk
    constexpr int NPC      = (NPIECE + 31)/32;       // pieces per lane per chunk
    constexpr int SW_BYTES = NWARP*ROWS*CBYTES;      // all warps' weight slices
    static_assert(CBYTES % 16 == 0, "a staged row chunk must be a whole number of 16-byte pieces");
    static_assert(CB*QK_PN3T % PER_WORD == 0, "the activation pattern must repeat every chunk");

    extern __shared__ __align__(16) uint8_t pn_wint_smem[];
    uint8_t * const sw   = pn_wint_smem;
    float   * const sred = reinterpret_cast<float *>(pn_wint_smem + SW_BYTES);
    int16_t * const xq   = reinterpret_cast<int16_t *>(pn_wint_smem + SW_BYTES + NWARP*sizeof(float));

    const int     lane   = threadIdx.x;
    const int     warp   = threadIdx.y;
    const int     tid    = warp*32 + lane;
    // 32-bit indices: sm_60 emulates 64-bit multiplies and divides with XMAD sequences, and the caller keeps
    // m*nblk*PN_BLOCK_BYTES below 2^32.
    const int cta_r0 = (int) blockIdx.x*NROWS;
    const int nblk   = k / QK_PN3T;
    const int     nchunk = (int) (nblk / CB);

    uint4 wA[NPC], wB[NPC];

    // Staging map, as in pn_gemv_wpriv_kernel.
    uint32_t goff[NPC];
    bool     wok[NPC];
#pragma unroll
    for (int j = 0; j < NPC; ++j) {
        const int     p   = lane + j*32;
        const int rr  = cta_r0 + warp*ROWS + p / CPIECE;
        const int row = rr < nrows ? rr : nrows - 1;
        goff[j] = (uint32_t) row*nblk*PN_BLOCK_BYTES + 16*(p % CPIECE);
        wok[j]  = p < NPIECE;
    }

    const uint8_t * wnext = w;
    auto load_w = [&](uint4 (&wr)[NPC]) {
        const uint8_t * wc = wnext;
        wnext += CBYTES;
#pragma unroll
        for (int j = 0; j < NPC; ++j) {
            if (wok[j]) {
                wr[j] = *reinterpret_cast<const uint4 *>(wc + goff[j]);
            }
        }
    };
    auto store_w = [&](uint4 (&wr)[NPC]) {
#pragma unroll
        for (int j = 0; j < NPC; ++j) {
            if (wok[j]) {
                *reinterpret_cast<uint4 *>(sw + warp*ROWS*CBYTES + 16*(lane + j*32)) = wr[j];
            }
        }
    };

    // Prologue: the activations are read once into registers (x is 16-byte aligned, k a multiple of 4 and at most
    // 4*XQ*NTHR), ahead of the first two weight chunks so that their loads do not delay the reads of x. XQ is the
    // number of float4 slots per thread that the host sized to k (ceil(k/(4*NTHR))); slots past k/4 are not read.
    const int nv = k/4;
    const float4 * x4 = reinterpret_cast<const float4 *>(x);
    float4 xr[XQ];
#pragma unroll
    for (int j = 0; j < XQ; ++j) {
        const int p = tid + j*NTHR;
        xr[j] = p < nv ? x4[p] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    load_w(wA);
    if (nchunk > 1) {
        load_w(wB);
    }

    // amax of x over the CTA, then the int16 quantization of the same registers.
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < XQ; ++j) {
        const float4 v = xr[j];
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }
    if (lane == 0) {
        sred[warp] = amax;
    }
    __syncthreads();
    float amax_all = 0.0f;
#pragma unroll
    for (int i = 0; i < NWARP; ++i) {
        amax_all = fmaxf(amax_all, sred[i]);
    }
    const float inv = amax_all > 0.0f ? 32767.0f/amax_all : 0.0f;
#pragma unroll
    for (int j = 0; j < XQ; ++j) {
        const int p = tid + j*NTHR;
        const float4 v = xr[j];
        const int a = min(max(__float2int_rn(v.x*inv), -32767), 32767);
        const int b = min(max(__float2int_rn(v.y*inv), -32767), 32767);
        const int c = min(max(__float2int_rn(v.z*inv), -32767), 32767);
        const int d = min(max(__float2int_rn(v.w*inv), -32767), 32767);
        if (p < nv) {
            reinterpret_cast<uint2 *>(xq)[p] = make_uint2((uint32_t) (uint16_t) a | ((uint32_t) (uint16_t) b << 16),
                                                          (uint32_t) (uint16_t) c | ((uint32_t) (uint16_t) d << 16));
        }
    }
    __syncthreads();

    store_w(wA);                                 // chunk 0 into this warp's slice
    __syncwarp();

    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        acc[r] = 0.0f;
    }
    const int wi = lane & (WORDS - 1);
    const int bl = lane >> 3;
    const uint8_t * const sw_lane = sw + warp*ROWS*CBYTES + bl*PN_BLOCK_BYTES;
    const int             wofs    = 2 + 4*wi;         // byte offset of the word inside its block

    // Step c, as in pn_gemv_wpriv_kernel: chunk c is decoded from this warp's slice, chunk c+2 goes into slot wl
    // (= slot c&1), and chunk c+1 is stored from slot ws once every lane is done with chunk c.
    auto step = [&](const int c, uint4 (&wl)[NPC], uint4 (&ws)[NPC]) {
        if (c + 2 < nchunk) {
            load_w(wl);
        }
        const int16_t * xbase = xq + (int) (c*CB + bl)*QK_PN3T + wi*PER_WORD;
#pragma unroll
        for (int it = 0; it < NIT; ++it) {
            const int16_t * xp = xbase + it*4*QK_PN3T;   // block c*CB + it*4 + bl; lane = 8*bl + wi as in the fp32 kernel
            int xv[PER_WORD];                            // the word's activations (the int16 pairs are 4-byte aligned)
            int xsum = 0;
#pragma unroll
            for (int j = 0; j < PER_WORD/2; ++j) {
                const uint32_t v = *reinterpret_cast<const uint32_t *>(xp + 2*j);
                xv[2*j]     = (int) (int16_t) (v & 0xffffu);
                xv[2*j + 1] = (int) v >> 16;
                xsum += xv[2*j] + xv[2*j + 1];
            }
            const int zoff = -Z*xsum;                    // the zero point of the word, exact in int
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const uint8_t * blk = sw_lane + r*CBYTES + it*4*PN_BLOCK_BYTES;
                const float    d = __half2float(*reinterpret_cast<const half *>(blk));
                const uint32_t lo = pn_ld_half(blk + wofs);      // codes 0, 2, ..., 8 of the word (3 bits each)
                const uint32_t hi = pn_ld_half(blk + wofs + 2);  // codes 1, 3, ..., 9
                int bs = zoff;
#pragma unroll
                for (int t = 0; t < PER_WORD; ++t) {
                    const uint32_t src = (t & 1) ? hi : lo;
                    const int      n   = (int) ((src >> (3*(t >> 1))) & 7u);
                    bs = pn_mad16(n, xv[t], bs);
                }
                acc[r] = fmaf(d, (float) bs, acc[r]);
            }
        }
        __syncwarp();                                    // every lane is done reading chunk c from the slice
        if (c + 1 < nchunk) {
            store_w(ws);
        }
        __syncwarp();                                    // chunk c+1 is visible to the warp
    };

    for (int c = 0; c < nchunk; c += 2) {
        step(c, wA, wB);
        if (c + 1 < nchunk) {
            step(c + 1, wB, wA);
        }
    }

    const float sc = amax_all * (1.0f/32767.0f);
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = cta_r0 + warp*ROWS + r;
            if (row < nrows) {
                dst[row] = acc[r]*sc;
            }
        }
    }
}

// fp32 reference GEMV over an already-dequantized row-major weight buffer. Lane l walks p = l, l+32, ...
// so both the weight and the activation reads are coalesced; accumulation stays fp32.
template <int NC>
static __global__ void pn_gemv_f32_kernel(
        const float * GGML_RESTRICT w, const float * GGML_RESTRICT x, float * GGML_RESTRICT dst,
        const int64_t k, const int64_t nrows) {
    const int64_t row = (int64_t) blockIdx.x*blockDim.y + threadIdx.y;
    if (row >= nrows) {
        return;
    }

    const int lane = threadIdx.x;
    const float * GGML_RESTRICT wr = w + row*k;

    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        acc[c] = 0.0f;
    }

    for (int64_t p = lane; p < k; p += 32) {
        const float wv = wr[p];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            acc[c] += wv * x[c*k + p];
        }
    }

#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc[c] += __shfl_down_sync(0xffffffffu, acc[c], off);
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            dst[c*nrows + row] = acc[c];
        }
    }
}

static bool ggml_cuda_pn_ref() {
    static const bool on = getenv("GGML_CUDA_PN_REF") != nullptr;
    return on;
}

static void ggml_cuda_pn_stat(const char * type, const char * kernel, int64_t m, int64_t k, int64_t ncols) {
    static const bool on = getenv("GGML_CUDA_PN_STATS") != nullptr;
    if (!on) {
        return;
    }
    static std::mutex mu;
    static std::unordered_set<std::string> seen;
    char key[128];
    snprintf(key, sizeof(key), "%s %s %lld %lld %lld", type, kernel,
             (long long) m, (long long) k, (long long) ncols);
    std::lock_guard<std::mutex> lock(mu);
    if (seen.insert(key).second) {
        fprintf(stderr, "pn_gemv: type=%s kernel=%s m=%lld k=%lld ncols=%lld\n", type, kernel,
                (long long) m, (long long) k, (long long) ncols);
        fflush(stderr);
    }
}

bool ggml_cuda_pn_sm60_try(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const ggml_type type = src0->type;
    if (type != GGML_TYPE_PN3T && type != GGML_TYPE_PN4T) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 ||
        src1->ne[2] != 1 || src1->ne[3] != 1 ||
        dst ->ne[2] != 1 || dst ->ne[3] != 1) {
        return false; // batched: leave it to the dequant + cuBLAS path
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t k     = src0->ne[0];
    const int64_t m     = src0->ne[1];
    const int64_t ncols = src1->ne[1];

    if (src1->ne[0] != k || dst->ne[0] != m || dst->ne[1] != ncols) {
        return false;
    }
    if (ncols < 1 || ncols > 4) {
        return false; // wider batches: dequantize once + cuBLAS
    }
    if (k % (type == GGML_TYPE_PN3T ? QK_PN3T : QK_PN4T) != 0) {
        return false;
    }

    cudaStream_t      stream = ctx.stream();
    const uint8_t   * w      = (const uint8_t *) src0->data;
    const float     * x      = (const float   *) src1->data;
    float           * out    = (float         *) dst->data;

    const bool     ref  = ggml_cuda_pn_ref();
    const char   * name = (type == GGML_TYPE_PN3T) ? "pn3t" : "pn4t";

    if (ref) {
        ggml_cuda_pn_stat(name, "ref", m, k, ncols);
        const auto to_f32 = ggml_get_to_fp32_cuda(type);
        GGML_ASSERT(to_f32 != nullptr);
        ggml_cuda_pool_alloc<float> src0_f32(ctx.pool());
        src0_f32.alloc((size_t) ggml_nelements(src0));
        to_f32(src0->data, src0_f32.get(), ggml_nelements(src0), stream);

        const int nwarp = 8;
        const dim3 block(32, nwarp);
        const dim3 grid((unsigned) ((m + nwarp - 1)/nwarp));
        switch (ncols) {
            case 1: pn_gemv_f32_kernel<1><<<grid, block, 0, stream>>>(src0_f32.get(), x, out, k, m); break;
            case 2: pn_gemv_f32_kernel<2><<<grid, block, 0, stream>>>(src0_f32.get(), x, out, k, m); break;
            case 3: pn_gemv_f32_kernel<3><<<grid, block, 0, stream>>>(src0_f32.get(), x, out, k, m); break;
            default: pn_gemv_f32_kernel<4><<<grid, block, 0, stream>>>(src0_f32.get(), x, out, k, m); break;
        }
        CUDA_CHECK(cudaGetLastError());
        return true;
    }

    ggml_cuda_pn_stat(name, (type == GGML_TYPE_PN3T) ? "pn3t_warp" : "pn4t_warp", m, k, ncols);

    constexpr int NWARP = 8;    // warps per CTA on the staged T=1 path
    const int  nwarp = 8;
    const dim3 block(32, nwarp);
    const dim3 block1(32, NWARP);

    if (ncols == 1) {
        // Rows per warp on the staged T=1 path, measured on the contiguous-row mapping: pn3t is fastest
        // at 4 rows/warp on the plain traversal (1.15-1.18x over 3) and at 3 on the pipelined one,
        // pn4t at 3 (4 rows loses 6-15%). 5 and 6 rows are slower on both types.
        const dim3 grid1((unsigned) ((m + NWARP*3 - 1)/(NWARP*3)));
        // Staged path: CB=8 blocks per chunk (about 17 KB smem per CTA, so three CTAs per SM). Needs whole
        // chunks (nblk a multiple of 8, which also keeps every row 16-byte aligned) and 16-byte aligned w and x
        // for its 16-byte loads, and a 32-bit byte offset per staged piece (the kernel's staging map).
        // Other shapes keep the pipelined and plain smem paths below.
        const int64_t nblk = k / (type == GGML_TYPE_PN3T ? QK_PN3T : QK_PN4T);
        if (nblk % 8 == 0 && ((((uintptr_t) w) | ((uintptr_t) x)) & 15) == 0 &&
            m*nblk*PN_BLOCK_BYTES < ((int64_t) 1 << 32)) {
            if (type == GGML_TYPE_PN3T) {
                // 3 rows/warp: 17 KB smem and 70 registers per CTA, so three CTAs fit per SM. Same per-row
                // arithmetic, so output is bit-identical.
                constexpr int WROWS3 = 3;
                const dim3 gridw3((unsigned) ((m + NWARP*WROWS3 - 1)/(NWARP*WROWS3)));
                if (k <= 8192) {
                    // Integer-domain GEMV: int16 activations with one scale per vector (see pn3t_gemv_wint_kernel).
                    // Its smem is the weight slices, one float per warp, then the k int16 activations.
                    // Five rows per warp (40 per CTA). Five rows keep 80 registers and 21 KB of smem per CTA, so three
                    // CTAs fit per SM (120 resident rows against 96 at six rows, which fit two). Six rows is 96
                    // registers and 23 KB, two CTAs per SM. Per-row arithmetic is unchanged, so the output is
                    // bit-identical. nvprof per call on the PN3M19 k=5120 shapes, six to five rows: m=6144 45.1 to
                    // 38.2 us, m=10240 52.1 to 53.0, m=12288 63.4 to 62.5, m=17408 86.2 to 83.5.
                    constexpr int WROWS_I = 5;
                    const dim3 gridi((unsigned) ((m + NWARP*WROWS_I - 1)/(NWARP*WROWS_I)));
                    const size_t smem = (size_t) NWARP*WROWS_I*8*PN_BLOCK_BYTES + NWARP*sizeof(float) + 2*(size_t) k;
                    // The kernel's activation prologue is specialized to the slots that hold data: k=5120 is five
                    // float4 slots per thread at 256 threads, not eight. Same quantization, so output is bit-identical.
                    const int nslot = (int) ((k/4 + 32*NWARP - 1)/(32*NWARP));
                    switch (nslot) {
                        case 1: pn3t_gemv_wint_kernel<8, WROWS_I, NWARP, 1><<<gridi, block1, smem, stream>>>(w, x, out, k, m); break;
                        case 2: pn3t_gemv_wint_kernel<8, WROWS_I, NWARP, 2><<<gridi, block1, smem, stream>>>(w, x, out, k, m); break;
                        case 3: pn3t_gemv_wint_kernel<8, WROWS_I, NWARP, 3><<<gridi, block1, smem, stream>>>(w, x, out, k, m); break;
                        case 4: pn3t_gemv_wint_kernel<8, WROWS_I, NWARP, 4><<<gridi, block1, smem, stream>>>(w, x, out, k, m); break;
                        case 5: pn3t_gemv_wint_kernel<8, WROWS_I, NWARP, 5><<<gridi, block1, smem, stream>>>(w, x, out, k, m); break;
                        case 6: pn3t_gemv_wint_kernel<8, WROWS_I, NWARP, 6><<<gridi, block1, smem, stream>>>(w, x, out, k, m); break;
                        case 7: pn3t_gemv_wint_kernel<8, WROWS_I, NWARP, 7><<<gridi, block1, smem, stream>>>(w, x, out, k, m); break;
                        default: pn3t_gemv_wint_kernel<8, WROWS_I, NWARP, 8><<<gridi, block1, smem, stream>>>(w, x, out, k, m); break;
                    }
                } else {
                    pn_gemv_wpriv_kernel<QK_PN3T, 3, 8, WROWS3, NWARP><<<gridw3, block1, 0, stream>>>(w, x, out, k, m);
                }
            } else {
                constexpr int WROWS = 4;   // rows per warp on the staged path
                const dim3 gridw((unsigned) ((m + NWARP*WROWS - 1)/(NWARP*WROWS)));
                pn_gemv_wpriv_kernel<QK_PN4T, 4, 8, WROWS, NWARP><<<gridw, block1, 0, stream>>>(w, x, out, k, m);
            }
        } else if (type == GGML_TYPE_PN3T) {
            // pn3t takes the software-pipelined traversal: CH=32 (eight word positions per lane per
            // chunk) with two positions fetched ahead and 3 rows/warp. In the sm_60 microbench that is
            // 1.08-1.14x over the plain CH=16/4-row traversal on the four PN3M19 pn3t shapes
            // (k=5120, m=6144/10240/12288/17408) and 1.4-1.7x on the small-m ones, bit-identical
            // output. pn4t keeps the plain traversal: the same pipelining measured 0.90-1.04x there,
            // i.e. no consistent win.
            pn_gemv_smem_pf_kernel<QK_PN3T, 3, 32, 3, NWARP><<<grid1, block1, 0, stream>>>(w, x, out, k, m);
        } else {
            pn_gemv_smem_kernel<QK_PN4T, 4, 16, 3, NWARP><<<grid1, block1, 0, stream>>>(w, x, out, k, m);
        }
        CUDA_CHECK(cudaGetLastError());
        return true;
    }

    const dim3 grid((unsigned) ((m + nwarp - 1)/nwarp));
    if (type == GGML_TYPE_PN3T) {
        switch (ncols) {
            case 2: pn_gemv_kernel<QK_PN3T, 3, 2><<<grid, block, 0, stream>>>(w, x, out, k, m); break;
            case 3: pn_gemv_kernel<QK_PN3T, 3, 3><<<grid, block, 0, stream>>>(w, x, out, k, m); break;
            default: pn_gemv_kernel<QK_PN3T, 3, 4><<<grid, block, 0, stream>>>(w, x, out, k, m); break;
        }
    } else {
        switch (ncols) {
            case 2: pn_gemv_kernel<QK_PN4T, 4, 2><<<grid, block, 0, stream>>>(w, x, out, k, m); break;
            case 3: pn_gemv_kernel<QK_PN4T, 4, 3><<<grid, block, 0, stream>>>(w, x, out, k, m); break;
            default: pn_gemv_kernel<QK_PN4T, 4, 4><<<grid, block, 0, stream>>>(w, x, out, k, m); break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
