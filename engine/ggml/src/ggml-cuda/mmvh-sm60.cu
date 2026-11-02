// P100 v2 custom kernel K-B: sm_60 IQ3_XXS GEMV with fp16 activations and shared-memory tables (see mmvh-sm60.cuh
// and the standalone harness in the research repository). Opt-in with GGML_CUDA_MMV_SM60=1 until the v2 quality gate passes.
//
// Replaces, for cc 600 / IQ3_XXS / ncols 1..4, the stock mul_mat_vec_q path (q8_1 activation + emulated dp4a +
// global table reads). The fp16 activation keeps 11 mantissa bits instead of 8 and removes the quantize_q8_1 launch.

#include "mmvq.cuh"
#include "mmvh-sm60.cuh"

#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

static int ggml_cuda_mmvh_sm60_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_CUDA_MMV_SM60");
        return e ? atoi(e) : 0;
    }();
    return mode;
}

static bool ggml_cuda_mmvh_sm60_stats() {
    static const bool on = getenv("GGML_CUDA_MMV_SM60_STATS") != nullptr;
    return on;
}

// IQ3_XXS / IQ3_S weights can be repacked once into a side buffer (see mmvh-sm60.cuh) so the decode GEMV reads
// wide aligned words instead of many two-byte pieces. Opt out with GGML_CUDA_MMV_SM60_REPACK=0. Both buffers do not
// fit together (~3.05 GB + ~2.6 GB vs ~4.2 GB free), so the default repacks IQ3_S (the larger gain); use "xxs" or
// "both" to change that. The buffers are process-lifetime: the model tensors they mirror are static.
struct mmvh_rep_entry { void * ptr; int64_t ne0; int64_t ne1; };
// The rep is a function of the tensor's DATA, so the cache must key on the tensor and not only on
// src0->data: ggml-alloc reuses a dead tensor's address, and two same-shape tensors would then
// share one entry (the X-CACHE-ALIAS hazard, one cache over).
static std::unordered_map<const void *, mmvh_rep_entry> g_mmvh_iq3xxs_rep;
static std::unordered_map<const void *, mmvh_rep_entry> g_mmvh_iq3s_rep;
static std::unordered_map<const void *, mmvh_rep_entry> g_mmvh_q3k_rep;
static std::unordered_map<const void *, mmvh_rep_entry> g_mmvh_q6k_rep;
static std::unordered_map<const void *, mmvh_rep_entry> g_mmvh_dhead_rep;
static size_t g_mmvh_iq3xxs_rep_bytes = 0;

// Opt-in: write the IQ3_XXS repack IN PLACE over the tensor's own (enlarged) storage instead of a side
// buffer, which removes the ~1 GB XXS side allocation. The allocation hook that makes the 100-byte record
// fit lives in ggml-cuda.cu; this is the matching admission flag.
static bool ggml_cuda_mmvh_sm60_inplace_xxs() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_MMV_SM60_INPLACE_XXS");
        return e != nullptr && atoi(e) != 0;
    }();
    return on;
}

// R-37: the IQ3_S twin, removing the ~2.6 GB IQ3_S side allocation. The allocation hook (112 vs 110 bytes
// per 256 weights) lives in ggml-cuda.cu beside the XXS one.
static bool ggml_cuda_mmvh_sm60_inplace_iq3s() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_MMV_SM60_INPLACE_IQ3S");
        return e != nullptr && atoi(e) != 0;
    }();
    return on;
}

// Repack only while the allocation leaves at least 700 MiB free (the residency guard is 512 MiB). The server runs
// one process per cell, so this sees each cell's real free memory - the MTP/DFlash drafter lanes have less of it.
static bool ggml_cuda_mmvh_sm60_repack_mem_ok(size_t bytes) {
    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess) {
        return false;
    }
    return free_mem >= bytes + (700u << 20);
}

// Upper bound on the IQ3_XXS side buffer; the free-memory check above is the real limiter.
static size_t ggml_cuda_mmvh_sm60_repack_xxs_cap() {
    static const size_t cap = [] {
        const char * e = getenv("GGML_CUDA_MMV_SM60_REPACK_XXS_MIB");
        return (size_t) (e ? atoi(e) : 1024) << 20;
    }();
    return cap;
}

static int ggml_cuda_mmvh_sm60_repack_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_CUDA_MMV_SM60_REPACK");
        if (e == nullptr)    { return 3; }  // default: IQ3_S + partial IQ3_XXS
        if (e[0] == '0')     { return 0; }
        if (strcmp(e, "xxs")  == 0) { return 2; }
        if (strcmp(e, "s")    == 0) { return 1; }
        return 3;
    }();
    return mode;
}

// Returns true when the sm_60 IQ3_XXS GEMV handled the call; false keeps the stock path.
const void * ggml_cuda_mmvh_sm60_iq3xxs_rep_lookup(const void * data, int64_t ne0, int64_t ne1, int64_t * row_bytes) {
    auto it = g_mmvh_iq3xxs_rep.find(data);
    if (it == g_mmvh_iq3xxs_rep.end() || it->second.ne0 != ne0 || it->second.ne1 != ne1) {
        return nullptr;
    }
    *row_bytes = (ne0 / sm60mmv::QK) * 100;
    return it->second.ptr;
}

// R-08: a rep is a function of the tensor's DATA, so any write to, clear of or free of the storage it
// was built from must invalidate it. Keyed at the WRITE side because two consumers read these maps -
// the mmvh dispatch and mmvq's lookup - and mmvq's is keyed on address and shape, so a stale entry
// reaches the stock kernel whatever the dispatch decides.
void ggml_cuda_mmvh_sm60_rep_erase(const void * addr, size_t size) {
    if (size == 0) {
        return;
    }
    const uintptr_t lo = (uintptr_t) addr;
    const uintptr_t hi = lo + size;
    auto sweep = [&](std::unordered_map<const void *, mmvh_rep_entry> & m, bool is_xxs, bool no_free = false) {
        for (auto it = m.begin(); it != m.end(); ) {
            const uintptr_t k = (uintptr_t) it->first;
            if (k >= lo && k < hi) {
                // A side buffer owns memory in BOTH maps: IQ3_XXS mallocs one at the non-inplace path and
                // IQ3_S always does, so freeing only for is_xxs leaks every IQ3_S entry. An in-place entry
                // has ptr == key and owns nothing. Only the IQ3_XXS buffer is counted in the byte budget.
                if (it->second.ptr != it->first && !no_free) {
                    cudaFree(it->second.ptr);
                    if (is_xxs) {
                        const size_t bytes = (size_t) it->second.ne1 *
                                             ((size_t) it->second.ne0 / sm60mmv::QK) * 100;
                        g_mmvh_iq3xxs_rep_bytes -= bytes < g_mmvh_iq3xxs_rep_bytes ? bytes : g_mmvh_iq3xxs_rep_bytes;
                    }
                }
                it = m.erase(it);
            } else {
                ++it;
            }
        }
    };
    sweep(g_mmvh_iq3xxs_rep, true);
    sweep(g_mmvh_iq3s_rep, false);
    sweep(g_mmvh_q3k_rep, false);
    // q6k entries are NOT freed on erase: a captured graph may still reference the buffer.
    // The leak is bounded (6 drafter tensors x ~60 MB, rekeyed at most per shape change).
    sweep(g_mmvh_q6k_rep, false, true);
    // the dhead map is keyed on ids->data but its CONTENT is a function of src0's rows too:
    // draft_vocab_refresh (llama-context.cpp) rewrites hot slots of the ids tensor IN PLACE at
    // offset base*4, so an ids key can sit BELOW the written range while its gathered rows come
    // from inside it. Overlap test against the src0 range, not the ids key range. Entries with
    // ne1 == 0 are un-owned views (identity prefix maps); erasing them frees nothing.
    for (auto it = g_mmvh_dhead_rep.begin(); it != g_mmvh_dhead_rep.end(); ) {
        const uintptr_t base = (uintptr_t) it->second.ne1;
        const bool owned = it->second.ne1 != 0;
        const bool ids_hit = (uintptr_t) it->first >= lo && (uintptr_t) it->first < hi;
        const bool src_hit = owned && base >= lo && base < hi;
        if (ids_hit || src_hit) {
            if (owned && it->second.ptr != it->first) { cudaFree(it->second.ptr); }
            it = g_mmvh_dhead_rep.erase(it);
        } else {
            ++it;
        }
    }
}

const void * ggml_cuda_mmvh_sm60_iq3s_rep_lookup(const void * data, int64_t ne0, int64_t ne1, int64_t * row_bytes) {
    auto it = g_mmvh_iq3s_rep.find(data);
    if (it == g_mmvh_iq3s_rep.end() || it->second.ne0 != ne0 || it->second.ne1 != ne1) {
        return nullptr;
    }
    *row_bytes = (ne0 / sm60mmv::QK) * sm60mmv::SB_BYTES_S_REP;
    return it->second.ptr;
}


// R-38 / S10-Q4K: the qr0n 0007 geometry for Q4_K at ncols == 1 with many rows - the 248320x5120 output
// head, whose stock mul_mat_vec_q call is 2035 us. The weights decode as fp16 subnormals straight from the
// nibble (no magic subtraction) and x is prescaled by a power of two, so the dot is one __hfma2 chain.
// Opt in with GGML_CUDA_MMV_SM60_Q4K_HEAD=1. Dispatch is deliberately narrow: ncols == 1 and many rows
// only, because the geometry's advantage comes from filling the machine and it loses on narrow shapes.
namespace sm60q4k {

static bool q4k_head_on() {
    static const bool on = getenv("GGML_CUDA_MMV_SM60_Q4K_HEAD") != nullptr &&
                           atoi(getenv("GGML_CUDA_MMV_SM60_Q4K_HEAD")) != 0;
    return on;
}

// The small-shape extension (m in [1024, 65536)) was REJECTED by its calibrated D1 gate
// (q4k-small-code-32k-20261002T112036Z: cand-vs-ref dnext_p99 0.1207 > envelope 0.1120, the other five
// checks pass). The head kernel's fp16-subnormal class sits closer to the tile anchor than to stock
// mmvq, but the pre-registered tail check fails; do not re-widen the dispatch without a lower-error
// variant (attn_out's standalone rel err 4.3e-6 is the likely tail driver).

__device__ __forceinline__ void q4k_gsm4r(int j, unsigned hy, unsigned hz, unsigned hw, int & d, int & m) {
    const int k = j & 3;
    const unsigned by = (hy >> (8*k)) & 0xFFu;   // q[j]   for j < 4,  q[j-4] for j >= 4
    const unsigned bz = (hz >> (8*k)) & 0xFFu;   // q[j+4] for j < 4,  q[j]   for j >= 4
    const unsigned bw = (hw >> (8*k)) & 0xFFu;   // q[j+4] for j >= 4
    if (j < 4) { d = (int) (by & 63u);                       m = (int) (bz & 63u); }
    else       { d = (int) ((bw & 15u) | ((by >> 6) << 4));  m = (int) ((bw >> 4) | ((bz >> 6) << 4)); }
}

__global__ void q4k_prep_x(const float * __restrict__ x, __half * __restrict__ y,
                          float2 * __restrict__ ys, const int nblk) {
    // one warp per 32-x-block. Bit-exact to the old per-thread version: the fmaxf butterfly is exact
    // in any order, each half is per-element, and sum replays the original sequential 32-add fp32
    // chain via __shfl_sync pulls (memcmp-gated byte-exact at K=5120/6144/17408, kernels/kq-nc).
    const int b = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (b >= nblk) return;
    const int lane = threadIdx.x & 31;
    const float v = x[(size_t) b * 32 + lane];
    float mx = fabsf(v);
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    int e = 0; while (e < 30 && ldexpf(1.f, e) < mx) e++;      // max|x| in [2^(e-1), 2^e)
    const int k = 15 - e;
    const float s = ldexpf(1.f, k);
    y[(size_t) b * 32 + lane] = __float2half(v * s);
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < 32; i++) sum += __shfl_sync(0xffffffffu, v, i);
    if (lane == 0) ys[b] = make_float2(ldexpf(1.f, 24 - k), sum);
}

template <int R, int NC>
__global__ void head_kernel(const uint8_t * __restrict__ w, const __half * __restrict__ y,
                        const float2 * __restrict__ ys, float * __restrict__ out,
                        const int nrows, const int nblk_x, const int ldy, const int lds) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int t = lane & 7, jj = t >> 1, hf = t & 1;
    const int g = lane >> 3;
    const int row0 = (blockIdx.x * 4 + warp) * R;
    float acc[NC][R];
#pragma unroll
    for (int c = 0; c < NC; c++)
#pragma unroll
        for (int r = 0; r < R; r++) acc[c][r] = 0.f;
#pragma unroll 1
    for (int sb = g; sb < nblk_x; sb += 4) {
        const __half * yb0 = y + sb*256 + 64*jj + 16*hf;
        uint4 yA[NC], yB[NC], yC[NC], yD[NC];
        float4 ysq[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const half2 * yp = (const half2 *) (yb0 + (size_t) c * ldy);
            yA[c] = __ldg((const uint4 *) yp + 0);
            yB[c] = __ldg((const uint4 *) yp + 1);
            yC[c] = __ldg((const uint4 *) yp + 4);
            yD[c] = __ldg((const uint4 *) yp + 5);
            ysq[c] = __ldg((const float4 *) (ys + c*lds + sb*8 + 2*jj));
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int row = row0 + r;
            if (row >= nrows) break;
            const uint8_t * rec = w + ((size_t) row * nblk_x + sb) * 144;
            const uint4 hd = __ldg((const uint4 *) (rec + 0));
            const float d    = __half2float(*(const __half *) &hd.x);
            const float dmin = __half2float(*(((const __half *) &hd.x) + 1));
            int sc_lo, m_lo, sc_hi, m_hi;
            q4k_gsm4r(2*jj,     hd.y, hd.z, hd.w, sc_lo, m_lo);
            q4k_gsm4r(2*jj + 1, hd.y, hd.z, hd.w, sc_hi, m_hi);
            const uint4 qw = __ldg((const uint4 *) (rec + 16 + 32*jj + 16*hf));
            const unsigned q[4] = {qw.x, qw.y, qw.z, qw.w};
            half2 lo[NC], hi[NC];
#pragma unroll
            for (int c = 0; c < NC; c++) { lo[c] = __float2half2_rn(0.f); hi[c] = __float2half2_rn(0.f); }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const unsigned vlo = q[i] & 0x0F0F0F0Fu;
                const unsigned vhi = (q[i] >> 4) & 0x0F0F0F0Fu;
                const unsigned l01 = __byte_perm(vlo, 0, 0x4140);
                const unsigned l23 = __byte_perm(vlo, 0, 0x4342);
                const unsigned h01 = __byte_perm(vhi, 0, 0x4140);
                const unsigned h23 = __byte_perm(vhi, 0, 0x4342);
                const half2 wl2 = *reinterpret_cast<const half2 *>(&l01);
                const half2 wl3 = *reinterpret_cast<const half2 *>(&l23);
                const half2 wh2 = *reinterpret_cast<const half2 *>(&h01);
                const half2 wh3 = *reinterpret_cast<const half2 *>(&h23);
                const bool odd = (i & 1) != 0;
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    const uint4 sa  = (i < 2) ? yA[c] : yB[c];
                    const uint4 sc4 = (i < 2) ? yC[c] : yD[c];
                    const unsigned w0 = odd ? sa.z : sa.x, w1 = odd ? sa.w : sa.y;
                    const unsigned z0 = odd ? sc4.z : sc4.x, z1 = odd ? sc4.w : sc4.y;
                    const half2 ya_lo = *reinterpret_cast<const half2 *>(&w0);
                    const half2 ya_hi = *reinterpret_cast<const half2 *>(&w1);
                    const half2 yb_lo = *reinterpret_cast<const half2 *>(&z0);
                    const half2 yb_hi = *reinterpret_cast<const half2 *>(&z1);
                    lo[c] = __hfma2(wl2, ya_lo, lo[c]);
                    lo[c] = __hfma2(wl3, ya_hi, lo[c]);
                    hi[c] = __hfma2(wh2, yb_lo, hi[c]);
                    hi[c] = __hfma2(wh3, yb_hi, hi[c]);
                }
            }
#pragma unroll
            for (int c = 0; c < NC; c++) {
                const float sl = __low2float(lo[c]) + __high2float(lo[c]);
                const float sh = __low2float(hi[c]) + __high2float(hi[c]);
                acc[c][r] += d * (float) sc_lo * sl * ysq[c].x - (hf == 0 ? dmin * (float) m_lo * ysq[c].y : 0.f)
                           + d * (float) sc_hi * sh * ysq[c].z - (hf == 0 ? dmin * (float) m_hi * ysq[c].w : 0.f);
            }
        }
    }
    // the fold order is fixed and contains no NC: per-column equality is structural
#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
            float v = acc[c][r];
            v += __shfl_down_sync(0xffffffffu, v, 4);
            v += __shfl_down_sync(0xffffffffu, v, 2);
            v += __shfl_down_sync(0xffffffffu, v, 1);
            v += __shfl_down_sync(0xffffffffu, v, 8);
            v += __shfl_down_sync(0xffffffffu, v, 16);
            if (lane == 0 && row0 + r < nrows) out[row0 + r + (size_t) c * nrows] = v;
        }
    }
}

// validate, allocate the prescale buffer, and launch. false means "not mine", so stock runs.
static bool try_head(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t k = src0->ne[0], m = src0->ne[1];
    if (k % 256 != 0) return false;
    if (src0->nb[0] != ggml_type_size(src0->type) || (size_t) src0->nb[1] != ggml_row_size(src0->type, k)) return false;
    if (src1->ne[0] != k || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (src1->nb[0] != sizeof(float) || (size_t) src1->nb[1] != (size_t) k*sizeof(float)) return false;
    if (!ggml_is_contiguous(dst) || dst->ne[0] != m || dst->ne[1] != src1->ne[1]) return false;

    const int     nc      = (int) src1->ne[1];
    const int     nblk_x  = (int) (k / 256);
    const int64_t y_bytes = k * (int64_t) sizeof(__half);
    const int64_t s_bytes = (k / 32) * (int64_t) sizeof(float2);
    // the buffer holds FOUR columns' worth, laid out as all the y first and then all the ys, so the ys base
    // is g_buf + 4*y_bytes and NOT g_buf + y_bytes. With the single-column offset, column 1's y (written at
    // bytes 10240..20480 for k=5120) landed on top of both columns' ys at 10240..12800, so every verify read
    // garbage scales: the MTP lane emitted random tokens and rejected every draft.
    const size_t  bytes   = (size_t) 4 * (size_t) (y_bytes + s_bytes);
    (void) nc;

    // process-lifetime scratch: the head is one tensor, so one buffer is enough, and it must not be freed
    // while a queued launch still reads it.
    static char * g_buf = nullptr;
    static size_t g_cap = 0;
    if (g_cap < bytes) {
        if (g_buf != nullptr) { cudaFree(g_buf); }
        g_buf = nullptr; g_cap = 0;
        if (cudaMalloc((void **) &g_buf, bytes) != cudaSuccess) { g_buf = nullptr; return false; }
        g_cap = bytes;
    }
    __half * yh = (__half *) g_buf;
    float2 * ys = (float2 *) (g_buf + 4*y_bytes);   // past all FOUR columns' y, matching the sizing above
    cudaStream_t stream = ctx.stream();

    const int     nb = (int) (k / 32);
    const int64_t ld = (int64_t) (src1->nb[1] / sizeof(float));
    // prep cache: sibling GEMVs share one src1 (attn_k then attn_v on the same normed hidden), so
    // re-preparing per call doubles the cost. Single-entry, keyed like the iq3 x-cache: a hit means
    // g_buf currently holds THIS (tensor, data, k, nc, epoch)'s prep, so skipping is safe.
    static const void * g_p_tensor = nullptr;
    static const void * g_p_src = nullptr;
    static int64_t g_p_k = 0, g_p_nc = 0;
    static int g_p_epoch = -1;
    if (!(g_p_tensor == src1 && g_p_src == src1->data && g_p_k == k && g_p_nc == nc && g_p_epoch == ctx.graph_epoch)) {
        for (int c = 0; c < nc; c++) {
            q4k_prep_x<<<(nb + 3) / 4, 128, 0, stream>>>((const float *) src1->data + (size_t) c * ld,
                                                         yh + (size_t) c * k, ys + (size_t) c * (k / 32), nb);
        }
        g_p_tensor = src1; g_p_src = src1->data; g_p_k = k; g_p_nc = nc; g_p_epoch = ctx.graph_epoch;
    }
    // nc == 1 keeps R=8 (REG 63, 8 CTAs/SM) - that path is the kept arm and is validated. For nc >= 2 the
    // accumulators are per column, so R=8 pushed REG to 96 and cost occupancy (5 CTAs/SM); R=4 halves the
    // live accumulators and should buy CTAs back. R only decides which rows a warp owns, never a row's K
    // order, so every column is bit-identical to its nc==1 value at any R.
    constexpr int R1 = 8;
    constexpr int R2 = 4;
    const int     ldy = (int) k, lds = (int) (k / 32);
    const int     g1  = (int) ((m + 4*R1 - 1) / (4*R1));
    const int     g2  = (int) ((m + 4*R2 - 1) / (4*R2));
    uint8_t * W = (uint8_t *) src0->data;
    float   * O = (float *) dst->data;
    if      (nc == 1) head_kernel<R1,1><<<g1, 128, 0, stream>>>(W, yh, ys, O, (int) m, nblk_x, ldy, lds);
    else if (nc == 2) head_kernel<R2,2><<<g2, 128, 0, stream>>>(W, yh, ys, O, (int) m, nblk_x, ldy, lds);
    else if (nc == 3) head_kernel<R2,3><<<g2, 128, 0, stream>>>(W, yh, ys, O, (int) m, nblk_x, ldy, lds);
    else              head_kernel<R2,4><<<g2, 128, 0, stream>>>(W, yh, ys, O, (int) m, nblk_x, ldy, lds);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace sm60q4k

// R-38 second arm: the same 0007 geometry on Q3_K, whose six A1 GEMV tensors (3x ssm_out [6144,5120],
// 2x ffn_up [5120,17408], 1x ffn_down [17408,5120], 155 MB/token) run stock mul_mat_vec_q at ~163 GB/s.
// Stock Q3_K records are 110 bytes, which cannot give aligned uint4 loads, so the weights are repacked
// once into a side buffer of 112-byte records: same field offsets plus 2 pad bytes, a pure block-wise
// memcpy (lossless), every stock reader untouched. The weight decodes as the fp16 subnormal n*2^-24
// (n = 2-bit code + 4*hmask bit) against a per-32-block power-of-two prescaled activation. Changes
// numerics: gated by the calibrated D1 gate like the head arm. Opt in with GGML_CUDA_MMV_SM60_Q3K=1.
namespace sm60q3k {

static bool q3k_on() {
    static const bool on = getenv("GGML_CUDA_MMV_SM60_Q3K") != nullptr &&
                           atoi(getenv("GGML_CUDA_MMV_SM60_Q3K")) != 0;
    return on;
}

__global__ void q3k_repack(const uint8_t * __restrict__ src, uint8_t * __restrict__ dst, const int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    memcpy(dst + (size_t) b * 112, src + (size_t) b * 110, 110);
}

__global__ void q3k_prep_x(const float * __restrict__ x, __half * __restrict__ y,
                           float4 * __restrict__ ys, const int nblk32) {
    // one warp per 32-x-block: 32 prescaled halves plus (2^(24-k), 4*sum(raw lo 16), 4*sum(raw hi 16), 0).
    // Same math as the per-thread version (warp max, half rounding, fp32 sequential sums), but a warp
    // per block spreads the 544-block grid over the SMs instead of 5 blocks of 128 threads.
    const int b = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (b >= nblk32) return;
    const int lane = threadIdx.x & 31;
    const float v = x[(size_t) b * 32 + lane];
    float mx = fabsf(v);
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    int e = 0; while (e < 30 && ldexpf(1.f, e) < mx) e++;
    const int k = 15 - e;
    const __half h = __float2half(v * ldexpf(1.f, k));
    y[(size_t) b * 32 + lane] = h;
    float s16 = __half2float(h);
#pragma unroll
    for (int o = 8; o; o >>= 1) s16 += __shfl_xor_sync(0xffffffffu, s16, o, 16);   // within each 16-lane half
    const float slo = __shfl_sync(0xffffffffu, s16, 0);
    const float shi = __shfl_sync(0xffffffffu, s16, 16);
    if (lane == 0) {
        const float c = 4.f * ldexpf(1.f, -k);
        ys[b] = make_float4(ldexpf(1.f, 24 - k), c * slo, c * shi, 0.f);
    }
}

// Q3_K weight (h,j,sub,l) at y position 128h + 32j + 16sub + l:
//   code = ((q[l + 16sub + 32h] >> 2j) & 3) - ((hm[l + 16sub] & (1 << (4h+j))) ? 0 : 4)
//   value = d * (sc_shuffle(scales)[8h + 2j + sub] - 32) * code, sc_shuffle as in ggml-quants.c.
// Lane t owns sub-blocks isc = t (h=0) and t+8 (h=1), x chunks [16t,16t+16) and [128+16t,144+16t).
template <int R, int NC>
__global__ void q3k_head(const uint8_t * __restrict__ w, const __half * __restrict__ y,
                         const float4 * __restrict__ ys, float * __restrict__ out,
                         const int nrows, const int nblk_x, const int ldy, const int lds) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int t = lane & 7, g = lane >> 3;
    const int sub = t & 1, shift = 2 * (t >> 1);
    const int qoff = 16 * sub, hoff = 16 * sub;
    const int row0 = (blockIdx.x * 4 + warp) * R;
    float acc[NC][R];
#pragma unroll
    for (int c = 0; c < NC; c++)
#pragma unroll
        for (int r = 0; r < R; r++) acc[c][r] = 0.f;
#pragma unroll 1
    for (int sb = g; sb < nblk_x; sb += 4) {
        float4 ysq[2][NC];
        const half * xbase[2][NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            ysq[0][c] = __ldg(ys + (size_t) c * lds + sb * 8 + (t >> 1));
            ysq[1][c] = __ldg(ys + (size_t) c * lds + sb * 8 + 4 + (t >> 1));
            xbase[0][c] = y + (size_t) c * ldy + sb * 256 + 16 * t;
            xbase[1][c] = xbase[0][c] + 128;
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int row = row0 + r;
            if (row >= nrows) break;
            const uint8_t * rec = w + ((size_t) row * nblk_x + sb) * 112;
            const float d3 = __half2float(*(const __half *) (rec + 108));
            const uint4 sraw = __ldg((const uint4 *) (rec + 96));
            uint32_t a0 = sraw.x, a1 = sraw.y, a2s = sraw.z;
            const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
            const uint32_t tmp = a2s;
            a2s = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            a0 = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
            a1 = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
            const uint32_t auxs[4] = {a0, a1, a2s, (uint32_t) (((sraw.y >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4))};
            const uint4 hv0 = __ldg((const uint4 *) (rec + hoff));   // hmask bytes depend on sub only
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int isc = t + 8 * h;
                const int p = 4 * h + (t >> 1);
                const uint4 qv = __ldg((const uint4 *) (rec + 32 + 32 * h + qoff));
                const uint4 hv = hv0;
                const float sc = (float) ((int) (int8_t) ((auxs[isc >> 2] >> (8 * (isc & 3))) & 0xFFu) - 32);
                const unsigned vu[4] = {qv.x, qv.y, qv.z, qv.w};
                const unsigned hu[4] = {hv.x, hv.y, hv.z, hv.w};
                half2 a2[NC];
#pragma unroll
                for (int c = 0; c < NC; c++) a2[c] = __float2half2_rn(0.f);
                const half2 * xp[NC];
#pragma unroll
                for (int c = 0; c < NC; c++) xp[c] = (const half2 *) xbase[h][c];
#pragma unroll
                for (int wq = 0; wq < 4; ++wq) {
                    const unsigned nu = ((vu[wq] >> shift) & 0x03030303u) | (((hu[wq] >> p) & 0x01010101u) << 2);
                    const unsigned p01 = __byte_perm(nu, 0, 0x4140);   // half2 {n_4wq, n_4wq+1} * 2^-24
                    const unsigned p23 = __byte_perm(nu, 0, 0x4342);
                    const half2 n01 = *reinterpret_cast<const half2 *>(&p01);
                    const half2 n23 = *reinterpret_cast<const half2 *>(&p23);
#pragma unroll
                    for (int c = 0; c < NC; c++) {
                        a2[c] = __hfma2(n01, xp[c][2 * wq + 0], a2[c]);
                        a2[c] = __hfma2(n23, xp[c][2 * wq + 1], a2[c]);
                    }
                }
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    const float hf = __low2float(a2[c]) + __high2float(a2[c]);
                    const float4 s4 = ysq[h][c];
                    const float c16 = (sub == 0) ? s4.y : s4.z;
                    acc[c][r] += d3 * sc * s4.x * hf - d3 * sc * c16;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
            float v = acc[c][r];
            v += __shfl_down_sync(0xffffffffu, v, 4);
            v += __shfl_down_sync(0xffffffffu, v, 2);
            v += __shfl_down_sync(0xffffffffu, v, 1);
            v += __shfl_down_sync(0xffffffffu, v, 8);
            v += __shfl_down_sync(0xffffffffu, v, 16);
            if (lane == 0 && row0 + r < nrows) out[row0 + r + (size_t) c * nrows] = v;
        }
    }
}

static bool try_q3k(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t k = src0->ne[0], m = src0->ne[1];
    if (k % 256 != 0) return false;
    if (src0->nb[0] != ggml_type_size(src0->type) || (size_t) src0->nb[1] != ggml_row_size(src0->type, k)) return false;
    if (src1->ne[0] != k || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (src1->nb[0] != sizeof(float) || (size_t) src1->nb[1] != (size_t) k*sizeof(float)) return false;
    if (!ggml_is_contiguous(dst) || dst->ne[0] != m || dst->ne[1] != src1->ne[1]) return false;

    const int     nc      = (int) src1->ne[1];
    const int     nblk_x  = (int) (k / 256);
    const int64_t y_bytes = k * (int64_t) sizeof(__half);
    const int64_t s_bytes = (k / 32) * (int64_t) sizeof(float4);
    const size_t  bytes   = (size_t) 4 * (size_t) (y_bytes + s_bytes);

    // sized once for the largest shape seen; grown WITHOUT freeing (a captured graph may
    // still reference the old block - the #177 lesson on the rep side)
    static char * g_buf = nullptr;
    static size_t g_cap = 0;
    if (g_cap < bytes) {
        void * p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return false;
        g_buf = (char *) p; g_cap = bytes;
    }
    __half * yh = (__half *) g_buf;
    float4 * ys = (float4 *) (g_buf + 4*y_bytes);
    cudaStream_t stream = ctx.stream();

    // side rep: 112-byte records keyed like the IQ3 maps (write-side erase covers invalidation)
    uint8_t * W = (uint8_t *) src0->data;
    auto it = g_mmvh_q3k_rep.find(src0->data);
    if (it == g_mmvh_q3k_rep.end() || it->second.ne0 != k || it->second.ne1 != m) {
        const int64_t nblocks = m * nblk_x;
        const size_t  rbytes  = (size_t) nblocks * 112;
        if (src0->buffer == nullptr ||
            ggml_backend_buffer_get_usage(src0->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
            !ggml_cuda_mmvh_sm60_repack_mem_ok(rbytes)) {
            return false;
        }
        void * p = nullptr;
        if (cudaMalloc(&p, rbytes) != cudaSuccess) return false;
        q3k_repack<<<(int) ((nblocks + 255) / 256), 256, 0, stream>>>((const uint8_t *) src0->data, (uint8_t *) p, (int) nblocks);
        if (cudaGetLastError() != cudaSuccess) { cudaFree(p); return false; }
        g_mmvh_q3k_rep[src0->data] = { p, k, m };
        W = (uint8_t *) p;
    } else {
        W = (uint8_t *) it->second.ptr;
    }

    const int     nb  = (int) (k / 32);
    const int64_t ld  = (int64_t) (src1->nb[1] / sizeof(float));
    for (int c = 0; c < nc; c++) {
        q3k_prep_x<<<(nb + 3) / 4, 128, 0, stream>>>((const float *) src1->data + (size_t) c * ld,
                                                     yh + (size_t) c * k, ys + (size_t) c * (k / 32), nb);
    }
    // R=4 at every real shape: the standalone sweep read 65.4/177.6/167.9 us vs R=8's 92.7/190.5/253.6.
    constexpr int R = 4;
    const int     ldy = (int) k, lds = (int) (k / 32);
    const int     gr  = (int) ((m + 4*R - 1) / (4*R));
    float   * O = (float *) dst->data;
    if      (nc == 1) q3k_head<R,1><<<gr, 128, 0, stream>>>(W, yh, ys, O, (int) m, nblk_x, ldy, lds);
    else if (nc == 2) q3k_head<R,2><<<gr, 128, 0, stream>>>(W, yh, ys, O, (int) m, nblk_x, ldy, lds);
    else if (nc == 3) q3k_head<R,3><<<gr, 128, 0, stream>>>(W, yh, ys, O, (int) m, nblk_x, ldy, lds);
    else              q3k_head<R,4><<<gr, 128, 0, stream>>>(W, yh, ys, O, (int) m, nblk_x, ldy, lds);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace sm60q3k

// Lever 2: Q6_K via the 0007 geometry for the drafter's six n=1/n=2 GEMVs (blk.64, 339.7 MB/cycle
// at 234.6 GB/s stock, measured per shape in dhead-proof3). Standalone (kernels/kq-q6k) at the real
// shapes: blended 1.52x with R=8 (per-shape 1.43-1.65x), n1-vs-n2 BITWISE per column, ref checked
// against ggml's own dequantize_row_q6_K. Stock Q6_K records are 210 bytes (ql@0, qh@128, scales@192,
// d fp16 @208), which cannot give aligned uint4 loads, so the weights are repacked once into
// 224-byte records (14 pad bytes, same field offsets, a pure block-wise copy).
// The 6-bit code q decodes directly as the fp16 subnormal q*2^-24 (no byte_perm shift needed); x is
// prescaled per 16-chunk; the fold d*sc*(e*A - t) is exact fp32. A1 carries q6_K only in blk.64, so
// this path cannot touch the main model. Changes numerics: gated by the calibrated D1 gate.
// Opt in with GGML_CUDA_MMV_SM60_Q6K=1.
namespace sm60q6k {

static bool q6k_on() {
    static const bool on = getenv("GGML_CUDA_MMV_SM60_Q6K") != nullptr &&
                           atoi(getenv("GGML_CUDA_MMV_SM60_Q6K")) != 0;
    return on;
}

__global__ void q6k_repack(const uint8_t * __restrict__ src, uint8_t * __restrict__ dst, const int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    memcpy(dst + (size_t) b * 224, src + (size_t) b * 210, 210);
}

__global__ void q6k_prep_x(const float * __restrict__ x, __half * __restrict__ y,
                           float4 * __restrict__ ys2, const int nblk16) {
    // one warp per 16-x-chunk: 16 prescaled halves plus (2^(24-k), 32*2^-k*sum(half(x*2^k)), 0, 0)
    const int b = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (b >= nblk16) return;
    const int lane = threadIdx.x & 31;
    float v = 0.f;
    if (lane < 16) v = x[(size_t) b * 16 + lane];
    float mx = fabsf(v);
#pragma unroll
    for (int o = 8; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    int e = 0; while (e < 30 && ldexpf(1.f, e) < mx) e++;
    const int k = 15 - e;
    const __half h = lane < 16 ? __float2half(v * ldexpf(1.f, k)) : __float2half(0.f);
    if (lane < 16) y[(size_t) b * 16 + lane] = h;
    float s = __half2float(h);
#pragma unroll
    for (int o = 8; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o, 16);
    if (lane == 0) ys2[b] = make_float4(ldexpf(1.f, 24 - k), 32.f * ldexpf(1.f, -k) * s, 0.f, 0.f);
}

// Weight w, sub-block j = 8h + t (y[128h + 16t + m], scale scales[j]): codes come from 16
// consecutive ql bytes at h*64 + 16*(t%4) + m (LOW nibble for t<4, HIGH for t>=4) and 16
// consecutive qh bytes at h*32 + 16*(t%2) + m, all with shift 2*(t/2) (verified by emulation
// against dequantize_row_q6_K). The fold order (wq loop, shfl 4,2,1,8,16) contains no NC.
template <int R, int NC>
__global__ void q6k_head(const uint8_t * __restrict__ w, const __half * __restrict__ y,
                         const float4 * __restrict__ ys2, float * __restrict__ out,
                         const int nrows, const int nblk_x, const int ldy, const int lds) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int t = lane & 7, g = lane >> 3;
    const int subn = t >> 2;
    const int shift2 = 2 * (t >> 1);
    const int row0 = (blockIdx.x * 4 + warp) * R;
    float acc[NC][R];
#pragma unroll
    for (int c = 0; c < NC; c++)
#pragma unroll
        for (int r = 0; r < R; r++) acc[c][r] = 0.f;
#pragma unroll 1
    for (int sb = g; sb < nblk_x; sb += 4) {
        float4 et[2][NC];
        const half * xbase[2][NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            et[0][c] = __ldg(ys2 + (size_t) c * lds + sb * 16 + t);
            et[1][c] = __ldg(ys2 + (size_t) c * lds + sb * 16 + 8 + t);
            xbase[0][c] = y + (size_t) c * ldy + sb * 256 + 16 * t;
            xbase[1][c] = xbase[0][c] + 128;
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int row = row0 + r;
            if (row >= nrows) break;
            const uint8_t * rec = w + ((size_t) row * nblk_x + sb) * 224;
            const float d6 = __half2float(*(const __half *) (rec + 208));
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint4 qlv = __ldg((const uint4 *) (rec + h * 64 + 16 * (t & 3)));
                const uint4 qhv = __ldg((const uint4 *) (rec + 128 + h * 32 + 16 * (t & 1)));
                const float sc = (float) ((const int8_t *) rec)[192 + 8 * h + t];
                const unsigned qu[4] = {qlv.x, qlv.y, qlv.z, qlv.w};
                const unsigned hu[4] = {qhv.x, qhv.y, qhv.z, qhv.w};
                half2 a2[NC];
#pragma unroll
                for (int c = 0; c < NC; c++) a2[c] = __float2half2_rn(0.f);
                const half2 * xp[NC];
#pragma unroll
                for (int c = 0; c < NC; c++) xp[c] = (const half2 *) xbase[h][c];
#pragma unroll
                for (int wq = 0; wq < 4; ++wq) {
                    // q per byte = (ql nibble) | ((qh 2 bits) << 4); fp16 subnormal bits = q
                    const unsigned nu = ((qu[wq] >> (4 * subn)) & 0x0F0F0F0Fu)
                                      | ((((hu[wq] >> shift2) & 0x03030303u) << 4));
                    const unsigned p01 = __byte_perm(nu, 0, 0x4140);   // half2 {q,4wq+0, +1} * 2^-24
                    const unsigned p23 = __byte_perm(nu, 0, 0x4342);   // half2 {q,4wq+2, +3}
                    const half2 n01 = *reinterpret_cast<const half2 *>(&p01);
                    const half2 n23 = *reinterpret_cast<const half2 *>(&p23);
#pragma unroll
                    for (int c = 0; c < NC; c++) {
                        a2[c] = __hfma2(n01, xp[c][2 * wq + 0], a2[c]);
                        a2[c] = __hfma2(n23, xp[c][2 * wq + 1], a2[c]);
                    }
                }
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    const float hf = __low2float(a2[c]) + __high2float(a2[c]);
                    const float4 s4 = et[h][c];
                    acc[c][r] += d6 * sc * (s4.x * hf - s4.y);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
            float v = acc[c][r];
            v += __shfl_down_sync(0xffffffffu, v, 4);
            v += __shfl_down_sync(0xffffffffu, v, 2);
            v += __shfl_down_sync(0xffffffffu, v, 1);
            v += __shfl_down_sync(0xffffffffu, v, 8);
            v += __shfl_down_sync(0xffffffffu, v, 16);
            if (lane == 0 && row0 + r < nrows) out[row0 + r + (size_t) c * nrows] = v;
        }
    }
}

static bool try_q6k(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t k = src0->ne[0], m = src0->ne[1];
    if (k % 256 != 0) return false;
    if (src0->nb[0] != ggml_type_size(src0->type) || (size_t) src0->nb[1] != ggml_row_size(src0->type, k)) return false;
    if (src1->ne[0] != k || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (src1->nb[0] != sizeof(float) || (size_t) src1->nb[1] != (size_t) k*sizeof(float)) return false;
    if (!ggml_is_contiguous(dst) || dst->ne[0] != m || dst->ne[1] != src1->ne[1]) return false;

    const int     nc      = (int) src1->ne[1];
    const int     nblk_x  = (int) (k / 256);
    const int64_t y_bytes = k * (int64_t) sizeof(__half);
    const int64_t s_bytes = (k / 16) * (int64_t) sizeof(float4);
    const size_t  bytes   = (size_t) 4 * (size_t) (y_bytes + s_bytes);

    // sized once for the largest shape seen; grown WITHOUT freeing (a captured graph may
    // still reference the old block - the #177 lesson on the rep side)
    static char * g_buf = nullptr;
    static size_t g_cap = 0;
    if (g_cap < bytes) {
        void * p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return false;
        g_buf = (char *) p; g_cap = bytes;
    }
    __half * yh = (__half *) g_buf;
    float4 * ys = (float4 *) (g_buf + 4*y_bytes);
    cudaStream_t stream = ctx.stream();

    // side rep: 224-byte records keyed like the Q3_K map (write-side erase covers invalidation)
    uint8_t * W = (uint8_t *) src0->data;
    auto it = g_mmvh_q6k_rep.find(src0->data);
    if (it == g_mmvh_q6k_rep.end() || it->second.ne0 != k || it->second.ne1 != m) {
        const int64_t nblocks = m * nblk_x;
        const size_t  rbytes  = (size_t) nblocks * 224;
        if (src0->buffer == nullptr ||
            ggml_backend_buffer_get_usage(src0->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
            !ggml_cuda_mmvh_sm60_repack_mem_ok(rbytes)) {
            return false;
        }
        void * p = nullptr;
        if (cudaMalloc(&p, rbytes) != cudaSuccess) return false;
        q6k_repack<<<(int) ((nblocks + 255) / 256), 256, 0, stream>>>((const uint8_t *) src0->data, (uint8_t *) p, (int) nblocks);
        if (cudaGetLastError() != cudaSuccess) { cudaFree(p); return false; }
        // never free a previous rep for this key: a captured graph may reference it
        g_mmvh_q6k_rep[src0->data] = { p, k, m };
        W = (uint8_t *) p;
    } else {
        W = (uint8_t *) it->second.ptr;
    }

    const int     nb  = (int) (k / 16);
    const int64_t ld  = (int64_t) (src1->nb[1] / sizeof(float));
    for (int c = 0; c < nc; c++) {
        q6k_prep_x<<<(nb + 7) / 8, 256, 0, stream>>>((const float *) src1->data + (size_t) c * ld,
                                                     yh + (size_t) c * k, ys + (size_t) c * (k / 16), nb);
    }
    // R=8 everywhere: the standalone sweep read R=8 best at 5 of 6 shapes and within 8% at the sixth;
    // per-shape R sniffing was rejected as benchmark tuning.
    constexpr int R = 8;
    const int     ldy = (int) k, lds = (int) (k / 16);
    const int     gr  = (int) ((m + 4*R - 1) / (4*R));
    float   * O = (float *) dst->data;
    if      (nc == 1) q6k_head<R,1><<<gr, 128, 0, stream>>>(W, yh, ys, O, (int) m, nblk_x, ldy, lds);
    else if (nc == 2) q6k_head<R,2><<<gr, 128, 0, stream>>>(W, yh, ys, O, (int) m, nblk_x, ldy, lds);
    else return false;
    return cudaGetLastError() == cudaSuccess;
}

} // namespace sm60q6k

// B-06 follow-up: the MTP draft head scores a fixed id subset of the Q4_K lm_head through MUL_MAT_ID
// one-row experts (mul_mat_vec_q_indexed_rows, ~242 GB/s at 64K rows). The subset is fixed per map,
// so the selected rows are gathered once into a contiguous [count, k] buffer and the kept fp16 head
// kernel runs on it (its 565 GB/s class). Only the draft logits' arithmetic class changes; the verify
// path and plain lane are untouched. Opt in with GGML_CUDA_MMV_SM60_Q4K_DHEAD=1.
static bool ggml_cuda_mmvh_sm60_q4k_dhead_on() {
    static const bool on = getenv("GGML_CUDA_MMV_SM60_Q4K_DHEAD") != nullptr &&
                           atoi(getenv("GGML_CUDA_MMV_SM60_Q4K_DHEAD")) != 0;
    return on;
}

__global__ void dhead_gather(const uint8_t * __restrict__ src, const int32_t * __restrict__ ids,
                             uint8_t * __restrict__ dst, const int row_bytes) {
    const uint8_t * s = src + (size_t) ids[blockIdx.x] * row_bytes;
    uint8_t * d = dst + (size_t) blockIdx.x * row_bytes;
    const int n4 = row_bytes / 16;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) { ((uint4 *) d)[i] = __ldg((const uint4 *) s + i); }
}

// True when ids is the identity prefix 0..count-1: the rep would be a byte-identical copy of
// src0's first count rows, so the head kernel can read src0->data directly through a view and
// the 184 MiB (at 64K) stays free. One 256 KB D2H of ids at rep-build time only.
static bool dhead_ids_are_prefix(const ggml_tensor * ids, int64_t count) {
    std::vector<int32_t> h((size_t) count);
    cudaMemcpy(h.data(), ids->data, (size_t) count * sizeof(int32_t), cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < count; ++i) {
        if (h[(size_t) i] != (int32_t) i) return false;
    }
    return true;
}

bool ggml_cuda_mmvh_sm60_q4k_dhead(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                   const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    if (ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_PASCAL) return false;
    if (src0->type != GGML_TYPE_Q4_K || src0->ne[1] != 1 || src0->ne[3] != 1 || !ggml_is_contiguous(src0)) return false;
    if (src1->type != GGML_TYPE_F32 || src1->ne[0] != src0->ne[0] || src1->ne[1] != 1 ||
        src1->ne[2] != 1 || src1->ne[3] != 1 || src1->nb[0] != sizeof(float) ||
        (size_t) src1->nb[1] != (size_t) src1->ne[0] * sizeof(float)) return false;
    if (ids->type != GGML_TYPE_I32 || ids->ne[1] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        ids->nb[0] != ggml_type_size(ids->type) || !ggml_is_contiguous(ids)) return false;
    if (dst->type != GGML_TYPE_F32 || dst->ne[0] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1 ||
        dst->ne[1] != ids->ne[0]) return false;
    const int64_t k = src0->ne[0];
    if (k % 256 != 0) return false;
    const int64_t count = ids->ne[0];
    const int64_t row_bytes = ggml_row_size(GGML_TYPE_Q4_K, k);

    uint8_t * W;
    auto it = g_mmvh_dhead_rep.find(ids->data);
    if (it != g_mmvh_dhead_rep.end() && it->second.ne0 == count &&
            (it->second.ne1 == (int64_t) src0->data || (it->second.ne1 == 0 && it->second.ptr == src0->data))) {
        W = (uint8_t *) it->second.ptr;
    } else if (dhead_ids_are_prefix(ids, count)) {
        // identity prefix map: view the original storage, no side buffer at all. ne1 == 0 marks
        // an un-owned view so rep_erase never frees it.
        g_mmvh_dhead_rep[ids->data] = { (void *) src0->data, count, 0 };
        W = (uint8_t *) src0->data;
    } else {
        // same admission rule as the other reps: weights buffer, and the allocation must leave
        // the residency margin intact
        if (src0->buffer == nullptr ||
            ggml_backend_buffer_get_usage(src0->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
            !ggml_cuda_mmvh_sm60_repack_mem_ok((size_t) count * row_bytes)) {
            return false;
        }
        void * p = nullptr;
        if (cudaMalloc(&p, (size_t) count * row_bytes) != cudaSuccess) return false;
        dhead_gather<<<(int) count, 128, 0, ctx.stream()>>>((const uint8_t *) src0->data,
                                                            (const int32_t *) ids->data, (uint8_t *) p, (int) row_bytes);
        if (cudaGetLastError() != cudaSuccess) { cudaFree(p); return false; }
        g_mmvh_dhead_rep[ids->data] = { p, count, (int64_t) src0->data };
        W = (uint8_t *) p;
    }

    // [k, count] view of the rep and a [count, 1] view of the MUL_MAT_ID dst, so try_head's
    // [m, k] x [k, nc] contract sees a plain GEMV. Slot i of the dst is rep row i.
    ggml_tensor fake_src0 = {};
    fake_src0.type = GGML_TYPE_Q4_K;
    fake_src0.ne[0] = k;       fake_src0.ne[1] = count;
    fake_src0.nb[0] = ggml_type_size(GGML_TYPE_Q4_K);
    fake_src0.nb[1] = row_bytes;
    fake_src0.data = W;
    ggml_tensor fake_dst = {};
    fake_dst.type = GGML_TYPE_F32;
    fake_dst.ne[0] = count;    fake_dst.ne[1] = 1;
    fake_dst.nb[0] = sizeof(float);
    fake_dst.nb[1] = sizeof(float) * count;
    fake_dst.data = dst->data;
    for (int d = 2; d < GGML_MAX_DIMS; ++d) {   // {} zeroes these; 0 breaks ggml_is_contiguous
        fake_src0.ne[d] = 1;  fake_src0.nb[d] = fake_src0.nb[d-1] * fake_src0.ne[d-1];
        fake_dst.ne[d]  = 1;  fake_dst.nb[d]  = fake_dst.nb[d-1]  * fake_dst.ne[d-1];
    }
    return sm60q4k::try_head(ctx, &fake_src0, src1, &fake_dst);
}

bool ggml_cuda_mmvh_sm60_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (ggml_cuda_mmvh_sm60_mode() == 0) {
        return false;
    }
    // R-38 head path: Q4_K, ncols == 1, many rows. Checked before the IQ3 type gate so the head can reach it.
    // (The m in [1024, 65536) extension failed its D1 gate - see q4k_small_on above.)
    if (sm60q4k::q4k_head_on() && src0->type == GGML_TYPE_Q4_K && src1->ne[1] >= 1 && src1->ne[1] <= 4 && src0->ne[1] >= 65536 &&
        ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_PASCAL) {
        if (sm60q4k::try_head(ctx, src0, src1, dst)) {
            return true;
        }
    }
    // R-38 second arm: Q3_K at ncols 1..4 via the 112-byte rep. Before the IQ3 type gate; numerics-changing.
    if (sm60q3k::q3k_on() && src0->type == GGML_TYPE_Q3_K && src1->ne[1] >= 1 && src1->ne[1] <= 4 &&
        ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_PASCAL) {
        if (sm60q3k::try_q3k(ctx, src0, src1, dst)) {
            return true;
        }
    }
    // Lever 2: Q6_K at ncols 1..2 via the 224-byte rep (the drafter's widths; A1 has q6_K only in
    // blk.64). Before the IQ3 type gate; numerics-changing, D1-gated.
    static const int q6k_nc_max = [] {
        const char * e = getenv("GGML_CUDA_MMV_SM60_Q6K_NC");
        return e ? atoi(e) : 2;
    }();
    if (sm60q6k::q6k_on() && src0->type == GGML_TYPE_Q6_K && src1->ne[1] >= 1 && src1->ne[1] <= q6k_nc_max &&
        ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_PASCAL) {
        if (sm60q6k::try_q6k(ctx, src0, src1, dst)) {
            return true;
        }
    }
    if (src0->type != GGML_TYPE_IQ3_XXS && src0->type != GGML_TYPE_IQ3_S) {
        return false;
    }
    if (ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_PASCAL) {
        return false;
    }

    const int64_t k     = src0->ne[0];
    const int64_t m     = src0->ne[1];
    const int64_t ncols = src1->ne[1];
    const bool    is_s  = src0->type == GGML_TYPE_IQ3_S;

    // P3.15 control: leave IQ3_XXS to the stock mmvq path (with the IQ3 smem grid + the ksigns
    // table), keeping K-B only for IQ3_S. The SBB-ARM reference shows the stock path ~6% faster
    // than K-B on IQ3_XXS per T=1 pass.
    static const bool xxs_stock = getenv("GGML_CUDA_MMV_SM60_XXS_STOCK") != nullptr &&
                                  atoi(getenv("GGML_CUDA_MMV_SM60_XXS_STOCK")) != 0;
    if (!is_s) {
        // ncols == 1 goes straight to the STOCK path. Wider batches fall through so the repack block
        // below materialises the 100-byte side buffer the stock n=1 path reads; the K-B launch is
        // declined for them too, just before the launch (so the x-cache claim is dropped first).
        if (xxs_stock && src1->ne[1] == 1) {
            return false;
        }
    } else {
        // IQ3_S: the stock mmvq path wins at ncols == 1 (11.26 vs 12.94 ms per T=1 pass) but loses
        // at ncols == 2 (166.9 vs 128.8 us/call), so the switch is shape-aware: only single-token
        // decode moves; the verify widths stay on K-B.
        static const bool iq3s_stock = getenv("GGML_CUDA_MMV_SM60_IQ3S_STOCK") != nullptr &&
                                       atoi(getenv("GGML_CUDA_MMV_SM60_IQ3S_STOCK")) != 0;
        if (iq3s_stock && ncols == 1) {
            return false;
        }
    }

    if (ncols < 1 || ncols > 4) {
        return false;
    }
    if (k % sm60mmv::QK != 0) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1) {
        return false;
    }
    if (src0->nb[0] != ggml_type_size(src0->type) || (size_t) src0->nb[1] != ggml_row_size(src0->type, k)) {
        return false;
    }
    if (src1->ne[0] != k || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src1->nb[0] != sizeof(float) || (size_t) src1->nb[1] != (size_t) k*sizeof(float)) {
        return false;
    }
    if (!ggml_is_contiguous(dst) || dst->ne[0] != m || dst->ne[1] != ncols || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }

    const int64_t xh_bytes = ncols*k*sizeof(half);
    const int64_t xs_bytes = (ncols*k/32)*sizeof(float);
    const size_t  x_bytes  = (size_t) (xh_bytes + xs_bytes);

    // Reuse the converted activation while the same src1 is consumed in the same graph epoch (gate and up
    // projections share it), so the second call skips convert_x. Falls back to the pool buffer if cudaMalloc fails.
    static char * g_x_ptr = nullptr;
    static size_t g_x_cap = 0;
    static const void * g_x_tensor = nullptr;
    static const void * g_x_src = nullptr;
    static int64_t g_x_k = 0, g_x_ncols = 0;
    static int g_x_epoch = -1;

    // Key on the tensor itself, not only its data address: ggml-alloc reuses a dead tensor's address within
    // one graph, so other GEMVs can leave src1->data pointing at a different tensor's storage.
    const bool x_hit = g_x_ptr != nullptr && g_x_tensor == src1 && g_x_src == src1->data && g_x_k == k &&
                       g_x_ncols == ncols && g_x_epoch == ctx.graph_epoch;
    ggml_cuda_pool_alloc<char> scratch(ctx.pool());
    half  * xh;
    float * xs;
    if (x_hit) {
        xh = (half *) g_x_ptr;
        xs = (float *) (g_x_ptr + xh_bytes);
    } else {
        if (g_x_cap < x_bytes) {
            if (g_x_ptr != nullptr) { cudaFree(g_x_ptr); }
            g_x_ptr = nullptr;
            g_x_cap = 0;
            if (cudaMalloc((void **) &g_x_ptr, x_bytes) == cudaSuccess) { g_x_cap = x_bytes; }
        }
        if (g_x_ptr != nullptr) {
            xh = (half *) g_x_ptr;
            xs = (float *) (g_x_ptr + xh_bytes);
            g_x_src = src1->data; g_x_k = k; g_x_ncols = ncols; g_x_epoch = ctx.graph_epoch; g_x_tensor = src1;
        } else {
            scratch.alloc(x_bytes);
            xh = (half *) scratch.ptr;
            xs = (float *) (scratch.ptr + xh_bytes);
        }
    }
    // R-08 diagnostic: force the conversion so a stale x-cache hit cannot hide. Default off.
    static const bool force_convert = getenv("GGML_CUDA_MMV_SM60_FORCE_CONVERT") != nullptr;
    const bool do_convert = force_convert || !x_hit;

    const uint8_t * W      = (const uint8_t *) src0->data;
    int64_t         rb     = (int64_t) src0->nb[1];
    const float   * x      = (const float *) src1->data;
    float         * y      = (float *) dst->data;
    cudaStream_t    stream = ctx.stream();
    bool            repacked = false;

    const int  rmode     = ggml_cuda_mmvh_sm60_repack_mode();
    const bool want_rep  = is_s ? (rmode == 1 || rmode == 3) : (rmode == 2 || rmode == 3);

    if (want_rep && src0->buffer != nullptr &&
        ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        auto & cache = is_s ? g_mmvh_iq3s_rep : g_mmvh_iq3xxs_rep;
        auto it = cache.find(src0->data);
        void * rp = nullptr;
        bool try_side = true;
        if (it != cache.end() && it->second.ne0 == k && it->second.ne1 == m) {
            rp = it->second.ptr;
            try_side = false;
        } else if (!is_s && ggml_cuda_mmvh_sm60_inplace_xxs()) {
            try_side = false;
            // In-place: repack into a one-tensor scratch, copy it back over the tensor's own storage (which
            // the allocation hook sized for 100-byte records), free the scratch, and register the tensor as
            // its own rep pointer. The decode reads it with the 100-byte stride; the cuBLAS prefill routes
            // through ggml_get_to_fp16_cuda_rep for tensors found in this map.
            const int64_t nb    = k / sm60mmv::QK;
            const int64_t bytes = m * nb * 100;
            void * scratch = nullptr;
            if (cudaMalloc(&scratch, (size_t) bytes) == cudaSuccess) {
                const int nblocks = (int) (m * nb);
                sm60mmv::repack_iq3xxs<<<(nblocks + 255)/256, 256, 0, stream>>>((const uint8_t *) src0->data, (uint8_t *) scratch, nblocks);
                if (cudaGetLastError() == cudaSuccess &&
                    cudaMemcpyAsync((void *) src0->data, scratch, (size_t) bytes, cudaMemcpyDeviceToDevice, stream) == cudaSuccess) {
                    cache[src0->data] = { (void *) src0->data, k, m };
                    rp = (void *) src0->data;
                }
                cudaFree(scratch);
            }
        } else if (is_s && ggml_cuda_mmvh_sm60_inplace_iq3s()) {
            try_side = false;
            // R-37: the IQ3_S twin. Same shape as the XXS branch, but a failed scratch malloc logs LOUDLY
            // and falls back to the side-buffer repack below: this route exists to remove that buffer, so
            // it coming back one tensor at a time must be visible.
            const int64_t nb    = k / sm60mmv::QK;
            const int64_t bytes = m * nb * sm60mmv::SB_BYTES_S_REP;
            void * scratch = nullptr;
            cudaError_t merr = cudaMalloc(&scratch, (size_t) bytes);
            if (merr == cudaSuccess) {
                const int nblocks = (int) (m * nb);
                sm60mmv::repack_iq3s<<<(nblocks + 255)/256, 256, 0, stream>>>((const uint8_t *) src0->data, (uint8_t *) scratch, nblocks);
                if (cudaGetLastError() == cudaSuccess &&
                    cudaMemcpyAsync((void *) src0->data, scratch, (size_t) bytes, cudaMemcpyDeviceToDevice, stream) == cudaSuccess) {
                    cache[src0->data] = { (void *) src0->data, k, m };
                    rp = (void *) src0->data;
                }
                cudaFree(scratch);
            } else {
                fprintf(stderr, "mmvh-sm60: IQ3_S in-place repack scratch malloc of %.1f MiB failed: %s; falling back to the side buffer\n",
                        (double) bytes / 1048576.0, cudaGetErrorString(merr));
                try_side = true;
            }
        } else if (!is_s && g_mmvh_iq3xxs_rep_bytes >= ggml_cuda_mmvh_sm60_repack_xxs_cap()) {
            try_side = false;
            // over the IQ3_XXS budget: fall back to the original layout
        }
        if (try_side) {
            const int64_t nb    = k / sm60mmv::QK;
            const int64_t bpb   = is_s ? sm60mmv::SB_BYTES_S_REP : 100;
            const int64_t bytes = m * nb * bpb;
            void * p = nullptr;
            if (ggml_cuda_mmvh_sm60_repack_mem_ok((size_t) bytes) && cudaMalloc(&p, (size_t) bytes) == cudaSuccess) {
                const int nblocks = (int) (m * nb);
                if (is_s) {
                    sm60mmv::repack_iq3s<<<(nblocks + 255)/256, 256, 0, stream>>>((const uint8_t *) src0->data, (uint8_t *) p, nblocks);
                } else {
                    sm60mmv::repack_iq3xxs<<<(nblocks + 255)/256, 256, 0, stream>>>((const uint8_t *) src0->data, (uint8_t *) p, nblocks);
                }
                if (cudaGetLastError() == cudaSuccess) {
                    cache[src0->data] = { p, k, m };
                    if (!is_s) {
                        g_mmvh_iq3xxs_rep_bytes += (size_t) bytes;
                    }
                    rp = p;
                } else {
                    cudaFree(p);
                }
            }
        }
        if (rp != nullptr) {
            W        = (const uint8_t *) rp;
            rb       = (k / sm60mmv::QK) * (is_s ? sm60mmv::SB_BYTES_S_REP : 100);
            repacked = true;
        }
    }

    // Opt-in: let K-B take IQ3_XXS at ncols 2..3 for tensors that were NOT repacked (the repacked
    // ones stay on stock, where the rep read is already cheap). Default off; changes numerics (fp16 x).
    if (!is_s && xxs_stock && !(sm60mmv::kb_wide_xxs() && !repacked && (ncols == 2 || ncols == 3))) {
        // The x-cache block above has claimed g_x_ptr for this src1, but convert_x only runs inside
        // launch_* when do_convert is true. Declining here with that claim live would make the next
        // call on the same src1 falsely x-hit and read the PREVIOUS activation (25 layers mix
        // IQ3_XXS and IQ3_S on one input), so drop the claim: this call does no conversion.
        g_x_tensor = nullptr;
        g_x_epoch  = -1;
        return false;
    }

    cudaError_t err;
    if (repacked && is_s) {
        err = sm60mmv::launch_iq3s_rep(W, rb, x, xh, xs, y, (int) m, (int) k, (int) ncols, stream, do_convert);
    } else if (repacked) {
        err = sm60mmv::launch_iq3xxs_rep(W, rb, x, xh, xs, y, (int) m, (int) k, (int) ncols, stream, do_convert);
    } else if (is_s) {
        err = sm60mmv::launch_iq3s(W, rb, x, xh, xs, y, (int) m, (int) k, (int) ncols, stream, do_convert);
    } else {
        err = sm60mmv::launch(W, rb, x, xh, xs, y, (int) m, (int) k, (int) ncols, stream, do_convert);
    }
    CUDA_CHECK(err);

    if (ggml_cuda_mmvh_sm60_stats()) {
        static bool printed = false;
        if (!printed) {
            printed = true;
            fprintf(stderr, "mmv_sm60: kernel=%s m=%d k=%d ncols=%d\n",
                    repacked ? (is_s ? "gemv_iq3s_rows_rep" : "gemv_iq3xxs_rows_rep")
                             : (is_s ? "gemv_iq3s_rows" : "gemv_iq3xxs_rows"),
                    (int) m, (int) k, (int) ncols);
        }
    }
    return true;
}