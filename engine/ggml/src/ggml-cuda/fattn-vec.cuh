#include "common.cuh"
#include "fattn-common.cuh"

static int ggml_cuda_fattn_vec_get_nthreads_host(const int cc) {
    return 128;
    GGML_UNUSED(cc);
}

static constexpr __device__ int ggml_cuda_fattn_vec_get_nthreads_device() {
    return 128;
}

// q8_0 KQ dot for two query heads that share one K row. Each K int is loaded once and dotted with
// both heads' Q; per head the products, their rounding and the sum order are those of
// vec_dot_fattn_vec_KQ_q8_0, so the two scores equal the single-head scores bit for bit.
template <int D, int nthreads>
static __device__ __forceinline__ void vec_dot_fattn_vec_KQ_q8_0_pair(
    const char * __restrict__ K_c, const int * __restrict__ Q_q8_a, const float2 * __restrict__ Q_ds_a,
    const int * __restrict__ Q_q8_b, const float2 * __restrict__ Q_ds_b, float & sum_a, float & sum_b) {

    const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;

    sum_a = 0.0f;
    sum_b = 0.0f;
#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib  = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        int v;
        ggml_cuda_memcpy_1<sizeof(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        const float d_K = __half2float(K_q8_0[ib].d);

        const float d_KQ_a = __fmul_rn(d_K, Q_ds_a[k_KQ_0/nthreads].x);
        sum_a = __fadd_rn(sum_a, __fmul_rn(d_KQ_a, float(ggml_cuda_dp4a(v, Q_q8_a[k_KQ_0/nthreads], 0))));

        const float d_KQ_b = __fmul_rn(d_K, Q_ds_b[k_KQ_0/nthreads].x);
        sum_b = __fadd_rn(sum_b, __fmul_rn(d_KQ_b, float(ggml_cuda_dp4a(v, Q_q8_b[k_KQ_0/nthreads], 0))));
    }
}

// q8_0 KQ for one K row (D=128) and both heads of a GQA pair, computed by one lane. The row is read as 17
// aligned 8-byte loads; the 34-byte blocks' d and int-aligned quants are taken from the words with shifts.
// Per head the arithmetic is that of vec_dot_fattn_vec_KQ_q8_0 on 16 lanes: lane l forms s_l = (0+p_l)+p_{l+16}
// and warp_reduce_sum<16> adds the s_l in the butterfly tree (offsets 8, 4, 2, 1), rebuilt below in the same order.
template <int D>
static __device__ __forceinline__ void kq_row_q8_0_pair16_tree(
    const uint2 * __restrict__ K_w, const int (&Q_a)[D/sizeof(int)], const float (&Qd_a)[D/QK8_1],
    const int (&Q_b)[D/sizeof(int)], const float (&Qd_b)[D/QK8_1], float & score_a, float & score_b) {
    static_assert(D == 128, "whole-row KQ is written for head size 128");

    uint32_t w[(D/QK8_0)*sizeof(block_q8_0)/sizeof(uint32_t)];
#pragma unroll
    for (int i = 0; i < int(sizeof(w)/sizeof(uint2)); ++i) {
        const uint2 t = K_w[i];
        w[2*i]   = t.x;
        w[2*i+1] = t.y;
    }

    int   v[D/sizeof(int)];
    float dK[D/QK8_0];
#pragma unroll
    for (int ib = 0; ib < int(D/QK8_0); ++ib) {
        const int off_d = ib*int(sizeof(block_q8_0));
        const uint32_t dw = w[off_d/4];
        dK[ib] = __half2float(__ushort_as_half(uint16_t((off_d & 2) ? (dw >> 16) : (dw & 0xFFFF))));
#pragma unroll
        for (int iqs = 0; iqs < QI8_0; ++iqs) {
            const int off = off_d + 2 + 4*iqs;
            v[ib*QI8_0 + iqs] = (off & 3) == 0 ? int(w[off/4]) : int(__byte_perm(w[off/4], w[off/4 + 1], 0x5432));
        }
    }

    float s_a[16];
    float s_b[16];
#pragma unroll
    for (int l = 0; l < 16; ++l) {
        float sum_a = 0.0f;
        float sum_b = 0.0f;
#pragma unroll
        for (int k = l; k < int(D/sizeof(int)); k += 16) {
            const int ib = k / QI8_0;
            const float d_KQ_a = __fmul_rn(dK[ib], Qd_a[ib]);
            const float d_KQ_b = __fmul_rn(dK[ib], Qd_b[ib]);
            sum_a = __fadd_rn(sum_a, __fmul_rn(d_KQ_a, float(ggml_cuda_dp4a(v[k], Q_a[k], 0))));
            sum_b = __fadd_rn(sum_b, __fmul_rn(d_KQ_b, float(ggml_cuda_dp4a(v[k], Q_b[k], 0))));
        }
        s_a[l] = sum_a;
        s_b[l] = sum_b;
    }

#pragma unroll
    for (int l = 0; l < 8; ++l) {
        s_a[l] = __fadd_rn(s_a[l], s_a[l + 8]);
        s_b[l] = __fadd_rn(s_b[l], s_b[l + 8]);
    }
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        s_a[l] = __fadd_rn(s_a[l], s_a[l + 4]);
        s_b[l] = __fadd_rn(s_b[l], s_b[l + 4]);
    }
#pragma unroll
    for (int l = 0; l < 2; ++l) {
        s_a[l] = __fadd_rn(s_a[l], s_a[l + 2]);
        s_b[l] = __fadd_rn(s_b[l], s_b[l + 2]);
    }
    score_a = __fadd_rn(s_a[0], s_a[1]);
    score_b = __fadd_rn(s_b[0], s_b[1]);
}

// q8_0 KQ for one K row (D=128) and all four heads of a GQA quad, computed by one lane. Same arithmetic and
// reduction tree as kq_row_q8_0_pair16_tree per head (lane l forms s_l = (0+p_l)+p_{l+16}, then the butterfly
// offsets 8, 4, 2, 1), so each head's score equals the pair kernel's score bit for bit. Q comes from shared memory.
template <int D>
static __device__ __forceinline__ void kq_row_q8_0_quad16_tree(
    const uint2 * __restrict__ K_w, const int (&Q_s)[4][D/sizeof(int)], const float (&Qd_s)[4][D/QK8_1], float (&score)[4]) {
    static_assert(D == 128, "whole-row KQ is written for head size 128");

    uint32_t w[(D/QK8_0)*sizeof(block_q8_0)/sizeof(uint32_t)];
#pragma unroll
    for (int i = 0; i < int(sizeof(w)/sizeof(uint2)); ++i) {
        const uint2 t = K_w[i];
        w[2*i]   = t.x;
        w[2*i+1] = t.y;
    }

    int   v[D/sizeof(int)];
    float dK[D/QK8_0];
#pragma unroll
    for (int ib = 0; ib < int(D/QK8_0); ++ib) {
        const int off_d = ib*int(sizeof(block_q8_0));
        const uint32_t dw = w[off_d/4];
        dK[ib] = __half2float(__ushort_as_half(uint16_t((off_d & 2) ? (dw >> 16) : (dw & 0xFFFF))));
#pragma unroll
        for (int iqs = 0; iqs < QI8_0; ++iqs) {
            const int off = off_d + 2 + 4*iqs;
            v[ib*QI8_0 + iqs] = (off & 3) == 0 ? int(w[off/4]) : int(__byte_perm(w[off/4], w[off/4 + 1], 0x5432));
        }
    }

#pragma unroll
    for (int h = 0; h < 4; ++h) {
        float s[16];
#pragma unroll
        for (int l = 0; l < 16; ++l) {
            float sum = 0.0f;
#pragma unroll
            for (int k = l; k < int(D/sizeof(int)); k += 16) {
                const int ib = k / QI8_0;
                const float d_KQ = __fmul_rn(dK[ib], Qd_s[h][ib]);
                sum = __fadd_rn(sum, __fmul_rn(d_KQ, float(ggml_cuda_dp4a(v[k], Q_s[h][k], 0))));
            }
            s[l] = sum;
        }
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            s[l] = __fadd_rn(s[l], s[l + 8]);
        }
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            s[l] = __fadd_rn(s[l], s[l + 4]);
        }
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            s[l] = __fadd_rn(s[l], s[l + 2]);
        }
        score[h] = __fadd_rn(s[0], s[1]);
    }
}

// Currently llvm with the amdgcn target does not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
// heads_as_cols: the ncols=2 columns are two sibling query heads of one KV head (GQA pair, one token)
// instead of two tokens. Each column keeps its own state; the pair shares one K load per int.
template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap, bool heads_as_cols = false> // D == head size
// minBlocksPerSM=1 matches upstream plain f16 decode (poolside board). minBlocks=2 was a
// turbo-path occupancy nudge that can regress f16/f16 TG on GB10 (~1% decode).
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), 1)
static __global__ void flash_attn_ext_vec(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
                            const fattn_ibias ibias) {
    GGML_UNUSED(ibias);
    ggml_cuda_pdl_lc();
#ifdef FLASH_ATTN_AVAILABLE
    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

#ifdef GGML_USE_HIP
#ifdef RDNA
    constexpr int nthreads_KQ_q = 2;
#else
    constexpr int nthreads_KQ_q = 4;
#endif // RDNA
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#else
    // q8_0 K at D=128, single column: 16 lanes per KQ dot, so each warp scores two rows at once.
    constexpr int nthreads_KQ_q = (type_K == GGML_TYPE_Q8_0 && D == 128 && (ncols == 1 || heads_as_cols)) ? 16 : (D/4 < 32 ? D/4 : 32);
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#endif // GGML_USE_HIP

    constexpr int nthreads    = ggml_cuda_fattn_vec_get_nthreads_device();
    // Turbo3 uses the float Q path (like f16/bf16), not q8_1 integer path
    constexpr bool K_is_unquantized = (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_BF16 || type_K == GGML_TYPE_TURBO3_0 || type_K == GGML_TYPE_TURBO2_0 || type_K == GGML_TYPE_TURBO4_0);
    constexpr bool V_is_unquantized = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16 || type_V == GGML_TYPE_TURBO3_0 || type_V == GGML_TYPE_TURBO2_0 || type_V == GGML_TYPE_TURBO4_0);
    constexpr bool K_is_turbo = (type_K == GGML_TYPE_TURBO3_0 || type_K == GGML_TYPE_TURBO2_0 || type_K == GGML_TYPE_TURBO4_0);
    // Turbo KQ dot does byte extraction + centroid lookup + scalar mul, not vectorized f16 loads.
    // nthreads_KQ=1: each thread computes a full KQ product alone — eliminates warp_reduce_sum
    // shuffle and halves KQ loop iterations. Each thread holds full Q vector in registers.
    //
    // That last property is what makes it unaffordable in general: Q_reg below is
    // [ncols][(D/2)/nthreads_KQ], so at nthreads_KQ=1 the per-lane Q working set is the whole
    // Q vector. Measured spill for turbo K (issue #294): ~4.3 KB/lane at ncols=2 on SM 86,
    // 295-786 VGPRs across CDNA/RDNA2/RDNA3/RDNA4, at both hsk=128 and hsk=256.
    //
    // The shared-memory LUT paths below are the exception. They compute the full D-length dot
    // per lane with no nthreads_KQ striding and never call warp_reduce_sum, so they structurally
    // require nthreads_KQ==1; raising it there makes every lane redundantly recompute the whole
    // dot (correct, but measured at -12.6% / -31.1% turbo2 decode at 8K / 32K on SM 86).
    // Those same shapes are also the ones that do not spill. So the split is gated on the LUT
    // being inactive rather than on D: keep 1 exactly where the LUT runs, split everywhere else.
    //
    // Kept in sync with n_centroids_lut below; turbo4 never gets a LUT (shmem budget).
    constexpr bool turbo_lut_active = (ncols == 1) && (D <= 256) &&
        (type_K == GGML_TYPE_TURBO3_0 || type_K == GGML_TYPE_TURBO2_0);
    constexpr int nthreads_KQ = K_is_turbo ? (turbo_lut_active ? 1 : 128 / cpy_nb)
                                           : (K_is_unquantized ? 128 / cpy_nb : nthreads_KQ_q);
    constexpr bool V_is_turbo = (type_V == GGML_TYPE_TURBO3_0 || type_V == GGML_TYPE_TURBO2_0 || type_V == GGML_TYPE_TURBO4_0);
    // Turbo V dequant is scalar (byte extract + LUT), not vectorized loads.
    // Halve nthreads_V to double V_cols_per_iter (process 2 V rows per loop iteration),
    // reducing loop overhead and improving ILP in the V aggregation phase.
    // Eighth nthreads_V for turbo: V_cols_per_iter goes from 4→8, processing 8 V positions
    // per outer loop iteration. Halves outer loop count again, more ILP from concurrent V rows.
    constexpr int nthreads_V  = V_is_unquantized ? (V_is_turbo ? (nthreads_V_q / 8 < 1 ? 1 : nthreads_V_q / 8) : 128 / cpy_nb) : nthreads_V_q;

    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");

#ifdef V_DOT2_F32_F16_AVAILABLE
    constexpr bool V_uses_four_rows = type_V == GGML_TYPE_TURBO3_0 || type_V == GGML_TYPE_TURBO2_0;
#else
    constexpr bool V_uses_four_rows = type_V == GGML_TYPE_TURBO3_0 || type_V == GGML_TYPE_TURBO2_0 || type_V == GGML_TYPE_TURBO4_0;
#endif // V_DOT2_F32_F16_AVAILABLE
    constexpr int V_rows_per_thread = V_is_unquantized ? (V_uses_four_rows ? 4 : 2*cpy_ne) : 4;
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ>();
    constexpr bool Q_q8_1 = !K_is_unquantized;
#ifdef V_DOT2_F32_F16_AVAILABLE
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, half,  V_rows_per_thread>();
#else
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, float, V_rows_per_thread>();
#endif // V_DOT2_F32_F16_AVAILABLE

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    constexpr bool pair = heads_as_cols;
    if constexpr (pair) {
        // Only the first head of each sibling group runs; it also computes the other ncols-1 heads (same KV head).
        if (head % ncols) {
            return;
        }
    }
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const half * maskh  = (const half  *) (mask + nb33*(sequence % ne33) + nb31*ic0);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = nthreads / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    constexpr int ne_KQ      = ncols*D;
    constexpr int ne_combine = nwarps*V_cols_per_iter*D;
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2            VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ half   KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#else
    float2           VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ float  KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#endif // V_DOT2_F32_F16_AVAILABLE

    // Shared-memory LUT for turbo KQ scoring: precompute Q[d] * centroid[c] once,
    // then the hot loop does turbo_lut[d][idx] (shmem read, no multiply).
    // turbo4 excluded: 16 centroids × D exceeds shmem budget.
    // Stride = n_centroids+1 to avoid bank conflicts.
    constexpr int n_centroids_lut = !turbo_lut_active ? 0 :
                                    (type_K == GGML_TYPE_TURBO3_0) ? 8 : 4;
    constexpr int lut_stride = n_centroids_lut > 0 ? n_centroids_lut + 1 : 1;
    __shared__ half turbo_lut[n_centroids_lut > 0 ? D : 1][lut_stride];

    // Sparse-V pruning is disabled: it was measured to corrupt output on SM60. With quantized V (q8_0)
    // at head size 256 and query width 2 -- the vector kernel's multi-column configuration -- the pruned
    // cases returned non-deterministic NaN / +inf / -inf against a finite CPU reference, while the same
    // shapes with unquantized V (which compile the prune out) were clean. The obvious alternatives were
    // ruled out before blaming the prune: compute-sanitizer memcheck reported no invalid access, and
    // zero-initializing Q_reg/Q_i32/Q_ds/KQ_reg did not change the outcome, so the garbage is produced
    // downstream of the KQ stage. Forcing the prune off turns the failing cases green (4/4). Re-enabling
    // it needs a corrected predicate plus a re-qualified operator suite, not just a faster benchmark.
    constexpr bool sparse_v_prune_enabled = false;

    // Sparse V: skip V dequant for positions with negligible attention weights.
    // At long context, most V positions contribute < 1e-6 to the output — skipping
    // their dequant saves significant compute (especially for quantized V types).
    constexpr float sparse_v_threshold_f = 1e-6f;
#ifdef V_DOT2_F32_F16_AVAILABLE
    const     half  sparse_v_threshold_h = __float2half(sparse_v_threshold_f);
#endif

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2  Q_reg[ncols][(D/2)/nthreads_KQ]; // Will be initialized completely.
#else
    __align__(16) float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // V_DOT2_F32_F16_AVAILABLE
    int    Q_i32[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    float2  Q_ds[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    // Whole-row Q for the pair whole-row KQ path: every lane scores its own row against all of Q.
    constexpr bool pair_row_kq = pair && nthreads_KQ == 16;
    // Quad (four heads of a GQA group): all four heads' Q sits in shared memory, not in registers.
    constexpr bool quad = pair && ncols == 4;
    constexpr bool pair_row_kq_regs = pair_row_kq && !quad;
    int    Q_all_i32[pair_row_kq_regs ? ncols : 1][pair_row_kq_regs ? D/sizeof(int) : 1];
    float  Q_all_d[pair_row_kq_regs ? ncols : 1][pair_row_kq_regs ? D/QK8_1 : 1];
    __shared__ __align__(16) int  Q_sh_i32[quad ? ncols : 1][quad ? D/sizeof(int) : 1];
    __shared__ float Q_sh_d[quad ? ncols : 1][quad ? D/QK8_1 : 1];
    // K tile of the whole-row pair path: nthreads rows of one 136-byte q8_0 row each, staged with coalesced loads.
    constexpr int K_row_words = pair_row_kq ? int((D/QK8_0)*sizeof(block_q8_0)/sizeof(uint2)) : 1;
    __shared__ uint2 K_tile[pair_row_kq ? nthreads : 1][K_row_words];
    // Registers holding the next K tile, loaded one iteration ahead so the global-load latency overlaps the compute.
    uint2 K_pre[pair_row_kq ? K_row_words : 1];

    ggml_cuda_pdl_sync();
    if constexpr (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (!pair && ncols > 1 && ic0 + j >= int(ne01.z)) {
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    if (i0 + WARP_SIZE <= int(D/sizeof(int)) || i < int(D/sizeof(int))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
            } else {
                const float * Q_f = (const float *) (Q + j*(pair ? nb02 : nb01));
                constexpr int nthreads_quantize = D/sizeof(int) < WARP_SIZE ? D/sizeof(int) : WARP_SIZE;
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<float2, nthreads_quantize>
                        (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }

            if constexpr (quad) {
                for (int i = tid; i < int(D/sizeof(int)); i += nthreads) {
                    Q_sh_i32[j][i] = tmp_q_i32[i];
                }
                for (int b = tid; b < int(D/QK8_1); b += nthreads) {
                    Q_sh_d[j][b] = tmp_q_ds[b].x;
                }
            } else if constexpr (pair_row_kq) {
#pragma unroll
                for (int i = 0; i < int(D/sizeof(int)); ++i) {
                    Q_all_i32[j][i] = tmp_q_i32[i];
                }
#pragma unroll
                for (int b = 0; b < int(D/QK8_1); ++b) {
                    Q_all_d[j][b] = tmp_q_ds[b].x;
                }
            }
        }

        __syncthreads();
    } else {
#ifdef V_DOT2_F32_F16_AVAILABLE
        const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*(pair ? nb02 : nb01));
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;

                __align__(16) float2 tmp[cpy_ne] = {{0.0f, 0.0f}};
                if (pair || ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0/nthreads_KQ + i1] = make_half2(tmp[i1].x, tmp[i1].y);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#else
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*(pair ? nb02 : nb01));
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;
                if (pair || ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x *= scale;
                Q_reg[j][k].y *= scale;
            }
        }
#endif // V_DOT2_F32_F16_AVAILABLE
    }

    // Build shared-memory LUT: turbo_lut[d][c] = half(Q[d] * scale * centroid[c])
    if constexpr (n_centroids_lut > 0 && ncols == 1) {
        const float * centroids_ptr = (type_K == GGML_TYPE_TURBO3_0) ? TURBO_CENTROIDS_3BIT :
                                      TURBO_CENTROIDS_2BIT;
        const float * Q_f = (const float *)(Q + 0*nb01);
        for (int d = tid; d < D; d += nthreads) {
            const float q_val = Q_f[d] * scale;
            for (int c = 0; c < n_centroids_lut; c++) {
                turbo_lut[d][c] = __float2half(q_val * centroids_ptr[c]);
            }
        }
        __syncthreads();
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    K     += blockIdx.y*nthreads * nb11;
    V     += blockIdx.y*nthreads * nb21;
    maskh += blockIdx.y*nthreads;
    if constexpr (pair_row_kq) {
        if (blockIdx.y*nthreads < k_VKQ_max) {
#pragma unroll
            for (int u = 0; u < K_row_words; ++u) {
                const int i = tid + u*nthreads;
                const int r = i / K_row_words;
                K_pre[u] = *((const uint2 *)(K + r*nb11) + (i - r*K_row_words));
            }
        }
    }
    for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,
             // Increment pointers after each loop:
             K += gridDim.y*nthreads*nb11, V += gridDim.y*nthreads*nb21, maskh += gridDim.y*nthreads) {

        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[ncols]; // KQ in registers.

        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

        if constexpr (pair_row_kq) {
            // K_pre holds this tile (loaded during the previous iteration): stage it, then start loading the next tile.
            __syncthreads(); // the previous tile's readers are done before its rows are overwritten
#pragma unroll
            for (int u = 0; u < K_row_words; ++u) {
                const int i = tid + u*nthreads;
                const int r = i / K_row_words;
                K_tile[r][i - r*K_row_words] = K_pre[u];
            }
            __syncthreads();
            if (k_VKQ_0 + int(gridDim.y)*nthreads < k_VKQ_max) {
#pragma unroll
                for (int u = 0; u < K_row_words; ++u) {
                    const int i = tid + u*nthreads;
                    const int r = i / K_row_words;
                    K_pre[u] = *((const uint2 *)(K + int(gridDim.y)*nthreads*nb11 + r*nb11) + (i - r*K_row_words));
                }
            }
        }

        if constexpr (pair && nthreads_KQ == 16) {
            // Pair path, whole row per lane: lane x owns row threadIdx.y*32 + x and rebuilds the 16-lane
            // butterfly sum of its row itself (see kq_row_q8_0_pair16_tree): no shuffles, K read once per row.
            // Only the owner lane applies softcap, mask and max for its row.
            const int i_KQ = threadIdx.y*WARP_SIZE + threadIdx.x;
            float score[ncols];
            if constexpr (quad) {
                kq_row_q8_0_quad16_tree<D>(K_tile[threadIdx.y*WARP_SIZE + threadIdx.x], Q_sh_i32, Q_sh_d, score);
            } else {
                kq_row_q8_0_pair16_tree<D>(K_tile[threadIdx.y*WARP_SIZE + threadIdx.x], Q_all_i32[0], Q_all_d[0], Q_all_i32[1], Q_all_d[1], score[0], score[1]);
            }
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = score[j];

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                if (mask) {
                    sum += slope*__half2float(maskh[i_KQ]);
                }

                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);
                KQ_reg[j] = sum;
            }
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
#pragma unroll
                for (int offset = 8; offset > 0; offset >>= 1) {
                    KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
                }
            }
        } else
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ-1))) + i_KQ_0;

            float sum_pair[ncols];
            if constexpr (pair) {
                vec_dot_fattn_vec_KQ_q8_0_pair<D, nthreads_KQ>(K + i_KQ*nb11, Q_i32[0], Q_ds[0], Q_i32[1], Q_ds[1], sum_pair[0], sum_pair[1]);
                sum_pair[0] = warp_reduce_sum<nthreads_KQ>(sum_pair[0]);
                sum_pair[1] = warp_reduce_sum<nthreads_KQ>(sum_pair[1]);
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum;
                if constexpr (n_centroids_lut > 0 && ncols == 1 && type_K == GGML_TYPE_TURBO3_0) {
                    // LUT scoring: 8 elements per iteration (2 qs bytes + 1 signs byte)
                    const block_turbo3_0 * K_turbo = (const block_turbo3_0 *)(K + i_KQ*nb11);
                    sum = 0.0f;
                    for (int d0 = 0; d0 < D; d0 += 8) {
                        const int ib = d0 / QK_TURBO3;
                        const int jj = d0 % QK_TURBO3;
                        const float norm = __half2float(K_turbo[ib].norm);
                        const uint8_t qs0 = K_turbo[ib].qs[jj / 4];
                        const uint8_t qs1 = K_turbo[ib].qs[jj / 4 + 1];
                        const uint8_t sgn = K_turbo[ib].signs[jj / 8];
                        sum += (__half2float(turbo_lut[d0  ][((qs0>>0)&3)|((sgn>>0&1)<<2)]) +
                                __half2float(turbo_lut[d0+1][((qs0>>2)&3)|((sgn>>1&1)<<2)]) +
                                __half2float(turbo_lut[d0+2][((qs0>>4)&3)|((sgn>>2&1)<<2)]) +
                                __half2float(turbo_lut[d0+3][((qs0>>6)&3)|((sgn>>3&1)<<2)]) +
                                __half2float(turbo_lut[d0+4][((qs1>>0)&3)|((sgn>>4&1)<<2)]) +
                                __half2float(turbo_lut[d0+5][((qs1>>2)&3)|((sgn>>5&1)<<2)]) +
                                __half2float(turbo_lut[d0+6][((qs1>>4)&3)|((sgn>>6&1)<<2)]) +
                                __half2float(turbo_lut[d0+7][((qs1>>6)&3)|((sgn>>7&1)<<2)])) * norm;
                    }
                } else if constexpr (n_centroids_lut > 0 && ncols == 1 && type_K == GGML_TYPE_TURBO2_0) {
                    // LUT scoring for turbo2: 8 elements per iteration (2 qs bytes, no signs)
                    const block_turbo2_0 * K_turbo = (const block_turbo2_0 *)(K + i_KQ*nb11);
                    sum = 0.0f;
                    for (int d0 = 0; d0 < D; d0 += 8) {
                        const int ib = d0 / QK_TURBO2;
                        const int jj = d0 % QK_TURBO2;
                        const float norm = __half2float(K_turbo[ib].norm);
                        const uint8_t qs0 = K_turbo[ib].qs[jj / 4];
                        const uint8_t qs1 = K_turbo[ib].qs[jj / 4 + 1];
                        sum += (__half2float(turbo_lut[d0  ][(qs0>>0)&3]) +
                                __half2float(turbo_lut[d0+1][(qs0>>2)&3]) +
                                __half2float(turbo_lut[d0+2][(qs0>>4)&3]) +
                                __half2float(turbo_lut[d0+3][(qs0>>6)&3]) +
                                __half2float(turbo_lut[d0+4][(qs1>>0)&3]) +
                                __half2float(turbo_lut[d0+5][(qs1>>2)&3]) +
                                __half2float(turbo_lut[d0+6][(qs1>>4)&3]) +
                                __half2float(turbo_lut[d0+7][(qs1>>6)&3])) * norm;
                    }
                } else if constexpr (pair) {
                    sum = sum_pair[j];
                } else {
                    sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                    sum = warp_reduce_sum<nthreads_KQ>(sum);
                }

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                if (mask && (pair || ncols == 1 || ic0 + j < int(ne01.z))) {
                    sum += slope*__half2float(maskh[(pair ? 0 : j)*ne11 + i_KQ]);
                }

                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);

                if ((nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ) == uint32_t(i_KQ_0)) {
                    KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = __expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = __expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            if constexpr (!V_is_turbo) { KQ[j*nthreads + tid] = KQ_reg[j]; }

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }

            if constexpr (!V_is_turbo) { ggml_cuda_syncwarp(); }

        // this thread's first V row and element offset, formed once: the row steps by nb21 per k0 step
        const char * V_k = V + (threadIdx.y*WARP_SIZE + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V))*nb21;
        const int V_lane = (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread;

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter, V_k += V_cols_per_iter*nb21) {
            const int k = threadIdx.y*WARP_SIZE + k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);

#ifdef V_DOT2_F32_F16_AVAILABLE
            half2 KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                if constexpr (V_is_turbo) {
                    const float kq_val = __shfl_sync(0xFFFFFFFF, KQ_reg[j], k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V), WARP_SIZE);
                    KQ_k[j] = make_half2(__float2half(kq_val), __float2half(kq_val));
                } else {
                    KQ_k[j] = __half2half2(KQ[j*nthreads + k]);
                }
            }

            // Sparse V: skip V dequant if all attention weights for this position are negligible.
            // Compiled out for unquantized V (f16/bf16/turbo): at short decode depth the
            // threshold almost never fires, so the branch is pure overhead and regresses
            // plain f16/f16 TG vs upstream. Keep for quantized V only.
            if constexpr (!V_is_unquantized) {
                bool dominated = true;
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    if (__hgt(__low2half(KQ_k[j]), sparse_v_threshold_h)) { dominated = false; break; }
                }
                if (sparse_v_prune_enabled && dominated) { continue; }
            }

#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                half2 tmp[V_rows_per_thread/2];
                if constexpr (type_V == GGML_TYPE_BF16) {
                    float2 tmp_f[V_rows_per_thread/2];
                    dequantize_V(V_k, tmp_f,
                        2*i_VKQ_0 + V_lane);
#pragma unroll
                    for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
                        tmp[i_VKQ_1] = __float22half2_rn(tmp_f[i_VKQ_1]);
                    }
                } else {
                    dequantize_V(V_k, tmp,
                        2*i_VKQ_0 + V_lane);
                }
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#else
            float KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                if constexpr (V_is_turbo) {
                    KQ_k[j] = __shfl_sync(0xFFFFFFFF, KQ_reg[j], k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V), WARP_SIZE);
                } else {
                    KQ_k[j] = KQ[j*nthreads + k];
                }
            }

            // Sparse V: quantized V only — see half2 path comment above.
            if constexpr (!V_is_unquantized) {
                bool dominated = true;
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    if (KQ_k[j] >= sparse_v_threshold_f) { dominated = false; break; }
                }
                if (sparse_v_prune_enabled && dominated) { continue; }
            }

            // Turbo V path: precompute scaled centroids once per block to eliminate
            // per-element norm multiply.  centroid[idx]*norm is computed 8/4/16 times
            // (once per centroid) instead of D times (once per element).
            if constexpr (type_V == GGML_TYPE_TURBO3_0) {
                const block_turbo3_0 * vb = (const block_turbo3_0 *)(V + k*nb21);
                int prev_ib = -1;
                float sc[8];

#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                    const int i0 = 2*i_VKQ_0 + (threadIdx.x % nthreads_V)*V_rows_per_thread;
                    const int ib = i0 / QK_TURBO3;
                    const int j0 = i0 % QK_TURBO3;

                    if (ib != prev_ib) {
                        prev_ib = ib;
                        const float norm = __half2float(vb[ib].norm);
#pragma unroll
                        for (int c = 0; c < 8; ++c) { sc[c] = TURBO_CENTROIDS_3BIT[c] * norm; }
                    }

                    const uint8_t qs_byte  = vb[ib].qs[j0 / 4];
                    const uint8_t sgn_byte = vb[ib].signs[j0 / 8];
                    const int     shift_s  = j0 % 8;

                    const uint8_t idx0 = ((qs_byte >> 0) & 0x3) | (((sgn_byte >> (shift_s+0)) & 0x1) << 2);
                    const uint8_t idx1 = ((qs_byte >> 2) & 0x3) | (((sgn_byte >> (shift_s+1)) & 0x1) << 2);
                    const uint8_t idx2 = ((qs_byte >> 4) & 0x3) | (((sgn_byte >> (shift_s+2)) & 0x1) << 2);
                    const uint8_t idx3 = ((qs_byte >> 6) & 0x3) | (((sgn_byte >> (shift_s+3)) & 0x1) << 2);

#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + 0].x += sc[idx0]*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + 0].y += sc[idx1]*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + 1].x += sc[idx2]*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + 1].y += sc[idx3]*KQ_k[j];
                    }
                }
            } else if constexpr (type_V == GGML_TYPE_TURBO2_0) {
                const block_turbo2_0 * vb = (const block_turbo2_0 *)(V + k*nb21);
                int prev_ib = -1;
                float sc[4];

#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                    const int i0 = 2*i_VKQ_0 + (threadIdx.x % nthreads_V)*V_rows_per_thread;
                    const int ib = i0 / QK_TURBO2;
                    const int j0 = i0 % QK_TURBO2;

                    if (ib != prev_ib) {
                        prev_ib = ib;
                        const float norm = __half2float(vb[ib].norm);
#pragma unroll
                        for (int c = 0; c < 4; ++c) { sc[c] = TURBO_CENTROIDS_2BIT[c] * norm; }
                    }

                    const uint8_t qs_byte = vb[ib].qs[j0 / 4];

                    const uint8_t idx0 = (qs_byte >> 0) & 0x3;
                    const uint8_t idx1 = (qs_byte >> 2) & 0x3;
                    const uint8_t idx2 = (qs_byte >> 4) & 0x3;
                    const uint8_t idx3 = (qs_byte >> 6) & 0x3;

#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + 0].x += sc[idx0]*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + 0].y += sc[idx1]*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + 1].x += sc[idx2]*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + 1].y += sc[idx3]*KQ_k[j];
                    }
                }
            } else if constexpr (type_V == GGML_TYPE_TURBO4_0) {
                static_assert(V_rows_per_thread == 4);
                const block_turbo4_0 * vb = (const block_turbo4_0 *)(V + k*nb21);
                int prev_ib = -1;
                float sc[16];

#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                    const int i0 = 2*i_VKQ_0 + (threadIdx.x % nthreads_V)*V_rows_per_thread;
                    const int ib = i0 / QK_TURBO4;
                    const int j0 = i0 % QK_TURBO4;

                    if (ib != prev_ib) {
                        prev_ib = ib;
                        const float norm = __half2float(vb[ib].norm);
#pragma unroll
                        for (int c = 0; c < 16; ++c) { sc[c] = TURBO_CENTROIDS_4BIT[c] * norm; }
                    }

                    const uint8_t qs_byte0 = vb[ib].qs[j0 / 2];
                    const uint8_t qs_byte1 = vb[ib].qs[j0 / 2 + 1];

                    const uint8_t idx0 = (qs_byte0 >> 0) & 0xF;
                    const uint8_t idx1 = (qs_byte0 >> 4) & 0xF;
                    const uint8_t idx2 = (qs_byte1 >> 0) & 0xF;
                    const uint8_t idx3 = (qs_byte1 >> 4) & 0xF;

#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + 0].x += sc[idx0]*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + 0].y += sc[idx1]*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + 1].x += sc[idx2]*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + 1].y += sc[idx3]*KQ_k[j];
                    }
                }
            } else {
#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                    float2 tmp[V_rows_per_thread/2];
                    dequantize_V(V_k, tmp,
                        2*i_VKQ_0 + V_lane);
#pragma unroll
                    for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                        for (int j = 0; j < ncols; ++j) {
                            VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x*KQ_k[j];
                            VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y*KQ_k[j];
                        }
                    }
                }
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    if (sinks && blockIdx.y == 0) {
        const float sink = ((const float *) sinks)[head];

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            const float kqmax_new_j = fmaxf(sink, KQ_max[j]);
            const float KQ_max_scale = __expf(KQ_max[j] - kqmax_new_j);
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + (threadIdx.x == 0 ? __expf(sink - KQ_max[j]) : 0.0f);

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (!pair && ncols > 1 && ic0 + j_VKQ >= int(ne01.z)) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = __expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

#ifdef V_DOT2_F32_F16_AVAILABLE
        half2 * VKQ_tmp = (half2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

        const half2 kqmax_scale_h2 = make_half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread*sizeof(half)>(VKQ_tmp + i_VKQ, &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
        }
#else
        float2 * VKQ_tmp = (float2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // V_DOT2_F32_F16_AVAILABLE

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }

        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                if (gridDim.y == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence*int(ne01.z) + ic0 + (pair ? 0 : j_VKQ))*ne02 + head + (pair ? j_VKQ : 0))*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;
            }
        }

        if (j_VKQ < ncols-1) {
            __syncthreads();
        }

    }

    if (gridDim.y != 1 && tid < ncols && (pair || ncols == 1 || ic0 + tid < int(ne01.z))) {
        dst_meta[((sequence*int(ne01.z) + ic0 + (pair ? 0 : tid))*ne02 + head + (pair ? tid : 0))*gridDim.y + blockIdx.y] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

// Single-token decode with the two query heads of each GQA pair as the two columns (K loaded once per pair).
template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_pair_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(cc);
    const int nwarps   = nthreads / WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_ext_vec<D, 2, type_K, type_V, false, true>;
    const bool need_f16_K = false;
    const bool need_f16_V = false;
    constexpr size_t nbytes_shared = 0;
    // Split the KV range exactly as the single-head kernel would, so the combine order matches.
    fattn_kernel_t occupancy_kernel = flash_attn_ext_vec<D, 1, type_K, type_V, false, false>;
    // K views of 8192 and up (the full-attention layers at depth) take 7 splits: one wave of 112 CTAs on 56 SMs at 2 CTAs/SM.
    const ggml_tensor * K = dst->src[1];
    const int fixed_parallel_blocks = K->ne[1] >= 8192 ? 7 : 0;
    launch_fattn<D, 2, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false, false,
        WARP_SIZE, occupancy_kernel, fixed_parallel_blocks);
}

// Single-token decode at K views of 8192 and up with the four query heads of each GQA group as the four columns
// (K loaded once per group, 14 splits: 8 groups x 14 = 112 CTAs in one wave). Shorter views keep the pair path.
template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_quad_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(cc);
    const int nwarps   = nthreads / WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_ext_vec<D, 4, type_K, type_V, false, true>;
    const bool need_f16_K = false;
    const bool need_f16_V = false;
    constexpr size_t nbytes_shared = 0;
    fattn_kernel_t occupancy_kernel = flash_attn_ext_vec<D, 1, type_K, type_V, false, false>;
    launch_fattn<D, 2, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false, false,
        WARP_SIZE, occupancy_kernel, 14);
}

template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(cc);
    const int nwarps   = nthreads / WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap>;
    const bool need_f16_K = type_K == GGML_TYPE_F16;
    const bool need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<D, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false, false);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if constexpr (D == 128 && type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
        const ggml_tensor * K = dst->src[1];
        float max_bias;
        memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
        if (Q->ne[1] == 1 && logit_softcap == 0.0f && max_bias == 0.0f && dst->src[4] == nullptr &&
                Q->ne[2] % K->ne[2] == 0 && (Q->ne[2] / K->ne[2]) % 4 == 0 && K->ne[1] >= 12288) {
            ggml_cuda_flash_attn_ext_vec_quad_impl<D, type_K, type_V>(ctx, dst);
            return;
        }
        if (Q->ne[1] == 1 && logit_softcap == 0.0f && max_bias == 0.0f && dst->src[4] == nullptr &&
                Q->ne[2] % K->ne[2] == 0 && (Q->ne[2] / K->ne[2]) % 2 == 0) {
            ggml_cuda_flash_attn_ext_vec_pair_impl<D, type_K, type_V>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 2;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template void ggml_cuda_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_BF16); \

EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_BF16)

// TurboQuant3 — turbo3 K + turbo3 V (KV cache uses same type)
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);

// Mixed turbo3/q8_0 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0);

// Mixed f16/turbo3 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_F16, GGML_TYPE_TURBO3_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO3_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_F16);

// TurboQuant2 -- turbo2 K + turbo2 V
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);

// Mixed turbo2/q8_0 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_0);

// Mixed f16/turbo2 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_F16, GGML_TYPE_TURBO2_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO2_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_F16);

// Mixed turbo3/turbo2 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0);

// TurboQuant4 — turbo4 K + turbo4 V (KV cache uses same type)
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);

// Mixed turbo4/q8_0 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0);

// Mixed f16/turbo4 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_F16, GGML_TYPE_TURBO4_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO4_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_F16);

// Mixed turbo4/turbo3 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0);

// Mixed turbo4/turbo2 KV cache types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0);
