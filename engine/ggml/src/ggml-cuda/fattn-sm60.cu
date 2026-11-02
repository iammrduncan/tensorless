// P100 v2 custom kernel K-A: split-KV flash-decoding for sm_60 with q4_0 K/V (see fattn-sm60.cuh and
// the standalone harness in the research repository). Opt-in with GGML_CUDA_FATTN_SM60=1 until the v2 quality gate passes.
//
// Replaces, for D=256 / GQA 6 / 1..4 query tokens on cc 600:
//   * T<=2: flash_attn_ext_vec<256,...,q4_0,q4_0> (~38 GB/s effective on sm_60, latency-bound), and
//   * T>=3: the TILE kernel, which first converts the whole q4_0 cache to f16 on every call (and reserves that
//           scratch in the compute buffer via ggml_cuda_flash_attn_ext_get_alloc_size).

#include "common.cuh"
#include "fattn-sm60.cuh"
#include "fattn.cuh"

#include <cstdlib>
#include <cstring>

static int ggml_cuda_fattn_sm60_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_CUDA_FATTN_SM60");
        return e ? atoi(e) : 0;
    }();
    return mode;
}

bool ggml_cuda_fattn_sm60_applicable(int device, const ggml_tensor * dst) {
    if (ggml_cuda_fattn_sm60_mode() == 0) {
        return false;
    }
    if (ggml_cuda_info().devices[device].cc != GGML_CUDA_CC_PASCAL) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    if (!Q || !K || !V || sinks) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_Q4_0 || V->type != GGML_TYPE_Q4_0 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->ne[0] != sm60fa::D || K->ne[0] != sm60fa::D || V->ne[0] != sm60fa::D) {
        return false;
    }
    if (K->ne[2] <= 0 || V->ne[2] != K->ne[2] || Q->ne[2] != 6 * K->ne[2]) {
        return false;
    }
    if (Q->ne[1] < 1 || Q->ne[1] > sm60fa::T_SUPPORTED || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) {
        return false;
    }
    if (V->ne[1] != K->ne[1]) {
        return false;
    }
    float scale = 0.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] ||
                 mask->ne[2] != 1 || mask->ne[3] != 1 || mask->nb[1] % 2 != 0)) {
        return false;
    }
    if (ggml_row_size(K->type, sm60fa::D) != (size_t) sm60fa::ROW_BYTES ||
        K->nb[1] % 16 != 0 || K->nb[2] % 16 != 0 || V->nb[1] % 16 != 0 || V->nb[2] % 16 != 0) {
        return false;
    }
    if (Q->nb[0] != sizeof(float) || Q->nb[1] % sizeof(float) != 0 || Q->nb[2] % sizeof(float) != 0) {
        return false;
    }
    if (!ggml_is_contiguous(dst) || dst->ne[0] != sm60fa::D || dst->ne[1] != Q->ne[2] || dst->ne[2] != Q->ne[1]) {
        return false;
    }
    return true;
}

bool ggml_cuda_fattn_sm60(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (((uintptr_t) K->data) % 16 != 0 || ((uintptr_t) V->data) % 16 != 0) {
        return false;  // the kernel loads rows with 16-byte vector loads
    }

    sm60fa::Params p{};
    p.Q         = (const float *) Q->data;
    p.K         = (const char *) K->data;
    p.V         = (const char *) V->data;
    p.mask      = mask ? (const half *) mask->data : nullptr;
    p.dst       = (float *) dst->data;
    p.n_kv      = (int) K->ne[1];
    p.n_head    = (int) Q->ne[2];
    p.n_head_kv = (int) K->ne[2];
    p.T         = (int) Q->ne[1];
    p.q_tok     = (long) (Q->nb[1] / sizeof(float));
    p.q_head    = (long) (Q->nb[2] / sizeof(float));
    p.k_pos     = (long) K->nb[1];
    p.k_head    = (long) K->nb[2];
    p.v_pos     = (long) V->nb[1];
    p.v_head    = (long) V->nb[2];
    p.m_row     = mask ? (long) (mask->nb[1] / sizeof(half)) : 0;
    memcpy(&p.scale, (const float *) dst->op_params + 0, sizeof(float));

    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
    // CTAs per SM to aim for. 8 gives 416-432 blocks at 32K/65K, i.e. 1.86-1.93 waves of the
    // 4-CTA/SM occupancy; 4 aims for one full wave instead.
    static const int cta_mult = [] {
        const char * e = getenv("GGML_CUDA_FATTN_SM60_CTAS");
        return e != nullptr ? atoi(e) : 8;
    }();
    // R-28: op_params slot 4 carries the allocated kv length (ggml_flash_attn_ext_set_n_kv_max, set in
    // llama-graph.cpp), so the split can be pinned to the cell rather than following this op's padded n_kv.
    const int32_t n_kv_alloc = ggml_get_op_params_i32(dst, 4);
    // R-28 is PARKED DEFAULT-OFF. The pin is not a contract no-op: n_kv steps 1152 -> 1280 partway through a
    // 65K cell (the receipts put prompt_tokens at 64448 / 64386, so n_kv = PAD(used+1, 256) = 64512 for the
    // 4-token prompt tail and every decode position up to 64511), and the old derivation gives 1152 (56
    // splits) there while the pin gives 1280 (51). That is 64 code-65k and 126 prose-65k decode steps per
    // lane on a different K-A grid and reduction order. The card's A/B gate has NOT run - my D1 was
    // bit-identical only because all 256 of its positions sat above 64512 and so never exercised the pin.
    // Opt in with GGML_CUDA_FATTN_SM60_PIN=1; enable it by default only after the card's gate passes.
    static const bool pin_split = [] {
        const char * e = getenv("GGML_CUDA_FATTN_SM60_PIN");
        return e != nullptr && atoi(e) != 0;
    }();
    if (pin_split) {
        static const bool pin_dbg = [] { const char * e = getenv("GGML_CUDA_FATTN_SM60_PIN_DBG"); return e != nullptr && atoi(e) != 0; }();
        if (pin_dbg) { static bool once = false; if (!once) { once = true; fprintf(stderr, "sm60fa pin: first call n_kv=%d n_kv_alloc=%d\n", (int) p.n_kv, n_kv_alloc); } }
        sm60fa::choose_split_pinned(p.n_kv, (int) n_kv_alloc, p.n_head_kv, cta_mult * nsm, p.kv_per_split, p.n_splits);
    } else {
        sm60fa::choose_split(p.n_kv, p.n_head_kv, cta_mult * nsm, p.kv_per_split, p.n_splits);
    }

    const int    C       = 6 * p.T;
    const size_t acc_b   = (size_t) p.n_head_kv * p.n_splits * C * sm60fa::D * sizeof(float);
    const size_t ml_b    = (size_t) p.n_head_kv * p.n_splits * C * sizeof(float2);
    ggml_cuda_pool_alloc<char> ws(ctx.pool(), acc_b + ml_b);
    p.part_acc = (float *) ws.ptr;
    p.part_ml  = (float2 *) (ws.ptr + acc_b);

    CUDA_CHECK(sm60fa::launch(p, ctx.stream()));
    return true;
}
