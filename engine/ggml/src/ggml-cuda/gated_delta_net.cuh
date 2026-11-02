#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// SBB-24 (shinbunbun): description of a folded-in SSM state GET_ROWS. gdn reads row rows[seq] straight out of
// base and writes the same values to gather_dst, so the folded GET_ROWS node is preserved.
struct ggml_cuda_gated_delta_net_fused_gather {
    const float *   base       = nullptr;
    const int32_t * rows       = nullptr;
    int64_t         row_stride = 0;
    float *         gather_dst = nullptr;
};

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// SBB-24: same as fused_cache, plus the folded-in SSM state GET_ROWS (gdn reads the cache row directly and also
// writes the folded node's dst).
void ggml_cuda_op_gated_delta_net_fused_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                               const ggml_cuda_gated_delta_net_fused_cache * cache,
                                               const ggml_cuda_gated_delta_net_fused_gather * gather);
