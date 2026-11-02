#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

// RMS_NORM -> SCALE (bias 0) in one launch; rms is not written unless the shape falls back.
void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms, ggml_tensor * scale);

// RMS_NORM -> MUL -> SILU -> MUL (gated norm) in one launch; the rms, weight-mul and silu outputs are not written.
bool ggml_cuda_rms_norm_gate_ok(const ggml_tensor * rms);
void ggml_cuda_op_rms_norm_gate_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms, ggml_tensor * weight_mul,
                                      ggml_tensor * gate_silu, ggml_tensor * out);

// SBB-19: fuse ADD -> RMS_NORM -> MUL (residual add folded into rms_norm's prologue).
// NORM-EMITS-STEP2: q8_emit/q8_npad/q8_swizzle optionally make the same launch also produce the
// q8_1 form of its output (the GEMVs' src1), registered in the shared q8_cache.
void ggml_cuda_op_rms_norm_fused_pre_add(ggml_backend_cuda_context & ctx,
                                         ggml_tensor *               add_tensor,
                                         ggml_tensor *               dst,
                                         ggml_tensor *               mul_tensor,
                                         char *                      q8_emit     = nullptr,
                                         int64_t                     q8_npad     = 0,
                                         bool                        q8_swizzle  = false,
                                         bool                        q8_barrier  = false);


void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
