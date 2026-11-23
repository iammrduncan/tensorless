#include "common.cuh"

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst);

// Decode tail (n_tokens == 1, eight experts): dst = sum_k(experts[k] * weights[k]) + residual, residual may be null.
void ggml_cuda_op_moe_tail_decode(ggml_backend_cuda_context & ctx,
                                  const ggml_tensor *         experts,
                                  const ggml_tensor *         weights,
                                  const ggml_tensor *         residual,
                                  ggml_tensor *               dst);
