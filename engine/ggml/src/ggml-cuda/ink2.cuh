#include "common.cuh"

void ggml_cuda_op_ink2_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_ink2_rel_bias(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_ink2_rht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_ink2_glue(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
