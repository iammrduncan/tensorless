#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

// P100 v2 K-A (fattn-sm60.cu): split-KV flash-decoding for cc 600, q4_0 K/V, D=256, GQA 6, 1..4 query tokens.
// Opt-in via GGML_CUDA_FATTN_SM60=1.
bool ggml_cuda_fattn_sm60_applicable(int device, const ggml_tensor * dst);
bool ggml_cuda_fattn_sm60(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
