#include "moe-weighted-reduction.cuh"

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}

// Decode tail of build_moe_ffn() for eight experts (n_tokens == 1): dst[col] = sum_k(experts[k][col] * weights[k]),
// plus the residual when one is given. Each product is rounded once (__fmul_rn). The sum is the 32-lane xor
// butterfly that sum_rows runs over an eight-element row: lanes 8..31 hold +0 there, so only the offsets 4, 2 and 1
// change a value, and leaves start at 0 + product as reduce_row_f32() starts its accumulators at zero. The residual
// is added last, which is the graph's own order of the add (f32 add is commutative).
// Block (8, 32): threadIdx.x is the expert and 8 lanes make one column, so a warp holds four columns.
template <bool has_residual>
static __global__ void moe_tail_decode_k8(const float * __restrict__ experts,
                                          const float * __restrict__ weights,
                                          const float * __restrict__ residual,
                                          float * __restrict__ dst,
                                          const int64_t n_embd) {
    const int     k   = threadIdx.x;
    const int64_t col = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    float v = 0.0f + __fmul_rn(experts[k * n_embd + col], weights[k]);
    v += __shfl_xor_sync(0xffffffff, v, 4, 8);
    v += __shfl_xor_sync(0xffffffff, v, 2, 8);
    v += __shfl_xor_sync(0xffffffff, v, 1, 8);

    if (k == 0) {
        dst[col] = has_residual ? v + residual[col] : v;
    }
}

void ggml_cuda_op_moe_tail_decode(ggml_backend_cuda_context & ctx,
                                  const ggml_tensor *         experts,
                                  const ggml_tensor *         weights,
                                  const ggml_tensor *         residual,
                                  ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32 && weights->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(experts->ne[1] == 8 && experts->ne[2] == 1 && experts->ne[3] == 1);
    GGML_ASSERT(experts->ne[0] % 32 == 0);
    GGML_ASSERT(ggml_is_contiguous(experts) && ggml_is_contiguous(weights) && ggml_is_contiguous(dst));

    const int64_t n_embd = experts->ne[0];
    const dim3    block_dims(8, 32, 1);
    const dim3    block_nums((unsigned int) (n_embd / 32), 1, 1);
    cudaStream_t  stream = ctx.stream();

    if (residual != nullptr) {
        moe_tail_decode_k8<true><<<block_nums, block_dims, 0, stream>>>(
            (const float *) experts->data, (const float *) weights->data, (const float *) residual->data,
            (float *) dst->data, n_embd);
    } else {
        moe_tail_decode_k8<false><<<block_nums, block_dims, 0, stream>>>(
            (const float *) experts->data, (const float *) weights->data, nullptr,
            (float *) dst->data, n_embd);
    }
    CUDA_CHECK(cudaGetLastError());
}
