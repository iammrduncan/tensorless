#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion = nullptr, bool convrot = false);

// P100 v2 K-B (mmvh-sm60.cu): sm_60 IQ3_XXS GEMV, opt-in GGML_CUDA_MMV_SM60=1. Returns false to keep the stock path.
bool ggml_cuda_mmvh_sm60_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// PN types 124/125 (pn-sm60.cu): decode GEMV over the packed PN blocks, ncols 1..4, or the fp32
// reference kernel with GGML_CUDA_PN_REF=1. Returns false for anything else (dequant + cuBLAS takes over).
bool ggml_cuda_pn_sm60_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// P3.16: expose the IQ3_S repack side buffer so the STOCK mmvq path can read it too. Returns the base
// for (data, k, m) when the tensor has already been repacked (the K-B prompt pass builds them), else
// nullptr; *row_bytes gets k/256*SB_BYTES_S_REP. Same data, different layout -> bit-identical.
void ggml_cuda_mmvh_sm60_rep_erase(const void * addr, size_t size);
const void * ggml_cuda_mmvh_sm60_iq3s_rep_lookup(const void * data, int64_t ne0, int64_t ne1, int64_t * row_bytes);
const void * ggml_cuda_mmvh_sm60_iq3xxs_rep_lookup(const void * data, int64_t ne0, int64_t ne1, int64_t * row_bytes);

// B-06 follow-up: draft-head repack - the MUL_MAT_ID one-row-expert path for the MTP draft head
// (fixed id subset of the Q4_K lm_head) gathers the selected rows once and runs the fp16 head
// kernel. Draft logits only; opt in with GGML_CUDA_MMV_SM60_Q4K_DHEAD=1. Returns false to keep
// the stock indexed-rows path.
bool ggml_cuda_mmvh_sm60_q4k_dhead(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                   const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// Runs one batched matvec over slot-pool experts selected by device-side IDs.
// Output row c is W[ids[c]] times activation row act_ids[c].
// slot_stride_bytes is the original tensor nb[2], and act_q8 uses the padded q8_1 layout from quantize_row_q8_1_cuda.
void ggml_cuda_moe_cache_mmv(
    const void * pool, ggml_type type0, const char * act_q8,
    const int32_t * ids_dev, const int32_t * act_ids_dev,
    float * dst_dev, int64_t n_in, int64_t n_out, int64_t n_slots,
    int64_t slot_stride_bytes, int64_t n_hits, int64_t act_rows, cudaStream_t stream);

// Runs up * GLU(gate) for rows whose two independently cached experts are resident.
void ggml_cuda_moe_cache_mmv_fused(
    const void * up_pool, const void * gate_pool, ggml_type type0,
    const char * act_q8, const int32_t * up_ids_dev,
    const int32_t * gate_ids_dev, const int32_t * act_ids_dev,
    float * dst_dev, int64_t n_in, int64_t n_out,
    int64_t slot_stride_bytes, int64_t n_hits, int64_t act_rows,
    float up_min, float up_max, float gate_min, float gate_max,
    cudaStream_t stream);
