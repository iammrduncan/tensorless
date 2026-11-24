// GGML_OP_INK2_GLUE: the Inkling-2 decode glue (n_tokens <= 8), each sub-op one kernel in place of the chain of small
// ggml ops it replaces. Every sub-op computes the same f32 math as its chain (rms norms as 1/sqrt(mean + eps), the
// same tanh/sigmoid forms), not the same reduction order: the results differ from the chains at rounding level.

#include "ink2.cuh"

#define INK2_GLUE_T     8 // most tokens
#define INK2_MIX_S      8 // most states of a depth mix
#define INK2_MIX_Q      4 // most queries of a depth mix

static __device__ __forceinline__ float ink2_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// ---- clip: y = (x * s) * min(1, thr / (|x * s| + 1e-6)) [* tau[t]], one warp per row of d <= 128
static __global__ void k_ink2_clip(const float * x, const float * sc, int sc_n, const float * tau, float * y,
                                   int d, int H, int rows, float thr) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (row >= rows) {
        return;
    }
    const int h = row % H, t = row / H;
    const float * xr = x + (int64_t) row * d;
    float v[4];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int k = lane + WARP_SIZE * j;
        v[j] = 0.0f;
        if (k < d) {
            v[j] = sc ? xr[k] * sc[sc_n == d ? k : h * d + k] : xr[k];
        }
        ss += v[j] * v[j];
    }
    ss = warp_reduce_sum(ss);
    const float nrm = sqrtf(ss) * 1.0f + 1e-6f;
    const float f = fminf(thr / nrm, 1.0f);
    float * yr = y + (int64_t) row * d;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int k = lane + WARP_SIZE * j;
        if (k < d) {
            yr[k] = tau ? (v[j] * f) * tau[t] : v[j] * f;
        }
    }
}

// ---- depth mix: out[q][t] = sum_s softmax_s(q . norm(s_t) / sqrt(n)) norm(s_t), result [n, T, n_q]
struct ink2_mix_args {
    const float * s[INK2_MIX_S];
    int64_t       st[INK2_MIX_S]; // row stride of each state, floats
};

static __global__ void __launch_bounds__(512) k_ink2_mix(const ink2_mix_args a, const float * q, float * y, int n, int T,
                                                         int S, int NQ, float eps, float scale) {
    constexpr int NP = INK2_MIX_S + INK2_MIX_S * INK2_MIX_Q;
    const int t = blockIdx.y;
    float part[NP];
#pragma unroll
    for (int p = 0; p < NP; ++p) {
        part[p] = 0.0f;
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float qv[INK2_MIX_Q];
#pragma unroll
        for (int k = 0; k < INK2_MIX_Q; ++k) {
            qv[k] = k < NQ ? q[k * n + i] : 0.0f;
        }
#pragma unroll
        for (int s = 0; s < INK2_MIX_S; ++s) {
            if (s < S) {
                const float v = a.s[s][t * a.st[s] + i];
                part[s] += v * v;
#pragma unroll
                for (int k = 0; k < INK2_MIX_Q; ++k) {
                    part[INK2_MIX_S + s * INK2_MIX_Q + k] += v * qv[k];
                }
            }
        }
    }
    __shared__ float red[512 / WARP_SIZE][NP];
    __shared__ float res[NP];
    __shared__ float w[INK2_MIX_Q][INK2_MIX_S];
    __shared__ float rinv[INK2_MIX_S];
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int n_warps = blockDim.x / WARP_SIZE;
#pragma unroll
    for (int p = 0; p < NP; ++p) {
        const float v = warp_reduce_sum(part[p]);
        if (lane == 0) {
            red[warp][p] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x < NP) {
        float v = 0.0f;
        for (int k = 0; k < n_warps; ++k) {
            v += red[k][threadIdx.x];
        }
        res[threadIdx.x] = v;
    }
    __syncthreads();
    if (threadIdx.x < S) {
        rinv[threadIdx.x] = rsqrtf(res[threadIdx.x] / n + eps);
    }
    __syncthreads();
    if (threadIdx.x < NQ) {
        const int k = threadIdx.x;
        float sc[INK2_MIX_S];
        float mx = -INFINITY;
        for (int s = 0; s < S; ++s) {
            sc[s] = res[INK2_MIX_S + s * INK2_MIX_Q + k] * rinv[s] * scale;
            mx = fmaxf(mx, sc[s]);
        }
        float sum = 0.0f;
        for (int s = 0; s < S; ++s) {
            sc[s] = expf(sc[s] - mx);
            sum += sc[s];
        }
        for (int s = 0; s < S; ++s) {
            w[k][s] = sc[s] / sum;
        }
    }
    __syncthreads();
    const int chunk = (n + gridDim.x - 1) / gridDim.x;
    const int i1 = min(n, (int) (blockIdx.x + 1) * chunk);
    for (int i = blockIdx.x * chunk + threadIdx.x; i < i1; i += blockDim.x) {
        float v[INK2_MIX_S];
#pragma unroll
        for (int s = 0; s < INK2_MIX_S; ++s) {
            v[s] = s < S ? a.s[s][t * a.st[s] + i] * rinv[s] : 0.0f;
        }
        for (int k = 0; k < NQ; ++k) {
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < INK2_MIX_S; ++s) {
                if (s < S) {
                    acc += w[k][s] * v[s];
                }
            }
            y[((int64_t) k * T + t) * n + i] = acc;
        }
    }
}

// ---- control gate, part 1: split-K partials of g[r, t] = down[r] . [cur_t, msg_t] and of the sums of squares of
// cur_t and msg_t; one warp per (row, chunk). Per token the result holds dc [R][S], dm [R][S], ssc [S], ssm [S].
#define INK2_CTL_S 16
static __global__ void __launch_bounds__(128) k_ink2_ctl_part(const float * cur, int64_t cs, const float * msg, int64_t ms,
                                                              const float * down, float * P, int n, int R, int T) {
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int S = gridDim.x, chunk = blockIdx.x;
    const int clen = (n + S - 1) / S;
    const int i0 = chunk * clen, i1 = min(n, i0 + clen);
    const int64_t PT = 2 * (int64_t) R * S + 2 * S;
    const int r = blockIdx.y * (blockDim.x / WARP_SIZE) + warp;
    if (r < R) {
        const float * dr = down + (int64_t) r * 2 * n;
        float ac[INK2_GLUE_T], am[INK2_GLUE_T];
#pragma unroll
        for (int t = 0; t < INK2_GLUE_T; ++t) {
            ac[t] = 0.0f;
            am[t] = 0.0f;
        }
        for (int i = i0 + lane; i < i1; i += WARP_SIZE) {
            const float wc = dr[i], wm = dr[n + i];
#pragma unroll
            for (int t = 0; t < INK2_GLUE_T; ++t) {
                if (t < T) {
                    ac[t] += wc * cur[t * cs + i];
                    am[t] += wm * msg[t * ms + i];
                }
            }
        }
#pragma unroll
        for (int t = 0; t < INK2_GLUE_T; ++t) {
            if (t < T) {
                const float vc = warp_reduce_sum(ac[t]);
                const float vm = warp_reduce_sum(am[t]);
                if (lane == 0) {
                    P[t * PT + r * S + chunk] = vc;
                    P[t * PT + (int64_t) R * S + r * S + chunk] = vm;
                }
            }
        }
    }
    if (blockIdx.y == 0 && warp == 0) {
        float sc[INK2_GLUE_T], sm[INK2_GLUE_T];
#pragma unroll
        for (int t = 0; t < INK2_GLUE_T; ++t) {
            sc[t] = 0.0f;
            sm[t] = 0.0f;
        }
        for (int i = i0 + lane; i < i1; i += WARP_SIZE) {
#pragma unroll
            for (int t = 0; t < INK2_GLUE_T; ++t) {
                if (t < T) {
                    const float c = cur[t * cs + i], m = msg[t * ms + i];
                    sc[t] += c * c;
                    sm[t] += m * m;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < INK2_GLUE_T; ++t) {
            if (t < T) {
                const float vc = warp_reduce_sum(sc[t]);
                const float vm = warp_reduce_sum(sm[t]);
                if (lane == 0) {
                    P[t * PT + 2 * (int64_t) R * S + chunk] = vc;
                    P[t * PT + 2 * (int64_t) R * S + S + chunk] = vm;
                }
            }
        }
    }
}

// ---- control gate, part 2: out = cur - tanh(forget . g) cur + tanh(gain . g) msg, one warp per output row
static __global__ void __launch_bounds__(512) k_ink2_ctl_blend(const float * cur, int64_t cs, const float * msg, int64_t ms,
                                                               const float * P, int S, const float * gain,
                                                               const float * forget, float * y, int n, int R, int T,
                                                               float eps) {
    extern __shared__ float gs[]; // [T, R]
    __shared__ float rn[2 * INK2_GLUE_T];
    const int64_t PT = 2 * (int64_t) R * S + 2 * S;
    if (threadIdx.x < 2 * T) {
        const int t = threadIdx.x / 2, m = threadIdx.x % 2;
        const float * ss = P + t * PT + 2 * (int64_t) R * S + m * S;
        float v = 0.0f;
        for (int k = 0; k < S; ++k) {
            v += ss[k];
        }
        rn[threadIdx.x] = rsqrtf(v / n + eps);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < R * T; j += blockDim.x) {
        const int t = j / R, r = j % R;
        const float * dc = P + t * PT + r * S;
        const float * dm = dc + (int64_t) R * S;
        float vc = 0.0f, vm = 0.0f;
        for (int k = 0; k < S; ++k) {
            vc += dc[k];
            vm += dm[k];
        }
        gs[j] = rn[2 * t] * vc + rn[2 * t + 1] * vm;
    }
    __syncthreads();
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int i = blockIdx.x * (blockDim.x / WARP_SIZE) + warp;
    if (i >= n) {
        return;
    }
    const float * gr = gain + (int64_t) i * R;
    const float * fr = forget + (int64_t) i * R;
    float ga[INK2_GLUE_T], fa[INK2_GLUE_T];
#pragma unroll
    for (int t = 0; t < INK2_GLUE_T; ++t) {
        ga[t] = 0.0f;
        fa[t] = 0.0f;
    }
    for (int k = lane; k < R; k += WARP_SIZE) {
        const float wg = gr[k], wf = fr[k];
#pragma unroll
        for (int t = 0; t < INK2_GLUE_T; ++t) {
            if (t < T) {
                ga[t] += wg * gs[t * R + k];
                fa[t] += wf * gs[t * R + k];
            }
        }
    }
#pragma unroll
    for (int t = 0; t < INK2_GLUE_T; ++t) {
        if (t < T) {
            const float gv = tanhf(warp_reduce_sum(ga[t]));
            const float fv = tanhf(warp_reduce_sum(fa[t]));
            if (lane == 0) {
                const float c = cur[t * cs + i];
                y[(int64_t) t * n + i] = (c - fv * c) + gv * msg[t * ms + i];
            }
        }
    }
}

// ---- relative-bias table: out[x, h, t] = (x < ext ? proj[:, x] . r[:, h, t] : 0) - sink[h, t]
static __global__ void k_ink2_table(const float * proj, const float * r, const float * sink, float * y, int d_rel, int ext,
                                    int HT) {
    extern __shared__ float rs[]; // [d_rel]
    const int ht = blockIdx.y;
    for (int d = threadIdx.x; d < d_rel; d += blockDim.x) {
        rs[d] = r[(int64_t) ht * d_rel + d];
    }
    __syncthreads();
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x > ext) {
        return;
    }
    float acc = 0.0f;
    if (x < ext) {
        const float * pr = proj + (int64_t) x * d_rel;
        if (d_rel == 16) {
            float4 p[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                p[j] = __ldg((const float4 *) pr + j);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc += p[j].x * rs[4 * j] + p[j].y * rs[4 * j + 1] + p[j].z * rs[4 * j + 2] + p[j].w * rs[4 * j + 3];
            }
        } else {
            for (int d = 0; d < d_rel; ++d) {
                acc += pr[d] * rs[d];
            }
        }
    }
    y[(int64_t) ht * (ext + 1) + x] = acc - sink[ht];
}

// ---- soft-capped SwiGLU: softcap(g, 4) * sigmoid(g) * softcap(u, 25)
static __global__ void k_ink2_swiglu(const float * gu, float * y, int n_ff, int64_t n) {
    const int64_t j = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) {
        return;
    }
    const int64_t col = j / n_ff, i = j - col * n_ff;
    const float g = gu[col * 2 * n_ff + i], u = gu[col * 2 * n_ff + n_ff + i];
    const float gc = tanhf(g * 0.25f) * 4.0f;
    const float uc = tanhf(u * (1.0f / 25.0f)) * 25.0f;
    y[j] = (gc * ink2_sigmoid(g)) * uc;
}

// ---- router: the n_used experts of largest sigmoid(logit) + bias, in descending order, one warp per token
#define INK2_TOPK_PER_LANE 8
static __global__ void k_ink2_topk(const float * logits, const float * bias, int32_t * ids, int n_exp, int n_used, int T) {
    const int t = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (t >= T) {
        return;
    }
    float v[INK2_TOPK_PER_LANE];
#pragma unroll
    for (int j = 0; j < INK2_TOPK_PER_LANE; ++j) {
        const int e = j * WARP_SIZE + lane;
        v[j] = e < n_exp ? ink2_sigmoid(logits[(int64_t) t * n_exp + e]) + bias[e] : -INFINITY;
    }
    for (int k = 0; k < n_used; ++k) {
        float best = -INFINITY;
        int bi = INT_MAX;
#pragma unroll
        for (int j = 0; j < INK2_TOPK_PER_LANE; ++j) {
            const int e = j * WARP_SIZE + lane;
            if (e < n_exp && (v[j] > best || (v[j] == best && e < bi))) {
                best = v[j];
                bi = e;
            }
        }
#pragma unroll
        for (int o = WARP_SIZE / 2; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffff, best, o);
            const int   oi = __shfl_xor_sync(0xffffffff, bi, o);
            if (ob > best || (ob == best && oi < bi)) {
                best = ob;
                bi = oi;
            }
        }
        if (bi % WARP_SIZE == lane) {
#pragma unroll
            for (int j = 0; j < INK2_TOPK_PER_LANE; ++j) {
                if (j == bi / WARP_SIZE) {
                    v[j] = -INFINITY;
                }
            }
        }
        if (lane == 0) {
            ids[t * n_used + k] = bi;
        }
    }
}

// ---- expert outputs times their normalized router weights, summed over the slots
static __global__ void k_ink2_slotsum(const float * dn, const float * logits, const int32_t * ids, float * y, int n,
                                      int n_used, int n_exp, float scale) {
    __shared__ float wt[32];
    const int t = blockIdx.y;
    if (threadIdx.x == 0) {
        float p[32];
        float sum = 0.0f;
        for (int s = 0; s < n_used; ++s) {
            p[s] = ink2_sigmoid(logits[(int64_t) t * n_exp + ids[t * n_used + s]]);
            sum += p[s];
        }
        sum = sum * 1.0f + 1e-20f;
        for (int s = 0; s < n_used; ++s) {
            wt[s] = (p[s] / sum) * scale;
        }
    }
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float * d = dn + (int64_t) t * n_used * n + i;
    float acc = d[0] * wt[0];
    for (int s = 1; s < n_used; ++s) {
        acc += d[(int64_t) s * n] * wt[s];
    }
    y[(int64_t) t * n + i] = acc;
}

// ---- causal 4-tap convolution with residual, dynamic per-head taps, and the state snapshots; one thread a channel
struct ink2_conv_args {
    float * snap[4];
};

static __global__ void k_ink2_conv(const float * cur, const float * past, int64_t ps, const float * kern,
                                   const float * dyn, int n_heads, const ink2_conv_args sa, int n_snap, float * y,
                                   int C, int T, float scale) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) {
        return;
    }
    float x[3 + INK2_GLUE_T];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        x[j] = past[c * ps + j];
    }
#pragma unroll
    for (int t = 0; t < INK2_GLUE_T; ++t) {
        x[3 + t] = t < T ? cur[(int64_t) t * C + c] : 0.0f;
    }
    float k[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        k[j] = kern[c * 4 + j];
    }
    const int h = dyn ? c / (C / n_heads) : 0;
#pragma unroll
    for (int t = 0; t < INK2_GLUE_T; ++t) {
        if (t < T) {
            float o = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                o += x[t + j] * k[j];
            }
            if (dyn) {
                const float * dt = dyn + ((int64_t) t * n_heads + h) * 4;
#pragma unroll
                for (int lag = 0; lag < 4; ++lag) {
                    o += x[t + 3 - lag] * dt[lag];
                }
            }
            y[(int64_t) t * C + c] = (x[3 + t] + o) * scale;
        }
    }
    for (int s = 0; s < n_snap; ++s) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            float v = 0.0f;
#pragma unroll
            for (int q = 0; q < 3 + INK2_GLUE_T; ++q) {
                if (q == T - s + j) {
                    v = x[q];
                }
            }
            sa.snap[s][c * 3 + j] = v;
        }
    }
}

static int64_t ink2_row_stride(const ggml_tensor * t) {
    GGML_ASSERT(t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float));
    return t->ne[1] > 1 ? (int64_t) (t->nb[1] / sizeof(float)) : t->ne[0];
}

void ggml_cuda_op_ink2_glue(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int op = ggml_get_op_params_i32(dst, 0);
    const int i0 = ggml_get_op_params_i32(dst, 1);
    const float f0 = ggml_get_op_params_f32(dst, 2);
    cudaStream_t stream = ctx.stream();
    ggml_tensor ** src = dst->src;
    float * y = (float *) dst->data;

    switch (op) {
        case GGML_INK2_GLUE_CLIP: {
            const ggml_tensor * x = src[0];
            GGML_ASSERT(ggml_is_contiguous(x) && x->ne[0] <= 4 * WARP_SIZE);
            const int d = x->ne[0], H = x->ne[1];
            const int rows = (int) (ggml_nelements(x) / d);
            const float * sc = src[1] ? (const float *) src[1]->data : nullptr;
            const int sc_n = src[1] ? (int) ggml_nelements(src[1]) : 0;
            GGML_ASSERT(!sc || sc_n == d || sc_n == d * H);
            const float * tau = src[2] ? (const float *) src[2]->data : nullptr;
            const int threads = 256, rpb = threads / WARP_SIZE;
            k_ink2_clip<<<(rows + rpb - 1) / rpb, threads, 0, stream>>>((const float *) x->data, sc, sc_n, tau, y, d, H,
                                                                        rows, f0);
        } break;
        case GGML_INK2_GLUE_MIX: {
            const ggml_tensor * q = src[0];
            const int n = q->ne[0];
            const int NQ = (int) (ggml_nelements(q) / n);
            GGML_ASSERT(ggml_is_contiguous(q) && NQ <= INK2_MIX_Q);
            ink2_mix_args a = {};
            int S = 0;
            for (int i = 1; i < GGML_MAX_SRC && src[i]; ++i) {
                GGML_ASSERT(S < INK2_MIX_S && src[i]->ne[0] == n);
                a.s[S]  = (const float *) src[i]->data;
                a.st[S] = ink2_row_stride(src[i]);
                ++S;
            }
            const int T = dst->ne[1];
            GGML_ASSERT(T <= INK2_GLUE_T);
            k_ink2_mix<<<dim3(8, T), 512, 0, stream>>>(a, (const float *) q->data, y, n, T, S, NQ, 1e-6f,
                                                       1.0f / sqrtf((float) n));
        } break;
        case GGML_INK2_GLUE_CTL_DOWN: {
            const ggml_tensor * cur = src[0], * msg = src[1], * down = src[2];
            const int n = cur->ne[0], T = cur->ne[1], R = down->ne[1];
            GGML_ASSERT(T <= INK2_GLUE_T && down->ne[0] == 2 * n && ggml_is_contiguous(down));
            GGML_ASSERT(dst->ne[0] == 2 * R * INK2_CTL_S + 2 * INK2_CTL_S);
            const int wpb = 4;
            k_ink2_ctl_part<<<dim3(INK2_CTL_S, (R + wpb - 1) / wpb), wpb * WARP_SIZE, 0, stream>>>(
                (const float *) cur->data, ink2_row_stride(cur), (const float *) msg->data, ink2_row_stride(msg),
                (const float *) down->data, y, n, R, T);
        } break;
        case GGML_INK2_GLUE_CTL_BLEND: {
            const ggml_tensor * cur = src[0], * msg = src[1], * g = src[2], * gain = src[3], * forget = src[4];
            const int n = cur->ne[0], T = cur->ne[1], R = gain->ne[0];
            GGML_ASSERT(T <= INK2_GLUE_T && forget->ne[0] == R && gain->ne[1] == n);
            GGML_ASSERT(g->ne[0] == 2 * R * INK2_CTL_S + 2 * INK2_CTL_S);
            GGML_ASSERT(ggml_is_contiguous(g) && ggml_is_contiguous(gain) && ggml_is_contiguous(forget));
            const int wpb = 16;
            k_ink2_ctl_blend<<<(n + wpb - 1) / wpb, wpb * WARP_SIZE, R * T * sizeof(float), stream>>>(
                (const float *) cur->data, ink2_row_stride(cur), (const float *) msg->data, ink2_row_stride(msg),
                (const float *) g->data, INK2_CTL_S, (const float *) gain->data, (const float *) forget->data, y, n, R,
                T, 1e-6f);
        } break;
        case GGML_INK2_GLUE_SWIGLU: {
            const ggml_tensor * gu = src[0];
            GGML_ASSERT(ggml_is_contiguous(gu));
            const int n_ff = dst->ne[0];
            const int64_t n = ggml_nelements(dst);
            k_ink2_swiglu<<<(n + 255) / 256, 256, 0, stream>>>((const float *) gu->data, y, n_ff, n);
        } break;
        case GGML_INK2_GLUE_TOPK: {
            const ggml_tensor * logits = src[0], * bias = src[1];
            const int n_exp = logits->ne[0], T = logits->ne[1];
            GGML_ASSERT(ggml_is_contiguous(logits) && n_exp <= INK2_TOPK_PER_LANE * WARP_SIZE && i0 <= n_exp);
            const int wpb = 4;
            k_ink2_topk<<<(T + wpb - 1) / wpb, wpb * WARP_SIZE, 0, stream>>>(
                (const float *) logits->data, (const float *) bias->data, (int32_t *) dst->data, n_exp, i0, T);
        } break;
        case GGML_INK2_GLUE_SLOTSUM: {
            const ggml_tensor * dn = src[0], * logits = src[1], * ids = src[2];
            const int n = dn->ne[0], n_used = dn->ne[1], T = dn->ne[2];
            GGML_ASSERT(ggml_is_contiguous(dn) && n_used <= 32 && ids->ne[0] == n_used);
            k_ink2_slotsum<<<dim3((n + 255) / 256, T), 256, 0, stream>>>(
                (const float *) dn->data, (const float *) logits->data, (const int32_t *) ids->data, y, n, n_used,
                (int) logits->ne[0], f0);
        } break;
        case GGML_INK2_GLUE_CONV: {
            const ggml_tensor * cur = src[0], * past = src[1], * kern = src[2], * dyn = src[3];
            const int C = cur->ne[0], T = cur->ne[1];
            GGML_ASSERT(ggml_is_contiguous(cur) && T <= INK2_GLUE_T && past->nb[0] == sizeof(float));
            GGML_ASSERT(kern->ne[0] == 4 && kern->ne[1] == C && ggml_is_contiguous(kern));
            GGML_ASSERT(i0 >= 0 && i0 <= 4 && i0 <= T);
            ink2_conv_args sa = {};
            for (int s = 0; s < i0; ++s) {
                sa.snap[s] = (float *) src[4 + s]->data;
            }
            const int n_heads = dyn ? (int) (dyn->ne[0] / 4) : 1;
            GGML_ASSERT(!dyn || (ggml_is_contiguous(dyn) && C % n_heads == 0));
            k_ink2_conv<<<(C + 127) / 128, 128, 0, stream>>>(
                (const float *) cur->data, (const float *) past->data, (int64_t) (past->nb[1] / sizeof(float)),
                (const float *) kern->data, dyn ? (const float *) dyn->data : nullptr, n_heads, sa, i0, y, C, T, f0);
        } break;
        case GGML_INK2_GLUE_TABLE: {
            const ggml_tensor * proj = src[0], * r = src[1], * sink = src[2];
            const int d_rel = proj->ne[0], ext = proj->ne[1];
            const int HT = (int) (ggml_nelements(r) / d_rel);
            GGML_ASSERT(ggml_is_contiguous(proj) && ggml_is_contiguous(r) && ggml_is_contiguous(sink));
            GGML_ASSERT(r->ne[0] == d_rel && ggml_nelements(sink) == HT && dst->ne[0] == ext + 1);
            GGML_ASSERT(d_rel != 16 || (uintptr_t) proj->data % 16 == 0);
            k_ink2_table<<<dim3((ext + 1 + 255) / 256, HT), 256, d_rel * sizeof(float), stream>>>(
                (const float *) proj->data, (const float *) r->data, (const float *) sink->data, y, d_rel, ext, HT);
        } break;
        default:
            GGML_ABORT("ink2_glue: unknown sub-op %d", op);
    }
    CUDA_CHECK(cudaGetLastError());
}
