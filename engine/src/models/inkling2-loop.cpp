// Inkling-2-Mini (inkling2_loop) from LookAtMyQuant's packed format (inkling2-packed-v1).
//
// The weights stay in the pack's files: tools/inkling-prepare/prepare.py writes a GGUF that names every array by file
// and offset, and load_arch_tensors() memory-maps them and uploads them as they are (codes, f16 scales, tables), so
// the packed matrices are multiplied in their stored, rotated basis by GGML_OP_INK2_MUL_MAT and the randomized
// Hadamard rotations are applied as signs * (A (x) B) with the constant Sylvester and Paley factors in fp32.
// The embedding is decoded and rotated back once at load, kept on the host in f32.
//
// The graph follows transformers' modeling_inkling2_loop.py and upstream llama.cpp's inkling2(-loop).cpp: 48 cache
// slots (24 layers, then layers 6-17 for passes 2 and 3), message gates, block attention residuals, short causal
// convolutions with their state in the recurrent memory, a dense relative-position KQ bias with per-token sinks, and
// the query rows of the full-attention slots processed in chunks so that the bias and KQ stay small at 32K.

#include "models.h"

#include "../llama-memory-hybrid-iswa.h"
#include "../llama-kv-cache.h"
#include "../llama-kv-cache-iswa.h"
#include "../llama-memory-recurrent.h"
#include "ggml-inkling2.h"
#include "ggml-cpp.h"

#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

constexpr int INK2_N_PHYS     = 24;
constexpr int INK2_CORE_START = 6;
constexpr int INK2_CORE_END   = 18;
constexpr int INK2_N_PASSES   = 3;
constexpr int INK2_N_SLOTS    = INK2_N_PHYS + (INK2_N_PASSES - 1) * (INK2_CORE_END - INK2_CORE_START);
constexpr int INK2_BLOCK      = 6;
constexpr int INK2_CONV       = 4;
// conv state channels per slot: K (1024), V (1024), attention output (2304), MoE output (2304)
constexpr int INK2_CONV_CH    = 1024 + 1024 + 2304 + 2304;

static int ink2_layer_phys(int il) {
    return il < INK2_N_PHYS ? il : INK2_CORE_START + (il - INK2_N_PHYS) % (INK2_CORE_END - INK2_CORE_START);
}

struct ink2_lin {
    int fmt = -1, m = 0, n = 0;
    ggml_tensor * codes = nullptr, * scales = nullptr, * table = nullptr, * signs_in = nullptr, * signs_out = nullptr;
};

struct ink2_slab {
    int fmt = -1;
    ggml_tensor * codes = nullptr, * scales = nullptr, * table = nullptr, * aux = nullptr;
};

struct ink2_side {
    ggml_tensor * map = nullptr, * signs = nullptr;
    std::vector<ink2_slab> slabs;
};

struct ink2_gate {
    ggml_tensor * down = nullptr, * forget = nullptr, * gain = nullptr;
};

struct ink2_layer {
    bool swa = true;
    ink2_lin q, k, v, o, og, rel;
    ggml_tensor * attn_res_query = nullptr, * ffn_res_query = nullptr;
    ggml_tensor * attn_norm = nullptr, * q_norm = nullptr, * k_norm = nullptr, * v_norm = nullptr, * ffn_norm = nullptr;
    ggml_tensor * post_attn_conv = nullptr, * post_ffn_conv = nullptr;
    ggml_tensor * k_conv = nullptr, * v_conv = nullptr, * k_conv_proj = nullptr, * v_conv_proj = nullptr;
    ggml_tensor * q_scale = nullptr, * rel_scale = nullptr, * rel_proj = nullptr, * sink = nullptr;
    ggml_tensor * router = nullptr, * router_bias = nullptr;
    ink2_side gu, dn;
    ink2_gate gates[4]; // q, k, v, ffn (loop layers)
};

struct ink2_hadamard {
    int a = 0, b = 0;
    ggml_tensor * A = nullptr, * At = nullptr, * B = nullptr, * Bt = nullptr;
};

struct ink2_weights {
    std::vector<ink2_layer> layers;
    ggml_tensor * embed_norm = nullptr, * final_query = nullptr, * out_norm = nullptr, * msg_query = nullptr;
    std::vector<ink2_lin> head;
    ggml_tensor * head_order = nullptr;
    std::map<int, ink2_hadamard> had;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;
    std::vector<std::pair<void *, size_t>> maps;

    ~ink2_weights() {
        for (auto & m : maps) {
            munmap(m.first, m.second);
        }
    }
};

namespace {

// ---------------------------------------------------------------------------------------------- Hadamard factors

bool ink2_is_prime(int q) {
    if (q < 2) return false;
    for (int d = 2; d * d <= q; ++d) if (q % d == 0) return false;
    return true;
}

std::vector<double> ink2_sylvester(int a) {
    std::vector<double> H = {1.0};
    int n = 1;
    while (n < (1 << a)) {
        std::vector<double> G((size_t) 4 * n * n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const double h = H[(size_t) i * n + j];
                G[(size_t) i * 2 * n + j]           = h;
                G[(size_t) i * 2 * n + j + n]       = h;
                G[(size_t) (i + n) * 2 * n + j]     = h;
                G[(size_t) (i + n) * 2 * n + j + n] = -h;
            }
        }
        H.swap(G);
        n *= 2;
    }
    return H;
}

std::vector<double> ink2_paley(int q) {
    std::vector<int> sq(q, 0);
    for (int x = 1; x < q; ++x) sq[(x * x) % q] = 1;
    std::vector<double> chi(q);
    chi[0] = 0.0;
    for (int a = 1; a < q; ++a) chi[a] = sq[a] ? 1.0 : -1.0;
    const int n = q + 1;
    std::vector<double> S((size_t) n * n, 0.0);
    for (int j = 1; j < n; ++j) S[j] = 1.0;
    for (int i = 1; i < n; ++i) S[(size_t) i * n] = -1.0;
    for (int i = 1; i < n; ++i) {
        for (int j = 1; j < n; ++j) {
            S[(size_t) i * n + j] = chi[(((j - 1) - (i - 1)) % q + q) % q];
        }
    }
    for (int i = 0; i < n; ++i) S[(size_t) i * n + i] += 1.0;
    return S;
}

// formats.py's hadamard_factors_np: (A, B) row-major, H = A (x) B
void ink2_hadamard_factors(int n, std::vector<double> & A, int & a_n, std::vector<double> & B, int & b_n) {
    for (int a = (int) std::floor(std::log2((double) n)); a >= 0; --a) {
        if (n % (1 << a)) continue;
        const int p = n / (1 << a);
        if (p == 1 || p == 2 || (ink2_is_prime(p - 1) && (p - 1) % 4 == 3)) {
            if (p == 1 && a > 1) {
                A = ink2_sylvester(a / 2);     a_n = 1 << (a / 2);
                B = ink2_sylvester(a - a / 2); b_n = 1 << (a - a / 2);
                return;
            }
            A = ink2_sylvester(a); a_n = 1 << a;
            if (p == 2) {
                B = {1.0, 1.0, 1.0, -1.0}; b_n = 2;
            } else if (p == 1) {
                B = {1.0}; b_n = 1;
            } else {
                B = ink2_paley(p - 1); b_n = p;
            }
            return;
        }
    }
    throw std::runtime_error(format("inkling2: no Hadamard construction for n=%d", n));
}

// ---------------------------------------------------------------------------------------------- pack table

struct ink2_entry {
    int file;
    uint64_t offset, nbytes;
    std::string dtype;
    std::vector<int64_t> shape; // numpy order
};

struct ink2_upload {
    ggml_tensor * t;
    std::function<void(void *)> fill;
};

int64_t ink2_key(const gguf_context * ctx, const char * k) {
    const int64_t id = gguf_find_key(ctx, k);
    if (id < 0) {
        throw std::runtime_error(std::string("inkling2: key not found: ") + k);
    }
    return id;
}

float ink2_bf16(uint16_t h) {
    uint32_t b = (uint32_t) h << 16;
    float f;
    memcpy(&f, &b, 4);
    return f;
}

} // namespace

void llama_model_inkling2_loop::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
    type = LLM_TYPE_UNKNOWN;

    const gguf_context * ctx = ml.metadata;
    const int64_t ks = ink2_key(ctx, "inkling2_loop.swa_layers");
    if (gguf_get_arr_n(ctx, ks) != (size_t) INK2_N_PHYS || hparams.n_layer_all != (uint32_t) INK2_N_PHYS) {
        throw std::runtime_error("inkling2_loop: expected 24 layers");
    }
    const int8_t * swa = (const int8_t *) gguf_get_arr_data(ctx, ks);

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.rope_pattern.fill(false);
    hparams.n_layer_all = INK2_N_SLOTS;
    // the recurrent memory holds the convolution states of each slot: 3 taps x INK2_CONV_CH channels
    hparams.ssm_d_conv  = INK2_CONV;
    hparams.ssm_d_inner = INK2_CONV_CH;
    hparams.ssm_d_state = 1;
    hparams.ssm_n_group = 0;
    for (int il = 0; il < INK2_N_SLOTS; ++il) {
        const int p = ink2_layer_phys(il);
        hparams.n_head_arr[il]        = hparams.n_head_arr[0];
        hparams.n_head_kv_arr[il]     = hparams.n_head_kv_arr[0];
        hparams.n_ff_arr[il]          = hparams.n_ff_arr[0];
        hparams.n_ff_exp_arr[il]      = hparams.n_ff_exp_arr[0];
        hparams.n_expert_used_arr[il] = hparams.n_expert_used_arr[0];
        hparams.is_swa_impl[il]       = swa[p] ? 1 : 0;
        hparams.is_recr_impl[il]      = 1;
    }
}

void llama_model_inkling2_loop::load_arch_tensors(llama_model_loader & ml) {
    const gguf_context * g = ml.metadata;
    auto W = std::make_shared<ink2_weights>();

    // ---- the table
    const std::string dir = gguf_get_val_str(g, ink2_key(g, "inkling2.pack_dir"));
    std::vector<std::string> files;
    {
        const int64_t kf = ink2_key(g, "inkling2.files");
        for (size_t i = 0; i < gguf_get_arr_n(g, kf); ++i) files.push_back(gguf_get_arr_str(g, kf, i));
    }
    std::map<std::string, ink2_entry> table;
    {
        const int64_t kn = ink2_key(g, "inkling2.t.names");
        const size_t n = gguf_get_arr_n(g, kn);
        const uint32_t * tf = (const uint32_t *) gguf_get_arr_data(g, ink2_key(g, "inkling2.t.file"));
        const uint64_t * to = (const uint64_t *) gguf_get_arr_data(g, ink2_key(g, "inkling2.t.offset"));
        const uint64_t * tb = (const uint64_t *) gguf_get_arr_data(g, ink2_key(g, "inkling2.t.nbytes"));
        const uint32_t * nd = (const uint32_t *) gguf_get_arr_data(g, ink2_key(g, "inkling2.t.ndim"));
        const uint64_t * dims = (const uint64_t *) gguf_get_arr_data(g, ink2_key(g, "inkling2.t.dims"));
        const int64_t kd = ink2_key(g, "inkling2.t.dtype");
        size_t di = 0;
        for (size_t i = 0; i < n; ++i) {
            ink2_entry e;
            e.file = (int) tf[i];
            e.offset = to[i];
            e.nbytes = tb[i];
            e.dtype = gguf_get_arr_str(g, kd, i);
            for (uint32_t d = 0; d < nd[i]; ++d) e.shape.push_back((int64_t) dims[di++]);
            table[gguf_get_arr_str(g, kn, i)] = e;
        }
    }
    struct mat_info { std::string prefix; int fmt, rows, cols, out_rot; };
    std::map<std::string, mat_info> mats;
    {
        const int64_t km = ink2_key(g, "inkling2.m.names");
        const int64_t kp = ink2_key(g, "inkling2.m.prefix");
        const uint32_t * mf = (const uint32_t *) gguf_get_arr_data(g, ink2_key(g, "inkling2.m.fmt"));
        const uint32_t * mr = (const uint32_t *) gguf_get_arr_data(g, ink2_key(g, "inkling2.m.rows"));
        const uint32_t * mc = (const uint32_t *) gguf_get_arr_data(g, ink2_key(g, "inkling2.m.cols"));
        const uint32_t * mo = (const uint32_t *) gguf_get_arr_data(g, ink2_key(g, "inkling2.m.out_rot"));
        for (size_t i = 0; i < gguf_get_arr_n(g, km); ++i) {
            mats[gguf_get_arr_str(g, km, i)] = { gguf_get_arr_str(g, kp, i), (int) mf[i], (int) mr[i], (int) mc[i], (int) mo[i] };
        }
    }

    // ---- map the pack files
    std::vector<const uint8_t *> fmap(files.size(), nullptr);
    auto data_of = [&](const ink2_entry & e) -> const uint8_t * {
        if (!fmap[e.file]) {
            const std::string p = dir + "/" + files[e.file];
            const int fd = open(p.c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error("inkling2: cannot open " + p);
            struct stat st;
            fstat(fd, &st);
            void * m = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
            close(fd);
            if (m == MAP_FAILED) throw std::runtime_error("inkling2: cannot map " + p);
            W->maps.emplace_back(m, (size_t) st.st_size);
            fmap[e.file] = (const uint8_t *) m;
        }
        return fmap[e.file] + e.offset;
    };
    auto entry = [&](const std::string & name) -> const ink2_entry & {
        auto it = table.find(name);
        if (it == table.end()) throw std::runtime_error("inkling2: the pack has no array " + name);
        return it->second;
    };
    auto has = [&](const std::string & name) { return table.count(name) > 0; };

    // ---- device and host contexts
    ggml_backend_buffer_type_t buft_dev = ggml_backend_cpu_buffer_type();
    for (const auto & d : devices) {
        if (!d.is_meta && ggml_backend_dev_type(d.dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            buft_dev = ggml_backend_dev_buffer_type(d.dev);
            break;
        }
    }
    if (params.n_gpu_layers == 0) {
        buft_dev = ggml_backend_cpu_buffer_type();
    }
    const size_t n_max_tensors = 4096;
    ggml_init_params ip = { ggml_tensor_overhead() * n_max_tensors, nullptr, true };
    ggml_context * ctx_dev  = ggml_init(ip);
    ggml_context * ctx_host = ggml_init(ip);
    W->ctxs.emplace_back(ctx_dev);
    W->ctxs.emplace_back(ctx_host);
    std::vector<ink2_upload> uploads_dev, uploads_host;

    // an array as it is stored (bit patterns kept: u32 -> I32, u8 -> I8), ggml order = reversed numpy order
    auto raw = [&](const std::string & name) -> ggml_tensor * {
        const ink2_entry & e = entry(name);
        ggml_type t;
        if (e.dtype == "u32" || e.dtype == "i32") t = GGML_TYPE_I32;
        else if (e.dtype == "u8" || e.dtype == "i8") t = GGML_TYPE_I8;
        else if (e.dtype == "f16") t = GGML_TYPE_F16;
        else if (e.dtype == "f32") t = GGML_TYPE_F32;
        else if (e.dtype == "i16") t = GGML_TYPE_I16;
        else throw std::runtime_error("inkling2: unexpected dtype " + e.dtype + " of " + name);
        int64_t ne[4] = {1, 1, 1, 1};
        const int nd = (int) e.shape.size();
        for (int d = 0; d < nd; ++d) ne[d] = e.shape[nd - 1 - d];
        ggml_tensor * x = ggml_new_tensor(ctx_dev, t, std::max(1, nd), ne);
        ggml_set_name(x, name.c_str());
        GGML_ASSERT(ggml_nbytes(x) == e.nbytes);
        const ink2_entry * ep = &e;
        uploads_dev.push_back({x, [&, ep](void * dst) { memcpy(dst, data_of(*ep), ep->nbytes); }});
        return x;
    };
    // an array converted to f32 (bf16, f16, f32, i8 signs), optionally with a custom shape and element map
    auto f32 = [&](const std::string & name, std::vector<int64_t> ne_override = {},
                   std::function<int64_t(int64_t)> src_index = nullptr) -> ggml_tensor * {
        const ink2_entry & e = entry(name);
        int64_t n = 1;
        for (auto s : e.shape) n *= s;
        int64_t ne[4] = {1, 1, 1, 1};
        int nd = (int) e.shape.size();
        if (!ne_override.empty()) {
            nd = (int) ne_override.size();
            for (int d = 0; d < nd; ++d) ne[d] = ne_override[d];
        } else {
            for (int d = 0; d < nd; ++d) ne[d] = e.shape[nd - 1 - d];
        }
        ggml_tensor * x = ggml_new_tensor(ctx_dev, GGML_TYPE_F32, std::max(1, nd), ne);
        ggml_set_name(x, ("f32." + name).c_str());
        GGML_ASSERT(ggml_nelements(x) == n);
        const ink2_entry * ep = &e;
        uploads_dev.push_back({x, [&, ep, n, src_index](void * dst) {
            const uint8_t * src = data_of(*ep);
            float * out = (float *) dst;
            for (int64_t i = 0; i < n; ++i) {
                const int64_t s = src_index ? src_index(i) : i;
                float v;
                if (ep->dtype == "bf16")      v = ink2_bf16(((const uint16_t *) src)[s]);
                else if (ep->dtype == "f16")  v = ink2_f16_to_f32(((const uint16_t *) src)[s]);
                else if (ep->dtype == "f32")  v = ((const float *) src)[s];
                else if (ep->dtype == "i8")   v = (float) ((const int8_t *) src)[s];
                else throw std::runtime_error("inkling2: cannot convert " + ep->dtype);
                out[i] = v;
            }
        }});
        return x;
    };

    // the code tables
    std::map<std::string, ggml_tensor *> luts;
    for (const char * l : {"ht1", "ht2", "ht3", "rtr1", "rtr2", "rtr3", "rtr4"}) {
        if (has(std::string("lut.") + l)) luts[l] = raw(std::string("lut.") + l);
    }
    ggml_tensor * e4m3 = raw("lut.nvfp4_e4m3");
    static const char * fmt_names[] = {"ht1", "ht2", "ht3", "rtr1", "rtr2", "rtr3", "rtr4", "nvfp4"};

    auto lin = [&](const std::string & mname) -> ink2_lin {
        auto it = mats.find(mname);
        if (it == mats.end()) throw std::runtime_error("inkling2: no matrix " + mname);
        const mat_info & mi = it->second;
        ink2_lin L;
        L.fmt = mi.fmt; L.m = mi.rows; L.n = mi.cols;
        L.codes  = raw(mi.prefix + "_codes");
        L.scales = raw(mi.prefix + "_scales");
        if (mi.fmt == INK2_LM6 || mi.fmt == INK2_LM8) {
            L.table = raw(mi.prefix + "_levels");
        } else {
            L.table = luts.at(fmt_names[mi.fmt]);
        }
        L.signs_in = f32(mi.prefix + "_signs_in");
        if (mi.out_rot) L.signs_out = f32(mi.prefix + "_signs_out");
        return L;
    };

    const int64_t n_embd = hparams.n_embd;
    std::set<int> had_sizes = {(int) n_embd, 1152};

    W->layers.resize(INK2_N_PHYS);
    for (int i = 0; i < INK2_N_PHYS; ++i) {
        ink2_layer & L = W->layers[i];
        char lp[32], sp[48];
        snprintf(lp, sizeof(lp), "layer_%02d.", i);
        snprintf(sp, sizeof(sp), "st.layers.%d.", i);
        const std::string LP = lp, SP = sp;
        char np[8];
        snprintf(np, sizeof(np), "L%02d.", i);
        const std::string NP = np;
        L.swa = hparams.is_swa_impl[i] != 0;

        L.q   = lin(LP + "self_attn.q_proj.linear");
        L.k   = lin(LP + "self_attn.k_proj.linear");
        L.v   = lin(LP + "self_attn.v_proj.linear");
        L.o   = lin(LP + "self_attn.o_proj.linear");
        L.og  = lin(LP + "self_attn.o_proj.gate_proj");
        L.rel = lin(LP + "self_attn.relative_position_bias.linear");
        for (const ink2_lin * l : {&L.q, &L.k, &L.v, &L.o, &L.og, &L.rel}) {
            had_sizes.insert(l->n);
            if (l->signs_out) had_sizes.insert(l->m);
        }

        if (i > 0) L.attn_res_query = f32(SP + "attention_input.query");
        L.ffn_res_query = f32(SP + "mlp_input.query");
        if (L.swa) {
            L.attn_norm = f32(SP + "input_layernorm.weight");
            L.k_conv = f32(SP + "self_attn.k_proj.sconv.conv1d.weight", {INK2_CONV, 1024});
            L.v_conv = f32(SP + "self_attn.v_proj.sconv.conv1d.weight", {INK2_CONV, 1024});
        } else {
            L.q_norm = f32(SP + "q_input_layernorm.weight");
            L.k_norm = f32(SP + "k_input_layernorm.weight");
            L.v_norm = f32(SP + "v_input_layernorm.weight");
            // static_weight [4, C] (index = lag) -> ssm_conv kernel [4, C] (index 3 = current token)
            auto lagrev = [](int64_t i) { const int64_t c = i / INK2_CONV, k = i % INK2_CONV; return (INK2_CONV - 1 - k) * 1024 + c; };
            L.k_conv = f32(SP + "self_attn.k_proj.sconv.static_weight", {INK2_CONV, 1024}, lagrev);
            L.v_conv = f32(SP + "self_attn.v_proj.sconv.static_weight", {INK2_CONV, 1024}, lagrev);
            L.k_conv_proj = f32(SP + "self_attn.k_proj.sconv.weight_proj.weight");
            L.v_conv_proj = f32(SP + "self_attn.v_proj.sconv.weight_proj.weight");
        }
        L.ffn_norm = f32(SP + "post_attention_layernorm.weight");
        L.post_attn_conv = f32(SP + "post_attention_convolution.conv1d.weight", {INK2_CONV, n_embd});
        L.post_ffn_conv  = f32(SP + "post_mlp_convolution.conv1d.weight", {INK2_CONV, n_embd});
        L.q_scale   = f32(SP + "self_attn.q_proj.norm.scale");
        L.rel_scale = f32(SP + "self_attn.relative_position_bias.norm.scale");
        {
            // proj [d_rel, extent] -> [d_rel (ne0), extent]
            const auto & e = entry(SP + "self_attn.relative_position_bias.proj");
            const int64_t d_rel = e.shape[0], ext = e.shape[1];
            L.rel_proj = f32(SP + "self_attn.relative_position_bias.proj", {d_rel, ext},
                    [d_rel, ext](int64_t i) { const int64_t d = i % d_rel, x = i / d_rel; return d * ext + x; });
        }
        L.sink        = f32(SP + "self_attn.sink_proj.weight");
        L.router      = f32(SP + "mlp.gate.weight");
        L.router_bias = f32(SP + "mlp.gate.e_score_correction_bias");

        for (int s = 0; s < 2; ++s) {
            ink2_side & S = s == 0 ? L.gu : L.dn;
            const std::string side = s == 0 ? "gu" : "dn";
            S.map   = raw(NP + side + "_map");
            S.signs = f32(NP + side + "_signs_in");
            for (int f = 0; f <= 7; ++f) {
                const std::string base = NP + side + "_" + fmt_names[f];
                if (!has(base + "_codes")) continue;
                ink2_slab sl;
                sl.fmt = f;
                sl.codes = raw(base + "_codes");
                sl.scales = raw(base + "_scales");
                if (f == INK2_NVFP4) {
                    sl.table = raw(base + "_outer");
                    sl.aux = e4m3;
                } else {
                    sl.table = luts.at(fmt_names[f]);
                }
                S.slabs.push_back(sl);
            }
        }

        if (i >= INK2_CORE_START && i < INK2_CORE_END) {
            const char * gn[4] = {"q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate"};
            for (int c = 0; c < 4; ++c) {
                L.gates[c].down   = f32(SP + gn[c] + ".down_proj.weight");
                L.gates[c].forget = f32(SP + gn[c] + ".forget_gate_proj.weight");
                L.gates[c].gain   = f32(SP + gn[c] + ".message_gate_proj.weight");
            }
        }
    }
    W->embed_norm  = f32("st.embed_tokens.embed_norm.weight");
    W->final_query = f32("st.final_input.query");
    W->out_norm    = f32("st.norm.weight");
    W->msg_query   = f32("st.message_input.query");

    // the head, bucket by bucket, and the permutation back to token order
    int64_t n_head_rows = 0;
    for (int b = 0; b < 64; ++b) {
        const std::string mn = "head.b" + std::to_string(b);
        if (!mats.count(mn)) break;
        W->head.push_back(lin(mn));
        n_head_rows += W->head.back().m;
    }
    if (n_head_rows != (int64_t) vocab.n_tokens()) {
        throw std::runtime_error(format("inkling2: the head has %lld rows, the vocabulary %u tokens", (long long) n_head_rows, vocab.n_tokens()));
    }
    {
        const ink2_entry & e = entry("head.order");
        GGML_ASSERT(e.dtype == "i32" && e.shape.size() == 1 && e.shape[0] == n_head_rows);
        W->head_order = raw("head.order");
    }

    // Hadamard factors, row-major, and their transposes
    for (int n : had_sizes) {
        std::vector<double> A, B;
        int a_n, b_n;
        ink2_hadamard_factors(n, A, a_n, B, b_n);
        ink2_hadamard h;
        h.a = a_n; h.b = b_n;
        auto mk = [&](const std::vector<double> & M, int k, bool tr, const char * nm) {
            ggml_tensor * t = ggml_new_tensor_2d(ctx_dev, GGML_TYPE_F32, k, k);
            ggml_set_name(t, format("had.%d.%s", n, nm).c_str());
            uploads_dev.push_back({t, [M, k, tr](void * dst) {
                float * o = (float *) dst;
                for (int i = 0; i < k; ++i) for (int j = 0; j < k; ++j) o[i * k + j] = (float) (tr ? M[(size_t) j * k + i] : M[(size_t) i * k + j]);
            }});
            return t;
        };
        h.A = mk(A, a_n, false, "A"); h.At = mk(A, a_n, true, "At");
        h.B = mk(B, b_n, false, "B"); h.Bt = mk(B, b_n, true, "Bt");
        W->had[n] = h;
    }

    // the embedding: every bucket decoded and rotated back, token-indexed, f32 on the host
    int64_t n_embd_rows = 0;
    std::vector<mat_info> ebuckets;
    for (int b = 0; b < 64; ++b) {
        const std::string mn = "embed.b" + std::to_string(b);
        if (!mats.count(mn)) break;
        ebuckets.push_back(mats[mn]);
        n_embd_rows += mats[mn].rows;
    }
    tok_embd = ggml_new_tensor_2d(ctx_host, GGML_TYPE_F32, n_embd, n_embd_rows);
    ggml_set_name(tok_embd, "token_embd.weight");
    uploads_host.push_back({tok_embd, [&, ebuckets, n_embd_rows](void * dst) {
        float * out = (float *) dst;
        const int32_t * order = (const int32_t *) data_of(entry("embed.order"));
        std::vector<double> A, B;
        int a_n, b_n;
        ink2_hadamard_factors((int) n_embd, A, a_n, B, b_n);
        struct job { ink2_matrix m; std::vector<float> signs; int64_t lo, rows; };
        std::vector<job> jobs;
        int64_t lo = 0;
        for (const auto & mi : ebuckets) {
            job j;
            j.m.fmt = mi.fmt; j.m.n = mi.cols;
            j.m.codes  = data_of(entry(mi.prefix + "_codes"));
            j.m.scales = data_of(entry(mi.prefix + "_scales"));
            j.m.table  = (mi.fmt == INK2_LM6 || mi.fmt == INK2_LM8) ? (const void *) data_of(entry(mi.prefix + "_levels"))
                                                                    : (const void *) data_of(entry(std::string("lut.") + fmt_names[mi.fmt]));
            j.m.e4m3 = nullptr; j.m.outer = 1.0f;
            const int8_t * s = (const int8_t *) data_of(entry(mi.prefix + "_signs_in"));
            j.signs.assign(s, s + n_embd);
            j.lo = lo; j.rows = mi.rows;
            lo += mi.rows;
            jobs.push_back(std::move(j));
        }
        const int n_thr = std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
        std::vector<std::thread> thr;
        const float inv = 1.0f / std::sqrt((float) n_embd);
        std::vector<float> Af(A.begin(), A.end()), Bf(B.begin(), B.end());
        for (int t = 0; t < n_thr; ++t) {
            thr.emplace_back([&, t]() {
                std::vector<float> q(n_embd), z((size_t) a_n * b_n), y(n_embd);
                for (const auto & j : jobs) {
                    for (int64_t r = t; r < j.rows; r += n_thr) {
                        ink2_decode_row(&j.m, r, q.data());
                        // z = A^T Q B, Q [a, b] row-major
                        std::fill(z.begin(), z.end(), 0.0f);
                        std::vector<float> qb((size_t) a_n * b_n, 0.0f);
                        for (int ia = 0; ia < a_n; ++ia) {
                            for (int ib = 0; ib < b_n; ++ib) {
                                const float v = q[(size_t) ia * b_n + ib];
                                if (v == 0.0f) continue;
                                float * row = &qb[(size_t) ia * b_n];
                                const float * Brow = &Bf[(size_t) ib * b_n];
                                for (int jb = 0; jb < b_n; ++jb) row[jb] += v * Brow[jb];
                            }
                        }
                        for (int ia = 0; ia < a_n; ++ia) {
                            const float * Arow = &Af[(size_t) ia * a_n];
                            const float * src = &qb[(size_t) ia * b_n];
                            for (int ja = 0; ja < a_n; ++ja) {
                                const float av = Arow[ja];
                                float * dz = &z[(size_t) ja * b_n];
                                for (int jb = 0; jb < b_n; ++jb) dz[jb] += av * src[jb];
                            }
                        }
                        const int64_t tok = order[j.lo + r];
                        float * o = out + tok * n_embd;
                        for (int64_t c = 0; c < n_embd; ++c) o[c] = z[c] * (j.signs[c] * inv);
                    }
                }
            });
        }
        for (auto & th : thr) th.join();
    }});

    // ---- allocate and fill
    auto alloc_fill = [&](ggml_context * ctx, ggml_backend_buffer_type_t buft, std::vector<ink2_upload> & ups) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) throw std::runtime_error("inkling2: cannot allocate the weights");
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        W->bufs.emplace_back(buf);
        LLAMA_LOG_INFO("%s: inkling2 %s buffer %.1f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf) / 1048576.0);
        const bool host = ggml_backend_buffer_is_host(buf);
        std::vector<uint8_t> tmp;
        for (auto & u : ups) {
            if (host) {
                u.fill(u.t->data);
            } else {
                tmp.resize(ggml_nbytes(u.t));
                u.fill(tmp.data());
                ggml_backend_tensor_set(u.t, tmp.data(), 0, tmp.size());
            }
        }
    };
    alloc_fill(ctx_dev, buft_dev, uploads_dev);
    alloc_fill(ctx_host, ggml_backend_cpu_buffer_type(), uploads_host);

    ink2 = W;
}

// ---------------------------------------------------------------------------------------------- the graph

namespace {

struct llm_graph_input_ink2_rel : public llm_graph_input_i {
    const llama_kv_cache_context * base = nullptr;
    const llama_kv_cache_context * swa  = nullptr;
    ggml_tensor * rel_full = nullptr; // I32 [n_kv, n_tokens]
    ggml_tensor * rel_swa  = nullptr;
    int32_t ext_full = 1024, ext_swa = 1024;

    void set_input(const llama_ubatch * ubatch) override {
        base->set_input_rel_pos(rel_full, ubatch, ext_full);
        swa->set_input_rel_pos(rel_swa, ubatch, ext_swa);
    }

    bool can_reuse(const llm_graph_params & params) override {
        const auto * mctx = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);
        base = mctx->get_attn()->get_base();
        swa  = mctx->get_attn()->get_swa();
        return rel_full->ne[0] == (int64_t) base->get_n_kv() && rel_full->ne[1] == params.ubatch.n_tokens &&
               rel_swa->ne[0]  == (int64_t) swa->get_n_kv()  && rel_swa->ne[1]  == params.ubatch.n_tokens;
    }
};

struct ink2_graph : public llm_graph_context {
    const ink2_weights & w;
    llm_graph_input_mem_hybrid_iswa * inp = nullptr;
    llm_graph_input_ink2_rel * rel = nullptr;
    ggml_tensor * tau = nullptr;
    // decode glue: ubatches of <= 8 tokens (the small-N GEMV's range) with the weights on a GPU run the glue between
    // the matrix products as fused GGML_OP_INK2_GLUE kernels; prefill keeps the op chains, bitwise as before.
    // INK2_GLUE_OFF=1 keeps the chains everywhere.
    // INK2_GLUE_MASK selects sub-ops for testing (bits: 1 clip, 2 mix, 4 control, 8 moe, 16 conv, 32 bias table)
    bool glue = false;
    int glue_mask = 0;
    bool use(int bit) const { return glue && (glue_mask & bit); }
    ggml_tensor * zero_sinks = nullptr;

    ink2_graph(const llama_model_inkling2_loop & model, const llm_graph_params & params) : llm_graph_context(params), w(*model.ink2) {
        static const bool glue_off = getenv("INK2_GLUE_OFF") != nullptr;
        const ggml_backend_buffer_t wbuf = w.layers[0].router->buffer;
        glue = !glue_off && n_tokens <= 8 && wbuf && !ggml_backend_buffer_is_host(wbuf);
        static const int mask = getenv("INK2_GLUE_MASK") ? atoi(getenv("INK2_GLUE_MASK")) : 63;
        glue_mask = mask;
        build(model);
    }

    ggml_tensor * rot(ggml_tensor * x, ggml_tensor * signs, bool out) {
        const int64_t n = x->ne[0];
        const int64_t cnt = ggml_nelements(x) / n;
        const ink2_hadamard & h = w.had.at((int) n);
        // decode-sized ubatches (the small-N packed GEMV's range): the whole rotation as one GGML_OP_INK2_RHT node
        // (A is Sylvester, so At == A); prefill keeps the composition below, bitwise as before.
        // INK2_RHT_COMPOSE=1 forces the composition everywhere.
        static const bool compose = getenv("INK2_RHT_COMPOSE") != nullptr;
        if (!compose && n_tokens <= 8) {
            if (!ggml_is_contiguous(x)) x = ggml_cont(ctx0, x);
            x = ggml_reshape_2d(ctx0, x, n, cnt);
            return ggml_ink2_rht(ctx0, x, out ? nullptr : signs, out ? h.B : h.Bt, out ? signs : nullptr, h.a,
                                 1.0f / sqrtf((float) n));
        }
        x = ggml_reshape_2d(ctx0, ggml_cont(ctx0, x), n, cnt);
        if (!out) x = ggml_mul(ctx0, x, signs);
        // [b, a*cnt]: contract the B index, then the A index
        ggml_tensor * z = ggml_mul_mat(ctx0, out ? h.Bt : h.B, ggml_reshape_2d(ctx0, x, h.b, h.a * cnt));
        z = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, z, h.b, h.a, cnt), 1, 0, 2, 3)); // [a, b, cnt]
        z = ggml_mul_mat(ctx0, out ? h.At : h.A, ggml_reshape_2d(ctx0, z, h.a, h.b * cnt));
        z = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, z, h.a, h.b, cnt), 1, 0, 2, 3)); // [b, a, cnt]
        z = ggml_reshape_2d(ctx0, z, n, cnt);
        z = ggml_scale(ctx0, z, 1.0f / sqrtf((float) n));
        if (out) z = ggml_mul(ctx0, z, signs);
        return z;
    }

    // xr: x already rotated by L.signs_in (rot_multi), or null
    ggml_tensor * linear(const ink2_lin & L, ggml_tensor * x, ggml_tensor * xr = nullptr) {
        const int64_t T = ggml_nelements(x) / x->ne[0];
        if (!xr) xr = rot(x, L.signs_in, false);
        ggml_tensor * y = ggml_ink2_mul_mat(ctx0, L.codes, L.scales, L.table, nullptr, xr, nullptr, nullptr, L.fmt, L.m);
        if (L.signs_out) y = rot(y, L.signs_out, true);
        return ggml_reshape_2d(ctx0, y, L.m, T);
    }

    // decode-sized ubatches: the input rotations of the linears in L (all reading x) as one GGML_OP_INK2_RHT node;
    // each slice equals rot(x, L[i]->signs_in, false). Empty when rot would not use the fused op.
    std::vector<ggml_tensor *> rot_multi(ggml_tensor * x, const std::vector<const ink2_lin *> & L) {
        static const bool compose = getenv("INK2_RHT_COMPOSE") != nullptr || getenv("INK2_RHT_NO_MULTI") != nullptr;
        const int64_t n = x->ne[0];
        const int64_t cnt = ggml_nelements(x) / n;
        if (compose || n_tokens > 8 || L.size() < 2 || L.size() > 5) {
            return {};
        }
        ggml_tensor * signs[5];
        for (size_t i = 0; i < L.size(); ++i) {
            if (!L[i]->signs_in || ggml_nelements(L[i]->signs_in) != n) {
                return {};
            }
            signs[i] = L[i]->signs_in;
        }
        const ink2_hadamard & h = w.had.at((int) n);
        if (!ggml_is_contiguous(x)) x = ggml_cont(ctx0, x);
        x = ggml_reshape_2d(ctx0, x, n, cnt);
        ggml_tensor * r = ggml_ink2_rht_multi(ctx0, x, signs, (int) L.size(), h.Bt, h.a, 1.0f / sqrtf((float) n));
        std::vector<ggml_tensor *> out;
        for (size_t i = 0; i < L.size(); ++i) {
            out.push_back(ggml_view_2d(ctx0, r, n, cnt, r->nb[1], i * r->nb[2]));
        }
        return out;
    }

    ggml_tensor * norm_w(ggml_tensor * x, ggml_tensor * wt) {
        return ggml_mul(ctx0, ggml_rms_norm(ctx0, x, 1e-6f), wt);
    }

    ggml_tensor * glue_op(ggml_ink2_glue_op op, std::initializer_list<ggml_tensor *> srcs, ggml_type type,
                          int64_t ne0, int64_t ne1, int64_t ne2, int32_t i0 = 0, float f0 = 0.0f) {
        ggml_tensor * v[GGML_MAX_SRC] = {};
        int n = 0;
        for (ggml_tensor * t : srcs) {
            GGML_ASSERT(n < GGML_MAX_SRC);
            v[n++] = t;
        }
        return ggml_ink2_glue(ctx0, op, v, n, type, ne0, ne1, ne2, i0, f0);
    }

    // glue: tau_mul (full-attention layers) multiplies the clipped rows in the same kernel
    ggml_tensor * clip(ggml_tensor * cur, ggml_tensor * scale, float thr, ggml_tensor * tau_mul = nullptr) {
        if (use(1)) {
            return glue_op(GGML_INK2_GLUE_CLIP, {cur, scale, tau_mul}, GGML_TYPE_F32, cur->ne[0], cur->ne[1], cur->ne[2],
                           0, thr);
        }
        if (scale) cur = ggml_mul(ctx0, cur, scale);
        ggml_tensor * nrm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur)));
        nrm = ggml_scale_bias(ctx0, nrm, 1.0f, 1e-6f);
        ggml_tensor * f = ggml_div(ctx0, ggml_fill(ctx0, nrm, thr), nrm);
        return ggml_mul(ctx0, cur, ggml_clamp(ctx0, f, -INFINITY, 1.0f));
    }

    // attention over the depth: [n_embd, n_q, n_tokens]
    // (glue: the result is [n_embd, n_tokens, n_q], so each query's slice is contiguous; view_q reads either form)
    ggml_tensor * mix(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query) {
        if (use(2) && blocks.size() + 1 <= GGML_MAX_SRC - 2) {
            ggml_tensor * v[GGML_MAX_SRC] = {};
            int n = 0;
            v[n++] = query;
            for (ggml_tensor * b : blocks) v[n++] = b;
            v[n++] = cur;
            const int64_t n_q = ggml_nelements(query) / n_embd;
            return ggml_ink2_glue(ctx0, GGML_INK2_GLUE_MIX, v, n, GGML_TYPE_F32, n_embd, n_tokens, n_q, 0, 0.0f);
        }
        ggml_tensor * states = nullptr;
        for (ggml_tensor * b : blocks) {
            ggml_tensor * s = ggml_reshape_3d(ctx0, b, n_embd, 1, n_tokens);
            states = states ? ggml_concat(ctx0, states, s, 1) : s;
        }
        ggml_tensor * c = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
        states = states ? ggml_concat(ctx0, states, c, 1) : c;
        states = ggml_rms_norm(ctx0, states, 1e-6f);
        const int64_t n_q = ggml_nelements(query) / n_embd;
        ggml_tensor * q = ggml_repeat(ctx0, ggml_reshape_3d(ctx0, query, n_embd, n_q, 1), ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, n_q, n_tokens));
        ggml_tensor * scores = ggml_mul_mat(ctx0, states, q); // [n_states, n_q, T]
        scores = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f / sqrtf((float) n_embd), 0.0f);
        return ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, states)), scores); // [n_embd, n_q, T]
    }

    ggml_tensor * view_q(ggml_tensor * mixed, int j) {
        if (mixed->op == GGML_OP_INK2_GLUE) {
            return ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[1], j * mixed->nb[2]);
        }
        return ggml_cont(ctx0, ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[2], j * mixed->nb[1]));
    }

    // causal 4-tap convolution with residual; the last 3 inputs per channel live in the slot's recurrent state
    ggml_tensor * convolve(ggml_tensor * cur, ggml_tensor * kernel, ggml_tensor * dynamic, ggml_tensor * state,
                           ggml_tensor * history, int64_t offset) {
        const auto * mctx_recr = inp->mctx->get_recr();
        const int64_t d_conv = INK2_CONV - 1;
        const int64_t C = cur->ne[0];
        const int64_t n_seq_tokens = ubatch.n_seq_tokens;
        const int64_t n_seqs = ubatch.n_seqs;

        ggml_tensor * past = ggml_view_3d(ctx0, history, d_conv, C, n_seqs, history->nb[1], history->nb[2], offset * history->nb[1]);
        ggml_tensor * x = ggml_transpose(ctx0, ggml_reshape_3d(ctx0, cur, C, n_seq_tokens, n_seqs));
        x = ggml_concat(ctx0, past, x, 0); // [3 + T, C, n_seqs]

        const int64_t K = cparams.n_rs_seq > 0 ? (int64_t) cparams.n_rs_seq + 1 : 1;
        const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);
        const auto mem_size = mctx_recr->get_size();
        const uint32_t kv_head = mctx_recr->get_head();
        const size_t row_size = ggml_row_size(state->type, hparams.n_embd_r());
        const float out_scale = 1.0f / sqrtf((float) ((dynamic ? 2 : 1) * INK2_CONV + 1));

        if (use(16) && n_seqs == 1 && n_written <= 4 && ggml_is_contiguous(cur) && state->type == GGML_TYPE_F32) {
            // the convolution, the dynamic taps, the residual and the state snapshots in one kernel
            ggml_tensor * v[GGML_MAX_SRC] = {};
            v[0] = cur;
            v[1] = past;
            v[2] = kernel;
            v[3] = dynamic;
            for (int64_t slot = 0; slot < n_written; ++slot) {
                v[4 + slot] = ggml_view_2d(ctx0, state, d_conv * C, n_seqs, state->nb[1],
                        ((size_t) slot * mem_size + kv_head) * row_size + offset * d_conv * ggml_element_size(state));
            }
            ggml_tensor * out = ggml_ink2_glue(ctx0, GGML_INK2_GLUE_CONV, v, 4 + (int) n_written, GGML_TYPE_F32, C,
                                               n_tokens, 1, (int32_t) n_written, out_scale);
            return ggml_reshape_2d(ctx0, out, C, n_tokens);
        }
        for (int64_t slot = 0; slot < n_written; ++slot) {
            ggml_tensor * snap = ggml_view_3d(ctx0, x, d_conv, C, n_seqs, x->nb[1], x->nb[2], (x->ne[0] - d_conv - slot) * ggml_element_size(x));
            ggml_tensor * dst = ggml_view_2d(ctx0, state, d_conv * C, n_seqs, state->nb[1],
                    ((size_t) slot * mem_size + kv_head) * row_size + offset * d_conv * ggml_element_size(state));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, snap, dst));
        }

        ggml_tensor * out = ggml_ssm_conv(ctx0, x, kernel);
        out = ggml_reshape_2d(ctx0, out, C, n_tokens);

        if (dynamic) {
            const int64_t n_heads = dynamic->ne[0] / INK2_CONV;
            dynamic = ggml_reshape_3d(ctx0, dynamic, INK2_CONV, n_heads, n_tokens);
            for (int64_t lag = 0; lag < INK2_CONV; ++lag) {
                ggml_tensor * tap = ggml_view_3d(ctx0, dynamic, 1, n_heads, n_tokens, dynamic->nb[1], dynamic->nb[2], lag * dynamic->nb[0]);
                ggml_tensor * win = ggml_view_3d(ctx0, x, n_seq_tokens, C, n_seqs, x->nb[1], x->nb[2], (d_conv - lag) * x->nb[0]);
                win = ggml_cont(ctx0, ggml_transpose(ctx0, win));
                win = ggml_reshape_3d(ctx0, win, C / n_heads, n_heads, n_tokens);
                out = ggml_add(ctx0, out, ggml_reshape_2d(ctx0, ggml_mul(ctx0, win, tap), C, n_tokens));
            }
        }
        return ggml_scale(ctx0, ggml_add(ctx0, cur, out), 1.0f / sqrtf((float) ((dynamic ? 2 : 1) * INK2_CONV + 1)));
    }

    // (1 - tanh(forget)) * cur + tanh(gain) * message, gated on [norm(cur), norm(message)]
    ggml_tensor * control(ggml_tensor * cur, ggml_tensor * message, const ink2_gate & gt) {
        if (use(4)) {
            const int64_t R = gt.down->ne[1];
            const int64_t n_split = 16; // the CUDA kernel's split-K count: partials of the dots and the norms
            ggml_tensor * g = glue_op(GGML_INK2_GLUE_CTL_DOWN, {cur, message, gt.down}, GGML_TYPE_F32,
                                      2 * R * n_split + 2 * n_split, n_tokens, 1);
            return glue_op(GGML_INK2_GLUE_CTL_BLEND, {cur, message, g, gt.gain, gt.forget}, GGML_TYPE_F32, n_embd,
                           n_tokens, 1);
        }
        ggml_tensor * g = ggml_concat(ctx0, ggml_rms_norm(ctx0, cur, 1e-6f), ggml_rms_norm(ctx0, message, 1e-6f), 0);
        g = ggml_mul_mat(ctx0, gt.down, g);
        ggml_tensor * gain   = ggml_tanh(ctx0, ggml_mul_mat(ctx0, gt.gain, g));
        ggml_tensor * forget = ggml_tanh(ctx0, ggml_mul_mat(ctx0, gt.forget, g));
        return ggml_add(ctx0, ggml_sub(ctx0, cur, ggml_mul(ctx0, forget, cur)), ggml_mul(ctx0, gain, message));
    }

    ggml_tensor * moe(const ink2_layer & L, ggml_tensor * x, int il) {
        const int64_t n_exp = L.router->ne[1];
        const int64_t n_used = hparams.n_expert_used(il);
        const int64_t n_ff = 1152;
        ggml_tensor * logits = ggml_mul_mat(ctx0, L.router, x); // [n_exp, T]
        if (use(8)) {
            // top-k, SwiGLU and the weighted slot sum fused; each side's slabs fill one output between them
            ggml_tensor * ids = glue_op(GGML_INK2_GLUE_TOPK, {logits, L.router_bias}, GGML_TYPE_I32, n_used, n_tokens, 1,
                                        (int32_t) n_used);
            auto side = [&](const ink2_side & S, ggml_tensor * raw, ggml_tensor * rotd, int64_t M) {
                ggml_tensor * acc = nullptr;
                for (const auto & sl : S.slabs) {
                    acc = ggml_ink2_mul_mat_into(ctx0, sl.codes, sl.scales, sl.table, sl.aux,
                            sl.fmt == INK2_NVFP4 ? raw : rotd, ids, S.map, acc, sl.fmt, M);
                }
                return acc;
            };
            ggml_tensor * x3 = ggml_reshape_3d(ctx0, x, n_embd, 1, n_tokens);
            ggml_tensor * xr = ggml_reshape_3d(ctx0, rot(x, L.gu.signs, false), n_embd, 1, n_tokens);
            ggml_tensor * gu = side(L.gu, x3, xr, 2 * n_ff);
            ggml_tensor * act = glue_op(GGML_INK2_GLUE_SWIGLU, {gu}, GGML_TYPE_F32, n_ff, n_used, n_tokens);
            ggml_tensor * actr = ggml_reshape_3d(ctx0, rot(act, L.dn.signs, false), n_ff, n_used, n_tokens);
            ggml_tensor * dn = side(L.dn, act, actr, n_embd);
            return glue_op(GGML_INK2_GLUE_SLOTSUM, {dn, logits, ids}, GGML_TYPE_F32, n_embd, n_tokens, 1, 0,
                           hparams.expert_weights_scale);
        }
        ggml_tensor * probs = ggml_sigmoid(ctx0, logits);
        ggml_tensor * sel = ggml_add(ctx0, probs, L.router_bias);
        ggml_tensor * ids = ggml_cont(ctx0, ggml_argsort_top_k(ctx0, sel, n_used)); // [n_used, T]
        ggml_tensor * wts = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, probs, 1, n_exp, n_tokens), ids); // [1, n_used, T]
        ggml_tensor * wsum = ggml_sum_rows(ctx0, ggml_reshape_2d(ctx0, wts, n_used, n_tokens));
        wsum = ggml_scale_bias(ctx0, wsum, 1.0f, 1e-20f);
        wts = ggml_div(ctx0, ggml_reshape_2d(ctx0, wts, n_used, n_tokens), wsum);
        wts = ggml_scale(ctx0, wts, hparams.expert_weights_scale);
        wts = ggml_reshape_3d(ctx0, wts, 1, n_used, n_tokens);

        auto side = [&](const ink2_side & S, ggml_tensor * raw, ggml_tensor * rotd, int64_t M) {
            ggml_tensor * acc = nullptr;
            for (const auto & sl : S.slabs) {
                ggml_tensor * y = ggml_ink2_mul_mat(ctx0, sl.codes, sl.scales, sl.table, sl.aux,
                        sl.fmt == INK2_NVFP4 ? raw : rotd, ids, S.map, sl.fmt, M);
                acc = acc ? ggml_add(ctx0, acc, y) : y;
            }
            return acc;
        };
        ggml_tensor * x3 = ggml_reshape_3d(ctx0, x, n_embd, 1, n_tokens);
        ggml_tensor * xr = ggml_reshape_3d(ctx0, rot(x, L.gu.signs, false), n_embd, 1, n_tokens);
        ggml_tensor * gu = side(L.gu, x3, xr, 2 * n_ff); // [2*n_ff, n_used, T]
        ggml_tensor * g = ggml_cont(ctx0, ggml_view_3d(ctx0, gu, n_ff, n_used, n_tokens, gu->nb[1], gu->nb[2], 0));
        ggml_tensor * u = ggml_cont(ctx0, ggml_view_3d(ctx0, gu, n_ff, n_used, n_tokens, gu->nb[1], gu->nb[2], n_ff * gu->nb[0]));
        // softcap(g, 4) * sigmoid(g) * softcap(u, 25)
        ggml_tensor * gc = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, g, 0.25f)), 4.0f);
        ggml_tensor * uc = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, u, 1.0f / 25.0f)), 25.0f);
        ggml_tensor * act = ggml_mul(ctx0, ggml_mul(ctx0, gc, ggml_sigmoid(ctx0, g)), uc); // [n_ff, n_used, T]
        ggml_tensor * actr = ggml_reshape_3d(ctx0, rot(act, L.dn.signs, false), n_ff, n_used, n_tokens);
        ggml_tensor * dn = side(L.dn, act, actr, n_embd); // [n_embd, n_used, T]
        dn = ggml_mul(ctx0, dn, wts);
        ggml_tensor * out = nullptr;
        for (int64_t s = 0; s < n_used; ++s) {
            ggml_tensor * v = ggml_view_2d(ctx0, dn, n_embd, n_tokens, dn->nb[2], s * dn->nb[1]);
            out = out ? ggml_add(ctx0, out, v) : ggml_cont(ctx0, v);
        }
        return out;
    }

    // softmax(q k / sqrt(d) + bias + mask) with a per-token sink, q rows in chunks on the full-attention slots
    ggml_tensor * attention(int il, ggml_tensor * Q, ggml_tensor * Kc, ggml_tensor * Vc, ggml_tensor * table) {
        const bool is_swa = hparams.is_swa(il);
        auto * ia = inp->get_attn();
        const auto * mctx_cur = is_swa ? ia->mctx->get_swa() : ia->mctx->get_base();
        ggml_build_forward_expand(gf, Q);
        ggml_build_forward_expand(gf, Kc);
        ggml_build_forward_expand(gf, Vc);
        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kc, is_swa ? ia->get_k_idxs_swa() : ia->get_k_idxs(), il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, Vc, is_swa ? ia->get_v_idxs_swa() : ia->get_v_idxs(), il));
        ggml_tensor * mask = is_swa ? ia->get_kq_mask_swa() : ia->get_kq_mask();
        ggml_tensor * k = mctx_cur->get_k(ctx0, il);
        ggml_tensor * v = mctx_cur->get_v(ctx0, il);
        ggml_tensor * rp = is_swa ? rel->rel_swa : rel->rel_full;
        const int64_t n_kv = rp->ne[0];
        const int64_t n_head = Q->ne[1], d = Q->ne[0];
        ggml_tensor * zero_sinks = glue && this->zero_sinks && this->zero_sinks->ne[0] == n_head ? this->zero_sinks :
                ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head), 0.0f);
        if (glue) this->zero_sinks = zero_sinks;

        if (cparams.flash_attn) {
            // flash attention with the bias table [extent + 1, n_head, T] read by the relative index in the kernel
            // (ggml_flash_attn_ext_add_indexed_bias): no KQ or dense bias is materialized
            ggml_tensor * q  = ggml_permute(ctx0, Q, 0, 2, 1, 3); // [d, T, n_head]
            ggml_tensor * kp = ggml_permute(ctx0, k, 0, 2, 1, 3); // [d, n_kv, n_head_kv]
            ggml_tensor * vp = ggml_permute(ctx0, v, 0, 2, 1, 3);
            ggml_tensor * fa = ggml_flash_attn_ext(ctx0, q, kp, vp, mask, 1.0f / sqrtf((float) d), 0.0f, 0.0f);
            ggml_flash_attn_ext_add_sinks(fa, zero_sinks);
            ggml_flash_attn_ext_add_indexed_bias(fa, table, rp);
            ggml_prec_set_acc(fa, GGML_PREC_F32);
            return ggml_reshape_2d(ctx0, fa, d * n_head, n_tokens); // [d*n_head, T]
        }

        Q = ggml_scale(ctx0, Q, 1.0f / sqrtf((float) d));
        // keep each chunk's KQ, bias and softmax near 50 MiB at 32K context
        const int64_t chunk = std::max<int64_t>(1, std::min<int64_t>(n_tokens, (int64_t) (12u << 20) / (n_kv * n_head)));
        ggml_tensor * out = nullptr;
        for (int64_t c0 = 0; c0 < n_tokens; c0 += chunk) {
            const int64_t nc = std::min(chunk, n_tokens - c0);
            ggml_tensor * q = ggml_view_3d(ctx0, Q, d, n_head, nc, Q->nb[1], Q->nb[2], c0 * Q->nb[2]);
            ggml_tensor * tb = ggml_view_3d(ctx0, table, table->ne[0], table->ne[1], nc, table->nb[1], table->nb[2], c0 * table->nb[2]);
            ggml_tensor * ri = ggml_view_2d(ctx0, rp, n_kv, nc, rp->nb[1], c0 * rp->nb[1]);
            ggml_tensor * m = ggml_view_4d(ctx0, mask, mask->ne[0], nc, mask->ne[2], mask->ne[3], mask->nb[1], mask->nb[2], mask->nb[3], c0 * mask->nb[1]);
            // non-flash attention on this chunk (V is stored transposed when flash attention is off)
            ggml_tensor * qp = ggml_permute(ctx0, ggml_cont(ctx0, q), 0, 2, 1, 3);   // [d, nc, n_head]
            ggml_tensor * kp = ggml_permute(ctx0, k, 0, 2, 1, 3);                    // [d, n_kv, n_head_kv]
            ggml_tensor * kq = ggml_mul_mat(ctx0, kp, qp);                           // [n_kv, nc, n_head]
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            kq = ggml_ink2_rel_bias(ctx0, kq, tb, ri);
            kq = ggml_soft_max_ext_inplace(ctx0, kq, m, 1.0f, 0.0f);
            ggml_soft_max_add_sinks(kq, zero_sinks);
            ggml_tensor * vp = ggml_permute(ctx0, v, 0, 2, 1, 3);
            if (v->nb[1] <= v->nb[2]) {
                vp = ggml_cont(ctx0, ggml_transpose(ctx0, vp));
            }
            ggml_tensor * kqv = ggml_mul_mat(ctx0, vp, kq);                          // [d, nc, n_head]
            ggml_tensor * o = ggml_cont_2d(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3), d * n_head, nc);
            out = out ? ggml_concat(ctx0, out, o, 1) : o;
        }
        return out; // [d*n_head, T]
    }

    ggml_tensor * layer(int il, int ip, ggml_tensor * cur, std::vector<ggml_tensor *> & blocks, ggml_tensor * messages) {
        const ink2_layer & L = w.layers[ip];
        const bool is_swa = L.swa;
        const int64_t n_head = hparams.n_head(il), n_head_kv = hparams.n_head_kv(il), d = hparams.n_embd_head_k(il);
        const int64_t n_kvc = d * n_head_kv;
        const ink2_gate * gates = messages ? L.gates : nullptr;

        auto message = [&](int c) {
            return view_q(messages, c);
        };

        ggml_tensor * state   = inp->mctx->get_recr()->get_r_l(il);
        ggml_tensor * history = build_rs(inp->get_recr(), state, hparams.n_embd_r(), ubatch.n_seqs);
        history = ggml_reshape_3d(ctx0, history, INK2_CONV - 1, INK2_CONV_CH, ubatch.n_seqs);

        ggml_tensor * inpSA = cur;
        ggml_tensor * in[3] = {cur, cur, cur};
        if (L.attn_res_query) {
            ggml_tensor * mixed = mix(blocks, cur, L.attn_res_query);
            for (int j = 0; j < 3; ++j) in[j] = view_q(mixed, is_swa ? 0 : j);
        }
        if (gates) {
            for (int j = 0; j < 3; ++j) in[j] = control(in[j], message(j), gates[j]);
        }
        if (is_swa) {
            if (gates) {
                for (int j = 0; j < 3; ++j) in[j] = norm_w(in[j], L.attn_norm);
            } else {
                in[0] = norm_w(in[0], L.attn_norm);
                in[1] = in[2] = in[0];
            }
        } else {
            in[0] = norm_w(in[0], L.q_norm);
            in[1] = norm_w(in[1], L.k_norm);
            in[2] = norm_w(in[2], L.v_norm);
        }

        // q, rel and og read in[0] (k and v too when they share it): their input rotations in one node
        const bool kv_shared = in[1] == in[0] && in[2] == in[0];
        std::vector<ggml_tensor *> xr = kv_shared ? rot_multi(in[0], {&L.q, &L.rel, &L.og, &L.k, &L.v})
                                                  : rot_multi(in[0], {&L.q, &L.rel, &L.og});
        auto xr_at = [&](size_t i) { return i < xr.size() ? xr[i] : nullptr; };
        ggml_tensor * Q = linear(L.q, in[0], xr_at(0));
        ggml_tensor * Kc = convolve(linear(L.k, in[1], xr_at(3)), L.k_conv, is_swa ? nullptr : ggml_mul_mat(ctx0, L.k_conv_proj, in[1]), state, history, 0);
        ggml_tensor * Vc = convolve(linear(L.v, in[2], xr_at(4)), L.v_conv, is_swa ? nullptr : ggml_mul_mat(ctx0, L.v_conv_proj, in[2]), state, history, n_kvc);
        ggml_tensor * R = linear(L.rel, in[0], xr_at(1));
        cb(Kc, "Kconv", il);

        const float qk_clip = 22.627416997969522f;
        ggml_tensor * tau_clip = use(1) && !is_swa ? tau : nullptr; // glue: tau inside the clip kernel
        Q  = clip(ggml_reshape_3d(ctx0, Q, d, n_head, n_tokens), L.q_scale, qk_clip, tau_clip);
        Kc = clip(ggml_reshape_3d(ctx0, Kc, d, n_head_kv, n_tokens), nullptr, qk_clip);
        Vc = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, Vc, d, n_head_kv, n_tokens), hparams.f_norm_rms_eps);
        const int64_t d_rel = L.rel_proj->ne[0];
        R  = clip(ggml_reshape_3d(ctx0, R, d_rel, n_head, n_tokens), L.rel_scale, 8.0f, tau_clip);
        if (!is_swa && !use(1)) {
            Q = ggml_mul(ctx0, Q, tau);
            R = ggml_mul(ctx0, R, tau);
        }

        // bias table [extent + 1, n_head, T] (row `extent` is zero), minus the sink logit; the dense path reads it
        // as [n_head, extent + 1, T]
        ggml_tensor * sinks = ggml_mul_mat(ctx0, L.sink, in[0]); // [n_head, T]
        ggml_tensor * table;
        if (use(32) && cparams.flash_attn) {
            table = glue_op(GGML_INK2_GLUE_TABLE, {L.rel_proj, R, sinks}, GGML_TYPE_F32, L.rel_proj->ne[1] + 1, n_head,
                            n_tokens);
        } else {
            table = ggml_mul_mat(ctx0, ggml_pad(ctx0, L.rel_proj, 0, 1, 0, 0), R);
            table = ggml_sub(ctx0, table, ggml_reshape_3d(ctx0, sinks, 1, n_head, n_tokens));
        }
        if (!cparams.flash_attn) {
            table = ggml_cont(ctx0, ggml_permute(ctx0, table, 1, 0, 2, 3));
        }

        cb(Q, "Qcur", il);
        cb(Kc, "Kcur", il);
        cb(Vc, "Vcur", il);
        cur = attention(il, Q, Kc, Vc, table);
        cb(cur, "attn_raw", il);
        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, linear(L.og, in[0], xr_at(2))));
        cb(cur, "attn_gated", il);
        cur = linear(L.o, cur);
        cb(cur, "attn_o", il);
        cur = convolve(cur, L.post_attn_conv, nullptr, state, history, 2 * n_kvc);
        cb(cur, "attn_conv", il);

        if (ip % INK2_BLOCK == 0) {
            blocks.push_back(inpSA);
        } else {
            cur = ggml_add(ctx0, inpSA, cur);
        }
        ggml_tensor * ffn_inp = cur;

        cur = ggml_reshape_2d(ctx0, mix(blocks, cur, L.ffn_res_query), n_embd, n_tokens);
        if (gates) cur = control(cur, message(3), gates[3]);
        cur = norm_w(cur, L.ffn_norm);
        cb(cur, "moe_in", il);
        cur = moe(L, cur, il);
        cb(cur, "moe_out", il);
        cur = convolve(cur, L.post_ffn_conv, nullptr, state, history, 2 * n_kvc + n_embd);
        cur = ggml_add(ctx0, ffn_inp, cur);
        cb(cur, "l_out", il);
        return cur;
    }

    void build(const llama_model_inkling2_loop & model) {
        // inputs
        {
            const auto * mctx_cur = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);
            inp = build_inp_mem_hybrid_iswa();
            auto r = std::make_unique<llm_graph_input_ink2_rel>();
            r->base = mctx_cur->get_attn()->get_base();
            r->swa  = mctx_cur->get_attn()->get_swa();
            const int64_t n_kv_full = r->base->get_n_kv();
            const int64_t n_kv_swa  = r->swa->get_n_kv();
            r->rel_full = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv_full, n_tokens);
            r->rel_swa  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv_swa, n_tokens);
            ggml_set_input(r->rel_full);
            ggml_set_input(r->rel_swa);
            r->ext_full = (int32_t) w.layers[5].rel_proj->ne[1];
            r->ext_swa  = (int32_t) w.layers[0].rel_proj->ne[1];
            rel = (llm_graph_input_ink2_rel *) res->add_input(std::move(r));
        }
        ggml_tensor * inp_pos = build_inp_pos();
        ggml_tensor * inp_out_ids = build_inp_out_ids();
        tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, inp_pos, GGML_TYPE_F32), 1.0f, 1.0f);
        tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f / 8192.0f), 1.0f, INFINITY));
        tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, 0.1f, 1.0f), 1, 1, n_tokens);

        ggml_tensor * cur = build_inp_embd(model.tok_embd);
        cur = norm_w(cur, w.embed_norm);
        cb(cur, "inp_normed", -1);

        std::vector<ggml_tensor *> blocks;
        for (int il = 0; il < INK2_CORE_START; ++il) {
            cur = layer(il, il, cur, blocks, nullptr);
        }
        ggml_tensor * prelude = cur;
        const auto prelude_blocks = blocks;
        for (int pass = 0; pass < INK2_N_PASSES; ++pass) {
            std::vector<ggml_tensor *> bank;
            ggml_tensor * bank_out = nullptr;
            if (pass > 0) {
                bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
                bank_out = cur;
                cur = prelude;
                blocks = prelude_blocks;
            }
            for (int ip = INK2_CORE_START; ip < INK2_CORE_END; ++ip) {
                const int il = pass == 0 ? ip : INK2_N_PHYS + (pass - 1) * (INK2_CORE_END - INK2_CORE_START) + (ip - INK2_CORE_START);
                ggml_tensor * messages = nullptr;
                if (pass > 0) {
                    const int li = ip - INK2_CORE_START;
                    ggml_tensor * q = ggml_view_2d(ctx0, w.msg_query, n_embd, 4, w.msg_query->nb[1], li * w.msg_query->nb[2]);
                    messages = mix(bank, bank_out, use(2) ? q : ggml_cont(ctx0, q)); // [n_embd, 4, T]
                }
                cur = layer(il, ip, cur, blocks, messages);
            }
        }
        for (int il = INK2_CORE_END; il < INK2_N_PHYS; ++il) {
            cur = layer(il, il, cur, blocks, nullptr);
        }

        cur = ggml_reshape_2d(ctx0, mix(blocks, cur, w.final_query), n_embd, n_tokens);
        cur = norm_w(cur, w.out_norm);
        if (inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        }
        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // head: buckets in stored order, then back to token order
        // each bucket writes its rows straight to their tokens: row r of bucket b is token order[lo_b + r]
        ggml_tensor * logits = nullptr;
        int64_t lo = 0;
        for (const auto & b : w.head) {
            ggml_tensor * xr = rot(cur, b.signs_in, false);
            ggml_tensor * rows = ggml_view_1d(ctx0, w.head_order, b.m, lo * ggml_element_size(w.head_order));
            logits = ggml_ink2_mul_mat_rows(ctx0, b.codes, b.scales, b.table, xr, rows, logits, b.fmt, b.m, w.head_order->ne[0]);
            lo += b.m;
        }
        const float cap = hparams.f_final_logit_softcapping;
        if (cap > 0.0f) {
            logits = ggml_scale_inplace(ctx0, ggml_tanh_inplace(ctx0, ggml_scale_inplace(ctx0, logits, 1.0f / cap)), cap);
        }
        cb(logits, "result_output", -1);
        res->t_logits = logits;
        ggml_build_forward_expand(gf, logits);
    }
};

} // namespace

std::unique_ptr<llm_graph_context> llama_model_inkling2_loop::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<ink2_graph>(*this, params);
}
