// llama-inkling-logits: prefill logits of every position, for checking the Inkling-2 port against a reference.
//
//   INK_PROMPTS=<file> INK_LOGITS_OUT=<file> llama-inkling-logits -m <model> [-c N] [-ngl N] [-ub N]
//
// INK_PROMPTS: one prompt per line, "name id id id ...". Output, per prompt: int32 name length, name bytes,
// int32 n_tokens, int32 n_vocab, then n_tokens x n_vocab float32 logits (prefill, every position asks for logits;
// with INK_DECODE=1 the tokens go one at a time).

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string model;
    int n_ctx = 4096, ngl = 99, n_ubatch = 512;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "-m") model = argv[i + 1];
        else if (a == "-c") n_ctx = atoi(argv[i + 1]);
        else if (a == "-ngl") ngl = atoi(argv[i + 1]);
        else if (a == "-ub") n_ubatch = atoi(argv[i + 1]);
    }
    const char * pin = getenv("INK_PROMPTS");
    const char * pout = getenv("INK_LOGITS_OUT");
    if (model.empty() || !pin || !pout) {
        fprintf(stderr, "usage: INK_PROMPTS=f INK_LOGITS_OUT=f %s -m model\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * m = llama_model_load_from_file(model.c_str(), mp);
    if (!m) return 1;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = 512;
    cp.n_ubatch = n_ubatch;
    // INK_NOFA=1: the dense (non-flash) attention path
    if (getenv("INK_NOFA")) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(m, cp);
    if (!ctx) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(m);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    FILE * out = fopen(pout, "wb");
    std::ifstream in(pin);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string name;
        if (!(ss >> name)) continue;
        std::vector<llama_token> ids;
        int t;
        while (ss >> t) ids.push_back(t);
        llama_memory_clear(llama_get_memory(ctx), true);
        // INK_GREEDY=N: prefill the prompt in batches of 512, then N greedy tokens one at a time; the logits written
        // are the N decode steps' (the step before each chosen token)
        if (const char * g = getenv("INK_GREEDY")) {
            const int n_gen = atoi(g);
            std::vector<float> logits((size_t) n_gen * n_vocab);
            for (size_t p0 = 0; p0 < ids.size(); p0 += 512) {
                const int n = (int) std::min<size_t>(512, ids.size() - p0);
                llama_batch b = llama_batch_init(n, 0, 1);
                for (int i = 0; i < n; ++i) {
                    b.token[i] = ids[p0 + i];
                    b.pos[i] = (llama_pos) (p0 + i);
                    b.n_seq_id[i] = 1;
                    b.seq_id[i][0] = 0;
                    b.logits[i] = i == n - 1;
                }
                b.n_tokens = n;
                if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
                llama_batch_free(b);
            }
            const float * l = llama_get_logits_ith(ctx, -1);
            std::string toks;
            for (int k = 0; k < n_gen; ++k) {
                std::copy(l, l + n_vocab, logits.begin() + (size_t) k * n_vocab);
                int best = 0;
                for (int v = 1; v < n_vocab; ++v) if (l[v] > l[best]) best = v;
                toks += " " + std::to_string(best);
                if (k + 1 == n_gen) break;
                llama_batch b = llama_batch_init(1, 0, 1);
                b.token[0] = best;
                b.pos[0] = (llama_pos) (ids.size() + k);
                b.n_seq_id[0] = 1;
                b.seq_id[0][0] = 0;
                b.logits[0] = 1;
                b.n_tokens = 1;
                if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
                llama_batch_free(b);
                l = llama_get_logits_ith(ctx, 0);
            }
            const int32_t nl = (int32_t) name.size(), nt = n_gen;
            fwrite(&nl, 4, 1, out);
            fwrite(name.data(), 1, nl, out);
            fwrite(&nt, 4, 1, out);
            fwrite(&n_vocab, 4, 1, out);
            fwrite(logits.data(), 4, logits.size(), out);
            fprintf(stderr, "%s: greedy%s\n", name.c_str(), toks.c_str());
            continue;
        }
        std::vector<float> logits((size_t) ids.size() * n_vocab);
        // INK_DECODE=1: one token per llama_decode (the decode path) instead of prefill batches
        const size_t step = getenv("INK_DECODE") ? 1 : 512;
        for (size_t p0 = 0; p0 < ids.size(); p0 += step) {
            const int n = (int) std::min<size_t>(step, ids.size() - p0);
            llama_batch b = llama_batch_init(n, 0, 1);
            for (int i = 0; i < n; ++i) {
                b.token[i] = ids[p0 + i];
                b.pos[i] = (llama_pos) (p0 + i);
                b.n_seq_id[i] = 1;
                b.seq_id[i][0] = 0;
                b.logits[i] = 1;
            }
            b.n_tokens = n;
            if (llama_decode(ctx, b) != 0) {
                fprintf(stderr, "decode failed\n");
                return 1;
            }
            for (int i = 0; i < n; ++i) {
                const float * l = llama_get_logits_ith(ctx, i);
                std::copy(l, l + n_vocab, logits.begin() + (p0 + i) * n_vocab);
            }
            llama_batch_free(b);
        }
        const int32_t nl = (int32_t) name.size(), nt = (int32_t) ids.size();
        fwrite(&nl, 4, 1, out);
        fwrite(name.data(), 1, nl, out);
        fwrite(&nt, 4, 1, out);
        fwrite(&n_vocab, 4, 1, out);
        fwrite(logits.data(), 4, logits.size(), out);
        fprintf(stderr, "%s: %d tokens\n", name.c_str(), nt);
    }
    fclose(out);
    llama_free(ctx);
    llama_model_free(m);
    return 0;
}
