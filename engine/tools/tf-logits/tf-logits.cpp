// llama-tf-logits: teacher-forced next-token distributions on the single-token (T=1) decode path.
//
// P100 v2 quality gate D1 (DIRECTION.md section 3). The prompt (-f FILE) is prefilled in ubatches; then N (= -n) tokens
// are fed ONE AT A TIME (llama_decode with n_tokens = 1, i.e. the same attention/GEMV kernels as generation). At every
// step the full next-token distribution is reduced to: log-sum-exp, the top-K (id, logprob), the logprob of the next
// forced token, and (if a reference dump is given) the candidate's logprobs for the reference's top-K ids.
//
// Reference mode (no TF_FORCED_IN): tokens are chosen greedily and written to TF_FORCED_OUT, so a candidate run can be
// forced along exactly the same history. Candidate mode: TF_FORCED_IN = that token file, TF_REF_IN = the reference dump.
//
// Environment:
//   TF_OUT         output JSONL (required)
//   TF_TOPK        top-K per position (default 64)
//   TF_FORCED_IN   JSON list of token ids to force (candidate mode)
//   TF_FORCED_OUT  write the greedy reference tokens here (reference mode)
//   TF_REF_IN      reference JSONL: record candidate logprobs for the reference's top-K ids (exact KL on that support)

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int> parse_int_list(const std::string & s) {
    std::vector<int> out;
    std::string num;
    for (char ch : s) {
        if ((ch >= '0' && ch <= '9') || ch == '-') {
            num.push_back(ch);
        } else if (!num.empty()) {
            out.push_back(std::stoi(num));
            num.clear();
        }
    }
    if (!num.empty()) {
        out.push_back(std::stoi(num));
    }
    return out;
}

// reference dump: one JSON object per line with "top":[[id,lp],...]; we only need the ids per position
static std::vector<std::vector<int>> load_ref_top_ids(const std::string & path) {
    std::vector<std::vector<int>> ids;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const size_t a = line.find("\"top\":[");
        const size_t b = line.find("]]", a);
        std::vector<int> row;
        if (a != std::string::npos && b != std::string::npos) {
            const std::string body = line.substr(a + 7, b - (a + 7));
            // body: [id,lp],[id,lp],...  -> take the first number of every pair
            size_t pos = 0;
            while ((pos = body.find('[', pos)) != std::string::npos) {
                row.push_back(std::atoi(body.c_str() + pos + 1));
                pos += 1;
            }
        }
        ids.push_back(row);
    }
    return ids;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_predict = 256;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    const char * out_path = getenv("TF_OUT");
    if (!out_path) {
        fprintf(stderr, "TF_OUT is required\n");
        return 1;
    }
    const int topk = getenv("TF_TOPK") ? atoi(getenv("TF_TOPK")) : 64;
    std::vector<int> forced;
    if (const char * fi = getenv("TF_FORCED_IN")) {
        std::ifstream in(fi);
        std::stringstream ss;
        ss << in.rdbuf();
        forced = parse_int_list(ss.str());
    }
    std::vector<std::vector<int>> ref_top;
    if (const char * ri = getenv("TF_REF_IN")) {
        ref_top = load_ref_top_ids(ri);
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    common_init_result_ptr init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();
    if (!model || !ctx) {
        fprintf(stderr, "failed to initialise model/context\n");
        return 1;
    }
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> prompt = common_tokenize(ctx, params.prompt, true, true);
    const int n_prompt = (int) prompt.size();
    const int n_steps  = forced.empty() ? params.n_predict : std::min<int>(params.n_predict, (int) forced.size());
    if (n_prompt + n_steps + 1 > (int) llama_n_ctx(ctx)) {
        fprintf(stderr, "prompt (%d) + steps (%d) exceed n_ctx (%d)\n", n_prompt, n_steps, (int) llama_n_ctx(ctx));
        return 1;
    }

    // prefill (batched; only the last prompt token needs logits)
    const auto t0 = std::chrono::steady_clock::now();
    const int n_batch = std::max(1, (int) params.n_batch);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (int i = 0; i < n_prompt; i += n_batch) {
        const int n = std::min(n_batch, n_prompt - i);
        common_batch_clear(batch);
        for (int j = 0; j < n; ++j) {
            common_batch_add(batch, prompt[i + j], i + j, {0}, i + j == n_prompt - 1);
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "prefill decode failed at %d\n", i);
            return 1;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();

    FILE * out = fopen(out_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open %s\n", out_path);
        return 1;
    }
    std::vector<llama_token> chosen;
    std::vector<std::pair<float, int>> scored(n_vocab);
    int pos = n_prompt;
    int logit_idx = batch.n_tokens - 1;  // index of the last prompt token's logits in the final prefill batch
    for (int step = 0; step < n_steps; ++step) {
        const float * logits = llama_get_logits_ith(ctx, logit_idx);
        float mx = -INFINITY;
        for (int v = 0; v < n_vocab; ++v) {
            mx = std::max(mx, logits[v]);
        }
        double sum = 0.0;
        for (int v = 0; v < n_vocab; ++v) {
            sum += std::exp((double) logits[v] - mx);
        }
        const double lse = mx + std::log(sum);
        for (int v = 0; v < n_vocab; ++v) {
            scored[v] = {logits[v], v};
        }
        std::partial_sort(scored.begin(), scored.begin() + topk, scored.end(),
                          [](const auto & a, const auto & b) { return a.first > b.first; });
        const llama_token next = forced.empty() ? (llama_token) scored[0].second : (llama_token) forced[step];
        chosen.push_back(next);

        fprintf(out, "{\"i\":%d,\"pos\":%d,\"lse\":%.6f,\"next\":%d,\"next_lp\":%.6f,\"top\":[", step, pos, lse, next,
                (double) logits[next] - lse);
        for (int k = 0; k < topk; ++k) {
            fprintf(out, "%s[%d,%.6f]", k ? "," : "", scored[k].second, (double) scored[k].first - lse);
        }
        fprintf(out, "]");
        if (step < (int) ref_top.size() && !ref_top[step].empty()) {
            fprintf(out, ",\"ref_lp\":[");
            for (size_t k = 0; k < ref_top[step].size(); ++k) {
                const int id = ref_top[step][k];
                fprintf(out, "%s%.6f", k ? "," : "", (id >= 0 && id < n_vocab) ? (double) logits[id] - lse : -1e30);
            }
            fprintf(out, "]");
        }
        fprintf(out, "}\n");

        // feed the chosen/forced token on the single-token decode path
        common_batch_clear(batch);
        common_batch_add(batch, next, pos, {0}, true);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at step %d\n", step);
            return 1;
        }
        logit_idx = 0;
        ++pos;
    }
    fclose(out);
    const auto t2 = std::chrono::steady_clock::now();

    if (const char * fo = getenv("TF_FORCED_OUT")) {
        FILE * f = fopen(fo, "w");
        fprintf(f, "[");
        for (size_t i = 0; i < chosen.size(); ++i) {
            fprintf(f, "%s%d", i ? "," : "", chosen[i]);
        }
        fprintf(f, "]\n");
        fclose(f);
    }
    fprintf(stderr, "tf-logits: prompt %d tokens, %d forced steps, prefill %.1f s, decode %.1f s (%.1f ms/step)\n", n_prompt,
            n_steps, std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count(),
            1000.0 * std::chrono::duration<double>(t2 - t1).count() / std::max(1, n_steps));
    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
