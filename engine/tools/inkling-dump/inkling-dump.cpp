// llama-inkling-dump: decode rows of LookAtMyQuant's packed Inkling-2-Mini matrices to float32.
//
//   INK_ROWS=<requests> INK_OUT=<output> llama-inkling-dump -m <prepared gguf> [-ngl N] [-t N]
//
// Each request line is "<matrix> <row>"; the output holds, per line in order, an int32 count n and n float32 values:
// that row as decoded from the pack's arrays (code value times the row's scale, times the expert's outer scale for
// NVFP4), in the stored (rotated) basis. Matrix names:
//   layer_LL.<linear>          a key of layer LL's meta "linears"
//   layer_LL.experts.gu.E      expert E's gate+up (2304 rows), or .dn.E its down projection
//   embed.bK / head.bK         bucket K of the embedding or the output head, rows in stored order
//
// The model is the GGUF that tools/inkling-prepare/prepare.py writes: its table names every array of the pack by
// file and absolute offset, and the arrays are memory-mapped in place from the pack folder.

#include "gguf.h"
#include "ggml-inkling2.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct pack_tensor {
    int file;
    uint64_t offset, nbytes;
    std::string dtype;
    std::vector<int64_t> shape;
};

struct pack_matrix {
    std::string prefix;
    int fmt, rows, cols, out_rot;
};

struct pack {
    std::string dir;
    std::vector<std::string> files;
    std::vector<const uint8_t *> maps;
    std::map<std::string, pack_tensor> tensors;
    std::map<std::string, pack_matrix> matrices;

    static int64_t key(gguf_context * ctx, const char * k) {
        const int64_t id = gguf_find_key(ctx, k);
        if (id < 0) {
            throw std::runtime_error(std::string("key not found: ") + k);
        }
        return id;
    }

    void load(const std::string & path) {
        gguf_init_params params = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
        gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
        if (!ctx) {
            throw std::runtime_error("cannot read " + path);
        }
        dir = gguf_get_val_str(ctx, key(ctx, "inkling2.pack_dir"));
        const int64_t kf = key(ctx, "inkling2.files");
        for (size_t i = 0; i < gguf_get_arr_n(ctx, kf); ++i) {
            files.push_back(gguf_get_arr_str(ctx, kf, i));
        }
        maps.assign(files.size(), nullptr);
        const int64_t kn = key(ctx, "inkling2.t.names");
        const size_t n = gguf_get_arr_n(ctx, kn);
        const uint32_t * tf = (const uint32_t *) gguf_get_arr_data(ctx, key(ctx, "inkling2.t.file"));
        const uint64_t * to = (const uint64_t *) gguf_get_arr_data(ctx, key(ctx, "inkling2.t.offset"));
        const uint64_t * tb = (const uint64_t *) gguf_get_arr_data(ctx, key(ctx, "inkling2.t.nbytes"));
        const uint32_t * nd = (const uint32_t *) gguf_get_arr_data(ctx, key(ctx, "inkling2.t.ndim"));
        const uint64_t * dims = (const uint64_t *) gguf_get_arr_data(ctx, key(ctx, "inkling2.t.dims"));
        const int64_t kd = key(ctx, "inkling2.t.dtype");
        size_t di = 0;
        for (size_t i = 0; i < n; ++i) {
            pack_tensor t;
            t.file = (int) tf[i];
            t.offset = to[i];
            t.nbytes = tb[i];
            t.dtype = gguf_get_arr_str(ctx, kd, i);
            for (uint32_t d = 0; d < nd[i]; ++d) {
                t.shape.push_back((int64_t) dims[di++]);
            }
            tensors[gguf_get_arr_str(ctx, kn, i)] = t;
        }
        const int64_t km = key(ctx, "inkling2.m.names");
        const int64_t kp = key(ctx, "inkling2.m.prefix");
        const uint32_t * mf = (const uint32_t *) gguf_get_arr_data(ctx, key(ctx, "inkling2.m.fmt"));
        const uint32_t * mr = (const uint32_t *) gguf_get_arr_data(ctx, key(ctx, "inkling2.m.rows"));
        const uint32_t * mc = (const uint32_t *) gguf_get_arr_data(ctx, key(ctx, "inkling2.m.cols"));
        const uint32_t * mo = (const uint32_t *) gguf_get_arr_data(ctx, key(ctx, "inkling2.m.out_rot"));
        for (size_t i = 0; i < gguf_get_arr_n(ctx, km); ++i) {
            matrices[gguf_get_arr_str(ctx, km, i)] =
                pack_matrix{ gguf_get_arr_str(ctx, kp, i), (int) mf[i], (int) mr[i], (int) mc[i], (int) mo[i] };
        }
        gguf_free(ctx);
    }

    const pack_tensor & tensor(const std::string & name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            throw std::runtime_error("no array " + name);
        }
        return it->second;
    }

    const uint8_t * data(const std::string & name) {
        const pack_tensor & t = tensor(name);
        if (!maps[t.file]) {
            const std::string p = dir + "/" + files[t.file];
            const int fd = open(p.c_str(), O_RDONLY);
            if (fd < 0) {
                throw std::runtime_error("cannot open " + p);
            }
            struct stat st;
            fstat(fd, &st);
            void * m = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
            close(fd);
            if (m == MAP_FAILED) {
                throw std::runtime_error("cannot map " + p);
            }
            maps[t.file] = (const uint8_t *) m;
        }
        return maps[t.file] + t.offset;
    }
};

static std::vector<float> decode_matrix_row(pack & pk, const pack_matrix & m, int64_t r) {
    if (r < 0 || r >= m.rows) {
        throw std::runtime_error("row out of range");
    }
    std::vector<float> out(m.cols);
    const uint16_t * scales = (const uint16_t *) pk.data(m.prefix + "_scales");
    if (m.fmt == INK2_LM6 || m.fmt == INK2_LM8) {
        const int k = m.fmt == INK2_LM6 ? 6 : 8;
        ink2_decode_lm_row(pk.data(m.prefix + "_codes"), scales[r], (const float *) pk.data(m.prefix + "_levels"), k,
                           m.cols, r, out.data());
    } else if (ink2_is_trellis(m.fmt)) {
        static const char * lut_names[] = { "ht1", "ht2", "ht3", "rtr1", "rtr2", "rtr3", "rtr4" };
        const ink2_trellis_params p = ink2_trellis(m.fmt);
        const int64_t wpr = (int64_t) m.cols * p.k / 32;
        const uint32_t * codes = (const uint32_t *) pk.data(m.prefix + "_codes");
        const uint16_t * lut = (const uint16_t *) pk.data(std::string("lut.") + lut_names[m.fmt]);
        ink2_decode_trellis_row(codes + r * wpr, scales[r], lut, m.fmt, m.cols, out.data());
    } else {
        throw std::runtime_error("unexpected matrix format");
    }
    return out;
}

static std::vector<float> decode_expert_row(pack & pk, int layer, const std::string & side, int e, int64_t r) {
    static const char * fmt_names[] = { "ht1", "ht2", "ht3", "rtr1", "rtr2", "rtr3", "rtr4", "nvfp4" };
    char pre[16];
    snprintf(pre, sizeof(pre), "L%02d.", layer);
    const std::string P = pre + side;
    const int16_t * map = (const int16_t *) pk.data(P + "_map");
    const int fid = map[2 * e], loc = map[2 * e + 1];
    const int rows = 2304;
    const int n = side == "gu" ? 2304 : 1152;
    if (r < 0 || r >= rows || fid < 0 || fid > 7) {
        throw std::runtime_error("bad expert request");
    }
    std::vector<float> out(n);
    const std::string S = P + "_" + fmt_names[fid];
    if (fid == INK2_NVFP4) {
        static float e2m1[16], e4m3[256];
        memcpy(e2m1, pk.data("lut.nvfp4_e2m1"), sizeof(e2m1));
        memcpy(e4m3, pk.data("lut.nvfp4_e4m3"), sizeof(e4m3));
        const uint8_t * codes = pk.data(S + "_codes") + ((int64_t) loc * rows + r) * (n / 2);
        const uint8_t * sc = pk.data(S + "_scales") + ((int64_t) loc * rows + r) * (n / 16);
        const float outer = ((const float *) pk.data(S + "_outer"))[loc];
        ink2_decode_nvfp4_row(codes, sc, outer, e2m1, e4m3, n, out.data());
    } else {
        const ink2_trellis_params p = ink2_trellis(fid);
        const int64_t wpr = (int64_t) n * p.k / 32;
        const uint32_t * codes = (const uint32_t *) pk.data(S + "_codes") + ((int64_t) loc * rows + r) * wpr;
        const uint16_t * scales = (const uint16_t *) pk.data(S + "_scales");
        const uint16_t * lut = (const uint16_t *) pk.data(std::string("lut.") + fmt_names[fid]);
        ink2_decode_trellis_row(codes, scales[(int64_t) loc * rows + r], lut, fid, n, out.data());
    }
    return out;
}

int main(int argc, char ** argv) {
    std::string model;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "-m" || a == "--model") && i + 1 < argc) {
            model = argv[++i];
        } else if ((a == "-ngl" || a == "-t" || a == "--threads" || a == "--n-gpu-layers") && i + 1 < argc) {
            ++i; // accepted for the common command line; decoding runs on the CPU
        }
    }
    const char * rows_path = getenv("INK_ROWS");
    const char * out_path = getenv("INK_OUT");
    if (model.empty() || !rows_path || !out_path) {
        fprintf(stderr, "usage: INK_ROWS=<requests> INK_OUT=<output> %s -m <model.gguf>\n", argv[0]);
        return 1;
    }
    struct stat st;
    if (stat(model.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
        model += "/inkling2.gguf";
    }
    try {
        pack pk;
        pk.load(model);
        std::ifstream in(rows_path);
        FILE * out = fopen(out_path, "wb");
        if (!in || !out) {
            fprintf(stderr, "cannot open the request or output file\n");
            return 1;
        }
        std::string line;
        int n_rows = 0;
        while (std::getline(in, line)) {
            std::istringstream ss(line);
            std::string name;
            int64_t row;
            if (!(ss >> name >> row)) {
                continue;
            }
            std::vector<float> v;
            int layer, e;
            char side[8];
            if (sscanf(name.c_str(), "layer_%d.experts.%2[gudn].%d", &layer, side, &e) == 3) {
                v = decode_expert_row(pk, layer, side, e, row);
            } else {
                auto it = pk.matrices.find(name);
                if (it == pk.matrices.end()) {
                    throw std::runtime_error("unknown matrix " + name);
                }
                v = decode_matrix_row(pk, it->second, row);
            }
            const int32_t cnt = (int32_t) v.size();
            fwrite(&cnt, 4, 1, out);
            fwrite(v.data(), 4, v.size(), out);
            n_rows++;
        }
        fclose(out);
        fprintf(stderr, "decoded %d rows\n", n_rows);
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
