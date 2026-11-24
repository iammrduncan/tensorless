#pragma once

// Row decoders for LookAtMyQuant's packed Inkling-2-Mini formats (inkling2-packed-v1), exact ports of
// runtime_cuda/inkling/formats.py: every value is the code's table value times the row's f16 scale, in float32,
// in the stored (rotated) basis. Header-only so that host code and CUDA kernels share one definition.
//
// Format ids: trellis 0-6 (ht1, ht2, ht3, rtr1, rtr2, rtr3, rtr4: the pack's own ids), NVFP4 7, lm6 8, lm8 9.

#include <stdint.h>
#include <string.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define INK2_FN static __host__ __device__ inline
#else
#define INK2_FN static inline
#endif

enum ink2_fmt {
    INK2_HT1 = 0, INK2_HT2 = 1, INK2_HT3 = 2,
    INK2_RTR1 = 3, INK2_RTR2 = 4, INK2_RTR3 = 5, INK2_RTR4 = 6,
    INK2_NVFP4 = 7, INK2_LM6 = 8, INK2_LM8 = 9,
};

// (k bits per weight, g weights per state, log2 of the table size, hashed)
struct ink2_trellis_params { int k, g, log2K, hashed; };

INK2_FN struct ink2_trellis_params ink2_trellis(int fmt) {
    struct ink2_trellis_params p = {0, 0, 0, 0};
    switch (fmt) {
        case INK2_HT1:  p.k = 1; p.g = 4; p.log2K =  9; p.hashed = 1; break;
        case INK2_HT2:  p.k = 2; p.g = 4; p.log2K =  9; p.hashed = 1; break;
        case INK2_HT3:  p.k = 3; p.g = 2; p.log2K = 10; p.hashed = 1; break;
        case INK2_RTR1: p.k = 1; p.g = 1; p.log2K = 12; p.hashed = 0; break;
        case INK2_RTR2: p.k = 2; p.g = 1; p.log2K = 12; p.hashed = 0; break;
        case INK2_RTR3: p.k = 3; p.g = 1; p.log2K = 12; p.hashed = 0; break;
        case INK2_RTR4: p.k = 4; p.g = 1; p.log2K = 12; p.hashed = 0; break;
        default: break;
    }
    return p;
}

INK2_FN int ink2_is_trellis(int fmt) { return fmt >= 0 && fmt <= 6; }

INK2_FN float ink2_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else { // subnormal: normalize
            int e = -1;
            do { mant <<= 1; e++; } while ((mant & 0x400u) == 0);
            mant &= 0x3ffu;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// The table index of the group of g columns ending at column `last`, for a row of n weights stored as n*k/32 words.
INK2_FN uint32_t ink2_trellis_index(const uint32_t * words, int n, int last, struct ink2_trellis_params p) {
    const int wpr = (n * p.k) / 32;
    const int bp  = (n - 1 - last) * p.k;
    const int wi  = bp >> 5, off = bp & 31;
    const uint64_t lo = words[wi];
    const uint64_t hi = wi + 1 < wpr ? (uint64_t) words[wi + 1] : 0; // bits past the end read as zero
    const uint32_t state = (uint32_t) (((lo | (hi << 32)) >> off) & 4095u);
    if (!p.hashed) {
        return state;
    }
    return (uint32_t) (state * 0x9E3779B1u) >> (32 - p.log2K);
}

// One trellis row: out[j] = lut[idx(j)][j % g] * scale. lut is f16 [K, g].
INK2_FN void ink2_decode_trellis_row(const uint32_t * words, uint16_t scale_f16, const uint16_t * lut, int fmt, int n,
                                     float * out) {
    const struct ink2_trellis_params p = ink2_trellis(fmt);
    const float s = ink2_f16_to_f32(scale_f16);
    for (int j = 0; j < n; j += p.g) {
        const uint32_t idx = ink2_trellis_index(words, n, j + p.g - 1, p);
        for (int t = 0; t < p.g; ++t) {
            out[j + t] = ink2_f16_to_f32(lut[idx * p.g + t]) * s;
        }
    }
}

// Lloyd-Max index j of a row (row bytes start at the row's first code byte; n is a multiple of 4 for lm6).
INK2_FN int ink2_lm_index(const uint8_t * row, int k, int j) {
    if (k == 8) {
        return row[j];
    }
    const uint8_t * t = row + (j >> 2) * 3;
    const uint32_t v = (uint32_t) t[0] | ((uint32_t) t[1] << 8) | ((uint32_t) t[2] << 16);
    return (int) ((v >> (6 * (j & 3))) & 63u);
}

// One Lloyd-Max row: out[j] = levels[idx] * scale. codes is the matrix's flat code bytes.
INK2_FN void ink2_decode_lm_row(const uint8_t * codes, uint16_t scale_f16, const float * levels, int k, int n, int64_t r,
                                float * out) {
    const uint8_t * row = codes + (r * (int64_t) n * k) / 8;
    const float s = ink2_f16_to_f32(scale_f16);
    for (int j = 0; j < n; ++j) {
        out[j] = levels[ink2_lm_index(row, k, j)] * s;
    }
}

// One NVFP4 row: (e2m1[nibble] * e4m3[block scale]) * outer, low nibble first, 16 weights a block.
INK2_FN void ink2_decode_nvfp4_row(const uint8_t * codes_row, const uint8_t * scales_row, float outer,
                                   const float * e2m1, const float * e4m3, int n, float * out) {
    for (int j = 0; j < n; ++j) {
        const uint8_t c = codes_row[j >> 1];
        const float v = e2m1[(j & 1) ? (c >> 4) : (c & 0xF)];
        out[j] = (v * e4m3[scales_row[j >> 4]]) * outer;
    }
}

// The base of matrix `e` of a slab (or the matrix itself when e < 0), and its row scales, for any format.
// Rows of a matrix: trellis n*k/32 words, Lloyd-Max n*k/8 bytes, NVFP4 n/2 bytes (and n/16 scale bytes).
struct ink2_matrix {
    int fmt, n;               // format and columns
    const uint8_t  * codes;   // first code byte of the matrix
    const uint8_t  * scales;  // f16 row scales, or NVFP4 block scales
    const void     * table;   // f16 LUT, f32 levels
    const float    * e4m3;    // NVFP4
    float            outer;   // NVFP4
};

INK2_FN void ink2_decode_row(const struct ink2_matrix * m, int64_t r, float * out) {
    const int n = m->n;
    if (ink2_is_trellis(m->fmt)) {
        const struct ink2_trellis_params p = ink2_trellis(m->fmt);
        const int64_t wpr = (int64_t) n * p.k / 32;
        ink2_decode_trellis_row((const uint32_t *) m->codes + r * wpr, ((const uint16_t *) m->scales)[r],
                                (const uint16_t *) m->table, m->fmt, n, out);
    } else if (m->fmt == INK2_LM6 || m->fmt == INK2_LM8) {
        ink2_decode_lm_row(m->codes, ((const uint16_t *) m->scales)[r], (const float *) m->table,
                           m->fmt == INK2_LM6 ? 6 : 8, n, r, out);
    } else {
        const float e2m1[16] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };
        ink2_decode_nvfp4_row(m->codes + r * (n / 2), m->scales + r * (n / 16), m->outer, e2m1, m->e4m3, n, out);
    }
}
