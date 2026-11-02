// Golden tests for the PN signed mid-tread types 124 (pn3t) / 125 (pn4t).
//
// Goldens and the normative decoder come from the PNX research reference (see tensorless docs/pn-format.md):
//   pnx_lib.py, pnx_verify.out ([11], [12], [15]), pnx_ref_check.c.
// Everything here is bit-level: the dequantized fp32 words are compared bit for bit (including the
// sign of a zero result, which comes from the sign of the fp16 scale), and ggml_validate_row_data is
// checked against the accept / reject rules of the format.
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>

static int failures = 0;

static void fail(const char * what) {
    ++failures;
    printf("FAIL: %s\n", what);
}

static uint32_t bits_of(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

static std::string unhex(const char * hex) {
    std::string out;
    for (const char * p = hex; *p; p += 2) {
        out.push_back((char) strtol(std::string(p, 2).c_str(), nullptr, 16));
    }
    return out;
}

static float f16_val(uint16_t b) {
    // binary16 -> float, exact
    const uint32_t sign = (uint32_t) (b & 0x8000) << 16;
    const uint32_t exp  = (b >> 10) & 0x1F;
    const uint32_t man  = b & 0x3FF;
    uint32_t out;
    if (exp == 0) {
        if (man == 0) {
            out = sign;
        } else {
            int e = -14;
            uint32_t m = man;
            while (!(m & 0x400)) { m <<= 1; --e; }
            m &= 0x3FF;
            out = sign | ((uint32_t) (e + 127) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        out = sign | 0x7F800000u | (man << 13);
    } else {
        out = sign | ((exp + 112) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &out, 4);
    return f;
}

// reference value of the format: w = d * (n - z), both steps exact in fp32
static float ref_value(uint16_t d_bits, int n, int z) {
    return f16_val(d_bits) * ((float) n - (float) z);
}

static const uint16_t D_PN3T = 0xA400; // -0.015625
static const uint16_t D_PN4T = 0xA800; // -0.03125

static void check_dequant(ggml_type type, const char * block_hex, int group, int bits, int z,
                          const char * what) {
    const std::string blk = unhex(block_hex);
    if (blk.size() != 34) {
        fail(what);
        return;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    if (!traits->is_quantized || traits->to_float == nullptr) {
        fail(what);
        return;
    }
    if (ggml_blck_size(type) != group || ggml_type_size(type) != 34) {
        printf("FAIL: %s: blck_size=%lld type_size=%lld\n", what,
               (long long) ggml_blck_size(type), (long long) ggml_type_size(type));
        ++failures;
        return;
    }

    std::vector<float> y(group);
    traits->to_float(blk.data(), y.data(), group);

    const uint16_t d_bits = (uint16_t) ((uint8_t) blk[0] | ((uint16_t) (uint8_t) blk[1] << 8));
    const int per_word = (bits == 3) ? 10 : 8;
    const int mask = (1 << bits) - 1;

    for (int j = 0; j < group; ++j) {
        const int w = j / per_word;
        const uint32_t * wp;
        uint32_t v;
        memcpy(&v, blk.data() + 2 + 4*w, 4);
        wp = &v;
        const int sh = ((bits == 3) ? 3*((j % 10) >> 1) : 4*((j % 8) >> 1)) + ((j & 1) ? 16 : 0);
        const int n = (*wp >> sh) & mask;
        const float want = ref_value(d_bits, n, z);
        if (bits_of(want) != bits_of(y[j])) {
            printf("FAIL: %s: element %d code %d: got 0x%08X want 0x%08X\n", what, j, n,
                   bits_of(y[j]), bits_of(want));
            ++failures;
            return;
        }
    }
    printf("ok: %s dequantizes %d weights bit-exactly (d = 0x%04X)\n", what, group, d_bits);
}

static void check_zero_sign(ggml_type type, uint16_t d_bits, int group, int zero_code, const char * what) {
    // a code at the zero level must dequantize to d*0, i.e. keep the sign of d (pnx_verify.out [11])
    std::string blk(34, '\0');
    blk[0] = (char) (d_bits & 0xFF);
    blk[1] = (char) (d_bits >> 8);
    const int per_word = (group == 80) ? 10 : 8;
    const int bits     = (group == 80) ? 3 : 4;
    for (int j = 0; j < group; ++j) {
        const int w = j / per_word;
        uint32_t v;
        memcpy(&v, blk.data() + 2 + 4*w, 4);
        const int sh = ((bits == 3) ? 3*((j % 10) >> 1) : 4*((j % 8) >> 1)) + ((j & 1) ? 16 : 0);
        v |= (uint32_t) zero_code << sh;
        memcpy((char *) blk.data() + 2 + 4*w, &v, 4);
    }
    std::vector<float> y(group);
    ggml_get_type_traits(type)->to_float(blk.data(), y.data(), group);
    const bool want_negative_zero = (d_bits & 0x8000) != 0;
    for (int j = 0; j < group; ++j) {
        const uint32_t u = bits_of(y[j]);
        if (u != (want_negative_zero ? 0x80000000u : 0x00000000u)) {
            printf("FAIL: %s: element %d is 0x%08X, expected %s zero\n", what, j, u,
                   want_negative_zero ? "-0.0" : "+0.0");
            ++failures;
            return;
        }
    }
    printf("ok: %s keeps the sign of zero\n", what);
}

static void check_validate(ggml_type type, const std::string & row, bool want, const char * what) {
    const bool got = ggml_validate_row_data(type, row.data(), row.size());
    if (got != want) {
        printf("FAIL: %s: validate returned %d, expected %d\n", what, (int) got, (int) want);
        ++failures;
        return;
    }
    printf("ok: %s -> validate %d\n", what, (int) got);
}

int main(void) {
    // [11] exact goldens (pnx_verify.out:45-54)
    check_dequant(GGML_TYPE_PN3T,
        "00a4eb33100d7d56a221cf7a3444591f8668eb33100d7d56a221cf7a3444591f8668", 80, 3, 4,
        "[11] pn3t block (codes (5k+3)%8, d = -0.015625)");
    check_dequant(GGML_TYPE_PN4T,
        "00a8f1bd68247935e0acf1bd68247935e0acf1bd68247935e0acf1bd68247935e0ac", 64, 4, 8,
        "[11] pn4t block (codes (7k+1)%16, d = -0.03125)");
    // [12] rounding-sensitive goldens (scale 0xA24C)
    check_dequant(GGML_TYPE_PN3T,
        "4ca2f82ec6094f5d1d64a67a3c35c756863875647f377854602cf97c513d1429ca25", 80, 3, 4,
        "[12] pn3t block (d = -0.012298583984375)");
    check_dequant(GGML_TYPE_PN4T,
        "4ca2ee556e3f4ee1295c100612e76a097e2fde78289a30c3c22960c160cf6d9a82a2", 64, 4, 8,
        "[12] pn4t block (d = -0.012298583984375)");

    // first six dequantized values of [11], as printed by the reference
    {
        std::string blk = unhex("00a4eb33100d7d56a221cf7a3444591f8668eb33100d7d56a221cf7a3444591f8668");
        float y[80];
        ggml_get_type_traits(GGML_TYPE_PN3T)->to_float(blk.data(), y, 80);
        const float want[6] = { 0.015625f, 0.0625f, -0.015625f, 0.03125f, -0.046875f, -0.0f };
        for (int j = 0; j < 6; ++j) {
            if (bits_of(y[j]) != bits_of(want[j])) {
                printf("FAIL: [11] pn3t dequant[%d] = 0x%08X want 0x%08X\n", j, bits_of(y[j]), bits_of(want[j]));
                ++failures;
            }
        }
        blk = unhex("00a8f1bd68247935e0acf1bd68247935e0acf1bd68247935e0acf1bd68247935e0ac");
        ggml_get_type_traits(GGML_TYPE_PN4T)->to_float(blk.data(), y, 64);
        const float want4[6] = { 0.21875f, -0.0f, -0.21875f, 0.0625f, -0.15625f, 0.125f };
        for (int j = 0; j < 6; ++j) {
            if (bits_of(y[j]) != bits_of(want4[j])) {
                printf("FAIL: [11] pn4t dequant[%d] = 0x%08X want 0x%08X\n", j, bits_of(y[j]), bits_of(want4[j]));
                ++failures;
            }
        }
    }

    check_zero_sign(GGML_TYPE_PN3T, 0xA400, 80, 4, "pn3t code 4 with d < 0");
    check_zero_sign(GGML_TYPE_PN3T, 0x2400, 80, 4, "pn3t code 4 with d > 0");
    check_zero_sign(GGML_TYPE_PN4T, 0xA800, 64, 8, "pn4t code 8 with d < 0");
    check_zero_sign(GGML_TYPE_PN4T, 0x0000, 64, 8, "pn4t canonical zero group");

    // [15] validate rules
    check_validate(GGML_TYPE_PN3T, unhex("00a4eb33100d7d56a221cf7a3444591f8668eb33100d7d56a221cf7a3444591f8668"),
                   true, "[15] signed mid-tread pn3t golden");
    check_validate(GGML_TYPE_PN4T, unhex("00a8f1bd68247935e0acf1bd68247935e0acf1bd68247935e0acf1bd68247935e0ac"),
                   true, "[15] signed mid-tread pn4t golden");
    {
        std::string blk = unhex("00a8f1bd68247935e0acf1bd68247935e0acf1bd68247935e0acf1bd68247935e0ac");
        std::string nan = blk; nan[0] = (char) 0x00; nan[1] = (char) 0x7E;  // d = NaN
        check_validate(GGML_TYPE_PN4T, nan, false, "[15] NaN d in pn4t");
        std::string inf = blk; inf[0] = (char) 0x00; inf[1] = (char) 0xFC;  // d = -Inf
        check_validate(GGML_TYPE_PN4T, inf, false, "[15] -Inf d in pn4t");

        std::string res = blk; res[2] = (char) ((uint8_t) res[2] | 0x80);   // reserved bit 15 of word 0
        check_validate(GGML_TYPE_PN3T, res, true == false, "[15] pn3t reserved bit 15 set");
        std::string res31 = blk; res31[5] = (char) ((uint8_t) res31[5] | 0x80); // reserved bit 31 of word 0
        check_validate(GGML_TYPE_PN3T, res31, false, "[15] pn3t reserved bit 31 set");
        // pn4t has no reserved bits: the same byte pattern is valid
        check_validate(GGML_TYPE_PN4T, res, true, "[15] pn4t nibble word with bit 15 set");
    }

    // row geometry
    if (ggml_row_size(GGML_TYPE_PN3T, 5120) != 64*34 || ggml_row_size(GGML_TYPE_PN4T, 17408) != 272*34) {
        printf("FAIL: ggml_row_size: pn3t(5120)=%zu pn4t(17408)=%zu\n",
               ggml_row_size(GGML_TYPE_PN4T, 17408), ggml_row_size(GGML_TYPE_PN3T, 5120));
        ++failures;
    }
    if (!ggml_is_quantized(GGML_TYPE_PN3T) || !ggml_is_quantized(GGML_TYPE_PN4T)) {
        fail("pn3t / pn4t must be quantized types");
    }
    if (std::strcmp(ggml_type_name(GGML_TYPE_PN3T), "pn3t") || std::strcmp(ggml_type_name(GGML_TYPE_PN4T), "pn4t")) {
        fail("type names");
    }

    // round trip: quantize_row_pn3t_ref / pn4t_ref must reproduce the golden block for the [11] code sets
    {
        float w[80];
        for (int k = 0; k < 80; ++k) w[k] = ref_value(D_PN3T, (5*k + 3) % 8, 4);
        std::vector<uint8_t> out(ggml_row_size(GGML_TYPE_PN3T, 80));
        ggml_quantize_chunk(GGML_TYPE_PN3T, w, out.data(), 0, 1, 80, nullptr);
        const std::string want = unhex("00a4eb33100d7d56a221cf7a3444591f8668eb33100d7d56a221cf7a3444591f8668");
        if (std::memcmp(out.data(), want.data(), 34) != 0) {
            printf("FAIL: quantize_row_pn3t_ref did not reproduce the [11] block\n");
            for (int i = 0; i < 34; ++i) printf("%02x", out[i]);
            printf("\n");
            ++failures;
        } else {
            printf("ok: quantize_row_pn3t_ref reproduces the [11] pn3t block byte for byte\n");
        }
    }

    if (failures) {
        printf("%d PN golden checks FAILED\n", failures);
        return 1;
    }
    printf("PN golden tests passed\n");
    return 0;
}
