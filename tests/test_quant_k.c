/* Numerical oracle for qw3 scalar K-quant decoders.
 *
 * This test intentionally includes qw3.c so its internal reference decoders
 * can be checked directly against the exported ggml routines from the local
 * llama.cpp build. It is not part of the runtime binary.
 */
#include "../qw3.c"

extern void dequantize_row_q2_K(const void *x, float *y, int64_t k);
extern void dequantize_row_q3_K(const void *x, float *y, int64_t k);
extern void dequantize_row_q4_K(const void *x, float *y, int64_t k);
extern void dequantize_row_q5_K(const void *x, float *y, int64_t k);
extern void dequantize_row_iq2_xxs(const void *x, float *y, int64_t k);
extern void dequantize_row_iq2_xs(const void *x, float *y, int64_t k);
extern void dequantize_row_iq2_s(const void *x, float *y, int64_t k);
extern void dequantize_row_iq3_xxs(const void *x, float *y, int64_t k);
extern void dequantize_row_iq4_nl(const void *x, float *y, int64_t k);
extern void dequantize_row_iq3_s(const void *x, float *y, int64_t k);
extern void dequantize_row_iq4_xs(const void *x, float *y, int64_t k);
extern void dequantize_row_q6_K(const void *x, float *y, int64_t k);

typedef void (*reference_fn)(const void *, float *, int64_t);

static uint32_t test_rng = 0x51a7c3d9u;

static uint8_t next_byte(void) {
    test_rng = test_rng * 1664525u + 1013904223u;
    return (uint8_t)(test_rng >> 24);
}

static void store_u16(uint8_t *p, uint16_t v) {
    memcpy(p, &v, sizeof(v));
}

static void set_finite_scales(uint32_t type, uint8_t *block) {
    const uint16_t one = 0x3c00u;
    const uint16_t half = 0x3800u;
    switch (type) {
    case QW3_TENSOR_Q2_K:
        store_u16(block + 80, one);
        store_u16(block + 82, half);
        break;
    case QW3_TENSOR_Q3_K:
        store_u16(block + 108, one);
        break;
    case QW3_TENSOR_Q4_K:
    case QW3_TENSOR_Q5_K:
        store_u16(block, one);
        store_u16(block + 2, half);
        break;
    case QW3_TENSOR_IQ2_XXS:
    case QW3_TENSOR_IQ2_XS:
    case QW3_TENSOR_IQ2_S:
    case QW3_TENSOR_IQ3_XXS:
    case QW3_TENSOR_IQ4_NL:
        store_u16(block, one);
        break;
    default:
        abort();
    }
}

static int check_type(uint32_t type, reference_fn reference) {
    uint64_t bytes = cpu_quant_k_block_size(type);
    uint64_t elems = QW3_QK_K;
    bool is_iq = false;
    if (bytes == 0) {
        bytes = cpu_quant_iq_block_size(type);
        elems = cpu_quant_iq_block_elems(type);
        is_iq = true;
    }
    uint8_t block[176];
    float actual[QW3_QK_K];
    float expected[QW3_QK_K];

    for (int trial = 0; trial < 64; trial++) {
        for (uint64_t i = 0; i < bytes; i++) block[i] = next_byte();
        set_finite_scales(type, block);
        if (!(is_iq ? cpu_dequant_iq_block(type, block, actual) :
                      cpu_dequant_k_block(type, block, actual))) return 1;
        reference(block, expected, (int64_t)elems);
        for (uint64_t i = 0; i < elems; i++) {
            const float diff = fabsf(actual[i] - expected[i]);
            if (diff > 1.0e-6f) {
                fprintf(stderr,
                        "test-quant-k: %s trial=%d element=%d "
                        "actual=%.9g expected=%.9g diff=%.9g\n",
                        tensor_type_name(type), trial, (int)i, actual[i],
                        expected[i], diff);
                return 1;
            }
        }
    }
    return 0;
}

static int check_dot_type(uint32_t type, uint64_t bytes,
                          reference_fn reference) {
    uint8_t block[210];
    float values[QW3_QK_K];
    float x[QW3_QK_K];
    for (int trial = 0; trial < 32; trial++) {
        for (uint64_t i = 0; i < bytes; i++) block[i] = next_byte();
        if (type == QW3_TENSOR_Q6_K) store_u16(block + 208, 0x3c00u);
        else store_u16(block, 0x3c00u);
        for (int i = 0; i < QW3_QK_K; i++) {
            x[i] = ((int)next_byte() - 127) / 127.0f;
        }
        reference(block, values, QW3_QK_K);
        float expected = 0.0f;
        for (int i = 0; i < QW3_QK_K; i++) expected += values[i] * x[i];
        float actual = type == QW3_TENSOR_Q6_K ?
            cpu_dot_q6_k_row(block, x, QW3_QK_K) :
            type == QW3_TENSOR_IQ3_S ?
                cpu_dot_iq3_s_row(block, x, QW3_QK_K) :
                cpu_dot_iq4_xs_row(block, x, QW3_QK_K);
        const float tolerance = 1.0e-4f * fmaxf(1.0f, fabsf(expected));
        if (fabsf(actual - expected) > tolerance) {
            fprintf(stderr,
                    "test-quant-dot: %s trial=%d actual=%.9g "
                    "expected=%.9g diff=%.9g\n",
                    tensor_type_name(type), trial, actual, expected,
                    fabsf(actual - expected));
            return 1;
        }
    }
    return 0;
}

int main(void) {
    if (check_type(QW3_TENSOR_Q2_K, dequantize_row_q2_K) != 0 ||
        check_type(QW3_TENSOR_Q3_K, dequantize_row_q3_K) != 0 ||
        check_type(QW3_TENSOR_Q4_K, dequantize_row_q4_K) != 0 ||
        check_type(QW3_TENSOR_Q5_K, dequantize_row_q5_K) != 0 ||
        check_type(QW3_TENSOR_IQ2_XXS, dequantize_row_iq2_xxs) != 0 ||
        check_type(QW3_TENSOR_IQ2_XS, dequantize_row_iq2_xs) != 0 ||
        check_type(QW3_TENSOR_IQ2_S, dequantize_row_iq2_s) != 0 ||
        check_type(QW3_TENSOR_IQ3_XXS, dequantize_row_iq3_xxs) != 0 ||
        check_type(QW3_TENSOR_IQ4_NL, dequantize_row_iq4_nl) != 0 ||
        check_dot_type(QW3_TENSOR_IQ3_S, 110, dequantize_row_iq3_s) != 0 ||
        check_dot_type(QW3_TENSOR_IQ4_XS, 136, dequantize_row_iq4_xs) != 0 ||
        check_dot_type(QW3_TENSOR_Q6_K, 210, dequantize_row_q6_K) != 0) {
        return 1;
    }
    puts("test-quant-reference: ok (Q2_K Q3_K Q4_K Q5_K "
         "IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_NL IQ4_XS Q6_K)");
    return 0;
}
