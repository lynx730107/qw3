/* =========================================================================
 * qw3.c - Qwen3.6-35B-A3B inference engine.
 * =========================================================================
 *
 * A minimalist, single-model inference engine following the ds4.c philosophy:
 * one model, zero frameworks, vertical code.
 *
 * Qwen3.6-35B-A3B is a hybrid Mixture-of-Experts transformer with two
 * attention mechanisms alternating in a 3:1 pattern across 40 layers:
 *
 *   - 30 layers of Gated DeltaNet (linear attention with recurrent state)
 *   - 10 layers of standard GQA (Grouped Query Attention)
 *   - All 40 layers use MoE FFN: 256 experts, top-8 routing + 1 shared
 *
 * This file is deliberately vertical: it owns GGUF loading, the fixed
 * Qwen3.6-35B-A3B tensor layout, CPU reference kernels, the whole-model
 * Metal graph driver, and tokenizer wiring.
 */

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <ctype.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>

#include "qw3.h"

#ifndef QW3_NO_METAL
#include "qw3_metal.h"
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* =========================================================================
 * Fixed Qwen3.6-35B-A3B Shape.
 * =========================================================================
 *
 * These constants define the single model this program accepts.  The weight
 * binder and metadata validator check the GGUF against the same numbers so
 * the rest of the inference code can use simple fixed-size paths.
 *
 * Architecture: qwen3_5_moe (hybrid Gated DeltaNet + GQA + Sparse MoE)
 */

#define QW3_NEG_INF (-1.0e30f)
#define QW3_RMS_EPS ( 1.0e-6f)

enum {
    QW3_N_LAYER            = 40,
    QW3_N_EMBD             = 2048,
    QW3_N_VOCAB            = 248320,

    /* Full attention (GQA) — layers at indices 3,7,11,...,39 */
    QW3_N_HEAD             = 16,      /* query heads */
    QW3_N_HEAD_KV          = 2,       /* key/value heads (GQA ratio 8:1) */
    QW3_N_HEAD_DIM         = 256,     /* per-head dimension */

    /* Linear attention (Gated DeltaNet) — layers at indices 0,1,2,4,5,6,... */
    QW3_N_LINEAR_QK_HEADS  = 16,      /* query/key heads */
    QW3_N_LINEAR_V_HEADS   = 32,      /* value heads */
    QW3_N_LINEAR_HEAD_DIM  = 128,     /* per-head dimension */
    QW3_N_LINEAR_CONV_K    = 4,       /* short causal convolution kernel */

    /* MoE (all layers) */
    QW3_N_EXPERT           = 256,
    QW3_N_EXPERT_USED      = 8,       /* top-k routing */
    QW3_N_EXPERT_SHARED    = 1,
    QW3_N_FF_EXP           = 512,     /* expert intermediate dim */
    QW3_N_FF_SHARED        = 512,     /* shared expert intermediate dim */

    /* Full-attention interval: every 4th layer is full attention */
    QW3_FULL_ATTN_INTERVAL = 4,
    QW3_EXPECTED_TENSORS   = 733,
};

#define QW3_QK_K 256

/* RoPE config — partial rotation (25% of head_dim). */
#define QW3_ROPE_PARTIAL_FACTOR (0.25f)
#define QW3_ROPE_THETA          (10000000.0f)
#define QW3_ROPE_DIM            ((int)(QW3_N_HEAD_DIM * QW3_ROPE_PARTIAL_FACTOR))
/* QW3_ROPE_DIM = 64 (25% of 256) */

/* Layer type: true = full attention (GQA), false = linear attention (DeltaNet).
 * Pattern: [linear, linear, linear, full] × 10 */
static inline bool qw3_layer_is_full_attention(uint32_t il) {
    return ((il + 1) % QW3_FULL_ATTN_INTERVAL) == 0;
}

/* =========================================================================
 * Shared Helpers — allocation, errors, timing.
 * =========================================================================
 * Mirrors ds4.c utilities.
 */

#define QW3_GGUF_MAGIC 0x46554747u /* "GGUF", little endian. */
#define QW3_MAX_DIMS   8
#define QW3_TENSOR_ANY UINT32_MAX

typedef struct {
    const char *ptr;
    uint64_t len;
} qw3_str;

typedef qw3_tokens token_vec;

typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;
    char error[256];
} qw3_cursor;

static void qw3_die(const char *msg) {
    fprintf(stderr, "qw3: %s\n", msg);
    exit(1);
}

static void qw3_die_errno(const char *what, const char *path) {
    fprintf(stderr, "qw3: %s '%s': %s\n", what, path, strerror(errno));
    exit(1);
}

static void *qw3_xcalloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) qw3_die("out of memory");
    return p;
}

static void *qw3_xmalloc(size_t size) {
    void *p = malloc(size);
    if (!p) qw3_die("out of memory");
    return p;
}

static void *qw3_xrealloc(void *ptr, size_t size) {
    void *p = realloc(ptr, size);
    if (!p) qw3_die("out of memory");
    return p;
}

static double qw3_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static uint64_t tensor_cols_qg(void) {
    return (uint64_t)QW3_N_HEAD * QW3_N_HEAD_DIM * 2;
}

static uint64_t tensor_cols_kv(void) {
    return (uint64_t)QW3_N_HEAD_KV * QW3_N_HEAD_DIM;
}

static uint64_t tensor_linear_inner(void) {
    return (uint64_t)QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM;
}

static uint64_t tensor_linear_qkv(void) {
    return (uint64_t)QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM * 2 +
           tensor_linear_inner();
}

static uint16_t qw3_load_u16(const void *p) {
    uint16_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static float qw3_f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp  = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static float qw3_bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* =========================================================================
 * Logging.
 * =========================================================================
 */

static const char *qw3_log_color_code(qw3_log_type type) {
    switch (type) {
    case QW3_LOG_PREFILL:
    case QW3_LOG_TIMING:    return "\x1b[36m";
    case QW3_LOG_GENERATION:
    case QW3_LOG_OK:        return "\x1b[32m";
    case QW3_LOG_KVCACHE:   return "\x1b[33m";
    case QW3_LOG_TOOL:      return "\x1b[90m";
    case QW3_LOG_WARNING:   return "\x1b[38;5;208m";
    case QW3_LOG_ERROR:     return "\x1b[31m";
    default:                return "";
    }
}

bool qw3_log_is_tty(FILE *fp) {
    int fd = fileno(fp);
    return fd >= 0 && isatty(fd) != 0;
}

void qw3_log(FILE *fp, qw3_log_type type, const char *fmt, ...) {
    const bool colorize = type != QW3_LOG_DEFAULT && qw3_log_is_tty(fp);
    if (colorize) fputs(qw3_log_color_code(type), fp);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(fp, fmt, ap);
    va_end(ap);
    if (colorize) fputs("\x1b[0m", fp);
}

/* =========================================================================
 * GGUF Cursor — byte-level reading of GGUF header/metadata.
 * =========================================================================
 */

static void cursor_error(qw3_cursor *c, const char *msg) {
    if (c->error[0] == '\0') {
        snprintf(c->error, sizeof(c->error), "%s at byte %" PRIu64,
                 msg, c->pos);
    }
}

static bool cursor_has(qw3_cursor *c, uint64_t n) {
    if (n > c->size || c->pos > c->size - n) {
        cursor_error(c, "truncated GGUF file");
        return false;
    }
    return true;
}

static bool cursor_read(qw3_cursor *c, void *dst, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    memcpy(dst, c->base + c->pos, (size_t)n);
    c->pos += n;
    return true;
}

static bool cursor_skip(qw3_cursor *c, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    c->pos += n;
    return true;
}

static bool cursor_u32(qw3_cursor *c, uint32_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_u64(qw3_cursor *c, uint64_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_string(qw3_cursor *c, qw3_str *s) {
    uint64_t len;
    if (!cursor_u64(c, &len)) return false;
    if (!cursor_has(c, len)) return false;
    s->ptr = (const char *)(c->base + c->pos);
    s->len = len;
    c->pos += len;
    return true;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    uint64_t rem = value % alignment;
    return rem ? value + alignment - rem : value;
}

static bool qw3_streq(qw3_str s, const char *z) {
    size_t n = strlen(z);
    return s.len == n && memcmp(s.ptr, z, n) == 0;
}

static uint64_t hash_bytes(const char *p, uint64_t len) {
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < len; i++) {
        h ^= (uint8_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static bool bytes_equal(const char *a, uint64_t alen,
                        const char *b, uint64_t blen) {
    return alen == blen && memcmp(a, b, (size_t)alen) == 0;
}

/* =========================================================================
 * Token vector helpers.
 * =========================================================================
 */

static void token_vec_push(token_vec *tv, int token) {
    if (tv->len == tv->cap) {
        tv->cap = tv->cap ? tv->cap * 2 : 64;
        tv->v = qw3_xrealloc(tv->v, (size_t)tv->cap * sizeof(int));
    }
    tv->v[tv->len++] = token;
}

void qw3_tokens_push(qw3_tokens *tv, int token) { token_vec_push(tv, token); }

void qw3_tokens_free(qw3_tokens *tv) {
    free(tv->v);
    tv->v = NULL;
    tv->len = tv->cap = 0;
}

void qw3_tokens_copy(qw3_tokens *dst, const qw3_tokens *src) {
    qw3_tokens_free(dst);
    if (src->len == 0) return;
    dst->cap = src->len;
    dst->len = src->len;
    dst->v = qw3_xmalloc((size_t)dst->cap * sizeof(int));
    memcpy(dst->v, src->v, (size_t)dst->len * sizeof(int));
}

bool qw3_tokens_starts_with(const qw3_tokens *tokens,
                             const qw3_tokens *prefix) {
    if (prefix->len > tokens->len) return false;
    return memcmp(tokens->v, prefix->v,
                  (size_t)prefix->len * sizeof(int)) == 0;
}

/* =========================================================================
 * GGUF Tensor types — the quant formats this engine reads.
 * =========================================================================
 *
 * For IQ4_XS (the target quantization), we need:
 *   - IQ4_XS blocks for quantized expert weights
 *   - F16 for embeddings, norms, projections
 *   - F32 for biases and small tensors
 *   - BF16 as the model's native dtype
 */

enum {
    QW3_TENSOR_F32     = 0,
    QW3_TENSOR_F16     = 1,
    QW3_TENSOR_Q4_0    = 2,
    QW3_TENSOR_Q4_1    = 3,
    QW3_TENSOR_Q5_0    = 6,
    QW3_TENSOR_Q5_1    = 7,
    QW3_TENSOR_Q8_0    = 8,
    QW3_TENSOR_Q2_K    = 10,
    QW3_TENSOR_Q3_K    = 11,
    QW3_TENSOR_Q4_K    = 12,
    QW3_TENSOR_Q5_K    = 13,
    QW3_TENSOR_Q6_K    = 14,
    QW3_TENSOR_Q8_K    = 15,
    QW3_TENSOR_IQ2_XXS = 16,
    QW3_TENSOR_IQ2_XS  = 17,
    QW3_TENSOR_IQ3_XXS = 18,
    QW3_TENSOR_IQ1_S   = 19,
    QW3_TENSOR_IQ4_NL  = 20,
    QW3_TENSOR_IQ3_S   = 21,
    QW3_TENSOR_IQ2_S   = 22,
    QW3_TENSOR_IQ4_XS  = 23,
    QW3_TENSOR_I8      = 24,
    QW3_TENSOR_I16     = 25,
    QW3_TENSOR_I32     = 26,
    QW3_TENSOR_I64     = 27,
    QW3_TENSOR_F64     = 28,
    QW3_TENSOR_BF16    = 30,
};

/* =========================================================================
 * GGUF Model — mmap'd model file with parsed tensor directory.
 * =========================================================================
 */

typedef struct {
    const char *name;
    uint64_t name_len;
    uint32_t type;
    uint32_t ndim;
    uint64_t dim[QW3_MAX_DIMS];
    uint64_t elements;
    uint64_t rel_offset;
    uint64_t offset; /* byte offset from tensor data start */
} qw3_tensor;

typedef struct {
    int fd;
    uint8_t *map;
    uint64_t map_size;
    uint64_t tensor_data_offset;
    qw3_tensor *tensors;
    uint32_t n_tensors;
    uint64_t n_kv;
    struct {
        qw3_str architecture;
        qw3_str tokenizer_model;
        qw3_str tokenizer_pre;
        int64_t block_count;
        int64_t context_length;
        int64_t embedding_length;
        int64_t head_count;
        int64_t head_count_kv;
        int64_t key_length;
        int64_t value_length;
        int64_t expert_count;
        int64_t expert_used_count;
        int64_t expert_ffn_length;
        int64_t expert_shared_ffn_length;
        int64_t ssm_conv_kernel;
        int64_t ssm_state_size;
        int64_t ssm_group_count;
        int64_t ssm_time_step_rank;
        int64_t ssm_inner_size;
        int64_t full_attention_interval;
        int64_t rope_dimension_count;
        int64_t eos_token_id;
        int64_t bos_token_id;
        int64_t padding_token_id;
        int64_t tokenizer_token_count;
        int64_t tokenizer_merge_count;
        int64_t tokenizer_token_type_count;
        float rope_freq_base;
        float rms_eps;
        bool add_bos_token;
        int rope_sections[4];
        int rope_sections_len;
        qw3_str *token_texts;
        uint32_t *token_types;
        qw3_str *merge_texts;
    } meta;
} qw3_model;

/* =========================================================================
 * Weight binding — fixed Qwen3.6-35B-A3B tensor layout.
 * =========================================================================
 *
 * Every layer has both attention weights (GQA or DeltaNet depending on
 * the layer index) and MoE FFN weights.  The weight binder maps GGUF
 * tensor names to direct pointers, and the validator checks dimensions
 * against the fixed model constants.
 */

typedef struct {
    qw3_tensor *attn_norm;          /* blk.X.attn_norm.weight */
    qw3_tensor *ffn_norm;           /* blk.X.post_attention_norm.weight */

    /* === Full Attention (GQA) weights — only for full_attention layers === */
    qw3_tensor *attn_q_proj;        /* blk.X.attn_q.weight */
    qw3_tensor *attn_q_norm;        /* blk.X.attn_q_norm.weight */
    qw3_tensor *attn_k_proj;        /* blk.X.attn_k.weight */
    qw3_tensor *attn_k_norm;        /* blk.X.attn_k_norm.weight */
    qw3_tensor *attn_v_proj;        /* blk.X.attn_v.weight */
    qw3_tensor *attn_o_proj;        /* blk.X.attn_output.weight */

    /* === Linear Attention (Gated DeltaNet) weights — only for linear layers */
    qw3_tensor *linear_qkv_proj;    /* blk.X.attn_qkv.weight */
    qw3_tensor *linear_gate_proj;   /* blk.X.attn_gate.weight */
    qw3_tensor *linear_ssm_a;       /* blk.X.ssm_a */
    qw3_tensor *linear_ssm_dt_bias; /* blk.X.ssm_dt.bias */
    qw3_tensor *linear_ssm_out;     /* blk.X.ssm_out.weight */
    qw3_tensor *linear_ssm_norm;    /* blk.X.ssm_norm.weight */
    qw3_tensor *linear_ssm_alpha;   /* blk.X.ssm_alpha.weight */
    qw3_tensor *linear_ssm_beta;    /* blk.X.ssm_beta.weight */
    qw3_tensor *linear_conv_weight; /* blk.X.ssm_conv1d.weight */

    /* === MoE FFN weights — all layers === */
    qw3_tensor *ffn_gate_inp;       /* blk.X.ffn_gate_inp.weight */
    qw3_tensor *ffn_gate_inp_shexp; /* blk.X.ffn_gate_inp_shexp.weight */
    qw3_tensor *ffn_gate_exps;      /* blk.X.ffn_gate_exps.weight */
    qw3_tensor *ffn_up_exps;        /* blk.X.ffn_up_exps.weight */
    qw3_tensor *ffn_down_exps;      /* blk.X.ffn_down_exps.weight */
    qw3_tensor *ffn_gate_shared;    /* blk.X.ffn_gate_shexp.weight */
    qw3_tensor *ffn_up_shared;      /* blk.X.ffn_up_shexp.weight */
    qw3_tensor *ffn_down_shared;    /* blk.X.ffn_down_shexp.weight */
} qw3_layer_weights;

typedef struct {
    qw3_tensor *token_embd;
    qw3_tensor *output_norm;
    qw3_tensor *output;             /* lm_head */
    qw3_layer_weights layer[QW3_N_LAYER];
} qw3_weights;

/* =========================================================================
 * Vocab / Tokenizer.
 * =========================================================================
 */

typedef struct {
    uint64_t hash;
    char *key;
    uint64_t key_len;
    int value;
} qw3_table_entry;

typedef struct {
    qw3_table_entry *entries;
    uint32_t cap;
    uint32_t count;
} qw3_table;

typedef struct {
    const char **id_to_text;
    uint64_t *id_to_text_len;
    uint32_t n_vocab;
    qw3_table token_to_id;
    qw3_table merge_rank;
    int bos_id;
    int eos_id;
    int im_start_id;
    int im_end_id;
    int turn_start_id;
    int turn_end_id;
    int think_id;
    int think_end_id;
    int channel_start_id;
    int channel_end_id;
} qw3_vocab;

/* =========================================================================
 * Small string -> int hash table.
 * =========================================================================
 */

static uint32_t table_pow2_for(uint32_t count) {
    uint32_t cap = 16;
    while (cap < count * 2) cap <<= 1;
    return cap;
}

static void table_init(qw3_table *t, uint32_t count_hint) {
    t->cap = table_pow2_for(count_hint ? count_hint : 16);
    t->count = 0;
    t->entries = qw3_xcalloc(t->cap, sizeof(t->entries[0]));
}

static void table_free(qw3_table *t) {
    free(t->entries);
    memset(t, 0, sizeof(*t));
}

static void table_insert(qw3_table *t, const char *key,
                         uint64_t key_len, int value) {
    uint64_t h = hash_bytes(key, key_len);
    uint32_t mask = t->cap - 1;
    for (uint32_t probe = 0; probe < t->cap; probe++) {
        uint32_t slot = (uint32_t)(h + probe) & mask;
        qw3_table_entry *e = &t->entries[slot];
        if (e->key == NULL) {
            e->hash = h;
            e->key = (char *)key;
            e->key_len = key_len;
            e->value = value;
            t->count++;
            return;
        }
        if (e->hash == h && bytes_equal(e->key, e->key_len, key, key_len)) {
            e->value = value;
            return;
        }
    }
    qw3_die("token table is full");
}

static bool table_get(const qw3_table *t, const char *key,
                      uint64_t key_len, int *value) {
    if (t->cap == 0) return false;
    uint64_t h = hash_bytes(key, key_len);
    uint32_t mask = t->cap - 1;
    for (uint32_t probe = 0; probe < t->cap; probe++) {
        uint32_t slot = (uint32_t)(h + probe) & mask;
        const qw3_table_entry *e = &t->entries[slot];
        if (e->key == NULL) return false;
        if (e->hash == h && bytes_equal(e->key, e->key_len, key, key_len)) {
            *value = e->value;
            return true;
        }
    }
    return false;
}

/* =========================================================================
 * DeltaNet Recurrent State.
 * =========================================================================
 *
 * The 30 linear-attention layers each maintain a recurrent state matrix:
 *   S[head][key_dim][value_dim]  (float32 for numerical stability)
 *
 * Plus a short-conv state buffer for the causal convolution:
 *   conv_state[hidden][(conv_k - 1)]
 *
 * This state is independent of context length — a key advantage.
 */

#define QW3_N_LINEAR_LAYERS 30  /* 40 total - 10 full attention */

typedef struct {
    /* Recurrent state: S[layer][head][k_dim * v_dim]
     * Total: 30 × 32 × 128 × 128 = 15,728,640 floats ≈ 60MB */
    float *state;
    /* Short conv state: conv[layer][hidden * (conv_k-1)]
     * Total: 30 × 2048 × 3 = 184,320 floats ≈ 720KB */
    float *conv_state;
    /* Which linear-layer index (0..29) corresponds to each model layer. */
    int linear_layer_map[QW3_N_LAYER];
    int n_linear_layers;
} qw3_deltanet_state;

/* =========================================================================
 * GQA KV Cache — for the 10 full-attention layers.
 * =========================================================================
 *
 * Standard ring-buffer KV cache.  Much simpler than DS4's compressed cache.
 * Each full-attention layer has 2 KV heads × 256 dim.
 */

#define QW3_N_FULL_ATTN_LAYERS 10

typedef struct {
    /* k_cache[layer][pos][n_kv_head * head_dim] */
    float *k_cache;
    /* v_cache[layer][pos][n_kv_head * head_dim] */
    float *v_cache;
    int ctx_size;
    int pos;   /* next write position */
    /* Which full-attention-layer index (0..9) corresponds to each model layer */
    int full_layer_map[QW3_N_LAYER];
    int n_full_layers;
} qw3_kv_cache;

/* =========================================================================
 * Session — one mutable inference timeline.
 * =========================================================================
 */

struct qw3_session {
    qw3_engine *engine;
    qw3_kv_cache kv;
    qw3_deltanet_state dn;
#ifndef QW3_NO_METAL
    qw3_metal_session *metal;
    int metal_n_gpu_layers;
#endif
    qw3_tokens tokens;
    float *logits;      /* [QW3_N_VOCAB] */
    int ctx_size;
    bool valid;
    qw3_session_progress_fn progress_fn;
    void *progress_ud;
};

#define QW3_METAL_LOGITS_DEFER 0
#define QW3_METAL_LOGITS_GPU   1
#define QW3_METAL_LOGITS_READ  2

#ifndef QW3_NO_METAL
static int qw3_metal_session_eval_token_slow_ex(qw3_session *s, int token,
                                                char *err, size_t errlen,
                                                int read_logits);
static int qw3_metal_session_eval_token_defer_logits(qw3_session *s, int token,
                                                     char *err, size_t errlen);
static int qw3_metal_session_eval_prefill_batch_mode(qw3_session *s,
                                                     const int *tokens,
                                                     int n_tokens,
                                                     char *err,
                                                     size_t errlen,
                                                     int logits_mode);
static int qw3_metal_env_n_gpu_layers(void);
static int qw3_session_uses_partial_metal(const qw3_session *s);
#endif
static void trace_emit(FILE *fp, bool json, bool *first_event,
                       const char *name, int il, const float *x, int n);

/* =========================================================================
 * Engine — the loaded model.
 * =========================================================================
 */

struct qw3_engine {
    qw3_model model;
    qw3_vocab vocab;
    qw3_weights weights;
    qw3_backend backend;
    bool metal_ready;
};

/* =========================================================================
 * Public API — think mode helpers.
 * =========================================================================
 */

bool qw3_think_mode_enabled(qw3_think_mode mode) {
    return mode != QW3_THINK_NONE;
}

qw3_think_mode qw3_think_mode_for_context(qw3_think_mode mode, int ctx_size) {
    if (mode == QW3_THINK_HIGH && ctx_size > 8192)
        return QW3_THINK_ON;
    if (mode == QW3_THINK_HIGH && ctx_size > 16384)
        return QW3_THINK_NONE;
    return mode;
}

const char *qw3_think_mode_name(qw3_think_mode mode) {
    switch (mode) {
    case QW3_THINK_NONE: return "none";
    case QW3_THINK_ON:   return "think";
    default:             return "unknown";
    }
}

const char *qw3_backend_name(qw3_backend backend) {
    switch (backend) {
    case QW3_BACKEND_METAL: return "Metal";
    case QW3_BACKEND_CPU:   return "CPU";
    default:                return "unknown";
    }
}

bool qw3_backend_supported(qw3_backend backend) {
    switch (backend) {
    case QW3_BACKEND_CPU:
        return true;
    case QW3_BACKEND_METAL:
#ifdef QW3_NO_METAL
        return false;
#else
        return true;
#endif
    default:
        return false;
    }
}

static int qw3_prefill_defer_interval(void) {
    const char *env = getenv("QW3_METAL_PREFILL_DEFER_INTERVAL");
    if (!env || !env[0]) return 16;
    char *end = NULL;
    long v = strtol(env, &end, 10);
    if (end == env || v < 1) return 1;
    if (v > 1024) return 1024;
    return (int)v;
}

static int qw3_metal_prefill_batch_size(void) {
    const char *env = getenv("QW3_METAL_PREFILL_BATCH");
    if (!env || !env[0]) return 4096;
    char *end = NULL;
    long v = strtol(env, &end, 10);
    if (end == env || v < 1) return 1;
    if (v > 4096) return 4096;
    return (int)v;
}

static void qw3_count_layer_types_before(int n_layers,
                                         int *n_full,
                                         int *n_linear) {
    if (n_layers < 0) n_layers = 0;
    if (n_layers > QW3_N_LAYER) n_layers = QW3_N_LAYER;
    int full = 0;
    int linear = 0;
    for (int il = 0; il < n_layers; il++) {
        if (qw3_layer_is_full_attention((uint32_t)il)) full++;
        else linear++;
    }
    if (n_full) *n_full = full;
    if (n_linear) *n_linear = linear;
}

#ifndef QW3_NO_METAL
static int qw3_metal_env_n_gpu_layers(void) {
    const char *env = getenv("QW3_METAL_NGL");
    if (!env || !env[0]) return QW3_N_LAYER;
    char *end = NULL;
    long v = strtol(env, &end, 10);
    if (end == env) return QW3_N_LAYER;
    if (v < 0) return QW3_N_LAYER;
    if (v > QW3_N_LAYER) return QW3_N_LAYER;
    return (int)v;
}

static int qw3_session_uses_partial_metal(const qw3_session *s) {
    return s && s->engine && s->engine->backend == QW3_BACKEND_METAL &&
           s->metal && s->metal_n_gpu_layers >= 0 &&
           s->metal_n_gpu_layers < QW3_N_LAYER;
}
#endif

/* =========================================================================
 * Memory estimation.
 * =========================================================================
 */

qw3_context_memory qw3_context_memory_estimate(qw3_backend backend,
                                                int ctx_size) {
    qw3_context_memory mem = {0};

    /* GQA KV cache: 10 layers × 2 kv_heads × 256 dim × ctx × 2 (K+V) × 4 */
    mem.gqa_kv_bytes = (uint64_t)QW3_N_FULL_ATTN_LAYERS * QW3_N_HEAD_KV *
                       QW3_N_HEAD_DIM * (uint64_t)ctx_size * 2 * sizeof(float);

    /* DeltaNet state: 30 layers × 32 heads × 128 × 128 × 4 bytes */
    mem.deltanet_state_bytes = (uint64_t)QW3_N_LINEAR_LAYERS *
                               QW3_N_LINEAR_V_HEADS *
                               QW3_N_LINEAR_HEAD_DIM *
                               QW3_N_LINEAR_HEAD_DIM * sizeof(float);

    /* Scratch buffers estimate. */
    mem.scratch_bytes = (uint64_t)QW3_N_EMBD * 4 * sizeof(float) +
                        (uint64_t)QW3_N_VOCAB * sizeof(float) +
                        (uint64_t)QW3_N_EXPERT * sizeof(float) +
                        (uint64_t)QW3_N_EXPERT_USED * QW3_N_FF_EXP *
                        sizeof(float);

    mem.total_bytes = mem.gqa_kv_bytes + mem.deltanet_state_bytes +
                      mem.scratch_bytes;
#ifndef QW3_NO_METAL
    if (backend == QW3_BACKEND_METAL) {
        int metal_full_layers = QW3_N_FULL_ATTN_LAYERS;
        int metal_linear_layers = QW3_N_LINEAR_LAYERS;
        qw3_count_layer_types_before(qw3_metal_env_n_gpu_layers(),
                                     &metal_full_layers,
                                     &metal_linear_layers);
        mem.deltanet_state_bytes = (uint64_t)metal_linear_layers *
                                   QW3_N_LINEAR_V_HEADS *
                                   QW3_N_LINEAR_HEAD_DIM *
                                   QW3_N_LINEAR_HEAD_DIM * sizeof(float);
        const char *kv_q8_env = getenv("QW3_METAL_KV_Q8_0");
        if (kv_q8_env && strcmp(kv_q8_env, "0") != 0) {
            mem.gqa_kv_bytes = (uint64_t)metal_full_layers *
                               QW3_N_HEAD_KV * (QW3_N_HEAD_DIM / 32) *
                               34ull * (uint64_t)ctx_size * 2ull;
            if (getenv("QW3_METAL_LEGACY_Q8_ATTN") == NULL) {
                const uint64_t q8_splits =
                    getenv("QW3_METAL_Q8_SPLIT_32") != NULL ? 32ull :
                    (getenv("QW3_METAL_Q8_SPLIT_64") != NULL ? 64ull :
                     (getenv("QW3_METAL_Q8_SPLIT_128") != NULL ? 128ull : 256ull));
                mem.scratch_bytes += q8_splits * QW3_N_HEAD *
                                     (QW3_N_HEAD_DIM + 2ull) * sizeof(float);
            }
        } else {
            const char *kv_f16_env = getenv("QW3_METAL_KV_F16");
            if (!kv_f16_env || strcmp(kv_f16_env, "0") != 0) {
                mem.gqa_kv_bytes = (uint64_t)metal_full_layers *
                                   QW3_N_HEAD_KV * QW3_N_HEAD_DIM *
                                   (uint64_t)ctx_size * 2ull * sizeof(uint16_t);
            } else {
                mem.gqa_kv_bytes = (uint64_t)metal_full_layers *
                                   QW3_N_HEAD_KV * QW3_N_HEAD_DIM *
                                   (uint64_t)ctx_size * 2ull * sizeof(float);
            }
        }
        const uint64_t conv_bytes = (uint64_t)metal_linear_layers *
                                    tensor_linear_qkv() *
                                    (QW3_N_LINEAR_CONV_K - 1) *
                                    sizeof(float);
        mem.scratch_bytes += conv_bytes +
                             (uint64_t)QW3_N_EMBD * 4 * sizeof(float);
        mem.total_bytes = mem.gqa_kv_bytes + mem.deltanet_state_bytes +
                          mem.scratch_bytes;
    }
#else
    (void)backend;
#endif
    return mem;
}

/* =========================================================================
 * GGUF Metadata.
 * =========================================================================
 */

enum {
    QW3_GGUF_TYPE_UINT8   = 0,
    QW3_GGUF_TYPE_INT8    = 1,
    QW3_GGUF_TYPE_UINT16  = 2,
    QW3_GGUF_TYPE_INT16   = 3,
    QW3_GGUF_TYPE_UINT32  = 4,
    QW3_GGUF_TYPE_INT32   = 5,
    QW3_GGUF_TYPE_FLOAT32 = 6,
    QW3_GGUF_TYPE_BOOL    = 7,
    QW3_GGUF_TYPE_STRING  = 8,
    QW3_GGUF_TYPE_ARRAY   = 9,
    QW3_GGUF_TYPE_UINT64  = 10,
    QW3_GGUF_TYPE_INT64   = 11,
    QW3_GGUF_TYPE_FLOAT64 = 12,
};

static bool qw3_key_is(qw3_str key, const char *z) {
    return qw3_streq(key, z);
}

static uint64_t gguf_scalar_size(uint32_t type) {
    switch (type) {
    case QW3_GGUF_TYPE_UINT8:
    case QW3_GGUF_TYPE_INT8:
    case QW3_GGUF_TYPE_BOOL:    return 1;
    case QW3_GGUF_TYPE_UINT16:
    case QW3_GGUF_TYPE_INT16:   return 2;
    case QW3_GGUF_TYPE_UINT32:
    case QW3_GGUF_TYPE_INT32:
    case QW3_GGUF_TYPE_FLOAT32: return 4;
    case QW3_GGUF_TYPE_UINT64:
    case QW3_GGUF_TYPE_INT64:
    case QW3_GGUF_TYPE_FLOAT64: return 8;
    default:                    return 0;
    }
}

static int64_t gguf_read_i64(qw3_cursor *c, uint32_t type) {
    switch (type) {
    case QW3_GGUF_TYPE_UINT8:  { uint8_t  v; if (!cursor_read(c, &v, 1)) qw3_die(c->error); return v; }
    case QW3_GGUF_TYPE_INT8:   { int8_t   v; if (!cursor_read(c, &v, 1)) qw3_die(c->error); return v; }
    case QW3_GGUF_TYPE_UINT16: { uint16_t v; if (!cursor_read(c, &v, 2)) qw3_die(c->error); return v; }
    case QW3_GGUF_TYPE_INT16:  { int16_t  v; if (!cursor_read(c, &v, 2)) qw3_die(c->error); return v; }
    case QW3_GGUF_TYPE_UINT32: { uint32_t v; if (!cursor_read(c, &v, 4)) qw3_die(c->error); return v; }
    case QW3_GGUF_TYPE_INT32:  { int32_t  v; if (!cursor_read(c, &v, 4)) qw3_die(c->error); return v; }
    case QW3_GGUF_TYPE_UINT64: { uint64_t v; if (!cursor_read(c, &v, 8)) qw3_die(c->error); return (int64_t)v; }
    case QW3_GGUF_TYPE_INT64:  { int64_t  v; if (!cursor_read(c, &v, 8)) qw3_die(c->error); return v; }
    default:
        qw3_die("metadata value is not an integer");
        return 0;
    }
}

static float gguf_read_f32(qw3_cursor *c, uint32_t type) {
    switch (type) {
    case QW3_GGUF_TYPE_FLOAT32: {
        float v;
        if (!cursor_read(c, &v, sizeof(v))) qw3_die(c->error);
        return v;
    }
    case QW3_GGUF_TYPE_FLOAT64: {
        double v;
        if (!cursor_read(c, &v, sizeof(v))) qw3_die(c->error);
        return (float)v;
    }
    default:
        qw3_die("metadata value is not a float");
        return 0.0f;
    }
}

static bool gguf_read_bool(qw3_cursor *c, uint32_t type) {
    if (type != QW3_GGUF_TYPE_BOOL) qw3_die("metadata value is not a bool");
    uint8_t v;
    if (!cursor_read(c, &v, sizeof(v))) qw3_die(c->error);
    return v != 0;
}

static void gguf_skip_value(qw3_cursor *c, uint32_t type) {
    if (type == QW3_GGUF_TYPE_STRING) {
        qw3_str ignored;
        if (!cursor_string(c, &ignored)) qw3_die(c->error);
        return;
    }
    if (type == QW3_GGUF_TYPE_ARRAY) {
        uint32_t arr_type;
        uint64_t arr_len;
        if (!cursor_u32(c, &arr_type)) qw3_die(c->error);
        if (!cursor_u64(c, &arr_len)) qw3_die(c->error);
        if (arr_type == QW3_GGUF_TYPE_STRING) {
            for (uint64_t i = 0; i < arr_len; i++) {
                qw3_str ignored;
                if (!cursor_string(c, &ignored)) qw3_die(c->error);
            }
            return;
        }
        uint64_t item_size = gguf_scalar_size(arr_type);
        if (item_size == 0 || arr_len > UINT64_MAX / item_size) {
            qw3_die("unsupported GGUF metadata array type");
        }
        if (!cursor_skip(c, arr_len * item_size)) qw3_die(c->error);
        return;
    }
    uint64_t size = gguf_scalar_size(type);
    if (size == 0) qw3_die("unsupported GGUF metadata value type");
    if (!cursor_skip(c, size)) qw3_die(c->error);
}

static qw3_str *gguf_read_string_array(qw3_cursor *c, uint64_t *len_out) {
    uint32_t arr_type;
    uint64_t arr_len;
    if (!cursor_u32(c, &arr_type)) qw3_die(c->error);
    if (!cursor_u64(c, &arr_len)) qw3_die(c->error);
    if (arr_type != QW3_GGUF_TYPE_STRING) {
        qw3_die("expected GGUF string array");
    }
    if (arr_len > UINT32_MAX) qw3_die("GGUF string array is too large");
    qw3_str *items = qw3_xcalloc((size_t)arr_len, sizeof(items[0]));
    for (uint64_t i = 0; i < arr_len; i++) {
        if (!cursor_string(c, &items[i])) qw3_die(c->error);
    }
    *len_out = arr_len;
    return items;
}

static uint32_t *gguf_read_u32_array(qw3_cursor *c, uint64_t *len_out) {
    uint32_t arr_type;
    uint64_t arr_len;
    if (!cursor_u32(c, &arr_type)) qw3_die(c->error);
    if (!cursor_u64(c, &arr_len)) qw3_die(c->error);
    if (arr_len > UINT32_MAX) qw3_die("GGUF integer array is too large");
    uint32_t *items = qw3_xcalloc((size_t)arr_len, sizeof(items[0]));
    for (uint64_t i = 0; i < arr_len; i++) {
        items[i] = (uint32_t)gguf_read_i64(c, arr_type);
    }
    *len_out = arr_len;
    return items;
}

static void metadata_parse_one(qw3_model *m, qw3_cursor *c,
                               qw3_str key, uint32_t type) {
#define META_I64(name, field) \
    do { if (qw3_key_is(key, name)) { m->meta.field = gguf_read_i64(c, type); return; } } while (0)
#define META_F32(name, field) \
    do { if (qw3_key_is(key, name)) { m->meta.field = gguf_read_f32(c, type); return; } } while (0)
#define META_STR(name, field) \
    do { if (qw3_key_is(key, name)) { if (type != QW3_GGUF_TYPE_STRING) qw3_die("metadata string has wrong type"); if (!cursor_string(c, &m->meta.field)) qw3_die(c->error); return; } } while (0)

    META_STR("general.architecture", architecture);
    META_STR("tokenizer.ggml.model", tokenizer_model);
    META_STR("tokenizer.ggml.pre", tokenizer_pre);

    META_I64("qwen35moe.block_count", block_count);
    META_I64("qwen35moe.context_length", context_length);
    META_I64("qwen35moe.embedding_length", embedding_length);
    META_I64("qwen35moe.attention.head_count", head_count);
    META_I64("qwen35moe.attention.head_count_kv", head_count_kv);
    META_I64("qwen35moe.attention.key_length", key_length);
    META_I64("qwen35moe.attention.value_length", value_length);
    META_I64("qwen35moe.expert_count", expert_count);
    META_I64("qwen35moe.expert_used_count", expert_used_count);
    META_I64("qwen35moe.expert_feed_forward_length", expert_ffn_length);
    META_I64("qwen35moe.expert_shared_feed_forward_length", expert_shared_ffn_length);
    META_I64("qwen35moe.ssm.conv_kernel", ssm_conv_kernel);
    META_I64("qwen35moe.ssm.state_size", ssm_state_size);
    META_I64("qwen35moe.ssm.group_count", ssm_group_count);
    META_I64("qwen35moe.ssm.time_step_rank", ssm_time_step_rank);
    META_I64("qwen35moe.ssm.inner_size", ssm_inner_size);
    META_I64("qwen35moe.full_attention_interval", full_attention_interval);
    META_I64("qwen35moe.rope.dimension_count", rope_dimension_count);
    META_I64("tokenizer.ggml.eos_token_id", eos_token_id);
    META_I64("tokenizer.ggml.bos_token_id", bos_token_id);
    META_I64("tokenizer.ggml.padding_token_id", padding_token_id);

    META_F32("qwen35moe.rope.freq_base", rope_freq_base);
    META_F32("qwen35moe.attention.layer_norm_rms_epsilon", rms_eps);

    if (qw3_key_is(key, "tokenizer.ggml.add_bos_token")) {
        m->meta.add_bos_token = gguf_read_bool(c, type);
        return;
    }

    if (qw3_key_is(key, "qwen35moe.rope.dimension_sections")) {
        if (type != QW3_GGUF_TYPE_ARRAY) qw3_die("rope sections metadata is not an array");
        uint32_t arr_type;
        uint64_t arr_len;
        if (!cursor_u32(c, &arr_type)) qw3_die(c->error);
        if (!cursor_u64(c, &arr_len)) qw3_die(c->error);
        if (arr_len > 4) qw3_die("too many rope dimension sections");
        for (uint64_t i = 0; i < arr_len; i++) {
            m->meta.rope_sections[i] = (int)gguf_read_i64(c, arr_type);
        }
        m->meta.rope_sections_len = (int)arr_len;
        return;
    }

    if (qw3_key_is(key, "tokenizer.ggml.tokens")) {
        if (type != QW3_GGUF_TYPE_ARRAY) qw3_die("tokenizer metadata is not an array");
        uint64_t arr_len;
        m->meta.token_texts = gguf_read_string_array(c, &arr_len);
        m->meta.tokenizer_token_count = (int64_t)arr_len;
        return;
    }

    if (qw3_key_is(key, "tokenizer.ggml.token_type")) {
        if (type != QW3_GGUF_TYPE_ARRAY) qw3_die("tokenizer metadata is not an array");
        uint64_t arr_len;
        m->meta.token_types = gguf_read_u32_array(c, &arr_len);
        m->meta.tokenizer_token_type_count = (int64_t)arr_len;
        return;
    }

    if (qw3_key_is(key, "tokenizer.ggml.merges")) {
        if (type != QW3_GGUF_TYPE_ARRAY) qw3_die("tokenizer metadata is not an array");
        uint64_t arr_len;
        m->meta.merge_texts = gguf_read_string_array(c, &arr_len);
        m->meta.tokenizer_merge_count = (int64_t)arr_len;
        return;
    }

    gguf_skip_value(c, type);

#undef META_I64
#undef META_F32
#undef META_STR
}

static void metadata_validate(const qw3_model *m) {
#define CHECK_I64(field, expected) \
    do { \
        if (m->meta.field != (expected)) { \
            fprintf(stderr, "qw3: GGUF metadata %s is %" PRId64 " (expected %d)\n", \
                    #field, m->meta.field, (int)(expected)); \
            exit(1); \
        } \
    } while (0)

    if (!qw3_streq(m->meta.architecture, "qwen35moe")) {
        fprintf(stderr, "qw3: GGUF architecture is '%.*s' (expected qwen35moe)\n",
                (int)m->meta.architecture.len, m->meta.architecture.ptr);
        exit(1);
    }
    CHECK_I64(block_count, QW3_N_LAYER);
    CHECK_I64(embedding_length, QW3_N_EMBD);
    CHECK_I64(head_count, QW3_N_HEAD);
    CHECK_I64(head_count_kv, QW3_N_HEAD_KV);
    CHECK_I64(key_length, QW3_N_HEAD_DIM);
    CHECK_I64(value_length, QW3_N_HEAD_DIM);
    CHECK_I64(expert_count, QW3_N_EXPERT);
    CHECK_I64(expert_used_count, QW3_N_EXPERT_USED);
    CHECK_I64(expert_ffn_length, QW3_N_FF_EXP);
    CHECK_I64(expert_shared_ffn_length, QW3_N_FF_SHARED);
    CHECK_I64(ssm_conv_kernel, QW3_N_LINEAR_CONV_K);
    CHECK_I64(ssm_state_size, QW3_N_LINEAR_HEAD_DIM);
    CHECK_I64(ssm_group_count, QW3_N_LINEAR_QK_HEADS);
    CHECK_I64(ssm_time_step_rank, QW3_N_LINEAR_V_HEADS);
    CHECK_I64(ssm_inner_size, QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM);
    CHECK_I64(full_attention_interval, QW3_FULL_ATTN_INTERVAL);
    CHECK_I64(rope_dimension_count, QW3_ROPE_DIM);
    CHECK_I64(tokenizer_token_count, QW3_N_VOCAB);
    CHECK_I64(tokenizer_token_type_count, QW3_N_VOCAB);

    if (m->n_tensors != QW3_EXPECTED_TENSORS) {
        fprintf(stderr, "qw3: GGUF tensor count is %u (expected %d)\n",
                m->n_tensors, QW3_EXPECTED_TENSORS);
        exit(1);
    }

    if (fabsf(m->meta.rope_freq_base - QW3_ROPE_THETA) > 0.5f) {
        fprintf(stderr, "qw3: rope freq base is %.1f (expected %.1f)\n",
                m->meta.rope_freq_base, QW3_ROPE_THETA);
        exit(1);
    }
    if (fabsf(m->meta.rms_eps - QW3_RMS_EPS) > 1.0e-12f) {
        fprintf(stderr, "qw3: RMS eps is %.9g (expected %.9g)\n",
                m->meta.rms_eps, QW3_RMS_EPS);
        exit(1);
    }

#undef CHECK_I64
}

/* Read the tensor directory and convert relative GGUF offsets to absolute
 * mmap offsets.  Tensor bytes are still never copied here. */
static void parse_tensors(qw3_model *m, qw3_cursor *c) {
    m->tensors = qw3_xcalloc((size_t)m->n_tensors, sizeof(m->tensors[0]));

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        qw3_tensor *t = &m->tensors[i];
        qw3_str s;

        if (!cursor_string(c, &s)) qw3_die(c->error);
        t->name = s.ptr;
        t->name_len = s.len;
        
        if (!cursor_u32(c, &t->ndim)) qw3_die(c->error);
        if (t->ndim == 0 || t->ndim > QW3_MAX_DIMS) {
            qw3_die("tensor has an unsupported number of dimensions");
        }

        t->elements = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!cursor_u64(c, &t->dim[d])) qw3_die(c->error);
            if (t->dim[d] != 0 && t->elements > UINT64_MAX / t->dim[d]) {
                qw3_die("tensor element count overflow");
            }
            t->elements *= t->dim[d];
        }

        if (!cursor_u32(c, &t->type)) qw3_die(c->error);
        if (!cursor_u64(c, &t->rel_offset)) qw3_die(c->error);

        /* skip tensor nbytes warning since we don't need it */
    }

    m->tensor_data_offset = align_up(c->pos, 32);

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        qw3_tensor *t = &m->tensors[i];
        if (t->rel_offset > UINT64_MAX - m->tensor_data_offset) {
            qw3_die("tensor offset overflow");
        }
        t->offset = m->tensor_data_offset + t->rel_offset;
        if (t->offset > m->map_size)
        {
            qw3_die("tensor points outside GGUF file");
        }
    }
}

static void model_open(qw3_model *m, const char *path, bool metal_mapping,
                       bool prefetch_cpu) {
    (void)prefetch_cpu;
    memset(m, 0, sizeof(*m));
    m->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd == -1) qw3_die_errno("cannot open model", path);

    struct stat st;
    if (fstat(fd, &st) == -1) qw3_die_errno("cannot stat model", path);
    if (st.st_size < 32) qw3_die("model file is too small to be GGUF");

    const int mmap_flags = metal_mapping ? MAP_SHARED : MAP_PRIVATE;
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, mmap_flags, fd, 0);
    if (map == MAP_FAILED) qw3_die_errno("cannot mmap model", path);

    m->fd = fd;
    m->map = map;
    m->map_size = (uint64_t)st.st_size;

    qw3_cursor c = {
        .base = m->map,
        .size = m->map_size,
        .pos = 0,
        .error = {0},
    };
    uint32_t magic, version;
    uint64_t n_tensors, n_kv;
    if (!cursor_u32(&c, &magic)) qw3_die(c.error);
    if (magic != QW3_GGUF_MAGIC) qw3_die("model is not a GGUF file");
    if (!cursor_u32(&c, &version)) qw3_die(c.error);
    if (!cursor_u64(&c, &n_tensors)) qw3_die(c.error);
    if (!cursor_u64(&c, &n_kv)) qw3_die(c.error);

    if (version != 3) qw3_die("only GGUF v3 is supported");

    m->n_tensors = n_tensors;
    m->n_kv = n_kv;
    
    /* Parse only metadata needed to prove this is exactly the model we run.
     * Large tokenizer arrays are skipped in-place after recording their length. */
    for (uint64_t i = 0; i < n_kv; i++) {
        qw3_str key;
        uint32_t kv_type;
        if (!cursor_string(&c, &key)) qw3_die(c.error);
        if (!cursor_u32(&c, &kv_type)) qw3_die(c.error);
        metadata_parse_one(m, &c, key, kv_type);
    }
    metadata_validate(m);

    parse_tensors(m, &c);
}

static qw3_tensor *model_find_tensor(const qw3_model *m, const char *name) {
    const size_t len = strlen(name);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (m->tensors[i].name_len == len &&
            memcmp(m->tensors[i].name, name, len) == 0) {
            return &m->tensors[i];
        }
    }
    return NULL;
}

/* =========================================================================
 * Weight Binding & Validation.
 * =========================================================================
 */

static void require_tensor(const qw3_model *m, const char *name,
                           uint32_t type, int ndim,
                           uint64_t d0, uint64_t d1, uint64_t d2,
                           qw3_tensor **out) {
    qw3_tensor *t = model_find_tensor(m, name);
    if (!t) qw3_die_errno("missing required tensor", name);

    if (t->type != type && type != QW3_TENSOR_ANY) {
        fprintf(stderr, "qw3: tensor '%s' has unexpected type %d (expected %d)\n",
                name, t->type, type);
        exit(1);
    }
    if ((int)t->ndim != ndim) {
        fprintf(stderr, "qw3: tensor '%s' has %d dims (expected %d)\n",
                name, t->ndim, ndim);
        exit(1);
    }
    if (ndim >= 1 && d0 != 0 && t->dim[0] != d0) {
        fprintf(stderr, "qw3: tensor '%s' dim 0 is %" PRIu64 " (expected %" PRIu64 ")\n",
                name, t->dim[0], d0);
        exit(1);
    }
    if (ndim >= 2 && d1 != 0 && t->dim[1] != d1) {
        fprintf(stderr, "qw3: tensor '%s' dim 1 is %" PRIu64 " (expected %" PRIu64 ")\n",
                name, t->dim[1], d1);
        exit(1);
    }
    if (ndim >= 3 && d2 != 0 && t->dim[2] != d2) {
        fprintf(stderr, "qw3: tensor '%s' dim 2 is %" PRIu64 " (expected %" PRIu64 ")\n",
                name, t->dim[2], d2);
        exit(1);
    }
    *out = t;
}

static void require_optional_tensor(const qw3_model *m, const char *name,
                                    uint32_t type, int ndim,
                                    uint64_t d0, uint64_t d1, uint64_t d2,
                                    qw3_tensor **out) {
    qw3_tensor *t = model_find_tensor(m, name);
    if (!t) {
        *out = NULL;
        return;
    }
    require_tensor(m, name, type, ndim, d0, d1, d2, out);
}

static void weights_bind(qw3_engine *e) {
    const qw3_model *m = &e->model;
    qw3_weights *w = &e->weights;

    /* Base tokens / output. */
    require_tensor(m, "token_embd.weight", QW3_TENSOR_ANY, 2, QW3_N_EMBD, QW3_N_VOCAB, 0, &w->token_embd);
    require_tensor(m, "output_norm.weight", QW3_TENSOR_F32, 1, QW3_N_EMBD, 0, 0, &w->output_norm);
    require_tensor(m, "output.weight", QW3_TENSOR_ANY, 2, QW3_N_EMBD, QW3_N_VOCAB, 0, &w->output);

    /* Layers. */
    char name[128];
    for (int i = 0; i < QW3_N_LAYER; i++) {
        qw3_layer_weights *lw = &w->layer[i];
        bool is_full = qw3_layer_is_full_attention(i);

        /* Norms (common) */
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", i);
        require_tensor(m, name, QW3_TENSOR_F32, 1, QW3_N_EMBD, 0, 0, &lw->attn_norm);

        snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", i);
        require_tensor(m, name, QW3_TENSOR_F32, 1, QW3_N_EMBD, 0, 0, &lw->ffn_norm);

        if (is_full) {
            /* GQA Attention */
            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", i);
            require_tensor(m, name, QW3_TENSOR_ANY, 2,
                           QW3_N_EMBD, tensor_cols_qg(), 0, &lw->attn_q_proj);

            snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", i);
            require_tensor(m, name, QW3_TENSOR_F32, 1,
                           QW3_N_HEAD_DIM, 0, 0, &lw->attn_q_norm);

            snprintf(name, sizeof(name), "blk.%d.attn_k.weight", i);
            require_tensor(m, name, QW3_TENSOR_ANY, 2,
                           QW3_N_EMBD, tensor_cols_kv(), 0, &lw->attn_k_proj);

            snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", i);
            require_tensor(m, name, QW3_TENSOR_F32, 1,
                           QW3_N_HEAD_DIM, 0, 0, &lw->attn_k_norm);

            snprintf(name, sizeof(name), "blk.%d.attn_v.weight", i);
            require_tensor(m, name, QW3_TENSOR_ANY, 2,
                           QW3_N_EMBD, tensor_cols_kv(), 0, &lw->attn_v_proj);

            snprintf(name, sizeof(name), "blk.%d.attn_output.weight", i);
            require_tensor(m, name, QW3_TENSOR_ANY, 2,
                           tensor_linear_inner(), QW3_N_EMBD, 0, &lw->attn_o_proj);
        } else {
            /* Linear Attention (Gated DeltaNet) */
            snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", i);
            require_tensor(m, name, QW3_TENSOR_ANY, 2,
                           QW3_N_EMBD, tensor_linear_qkv(), 0, &lw->linear_qkv_proj);

            snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", i);
            require_tensor(m, name, QW3_TENSOR_ANY, 2,
                           QW3_N_EMBD, tensor_linear_inner(), 0, &lw->linear_gate_proj);

            snprintf(name, sizeof(name), "blk.%d.ssm_a", i);
            require_tensor(m, name, QW3_TENSOR_F32, 1,
                           QW3_N_LINEAR_V_HEADS, 0, 0, &lw->linear_ssm_a);

            snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", i);
            require_tensor(m, name, QW3_TENSOR_F32, 1,
                           QW3_N_LINEAR_V_HEADS, 0, 0, &lw->linear_ssm_dt_bias);

            snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", i);
            require_tensor(m, name, QW3_TENSOR_ANY, 2,
                           tensor_linear_inner(), QW3_N_EMBD, 0, &lw->linear_ssm_out);

            snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", i);
            require_tensor(m, name, QW3_TENSOR_F32, 1,
                           QW3_N_LINEAR_HEAD_DIM, 0, 0, &lw->linear_ssm_norm);

            snprintf(name, sizeof(name), "blk.%d.ssm_alpha.weight", i);
            require_tensor(m, name, QW3_TENSOR_ANY, 2,
                           QW3_N_EMBD, QW3_N_LINEAR_V_HEADS, 0, &lw->linear_ssm_alpha);

            snprintf(name, sizeof(name), "blk.%d.ssm_beta.weight", i);
            require_tensor(m, name, QW3_TENSOR_ANY, 2,
                           QW3_N_EMBD, QW3_N_LINEAR_V_HEADS, 0, &lw->linear_ssm_beta);

            snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", i);
            require_tensor(m, name, QW3_TENSOR_ANY, 2,
                           QW3_N_LINEAR_CONV_K, tensor_linear_qkv(), 0,
                           &lw->linear_conv_weight);
        }

        /* MoE FFN */
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", i);
        require_tensor(m, name, QW3_TENSOR_F32, 2,
                       QW3_N_EMBD, QW3_N_EXPERT, 0, &lw->ffn_gate_inp);

        snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp_shexp.weight", i);
        require_optional_tensor(m, name, QW3_TENSOR_F32, 1,
                                QW3_N_EMBD, 0, 0, &lw->ffn_gate_inp_shexp);

        snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", i);
        require_tensor(m, name, QW3_TENSOR_ANY, 3,
                       QW3_N_EMBD, QW3_N_FF_EXP, QW3_N_EXPERT, &lw->ffn_gate_exps);

        snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", i);
        require_tensor(m, name, QW3_TENSOR_ANY, 3,
                       QW3_N_EMBD, QW3_N_FF_EXP, QW3_N_EXPERT, &lw->ffn_up_exps);

        snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", i);
        require_tensor(m, name, QW3_TENSOR_ANY, 3,
                       QW3_N_FF_EXP, QW3_N_EMBD, QW3_N_EXPERT, &lw->ffn_down_exps);

        /* Shared Expert */
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", i);
        require_tensor(m, name, QW3_TENSOR_ANY, 2,
                       QW3_N_EMBD, QW3_N_FF_SHARED, 0, &lw->ffn_gate_shared);

        snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", i);
        require_tensor(m, name, QW3_TENSOR_ANY, 2,
                       QW3_N_EMBD, QW3_N_FF_SHARED, 0, &lw->ffn_up_shared);

        snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", i);
        require_tensor(m, name, QW3_TENSOR_ANY, 2,
                       QW3_N_FF_SHARED, QW3_N_EMBD, 0, &lw->ffn_down_shared);
    }
}

/* =========================================================================
 * CPU reference kernels — deliberately small and model-specific.
 * =========================================================================
 */

static const char *tensor_type_name(uint32_t type) {
    switch (type) {
    case QW3_TENSOR_F32:     return "f32";
    case QW3_TENSOR_F16:     return "f16";
    case QW3_TENSOR_Q8_0:    return "q8_0";
    case QW3_TENSOR_Q6_K:    return "q6_K";
    case QW3_TENSOR_IQ3_S:   return "iq3_s";
    case QW3_TENSOR_IQ4_XS:  return "iq4_xs";
    case QW3_TENSOR_BF16:    return "bf16";
    default:                 return "unsupported";
    }
}

static const uint8_t *tensor_data(const qw3_model *m, const qw3_tensor *t) {
    return m->map + t->offset;
}

static bool tensor_is_dense_float(uint32_t type) {
    return type == QW3_TENSOR_F32 ||
           type == QW3_TENSOR_F16 ||
           type == QW3_TENSOR_BF16;
}

static bool tensor_read_dense_row(const qw3_model *m, const qw3_tensor *t,
                                  uint64_t row, float *dst) {
    if (t->ndim < 2 || row >= t->dim[1] || !tensor_is_dense_float(t->type)) {
        if (t->type != QW3_TENSOR_Q8_0) return false;
    }

    const uint64_t n = t->dim[0];
    const uint8_t *p = tensor_data(m, t);
    if (t->type == QW3_TENSOR_F32) {
        const float *src = (const float *)p + row * n;
        memcpy(dst, src, (size_t)n * sizeof(float));
        return true;
    }
    if (t->type == QW3_TENSOR_F16) {
        const uint8_t *src = p + row * n * sizeof(uint16_t);
        for (uint64_t i = 0; i < n; i++) {
            dst[i] = qw3_f16_to_f32(qw3_load_u16(src + i * sizeof(uint16_t)));
        }
        return true;
    }
    if (t->type == QW3_TENSOR_BF16) {
        const uint8_t *src = p + row * n * sizeof(uint16_t);
        for (uint64_t i = 0; i < n; i++) {
            dst[i] = qw3_bf16_to_f32(qw3_load_u16(src + i * sizeof(uint16_t)));
        }
        return true;
    }
    if (t->type == QW3_TENSOR_Q8_0) {
        if ((n % 32) != 0) return false;
        const uint64_t blocks_per_row = n / 32;
        const uint64_t block_size = sizeof(uint16_t) + 32;
        const uint8_t *src = p + row * blocks_per_row * block_size;
        for (uint64_t b = 0; b < blocks_per_row; b++) {
            const uint8_t *blk = src + b * block_size;
            float d = qw3_f16_to_f32(qw3_load_u16(blk));
            const int8_t *qs = (const int8_t *)(blk + sizeof(uint16_t));
            for (uint64_t i = 0; i < 32; i++) {
                dst[b * 32 + i] = d * (float)qs[i];
            }
        }
        return true;
    }
    return false;
}

static float tensor_read_dense_1d(const qw3_model *m, const qw3_tensor *t,
                                  uint64_t i) {
    const uint8_t *p = tensor_data(m, t);
    if (t->type == QW3_TENSOR_F32) {
        return ((const float *)p)[i];
    }
    if (t->type == QW3_TENSOR_F16) {
        return qw3_f16_to_f32(qw3_load_u16(p + i * sizeof(uint16_t)));
    }
    if (t->type == QW3_TENSOR_BF16) {
        return qw3_bf16_to_f32(qw3_load_u16(p + i * sizeof(uint16_t)));
    }
    return 0.0f;
}

static float tensor_read_dense_linear(const qw3_model *m, const qw3_tensor *t,
                                      uint64_t i) {
    const uint8_t *p = tensor_data(m, t);
    if (t->type == QW3_TENSOR_F32) {
        return ((const float *)p)[i];
    }
    if (t->type == QW3_TENSOR_F16) {
        return qw3_f16_to_f32(qw3_load_u16(p + i * sizeof(uint16_t)));
    }
    if (t->type == QW3_TENSOR_BF16) {
        return qw3_bf16_to_f32(qw3_load_u16(p + i * sizeof(uint16_t)));
    }
    return 0.0f;
}

static bool cpu_matvec_dense(const qw3_model *m, const qw3_tensor *w,
                             const float *x, float *y) {
    if (w->ndim != 2 || !tensor_is_dense_float(w->type)) return false;
    const uint64_t n_in = w->dim[0];
    const uint64_t n_out = w->dim[1];
    for (uint64_t row = 0; row < n_out; row++) {
        const uint8_t *base = tensor_data(m, w);
        float sum = 0.0f;
        if (w->type == QW3_TENSOR_F32) {
            const float *wr = (const float *)base + row * n_in;
            for (uint64_t i = 0; i < n_in; i++) sum += wr[i] * x[i];
        } else if (w->type == QW3_TENSOR_F16) {
            const uint8_t *wr = base + row * n_in * sizeof(uint16_t);
            for (uint64_t i = 0; i < n_in; i++) {
                sum += qw3_f16_to_f32(qw3_load_u16(wr + i * sizeof(uint16_t))) * x[i];
            }
        } else if (w->type == QW3_TENSOR_BF16) {
            const uint8_t *wr = base + row * n_in * sizeof(uint16_t);
            for (uint64_t i = 0; i < n_in; i++) {
                sum += qw3_bf16_to_f32(qw3_load_u16(wr + i * sizeof(uint16_t))) * x[i];
            }
        }
        y[row] = sum;
    }
    return true;
}

static bool cpu_dot_dense_1d(const qw3_model *m, const qw3_tensor *w,
                             const float *x, float *y) {
    if (!w || w->ndim != 1 || !tensor_is_dense_float(w->type)) return false;
    const uint64_t n = w->dim[0];
    float sum = 0.0f;
    const uint8_t *base = tensor_data(m, w);
    if (w->type == QW3_TENSOR_F32) {
        const float *ww = (const float *)base;
        for (uint64_t i = 0; i < n; i++) sum += ww[i] * x[i];
    } else if (w->type == QW3_TENSOR_F16) {
        for (uint64_t i = 0; i < n; i++) {
            sum += qw3_f16_to_f32(qw3_load_u16(base + i * sizeof(uint16_t))) * x[i];
        }
    } else if (w->type == QW3_TENSOR_BF16) {
        for (uint64_t i = 0; i < n; i++) {
            sum += qw3_bf16_to_f32(qw3_load_u16(base + i * sizeof(uint16_t))) * x[i];
        }
    }
    *y = sum;
    return true;
}

static bool cpu_matvec_q8_0(const qw3_model *m, const qw3_tensor *w,
                            const float *x, float *y) {
    if (w->ndim != 2 || w->type != QW3_TENSOR_Q8_0) return false;
    const uint64_t n_in = w->dim[0];
    const uint64_t n_out = w->dim[1];
    if ((n_in % 32) != 0) return false;

    const uint64_t blocks_per_row = n_in / 32;
    const uint64_t block_size = sizeof(uint16_t) + 32;
    const uint8_t *base = tensor_data(m, w);
    for (uint64_t row = 0; row < n_out; row++) {
        const uint8_t *src = base + row * blocks_per_row * block_size;
        float sum = 0.0f;
        for (uint64_t b = 0; b < blocks_per_row; b++) {
            const uint8_t *blk = src + b * block_size;
            const float d = qw3_f16_to_f32(qw3_load_u16(blk));
            const int8_t *qs = (const int8_t *)(blk + sizeof(uint16_t));
            const float *xx = x + b * 32;
            for (uint64_t i = 0; i < 32; i++) {
                sum += d * (float)qs[i] * xx[i];
            }
        }
        y[row] = sum;
    }
    return true;
}

static const int8_t qw3_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

static const uint8_t qw3_kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128,
};

static const uint16_t qw3_iq3s_kgrid[512] = {
       0,    1,    2,    5,    7,    8,    9,   10,   12,   14,   16,   17,   21,   27,   32,   34,
      37,   39,   41,   43,   48,   50,   57,   60,   63,   64,   65,   66,   68,   72,   73,   77,
      80,   83,   87,   89,   93,  100,  113,  117,  122,  128,  129,  133,  135,  136,  139,  142,
     145,  149,  152,  156,  162,  165,  167,  169,  171,  184,  187,  195,  201,  205,  208,  210,
     217,  219,  222,  228,  232,  234,  247,  249,  253,  256,  267,  271,  273,  276,  282,  288,
     291,  297,  312,  322,  324,  336,  338,  342,  347,  353,  357,  359,  374,  379,  390,  393,
     395,  409,  426,  441,  448,  450,  452,  464,  466,  470,  475,  488,  492,  512,  513,  514,
     516,  520,  521,  523,  525,  527,  528,  530,  537,  540,  542,  556,  558,  561,  570,  576,
     577,  579,  582,  584,  588,  593,  600,  603,  609,  616,  618,  632,  638,  640,  650,  653,
     655,  656,  660,  666,  672,  675,  685,  688,  698,  705,  708,  711,  712,  715,  721,  727,
     728,  732,  737,  754,  760,  771,  773,  778,  780,  793,  795,  802,  806,  808,  812,  833,
     840,  843,  849,  856,  858,  873,  912,  916,  919,  932,  934,  961,  963,  968,  970,  977,
     989,  993, 1010, 1016, 1024, 1025, 1027, 1029, 1031, 1032, 1034, 1036, 1038, 1041, 1043, 1047,
    1048, 1050, 1057, 1059, 1061, 1064, 1066, 1079, 1080, 1083, 1085, 1088, 1090, 1096, 1099, 1103,
    1106, 1109, 1113, 1116, 1122, 1129, 1153, 1156, 1159, 1169, 1171, 1176, 1183, 1185, 1195, 1199,
    1209, 1212, 1216, 1218, 1221, 1225, 1234, 1236, 1241, 1243, 1250, 1256, 1270, 1281, 1287, 1296,
    1299, 1306, 1309, 1313, 1338, 1341, 1348, 1353, 1362, 1375, 1376, 1387, 1400, 1408, 1410, 1415,
    1425, 1453, 1457, 1477, 1481, 1494, 1496, 1507, 1512, 1538, 1545, 1547, 1549, 1551, 1554, 1561,
    1563, 1565, 1570, 1572, 1575, 1577, 1587, 1593, 1601, 1603, 1605, 1612, 1617, 1619, 1632, 1648,
    1658, 1662, 1664, 1674, 1680, 1690, 1692, 1704, 1729, 1736, 1740, 1745, 1747, 1751, 1752, 1761,
    1763, 1767, 1773, 1787, 1795, 1801, 1806, 1810, 1817, 1834, 1840, 1844, 1857, 1864, 1866, 1877,
    1882, 1892, 1902, 1915, 1934, 1953, 1985, 1987, 2000, 2002, 2013, 2048, 2052, 2058, 2064, 2068,
    2071, 2074, 2081, 2088, 2104, 2114, 2119, 2121, 2123, 2130, 2136, 2141, 2147, 2153, 2157, 2177,
    2179, 2184, 2189, 2193, 2203, 2208, 2223, 2226, 2232, 2244, 2249, 2251, 2256, 2258, 2265, 2269,
    2304, 2306, 2324, 2335, 2336, 2361, 2373, 2375, 2385, 2418, 2443, 2460, 2480, 2504, 2509, 2520,
    2531, 2537, 2562, 2568, 2572, 2578, 2592, 2596, 2599, 2602, 2614, 2620, 2625, 2627, 2629, 2634,
    2641, 2650, 2682, 2688, 2697, 2707, 2712, 2718, 2731, 2754, 2759, 2760, 2775, 2788, 2793, 2805,
    2811, 2817, 2820, 2832, 2842, 2854, 2890, 2902, 2921, 2923, 2978, 3010, 3012, 3026, 3081, 3083,
    3085, 3097, 3099, 3120, 3136, 3152, 3159, 3188, 3210, 3228, 3234, 3245, 3250, 3256, 3264, 3276,
    3281, 3296, 3349, 3363, 3378, 3392, 3395, 3420, 3440, 3461, 3488, 3529, 3531, 3584, 3588, 3591,
    3600, 3602, 3614, 3616, 3628, 3634, 3650, 3657, 3668, 3683, 3685, 3713, 3716, 3720, 3726, 3729,
    3736, 3753, 3778, 3802, 3805, 3819, 3841, 3845, 3851, 3856, 3880, 3922, 3938, 3970, 3993, 4032,
};

static void iq3s_grid4(uint16_t idx, uint8_t out[4]) {
    uint16_t packed = qw3_iq3s_kgrid[idx & 511u];
    for (int i = 0; i < 4; i++) {
        out[i] = (uint8_t)(2 * ((packed >> (3 * i)) & 7u) + 1u);
    }
}

static float cpu_dot_iq4_xs_row(const uint8_t *src, const float *x,
                                uint64_t n_in) {
    const uint64_t blocks_per_row = n_in / QW3_QK_K;
    const uint64_t block_size = sizeof(uint16_t) + sizeof(uint16_t) +
                                QW3_QK_K / 64 + QW3_QK_K / 2;
    float sum = 0.0f;
    for (uint64_t b = 0; b < blocks_per_row; b++) {
        const uint8_t *blk = src + b * block_size;
        const float d = qw3_f16_to_f32(qw3_load_u16(blk));
        const uint16_t scales_h = qw3_load_u16(blk + sizeof(uint16_t));
        const uint8_t *scales_l = blk + 2 * sizeof(uint16_t);
        const uint8_t *qs = scales_l + QW3_QK_K / 64;
        const float *xx = x + b * QW3_QK_K;

        for (int ib = 0; ib < QW3_QK_K / 32; ib++) {
            const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0f) |
                           (((scales_h >> (2 * ib)) & 0x03) << 4);
            const float dl = d * (float)(ls - 32);
            const uint8_t *q = qs + ib * 16;
            const float *xg = xx + ib * 32;
            for (int j = 0; j < 16; j++) {
                sum += dl * (float)qw3_iq4nl[q[j] & 0x0f] * xg[j];
                sum += dl * (float)qw3_iq4nl[q[j] >> 4] * xg[j + 16];
            }
        }
    }
    return sum;
}

static bool cpu_matvec_iq4_xs(const qw3_model *m, const qw3_tensor *w,
                              const float *x, float *y) {
    if (w->ndim != 2 || w->type != QW3_TENSOR_IQ4_XS) return false;
    const uint64_t n_in = w->dim[0];
    const uint64_t n_out = w->dim[1];
    if ((n_in % QW3_QK_K) != 0) return false;

    const uint64_t blocks_per_row = n_in / QW3_QK_K;
    const uint64_t block_size = sizeof(uint16_t) + sizeof(uint16_t) +
                                QW3_QK_K / 64 + QW3_QK_K / 2;
    const uint8_t *base = tensor_data(m, w);
    for (uint64_t row = 0; row < n_out; row++) {
        y[row] = cpu_dot_iq4_xs_row(base + row * blocks_per_row * block_size,
                                    x, n_in);
    }
    return true;
}

static bool cpu_matvec_iq4_xs_expert(const qw3_model *m, const qw3_tensor *w,
                                     int expert, const float *x, float *y) {
    if (w->ndim != 3 || w->type != QW3_TENSOR_IQ4_XS) return false;
    const uint64_t n_in = w->dim[0];
    const uint64_t n_out = w->dim[1];
    const uint64_t n_expert = w->dim[2];
    if (expert < 0 || (uint64_t)expert >= n_expert) return false;
    if ((n_in % QW3_QK_K) != 0) return false;

    const uint64_t blocks_per_row = n_in / QW3_QK_K;
    const uint64_t block_size = sizeof(uint16_t) + sizeof(uint16_t) +
                                QW3_QK_K / 64 + QW3_QK_K / 2;
    const uint8_t *base = tensor_data(m, w) +
                          (uint64_t)expert * n_out * blocks_per_row * block_size;
    for (uint64_t row = 0; row < n_out; row++) {
        y[row] = cpu_dot_iq4_xs_row(base + row * blocks_per_row * block_size,
                                    x, n_in);
    }
    return true;
}

static float cpu_dot_q6_k_row(const uint8_t *src, const float *x,
                              uint64_t n_in) {
    const uint64_t blocks_per_row = n_in / QW3_QK_K;
    const uint64_t block_size = QW3_QK_K / 2 + QW3_QK_K / 4 +
                                QW3_QK_K / 16 + sizeof(uint16_t);
    float sum = 0.0f;

    for (uint64_t b = 0; b < blocks_per_row; b++) {
        const uint8_t *blk = src + b * block_size;
        const uint8_t *ql = blk;
        const uint8_t *qh = ql + QW3_QK_K / 2;
        const int8_t *sc = (const int8_t *)(qh + QW3_QK_K / 4);
        const float d = qw3_f16_to_f32(qw3_load_u16(sc + QW3_QK_K / 16));
        const float *xx = x + b * QW3_QK_K;

        for (int n = 0; n < QW3_QK_K; n += 128) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0x0f) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                sum += d * (float)sc[is + 0] * (float)q1 * xx[n + l +  0];
                sum += d * (float)sc[is + 2] * (float)q2 * xx[n + l + 32];
                sum += d * (float)sc[is + 4] * (float)q3 * xx[n + l + 64];
                sum += d * (float)sc[is + 6] * (float)q4 * xx[n + l + 96];
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return sum;
}

static bool cpu_matvec_q6_k(const qw3_model *m, const qw3_tensor *w,
                            const float *x, float *y) {
    if (w->ndim != 2 || w->type != QW3_TENSOR_Q6_K) return false;
    const uint64_t n_in = w->dim[0];
    const uint64_t n_out = w->dim[1];
    if ((n_in % QW3_QK_K) != 0) return false;

    const uint64_t blocks_per_row = n_in / QW3_QK_K;
    const uint64_t block_size = QW3_QK_K / 2 + QW3_QK_K / 4 +
                                QW3_QK_K / 16 + sizeof(uint16_t);
    const uint8_t *base = tensor_data(m, w);
    for (uint64_t row = 0; row < n_out; row++) {
        y[row] = cpu_dot_q6_k_row(base + row * blocks_per_row * block_size,
                                  x, n_in);
    }
    return true;
}

static bool cpu_matvec_q6_k_expert(const qw3_model *m, const qw3_tensor *w,
                                   int expert, const float *x, float *y) {
    if (w->ndim != 3 || w->type != QW3_TENSOR_Q6_K) return false;
    const uint64_t n_in = w->dim[0];
    const uint64_t n_out = w->dim[1];
    const uint64_t n_expert = w->dim[2];
    if (expert < 0 || (uint64_t)expert >= n_expert) return false;
    if ((n_in % QW3_QK_K) != 0) return false;

    const uint64_t blocks_per_row = n_in / QW3_QK_K;
    const uint64_t block_size = QW3_QK_K / 2 + QW3_QK_K / 4 +
                                QW3_QK_K / 16 + sizeof(uint16_t);
    const uint8_t *base = tensor_data(m, w) +
                          (uint64_t)expert * n_out *
                          blocks_per_row * block_size;
    for (uint64_t row = 0; row < n_out; row++) {
        y[row] = cpu_dot_q6_k_row(base + row * blocks_per_row * block_size,
                                  x, n_in);
    }
    return true;
}

static float cpu_dot_iq3_s_row(const uint8_t *src, const float *x,
                               uint64_t n_in) {
    const uint64_t blocks_per_row = n_in / QW3_QK_K;
    const uint64_t block_size = sizeof(uint16_t) + 64 + 8 + 32 + 4;
    float sum = 0.0f;

    for (uint64_t b = 0; b < blocks_per_row; b++) {
        const uint8_t *blk = src + b * block_size;
        const float d = qw3_f16_to_f32(qw3_load_u16(blk));
        const uint8_t *qs = blk + sizeof(uint16_t);
        const uint8_t *qh = qs + 64;
        const uint8_t *signs = qh + 8;
        const uint8_t *scales = signs + 32;
        const float *xx = x + b * QW3_QK_K;
        int out = 0;

        for (int ib32 = 0; ib32 < QW3_QK_K / 32; ib32 += 2) {
            const float db1 = d * (float)(1 + 2 * (scales[ib32 / 2] & 0x0f));
            const float db2 = d * (float)(1 + 2 * (scales[ib32 / 2] >> 4));
            const uint8_t qh0 = qh[0];
            const uint8_t qh1 = qh[1];

            for (int l = 0; l < 4; l++) {
                uint8_t grid1[4], grid2[4];
                uint16_t idx1 = (uint16_t)(qs[2 * l + 0] | ((qh0 << (8 - 2 * l)) & 256));
                uint16_t idx2 = (uint16_t)(qs[2 * l + 1] | ((qh0 << (7 - 2 * l)) & 256));
                iq3s_grid4(idx1, grid1);
                iq3s_grid4(idx2, grid2);
                for (int j = 0; j < 4; j++) {
                    float sign1 = (signs[l] & qw3_kmask_iq2xs[j + 0]) ? -1.0f : 1.0f;
                    float sign2 = (signs[l] & qw3_kmask_iq2xs[j + 4]) ? -1.0f : 1.0f;
                    sum += db1 * (float)grid1[j] * sign1 * xx[out + j + 0];
                    sum += db1 * (float)grid2[j] * sign2 * xx[out + j + 4];
                }
                out += 8;
            }
            qs += 8;
            signs += 4;

            for (int l = 0; l < 4; l++) {
                uint8_t grid1[4], grid2[4];
                uint16_t idx1 = (uint16_t)(qs[2 * l + 0] | ((qh1 << (8 - 2 * l)) & 256));
                uint16_t idx2 = (uint16_t)(qs[2 * l + 1] | ((qh1 << (7 - 2 * l)) & 256));
                iq3s_grid4(idx1, grid1);
                iq3s_grid4(idx2, grid2);
                for (int j = 0; j < 4; j++) {
                    float sign1 = (signs[l] & qw3_kmask_iq2xs[j + 0]) ? -1.0f : 1.0f;
                    float sign2 = (signs[l] & qw3_kmask_iq2xs[j + 4]) ? -1.0f : 1.0f;
                    sum += db2 * (float)grid1[j] * sign1 * xx[out + j + 0];
                    sum += db2 * (float)grid2[j] * sign2 * xx[out + j + 4];
                }
                out += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
    return sum;
}

static bool cpu_matvec_iq3_s_expert(const qw3_model *m, const qw3_tensor *w,
                                    int expert, const float *x, float *y) {
    if (w->ndim != 3 || w->type != QW3_TENSOR_IQ3_S) return false;
    const uint64_t n_in = w->dim[0];
    const uint64_t n_out = w->dim[1];
    const uint64_t n_expert = w->dim[2];
    if (expert < 0 || (uint64_t)expert >= n_expert) return false;
    if ((n_in % QW3_QK_K) != 0) return false;

    const uint64_t blocks_per_row = n_in / QW3_QK_K;
    const uint64_t block_size = sizeof(uint16_t) + 64 + 8 + 32 + 4;
    const uint8_t *base = tensor_data(m, w) +
                          (uint64_t)expert * n_out * blocks_per_row * block_size;
    for (uint64_t row = 0; row < n_out; row++) {
        y[row] = cpu_dot_iq3_s_row(base + row * blocks_per_row * block_size,
                                   x, n_in);
    }
    return true;
}

static bool cpu_matvec(const qw3_model *m, const qw3_tensor *w,
                       const float *x, float *y) {
    if (tensor_is_dense_float(w->type)) return cpu_matvec_dense(m, w, x, y);
    if (w->type == QW3_TENSOR_Q8_0) return cpu_matvec_q8_0(m, w, x, y);
    if (w->type == QW3_TENSOR_Q6_K) return cpu_matvec_q6_k(m, w, x, y);
    if (w->type == QW3_TENSOR_IQ4_XS) return cpu_matvec_iq4_xs(m, w, x, y);
    return false;
}

static float cpu_silu(float x) {
    return x / (1.0f + expf(-x));
}

static float cpu_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static void cpu_rmsnorm(float *dst, const float *x, const qw3_model *m,
                        const qw3_tensor *weight, uint64_t n) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + QW3_RMS_EPS);
    for (uint64_t i = 0; i < n; i++) {
        dst[i] = x[i] * scale * tensor_read_dense_1d(m, weight, i);
    }
}

static void topk_desc(const float *scores, int n, int k,
                      int *ids, float *vals) {
    for (int i = 0; i < k; i++) {
        ids[i] = -1;
        vals[i] = -FLT_MAX;
    }
    for (int i = 0; i < n; i++) {
        float v = scores[i];
        for (int j = 0; j < k; j++) {
            if (v > vals[j]) {
                for (int s = k - 1; s > j; s--) {
                    vals[s] = vals[s - 1];
                    ids[s] = ids[s - 1];
                }
                vals[j] = v;
                ids[j] = i;
                break;
            }
        }
    }
}

static bool cpu_moe_layer(qw3_engine *e, int il, const float *x,
                          float *out, int *top_ids, float *top_scores) {
    if (!e || il < 0 || il >= QW3_N_LAYER || !x || !out) return false;
    const qw3_model *m = &e->model;
    const qw3_layer_weights *lw = &e->weights.layer[il];

    float *router = qw3_xmalloc((size_t)QW3_N_EXPERT * sizeof(float));
    float *gate = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *up = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *sgate = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *sup = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *shidden = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));

    memset(out, 0, (size_t)QW3_N_EMBD * sizeof(float));

    bool ok = cpu_matvec(m, lw->ffn_gate_inp, x, router);
    if (!ok) goto done;

    int ids[QW3_N_EXPERT_USED];
    float scores[QW3_N_EXPERT_USED];
    topk_desc(router, QW3_N_EXPERT, QW3_N_EXPERT_USED, ids, scores);
    if (top_ids) memcpy(top_ids, ids, sizeof(ids));
    if (top_scores) memcpy(top_scores, scores, sizeof(scores));

    ok = cpu_matvec(m, lw->ffn_gate_shared, x, sgate) &&
         cpu_matvec(m, lw->ffn_up_shared, x, sup);
    if (!ok) goto done;
    for (int i = 0; i < QW3_N_FF_SHARED; i++) {
        shidden[i] = cpu_silu(sgate[i]) * sup[i];
    }
    ok = cpu_matvec(m, lw->ffn_down_shared, shidden, out);
    if (!ok) goto done;
    if (lw->ffn_gate_inp_shexp) {
        float shared_gate = 0.0f;
        ok = cpu_dot_dense_1d(m, lw->ffn_gate_inp_shexp, x, &shared_gate);
        if (!ok) goto done;
        shared_gate = 1.0f / (1.0f + expf(-shared_gate));
        for (int i = 0; i < QW3_N_EMBD; i++) {
            out[i] *= shared_gate;
        }
    }

    float max_route = scores[0];
    float route_sum = 0.0f;
    float route_w[QW3_N_EXPERT_USED];
    for (int k = 0; k < QW3_N_EXPERT_USED; k++) {
        route_w[k] = expf(scores[k] - max_route);
        route_sum += route_w[k];
    }

    for (int k = 0; k < QW3_N_EXPERT_USED; k++) {
        route_w[k] /= route_sum;
        ok = cpu_matvec_iq3_s_expert(m, lw->ffn_gate_exps, ids[k], x, gate) &&
             cpu_matvec_iq3_s_expert(m, lw->ffn_up_exps, ids[k], x, up);
        if (!ok) goto done;
        for (int i = 0; i < QW3_N_FF_EXP; i++) {
            hidden[i] = cpu_silu(gate[i]) * up[i];
        }
        if (lw->ffn_down_exps->type == QW3_TENSOR_IQ4_XS) {
            ok = cpu_matvec_iq4_xs_expert(m, lw->ffn_down_exps, ids[k], hidden, down);
        } else if (lw->ffn_down_exps->type == QW3_TENSOR_Q6_K) {
            ok = cpu_matvec_q6_k_expert(m, lw->ffn_down_exps, ids[k], hidden, down);
        } else {
            ok = false;
        }
        if (!ok) goto done;
        for (int i = 0; i < QW3_N_EMBD; i++) {
            out[i] += route_w[k] * down[i];
        }
    }

done:
    free(shidden);
    free(sup);
    free(sgate);
    free(down);
    free(hidden);
    free(up);
    free(gate);
    free(router);
    return ok;
}

static void cpu_rope_head(float *x, int pos) {
    for (int i = 0; i < QW3_ROPE_DIM; i += 2) {
        const float freq = powf(QW3_ROPE_THETA, -(float)i / (float)QW3_ROPE_DIM);
        const float ang = (float)pos * freq;
        const float c = cosf(ang);
        const float s = sinf(ang);
        const float x0 = x[i + 0];
        const float x1 = x[i + 1];
        x[i + 0] = x0 * c - x1 * s;
        x[i + 1] = x0 * s + x1 * c;
    }
}

static bool cpu_gqa_project_token(qw3_engine *e, int il, int pos,
                                  const float *x,
                                  float *q, float *k, float *v,
                                  float *gate) {
    if (!e || il < 0 || il >= QW3_N_LAYER || !x || !q || !k || !v || !gate) {
        return false;
    }
    if (!qw3_layer_is_full_attention((uint32_t)il)) return false;

    const qw3_model *m = &e->model;
    const qw3_layer_weights *lw = &e->weights.layer[il];
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qfull = qw3_xmalloc((size_t)tensor_cols_qg() * sizeof(float));

    cpu_rmsnorm(xn, x, m, lw->attn_norm, QW3_N_EMBD);
    bool ok = cpu_matvec(m, lw->attn_q_proj, xn, qfull) &&
              cpu_matvec(m, lw->attn_k_proj, xn, k) &&
              cpu_matvec(m, lw->attn_v_proj, xn, v);
    if (!ok) goto done;

    for (int h = 0; h < QW3_N_HEAD; h++) {
        cpu_rmsnorm(q + h * QW3_N_HEAD_DIM,
                    qfull + h * QW3_N_HEAD_DIM * 2,
                    m, lw->attn_q_norm, QW3_N_HEAD_DIM);
        memcpy(gate + h * QW3_N_HEAD_DIM,
               qfull + h * QW3_N_HEAD_DIM * 2 + QW3_N_HEAD_DIM,
               QW3_N_HEAD_DIM * sizeof(float));
        cpu_rope_head(q + h * QW3_N_HEAD_DIM, pos);
    }
    for (int h = 0; h < QW3_N_HEAD_KV; h++) {
        float tmp[QW3_N_HEAD_DIM];
        cpu_rmsnorm(tmp, k + h * QW3_N_HEAD_DIM,
                    m, lw->attn_k_norm, QW3_N_HEAD_DIM);
        memcpy(k + h * QW3_N_HEAD_DIM, tmp, sizeof(tmp));
        cpu_rope_head(k + h * QW3_N_HEAD_DIM, pos);
    }

done:
    free(qfull);
    free(xn);
    return ok;
}

static void cpu_gqa_attend_inner(const float *q, const float *gate,
                                 const float *k_cache,
                                 const float *v_cache,
                                 int n_ctx, float *inner) {
    const float scale = 1.0f / sqrtf((float)QW3_N_HEAD_DIM);
    if (n_ctx <= 0) return;
    float *scores = qw3_xmalloc((size_t)n_ctx * sizeof(float));

    for (int h = 0; h < QW3_N_HEAD; h++) {
        const int kvh = h / (QW3_N_HEAD / QW3_N_HEAD_KV);
        const float *qh = q + h * QW3_N_HEAD_DIM;
        float max_score = -FLT_MAX;
        for (int t = 0; t < n_ctx; t++) {
            const float *kh = k_cache +
                              ((t * QW3_N_HEAD_KV + kvh) * QW3_N_HEAD_DIM);
            float dot = 0.0f;
            for (int i = 0; i < QW3_N_HEAD_DIM; i++) dot += qh[i] * kh[i];
            scores[t] = dot * scale;
            if (scores[t] > max_score) max_score = scores[t];
        }
        float denom = 0.0f;
        for (int t = 0; t < n_ctx; t++) {
            scores[t] = expf(scores[t] - max_score);
            denom += scores[t];
        }

        float *dst = inner + h * QW3_N_HEAD_DIM;
        const float *gh = gate + h * QW3_N_HEAD_DIM;
        for (int i = 0; i < QW3_N_HEAD_DIM; i++) {
            float acc = 0.0f;
            for (int t = 0; t < n_ctx; t++) {
                const float *vh = v_cache +
                                  ((t * QW3_N_HEAD_KV + kvh) * QW3_N_HEAD_DIM);
                acc += (scores[t] / denom) * vh[i];
            }
            const float g = 1.0f / (1.0f + expf(-gh[i]));
            dst[i] = acc * g;
        }
    }
    free(scores);
}

static bool cpu_gqa_single_token_layer(qw3_engine *e, int il, int pos,
                                       const float *x, float *out) {
    if (!e || il < 0 || il >= QW3_N_LAYER || !x || !out) return false;
    if (!qw3_layer_is_full_attention((uint32_t)il)) return false;

    const qw3_model *m = &e->model;
    const qw3_layer_weights *lw = &e->weights.layer[il];
    float *q = qw3_xmalloc((size_t)(QW3_N_HEAD * QW3_N_HEAD_DIM) * sizeof(float));
    float *k = qw3_xmalloc((size_t)tensor_cols_kv() * sizeof(float));
    float *v = qw3_xmalloc((size_t)tensor_cols_kv() * sizeof(float));
    float *gate = qw3_xmalloc((size_t)(QW3_N_HEAD * QW3_N_HEAD_DIM) * sizeof(float));
    float *inner = qw3_xmalloc((size_t)tensor_linear_inner() * sizeof(float));

    bool ok = cpu_gqa_project_token(e, il, pos, x, q, k, v, gate);
    if (!ok) goto done;
    cpu_gqa_attend_inner(q, gate, k, v, 1, inner);

    ok = cpu_matvec(m, lw->attn_o_proj, inner, out);

done:
    free(inner);
    free(gate);
    free(v);
    free(k);
    free(q);
    return ok;
}

static void cpu_l2_norm_head(float *dst, const float *src, int n) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)src[i] * (double)src[i];
    const float scale = 1.0f / fmaxf(sqrtf((float)ss), QW3_RMS_EPS);
    for (int i = 0; i < n; i++) dst[i] = src[i] * scale;
}

static bool cpu_deltanet_conv1d_step(const qw3_model *m,
                                     const qw3_layer_weights *lw,
                                     const float *qkv,
                                     float *conv_state,
                                     float *conv_out) {
    if (!m || !lw || !qkv || !conv_state || !conv_out) return false;
    const qw3_tensor *w = lw->linear_conv_weight;
    if (!w || w->ndim != 2 || !tensor_is_dense_float(w->type)) return false;
    if (w->dim[0] != QW3_N_LINEAR_CONV_K || w->dim[1] != tensor_linear_qkv()) {
        return false;
    }

    for (uint64_t ch = 0; ch < tensor_linear_qkv(); ch++) {
        float *st = conv_state + ch * (QW3_N_LINEAR_CONV_K - 1);
        float sum = 0.0f;
        for (int k = 0; k < QW3_N_LINEAR_CONV_K - 1; k++) {
            sum += st[k] * tensor_read_dense_linear(m, w,
                                                    ch * QW3_N_LINEAR_CONV_K + (uint64_t)k);
        }
        sum += qkv[ch] * tensor_read_dense_linear(m, w,
                                                  ch * QW3_N_LINEAR_CONV_K +
                                                  (QW3_N_LINEAR_CONV_K - 1));
        for (int k = 0; k < QW3_N_LINEAR_CONV_K - 2; k++) st[k] = st[k + 1];
        st[QW3_N_LINEAR_CONV_K - 2] = qkv[ch];
        conv_out[ch] = cpu_silu(sum);
    }
    return true;
}

static bool cpu_deltanet_layer(qw3_engine *e, int il, const float *x,
                               float *conv_state, float *state, float *out) {
    if (!e || !x || !conv_state || !state || !out) return false;
    if (il < 0 || il >= QW3_N_LAYER) return false;
    if (qw3_layer_is_full_attention((uint32_t)il)) return false;

    const qw3_model *m = &e->model;
    const qw3_layer_weights *lw = &e->weights.layer[il];
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)tensor_linear_qkv() * sizeof(float));
    float *z = qw3_xmalloc((size_t)tensor_linear_inner() * sizeof(float));
    float *alpha = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *conv = qw3_xmalloc((size_t)tensor_linear_qkv() * sizeof(float));
    float *qnorm = qw3_xmalloc((size_t)(QW3_N_LINEAR_QK_HEADS *
                                        QW3_N_LINEAR_HEAD_DIM) * sizeof(float));
    float *knorm = qw3_xmalloc((size_t)(QW3_N_LINEAR_QK_HEADS *
                                        QW3_N_LINEAR_HEAD_DIM) * sizeof(float));
    float *core = qw3_xmalloc((size_t)tensor_linear_inner() * sizeof(float));
    float *inner = qw3_xmalloc((size_t)tensor_linear_inner() * sizeof(float));
    float *attn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *resid = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *ffn_in = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *moe = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));

    bool ok = true;
    cpu_rmsnorm(xn, x, m, lw->attn_norm, QW3_N_EMBD);
    ok = cpu_matvec(m, lw->linear_qkv_proj, xn, qkv) &&
         cpu_matvec(m, lw->linear_gate_proj, xn, z) &&
         cpu_matvec(m, lw->linear_ssm_alpha, xn, alpha) &&
         cpu_matvec(m, lw->linear_ssm_beta, xn, beta);
    if (!ok) goto done;

    ok = cpu_deltanet_conv1d_step(m, lw, qkv, conv_state, conv);
    if (!ok) goto done;

    const float *qraw = conv;
    const float *kraw = conv + QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const float *vraw = kraw + QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    for (int h = 0; h < QW3_N_LINEAR_QK_HEADS; h++) {
        cpu_l2_norm_head(qnorm + h * QW3_N_LINEAR_HEAD_DIM,
                         qraw + h * QW3_N_LINEAR_HEAD_DIM,
                         QW3_N_LINEAR_HEAD_DIM);
        cpu_l2_norm_head(knorm + h * QW3_N_LINEAR_HEAD_DIM,
                         kraw + h * QW3_N_LINEAR_HEAD_DIM,
                         QW3_N_LINEAR_HEAD_DIM);
    }

    for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
        const int hk = hv % QW3_N_LINEAR_QK_HEADS;
        const float *qh = qnorm + hk * QW3_N_LINEAR_HEAD_DIM;
        const float *kh = knorm + hk * QW3_N_LINEAR_HEAD_DIM;
        const float *vh = vraw + hv * QW3_N_LINEAR_HEAD_DIM;
        float *sh = state + (uint64_t)hv * QW3_N_LINEAR_HEAD_DIM *
                    QW3_N_LINEAR_HEAD_DIM;
        float *oh = core + hv * QW3_N_LINEAR_HEAD_DIM;
        const float bh = 1.0f / (1.0f + expf(-beta[hv]));
        const float ah = cpu_softplus(alpha[hv] +
                                      tensor_read_dense_1d(m, lw->linear_ssm_dt_bias,
                                                           (uint64_t)hv));
        const float gh = expf(ah * tensor_read_dense_1d(m, lw->linear_ssm_a,
                                                        (uint64_t)hv));
        float sk[QW3_N_LINEAR_HEAD_DIM];

        for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
            for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
                sh[i * QW3_N_LINEAR_HEAD_DIM + j] *= gh;
            }
        }
        for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
            float acc = 0.0f;
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                acc += sh[i * QW3_N_LINEAR_HEAD_DIM + j] * kh[i];
            }
            sk[j] = acc;
        }
        for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
            const float d = (vh[j] - sk[j]) * bh;
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                sh[i * QW3_N_LINEAR_HEAD_DIM + j] += kh[i] * d;
            }
        }
        for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
            float acc = 0.0f;
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                acc += sh[i * QW3_N_LINEAR_HEAD_DIM + j] *
                       (qh[i] / sqrtf((float)QW3_N_LINEAR_HEAD_DIM));
            }
            oh[j] = acc;
        }
    }

    for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
        float *dst = inner + hv * QW3_N_LINEAR_HEAD_DIM;
        const float *src = core + hv * QW3_N_LINEAR_HEAD_DIM;
        const float *zh = z + hv * QW3_N_LINEAR_HEAD_DIM;
        double ss = 0.0;
        for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
            ss += (double)src[i] * (double)src[i];
        }
        const float scale = 1.0f /
                            sqrtf((float)(ss / QW3_N_LINEAR_HEAD_DIM) +
                                  QW3_RMS_EPS);
        for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
            dst[i] = src[i] * scale *
                     tensor_read_dense_1d(m, lw->linear_ssm_norm, (uint64_t)i) *
                     cpu_silu(zh[i]);
        }
    }

    ok = cpu_matvec(m, lw->linear_ssm_out, inner, attn);
    if (!ok) goto done;
    for (int i = 0; i < QW3_N_EMBD; i++) resid[i] = x[i] + attn[i];
    cpu_rmsnorm(ffn_in, resid, m, lw->ffn_norm, QW3_N_EMBD);
    ok = cpu_moe_layer(e, il, ffn_in, moe, NULL, NULL);
    if (!ok) goto done;
    for (int i = 0; i < QW3_N_EMBD; i++) out[i] = resid[i] + moe[i];

done:
    free(moe);
    free(ffn_in);
    free(resid);
    free(attn);
    free(inner);
    free(core);
    free(knorm);
    free(qnorm);
    free(conv);
    free(beta);
    free(alpha);
    free(z);
    free(qkv);
    free(xn);
    return ok;
}

static bool cpu_full_attention_layer(qw3_engine *e, int il, int pos,
                                     const float *x,
                                     float *k_cache, float *v_cache,
                                     int n_ctx, float *out) {
    if (!e || !x || !k_cache || !v_cache || !out) return false;
    if (pos < 0 || pos >= n_ctx) return false;
    if (!qw3_layer_is_full_attention((uint32_t)il)) return false;

    const qw3_model *m = &e->model;
    const qw3_layer_weights *lw = &e->weights.layer[il];
    float *q = qw3_xmalloc((size_t)(QW3_N_HEAD * QW3_N_HEAD_DIM) * sizeof(float));
    float *k = qw3_xmalloc((size_t)tensor_cols_kv() * sizeof(float));
    float *v = qw3_xmalloc((size_t)tensor_cols_kv() * sizeof(float));
    float *gate = qw3_xmalloc((size_t)(QW3_N_HEAD * QW3_N_HEAD_DIM) * sizeof(float));
    float *inner = qw3_xmalloc((size_t)tensor_linear_inner() * sizeof(float));
    float *attn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *resid = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *ffn_in = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *moe = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));

    bool ok = cpu_gqa_project_token(e, il, pos, x, q, k, v, gate);
    if (!ok) goto done;

    memcpy(k_cache + (uint64_t)pos * tensor_cols_kv(), k,
           (size_t)tensor_cols_kv() * sizeof(float));
    memcpy(v_cache + (uint64_t)pos * tensor_cols_kv(), v,
           (size_t)tensor_cols_kv() * sizeof(float));

    cpu_gqa_attend_inner(q, gate, k_cache, v_cache, pos + 1, inner);
    ok = cpu_matvec(m, lw->attn_o_proj, inner, attn);
    if (!ok) goto done;

    for (int i = 0; i < QW3_N_EMBD; i++) resid[i] = x[i] + attn[i];
    cpu_rmsnorm(ffn_in, resid, m, lw->ffn_norm, QW3_N_EMBD);
    ok = cpu_moe_layer(e, il, ffn_in, moe, NULL, NULL);
    if (!ok) goto done;
    for (int i = 0; i < QW3_N_EMBD; i++) out[i] = resid[i] + moe[i];

done:
    free(moe);
    free(ffn_in);
    free(resid);
    free(attn);
    free(inner);
    free(gate);
    free(v);
    free(k);
    free(q);
    return ok;
}

static bool qw3_cpu_eval_layer_range(qw3_session *s, int first_layer,
                                     const float *input, float *out,
                                     char *err, size_t errlen,
                                     FILE *trace, bool trace_json,
                                     bool *trace_first_event) {
    if (!s || !s->engine || !input || !out) return false;
    if (first_layer < 0) first_layer = 0;
    if (first_layer > QW3_N_LAYER) first_layer = QW3_N_LAYER;

    qw3_engine *e = s->engine;
    float *x0 = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *x1 = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    memcpy(x0, input, (size_t)QW3_N_EMBD * sizeof(float));

    bool ok = true;
    for (int il = first_layer; il < QW3_N_LAYER; il++) {
        if (s->progress_fn) s->progress_fn(s->progress_ud, "layer", il, QW3_N_LAYER);
        if (qw3_layer_is_full_attention((uint32_t)il)) {
            int fl = s->kv.full_layer_map[il];
            float *k_cache = s->kv.k_cache +
                             (uint64_t)fl * (uint64_t)s->ctx_size *
                             tensor_cols_kv();
            float *v_cache = s->kv.v_cache +
                             (uint64_t)fl * (uint64_t)s->ctx_size *
                             tensor_cols_kv();
            ok = cpu_full_attention_layer(e, il, s->kv.pos, x0,
                                          k_cache, v_cache, s->ctx_size, x1);
        } else {
            int dl = s->dn.linear_layer_map[il];
            float *conv_state = s->dn.conv_state +
                                (uint64_t)dl * tensor_linear_qkv() *
                                (QW3_N_LINEAR_CONV_K - 1);
            float *state = s->dn.state +
                           (uint64_t)dl * QW3_N_LINEAR_V_HEADS *
                           QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM;
            ok = cpu_deltanet_layer(e, il, x0, conv_state, state, x1);
        }
        if (!ok) {
            if (err && errlen) snprintf(err, errlen, "layer %d forward failed", il);
            break;
        }
        trace_emit(trace, trace_json, trace_first_event,
                   qw3_layer_is_full_attention((uint32_t)il) ? "full" : "linear",
                   il, x1, QW3_N_EMBD);
        float *tmp = x0;
        x0 = x1;
        x1 = tmp;
    }

    if (ok) memcpy(out, x0, (size_t)QW3_N_EMBD * sizeof(float));
    free(x1);
    free(x0);
    return ok;
}

static bool qw3_cpu_output_logits(qw3_session *s, const float *x,
                                  char *err, size_t errlen) {
    if (!s || !s->engine || !x || !s->logits) return false;
    const qw3_model *m = &s->engine->model;
    const qw3_weights *w = &s->engine->weights;
    float *norm = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    cpu_rmsnorm(norm, x, m, w->output_norm, QW3_N_EMBD);
    bool ok = cpu_matvec(m, w->output, norm, s->logits);
    if (!ok && err && errlen) {
        snprintf(err, errlen, "output projection failed for tensor type %s",
                 tensor_type_name(w->output->type));
    }
    free(norm);
    return ok;
}

static void vocab_bind(qw3_engine *e) {
    const qw3_model *m = &e->model;
    qw3_vocab *v = &e->vocab;

    v->n_vocab = (uint32_t)m->meta.tokenizer_token_count;
    v->bos_id = (int)m->meta.bos_token_id;
    v->eos_id = (int)m->meta.eos_token_id;
    v->im_start_id = -1;
    v->im_end_id = -1;
    v->turn_start_id = -1;
    v->turn_end_id = -1;
    v->think_id = -1;
    v->think_end_id = -1;
    v->channel_start_id = -1;
    v->channel_end_id = -1;
    v->id_to_text = qw3_xcalloc(v->n_vocab, sizeof(v->id_to_text[0]));
    v->id_to_text_len = qw3_xcalloc(v->n_vocab, sizeof(v->id_to_text_len[0]));

    table_init(&v->token_to_id, v->n_vocab);
    for (uint32_t i = 0; i < v->n_vocab; i++) {
        qw3_str tok = m->meta.token_texts[i];
        v->id_to_text[i] = tok.ptr;
        v->id_to_text_len[i] = tok.len;
        table_insert(&v->token_to_id, tok.ptr, tok.len, (int)i);
    }

    table_init(&v->merge_rank, (uint32_t)m->meta.tokenizer_merge_count);
    for (int64_t i = 0; i < m->meta.tokenizer_merge_count; i++) {
        qw3_str merge = m->meta.merge_texts[i];
        table_insert(&v->merge_rank, merge.ptr, merge.len, (int)i);
    }

    table_get(&v->token_to_id, "<|im_start|>", strlen("<|im_start|>"), &v->im_start_id);
    table_get(&v->token_to_id, "<|im_end|>", strlen("<|im_end|>"), &v->im_end_id);
    table_get(&v->token_to_id, "<|turn>", strlen("<|turn>"), &v->turn_start_id);
    table_get(&v->token_to_id, "<turn|>", strlen("<turn|>"), &v->turn_end_id);
    table_get(&v->token_to_id, "<think>", strlen("<think>"), &v->think_id);
    table_get(&v->token_to_id, "</think>", strlen("</think>"), &v->think_end_id);
    table_get(&v->token_to_id, "<|channel>", strlen("<|channel>"), &v->channel_start_id);
    table_get(&v->token_to_id, "<channel|>", strlen("<channel|>"), &v->channel_end_id);
}

static void vocab_free(qw3_vocab *v) {
    free(v->id_to_text);
    free(v->id_to_text_len);
    table_free(&v->token_to_id);
    table_free(&v->merge_rank);
    memset(v, 0, sizeof(*v));
}

/* =========================================================================
 * Qwen35 byte-level BPE tokenizer.
 * =========================================================================
 *
 * This is intentionally model-specific: Qwen35 uses GPT-2 byte encoding plus
 * the qwen35 pre-tokenizer pattern.  We keep the implementation in this file
 * instead of importing a tokenizer framework so the engine remains vertical.
 */

static void utf8_put(char **p, uint32_t cp) {
    if (cp <= 0x7f) {
        *(*p)++ = (char)cp;
    } else if (cp <= 0x7ff) {
        *(*p)++ = (char)(0xc0 | (cp >> 6));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        *(*p)++ = (char)(0xe0 | (cp >> 12));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else {
        *(*p)++ = (char)(0xf0 | (cp >> 18));
        *(*p)++ = (char)(0x80 | ((cp >> 12) & 0x3f));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    }
}

static uint32_t gpt2_byte_to_codepoint(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) {
        return b;
    }
    uint32_t n = 0;
    for (uint32_t x = 0; x < 256; x++) {
        if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || x >= 174) {
            continue;
        }
        if (x == b) return 256 + n;
        n++;
    }
    return b;
}

static char *byte_encode(qw3_str in, uint64_t *out_len) {
    char *out = qw3_xmalloc((size_t)in.len * 4 + 1);
    char *p = out;
    for (uint64_t i = 0; i < in.len; i++) {
        utf8_put(&p, gpt2_byte_to_codepoint((uint8_t)in.ptr[i]));
    }
    *p = '\0';
    *out_len = (uint64_t)(p - out);
    return out;
}

static int utf8_len_from_first_byte(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

typedef struct {
    char *ptr;
    uint64_t len;
} owned_str;

static owned_str owned_copy(const char *ptr, uint64_t len) {
    owned_str s;
    s.ptr = qw3_xmalloc((size_t)len);
    memcpy(s.ptr, ptr, (size_t)len);
    s.len = len;
    return s;
}

static int bpe_rank(const qw3_vocab *vocab, const owned_str *a, const owned_str *b) {
    uint64_t len = a->len + 1 + b->len;
    char stack[512];
    char *buf = len <= sizeof(stack) ? stack : qw3_xmalloc((size_t)len);

    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1, b->ptr, (size_t)b->len);

    int rank = -1;
    table_get(&vocab->merge_rank, buf, len, &rank);
    if (buf != stack) free(buf);
    return rank;
}

static void bpe_emit_piece(const qw3_vocab *vocab, qw3_str raw_piece,
                           token_vec *out) {
    uint64_t encoded_len = 0;
    char *encoded = byte_encode(raw_piece, &encoded_len);

    int n_sym = 0;
    int cap_sym = 32;
    owned_str *sym = qw3_xcalloc((size_t)cap_sym, sizeof(sym[0]));

    for (uint64_t off = 0; off < encoded_len;) {
        int n = utf8_len_from_first_byte((uint8_t)encoded[off]);
        if (off + (uint64_t)n > encoded_len) n = 1;
        if (n_sym == cap_sym) {
            cap_sym *= 2;
            sym = qw3_xrealloc(sym, (size_t)cap_sym * sizeof(sym[0]));
        }
        sym[n_sym++] = owned_copy(encoded + off, (uint64_t)n);
        off += (uint64_t)n;
    }

    for (;;) {
        int best_i = -1;
        int best_rank = INT32_MAX;
        for (int i = 0; i + 1 < n_sym; i++) {
            int rank = bpe_rank(vocab, &sym[i], &sym[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }
        if (best_i < 0) break;

        owned_str merged;
        merged.len = sym[best_i].len + sym[best_i + 1].len;
        merged.ptr = qw3_xmalloc((size_t)merged.len);
        memcpy(merged.ptr, sym[best_i].ptr, (size_t)sym[best_i].len);
        memcpy(merged.ptr + sym[best_i].len,
               sym[best_i + 1].ptr, (size_t)sym[best_i + 1].len);

        free(sym[best_i].ptr);
        free(sym[best_i + 1].ptr);
        sym[best_i] = merged;
        for (int j = best_i + 1; j + 1 < n_sym; j++) sym[j] = sym[j + 1];
        n_sym--;
    }

    for (int i = 0; i < n_sym; i++) {
        int token = -1;
        if (table_get(&vocab->token_to_id, sym[i].ptr, sym[i].len, &token)) {
            token_vec_push(out, token);
        } else {
            for (uint64_t j = 0; j < sym[i].len; j++) {
                if (table_get(&vocab->token_to_id, sym[i].ptr + j, 1, &token)) {
                    token_vec_push(out, token);
                }
            }
        }
        free(sym[i].ptr);
    }

    free(sym);
    free(encoded);
}

static bool ascii_alpha(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static bool ascii_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static bool ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}

static bool ascii_newline(uint8_t c) {
    return c == '\n' || c == '\r';
}

static uint8_t ascii_tolower_u8(uint8_t c) {
    return (c >= 'A' && c <= 'Z') ? (uint8_t)(c + 32) : c;
}

static bool qwen35_letter_like_at(const char *s, uint64_t len, uint64_t pos) {
    uint8_t c = (uint8_t)s[pos];
    if (c < 128) return ascii_alpha(c);
    (void)len;
    return true;
}

static uint64_t next_utf8_char(const char *s, uint64_t len, uint64_t pos) {
    int n = utf8_len_from_first_byte((uint8_t)s[pos]);
    if (pos + (uint64_t)n > len) n = 1;
    return pos + (uint64_t)n;
}

static uint64_t qwen35_consume_letters(const char *s, uint64_t len,
                                       uint64_t pos) {
    while (pos < len && qwen35_letter_like_at(s, len, pos)) {
        pos = next_utf8_char(s, len, pos);
    }
    return pos;
}

static void bpe_tokenize_text(const qw3_vocab *vocab, const char *text,
                              token_vec *out) {
    const uint64_t len = text ? strlen(text) : 0;
    uint64_t pos = 0;

    while (pos < len) {
        uint64_t start = pos;
        uint8_t c = (uint8_t)text[pos];

        if (c == '\'' && pos + 1 < len) {
            uint8_t c1 = ascii_tolower_u8((uint8_t)text[pos + 1]);
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                pos += 2;
            } else if (pos + 2 < len) {
                uint8_t c2 = ascii_tolower_u8((uint8_t)text[pos + 2]);
                if ((c1 == 'r' && c2 == 'e') ||
                    (c1 == 'v' && c2 == 'e') ||
                    (c1 == 'l' && c2 == 'l')) {
                    pos += 3;
                }
            }
        }

        if (pos == start) {
            if (!ascii_newline(c) && !ascii_digit(c)) {
                if (qwen35_letter_like_at(text, len, pos) ||
                    (pos + 1 < len && qwen35_letter_like_at(text, len, pos + 1))) {
                    if (!qwen35_letter_like_at(text, len, pos)) pos++;
                    pos = qwen35_consume_letters(text, len, pos);
                }
            }
        }

        if (pos == start && ascii_digit(c)) {
            pos++;
        }

        if (pos == start) {
            uint64_t p = pos + (c == ' ' && pos + 1 < len ? 1 : 0);
            if (p < len) {
                uint8_t pc = (uint8_t)text[p];
                if (!ascii_space(pc) && !qwen35_letter_like_at(text, len, p) &&
                    !ascii_digit(pc)) {
                    pos = p;
                    while (pos < len) {
                        uint8_t x = (uint8_t)text[pos];
                        if (ascii_space(x) || qwen35_letter_like_at(text, len, pos) ||
                            ascii_digit(x)) break;
                        pos = next_utf8_char(text, len, pos);
                    }
                    while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
                }
            }
        }

        if (pos == start && ascii_space(c)) {
            uint64_t p = pos;
            uint64_t last_newline_end = 0;
            while (p < len && ascii_space((uint8_t)text[p])) {
                uint8_t sc = (uint8_t)text[p++];
                if (ascii_newline(sc)) last_newline_end = p;
            }
            if (last_newline_end) {
                pos = last_newline_end;
            } else if (p < len && p > pos + 1) {
                pos = p - 1;
            } else {
                pos = p;
            }
        }

        if (pos == start) pos = next_utf8_char(text, len, pos);
        bpe_emit_piece(vocab, (qw3_str){ text + start, pos - start }, out);
    }
}

static bool special_token_text_at(const qw3_vocab *vocab, const char *p,
                                  int *token, size_t *len) {
    struct special {
        const char *text;
        int token;
    } specials[] = {
        {"<|im_start|>", vocab->im_start_id},
        {"<|im_end|>",   vocab->im_end_id},
        {"<|turn>",    vocab->turn_start_id},
        {"<turn|>",    vocab->turn_end_id},
        {"<think>",    vocab->think_id},
        {"</think>",   vocab->think_end_id},
        {"<|channel>", vocab->channel_start_id},
        {"<channel|>", vocab->channel_end_id},
    };

    for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); i++) {
        if (specials[i].token < 0) continue;
        size_t n = strlen(specials[i].text);
        if (!strncmp(p, specials[i].text, n)) {
            *token = specials[i].token;
            *len = n;
            return true;
        }
    }
    return false;
}

static void tokenize_span(const qw3_vocab *vocab, const char *p, size_t n,
                          token_vec *out) {
    if (n == 0) return;
    char *tmp = qw3_xmalloc(n + 1);
    memcpy(tmp, p, n);
    tmp[n] = '\0';
    bpe_tokenize_text(vocab, tmp, out);
    free(tmp);
}

static void tokenize_rendered_chat_vocab(const qw3_vocab *vocab,
                                         const char *text,
                                         token_vec *out) {
    if (!text) text = "";
    const char *span = text;
    const char *p = text;

    while (*p) {
        int token = -1;
        size_t n = 0;
        if (special_token_text_at(vocab, p, &token, &n)) {
            tokenize_span(vocab, span, (size_t)(p - span), out);
            token_vec_push(out, token);
            p += n;
            span = p;
            continue;
        }
        p++;
    }
    tokenize_span(vocab, span, (size_t)(p - span), out);
}

static void emit_text(const qw3_vocab *vocab, const char *text, token_vec *out) {
    bpe_tokenize_text(vocab, text ? text : "", out);
}

static void emit_special_or_text(const qw3_vocab *vocab, const char *text,
                                 int token, token_vec *out) {
    if (token >= 0) {
        token_vec_push(out, token);
    } else {
        emit_text(vocab, text, out);
    }
}

static void emit_trimmed_text(const qw3_vocab *vocab, const char *text,
                              token_vec *out) {
    if (!text) return;
    const char *start = text;
    while (*start && ascii_space((uint8_t)*start)) start++;
    const char *end = start + strlen(start);
    while (end > start && ascii_space((uint8_t)end[-1])) end--;
    tokenize_span(vocab, start, (size_t)(end - start), out);
}

/* =========================================================================
 * Engine open.
 * =========================================================================
 */

int qw3_engine_open(qw3_engine **out, const qw3_engine_options *opt) {
    if (!out || !opt || !qw3_backend_supported(opt->backend)) {
        return -1;
    }
    qw3_engine *e = qw3_xcalloc(1, sizeof(*e));
    e->backend = opt->backend;
    double t0 = qw3_now_sec();

    /* Re-use model_open and structure from ds4 to avoid repeating boiler. */
    /* Map the file. metal_mapping=true if backend is METAL. */
    model_open(&e->model, opt->model_path, opt->backend == QW3_BACKEND_METAL, opt->warm_weights);

    qw3_log(stderr, QW3_LOG_TIMING, "qw3: loaded %s (size=%.1f GiB, %" PRIu64 " tensors, %.3fs)\n",
            opt->model_path, (double)e->model.map_size / (1024.0*1024.0*1024.0),
            e->model.n_tensors, qw3_now_sec() - t0);

    weights_bind(e);
    vocab_bind(e);

#ifndef QW3_NO_METAL
    if (e->backend == QW3_BACKEND_METAL) {
        if (!qw3_metal_init()) {
            qw3_engine_close(e);
            return -1;
        }
        uint64_t data_offset = e->model.tensor_data_offset;
        uint64_t data_size = e->model.map_size - data_offset;
        if (!qw3_metal_set_model_map_range(e->model.map, e->model.map_size,
                                           data_offset, data_size)) {
            qw3_engine_close(e);
            return -1;
        }
        e->metal_ready = true;
        qw3_log(stderr, QW3_LOG_OK,
                "qw3: Metal backend initialized for graph bring-up (%s)\n",
                qw3_metal_device_name());
    }
#endif
    
    *out = e;
    return 0;
}

void qw3_engine_close(qw3_engine *e) {
    if (!e) return;
#ifndef QW3_NO_METAL
    if (e->metal_ready) {
        qw3_metal_cleanup();
        e->metal_ready = false;
    }
#endif
    vocab_free(&e->vocab);
    free(e->model.meta.token_texts);
    free(e->model.meta.token_types);
    free(e->model.meta.merge_texts);
    free(e->model.tensors);
    if (e->model.map && e->model.map_size) {
        munmap(e->model.map, (size_t)e->model.map_size);
    }
    if (e->model.fd >= 0) {
        close(e->model.fd);
    }
    free(e);
}

char *qw3_token_text(qw3_engine *e, int token, size_t *len) {
    if (!e || token < 0 || (uint32_t)token >= e->vocab.n_vocab) {
        if (len) *len = 0;
        return NULL;
    }
    uint64_t n = e->vocab.id_to_text_len[token];
    char *out = qw3_xmalloc((size_t)n + 1);
    memcpy(out, e->vocab.id_to_text[token], (size_t)n);
    out[n] = '\0';
    if (len) *len = (size_t)n;
    return out;
}

int qw3_token_eos(qw3_engine *e) {
    return e ? e->vocab.eos_id : -1;
}

int qw3_vocab_size(qw3_engine *e) {
    return e ? (int)e->vocab.n_vocab : 0;
}

void qw3_tokenize_text(qw3_engine *e, const char *text, qw3_tokens *out) {
    if (!e || !text || !out) return;
    bpe_tokenize_text(&e->vocab, text, out);
}

void qw3_tokenize_rendered_chat(qw3_engine *e, const char *text,
                                qw3_tokens *out) {
    if (!e || !out) return;
    tokenize_rendered_chat_vocab(&e->vocab, text, out);
}

void qw3_encode_chat_prompt(qw3_engine *e,
                            const char *system,
                            const char *prompt,
                            qw3_think_mode think_mode,
                            qw3_tokens *out) {
    if (!e || !out) return;
    const qw3_vocab *v = &e->vocab;

    if (system && system[0]) {
        emit_special_or_text(v, "<|im_start|>", v->im_start_id, out);
        emit_text(v, "system\n", out);
        emit_trimmed_text(v, system, out);
        emit_special_or_text(v, "<|im_end|>", v->im_end_id, out);
        emit_text(v, "\n", out);
    }

    emit_special_or_text(v, "<|im_start|>", v->im_start_id, out);
    emit_text(v, "user\n", out);
    emit_trimmed_text(v, prompt ? prompt : "", out);
    emit_special_or_text(v, "<|im_end|>", v->im_end_id, out);
    emit_text(v, "\n", out);

    qw3_chat_append_assistant_prefix(e, out, think_mode);
}

void qw3_chat_append_message(qw3_engine *e, qw3_tokens *tokens,
                             const char *role, const char *content) {
    if (!e || !tokens) return;
    const qw3_vocab *v = &e->vocab;
    if (!role) role = "user";
    if (!strcmp(role, "assistant")) role = "model";

    emit_special_or_text(v, "<|im_start|>", v->im_start_id, tokens);
    emit_text(v, !strcmp(role, "model") ? "assistant" : role, tokens);
    emit_text(v, "\n", tokens);
    emit_trimmed_text(v, content ? content : "", tokens);
    emit_special_or_text(v, "<|im_end|>", v->im_end_id, tokens);
    emit_text(v, "\n", tokens);
}

void qw3_chat_append_assistant_prefix(qw3_engine *e, qw3_tokens *tokens,
                                      qw3_think_mode think_mode) {
    if (!e || !tokens) return;
    const qw3_vocab *v = &e->vocab;

    emit_special_or_text(v, "<|im_start|>", v->im_start_id, tokens);
    emit_text(v, "assistant\n", tokens);
    if (qw3_think_mode_enabled(think_mode)) {
        emit_special_or_text(v, "<think>", v->think_id, tokens);
        emit_text(v, "\n", tokens);
    } else {
        emit_special_or_text(v, "<think>", v->think_id, tokens);
        emit_text(v, "\n\n", tokens);
        emit_special_or_text(v, "</think>", v->think_end_id, tokens);
        emit_text(v, "\n\n", tokens);
    }
}

void qw3_engine_summary(qw3_engine *e) {
    if (!e) return;
    qw3_log(stderr, QW3_LOG_OK,
            "qw3: Qwen3.6-35B-A3B  layers=%d  embd=%d  vocab=%d\n",
            QW3_N_LAYER, QW3_N_EMBD, QW3_N_VOCAB);
    qw3_log(stderr, QW3_LOG_OK,
            "qw3: full_attn_layers=%d (GQA: %d Q-heads, %d KV-heads, dim=%d)\n",
            QW3_N_FULL_ATTN_LAYERS, QW3_N_HEAD, QW3_N_HEAD_KV, QW3_N_HEAD_DIM);
    qw3_log(stderr, QW3_LOG_OK,
            "qw3: linear_layers=%d (DeltaNet: %d QK-heads, %d V-heads, dim=%d)\n",
            QW3_N_LINEAR_LAYERS, QW3_N_LINEAR_QK_HEADS,
            QW3_N_LINEAR_V_HEADS, QW3_N_LINEAR_HEAD_DIM);
    qw3_log(stderr, QW3_LOG_OK,
            "qw3: moe: %d experts, top-%d routing + %d shared, ff_dim=%d\n",
            QW3_N_EXPERT, QW3_N_EXPERT_USED, QW3_N_EXPERT_SHARED, QW3_N_FF_EXP);
}

void qw3_engine_inspect(qw3_engine *e, FILE *fp) {
    if (!e) return;
    const qw3_model *m = &e->model;
    fprintf(fp, "architecture: %.*s\n",
            (int)m->meta.architecture.len, m->meta.architecture.ptr);
    fprintf(fp, "gguf: kv=%" PRIu64 " tensors=%u data_offset=%" PRIu64 "\n",
            m->n_kv, m->n_tensors, m->tensor_data_offset);
    fprintf(fp, "shape: layers=%" PRId64 " embd=%" PRId64
                " ctx=%" PRId64 " vocab=%" PRId64 "\n",
            m->meta.block_count, m->meta.embedding_length,
            m->meta.context_length, m->meta.tokenizer_token_count);
    fprintf(fp, "attention: heads=%" PRId64 " kv_heads=%" PRId64
                " head_dim=%" PRId64 " full_interval=%" PRId64 "\n",
            m->meta.head_count, m->meta.head_count_kv,
            m->meta.key_length, m->meta.full_attention_interval);
    fprintf(fp, "deltanet: conv=%" PRId64 " state=%" PRId64
                " groups=%" PRId64 " value_heads=%" PRId64
                " inner=%" PRId64 "\n",
            m->meta.ssm_conv_kernel, m->meta.ssm_state_size,
            m->meta.ssm_group_count, m->meta.ssm_time_step_rank,
            m->meta.ssm_inner_size);
    fprintf(fp, "moe: experts=%" PRId64 " active=%" PRId64
                " expert_ff=%" PRId64 " shared_ff=%" PRId64 "\n",
            m->meta.expert_count, m->meta.expert_used_count,
            m->meta.expert_ffn_length, m->meta.expert_shared_ffn_length);
    fprintf(fp, "rope: dim=%" PRId64 " theta=%.1f sections=",
            m->meta.rope_dimension_count, m->meta.rope_freq_base);
    for (int i = 0; i < m->meta.rope_sections_len; i++) {
        fprintf(fp, "%s%d", i ? "," : "", m->meta.rope_sections[i]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "tokenizer: model=%.*s pre=%.*s bos=%" PRId64
                " eos=%" PRId64 " pad=%" PRId64
                " merges=%" PRId64 " add_bos=%s\n",
            (int)m->meta.tokenizer_model.len, m->meta.tokenizer_model.ptr,
            (int)m->meta.tokenizer_pre.len, m->meta.tokenizer_pre.ptr,
            m->meta.bos_token_id, m->meta.eos_token_id,
            m->meta.padding_token_id, m->meta.tokenizer_merge_count,
            m->meta.add_bos_token ? "true" : "false");
}

void qw3_engine_layer_types(qw3_engine *e, int layer, FILE *fp) {
    if (!e || !fp) return;
    if (layer < 0 || layer >= QW3_N_LAYER) {
        fprintf(fp, "layer-types: layer %d is outside 0..%d\n",
                layer, QW3_N_LAYER - 1);
        return;
    }

    const qw3_layer_weights *lw = &e->weights.layer[layer];
    fprintf(fp, "layer %d: kind=%s\n", layer,
            qw3_layer_is_full_attention((uint32_t)layer) ? "gqa" : "deltanet");
    fprintf(fp, "layer %d moe: router=%s shared_router=%s gate=%s up=%s down=%s shared_gate=%s shared_up=%s shared_down=%s\n",
            layer,
            lw->ffn_gate_inp ? tensor_type_name(lw->ffn_gate_inp->type) : "none",
            lw->ffn_gate_inp_shexp ? tensor_type_name(lw->ffn_gate_inp_shexp->type) : "none",
            lw->ffn_gate_exps ? tensor_type_name(lw->ffn_gate_exps->type) : "none",
            lw->ffn_up_exps ? tensor_type_name(lw->ffn_up_exps->type) : "none",
            lw->ffn_down_exps ? tensor_type_name(lw->ffn_down_exps->type) : "none",
            lw->ffn_gate_shared ? tensor_type_name(lw->ffn_gate_shared->type) : "none",
            lw->ffn_up_shared ? tensor_type_name(lw->ffn_up_shared->type) : "none",
            lw->ffn_down_shared ? tensor_type_name(lw->ffn_down_shared->type) : "none");
    if (qw3_layer_is_full_attention((uint32_t)layer)) {
        fprintf(fp, "layer %d gqa: q=%s k=%s v=%s o=%s q_norm=%s k_norm=%s attn_norm=%s ffn_norm=%s\n",
                layer,
                lw->attn_q_proj ? tensor_type_name(lw->attn_q_proj->type) : "none",
                lw->attn_k_proj ? tensor_type_name(lw->attn_k_proj->type) : "none",
                lw->attn_v_proj ? tensor_type_name(lw->attn_v_proj->type) : "none",
                lw->attn_o_proj ? tensor_type_name(lw->attn_o_proj->type) : "none",
                lw->attn_q_norm ? tensor_type_name(lw->attn_q_norm->type) : "none",
                lw->attn_k_norm ? tensor_type_name(lw->attn_k_norm->type) : "none",
                lw->attn_norm ? tensor_type_name(lw->attn_norm->type) : "none",
                lw->ffn_norm ? tensor_type_name(lw->ffn_norm->type) : "none");
    } else {
        fprintf(fp, "layer %d deltanet: qkv=%s z=%s alpha=%s beta=%s conv=%s out=%s norm=%s a=%s dt=%s attn_norm=%s ffn_norm=%s\n",
                layer,
                lw->linear_qkv_proj ? tensor_type_name(lw->linear_qkv_proj->type) : "none",
                lw->linear_gate_proj ? tensor_type_name(lw->linear_gate_proj->type) : "none",
                lw->linear_ssm_alpha ? tensor_type_name(lw->linear_ssm_alpha->type) : "none",
                lw->linear_ssm_beta ? tensor_type_name(lw->linear_ssm_beta->type) : "none",
                lw->linear_conv_weight ? tensor_type_name(lw->linear_conv_weight->type) : "none",
                lw->linear_ssm_out ? tensor_type_name(lw->linear_ssm_out->type) : "none",
                lw->linear_ssm_norm ? tensor_type_name(lw->linear_ssm_norm->type) : "none",
                lw->linear_ssm_a ? tensor_type_name(lw->linear_ssm_a->type) : "none",
                lw->linear_ssm_dt_bias ? tensor_type_name(lw->linear_ssm_dt_bias->type) : "none",
                lw->attn_norm ? tensor_type_name(lw->attn_norm->type) : "none",
                lw->ffn_norm ? tensor_type_name(lw->ffn_norm->type) : "none");
    }
}

void qw3_engine_probe_token(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return;
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "probe: token %d is outside vocab\n", token);
        return;
    }

    const qw3_model *m = &e->model;
    const qw3_weights *w = &e->weights;
    const qw3_tensor *emb = w->token_embd;
    fprintf(fp, "probe: token=%d\n", token);
    fprintf(fp, "token_embd: type=%s dims=[%" PRIu64 ",%" PRIu64 "]\n",
            tensor_type_name(emb->type), emb->dim[0], emb->dim[1]);
    fprintf(fp, "layer0 moe tensors: router=%s shared_router=%s gate=%s up=%s down=%s shared_gate=%s shared_up=%s shared_down=%s\n",
            tensor_type_name(w->layer[0].ffn_gate_inp->type),
            w->layer[0].ffn_gate_inp_shexp ?
                tensor_type_name(w->layer[0].ffn_gate_inp_shexp->type) : "none",
            tensor_type_name(w->layer[0].ffn_gate_exps->type),
            tensor_type_name(w->layer[0].ffn_up_exps->type),
            tensor_type_name(w->layer[0].ffn_down_exps->type),
            tensor_type_name(w->layer[0].ffn_gate_shared->type),
            tensor_type_name(w->layer[0].ffn_up_shared->type),
            tensor_type_name(w->layer[0].ffn_down_shared->type));
    fprintf(fp, "layer0 deltanet tensors: qkv=%s z=%s alpha=%s beta=%s conv=%s out=%s norm=%s a=%s dt=%s\n",
            tensor_type_name(w->layer[0].linear_qkv_proj->type),
            tensor_type_name(w->layer[0].linear_gate_proj->type),
            tensor_type_name(w->layer[0].linear_ssm_alpha->type),
            tensor_type_name(w->layer[0].linear_ssm_beta->type),
            tensor_type_name(w->layer[0].linear_conv_weight->type),
            tensor_type_name(w->layer[0].linear_ssm_out->type),
            tensor_type_name(w->layer[0].linear_ssm_norm->type),
            tensor_type_name(w->layer[0].linear_ssm_a->type),
            tensor_type_name(w->layer[0].linear_ssm_dt_bias->type));
    fprintf(fp, "layer3 gqa tensors: q=%s k=%s v=%s o=%s q_norm=%s k_norm=%s\n",
            tensor_type_name(w->layer[3].attn_q_proj->type),
            tensor_type_name(w->layer[3].attn_k_proj->type),
            tensor_type_name(w->layer[3].attn_v_proj->type),
            tensor_type_name(w->layer[3].attn_o_proj->type),
            tensor_type_name(w->layer[3].attn_q_norm->type),
            tensor_type_name(w->layer[3].attn_k_norm->type));
    fprintf(fp, "layer_pattern:");
    for (int i = 0; i < QW3_N_LAYER; i++) {
        fprintf(fp, "%s%c", i ? "," : " ",
                qw3_layer_is_full_attention((uint32_t)i) ? 'F' : 'L');
    }
    fprintf(fp, "\n");

    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *router = qw3_xmalloc((size_t)QW3_N_EXPERT * sizeof(float));
    float *sg = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *su = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *sh = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *sd = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *shared_out = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *sparse_out = qw3_xcalloc(QW3_N_EMBD, sizeof(float));

    if (!tensor_read_dense_row(m, emb, (uint64_t)token, x)) {
        fprintf(fp, "probe: embedding read needs dequant for tensor type %s\n",
                tensor_type_name(emb->type));
        free(sparse_out);
        free(shared_out);
        free(sd);
        free(sh);
        free(su);
        free(sg);
        free(router);
        free(xn);
        free(x);
        return;
    }

    double mean = 0.0;
    double ss = 0.0;
    for (int i = 0; i < QW3_N_EMBD; i++) {
        mean += x[i];
        ss += (double)x[i] * (double)x[i];
    }
    mean /= QW3_N_EMBD;
    fprintf(fp, "embedding: mean=%.7g rms=%.7g first=[%.7g %.7g %.7g %.7g]\n",
            mean, sqrt(ss / QW3_N_EMBD), x[0], x[1], x[2], x[3]);

    float *lin_in = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *lin_qkv = qw3_xmalloc((size_t)tensor_linear_qkv() * sizeof(float));
    float *lin_z = qw3_xmalloc((size_t)tensor_linear_inner() * sizeof(float));
    float *lin_alpha = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *lin_beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    cpu_rmsnorm(lin_in, x, m, w->layer[0].attn_norm, QW3_N_EMBD);
    if (cpu_matvec(m, w->layer[0].linear_qkv_proj, lin_in, lin_qkv) &&
        cpu_matvec(m, w->layer[0].linear_gate_proj, lin_in, lin_z) &&
        cpu_matvec(m, w->layer[0].linear_ssm_alpha, lin_in, lin_alpha) &&
        cpu_matvec(m, w->layer[0].linear_ssm_beta, lin_in, lin_beta)) {
        double qrms = 0.0;
        double krms = 0.0;
        double vrms = 0.0;
        double zrms = 0.0;
        const float *lin_q = lin_qkv;
        const float *lin_k = lin_qkv + QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
        const float *lin_v = lin_k + QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
        for (int i = 0; i < QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM; i++) {
            qrms += (double)lin_q[i] * (double)lin_q[i];
            krms += (double)lin_k[i] * (double)lin_k[i];
        }
        for (int i = 0; i < QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM; i++) {
            vrms += (double)lin_v[i] * (double)lin_v[i];
            zrms += (double)lin_z[i] * (double)lin_z[i];
        }
        float beta0[4];
        float alpha0[4];
        float gate0[4];
        for (int i = 0; i < 4; i++) {
            beta0[i] = 1.0f / (1.0f + expf(-lin_beta[i]));
            alpha0[i] = cpu_softplus(lin_alpha[i] +
                                     tensor_read_dense_1d(m, w->layer[0].linear_ssm_dt_bias, (uint64_t)i));
            gate0[i] = alpha0[i] *
                       tensor_read_dense_1d(m, w->layer[0].linear_ssm_a, (uint64_t)i);
        }
        fprintf(fp, "layer0 deltanet projection: q_rms=%.7g k_rms=%.7g v_rms=%.7g z_rms=%.7g q0=[%.7g %.7g %.7g %.7g]\n",
                sqrt(qrms / (QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM)),
                sqrt(krms / (QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM)),
                sqrt(vrms / (QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM)),
                sqrt(zrms / (QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM)),
                lin_q[0], lin_q[1], lin_q[2], lin_q[3]);
        fprintf(fp, "layer0 deltanet gates: beta=[%.7g %.7g %.7g %.7g] alpha=[%.7g %.7g %.7g %.7g] dt_a=[%.7g %.7g %.7g %.7g]\n",
                beta0[0], beta0[1], beta0[2], beta0[3],
                alpha0[0], alpha0[1], alpha0[2], alpha0[3],
                gate0[0], gate0[1], gate0[2], gate0[3]);
    } else {
        fprintf(fp, "probe: layer0 deltanet projection failed\n");
    }
    free(lin_beta);
    free(lin_alpha);
    free(lin_z);
    free(lin_qkv);
    free(lin_in);

    float *dn_conv_state = qw3_xcalloc((size_t)tensor_linear_qkv() *
                                       (QW3_N_LINEAR_CONV_K - 1), sizeof(float));
    float *dn_state = qw3_xcalloc((size_t)QW3_N_LINEAR_V_HEADS *
                                  QW3_N_LINEAR_HEAD_DIM *
                                  QW3_N_LINEAR_HEAD_DIM, sizeof(float));
    float *dn_out = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    if (cpu_deltanet_layer(e, 0, x, dn_conv_state, dn_state, dn_out)) {
        double dmean = 0.0;
        double drms = 0.0;
        for (int i = 0; i < QW3_N_EMBD; i++) {
            dmean += dn_out[i];
            drms += (double)dn_out[i] * (double)dn_out[i];
        }
        fprintf(fp, "layer0 cpu_deltanet_layer: mean=%.7g rms=%.7g first=[%.7g %.7g %.7g %.7g]\n",
                dmean / QW3_N_EMBD, sqrt(drms / QW3_N_EMBD),
                dn_out[0], dn_out[1], dn_out[2], dn_out[3]);
    } else {
        fprintf(fp, "probe: cpu_deltanet_layer failed\n");
    }
    free(dn_out);
    free(dn_state);
    free(dn_conv_state);

    cpu_rmsnorm(xn, x, m, w->layer[0].ffn_norm, QW3_N_EMBD);
    if (!cpu_matvec(m, w->layer[0].ffn_gate_inp, xn, router)) {
        fprintf(fp, "probe: router matvec needs dequant for tensor type %s\n",
                tensor_type_name(w->layer[0].ffn_gate_inp->type));
        free(sparse_out);
        free(shared_out);
        free(sd);
        free(sh);
        free(su);
        free(sg);
        free(router);
        free(xn);
        free(x);
        return;
    }

    int ids[QW3_N_EXPERT_USED];
    float vals[QW3_N_EXPERT_USED];
    topk_desc(router, QW3_N_EXPERT, QW3_N_EXPERT_USED, ids, vals);
    fprintf(fp, "layer0 router top%d:", QW3_N_EXPERT_USED);
    for (int i = 0; i < QW3_N_EXPERT_USED; i++) {
        fprintf(fp, " %d(%.7g)", ids[i], vals[i]);
    }
    fprintf(fp, "\n");

    if (!cpu_matvec(m, w->layer[0].ffn_gate_shared, xn, sg) ||
        !cpu_matvec(m, w->layer[0].ffn_up_shared, xn, su)) {
        fprintf(fp, "probe: shared expert gate/up matvec needs dequant\n");
        free(sparse_out);
        free(shared_out);
        free(sd);
        free(sh);
        free(su);
        free(sg);
        free(router);
        free(xn);
        free(x);
        return;
    }
    for (int i = 0; i < QW3_N_FF_SHARED; i++) {
        sh[i] = cpu_silu(sg[i]) * su[i];
    }
    if (!cpu_matvec(m, w->layer[0].ffn_down_shared, sh, sd)) {
        fprintf(fp, "probe: shared expert down matvec needs dequant\n");
        free(sparse_out);
        free(shared_out);
        free(sd);
        free(sh);
        free(su);
        free(sg);
        free(router);
        free(xn);
        free(x);
        return;
    }
    float shared_gate0 = 1.0f;
    if (w->layer[0].ffn_gate_inp_shexp) {
        if (!cpu_dot_dense_1d(m, w->layer[0].ffn_gate_inp_shexp, xn,
                              &shared_gate0)) {
            fprintf(fp, "probe: shared expert scalar gate failed\n");
            free(sparse_out);
            free(shared_out);
            free(sd);
            free(sh);
            free(su);
            free(sg);
            free(router);
            free(xn);
            free(x);
            return;
        }
        shared_gate0 = 1.0f / (1.0f + expf(-shared_gate0));
        for (int i = 0; i < QW3_N_EMBD; i++) sd[i] *= shared_gate0;
    }
    double smean = 0.0;
    double srms = 0.0;
    for (int i = 0; i < QW3_N_EMBD; i++) {
        smean += sd[i];
        srms += (double)sd[i] * (double)sd[i];
    }
    fprintf(fp, "layer0 shared expert: gate=%.7g mean=%.7g rms=%.7g first=[%.7g %.7g %.7g %.7g]\n",
            shared_gate0,
            smean / QW3_N_EMBD, sqrt(srms / QW3_N_EMBD),
            sd[0], sd[1], sd[2], sd[3]);
    memcpy(shared_out, sd, (size_t)QW3_N_EMBD * sizeof(float));

    if (cpu_matvec_iq4_xs_expert(m, w->layer[0].ffn_down_exps, ids[0], sh, sd)) {
        smean = 0.0;
        srms = 0.0;
        for (int i = 0; i < QW3_N_EMBD; i++) {
            smean += sd[i];
            srms += (double)sd[i] * (double)sd[i];
        }
        fprintf(fp, "layer0 iq4_xs down expert %d probe: mean=%.7g rms=%.7g first=[%.7g %.7g %.7g %.7g]\n",
                ids[0], smean / QW3_N_EMBD, sqrt(srms / QW3_N_EMBD),
                sd[0], sd[1], sd[2], sd[3]);
    } else {
        fprintf(fp, "probe: iq4_xs expert down matvec unavailable\n");
    }

    if (cpu_matvec_iq3_s_expert(m, w->layer[0].ffn_gate_exps, ids[0], xn, sg) &&
        cpu_matvec_iq3_s_expert(m, w->layer[0].ffn_up_exps, ids[0], xn, su)) {
        for (int i = 0; i < QW3_N_FF_EXP; i++) {
            sh[i] = cpu_silu(sg[i]) * su[i];
        }
        if (cpu_matvec_iq4_xs_expert(m, w->layer[0].ffn_down_exps, ids[0], sh, sd)) {
            smean = 0.0;
            srms = 0.0;
            for (int i = 0; i < QW3_N_EMBD; i++) {
                smean += sd[i];
                srms += (double)sd[i] * (double)sd[i];
            }
            fprintf(fp, "layer0 sparse expert %d: mean=%.7g rms=%.7g first=[%.7g %.7g %.7g %.7g]\n",
                    ids[0], smean / QW3_N_EMBD, sqrt(srms / QW3_N_EMBD),
                    sd[0], sd[1], sd[2], sd[3]);
        }
    } else {
        fprintf(fp, "probe: iq3_s expert gate/up matvec unavailable\n");
    }

    float max_route = vals[0];
    float route_sum = 0.0f;
    float route_w[QW3_N_EXPERT_USED];
    for (int k = 0; k < QW3_N_EXPERT_USED; k++) {
        route_w[k] = expf(vals[k] - max_route);
        route_sum += route_w[k];
    }
    for (int k = 0; k < QW3_N_EXPERT_USED; k++) {
        route_w[k] /= route_sum;
        if (!cpu_matvec_iq3_s_expert(m, w->layer[0].ffn_gate_exps, ids[k], xn, sg) ||
            !cpu_matvec_iq3_s_expert(m, w->layer[0].ffn_up_exps, ids[k], xn, su)) {
            continue;
        }
        for (int i = 0; i < QW3_N_FF_EXP; i++) {
            sh[i] = cpu_silu(sg[i]) * su[i];
        }
        if (!cpu_matvec_iq4_xs_expert(m, w->layer[0].ffn_down_exps, ids[k], sh, sd)) {
            continue;
        }
        for (int i = 0; i < QW3_N_EMBD; i++) {
            sparse_out[i] += route_w[k] * sd[i];
        }
    }
    smean = 0.0;
    srms = 0.0;
    for (int i = 0; i < QW3_N_EMBD; i++) {
        smean += sparse_out[i];
        srms += (double)sparse_out[i] * (double)sparse_out[i];
    }
    fprintf(fp, "layer0 sparse top8: mean=%.7g rms=%.7g first=[%.7g %.7g %.7g %.7g]\n",
            smean / QW3_N_EMBD, sqrt(srms / QW3_N_EMBD),
            sparse_out[0], sparse_out[1], sparse_out[2], sparse_out[3]);

    smean = 0.0;
    srms = 0.0;
    for (int i = 0; i < QW3_N_EMBD; i++) {
        float v = shared_out[i] + sparse_out[i];
        smean += v;
        srms += (double)v * (double)v;
    }
    fprintf(fp, "layer0 moe total probe: mean=%.7g rms=%.7g first=[%.7g %.7g %.7g %.7g]\n",
            smean / QW3_N_EMBD, sqrt(srms / QW3_N_EMBD),
            shared_out[0] + sparse_out[0],
            shared_out[1] + sparse_out[1],
            shared_out[2] + sparse_out[2],
            shared_out[3] + sparse_out[3]);

    float *moe_out = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    int moe_ids[QW3_N_EXPERT_USED];
    float moe_scores[QW3_N_EXPERT_USED];
    if (cpu_moe_layer(e, 0, xn, moe_out, moe_ids, moe_scores)) {
        double diff = 0.0;
        smean = 0.0;
        srms = 0.0;
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float manual = shared_out[i] + sparse_out[i];
            float d = fabsf(moe_out[i] - manual);
            if (d > diff) diff = d;
            smean += moe_out[i];
            srms += (double)moe_out[i] * (double)moe_out[i];
        }
        fprintf(fp, "layer0 cpu_moe_layer: mean=%.7g rms=%.7g maxdiff=%.7g top0=%d\n",
                smean / QW3_N_EMBD, sqrt(srms / QW3_N_EMBD),
                diff, moe_ids[0]);
    } else {
        fprintf(fp, "probe: cpu_moe_layer failed\n");
    }
    free(moe_out);

    float *xa = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qfull = qw3_xmalloc((size_t)tensor_cols_qg() * sizeof(float));
    float *qnorm = qw3_xmalloc((size_t)(QW3_N_HEAD * QW3_N_HEAD_DIM) * sizeof(float));
    float *kproj = qw3_xmalloc((size_t)tensor_cols_kv() * sizeof(float));
    float *knorm = qw3_xmalloc((size_t)tensor_cols_kv() * sizeof(float));
    float *vproj = qw3_xmalloc((size_t)tensor_cols_kv() * sizeof(float));

    cpu_rmsnorm(xa, x, m, w->layer[3].attn_norm, QW3_N_EMBD);
    if (cpu_matvec(m, w->layer[3].attn_q_proj, xa, qfull) &&
        cpu_matvec(m, w->layer[3].attn_k_proj, xa, kproj) &&
        cpu_matvec(m, w->layer[3].attn_v_proj, xa, vproj)) {
        for (int h = 0; h < QW3_N_HEAD; h++) {
            cpu_rmsnorm(qnorm + h * QW3_N_HEAD_DIM,
                        qfull + h * QW3_N_HEAD_DIM * 2,
                        m, w->layer[3].attn_q_norm, QW3_N_HEAD_DIM);
        }
        for (int h = 0; h < QW3_N_HEAD_KV; h++) {
            cpu_rmsnorm(knorm + h * QW3_N_HEAD_DIM,
                        kproj + h * QW3_N_HEAD_DIM,
                        m, w->layer[3].attn_k_norm, QW3_N_HEAD_DIM);
        }
        double qrms = 0.0;
        double krms = 0.0;
        double vrms = 0.0;
        for (int i = 0; i < QW3_N_HEAD * QW3_N_HEAD_DIM; i++) {
            qrms += (double)qnorm[i] * (double)qnorm[i];
        }
        for (int i = 0; i < (int)tensor_cols_kv(); i++) {
            krms += (double)knorm[i] * (double)knorm[i];
            vrms += (double)vproj[i] * (double)vproj[i];
        }
        float gate0 = qfull[QW3_N_HEAD_DIM];
        float gate0_sigmoid = 1.0f / (1.0f + expf(-gate0));
        fprintf(fp, "layer3 gqa projection: q_rms=%.7g k_rms=%.7g v_rms=%.7g gate0_sigmoid=%.7g q0=[%.7g %.7g %.7g %.7g]\n",
                sqrt(qrms / (QW3_N_HEAD * QW3_N_HEAD_DIM)),
                sqrt(krms / (double)tensor_cols_kv()),
                sqrt(vrms / (double)tensor_cols_kv()),
                gate0_sigmoid, qnorm[0], qnorm[1], qnorm[2], qnorm[3]);

        cpu_rope_head(qnorm, 1);
        cpu_rope_head(knorm, 1);
        fprintf(fp, "layer3 rope probe pos=1: q0=[%.7g %.7g %.7g %.7g] k0=[%.7g %.7g %.7g %.7g]\n",
                qnorm[0], qnorm[1], qnorm[2], qnorm[3],
                knorm[0], knorm[1], knorm[2], knorm[3]);
    } else {
        fprintf(fp, "probe: layer3 gqa projection failed\n");
    }

    float *gqa_out = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    if (cpu_gqa_single_token_layer(e, 3, 1, x, gqa_out)) {
        double amean = 0.0;
        double arms = 0.0;
        for (int i = 0; i < QW3_N_EMBD; i++) {
            amean += gqa_out[i];
            arms += (double)gqa_out[i] * (double)gqa_out[i];
        }
        fprintf(fp, "layer3 gqa single-token: mean=%.7g rms=%.7g first=[%.7g %.7g %.7g %.7g]\n",
                amean / QW3_N_EMBD, sqrt(arms / QW3_N_EMBD),
                gqa_out[0], gqa_out[1], gqa_out[2], gqa_out[3]);
    } else {
        fprintf(fp, "probe: layer3 gqa single-token failed\n");
    }
    free(gqa_out);

    float *x2 = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *q1 = qw3_xmalloc((size_t)(QW3_N_HEAD * QW3_N_HEAD_DIM) * sizeof(float));
    float *q2 = qw3_xmalloc((size_t)(QW3_N_HEAD * QW3_N_HEAD_DIM) * sizeof(float));
    float *g1 = qw3_xmalloc((size_t)(QW3_N_HEAD * QW3_N_HEAD_DIM) * sizeof(float));
    float *g2 = qw3_xmalloc((size_t)(QW3_N_HEAD * QW3_N_HEAD_DIM) * sizeof(float));
    float *k1 = qw3_xmalloc((size_t)tensor_cols_kv() * sizeof(float));
    float *k2 = qw3_xmalloc((size_t)tensor_cols_kv() * sizeof(float));
    float *v1 = qw3_xmalloc((size_t)tensor_cols_kv() * sizeof(float));
    float *v2 = qw3_xmalloc((size_t)tensor_cols_kv() * sizeof(float));
    float *kcache2 = qw3_xmalloc((size_t)(2 * tensor_cols_kv()) * sizeof(float));
    float *vcache2 = qw3_xmalloc((size_t)(2 * tensor_cols_kv()) * sizeof(float));
    float *inner2 = qw3_xmalloc((size_t)tensor_linear_inner() * sizeof(float));
    float *out2 = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    const int token2 = 1814; /* " world" in the local qwen35 tokenizer. */
    if (tensor_read_dense_row(m, emb, token2, x2) &&
        cpu_gqa_project_token(e, 3, 0, x,  q1, k1, v1, g1) &&
        cpu_gqa_project_token(e, 3, 1, x2, q2, k2, v2, g2)) {
        memcpy(kcache2, k1, (size_t)tensor_cols_kv() * sizeof(float));
        memcpy(kcache2 + tensor_cols_kv(), k2,
               (size_t)tensor_cols_kv() * sizeof(float));
        memcpy(vcache2, v1, (size_t)tensor_cols_kv() * sizeof(float));
        memcpy(vcache2 + tensor_cols_kv(), v2,
               (size_t)tensor_cols_kv() * sizeof(float));
        cpu_gqa_attend_inner(q2, g2, kcache2, vcache2, 2, inner2);
        if (cpu_matvec(m, w->layer[3].attn_o_proj, inner2, out2)) {
            double amean = 0.0;
            double arms = 0.0;
            for (int i = 0; i < QW3_N_EMBD; i++) {
                amean += out2[i];
                arms += (double)out2[i] * (double)out2[i];
            }
            fprintf(fp, "layer3 gqa two-token cache: mean=%.7g rms=%.7g first=[%.7g %.7g %.7g %.7g]\n",
                    amean / QW3_N_EMBD, sqrt(arms / QW3_N_EMBD),
                    out2[0], out2[1], out2[2], out2[3]);
        }
    } else {
        fprintf(fp, "probe: layer3 gqa two-token cache failed\n");
    }

    float *resid3 = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *ffn3_in = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *moe3 = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *full3 = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *full3_k = qw3_xcalloc((size_t)tensor_cols_kv(), sizeof(float));
    float *full3_v = qw3_xcalloc((size_t)tensor_cols_kv(), sizeof(float));
    int moe3_ids[QW3_N_EXPERT_USED];
    float moe3_scores[QW3_N_EXPERT_USED];
    if (cpu_gqa_single_token_layer(e, 3, 1, x, out2)) {
        for (int i = 0; i < QW3_N_EMBD; i++) resid3[i] = x[i] + out2[i];
        cpu_rmsnorm(ffn3_in, resid3, m, w->layer[3].ffn_norm, QW3_N_EMBD);
        if (cpu_moe_layer(e, 3, ffn3_in, moe3, moe3_ids, moe3_scores)) {
            double lmean = 0.0;
            double lrms = 0.0;
            for (int i = 0; i < QW3_N_EMBD; i++) {
                float v = resid3[i] + moe3[i];
                lmean += v;
                lrms += (double)v * (double)v;
            }
            fprintf(fp, "layer3 isolated full layer: mean=%.7g rms=%.7g top0=%d first=[%.7g %.7g %.7g %.7g]\n",
                    lmean / QW3_N_EMBD, sqrt(lrms / QW3_N_EMBD), moe3_ids[0],
                    resid3[0] + moe3[0], resid3[1] + moe3[1],
                    resid3[2] + moe3[2], resid3[3] + moe3[3]);
        }
    } else {
        fprintf(fp, "probe: layer3 isolated full layer failed\n");
    }
    if (cpu_full_attention_layer(e, 3, 0, x, full3_k, full3_v, 1, full3)) {
        double lmean = 0.0;
        double lrms = 0.0;
        for (int i = 0; i < QW3_N_EMBD; i++) {
            lmean += full3[i];
            lrms += (double)full3[i] * (double)full3[i];
        }
        fprintf(fp, "layer3 cpu_full_attention_layer: mean=%.7g rms=%.7g first=[%.7g %.7g %.7g %.7g]\n",
                lmean / QW3_N_EMBD, sqrt(lrms / QW3_N_EMBD),
                full3[0], full3[1], full3[2], full3[3]);
    } else {
        fprintf(fp, "probe: cpu_full_attention_layer failed\n");
    }
    free(full3_v);
    free(full3_k);
    free(full3);
    free(moe3);
    free(ffn3_in);
    free(resid3);

    free(out2);
    free(inner2);
    free(vcache2);
    free(kcache2);
    free(v2);
    free(v1);
    free(k2);
    free(k1);
    free(g2);
    free(g1);
    free(q2);
    free(q1);
    free(x2);

    free(vproj);
    free(knorm);
    free(kproj);
    free(qnorm);
    free(qfull);
    free(xa);

    free(sparse_out);
    free(shared_out);
    free(sd);
    free(sh);
    free(su);
    free(sg);
    free(router);
    free(xn);
    free(x);
}

int qw3_session_create(qw3_session **out, qw3_engine *e, int ctx_size) {
    if (!out || !e || ctx_size <= 0) return -1;
    qw3_session *s = qw3_xcalloc(1, sizeof(*s));
    s->engine = e;
    s->ctx_size = ctx_size;
    s->valid = true;

    s->kv.ctx_size = ctx_size;
    s->kv.pos = 0;
    s->kv.n_full_layers = 0;
    s->dn.n_linear_layers = 0;
    for (int il = 0; il < QW3_N_LAYER; il++) {
        if (qw3_layer_is_full_attention((uint32_t)il)) {
            s->kv.full_layer_map[il] = s->kv.n_full_layers++;
            s->dn.linear_layer_map[il] = -1;
        } else {
            s->dn.linear_layer_map[il] = s->dn.n_linear_layers++;
            s->kv.full_layer_map[il] = -1;
        }
    }

    const uint64_t kv_floats = (uint64_t)QW3_N_FULL_ATTN_LAYERS *
                               (uint64_t)ctx_size *
                               QW3_N_HEAD_KV * QW3_N_HEAD_DIM;
    s->kv.k_cache = qw3_xcalloc((size_t)kv_floats, sizeof(float));
    s->kv.v_cache = qw3_xcalloc((size_t)kv_floats, sizeof(float));

    const uint64_t dn_floats = (uint64_t)QW3_N_LINEAR_LAYERS *
                               QW3_N_LINEAR_V_HEADS *
                               QW3_N_LINEAR_HEAD_DIM *
                               QW3_N_LINEAR_HEAD_DIM;
    const uint64_t conv_floats = (uint64_t)QW3_N_LINEAR_LAYERS *
                                 tensor_linear_qkv() *
                                 (QW3_N_LINEAR_CONV_K - 1);
    s->dn.state = qw3_xcalloc((size_t)dn_floats, sizeof(float));
    s->dn.conv_state = qw3_xcalloc((size_t)conv_floats, sizeof(float));
    s->logits = qw3_xcalloc(QW3_N_VOCAB, sizeof(float));

#ifndef QW3_NO_METAL
    if (e->backend == QW3_BACKEND_METAL) {
        if (!e->metal_ready) {
            qw3_session_free(s);
            return -1;
        }
        s->metal_n_gpu_layers = qw3_metal_env_n_gpu_layers();
        s->metal = qw3_metal_session_create((uint32_t)ctx_size,
                                            (uint32_t)QW3_N_VOCAB);
        if (!s->metal) {
            qw3_session_free(s);
            return -1;
        }
    }
#endif

    *out = s;
    return 0;
}

void qw3_session_free(qw3_session *s) {
    if (!s) return;
#ifndef QW3_NO_METAL
    qw3_metal_session_free(s->metal);
#endif
    free(s->kv.k_cache);
    free(s->kv.v_cache);
    free(s->dn.state);
    free(s->dn.conv_state);
    qw3_tokens_free(&s->tokens);
    free(s->logits);
    free(s);
}

void qw3_session_set_progress(qw3_session *s, qw3_session_progress_fn fn,
                              void *ud) {
    if (!s) return;
    s->progress_fn = fn;
    s->progress_ud = ud;
}

static void qw3_session_progress(qw3_session *s, const char *event,
                                 int current, int total) {
    if (s && s->progress_fn) {
        s->progress_fn(s->progress_ud, event, current, total);
    }
}

int qw3_session_common_prefix(qw3_session *s, const qw3_tokens *prompt) {
    if (!s || !prompt) return 0;
    int n = s->tokens.len < prompt->len ? s->tokens.len : prompt->len;
    int i = 0;
    while (i < n && s->tokens.v[i] == prompt->v[i]) i++;
    return i;
}

void qw3_session_invalidate(qw3_session *s) {
    if (!s) return;
    s->valid = false;
    s->kv.pos = 0;
    qw3_tokens_free(&s->tokens);
    memset(s->logits, 0, (size_t)QW3_N_VOCAB * sizeof(float));
    if (s->kv.k_cache) {
        const uint64_t kv_floats = (uint64_t)QW3_N_FULL_ATTN_LAYERS *
                                   (uint64_t)s->ctx_size *
                                   QW3_N_HEAD_KV * QW3_N_HEAD_DIM;
        memset(s->kv.k_cache, 0, (size_t)kv_floats * sizeof(float));
        memset(s->kv.v_cache, 0, (size_t)kv_floats * sizeof(float));
    }
    if (s->dn.state) {
        const uint64_t dn_floats = (uint64_t)QW3_N_LINEAR_LAYERS *
                                   QW3_N_LINEAR_V_HEADS *
                                   QW3_N_LINEAR_HEAD_DIM *
                                   QW3_N_LINEAR_HEAD_DIM;
        memset(s->dn.state, 0, (size_t)dn_floats * sizeof(float));
    }
    if (s->dn.conv_state) {
        const uint64_t conv_floats = (uint64_t)QW3_N_LINEAR_LAYERS *
                                     tensor_linear_qkv() *
                                     (QW3_N_LINEAR_CONV_K - 1);
        memset(s->dn.conv_state, 0, (size_t)conv_floats * sizeof(float));
    }
#ifndef QW3_NO_METAL
    if (s->metal) {
        (void)qw3_metal_session_clear(s->metal);
    }
#endif
}

void qw3_session_rewind(qw3_session *s, int pos) {
    if (!s) return;
    if (pos < 0) pos = 0;
    if (pos > s->tokens.len) pos = s->tokens.len;
    s->tokens.len = pos;
    s->kv.pos = pos;
    s->valid = false;
}

int qw3_session_pos(qw3_session *s) {
    return s ? s->tokens.len : 0;
}

int qw3_session_ctx(qw3_session *s) {
    return s ? s->ctx_size : 0;
}

const qw3_tokens *qw3_session_tokens(qw3_session *s) {
    return s ? &s->tokens : NULL;
}

static void trace_vec(FILE *fp, const char *name, int il, const float *x, int n) {
    if (!fp || !x || n <= 0) return;
    double mean = 0.0;
    double ss = 0.0;
    for (int i = 0; i < n; i++) {
        mean += x[i];
        ss += (double)x[i] * (double)x[i];
    }
    fprintf(fp, "%s %02d: mean=%.7g rms=%.7g first=[%.7g %.7g %.7g %.7g]\n",
            name, il, mean / n, sqrt(ss / n), x[0], x[1], x[2], x[3]);
}

static void trace_vec_json(FILE *fp, bool *first_event, const char *name,
                           int il, const float *x, int n) {
    if (!fp || !first_event || !x || n <= 0) return;
    double mean = 0.0;
    double ss = 0.0;
    for (int i = 0; i < n; i++) {
        mean += x[i];
        ss += (double)x[i] * (double)x[i];
    }
    if (!*first_event) fprintf(fp, ",\n");
    *first_event = false;
    fprintf(fp,
            "    {\"name\":\"%s\",\"layer\":%d,\"mean\":%.9g,\"rms\":%.9g,\"first\":[",
            name, il, mean / n, sqrt(ss / n));
    int nf = n < 8 ? n : 8;
    for (int i = 0; i < nf; i++) {
        if (i) fputc(',', fp);
        fprintf(fp, "%.9g", x[i]);
    }
    fprintf(fp, "]}");
}

static void trace_emit(FILE *fp, bool json, bool *first_event,
                       const char *name, int il, const float *x, int n) {
    if (json) {
        trace_vec_json(fp, first_event, name, il, x, n);
    } else {
        trace_vec(fp, name, il, x, n);
    }
}

static int qw3_session_eval_inner(qw3_session *s, int token,
                                  char *err, size_t errlen, FILE *trace,
                                  bool trace_json, bool *trace_first_event) {
    if (!s) return -1;
    if (token < 0 || token >= QW3_N_VOCAB) {
        if (err && errlen) snprintf(err, errlen, "token %d is outside vocab", token);
        return -1;
    }
    if (s->kv.pos >= s->ctx_size) {
        if (err && errlen) snprintf(err, errlen, "context is full");
        return -1;
    }

    qw3_engine *e = s->engine;
    const qw3_model *m = &e->model;
    const qw3_weights *w = &e->weights;
    float *x0 = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *x1 = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *norm = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));

    bool ok = tensor_read_dense_row(m, w->token_embd, (uint64_t)token, x0);
    if (!ok) {
        if (err && errlen) {
            snprintf(err, errlen, "embedding read failed for tensor type %s",
                     tensor_type_name(w->token_embd->type));
        }
        goto done;
    }
    trace_emit(trace, trace_json, trace_first_event,
               "embedding", -1, x0, QW3_N_EMBD);

    for (int il = 0; il < QW3_N_LAYER; il++) {
        if (s->progress_fn) s->progress_fn(s->progress_ud, "layer", il, QW3_N_LAYER);
        if (qw3_layer_is_full_attention((uint32_t)il)) {
            int fl = s->kv.full_layer_map[il];
            float *k_cache = s->kv.k_cache +
                             (uint64_t)fl * (uint64_t)s->ctx_size *
                             tensor_cols_kv();
            float *v_cache = s->kv.v_cache +
                             (uint64_t)fl * (uint64_t)s->ctx_size *
                             tensor_cols_kv();
            ok = cpu_full_attention_layer(e, il, s->kv.pos, x0,
                                          k_cache, v_cache, s->ctx_size, x1);
        } else {
            int dl = s->dn.linear_layer_map[il];
            float *conv_state = s->dn.conv_state +
                                (uint64_t)dl * tensor_linear_qkv() *
                                (QW3_N_LINEAR_CONV_K - 1);
            float *state = s->dn.state +
                           (uint64_t)dl * QW3_N_LINEAR_V_HEADS *
                           QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM;
            ok = cpu_deltanet_layer(e, il, x0, conv_state, state, x1);
        }
        if (!ok) {
            if (err && errlen) snprintf(err, errlen, "layer %d forward failed", il);
            goto done;
        }
        trace_emit(trace, trace_json, trace_first_event,
                   qw3_layer_is_full_attention((uint32_t)il) ? "full" : "linear",
                   il, x1, QW3_N_EMBD);
        float *tmp = x0;
        x0 = x1;
        x1 = tmp;
    }

    cpu_rmsnorm(norm, x0, m, w->output_norm, QW3_N_EMBD);
    ok = cpu_matvec(m, w->output, norm, s->logits);
    if (!ok) {
        if (err && errlen) {
            snprintf(err, errlen, "output projection failed for tensor type %s",
                     tensor_type_name(w->output->type));
        }
        goto done;
    }

    token_vec_push(&s->tokens, token);
    s->kv.pos++;
    s->valid = true;
    if (s->progress_fn) s->progress_fn(s->progress_ud, "layer", QW3_N_LAYER, QW3_N_LAYER);

done:
    free(norm);
    free(x1);
    free(x0);
    return ok ? 0 : -1;
}

int qw3_session_eval(qw3_session *s, int token, char *err, size_t errlen) {
#ifndef QW3_NO_METAL
    if (s && s->engine && s->engine->backend == QW3_BACKEND_METAL && s->metal) {
        return qw3_metal_session_eval_token_slow_ex(s, token, err, errlen, 1);
    }
#endif
    return qw3_session_eval_inner(s, token, err, errlen, NULL, false, NULL);
}

int qw3_session_sync(qw3_session *s, const qw3_tokens *prompt,
                     char *err, size_t errlen) {
    if (!s || !prompt) return -1;
    int common = qw3_session_common_prefix(s, prompt);
    if (common != s->tokens.len) {
        qw3_session_invalidate(s);
        common = 0;
    }
    const int total_prefill = prompt->len - common;
    int done_prefill = 0;
    if (total_prefill > 0) {
        qw3_session_progress(s, "prefill_chunk", 0, total_prefill);
    }
#ifndef QW3_NO_METAL
    if (s->engine && s->engine->backend == QW3_BACKEND_METAL && s->metal) {
        const int prefill_batch = qw3_session_uses_partial_metal(s) ?
            1 : qw3_metal_prefill_batch_size();
        if (prefill_batch > 1) {
            for (int i = common; i < prompt->len;) {
                int n = prompt->len - i;
                if (n > prefill_batch) n = prefill_batch;
                const int last = i + n == prompt->len;
                int rc = 0;
                if (n > 1) {
                    rc = qw3_metal_session_eval_prefill_batch_mode(
                        s, prompt->v + i, n, err, errlen,
                        last ? QW3_METAL_LOGITS_READ :
                               QW3_METAL_LOGITS_DEFER);
                } else {
                    rc = last ?
                        qw3_metal_session_eval_token_slow_ex(
                            s, prompt->v[i], err, errlen, 1) :
                        qw3_metal_session_eval_token_defer_logits(
                            s, prompt->v[i], err, errlen);
                }
                if (rc != 0) return -1;
                i += n;
                done_prefill += n;
                qw3_session_progress(s, "prefill_chunk",
                                     done_prefill, total_prefill);
            }
        } else {
            const int defer_interval = qw3_prefill_defer_interval();
            int deferred = 0;
            for (int i = common; i < prompt->len; i++) {
                const int last = i + 1 == prompt->len;
                int rc = last ?
                    qw3_metal_session_eval_token_slow_ex(
                        s, prompt->v[i], err, errlen, 1) :
                    qw3_metal_session_eval_token_defer_logits(
                        s, prompt->v[i], err, errlen);
                if (rc != 0) return -1;
                if (!last && ++deferred >= defer_interval) {
                    if (!qw3_metal_synchronize()) {
                        if (err && errlen) snprintf(err, errlen,
                                                    "Metal deferred prefill failed");
                        return -1;
                    }
                    deferred = 0;
                }
                done_prefill++;
                qw3_session_progress(s, "prefill_chunk",
                                     done_prefill, total_prefill);
            }
        }
        return 0;
    }
#endif
    for (int i = common; i < prompt->len; i++) {
        if (qw3_session_eval(s, prompt->v[i], err, errlen) != 0) return -1;
        done_prefill++;
        qw3_session_progress(s, "prefill_chunk", done_prefill, total_prefill);
    }
    return 0;
}

int qw3_engine_trace_prompt(qw3_engine *e, const qw3_tokens *prompt,
                            int ctx_size, FILE *fp) {
    if (!e || !prompt || !fp || ctx_size <= 0) return -1;
    qw3_session *s = NULL;
    char err[256] = {0};
    if (qw3_session_create(&s, e, ctx_size) != 0) return -1;

    fprintf(fp, "trace: tokens=%d ctx=%d\n", prompt->len, ctx_size);
    for (int i = 0; i < prompt->len; i++) {
        fprintf(fp, "trace token %d/%d id=%d\n", i + 1, prompt->len, prompt->v[i]);
        if (qw3_session_eval_inner(s, prompt->v[i], err, sizeof(err),
                                   i == prompt->len - 1 ? fp : NULL,
                                   false, NULL) != 0) {
            fprintf(fp, "trace failed: %s\n", err);
            qw3_session_free(s);
            return -1;
        }
    }

    qw3_token_score scores[8];
    int n = qw3_session_top_logprobs(s, scores, 8);
    fprintf(fp, "trace top%d:", n);
    for (int i = 0; i < n; i++) {
        fprintf(fp, " %d(%.7g)", scores[i].id, scores[i].logit);
    }
    fprintf(fp, "\n");
    qw3_session_free(s);
    return 0;
}

int qw3_engine_trace_prompt_json(qw3_engine *e, const qw3_tokens *prompt,
                                 int ctx_size, FILE *fp) {
    if (!e || !prompt || !fp || ctx_size <= 0) return -1;
    qw3_session *s = NULL;
    char err[256] = {0};
    if (qw3_session_create(&s, e, ctx_size) != 0) return -1;

    fprintf(fp, "{\n");
    fprintf(fp, "  \"schema\":\"qw3-local-trace-v1\",\n");
    fprintf(fp, "  \"ctx_size\":%d,\n", ctx_size);
    fprintf(fp, "  \"prompt_tokens\":[");
    for (int i = 0; i < prompt->len; i++) {
        if (i) fputc(',', fp);
        fprintf(fp, "%d", prompt->v[i]);
    }
    fprintf(fp, "],\n");
    fprintf(fp, "  \"traced_token_index\":%d,\n", prompt->len - 1);
    fprintf(fp, "  \"traced_token_id\":%d,\n", prompt->len > 0 ? prompt->v[prompt->len - 1] : -1);
    fprintf(fp, "  \"events\":[\n");

    bool first_event = true;
    for (int i = 0; i < prompt->len; i++) {
        FILE *trace = i == prompt->len - 1 ? fp : NULL;
        if (qw3_session_eval_inner(s, prompt->v[i], err, sizeof(err),
                                   trace, true, &first_event) != 0) {
            fprintf(fp, "\n  ],\n  \"error\":\"%s\"\n}\n", err);
            qw3_session_free(s);
            return -1;
        }
    }

    fprintf(fp, "\n  ],\n  \"top_logits\":[");
    qw3_token_score scores[8];
    int n = qw3_session_top_logprobs(s, scores, 8);
    for (int i = 0; i < n; i++) {
        if (i) fputc(',', fp);
        fprintf(fp, "{\"id\":%d,\"logit\":%.9g}", scores[i].id, scores[i].logit);
    }
    fprintf(fp, "]\n}\n");
    qw3_session_free(s);
    return 0;
}

int qw3_engine_metal_rmsnorm_test(qw3_engine *e, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e;
    fprintf(fp, "metal rmsnorm: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal rmsnorm: Metal backend is not initialized\n");
        return -1;
    }
    enum { N = 2048 };
    float *x = qw3_xmalloc((size_t)N * sizeof(float));
    float *cpu = qw3_xmalloc((size_t)N * sizeof(float));
    float *gpu = qw3_xmalloc((size_t)N * sizeof(float));
    double ss = 0.0;
    for (int i = 0; i < N; i++) {
        x[i] = sinf((float)i * 0.013f) * 0.25f + cosf((float)i * 0.017f) * 0.125f;
        ss += (double)x[i] * (double)x[i];
    }
    float scale = 1.0f / sqrtf((float)(ss / (double)N) + QW3_RMS_EPS);
    for (int i = 0; i < N; i++) cpu[i] = x[i] * scale;
    int ok = qw3_metal_rmsnorm_plain(x, gpu, N, QW3_RMS_EPS);
    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    if (ok) {
        for (int i = 0; i < N; i++) {
            float d = fabsf(cpu[i] - gpu[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * (double)d;
        }
        rmsdiff = sqrt(rmsdiff / (double)N);
    }
    fprintf(fp,
            "metal rmsnorm: %s n=%d maxdiff=%.7g rmsdiff=%.7g first=[%.7g %.7g %.7g %.7g]\n",
            ok ? "ok" : "failed", N, maxdiff, rmsdiff,
            gpu[0], gpu[1], gpu[2], gpu[3]);
    free(gpu);
    free(cpu);
    free(x);
    return ok ? 0 : -1;
#endif
}

int qw3_engine_metal_embed_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal embed: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal embed: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal embed: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_tensor *emb = e->weights.token_embd;
    if (emb->type != QW3_TENSOR_Q8_0 || emb->dim[0] != QW3_N_EMBD) {
        fprintf(fp, "metal embed: expected q8_0 embedding tensor\n");
        return -1;
    }
    float *cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    bool cpu_ok = tensor_read_dense_row(&e->model, emb, (uint64_t)token, cpu);
    int gpu_ok = cpu_ok && qw3_metal_embed_q8_0(emb->offset, (uint32_t)token,
                                                QW3_N_EMBD, gpu);
    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu[i] - gpu[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * (double)d;
        }
        rmsdiff = sqrt(rmsdiff / (double)QW3_N_EMBD);
    }
    fprintf(fp,
            "metal embed: %s token=%d n=%d maxdiff=%.7g rmsdiff=%.7g first=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, QW3_N_EMBD,
            maxdiff, rmsdiff, gpu[0], gpu[1], gpu[2], gpu[3]);
    free(gpu);
    free(cpu);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_rmsnorm_weight_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal weighted rmsnorm: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal weighted rmsnorm: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal weighted rmsnorm: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_tensor *emb = e->weights.token_embd;
    const qw3_tensor *weight = e->weights.layer[0].attn_norm;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    bool ok = tensor_read_dense_row(&e->model, emb, (uint64_t)token, x);
    if (ok) {
        cpu_rmsnorm(cpu, x, &e->model, weight, QW3_N_EMBD);
    }
    int gpu_ok = ok && qw3_metal_rmsnorm_weight_f32(x, weight->offset,
                                                    gpu, QW3_N_EMBD, QW3_RMS_EPS);
    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu[i] - gpu[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * (double)d;
        }
        rmsdiff = sqrt(rmsdiff / (double)QW3_N_EMBD);
    }
    fprintf(fp,
            "metal weighted rmsnorm: %s token=%d n=%d maxdiff=%.7g rmsdiff=%.7g first=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, QW3_N_EMBD,
            maxdiff, rmsdiff, gpu[0], gpu[1], gpu[2], gpu[3]);
    free(gpu);
    free(cpu);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_matvec_q8_0_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal q8_0 matvec: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal q8_0 matvec: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal q8_0 matvec: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_tensor *emb = e->weights.token_embd;
    const qw3_tensor *norm_w = e->weights.layer[0].attn_norm;
    const qw3_tensor *proj = e->weights.layer[0].linear_qkv_proj;
    const uint32_t n_out = (uint32_t)tensor_linear_qkv();
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu = qw3_xmalloc((size_t)n_out * sizeof(float));
    float *gpu = qw3_xmalloc((size_t)n_out * sizeof(float));
    bool ok = tensor_read_dense_row(&e->model, emb, (uint64_t)token, x);
    if (ok) {
        cpu_rmsnorm(xn, x, &e->model, norm_w, QW3_N_EMBD);
        ok = cpu_matvec_q8_0(&e->model, proj, xn, cpu);
    }
    int gpu_ok = ok && qw3_metal_matvec_q8_0(proj->offset, xn, QW3_N_EMBD, n_out, gpu);
    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < n_out; i++) {
            float d = fabsf(cpu[i] - gpu[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * (double)d;
        }
        rmsdiff = sqrt(rmsdiff / (double)n_out);
    }
    fprintf(fp,
            "metal q8_0 matvec: %s token=%d n_in=%d n_out=%u maxdiff=%.7g rmsdiff=%.7g first=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, QW3_N_EMBD, n_out,
            maxdiff, rmsdiff, gpu[0], gpu[1], gpu[2], gpu[3]);
    free(gpu);
    free(cpu);
    free(xn);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_deltanet_proj_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal deltanet projection: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal deltanet projection: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal deltanet projection: token %d is outside vocab\n", token);
        return -1;
    }

    const qw3_layer_weights *lw = &e->weights.layer[0];
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t n_z = (uint32_t)tensor_linear_inner();
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *gpu_qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *cpu_z = qw3_xmalloc((size_t)n_z * sizeof(float));
    float *gpu_z = qw3_xmalloc((size_t)n_z * sizeof(float));

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd, (uint64_t)token, x);
    if (ok) {
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        ok = cpu_matvec_q8_0(&e->model, lw->linear_qkv_proj, xn, cpu_qkv) &&
             cpu_matvec_q8_0(&e->model, lw->linear_gate_proj, xn, cpu_z);
    }
    int gpu_ok = ok &&
        qw3_metal_matvec_q8_0(lw->linear_qkv_proj->offset, xn, QW3_N_EMBD, n_qkv, gpu_qkv) &&
        qw3_metal_matvec_q8_0(lw->linear_gate_proj->offset, xn, QW3_N_EMBD, n_z, gpu_z);

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    uint64_t ndiff = 0;
    double qrms = 0.0, krms = 0.0, vrms = 0.0, zrms = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < n_qkv; i++) {
            float d = fabsf(cpu_qkv[i] - gpu_qkv[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * (double)d;
            ndiff++;
        }
        for (uint32_t i = 0; i < n_z; i++) {
            float d = fabsf(cpu_z[i] - gpu_z[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * (double)d;
            ndiff++;
        }
        const float *q = gpu_qkv;
        const float *k = q + QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
        const float *v = k + QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
        for (int i = 0; i < QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM; i++) {
            qrms += (double)q[i] * (double)q[i];
            krms += (double)k[i] * (double)k[i];
        }
        for (int i = 0; i < QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM; i++) {
            vrms += (double)v[i] * (double)v[i];
            zrms += (double)gpu_z[i] * (double)gpu_z[i];
        }
        rmsdiff = sqrt(rmsdiff / (double)ndiff);
        qrms = sqrt(qrms / (QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM));
        krms = sqrt(krms / (QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM));
        vrms = sqrt(vrms / (QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM));
        zrms = sqrt(zrms / (QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM));
    }
    fprintf(fp,
            "metal deltanet projection: %s token=%d maxdiff=%.7g rmsdiff=%.7g q_rms=%.7g k_rms=%.7g v_rms=%.7g z_rms=%.7g q0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, maxdiff, rmsdiff,
            qrms, krms, vrms, zrms,
            gpu_qkv[0], gpu_qkv[1], gpu_qkv[2], gpu_qkv[3]);

    free(gpu_z);
    free(cpu_z);
    free(gpu_qkv);
    free(cpu_qkv);
    free(xn);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_deltanet_conv_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal deltanet conv1d: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal deltanet conv1d: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal deltanet conv1d: token %d is outside vocab\n", token);
        return -1;
    }

    const qw3_layer_weights *lw = &e->weights.layer[0];
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *gpu_qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *cpu_conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *gpu_conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *conv_state = qw3_xcalloc((size_t)n_qkv * (QW3_N_LINEAR_CONV_K - 1),
                                    sizeof(float));

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd, (uint64_t)token, x);
    if (ok) {
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        ok = cpu_matvec_q8_0(&e->model, lw->linear_qkv_proj, xn, cpu_qkv) &&
             cpu_deltanet_conv1d_step(&e->model, lw, cpu_qkv, conv_state, cpu_conv);
    }
    int gpu_ok = ok &&
        qw3_metal_matvec_q8_0(lw->linear_qkv_proj->offset, xn, QW3_N_EMBD, n_qkv, gpu_qkv) &&
        qw3_metal_deltanet_conv1d_zero(lw->linear_conv_weight->offset, gpu_qkv, n_qkv, gpu_conv);

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    double qrms = 0.0, krms = 0.0, vrms = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < n_qkv; i++) {
            float d = fabsf(cpu_conv[i] - gpu_conv[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * (double)d;
        }
        const float *q = gpu_conv;
        const float *k = q + QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
        const float *v = k + QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
        for (int i = 0; i < QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM; i++) {
            qrms += (double)q[i] * (double)q[i];
            krms += (double)k[i] * (double)k[i];
        }
        for (int i = 0; i < QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM; i++) {
            vrms += (double)v[i] * (double)v[i];
        }
        rmsdiff = sqrt(rmsdiff / (double)n_qkv);
        qrms = sqrt(qrms / (QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM));
        krms = sqrt(krms / (QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM));
        vrms = sqrt(vrms / (QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM));
    }
    fprintf(fp,
            "metal deltanet conv1d: %s token=%d maxdiff=%.7g rmsdiff=%.7g q_rms=%.7g k_rms=%.7g v_rms=%.7g q0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, maxdiff, rmsdiff,
            qrms, krms, vrms,
            gpu_conv[0], gpu_conv[1], gpu_conv[2], gpu_conv[3]);

    free(conv_state);
    free(gpu_conv);
    free(cpu_conv);
    free(gpu_qkv);
    free(cpu_qkv);
    free(xn);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_deltanet_conv_step_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal deltanet conv1d step: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal deltanet conv1d step: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal deltanet conv1d step: token %d is outside vocab\n", token);
        return -1;
    }

    const qw3_layer_weights *lw = &e->weights.layer[0];
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t state_n = n_qkv * (QW3_N_LINEAR_CONV_K - 1);
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *cpu_state = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *gpu_state = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *cpu_state_out = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *gpu_state_out = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *cpu_conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *gpu_conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));

    for (uint32_t i = 0; i < state_n; i++) {
        int r = (int)((i * 19u + 7u) % 101u) - 50;
        cpu_state[i] = (float)r * 0.0009f;
        gpu_state[i] = cpu_state[i];
    }
    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd, (uint64_t)token, x);
    if (ok) {
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        ok = cpu_matvec_q8_0(&e->model, lw->linear_qkv_proj, xn, qkv);
    }
    if (ok) {
        memcpy(cpu_state_out, cpu_state, (size_t)state_n * sizeof(float));
        ok = cpu_deltanet_conv1d_step(&e->model, lw, qkv, cpu_state_out, cpu_conv);
    }
    int gpu_ok = ok &&
        qw3_metal_deltanet_conv1d_step(lw->linear_conv_weight->offset,
                                       qkv, gpu_state, n_qkv,
                                       gpu_conv, gpu_state_out);

    float conv_max = 0.0f, state_max = 0.0f;
    double conv_rmsdiff = 0.0, state_rmsdiff = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < n_qkv; i++) {
            float d = fabsf(cpu_conv[i] - gpu_conv[i]);
            if (d > conv_max) conv_max = d;
            conv_rmsdiff += (double)d * d;
        }
        for (uint32_t i = 0; i < state_n; i++) {
            float d = fabsf(cpu_state_out[i] - gpu_state_out[i]);
            if (d > state_max) state_max = d;
            state_rmsdiff += (double)d * d;
        }
        conv_rmsdiff = sqrt(conv_rmsdiff / n_qkv);
        state_rmsdiff = sqrt(state_rmsdiff / state_n);
    }
    fprintf(fp,
            "metal deltanet conv1d step: %s token=%d conv_max=%.7g conv_rmsdiff=%.7g state_max=%.7g state_rmsdiff=%.7g conv0=[%.7g %.7g %.7g %.7g] state0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token,
            conv_max, conv_rmsdiff, state_max, state_rmsdiff,
            gpu_conv[0], gpu_conv[1], gpu_conv[2], gpu_conv[3],
            gpu_state_out[0], gpu_state_out[1], gpu_state_out[2], gpu_state_out[3]);

    free(gpu_conv);
    free(cpu_conv);
    free(gpu_state_out);
    free(cpu_state_out);
    free(gpu_state);
    free(cpu_state);
    free(qkv);
    free(xn);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_deltanet_l2norm_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal deltanet l2norm: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal deltanet l2norm: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal deltanet l2norm: token %d is outside vocab\n", token);
        return -1;
    }

    const qw3_layer_weights *lw = &e->weights.layer[0];
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *cpu_q = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *cpu_k = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *gpu_q = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *gpu_k = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *conv_state = qw3_xcalloc((size_t)n_qkv * (QW3_N_LINEAR_CONV_K - 1),
                                    sizeof(float));

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd, (uint64_t)token, x);
    if (ok) {
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        ok = cpu_matvec_q8_0(&e->model, lw->linear_qkv_proj, xn, qkv) &&
             cpu_deltanet_conv1d_step(&e->model, lw, qkv, conv_state, conv);
    }
    if (ok) {
        const float *qraw = conv;
        const float *kraw = conv + qk_n;
        for (int h = 0; h < QW3_N_LINEAR_QK_HEADS; h++) {
            cpu_l2_norm_head(cpu_q + h * QW3_N_LINEAR_HEAD_DIM,
                             qraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
            cpu_l2_norm_head(cpu_k + h * QW3_N_LINEAR_HEAD_DIM,
                             kraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
        }
    }
    int gpu_ok = ok &&
        qw3_metal_l2norm_heads(conv, QW3_N_LINEAR_QK_HEADS,
                               QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS, gpu_q) &&
        qw3_metal_l2norm_heads(conv + qk_n, QW3_N_LINEAR_QK_HEADS,
                               QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS, gpu_k);

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    double qnorm0 = 0.0, knorm0 = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < qk_n; i++) {
            float dq = fabsf(cpu_q[i] - gpu_q[i]);
            float dk = fabsf(cpu_k[i] - gpu_k[i]);
            if (dq > maxdiff) maxdiff = dq;
            if (dk > maxdiff) maxdiff = dk;
            rmsdiff += (double)dq * (double)dq + (double)dk * (double)dk;
        }
        for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
            qnorm0 += (double)gpu_q[i] * (double)gpu_q[i];
            knorm0 += (double)gpu_k[i] * (double)gpu_k[i];
        }
        rmsdiff = sqrt(rmsdiff / (double)(2 * qk_n));
        qnorm0 = sqrt(qnorm0);
        knorm0 = sqrt(knorm0);
    }
    fprintf(fp,
            "metal deltanet l2norm: %s token=%d maxdiff=%.7g rmsdiff=%.7g q0_norm=%.7g k0_norm=%.7g q0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, maxdiff, rmsdiff,
            qnorm0, knorm0, gpu_q[0], gpu_q[1], gpu_q[2], gpu_q[3]);

    free(conv_state);
    free(gpu_k);
    free(gpu_q);
    free(cpu_k);
    free(cpu_q);
    free(conv);
    free(qkv);
    free(xn);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_deltanet_gates_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal deltanet gates: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal deltanet gates: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal deltanet gates: token %d is outside vocab\n", token);
        return -1;
    }

    const qw3_layer_weights *lw = &e->weights.layer[0];
    const uint32_t n_gates = QW3_N_LINEAR_V_HEADS;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_alpha = qw3_xmalloc((size_t)n_gates * sizeof(float));
    float *cpu_beta = qw3_xmalloc((size_t)n_gates * sizeof(float));
    float *gpu_alpha = qw3_xmalloc((size_t)n_gates * sizeof(float));
    float *gpu_beta = qw3_xmalloc((size_t)n_gates * sizeof(float));

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd, (uint64_t)token, x);
    if (ok) {
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        ok = cpu_matvec_dense(&e->model, lw->linear_ssm_alpha, xn, cpu_alpha) &&
             cpu_matvec_dense(&e->model, lw->linear_ssm_beta, xn, cpu_beta);
    }
    int gpu_ok = ok &&
        qw3_metal_matvec_f32(lw->linear_ssm_alpha->offset, xn, QW3_N_EMBD,
                             n_gates, gpu_alpha) &&
        qw3_metal_matvec_f32(lw->linear_ssm_beta->offset, xn, QW3_N_EMBD,
                             n_gates, gpu_beta);

    float raw_maxdiff = 0.0f;
    float gate_maxdiff = 0.0f;
    double raw_rmsdiff = 0.0;
    double gate_rmsdiff = 0.0;
    float beta_sig0 = 0.0f;
    float dt_a0 = 0.0f;
    if (gpu_ok) {
        for (uint32_t i = 0; i < n_gates; i++) {
            float da = fabsf(cpu_alpha[i] - gpu_alpha[i]);
            float db = fabsf(cpu_beta[i] - gpu_beta[i]);
            if (da > raw_maxdiff) raw_maxdiff = da;
            if (db > raw_maxdiff) raw_maxdiff = db;
            raw_rmsdiff += (double)da * (double)da + (double)db * (double)db;

            const float cb = 1.0f / (1.0f + expf(-cpu_beta[i]));
            const float gb = 1.0f / (1.0f + expf(-gpu_beta[i]));
            const float cdt = cpu_softplus(cpu_alpha[i] +
                tensor_read_dense_1d(&e->model, lw->linear_ssm_dt_bias, (uint64_t)i)) *
                tensor_read_dense_1d(&e->model, lw->linear_ssm_a, (uint64_t)i);
            const float gdt = cpu_softplus(gpu_alpha[i] +
                tensor_read_dense_1d(&e->model, lw->linear_ssm_dt_bias, (uint64_t)i)) *
                tensor_read_dense_1d(&e->model, lw->linear_ssm_a, (uint64_t)i);
            float dgb = fabsf(cb - gb);
            float ddt = fabsf(cdt - gdt);
            if (dgb > gate_maxdiff) gate_maxdiff = dgb;
            if (ddt > gate_maxdiff) gate_maxdiff = ddt;
            gate_rmsdiff += (double)dgb * (double)dgb + (double)ddt * (double)ddt;
        }
        raw_rmsdiff = sqrt(raw_rmsdiff / (double)(2 * n_gates));
        gate_rmsdiff = sqrt(gate_rmsdiff / (double)(2 * n_gates));
        beta_sig0 = 1.0f / (1.0f + expf(-gpu_beta[0]));
        dt_a0 = cpu_softplus(gpu_alpha[0] +
            tensor_read_dense_1d(&e->model, lw->linear_ssm_dt_bias, 0)) *
            tensor_read_dense_1d(&e->model, lw->linear_ssm_a, 0);
    }
    fprintf(fp,
            "metal deltanet gates: %s token=%d raw_maxdiff=%.7g raw_rmsdiff=%.7g gate_maxdiff=%.7g gate_rmsdiff=%.7g beta_sig0=%.7g dt_a0=%.7g alpha0=[%.7g %.7g %.7g %.7g] beta0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token,
            raw_maxdiff, raw_rmsdiff, gate_maxdiff, gate_rmsdiff,
            beta_sig0, dt_a0,
            gpu_alpha[0], gpu_alpha[1], gpu_alpha[2], gpu_alpha[3],
            gpu_beta[0], gpu_beta[1], gpu_beta[2], gpu_beta[3]);

    free(gpu_beta);
    free(gpu_alpha);
    free(cpu_beta);
    free(cpu_alpha);
    free(xn);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_deltanet_recur_zero_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal deltanet recur zero: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal deltanet recur zero: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal deltanet recur zero: token %d is outside vocab\n", token);
        return -1;
    }

    const qw3_layer_weights *lw = &e->weights.layer[0];
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t core_n = QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t state_n = QW3_N_LINEAR_V_HEADS *
                             QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *qnorm = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *knorm = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *alpha = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta_sig = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *cpu_state = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *gpu_state = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *cpu_core = qw3_xmalloc((size_t)core_n * sizeof(float));
    float *gpu_core = qw3_xmalloc((size_t)core_n * sizeof(float));
    float *conv_state = qw3_xcalloc((size_t)n_qkv * (QW3_N_LINEAR_CONV_K - 1),
                                    sizeof(float));

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd, (uint64_t)token, x);
    if (ok) {
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        ok = cpu_matvec_q8_0(&e->model, lw->linear_qkv_proj, xn, qkv) &&
             cpu_matvec_dense(&e->model, lw->linear_ssm_alpha, xn, alpha) &&
             cpu_matvec_dense(&e->model, lw->linear_ssm_beta, xn, beta) &&
             cpu_deltanet_conv1d_step(&e->model, lw, qkv, conv_state, conv);
    }
    if (ok) {
        const float *qraw = conv;
        const float *kraw = conv + qk_n;
        for (int h = 0; h < QW3_N_LINEAR_QK_HEADS; h++) {
            cpu_l2_norm_head(qnorm + h * QW3_N_LINEAR_HEAD_DIM,
                             qraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
            cpu_l2_norm_head(knorm + h * QW3_N_LINEAR_HEAD_DIM,
                             kraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            const int hk = hv % QW3_N_LINEAR_QK_HEADS;
            const float *qh = qnorm + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *kh = knorm + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *vh = conv + qk_n * 2 + hv * QW3_N_LINEAR_HEAD_DIM;
            float *sh = cpu_state + (uint64_t)hv * QW3_N_LINEAR_HEAD_DIM *
                        QW3_N_LINEAR_HEAD_DIM;
            float *oh = cpu_core + hv * QW3_N_LINEAR_HEAD_DIM;
            const float bh = 1.0f / (1.0f + expf(-beta[hv]));
            beta_sig[hv] = bh;
            float dot = 0.0f;
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) dot += qh[i] * kh[i];
            const float scale = dot / sqrtf((float)QW3_N_LINEAR_HEAD_DIM) * bh;
            for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
                oh[j] = scale * vh[j];
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    sh[i * QW3_N_LINEAR_HEAD_DIM + j] = kh[i] * bh * vh[j];
                }
            }
        }
    }
    int gpu_ok = ok &&
        qw3_metal_deltanet_recur_zero(qnorm, knorm, conv + qk_n * 2, beta_sig,
                                      QW3_N_LINEAR_QK_HEADS,
                                      QW3_N_LINEAR_V_HEADS,
                                      QW3_N_LINEAR_HEAD_DIM,
                                      gpu_state, gpu_core);

    float core_maxdiff = 0.0f;
    float state_maxdiff = 0.0f;
    double core_rmsdiff = 0.0;
    double state_rmsdiff = 0.0;
    double core_rms = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < core_n; i++) {
            float d = fabsf(cpu_core[i] - gpu_core[i]);
            if (d > core_maxdiff) core_maxdiff = d;
            core_rmsdiff += (double)d * (double)d;
            core_rms += (double)gpu_core[i] * (double)gpu_core[i];
        }
        for (uint32_t i = 0; i < state_n; i++) {
            float d = fabsf(cpu_state[i] - gpu_state[i]);
            if (d > state_maxdiff) state_maxdiff = d;
            state_rmsdiff += (double)d * (double)d;
        }
        core_rmsdiff = sqrt(core_rmsdiff / (double)core_n);
        state_rmsdiff = sqrt(state_rmsdiff / (double)state_n);
        core_rms = sqrt(core_rms / (double)core_n);
    }
    fprintf(fp,
            "metal deltanet recur zero: %s token=%d core_maxdiff=%.7g core_rmsdiff=%.7g state_maxdiff=%.7g state_rmsdiff=%.7g core_rms=%.7g core0=[%.7g %.7g %.7g %.7g] state0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token,
            core_maxdiff, core_rmsdiff, state_maxdiff, state_rmsdiff, core_rms,
            gpu_core[0], gpu_core[1], gpu_core[2], gpu_core[3],
            gpu_state[0], gpu_state[1], gpu_state[2], gpu_state[3]);

    free(conv_state);
    free(gpu_core);
    free(cpu_core);
    free(gpu_state);
    free(cpu_state);
    free(beta_sig);
    free(beta);
    free(alpha);
    free(knorm);
    free(qnorm);
    free(conv);
    free(qkv);
    free(xn);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_deltanet_recur_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal deltanet recur: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal deltanet recur: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal deltanet recur: token %d is outside vocab\n", token);
        return -1;
    }

    const qw3_layer_weights *lw = &e->weights.layer[0];
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t core_n = QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t state_n = QW3_N_LINEAR_V_HEADS *
                             QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *qnorm = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *knorm = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *alpha = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta_sig = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *gamma = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *state_in = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *cpu_state = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *gpu_state = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *cpu_core = qw3_xmalloc((size_t)core_n * sizeof(float));
    float *gpu_core = qw3_xmalloc((size_t)core_n * sizeof(float));
    float *conv_state = qw3_xcalloc((size_t)n_qkv * (QW3_N_LINEAR_CONV_K - 1),
                                    sizeof(float));

    for (uint32_t i = 0; i < state_n; i++) {
        int r = (int)((i * 17u + 13u) % 97u) - 48;
        state_in[i] = (float)r * 0.0007f;
    }

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd, (uint64_t)token, x);
    if (ok) {
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        ok = cpu_matvec_q8_0(&e->model, lw->linear_qkv_proj, xn, qkv) &&
             cpu_matvec_dense(&e->model, lw->linear_ssm_alpha, xn, alpha) &&
             cpu_matvec_dense(&e->model, lw->linear_ssm_beta, xn, beta) &&
             cpu_deltanet_conv1d_step(&e->model, lw, qkv, conv_state, conv);
    }
    if (ok) {
        const float *qraw = conv;
        const float *kraw = conv + qk_n;
        for (int h = 0; h < QW3_N_LINEAR_QK_HEADS; h++) {
            cpu_l2_norm_head(qnorm + h * QW3_N_LINEAR_HEAD_DIM,
                             qraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
            cpu_l2_norm_head(knorm + h * QW3_N_LINEAR_HEAD_DIM,
                             kraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            const int hk = hv % QW3_N_LINEAR_QK_HEADS;
            const float *qh = qnorm + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *kh = knorm + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *vh = conv + qk_n * 2 + hv * QW3_N_LINEAR_HEAD_DIM;
            const float bh = 1.0f / (1.0f + expf(-beta[hv]));
            const float ah = cpu_softplus(alpha[hv] +
                tensor_read_dense_1d(&e->model, lw->linear_ssm_dt_bias, (uint64_t)hv));
            const float gh = expf(ah *
                tensor_read_dense_1d(&e->model, lw->linear_ssm_a, (uint64_t)hv));
            beta_sig[hv] = bh;
            gamma[hv] = gh;
            float *sh_out = cpu_state + (uint64_t)hv * QW3_N_LINEAR_HEAD_DIM *
                            QW3_N_LINEAR_HEAD_DIM;
            const float *sh_in = state_in + (uint64_t)hv * QW3_N_LINEAR_HEAD_DIM *
                                 QW3_N_LINEAR_HEAD_DIM;
            float *oh = cpu_core + hv * QW3_N_LINEAR_HEAD_DIM;
            for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
                float sk = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    sk += sh_in[i * QW3_N_LINEAR_HEAD_DIM + j] * gh * kh[i];
                }
                const float d = bh * (vh[j] - sk);
                float out = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    const int idx = i * QW3_N_LINEAR_HEAD_DIM + j;
                    const float sv = sh_in[idx] * gh + kh[i] * d;
                    sh_out[idx] = sv;
                    out += sv * qh[i];
                }
                oh[j] = out / sqrtf((float)QW3_N_LINEAR_HEAD_DIM);
            }
        }
    }
    int gpu_ok = ok &&
        qw3_metal_deltanet_recur(state_in, qnorm, knorm, conv + qk_n * 2,
                                 beta_sig, gamma,
                                 QW3_N_LINEAR_QK_HEADS,
                                 QW3_N_LINEAR_V_HEADS,
                                 QW3_N_LINEAR_HEAD_DIM,
                                 gpu_state, gpu_core);

    float core_maxdiff = 0.0f;
    float state_maxdiff = 0.0f;
    double core_rmsdiff = 0.0;
    double state_rmsdiff = 0.0;
    double core_rms = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < core_n; i++) {
            float d = fabsf(cpu_core[i] - gpu_core[i]);
            if (d > core_maxdiff) core_maxdiff = d;
            core_rmsdiff += (double)d * (double)d;
            core_rms += (double)gpu_core[i] * (double)gpu_core[i];
        }
        for (uint32_t i = 0; i < state_n; i++) {
            float d = fabsf(cpu_state[i] - gpu_state[i]);
            if (d > state_maxdiff) state_maxdiff = d;
            state_rmsdiff += (double)d * (double)d;
        }
        core_rmsdiff = sqrt(core_rmsdiff / (double)core_n);
        state_rmsdiff = sqrt(state_rmsdiff / (double)state_n);
        core_rms = sqrt(core_rms / (double)core_n);
    }
    fprintf(fp,
            "metal deltanet recur: %s token=%d core_maxdiff=%.7g core_rmsdiff=%.7g state_maxdiff=%.7g state_rmsdiff=%.7g core_rms=%.7g gamma0=%.7g core0=[%.7g %.7g %.7g %.7g] state0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token,
            core_maxdiff, core_rmsdiff, state_maxdiff, state_rmsdiff, core_rms,
            gamma[0],
            gpu_core[0], gpu_core[1], gpu_core[2], gpu_core[3],
            gpu_state[0], gpu_state[1], gpu_state[2], gpu_state[3]);

    free(conv_state);
    free(gpu_core);
    free(cpu_core);
    free(gpu_state);
    free(cpu_state);
    free(state_in);
    free(gamma);
    free(beta_sig);
    free(beta);
    free(alpha);
    free(knorm);
    free(qnorm);
    free(conv);
    free(qkv);
    free(xn);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_deltanet_recur_step_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal deltanet recur step: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal deltanet recur step: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal deltanet recur step: token %d is outside vocab\n", token);
        return -1;
    }

    const qw3_layer_weights *lw = &e->weights.layer[0];
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t core_n = QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t conv_state_n = n_qkv * (QW3_N_LINEAR_CONV_K - 1);
    const uint32_t state_n = QW3_N_LINEAR_V_HEADS *
                             QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *alpha = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta_sig = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *gamma = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *conv_state = qw3_xmalloc((size_t)conv_state_n * sizeof(float));
    float *conv_state_cpu = qw3_xmalloc((size_t)conv_state_n * sizeof(float));
    float *conv_state_gpu = qw3_xmalloc((size_t)conv_state_n * sizeof(float));
    float *conv_cpu = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *conv_gpu = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *q_cpu = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *k_cpu = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *q_gpu = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *k_gpu = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *state_in = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *cpu_state = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *gpu_state = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *cpu_core = qw3_xmalloc((size_t)core_n * sizeof(float));
    float *gpu_core = qw3_xmalloc((size_t)core_n * sizeof(float));

    for (uint32_t i = 0; i < conv_state_n; i++) {
        int r = (int)((i * 23u + 11u) % 103u) - 51;
        conv_state[i] = (float)r * 0.0008f;
    }
    for (uint32_t i = 0; i < state_n; i++) {
        int r = (int)((i * 17u + 13u) % 97u) - 48;
        state_in[i] = (float)r * 0.0007f;
    }

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd, (uint64_t)token, x);
    if (ok) {
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        ok = cpu_matvec_q8_0(&e->model, lw->linear_qkv_proj, xn, qkv) &&
             cpu_matvec_dense(&e->model, lw->linear_ssm_alpha, xn, alpha) &&
             cpu_matvec_dense(&e->model, lw->linear_ssm_beta, xn, beta);
    }
    if (ok) {
        memcpy(conv_state_cpu, conv_state, (size_t)conv_state_n * sizeof(float));
        ok = cpu_deltanet_conv1d_step(&e->model, lw, qkv, conv_state_cpu, conv_cpu);
    }
    if (ok) {
        for (int h = 0; h < QW3_N_LINEAR_QK_HEADS; h++) {
            cpu_l2_norm_head(q_cpu + h * QW3_N_LINEAR_HEAD_DIM,
                             conv_cpu + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
            cpu_l2_norm_head(k_cpu + h * QW3_N_LINEAR_HEAD_DIM,
                             conv_cpu + qk_n + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            const int hk = hv % QW3_N_LINEAR_QK_HEADS;
            const float *qh = q_cpu + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *kh = k_cpu + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *vh = conv_cpu + qk_n * 2 + hv * QW3_N_LINEAR_HEAD_DIM;
            const float bh = 1.0f / (1.0f + expf(-beta[hv]));
            const float ah = cpu_softplus(alpha[hv] +
                tensor_read_dense_1d(&e->model, lw->linear_ssm_dt_bias, (uint64_t)hv));
            const float gh = expf(ah *
                tensor_read_dense_1d(&e->model, lw->linear_ssm_a, (uint64_t)hv));
            beta_sig[hv] = bh;
            gamma[hv] = gh;
            float *sh_out = cpu_state + (uint64_t)hv * QW3_N_LINEAR_HEAD_DIM *
                            QW3_N_LINEAR_HEAD_DIM;
            const float *sh_in = state_in + (uint64_t)hv * QW3_N_LINEAR_HEAD_DIM *
                                 QW3_N_LINEAR_HEAD_DIM;
            float *oh = cpu_core + hv * QW3_N_LINEAR_HEAD_DIM;
            for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
                float sk = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) sk += sh_in[i * QW3_N_LINEAR_HEAD_DIM + j] * gh * kh[i];
                const float d = bh * (vh[j] - sk);
                float out = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    const int idx = i * QW3_N_LINEAR_HEAD_DIM + j;
                    const float sv = sh_in[idx] * gh + kh[i] * d;
                    sh_out[idx] = sv;
                    out += sv * qh[i];
                }
                oh[j] = out / sqrtf((float)QW3_N_LINEAR_HEAD_DIM);
            }
        }
    }
    int gpu_ok = ok &&
        qw3_metal_deltanet_conv1d_step(lw->linear_conv_weight->offset,
                                       qkv, conv_state, n_qkv,
                                       conv_gpu, conv_state_gpu) &&
        qw3_metal_l2norm_heads(conv_gpu, QW3_N_LINEAR_QK_HEADS,
                               QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS, q_gpu) &&
        qw3_metal_l2norm_heads(conv_gpu + qk_n, QW3_N_LINEAR_QK_HEADS,
                               QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS, k_gpu) &&
        qw3_metal_deltanet_recur(state_in, q_gpu, k_gpu, conv_gpu + qk_n * 2,
                                 beta_sig, gamma,
                                 QW3_N_LINEAR_QK_HEADS,
                                 QW3_N_LINEAR_V_HEADS,
                                 QW3_N_LINEAR_HEAD_DIM,
                                 gpu_state, gpu_core);

    float core_max = 0.0f, state_max = 0.0f, conv_max = 0.0f, conv_state_max = 0.0f;
    double core_rmsdiff = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < core_n; i++) {
            float d = fabsf(cpu_core[i] - gpu_core[i]);
            if (d > core_max) core_max = d;
            core_rmsdiff += (double)d * d;
        }
        for (uint32_t i = 0; i < state_n; i++) {
            float d = fabsf(cpu_state[i] - gpu_state[i]);
            if (d > state_max) state_max = d;
        }
        for (uint32_t i = 0; i < n_qkv; i++) {
            float d = fabsf(conv_cpu[i] - conv_gpu[i]);
            if (d > conv_max) conv_max = d;
        }
        for (uint32_t i = 0; i < conv_state_n; i++) {
            float d = fabsf(conv_state_cpu[i] - conv_state_gpu[i]);
            if (d > conv_state_max) conv_state_max = d;
        }
        core_rmsdiff = sqrt(core_rmsdiff / core_n);
    }
    fprintf(fp,
            "metal deltanet recur step: %s token=%d conv_max=%.7g conv_state_max=%.7g core_max=%.7g core_rmsdiff=%.7g state_max=%.7g core0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token,
            conv_max, conv_state_max, core_max, core_rmsdiff, state_max,
            gpu_core[0], gpu_core[1], gpu_core[2], gpu_core[3]);

    free(gpu_core); free(cpu_core); free(gpu_state); free(cpu_state);
    free(state_in); free(k_gpu); free(q_gpu); free(k_cpu); free(q_cpu);
    free(conv_gpu); free(conv_cpu); free(conv_state_gpu); free(conv_state_cpu);
    free(conv_state); free(gamma); free(beta_sig); free(beta); free(alpha);
    free(qkv); free(xn); free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_deltanet_gated_norm_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal deltanet gated norm: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal deltanet gated norm: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal deltanet gated norm: token %d is outside vocab\n", token);
        return -1;
    }

    const qw3_layer_weights *lw = &e->weights.layer[0];
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t core_n = QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t state_n = QW3_N_LINEAR_V_HEADS *
                             QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *z = qw3_xmalloc((size_t)core_n * sizeof(float));
    float *conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *qnorm = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *knorm = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *alpha = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *state_in = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *state_out = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *core = qw3_xmalloc((size_t)core_n * sizeof(float));
    float *cpu_inner = qw3_xmalloc((size_t)core_n * sizeof(float));
    float *gpu_inner = qw3_xmalloc((size_t)core_n * sizeof(float));
    float *conv_state = qw3_xcalloc((size_t)n_qkv * (QW3_N_LINEAR_CONV_K - 1),
                                    sizeof(float));

    for (uint32_t i = 0; i < state_n; i++) {
        int r = (int)((i * 17u + 13u) % 97u) - 48;
        state_in[i] = (float)r * 0.0007f;
    }

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd, (uint64_t)token, x);
    if (ok) {
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        ok = cpu_matvec_q8_0(&e->model, lw->linear_qkv_proj, xn, qkv) &&
             cpu_matvec_q8_0(&e->model, lw->linear_gate_proj, xn, z) &&
             cpu_matvec_dense(&e->model, lw->linear_ssm_alpha, xn, alpha) &&
             cpu_matvec_dense(&e->model, lw->linear_ssm_beta, xn, beta) &&
             cpu_deltanet_conv1d_step(&e->model, lw, qkv, conv_state, conv);
    }
    if (ok) {
        const float *qraw = conv;
        const float *kraw = conv + qk_n;
        for (int h = 0; h < QW3_N_LINEAR_QK_HEADS; h++) {
            cpu_l2_norm_head(qnorm + h * QW3_N_LINEAR_HEAD_DIM,
                             qraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
            cpu_l2_norm_head(knorm + h * QW3_N_LINEAR_HEAD_DIM,
                             kraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            const int hk = hv % QW3_N_LINEAR_QK_HEADS;
            const float *qh = qnorm + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *kh = knorm + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *vh = conv + qk_n * 2 + hv * QW3_N_LINEAR_HEAD_DIM;
            const float bh = 1.0f / (1.0f + expf(-beta[hv]));
            const float ah = cpu_softplus(alpha[hv] +
                tensor_read_dense_1d(&e->model, lw->linear_ssm_dt_bias, (uint64_t)hv));
            const float gh = expf(ah *
                tensor_read_dense_1d(&e->model, lw->linear_ssm_a, (uint64_t)hv));
            float *sh_out = state_out + (uint64_t)hv * QW3_N_LINEAR_HEAD_DIM *
                            QW3_N_LINEAR_HEAD_DIM;
            const float *sh_in = state_in + (uint64_t)hv * QW3_N_LINEAR_HEAD_DIM *
                                 QW3_N_LINEAR_HEAD_DIM;
            float *oh = core + hv * QW3_N_LINEAR_HEAD_DIM;
            for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
                float sk = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    sk += sh_in[i * QW3_N_LINEAR_HEAD_DIM + j] * gh * kh[i];
                }
                const float d = bh * (vh[j] - sk);
                float out = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    const int idx = i * QW3_N_LINEAR_HEAD_DIM + j;
                    const float sv = sh_in[idx] * gh + kh[i] * d;
                    sh_out[idx] = sv;
                    out += sv * qh[i];
                }
                oh[j] = out / sqrtf((float)QW3_N_LINEAR_HEAD_DIM);
            }
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            float *dst = cpu_inner + hv * QW3_N_LINEAR_HEAD_DIM;
            const float *src = core + hv * QW3_N_LINEAR_HEAD_DIM;
            const float *zh = z + hv * QW3_N_LINEAR_HEAD_DIM;
            double ss = 0.0;
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                ss += (double)src[i] * (double)src[i];
            }
            const float scale = 1.0f /
                sqrtf((float)(ss / QW3_N_LINEAR_HEAD_DIM) + QW3_RMS_EPS);
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                dst[i] = src[i] * scale *
                         tensor_read_dense_1d(&e->model, lw->linear_ssm_norm, (uint64_t)i) *
                         cpu_silu(zh[i]);
            }
        }
    }
    int gpu_ok = ok &&
        qw3_metal_deltanet_gated_rmsnorm(lw->linear_ssm_norm->offset,
                                         core, z,
                                         QW3_N_LINEAR_V_HEADS,
                                         QW3_N_LINEAR_HEAD_DIM,
                                         QW3_RMS_EPS, gpu_inner);

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    double inner_rms = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < core_n; i++) {
            float d = fabsf(cpu_inner[i] - gpu_inner[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * (double)d;
            inner_rms += (double)gpu_inner[i] * (double)gpu_inner[i];
        }
        rmsdiff = sqrt(rmsdiff / (double)core_n);
        inner_rms = sqrt(inner_rms / (double)core_n);
    }
    fprintf(fp,
            "metal deltanet gated norm: %s token=%d maxdiff=%.7g rmsdiff=%.7g inner_rms=%.7g inner0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, maxdiff, rmsdiff, inner_rms,
            gpu_inner[0], gpu_inner[1], gpu_inner[2], gpu_inner[3]);

    free(conv_state);
    free(gpu_inner);
    free(cpu_inner);
    free(core);
    free(state_out);
    free(state_in);
    free(beta);
    free(alpha);
    free(knorm);
    free(qnorm);
    free(conv);
    free(z);
    free(qkv);
    free(xn);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_deltanet_out_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal deltanet out: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal deltanet out: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal deltanet out: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_layer_weights *lw = &e->weights.layer[0];
    if (lw->linear_ssm_out->type != QW3_TENSOR_Q8_0) {
        fprintf(fp, "metal deltanet out: expected q8_0 ssm_out, got %s\n",
                tensor_type_name(lw->linear_ssm_out->type));
        return -1;
    }

    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t inner_n = QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t state_n = QW3_N_LINEAR_V_HEADS *
                             QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *z = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *qnorm = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *knorm = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *alpha = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *state_in = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *state_out = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *core = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *cpu_inner = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *gpu_inner = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *cpu_attn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_attn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *conv_state = qw3_xcalloc((size_t)n_qkv * (QW3_N_LINEAR_CONV_K - 1),
                                    sizeof(float));

    for (uint32_t i = 0; i < state_n; i++) {
        int r = (int)((i * 17u + 13u) % 97u) - 48;
        state_in[i] = (float)r * 0.0007f;
    }

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd, (uint64_t)token, x);
    if (ok) {
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        ok = cpu_matvec_q8_0(&e->model, lw->linear_qkv_proj, xn, qkv) &&
             cpu_matvec_q8_0(&e->model, lw->linear_gate_proj, xn, z) &&
             cpu_matvec_dense(&e->model, lw->linear_ssm_alpha, xn, alpha) &&
             cpu_matvec_dense(&e->model, lw->linear_ssm_beta, xn, beta) &&
             cpu_deltanet_conv1d_step(&e->model, lw, qkv, conv_state, conv);
    }
    if (ok) {
        const float *qraw = conv;
        const float *kraw = conv + qk_n;
        for (int h = 0; h < QW3_N_LINEAR_QK_HEADS; h++) {
            cpu_l2_norm_head(qnorm + h * QW3_N_LINEAR_HEAD_DIM,
                             qraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
            cpu_l2_norm_head(knorm + h * QW3_N_LINEAR_HEAD_DIM,
                             kraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            const int hk = hv % QW3_N_LINEAR_QK_HEADS;
            const float *qh = qnorm + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *kh = knorm + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *vh = conv + qk_n * 2 + hv * QW3_N_LINEAR_HEAD_DIM;
            const float bh = 1.0f / (1.0f + expf(-beta[hv]));
            const float ah = cpu_softplus(alpha[hv] +
                tensor_read_dense_1d(&e->model, lw->linear_ssm_dt_bias, (uint64_t)hv));
            const float gh = expf(ah *
                tensor_read_dense_1d(&e->model, lw->linear_ssm_a, (uint64_t)hv));
            float *sh_out = state_out + (uint64_t)hv * QW3_N_LINEAR_HEAD_DIM *
                            QW3_N_LINEAR_HEAD_DIM;
            const float *sh_in = state_in + (uint64_t)hv * QW3_N_LINEAR_HEAD_DIM *
                                 QW3_N_LINEAR_HEAD_DIM;
            float *oh = core + hv * QW3_N_LINEAR_HEAD_DIM;
            for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
                float sk = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    sk += sh_in[i * QW3_N_LINEAR_HEAD_DIM + j] * gh * kh[i];
                }
                const float d = bh * (vh[j] - sk);
                float out = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    const int idx = i * QW3_N_LINEAR_HEAD_DIM + j;
                    const float sv = sh_in[idx] * gh + kh[i] * d;
                    sh_out[idx] = sv;
                    out += sv * qh[i];
                }
                oh[j] = out / sqrtf((float)QW3_N_LINEAR_HEAD_DIM);
            }
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            float *dst = cpu_inner + hv * QW3_N_LINEAR_HEAD_DIM;
            const float *src = core + hv * QW3_N_LINEAR_HEAD_DIM;
            const float *zh = z + hv * QW3_N_LINEAR_HEAD_DIM;
            double ss = 0.0;
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                ss += (double)src[i] * (double)src[i];
            }
            const float scale = 1.0f /
                sqrtf((float)(ss / QW3_N_LINEAR_HEAD_DIM) + QW3_RMS_EPS);
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                dst[i] = src[i] * scale *
                         tensor_read_dense_1d(&e->model, lw->linear_ssm_norm, (uint64_t)i) *
                         cpu_silu(zh[i]);
            }
        }
        ok = cpu_matvec_q8_0(&e->model, lw->linear_ssm_out, cpu_inner, cpu_attn);
    }

    int gpu_ok = ok &&
        qw3_metal_deltanet_gated_rmsnorm(lw->linear_ssm_norm->offset,
                                         core, z,
                                         QW3_N_LINEAR_V_HEADS,
                                         QW3_N_LINEAR_HEAD_DIM,
                                         QW3_RMS_EPS, gpu_inner) &&
        qw3_metal_matvec_q8_0(lw->linear_ssm_out->offset, gpu_inner,
                              inner_n, QW3_N_EMBD, gpu_attn);

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    double attn_rms = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu_attn[i] - gpu_attn[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * (double)d;
            attn_rms += (double)gpu_attn[i] * (double)gpu_attn[i];
        }
        rmsdiff = sqrt(rmsdiff / (double)QW3_N_EMBD);
        attn_rms = sqrt(attn_rms / (double)QW3_N_EMBD);
    }
    fprintf(fp,
            "metal deltanet out: %s token=%d maxdiff=%.7g rmsdiff=%.7g attn_rms=%.7g attn0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, maxdiff, rmsdiff, attn_rms,
            gpu_attn[0], gpu_attn[1], gpu_attn[2], gpu_attn[3]);

    free(conv_state);
    free(gpu_attn);
    free(cpu_attn);
    free(gpu_inner);
    free(cpu_inner);
    free(core);
    free(state_out);
    free(state_in);
    free(beta);
    free(alpha);
    free(knorm);
    free(qnorm);
    free(conv);
    free(z);
    free(qkv);
    free(xn);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_deltanet_branch_step_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal deltanet branch step: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal deltanet branch step: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal deltanet branch step: token %d is outside vocab\n", token);
        return -1;
    }

    const qw3_layer_weights *lw = &e->weights.layer[0];
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t inner_n = QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t conv_state_n = n_qkv * (QW3_N_LINEAR_CONV_K - 1);
    const uint32_t state_n = QW3_N_LINEAR_V_HEADS *
                             QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *z = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *alpha = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta_sig = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *gamma = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *conv_state = qw3_xmalloc((size_t)conv_state_n * sizeof(float));
    float *conv_state_cpu = qw3_xmalloc((size_t)conv_state_n * sizeof(float));
    float *conv_state_gpu = qw3_xmalloc((size_t)conv_state_n * sizeof(float));
    float *conv_cpu = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *conv_gpu = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *q_cpu = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *k_cpu = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *q_gpu = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *k_gpu = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *state_in = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *cpu_state = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *gpu_state = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *cpu_core = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *gpu_core = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *cpu_inner = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *gpu_inner = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *cpu_attn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_attn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));

    for (uint32_t i = 0; i < conv_state_n; i++) {
        int r = (int)((i * 23u + 11u) % 103u) - 51;
        conv_state[i] = (float)r * 0.0008f;
    }
    for (uint32_t i = 0; i < state_n; i++) {
        int r = (int)((i * 17u + 13u) % 97u) - 48;
        state_in[i] = (float)r * 0.0007f;
    }

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd, (uint64_t)token, x);
    if (ok) {
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        ok = cpu_matvec_q8_0(&e->model, lw->linear_qkv_proj, xn, qkv) &&
             cpu_matvec_q8_0(&e->model, lw->linear_gate_proj, xn, z) &&
             cpu_matvec_dense(&e->model, lw->linear_ssm_alpha, xn, alpha) &&
             cpu_matvec_dense(&e->model, lw->linear_ssm_beta, xn, beta);
    }
    if (ok) {
        memcpy(conv_state_cpu, conv_state, (size_t)conv_state_n * sizeof(float));
        ok = cpu_deltanet_conv1d_step(&e->model, lw, qkv, conv_state_cpu, conv_cpu);
    }
    if (ok) {
        for (int h = 0; h < QW3_N_LINEAR_QK_HEADS; h++) {
            cpu_l2_norm_head(q_cpu + h * QW3_N_LINEAR_HEAD_DIM,
                             conv_cpu + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
            cpu_l2_norm_head(k_cpu + h * QW3_N_LINEAR_HEAD_DIM,
                             conv_cpu + qk_n + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            const int hk = hv % QW3_N_LINEAR_QK_HEADS;
            const float *qh = q_cpu + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *kh = k_cpu + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *vh = conv_cpu + qk_n * 2 + hv * QW3_N_LINEAR_HEAD_DIM;
            const float bh = 1.0f / (1.0f + expf(-beta[hv]));
            const float ah = cpu_softplus(alpha[hv] +
                tensor_read_dense_1d(&e->model, lw->linear_ssm_dt_bias, (uint64_t)hv));
            const float gh = expf(ah *
                tensor_read_dense_1d(&e->model, lw->linear_ssm_a, (uint64_t)hv));
            beta_sig[hv] = bh;
            gamma[hv] = gh;
            float *sh_out = cpu_state + (uint64_t)hv * QW3_N_LINEAR_HEAD_DIM *
                            QW3_N_LINEAR_HEAD_DIM;
            const float *sh_in = state_in + (uint64_t)hv * QW3_N_LINEAR_HEAD_DIM *
                                 QW3_N_LINEAR_HEAD_DIM;
            float *oh = cpu_core + hv * QW3_N_LINEAR_HEAD_DIM;
            for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
                float sk = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) sk += sh_in[i * QW3_N_LINEAR_HEAD_DIM + j] * gh * kh[i];
                const float d = bh * (vh[j] - sk);
                float out = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    const int idx = i * QW3_N_LINEAR_HEAD_DIM + j;
                    const float sv = sh_in[idx] * gh + kh[i] * d;
                    sh_out[idx] = sv;
                    out += sv * qh[i];
                }
                oh[j] = out / sqrtf((float)QW3_N_LINEAR_HEAD_DIM);
            }
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            float *dst = cpu_inner + hv * QW3_N_LINEAR_HEAD_DIM;
            const float *src = cpu_core + hv * QW3_N_LINEAR_HEAD_DIM;
            const float *zh = z + hv * QW3_N_LINEAR_HEAD_DIM;
            double ss = 0.0;
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) ss += (double)src[i] * src[i];
            const float scale = 1.0f / sqrtf((float)(ss / QW3_N_LINEAR_HEAD_DIM) + QW3_RMS_EPS);
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                dst[i] = src[i] * scale *
                         tensor_read_dense_1d(&e->model, lw->linear_ssm_norm, (uint64_t)i) *
                         cpu_silu(zh[i]);
            }
        }
        ok = cpu_matvec_q8_0(&e->model, lw->linear_ssm_out, cpu_inner, cpu_attn);
    }

    int gpu_ok = ok &&
        qw3_metal_deltanet_conv1d_step(lw->linear_conv_weight->offset,
                                       qkv, conv_state, n_qkv,
                                       conv_gpu, conv_state_gpu) &&
        qw3_metal_l2norm_heads(conv_gpu, QW3_N_LINEAR_QK_HEADS,
                               QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS, q_gpu) &&
        qw3_metal_l2norm_heads(conv_gpu + qk_n, QW3_N_LINEAR_QK_HEADS,
                               QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS, k_gpu) &&
        qw3_metal_deltanet_recur(state_in, q_gpu, k_gpu, conv_gpu + qk_n * 2,
                                 beta_sig, gamma,
                                 QW3_N_LINEAR_QK_HEADS,
                                 QW3_N_LINEAR_V_HEADS,
                                 QW3_N_LINEAR_HEAD_DIM,
                                 gpu_state, gpu_core) &&
        qw3_metal_deltanet_gated_rmsnorm(lw->linear_ssm_norm->offset,
                                         gpu_core, z,
                                         QW3_N_LINEAR_V_HEADS,
                                         QW3_N_LINEAR_HEAD_DIM,
                                         QW3_RMS_EPS, gpu_inner) &&
        qw3_metal_matvec_q8_0(lw->linear_ssm_out->offset, gpu_inner,
                              inner_n, QW3_N_EMBD, gpu_attn);

    float attn_max = 0.0f, state_max = 0.0f, conv_state_max = 0.0f;
    double attn_rmsdiff = 0.0, attn_rms = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu_attn[i] - gpu_attn[i]);
            if (d > attn_max) attn_max = d;
            attn_rmsdiff += (double)d * d;
            attn_rms += (double)gpu_attn[i] * gpu_attn[i];
        }
        for (uint32_t i = 0; i < state_n; i++) {
            float d = fabsf(cpu_state[i] - gpu_state[i]);
            if (d > state_max) state_max = d;
        }
        for (uint32_t i = 0; i < conv_state_n; i++) {
            float d = fabsf(conv_state_cpu[i] - conv_state_gpu[i]);
            if (d > conv_state_max) conv_state_max = d;
        }
        attn_rmsdiff = sqrt(attn_rmsdiff / QW3_N_EMBD);
        attn_rms = sqrt(attn_rms / QW3_N_EMBD);
    }
    fprintf(fp,
            "metal deltanet branch step: %s token=%d attn_max=%.7g attn_rmsdiff=%.7g attn_rms=%.7g state_max=%.7g conv_state_max=%.7g attn0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token,
            attn_max, attn_rmsdiff, attn_rms, state_max, conv_state_max,
            gpu_attn[0], gpu_attn[1], gpu_attn[2], gpu_attn[3]);

    free(gpu_attn); free(cpu_attn); free(gpu_inner); free(cpu_inner);
    free(gpu_core); free(cpu_core); free(gpu_state); free(cpu_state);
    free(state_in); free(k_gpu); free(q_gpu); free(k_cpu); free(q_cpu);
    free(conv_gpu); free(conv_cpu); free(conv_state_gpu); free(conv_state_cpu);
    free(conv_state); free(gamma); free(beta_sig); free(beta); free(alpha);
    free(z); free(qkv); free(xn); free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_deltanet_layer_step_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal deltanet layer step: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal deltanet layer step: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal deltanet layer step: token %d is outside vocab\n", token);
        return -1;
    }

    const qw3_layer_weights *lw = &e->weights.layer[0];
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t inner_n = QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t conv_state_n = n_qkv * (QW3_N_LINEAR_CONV_K - 1);
    const uint32_t state_n = QW3_N_LINEAR_V_HEADS *
                             QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM;

    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_layer = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_conv_state = qw3_xmalloc((size_t)conv_state_n * sizeof(float));
    float *cpu_state = qw3_xmalloc((size_t)state_n * sizeof(float));

    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *z = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *alpha = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta_sig = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *gamma = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *conv_state = qw3_xmalloc((size_t)conv_state_n * sizeof(float));
    float *conv_state_out = qw3_xmalloc((size_t)conv_state_n * sizeof(float));
    float *conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *q = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *k = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *state_in = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *state_out = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *core = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *inner = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *attn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *ffn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *router = qw3_xmalloc((size_t)QW3_N_EXPERT * sizeof(float));
    float *egate = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *up = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *sparse = qw3_xcalloc((size_t)QW3_N_EMBD, sizeof(float));
    float *sh_gate = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *sh_up = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *sh_hidden = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *shared_down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *shared = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *moe = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_layer = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));

    for (uint32_t i = 0; i < conv_state_n; i++) {
        int r = (int)((i * 23u + 11u) % 103u) - 51;
        conv_state[i] = (float)r * 0.0008f;
        cpu_conv_state[i] = conv_state[i];
    }
    for (uint32_t i = 0; i < state_n; i++) {
        int r = (int)((i * 17u + 13u) % 97u) - 48;
        state_in[i] = (float)r * 0.0007f;
        cpu_state[i] = state_in[i];
    }

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd, (uint64_t)token, x) &&
              cpu_deltanet_layer(e, 0, x, cpu_conv_state, cpu_state, cpu_layer);
    int gpu_ok = ok &&
        qw3_metal_rmsnorm_weight_f32(x, lw->attn_norm->offset,
                                     xn, QW3_N_EMBD, QW3_RMS_EPS) &&
        qw3_metal_matvec_q8_0(lw->linear_qkv_proj->offset, xn,
                              QW3_N_EMBD, n_qkv, qkv) &&
        qw3_metal_matvec_q8_0(lw->linear_gate_proj->offset, xn,
                              QW3_N_EMBD, inner_n, z) &&
        qw3_metal_matvec_f32(lw->linear_ssm_alpha->offset, xn,
                             QW3_N_EMBD, QW3_N_LINEAR_V_HEADS, alpha) &&
        qw3_metal_matvec_f32(lw->linear_ssm_beta->offset, xn,
                             QW3_N_EMBD, QW3_N_LINEAR_V_HEADS, beta) &&
        qw3_metal_deltanet_conv1d_step(lw->linear_conv_weight->offset,
                                       qkv, conv_state, n_qkv,
                                       conv, conv_state_out) &&
        qw3_metal_l2norm_heads(conv, QW3_N_LINEAR_QK_HEADS,
                               QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS, q) &&
        qw3_metal_l2norm_heads(conv + qk_n, QW3_N_LINEAR_QK_HEADS,
                               QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS, k);
    if (gpu_ok) {
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            beta_sig[hv] = 1.0f / (1.0f + expf(-beta[hv]));
            const float ah = cpu_softplus(alpha[hv] +
                tensor_read_dense_1d(&e->model, lw->linear_ssm_dt_bias, (uint64_t)hv));
            gamma[hv] = expf(ah *
                tensor_read_dense_1d(&e->model, lw->linear_ssm_a, (uint64_t)hv));
        }
        gpu_ok =
            qw3_metal_deltanet_recur(state_in, q, k, conv + qk_n * 2,
                                     beta_sig, gamma,
                                     QW3_N_LINEAR_QK_HEADS,
                                     QW3_N_LINEAR_V_HEADS,
                                     QW3_N_LINEAR_HEAD_DIM,
                                     state_out, core) &&
            qw3_metal_deltanet_gated_rmsnorm(lw->linear_ssm_norm->offset,
                                             core, z,
                                             QW3_N_LINEAR_V_HEADS,
                                             QW3_N_LINEAR_HEAD_DIM,
                                             QW3_RMS_EPS, inner) &&
            qw3_metal_matvec_q8_0(lw->linear_ssm_out->offset, inner,
                                  inner_n, QW3_N_EMBD, attn) &&
            qw3_metal_residual_rmsnorm_weight_f32(x, attn,
                                                  lw->ffn_norm->offset,
                                                  ffn, QW3_N_EMBD,
                                                  QW3_RMS_EPS) &&
            qw3_metal_matvec_f32(lw->ffn_gate_inp->offset, ffn,
                                 QW3_N_EMBD, QW3_N_EXPERT, router);
    }

    int ids[QW3_N_EXPERT_USED];
    float vals[QW3_N_EXPERT_USED], weights[QW3_N_EXPERT_USED];
    memset(ids, 0, sizeof(ids));
    if (gpu_ok) {
        topk_desc(router, QW3_N_EXPERT, QW3_N_EXPERT_USED, ids, vals);
        float wsum = 0.0f;
        for (int kk = 0; kk < QW3_N_EXPERT_USED; kk++) {
            weights[kk] = expf(vals[kk] - vals[0]);
            wsum += weights[kk];
        }
        for (int kk = 0; kk < QW3_N_EXPERT_USED; kk++) weights[kk] /= wsum;
    }
    for (int kk = 0; gpu_ok && kk < QW3_N_EXPERT_USED; kk++) {
        gpu_ok =
            qw3_metal_matvec_iq3_s_expert(lw->ffn_gate_exps->offset, (uint32_t)ids[kk],
                                          ffn, QW3_N_EMBD, QW3_N_FF_EXP, egate) &&
            qw3_metal_matvec_iq3_s_expert(lw->ffn_up_exps->offset, (uint32_t)ids[kk],
                                          ffn, QW3_N_EMBD, QW3_N_FF_EXP, up) &&
            qw3_metal_silu_mul(egate, up, QW3_N_FF_EXP, hidden) &&
            qw3_metal_matvec_iq4_xs_expert(lw->ffn_down_exps->offset, (uint32_t)ids[kk],
                                           hidden, QW3_N_FF_EXP, QW3_N_EMBD, down);
        if (gpu_ok) for (int i = 0; i < QW3_N_EMBD; i++) sparse[i] += weights[kk] * down[i];
    }
    gpu_ok = gpu_ok &&
        qw3_metal_matvec_q8_0(lw->ffn_gate_shared->offset, ffn,
                              QW3_N_EMBD, QW3_N_FF_SHARED, sh_gate) &&
        qw3_metal_matvec_q8_0(lw->ffn_up_shared->offset, ffn,
                              QW3_N_EMBD, QW3_N_FF_SHARED, sh_up) &&
        qw3_metal_silu_mul(sh_gate, sh_up, QW3_N_FF_SHARED, sh_hidden) &&
        qw3_metal_matvec_q8_0(lw->ffn_down_shared->offset, sh_hidden,
                              QW3_N_FF_SHARED, QW3_N_EMBD, shared_down);
    float shared_raw = 0.0f;
    if (gpu_ok) gpu_ok = qw3_metal_matvec_f32(lw->ffn_gate_inp_shexp->offset, ffn,
                                             QW3_N_EMBD, 1, &shared_raw);
    if (gpu_ok) gpu_ok = qw3_metal_scale(shared_down, QW3_N_EMBD,
                                         1.0f / (1.0f + expf(-shared_raw)), shared);
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            moe[i] = sparse[i] + shared[i];
            gpu_layer[i] = x[i] + attn[i] + moe[i];
        }
    }

    float layer_max = 0.0f, state_max = 0.0f, conv_state_max = 0.0f;
    double layer_rmsdiff = 0.0, layer_rms = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu_layer[i] - gpu_layer[i]);
            if (d > layer_max) layer_max = d;
            layer_rmsdiff += (double)d * d;
            layer_rms += (double)gpu_layer[i] * gpu_layer[i];
        }
        for (uint32_t i = 0; i < state_n; i++) {
            float d = fabsf(cpu_state[i] - state_out[i]);
            if (d > state_max) state_max = d;
        }
        for (uint32_t i = 0; i < conv_state_n; i++) {
            float d = fabsf(cpu_conv_state[i] - conv_state_out[i]);
            if (d > conv_state_max) conv_state_max = d;
        }
        layer_rmsdiff = sqrt(layer_rmsdiff / QW3_N_EMBD);
        layer_rms = sqrt(layer_rms / QW3_N_EMBD);
    }
    fprintf(fp,
            "metal deltanet layer step: %s token=%d top0=%d layer_max=%.7g layer_rmsdiff=%.7g layer_rms=%.7g state_max=%.7g conv_state_max=%.7g out0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, ids[0],
            layer_max, layer_rmsdiff, layer_rms, state_max, conv_state_max,
            gpu_layer[0], gpu_layer[1], gpu_layer[2], gpu_layer[3]);

    free(gpu_layer); free(moe); free(shared); free(shared_down); free(sh_hidden);
    free(sh_up); free(sh_gate); free(sparse); free(down); free(hidden); free(up);
    free(egate); free(router); free(ffn); free(attn); free(inner); free(core);
    free(state_out); free(state_in); free(k); free(q); free(conv); free(conv_state_out);
    free(conv_state); free(gamma); free(beta_sig); free(beta); free(alpha); free(z);
    free(qkv); free(xn); free(cpu_state); free(cpu_conv_state); free(cpu_layer); free(x);
    return gpu_ok ? 0 : -1;
#endif
}

static int qw3_engine_metal_deltanet_layer_n_test(qw3_engine *e, int token,
                                                  int n_steps, const char *name,
                                                  FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token; (void)n_steps; (void)name;
    fprintf(fp, "%s: unavailable in QW3_NO_METAL build\n", name);
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "%s: Metal backend is not initialized\n", name);
        return -1;
    }
    if (n_steps <= 0 || token < 0 || token + n_steps - 1 >= QW3_N_VOCAB) {
        fprintf(fp, "%s: token %d cannot form a %d-token run\n", name, token, n_steps);
        return -1;
    }

    const qw3_layer_weights *lw = &e->weights.layer[0];
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t inner_n = QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t conv_state_n = n_qkv * (QW3_N_LINEAR_CONV_K - 1);
    const uint32_t state_n = QW3_N_LINEAR_V_HEADS *
                             QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM;

    float *cpu_x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_out = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_conv_state = qw3_xmalloc((size_t)conv_state_n * sizeof(float));
    float *cpu_state = qw3_xmalloc((size_t)state_n * sizeof(float));

    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *z = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *alpha = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta_sig = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *gamma = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *conv_state = qw3_xmalloc((size_t)conv_state_n * sizeof(float));
    float *conv_state_next = qw3_xmalloc((size_t)conv_state_n * sizeof(float));
    float *conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *q = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *k = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *state = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *state_next = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *core = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *inner = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *attn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *ffn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *router = qw3_xmalloc((size_t)QW3_N_EXPERT * sizeof(float));
    float *egate = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *up = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *sparse = qw3_xcalloc((size_t)QW3_N_EMBD, sizeof(float));
    float *sh_gate = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *sh_up = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *sh_hidden = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *shared_down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *shared = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_out = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));

    for (uint32_t i = 0; i < conv_state_n; i++) {
        int r = (int)((i * 23u + 11u) % 103u) - 51;
        cpu_conv_state[i] = (float)r * 0.0008f;
        conv_state[i] = cpu_conv_state[i];
    }
    for (uint32_t i = 0; i < state_n; i++) {
        int r = (int)((i * 17u + 13u) % 97u) - 48;
        cpu_state[i] = (float)r * 0.0007f;
        state[i] = cpu_state[i];
    }

    bool ok = true;
    for (int t = 0; ok && t < n_steps; t++) {
        ok = tensor_read_dense_row(&e->model, e->weights.token_embd,
                                   (uint64_t)(token + t), cpu_x) &&
             cpu_deltanet_layer(e, 0, cpu_x, cpu_conv_state, cpu_state, cpu_out);
    }

    int gpu_ok = ok;
    int last_top0 = -1;
    for (int t = 0; gpu_ok && t < n_steps; t++) {
        gpu_ok =
            qw3_metal_embed_q8_0(e->weights.token_embd->offset,
                                 (uint32_t)(token + t), QW3_N_EMBD, x) &&
            qw3_metal_rmsnorm_weight_f32(x, lw->attn_norm->offset,
                                         xn, QW3_N_EMBD, QW3_RMS_EPS) &&
            qw3_metal_matvec_q8_0(lw->linear_qkv_proj->offset, xn,
                                  QW3_N_EMBD, n_qkv, qkv) &&
            qw3_metal_matvec_q8_0(lw->linear_gate_proj->offset, xn,
                                  QW3_N_EMBD, inner_n, z) &&
            qw3_metal_matvec_f32(lw->linear_ssm_alpha->offset, xn,
                                 QW3_N_EMBD, QW3_N_LINEAR_V_HEADS, alpha) &&
            qw3_metal_matvec_f32(lw->linear_ssm_beta->offset, xn,
                                 QW3_N_EMBD, QW3_N_LINEAR_V_HEADS, beta) &&
            qw3_metal_deltanet_conv1d_step(lw->linear_conv_weight->offset,
                                           qkv, conv_state, n_qkv,
                                           conv, conv_state_next) &&
            qw3_metal_l2norm_heads(conv, QW3_N_LINEAR_QK_HEADS,
                                   QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS, q) &&
            qw3_metal_l2norm_heads(conv + qk_n, QW3_N_LINEAR_QK_HEADS,
                                   QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS, k);
        if (gpu_ok) {
            for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
                beta_sig[hv] = 1.0f / (1.0f + expf(-beta[hv]));
                const float ah = cpu_softplus(alpha[hv] +
                    tensor_read_dense_1d(&e->model, lw->linear_ssm_dt_bias, (uint64_t)hv));
                gamma[hv] = expf(ah *
                    tensor_read_dense_1d(&e->model, lw->linear_ssm_a, (uint64_t)hv));
            }
            gpu_ok =
                qw3_metal_deltanet_recur(state, q, k, conv + qk_n * 2,
                                         beta_sig, gamma,
                                         QW3_N_LINEAR_QK_HEADS,
                                         QW3_N_LINEAR_V_HEADS,
                                         QW3_N_LINEAR_HEAD_DIM,
                                         state_next, core) &&
                qw3_metal_deltanet_gated_rmsnorm(lw->linear_ssm_norm->offset,
                                                 core, z,
                                                 QW3_N_LINEAR_V_HEADS,
                                                 QW3_N_LINEAR_HEAD_DIM,
                                                 QW3_RMS_EPS, inner) &&
                qw3_metal_matvec_q8_0(lw->linear_ssm_out->offset, inner,
                                      inner_n, QW3_N_EMBD, attn) &&
                qw3_metal_residual_rmsnorm_weight_f32(x, attn,
                                                      lw->ffn_norm->offset,
                                                      ffn, QW3_N_EMBD,
                                                      QW3_RMS_EPS) &&
                qw3_metal_matvec_f32(lw->ffn_gate_inp->offset, ffn,
                                     QW3_N_EMBD, QW3_N_EXPERT, router);
        }
        int ids[QW3_N_EXPERT_USED];
        float vals[QW3_N_EXPERT_USED], weights[QW3_N_EXPERT_USED];
        memset(ids, 0, sizeof(ids));
        if (gpu_ok) {
            topk_desc(router, QW3_N_EXPERT, QW3_N_EXPERT_USED, ids, vals);
            last_top0 = ids[0];
            float wsum = 0.0f;
            for (int kk = 0; kk < QW3_N_EXPERT_USED; kk++) {
                weights[kk] = expf(vals[kk] - vals[0]);
                wsum += weights[kk];
            }
            for (int kk = 0; kk < QW3_N_EXPERT_USED; kk++) weights[kk] /= wsum;
            memset(sparse, 0, (size_t)QW3_N_EMBD * sizeof(float));
        }
        for (int kk = 0; gpu_ok && kk < QW3_N_EXPERT_USED; kk++) {
            gpu_ok =
                qw3_metal_matvec_iq3_s_expert(lw->ffn_gate_exps->offset, (uint32_t)ids[kk],
                                              ffn, QW3_N_EMBD, QW3_N_FF_EXP, egate) &&
                qw3_metal_matvec_iq3_s_expert(lw->ffn_up_exps->offset, (uint32_t)ids[kk],
                                              ffn, QW3_N_EMBD, QW3_N_FF_EXP, up) &&
                qw3_metal_silu_mul(egate, up, QW3_N_FF_EXP, hidden) &&
                qw3_metal_matvec_iq4_xs_expert(lw->ffn_down_exps->offset, (uint32_t)ids[kk],
                                               hidden, QW3_N_FF_EXP, QW3_N_EMBD, down);
            if (gpu_ok) for (int i = 0; i < QW3_N_EMBD; i++) sparse[i] += weights[kk] * down[i];
        }
        gpu_ok = gpu_ok &&
            qw3_metal_matvec_q8_0(lw->ffn_gate_shared->offset, ffn,
                                  QW3_N_EMBD, QW3_N_FF_SHARED, sh_gate) &&
            qw3_metal_matvec_q8_0(lw->ffn_up_shared->offset, ffn,
                                  QW3_N_EMBD, QW3_N_FF_SHARED, sh_up) &&
            qw3_metal_silu_mul(sh_gate, sh_up, QW3_N_FF_SHARED, sh_hidden) &&
            qw3_metal_matvec_q8_0(lw->ffn_down_shared->offset, sh_hidden,
                                  QW3_N_FF_SHARED, QW3_N_EMBD, shared_down);
        float shared_raw = 0.0f;
        if (gpu_ok) gpu_ok = qw3_metal_matvec_f32(lw->ffn_gate_inp_shexp->offset, ffn,
                                                 QW3_N_EMBD, 1, &shared_raw);
        if (gpu_ok) gpu_ok = qw3_metal_scale(shared_down, QW3_N_EMBD,
                                             1.0f / (1.0f + expf(-shared_raw)), shared);
        if (gpu_ok) {
            for (int i = 0; i < QW3_N_EMBD; i++) gpu_out[i] = x[i] + attn[i] + sparse[i] + shared[i];
            memcpy(conv_state, conv_state_next, (size_t)conv_state_n * sizeof(float));
            memcpy(state, state_next, (size_t)state_n * sizeof(float));
        }
    }

    float layer_max = 0.0f, state_max = 0.0f, conv_state_max = 0.0f;
    double layer_rmsdiff = 0.0, layer_rms = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu_out[i] - gpu_out[i]);
            if (d > layer_max) layer_max = d;
            layer_rmsdiff += (double)d * d;
            layer_rms += (double)gpu_out[i] * gpu_out[i];
        }
        for (uint32_t i = 0; i < state_n; i++) {
            float d = fabsf(cpu_state[i] - state[i]);
            if (d > state_max) state_max = d;
        }
        for (uint32_t i = 0; i < conv_state_n; i++) {
            float d = fabsf(cpu_conv_state[i] - conv_state[i]);
            if (d > conv_state_max) conv_state_max = d;
        }
        layer_rmsdiff = sqrt(layer_rmsdiff / QW3_N_EMBD);
        layer_rms = sqrt(layer_rms / QW3_N_EMBD);
    }
    fprintf(fp,
            "%s: %s token=%d..%d top0=%d layer_max=%.7g layer_rmsdiff=%.7g layer_rms=%.7g state_max=%.7g conv_state_max=%.7g out0=[%.7g %.7g %.7g %.7g]\n",
            name, gpu_ok ? "ok" : "failed", token, token + n_steps - 1, last_top0,
            layer_max, layer_rmsdiff, layer_rms, state_max, conv_state_max,
            gpu_out[0], gpu_out[1], gpu_out[2], gpu_out[3]);

    free(gpu_out); free(shared); free(shared_down); free(sh_hidden); free(sh_up);
    free(sh_gate); free(sparse); free(down); free(hidden); free(up); free(egate);
    free(router); free(ffn); free(attn); free(inner); free(core); free(state_next);
    free(state); free(k); free(q); free(conv); free(conv_state_next); free(conv_state);
    free(gamma); free(beta_sig); free(beta); free(alpha); free(z); free(qkv); free(xn);
    free(x); free(cpu_state); free(cpu_conv_state); free(cpu_out); free(cpu_x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_deltanet_layer2_test(qw3_engine *e, int token, FILE *fp) {
    return qw3_engine_metal_deltanet_layer_n_test(e, token, 2,
                                                  "metal deltanet layer2", fp);
}

int qw3_engine_metal_deltanet_layer4_test(qw3_engine *e, int token, FILE *fp) {
    return qw3_engine_metal_deltanet_layer_n_test(e, token, 4,
                                                  "metal deltanet layer4", fp);
}

int qw3_engine_metal_deltanet_layer8_test(qw3_engine *e, int token, FILE *fp) {
    return qw3_engine_metal_deltanet_layer_n_test(e, token, 8,
                                                  "metal deltanet layer8", fp);
}

int qw3_engine_metal_deltanet_branch_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal deltanet branch: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal deltanet branch: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal deltanet branch: token %d is outside vocab\n", token);
        return -1;
    }

    const qw3_layer_weights *lw = &e->weights.layer[0];
    if (lw->linear_ssm_out->type != QW3_TENSOR_Q8_0) {
        fprintf(fp, "metal deltanet branch: expected q8_0 ssm_out, got %s\n",
                tensor_type_name(lw->linear_ssm_out->type));
        return -1;
    }

    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t inner_n = QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t state_n = QW3_N_LINEAR_V_HEADS *
                             QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM;
    float *x_cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn_cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv_cpu = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *z_cpu = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *alpha_cpu = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta_cpu = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *conv_cpu = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *q_cpu = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *k_cpu = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *core_cpu = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *inner_cpu = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *attn_cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *state_in = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *state_cpu = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *conv_state = qw3_xcalloc((size_t)n_qkv * (QW3_N_LINEAR_CONV_K - 1),
                                    sizeof(float));

    float *x_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv_gpu = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *z_gpu = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *alpha_gpu = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta_gpu = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta_sig = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *gamma = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *conv_gpu = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *q_gpu = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *k_gpu = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *core_gpu = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *inner_gpu = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *attn_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *state_gpu = qw3_xmalloc((size_t)state_n * sizeof(float));

    for (uint32_t i = 0; i < state_n; i++) {
        int r = (int)((i * 17u + 13u) % 97u) - 48;
        state_in[i] = (float)r * 0.0007f;
    }

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd,
                                    (uint64_t)token, x_cpu);
    if (ok) {
        cpu_rmsnorm(xn_cpu, x_cpu, &e->model, lw->attn_norm, QW3_N_EMBD);
        ok = cpu_matvec_q8_0(&e->model, lw->linear_qkv_proj, xn_cpu, qkv_cpu) &&
             cpu_matvec_q8_0(&e->model, lw->linear_gate_proj, xn_cpu, z_cpu) &&
             cpu_matvec_dense(&e->model, lw->linear_ssm_alpha, xn_cpu, alpha_cpu) &&
             cpu_matvec_dense(&e->model, lw->linear_ssm_beta, xn_cpu, beta_cpu) &&
             cpu_deltanet_conv1d_step(&e->model, lw, qkv_cpu, conv_state, conv_cpu);
    }
    if (ok) {
        const float *qraw = conv_cpu;
        const float *kraw = conv_cpu + qk_n;
        for (int h = 0; h < QW3_N_LINEAR_QK_HEADS; h++) {
            cpu_l2_norm_head(q_cpu + h * QW3_N_LINEAR_HEAD_DIM,
                             qraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
            cpu_l2_norm_head(k_cpu + h * QW3_N_LINEAR_HEAD_DIM,
                             kraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            const int hk = hv % QW3_N_LINEAR_QK_HEADS;
            const float *qh = q_cpu + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *kh = k_cpu + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *vh = conv_cpu + qk_n * 2 + hv * QW3_N_LINEAR_HEAD_DIM;
            const float bh = 1.0f / (1.0f + expf(-beta_cpu[hv]));
            const float ah = cpu_softplus(alpha_cpu[hv] +
                tensor_read_dense_1d(&e->model, lw->linear_ssm_dt_bias, (uint64_t)hv));
            const float gh = expf(ah *
                tensor_read_dense_1d(&e->model, lw->linear_ssm_a, (uint64_t)hv));
            float *sh_out = state_cpu + (uint64_t)hv * QW3_N_LINEAR_HEAD_DIM *
                            QW3_N_LINEAR_HEAD_DIM;
            const float *sh_in = state_in + (uint64_t)hv * QW3_N_LINEAR_HEAD_DIM *
                                 QW3_N_LINEAR_HEAD_DIM;
            float *oh = core_cpu + hv * QW3_N_LINEAR_HEAD_DIM;
            for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
                float sk = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    sk += sh_in[i * QW3_N_LINEAR_HEAD_DIM + j] * gh * kh[i];
                }
                const float d = bh * (vh[j] - sk);
                float out = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    const int idx = i * QW3_N_LINEAR_HEAD_DIM + j;
                    const float sv = sh_in[idx] * gh + kh[i] * d;
                    sh_out[idx] = sv;
                    out += sv * qh[i];
                }
                oh[j] = out / sqrtf((float)QW3_N_LINEAR_HEAD_DIM);
            }
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            float *dst = inner_cpu + hv * QW3_N_LINEAR_HEAD_DIM;
            const float *src = core_cpu + hv * QW3_N_LINEAR_HEAD_DIM;
            const float *zh = z_cpu + hv * QW3_N_LINEAR_HEAD_DIM;
            double ss = 0.0;
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                ss += (double)src[i] * (double)src[i];
            }
            const float scale = 1.0f /
                sqrtf((float)(ss / QW3_N_LINEAR_HEAD_DIM) + QW3_RMS_EPS);
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                dst[i] = src[i] * scale *
                         tensor_read_dense_1d(&e->model, lw->linear_ssm_norm, (uint64_t)i) *
                         cpu_silu(zh[i]);
            }
        }
        ok = cpu_matvec_q8_0(&e->model, lw->linear_ssm_out, inner_cpu, attn_cpu);
    }

    int gpu_ok = ok &&
        qw3_metal_embed_q8_0(e->weights.token_embd->offset, (uint32_t)token,
                             QW3_N_EMBD, x_gpu) &&
        qw3_metal_rmsnorm_weight_f32(x_gpu, lw->attn_norm->offset,
                                     xn_gpu, QW3_N_EMBD, QW3_RMS_EPS) &&
        qw3_metal_matvec_q8_0(lw->linear_qkv_proj->offset, xn_gpu,
                              QW3_N_EMBD, n_qkv, qkv_gpu) &&
        qw3_metal_matvec_q8_0(lw->linear_gate_proj->offset, xn_gpu,
                              QW3_N_EMBD, inner_n, z_gpu) &&
        qw3_metal_matvec_f32(lw->linear_ssm_alpha->offset, xn_gpu,
                             QW3_N_EMBD, QW3_N_LINEAR_V_HEADS, alpha_gpu) &&
        qw3_metal_matvec_f32(lw->linear_ssm_beta->offset, xn_gpu,
                             QW3_N_EMBD, QW3_N_LINEAR_V_HEADS, beta_gpu) &&
        qw3_metal_deltanet_conv1d_zero(lw->linear_conv_weight->offset,
                                       qkv_gpu, n_qkv, conv_gpu) &&
        qw3_metal_l2norm_heads(conv_gpu, QW3_N_LINEAR_QK_HEADS,
                               QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS, q_gpu) &&
        qw3_metal_l2norm_heads(conv_gpu + qk_n, QW3_N_LINEAR_QK_HEADS,
                               QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS, k_gpu);
    if (gpu_ok) {
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            beta_sig[hv] = 1.0f / (1.0f + expf(-beta_gpu[hv]));
            const float ah = cpu_softplus(alpha_gpu[hv] +
                tensor_read_dense_1d(&e->model, lw->linear_ssm_dt_bias, (uint64_t)hv));
            gamma[hv] = expf(ah *
                tensor_read_dense_1d(&e->model, lw->linear_ssm_a, (uint64_t)hv));
        }
        gpu_ok =
            qw3_metal_deltanet_recur(state_in, q_gpu, k_gpu, conv_gpu + qk_n * 2,
                                     beta_sig, gamma,
                                     QW3_N_LINEAR_QK_HEADS,
                                     QW3_N_LINEAR_V_HEADS,
                                     QW3_N_LINEAR_HEAD_DIM,
                                     state_gpu, core_gpu) &&
            qw3_metal_deltanet_gated_rmsnorm(lw->linear_ssm_norm->offset,
                                             core_gpu, z_gpu,
                                             QW3_N_LINEAR_V_HEADS,
                                             QW3_N_LINEAR_HEAD_DIM,
                                             QW3_RMS_EPS, inner_gpu) &&
            qw3_metal_matvec_q8_0(lw->linear_ssm_out->offset, inner_gpu,
                                  inner_n, QW3_N_EMBD, attn_gpu);
    }

    float attn_maxdiff = 0.0f;
    double attn_rmsdiff = 0.0;
    double attn_rms = 0.0;
    float state_maxdiff = 0.0f;
    if (gpu_ok) {
        for (uint32_t i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(attn_cpu[i] - attn_gpu[i]);
            if (d > attn_maxdiff) attn_maxdiff = d;
            attn_rmsdiff += (double)d * (double)d;
            attn_rms += (double)attn_gpu[i] * (double)attn_gpu[i];
        }
        for (uint32_t i = 0; i < state_n; i++) {
            float d = fabsf(state_cpu[i] - state_gpu[i]);
            if (d > state_maxdiff) state_maxdiff = d;
        }
        attn_rmsdiff = sqrt(attn_rmsdiff / (double)QW3_N_EMBD);
        attn_rms = sqrt(attn_rms / (double)QW3_N_EMBD);
    }
    fprintf(fp,
            "metal deltanet branch: %s token=%d attn_maxdiff=%.7g attn_rmsdiff=%.7g state_maxdiff=%.7g attn_rms=%.7g attn0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, attn_maxdiff, attn_rmsdiff,
            state_maxdiff, attn_rms,
            attn_gpu[0], attn_gpu[1], attn_gpu[2], attn_gpu[3]);

    free(state_gpu);
    free(attn_gpu);
    free(inner_gpu);
    free(core_gpu);
    free(k_gpu);
    free(q_gpu);
    free(conv_gpu);
    free(gamma);
    free(beta_sig);
    free(beta_gpu);
    free(alpha_gpu);
    free(z_gpu);
    free(qkv_gpu);
    free(xn_gpu);
    free(x_gpu);
    free(conv_state);
    free(state_cpu);
    free(state_in);
    free(attn_cpu);
    free(inner_cpu);
    free(core_cpu);
    free(k_cpu);
    free(q_cpu);
    free(conv_cpu);
    free(beta_cpu);
    free(alpha_cpu);
    free(z_cpu);
    free(qkv_cpu);
    free(xn_cpu);
    free(x_cpu);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_deltanet_residual_norm_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal deltanet residual norm: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal deltanet residual norm: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal deltanet residual norm: token %d is outside vocab\n", token);
        return -1;
    }

    const qw3_layer_weights *lw = &e->weights.layer[0];
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t inner_n = QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t state_n = QW3_N_LINEAR_V_HEADS *
                             QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *z = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *alpha = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *qnorm = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *knorm = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *state = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *core = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *inner = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *attn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *resid = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_ffn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_ffn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *conv_state = qw3_xcalloc((size_t)n_qkv * (QW3_N_LINEAR_CONV_K - 1),
                                    sizeof(float));

    for (uint32_t i = 0; i < state_n; i++) {
        int r = (int)((i * 17u + 13u) % 97u) - 48;
        state[i] = (float)r * 0.0007f;
    }

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd, (uint64_t)token, x);
    if (ok) {
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        ok = cpu_matvec_q8_0(&e->model, lw->linear_qkv_proj, xn, qkv) &&
             cpu_matvec_q8_0(&e->model, lw->linear_gate_proj, xn, z) &&
             cpu_matvec_dense(&e->model, lw->linear_ssm_alpha, xn, alpha) &&
             cpu_matvec_dense(&e->model, lw->linear_ssm_beta, xn, beta) &&
             cpu_deltanet_conv1d_step(&e->model, lw, qkv, conv_state, conv);
    }
    if (ok) {
        const float *qraw = conv;
        const float *kraw = conv + qk_n;
        const float *vraw = conv + qk_n * 2;
        for (int h = 0; h < QW3_N_LINEAR_QK_HEADS; h++) {
            cpu_l2_norm_head(qnorm + h * QW3_N_LINEAR_HEAD_DIM,
                             qraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
            cpu_l2_norm_head(knorm + h * QW3_N_LINEAR_HEAD_DIM,
                             kraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            const int hk = hv % QW3_N_LINEAR_QK_HEADS;
            const float *qh = qnorm + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *kh = knorm + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *vh = vraw + hv * QW3_N_LINEAR_HEAD_DIM;
            float *sh = state + (uint64_t)hv * QW3_N_LINEAR_HEAD_DIM *
                        QW3_N_LINEAR_HEAD_DIM;
            float *oh = core + hv * QW3_N_LINEAR_HEAD_DIM;
            const float bh = 1.0f / (1.0f + expf(-beta[hv]));
            const float ah = cpu_softplus(alpha[hv] +
                tensor_read_dense_1d(&e->model, lw->linear_ssm_dt_bias, (uint64_t)hv));
            const float gh = expf(ah *
                tensor_read_dense_1d(&e->model, lw->linear_ssm_a, (uint64_t)hv));
            for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
                float sk = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    sh[i * QW3_N_LINEAR_HEAD_DIM + j] *= gh;
                    sk += sh[i * QW3_N_LINEAR_HEAD_DIM + j] * kh[i];
                }
                const float d = bh * (vh[j] - sk);
                float out = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    sh[i * QW3_N_LINEAR_HEAD_DIM + j] += kh[i] * d;
                    out += sh[i * QW3_N_LINEAR_HEAD_DIM + j] * qh[i];
                }
                oh[j] = out / sqrtf((float)QW3_N_LINEAR_HEAD_DIM);
            }
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            float *dst = inner + hv * QW3_N_LINEAR_HEAD_DIM;
            const float *src = core + hv * QW3_N_LINEAR_HEAD_DIM;
            const float *zh = z + hv * QW3_N_LINEAR_HEAD_DIM;
            double ss = 0.0;
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) ss += (double)src[i] * src[i];
            const float scale = 1.0f /
                sqrtf((float)(ss / QW3_N_LINEAR_HEAD_DIM) + QW3_RMS_EPS);
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                dst[i] = src[i] * scale *
                         tensor_read_dense_1d(&e->model, lw->linear_ssm_norm, (uint64_t)i) *
                         cpu_silu(zh[i]);
            }
        }
        ok = cpu_matvec_q8_0(&e->model, lw->linear_ssm_out, inner, attn);
    }
    if (ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) resid[i] = x[i] + attn[i];
        cpu_rmsnorm(cpu_ffn, resid, &e->model, lw->ffn_norm, QW3_N_EMBD);
    }
    int gpu_ok = ok &&
        qw3_metal_residual_rmsnorm_weight_f32(x, attn, lw->ffn_norm->offset,
                                              gpu_ffn, QW3_N_EMBD, QW3_RMS_EPS);
    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    double ffn_rms = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu_ffn[i] - gpu_ffn[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * d;
            ffn_rms += (double)gpu_ffn[i] * gpu_ffn[i];
        }
        rmsdiff = sqrt(rmsdiff / QW3_N_EMBD);
        ffn_rms = sqrt(ffn_rms / QW3_N_EMBD);
    }
    fprintf(fp,
            "metal deltanet residual norm: %s token=%d maxdiff=%.7g rmsdiff=%.7g ffn_rms=%.7g ffn0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, maxdiff, rmsdiff, ffn_rms,
            gpu_ffn[0], gpu_ffn[1], gpu_ffn[2], gpu_ffn[3]);

    free(conv_state);
    free(gpu_ffn);
    free(cpu_ffn);
    free(resid);
    free(attn);
    free(inner);
    free(core);
    free(state);
    free(knorm);
    free(qnorm);
    free(conv);
    free(beta);
    free(alpha);
    free(z);
    free(qkv);
    free(xn);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_moe_router_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal moe router: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal moe router: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal moe router: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_layer_weights *lw = &e->weights.layer[0];
    if (lw->ffn_gate_inp->type != QW3_TENSOR_F32) {
        fprintf(fp, "metal moe router: expected f32 router, got %s\n",
                tensor_type_name(lw->ffn_gate_inp->type));
        return -1;
    }

    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t inner_n = QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t state_n = QW3_N_LINEAR_V_HEADS *
                             QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *z = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *alpha = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *qnorm = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *knorm = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *state = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *core = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *inner = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *attn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *resid = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_ffn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_ffn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_router = qw3_xmalloc((size_t)QW3_N_EXPERT * sizeof(float));
    float *gpu_router = qw3_xmalloc((size_t)QW3_N_EXPERT * sizeof(float));
    float *conv_state = qw3_xcalloc((size_t)n_qkv * (QW3_N_LINEAR_CONV_K - 1),
                                    sizeof(float));

    for (uint32_t i = 0; i < state_n; i++) {
        int r = (int)((i * 17u + 13u) % 97u) - 48;
        state[i] = (float)r * 0.0007f;
    }

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd, (uint64_t)token, x);
    if (ok) {
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        ok = cpu_matvec_q8_0(&e->model, lw->linear_qkv_proj, xn, qkv) &&
             cpu_matvec_q8_0(&e->model, lw->linear_gate_proj, xn, z) &&
             cpu_matvec_dense(&e->model, lw->linear_ssm_alpha, xn, alpha) &&
             cpu_matvec_dense(&e->model, lw->linear_ssm_beta, xn, beta) &&
             cpu_deltanet_conv1d_step(&e->model, lw, qkv, conv_state, conv);
    }
    if (ok) {
        const float *qraw = conv;
        const float *kraw = conv + qk_n;
        const float *vraw = conv + qk_n * 2;
        for (int h = 0; h < QW3_N_LINEAR_QK_HEADS; h++) {
            cpu_l2_norm_head(qnorm + h * QW3_N_LINEAR_HEAD_DIM,
                             qraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
            cpu_l2_norm_head(knorm + h * QW3_N_LINEAR_HEAD_DIM,
                             kraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            const int hk = hv % QW3_N_LINEAR_QK_HEADS;
            const float *qh = qnorm + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *kh = knorm + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *vh = vraw + hv * QW3_N_LINEAR_HEAD_DIM;
            float *sh = state + (uint64_t)hv * QW3_N_LINEAR_HEAD_DIM *
                        QW3_N_LINEAR_HEAD_DIM;
            float *oh = core + hv * QW3_N_LINEAR_HEAD_DIM;
            const float bh = 1.0f / (1.0f + expf(-beta[hv]));
            const float ah = cpu_softplus(alpha[hv] +
                tensor_read_dense_1d(&e->model, lw->linear_ssm_dt_bias, (uint64_t)hv));
            const float gh = expf(ah *
                tensor_read_dense_1d(&e->model, lw->linear_ssm_a, (uint64_t)hv));
            for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
                float sk = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    sh[i * QW3_N_LINEAR_HEAD_DIM + j] *= gh;
                    sk += sh[i * QW3_N_LINEAR_HEAD_DIM + j] * kh[i];
                }
                const float d = bh * (vh[j] - sk);
                float out = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    sh[i * QW3_N_LINEAR_HEAD_DIM + j] += kh[i] * d;
                    out += sh[i * QW3_N_LINEAR_HEAD_DIM + j] * qh[i];
                }
                oh[j] = out / sqrtf((float)QW3_N_LINEAR_HEAD_DIM);
            }
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            float *dst = inner + hv * QW3_N_LINEAR_HEAD_DIM;
            const float *src = core + hv * QW3_N_LINEAR_HEAD_DIM;
            const float *zh = z + hv * QW3_N_LINEAR_HEAD_DIM;
            double ss = 0.0;
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) ss += (double)src[i] * src[i];
            const float scale = 1.0f /
                sqrtf((float)(ss / QW3_N_LINEAR_HEAD_DIM) + QW3_RMS_EPS);
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                dst[i] = src[i] * scale *
                         tensor_read_dense_1d(&e->model, lw->linear_ssm_norm, (uint64_t)i) *
                         cpu_silu(zh[i]);
            }
        }
        ok = cpu_matvec_q8_0(&e->model, lw->linear_ssm_out, inner, attn);
    }
    if (ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) resid[i] = x[i] + attn[i];
        cpu_rmsnorm(cpu_ffn, resid, &e->model, lw->ffn_norm, QW3_N_EMBD);
        ok = cpu_matvec_dense(&e->model, lw->ffn_gate_inp, cpu_ffn, cpu_router);
    }
    int gpu_ok = ok &&
        qw3_metal_residual_rmsnorm_weight_f32(x, attn, lw->ffn_norm->offset,
                                              gpu_ffn, QW3_N_EMBD, QW3_RMS_EPS) &&
        qw3_metal_matvec_f32(lw->ffn_gate_inp->offset, gpu_ffn, QW3_N_EMBD,
                             QW3_N_EXPERT, gpu_router);

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    int cpu_ids[QW3_N_EXPERT_USED], gpu_ids[QW3_N_EXPERT_USED];
    float cpu_vals[QW3_N_EXPERT_USED], gpu_vals[QW3_N_EXPERT_USED];
    float cpu_w[QW3_N_EXPERT_USED], gpu_w[QW3_N_EXPERT_USED];
    int top_match = 1;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EXPERT; i++) {
            float d = fabsf(cpu_router[i] - gpu_router[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * d;
        }
        rmsdiff = sqrt(rmsdiff / QW3_N_EXPERT);
        topk_desc(cpu_router, QW3_N_EXPERT, QW3_N_EXPERT_USED, cpu_ids, cpu_vals);
        topk_desc(gpu_router, QW3_N_EXPERT, QW3_N_EXPERT_USED, gpu_ids, gpu_vals);
        float cpu_sum = 0.0f, gpu_sum = 0.0f;
        for (int k = 0; k < QW3_N_EXPERT_USED; k++) {
            if (cpu_ids[k] != gpu_ids[k]) top_match = 0;
            cpu_w[k] = expf(cpu_vals[k] - cpu_vals[0]);
            gpu_w[k] = expf(gpu_vals[k] - gpu_vals[0]);
            cpu_sum += cpu_w[k];
            gpu_sum += gpu_w[k];
        }
        for (int k = 0; k < QW3_N_EXPERT_USED; k++) {
            cpu_w[k] /= cpu_sum;
            gpu_w[k] /= gpu_sum;
        }
    } else {
        memset(cpu_ids, 0, sizeof(cpu_ids));
        memset(gpu_ids, 0, sizeof(gpu_ids));
        memset(cpu_w, 0, sizeof(cpu_w));
        memset(gpu_w, 0, sizeof(gpu_w));
    }
    fprintf(fp,
            "metal moe router: %s token=%d maxdiff=%.7g rmsdiff=%.7g top_match=%s top8=%d,%d,%d,%d,%d,%d,%d,%d w0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, maxdiff, rmsdiff,
            top_match ? "yes" : "no",
            gpu_ids[0], gpu_ids[1], gpu_ids[2], gpu_ids[3],
            gpu_ids[4], gpu_ids[5], gpu_ids[6], gpu_ids[7],
            gpu_w[0], gpu_w[1], gpu_w[2], gpu_w[3]);

    free(conv_state);
    free(gpu_router);
    free(cpu_router);
    free(gpu_ffn);
    free(cpu_ffn);
    free(resid);
    free(attn);
    free(inner);
    free(core);
    free(state);
    free(knorm);
    free(qnorm);
    free(conv);
    free(beta);
    free(alpha);
    free(z);
    free(qkv);
    free(xn);
    free(x);
    return gpu_ok && top_match ? 0 : -1;
#endif
}

int qw3_engine_metal_moe_shared_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal moe shared: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal moe shared: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal moe shared: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_layer_weights *lw = &e->weights.layer[0];
    if (lw->ffn_gate_shared->type != QW3_TENSOR_Q8_0 ||
        lw->ffn_up_shared->type != QW3_TENSOR_Q8_0 ||
        lw->ffn_down_shared->type != QW3_TENSOR_Q8_0) {
        fprintf(fp, "metal moe shared: expected q8_0 shared tensors, got gate=%s up=%s down=%s\n",
                tensor_type_name(lw->ffn_gate_shared->type),
                tensor_type_name(lw->ffn_up_shared->type),
                tensor_type_name(lw->ffn_down_shared->type));
        return -1;
    }

    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *ffn_in = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_gate = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *cpu_up = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *cpu_hidden = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *cpu_out = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_gate = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *gpu_up = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *gpu_hidden = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *gpu_down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_out = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float cpu_shared_raw = 0.0f;
    float gpu_shared_raw = 0.0f;

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd, (uint64_t)token, x);
    if (ok) {
        cpu_rmsnorm(ffn_in, x, &e->model, lw->ffn_norm, QW3_N_EMBD);
        ok = cpu_matvec_q8_0(&e->model, lw->ffn_gate_shared, ffn_in, cpu_gate) &&
             cpu_matvec_q8_0(&e->model, lw->ffn_up_shared, ffn_in, cpu_up) &&
             cpu_dot_dense_1d(&e->model, lw->ffn_gate_inp_shexp, ffn_in, &cpu_shared_raw);
    }
    if (ok) {
        for (int i = 0; i < QW3_N_FF_SHARED; i++) {
            cpu_hidden[i] = cpu_silu(cpu_gate[i]) * cpu_up[i];
        }
        ok = cpu_matvec_q8_0(&e->model, lw->ffn_down_shared, cpu_hidden, cpu_out);
    }
    const float cpu_shared_gate = 1.0f / (1.0f + expf(-cpu_shared_raw));
    if (ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) cpu_out[i] *= cpu_shared_gate;
    }

    int gpu_ok = ok &&
        qw3_metal_matvec_q8_0(lw->ffn_gate_shared->offset, ffn_in,
                              QW3_N_EMBD, QW3_N_FF_SHARED, gpu_gate) &&
        qw3_metal_matvec_q8_0(lw->ffn_up_shared->offset, ffn_in,
                              QW3_N_EMBD, QW3_N_FF_SHARED, gpu_up) &&
        qw3_metal_silu_mul(gpu_gate, gpu_up, QW3_N_FF_SHARED, gpu_hidden) &&
        qw3_metal_matvec_q8_0(lw->ffn_down_shared->offset, gpu_hidden,
                              QW3_N_FF_SHARED, QW3_N_EMBD, gpu_down) &&
        qw3_metal_matvec_f32(lw->ffn_gate_inp_shexp->offset, ffn_in,
                             QW3_N_EMBD, 1, &gpu_shared_raw);
    const float gpu_shared_gate = 1.0f / (1.0f + expf(-gpu_shared_raw));
    if (gpu_ok) {
        gpu_ok = qw3_metal_scale(gpu_down, QW3_N_EMBD, gpu_shared_gate, gpu_out);
    }

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    double out_rms = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu_out[i] - gpu_out[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * d;
            out_rms += (double)gpu_out[i] * gpu_out[i];
        }
        rmsdiff = sqrt(rmsdiff / QW3_N_EMBD);
        out_rms = sqrt(out_rms / QW3_N_EMBD);
    }
    fprintf(fp,
            "metal moe shared: %s token=%d maxdiff=%.7g rmsdiff=%.7g gate=%.7g out_rms=%.7g out0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, maxdiff, rmsdiff,
            gpu_shared_gate, out_rms,
            gpu_out[0], gpu_out[1], gpu_out[2], gpu_out[3]);

    free(gpu_out);
    free(gpu_down);
    free(gpu_hidden);
    free(gpu_up);
    free(gpu_gate);
    free(cpu_out);
    free(cpu_hidden);
    free(cpu_up);
    free(cpu_gate);
    free(ffn_in);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_moe_iq4_down_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal moe iq4 down: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal moe iq4 down: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal moe iq4 down: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_layer_weights *lw = &e->weights.layer[0];
    if (lw->ffn_gate_inp->type != QW3_TENSOR_F32 ||
        lw->ffn_gate_exps->type != QW3_TENSOR_IQ3_S ||
        lw->ffn_up_exps->type != QW3_TENSOR_IQ3_S ||
        lw->ffn_down_exps->type != QW3_TENSOR_IQ4_XS) {
        fprintf(fp,
                "metal moe iq4 down: expected router=f32 gate/up=iq3_s down=iq4_xs, got router=%s gate=%s up=%s down=%s\n",
                tensor_type_name(lw->ffn_gate_inp->type),
                tensor_type_name(lw->ffn_gate_exps->type),
                tensor_type_name(lw->ffn_up_exps->type),
                tensor_type_name(lw->ffn_down_exps->type));
        return -1;
    }

    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *ffn_in = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *router = qw3_xmalloc((size_t)QW3_N_EXPERT * sizeof(float));
    float *gate = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *up = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *cpu_out = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_out = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    int ids[QW3_N_EXPERT_USED];
    float vals[QW3_N_EXPERT_USED];
    memset(ids, 0, sizeof(ids));
    memset(vals, 0, sizeof(vals));

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd, (uint64_t)token, x);
    if (ok) {
        cpu_rmsnorm(ffn_in, x, &e->model, lw->ffn_norm, QW3_N_EMBD);
        ok = cpu_matvec_dense(&e->model, lw->ffn_gate_inp, ffn_in, router);
    }
    if (ok) {
        topk_desc(router, QW3_N_EXPERT, QW3_N_EXPERT_USED, ids, vals);
        ok = cpu_matvec_iq3_s_expert(&e->model, lw->ffn_gate_exps, ids[0], ffn_in, gate) &&
             cpu_matvec_iq3_s_expert(&e->model, lw->ffn_up_exps, ids[0], ffn_in, up);
    }
    if (ok) {
        for (int i = 0; i < QW3_N_FF_EXP; i++) hidden[i] = cpu_silu(gate[i]) * up[i];
        ok = cpu_matvec_iq4_xs_expert(&e->model, lw->ffn_down_exps, ids[0], hidden, cpu_out);
    }
    int gpu_ok = ok &&
        qw3_metal_matvec_iq4_xs_expert(lw->ffn_down_exps->offset, (uint32_t)ids[0],
                                       hidden, QW3_N_FF_EXP, QW3_N_EMBD, gpu_out);

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    double out_rms = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu_out[i] - gpu_out[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * d;
            out_rms += (double)gpu_out[i] * gpu_out[i];
        }
        rmsdiff = sqrt(rmsdiff / QW3_N_EMBD);
        out_rms = sqrt(out_rms / QW3_N_EMBD);
    }
    fprintf(fp,
            "metal moe iq4 down: %s token=%d expert=%d maxdiff=%.7g rmsdiff=%.7g out_rms=%.7g out0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, ids[0], maxdiff, rmsdiff,
            out_rms, gpu_out[0], gpu_out[1], gpu_out[2], gpu_out[3]);

    free(gpu_out);
    free(cpu_out);
    free(hidden);
    free(up);
    free(gate);
    free(router);
    free(ffn_in);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_moe_q6_down_test(qw3_engine *e, int expert, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)expert;
    fprintf(fp, "metal moe q6 down: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal moe q6 down: Metal backend is not initialized\n");
        return -1;
    }
    if (expert < 0 || expert >= QW3_N_EXPERT) {
        fprintf(fp, "metal moe q6 down: expert %d is outside range\n", expert);
        return -1;
    }
    const int il = 34;
    const qw3_layer_weights *lw = &e->weights.layer[il];
    if (lw->ffn_down_exps->type != QW3_TENSOR_Q6_K) {
        fprintf(fp, "metal moe q6 down: layer %d down type is %s, expected q6_K\n",
                il, tensor_type_name(lw->ffn_down_exps->type));
        return -1;
    }

    float *hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *cpu_out = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_out = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    for (int i = 0; i < QW3_N_FF_EXP; i++) {
        int r = (int)((i * 17u + 13u) % 97u) - 48;
        hidden[i] = (float)r * 0.003f;
    }

    bool ok = cpu_matvec_q6_k_expert(&e->model, lw->ffn_down_exps,
                                     expert, hidden, cpu_out);
    const uint64_t q6_block_size = QW3_QK_K / 2 + QW3_QK_K / 4 +
                                   QW3_QK_K / 16 + sizeof(uint16_t);
    const uint64_t q6_row_bytes = (uint64_t)(QW3_N_FF_EXP / QW3_QK_K) *
                                  q6_block_size;
    const uint8_t *q6_row = tensor_data(&e->model, lw->ffn_down_exps) +
                            (uint64_t)expert * QW3_N_EMBD * q6_row_bytes;
    const float cpu_d0 = qw3_f16_to_f32(qw3_load_u16(q6_row + 208));
    const float cpu_d1 = qw3_f16_to_f32(qw3_load_u16(q6_row + q6_block_size + 208));
    int gpu_ok = ok &&
        qw3_metal_matvec_q6_k_expert(lw->ffn_down_exps->offset,
                                     (uint32_t)expert, hidden,
                                     QW3_N_FF_EXP, QW3_N_EMBD, gpu_out);

    float maxdiff = 0.0f;
    double rmsdiff = 0.0, out_rms = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu_out[i] - gpu_out[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * d;
            out_rms += (double)gpu_out[i] * gpu_out[i];
        }
        rmsdiff = sqrt(rmsdiff / QW3_N_EMBD);
        out_rms = sqrt(out_rms / QW3_N_EMBD);
    }
    fprintf(fp,
            "metal moe q6 down: %s layer=%d expert=%d cpu_d=[%.7g %.7g] maxdiff=%.7g rmsdiff=%.7g out_rms=%.7g cpu0=[%.7g %.7g %.7g %.7g] out0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", il, expert, cpu_d0, cpu_d1,
            maxdiff, rmsdiff, out_rms,
            cpu_out[0], cpu_out[1], cpu_out[2], cpu_out[3],
            gpu_out[0], gpu_out[1], gpu_out[2], gpu_out[3]);

    free(gpu_out);
    free(cpu_out);
    free(hidden);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_moe_iq3_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal moe iq3: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal moe iq3: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal moe iq3: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_layer_weights *lw = &e->weights.layer[0];
    if (lw->ffn_gate_inp->type != QW3_TENSOR_F32 ||
        lw->ffn_gate_exps->type != QW3_TENSOR_IQ3_S ||
        lw->ffn_up_exps->type != QW3_TENSOR_IQ3_S) {
        fprintf(fp,
                "metal moe iq3: expected router=f32 gate/up=iq3_s, got router=%s gate=%s up=%s\n",
                tensor_type_name(lw->ffn_gate_inp->type),
                tensor_type_name(lw->ffn_gate_exps->type),
                tensor_type_name(lw->ffn_up_exps->type));
        return -1;
    }

    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *ffn_in = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *router = qw3_xmalloc((size_t)QW3_N_EXPERT * sizeof(float));
    float *cpu_gate = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *cpu_up = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *cpu_hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *gpu_gate = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *gpu_up = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *gpu_hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    int ids[QW3_N_EXPERT_USED];
    float vals[QW3_N_EXPERT_USED];
    memset(ids, 0, sizeof(ids));
    memset(vals, 0, sizeof(vals));

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd, (uint64_t)token, x);
    if (ok) {
        cpu_rmsnorm(ffn_in, x, &e->model, lw->ffn_norm, QW3_N_EMBD);
        ok = cpu_matvec_dense(&e->model, lw->ffn_gate_inp, ffn_in, router);
    }
    if (ok) {
        topk_desc(router, QW3_N_EXPERT, QW3_N_EXPERT_USED, ids, vals);
        ok = cpu_matvec_iq3_s_expert(&e->model, lw->ffn_gate_exps, ids[0], ffn_in, cpu_gate) &&
             cpu_matvec_iq3_s_expert(&e->model, lw->ffn_up_exps, ids[0], ffn_in, cpu_up);
    }
    if (ok) {
        for (int i = 0; i < QW3_N_FF_EXP; i++) cpu_hidden[i] = cpu_silu(cpu_gate[i]) * cpu_up[i];
    }
    int gpu_ok = ok &&
        qw3_metal_matvec_iq3_s_expert(lw->ffn_gate_exps->offset, (uint32_t)ids[0],
                                      ffn_in, QW3_N_EMBD, QW3_N_FF_EXP, gpu_gate) &&
        qw3_metal_matvec_iq3_s_expert(lw->ffn_up_exps->offset, (uint32_t)ids[0],
                                      ffn_in, QW3_N_EMBD, QW3_N_FF_EXP, gpu_up) &&
        qw3_metal_silu_mul(gpu_gate, gpu_up, QW3_N_FF_EXP, gpu_hidden);

    float gate_max = 0.0f, up_max = 0.0f, hidden_max = 0.0f;
    double hidden_rmsdiff = 0.0, hidden_rms = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_FF_EXP; i++) {
            float dg = fabsf(cpu_gate[i] - gpu_gate[i]);
            float du = fabsf(cpu_up[i] - gpu_up[i]);
            float dh = fabsf(cpu_hidden[i] - gpu_hidden[i]);
            if (dg > gate_max) gate_max = dg;
            if (du > up_max) up_max = du;
            if (dh > hidden_max) hidden_max = dh;
            hidden_rmsdiff += (double)dh * dh;
            hidden_rms += (double)gpu_hidden[i] * gpu_hidden[i];
        }
        hidden_rmsdiff = sqrt(hidden_rmsdiff / QW3_N_FF_EXP);
        hidden_rms = sqrt(hidden_rms / QW3_N_FF_EXP);
    }
    fprintf(fp,
            "metal moe iq3: %s token=%d expert=%d gate_max=%.7g up_max=%.7g hidden_max=%.7g hidden_rmsdiff=%.7g hidden_rms=%.7g hidden0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, ids[0], gate_max, up_max,
            hidden_max, hidden_rmsdiff, hidden_rms,
            gpu_hidden[0], gpu_hidden[1], gpu_hidden[2], gpu_hidden[3]);

    free(gpu_hidden);
    free(gpu_up);
    free(gpu_gate);
    free(cpu_hidden);
    free(cpu_up);
    free(cpu_gate);
    free(router);
    free(ffn_in);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_moe_sparse_top1_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal moe sparse top1: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal moe sparse top1: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal moe sparse top1: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_layer_weights *lw = &e->weights.layer[0];
    if (lw->ffn_gate_inp->type != QW3_TENSOR_F32 ||
        lw->ffn_gate_exps->type != QW3_TENSOR_IQ3_S ||
        lw->ffn_up_exps->type != QW3_TENSOR_IQ3_S ||
        lw->ffn_down_exps->type != QW3_TENSOR_IQ4_XS) {
        fprintf(fp,
                "metal moe sparse top1: expected router=f32 gate/up=iq3_s down=iq4_xs, got router=%s gate=%s up=%s down=%s\n",
                tensor_type_name(lw->ffn_gate_inp->type),
                tensor_type_name(lw->ffn_gate_exps->type),
                tensor_type_name(lw->ffn_up_exps->type),
                tensor_type_name(lw->ffn_down_exps->type));
        return -1;
    }

    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *ffn_in = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *router = qw3_xmalloc((size_t)QW3_N_EXPERT * sizeof(float));
    float *cpu_gate = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *cpu_up = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *cpu_hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *cpu_out = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_gate = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *gpu_up = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *gpu_hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *gpu_out = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    int ids[QW3_N_EXPERT_USED];
    float vals[QW3_N_EXPERT_USED];
    memset(ids, 0, sizeof(ids));
    memset(vals, 0, sizeof(vals));

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd, (uint64_t)token, x);
    if (ok) {
        cpu_rmsnorm(ffn_in, x, &e->model, lw->ffn_norm, QW3_N_EMBD);
        ok = cpu_matvec_dense(&e->model, lw->ffn_gate_inp, ffn_in, router);
    }
    if (ok) {
        topk_desc(router, QW3_N_EXPERT, QW3_N_EXPERT_USED, ids, vals);
        ok = cpu_matvec_iq3_s_expert(&e->model, lw->ffn_gate_exps, ids[0], ffn_in, cpu_gate) &&
             cpu_matvec_iq3_s_expert(&e->model, lw->ffn_up_exps, ids[0], ffn_in, cpu_up);
    }
    if (ok) {
        for (int i = 0; i < QW3_N_FF_EXP; i++) cpu_hidden[i] = cpu_silu(cpu_gate[i]) * cpu_up[i];
        ok = cpu_matvec_iq4_xs_expert(&e->model, lw->ffn_down_exps, ids[0], cpu_hidden, cpu_out);
    }

    int gpu_ok = ok &&
        qw3_metal_matvec_iq3_s_expert(lw->ffn_gate_exps->offset, (uint32_t)ids[0],
                                      ffn_in, QW3_N_EMBD, QW3_N_FF_EXP, gpu_gate) &&
        qw3_metal_matvec_iq3_s_expert(lw->ffn_up_exps->offset, (uint32_t)ids[0],
                                      ffn_in, QW3_N_EMBD, QW3_N_FF_EXP, gpu_up) &&
        qw3_metal_silu_mul(gpu_gate, gpu_up, QW3_N_FF_EXP, gpu_hidden) &&
        qw3_metal_matvec_iq4_xs_expert(lw->ffn_down_exps->offset, (uint32_t)ids[0],
                                       gpu_hidden, QW3_N_FF_EXP, QW3_N_EMBD, gpu_out);

    float maxdiff = 0.0f;
    double rmsdiff = 0.0, out_rms = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu_out[i] - gpu_out[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * d;
            out_rms += (double)gpu_out[i] * gpu_out[i];
        }
        rmsdiff = sqrt(rmsdiff / QW3_N_EMBD);
        out_rms = sqrt(out_rms / QW3_N_EMBD);
    }
    fprintf(fp,
            "metal moe sparse top1: %s token=%d expert=%d maxdiff=%.7g rmsdiff=%.7g out_rms=%.7g out0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, ids[0], maxdiff, rmsdiff,
            out_rms, gpu_out[0], gpu_out[1], gpu_out[2], gpu_out[3]);

    free(gpu_out);
    free(gpu_hidden);
    free(gpu_up);
    free(gpu_gate);
    free(cpu_out);
    free(cpu_hidden);
    free(cpu_up);
    free(cpu_gate);
    free(router);
    free(ffn_in);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_moe_sparse_top8_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal moe sparse top8: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal moe sparse top8: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal moe sparse top8: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_layer_weights *lw = &e->weights.layer[0];
    if (lw->ffn_gate_inp->type != QW3_TENSOR_F32 ||
        lw->ffn_gate_exps->type != QW3_TENSOR_IQ3_S ||
        lw->ffn_up_exps->type != QW3_TENSOR_IQ3_S ||
        lw->ffn_down_exps->type != QW3_TENSOR_IQ4_XS) {
        fprintf(fp,
                "metal moe sparse top8: expected router=f32 gate/up=iq3_s down=iq4_xs, got router=%s gate=%s up=%s down=%s\n",
                tensor_type_name(lw->ffn_gate_inp->type),
                tensor_type_name(lw->ffn_gate_exps->type),
                tensor_type_name(lw->ffn_up_exps->type),
                tensor_type_name(lw->ffn_down_exps->type));
        return -1;
    }

    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *ffn_in = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *router = qw3_xmalloc((size_t)QW3_N_EXPERT * sizeof(float));
    float *cpu_gate = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *cpu_up = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *cpu_hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *cpu_down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_out = qw3_xcalloc((size_t)QW3_N_EMBD, sizeof(float));
    float *gpu_gate = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *gpu_up = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *gpu_hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *gpu_down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_out = qw3_xcalloc((size_t)QW3_N_EMBD, sizeof(float));
    int ids[QW3_N_EXPERT_USED];
    float vals[QW3_N_EXPERT_USED];
    float weights[QW3_N_EXPERT_USED];
    memset(ids, 0, sizeof(ids));
    memset(vals, 0, sizeof(vals));
    memset(weights, 0, sizeof(weights));

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd, (uint64_t)token, x);
    if (ok) {
        cpu_rmsnorm(ffn_in, x, &e->model, lw->ffn_norm, QW3_N_EMBD);
        ok = cpu_matvec_dense(&e->model, lw->ffn_gate_inp, ffn_in, router);
    }
    if (ok) {
        topk_desc(router, QW3_N_EXPERT, QW3_N_EXPERT_USED, ids, vals);
        float wsum = 0.0f;
        for (int k = 0; k < QW3_N_EXPERT_USED; k++) {
            weights[k] = expf(vals[k] - vals[0]);
            wsum += weights[k];
        }
        for (int k = 0; k < QW3_N_EXPERT_USED; k++) weights[k] /= wsum;
    }

    int gpu_ok = ok;
    for (int k = 0; ok && k < QW3_N_EXPERT_USED; k++) {
        ok = cpu_matvec_iq3_s_expert(&e->model, lw->ffn_gate_exps, ids[k], ffn_in, cpu_gate) &&
             cpu_matvec_iq3_s_expert(&e->model, lw->ffn_up_exps, ids[k], ffn_in, cpu_up);
        if (ok) {
            for (int i = 0; i < QW3_N_FF_EXP; i++) cpu_hidden[i] = cpu_silu(cpu_gate[i]) * cpu_up[i];
            ok = cpu_matvec_iq4_xs_expert(&e->model, lw->ffn_down_exps, ids[k], cpu_hidden, cpu_down);
        }
        if (ok) {
            for (int i = 0; i < QW3_N_EMBD; i++) cpu_out[i] += weights[k] * cpu_down[i];
        }
    }
    for (int k = 0; gpu_ok && k < QW3_N_EXPERT_USED; k++) {
        gpu_ok =
            qw3_metal_matvec_iq3_s_expert(lw->ffn_gate_exps->offset, (uint32_t)ids[k],
                                          ffn_in, QW3_N_EMBD, QW3_N_FF_EXP, gpu_gate) &&
            qw3_metal_matvec_iq3_s_expert(lw->ffn_up_exps->offset, (uint32_t)ids[k],
                                          ffn_in, QW3_N_EMBD, QW3_N_FF_EXP, gpu_up) &&
            qw3_metal_silu_mul(gpu_gate, gpu_up, QW3_N_FF_EXP, gpu_hidden) &&
            qw3_metal_matvec_iq4_xs_expert(lw->ffn_down_exps->offset, (uint32_t)ids[k],
                                           gpu_hidden, QW3_N_FF_EXP, QW3_N_EMBD, gpu_down);
        if (gpu_ok) {
            for (int i = 0; i < QW3_N_EMBD; i++) gpu_out[i] += weights[k] * gpu_down[i];
        }
    }
    gpu_ok = gpu_ok && ok;

    float maxdiff = 0.0f;
    double rmsdiff = 0.0, out_rms = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu_out[i] - gpu_out[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * d;
            out_rms += (double)gpu_out[i] * gpu_out[i];
        }
        rmsdiff = sqrt(rmsdiff / QW3_N_EMBD);
        out_rms = sqrt(out_rms / QW3_N_EMBD);
    }
    fprintf(fp,
            "metal moe sparse top8: %s token=%d experts=%d,%d,%d,%d,%d,%d,%d,%d w0=[%.7g %.7g %.7g %.7g] maxdiff=%.7g rmsdiff=%.7g out_rms=%.7g out0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token,
            ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7],
            weights[0], weights[1], weights[2], weights[3],
            maxdiff, rmsdiff, out_rms,
            gpu_out[0], gpu_out[1], gpu_out[2], gpu_out[3]);

    free(gpu_out);
    free(gpu_down);
    free(gpu_hidden);
    free(gpu_up);
    free(gpu_gate);
    free(cpu_out);
    free(cpu_down);
    free(cpu_hidden);
    free(cpu_up);
    free(cpu_gate);
    free(router);
    free(ffn_in);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_moe_layer_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal moe layer: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal moe layer: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal moe layer: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_layer_weights *lw = &e->weights.layer[0];
    if (lw->ffn_gate_inp->type != QW3_TENSOR_F32 ||
        !lw->ffn_gate_inp_shexp ||
        lw->ffn_gate_exps->type != QW3_TENSOR_IQ3_S ||
        lw->ffn_up_exps->type != QW3_TENSOR_IQ3_S ||
        lw->ffn_down_exps->type != QW3_TENSOR_IQ4_XS ||
        lw->ffn_gate_shared->type != QW3_TENSOR_Q8_0 ||
        lw->ffn_up_shared->type != QW3_TENSOR_Q8_0 ||
        lw->ffn_down_shared->type != QW3_TENSOR_Q8_0) {
        fprintf(fp,
                "metal moe layer: expected router=f32 shexp=f32 sparse=iq3_s/iq4_xs shared=q8_0, got router=%s sparse=%s/%s/%s shared=%s/%s/%s\n",
                tensor_type_name(lw->ffn_gate_inp->type),
                tensor_type_name(lw->ffn_gate_exps->type),
                tensor_type_name(lw->ffn_up_exps->type),
                tensor_type_name(lw->ffn_down_exps->type),
                tensor_type_name(lw->ffn_gate_shared->type),
                tensor_type_name(lw->ffn_up_shared->type),
                tensor_type_name(lw->ffn_down_shared->type));
        return -1;
    }

    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *ffn_in = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *router = qw3_xmalloc((size_t)QW3_N_EXPERT * sizeof(float));
    float *cpu_gate = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *cpu_up = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *cpu_hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *cpu_down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_sparse = qw3_xcalloc((size_t)QW3_N_EMBD, sizeof(float));
    float *cpu_sh_gate = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *cpu_sh_up = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *cpu_sh_hidden = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *cpu_shared = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_out = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));

    float *gpu_gate = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *gpu_up = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *gpu_hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *gpu_down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_sparse = qw3_xcalloc((size_t)QW3_N_EMBD, sizeof(float));
    float *gpu_sh_gate = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *gpu_sh_up = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *gpu_sh_hidden = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *gpu_shared_down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_shared = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_out = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));

    int ids[QW3_N_EXPERT_USED];
    float vals[QW3_N_EXPERT_USED];
    float weights[QW3_N_EXPERT_USED];
    memset(ids, 0, sizeof(ids));
    memset(vals, 0, sizeof(vals));
    memset(weights, 0, sizeof(weights));
    float cpu_shared_raw = 0.0f;
    float gpu_shared_raw = 0.0f;

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd, (uint64_t)token, x);
    if (ok) {
        cpu_rmsnorm(ffn_in, x, &e->model, lw->ffn_norm, QW3_N_EMBD);
        ok = cpu_matvec_dense(&e->model, lw->ffn_gate_inp, ffn_in, router);
    }
    if (ok) {
        topk_desc(router, QW3_N_EXPERT, QW3_N_EXPERT_USED, ids, vals);
        float wsum = 0.0f;
        for (int k = 0; k < QW3_N_EXPERT_USED; k++) {
            weights[k] = expf(vals[k] - vals[0]);
            wsum += weights[k];
        }
        for (int k = 0; k < QW3_N_EXPERT_USED; k++) weights[k] /= wsum;
    }
    for (int k = 0; ok && k < QW3_N_EXPERT_USED; k++) {
        ok = cpu_matvec_iq3_s_expert(&e->model, lw->ffn_gate_exps, ids[k], ffn_in, cpu_gate) &&
             cpu_matvec_iq3_s_expert(&e->model, lw->ffn_up_exps, ids[k], ffn_in, cpu_up);
        if (ok) {
            for (int i = 0; i < QW3_N_FF_EXP; i++) cpu_hidden[i] = cpu_silu(cpu_gate[i]) * cpu_up[i];
            ok = cpu_matvec_iq4_xs_expert(&e->model, lw->ffn_down_exps, ids[k], cpu_hidden, cpu_down);
        }
        if (ok) {
            for (int i = 0; i < QW3_N_EMBD; i++) cpu_sparse[i] += weights[k] * cpu_down[i];
        }
    }
    if (ok) {
        ok = cpu_matvec_q8_0(&e->model, lw->ffn_gate_shared, ffn_in, cpu_sh_gate) &&
             cpu_matvec_q8_0(&e->model, lw->ffn_up_shared, ffn_in, cpu_sh_up) &&
             cpu_dot_dense_1d(&e->model, lw->ffn_gate_inp_shexp, ffn_in, &cpu_shared_raw);
    }
    if (ok) {
        for (int i = 0; i < QW3_N_FF_SHARED; i++) {
            cpu_sh_hidden[i] = cpu_silu(cpu_sh_gate[i]) * cpu_sh_up[i];
        }
        ok = cpu_matvec_q8_0(&e->model, lw->ffn_down_shared, cpu_sh_hidden, cpu_shared);
    }
    const float cpu_shared_gate = 1.0f / (1.0f + expf(-cpu_shared_raw));
    if (ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            cpu_shared[i] *= cpu_shared_gate;
            cpu_out[i] = cpu_sparse[i] + cpu_shared[i];
        }
    }

    int gpu_ok = ok;
    for (int k = 0; gpu_ok && k < QW3_N_EXPERT_USED; k++) {
        gpu_ok =
            qw3_metal_matvec_iq3_s_expert(lw->ffn_gate_exps->offset, (uint32_t)ids[k],
                                          ffn_in, QW3_N_EMBD, QW3_N_FF_EXP, gpu_gate) &&
            qw3_metal_matvec_iq3_s_expert(lw->ffn_up_exps->offset, (uint32_t)ids[k],
                                          ffn_in, QW3_N_EMBD, QW3_N_FF_EXP, gpu_up) &&
            qw3_metal_silu_mul(gpu_gate, gpu_up, QW3_N_FF_EXP, gpu_hidden) &&
            qw3_metal_matvec_iq4_xs_expert(lw->ffn_down_exps->offset, (uint32_t)ids[k],
                                           gpu_hidden, QW3_N_FF_EXP, QW3_N_EMBD, gpu_down);
        if (gpu_ok) {
            for (int i = 0; i < QW3_N_EMBD; i++) gpu_sparse[i] += weights[k] * gpu_down[i];
        }
    }
    gpu_ok = gpu_ok &&
        qw3_metal_matvec_q8_0(lw->ffn_gate_shared->offset, ffn_in,
                              QW3_N_EMBD, QW3_N_FF_SHARED, gpu_sh_gate) &&
        qw3_metal_matvec_q8_0(lw->ffn_up_shared->offset, ffn_in,
                              QW3_N_EMBD, QW3_N_FF_SHARED, gpu_sh_up) &&
        qw3_metal_silu_mul(gpu_sh_gate, gpu_sh_up, QW3_N_FF_SHARED, gpu_sh_hidden) &&
        qw3_metal_matvec_q8_0(lw->ffn_down_shared->offset, gpu_sh_hidden,
                              QW3_N_FF_SHARED, QW3_N_EMBD, gpu_shared_down) &&
        qw3_metal_matvec_f32(lw->ffn_gate_inp_shexp->offset, ffn_in,
                             QW3_N_EMBD, 1, &gpu_shared_raw);
    const float gpu_shared_gate = 1.0f / (1.0f + expf(-gpu_shared_raw));
    if (gpu_ok) {
        gpu_ok = qw3_metal_scale(gpu_shared_down, QW3_N_EMBD, gpu_shared_gate, gpu_shared);
    }
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) gpu_out[i] = gpu_sparse[i] + gpu_shared[i];
    }

    float maxdiff = 0.0f, sparse_max = 0.0f, shared_max = 0.0f;
    double rmsdiff = 0.0, out_rms = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float ds = fabsf(cpu_sparse[i] - gpu_sparse[i]);
            float dh = fabsf(cpu_shared[i] - gpu_shared[i]);
            float d = fabsf(cpu_out[i] - gpu_out[i]);
            if (ds > sparse_max) sparse_max = ds;
            if (dh > shared_max) shared_max = dh;
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * d;
            out_rms += (double)gpu_out[i] * gpu_out[i];
        }
        rmsdiff = sqrt(rmsdiff / QW3_N_EMBD);
        out_rms = sqrt(out_rms / QW3_N_EMBD);
    }
    fprintf(fp,
            "metal moe layer: %s token=%d experts=%d,%d,%d,%d,%d,%d,%d,%d gate=%.7g maxdiff=%.7g sparse_max=%.7g shared_max=%.7g rmsdiff=%.7g out_rms=%.7g out0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token,
            ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7],
            gpu_shared_gate, maxdiff, sparse_max, shared_max, rmsdiff, out_rms,
            gpu_out[0], gpu_out[1], gpu_out[2], gpu_out[3]);

    free(gpu_out);
    free(gpu_shared);
    free(gpu_shared_down);
    free(gpu_sh_hidden);
    free(gpu_sh_up);
    free(gpu_sh_gate);
    free(gpu_sparse);
    free(gpu_down);
    free(gpu_hidden);
    free(gpu_up);
    free(gpu_gate);
    free(cpu_out);
    free(cpu_shared);
    free(cpu_sh_hidden);
    free(cpu_sh_up);
    free(cpu_sh_gate);
    free(cpu_sparse);
    free(cpu_down);
    free(cpu_hidden);
    free(cpu_up);
    free(cpu_gate);
    free(router);
    free(ffn_in);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_moe_real_layer_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal moe real layer: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal moe real layer: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal moe real layer: token %d is outside vocab\n", token);
        return -1;
    }

    const qw3_layer_weights *lw = &e->weights.layer[0];
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t inner_n = QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t state_n = QW3_N_LINEAR_V_HEADS *
                             QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM;

    float *x_cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn_cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv_cpu = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *z_cpu = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *alpha_cpu = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta_cpu = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *conv_cpu = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *q_cpu = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *k_cpu = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *core_cpu = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *inner_cpu = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *attn_cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *resid_cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *ffn_cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *moe_cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *layer_cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *state_in = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *state_cpu = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *conv_state = qw3_xcalloc((size_t)n_qkv * (QW3_N_LINEAR_CONV_K - 1),
                                    sizeof(float));

    float *x_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv_gpu = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *z_gpu = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *alpha_gpu = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta_gpu = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta_sig = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *gamma = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *conv_gpu = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *q_gpu = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *k_gpu = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *core_gpu = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *inner_gpu = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *attn_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *ffn_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *state_gpu = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *router_gpu = qw3_xmalloc((size_t)QW3_N_EXPERT * sizeof(float));

    float *gate = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *up = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *sparse_gpu = qw3_xcalloc((size_t)QW3_N_EMBD, sizeof(float));
    float *sh_gate = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *sh_up = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *sh_hidden = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *shared_down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *shared_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *moe_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *layer_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));

    for (uint32_t i = 0; i < state_n; i++) {
        int r = (int)((i * 17u + 13u) % 97u) - 48;
        state_in[i] = (float)r * 0.0007f;
    }

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd,
                                    (uint64_t)token, x_cpu);
    if (ok) {
        cpu_rmsnorm(xn_cpu, x_cpu, &e->model, lw->attn_norm, QW3_N_EMBD);
        ok = cpu_matvec_q8_0(&e->model, lw->linear_qkv_proj, xn_cpu, qkv_cpu) &&
             cpu_matvec_q8_0(&e->model, lw->linear_gate_proj, xn_cpu, z_cpu) &&
             cpu_matvec_dense(&e->model, lw->linear_ssm_alpha, xn_cpu, alpha_cpu) &&
             cpu_matvec_dense(&e->model, lw->linear_ssm_beta, xn_cpu, beta_cpu) &&
             cpu_deltanet_conv1d_step(&e->model, lw, qkv_cpu, conv_state, conv_cpu);
    }
    if (ok) {
        const float *qraw = conv_cpu;
        const float *kraw = conv_cpu + qk_n;
        for (int h = 0; h < QW3_N_LINEAR_QK_HEADS; h++) {
            cpu_l2_norm_head(q_cpu + h * QW3_N_LINEAR_HEAD_DIM,
                             qraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
            cpu_l2_norm_head(k_cpu + h * QW3_N_LINEAR_HEAD_DIM,
                             kraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            const int hk = hv % QW3_N_LINEAR_QK_HEADS;
            const float *qh = q_cpu + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *kh = k_cpu + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *vh = conv_cpu + qk_n * 2 + hv * QW3_N_LINEAR_HEAD_DIM;
            const float bh = 1.0f / (1.0f + expf(-beta_cpu[hv]));
            const float ah = cpu_softplus(alpha_cpu[hv] +
                tensor_read_dense_1d(&e->model, lw->linear_ssm_dt_bias, (uint64_t)hv));
            const float gh = expf(ah *
                tensor_read_dense_1d(&e->model, lw->linear_ssm_a, (uint64_t)hv));
            float *sh_out = state_cpu + (uint64_t)hv * QW3_N_LINEAR_HEAD_DIM *
                            QW3_N_LINEAR_HEAD_DIM;
            const float *sh_in = state_in + (uint64_t)hv * QW3_N_LINEAR_HEAD_DIM *
                                 QW3_N_LINEAR_HEAD_DIM;
            float *oh = core_cpu + hv * QW3_N_LINEAR_HEAD_DIM;
            for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
                float sk = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    sk += sh_in[i * QW3_N_LINEAR_HEAD_DIM + j] * gh * kh[i];
                }
                const float d = bh * (vh[j] - sk);
                float out = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    const int idx = i * QW3_N_LINEAR_HEAD_DIM + j;
                    const float sv = sh_in[idx] * gh + kh[i] * d;
                    sh_out[idx] = sv;
                    out += sv * qh[i];
                }
                oh[j] = out / sqrtf((float)QW3_N_LINEAR_HEAD_DIM);
            }
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            float *dst = inner_cpu + hv * QW3_N_LINEAR_HEAD_DIM;
            const float *src = core_cpu + hv * QW3_N_LINEAR_HEAD_DIM;
            const float *zh = z_cpu + hv * QW3_N_LINEAR_HEAD_DIM;
            double ss = 0.0;
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) ss += (double)src[i] * src[i];
            const float scale = 1.0f /
                sqrtf((float)(ss / QW3_N_LINEAR_HEAD_DIM) + QW3_RMS_EPS);
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                dst[i] = src[i] * scale *
                         tensor_read_dense_1d(&e->model, lw->linear_ssm_norm, (uint64_t)i) *
                         cpu_silu(zh[i]);
            }
        }
        ok = cpu_matvec_q8_0(&e->model, lw->linear_ssm_out, inner_cpu, attn_cpu);
    }
    if (ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) resid_cpu[i] = x_cpu[i] + attn_cpu[i];
        cpu_rmsnorm(ffn_cpu, resid_cpu, &e->model, lw->ffn_norm, QW3_N_EMBD);
        ok = cpu_moe_layer(e, 0, ffn_cpu, moe_cpu, NULL, NULL);
        if (ok) {
            for (int i = 0; i < QW3_N_EMBD; i++) layer_cpu[i] = resid_cpu[i] + moe_cpu[i];
        }
    }

    int gpu_ok = ok &&
        qw3_metal_embed_q8_0(e->weights.token_embd->offset, (uint32_t)token,
                             QW3_N_EMBD, x_gpu) &&
        qw3_metal_rmsnorm_weight_f32(x_gpu, lw->attn_norm->offset,
                                     xn_gpu, QW3_N_EMBD, QW3_RMS_EPS) &&
        qw3_metal_matvec_q8_0(lw->linear_qkv_proj->offset, xn_gpu,
                              QW3_N_EMBD, n_qkv, qkv_gpu) &&
        qw3_metal_matvec_q8_0(lw->linear_gate_proj->offset, xn_gpu,
                              QW3_N_EMBD, inner_n, z_gpu) &&
        qw3_metal_matvec_f32(lw->linear_ssm_alpha->offset, xn_gpu,
                             QW3_N_EMBD, QW3_N_LINEAR_V_HEADS, alpha_gpu) &&
        qw3_metal_matvec_f32(lw->linear_ssm_beta->offset, xn_gpu,
                             QW3_N_EMBD, QW3_N_LINEAR_V_HEADS, beta_gpu) &&
        qw3_metal_deltanet_conv1d_zero(lw->linear_conv_weight->offset,
                                       qkv_gpu, n_qkv, conv_gpu) &&
        qw3_metal_l2norm_heads(conv_gpu, QW3_N_LINEAR_QK_HEADS,
                               QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS, q_gpu) &&
        qw3_metal_l2norm_heads(conv_gpu + qk_n, QW3_N_LINEAR_QK_HEADS,
                               QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS, k_gpu);
    if (gpu_ok) {
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            beta_sig[hv] = 1.0f / (1.0f + expf(-beta_gpu[hv]));
            const float ah = cpu_softplus(alpha_gpu[hv] +
                tensor_read_dense_1d(&e->model, lw->linear_ssm_dt_bias, (uint64_t)hv));
            gamma[hv] = expf(ah *
                tensor_read_dense_1d(&e->model, lw->linear_ssm_a, (uint64_t)hv));
        }
        gpu_ok =
            qw3_metal_deltanet_recur(state_in, q_gpu, k_gpu, conv_gpu + qk_n * 2,
                                     beta_sig, gamma,
                                     QW3_N_LINEAR_QK_HEADS,
                                     QW3_N_LINEAR_V_HEADS,
                                     QW3_N_LINEAR_HEAD_DIM,
                                     state_gpu, core_gpu) &&
            qw3_metal_deltanet_gated_rmsnorm(lw->linear_ssm_norm->offset,
                                             core_gpu, z_gpu,
                                             QW3_N_LINEAR_V_HEADS,
                                             QW3_N_LINEAR_HEAD_DIM,
                                             QW3_RMS_EPS, inner_gpu) &&
            qw3_metal_matvec_q8_0(lw->linear_ssm_out->offset, inner_gpu,
                                  inner_n, QW3_N_EMBD, attn_gpu) &&
            qw3_metal_residual_rmsnorm_weight_f32(x_gpu, attn_gpu,
                                                  lw->ffn_norm->offset,
                                                  ffn_gpu, QW3_N_EMBD,
                                                  QW3_RMS_EPS) &&
            qw3_metal_matvec_f32(lw->ffn_gate_inp->offset, ffn_gpu,
                                 QW3_N_EMBD, QW3_N_EXPERT, router_gpu);
    }

    int ids[QW3_N_EXPERT_USED], cpu_ids[QW3_N_EXPERT_USED];
    float vals[QW3_N_EXPERT_USED], weights[QW3_N_EXPERT_USED];
    memset(ids, 0, sizeof(ids));
    memset(cpu_ids, 0, sizeof(cpu_ids));
    memset(vals, 0, sizeof(vals));
    memset(weights, 0, sizeof(weights));
    int top_match = 1;
    if (gpu_ok) {
        float cpu_router[QW3_N_EXPERT];
        float cpu_vals[QW3_N_EXPERT_USED];
        if (!cpu_matvec_dense(&e->model, lw->ffn_gate_inp, ffn_cpu, cpu_router)) {
            gpu_ok = 0;
        } else {
            topk_desc(cpu_router, QW3_N_EXPERT, QW3_N_EXPERT_USED, cpu_ids, cpu_vals);
            topk_desc(router_gpu, QW3_N_EXPERT, QW3_N_EXPERT_USED, ids, vals);
            float wsum = 0.0f;
            for (int k = 0; k < QW3_N_EXPERT_USED; k++) {
                if (ids[k] != cpu_ids[k]) top_match = 0;
                weights[k] = expf(vals[k] - vals[0]);
                wsum += weights[k];
            }
            for (int k = 0; k < QW3_N_EXPERT_USED; k++) weights[k] /= wsum;
        }
    }
    for (int k = 0; gpu_ok && k < QW3_N_EXPERT_USED; k++) {
        gpu_ok =
            qw3_metal_matvec_iq3_s_expert(lw->ffn_gate_exps->offset, (uint32_t)ids[k],
                                          ffn_gpu, QW3_N_EMBD, QW3_N_FF_EXP, gate) &&
            qw3_metal_matvec_iq3_s_expert(lw->ffn_up_exps->offset, (uint32_t)ids[k],
                                          ffn_gpu, QW3_N_EMBD, QW3_N_FF_EXP, up) &&
            qw3_metal_silu_mul(gate, up, QW3_N_FF_EXP, hidden) &&
            qw3_metal_matvec_iq4_xs_expert(lw->ffn_down_exps->offset, (uint32_t)ids[k],
                                           hidden, QW3_N_FF_EXP, QW3_N_EMBD, down);
        if (gpu_ok) {
            for (int i = 0; i < QW3_N_EMBD; i++) sparse_gpu[i] += weights[k] * down[i];
        }
    }
    gpu_ok = gpu_ok &&
        qw3_metal_matvec_q8_0(lw->ffn_gate_shared->offset, ffn_gpu,
                              QW3_N_EMBD, QW3_N_FF_SHARED, sh_gate) &&
        qw3_metal_matvec_q8_0(lw->ffn_up_shared->offset, ffn_gpu,
                              QW3_N_EMBD, QW3_N_FF_SHARED, sh_up) &&
        qw3_metal_silu_mul(sh_gate, sh_up, QW3_N_FF_SHARED, sh_hidden) &&
        qw3_metal_matvec_q8_0(lw->ffn_down_shared->offset, sh_hidden,
                              QW3_N_FF_SHARED, QW3_N_EMBD, shared_down);
    float shared_raw = 0.0f;
    if (gpu_ok) {
        gpu_ok = qw3_metal_matvec_f32(lw->ffn_gate_inp_shexp->offset, ffn_gpu,
                                      QW3_N_EMBD, 1, &shared_raw);
    }
    const float shared_gate = 1.0f / (1.0f + expf(-shared_raw));
    if (gpu_ok) {
        gpu_ok = qw3_metal_scale(shared_down, QW3_N_EMBD, shared_gate, shared_gpu);
    }
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) moe_gpu[i] = sparse_gpu[i] + shared_gpu[i];
        for (int i = 0; i < QW3_N_EMBD; i++) layer_gpu[i] = x_gpu[i] + attn_gpu[i] + moe_gpu[i];
    }

    float ffn_max = 0.0f, moe_max = 0.0f, layer_max = 0.0f;
    double moe_rmsdiff = 0.0, layer_rmsdiff = 0.0, layer_rms = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float df = fabsf(ffn_cpu[i] - ffn_gpu[i]);
            float dm = fabsf(moe_cpu[i] - moe_gpu[i]);
            float dl = fabsf(layer_cpu[i] - layer_gpu[i]);
            if (df > ffn_max) ffn_max = df;
            if (dm > moe_max) moe_max = dm;
            if (dl > layer_max) layer_max = dl;
            moe_rmsdiff += (double)dm * dm;
            layer_rmsdiff += (double)dl * dl;
            layer_rms += (double)layer_gpu[i] * layer_gpu[i];
        }
        moe_rmsdiff = sqrt(moe_rmsdiff / QW3_N_EMBD);
        layer_rmsdiff = sqrt(layer_rmsdiff / QW3_N_EMBD);
        layer_rms = sqrt(layer_rms / QW3_N_EMBD);
    }
    fprintf(fp,
            "metal moe real layer: %s token=%d top_match=%s experts=%d,%d,%d,%d,%d,%d,%d,%d ffn_max=%.7g moe_max=%.7g layer_max=%.7g moe_rmsdiff=%.7g layer_rmsdiff=%.7g layer_rms=%.7g out0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, top_match ? "yes" : "no",
            ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7],
            ffn_max, moe_max, layer_max, moe_rmsdiff, layer_rmsdiff, layer_rms,
            layer_gpu[0], layer_gpu[1], layer_gpu[2], layer_gpu[3]);

    free(layer_gpu);
    free(moe_gpu);
    free(shared_gpu);
    free(shared_down);
    free(sh_hidden);
    free(sh_up);
    free(sh_gate);
    free(sparse_gpu);
    free(down);
    free(hidden);
    free(up);
    free(gate);
    free(router_gpu);
    free(state_gpu);
    free(ffn_gpu);
    free(attn_gpu);
    free(inner_gpu);
    free(core_gpu);
    free(k_gpu);
    free(q_gpu);
    free(conv_gpu);
    free(gamma);
    free(beta_sig);
    free(beta_gpu);
    free(alpha_gpu);
    free(z_gpu);
    free(qkv_gpu);
    free(xn_gpu);
    free(x_gpu);
    free(conv_state);
    free(state_cpu);
    free(state_in);
    free(moe_cpu);
    free(layer_cpu);
    free(ffn_cpu);
    free(resid_cpu);
    free(attn_cpu);
    free(inner_cpu);
    free(core_cpu);
    free(k_cpu);
    free(q_cpu);
    free(conv_cpu);
    free(beta_cpu);
    free(alpha_cpu);
    free(z_cpu);
    free(qkv_cpu);
    free(xn_cpu);
    free(x_cpu);
    return gpu_ok && top_match ? 0 : -1;
#endif
}

int qw3_engine_metal_deltanet3_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal deltanet3: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal deltanet3: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal deltanet3: token %d is outside vocab\n", token);
        return -1;
    }
    for (int il = 0; il < 3; il++) {
        if (qw3_layer_is_full_attention((uint32_t)il)) {
            fprintf(fp, "metal deltanet3: layer %d is not a DeltaNet layer\n", il);
            return -1;
        }
    }

    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t inner_n = QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t state_n = QW3_N_LINEAR_V_HEADS *
                             QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t conv_state_n = n_qkv * (QW3_N_LINEAR_CONV_K - 1);

    float *cpu_a = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_b = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_a = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_b = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_conv_state = qw3_xcalloc((size_t)3 * conv_state_n, sizeof(float));
    float *cpu_state = qw3_xcalloc((size_t)3 * state_n, sizeof(float));

    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *z = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *alpha = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta_sig = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *q = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *k = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *state_out = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *core = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *inner = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *attn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *ffn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *router = qw3_xmalloc((size_t)QW3_N_EXPERT * sizeof(float));

    float *gate = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *up = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *sparse = qw3_xcalloc((size_t)QW3_N_EMBD, sizeof(float));
    float *sh_gate = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *sh_up = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *sh_hidden = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *shared_down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *shared = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd,
                                    (uint64_t)token, cpu_a);
    int gpu_ok = ok &&
        qw3_metal_embed_q8_0(e->weights.token_embd->offset, (uint32_t)token,
                             QW3_N_EMBD, gpu_a);

    for (int il = 0; ok && il < 3; il++) {
        ok = cpu_deltanet_layer(e, il, cpu_a,
                                cpu_conv_state + (uint64_t)il * conv_state_n,
                                cpu_state + (uint64_t)il * state_n,
                                cpu_b);
        float *tmp = cpu_a;
        cpu_a = cpu_b;
        cpu_b = tmp;
    }

    int ids[QW3_N_EXPERT_USED];
    float vals[QW3_N_EXPERT_USED];
    float weights[QW3_N_EXPERT_USED];
    memset(ids, 0, sizeof(ids));
    memset(vals, 0, sizeof(vals));
    memset(weights, 0, sizeof(weights));
    int last_top0 = -1;

    for (int il = 0; gpu_ok && il < 3; il++) {
        const qw3_layer_weights *lw = &e->weights.layer[il];
        gpu_ok =
            qw3_metal_rmsnorm_weight_f32(gpu_a, lw->attn_norm->offset,
                                         xn, QW3_N_EMBD, QW3_RMS_EPS) &&
            qw3_metal_matvec_q8_0(lw->linear_qkv_proj->offset, xn,
                                  QW3_N_EMBD, n_qkv, qkv) &&
            qw3_metal_matvec_q8_0(lw->linear_gate_proj->offset, xn,
                                  QW3_N_EMBD, inner_n, z) &&
            qw3_metal_matvec_f32(lw->linear_ssm_alpha->offset, xn,
                                 QW3_N_EMBD, QW3_N_LINEAR_V_HEADS, alpha) &&
            qw3_metal_matvec_f32(lw->linear_ssm_beta->offset, xn,
                                 QW3_N_EMBD, QW3_N_LINEAR_V_HEADS, beta) &&
            qw3_metal_deltanet_conv1d_zero(lw->linear_conv_weight->offset,
                                           qkv, n_qkv, conv) &&
            qw3_metal_l2norm_heads(conv, QW3_N_LINEAR_QK_HEADS,
                                   QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS, q) &&
            qw3_metal_l2norm_heads(conv + qk_n, QW3_N_LINEAR_QK_HEADS,
                                   QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS, k);
        if (gpu_ok) {
            for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
                beta_sig[hv] = 1.0f / (1.0f + expf(-beta[hv]));
            }
            gpu_ok =
                qw3_metal_deltanet_recur_zero(q, k, conv + qk_n * 2,
                                              beta_sig,
                                              QW3_N_LINEAR_QK_HEADS,
                                              QW3_N_LINEAR_V_HEADS,
                                              QW3_N_LINEAR_HEAD_DIM,
                                              state_out, core) &&
                qw3_metal_deltanet_gated_rmsnorm(lw->linear_ssm_norm->offset,
                                                 core, z,
                                                 QW3_N_LINEAR_V_HEADS,
                                                 QW3_N_LINEAR_HEAD_DIM,
                                                 QW3_RMS_EPS, inner) &&
                qw3_metal_matvec_q8_0(lw->linear_ssm_out->offset, inner,
                                      inner_n, QW3_N_EMBD, attn) &&
                qw3_metal_residual_rmsnorm_weight_f32(gpu_a, attn,
                                                      lw->ffn_norm->offset,
                                                      ffn, QW3_N_EMBD,
                                                      QW3_RMS_EPS) &&
                qw3_metal_matvec_f32(lw->ffn_gate_inp->offset, ffn,
                                     QW3_N_EMBD, QW3_N_EXPERT, router);
        }
        if (gpu_ok) {
            topk_desc(router, QW3_N_EXPERT, QW3_N_EXPERT_USED, ids, vals);
            last_top0 = ids[0];
            float wsum = 0.0f;
            for (int kk = 0; kk < QW3_N_EXPERT_USED; kk++) {
                weights[kk] = expf(vals[kk] - vals[0]);
                wsum += weights[kk];
            }
            for (int kk = 0; kk < QW3_N_EXPERT_USED; kk++) weights[kk] /= wsum;
            memset(sparse, 0, (size_t)QW3_N_EMBD * sizeof(float));
        }
        for (int kk = 0; gpu_ok && kk < QW3_N_EXPERT_USED; kk++) {
            gpu_ok =
                qw3_metal_matvec_iq3_s_expert(lw->ffn_gate_exps->offset, (uint32_t)ids[kk],
                                              ffn, QW3_N_EMBD, QW3_N_FF_EXP, gate) &&
                qw3_metal_matvec_iq3_s_expert(lw->ffn_up_exps->offset, (uint32_t)ids[kk],
                                              ffn, QW3_N_EMBD, QW3_N_FF_EXP, up) &&
                qw3_metal_silu_mul(gate, up, QW3_N_FF_EXP, hidden) &&
                qw3_metal_matvec_iq4_xs_expert(lw->ffn_down_exps->offset, (uint32_t)ids[kk],
                                               hidden, QW3_N_FF_EXP, QW3_N_EMBD, down);
            if (gpu_ok) {
                for (int i = 0; i < QW3_N_EMBD; i++) sparse[i] += weights[kk] * down[i];
            }
        }
        gpu_ok = gpu_ok &&
            qw3_metal_matvec_q8_0(lw->ffn_gate_shared->offset, ffn,
                                  QW3_N_EMBD, QW3_N_FF_SHARED, sh_gate) &&
            qw3_metal_matvec_q8_0(lw->ffn_up_shared->offset, ffn,
                                  QW3_N_EMBD, QW3_N_FF_SHARED, sh_up) &&
            qw3_metal_silu_mul(sh_gate, sh_up, QW3_N_FF_SHARED, sh_hidden) &&
            qw3_metal_matvec_q8_0(lw->ffn_down_shared->offset, sh_hidden,
                                  QW3_N_FF_SHARED, QW3_N_EMBD, shared_down);
        float shared_raw = 0.0f;
        if (gpu_ok) {
            gpu_ok = qw3_metal_matvec_f32(lw->ffn_gate_inp_shexp->offset, ffn,
                                          QW3_N_EMBD, 1, &shared_raw);
        }
        const float shared_gate = 1.0f / (1.0f + expf(-shared_raw));
        if (gpu_ok) {
            gpu_ok = qw3_metal_scale(shared_down, QW3_N_EMBD, shared_gate, shared);
        }
        if (gpu_ok) {
            for (int i = 0; i < QW3_N_EMBD; i++) {
                gpu_b[i] = gpu_a[i] + attn[i] + sparse[i] + shared[i];
            }
            float *tmp = gpu_a;
            gpu_a = gpu_b;
            gpu_b = tmp;
        }
    }

    float maxdiff = 0.0f;
    double rmsdiff = 0.0, out_rms = 0.0;
    if (gpu_ok && ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu_a[i] - gpu_a[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * d;
            out_rms += (double)gpu_a[i] * gpu_a[i];
        }
        rmsdiff = sqrt(rmsdiff / QW3_N_EMBD);
        out_rms = sqrt(out_rms / QW3_N_EMBD);
    }
    fprintf(fp,
            "metal deltanet3: %s token=%d layers=0,1,2 maxdiff=%.7g rmsdiff=%.7g out_rms=%.7g last_top0=%d out0=[%.7g %.7g %.7g %.7g]\n",
            (gpu_ok && ok) ? "ok" : "failed", token,
            maxdiff, rmsdiff, out_rms, last_top0,
            gpu_a[0], gpu_a[1], gpu_a[2], gpu_a[3]);

    free(shared);
    free(shared_down);
    free(sh_hidden);
    free(sh_up);
    free(sh_gate);
    free(sparse);
    free(down);
    free(hidden);
    free(up);
    free(gate);
    free(router);
    free(ffn);
    free(attn);
    free(inner);
    free(core);
    free(state_out);
    free(k);
    free(q);
    free(conv);
    free(beta_sig);
    free(beta);
    free(alpha);
    free(z);
    free(qkv);
    free(xn);
    free(cpu_state);
    free(cpu_conv_state);
    free(gpu_b == gpu_a ? NULL : gpu_b);
    free(gpu_a);
    free(cpu_b == cpu_a ? NULL : cpu_b);
    free(cpu_a);
    return (gpu_ok && ok) ? 0 : -1;
#endif
}

static int qw3_engine_metal_mixed_n_hidden(qw3_engine *e, int token,
                                           int n_layers, const char *name,
                                           float *cpu_hidden_out,
                                           float *gpu_hidden_out,
                                           FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token; (void)n_layers; (void)name;
    (void)cpu_hidden_out; (void)gpu_hidden_out;
    fprintf(fp, "%s: unavailable in QW3_NO_METAL build\n", name);
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "%s: Metal backend is not initialized\n", name);
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "%s: token %d is outside vocab\n", name, token);
        return -1;
    }
    if (n_layers <= 0 || n_layers > QW3_N_LAYER) {
        fprintf(fp, "%s: invalid layer count %d\n", name, n_layers);
        return -1;
    }

    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t inner_n = QW3_N_LINEAR_V_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t state_n = QW3_N_LINEAR_V_HEADS *
                             QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t conv_state_n = n_qkv * (QW3_N_LINEAR_CONV_K - 1);
    const uint32_t gqa_qg_n = (uint32_t)tensor_cols_qg();
    const uint32_t gqa_q_n = QW3_N_HEAD * QW3_N_HEAD_DIM;
    const uint32_t gqa_kv_n = (uint32_t)tensor_cols_kv();

    float *cpu0 = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu1 = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu0 = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu1 = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_trace = qw3_xmalloc((size_t)(n_layers + 1) *
                                   QW3_N_EMBD * sizeof(float));
    float *cpu_conv_state = qw3_xcalloc((size_t)QW3_N_LINEAR_LAYERS * conv_state_n,
                                        sizeof(float));
    float *cpu_state = qw3_xcalloc((size_t)QW3_N_LINEAR_LAYERS * state_n,
                                   sizeof(float));
    float *gqa_k_cache = qw3_xmalloc((size_t)QW3_N_FULL_ATTN_LAYERS *
                                     gqa_kv_n * sizeof(float));
    float *gqa_v_cache = qw3_xmalloc((size_t)QW3_N_FULL_ATTN_LAYERS *
                                     gqa_kv_n * sizeof(float));

    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *z = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *alpha = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta_sig = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *q = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *k = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *state_out = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *core = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *inner = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *attn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *ffn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *router = qw3_xmalloc((size_t)QW3_N_EXPERT * sizeof(float));
    float *gate = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *up = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *sparse = qw3_xcalloc((size_t)QW3_N_EMBD, sizeof(float));
    float *sh_gate = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *sh_up = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *sh_hidden = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *shared_down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *shared = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qfull = qw3_xmalloc((size_t)gqa_qg_n * sizeof(float));
    float *gqa_gate = qw3_xmalloc((size_t)gqa_q_n * sizeof(float));
    float *v = qw3_xmalloc((size_t)gqa_kv_n * sizeof(float));

    float *ca = cpu0, *cb = cpu1, *ga = gpu0, *gb = gpu1;
    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd,
                                    (uint64_t)token, ca);
    if (ok) memcpy(cpu_trace, ca, (size_t)QW3_N_EMBD * sizeof(float));
    int gpu_ok = ok &&
        qw3_metal_embed_q8_0(e->weights.token_embd->offset, (uint32_t)token,
                             QW3_N_EMBD, ga);

    int cpu_linear_slot = 0;
    int cpu_full_slot = 0;
    for (int il = 0; ok && il < n_layers; il++) {
        if (qw3_layer_is_full_attention((uint32_t)il)) {
            ok = cpu_full_attention_layer(e, il, 0, ca,
                                          gqa_k_cache + (size_t)cpu_full_slot * gqa_kv_n,
                                          gqa_v_cache + (size_t)cpu_full_slot * gqa_kv_n,
                                          1, cb);
            cpu_full_slot++;
        } else {
            ok = cpu_deltanet_layer(e, il, ca,
                                    cpu_conv_state + (uint64_t)cpu_linear_slot * conv_state_n,
                                    cpu_state + (uint64_t)cpu_linear_slot * state_n, cb);
            cpu_linear_slot++;
        }
        if (ok) {
            float *tmp = ca; ca = cb; cb = tmp;
            memcpy(cpu_trace + (size_t)(il + 1) * QW3_N_EMBD, ca,
                   (size_t)QW3_N_EMBD * sizeof(float));
        }
    }

    int ids[QW3_N_EXPERT_USED];
    float vals[QW3_N_EXPERT_USED], weights[QW3_N_EXPERT_USED];
    memset(ids, 0, sizeof(ids));
    memset(vals, 0, sizeof(vals));
    memset(weights, 0, sizeof(weights));
    int last_top0 = -1;
    int first_bad_layer = -1;
    float first_bad_maxdiff = 0.0f;
    double first_bad_rmsdiff = 0.0;

    for (int il = 0; gpu_ok && il < n_layers; il++) {
        const qw3_layer_weights *lw = &e->weights.layer[il];
        if (qw3_layer_is_full_attention((uint32_t)il)) {
            gpu_ok =
                qw3_metal_rmsnorm_weight_f32(ga, lw->attn_norm->offset,
                                             xn, QW3_N_EMBD, QW3_RMS_EPS) &&
                qw3_metal_matvec_q8_0(lw->attn_q_proj->offset, xn,
                                      QW3_N_EMBD, gqa_qg_n, qfull) &&
                qw3_metal_matvec_q8_0(lw->attn_v_proj->offset, xn,
                                      QW3_N_EMBD, gqa_kv_n, v);
            if (gpu_ok) {
                for (int h = 0; h < QW3_N_HEAD; h++) {
                    memcpy(gqa_gate + h * QW3_N_HEAD_DIM,
                           qfull + h * QW3_N_HEAD_DIM * 2 + QW3_N_HEAD_DIM,
                           (size_t)QW3_N_HEAD_DIM * sizeof(float));
                }
                gpu_ok =
                    qw3_metal_gqa_single_token_inner(gqa_gate, v,
                                                     QW3_N_HEAD, QW3_N_HEAD_KV,
                                                     QW3_N_HEAD_DIM, inner) &&
                    qw3_metal_matvec_q8_0(lw->attn_o_proj->offset, inner,
                                          inner_n, QW3_N_EMBD, attn) &&
                    qw3_metal_residual_rmsnorm_weight_f32(ga, attn,
                                                          lw->ffn_norm->offset,
                                                          ffn, QW3_N_EMBD,
                                                          QW3_RMS_EPS) &&
                    qw3_metal_matvec_f32(lw->ffn_gate_inp->offset, ffn,
                                         QW3_N_EMBD, QW3_N_EXPERT, router);
            }
        } else {
            gpu_ok =
                qw3_metal_rmsnorm_weight_f32(ga, lw->attn_norm->offset,
                                             xn, QW3_N_EMBD, QW3_RMS_EPS) &&
                qw3_metal_matvec_q8_0(lw->linear_qkv_proj->offset, xn,
                                      QW3_N_EMBD, n_qkv, qkv) &&
                qw3_metal_matvec_q8_0(lw->linear_gate_proj->offset, xn,
                                      QW3_N_EMBD, inner_n, z) &&
                qw3_metal_matvec_f32(lw->linear_ssm_alpha->offset, xn,
                                     QW3_N_EMBD, QW3_N_LINEAR_V_HEADS, alpha) &&
                qw3_metal_matvec_f32(lw->linear_ssm_beta->offset, xn,
                                     QW3_N_EMBD, QW3_N_LINEAR_V_HEADS, beta) &&
                qw3_metal_deltanet_conv1d_zero(lw->linear_conv_weight->offset,
                                               qkv, n_qkv, conv) &&
                qw3_metal_l2norm_heads(conv, QW3_N_LINEAR_QK_HEADS,
                                       QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS, q) &&
                qw3_metal_l2norm_heads(conv + qk_n, QW3_N_LINEAR_QK_HEADS,
                                       QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS, k);
            if (gpu_ok) {
                for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
                    beta_sig[hv] = 1.0f / (1.0f + expf(-beta[hv]));
                }
                gpu_ok =
                    qw3_metal_deltanet_recur_zero(q, k, conv + qk_n * 2,
                                                  beta_sig,
                                                  QW3_N_LINEAR_QK_HEADS,
                                                  QW3_N_LINEAR_V_HEADS,
                                                  QW3_N_LINEAR_HEAD_DIM,
                                                  state_out, core) &&
                    qw3_metal_deltanet_gated_rmsnorm(lw->linear_ssm_norm->offset,
                                                     core, z,
                                                     QW3_N_LINEAR_V_HEADS,
                                                     QW3_N_LINEAR_HEAD_DIM,
                                                     QW3_RMS_EPS, inner) &&
                    qw3_metal_matvec_q8_0(lw->linear_ssm_out->offset, inner,
                                          inner_n, QW3_N_EMBD, attn) &&
                    qw3_metal_residual_rmsnorm_weight_f32(ga, attn,
                                                          lw->ffn_norm->offset,
                                                          ffn, QW3_N_EMBD,
                                                          QW3_RMS_EPS) &&
                    qw3_metal_matvec_f32(lw->ffn_gate_inp->offset, ffn,
                                         QW3_N_EMBD, QW3_N_EXPERT, router);
            }
        }
        if (gpu_ok) {
            topk_desc(router, QW3_N_EXPERT, QW3_N_EXPERT_USED, ids, vals);
            last_top0 = ids[0];
            float wsum = 0.0f;
            for (int kk = 0; kk < QW3_N_EXPERT_USED; kk++) {
                weights[kk] = expf(vals[kk] - vals[0]);
                wsum += weights[kk];
            }
            for (int kk = 0; kk < QW3_N_EXPERT_USED; kk++) weights[kk] /= wsum;
            memset(sparse, 0, (size_t)QW3_N_EMBD * sizeof(float));
        }
        for (int kk = 0; gpu_ok && kk < QW3_N_EXPERT_USED; kk++) {
            gpu_ok =
                qw3_metal_matvec_iq3_s_expert(lw->ffn_gate_exps->offset, (uint32_t)ids[kk],
                                              ffn, QW3_N_EMBD, QW3_N_FF_EXP, gate) &&
                qw3_metal_matvec_iq3_s_expert(lw->ffn_up_exps->offset, (uint32_t)ids[kk],
                                              ffn, QW3_N_EMBD, QW3_N_FF_EXP, up) &&
                qw3_metal_silu_mul(gate, up, QW3_N_FF_EXP, hidden);
            if (gpu_ok && lw->ffn_down_exps->type == QW3_TENSOR_IQ4_XS) {
                gpu_ok = qw3_metal_matvec_iq4_xs_expert(
                    lw->ffn_down_exps->offset, (uint32_t)ids[kk],
                    hidden, QW3_N_FF_EXP, QW3_N_EMBD, down);
            } else if (gpu_ok && lw->ffn_down_exps->type == QW3_TENSOR_Q6_K) {
                gpu_ok = qw3_metal_matvec_q6_k_expert(
                    lw->ffn_down_exps->offset, (uint32_t)ids[kk],
                    hidden, QW3_N_FF_EXP, QW3_N_EMBD, down);
            } else if (gpu_ok) {
                fprintf(fp, "%s: layer %d sparse down type %s is not supported by the current Metal mixed helper\n",
                        name, il, tensor_type_name(lw->ffn_down_exps->type));
                gpu_ok = 0;
            }
            if (gpu_ok) for (int i = 0; i < QW3_N_EMBD; i++) sparse[i] += weights[kk] * down[i];
        }
        gpu_ok = gpu_ok &&
            qw3_metal_matvec_q8_0(lw->ffn_gate_shared->offset, ffn,
                                  QW3_N_EMBD, QW3_N_FF_SHARED, sh_gate) &&
            qw3_metal_matvec_q8_0(lw->ffn_up_shared->offset, ffn,
                                  QW3_N_EMBD, QW3_N_FF_SHARED, sh_up) &&
            qw3_metal_silu_mul(sh_gate, sh_up, QW3_N_FF_SHARED, sh_hidden) &&
            qw3_metal_matvec_q8_0(lw->ffn_down_shared->offset, sh_hidden,
                                  QW3_N_FF_SHARED, QW3_N_EMBD, shared_down);
        float shared_raw = 0.0f;
        if (gpu_ok) gpu_ok = qw3_metal_matvec_f32(lw->ffn_gate_inp_shexp->offset, ffn,
                                                 QW3_N_EMBD, 1, &shared_raw);
        if (gpu_ok) gpu_ok = qw3_metal_scale(shared_down, QW3_N_EMBD,
                                             1.0f / (1.0f + expf(-shared_raw)), shared);
        if (gpu_ok) {
            for (int i = 0; i < QW3_N_EMBD; i++) gb[i] = ga[i] + attn[i] + sparse[i] + shared[i];
            float *tmp = ga; ga = gb; gb = tmp;
            if (first_bad_layer < 0) {
                const float *cpu_ref = cpu_trace + (size_t)(il + 1) * QW3_N_EMBD;
                float layer_max = 0.0f;
                double layer_rms = 0.0;
                for (int i = 0; i < QW3_N_EMBD; i++) {
                    float d = fabsf(cpu_ref[i] - ga[i]);
                    if (d > layer_max) layer_max = d;
                    layer_rms += (double)d * d;
                }
                layer_rms = sqrt(layer_rms / QW3_N_EMBD);
                if (layer_max > 1.0e-3f || layer_rms > 1.0e-5) {
                    first_bad_layer = il;
                    first_bad_maxdiff = layer_max;
                    first_bad_rmsdiff = layer_rms;
                }
            }
        }
    }

    float maxdiff = 0.0f;
    double rmsdiff = 0.0, out_rms = 0.0;
    if (gpu_ok && ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(ca[i] - ga[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * d;
            out_rms += (double)ga[i] * ga[i];
        }
        rmsdiff = sqrt(rmsdiff / QW3_N_EMBD);
        out_rms = sqrt(out_rms / QW3_N_EMBD);
    }
    fprintf(fp,
            "%s: %s token=%d layers=0..%d maxdiff=%.7g rmsdiff=%.7g out_rms=%.7g last_top0=%d first_bad_layer=%d first_bad_maxdiff=%.7g first_bad_rmsdiff=%.7g out0=[%.7g %.7g %.7g %.7g]\n",
            name, (gpu_ok && ok) ? "ok" : "failed", token, n_layers - 1,
            maxdiff, rmsdiff,
            out_rms, last_top0, first_bad_layer,
            first_bad_maxdiff, first_bad_rmsdiff,
            ga[0], ga[1], ga[2], ga[3]);
    if (gpu_ok && ok && cpu_hidden_out && gpu_hidden_out) {
        memcpy(cpu_hidden_out, ca, (size_t)QW3_N_EMBD * sizeof(float));
        memcpy(gpu_hidden_out, ga, (size_t)QW3_N_EMBD * sizeof(float));
    }

    free(v); free(gqa_gate); free(qfull); free(shared); free(shared_down);
    free(sh_hidden); free(sh_up); free(sh_gate); free(sparse); free(down);
    free(hidden); free(up); free(gate); free(router); free(ffn); free(attn);
    free(inner); free(core); free(state_out); free(k); free(q); free(conv);
    free(beta_sig); free(beta); free(alpha); free(z); free(qkv); free(xn);
    free(gqa_v_cache); free(gqa_k_cache); free(cpu_state); free(cpu_conv_state);
    free(cpu_trace); free(gpu1); free(gpu0); free(cpu1); free(cpu0);
    return (gpu_ok && ok) ? 0 : -1;
#endif
}

static int qw3_engine_metal_mixed_n_test(qw3_engine *e, int token,
                                         int n_layers, const char *name,
                                         FILE *fp) {
    return qw3_engine_metal_mixed_n_hidden(e, token, n_layers, name,
                                           NULL, NULL, fp);
}

int qw3_engine_metal_mixed4_test(qw3_engine *e, int token, FILE *fp) {
    return qw3_engine_metal_mixed_n_test(e, token, 4, "metal mixed4", fp);
}

int qw3_engine_metal_mixed8_test(qw3_engine *e, int token, FILE *fp) {
    return qw3_engine_metal_mixed_n_test(e, token, 8, "metal mixed8", fp);
}

int qw3_engine_metal_mixed16_test(qw3_engine *e, int token, FILE *fp) {
    return qw3_engine_metal_mixed_n_test(e, token, 16, "metal mixed16", fp);
}

int qw3_engine_metal_mixed32_test(qw3_engine *e, int token, FILE *fp) {
    return qw3_engine_metal_mixed_n_test(e, token, 32, "metal mixed32", fp);
}

int qw3_engine_metal_mixed33_test(qw3_engine *e, int token, FILE *fp) {
    return qw3_engine_metal_mixed_n_test(e, token, 33, "metal mixed33", fp);
}

int qw3_engine_metal_mixed34_test(qw3_engine *e, int token, FILE *fp) {
    return qw3_engine_metal_mixed_n_test(e, token, 34, "metal mixed34", fp);
}

int qw3_engine_metal_mixed35_test(qw3_engine *e, int token, FILE *fp) {
    return qw3_engine_metal_mixed_n_test(e, token, 35, "metal mixed35", fp);
}

int qw3_engine_metal_mixed36_test(qw3_engine *e, int token, FILE *fp) {
    return qw3_engine_metal_mixed_n_test(e, token, 36, "metal mixed36", fp);
}

int qw3_engine_metal_mixed40_test(qw3_engine *e, int token, FILE *fp) {
    return qw3_engine_metal_mixed_n_test(e, token, QW3_N_LAYER, "metal mixed40", fp);
}

int qw3_engine_metal_logits_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal logits: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal logits: Metal backend is not initialized\n");
        return -1;
    }
    if (e->weights.output->type != QW3_TENSOR_Q8_0 &&
        e->weights.output->type != QW3_TENSOR_Q6_K) {
        fprintf(fp, "metal logits: unsupported output.weight type %s\n",
                tensor_type_name(e->weights.output->type));
        return -1;
    }

    float *cpu_hidden = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_hidden = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_norm = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_norm = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_logits = qw3_xmalloc((size_t)QW3_N_VOCAB * sizeof(float));
    float *gpu_logits = qw3_xmalloc((size_t)QW3_N_VOCAB * sizeof(float));

    int rc = qw3_engine_metal_mixed_n_hidden(e, token, QW3_N_LAYER,
                                             "metal logits/mixed40",
                                             cpu_hidden, gpu_hidden, fp);
    bool ok = (rc == 0);
    if (ok) {
        cpu_rmsnorm(cpu_norm, cpu_hidden, &e->model, e->weights.output_norm,
                    QW3_N_EMBD);
        ok = cpu_matvec(&e->model, e->weights.output, cpu_norm, cpu_logits);
    }
    if (ok) {
        ok = qw3_metal_rmsnorm_weight_f32(gpu_hidden,
                                          e->weights.output_norm->offset,
                                          gpu_norm, QW3_N_EMBD, QW3_RMS_EPS);
        if (ok && e->weights.output->type == QW3_TENSOR_Q8_0) {
            ok = qw3_metal_matvec_q8_0(e->weights.output->offset, gpu_norm,
                                       QW3_N_EMBD, QW3_N_VOCAB, gpu_logits);
        } else if (ok) {
            ok = qw3_metal_matvec_q6_k(e->weights.output->offset, gpu_norm,
                                       QW3_N_EMBD, QW3_N_VOCAB, gpu_logits);
        }
    }

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    int cpu_ids[8], gpu_ids[8];
    float cpu_vals[8], gpu_vals[8];
    uint32_t metal_argmax = 0;
    float metal_argmax_val = 0.0f;
    int metal_argmax_ok = 0;
    for (int i = 0; i < 8; i++) {
        cpu_ids[i] = -1;
        gpu_ids[i] = -1;
    }
    memset(cpu_vals, 0, sizeof(cpu_vals));
    memset(gpu_vals, 0, sizeof(gpu_vals));
    if (ok) {
        for (int i = 0; i < QW3_N_VOCAB; i++) {
            float d = fabsf(cpu_logits[i] - gpu_logits[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * d;
        }
        rmsdiff = sqrt(rmsdiff / QW3_N_VOCAB);
        topk_desc(cpu_logits, QW3_N_VOCAB, 8, cpu_ids, cpu_vals);
        topk_desc(gpu_logits, QW3_N_VOCAB, 8, gpu_ids, gpu_vals);
        metal_argmax_ok = qw3_metal_argmax(gpu_logits, QW3_N_VOCAB,
                                           &metal_argmax, &metal_argmax_val);
        ok = metal_argmax_ok && metal_argmax == (uint32_t)gpu_ids[0];
    }

    fprintf(fp,
            "metal logits: %s token=%d maxdiff=%.7g rmsdiff=%.7g cpu_top0=%d gpu_top0=%d metal_argmax=%u\n",
            ok ? "ok" : "failed", token, maxdiff, rmsdiff,
            cpu_ids[0], gpu_ids[0], metal_argmax);
    if (ok) {
        fprintf(fp, "metal logits cpu top8:");
        for (int i = 0; i < 8; i++) fprintf(fp, " %d:%.7g", cpu_ids[i], cpu_vals[i]);
        fprintf(fp, "\nmetal logits gpu top8:");
        for (int i = 0; i < 8; i++) fprintf(fp, " %d:%.7g", gpu_ids[i], gpu_vals[i]);
        fprintf(fp, "\nmetal logits argmax: %u:%.7g\n",
                metal_argmax, metal_argmax_val);
    }

    free(gpu_logits);
    free(cpu_logits);
    free(gpu_norm);
    free(cpu_norm);
    free(gpu_hidden);
    free(cpu_hidden);
    return ok ? 0 : -1;
#endif
}

static int qw3_engine_metal_decode_inner(qw3_engine *e, const qw3_tokens *prompt,
                                         int ctx_size, const char *name,
                                         bool compare_cpu,
                                         bool print_top8,
                                         bool verbose,
                                         int *cpu_top0_out,
                                         int *gpu_top0_out,
                                         float *maxdiff_out,
                                         double *rmsdiff_out,
                                         FILE *fp) {
    if (!e || !prompt || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)prompt; (void)ctx_size; (void)compare_cpu; (void)print_top8;
    (void)verbose;
    (void)cpu_top0_out; (void)gpu_top0_out;
    (void)maxdiff_out; (void)rmsdiff_out;
    if (fp) {
        fprintf(fp, "%s: unavailable in QW3_NO_METAL build\n",
                name ? name : "metal decode");
    }
    return -1;
#else
    const char *label = name ? name : "metal decode";
    FILE *log = fp ? fp : stderr;
    const double t_total0 = qw3_now_sec();
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(log, "%s: Metal backend is not initialized\n", label);
        return -1;
    }
    if (prompt->len <= 0 || prompt->len > ctx_size) {
        fprintf(log, "%s: invalid prompt length %d for ctx %d\n",
                label, prompt->len, ctx_size);
        return -1;
    }
    if (e->weights.output->type != QW3_TENSOR_Q8_0 &&
        e->weights.output->type != QW3_TENSOR_Q6_K) {
        fprintf(log, "%s: unsupported output.weight type %s\n",
                label, tensor_type_name(e->weights.output->type));
        return -1;
    }

    qw3_session *cpu = NULL;
    char err[256] = {0};
    const double t_cpu0 = qw3_now_sec();
    bool ok = true;
    if (compare_cpu) {
        ok = qw3_session_create(&cpu, e, ctx_size) == 0 &&
             qw3_session_sync(cpu, prompt, err, sizeof(err)) == 0;
    }
    const double t_cpu1 = qw3_now_sec();

    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t inner_n = (uint32_t)tensor_linear_inner();
    const uint32_t state_n = QW3_N_LINEAR_V_HEADS *
                             QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t conv_state_n = n_qkv * (QW3_N_LINEAR_CONV_K - 1);
    const uint32_t gqa_qg_n = (uint32_t)tensor_cols_qg();
    const uint32_t gqa_q_n = QW3_N_HEAD * QW3_N_HEAD_DIM;
    const uint32_t gqa_kv_n = (uint32_t)tensor_cols_kv();

    float *x0 = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *x1 = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *attn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *ffn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *router = qw3_xmalloc((size_t)QW3_N_EXPERT * sizeof(float));
    float *egate = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *up = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *sparse = qw3_xcalloc((size_t)QW3_N_EMBD, sizeof(float));
    float *sh_gate = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *sh_up = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *sh_hidden = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *shared_down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *shared = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *norm = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_logits = qw3_xmalloc((size_t)QW3_N_VOCAB * sizeof(float));

    float *conv_states = qw3_xcalloc((size_t)QW3_N_LINEAR_LAYERS * conv_state_n,
                                     sizeof(float));
    float *states = qw3_xcalloc((size_t)QW3_N_LINEAR_LAYERS * state_n,
                                sizeof(float));
    float *gqa_k_cache = qw3_xcalloc((size_t)QW3_N_FULL_ATTN_LAYERS *
                                     ctx_size * gqa_kv_n, sizeof(float));
    float *gqa_v_cache = qw3_xcalloc((size_t)QW3_N_FULL_ATTN_LAYERS *
                                     ctx_size * gqa_kv_n, sizeof(float));

    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *z = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *alpha = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta_sig = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *gamma = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *conv_next = qw3_xmalloc((size_t)conv_state_n * sizeof(float));
    float *q = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *k = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *state_next = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *core = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *inner = qw3_xmalloc((size_t)inner_n * sizeof(float));

    float *qfull = qw3_xmalloc((size_t)gqa_qg_n * sizeof(float));
    float *gqa_gate = qw3_xmalloc((size_t)gqa_q_n * sizeof(float));
    float *qnorm = qw3_xmalloc((size_t)gqa_q_n * sizeof(float));
    float *qrope = qw3_xmalloc((size_t)gqa_q_n * sizeof(float));
    float *kproj = qw3_xmalloc((size_t)gqa_kv_n * sizeof(float));
    float *knorm = qw3_xmalloc((size_t)gqa_kv_n * sizeof(float));
    float *krope = qw3_xmalloc((size_t)gqa_kv_n * sizeof(float));
    float *v = qw3_xmalloc((size_t)gqa_kv_n * sizeof(float));

    int gpu_ok = ok;
    int last_top0 = -1;
    int last_token = prompt->v[prompt->len - 1];
    const double t_metal0 = qw3_now_sec();
    for (int pos = 0; gpu_ok && pos < prompt->len; pos++) {
        int token = prompt->v[pos];
        if (token < 0 || token >= QW3_N_VOCAB) {
            fprintf(log, "%s: token %d at pos %d is outside vocab\n",
                    label, token, pos);
            gpu_ok = 0;
            break;
        }
        gpu_ok = qw3_metal_embed_q8_0(e->weights.token_embd->offset,
                                      (uint32_t)token, QW3_N_EMBD, x0);
        int linear_slot = 0;
        int full_slot = 0;
        for (int il = 0; gpu_ok && il < QW3_N_LAYER; il++) {
            const qw3_layer_weights *lw = &e->weights.layer[il];
            if (qw3_layer_is_full_attention((uint32_t)il)) {
                float *kc = gqa_k_cache +
                    ((uint64_t)full_slot * (uint64_t)ctx_size + (uint64_t)pos) *
                    gqa_kv_n;
                float *vc = gqa_v_cache +
                    ((uint64_t)full_slot * (uint64_t)ctx_size + (uint64_t)pos) *
                    gqa_kv_n;
                gpu_ok =
                    qw3_metal_rmsnorm_weight_f32(x0, lw->attn_norm->offset,
                                                 xn, QW3_N_EMBD, QW3_RMS_EPS) &&
                    qw3_metal_matvec_q8_0(lw->attn_q_proj->offset, xn,
                                          QW3_N_EMBD, gqa_qg_n, qfull) &&
                    qw3_metal_matvec_q8_0(lw->attn_k_proj->offset, xn,
                                          QW3_N_EMBD, gqa_kv_n, kproj) &&
                    qw3_metal_matvec_q8_0(lw->attn_v_proj->offset, xn,
                                          QW3_N_EMBD, gqa_kv_n, v);
                for (int h = 0; gpu_ok && h < QW3_N_HEAD; h++) {
                    memcpy(gqa_gate + h * QW3_N_HEAD_DIM,
                           qfull + h * QW3_N_HEAD_DIM * 2 + QW3_N_HEAD_DIM,
                           (size_t)QW3_N_HEAD_DIM * sizeof(float));
                    gpu_ok = qw3_metal_rmsnorm_weight_f32(
                        qfull + h * QW3_N_HEAD_DIM * 2,
                        lw->attn_q_norm->offset,
                        qnorm + h * QW3_N_HEAD_DIM,
                        QW3_N_HEAD_DIM, QW3_RMS_EPS);
                }
                for (int h = 0; gpu_ok && h < QW3_N_HEAD_KV; h++) {
                    gpu_ok = qw3_metal_rmsnorm_weight_f32(
                        kproj + h * QW3_N_HEAD_DIM,
                        lw->attn_k_norm->offset,
                        knorm + h * QW3_N_HEAD_DIM,
                        QW3_N_HEAD_DIM, QW3_RMS_EPS);
                }
                if (gpu_ok) {
                    gpu_ok =
                        qw3_metal_rope_heads(qnorm, QW3_N_HEAD,
                                             QW3_N_HEAD_DIM, QW3_ROPE_DIM,
                                             pos, QW3_ROPE_THETA, qrope) &&
                        qw3_metal_rope_heads(knorm, QW3_N_HEAD_KV,
                                             QW3_N_HEAD_DIM, QW3_ROPE_DIM,
                                             pos, QW3_ROPE_THETA, krope);
                }
                if (gpu_ok) {
                    memcpy(kc, krope, (size_t)gqa_kv_n * sizeof(float));
                    memcpy(vc, v, (size_t)gqa_kv_n * sizeof(float));
                    gpu_ok =
                        qw3_metal_gqa_attend_n_inner(
                            qrope, gqa_gate,
                            gqa_k_cache + (uint64_t)full_slot * ctx_size * gqa_kv_n,
                            gqa_v_cache + (uint64_t)full_slot * ctx_size * gqa_kv_n,
                            pos + 1, QW3_N_HEAD, QW3_N_HEAD_KV,
                            QW3_N_HEAD_DIM, inner) &&
                        qw3_metal_matvec_q8_0(lw->attn_o_proj->offset, inner,
                                              inner_n, QW3_N_EMBD, attn) &&
                        qw3_metal_residual_rmsnorm_weight_f32(
                            x0, attn, lw->ffn_norm->offset, ffn,
                            QW3_N_EMBD, QW3_RMS_EPS) &&
                        qw3_metal_matvec_f32(lw->ffn_gate_inp->offset, ffn,
                                             QW3_N_EMBD, QW3_N_EXPERT, router);
                }
                full_slot++;
            } else {
                float *conv_state = conv_states +
                    (uint64_t)linear_slot * conv_state_n;
                float *state = states + (uint64_t)linear_slot * state_n;
                gpu_ok =
                    qw3_metal_rmsnorm_weight_f32(x0, lw->attn_norm->offset,
                                                 xn, QW3_N_EMBD, QW3_RMS_EPS) &&
                    qw3_metal_matvec_q8_0(lw->linear_qkv_proj->offset, xn,
                                          QW3_N_EMBD, n_qkv, qkv) &&
                    qw3_metal_matvec_q8_0(lw->linear_gate_proj->offset, xn,
                                          QW3_N_EMBD, inner_n, z) &&
                    qw3_metal_matvec_f32(lw->linear_ssm_alpha->offset, xn,
                                         QW3_N_EMBD, QW3_N_LINEAR_V_HEADS,
                                         alpha) &&
                    qw3_metal_matvec_f32(lw->linear_ssm_beta->offset, xn,
                                         QW3_N_EMBD, QW3_N_LINEAR_V_HEADS,
                                         beta) &&
                    qw3_metal_deltanet_conv1d_step(
                        lw->linear_conv_weight->offset, qkv, conv_state,
                        n_qkv, conv, conv_next) &&
                    qw3_metal_l2norm_heads(conv, QW3_N_LINEAR_QK_HEADS,
                                           QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS,
                                           q) &&
                    qw3_metal_l2norm_heads(conv + qk_n, QW3_N_LINEAR_QK_HEADS,
                                           QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS,
                                           k);
                if (gpu_ok) {
                    for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
                        beta_sig[hv] = 1.0f / (1.0f + expf(-beta[hv]));
                        float ah = cpu_softplus(alpha[hv] +
                            tensor_read_dense_1d(&e->model,
                                                 lw->linear_ssm_dt_bias,
                                                 (uint64_t)hv));
                        gamma[hv] = expf(ah *
                            tensor_read_dense_1d(&e->model,
                                                 lw->linear_ssm_a,
                                                 (uint64_t)hv));
                    }
                    gpu_ok =
                        qw3_metal_deltanet_recur(
                            state, q, k, conv + qk_n * 2, beta_sig, gamma,
                            QW3_N_LINEAR_QK_HEADS, QW3_N_LINEAR_V_HEADS,
                            QW3_N_LINEAR_HEAD_DIM, state_next, core) &&
                        qw3_metal_deltanet_gated_rmsnorm(
                            lw->linear_ssm_norm->offset, core, z,
                            QW3_N_LINEAR_V_HEADS, QW3_N_LINEAR_HEAD_DIM,
                            QW3_RMS_EPS, inner) &&
                        qw3_metal_matvec_q8_0(lw->linear_ssm_out->offset,
                                              inner, inner_n, QW3_N_EMBD,
                                              attn) &&
                        qw3_metal_residual_rmsnorm_weight_f32(
                            x0, attn, lw->ffn_norm->offset, ffn,
                            QW3_N_EMBD, QW3_RMS_EPS) &&
                        qw3_metal_matvec_f32(lw->ffn_gate_inp->offset, ffn,
                                             QW3_N_EMBD, QW3_N_EXPERT, router);
                }
                if (gpu_ok) {
                    memcpy(conv_state, conv_next,
                           (size_t)conv_state_n * sizeof(float));
                    memcpy(state, state_next, (size_t)state_n * sizeof(float));
                }
                linear_slot++;
            }

            int ids[QW3_N_EXPERT_USED];
            float vals[QW3_N_EXPERT_USED], weights[QW3_N_EXPERT_USED];
            memset(ids, 0, sizeof(ids));
            memset(vals, 0, sizeof(vals));
            if (gpu_ok) {
                topk_desc(router, QW3_N_EXPERT, QW3_N_EXPERT_USED, ids, vals);
                if (pos == prompt->len - 1) last_top0 = ids[0];
                float wsum = 0.0f;
                for (int kk = 0; kk < QW3_N_EXPERT_USED; kk++) {
                    weights[kk] = expf(vals[kk] - vals[0]);
                    wsum += weights[kk];
                }
                for (int kk = 0; kk < QW3_N_EXPERT_USED; kk++) weights[kk] /= wsum;
                memset(sparse, 0, (size_t)QW3_N_EMBD * sizeof(float));
            }
            for (int kk = 0; gpu_ok && kk < QW3_N_EXPERT_USED; kk++) {
                gpu_ok =
                    qw3_metal_matvec_iq3_s_expert(
                        lw->ffn_gate_exps->offset, (uint32_t)ids[kk],
                        ffn, QW3_N_EMBD, QW3_N_FF_EXP, egate) &&
                    qw3_metal_matvec_iq3_s_expert(
                        lw->ffn_up_exps->offset, (uint32_t)ids[kk],
                        ffn, QW3_N_EMBD, QW3_N_FF_EXP, up) &&
                    qw3_metal_silu_mul(egate, up, QW3_N_FF_EXP, hidden);
                if (gpu_ok && lw->ffn_down_exps->type == QW3_TENSOR_IQ4_XS) {
                    gpu_ok = qw3_metal_matvec_iq4_xs_expert(
                        lw->ffn_down_exps->offset, (uint32_t)ids[kk],
                        hidden, QW3_N_FF_EXP, QW3_N_EMBD, down);
                } else if (gpu_ok && lw->ffn_down_exps->type == QW3_TENSOR_Q6_K) {
                    gpu_ok = qw3_metal_matvec_q6_k_expert(
                        lw->ffn_down_exps->offset, (uint32_t)ids[kk],
                        hidden, QW3_N_FF_EXP, QW3_N_EMBD, down);
                } else if (gpu_ok) {
                    fprintf(log, "%s: layer %d sparse down type %s unsupported\n",
                            label, il, tensor_type_name(lw->ffn_down_exps->type));
                    gpu_ok = 0;
                }
                if (gpu_ok) {
                    for (int i = 0; i < QW3_N_EMBD; i++) {
                        sparse[i] += weights[kk] * down[i];
                    }
                }
            }
            gpu_ok = gpu_ok &&
                qw3_metal_matvec_q8_0(lw->ffn_gate_shared->offset, ffn,
                                      QW3_N_EMBD, QW3_N_FF_SHARED, sh_gate) &&
                qw3_metal_matvec_q8_0(lw->ffn_up_shared->offset, ffn,
                                      QW3_N_EMBD, QW3_N_FF_SHARED, sh_up) &&
                qw3_metal_silu_mul(sh_gate, sh_up, QW3_N_FF_SHARED, sh_hidden) &&
                qw3_metal_matvec_q8_0(lw->ffn_down_shared->offset, sh_hidden,
                                      QW3_N_FF_SHARED, QW3_N_EMBD, shared_down);
            float shared_raw = 0.0f;
            if (gpu_ok) {
                gpu_ok = qw3_metal_matvec_f32(lw->ffn_gate_inp_shexp->offset,
                                              ffn, QW3_N_EMBD, 1, &shared_raw);
            }
            if (gpu_ok) {
                gpu_ok = qw3_metal_scale(
                    shared_down, QW3_N_EMBD,
                    1.0f / (1.0f + expf(-shared_raw)), shared);
            }
            if (gpu_ok) {
                for (int i = 0; i < QW3_N_EMBD; i++) {
                    x1[i] = x0[i] + attn[i] + sparse[i] + shared[i];
                }
                float *tmp = x0; x0 = x1; x1 = tmp;
            }
        }
    }

    if (gpu_ok) {
        gpu_ok = qw3_metal_rmsnorm_weight_f32(x0, e->weights.output_norm->offset,
                                              norm, QW3_N_EMBD, QW3_RMS_EPS);
        if (gpu_ok && e->weights.output->type == QW3_TENSOR_Q8_0) {
            gpu_ok = qw3_metal_matvec_q8_0(e->weights.output->offset, norm,
                                           QW3_N_EMBD, QW3_N_VOCAB,
                                           gpu_logits);
        } else if (gpu_ok) {
            gpu_ok = qw3_metal_matvec_q6_k(e->weights.output->offset, norm,
                                           QW3_N_EMBD, QW3_N_VOCAB,
                                           gpu_logits);
        }
    }
    const double t_metal1 = qw3_now_sec();

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    int cpu_ids[8], gpu_ids[8];
    float cpu_vals[8], gpu_vals[8];
    uint32_t metal_argmax = 0;
    float metal_argmax_val = 0.0f;
    int metal_argmax_ok = 0;
    memset(cpu_ids, 0, sizeof(cpu_ids));
    memset(gpu_ids, 0, sizeof(gpu_ids));
    memset(cpu_vals, 0, sizeof(cpu_vals));
    memset(gpu_vals, 0, sizeof(gpu_vals));
    if (gpu_ok) {
        topk_desc(gpu_logits, QW3_N_VOCAB, 8, gpu_ids, gpu_vals);
        metal_argmax_ok = qw3_metal_argmax(gpu_logits, QW3_N_VOCAB,
                                           &metal_argmax, &metal_argmax_val);
        gpu_ok = metal_argmax_ok && metal_argmax == (uint32_t)gpu_ids[0];
    }
    if (gpu_ok && ok && compare_cpu) {
        for (int i = 0; i < QW3_N_VOCAB; i++) {
            float d = fabsf(cpu->logits[i] - gpu_logits[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * d;
        }
        rmsdiff = sqrt(rmsdiff / QW3_N_VOCAB);
        topk_desc(cpu->logits, QW3_N_VOCAB, 8, cpu_ids, cpu_vals);
    }

    if (verbose) {
        fprintf(log,
                "%s: %s tokens=%d ctx=%d last_token=%d maxdiff=%.7g rmsdiff=%.7g cpu_top0=%d gpu_top0=%d gpu_top0_val=%.7g last_router_top0=%d cpu_ms=%.1f metal_ms=%.1f total_ms=%.1f\n",
                label, (gpu_ok && ok) ? "ok" : "failed", prompt->len, ctx_size,
                last_token, maxdiff, rmsdiff, cpu_ids[0],
                metal_argmax_ok ? (int)metal_argmax : gpu_ids[0],
                metal_argmax_ok ? metal_argmax_val : gpu_vals[0], last_top0,
                (t_cpu1 - t_cpu0) * 1000.0,
                (t_metal1 - t_metal0) * 1000.0,
                (qw3_now_sec() - t_total0) * 1000.0);
        if (!ok && err[0]) fprintf(log, "%s cpu error: %s\n", label, err);
        if (gpu_ok && ok && print_top8 && compare_cpu) {
            fprintf(log, "%s cpu top8:", label);
            for (int i = 0; i < 8; i++) fprintf(log, " %d:%.7g", cpu_ids[i], cpu_vals[i]);
            fprintf(log, "\n%s gpu top8:", label);
            for (int i = 0; i < 8; i++) fprintf(log, " %d:%.7g", gpu_ids[i], gpu_vals[i]);
            fprintf(log, "\n");
        } else if (gpu_ok && ok && print_top8) {
            fprintf(log, "%s gpu top8:", label);
            for (int i = 0; i < 8; i++) fprintf(log, " %d:%.7g", gpu_ids[i], gpu_vals[i]);
            fprintf(log, "\n");
        }
    }
    if (gpu_ok && ok) {
        if (cpu_top0_out) *cpu_top0_out = cpu_ids[0];
        if (gpu_top0_out) *gpu_top0_out = (int)metal_argmax;
        if (maxdiff_out) *maxdiff_out = maxdiff;
        if (rmsdiff_out) *rmsdiff_out = rmsdiff;
    }

    free(v); free(krope); free(knorm); free(kproj); free(qrope); free(qnorm);
    free(gqa_gate); free(qfull); free(inner); free(core); free(state_next);
    free(k); free(q); free(conv_next); free(conv); free(gamma); free(beta_sig);
    free(beta); free(alpha); free(z); free(qkv); free(gqa_v_cache);
    free(gqa_k_cache); free(states); free(conv_states); free(gpu_logits);
    free(norm); free(shared); free(shared_down); free(sh_hidden); free(sh_up);
    free(sh_gate); free(sparse); free(down); free(hidden); free(up); free(egate);
    free(router); free(ffn); free(attn); free(xn); free(x1); free(x0);
    qw3_session_free(cpu);
    return (gpu_ok && ok) ? 0 : -1;
#endif
}

int qw3_engine_metal_decode_test(qw3_engine *e, const qw3_tokens *prompt,
                                 int ctx_size, FILE *fp) {
    return qw3_engine_metal_decode_inner(e, prompt, ctx_size, "metal decode",
                                         true, true, true, NULL, NULL, NULL,
                                         NULL, fp);
}

int qw3_engine_metal_session_test(qw3_engine *e, int ctx_size, FILE *fp) {
    if (!e || !fp || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)ctx_size;
    fprintf(fp, "metal session: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal session: Metal backend is not initialized\n");
        return -1;
    }
    qw3_session *s = NULL;
    if (qw3_session_create(&s, e, ctx_size) != 0 || !s || !s->metal) {
        qw3_session_free(s);
        fprintf(fp, "metal session: allocation failed ctx=%d\n", ctx_size);
        return -1;
    }
    qw3_metal_session_info info = qw3_metal_session_get_info(s->metal);
    fprintf(fp,
            "metal session: ok ctx=%d total=%.2f MiB gqa_kv=%.2f MiB deltanet=%.2f MiB conv=%.2f MiB logits=%.2f MiB scratch=%.2f MiB\n",
            ctx_size,
            (double)info.total_bytes / (1024.0 * 1024.0),
            (double)info.gqa_kv_bytes / (1024.0 * 1024.0),
            (double)info.deltanet_state_bytes / (1024.0 * 1024.0),
            (double)info.conv_state_bytes / (1024.0 * 1024.0),
            (double)info.logits_bytes / (1024.0 * 1024.0),
            (double)info.scratch_bytes / (1024.0 * 1024.0));
    qw3_session_free(s);
    return 0;
#endif
}

int qw3_engine_metal_session_embed_test(qw3_engine *e, int token,
                                        int ctx_size, FILE *fp) {
    if (!e || !fp || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token; (void)ctx_size;
    fprintf(fp, "metal session embed: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal session embed: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal session embed: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_tensor *emb = e->weights.token_embd;
    if (emb->type != QW3_TENSOR_Q8_0 || emb->dim[0] != QW3_N_EMBD) {
        fprintf(fp, "metal session embed: expected q8_0 embedding tensor\n");
        return -1;
    }

    qw3_session *s = NULL;
    float *cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    bool cpu_ok = tensor_read_dense_row(&e->model, emb, (uint64_t)token, cpu);
    int gpu_ok = 0;
    if (cpu_ok && qw3_session_create(&s, e, ctx_size) == 0 && s && s->metal) {
        gpu_ok = qw3_metal_session_embed_q8_0(s->metal, emb->offset,
                                              (uint32_t)token,
                                              QW3_N_EMBD, gpu);
    }

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu[i] - gpu[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * (double)d;
        }
        rmsdiff = sqrt(rmsdiff / (double)QW3_N_EMBD);
    }
    fprintf(fp,
            "metal session embed: %s token=%d n=%d maxdiff=%.7g rmsdiff=%.7g first=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, QW3_N_EMBD,
            maxdiff, rmsdiff, gpu[0], gpu[1], gpu[2], gpu[3]);
    qw3_session_free(s);
    free(gpu);
    free(cpu);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_session_rmsnorm_test(qw3_engine *e, int token,
                                          int ctx_size, FILE *fp) {
    if (!e || !fp || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token; (void)ctx_size;
    fprintf(fp, "metal session rmsnorm: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal session rmsnorm: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal session rmsnorm: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_tensor *emb = e->weights.token_embd;
    const qw3_tensor *weight = e->weights.layer[0].attn_norm;
    if (emb->type != QW3_TENSOR_Q8_0 || emb->dim[0] != QW3_N_EMBD) {
        fprintf(fp, "metal session rmsnorm: expected q8_0 embedding tensor\n");
        return -1;
    }

    qw3_session *s = NULL;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    bool cpu_ok = tensor_read_dense_row(&e->model, emb, (uint64_t)token, x);
    if (cpu_ok) {
        cpu_rmsnorm(cpu, x, &e->model, weight, QW3_N_EMBD);
    }
    int gpu_ok = 0;
    if (cpu_ok && qw3_session_create(&s, e, ctx_size) == 0 && s && s->metal) {
        gpu_ok =
            qw3_metal_session_embed_q8_0(s->metal, emb->offset,
                                         (uint32_t)token, QW3_N_EMBD, NULL) &&
            qw3_metal_session_rmsnorm_weight_f32(s->metal, weight->offset,
                                                 QW3_N_EMBD, QW3_RMS_EPS, gpu);
    }

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu[i] - gpu[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * (double)d;
        }
        rmsdiff = sqrt(rmsdiff / (double)QW3_N_EMBD);
    }
    fprintf(fp,
            "metal session rmsnorm: %s token=%d n=%d maxdiff=%.7g rmsdiff=%.7g first=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, QW3_N_EMBD,
            maxdiff, rmsdiff, gpu[0], gpu[1], gpu[2], gpu[3]);
    qw3_session_free(s);
    free(gpu);
    free(cpu);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_session_qkv_test(qw3_engine *e, int token,
                                      int ctx_size, FILE *fp) {
    if (!e || !fp || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token; (void)ctx_size;
    fprintf(fp, "metal session qkv: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal session qkv: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal session qkv: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_tensor *emb = e->weights.token_embd;
    const qw3_tensor *norm_w = e->weights.layer[0].attn_norm;
    const qw3_tensor *proj = e->weights.layer[0].linear_qkv_proj;
    const uint32_t n_out = (uint32_t)tensor_linear_qkv();
    if (emb->type != QW3_TENSOR_Q8_0 || proj->type != QW3_TENSOR_Q8_0) {
        fprintf(fp, "metal session qkv: expected q8_0 embedding/projection tensors\n");
        return -1;
    }

    qw3_session *s = NULL;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu = qw3_xmalloc((size_t)n_out * sizeof(float));
    float *gpu = qw3_xmalloc((size_t)n_out * sizeof(float));
    bool cpu_ok = tensor_read_dense_row(&e->model, emb, (uint64_t)token, x);
    if (cpu_ok) {
        cpu_rmsnorm(xn, x, &e->model, norm_w, QW3_N_EMBD);
        cpu_ok = cpu_matvec_q8_0(&e->model, proj, xn, cpu);
    }
    int gpu_ok = 0;
    if (cpu_ok && qw3_session_create(&s, e, ctx_size) == 0 && s && s->metal) {
        gpu_ok =
            qw3_metal_session_embed_q8_0(s->metal, emb->offset,
                                         (uint32_t)token, QW3_N_EMBD, NULL) &&
            qw3_metal_session_rmsnorm_weight_f32(s->metal, norm_w->offset,
                                                 QW3_N_EMBD, QW3_RMS_EPS, NULL) &&
            qw3_metal_session_matvec_q8_0_x1(s->metal, proj->offset,
                                             QW3_N_EMBD, n_out, gpu);
    }

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < n_out; i++) {
            float d = fabsf(cpu[i] - gpu[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * (double)d;
        }
        rmsdiff = sqrt(rmsdiff / (double)n_out);
    }
    fprintf(fp,
            "metal session qkv: %s token=%d n_in=%d n_out=%u maxdiff=%.7g rmsdiff=%.7g first=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, QW3_N_EMBD, n_out,
            maxdiff, rmsdiff, gpu[0], gpu[1], gpu[2], gpu[3]);
    qw3_session_free(s);
    free(gpu);
    free(cpu);
    free(xn);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_session_prefill_q8_batch_test(qw3_engine *e, int token,
                                                   int ctx_size, FILE *fp) {
    if (!e || !fp || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token; (void)ctx_size;
    fprintf(fp, "metal session prefill q8 batch: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal session prefill q8 batch: Metal backend is not initialized\n");
        return -1;
    }
    uint32_t n_tokens = 4;
    const char *ntok_env = getenv("QW3_METAL_PREFILL_TEST_TOKENS");
    if (ntok_env && ntok_env[0]) {
        char *end = NULL;
        long v = strtol(ntok_env, &end, 10);
        if (end != ntok_env && v > 0 && v <= 256) {
            n_tokens = (uint32_t)v;
        }
    }
    if (token < 0 || token + (int)n_tokens - 1 >= QW3_N_VOCAB) {
        fprintf(fp, "metal session prefill q8 batch: token %d cannot form a %u-token run\n",
                token, n_tokens);
        return -1;
    }
    const qw3_layer_weights *lw = &e->weights.layer[0];
    const qw3_tensor *emb = e->weights.token_embd;
    const qw3_tensor *norm_w = lw->attn_norm;
    const qw3_tensor *proj = lw->linear_qkv_proj;
    const qw3_tensor *gate = lw->linear_gate_proj;
    const qw3_tensor *alpha = lw->linear_ssm_alpha;
    const qw3_tensor *beta = lw->linear_ssm_beta;
    const qw3_tensor *conv_w = lw->linear_conv_weight;
    const qw3_tensor *dt_bias = lw->linear_ssm_dt_bias;
    const qw3_tensor *ssm_a = lw->linear_ssm_a;
    const qw3_tensor *ssm_norm = lw->linear_ssm_norm;
    const qw3_tensor *out_proj = lw->linear_ssm_out;
    const qw3_tensor *router_proj = lw->ffn_gate_inp;
    const uint32_t n_out = (uint32_t)tensor_linear_qkv();
    const uint32_t n_z = (uint32_t)tensor_linear_inner();
    const uint32_t n_gates = QW3_N_LINEAR_V_HEADS;
    const uint32_t n_gate_pair = n_gates * 2u;
    const uint32_t gate_offset = n_out;
    const uint32_t alpha_offset = gate_offset + n_z;
    const uint32_t beta_offset = alpha_offset + n_gates;
    const uint32_t conv_offset = beta_offset + n_gates;
    const uint32_t inner_offset = conv_offset + n_out;
    const uint32_t attn_offset = inner_offset + n_z;
    const uint32_t router_offset = attn_offset + QW3_N_EMBD;
    const uint32_t moe_hidden_offset = router_offset + QW3_N_EXPERT;
    const uint32_t shared_gate_offset =
        moe_hidden_offset + QW3_N_EXPERT_USED * QW3_N_FF_EXP;
    const uint32_t shared_up_offset = shared_gate_offset + QW3_N_FF_SHARED;
    const uint32_t shared_hidden_offset = shared_up_offset + QW3_N_FF_SHARED;
    const uint32_t shared_down_offset =
        shared_hidden_offset + QW3_N_FF_SHARED;
    const uint32_t shared_scalar_offset = shared_down_offset + QW3_N_EMBD;
    const uint32_t stage_stride = shared_scalar_offset + 1u;
    if (emb->type != QW3_TENSOR_Q8_0 || proj->type != QW3_TENSOR_Q8_0 ||
        gate->type != QW3_TENSOR_Q8_0 || out_proj->type != QW3_TENSOR_Q8_0) {
        fprintf(fp, "metal session prefill q8 batch: expected q8_0 embedding/projection tensors\n");
        return -1;
    }
    if (alpha->type != QW3_TENSOR_F32 || beta->type != QW3_TENSOR_F32) {
        fprintf(fp, "metal session prefill q8 batch: expected f32 alpha/beta tensors\n");
        return -1;
    }
    if (router_proj->type != QW3_TENSOR_F32) {
        fprintf(fp, "metal session prefill q8 batch: expected f32 router tensor\n");
        return -1;
    }
    if (lw->ffn_gate_exps->type != QW3_TENSOR_IQ3_S ||
        lw->ffn_up_exps->type != QW3_TENSOR_IQ3_S ||
        lw->ffn_down_exps->type != QW3_TENSOR_IQ4_XS) {
        fprintf(fp, "metal session prefill q8 batch: expected sparse MoE iq3_s/iq4_xs tensors\n");
        return -1;
    }
    if (!lw->ffn_gate_inp_shexp ||
        lw->ffn_gate_inp_shexp->type != QW3_TENSOR_F32 ||
        lw->ffn_gate_shared->type != QW3_TENSOR_Q8_0 ||
        lw->ffn_up_shared->type != QW3_TENSOR_Q8_0 ||
        lw->ffn_down_shared->type != QW3_TENSOR_Q8_0) {
        fprintf(fp, "metal session prefill q8 batch: expected shared MoE q8_0/f32 tensors\n");
        return -1;
    }
    if (!tensor_is_dense_float(conv_w->type)) {
        fprintf(fp, "metal session prefill q8 batch: expected f32 conv tensor\n");
        return -1;
    }
    if (!tensor_is_dense_float(dt_bias->type) ||
        !tensor_is_dense_float(ssm_a->type) ||
        !tensor_is_dense_float(ssm_norm->type)) {
        fprintf(fp, "metal session prefill q8 batch: expected f32 ssm tensors\n");
        return -1;
    }

    uint32_t *toks = qw3_xmalloc((size_t)n_tokens * sizeof(uint32_t));
    for (uint32_t i = 0; i < n_tokens; i++) {
        toks[i] = (uint32_t)token + i;
    }
    qw3_session *s = NULL;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_hidden =
        qw3_xmalloc((size_t)n_tokens * QW3_N_EMBD * sizeof(float));
    float *cpu_ffn =
        qw3_xmalloc((size_t)n_tokens * QW3_N_EMBD * sizeof(float));
    float *gpu_ffn =
        qw3_xmalloc((size_t)n_tokens * QW3_N_EMBD * sizeof(float));
    float *cpu_layer =
        qw3_xcalloc((size_t)n_tokens * QW3_N_EMBD, sizeof(float));
    float *gpu_layer =
        qw3_xcalloc((size_t)n_tokens * QW3_N_EMBD, sizeof(float));
    float *cpu_stage =
        qw3_xcalloc((size_t)n_tokens * stage_stride, sizeof(float));
    float *gpu_stage =
        qw3_xcalloc((size_t)n_tokens * stage_stride, sizeof(float));
    float *conv_state =
        qw3_xcalloc((size_t)n_out * (QW3_N_LINEAR_CONV_K - 1),
                    sizeof(float));
    float *dn_state =
        qw3_xcalloc((size_t)QW3_N_LINEAR_V_HEADS *
                        QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM,
                    sizeof(float));
    float *moe_gate = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *moe_up = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *moe_hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *moe_down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *shared_gate =
        qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *shared_up =
        qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *shared_hidden =
        qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *shared_down =
        qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    bool cpu_ok = true;
    for (uint32_t t = 0; cpu_ok && t < n_tokens; t++) {
        cpu_ok = tensor_read_dense_row(&e->model, emb, toks[t], x);
        if (cpu_ok) {
            memcpy(cpu_hidden + (size_t)t * QW3_N_EMBD, x,
                   (size_t)QW3_N_EMBD * sizeof(float));
            cpu_rmsnorm(x, x, &e->model, norm_w, QW3_N_EMBD);
            cpu_ok = cpu_matvec_q8_0(&e->model, proj, x,
                                     cpu_stage + (size_t)t * stage_stride) &&
                     cpu_matvec_q8_0(&e->model, gate, x,
                                     cpu_stage + (size_t)t * stage_stride +
                                         gate_offset) &&
                     cpu_matvec_dense(&e->model, alpha, x,
                                      cpu_stage + (size_t)t * stage_stride +
                                          alpha_offset) &&
                     cpu_matvec_dense(&e->model, beta, x,
                                      cpu_stage + (size_t)t * stage_stride +
                                          beta_offset) &&
                     cpu_deltanet_conv1d_step(
                         &e->model, lw,
                         cpu_stage + (size_t)t * stage_stride,
                         conv_state,
                         cpu_stage + (size_t)t * stage_stride + conv_offset);
            if (cpu_ok) {
                float *conv = cpu_stage + (size_t)t * stage_stride + conv_offset;
                const uint32_t qk_n =
                    QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
                for (int h = 0; h < QW3_N_LINEAR_QK_HEADS; h++) {
                    cpu_l2_norm_head(conv + h * QW3_N_LINEAR_HEAD_DIM,
                                     conv + h * QW3_N_LINEAR_HEAD_DIM,
                                     QW3_N_LINEAR_HEAD_DIM);
                    cpu_l2_norm_head(conv + qk_n +
                                         h * QW3_N_LINEAR_HEAD_DIM,
                                     conv + qk_n +
                                         h * QW3_N_LINEAR_HEAD_DIM,
                                         QW3_N_LINEAR_HEAD_DIM);
                }
                float *inner = cpu_stage + (size_t)t * stage_stride +
                               inner_offset;
                const float *z = cpu_stage + (size_t)t * stage_stride +
                                 gate_offset;
                const float *alpha_v = cpu_stage + (size_t)t * stage_stride +
                                       alpha_offset;
                const float *beta_v = cpu_stage + (size_t)t * stage_stride +
                                      beta_offset;
                for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
                    const int hk = hv % QW3_N_LINEAR_QK_HEADS;
                    const float *qh = conv + hk * QW3_N_LINEAR_HEAD_DIM;
                    const float *kh = conv + qk_n +
                                      hk * QW3_N_LINEAR_HEAD_DIM;
                    const float *vh = conv + 2 * qk_n +
                                      hv * QW3_N_LINEAR_HEAD_DIM;
                    float *sh = dn_state +
                                (size_t)hv * QW3_N_LINEAR_HEAD_DIM *
                                    QW3_N_LINEAR_HEAD_DIM;
                    const float b = 1.0f / (1.0f + expf(-beta_v[hv]));
                    const float a_raw = alpha_v[hv] +
                        tensor_read_dense_1d(&e->model, dt_bias, (uint64_t)hv);
                    const float g = expf(cpu_softplus(a_raw) *
                        tensor_read_dense_1d(&e->model, ssm_a, (uint64_t)hv));
                    float core[QW3_N_LINEAR_HEAD_DIM];
                    for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                        for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
                            sh[i * QW3_N_LINEAR_HEAD_DIM + j] *= g;
                        }
                    }
                    for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
                        float sk = 0.0f;
                        for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                            sk += sh[i * QW3_N_LINEAR_HEAD_DIM + j] * kh[i];
                        }
                        const float d = (vh[j] - sk) * b;
                        for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                            sh[i * QW3_N_LINEAR_HEAD_DIM + j] += kh[i] * d;
                        }
                    }
                    double ss = 0.0;
                    for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
                        float acc = 0.0f;
                        for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                            acc += sh[i * QW3_N_LINEAR_HEAD_DIM + j] * qh[i];
                        }
                        core[j] = acc / sqrtf((float)QW3_N_LINEAR_HEAD_DIM);
                        ss += (double)core[j] * (double)core[j];
                    }
                    const float scale = 1.0f /
                        sqrtf((float)(ss / QW3_N_LINEAR_HEAD_DIM) +
                              QW3_RMS_EPS);
                    for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
                        inner[hv * QW3_N_LINEAR_HEAD_DIM + j] =
                            core[j] * scale *
                            tensor_read_dense_1d(&e->model, ssm_norm,
                                                 (uint64_t)j) *
                            cpu_silu(z[hv * QW3_N_LINEAR_HEAD_DIM + j]);
                    }
                }
                cpu_ok = cpu_matvec_q8_0(
                    &e->model, out_proj, inner,
                    cpu_stage + (size_t)t * stage_stride + attn_offset);
                if (cpu_ok) {
                    float *resid = cpu_stage + (size_t)t * stage_stride +
                                   attn_offset;
                    for (int i = 0; i < QW3_N_EMBD; i++) {
                        resid[i] += cpu_hidden[(size_t)t * QW3_N_EMBD + i];
                        cpu_layer[(size_t)t * QW3_N_EMBD + i] = resid[i];
                    }
                    cpu_rmsnorm(cpu_ffn + (size_t)t * QW3_N_EMBD, resid,
                                &e->model, lw->ffn_norm, QW3_N_EMBD);
                    cpu_ok = cpu_matvec_dense(
                        &e->model, router_proj,
                        cpu_ffn + (size_t)t * QW3_N_EMBD,
                        cpu_stage + (size_t)t * stage_stride +
                            router_offset);
                    if (cpu_ok) {
                        int ids[QW3_N_EXPERT_USED];
                        float scores[QW3_N_EXPERT_USED];
                        float weights[QW3_N_EXPERT_USED];
                        topk_desc(cpu_stage + (size_t)t * stage_stride +
                                      router_offset,
                                  QW3_N_EXPERT, QW3_N_EXPERT_USED, ids,
                                  scores);
                        float max_route = scores[0];
                        float route_sum = 0.0f;
                        for (int k = 0; k < QW3_N_EXPERT_USED; k++) {
                            weights[k] = expf(scores[k] - max_route);
                            route_sum += weights[k];
                        }
                        for (int k = 0; cpu_ok && k < QW3_N_EXPERT_USED; k++) {
                            weights[k] /= route_sum;
                            cpu_ok =
                                cpu_matvec_iq3_s_expert(
                                    &e->model, lw->ffn_gate_exps, ids[k],
                                    cpu_ffn + (size_t)t * QW3_N_EMBD,
                                    moe_gate) &&
                                cpu_matvec_iq3_s_expert(
                                    &e->model, lw->ffn_up_exps, ids[k],
                                    cpu_ffn + (size_t)t * QW3_N_EMBD,
                                    moe_up);
                            if (!cpu_ok) break;
                            for (int i = 0; i < QW3_N_FF_EXP; i++) {
                                moe_hidden[i] = cpu_silu(moe_gate[i]) * moe_up[i];
                                cpu_stage[(size_t)t * stage_stride +
                                          moe_hidden_offset +
                                          (size_t)k * QW3_N_FF_EXP + i] =
                                    moe_hidden[i];
                            }
                            cpu_ok = cpu_matvec_iq4_xs_expert(
                                &e->model, lw->ffn_down_exps, ids[k],
                                moe_hidden, moe_down);
                            if (!cpu_ok) break;
                            for (int i = 0; i < QW3_N_EMBD; i++) {
                                cpu_layer[(size_t)t * QW3_N_EMBD + i] +=
                                    weights[k] * moe_down[i];
                            }
                        }
                        if (cpu_ok) {
                            cpu_ok =
                                cpu_matvec_q8_0(
                                    &e->model, lw->ffn_gate_shared,
                                    cpu_ffn + (size_t)t * QW3_N_EMBD,
                                    shared_gate) &&
                                cpu_matvec_q8_0(
                                    &e->model, lw->ffn_up_shared,
                                    cpu_ffn + (size_t)t * QW3_N_EMBD,
                                    shared_up);
                        }
                        if (cpu_ok) {
                            for (int i = 0; i < QW3_N_FF_SHARED; i++) {
                                shared_hidden[i] =
                                    cpu_silu(shared_gate[i]) * shared_up[i];
                                cpu_stage[(size_t)t * stage_stride +
                                          shared_hidden_offset + i] =
                                    shared_hidden[i];
                            }
                            cpu_ok = cpu_matvec_q8_0(
                                &e->model, lw->ffn_down_shared,
                                shared_hidden, shared_down);
                        }
                        if (cpu_ok) {
                            float shared_raw = 0.0f;
                            cpu_ok = cpu_dot_dense_1d(
                                &e->model, lw->ffn_gate_inp_shexp,
                                cpu_ffn + (size_t)t * QW3_N_EMBD,
                                &shared_raw);
                            cpu_stage[(size_t)t * stage_stride +
                                      shared_scalar_offset] = shared_raw;
                            const float shared_scale =
                                1.0f / (1.0f + expf(-shared_raw));
                            for (int i = 0; i < QW3_N_EMBD; i++) {
                                cpu_layer[(size_t)t * QW3_N_EMBD + i] +=
                                    shared_scale * shared_down[i];
                            }
                        }
                    }
                }
            }
        }
    }

    int gpu_ok = 0;
    if (cpu_ok && qw3_session_create(&s, e, ctx_size) == 0 && s && s->metal) {
        gpu_ok = qw3_metal_session_batch_embed_q8_0(
                     s->metal, emb->offset, toks, n_tokens, QW3_N_EMBD) &&
                 qw3_metal_session_batch_rmsnorm_weight_f32_x0_to_x1(
                     s->metal, norm_w->offset, n_tokens, QW3_N_EMBD,
                     QW3_RMS_EPS) &&
                 qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(
                     s->metal, proj->offset, n_tokens, QW3_N_EMBD, n_out,
                     0, stage_stride) &&
                 qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(
                     s->metal, gate->offset, n_tokens, QW3_N_EMBD, n_z,
                     gate_offset, stage_stride) &&
                 qw3_metal_session_batch_matmul_f32_pair_x1_to_scratch(
                     s->metal, alpha->offset, beta->offset, n_tokens,
                     QW3_N_EMBD, n_gates, alpha_offset, beta_offset,
                     stage_stride) &&
                 qw3_metal_session_batch_conv1d_step_from_scratch(
                     s->metal, conv_w->offset, 0, n_tokens, n_out, 0,
                     conv_offset, stage_stride) &&
                 qw3_metal_session_batch_l2norm_qk_from_scratch(
                     s->metal, n_tokens, conv_offset, stage_stride,
                     QW3_N_LINEAR_QK_HEADS, QW3_N_LINEAR_HEAD_DIM,
                     QW3_RMS_EPS) &&
                 qw3_metal_session_batch_deltanet_fused_gdn_from_scratch(
                     s->metal, dt_bias->offset, ssm_a->offset,
                     ssm_norm->offset, 0, n_tokens, conv_offset,
                     gate_offset, alpha_offset, beta_offset, inner_offset,
                     stage_stride, QW3_N_LINEAR_QK_HEADS,
                     QW3_N_LINEAR_V_HEADS, QW3_N_LINEAR_HEAD_DIM,
                     QW3_RMS_EPS) &&
                 qw3_metal_session_batch_matmul_q8_0_scratch_to_scratch(
                     s->metal, out_proj->offset, n_tokens, n_z,
                     QW3_N_EMBD, inner_offset, attn_offset, stage_stride) &&
                 qw3_metal_session_batch_residual_rmsnorm_update_x0_from_scratch(
                     s->metal, lw->ffn_norm->offset, n_tokens,
                     QW3_N_EMBD, attn_offset, stage_stride, QW3_RMS_EPS) &&
                 qw3_metal_session_batch_matmul_f32_x1_to_scratch(
                     s->metal, router_proj->offset, n_tokens, QW3_N_EMBD,
                     QW3_N_EXPERT, router_offset, stage_stride) &&
                 qw3_metal_session_batch_sparse_moe_topk_from_router_scratch(
                     s->metal, lw->ffn_gate_exps->offset,
                     lw->ffn_up_exps->offset, lw->ffn_down_exps->offset,
                     (uint32_t)lw->ffn_down_exps->type, n_tokens,
                     QW3_N_EXPERT_USED, QW3_N_EMBD, QW3_N_FF_EXP,
                     router_offset, moe_hidden_offset, stage_stride) &&
                 qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(
                     s->metal, lw->ffn_gate_shared->offset, n_tokens,
                     QW3_N_EMBD, QW3_N_FF_SHARED, shared_gate_offset,
                     stage_stride) &&
                 qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(
                     s->metal, lw->ffn_up_shared->offset, n_tokens,
                     QW3_N_EMBD, QW3_N_FF_SHARED, shared_up_offset,
                     stage_stride) &&
                 qw3_metal_session_batch_silu_mul_scratch_to_scratch(
                     s->metal, n_tokens, QW3_N_FF_SHARED,
                     shared_gate_offset, shared_up_offset,
                     shared_hidden_offset, stage_stride) &&
                 qw3_metal_session_batch_matmul_q8_0_scratch_to_scratch(
                     s->metal, lw->ffn_down_shared->offset, n_tokens,
                     QW3_N_FF_SHARED, QW3_N_EMBD, shared_hidden_offset,
                     shared_down_offset, stage_stride) &&
                 qw3_metal_session_batch_matmul_f32_x1_to_scratch(
                     s->metal, lw->ffn_gate_inp_shexp->offset, n_tokens,
                     QW3_N_EMBD, 1, shared_scalar_offset, stage_stride) &&
                 qw3_metal_session_batch_sigmoid_scale_scratch_add_x0(
                     s->metal, n_tokens, QW3_N_EMBD, shared_down_offset,
                     shared_scalar_offset, stage_stride) &&
                 qw3_metal_session_read_batch_x0(s->metal, gpu_layer,
                                                 n_tokens, QW3_N_EMBD) &&
                 qw3_metal_session_read_batch_x1(s->metal, gpu_ffn, n_tokens,
                                                 QW3_N_EMBD) &&
                 qw3_metal_session_read_batch_scratch(s->metal, gpu_stage,
                                                      n_tokens, stage_stride);
    }

    float qkv_maxdiff = 0.0f, gate_maxdiff = 0.0f, f32_maxdiff = 0.0f;
    float convnorm_maxdiff = 0.0f, inner_maxdiff = 0.0f, attn_maxdiff = 0.0f;
    float ffn_maxdiff = 0.0f, router_maxdiff = 0.0f, layer_maxdiff = 0.0f;
    double qkv_rmsdiff = 0.0, gate_rmsdiff = 0.0, f32_rmsdiff = 0.0;
    double convnorm_rmsdiff = 0.0, inner_rmsdiff = 0.0, attn_rmsdiff = 0.0;
    double ffn_rmsdiff = 0.0, router_rmsdiff = 0.0, layer_rmsdiff = 0.0;
    uint64_t qkv_count = 0, gate_count = 0, f32_count = 0, conv_count = 0;
    uint64_t inner_count = 0, attn_count = 0, ffn_count = 0, router_count = 0;
    uint64_t layer_count = 0;
    if (gpu_ok) {
        for (uint32_t t = 0; t < n_tokens; t++) {
            const size_t base = (size_t)t * stage_stride;
            for (uint32_t i = 0; i < n_out; i++) {
                float d = fabsf(cpu_stage[base + i] - gpu_stage[base + i]);
                if (d > qkv_maxdiff) qkv_maxdiff = d;
                qkv_rmsdiff += (double)d * (double)d;
                qkv_count++;
            }
            for (uint32_t i = 0; i < n_z; i++) {
                float d = fabsf(cpu_stage[base + gate_offset + i] -
                                gpu_stage[base + gate_offset + i]);
                if (d > gate_maxdiff) gate_maxdiff = d;
                gate_rmsdiff += (double)d * (double)d;
                gate_count++;
            }
            for (uint32_t i = 0; i < n_gate_pair; i++) {
                float d = fabsf(cpu_stage[base + alpha_offset + i] -
                                gpu_stage[base + alpha_offset + i]);
                if (d > f32_maxdiff) f32_maxdiff = d;
                f32_rmsdiff += (double)d * (double)d;
                f32_count++;
            }
            for (uint32_t i = 0; i < n_out; i++) {
                float d = fabsf(cpu_stage[base + conv_offset + i] -
                                gpu_stage[base + conv_offset + i]);
                if (d > convnorm_maxdiff) convnorm_maxdiff = d;
                convnorm_rmsdiff += (double)d * (double)d;
                conv_count++;
            }
            for (uint32_t i = 0; i < n_z; i++) {
                float d = fabsf(cpu_stage[base + inner_offset + i] -
                                gpu_stage[base + inner_offset + i]);
                if (d > inner_maxdiff) inner_maxdiff = d;
                inner_rmsdiff += (double)d * (double)d;
                inner_count++;
            }
            for (uint32_t i = 0; i < QW3_N_EMBD; i++) {
                float cpu_attn = cpu_stage[base + attn_offset + i] -
                                 cpu_hidden[(size_t)t * QW3_N_EMBD + i];
                float d = fabsf(cpu_attn - gpu_stage[base + attn_offset + i]);
                if (d > attn_maxdiff) attn_maxdiff = d;
                attn_rmsdiff += (double)d * (double)d;
                attn_count++;
            }
            for (uint32_t i = 0; i < QW3_N_EMBD; i++) {
                float d = fabsf(cpu_ffn[(size_t)t * QW3_N_EMBD + i] -
                                gpu_ffn[(size_t)t * QW3_N_EMBD + i]);
                if (d > ffn_maxdiff) ffn_maxdiff = d;
                ffn_rmsdiff += (double)d * (double)d;
                ffn_count++;
            }
            for (uint32_t i = 0; i < QW3_N_EXPERT; i++) {
                float d = fabsf(cpu_stage[base + router_offset + i] -
                                gpu_stage[base + router_offset + i]);
                if (d > router_maxdiff) router_maxdiff = d;
                router_rmsdiff += (double)d * (double)d;
                router_count++;
            }
            for (uint32_t i = 0; i < QW3_N_EMBD; i++) {
                float d = fabsf(cpu_layer[(size_t)t * QW3_N_EMBD + i] -
                                gpu_layer[(size_t)t * QW3_N_EMBD + i]);
                if (d > layer_maxdiff) layer_maxdiff = d;
                layer_rmsdiff += (double)d * (double)d;
                layer_count++;
            }
        }
        qkv_rmsdiff = sqrt(qkv_rmsdiff / (double)qkv_count);
        gate_rmsdiff = sqrt(gate_rmsdiff / (double)gate_count);
        f32_rmsdiff = sqrt(f32_rmsdiff / (double)f32_count);
        convnorm_rmsdiff = sqrt(convnorm_rmsdiff / (double)conv_count);
        inner_rmsdiff = sqrt(inner_rmsdiff / (double)inner_count);
        attn_rmsdiff = sqrt(attn_rmsdiff / (double)attn_count);
        ffn_rmsdiff = sqrt(ffn_rmsdiff / (double)ffn_count);
        router_rmsdiff = sqrt(router_rmsdiff / (double)router_count);
        layer_rmsdiff = sqrt(layer_rmsdiff / (double)layer_count);
    }
    fprintf(fp,
            "metal session prefill q8 batch: %s token=%d n_tokens=%u stride=%u qkv_maxdiff=%.7g qkv_rmsdiff=%.7g gate_maxdiff=%.7g gate_rmsdiff=%.7g f32_maxdiff=%.7g f32_rmsdiff=%.7g convnorm_maxdiff=%.7g convnorm_rmsdiff=%.7g inner_maxdiff=%.7g inner_rmsdiff=%.7g attn_maxdiff=%.7g attn_rmsdiff=%.7g ffn_maxdiff=%.7g ffn_rmsdiff=%.7g router_maxdiff=%.7g router_rmsdiff=%.7g layer_maxdiff=%.7g layer_rmsdiff=%.7g qkv_first=[%.7g %.7g %.7g %.7g] router_first=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, n_tokens, stage_stride,
            qkv_maxdiff, qkv_rmsdiff, gate_maxdiff, gate_rmsdiff,
            f32_maxdiff, f32_rmsdiff, convnorm_maxdiff, convnorm_rmsdiff,
            inner_maxdiff, inner_rmsdiff, attn_maxdiff, attn_rmsdiff,
            ffn_maxdiff, ffn_rmsdiff, router_maxdiff, router_rmsdiff,
            layer_maxdiff, layer_rmsdiff,
            gpu_stage[0], gpu_stage[1], gpu_stage[2], gpu_stage[3],
            gpu_stage[router_offset], gpu_stage[router_offset + 1],
            gpu_stage[router_offset + 2], gpu_stage[router_offset + 3]);
    qw3_session_free(s);
    free(shared_down);
    free(shared_hidden);
    free(shared_up);
    free(shared_gate);
    free(moe_down);
    free(moe_hidden);
    free(moe_up);
    free(moe_gate);
    free(dn_state);
    free(conv_state);
    free(gpu_stage);
    free(cpu_stage);
    free(gpu_layer);
    free(cpu_layer);
    free(gpu_ffn);
    free(cpu_ffn);
    free(cpu_hidden);
    free(toks);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_session_gqa_prefill_batch_test(qw3_engine *e, int token,
                                                    int ctx_size, FILE *fp) {
    if (!e || !fp || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token; (void)ctx_size;
    fprintf(fp, "metal session gqa prefill batch: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal session gqa prefill batch: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token + 3 >= QW3_N_VOCAB) {
        fprintf(fp, "metal session gqa prefill batch: token %d cannot form a 4-token run\n",
                token);
        return -1;
    }

    const int il = 3;
    const uint32_t n_tokens = 4;
    const qw3_layer_weights *lw = &e->weights.layer[il];
    const qw3_tensor *emb = e->weights.token_embd;
    const uint32_t q_n = QW3_N_HEAD * QW3_N_HEAD_DIM;
    const uint32_t qg_n = (uint32_t)tensor_cols_qg();
    const uint32_t kv_n = (uint32_t)tensor_cols_kv();
    const uint32_t qg_offset = 0;
    const uint32_t k_offset = qg_offset + qg_n;
    const uint32_t v_offset = k_offset + kv_n;
    const uint32_t q_tmp_offset = v_offset + kv_n;
    const uint32_t k_tmp_offset = q_tmp_offset + q_n;
    const uint32_t q_rope_offset = k_tmp_offset + kv_n;
    const uint32_t k_rope_offset = q_rope_offset + q_n;
    const uint32_t gate_offset = k_rope_offset + kv_n;
    const uint32_t inner_offset = gate_offset + q_n;
    const uint32_t out_offset = inner_offset + q_n;
    const uint32_t stage_stride = out_offset + QW3_N_EMBD;

    if (qg_n != 2u * q_n ||
        emb->type != QW3_TENSOR_Q8_0 ||
        lw->attn_q_proj->type != QW3_TENSOR_Q8_0 ||
        lw->attn_k_proj->type != QW3_TENSOR_Q8_0 ||
        lw->attn_v_proj->type != QW3_TENSOR_Q8_0 ||
        lw->attn_o_proj->type != QW3_TENSOR_Q8_0 ||
        !tensor_is_dense_float(lw->attn_q_norm->type) ||
        !tensor_is_dense_float(lw->attn_k_norm->type)) {
        fprintf(fp, "metal session gqa prefill batch: unexpected layer-3 tensor layout\n");
        return -1;
    }

    uint32_t toks[4] = {
        (uint32_t)token,
        (uint32_t)token + 1u,
        (uint32_t)token + 2u,
        (uint32_t)token + 3u,
    };
    qw3_session *s = NULL;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_q = qw3_xmalloc((size_t)n_tokens * q_n * sizeof(float));
    float *cpu_k = qw3_xmalloc((size_t)n_tokens * kv_n * sizeof(float));
    float *cpu_v = qw3_xmalloc((size_t)n_tokens * kv_n * sizeof(float));
    float *cpu_gate = qw3_xmalloc((size_t)n_tokens * q_n * sizeof(float));
    float *cpu_inner = qw3_xmalloc((size_t)n_tokens * q_n * sizeof(float));
    float *cpu_out = qw3_xmalloc((size_t)n_tokens * QW3_N_EMBD * sizeof(float));
    float *gpu_stage =
        qw3_xcalloc((size_t)n_tokens * stage_stride, sizeof(float));

    bool cpu_ok = true;
    for (uint32_t t = 0; cpu_ok && t < n_tokens; t++) {
        cpu_ok = tensor_read_dense_row(&e->model, emb, toks[t], x) &&
                 cpu_gqa_project_token(
                     e, il, (int)t, x,
                     cpu_q + (size_t)t * q_n,
                     cpu_k + (size_t)t * kv_n,
                     cpu_v + (size_t)t * kv_n,
                     cpu_gate + (size_t)t * q_n);
        if (cpu_ok) {
            cpu_gqa_attend_inner(
                cpu_q + (size_t)t * q_n,
                cpu_gate + (size_t)t * q_n,
                cpu_k, cpu_v, (int)t + 1,
                cpu_inner + (size_t)t * q_n);
            cpu_ok = cpu_matvec_q8_0(
                &e->model, lw->attn_o_proj,
                cpu_inner + (size_t)t * q_n,
                cpu_out + (size_t)t * QW3_N_EMBD);
        }
    }

    int gpu_ok = 0;
    if (cpu_ok && qw3_session_create(&s, e, ctx_size) == 0 && s && s->metal) {
        gpu_ok = qw3_metal_session_batch_embed_q8_0(
                     s->metal, emb->offset, toks, n_tokens, QW3_N_EMBD) &&
                 qw3_metal_session_batch_rmsnorm_weight_f32_x0_to_x1(
                     s->metal, lw->attn_norm->offset, n_tokens,
                     QW3_N_EMBD, QW3_RMS_EPS) &&
                 qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(
                     s->metal, lw->attn_q_proj->offset, n_tokens,
                     QW3_N_EMBD, qg_n, qg_offset, stage_stride) &&
                 qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(
                     s->metal, lw->attn_k_proj->offset, n_tokens,
                     QW3_N_EMBD, kv_n, k_offset, stage_stride) &&
                 qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(
                     s->metal, lw->attn_v_proj->offset, n_tokens,
                     QW3_N_EMBD, kv_n, v_offset, stage_stride) &&
                 qw3_metal_session_batch_gqa_norm_rope_from_scratch(
                     s->metal, lw->attn_q_norm->offset,
                     lw->attn_k_norm->offset, n_tokens, QW3_N_HEAD,
                     QW3_N_HEAD_KV, QW3_N_HEAD_DIM, QW3_ROPE_DIM, 0,
                     QW3_ROPE_THETA, QW3_RMS_EPS, qg_offset, k_offset,
                     q_tmp_offset, k_tmp_offset, q_rope_offset,
                     k_rope_offset, gate_offset, stage_stride) &&
                 qw3_metal_session_batch_gqa_write_cache_from_scratch(
                     s->metal, 0, 0, n_tokens, QW3_N_HEAD_KV,
                     QW3_N_HEAD_DIM, k_rope_offset, v_offset,
                     stage_stride) &&
                 qw3_metal_session_batch_gqa_cached_attn_from_scratch(
                     s->metal, 0, 0, n_tokens, QW3_N_HEAD,
                     QW3_N_HEAD_KV, QW3_N_HEAD_DIM, q_rope_offset,
                     gate_offset, inner_offset, stage_stride) &&
                 qw3_metal_session_batch_matmul_q8_0_scratch_to_scratch(
                     s->metal, lw->attn_o_proj->offset, n_tokens,
                     q_n, QW3_N_EMBD, inner_offset, out_offset,
                     stage_stride) &&
                 qw3_metal_session_read_batch_scratch(
                     s->metal, gpu_stage, n_tokens, stage_stride);
    }

    float q_maxdiff = 0.0f, k_maxdiff = 0.0f, v_maxdiff = 0.0f;
    float gate_maxdiff = 0.0f, inner_maxdiff = 0.0f, out_maxdiff = 0.0f;
    double q_rmsdiff = 0.0, k_rmsdiff = 0.0, v_rmsdiff = 0.0;
    double gate_rmsdiff = 0.0, inner_rmsdiff = 0.0, out_rmsdiff = 0.0;
    uint64_t q_count = 0, k_count = 0, v_count = 0;
    uint64_t gate_count = 0, inner_count = 0, out_count = 0;
    if (gpu_ok) {
        for (uint32_t t = 0; t < n_tokens; t++) {
            const size_t base = (size_t)t * stage_stride;
            for (uint32_t i = 0; i < q_n; i++) {
                float d = fabsf(cpu_q[(size_t)t * q_n + i] -
                                gpu_stage[base + q_rope_offset + i]);
                if (d > q_maxdiff) q_maxdiff = d;
                q_rmsdiff += (double)d * (double)d;
                q_count++;

                d = fabsf(cpu_gate[(size_t)t * q_n + i] -
                          gpu_stage[base + gate_offset + i]);
                if (d > gate_maxdiff) gate_maxdiff = d;
                gate_rmsdiff += (double)d * (double)d;
                gate_count++;

                d = fabsf(cpu_inner[(size_t)t * q_n + i] -
                          gpu_stage[base + inner_offset + i]);
                if (d > inner_maxdiff) inner_maxdiff = d;
                inner_rmsdiff += (double)d * (double)d;
                inner_count++;
            }
            for (uint32_t i = 0; i < kv_n; i++) {
                float d = fabsf(cpu_k[(size_t)t * kv_n + i] -
                                gpu_stage[base + k_rope_offset + i]);
                if (d > k_maxdiff) k_maxdiff = d;
                k_rmsdiff += (double)d * (double)d;
                k_count++;

                d = fabsf(cpu_v[(size_t)t * kv_n + i] -
                          gpu_stage[base + v_offset + i]);
                if (d > v_maxdiff) v_maxdiff = d;
                v_rmsdiff += (double)d * (double)d;
                v_count++;
            }
            for (uint32_t i = 0; i < QW3_N_EMBD; i++) {
                float d = fabsf(cpu_out[(size_t)t * QW3_N_EMBD + i] -
                                gpu_stage[base + out_offset + i]);
                if (d > out_maxdiff) out_maxdiff = d;
                out_rmsdiff += (double)d * (double)d;
                out_count++;
            }
        }
        q_rmsdiff = sqrt(q_rmsdiff / (double)q_count);
        k_rmsdiff = sqrt(k_rmsdiff / (double)k_count);
        v_rmsdiff = sqrt(v_rmsdiff / (double)v_count);
        gate_rmsdiff = sqrt(gate_rmsdiff / (double)gate_count);
        inner_rmsdiff = sqrt(inner_rmsdiff / (double)inner_count);
        out_rmsdiff = sqrt(out_rmsdiff / (double)out_count);
    }

    fprintf(fp,
            "metal session gqa prefill batch: %s token=%d n_tokens=%u stride=%u q_maxdiff=%.7g q_rmsdiff=%.7g k_maxdiff=%.7g k_rmsdiff=%.7g v_maxdiff=%.7g v_rmsdiff=%.7g gate_maxdiff=%.7g gate_rmsdiff=%.7g inner_maxdiff=%.7g inner_rmsdiff=%.7g out_maxdiff=%.7g out_rmsdiff=%.7g q0=[%.7g %.7g %.7g %.7g] inner0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, n_tokens, stage_stride,
            q_maxdiff, q_rmsdiff, k_maxdiff, k_rmsdiff,
            v_maxdiff, v_rmsdiff, gate_maxdiff, gate_rmsdiff,
            inner_maxdiff, inner_rmsdiff, out_maxdiff, out_rmsdiff,
            gpu_stage[q_rope_offset], gpu_stage[q_rope_offset + 1],
            gpu_stage[q_rope_offset + 2], gpu_stage[q_rope_offset + 3],
            gpu_stage[inner_offset], gpu_stage[inner_offset + 1],
            gpu_stage[inner_offset + 2], gpu_stage[inner_offset + 3]);

    qw3_session_free(s);
    free(gpu_stage);
    free(cpu_out);
    free(cpu_inner);
    free(cpu_gate);
    free(cpu_v);
    free(cpu_k);
    free(cpu_q);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_session_z_test(qw3_engine *e, int token,
                                    int ctx_size, FILE *fp) {
    if (!e || !fp || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token; (void)ctx_size;
    fprintf(fp, "metal session z: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal session z: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal session z: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_layer_weights *lw = &e->weights.layer[0];
    const qw3_tensor *emb = e->weights.token_embd;
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t n_z = (uint32_t)tensor_linear_inner();
    if (emb->type != QW3_TENSOR_Q8_0 ||
        lw->linear_qkv_proj->type != QW3_TENSOR_Q8_0 ||
        lw->linear_gate_proj->type != QW3_TENSOR_Q8_0) {
        fprintf(fp, "metal session z: expected q8_0 embedding/qkv/z tensors\n");
        return -1;
    }

    qw3_session *s = NULL;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_z = qw3_xmalloc((size_t)n_z * sizeof(float));
    float *gpu_z = qw3_xmalloc((size_t)n_z * sizeof(float));
    bool cpu_ok = tensor_read_dense_row(&e->model, emb, (uint64_t)token, x);
    if (cpu_ok) {
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        cpu_ok = cpu_matvec_q8_0(&e->model, lw->linear_gate_proj, xn, cpu_z);
    }
    int gpu_ok = 0;
    if (cpu_ok && qw3_session_create(&s, e, ctx_size) == 0 && s && s->metal) {
        gpu_ok =
            qw3_metal_session_embed_q8_0(s->metal, emb->offset,
                                         (uint32_t)token, QW3_N_EMBD, NULL) &&
            qw3_metal_session_rmsnorm_weight_f32(s->metal, lw->attn_norm->offset,
                                                 QW3_N_EMBD, QW3_RMS_EPS, NULL) &&
            qw3_metal_session_matvec_q8_0_x1_to_scratch(
                s->metal, lw->linear_qkv_proj->offset,
                QW3_N_EMBD, n_qkv, 0, NULL) &&
            qw3_metal_session_matvec_q8_0_x1_to_scratch(
                s->metal, lw->linear_gate_proj->offset,
                QW3_N_EMBD, n_z, n_qkv, gpu_z);
    }

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < n_z; i++) {
            float d = fabsf(cpu_z[i] - gpu_z[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * (double)d;
        }
        rmsdiff = sqrt(rmsdiff / (double)n_z);
    }
    fprintf(fp,
            "metal session z: %s token=%d n_in=%d n_out=%u scratch_offset=%u maxdiff=%.7g rmsdiff=%.7g first=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, QW3_N_EMBD, n_z, n_qkv,
            maxdiff, rmsdiff, gpu_z[0], gpu_z[1], gpu_z[2], gpu_z[3]);
    qw3_session_free(s);
    free(gpu_z);
    free(cpu_z);
    free(xn);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_session_conv_test(qw3_engine *e, int token,
                                       int ctx_size, FILE *fp) {
    if (!e || !fp || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token; (void)ctx_size;
    fprintf(fp, "metal session conv1d: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal session conv1d: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal session conv1d: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_layer_weights *lw = &e->weights.layer[0];
    const qw3_tensor *emb = e->weights.token_embd;
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    if (emb->type != QW3_TENSOR_Q8_0 ||
        lw->linear_qkv_proj->type != QW3_TENSOR_Q8_0) {
        fprintf(fp, "metal session conv1d: expected q8_0 embedding/qkv tensors\n");
        return -1;
    }

    qw3_session *s = NULL;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *cpu = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *gpu = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *conv_state = qw3_xcalloc((size_t)n_qkv * (QW3_N_LINEAR_CONV_K - 1),
                                    sizeof(float));
    bool cpu_ok = tensor_read_dense_row(&e->model, emb, (uint64_t)token, x);
    if (cpu_ok) {
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        cpu_ok = cpu_matvec_q8_0(&e->model, lw->linear_qkv_proj, xn, qkv) &&
                 cpu_deltanet_conv1d_step(&e->model, lw, qkv, conv_state, cpu);
    }
    int gpu_ok = 0;
    if (cpu_ok && qw3_session_create(&s, e, ctx_size) == 0 && s && s->metal) {
        gpu_ok =
            qw3_metal_session_embed_q8_0(s->metal, emb->offset,
                                         (uint32_t)token, QW3_N_EMBD, NULL) &&
            qw3_metal_session_rmsnorm_weight_f32(s->metal, lw->attn_norm->offset,
                                                 QW3_N_EMBD, QW3_RMS_EPS, NULL) &&
            qw3_metal_session_matvec_q8_0_x1_to_scratch(
                s->metal, lw->linear_qkv_proj->offset,
                QW3_N_EMBD, n_qkv, 0, NULL) &&
            qw3_metal_session_conv1d_zero_from_scratch(
                s->metal, lw->linear_conv_weight->offset, n_qkv, gpu);
    }

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < n_qkv; i++) {
            float d = fabsf(cpu[i] - gpu[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * (double)d;
        }
        rmsdiff = sqrt(rmsdiff / (double)n_qkv);
    }
    fprintf(fp,
            "metal session conv1d: %s token=%d n=%u maxdiff=%.7g rmsdiff=%.7g first=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, n_qkv,
            maxdiff, rmsdiff, gpu[0], gpu[1], gpu[2], gpu[3]);
    qw3_session_free(s);
    free(conv_state);
    free(gpu);
    free(cpu);
    free(qkv);
    free(xn);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_session_l2norm_test(qw3_engine *e, int token,
                                         int ctx_size, FILE *fp) {
    if (!e || !fp || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token; (void)ctx_size;
    fprintf(fp, "metal session l2norm: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal session l2norm: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal session l2norm: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_layer_weights *lw = &e->weights.layer[0];
    const qw3_tensor *emb = e->weights.token_embd;
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    if (emb->type != QW3_TENSOR_Q8_0 ||
        lw->linear_qkv_proj->type != QW3_TENSOR_Q8_0) {
        fprintf(fp, "metal session l2norm: expected q8_0 embedding/qkv tensors\n");
        return -1;
    }

    qw3_session *s = NULL;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *cpu_q = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *cpu_k = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *gpu_q = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *gpu_k = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *conv_state = qw3_xcalloc((size_t)n_qkv * (QW3_N_LINEAR_CONV_K - 1),
                                    sizeof(float));
    bool cpu_ok = tensor_read_dense_row(&e->model, emb, (uint64_t)token, x);
    if (cpu_ok) {
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        cpu_ok = cpu_matvec_q8_0(&e->model, lw->linear_qkv_proj, xn, qkv) &&
                 cpu_deltanet_conv1d_step(&e->model, lw, qkv, conv_state, conv);
    }
    if (cpu_ok) {
        const float *qraw = conv;
        const float *kraw = conv + qk_n;
        for (int h = 0; h < QW3_N_LINEAR_QK_HEADS; h++) {
            cpu_l2_norm_head(cpu_q + h * QW3_N_LINEAR_HEAD_DIM,
                             qraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
            cpu_l2_norm_head(cpu_k + h * QW3_N_LINEAR_HEAD_DIM,
                             kraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
        }
    }

    int gpu_ok = 0;
    if (cpu_ok && qw3_session_create(&s, e, ctx_size) == 0 && s && s->metal) {
        gpu_ok =
            qw3_metal_session_embed_q8_0(s->metal, emb->offset,
                                         (uint32_t)token, QW3_N_EMBD, NULL) &&
            qw3_metal_session_rmsnorm_weight_f32(s->metal, lw->attn_norm->offset,
                                                 QW3_N_EMBD, QW3_RMS_EPS, NULL) &&
            qw3_metal_session_matvec_q8_0_x1_to_scratch(
                s->metal, lw->linear_qkv_proj->offset,
                QW3_N_EMBD, n_qkv, 0, NULL) &&
            qw3_metal_session_conv1d_zero_from_scratch(
                s->metal, lw->linear_conv_weight->offset, n_qkv, NULL) &&
            qw3_metal_session_l2norm_qk_from_conv(
                s->metal, QW3_N_LINEAR_QK_HEADS, QW3_N_LINEAR_HEAD_DIM,
                QW3_RMS_EPS, gpu_q, gpu_k);
    }

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    double qnorm0 = 0.0, knorm0 = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < qk_n; i++) {
            float dq = fabsf(cpu_q[i] - gpu_q[i]);
            float dk = fabsf(cpu_k[i] - gpu_k[i]);
            if (dq > maxdiff) maxdiff = dq;
            if (dk > maxdiff) maxdiff = dk;
            rmsdiff += (double)dq * (double)dq + (double)dk * (double)dk;
        }
        for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
            qnorm0 += (double)gpu_q[i] * (double)gpu_q[i];
            knorm0 += (double)gpu_k[i] * (double)gpu_k[i];
        }
        rmsdiff = sqrt(rmsdiff / (double)(2 * qk_n));
        qnorm0 = sqrt(qnorm0);
        knorm0 = sqrt(knorm0);
    }
    fprintf(fp,
            "metal session l2norm: %s token=%d n=%u maxdiff=%.7g rmsdiff=%.7g q0_norm=%.7g k0_norm=%.7g q0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, qk_n,
            maxdiff, rmsdiff, qnorm0, knorm0,
            gpu_q[0], gpu_q[1], gpu_q[2], gpu_q[3]);
    qw3_session_free(s);
    free(conv_state);
    free(gpu_k);
    free(gpu_q);
    free(cpu_k);
    free(cpu_q);
    free(conv);
    free(qkv);
    free(xn);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_session_gates_test(qw3_engine *e, int token,
                                        int ctx_size, FILE *fp) {
    if (!e || !fp || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token; (void)ctx_size;
    fprintf(fp, "metal session gates: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal session gates: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal session gates: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_layer_weights *lw = &e->weights.layer[0];
    const qw3_tensor *emb = e->weights.token_embd;
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t n_z = (uint32_t)tensor_linear_inner();
    const uint32_t alpha_offset = n_qkv + n_z;
    const uint32_t beta_offset = alpha_offset + QW3_N_LINEAR_V_HEADS;

    qw3_session *s = NULL;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_alpha = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *cpu_beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *gpu_alpha = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *gpu_beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));

    bool cpu_ok = tensor_read_dense_row(&e->model, emb, (uint64_t)token, x);
    if (cpu_ok) {
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        cpu_ok = cpu_matvec_dense(&e->model, lw->linear_ssm_alpha, xn, cpu_alpha) &&
                 cpu_matvec_dense(&e->model, lw->linear_ssm_beta, xn, cpu_beta);
    }
    int gpu_ok = 0;
    if (cpu_ok && qw3_session_create(&s, e, ctx_size) == 0 && s && s->metal) {
        gpu_ok =
            qw3_metal_session_embed_q8_0(s->metal, emb->offset,
                                         (uint32_t)token, QW3_N_EMBD, NULL) &&
            qw3_metal_session_rmsnorm_weight_f32(s->metal, lw->attn_norm->offset,
                                                 QW3_N_EMBD, QW3_RMS_EPS, NULL) &&
            qw3_metal_session_matvec_f32_x1_to_scratch(
                s->metal, lw->linear_ssm_alpha->offset,
                QW3_N_EMBD, QW3_N_LINEAR_V_HEADS, alpha_offset, gpu_alpha) &&
            qw3_metal_session_matvec_f32_x1_to_scratch(
                s->metal, lw->linear_ssm_beta->offset,
                QW3_N_EMBD, QW3_N_LINEAR_V_HEADS, beta_offset, gpu_beta);
    }

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_LINEAR_V_HEADS; i++) {
            float da = fabsf(cpu_alpha[i] - gpu_alpha[i]);
            float db = fabsf(cpu_beta[i] - gpu_beta[i]);
            if (da > maxdiff) maxdiff = da;
            if (db > maxdiff) maxdiff = db;
            rmsdiff += (double)da * (double)da + (double)db * (double)db;
        }
        rmsdiff = sqrt(rmsdiff / (double)(2 * QW3_N_LINEAR_V_HEADS));
    }
    fprintf(fp,
            "metal session gates: %s token=%d maxdiff=%.7g rmsdiff=%.7g alpha0=[%.7g %.7g %.7g %.7g] beta0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, maxdiff, rmsdiff,
            gpu_alpha[0], gpu_alpha[1], gpu_alpha[2], gpu_alpha[3],
            gpu_beta[0], gpu_beta[1], gpu_beta[2], gpu_beta[3]);
    qw3_session_free(s);
    free(gpu_beta); free(gpu_alpha); free(cpu_beta); free(cpu_alpha);
    free(xn); free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_session_recur_zero_test(qw3_engine *e, int token,
                                             int ctx_size, FILE *fp) {
    if (!e || !fp || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token; (void)ctx_size;
    fprintf(fp, "metal session recur zero: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal session recur zero: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal session recur zero: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_layer_weights *lw = &e->weights.layer[0];
    const qw3_tensor *emb = e->weights.token_embd;
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t inner_n = (uint32_t)tensor_linear_inner();
    const uint32_t alpha_offset = n_qkv + inner_n;
    const uint32_t beta_offset = alpha_offset + QW3_N_LINEAR_V_HEADS;
    const uint32_t state_n = QW3_N_LINEAR_V_HEADS *
                             QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM;

    qw3_session *s = NULL;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *q = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *k = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta_sig = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *cpu_state = qw3_xcalloc((size_t)state_n, sizeof(float));
    float *gpu_state = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *cpu_core = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *gpu_core = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *conv_state = qw3_xcalloc((size_t)n_qkv * (QW3_N_LINEAR_CONV_K - 1),
                                    sizeof(float));

    bool cpu_ok = tensor_read_dense_row(&e->model, emb, (uint64_t)token, x);
    if (cpu_ok) {
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        cpu_ok = cpu_matvec_q8_0(&e->model, lw->linear_qkv_proj, xn, qkv) &&
                 cpu_matvec_dense(&e->model, lw->linear_ssm_beta, xn, beta) &&
                 cpu_deltanet_conv1d_step(&e->model, lw, qkv, conv_state, conv);
    }
    if (cpu_ok) {
        const float *qraw = conv;
        const float *kraw = conv + qk_n;
        for (int h = 0; h < QW3_N_LINEAR_QK_HEADS; h++) {
            cpu_l2_norm_head(q + h * QW3_N_LINEAR_HEAD_DIM,
                             qraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
            cpu_l2_norm_head(k + h * QW3_N_LINEAR_HEAD_DIM,
                             kraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            const int hk = hv % QW3_N_LINEAR_QK_HEADS;
            const float *qh = q + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *kh = k + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *vh = conv + qk_n * 2 + hv * QW3_N_LINEAR_HEAD_DIM;
            float *sh = cpu_state + (uint64_t)hv * QW3_N_LINEAR_HEAD_DIM *
                        QW3_N_LINEAR_HEAD_DIM;
            float *oh = cpu_core + hv * QW3_N_LINEAR_HEAD_DIM;
            const float bh = 1.0f / (1.0f + expf(-beta[hv]));
            beta_sig[hv] = bh;
            for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
                const float d = bh * vh[j];
                float out = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    const int idx = i * QW3_N_LINEAR_HEAD_DIM + j;
                    const float sv = kh[i] * d;
                    sh[idx] = sv;
                    out += sv * qh[i];
                }
                oh[j] = out / sqrtf((float)QW3_N_LINEAR_HEAD_DIM);
            }
        }
    }

    int gpu_ok = 0;
    if (cpu_ok && qw3_session_create(&s, e, ctx_size) == 0 && s && s->metal) {
        gpu_ok =
            qw3_metal_session_embed_q8_0(s->metal, emb->offset,
                                         (uint32_t)token, QW3_N_EMBD, NULL) &&
            qw3_metal_session_rmsnorm_weight_f32(s->metal, lw->attn_norm->offset,
                                                 QW3_N_EMBD, QW3_RMS_EPS, NULL) &&
            qw3_metal_session_matvec_q8_0_x1_to_scratch(
                s->metal, lw->linear_qkv_proj->offset,
                QW3_N_EMBD, n_qkv, 0, NULL) &&
            qw3_metal_session_matvec_f32_x1_to_scratch(
                s->metal, lw->linear_ssm_beta->offset,
                QW3_N_EMBD, QW3_N_LINEAR_V_HEADS, beta_offset, NULL) &&
            qw3_metal_session_conv1d_zero_from_scratch(
                s->metal, lw->linear_conv_weight->offset, n_qkv, NULL) &&
            qw3_metal_session_l2norm_qk_from_conv(
                s->metal, QW3_N_LINEAR_QK_HEADS, QW3_N_LINEAR_HEAD_DIM,
                QW3_RMS_EPS, NULL, NULL) &&
            qw3_metal_session_deltanet_recur_zero_from_buffers(
                s->metal, beta_sig, QW3_N_LINEAR_QK_HEADS,
                QW3_N_LINEAR_V_HEADS, QW3_N_LINEAR_HEAD_DIM,
                gpu_state, gpu_core);
    }

    float core_maxdiff = 0.0f, state_maxdiff = 0.0f;
    double core_rmsdiff = 0.0, state_rmsdiff = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < inner_n; i++) {
            float d = fabsf(cpu_core[i] - gpu_core[i]);
            if (d > core_maxdiff) core_maxdiff = d;
            core_rmsdiff += (double)d * (double)d;
        }
        for (uint32_t i = 0; i < state_n; i++) {
            float d = fabsf(cpu_state[i] - gpu_state[i]);
            if (d > state_maxdiff) state_maxdiff = d;
            state_rmsdiff += (double)d * (double)d;
        }
        core_rmsdiff = sqrt(core_rmsdiff / (double)inner_n);
        state_rmsdiff = sqrt(state_rmsdiff / (double)state_n);
    }
    fprintf(fp,
            "metal session recur zero: %s token=%d core_maxdiff=%.7g core_rmsdiff=%.7g state_maxdiff=%.7g state_rmsdiff=%.7g core0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, core_maxdiff, core_rmsdiff,
            state_maxdiff, state_rmsdiff,
            gpu_core[0], gpu_core[1], gpu_core[2], gpu_core[3]);
    qw3_session_free(s);
    free(conv_state); free(gpu_core); free(cpu_core); free(gpu_state);
    free(cpu_state); free(beta_sig); free(beta); free(k); free(q); free(conv);
    free(qkv); free(xn); free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_session_recur_step_test(qw3_engine *e, int token,
                                             int ctx_size, FILE *fp) {
    if (!e || !fp || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token; (void)ctx_size;
    fprintf(fp, "metal session recur step: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal session recur step: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal session recur step: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_layer_weights *lw = &e->weights.layer[0];
    const qw3_tensor *emb = e->weights.token_embd;
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t inner_n = (uint32_t)tensor_linear_inner();
    const uint32_t alpha_offset = n_qkv + inner_n;
    const uint32_t beta_offset = alpha_offset + QW3_N_LINEAR_V_HEADS;
    const uint32_t conv_state_n = n_qkv * (QW3_N_LINEAR_CONV_K - 1);
    const uint32_t state_n = QW3_N_LINEAR_V_HEADS *
                             QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM;

    qw3_session *s = NULL;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *q = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *k = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *alpha = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta_sig = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *gamma = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *cpu_conv_state = qw3_xcalloc((size_t)conv_state_n, sizeof(float));
    float *gpu_conv_state = qw3_xmalloc((size_t)conv_state_n * sizeof(float));
    float *cpu_state = qw3_xcalloc((size_t)state_n, sizeof(float));
    float *gpu_state = qw3_xmalloc((size_t)state_n * sizeof(float));
    float *cpu_core = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *gpu_core = qw3_xmalloc((size_t)inner_n * sizeof(float));

    bool cpu_ok = true;
    for (int step = 0; cpu_ok && step < 2; step++) {
        int tok = (token + step) % QW3_N_VOCAB;
        cpu_ok = tensor_read_dense_row(&e->model, emb, (uint64_t)tok, x);
        if (!cpu_ok) break;
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        cpu_ok = cpu_matvec_q8_0(&e->model, lw->linear_qkv_proj, xn, qkv) &&
                 cpu_matvec_dense(&e->model, lw->linear_ssm_alpha, xn, alpha) &&
                 cpu_matvec_dense(&e->model, lw->linear_ssm_beta, xn, beta) &&
                 cpu_deltanet_conv1d_step(&e->model, lw, qkv,
                                          cpu_conv_state, conv);
        if (!cpu_ok) break;
        const float *qraw = conv;
        const float *kraw = conv + qk_n;
        const float *vraw = conv + qk_n * 2;
        for (int h = 0; h < QW3_N_LINEAR_QK_HEADS; h++) {
            cpu_l2_norm_head(q + h * QW3_N_LINEAR_HEAD_DIM,
                             qraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
            cpu_l2_norm_head(k + h * QW3_N_LINEAR_HEAD_DIM,
                             kraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            const int hk = hv % QW3_N_LINEAR_QK_HEADS;
            const float *qh = q + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *kh = k + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *vh = vraw + hv * QW3_N_LINEAR_HEAD_DIM;
            float *sh = cpu_state + (uint64_t)hv * QW3_N_LINEAR_HEAD_DIM *
                        QW3_N_LINEAR_HEAD_DIM;
            float *oh = cpu_core + hv * QW3_N_LINEAR_HEAD_DIM;
            const float bh = 1.0f / (1.0f + expf(-beta[hv]));
            const float ah = cpu_softplus(alpha[hv] +
                tensor_read_dense_1d(&e->model, lw->linear_ssm_dt_bias,
                                     (uint64_t)hv));
            const float gh = expf(ah *
                tensor_read_dense_1d(&e->model, lw->linear_ssm_a,
                                     (uint64_t)hv));
            beta_sig[hv] = bh;
            gamma[hv] = gh;
            for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
                float sk = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    sk += sh[i * QW3_N_LINEAR_HEAD_DIM + j] * gh * kh[i];
                }
                const float d = (vh[j] - sk) * bh;
                float out = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    const int idx = i * QW3_N_LINEAR_HEAD_DIM + j;
                    const float sv = sh[idx] * gh + kh[i] * d;
                    sh[idx] = sv;
                    out += sv * qh[i];
                }
                oh[j] = out / sqrtf((float)QW3_N_LINEAR_HEAD_DIM);
            }
        }
    }

    int gpu_ok = 0;
    const char *gpu_stage = "not-started";
    if (cpu_ok && qw3_session_create(&s, e, ctx_size) == 0 && s && s->metal) {
        gpu_ok = 1;
        for (int step = 0; gpu_ok && step < 2; step++) {
            int tok = (token + step) % QW3_N_VOCAB;
            gpu_stage = "embed";
            gpu_ok = qw3_metal_session_embed_q8_0(
                s->metal, emb->offset, (uint32_t)tok, QW3_N_EMBD, NULL);
            if (!gpu_ok) break;
            gpu_stage = "rmsnorm";
            gpu_ok = qw3_metal_session_rmsnorm_weight_f32(
                s->metal, lw->attn_norm->offset,
                QW3_N_EMBD, QW3_RMS_EPS, NULL);
            if (!gpu_ok) break;
            gpu_stage = "qkv";
            gpu_ok = qw3_metal_session_matvec_q8_0_x1_to_scratch(
                s->metal, lw->linear_qkv_proj->offset,
                QW3_N_EMBD, n_qkv, 0, NULL);
            if (!gpu_ok) break;
            gpu_stage = "alpha";
            gpu_ok = qw3_metal_session_matvec_f32_x1_to_scratch(
                s->metal, lw->linear_ssm_alpha->offset,
                QW3_N_EMBD, QW3_N_LINEAR_V_HEADS,
                alpha_offset, alpha);
            if (!gpu_ok) break;
            gpu_stage = "beta";
            gpu_ok = qw3_metal_session_matvec_f32_x1_to_scratch(
                s->metal, lw->linear_ssm_beta->offset,
                QW3_N_EMBD, QW3_N_LINEAR_V_HEADS,
                beta_offset, beta);
            if (!gpu_ok) break;
            gpu_stage = "conv-step";
            gpu_ok = qw3_metal_session_conv1d_step_from_scratch(
                s->metal, lw->linear_conv_weight->offset, 0,
                n_qkv, NULL, step == 1 ? gpu_conv_state : NULL);
            if (!gpu_ok) break;
            gpu_stage = "l2norm";
            gpu_ok = qw3_metal_session_l2norm_qk_from_conv(
                s->metal, QW3_N_LINEAR_QK_HEADS,
                QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS, NULL, NULL);
            if (!gpu_ok) break;
            for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
                beta_sig[hv] = 1.0f / (1.0f + expf(-beta[hv]));
                float ah = cpu_softplus(alpha[hv] +
                    tensor_read_dense_1d(&e->model, lw->linear_ssm_dt_bias,
                                         (uint64_t)hv));
                gamma[hv] = expf(ah *
                    tensor_read_dense_1d(&e->model, lw->linear_ssm_a,
                                         (uint64_t)hv));
            }
            gpu_stage = "recur-step";
            gpu_ok = qw3_metal_session_deltanet_recur_from_buffers(
                s->metal, beta_sig, gamma, 0, QW3_N_LINEAR_QK_HEADS,
                QW3_N_LINEAR_V_HEADS, QW3_N_LINEAR_HEAD_DIM,
                step == 1 ? gpu_state : NULL, step == 1 ? gpu_core : NULL);
        }
    }

    float core_maxdiff = 0.0f, state_maxdiff = 0.0f, conv_maxdiff = 0.0f;
    double core_rmsdiff = 0.0, state_rmsdiff = 0.0, conv_rmsdiff = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < inner_n; i++) {
            float d = fabsf(cpu_core[i] - gpu_core[i]);
            if (d > core_maxdiff) core_maxdiff = d;
            core_rmsdiff += (double)d * (double)d;
        }
        for (uint32_t i = 0; i < state_n; i++) {
            float d = fabsf(cpu_state[i] - gpu_state[i]);
            if (d > state_maxdiff) state_maxdiff = d;
            state_rmsdiff += (double)d * (double)d;
        }
        for (uint32_t i = 0; i < conv_state_n; i++) {
            float d = fabsf(cpu_conv_state[i] - gpu_conv_state[i]);
            if (d > conv_maxdiff) conv_maxdiff = d;
            conv_rmsdiff += (double)d * (double)d;
        }
        core_rmsdiff = sqrt(core_rmsdiff / (double)inner_n);
        state_rmsdiff = sqrt(state_rmsdiff / (double)state_n);
        conv_rmsdiff = sqrt(conv_rmsdiff / (double)conv_state_n);
    }
    fprintf(fp,
            "metal session recur step: %s token=%d,%d stage=%s core_maxdiff=%.7g core_rmsdiff=%.7g state_maxdiff=%.7g state_rmsdiff=%.7g conv_maxdiff=%.7g conv_rmsdiff=%.7g core0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, (token + 1) % QW3_N_VOCAB,
            gpu_stage, core_maxdiff, core_rmsdiff, state_maxdiff, state_rmsdiff,
            conv_maxdiff, conv_rmsdiff,
            gpu_core[0], gpu_core[1], gpu_core[2], gpu_core[3]);

    qw3_session_free(s);
    free(gpu_core); free(cpu_core); free(gpu_state); free(cpu_state);
    free(gpu_conv_state); free(cpu_conv_state); free(gamma); free(beta_sig);
    free(beta); free(alpha); free(k); free(q); free(conv); free(qkv);
    free(xn); free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_session_gated_rmsnorm_test(qw3_engine *e, int token,
                                                int ctx_size, FILE *fp) {
    if (!e || !fp || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token; (void)ctx_size;
    fprintf(fp, "metal session gated RMSNorm: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal session gated RMSNorm: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal session gated RMSNorm: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_layer_weights *lw = &e->weights.layer[0];
    const qw3_tensor *emb = e->weights.token_embd;
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t inner_n = (uint32_t)tensor_linear_inner();
    const uint32_t beta_offset = n_qkv + inner_n + QW3_N_LINEAR_V_HEADS;

    qw3_session *s = NULL;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *z = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *q = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *k = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta_sig = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *core = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *cpu_inner = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *gpu_inner = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *conv_state = qw3_xcalloc((size_t)n_qkv * (QW3_N_LINEAR_CONV_K - 1),
                                    sizeof(float));

    bool cpu_ok = tensor_read_dense_row(&e->model, emb, (uint64_t)token, x);
    if (cpu_ok) {
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        cpu_ok = cpu_matvec_q8_0(&e->model, lw->linear_qkv_proj, xn, qkv) &&
                 cpu_matvec_q8_0(&e->model, lw->linear_gate_proj, xn, z) &&
                 cpu_matvec_dense(&e->model, lw->linear_ssm_beta, xn, beta) &&
                 cpu_deltanet_conv1d_step(&e->model, lw, qkv, conv_state, conv);
    }
    if (cpu_ok) {
        const float *qraw = conv;
        const float *kraw = conv + qk_n;
        for (int h = 0; h < QW3_N_LINEAR_QK_HEADS; h++) {
            cpu_l2_norm_head(q + h * QW3_N_LINEAR_HEAD_DIM,
                             qraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
            cpu_l2_norm_head(k + h * QW3_N_LINEAR_HEAD_DIM,
                             kraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            const int hk = hv % QW3_N_LINEAR_QK_HEADS;
            const float *qh = q + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *kh = k + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *vh = conv + qk_n * 2 + hv * QW3_N_LINEAR_HEAD_DIM;
            float *oh = core + hv * QW3_N_LINEAR_HEAD_DIM;
            const float bh = 1.0f / (1.0f + expf(-beta[hv]));
            beta_sig[hv] = bh;
            for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
                const float d = bh * vh[j];
                float out = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    out += kh[i] * d * qh[i];
                }
                oh[j] = out / sqrtf((float)QW3_N_LINEAR_HEAD_DIM);
            }
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            float *dst = cpu_inner + hv * QW3_N_LINEAR_HEAD_DIM;
            const float *src = core + hv * QW3_N_LINEAR_HEAD_DIM;
            const float *zh = z + hv * QW3_N_LINEAR_HEAD_DIM;
            double ss = 0.0;
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                ss += (double)src[i] * (double)src[i];
            }
            const float scale = 1.0f /
                sqrtf((float)(ss / QW3_N_LINEAR_HEAD_DIM) + QW3_RMS_EPS);
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                dst[i] = src[i] * scale *
                         tensor_read_dense_1d(&e->model, lw->linear_ssm_norm, (uint64_t)i) *
                         cpu_silu(zh[i]);
            }
        }
    }

    int gpu_ok = 0;
    if (cpu_ok && qw3_session_create(&s, e, ctx_size) == 0 && s && s->metal) {
        gpu_ok =
            qw3_metal_session_embed_q8_0(s->metal, emb->offset,
                                         (uint32_t)token, QW3_N_EMBD, NULL) &&
            qw3_metal_session_rmsnorm_weight_f32(s->metal, lw->attn_norm->offset,
                                                 QW3_N_EMBD, QW3_RMS_EPS, NULL) &&
            qw3_metal_session_matvec_q8_0_x1_to_scratch(
                s->metal, lw->linear_qkv_proj->offset,
                QW3_N_EMBD, n_qkv, 0, NULL) &&
            qw3_metal_session_matvec_q8_0_x1_to_scratch(
                s->metal, lw->linear_gate_proj->offset,
                QW3_N_EMBD, inner_n, n_qkv, NULL) &&
            qw3_metal_session_matvec_f32_x1_to_scratch(
                s->metal, lw->linear_ssm_beta->offset,
                QW3_N_EMBD, QW3_N_LINEAR_V_HEADS, beta_offset, NULL) &&
            qw3_metal_session_conv1d_zero_from_scratch(
                s->metal, lw->linear_conv_weight->offset, n_qkv, NULL) &&
            qw3_metal_session_l2norm_qk_from_conv(
                s->metal, QW3_N_LINEAR_QK_HEADS, QW3_N_LINEAR_HEAD_DIM,
                QW3_RMS_EPS, NULL, NULL) &&
            qw3_metal_session_deltanet_recur_zero_from_buffers(
                s->metal, beta_sig, QW3_N_LINEAR_QK_HEADS,
                QW3_N_LINEAR_V_HEADS, QW3_N_LINEAR_HEAD_DIM,
                NULL, NULL) &&
            qw3_metal_session_deltanet_gated_rmsnorm_from_buffers(
                s->metal, lw->linear_ssm_norm->offset, n_qkv,
                QW3_N_LINEAR_V_HEADS, QW3_N_LINEAR_HEAD_DIM,
                QW3_RMS_EPS, gpu_inner);
    }

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < inner_n; i++) {
            float d = fabsf(cpu_inner[i] - gpu_inner[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * (double)d;
        }
        rmsdiff = sqrt(rmsdiff / (double)inner_n);
    }
    fprintf(fp,
            "metal session gated RMSNorm: %s token=%d n=%u maxdiff=%.7g rmsdiff=%.7g first=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, inner_n, maxdiff, rmsdiff,
            gpu_inner[0], gpu_inner[1], gpu_inner[2], gpu_inner[3]);
    qw3_session_free(s);
    free(conv_state); free(gpu_inner); free(cpu_inner); free(core);
    free(beta_sig); free(beta); free(k); free(q); free(conv); free(z);
    free(qkv); free(xn); free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_session_attn_out_test(qw3_engine *e, int token,
                                           int ctx_size, FILE *fp) {
    if (!e || !fp || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token; (void)ctx_size;
    fprintf(fp, "metal session attn out: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal session attn out: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal session attn out: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_layer_weights *lw = &e->weights.layer[0];
    const qw3_tensor *emb = e->weights.token_embd;
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t inner_n = (uint32_t)tensor_linear_inner();
    const uint32_t beta_offset = n_qkv + inner_n + QW3_N_LINEAR_V_HEADS;

    qw3_session *s = NULL;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *z = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *q = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *k = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta_sig = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *core = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *inner = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *cpu_attn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_attn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *conv_state = qw3_xcalloc((size_t)n_qkv * (QW3_N_LINEAR_CONV_K - 1),
                                    sizeof(float));

    bool cpu_ok = tensor_read_dense_row(&e->model, emb, (uint64_t)token, x);
    if (cpu_ok) {
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        cpu_ok = cpu_matvec_q8_0(&e->model, lw->linear_qkv_proj, xn, qkv) &&
                 cpu_matvec_q8_0(&e->model, lw->linear_gate_proj, xn, z) &&
                 cpu_matvec_dense(&e->model, lw->linear_ssm_beta, xn, beta) &&
                 cpu_deltanet_conv1d_step(&e->model, lw, qkv, conv_state, conv);
    }
    if (cpu_ok) {
        const float *qraw = conv;
        const float *kraw = conv + qk_n;
        for (int h = 0; h < QW3_N_LINEAR_QK_HEADS; h++) {
            cpu_l2_norm_head(q + h * QW3_N_LINEAR_HEAD_DIM,
                             qraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
            cpu_l2_norm_head(k + h * QW3_N_LINEAR_HEAD_DIM,
                             kraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            const int hk = hv % QW3_N_LINEAR_QK_HEADS;
            const float *qh = q + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *kh = k + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *vh = conv + qk_n * 2 + hv * QW3_N_LINEAR_HEAD_DIM;
            float *oh = core + hv * QW3_N_LINEAR_HEAD_DIM;
            const float bh = 1.0f / (1.0f + expf(-beta[hv]));
            beta_sig[hv] = bh;
            for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
                const float d = bh * vh[j];
                float out = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    out += kh[i] * d * qh[i];
                }
                oh[j] = out / sqrtf((float)QW3_N_LINEAR_HEAD_DIM);
            }
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            float *dst = inner + hv * QW3_N_LINEAR_HEAD_DIM;
            const float *src = core + hv * QW3_N_LINEAR_HEAD_DIM;
            const float *zh = z + hv * QW3_N_LINEAR_HEAD_DIM;
            double ss = 0.0;
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) ss += (double)src[i] * src[i];
            const float scale = 1.0f /
                sqrtf((float)(ss / QW3_N_LINEAR_HEAD_DIM) + QW3_RMS_EPS);
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                dst[i] = src[i] * scale *
                         tensor_read_dense_1d(&e->model, lw->linear_ssm_norm, (uint64_t)i) *
                         cpu_silu(zh[i]);
            }
        }
        cpu_ok = cpu_matvec_q8_0(&e->model, lw->linear_ssm_out, inner, cpu_attn);
    }

    int gpu_ok = 0;
    if (cpu_ok && qw3_session_create(&s, e, ctx_size) == 0 && s && s->metal) {
        gpu_ok =
            qw3_metal_session_embed_q8_0(s->metal, emb->offset,
                                         (uint32_t)token, QW3_N_EMBD, NULL) &&
            qw3_metal_session_rmsnorm_weight_f32(s->metal, lw->attn_norm->offset,
                                                 QW3_N_EMBD, QW3_RMS_EPS, NULL) &&
            qw3_metal_session_matvec_q8_0_x1_to_scratch(
                s->metal, lw->linear_qkv_proj->offset,
                QW3_N_EMBD, n_qkv, 0, NULL) &&
            qw3_metal_session_matvec_q8_0_x1_to_scratch(
                s->metal, lw->linear_gate_proj->offset,
                QW3_N_EMBD, inner_n, n_qkv, NULL) &&
            qw3_metal_session_matvec_f32_x1_to_scratch(
                s->metal, lw->linear_ssm_beta->offset,
                QW3_N_EMBD, QW3_N_LINEAR_V_HEADS, beta_offset, NULL) &&
            qw3_metal_session_conv1d_zero_from_scratch(
                s->metal, lw->linear_conv_weight->offset, n_qkv, NULL) &&
            qw3_metal_session_l2norm_qk_from_conv(
                s->metal, QW3_N_LINEAR_QK_HEADS, QW3_N_LINEAR_HEAD_DIM,
                QW3_RMS_EPS, NULL, NULL) &&
            qw3_metal_session_deltanet_recur_zero_from_buffers(
                s->metal, beta_sig, QW3_N_LINEAR_QK_HEADS,
                QW3_N_LINEAR_V_HEADS, QW3_N_LINEAR_HEAD_DIM,
                NULL, NULL) &&
            qw3_metal_session_deltanet_gated_rmsnorm_from_buffers(
                s->metal, lw->linear_ssm_norm->offset, n_qkv,
                QW3_N_LINEAR_V_HEADS, QW3_N_LINEAR_HEAD_DIM,
                QW3_RMS_EPS, NULL) &&
            qw3_metal_session_matvec_q8_0_inner_to_x1(
                s->metal, lw->linear_ssm_out->offset,
                inner_n, QW3_N_EMBD, gpu_attn);
    }

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu_attn[i] - gpu_attn[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * (double)d;
        }
        rmsdiff = sqrt(rmsdiff / (double)QW3_N_EMBD);
    }
    fprintf(fp,
            "metal session attn out: %s token=%d maxdiff=%.7g rmsdiff=%.7g first=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, maxdiff, rmsdiff,
            gpu_attn[0], gpu_attn[1], gpu_attn[2], gpu_attn[3]);
    qw3_session_free(s);
    free(conv_state); free(gpu_attn); free(cpu_attn); free(inner); free(core);
    free(beta_sig); free(beta); free(k); free(q); free(conv); free(z);
    free(qkv); free(xn); free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_session_ffn_norm_test(qw3_engine *e, int token,
                                           int ctx_size, FILE *fp) {
    if (!e || !fp || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token; (void)ctx_size;
    fprintf(fp, "metal session ffn norm: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal session ffn norm: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal session ffn norm: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_layer_weights *lw = &e->weights.layer[0];
    const qw3_tensor *emb = e->weights.token_embd;
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t qk_n = QW3_N_LINEAR_QK_HEADS * QW3_N_LINEAR_HEAD_DIM;
    const uint32_t inner_n = (uint32_t)tensor_linear_inner();
    const uint32_t beta_offset = n_qkv + inner_n + QW3_N_LINEAR_V_HEADS;

    qw3_session *s = NULL;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qkv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *z = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *conv = qw3_xmalloc((size_t)n_qkv * sizeof(float));
    float *q = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *k = qw3_xmalloc((size_t)qk_n * sizeof(float));
    float *beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta_sig = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *core = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *inner = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *attn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_ffn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_ffn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *conv_state = qw3_xcalloc((size_t)n_qkv * (QW3_N_LINEAR_CONV_K - 1),
                                    sizeof(float));

    bool cpu_ok = tensor_read_dense_row(&e->model, emb, (uint64_t)token, x);
    if (cpu_ok) {
        cpu_rmsnorm(xn, x, &e->model, lw->attn_norm, QW3_N_EMBD);
        cpu_ok = cpu_matvec_q8_0(&e->model, lw->linear_qkv_proj, xn, qkv) &&
                 cpu_matvec_q8_0(&e->model, lw->linear_gate_proj, xn, z) &&
                 cpu_matvec_dense(&e->model, lw->linear_ssm_beta, xn, beta) &&
                 cpu_deltanet_conv1d_step(&e->model, lw, qkv, conv_state, conv);
    }
    if (cpu_ok) {
        const float *qraw = conv;
        const float *kraw = conv + qk_n;
        for (int h = 0; h < QW3_N_LINEAR_QK_HEADS; h++) {
            cpu_l2_norm_head(q + h * QW3_N_LINEAR_HEAD_DIM,
                             qraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
            cpu_l2_norm_head(k + h * QW3_N_LINEAR_HEAD_DIM,
                             kraw + h * QW3_N_LINEAR_HEAD_DIM,
                             QW3_N_LINEAR_HEAD_DIM);
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            const int hk = hv % QW3_N_LINEAR_QK_HEADS;
            const float *qh = q + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *kh = k + hk * QW3_N_LINEAR_HEAD_DIM;
            const float *vh = conv + qk_n * 2 + hv * QW3_N_LINEAR_HEAD_DIM;
            float *oh = core + hv * QW3_N_LINEAR_HEAD_DIM;
            const float bh = 1.0f / (1.0f + expf(-beta[hv]));
            beta_sig[hv] = bh;
            for (int j = 0; j < QW3_N_LINEAR_HEAD_DIM; j++) {
                const float d = bh * vh[j];
                float out = 0.0f;
                for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                    out += kh[i] * d * qh[i];
                }
                oh[j] = out / sqrtf((float)QW3_N_LINEAR_HEAD_DIM);
            }
        }
        for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
            float *dst = inner + hv * QW3_N_LINEAR_HEAD_DIM;
            const float *src = core + hv * QW3_N_LINEAR_HEAD_DIM;
            const float *zh = z + hv * QW3_N_LINEAR_HEAD_DIM;
            double ss = 0.0;
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) ss += (double)src[i] * src[i];
            const float scale = 1.0f /
                sqrtf((float)(ss / QW3_N_LINEAR_HEAD_DIM) + QW3_RMS_EPS);
            for (int i = 0; i < QW3_N_LINEAR_HEAD_DIM; i++) {
                dst[i] = src[i] * scale *
                         tensor_read_dense_1d(&e->model, lw->linear_ssm_norm, (uint64_t)i) *
                         cpu_silu(zh[i]);
            }
        }
        cpu_ok = cpu_matvec_q8_0(&e->model, lw->linear_ssm_out, inner, attn);
        if (cpu_ok) {
            float *resid = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
            for (int i = 0; i < QW3_N_EMBD; i++) resid[i] = x[i] + attn[i];
            cpu_rmsnorm(cpu_ffn, resid, &e->model, lw->ffn_norm, QW3_N_EMBD);
            free(resid);
        }
    }

    int gpu_ok = 0;
    if (cpu_ok && qw3_session_create(&s, e, ctx_size) == 0 && s && s->metal) {
        gpu_ok =
            qw3_metal_session_embed_q8_0(s->metal, emb->offset,
                                         (uint32_t)token, QW3_N_EMBD, NULL) &&
            qw3_metal_session_rmsnorm_weight_f32(s->metal, lw->attn_norm->offset,
                                                 QW3_N_EMBD, QW3_RMS_EPS, NULL) &&
            qw3_metal_session_matvec_q8_0_x1_to_scratch(
                s->metal, lw->linear_qkv_proj->offset,
                QW3_N_EMBD, n_qkv, 0, NULL) &&
            qw3_metal_session_matvec_q8_0_x1_to_scratch(
                s->metal, lw->linear_gate_proj->offset,
                QW3_N_EMBD, inner_n, n_qkv, NULL) &&
            qw3_metal_session_matvec_f32_x1_to_scratch(
                s->metal, lw->linear_ssm_beta->offset,
                QW3_N_EMBD, QW3_N_LINEAR_V_HEADS, beta_offset, NULL) &&
            qw3_metal_session_conv1d_zero_from_scratch(
                s->metal, lw->linear_conv_weight->offset, n_qkv, NULL) &&
            qw3_metal_session_l2norm_qk_from_conv(
                s->metal, QW3_N_LINEAR_QK_HEADS, QW3_N_LINEAR_HEAD_DIM,
                QW3_RMS_EPS, NULL, NULL) &&
            qw3_metal_session_deltanet_recur_zero_from_buffers(
                s->metal, beta_sig, QW3_N_LINEAR_QK_HEADS,
                QW3_N_LINEAR_V_HEADS, QW3_N_LINEAR_HEAD_DIM,
                NULL, NULL) &&
            qw3_metal_session_deltanet_gated_rmsnorm_from_buffers(
                s->metal, lw->linear_ssm_norm->offset, n_qkv,
                QW3_N_LINEAR_V_HEADS, QW3_N_LINEAR_HEAD_DIM,
                QW3_RMS_EPS, NULL) &&
            qw3_metal_session_matvec_q8_0_inner_to_x1(
                s->metal, lw->linear_ssm_out->offset,
                inner_n, QW3_N_EMBD, NULL) &&
            qw3_metal_session_residual_rmsnorm_x0_x1(
                s->metal, lw->ffn_norm->offset, QW3_N_EMBD,
                QW3_RMS_EPS, gpu_ffn);
    }

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu_ffn[i] - gpu_ffn[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * (double)d;
        }
        rmsdiff = sqrt(rmsdiff / (double)QW3_N_EMBD);
    }
    fprintf(fp,
            "metal session ffn norm: %s token=%d maxdiff=%.7g rmsdiff=%.7g first=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, maxdiff, rmsdiff,
            gpu_ffn[0], gpu_ffn[1], gpu_ffn[2], gpu_ffn[3]);
    qw3_session_free(s);
    free(conv_state); free(gpu_ffn); free(cpu_ffn); free(attn); free(inner);
    free(core); free(beta_sig); free(beta); free(k); free(q); free(conv);
    free(z); free(qkv); free(xn); free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_session_layer0_test(qw3_engine *e, int token,
                                         int ctx_size, FILE *fp) {
    if (!e || !fp || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token; (void)ctx_size;
    fprintf(fp, "metal session layer0: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal session layer0: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal session layer0: token %d is outside vocab\n", token);
        return -1;
    }
    const qw3_layer_weights *lw = &e->weights.layer[0];
    const qw3_tensor *emb = e->weights.token_embd;
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t inner_n = (uint32_t)tensor_linear_inner();
    const uint32_t beta_offset = n_qkv + inner_n + QW3_N_LINEAR_V_HEADS;
    const uint32_t conv_state_n = n_qkv * (QW3_N_LINEAR_CONV_K - 1);
    const uint32_t state_n = QW3_N_LINEAR_V_HEADS *
                             QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM;

    qw3_session *s = NULL;
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_layer = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_conv_state = qw3_xcalloc((size_t)conv_state_n, sizeof(float));
    float *cpu_state = qw3_xcalloc((size_t)state_n, sizeof(float));

    float *ffn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *attn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *router = qw3_xmalloc((size_t)QW3_N_EXPERT * sizeof(float));
    float *egate = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *up = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *sparse = qw3_xcalloc((size_t)QW3_N_EMBD, sizeof(float));
    float *sh_gate = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *sh_up = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *sh_hidden = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *shared_down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *shared = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_layer = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *beta_sig = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));
    float *beta = qw3_xmalloc((size_t)QW3_N_LINEAR_V_HEADS * sizeof(float));

    bool cpu_ok = tensor_read_dense_row(&e->model, emb, (uint64_t)token, x) &&
                  cpu_deltanet_layer(e, 0, x, cpu_conv_state, cpu_state,
                                     cpu_layer);

    int gpu_ok = 0;
    if (cpu_ok && qw3_session_create(&s, e, ctx_size) == 0 && s && s->metal) {
        gpu_ok =
            qw3_metal_session_embed_q8_0(s->metal, emb->offset,
                                         (uint32_t)token, QW3_N_EMBD, NULL) &&
            qw3_metal_session_rmsnorm_weight_f32(s->metal, lw->attn_norm->offset,
                                                 QW3_N_EMBD, QW3_RMS_EPS, NULL) &&
            qw3_metal_session_matvec_q8_0_x1_to_scratch(
                s->metal, lw->linear_qkv_proj->offset,
                QW3_N_EMBD, n_qkv, 0, NULL) &&
            qw3_metal_session_matvec_q8_0_x1_to_scratch(
                s->metal, lw->linear_gate_proj->offset,
                QW3_N_EMBD, inner_n, n_qkv, NULL) &&
            qw3_metal_session_matvec_f32_x1_to_scratch(
                s->metal, lw->linear_ssm_beta->offset,
                QW3_N_EMBD, QW3_N_LINEAR_V_HEADS, beta_offset, beta) &&
            qw3_metal_session_conv1d_zero_from_scratch(
                s->metal, lw->linear_conv_weight->offset, n_qkv, NULL) &&
            qw3_metal_session_l2norm_qk_from_conv(
                s->metal, QW3_N_LINEAR_QK_HEADS, QW3_N_LINEAR_HEAD_DIM,
                QW3_RMS_EPS, NULL, NULL);
        if (gpu_ok) {
            for (int hv = 0; hv < QW3_N_LINEAR_V_HEADS; hv++) {
                beta_sig[hv] = 1.0f / (1.0f + expf(-beta[hv]));
            }
            gpu_ok =
                qw3_metal_session_deltanet_recur_zero_from_buffers(
                    s->metal, beta_sig, QW3_N_LINEAR_QK_HEADS,
                    QW3_N_LINEAR_V_HEADS, QW3_N_LINEAR_HEAD_DIM,
                    NULL, NULL) &&
                qw3_metal_session_deltanet_gated_rmsnorm_from_buffers(
                    s->metal, lw->linear_ssm_norm->offset, n_qkv,
                    QW3_N_LINEAR_V_HEADS, QW3_N_LINEAR_HEAD_DIM,
                    QW3_RMS_EPS, NULL) &&
                qw3_metal_session_matvec_q8_0_inner_to_x1(
                    s->metal, lw->linear_ssm_out->offset,
                    inner_n, QW3_N_EMBD, attn) &&
                qw3_metal_session_residual_rmsnorm_x0_x1(
                    s->metal, lw->ffn_norm->offset, QW3_N_EMBD,
                    QW3_RMS_EPS, ffn) &&
                qw3_metal_matvec_f32(lw->ffn_gate_inp->offset, ffn,
                                     QW3_N_EMBD, QW3_N_EXPERT, router);
        }
    }

    int ids[QW3_N_EXPERT_USED];
    float vals[QW3_N_EXPERT_USED], weights[QW3_N_EXPERT_USED];
    memset(ids, 0, sizeof(ids));
    if (gpu_ok) {
        topk_desc(router, QW3_N_EXPERT, QW3_N_EXPERT_USED, ids, vals);
        float wsum = 0.0f;
        for (int kk = 0; kk < QW3_N_EXPERT_USED; kk++) {
            weights[kk] = expf(vals[kk] - vals[0]);
            wsum += weights[kk];
        }
        for (int kk = 0; kk < QW3_N_EXPERT_USED; kk++) weights[kk] /= wsum;
    }
    for (int kk = 0; gpu_ok && kk < QW3_N_EXPERT_USED; kk++) {
        gpu_ok =
            qw3_metal_matvec_iq3_s_expert(
                lw->ffn_gate_exps->offset, (uint32_t)ids[kk],
                ffn, QW3_N_EMBD, QW3_N_FF_EXP, egate) &&
            qw3_metal_matvec_iq3_s_expert(
                lw->ffn_up_exps->offset, (uint32_t)ids[kk],
                ffn, QW3_N_EMBD, QW3_N_FF_EXP, up) &&
            qw3_metal_silu_mul(egate, up, QW3_N_FF_EXP, hidden);
        if (gpu_ok && lw->ffn_down_exps->type == QW3_TENSOR_IQ4_XS) {
            gpu_ok = qw3_metal_matvec_iq4_xs_expert(
                lw->ffn_down_exps->offset, (uint32_t)ids[kk],
                hidden, QW3_N_FF_EXP, QW3_N_EMBD, down);
        } else if (gpu_ok && lw->ffn_down_exps->type == QW3_TENSOR_Q6_K) {
            gpu_ok = qw3_metal_matvec_q6_k_expert(
                lw->ffn_down_exps->offset, (uint32_t)ids[kk],
                hidden, QW3_N_FF_EXP, QW3_N_EMBD, down);
        } else if (gpu_ok) {
            gpu_ok = 0;
        }
        if (gpu_ok) {
            for (int i = 0; i < QW3_N_EMBD; i++) {
                sparse[i] += weights[kk] * down[i];
            }
        }
    }
    gpu_ok = gpu_ok &&
        qw3_metal_matvec_q8_0(lw->ffn_gate_shared->offset, ffn,
                              QW3_N_EMBD, QW3_N_FF_SHARED, sh_gate) &&
        qw3_metal_matvec_q8_0(lw->ffn_up_shared->offset, ffn,
                              QW3_N_EMBD, QW3_N_FF_SHARED, sh_up) &&
        qw3_metal_silu_mul(sh_gate, sh_up, QW3_N_FF_SHARED, sh_hidden) &&
        qw3_metal_matvec_q8_0(lw->ffn_down_shared->offset, sh_hidden,
                              QW3_N_FF_SHARED, QW3_N_EMBD, shared_down);
    float shared_raw = 0.0f;
    if (gpu_ok) {
        gpu_ok = qw3_metal_matvec_f32(lw->ffn_gate_inp_shexp->offset, ffn,
                                      QW3_N_EMBD, 1, &shared_raw);
    }
    if (gpu_ok) {
        gpu_ok = qw3_metal_scale(shared_down, QW3_N_EMBD,
                                 1.0f / (1.0f + expf(-shared_raw)), shared);
    }
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            gpu_layer[i] = x[i] + attn[i] + sparse[i] + shared[i];
        }
    }

    float maxdiff = 0.0f;
    double rmsdiff = 0.0, out_rms = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu_layer[i] - gpu_layer[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * (double)d;
            out_rms += (double)gpu_layer[i] * (double)gpu_layer[i];
        }
        rmsdiff = sqrt(rmsdiff / (double)QW3_N_EMBD);
        out_rms = sqrt(out_rms / (double)QW3_N_EMBD);
    }
    fprintf(fp,
            "metal session layer0: %s token=%d top0=%d maxdiff=%.7g rmsdiff=%.7g out_rms=%.7g out0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, ids[0], maxdiff, rmsdiff,
            out_rms, gpu_layer[0], gpu_layer[1], gpu_layer[2], gpu_layer[3]);

    qw3_session_free(s);
    free(beta); free(beta_sig); free(gpu_layer); free(shared); free(shared_down);
    free(sh_hidden); free(sh_up); free(sh_gate); free(sparse); free(down);
    free(hidden); free(up); free(egate); free(router); free(attn); free(ffn);
    free(cpu_state); free(cpu_conv_state); free(cpu_layer); free(x);
    return gpu_ok ? 0 : -1;
#endif
}

static int __attribute__((unused))
qw3_metal_eval_sparse_moe_from_ffn(qw3_engine *e,
                                   const qw3_layer_weights *lw,
                                   const float *ffn,
                                   const float *router_in,
                                   float *moe,
                                   int *top0_out) {
#ifdef QW3_NO_METAL
    (void)e; (void)lw; (void)ffn; (void)router_in; (void)moe; (void)top0_out;
    return 0;
#else
    if (!e || !lw || !ffn || !moe) return 0;
    float *router = router_in ? NULL :
        qw3_xmalloc((size_t)QW3_N_EXPERT * sizeof(float));
    float *egate = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *up = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *sparse = qw3_xcalloc((size_t)QW3_N_EMBD, sizeof(float));

    int ok = 1;
    if (!router_in) {
        ok = qw3_metal_matvec_f32(lw->ffn_gate_inp->offset, ffn,
                                  QW3_N_EMBD, QW3_N_EXPERT, router);
    }
    const float *router_scores = router_in ? router_in : router;
    int ids[QW3_N_EXPERT_USED];
    float vals[QW3_N_EXPERT_USED], weights[QW3_N_EXPERT_USED];
    memset(ids, 0, sizeof(ids));
    if (ok) {
        topk_desc(router_scores, QW3_N_EXPERT, QW3_N_EXPERT_USED, ids, vals);
        if (top0_out) *top0_out = ids[0];
        float wsum = 0.0f;
        for (int kk = 0; kk < QW3_N_EXPERT_USED; kk++) {
            weights[kk] = expf(vals[kk] - vals[0]);
            wsum += weights[kk];
        }
        for (int kk = 0; kk < QW3_N_EXPERT_USED; kk++) weights[kk] /= wsum;
    }
    for (int kk = 0; ok && kk < QW3_N_EXPERT_USED; kk++) {
        ok =
            qw3_metal_matvec_iq3_s_expert(
                lw->ffn_gate_exps->offset, (uint32_t)ids[kk],
                ffn, QW3_N_EMBD, QW3_N_FF_EXP, egate) &&
            qw3_metal_matvec_iq3_s_expert(
                lw->ffn_up_exps->offset, (uint32_t)ids[kk],
                ffn, QW3_N_EMBD, QW3_N_FF_EXP, up) &&
            qw3_metal_silu_mul(egate, up, QW3_N_FF_EXP, hidden);
        if (ok && lw->ffn_down_exps->type == QW3_TENSOR_IQ4_XS) {
            ok = qw3_metal_matvec_iq4_xs_expert(
                lw->ffn_down_exps->offset, (uint32_t)ids[kk],
                hidden, QW3_N_FF_EXP, QW3_N_EMBD, down);
        } else if (ok && lw->ffn_down_exps->type == QW3_TENSOR_Q6_K) {
            ok = qw3_metal_matvec_q6_k_expert(
                lw->ffn_down_exps->offset, (uint32_t)ids[kk],
                hidden, QW3_N_FF_EXP, QW3_N_EMBD, down);
        } else if (ok) {
            ok = 0;
        }
        if (ok) {
            for (int i = 0; i < QW3_N_EMBD; i++) {
                sparse[i] += weights[kk] * down[i];
            }
        }
    }
    if (ok) memcpy(moe, sparse, (size_t)QW3_N_EMBD * sizeof(float));

    free(sparse); free(down); free(hidden); free(up); free(egate); free(router);
    return ok;
#endif
}

static int __attribute__((unused))
qw3_metal_session_sparse_moe_from_router(qw3_session *s,
                                         const qw3_layer_weights *lw,
                                         const float *router_scores,
                                         int *top0_out) {
#ifdef QW3_NO_METAL
    (void)s; (void)lw; (void)router_scores; (void)top0_out;
    return 0;
#else
    if (!s || !s->metal || !lw || !router_scores) return 0;

    int ids[QW3_N_EXPERT_USED];
    float vals[QW3_N_EXPERT_USED], weights[QW3_N_EXPERT_USED];
    memset(ids, 0, sizeof(ids));
    topk_desc(router_scores, QW3_N_EXPERT, QW3_N_EXPERT_USED, ids, vals);
    if (top0_out) *top0_out = ids[0];

    float wsum = 0.0f;
    for (int kk = 0; kk < QW3_N_EXPERT_USED; kk++) {
        weights[kk] = expf(vals[kk] - vals[0]);
        wsum += weights[kk];
    }
    for (int kk = 0; kk < QW3_N_EXPERT_USED; kk++) weights[kk] /= wsum;

    return qw3_metal_session_sparse_moe_topk(
        s->metal, lw->ffn_gate_exps->offset, lw->ffn_up_exps->offset,
        lw->ffn_down_exps->offset, (uint32_t)lw->ffn_down_exps->type,
        ids, weights, QW3_N_EXPERT_USED, QW3_N_EMBD, QW3_N_FF_EXP);
#endif
}

int qw3_engine_metal_session_gqa_project_test(qw3_engine *e, int token,
                                              int ctx_size, FILE *fp) {
    if (!e || !fp || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token; (void)ctx_size;
    fprintf(fp, "metal session gqa project: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal session gqa project: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal session gqa project: token %d is outside vocab\n", token);
        return -1;
    }
    const int il = 3;
    const int pos = 1;
    const qw3_layer_weights *lw = &e->weights.layer[il];
    const uint32_t qg_n = (uint32_t)tensor_cols_qg();
    const uint32_t q_n = QW3_N_HEAD * QW3_N_HEAD_DIM;
    const uint32_t kv_n = (uint32_t)tensor_cols_kv();

    qw3_session *s = NULL;
    float *x_cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *q_cpu = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *k_cpu = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *v_cpu = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *gate_cpu = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *q_gpu = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *k_gpu = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *v_gpu = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *gate_gpu = qw3_xmalloc((size_t)q_n * sizeof(float));

    bool cpu_ok = tensor_read_dense_row(&e->model, e->weights.token_embd,
                                        (uint64_t)token, x_cpu) &&
                  cpu_gqa_project_token(e, il, pos, x_cpu,
                                        q_cpu, k_cpu, v_cpu, gate_cpu);
    int gpu_ok = 0;
    if (cpu_ok && qw3_session_create(&s, e, ctx_size) == 0 && s && s->metal) {
        gpu_ok =
            qw3_metal_session_embed_q8_0(s->metal, e->weights.token_embd->offset,
                                         (uint32_t)token, QW3_N_EMBD, NULL) &&
            qw3_metal_session_rmsnorm_weight_f32(s->metal, lw->attn_norm->offset,
                                                 QW3_N_EMBD, QW3_RMS_EPS, NULL) &&
            qw3_metal_session_gqa_project_cache(
                s->metal, lw->attn_q_proj->offset, lw->attn_k_proj->offset,
                lw->attn_v_proj->offset, lw->attn_q_norm->offset,
                lw->attn_k_norm->offset, qg_n, q_n, kv_n,
                QW3_N_HEAD, QW3_N_HEAD_KV, QW3_N_HEAD_DIM,
                QW3_ROPE_DIM, 0, (uint32_t)pos, QW3_ROPE_THETA,
                QW3_RMS_EPS, q_gpu, k_gpu, v_gpu, gate_gpu);
    }

    float q_max = 0.0f, k_max = 0.0f, v_max = 0.0f, gate_max = 0.0f;
    double q_rms = 0.0, k_rms = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < q_n; i++) {
            float dq = fabsf(q_cpu[i] - q_gpu[i]);
            float dg = fabsf(gate_cpu[i] - gate_gpu[i]);
            if (dq > q_max) q_max = dq;
            if (dg > gate_max) gate_max = dg;
            q_rms += (double)q_gpu[i] * q_gpu[i];
        }
        for (uint32_t i = 0; i < kv_n; i++) {
            float dk = fabsf(k_cpu[i] - k_gpu[i]);
            float dv = fabsf(v_cpu[i] - v_gpu[i]);
            if (dk > k_max) k_max = dk;
            if (dv > v_max) v_max = dv;
            k_rms += (double)k_gpu[i] * k_gpu[i];
        }
        q_rms = sqrt(q_rms / q_n);
        k_rms = sqrt(k_rms / kv_n);
    }
    fprintf(fp,
            "metal session gqa project: %s token=%d layer=3 pos=%d q_max=%.7g k_max=%.7g v_max=%.7g gate_max=%.7g q_rms=%.7g k_rms=%.7g q0=[%.7g %.7g %.7g %.7g] k0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, pos,
            q_max, k_max, v_max, gate_max, q_rms, k_rms,
            q_gpu[0], q_gpu[1], q_gpu[2], q_gpu[3],
            k_gpu[0], k_gpu[1], k_gpu[2], k_gpu[3]);
    qw3_session_free(s);
    free(gate_gpu); free(v_gpu); free(k_gpu); free(q_gpu);
    free(gate_cpu); free(v_cpu); free(k_cpu); free(q_cpu); free(x_cpu);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_session_gqa_single_test(qw3_engine *e, int token,
                                             int ctx_size, FILE *fp) {
    if (!e || !fp || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token; (void)ctx_size;
    fprintf(fp, "metal session gqa single: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal session gqa single: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal session gqa single: token %d is outside vocab\n", token);
        return -1;
    }
    const int il = 3;
    const int pos = 1;
    const qw3_layer_weights *lw = &e->weights.layer[il];
    const uint32_t qg_n = (uint32_t)tensor_cols_qg();
    const uint32_t q_n = QW3_N_HEAD * QW3_N_HEAD_DIM;
    const uint32_t kv_n = (uint32_t)tensor_cols_kv();

    qw3_session *s = NULL;
    float *x_cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    bool cpu_ok = tensor_read_dense_row(&e->model, e->weights.token_embd,
                                        (uint64_t)token, x_cpu) &&
                  cpu_gqa_single_token_layer(e, il, pos, x_cpu, cpu);
    int gpu_ok = 0;
    if (cpu_ok && qw3_session_create(&s, e, ctx_size) == 0 && s && s->metal) {
        gpu_ok =
            qw3_metal_session_embed_q8_0(s->metal, e->weights.token_embd->offset,
                                         (uint32_t)token, QW3_N_EMBD, NULL) &&
            qw3_metal_session_rmsnorm_weight_f32(s->metal, lw->attn_norm->offset,
                                                 QW3_N_EMBD, QW3_RMS_EPS, NULL) &&
            qw3_metal_session_gqa_project_cache(
                s->metal, lw->attn_q_proj->offset, lw->attn_k_proj->offset,
                lw->attn_v_proj->offset, lw->attn_q_norm->offset,
                lw->attn_k_norm->offset, qg_n, q_n, kv_n,
                QW3_N_HEAD, QW3_N_HEAD_KV, QW3_N_HEAD_DIM,
                QW3_ROPE_DIM, 0, (uint32_t)pos, QW3_ROPE_THETA,
                QW3_RMS_EPS, NULL, NULL, NULL, NULL) &&
            qw3_metal_session_gqa_single_attn_out(
                s->metal, lw->attn_o_proj->offset,
                QW3_N_HEAD, QW3_N_HEAD_KV, QW3_N_HEAD_DIM,
                QW3_N_EMBD, gpu);
    }

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu[i] - gpu[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * (double)d;
        }
        rmsdiff = sqrt(rmsdiff / (double)QW3_N_EMBD);
    }
    fprintf(fp,
            "metal session gqa single: %s token=%d layer=3 pos=%d maxdiff=%.7g rmsdiff=%.7g first=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, pos, maxdiff, rmsdiff,
            gpu[0], gpu[1], gpu[2], gpu[3]);
    qw3_session_free(s);
    free(gpu);
    free(cpu);
    free(x_cpu);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_session_gqa_cached2_test(qw3_engine *e, int token,
                                              int ctx_size, FILE *fp) {
    if (!e || !fp || ctx_size <= 1) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token; (void)ctx_size;
    fprintf(fp, "metal session gqa cached2: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal session gqa cached2: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal session gqa cached2: token %d is outside vocab\n", token);
        return -1;
    }
    const int il = 3;
    const qw3_layer_weights *lw = &e->weights.layer[il];
    const uint32_t qg_n = (uint32_t)tensor_cols_qg();
    const uint32_t q_n = QW3_N_HEAD * QW3_N_HEAD_DIM;
    const uint32_t kv_n = (uint32_t)tensor_cols_kv();

    qw3_session *s = NULL;
    float *x0 = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *x1 = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *q0 = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *k0 = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *v0 = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *g0 = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *q1 = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *k1 = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *v1 = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *g1 = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *kcache = qw3_xmalloc((size_t)(2 * kv_n) * sizeof(float));
    float *vcache = qw3_xmalloc((size_t)(2 * kv_n) * sizeof(float));
    float *inner = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));

    bool cpu_ok = tensor_read_dense_row(&e->model, e->weights.token_embd,
                                        (uint64_t)token, x0);
    if (cpu_ok) {
        memcpy(x1, x0, (size_t)QW3_N_EMBD * sizeof(float));
        for (int i = 0; i < QW3_N_EMBD; i++) {
            x1[i] += 0.001f * sinf((float)(i % 17));
        }
        cpu_ok = cpu_gqa_project_token(e, il, 0, x0, q0, k0, v0, g0) &&
                 cpu_gqa_project_token(e, il, 1, x1, q1, k1, v1, g1);
    }
    if (cpu_ok) {
        memcpy(kcache, k0, (size_t)kv_n * sizeof(float));
        memcpy(kcache + kv_n, k1, (size_t)kv_n * sizeof(float));
        memcpy(vcache, v0, (size_t)kv_n * sizeof(float));
        memcpy(vcache + kv_n, v1, (size_t)kv_n * sizeof(float));
        cpu_gqa_attend_inner(q1, g1, kcache, vcache, 2, inner);
        cpu_ok = cpu_matvec_q8_0(&e->model, lw->attn_o_proj, inner, cpu);
    }

    int gpu_ok = 0;
    if (cpu_ok && qw3_session_create(&s, e, ctx_size) == 0 && s && s->metal) {
        gpu_ok =
            qw3_metal_session_embed_q8_0(s->metal, e->weights.token_embd->offset,
                                         (uint32_t)token, QW3_N_EMBD, NULL) &&
            qw3_metal_session_rmsnorm_weight_f32(s->metal, lw->attn_norm->offset,
                                                 QW3_N_EMBD, QW3_RMS_EPS, NULL) &&
            qw3_metal_session_gqa_project_cache(
                s->metal, lw->attn_q_proj->offset, lw->attn_k_proj->offset,
                lw->attn_v_proj->offset, lw->attn_q_norm->offset,
                lw->attn_k_norm->offset, qg_n, q_n, kv_n,
                QW3_N_HEAD, QW3_N_HEAD_KV, QW3_N_HEAD_DIM,
                QW3_ROPE_DIM, 0, 0, QW3_ROPE_THETA, QW3_RMS_EPS,
                NULL, NULL, NULL, NULL);
        if (gpu_ok) {
            /*
             * Reuse x0 as an exact CPU-side synthetic second token by writing it
             * through the embedding/rms path is impossible; use token+1 when the
             * diagnostic needs an independent cached position on-device.
             */
            const int token2 = (token + 1 < QW3_N_VOCAB) ? token + 1 : token;
            float *x2 = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
            float *q2 = qw3_xmalloc((size_t)q_n * sizeof(float));
            float *k2 = qw3_xmalloc((size_t)kv_n * sizeof(float));
            float *v2 = qw3_xmalloc((size_t)kv_n * sizeof(float));
            float *g2 = qw3_xmalloc((size_t)q_n * sizeof(float));
            float *inner2 = qw3_xmalloc((size_t)q_n * sizeof(float));
            cpu_ok = tensor_read_dense_row(&e->model, e->weights.token_embd,
                                           (uint64_t)token2, x2) &&
                     cpu_gqa_project_token(e, il, 1, x2, q2, k2, v2, g2);
            if (cpu_ok) {
                memcpy(kcache + kv_n, k2, (size_t)kv_n * sizeof(float));
                memcpy(vcache + kv_n, v2, (size_t)kv_n * sizeof(float));
                cpu_gqa_attend_inner(q2, g2, kcache, vcache, 2, inner2);
                cpu_ok = cpu_matvec_q8_0(&e->model, lw->attn_o_proj, inner2, cpu);
            }
            gpu_ok = cpu_ok &&
                qw3_metal_session_embed_q8_0(s->metal, e->weights.token_embd->offset,
                                             (uint32_t)token2, QW3_N_EMBD, NULL) &&
                qw3_metal_session_rmsnorm_weight_f32(s->metal, lw->attn_norm->offset,
                                                     QW3_N_EMBD, QW3_RMS_EPS, NULL) &&
                qw3_metal_session_gqa_project_cache(
                    s->metal, lw->attn_q_proj->offset, lw->attn_k_proj->offset,
                    lw->attn_v_proj->offset, lw->attn_q_norm->offset,
                    lw->attn_k_norm->offset, qg_n, q_n, kv_n,
                    QW3_N_HEAD, QW3_N_HEAD_KV, QW3_N_HEAD_DIM,
                    QW3_ROPE_DIM, 0, 1, QW3_ROPE_THETA, QW3_RMS_EPS,
                    NULL, NULL, NULL, NULL) &&
                qw3_metal_session_gqa_cached_attn_out(
                    s->metal, lw->attn_o_proj->offset, 2, 0,
                    QW3_N_HEAD, QW3_N_HEAD_KV, QW3_N_HEAD_DIM,
                    QW3_N_EMBD, gpu);
            free(inner2); free(g2); free(v2); free(k2); free(q2); free(x2);
        }
    }

    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu[i] - gpu[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * d;
        }
        rmsdiff = sqrt(rmsdiff / (double)QW3_N_EMBD);
    }
    fprintf(fp,
            "metal session gqa cached2: %s token=%d layer=3 n_ctx=2 maxdiff=%.7g rmsdiff=%.7g first=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, maxdiff, rmsdiff,
            gpu[0], gpu[1], gpu[2], gpu[3]);
    qw3_session_free(s);
    free(gpu); free(cpu); free(inner); free(vcache); free(kcache);
    free(g1); free(v1); free(k1); free(q1); free(g0); free(v0); free(k0);
    free(q0); free(x1); free(x0);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_session_gqa_cached_bench(qw3_engine *e, int token,
                                              int n_ctx, int ctx_size,
                                              FILE *fp) {
    if (!e || !fp || n_ctx < 2 || ctx_size < n_ctx) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token; (void)n_ctx; (void)ctx_size;
    fprintf(fp, "metal session gqa cached bench: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal session gqa cached bench: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal session gqa cached bench: token %d is outside vocab\n", token);
        return -1;
    }

    int il = 3;
    const char *layer_env = getenv("QW3_METAL_GQA_BENCH_LAYER");
    if (layer_env && layer_env[0]) {
        char *end = NULL;
        long v = strtol(layer_env, &end, 10);
        if (end != layer_env && v >= 0 && v < QW3_N_LAYER) il = (int)v;
    }
    if (!qw3_layer_is_full_attention((uint32_t)il)) {
        fprintf(fp,
                "metal session gqa cached bench: layer %d is not full-attention\n",
                il);
        return -1;
    }
    uint32_t full_slot = 0;
    for (int j = 0; j < il; j++) {
        if (qw3_layer_is_full_attention((uint32_t)j)) full_slot++;
    }
    const int iters = 64;
    const qw3_layer_weights *lw = &e->weights.layer[il];
    const uint32_t qg_n = (uint32_t)tensor_cols_qg();
    const uint32_t q_n = QW3_N_HEAD * QW3_N_HEAD_DIM;
    const uint32_t kv_n = (uint32_t)tensor_cols_kv();
    qw3_session *s = NULL;
    int ok = qw3_session_create(&s, e, ctx_size) == 0 && s && s->metal;
    float *q_last = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *gate_last = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *k_cache = qw3_xmalloc((size_t)n_ctx * kv_n * sizeof(float));
    float *v_cache = qw3_xmalloc((size_t)n_ctx * kv_n * sizeof(float));
    float *cpu_inner = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *cpu_out = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_out = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    memset(q_last, 0, (size_t)q_n * sizeof(float));
    memset(gate_last, 0, (size_t)q_n * sizeof(float));
    memset(k_cache, 0, (size_t)n_ctx * kv_n * sizeof(float));
    memset(v_cache, 0, (size_t)n_ctx * kv_n * sizeof(float));
    memset(cpu_inner, 0, (size_t)q_n * sizeof(float));
    memset(cpu_out, 0, (size_t)QW3_N_EMBD * sizeof(float));
    memset(gpu_out, 0, (size_t)QW3_N_EMBD * sizeof(float));

    double fill_ms = 0.0;
    if (ok) {
        ok = qw3_metal_session_embed_q8_0(
                 s->metal, e->weights.token_embd->offset,
                 (uint32_t)token, QW3_N_EMBD, NULL) &&
             qw3_metal_session_rmsnorm_weight_f32(
                 s->metal, lw->attn_norm->offset,
                 QW3_N_EMBD, QW3_RMS_EPS, NULL);
    }
    if (ok) {
        const double t0 = qw3_now_sec();
        for (int pos = 0; ok && pos < n_ctx; pos++) {
            float *q_out = (pos + 1 == n_ctx) ? q_last : NULL;
            float *gate_out = (pos + 1 == n_ctx) ? gate_last : NULL;
            ok = qw3_metal_session_gqa_project_cache(
                s->metal, lw->attn_q_proj->offset, lw->attn_k_proj->offset,
                lw->attn_v_proj->offset, lw->attn_q_norm->offset,
                lw->attn_k_norm->offset, qg_n, q_n, kv_n,
                QW3_N_HEAD, QW3_N_HEAD_KV, QW3_N_HEAD_DIM,
                QW3_ROPE_DIM, full_slot, (uint32_t)pos, QW3_ROPE_THETA,
                QW3_RMS_EPS, q_out,
                k_cache + (size_t)pos * kv_n,
                v_cache + (size_t)pos * kv_n,
                gate_out);
        }
        if (ok) ok = qw3_metal_synchronize();
        fill_ms = (qw3_now_sec() - t0) * 1000.0;
    }
    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    if (ok) {
        cpu_gqa_attend_inner(q_last, gate_last, k_cache, v_cache, n_ctx,
                             cpu_inner);
        ok = cpu_matvec_q8_0(&e->model, lw->attn_o_proj, cpu_inner, cpu_out);
    }
    if (ok) {
        ok = qw3_metal_session_gqa_cached_attn_out(
                 s->metal, lw->attn_o_proj->offset, (uint32_t)n_ctx, full_slot,
                 QW3_N_HEAD, QW3_N_HEAD_KV, QW3_N_HEAD_DIM,
                 QW3_N_EMBD, gpu_out) &&
             qw3_metal_synchronize();
    }
    if (ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu_out[i] - gpu_out[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * d;
        }
        rmsdiff = sqrt(rmsdiff / (double)QW3_N_EMBD);
        if (maxdiff > 0.05f || rmsdiff > 0.005) ok = 0;
    }

    double attend_ms = 0.0;
    if (ok) {
        const double t0 = qw3_now_sec();
        for (int i = 0; ok && i < iters; i++) {
            ok = qw3_metal_session_gqa_cached_attn_out(
                s->metal, lw->attn_o_proj->offset, (uint32_t)n_ctx, full_slot,
                QW3_N_HEAD, QW3_N_HEAD_KV, QW3_N_HEAD_DIM,
                QW3_N_EMBD, NULL);
        }
        if (ok) ok = qw3_metal_synchronize();
        attend_ms = (qw3_now_sec() - t0) * 1000.0;
    }

    fprintf(fp,
            "metal session gqa cached bench: %s token=%d layer=%d slot=%u n_ctx=%d iters=%d fill_ms=%.3f attend_total_ms=%.3f attend_ms=%.4f maxdiff=%.7g rmsdiff=%.7g out0=[%.7g %.7g %.7g %.7g]\n",
            ok ? "ok" : "failed", token, il, full_slot, n_ctx, iters, fill_ms,
            attend_ms, attend_ms / (double)iters, maxdiff, rmsdiff,
            gpu_out[0], gpu_out[1], gpu_out[2], gpu_out[3]);
    qw3_session_free(s);
    free(gpu_out);
    free(cpu_out);
    free(cpu_inner);
    free(v_cache);
    free(k_cache);
    free(gate_last);
    free(q_last);
    return ok ? 0 : -1;
#endif
}


static int
qw3_metal_session_eval_prefill_batch_mode(qw3_session *s, const int *tokens,
                                          int n_tokens, char *err,
                                          size_t errlen, int logits_mode) {
#ifdef QW3_NO_METAL
    (void)s; (void)tokens; (void)n_tokens; (void)err; (void)errlen;
    (void)logits_mode;
    return -1;
#else
    if (!s || !s->engine || !s->metal || !tokens || n_tokens <= 0) return -1;
    if (qw3_session_uses_partial_metal(s)) {
        if (err && errlen) {
            snprintf(err, errlen,
                     "Metal batched prefill is disabled with partial layer offload");
        }
        return -1;
    }
    if (s->kv.pos + (uint64_t)n_tokens > (uint64_t)s->ctx_size) {
        if (err && errlen) snprintf(err, errlen, "context is full");
        return -1;
    }
    for (int i = 0; i < n_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= QW3_N_VOCAB) {
            if (err && errlen) {
                snprintf(err, errlen, "token %d is outside vocab", tokens[i]);
            }
            return -1;
        }
    }

    qw3_engine *e = s->engine;
    const uint32_t pos0 = (uint32_t)s->kv.pos;
    const uint32_t ntok = (uint32_t)n_tokens;
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t n_z = (uint32_t)tensor_linear_inner();
    const uint32_t q_n = QW3_N_HEAD * QW3_N_HEAD_DIM;
    const uint32_t qg_n = (uint32_t)tensor_cols_qg();
    const uint32_t kv_n = (uint32_t)tensor_cols_kv();
    if (qg_n != 2u * q_n) {
        if (err && errlen) snprintf(err, errlen, "unexpected GQA q/g layout");
        return -1;
    }

    const uint32_t lin_gate_offset = n_qkv;
    const uint32_t lin_alpha_offset = lin_gate_offset + n_z;
    const uint32_t lin_beta_offset = lin_alpha_offset + QW3_N_LINEAR_V_HEADS;
    const uint32_t lin_conv_offset = lin_beta_offset + QW3_N_LINEAR_V_HEADS;
    const uint32_t lin_inner_offset = lin_conv_offset + n_qkv;
    const uint32_t lin_attn_offset = lin_inner_offset + n_z;

    const uint32_t gqa_qg_offset = 0;
    const uint32_t gqa_k_offset = gqa_qg_offset + qg_n;
    const uint32_t gqa_v_offset = gqa_k_offset + kv_n;
    const uint32_t gqa_q_tmp_offset = gqa_v_offset + kv_n;
    const uint32_t gqa_k_tmp_offset = gqa_q_tmp_offset + q_n;
    const uint32_t gqa_q_rope_offset = gqa_k_tmp_offset + kv_n;
    const uint32_t gqa_k_rope_offset = gqa_q_rope_offset + q_n;
    const uint32_t gqa_gate_offset = gqa_k_rope_offset + kv_n;
    const uint32_t gqa_inner_offset = gqa_gate_offset + q_n;
    const uint32_t gqa_attn_offset = gqa_inner_offset + q_n;

    const uint32_t lin_attn_end = lin_attn_offset + QW3_N_EMBD;
    const uint32_t gqa_attn_end = gqa_attn_offset + QW3_N_EMBD;
    const uint32_t router_offset =
        lin_attn_end > gqa_attn_end ? lin_attn_end : gqa_attn_end;
    const uint32_t moe_hidden_offset = router_offset + QW3_N_EXPERT;
    const uint32_t shared_gate_offset =
        moe_hidden_offset + QW3_N_EXPERT_USED * QW3_N_FF_EXP;
    const uint32_t shared_up_offset = shared_gate_offset + QW3_N_FF_SHARED;
    const uint32_t shared_hidden_offset = shared_up_offset + QW3_N_FF_SHARED;
    const uint32_t shared_down_offset =
        shared_hidden_offset + QW3_N_FF_SHARED;
    const uint32_t shared_scalar_offset = shared_down_offset + QW3_N_EMBD;
    const uint32_t stage_stride = shared_scalar_offset + 1u;

    uint32_t *btoks = qw3_xmalloc((size_t)ntok * sizeof(uint32_t));
    for (uint32_t i = 0; i < ntok; i++) btoks[i] = (uint32_t)tokens[i];

    const char *concurrent_env = getenv("QW3_METAL_PREFILL_CONCURRENT");
    const int concurrent_prefill =
        getenv("QW3_METAL_PREFILL_CONCURRENT_DISABLE") == NULL &&
        getenv("GGML_METAL_CONCURRENCY_DISABLE") == NULL &&
        (!concurrent_env || strcmp(concurrent_env, "0") != 0);
    int ok = (concurrent_prefill ?
              qw3_metal_begin_commands_concurrent() :
              qw3_metal_begin_commands()) &&
             qw3_metal_session_batch_embed_q8_0(
                 s->metal, e->weights.token_embd->offset, btoks, ntok,
                 QW3_N_EMBD);
    const int profile = getenv("QW3_METAL_PREFILL_PROFILE") != NULL;
    const int profile_gqa =
        getenv("QW3_METAL_PROFILE_PREFILL_GQA_SYNC") != NULL;
    const int profile_linear_proj =
        getenv("QW3_METAL_PROFILE_PREFILL_LINEAR_PROJ_SYNC") != NULL;
    const int profile_linear =
        getenv("QW3_METAL_PROFILE_PREFILL_LINEAR_SYNC") != NULL ||
        profile_linear_proj;
    double profile_t0 = profile ? qw3_now_sec() : 0.0;
    double profile_gqa_t0 = 0.0;
    double profile_linear_t0 = 0.0;
#define QW3_PREFILL_BARRIER() do {                                          \
        if (concurrent_prefill && ok) ok = qw3_metal_batch_barrier();        \
    } while (0)
#define QW3_PREFILL_PROFILE_STAGE(stage_name, layer_id) do {                 \
        if (profile && ok) {                                                 \
            ok = qw3_metal_synchronize();                                    \
            const double profile_t1 = qw3_now_sec();                         \
            fprintf(stderr,                                                  \
                    "qw3 metal prefill profile pos=%u ntok=%u layer=%d "     \
                    "stage=%s ms=%.3f\n",                                   \
                    pos0, ntok, (layer_id), (stage_name),                    \
                    (profile_t1 - profile_t0) * 1000.0);                    \
            profile_t0 = profile_t1;                                         \
            if (ok) ok = concurrent_prefill ?                                \
                qw3_metal_begin_commands_concurrent() :                      \
                qw3_metal_begin_commands();                                  \
        }                                                                    \
    } while (0)
#define QW3_PREFILL_PROFILE_GQA_START(layer_id) do {                         \
        if (profile_gqa && ok &&                                             \
            qw3_layer_is_full_attention((uint32_t)(layer_id))) {             \
            profile_gqa_t0 = qw3_now_sec();                                  \
        }                                                                    \
    } while (0)
#define QW3_PREFILL_PROFILE_GQA_STAGE(stage_name, layer_id) do {             \
        if (profile_gqa && ok) {                                             \
            ok = qw3_metal_synchronize();                                    \
            const double profile_gqa_t1 = qw3_now_sec();                     \
            fprintf(stderr,                                                  \
                    "qw3 metal prefill gqa profile pos=%u ntok=%u "          \
                    "layer=%d stage=%s ms=%.3f\n",                          \
                    pos0, ntok, (layer_id), (stage_name),                    \
                    (profile_gqa_t1 - profile_gqa_t0) * 1000.0);             \
            profile_gqa_t0 = profile_gqa_t1;                                 \
            if (ok) ok = concurrent_prefill ?                                \
                qw3_metal_begin_commands_concurrent() :                      \
                qw3_metal_begin_commands();                                  \
        }                                                                    \
    } while (0)
#define QW3_PREFILL_PROFILE_LINEAR_START(layer_id, is_full_layer) do {       \
        if (profile_linear && ok && !(is_full_layer)) {                      \
            profile_linear_t0 = qw3_now_sec();                               \
        }                                                                    \
    } while (0)
#define QW3_PREFILL_PROFILE_LINEAR_STAGE(stage_name, layer_id) do {          \
        if (profile_linear && ok) {                                          \
            ok = qw3_metal_synchronize();                                    \
            const double profile_linear_t1 = qw3_now_sec();                  \
            fprintf(stderr,                                                  \
                    "qw3 metal prefill linear profile pos=%u ntok=%u "       \
                    "layer=%d stage=%s ms=%.3f\n",                          \
                    pos0, ntok, (layer_id), (stage_name),                    \
                    (profile_linear_t1 - profile_linear_t0) * 1000.0);       \
            profile_linear_t0 = profile_linear_t1;                           \
            if (ok) ok = concurrent_prefill ?                                \
                qw3_metal_begin_commands_concurrent() :                      \
                qw3_metal_begin_commands();                                  \
        }                                                                    \
    } while (0)
    QW3_PREFILL_PROFILE_STAGE("embed", -1);
    QW3_PREFILL_BARRIER();
    int full_slot = 0;
    int linear_slot = 0;
    for (int il = 0; ok && il < QW3_N_LAYER; il++) {
        const qw3_layer_weights *lw = &e->weights.layer[il];
        const int is_full_attn = qw3_layer_is_full_attention((uint32_t)il);
        QW3_PREFILL_PROFILE_GQA_START(il);
        QW3_PREFILL_PROFILE_LINEAR_START(il, is_full_attn);
        ok = qw3_metal_session_batch_rmsnorm_weight_f32_x0_to_x1(
            s->metal, lw->attn_norm->offset, ntok, QW3_N_EMBD,
            QW3_RMS_EPS);
        QW3_PREFILL_BARRIER();
        if (ok && is_full_attn) {
            QW3_PREFILL_PROFILE_GQA_STAGE("attn_norm", il);
            ok =
                qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(
                    s->metal, lw->attn_q_proj->offset, ntok, QW3_N_EMBD,
                    qg_n, gqa_qg_offset, stage_stride) &&
                qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(
                    s->metal, lw->attn_k_proj->offset, ntok, QW3_N_EMBD,
                    kv_n, gqa_k_offset, stage_stride) &&
                qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(
                    s->metal, lw->attn_v_proj->offset, ntok, QW3_N_EMBD,
                    kv_n, gqa_v_offset, stage_stride);
            QW3_PREFILL_BARRIER();
            QW3_PREFILL_PROFILE_GQA_STAGE("qkv_proj", il);
            if (ok) {
                ok = qw3_metal_session_batch_gqa_norm_rope_from_scratch(
                    s->metal, lw->attn_q_norm->offset,
                    lw->attn_k_norm->offset, ntok, QW3_N_HEAD,
                    QW3_N_HEAD_KV, QW3_N_HEAD_DIM, QW3_ROPE_DIM,
                    pos0, QW3_ROPE_THETA, QW3_RMS_EPS, gqa_qg_offset,
                    gqa_k_offset, gqa_q_tmp_offset, gqa_k_tmp_offset,
                    gqa_q_rope_offset, gqa_k_rope_offset, gqa_gate_offset,
                    stage_stride);
            }
            QW3_PREFILL_BARRIER();
            QW3_PREFILL_PROFILE_GQA_STAGE("norm_rope", il);
            if (ok) {
                ok = qw3_metal_session_batch_gqa_write_cache_from_scratch(
                    s->metal, (uint32_t)full_slot, pos0, ntok,
                    QW3_N_HEAD_KV, QW3_N_HEAD_DIM, gqa_k_rope_offset,
                    gqa_v_offset, stage_stride);
            }
            QW3_PREFILL_BARRIER();
            QW3_PREFILL_PROFILE_GQA_STAGE("write_cache", il);
            if (ok) {
                ok = qw3_metal_session_batch_gqa_cached_attn_from_scratch(
                    s->metal, (uint32_t)full_slot, pos0, ntok,
                    QW3_N_HEAD, QW3_N_HEAD_KV, QW3_N_HEAD_DIM,
                    gqa_q_rope_offset, gqa_gate_offset, gqa_inner_offset,
                    stage_stride);
            }
            QW3_PREFILL_BARRIER();
            QW3_PREFILL_PROFILE_GQA_STAGE("attend", il);
            if (ok) {
                ok = qw3_metal_session_batch_matmul_q8_0_scratch_to_scratch(
                    s->metal, lw->attn_o_proj->offset, ntok, q_n,
                    QW3_N_EMBD, gqa_inner_offset, gqa_attn_offset,
                    stage_stride);
            }
            QW3_PREFILL_BARRIER();
            QW3_PREFILL_PROFILE_GQA_STAGE("out_proj", il);
            if (ok) {
                ok = qw3_metal_session_batch_residual_rmsnorm_update_x0_from_scratch(
                    s->metal, lw->ffn_norm->offset, ntok, QW3_N_EMBD,
                    gqa_attn_offset, stage_stride, QW3_RMS_EPS);
            }
            QW3_PREFILL_BARRIER();
            QW3_PREFILL_PROFILE_GQA_STAGE("residual_norm", il);
            QW3_PREFILL_PROFILE_STAGE("full_attn", il);
            full_slot++;
        } else if (ok) {
            QW3_PREFILL_PROFILE_LINEAR_STAGE("attn_norm", il);
            if (profile_linear_proj) {
                ok = qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(
                    s->metal, lw->linear_qkv_proj->offset, ntok,
                    QW3_N_EMBD, n_qkv, 0, stage_stride);
                QW3_PREFILL_BARRIER();
                QW3_PREFILL_PROFILE_LINEAR_STAGE("qkv_proj", il);
                if (ok) {
                    ok = qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(
                        s->metal, lw->linear_gate_proj->offset, ntok,
                        QW3_N_EMBD, n_z, lin_gate_offset, stage_stride);
                }
                QW3_PREFILL_BARRIER();
                QW3_PREFILL_PROFILE_LINEAR_STAGE("gate_proj", il);
                if (ok) {
                    ok = qw3_metal_session_batch_matmul_f32_pair_x1_to_scratch(
                        s->metal, lw->linear_ssm_alpha->offset,
                        lw->linear_ssm_beta->offset, ntok, QW3_N_EMBD,
                        QW3_N_LINEAR_V_HEADS, lin_alpha_offset,
                        lin_beta_offset, stage_stride);
                }
                QW3_PREFILL_BARRIER();
                QW3_PREFILL_PROFILE_LINEAR_STAGE("alpha_beta_proj", il);
            } else {
                ok =
                    qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(
                        s->metal, lw->linear_qkv_proj->offset, ntok,
                        QW3_N_EMBD, n_qkv, 0, stage_stride) &&
                    qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(
                        s->metal, lw->linear_gate_proj->offset, ntok,
                        QW3_N_EMBD, n_z, lin_gate_offset, stage_stride) &&
                    qw3_metal_session_batch_matmul_f32_pair_x1_to_scratch(
                        s->metal, lw->linear_ssm_alpha->offset,
                        lw->linear_ssm_beta->offset, ntok, QW3_N_EMBD,
                        QW3_N_LINEAR_V_HEADS, lin_alpha_offset,
                        lin_beta_offset, stage_stride);
                QW3_PREFILL_BARRIER();
                QW3_PREFILL_PROFILE_LINEAR_STAGE("qkv_gate_alpha_beta_proj", il);
            }
            if (ok) {
                ok = qw3_metal_session_batch_conv1d_step_from_scratch(
                    s->metal, lw->linear_conv_weight->offset,
                    (uint32_t)linear_slot, ntok, n_qkv, 0,
                    lin_conv_offset, stage_stride);
            }
            QW3_PREFILL_BARRIER();
            QW3_PREFILL_PROFILE_LINEAR_STAGE("conv1d", il);
            if (ok) {
                ok = qw3_metal_session_batch_l2norm_qk_from_scratch(
                    s->metal, ntok, lin_conv_offset, stage_stride,
                    QW3_N_LINEAR_QK_HEADS, QW3_N_LINEAR_HEAD_DIM,
                    QW3_RMS_EPS);
            }
            QW3_PREFILL_BARRIER();
            QW3_PREFILL_PROFILE_LINEAR_STAGE("l2norm_qk", il);
            if (ok) {
                ok = qw3_metal_session_batch_deltanet_fused_gdn_from_scratch(
                    s->metal, lw->linear_ssm_dt_bias->offset,
                    lw->linear_ssm_a->offset, lw->linear_ssm_norm->offset,
                    (uint32_t)linear_slot, ntok, lin_conv_offset,
                    lin_gate_offset, lin_alpha_offset, lin_beta_offset,
                    lin_inner_offset, stage_stride, QW3_N_LINEAR_QK_HEADS,
                    QW3_N_LINEAR_V_HEADS, QW3_N_LINEAR_HEAD_DIM,
                    QW3_RMS_EPS);
            }
            QW3_PREFILL_BARRIER();
            QW3_PREFILL_PROFILE_LINEAR_STAGE("deltanet_gdn", il);
            if (ok) {
                ok = qw3_metal_session_batch_matmul_q8_0_scratch_to_scratch(
                    s->metal, lw->linear_ssm_out->offset, ntok, n_z,
                    QW3_N_EMBD, lin_inner_offset, lin_attn_offset,
                    stage_stride);
            }
            QW3_PREFILL_BARRIER();
            QW3_PREFILL_PROFILE_LINEAR_STAGE("out_proj", il);
            if (ok) {
                ok = qw3_metal_session_batch_residual_rmsnorm_update_x0_from_scratch(
                    s->metal, lw->ffn_norm->offset, ntok, QW3_N_EMBD,
                    lin_attn_offset, stage_stride, QW3_RMS_EPS);
            }
            QW3_PREFILL_BARRIER();
            QW3_PREFILL_PROFILE_LINEAR_STAGE("residual_norm", il);
            QW3_PREFILL_PROFILE_STAGE("linear_attn", il);
            linear_slot++;
        }

        if (ok) {
            ok =
                qw3_metal_session_batch_matmul_f32_x1_to_scratch(
                    s->metal, lw->ffn_gate_inp->offset, ntok, QW3_N_EMBD,
                    QW3_N_EXPERT, router_offset, stage_stride);
            QW3_PREFILL_BARRIER();
            if (ok) {
                ok = qw3_metal_session_batch_sparse_moe_topk_from_router_scratch(
                    s->metal, lw->ffn_gate_exps->offset,
                    lw->ffn_up_exps->offset, lw->ffn_down_exps->offset,
                    (uint32_t)lw->ffn_down_exps->type, ntok,
                    QW3_N_EXPERT_USED, QW3_N_EMBD, QW3_N_FF_EXP,
                    router_offset, moe_hidden_offset, stage_stride);
            }
            QW3_PREFILL_BARRIER();
            QW3_PREFILL_PROFILE_STAGE("moe_sparse", il);
        }
        if (ok) {
            ok =
                qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(
                    s->metal, lw->ffn_gate_shared->offset, ntok,
                    QW3_N_EMBD, QW3_N_FF_SHARED, shared_gate_offset,
                    stage_stride) &&
                qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(
                    s->metal, lw->ffn_up_shared->offset, ntok,
                    QW3_N_EMBD, QW3_N_FF_SHARED, shared_up_offset,
                    stage_stride) &&
                qw3_metal_session_batch_matmul_f32_x1_to_scratch(
                    s->metal, lw->ffn_gate_inp_shexp->offset, ntok,
                    QW3_N_EMBD, 1, shared_scalar_offset, stage_stride);
            QW3_PREFILL_BARRIER();
            if (ok) {
                ok = qw3_metal_session_batch_silu_mul_scratch_to_scratch(
                    s->metal, ntok, QW3_N_FF_SHARED, shared_gate_offset,
                    shared_up_offset, shared_hidden_offset, stage_stride);
            }
            QW3_PREFILL_BARRIER();
            if (ok) {
                ok = qw3_metal_session_batch_matmul_q8_0_scratch_to_scratch(
                    s->metal, lw->ffn_down_shared->offset, ntok,
                    QW3_N_FF_SHARED, QW3_N_EMBD, shared_hidden_offset,
                    shared_down_offset, stage_stride);
            }
            QW3_PREFILL_BARRIER();
            if (ok) {
                ok = qw3_metal_session_batch_sigmoid_scale_scratch_add_x0(
                    s->metal, ntok, QW3_N_EMBD, shared_down_offset,
                    shared_scalar_offset, stage_stride);
            }
            QW3_PREFILL_BARRIER();
            QW3_PREFILL_PROFILE_STAGE("moe_shared", il);
        }
    }
#undef QW3_PREFILL_BARRIER
#undef QW3_PREFILL_PROFILE_GQA_START
#undef QW3_PREFILL_PROFILE_GQA_STAGE
#undef QW3_PREFILL_PROFILE_LINEAR_START
#undef QW3_PREFILL_PROFILE_LINEAR_STAGE
#undef QW3_PREFILL_PROFILE_STAGE

    if (ok) ok = qw3_metal_synchronize();
    if (ok && logits_mode != QW3_METAL_LOGITS_DEFER) {
        ok = qw3_metal_session_copy_batch_x0_to_x0(
                 s->metal, ntok - 1u, QW3_N_EMBD) &&
             qw3_metal_session_rmsnorm_weight_f32(
                 s->metal, e->weights.output_norm->offset,
                 QW3_N_EMBD, QW3_RMS_EPS, NULL);
        if (ok && e->weights.output->type == QW3_TENSOR_Q8_0) {
            ok = qw3_metal_session_matvec_q8_0_x1_to_logits(
                s->metal, e->weights.output->offset, QW3_N_EMBD,
                QW3_N_VOCAB,
                logits_mode == QW3_METAL_LOGITS_READ ? s->logits : NULL);
        } else if (ok && e->weights.output->type == QW3_TENSOR_Q6_K) {
            ok = qw3_metal_session_matvec_q6_k_x1_to_logits(
                s->metal, e->weights.output->offset, QW3_N_EMBD,
                QW3_N_VOCAB,
                logits_mode == QW3_METAL_LOGITS_READ ? s->logits : NULL);
        } else {
            ok = 0;
        }
    }

    if (ok) {
        for (int i = 0; i < n_tokens; i++) token_vec_push(&s->tokens, tokens[i]);
        s->kv.pos += n_tokens;
        s->valid = true;
    } else if (err && errlen) {
        snprintf(err, errlen, "Metal session batch prefill failed");
    }

    free(btoks);
    return ok ? 0 : -1;
#endif
}

static int
qw3_metal_session_eval_token_mode(qw3_session *s, int token,
                                  char *err, size_t errlen,
                                  int logits_mode) {
#ifdef QW3_NO_METAL
    (void)s; (void)token; (void)err; (void)errlen; (void)logits_mode;
    return -1;
#else
    if (!s || !s->engine || !s->metal) return -1;
    if (token < 0 || token >= QW3_N_VOCAB) {
        if (err && errlen) snprintf(err, errlen, "token %d is outside vocab", token);
        return -1;
    }
    if (s->kv.pos >= s->ctx_size) {
        if (err && errlen) snprintf(err, errlen, "context is full");
        return -1;
    }

    qw3_engine *e = s->engine;
    const uint32_t n_qkv = (uint32_t)tensor_linear_qkv();
    const uint32_t inner_n = (uint32_t)tensor_linear_inner();
    const uint32_t qg_n = (uint32_t)tensor_cols_qg();
    const uint32_t q_n = QW3_N_HEAD * QW3_N_HEAD_DIM;
    const uint32_t kv_n = (uint32_t)tensor_cols_kv();
    const uint32_t alpha_offset = n_qkv + inner_n;
    const uint32_t beta_offset = alpha_offset + QW3_N_LINEAR_V_HEADS;

    float *router = qw3_xmalloc((size_t)QW3_N_EXPERT * sizeof(float));
    const int profile = getenv("QW3_METAL_PROFILE") != NULL;
    const int graph_token_profile =
        getenv("QW3_METAL_GRAPH_TOKEN_PROFILE") != NULL;
    const int gpu_router_topk = getenv("QW3_METAL_GPU_ROUTER_TOPK") != NULL;
    const char *dynamic_router_env = getenv("QW3_METAL_DYNAMIC_ROUTER");
    const int cpu_router =
        getenv("QW3_METAL_CPU_ROUTER") != NULL ||
        (dynamic_router_env && strcmp(dynamic_router_env, "0") == 0);
    const int dynamic_router = !gpu_router_topk && !cpu_router;
    const int layer_flush = getenv("QW3_METAL_NO_LAYER_FLUSH") == NULL;
    const int profile_layers = getenv("QW3_METAL_PROFILE_LAYERS") != NULL;
    const int profile_layer_sync = getenv("QW3_METAL_PROFILE_LAYER_SYNC") != NULL;
    const int profile_stage_sync = getenv("QW3_METAL_PROFILE_STAGE_SYNC") != NULL;
    const int profile_attn_sync = getenv("QW3_METAL_PROFILE_ATTN_SYNC") != NULL;
    const int profile_proj_sync = getenv("QW3_METAL_PROFILE_PROJ_SYNC") != NULL;
    const int fused_gdn = getenv("QW3_METAL_LEGACY_GDN") == NULL;
    const int fused_shared_gate =
        getenv("QW3_METAL_LEGACY_SHARED_GATE") == NULL;
    const double t_eval0 = profile ? qw3_now_sec() : 0.0;
    const double t_graph_token0 = graph_token_profile ? qw3_now_sec() : 0.0;
    double t_router_sync = 0.0;
    double t_router_matvec = 0.0;
    double t_sparse_encode = 0.0;
    double t_pre_logits_sync = 0.0;
    double t_output_norm_sync = 0.0;
    double t_logits = 0.0;
    double t_graph_encoded = 0.0;
    double t_graph_done = 0.0;
    int graph_flushes = 0;

    int metal_layers = s->metal_n_gpu_layers;
    if (metal_layers < 0 || metal_layers > QW3_N_LAYER) {
        metal_layers = QW3_N_LAYER;
    }
    float *cpu_tail_in = NULL;
    int batch_open = 0;
    int ok = 1;
    if (metal_layers > 0) {
        ok = qw3_metal_begin_commands();
        if (!ok && qw3_metal_synchronize()) {
            ok = qw3_metal_begin_commands();
        }
        if (!ok && err && errlen && !err[0]) {
            snprintf(err, errlen, "Metal command buffer begin failed");
        }
        if (ok) {
            ok = qw3_metal_session_embed_q8_0(
                s->metal, e->weights.token_embd->offset, (uint32_t)token,
                QW3_N_EMBD, NULL);
            if (!ok && err && errlen && !err[0]) {
                snprintf(err, errlen, "Metal token embedding failed");
            }
        }
        batch_open = ok;
    } else {
        cpu_tail_in = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
        ok = tensor_read_dense_row(&e->model, e->weights.token_embd,
                                   (uint64_t)token, cpu_tail_in);
        if (!ok && err && errlen) {
            snprintf(err, errlen, "embedding read failed for tensor type %s",
                     tensor_type_name(e->weights.token_embd->type));
        }
    }
    int full_slot = 0;
    int linear_slot = 0;
    int last_metal_layer = -1;
    for (int il = 0; ok && il < metal_layers; il++) {
        last_metal_layer = il;
        const double t_layer_sync0 = profile_layer_sync ? qw3_now_sec() : 0.0;
        double t_stage_sync0 = profile_stage_sync ? qw3_now_sec() : 0.0;
        double t_attn_sync0 = profile_attn_sync ? qw3_now_sec() : 0.0;
        const qw3_layer_weights *lw = &e->weights.layer[il];
        if (qw3_layer_is_full_attention((uint32_t)il)) {
            ok =
                qw3_metal_session_rmsnorm_weight_f32(
                    s->metal, lw->attn_norm->offset,
                    QW3_N_EMBD, QW3_RMS_EPS, NULL) &&
                qw3_metal_session_gqa_project_cache(
                    s->metal, lw->attn_q_proj->offset,
                    lw->attn_k_proj->offset, lw->attn_v_proj->offset,
                    lw->attn_q_norm->offset, lw->attn_k_norm->offset,
                    qg_n, q_n, kv_n, QW3_N_HEAD, QW3_N_HEAD_KV,
                    QW3_N_HEAD_DIM, QW3_ROPE_DIM, (uint32_t)full_slot,
                    (uint32_t)s->kv.pos, QW3_ROPE_THETA, QW3_RMS_EPS,
                    NULL, NULL, NULL, NULL);
            if (ok && profile_attn_sync) {
                ok = qw3_metal_synchronize();
                batch_open = 0;
                fprintf(stderr,
                        "qw3 metal attn profile token=%d pos=%llu layer=%d kind=gqa project_ms=%.3f\n",
                        token, (unsigned long long)s->kv.pos, il,
                        (qw3_now_sec() - t_attn_sync0) * 1000.0);
                t_attn_sync0 = qw3_now_sec();
                if (ok) {
                    ok = qw3_metal_begin_commands();
                    batch_open = ok;
                }
            }
            if (ok && s->kv.pos == 0) {
                ok = qw3_metal_session_gqa_single_attn_out(
                    s->metal, lw->attn_o_proj->offset,
                    QW3_N_HEAD, QW3_N_HEAD_KV, QW3_N_HEAD_DIM,
                    QW3_N_EMBD, NULL);
            } else if (ok) {
                ok = qw3_metal_session_gqa_cached_attn_out(
                    s->metal, lw->attn_o_proj->offset,
                    (uint32_t)(s->kv.pos + 1), (uint32_t)full_slot,
                    QW3_N_HEAD, QW3_N_HEAD_KV, QW3_N_HEAD_DIM,
                    QW3_N_EMBD, NULL);
            }
            if (ok && profile_attn_sync) {
                ok = qw3_metal_synchronize();
                batch_open = 0;
                fprintf(stderr,
                        "qw3 metal attn profile token=%d pos=%llu layer=%d kind=gqa attend_out_ms=%.3f\n",
                        token, (unsigned long long)s->kv.pos, il,
                        (qw3_now_sec() - t_attn_sync0) * 1000.0);
                t_attn_sync0 = qw3_now_sec();
                if (ok) {
                    ok = qw3_metal_begin_commands();
                    batch_open = ok;
                }
            }
            full_slot++;
        } else {
            if (profile_proj_sync) {
                double t_proj_sync0 = qw3_now_sec();
                ok = qw3_metal_session_rmsnorm_weight_f32(
                    s->metal, lw->attn_norm->offset,
                    QW3_N_EMBD, QW3_RMS_EPS, NULL);
                if (ok) {
                    ok = qw3_metal_synchronize();
                    batch_open = 0;
                    fprintf(stderr,
                            "qw3 metal proj profile token=%d pos=%llu layer=%d kind=linear norm_ms=%.3f\n",
                            token, (unsigned long long)s->kv.pos, il,
                            (qw3_now_sec() - t_proj_sync0) * 1000.0);
                    t_proj_sync0 = qw3_now_sec();
                    if (ok) {
                        ok = qw3_metal_begin_commands();
                        batch_open = ok;
                    }
                }
                if (ok) {
                    ok = qw3_metal_session_matvec_q8_0_x1_to_scratch(
                        s->metal, lw->linear_qkv_proj->offset,
                        QW3_N_EMBD, n_qkv, 0, NULL);
                }
                if (ok) {
                    ok = qw3_metal_synchronize();
                    batch_open = 0;
                    fprintf(stderr,
                            "qw3 metal proj profile token=%d pos=%llu layer=%d kind=linear qkv_ms=%.3f\n",
                            token, (unsigned long long)s->kv.pos, il,
                            (qw3_now_sec() - t_proj_sync0) * 1000.0);
                    t_proj_sync0 = qw3_now_sec();
                    if (ok) {
                        ok = qw3_metal_begin_commands();
                        batch_open = ok;
                    }
                }
                if (ok) {
                    ok = qw3_metal_session_matvec_q8_0_x1_to_scratch(
                        s->metal, lw->linear_gate_proj->offset,
                        QW3_N_EMBD, inner_n, n_qkv, NULL);
                }
                if (ok) {
                    ok = qw3_metal_synchronize();
                    batch_open = 0;
                    fprintf(stderr,
                            "qw3 metal proj profile token=%d pos=%llu layer=%d kind=linear gate_ms=%.3f\n",
                            token, (unsigned long long)s->kv.pos, il,
                            (qw3_now_sec() - t_proj_sync0) * 1000.0);
                    t_proj_sync0 = qw3_now_sec();
                    if (ok) {
                        ok = qw3_metal_begin_commands();
                        batch_open = ok;
                    }
                }
                if (ok) {
                    ok = qw3_metal_session_matvec_f32_pair_x1_to_scratch(
                        s->metal, lw->linear_ssm_alpha->offset,
                        lw->linear_ssm_beta->offset, QW3_N_EMBD,
                        QW3_N_LINEAR_V_HEADS, alpha_offset, beta_offset);
                }
                if (ok) {
                    ok = qw3_metal_synchronize();
                    batch_open = 0;
                    fprintf(stderr,
                            "qw3 metal proj profile token=%d pos=%llu layer=%d kind=linear gates_f32_pair_ms=%.3f\n",
                            token, (unsigned long long)s->kv.pos, il,
                            (qw3_now_sec() - t_proj_sync0) * 1000.0);
                    if (ok) {
                        ok = qw3_metal_begin_commands();
                        batch_open = ok;
                    }
                }
                if (profile_attn_sync) t_attn_sync0 = qw3_now_sec();
            } else {
                ok =
                    qw3_metal_session_rmsnorm_weight_f32(
                        s->metal, lw->attn_norm->offset,
                        QW3_N_EMBD, QW3_RMS_EPS, NULL) &&
                    qw3_metal_session_matvec_q8_0_x1_to_scratch(
                        s->metal, lw->linear_qkv_proj->offset,
                        QW3_N_EMBD, n_qkv, 0, NULL) &&
                    qw3_metal_session_matvec_q8_0_x1_to_scratch(
                        s->metal, lw->linear_gate_proj->offset,
                        QW3_N_EMBD, inner_n, n_qkv, NULL) &&
                    qw3_metal_session_matvec_f32_pair_x1_to_scratch(
                        s->metal, lw->linear_ssm_alpha->offset,
                        lw->linear_ssm_beta->offset, QW3_N_EMBD,
                        QW3_N_LINEAR_V_HEADS, alpha_offset, beta_offset);
            }
            if (ok && profile_attn_sync && !profile_proj_sync) {
                ok = qw3_metal_synchronize();
                batch_open = 0;
                fprintf(stderr,
                        "qw3 metal attn profile token=%d pos=%llu layer=%d kind=linear project_ms=%.3f\n",
                        token, (unsigned long long)s->kv.pos, il,
                        (qw3_now_sec() - t_attn_sync0) * 1000.0);
                t_attn_sync0 = qw3_now_sec();
                if (ok) {
                    ok = qw3_metal_begin_commands();
                    batch_open = ok;
                }
            }
            ok = ok &&
                qw3_metal_session_conv1d_step_from_scratch(
                    s->metal, lw->linear_conv_weight->offset,
                    (uint32_t)linear_slot, n_qkv, NULL, NULL) &&
                qw3_metal_session_l2norm_qk_from_conv(
                    s->metal, QW3_N_LINEAR_QK_HEADS,
                    QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS, NULL, NULL);
            if (ok && profile_attn_sync) {
                ok = qw3_metal_synchronize();
                batch_open = 0;
                fprintf(stderr,
                        "qw3 metal attn profile token=%d pos=%llu layer=%d kind=linear conv_norm_ms=%.3f\n",
                        token, (unsigned long long)s->kv.pos, il,
                        (qw3_now_sec() - t_attn_sync0) * 1000.0);
                t_attn_sync0 = qw3_now_sec();
                if (ok) {
                    ok = qw3_metal_begin_commands();
                    batch_open = ok;
                }
            }
            if (ok) {
                if (fused_gdn) {
                    ok = qw3_metal_session_deltanet_tiled_gdn_from_scratch(
                        s->metal, lw->linear_ssm_dt_bias->offset,
                        lw->linear_ssm_a->offset, lw->linear_ssm_norm->offset,
                        n_qkv, alpha_offset, beta_offset,
                        (uint32_t)linear_slot,
                        QW3_N_LINEAR_QK_HEADS, QW3_N_LINEAR_V_HEADS,
                        QW3_N_LINEAR_HEAD_DIM, QW3_RMS_EPS);
                } else {
                    ok =
                        qw3_metal_session_deltanet_recur_from_scratch_gates(
                            s->metal, lw->linear_ssm_dt_bias->offset,
                            lw->linear_ssm_a->offset, alpha_offset, beta_offset,
                            (uint32_t)linear_slot,
                            QW3_N_LINEAR_QK_HEADS, QW3_N_LINEAR_V_HEADS,
                            QW3_N_LINEAR_HEAD_DIM, NULL, NULL) &&
                        qw3_metal_session_deltanet_gated_rmsnorm_from_buffers(
                        s->metal, lw->linear_ssm_norm->offset, n_qkv,
                        QW3_N_LINEAR_V_HEADS, QW3_N_LINEAR_HEAD_DIM,
                        QW3_RMS_EPS, NULL);
                }
                ok = ok &&
                    qw3_metal_session_matvec_q8_0_inner_to_x1(
                        s->metal, lw->linear_ssm_out->offset,
                        inner_n, QW3_N_EMBD, NULL);
            }
            if (ok && profile_attn_sync) {
                ok = qw3_metal_synchronize();
                batch_open = 0;
                fprintf(stderr,
                        "qw3 metal attn profile token=%d pos=%llu layer=%d kind=linear recur_out_ms=%.3f\n",
                        token, (unsigned long long)s->kv.pos, il,
                        (qw3_now_sec() - t_attn_sync0) * 1000.0);
                t_attn_sync0 = qw3_now_sec();
                if (ok) {
                    ok = qw3_metal_begin_commands();
                    batch_open = ok;
                }
            }
            linear_slot++;
        }
        if (ok) {
            double t0 = profile ? qw3_now_sec() : 0.0;
            const double layer_sync_t0 = profile_layers ? qw3_now_sec() : 0.0;
            ok = qw3_metal_session_residual_rmsnorm_update_x0_x1(
                s->metal, lw->ffn_norm->offset, QW3_N_EMBD,
                QW3_RMS_EPS, NULL);
            if (ok && profile_attn_sync) {
                ok = qw3_metal_synchronize();
                batch_open = 0;
                fprintf(stderr,
                        "qw3 metal attn profile token=%d pos=%llu layer=%d kind=%s residual_norm_ms=%.3f\n",
                        token, (unsigned long long)s->kv.pos, il,
                        qw3_layer_is_full_attention((uint32_t)il) ? "gqa" : "linear",
                        (qw3_now_sec() - t_attn_sync0) * 1000.0);
                if (ok) {
                    ok = qw3_metal_begin_commands();
                    batch_open = ok;
                }
            }
            if (ok && profile_stage_sync) {
                ok = qw3_metal_synchronize();
                batch_open = 0;
                fprintf(stderr,
                        "qw3 metal stage profile token=%d pos=%llu layer=%d kind=%s attention_ms=%.3f\n",
                        token, (unsigned long long)s->kv.pos, il,
                        qw3_layer_is_full_attention((uint32_t)il) ? "gqa" : "linear",
                        (qw3_now_sec() - t_stage_sync0) * 1000.0);
                t_stage_sync0 = qw3_now_sec();
                if (ok) {
                    ok = qw3_metal_begin_commands();
                    batch_open = ok;
                }
            }
            if (ok && !dynamic_router) ok = qw3_metal_synchronize();
            if (profile) t_router_sync += qw3_now_sec() - t0;
            if (profile_layers) {
                fprintf(stderr,
                        "qw3 metal layer profile token=%d pos=%llu layer=%d kind=%s pre_router_ms=%.3f ok=%d\n",
                        token, (unsigned long long)s->kv.pos, il,
                        qw3_layer_is_full_attention((uint32_t)il) ? "gqa" : "linear",
                        (qw3_now_sec() - layer_sync_t0) * 1000.0, ok);
            }
            t0 = profile ? qw3_now_sec() : 0.0;
            int ids[QW3_N_EXPERT_USED];
            float weights[QW3_N_EXPERT_USED];
            if (ok && dynamic_router) {
                ok = qw3_metal_session_matvec_f32_x1_to_scratch(
                    s->metal, lw->ffn_gate_inp->offset,
                    QW3_N_EMBD, QW3_N_EXPERT, 0, NULL);
            } else if (ok && gpu_router_topk) {
                ok = qw3_metal_begin_commands() &&
                     qw3_metal_session_matvec_f32_x1_to_scratch(
                         s->metal, lw->ffn_gate_inp->offset,
                         QW3_N_EMBD, QW3_N_EXPERT, 0, NULL) &&
                     qw3_metal_session_router_topk_from_scratch(
                         s->metal, 0, QW3_N_EXPERT, QW3_N_EXPERT_USED,
                         ids, weights);
            } else if (ok) {
                ok = qw3_metal_session_matvec_f32_x1_to_scratch(
                    s->metal, lw->ffn_gate_inp->offset,
                    QW3_N_EMBD, QW3_N_EXPERT, 0, router);
            }
            if (profile) t_router_matvec += qw3_now_sec() - t0;
            t0 = profile ? qw3_now_sec() : 0.0;
            if (dynamic_router) {
                ok = ok &&
                     qw3_metal_session_sparse_moe_topk_from_router_scratch(
                         s->metal, lw->ffn_gate_exps->offset,
                         lw->ffn_up_exps->offset, lw->ffn_down_exps->offset,
                         (uint32_t)lw->ffn_down_exps->type,
                         QW3_N_EXPERT_USED, QW3_N_EMBD, QW3_N_FF_EXP);
            } else if (gpu_router_topk) {
                ok = ok &&
                     qw3_metal_begin_commands() &&
                     qw3_metal_session_sparse_moe_topk(
                         s->metal, lw->ffn_gate_exps->offset,
                         lw->ffn_up_exps->offset, lw->ffn_down_exps->offset,
                         (uint32_t)lw->ffn_down_exps->type,
                         ids, weights, QW3_N_EXPERT_USED,
                         QW3_N_EMBD, QW3_N_FF_EXP);
            } else {
                ok = ok &&
                     qw3_metal_begin_commands() &&
                     qw3_metal_session_sparse_moe_from_router(s, lw, router, NULL);
            }
            if (profile) t_sparse_encode += qw3_now_sec() - t0;
            batch_open = ok;
            if (ok && profile_stage_sync) {
                ok = qw3_metal_synchronize();
                batch_open = 0;
                fprintf(stderr,
                        "qw3 metal stage profile token=%d pos=%llu layer=%d kind=%s sparse_ms=%.3f\n",
                        token, (unsigned long long)s->kv.pos, il,
                        qw3_layer_is_full_attention((uint32_t)il) ? "gqa" : "linear",
                        (qw3_now_sec() - t_stage_sync0) * 1000.0);
                t_stage_sync0 = qw3_now_sec();
                if (ok) {
                    ok = qw3_metal_begin_commands();
                    batch_open = ok;
                }
            }
        }
        if (ok) {
            const uint32_t sh_scalar_off = QW3_N_FF_SHARED * 2;
            if (fused_shared_gate) {
                ok =
                    qw3_metal_session_shared_gate_up_silu_x1_to_inner(
                        s->metal, lw->ffn_gate_shared->offset,
                        lw->ffn_up_shared->offset,
                        lw->ffn_gate_inp_shexp->offset, QW3_N_EMBD,
                        QW3_N_FF_SHARED, sh_scalar_off) &&
                    qw3_metal_session_matvec_q8_0_inner_scale_add_x0(
                        s->metal, lw->ffn_down_shared->offset,
                        QW3_N_FF_SHARED, QW3_N_EMBD, sh_scalar_off);
            } else {
                ok =
                    qw3_metal_session_matvec_q8_0_pair_silu_x1_to_inner(
                        s->metal, lw->ffn_gate_shared->offset,
                        lw->ffn_up_shared->offset, QW3_N_EMBD,
                        QW3_N_FF_SHARED) &&
                    qw3_metal_session_matvec_f32_x1_to_scratch(
                        s->metal, lw->ffn_gate_inp_shexp->offset,
                        QW3_N_EMBD, 1, sh_scalar_off, NULL) &&
                    qw3_metal_session_matvec_q8_0_inner_scale_add_x0(
                        s->metal, lw->ffn_down_shared->offset,
                        QW3_N_FF_SHARED, QW3_N_EMBD, sh_scalar_off);
            }
            if (ok && (layer_flush || profile_layer_sync || profile_stage_sync)) {
                ok = qw3_metal_flush_commands();
                if (ok && graph_token_profile) graph_flushes++;
                batch_open = ok;
            }
            if (ok && (profile_layer_sync || profile_stage_sync)) {
                ok = qw3_metal_synchronize();
                batch_open = 0;
                if (profile_stage_sync) {
                    fprintf(stderr,
                            "qw3 metal stage profile token=%d pos=%llu layer=%d kind=%s shared_ms=%.3f\n",
                            token, (unsigned long long)s->kv.pos, il,
                            qw3_layer_is_full_attention((uint32_t)il) ? "gqa" : "linear",
                            (qw3_now_sec() - t_stage_sync0) * 1000.0);
                }
                if (profile_layer_sync) {
                    fprintf(stderr,
                            "qw3 metal layer sync profile token=%d pos=%llu layer=%d kind=%s total_ms=%.3f\n",
                            token, (unsigned long long)s->kv.pos, il,
                            qw3_layer_is_full_attention((uint32_t)il) ? "gqa" : "linear",
                            (qw3_now_sec() - t_layer_sync0) * 1000.0);
                }
                if (ok && il + 1 < metal_layers) {
                    ok = qw3_metal_begin_commands();
                    batch_open = ok;
                }
            }
        }
    }

    if (!ok && err && errlen && !err[0]) {
        snprintf(err, errlen, "Metal layer %d failed before CPU offload",
                 last_metal_layer);
    }

    if (ok && metal_layers < QW3_N_LAYER) {
        if (!cpu_tail_in) {
            cpu_tail_in = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
            if (batch_open) {
                ok = qw3_metal_synchronize();
                batch_open = 0;
                if (!ok && err && errlen && !err[0]) {
                    snprintf(err, errlen,
                             "Metal command buffer sync failed before CPU offload");
                }
            }
            if (ok) {
                ok = qw3_metal_session_read_x0(s->metal, cpu_tail_in, QW3_N_EMBD);
            }
            if (!ok && err && errlen && !err[0]) {
                snprintf(err, errlen, "Metal activation readback failed at layer %d",
                         metal_layers);
            }
        }
        float *cpu_tail_out = ok ?
            qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float)) : NULL;
        if (ok) {
            ok = qw3_cpu_eval_layer_range(s, metal_layers, cpu_tail_in,
                                          cpu_tail_out, err, errlen,
                                          NULL, false, NULL);
            if (!ok && err && errlen && !err[0]) {
                snprintf(err, errlen, "CPU tail failed from layer %d",
                         metal_layers);
            }
        }
        if (ok) {
            ok = qw3_cpu_output_logits(s, cpu_tail_out, err, errlen);
            if (!ok && err && errlen && !err[0]) {
                snprintf(err, errlen, "CPU output logits failed after layer %d",
                         QW3_N_LAYER - 1);
            }
        }
        free(cpu_tail_out);
    } else if (ok && logits_mode == QW3_METAL_LOGITS_DEFER) {
        ok = batch_open ? qw3_metal_commit_commands() : 1;
        batch_open = 0;
    } else if (ok && logits_mode == QW3_METAL_LOGITS_GPU) {
        double t0 = profile ? qw3_now_sec() : 0.0;
        if (profile) {
            ok = qw3_metal_synchronize();
            batch_open = 0;
            t_pre_logits_sync += qw3_now_sec() - t0;
            t0 = qw3_now_sec();
        }
        if (!batch_open) {
            ok = qw3_metal_begin_commands();
            batch_open = ok;
        }
        ok = ok &&
             qw3_metal_session_rmsnorm_weight_f32(
                 s->metal, e->weights.output_norm->offset,
                 QW3_N_EMBD, QW3_RMS_EPS, NULL);
        if (profile && ok) {
            ok = qw3_metal_synchronize();
            batch_open = 0;
        }
        if (profile) t_output_norm_sync += qw3_now_sec() - t0;
        t0 = profile ? qw3_now_sec() : 0.0;
        if (ok && !batch_open) {
            ok = qw3_metal_begin_commands();
            batch_open = ok;
        }
        if (ok && e->weights.output->type == QW3_TENSOR_Q8_0) {
            ok = qw3_metal_session_matvec_q8_0_x1_to_logits(
                s->metal, e->weights.output->offset,
                QW3_N_EMBD, QW3_N_VOCAB, NULL);
        } else if (ok && e->weights.output->type == QW3_TENSOR_Q6_K) {
            ok = qw3_metal_session_matvec_q6_k_x1_to_logits(
                s->metal, e->weights.output->offset,
                QW3_N_EMBD, QW3_N_VOCAB, NULL);
        } else {
            ok = 0;
        }
        if (ok && graph_token_profile) t_graph_encoded = qw3_now_sec();
        if (ok) ok = qw3_metal_synchronize();
        if (ok && graph_token_profile) t_graph_done = qw3_now_sec();
        if (profile) t_logits += qw3_now_sec() - t0;
        batch_open = 0;
    } else if (ok) {
        double t0 = profile ? qw3_now_sec() : 0.0;
        ok = qw3_metal_synchronize();
        if (profile) t_pre_logits_sync += qw3_now_sec() - t0;
        t0 = profile ? qw3_now_sec() : 0.0;
        ok = ok &&
             qw3_metal_begin_commands() &&
             qw3_metal_session_rmsnorm_weight_f32(
                 s->metal, e->weights.output_norm->offset,
                 QW3_N_EMBD, QW3_RMS_EPS, NULL) &&
             qw3_metal_synchronize();
        if (profile) t_output_norm_sync += qw3_now_sec() - t0;
        batch_open = 0;
        t0 = profile ? qw3_now_sec() : 0.0;
        if (ok && e->weights.output->type == QW3_TENSOR_Q8_0) {
            ok = qw3_metal_session_matvec_q8_0_x1_to_logits(
                s->metal, e->weights.output->offset,
                QW3_N_EMBD, QW3_N_VOCAB, s->logits);
        } else if (ok && e->weights.output->type == QW3_TENSOR_Q6_K) {
            ok = qw3_metal_session_matvec_q6_k_x1_to_logits(
                s->metal, e->weights.output->offset,
                QW3_N_EMBD, QW3_N_VOCAB, s->logits);
        } else {
            ok = 0;
        }
        if (profile) t_logits += qw3_now_sec() - t0;
    }
    if (!ok && batch_open) (void)qw3_metal_synchronize();
    if (ok) {
        token_vec_push(&s->tokens, token);
        s->kv.pos++;
        s->valid = true;
    } else if (err && errlen && !err[0]) {
        snprintf(err, errlen, "Metal session slow eval failed");
    }

    free(router);
    free(cpu_tail_in);
    if (profile) {
        const double total = qw3_now_sec() - t_eval0;
        fprintf(stderr,
                "qw3 metal profile token=%d pos=%d total_ms=%.1f router_sync_ms=%.1f router_matvec_ms=%.1f sparse_encode_ms=%.1f pre_logits_sync_ms=%.1f output_norm_sync_ms=%.1f logits_ms=%.1f\n",
                token, s->kv.pos, total * 1000.0, t_router_sync * 1000.0,
                t_router_matvec * 1000.0, t_sparse_encode * 1000.0,
                t_pre_logits_sync * 1000.0, t_output_norm_sync * 1000.0,
                t_logits * 1000.0);
    }
    if (graph_token_profile &&
        logits_mode == QW3_METAL_LOGITS_GPU && t_graph_done != 0.0) {
        fprintf(stderr,
                "qw3: metal graph token token=%d pos=%d flushes=%d "
                "submit_ms=%.3f final_wait_ms=%.3f total_ms=%.3f\n",
                token, s->kv.pos - 1, graph_flushes,
                (t_graph_encoded - t_graph_token0) * 1000.0,
                (t_graph_done - t_graph_encoded) * 1000.0,
                (t_graph_done - t_graph_token0) * 1000.0);
    }
    return ok ? 0 : -1;
#endif
}

static int
qw3_metal_session_eval_token_slow_ex(qw3_session *s, int token,
                                     char *err, size_t errlen,
                                     int read_logits) {
    return qw3_metal_session_eval_token_mode(
        s, token, err, errlen,
        read_logits < 0 ? QW3_METAL_LOGITS_DEFER :
        (read_logits ? QW3_METAL_LOGITS_READ : QW3_METAL_LOGITS_GPU));
}

static int
qw3_metal_session_eval_token_defer_logits(qw3_session *s, int token,
                                          char *err, size_t errlen) {
    return qw3_metal_session_eval_token_slow_ex(s, token, err, errlen, -1);
}

static int __attribute__((unused))
qw3_metal_session_eval_token_slow(qw3_session *s, int token,
                                  char *err, size_t errlen) {
    return qw3_metal_session_eval_token_slow_ex(s, token, err, errlen, 1);
}

int qw3_engine_metal_session_decode_test(qw3_engine *e,
                                         const qw3_tokens *prompt,
                                         int ctx_size, FILE *fp) {
    if (!e || !prompt || !fp || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)prompt; (void)ctx_size;
    fprintf(fp, "metal session decode: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal session decode: Metal backend is not initialized\n");
        return -1;
    }
    qw3_session *cpu = NULL;
    qw3_session *gpu = NULL;
    char err[256] = {0};
    int ok = prompt->len > 0 &&
             qw3_session_create(&cpu, e, ctx_size) == 0 &&
             qw3_session_create(&gpu, e, ctx_size) == 0;
    for (int i = 0; ok && i < prompt->len; i++) {
        ok = qw3_metal_session_eval_token_slow(cpu, prompt->v[i],
                                               err, sizeof(err)) == 0;
    }
    if (ok) {
        ok = qw3_session_sync(gpu, prompt, err, sizeof(err)) == 0;
    }
    float maxdiff = 0.0f;
    double rmsdiff = 0.0;
    int cpu_ids[8] = {0}, gpu_ids[8] = {0};
    float cpu_vals[8] = {0}, gpu_vals[8] = {0};
    if (ok) {
        for (int i = 0; i < QW3_N_VOCAB; i++) {
            float d = fabsf(cpu->logits[i] - gpu->logits[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * (double)d;
        }
        rmsdiff = sqrt(rmsdiff / (double)QW3_N_VOCAB);
        topk_desc(cpu->logits, QW3_N_VOCAB, 8, cpu_ids, cpu_vals);
        topk_desc(gpu->logits, QW3_N_VOCAB, 8, gpu_ids, gpu_vals);
        ok = cpu_ids[0] == gpu_ids[0];
    }
    fprintf(fp,
            "metal session decode: %s tokens=%d ctx=%d pos=%d maxdiff=%.7g rmsdiff=%.7g cpu_top0=%d gpu_top0=%d\n",
            ok ? "ok" : "failed", prompt->len, ctx_size,
            gpu ? gpu->kv.pos : -1, maxdiff, rmsdiff,
            cpu_ids[0], gpu_ids[0]);
    if (!ok && err[0]) fprintf(fp, "metal session decode error: %s\n", err);
    if (ok) {
        fprintf(fp, "metal session decode cpu top8:");
        for (int i = 0; i < 8; i++) fprintf(fp, " %d:%.7g", cpu_ids[i], cpu_vals[i]);
        fprintf(fp, "\nmetal session decode gpu top8:");
        for (int i = 0; i < 8; i++) fprintf(fp, " %d:%.7g", gpu_ids[i], gpu_vals[i]);
        fprintf(fp, "\n");
    }
    qw3_session_free(gpu);
    qw3_session_free(cpu);
    return ok ? 0 : -1;
#endif
}

#ifndef QW3_NO_METAL
static void qw3_metal_debug_conv_state_diff(qw3_session *ref,
                                            qw3_session *gpu,
                                            FILE *fp) {
    if (!ref || !gpu || !ref->metal || !gpu->metal || !fp) return;
    const uint32_t n_channels = (uint32_t)tensor_linear_qkv();
    const size_t n = (size_t)n_channels * 3u;
    float *a = qw3_xmalloc(n * sizeof(float));
    float *b = qw3_xmalloc(n * sizeof(float));
    for (uint32_t slot = 0; slot < QW3_N_LINEAR_LAYERS; slot++) {
        if (!qw3_metal_session_read_conv_state(ref->metal, slot, n_channels, a) ||
            !qw3_metal_session_read_conv_state(gpu->metal, slot, n_channels, b)) {
            fprintf(fp, "metal state conv: read failed slot=%u\n", slot);
            break;
        }
        float maxdiff = 0.0f;
        double rmsdiff = 0.0;
        size_t maxi = 0;
        for (size_t i = 0; i < n; i++) {
            float d = fabsf(a[i] - b[i]);
            if (d > maxdiff) {
                maxdiff = d;
                maxi = i;
            }
            rmsdiff += (double)d * (double)d;
        }
        rmsdiff = sqrt(rmsdiff / (double)n);
        if (maxdiff > 1e-4f) {
            fprintf(fp,
                    "metal state conv: slot=%u maxdiff=%.7g rmsdiff=%.7g idx=%zu ref=%.7g gpu=%.7g\n",
                    slot, maxdiff, rmsdiff, maxi, a[maxi], b[maxi]);
        }
    }
    free(b);
    free(a);
}

static void qw3_metal_debug_deltanet_state_diff(qw3_session *ref,
                                                qw3_session *gpu,
                                                FILE *fp) {
    if (!ref || !gpu || !ref->metal || !gpu->metal || !fp) return;
    const uint32_t v_heads = QW3_N_LINEAR_V_HEADS;
    const uint32_t head_dim = QW3_N_LINEAR_HEAD_DIM;
    const size_t n = (size_t)v_heads * head_dim * head_dim;
    float *a = qw3_xmalloc(n * sizeof(float));
    float *b = qw3_xmalloc(n * sizeof(float));
    for (uint32_t slot = 0; slot < QW3_N_LINEAR_LAYERS; slot++) {
        if (!qw3_metal_session_read_deltanet_state(ref->metal, slot, v_heads,
                                                   head_dim, a) ||
            !qw3_metal_session_read_deltanet_state(gpu->metal, slot, v_heads,
                                                   head_dim, b)) {
            fprintf(fp, "metal state deltanet: read failed slot=%u\n", slot);
            break;
        }
        float maxdiff = 0.0f;
        double rmsdiff = 0.0;
        size_t maxi = 0;
        for (size_t i = 0; i < n; i++) {
            float d = fabsf(a[i] - b[i]);
            if (d > maxdiff) {
                maxdiff = d;
                maxi = i;
            }
            rmsdiff += (double)d * (double)d;
        }
        rmsdiff = sqrt(rmsdiff / (double)n);
        if (maxdiff > 1e-4f) {
            fprintf(fp,
                    "metal state deltanet: slot=%u maxdiff=%.7g rmsdiff=%.7g idx=%zu ref=%.7g gpu=%.7g\n",
                    slot, maxdiff, rmsdiff, maxi, a[maxi], b[maxi]);
        }
    }
    free(b);
    free(a);
}
#endif

int qw3_engine_metal_greedy_test(qw3_engine *e, const qw3_tokens *prompt,
                                 int ctx_size, int n_steps, FILE *fp) {
    if (!e || !prompt || !fp || ctx_size <= 0 || n_steps < 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)prompt; (void)ctx_size; (void)n_steps;
    fprintf(fp, "metal greedy: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (n_steps == 0) n_steps = 1;
    const double t0 = qw3_now_sec();
    qw3_session *cpu = NULL;
    qw3_session *gpu = NULL;
    char err[256] = {0};
    int ok = qw3_session_create(&cpu, e, ctx_size) == 0 &&
             qw3_session_create(&gpu, e, ctx_size) == 0;
    for (int i = 0; ok && i < prompt->len; i++) {
        ok = qw3_metal_session_eval_token_slow(cpu, prompt->v[i],
                                               err, sizeof(err)) == 0;
    }
    if (ok) {
        ok = qw3_session_sync(gpu, prompt, err, sizeof(err)) == 0;
    }
#ifndef QW3_NO_METAL
    if (ok && getenv("QW3_METAL_STATE_DIFF_PROMPT") != NULL) {
        qw3_metal_debug_deltanet_state_diff(cpu, gpu, fp);
        qw3_metal_debug_conv_state_diff(cpu, gpu, fp);
    }
#endif
    for (int step = 0; ok && step < n_steps; step++) {
        if (gpu->kv.pos >= ctx_size) {
            fprintf(fp, "metal greedy: context full at step %d len=%d ctx=%d\n",
                    step, gpu->kv.pos, ctx_size);
            ok = 0;
            break;
        }
        int cpu_ids[8] = {0};
        float cpu_vals[8] = {0};
        topk_desc(cpu->logits, QW3_N_VOCAB, 8, cpu_ids, cpu_vals);
        uint32_t gpu_top0 = 0;
        float gpu_top0_val = 0.0f;
        ok = qw3_metal_argmax(gpu->logits, QW3_N_VOCAB,
                              &gpu_top0, &gpu_top0_val);
        if (!ok) break;
        float maxdiff = 0.0f;
        double rmsdiff = 0.0;
        for (int i = 0; i < QW3_N_VOCAB; i++) {
            float d = fabsf(cpu->logits[i] - gpu->logits[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * (double)d;
        }
        rmsdiff = sqrt(rmsdiff / (double)QW3_N_VOCAB);
        fprintf(fp,
                "metal greedy step %d: cpu_top0=%d gpu_top0=%u gpu_top0_val=%.7g maxdiff=%.7g rmsdiff=%.7g\n",
                step, cpu_ids[0], gpu_top0, gpu_top0_val, maxdiff, rmsdiff);
        if (cpu_ids[0] != (int)gpu_top0) {
            fprintf(fp,
                    "metal greedy: mismatch step=%d cpu_top0=%d gpu_top0=%u maxdiff=%.7g rmsdiff=%.7g\n",
                    step, cpu_ids[0], gpu_top0, maxdiff, rmsdiff);
#ifndef QW3_NO_METAL
            if (getenv("QW3_METAL_STATE_DIFF") != NULL) {
                qw3_metal_debug_deltanet_state_diff(cpu, gpu, fp);
                qw3_metal_debug_conv_state_diff(cpu, gpu, fp);
            }
#endif
            ok = 0;
            break;
        }
        ok = qw3_session_eval(cpu, (int)gpu_top0, err, sizeof(err)) == 0 &&
             qw3_metal_session_eval_token_slow(gpu, (int)gpu_top0,
                                               err, sizeof(err)) == 0;
    }
    fprintf(fp, "metal greedy: %s prompt_tokens=%d generated=%d final_tokens=%d total_ms=%.1f\n",
            ok ? "ok" : "failed", prompt->len,
            ok && gpu ? n_steps : (gpu ? gpu->tokens.len - prompt->len : 0),
            gpu ? gpu->tokens.len : 0, (qw3_now_sec() - t0) * 1000.0);
    if (!ok && err[0]) fprintf(fp, "metal greedy error: %s\n", err);
    if (gpu && gpu->tokens.len > prompt->len) {
        fprintf(fp, "metal greedy generated ids:");
        for (int i = prompt->len; i < gpu->tokens.len; i++) {
            fprintf(fp, " %d", gpu->tokens.v[i]);
        }
        fprintf(fp, "\n");
    }
    qw3_session_free(gpu);
    qw3_session_free(cpu);
    return ok ? 0 : -1;
#endif
}

int qw3_engine_metal_greedy_run(qw3_engine *e, const qw3_tokens *prompt,
                                int ctx_size, int n_steps, int quiet,
                                FILE *fp) {
    if (!e || !prompt || !fp || ctx_size <= 0 || n_steps <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)prompt; (void)ctx_size; (void)n_steps; (void)quiet;
    fprintf(fp, "metal run: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    const double t0 = qw3_now_sec();
    const int profile = getenv("QW3_METAL_PROFILE") != NULL;
    double t_argmax = 0.0;
    int n_argmax = 0;
    qw3_session *gpu = NULL;
    char err[256] = {0};
    int ok = qw3_session_create(&gpu, e, ctx_size) == 0;
    const double t_prefill0 = qw3_now_sec();
    const int prefill_batch = qw3_session_uses_partial_metal(gpu) ?
        1 : qw3_metal_prefill_batch_size();
    if (prefill_batch > 1) {
        for (int i = 0; ok && i < prompt->len;) {
            int n = prompt->len - i;
            if (n > prefill_batch) n = prefill_batch;
            const int last = i + n == prompt->len;
            int rc = 0;
            if (n > 1) {
                rc = qw3_metal_session_eval_prefill_batch_mode(
                    gpu, prompt->v + i, n, err, sizeof(err),
                    last ? QW3_METAL_LOGITS_GPU : QW3_METAL_LOGITS_DEFER);
            } else {
                rc = last ?
                    qw3_metal_session_eval_token_slow_ex(
                        gpu, prompt->v[i], err, sizeof(err), 0) :
                    qw3_metal_session_eval_token_defer_logits(
                        gpu, prompt->v[i], err, sizeof(err));
            }
            ok = rc == 0;
            i += n;
        }
    } else {
        for (int i = 0; ok && i < prompt->len; i++) {
            ok = qw3_metal_session_eval_token_slow_ex(gpu, prompt->v[i],
                                                      err, sizeof(err), 0) == 0;
        }
    }
    const double t_prefill1 = qw3_now_sec();
    const double t_gen0 = qw3_now_sec();
    int generated = 0;
    for (int step = 0; ok && step < n_steps; step++) {
        if (gpu->kv.pos >= ctx_size) {
            fprintf(fp, "metal run: context full at step %d len=%d ctx=%d\n",
                    step, gpu->kv.pos, ctx_size);
            ok = 0;
            break;
        }
        int top0 = -1;
        float top0_val = 0.0f;
        const double t_argmax0 = profile ? qw3_now_sec() : 0.0;
        if (qw3_session_uses_partial_metal(gpu)) {
            top0 = qw3_session_argmax(gpu);
            top0_val = top0 >= 0 ? gpu->logits[top0] : 0.0f;
            ok = top0 >= 0;
        } else {
            uint32_t gpu_top0 = 0;
            ok = qw3_metal_session_argmax_logits(gpu->metal, QW3_N_VOCAB,
                                                 &gpu_top0, &top0_val);
            top0 = (int)gpu_top0;
        }
        if (profile) {
            t_argmax += qw3_now_sec() - t_argmax0;
            n_argmax++;
        }
        if (!ok) break;
        if (!quiet) {
            fprintf(fp, "metal run step %d: top0=%d top0_val=%.7g\n",
                    step, top0, top0_val);
        }
        ok = qw3_metal_session_eval_token_slow_ex(gpu, top0,
                                                  err, sizeof(err), 0) == 0;
        if (ok) generated++;
    }
    const double t_gen1 = qw3_now_sec();
    const double prefill_ms = (t_prefill1 - t_prefill0) * 1000.0;
    const double gen_ms = (t_gen1 - t_gen0) * 1000.0;
    const double total_ms = (t_gen1 - t0) * 1000.0;
    const double gen_tps = gen_ms > 0.0 ? (double)generated * 1000.0 / gen_ms : 0.0;
    const double avg_decode_ms = generated > 0 ? gen_ms / (double)generated : 0.0;
    fprintf(fp,
            "metal run: %s prompt_tokens=%d generated=%d final_tokens=%d "
            "prefill_ms=%.1f generation_ms=%.1f avg_decode_ms=%.2f "
            "generation_tok_s=%.2f total_ms=%.1f\n",
            ok ? "ok" : "failed", prompt->len, generated,
            gpu ? gpu->tokens.len : 0, prefill_ms, gen_ms,
            avg_decode_ms, gen_tps, total_ms);
    if (profile) {
        fprintf(stderr,
                "qw3 metal argmax profile calls=%d total_ms=%.1f avg_ms=%.3f\n",
                n_argmax, t_argmax * 1000.0,
                n_argmax ? t_argmax * 1000.0 / (double)n_argmax : 0.0);
    }
    if (!ok && err[0]) fprintf(fp, "metal run error: %s\n", err);
    if (!quiet && gpu && gpu->tokens.len > prompt->len) {
        fprintf(fp, "metal run generated ids:");
        for (int i = prompt->len; i < gpu->tokens.len; i++) {
            fprintf(fp, " %d", gpu->tokens.v[i]);
        }
        fprintf(fp, "\n");
    }
    qw3_session_free(gpu);
    return ok ? 0 : -1;
#endif
}

int qw3_engine_metal_generate_argmax(qw3_engine *e, const qw3_tokens *prompt,
                                     int n_predict, int ctx_size,
                                     qw3_token_emit_fn emit,
                                     qw3_generation_done_fn done,
                                     void *emit_ud) {
    if (!e || !prompt || n_predict < 0 || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)prompt; (void)n_predict; (void)ctx_size;
    (void)emit; (void)done; (void)emit_ud;
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) return -1;

    qw3_session *s = NULL;
    char err[256] = {0};
    const int eos = qw3_token_eos(e);
    int rc = 0;
    if (qw3_session_create(&s, e, ctx_size) != 0) return -1;

    /* --- Prefill phase --- */
    const double t_prefill_start = qw3_now_sec();
    const int prefill_batch = qw3_session_uses_partial_metal(s) ?
        1 : qw3_metal_prefill_batch_size();
     
    if (prefill_batch > 1) {
        for (int i = 0; i < prompt->len;) {
            int n = prompt->len - i;
            if (n > prefill_batch) n = prefill_batch;
            const int last = i + n == prompt->len;
            int prc = 0;
            if (n > 1) {
                prc = qw3_metal_session_eval_prefill_batch_mode(
                    s, prompt->v + i, n, err, sizeof(err),
                    last ? QW3_METAL_LOGITS_GPU : QW3_METAL_LOGITS_DEFER);
            } else {
                prc = last ?
                    qw3_metal_session_eval_token_slow_ex(
                        s, prompt->v[i], err, sizeof(err), 0) :
                    qw3_metal_session_eval_token_defer_logits(
                        s, prompt->v[i], err, sizeof(err));
            }
            if (prc != 0) {
                fprintf(stderr, "qw3: Metal session prefill failed: %s\n", err);
                qw3_session_free(s);
                return -1;
            }
            i += n;
        }
    } else {
        const int defer_interval = qw3_prefill_defer_interval();
        int deferred = 0;
        for (int i = 0; i < prompt->len; i++) {
            const int last = i + 1 == prompt->len;
            int prc = last ?
                qw3_metal_session_eval_token_slow_ex(s, prompt->v[i],
                                                     err, sizeof(err), 0) :
                qw3_metal_session_eval_token_defer_logits(s, prompt->v[i],
                                                          err, sizeof(err));
            if (prc != 0) {
                fprintf(stderr, "qw3: Metal session prefill failed: %s\n", err);
                qw3_session_free(s);
                return -1;
            }
            if (!last && ++deferred >= defer_interval) {
                if (!qw3_metal_synchronize()) {
                    fprintf(stderr, "qw3: Metal deferred prefill failed\n");
                    qw3_session_free(s);
                    return -1;
                }
                deferred = 0;
            }
        }
    }
    const double t_prefill_end = qw3_now_sec();

    /* --- Generation phase --- */
    const double t_gen_start = qw3_now_sec();
    int n_generated = 0;
    for (int step = 0; step < n_predict; step++) {
        if (s->kv.pos >= ctx_size) {
            fprintf(stderr, "qw3: Metal generation context full at step %d len=%d ctx=%d\n",
                    step, s->kv.pos, ctx_size);
            rc = -1;
            break;
        }

        int top0 = qw3_session_argmax(s);
        if (top0 < 0) {
            rc = -1;
            break;
        }

        if (top0 == eos) break;
        n_generated++;
        if (emit) emit(emit_ud, top0);
        if (qw3_metal_session_eval_token_slow_ex(s, top0,
                                                 err, sizeof(err), 0) != 0) {
            fprintf(stderr, "qw3: Metal session decode failed: %s\n", err);
            rc = -1;
            break;
        }
    }
    const double t_gen_end = qw3_now_sec();

    if (done) done(emit_ud);

    /* --- Timing summary --- */
    const double dt_prefill = t_prefill_end - t_prefill_start;
    const double dt_gen = t_gen_end - t_gen_start;
    const double dt_total = t_gen_end - t_prefill_start;
    const double prefill_tps = (dt_prefill > 0.0) ? (double)prompt->len / dt_prefill : 0.0;
    const double gen_tps = (dt_gen > 0.0) ? (double)n_generated / dt_gen : 0.0;
    qw3_log(stderr, QW3_LOG_TIMING,
            "qw3: Metal argmax timing: "
            "prefill=%d tokens  %.1f ms  (%.2f tok/s)  |  "
            "generation=%d tokens  %.1f ms  (%.2f tok/s)  |  "
            "total=%.1f ms\n",
            prompt->len, dt_prefill * 1000.0, prefill_tps,
            n_generated, dt_gen * 1000.0, gen_tps,
            dt_total * 1000.0);

    qw3_session_free(s);
    return rc;
#endif
}

int qw3_engine_metal_generate_sample(qw3_engine *e, const qw3_tokens *prompt,
                                     int n_predict, int ctx_size,
                                     float temperature, int top_k,
                                     float top_p, float min_p,
                                     uint64_t *rng,
                                     qw3_token_emit_fn emit,
                                     qw3_generation_done_fn done,
                                     void *emit_ud) {
    if (!e || !prompt || n_predict < 0 || ctx_size <= 0) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)prompt; (void)n_predict; (void)ctx_size;
    (void)temperature; (void)top_k; (void)top_p; (void)min_p; (void)rng;
    (void)emit; (void)done; (void)emit_ud;
    return -1;
#else
    if (temperature <= 0.0f) {
        return qw3_engine_metal_generate_argmax(e, prompt, n_predict, ctx_size,
                                                emit, done, emit_ud);
    }
    if (!rng) return -1;
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) return -1;

    qw3_session *s = NULL;
    char err[256] = {0};
    const int eos = qw3_token_eos(e);
    int rc = 0;
    if (qw3_session_create(&s, e, ctx_size) != 0) return -1;

    /* --- Prefill phase --- */
    const double t_prefill_start = qw3_now_sec();
    const int prefill_batch = qw3_session_uses_partial_metal(s) ?
        1 : qw3_metal_prefill_batch_size();
    if (prefill_batch > 1) {
        for (int i = 0; i < prompt->len;) {
            int n = prompt->len - i;
            if (n > prefill_batch) n = prefill_batch;
            const int last = i + n == prompt->len;
            int prc = 0;
            if (n > 1) {
                prc = qw3_metal_session_eval_prefill_batch_mode(
                    s, prompt->v + i, n, err, sizeof(err),
                    last ? QW3_METAL_LOGITS_READ : QW3_METAL_LOGITS_DEFER);
            } else {
                prc = last ?
                    qw3_metal_session_eval_token_slow_ex(
                        s, prompt->v[i], err, sizeof(err), 1) :
                    qw3_metal_session_eval_token_defer_logits(
                        s, prompt->v[i], err, sizeof(err));
            }
            if (prc != 0) {
                fprintf(stderr, "qw3: Metal session prefill failed: %s\n", err);
                qw3_session_free(s);
                return -1;
            }
            i += n;
        }
    } else {
        const int defer_interval = qw3_prefill_defer_interval();
        int deferred = 0;
        for (int i = 0; i < prompt->len; i++) {
            const int last = i + 1 == prompt->len;
            int prc = last ?
                qw3_metal_session_eval_token_slow_ex(s, prompt->v[i],
                                                     err, sizeof(err), 1) :
                qw3_metal_session_eval_token_defer_logits(s, prompt->v[i],
                                                          err, sizeof(err));
            if (prc != 0) {
                fprintf(stderr, "qw3: Metal session prefill failed: %s\n", err);
                qw3_session_free(s);
                return -1;
            }
            if (!last && ++deferred >= defer_interval) {
                if (!qw3_metal_synchronize()) {
                    fprintf(stderr, "qw3: Metal deferred prefill failed\n");
                    qw3_session_free(s);
                    return -1;
                }
                deferred = 0;
            }
        }
    }
    const double t_prefill_end = qw3_now_sec();

    /* --- Generation phase --- */
    const double t_gen_start = qw3_now_sec();
    int n_generated = 0;
    for (int step = 0; step < n_predict; step++) {
        if (s->kv.pos >= ctx_size) {
            fprintf(stderr, "qw3: Metal generation context full at step %d len=%d ctx=%d\n",
                    step, s->kv.pos, ctx_size);
            rc = -1;
            break;
        }

        int token = qw3_session_sample(s, temperature, top_k, top_p, min_p, rng);
        if (token < 0) {
            rc = -1;
            break;
        }
        if (token == eos) break;
        n_generated++;
        if (emit) emit(emit_ud, token);
        if (qw3_metal_session_eval_token_slow(s, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "qw3: Metal session decode failed: %s\n", err);
            rc = -1;
            break;
        }
    }
    const double t_gen_end = qw3_now_sec();

    if (done) done(emit_ud);

    /* --- Timing summary --- */
    const double dt_prefill = t_prefill_end - t_prefill_start;
    const double dt_gen = t_gen_end - t_gen_start;
    const double dt_total = t_gen_end - t_prefill_start;
    const double prefill_tps = (dt_prefill > 0.0) ? (double)prompt->len / dt_prefill : 0.0;
    const double gen_tps = (dt_gen > 0.0) ? (double)n_generated / dt_gen : 0.0;
    qw3_log(stderr, QW3_LOG_TIMING,
            "qw3: Metal sample timing: "
            "prefill=%d tokens  %.1f ms  (%.2f tok/s)  |  "
            "generation=%d tokens  %.1f ms  (%.2f tok/s)  |  "
            "total=%.1f ms\n",
            prompt->len, dt_prefill * 1000.0, prefill_tps,
            n_generated, dt_gen * 1000.0, gen_tps,
            dt_total * 1000.0);

    qw3_session_free(s);
    return rc;
#endif
}

int qw3_engine_metal_gqa_project_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal gqa project: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal gqa project: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal gqa project: token %d is outside vocab\n", token);
        return -1;
    }
    const int il = 3;
    const int pos = 1;
    const qw3_layer_weights *lw = &e->weights.layer[il];
    if (!qw3_layer_is_full_attention((uint32_t)il) ||
        lw->attn_q_proj->type != QW3_TENSOR_Q8_0 ||
        lw->attn_k_proj->type != QW3_TENSOR_Q8_0 ||
        lw->attn_v_proj->type != QW3_TENSOR_Q8_0) {
        fprintf(fp, "metal gqa project: layer3 expected full-attn q/k/v q8_0 tensors\n");
        return -1;
    }

    const uint32_t qg_n = (uint32_t)tensor_cols_qg();
    const uint32_t q_n = QW3_N_HEAD * QW3_N_HEAD_DIM;
    const uint32_t kv_n = (uint32_t)tensor_cols_kv();
    float *x_cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *q_cpu = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *k_cpu = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *v_cpu = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *gate_cpu = qw3_xmalloc((size_t)q_n * sizeof(float));

    float *x_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qfull_gpu = qw3_xmalloc((size_t)qg_n * sizeof(float));
    float *qnorm_gpu = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *q_gpu = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *kproj_gpu = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *knorm_gpu = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *k_gpu = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *v_gpu = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *gate_gpu = qw3_xmalloc((size_t)q_n * sizeof(float));

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd,
                                    (uint64_t)token, x_cpu) &&
              cpu_gqa_project_token(e, il, pos, x_cpu,
                                    q_cpu, k_cpu, v_cpu, gate_cpu);
    int gpu_ok = ok &&
        qw3_metal_embed_q8_0(e->weights.token_embd->offset, (uint32_t)token,
                             QW3_N_EMBD, x_gpu) &&
        qw3_metal_rmsnorm_weight_f32(x_gpu, lw->attn_norm->offset,
                                     xn_gpu, QW3_N_EMBD, QW3_RMS_EPS) &&
        qw3_metal_matvec_q8_0(lw->attn_q_proj->offset, xn_gpu,
                              QW3_N_EMBD, qg_n, qfull_gpu) &&
        qw3_metal_matvec_q8_0(lw->attn_k_proj->offset, xn_gpu,
                              QW3_N_EMBD, kv_n, kproj_gpu) &&
        qw3_metal_matvec_q8_0(lw->attn_v_proj->offset, xn_gpu,
                              QW3_N_EMBD, kv_n, v_gpu);
    if (gpu_ok) {
        for (int h = 0; h < QW3_N_HEAD; h++) {
            memcpy(gate_gpu + h * QW3_N_HEAD_DIM,
                   qfull_gpu + h * QW3_N_HEAD_DIM * 2 + QW3_N_HEAD_DIM,
                   (size_t)QW3_N_HEAD_DIM * sizeof(float));
            gpu_ok = qw3_metal_rmsnorm_weight_f32(
                qfull_gpu + h * QW3_N_HEAD_DIM * 2,
                lw->attn_q_norm->offset,
                qnorm_gpu + h * QW3_N_HEAD_DIM,
                QW3_N_HEAD_DIM, QW3_RMS_EPS);
            if (!gpu_ok) break;
        }
    }
    if (gpu_ok) {
        for (int h = 0; h < QW3_N_HEAD_KV; h++) {
            gpu_ok = qw3_metal_rmsnorm_weight_f32(
                kproj_gpu + h * QW3_N_HEAD_DIM,
                lw->attn_k_norm->offset,
                knorm_gpu + h * QW3_N_HEAD_DIM,
                QW3_N_HEAD_DIM, QW3_RMS_EPS);
            if (!gpu_ok) break;
        }
    }
    if (gpu_ok) {
        gpu_ok =
            qw3_metal_rope_heads(qnorm_gpu, QW3_N_HEAD, QW3_N_HEAD_DIM,
                                 QW3_ROPE_DIM, pos, QW3_ROPE_THETA, q_gpu) &&
            qw3_metal_rope_heads(knorm_gpu, QW3_N_HEAD_KV, QW3_N_HEAD_DIM,
                                 QW3_ROPE_DIM, pos, QW3_ROPE_THETA, k_gpu);
    }

    float q_max = 0.0f, k_max = 0.0f, v_max = 0.0f, gate_max = 0.0f;
    double q_rms = 0.0, k_rms = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < q_n; i++) {
            float dq = fabsf(q_cpu[i] - q_gpu[i]);
            float dg = fabsf(gate_cpu[i] - gate_gpu[i]);
            if (dq > q_max) q_max = dq;
            if (dg > gate_max) gate_max = dg;
            q_rms += (double)q_gpu[i] * q_gpu[i];
        }
        for (uint32_t i = 0; i < kv_n; i++) {
            float dk = fabsf(k_cpu[i] - k_gpu[i]);
            float dv = fabsf(v_cpu[i] - v_gpu[i]);
            if (dk > k_max) k_max = dk;
            if (dv > v_max) v_max = dv;
            k_rms += (double)k_gpu[i] * k_gpu[i];
        }
        q_rms = sqrt(q_rms / q_n);
        k_rms = sqrt(k_rms / kv_n);
    }
    fprintf(fp,
            "metal gqa project: %s token=%d layer=3 pos=%d q_max=%.7g k_max=%.7g v_max=%.7g gate_max=%.7g q_rms=%.7g k_rms=%.7g q0=[%.7g %.7g %.7g %.7g] k0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, pos,
            q_max, k_max, v_max, gate_max, q_rms, k_rms,
            q_gpu[0], q_gpu[1], q_gpu[2], q_gpu[3],
            k_gpu[0], k_gpu[1], k_gpu[2], k_gpu[3]);

    free(gate_gpu);
    free(v_gpu);
    free(k_gpu);
    free(knorm_gpu);
    free(kproj_gpu);
    free(q_gpu);
    free(qnorm_gpu);
    free(qfull_gpu);
    free(xn_gpu);
    free(x_gpu);
    free(gate_cpu);
    free(v_cpu);
    free(k_cpu);
    free(q_cpu);
    free(x_cpu);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_gqa_single_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal gqa single: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal gqa single: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal gqa single: token %d is outside vocab\n", token);
        return -1;
    }
    const int il = 3;
    const int pos = 1;
    const qw3_layer_weights *lw = &e->weights.layer[il];
    if (!qw3_layer_is_full_attention((uint32_t)il) ||
        lw->attn_q_proj->type != QW3_TENSOR_Q8_0 ||
        lw->attn_v_proj->type != QW3_TENSOR_Q8_0 ||
        lw->attn_o_proj->type != QW3_TENSOR_Q8_0) {
        fprintf(fp, "metal gqa single: layer3 expected full-attn q/v/o q8_0 tensors\n");
        return -1;
    }

    const uint32_t qg_n = (uint32_t)tensor_cols_qg();
    const uint32_t q_n = QW3_N_HEAD * QW3_N_HEAD_DIM;
    const uint32_t kv_n = (uint32_t)tensor_cols_kv();
    const uint32_t inner_n = (uint32_t)tensor_linear_inner();
    float *x_cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *q_cpu = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *k_cpu = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *v_cpu = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *gate_cpu = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *inner_cpu = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *out_cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));

    float *x_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qfull_gpu = qw3_xmalloc((size_t)qg_n * sizeof(float));
    float *gate_gpu = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *v_gpu = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *inner_gpu = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *out_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd,
                                    (uint64_t)token, x_cpu) &&
              cpu_gqa_project_token(e, il, pos, x_cpu,
                                    q_cpu, k_cpu, v_cpu, gate_cpu) &&
              cpu_gqa_single_token_layer(e, il, pos, x_cpu, out_cpu);
    if (ok) {
        cpu_gqa_attend_inner(q_cpu, gate_cpu, k_cpu, v_cpu, 1, inner_cpu);
    }

    int gpu_ok = ok &&
        qw3_metal_embed_q8_0(e->weights.token_embd->offset, (uint32_t)token,
                             QW3_N_EMBD, x_gpu) &&
        qw3_metal_rmsnorm_weight_f32(x_gpu, lw->attn_norm->offset,
                                     xn_gpu, QW3_N_EMBD, QW3_RMS_EPS) &&
        qw3_metal_matvec_q8_0(lw->attn_q_proj->offset, xn_gpu,
                              QW3_N_EMBD, qg_n, qfull_gpu) &&
        qw3_metal_matvec_q8_0(lw->attn_v_proj->offset, xn_gpu,
                              QW3_N_EMBD, kv_n, v_gpu);
    if (gpu_ok) {
        for (int h = 0; h < QW3_N_HEAD; h++) {
            memcpy(gate_gpu + h * QW3_N_HEAD_DIM,
                   qfull_gpu + h * QW3_N_HEAD_DIM * 2 + QW3_N_HEAD_DIM,
                   (size_t)QW3_N_HEAD_DIM * sizeof(float));
        }
        gpu_ok =
            qw3_metal_gqa_single_token_inner(gate_gpu, v_gpu,
                                             QW3_N_HEAD, QW3_N_HEAD_KV,
                                             QW3_N_HEAD_DIM, inner_gpu) &&
            qw3_metal_matvec_q8_0(lw->attn_o_proj->offset, inner_gpu,
                                  inner_n, QW3_N_EMBD, out_gpu);
    }

    float inner_max = 0.0f, out_max = 0.0f;
    double inner_rms = 0.0, out_rms = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < inner_n; i++) {
            float d = fabsf(inner_cpu[i] - inner_gpu[i]);
            if (d > inner_max) inner_max = d;
            inner_rms += (double)d * d;
        }
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(out_cpu[i] - out_gpu[i]);
            if (d > out_max) out_max = d;
            out_rms += (double)d * d;
        }
        inner_rms = sqrt(inner_rms / inner_n);
        out_rms = sqrt(out_rms / QW3_N_EMBD);
    }
    fprintf(fp,
            "metal gqa single: %s token=%d layer=3 pos=%d inner_max=%.7g inner_rms=%.7g out_max=%.7g out_rms=%.7g out0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, pos,
            inner_max, inner_rms, out_max, out_rms,
            out_gpu[0], out_gpu[1], out_gpu[2], out_gpu[3]);

    free(out_gpu);
    free(inner_gpu);
    free(v_gpu);
    free(gate_gpu);
    free(qfull_gpu);
    free(xn_gpu);
    free(x_gpu);
    free(out_cpu);
    free(inner_cpu);
    free(gate_cpu);
    free(v_cpu);
    free(k_cpu);
    free(q_cpu);
    free(x_cpu);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_gqa_attend2_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal gqa attend2: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal gqa attend2: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token + 1 >= QW3_N_VOCAB) {
        fprintf(fp, "metal gqa attend2: token %d cannot form token/token+1 pair\n", token);
        return -1;
    }
    const int il = 3;
    const uint32_t q_n = QW3_N_HEAD * QW3_N_HEAD_DIM;
    const uint32_t kv_n = (uint32_t)tensor_cols_kv();
    float *x0 = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *x1 = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *q0 = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *q1 = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *gate0 = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *gate1 = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *k0 = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *k1 = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *v0 = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *v1 = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *k_cache = qw3_xmalloc((size_t)2 * kv_n * sizeof(float));
    float *v_cache = qw3_xmalloc((size_t)2 * kv_n * sizeof(float));
    float *cpu_inner = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *gpu_inner = qw3_xmalloc((size_t)q_n * sizeof(float));

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd,
                                    (uint64_t)token, x0) &&
              tensor_read_dense_row(&e->model, e->weights.token_embd,
                                    (uint64_t)(token + 1), x1) &&
              cpu_gqa_project_token(e, il, 0, x0, q0, k0, v0, gate0) &&
              cpu_gqa_project_token(e, il, 1, x1, q1, k1, v1, gate1);
    if (ok) {
        memcpy(k_cache, k0, (size_t)kv_n * sizeof(float));
        memcpy(k_cache + kv_n, k1, (size_t)kv_n * sizeof(float));
        memcpy(v_cache, v0, (size_t)kv_n * sizeof(float));
        memcpy(v_cache + kv_n, v1, (size_t)kv_n * sizeof(float));
        cpu_gqa_attend_inner(q1, gate1, k_cache, v_cache, 2, cpu_inner);
    }
    int gpu_ok = ok &&
        qw3_metal_gqa_attend2_inner(q1, gate1, k_cache, v_cache,
                                    QW3_N_HEAD, QW3_N_HEAD_KV,
                                    QW3_N_HEAD_DIM, gpu_inner);

    float maxdiff = 0.0f;
    double rmsdiff = 0.0, out_rms = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < q_n; i++) {
            float d = fabsf(cpu_inner[i] - gpu_inner[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * d;
            out_rms += (double)gpu_inner[i] * gpu_inner[i];
        }
        rmsdiff = sqrt(rmsdiff / q_n);
        out_rms = sqrt(out_rms / q_n);
    }
    fprintf(fp,
            "metal gqa attend2: %s token=%d,%d layer=3 maxdiff=%.7g rmsdiff=%.7g out_rms=%.7g out0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, token + 1,
            maxdiff, rmsdiff, out_rms,
            gpu_inner[0], gpu_inner[1], gpu_inner[2], gpu_inner[3]);

    free(gpu_inner);
    free(cpu_inner);
    free(v_cache);
    free(k_cache);
    free(v1);
    free(v0);
    free(k1);
    free(k0);
    free(gate1);
    free(gate0);
    free(q1);
    free(q0);
    free(x1);
    free(x0);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_gqa_attend4_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal gqa attend4: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    const int n_ctx = 4;
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal gqa attend4: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token + n_ctx - 1 >= QW3_N_VOCAB) {
        fprintf(fp, "metal gqa attend4: token %d cannot form a 4-token run\n", token);
        return -1;
    }
    const int il = 3;
    const uint32_t q_n = QW3_N_HEAD * QW3_N_HEAD_DIM;
    const uint32_t kv_n = (uint32_t)tensor_cols_kv();
    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *q = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *gate = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *k = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *v = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *k_cache = qw3_xmalloc((size_t)n_ctx * kv_n * sizeof(float));
    float *v_cache = qw3_xmalloc((size_t)n_ctx * kv_n * sizeof(float));
    float *cpu_inner = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *gpu_inner = qw3_xmalloc((size_t)q_n * sizeof(float));

    bool ok = true;
    for (int t = 0; ok && t < n_ctx; t++) {
        ok = tensor_read_dense_row(&e->model, e->weights.token_embd,
                                   (uint64_t)(token + t), x) &&
             cpu_gqa_project_token(e, il, t, x, q, k, v, gate);
        if (ok) {
            memcpy(k_cache + (size_t)t * kv_n, k, (size_t)kv_n * sizeof(float));
            memcpy(v_cache + (size_t)t * kv_n, v, (size_t)kv_n * sizeof(float));
        }
    }
    if (ok) {
        cpu_gqa_attend_inner(q, gate, k_cache, v_cache, n_ctx, cpu_inner);
    }
    int gpu_ok = ok &&
        qw3_metal_gqa_attend_n_inner(q, gate, k_cache, v_cache,
                                     n_ctx, QW3_N_HEAD, QW3_N_HEAD_KV,
                                     QW3_N_HEAD_DIM, gpu_inner);

    float maxdiff = 0.0f;
    double rmsdiff = 0.0, out_rms = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < q_n; i++) {
            float d = fabsf(cpu_inner[i] - gpu_inner[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * d;
            out_rms += (double)gpu_inner[i] * gpu_inner[i];
        }
        rmsdiff = sqrt(rmsdiff / q_n);
        out_rms = sqrt(out_rms / q_n);
    }
    fprintf(fp,
            "metal gqa attend4: %s token=%d..%d layer=3 maxdiff=%.7g rmsdiff=%.7g out_rms=%.7g out0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, token + n_ctx - 1,
            maxdiff, rmsdiff, out_rms,
            gpu_inner[0], gpu_inner[1], gpu_inner[2], gpu_inner[3]);

    free(gpu_inner);
    free(cpu_inner);
    free(v_cache);
    free(k_cache);
    free(v);
    free(k);
    free(gate);
    free(q);
    free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_gqa_branch4_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal gqa branch4: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    const int n_ctx = 4;
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal gqa branch4: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token + n_ctx - 1 >= QW3_N_VOCAB) {
        fprintf(fp, "metal gqa branch4: token %d cannot form a 4-token run\n", token);
        return -1;
    }
    const int il = 3;
    const qw3_layer_weights *lw = &e->weights.layer[il];
    const uint32_t qg_n = (uint32_t)tensor_cols_qg();
    const uint32_t q_n = QW3_N_HEAD * QW3_N_HEAD_DIM;
    const uint32_t kv_n = (uint32_t)tensor_cols_kv();
    const uint32_t inner_n = (uint32_t)tensor_linear_inner();

    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *q = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *gate = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *k = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *v = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *cpu_k_cache = qw3_xmalloc((size_t)n_ctx * kv_n * sizeof(float));
    float *cpu_v_cache = qw3_xmalloc((size_t)n_ctx * kv_n * sizeof(float));
    float *cpu_inner = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *cpu_attn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));

    float *xg = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qfull = qw3_xmalloc((size_t)qg_n * sizeof(float));
    float *qnorm = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *qg = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *kg_proj = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *knorm = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *kg = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *vg = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *ggate = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *gpu_k_cache = qw3_xmalloc((size_t)n_ctx * kv_n * sizeof(float));
    float *gpu_v_cache = qw3_xmalloc((size_t)n_ctx * kv_n * sizeof(float));
    float *gpu_inner = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *gpu_attn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));

    bool ok = true;
    for (int t = 0; ok && t < n_ctx; t++) {
        ok = tensor_read_dense_row(&e->model, e->weights.token_embd,
                                   (uint64_t)(token + t), x) &&
             cpu_gqa_project_token(e, il, t, x, q, k, v, gate);
        if (ok) {
            memcpy(cpu_k_cache + (size_t)t * kv_n, k, (size_t)kv_n * sizeof(float));
            memcpy(cpu_v_cache + (size_t)t * kv_n, v, (size_t)kv_n * sizeof(float));
        }
    }
    if (ok) {
        cpu_gqa_attend_inner(q, gate, cpu_k_cache, cpu_v_cache, n_ctx, cpu_inner);
        ok = cpu_matvec(&e->model, lw->attn_o_proj, cpu_inner, cpu_attn);
    }

    int gpu_ok = ok;
    for (int t = 0; gpu_ok && t < n_ctx; t++) {
        gpu_ok =
            qw3_metal_embed_q8_0(e->weights.token_embd->offset,
                                 (uint32_t)(token + t), QW3_N_EMBD, xg) &&
            qw3_metal_rmsnorm_weight_f32(xg, lw->attn_norm->offset,
                                         xn, QW3_N_EMBD, QW3_RMS_EPS) &&
            qw3_metal_matvec_q8_0(lw->attn_q_proj->offset, xn,
                                  QW3_N_EMBD, qg_n, qfull) &&
            qw3_metal_matvec_q8_0(lw->attn_k_proj->offset, xn,
                                  QW3_N_EMBD, kv_n, kg_proj) &&
            qw3_metal_matvec_q8_0(lw->attn_v_proj->offset, xn,
                                  QW3_N_EMBD, kv_n, vg);
        if (gpu_ok) {
            for (int h = 0; h < QW3_N_HEAD; h++) {
                memcpy(ggate + h * QW3_N_HEAD_DIM,
                       qfull + h * QW3_N_HEAD_DIM * 2 + QW3_N_HEAD_DIM,
                       (size_t)QW3_N_HEAD_DIM * sizeof(float));
                gpu_ok = qw3_metal_rmsnorm_weight_f32(
                    qfull + h * QW3_N_HEAD_DIM * 2,
                    lw->attn_q_norm->offset,
                    qnorm + h * QW3_N_HEAD_DIM,
                    QW3_N_HEAD_DIM, QW3_RMS_EPS);
                if (!gpu_ok) break;
            }
        }
        for (int h = 0; gpu_ok && h < QW3_N_HEAD_KV; h++) {
            gpu_ok = qw3_metal_rmsnorm_weight_f32(
                kg_proj + h * QW3_N_HEAD_DIM,
                lw->attn_k_norm->offset,
                knorm + h * QW3_N_HEAD_DIM,
                QW3_N_HEAD_DIM, QW3_RMS_EPS);
        }
        if (gpu_ok) {
            gpu_ok =
                qw3_metal_rope_heads(qnorm, QW3_N_HEAD, QW3_N_HEAD_DIM,
                                     QW3_ROPE_DIM, t, QW3_ROPE_THETA, qg) &&
                qw3_metal_rope_heads(knorm, QW3_N_HEAD_KV, QW3_N_HEAD_DIM,
                                     QW3_ROPE_DIM, t, QW3_ROPE_THETA, kg);
        }
        if (gpu_ok) {
            memcpy(gpu_k_cache + (size_t)t * kv_n, kg, (size_t)kv_n * sizeof(float));
            memcpy(gpu_v_cache + (size_t)t * kv_n, vg, (size_t)kv_n * sizeof(float));
        }
    }
    if (gpu_ok) {
        gpu_ok =
            qw3_metal_gqa_attend_n_inner(qg, ggate, gpu_k_cache, gpu_v_cache,
                                         n_ctx, QW3_N_HEAD, QW3_N_HEAD_KV,
                                         QW3_N_HEAD_DIM, gpu_inner) &&
            qw3_metal_matvec_q8_0(lw->attn_o_proj->offset, gpu_inner,
                                  inner_n, QW3_N_EMBD, gpu_attn);
    }

    float inner_max = 0.0f, attn_max = 0.0f;
    double attn_rmsdiff = 0.0, attn_rms = 0.0;
    if (gpu_ok) {
        for (uint32_t i = 0; i < inner_n; i++) {
            float d = fabsf(cpu_inner[i] - gpu_inner[i]);
            if (d > inner_max) inner_max = d;
        }
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu_attn[i] - gpu_attn[i]);
            if (d > attn_max) attn_max = d;
            attn_rmsdiff += (double)d * d;
            attn_rms += (double)gpu_attn[i] * gpu_attn[i];
        }
        attn_rmsdiff = sqrt(attn_rmsdiff / QW3_N_EMBD);
        attn_rms = sqrt(attn_rms / QW3_N_EMBD);
    }
    fprintf(fp,
            "metal gqa branch4: %s token=%d..%d layer=3 inner_max=%.7g attn_max=%.7g attn_rmsdiff=%.7g attn_rms=%.7g out0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, token + n_ctx - 1,
            inner_max, attn_max, attn_rmsdiff, attn_rms,
            gpu_attn[0], gpu_attn[1], gpu_attn[2], gpu_attn[3]);

    free(gpu_attn); free(gpu_inner); free(gpu_v_cache); free(gpu_k_cache);
    free(ggate); free(vg); free(kg); free(knorm); free(kg_proj); free(qg);
    free(qnorm); free(qfull); free(xn); free(xg); free(cpu_attn);
    free(cpu_inner); free(cpu_v_cache); free(cpu_k_cache); free(v); free(k);
    free(gate); free(q); free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_gqa_layer4_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal gqa layer4: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    const int n_ctx = 4;
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal gqa layer4: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token + n_ctx - 1 >= QW3_N_VOCAB) {
        fprintf(fp, "metal gqa layer4: token %d cannot form a 4-token run\n", token);
        return -1;
    }
    const int il = 3;
    const qw3_layer_weights *lw = &e->weights.layer[il];
    const uint32_t qg_n = (uint32_t)tensor_cols_qg();
    const uint32_t q_n = QW3_N_HEAD * QW3_N_HEAD_DIM;
    const uint32_t kv_n = (uint32_t)tensor_cols_kv();
    const uint32_t inner_n = (uint32_t)tensor_linear_inner();

    float *x = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *x_last = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *cpu_k_cache = qw3_xmalloc((size_t)n_ctx * kv_n * sizeof(float));
    float *cpu_v_cache = qw3_xmalloc((size_t)n_ctx * kv_n * sizeof(float));
    float *cpu_layer = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *tmp_layer = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *q = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *gate0 = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *k0 = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *v0 = qw3_xmalloc((size_t)kv_n * sizeof(float));

    float *xg = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xg_last = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qfull = qw3_xmalloc((size_t)qg_n * sizeof(float));
    float *qnorm = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *qg = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *kg_proj = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *knorm = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *kg = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *vg = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *ggate = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *gpu_k_cache = qw3_xmalloc((size_t)n_ctx * kv_n * sizeof(float));
    float *gpu_v_cache = qw3_xmalloc((size_t)n_ctx * kv_n * sizeof(float));
    float *inner = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *attn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *ffn = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *router = qw3_xmalloc((size_t)QW3_N_EXPERT * sizeof(float));
    float *egate = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *up = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *sparse = qw3_xcalloc((size_t)QW3_N_EMBD, sizeof(float));
    float *sh_gate = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *sh_up = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *sh_hidden = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *shared_down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *shared = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *moe = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *gpu_layer = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));

    bool ok = true;
    for (int t = 0; ok && t < n_ctx; t++) {
        ok = tensor_read_dense_row(&e->model, e->weights.token_embd,
                                   (uint64_t)(token + t), x);
        if (ok && t == n_ctx - 1) memcpy(x_last, x, (size_t)QW3_N_EMBD * sizeof(float));
        ok = ok && cpu_full_attention_layer(e, il, t, x, cpu_k_cache,
                                            cpu_v_cache, n_ctx, tmp_layer);
        if (ok && t == n_ctx - 1) {
            memcpy(cpu_layer, tmp_layer, (size_t)QW3_N_EMBD * sizeof(float));
        }
    }

    int gpu_ok = ok;
    for (int t = 0; gpu_ok && t < n_ctx; t++) {
        gpu_ok =
            qw3_metal_embed_q8_0(e->weights.token_embd->offset,
                                 (uint32_t)(token + t), QW3_N_EMBD, xg) &&
            qw3_metal_rmsnorm_weight_f32(xg, lw->attn_norm->offset,
                                         xn, QW3_N_EMBD, QW3_RMS_EPS) &&
            qw3_metal_matvec_q8_0(lw->attn_q_proj->offset, xn,
                                  QW3_N_EMBD, qg_n, qfull) &&
            qw3_metal_matvec_q8_0(lw->attn_k_proj->offset, xn,
                                  QW3_N_EMBD, kv_n, kg_proj) &&
            qw3_metal_matvec_q8_0(lw->attn_v_proj->offset, xn,
                                  QW3_N_EMBD, kv_n, vg);
        if (gpu_ok && t == n_ctx - 1) memcpy(xg_last, xg, (size_t)QW3_N_EMBD * sizeof(float));
        if (gpu_ok) {
            for (int h = 0; h < QW3_N_HEAD; h++) {
                memcpy(ggate + h * QW3_N_HEAD_DIM,
                       qfull + h * QW3_N_HEAD_DIM * 2 + QW3_N_HEAD_DIM,
                       (size_t)QW3_N_HEAD_DIM * sizeof(float));
                gpu_ok = qw3_metal_rmsnorm_weight_f32(
                    qfull + h * QW3_N_HEAD_DIM * 2, lw->attn_q_norm->offset,
                    qnorm + h * QW3_N_HEAD_DIM, QW3_N_HEAD_DIM, QW3_RMS_EPS);
                if (!gpu_ok) break;
            }
        }
        for (int h = 0; gpu_ok && h < QW3_N_HEAD_KV; h++) {
            gpu_ok = qw3_metal_rmsnorm_weight_f32(
                kg_proj + h * QW3_N_HEAD_DIM, lw->attn_k_norm->offset,
                knorm + h * QW3_N_HEAD_DIM, QW3_N_HEAD_DIM, QW3_RMS_EPS);
        }
        if (gpu_ok) {
            gpu_ok =
                qw3_metal_rope_heads(qnorm, QW3_N_HEAD, QW3_N_HEAD_DIM,
                                     QW3_ROPE_DIM, t, QW3_ROPE_THETA, qg) &&
                qw3_metal_rope_heads(knorm, QW3_N_HEAD_KV, QW3_N_HEAD_DIM,
                                     QW3_ROPE_DIM, t, QW3_ROPE_THETA, kg);
        }
        if (gpu_ok) {
            memcpy(gpu_k_cache + (size_t)t * kv_n, kg, (size_t)kv_n * sizeof(float));
            memcpy(gpu_v_cache + (size_t)t * kv_n, vg, (size_t)kv_n * sizeof(float));
        }
    }
    if (gpu_ok) {
        gpu_ok =
            qw3_metal_gqa_attend_n_inner(qg, ggate, gpu_k_cache, gpu_v_cache,
                                         n_ctx, QW3_N_HEAD, QW3_N_HEAD_KV,
                                         QW3_N_HEAD_DIM, inner) &&
            qw3_metal_matvec_q8_0(lw->attn_o_proj->offset, inner,
                                  inner_n, QW3_N_EMBD, attn) &&
            qw3_metal_residual_rmsnorm_weight_f32(xg_last, attn,
                                                  lw->ffn_norm->offset,
                                                  ffn, QW3_N_EMBD,
                                                  QW3_RMS_EPS) &&
            qw3_metal_matvec_f32(lw->ffn_gate_inp->offset, ffn,
                                 QW3_N_EMBD, QW3_N_EXPERT, router);
    }
    int ids[QW3_N_EXPERT_USED];
    float vals[QW3_N_EXPERT_USED], weights[QW3_N_EXPERT_USED];
    memset(ids, 0, sizeof(ids));
    if (gpu_ok) {
        topk_desc(router, QW3_N_EXPERT, QW3_N_EXPERT_USED, ids, vals);
        float wsum = 0.0f;
        for (int kk = 0; kk < QW3_N_EXPERT_USED; kk++) {
            weights[kk] = expf(vals[kk] - vals[0]);
            wsum += weights[kk];
        }
        for (int kk = 0; kk < QW3_N_EXPERT_USED; kk++) weights[kk] /= wsum;
    }
    for (int kk = 0; gpu_ok && kk < QW3_N_EXPERT_USED; kk++) {
        gpu_ok =
            qw3_metal_matvec_iq3_s_expert(lw->ffn_gate_exps->offset, (uint32_t)ids[kk],
                                          ffn, QW3_N_EMBD, QW3_N_FF_EXP, egate) &&
            qw3_metal_matvec_iq3_s_expert(lw->ffn_up_exps->offset, (uint32_t)ids[kk],
                                          ffn, QW3_N_EMBD, QW3_N_FF_EXP, up) &&
            qw3_metal_silu_mul(egate, up, QW3_N_FF_EXP, hidden) &&
            qw3_metal_matvec_iq4_xs_expert(lw->ffn_down_exps->offset, (uint32_t)ids[kk],
                                           hidden, QW3_N_FF_EXP, QW3_N_EMBD, down);
        if (gpu_ok) for (int i = 0; i < QW3_N_EMBD; i++) sparse[i] += weights[kk] * down[i];
    }
    gpu_ok = gpu_ok &&
        qw3_metal_matvec_q8_0(lw->ffn_gate_shared->offset, ffn,
                              QW3_N_EMBD, QW3_N_FF_SHARED, sh_gate) &&
        qw3_metal_matvec_q8_0(lw->ffn_up_shared->offset, ffn,
                              QW3_N_EMBD, QW3_N_FF_SHARED, sh_up) &&
        qw3_metal_silu_mul(sh_gate, sh_up, QW3_N_FF_SHARED, sh_hidden) &&
        qw3_metal_matvec_q8_0(lw->ffn_down_shared->offset, sh_hidden,
                              QW3_N_FF_SHARED, QW3_N_EMBD, shared_down);
    float shared_raw = 0.0f;
    if (gpu_ok) gpu_ok = qw3_metal_matvec_f32(lw->ffn_gate_inp_shexp->offset, ffn,
                                             QW3_N_EMBD, 1, &shared_raw);
    if (gpu_ok) gpu_ok = qw3_metal_scale(shared_down, QW3_N_EMBD,
                                         1.0f / (1.0f + expf(-shared_raw)), shared);
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            moe[i] = sparse[i] + shared[i];
            gpu_layer[i] = xg_last[i] + attn[i] + moe[i];
        }
    }

    float maxdiff = 0.0f;
    double rmsdiff = 0.0, out_rms = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float d = fabsf(cpu_layer[i] - gpu_layer[i]);
            if (d > maxdiff) maxdiff = d;
            rmsdiff += (double)d * d;
            out_rms += (double)gpu_layer[i] * gpu_layer[i];
        }
        rmsdiff = sqrt(rmsdiff / QW3_N_EMBD);
        out_rms = sqrt(out_rms / QW3_N_EMBD);
    }
    fprintf(fp,
            "metal gqa layer4: %s token=%d..%d layer=3 maxdiff=%.7g rmsdiff=%.7g out_rms=%.7g top0=%d out0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, token + n_ctx - 1,
            maxdiff, rmsdiff, out_rms, ids[0],
            gpu_layer[0], gpu_layer[1], gpu_layer[2], gpu_layer[3]);

    free(gpu_layer); free(moe); free(shared); free(shared_down); free(sh_hidden);
    free(sh_up); free(sh_gate); free(sparse); free(down); free(hidden); free(up);
    free(egate); free(router); free(ffn); free(attn); free(inner); free(gpu_v_cache);
    free(gpu_k_cache); free(ggate); free(vg); free(kg); free(knorm); free(kg_proj);
    free(qg); free(qnorm); free(qfull); free(xn); free(xg_last); free(xg);
    free(v0); free(k0); free(gate0); free(q); free(tmp_layer); free(cpu_layer);
    free(cpu_v_cache); free(cpu_k_cache); free(x_last); free(x);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_engine_metal_gqa_real_layer_test(qw3_engine *e, int token, FILE *fp) {
    if (!e || !fp) return -1;
#ifdef QW3_NO_METAL
    (void)e; (void)token;
    fprintf(fp, "metal gqa real layer: unavailable in QW3_NO_METAL build\n");
    return -1;
#else
    if (e->backend != QW3_BACKEND_METAL || !e->metal_ready) {
        fprintf(fp, "metal gqa real layer: Metal backend is not initialized\n");
        return -1;
    }
    if (token < 0 || token >= QW3_N_VOCAB) {
        fprintf(fp, "metal gqa real layer: token %d is outside vocab\n", token);
        return -1;
    }
    const int il = 3;
    const int pos = 1;
    const qw3_layer_weights *lw = &e->weights.layer[il];
    if (!qw3_layer_is_full_attention((uint32_t)il) ||
        lw->attn_q_proj->type != QW3_TENSOR_Q8_0 ||
        lw->attn_v_proj->type != QW3_TENSOR_Q8_0 ||
        lw->attn_o_proj->type != QW3_TENSOR_Q8_0 ||
        lw->ffn_gate_inp->type != QW3_TENSOR_F32 ||
        !lw->ffn_gate_inp_shexp ||
        lw->ffn_gate_exps->type != QW3_TENSOR_IQ3_S ||
        lw->ffn_up_exps->type != QW3_TENSOR_IQ3_S ||
        lw->ffn_down_exps->type != QW3_TENSOR_IQ4_XS ||
        lw->ffn_gate_shared->type != QW3_TENSOR_Q8_0 ||
        lw->ffn_up_shared->type != QW3_TENSOR_Q8_0 ||
        lw->ffn_down_shared->type != QW3_TENSOR_Q8_0) {
        fprintf(fp, "metal gqa real layer: layer3 tensor layout is not the expected q8/f32/iq3/iq4 mix\n");
        return -1;
    }

    const uint32_t qg_n = (uint32_t)tensor_cols_qg();
    const uint32_t q_n = QW3_N_HEAD * QW3_N_HEAD_DIM;
    const uint32_t kv_n = (uint32_t)tensor_cols_kv();
    const uint32_t inner_n = (uint32_t)tensor_linear_inner();
    float *x_cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *attn_cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *resid_cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *ffn_cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *moe_cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *layer_cpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    int cpu_ids[QW3_N_EXPERT_USED];
    float cpu_scores[QW3_N_EXPERT_USED];

    float *x_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *xn_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *qfull_gpu = qw3_xmalloc((size_t)qg_n * sizeof(float));
    float *gate_gpu = qw3_xmalloc((size_t)q_n * sizeof(float));
    float *v_gpu = qw3_xmalloc((size_t)kv_n * sizeof(float));
    float *inner_gpu = qw3_xmalloc((size_t)inner_n * sizeof(float));
    float *attn_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *ffn_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *router_gpu = qw3_xmalloc((size_t)QW3_N_EXPERT * sizeof(float));

    float *gate = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *up = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *hidden = qw3_xmalloc((size_t)QW3_N_FF_EXP * sizeof(float));
    float *down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *sparse_gpu = qw3_xcalloc((size_t)QW3_N_EMBD, sizeof(float));
    float *sh_gate = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *sh_up = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *sh_hidden = qw3_xmalloc((size_t)QW3_N_FF_SHARED * sizeof(float));
    float *shared_down = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *shared_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *moe_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));
    float *layer_gpu = qw3_xmalloc((size_t)QW3_N_EMBD * sizeof(float));

    bool ok = tensor_read_dense_row(&e->model, e->weights.token_embd,
                                    (uint64_t)token, x_cpu) &&
              cpu_gqa_single_token_layer(e, il, pos, x_cpu, attn_cpu);
    if (ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) resid_cpu[i] = x_cpu[i] + attn_cpu[i];
        cpu_rmsnorm(ffn_cpu, resid_cpu, &e->model, lw->ffn_norm, QW3_N_EMBD);
        ok = cpu_moe_layer(e, il, ffn_cpu, moe_cpu, cpu_ids, cpu_scores);
        if (ok) {
            for (int i = 0; i < QW3_N_EMBD; i++) {
                layer_cpu[i] = resid_cpu[i] + moe_cpu[i];
            }
        }
    }

    int gpu_ok = ok &&
        qw3_metal_embed_q8_0(e->weights.token_embd->offset, (uint32_t)token,
                             QW3_N_EMBD, x_gpu) &&
        qw3_metal_rmsnorm_weight_f32(x_gpu, lw->attn_norm->offset,
                                     xn_gpu, QW3_N_EMBD, QW3_RMS_EPS) &&
        qw3_metal_matvec_q8_0(lw->attn_q_proj->offset, xn_gpu,
                              QW3_N_EMBD, qg_n, qfull_gpu) &&
        qw3_metal_matvec_q8_0(lw->attn_v_proj->offset, xn_gpu,
                              QW3_N_EMBD, kv_n, v_gpu);
    if (gpu_ok) {
        for (int h = 0; h < QW3_N_HEAD; h++) {
            memcpy(gate_gpu + h * QW3_N_HEAD_DIM,
                   qfull_gpu + h * QW3_N_HEAD_DIM * 2 + QW3_N_HEAD_DIM,
                   (size_t)QW3_N_HEAD_DIM * sizeof(float));
        }
        gpu_ok =
            qw3_metal_gqa_single_token_inner(gate_gpu, v_gpu,
                                             QW3_N_HEAD, QW3_N_HEAD_KV,
                                             QW3_N_HEAD_DIM, inner_gpu) &&
            qw3_metal_matvec_q8_0(lw->attn_o_proj->offset, inner_gpu,
                                  inner_n, QW3_N_EMBD, attn_gpu) &&
            qw3_metal_residual_rmsnorm_weight_f32(x_gpu, attn_gpu,
                                                  lw->ffn_norm->offset,
                                                  ffn_gpu, QW3_N_EMBD,
                                                  QW3_RMS_EPS) &&
            qw3_metal_matvec_f32(lw->ffn_gate_inp->offset, ffn_gpu,
                                 QW3_N_EMBD, QW3_N_EXPERT, router_gpu);
    }

    int ids[QW3_N_EXPERT_USED];
    float vals[QW3_N_EXPERT_USED], weights[QW3_N_EXPERT_USED];
    memset(ids, 0, sizeof(ids));
    memset(vals, 0, sizeof(vals));
    memset(weights, 0, sizeof(weights));
    int top_match = 1;
    if (gpu_ok) {
        topk_desc(router_gpu, QW3_N_EXPERT, QW3_N_EXPERT_USED, ids, vals);
        float wsum = 0.0f;
        for (int k = 0; k < QW3_N_EXPERT_USED; k++) {
            if (ids[k] != cpu_ids[k]) top_match = 0;
            weights[k] = expf(vals[k] - vals[0]);
            wsum += weights[k];
        }
        for (int k = 0; k < QW3_N_EXPERT_USED; k++) weights[k] /= wsum;
    }
    for (int k = 0; gpu_ok && k < QW3_N_EXPERT_USED; k++) {
        gpu_ok =
            qw3_metal_matvec_iq3_s_expert(lw->ffn_gate_exps->offset, (uint32_t)ids[k],
                                          ffn_gpu, QW3_N_EMBD, QW3_N_FF_EXP, gate) &&
            qw3_metal_matvec_iq3_s_expert(lw->ffn_up_exps->offset, (uint32_t)ids[k],
                                          ffn_gpu, QW3_N_EMBD, QW3_N_FF_EXP, up) &&
            qw3_metal_silu_mul(gate, up, QW3_N_FF_EXP, hidden) &&
            qw3_metal_matvec_iq4_xs_expert(lw->ffn_down_exps->offset, (uint32_t)ids[k],
                                           hidden, QW3_N_FF_EXP, QW3_N_EMBD, down);
        if (gpu_ok) {
            for (int i = 0; i < QW3_N_EMBD; i++) sparse_gpu[i] += weights[k] * down[i];
        }
    }
    gpu_ok = gpu_ok &&
        qw3_metal_matvec_q8_0(lw->ffn_gate_shared->offset, ffn_gpu,
                              QW3_N_EMBD, QW3_N_FF_SHARED, sh_gate) &&
        qw3_metal_matvec_q8_0(lw->ffn_up_shared->offset, ffn_gpu,
                              QW3_N_EMBD, QW3_N_FF_SHARED, sh_up) &&
        qw3_metal_silu_mul(sh_gate, sh_up, QW3_N_FF_SHARED, sh_hidden) &&
        qw3_metal_matvec_q8_0(lw->ffn_down_shared->offset, sh_hidden,
                              QW3_N_FF_SHARED, QW3_N_EMBD, shared_down);
    float shared_raw = 0.0f;
    if (gpu_ok) {
        gpu_ok = qw3_metal_matvec_f32(lw->ffn_gate_inp_shexp->offset, ffn_gpu,
                                      QW3_N_EMBD, 1, &shared_raw);
    }
    if (gpu_ok) {
        const float shared_gate = 1.0f / (1.0f + expf(-shared_raw));
        gpu_ok = qw3_metal_scale(shared_down, QW3_N_EMBD, shared_gate, shared_gpu);
    }
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) moe_gpu[i] = sparse_gpu[i] + shared_gpu[i];
        for (int i = 0; i < QW3_N_EMBD; i++) layer_gpu[i] = x_gpu[i] + attn_gpu[i] + moe_gpu[i];
    }

    float attn_max = 0.0f, ffn_max = 0.0f, moe_max = 0.0f, layer_max = 0.0f;
    double layer_rmsdiff = 0.0, layer_rms = 0.0;
    if (gpu_ok) {
        for (int i = 0; i < QW3_N_EMBD; i++) {
            float da = fabsf(attn_cpu[i] - attn_gpu[i]);
            float df = fabsf(ffn_cpu[i] - ffn_gpu[i]);
            float dm = fabsf(moe_cpu[i] - moe_gpu[i]);
            float dl = fabsf(layer_cpu[i] - layer_gpu[i]);
            if (da > attn_max) attn_max = da;
            if (df > ffn_max) ffn_max = df;
            if (dm > moe_max) moe_max = dm;
            if (dl > layer_max) layer_max = dl;
            layer_rmsdiff += (double)dl * dl;
            layer_rms += (double)layer_gpu[i] * layer_gpu[i];
        }
        layer_rmsdiff = sqrt(layer_rmsdiff / QW3_N_EMBD);
        layer_rms = sqrt(layer_rms / QW3_N_EMBD);
    }
    fprintf(fp,
            "metal gqa real layer: %s token=%d top_match=%s experts=%d,%d,%d,%d,%d,%d,%d,%d attn_max=%.7g ffn_max=%.7g moe_max=%.7g layer_max=%.7g layer_rmsdiff=%.7g layer_rms=%.7g out0=[%.7g %.7g %.7g %.7g]\n",
            gpu_ok ? "ok" : "failed", token, top_match ? "yes" : "no",
            ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7],
            attn_max, ffn_max, moe_max, layer_max, layer_rmsdiff, layer_rms,
            layer_gpu[0], layer_gpu[1], layer_gpu[2], layer_gpu[3]);

    free(layer_gpu);
    free(moe_gpu);
    free(shared_gpu);
    free(shared_down);
    free(sh_hidden);
    free(sh_up);
    free(sh_gate);
    free(sparse_gpu);
    free(down);
    free(hidden);
    free(up);
    free(gate);
    free(router_gpu);
    free(ffn_gpu);
    free(attn_gpu);
    free(inner_gpu);
    free(v_gpu);
    free(gate_gpu);
    free(qfull_gpu);
    free(xn_gpu);
    free(x_gpu);
    free(layer_cpu);
    free(moe_cpu);
    free(ffn_cpu);
    free(resid_cpu);
    free(attn_cpu);
    free(x_cpu);
    return gpu_ok ? 0 : -1;
#endif
}

int qw3_session_argmax(qw3_session *s) {
    if (!s) return -1;
#ifndef QW3_NO_METAL
    if (s->engine && s->engine->backend == QW3_BACKEND_METAL && s->metal &&
        !qw3_session_uses_partial_metal(s)) {
        uint32_t idx = 0;
        float val = 0.0f;
        if (qw3_metal_session_argmax_logits(s->metal, QW3_N_VOCAB,
                                            &idx, &val)) {
            return (int)idx;
        }
        return -1;
    }
#endif
    int best = 0;
    float bestv = s->logits[0];
    for (int i = 1; i < QW3_N_VOCAB; i++) {
        if (s->logits[i] > bestv) {
            bestv = s->logits[i];
            best = i;
        }
    }
    return best;
}

typedef struct {
    int id;
    float logit;
    float prob;
} qw3_sample_item;

static int sample_item_cmp_desc(const void *a, const void *b) {
    const qw3_sample_item *ia = (const qw3_sample_item *)a;
    const qw3_sample_item *ib = (const qw3_sample_item *)b;
    return (ia->logit < ib->logit) - (ia->logit > ib->logit);
}

static uint64_t sample_rng_next(uint64_t *rng) {
    uint64_t x = *rng;
    if (!x) x = 0x9e3779b97f4a7c15ull;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *rng = x;
    return x * 2685821657736338717ull;
}

static float sample_rng_float(uint64_t *rng) {
    uint64_t x = sample_rng_next(rng);
    return (float)((x >> 11) * (1.0 / 9007199254740992.0));
}

static float sample_apply_repeat_penalty(float logit, float penalty) {
    return logit < 0.0f ? logit * penalty : logit / penalty;
}

static unsigned char *sample_recent_token_map(const int *recent_tokens,
                                              int n_recent_tokens,
                                              float repeat_penalty) {
    if (!recent_tokens || n_recent_tokens <= 0 || repeat_penalty <= 1.0f) {
        return NULL;
    }
    unsigned char *seen = calloc((size_t)QW3_N_VOCAB, 1);
    if (!seen) return NULL;
    for (int i = 0; i < n_recent_tokens; i++) {
        int tok = recent_tokens[i];
        if (tok >= 0 && tok < QW3_N_VOCAB) seen[tok] = 1;
    }
    return seen;
}

static int qw3_session_argmax_with_repetition(qw3_session *s,
                                              const unsigned char *seen,
                                              float repeat_penalty) {
    if (!seen || repeat_penalty <= 1.0f) return qw3_session_argmax(s);
    int best = 0;
    float bestv = seen[0] ? sample_apply_repeat_penalty(s->logits[0],
                                                        repeat_penalty)
                          : s->logits[0];
    for (int i = 1; i < QW3_N_VOCAB; i++) {
        float v = seen[i] ? sample_apply_repeat_penalty(s->logits[i],
                                                        repeat_penalty)
                          : s->logits[i];
        if (v > bestv) {
            bestv = v;
            best = i;
        }
    }
    return best;
}

int qw3_session_sample(qw3_session *s, float temperature, int top_k,
                       float top_p, float min_p, uint64_t *rng) {
    if (!s) return -1;
    if (temperature <= 0.0f || !rng) return qw3_session_argmax(s);
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k <= 0 || top_k > QW3_N_VOCAB) top_k = QW3_N_VOCAB;

    qw3_sample_item *items = qw3_xmalloc((size_t)QW3_N_VOCAB * sizeof(*items));
    for (int i = 0; i < QW3_N_VOCAB; i++) {
        items[i].id = i;
        items[i].logit = s->logits[i];
        items[i].prob = 0.0f;
    }
    qsort(items, QW3_N_VOCAB, sizeof(*items), sample_item_cmp_desc);

    int n = top_k;
    const float max_logit = items[0].logit;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        float p = expf((items[i].logit - max_logit) / temperature);
        items[i].prob = p;
        sum += p;
    }
    if (sum <= 0.0) {
        int id = items[0].id;
        free(items);
        return id;
    }
    for (int i = 0; i < n; i++) items[i].prob = (float)(items[i].prob / sum);

    if (min_p > 0.0f) {
        const float floor_p = items[0].prob * min_p;
        int kept = 1;
        while (kept < n && items[kept].prob >= floor_p) kept++;
        n = kept;
    }
    if (top_p < 1.0f) {
        float cdf = 0.0f;
        int kept = 0;
        while (kept < n) {
            cdf += items[kept].prob;
            kept++;
            if (cdf >= top_p) break;
        }
        if (kept > 0) n = kept;
    }

    sum = 0.0;
    for (int i = 0; i < n; i++) sum += items[i].prob;
    float r = sample_rng_float(rng) * (float)sum;
    int id = items[n - 1].id;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += items[i].prob;
        if (r <= cdf) {
            id = items[i].id;
            break;
        }
    }
    free(items);
    return id;
}

int qw3_session_sample_repetition(qw3_session *s, float temperature, int top_k,
                                  float top_p, float min_p, uint64_t *rng,
                                  const int *recent_tokens,
                                  int n_recent_tokens,
                                  float repeat_penalty) {
    if (!s) return -1;
    if (repeat_penalty <= 1.0f || !recent_tokens || n_recent_tokens <= 0) {
        return qw3_session_sample(s, temperature, top_k, top_p, min_p, rng);
    }
    unsigned char *seen = sample_recent_token_map(recent_tokens,
                                                  n_recent_tokens,
                                                  repeat_penalty);
    if (temperature <= 0.0f || !rng) {
        int id = qw3_session_argmax_with_repetition(s, seen, repeat_penalty);
        free(seen);
        return id;
    }
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k <= 0 || top_k > QW3_N_VOCAB) top_k = QW3_N_VOCAB;

    qw3_sample_item *items = qw3_xmalloc((size_t)QW3_N_VOCAB * sizeof(*items));
    for (int i = 0; i < QW3_N_VOCAB; i++) {
        items[i].id = i;
        items[i].logit = seen && seen[i] ?
            sample_apply_repeat_penalty(s->logits[i], repeat_penalty) :
            s->logits[i];
        items[i].prob = 0.0f;
    }
    free(seen);
    qsort(items, QW3_N_VOCAB, sizeof(*items), sample_item_cmp_desc);

    int n = top_k;
    const float max_logit = items[0].logit;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        float p = expf((items[i].logit - max_logit) / temperature);
        items[i].prob = p;
        sum += p;
    }
    if (sum <= 0.0) {
        int id = items[0].id;
        free(items);
        return id;
    }
    for (int i = 0; i < n; i++) items[i].prob = (float)(items[i].prob / sum);

    if (min_p > 0.0f) {
        const float floor_p = items[0].prob * min_p;
        int kept = 1;
        while (kept < n && items[kept].prob >= floor_p) kept++;
        n = kept;
    }
    if (top_p < 1.0f) {
        float cdf = 0.0f;
        int kept = 0;
        while (kept < n) {
            cdf += items[kept].prob;
            kept++;
            if (cdf >= top_p) break;
        }
        if (kept > 0) n = kept;
    }

    sum = 0.0;
    for (int i = 0; i < n; i++) sum += items[i].prob;
    float r = sample_rng_float(rng) * (float)sum;
    int id = items[n - 1].id;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += items[i].prob;
        if (r <= cdf) {
            id = items[i].id;
            break;
        }
    }
    free(items);
    return id;
}

int qw3_session_top_logprobs(qw3_session *s, qw3_token_score *out, int k) {
    if (!s || !out || k <= 0) return 0;
    int *ids = qw3_xmalloc((size_t)k * sizeof(int));
    float *vals = qw3_xmalloc((size_t)k * sizeof(float));
    topk_desc(s->logits, QW3_N_VOCAB, k, ids, vals);
    for (int i = 0; i < k; i++) {
        out[i].id = ids[i];
        out[i].logit = vals[i];
        out[i].logprob = 0.0f;
    }
    free(vals);
    free(ids);
    return k;
}

enum {
    QW3_SESSION_PAYLOAD_VERSION = 1,
};

#define QW3_SESSION_PAYLOAD_MAGIC 0x3357515345535349ull /* "ISSESQW3" */

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t header_size;
    uint64_t total_bytes;
    int32_t ctx_size;
    int32_t kv_pos;
    int32_t token_len;
    int32_t token_cap_reserved;
    uint64_t kv_floats;
    uint64_t dn_floats;
    uint64_t conv_floats;
    uint64_t logits_floats;
} qw3_session_payload_header;

static uint64_t session_kv_floats(const qw3_session *s) {
    return (uint64_t)QW3_N_FULL_ATTN_LAYERS * (uint64_t)s->ctx_size *
           QW3_N_HEAD_KV * QW3_N_HEAD_DIM;
}

static uint64_t session_dn_floats(void) {
    return (uint64_t)QW3_N_LINEAR_LAYERS * QW3_N_LINEAR_V_HEADS *
           QW3_N_LINEAR_HEAD_DIM * QW3_N_LINEAR_HEAD_DIM;
}

static uint64_t session_conv_floats(void) {
    return (uint64_t)QW3_N_LINEAR_LAYERS * tensor_linear_qkv() *
           (QW3_N_LINEAR_CONV_K - 1);
}

static int payload_write(FILE *fp, const void *p, uint64_t n,
                         char *err, size_t errlen) {
    if (n == 0) return 0;
    if (fwrite(p, 1, (size_t)n, fp) != (size_t)n) {
        if (err && errlen) snprintf(err, errlen, "payload write failed");
        return -1;
    }
    return 0;
}

static int payload_read(FILE *fp, void *p, uint64_t n,
                        char *err, size_t errlen) {
    if (n == 0) return 0;
    if (fread(p, 1, (size_t)n, fp) != (size_t)n) {
        if (err && errlen) snprintf(err, errlen, "payload read failed");
        return -1;
    }
    return 0;
}

uint64_t qw3_session_payload_bytes(qw3_session *s) {
    if (!s) return 0;
    return sizeof(qw3_session_payload_header) +
           (uint64_t)s->tokens.len * sizeof(int32_t) +
           2 * session_kv_floats(s) * sizeof(float) +
           session_dn_floats() * sizeof(float) +
           session_conv_floats() * sizeof(float) +
           (uint64_t)QW3_N_VOCAB * sizeof(float);
}

int qw3_session_save_payload(qw3_session *s, FILE *fp,
                             char *err, size_t errlen) {
    if (!s || !fp) return -1;
    qw3_session_payload_header h = {
        .magic = QW3_SESSION_PAYLOAD_MAGIC,
        .version = QW3_SESSION_PAYLOAD_VERSION,
        .header_size = sizeof(qw3_session_payload_header),
        .total_bytes = qw3_session_payload_bytes(s),
        .ctx_size = s->ctx_size,
        .kv_pos = s->kv.pos,
        .token_len = s->tokens.len,
        .token_cap_reserved = 0,
        .kv_floats = session_kv_floats(s),
        .dn_floats = session_dn_floats(),
        .conv_floats = session_conv_floats(),
        .logits_floats = QW3_N_VOCAB,
    };

    if (payload_write(fp, &h, sizeof(h), err, errlen) != 0) return -1;
    for (int i = 0; i < s->tokens.len; i++) {
        int32_t tok = s->tokens.v[i];
        if (payload_write(fp, &tok, sizeof(tok), err, errlen) != 0) return -1;
    }
    if (payload_write(fp, s->kv.k_cache, h.kv_floats * sizeof(float), err, errlen) != 0) return -1;
    if (payload_write(fp, s->kv.v_cache, h.kv_floats * sizeof(float), err, errlen) != 0) return -1;
    if (payload_write(fp, s->dn.state, h.dn_floats * sizeof(float), err, errlen) != 0) return -1;
    if (payload_write(fp, s->dn.conv_state, h.conv_floats * sizeof(float), err, errlen) != 0) return -1;
    if (payload_write(fp, s->logits, h.logits_floats * sizeof(float), err, errlen) != 0) return -1;
    return 0;
}

int qw3_session_load_payload(qw3_session *s, FILE *fp,
                             uint64_t payload_bytes,
                             char *err, size_t errlen) {
    if (!s || !fp) return -1;
    qw3_session_payload_header h;
    if (payload_read(fp, &h, sizeof(h), err, errlen) != 0) return -1;

    if (h.magic != QW3_SESSION_PAYLOAD_MAGIC ||
        h.version != QW3_SESSION_PAYLOAD_VERSION ||
        h.header_size != sizeof(qw3_session_payload_header)) {
        if (err && errlen) snprintf(err, errlen, "unsupported session payload");
        return -1;
    }
    if (payload_bytes && payload_bytes != h.total_bytes) {
        if (err && errlen) snprintf(err, errlen, "session payload size mismatch");
        return -1;
    }
    if (h.ctx_size != s->ctx_size ||
        h.kv_floats != session_kv_floats(s) ||
        h.dn_floats != session_dn_floats() ||
        h.conv_floats != session_conv_floats() ||
        h.logits_floats != QW3_N_VOCAB ||
        h.kv_pos < 0 || h.kv_pos > s->ctx_size ||
        h.token_len < 0 || h.token_len > s->ctx_size) {
        if (err && errlen) snprintf(err, errlen, "session payload shape mismatch");
        return -1;
    }

    qw3_tokens_free(&s->tokens);
    s->tokens.cap = h.token_len;
    s->tokens.len = h.token_len;
    s->tokens.v = h.token_len ? qw3_xmalloc((size_t)h.token_len * sizeof(int)) : NULL;
    for (int i = 0; i < h.token_len; i++) {
        int32_t tok = 0;
        if (payload_read(fp, &tok, sizeof(tok), err, errlen) != 0) return -1;
        s->tokens.v[i] = tok;
    }
    if (payload_read(fp, s->kv.k_cache, h.kv_floats * sizeof(float), err, errlen) != 0) return -1;
    if (payload_read(fp, s->kv.v_cache, h.kv_floats * sizeof(float), err, errlen) != 0) return -1;
    if (payload_read(fp, s->dn.state, h.dn_floats * sizeof(float), err, errlen) != 0) return -1;
    if (payload_read(fp, s->dn.conv_state, h.conv_floats * sizeof(float), err, errlen) != 0) return -1;
    if (payload_read(fp, s->logits, h.logits_floats * sizeof(float), err, errlen) != 0) return -1;
    s->kv.pos = h.kv_pos;
    s->valid = true;
    return 0;
}

int qw3_engine_generate_argmax(qw3_engine *e, const qw3_tokens *prompt,
                               int n_predict, int ctx_size,
                               qw3_token_emit_fn emit,
                               qw3_generation_done_fn done,
                               void *emit_ud,
                               qw3_session_progress_fn progress,
                               void *progress_ud) {
    if (!e || !prompt || prompt->len <= 0 || n_predict < 0) return -1;

    qw3_session *s = NULL;
    if (qw3_session_create(&s, e, ctx_size) != 0) return -1;
    qw3_session_set_progress(s, progress, progress_ud);

    char err[256];
    int rc = qw3_session_sync(s, prompt, err, sizeof(err));
    if (rc != 0) {
        fprintf(stderr, "qw3: generation prefill failed: %s\n", err);
        qw3_session_free(s);
        return -1;
    }

    const int eos = qw3_token_eos(e);
    for (int i = 0; i < n_predict; i++) {
        int token = qw3_session_argmax(s);
        if (token < 0) {
            rc = -1;
            break;
        }
        if (token == eos) break;
        if (emit) emit(emit_ud, token);
        rc = qw3_session_eval(s, token, err, sizeof(err));
        if (rc != 0) {
            fprintf(stderr, "qw3: generation step failed: %s\n", err);
            break;
        }
    }

    if (done) done(emit_ud);
    qw3_session_free(s);
    return rc;
}
