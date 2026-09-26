/* Differential dense-logits test against the pinned local llama.cpp build.
 *
 * The two runtimes are opened in sequence to keep peak memory low.  llama.cpp
 * produces the reference vector first on Metal; qw3 then evaluates the same
 * raw token sequence with an empty recurrent/KV state.  The Metal reference
 * intentionally avoids ggml's CPU Q8 activation conversion, matching qw3's
 * direct float-activation quantized matvec semantics.  QW3_LLAMA_NGL can
 * override the default of 99 for diagnostics.
 */
#include "../qw3.h"
#include "llama.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TEST_CTX = 128,
    TEST_THREADS = 16,
    TEST_TOP = 8,
    TEST_GREEDY = 8,
};

typedef struct {
    int id;
    float value;
} ranked_logit;

static void quiet_log(enum ggml_log_level level, const char *text,
                      void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

static bool trace_llama_tensor(struct ggml_tensor *tensor, bool ask,
                               void *user_data) {
    (void)user_data;
    const char *name = tensor->name;
    const bool wanted = getenv("QW3_DENSE_TRACE") &&
        (!strncmp(name, "model.input_embed", 17) ||
         !strncmp(name, "l_out-", 6) ||
         !strncmp(name, "result_norm", 11));
    if (ask) return wanted;
    if (!wanted || tensor->type != GGML_TYPE_F32) return true;

    const int64_t n = ggml_nelements(tensor);
    float *values = malloc((size_t)n * sizeof(*values));
    if (!values) abort();
    ggml_backend_tensor_get(tensor, values, 0, (size_t)n * sizeof(*values));
    double mean = 0.0;
    double ss = 0.0;
    for (int64_t i = 0; i < n; i++) {
        mean += values[i];
        ss += (double)values[i] * values[i];
    }
    printf("llama %-24s mean=%.9g rms=%.9g first=[%.9g %.9g %.9g %.9g]\n",
           name, mean / n, sqrt(ss / n), values[0], values[1],
           values[2], values[3]);
    free(values);
    return true;
}

static int ranked_logit_desc(const void *a, const void *b) {
    const ranked_logit *la = a;
    const ranked_logit *lb = b;
    if (la->value < lb->value) return 1;
    if (la->value > lb->value) return -1;
    return la->id - lb->id;
}

static void top_logits(const float *logits, int n_vocab,
                       ranked_logit out[TEST_TOP]) {
    ranked_logit *all = malloc((size_t)n_vocab * sizeof(*all));
    if (!all) abort();
    for (int i = 0; i < n_vocab; i++) {
        all[i].id = i;
        all[i].value = logits[i];
    }
    qsort(all, (size_t)n_vocab, sizeof(*all), ranked_logit_desc);
    for (int i = 0; i < TEST_TOP; i++) out[i] = all[i];
    free(all);
}

static int argmax(const float *logits, int n_vocab) {
    int best = 0;
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

static int load_llama_logits(const char *model_path, const llama_token *tokens,
                             int n_tokens, float **out, int *out_vocab,
                             int greedy[TEST_GREEDY]) {
    struct llama_model_params mp = llama_model_default_params();
    const char *ngl_env = getenv("QW3_LLAMA_NGL");
    mp.n_gpu_layers = ngl_env ? atoi(ngl_env) : 99;
    mp.use_extra_bufts = false;

    struct llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "test-dense-logits: llama.cpp model load failed\n");
        return 1;
    }
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = TEST_CTX;
    cp.n_batch = (uint32_t)n_tokens;
    cp.n_ubatch = (uint32_t)n_tokens;
    cp.n_threads = TEST_THREADS;
    cp.n_threads_batch = TEST_THREADS;
    cp.offload_kqv = mp.n_gpu_layers != 0;
    cp.cb_eval = trace_llama_tensor;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "test-dense-logits: llama.cpp context creation failed\n");
        llama_model_free(model);
        return 1;
    }

    struct llama_batch batch = llama_batch_get_one((llama_token *)tokens,
                                                    n_tokens);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "test-dense-logits: llama.cpp decode failed\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    float *logits = llama_get_logits_ith(ctx, -1);
    float *copy = malloc((size_t)n_vocab * sizeof(*copy));
    if (!logits || !copy) abort();
    for (int i = 0; i < n_vocab; i++) copy[i] = logits[i];

    for (int step = 0; step < TEST_GREEDY; step++) {
        logits = llama_get_logits_ith(ctx, -1);
        greedy[step] = argmax(logits, n_vocab);
        if (step + 1 < TEST_GREEDY) {
            llama_token next = (llama_token)greedy[step];
            batch = llama_batch_get_one(&next, 1);
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr,
                        "test-dense-logits: llama.cpp greedy decode failed "
                        "at step %d\n", step + 1);
                free(copy);
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    *out = copy;
    *out_vocab = n_vocab;
    return 0;
}

static int load_qw3_logits(const char *model_path, const int *tokens,
                           int n_tokens, float **out, int *out_vocab,
                           int greedy[TEST_GREEDY]) {
    const qw3_engine_options opt = {
        .model_path = model_path,
        .backend = QW3_BACKEND_CPU,
        .n_threads = TEST_THREADS,
        .warm_weights = false,
    };
    qw3_engine *engine = NULL;
    qw3_session *session = NULL;
    char err[256] = {0};
    if (qw3_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "test-dense-logits: qw3 model load failed\n");
        return 1;
    }
    const int n_vocab = qw3_vocab_size(engine);
    if (qw3_session_create(&session, engine, TEST_CTX) != 0) {
        fprintf(stderr, "test-dense-logits: qw3 session creation failed\n");
        qw3_engine_close(engine);
        return 1;
    }
    for (int i = 0; i < n_tokens; i++) {
        if (qw3_session_eval(session, tokens[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "test-dense-logits: qw3 eval failed: %s\n", err);
            qw3_session_free(session);
            qw3_engine_close(engine);
            return 1;
        }
    }
    if (getenv("QW3_DENSE_TRACE")) {
        qw3_tokens prompt = {
            .v = (int *)tokens,
            .len = n_tokens,
            .cap = n_tokens,
        };
        if (qw3_engine_trace_prompt(engine, &prompt, TEST_CTX, stdout) != 0) {
            fprintf(stderr, "test-dense-logits: qw3 trace failed\n");
        }
    }
    float *copy = malloc((size_t)n_vocab * sizeof(*copy));
    if (!copy) abort();
    if (qw3_session_copy_logits(session, copy, n_vocab) != n_vocab) {
        fprintf(stderr, "test-dense-logits: qw3 logits copy failed\n");
        free(copy);
        qw3_session_free(session);
        qw3_engine_close(engine);
        return 1;
    }
    for (int step = 0; step < TEST_GREEDY; step++) {
        greedy[step] = qw3_session_argmax(session);
        if (step + 1 < TEST_GREEDY &&
            qw3_session_eval(session, greedy[step], err, sizeof(err)) != 0) {
            fprintf(stderr,
                    "test-dense-logits: qw3 greedy eval failed at step %d: %s\n",
                    step + 1, err);
            free(copy);
            qw3_session_free(session);
            qw3_engine_close(engine);
            return 1;
        }
    }
    qw3_session_free(session);
    qw3_engine_close(engine);
    *out = copy;
    *out_vocab = n_vocab;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL TOKEN [TOKEN ...]\n", argv[0]);
        return 2;
    }
    const int n_tokens = argc - 2;
    int *tokens = malloc((size_t)n_tokens * sizeof(*tokens));
    llama_token *llama_tokens =
        malloc((size_t)n_tokens * sizeof(*llama_tokens));
    if (!tokens || !llama_tokens) abort();
    for (int i = 0; i < n_tokens; i++) {
        tokens[i] = atoi(argv[i + 2]);
        llama_tokens[i] = (llama_token)tokens[i];
    }

    llama_log_set(quiet_log, NULL);
    ggml_log_set(quiet_log, NULL);
    ggml_backend_load_all();
    float *reference = NULL;
    float *actual = NULL;
    int ref_vocab = 0;
    int actual_vocab = 0;
    int ref_greedy[TEST_GREEDY];
    int actual_greedy[TEST_GREEDY];
    if (load_llama_logits(argv[1], llama_tokens, n_tokens,
                          &reference, &ref_vocab, ref_greedy) != 0 ||
        load_qw3_logits(argv[1], tokens, n_tokens,
                        &actual, &actual_vocab, actual_greedy) != 0) {
        free(llama_tokens);
        free(tokens);
        free(reference);
        free(actual);
        return 1;
    }
    if (ref_vocab != actual_vocab) {
        fprintf(stderr, "test-dense-logits: vocab mismatch %d != %d\n",
                actual_vocab, ref_vocab);
        return 1;
    }

    double squared_error = 0.0;
    float max_abs = 0.0f;
    int max_id = -1;
    for (int i = 0; i < ref_vocab; i++) {
        const float diff = fabsf(actual[i] - reference[i]);
        if (!isfinite(actual[i]) || !isfinite(reference[i])) {
            fprintf(stderr, "test-dense-logits: non-finite logit at %d\n", i);
            return 1;
        }
        squared_error += (double)diff * diff;
        if (diff > max_abs) {
            max_abs = diff;
            max_id = i;
        }
    }
    ranked_logit ref_top[TEST_TOP];
    ranked_logit actual_top[TEST_TOP];
    top_logits(reference, ref_vocab, ref_top);
    top_logits(actual, actual_vocab, actual_top);
    int top_match = 0;
    while (top_match < TEST_TOP &&
           ref_top[top_match].id == actual_top[top_match].id) {
        top_match++;
    }
    int greedy_match = 0;
    while (greedy_match < TEST_GREEDY &&
           ref_greedy[greedy_match] == actual_greedy[greedy_match]) {
        greedy_match++;
    }
    printf("test-dense-logits: tokens=");
    for (int i = 0; i < n_tokens; i++)
        printf("%s%d", i ? "," : "", tokens[i]);
    printf(" top_match=%d/%d greedy_match=%d/%d rmse=%.9g "
           "max_abs=%.9g max_id=%d\n",
           top_match, TEST_TOP, greedy_match, TEST_GREEDY,
           sqrt(squared_error / ref_vocab), max_abs, max_id);
    printf("  greedy llama=");
    for (int i = 0; i < TEST_GREEDY; i++)
        printf("%s%d", i ? "," : "", ref_greedy[i]);
    printf(" qw3=");
    for (int i = 0; i < TEST_GREEDY; i++)
        printf("%s%d", i ? "," : "", actual_greedy[i]);
    putchar('\n');
    for (int i = 0; i < TEST_TOP; i++) {
        printf("  %d llama=%d:%.9g qw3=%d:%.9g\n", i + 1,
               ref_top[i].id, ref_top[i].value,
               actual_top[i].id, actual_top[i].value);
    }

    free(actual);
    free(reference);
    free(llama_tokens);
    free(tokens);
    const double rmse = sqrt(squared_error / ref_vocab);
    if (top_match != TEST_TOP || greedy_match != TEST_GREEDY ||
        rmse > 0.001 || max_abs > 0.005f) {
        fprintf(stderr,
                "test-dense-logits: FAIL: expected identical top-%d and "
                "%d greedy tokens, rmse <= 0.001, max_abs <= 0.005\n",
                TEST_TOP, TEST_GREEDY);
        return 1;
    }
    puts("test-dense-logits: ok");
    return 0;
}
