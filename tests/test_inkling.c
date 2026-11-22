#include "../qw3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int verify_logits(qw3_session *s, float *logits, int vocab, unsigned step) {
    // Unsloth llama.cpp b12588, the original UD-Q3_K_XL, serial Metal decode.
    static const int ids[3][5] = {
        {11, 156558, 0, 149971, 13},
        {0, 68014, 149881, 11, 13},
        {68014, 198, 149904, 220, 150078},
    };
    static const float reference[3][5] = {
        {27.187273f, 25.961412f, 25.767412f, 25.427490f, 25.339937f},
        {26.763622f, 25.899595f, 24.664217f, 24.593926f, 24.166052f},
        {26.582550f, 25.129131f, 24.956081f, 24.158016f, 23.856260f},
    };
    qw3_token_score top[5];
    if (qw3_session_copy_logits(s, logits, vocab) != vocab ||
        qw3_session_top_logprobs(s, top, 5) != 5 ||
        qw3_session_argmax(s) != ids[step][0]) return 1;
    for (int i = 0; i < vocab; i++) {
        if (!isfinite(logits[i])) return 1;
    }
    for (unsigned i = 0; i < 5; i++) {
        if (top[i].id != ids[step][i] || fabsf(top[i].logit - reference[step][i]) > 0.01f) {
            fprintf(stderr, "Inkling step %u rank %u: %d %.7f vs %d %.7f\n",
                    step, i, top[i].id, top[i].logit, ids[step][i], reference[step][i]);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    const char *backend_name = argv[2];
    qw3_backend backend;
    if (!strcmp(backend_name, "cpu")) backend = QW3_BACKEND_CPU;
    else if (!strcmp(backend_name, "metal")) backend = QW3_BACKEND_METAL;
    else return 2;
    qw3_engine_options opts = {.model_path = argv[1], .backend = backend,
                               .enable_inkling = true};
    qw3_engine *engine = NULL;
    qw3_session *session = NULL;
    qw3_tokens prompt = {0};
    float *logits = NULL, *before = NULL;
    char err[256] = {0};
    int status = 1;
    if (qw3_engine_open(&engine, &opts) != 0 ||
        !qw3_engine_is_inkling(engine) || qw3_engine_layer_count(engine) != 48 ||
        qw3_session_create(&session, engine, 16) != 0) goto done;
    const int vocab = qw3_vocab_size(engine);
    if (vocab != 200058) goto done;
    const qw3_context_memory memory = qw3_engine_context_memory_estimate(engine, 65536);
    const uint64_t kv_rows = 8 * 65536 + 40 * 1024;
    const uint64_t kv_bytes = kv_rows * 2 * 1024 * (backend == QW3_BACKEND_METAL ? 2 : 4);
    if (memory.gqa_kv_bytes != kv_bytes) {
        fprintf(stderr, "Inkling KV estimate disagrees with the full and sliding visit capacities\n");
        goto done;
    }
    const size_t bytes = (size_t)vocab * sizeof(float);
    logits = malloc(bytes);
    before = malloc(bytes);
    if (!logits || !before) goto done;

    const int tokens[] = {32269, 150770, 13}; // "Hello world."
#ifdef __APPLE__
    qw3_tokenize_text(engine, "Hello world.", &prompt);
    if (prompt.len != 3 || memcmp(prompt.v, tokens, sizeof(tokens)) != 0) goto done;
#else
    for (unsigned i = 0; i < 3; i++) qw3_tokens_push(&prompt, tokens[i]);
#endif
    for (unsigned i = 0; i < 3; i++) {
        if (qw3_session_eval(session, tokens[i], err, sizeof(err)) != 0 ||
            verify_logits(session, logits, vocab, i) != 0) goto done;
    }
    memcpy(before, logits, bytes);
    qw3_session_rewind(session, 1);
    if (qw3_session_sync(session, &prompt, err, sizeof(err)) != 0 ||
        qw3_session_copy_logits(session, logits, vocab) != vocab ||
        memcmp(before, logits, bytes) != 0) goto done;
    printf("test-inkling-%s: tokenizer, KV estimate, reference top-5 and exact replay ok\n", backend_name);
    status = 0;

done:
    if (status) fprintf(stderr, "test-inkling-%s: failed %s\n", backend_name, err);
    free(before);
    free(logits);
    qw3_tokens_free(&prompt);
    qw3_session_free(session);
    qw3_engine_close(engine);
    return status;
}
