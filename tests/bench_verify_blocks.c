/* Diagnostic only: paired dense backbone block cost, NOT speculative decoding.
 * Both sides compute just the last LM head. Production APIs are unchanged. */
#include "../qw3.c"
#include <inttypes.h>

static int block_argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

static int block_compare(const float *a, const float *b, int n,
                         double *maximum, double *rms) {
    double ss = 0.0;
    *maximum = 0.0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i])) {
            *maximum = *rms = INFINITY;
            return 0;
        }
        const double d = fabs((double)a[i] - b[i]);
        if (d > *maximum) *maximum = d;
        ss += d * d;
    }
    *rms = sqrt(ss / n);
    return *maximum <= 0.005 && *rms <= 0.001;
}

static int block_history_matches(qw3_session *s, const qw3_tokens *prefix,
                                  const int *suffix, int n) {
    if (!s->valid || s->kv.pos != prefix->len + n ||
        s->tokens.len != prefix->len + n) return 0;
    return !memcmp(s->tokens.v, prefix->v, prefix->len * sizeof(int)) &&
           !memcmp(s->tokens.v + prefix->len, suffix, n * sizeof(int));
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: %s MODEL [REPEATS=3] [BLOCK=2,4,8,16,32]\n", argv[0]);
        return 1;
    }
    char *end = NULL;
    const long repeats = argc >= 3 ? strtol(argv[2], &end, 10) : 3;
    if (repeats < 1 || repeats > 20 || (end && (end == argv[2] || *end))) return 1;
    end = NULL;
    const long selected = argc == 4 ? strtol(argv[3], &end, 10) : 0;
    if ((argc == 4 && selected != 2 && selected != 4 && selected != 8 &&
         selected != 16 && selected != 32) || (end && (end == argv[3] || *end))) return 1;
    qw3_engine *engine = NULL;
    qw3_session *warm = NULL, *sessions[2] = {NULL, NULL};
    qw3_tokens prompt = {0};
    qw3_engine_options options = {.model_path = argv[1], .backend = QW3_BACKEND_METAL};
    float *logits[2] = {NULL, NULL};
    int suffix[33], status = 1, quality_ok = 1;
    char error[512] = {0};
    if (qw3_engine_open(&engine, &options) != 0 ||
        engine->model.spec->ffn_kind != QW3_FFN_DENSE) goto done;
#ifdef QW3_BLOCK_INIT
    if (!QW3_BLOCK_INIT()) goto done;
#endif
#ifdef QW3_BLOCK_SELECT
    QW3_BLOCK_SELECT(0);
#endif
#ifdef QW3_BLOCK_QUANT_CHECK
    if (!QW3_BLOCK_QUANT_CHECK(engine)) goto done;
    if (getenv("QW3_BLOCK_QUANT_ONLY")) { status = 0; goto done; }
#endif
    const int baseline_batch = getenv("QW3_BLOCK_BASELINE_BATCH") != NULL;
    const int vocab = engine->model.spec->n_vocab;
    const char *prompt_text = getenv("QW3_BLOCK_PROMPT");
    if (!prompt_text) prompt_text =
        "Spiega dettagliatamente come implementare una tabella hash in C.";
    qw3_encode_chat_prompt(engine, NULL, prompt_text, QW3_THINK_NONE, &prompt);
    if (prompt.len < 1 || prompt.len + 33 > 512 ||
        qw3_session_create(&warm, engine, 512) != 0 ||
        warm->metal_n_gpu_layers != engine->model.spec->n_layer ||
        qw3_session_sync(warm, &prompt, error, sizeof(error)) != 0) goto done;
    uint64_t hash = 1469598103934665603ULL;
    for (int i = 0; i < 33; i++) {
        suffix[i] = qw3_session_argmax(warm);
        if (suffix[i] < 0 || suffix[i] >= vocab) goto done;
        hash = (hash ^ (uint32_t)suffix[i]) * 1099511628211ULL;
        if (i < 32 && qw3_session_eval_gpu_logits(warm, suffix[i],
                                                error, sizeof(error)) != 0) goto done;
    }
    qw3_session_free(warm); warm = NULL;
    for (int v = 0; v < 2; v++) {
        if (qw3_session_create(&sessions[v], engine, 512) != 0) goto done;
        logits[v] = calloc((size_t)vocab, sizeof(float));
        if (!logits[v]) goto done;
    }
    printf("block_probe prefix=%d suffix_hash=%" PRIx64 " repeats=%ld baseline_kind=%s "
           "last_logits_only=1 not_speculative_decode=1\n", prompt.len, hash, repeats,
           baseline_batch ? "batch" : "serial");
    fflush(stdout);
    const int sizes[] = {2,4,8,16,32};
    for (int shape = 0; shape < 5; shape++) {
        const int n = sizes[shape];
        if (selected && n != selected) continue;
        double elapsed[2] = {0.0,0.0}, worst_max = 0.0, worst_rms = 0.0;
        int shape_ok = 1;
        /* One untimed paired trial per shape warms pipelines and buffers. */
        for (int trial = -1; trial < repeats; trial++) {
#ifdef QW3_BLOCK_SELECT
            QW3_BLOCK_SELECT(0);
#endif
            for (int v = 0; v < 2; v++) {
                qw3_session_invalidate(sessions[v]);
                if (qw3_session_sync(sessions[v], &prompt, error, sizeof(error)) != 0)
                    goto done;
                if (!block_history_matches(sessions[v], &prompt, suffix, 0) ||
                    qw3_session_copy_logits(sessions[v], logits[v], vocab) != vocab)
                    goto done;
            }
            /* Each trial must start from the same prompt computation/state.
             * This readback is outside the timed suffix. */
            for (int i = 0; i < vocab; i++) {
                if (!isfinite(logits[0][i]) || !isfinite(logits[1][i]) ||
                    logits[0][i] != logits[1][i]) {
                    snprintf(error, sizeof(error), "initial prefix logits differ at %d", i);
                    goto done;
                }
            }
            double times[2];
            for (int order = 0; order < 2; order++) {
                const int v = order ^ ((trial + 1) & 1);
#ifdef QW3_BLOCK_SELECT
                QW3_BLOCK_SELECT(v);
#endif
                const double start = qw3_now_sec();
                if (v == 0 && !baseline_batch) {
                    for (int t = 0; t < n; t++)
                        if (qw3_metal_session_eval_dense_token(sessions[v], suffix[t],
                                error, sizeof(error), t + 1 == n ?
                                QW3_METAL_LOGITS_GPU : QW3_METAL_LOGITS_DEFER) != 0)
                            goto done;
                } else if (qw3_metal_session_eval_dense_prefill_batch(sessions[v],
                               suffix, n, error, sizeof(error), QW3_METAL_LOGITS_GPU) != 0)
                    goto done;
                times[v] = qw3_now_sec() - start;
                if (!block_history_matches(sessions[v], &prompt, suffix, n)) goto done;
                if (trial >= 0) elapsed[v] += times[v];
            }
            if (qw3_session_copy_logits(sessions[0], logits[0], vocab) != vocab ||
                qw3_session_copy_logits(sessions[1], logits[1], vocab) != vocab) goto done;
            double maximum, rms;
            int ok = block_compare(logits[0], logits[1], vocab, &maximum, &rms);
            const int id0 = block_argmax(logits[0], vocab), id1 = block_argmax(logits[1], vocab);
            ok &= id0 == id1;
            if (maximum > worst_max) worst_max = maximum;
            if (rms > worst_rms) worst_rms = rms;
            printf("block_trial n=%d trial=%d baseline_ms=%.3f batch_ms=%.3f "
                   "last_max=%.9g last_rms=%.9g ids=%d/%d ok=%d\n",
                   n, trial, times[0]*1000, times[1]*1000, maximum, rms, id0, id1, ok);
            shape_ok &= ok;
            /* The next serial token also checks recurrent/cache continuation.
             * It is outside the timed suffix on both sides. */
#ifdef QW3_BLOCK_SELECT
            QW3_BLOCK_SELECT(0);
#endif
            for (int v = 0; v < 2; v++) {
                if (qw3_session_eval_gpu_logits(sessions[v], suffix[n], error, sizeof(error)) != 0 ||
                    !block_history_matches(sessions[v], &prompt, suffix, n + 1) ||
                    qw3_session_copy_logits(sessions[v], logits[v], vocab) != vocab) goto done;
            }
            ok = block_compare(logits[0], logits[1], vocab, &maximum, &rms);
            const int next0 = block_argmax(logits[0], vocab), next1 = block_argmax(logits[1], vocab);
            ok &= next0 == next1;
            if (maximum > worst_max) worst_max = maximum;
            if (rms > worst_rms) worst_rms = rms;
            printf("block_continuation n=%d trial=%d max=%.9g rms=%.9g ids=%d/%d ok=%d\n",
                   n, trial, maximum, rms, next0, next1, ok);
            shape_ok &= ok;
            fflush(stdout);
        }
        printf("block_summary n=%d baseline_ms=%.3f batch_ms=%.3f backbone_ratio=%.3f "
               "worst_max=%.9g worst_rms=%.9g quality_ok=%d last_logits_only=1\n",
               n, elapsed[0]*1000/repeats, elapsed[1]*1000/repeats,
               elapsed[0]/elapsed[1], worst_max, worst_rms, shape_ok);
        quality_ok &= shape_ok;
        fflush(stdout);
    }
    status = quality_ok ? 0 : 2;
done:
    if (status) fprintf(stderr, "block probe failed: %s%s\n", error,
                        status == 2 ? "full-vector equivalence limits exceeded" : "");
    for (int v = 0; v < 2; v++) {
        free(logits[v]);
        if (sessions[v]) qw3_session_free(sessions[v]);
    }
    if (warm) qw3_session_free(warm);
    qw3_tokens_free(&prompt);
    if (engine) qw3_engine_close(engine);
    return status;
}
