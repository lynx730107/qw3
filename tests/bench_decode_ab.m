/* Diagnostic-only access to Metal dispatch choices. No production API. */
#include "../qw3_metal.m"
#include "../qw3.h"
#include <inttypes.h>
#include <time.h>

static double bench_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 2) {
            fprintf(stderr, "usage: %s MODEL\n", argv[0]);
            return 1;
        }
        setenv("QW3_METAL_DENSE_IQ2_SIMDGROUPS", "8", 1);
        setenv("QW3_METAL_DENSE_IQ2_VECTOR", "1", 1);
        qw3_engine *engine = NULL;
        qw3_session *sessions[2] = {NULL, NULL};
        qw3_tokens prompt = {0};
        qw3_engine_options options = {.model_path = argv[1], .backend = QW3_BACKEND_METAL};
        char error[512] = {0};
        int status = 1;
        if (qw3_engine_open(&engine, &options) != 0) goto done;
        if (!g_matvec_iq2_xs_sg8_pipeline || !g_matvec_iq2_s_sg8_pipeline) {
            fprintf(stderr, "candidate unavailable\n");
            goto done;
        }
        qw3_encode_chat_prompt(engine, NULL,
            "Spiega dettagliatamente come implementare una tabella hash in C.",
            QW3_THINK_NONE, &prompt);
        for (int variant = 0; variant < 2; variant++) {
            g_dense_iq2_simdgroups = variant ? 8u : 2u;
            if (qw3_session_create(&sessions[variant], engine, 512) != 0 ||
                qw3_session_sync(sessions[variant], &prompt, error, sizeof(error)) != 0) goto done;
        }
        uint64_t rng[2] = {42, 42};
        uint64_t hash = 1469598103934665603ULL;
        double forward[2] = {0}, sampling[2] = {0}, block[2] = {0};
        for (int i = 0; i < 128; i++) {
            int token[2];
            for (int variant = 0; variant < 2; variant++) {
                const qw3_tokens *history = qw3_session_tokens(sessions[variant]);
                const int count = history->len > 1024 ? 1024 : history->len;
                double start = bench_now();
                token[variant] = qw3_session_sample_repetition(sessions[variant],
                    0.6f, 20, 0.95f, 0, &rng[variant],
                    history->v + history->len - count, count, 1.06f);
                if (i >= 8) sampling[variant] += bench_now() - start;
            }
            if (token[0] < 0 || token[0] != token[1]) {
                fprintf(stderr, "token mismatch at %d: %d/%d\n", i, token[0], token[1]);
                goto done;
            }
            hash = (hash ^ (uint32_t)token[0]) * 1099511628211ULL;
            /* AB, BA ordering cancels first/second dispatch bias. */
            for (int order = 0; order < 2; order++) {
                const int variant = order ^ (i & 1);
                g_dense_iq2_simdgroups = variant ? 8u : 2u;
                const double start = bench_now();
                if (qw3_session_eval_gpu_logits(sessions[variant], token[variant],
                                                error, sizeof(error)) != 0) goto done;
                const double elapsed = bench_now() - start;
                if (i >= 8) { forward[variant] += elapsed; block[variant] += elapsed; }
            }
            if (i >= 8 && ((i - 7) % 16 == 0 || i == 127)) {
                printf("block_end=%d baseline_ms=%.2f candidate_ms=%.2f gain_pct=%.2f\n",
                    i + 1, block[0] * 1000, block[1] * 1000, (block[0] / block[1] - 1) * 100);
                block[0] = block[1] = 0;
                fflush(stdout);
            }
        }
        printf("mode=iq2 tokens=128 timed=120 baseline=%.3f candidate=%.3f gain_pct=%.2f hash=%" PRIx64 "\n",
            120 / (forward[0] + sampling[0]),
            120 / (forward[1] + sampling[1]),
            (forward[0] / forward[1] - 1) * 100, hash);
        status = 0;
done:
        if (status) fprintf(stderr, "A/B benchmark failed: %s\n", error);
        qw3_tokens_free(&prompt);
        for (int i = 0; i < 2; i++) if (sessions[i]) qw3_session_free(sessions[i]);
        if (engine) qw3_engine_close(engine);
        return status;
    }
}
