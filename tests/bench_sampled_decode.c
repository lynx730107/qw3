#include "../qw3.h"
#include <inttypes.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Fixed-length sampled decode; EOS does not shorten the measurement. */
int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL\n", argv[0]);
        return 1;
    }
    qw3_engine *engine = NULL;
    qw3_session *session = NULL;
    qw3_tokens prompt = {0};
    qw3_engine_options options = {.model_path = argv[1], .backend = QW3_BACKEND_METAL};
    char error[512] = {0};
    int status = 1;
    if (qw3_engine_open(&engine, &options) != 0) goto done;
    if (qw3_session_create(&session, engine, 4096) != 0) goto done;
    qw3_encode_chat_prompt(engine, NULL,
        "Spiega dettagliatamente come implementare una tabella hash in C.",
        QW3_THINK_NONE, &prompt);
    if (qw3_session_sync(session, &prompt, error, sizeof(error)) != 0) goto done;
    uint64_t rng = 42, hash = 1469598103934665603ULL;
    double sampling = 0, forward = 0;
    for (int i = 0; i < 128; i++) {
        const qw3_tokens *history = qw3_session_tokens(session);
        int count = history->len > 1024 ? 1024 : history->len;
        double t = now();
        int token = qw3_session_sample_repetition(session, 0.6f, 20, 0.95f, 0,
            &rng, history->v + history->len - count, count, 1.06f);
        sampling += now() - t;
        if (token < 0) goto done;
        hash = (hash ^ (uint32_t)token) * 1099511628211ULL;
        t = now();
        if (qw3_session_eval(session, token, error, sizeof(error)) != 0) goto done;
        forward += now() - t;
    }
    printf("tokens=128 sample_ms=%.3f forward_ms=%.3f tok_s=%.3f hash=%" PRIx64 "\n",
        sampling * 1000 / 128, forward * 1000 / 128,
        128 / (sampling + forward), hash);
    status = 0;
done:
    if (status) fprintf(stderr, "sampled benchmark failed: %s\n", error);
    qw3_tokens_free(&prompt);
    if (session) qw3_session_free(session);
    if (engine) qw3_engine_close(engine);
    return status;
}
