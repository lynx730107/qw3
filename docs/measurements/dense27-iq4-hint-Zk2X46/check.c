#include "qw3.h"
#include <stdlib.h>
#include <math.h>
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    qw3_engine_options opt = {.model_path = argv[1], .backend = QW3_BACKEND_METAL, .n_threads = 4};
    qw3_engine *engine;
    qw3_session *session;
    if (qw3_engine_open(&engine, &opt) || qw3_session_create(&session, engine, 256)) return 1;
    int vocab = qw3_vocab_size(engine);
    float *logits = malloc(sizeof(*logits) * vocab);
    if (!logits) return 1;
    char err[256];
    int token = 42;
    for (int i = 0; i < 16; i++) {
        if (qw3_session_eval(session, token, err, sizeof(err))) {
            fprintf(stderr, "%s\n", err);
            return 1;
        }
        if (qw3_session_copy_logits(session, logits, vocab) != vocab) return 1;
        for (int j = 0; j < vocab; j++) if (!isfinite(logits[j])) return 1;
        if (fwrite(logits, sizeof(*logits), vocab, stdout) != (size_t)vocab) return 1;
        token = qw3_session_argmax(session);
    }
    free(logits);
    qw3_session_free(session);
    qw3_engine_close(engine);
    return fflush(stdout) != 0;
}
