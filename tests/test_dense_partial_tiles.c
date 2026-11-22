/* Compare final and continuation logits across the partial-tile boundaries. */
#include "../qw3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int run(const char *model, FILE *oracle, int compare) {
    qw3_engine_options opt = {.model_path = model, .backend = QW3_BACKEND_METAL,
                              .n_threads = 4};
    const int lengths[] = {63, 64, 65, 95, 96, 127, 128, 129, 193};
    int ids[193];
    qw3_engine *engine = NULL;
    qw3_session *session = NULL;
    float *actual = NULL, *expected = NULL;
    char err[256] = {0};
    int ok = 0;
    if (setenv("QW3_METAL_DENSE_PARTIAL_TILES", compare ? "1" : "0", 1) ||
        qw3_engine_open(&engine, &opt) || qw3_session_create(&session, engine, 256))
        goto done;
    int vocab = qw3_vocab_size(engine);
    size_t bytes = (size_t)vocab * sizeof(float);
    actual = malloc(bytes);
    expected = malloc(bytes);
    if (!actual || !expected) goto done;
    for (int i = 0; i < 193; i++) ids[i] = (777 + i * 379) % vocab;
    for (size_t i = 0; i < sizeof(lengths)/sizeof(*lengths); i++) {
        qw3_tokens prefix = {.v = ids, .len = lengths[i], .cap = 193};
        qw3_session_invalidate(session);
        if (qw3_session_sync(session, &prefix, err, sizeof(err))) goto done;
        for (int step = 0; step < 2; step++) {
            if (step && qw3_session_eval(session, 100, err, sizeof(err))) goto done;
            if (qw3_session_copy_logits(session, actual, vocab) != vocab) goto done;
            for (int j = 0; j < vocab; j++) if (!isfinite(actual[j])) goto done;
            if (compare) {
                if (fread(expected, 1, bytes, oracle) != bytes ||
                    memcmp(actual, expected, bytes)) {
                    fprintf(stderr, "logit mismatch at length %d, step %d\n", lengths[i], step);
                    goto done;
                }
            } else if (fwrite(actual, 1, bytes, oracle) != bytes) goto done;
        }
    }
    ok = 1;
done:
    if (!ok) fprintf(stderr, "partial tiles phase %d failed %s\n", compare, err);
    free(actual);
    free(expected);
    if (session) qw3_session_free(session);
    if (engine) qw3_engine_close(engine);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    FILE *oracle = tmpfile();
    if (!oracle) return 1;
    int ok = run(argv[1], oracle, 0) && !fseek(oracle, 0, SEEK_SET) && run(argv[1], oracle, 1);
    fclose(oracle);
    if (ok) puts("PASS partial tiles, 18 full vocabulary comparisons");
    return ok ? 0 : 1;
}
