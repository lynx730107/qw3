/* Compare an uninterrupted decode against three weight-view release/rebinds. */
#import <Foundation/Foundation.h>
#include "../qw3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    qw3_engine *engine = NULL;
    qw3_session *reference = NULL, *resumed = NULL;
    qw3_tokens prompt = {0};
    float *expected = NULL, *actual = NULL;
    char error[256] = {0};
    int rc = 1;
    @autoreleasepool {
        qw3_engine_options options = {.model_path = argv[1], .backend = QW3_BACKEND_METAL};
        if (qw3_engine_open(&engine, &options) != 0) goto done;
        if (qw3_session_create(&reference, engine, 64) != 0 ||
            qw3_session_create(&resumed, engine, 64) != 0) goto done;
        qw3_tokenize_text(engine, "Ciao, come stai?", &prompt);
        if (qw3_session_sync(reference, &prompt, error, sizeof(error)) != 0 ||
            qw3_session_sync(resumed, &prompt, error, sizeof(error)) != 0) goto done;
    }
    int count = qw3_vocab_size(engine);
    expected = malloc((size_t)count * sizeof(float));
    actual = malloc((size_t)count * sizeof(float));
    if (!expected || !actual) goto done;
    for (int step = 0; step < 3; step++) {
        int token = qw3_session_argmax(reference);
        @autoreleasepool {
            if (qw3_session_eval(reference, token, error, sizeof(error)) != 0 ||
                qw3_session_copy_logits(reference, expected, count) != count) goto done;
        }
        int position = qw3_session_pos(resumed);
        if (qw3_engine_set_weights_active(engine, false) != 0 ||
            qw3_engine_set_weights_active(engine, false) != 0 ||
            qw3_session_pos(resumed) != position ||
            qw3_session_common_prefix(resumed, &prompt) != prompt.len) goto done;
        @autoreleasepool {
            if (qw3_engine_set_weights_active(engine, true) != 0 ||
                qw3_engine_set_weights_active(engine, true) != 0 ||
                qw3_session_eval(resumed, token, error, sizeof(error)) != 0 ||
                qw3_session_copy_logits(resumed, actual, count) != count) goto done;
        }
        if (memcmp(expected, actual, (size_t)count * sizeof(float)) != 0) {
            fprintf(stderr, "voice residency: logits changed at step %d\n", step);
            goto done;
        }
    }
    puts("voice residency: exact logits after three release/rebind cycles; KV and DeltaNet preserved");
    rc = 0;
done:
    if (rc && error[0]) fprintf(stderr, "%s\n", error);
    free(expected);
    free(actual);
    qw3_tokens_free(&prompt);
    qw3_session_free(reference);
    qw3_session_free(resumed);
    qw3_engine_close(engine);
    return rc;
}
