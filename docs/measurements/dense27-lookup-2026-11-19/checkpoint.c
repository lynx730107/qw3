#include "qw3.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return 1000.0 * t.tv_sec + t.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    char err[256] = {0};
    qw3_engine_options opt = {.model_path=argv[1], .backend=QW3_BACKEND_METAL, .n_threads=4};
    qw3_engine *engine;
    qw3_session *session;
    int ids[8192];
    int count = 0;
    while (count < 8192 && scanf("%d", &ids[count]) == 1) count++;
    if (count == 0 || count >= 8192) return 2;
    qw3_tokens prompt = {.v=ids, .len=count, .cap=count};
    if (qw3_engine_open(&engine, &opt) || qw3_session_create(&session, engine, 8192)) return 1;
    if (qw3_session_sync(session, &prompt, err, sizeof(err))) goto failed;
    const int vocab = qw3_vocab_size(engine);
    const size_t bytes = sizeof(float) * (size_t)vocab;
    float *oracle = malloc(bytes);
    float *actual = malloc(bytes);
    if (!oracle || !actual) return 1;
    int next = qw3_session_argmax(session);
    uint64_t payload = qw3_session_payload_bytes(session);
    for (int i = 0; i < 3; i++) {
        FILE *fp = tmpfile();
        if (!fp) return 1;
        double t = ms();
        if (qw3_session_save_payload(session, fp, err, sizeof(err)) || fflush(fp)) goto failed;
        double write_ms = ms() - t;
        if (qw3_session_eval(session,next,err,sizeof(err)) ||
            qw3_session_copy_logits(session,actual,vocab) != vocab) goto failed;
        if (i == 0) memcpy(oracle,actual,bytes);
        if (memcmp(actual,oracle,bytes)) return 3;
        rewind(fp);
        t = ms();
        if (qw3_session_load_payload(session,fp,payload,err,sizeof(err))) goto failed;
        double read_ms = ms() - t;
        if (qw3_session_pos(session) != count || qw3_session_argmax(session) != next) return 3;
        printf("prefix=%d bytes=%llu save_ms=%.2f restore_ms=%.2f\n",count,
               (unsigned long long)payload,write_ms,read_ms);
        fclose(fp);
    }
    if (qw3_session_eval(session,next,err,sizeof(err)) ||
        qw3_session_copy_logits(session,actual,vocab) != vocab || memcmp(actual,oracle,bytes)) return 3;
    free(oracle);
    free(actual);
    qw3_session_free(session);
    qw3_engine_close(engine);
    return 0;
failed:
    fprintf(stderr, "%s\n",err);
    return 1;
}
