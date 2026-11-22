/* Isolated diagnostic. Time only prefix extension and selection of the first greedy token. */
#include "qw3_metal.m"
#define main saved_bench_main
#include "qw3_bench.c"
#undef main

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    const int cached = atoi(argv[2]), added = atoi(argv[3]);
    if (cached < 0 || added <= 0 || cached + added > 8192) return 2;
    qw3_engine *engine = NULL;
    qw3_session *session = NULL;
    qw3_tokens tokens = {0};
    char error[512] = {0};
    qw3_engine_options options = {.model_path=argv[1], .backend=QW3_BACKEND_METAL, .n_threads=4};
    int rc = 1, expected = -1;
    FILE *prefix = NULL;
    uint64_t bytes = 0;
    if (qw3_engine_open(&engine,&options) || qw3_session_create(&session,engine,MAX(256,cached+added+1)) ||
        fill_synthetic_tokens(&tokens,cached+added,qw3_vocab_size(engine),qw3_token_eos(engine),1u)) goto done;
    if (cached) {
        qw3_tokens initial = {.v=tokens.v,.len=cached,.cap=cached};
        if (qw3_session_sync(session,&initial,error,sizeof(error))) goto done;
        prefix=tmpfile();
        if (!prefix || qw3_session_save_payload(session,prefix,error,sizeof(error))) goto done;
        bytes=qw3_session_payload_bytes(session);
    }
    const int order[]={0,3,0,3,3,0,3,0,0,3};
    for (unsigned i=0; i<sizeof(order)/sizeof(*order); i++) {
        partial_mode=order[i];
        qw3_session_invalidate(session);
        if (cached && (fseek(prefix,0,SEEK_SET) || qw3_session_load_payload(session,prefix,bytes,error,sizeof(error)))) goto done;
        if (qw3_session_common_prefix(session,&tokens)!=cached) goto done;
        double start=bench_now_sec();
        if (qw3_session_sync(session,&tokens,error,sizeof(error))) goto done;
        int id=qw3_session_argmax(session);
        double elapsed=(bench_now_sec()-start)*1000;
        if (!i) expected=id;
        if (id<0 || id!=expected) goto done;
        printf("sample=%u warmup=%d cached=%d added=%d mode=%u ms=%.6f first=%d\n",i,i<2,cached,added,partial_mode,elapsed,id);
        fflush(stdout);
    }
    rc=0;
done:
    if (rc) fprintf(stderr,"turn failed %s\n",error);
    if (prefix) fclose(prefix);
    if (session) qw3_session_free(session);
    if (engine) qw3_engine_close(engine);
    qw3_tokens_free(&tokens);
    return rc;
}
