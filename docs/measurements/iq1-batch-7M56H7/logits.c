#define main original_benchmark_main
#include "qw3_bench.c"
#undef main

/* Save four consecutive full vocabularies for identical teacher-forced tokens. */
int main(int argc, char **argv) {
    if (argc != 4) return 2;
    int depth = atoi(argv[2]);
    qw3_engine_options opt = {.model_path=argv[1], .backend=QW3_BACKEND_METAL, .n_threads=4};
    qw3_engine *engine=NULL;
    qw3_session *s=NULL;
    qw3_tokens seq={0};
    char error[256]={0};
    if (qw3_engine_open(&engine,&opt)) return 1;
    int vocab=qw3_vocab_size(engine);
    if (fill_synthetic_tokens(&seq,depth+3,vocab,qw3_token_eos(engine),1u) ||
        qw3_session_create(&s,engine,depth+8)) return 1;
    qw3_tokens prefix={.v=seq.v,.len=depth,.cap=depth};
    if (qw3_session_sync(s,&prefix,error,sizeof(error))) return 1;
    FILE *file=fopen(argv[3],"wb");
    float *logits=malloc(vocab*sizeof(*logits));
    if (!file || !logits) return 1;
    for (int i=0;i<4;++i) {
        if (i && qw3_session_eval(s,seq.v[depth+i-1],error,sizeof(error))) return 1;
        if (qw3_session_copy_logits(s,logits,vocab)!=vocab ||
            fwrite(logits,sizeof(float),vocab,file)!=(size_t)vocab) return 1;
    }
    int rc=fclose(file);
    free(logits);
    qw3_session_free(s);
    qw3_tokens_free(&seq);
    qw3_engine_close(engine);
    return rc!=0;
}
