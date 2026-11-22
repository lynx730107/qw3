/* Within-process comparison. Both orders share one session and the same 160 IDs. */
#define main saved_bench_main
#include "../../qw3_bench.c"
#undef main
int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=2) return 2;
        int rc=1;
        char error[512]={0};
        qw3_engine *engine=NULL;qw3_session *session=NULL;
        qw3_tokens tokens={0};
        qw3_engine_options opt={.model_path=argv[1],.backend=QW3_BACKEND_METAL,.n_threads=4};
        if (!CANDIDATE_COMPILE() || qw3_engine_open(&engine,&opt) ||
            qw3_session_create(&session,engine,256) || session->metal_n_gpu_layers!=64 ||
            fill_synthetic_tokens(&tokens,160,qw3_vocab_size(engine),qw3_token_eos(engine),1u)) goto done;
        const int modes[]={0,1,0,1,1,0,1,0,0,1,0,1,1,0};
        qw3_tokens prefix={.v=tokens.v,.len=128,.cap=tokens.cap};
        for (unsigned i=0;i<sizeof(modes)/sizeof(*modes);i++) {
            CANDIDATE_VARIANT=1;
            scratch_select(modes[i]);
            qw3_session_invalidate(session);
            if (qw3_session_sync(session,&prefix,error,sizeof(error))) goto done;
            uint64_t calls=CANDIDATE_CALLS;
            double start=bench_now_sec();
            for (int j=128;j<160;j++) if (qw3_session_eval(session,tokens.v[j],error,sizeof(error))) goto done;
            double ms=(bench_now_sec()-start)*1000;
            uint64_t used=CANDIDATE_CALLS-calls;
            if (used!=(uint64_t)(modes[i]*32*CANDIDATE_LAYER_CALLS)) goto done;
            printf("rep=%u warmup=%d mode=%d ms=%.6f calls=%llu\n",i,i<2,modes[i],ms,(unsigned long long)used);
            fflush(stdout);
        }
        rc=0;
done:
        if (rc) fprintf(stderr,"paired decode failed %s\n",error);
        if (session) qw3_session_free(session);
        if (engine) qw3_engine_close(engine);
        qw3_tokens_free(&tokens);return rc;
    }
}
