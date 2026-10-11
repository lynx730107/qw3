/* Private decode experiment. Exact checks are outside the forward timer. */
#define main groups_original_quality_main
#include ".iq3_groups/quality.inc"
#undef main
#define main groups_original_bench_main
#include "../qw3_bench.c"
#undef main
int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=3) return 2;
        groups_width=atoi(argv[2]);if (groups_width!=4 && groups_width!=8) return 2;
        qw3_engine *e=NULL;qw3_session *s[2]={NULL,NULL};qw3_tokens prefix={0};
        qw3_engine_options opt={.model_path=argv[1],.backend=QW3_BACKEND_METAL,.n_threads=4};
        int rc=1;char err[512]={0};double total[2]={0,0};
        setenv("QW3_METAL_NGL","64",1);
        if (qw3_engine_open(&e,&opt) || !groups_compile() ||
            fill_synthetic_tokens(&prefix,128,qw3_vocab_size(e),qw3_token_eos(e),1u)) goto done;
        for (int v=0;v<2;v++) {
            scratch_select(v);
            if (qw3_session_create(&s[v],e,512) || qw3_session_sync(s[v],&prefix,err,sizeof(err))) goto done;
        }
        if (!logits_equal(s,qw3_vocab_size(e)) || !states_equal(s)) goto done;
        groups_calls=0;
        printf("token,order,baseline_ms,candidate_ms,saved_ms\n");fflush(stdout);
        for (int i=0;i<72;i++) {
            double ms[2]={0,0};int token=qw3_session_argmax(s[0]);
            if (token<0 || token!=qw3_session_argmax(s[1])) goto done;
            for (int j=0;j<2;j++) {
                int v=j^(i&1);scratch_select(v);double start=bench_now_sec();
                if (qw3_session_eval_gpu_logits(s[v],token,err,sizeof(err))) goto done;
                ms[v]=(bench_now_sec()-start)*1000;
            }
            if (!logits_equal(s,qw3_vocab_size(e))) goto done;
            if (i>=8) {
                total[0]+=ms[0];total[1]+=ms[1];
                printf("%d,%s,%.9f,%.9f,%.9f\n",i,i&1 ? "BA" : "AB",ms[0],ms[1],ms[0]-ms[1]);fflush(stdout);
            }
        }
        if (!states_equal(s) || groups_calls!=72u*19u) goto done;
        fprintf(stderr,"PASS groups=%u timed_tokens=64 baseline_tps=%.9f candidate_tps=%.9f gain_pct=%.9f calls=%llu full_logits=bit_exact live_state=exact\n",
            groups_width,64000/total[0],64000/total[1],100*(total[0]/total[1]-1),(unsigned long long)groups_calls);rc=0;
done:
        if (rc) fprintf(stderr,"IQ3 groups failed: %s calls=%llu\n",err,(unsigned long long)groups_calls);
        for (int v=0;v<2;v++) if (s[v]) qw3_session_free(s[v]);
        if (e) qw3_engine_close(e);qw3_tokens_free(&prefix);return rc;
    }
}
