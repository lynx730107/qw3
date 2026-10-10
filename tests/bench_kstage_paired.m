/* Adjacent AB/BA replication. Two sessions share one engine/model; this
 * estimates a paired effect, not ordinary single-session absolute throughput. */
#define main stage_original_quality_main
#include ".kstage_candidates/quality.inc"
#undef main
#define main stage_original_benchmark_main
#include "../qw3_bench.c"
#undef main
int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=4) return 2;
        stage_width=atoi(argv[2]);
        int mode=scratch_parse_mode(argv[3]);
        if ((stage_width!=64 && stage_width!=128) || mode<1) return 2;
        qw3_engine *e=NULL;qw3_session *s[2]={NULL,NULL};qw3_tokens prefix={0};
        qw3_engine_options opt={.model_path=argv[1],.backend=QW3_BACKEND_METAL,.n_threads=4};
        int rc=1;char err[512]={0};
        setenv("QW3_METAL_NGL","64",1);setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1);
        if (qw3_engine_open(&e,&opt) || !stage_compile() ||
            fill_synthetic_tokens(&prefix,128,qw3_vocab_size(e),qw3_token_eos(e),1u)) goto done;
        for (int v=0;v<2;v++) if (qw3_session_create(&s[v],e,512)) goto done;
        printf("trial,order,baseline_ms,candidate_ms,saved_ms\n");fflush(stdout);
        for (int trial=-1;trial<8;trial++) {
            double ms[2]={0,0};
            for (int j=0;j<2;j++) {
                int v=trial>=0 && (trial&1) ? 1-j : j;
                scratch_select(v ? mode : 0);qw3_session_invalidate(s[v]);
                double start=bench_now_sec();
                if (qw3_session_sync(s[v],&prefix,err,sizeof(err))) goto done;
                ms[v]=(bench_now_sec()-start)*1000;
            }
            if (!logits_equal(s,qw3_vocab_size(e))) goto done;
            if (trial>=0) {
                printf("%d,%s,%.9f,%.9f,%.9f\n",trial,trial&1 ? "BA" : "AB",ms[0],ms[1],ms[0]-ms[1]);
                fflush(stdout);
            }
        }
        if (!states_equal(s) || ((mode&1) && !stage_iq3_calls) || ((mode&2) && !stage_iq4_calls)) goto done;
        fprintf(stderr,"PASS paired_trials=8 full_logits=bit_exact live_state=exact stage_k=%d iq3=%llu iq4=%llu\n",
            stage_width,(unsigned long long)stage_iq3_calls,(unsigned long long)stage_iq4_calls);rc=0;
done:
        if (rc) fprintf(stderr,"K-stage paired trial failed: %s\n",err);
        for (int v=0;v<2;v++) if (s[v]) qw3_session_free(s[v]);
        if (e) qw3_engine_close(e);qw3_tokens_free(&prefix);return rc;
    }
}
