/* Diagnostic GPU stage attribution; encoder boundaries differ from native. */
#include ".decode_profile/hooks.inc"
#define qw3_metal_env_n_gpu_layers profile_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled profile_core_llama_split_enabled
#define qw3_metal_llama_split_start profile_core_llama_split_start
#include "../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
/* Reuse the exact full-vocabulary and canonical-state oracle. */
#include ".decode_profile/oracle.inc"
#define main profile_original_bench_main
#include "../qw3_bench.c"
#undef main
int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=2) return 2;
        qw3_engine *e=NULL;qw3_session *s[2]={NULL,NULL};qw3_tokens prefix={0},tokens={0};
        qw3_engine_options opt={.model_path=argv[1],.backend=QW3_BACKEND_METAL,.n_threads=4};
        int rc=1;char err[512]={0};
        if (qw3_engine_open(&e,&opt) || !qw3_metal_compile_kernels() || !probe_profile_initialize() ||
            fill_synthetic_tokens(&prefix,128,qw3_vocab_size(e),qw3_token_eos(e),1u) ||
            fill_synthetic_tokens(&tokens,32,qw3_vocab_size(e),qw3_token_eos(e),2u)) goto done;
        for (int v=0;v<2;v++) if (qw3_session_create(&s[v],e,512) ||
            s[v]->metal_n_gpu_layers!=64 || qw3_session_sync(s[v],&prefix,err,sizeof(err))) goto done;
        if (!logits_equal(s,qw3_vocab_size(e)) || !states_equal(s)) goto done;
        for (int i=0;i<32;i++) {
            if (qw3_session_eval_gpu_logits(s[0],tokens.v[i],err,sizeof(err))) goto done;
            int capture=i%8==0;
            if (capture) probe_profile_start();
            double start=bench_now_sec();
            if (qw3_session_eval_gpu_logits(s[1],tokens.v[i],err,sizeof(err))) goto done;
            double wall=bench_now_sec()-start;
            if (capture) probe_profile_end(i,wall);
            if (!logits_equal(s,qw3_vocab_size(e))) goto done;
        }
        if (!states_equal(s)) goto done;
        printf("PASS decode_profile depth=128 tokens=32 profiled_steps=0,8,16,24 full_logits=finite_bit_exact canonical_state=exact native_k64=1 native_gdn_cache64=1\n");rc=0;
done:
        probe_capture=0;
        if (rc) fprintf(stderr,"decode profile failed: %s\n",err);
        for (int v=0;v<2;v++) if (s[v]) qw3_session_free(s[v]);
        if (e) qw3_engine_close(e);qw3_tokens_free(&prefix);qw3_tokens_free(&tokens);return rc;
    }
}
