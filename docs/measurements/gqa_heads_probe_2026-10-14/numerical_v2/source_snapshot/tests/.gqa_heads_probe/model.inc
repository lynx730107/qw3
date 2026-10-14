#include "../gqa_heads_numeric_oracles.inc"

int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=3) return 2;
        groups_variant=atoi(argv[2]);if (groups_variant<1 || groups_variant>2) return 2;
        qw3_engine *e=NULL;qw3_session *s=NULL;qw3_tokens prefix={0};FILE *oracle=NULL;
        printf("gqa_heads_probe_variant=%d\n",groups_variant);
        int rc=1;char err[512]={0};uint32_t rng=1;
        qw3_engine_options opt={.model_path=argv[1],.backend=QW3_BACKEND_METAL,.n_threads=4};
        if (!gqa_groups_compile() || qw3_engine_open(&e,&opt) ||
            qw3_session_create(&s,e,256) || s->metal_n_gpu_layers!=64) goto done;
        int vocab=qw3_vocab_size(e);
        for (int i=0;i<128;i++) {
            rng=rng*1664525u+1013904223u;int token=rng%(uint32_t)vocab;
            if (token==qw3_token_eos(e)) token=(token+1)%vocab;
            qw3_tokens_push(&prefix,token);
        }
        oracle=tmpfile();if (!oracle) goto done;
        for (int phase=0;phase<=1;phase++) {
            if (phase && (fflush(oracle) || fseek(oracle,0,SEEK_SET))) goto done;
            scratch_select(phase!=0);
            qw3_session_invalidate(s);
            if (qw3_session_sync(s,&prefix,err,sizeof(err)) ||
                !grid_oracle_logits(oracle,phase==0,s,vocab) ||
                !grid_oracle_state(oracle,phase==0,s)) goto done;
            gqa_groups_calls=0;gqa_groups_fallbacks=0;
            for (int step=0;step<32;step++) {
                int token=qw3_session_argmax(s),expected=token;
                if (token<0) goto done;
                if (phase==0) {
                    if (fwrite(&token,sizeof(token),1,oracle)!=1) goto done;
                } else { if (fread(&expected,sizeof(expected),1,oracle)!=1) goto done; token=expected; }
                if (qw3_session_eval_gpu_logits(s,token,err,sizeof(err)) ||
                    !grid_oracle_logits(oracle,phase==0,s,vocab)) goto done;
            }
            if (!grid_oracle_state(oracle,phase==0,s) ||
                gqa_groups_calls!=(phase ? 32u*16u : 0u)) goto done;
            if (gqa_groups_fallbacks) goto done;
            gqa_groups_print_counts();
            if (phase && (fgetc(oracle)!=EOF || ferror(oracle))) goto done;
            printf("PASS gqa_heads_probe_model phase=%d prefix=128 continuation=32 finite_logits=numerically_valid canonical_state=numerically_valid input_tokens=teacher_forced history=exact calls=%llu sessions=1 additional_weight_buffer_bytes=0\n",
                phase,(unsigned long long)gqa_groups_calls);fflush(stdout);
        }
        grid_numeric_report();rc=0;
done:
        if (rc) gqa_groups_print_counts();
        if (rc) fprintf(stderr,"GQA heads probe model failed: %s calls=%llu\n",err,(unsigned long long)gqa_groups_calls);
        if (oracle) fclose(oracle);
        if (s) qw3_session_free(s);if (e) qw3_engine_close(e);qw3_tokens_free(&prefix);return rc;
    }
}
