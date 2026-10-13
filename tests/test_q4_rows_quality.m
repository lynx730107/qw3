#include ".q4_rows/hooks.inc"
#define main q4_rows_screen_main
#include ".q4_rows/model.inc"
#undef main

static int checkpoint_exact(qw3_session *s,int vocab,char *err,size_t errlen) {
    FILE *reference=tmpfile(),*payload=tmpfile();
    int ok=reference && payload;
    uint64_t bytes=qw3_session_payload_bytes(s);
    if (ok) ok=grid_oracle_logits(reference,1,s,vocab) && grid_oracle_state(reference,1,s) &&
        !qw3_session_save_payload(s,payload,err,errlen) && !fflush(payload);
    if (ok) {
        qw3_session_invalidate(s);rewind(payload);
        ok=!qw3_session_load_payload(s,payload,bytes,err,errlen);
    }
    if (ok) {
        rewind(reference);
        ok=grid_oracle_logits(reference,0,s,vocab) && grid_oracle_state(reference,0,s) &&
           fgetc(reference)==EOF && !ferror(reference);
    }
    if (reference) fclose(reference);
    if (payload) fclose(payload);
    return ok;
}

int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=2) return 2;
        qw3_engine *e=NULL;qw3_session *s=NULL;qw3_tokens prompt={0};FILE *oracle=NULL;
        int rc=1,trial=-1,phase=-1;char err[512]={0};uint32_t rng=1;
        qw3_engine_options options={.model_path=argv[1],.backend=QW3_BACKEND_METAL,.n_threads=4};
        if (qw3_engine_open(&e,&options) || !q4_rows_compile() ||
            qw3_session_create(&s,e,256) || s->metal_n_gpu_layers!=64) goto done;
        int vocab=qw3_vocab_size(e);
        for (int i=0;i<159;i++) {
            rng=rng*1664525u+1013904223u;int token=rng%(uint32_t)vocab;
            if (token==qw3_token_eos(e)) token=(token+1)%vocab;
            qw3_tokens_push(&prompt,token);
        }
        const int lengths[]={1,15,16,31,127,128,129,159};
        for (trial=0;trial<8;trial++) {
            oracle=tmpfile();if (!oracle) goto done;
            for (phase=0;phase<=1;phase++) {
                if (phase && (fflush(oracle) || fseek(oracle,0,SEEK_SET))) goto done;
                scratch_select(phase);qw3_session_invalidate(s);
                qw3_tokens prefix={.v=prompt.v,.len=lengths[trial],.cap=prompt.cap};
                if (qw3_session_sync(s,&prefix,err,sizeof(err)) ||
                    !grid_oracle_logits(oracle,phase==0,s,vocab) ||
                    !grid_oracle_state(oracle,phase==0,s)) goto done;
                if (lengths[trial]==128) {
                    prefix.len=159;
                    if (qw3_session_sync(s,&prefix,err,sizeof(err)) ||
                        !grid_oracle_logits(oracle,phase==0,s,vocab) ||
                        !grid_oracle_state(oracle,phase==0,s)) goto done;
                }
                if (!checkpoint_exact(s,vocab,err,sizeof(err))) goto done;
                q4_rows_calls=0;q4_rows_fallbacks=0;
                for (int step=0;step<32;step++) {
                    int token=qw3_session_argmax(s),expected=token;
                    if (token<0) goto done;
                    if (!phase) {
                        if (fwrite(&token,sizeof(token),1,oracle)!=1) goto done;
                    } else if (fread(&expected,sizeof(expected),1,oracle)!=1 || expected!=token) goto done;
                    if (qw3_session_eval_gpu_logits(s,token,err,sizeof(err)) ||
                        !grid_oracle_logits(oracle,phase==0,s,vocab)) goto done;
                }
                if (!grid_oracle_state(oracle,phase==0,s) ||
                    q4_rows_calls!=(phase ? 1088u : 0u) ||
                    q4_rows_fallbacks!=(phase ? 64u : 0u)) goto done;
                if (phase && (fgetc(oracle)!=EOF || ferror(oracle))) goto done;
                printf("full_exact phase=%d prefix=%d extended_prefix=%d continuation=32 finite_logits=bit_exact canonical_state=exact history=exact checkpoint=exact calls=%llu fallbacks=%llu sessions=1\n",phase,lengths[trial],prefix.len,(unsigned long long)q4_rows_calls,(unsigned long long)q4_rows_fallbacks);fflush(stdout);
            }
            fclose(oracle);oracle=NULL;
        }
        puts("PASS q4_rows_quality eight_prefixes=exact checkpoint_roundtrips=16 sessions=1 additional_weight_buffer_bytes=0");rc=0;
done:
        if (rc) fprintf(stderr,"Q4 rows quality failed trial=%d phase=%d error=%s\n",trial,phase,err);
        if (oracle) fclose(oracle);
        if (s) qw3_session_free(s);
        if (e) qw3_engine_close(e);
        qw3_tokens_free(&prompt);return rc;
    }
}
