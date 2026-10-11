#include "hooks.inc"

static int scratch_parse_mode(const char *name) {
    return name && !strcmp(name, "all") ? 3 : -1;
}

static int logits_equal(qw3_session **s,int vocab) {
    float *a=malloc(vocab*sizeof(float)),*b=malloc(vocab*sizeof(float));
    int ok=a && b && qw3_session_copy_logits(s[0],a,vocab)==vocab &&
        qw3_session_copy_logits(s[1],b,vocab)==vocab;
    for (int i=0;ok && i<vocab;i++) if (!isfinite(a[i]) || !isfinite(b[i]) || memcmp(a+i,b+i,sizeof(float))) {
        fprintf(stderr,"candidate logit mismatch id=%d baseline=%.9g candidate=%.9g\n",i,a[i],b[i]);ok=0;
    }
    free(a);free(b);
    return ok;
}

static int states_equal(qw3_session **s) {
    qw3_metal_state_info info={0};
    if (!qw3_metal_session_get_state_info(s[0]->metal,&info) || info.kv_type!=1u) return 0;
    const size_t chunk=4u*1024u*1024u;
    unsigned char *data[2]={malloc(chunk),malloc(chunk)};
    int ok=data[0] && data[1];
    if (s[0]->tokens.len!=s[1]->tokens.len || s[0]->kv.pos!=s[1]->kv.pos ||
        memcmp(s[0]->tokens.v,s[1]->tokens.v,s[0]->tokens.len*sizeof(int))) ok=0;
    for (int kind=QW3_METAL_STATE_GQA_K;ok && kind<=QW3_METAL_STATE_CONV;kind++) {
        if (kind==QW3_METAL_STATE_GQA_FLASH_TAIL) continue;
        const uint64_t bytes=kind==QW3_METAL_STATE_DELTANET ? info.deltanet_bytes :
            kind==QW3_METAL_STATE_CONV ? info.conv_bytes : info.kv_row_bytes*s[0]->tokens.len;
        const uint32_t layers=kind<=QW3_METAL_STATE_GQA_V ? info.n_full_layers : 1u;
        for (uint32_t l=0;ok && l<layers;l++) for (uint64_t off=0;ok && off<bytes;off+=chunk) {
            uint64_t size=MIN((uint64_t)chunk,bytes-off);
            for (int v=0;ok && v<2;v++) ok=qw3_metal_session_read_state(s[v]->metal,
                (qw3_metal_state_buffer)kind,l,off,data[v],size);
            if (ok && memcmp(data[0],data[1],size)) {
                fprintf(stderr,"canonical state mismatch kind=%d layer=%u offset=%llu\n",kind,l,(unsigned long long)off);ok=0;
            }
        }
    }
    free(data[0]);free(data[1]);return ok;
}

int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=4) return 2;
        layout_variant=atoi(argv[3]);if (layout_variant<1 || layout_variant>2) return 2;
        int mode=scratch_parse_mode(argv[2]),rc=1;
        if (mode<1) return 2;
        qw3_engine *e=NULL;qw3_session *s[2]={NULL,NULL};qw3_tokens prompt={0};char err[512]={0};
        qw3_engine_options options={.model_path=argv[1],.backend=QW3_BACKEND_METAL};
        if (qw3_engine_open(&e,&options) || !iq3_layout_compile() || !iq3_layout_prepare(e) ||
            !g_deltanet_batch_recur_packed8_pipeline) goto done;
        for (int v=0;v<2;v++) if (qw3_session_create(&s[v],e,512) || s[v]->metal_n_gpu_layers!=64) goto done;
        uint32_t rng=1;int vocab=e->model.spec->n_vocab;
        for (int i=0;i<159;i++) {
            rng=rng*1664525u+1013904223u;int token=rng%(uint32_t)vocab;
            if (token==qw3_token_eos(e)) token=(token+1)%vocab;
            qw3_tokens_push(&prompt,token);
        }
        int lengths[]={1,15,16,31,127,128,129,159};
        for (int trial=0;trial<8;trial++) {
            const int count=lengths[trial];
            qw3_tokens prefix={.v=prompt.v,.len=count,.cap=prompt.cap};
            for (int v=0;v<2;v++) {
                scratch_select(v ? mode : 0);qw3_session_invalidate(s[v]);
                if (qw3_session_sync(s[v],&prefix,err,sizeof(err))) goto done;
            }
            if (!logits_equal(s,vocab) || !states_equal(s)) goto done;
            if (count==128) {
                prefix.len=159;
                for (int v=0;v<2;v++) {
                    scratch_select(v ? mode : 0);
                    if (qw3_session_sync(s[v],&prefix,err,sizeof(err))) goto done;
                }
                if (!logits_equal(s,vocab) || !states_equal(s)) goto done;
            }
            FILE *checkpoint=tmpfile();
            uint64_t bytes=qw3_session_payload_bytes(s[1]);
            if (!checkpoint || qw3_session_save_payload(s[1],checkpoint,err,sizeof(err))) goto done;
            qw3_session_invalidate(s[1]); rewind(checkpoint);
            if (qw3_session_load_payload(s[1],checkpoint,bytes,err,sizeof(err))) { fclose(checkpoint); goto done; }
            fclose(checkpoint);
            if (!logits_equal(s,vocab) || !states_equal(s)) goto done;
            for (int step=0;step<32;step++) {
                int token=qw3_session_argmax(s[0]);
                if (token<0 || token!=qw3_session_argmax(s[1])) goto done;
                for (int v=0;v<2;v++) {
                    scratch_select(v ? mode : 0);
                    if (qw3_session_eval_gpu_logits(s[v],token,err,sizeof(err))) goto done;
                }
                if (!logits_equal(s,vocab)) goto done;
            }
            if (!states_equal(s)) goto done;
            printf("full_exact mode=%s prefix=%d extended_prefix=%d continuation=32 logits=bit_exact live_kv=exact deltanet=exact conv=exact\n",argv[2],count,prefix.len);
            fflush(stdout);
        }
        if (!iq3_layout_calls[0] || !iq3_layout_calls[1] || !iq3_layout_redirects) goto done;
        for (id<MTLBuffer> buffer in iq3_layout_weights.allValues) if (!iq3_layout_guards(buffer)) goto done;
        printf("PASS IQ3_S layout variant=%d single=%llu pair=%llu redirects=%llu\n",layout_variant,
            (unsigned long long)iq3_layout_calls[0],(unsigned long long)iq3_layout_calls[1],(unsigned long long)iq3_layout_redirects);
        rc=0;
done:
        if (rc) fprintf(stderr,"candidate quality failed mode=%s error=%s\n",argv[2],err);
        for (int v=0;v<2;v++) if (s[v]) qw3_session_free(s[v]);
        if (e) qw3_engine_close(e);qw3_tokens_free(&prompt);return rc;
    }
}
