#include "dense_candidates.inc"

static int logits_equal(qw3_session **s,int vocab) {
    float *a=malloc(vocab*sizeof(float)),*b=malloc(vocab*sizeof(float));
    int ok=a && b && qw3_session_copy_logits(s[0],a,vocab)==vocab &&
        qw3_session_copy_logits(s[1],b,vocab)==vocab;
    for (int i=0;ok && i<vocab;i++) if (!isfinite(a[i]) || !isfinite(b[i]) || a[i]!=b[i]) {
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

static int quant_checks(qw3_engine *e,qw3_session *s) {
    const qw3_tensor *gate=NULL,*up=NULL,*down=NULL;
    for (int l=0;l<64;l++) {
        if (e->weights.layer[l].ffn_gate->type==23u && e->weights.layer[l].ffn_up->type==23u) {
            gate=e->weights.layer[l].ffn_gate;up=e->weights.layer[l].ffn_up;
        }
        if (e->weights.layer[l].ffn_down->type==23u) down=e->weights.layer[l].ffn_down;
    }
    if (!gate || !up || !down) return 0;
    QW3MetalSessionObj *obj=(__bridge QW3MetalSessionObj *)s->metal->obj;
    id<MTLBuffer> x1=obj.x1,inner=obj.inner,scratch=obj.scratch,x0=obj.x0;
    const uint32_t shapes[][2]={{256,1},{4352,33},{5120,4097},{5120,4111},{5120,17408},{17408,5120}};
    int ok=1;
    for (int shape=0;ok && shape<6;shape++) for (int kind=0;ok && kind<2;kind++) {
        uint32_t n=shapes[shape][0],m=shapes[shape][1],outputs=kind ? m : m*2;
        obj.x1=[g_device newBufferWithLength:n*sizeof(float) options:MTLResourceStorageModeShared];
        obj.inner=[g_device newBufferWithLength:n*sizeof(float) options:MTLResourceStorageModeShared];
        obj.scratch=[g_device newBufferWithLength:(outputs+16)*sizeof(float) options:MTLResourceStorageModeShared];
        obj.x0=[g_device newBufferWithLength:(outputs+16)*sizeof(float) options:MTLResourceStorageModeShared];
        float *a=malloc((outputs+16)*sizeof(float));
        if (!obj.x1 || !obj.inner || !obj.scratch || !obj.x0 || !a) {free(a);ok=0;break;}
        for (uint32_t i=0;i<n;i++) {
            ((float *)obj.x1.contents)[i]=sinf(i*.17f)*.03f;
            ((float *)obj.inner.contents)[i]=cosf(i*.11f)*.02f;
        }
        for (int v=0;ok && v<2;v++) {
            for (uint32_t i=0;i<outputs+16;i++) {
                ((float *)obj.scratch.contents)[i]=121.75f;
                ((float *)obj.x0.contents)[i]=i<outputs ? sinf(i*.13f)*.05f : 121.75f;
            }
            probe_select(v);
            ok=kind ? qw3_metal_session_matvec_iq4_xs_inner_add_x0(s->metal,down->offset,n,m) :
                qw3_metal_session_matvec_iq4_xs_pair_x1_to_scratch(s->metal,gate->offset,up->offset,n,m);
            float *data=kind ? obj.x0.contents : obj.scratch.contents;
            if (v==0) memcpy(a,data,(outputs+16)*sizeof(float));
            for (uint32_t i=0;ok && i<outputs+16;i++) {
                if (!isfinite(data[i]) || (v && data[i]!=a[i]) || (i>=outputs && data[i]!=121.75f)) ok=0;
            }
        }
        free(a);
        if (ok) printf("quant_exact kind=%s n=%u m=%u guards=1\n",kind ? "add" : "pair",n,m);
    }
    obj.x1=x1;obj.inner=inner;obj.scratch=scratch;obj.x0=x0;probe_select(0);
    return ok;
}

int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=3 || (strcmp(argv[2],"ffn") && strcmp(argv[2],"gdn"))) return 2;
        int mode=!strcmp(argv[2],"ffn") ? 1 : 2,rc=1;
        qw3_engine *e=NULL;qw3_session *s[2]={NULL,NULL};qw3_tokens prompt={0};char err[512]={0};
        qw3_engine_options options={.model_path=argv[1],.backend=QW3_BACKEND_METAL};
        if (qw3_engine_open(&e,&options)) goto done;
        for (int v=0;v<2;v++) if (qw3_session_create(&s[v],e,512) || s[v]->metal_n_gpu_layers!=64) goto done;
        if (mode==1 && !quant_checks(e,s[0])) goto done;
        uint32_t rng=1;int vocab=e->model.spec->n_vocab;
        for (int i=0;i<159;i++) {
            rng=rng*1664525u+1013904223u;int token=rng%(uint32_t)vocab;
            if (token==qw3_token_eos(e)) token=(token+1)%vocab;
            qw3_tokens_push(&prompt,token);
        }
        int lengths[]={16,31,128};
        for (int trial=0;trial<(mode==1 ? 1 : 3);trial++) {
            const int count=mode==1 ? 128 : lengths[trial];
            qw3_tokens prefix={.v=prompt.v,.len=count,.cap=prompt.cap};
            for (int v=0;v<2;v++) {
                probe_select(v ? mode : 0);qw3_session_invalidate(s[v]);
                if (qw3_session_sync(s[v],&prefix,err,sizeof(err))) goto done;
            }
            if (!logits_equal(s,vocab) || !states_equal(s)) goto done;
            if (mode==2 && count==128) {
                prefix.len=159;
                for (int v=0;v<2;v++) {
                    probe_select(v ? mode : 0);
                    if (qw3_session_sync(s[v],&prefix,err,sizeof(err))) goto done;
                }
                if (!logits_equal(s,vocab) || !states_equal(s)) goto done;
            }
            for (int step=0;step<32;step++) {
                int token=qw3_session_argmax(s[0]);
                if (token<0 || token!=qw3_session_argmax(s[1])) goto done;
                for (int v=0;v<2;v++) {
                    probe_select(v ? mode : 0);
                    if (qw3_session_eval_gpu_logits(s[v],token,err,sizeof(err))) goto done;
                }
                if (!logits_equal(s,vocab)) goto done;
            }
            if (!states_equal(s)) goto done;
            printf("full_exact mode=%s prefix=%d continuation=32 logits=exact live_kv=exact deltanet=exact conv=exact\n",argv[2],prefix.len);
            fflush(stdout);
        }
        if ((mode==1 && (!probe_pair_calls || !probe_add_calls)) || (mode==2 && !probe_gdn_calls)) goto done;
        printf("PASS mode=%s pair_calls=%llu add_calls=%llu gdn_calls=%llu\n",argv[2],
            (unsigned long long)probe_pair_calls,(unsigned long long)probe_add_calls,(unsigned long long)probe_gdn_calls);
        rc=0;
done:
        if (rc) fprintf(stderr,"candidate quality failed mode=%s error=%s\n",argv[2],err);
        for (int v=0;v<2;v++) if (s[v]) qw3_session_free(s[v]);
        if (e) qw3_engine_close(e);qw3_tokens_free(&prompt);return rc;
    }
}
