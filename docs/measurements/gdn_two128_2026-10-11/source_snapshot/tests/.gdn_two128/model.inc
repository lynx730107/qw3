/* Low-memory screen: one reused session, sequential baseline/cache32/cache64
 * passes, and a disk oracle. This is not the eight-prefix checkpoint suite. */

static int grid_oracle_blob(FILE *oracle,int record,const void *data,size_t bytes) {
    if (record) return fwrite(data,1,bytes,oracle)==bytes;
    unsigned char expected[65536];
    if (bytes>sizeof(expected) || fread(expected,1,bytes,oracle)!=bytes) return 0;
    return !memcmp(expected,data,bytes);
}

static int grid_oracle_logits(FILE *oracle,int record,qw3_session *s,int vocab) {
    float data[16384];
    for (int off=0;off<vocab;off+=16384) {
        int count=MIN(16384,vocab-off);
        if (!qw3_metal_session_read_state(s->metal,QW3_METAL_STATE_LOGITS,0,
                (uint64_t)off*4u,data,(uint64_t)count*4u)) return 0;
        for (int i=0;i<count;i++) if (!isfinite(data[i])) return 0;
        if (!grid_oracle_blob(oracle,record,data,(size_t)count*4u)) {
            fprintf(stderr,"GDN two128 logit mismatch offset=%d\n",off);return 0;
        }
    }
    return 1;
}

static int grid_oracle_state(FILE *oracle,int record,qw3_session *s) {
    qw3_metal_state_info info={0};unsigned char data[65536];
    if (!qw3_metal_session_get_state_info(s->metal,&info) || info.kv_type!=1u) return 0;
    int32_t header[2]={s->tokens.len,s->kv.pos};
    if (!grid_oracle_blob(oracle,record,header,sizeof(header)) ||
        s->tokens.len>256 || s->kv.pos!=s->tokens.len ||
        !grid_oracle_blob(oracle,record,s->tokens.v,(size_t)s->tokens.len*sizeof(int))) return 0;
    for (int kind=QW3_METAL_STATE_GQA_K;kind<=QW3_METAL_STATE_CONV;kind++) {
        if (kind==QW3_METAL_STATE_GQA_FLASH_TAIL) continue;
        uint64_t bytes=kind==QW3_METAL_STATE_DELTANET ? info.deltanet_bytes :
            kind==QW3_METAL_STATE_CONV ? info.conv_bytes : info.kv_row_bytes*s->tokens.len;
        uint32_t layers=kind<=QW3_METAL_STATE_GQA_V ? info.n_full_layers : 1u;
        for (uint32_t l=0;l<layers;l++) for (uint64_t off=0;off<bytes;off+=sizeof(data)) {
            size_t n=(size_t)MIN((uint64_t)sizeof(data),bytes-off);
            if (!qw3_metal_session_read_state(s->metal,(qw3_metal_state_buffer)kind,l,off,data,n) ||
                !grid_oracle_blob(oracle,record,data,n)) {
                fprintf(stderr,"GDN two128 state mismatch kind=%d layer=%u offset=%llu\n",
                    kind,l,(unsigned long long)off);return 0;
            }
        }
    }
    return 1;
}

int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc<2 || argc>3) return 2;
        int filter=argc==3 ? atoi(argv[2]) : 0;
        if (argc==3 && (filter<1 || filter>2)) return 2;
        qw3_engine *e=NULL;qw3_session *s=NULL;qw3_tokens prefix={0};FILE *oracle=NULL;
        int rc=1;char err[512]={0};uint32_t rng=1;
        qw3_engine_options opt={.model_path=argv[1],.backend=QW3_BACKEND_METAL,.n_threads=4};
        if (qw3_engine_open(&e,&opt) || !two128_compile() ||
            qw3_session_create(&s,e,256) || s->metal_n_gpu_layers!=64) goto done;
        int vocab=qw3_vocab_size(e);
        for (int i=0;i<128;i++) {
            rng=rng*1664525u+1013904223u;int token=rng%(uint32_t)vocab;
            if (token==qw3_token_eos(e)) token=(token+1)%vocab;
            qw3_tokens_push(&prefix,token);
        }
        oracle=tmpfile();if (!oracle) goto done;
        for (int phase=0;phase<=2;phase++) {
            if (phase && filter && phase!=filter) continue;
            if (phase && (fflush(oracle) || fseek(oracle,0,SEEK_SET))) goto done;
            two128_variant=phase ? phase : 1;scratch_select(phase!=0);
            qw3_session_invalidate(s);
            if (qw3_session_sync(s,&prefix,err,sizeof(err)) ||
                !grid_oracle_logits(oracle,phase==0,s,vocab) ||
                !grid_oracle_state(oracle,phase==0,s)) goto done;
            two128_calls=0;
            for (int step=0;step<32;step++) {
                int token=qw3_session_argmax(s),expected=token;
                if (token<0) goto done;
                if (phase==0) {
                    if (fwrite(&token,sizeof(token),1,oracle)!=1) goto done;
                } else if (fread(&expected,sizeof(expected),1,oracle)!=1 || expected!=token) goto done;
                if (qw3_session_eval_gpu_logits(s,token,err,sizeof(err)) ||
                    !grid_oracle_logits(oracle,phase==0,s,vocab)) goto done;
            }
            if (!grid_oracle_state(oracle,phase==0,s) ||
                two128_calls!=(phase ? 32u*48u : 0u)) goto done;
            if (phase && (fgetc(oracle)!=EOF || ferror(oracle))) goto done;
            printf("PASS gdn_two128_model phase=%d prefix=128 continuation=32 finite_logits=bit_exact canonical_state=exact history=exact calls=%llu sessions=1 additional_weight_buffer_bytes=0\n",
                phase,(unsigned long long)two128_calls);fflush(stdout);
        }
        rc=0;
done:
        if (rc) fprintf(stderr,"GDN two128 model failed: %s calls=%llu\n",err,(unsigned long long)two128_calls);
        if (oracle) fclose(oracle);
        if (s) qw3_session_free(s);if (e) qw3_engine_close(e);qw3_tokens_free(&prefix);return rc;
    }
}
