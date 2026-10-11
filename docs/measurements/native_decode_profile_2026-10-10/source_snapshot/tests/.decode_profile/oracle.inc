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

