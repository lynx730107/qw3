/* Alternating all-row verifier A/B: same precise backbone, proposals and
 * initial checkpoint. Only head implementation differs. Not MTP throughput. */
#define QW3_MTP_DECODE_ENTRYPOINT qw3_mtp_decoder_probe_entrypoint
#include "bench_mtp_decode.m"

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc<2 || argc>4) {
            fprintf(stderr,"usage: %s MODEL [PAIRS=5] [BLOCK=8]\n",argv[0]); return 1;
        }
        const int pairs=argc>2 ? decode_int(argv[2],1,20) : 5;
        const int n=argc>3 ? decode_int(argv[3],2,8) : 8;
        if (pairs<0 || n<0) return 1;
        qw3_engine *e=NULL;
        qw3_session *warm=NULL, *s[2]={NULL,NULL};
        qw3_tokens prompt={0};
        NSArray<id<MTLBuffer>> *live[2]={nil,nil}, *backup=nil;
        id<MTLBuffer> log_host[2]={nil,nil}, h_host[2]={nil,nil};
        float *logits[2]={NULL,NULL}, *hidden[2]={NULL,NULL};
        int tokens[8], status=1;
        double elapsed[2]={0}, worst_max=0, worst_rms=0;
        char error[512]={0};
        qw3_engine_options options={.model_path=argv[1],.backend=QW3_BACKEND_METAL};
        decode_short_f32=1; decode_short_vector=3; decode_batch_head=1;
        decode_cached_gdn=0; decode_head_group=4;
        if (getenv("QW3_MTP_HEAD_GROUP")) {
            const int group=decode_int(getenv("QW3_MTP_HEAD_GROUP"),1,4);
            if (group!=1 && group!=2 && group!=4) goto done;
            decode_head_group=(uint32_t)group;
        }
        if (qw3_engine_open(&e,&options)!=0 || !qw3_metal_compile_kernels() || !decode_short_init() ||
            e->model.spec->ffn_kind!=QW3_FFN_DENSE || e->weights.output->type!=QW3_TENSOR_Q5_K) goto done;
        const int d=e->model.spec->n_embd, v=e->model.spec->n_vocab;
        qw3_encode_chat_prompt(e,NULL,"Spiega dettagliatamente come implementare una tabella hash in C.",QW3_THINK_NONE,&prompt);
        if (prompt.len<1 || prompt.len+n+1>512 || qw3_session_create(&warm,e,512)!=0 ||
            warm->metal_n_gpu_layers!=e->model.spec->n_layer ||
            qw3_session_sync(warm,&prompt,error,sizeof(error))!=0) goto done;
        backup=decode_backup(decode_state_buffers(warm->metal));
        if (!backup || !decode_copy_state(decode_state_buffers(warm->metal),backup)) goto done;
        for (int t=0; t<n; t++) {
            tokens[t]=qw3_session_argmax(warm);
            if (tokens[t]<0 || qw3_session_eval_gpu_logits(warm,tokens[t],error,sizeof(error))!=0) goto done;
        }
        for (int side=0; side<2; side++) {
            if (qw3_session_create(&s[side],e,512)!=0) goto done;
            live[side]=decode_state_buffers(s[side]->metal);
            for (int t=0; t<prompt.len; t++) token_vec_push(&s[side]->tokens,prompt.v[t]);
            log_host[side]=[g_device newBufferWithLength:(NSUInteger)n*v*sizeof(float) options:MTLResourceStorageModeShared];
            h_host[side]=[g_device newBufferWithLength:(NSUInteger)n*d*sizeof(float) options:MTLResourceStorageModeShared];
            logits[side]=calloc((size_t)n*v,sizeof(float));
            hidden[side]=calloc((size_t)n*d,sizeof(float));
            if (!log_host[side] || !h_host[side] || !logits[side] || !hidden[side]) goto done;
        }
        qw3_session_free(warm); warm=NULL;
        printf("mtp_head_verify block=%d pairs=%d group=%u all_rows=1 same_backbone=1 not_speculative_decode=1\n",n,pairs,decode_head_group);
        for (int trial=-1; trial<pairs; trial++) {
            double pair[2]={0};
            for (int order=0; order<2; order++) {
                const int side=order^(trial>=0 ? trial&1 : 0);
                if (!decode_copy_state(backup,live[side])) goto done;
                decode_rewind(s[side],prompt.len);
                decode_batch_head=side;
                const double start=qw3_now_sec();
                if (!decode_verify(s[side],tokens,n,1,log_host[side],h_host[side],
                    logits[side],hidden[side],error,sizeof(error))) goto done;
                for (int t=0; t<n; t++) if (decode_top(logits[side]+(size_t)t*v,v)<0) goto done;
                pair[side]=qw3_now_sec()-start;
                if (trial>=0) elapsed[side]+=pair[side];
                if (s[side]->kv.pos!=prompt.len+n || s[side]->tokens.len!=prompt.len+n ||
                    memcmp(s[side]->tokens.v,prompt.v,prompt.len*sizeof(int)) ||
                    memcmp(s[side]->tokens.v+prompt.len,tokens,n*sizeof(int))) goto done;
            }
            double maximum=0, rms=0;
            if (!decode_vector_error(logits[0],logits[1],n*v,&maximum,&rms) || maximum>0.0001 || rms>0.00001 ||
                memcmp(hidden[0],hidden[1],(size_t)n*d*sizeof(float))) goto done;
            for (int t=0; t<n; t++) if (decode_top(logits[0]+(size_t)t*v,v)!=decode_top(logits[1]+(size_t)t*v,v)) goto done;
            if (maximum>worst_max) worst_max=maximum;
            if (rms>worst_rms) worst_rms=rms;
            /* Check installed last logits, then ordinary-decode continuation.
             * This entire comparison is outside the alternating timed region. */
            for (int side=0; side<2; side++) {
                if (qw3_session_copy_logits(s[side],logits[side],v)!=v ||
                    memcmp(logits[side],(const float *)log_host[side].contents+(size_t)(n-1)*v,v*sizeof(float))) goto done;
            }
            const int next=decode_top(logits[0],v);
            if (next<0 || next!=decode_top(logits[1],v)) goto done;
            for (int side=0; side<2; side++) if (qw3_session_eval_gpu_logits(s[side],next,error,sizeof(error))!=0 ||
                qw3_session_copy_logits(s[side],logits[side],v)!=v) goto done;
            if (memcmp(logits[0],logits[1],v*sizeof(float))) goto done;
            printf("mtp_head_verify_pair trial=%d scalar_ms=%.3f shared_ms=%.3f max=%.9g rms=%.9g hidden_exact=1 continuation_exact=1\n",
                trial,pair[0]*1000,pair[1]*1000,maximum,rms);
            fflush(stdout);
        }
        printf("mtp_head_verify_summary pairs=%d block=%d scalar_ms=%.3f shared_ms=%.3f ratio=%.3f max=%.9g rms=%.9g quality_ok=1 all_rows=1 not_speculative_decode=1\n",
            pairs,n,elapsed[0]*1000/pairs,elapsed[1]*1000/pairs,elapsed[0]/elapsed[1],worst_max,worst_rms);
        status=0;
done:
        if (status) fprintf(stderr,"MTP head paired probe failed: %s\n",error);
        if (e) (void)qw3_metal_synchronize();
        if (warm) qw3_session_free(warm);
        for (int side=0; side<2; side++) {
            if (s[side]) qw3_session_free(s[side]);
            free(logits[side]); free(hidden[side]);
        }
        if (e) qw3_engine_close(e);
        qw3_tokens_free(&prompt);
        return status;
    }
}
