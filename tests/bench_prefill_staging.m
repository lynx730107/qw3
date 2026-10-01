/* Paired pp128 vector-store probe. Production shaders stay unchanged. */
#include "../qw3_metal.m"
#define qw3_metal_env_n_gpu_layers qw3_memory_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled qw3_memory_core_llama_split_enabled
#define qw3_metal_llama_split_start qw3_memory_core_llama_split_start
#include "../qw3.c"

static id<MTLComputePipelineState> prefill_iq3[2],prefill_iq4[2];
static const char *prefill_mode="vector_store";
static int prefill_initialize(void) {
    if (!qw3_metal_compile_kernels() || !g_metal4_tensor_api_enabled) return 0;
    prefill_iq3[0]=g_matmul_iq3_s_mpp_staged128_pipeline;
    prefill_iq4[0]=g_matmul_iq4_xs_mpp_staged128_pipeline;
    NSMutableString *source=[qw3_metal_kernel_source() mutableCopy];
    NSString *old=@"#pragma unroll\n            for (int i = 0; i < 16; ++i)\n                sa[row * NK + half_k * 16 + i] =\n                    values[i / 4][i & 3];";
    NSString *replacement=@"threadgroup half4 *dst = (threadgroup half4 *)(sa + row * NK + half_k * 16);\n#pragma unroll\n            for (int i = 0; i < 4; ++i) dst[i] = values[i];";
    const NSUInteger changes=[source replaceOccurrencesOfString:old withString:replacement options:0 range:NSMakeRange(0,source.length)];
    if (changes!=4) {fprintf(stderr,"unexpected vector-store replacement count: %lu\n",(unsigned long)changes);return 0;}
    if (!strcmp(prefill_mode,"vector_dequant")) {
        NSRange start=[source rangeOfString:@"inline half4x4 qw3_iq3s_dequant4x4_expanded("];
        if (start.location==NSNotFound) return 0;
        NSRange end=[source rangeOfString:@"\n#ifdef QW3_METAL_HAS_TENSOR" options:0 range:NSMakeRange(start.location,source.length-start.location)];
        if (end.location==NSNotFound) return 0;
        NSString *body=@"inline half4x4 qw3_iq3s_dequant4x4_expanded(float dl, device const uchar *qs16, uchar qh16, device const uchar *signs16, device const uchar *kgrid) {\n"
            "    half4x4 reg;\n"
            "#pragma unroll\n"
            "    for (uint qn=0u; qn<4u; qn++) {\n"
            "        const uint idx=uint(qs16[qn]) | ((uint(qh16) << (8u-qn)) & 256u);\n"
            "        const uint s=uint(signs16[qn >> 1u]) >> (4u*(qn&1u));\n"
            "        const uchar4 g=*((device const uchar4 *)(kgrid+idx*4u));\n"
            "        const float4 sign=select(float4(1.0f),float4(-1.0f),(uint4(s)&uint4(1u,2u,4u,8u))!=0u);\n"
            "        reg[qn]=half4(dl*float4(g)*sign);\n"
            "    }\n"
            "    return reg;\n"
            "}\n";
        [source replaceCharactersInRange:NSMakeRange(start.location,end.location-start.location) withString:body];
    }
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    id<MTLLibrary> library=[g_device newLibraryWithSource:source options:options error:&error];
    id<MTLFunction> iq3=[library newFunctionWithName:@"qw3_matmul_iq3_s_mpp_staged128"];
    id<MTLFunction> iq4=[library newFunctionWithName:@"qw3_matmul_iq4_xs_mpp_staged128"];
    prefill_iq3[1]=iq3 ? [g_device newComputePipelineStateWithFunction:iq3 error:&error] : nil;
    prefill_iq4[1]=iq4 ? [g_device newComputePipelineStateWithFunction:iq4 error:&error] : nil;
    if (!prefill_iq3[1] || !prefill_iq4[1] || prefill_iq3[1].maxTotalThreadsPerThreadgroup<128 || prefill_iq4[1].maxTotalThreadsPerThreadgroup<128) {
        fprintf(stderr,"prefill candidate unavailable: %s\n",error ? error.localizedDescription.UTF8String : "");return 0;
    }
    return 1;
}
static void prefill_select(int v) {
    g_matmul_iq3_s_mpp_staged128_pipeline=prefill_iq3[v];
    g_matmul_iq4_xs_mpp_staged128_pipeline=prefill_iq4[v];
}

static int memory_logits_equal(qw3_session **s,float **logits,int vocab) {
    for (int v=0;v<2;v++) if (qw3_session_copy_logits(s[v],logits[v],vocab)!=vocab) return 0;
    for (int i=0;i<vocab;i++) if (!isfinite(logits[0][i]) || !isfinite(logits[1][i]) || logits[0][i]!=logits[1][i]) {
        fprintf(stderr,"memory logits mismatch at %d: %.9g/%.9g\n",i,logits[0][i],logits[1][i]);return 0;
    }
    for (int v=0;v<2;v++) if (!s[v]->valid || s[v]->kv.pos!=s[v]->tokens.len) return 0;
    return s[0]->tokens.len==s[1]->tokens.len &&
           !memcmp(s[0]->tokens.v,s[1]->tokens.v,s[0]->tokens.len*sizeof(int));
}
int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=2 && argc!=3) {fprintf(stderr,"usage: %s MODEL [vector_store|vector_dequant]\n",argv[0]);return 1;}
        if (argc==3) {if (strcmp(argv[2],"vector_store") && strcmp(argv[2],"vector_dequant")) return 1;prefill_mode=argv[2];}
        qw3_engine *engine=NULL;qw3_session *sessions[2]={NULL,NULL};qw3_tokens prompt={0};
        qw3_engine_options options={.model_path=argv[1],.backend=QW3_BACKEND_METAL};
        float *logits[2]={NULL,NULL};char error[512]={0};int status=1;
        if (qw3_engine_open(&engine,&options)!=0 || engine->model.spec->ffn_kind!=QW3_FFN_DENSE || !prefill_initialize()) goto done;
        const int vocab=engine->model.spec->n_vocab;uint32_t rng=1;
        for (int i=0;i<128;i++) {
            rng=rng*1664525u+1013904223u;
            int token=rng%(uint32_t)vocab;
            if (token==qw3_token_eos(engine)) token=(token+1)%vocab;
            qw3_tokens_push(&prompt,token);
        }
        for (int v=0;v<2;v++) {
            logits[v]=calloc(vocab,sizeof(float));
            if (!logits[v] || qw3_session_create(&sessions[v],engine,256)!=0 ||
                sessions[v]->metal_n_gpu_layers!=engine->model.spec->n_layer) goto done;
            qw3_metal_state_info info={0};
            if (!qw3_metal_session_get_state_info(sessions[v]->metal,&info) || info.kv_type!=1u) {
                fprintf(stderr,"prefill probe requires F16 KV\n");goto done;
            }
        }
        double elapsed[2]={0},pairs[8];
        for (int trial=-1;trial<8;trial++) {
            double times[2]={0};
            for (int order=0;order<2;order++) {
                const int v=order^(trial&1);
                prefill_select(v);
                qw3_session_invalidate(sessions[v]);
                const double start=qw3_now_sec();
                if (qw3_session_sync(sessions[v],&prompt,error,sizeof(error))!=0) goto done;
                times[v]=qw3_now_sec()-start;
                if (trial>=0) elapsed[v]+=times[v];
            }
            if (!memory_logits_equal(sessions,logits,vocab)) goto done;
            for (int step=0;step<8;step++) {
                const int token=qw3_session_argmax(sessions[0]);
                if (token<0 || token!=qw3_session_argmax(sessions[1])) goto done;
                for (int v=0;v<2;v++) {
                    prefill_select(v);
                    if (qw3_session_eval_gpu_logits(sessions[v],token,error,sizeof(error))!=0) goto done;
                }
                if (!memory_logits_equal(sessions,logits,vocab)) goto done;
            }
            if (trial>=0) {
                pairs[trial]=(times[0]-times[1])*1000;
                printf("prefill_pair mode=%s trial=%d baseline_ms=%.3f candidate_ms=%.3f gain_pct=%.2f logits_exact=1 continuation=8\n",
                       prefill_mode,trial,times[0]*1000,times[1]*1000,(times[0]/times[1]-1)*100);fflush(stdout);
            }
        }
        double mean=0,ss=0;
        for (int i=0;i<8;i++) mean+=pairs[i]/8;
        for (int i=0;i<8;i++) ss+=(pairs[i]-mean)*(pairs[i]-mean);
        printf("prefill_pp128 mode=%s pairs=8 baseline_tps=%.3f candidate_tps=%.3f gain_pct=%.2f saved_ms=%.3f ci95_ms=%.3f logits_exact=1\n",
               prefill_mode,8*128/elapsed[0],8*128/elapsed[1],(elapsed[0]/elapsed[1]-1)*100,mean,2.365*sqrt(ss/7/8));
        status=0;
done:
        if (prefill_iq3[0] && prefill_iq4[0]) prefill_select(0);
        if (status) fprintf(stderr,"prefill staging benchmark failed: %s\n",error);
        for (int v=0;v<2;v++) {free(logits[v]);if (sessions[v]) qw3_session_free(sessions[v]);}
        qw3_tokens_free(&prompt);if (engine) qw3_engine_close(engine);
        return status;
    }
}
