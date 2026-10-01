/* Private experiment harness: normal CLI/agent paths are not modified. */
#include "../qw3_metal.m"
#define qw3_metal_env_n_gpu_layers qw3_stream_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled qw3_stream_core_llama_split_enabled
#define qw3_metal_llama_split_start qw3_stream_core_llama_split_start
#include "../qw3.c"
#include <inttypes.h>

static id<MTLComputePipelineState> stream_base,stream_pair_base,stream_pipelines[6];
static const char *stream_names[6]={"iq3_bytes","iq3_words","iq3_prefetch",
                                  "iq3_float_prefetch","iq3_float_words","iq3s_bytes"};
static void stream_select(int selected,int variant) {
    g_matvec_iq3_xxs_pipeline=variant && selected<5 ? stream_pipelines[selected] : stream_base;
    g_matvec_iq3_s_pair_pipeline=variant && selected==5 ? stream_pipelines[selected] : stream_pair_base;
}
static double stream_now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+t.tv_nsec*1e-9;
}
static int stream_compare(const float *a,const float *b,uint32_t n,
                          double *maximum,double *rms) {
    *maximum=0;*rms=INFINITY;double ss=0;
    for (uint32_t i=0;i<n;i++) {
        if (!isfinite(a[i]) || !isfinite(b[i])) return 0;
        const double d=fabs((double)a[i]-b[i]);
        if (d>*maximum) *maximum=d;
        ss+=d*d;
    }
    *rms=sqrt(ss/n);
    return *maximum<=0.0001 && *rms<=0.00001;
}
static int stream_initialize(void) {
    if (!qw3_metal_compile_kernels()) return 0;
    stream_base=g_matvec_iq3_xxs_pipeline;
    stream_pair_base=g_matvec_iq3_s_pair_pipeline;
    NSError *error=nil;
    NSString *source=[NSString stringWithContentsOfFile:@"tests/iq3_decode_candidates.metal"
                                             encoding:NSUTF8StringEncoding error:&error];
    MTLCompileOptions *options=[[MTLCompileOptions alloc] init];
    options.mathMode=MTLMathModeFast;
    id<MTLLibrary> library=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int i=0;i<6;i++) {
        id<MTLFunction> fn=[library newFunctionWithName:[NSString stringWithUTF8String:stream_names[i]]];
        stream_pipelines[i]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!stream_pipelines[i] || stream_pipelines[i].maxTotalThreadsPerThreadgroup<64) {
            fprintf(stderr,"stream pipeline unavailable: %s %s\n",stream_names[i],
                    error ? error.localizedDescription.UTF8String : "");return 0;
        }
    }
    return 1;
}
static double stream_cpu_iq3s_dot(const uint8_t *src,const float *x,uint32_t ni) {
    double sum=0;
    for (uint32_t b=0;b<ni/256u;b++) {
        const uint8_t *blk=src+(uint64_t)b*110u;
        const float d=qw3_f16_to_f32(qw3_load_u16(blk));
        for (uint32_t ib=0;ib<8u;ib++) {
            const float db=d*(1u+2u*((blk[106u+ib/2u]>>(4u*(ib&1u)))&15u));
            for (uint32_t l=0;l<4u;l++) {
                const uint32_t qhb=blk[66u+ib],s=blk[74u+ib*4u+l];
                uint8_t lo[4],hi[4];
                iq3s_grid4(blk[2u+ib*8u+2u*l]|((qhb<<(8u-2u*l))&256u),lo);
                iq3s_grid4(blk[3u+ib*8u+2u*l]|((qhb<<(7u-2u*l))&256u),hi);
                for (uint32_t j=0;j<4u;j++) {
                    const uint32_t index=b*256u+ib*32u+l*8u+j;
                    const float wl=db*lo[j]*((s&(1u<<j)) ? -1.0f : 1.0f);
                    const float wh=db*hi[j]*((s&(1u<<(j+4u))) ? -1.0f : 1.0f);
                    sum+=(double)wl*x[index];sum+=(double)wh*x[index+4u];
                }
            }
        }
    }
    return sum;
}
static int stream_pair_micro(qw3_engine *engine) {
    const qw3_tensor *weights[2]={NULL,NULL};
    for (int il=0;il<engine->model.spec->n_layer && !weights[0];il++) {
        const qw3_layer_weights *lw=&engine->weights.layer[il];
        if (lw->ffn_gate->type==QW3_TENSOR_IQ3_S && lw->ffn_up->type==QW3_TENSOR_IQ3_S) {
            weights[0]=lw->ffn_gate;weights[1]=lw->ffn_up;
        }
    }
    id<MTLBuffer> grid=qw3_metal_iq3s_kgrid_buffer();
    if (!weights[0] || !grid) {fprintf(stderr,"no IQ3_S gate/up pair in this model\n");return 0;}
    const uint32_t shapes[][2]={{256,1},{4352,33},{5120,4097},{5120,4111},{5120,17408},{17408,5120}};
    for (int shape=0;shape<6;shape++) {
        uint32_t ni=shapes[shape][0],no=shapes[shape][1];
        if ((uint64_t)ni*no>weights[0]->elements || (uint64_t)ni*no>weights[1]->elements) return 0;
        uint64_t offsets[2];id<MTLBuffer> wb[2];
        for (int v=0;v<2;v++) wb[v]=qw3_metal_model_view_for(weights[v]->offset,(uint64_t)ni/256u*110u*no,&offsets[v]);
        id<MTLBuffer> x=[g_device newBufferWithLength:(ni+8u)*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputs[2];
        for (int v=0;v<2;v++) outputs[v]=[g_device newBufferWithLength:(2u*no+12u)*sizeof(float) options:MTLResourceStorageModeShared];
        if (!wb[0] || !wb[1] || !x || !outputs[0] || !outputs[1]) {fprintf(stderr,"pair buffers unavailable shape=%d\n",shape);return 0;}
        float *input=x.contents;
        for (uint32_t i=0;i<ni+8u;i++) input[i]=((int)((i*37u)%101u)-50)/31.0f;
        double elapsed[2]={0},gpu[2]={0};
        for (int repeat=-4;repeat<40;repeat++) for (int order=0;order<2;order++) {
            int v=order^(repeat&1);double start=stream_now();
            float *out=outputs[v].contents;
            if (repeat==-4) for (uint32_t i=0;i<2u*no+12u;i++) out[i]=123456.0f;
            struct {uint32_t ni,no,rb;} args={ni,no,ni/256u*110u};
            int owned=0;id<MTLCommandBuffer> cb=qw3_metal_command_buffer(&owned);
            id<MTLComputeCommandEncoder> enc=qw3_metal_compute_encoder(cb);
            [enc setComputePipelineState:v ? stream_pipelines[5] : stream_pair_base];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:wb[0] offset:offsets[0] atIndex:1];
            [enc setBuffer:wb[1] offset:offsets[1] atIndex:2];
            [enc setBuffer:x offset:16 atIndex:3];
            [enc setBuffer:outputs[v] offset:20 atIndex:4];
            [enc setBuffer:grid offset:0 atIndex:5];
            [enc setThreadgroupMemoryLength:128 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((no+7u)/8u,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            qw3_metal_end_compute_encoder(cb,enc);
            if (!qw3_metal_finish_command_buffer(cb,owned,"IQ3_S pair stream") || cb.status==MTLCommandBufferStatusError) {fprintf(stderr,"pair dispatch failed shape=%d repeat=%d variant=%d\n",shape,repeat,v);return 0;}
            if (repeat>=0) {elapsed[v]+=stream_now()-start;gpu[v]+=cb.GPUEndTime-cb.GPUStartTime;}
        }
        for (int v=0;v<2;v++) {
            float *out=outputs[v].contents;
            for (uint32_t i=0;i<2u*no+12u;i++) if ((i<5u || i>=2u*no+5u) && out[i]!=123456.0f) {
                fprintf(stderr,"pair guard failed shape=%d variant=%d index=%u value=%.9g\n",shape,v,i,out[i]);return 0;
            }
        }
        double maximum,rms;
        if (!stream_compare((float *)outputs[0].contents+5,(float *)outputs[1].contents+5,2u*no,&maximum,&rms)) {
            fprintf(stderr,"pair comparison failed n_in=%u n_out=%u max=%.9g rms=%.9g baseline_first=%.9g candidate_first=%.9g\n",
                    ni,no,maximum,rms,((float *)outputs[0].contents)[5],((float *)outputs[1].contents)[5]);return 0;
        }
        double cpu_max=0;
        const uint32_t rows[]={0u,no/2u,no-1u};
        for (int w=0;w<2;w++) {
            for (int r=0;r<3;r++) {
                const uint8_t *src=tensor_data(&engine->model,weights[w])+(uint64_t)rows[r]*ni/256u*110u;
                const double sum=stream_cpu_iq3s_dot(src,input+4u,ni);
                double d=fabs(sum-((float *)outputs[1].contents)[5u+w*no+rows[r]]);
                if (!isfinite(sum) || !isfinite(d)) return 0;
                if (d>cpu_max) cpu_max=d;
            }
        }
        printf("stream_quant mode=iq3s_bytes n_in=%u n_out=%u max=%.9g rms=%.9g cpu_max=%.9g guards=ok\n",ni,no,maximum,rms,cpu_max);
        if (cpu_max>0.0001) return 0;
        printf("stream_micro mode=iq3s_bytes n_in=%u n_out=%u baseline_ms=%.4f candidate_ms=%.4f gain_pct=%.2f gpu_base_ms=%.4f gpu_candidate_ms=%.4f\n",
               ni,no,elapsed[0]*25,elapsed[1]*25,(elapsed[0]/elapsed[1]-1)*100,gpu[0]*25,gpu[1]*25);
        fflush(stdout);
    }
    return 1;
}
static int stream_dispatch(id<MTLComputePipelineState> pipeline,id<MTLBuffer> weight,
    uint64_t offset,id<MTLBuffer> x,id<MTLBuffer> out,id<MTLBuffer> grid,id<MTLBuffer> signs,
    uint32_t ni,uint32_t no,double *gpu) {
    struct {uint32_t ni,no,rb;} args={ni,no,ni/256u*98u};
    int owned=0; id<MTLCommandBuffer> cb=qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc=qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:weight offset:offset atIndex:1];
    [enc setBuffer:x offset:16 atIndex:2];
    [enc setBuffer:out offset:20 atIndex:3];
    [enc setBuffer:grid offset:0 atIndex:4];
    [enc setBuffer:signs offset:0 atIndex:5];
    [enc setThreadgroupMemoryLength:128 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((no+3u)/4u,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    qw3_metal_end_compute_encoder(cb,enc);
    if (!qw3_metal_finish_command_buffer(cb,owned,"IQ3 stream probe") || cb.status==MTLCommandBufferStatusError) return 0;
    if (gpu) *gpu=cb.GPUEndTime-cb.GPUStartTime;
    return 1;
}
static int stream_micro(qw3_engine *engine,int selected) {
    if (selected==5) return stream_pair_micro(engine);
    const qw3_tensor *weight=NULL;
    for (int layer=0;layer<engine->model.spec->n_layer && !weight;layer++) {
        const qw3_layer_weights *lw=&engine->weights.layer[layer];
        const qw3_tensor *possibilities[]={lw->ffn_gate,lw->ffn_up,lw->ffn_down};
        for (int p=0;p<3;p++) if (possibilities[p] && possibilities[p]->type==QW3_TENSOR_IQ3_XXS) {
            weight=possibilities[p];break;
        }
    }
    id<MTLBuffer> grid=nil,signs=nil;
    if (!weight || !qw3_metal_dense_iq_buffers(&signs,NULL,NULL,&grid)) return 0;
    const uint32_t shapes[][2]={{256,1},{4352,33},{5120,4097},{5120,4111},{5120,17408},{17408,5120}};
    for (int shape=0;shape<6;shape++) {
        uint32_t ni=shapes[shape][0],no=shapes[shape][1];
        if ((uint64_t)ni*no>weight->elements) return 0;
        uint64_t offset=0;
        id<MTLBuffer> wb=qw3_metal_model_view_for(weight->offset,(uint64_t)ni/256u*98u*no,&offset);
        id<MTLBuffer> x=[g_device newBufferWithLength:(ni+8u)*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputs[2];
        for (int v=0;v<2;v++) outputs[v]=[g_device newBufferWithLength:(no+12u)*sizeof(float) options:MTLResourceStorageModeShared];
        if (!wb || !x || !outputs[0] || !outputs[1]) return 0;
        float *input=x.contents;
        for (uint32_t i=0;i<ni+8u;i++) input[i]=((int)((i*37u)%101u)-50)/31.0f;
        for (int mode=0;mode<5;mode++) {
            if (selected>=0 && mode!=selected) continue;
            for (int v=0;v<2;v++) {
                float *out=outputs[v].contents;
                for (uint32_t i=0;i<no+12u;i++) out[i]=123456.0f;
                if (!stream_dispatch(v ? stream_pipelines[mode] : stream_base,wb,offset,x,outputs[v],grid,signs,ni,no,NULL)) return 0;
                for (uint32_t i=0;i<no+12u;i++) if ((i<5u || i>=no+5u) && out[i]!=123456.0f) return 0;
            }
            double maximum,rms;
            if (!stream_compare((float *)outputs[0].contents+5,(float *)outputs[1].contents+5,no,&maximum,&rms)) return 0;
            qw3_tensor view=*weight;view.dim[0]=ni;view.dim[1]=no;view.elements=(uint64_t)ni*no;
            float *row=malloc(ni*sizeof(float));if (!row) return 0;
            const uint32_t rows[]={0u,no/2u,no-1u};double cpu_max=0;
            for (int r=0;r<3;r++) {
                if (!tensor_read_dense_row(&engine->model,&view,rows[r],row)) {free(row);return 0;}
                double sum=0;
                for (uint32_t j=0;j<ni;j++) sum+=(double)row[j]*input[4u+j];
                double d=fabs(sum-((float *)outputs[1].contents)[5u+rows[r]]);
                if (!isfinite(sum) || !isfinite(d)) {free(row);return 0;}
                if (d>cpu_max) cpu_max=d;
            }
            free(row);
            printf("stream_quant mode=%s n_in=%u n_out=%u max=%.9g rms=%.9g cpu_max=%.9g guards=ok\n",stream_names[mode],ni,no,maximum,rms,cpu_max);
            if (cpu_max>0.0001) return 0;
            if (shape<2) continue;
            double elapsed[2]={0},gpu[2]={0};
            for (int repeat=-4;repeat<40;repeat++) for (int order=0;order<2;order++) {
                int v=order ^ (repeat&1);double device_time=0,start=stream_now();
                if (!stream_dispatch(v ? stream_pipelines[mode] : stream_base,wb,offset,x,outputs[v],grid,signs,ni,no,&device_time)) return 0;
                if (repeat>=0) {elapsed[v]+=stream_now()-start;gpu[v]+=device_time;}
            }
            printf("stream_micro mode=%s n_in=%u n_out=%u baseline_ms=%.4f candidate_ms=%.4f gain_pct=%.2f gpu_base_ms=%.4f gpu_candidate_ms=%.4f\n",
                   stream_names[mode],ni,no,elapsed[0]*25,elapsed[1]*25,(elapsed[0]/elapsed[1]-1)*100,gpu[0]*25,gpu[1]*25);
            fflush(stdout);
        }
    }
    return selected<0 ? stream_pair_micro(engine) : 1;
}
int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=2 && argc!=4) {fprintf(stderr,"usage: %s MODEL [--full MODE]\n",argv[0]);return 1;}
        int selected=-1;
        if (argc==4) {
            if (strcmp(argv[2],"--full")) return 1;
            for (int m=0;m<6;m++) if (!strcmp(argv[3],stream_names[m])) selected=m;
            if (selected<0) return 1;
        }
        qw3_engine *engine=NULL;qw3_session *sessions[2]={NULL,NULL};qw3_tokens prompt={0};
        qw3_engine_options options={.model_path=argv[1],.backend=QW3_BACKEND_METAL};
        float *logits[2]={NULL,NULL};char error[512]={0};int status=1;
        if (qw3_engine_open(&engine,&options)!=0 || engine->model.spec->ffn_kind!=QW3_FFN_DENSE || !stream_initialize()) goto done;
        for (int type=0;type<32;type++) {
            int gate=0,up=0,down=0;
            for (int il=0;il<engine->model.spec->n_layer;il++) {
                const qw3_layer_weights *lw=&engine->weights.layer[il];
                gate+=lw->ffn_gate->type==(uint32_t)type;
                up+=lw->ffn_up->type==(uint32_t)type;
                down+=lw->ffn_down->type==(uint32_t)type;
            }
            if (gate+up+down) printf("stream_model type=%d gate_layers=%d up_layers=%d down_layers=%d\n",type,gate,up,down);
        }
        fflush(stdout);
        if (!stream_micro(engine,selected)) goto done;
        if (selected<0) {status=0;goto done;}
        const uint32_t vocab=engine->model.spec->n_vocab;
        qw3_encode_chat_prompt(engine,NULL,"Spiega dettagliatamente come implementare una tabella hash in C.",QW3_THINK_NONE,&prompt);
        for (int v=0;v<2;v++) {
            stream_select(selected,v);
            logits[v]=calloc(vocab,sizeof(float));
            if (!logits[v] || qw3_session_create(&sessions[v],engine,512)!=0 ||
                sessions[v]->metal_n_gpu_layers!=engine->model.spec->n_layer ||
                qw3_session_sync(sessions[v],&prompt,error,sizeof(error))!=0) goto done;
        }
        uint64_t rng[2]={42,42},hash=1469598103934665603ULL;
        double forward[2]={0},sampling[2]={0},block[2]={0},worst_max=0,worst_rms=0;
        for (int i=0;i<128;i++) {
            int tokens[2];
            for (int v=0;v<2;v++) {
                const qw3_tokens *history=qw3_session_tokens(sessions[v]);
                const int count=history->len>1024 ? 1024 : history->len;
                const double start=stream_now();
                tokens[v]=qw3_session_sample_repetition(sessions[v],0.6f,20,0.95f,0,&rng[v],history->v+history->len-count,count,1.06f);
                if (i>=8) sampling[v]+=stream_now()-start;
            }
            if (tokens[0]<0 || tokens[0]!=tokens[1]) {fprintf(stderr,"token mismatch at %d: %d/%d\n",i,tokens[0],tokens[1]);goto done;}
            hash=(hash^(uint32_t)tokens[0])*1099511628211ULL;
            for (int order=0;order<2;order++) {
                const int v=order^(i&1);
                stream_select(selected,v);
                const double start=stream_now();
                if (qw3_session_eval_gpu_logits(sessions[v],tokens[v],error,sizeof(error))!=0) goto done;
                if (i>=8) {double t=stream_now()-start;forward[v]+=t;block[v]+=t;}
            }
            // All vocabulary logits at every step, not just matching sampled IDs.
            for (int v=0;v<2;v++) if (qw3_session_copy_logits(sessions[v],logits[v],vocab)!=(int)vocab) goto done;
            double maximum=INFINITY,rms=INFINITY;
            if (!stream_compare(logits[0],logits[1],vocab,&maximum,&rms)) {fprintf(stderr,"full logits differ at %d: %.9g/%.9g\n",i,maximum,rms);goto done;}
            if (maximum>worst_max) worst_max=maximum;
            if (rms>worst_rms) worst_rms=rms;
            if (i>=8 && ((i-7)%16==0 || i==127)) {
                printf("stream_block end=%d baseline_ms=%.3f candidate_ms=%.3f gain_pct=%.2f\n",i+1,block[0]*1000,block[1]*1000,(block[0]/block[1]-1)*100);
                block[0]=block[1]=0;fflush(stdout);
            }
        }
        printf("stream_full mode=%s tokens=128 timed=120 baseline=%.3f candidate=%.3f gain_pct=%.2f hash=%" PRIx64 " max=%.9g rms=%.9g\n",
               stream_names[selected],120/(forward[0]+sampling[0]),120/(forward[1]+sampling[1]),(forward[0]/forward[1]-1)*100,hash,worst_max,worst_rms);
        status=0;
done:
        g_matvec_iq3_xxs_pipeline=stream_base;
        g_matvec_iq3_s_pair_pipeline=stream_pair_base;
        if (status) fprintf(stderr,"stream benchmark failed: %s\n",error);
        for (int v=0;v<2;v++) {free(logits[v]);if (sessions[v]) qw3_session_free(sessions[v]);}
        qw3_tokens_free(&prompt);if (engine) qw3_engine_close(engine);
        return status;
    }
}
