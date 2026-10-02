/* Private lookup/scheduling experiments. No production runtime selector. */
#define qw3_metal_synchronize lut_baseline_synchronize
#include "../qw3_metal.m"
#undef qw3_metal_synchronize
static int lut_observe;
static double lut_forward_start,lut_submit_ms,lut_gpu_ms,lut_wait_ms;
static double lut_clock(void) {return [NSDate timeIntervalSinceReferenceDate];}
int qw3_metal_synchronize(void) {
    if (!lut_observe) return lut_baseline_synchronize();
    id<MTLCommandBuffer> last=g_batch_cb;
    NSArray<id<MTLCommandBuffer>> *prefixes=[g_pending_cbs copy];
    const double start=lut_clock();
    lut_submit_ms=(start-lut_forward_start)*1000;
    const int ok=lut_baseline_synchronize();
    lut_wait_ms=(lut_clock()-start)*1000;
    lut_gpu_ms=last ? (last.GPUEndTime-last.GPUStartTime)*1000 : 0;
    for (id<MTLCommandBuffer> prefix in prefixes)
        lut_gpu_ms+=(prefix.GPUEndTime-prefix.GPUStartTime)*1000;
    return ok;
}
#define qw3_metal_env_n_gpu_layers qw3_lut_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled qw3_lut_core_llama_split_enabled
#define qw3_metal_llama_split_start qw3_lut_core_llama_split_start
static int lut_flush_interval;
static int lut_layer_done(int layer,int layers) {
    return !lut_flush_interval || layer+1==layers || (layer+1)%lut_flush_interval || qw3_metal_flush_commands();
}
#define QW3_METAL_DENSE_DECODE_LAYER_DONE(layer, layers) lut_layer_done(layer,layers)
#include "iq4_layout_probe.inc"
#include "iq4_prefetch_probe.inc"
#include "../qw3.c"
#include <inttypes.h>

static id<MTLComputePipelineState> lut_pipeline[3][3]; /* baseline, constant, shuffle */
static id<MTLComputePipelineState> lut_iq3_base,lut_iq3_words;
static const char *lut_modes[]={"shared","constant","shuffle","flush16","flush8","flush32","iq3s_words",
    "blocks4","blocks16","planes4","planes16","iq4_prefetch","native_copy"};
static const char *lut_names[]={"qw3_matvec_iq4_xs_fast","qw3_matvec_iq4_xs_pair_fast",
                              "qw3_matvec_iq4_xs_add_x0_fast"};
static int lut_is_layout(int mode) {return (mode>=7 && mode<=10) || mode==12;}
static int lut_layout_variant(int mode) {return mode==12 ? 4 : mode-7;}
static void lut_select(int mode) {
    layout_selected=lut_is_layout(mode) ? lut_layout_variant(mode) : -1;
    g_matvec_iq3_s_pair_pipeline=mode==6 ? lut_iq3_words : lut_iq3_base;
    lut_flush_interval=mode==3 ? 16 : mode==4 ? 8 : mode==5 ? 32 : 0;
    const int iq4=mode==11;
    if (mode>2) mode=0;
    g_matvec_iq4_xs_pipeline=iq4 ? prefetch_iq4_pipeline[0] : lut_pipeline[mode][0];
    g_matvec_iq4_xs_pair_pipeline=iq4 ? prefetch_iq4_pipeline[1] : lut_pipeline[mode][1];
    g_matvec_iq4_xs_add_x0_pipeline=iq4 ? prefetch_iq4_pipeline[2] : lut_pipeline[mode][2];
}
static NSString *lut_extract(NSString *source,NSString *marker,NSString *end_marker) {
    NSRange start=[source rangeOfString:marker];
    if (start.location==NSNotFound) return nil;
    NSRange end=[source rangeOfString:end_marker options:0 range:NSMakeRange(start.location,source.length-start.location)];
    if (end.location==NSNotFound) return nil;
    return [source substringWithRange:NSMakeRange(start.location,end.location+end.length-start.location)];
}
static int lut_replace(NSMutableString *source,NSString *old,NSString *replacement,NSUInteger expected) {
    NSUInteger count=[source replaceOccurrencesOfString:old withString:replacement options:0 range:NSMakeRange(0,source.length)];
    if (count!=expected) {fprintf(stderr,"lookup replacement count=%lu expected=%lu\n",(unsigned long)count,(unsigned long)expected);return 0;}
    return 1;
}
static int lut_initialize(void) {
    if (!qw3_metal_compile_kernels()) return 0;
    lut_pipeline[0][0]=g_matvec_iq4_xs_pipeline;
    lut_pipeline[0][1]=g_matvec_iq4_xs_pair_pipeline;
    lut_pipeline[0][2]=g_matvec_iq4_xs_add_x0_pipeline;
    lut_iq3_base=g_matvec_iq3_s_pair_pipeline;
    NSString *all=qw3_metal_kernel_source();
    NSString *table=lut_extract(all,@"constant float qw3_iq4nl_table[16]",@"\n};");
    NSString *value=lut_extract(all,@"inline float qw3_iq4nl_val(",@"\n}\n");
    NSString *helper=lut_extract(all,@"inline float qw3_iq4xs_dot8_vector(",@"\n}\n");
    if (!table || !value || !helper) return 0;
    for (int mode=1;mode<=2;mode++) {
        NSMutableString *source=[@"#include <metal_stdlib>\nusing namespace metal;\nstruct qw3_matvec_q8_0_args {uint n_in,n_out,row_bytes;};\n" mutableCopy];
        [source appendString:table];[source appendString:@"\n"];[source appendString:value];
        NSMutableString *dot=[helper mutableCopy];
        if (!lut_replace(dot,@"threadgroup const float *table",@"float table",1)) return 0;
        for (NSString *q in @[@"qlo.x",@"qlo.y",@"qlo.z",@"qlo.w",@"qhi.x",@"qhi.y",@"qhi.z",@"qhi.w"]) {
            NSString *old=[NSString stringWithFormat:@"table[%@]",q];
            NSString *replacement=mode==1 ? [NSString stringWithFormat:@"qw3_iq4nl_val(uint(%@))",q] :
                [NSString stringWithFormat:@"simd_shuffle(table, uint(%@))",q];
            if (!lut_replace(dot,old,replacement,1)) return 0;
        }
        [source appendString:dot];
        for (int kind=0;kind<3;kind++) {
            NSString *body=lut_extract(all,[@"kernel void " stringByAppendingString:[NSString stringWithUTF8String:lut_names[kind]]],@"\n}\n");
            if (!body) return 0;
            NSMutableString *kernel=[body mutableCopy];
            NSString *init=@"if (simd_idx == 0u && lane < 16u) sh[lane] = qw3_iq4nl_val(uint(lane));\n    threadgroup_barrier(mem_flags::mem_threadgroup);";
            NSString *replacement=mode==1 ? @"(void)sh;\n    const float table = 0.0f;" :
                @"(void)sh;\n    const float table = qw3_iq4nl_val(uint(lane) & 15u);";
            if (!lut_replace(kernel,init,replacement,1) ||
                !lut_replace(kernel,@"xl, sh, ib, il",@"xl, table, ib, il",kind==1 ? 4 : 2)) return 0;
            [source appendString:kernel];
        }
        NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
        id<MTLLibrary> library=[g_device newLibraryWithSource:source options:options error:&error];
        for (int kind=0;kind<3;kind++) {
            id<MTLFunction> fn=[library newFunctionWithName:[NSString stringWithUTF8String:lut_names[kind]]];
            lut_pipeline[mode][kind]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
            if (!lut_pipeline[mode][kind] || lut_pipeline[mode][kind].maxTotalThreadsPerThreadgroup<64) {
                fprintf(stderr,"lookup pipeline failed %s/%d: %s\n",lut_modes[mode],kind,error ? error.localizedDescription.UTF8String : "");return 0;
            }
        }
    }
    NSMutableString *iq3=[@"#include <metal_stdlib>\nusing namespace metal;\nstruct qw3_matvec_q8_0_args {uint n_in,n_out,row_bytes;};\n" mutableCopy];
    NSString *grid=lut_extract(all,@"inline float4 qw3_iq3s_float_grid(",@"\n}\n");
    NSString *pair=lut_extract(all,@"kernel void qw3_matvec_iq3_s_pair_fast(",@"\n}\n");
    NSMutableString *dot=[lut_extract(all,@"inline float qw3_iq3s_dot32_vector(",@"\n}\n") mutableCopy];
    if (!grid || !pair || !dot ||
        !lut_replace(dot,@"const uint idx1 = uint(qs[2u * l])",@"const uint q = uint(((device const ushort *)qs)[l]);\n        const uint idx1 = (q & 255u)",1) ||
        !lut_replace(dot,@"uint(qs[2u * l + 1u])",@"(q >> 8u)",1) ||
        !lut_replace(dot,@"uint(signs[l])",@"((uint(((device const ushort *)signs)[l >> 1u]) >> (8u * (l & 1u))) & 255u)",1)) return 0;
    [iq3 appendString:grid];[iq3 appendString:dot];[iq3 appendString:pair];
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    id<MTLLibrary> library=[g_device newLibraryWithSource:iq3 options:options error:&error];
    id<MTLFunction> fn=[library newFunctionWithName:@"qw3_matvec_iq3_s_pair_fast"];
    lut_iq3_words=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
    if (!lut_iq3_words || lut_iq3_words.maxTotalThreadsPerThreadgroup<64) {
        fprintf(stderr,"IQ3_S word pipeline failed: %s\n",error ? error.localizedDescription.UTF8String : "");return 0;
    }
    return 1;
}
static double lut_cpu_dot(const uint8_t *src,const float *x,uint32_t n) {
    double sum=0;
    for (uint32_t b=0;b<n/256;b++) {
        const uint8_t *blk=src+(uint64_t)b*136;
        const float d=qw3_f16_to_f32(qw3_load_u16(blk));
        const uint32_t hi=qw3_load_u16(blk+2);
        for (uint32_t ib=0;ib<8;ib++) {
            const int scale=((blk[4+ib/2]>>(4*(ib&1)))&15)|(((hi>>(2*ib))&3)<<4);
            const float dl=d*(scale-32);
            for (uint32_t j=0;j<16;j++) {
                const uint32_t q=blk[8+ib*16+j],i=b*256+ib*32+j;
                sum+=(double)(dl*qw3_iq4nl[q&15])*x[i];
                sum+=(double)(dl*qw3_iq4nl[q>>4])*x[i+16];
            }
        }
    }
    return sum;
}
static int lut_dispatch(int mode,int kind,id<MTLBuffer> __strong *weights,uint64_t *offset,
    id<MTLBuffer> input,id<MTLBuffer> output,uint32_t ni,uint32_t no,double *gpu) {
    struct {uint32_t ni,no,rb;} args={ni,no,ni/256*(kind==3 ? 110 : 136)};
    const float scale=-0.25f;
    int owned=0;id<MTLCommandBuffer> cb=qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc=qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:kind==3 ? (mode ? lut_iq3_words : lut_iq3_base) :
        mode==11 ? prefetch_iq4_pipeline[kind] : lut_pipeline[mode][kind]];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    if (kind==2) [enc setBytes:&scale length:sizeof(scale) atIndex:1];
    [enc setBuffer:weights[0] offset:offset[0] atIndex:kind==2 ? 2 : 1];
    if (kind==1 || kind==3) [enc setBuffer:weights[1] offset:offset[1] atIndex:2];
    [enc setBuffer:input offset:16 atIndex:kind ? 3 : 2];
    [enc setBuffer:output offset:20 atIndex:kind ? 4 : 3];
    if (kind==3) [enc setBuffer:qw3_metal_iq3s_kgrid_buffer() offset:0 atIndex:5];
    [enc setThreadgroupMemoryLength:128 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(kind==3 ? (no+7)/8 : (no+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    qw3_metal_end_compute_encoder(cb,enc);
    if (!qw3_metal_finish_command_buffer(cb,owned,"IQ4 lookup") || cb.status==MTLCommandBufferStatusError) return 0;
    *gpu=cb.GPUEndTime-cb.GPUStartTime;
    return 1;
}
static double lut_cpu_iq3_dot(const uint8_t *src,const float *x,uint32_t ni) {
    double sum=0;
    for (uint32_t b=0;b<ni/256;b++) {
        const uint8_t *blk=src+(uint64_t)b*110;
        const float d=qw3_f16_to_f32(qw3_load_u16(blk));
        for (uint32_t ib=0;ib<8;ib++) {
            const float db=d*(1+2*((blk[106+ib/2]>>(4*(ib&1)))&15));
            for (uint32_t l=0;l<4;l++) {
                const uint32_t qh=blk[66+ib],s=blk[74+ib*4+l];uint8_t lo[4],hi[4];
                iq3s_grid4(blk[2+ib*8+2*l]|((qh<<(8-2*l))&256),lo);
                iq3s_grid4(blk[3+ib*8+2*l]|((qh<<(7-2*l))&256),hi);
                for (uint32_t j=0;j<4;j++) {
                    const uint32_t i=b*256+ib*32+l*8+j;
                    sum+=(double)(db*lo[j]*((s&(1<<j)) ? -1.f : 1.f))*x[i];
                    sum+=(double)(db*hi[j]*((s&(1<<(j+4))) ? -1.f : 1.f))*x[i+4];
                }
            }
        }
    }
    return sum;
}
static int lut_iq3_micro(qw3_engine *engine) {
    const qw3_tensor *w[2]={NULL,NULL};
    for (int il=0;il<engine->model.spec->n_layer && !w[0];il++) {
        const qw3_layer_weights *lw=&engine->weights.layer[il];
        if (lw->ffn_gate->type==21 && lw->ffn_up->type==21) {w[0]=lw->ffn_gate;w[1]=lw->ffn_up;}
    }
    if (!w[0]) return 0;
    const uint32_t shapes[][2]={{256,1},{4352,33},{5120,4097},{5120,4111},{5120,17408},{17408,5120}};
    for (int shape=0;shape<6;shape++) {
        const uint32_t ni=shapes[shape][0],no=shapes[shape][1],n=no*2;
        if ((uint64_t)ni*no>w[0]->elements || (uint64_t)ni*no>w[1]->elements) return 0;
        uint64_t offset[2];id<MTLBuffer> weights[2],output[2];
        for (int v=0;v<2;v++) {
            weights[v]=qw3_metal_model_view_for(w[v]->offset,(uint64_t)ni/256*110*no,&offset[v]);
            output[v]=[g_device newBufferWithLength:(n+12)*4 options:MTLResourceStorageModeShared];
            if (!weights[v] || !output[v]) return 0;
        }
        id<MTLBuffer> input=[g_device newBufferWithLength:(ni+8)*4 options:MTLResourceStorageModeShared];
        if (!input) return 0;
        float *x=input.contents;for (uint32_t i=0;i<ni+8;i++) x[i]=((int)(i*37%101)-50)/31.f;
        double times[2]={0},gpu[2]={0};
        for (int repeat=-4;repeat<40;repeat++) for (int order=0;order<2;order++) {
            const int v=order^(repeat&1);float *out=output[v].contents;
            for (uint32_t i=0;i<n+12;i++) out[i]=123456;
            const double start=qw3_now_sec();double device;
            if (!lut_dispatch(v,3,weights,offset,input,output[v],ni,no,&device)) return 0;
            if (repeat>=0) {times[v]+=qw3_now_sec()-start;gpu[v]+=device;}
        }
        float *a=output[0].contents,*b=output[1].contents;double cpu_max=0;
        for (uint32_t i=0;i<n+12;i++) {
            if (i<5 || i>=n+5) {if (a[i]!=123456 || b[i]!=123456) return 0;}
            else if (!isfinite(a[i]) || !isfinite(b[i]) || a[i]!=b[i]) return 0;
        }
        const uint32_t rows[]={0,no/2,no-1};
        for (int v=0;v<2;v++) for (int r=0;r<3;r++) {
            const double reference=lut_cpu_iq3_dot(tensor_data(&engine->model,w[v])+(uint64_t)rows[r]*ni/256*110,x+4,ni);
            const double d=fabs(reference-b[5+v*no+rows[r]]);
            if (!isfinite(d)) return 0;cpu_max=fmax(cpu_max,d);
        }
        printf("lut_quant mode=iq3s_words kind=3 ni=%u no=%u max=0 rms=0 cpu_max=%.9g guards=ok\n",ni,no,cpu_max);
        if (cpu_max>0.0001) return 0;
        printf("lut_micro mode=iq3s_words kind=3 ni=%u no=%u baseline_ms=%.4f candidate_ms=%.4f gpu_base_ms=%.4f gpu_candidate_ms=%.4f gain_pct=%.2f\n",ni,no,times[0]*25,times[1]*25,gpu[0]*25,gpu[1]*25,(times[0]/times[1]-1)*100);fflush(stdout);
    }
    return 1;
}
static int lut_states_equal(qw3_session **s) {
    qw3_metal_state_info info={0};
    if (!qw3_metal_session_get_state_info(s[0]->metal,&info)) return 0;
    const size_t chunk=4u*1024u*1024u;unsigned char *data[2]={malloc(chunk),malloc(chunk)};
    int ok=data[0] && data[1];
    for (int kind=1;ok && kind<=5;kind++) {
        if (kind==3) continue; /* derived FlashAttention scratch, not canonical KV */
        const uint64_t bytes=kind==4 ? info.deltanet_bytes : kind==5 ? info.conv_bytes : info.kv_row_bytes*s[0]->tokens.len;
        const uint32_t layers=kind<=2 ? info.n_full_layers : 1;
        for (uint32_t layer=0;ok && layer<layers;layer++) for (uint64_t offset=0;ok && offset<bytes;offset+=chunk) {
            const uint64_t count=MIN((uint64_t)chunk,bytes-offset);
            for (int v=0;ok && v<2;v++) ok=qw3_metal_session_read_state(s[v]->metal,(qw3_metal_state_buffer)kind,layer,offset,data[v],count);
            if (ok && memcmp(data[0],data[1],count)) {fprintf(stderr,"decode state differs kind=%d layer=%u\n",kind,layer);ok=0;}
        }
    }
    free(data[0]);free(data[1]);
    if (ok) printf("lut_state deltanet=exact conv=exact live_kv=exact tokens=%d\n",s[0]->tokens.len);
    return ok;
}
static int lut_micro(qw3_engine *engine,int only_prefetch) {
    const qw3_tensor *weight[2]={NULL,NULL};
    for (int il=0;il<engine->model.spec->n_layer && !weight[0];il++) {
        const qw3_layer_weights *lw=&engine->weights.layer[il];
        if (lw->ffn_gate->type==23 && lw->ffn_up->type==23) {weight[0]=lw->ffn_gate;weight[1]=lw->ffn_up;}
    }
    if (!weight[0]) return 0;
    const uint32_t shapes[][2]={{256,1},{4352,33},{5120,4097},{5120,4111},{5120,17408},{17408,5120}};
    for (int shape=0;shape<6;shape++) {
        const uint32_t ni=shapes[shape][0],no=shapes[shape][1];
        if ((uint64_t)ni*no>weight[0]->elements || (uint64_t)ni*no>weight[1]->elements) return 0;
        uint64_t offset[2];id<MTLBuffer> weights[2];
        for (int w=0;w<2;w++) weights[w]=qw3_metal_model_view_for(weight[w]->offset,(uint64_t)ni/256*136*no,&offset[w]);
        id<MTLBuffer> input=[g_device newBufferWithLength:(ni+8)*4 options:MTLResourceStorageModeShared];
        if (!weights[0] || !weights[1] || !input) return 0;
        float *x=input.contents;
        for (uint32_t i=0;i<ni+8;i++) x[i]=((int)(i*37%101)-50)/31.0f;
        for (int kind=0;kind<3;kind++) for (int mode=only_prefetch ? 11 : 1;mode<=(only_prefetch ? 11 : 2);mode++) {
            const uint32_t n=kind==1 ? no*2 : no;
            id<MTLBuffer> output[2];double times[2]={0},gpu[2]={0};
            for (int v=0;v<2;v++) output[v]=[g_device newBufferWithLength:(n+12)*4 options:MTLResourceStorageModeShared];
            if (!output[0] || !output[1]) return 0;
            for (int repeat=-4;repeat<40;repeat++) for (int order=0;order<2;order++) {
                const int v=order^(repeat&1);float *out=output[v].contents;
                /* Residual-add is in-place, so reset the same input each time. */
                for (uint32_t i=0;i<n+12;i++) out[i]=kind==2 && i>=5 && i<no+5 ? (float)(i%17)/13 : 123456.0f;
                const double start=qw3_now_sec();double device=0;
                if (!lut_dispatch(v ? mode : 0,kind,weights,offset,input,output[v],ni,no,&device)) return 0;
                if (repeat>=0) {times[v]+=qw3_now_sec()-start;gpu[v]+=device;}
            }
            double max=0,ss=0,cpu_max=0;float *a=output[0].contents,*b=output[1].contents;
            for (int v=0;v<2;v++) {
                float *out=output[v].contents;
                for (uint32_t i=0;i<n+12;i++) if ((i<5 || i>=n+5) && out[i]!=123456.0f) return 0;
            }
            for (uint32_t i=0;i<n;i++) {
                if (!isfinite(a[i+5]) || !isfinite(b[i+5])) return 0;
                const double d=fabs((double)a[i+5]-b[i+5]);max=fmax(max,d);ss+=d*d;
            }
            const uint32_t rows[]={0,no/2,no-1};
            for (int w=0;w<(kind==1 ? 2 : 1);w++) for (int r=0;r<3;r++) {
                double reference=lut_cpu_dot(tensor_data(&engine->model,weight[w])+(uint64_t)rows[r]*ni/256*136,x+4,ni);
                if (kind==2) reference=(float)((rows[r]+5)%17)/13+reference*(-0.25);
                const double d=fabs(reference-b[5+w*no+rows[r]]);
                if (!isfinite(reference) || !isfinite(d)) return 0;
                cpu_max=fmax(cpu_max,d);
            }
            printf("lut_quant mode=%s kind=%d ni=%u no=%u max=%.9g rms=%.9g cpu_max=%.9g guards=ok\n",lut_modes[mode],kind,ni,no,max,sqrt(ss/n),cpu_max);
            if (max!=0 || cpu_max>0.0001) return 0;
            printf("lut_micro mode=%s kind=%d ni=%u no=%u baseline_ms=%.4f candidate_ms=%.4f gpu_base_ms=%.4f gpu_candidate_ms=%.4f gain_pct=%.2f\n",
                lut_modes[mode],kind,ni,no,times[0]*25,times[1]*25,gpu[0]*25,gpu[1]*25,(times[0]/times[1]-1)*100);fflush(stdout);
        }
    }
    return 1;
}
static int lut_layout_buffer_guard(id<MTLBuffer> buffer) {
    if (!buffer || buffer.length<64) return 0;
    const unsigned char *data=buffer.contents;
    for (uint32_t i=0;i<32;i++) if (data[i]!=0xa5 || data[buffer.length-32+i]!=0xa5) return 0;
    return 1;
}
static int lut_layout_guards(void) {
    for (NSNumber *key in layout_weights) if (!lut_layout_buffer_guard(layout_weights[key])) return 0;
    return 1;
}
static int lut_layout_prepare(qw3_engine *engine,int variant) {
    const double start=qw3_now_sec();
    layout_weights=[NSMutableDictionary new];layout_shapes=[NSMutableDictionary new];
    layout_bytes=0;layout_calls=0;
    for (int il=0;il<engine->model.spec->n_layer;il++) {
        const qw3_layer_weights *lw=&engine->weights.layer[il];
        const qw3_tensor *tensors[]={lw->ffn_gate,lw->ffn_up,lw->ffn_down};
        for (int kind=0;kind<3;kind++) {
            const qw3_tensor *tensor=tensors[kind];
            if (!tensor || tensor->type!=23 || layout_weights[@(tensor->offset)]) continue;
            const uint32_t ni=kind==2 ? engine->model.spec->n_ff_dense : engine->model.spec->n_embd;
            const uint32_t no=kind==2 ? engine->model.spec->n_embd : engine->model.spec->n_ff_dense;
            id<MTLBuffer> buffer=layout_pack(tensor->offset,ni,no,variant);
            if (!buffer || !lut_layout_buffer_guard(buffer)) return 0;
            layout_weights[@(tensor->offset)]=buffer;layout_shapes[@(tensor->offset)]=@[@(ni),@(no)];
            layout_bytes+=(uint64_t)ni/256*136*no;
        }
    }
    printf("layout_prepare mode=%s tensors=%lu weight_mib=%.2f seconds=%.3f bytes_exact=1\n",layout_names[variant],(unsigned long)layout_weights.count,layout_bytes/1048576.0,qw3_now_sec()-start);fflush(stdout);
    return layout_weights.count>0;
}
static uint32_t lut_layout_calls_per_token(qw3_engine *engine) {
    const char *pair=getenv("QW3_METAL_DENSE_IQ4_XS_PAIR");
    const char *down=getenv("QW3_METAL_DENSE_IQ4_DOWN_ADD");
    if ((pair && !strcmp(pair,"0")) || (down && !strcmp(down,"0"))) return 0;
    uint32_t calls=0;
    for (int il=0;il<engine->model.spec->n_layer;il++) {
        const qw3_layer_weights *lw=&engine->weights.layer[il];
        const int gate=lw->ffn_gate->type==23,up=lw->ffn_up->type==23;
        calls+=(gate && up ? 1 : gate+up)+(lw->ffn_down->type==23);
    }
    return calls;
}
static int lut_layout_micro(qw3_engine *engine,int variant) {
    const qw3_tensor *weight[2]={NULL,NULL};
    for (int il=0;il<engine->model.spec->n_layer && !weight[0];il++) {
        const qw3_layer_weights *lw=&engine->weights.layer[il];
        if (lw->ffn_gate->type==23 && lw->ffn_up->type==23) {weight[0]=lw->ffn_gate;weight[1]=lw->ffn_up;}
    }
    if (!weight[0]) return 0;
    const uint32_t shapes[][2]={{256,1},{4352,33},{5120,4097},{5120,4111},{5120,17408},{17408,5120}};
    for (int shape=0;shape<6;shape++) {
        const uint32_t ni=shapes[shape][0],no=shapes[shape][1];
        if ((uint64_t)ni*no>weight[0]->elements || (uint64_t)ni*no>weight[1]->elements) return 0;
        uint64_t offset[2];id<MTLBuffer> weights[2],packed[2];
        for (int w=0;w<2;w++) {
            weights[w]=qw3_metal_model_view_for(weight[w]->offset,(uint64_t)ni/256*136*no,&offset[w]);
            packed[w]=layout_pack(weight[w]->offset,ni,no,variant);
            if (!weights[w] || !packed[w]) return 0;
        }
        id<MTLBuffer> input=[g_device newBufferWithLength:(ni+8)*4 options:MTLResourceStorageModeShared];
        if (!input) return 0;
        float *x=input.contents;
        for (uint32_t i=0;i<ni+8;i++) x[i]=((int)(i*37%101)-50)/31.f;
        for (int kind=0;kind<3;kind++) {
            const uint32_t n=kind==1 ? no*2 : no;
            id<MTLBuffer> output[2];double times[2]={0},gpu[2]={0};
            for (int v=0;v<2;v++) output[v]=[g_device newBufferWithLength:(n+12)*4 options:MTLResourceStorageModeShared];
            if (!output[0] || !output[1]) return 0;
            for (int repeat=-4;repeat<40;repeat++) for (int order=0;order<2;order++) {
                const int v=order^(repeat&1);float *out=output[v].contents;
                for (uint32_t i=0;i<n+12;i++) out[i]=kind==2 && i>=5 && i<no+5 ? (float)(i%17)/13 : 123456;
                const double start=qw3_now_sec();double device=0;
                const int ok=v ? layout_dispatch(variant,kind,packed[0],packed[1],32,32,input,16,output[v],20,ni,no,-.25f,&device) :
                    lut_dispatch(0,kind,weights,offset,input,output[v],ni,no,&device);
                if (!ok) return 0;
                if (repeat>=0) {times[v]+=qw3_now_sec()-start;gpu[v]+=device;}
            }
            double cpu_max=0;float *a=output[0].contents,*b=output[1].contents;
            for (int v=0;v<2;v++) {
                if (!lut_layout_buffer_guard(packed[v])) return 0;
                float *out=output[v].contents;
                for (uint32_t i=0;i<n+12;i++) if ((i<5 || i>=n+5) && out[i]!=123456) return 0;
            }
            for (uint32_t i=0;i<n;i++) if (!isfinite(a[i+5]) || !isfinite(b[i+5]) || a[i+5]!=b[i+5]) {
                fprintf(stderr,"layout micro differs mode=%s shape=%d kind=%d row=%u old=%.9g new=%.9g\n",layout_names[variant],shape,kind,i,a[i+5],b[i+5]);return 0;
            }
            const uint32_t rows[]={0,no/2,no-1};
            for (int w=0;w<(kind==1 ? 2 : 1);w++) for (int r=0;r<3;r++) {
                double reference=lut_cpu_dot(tensor_data(&engine->model,weight[w])+(uint64_t)rows[r]*ni/256*136,x+4,ni);
                if (kind==2) reference=(float)((rows[r]+5)%17)/13+reference*(-.25);
                const double d=fabs(reference-b[5+w*no+rows[r]]);
                if (!isfinite(d)) return 0;cpu_max=fmax(cpu_max,d);
            }
            printf("layout_quant mode=%s kind=%d ni=%u no=%u max=0 rms=0 cpu_max=%.9g guards=ok bytes_exact=1\n",layout_names[variant],kind,ni,no,cpu_max);
            if (cpu_max>1e-4) return 0;
            printf("layout_micro mode=%s kind=%d ni=%u no=%u baseline_ms=%.4f candidate_ms=%.4f gpu_base_ms=%.4f gpu_candidate_ms=%.4f gain_pct=%.2f\n",layout_names[variant],kind,ni,no,times[0]*25,times[1]*25,gpu[0]*25,gpu[1]*25,(times[0]/times[1]-1)*100);fflush(stdout);
        }
    }
    return 1;
}
int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=2 && argc!=4) {fprintf(stderr,"usage: %s MODEL [--full MODE|--micro blocks4|blocks16|planes4|planes16|iq4_prefetch|native_copy]\n",argv[0]);return 1;}
        int mode=0,micro=0;
        if (argc==4) {
            micro=!strcmp(argv[2],"--micro");
            if (!micro && strcmp(argv[2],"--full")) return 1;
            for (int m=1;m<=12;m++) if (!strcmp(argv[3],lut_modes[m])) mode=m;
            if (!mode || (micro && mode<7)) return 1;
        }
        qw3_engine *engine=NULL;qw3_session *s[2]={NULL,NULL};qw3_tokens prompt={0};
        qw3_engine_options options={.model_path=argv[1],.backend=QW3_BACKEND_METAL};
        float *logits[2]={NULL,NULL};char error[512]={0};int status=1;
        if (qw3_engine_open(&engine,&options)!=0 || engine->model.spec->ffn_kind!=QW3_FFN_DENSE || !lut_initialize()) goto done;
        if (lut_is_layout(mode) && !layout_initialize()) goto done;
        if (mode==11 && !prefetch_iq4_initialize()) goto done;
        if (!mode) {if (lut_micro(engine,0) && lut_iq3_micro(engine)) status=0;goto done;}
        if (micro) {if (mode==11 ? lut_micro(engine,1) : lut_layout_micro(engine,lut_layout_variant(mode))) status=0;goto done;}
        if (lut_is_layout(mode) && !lut_layout_prepare(engine,lut_layout_variant(mode))) goto done;
        const uint32_t layout_expected=lut_is_layout(mode) ? lut_layout_calls_per_token(engine) : 0;
        if (lut_is_layout(mode) && !layout_expected) goto done;
        const int vocab=engine->model.spec->n_vocab;
        qw3_encode_chat_prompt(engine,NULL,"Spiega dettagliatamente come implementare una tabella hash in C.",QW3_THINK_NONE,&prompt);
        for (int v=0;v<2;v++) {
            lut_select(v ? mode : 0);logits[v]=calloc(vocab,sizeof(float));
            if (!logits[v] || qw3_session_create(&s[v],engine,512)!=0 || s[v]->metal_n_gpu_layers!=engine->model.spec->n_layer ||
                qw3_session_sync(s[v],&prompt,error,sizeof(error))!=0) goto done;
            qw3_metal_state_info info={0};
            if (!qw3_metal_session_get_state_info(s[v]->metal,&info) || info.kv_type!=1) goto done;
            if (qw3_session_copy_logits(s[v],logits[v],vocab)!=vocab) goto done;
        }
        for (int k=0;k<vocab;k++) if (!isfinite(logits[0][k]) || !isfinite(logits[1][k]) || logits[0][k]!=logits[1][k]) goto done;
        uint64_t rng[2]={42,42},hash=1469598103934665603ULL;
        double forward[2]={0},sampling[2]={0},pairs[120],block[2]={0};
        double submit[2]={0},gpu[2]={0},wait[2]={0};
        for (int i=0;i<128;i++) {
            int token[2];double elapsed[2]={0};
            for (int v=0;v<2;v++) {
                const qw3_tokens *history=qw3_session_tokens(s[v]);
                const int n=MIN(history->len,1024);const double start=qw3_now_sec();
                token[v]=qw3_session_sample_repetition(s[v],0.6f,20,0.95f,0,&rng[v],history->v+history->len-n,n,1.06f);
                if (i>=8) sampling[v]+=qw3_now_sec()-start;
            }
            if (token[0]<0 || token[0]!=token[1]) goto done;
            hash=(hash^(uint32_t)token[0])*1099511628211ULL;
            for (int order=0;order<2;order++) {
                const int v=order^(i&1);lut_select(v ? mode : 0);const double start=qw3_now_sec();
                const uint64_t calls_before=layout_calls;
                lut_forward_start=lut_clock();lut_observe=1;
                const int result=qw3_session_eval_gpu_logits(s[v],token[v],error,sizeof(error));
                lut_observe=0;
                if (result!=0) goto done;
                if (lut_is_layout(mode) && layout_calls-calls_before!=(v ? layout_expected : 0)) {
                    fprintf(stderr,"layout dispatch coverage failed step=%d expected=%u actual=%llu\n",i,v ? layout_expected : 0,(unsigned long long)(layout_calls-calls_before));goto done;
                }
                elapsed[v]=qw3_now_sec()-start;
                if (i>=8) {
                    forward[v]+=elapsed[v];block[v]+=elapsed[v];
                    submit[v]+=lut_submit_ms;gpu[v]+=lut_gpu_ms;wait[v]+=lut_wait_ms;
                }
            }
            for (int v=0;v<2;v++) if (qw3_session_copy_logits(s[v],logits[v],vocab)!=vocab ||
                !s[v]->valid || s[v]->kv.pos!=s[v]->tokens.len) goto done;
            if (s[0]->tokens.len!=s[1]->tokens.len || memcmp(s[0]->tokens.v,s[1]->tokens.v,s[0]->tokens.len*sizeof(int))) goto done;
            for (int k=0;k<vocab;k++) if (!isfinite(logits[0][k]) || !isfinite(logits[1][k]) || logits[0][k]!=logits[1][k]) {
                fprintf(stderr,"lookup full logits differ step=%d id=%d\n",i,k);goto done;
            }
            if (i>=8) {
                pairs[i-8]=(elapsed[0]-elapsed[1])*1000;
                printf("lut_pair mode=%s step=%d saved_ms=%.6f\n",lut_modes[mode],i,pairs[i-8]);
            }
            if (i>=8 && ((i-7)%16==0 || i==127)) {
                printf("lut_block mode=%s end=%d baseline_ms=%.3f candidate_ms=%.3f gain_pct=%.2f\n",lut_modes[mode],i+1,block[0]*1000,block[1]*1000,(block[0]/block[1]-1)*100);
                block[0]=block[1]=0;fflush(stdout);
            }
        }
        double mean=0,ss=0;
        for (int i=0;i<120;i++) mean+=pairs[i]/120;
        for (int i=0;i<120;i++) ss+=(pairs[i]-mean)*(pairs[i]-mean);
        printf("lut_full mode=%s tokens=128 timed=120 baseline=%.3f candidate=%.3f gain_pct=%.2f saved_ms=%.6f naive_ci95_ms=%.6f hash=%" PRIx64 " logits_exact=1\n",
            lut_modes[mode],120/(forward[0]+sampling[0]),120/(forward[1]+sampling[1]),(forward[0]/forward[1]-1)*100,mean,1.980*sqrt(ss/119/120),hash);
        printf("lut_schedule mode=%s baseline_submit_ms=%.6f candidate_submit_ms=%.6f baseline_wait_ms=%.6f candidate_wait_ms=%.6f baseline_gpu_ms=%.6f candidate_gpu_ms=%.6f\n",
            lut_modes[mode],submit[0]/120,submit[1]/120,wait[0]/120,wait[1]/120,gpu[0]/120,gpu[1]/120);
        /* Per-token intervals are descriptive: adjacent steps are correlated. */
        if (!lut_states_equal(s)) goto done;
        if (lut_is_layout(mode)) printf("layout_full mode=%s calls=%llu weight_mib=%.2f guards=%s\n",lut_modes[mode],(unsigned long long)layout_calls,layout_bytes/1048576.0,lut_layout_guards() ? "ok" : "failed");
        if (lut_is_layout(mode) && (!layout_calls || !lut_layout_guards())) goto done;
        status=0;
done:
        if (lut_pipeline[0][0]) lut_select(0);
        if (status) fprintf(stderr,"IQ4 lookup benchmark failed: %s\n",error);
        for (int v=0;v<2;v++) {free(logits[v]);if (s[v]) qw3_session_free(s[v]);}
        layout_weights=nil;layout_shapes=nil;
        qw3_tokens_free(&prompt);if (engine) qw3_engine_close(engine);
        return status;
    }
}
