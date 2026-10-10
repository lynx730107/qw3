/* Paired pp128 staging/geometry probe. Production shaders stay unchanged. */
#import <Metal/Metal.h>
#include <stdint.h>
static uint32_t prefill_columns(id<MTLComputePipelineState> pipeline);
static uint32_t prefill_threads(id<MTLComputePipelineState> pipeline);
#define QW3_METAL_STAGED128_COLUMNS(pipeline) prefill_columns(pipeline)
#define QW3_METAL_STAGED128_THREADS(pipeline) prefill_threads(pipeline)
#define qw3_metal_session_stage_dense_gqa_projections prefill_blit_gqa
#define qw3_metal_session_stage_dense_deltanet_projections prefill_blit_dn
#define qw3_metal_session_copy_batch_scratch_to_x1 prefill_blit_compact
#include "../qw3_metal.m"
#undef qw3_metal_session_stage_dense_gqa_projections
#undef qw3_metal_session_stage_dense_deltanet_projections
#undef qw3_metal_session_copy_batch_scratch_to_x1
#include "prefill_copy_probe.inc"
#define qw3_metal_env_n_gpu_layers qw3_memory_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled qw3_memory_core_llama_split_enabled
#define qw3_metal_llama_split_start qw3_memory_core_llama_split_start
#include "../qw3.c"

static id<MTLComputePipelineState> prefill_iq3[2],prefill_iq4[2];
static const char *prefill_mode="vector_store";
static uint32_t prefill_tile=64;
static uint32_t prefill_k=32;
static uint32_t prefill_simdgroups=4;
static uint32_t prefill_columns(id<MTLComputePipelineState> pipeline) {
    return pipeline==prefill_iq3[1] || pipeline==prefill_iq4[1] ? prefill_tile : 64u;
}
static uint32_t prefill_threads(id<MTLComputePipelineState> pipeline) {
    return pipeline==prefill_iq3[1] || pipeline==prefill_iq4[1] ? prefill_simdgroups*32u : 128u;
}
static int prefill_replace(NSMutableString *source,NSString *old,NSString *replacement,NSUInteger expected) {
    NSUInteger changes=[source replaceOccurrencesOfString:old withString:replacement options:0 range:NSMakeRange(0,source.length)];
    if (changes!=expected) {fprintf(stderr,"unexpected replacement count: %lu expected %lu\n",(unsigned long)changes,(unsigned long)expected);return 0;}
    return 1;
}
static int prefill_resize(NSMutableString *source,NSString *name) {
    NSRange start=[source rangeOfString:[@"kernel void " stringByAppendingString:name]];
    if (start.location==NSNotFound) return 0;
    NSRange end=[source rangeOfString:@"\n}\n" options:0 range:NSMakeRange(start.location,source.length-start.location)];
    if (end.location==NSNotFound) return 0;
    NSRange range=NSMakeRange(start.location,end.location+end.length-start.location);
    NSMutableString *body=[[source substringWithRange:range] mutableCopy];
    if (!prefill_replace(body,@"constexpr int NR0 = 64;",[NSString stringWithFormat:@"constexpr int NR0 = %u;",prefill_tile],1)) return 0;
    if (prefill_simdgroups!=4) {
        if (!prefill_replace(body,@"constexpr int ROWS_PER_SG = 32;",[NSString stringWithFormat:@"constexpr int ROWS_PER_SG = %u;",128u/prefill_simdgroups],1) ||
            !prefill_replace(body,@"(4 * ROWS_PER_SG)",[NSString stringWithFormat:@"(%u * ROWS_PER_SG)",prefill_simdgroups],1) ||
            !prefill_replace(body,@"if (r0 + row < int(args.n_out))",@"if (row < NR0 && r0 + row < int(args.n_out))",1) ||
            !prefill_replace(body,@"} else {",@"} else if (row < NR0) {",1)) return 0;
    }
    if (prefill_tile<64) {
        if (!prefill_replace(body,@"if (r0 + row < int(args.n_out))",@"if (row < NR0 && r0 + row < int(args.n_out))",1) ||
            !prefill_replace(body,@"} else {",@"} else if (row < NR0) {",1)) return 0;
    } else if (prefill_tile>64) {
        if (!prefill_replace(body,@"const int row = int(tid) / 2;",@"for (int row = int(tid) / 2; row < NR0; row += 64) {",1) ||
            !prefill_replace(body,@"        threadgroup_barrier(mem_flags::mem_threadgroup);\n        auto mA",@"        }\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n        auto mA",1)) return 0;
    }
    if (prefill_k==64) {
        if (!prefill_replace(body,@"constexpr int NK = 32;",@"constexpr int NK = 64;",1) ||
            !prefill_replace(body,@"const int half_k = int(tid) & 1;",@"const int half_k = int(tid) & 1;\n        for (int part = 0; part < 2; ++part) {",1) ||
            !prefill_replace(body,@"half_k * 16",@"half_k * 32 + part * 16",3) ||
            !prefill_replace(body,@"        threadgroup_barrier(mem_flags::mem_threadgroup);\n        auto mA",@"        }\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n        auto mA",1)) return 0;
    }
    [source replaceCharactersInRange:range withString:body];
    return 1;
}
static int prefill_stage_input(NSMutableString *source,NSString *name,int prefetch,int half_input) {
    NSRange start=[source rangeOfString:[@"kernel void " stringByAppendingString:name]];
    if (start.location==NSNotFound) return 0;
    NSRange end=[source rangeOfString:@"\n}\n" options:0 range:NSMakeRange(start.location,source.length-start.location)];
    if (end.location==NSNotFound) return 0;
    NSRange range=NSMakeRange(start.location,end.location+end.length-start.location);
    NSMutableString *body=[[source substringWithRange:range] mutableCopy];
    NSString *old=@"auto tB = tensor(ptrB,\n                    dextents<int32_t, 2>(int(args.n_in), int(args.n_tokens)),\n                    array<int, 2>({1, int(args.in_stride)}));";
    NSString *replacement=@"threadgroup float input_stage[128 * NK];\n    auto tB = tensor(input_stage,\n                    dextents<int32_t, 2>(NK, 128),\n                    array<int, 2>({1, NK}));";
    if (!prefill_replace(body,old,replacement,1) ||
        !prefill_replace(body,@"auto mB = tB.slice(int(loop_k),\n                           r1 + int(simd_group) * ROWS_PER_SG);",
            @"auto mB = tB.slice(0, int(simd_group) * ROWS_PER_SG);",1)) return 0;
    NSString *load=@"#pragma unroll\n        for (uint slot = 0; slot < 32; ++slot) {\n            const uint index = uint(tid) + slot * 128u;\n            const uint row = index / uint(NK), column = index % uint(NK);\n            input_stage[index] = ptrB[uint64_t(r1 + int(row)) * args.in_stride + loop_k + column];\n        }\n";
    if (prefetch) {
        NSString *initial=@"float next_input[32];\n#pragma unroll\n    for (uint slot = 0; slot < 32; ++slot) {\n        const uint index = uint(tid) + slot * 128u;\n        next_input[slot] = ptrB[uint64_t(r1 + int(index / uint(NK))) * args.in_stride + index % uint(NK)];\n    }\n    for (uint loop_k = 0u; loop_k < args.n_in; loop_k += NK) {\n#pragma unroll\n        for (uint slot = 0; slot < 32; ++slot) input_stage[uint(tid) + slot * 128u] = next_input[slot];";
        NSString *ahead=@"        threadgroup_barrier(mem_flags::mem_threadgroup);\n        if (loop_k + NK < args.n_in) {\n#pragma unroll\n            for (uint slot = 0; slot < 32; ++slot) {\n                const uint index = uint(tid) + slot * 128u;\n                next_input[slot] = ptrB[uint64_t(r1 + int(index / uint(NK))) * args.in_stride + loop_k + NK + index % uint(NK)];\n            }\n        }\n        auto mA";
        if (!prefill_replace(body,@"for (uint loop_k = 0u; loop_k < args.n_in; loop_k += NK) {",initial,1) ||
            !prefill_replace(body,@"        threadgroup_barrier(mem_flags::mem_threadgroup);\n        auto mA",ahead,1)) return 0;
    } else if (!prefill_replace(body,@"for (uint loop_k = 0u; loop_k < args.n_in; loop_k += NK) {",
        [@"for (uint loop_k = 0u; loop_k < args.n_in; loop_k += NK) {\n" stringByAppendingString:load],1)) return 0;
    if (half_input && (!prefill_replace(body,@"threadgroup float input_stage",@"threadgroup half input_stage",1) ||
        (prefetch && !prefill_replace(body,@"float next_input[32]",@"half next_input[32]",1)))) return 0;
    [source replaceCharactersInRange:range withString:body];
    return 1;
}
static int prefill_initialize(void) {
    if (!qw3_metal_compile_kernels() || !g_metal4_tensor_api_enabled) return 0;
    prefill_iq3[0]=g_matmul_iq3_s_mpp_staged128_pipeline;
    prefill_iq4[0]=g_matmul_iq4_xs_mpp_staged128_pipeline;
    NSMutableString *source=[qw3_metal_kernel_source() mutableCopy];
    NSString *old=@"#pragma unroll\n            for (int i = 0; i < 16; ++i)\n                sa[row * NK + half_k * 16 + i] =\n                    values[i / 4][i & 3];";
    NSString *replacement=@"threadgroup half4 *dst = (threadgroup half4 *)(sa + row * NK + half_k * 16);\n#pragma unroll\n            for (int i = 0; i < 4; ++i) dst[i] = values[i];";
    if (prefill_tile!=64 || prefill_k!=32 || prefill_simdgroups!=4) {
        if (!prefill_resize(source,@"qw3_matmul_iq3_s_mpp_staged128") ||
            !prefill_resize(source,@"qw3_matmul_iq4_xs_mpp_staged128")) return 0;
    } else if (!strcmp(prefill_mode,"input_stage") || !strcmp(prefill_mode,"input_prefetch") ||
               !strcmp(prefill_mode,"input_half") || !strcmp(prefill_mode,"input_half_prefetch")) {
        const int prefetch=!strcmp(prefill_mode,"input_prefetch") || !strcmp(prefill_mode,"input_half_prefetch");
        const int half_input=!strcmp(prefill_mode,"input_half") || !strcmp(prefill_mode,"input_half_prefetch");
        if (!prefill_stage_input(source,@"qw3_matmul_iq3_s_mpp_staged128",prefetch,half_input) ||
            !prefill_stage_input(source,@"qw3_matmul_iq4_xs_mpp_staged128",prefetch,half_input)) return 0;
    } else if (strcmp(prefill_mode,"copies") && strcmp(prefill_mode,"production_copy") && !prefill_replace(source,old,replacement,4)) return 0;
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
    [source appendString:prefill_copy_source];
    id<MTLLibrary> library=[g_device newLibraryWithSource:source options:options error:&error];
    id<MTLFunction> copy=[library newFunctionWithName:@"prefill_copy_rows"];
    prefill_copy_pipeline=copy ? [g_device newComputePipelineStateWithFunction:copy error:&error] : nil;
    id<MTLFunction> iq3=[library newFunctionWithName:@"qw3_matmul_iq3_s_mpp_staged128"];
    id<MTLFunction> iq4=[library newFunctionWithName:@"qw3_matmul_iq4_xs_mpp_staged128"];
    prefill_iq3[1]=iq3 ? [g_device newComputePipelineStateWithFunction:iq3 error:&error] : nil;
    prefill_iq4[1]=iq4 ? [g_device newComputePipelineStateWithFunction:iq4 error:&error] : nil;
    if (!prefill_copy_pipeline || !prefill_iq3[1] || !prefill_iq4[1] || prefill_iq3[1].maxTotalThreadsPerThreadgroup<prefill_simdgroups*32u || prefill_iq4[1].maxTotalThreadsPerThreadgroup<prefill_simdgroups*32u) {
        fprintf(stderr,"prefill candidate unavailable: %s\n",error ? error.localizedDescription.UTF8String : "");return 0;
    }
    return prefill_copy_selftest(0) && prefill_copy_selftest(1);
}
static void prefill_select(int v) {
    /* This experiment compares its own K32 pipelines. */
    setenv("QW3_METAL_DENSE_PREFILL_K64", "0", 1);
    setenv("QW3_METAL_DENSE_GPU_COPY",v && !strcmp(prefill_mode,"production_copy") ? "1" : "0",1);
    prefill_copy_enabled=v && !strcmp(prefill_mode,"copies");
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
static int prefill_states_equal(qw3_session **s) {
    qw3_metal_state_info info={0};
    if (!qw3_metal_session_get_state_info(s[0]->metal,&info)) return 0;
    const size_t chunk=4u*1024u*1024u;
    unsigned char *data[2]={malloc(chunk),malloc(chunk)};
    int ok=data[0] && data[1];
    for (int kind=QW3_METAL_STATE_GQA_K;ok && kind<=QW3_METAL_STATE_CONV;kind++) {
        if (kind==QW3_METAL_STATE_GQA_FLASH_TAIL) continue;
        const uint64_t bytes=kind==QW3_METAL_STATE_DELTANET ? info.deltanet_bytes :
            kind==QW3_METAL_STATE_CONV ? info.conv_bytes : info.kv_row_bytes*s[0]->tokens.len;
        const uint32_t layers=kind<=QW3_METAL_STATE_GQA_V ? info.n_full_layers : 1u;
        for (uint32_t layer=0;ok && layer<layers;layer++) for (uint64_t offset=0;ok && offset<bytes;offset+=chunk) {
            const uint64_t size=MIN((uint64_t)chunk,bytes-offset);
            for (int v=0;ok && v<2;v++) ok=qw3_metal_session_read_state(s[v]->metal,
                (qw3_metal_state_buffer)kind,layer,offset,data[v],size);
            if (ok && memcmp(data[0],data[1],size)) {
                fprintf(stderr,"prefill state mismatch kind=%d layer=%u offset=%llu\n",kind,layer,(unsigned long long)offset);ok=0;
            }
        }
    }
    free(data[0]);free(data[1]);
    if (ok) printf("prefill_state deltanet=exact conv=exact live_kv=exact tokens=%d\n",s[0]->tokens.len);
    return ok;
}
int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=2 && argc!=3) {fprintf(stderr,"usage: %s MODEL [vector_store|vector_dequant|tile32|tile16|tile128|k64|sg8|input_stage|input_prefetch|input_half|input_half_prefetch|copies|production_copy]\n",argv[0]);return 1;}
        if (argc==3) {
            prefill_mode=argv[2];
            if (!strcmp(prefill_mode,"tile32")) prefill_tile=32;
            else if (!strcmp(prefill_mode,"tile16")) prefill_tile=16;
            else if (!strcmp(prefill_mode,"tile128")) prefill_tile=128;
            else if (!strcmp(prefill_mode,"k64")) prefill_k=64;
            else if (!strcmp(prefill_mode,"sg8")) prefill_simdgroups=8;
            else if (strcmp(prefill_mode,"vector_store") && strcmp(prefill_mode,"vector_dequant") && strcmp(prefill_mode,"input_stage") && strcmp(prefill_mode,"input_prefetch") && strcmp(prefill_mode,"input_half") && strcmp(prefill_mode,"input_half_prefetch") && strcmp(prefill_mode,"copies") && strcmp(prefill_mode,"production_copy")) return 1;
        }
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
        if (!prefill_states_equal(sessions)) goto done;
        status=0;
done:
        if (prefill_iq3[0] && prefill_iq4[0]) prefill_select(0);
        if (status) fprintf(stderr,"prefill staging benchmark failed: %s\n",error);
        for (int v=0;v<2;v++) {free(logits[v]);if (sessions[v]) qw3_session_free(sessions[v]);}
        qw3_tokens_free(&prompt);if (engine) qw3_engine_close(engine);
        return status;
    }
}
