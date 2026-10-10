/* Private graph generated with exact anchors; baseline keeps production dispatch. */
#include "../qw3_metal.m"
static int scratch_mode;
static uint64_t micro_prefill_calls, micro_decode_calls, micro_conv_calls;
static id<MTLComputePipelineState> micro_batch_pipeline, micro_decode_pipeline, micro_conv_pipeline;
static void scratch_select(int mode) { scratch_mode=mode; setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1); }
static int micro_compile(void) {
    if (micro_batch_pipeline && micro_decode_pipeline && micro_conv_pipeline) return 1;
    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;
    NSError *error=nil;
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/micro_candidates.metal" encoding:NSUTF8StringEncoding error:&error];
    if (!extra) return 0;
    NSMutableString *source=[qw3_metal_kernel_source() mutableCopy];
    [source appendString:extra];
    MTLCompileOptions *opt=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) opt.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    id<MTLLibrary> lib=[g_device newLibraryWithSource:source options:opt error:&error];
    if (!lib) { fprintf(stderr,"micro library compile: %s\n",error.localizedDescription.UTF8String); return 0; }
    micro_batch_pipeline=[g_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"micro_q8_batch_pair"] error:&error];
    micro_decode_pipeline=[g_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"micro_q8_decode_pair"] error:&error];
    micro_conv_pipeline=[g_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"micro_conv_norm"] error:&error];
    if (!micro_batch_pipeline || !micro_decode_pipeline) fprintf(stderr,"micro compile: %s\n",error.localizedDescription.UTF8String);
    return micro_batch_pipeline && micro_decode_pipeline && micro_conv_pipeline;
}
static int micro_batch_pair(qw3_metal_session *s, uint64_t a, uint64_t b,
    uint32_t nt, uint32_t ni, uint32_t no, uint32_t oa, uint32_t ob, uint32_t stride) {
    if (!(scratch_mode&1) || no>=64u) return
        qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(s,a,nt,ni,no,oa,stride) &&
        qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(s,b,nt,ni,no,ob,stride);
    if (!s || !s->obj || !nt || !ni || !no || ni%32u || oa>stride || ob>stride || no>stride-oa || no>stride-ob || !micro_compile()) return 0;
    QW3MetalSessionObj *obj=(__bridge QW3MetalSessionObj *)s->obj;
    if (!obj.prefillX1 || obj.prefillX1.length<(uint64_t)nt*ni*4ull ||
        !qw3_metal_session_ensure_prefill_scratch(obj,nt,(uint64_t)nt*stride*4ull)) return 0;
    uint64_t row=(uint64_t)(ni/32u)*34ull, ia=0,ib=0;
    id<MTLBuffer> wa=qw3_metal_model_view_for(a,row*no,&ia),wb=qw3_metal_model_view_for(b,row*no,&ib);
    if (!wa || !wb) return 0;
    struct { uint32_t ni,no,row,nt,is,os,oa,ob; } args={ni,no,(uint32_t)row,nt,ni,stride,oa,ob};
    int owned=0;
    id<MTLCommandBuffer> cb=qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc=qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:micro_batch_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wa offset:ia atIndex:1]; [enc setBuffer:wb offset:ib atIndex:2];
    [enc setBuffer:obj.prefillX1 offset:0 atIndex:3]; [enc setBuffer:obj.prefillScratch offset:0 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((no+3u)/4u,(nt+3u)/4u,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];
    qw3_metal_end_compute_encoder(cb,enc); micro_prefill_calls++;
    return qw3_metal_finish_command_buffer(cb,owned,"micro batch pair") && cb.status!=MTLCommandBufferStatusError;
}
static int micro_decode_pair(qw3_metal_session *s, uint64_t a, uint64_t b,
    uint32_t ta, uint32_t tb, uint32_t ni, uint32_t no, uint32_t oa, uint32_t ob) {
    if (!(scratch_mode&2) || ta!=8u || tb!=8u) return
        qw3_metal_session_matvec_quant_x1_to_scratch(s,a,ta,ni,no,oa,NULL) &&
        qw3_metal_session_matvec_quant_x1_to_scratch(s,b,tb,ni,no,ob,NULL);
    if (!s || !s->obj || !ni || !no || ni%32u || !micro_compile()) return 0;
    QW3MetalSessionObj *obj=(__bridge QW3MetalSessionObj *)s->obj;
    if (!obj.x1 || !obj.scratch || obj.x1.length<(uint64_t)ni*4ull ||
        obj.scratch.length<((uint64_t)oa+no)*4ull || obj.scratch.length<((uint64_t)ob+no)*4ull) return 0;
    uint64_t row=(uint64_t)(ni/32u)*34ull,ia=0,ib=0;
    id<MTLBuffer> wa=qw3_metal_model_view_for(a,row*no,&ia),wb=qw3_metal_model_view_for(b,row*no,&ib);
    if (!wa || !wb) return 0;
    struct { uint32_t ni,no,row,oa,ob; } args={ni,no,(uint32_t)row,oa,ob};
    int owned=0;
    id<MTLCommandBuffer> cb=qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc=qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:micro_decode_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wa offset:ia atIndex:1]; [enc setBuffer:wb offset:ib atIndex:2];
    [enc setBuffer:obj.x1 offset:0 atIndex:3]; [enc setBuffer:obj.scratch offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
    NSUInteger threads=qw3_metal_matvec_threads(g_matvec_q8_0_pipeline,ni/32u);
    if (threads>micro_decode_pipeline.maxTotalThreadsPerThreadgroup) return 0;
    [enc dispatchThreadgroups:MTLSizeMake(no*2u,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    qw3_metal_end_compute_encoder(cb,enc); micro_decode_calls++;
    return qw3_metal_finish_command_buffer(cb,owned,"micro decode pair") && cb.status!=MTLCommandBufferStatusError;
}
static int micro_conv_norm(qw3_metal_session *s, uint64_t weight, uint32_t slot,
    uint32_t nt, uint32_t channels, uint32_t qkv, uint32_t conv, uint32_t stride,
    uint32_t qheads, uint32_t dim, float eps) {
    if (!(scratch_mode&4) || dim!=128u) return
        qw3_metal_session_batch_conv1d_step_from_scratch(s,weight,slot,nt,channels,qkv,conv,stride) &&
        qw3_metal_session_batch_l2norm_qk_from_scratch(s,nt,conv,stride,qheads,dim,eps);
    if (!s || !s->obj || !nt || !channels || !qheads || channels%dim ||
        qkv>stride || conv>stride || channels>stride-qkv || channels>stride-conv ||
        !qw3_metal_session_linear_slot_valid(s,slot) || !micro_compile()) return 0;
    QW3MetalSessionObj *obj=(__bridge QW3MetalSessionObj *)s->obj;
    uint64_t state_bytes=(uint64_t)channels*3ull*4ull,state_offset=slot*state_bytes;
    if (!obj.prefillScratch || obj.prefillScratch.length<(uint64_t)nt*stride*4ull ||
        !obj.convState || obj.convState.length<state_offset+state_bytes) return 0;
    uint64_t wi=0; id<MTLBuffer> wb=qw3_metal_model_view_for(weight,(uint64_t)channels*4ull*4ull,&wi);
    if (!wb) return 0;
    struct { uint32_t channels,nt,qkv,conv,stride,qheads,dim; float eps; } args={channels,nt,qkv,conv,stride,qheads,dim,eps};
    int owned=0;id<MTLCommandBuffer> cb=qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc=qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:micro_conv_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:wi atIndex:1]; [enc setBuffer:obj.prefillScratch offset:0 atIndex:2];
    [enc setBuffer:obj.convState offset:state_offset atIndex:3];
    [enc setThreadgroupMemoryLength:32u*sizeof(float) atIndex:0];
    if (micro_conv_pipeline.maxTotalThreadsPerThreadgroup<256u) return 0;
    [enc dispatchThreadgroups:MTLSizeMake(channels/dim,nt,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    struct { uint32_t channels,nt,qkv,conv,stride; } state_args={channels,nt,qkv,conv,stride};
    [enc setComputePipelineState:g_deltanet_conv1d_batch_state_pipeline];
    [enc setBytes:&state_args length:sizeof(state_args) atIndex:0];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:1]; [enc setBuffer:obj.convState offset:state_offset atIndex:2];
    NSUInteger threads=g_deltanet_conv1d_batch_state_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads>256u) threads=256u;
    [enc dispatchThreads:MTLSizeMake(channels,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    qw3_metal_end_compute_encoder(cb,enc);micro_conv_calls++;
    return qw3_metal_finish_command_buffer(cb,owned,"micro conv norm") && cb.status!=MTLCommandBufferStatusError;
}
#define qw3_metal_env_n_gpu_layers micro_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled micro_core_llama_split_enabled
#define qw3_metal_llama_split_start micro_core_llama_split_start
#include ".micro_candidates/core.inc"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
