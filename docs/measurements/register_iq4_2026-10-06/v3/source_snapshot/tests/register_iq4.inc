#include "../qw3_metal.m"

static id<MTLComputePipelineState> register_iq4_pipeline;
static int register_iq4_compile(void) {
    if (register_iq4_pipeline) return 1;
    if (!qw3_metal_compile_kernels() || !g_matmul_iq4_xs_mpp_staged128_pipeline) return 0;
    NSError *error=nil;
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.register_iq4/kernel.metal"
        encoding:NSUTF8StringEncoding error:&error];
    if (!extra) return 0;
    NSMutableString *source=[qw3_metal_kernel_source() mutableCopy];[source appendString:extra];
    MTLCompileOptions *opt=[MTLCompileOptions new];
    opt.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    id<MTLLibrary> lib=[g_device newLibraryWithSource:source options:opt error:&error];
    if (lib) register_iq4_pipeline=[g_device newComputePipelineStateWithFunction:
        [lib newFunctionWithName:@"register_iq4_staged128"] error:&error];
    if (!register_iq4_pipeline) fprintf(stderr,"register IQ4 compile: %s\n",error.localizedDescription.UTF8String);
    return register_iq4_pipeline!=nil;
}

static int register_iq4_encode(QW3MetalSessionObj *obj,id<MTLCommandBuffer> cb,int candidate,
    uint64_t offset,uint32_t nt,uint32_t ni,uint32_t no,id<MTLBuffer> src,id<MTLBuffer> dst,
    uint32_t io,uint32_t is,uint32_t oo,uint32_t os) {
    if (!candidate) return qw3_metal_encode_dense_batch_matmul_strided(obj,cb,offset,23u,
        nt,ni,no,src,dst,io,is,oo,os);
    if (!obj || !cb || !src || !dst || !nt || nt%128u || !ni || ni%256u || !no ||
        io>is || ni>is-io || oo>os || no>os-oo ||
        ((uint64_t)(nt-1u)*is+io+ni)*4u>src.length ||
        ((uint64_t)(nt-1u)*os+oo+no)*4u>dst.length || !register_iq4_compile()) return 0;
    const uint64_t row=(uint64_t)(ni/256u)*136u;uint64_t inner=0;
    id<MTLBuffer> weights=qw3_metal_model_view_for(offset,row*no,&inner);
    if (!weights || row>UINT32_MAX) return 0;
    struct {uint32_t ni,no,row,nt,io,is,oo,os;} a={ni,no,(uint32_t)row,nt,io,is,oo,os};
    id<MTLComputeCommandEncoder> enc=qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:register_iq4_pipeline];
    [enc setBytes:&a length:sizeof(a) atIndex:0];[enc setBuffer:weights offset:inner atIndex:1];
    [enc setBuffer:src offset:0 atIndex:2];[enc setBuffer:dst offset:0 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(nt/128u,(no+63u)/64u,1)
        threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    qw3_metal_end_compute_encoder(cb,enc);return 1;
}

#define qw3_metal_env_n_gpu_layers register_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled register_core_llama_split_enabled
#define qw3_metal_llama_split_start register_core_llama_split_start
#include "../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
