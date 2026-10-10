#define qw3_metal_session_deltanet_fused_gdn_from_scratch cache_original_gdn
#include "../qw3_metal.m"
#undef qw3_metal_session_deltanet_fused_gdn_from_scratch
static uint32_t cache_width=64;
static int cache_selected;
static uint64_t cache_calls;
static id<MTLComputePipelineState> cache_base,cache_pipeline[6];
static int cache_compile(void) {
    if (cache_pipeline[5]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    cache_base=g_deltanet_fused_gdn_scratch_pipeline;
    NSError *error=nil;
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.gdn_cache/kernels.metal"
        encoding:NSUTF8StringEncoding error:&error];
    if (!extra) return 0;
    NSMutableString *source=[qw3_metal_kernel_source() mutableCopy];[source appendString:extra];
    MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    id<MTLLibrary> lib=[g_device newLibraryWithSource:source options:options error:&error];
    NSArray *names=@[@"private_gdn_cache0",@"private_gdn_cache32",@"private_gdn_cache64",@"private_gdn_cache128",@"private_gdn_poison",@"private_gdn_unroll"];
    for (int i=0;i<6;i++) {
        id<MTLFunction> fn=[lib newFunctionWithName:names[i]];
        cache_pipeline[i]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!cache_pipeline[i] || cache_pipeline[i].maxTotalThreadsPerThreadgroup<128u) {
            fprintf(stderr,"GDN cache compile: %s\n",error.localizedDescription.UTF8String);return 0;
        }
    }
    return cache_base!=nil;
}
static void scratch_select(int candidate) {
    /* Keep native cache64 from masking the diagnostic's selected pipeline. */
    setenv("QW3_METAL_DENSE_GDN_CACHE","0",1);
    cache_selected=candidate!=0;
    unsigned i=cache_width==0 ? 0 : cache_width==16 ? 5 : cache_width==32 ? 1 : cache_width==64 ? 2 : 3;
    g_deltanet_fused_gdn_scratch_pipeline=candidate ? cache_pipeline[i] : cache_base;
    setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1);
}
int qw3_metal_session_deltanet_fused_gdn_from_scratch(
    qw3_metal_session *s,uint64_t dt,uint64_t a,uint64_t w,
    uint32_t z,uint32_t alpha,uint32_t beta,uint32_t slot,
    uint32_t q_heads,uint32_t v_heads,uint32_t head_dim,float eps) {
    const int active=cache_selected && (cache_width==0 || head_dim==128u);
    cache_calls+=active;
    id<MTLComputePipelineState> selected=g_deltanet_fused_gdn_scratch_pipeline;
    if (!active && cache_base) g_deltanet_fused_gdn_scratch_pipeline=cache_base;
    int ok=cache_original_gdn(s,dt,a,w,z,alpha,beta,slot,q_heads,v_heads,head_dim,eps);
    g_deltanet_fused_gdn_scratch_pipeline=selected;
    return ok;
}
#define qw3_metal_env_n_gpu_layers cache_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled cache_core_llama_split_enabled
#define qw3_metal_llama_split_start cache_core_llama_split_start
#include "../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
