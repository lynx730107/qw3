/* Private probes: same native cache64 baseline and unchanged FP32 ordering. */
#define qw3_metal_session_deltanet_fused_gdn_from_scratch window_original_gdn
#include "../qw3_metal.m"
#undef qw3_metal_session_deltanet_fused_gdn_from_scratch
static uint32_t window_variant=1;
static int window_selected;
static uint64_t window_calls,window_baseline_calls;
static id<MTLComputePipelineState> window_base,window_fallback,window_pipeline[5];
static int window_compile(void) {
    if (window_pipeline[4]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    window_base=g_deltanet_fused_gdn_cache64_pipeline;
    window_fallback=g_deltanet_fused_gdn_scratch_pipeline;
    if (!window_base || !window_fallback) return 0;
    NSError *error=nil;
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.gdn_windows/kernels.metal"
        encoding:NSUTF8StringEncoding error:&error];
    if (!extra) return 0;
    NSMutableString *source=[qw3_metal_kernel_source() mutableCopy];[source appendString:extra];
    MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    id<MTLLibrary> lib=[g_device newLibraryWithSource:source options:options error:&error];
    NSArray *names=@[@"private_gdn_first48",@"private_gdn_first80",@"private_gdn_first96",
                     @"private_gdn_middle64",@"private_gdn_last64"];
    for (int i=0;i<5;i++) {
        id<MTLFunction> fn=[lib newFunctionWithName:names[i]];
        window_pipeline[i]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!window_pipeline[i] || window_pipeline[i].maxTotalThreadsPerThreadgroup<128u) {
            fprintf(stderr,"GDN windows compile: %s\n",error.localizedDescription.UTF8String);return 0;
        }
    }
    return 1;
}
static void scratch_select(int candidate) {
    setenv("QW3_METAL_DENSE_GDN_CACHE","0",1);
    setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1);
    window_selected=candidate!=0;
}
int qw3_metal_session_deltanet_fused_gdn_from_scratch(
    qw3_metal_session *s,uint64_t dt,uint64_t a,uint64_t w,
    uint32_t z,uint32_t alpha,uint32_t beta,uint32_t slot,
    uint32_t q_heads,uint32_t v_heads,uint32_t head_dim,float eps) {
    QW3MetalSessionObj *obj=s ? (__bridge QW3MetalSessionObj *)s->obj : nil;
    const int supported=obj.densePrefill && head_dim==128u;
    window_calls+=supported && window_selected;
    window_baseline_calls+=supported && !window_selected;
    id<MTLComputePipelineState> saved=g_deltanet_fused_gdn_scratch_pipeline;
    g_deltanet_fused_gdn_scratch_pipeline=supported ?
        (window_selected ? window_pipeline[window_variant-1] : window_base) : window_fallback;
    int ok=window_original_gdn(s,dt,a,w,z,alpha,beta,slot,q_heads,v_heads,head_dim,eps);
    g_deltanet_fused_gdn_scratch_pipeline=saved;
    return ok;
}
#define qw3_metal_env_n_gpu_layers window_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled window_core_llama_split_enabled
#define qw3_metal_llama_split_start window_core_llama_split_start
#include "../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
