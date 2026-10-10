/* Private gate scheduling against the native cache64 baseline. */
#define qw3_metal_session_deltanet_fused_gdn_from_scratch local_original_gdn
#include "../qw3_metal.m"
#undef qw3_metal_session_deltanet_fused_gdn_from_scratch
static uint32_t local_variant=1;
static int local_selected;
static uint64_t local_calls,local_baseline_calls;
static id<MTLComputePipelineState> local_base,local_fallback,local_pipeline[2];
static int local_compile(void) {
    if (local_pipeline[1]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    local_base=g_deltanet_fused_gdn_cache64_pipeline;
    local_fallback=g_deltanet_fused_gdn_scratch_pipeline;
    if (!local_base || !local_fallback) return 0;
    NSError *error=nil;
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.gdn_local_gates/kernels.metal"
        encoding:NSUTF8StringEncoding error:&error];
    if (!extra) return 0;
    NSMutableString *source=[qw3_metal_kernel_source() mutableCopy];[source appendString:extra];
    MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    id<MTLLibrary> lib=[g_device newLibraryWithSource:source options:options error:&error];
    NSArray *names=@[@"private_gdn_local_all",@"private_gdn_local_simd"];
    for (int i=0;i<2;i++) {
        id<MTLFunction> fn=[lib newFunctionWithName:names[i]];
        local_pipeline[i]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!local_pipeline[i] || local_pipeline[i].maxTotalThreadsPerThreadgroup<128u) {
            fprintf(stderr,"GDN local gates compile: %s\n",error.localizedDescription.UTF8String);return 0;
        }
    }
    return 1;
}
static void scratch_select(int candidate) {
    setenv("QW3_METAL_DENSE_GDN_CACHE","0",1);
    setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1);
    local_selected=candidate!=0;
}
int qw3_metal_session_deltanet_fused_gdn_from_scratch(
    qw3_metal_session *s,uint64_t dt,uint64_t a,uint64_t w,
    uint32_t z,uint32_t alpha,uint32_t beta,uint32_t slot,
    uint32_t q_heads,uint32_t v_heads,uint32_t head_dim,float eps) {
    QW3MetalSessionObj *obj=s ? (__bridge QW3MetalSessionObj *)s->obj : nil;
    const int supported=obj.densePrefill && head_dim==128u;
    local_calls+=supported && local_selected;
    local_baseline_calls+=supported && !local_selected;
    id<MTLComputePipelineState> saved=g_deltanet_fused_gdn_scratch_pipeline;
    g_deltanet_fused_gdn_scratch_pipeline=supported ?
        (local_selected ? local_pipeline[local_variant-1] : local_base) : local_fallback;
    int ok=local_original_gdn(s,dt,a,w,z,alpha,beta,slot,q_heads,v_heads,head_dim,eps);
    g_deltanet_fused_gdn_scratch_pipeline=saved;
    return ok;
}
#define qw3_metal_env_n_gpu_layers local_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled local_core_llama_split_enabled
#define qw3_metal_llama_split_start local_core_llama_split_start
#include "../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
