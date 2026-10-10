#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static int stage_width=64;
static uint64_t stage_iq3_calls,stage_iq4_calls;
static id<MTLComputePipelineState> stage_iq3_base,stage_iq4_base;
static id<MTLComputePipelineState> stage_iq3[2],stage_iq4[2];
static NSUInteger stage_pipeline_memory(id<MTLComputePipelineState> pipeline) {
    for (int i=0;i<2;i++) {
        if (pipeline==stage_iq3[i] && pipeline) { stage_iq3_calls++;return (i ? 128u : 64u)*64u*sizeof(__fp16); }
        if (pipeline==stage_iq4[i] && pipeline) { stage_iq4_calls++;return (i ? 128u : 64u)*64u*sizeof(__fp16); }
    }
    return 8192u;
}
#include ".kstage_candidates/bridge.inc"
#define qw3_metal_env_n_gpu_layers stage_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled stage_core_llama_split_enabled
#define qw3_metal_llama_split_start stage_core_llama_split_start
#include "../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int stage_compile(void) {
    if (stage_iq3[0] && stage_iq4[1]) return 1;
    if (!qw3_metal_compile_kernels() || !g_metal4_tensor_api_enabled) return 0;
    stage_iq3_base=g_matmul_iq3_s_mpp_staged128_pipeline;
    stage_iq4_base=g_matmul_iq4_xs_mpp_staged128_pipeline;
    if (!stage_iq3_base || !stage_iq4_base) return 0;
    NSError *error=nil;
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.kstage_candidates/kernels.metal"
        encoding:NSUTF8StringEncoding error:&error];
    if (!extra) return 0;
    NSMutableString *source=[qw3_metal_kernel_source() mutableCopy];[source appendString:extra];
    MTLCompileOptions *opt=[MTLCompileOptions new];opt.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    id<MTLLibrary> lib=[g_device newLibraryWithSource:source options:opt error:&error];
    for (int i=0;i<2;i++) {
        NSString *names[]={[NSString stringWithFormat:@"kstage_iq3_s_%d",i ? 128 : 64],
            [NSString stringWithFormat:@"kstage_iq4_xs_%d",i ? 128 : 64]};
        id<MTLComputePipelineState> p[2]={nil,nil};
        for (int j=0;j<2 && lib;j++) {
            id<MTLFunction> fn=[lib newFunctionWithName:names[j]];
            if (fn) p[j]=[g_device newComputePipelineStateWithFunction:fn error:&error];
        }
        stage_iq3[i]=p[0];stage_iq4[i]=p[1];
        if (!p[0] || !p[1]) { fprintf(stderr,"K-stage compile: %s\n",error.localizedDescription.UTF8String);return 0; }
    }
    return 1;
}
static void scratch_select(int mode) {
    setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1);
    int i=stage_width==128;
    g_matmul_iq3_s_mpp_staged128_pipeline=(mode&1) ? stage_iq3[i] : stage_iq3_base;
    g_matmul_iq4_xs_mpp_staged128_pipeline=(mode&2) ? stage_iq4[i] : stage_iq4_base;
}
