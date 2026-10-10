/* Private two-column geometry; original scalar cache64 is the baseline. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
static uint32_t two_variant=1;
static int two_selected,two_live;
static uint64_t two_calls,two_baseline_calls;
static id<MTLComputePipelineState> two_base,two_pipeline[3];
#define QW3_METAL_GDN_PIPELINE(pipeline) \
    (two_live=two_selected && obj.densePrefill && head_dim==128u, \
     two_calls+=two_live, \
     two_baseline_calls+=!two_live && (pipeline)==g_deltanet_fused_gdn_cache64_pipeline, \
     two_live ? two_pipeline[two_variant-1] : (pipeline))
#include ".gdn_two_columns/core.inc"
static int two_compile(void) {
    if (two_pipeline[2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    two_base=g_deltanet_fused_gdn_cache64_pipeline;
    if (!two_base) return 0;
    NSError *error=nil;
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.gdn_two_columns/kernels.metal"
        encoding:NSUTF8StringEncoding error:&error];
    if (!extra) return 0;
    NSMutableString *source=[qw3_metal_kernel_source() mutableCopy];[source appendString:extra];
    MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    id<MTLLibrary> lib=[g_device newLibraryWithSource:source options:options error:&error];
    NSArray *names=@[@"private_gdn_two_cache32",@"private_gdn_two_cache48",@"private_gdn_two_cache64"];
    for (int i=0;i<3;i++) {
        id<MTLFunction> fn=[lib newFunctionWithName:names[i]];
        two_pipeline[i]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!two_pipeline[i] || two_pipeline[i].maxTotalThreadsPerThreadgroup<64u) {
            fprintf(stderr,"GDN two columns compile: %s\n",error.localizedDescription.UTF8String);return 0;
        }
    }
    return 1;
}
static void scratch_select(int candidate) {
    setenv("QW3_METAL_DENSE_GDN_CACHE","1",1);
    setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1);
    two_selected=candidate!=0;
}
#define qw3_metal_env_n_gpu_layers two_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled two_core_llama_split_enabled
#define qw3_metal_llama_split_start two_core_llama_split_start
#include "../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
