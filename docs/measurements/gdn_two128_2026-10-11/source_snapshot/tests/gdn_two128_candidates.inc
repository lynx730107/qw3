/* Original buffers and dispatch; two columns in half the 128 threads. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
static int two128_variant=1,two128_selected;
static uint64_t two128_calls;
static id<MTLComputePipelineState> two128_base,two128_pipeline[2];
#define QW3_METAL_GDN_PIPELINE(pipeline) \
    ((two128_selected && (pipeline)==g_deltanet_fused_gdn_cache64_pipeline) \
        ? (two128_calls++,two128_pipeline[two128_variant-1]) : (pipeline))
#include "../qw3_metal.m"
static int two128_compile(void) {
    if (two128_pipeline[1]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    two128_base=g_deltanet_fused_gdn_cache64_pipeline;
    if (!two128_base) return 0;
    NSError *error=nil;
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.gdn_two128/kernels.metal"
        encoding:NSUTF8StringEncoding error:&error];
    if (!extra) return 0;
    NSMutableString *source=[qw3_metal_full_kernel_source() mutableCopy];[source appendString:extra];
    MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    id<MTLLibrary> lib=[g_device newLibraryWithSource:source options:options error:&error];
    if (!lib) {fprintf(stderr,"GDN two128 library: %s\n",error.localizedDescription.UTF8String);return 0;}
    NSArray *names=@[@"private_gdn_two128_cache32",@"private_gdn_two128_cache64"];
    for (int i=0;i<2;i++) {
        id<MTLFunction> fn=[lib newFunctionWithName:names[i]];
        two128_pipeline[i]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!two128_pipeline[i] || two128_pipeline[i].maxTotalThreadsPerThreadgroup<128u ||
            two128_pipeline[i].threadExecutionWidth!=two128_base.threadExecutionWidth ||
            two128_pipeline[i].staticThreadgroupMemoryLength!=two128_base.staticThreadgroupMemoryLength) {
            fprintf(stderr,"GDN two128 pipeline: %s\n",error.localizedDescription.UTF8String);return 0;
        }
    }
    for (int v=0;v<=2;v++) {
        id<MTLComputePipelineState> p=v ? two128_pipeline[v-1] : two128_base;
        fprintf(stderr,"gdn_two128_metadata variant=%d thread_width=%lu max_threads=%lu static_threadgroup_bytes=%lu dispatch_threads=128 dynamic_threadgroup_bytes=128\n",
                v,(unsigned long)p.threadExecutionWidth,(unsigned long)p.maxTotalThreadsPerThreadgroup,
                (unsigned long)p.staticThreadgroupMemoryLength);
    }
    return 1;
}
static __attribute__((unused)) void scratch_select(int candidate) {
    setenv("QW3_METAL_DENSE_GDN_CACHE","1",1);
    setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1);
    two128_selected=candidate!=0;
}
#define qw3_metal_env_n_gpu_layers two128_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled two128_core_llama_split_enabled
#define qw3_metal_llama_split_start two128_core_llama_split_start
#include "../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
