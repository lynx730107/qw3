/* Two IQ3_S rows per SIMD group; 4/8 groups share the original packed grid. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> iq3_share_pipeline[3];
static uint64_t iq3_share_calls;
static int share_variant=1;
@protocol QW3IQ3ShareHooks
- (void)iq3ShareSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq3ShareEncoder;
- (id)iq3ShareEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq3ShareCommandBuffer;
- (id)iq3ShareCommandBufferUnretained;
- (id)iq3ShareCommandQueue;
@end
@interface NSObject (QW3IQ3ShareProbe) <QW3IQ3ShareHooks>
@end
@implementation NSObject (QW3IQ3ShareProbe)
- (id)iq3ShareCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq3ShareCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq3ShareCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq3ShareEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq3ShareEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq3ShareSetPipeline:(id<MTLComputePipelineState>)pipeline {
    if (pipeline==iq3_share_pipeline[1] || pipeline==iq3_share_pipeline[2]) iq3_share_calls++;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
@end
static id iq3_share_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq3_share_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq3_share_default_device
#define MTLCopyAllDevices iq3_share_all_devices
#define MTLDevice MTLDevice,QW3IQ3ShareHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ3ShareHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ3ShareHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ3ShareHooks
#define newCommandQueue iq3ShareCommandQueue
#define commandBuffer iq3ShareCommandBuffer
#define commandBufferWithUnretainedReferences iq3ShareCommandBufferUnretained
#define setComputePipelineState iq3ShareSetPipeline
#define computeCommandEncoder iq3ShareEncoder
#define computeCommandEncoderWithDispatchType iq3ShareEncoderWithDispatchType
static NSUInteger iq3_share_rows_per_group=8u;
static NSUInteger iq3_share_thread_count=64u;
static NSUInteger iq3_share_groups(NSUInteger n_out) {
    return (n_out+iq3_share_rows_per_group-1u)/iq3_share_rows_per_group;
}
static NSUInteger iq3_share_threads(void) { return iq3_share_thread_count; }
#include "core.inc"
#undef computeCommandEncoderWithDispatchType
#undef computeCommandEncoder
#undef setComputePipelineState
#undef MTLCommandBuffer
#undef MTLComputeCommandEncoder
#undef commandBufferWithUnretainedReferences
#undef commandBuffer
#undef newCommandQueue
#undef MTLCommandQueue
#undef MTLDevice
#undef MTLCopyAllDevices
#undef MTLCreateSystemDefaultDevice
#define qw3_metal_env_n_gpu_layers iq3_share_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq3_share_core_llama_split_enabled
#define qw3_metal_llama_split_start iq3_share_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int iq3_share_compile(void) {
    if (iq3_share_pipeline[2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq3_share_pipeline[0]=g_matvec_iq3_s_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.iq3_share/kernels.metal"
        encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) {
        NSString *name=[NSString stringWithFormat:@"private_iq3_share_fast%d",v];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        iq3_share_pipeline[v]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!iq3_share_pipeline[v] || iq3_share_pipeline[v].maxTotalThreadsPerThreadgroup<(v==1 ? 128u : 256u) ||
            iq3_share_pipeline[v].threadExecutionWidth!=32 ||
            iq3_share_pipeline[v].staticThreadgroupMemoryLength!=iq3_share_pipeline[0].staticThreadgroupMemoryLength) {
            fprintf(stderr,"IQ3 share compile variant=%d: %s\n",v,error.localizedDescription.UTF8String);return 0;
        }
    }
    for (int v=0;v<=2;v++) {
        id<MTLComputePipelineState> p=iq3_share_pipeline[v];
        fprintf(stderr,"iq3_share_metadata variant=%d thread_width=%lu max_threads=%lu static_threadgroup_bytes=%lu rows_per_simd=%d simdgroups=%d threads=%d\n",
            v,(unsigned long)p.threadExecutionWidth,(unsigned long)p.maxTotalThreadsPerThreadgroup,
            (unsigned long)p.staticThreadgroupMemoryLength,v ? 2 : 4,v==1 ? 4 : v==2 ? 8 : 2,v==1 ? 128 : v==2 ? 256 : 64);
    }
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    g_matvec_iq3_s_pipeline=iq3_share_pipeline[mode ? share_variant : 0];
    iq3_share_rows_per_group=mode && share_variant==2 ? 16u : 8u;
    iq3_share_thread_count=mode ? (share_variant==1 ? 128u : 256u) : 64u;
}
