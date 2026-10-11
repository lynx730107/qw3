/* Three IQ4_XS pipelines: native row arithmetic, LUT and data buffers. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> iq4_share_pipeline[3][3];
static uint64_t iq4_share_calls,iq4_share_calls_by_kind[3];
static int share_variant=1;
@protocol QW3IQ4ShareHooks
- (void)iq4ShareSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq4ShareEncoder;
- (id)iq4ShareEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq4ShareCommandBuffer;
- (id)iq4ShareCommandBufferUnretained;
- (id)iq4ShareCommandQueue;
@end
@interface NSObject (QW3IQ4ShareProbe) <QW3IQ4ShareHooks>
@end
@implementation NSObject (QW3IQ4ShareProbe)
- (id)iq4ShareCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq4ShareCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq4ShareCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq4ShareEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq4ShareEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq4ShareSetPipeline:(id<MTLComputePipelineState>)pipeline {
    for (int v=1;v<=2;v++) for (int kind=0;kind<3;kind++)
        if (pipeline==iq4_share_pipeline[v][kind]) {iq4_share_calls++;iq4_share_calls_by_kind[kind]++;}
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
@end
static id iq4_share_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq4_share_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq4_share_default_device
#define MTLCopyAllDevices iq4_share_all_devices
#define MTLDevice MTLDevice,QW3IQ4ShareHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ4ShareHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ4ShareHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ4ShareHooks
#define newCommandQueue iq4ShareCommandQueue
#define commandBuffer iq4ShareCommandBuffer
#define commandBufferWithUnretainedReferences iq4ShareCommandBufferUnretained
#define setComputePipelineState iq4ShareSetPipeline
#define computeCommandEncoder iq4ShareEncoder
#define computeCommandEncoderWithDispatchType iq4ShareEncoderWithDispatchType
static NSUInteger iq4_share_rows_per_group=4u;
static NSUInteger iq4_share_thread_count=64u;
static NSUInteger iq4_share_groups(NSUInteger n_out) {
    return (n_out+iq4_share_rows_per_group-1u)/iq4_share_rows_per_group;
}
static NSUInteger iq4_share_threads(void) { return iq4_share_thread_count; }
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
#define qw3_metal_env_n_gpu_layers iq4_share_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq4_share_core_llama_split_enabled
#define qw3_metal_llama_split_start iq4_share_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int iq4_share_compile(void) {
    if (iq4_share_pipeline[2][2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq4_share_pipeline[0][0]=g_matvec_iq4_xs_pipeline;
    iq4_share_pipeline[0][1]=g_matvec_iq4_xs_pair_pipeline;
    iq4_share_pipeline[0][2]=g_matvec_iq4_xs_add_x0_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.iq4_share/kernels.metal"
        encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) for (int kind=0;kind<3;kind++) {
        NSString *name=[NSString stringWithFormat:@"private_iq4_share_fast%d_%d",v,kind];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        id<MTLComputePipelineState> p=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!p || p.maxTotalThreadsPerThreadgroup<(v==1 ? 128u : 256u) || p.threadExecutionWidth!=32 ||
            p.staticThreadgroupMemoryLength!=iq4_share_pipeline[0][kind].staticThreadgroupMemoryLength) {
            fprintf(stderr,"IQ4 share compile variant=%d kind=%d: %s\n",v,kind,error.localizedDescription.UTF8String);return 0;
        }
        iq4_share_pipeline[v][kind]=p;
    }
    for (int v=0;v<=2;v++) for (int kind=0;kind<3;kind++) {
        id<MTLComputePipelineState> p=iq4_share_pipeline[v][kind];
        fprintf(stderr,"iq4_share_metadata variant=%d kind=%d thread_width=%lu max_threads=%lu static_threadgroup_bytes=%lu rows_per_simd=2 simdgroups=%d threads=%d dynamic_threadgroup_bytes=128\n",
            v,kind,(unsigned long)p.threadExecutionWidth,(unsigned long)p.maxTotalThreadsPerThreadgroup,
            (unsigned long)p.staticThreadgroupMemoryLength,v==1 ? 4 : v==2 ? 8 : 2,v==1 ? 128 : v==2 ? 256 : 64);
    }
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    int v=mode ? share_variant : 0;
    g_matvec_iq4_xs_pipeline=iq4_share_pipeline[v][0];
    g_matvec_iq4_xs_pair_pipeline=iq4_share_pipeline[v][1];
    g_matvec_iq4_xs_add_x0_pipeline=iq4_share_pipeline[v][2];
    iq4_share_rows_per_group=mode ? (share_variant==1 ? 8u : 16u) : 4u;
    iq4_share_thread_count=mode ? (share_variant==1 ? 128u : 256u) : 64u;
}
static __attribute__((unused)) void iq4_share_print_counts(void) {
    fprintf(stderr,"iq4_share_dispatch_counts single=%llu pair=%llu add=%llu\n",
        (unsigned long long)iq4_share_calls_by_kind[0],(unsigned long long)iq4_share_calls_by_kind[1],
        (unsigned long long)iq4_share_calls_by_kind[2]);
}
