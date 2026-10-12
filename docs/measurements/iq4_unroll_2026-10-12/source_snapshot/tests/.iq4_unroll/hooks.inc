/* Three IQ4_XS pipelines: K-loop unroll hints only; native arithmetic and buffers. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> iq4_unroll_pipeline[3][3];
static uint64_t iq4_unroll_calls,iq4_unroll_calls_by_kind[3];
static int unroll_variant=1;
@protocol QW3IQ4UnrollHooks
- (void)iq4UnrollSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq4UnrollEncoder;
- (id)iq4UnrollEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq4UnrollCommandBuffer;
- (id)iq4UnrollCommandBufferUnretained;
- (id)iq4UnrollCommandQueue;
@end
@interface NSObject (QW3IQ4UnrollProbe) <QW3IQ4UnrollHooks>
@end
@implementation NSObject (QW3IQ4UnrollProbe)
- (id)iq4UnrollCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq4UnrollCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq4UnrollCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq4UnrollEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq4UnrollEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq4UnrollSetPipeline:(id<MTLComputePipelineState>)pipeline {
    for (int v=1;v<=2;v++) for (int kind=0;kind<3;kind++)
        if (pipeline==iq4_unroll_pipeline[v][kind]) {iq4_unroll_calls++;iq4_unroll_calls_by_kind[kind]++;}
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
@end
static id iq4_unroll_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq4_unroll_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq4_unroll_default_device
#define MTLCopyAllDevices iq4_unroll_all_devices
#define MTLDevice MTLDevice,QW3IQ4UnrollHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ4UnrollHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ4UnrollHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ4UnrollHooks
#define newCommandQueue iq4UnrollCommandQueue
#define commandBuffer iq4UnrollCommandBuffer
#define commandBufferWithUnretainedReferences iq4UnrollCommandBufferUnretained
#define setComputePipelineState iq4UnrollSetPipeline
#define computeCommandEncoder iq4UnrollEncoder
#define computeCommandEncoderWithDispatchType iq4UnrollEncoderWithDispatchType
#include "../../qw3_metal.m"
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
#define qw3_metal_env_n_gpu_layers iq4_unroll_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq4_unroll_core_llama_split_enabled
#define qw3_metal_llama_split_start iq4_unroll_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int iq4_unroll_compile(void) {
    if (iq4_unroll_pipeline[2][2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq4_unroll_pipeline[0][0]=g_matvec_iq4_xs_pipeline;
    iq4_unroll_pipeline[0][1]=g_matvec_iq4_xs_pair_pipeline;
    iq4_unroll_pipeline[0][2]=g_matvec_iq4_xs_add_x0_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.iq4_unroll/kernels.metal"
        encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) for (int kind=0;kind<3;kind++) {
        NSString *name=[NSString stringWithFormat:@"private_iq4_unroll_fast%d_%d",v,kind];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        id<MTLComputePipelineState> p=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!p || p.maxTotalThreadsPerThreadgroup<64u || p.threadExecutionWidth!=32 ||
            p.staticThreadgroupMemoryLength!=iq4_unroll_pipeline[0][kind].staticThreadgroupMemoryLength) {
            fprintf(stderr,"IQ4 unroll compile variant=%d kind=%d: %s\n",v,kind,error.localizedDescription.UTF8String);return 0;
        }
        iq4_unroll_pipeline[v][kind]=p;
    }
    for (int v=0;v<=2;v++) for (int kind=0;kind<3;kind++) {
        id<MTLComputePipelineState> p=iq4_unroll_pipeline[v][kind];
        fprintf(stderr,"iq4_unroll_metadata variant=%d kind=%d thread_width=%lu max_threads=%lu static_threadgroup_bytes=%lu rows_per_simd=2 simdgroups=2 threads=64 dynamic_threadgroup_bytes=128 loop_unroll=%d\n",
            v,kind,(unsigned long)p.threadExecutionWidth,(unsigned long)p.maxTotalThreadsPerThreadgroup,
            (unsigned long)p.staticThreadgroupMemoryLength,v==1 ? 2 : v==2 ? 4 : 0);
    }
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    int v=mode ? unroll_variant : 0;
    g_matvec_iq4_xs_pipeline=iq4_unroll_pipeline[v][0];
    g_matvec_iq4_xs_pair_pipeline=iq4_unroll_pipeline[v][1];
    g_matvec_iq4_xs_add_x0_pipeline=iq4_unroll_pipeline[v][2];
}
static __attribute__((unused)) void iq4_unroll_print_counts(void) {
    fprintf(stderr,"iq4_unroll_dispatch_counts single=%llu pair=%llu add=%llu\n",
        (unsigned long long)iq4_unroll_calls_by_kind[0],(unsigned long long)iq4_unroll_calls_by_kind[1],
        (unsigned long long)iq4_unroll_calls_by_kind[2]);
}
