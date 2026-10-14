/* Original Q5_K shader/data buffers; descriptor hints only. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> q5_hint_pipeline[3];
static uint64_t q5_hint_calls;
static int hint_variant=1;
@protocol QW3Q5HintHooks
- (void)q5HintSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)q5HintEncoder;
- (id)q5HintEncoderWithDispatchType:(MTLDispatchType)type;
- (id)q5HintCommandBuffer;
- (id)q5HintCommandBufferUnretained;
- (id)q5HintCommandQueue;
@end
@interface NSObject (QW3Q5HintProbe) <QW3Q5HintHooks>
@end
@implementation NSObject (QW3Q5HintProbe)
- (id)q5HintCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)q5HintCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)q5HintCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)q5HintEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)q5HintEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)q5HintSetPipeline:(id<MTLComputePipelineState>)pipeline {
    if (pipeline==q5_hint_pipeline[1] || pipeline==q5_hint_pipeline[2]) q5_hint_calls++;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
@end
static id q5_hint_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id q5_hint_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice q5_hint_default_device
#define MTLCopyAllDevices q5_hint_all_devices
#define MTLDevice MTLDevice,QW3Q5HintHooks
#define MTLCommandQueue MTLCommandQueue,QW3Q5HintHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3Q5HintHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3Q5HintHooks
#define newCommandQueue q5HintCommandQueue
#define commandBuffer q5HintCommandBuffer
#define commandBufferWithUnretainedReferences q5HintCommandBufferUnretained
#define setComputePipelineState q5HintSetPipeline
#define computeCommandEncoder q5HintEncoder
#define computeCommandEncoderWithDispatchType q5HintEncoderWithDispatchType
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
#define qw3_metal_env_n_gpu_layers q5_hint_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled q5_hint_core_llama_split_enabled
#define qw3_metal_llama_split_start q5_hint_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int q5_hint_compile(void) {
    if (q5_hint_pipeline[2]) return 1;
    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;
    q5_hint_pipeline[0]=g_matvec_q5_k_pipeline;
    id<MTLFunction> fn=[g_library newFunctionWithName:@"qw3_matvec_q5_k_fast"];
    if (!fn) return 0;
    NSError *error=nil;
    for (int v=1;v<=2;v++) {
        MTLComputePipelineDescriptor *descriptor=[MTLComputePipelineDescriptor new];
        descriptor.computeFunction=fn;
        descriptor.maxTotalThreadsPerThreadgroup=64u;
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth=(v==2);
        q5_hint_pipeline[v]=[g_device newComputePipelineStateWithDescriptor:descriptor
            options:MTLPipelineOptionNone reflection:nil error:&error];
        if (!q5_hint_pipeline[v] || q5_hint_pipeline[v].threadExecutionWidth!=32u ||
            q5_hint_pipeline[v].maxTotalThreadsPerThreadgroup!=64u ||
            q5_hint_pipeline[v].staticThreadgroupMemoryLength!=q5_hint_pipeline[0].staticThreadgroupMemoryLength) {
            fprintf(stderr,"Q5 hint compile variant=%d: %s\n",v,error.localizedDescription.UTF8String);return 0;
        }
    }
    for (int v=0;v<=2;v++) fprintf(stderr,"q5_hint_metadata variant=%d width=%lu max_threads=%lu static_threadgroup_bytes=%lu dynamic_threadgroup_bytes=128 threads=64\n",v,(unsigned long)q5_hint_pipeline[v].threadExecutionWidth,(unsigned long)q5_hint_pipeline[v].maxTotalThreadsPerThreadgroup,(unsigned long)q5_hint_pipeline[v].staticThreadgroupMemoryLength);
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    g_matvec_q5_k_pipeline=q5_hint_pipeline[mode ? hint_variant : 0];
}
