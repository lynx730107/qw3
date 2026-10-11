/* Original shader and buffers; explicit compute-pipeline threadgroup hints only. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> iq3_hint_pipeline[3];
static uint64_t iq3_hint_calls;
static int hint_variant=1;
@protocol QW3IQ3HintHooks
- (void)iq3HintSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq3HintEncoder;
- (id)iq3HintEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq3HintCommandBuffer;
- (id)iq3HintCommandBufferUnretained;
- (id)iq3HintCommandQueue;
@end
@interface NSObject (QW3IQ3HintProbe) <QW3IQ3HintHooks>
@end
@implementation NSObject (QW3IQ3HintProbe)
- (id)iq3HintCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq3HintCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq3HintCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq3HintEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq3HintEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq3HintSetPipeline:(id<MTLComputePipelineState>)pipeline {
    if (pipeline==iq3_hint_pipeline[1] || pipeline==iq3_hint_pipeline[2]) iq3_hint_calls++;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
@end
static id iq3_hint_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq3_hint_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq3_hint_default_device
#define MTLCopyAllDevices iq3_hint_all_devices
#define MTLDevice MTLDevice,QW3IQ3HintHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ3HintHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ3HintHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ3HintHooks
#define newCommandQueue iq3HintCommandQueue
#define commandBuffer iq3HintCommandBuffer
#define commandBufferWithUnretainedReferences iq3HintCommandBufferUnretained
#define setComputePipelineState iq3HintSetPipeline
#define computeCommandEncoder iq3HintEncoder
#define computeCommandEncoderWithDispatchType iq3HintEncoderWithDispatchType
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
#define qw3_metal_env_n_gpu_layers iq3_hint_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq3_hint_core_llama_split_enabled
#define qw3_metal_llama_split_start iq3_hint_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int iq3_hint_compile(void) {
    if (iq3_hint_pipeline[2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq3_hint_pipeline[0]=g_matvec_iq3_s_pipeline;
    id<MTLFunction> fn=[g_library newFunctionWithName:@"qw3_matvec_iq3_s_fast"];
    if (!fn) return 0;
    NSError *error=nil;
    for (int v=1;v<=2;v++) {
        MTLComputePipelineDescriptor *descriptor=[MTLComputePipelineDescriptor new];
        descriptor.computeFunction=fn;
        descriptor.maxTotalThreadsPerThreadgroup=64;
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth=(v==2);
        iq3_hint_pipeline[v]=[g_device newComputePipelineStateWithDescriptor:descriptor
            options:MTLPipelineOptionNone reflection:nil error:&error];
        if (!iq3_hint_pipeline[v] || iq3_hint_pipeline[v].maxTotalThreadsPerThreadgroup!=64 ||
            iq3_hint_pipeline[v].threadExecutionWidth!=32 ||
            iq3_hint_pipeline[v].staticThreadgroupMemoryLength!=iq3_hint_pipeline[0].staticThreadgroupMemoryLength) {
            fprintf(stderr,"IQ3 hint compile variant=%d: %s\n",v,error.localizedDescription.UTF8String);return 0;
        }
    }
    for (int v=0;v<=2;v++) {
        id<MTLComputePipelineState> p=iq3_hint_pipeline[v];
        fprintf(stderr,"iq3_hint_metadata variant=%d thread_width=%lu max_threads=%lu static_threadgroup_bytes=%lu\n",
            v,(unsigned long)p.threadExecutionWidth,(unsigned long)p.maxTotalThreadsPerThreadgroup,
            (unsigned long)p.staticThreadgroupMemoryLength);
    }
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    g_matvec_iq3_s_pipeline=iq3_hint_pipeline[mode ? hint_variant : 0];
}
