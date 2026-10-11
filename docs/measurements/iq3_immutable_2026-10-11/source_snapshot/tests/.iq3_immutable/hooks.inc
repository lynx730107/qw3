/* Original shader and buffers; immutable descriptors for existing readonly weights/grid only. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> iq3_immutable_pipeline[3];
static uint64_t iq3_immutable_calls;
static int immutable_variant=1;
@protocol QW3IQ3ImmutableHooks
- (void)iq3ImmutableSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq3ImmutableEncoder;
- (id)iq3ImmutableEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq3ImmutableCommandBuffer;
- (id)iq3ImmutableCommandBufferUnretained;
- (id)iq3ImmutableCommandQueue;
@end
@interface NSObject (QW3IQ3ImmutableProbe) <QW3IQ3ImmutableHooks>
@end
@implementation NSObject (QW3IQ3ImmutableProbe)
- (id)iq3ImmutableCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq3ImmutableCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq3ImmutableCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq3ImmutableEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq3ImmutableEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq3ImmutableSetPipeline:(id<MTLComputePipelineState>)pipeline {
    if (pipeline==iq3_immutable_pipeline[1] || pipeline==iq3_immutable_pipeline[2]) iq3_immutable_calls++;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
@end
static id iq3_immutable_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq3_immutable_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq3_immutable_default_device
#define MTLCopyAllDevices iq3_immutable_all_devices
#define MTLDevice MTLDevice,QW3IQ3ImmutableHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ3ImmutableHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ3ImmutableHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ3ImmutableHooks
#define newCommandQueue iq3ImmutableCommandQueue
#define commandBuffer iq3ImmutableCommandBuffer
#define commandBufferWithUnretainedReferences iq3ImmutableCommandBufferUnretained
#define setComputePipelineState iq3ImmutableSetPipeline
#define computeCommandEncoder iq3ImmutableEncoder
#define computeCommandEncoderWithDispatchType iq3ImmutableEncoderWithDispatchType
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
#define qw3_metal_env_n_gpu_layers iq3_immutable_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq3_immutable_core_llama_split_enabled
#define qw3_metal_llama_split_start iq3_immutable_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int iq3_immutable_compile(void) {
    if (iq3_immutable_pipeline[2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq3_immutable_pipeline[0]=g_matvec_iq3_s_pipeline;
    id<MTLFunction> fn=[g_library newFunctionWithName:@"qw3_matvec_iq3_s_fast"];
    if (!fn) return 0;
    NSError *error=nil;
    for (int v=1;v<=2;v++) {
        MTLComputePipelineDescriptor *descriptor=[MTLComputePipelineDescriptor new];
        descriptor.computeFunction=fn;
        descriptor.buffers[1].mutability=MTLMutabilityImmutable;
        if (v==2) descriptor.buffers[4].mutability=MTLMutabilityImmutable;
        if (descriptor.maxTotalThreadsPerThreadgroup!=0 ||
            descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth ||
            descriptor.buffers[0].mutability!=MTLMutabilityDefault ||
            descriptor.buffers[2].mutability!=MTLMutabilityDefault ||
            descriptor.buffers[3].mutability!=MTLMutabilityDefault) return 0;
        iq3_immutable_pipeline[v]=[g_device newComputePipelineStateWithDescriptor:descriptor
            options:MTLPipelineOptionNone reflection:nil error:&error];
        if (!iq3_immutable_pipeline[v] || iq3_immutable_pipeline[v].maxTotalThreadsPerThreadgroup<64 ||
            iq3_immutable_pipeline[v].threadExecutionWidth!=32 ||
            iq3_immutable_pipeline[v].staticThreadgroupMemoryLength!=iq3_immutable_pipeline[0].staticThreadgroupMemoryLength) {
            fprintf(stderr,"IQ3 immutable compile variant=%d: %s\n",v,error.localizedDescription.UTF8String);return 0;
        }
    }
    for (int v=0;v<=2;v++) {
        id<MTLComputePipelineState> p=iq3_immutable_pipeline[v];
        fprintf(stderr,"iq3_immutable_metadata variant=%d thread_width=%lu max_threads=%lu static_threadgroup_bytes=%lu immutable_weight=%d immutable_grid=%d\n",
            v,(unsigned long)p.threadExecutionWidth,(unsigned long)p.maxTotalThreadsPerThreadgroup,
            (unsigned long)p.staticThreadgroupMemoryLength,v!=0,v==2);
    }
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    g_matvec_iq3_s_pipeline=iq3_immutable_pipeline[mode ? immutable_variant : 0];
}
