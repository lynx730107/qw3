/* Observe the actual public pipeline; never replace a kernel or change a dispatch. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> gqa_groups_pipeline[3];
static uint64_t gqa_groups_calls,gqa_groups_fallbacks;
static int groups_variant=2;

@protocol QW3GQAGroupsHooks
- (void)gqaGroupsDispatchThreadgroups:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads;
- (void)gqaGroupsSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)gqaGroupsSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)gqaGroupsEncoder;
- (id)gqaGroupsEncoderWithDispatchType:(MTLDispatchType)type;
- (id)gqaGroupsCommandBuffer;
- (id)gqaGroupsCommandBufferUnretained;
- (id)gqaGroupsCommandQueue;
@end
@interface NSObject (QW3GQAGroupsProbe) <QW3GQAGroupsHooks>
@end
@implementation NSObject (QW3GQAGroupsProbe)
- (id)gqaGroupsCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)gqaGroupsCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)gqaGroupsCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)gqaGroupsEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)gqaGroupsEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)gqaGroupsSetPipeline:(id<MTLComputePipelineState>)pipeline {
    if (pipeline==gqa_groups_pipeline[2]) gqa_groups_calls++;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
- (void)gqaGroupsSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    [(id<MTLComputeCommandEncoder>)self setBytes:bytes length:length atIndex:index];
}
- (void)gqaGroupsDispatchThreadgroups:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads {
    [(id<MTLComputeCommandEncoder>)self dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

@end
static id gqa_groups_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id gqa_groups_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice gqa_groups_default_device
#define MTLCopyAllDevices gqa_groups_all_devices
#define MTLDevice MTLDevice,QW3GQAGroupsHooks
#define MTLCommandQueue MTLCommandQueue,QW3GQAGroupsHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3GQAGroupsHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3GQAGroupsHooks
#define newCommandQueue gqaGroupsCommandQueue
#define commandBuffer gqaGroupsCommandBuffer
#define commandBufferWithUnretainedReferences gqaGroupsCommandBufferUnretained
#define dispatchThreadgroups gqaGroupsDispatchThreadgroups
#define setBytes gqaGroupsSetBytes
#define setComputePipelineState gqaGroupsSetPipeline
#define computeCommandEncoder gqaGroupsEncoder
#define computeCommandEncoderWithDispatchType gqaGroupsEncoderWithDispatchType
#include "../../qw3_metal.m"
#undef computeCommandEncoderWithDispatchType
#undef computeCommandEncoder
#undef dispatchThreadgroups
#undef setBytes
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
#define qw3_metal_env_n_gpu_layers gqa_groups_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled gqa_groups_core_llama_split_enabled
#define qw3_metal_llama_split_start gqa_groups_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int gqa_groups_compile(void) {
    if (gqa_groups_pipeline[2]) return 1;
    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;
    gqa_groups_pipeline[0]=g_gqa_attend_n_inner_pipeline;
    gqa_groups_pipeline[2]=g_gqa_attend_n_inner_heads3_pipeline;
    if (!gqa_groups_pipeline[2]) return 0;
    fprintf(stderr,"public_heads3_metadata width=%lu max_threads=%lu static_bytes=%lu dynamic_bytes=256\n",
      (unsigned long)gqa_groups_pipeline[2].threadExecutionWidth,
      (unsigned long)gqa_groups_pipeline[2].maxTotalThreadsPerThreadgroup,
      (unsigned long)gqa_groups_pipeline[2].staticThreadgroupMemoryLength);
    return 1;
}
static void scratch_select(int mode) {
    g_gqa_attend_n_inner_heads3_pipeline=mode ? gqa_groups_pipeline[2] : nil;
}
static __attribute__((unused)) void gqa_groups_print_counts(void) {
    fprintf(stderr,"public_heads3_dispatch_counts calls=%llu fallbacks=%llu\n",
       (unsigned long long)gqa_groups_calls,(unsigned long long)gqa_groups_fallbacks);
}
