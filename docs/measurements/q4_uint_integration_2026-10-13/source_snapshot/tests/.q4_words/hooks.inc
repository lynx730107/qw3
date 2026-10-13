/* Aligned Q4_K payload words; native geometry, scalar FP32 arithmetic and buffers. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> q4_words_pipeline[3];
static id<MTLComputePipelineState> g_matvec_q4_k_words_pipeline;
static uint64_t q4_words_calls,q4_words_fallbacks;
static int q4_words_mode,words_variant=1;

@protocol QW3Q4WordsHooks
- (void)q4WordsSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)q4WordsEncoder;
- (id)q4WordsEncoderWithDispatchType:(MTLDispatchType)type;
- (id)q4WordsCommandBuffer;
- (id)q4WordsCommandBufferUnretained;
- (id)q4WordsCommandQueue;
@end
@interface NSObject (QW3Q4WordsProbe) <QW3Q4WordsHooks>
@end
@implementation NSObject (QW3Q4WordsProbe)
- (id)q4WordsCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)q4WordsCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)q4WordsCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)q4WordsEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)q4WordsEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)q4WordsSetPipeline:(id<MTLComputePipelineState>)pipeline {
    if (q4_words_mode) {
        if (pipeline==g_matvec_q4_k_words_pipeline) q4_words_calls++;
        else if (pipeline==q4_words_pipeline[0]) q4_words_fallbacks++;
    }
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}

@end
static id q4_words_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id q4_words_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice q4_words_default_device
#define MTLCopyAllDevices q4_words_all_devices
#define MTLDevice MTLDevice,QW3Q4WordsHooks
#define MTLCommandQueue MTLCommandQueue,QW3Q4WordsHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3Q4WordsHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3Q4WordsHooks
#define newCommandQueue q4WordsCommandQueue
#define commandBuffer q4WordsCommandBuffer
#define commandBufferWithUnretainedReferences q4WordsCommandBufferUnretained
#define setComputePipelineState q4WordsSetPipeline
#define computeCommandEncoder q4WordsEncoder
#define computeCommandEncoderWithDispatchType q4WordsEncoderWithDispatchType
#include "../../qw3_metal.m"
#undef computeCommandEncoderWithDispatchType
#undef computeCommandEncoder
#undef dispatchThreadgroups
#undef setBuffer
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
#define qw3_metal_env_n_gpu_layers q4_words_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled q4_words_core_llama_split_enabled
#define qw3_metal_llama_split_start q4_words_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int q4_words_compile(void) {
    if (!qw3_metal_init() || !qw3_metal_compile_kernels() || !g_matvec_q4_k_words_pipeline) return 0;
    q4_words_pipeline[0]=g_matvec_q4_k_pipeline;
    q4_words_pipeline[2]=g_matvec_q4_k_words_pipeline;
    return 1;
}
static void scratch_select(int mode) {g_dense_q4_words=mode!=0;q4_words_mode=mode!=0;}
static void q4_words_print_counts(void) {
    fprintf(stderr,"integrated_q4_words calls=%llu fallbacks=%llu\n",(unsigned long long)q4_words_calls,(unsigned long long)q4_words_fallbacks);
}
