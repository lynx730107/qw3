/* Aligned integer loads from original IQ3_S weights; native grid and FP32 math. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> iq3_words_pipeline[3];
static uint64_t iq3_words_calls;
static int words_variant=1;
@protocol QW3IQ3WordsHooks
- (void)iq3WordsSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq3WordsEncoder;
- (id)iq3WordsEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq3WordsCommandBuffer;
- (id)iq3WordsCommandBufferUnretained;
- (id)iq3WordsCommandQueue;
@end
@interface NSObject (QW3IQ3WordsProbe) <QW3IQ3WordsHooks>
@end
@implementation NSObject (QW3IQ3WordsProbe)
- (id)iq3WordsCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq3WordsCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq3WordsCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq3WordsEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq3WordsEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq3WordsSetPipeline:(id<MTLComputePipelineState>)pipeline {
    if (pipeline==iq3_words_pipeline[1] || pipeline==iq3_words_pipeline[2]) iq3_words_calls++;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
@end
static id iq3_words_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq3_words_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq3_words_default_device
#define MTLCopyAllDevices iq3_words_all_devices
#define MTLDevice MTLDevice,QW3IQ3WordsHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ3WordsHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ3WordsHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ3WordsHooks
#define newCommandQueue iq3WordsCommandQueue
#define commandBuffer iq3WordsCommandBuffer
#define commandBufferWithUnretainedReferences iq3WordsCommandBufferUnretained
#define setComputePipelineState iq3WordsSetPipeline
#define computeCommandEncoder iq3WordsEncoder
#define computeCommandEncoderWithDispatchType iq3WordsEncoderWithDispatchType
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
#define qw3_metal_env_n_gpu_layers iq3_words_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq3_words_core_llama_split_enabled
#define qw3_metal_llama_split_start iq3_words_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int iq3_words_compile(void) {
    if (iq3_words_pipeline[2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq3_words_pipeline[0]=g_matvec_iq3_s_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    NSString *source=[NSString stringWithContentsOfFile:@"tests/.iq3_words/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) {
        NSString *name=[NSString stringWithFormat:@"private_iq3_words_fast%d",v];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        iq3_words_pipeline[v]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!iq3_words_pipeline[v] || iq3_words_pipeline[v].maxTotalThreadsPerThreadgroup<64) {
            fprintf(stderr,"IQ3 words compile variant=%d: %s\n",v,error.localizedDescription.UTF8String);return 0;
        }
    }
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    g_matvec_iq3_s_pipeline=iq3_words_pipeline[mode ? words_variant : 0];
}
