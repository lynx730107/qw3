/* Fold exact integer signs in a private IQ3_S shader; numerical gates remain strict. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> iq3_signed_pipeline[3];
static uint64_t iq3_signed_calls;
static int signed_variant=1;
@protocol QW3IQ3SignedHooks
- (void)iq3SignedSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq3SignedEncoder;
- (id)iq3SignedEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq3SignedCommandBuffer;
- (id)iq3SignedCommandBufferUnretained;
- (id)iq3SignedCommandQueue;
@end
@interface NSObject (QW3IQ3SignedProbe) <QW3IQ3SignedHooks>
@end
@implementation NSObject (QW3IQ3SignedProbe)
- (id)iq3SignedCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq3SignedCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq3SignedCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq3SignedEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq3SignedEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq3SignedSetPipeline:(id<MTLComputePipelineState>)pipeline {
    if (pipeline==iq3_signed_pipeline[1] || pipeline==iq3_signed_pipeline[2]) iq3_signed_calls++;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
@end
static id iq3_signed_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq3_signed_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq3_signed_default_device
#define MTLCopyAllDevices iq3_signed_all_devices
#define MTLDevice MTLDevice,QW3IQ3SignedHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ3SignedHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ3SignedHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ3SignedHooks
#define newCommandQueue iq3SignedCommandQueue
#define commandBuffer iq3SignedCommandBuffer
#define commandBufferWithUnretainedReferences iq3SignedCommandBufferUnretained
#define setComputePipelineState iq3SignedSetPipeline
#define computeCommandEncoder iq3SignedEncoder
#define computeCommandEncoderWithDispatchType iq3SignedEncoderWithDispatchType
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
#define qw3_metal_env_n_gpu_layers iq3_signed_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq3_signed_core_llama_split_enabled
#define qw3_metal_llama_split_start iq3_signed_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int iq3_signed_compile(void) {
    if (iq3_signed_pipeline[2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq3_signed_pipeline[0]=g_matvec_iq3_s_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    NSString *source=[NSString stringWithContentsOfFile:@"tests/.iq3_signed/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) {
        NSString *name=[NSString stringWithFormat:@"private_iq3_signed_fast%d",v];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        iq3_signed_pipeline[v]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!iq3_signed_pipeline[v] || iq3_signed_pipeline[v].maxTotalThreadsPerThreadgroup<64) {
            fprintf(stderr,"IQ3 signed compile variant=%d: %s\n",v,error.localizedDescription.UTF8String);return 0;
        }
    }
    return 1;
}
static void scratch_select(int mode) {
    setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1);
    g_matvec_iq3_s_pipeline=iq3_signed_pipeline[mode ? signed_variant : 0];
}
