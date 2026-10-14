/* IQ3_S predecoded nibbles; same 1024-byte local grid and original FP32 scalar order. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> iq3_nibble_pipeline[2];
static uint64_t iq3_nibble_calls;
static int nibble_variant=1;
@protocol QW3IQ3NibbleHooks
- (void)iq3NibbleSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq3NibbleEncoder;
- (id)iq3NibbleEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq3NibbleCommandBuffer;
- (id)iq3NibbleCommandBufferUnretained;
- (id)iq3NibbleCommandQueue;
@end
@interface NSObject (QW3IQ3NibbleProbe) <QW3IQ3NibbleHooks>
@end
@implementation NSObject (QW3IQ3NibbleProbe)
- (id)iq3NibbleCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq3NibbleCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq3NibbleCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq3NibbleEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq3NibbleEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq3NibbleSetPipeline:(id<MTLComputePipelineState>)pipeline {
    if (pipeline==iq3_nibble_pipeline[1]) iq3_nibble_calls++;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
@end
static id iq3_nibble_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq3_nibble_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq3_nibble_default_device
#define MTLCopyAllDevices iq3_nibble_all_devices
#define MTLDevice MTLDevice,QW3IQ3NibbleHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ3NibbleHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ3NibbleHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ3NibbleHooks
#define newCommandQueue iq3NibbleCommandQueue
#define commandBuffer iq3NibbleCommandBuffer
#define commandBufferWithUnretainedReferences iq3NibbleCommandBufferUnretained
#define setComputePipelineState iq3NibbleSetPipeline
#define computeCommandEncoder iq3NibbleEncoder
#define computeCommandEncoderWithDispatchType iq3NibbleEncoderWithDispatchType
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
#define qw3_metal_env_n_gpu_layers iq3_nibble_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq3_nibble_core_llama_split_enabled
#define qw3_metal_llama_split_start iq3_nibble_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int iq3_nibble_compile(void) {
    if (iq3_nibble_pipeline[1]) return 1;
    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;
    iq3_nibble_pipeline[0]=g_matvec_iq3_s_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.iq3_nibble/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=1;v++) {
        NSString *name=[NSString stringWithFormat:@"private_iq3_nibble_fast%d",v];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        iq3_nibble_pipeline[v]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!iq3_nibble_pipeline[v] || iq3_nibble_pipeline[v].maxTotalThreadsPerThreadgroup<64 || iq3_nibble_pipeline[v].threadExecutionWidth!=32 || iq3_nibble_pipeline[v].staticThreadgroupMemoryLength!=1024u) {
            fprintf(stderr,"IQ3 nibble compile variant=%d: %s\n",v,error.localizedDescription.UTF8String);return 0;
        }
    }
    for (int v=0;v<=1;v++) fprintf(stderr,"iq3_nibble_metadata variant=%d width=%lu max_threads=%lu static_threadgroup_bytes=%lu dynamic_threadgroup_bytes=128 threads=64\n",v,(unsigned long)iq3_nibble_pipeline[v].threadExecutionWidth,(unsigned long)iq3_nibble_pipeline[v].maxTotalThreadsPerThreadgroup,(unsigned long)iq3_nibble_pipeline[v].staticThreadgroupMemoryLength);
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    g_matvec_iq3_s_pipeline=iq3_nibble_pipeline[mode ? nibble_variant : 0];
}
