/* Packed input loads; native weights, grid, FP32 arithmetic and data buffers. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> iq3_input_pipeline[3];
static uint64_t iq3_input_calls;
static int input_variant=1;
@protocol QW3IQ3InputHooks
- (void)iq3InputSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq3InputEncoder;
- (id)iq3InputEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq3InputCommandBuffer;
- (id)iq3InputCommandBufferUnretained;
- (id)iq3InputCommandQueue;
@end
@interface NSObject (QW3IQ3InputProbe) <QW3IQ3InputHooks>
@end
@implementation NSObject (QW3IQ3InputProbe)
- (id)iq3InputCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq3InputCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq3InputCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq3InputEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq3InputEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq3InputSetPipeline:(id<MTLComputePipelineState>)pipeline {
    if (pipeline==iq3_input_pipeline[1] || pipeline==iq3_input_pipeline[2]) iq3_input_calls++;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
@end
static id iq3_input_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq3_input_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq3_input_default_device
#define MTLCopyAllDevices iq3_input_all_devices
#define MTLDevice MTLDevice,QW3IQ3InputHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ3InputHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ3InputHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ3InputHooks
#define newCommandQueue iq3InputCommandQueue
#define commandBuffer iq3InputCommandBuffer
#define commandBufferWithUnretainedReferences iq3InputCommandBufferUnretained
#define setComputePipelineState iq3InputSetPipeline
#define computeCommandEncoder iq3InputEncoder
#define computeCommandEncoderWithDispatchType iq3InputEncoderWithDispatchType
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
#define qw3_metal_env_n_gpu_layers iq3_input_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq3_input_core_llama_split_enabled
#define qw3_metal_llama_split_start iq3_input_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int iq3_input_compile(void) {
    if (iq3_input_pipeline[2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq3_input_pipeline[0]=g_matvec_iq3_s_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.iq3_input/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) {
        NSString *name=[NSString stringWithFormat:@"private_iq3_input_fast%d",v];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        iq3_input_pipeline[v]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!iq3_input_pipeline[v] || iq3_input_pipeline[v].maxTotalThreadsPerThreadgroup<64 || iq3_input_pipeline[v].threadExecutionWidth!=32 || iq3_input_pipeline[v].staticThreadgroupMemoryLength!=iq3_input_pipeline[0].staticThreadgroupMemoryLength) {
            fprintf(stderr,"IQ3 input compile variant=%d: %s\n",v,error.localizedDescription.UTF8String);return 0;
        }
    }
    for (int v=0;v<=2;v++) fprintf(stderr,"iq3_input_metadata variant=%d width=%lu max_threads=%lu static_threadgroup_bytes=%lu dynamic_threadgroup_bytes=256 threads=64\n",v,(unsigned long)iq3_input_pipeline[v].threadExecutionWidth,(unsigned long)iq3_input_pipeline[v].maxTotalThreadsPerThreadgroup,(unsigned long)iq3_input_pipeline[v].staticThreadgroupMemoryLength);
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    g_matvec_iq3_s_pipeline=iq3_input_pipeline[mode ? input_variant : 0];
}
