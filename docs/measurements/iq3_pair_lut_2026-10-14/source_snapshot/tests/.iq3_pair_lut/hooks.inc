/* Exact half4 or expanded-byte paired grid; native FP32 dot expressions. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> iq3_pair_lut_pipeline[3];
static uint64_t iq3_pair_lut_calls;
static int pair_lut_variant=1;
@protocol QW3IQ3PairLutHooks
- (void)iq3PairLutSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq3PairLutEncoder;
- (id)iq3PairLutEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq3PairLutCommandBuffer;
- (id)iq3PairLutCommandBufferUnretained;
- (id)iq3PairLutCommandQueue;
@end
@interface NSObject (QW3IQ3PairLutProbe) <QW3IQ3PairLutHooks>
@end
@implementation NSObject (QW3IQ3PairLutProbe)
- (id)iq3PairLutCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq3PairLutCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq3PairLutCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq3PairLutEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq3PairLutEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq3PairLutSetPipeline:(id<MTLComputePipelineState>)pipeline {
    if (pipeline==iq3_pair_lut_pipeline[1] || pipeline==iq3_pair_lut_pipeline[2]) iq3_pair_lut_calls++;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
@end
static id iq3_pair_lut_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq3_pair_lut_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq3_pair_lut_default_device
#define MTLCopyAllDevices iq3_pair_lut_all_devices
#define MTLDevice MTLDevice,QW3IQ3PairLutHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ3PairLutHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ3PairLutHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ3PairLutHooks
#define newCommandQueue iq3PairLutCommandQueue
#define commandBuffer iq3PairLutCommandBuffer
#define commandBufferWithUnretainedReferences iq3PairLutCommandBufferUnretained
#define setComputePipelineState iq3PairLutSetPipeline
#define computeCommandEncoder iq3PairLutEncoder
#define computeCommandEncoderWithDispatchType iq3PairLutEncoderWithDispatchType
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
#define qw3_metal_env_n_gpu_layers iq3_pair_lut_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq3_pair_lut_core_llama_split_enabled
#define qw3_metal_llama_split_start iq3_pair_lut_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int iq3_pair_lut_compile(void) {
    if (iq3_pair_lut_pipeline[2]) return 1;
    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;
    iq3_pair_lut_pipeline[0]=g_matvec_iq3_s_pair_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.iq3_pair_lut/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) {
        NSString *name=[NSString stringWithFormat:@"private_iq3_pair_lut_fast%d",v];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        iq3_pair_lut_pipeline[v]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!iq3_pair_lut_pipeline[v] || iq3_pair_lut_pipeline[v].maxTotalThreadsPerThreadgroup<64 || iq3_pair_lut_pipeline[v].threadExecutionWidth!=32 || iq3_pair_lut_pipeline[v].staticThreadgroupMemoryLength!=(v==1 ? 4096u : 2048u)) {
            fprintf(stderr,"IQ3 pair lut compile variant=%d: %s\n",v,error.localizedDescription.UTF8String);return 0;
        }
    }
    for (int v=0;v<=2;v++) fprintf(stderr,"iq3_pair_lut_metadata variant=%d width=%lu max_threads=%lu static_threadgroup_bytes=%lu dynamic_threadgroup_bytes=128 threads=64\n",v,(unsigned long)iq3_pair_lut_pipeline[v].threadExecutionWidth,(unsigned long)iq3_pair_lut_pipeline[v].maxTotalThreadsPerThreadgroup,(unsigned long)iq3_pair_lut_pipeline[v].staticThreadgroupMemoryLength);
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    g_matvec_iq3_s_pair_pipeline=iq3_pair_lut_pipeline[mode ? pair_lut_variant : 0];
}
