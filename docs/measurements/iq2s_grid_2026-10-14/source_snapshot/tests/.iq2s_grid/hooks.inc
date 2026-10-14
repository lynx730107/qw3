/* IQ2_S sg8 packed/device grid; original FP32 vector dot and data buffers. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> iq2s_grid_pipeline[3];
static uint64_t iq2s_grid_calls;
static int grid_variant=1;
@protocol QW3IQ2SGridHooks
- (void)iq2sGridSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq2sGridEncoder;
- (id)iq2sGridEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq2sGridCommandBuffer;
- (id)iq2sGridCommandBufferUnretained;
- (id)iq2sGridCommandQueue;
@end
@interface NSObject (QW3IQ2SGridProbe) <QW3IQ2SGridHooks>
@end
@implementation NSObject (QW3IQ2SGridProbe)
- (id)iq2sGridCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq2sGridCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq2sGridCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq2sGridEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq2sGridEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq2sGridSetPipeline:(id<MTLComputePipelineState>)pipeline {
    if (pipeline==iq2s_grid_pipeline[1] || pipeline==iq2s_grid_pipeline[2]) iq2s_grid_calls++;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
@end
static id iq2s_grid_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq2s_grid_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq2s_grid_default_device
#define MTLCopyAllDevices iq2s_grid_all_devices
#define MTLDevice MTLDevice,QW3IQ2SGridHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ2SGridHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ2SGridHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ2SGridHooks
#define newCommandQueue iq2sGridCommandQueue
#define commandBuffer iq2sGridCommandBuffer
#define commandBufferWithUnretainedReferences iq2sGridCommandBufferUnretained
#define setComputePipelineState iq2sGridSetPipeline
#define computeCommandEncoder iq2sGridEncoder
#define computeCommandEncoderWithDispatchType iq2sGridEncoderWithDispatchType
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
#define qw3_metal_env_n_gpu_layers iq2s_grid_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq2s_grid_core_llama_split_enabled
#define qw3_metal_llama_split_start iq2s_grid_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int iq2s_grid_compile(void) {
    if (iq2s_grid_pipeline[2]) return 1;
    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;
    iq2s_grid_pipeline[0]=g_matvec_iq2_s_sg8_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.iq2s_grid/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) {
        NSString *name=[NSString stringWithFormat:@"private_iq2s_grid_fast%d",v];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        iq2s_grid_pipeline[v]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!iq2s_grid_pipeline[v] || iq2s_grid_pipeline[v].maxTotalThreadsPerThreadgroup<256 || iq2s_grid_pipeline[v].threadExecutionWidth!=32 || iq2s_grid_pipeline[v].staticThreadgroupMemoryLength!=(v==1 ? 2048u : 0u)) {
            fprintf(stderr,"IQ2S grid compile variant=%d: %s\n",v,error.localizedDescription.UTF8String);return 0;
        }
    }
    for (int v=0;v<=2;v++) fprintf(stderr,"iq2s_grid_metadata variant=%d width=%lu max_threads=%lu static_threadgroup_bytes=%lu dynamic_threadgroup_bytes=128 threads=256\n",v,(unsigned long)iq2s_grid_pipeline[v].threadExecutionWidth,(unsigned long)iq2s_grid_pipeline[v].maxTotalThreadsPerThreadgroup,(unsigned long)iq2s_grid_pipeline[v].staticThreadgroupMemoryLength);
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    g_matvec_iq2_s_sg8_pipeline=iq2s_grid_pipeline[mode ? grid_variant : 0];
}
