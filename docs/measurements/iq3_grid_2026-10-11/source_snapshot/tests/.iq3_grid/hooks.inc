/* Direct reads of the existing packed grid; no extra weight/session buffers. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> iq3_grid_pipeline[3];
static uint64_t iq3_grid_calls;
static int grid_variant=1;
@protocol QW3IQ3GridHooks
- (void)iq3GridSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq3GridEncoder;
- (id)iq3GridEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq3GridCommandBuffer;
- (id)iq3GridCommandBufferUnretained;
- (id)iq3GridCommandQueue;
@end
@interface NSObject (QW3IQ3GridProbe) <QW3IQ3GridHooks>
@end
@implementation NSObject (QW3IQ3GridProbe)
- (id)iq3GridCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq3GridCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq3GridCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq3GridEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq3GridEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq3GridSetPipeline:(id<MTLComputePipelineState>)pipeline {
    if (pipeline==iq3_grid_pipeline[1] || pipeline==iq3_grid_pipeline[2]) iq3_grid_calls++;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
@end
static id iq3_grid_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq3_grid_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq3_grid_default_device
#define MTLCopyAllDevices iq3_grid_all_devices
#define MTLDevice MTLDevice,QW3IQ3GridHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ3GridHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ3GridHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ3GridHooks
#define newCommandQueue iq3GridCommandQueue
#define commandBuffer iq3GridCommandBuffer
#define commandBufferWithUnretainedReferences iq3GridCommandBufferUnretained
#define setComputePipelineState iq3GridSetPipeline
#define computeCommandEncoder iq3GridEncoder
#define computeCommandEncoderWithDispatchType iq3GridEncoderWithDispatchType
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
#define qw3_metal_env_n_gpu_layers iq3_grid_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq3_grid_core_llama_split_enabled
#define qw3_metal_llama_split_start iq3_grid_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int iq3_grid_compile(void) {
    if (iq3_grid_pipeline[2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq3_grid_pipeline[0]=g_matvec_iq3_s_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    NSString *source=[NSString stringWithContentsOfFile:@"tests/.iq3_grid/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) {
        NSString *name=[NSString stringWithFormat:@"private_iq3_grid_fast%d",v];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        iq3_grid_pipeline[v]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!iq3_grid_pipeline[v] || iq3_grid_pipeline[v].maxTotalThreadsPerThreadgroup<64) {
            fprintf(stderr,"IQ3 grid compile variant=%d: %s\n",v,error.localizedDescription.UTF8String);return 0;
        }
    }
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    g_matvec_iq3_s_pipeline=iq3_grid_pipeline[mode ? grid_variant : 0];
}
