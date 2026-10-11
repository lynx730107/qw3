/* Single IQ3_S row geometry only; native weights, grid, lane sums and buffers. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> iq3_rows_pipeline[3];
static uint64_t iq3_rows_calls;
static int rows_variant=1;
@protocol QW3IQ3RowsHooks
- (void)iq3RowsSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq3RowsEncoder;
- (id)iq3RowsEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq3RowsCommandBuffer;
- (id)iq3RowsCommandBufferUnretained;
- (id)iq3RowsCommandQueue;
@end
@interface NSObject (QW3IQ3RowsProbe) <QW3IQ3RowsHooks>
@end
@implementation NSObject (QW3IQ3RowsProbe)
- (id)iq3RowsCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq3RowsCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq3RowsCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq3RowsEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq3RowsEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq3RowsSetPipeline:(id<MTLComputePipelineState>)pipeline {
    if (pipeline==iq3_rows_pipeline[1] || pipeline==iq3_rows_pipeline[2]) iq3_rows_calls++;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
@end
static id iq3_rows_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq3_rows_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq3_rows_default_device
#define MTLCopyAllDevices iq3_rows_all_devices
#define MTLDevice MTLDevice,QW3IQ3RowsHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ3RowsHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ3RowsHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ3RowsHooks
#define newCommandQueue iq3RowsCommandQueue
#define commandBuffer iq3RowsCommandBuffer
#define commandBufferWithUnretainedReferences iq3RowsCommandBufferUnretained
#define setComputePipelineState iq3RowsSetPipeline
#define computeCommandEncoder iq3RowsEncoder
#define computeCommandEncoderWithDispatchType iq3RowsEncoderWithDispatchType
static NSUInteger iq3_rows_per_simd=4u;
static NSUInteger iq3_rows_groups(NSUInteger n_out) {
    NSUInteger rows=2u*iq3_rows_per_simd;
    return (n_out+rows-1u)/rows;
}
#include "core.inc"
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
#define qw3_metal_env_n_gpu_layers iq3_rows_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq3_rows_core_llama_split_enabled
#define qw3_metal_llama_split_start iq3_rows_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int iq3_rows_compile(void) {
    if (iq3_rows_pipeline[2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq3_rows_pipeline[0]=g_matvec_iq3_s_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.iq3_rows/kernels.metal"
        encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) {
        NSString *name=[NSString stringWithFormat:@"private_iq3_rows_fast%d",v];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        iq3_rows_pipeline[v]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!iq3_rows_pipeline[v] || iq3_rows_pipeline[v].maxTotalThreadsPerThreadgroup<64 ||
            iq3_rows_pipeline[v].threadExecutionWidth!=32 ||
            iq3_rows_pipeline[v].staticThreadgroupMemoryLength!=iq3_rows_pipeline[0].staticThreadgroupMemoryLength) {
            fprintf(stderr,"IQ3 rows compile variant=%d: %s\n",v,error.localizedDescription.UTF8String);return 0;
        }
    }
    for (int v=0;v<=2;v++) {
        id<MTLComputePipelineState> p=iq3_rows_pipeline[v];
        fprintf(stderr,"iq3_rows_metadata variant=%d thread_width=%lu max_threads=%lu static_threadgroup_bytes=%lu rows_per_simd=%d threads=64\n",
            v,(unsigned long)p.threadExecutionWidth,(unsigned long)p.maxTotalThreadsPerThreadgroup,
            (unsigned long)p.staticThreadgroupMemoryLength,v==1 ? 2 : v==2 ? 8 : 4);
    }
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    g_matvec_iq3_s_pipeline=iq3_rows_pipeline[mode ? rows_variant : 0];
    iq3_rows_per_simd=mode ? (rows_variant==1 ? 2u : 8u) : 4u;
}
