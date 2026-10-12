/* Three IQ4_XS pipelines: Per-SIMD LUT and optional unroll4; native arithmetic and data buffers. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> iq4_simd_lut_pipeline[3][3];
static uint64_t iq4_simd_lut_calls,iq4_simd_lut_calls_by_kind[3];
static int lut_variant=1;
@protocol QW3IQ4SimdLutHooks
- (void)iq4SimdLutSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq4SimdLutEncoder;
- (id)iq4SimdLutEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq4SimdLutCommandBuffer;
- (id)iq4SimdLutCommandBufferUnretained;
- (id)iq4SimdLutCommandQueue;
@end
@interface NSObject (QW3IQ4SimdLutProbe) <QW3IQ4SimdLutHooks>
@end
@implementation NSObject (QW3IQ4SimdLutProbe)
- (id)iq4SimdLutCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq4SimdLutCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq4SimdLutCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq4SimdLutEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq4SimdLutEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq4SimdLutSetPipeline:(id<MTLComputePipelineState>)pipeline {
    for (int v=1;v<=2;v++) for (int kind=0;kind<3;kind++)
        if (pipeline==iq4_simd_lut_pipeline[v][kind]) {iq4_simd_lut_calls++;iq4_simd_lut_calls_by_kind[kind]++;}
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
@end
static id iq4_simd_lut_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq4_simd_lut_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq4_simd_lut_default_device
#define MTLCopyAllDevices iq4_simd_lut_all_devices
#define MTLDevice MTLDevice,QW3IQ4SimdLutHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ4SimdLutHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ4SimdLutHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ4SimdLutHooks
#define newCommandQueue iq4SimdLutCommandQueue
#define commandBuffer iq4SimdLutCommandBuffer
#define commandBufferWithUnretainedReferences iq4SimdLutCommandBufferUnretained
#define setComputePipelineState iq4SimdLutSetPipeline
#define computeCommandEncoder iq4SimdLutEncoder
#define computeCommandEncoderWithDispatchType iq4SimdLutEncoderWithDispatchType
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
#define qw3_metal_env_n_gpu_layers iq4_simd_lut_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq4_simd_lut_core_llama_split_enabled
#define qw3_metal_llama_split_start iq4_simd_lut_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int iq4_simd_lut_compile(void) {
    if (iq4_simd_lut_pipeline[2][2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq4_simd_lut_pipeline[0][0]=g_matvec_iq4_xs_pipeline;
    iq4_simd_lut_pipeline[0][1]=g_matvec_iq4_xs_pair_pipeline;
    iq4_simd_lut_pipeline[0][2]=g_matvec_iq4_xs_add_x0_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.iq4_simd_lut/kernels.metal"
        encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) for (int kind=0;kind<3;kind++) {
        NSString *name=[NSString stringWithFormat:@"private_iq4_simd_lut_fast%d_%d",v,kind];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        id<MTLComputePipelineState> p=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!p || p.maxTotalThreadsPerThreadgroup<64u || p.threadExecutionWidth!=32 ||
            p.staticThreadgroupMemoryLength!=iq4_simd_lut_pipeline[0][kind].staticThreadgroupMemoryLength) {
            fprintf(stderr,"IQ4 SIMD LUT compile variant=%d kind=%d: %s\n",v,kind,error.localizedDescription.UTF8String);return 0;
        }
        iq4_simd_lut_pipeline[v][kind]=p;
    }
    for (int v=0;v<=2;v++) for (int kind=0;kind<3;kind++) {
        id<MTLComputePipelineState> p=iq4_simd_lut_pipeline[v][kind];
        fprintf(stderr,"iq4_simd_lut_metadata variant=%d kind=%d thread_width=%lu max_threads=%lu static_threadgroup_bytes=%lu rows_per_simd=2 simdgroups=2 threads=64 dynamic_threadgroup_bytes=128 lut_bytes=%d\n",
            v,kind,(unsigned long)p.threadExecutionWidth,(unsigned long)p.maxTotalThreadsPerThreadgroup,
            (unsigned long)p.staticThreadgroupMemoryLength,v ? 128 : 64);
    }
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    int v=mode ? lut_variant : 0;
    g_matvec_iq4_xs_pipeline=iq4_simd_lut_pipeline[v][0];
    g_matvec_iq4_xs_pair_pipeline=iq4_simd_lut_pipeline[v][1];
    g_matvec_iq4_xs_add_x0_pipeline=iq4_simd_lut_pipeline[v][2];
}
static __attribute__((unused)) void iq4_simd_lut_print_counts(void) {
    fprintf(stderr,"iq4_simd_lut_dispatch_counts single=%llu pair=%llu add=%llu\n",
        (unsigned long long)iq4_simd_lut_calls_by_kind[0],(unsigned long long)iq4_simd_lut_calls_by_kind[1],
        (unsigned long long)iq4_simd_lut_calls_by_kind[2]);
}
