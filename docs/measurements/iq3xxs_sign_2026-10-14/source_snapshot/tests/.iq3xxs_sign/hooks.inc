/* IQ3_XXS sign parity/direct table; original FP32 vector dot and data buffers. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> iq3xxs_sign_pipeline[3];
static uint64_t iq3xxs_sign_calls;
static int sign_variant=1;
@protocol QW3IQ3XXSSignHooks
- (void)iq3xxsSignSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq3xxsSignEncoder;
- (id)iq3xxsSignEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq3xxsSignCommandBuffer;
- (id)iq3xxsSignCommandBufferUnretained;
- (id)iq3xxsSignCommandQueue;
@end
@interface NSObject (QW3IQ3XXSSignProbe) <QW3IQ3XXSSignHooks>
@end
@implementation NSObject (QW3IQ3XXSSignProbe)
- (id)iq3xxsSignCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq3xxsSignCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq3xxsSignCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq3xxsSignEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq3xxsSignEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq3xxsSignSetPipeline:(id<MTLComputePipelineState>)pipeline {
    if (pipeline==iq3xxs_sign_pipeline[1] || pipeline==iq3xxs_sign_pipeline[2]) iq3xxs_sign_calls++;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
@end
static id iq3xxs_sign_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq3xxs_sign_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq3xxs_sign_default_device
#define MTLCopyAllDevices iq3xxs_sign_all_devices
#define MTLDevice MTLDevice,QW3IQ3XXSSignHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ3XXSSignHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ3XXSSignHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ3XXSSignHooks
#define newCommandQueue iq3xxsSignCommandQueue
#define commandBuffer iq3xxsSignCommandBuffer
#define commandBufferWithUnretainedReferences iq3xxsSignCommandBufferUnretained
#define setComputePipelineState iq3xxsSignSetPipeline
#define computeCommandEncoder iq3xxsSignEncoder
#define computeCommandEncoderWithDispatchType iq3xxsSignEncoderWithDispatchType
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
#define qw3_metal_env_n_gpu_layers iq3xxs_sign_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq3xxs_sign_core_llama_split_enabled
#define qw3_metal_llama_split_start iq3xxs_sign_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int iq3xxs_sign_compile(void) {
    if (iq3xxs_sign_pipeline[2]) return 1;
    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;
    iq3xxs_sign_pipeline[0]=g_matvec_iq3_xxs_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.iq3xxs_sign/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) {
        NSString *name=[NSString stringWithFormat:@"private_iq3xxs_sign_fast%d",v];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        iq3xxs_sign_pipeline[v]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!iq3xxs_sign_pipeline[v] || iq3xxs_sign_pipeline[v].maxTotalThreadsPerThreadgroup<64 || iq3xxs_sign_pipeline[v].threadExecutionWidth!=32 || iq3xxs_sign_pipeline[v].staticThreadgroupMemoryLength!=4096u) {
            fprintf(stderr,"IQ3XXS sign compile variant=%d: %s\n",v,error.localizedDescription.UTF8String);return 0;
        }
    }
    for (int v=0;v<=2;v++) fprintf(stderr,"iq3xxs_sign_metadata variant=%d width=%lu max_threads=%lu static_threadgroup_bytes=%lu dynamic_threadgroup_bytes=128 threads=64\n",v,(unsigned long)iq3xxs_sign_pipeline[v].threadExecutionWidth,(unsigned long)iq3xxs_sign_pipeline[v].maxTotalThreadsPerThreadgroup,(unsigned long)iq3xxs_sign_pipeline[v].staticThreadgroupMemoryLength);
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    g_matvec_iq3_xxs_pipeline=iq3xxs_sign_pipeline[mode ? sign_variant : 0];
}
