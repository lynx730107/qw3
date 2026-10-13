/* Aligned Q5_K payload words; native geometry, FP32 arithmetic and buffers. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> q5_words_pipeline[3];
static uint64_t q5_words_calls,q5_words_fallbacks;
static int q5_words_mode,words_variant=1;
static _Thread_local void *q5_words_encoder;
static _Thread_local int q5_words_seen,q5_words_eligible;
static _Thread_local int q5_words_header,q5_words_weight_aligned;

@protocol QW3Q5WordsHooks
- (void)q5WordsSetBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)q5WordsDispatch:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads;
- (void)q5WordsSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)q5WordsSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)q5WordsEncoder;
- (id)q5WordsEncoderWithDispatchType:(MTLDispatchType)type;
- (id)q5WordsCommandBuffer;
- (id)q5WordsCommandBufferUnretained;
- (id)q5WordsCommandQueue;
@end
@interface NSObject (QW3Q5WordsProbe) <QW3Q5WordsHooks>
@end
@implementation NSObject (QW3Q5WordsProbe)
- (id)q5WordsCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)q5WordsCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)q5WordsCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)q5WordsEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)q5WordsEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)q5WordsSetPipeline:(id<MTLComputePipelineState>)pipeline {
    q5_words_encoder=(__bridge void *)self;
    q5_words_seen=(pipeline==q5_words_pipeline[0]);
    q5_words_header=0;q5_words_weight_aligned=0;q5_words_eligible=0;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
- (void)q5WordsSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    if (q5_words_mode && q5_words_encoder==(__bridge void *)self &&
        q5_words_seen && index==0 && length==3u*sizeof(uint32_t)) {
        const uint32_t *a=bytes;
        uint32_t alignment=8u;
        q5_words_header=a[0]>0u && a[0]%256u==0u && a[1]>0u && a[2]%alignment==0u;
    }
    [(id<MTLComputeCommandEncoder>)self setBytes:bytes length:length atIndex:index];
}
- (void)q5WordsSetBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
    if (q5_words_mode && q5_words_encoder==(__bridge void *)self && q5_words_seen && index==1u) {
        NSUInteger alignment=8u;
        q5_words_weight_aligned=buffer!=nil && offset%alignment==0u;
    }
    [(id<MTLComputeCommandEncoder>)self setBuffer:buffer offset:offset atIndex:index];
}
- (void)q5WordsDispatch:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads {
    if (q5_words_mode && q5_words_encoder==(__bridge void *)self && q5_words_seen) {
        if (q5_words_header && q5_words_weight_aligned) {
            [(id<MTLComputeCommandEncoder>)self setComputePipelineState:q5_words_pipeline[words_variant]];
            q5_words_calls++;
        } else q5_words_fallbacks++;
        q5_words_seen=0;
    }
    [(id<MTLComputeCommandEncoder>)self dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

@end
static id q5_words_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id q5_words_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice q5_words_default_device
#define MTLCopyAllDevices q5_words_all_devices
#define MTLDevice MTLDevice,QW3Q5WordsHooks
#define MTLCommandQueue MTLCommandQueue,QW3Q5WordsHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3Q5WordsHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3Q5WordsHooks
#define newCommandQueue q5WordsCommandQueue
#define commandBuffer q5WordsCommandBuffer
#define commandBufferWithUnretainedReferences q5WordsCommandBufferUnretained
#define dispatchThreadgroups q5WordsDispatch
#define setBuffer q5WordsSetBuffer
#define setBytes q5WordsSetBytes
#define setComputePipelineState q5WordsSetPipeline
#define computeCommandEncoder q5WordsEncoder
#define computeCommandEncoderWithDispatchType q5WordsEncoderWithDispatchType
#include "../../qw3_metal.m"
#undef computeCommandEncoderWithDispatchType
#undef computeCommandEncoder
#undef dispatchThreadgroups
#undef setBuffer
#undef setBytes
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
#define qw3_metal_env_n_gpu_layers q5_words_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled q5_words_core_llama_split_enabled
#define qw3_metal_llama_split_start q5_words_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int q5_words_compile(void) {
    if (q5_words_pipeline[2]) return 1;
    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;
    q5_words_pipeline[0]=g_matvec_q5_k_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.q5_words/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) {
        id<MTLFunction> fn=[lib newFunctionWithName:[NSString stringWithFormat:@"private_q5_words_fast%d",v]];
        id<MTLComputePipelineState> p=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!p || p.threadExecutionWidth!=32u || p.maxTotalThreadsPerThreadgroup<64u ||
            p.staticThreadgroupMemoryLength!=q5_words_pipeline[0].staticThreadgroupMemoryLength) {
            fprintf(stderr,"Q5 words compile variant=%d: %s\n",v,error.localizedDescription.UTF8String);return 0;
        }
        q5_words_pipeline[v]=p;
    }
    for (int v=0;v<=2;v++) fprintf(stderr,"q5_words_metadata variant=%d width=%lu max_threads=%lu static_threadgroup_bytes=%lu dynamic_threadgroup_bytes=128\n",v,(unsigned long)q5_words_pipeline[v].threadExecutionWidth,(unsigned long)q5_words_pipeline[v].maxTotalThreadsPerThreadgroup,(unsigned long)q5_words_pipeline[v].staticThreadgroupMemoryLength);
    return 1;
}
static void scratch_select(int mode) {
    q5_words_mode=mode!=0;q5_words_seen=0;q5_words_header=0;q5_words_weight_aligned=0;q5_words_encoder=NULL;
}
static void q5_words_print_counts(void) {
    fprintf(stderr,"q5_words_dispatch_counts calls=%llu fallbacks=%llu\n",(unsigned long long)q5_words_calls,(unsigned long long)q5_words_fallbacks);
}
