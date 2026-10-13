/* Aligned Q4_K payload words; native geometry, scalar FP32 arithmetic and buffers. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> q4_words_pipeline[3];
static uint64_t q4_words_calls,q4_words_fallbacks;
static int q4_words_mode,words_variant=1;
static _Thread_local void *q4_words_encoder;
static _Thread_local int q4_words_seen,q4_words_eligible;
static _Thread_local int q4_words_header,q4_words_weight_aligned;

@protocol QW3Q4WordsHooks
- (void)q4WordsSetBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)q4WordsDispatch:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads;
- (void)q4WordsSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)q4WordsSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)q4WordsEncoder;
- (id)q4WordsEncoderWithDispatchType:(MTLDispatchType)type;
- (id)q4WordsCommandBuffer;
- (id)q4WordsCommandBufferUnretained;
- (id)q4WordsCommandQueue;
@end
@interface NSObject (QW3Q4WordsProbe) <QW3Q4WordsHooks>
@end
@implementation NSObject (QW3Q4WordsProbe)
- (id)q4WordsCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)q4WordsCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)q4WordsCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)q4WordsEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)q4WordsEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)q4WordsSetPipeline:(id<MTLComputePipelineState>)pipeline {
    q4_words_encoder=(__bridge void *)self;
    q4_words_seen=(pipeline==q4_words_pipeline[0]);
    q4_words_header=0;q4_words_weight_aligned=0;q4_words_eligible=0;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
- (void)q4WordsSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    if (q4_words_mode && q4_words_encoder==(__bridge void *)self &&
        q4_words_seen && index==0 && length==3u*sizeof(uint32_t)) {
        const uint32_t *a=bytes;
        uint32_t alignment=words_variant==1 ? 2u : 4u;
        q4_words_header=a[0]>0u && a[0]%256u==0u && a[1]>0u && a[2]%alignment==0u;
    }
    [(id<MTLComputeCommandEncoder>)self setBytes:bytes length:length atIndex:index];
}
- (void)q4WordsSetBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
    if (q4_words_mode && q4_words_encoder==(__bridge void *)self && q4_words_seen && index==1u) {
        NSUInteger alignment=words_variant==1 ? 2u : 4u;
        q4_words_weight_aligned=buffer!=nil && offset%alignment==0u;
    }
    [(id<MTLComputeCommandEncoder>)self setBuffer:buffer offset:offset atIndex:index];
}
- (void)q4WordsDispatch:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads {
    if (q4_words_mode && q4_words_encoder==(__bridge void *)self && q4_words_seen) {
        if (q4_words_header && q4_words_weight_aligned) {
            [(id<MTLComputeCommandEncoder>)self setComputePipelineState:q4_words_pipeline[words_variant]];
            q4_words_calls++;
        } else q4_words_fallbacks++;
        q4_words_seen=0;
    }
    [(id<MTLComputeCommandEncoder>)self dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

@end
static id q4_words_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id q4_words_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice q4_words_default_device
#define MTLCopyAllDevices q4_words_all_devices
#define MTLDevice MTLDevice,QW3Q4WordsHooks
#define MTLCommandQueue MTLCommandQueue,QW3Q4WordsHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3Q4WordsHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3Q4WordsHooks
#define newCommandQueue q4WordsCommandQueue
#define commandBuffer q4WordsCommandBuffer
#define commandBufferWithUnretainedReferences q4WordsCommandBufferUnretained
#define dispatchThreadgroups q4WordsDispatch
#define setBuffer q4WordsSetBuffer
#define setBytes q4WordsSetBytes
#define setComputePipelineState q4WordsSetPipeline
#define computeCommandEncoder q4WordsEncoder
#define computeCommandEncoderWithDispatchType q4WordsEncoderWithDispatchType
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
#define qw3_metal_env_n_gpu_layers q4_words_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled q4_words_core_llama_split_enabled
#define qw3_metal_llama_split_start q4_words_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int q4_words_compile(void) {
    if (q4_words_pipeline[2]) return 1;
    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;
    q4_words_pipeline[0]=g_matvec_q4_k_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.q4_words/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) {
        id<MTLFunction> fn=[lib newFunctionWithName:[NSString stringWithFormat:@"private_q4_words_fast%d",v]];
        id<MTLComputePipelineState> p=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!p || p.threadExecutionWidth!=32u || p.maxTotalThreadsPerThreadgroup<256u ||
            p.staticThreadgroupMemoryLength!=q4_words_pipeline[0].staticThreadgroupMemoryLength) {
            fprintf(stderr,"Q4 words compile variant=%d: %s\n",v,error.localizedDescription.UTF8String);return 0;
        }
        q4_words_pipeline[v]=p;
    }
    for (int v=0;v<=2;v++) fprintf(stderr,"q4_words_metadata variant=%d width=%lu max_threads=%lu static_threadgroup_bytes=%lu dynamic_threadgroup_bytes=128\n",v,(unsigned long)q4_words_pipeline[v].threadExecutionWidth,(unsigned long)q4_words_pipeline[v].maxTotalThreadsPerThreadgroup,(unsigned long)q4_words_pipeline[v].staticThreadgroupMemoryLength);
    return 1;
}
static void scratch_select(int mode) {
    q4_words_mode=mode!=0;q4_words_seen=0;q4_words_header=0;q4_words_weight_aligned=0;q4_words_encoder=NULL;
}
static void q4_words_print_counts(void) {
    fprintf(stderr,"q4_words_dispatch_counts calls=%llu fallbacks=%llu\n",(unsigned long long)q4_words_calls,(unsigned long long)q4_words_fallbacks);
}
