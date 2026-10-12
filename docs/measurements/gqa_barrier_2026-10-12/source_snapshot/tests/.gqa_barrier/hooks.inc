/* Three IQ4_XS pipelines: K-loop unroll hints only; native arithmetic and buffers. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> gqa_barrier_pipeline[3];
static uint64_t gqa_barrier_calls,gqa_barrier_fallbacks;
static int barrier_variant=1,gqa_barrier_mode;
static _Thread_local void *gqa_barrier_encoder;
static _Thread_local int gqa_barrier_eligible;

@protocol QW3GQABarrierHooks
- (void)gqaBarrierSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)gqaBarrierSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)gqaBarrierEncoder;
- (id)gqaBarrierEncoderWithDispatchType:(MTLDispatchType)type;
- (id)gqaBarrierCommandBuffer;
- (id)gqaBarrierCommandBufferUnretained;
- (id)gqaBarrierCommandQueue;
@end
@interface NSObject (QW3GQABarrierProbe) <QW3GQABarrierHooks>
@end
@implementation NSObject (QW3GQABarrierProbe)
- (id)gqaBarrierCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)gqaBarrierCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)gqaBarrierCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)gqaBarrierEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)gqaBarrierEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)gqaBarrierSetPipeline:(id<MTLComputePipelineState>)pipeline {
    gqa_barrier_encoder=(__bridge void *)self;
    gqa_barrier_eligible=(pipeline==gqa_barrier_pipeline[0]);
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
- (void)gqaBarrierSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    if (gqa_barrier_mode && gqa_barrier_encoder==(__bridge void *)self &&
        gqa_barrier_eligible && index==0 && length==5u*sizeof(uint32_t)) {
        const uint32_t *a=bytes;
        if (a[2] && a[1]%a[2]==0 && a[1]/a[2]>0 && a[1]/a[2]<=7u) {
            [(id<MTLComputeCommandEncoder>)self setComputePipelineState:gqa_barrier_pipeline[barrier_variant]];
            gqa_barrier_calls++;
        } else gqa_barrier_fallbacks++;
        gqa_barrier_eligible=0;
    }
    [(id<MTLComputeCommandEncoder>)self setBytes:bytes length:length atIndex:index];
}

@end
static id gqa_barrier_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id gqa_barrier_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice gqa_barrier_default_device
#define MTLCopyAllDevices gqa_barrier_all_devices
#define MTLDevice MTLDevice,QW3GQABarrierHooks
#define MTLCommandQueue MTLCommandQueue,QW3GQABarrierHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3GQABarrierHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3GQABarrierHooks
#define newCommandQueue gqaBarrierCommandQueue
#define commandBuffer gqaBarrierCommandBuffer
#define commandBufferWithUnretainedReferences gqaBarrierCommandBufferUnretained
#define setBytes gqaBarrierSetBytes
#define setComputePipelineState gqaBarrierSetPipeline
#define computeCommandEncoder gqaBarrierEncoder
#define computeCommandEncoderWithDispatchType gqaBarrierEncoderWithDispatchType
#include "../../qw3_metal.m"
#undef computeCommandEncoderWithDispatchType
#undef computeCommandEncoder
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
#define qw3_metal_env_n_gpu_layers gqa_barrier_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled gqa_barrier_core_llama_split_enabled
#define qw3_metal_llama_split_start gqa_barrier_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int gqa_barrier_compile(void) {
    if (gqa_barrier_pipeline[2]) return 1;
    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;
    gqa_barrier_pipeline[0]=g_gqa_attend_n_inner_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.gqa_barrier/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) {
        NSString *name=[NSString stringWithFormat:@"private_gqa_barrier%d",v];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        id<MTLComputePipelineState> p=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!p || p.threadExecutionWidth!=32 || p.maxTotalThreadsPerThreadgroup<256 ||
            p.staticThreadgroupMemoryLength!=gqa_barrier_pipeline[0].staticThreadgroupMemoryLength) {
            fprintf(stderr,"GQA barrier compile variant=%d: %s\n",v,error.localizedDescription.UTF8String);return 0;
        }
        gqa_barrier_pipeline[v]=p;
    }
    for (int v=0;v<=2;v++) fprintf(stderr,
        "gqa_barrier_metadata variant=%d thread_width=%lu max_threads=%lu static_threadgroup_bytes=%lu dynamic_threadgroup_bytes=256 barriers_per_context=%d\n",
        v,(unsigned long)gqa_barrier_pipeline[v].threadExecutionWidth,
        (unsigned long)gqa_barrier_pipeline[v].maxTotalThreadsPerThreadgroup,
        (unsigned long)gqa_barrier_pipeline[v].staticThreadgroupMemoryLength,v ? 2 : 3);
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    gqa_barrier_mode=mode!=0;gqa_barrier_eligible=0;gqa_barrier_encoder=NULL;
}
static __attribute__((unused)) void gqa_barrier_print_counts(void) {
    fprintf(stderr,"gqa_barrier_dispatch_counts calls=%llu fallbacks=%llu\n",
        (unsigned long long)gqa_barrier_calls,(unsigned long long)gqa_barrier_fallbacks);
}
