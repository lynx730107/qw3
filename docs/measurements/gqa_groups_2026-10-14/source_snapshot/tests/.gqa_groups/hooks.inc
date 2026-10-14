/* Three IQ4_XS pipelines: K-loop unroll hints only; native arithmetic and buffers. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> gqa_groups_pipeline[3];
static uint64_t gqa_groups_calls,gqa_groups_fallbacks;
static int groups_variant=1,gqa_groups_mode;
static _Thread_local void *gqa_groups_encoder;
static _Thread_local int gqa_groups_eligible;
static _Thread_local uint32_t gqa_groups_multiplier=1;

@protocol QW3GQAGroupsHooks
- (void)gqaGroupsDispatchThreadgroups:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads;
- (void)gqaGroupsSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)gqaGroupsSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)gqaGroupsEncoder;
- (id)gqaGroupsEncoderWithDispatchType:(MTLDispatchType)type;
- (id)gqaGroupsCommandBuffer;
- (id)gqaGroupsCommandBufferUnretained;
- (id)gqaGroupsCommandQueue;
@end
@interface NSObject (QW3GQAGroupsProbe) <QW3GQAGroupsHooks>
@end
@implementation NSObject (QW3GQAGroupsProbe)
- (id)gqaGroupsCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)gqaGroupsCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)gqaGroupsCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)gqaGroupsEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)gqaGroupsEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)gqaGroupsSetPipeline:(id<MTLComputePipelineState>)pipeline {
    gqa_groups_multiplier=1;
    gqa_groups_encoder=(__bridge void *)self;
    gqa_groups_eligible=(pipeline==gqa_groups_pipeline[0]);
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
- (void)gqaGroupsSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    if (gqa_groups_mode && gqa_groups_encoder==(__bridge void *)self &&
        gqa_groups_eligible && index==0 && length==5u*sizeof(uint32_t)) {
        const uint32_t *a=bytes;
        if (a[2] && a[1]%a[2]==0 && a[1]/a[2]>0 && a[1]/a[2]<=8u) {
            [(id<MTLComputeCommandEncoder>)self setComputePipelineState:gqa_groups_pipeline[groups_variant]];
            const uint32_t cap=groups_variant==1 ? 1u : 3u;
            gqa_groups_multiplier=(a[1]/a[2]+cap-1u)/cap;
            gqa_groups_calls++;
        } else gqa_groups_fallbacks++;
        gqa_groups_eligible=0;
    }
    [(id<MTLComputeCommandEncoder>)self setBytes:bytes length:length atIndex:index];
}

- (void)gqaGroupsDispatchThreadgroups:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads {
    if (gqa_groups_encoder==(__bridge void *)self) groups.width*=gqa_groups_multiplier;
    [(id<MTLComputeCommandEncoder>)self dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

@end
static id gqa_groups_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id gqa_groups_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice gqa_groups_default_device
#define MTLCopyAllDevices gqa_groups_all_devices
#define MTLDevice MTLDevice,QW3GQAGroupsHooks
#define MTLCommandQueue MTLCommandQueue,QW3GQAGroupsHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3GQAGroupsHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3GQAGroupsHooks
#define newCommandQueue gqaGroupsCommandQueue
#define commandBuffer gqaGroupsCommandBuffer
#define commandBufferWithUnretainedReferences gqaGroupsCommandBufferUnretained
#define dispatchThreadgroups gqaGroupsDispatchThreadgroups
#define setBytes gqaGroupsSetBytes
#define setComputePipelineState gqaGroupsSetPipeline
#define computeCommandEncoder gqaGroupsEncoder
#define computeCommandEncoderWithDispatchType gqaGroupsEncoderWithDispatchType
#include "../../qw3_metal.m"
#undef computeCommandEncoderWithDispatchType
#undef computeCommandEncoder
#undef dispatchThreadgroups
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
#define qw3_metal_env_n_gpu_layers gqa_groups_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled gqa_groups_core_llama_split_enabled
#define qw3_metal_llama_split_start gqa_groups_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int gqa_groups_compile(void) {
    if (gqa_groups_pipeline[2]) return 1;
    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;
    gqa_groups_pipeline[0]=g_gqa_attend_n_inner_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.gqa_groups/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) {
        NSString *name=[NSString stringWithFormat:@"private_gqa_groups%d",v];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        id<MTLComputePipelineState> p=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!p || p.threadExecutionWidth!=32 || p.maxTotalThreadsPerThreadgroup<256 ||
            p.staticThreadgroupMemoryLength!=gqa_groups_pipeline[0].staticThreadgroupMemoryLength) {
            fprintf(stderr,"GQA groups compile variant=%d: %s\n",v,error.localizedDescription.UTF8String);return 0;
        }
        gqa_groups_pipeline[v]=p;
    }
    for (int v=0;v<=2;v++) fprintf(stderr,
        "gqa_groups_metadata variant=%d thread_width=%lu max_threads=%lu static_threadgroup_bytes=%lu dynamic_threadgroup_bytes=256 original_barriers_per_context=3 array_head_capacity=%d\n",
        v,(unsigned long)gqa_groups_pipeline[v].threadExecutionWidth,
        (unsigned long)gqa_groups_pipeline[v].maxTotalThreadsPerThreadgroup,
        (unsigned long)gqa_groups_pipeline[v].staticThreadgroupMemoryLength,v==1 ? 1 : v==2 ? 3 : 8);
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    gqa_groups_mode=mode!=0;gqa_groups_eligible=0;gqa_groups_encoder=NULL;
}
static __attribute__((unused)) void gqa_groups_print_counts(void) {
    fprintf(stderr,"gqa_groups_dispatch_counts calls=%llu fallbacks=%llu\n",
        (unsigned long long)gqa_groups_calls,(unsigned long long)gqa_groups_fallbacks);
}
