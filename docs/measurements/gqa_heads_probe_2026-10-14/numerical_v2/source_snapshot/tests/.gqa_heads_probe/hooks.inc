/* Private cap-two/cap-one probes, restricted to the actual public 27B F16 route. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> gqa_groups_pipeline[3];
static uint64_t gqa_groups_calls,gqa_groups_fallbacks;
static int groups_variant=1, probe_mode;
static _Thread_local uint32_t probe_grid,probe_base_grid;
static _Thread_local int probe_eligible;

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
    probe_grid=0;probe_base_grid=0;
    probe_eligible=probe_mode && pipeline==gqa_groups_pipeline[0];
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
- (void)gqaGroupsSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    if (probe_eligible && index==0 && length==5u*sizeof(uint32_t)) {
        const uint32_t *a=bytes;
        if (a[1]==24u && a[2]==4u && a[3]==256u && a[4]==1u) {
            const uint32_t cap=groups_variant==1 ? 2u : 1u;
            probe_base_grid=8u;probe_grid=4u*((6u+cap-1u)/cap);
            [(id<MTLComputeCommandEncoder>)self setComputePipelineState:gqa_groups_pipeline[groups_variant]];
            gqa_groups_calls++;
        } else gqa_groups_fallbacks++;
        probe_eligible=0;
    }
    [(id<MTLComputeCommandEncoder>)self setBytes:bytes length:length atIndex:index];
}
- (void)gqaGroupsDispatchThreadgroups:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads {
    if (probe_grid) {
        if (groups.width<probe_base_grid) gqa_groups_fallbacks++;
        else groups.width=probe_grid+(groups.width-probe_base_grid);
    }
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
    @autoreleasepool {
    if (gqa_groups_pipeline[2]) return 1;
    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;
    gqa_groups_pipeline[0]=g_gqa_attend_n_inner_heads3_pipeline;
    if (!gqa_groups_pipeline[0]) return 0;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.gqa_heads_probe/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) {
        NSString *name=[NSString stringWithFormat:@"private_gqa_heads_probe%d",v];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        id<MTLComputePipelineState> p=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!p || p.threadExecutionWidth!=32 || p.maxTotalThreadsPerThreadgroup<256 ||
            p.staticThreadgroupMemoryLength!=gqa_groups_pipeline[0].staticThreadgroupMemoryLength) return 0;
        gqa_groups_pipeline[v]=p;
    }
    for (int v=0;v<=2;v++) fprintf(stderr,
        "gqa_heads_probe_metadata variant=%d capacity=%d width=%lu max_threads=%lu static_bytes=%lu dynamic_bytes=256\n",
        v,v==0 ? 3 : v==1 ? 2 : 1,
        (unsigned long)gqa_groups_pipeline[v].threadExecutionWidth,
        (unsigned long)gqa_groups_pipeline[v].maxTotalThreadsPerThreadgroup,
        (unsigned long)gqa_groups_pipeline[v].staticThreadgroupMemoryLength);
    return 1;
    }
}
static void scratch_select(int mode) {
    probe_mode=mode!=0;probe_eligible=0;probe_grid=0;probe_base_grid=0;
}
static void gqa_groups_print_counts(void) {
    fprintf(stderr,"gqa_heads_probe_dispatch_counts calls=%llu fallbacks=%llu\n",
       (unsigned long long)gqa_groups_calls,(unsigned long long)gqa_groups_fallbacks);
}
