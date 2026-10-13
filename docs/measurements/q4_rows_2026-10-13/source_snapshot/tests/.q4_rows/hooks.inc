/* Two Q4_K rows per workgroup; native per-row arithmetic and data buffers. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> q4_rows_pipeline[2];
static uint64_t q4_rows_calls,q4_rows_fallbacks;
static int q4_rows_mode;
static _Thread_local void *q4_rows_encoder;
static _Thread_local int q4_rows_seen,q4_rows_eligible;
static _Thread_local uint32_t q4_rows_out;

@protocol QW3Q4RowsHooks
- (void)q4RowsDispatch:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads;
- (void)q4RowsSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)q4RowsSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)q4RowsEncoder;
- (id)q4RowsEncoderWithDispatchType:(MTLDispatchType)type;
- (id)q4RowsCommandBuffer;
- (id)q4RowsCommandBufferUnretained;
- (id)q4RowsCommandQueue;
@end
@interface NSObject (QW3Q4RowsProbe) <QW3Q4RowsHooks>
@end
@implementation NSObject (QW3Q4RowsProbe)
- (id)q4RowsCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)q4RowsCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)q4RowsCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)q4RowsEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)q4RowsEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)q4RowsSetPipeline:(id<MTLComputePipelineState>)pipeline {
    q4_rows_encoder=(__bridge void *)self;
    q4_rows_seen=(pipeline==q4_rows_pipeline[0]);
    q4_rows_eligible=0;q4_rows_out=0;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
- (void)q4RowsSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    if (q4_rows_mode && q4_rows_encoder==(__bridge void *)self &&
        q4_rows_seen && index==0 && length==3u*sizeof(uint32_t)) {
        const uint32_t *a=bytes;
        q4_rows_out=a[1];
        q4_rows_eligible=a[0]>0u && a[0]%256u==0u && a[0]<=8192u && a[1]>0u;
    }
    [(id<MTLComputeCommandEncoder>)self setBytes:bytes length:length atIndex:index];
}
- (void)q4RowsDispatch:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads {
    if (q4_rows_mode && q4_rows_encoder==(__bridge void *)self && q4_rows_seen) {
        if (q4_rows_eligible && groups.width>=q4_rows_out && groups.height==1u && groups.depth==1u &&
            threads.width==64u && threads.height==1u && threads.depth==1u) {
            [(id<MTLComputeCommandEncoder>)self setComputePipelineState:q4_rows_pipeline[1]];
            groups.width=(groups.width+1u)/2u;
            q4_rows_calls++;
        } else q4_rows_fallbacks++;
        q4_rows_seen=0;
    }
    [(id<MTLComputeCommandEncoder>)self dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

@end
static id q4_rows_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id q4_rows_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice q4_rows_default_device
#define MTLCopyAllDevices q4_rows_all_devices
#define MTLDevice MTLDevice,QW3Q4RowsHooks
#define MTLCommandQueue MTLCommandQueue,QW3Q4RowsHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3Q4RowsHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3Q4RowsHooks
#define newCommandQueue q4RowsCommandQueue
#define commandBuffer q4RowsCommandBuffer
#define commandBufferWithUnretainedReferences q4RowsCommandBufferUnretained
#define dispatchThreadgroups q4RowsDispatch
#define setBytes q4RowsSetBytes
#define setComputePipelineState q4RowsSetPipeline
#define computeCommandEncoder q4RowsEncoder
#define computeCommandEncoderWithDispatchType q4RowsEncoderWithDispatchType
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
#define qw3_metal_env_n_gpu_layers q4_rows_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled q4_rows_core_llama_split_enabled
#define qw3_metal_llama_split_start q4_rows_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int q4_rows_compile(void) {
    if (q4_rows_pipeline[1]) return 1;
    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;
    q4_rows_pipeline[0]=g_matvec_q4_k_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.q4_rows/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    id<MTLFunction> fn=[lib newFunctionWithName:@"private_q4_rows_fast"];
    id<MTLComputePipelineState> p=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
    if (!p || p.threadExecutionWidth!=32u || p.maxTotalThreadsPerThreadgroup<64u ||
        p.staticThreadgroupMemoryLength!=q4_rows_pipeline[0].staticThreadgroupMemoryLength) {
        fprintf(stderr,"Q4 rows compile: %s\n",error.localizedDescription.UTF8String);return 0;
    }
    q4_rows_pipeline[1]=p;
    for (int v=0;v<=1;v++) fprintf(stderr,"q4_rows_metadata variant=%d width=%lu max_threads=%lu static_threadgroup_bytes=%lu dynamic_threadgroup_bytes=128 threads=64\n",v,(unsigned long)q4_rows_pipeline[v].threadExecutionWidth,(unsigned long)q4_rows_pipeline[v].maxTotalThreadsPerThreadgroup,(unsigned long)q4_rows_pipeline[v].staticThreadgroupMemoryLength);
    return 1;
}
static void scratch_select(int mode) {
    q4_rows_mode=mode!=0;q4_rows_seen=0;q4_rows_eligible=0;q4_rows_encoder=NULL;
}
static void q4_rows_print_counts(void) {
    fprintf(stderr,"q4_rows_dispatch_counts calls=%llu fallbacks=%llu\n",(unsigned long long)q4_rows_calls,(unsigned long long)q4_rows_fallbacks);
}
