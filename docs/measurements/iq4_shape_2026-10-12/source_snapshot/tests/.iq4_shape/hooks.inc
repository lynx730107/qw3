/* Three IQ4_XS pipelines: K-loop unroll hints only; native arithmetic and buffers. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> iq4_shape_pipeline[3][5][3];
static uint64_t iq4_shape_calls,iq4_shape_calls_by_kind[3];
static int shape_variant=1, iq4_shape_mode;
static uint64_t iq4_shape_fallbacks;
static const uint32_t iq4_shape_inputs[5]={256u,4352u,5120u,6144u,17408u};
static _Thread_local void *iq4_shape_encoder;
static _Thread_local int iq4_shape_kind=-1;

@protocol QW3IQ4ShapeHooks
- (void)iq4ShapeSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)iq4ShapeSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq4ShapeEncoder;
- (id)iq4ShapeEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq4ShapeCommandBuffer;
- (id)iq4ShapeCommandBufferUnretained;
- (id)iq4ShapeCommandQueue;
@end
@interface NSObject (QW3IQ4ShapeProbe) <QW3IQ4ShapeHooks>
@end
@implementation NSObject (QW3IQ4ShapeProbe)
- (id)iq4ShapeCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq4ShapeCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq4ShapeCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq4ShapeEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq4ShapeEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq4ShapeSetPipeline:(id<MTLComputePipelineState>)pipeline {
    iq4_shape_encoder=(__bridge void *)self;
    iq4_shape_kind=-1;
    if (iq4_shape_mode) for (int kind=0;kind<3;kind++)
        if (pipeline==iq4_shape_pipeline[0][0][kind]) iq4_shape_kind=kind;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
- (void)iq4ShapeSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    if (iq4_shape_mode && iq4_shape_encoder==(__bridge void *)self &&
        iq4_shape_kind>=0 && index==0 && length==3u*sizeof(uint32_t)) {
        const uint32_t *args=bytes;
        int shape=-1;
        for (int i=0;i<5;i++) if (args[0]==iq4_shape_inputs[i] && args[2]==args[0]/256u*136u) shape=i;
        if (shape>=0) {
            id<MTLComputePipelineState> pipeline=iq4_shape_pipeline[shape_variant][shape][iq4_shape_kind];
            [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
            iq4_shape_calls++;iq4_shape_calls_by_kind[iq4_shape_kind]++;
        } else iq4_shape_fallbacks++;
        iq4_shape_kind=-1;
    }
    [(id<MTLComputeCommandEncoder>)self setBytes:bytes length:length atIndex:index];
}

@end
static id iq4_shape_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq4_shape_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq4_shape_default_device
#define MTLCopyAllDevices iq4_shape_all_devices
#define MTLDevice MTLDevice,QW3IQ4ShapeHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ4ShapeHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ4ShapeHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ4ShapeHooks
#define newCommandQueue iq4ShapeCommandQueue
#define commandBuffer iq4ShapeCommandBuffer
#define commandBufferWithUnretainedReferences iq4ShapeCommandBufferUnretained
#define setBytes iq4ShapeSetBytes
#define setComputePipelineState iq4ShapeSetPipeline
#define computeCommandEncoder iq4ShapeEncoder
#define computeCommandEncoderWithDispatchType iq4ShapeEncoderWithDispatchType
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
#define qw3_metal_env_n_gpu_layers iq4_shape_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq4_shape_core_llama_split_enabled
#define qw3_metal_llama_split_start iq4_shape_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int iq4_shape_compile(void) {
    if (iq4_shape_pipeline[2][4][2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    for (int i=0;i<5;i++) {
        iq4_shape_pipeline[0][i][0]=g_matvec_iq4_xs_pipeline;
        iq4_shape_pipeline[0][i][1]=g_matvec_iq4_xs_pair_pipeline;
        iq4_shape_pipeline[0][i][2]=g_matvec_iq4_xs_add_x0_pipeline;
    }
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.iq4_shape/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) for (int i=0;i<5;i++) for (int kind=0;kind<3;kind++) {
        uint32_t blocks=iq4_shape_inputs[i]/256u,stride=blocks*136u;
        MTLFunctionConstantValues *constants=[MTLFunctionConstantValues new];
        [constants setConstantValue:&blocks type:MTLDataTypeUInt atIndex:900];
        if (v==2) [constants setConstantValue:&stride type:MTLDataTypeUInt atIndex:901];
        NSString *name=[NSString stringWithFormat:@"private_iq4_shape_fast%d_%d",v,kind];
        id<MTLFunction> fn=lib ? [lib newFunctionWithName:name constantValues:constants error:&error] : nil;
        id<MTLComputePipelineState> p=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!p || p.maxTotalThreadsPerThreadgroup<64u || p.threadExecutionWidth!=32 ||
            p.staticThreadgroupMemoryLength!=iq4_shape_pipeline[0][0][kind].staticThreadgroupMemoryLength) {
            fprintf(stderr,"IQ4 shape compile variant=%d n_in=%u kind=%d: %s\n",v,iq4_shape_inputs[i],kind,error.localizedDescription.UTF8String);return 0;
        }
        iq4_shape_pipeline[v][i][kind]=p;
        fprintf(stderr,"iq4_shape_metadata variant=%d n_in=%u blocks=%u stride_constant=%u kind=%d thread_width=%lu max_threads=%lu static_threadgroup_bytes=%lu threads=64 dynamic_threadgroup_bytes=128\n",
            v,iq4_shape_inputs[i],blocks,v==2 ? stride : 0u,kind,(unsigned long)p.threadExecutionWidth,
            (unsigned long)p.maxTotalThreadsPerThreadgroup,(unsigned long)p.staticThreadgroupMemoryLength);
    }
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    iq4_shape_mode=mode!=0;iq4_shape_kind=-1;iq4_shape_encoder=NULL;
}
static __attribute__((unused)) void iq4_shape_print_counts(void) {
    fprintf(stderr,"iq4_shape_dispatch_counts single=%llu pair=%llu add=%llu fallbacks=%llu\n",
        (unsigned long long)iq4_shape_calls_by_kind[0],(unsigned long long)iq4_shape_calls_by_kind[1],
        (unsigned long long)iq4_shape_calls_by_kind[2],(unsigned long long)iq4_shape_fallbacks);
}
