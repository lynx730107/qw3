/* Integer shape constants; native weights, grid, FP32 arithmetic and data buffers. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> iq3_shape_pipeline[3][5];
static uint64_t iq3_shape_calls;
static int shape_variant=1,iq3_shape_mode;
static uint64_t iq3_shape_fallbacks;
static const uint32_t iq3_shape_inputs[5]={256u,4352u,5120u,6144u,17408u};
static _Thread_local void *iq3_shape_encoder;
static _Thread_local int iq3_shape_seen;

@protocol QW3IQ3ShapeHooks
- (void)iq3ShapeSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)iq3ShapeSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq3ShapeEncoder;
- (id)iq3ShapeEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq3ShapeCommandBuffer;
- (id)iq3ShapeCommandBufferUnretained;
- (id)iq3ShapeCommandQueue;
@end
@interface NSObject (QW3IQ3ShapeProbe) <QW3IQ3ShapeHooks>
@end
@implementation NSObject (QW3IQ3ShapeProbe)
- (id)iq3ShapeCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq3ShapeCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq3ShapeCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq3ShapeEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq3ShapeEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq3ShapeSetPipeline:(id<MTLComputePipelineState>)pipeline {
    iq3_shape_encoder=(__bridge void *)self;
    iq3_shape_seen=iq3_shape_mode && pipeline==iq3_shape_pipeline[0][0];
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
- (void)iq3ShapeSetBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    if (iq3_shape_seen && iq3_shape_encoder==(__bridge void *)self && index==0 && length==3u*sizeof(uint32_t)) {
        const uint32_t *a=bytes;int shape=-1;
        for (int i=0;i<5;i++) if (a[0]==iq3_shape_inputs[i] && a[2]==a[0]/256u*110u) shape=i;
        if (shape>=0) {
            [(id<MTLComputeCommandEncoder>)self setComputePipelineState:iq3_shape_pipeline[shape_variant][shape]];
            iq3_shape_calls++;
        } else iq3_shape_fallbacks++;
        iq3_shape_seen=0;
    }
    [(id<MTLComputeCommandEncoder>)self setBytes:bytes length:length atIndex:index];
}

@end
static id iq3_shape_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq3_shape_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq3_shape_default_device
#define MTLCopyAllDevices iq3_shape_all_devices
#define MTLDevice MTLDevice,QW3IQ3ShapeHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ3ShapeHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ3ShapeHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ3ShapeHooks
#define newCommandQueue iq3ShapeCommandQueue
#define commandBuffer iq3ShapeCommandBuffer
#define commandBufferWithUnretainedReferences iq3ShapeCommandBufferUnretained
#define setBytes iq3ShapeSetBytes
#define setComputePipelineState iq3ShapeSetPipeline
#define computeCommandEncoder iq3ShapeEncoder
#define computeCommandEncoderWithDispatchType iq3ShapeEncoderWithDispatchType
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
#define qw3_metal_env_n_gpu_layers iq3_shape_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq3_shape_core_llama_split_enabled
#define qw3_metal_llama_split_start iq3_shape_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int iq3_shape_compile(void) {
    if (iq3_shape_pipeline[2][4]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    for (int i=0;i<5;i++) iq3_shape_pipeline[0][i]=g_matvec_iq3_s_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.iq3_shape/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) for (int i=0;i<5;i++) {
        uint32_t nb32=iq3_shape_inputs[i]/32u,stride=iq3_shape_inputs[i]/256u*110u;
        MTLFunctionConstantValues *constants=[MTLFunctionConstantValues new];
        [constants setConstantValue:&nb32 type:MTLDataTypeUInt atIndex:902];
        if (v==2) [constants setConstantValue:&stride type:MTLDataTypeUInt atIndex:903];
        NSString *name=[NSString stringWithFormat:@"private_iq3_shape_fast%d",v];
        id<MTLFunction> fn=lib ? [lib newFunctionWithName:name constantValues:constants error:&error] : nil;
        id<MTLComputePipelineState> p=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!p || p.threadExecutionWidth!=32 || p.maxTotalThreadsPerThreadgroup<64 ||
            p.staticThreadgroupMemoryLength!=iq3_shape_pipeline[0][0].staticThreadgroupMemoryLength) {
            fprintf(stderr,"IQ3 shape compile variant=%d n_in=%u: %s\n",v,iq3_shape_inputs[i],error.localizedDescription.UTF8String);return 0;
        }
        iq3_shape_pipeline[v][i]=p;
        fprintf(stderr,"iq3_shape_metadata variant=%d n_in=%u nb32=%u stride_constant=%u width=%lu max_threads=%lu static_threadgroup_bytes=%lu matrix_dynamic_threadgroup_bytes=256 native_dynamic_threadgroup_bytes=128 threads=64\n",v,iq3_shape_inputs[i],nb32,v==2 ? stride : 0u,(unsigned long)p.threadExecutionWidth,(unsigned long)p.maxTotalThreadsPerThreadgroup,(unsigned long)p.staticThreadgroupMemoryLength);
    }
    return 1;
}
static __attribute__((unused)) void scratch_select(int mode) {
    iq3_shape_mode=mode!=0;iq3_shape_seen=0;iq3_shape_encoder=NULL;
}
