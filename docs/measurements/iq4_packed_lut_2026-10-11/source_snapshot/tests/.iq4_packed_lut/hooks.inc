/* Private packed integer lookup; native FP32 expressions are retained. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> iq4_packed_lut_pipeline[3][3];
static uint64_t iq4_packed_lut_calls[3];
static int iq4_variant=1;
@protocol QW3IQ4PackedLutHooks
- (void)iq4PackedLutSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq4PackedLutEncoder;
- (id)iq4PackedLutEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq4PackedLutCommandBuffer;
- (id)iq4PackedLutCommandBufferUnretained;
- (id)iq4PackedLutCommandQueue;
@end
@interface NSObject (QW3IQ4PackedLutProbe) <QW3IQ4PackedLutHooks>
@end
@implementation NSObject (QW3IQ4PackedLutProbe)
- (id)iq4PackedLutCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq4PackedLutCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq4PackedLutCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq4PackedLutEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq4PackedLutEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq4PackedLutSetPipeline:(id<MTLComputePipelineState>)pipeline {
    for (int kind=0;kind<3;kind++) for (int variant=1;variant<=2;variant++)
        if (pipeline==iq4_packed_lut_pipeline[variant][kind]) iq4_packed_lut_calls[kind]++;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
@end
static id iq4_packed_lut_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq4_packed_lut_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq4_packed_lut_default_device
#define MTLCopyAllDevices iq4_packed_lut_all_devices
#define MTLDevice MTLDevice,QW3IQ4PackedLutHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ4PackedLutHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ4PackedLutHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ4PackedLutHooks
#define newCommandQueue iq4PackedLutCommandQueue
#define commandBuffer iq4PackedLutCommandBuffer
#define commandBufferWithUnretainedReferences iq4PackedLutCommandBufferUnretained
#define setComputePipelineState iq4PackedLutSetPipeline
#define computeCommandEncoder iq4PackedLutEncoder
#define computeCommandEncoderWithDispatchType iq4PackedLutEncoderWithDispatchType
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
#define qw3_metal_env_n_gpu_layers iq4_packed_lut_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq4_packed_lut_core_llama_split_enabled
#define qw3_metal_llama_split_start iq4_packed_lut_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int iq4_packed_lut_compile(void) {
    if (iq4_packed_lut_pipeline[2][2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq4_packed_lut_pipeline[0][0]=g_matvec_iq4_xs_pipeline;
    iq4_packed_lut_pipeline[0][1]=g_matvec_iq4_xs_pair_pipeline;
    iq4_packed_lut_pipeline[0][2]=g_matvec_iq4_xs_add_x0_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    NSString *source=[NSString stringWithContentsOfFile:@"tests/.iq4_packed_lut/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) for (int kind=0;kind<3;kind++) {
        NSString *name=[NSString stringWithFormat:@"private_iq4_packed%d_%d",v,kind];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        iq4_packed_lut_pipeline[v][kind]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!iq4_packed_lut_pipeline[v][kind] || iq4_packed_lut_pipeline[v][kind].maxTotalThreadsPerThreadgroup<64) {
            fprintf(stderr,"IQ4 packed LUT compile variant=%d kind=%d: %s\n",v,kind,error.localizedDescription.UTF8String);return 0;
        }
    }
    return 1;
}
static void scratch_select(int mode) {
    setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1);
    int v=mode ? iq4_variant : 0;
    g_matvec_iq4_xs_pipeline=iq4_packed_lut_pipeline[v][0];
    g_matvec_iq4_xs_pair_pipeline=iq4_packed_lut_pipeline[v][1];
    g_matvec_iq4_xs_add_x0_pipeline=iq4_packed_lut_pipeline[v][2];
}

static int iq4_packed_lut_model_aligned(qw3_engine *engine) {
    uint32_t count=0;
    for (uint32_t i=0;i<engine->model.n_tensors;i++) {
        const qw3_tensor *w=&engine->model.tensors[i];
        if (w->type!=23u) continue;
        uint64_t offset=0;
        id<MTLBuffer> buffer=qw3_metal_model_view_for(w->offset,w->elements/256u*136u,&offset);
        if (!buffer || (offset&7u) || (w->dim[0]%256u)) return 0;
        count++;
    }
    fprintf(stderr,"IQ4 aligned tensors=%u block_bytes=136 alignment=8\n",count);
    return count>0;
}
