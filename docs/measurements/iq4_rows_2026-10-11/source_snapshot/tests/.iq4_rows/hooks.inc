/* Private row-reuse variants; preserve native FP32 expressions per row. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
static id<MTLComputePipelineState> iq4_rows_pipeline[3][3];
static uint64_t iq4_rows_calls[3];
static int iq4_variant=1;
static void *iq4_rows_factor_key=&iq4_rows_factor_key;
@protocol QW3IQ4RowsHooks
- (void)iq4RowsDispatch:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads;
- (void)iq4RowsSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq4RowsEncoder;
- (id)iq4RowsEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq4RowsCommandBuffer;
- (id)iq4RowsCommandBufferUnretained;
- (id)iq4RowsCommandQueue;
@end
@interface NSObject (QW3IQ4RowsProbe) <QW3IQ4RowsHooks>
@end
@implementation NSObject (QW3IQ4RowsProbe)
- (id)iq4RowsCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq4RowsCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq4RowsCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq4RowsEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq4RowsEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq4RowsSetPipeline:(id<MTLComputePipelineState>)pipeline {
    unsigned factor=1;
    for (int kind=0;kind<3;kind++) for (int variant=1;variant<=2;variant++)
        if (pipeline==iq4_rows_pipeline[variant][kind]) {iq4_rows_calls[kind]++;factor=variant==1 ? 2u : 4u;}
    objc_setAssociatedObject(self,iq4_rows_factor_key,@(factor),OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
- (void)iq4RowsDispatch:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads {
    unsigned factor=[objc_getAssociatedObject(self,iq4_rows_factor_key) unsignedIntValue];
    if (factor>1u) {
        if (groups.height!=1 || groups.depth!=1 || threads.width!=64) abort();
        groups.width=(groups.width+factor-1u)/factor;
    }
    [(id<MTLComputeCommandEncoder>)self dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}
@end
static id iq4_rows_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq4_rows_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq4_rows_default_device
#define MTLCopyAllDevices iq4_rows_all_devices
#define MTLDevice MTLDevice,QW3IQ4RowsHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ4RowsHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ4RowsHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ4RowsHooks
#define newCommandQueue iq4RowsCommandQueue
#define commandBuffer iq4RowsCommandBuffer
#define commandBufferWithUnretainedReferences iq4RowsCommandBufferUnretained
#define dispatchThreadgroups iq4RowsDispatch
#define setComputePipelineState iq4RowsSetPipeline
#define computeCommandEncoder iq4RowsEncoder
#define computeCommandEncoderWithDispatchType iq4RowsEncoderWithDispatchType
#include "../../qw3_metal.m"
#undef computeCommandEncoderWithDispatchType
#undef computeCommandEncoder
#undef dispatchThreadgroups
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
#define qw3_metal_env_n_gpu_layers iq4_rows_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq4_rows_core_llama_split_enabled
#define qw3_metal_llama_split_start iq4_rows_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int iq4_rows_compile(void) {
    if (iq4_rows_pipeline[2][2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq4_rows_pipeline[0][0]=g_matvec_iq4_xs_pipeline;
    iq4_rows_pipeline[0][1]=g_matvec_iq4_xs_pair_pipeline;
    iq4_rows_pipeline[0][2]=g_matvec_iq4_xs_add_x0_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    NSString *source=[NSString stringWithContentsOfFile:@"tests/.iq4_rows/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) for (int kind=0;kind<3;kind++) {
        NSString *name=[NSString stringWithFormat:@"private_iq4_rows%u_%d",v==1 ? 4u : 8u,kind];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        iq4_rows_pipeline[v][kind]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!iq4_rows_pipeline[v][kind] || iq4_rows_pipeline[v][kind].maxTotalThreadsPerThreadgroup<64) {
            fprintf(stderr,"IQ4 rows compile variant=%d kind=%d: %s\n",v,kind,error.localizedDescription.UTF8String);return 0;
        }
    }
    return 1;
}
static void scratch_select(int mode) {
    setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1);
    int v=mode ? iq4_variant : 0;
    g_matvec_iq4_xs_pipeline=iq4_rows_pipeline[v][0];
    g_matvec_iq4_xs_pair_pipeline=iq4_rows_pipeline[v][1];
    g_matvec_iq4_xs_add_x0_pipeline=iq4_rows_pipeline[v][2];
}

static int iq4_rows_model_aligned(qw3_engine *engine) {
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
