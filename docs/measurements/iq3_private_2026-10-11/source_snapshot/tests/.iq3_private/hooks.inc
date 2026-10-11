/* Private padded IQ3_S copy; preserve native FP32 expressions. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
static id<MTLComputePipelineState> iq3_private_pipeline[2][2];
static uint64_t iq3_private_calls[2];
static int layout_variant=1;
static uint64_t iq3_private_redirects,iq3_private_bytes;
static NSMutableDictionary<NSNumber *,id<MTLBuffer>> *iq3_private_weights;
static void *iq3_private_kind_key=&iq3_private_kind_key;
static id<MTLBuffer> iq3_private_bind(id<MTLBuffer>,NSUInteger *,NSUInteger,unsigned);
@protocol QW3IQ3PrivateHooks
- (void)iq3PrivateSetBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)iq3PrivateSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq3PrivateEncoder;
- (id)iq3PrivateEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq3PrivateCommandBuffer;
- (id)iq3PrivateCommandBufferUnretained;
- (id)iq3PrivateCommandQueue;
@end
@interface NSObject (QW3IQ3PrivateProbe) <QW3IQ3PrivateHooks>
@end
@implementation NSObject (QW3IQ3PrivateProbe)
- (id)iq3PrivateCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq3PrivateCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq3PrivateCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq3PrivateEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq3PrivateEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq3PrivateSetPipeline:(id<MTLComputePipelineState>)pipeline {
    unsigned active=0;
    for (int kind=0;kind<2;kind++) for (int variant=1;variant<=1;variant++)
        if (pipeline==iq3_private_pipeline[variant][kind]) {iq3_private_calls[kind]++;active=kind+1u;}
    objc_setAssociatedObject(self,iq3_private_kind_key,@(active),OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
- (void)iq3PrivateSetBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
    unsigned active=[objc_getAssociatedObject(self,iq3_private_kind_key) unsignedIntValue];
    if (active && index>=1u && index<=active) buffer=iq3_private_bind(buffer,&offset,index,active);
    [(id<MTLComputeCommandEncoder>)self setBuffer:buffer offset:offset atIndex:index];
}
@end
static id iq3_private_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq3_private_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq3_private_default_device
#define MTLCopyAllDevices iq3_private_all_devices
#define MTLDevice MTLDevice,QW3IQ3PrivateHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ3PrivateHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ3PrivateHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ3PrivateHooks
#define newCommandQueue iq3PrivateCommandQueue
#define commandBuffer iq3PrivateCommandBuffer
#define commandBufferWithUnretainedReferences iq3PrivateCommandBufferUnretained
#define setBuffer iq3PrivateSetBuffer
#define setComputePipelineState iq3PrivateSetPipeline
#define computeCommandEncoder iq3PrivateEncoder
#define computeCommandEncoderWithDispatchType iq3PrivateEncoderWithDispatchType
#include "../../qw3_metal.m"
#undef computeCommandEncoderWithDispatchType
#undef computeCommandEncoder
#undef setBuffer
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
#define qw3_metal_env_n_gpu_layers iq3_private_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq3_private_core_llama_split_enabled
#define qw3_metal_llama_split_start iq3_private_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int iq3_private_compile(void) {
    if (iq3_private_pipeline[1][1]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq3_private_pipeline[0][0]=g_matvec_iq3_s_pipeline;
    iq3_private_pipeline[0][1]=g_matvec_iq3_s_pair_pipeline;
    const char *names[]={"qw3_matvec_iq3_s_fast","qw3_matvec_iq3_s_pair_fast"};
    NSError *error=nil;
    for (int kind=0;kind<2;kind++) {
        /* Same already-compiled library and function as the baseline. */
        id<MTLFunction> fn=[g_library newFunctionWithName:[NSString stringWithUTF8String:names[kind]]];
        iq3_private_pipeline[1][kind]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!iq3_private_pipeline[1][kind] || iq3_private_pipeline[1][kind]==iq3_private_pipeline[0][kind] || iq3_private_pipeline[1][kind].maxTotalThreadsPerThreadgroup<64) {
            fprintf(stderr,"IQ3 private pipeline kind=%d: %s\n",kind,error.localizedDescription.UTF8String);return 0;
        }
    }
    return 1;
}
static id<MTLBuffer> iq3_private_pack(id<MTLBuffer> native,uint64_t offset,uint32_t ni,uint32_t no) {
    @autoreleasepool {
        if (!ni || ni%256u || !no) return nil;
        const uint64_t bytes=(uint64_t)ni/256u*no*110u;
        if (!native || offset>native.length || bytes>native.length-offset || bytes>NSUIntegerMax-80u) return nil;
        const NSUInteger length=(NSUInteger)((bytes+64u+15u)&~15ull);
        id<MTLBuffer> staging=[g_device newBufferWithLength:length options:MTLResourceStorageModeShared];
        id<MTLBuffer> private=[g_device newBufferWithLength:length options:MTLResourceStorageModePrivate];
        if (!staging || !private) return nil;
        const unsigned char *src=(const unsigned char *)native.contents+offset;
        unsigned char *dst=staging.contents;memset(dst,0xa5,length);memcpy(dst+32u,src,bytes);
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
        [blit copyFromBuffer:staging sourceOffset:0 toBuffer:private destinationOffset:0 size:length];
        [blit endEncoding];[cb commit];[cb waitUntilCompleted];
        if (cb.status!=MTLCommandBufferStatusCompleted) return nil;
        /* Verify actual GPU copy bytes through a GPU-to-CPU readback. */
        memset(dst,0,length);
        cb=[g_queue commandBuffer];blit=[cb blitCommandEncoder];
        [blit copyFromBuffer:private sourceOffset:0 toBuffer:staging destinationOffset:0 size:length];
        [blit endEncoding];[cb commit];[cb waitUntilCompleted];
        if (cb.status!=MTLCommandBufferStatusCompleted || memcmp(dst+32u,src,bytes)) return nil;
        for (NSUInteger i=0;i<32u;i++) if (dst[i]!=0xa5) return nil;
        for (NSUInteger i=32u+bytes;i<length;i++) if (dst[i]!=0xa5) return nil;
        if (private.storageMode!=MTLStorageModePrivate) return nil;
        return private;
    }
}
static int iq3_private_guards(id<MTLBuffer> buffer) {
    @autoreleasepool {
        if (!buffer || buffer.length<64u) return 0;
        id<MTLBuffer> result=[g_device newBufferWithLength:64u options:MTLResourceStorageModeShared];
        if (!result) return 0;
        id<MTLCommandBuffer> cb=[g_queue commandBuffer];id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
        [blit copyFromBuffer:buffer sourceOffset:0 toBuffer:result destinationOffset:0 size:32u];
        [blit copyFromBuffer:buffer sourceOffset:buffer.length-32u toBuffer:result destinationOffset:32u size:32u];
        [blit endEncoding];[cb commit];[cb waitUntilCompleted];
        if (cb.status!=MTLCommandBufferStatusCompleted) return 0;
        const unsigned char *p=result.contents;
        for (unsigned i=0;i<64u;i++) if (p[i]!=0xa5) return 0;
        return 1;
    }
}
static int iq3_private_prepare(qw3_engine *engine) {
    if (iq3_private_weights) return 1;
    if (g_model_view_count!=1u) return 0;
    iq3_private_weights=[NSMutableDictionary new];unsigned count=0;
    const double start=[NSDate timeIntervalSinceReferenceDate];
    const NSUInteger before=g_device.currentAllocatedSize;
    for (uint32_t i=0;i<engine->model.n_tensors;i++) {
        const qw3_tensor *w=&engine->model.tensors[i];if (w->type!=21u) continue;
        if (w->ndim!=2u || w->dim[0]%256u || w->dim[0]>UINT32_MAX || w->dim[1]>UINT32_MAX) return 0;
        uint64_t offset=0;id<MTLBuffer> native=qw3_metal_model_view_for(w->offset,w->elements/256u*110u,&offset);
        id<MTLBuffer> packed=iq3_private_pack(native,offset,(uint32_t)w->dim[0],(uint32_t)w->dim[1]);
        if (!packed || iq3_private_weights[@(offset)] || !iq3_private_guards(packed)) return 0;
        iq3_private_weights[@(offset)]=packed;count++;iq3_private_bytes+=packed.length;
    }
    fprintf(stderr,"IQ3 private prepared tensors=%u bytes=%llu seconds=%.6f all_source_bytes=exact\n",count,(unsigned long long)iq3_private_bytes,[NSDate timeIntervalSinceReferenceDate]-start);
    fprintf(stderr,"IQ3 private GPU allocated before=%llu after=%llu\n",(unsigned long long)before,(unsigned long long)g_device.currentAllocatedSize);
    return count>0;
}
static id<MTLBuffer> iq3_private_bind(id<MTLBuffer> buffer,NSUInteger *offset,NSUInteger index,unsigned active) {
    (void)index;(void)active;
    /* Matrix tests already bind explicit packed copies. Native model bindings
     * are redirected only while a private candidate pipeline is selected. */
    if (g_model_view_count!=1u || buffer!=g_model_buffers[0]) return buffer;
    id<MTLBuffer> packed=iq3_private_weights[@(*offset)];
    if (!packed) {fprintf(stderr,"IQ3 private missing packed tensor offset=%llu\n",(unsigned long long)*offset);abort();}
    *offset=32u;iq3_private_redirects++;return packed;
}
static void scratch_select(int mode) {
    setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1);
    g_matvec_iq3_s_pipeline=iq3_private_pipeline[mode ? layout_variant : 0][0];
    g_matvec_iq3_s_pair_pipeline=iq3_private_pipeline[mode ? layout_variant : 0][1];
}
