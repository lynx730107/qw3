/* Private padded IQ3_S copy; preserve native FP32 expressions. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
static id<MTLComputePipelineState> iq3_layout_pipeline[3][2];
static uint64_t iq3_layout_calls[2];
static int layout_variant=1;
static uint64_t iq3_layout_redirects,iq3_layout_bytes;
static NSMutableDictionary<NSNumber *,id<MTLBuffer>> *iq3_layout_weights;
static void *iq3_layout_kind_key=&iq3_layout_kind_key;
static id<MTLBuffer> iq3_layout_bind(id<MTLBuffer>,NSUInteger *,NSUInteger,unsigned);
@protocol QW3IQ3LayoutHooks
- (void)iq3LayoutSetBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)iq3LayoutSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq3LayoutEncoder;
- (id)iq3LayoutEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq3LayoutCommandBuffer;
- (id)iq3LayoutCommandBufferUnretained;
- (id)iq3LayoutCommandQueue;
@end
@interface NSObject (QW3IQ3LayoutProbe) <QW3IQ3LayoutHooks>
@end
@implementation NSObject (QW3IQ3LayoutProbe)
- (id)iq3LayoutCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq3LayoutCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq3LayoutCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq3LayoutEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq3LayoutEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq3LayoutSetPipeline:(id<MTLComputePipelineState>)pipeline {
    unsigned active=0;
    for (int kind=0;kind<2;kind++) for (int variant=1;variant<=2;variant++)
        if (pipeline==iq3_layout_pipeline[variant][kind]) {iq3_layout_calls[kind]++;active=kind+1u;}
    objc_setAssociatedObject(self,iq3_layout_kind_key,@(active),OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
- (void)iq3LayoutSetBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
    unsigned active=[objc_getAssociatedObject(self,iq3_layout_kind_key) unsignedIntValue];
    if (active && index>=1u && index<=active) buffer=iq3_layout_bind(buffer,&offset,index,active);
    [(id<MTLComputeCommandEncoder>)self setBuffer:buffer offset:offset atIndex:index];
}
@end
static id iq3_layout_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq3_layout_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq3_layout_default_device
#define MTLCopyAllDevices iq3_layout_all_devices
#define MTLDevice MTLDevice,QW3IQ3LayoutHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ3LayoutHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ3LayoutHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ3LayoutHooks
#define newCommandQueue iq3LayoutCommandQueue
#define commandBuffer iq3LayoutCommandBuffer
#define commandBufferWithUnretainedReferences iq3LayoutCommandBufferUnretained
#define setBuffer iq3LayoutSetBuffer
#define setComputePipelineState iq3LayoutSetPipeline
#define computeCommandEncoder iq3LayoutEncoder
#define computeCommandEncoderWithDispatchType iq3LayoutEncoderWithDispatchType
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
#define qw3_metal_env_n_gpu_layers iq3_layout_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq3_layout_core_llama_split_enabled
#define qw3_metal_llama_split_start iq3_layout_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int iq3_layout_compile(void) {
    if (iq3_layout_pipeline[2][1]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq3_layout_pipeline[0][0]=g_matvec_iq3_s_pipeline;
    iq3_layout_pipeline[0][1]=g_matvec_iq3_s_pair_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    NSString *source=[NSString stringWithContentsOfFile:@"tests/.iq3_layout/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) for (int kind=0;kind<2;kind++) {
        NSString *name=[NSString stringWithFormat:@"private_iq3_layout%d_%d",v,kind];
        id<MTLFunction> fn=[lib newFunctionWithName:name];
        iq3_layout_pipeline[v][kind]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!iq3_layout_pipeline[v][kind] || iq3_layout_pipeline[v][kind].maxTotalThreadsPerThreadgroup<64) {
            fprintf(stderr,"IQ3 layout compile variant=%d kind=%d: %s\n",v,kind,error.localizedDescription.UTF8String);return 0;
        }
    }
    return 1;
}
static id<MTLBuffer> iq3_layout_pack(id<MTLBuffer> native,uint64_t offset,uint32_t ni,uint32_t no) {
    if (!ni || ni%256u || !no) return nil;
    const uint64_t blocks=(uint64_t)ni/256u*no,bytes=blocks*112u;
    if (!native || offset>native.length || blocks*110u>native.length-offset || bytes>NSUIntegerMax-64u) return nil;
    id<MTLBuffer> packed=[g_device newBufferWithLength:bytes+64u options:MTLResourceStorageModeShared];
    if (!packed) return nil;
    const unsigned char *src=(const unsigned char *)native.contents+offset;
    unsigned char *allocation=packed.contents,*dst=allocation+32u;
    memset(allocation,0xa5,32u);memset(dst+bytes,0xa5,32u);
    for (uint64_t b=0;b<blocks;b++) {
        const unsigned char *s=src+b*110u;unsigned char *d=dst+b*112u;
        memcpy(d,s,2u);memset(d+2u,0,2u);memcpy(d+4u,s+106u,4u);
        memcpy(d+8u,s+66u,8u);memcpy(d+16u,s+2u,64u);memcpy(d+80u,s+74u,32u);
    }
    for (uint64_t b=0;b<blocks;b++) {
        const unsigned char *s=src+b*110u,*d=dst+b*112u;
        if (memcmp(d,s,2u) || d[2] || d[3] || memcmp(d+4u,s+106u,4u) ||
            memcmp(d+8u,s+66u,8u) || memcmp(d+16u,s+2u,64u) || memcmp(d+80u,s+74u,32u)) return nil;
    }
    return packed;
}
static int iq3_layout_guards(id<MTLBuffer> buffer) {
    const unsigned char *p=buffer.contents;
    if (!p || buffer.length<64u) return 0;
    for (unsigned i=0;i<32u;i++) if (p[i]!=0xa5 || p[buffer.length-32u+i]!=0xa5) return 0;
    return 1;
}
static int iq3_layout_prepare(qw3_engine *engine) {
    if (iq3_layout_weights) return 1;
    if (g_model_view_count!=1u) return 0;
    iq3_layout_weights=[NSMutableDictionary new];unsigned count=0;
    const double start=[NSDate timeIntervalSinceReferenceDate];
    for (uint32_t i=0;i<engine->model.n_tensors;i++) {
        const qw3_tensor *w=&engine->model.tensors[i];if (w->type!=21u) continue;
        if (w->ndim!=2u || w->dim[0]%256u || w->dim[0]>UINT32_MAX || w->dim[1]>UINT32_MAX) return 0;
        uint64_t offset=0;id<MTLBuffer> native=qw3_metal_model_view_for(w->offset,w->elements/256u*110u,&offset);
        id<MTLBuffer> packed=iq3_layout_pack(native,offset,(uint32_t)w->dim[0],(uint32_t)w->dim[1]);
        if (!packed || iq3_layout_weights[@(offset)] || !iq3_layout_guards(packed)) return 0;
        iq3_layout_weights[@(offset)]=packed;count++;iq3_layout_bytes+=packed.length;
    }
    fprintf(stderr,"IQ3 layout prepared tensors=%u bytes=%llu seconds=%.6f all_source_bytes=exact\n",count,(unsigned long long)iq3_layout_bytes,[NSDate timeIntervalSinceReferenceDate]-start);
    return count>0;
}
static id<MTLBuffer> iq3_layout_bind(id<MTLBuffer> buffer,NSUInteger *offset,NSUInteger index,unsigned active) {
    (void)index;(void)active;
    /* Matrix tests already bind explicit packed copies. Native model bindings
     * are redirected only while a private candidate pipeline is selected. */
    if (g_model_view_count!=1u || buffer!=g_model_buffers[0]) return buffer;
    id<MTLBuffer> packed=iq3_layout_weights[@(*offset)];
    if (!packed) {fprintf(stderr,"IQ3 layout missing packed tensor offset=%llu\n",(unsigned long long)*offset);abort();}
    *offset=32u;iq3_layout_redirects++;return packed;
}
static void scratch_select(int mode) {
    setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1);
    g_matvec_iq3_s_pipeline=iq3_layout_pipeline[mode ? layout_variant : 0][0];
    g_matvec_iq3_s_pair_pipeline=iq3_layout_pipeline[mode ? layout_variant : 0][1];
}
