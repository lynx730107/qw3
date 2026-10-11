/* Private integer-load variants. Preserve all native FP expressions. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> iq4_load_pipeline[4][3];
static uint64_t iq4_load_calls[3];
static int iq4_variant=1;
@protocol QW3IQ4LoadHooks
- (void)iq4LoadSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (id)iq4LoadEncoder;
- (id)iq4LoadEncoderWithDispatchType:(MTLDispatchType)type;
- (id)iq4LoadCommandBuffer;
- (id)iq4LoadCommandBufferUnretained;
- (id)iq4LoadCommandQueue;
@end
@interface NSObject (QW3IQ4LoadProbe) <QW3IQ4LoadHooks>
@end
@implementation NSObject (QW3IQ4LoadProbe)
- (id)iq4LoadCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)iq4LoadCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)iq4LoadCommandBufferUnretained {
    return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences];
}
- (id)iq4LoadEncoder { return [(id<MTLCommandBuffer>)self computeCommandEncoder]; }
- (id)iq4LoadEncoderWithDispatchType:(MTLDispatchType)type {
    return [(id<MTLCommandBuffer>)self computeCommandEncoderWithDispatchType:type];
}
- (void)iq4LoadSetPipeline:(id<MTLComputePipelineState>)pipeline {
    for (int kind=0;kind<3;kind++) for (int variant=1;variant<=3;variant++)
        if (pipeline==iq4_load_pipeline[variant][kind]) iq4_load_calls[kind]++;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:pipeline];
}
@end
static id iq4_load_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id iq4_load_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice iq4_load_default_device
#define MTLCopyAllDevices iq4_load_all_devices
#define MTLDevice MTLDevice,QW3IQ4LoadHooks
#define MTLCommandQueue MTLCommandQueue,QW3IQ4LoadHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3IQ4LoadHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3IQ4LoadHooks
#define newCommandQueue iq4LoadCommandQueue
#define commandBuffer iq4LoadCommandBuffer
#define commandBufferWithUnretainedReferences iq4LoadCommandBufferUnretained
#define setComputePipelineState iq4LoadSetPipeline
#define computeCommandEncoder iq4LoadEncoder
#define computeCommandEncoderWithDispatchType iq4LoadEncoderWithDispatchType
#include "../qw3_metal.m"
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
#define qw3_metal_env_n_gpu_layers iq4_load_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq4_load_core_llama_split_enabled
#define qw3_metal_llama_split_start iq4_load_core_llama_split_start
#include "../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static NSString *iq4_load_extract(NSString *s,NSString *marker,NSString *last) {
    NSRange first=[s rangeOfString:marker];if (first.location==NSNotFound) return nil;
    NSRange end=[s rangeOfString:last options:0 range:NSMakeRange(first.location,s.length-first.location)];
    if (end.location==NSNotFound) return nil;
    return [s substringWithRange:NSMakeRange(first.location,end.location+end.length-first.location)];
}
static int iq4_load_replace(NSMutableString *s,NSString *old,NSString *replacement) {
    return [s replaceOccurrencesOfString:old withString:replacement options:0 range:NSMakeRange(0,s.length)]==1;
}
static int iq4_load_compile(void) {
    if (iq4_load_pipeline[3][2]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    iq4_load_pipeline[0][0]=g_matvec_iq4_xs_pipeline;
    iq4_load_pipeline[0][1]=g_matvec_iq4_xs_pair_pipeline;
    iq4_load_pipeline[0][2]=g_matvec_iq4_xs_add_x0_pipeline;
    NSString *all=qw3_metal_kernel_source();
    NSString *table=iq4_load_extract(all,@"constant float qw3_iq4nl_table[16]",@"\n};");
    NSString *value=iq4_load_extract(all,@"inline float qw3_iq4nl_val(",@"\n}\n");
    NSString *helper=iq4_load_extract(all,@"inline float qw3_iq4xs_dot8_vector(",@"\n}\n");
    const char *names[]={"qw3_matvec_iq4_xs_fast","qw3_matvec_iq4_xs_pair_fast","qw3_matvec_iq4_xs_add_x0_fast"};
    if (!table || !value || !helper) return 0;
    for (int v=1;v<=3;v++) {
        NSMutableString *dot=[helper mutableCopy];
        if (v&1) {
            if (!iq4_load_replace(dot,@"const half d = *((device const half *)blk);\n    const ushort scales_h = *((device const ushort *)(blk + 2));\n    device const uchar *scales_l = blk + 4;",
                @"const uint2 header = *((device const uint2 *)blk);\n    const half d = as_type<half2>(header.x).x;\n    const ushort scales_h = ushort(header.x >> 16u);") ||
                !iq4_load_replace(dot,@"(uint(scales_l[ib / 2u]) >> (4u * (ib & 1u)))",@"(header.y >> (4u * ib))")) return 0;
        }
        if (v&2) if (!iq4_load_replace(dot,@"device const uint *qs = (device const uint *)(blk + 8 + ib * 16u + il * 8u);",
            @"const uint2 qs = *((device const uint2 *)(blk + 8 + ib * 16u + il * 8u));")) return 0;
        NSMutableString *source=[@"#include <metal_stdlib>\nusing namespace metal;\nstruct qw3_matvec_q8_0_args {uint n_in,n_out,row_bytes;};\n" mutableCopy];
        [source appendString:table];[source appendString:@"\n"];[source appendString:value];[source appendString:dot];
        for (int k=0;k<3;k++) {
            NSString *body=iq4_load_extract(all,[@"kernel void " stringByAppendingString:[NSString stringWithUTF8String:names[k]]],@"\n}\n");
            if (!body) return 0;[source appendString:body];
        }
        NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
        id<MTLLibrary> lib=[g_device newLibraryWithSource:source options:options error:&error];
        for (int k=0;k<3;k++) {
            id<MTLFunction> fn=[lib newFunctionWithName:[NSString stringWithUTF8String:names[k]]];
            iq4_load_pipeline[v][k]=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
            if (!iq4_load_pipeline[v][k] || iq4_load_pipeline[v][k].maxTotalThreadsPerThreadgroup<64) {
                fprintf(stderr,"IQ4 load compile v=%d kind=%d: %s\n",v,k,error.localizedDescription.UTF8String);return 0;
            }
        }
    }
    /* IQ4 blocks (136 bytes), rows and payload addresses are 8-byte aligned. */
    for (uint32_t i=0;i<g_model_view_count;i++) if (((uintptr_t)g_model_view_ptrs[i])&7u) return 0;
    return 1;
}
static void scratch_select(int mode) {
    setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1);
    int v=mode ? iq4_variant : 0;
    g_matvec_iq4_xs_pipeline=iq4_load_pipeline[v][0];
    g_matvec_iq4_xs_pair_pipeline=iq4_load_pipeline[v][1];
    g_matvec_iq4_xs_add_x0_pipeline=iq4_load_pipeline[v][2];
}

static int iq4_load_model_aligned(qw3_engine *engine) {
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
