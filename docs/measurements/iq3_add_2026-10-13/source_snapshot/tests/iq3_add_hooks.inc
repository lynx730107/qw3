/* Private FFN down/residual fusion. No extra model/session data buffers. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static id<MTLComputePipelineState> iq3_add_pipeline[3];
static int iq3_add_mode,add_variant=1;
static uint64_t iq3_add_calls;
#include "../qw3_metal.m"

static int iq3_add_compile(void) {
    if (iq3_add_pipeline[2]) return 1;
    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;
    iq3_add_pipeline[0]=g_matvec_iq3_s_pipeline;
    NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/.iq3_add/kernels.metal" encoding:NSUTF8StringEncoding error:&error];
    NSString *original=qw3_metal_full_kernel_source();
    NSString *source=extra && original ? [original stringByAppendingString:extra] : nil;
    id<MTLLibrary> lib=source ? [g_device newLibraryWithSource:source options:options error:&error] : nil;
    for (int v=1;v<=2;v++) {
        id<MTLFunction> fn=[lib newFunctionWithName:[NSString stringWithFormat:@"private_iq3_add_fast%d",v]];
        id<MTLComputePipelineState> p=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!p || p.threadExecutionWidth!=32 || p.maxTotalThreadsPerThreadgroup<64 ||
            p.staticThreadgroupMemoryLength!=iq3_add_pipeline[0].staticThreadgroupMemoryLength) {
            fprintf(stderr,"IQ3 add compile v=%d: %s\n",v,error.localizedDescription.UTF8String);return 0;
        }
        iq3_add_pipeline[v]=p;
    }
    for (int v=0;v<=2;v++) fprintf(stderr,"iq3_add_metadata variant=%d width=%lu max_threads=%lu static_threadgroup_bytes=%lu dynamic_threadgroup_bytes=128 threads=64\n",v,(unsigned long)iq3_add_pipeline[v].threadExecutionWidth,(unsigned long)iq3_add_pipeline[v].maxTotalThreadsPerThreadgroup,(unsigned long)iq3_add_pipeline[v].staticThreadgroupMemoryLength);
    return 1;
}
static int private_iq3_inner_add_x0(qw3_metal_session *s,uint64_t tensor_offset,uint32_t ni,uint32_t no) {
    if (!s || !s->obj || !ni || !no || ni%256u || !iq3_add_compile()) return 0;
    QW3MetalSessionObj *obj=(__bridge QW3MetalSessionObj *)s->obj;
    uint64_t rb=(uint64_t)(ni/256u)*110u,off=0;
    if (rb>UINT32_MAX || !obj.inner || !obj.x0 ||
        obj.inner.length<(uint64_t)ni*4u || obj.x0.length<(uint64_t)no*4u) return 0;
    id<MTLBuffer> w=qw3_metal_model_view_for(tensor_offset,rb*no,&off);
    id<MTLBuffer> grid=qw3_metal_iq3s_kgrid_buffer();
    if (!w || !grid) return 0;
    struct {uint32_t ni,no,rb;} args={ni,no,(uint32_t)rb};const float scale=1.0f;
    int owned=0;id<MTLCommandBuffer> cb=qw3_metal_command_buffer(&owned);if (!cb) return 0;
    id<MTLComputeCommandEncoder> enc=qw3_metal_compute_encoder(cb);if (!enc) return 0;
    [enc setComputePipelineState:iq3_add_pipeline[add_variant]];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:w offset:off atIndex:1];[enc setBuffer:obj.inner offset:0 atIndex:2];
    [enc setBuffer:obj.x0 offset:0 atIndex:3];[enc setBuffer:grid offset:0 atIndex:4];
    [enc setBytes:&scale length:sizeof(scale) atIndex:5];
    [enc setThreadgroupMemoryLength:128u atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((no+7u)/8u,1,1) threadsPerThreadgroup:MTLSizeMake(64u,1,1)];
    iq3_add_calls++;
    qw3_metal_end_compute_encoder(cb,enc);
    return qw3_metal_finish_command_buffer(cb,owned,"private IQ3 residual") && cb.status!=MTLCommandBufferStatusError;
}
static __attribute__((unused)) void scratch_select(int mode) { iq3_add_mode=mode!=0; }
#define qw3_metal_env_n_gpu_layers iq3_add_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled iq3_add_core_llama_split_enabled
#define qw3_metal_llama_split_start iq3_add_core_llama_split_start
#include ".iq3_add/core.inc"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
