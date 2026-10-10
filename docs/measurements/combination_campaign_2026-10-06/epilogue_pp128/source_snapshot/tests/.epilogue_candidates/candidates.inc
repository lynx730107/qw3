#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
@class QW3MetalSessionObj;
static int epilogue_saved;
static unsigned long long epilogue_calls;
static int epilogue_down(QW3MetalSessionObj *, id<MTLCommandBuffer>, uint64_t,
    uint32_t,uint32_t,uint32_t,uint32_t,id<MTLBuffer>,id<MTLBuffer>);
/* Private graph; both sides retain the native packed8/direct-scratch default. */
static int combo_mode, combo_round_ready;
static unsigned long long combo_residual_calls, combo_round_calls, combo_round_saved;
#define qw3_metal_session_matvec_q5_k_x1_to_logits sum8_base_logits
#include "bridge.inc"
#undef qw3_metal_session_matvec_q5_k_x1_to_logits

static id<MTLComputePipelineState> combo_norm_pipeline;
static void scratch_select(int mode) {
    combo_mode=mode; combo_round_ready=0;
    setenv("QW3_METAL_DENSE_PREFILL_FAST","1",1);
}
static int combo_compile(void) {
    if (combo_norm_pipeline) return 1;
    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;
    NSError *error=nil;
    NSString *extra=[NSString stringWithContentsOfFile:@"tests/combination_candidates.metal" encoding:NSUTF8StringEncoding error:&error];
    if (!extra) return 0;
    NSMutableString *source=[qw3_metal_kernel_source() mutableCopy]; [source appendString:extra];
    MTLCompileOptions *opt=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) opt.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    id<MTLLibrary> lib=[g_device newLibraryWithSource:source options:opt error:&error];
    if (lib) combo_norm_pipeline=[g_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"combo_residual_norm_round"] error:&error];
    if (!combo_norm_pipeline) fprintf(stderr,"combo compile: %s\n",error.localizedDescription.UTF8String);
    return combo_norm_pipeline != nil;
}
static int combo_output_add_layout(qw3_metal_session *s, uint64_t weight, uint32_t type,
    uint32_t nt, uint32_t ni, uint32_t ne, uint32_t offset, uint32_t stride) {
    if (!(combo_mode&1)) return qw3_metal_session_batch_dense_gqa_output_add_layout(s,weight,type,nt,ni,ne,offset,stride);
    if (!s || !s->obj || !nt || !ni || !ne || !qw3_metal_dense_quant_batch_available(type,nt)) return 0;
    QW3MetalSessionObj *obj=(__bridge QW3MetalSessionObj *)s->obj;
    uint64_t ib=(uint64_t)nt*ni*4ull,ob=(uint64_t)nt*ne*4ull;
    if ((!stride && (!obj.prefillX1 || obj.prefillX1.length<ib)) || !obj.prefillX0 || obj.prefillX0.length<ob) return 0;
    if (!obj.denseBatchDown || obj.denseBatchDown.length<ob) obj.denseBatchDown=qw3_metal_new_private_buffer(ob);
    if (!obj.denseBatchDown) return 0;
    int owned=0; id<MTLCommandBuffer> cb=qw3_metal_command_buffer(&owned);
    return cb && qw3_metal_encode_dense_batch_matmul_strided(obj,cb,weight,type,nt,ni,ne,
        stride ? obj.prefillScratch : obj.prefillX1,obj.denseBatchDown,
        stride ? offset : 0,stride ? stride : ni,0,ne) &&
        qw3_metal_finish_command_buffer(cb,owned,"combo attention projection");
}
static int combo_needs_round(uint32_t nt, uint32_t gt, uint32_t ut) {
    const char *env=getenv("QW3_METAL_DENSE_PREFILL_IQ3_PAIR");
    const int pair=env ? strcmp(env,"0")!=0 :
        !(nt%128u==0u && g_matmul_iq3_s_mpp_staged128_pipeline &&
          g_round_f16_inplace_pipeline && qw3_metal_staged128_enabled());
    return gt==21u && ut==21u && g_moe_iq3_s_prefill_pair_mapped_mpp_pipeline &&
        !pair && g_round_f16_inplace_pipeline;
}
static int combo_ffn_norm(qw3_metal_session *s, uint64_t weight, uint32_t nt,
    uint32_t n, float eps, uint32_t gt, uint32_t ut) {
    int round=(combo_mode&2) && combo_needs_round(nt,gt,ut);
    uint32_t flags=(combo_mode&1) | (round ? 2u : 0u);
    combo_round_ready=0;
    if (!flags) return qw3_metal_session_batch_rmsnorm_weight_f32_x0_to_x1(s,weight,nt,n,eps);
    if (!s || !s->obj || !nt || !n || !combo_compile()) return 0;
    QW3MetalSessionObj *obj=(__bridge QW3MetalSessionObj *)s->obj;
    uint64_t bytes=(uint64_t)nt*n*4ull,inner=0;
    if (!qw3_metal_session_ensure_prefill_buffers(obj,nt,bytes,bytes) ||
        ((flags&1) && (!obj.denseBatchDown || obj.denseBatchDown.length<bytes))) return 0;
    id<MTLBuffer> wb=qw3_metal_model_view_for(weight,(uint64_t)n*4ull,&inner);
    if (!wb) return 0;
    struct { uint32_t n; float eps; uint32_t rows,flags; } args={n,eps,nt,flags};
    int owned=0; id<MTLCommandBuffer> cb=qw3_metal_command_buffer(&owned);
    if (!cb) return 0;
    id<MTLComputeCommandEncoder> enc=qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:combo_norm_pipeline]; [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.prefillX0 offset:0 atIndex:1];
    [enc setBuffer:(flags&1) ? obj.denseBatchDown : obj.prefillX0 offset:0 atIndex:2];
    [enc setBuffer:wb offset:inner atIndex:3]; [enc setBuffer:obj.prefillX1 offset:0 atIndex:4];
    NSUInteger threads=MIN((NSUInteger)256,g_rmsnorm_weight_f32_rows_to_out_pipeline.maxTotalThreadsPerThreadgroup);
    if (threads>combo_norm_pipeline.maxTotalThreadsPerThreadgroup || threads%32u) return 0;
    [enc setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(nt,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    qw3_metal_end_compute_encoder(cb,enc);
    if (flags&1) combo_residual_calls++;
    if (round) { combo_round_calls++; combo_round_ready=1; }
    return qw3_metal_finish_command_buffer(cb,owned,"combo FFN norm");
}
#include "../sum8_dispatch.inc"
#include "../epilogue_dispatch.inc"
#define qw3_metal_env_n_gpu_layers combo_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled combo_core_llama_split_enabled
#define qw3_metal_llama_split_start combo_core_llama_split_start
#include "../.combination_candidates/core.inc"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
