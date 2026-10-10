/* Optional diagnostic binary only. Production engine/shaders are untouched. */
#define qw3_metal_session_batch_deltanet_fused_gdn_from_scratch probe_base_gdn
#include "../qw3_metal.m"
#undef qw3_metal_session_batch_deltanet_fused_gdn_from_scratch
static int probe_gdn;
static uint64_t probe_gdn_calls;
static id<MTLComputePipelineState> probe_gdn_pipelines[3];
static int probe_initialize(void) {
    if (probe_gdn_pipelines[0]) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    NSString *path=@"tests/gdn_prefill_candidates.metal";
    NSError *error=nil;
    NSString *extra=[NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
    if (!extra) { fprintf(stderr,"private GDN shader missing: %s\n",error.localizedDescription.UTF8String); return 0; }
    NSMutableString *source=[qw3_metal_kernel_source() mutableCopy]; [source appendString:extra];
    MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    id<MTLLibrary> library=[g_device newLibraryWithSource:source options:options error:&error];
    NSString *names[]={@"probe_gdn_packed4",@"probe_gdn_packed8",@"probe_gdn_pipeline"};
    for (int i=0;i<3;i++) {
        id<MTLFunction> f=[library newFunctionWithName:names[i]];
        probe_gdn_pipelines[i]=f ? [g_device newComputePipelineStateWithFunction:f error:&error] : nil;
        if (!probe_gdn_pipelines[i] || probe_gdn_pipelines[i].maxTotalThreadsPerThreadgroup<128) {
            fprintf(stderr,"private GDN compile failed: %s\n",error.localizedDescription.UTF8String); return 0;
        }
    }
    return 1;
}
int qw3_metal_session_batch_deltanet_fused_gdn_from_scratch(
    qw3_metal_session *s, uint64_t dt_bias_offset, uint64_t a_offset,
    uint64_t norm_weight_offset, uint32_t layer_slot, uint32_t n_tokens,
    uint32_t conv_offset, uint32_t z_offset, uint32_t alpha_offset,
    uint32_t beta_offset, uint32_t inner_offset, uint32_t stride,
    uint32_t q_heads, uint32_t v_heads, uint32_t head_dim, float eps) {
    if (!probe_gdn || head_dim != 128u || n_tokens < 16u)
        return probe_base_gdn(s,dt_bias_offset,a_offset,norm_weight_offset,layer_slot,n_tokens,
            conv_offset,z_offset,alpha_offset,beta_offset,inner_offset,stride,q_heads,v_heads,head_dim,eps);
    probe_gdn_calls++;
    if (!s || !s->obj || n_tokens == 0 || stride == 0 ||
        q_heads == 0 || v_heads == 0 || head_dim == 0 ||
        !qw3_metal_session_linear_slot_valid(s, layer_slot) ||
        head_dim > 256) {
        return 0;
    }
    const uint32_t qk_n = q_heads * head_dim;
    const uint32_t inner_n = v_heads * head_dim;
    if (conv_offset > stride || z_offset > stride ||
        alpha_offset > stride || beta_offset > stride ||
        inner_offset > stride ||
        2u * qk_n + inner_n > stride - conv_offset ||
        inner_n > stride - z_offset ||
        v_heads > stride - alpha_offset ||
        v_heads > stride - beta_offset ||
        inner_n > stride - inner_offset) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t scratch_bytes =
        (uint64_t)n_tokens * stride * sizeof(float);
    const uint64_t state_bytes =
        (uint64_t)v_heads * head_dim * head_dim * sizeof(float);
    const uint64_t state_offset = (uint64_t)layer_slot * state_bytes;
    if (!obj.prefillScratch || obj.prefillScratch.length < scratch_bytes ||
        !obj.deltanetState || obj.deltanetState.length < state_offset ||
        obj.deltanetState.length - state_offset < state_bytes) {
        return 0;
    }

    const uint64_t gates_bytes = (uint64_t)v_heads * sizeof(float);
    const uint64_t weight_bytes = (uint64_t)head_dim * sizeof(float);
    uint64_t dt_inner = 0, a_inner = 0, w_inner = 0;
    id<MTLBuffer> dtb =
        qw3_metal_model_view_for(dt_bias_offset, gates_bytes, &dt_inner);
    id<MTLBuffer> ab =
        qw3_metal_model_view_for(a_offset, gates_bytes, &a_inner);
    id<MTLBuffer> wb =
        qw3_metal_model_view_for(norm_weight_offset, weight_bytes, &w_inner);
    if (!dtb || !ab || !wb ||
        head_dim > g_deltanet_batch_fused_gdn_pipeline.maxTotalThreadsPerThreadgroup) {
        fprintf(stderr, "qw3: Metal batch Gated DeltaNet weights are outside mapped model views\n");
        return 0;
    }

    struct {
        uint32_t q_heads;
        uint32_t v_heads;
        uint32_t head_dim;
        uint32_t n_tokens;
        uint32_t conv_offset;
        uint32_t z_offset;
        uint32_t alpha_offset;
        uint32_t beta_offset;
        uint32_t inner_offset;
        uint32_t stride;
        float eps;
    } args = {
        q_heads, v_heads, head_dim, n_tokens, conv_offset, z_offset,
        alpha_offset, beta_offset, inner_offset, stride, eps
    };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    {
        [enc setComputePipelineState:probe_gdn_pipelines[probe_gdn-1]];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:obj.deltanetState offset:(NSUInteger)state_offset atIndex:1];
        [enc setBuffer:obj.prefillScratch offset:0 atIndex:2];
        [enc setBuffer:dtb offset:(NSUInteger)dt_inner atIndex:3];
        [enc setBuffer:ab offset:(NSUInteger)a_inner atIndex:4];
        uint columns = probe_gdn == 1 ? 32u : probe_gdn == 2 ? 16u : 8u;
        [enc dispatchThreadgroups:MTLSizeMake((head_dim+columns-1u)/columns,v_heads,1)
            threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:g_deltanet_batch_gated_rmsnorm_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:obj.prefillScratch offset:0 atIndex:1];
        [enc setBuffer:wb offset:(NSUInteger)w_inner atIndex:2];
        [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
        NSUInteger norm_threads =
            g_deltanet_batch_gated_rmsnorm_pipeline.maxTotalThreadsPerThreadgroup;
        if (norm_threads > 256) norm_threads = 256;
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens * v_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(norm_threads, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);
        if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
        if (cb.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "qw3: Metal batch tiled Gated DeltaNet command failed: %s\n",
                    [[cb.error localizedDescription] UTF8String]);
            return 0;
        }
        return 1;
    }

}

#define qw3_metal_env_n_gpu_layers qw3_probe_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled qw3_probe_core_llama_split_enabled
#define qw3_metal_llama_split_start qw3_probe_core_llama_split_start
#define qw3_engine_open probe_engine_open_base
#include "../qw3.c"
#undef qw3_engine_open
static void probe_select(int mode) { probe_gdn=mode; }
int qw3_engine_open(qw3_engine **engine,const qw3_engine_options *options) {
    int rc=probe_engine_open_base(engine,options); if (rc) return rc;
    if (!probe_initialize()) { qw3_engine_close(*engine); *engine=NULL; return -1; }
    const char *mode=getenv("QW3_PRIVATE_GDN_LAYOUT");
    probe_select(!mode ? 0 : !strcmp(mode,"packed4") ? 1 : !strcmp(mode,"packed8") ? 2 : !strcmp(mode,"pipeline") ? 3 : -1);
    if (probe_gdn<0) { qw3_engine_close(*engine); *engine=NULL; return -1; }
    fprintf(stderr,"private GDN layout=%s\n",mode ? mode : "off"); return 0;
}
