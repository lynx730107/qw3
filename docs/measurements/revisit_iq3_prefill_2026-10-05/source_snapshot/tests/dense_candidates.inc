#define QW3_PRIVATE_ORDINARY_PREFILL_BASELINE 1
/* Private candidates only; normal CLI/agent/engine sources are unchanged. */
#define qw3_metal_session_matvec_iq4_xs_pair_x1_to_scratch probe_base_iq4_pair
#define qw3_metal_session_matvec_iq4_xs_inner_add_x0 probe_base_iq4_add
#define qw3_metal_session_batch_deltanet_fused_gdn_from_scratch probe_base_gdn
#include "../qw3_metal.m"
#undef qw3_metal_session_matvec_iq4_xs_pair_x1_to_scratch
#undef qw3_metal_session_matvec_iq4_xs_inner_add_x0
#undef qw3_metal_session_batch_deltanet_fused_gdn_from_scratch

static int probe_ffn, probe_gdn;
static uint64_t probe_pair_calls, probe_add_calls, probe_gdn_calls;
static id<MTLComputePipelineState> probe_iq4_pair, probe_iq4_add;
static id<MTLComputePipelineState> probe_gdn_recur, probe_gdn_gate;

static int probe_replace_kernel(NSMutableString *source, NSString *name,
                                NSString *old, NSString *replacement) {
    NSRange start=[source rangeOfString:[@"kernel void " stringByAppendingString:name]];
    if (start.location==NSNotFound) return 0;
    NSRange end=[source rangeOfString:@"\n}\n" options:0
                                range:NSMakeRange(start.location,source.length-start.location)];
    if (end.location==NSNotFound) return 0;
    NSRange range=NSMakeRange(start.location,end.location+end.length-start.location);
    NSMutableString *body=[[source substringWithRange:range] mutableCopy];
    NSUInteger count=[body replaceOccurrencesOfString:old withString:replacement
                             options:0 range:NSMakeRange(0,body.length)];
    if (count!=1) {fprintf(stderr,"candidate replacement count=%lu expected=1\n",(unsigned long)count);return 0;}
    [source replaceCharactersInRange:range withString:body];
    return 1;
}

static int probe_initialize(void) {
    if (probe_iq4_pair) return 1;
    if (!qw3_metal_compile_kernels()) return 0;
    NSMutableString *source=[qw3_metal_kernel_source() mutableCopy];
    if (!probe_replace_kernel(source,@"qw3_matvec_iq4_xs_pair_fast",@"const uint nsg = 2u;",@"const uint nsg = 8u;") ||
        !probe_replace_kernel(source,@"qw3_matvec_iq4_xs_add_x0_fast",@"const uint nsg = 2u;",@"const uint nsg = 8u;")) return 0;
    NSString *gates=@"float beta_raw = scratch[base + args.beta_offset + hv];\n"
        "            b = 1.0f / (1.0f + exp(-beta_raw));\n"
        "            float alpha_raw = scratch[base + args.alpha_offset + hv] + dt_hv;\n"
        "            float sp = alpha_raw > 20.0f ? alpha_raw : (alpha_raw < -20.0f ? exp(alpha_raw) : log(1.0f + exp(alpha_raw)));\n"
        "            g = exp(sp * a_hv);";
    if (!probe_replace_kernel(source,@"qw3_deltanet_batch_recur_tiled2",gates,
        @"b = scratch[base + args.beta_offset + hv];\n            g = scratch[base + args.alpha_offset + hv];")) return 0;
    [source appendString:@"\nkernel void probe_batch_gates(constant qw3_batch_gdn_args &args,\n"
        "    device float *scratch, device const float *dt_bias, device const float *a, uint index [[thread_position_in_grid]]) {\n"
        "    if (index >= args.n_tokens * args.v_heads) return;\n"
        "    uint hv = index % args.v_heads;\n"
        "    uint64_t base = uint64_t(index / args.v_heads) * args.stride;\n"
        "    float beta_raw = scratch[base + args.beta_offset + hv];\n"
        "    float b = 1.0f / (1.0f + exp(-beta_raw));\n"
        "    float alpha_raw = scratch[base + args.alpha_offset + hv] + dt_bias[hv];\n"
        "    float sp = alpha_raw > 20.0f ? alpha_raw : (alpha_raw < -20.0f ? exp(alpha_raw) : log(1.0f + exp(alpha_raw)));\n"
        "    float g = exp(sp * a[hv]);\n"
        "    scratch[base + args.beta_offset + hv] = b;\n"
        "    scratch[base + args.alpha_offset + hv] = g;\n}\n"];
    NSError *error=nil;
    MTLCompileOptions *options=[MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) options.preprocessorMacros=@{@"QW3_METAL_HAS_TENSOR":@"1"};
    id<MTLLibrary> library=[g_device newLibraryWithSource:source options:options error:&error];
    for (int i=0;i<4;i++) {
        NSString *names[] = {@"qw3_matvec_iq4_xs_pair_fast",@"qw3_matvec_iq4_xs_add_x0_fast",@"qw3_deltanet_batch_recur_tiled2",@"probe_batch_gates"};
        id<MTLFunction> fn=[library newFunctionWithName:names[i]];
        id<MTLComputePipelineState> pipeline=fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!pipeline || (i<2 && pipeline.maxTotalThreadsPerThreadgroup<256)) {
            fprintf(stderr,"candidate compile failed: %s\n",error.localizedDescription.UTF8String);return 0;
        }
        if (i==0) probe_iq4_pair=pipeline;
        else if (i==1) probe_iq4_add=pipeline;
        else if (i==2) probe_gdn_recur=pipeline;
        else probe_gdn_gate=pipeline;
    }
    return 1;
}

int qw3_metal_session_batch_deltanet_fused_gdn_from_scratch(
    qw3_metal_session *s, uint64_t dt_bias_offset, uint64_t a_offset,
    uint64_t norm_weight_offset, uint32_t layer_slot, uint32_t n_tokens,
    uint32_t conv_offset, uint32_t z_offset, uint32_t alpha_offset,
    uint32_t beta_offset, uint32_t inner_offset, uint32_t stride,
    uint32_t q_heads, uint32_t v_heads, uint32_t head_dim, float eps) {
    const int enabled=probe_gdn && head_dim==128u && n_tokens>=16u;
    if (enabled) {
        if (!s || !s->obj || !q_heads || !v_heads || alpha_offset>stride || beta_offset>stride ||
            v_heads>stride-alpha_offset || v_heads>stride-beta_offset) return 0;
        QW3MetalSessionObj *obj=(__bridge QW3MetalSessionObj *)s->obj;
        if (obj.prefillScratch.length<(uint64_t)n_tokens*stride*sizeof(float)) return 0;
        uint64_t dt_inner=0,a_inner=0;
        id<MTLBuffer> dt=qw3_metal_model_view_for(dt_bias_offset,v_heads*sizeof(float),&dt_inner);
        id<MTLBuffer> a=qw3_metal_model_view_for(a_offset,v_heads*sizeof(float),&a_inner);
        if (!dt || !a) return 0;
        struct {uint32_t q_heads,v_heads,head_dim,n_tokens,conv_offset,z_offset,alpha_offset,beta_offset,inner_offset,stride;float eps;}
            args={q_heads,v_heads,head_dim,n_tokens,conv_offset,z_offset,alpha_offset,beta_offset,inner_offset,stride,eps};
        int owned=0;
        id<MTLCommandBuffer> cb=qw3_metal_command_buffer(&owned);
        id<MTLComputeCommandEncoder> enc=qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:probe_gdn_gate];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:obj.prefillScratch offset:0 atIndex:1];
        [enc setBuffer:dt offset:dt_inner atIndex:2];
        [enc setBuffer:a offset:a_inner atIndex:3];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n_tokens*v_heads,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        qw3_metal_end_compute_encoder(cb,enc);
        if (!qw3_metal_finish_command_buffer(cb,owned,"private DeltaNet gates")) return 0;
        probe_gdn_calls++;
    }
    id<MTLComputePipelineState> base=g_deltanet_batch_recur_tiled2_pipeline;
    if (enabled) g_deltanet_batch_recur_tiled2_pipeline=probe_gdn_recur;
    int ok=probe_base_gdn(s,dt_bias_offset,a_offset,norm_weight_offset,layer_slot,n_tokens,
        conv_offset,z_offset,alpha_offset,beta_offset,inner_offset,stride,q_heads,v_heads,head_dim,eps);
    g_deltanet_batch_recur_tiled2_pipeline=base;
    return ok;
}

/* IQ4 wrapper implementations follow, keeping normal validation/bindings. */

int qw3_metal_session_matvec_iq4_xs_pair_x1_to_scratch(
    qw3_metal_session *s, uint64_t gate_offset, uint64_t up_offset,
    uint32_t n_in, uint32_t n_out) {
    if (!probe_ffn) return probe_base_iq4_pair(s, gate_offset, up_offset, n_in, n_out);
    probe_pair_calls++;
    if (!s || !s->obj || !n_in || !n_out || (n_in % 256u) != 0u)
        return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;
    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_in * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_out * 2u * sizeof(float);
    const uint64_t row_bytes = (uint64_t)(n_in / 256u) * 136ull;
    const uint64_t weight_bytes = row_bytes * n_out;
    if (!obj.x1 || obj.x1.length < x_bytes ||
        !obj.scratch || obj.scratch.length < out_bytes ||
        row_bytes > UINT32_MAX || !g_matvec_iq4_xs_pair_pipeline)
        return 0;
    uint64_t gate_inner = 0, up_inner = 0;
    id<MTLBuffer> gate = qw3_metal_model_view_for(
        gate_offset, weight_bytes, &gate_inner);
    id<MTLBuffer> up = qw3_metal_model_view_for(
        up_offset, weight_bytes, &up_inner);
    if (!gate || !up) return 0;
    struct { uint32_t n_in, n_out, row_bytes; } args = {
        n_in, n_out, (uint32_t)row_bytes
    };
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    if (!cb) return 0;
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:probe_iq4_pair];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:gate offset:(NSUInteger)gate_inner atIndex:1];
    [enc setBuffer:up offset:(NSUInteger)up_inner atIndex:2];
    [enc setBuffer:obj.x1 offset:0 atIndex:3];
    [enc setBuffer:obj.scratch offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((n_out + 15u) / 16u, 1u, 1u)
         threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    qw3_metal_end_compute_encoder(cb, enc);
    return qw3_metal_finish_command_buffer(cb, owned,
                                            "dense IQ4_XS gate/up pair");
}

int qw3_metal_session_matvec_iq4_xs_inner_add_x0(
    qw3_metal_session *s, uint64_t tensor_offset,
    uint32_t n_in, uint32_t n_out) {
    if (!probe_ffn) return probe_base_iq4_add(s, tensor_offset, n_in, n_out);
    probe_add_calls++;
    if (!s || !s->obj || !n_in || !n_out || (n_in % 256u) != 0u)
        return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;
    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t in_bytes = (uint64_t)n_in * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_out * sizeof(float);
    const uint64_t row_bytes = (uint64_t)(n_in / 256u) * 136ull;
    const uint64_t tensor_bytes = row_bytes * n_out;
    if (!obj.inner || obj.inner.length < in_bytes ||
        !obj.x0 || obj.x0.length < out_bytes || row_bytes > UINT32_MAX)
        return 0;
    uint64_t model_offset = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(
        tensor_offset, tensor_bytes, &model_offset);
    if (!wb) return 0;
    struct { uint32_t n_in, n_out, row_bytes; } args = {
        n_in, n_out, (uint32_t)row_bytes
    };
    const float scale = 1.0f;
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    if (!cb) return 0;
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:probe_iq4_add];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBytes:&scale length:sizeof(scale) atIndex:1];
    [enc setBuffer:wb offset:(NSUInteger)model_offset atIndex:2];
    [enc setBuffer:obj.inner offset:0 atIndex:3];
    [enc setBuffer:obj.x0 offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((n_out + 15u) / 16u, 1u, 1u)
         threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "dense IQ4_XS down add"))
        return 0;
    return cb.status != MTLCommandBufferStatusError;
}

#define qw3_metal_env_n_gpu_layers qw3_probe_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled qw3_probe_core_llama_split_enabled
#define qw3_metal_llama_split_start qw3_probe_core_llama_split_start
#define qw3_engine_open probe_engine_open_base
#include "../qw3.c"
#undef qw3_engine_open

static void probe_select(int mode) { probe_ffn=mode==1; probe_gdn=mode==2; }
int qw3_engine_open(qw3_engine **engine, const qw3_engine_options *options) {
    int rc=probe_engine_open_base(engine,options);
    if (rc) return rc;
    if (!probe_initialize()) {qw3_engine_close(*engine);*engine=NULL;return -1;}
    const int f=getenv("QW3_PRIVATE_FFN_SG8")!=NULL;
    const int d=getenv("QW3_PRIVATE_GDN_GATES")!=NULL;
    if (f && d) {qw3_engine_close(*engine);*engine=NULL;return -1;}
    probe_select(f ? 1 : d ? 2 : 0);
    fprintf(stderr,"private candidate ffn_sg8=%d gdn_gates=%d\n",f,d);
    return 0;
}
