/* Private comparator: retain the compact baseline while reusing the
 * production strided dispatcher for candidate projections. */
#define QW3_PRIVATE_ORDINARY_PREFILL_BASELINE 1
#ifdef QW3_PRIVATE_WITH_GDN
#define qw3_metal_session_batch_deltanet_fused_gdn_from_scratch probe_base_gdn
#endif
#include "../qw3_metal.m"
#ifdef QW3_PRIVATE_WITH_GDN
#undef qw3_metal_session_batch_deltanet_fused_gdn_from_scratch
#include "gdn_prefill_dispatch.inc"
#endif

static int scratch_mode;
static uint64_t scratch_projection_calls, scratch_output_calls, scratch_copies_skipped;

static int scratch_matmul_available(uint32_t type, uint32_t n_tokens) {
    if (n_tokens < 16u) return 0;
    switch (type) {
        case 10: return g_matmul_q2_k_batch4_pipeline != nil;
        case 11: return g_matmul_q3_k_batch4_pipeline != nil;
        case 12: case 13: return g_matmul_q4_q5_k_mpp_pipeline != nil;
        case 14: return g_matmul_q6_k_mpp_pipeline != nil;
        case 16: return g_matmul_iq2_xxs_mpp_pipeline != nil;
        case 17: return g_matmul_iq2_xs_mpp_pipeline != nil;
        case 18: return g_matmul_iq3_xxs_batch4_pipeline != nil;
        case 20: return g_matmul_iq4_nl_mpp_pipeline != nil;
        case 21: return g_matmul_iq3_s_mpp_pipeline != nil;
        case 22: return g_matmul_iq2_s_batch4_pipeline != nil;
        case 23: return g_matmul_iq4_xs_mpp_pipeline != nil;
        default: return 0;
    }
}

static int scratch_projection_eligible(uint32_t n, int full,
                                       uint32_t a, uint32_t b, uint32_t c) {
    return (scratch_mode & 1) && full && scratch_matmul_available(a, n) &&
        scratch_matmul_available(b, n) && (!c || scratch_matmul_available(c, n));
}

static int scratch_output_eligible(uint32_t n, uint32_t type) {
    return (scratch_mode & 2) && scratch_matmul_available(type, n);
}

#include "scratch_matmul.inc"

static int scratch_batch_deltanet(qw3_metal_session *s,
    uint64_t qkv_weight, uint32_t qkv_type, uint64_t z_weight, uint32_t z_type,
    uint32_t n_tokens, uint32_t n_embd, uint32_t n_qkv, uint32_t n_z,
    uint32_t native_qkv_offset, uint32_t native_z_offset, uint32_t native_stride,
    uint32_t qkv_offset, uint32_t z_offset, uint32_t stride, int full) {
    (void)native_qkv_offset; (void)native_z_offset; (void)native_stride;
    if (!scratch_projection_eligible(n_tokens, full, qkv_type, z_type, 0))
        return qw3_metal_session_batch_dense_deltanet_projections(s,
            qkv_weight, qkv_type, z_weight, z_type, n_tokens, n_embd, n_qkv, n_z);
    if (!s || !s->obj) return 0;
    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    if (!qw3_metal_dense_batch_prepare_map(obj, n_tokens) ||
        !qw3_metal_session_ensure_prefill_scratch(obj, n_tokens,
            (uint64_t)n_tokens * stride * sizeof(float))) return 0;
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    if (!scratch_encode_matmul(obj, cb, qkv_weight, qkv_type, n_tokens,
            n_embd, n_qkv, obj.prefillX1, obj.prefillScratch,
            0, n_embd, qkv_offset, stride) ||
        !scratch_encode_matmul(obj, cb, z_weight, z_type, n_tokens,
            n_embd, n_z, obj.prefillX1, obj.prefillScratch,
            0, n_embd, z_offset, stride)) return 0;
    scratch_projection_calls += 2;
    return qw3_metal_finish_command_buffer(cb, owned, "private direct DeltaNet projections");
}

static int scratch_batch_gqa(qw3_metal_session *s,
    uint64_t q_weight, uint32_t q_type, uint64_t k_weight, uint32_t k_type,
    uint64_t v_weight, uint32_t v_type, uint32_t n_tokens, uint32_t n_embd,
    uint32_t qg_n, uint32_t kv_n, uint32_t mask,
    uint32_t native_q_offset, uint32_t native_k_offset, uint32_t native_v_offset, uint32_t native_stride,
    uint32_t q_offset, uint32_t k_offset, uint32_t v_offset,
    uint32_t stride, int full) {
    (void)native_q_offset; (void)native_k_offset;
    (void)native_v_offset; (void)native_stride;
    if (!scratch_projection_eligible(n_tokens, full, q_type, k_type, v_type))
        return qw3_metal_session_batch_dense_gqa_projections(s,
            q_weight, q_type, k_weight, k_type, v_weight, v_type,
            n_tokens, n_embd, qg_n, kv_n, mask);
    if (!s || !s->obj || mask != 7u) return 0;
    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    if (!qw3_metal_dense_batch_prepare_map(obj, n_tokens) ||
        !qw3_metal_session_ensure_prefill_scratch(obj, n_tokens,
            (uint64_t)n_tokens * stride * sizeof(float))) return 0;
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    if (!scratch_encode_matmul(obj, cb, q_weight, q_type, n_tokens,
            n_embd, qg_n, obj.prefillX1, obj.prefillScratch,
            0, n_embd, q_offset, stride) ||
        !scratch_encode_matmul(obj, cb, k_weight, k_type, n_tokens,
            n_embd, kv_n, obj.prefillX1, obj.prefillScratch,
            0, n_embd, k_offset, stride) ||
        !scratch_encode_matmul(obj, cb, v_weight, v_type, n_tokens,
            n_embd, kv_n, obj.prefillX1, obj.prefillScratch,
            0, n_embd, v_offset, stride)) return 0;
    scratch_projection_calls += 3;
    return qw3_metal_finish_command_buffer(cb, owned, "private direct GQA projections");
}

static int scratch_output_add(qw3_metal_session *s, uint64_t weight,
    uint32_t type, uint32_t n_tokens, uint32_t n_in, uint32_t n_embd,
    uint32_t native_input_offset, uint32_t native_input_stride,
    uint32_t input_offset, uint32_t stride) {
    (void)native_input_offset; (void)native_input_stride;
    if (!scratch_output_eligible(n_tokens, type))
        return qw3_metal_session_batch_dense_gqa_output_add(s, weight,
            type, n_tokens, n_in, n_embd);
    if (!s || !s->obj) return 0;
    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t out_bytes = (uint64_t)n_tokens * n_embd * sizeof(float);
    if (!obj.prefillX0 || obj.prefillX0.length < out_bytes) return 0;
    if (!obj.denseBatchDown || obj.denseBatchDown.length < out_bytes)
        obj.denseBatchDown = qw3_metal_new_private_buffer(out_bytes);
    if (!obj.denseBatchDown) return 0;
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    if (!scratch_encode_matmul(obj, cb, weight, type, n_tokens, n_in,
            n_embd, obj.prefillScratch, obj.denseBatchDown,
            input_offset, stride, 0, n_embd)) return 0;
    struct { uint32_t n; float scale; } args = {n_tokens * n_embd, 1.0f};
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_scale_x1_add_x0_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.prefillX0 offset:0 atIndex:1];
    [enc setBuffer:obj.denseBatchDown offset:0 atIndex:2];
    NSUInteger threads = MIN((NSUInteger)256,
        g_scale_x1_add_x0_pipeline.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(args.n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    scratch_output_calls++;
    return qw3_metal_finish_command_buffer(cb, owned, "private direct attention output");
}

static int scratch_skip_copies(uint32_t count) {
    scratch_copies_skipped += count;
    return 1;
}

/* Forward the exact layout already computed by the native prefill routine.
 * Partial attention batches and short chunks continue through native calls. */
#define qw3_metal_session_batch_dense_deltanet_projections_layout(...) \
    scratch_batch_deltanet(__VA_ARGS__, dense_dn_qkv_offset, dense_dn_z_offset, \
        dense_dn_stride, full_batch_deltanet)
#define qw3_metal_session_batch_dense_gqa_projections_layout(...) \
    scratch_batch_gqa(__VA_ARGS__, dense_gqa_qg_offset, dense_gqa_k_offset, \
        dense_gqa_v_offset, dense_gqa_stride, full_batch_gqa)
#define qw3_metal_session_stage_dense_deltanet_projections(...) \
    (scratch_projection_eligible(ntok, full_batch_deltanet, \
        lw->linear_qkv_proj->type, lw->linear_gate_proj->type, 0) ? \
        scratch_skip_copies(2) : qw3_metal_session_stage_dense_deltanet_projections(__VA_ARGS__))
#define qw3_metal_session_stage_dense_gqa_projections(...) \
    (scratch_projection_eligible(ntok, full_batch_gqa, lw->attn_q_proj->type, \
        lw->attn_k_proj->type, lw->attn_v_proj->type) ? scratch_skip_copies(3) : \
        qw3_metal_session_stage_dense_gqa_projections(__VA_ARGS__))
#define SCRATCH_OUTPUT_TYPE (is_full_attn ? lw->attn_o_proj->type : lw->linear_ssm_out->type)
#define qw3_metal_session_copy_batch_scratch_to_x1(...) \
    (scratch_output_eligible(ntok, SCRATCH_OUTPUT_TYPE) ? scratch_skip_copies(1) : \
        qw3_metal_session_copy_batch_scratch_to_x1(__VA_ARGS__))
#define qw3_metal_session_batch_dense_gqa_output_add_layout(...) \
    scratch_output_add(__VA_ARGS__, \
        is_full_attn ? dense_gqa_inner_offset : dense_dn_inner_offset, \
        is_full_attn ? dense_gqa_stride : dense_dn_stride)
#define qw3_metal_env_n_gpu_layers scratch_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled scratch_core_llama_split_enabled
#define qw3_metal_llama_split_start scratch_core_llama_split_start
#define qw3_engine_open scratch_engine_open_base
#include "../qw3.c"
#undef qw3_engine_open
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
#undef qw3_metal_session_batch_dense_deltanet_projections_layout
#undef qw3_metal_session_batch_dense_gqa_projections_layout
#undef qw3_metal_session_stage_dense_deltanet_projections
#undef qw3_metal_session_stage_dense_gqa_projections
#undef qw3_metal_session_copy_batch_scratch_to_x1
#undef qw3_metal_session_batch_dense_gqa_output_add_layout
#undef SCRATCH_OUTPUT_TYPE

static int scratch_parse_mode(const char *name) {
    if (!name || !strcmp(name, "off")) return 0;
    if (!strcmp(name, "projections")) return 1;
    if (!strcmp(name, "output")) return 2;
    if (!strcmp(name, "all")) return 3;
    return -1;
}

static void scratch_select(int mode) {
    scratch_mode = mode;
#ifdef QW3_PRIVATE_WITH_GDN
    probe_gdn = mode ? 2 : 0;
#endif
}

int qw3_engine_open(qw3_engine **engine, const qw3_engine_options *options) {
    int mode = scratch_parse_mode(getenv("QW3_PRIVATE_DIRECT_SCRATCH"));
#ifdef QW3_PRIVATE_WITH_GDN
    const char *combined = getenv("QW3_PRIVATE_PACKED8_SCRATCH");
    if (combined && strcmp(combined, "1")) return -1;
    if (combined) mode = 3;
#endif
    if (mode < 0) return -1;
    scratch_select(mode);
    int rc = scratch_engine_open_base(engine, options);
    if (rc) return rc;
    if (!qw3_metal_compile_kernels()) {
        qw3_engine_close(*engine);
        *engine = NULL;
        return -1;
    }
#ifdef QW3_PRIVATE_WITH_GDN
    if (!probe_initialize()) {
        qw3_engine_close(*engine);
        *engine = NULL;
        return -1;
    }
#endif
    return 0;
}
