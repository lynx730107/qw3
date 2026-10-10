/* Native direct/batch4 dispatch copied from qw3_metal.m. Only buffer extent
 * checks and the four layout fields differ; pipeline, arithmetic and geometry
 * stay identical. Private to the direct-scratch diagnostic. */
static int scratch_encode_matmul(
    QW3MetalSessionObj *obj, id<MTLCommandBuffer> cb,
    uint64_t tensor_offset, uint32_t tensor_type, uint32_t n_tokens,
    uint32_t n_in, uint32_t n_out, id<MTLBuffer> src, id<MTLBuffer> dst,
    uint32_t in_offset, uint32_t in_stride,
    uint32_t out_offset, uint32_t out_stride) {
    if (!obj || !cb || !src || !dst || !n_tokens || !n_in || !n_out ||
        (n_in % 256u) != 0 ||
        (tensor_type != 10u && tensor_type != 11u && tensor_type != 12u &&
         tensor_type != 13u && tensor_type != 14u &&
         tensor_type != 16u && tensor_type != 17u &&
         tensor_type != 18u && tensor_type != 20u &&
         tensor_type != 21u && tensor_type != 22u &&
         tensor_type != 23u)) return 0;
    if (in_offset > in_stride || n_in > in_stride - in_offset ||
        out_offset > out_stride || n_out > out_stride - out_offset) return 0;
    const uint64_t x_bytes = ((uint64_t)(n_tokens - 1u) * in_stride +
                             in_offset + n_in) * sizeof(float);
    const uint64_t out_bytes = ((uint64_t)(n_tokens - 1u) * out_stride +
                               out_offset + n_out) * sizeof(float);
    const uint64_t row_bytes = tensor_type == 20u ?
        (uint64_t)(n_in / 32u) * 18ull : (uint64_t)(n_in / 256u) *
        (tensor_type == 10u ? 84ull :
         tensor_type == 11u ? 110ull :
         tensor_type == 12u ? 144ull :
         tensor_type == 13u ? 176ull :
         tensor_type == 14u ? 210ull :
         tensor_type == 16u ? 66ull :
         tensor_type == 17u ? 74ull :
         tensor_type == 18u ? 98ull :
         tensor_type == 21u ? 110ull :
         tensor_type == 22u ? 82ull : 136ull);
    const uint64_t weight_bytes = row_bytes * n_out;
    if (row_bytes > UINT32_MAX || weight_bytes > UINT32_MAX ||
        src.length < x_bytes || dst.length < out_bytes ||
        !obj.denseBatchCount || !obj.denseBatchPairIds ||
        !obj.denseBatchBlockIds) return 0;
    uint64_t inner = 0;
    id<MTLBuffer> weights = qw3_metal_model_view_for(
        tensor_offset, weight_bytes, &inner);
    if (tensor_type == 12u || tensor_type == 13u) {
        if (!weights || !g_matmul_q4_q5_k_mpp_pipeline ||
            n_tokens < 16u) return 0;
        struct {
            uint32_t n_in, n_out, row_bytes, n_tokens;
            uint32_t in_offset, in_stride, out_offset, out_stride;
            uint32_t is_q5;
        } args = {n_in, n_out, (uint32_t)row_bytes, n_tokens,
                  in_offset, in_stride, out_offset, out_stride, tensor_type == 13u};
        const int use_staged128 = n_tokens % 128u == 0u &&
            g_matmul_q4_q5_k_mpp_staged128_pipeline &&
            qw3_metal_staged128_enabled();
        id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:use_staged128 ?
            g_matmul_q4_q5_k_mpp_staged128_pipeline :
            g_matmul_q4_q5_k_mpp_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weights offset:(NSUInteger)inner atIndex:1];
        [enc setBuffer:src offset:0 atIndex:2];
        [enc setBuffer:dst offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:8192u atIndex:0];
        [enc dispatchThreadgroups:
             MTLSizeMake(use_staged128 ? n_tokens / 128u :
                         (n_tokens + 63u) / 64u,
                         (n_out + 63u) / 64u, 1u)
             threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
        qw3_metal_end_compute_encoder(cb, enc);
        return 1;
    }
    if (tensor_type == 11u) {
        if (!weights || !g_matmul_q3_k_batch4_pipeline) return 0;
        const int use_mpp = n_tokens >= 16u &&
            g_matmul_q3_k_mpp_pipeline &&
            getenv("QW3_METAL_Q3_K_BATCH4") == NULL;
        const int use_staged128 = use_mpp && n_tokens % 128u == 0u &&
            g_matmul_q3_k_mpp_staged128_pipeline &&
            qw3_metal_staged128_enabled();
        struct {
            uint32_t n_in, n_out, row_bytes, n_tokens;
            uint32_t in_offset, in_stride, out_offset, out_stride;
        } args = {n_in, n_out, (uint32_t)row_bytes, n_tokens,
                  in_offset, in_stride, out_offset, out_stride};
        id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:use_staged128 ?
            g_matmul_q3_k_mpp_staged128_pipeline : (use_mpp ?
            g_matmul_q3_k_mpp_pipeline : g_matmul_q3_k_batch4_pipeline)];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weights offset:(NSUInteger)inner atIndex:1];
        [enc setBuffer:src offset:0 atIndex:2];
        [enc setBuffer:dst offset:0 atIndex:3];
        if (use_mpp) [enc setThreadgroupMemoryLength:8192u atIndex:0];
        [enc dispatchThreadgroups:use_staged128 ?
             MTLSizeMake(n_tokens / 128u, (n_out + 63u) / 64u, 1u) : (use_mpp ?
             MTLSizeMake((n_tokens + 63u) / 64u, (n_out + 63u) / 64u, 1u) :
             MTLSizeMake((n_out + 3u) / 4u, (n_tokens + 3u) / 4u, 1u)
             )
             threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
        qw3_metal_end_compute_encoder(cb, enc);
        return 1;
    }
    if (tensor_type == 10u) {
        if (!weights || !g_matmul_q2_k_batch4_pipeline) return 0;
        const int use_mpp = n_tokens >= 16u &&
            g_matmul_q2_k_mpp_pipeline &&
            getenv("QW3_METAL_Q2_K_BATCH4") == NULL;
        struct {
            uint32_t n_in, n_out, row_bytes, n_tokens;
            uint32_t in_offset, in_stride, out_offset, out_stride;
        } args = {n_in, n_out, (uint32_t)row_bytes, n_tokens,
                  in_offset, in_stride, out_offset, out_stride};
        id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:use_mpp ?
            g_matmul_q2_k_mpp_pipeline : g_matmul_q2_k_batch4_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weights offset:(NSUInteger)inner atIndex:1];
        [enc setBuffer:src offset:0 atIndex:2];
        [enc setBuffer:dst offset:0 atIndex:3];
        if (use_mpp) [enc setThreadgroupMemoryLength:8192u atIndex:0];
        [enc dispatchThreadgroups:use_mpp ?
             MTLSizeMake((n_tokens + 63u) / 64u, (n_out + 63u) / 64u, 1u) :
             MTLSizeMake((n_out + 3u) / 4u, (n_tokens + 3u) / 4u, 1u)
             threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
        qw3_metal_end_compute_encoder(cb, enc);
        return 1;
    }
    if (tensor_type == 20u) {
        if (!weights || !g_matmul_iq4_nl_mpp_pipeline ||
            n_tokens < 16u) return 0;
        struct {
            uint32_t n_in, n_out, row_bytes, n_tokens;
            uint32_t in_offset, in_stride, out_offset, out_stride;
        } args = {n_in, n_out, (uint32_t)row_bytes, n_tokens,
                  in_offset, in_stride, out_offset, out_stride};
        id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:g_matmul_iq4_nl_mpp_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weights offset:(NSUInteger)inner atIndex:1];
        [enc setBuffer:src offset:0 atIndex:2];
        [enc setBuffer:dst offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:8192u atIndex:0];
        [enc dispatchThreadgroups:
             MTLSizeMake((n_tokens + 63u) / 64u, (n_out + 63u) / 64u, 1u)
             threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
        qw3_metal_end_compute_encoder(cb, enc);
        return 1;
    }
    if (tensor_type == 14u) {
        if (!weights || !g_matmul_q6_k_mpp_pipeline ||
            n_tokens < 16u) return 0;
        struct {
            uint32_t n_in, n_out, row_bytes, n_tokens;
            uint32_t in_offset, in_stride, out_offset, out_stride;
        } args = {n_in, n_out, (uint32_t)row_bytes, n_tokens,
                  in_offset, in_stride, out_offset, out_stride};
        id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:g_matmul_q6_k_mpp_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weights offset:(NSUInteger)inner atIndex:1];
        [enc setBuffer:src offset:0 atIndex:2];
        [enc setBuffer:dst offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:8192u atIndex:0];
        [enc dispatchThreadgroups:
             MTLSizeMake((n_tokens + 63u) / 64u, (n_out + 63u) / 64u, 1u)
             threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
        qw3_metal_end_compute_encoder(cb, enc);
        return 1;
    }
    if (tensor_type == 16u || tensor_type == 17u) {
        id<MTLBuffer> signs = nil;
        id<MTLBuffer> grid = nil;
        id<MTLComputePipelineState> pipeline = tensor_type == 16u ?
            g_matmul_iq2_xxs_mpp_pipeline : g_matmul_iq2_xs_mpp_pipeline;
        if (!weights || !pipeline || n_tokens < 16u ||
            !qw3_metal_dense_iq_buffers(&signs,
                tensor_type == 17u ? &grid : NULL, NULL, NULL)) return 0;
        if (tensor_type == 16u) grid = g_iq2xxs_kgrid_buffer;
        struct {
            uint32_t n_in, n_out, row_bytes, n_tokens;
            uint32_t in_offset, in_stride, out_offset, out_stride;
        } args = {n_in, n_out, (uint32_t)row_bytes, n_tokens,
                  in_offset, in_stride, out_offset, out_stride};
        id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weights offset:(NSUInteger)inner atIndex:1];
        [enc setBuffer:src offset:0 atIndex:2];
        [enc setBuffer:dst offset:0 atIndex:3];
        [enc setBuffer:grid offset:0 atIndex:4];
        [enc setBuffer:signs offset:0 atIndex:5];
        [enc setThreadgroupMemoryLength:8192u atIndex:0];
        [enc dispatchThreadgroups:
             MTLSizeMake((n_tokens + 63u) / 64u, (n_out + 63u) / 64u, 1u)
             threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
        qw3_metal_end_compute_encoder(cb, enc);
        return 1;
    }
    if (tensor_type == 18u || tensor_type == 22u) {
        id<MTLBuffer> grid = nil;
        id<MTLBuffer> signs = nil;
        const int use_mpp = n_tokens >= 16u &&
            (tensor_type == 18u ?
             (g_matmul_iq3_xxs_mpp_pipeline &&
              getenv("QW3_METAL_IQ3_XXS_BATCH4") == NULL) :
             (g_matmul_iq2_s_mpp_pipeline &&
              getenv("QW3_METAL_IQ2_S_BATCH4") == NULL));
        id<MTLComputePipelineState> pipeline = use_mpp ?
            (tensor_type == 18u ? g_matmul_iq3_xxs_mpp_pipeline :
                                  g_matmul_iq2_s_mpp_pipeline) :
            (tensor_type == 18u ? g_matmul_iq3_xxs_batch4_pipeline :
                                  g_matmul_iq2_s_batch4_pipeline);
        if (!use_mpp && qw3_metal_dense_short_vector_enabled(n_tokens, n_in, n_out)) {
            id<MTLComputePipelineState> short_pipeline = tensor_type == 18u ?
                g_matmul_iq3_xxs_short_pipeline : g_matmul_iq2_s_short_pipeline;
            if (short_pipeline && short_pipeline.maxTotalThreadsPerThreadgroup >= 128u)
                pipeline = short_pipeline;
        }
        const int use_staged128 = tensor_type == 18u && use_mpp &&
            n_tokens % 128u == 0u &&
            g_matmul_iq3_xxs_mpp_staged128_pipeline &&
            qw3_metal_staged128_enabled();
        const int use_iq2s_staged128 = tensor_type == 22u && use_mpp &&
            n_tokens % 128u == 0u &&
            g_matmul_iq2_s_mpp_staged128_pipeline &&
            qw3_metal_staged128_enabled();
        if (use_staged128)
            pipeline = g_matmul_iq3_xxs_mpp_staged128_pipeline;
        else if (use_iq2s_staged128)
            pipeline = g_matmul_iq2_s_mpp_staged128_pipeline;
        if (!weights || !pipeline ||
            !qw3_metal_dense_iq_buffers(tensor_type == 18u ? &signs : NULL,
                NULL, tensor_type == 22u ? &grid : NULL,
                tensor_type == 18u ? &grid : NULL)) return 0;
        struct {
            uint32_t n_in, n_out, row_bytes, n_tokens;
            uint32_t in_offset, in_stride, out_offset, out_stride;
        } args = {n_in, n_out, (uint32_t)row_bytes, n_tokens,
                  in_offset, in_stride, out_offset, out_stride};
        id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weights offset:(NSUInteger)inner atIndex:1];
        [enc setBuffer:src offset:0 atIndex:2];
        [enc setBuffer:dst offset:0 atIndex:3];
        [enc setBuffer:grid offset:0 atIndex:4];
        if (tensor_type == 18u)
            [enc setBuffer:signs offset:0 atIndex:5];
        if (use_mpp) [enc setThreadgroupMemoryLength:8192u atIndex:0];
        [enc dispatchThreadgroups:(use_staged128 || use_iq2s_staged128) ?
             MTLSizeMake(n_tokens / 128u, (n_out + 63u) / 64u, 1u) : (use_mpp ?
             MTLSizeMake((n_tokens + 63u) / 64u, (n_out + 63u) / 64u, 1u) :
             MTLSizeMake((n_out + 3u) / 4u, (n_tokens + 3u) / 4u, 1u)
             )
             threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
        qw3_metal_end_compute_encoder(cb, enc);
        return 1;
    }
    if (tensor_type == 23u && n_tokens >= 16u &&
        g_matmul_iq4_xs_mpp_pipeline) {
        const int use_staged128 = n_tokens % 128u == 0u &&
            g_matmul_iq4_xs_mpp_staged128_pipeline &&
            qw3_metal_staged128_enabled();
        struct {
            uint32_t n_in, n_out, row_bytes, n_tokens;
            uint32_t in_offset, in_stride, out_offset, out_stride;
        } direct_args = {n_in, n_out, (uint32_t)row_bytes, n_tokens,
                         in_offset, in_stride, out_offset, out_stride};
        id<MTLComputeCommandEncoder> direct =
            qw3_metal_compute_encoder(cb);
        id<MTLComputePipelineState> selected_pipeline = use_staged128 ?
            g_matmul_iq4_xs_mpp_staged128_pipeline :
            g_matmul_iq4_xs_mpp_pipeline;
        const uint32_t columns = use_staged128 ?
            QW3_METAL_STAGED128_COLUMNS(selected_pipeline) : 64u;
        const uint32_t threads = use_staged128 ?
            QW3_METAL_STAGED128_THREADS(selected_pipeline) : 128u;
        [direct setComputePipelineState:selected_pipeline];
        [direct setBytes:&direct_args length:sizeof(direct_args) atIndex:0];
        [direct setBuffer:weights offset:(NSUInteger)inner atIndex:1];
        [direct setBuffer:src offset:0 atIndex:2];
        [direct setBuffer:dst offset:0 atIndex:3];
        [direct setThreadgroupMemoryLength:8192u atIndex:0];
        [direct dispatchThreadgroups:
            MTLSizeMake(use_staged128 ? n_tokens / 128u :
                        (n_tokens + 63u) / 64u,
                        (n_out + columns - 1u) / columns, 1u)
            threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
        qw3_metal_end_compute_encoder(cb, direct);
        return 1;
    }
    id<MTLBuffer> kgrid = tensor_type == 21u ?
        qw3_metal_iq3s_expanded_kgrid_buffer() : nil;
    if (!weights || (tensor_type == 21u && !kgrid)) return 0;
    if (tensor_type == 21u && n_tokens >= 16u &&
        g_matmul_iq3_s_mpp_pipeline) {
        const int use_staged128 = n_tokens % 128u == 0u &&
            g_matmul_iq3_s_mpp_staged128_pipeline &&
            qw3_metal_staged128_enabled();
        struct {
            uint32_t n_in, n_out, row_bytes, n_tokens;
            uint32_t in_offset, in_stride, out_offset, out_stride;
        } direct_args = {n_in, n_out, (uint32_t)row_bytes, n_tokens,
                         in_offset, in_stride, out_offset, out_stride};
        id<MTLComputeCommandEncoder> direct =
            qw3_metal_compute_encoder(cb);
        id<MTLComputePipelineState> selected_pipeline = use_staged128 ?
            g_matmul_iq3_s_mpp_staged128_pipeline :
            g_matmul_iq3_s_mpp_pipeline;
        const uint32_t columns = use_staged128 ?
            QW3_METAL_STAGED128_COLUMNS(selected_pipeline) : 64u;
        const uint32_t threads = use_staged128 ?
            QW3_METAL_STAGED128_THREADS(selected_pipeline) : 128u;
        [direct setComputePipelineState:selected_pipeline];
        [direct setBytes:&direct_args length:sizeof(direct_args) atIndex:0];
        [direct setBuffer:weights offset:(NSUInteger)inner atIndex:1];
        [direct setBuffer:src offset:0 atIndex:2];
        [direct setBuffer:dst offset:0 atIndex:3];
        [direct setBuffer:kgrid offset:0 atIndex:4];
        [direct setThreadgroupMemoryLength:8192u atIndex:0];
        [direct dispatchThreadgroups:
            MTLSizeMake(use_staged128 ? n_tokens / 128u :
                        (n_tokens + 63u) / 64u,
                        (n_out + columns - 1u) / columns, 1u)
            threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
        qw3_metal_end_compute_encoder(cb, direct);
        return 1;
    }
    return 0; /* Mapped fallback has a different ABI; use the native caller. */
}
