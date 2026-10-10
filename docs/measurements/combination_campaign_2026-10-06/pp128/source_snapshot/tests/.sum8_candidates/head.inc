int qw3_metal_session_matvec_q5_k_x1_to_logits(qw3_metal_session *s,
                                               uint64_t tensor_offset,
                                               uint32_t n_in,
                                               uint32_t n_out,
                                               float *out) {
    if (!(combo_mode & 4)) return sum8_base_logits(s,tensor_offset,n_in,n_out,out);
    if (!sum8_compile()) return 0;
    if (!s || !s->obj || n_in == 0 || n_out == 0 || (n_in % 256u) != 0u)
        return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_in * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_out * sizeof(float);
    if (!obj.x1 || !obj.logits || obj.x1.length < x_bytes ||
        obj.logits.length < out_bytes) return 0;

    const uint64_t row_bytes = (uint64_t)(n_in / 256u) * 176ull;
    const uint64_t tensor_bytes = row_bytes * (uint64_t)n_out;
    uint64_t model_offset = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(
        tensor_offset, tensor_bytes, &model_offset);
    if (!wb) return 0;
    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
    } args = { n_in, n_out, (uint32_t)row_bytes };

    uint64_t partial_bytes=(uint64_t)(n_in/8u)*sizeof(float);
    if (!sum8_buffer || sum8_buffer.length<partial_bytes)
        sum8_buffer=qw3_metal_new_private_buffer(partial_bytes);
    if (!sum8_buffer) return 0;
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    if (!cb) return 0;
    id<MTLComputeCommandEncoder> prep=qw3_metal_compute_encoder(cb);
    [prep setComputePipelineState:sum8_prepare_pipeline];
    [prep setBytes:&n_in length:sizeof(n_in) atIndex:0];
    [prep setBuffer:obj.x1 offset:0 atIndex:1];
    [prep setBuffer:sum8_buffer offset:0 atIndex:2];
    [prep dispatchThreads:MTLSizeMake(n_in/8u,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    qw3_metal_end_compute_encoder(cb,prep);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:sum8_head_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)model_offset atIndex:1];
    [enc setBuffer:obj.x1 offset:0 atIndex:2];
    [enc setBuffer:obj.logits offset:0 atIndex:3];
    [enc setBuffer:sum8_buffer offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((n_out + 1u) / 2u, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(64u, 1u, 1u)];
    qw3_metal_end_compute_encoder(cb, enc);

    id<MTLBuffer> readback = nil;
    if (out) {
        readback = [g_device newBufferWithLength:(NSUInteger)out_bytes
                                         options:MTLResourceStorageModeShared];
        if (!readback) return 0;
        qw3_metal_close_batch_encoder();
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:obj.logits sourceOffset:0
                    toBuffer:readback destinationOffset:0
                        size:(NSUInteger)out_bytes];
        [blit endEncoding];
    }
    if (out && !owned) {
        if (!qw3_metal_synchronize()) return 0;
    } else if (!qw3_metal_finish_command_buffer(cb, owned, "q5_K logits")) {
        return 0;
    }
    if (cb.status == MTLCommandBufferStatusError) return 0;
    if (out) memcpy(out, readback.contents, (size_t)out_bytes);
    sum8_head_calls++;
    return 1;
}

