int qw3_metal_session_matvec_iq3_s_pair_x1_to_scratch(
    qw3_metal_session *s, uint64_t gate_offset, uint64_t up_offset,
    uint32_t n_in, uint32_t n_out) {
    if (groups_selected==2) return groups_original_pair(s,gate_offset,up_offset,n_in,n_out);
    groups_calls++;
    if (!s || !s->obj || !n_in || !n_out || (n_in % 256u) != 0u)
        return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;
    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_in * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_out * 2u * sizeof(float);
    const uint64_t row_bytes = (uint64_t)(n_in / 256u) * 110ull;
    const uint64_t weight_bytes = row_bytes * n_out;
    if (!obj.x1 || obj.x1.length < x_bytes ||
        !obj.scratch || obj.scratch.length < out_bytes ||
        row_bytes > UINT32_MAX || !g_matvec_iq3_s_pair_pipeline) return 0;
    uint64_t gate_inner = 0, up_inner = 0;
    id<MTLBuffer> gate = qw3_metal_model_view_for(
        gate_offset, weight_bytes, &gate_inner);
    id<MTLBuffer> up = qw3_metal_model_view_for(
        up_offset, weight_bytes, &up_inner);
    id<MTLBuffer> kgrid = qw3_metal_iq3s_kgrid_buffer();
    if (!gate || !up || !kgrid) return 0;
    struct { uint32_t n_in, n_out, row_bytes; } args = {
        n_in, n_out, (uint32_t)row_bytes
    };
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    if (!cb) return 0;
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:groups_pipeline[groups_selected==8]];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:gate offset:(NSUInteger)gate_inner atIndex:1];
    [enc setBuffer:up offset:(NSUInteger)up_inner atIndex:2];
    [enc setBuffer:obj.x1 offset:0 atIndex:3];
    [enc setBuffer:obj.scratch offset:0 atIndex:4];
    [enc setBuffer:kgrid offset:0 atIndex:5];
    [enc setThreadgroupMemoryLength:64u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((n_out + groups_selected*4u - 1u)/(groups_selected*4u), 1u, 1u)
         threadsPerThreadgroup:MTLSizeMake(groups_selected*32u, 1u, 1u)];
    qw3_metal_end_compute_encoder(cb, enc);
    return qw3_metal_finish_command_buffer(cb, owned,
                                            "dense IQ3_S gate/up pair");
}
