inline float private_iq3xxs_sign_dot1(device const uchar *blk,
                                    thread const float4 *xx,
                                    threadgroup const float4 *grid,
                                    device const uchar *signs, uint ib) {
    const float d = float(*((device const half *)blk));
    device const uchar *qs = blk + 2u + 8u * ib;
    device const uchar *sp = blk + 66u + 4u * ib;
    const uint packed = uint(sp[0]) | (uint(sp[1]) << 8u) |
                        (uint(sp[2]) << 16u) | (uint(sp[3]) << 24u);
    const float db = d * (0.5f + float(packed >> 28u)) * 0.5f;
    float sum = 0.0f;
    for (uint l = 0; l < 4u; l++) {
        const uint sign_index = (packed >> (7u * l)) & 127u;
        const uint s = sign_index | ((popcount(sign_index) & 1u) << 7u);
        const float4 sign1 = select(float4(1.0f), float4(-1.0f),
                                   (uint4(s) & uint4(1u, 2u, 4u, 8u)) != 0u);
        const float4 sign2 = select(float4(1.0f), float4(-1.0f),
                                   (uint4(s) & uint4(16u, 32u, 64u, 128u)) != 0u);
        sum += dot(grid[uint(qs[2u * l])] * sign1, xx[2u * l]) +
               dot(grid[uint(qs[2u * l + 1u])] * sign2, xx[2u * l + 1u]);
    }
    return db * sum;
}
kernel void private_iq3xxs_sign_fast1(
                               constant qw3_matvec_q8_0_args &args,
                               device const uchar *weights,
                               device const float *x,
                               device float *out,
                               device const ushort *grid_table,
                               device const uchar *sign_table,
                               threadgroup float *sh,
                               uint group [[threadgroup_position_in_grid]],
                               ushort simd_idx [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    /* Split quant blocks into 32-value units so all lanes participate, even
     * for 5120 inputs (only 20 whole blocks). Reuse x across output rows. */
    threadgroup float4 grid[256];
    device const uchar *signs = sign_table;
    const uint tid = uint(simd_idx) * 32u + uint(lane);
    for (uint i = tid; i < 256u; i += 64u) {
        const ushort packed = grid_table[i];
        grid[i] = float4(qw3_iq3xxs_grid_value(packed, 0u),
                         qw3_iq3xxs_grid_value(packed, 1u),
                         qw3_iq3xxs_grid_value(packed, 2u),
                         qw3_iq3xxs_grid_value(packed, 3u));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row0 = (group * 2u + uint(simd_idx)) * 2u;
    uint row1 = row0 + 1u;
    float sum0 = 0.0f, sum1 = 0.0f;
    uint nb32 = args.n_in / 32u;
    for (uint b32 = uint(lane); b32 < nb32; b32 += 32u) {
        const uint b = b32 / 8u, ib = b32 & 7u;
        device const float4 *x4 = (device const float4 *)(x + uint64_t(b32) * 32ull);
        float4 xx[8];
        for (uint j = 0; j < 8u; j++) xx[j] = x4[j];
        if (row0 < args.n_out)
            sum0 += private_iq3xxs_sign_dot1(
                weights + uint64_t(row0) * uint64_t(args.row_bytes) +
                uint64_t(b) * 98ull, xx, grid, signs, ib);
        if (row1 < args.n_out)
            sum1 += private_iq3xxs_sign_dot1(
                weights + uint64_t(row1) * uint64_t(args.row_bytes) +
                uint64_t(b) * 98ull, xx, grid, signs, ib);
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (lane == 0) {
        if (row0 < args.n_out) out[row0] = sum0;
        if (row1 < args.n_out) out[row1] = sum1;
    }
}
inline float private_iq3xxs_sign_dot2(device const uchar *blk,
                                    thread const float4 *xx,
                                    threadgroup const float4 *grid,
                                    device const uchar *signs, uint ib) {
    const float d = float(*((device const half *)blk));
    device const uchar *qs = blk + 2u + 8u * ib;
    device const uchar *sp = blk + 66u + 4u * ib;
    const uint packed = uint(sp[0]) | (uint(sp[1]) << 8u) |
                        (uint(sp[2]) << 16u) | (uint(sp[3]) << 24u);
    const float db = d * (0.5f + float(packed >> 28u)) * 0.5f;
    float sum = 0.0f;
    for (uint l = 0; l < 4u; l++) {
        const uint s = uint(signs[(packed >> (7u * l)) & 127u]);
        const float4 sign1 = select(float4(1.0f), float4(-1.0f),
                                   (uint4(s) & uint4(1u, 2u, 4u, 8u)) != 0u);
        const float4 sign2 = select(float4(1.0f), float4(-1.0f),
                                   (uint4(s) & uint4(16u, 32u, 64u, 128u)) != 0u);
        sum += dot(grid[uint(qs[2u * l])] * sign1, xx[2u * l]) +
               dot(grid[uint(qs[2u * l + 1u])] * sign2, xx[2u * l + 1u]);
    }
    return db * sum;
}
kernel void private_iq3xxs_sign_fast2(
                               constant qw3_matvec_q8_0_args &args,
                               device const uchar *weights,
                               device const float *x,
                               device float *out,
                               device const ushort *grid_table,
                               device const uchar *sign_table,
                               threadgroup float *sh,
                               uint group [[threadgroup_position_in_grid]],
                               ushort simd_idx [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    /* Split quant blocks into 32-value units so all lanes participate, even
     * for 5120 inputs (only 20 whole blocks). Reuse x across output rows. */
    threadgroup float4 grid[256];
    device const uchar *signs = sign_table;
    const uint tid = uint(simd_idx) * 32u + uint(lane);
    for (uint i = tid; i < 256u; i += 64u) {
        const ushort packed = grid_table[i];
        grid[i] = float4(qw3_iq3xxs_grid_value(packed, 0u),
                         qw3_iq3xxs_grid_value(packed, 1u),
                         qw3_iq3xxs_grid_value(packed, 2u),
                         qw3_iq3xxs_grid_value(packed, 3u));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row0 = (group * 2u + uint(simd_idx)) * 2u;
    uint row1 = row0 + 1u;
    float sum0 = 0.0f, sum1 = 0.0f;
    uint nb32 = args.n_in / 32u;
    for (uint b32 = uint(lane); b32 < nb32; b32 += 32u) {
        const uint b = b32 / 8u, ib = b32 & 7u;
        device const float4 *x4 = (device const float4 *)(x + uint64_t(b32) * 32ull);
        float4 xx[8];
        for (uint j = 0; j < 8u; j++) xx[j] = x4[j];
        if (row0 < args.n_out)
            sum0 += private_iq3xxs_sign_dot2(
                weights + uint64_t(row0) * uint64_t(args.row_bytes) +
                uint64_t(b) * 98ull, xx, grid, signs, ib);
        if (row1 < args.n_out)
            sum1 += private_iq3xxs_sign_dot2(
                weights + uint64_t(row1) * uint64_t(args.row_bytes) +
                uint64_t(b) * 98ull, xx, grid, signs, ib);
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (lane == 0) {
        if (row0 < args.n_out) out[row0] = sum0;
        if (row1 < args.n_out) out[row1] = sum1;
    }
}
