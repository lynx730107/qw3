inline float private_iq3_pair_grid_dot1(device const uchar *blk,
                                  thread const float4 *xx,
                                  threadgroup const ushort *grid, uint ib) {
    const float d = float(*((device const half *)blk));
    device const uchar *qs = blk + 2u + ib * 8u;
    const uint qhb = uint(blk[66u + ib]);
    device const uchar *signs = blk + 74u + ib * 4u;
    const uint scales = uint(blk[106u + ib / 2u]);
    const float db = d * float(1u + 2u * ((scales >> (4u * (ib & 1u))) & 15u));
    float sum = 0.0f;
    for (uint l = 0; l < 4u; l++) {
        const uint idx1 = uint(qs[2u * l]) | ((qhb << (8u - 2u * l)) & 256u);
        const uint idx2 = uint(qs[2u * l + 1u]) | ((qhb << (7u - 2u * l)) & 256u);
        const uint s = uint(signs[l]);
        const float4 sign1 = select(float4(1.0f), float4(-1.0f),
                                   (uint4(s) & uint4(1u, 2u, 4u, 8u)) != 0u);
        const float4 sign2 = select(float4(1.0f), float4(-1.0f),
                                   (uint4(s) & uint4(16u, 32u, 64u, 128u)) != 0u);
        sum += dot(qw3_iq3s_float_grid(grid[idx1]) * sign1, xx[2u * l]) +
               dot(qw3_iq3s_float_grid(grid[idx2]) * sign2, xx[2u * l + 1u]);
    }
    return db * sum;
}
kernel void private_iq3_pair_grid_fast1(constant qw3_matvec_q8_0_args &args,
                                       device const uchar *gate_weights,
                                       device const uchar *up_weights,
                                       device const float *x,
                                       device float *out,
                                       device const ushort *kgrid,
                                       threadgroup float *sh,
                                       uint group [[threadgroup_position_in_grid]],
                                       ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                       ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    threadgroup ushort grid[512];
    const uint tid = uint(simd_idx) * 32u + uint(lane);
    for (uint i = tid; i < 512u; i += 64u) grid[i] = kgrid[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nr0 = 4u;
    const uint nsg = 2u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    float gate_sum0 = 0.0f, gate_sum1 = 0.0f, gate_sum2 = 0.0f, gate_sum3 = 0.0f;
    float up_sum0 = 0.0f, up_sum1 = 0.0f, up_sum2 = 0.0f, up_sum3 = 0.0f;
    uint nb32 = (args.n_in / 256u) * 8u;
    for (uint ib32 = uint(lane); ib32 < nb32; ib32 += 32u) {
        uint ibl = ib32 / 8u;
        uint ib = ib32 - ibl * 8u;
        float4 xx[8];
        device const float4 *x4 = (device const float4 *)(x + uint64_t(ib32) * 32ull);
        for (uint j = 0; j < 8u; j++) xx[j] = x4[j];
        uint row = first_row;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;
            gate_sum0 += private_iq3_pair_grid_dot1(gate_weights + off, xx, grid, ib);
            up_sum0 += private_iq3_pair_grid_dot1(up_weights + off, xx, grid, ib);
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;
            gate_sum1 += private_iq3_pair_grid_dot1(gate_weights + off, xx, grid, ib);
            up_sum1 += private_iq3_pair_grid_dot1(up_weights + off, xx, grid, ib);
        }
        row = first_row + 2u;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;
            gate_sum2 += private_iq3_pair_grid_dot1(gate_weights + off, xx, grid, ib);
            up_sum2 += private_iq3_pair_grid_dot1(up_weights + off, xx, grid, ib);
        }
        row = first_row + 3u;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;
            gate_sum3 += private_iq3_pair_grid_dot1(gate_weights + off, xx, grid, ib);
            up_sum3 += private_iq3_pair_grid_dot1(up_weights + off, xx, grid, ib);
        }
    }
    gate_sum0 = simd_sum(gate_sum0); up_sum0 = simd_sum(up_sum0);
    gate_sum1 = simd_sum(gate_sum1); up_sum1 = simd_sum(up_sum1);
    gate_sum2 = simd_sum(gate_sum2); up_sum2 = simd_sum(up_sum2);
    gate_sum3 = simd_sum(gate_sum3); up_sum3 = simd_sum(up_sum3);
    if (lane == 0) {
        uint row = first_row;
        if (row < args.n_out) { out[row] = gate_sum0; out[args.n_out + row] = up_sum0; }
        row = first_row + 1u;
        if (row < args.n_out) { out[row] = gate_sum1; out[args.n_out + row] = up_sum1; }
        row = first_row + 2u;
        if (row < args.n_out) { out[row] = gate_sum2; out[args.n_out + row] = up_sum2; }
        row = first_row + 3u;
        if (row < args.n_out) { out[row] = gate_sum3; out[args.n_out + row] = up_sum3; }
    }
}
inline float private_iq3_pair_grid_dot2(device const uchar *blk,
                                  thread const float4 *xx,
                                  device const ushort *grid, uint ib) {
    const float d = float(*((device const half *)blk));
    device const uchar *qs = blk + 2u + ib * 8u;
    const uint qhb = uint(blk[66u + ib]);
    device const uchar *signs = blk + 74u + ib * 4u;
    const uint scales = uint(blk[106u + ib / 2u]);
    const float db = d * float(1u + 2u * ((scales >> (4u * (ib & 1u))) & 15u));
    float sum = 0.0f;
    for (uint l = 0; l < 4u; l++) {
        const uint idx1 = uint(qs[2u * l]) | ((qhb << (8u - 2u * l)) & 256u);
        const uint idx2 = uint(qs[2u * l + 1u]) | ((qhb << (7u - 2u * l)) & 256u);
        const uint s = uint(signs[l]);
        const float4 sign1 = select(float4(1.0f), float4(-1.0f),
                                   (uint4(s) & uint4(1u, 2u, 4u, 8u)) != 0u);
        const float4 sign2 = select(float4(1.0f), float4(-1.0f),
                                   (uint4(s) & uint4(16u, 32u, 64u, 128u)) != 0u);
        sum += dot(qw3_iq3s_float_grid(grid[idx1]) * sign1, xx[2u * l]) +
               dot(qw3_iq3s_float_grid(grid[idx2]) * sign2, xx[2u * l + 1u]);
    }
    return db * sum;
}
kernel void private_iq3_pair_grid_fast2(constant qw3_matvec_q8_0_args &args,
                                       device const uchar *gate_weights,
                                       device const uchar *up_weights,
                                       device const float *x,
                                       device float *out,
                                       device const ushort *kgrid,
                                       threadgroup float *sh,
                                       uint group [[threadgroup_position_in_grid]],
                                       ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                       ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    device const ushort *grid = kgrid;
    const uint nr0 = 4u;
    const uint nsg = 2u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    float gate_sum0 = 0.0f, gate_sum1 = 0.0f, gate_sum2 = 0.0f, gate_sum3 = 0.0f;
    float up_sum0 = 0.0f, up_sum1 = 0.0f, up_sum2 = 0.0f, up_sum3 = 0.0f;
    uint nb32 = (args.n_in / 256u) * 8u;
    for (uint ib32 = uint(lane); ib32 < nb32; ib32 += 32u) {
        uint ibl = ib32 / 8u;
        uint ib = ib32 - ibl * 8u;
        float4 xx[8];
        device const float4 *x4 = (device const float4 *)(x + uint64_t(ib32) * 32ull);
        for (uint j = 0; j < 8u; j++) xx[j] = x4[j];
        uint row = first_row;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;
            gate_sum0 += private_iq3_pair_grid_dot2(gate_weights + off, xx, grid, ib);
            up_sum0 += private_iq3_pair_grid_dot2(up_weights + off, xx, grid, ib);
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;
            gate_sum1 += private_iq3_pair_grid_dot2(gate_weights + off, xx, grid, ib);
            up_sum1 += private_iq3_pair_grid_dot2(up_weights + off, xx, grid, ib);
        }
        row = first_row + 2u;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;
            gate_sum2 += private_iq3_pair_grid_dot2(gate_weights + off, xx, grid, ib);
            up_sum2 += private_iq3_pair_grid_dot2(up_weights + off, xx, grid, ib);
        }
        row = first_row + 3u;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;
            gate_sum3 += private_iq3_pair_grid_dot2(gate_weights + off, xx, grid, ib);
            up_sum3 += private_iq3_pair_grid_dot2(up_weights + off, xx, grid, ib);
        }
    }
    gate_sum0 = simd_sum(gate_sum0); up_sum0 = simd_sum(up_sum0);
    gate_sum1 = simd_sum(gate_sum1); up_sum1 = simd_sum(up_sum1);
    gate_sum2 = simd_sum(gate_sum2); up_sum2 = simd_sum(up_sum2);
    gate_sum3 = simd_sum(gate_sum3); up_sum3 = simd_sum(up_sum3);
    if (lane == 0) {
        uint row = first_row;
        if (row < args.n_out) { out[row] = gate_sum0; out[args.n_out + row] = up_sum0; }
        row = first_row + 1u;
        if (row < args.n_out) { out[row] = gate_sum1; out[args.n_out + row] = up_sum1; }
        row = first_row + 2u;
        if (row < args.n_out) { out[row] = gate_sum2; out[args.n_out + row] = up_sum2; }
        row = first_row + 3u;
        if (row < args.n_out) { out[row] = gate_sum3; out[args.n_out + row] = up_sum3; }
    }
}
