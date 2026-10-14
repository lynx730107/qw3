inline uchar4 private_iq2s_grid4(ushort packed, uint half_index) {
    constexpr uchar values[4] = { 8, 25, 43, 0 };
    const uint q = uint(packed) >> (8u * half_index);
    return uchar4(values[q & 3u], values[(q >> 2u) & 3u],
                  values[(q >> 4u) & 3u], values[(q >> 6u) & 3u]);
}
template<ushort Kind>
inline float private_iq2s_grid_dot1(device const uchar *blk,
                                  thread const float4 *xx,
                                  threadgroup const ushort *grid,
                                  threadgroup const uchar *signs, uint ib) {
    const float d = float(*((device const half *)blk));
    device const uchar *qs = blk + 2u;
    uint packed = 0u, scale = 0u;
    if (Kind == 0u) {
        device const uchar *sp = qs + 8u * ib + 4u;
        packed = uint(sp[0]) | (uint(sp[1]) << 8u) |
                 (uint(sp[2]) << 16u) | (uint(sp[3]) << 24u);
    } else {
        scale = uint(blk[(Kind == 1u ? 66u : 74u) + ib]);
    }
    float sum = 0.0f;
    for (uint l = 0; l < 4u; l++) {
        uint idx, s;
        if (Kind == 0u) {
            idx = uint(qs[8u * ib + l]);
            s = uint(signs[(packed >> (7u * l)) & 127u]);
        } else if (Kind == 1u) {
            const uint off = 2u * (4u * ib + l);
            const uint q = uint(qs[off]) | (uint(qs[off + 1u]) << 8u);
            idx = q & 511u;
            s = uint(signs[q >> 9u]);
        } else {
            idx = uint(qs[4u * ib + l]) |
                  ((uint(blk[66u + ib]) << (8u - 2u * l)) & 0x300u);
            s = uint(blk[34u + 4u * ib + l]);
        }
        const float4 sign1 = select(float4(1.0f), float4(-1.0f),
                                   (uint4(s) & uint4(1u, 2u, 4u, 8u)) != 0u);
        const float4 sign2 = select(float4(1.0f), float4(-1.0f),
                                   (uint4(s) & uint4(16u, 32u, 64u, 128u)) != 0u);
        const float v = dot(float4(private_iq2s_grid4(grid[idx], 0u)) * sign1, xx[2u * l]) +
                        dot(float4(private_iq2s_grid4(grid[idx], 1u)) * sign2, xx[2u * l + 1u]);
        if (Kind == 0u) sum += v;
        else sum += (0.5f + float((scale >> (4u * (l >> 1u))) & 15u)) * v;
    }
    return d * 0.25f * (Kind == 0u ? (0.5f + float(packed >> 28u)) * sum : sum);
}
template<ushort Kind, ushort Simdgroups = 2>
kernel void private_iq2s_grid_kernel1(
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
    constexpr uint NG = Kind == 0u ? 256u : Kind == 1u ? 512u : 1024u;
    constexpr uint NB = Kind == 0u ? 66u : Kind == 1u ? 74u : 82u;
    threadgroup ushort grid[NG];
    threadgroup uchar signs[128];
    const uint tid = uint(simd_idx) * 32u + uint(lane);
    for (uint i = tid; i < NG; i += Simdgroups * 32u)
        grid[i] = grid_table[i];
    if (Kind != 2u)
        for (uint i = tid; i < 128u; i += Simdgroups * 32u) signs[i] = sign_table[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row0 = (group * Simdgroups + uint(simd_idx)) * 2u;
    const uint row1 = row0 + 1u;
    float sum0 = 0.0f, sum1 = 0.0f;
    for (uint b32 = uint(lane); b32 < args.n_in / 32u; b32 += 32u) {
        const uint b = b32 / 8u, ib = b32 & 7u;
        device const float4 *x4 = (device const float4 *)(x + uint64_t(b32) * 32ull);
        float4 xx[8];
        for (uint j = 0; j < 8u; j++) xx[j] = x4[j];
        if (row0 < args.n_out)
            sum0 += private_iq2s_grid_dot1<Kind>(
                weights + uint64_t(row0) * args.row_bytes + uint64_t(b) * NB,
                xx, grid, signs, ib);
        if (row1 < args.n_out)
            sum1 += private_iq2s_grid_dot1<Kind>(
                weights + uint64_t(row1) * args.row_bytes + uint64_t(b) * NB,
                xx, grid, signs, ib);
    }
    sum0 = simd_sum(sum0); sum1 = simd_sum(sum1);
    if (lane == 0) {
        if (row0 < args.n_out) out[row0] = sum0;
        if (row1 < args.n_out) out[row1] = sum1;
    }
}
typedef decltype(private_iq2s_grid_kernel1<2, 8>) private_iq2s_grid_type1;
template [[host_name("private_iq2s_grid_fast1")]] kernel private_iq2s_grid_type1 private_iq2s_grid_kernel1<2, 8>;
template<ushort Kind>
inline float private_iq2s_grid_dot2(device const uchar *blk,
                                  thread const float4 *xx,
                                  device const ushort *grid,
                                  threadgroup const uchar *signs, uint ib) {
    const float d = float(*((device const half *)blk));
    device const uchar *qs = blk + 2u;
    uint packed = 0u, scale = 0u;
    if (Kind == 0u) {
        device const uchar *sp = qs + 8u * ib + 4u;
        packed = uint(sp[0]) | (uint(sp[1]) << 8u) |
                 (uint(sp[2]) << 16u) | (uint(sp[3]) << 24u);
    } else {
        scale = uint(blk[(Kind == 1u ? 66u : 74u) + ib]);
    }
    float sum = 0.0f;
    for (uint l = 0; l < 4u; l++) {
        uint idx, s;
        if (Kind == 0u) {
            idx = uint(qs[8u * ib + l]);
            s = uint(signs[(packed >> (7u * l)) & 127u]);
        } else if (Kind == 1u) {
            const uint off = 2u * (4u * ib + l);
            const uint q = uint(qs[off]) | (uint(qs[off + 1u]) << 8u);
            idx = q & 511u;
            s = uint(signs[q >> 9u]);
        } else {
            idx = uint(qs[4u * ib + l]) |
                  ((uint(blk[66u + ib]) << (8u - 2u * l)) & 0x300u);
            s = uint(blk[34u + 4u * ib + l]);
        }
        const float4 sign1 = select(float4(1.0f), float4(-1.0f),
                                   (uint4(s) & uint4(1u, 2u, 4u, 8u)) != 0u);
        const float4 sign2 = select(float4(1.0f), float4(-1.0f),
                                   (uint4(s) & uint4(16u, 32u, 64u, 128u)) != 0u);
        const float v = dot(float4(private_iq2s_grid4(grid[idx], 0u)) * sign1, xx[2u * l]) +
                        dot(float4(private_iq2s_grid4(grid[idx], 1u)) * sign2, xx[2u * l + 1u]);
        if (Kind == 0u) sum += v;
        else sum += (0.5f + float((scale >> (4u * (l >> 1u))) & 15u)) * v;
    }
    return d * 0.25f * (Kind == 0u ? (0.5f + float(packed >> 28u)) * sum : sum);
}
template<ushort Kind, ushort Simdgroups = 2>
kernel void private_iq2s_grid_kernel2(
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
    constexpr uint NG = Kind == 0u ? 256u : Kind == 1u ? 512u : 1024u;
    constexpr uint NB = Kind == 0u ? 66u : Kind == 1u ? 74u : 82u;
    device const ushort *grid = grid_table;
    threadgroup uchar signs[128];
    const uint tid = uint(simd_idx) * 32u + uint(lane);
    if (Kind != 2u)
        for (uint i = tid; i < 128u; i += Simdgroups * 32u) signs[i] = sign_table[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row0 = (group * Simdgroups + uint(simd_idx)) * 2u;
    const uint row1 = row0 + 1u;
    float sum0 = 0.0f, sum1 = 0.0f;
    for (uint b32 = uint(lane); b32 < args.n_in / 32u; b32 += 32u) {
        const uint b = b32 / 8u, ib = b32 & 7u;
        device const float4 *x4 = (device const float4 *)(x + uint64_t(b32) * 32ull);
        float4 xx[8];
        for (uint j = 0; j < 8u; j++) xx[j] = x4[j];
        if (row0 < args.n_out)
            sum0 += private_iq2s_grid_dot2<Kind>(
                weights + uint64_t(row0) * args.row_bytes + uint64_t(b) * NB,
                xx, grid, signs, ib);
        if (row1 < args.n_out)
            sum1 += private_iq2s_grid_dot2<Kind>(
                weights + uint64_t(row1) * args.row_bytes + uint64_t(b) * NB,
                xx, grid, signs, ib);
    }
    sum0 = simd_sum(sum0); sum1 = simd_sum(sum1);
    if (lane == 0) {
        if (row0 < args.n_out) out[row0] = sum0;
        if (row1 < args.n_out) out[row1] = sum1;
    }
}
typedef decltype(private_iq2s_grid_kernel2<2, 8>) private_iq2s_grid_type2;
template [[host_name("private_iq2s_grid_fast2")]] kernel private_iq2s_grid_type2 private_iq2s_grid_kernel2<2, 8>;
