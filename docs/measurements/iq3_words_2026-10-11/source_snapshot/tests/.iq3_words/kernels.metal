#include <metal_stdlib>
using namespace metal;
struct qw3_matvec_q8_0_args {uint n_in,n_out,row_bytes;};
inline float qw3_iq3s_grid_value(ushort packed, uint j) {
    return float(2u * ((uint(packed) >> (3u * j)) & 7u) + 1u);
}
template<typename Grid>
inline float private_iq3_words_dot1(device const uchar *blk,
                                  thread const float *xx,
                                  threadgroup const Grid *grid,
                                  uint ib) {
    const half d = *((device const half *)blk);
    device const uchar *qs = blk + 2 + ib * 8u;
    const uchar qhb = blk[66u + ib];
    device const uchar *signs = blk + 74 + ib * 4u;
    const uint scale = uint(blk[106u + ib / 2u]);
    const float db = float(d) * float(1u + 2u * ((scale >> (4u * (ib & 1u))) & 15u));
    float sum = 0.0f;
    for (uint l = 0; l < 4u; l++) {
        const uint packed_qs = uint(reinterpret_cast<device const ushort *>(qs)[l]);
        const uint idx1 = (packed_qs & 255u) | ((uint(qhb) << (8u - 2u * l)) & 256u);
        const uint idx2 = (packed_qs >> 8u) | ((uint(qhb) << (7u - 2u * l)) & 256u);
        const Grid g1 = grid[idx1], g2 = grid[idx2];
        const uchar s = signs[l];
        for (uint j = 0; j < 4u; j++) {
            const float sign1 = (uint(s) & (1u << j)) ? -1.0f : 1.0f;
            const float sign2 = (uint(s) & (1u << (j + 4u))) ? -1.0f : 1.0f;
            const float v1 = qw3_iq3s_grid_value(g1, j);
            const float v2 = qw3_iq3s_grid_value(g2, j);
            sum += db * v1 * sign1 * xx[8u * l + j];
            sum += db * v2 * sign2 * xx[8u * l + j + 4u];
        }
    }
    return sum;
}
kernel void private_iq3_words_fast1(constant qw3_matvec_q8_0_args &args,
                                  device const uchar *weights,
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
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    uint nb32 = (args.n_in / 256u) * 8u;
    for (uint ib32 = uint(lane); ib32 < nb32; ib32 += 32u) {
        uint ibl = ib32 / 8u;
        uint ib = ib32 - ibl * 8u;
        float xx[32];
        for (uint j = 0; j < 32u; j++) xx[j] = x[uint64_t(ib32) * 32ull + j];
        uint64_t block_off = uint64_t(ibl) * 110ull;
        uint row = first_row;
        if (row < args.n_out)
            sum0 += private_iq3_words_dot1(weights +
                uint64_t(row) * uint64_t(args.row_bytes) + block_off,
                xx, grid, ib);
        row = first_row + 1u;
        if (row < args.n_out)
            sum1 += private_iq3_words_dot1(weights +
                uint64_t(row) * uint64_t(args.row_bytes) + block_off,
                xx, grid, ib);
        row = first_row + 2u;
        if (row < args.n_out)
            sum2 += private_iq3_words_dot1(weights +
                uint64_t(row) * uint64_t(args.row_bytes) + block_off,
                xx, grid, ib);
        row = first_row + 3u;
        if (row < args.n_out)
            sum3 += private_iq3_words_dot1(weights +
                uint64_t(row) * uint64_t(args.row_bytes) + block_off,
                xx, grid, ib);
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2);
    sum3 = simd_sum(sum3);
    if (lane == 0) {
        uint row = first_row;
        if (row < args.n_out) out[row] = sum0;
        row = first_row + 1u;
        if (row < args.n_out) out[row] = sum1;
        row = first_row + 2u;
        if (row < args.n_out) out[row] = sum2;
        row = first_row + 3u;
        if (row < args.n_out) out[row] = sum3;
    }
}
template<typename Grid>
inline float private_iq3_words_dot2(device const uchar *blk,
                                  thread const float *xx,
                                  threadgroup const Grid *grid,
                                  uint ib) {
    const half d = *((device const half *)blk);
    device const uchar *qs = blk + 2 + ib * 8u;
    const uchar qhb = blk[66u + ib];
    device const uchar *signs = blk + 74 + ib * 4u;
    const uint packed_signs0 = uint(reinterpret_cast<device const ushort *>(signs)[0]);
    const uint packed_signs1 = uint(reinterpret_cast<device const ushort *>(signs)[1]);
    const uint scale = uint(blk[106u + ib / 2u]);
    const float db = float(d) * float(1u + 2u * ((scale >> (4u * (ib & 1u))) & 15u));
    float sum = 0.0f;
    for (uint l = 0; l < 4u; l++) {
        const uint packed_qs = uint(reinterpret_cast<device const ushort *>(qs)[l]);
        const uint idx1 = (packed_qs & 255u) | ((uint(qhb) << (8u - 2u * l)) & 256u);
        const uint idx2 = (packed_qs >> 8u) | ((uint(qhb) << (7u - 2u * l)) & 256u);
        const Grid g1 = grid[idx1], g2 = grid[idx2];
        const uint packed_signs = l < 2u ? packed_signs0 : packed_signs1;
        const uchar s = uchar((packed_signs >> (8u * (l & 1u))) & 255u);
        for (uint j = 0; j < 4u; j++) {
            const float sign1 = (uint(s) & (1u << j)) ? -1.0f : 1.0f;
            const float sign2 = (uint(s) & (1u << (j + 4u))) ? -1.0f : 1.0f;
            const float v1 = qw3_iq3s_grid_value(g1, j);
            const float v2 = qw3_iq3s_grid_value(g2, j);
            sum += db * v1 * sign1 * xx[8u * l + j];
            sum += db * v2 * sign2 * xx[8u * l + j + 4u];
        }
    }
    return sum;
}
kernel void private_iq3_words_fast2(constant qw3_matvec_q8_0_args &args,
                                  device const uchar *weights,
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
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    uint nb32 = (args.n_in / 256u) * 8u;
    for (uint ib32 = uint(lane); ib32 < nb32; ib32 += 32u) {
        uint ibl = ib32 / 8u;
        uint ib = ib32 - ibl * 8u;
        float xx[32];
        for (uint j = 0; j < 32u; j++) xx[j] = x[uint64_t(ib32) * 32ull + j];
        uint64_t block_off = uint64_t(ibl) * 110ull;
        uint row = first_row;
        if (row < args.n_out)
            sum0 += private_iq3_words_dot2(weights +
                uint64_t(row) * uint64_t(args.row_bytes) + block_off,
                xx, grid, ib);
        row = first_row + 1u;
        if (row < args.n_out)
            sum1 += private_iq3_words_dot2(weights +
                uint64_t(row) * uint64_t(args.row_bytes) + block_off,
                xx, grid, ib);
        row = first_row + 2u;
        if (row < args.n_out)
            sum2 += private_iq3_words_dot2(weights +
                uint64_t(row) * uint64_t(args.row_bytes) + block_off,
                xx, grid, ib);
        row = first_row + 3u;
        if (row < args.n_out)
            sum3 += private_iq3_words_dot2(weights +
                uint64_t(row) * uint64_t(args.row_bytes) + block_off,
                xx, grid, ib);
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2);
    sum3 = simd_sum(sum3);
    if (lane == 0) {
        uint row = first_row;
        if (row < args.n_out) out[row] = sum0;
        row = first_row + 1u;
        if (row < args.n_out) out[row] = sum1;
        row = first_row + 2u;
        if (row < args.n_out) out[row] = sum2;
        row = first_row + 3u;
        if (row < args.n_out) out[row] = sum3;
    }
}
