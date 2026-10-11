#include <metal_stdlib>
using namespace metal;
struct qw3_matvec_q8_0_args {uint n_in,n_out,row_bytes;};
inline float4 qw3_iq3s_float_grid(ushort packed) {
    const uint g = uint(packed);
    return float4(2u * (g & 7u) + 1u,
                  2u * ((g >> 3u) & 7u) + 1u,
                  2u * ((g >> 6u) & 7u) + 1u,
                  2u * ((g >> 9u) & 7u) + 1u);
}
inline float qw3_iq3s_grid_value(ushort packed, uint j) {
    return float(2u * ((uint(packed) >> (3u * j)) & 7u) + 1u);
}
inline float qw3_iq3s_grid_value(float4 magnitudes, uint j) {
    return magnitudes[j];
}
template<typename Grid>
inline float private_iq3_layout_dot1_0(device const uchar *blk,
                                  thread const float *xx,
                                  threadgroup const Grid *grid,
                                  uint ib) {
    const half d = *((device const half *)blk);
    device const uchar *qs = blk + 16 + ib * 8u;
    const uchar qhb = blk[8u + ib];
    device const uchar *signs = blk + 80 + ib * 4u;
    const uint scale = uint(blk[4u + ib / 2u]);
    const float db = float(d) * float(1u + 2u * ((scale >> (4u * (ib & 1u))) & 15u));
    float sum = 0.0f;
    for (uint l = 0; l < 4u; l++) {
        const uint idx1 = uint(qs[2u * l]) | ((uint(qhb) << (8u - 2u * l)) & 256u);
        const uint idx2 = uint(qs[2u * l + 1u]) | ((uint(qhb) << (7u - 2u * l)) & 256u);
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
kernel void private_iq3_layout1_0(constant qw3_matvec_q8_0_args &args,
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
        uint64_t block_off = uint64_t(ibl) * 112ull;
        uint row = first_row;
        if (row < args.n_out)
            sum0 += private_iq3_layout_dot1_0(weights +
                uint64_t(row) * (uint64_t(args.n_in / 256u) * 112ull) + block_off,
                xx, grid, ib);
        row = first_row + 1u;
        if (row < args.n_out)
            sum1 += private_iq3_layout_dot1_0(weights +
                uint64_t(row) * (uint64_t(args.n_in / 256u) * 112ull) + block_off,
                xx, grid, ib);
        row = first_row + 2u;
        if (row < args.n_out)
            sum2 += private_iq3_layout_dot1_0(weights +
                uint64_t(row) * (uint64_t(args.n_in / 256u) * 112ull) + block_off,
                xx, grid, ib);
        row = first_row + 3u;
        if (row < args.n_out)
            sum3 += private_iq3_layout_dot1_0(weights +
                uint64_t(row) * (uint64_t(args.n_in / 256u) * 112ull) + block_off,
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
inline float private_iq3_layout_dot1_1(device const uchar *blk,
                                  thread const float4 *xx,
                                  threadgroup const float4 *grid, uint ib) {
    const float d = float(*((device const half *)blk));
    device const uchar *qs = blk + 16u + ib * 8u;
    const uint qhb = uint(blk[8u + ib]);
    device const uchar *signs = blk + 80u + ib * 4u;
    const uint scales = uint(blk[4u + ib / 2u]);
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
        sum += dot(grid[idx1] * sign1, xx[2u * l]) +
               dot(grid[idx2] * sign2, xx[2u * l + 1u]);
    }
    return db * sum;
}
kernel void private_iq3_layout1_1(constant qw3_matvec_q8_0_args &args,
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
    threadgroup float4 grid[512];
    const uint tid = uint(simd_idx) * 32u + uint(lane);
    for (uint i = tid; i < 512u; i += 64u) grid[i] = qw3_iq3s_float_grid(kgrid[i]);
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
            uint64_t off = uint64_t(row) * (uint64_t(args.n_in / 256u) * 112ull) + uint64_t(ibl) * 112ull;
            gate_sum0 += private_iq3_layout_dot1_1(gate_weights + off, xx, grid, ib);
            up_sum0 += private_iq3_layout_dot1_1(up_weights + off, xx, grid, ib);
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * (uint64_t(args.n_in / 256u) * 112ull) + uint64_t(ibl) * 112ull;
            gate_sum1 += private_iq3_layout_dot1_1(gate_weights + off, xx, grid, ib);
            up_sum1 += private_iq3_layout_dot1_1(up_weights + off, xx, grid, ib);
        }
        row = first_row + 2u;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * (uint64_t(args.n_in / 256u) * 112ull) + uint64_t(ibl) * 112ull;
            gate_sum2 += private_iq3_layout_dot1_1(gate_weights + off, xx, grid, ib);
            up_sum2 += private_iq3_layout_dot1_1(up_weights + off, xx, grid, ib);
        }
        row = first_row + 3u;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * (uint64_t(args.n_in / 256u) * 112ull) + uint64_t(ibl) * 112ull;
            gate_sum3 += private_iq3_layout_dot1_1(gate_weights + off, xx, grid, ib);
            up_sum3 += private_iq3_layout_dot1_1(up_weights + off, xx, grid, ib);
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
template<typename Grid>
inline float private_iq3_layout_dot2_0(device const uchar *blk,
                                  thread const float *xx,
                                  threadgroup const Grid *grid,
                                  uint ib) {
    const uint4 header = *((device const uint4 *)blk);
    const half d = as_type<half2>(header.x).x;
    const uint2 qs = *((device const uint2 *)(blk + 16u + ib * 8u));
    const uchar qhb = uchar(header[2u + (ib >> 2u)] >> (8u * (ib & 3u)));
    const uint signs = *((device const uint *)(blk + 80u + ib * 4u));
    const uint scale = (header.y >> (8u * (ib / 2u))) & 255u;
    const float db = float(d) * float(1u + 2u * ((scale >> (4u * (ib & 1u))) & 15u));
    float sum = 0.0f;
    for (uint l = 0; l < 4u; l++) {
        const uint idx1 = ((qs[l >> 1u] >> (16u * (l & 1u))) & 255u) | ((uint(qhb) << (8u - 2u * l)) & 256u);
        const uint idx2 = ((qs[l >> 1u] >> (16u * (l & 1u) + 8u)) & 255u) | ((uint(qhb) << (7u - 2u * l)) & 256u);
        const Grid g1 = grid[idx1], g2 = grid[idx2];
        const uchar s = uchar(signs >> (8u * l));
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
kernel void private_iq3_layout2_0(constant qw3_matvec_q8_0_args &args,
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
        uint64_t block_off = uint64_t(ibl) * 112ull;
        uint row = first_row;
        if (row < args.n_out)
            sum0 += private_iq3_layout_dot2_0(weights +
                uint64_t(row) * (uint64_t(args.n_in / 256u) * 112ull) + block_off,
                xx, grid, ib);
        row = first_row + 1u;
        if (row < args.n_out)
            sum1 += private_iq3_layout_dot2_0(weights +
                uint64_t(row) * (uint64_t(args.n_in / 256u) * 112ull) + block_off,
                xx, grid, ib);
        row = first_row + 2u;
        if (row < args.n_out)
            sum2 += private_iq3_layout_dot2_0(weights +
                uint64_t(row) * (uint64_t(args.n_in / 256u) * 112ull) + block_off,
                xx, grid, ib);
        row = first_row + 3u;
        if (row < args.n_out)
            sum3 += private_iq3_layout_dot2_0(weights +
                uint64_t(row) * (uint64_t(args.n_in / 256u) * 112ull) + block_off,
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
inline float private_iq3_layout_dot2_1(device const uchar *blk,
                                  thread const float4 *xx,
                                  threadgroup const float4 *grid, uint ib) {
    const uint4 header = *((device const uint4 *)blk);
    const float d = float(as_type<half2>(header.x).x);
    const uint2 qs = *((device const uint2 *)(blk + 16u + ib * 8u));
    const uchar qhb = uchar(header[2u + (ib >> 2u)] >> (8u * (ib & 3u)));
    const uint signs = *((device const uint *)(blk + 80u + ib * 4u));
    const uint scales = (header.y >> (8u * (ib / 2u))) & 255u;
    const float db = d * float(1u + 2u * ((scales >> (4u * (ib & 1u))) & 15u));
    float sum = 0.0f;
    for (uint l = 0; l < 4u; l++) {
        const uint idx1 = ((qs[l >> 1u] >> (16u * (l & 1u))) & 255u) | ((qhb << (8u - 2u * l)) & 256u);
        const uint idx2 = ((qs[l >> 1u] >> (16u * (l & 1u) + 8u)) & 255u) | ((qhb << (7u - 2u * l)) & 256u);
        const uint s = (signs >> (8u * l)) & 255u;
        const float4 sign1 = select(float4(1.0f), float4(-1.0f),
                                   (uint4(s) & uint4(1u, 2u, 4u, 8u)) != 0u);
        const float4 sign2 = select(float4(1.0f), float4(-1.0f),
                                   (uint4(s) & uint4(16u, 32u, 64u, 128u)) != 0u);
        sum += dot(grid[idx1] * sign1, xx[2u * l]) +
               dot(grid[idx2] * sign2, xx[2u * l + 1u]);
    }
    return db * sum;
}
kernel void private_iq3_layout2_1(constant qw3_matvec_q8_0_args &args,
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
    threadgroup float4 grid[512];
    const uint tid = uint(simd_idx) * 32u + uint(lane);
    for (uint i = tid; i < 512u; i += 64u) grid[i] = qw3_iq3s_float_grid(kgrid[i]);
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
            uint64_t off = uint64_t(row) * (uint64_t(args.n_in / 256u) * 112ull) + uint64_t(ibl) * 112ull;
            gate_sum0 += private_iq3_layout_dot2_1(gate_weights + off, xx, grid, ib);
            up_sum0 += private_iq3_layout_dot2_1(up_weights + off, xx, grid, ib);
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * (uint64_t(args.n_in / 256u) * 112ull) + uint64_t(ibl) * 112ull;
            gate_sum1 += private_iq3_layout_dot2_1(gate_weights + off, xx, grid, ib);
            up_sum1 += private_iq3_layout_dot2_1(up_weights + off, xx, grid, ib);
        }
        row = first_row + 2u;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * (uint64_t(args.n_in / 256u) * 112ull) + uint64_t(ibl) * 112ull;
            gate_sum2 += private_iq3_layout_dot2_1(gate_weights + off, xx, grid, ib);
            up_sum2 += private_iq3_layout_dot2_1(up_weights + off, xx, grid, ib);
        }
        row = first_row + 3u;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * (uint64_t(args.n_in / 256u) * 112ull) + uint64_t(ibl) * 112ull;
            gate_sum3 += private_iq3_layout_dot2_1(gate_weights + off, xx, grid, ib);
            up_sum3 += private_iq3_layout_dot2_1(up_weights + off, xx, grid, ib);
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
