#include <metal_stdlib>
using namespace metal;
struct qw3_matvec_q8_0_args {uint n_in,n_out,row_bytes;};
inline float private_iq4_lut1(uint q) {
    const uint word = (q & 8u) ? ((q & 4u) ? 0x71594535u : 0x26190d01u)
                               : ((q & 4u) ? 0xf6eaddcfu : 0xbfad9881u);
    const int value = int(word << ((3u - (q & 3u)) * 8u)) >> 24;
    return float(value);
}
inline float private_iq4_dot1(device const uchar *blk,
                                  thread const float4 *xl,
                                  threadgroup const float *table,
                                  uint ib, uint il) {
    const half d = *((device const half *)blk);
    const ushort scales_h = *((device const ushort *)(blk + 2));
    device const uchar *scales_l = blk + 4;
    device const uint *qs = (device const uint *)(blk + 8 + ib * 16u + il * 8u);
    const uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) |
                    (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);
    float4 lo = float4(0.0f), hi = float4(0.0f);
    for (uint j = 0; j < 2u; j++) {
        const uint q = qs[j];
        const uchar4 qlo = as_type<uchar4>(q & 0x0f0f0f0fu);
        const uchar4 qhi = as_type<uchar4>((q >> 4u) & 0x0f0f0f0fu);
        lo += xl[2u * j] * float4(private_iq4_lut1(uint(qlo.x)), private_iq4_lut1(uint(qlo.y)), private_iq4_lut1(uint(qlo.z)), private_iq4_lut1(uint(qlo.w)));
        hi += xl[2u * j + 1u] * float4(private_iq4_lut1(uint(qhi.x)), private_iq4_lut1(uint(qhi.y)), private_iq4_lut1(uint(qhi.z)), private_iq4_lut1(uint(qhi.w)));
    }
    lo += hi;
    return float(d) * float(int(ls) - 32) * (lo.x + lo.y + lo.z + lo.w);
}
kernel void private_iq4_packed1_0(constant qw3_matvec_q8_0_args &args,
                                   device const uchar *weights,
                                   device const float *x,
                                   device float *out,
                                   threadgroup float *sh,
                                   uint group [[threadgroup_position_in_grid]],
                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    const uint nr0 = 2u;
    const uint nsg = 2u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    uint ix = uint(lane) >> 4u;
    uint it = uint(lane) & 15u;
    uint ib = it >> 1u;
    uint il = it & 1u;
    float sum0 = 0.0f, sum1 = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = ix; b < n_blocks; b += 2u) {
        device const float *xg = x + uint64_t(b) * 256ull +
                                 uint64_t(ib) * 32ull + uint64_t(il) * 8ull;
        device const float4 *x4 = (device const float4 *)xg;
        float4 xl[4] = {x4[0], x4[4], x4[1], x4[5]};
        uint row = first_row;
        uint64_t block_off = uint64_t(b) * 136ull;
        if (row < args.n_out)
            sum0 += private_iq4_dot1(weights +
                uint64_t(row) * uint64_t(args.row_bytes) + block_off,
                xl, sh, ib, il);
        row = first_row + 1u;
        if (row < args.n_out)
            sum1 += private_iq4_dot1(weights +
                uint64_t(row) * uint64_t(args.row_bytes) + block_off,
                xl, sh, ib, il);
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (lane == 0) {
        uint row = first_row;
        if (row < args.n_out) out[row] = sum0;
        row = first_row + 1u;
        if (row < args.n_out) out[row] = sum1;
    }
}
kernel void private_iq4_packed1_1(
                                   constant qw3_matvec_q8_0_args &args,
                                   device const uchar *gate_weights,
                                   device const uchar *up_weights,
                                   device const float *x,
                                   device float *out,
                                   threadgroup float *sh,
                                   uint group [[threadgroup_position_in_grid]],
                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    const uint nr0 = 2u;
    const uint nsg = 2u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    uint ix = uint(lane) >> 4u;
    uint it = uint(lane) & 15u;
    uint ib = it >> 1u;
    uint il = it & 1u;
    float gate0 = 0.0f, gate1 = 0.0f;
    float up0 = 0.0f, up1 = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = ix; b < n_blocks; b += 2u) {
        device const float *xg = x + uint64_t(b) * 256ull +
                                 uint64_t(ib) * 32ull + uint64_t(il) * 8ull;
        device const float4 *x4 = (device const float4 *)xg;
        float4 xl[4] = {x4[0], x4[4], x4[1], x4[5]};
        uint row = first_row;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) +
                           uint64_t(b) * 136ull;
            gate0 += private_iq4_dot1(gate_weights + off, xl, sh, ib, il);
            up0 += private_iq4_dot1(up_weights + off, xl, sh, ib, il);
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) +
                           uint64_t(b) * 136ull;
            gate1 += private_iq4_dot1(gate_weights + off, xl, sh, ib, il);
            up1 += private_iq4_dot1(up_weights + off, xl, sh, ib, il);
        }
    }
    gate0 = simd_sum(gate0);
    gate1 = simd_sum(gate1);
    up0 = simd_sum(up0);
    up1 = simd_sum(up1);
    if (lane == 0) {
        uint row = first_row;
        if (row < args.n_out) {
            out[row] = gate0;
            out[args.n_out + row] = up0;
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            out[row] = gate1;
            out[args.n_out + row] = up1;
        }
    }
}
kernel void private_iq4_packed1_2(constant qw3_matvec_q8_0_args &args,
                                          constant float &scale,
                                          device const uchar *weights,
                                          device const float *x,
                                          device float *x0,
                                          threadgroup float *sh,
                                          uint group [[threadgroup_position_in_grid]],
                                          ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                          ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    const uint nr0 = 2u;
    const uint nsg = 2u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    uint ix = uint(lane) >> 4u;
    uint it = uint(lane) & 15u;
    uint ib = it >> 1u;
    uint il = it & 1u;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = ix; b < n_blocks; b += 2u) {
        device const float *xg = x + uint64_t(b) * 256ull + uint64_t(ib) * 32ull + uint64_t(il) * 8ull;
        device const float4 *x4 = (device const float4 *)xg;
        float4 xl[4] = {x4[0], x4[4], x4[1], x4[5]};
        uint row = first_row;
        if (row < args.n_out) {
            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
            sum0 += private_iq4_dot1(blk, xl, sh, ib, il);
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
            sum1 += private_iq4_dot1(blk, xl, sh, ib, il);
        }
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (lane == 0) {
        uint row = first_row;
        if (row < args.n_out) x0[row] = x0[row] + sum0 * scale;
        row = first_row + 1u;
        if (row < args.n_out) x0[row] = x0[row] + sum1 * scale;
    }
}
inline float private_iq4_lut2(uint q) {
    const uint word = (q & 8u) ? ((q & 4u) ? 0xf1d9c5b5u : 0xa6998d81u)
                               : ((q & 4u) ? 0x766a5d4fu : 0x3f2d1801u);
    const int value = int((word >> ((q & 3u) * 8u)) & 255u) - 128;
    return float(value);
}
inline float private_iq4_dot2(device const uchar *blk,
                                  thread const float4 *xl,
                                  threadgroup const float *table,
                                  uint ib, uint il) {
    const half d = *((device const half *)blk);
    const ushort scales_h = *((device const ushort *)(blk + 2));
    device const uchar *scales_l = blk + 4;
    device const uint *qs = (device const uint *)(blk + 8 + ib * 16u + il * 8u);
    const uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) |
                    (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);
    float4 lo = float4(0.0f), hi = float4(0.0f);
    for (uint j = 0; j < 2u; j++) {
        const uint q = qs[j];
        const uchar4 qlo = as_type<uchar4>(q & 0x0f0f0f0fu);
        const uchar4 qhi = as_type<uchar4>((q >> 4u) & 0x0f0f0f0fu);
        lo += xl[2u * j] * float4(private_iq4_lut2(uint(qlo.x)), private_iq4_lut2(uint(qlo.y)), private_iq4_lut2(uint(qlo.z)), private_iq4_lut2(uint(qlo.w)));
        hi += xl[2u * j + 1u] * float4(private_iq4_lut2(uint(qhi.x)), private_iq4_lut2(uint(qhi.y)), private_iq4_lut2(uint(qhi.z)), private_iq4_lut2(uint(qhi.w)));
    }
    lo += hi;
    return float(d) * float(int(ls) - 32) * (lo.x + lo.y + lo.z + lo.w);
}
kernel void private_iq4_packed2_0(constant qw3_matvec_q8_0_args &args,
                                   device const uchar *weights,
                                   device const float *x,
                                   device float *out,
                                   threadgroup float *sh,
                                   uint group [[threadgroup_position_in_grid]],
                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    const uint nr0 = 2u;
    const uint nsg = 2u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    uint ix = uint(lane) >> 4u;
    uint it = uint(lane) & 15u;
    uint ib = it >> 1u;
    uint il = it & 1u;
    float sum0 = 0.0f, sum1 = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = ix; b < n_blocks; b += 2u) {
        device const float *xg = x + uint64_t(b) * 256ull +
                                 uint64_t(ib) * 32ull + uint64_t(il) * 8ull;
        device const float4 *x4 = (device const float4 *)xg;
        float4 xl[4] = {x4[0], x4[4], x4[1], x4[5]};
        uint row = first_row;
        uint64_t block_off = uint64_t(b) * 136ull;
        if (row < args.n_out)
            sum0 += private_iq4_dot2(weights +
                uint64_t(row) * uint64_t(args.row_bytes) + block_off,
                xl, sh, ib, il);
        row = first_row + 1u;
        if (row < args.n_out)
            sum1 += private_iq4_dot2(weights +
                uint64_t(row) * uint64_t(args.row_bytes) + block_off,
                xl, sh, ib, il);
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (lane == 0) {
        uint row = first_row;
        if (row < args.n_out) out[row] = sum0;
        row = first_row + 1u;
        if (row < args.n_out) out[row] = sum1;
    }
}
kernel void private_iq4_packed2_1(
                                   constant qw3_matvec_q8_0_args &args,
                                   device const uchar *gate_weights,
                                   device const uchar *up_weights,
                                   device const float *x,
                                   device float *out,
                                   threadgroup float *sh,
                                   uint group [[threadgroup_position_in_grid]],
                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    const uint nr0 = 2u;
    const uint nsg = 2u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    uint ix = uint(lane) >> 4u;
    uint it = uint(lane) & 15u;
    uint ib = it >> 1u;
    uint il = it & 1u;
    float gate0 = 0.0f, gate1 = 0.0f;
    float up0 = 0.0f, up1 = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = ix; b < n_blocks; b += 2u) {
        device const float *xg = x + uint64_t(b) * 256ull +
                                 uint64_t(ib) * 32ull + uint64_t(il) * 8ull;
        device const float4 *x4 = (device const float4 *)xg;
        float4 xl[4] = {x4[0], x4[4], x4[1], x4[5]};
        uint row = first_row;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) +
                           uint64_t(b) * 136ull;
            gate0 += private_iq4_dot2(gate_weights + off, xl, sh, ib, il);
            up0 += private_iq4_dot2(up_weights + off, xl, sh, ib, il);
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) +
                           uint64_t(b) * 136ull;
            gate1 += private_iq4_dot2(gate_weights + off, xl, sh, ib, il);
            up1 += private_iq4_dot2(up_weights + off, xl, sh, ib, il);
        }
    }
    gate0 = simd_sum(gate0);
    gate1 = simd_sum(gate1);
    up0 = simd_sum(up0);
    up1 = simd_sum(up1);
    if (lane == 0) {
        uint row = first_row;
        if (row < args.n_out) {
            out[row] = gate0;
            out[args.n_out + row] = up0;
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            out[row] = gate1;
            out[args.n_out + row] = up1;
        }
    }
}
kernel void private_iq4_packed2_2(constant qw3_matvec_q8_0_args &args,
                                          constant float &scale,
                                          device const uchar *weights,
                                          device const float *x,
                                          device float *x0,
                                          threadgroup float *sh,
                                          uint group [[threadgroup_position_in_grid]],
                                          ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                          ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    const uint nr0 = 2u;
    const uint nsg = 2u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    uint ix = uint(lane) >> 4u;
    uint it = uint(lane) & 15u;
    uint ib = it >> 1u;
    uint il = it & 1u;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = ix; b < n_blocks; b += 2u) {
        device const float *xg = x + uint64_t(b) * 256ull + uint64_t(ib) * 32ull + uint64_t(il) * 8ull;
        device const float4 *x4 = (device const float4 *)xg;
        float4 xl[4] = {x4[0], x4[4], x4[1], x4[5]};
        uint row = first_row;
        if (row < args.n_out) {
            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
            sum0 += private_iq4_dot2(blk, xl, sh, ib, il);
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
            sum1 += private_iq4_dot2(blk, xl, sh, ib, il);
        }
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (lane == 0) {
        uint row = first_row;
        if (row < args.n_out) x0[row] = x0[row] + sum0 * scale;
        row = first_row + 1u;
        if (row < args.n_out) x0[row] = x0[row] + sum1 * scale;
    }
}
