#include <metal_stdlib>
using namespace metal;
struct qw3_matvec_q8_0_args {uint n_in,n_out,row_bytes;};
constant float qw3_iq4nl_table[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f,
     -49.0f,  -35.0f, -22.0f, -10.0f,
       1.0f,   13.0f,  25.0f,  38.0f,
      53.0f,   69.0f,  89.0f, 113.0f
};
inline float qw3_iq4nl_val(uint q) {
    return qw3_iq4nl_table[q & 15u];
}
inline float qw3_iq4xs_dot8_vector(device const uchar *blk,
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
        lo += xl[2u * j] * float4(table[qlo.x], table[qlo.y], table[qlo.z], table[qlo.w]);
        hi += xl[2u * j + 1u] * float4(table[qhi.x], table[qhi.y], table[qhi.z], table[qhi.w]);
    }
    lo += hi;
    return float(d) * float(int(ls) - 32) * (lo.x + lo.y + lo.z + lo.w);
}
kernel void private_iq4_rows4_0(constant qw3_matvec_q8_0_args &args,
                                   device const uchar *weights,
                                   device const float *x,
                                   device float *out,
                                   threadgroup float *sh,
                                   uint group [[threadgroup_position_in_grid]],
                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]]) {
    if (simd_idx == 0u && lane < 16u) sh[lane] = qw3_iq4nl_val(uint(lane));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nr0 = 4u;
    const uint nsg = 2u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    uint ix = uint(lane) >> 4u;
    uint it = uint(lane) & 15u;
    uint ib = it >> 1u;
    uint il = it & 1u;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = ix; b < n_blocks; b += 2u) {
        device const float *xg = x + uint64_t(b) * 256ull +
                                 uint64_t(ib) * 32ull + uint64_t(il) * 8ull;
        device const float4 *x4 = (device const float4 *)xg;
        float4 xl[4] = {x4[0], x4[4], x4[1], x4[5]};
        {
            uint row = first_row + 0u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum0 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 1u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum1 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 2u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum2 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 3u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum3 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2);
    sum3 = simd_sum(sum3);
    if (lane == 0) {
        { uint row = first_row + 0u;
            if (row < args.n_out) {
                out[row] = sum0;
            }
        }
        { uint row = first_row + 1u;
            if (row < args.n_out) {
                out[row] = sum1;
            }
        }
        { uint row = first_row + 2u;
            if (row < args.n_out) {
                out[row] = sum2;
            }
        }
        { uint row = first_row + 3u;
            if (row < args.n_out) {
                out[row] = sum3;
            }
        }
    }
}
kernel void private_iq4_rows4_1(
                                   constant qw3_matvec_q8_0_args &args,
                                   device const uchar *gate_weights,
                                   device const uchar *up_weights,
                                   device const float *x,
                                   device float *out,
                                   threadgroup float *sh,
                                   uint group [[threadgroup_position_in_grid]],
                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]]) {
    if (simd_idx == 0u && lane < 16u) sh[lane] = qw3_iq4nl_val(uint(lane));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nr0 = 4u;
    const uint nsg = 2u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    uint ix = uint(lane) >> 4u;
    uint it = uint(lane) & 15u;
    uint ib = it >> 1u;
    uint il = it & 1u;
    float gate0 = 0.0f;
    float gate1 = 0.0f;
    float gate2 = 0.0f;
    float gate3 = 0.0f;
    float up0 = 0.0f;
    float up1 = 0.0f;
    float up2 = 0.0f;
    float up3 = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = ix; b < n_blocks; b += 2u) {
        device const float *xg = x + uint64_t(b) * 256ull +
                                 uint64_t(ib) * 32ull + uint64_t(il) * 8ull;
        device const float4 *x4 = (device const float4 *)xg;
        float4 xl[4] = {x4[0], x4[4], x4[1], x4[5]};
        {
            uint row = first_row + 0u;
            if (row < args.n_out) {
                uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                gate0 += qw3_iq4xs_dot8_vector(gate_weights + off, xl, sh, ib, il);
                up0 += qw3_iq4xs_dot8_vector(up_weights + off, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 1u;
            if (row < args.n_out) {
                uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                gate1 += qw3_iq4xs_dot8_vector(gate_weights + off, xl, sh, ib, il);
                up1 += qw3_iq4xs_dot8_vector(up_weights + off, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 2u;
            if (row < args.n_out) {
                uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                gate2 += qw3_iq4xs_dot8_vector(gate_weights + off, xl, sh, ib, il);
                up2 += qw3_iq4xs_dot8_vector(up_weights + off, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 3u;
            if (row < args.n_out) {
                uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                gate3 += qw3_iq4xs_dot8_vector(gate_weights + off, xl, sh, ib, il);
                up3 += qw3_iq4xs_dot8_vector(up_weights + off, xl, sh, ib, il);
            }
        }
    }
    gate0 = simd_sum(gate0);
    gate1 = simd_sum(gate1);
    gate2 = simd_sum(gate2);
    gate3 = simd_sum(gate3);
    up0 = simd_sum(up0);
    up1 = simd_sum(up1);
    up2 = simd_sum(up2);
    up3 = simd_sum(up3);
    if (lane == 0) {
        { uint row = first_row + 0u;
            if (row < args.n_out) {
                out[row] = gate0;
                out[args.n_out + row] = up0;
            }
        }
        { uint row = first_row + 1u;
            if (row < args.n_out) {
                out[row] = gate1;
                out[args.n_out + row] = up1;
            }
        }
        { uint row = first_row + 2u;
            if (row < args.n_out) {
                out[row] = gate2;
                out[args.n_out + row] = up2;
            }
        }
        { uint row = first_row + 3u;
            if (row < args.n_out) {
                out[row] = gate3;
                out[args.n_out + row] = up3;
            }
        }
    }
}
kernel void private_iq4_rows4_2(constant qw3_matvec_q8_0_args &args,
                                          constant float &scale,
                                          device const uchar *weights,
                                          device const float *x,
                                          device float *x0,
                                          threadgroup float *sh,
                                          uint group [[threadgroup_position_in_grid]],
                                          ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                          ushort lane [[thread_index_in_simdgroup]]) {
    if (simd_idx == 0u && lane < 16u) sh[lane] = qw3_iq4nl_val(uint(lane));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nr0 = 4u;
    const uint nsg = 2u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    uint ix = uint(lane) >> 4u;
    uint it = uint(lane) & 15u;
    uint ib = it >> 1u;
    uint il = it & 1u;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = ix; b < n_blocks; b += 2u) {
        device const float *xg = x + uint64_t(b) * 256ull + uint64_t(ib) * 32ull + uint64_t(il) * 8ull;
        device const float4 *x4 = (device const float4 *)xg;
        float4 xl[4] = {x4[0], x4[4], x4[1], x4[5]};
        {
            uint row = first_row + 0u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum0 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 1u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum1 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 2u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum2 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 3u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum3 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2);
    sum3 = simd_sum(sum3);
    if (lane == 0) {
        { uint row = first_row + 0u;
            if (row < args.n_out) {
                x0[row] = x0[row] + sum0 * scale;
            }
        }
        { uint row = first_row + 1u;
            if (row < args.n_out) {
                x0[row] = x0[row] + sum1 * scale;
            }
        }
        { uint row = first_row + 2u;
            if (row < args.n_out) {
                x0[row] = x0[row] + sum2 * scale;
            }
        }
        { uint row = first_row + 3u;
            if (row < args.n_out) {
                x0[row] = x0[row] + sum3 * scale;
            }
        }
    }
}
kernel void private_iq4_rows8_0(constant qw3_matvec_q8_0_args &args,
                                   device const uchar *weights,
                                   device const float *x,
                                   device float *out,
                                   threadgroup float *sh,
                                   uint group [[threadgroup_position_in_grid]],
                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]]) {
    if (simd_idx == 0u && lane < 16u) sh[lane] = qw3_iq4nl_val(uint(lane));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nr0 = 8u;
    const uint nsg = 2u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    uint ix = uint(lane) >> 4u;
    uint it = uint(lane) & 15u;
    uint ib = it >> 1u;
    uint il = it & 1u;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = ix; b < n_blocks; b += 2u) {
        device const float *xg = x + uint64_t(b) * 256ull +
                                 uint64_t(ib) * 32ull + uint64_t(il) * 8ull;
        device const float4 *x4 = (device const float4 *)xg;
        float4 xl[4] = {x4[0], x4[4], x4[1], x4[5]};
        {
            uint row = first_row + 0u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum0 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 1u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum1 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 2u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum2 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 3u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum3 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 4u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum4 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 5u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum5 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 6u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum6 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 7u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum7 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2);
    sum3 = simd_sum(sum3);
    sum4 = simd_sum(sum4);
    sum5 = simd_sum(sum5);
    sum6 = simd_sum(sum6);
    sum7 = simd_sum(sum7);
    if (lane == 0) {
        { uint row = first_row + 0u;
            if (row < args.n_out) {
                out[row] = sum0;
            }
        }
        { uint row = first_row + 1u;
            if (row < args.n_out) {
                out[row] = sum1;
            }
        }
        { uint row = first_row + 2u;
            if (row < args.n_out) {
                out[row] = sum2;
            }
        }
        { uint row = first_row + 3u;
            if (row < args.n_out) {
                out[row] = sum3;
            }
        }
        { uint row = first_row + 4u;
            if (row < args.n_out) {
                out[row] = sum4;
            }
        }
        { uint row = first_row + 5u;
            if (row < args.n_out) {
                out[row] = sum5;
            }
        }
        { uint row = first_row + 6u;
            if (row < args.n_out) {
                out[row] = sum6;
            }
        }
        { uint row = first_row + 7u;
            if (row < args.n_out) {
                out[row] = sum7;
            }
        }
    }
}
kernel void private_iq4_rows8_1(
                                   constant qw3_matvec_q8_0_args &args,
                                   device const uchar *gate_weights,
                                   device const uchar *up_weights,
                                   device const float *x,
                                   device float *out,
                                   threadgroup float *sh,
                                   uint group [[threadgroup_position_in_grid]],
                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]]) {
    if (simd_idx == 0u && lane < 16u) sh[lane] = qw3_iq4nl_val(uint(lane));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nr0 = 8u;
    const uint nsg = 2u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    uint ix = uint(lane) >> 4u;
    uint it = uint(lane) & 15u;
    uint ib = it >> 1u;
    uint il = it & 1u;
    float gate0 = 0.0f;
    float gate1 = 0.0f;
    float gate2 = 0.0f;
    float gate3 = 0.0f;
    float gate4 = 0.0f;
    float gate5 = 0.0f;
    float gate6 = 0.0f;
    float gate7 = 0.0f;
    float up0 = 0.0f;
    float up1 = 0.0f;
    float up2 = 0.0f;
    float up3 = 0.0f;
    float up4 = 0.0f;
    float up5 = 0.0f;
    float up6 = 0.0f;
    float up7 = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = ix; b < n_blocks; b += 2u) {
        device const float *xg = x + uint64_t(b) * 256ull +
                                 uint64_t(ib) * 32ull + uint64_t(il) * 8ull;
        device const float4 *x4 = (device const float4 *)xg;
        float4 xl[4] = {x4[0], x4[4], x4[1], x4[5]};
        {
            uint row = first_row + 0u;
            if (row < args.n_out) {
                uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                gate0 += qw3_iq4xs_dot8_vector(gate_weights + off, xl, sh, ib, il);
                up0 += qw3_iq4xs_dot8_vector(up_weights + off, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 1u;
            if (row < args.n_out) {
                uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                gate1 += qw3_iq4xs_dot8_vector(gate_weights + off, xl, sh, ib, il);
                up1 += qw3_iq4xs_dot8_vector(up_weights + off, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 2u;
            if (row < args.n_out) {
                uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                gate2 += qw3_iq4xs_dot8_vector(gate_weights + off, xl, sh, ib, il);
                up2 += qw3_iq4xs_dot8_vector(up_weights + off, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 3u;
            if (row < args.n_out) {
                uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                gate3 += qw3_iq4xs_dot8_vector(gate_weights + off, xl, sh, ib, il);
                up3 += qw3_iq4xs_dot8_vector(up_weights + off, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 4u;
            if (row < args.n_out) {
                uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                gate4 += qw3_iq4xs_dot8_vector(gate_weights + off, xl, sh, ib, il);
                up4 += qw3_iq4xs_dot8_vector(up_weights + off, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 5u;
            if (row < args.n_out) {
                uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                gate5 += qw3_iq4xs_dot8_vector(gate_weights + off, xl, sh, ib, il);
                up5 += qw3_iq4xs_dot8_vector(up_weights + off, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 6u;
            if (row < args.n_out) {
                uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                gate6 += qw3_iq4xs_dot8_vector(gate_weights + off, xl, sh, ib, il);
                up6 += qw3_iq4xs_dot8_vector(up_weights + off, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 7u;
            if (row < args.n_out) {
                uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                gate7 += qw3_iq4xs_dot8_vector(gate_weights + off, xl, sh, ib, il);
                up7 += qw3_iq4xs_dot8_vector(up_weights + off, xl, sh, ib, il);
            }
        }
    }
    gate0 = simd_sum(gate0);
    gate1 = simd_sum(gate1);
    gate2 = simd_sum(gate2);
    gate3 = simd_sum(gate3);
    gate4 = simd_sum(gate4);
    gate5 = simd_sum(gate5);
    gate6 = simd_sum(gate6);
    gate7 = simd_sum(gate7);
    up0 = simd_sum(up0);
    up1 = simd_sum(up1);
    up2 = simd_sum(up2);
    up3 = simd_sum(up3);
    up4 = simd_sum(up4);
    up5 = simd_sum(up5);
    up6 = simd_sum(up6);
    up7 = simd_sum(up7);
    if (lane == 0) {
        { uint row = first_row + 0u;
            if (row < args.n_out) {
                out[row] = gate0;
                out[args.n_out + row] = up0;
            }
        }
        { uint row = first_row + 1u;
            if (row < args.n_out) {
                out[row] = gate1;
                out[args.n_out + row] = up1;
            }
        }
        { uint row = first_row + 2u;
            if (row < args.n_out) {
                out[row] = gate2;
                out[args.n_out + row] = up2;
            }
        }
        { uint row = first_row + 3u;
            if (row < args.n_out) {
                out[row] = gate3;
                out[args.n_out + row] = up3;
            }
        }
        { uint row = first_row + 4u;
            if (row < args.n_out) {
                out[row] = gate4;
                out[args.n_out + row] = up4;
            }
        }
        { uint row = first_row + 5u;
            if (row < args.n_out) {
                out[row] = gate5;
                out[args.n_out + row] = up5;
            }
        }
        { uint row = first_row + 6u;
            if (row < args.n_out) {
                out[row] = gate6;
                out[args.n_out + row] = up6;
            }
        }
        { uint row = first_row + 7u;
            if (row < args.n_out) {
                out[row] = gate7;
                out[args.n_out + row] = up7;
            }
        }
    }
}
kernel void private_iq4_rows8_2(constant qw3_matvec_q8_0_args &args,
                                          constant float &scale,
                                          device const uchar *weights,
                                          device const float *x,
                                          device float *x0,
                                          threadgroup float *sh,
                                          uint group [[threadgroup_position_in_grid]],
                                          ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                          ushort lane [[thread_index_in_simdgroup]]) {
    if (simd_idx == 0u && lane < 16u) sh[lane] = qw3_iq4nl_val(uint(lane));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nr0 = 8u;
    const uint nsg = 2u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    uint ix = uint(lane) >> 4u;
    uint it = uint(lane) & 15u;
    uint ib = it >> 1u;
    uint il = it & 1u;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = ix; b < n_blocks; b += 2u) {
        device const float *xg = x + uint64_t(b) * 256ull + uint64_t(ib) * 32ull + uint64_t(il) * 8ull;
        device const float4 *x4 = (device const float4 *)xg;
        float4 xl[4] = {x4[0], x4[4], x4[1], x4[5]};
        {
            uint row = first_row + 0u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum0 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 1u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum1 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 2u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum2 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 3u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum3 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 4u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum4 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 5u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum5 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 6u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum6 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
        {
            uint row = first_row + 7u;
            if (row < args.n_out) {
                device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
                sum7 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
            }
        }
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2);
    sum3 = simd_sum(sum3);
    sum4 = simd_sum(sum4);
    sum5 = simd_sum(sum5);
    sum6 = simd_sum(sum6);
    sum7 = simd_sum(sum7);
    if (lane == 0) {
        { uint row = first_row + 0u;
            if (row < args.n_out) {
                x0[row] = x0[row] + sum0 * scale;
            }
        }
        { uint row = first_row + 1u;
            if (row < args.n_out) {
                x0[row] = x0[row] + sum1 * scale;
            }
        }
        { uint row = first_row + 2u;
            if (row < args.n_out) {
                x0[row] = x0[row] + sum2 * scale;
            }
        }
        { uint row = first_row + 3u;
            if (row < args.n_out) {
                x0[row] = x0[row] + sum3 * scale;
            }
        }
        { uint row = first_row + 4u;
            if (row < args.n_out) {
                x0[row] = x0[row] + sum4 * scale;
            }
        }
        { uint row = first_row + 5u;
            if (row < args.n_out) {
                x0[row] = x0[row] + sum5 * scale;
            }
        }
        { uint row = first_row + 6u;
            if (row < args.n_out) {
                x0[row] = x0[row] + sum6 * scale;
            }
        }
        { uint row = first_row + 7u;
            if (row < args.n_out) {
                x0[row] = x0[row] + sum7 * scale;
            }
        }
    }
}
