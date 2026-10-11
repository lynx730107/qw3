/* All helpers and declarations come from the full original Metal source. */
kernel void private_iq4_share_fast1_0(constant qw3_matvec_q8_0_args &args,
                                   device const uchar *weights,
                                   device const float *x,
                                   device float *out,
                                   threadgroup float *sh,
                                   uint group [[threadgroup_position_in_grid]],
                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]]) {
    if (simd_idx == 0u && lane < 16u) sh[lane] = qw3_iq4nl_val(uint(lane));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nr0 = 2u;
    const uint nsg = 4u;
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
            sum0 += qw3_iq4xs_dot8_vector(weights +
                uint64_t(row) * uint64_t(args.row_bytes) + block_off,
                xl, sh, ib, il);
        row = first_row + 1u;
        if (row < args.n_out)
            sum1 += qw3_iq4xs_dot8_vector(weights +
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
kernel void private_iq4_share_fast1_1(
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
    const uint nr0 = 2u;
    const uint nsg = 4u;
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
            gate0 += qw3_iq4xs_dot8_vector(gate_weights + off, xl, sh, ib, il);
            up0 += qw3_iq4xs_dot8_vector(up_weights + off, xl, sh, ib, il);
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) +
                           uint64_t(b) * 136ull;
            gate1 += qw3_iq4xs_dot8_vector(gate_weights + off, xl, sh, ib, il);
            up1 += qw3_iq4xs_dot8_vector(up_weights + off, xl, sh, ib, il);
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
kernel void private_iq4_share_fast1_2(constant qw3_matvec_q8_0_args &args,
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
    const uint nr0 = 2u;
    const uint nsg = 4u;
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
            sum0 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
            sum1 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
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
kernel void private_iq4_share_fast2_0(constant qw3_matvec_q8_0_args &args,
                                   device const uchar *weights,
                                   device const float *x,
                                   device float *out,
                                   threadgroup float *sh,
                                   uint group [[threadgroup_position_in_grid]],
                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]]) {
    if (simd_idx == 0u && lane < 16u) sh[lane] = qw3_iq4nl_val(uint(lane));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nr0 = 2u;
    const uint nsg = 8u;
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
            sum0 += qw3_iq4xs_dot8_vector(weights +
                uint64_t(row) * uint64_t(args.row_bytes) + block_off,
                xl, sh, ib, il);
        row = first_row + 1u;
        if (row < args.n_out)
            sum1 += qw3_iq4xs_dot8_vector(weights +
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
kernel void private_iq4_share_fast2_1(
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
    const uint nr0 = 2u;
    const uint nsg = 8u;
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
            gate0 += qw3_iq4xs_dot8_vector(gate_weights + off, xl, sh, ib, il);
            up0 += qw3_iq4xs_dot8_vector(up_weights + off, xl, sh, ib, il);
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) +
                           uint64_t(b) * 136ull;
            gate1 += qw3_iq4xs_dot8_vector(gate_weights + off, xl, sh, ib, il);
            up1 += qw3_iq4xs_dot8_vector(up_weights + off, xl, sh, ib, il);
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
kernel void private_iq4_share_fast2_2(constant qw3_matvec_q8_0_args &args,
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
    const uint nr0 = 2u;
    const uint nsg = 8u;
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
            sum0 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
            sum1 += qw3_iq4xs_dot8_vector(blk, xl, sh, ib, il);
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
