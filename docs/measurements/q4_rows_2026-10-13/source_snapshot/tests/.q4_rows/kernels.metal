kernel void private_q4_rows_fast(constant qw3_matvec_q8_0_args &args,
                            device const uchar *weights,
                            device const float *x,
                            device float *out,
                            threadgroup float *sh,
                            uint group [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort simd_idx [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]],
                            ushort nt [[threads_per_threadgroup]]) {
    if (group * 2u >= args.n_out) return;
    uint row = group * 2u + uint(simd_idx);
    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);
    float sum = 0.0f;
    for (uint b = uint(lane); row < args.n_out && b < args.n_in / 256u; b += nt) {
        device const uchar *blk = wr + uint64_t(b) * 144ull;
        ushort dbits = ushort(blk[0]) | (ushort(blk[1]) << 8u);
        ushort dmin_bits = ushort(blk[2]) | (ushort(blk[3]) << 8u);
        float d = qw3_f16_to_f32(dbits);
        float dmin = qw3_f16_to_f32(dmin_bits);
        device const uchar *scales = blk + 4u;
        device const uchar *qs = blk + 16u;
        device const float *xx = x + uint64_t(b) * 256ull;
        for (uint g = 0; g < 8u; g++) {
            uint sc;
            uint mn;
            if (g < 4u) {
                sc = uint(scales[g]) & 63u;
                mn = uint(scales[g + 4u]) & 63u;
            } else {
                sc = (uint(scales[g + 4u]) & 15u) |
                     ((uint(scales[g - 4u]) >> 6u) << 4u);
                mn = (uint(scales[g + 4u]) >> 4u) |
                     ((uint(scales[g]) >> 6u) << 4u);
            }
            float dl = d * float(sc);
            float ml = dmin * float(mn);
            uint qbase = (g >> 1u) * 32u;
            uint xbase = g * 32u;
            for (uint l = 0; l < 32u; l++) {
                uint packed = uint(qs[qbase + l]);
                uint q = (g & 1u) == 0u ? (packed & 15u) : (packed >> 4u);
                sum += (dl * float(q) - ml) * xx[xbase + l];
            }
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) sh[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane == 0u ? sh[simd_idx] : 0.0f;
    sum = simd_sum(sum);
    if (lane == 0u && row < args.n_out) out[row] = sum;
}
