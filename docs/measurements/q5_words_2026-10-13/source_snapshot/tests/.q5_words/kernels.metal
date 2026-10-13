kernel void private_q5_words_fast1(constant qw3_matvec_q8_0_args &args,
                                 device const uchar *weights,
                                 device const float *x,
                                 device float *out,
                                 threadgroup float *sh,
                                 uint group [[threadgroup_position_in_grid]],
                                 ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;
    const uint row = group * 2u + uint(simd_idx);
    if (row >= args.n_out) return;
    const uint tid = uint(lane) / 4u;
    const uint ix = uint(lane) & 3u;
    const uint iq = tid / 4u;
    const uint ir = tid & 3u;
    const uint l0 = 8u * ir;
    const uint q_offset = 32u * iq + l0;
    const uint x_offset = 64u * iq + l0;
    const uchar hm1 = uchar(1u << (2u * iq));
    const uchar hm2 = uchar(uint(hm1) << 1u);
    const uchar hm3 = uchar(uint(hm1) << 4u);
    const uchar hm4 = uchar(uint(hm2) << 4u);
    const uint nb = args.n_in / 256u;
    device const uchar *wr = weights +
        uint64_t(row) * uint64_t(args.row_bytes);
    device const float *x1 = x + ix * 256u + x_offset;
    float sum = 0.0f;

    for (uint ib = ix; ib < nb; ib += 4u) {
        device const uchar *blk = wr + uint64_t(ib) * 176ull;
        device const uchar *q1 = blk + 48u + q_offset;
        device const uchar *q2 = q1 + 64u;
        device const uchar *qh = blk + 16u + l0;
        device const ushort *a =
            (device const ushort *)(blk + 4u) + iq;
        device const float *x2 = x1 + 128u;
        float yl[16], yh[16];
        float4 sumx = float4(0.0f);
#pragma unroll
        for (uint l = 0; l < 8u; ++l) {
            yl[l] = x1[l];       sumx[0] += yl[l];
            yl[l + 8u] = x1[l + 32u];  sumx[1] += yl[l + 8u];
            yh[l] = x2[l];       sumx[2] += yh[l];
            yh[l + 8u] = x2[l + 32u];  sumx[3] += yh[l + 8u];
        }
        ushort sc16[4];
        sc16[0] = a[0] & kmask1;
        sc16[1] = a[2] & kmask1;
        sc16[2] = ((a[4] >> 0u) & kmask2) |
                  ((a[0] & kmask3) >> 2u);
        sc16[3] = ((a[4] >> 4u) & kmask2) |
                  ((a[2] & kmask3) >> 2u);
        thread const uchar *sc8 = (thread const uchar *)sc16;
        const uint2 q1_words = *((device const uint2 *)q1);
        const uint2 q2_words = *((device const uint2 *)q2);
        float4 acc1 = float4(0.0f), acc2 = float4(0.0f);
#pragma unroll
        for (uint l = 0; l < 8u; ++l) {
            const uchar q1_byte = uchar((q1_words[l >> 2u] >> (8u * (l & 3u))) & 255u);
            const uchar q2_byte = uchar((q2_words[l >> 2u] >> (8u * (l & 3u))) & 255u);
            const uchar h = qh[l];
            acc1[0] += yl[l]      * float(q1_byte & 0x0fu);
            acc1[1] += yl[l + 8u] * float(q1_byte & 0xf0u);
            acc1[2] += yh[l]      * float(q2_byte & 0x0fu);
            acc1[3] += yh[l + 8u] * float(q2_byte & 0xf0u);
            acc2[0] += (h & hm1) ? yl[l] : 0.0f;
            acc2[1] += (h & hm2) ? yl[l + 8u] : 0.0f;
            acc2[2] += (h & hm3) ? yh[l] : 0.0f;
            acc2[3] += (h & hm4) ? yh[l + 8u] : 0.0f;
        }
        const float d = float(*((device const half *)blk));
        const float dmin = float(*((device const half *)(blk + 2u)));
        sum += d *
            (float(sc8[0]) * (acc1[0] + 16.0f * acc2[0]) +
             float(sc8[1]) * (acc1[1] * (1.0f/16.0f) + 16.0f * acc2[1]) +
             float(sc8[4]) * (acc1[2] + 16.0f * acc2[2]) +
             float(sc8[5]) * (acc1[3] * (1.0f/16.0f) + 16.0f * acc2[3])) -
            dmin * (sumx[0] * float(sc8[2]) +
                    sumx[1] * float(sc8[3]) +
                    sumx[2] * float(sc8[6]) +
                    sumx[3] * float(sc8[7]));
        x1 += 4u * 256u;
    }
    sum = simd_sum(sum);
    if (lane == 0u) out[row] = sum;
}
kernel void private_q5_words_fast2(constant qw3_matvec_q8_0_args &args,
                                 device const uchar *weights,
                                 device const float *x,
                                 device float *out,
                                 threadgroup float *sh,
                                 uint group [[threadgroup_position_in_grid]],
                                 ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;
    const uint row = group * 2u + uint(simd_idx);
    if (row >= args.n_out) return;
    const uint tid = uint(lane) / 4u;
    const uint ix = uint(lane) & 3u;
    const uint iq = tid / 4u;
    const uint ir = tid & 3u;
    const uint l0 = 8u * ir;
    const uint q_offset = 32u * iq + l0;
    const uint x_offset = 64u * iq + l0;
    const uchar hm1 = uchar(1u << (2u * iq));
    const uchar hm2 = uchar(uint(hm1) << 1u);
    const uchar hm3 = uchar(uint(hm1) << 4u);
    const uchar hm4 = uchar(uint(hm2) << 4u);
    const uint nb = args.n_in / 256u;
    device const uchar *wr = weights +
        uint64_t(row) * uint64_t(args.row_bytes);
    device const float *x1 = x + ix * 256u + x_offset;
    float sum = 0.0f;

    for (uint ib = ix; ib < nb; ib += 4u) {
        device const uchar *blk = wr + uint64_t(ib) * 176ull;
        device const uchar *q1 = blk + 48u + q_offset;
        device const uchar *q2 = q1 + 64u;
        device const uchar *qh = blk + 16u + l0;
        device const ushort *a =
            (device const ushort *)(blk + 4u) + iq;
        device const float *x2 = x1 + 128u;
        float yl[16], yh[16];
        float4 sumx = float4(0.0f);
#pragma unroll
        for (uint l = 0; l < 8u; ++l) {
            yl[l] = x1[l];       sumx[0] += yl[l];
            yl[l + 8u] = x1[l + 32u];  sumx[1] += yl[l + 8u];
            yh[l] = x2[l];       sumx[2] += yh[l];
            yh[l + 8u] = x2[l + 32u];  sumx[3] += yh[l + 8u];
        }
        ushort sc16[4];
        sc16[0] = a[0] & kmask1;
        sc16[1] = a[2] & kmask1;
        sc16[2] = ((a[4] >> 0u) & kmask2) |
                  ((a[0] & kmask3) >> 2u);
        sc16[3] = ((a[4] >> 4u) & kmask2) |
                  ((a[2] & kmask3) >> 2u);
        thread const uchar *sc8 = (thread const uchar *)sc16;
        const uint2 q1_words = *((device const uint2 *)q1);
        const uint2 q2_words = *((device const uint2 *)q2);
        const uint2 qh_words = *((device const uint2 *)qh);
        float4 acc1 = float4(0.0f), acc2 = float4(0.0f);
#pragma unroll
        for (uint l = 0; l < 8u; ++l) {
            const uchar q1_byte = uchar((q1_words[l >> 2u] >> (8u * (l & 3u))) & 255u);
            const uchar q2_byte = uchar((q2_words[l >> 2u] >> (8u * (l & 3u))) & 255u);
            const uchar h = uchar((qh_words[l >> 2u] >> (8u * (l & 3u))) & 255u);
            acc1[0] += yl[l]      * float(q1_byte & 0x0fu);
            acc1[1] += yl[l + 8u] * float(q1_byte & 0xf0u);
            acc1[2] += yh[l]      * float(q2_byte & 0x0fu);
            acc1[3] += yh[l + 8u] * float(q2_byte & 0xf0u);
            acc2[0] += (h & hm1) ? yl[l] : 0.0f;
            acc2[1] += (h & hm2) ? yl[l + 8u] : 0.0f;
            acc2[2] += (h & hm3) ? yh[l] : 0.0f;
            acc2[3] += (h & hm4) ? yh[l + 8u] : 0.0f;
        }
        const float d = float(*((device const half *)blk));
        const float dmin = float(*((device const half *)(blk + 2u)));
        sum += d *
            (float(sc8[0]) * (acc1[0] + 16.0f * acc2[0]) +
             float(sc8[1]) * (acc1[1] * (1.0f/16.0f) + 16.0f * acc2[1]) +
             float(sc8[4]) * (acc1[2] + 16.0f * acc2[2]) +
             float(sc8[5]) * (acc1[3] * (1.0f/16.0f) + 16.0f * acc2[3])) -
            dmin * (sumx[0] * float(sc8[2]) +
                    sumx[1] * float(sc8[3]) +
                    sumx[2] * float(sc8[6]) +
                    sumx[3] * float(sc8[7]));
        x1 += 4u * 256u;
    }
    sum = simd_sum(sum);
    if (lane == 0u) out[row] = sum;
}
