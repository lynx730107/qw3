
#ifdef QW3_METAL_HAS_TENSOR
kernel void probe_q3_metadata(
    constant qw3_matmul_q8_0_batch_args &args,
    device const uchar *weights,
    device const float *x,
    device float *out,
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int ROWS_PER_SG = 32;
    constexpr int NK = 32;
    const int r0 = int(group.y) * NR0;
    const int r1 = int(group.x) * (4 * ROWS_PER_SG);
    if (r0 >= int(args.n_out) || r1 >= int(args.n_tokens)) return;
    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    device float *ptrB = (device float *)x + args.in_offset;
    auto tB = tensor(ptrB,
                    dextents<int32_t, 2>(int(args.n_in), int(args.n_tokens)),
                    array<int, 2>({1, int(args.in_stride)}));
    matmul2d<matmul2d_descriptor(ROWS_PER_SG, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<1>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<
        decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); i++)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0f0f0f0fu;
    const int row = int(tid) / 2;
    const int half_k = int(tid) % 2;
    for (uint block_k = 0; block_k < args.n_in; block_k += 256u) {
        uint aux[4] = {0u,0u,0u,0u};
        float d = 0.0f;
        device const uchar *blk = weights;
        if (r0 + row < int(args.n_out)) {
            device const uchar *wrow = weights +
                uint64_t(r0 + row) * args.row_bytes;
            blk = wrow + uint64_t(block_k >> 8u) * 110ull;
            device const uchar *packed = blk + 96u;
            uint aux0 = uint(packed[0]) | (uint(packed[1]) << 8u) |
                        (uint(packed[2]) << 16u) | (uint(packed[3]) << 24u);
            uint aux1 = uint(packed[4]) | (uint(packed[5]) << 8u) |
                        (uint(packed[6]) << 16u) | (uint(packed[7]) << 24u);
            uint tmp = uint(packed[8]) | (uint(packed[9]) << 8u) |
                       (uint(packed[10]) << 16u) | (uint(packed[11]) << 24u);
            aux[2] = ((aux0 >> 4u) & kmask2) |
                     (((tmp >> 4u) & kmask1) << 4u);
            aux[3] = ((aux1 >> 4u) & kmask2) |
                     (((tmp >> 6u) & kmask1) << 4u);
            aux[0] = (aux0 & kmask2) |
                     (((tmp >> 0u) & kmask1) << 4u);
            aux[1] = (aux1 & kmask2) |
                     (((tmp >> 2u) & kmask1) << 4u);
            d = float(*((device const half *)(blk + 108u)));
        }
        for (uint loop_k = block_k; loop_k < block_k + 256u; loop_k += NK) {
        if (r0 + row < int(args.n_out)) {
            const uint k0 = loop_k + uint(half_k * 16);
            for (int i = 0; i < 16; i++) {
                const uint local = (k0 + uint(i)) & 255u;
                const uint half_block = local >> 7u;
                const uint within = local & 127u;
                const uint grp = within >> 5u;
                const uint upper = (within & 31u) >= 16u ? 1u : 0u;
                const uint l = within & 15u;
                const uint scale_index = half_block * 8u + grp * 2u + upper;
                const uint scale = (aux[scale_index >> 2u] >>
                                    (8u * (scale_index & 3u))) & 255u;
                const uint shift = grp * 2u;
                const uint mask = 1u << (half_block * 4u + grp);
                const uint q_index = half_block * 32u + upper * 16u + l;
                const int low = int((uint(blk[32u + q_index]) >> shift) & 3u);
                const int q = low -
                    ((uint(blk[upper * 16u + l]) & mask) ? 0 : 4);
                sa[row * NK + half_k * 16 + i] =
                    half(d * float(int(scale) - 32) * float(q));
            }
        } else {
            for (int i = 0; i < 16; i++)
                sa[row * NK + half_k * 16 + i] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(int(loop_k),
                           r1 + int(simd_group) * ROWS_PER_SG);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out), int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    cT.store(tD.slice(r0, r1 + int(simd_group) * ROWS_PER_SG));
}

#endif
