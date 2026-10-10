#ifdef QW3_METAL_HAS_TENSOR
kernel void kstage_iq3_s_64(
    constant qw3_matmul_q8_0_batch_args &args,
    device const uchar *weights,
    device const float *x,
    device float *out,
    device const uchar *kgrid,
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int ROWS_PER_SG = 32;
    constexpr int NK = 32;
    constexpr int STAGE_K = 64;
    const int r0 = int(group.y) * NR0;
    const int r1 = int(group.x) * (4 * ROWS_PER_SG);
    if (r0 >= int(args.n_out) || r1 >= int(args.n_tokens)) return;
    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(STAGE_K, NR0));
    device float *ptrB = (device float *)x + args.in_offset;
    auto tB = tensor(ptrB,
                    dextents<int32_t, 2>(int(args.n_in), int(args.n_tokens)),
                    array<int, 2>({1, int(args.in_stride)}));
    matmul2d<matmul2d_descriptor(ROWS_PER_SG, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<1>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<
        decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    for (uint stage_k = 0u; stage_k < args.n_in; stage_k += STAGE_K) {
        for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
        const uint loop_k = stage_k + sub_k;
        const int row = int(tid) / 2;
        const int half_k = int(tid) & 1;
        if (r0 + row < int(args.n_out)) {
            const uint k0 = loop_k + uint(half_k * 16);
            device const uchar *wrow = weights +
                uint64_t(r0 + row) * args.row_bytes;
            device const uchar *blk =
                wrow + uint64_t(k0 >> 8u) * 110ull;
            const uint il = (k0 & 255u) >> 4u;
            const uint ib32 = il >> 1u;
            const uint half_il = il & 1u;
            const float d = float(*((device const half *)blk));
            device const uchar *qs = blk + 2u;
            device const uchar *qh = qs + 64u;
            device const uchar *signs = qh + 8u;
            device const uchar *scales = signs + 32u;
            const float dl = d * float(1u + 2u *
                ((uint(scales[ib32 >> 1u]) >>
                  (4u * (ib32 & 1u))) & 15u));
            device const uchar *qs16 =
                qs + 8u * ib32 + 4u * half_il;
            device const uchar *signs16 =
                signs + 4u * ib32 + 2u * half_il;
            const uchar qh16 =
                uchar(uint(qh[ib32]) >> (4u * half_il));
            const half4x4 values = qw3_iq3s_dequant4x4_expanded(
                dl, qs16, qh16, signs16, kgrid);
#pragma unroll
            for (int i = 0; i < 16; ++i)
                sa[row * STAGE_K + int(sub_k) + half_k * 16 + i] =
                    values[i / 4][i & 3];
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                sa[row * STAGE_K + int(sub_k) + half_k * 16 + i] = half(0.0f);
        }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
            auto mA = tA.slice(int(sub_k), 0);
            auto mB = tB.slice(int(stage_k + sub_k),
                               r1 + int(simd_group) * ROWS_PER_SG);
            mm.run(mB, mA, cT);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out),
                                         int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    cT.store(tD.slice(r0, r1 + int(simd_group) * ROWS_PER_SG));
}

kernel void kstage_iq3_s_128(
    constant qw3_matmul_q8_0_batch_args &args,
    device const uchar *weights,
    device const float *x,
    device float *out,
    device const uchar *kgrid,
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int ROWS_PER_SG = 32;
    constexpr int NK = 32;
    constexpr int STAGE_K = 128;
    const int r0 = int(group.y) * NR0;
    const int r1 = int(group.x) * (4 * ROWS_PER_SG);
    if (r0 >= int(args.n_out) || r1 >= int(args.n_tokens)) return;
    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(STAGE_K, NR0));
    device float *ptrB = (device float *)x + args.in_offset;
    auto tB = tensor(ptrB,
                    dextents<int32_t, 2>(int(args.n_in), int(args.n_tokens)),
                    array<int, 2>({1, int(args.in_stride)}));
    matmul2d<matmul2d_descriptor(ROWS_PER_SG, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<1>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<
        decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    for (uint stage_k = 0u; stage_k < args.n_in; stage_k += STAGE_K) {
        for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
        const uint loop_k = stage_k + sub_k;
        const int row = int(tid) / 2;
        const int half_k = int(tid) & 1;
        if (r0 + row < int(args.n_out)) {
            const uint k0 = loop_k + uint(half_k * 16);
            device const uchar *wrow = weights +
                uint64_t(r0 + row) * args.row_bytes;
            device const uchar *blk =
                wrow + uint64_t(k0 >> 8u) * 110ull;
            const uint il = (k0 & 255u) >> 4u;
            const uint ib32 = il >> 1u;
            const uint half_il = il & 1u;
            const float d = float(*((device const half *)blk));
            device const uchar *qs = blk + 2u;
            device const uchar *qh = qs + 64u;
            device const uchar *signs = qh + 8u;
            device const uchar *scales = signs + 32u;
            const float dl = d * float(1u + 2u *
                ((uint(scales[ib32 >> 1u]) >>
                  (4u * (ib32 & 1u))) & 15u));
            device const uchar *qs16 =
                qs + 8u * ib32 + 4u * half_il;
            device const uchar *signs16 =
                signs + 4u * ib32 + 2u * half_il;
            const uchar qh16 =
                uchar(uint(qh[ib32]) >> (4u * half_il));
            const half4x4 values = qw3_iq3s_dequant4x4_expanded(
                dl, qs16, qh16, signs16, kgrid);
#pragma unroll
            for (int i = 0; i < 16; ++i)
                sa[row * STAGE_K + int(sub_k) + half_k * 16 + i] =
                    values[i / 4][i & 3];
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                sa[row * STAGE_K + int(sub_k) + half_k * 16 + i] = half(0.0f);
        }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
            auto mA = tA.slice(int(sub_k), 0);
            auto mB = tB.slice(int(stage_k + sub_k),
                               r1 + int(simd_group) * ROWS_PER_SG);
            mm.run(mB, mA, cT);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out),
                                         int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    cT.store(tD.slice(r0, r1 + int(simd_group) * ROWS_PER_SG));
}

kernel void kstage_iq4_xs_64(
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
    constexpr int STAGE_K = 64;
    const int r0 = int(group.y) * NR0;
    const int r1 = int(group.x) * (4 * ROWS_PER_SG);
    if (r0 >= int(args.n_out) || r1 >= int(args.n_tokens)) return;
    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(STAGE_K, NR0));
    device float *ptrB = (device float *)x + args.in_offset;
    auto tB = tensor(ptrB,
                    dextents<int32_t, 2>(int(args.n_in), int(args.n_tokens)),
                    array<int, 2>({1, int(args.in_stride)}));
    matmul2d<matmul2d_descriptor(ROWS_PER_SG, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<1>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<
        decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    for (uint stage_k = 0u; stage_k < args.n_in; stage_k += STAGE_K) {
        for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
        const uint loop_k = stage_k + sub_k;
        const int row = int(tid) / 2;
        const int half_k = int(tid) & 1;
        if (r0 + row < int(args.n_out)) {
            const uint k0 = loop_k + uint(half_k * 16);
            device const uchar *wrow = weights +
                uint64_t(r0 + row) * args.row_bytes;
            device const uchar *blk =
                wrow + uint64_t(k0 >> 8u) * 136ull;
            const float d = float(*((device const half *)blk));
            const ushort scales_h =
                *((device const ushort *)(blk + 2u));
            device const uchar *scales_l = blk + 4u;
            const uint il = (k0 & 255u) >> 4u;
            const uint ib32 = il >> 1u;
            const bool hi = (il & 1u) != 0u;
            const uint ls =
                ((uint(scales_l[ib32 >> 1u]) >>
                  (4u * (ib32 & 1u))) & 15u) |
                (((uint(scales_h) >> (2u * ib32)) & 3u) << 4u);
            const float dl = d * float(int(ls) - 32);
            device const uchar *q16 = scales_l + 4u + ib32 * 16u;
            const half4x4 values = qw3_iq4xs_dequant4x4(dl, q16, hi);
#pragma unroll
            for (int i = 0; i < 16; ++i)
                sa[row * STAGE_K + int(sub_k) + half_k * 16 + i] =
                    values[i / 4][i & 3];
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                sa[row * STAGE_K + int(sub_k) + half_k * 16 + i] = half(0.0f);
        }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
            auto mA = tA.slice(int(sub_k), 0);
            auto mB = tB.slice(int(stage_k + sub_k),
                               r1 + int(simd_group) * ROWS_PER_SG);
            mm.run(mB, mA, cT);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out),
                                         int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    cT.store(tD.slice(r0, r1 + int(simd_group) * ROWS_PER_SG));
}

kernel void kstage_iq4_xs_128(
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
    constexpr int STAGE_K = 128;
    const int r0 = int(group.y) * NR0;
    const int r1 = int(group.x) * (4 * ROWS_PER_SG);
    if (r0 >= int(args.n_out) || r1 >= int(args.n_tokens)) return;
    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(STAGE_K, NR0));
    device float *ptrB = (device float *)x + args.in_offset;
    auto tB = tensor(ptrB,
                    dextents<int32_t, 2>(int(args.n_in), int(args.n_tokens)),
                    array<int, 2>({1, int(args.in_stride)}));
    matmul2d<matmul2d_descriptor(ROWS_PER_SG, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<1>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<
        decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    for (uint stage_k = 0u; stage_k < args.n_in; stage_k += STAGE_K) {
        for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
        const uint loop_k = stage_k + sub_k;
        const int row = int(tid) / 2;
        const int half_k = int(tid) & 1;
        if (r0 + row < int(args.n_out)) {
            const uint k0 = loop_k + uint(half_k * 16);
            device const uchar *wrow = weights +
                uint64_t(r0 + row) * args.row_bytes;
            device const uchar *blk =
                wrow + uint64_t(k0 >> 8u) * 136ull;
            const float d = float(*((device const half *)blk));
            const ushort scales_h =
                *((device const ushort *)(blk + 2u));
            device const uchar *scales_l = blk + 4u;
            const uint il = (k0 & 255u) >> 4u;
            const uint ib32 = il >> 1u;
            const bool hi = (il & 1u) != 0u;
            const uint ls =
                ((uint(scales_l[ib32 >> 1u]) >>
                  (4u * (ib32 & 1u))) & 15u) |
                (((uint(scales_h) >> (2u * ib32)) & 3u) << 4u);
            const float dl = d * float(int(ls) - 32);
            device const uchar *q16 = scales_l + 4u + ib32 * 16u;
            const half4x4 values = qw3_iq4xs_dequant4x4(dl, q16, hi);
#pragma unroll
            for (int i = 0; i < 16; ++i)
                sa[row * STAGE_K + int(sub_k) + half_k * 16 + i] =
                    values[i / 4][i & 3];
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                sa[row * STAGE_K + int(sub_k) + half_k * 16 + i] = half(0.0f);
        }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
            auto mA = tA.slice(int(sub_k), 0);
            auto mB = tB.slice(int(stage_k + sub_k),
                               r1 + int(simd_group) * ROWS_PER_SG);
            mm.run(mB, mA, cT);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out),
                                         int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    cT.store(tD.slice(r0, r1 + int(simd_group) * ROWS_PER_SG));
}
#endif
