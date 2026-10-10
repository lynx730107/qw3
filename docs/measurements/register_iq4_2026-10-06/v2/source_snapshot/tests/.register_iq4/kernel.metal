#ifdef QW3_METAL_HAS_TENSOR
kernel void register_iq4_staged128(
    constant qw3_matmul_q8_0_batch_args &args,
    device const uchar *weights,
    device const float *x,
    device float *out,
    uint2 group [[threadgroup_position_in_grid]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int ROWS_PER_SG = 32;
    constexpr int NK = 32;
    const int r0 = int(group.y) * NR0;
    const int r1 = int(group.x) * (4 * ROWS_PER_SG);
    if (r0 >= int(args.n_out) || r1 >= int(args.n_tokens)) return;
    using WeightTensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;
    device float *ptrB = (device float *)x + args.in_offset;
    auto tB = tensor(ptrB,
                    dextents<int32_t, 2>(int(args.n_in), int(args.n_tokens)),
                    array<int, 2>({1, int(args.in_stride)}));
    matmul2d<matmul2d_descriptor(ROWS_PER_SG, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<1>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<
        decltype(tB), WeightTensor, float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    auto cA = mm.get_right_input_cooperative_tensor<float, half, float>();
    for (uint loop_k = 0u; loop_k < args.n_in; loop_k += NK) {
        for (uint16_t i = 0; i < cA.get_capacity(); ++i) {
            if (cA.is_valid_element(i)) {
                auto index = cA.get_multidimensional_index(i);
                const uint k = loop_k + uint(index[0]);
                const uint row = uint(r0 + index[1]);
                cA[i] = row < args.n_out && k < args.n_in ?
                    qw3_iq4xs_dequant_k(weights + uint64_t(row) * args.row_bytes, k) : half(0.0f);
            }
        }
        auto mB = tB.slice(int(loop_k), r1 + int(simd_group) * ROWS_PER_SG);
        mm.run(mB, cA, cT);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out),
                                         int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    cT.store(tD.slice(r0, r1 + int(simd_group) * ROWS_PER_SG));
}
#endif
