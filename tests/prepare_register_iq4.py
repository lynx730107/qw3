"""Private IQ4_XS staged128 prototype with register-resident weight tiles."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
source = (root / 'metal/qw3_core_moe.metal').read_text()
name = 'qw3_matmul_iq4_xs_mpp_staged128'
start = source.index('kernel void ' + name + '(')
end = source.index('\n}\n', start) + 3
body = source[start:end].replace(name, 'register_iq4_staged128')
anchor = '    threadgroup char *shmem [[threadgroup(0)]],\n'
assert body.count(anchor) == 1
body = body.replace(anchor, '')
body = body.replace('    ushort tid [[thread_index_in_threadgroup]],\n', '')
anchor = '''    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));'''
assert body.count(anchor) == 1
body = body.replace(anchor, '    using WeightTensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;')
body = body.replace('decltype(tA)', 'WeightTensor')
start = body.index('    for (uint loop_k = 0u;')
end = body.index('    auto tD =', start)
body = body[:start] + '''    auto cA = mm.get_right_input_cooperative_tensor<float, half, float>();
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
''' + body[end:]
out = root / 'tests/.register_iq4'
out.mkdir(exist_ok=True)
(out / 'kernel.metal').write_text('#ifdef QW3_METAL_HAS_TENSOR\n' + body + '#endif\n')
