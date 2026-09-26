struct qw3_matvec_q8_0_args { uint n_in; uint n_out; uint row_bytes; };
kernel void qw3_matvec_q8_0(constant qw3_matvec_q8_0_args &args,
                            device const uchar *weights,
                            device const float *x,
                            device float *out,
                            threadgroup float *sh,
                            uint row [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort simd_idx [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]],
                            ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);
    float sum = 0.0f;
    uint n_blocks = args.n_in / 32;
    for (uint b = tid; b < n_blocks; b += nt) {
        device const uchar *blk = wr + b * 34;
        half d = *((device const half *)blk);
        for (uint i = 0; i < 32; i++) {
            char q = *((device const char *)(blk + 2 + i));
            sum += float(d) * float(q) * x[b * 32 + i];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) sh[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane < ((uint(nt) + 31u) / 32u) ? sh[lane] : 0.0f;
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}
struct qw3_matmul_q8_0_batch_args { uint n_in; uint n_out; uint row_bytes; uint n_tokens; uint in_offset; uint in_stride; uint out_offset; uint out_stride; };
kernel void qw3_matmul_q8_0_batch4(constant qw3_matmul_q8_0_batch_args &args,
                                   device const uchar *weights,
                                   device const float *x,
                                   device float *out,
                                   uint2 group [[threadgroup_position_in_grid]],
                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]]) {
    uint row = group.x * 4u + uint(simd_idx);
    if (row >= args.n_out) return;
    uint t0 = group.y * 4u;
    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    uint n_blocks = args.n_in / 32u;
    for (uint b = uint(lane); b < n_blocks; b += 32u) {
        device const uchar *blk = wr + uint64_t(b) * 34ull;
        float d = float(*((device const half *)blk));
        uint xb = b * 32u;
        for (uint i = 0; i < 32u; i++) {
            float wv = d * float(*((device const char *)(blk + 2u + i)));
            uint xi = xb + i;
            if (t0 + 0u < args.n_tokens) s0 += wv * x[uint64_t(t0 + 0u) * args.in_stride + args.in_offset + xi];
            if (t0 + 1u < args.n_tokens) s1 += wv * x[uint64_t(t0 + 1u) * args.in_stride + args.in_offset + xi];
            if (t0 + 2u < args.n_tokens) s2 += wv * x[uint64_t(t0 + 2u) * args.in_stride + args.in_offset + xi];
            if (t0 + 3u < args.n_tokens) s3 += wv * x[uint64_t(t0 + 3u) * args.in_stride + args.in_offset + xi];
        }
    }
    s0 = simd_sum(s0); s1 = simd_sum(s1); s2 = simd_sum(s2); s3 = simd_sum(s3);
    if (lane == 0) {
        if (t0 + 0u < args.n_tokens) out[uint64_t(t0 + 0u) * args.out_stride + args.out_offset + row] = s0;
        if (t0 + 1u < args.n_tokens) out[uint64_t(t0 + 1u) * args.out_stride + args.out_offset + row] = s1;
        if (t0 + 2u < args.n_tokens) out[uint64_t(t0 + 2u) * args.out_stride + args.out_offset + row] = s2;
        if (t0 + 3u < args.n_tokens) out[uint64_t(t0 + 3u) * args.out_stride + args.out_offset + row] = s3;
    }
}


constant bool qw3_mm_q8_bc_out [[function_constant(700)]];
struct qw3_block_q8_0 { half d; char qs[32]; };
struct qw3_matmul_q8_0_mm_args { uint n_in; uint n_out; uint row_bytes; uint n_tokens; uint in_stride; uint out_stride; };
static inline void qw3_dequant_q8_0_16(device const qw3_block_q8_0 *xb, short il, thread half4x4 &reg) {
    const float d = float(xb->d);
    float4x4 tmp;
    for (short i = 0; i < 16; i++) tmp[i / 4][i % 4] = float(xb->qs[i + 16 * il]) * d;
    reg = half4x4(tmp);
}
kernel void qw3_matmul_q8_0_mm(constant qw3_matmul_q8_0_mm_args &args,
                                device const char *weights,
                                device const char *xin,
                                device char *yout,
                                threadgroup char *shmem [[threadgroup(0)]],
                                uint3 tgpig [[threadgroup_position_in_grid]],
                                ushort tiitg [[thread_index_in_threadgroup]],
                                ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;
    const int K = int(args.n_in);
    const int M = int(args.n_out);
    const int N = int(args.n_tokens);
    const int r0 = int(tgpig.y) * NR0;
    const int r1 = int(tgpig.x) * NR1;
    const int nr0 = min(M - r0, NR0);
    const int nr1 = min(N - r1, NR1);
    const int lr0 = min(int(tiitg) / NL0, nr0 - 1);
    const int lr1 = min(int(tiitg) / NL1, nr1 - 1);
    const short il0 = short(tiitg % NL0);
    short il = il0;
    device const qw3_block_q8_0 *wblk = (device const qw3_block_q8_0 *)(weights + uint64_t(args.row_bytes) * uint64_t(r0 + lr0));
    const short iy = short(8 * (tiitg % NL1));
    device const float *yin = (device const float *)xin + uint64_t(args.in_stride) * uint64_t(r1 + lr1) + uint64_t(iy);
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        half4x4 temp_a;
        qw3_dequant_q8_0_16(wblk, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; i++) {
            const short sx = short(2 * il0 + i / 8);
            const short sy = short((tiitg / NL0) / 8);
            const short lx = short((tiitg / NL0) % 8);
            const short ly = short(i % 8);
            const short ib = short(8 * sx + sy);
            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
        }
        for (short i = 0; i < 8; i++) {
            const short sx = short(tiitg % NL1);
            const short sy = short((tiitg / NL1) / 8);
            const short lx = i;
            const short ly = short((tiitg / NL1) % 8);
            const short ib = short(4 * sx + sy);
            *(sb + 64 * ib + 8 * ly + lx) = half(yin[i]);
        }
        il = short((il + 2 < 2) ? il + 2 : il % 2);
        wblk = (il < 2) ? wblk + 1 : wblk;
        yin += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half *lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half *lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }
    device float *dst = (device float *)yout;
    if (!qw3_mm_q8_bc_out || (r0 + NR0 <= M && r1 + NR1 <= N)) {
        device float *C = dst + uint64_t(r0 + 32 * (sgitg & 1)) + uint64_t(r1 + 16 * (sgitg >> 1)) * uint64_t(args.out_stride);
        for (short i = 0; i < 8; i++) simdgroup_store(mc[i], C + 8 * (i % 4) + uint64_t(8 * (i / 4)) * uint64_t(args.out_stride), args.out_stride, 0, false);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float *tmp = ((threadgroup float *)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
        for (short i = 0; i < 8; i++) simdgroup_store(mc[i], tmp + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = int(tiitg); j < nr1; j += NR1) {
                device float *D = dst + uint64_t(r0) + uint64_t(r1 + j) * uint64_t(args.out_stride);
                threadgroup float *C = ((threadgroup float *)shmem) + j * NR0;
                int i = 0;
                device float4 *D4 = (device float4 *)D;
                threadgroup float4 *C4 = (threadgroup float4 *)C;
                for (; i < nr0 / 4; i++) D4[i] = C4[i];
                i *= 4;
                for (; i < nr0; i++) D[i] = C[i];
            }
        }
    }
}
#ifdef QW3_METAL_HAS_TENSOR
template<short NR1>
kernel void qw3_matmul_q8_0_nax_direct_rhs(constant qw3_matmul_q8_0_mm_args &args,
                                           device const char *weights,
                                           device const char *xin,
                                           device char *yout,
                                           threadgroup char *shmem [[threadgroup(0)]],
                                           uint2 group [[threadgroup_position_in_grid]],
                                           ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int NK = 32;
    constexpr int NL = NK / 16;
    constexpr int NUM_THREADS = 128;
    const int K = int(args.n_in);
    const int M = int(args.n_out);
    const int N = int(args.n_tokens);
    const int r0 = int(group.y) * NR0;
    const int r1 = int(group.x) * NR1;
    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    device float *ptrB = (device float *)xin;
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N),
                    array<int, 2>({1, int(args.in_stride)}));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); i++) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }
    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = int(tid); work < NR0 * NL; work += NUM_THREADS) {
            const int row = work / NL;
            const int k_chunk = work % NL;
            const int k_pos = loop_k + k_chunk * 16;
            const short k_base = short(k_chunk * 16);
            if (r0 + row < M) {
                const int block_idx = k_pos / 32;
                const short il = short((k_pos / 16) & 1);
                device const qw3_block_q8_0 *row_ptr =
                    (device const qw3_block_q8_0 *)(weights + uint64_t(r0 + row) * uint64_t(args.row_bytes));
                half4x4 temp_a;
                qw3_dequant_q8_0_16(row_ptr + block_idx, il, temp_a);
                for (short i = 0; i < 16; i++) {
                    sa[row * NK + k_base + i] = (k_pos + i < K) ? temp_a[i / 4][i % 4] : half(0.0f);
                }
            } else {
                for (short i = 0; i < 16; i++) sa[row * NK + k_base + i] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, r1);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float *dst = (device float *)yout;
    auto tD = tensor(dst, dextents<int32_t, 2>(M, N),
                    array<int, 2>({1, int(args.out_stride)}));
    auto mD = tD.slice(r0, r1);
    cT.store(mD);
}
typedef decltype(qw3_matmul_q8_0_nax_direct_rhs<32>) qw3_matmul_q8_0_nax_direct_rhs_t;
template [[host_name("qw3_matmul_q8_0_nax_direct_rhs")]] kernel qw3_matmul_q8_0_nax_direct_rhs_t qw3_matmul_q8_0_nax_direct_rhs<32>;
template [[host_name("qw3_matmul_q8_0_nax_direct_rhs_n64")]] kernel qw3_matmul_q8_0_nax_direct_rhs_t qw3_matmul_q8_0_nax_direct_rhs<64>;
template [[host_name("qw3_matmul_q8_0_nax_direct_rhs_n128")]] kernel qw3_matmul_q8_0_nax_direct_rhs_t qw3_matmul_q8_0_nax_direct_rhs<128>;
#endif
struct qw3_matvec_q8_0_pair_args { uint n_in; uint n_out; uint row_bytes; uint out_a_offset; uint out_b_offset; };
kernel void qw3_matvec_q8_0_pair(constant qw3_matvec_q8_0_pair_args &args,
                                device const uchar *weights_a,
                                device const uchar *weights_b,
                                device const float *x,
                                device float *out,
                                threadgroup float *sh,
                                uint row [[threadgroup_position_in_grid]],
                                ushort tid [[thread_index_in_threadgroup]],
                                ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]],
                                ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    device const uchar *wra = weights_a + uint64_t(row) * uint64_t(args.row_bytes);
    device const uchar *wrb = weights_b + uint64_t(row) * uint64_t(args.row_bytes);
    float suma = 0.0f;
    float sumb = 0.0f;
    uint n_blocks = args.n_in / 32;
    for (uint b = tid; b < n_blocks; b += nt) {
        device const float *xx = x + b * 32;
        device const uchar *blka = wra + b * 34;
        device const uchar *blkb = wrb + b * 34;
        half da = *((device const half *)blka);
        half db = *((device const half *)blkb);
        for (uint i = 0; i < 32; i++) {
            char qa = *((device const char *)(blka + 2 + i));
            char qb = *((device const char *)(blkb + 2 + i));
            float xv = xx[i];
            suma += float(da) * float(qa) * xv;
            sumb += float(db) * float(qb) * xv;
        }
    }
    suma = simd_sum(suma);
    sumb = simd_sum(sumb);
    if (lane == 0) {
        sh[simd_idx] = suma;
        sh[simd_idx + 32] = sumb;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    suma = lane < 32 ? sh[lane] : 0.0f;
    sumb = lane < 32 ? sh[lane + 32] : 0.0f;
    suma = simd_sum(suma);
    sumb = simd_sum(sumb);
    if (tid == 0) {
        out[args.out_a_offset + row] = suma;
        out[args.out_b_offset + row] = sumb;
    }
}
kernel void qw3_matvec_q8_0_pair_silu_fast(constant qw3_matvec_q8_0_pair_args &args,
                                             device const uchar *weights_a,
                                             device const uchar *weights_b,
                                             device const float *x,
                                             device float *inner,
                                             threadgroup float *sh,
                                             uint group [[threadgroup_position_in_grid]],
                                             ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                             ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    const uint nr0 = 2u;
    const uint nsg = 4u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    float ga0 = 0.0f, up0 = 0.0f, ga1 = 0.0f, up1 = 0.0f;
    uint n_blocks = args.n_in / 32u;
    for (uint b = uint(lane); b < n_blocks; b += 32u) {
        device const float *xx = x + uint64_t(b) * 32ull;
        uint row = first_row;
        if (row < args.n_out) {
            device const uchar *ba = weights_a + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;
            device const uchar *bb = weights_b + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;
            float da = float(*((device const half *)ba));
            float db = float(*((device const half *)bb));
            for (uint i = 0; i < 32u; i++) { float xv = xx[i]; ga0 += da * float(*((device const char *)(ba + 2u + i))) * xv; up0 += db * float(*((device const char *)(bb + 2u + i))) * xv; }
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            device const uchar *ba = weights_a + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;
            device const uchar *bb = weights_b + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;
            float da = float(*((device const half *)ba));
            float db = float(*((device const half *)bb));
            for (uint i = 0; i < 32u; i++) { float xv = xx[i]; ga1 += da * float(*((device const char *)(ba + 2u + i))) * xv; up1 += db * float(*((device const char *)(bb + 2u + i))) * xv; }
        }
    }
    ga0 = simd_sum(ga0); up0 = simd_sum(up0);
    ga1 = simd_sum(ga1); up1 = simd_sum(up1);
    if (lane == 0) {
        uint row = first_row;
        if (row < args.n_out) inner[row] = (ga0 / (1.0f + exp(-ga0))) * up0;
        row = first_row + 1u;
        if (row < args.n_out) inner[row] = (ga1 / (1.0f + exp(-ga1))) * up1;
    }
}
struct qw3_shared_gate_up_args { uint n_in; uint n_out; uint row_bytes; uint scalar_offset; };
kernel void qw3_shared_gate_up_silu_fast(constant qw3_shared_gate_up_args &args,
                                         device const uchar *weights_a,
                                         device const uchar *weights_b,
                                         device const float *scalar_weights,
                                         device const float *x,
                                         device float *inner,
                                         device float *scratch,
                                         threadgroup float *sh,
                                         uint group [[threadgroup_position_in_grid]],
                                         ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                         ushort lane [[thread_index_in_simdgroup]]) {
    const uint nr0 = 2u;
    const uint nsg = 4u;
    const uint pair_groups = (args.n_out + nr0 * nsg - 1u) / (nr0 * nsg);
    if (group == pair_groups) {
        uint tid = uint(simd_idx) * 32u + uint(lane);
        float sum = 0.0f;
        for (uint i = tid; i < args.n_in; i += 128u) sum += scalar_weights[i] * x[i];
        sum = simd_sum(sum);
        if (lane == 0) sh[simd_idx] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = lane < 4u ? sh[lane] : 0.0f;
        sum = simd_sum(sum);
        if (tid == 0u) scratch[args.scalar_offset] = sum;
        return;
    }
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    float ga0 = 0.0f, up0 = 0.0f, ga1 = 0.0f, up1 = 0.0f;
    uint n_blocks = args.n_in / 32u;
    for (uint b = uint(lane); b < n_blocks; b += 32u) {
        device const float *xx = x + uint64_t(b) * 32ull;
        uint row = first_row;
        if (row < args.n_out) {
            device const uchar *ba = weights_a + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;
            device const uchar *bb = weights_b + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;
            float da = float(*((device const half *)ba));
            float db = float(*((device const half *)bb));
            for (uint i = 0; i < 32u; i++) { float xv = xx[i]; ga0 += da * float(*((device const char *)(ba + 2u + i))) * xv; up0 += db * float(*((device const char *)(bb + 2u + i))) * xv; }
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            device const uchar *ba = weights_a + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;
            device const uchar *bb = weights_b + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;
            float da = float(*((device const half *)ba));
            float db = float(*((device const half *)bb));
            for (uint i = 0; i < 32u; i++) { float xv = xx[i]; ga1 += da * float(*((device const char *)(ba + 2u + i))) * xv; up1 += db * float(*((device const char *)(bb + 2u + i))) * xv; }
        }
    }
    ga0 = simd_sum(ga0); up0 = simd_sum(up0);
    ga1 = simd_sum(ga1); up1 = simd_sum(up1);
    if (lane == 0) {
        uint row = first_row;
        if (row < args.n_out) inner[row] = (ga0 / (1.0f + exp(-ga0))) * up0;
        row = first_row + 1u;
        if (row < args.n_out) inner[row] = (ga1 / (1.0f + exp(-ga1))) * up1;
    }
}
struct qw3_matvec_q8_0_scale_args { uint n_in; uint n_out; uint row_bytes; uint scalar_offset; };
kernel void qw3_matvec_q8_0_inner_scale_add_x0(constant qw3_matvec_q8_0_scale_args &args,
                                               device const uchar *weights,
                                               device const float *x,
                                               device const float *scratch,
                                               device float *x0,
                                               threadgroup float *sh,
                                               uint row [[threadgroup_position_in_grid]],
                                               ushort tid [[thread_index_in_threadgroup]],
                                               ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                               ushort lane [[thread_index_in_simdgroup]],
                                               ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);
    float sum = 0.0f;
    uint n_blocks = args.n_in / 32;
    for (uint b = tid; b < n_blocks; b += nt) {
        device const uchar *blk = wr + b * 34;
        half d = *((device const half *)blk);
        for (uint i = 0; i < 32; i++) {
            char q = *((device const char *)(blk + 2 + i));
            sum += float(d) * float(q) * x[b * 32 + i];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) sh[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane < ((uint(nt) + 31u) / 32u) ? sh[lane] : 0.0f;
    sum = simd_sum(sum);
    if (tid == 0) {
        float raw = scratch[args.scalar_offset];
        float scale = 1.0f / (1.0f + exp(-raw));
        x0[row] = x0[row] + sum * scale;
    }
}
kernel void qw3_matvec_q8_0_inner_scale_add_x0_fast(constant qw3_matvec_q8_0_scale_args &args,
                                                    device const uchar *weights,
                                                    device const float *x,
                                                    device const float *scratch,
                                                    device float *x0,
                                                    threadgroup float *sh,
                                                    uint group [[threadgroup_position_in_grid]],
                                                    ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                                    ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    const uint nr0 = 2u;
    const uint nsg = 4u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    uint n_blocks = args.n_in / 32u;
    for (uint b = uint(lane); b < n_blocks; b += 32u) {
        device const float *xx = x + uint64_t(b) * 32ull;
        uint row = first_row;
        if (row < args.n_out) {
            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;
            float d = float(*((device const half *)blk));
            for (uint i = 0; i < 32u; i++) sum0 += d * float(*((device const char *)(blk + 2u + i))) * xx[i];
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;
            float d = float(*((device const half *)blk));
            for (uint i = 0; i < 32u; i++) sum1 += d * float(*((device const char *)(blk + 2u + i))) * xx[i];
        }
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (lane == 0) {
        float raw = scratch[args.scalar_offset];
        float scale = 1.0f / (1.0f + exp(-raw));
        uint row = first_row;
        if (row < args.n_out) x0[row] = x0[row] + sum0 * scale;
        row = first_row + 1u;
        if (row < args.n_out) x0[row] = x0[row] + sum1 * scale;
    }
}
kernel void qw3_matvec_q8_0_fast(constant qw3_matvec_q8_0_args &args,
                                 device const uchar *weights,
                                 device const float *x,
                                 device float *out,
                                 threadgroup float *sh,
                                 uint group [[threadgroup_position_in_grid]],
                                 ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    const uint nr0 = 2u;
    const uint nsg = 4u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    uint n_blocks = args.n_in / 32u;
    for (uint b = uint(lane); b < n_blocks; b += 32u) {
        device const float *xx = x + uint64_t(b) * 32ull;
        uint row = first_row;
        if (row < args.n_out) {
            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;
            half d = *((device const half *)blk);
            float ds = float(d);
            for (uint i = 0; i < 32u; i++) { char q = *((device const char *)(blk + 2u + i)); sum0 += ds * float(q) * xx[i]; }
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;
            half d = *((device const half *)blk);
            float ds = float(d);
            for (uint i = 0; i < 32u; i++) { char q = *((device const char *)(blk + 2u + i)); sum1 += ds * float(q) * xx[i]; }
        }
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
inline float qw3_iq4nl_val(uint q) {
    return qw3_iq4nl_table[q & 15u];
}
inline float qw3_iq4xs_dot8(device const uchar *blk,
                            device const float *xg,
                            uint ib, uint il) {
    half d = *((device const half *)blk);
    ushort scales_h = *((device const ushort *)(blk + 2));
    device const uchar *scales_l = blk + 4;
    device const uchar *qs = scales_l + 4 + ib * 16u + il * 8u;
    uint ls = ((uint(scales_l[ib / 2u]) >>
                (4u * (ib & 1u))) & 15u) |
              (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);
    float dl = float(d) * float(int(ls) - 32);
    float acc = 0.0f;
    for (uint j = 0; j < 8u; j++) {
        uchar v = qs[j];
        acc += qw3_iq4nl_val(uint(v) & 15u) * xg[j];
        acc += qw3_iq4nl_val(uint(v) >> 4u) * xg[j + 16u];
    }
    return dl * acc;
}
kernel void qw3_matvec_iq4_xs(constant qw3_matvec_q8_0_args &args,
                              device const uchar *weights,
                              device const float *x,
                              device float *out,
                              threadgroup float *sh,
                              uint row [[threadgroup_position_in_grid]],
                              ushort tid [[thread_index_in_threadgroup]],
                              ushort simd_idx [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]],
                              ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);
    float sum = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = tid; b < n_blocks; b += nt) {
        device const uchar *blk = wr + uint64_t(b) * 136ull;
        half d = *((device const half *)blk);
        ushort scales_h = *((device const ushort *)(blk + 2));
        device const uchar *scales_l = blk + 4;
        device const uchar *qs = scales_l + 4;
        device const float *xx = x + uint64_t(b) * 256ull;
        for (uint ib = 0; ib < 8u; ib++) {
            uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) |
                      (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);
            float dl = float(d) * float(int(ls) - 32);
            device const uchar *q = qs + ib * 16u;
            device const float *xg = xx + ib * 32u;
            for (uint j = 0; j < 16u; j++) {
                uchar v = q[j];
                sum += dl * qw3_iq4nl_val(uint(v) & 15u) * xg[j];
                sum += dl * qw3_iq4nl_val(uint(v) >> 4u) * xg[j + 16u];
            }
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) sh[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane < ((uint(nt) + 31u) / 32u) ? sh[lane] : 0.0f;
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}
kernel void qw3_matvec_iq4_xs_add_x0(constant qw3_matvec_q8_0_args &args,
                                     constant float &scale,
                                     device const uchar *weights,
                                     device const float *x,
                                     device float *x0,
                                     threadgroup float *sh,
                                     uint row [[threadgroup_position_in_grid]],
                                     ushort tid [[thread_index_in_threadgroup]],
                                     ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                     ushort lane [[thread_index_in_simdgroup]],
                                     ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);
    float sum = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = tid; b < n_blocks; b += nt) {
        device const uchar *blk = wr + uint64_t(b) * 136ull;
        half d = *((device const half *)blk);
        ushort scales_h = *((device const ushort *)(blk + 2));
        device const uchar *scales_l = blk + 4;
        device const uchar *qs = scales_l + 4;
        device const float *xx = x + uint64_t(b) * 256ull;
        for (uint ib = 0; ib < 8u; ib++) {
            uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) |
                      (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);
            float dl = float(d) * float(int(ls) - 32);
            device const uchar *q = qs + ib * 16u;
            device const float *xg = xx + ib * 32u;
            for (uint j = 0; j < 16u; j++) {
                uchar v = q[j];
                sum += dl * qw3_iq4nl_val(uint(v) & 15u) * xg[j];
                sum += dl * qw3_iq4nl_val(uint(v) >> 4u) * xg[j + 16u];
            }
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) sh[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane < ((uint(nt) + 31u) / 32u) ? sh[lane] : 0.0f;
    sum = simd_sum(sum);
    if (tid == 0) x0[row] = x0[row] + sum * scale;
}
kernel void qw3_matvec_iq4_xs_fast(constant qw3_matvec_q8_0_args &args,
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
        uint row = first_row;
        uint64_t block_off = uint64_t(b) * 136ull;
        if (row < args.n_out)
            sum0 += qw3_iq4xs_dot8(weights +
                uint64_t(row) * uint64_t(args.row_bytes) + block_off,
                xg, ib, il);
        row = first_row + 1u;
        if (row < args.n_out)
            sum1 += qw3_iq4xs_dot8(weights +
                uint64_t(row) * uint64_t(args.row_bytes) + block_off,
                xg, ib, il);
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
kernel void qw3_matvec_iq4_xs_pair_fast(
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
        uint row = first_row;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) +
                           uint64_t(b) * 136ull;
            gate0 += qw3_iq4xs_dot8(gate_weights + off, xg, ib, il);
            up0 += qw3_iq4xs_dot8(up_weights + off, xg, ib, il);
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) +
                           uint64_t(b) * 136ull;
            gate1 += qw3_iq4xs_dot8(gate_weights + off, xg, ib, il);
            up1 += qw3_iq4xs_dot8(up_weights + off, xg, ib, il);
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
kernel void qw3_matvec_iq4_xs_add_x0_fast(constant qw3_matvec_q8_0_args &args,
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
        uint row = first_row;
        if (row < args.n_out) {
            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
            half d = *((device const half *)blk);
            ushort scales_h = *((device const ushort *)(blk + 2));
            device const uchar *scales_l = blk + 4;
            device const uchar *qs = scales_l + 4 + ib * 16u + il * 8u;
            uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);
            float dl = float(d) * float(int(ls) - 32);
            float acc = 0.0f;
            for (uint j = 0; j < 8u; j++) { uchar v = qs[j]; acc += qw3_iq4nl_val(uint(v) & 15u) * xg[j]; acc += qw3_iq4nl_val(uint(v) >> 4u) * xg[j + 16u]; }
            sum0 += dl * acc;
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
            half d = *((device const half *)blk);
            ushort scales_h = *((device const ushort *)(blk + 2));
            device const uchar *scales_l = blk + 4;
            device const uchar *qs = scales_l + 4 + ib * 16u + il * 8u;
            uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);
            float dl = float(d) * float(int(ls) - 32);
            float acc = 0.0f;
            for (uint j = 0; j < 8u; j++) { uchar v = qs[j]; acc += qw3_iq4nl_val(uint(v) & 15u) * xg[j]; acc += qw3_iq4nl_val(uint(v) >> 4u) * xg[j + 16u]; }
            sum1 += dl * acc;
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
inline float qw3_swiglu_val(device const float *scratch, uint n, uint idx) {
    float g = scratch[idx];
    return (g / (1.0f + exp(-g))) * scratch[n + idx];
}
kernel void qw3_matvec_iq4_xs_swiglu_add_x0(constant qw3_matvec_q8_0_args &args,
                                            constant float &scale,
                                            device const uchar *weights,
                                            device const float *scratch,
                                            device float *x0,
                                            threadgroup float *sh,
                                            uint row [[threadgroup_position_in_grid]],
                                            ushort tid [[thread_index_in_threadgroup]],
                                            ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                            ushort lane [[thread_index_in_simdgroup]],
                                            ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);
    float sum = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = tid; b < n_blocks; b += nt) {
        device const uchar *blk = wr + uint64_t(b) * 136ull;
        half d = *((device const half *)blk);
        ushort scales_h = *((device const ushort *)(blk + 2));
        device const uchar *scales_l = blk + 4;
        device const uchar *qs = scales_l + 4;
        uint xb = b * 256u;
        for (uint ib = 0; ib < 8u; ib++) {
            uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);
            float dl = float(d) * float(int(ls) - 32);
            device const uchar *q = qs + ib * 16u;
            uint xg = xb + ib * 32u;
            for (uint j = 0; j < 16u; j++) { uchar v = q[j]; sum += dl * qw3_iq4nl_val(uint(v) & 15u) * qw3_swiglu_val(scratch, args.n_in, xg + j); sum += dl * qw3_iq4nl_val(uint(v) >> 4u) * qw3_swiglu_val(scratch, args.n_in, xg + j + 16u); }
        }
    }
    sum = simd_sum(sum); if (lane == 0) sh[simd_idx] = sum; threadgroup_barrier(mem_flags::mem_threadgroup); sum = lane < ((uint(nt) + 31u) / 32u) ? sh[lane] : 0.0f; sum = simd_sum(sum); if (tid == 0) x0[row] = x0[row] + sum * scale;
}
kernel void qw3_matvec_q2_k(constant qw3_matvec_q8_0_args &args,
                            device const uchar *weights,
                            device const float *x,
                            device float *out,
                            threadgroup float *sh,
                            uint row [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort simd_idx [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]],
                            ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);
    float sum = 0.0f;
    for (uint b = tid; b < args.n_in / 256u; b += nt) {
        device const uchar *blk = wr + uint64_t(b) * 84ull;
        device const uchar *scales = blk;
        device const uchar *qs = blk + 16u;
        float d = qw3_f16_to_f32(ushort(blk[80]) | (ushort(blk[81]) << 8u));
        float dmin = qw3_f16_to_f32(ushort(blk[82]) | (ushort(blk[83]) << 8u));
        device const float *xx = x + uint64_t(b) * 256ull;
        for (uint half_block = 0; half_block < 2u; half_block++) {
            device const uchar *q = qs + half_block * 32u;
            uint base = half_block * 128u;
            for (uint j = 0; j < 4u; j++) {
                uint sc0 = uint(scales[half_block * 8u + 2u * j]);
                uint sc1 = uint(scales[half_block * 8u + 2u * j + 1u]);
                float dl0 = d * float(sc0 & 15u);
                float ml0 = dmin * float(sc0 >> 4u);
                float dl1 = d * float(sc1 & 15u);
                float ml1 = dmin * float(sc1 >> 4u);
                uint shift = 2u * j;
                for (uint l = 0; l < 16u; l++) {
                    sum += (dl0 * float((uint(q[l]) >> shift) & 3u) - ml0) *
                           xx[base + j * 32u + l];
                    sum += (dl1 * float((uint(q[l + 16u]) >> shift) & 3u) - ml1) *
                           xx[base + j * 32u + l + 16u];
                }
            }
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) sh[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane < ((uint(nt) + 31u) / 32u) ? sh[lane] : 0.0f;
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}
kernel void qw3_matmul_q2_k_batch4(
                            constant qw3_matmul_q8_0_batch_args &args,
                            device const uchar *weights,
                            device const float *x,
                            device float *out,
                            uint2 group [[threadgroup_position_in_grid]],
                            ushort simd_idx [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    uint row = group.x * 4u + uint(simd_idx);
    if (row >= args.n_out) return;
    uint t0 = group.y * 4u;
    uint t1 = min(t0 + 1u, args.n_tokens - 1u);
    uint t2 = min(t0 + 2u, args.n_tokens - 1u);
    uint t3 = min(t0 + 3u, args.n_tokens - 1u);
    device const float *x0 = x + uint64_t(t0) * args.in_stride + args.in_offset;
    device const float *x1 = x + uint64_t(t1) * args.in_stride + args.in_offset;
    device const float *x2 = x + uint64_t(t2) * args.in_stride + args.in_offset;
    device const float *x3 = x + uint64_t(t3) * args.in_stride + args.in_offset;
    device const uchar *wr = weights + uint64_t(row) * args.row_bytes;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (uint b = uint(lane); b < args.n_in / 256u; b += 32u) {
        device const uchar *blk = wr + uint64_t(b) * 84ull;
        device const uchar *scales = blk;
        device const uchar *qs = blk + 16u;
        float d = qw3_f16_to_f32(
            ushort(blk[80]) | (ushort(blk[81]) << 8u));
        float dmin = qw3_f16_to_f32(
            ushort(blk[82]) | (ushort(blk[83]) << 8u));
        uint xb = b * 256u;
        for (uint half_block = 0; half_block < 2u; half_block++) {
            device const uchar *q = qs + half_block * 32u;
            uint base = half_block * 128u;
            for (uint j = 0; j < 4u; j++) {
                uint sc0 = uint(scales[half_block * 8u + 2u * j]);
                uint sc1 = uint(scales[half_block * 8u + 2u * j + 1u]);
                float dl0 = d * float(sc0 & 15u);
                float ml0 = dmin * float(sc0 >> 4u);
                float dl1 = d * float(sc1 & 15u);
                float ml1 = dmin * float(sc1 >> 4u);
                uint shift = 2u * j;
                for (uint l = 0; l < 16u; l++) {
                    uint i0 = xb + base + j * 32u + l;
                    uint i1 = i0 + 16u;
                    float w0 = dl0 * float((uint(q[l]) >> shift) & 3u) - ml0;
                    float w1 = dl1 * float((uint(q[l + 16u]) >> shift) & 3u) - ml1;
                    s0 += w0 * x0[i0] + w1 * x0[i1];
                    s1 += w0 * x1[i0] + w1 * x1[i1];
                    s2 += w0 * x2[i0] + w1 * x2[i1];
                    s3 += w0 * x3[i0] + w1 * x3[i1];
                }
            }
        }
    }
    s0 = simd_sum(s0); s1 = simd_sum(s1);
    s2 = simd_sum(s2); s3 = simd_sum(s3);
    if (lane == 0) {
        out[uint64_t(t0) * args.out_stride + args.out_offset + row] = s0;
        if (t0 + 1u < args.n_tokens)
            out[uint64_t(t1) * args.out_stride + args.out_offset + row] = s1;
        if (t0 + 2u < args.n_tokens)
            out[uint64_t(t2) * args.out_stride + args.out_offset + row] = s2;
        if (t0 + 3u < args.n_tokens)
            out[uint64_t(t3) * args.out_stride + args.out_offset + row] = s3;
    }
}
kernel void qw3_matvec_q6_k(constant qw3_matvec_q8_0_args &args,
                            device const uchar *weights,
                            device const float *x,
                            device float *out,
                            threadgroup float *sh,
                            uint row [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort simd_idx [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]],
                            ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);
    float sum = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = tid; b < n_blocks; b += nt) {
        device const uchar *blk = wr + uint64_t(b) * 210ull;
        device const uchar *ql = blk;
        device const uchar *qh = ql + 128u;
        device const uchar *scb = qh + 64u;
        device const char *sc = (device const char *)scb;
        ushort dbits = ushort(scb[16u]) | (ushort(scb[17u]) << 8u);
        float d = qw3_f16_to_f32(dbits);
        device const float *xx = x + uint64_t(b) * 256ull;
        for (uint n = 0; n < 256u; n += 128u) {
            for (uint l = 0; l < 32u; l++) {
                uint is = l / 16u;
                int q1 = int((uint(ql[l +  0u]) & 15u) | (((uint(qh[l]) >> 0u) & 3u) << 4u)) - 32;
                int q2 = int((uint(ql[l + 32u]) & 15u) | (((uint(qh[l]) >> 2u) & 3u) << 4u)) - 32;
                int q3 = int((uint(ql[l +  0u]) >> 4u) | (((uint(qh[l]) >> 4u) & 3u) << 4u)) - 32;
                int q4 = int((uint(ql[l + 32u]) >> 4u) | (((uint(qh[l]) >> 6u) & 3u) << 4u)) - 32;
                sum += d * float(sc[is + 0u]) * float(q1) * xx[n + l +  0u];
                sum += d * float(sc[is + 2u]) * float(q2) * xx[n + l + 32u];
                sum += d * float(sc[is + 4u]) * float(q3) * xx[n + l + 64u];
                sum += d * float(sc[is + 6u]) * float(q4) * xx[n + l + 96u];
            }
            ql += 64u;
            qh += 32u;
            sc += 8u;
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) sh[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane < ((uint(nt) + 31u) / 32u) ? sh[lane] : 0.0f;
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}
kernel void qw3_matvec_q5_k(constant qw3_matvec_q8_0_args &args,
                            device const uchar *weights,
                            device const float *x,
                            device float *out,
                            threadgroup float *sh,
                            uint row [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort simd_idx [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]],
                            ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);
    float sum = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = tid; b < n_blocks; b += nt) {
        device const uchar *blk = wr + uint64_t(b) * 176ull;
        ushort dbits = ushort(blk[0]) | (ushort(blk[1]) << 8u);
        ushort dmin_bits = ushort(blk[2]) | (ushort(blk[3]) << 8u);
        float d = qw3_f16_to_f32(dbits);
        float dmin = qw3_f16_to_f32(dmin_bits);
        device const uchar *scales = blk + 4u;
        device const uchar *qh = blk + 16u;
        device const uchar *ql = blk + 48u;
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
            uint high = 1u << g;
            uint xbase = g * 32u;
            for (uint l = 0; l < 32u; l++) {
                uint packed = uint(ql[qbase + l]);
                uint q = ((g & 1u) == 0u ? (packed & 15u) : (packed >> 4u)) +
                         ((uint(qh[l]) & high) != 0u ? 16u : 0u);
                sum += (dl * float(q) - ml) * xx[xbase + l];
            }
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) sh[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane < ((uint(nt) + 31u) / 32u) ? sh[lane] : 0.0f;
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}
kernel void qw3_matvec_q4_k(constant qw3_matvec_q8_0_args &args,
                            device const uchar *weights,
                            device const float *x,
                            device float *out,
                            threadgroup float *sh,
                            uint row [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort simd_idx [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]],
                            ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);
    float sum = 0.0f;
    for (uint b = tid; b < args.n_in / 256u; b += nt) {
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
    sum = lane < ((uint(nt) + 31u) / 32u) ? sh[lane] : 0.0f;
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}
kernel void qw3_matvec_iq4_nl(constant qw3_matvec_q8_0_args &args,
                              device const uchar *weights,
                              device const float *x,
                              device float *out,
                              threadgroup float *sh,
                              uint row [[threadgroup_position_in_grid]],
                              ushort tid [[thread_index_in_threadgroup]],
                              ushort simd_idx [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]],
                              ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);
    float sum = 0.0f;
    for (uint b = tid; b < args.n_in / 32u; b += nt) {
        device const uchar *blk = wr + uint64_t(b) * 18ull;
        float d = float(*((device const half *)blk));
        device const uchar *qs = blk + 2u;
        device const float *xx = x + uint64_t(b) * 32ull;
        for (uint j = 0; j < 16u; j++) {
            uint q = uint(qs[j]);
            sum += d * qw3_iq4nl_table[q & 15u] * xx[j];
            sum += d * qw3_iq4nl_table[q >> 4u] * xx[j + 16u];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) sh[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane < ((uint(nt) + 31u) / 32u) ? sh[lane] : 0.0f;
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}
kernel void qw3_matvec_q3_k(constant qw3_matvec_q8_0_args &args,
                            device const uchar *weights,
                            device const float *x,
                            device float *out,
                            threadgroup float *sh,
                            uint row [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort simd_idx [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]],
                            ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);
    float sum = 0.0f;
    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0f0f0f0fu;
    for (uint b = tid; b < args.n_in / 256u; b += nt) {
        device const uchar *blk = wr + uint64_t(b) * 110ull;
        device const uchar *packed = blk + 96u;
        uint aux0 = uint(packed[0]) | (uint(packed[1]) << 8u) |
                    (uint(packed[2]) << 16u) | (uint(packed[3]) << 24u);
        uint aux1 = uint(packed[4]) | (uint(packed[5]) << 8u) |
                    (uint(packed[6]) << 16u) | (uint(packed[7]) << 24u);
        uint tmp = uint(packed[8]) | (uint(packed[9]) << 8u) |
                   (uint(packed[10]) << 16u) | (uint(packed[11]) << 24u);
        uint aux[4];
        aux[2] = ((aux0 >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u);
        aux[3] = ((aux1 >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u);
        aux[0] = (aux0 & kmask2) | (((tmp >> 0u) & kmask1) << 4u);
        aux[1] = (aux1 & kmask2) | (((tmp >> 2u) & kmask1) << 4u);
        float d = float(*((device const half *)(blk + 108u)));
        device const float *xx = x + uint64_t(b) * 256ull;
        for (uint local = 0; local < 256u; local++) {
            uint half_block = local >> 7u;
            uint within = local & 127u;
            uint group = within >> 5u;
            uint upper = (within & 31u) >= 16u ? 1u : 0u;
            uint l = within & 15u;
            uint scale_index = half_block * 8u + group * 2u + upper;
            uint scale = (aux[scale_index >> 2u] >>
                          (8u * (scale_index & 3u))) & 255u;
            uint shift = group * 2u;
            uint mask = 1u << (half_block * 4u + group);
            uint q_index = half_block * 32u + upper * 16u + l;
            int low = int((uint(blk[32u + q_index]) >> shift) & 3u);
            int q = low - ((uint(blk[upper * 16u + l]) & mask) ? 0 : 4);
            sum += d * float(int(scale) - 32) * float(q) * xx[local];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) sh[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane < ((uint(nt) + 31u) / 32u) ? sh[lane] : 0.0f;
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}
/* Reuse each Q3_K block across four prompt rows. This is also the fallback
 * for mixed dense FFNs whose Q3_K projection would otherwise run x1. */
kernel void qw3_matmul_q3_k_batch4(
    constant qw3_matmul_q8_0_batch_args &args,
    device const uchar *weights,
    device const float *x,
    device float *out,
    uint2 group [[threadgroup_position_in_grid]],
    ushort simd_idx [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x * 4u + uint(simd_idx);
    if (row >= args.n_out) return;
    const uint t0 = group.y * 4u;
    device const uchar *wr = weights + uint64_t(row) * args.row_bytes;
    float4 sum = float4(0.0f);
    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0f0f0f0fu;
    for (uint b = uint(lane); b < args.n_in / 256u; b += 32u) {
        device const uchar *blk = wr + uint64_t(b) * 110ull;
        device const uchar *packed = blk + 96u;
        uint aux0 = uint(packed[0]) | (uint(packed[1]) << 8u) |
                    (uint(packed[2]) << 16u) | (uint(packed[3]) << 24u);
        uint aux1 = uint(packed[4]) | (uint(packed[5]) << 8u) |
                    (uint(packed[6]) << 16u) | (uint(packed[7]) << 24u);
        uint tmp = uint(packed[8]) | (uint(packed[9]) << 8u) |
                   (uint(packed[10]) << 16u) | (uint(packed[11]) << 24u);
        uint aux[4];
        aux[2] = ((aux0 >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u);
        aux[3] = ((aux1 >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u);
        aux[0] = (aux0 & kmask2) | (((tmp >> 0u) & kmask1) << 4u);
        aux[1] = (aux1 & kmask2) | (((tmp >> 2u) & kmask1) << 4u);
        const float d = float(*((device const half *)(blk + 108u)));
        for (uint local = 0; local < 256u; local++) {
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
            const int q = low - ((uint(blk[upper * 16u + l]) & mask) ? 0 : 4);
            const float wv = d * float(int(scale) - 32) * float(q);
            const uint xi = b * 256u + local;
            for (uint r = 0; r < 4u; r++) {
                if (t0 + r < args.n_tokens)
                    sum[r] += wv * x[uint64_t(t0 + r) * args.in_stride +
                                     args.in_offset + xi];
            }
        }
    }
    sum.x = simd_sum(sum.x); sum.y = simd_sum(sum.y);
    sum.z = simd_sum(sum.z); sum.w = simd_sum(sum.w);
    if (lane == 0) {
        for (uint r = 0; r < 4u; r++) {
            if (t0 + r < args.n_tokens)
                out[uint64_t(t0 + r) * args.out_stride + args.out_offset + row] = sum[r];
        }
    }
}
#ifdef QW3_METAL_HAS_TENSOR
/* Decode a 64x32 Q3_K weight tile once, then multiply it by 32 prompt rows.
 * The input stays F32 to avoid an extra conversion and rounding pass. */
kernel void qw3_matmul_iq4_nl_mpp_direct(
    constant qw3_matmul_q8_0_batch_args &args,
    device const uchar *weights,
    device const float *x,
    device float *out,
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    const int r0 = int(group.y) * NR0;
    const int r1 = int(group.x) * NR1;
    if (r0 >= int(args.n_out) || r1 >= int(args.n_tokens)) return;
    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    device float *ptrB = (device float *)x + args.in_offset;
    auto tB = tensor(ptrB,
                    dextents<int32_t, 2>(int(args.n_in), int(args.n_tokens)),
                    array<int, 2>({1, int(args.in_stride)}));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<
        decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); i++)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    for (uint loop_k = 0; loop_k < args.n_in; loop_k += NK) {
        const int row = int(tid) / 2;
        const int half_k = int(tid) % 2;
        if (r0 + row < int(args.n_out)) {
            const uint k0 = loop_k + uint(half_k * 16);
            device const uchar *wrow = weights +
                uint64_t(r0 + row) * args.row_bytes;
            device const uchar *blk = wrow + uint64_t(k0 >> 5u) * 18ull;
            const float d = float(*((device const half *)blk));
            device const uchar *qs = blk + 2u;
            for (int i = 0; i < 16; i++) {
                const uint local = (k0 + uint(i)) & 31u;
                const uchar packed = qs[local & 15u];
                const uint q = local < 16u ? uint(packed) & 15u :
                                            uint(packed) >> 4u;
                sa[row * NK + half_k * 16 + i] =
                    half(d * qw3_iq4nl_val(q));
            }
        } else {
            for (int i = 0; i < 16; i++)
                sa[row * NK + half_k * 16 + i] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(int(loop_k), r1);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out), int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    cT.store(tD.slice(r0, r1));
}

kernel void qw3_matmul_q2_k_mpp_direct(
    constant qw3_matmul_q8_0_batch_args &args,
    device const uchar *weights,
    device const float *x,
    device float *out,
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    const int r0 = int(group.y) * NR0;
    const int r1 = int(group.x) * NR1;
    if (r0 >= int(args.n_out) || r1 >= int(args.n_tokens)) return;
    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    device float *ptrB = (device float *)x + args.in_offset;
    auto tB = tensor(ptrB,
                    dextents<int32_t, 2>(int(args.n_in), int(args.n_tokens)),
                    array<int, 2>({1, int(args.in_stride)}));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<
        decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); i++)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    for (uint loop_k = 0; loop_k < args.n_in; loop_k += NK) {
        const int row = int(tid) / 2;
        const int half_k = int(tid) % 2;
        if (r0 + row < int(args.n_out)) {
            const uint k0 = loop_k + uint(half_k * 16);
            device const uchar *wrow = weights +
                uint64_t(r0 + row) * args.row_bytes;
            device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 84ull;
            device const uchar *scales = blk;
            device const uchar *qs = blk + 16u;
            const float d = qw3_f16_to_f32(
                ushort(blk[80]) | (ushort(blk[81]) << 8u));
            const float dmin = qw3_f16_to_f32(
                ushort(blk[82]) | (ushort(blk[83]) << 8u));
            for (int i = 0; i < 16; i++) {
                const uint local = (k0 + uint(i)) & 255u;
                const uint half_block = local >> 7u;
                const uint within = local & 127u;
                const uint j = within >> 5u;
                const uint within32 = within & 31u;
                const uint upper = within32 >> 4u;
                const uint l = within32 & 15u;
                const uint sc = uint(scales[
                    half_block * 8u + 2u * j + upper]);
                const uint q = (uint(qs[half_block * 32u +
                                    upper * 16u + l]) >> (2u * j)) & 3u;
                sa[row * NK + half_k * 16 + i] =
                    half(d * float(sc & 15u) * float(q) -
                         dmin * float(sc >> 4u));
            }
        } else {
            for (int i = 0; i < 16; i++)
                sa[row * NK + half_k * 16 + i] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(int(loop_k), r1);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out), int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    cT.store(tD.slice(r0, r1));
}

kernel void qw3_matmul_q3_k_mpp_direct(
    constant qw3_matmul_q8_0_batch_args &args,
    device const uchar *weights,
    device const float *x,
    device float *out,
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    const int r0 = int(group.y) * NR0;
    const int r1 = int(group.x) * NR1;
    if (r0 >= int(args.n_out) || r1 >= int(args.n_tokens)) return;
    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    device float *ptrB = (device float *)x + args.in_offset;
    auto tB = tensor(ptrB,
                    dextents<int32_t, 2>(int(args.n_in), int(args.n_tokens)),
                    array<int, 2>({1, int(args.in_stride)}));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<
        decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); i++)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0f0f0f0fu;
    for (uint loop_k = 0; loop_k < args.n_in; loop_k += NK) {
        const int row = int(tid) / 2;
        const int half_k = int(tid) % 2;
        if (r0 + row < int(args.n_out)) {
            const uint k0 = loop_k + uint(half_k * 16);
            device const uchar *wrow = weights +
                uint64_t(r0 + row) * args.row_bytes;
            device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 110ull;
            device const uchar *packed = blk + 96u;
            uint aux0 = uint(packed[0]) | (uint(packed[1]) << 8u) |
                        (uint(packed[2]) << 16u) | (uint(packed[3]) << 24u);
            uint aux1 = uint(packed[4]) | (uint(packed[5]) << 8u) |
                        (uint(packed[6]) << 16u) | (uint(packed[7]) << 24u);
            uint tmp = uint(packed[8]) | (uint(packed[9]) << 8u) |
                       (uint(packed[10]) << 16u) | (uint(packed[11]) << 24u);
            uint aux[4];
            aux[2] = ((aux0 >> 4u) & kmask2) |
                     (((tmp >> 4u) & kmask1) << 4u);
            aux[3] = ((aux1 >> 4u) & kmask2) |
                     (((tmp >> 6u) & kmask1) << 4u);
            aux[0] = (aux0 & kmask2) |
                     (((tmp >> 0u) & kmask1) << 4u);
            aux[1] = (aux1 & kmask2) |
                     (((tmp >> 2u) & kmask1) << 4u);
            const float d = float(*((device const half *)(blk + 108u)));
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
        auto mB = tB.slice(int(loop_k), r1);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out), int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    cT.store(tD.slice(r0, r1));
}

/* Q4_K and Q5_K share scale/min packing; Q5_K adds one high bit per value.
 * A uniform format flag keeps the 64x32 tiled path usable for both GGUF types. */
struct qw3_matmul_k_batch_args {
    uint n_in, n_out, row_bytes, n_tokens;
    uint in_offset, in_stride, out_offset, out_stride;
    uint is_q5;
};
kernel void qw3_matmul_q4_q5_k_mpp_direct(
    constant qw3_matmul_k_batch_args &args,
    device const uchar *weights,
    device const float *x,
    device float *out,
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    const int r0 = int(group.y) * NR0;
    const int r1 = int(group.x) * NR1;
    if (r0 >= int(args.n_out) || r1 >= int(args.n_tokens)) return;
    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    device float *ptrB = (device float *)x + args.in_offset;
    auto tB = tensor(ptrB,
                    dextents<int32_t, 2>(int(args.n_in), int(args.n_tokens)),
                    array<int, 2>({1, int(args.in_stride)}));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<
        decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); i++)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    for (uint loop_k = 0; loop_k < args.n_in; loop_k += NK) {
        const int row = int(tid) / 2;
        const int half_k = int(tid) % 2;
        if (r0 + row < int(args.n_out)) {
            const uint k0 = loop_k + uint(half_k * 16);
            device const uchar *wrow = weights +
                uint64_t(r0 + row) * args.row_bytes;
            device const uchar *blk = wrow +
                uint64_t(k0 >> 8u) * (args.is_q5 ? 176ull : 144ull);
            const float d = qw3_f16_to_f32(
                ushort(blk[0]) | (ushort(blk[1]) << 8u));
            const float dmin = qw3_f16_to_f32(
                ushort(blk[2]) | (ushort(blk[3]) << 8u));
            device const uchar *scales = blk + 4u;
            device const uchar *qh = blk + 16u;
            device const uchar *ql = blk + (args.is_q5 ? 48u : 16u);
            for (int i = 0; i < 16; i++) {
                const uint local = (k0 + uint(i)) & 255u;
                const uint g = local >> 5u;
                const uint l = local & 31u;
                uint sc, mn;
                if (g < 4u) {
                    sc = uint(scales[g]) & 63u;
                    mn = uint(scales[g + 4u]) & 63u;
                } else {
                    sc = (uint(scales[g + 4u]) & 15u) |
                         ((uint(scales[g - 4u]) >> 6u) << 4u);
                    mn = (uint(scales[g + 4u]) >> 4u) |
                         ((uint(scales[g]) >> 6u) << 4u);
                }
                const uint packed = uint(ql[(g >> 1u) * 32u + l]);
                uint q = (g & 1u) ? packed >> 4u : packed & 15u;
                if (args.is_q5 && (uint(qh[l]) & (1u << g))) q += 16u;
                sa[row * NK + half_k * 16 + i] =
                    half(d * float(sc) * float(q) - dmin * float(mn));
            }
        } else {
            for (int i = 0; i < 16; i++)
                sa[row * NK + half_k * 16 + i] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(int(loop_k), r1);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out), int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    cT.store(tD.slice(r0, r1));
}
#endif
inline float qw3_iq2_grid_value(ushort packed, uint j) {
    constexpr float values[4] = { 8.0f, 25.0f, 43.0f, 0.0f };
    return values[(uint(packed) >> (2u * j)) & 3u];
}
inline float qw3_iq3xxs_grid_value(ushort packed, uint j) {
    constexpr float values[8] = {
        4.0f, 12.0f, 20.0f, 28.0f, 36.0f, 44.0f, 52.0f, 62.0f
    };
    return values[(uint(packed) >> (3u * j)) & 7u];
}
kernel void qw3_matvec_iq2_xxs(constant qw3_matvec_q8_0_args &args,
                               device const uchar *weights,
                               device const float *x,
                               device float *out,
                               device const ushort *grid_table,
                               device const uchar *sign_table,
                               threadgroup float *sh,
                               uint row [[threadgroup_position_in_grid]],
                               ushort tid [[thread_index_in_threadgroup]],
                               ushort simd_idx [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]],
                               ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);
    float sum = 0.0f;
    for (uint b = tid; b < args.n_in / 256u; b += nt) {
        device const uchar *blk = wr + uint64_t(b) * 66ull;
        float d = qw3_f16_to_f32(ushort(blk[0]) | (ushort(blk[1]) << 8u));
        device const uchar *qs = blk + 2u;
        device const float *xx = x + uint64_t(b) * 256ull;
        for (uint ib = 0; ib < 8u; ib++) {
            device const uchar *q = qs + ib * 8u;
            uint packed = uint(q[4]) | (uint(q[5]) << 8u) |
                          (uint(q[6]) << 16u) | (uint(q[7]) << 24u);
            float db = d * (0.5f + float(packed >> 28u)) * 0.25f;
            for (uint l = 0; l < 4u; l++) {
                ushort grid = grid_table[uint(q[l])];
                uint signs = uint(sign_table[(packed >> (7u * l)) & 127u]);
                uint xb = ib * 32u + l * 8u;
                for (uint j = 0; j < 8u; j++) {
                    float sign = (signs & (1u << j)) != 0u ? -1.0f : 1.0f;
                    sum += db * qw3_iq2_grid_value(grid, j) * sign * xx[xb + j];
                }
            }
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) sh[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane < ((uint(nt) + 31u) / 32u) ? sh[lane] : 0.0f;
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}
kernel void qw3_matvec_iq2_xs(constant qw3_matvec_q8_0_args &args,
                              device const uchar *weights,
                              device const float *x,
                              device float *out,
                              device const ushort *grid_table,
                              device const uchar *sign_table,
                              threadgroup float *sh,
                              uint row [[threadgroup_position_in_grid]],
                              ushort tid [[thread_index_in_threadgroup]],
                              ushort simd_idx [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]],
                              ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);
    float sum = 0.0f;
    for (uint b = tid; b < args.n_in / 256u; b += nt) {
        device const uchar *blk = wr + uint64_t(b) * 74ull;
        ushort dbits = ushort(blk[0]) | (ushort(blk[1]) << 8u);
        float d = qw3_f16_to_f32(dbits);
        device const uchar *qs = blk + 2u;
        device const uchar *scales = qs + 64u;
        device const float *xx = x + uint64_t(b) * 256ull;
        for (uint ib = 0; ib < 8u; ib++) {
            uint scale_byte = uint(scales[ib]);
            for (uint l = 0; l < 4u; l++) {
                uint off = 2u * (4u * ib + l);
                uint q = uint(qs[off]) | (uint(qs[off + 1u]) << 8u);
                ushort grid = grid_table[q & 511u];
                uint signs = uint(sign_table[q >> 9u]);
                float db = d * (0.5f + float((scale_byte >> (4u * (l >> 1u))) & 15u)) * 0.25f;
                uint xb = ib * 32u + l * 8u;
                for (uint j = 0; j < 8u; j++) {
                    float sign = (signs & (1u << j)) != 0u ? -1.0f : 1.0f;
                    sum += db * qw3_iq2_grid_value(grid, j) * sign * xx[xb + j];
                }
            }
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) sh[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane < ((uint(nt) + 31u) / 32u) ? sh[lane] : 0.0f;
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}
kernel void qw3_matvec_iq2_s(constant qw3_matvec_q8_0_args &args,
                             device const uchar *weights,
                             device const float *x,
                             device float *out,
                             device const ushort *grid_table,
                             device const uchar *sign_table,
                             threadgroup float *sh,
                             uint row [[threadgroup_position_in_grid]],
                             ushort tid [[thread_index_in_threadgroup]],
                             ushort simd_idx [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]],
                             ushort nt [[threads_per_threadgroup]]) {
    (void)sign_table;
    if (row >= args.n_out) return;
    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);
    float sum = 0.0f;
    for (uint b = tid; b < args.n_in / 256u; b += nt) {
        device const uchar *blk = wr + uint64_t(b) * 82ull;
        ushort dbits = ushort(blk[0]) | (ushort(blk[1]) << 8u);
        float d = qw3_f16_to_f32(dbits);
        device const uchar *qs = blk + 2u;
        device const uchar *signs = qs + 32u;
        device const uchar *qh = qs + 64u;
        device const uchar *scales = qh + 8u;
        device const float *xx = x + uint64_t(b) * 256ull;
        for (uint ib = 0; ib < 8u; ib++) {
            uint scale_byte = uint(scales[ib]);
            for (uint l = 0; l < 4u; l++) {
                uint idx = uint(qs[4u * ib + l]) |
                    ((uint(qh[ib]) << (8u - 2u * l)) & 0x300u);
                ushort grid = grid_table[idx];
                uint sign_bits = uint(signs[4u * ib + l]);
                float db = d * (0.5f + float((scale_byte >> (4u * (l >> 1u))) & 15u)) * 0.25f;
                uint xb = ib * 32u + l * 8u;
                for (uint j = 0; j < 8u; j++) {
                    float sign = (sign_bits & (1u << j)) != 0u ? -1.0f : 1.0f;
                    sum += db * qw3_iq2_grid_value(grid, j) * sign * xx[xb + j];
                }
            }
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) sh[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane < ((uint(nt) + 31u) / 32u) ? sh[lane] : 0.0f;
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}
/* Four prompt rows share each IQ2_S weight decode. A simdgroup owns one
 * output row; each lane handles complete 256-value quant blocks. */
kernel void qw3_matmul_iq2_s_batch4(
    constant qw3_matmul_q8_0_batch_args &args,
    device const uchar *weights,
    device const float *x,
    device float *out,
    device const ushort *grid_table,
    uint2 group [[threadgroup_position_in_grid]],
    ushort simd_idx [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x * 4u + uint(simd_idx);
    if (row >= args.n_out) return;
    const uint t0 = group.y * 4u;
    const uint t1 = min(t0 + 1u, args.n_tokens - 1u);
    const uint t2 = min(t0 + 2u, args.n_tokens - 1u);
    const uint t3 = min(t0 + 3u, args.n_tokens - 1u);
    device const float *x0 = x + uint64_t(t0) * args.in_stride + args.in_offset;
    device const float *x1 = x + uint64_t(t1) * args.in_stride + args.in_offset;
    device const float *x2 = x + uint64_t(t2) * args.in_stride + args.in_offset;
    device const float *x3 = x + uint64_t(t3) * args.in_stride + args.in_offset;
    device const uchar *wr = weights + uint64_t(row) * args.row_bytes;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (uint b = uint(lane); b < args.n_in / 256u; b += 32u) {
        device const uchar *blk = wr + uint64_t(b) * 82ull;
        const float d = qw3_f16_to_f32(
            ushort(blk[0]) | (ushort(blk[1]) << 8u));
        device const uchar *qs = blk + 2u;
        device const uchar *signs = qs + 32u;
        device const uchar *qh = qs + 64u;
        device const uchar *scales = qh + 8u;
        const uint base = b * 256u;
        for (uint ib = 0; ib < 8u; ib++) {
            const uint scale_byte = uint(scales[ib]);
            for (uint l = 0; l < 4u; l++) {
                const uint idx = uint(qs[4u * ib + l]) |
                    ((uint(qh[ib]) << (8u - 2u * l)) & 0x300u);
                const ushort grid = grid_table[idx];
                const uint sign_bits = uint(signs[4u * ib + l]);
                const float db = d *
                    (0.5f + float((scale_byte >> (4u * (l >> 1u))) & 15u)) *
                    0.25f;
                const uint xb = base + ib * 32u + l * 8u;
                for (uint j = 0; j < 8u; j++) {
                    const float sign = (sign_bits & (1u << j)) ? -1.0f : 1.0f;
                    const float wv = db * qw3_iq2_grid_value(grid, j) * sign;
                    s0 += wv * x0[xb + j];
                    s1 += wv * x1[xb + j];
                    s2 += wv * x2[xb + j];
                    s3 += wv * x3[xb + j];
                }
            }
        }
    }
    s0 = simd_sum(s0); s1 = simd_sum(s1);
    s2 = simd_sum(s2); s3 = simd_sum(s3);
    if (lane == 0) {
        out[uint64_t(t0) * args.out_stride + args.out_offset + row] = s0;
        if (t0 + 1u < args.n_tokens)
            out[uint64_t(t0 + 1u) * args.out_stride + args.out_offset + row] = s1;
        if (t0 + 2u < args.n_tokens)
            out[uint64_t(t0 + 2u) * args.out_stride + args.out_offset + row] = s2;
        if (t0 + 3u < args.n_tokens)
            out[uint64_t(t0 + 3u) * args.out_stride + args.out_offset + row] = s3;
    }
}
#ifdef QW3_METAL_HAS_TENSOR
kernel void qw3_matmul_iq2_xxs_mpp_direct(
    constant qw3_matmul_q8_0_batch_args &args,
    device const uchar *weights,
    device const float *x,
    device float *out,
    device const ushort *grid_table,
    device const uchar *sign_table,
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    const int r0 = int(group.y) * NR0;
    const int r1 = int(group.x) * NR1;
    if (r0 >= int(args.n_out) || r1 >= int(args.n_tokens)) return;
    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    device float *ptrB = (device float *)x + args.in_offset;
    auto tB = tensor(ptrB,
                    dextents<int32_t, 2>(int(args.n_in), int(args.n_tokens)),
                    array<int, 2>({1, int(args.in_stride)}));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<
        decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); i++)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    for (uint loop_k = 0; loop_k < args.n_in; loop_k += NK) {
        const int row = int(tid) / 2;
        const int half_k = int(tid) % 2;
        if (r0 + row < int(args.n_out)) {
            const uint k0 = loop_k + uint(half_k * 16);
            device const uchar *wrow = weights +
                uint64_t(r0 + row) * args.row_bytes;
            device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 66ull;
            const float d = qw3_f16_to_f32(
                ushort(blk[0]) | (ushort(blk[1]) << 8u));
            device const uchar *qs = blk + 2u;
            for (int i = 0; i < 16; i++) {
                const uint local = (k0 + uint(i)) & 255u;
                const uint ib = local >> 5u;
                const uint l = (local & 31u) >> 3u;
                const uint j = local & 7u;
                device const uchar *q = qs + ib * 8u;
                const uint packed = uint(q[4]) | (uint(q[5]) << 8u) |
                    (uint(q[6]) << 16u) | (uint(q[7]) << 24u);
                const uint sign_bits =
                    uint(sign_table[(packed >> (7u * l)) & 127u]);
                const float db = d *
                    (0.5f + float(packed >> 28u)) * 0.25f;
                const float sign = (sign_bits & (1u << j)) ? -1.0f : 1.0f;
                sa[row * NK + half_k * 16 + i] = half(
                    db * qw3_iq2_grid_value(grid_table[uint(q[l])], j) * sign);
            }
        } else {
            for (int i = 0; i < 16; i++)
                sa[row * NK + half_k * 16 + i] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(int(loop_k), r1);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out), int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    cT.store(tD.slice(r0, r1));
}

kernel void qw3_matmul_iq2_xs_mpp_direct(
    constant qw3_matmul_q8_0_batch_args &args,
    device const uchar *weights,
    device const float *x,
    device float *out,
    device const ushort *grid_table,
    device const uchar *sign_table,
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    const int r0 = int(group.y) * NR0;
    const int r1 = int(group.x) * NR1;
    if (r0 >= int(args.n_out) || r1 >= int(args.n_tokens)) return;
    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    device float *ptrB = (device float *)x + args.in_offset;
    auto tB = tensor(ptrB,
                    dextents<int32_t, 2>(int(args.n_in), int(args.n_tokens)),
                    array<int, 2>({1, int(args.in_stride)}));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<
        decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); i++)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    for (uint loop_k = 0; loop_k < args.n_in; loop_k += NK) {
        const int row = int(tid) / 2;
        const int half_k = int(tid) % 2;
        if (r0 + row < int(args.n_out)) {
            const uint k0 = loop_k + uint(half_k * 16);
            device const uchar *wrow = weights +
                uint64_t(r0 + row) * args.row_bytes;
            device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 74ull;
            const float d = qw3_f16_to_f32(
                ushort(blk[0]) | (ushort(blk[1]) << 8u));
            device const uchar *qs = blk + 2u;
            device const uchar *scales = qs + 64u;
            for (int i = 0; i < 16; i++) {
                const uint local = (k0 + uint(i)) & 255u;
                const uint ib = local >> 5u;
                const uint l = (local & 31u) >> 3u;
                const uint j = local & 7u;
                const uint off = 2u * (4u * ib + l);
                const uint q = uint(qs[off]) | (uint(qs[off + 1u]) << 8u);
                const uint sign_bits = uint(sign_table[q >> 9u]);
                const float db = d *
                    (0.5f + float((uint(scales[ib]) >>
                                  (4u * (l >> 1u))) & 15u)) * 0.25f;
                const float sign = (sign_bits & (1u << j)) ? -1.0f : 1.0f;
                sa[row * NK + half_k * 16 + i] = half(
                    db * qw3_iq2_grid_value(grid_table[q & 511u], j) * sign);
            }
        } else {
            for (int i = 0; i < 16; i++)
                sa[row * NK + half_k * 16 + i] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(int(loop_k), r1);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out), int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    cT.store(tD.slice(r0, r1));
}

kernel void qw3_matmul_iq2_s_mpp_direct(
    constant qw3_matmul_q8_0_batch_args &args,
    device const uchar *weights,
    device const float *x,
    device float *out,
    device const ushort *grid_table,
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    const int r0 = int(group.y) * NR0;
    const int r1 = int(group.x) * NR1;
    if (r0 >= int(args.n_out) || r1 >= int(args.n_tokens)) return;
    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    device float *ptrB = (device float *)x + args.in_offset;
    auto tB = tensor(ptrB,
                    dextents<int32_t, 2>(int(args.n_in), int(args.n_tokens)),
                    array<int, 2>({1, int(args.in_stride)}));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<
        decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); i++)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    for (uint loop_k = 0; loop_k < args.n_in; loop_k += NK) {
        const int row = int(tid) / 2;
        const int half_k = int(tid) % 2;
        if (r0 + row < int(args.n_out)) {
            const uint k0 = loop_k + uint(half_k * 16);
            device const uchar *wrow = weights +
                uint64_t(r0 + row) * args.row_bytes;
            device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 82ull;
            const float d = qw3_f16_to_f32(
                ushort(blk[0]) | (ushort(blk[1]) << 8u));
            device const uchar *qs = blk + 2u;
            device const uchar *signs = qs + 32u;
            device const uchar *qh = qs + 64u;
            device const uchar *scales = qh + 8u;
            for (int i = 0; i < 16; i++) {
                const uint local = (k0 + uint(i)) & 255u;
                const uint ib = local >> 5u;
                const uint l = (local & 31u) >> 3u;
                const uint j = local & 7u;
                const uint idx = uint(qs[4u * ib + l]) |
                    ((uint(qh[ib]) << (8u - 2u * l)) & 0x300u);
                const uint sign_bits = uint(signs[4u * ib + l]);
                const float db = d *
                    (0.5f + float((uint(scales[ib]) >>
                                  (4u * (l >> 1u))) & 15u)) * 0.25f;
                const float sign = (sign_bits & (1u << j)) ? -1.0f : 1.0f;
                sa[row * NK + half_k * 16 + i] = half(
                    db * qw3_iq2_grid_value(grid_table[idx], j) * sign);
            }
        } else {
            for (int i = 0; i < 16; i++)
                sa[row * NK + half_k * 16 + i] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(int(loop_k), r1);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out), int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    cT.store(tD.slice(r0, r1));
}
#endif
inline float qw3_iq3xxs_dot_block(device const uchar *blk,
                                  device const float *xx,
                                  device const ushort *grid_table,
                                  device const uchar *sign_table) {
    ushort dbits = ushort(blk[0]) | (ushort(blk[1]) << 8u);
    float d = qw3_f16_to_f32(dbits);
    device const uchar *qs = blk + 2u;
    device const uchar *ss = qs + 64u;
    float sum = 0.0f;
    for (uint ib = 0; ib < 8u; ib++) {
        device const uchar *sp = ss + 4u * ib;
        uint packed = uint(sp[0]) | (uint(sp[1]) << 8u) |
                      (uint(sp[2]) << 16u) | (uint(sp[3]) << 24u);
        float db = d * (0.5f + float(packed >> 28u)) * 0.5f;
        for (uint l = 0; l < 4u; l++) {
            ushort grid1 = grid_table[uint(qs[8u * ib + 2u * l])];
            ushort grid2 = grid_table[uint(qs[8u * ib + 2u * l + 1u])];
            uint sign_bits = uint(sign_table[(packed >> (7u * l)) & 127u]);
            uint xb = ib * 32u + l * 8u;
            for (uint j = 0; j < 4u; j++) {
                float sign1 = (sign_bits & (1u << j)) ? -1.0f : 1.0f;
                float sign2 = (sign_bits & (1u << (j + 4u))) ? -1.0f : 1.0f;
                sum += db * qw3_iq3xxs_grid_value(grid1, j) * sign1 * xx[xb + j];
                sum += db * qw3_iq3xxs_grid_value(grid2, j) * sign2 * xx[xb + j + 4u];
            }
        }
    }
    return sum;
}
kernel void qw3_matvec_iq3_xxs(constant qw3_matvec_q8_0_args &args,
                               device const uchar *weights,
                               device const float *x,
                               device float *out,
                               device const ushort *grid_table,
                               device const uchar *sign_table,
                               threadgroup float *sh,
                               uint row [[threadgroup_position_in_grid]],
                               ushort tid [[thread_index_in_threadgroup]],
                               ushort simd_idx [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]],
                               ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);
    float sum = 0.0f;
    for (uint b = tid; b < args.n_in / 256u; b += nt) {
        device const uchar *blk = wr + uint64_t(b) * 98ull;
        ushort dbits = ushort(blk[0]) | (ushort(blk[1]) << 8u);
        float d = qw3_f16_to_f32(dbits);
        device const uchar *qs = blk + 2u;
        device const uchar *ss = qs + 64u;
        device const float *xx = x + uint64_t(b) * 256ull;
        for (uint ib = 0; ib < 8u; ib++) {
            device const uchar *sp = ss + 4u * ib;
            uint packed = uint(sp[0]) | (uint(sp[1]) << 8u) |
                          (uint(sp[2]) << 16u) | (uint(sp[3]) << 24u);
            float db = d * (0.5f + float(packed >> 28u)) * 0.5f;
            for (uint l = 0; l < 4u; l++) {
                ushort grid1 = grid_table[uint(qs[8u * ib + 2u * l])];
                ushort grid2 = grid_table[uint(qs[8u * ib + 2u * l + 1u])];
                uint sign_bits = uint(sign_table[(packed >> (7u * l)) & 127u]);
                uint xb = ib * 32u + l * 8u;
                for (uint j = 0; j < 4u; j++) {
                    float sign1 = (sign_bits & (1u << j)) != 0u ? -1.0f : 1.0f;
                    float sign2 = (sign_bits & (1u << (j + 4u))) != 0u ? -1.0f : 1.0f;
                    sum += db * qw3_iq3xxs_grid_value(grid1, j) * sign1 * xx[xb + j];
                    sum += db * qw3_iq3xxs_grid_value(grid2, j) * sign2 * xx[xb + j + 4u];
                }
            }
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) sh[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane < ((uint(nt) + 31u) / 32u) ? sh[lane] : 0.0f;
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}
kernel void qw3_matvec_iq3_xxs_fast(
                               constant qw3_matvec_q8_0_args &args,
                               device const uchar *weights,
                               device const float *x,
                               device float *out,
                               device const ushort *grid_table,
                               device const uchar *sign_table,
                               threadgroup float *sh,
                               uint group [[threadgroup_position_in_grid]],
                               ushort simd_idx [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    uint row0 = (group * 2u + uint(simd_idx)) * 2u;
    uint row1 = row0 + 1u;
    float sum0 = 0.0f, sum1 = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = uint(lane); b < n_blocks; b += 32u) {
        device const float *xx = x + uint64_t(b) * 256ull;
        if (row0 < args.n_out)
            sum0 += qw3_iq3xxs_dot_block(
                weights + uint64_t(row0) * uint64_t(args.row_bytes) +
                uint64_t(b) * 98ull, xx, grid_table, sign_table);
        if (row1 < args.n_out)
            sum1 += qw3_iq3xxs_dot_block(
                weights + uint64_t(row1) * uint64_t(args.row_bytes) +
                uint64_t(b) * 98ull, xx, grid_table, sign_table);
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (lane == 0) {
        if (row0 < args.n_out) out[row0] = sum0;
        if (row1 < args.n_out) out[row1] = sum1;
    }
}
kernel void qw3_matmul_iq3_xxs_batch4(
    constant qw3_matmul_q8_0_batch_args &args,
    device const uchar *weights,
    device const float *x,
    device float *out,
    device const ushort *grid_table,
    device const uchar *sign_table,
    uint2 group [[threadgroup_position_in_grid]],
    ushort simd_idx [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x * 4u + uint(simd_idx);
    if (row >= args.n_out) return;
    const uint t0 = group.y * 4u;
    const uint t1 = min(t0 + 1u, args.n_tokens - 1u);
    const uint t2 = min(t0 + 2u, args.n_tokens - 1u);
    const uint t3 = min(t0 + 3u, args.n_tokens - 1u);
    device const float *x0 = x + uint64_t(t0) * args.in_stride + args.in_offset;
    device const float *x1 = x + uint64_t(t1) * args.in_stride + args.in_offset;
    device const float *x2 = x + uint64_t(t2) * args.in_stride + args.in_offset;
    device const float *x3 = x + uint64_t(t3) * args.in_stride + args.in_offset;
    device const uchar *wr = weights + uint64_t(row) * args.row_bytes;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (uint b = uint(lane); b < args.n_in / 256u; b += 32u) {
        device const uchar *blk = wr + uint64_t(b) * 98ull;
        const float d = qw3_f16_to_f32(
            ushort(blk[0]) | (ushort(blk[1]) << 8u));
        device const uchar *qs = blk + 2u;
        device const uchar *ss = qs + 64u;
        const uint base = b * 256u;
        for (uint ib = 0; ib < 8u; ib++) {
            device const uchar *sp = ss + 4u * ib;
            const uint packed = uint(sp[0]) | (uint(sp[1]) << 8u) |
                                (uint(sp[2]) << 16u) |
                                (uint(sp[3]) << 24u);
            const float db = d * (0.5f + float(packed >> 28u)) * 0.5f;
            for (uint l = 0; l < 4u; l++) {
                const ushort grid1 = grid_table[uint(qs[8u * ib + 2u * l])];
                const ushort grid2 = grid_table[uint(qs[8u * ib + 2u * l + 1u])];
                const uint sign_bits =
                    uint(sign_table[(packed >> (7u * l)) & 127u]);
                const uint xb = base + ib * 32u + l * 8u;
                for (uint j = 0; j < 4u; j++) {
                    const float sign1 = (sign_bits & (1u << j)) ? -1.0f : 1.0f;
                    const float sign2 = (sign_bits & (1u << (j + 4u))) ?
                        -1.0f : 1.0f;
                    const float w0 = db *
                        qw3_iq3xxs_grid_value(grid1, j) * sign1;
                    const float w1 = db *
                        qw3_iq3xxs_grid_value(grid2, j) * sign2;
                    s0 += w0 * x0[xb + j] + w1 * x0[xb + j + 4u];
                    s1 += w0 * x1[xb + j] + w1 * x1[xb + j + 4u];
                    s2 += w0 * x2[xb + j] + w1 * x2[xb + j + 4u];
                    s3 += w0 * x3[xb + j] + w1 * x3[xb + j + 4u];
                }
            }
        }
    }
    s0 = simd_sum(s0); s1 = simd_sum(s1);
    s2 = simd_sum(s2); s3 = simd_sum(s3);
    if (lane == 0) {
        out[uint64_t(t0) * args.out_stride + args.out_offset + row] = s0;
        if (t0 + 1u < args.n_tokens)
            out[uint64_t(t0 + 1u) * args.out_stride + args.out_offset + row] = s1;
        if (t0 + 2u < args.n_tokens)
            out[uint64_t(t0 + 2u) * args.out_stride + args.out_offset + row] = s2;
        if (t0 + 3u < args.n_tokens)
            out[uint64_t(t0 + 3u) * args.out_stride + args.out_offset + row] = s3;
    }
}
#ifdef QW3_METAL_HAS_TENSOR
kernel void qw3_matmul_iq3_xxs_mpp_direct(
    constant qw3_matmul_q8_0_batch_args &args,
    device const uchar *weights,
    device const float *x,
    device float *out,
    device const ushort *grid_table,
    device const uchar *sign_table,
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    const int r0 = int(group.y) * NR0;
    const int r1 = int(group.x) * NR1;
    if (r0 >= int(args.n_out) || r1 >= int(args.n_tokens)) return;
    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    device float *ptrB = (device float *)x + args.in_offset;
    auto tB = tensor(ptrB,
                    dextents<int32_t, 2>(int(args.n_in), int(args.n_tokens)),
                    array<int, 2>({1, int(args.in_stride)}));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<
        decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); i++)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    for (uint loop_k = 0; loop_k < args.n_in; loop_k += NK) {
        const int row = int(tid) / 2;
        const int half_k = int(tid) % 2;
        if (r0 + row < int(args.n_out)) {
            const uint k0 = loop_k + uint(half_k * 16);
            device const uchar *wrow = weights +
                uint64_t(r0 + row) * args.row_bytes;
            device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 98ull;
            const float d = qw3_f16_to_f32(
                ushort(blk[0]) | (ushort(blk[1]) << 8u));
            device const uchar *qs = blk + 2u;
            device const uchar *ss = qs + 64u;
            for (int i = 0; i < 16; i++) {
                const uint local = (k0 + uint(i)) & 255u;
                const uint ib = local >> 5u;
                const uint l = (local & 31u) >> 3u;
                const uint j = local & 7u;
                device const uchar *sp = ss + 4u * ib;
                const uint packed = uint(sp[0]) | (uint(sp[1]) << 8u) |
                    (uint(sp[2]) << 16u) | (uint(sp[3]) << 24u);
                const float db = d *
                    (0.5f + float(packed >> 28u)) * 0.5f;
                const uint sign_bits =
                    uint(sign_table[(packed >> (7u * l)) & 127u]);
                const uint grid_index =
                    uint(qs[8u * ib + 2u * l + (j >> 2u)]);
                const float sign = (sign_bits & (1u << j)) ? -1.0f : 1.0f;
                sa[row * NK + half_k * 16 + i] = half(
                    db * qw3_iq3xxs_grid_value(grid_table[grid_index],
                                                 j & 3u) * sign);
            }
        } else {
            for (int i = 0; i < 16; i++)
                sa[row * NK + half_k * 16 + i] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(int(loop_k), r1);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out), int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    cT.store(tD.slice(r0, r1));
}

kernel void qw3_matmul_q6_k_mpp_direct(
    constant qw3_matmul_q8_0_batch_args &args,
    device const uchar *weights,
    device const float *x,
    device float *out,
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    const int r0 = int(group.y) * NR0;
    const int r1 = int(group.x) * NR1;
    if (r0 >= int(args.n_out) || r1 >= int(args.n_tokens)) return;
    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    device float *ptrB = (device float *)x + args.in_offset;
    auto tB = tensor(ptrB,
                    dextents<int32_t, 2>(int(args.n_in), int(args.n_tokens)),
                    array<int, 2>({1, int(args.in_stride)}));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<
        decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); i++)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    for (uint loop_k = 0; loop_k < args.n_in; loop_k += NK) {
        const int row = int(tid) / 2;
        const int half_k = int(tid) % 2;
        if (r0 + row < int(args.n_out)) {
            const uint k0 = loop_k + uint(half_k * 16);
            device const uchar *wrow = weights +
                uint64_t(r0 + row) * args.row_bytes;
            device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 210ull;
            const float d = qw3_f16_to_f32(
                ushort(blk[208]) | (ushort(blk[209]) << 8u));
            for (int i = 0; i < 16; i++) {
                const uint local = (k0 + uint(i)) & 255u;
                const uint half_block = local >> 7u;
                const uint within = local & 127u;
                const uint part = within >> 5u;
                const uint l = within & 31u;
                const uint ql_index = half_block * 64u +
                    (part & 1u) * 32u + l;
                const uint packed = uint(blk[ql_index]);
                const uint low = part < 2u ? packed & 15u : packed >> 4u;
                const uint high = (uint(blk[128u + half_block * 32u + l]) >>
                                   (2u * part)) & 3u;
                const int q = int(low | (high << 4u)) - 32;
                const int scale = int(((device const char *)(blk + 192u))[
                    half_block * 8u + 2u * part + l / 16u]);
                sa[row * NK + half_k * 16 + i] =
                    half(d * float(scale) * float(q));
            }
        } else {
            for (int i = 0; i < 16; i++)
                sa[row * NK + half_k * 16 + i] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(int(loop_k), r1);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out), int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    cT.store(tD.slice(r0, r1));
}
#endif
kernel void qw3_matvec_q6_k_fast(constant qw3_matvec_q8_0_args &args,
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
    uint tid = uint(lane) >> 1u;
    uint ix = uint(lane) & 1u;
    uint ip = tid >> 3u;
    uint il = tid & 7u;
    uint l0 = 4u * il;
    uint is = 8u * ip + l0 / 16u;
    uint y_offset = 128u * ip + l0;
    uint q_offset_l = 64u * ip + l0;
    uint q_offset_h = 32u * ip + l0;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = ix; b < n_blocks; b += 2u) {
        device const float *yy = x + uint64_t(b) * 256ull + y_offset;
        uint row = first_row;
        if (row < args.n_out) {
            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 210ull;
            device const uchar *q1 = blk + q_offset_l;
            device const uchar *q2 = q1 + 32u;
            device const uchar *qh = blk + 128u + q_offset_h;
            device const char *sc = (device const char *)(blk + 192u + is);
            ushort dbits = ushort(blk[208u]) | (ushort(blk[209u]) << 8u);
            float d = qw3_f16_to_f32(dbits);
            float acc = 0.0f;
            for (uint l = 0; l < 4u; l++) {
                int qv1 = int((uint(q1[l]) & 15u) | (((uint(qh[l]) >> 0u) & 3u) << 4u)) - 32;
                int qv2 = int((uint(q2[l]) & 15u) | (((uint(qh[l]) >> 2u) & 3u) << 4u)) - 32;
                int qv3 = int((uint(q1[l]) >> 4u) | (((uint(qh[l]) >> 4u) & 3u) << 4u)) - 32;
                int qv4 = int((uint(q2[l]) >> 4u) | (((uint(qh[l]) >> 6u) & 3u) << 4u)) - 32;
                acc += float(sc[0]) * float(qv1) * yy[l + 0u];
                acc += float(sc[2]) * float(qv2) * yy[l + 32u];
                acc += float(sc[4]) * float(qv3) * yy[l + 64u];
                acc += float(sc[6]) * float(qv4) * yy[l + 96u];
            }
            sum0 += d * acc;
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 210ull;
            device const uchar *q1 = blk + q_offset_l;
            device const uchar *q2 = q1 + 32u;
            device const uchar *qh = blk + 128u + q_offset_h;
            device const char *sc = (device const char *)(blk + 192u + is);
            ushort dbits = ushort(blk[208u]) | (ushort(blk[209u]) << 8u);
            float d = qw3_f16_to_f32(dbits);
            float acc = 0.0f;
            for (uint l = 0; l < 4u; l++) {
                int qv1 = int((uint(q1[l]) & 15u) | (((uint(qh[l]) >> 0u) & 3u) << 4u)) - 32;
                int qv2 = int((uint(q2[l]) & 15u) | (((uint(qh[l]) >> 2u) & 3u) << 4u)) - 32;
                int qv3 = int((uint(q1[l]) >> 4u) | (((uint(qh[l]) >> 4u) & 3u) << 4u)) - 32;
                int qv4 = int((uint(q2[l]) >> 4u) | (((uint(qh[l]) >> 6u) & 3u) << 4u)) - 32;
                acc += float(sc[0]) * float(qv1) * yy[l + 0u];
                acc += float(sc[2]) * float(qv2) * yy[l + 32u];
                acc += float(sc[4]) * float(qv3) * yy[l + 64u];
                acc += float(sc[6]) * float(qv4) * yy[l + 96u];
            }
            sum1 += d * acc;
        }
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
kernel void qw3_matvec_q6_k_add_x0(constant qw3_matvec_q8_0_args &args,
                                   constant float &scale,
                                   device const uchar *weights,
                                   device const float *x,
                                   device float *x0,
                                   threadgroup float *sh,
                                   uint row [[threadgroup_position_in_grid]],
                                   ushort tid [[thread_index_in_threadgroup]],
                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]],
                                   ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);
    float sum = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = tid; b < n_blocks; b += nt) {
        device const uchar *blk = wr + uint64_t(b) * 210ull;
        device const uchar *ql = blk;
        device const uchar *qh = ql + 128u;
        device const uchar *scb = qh + 64u;
        device const char *sc = (device const char *)scb;
        ushort dbits = ushort(scb[16u]) | (ushort(scb[17u]) << 8u);
        float d = qw3_f16_to_f32(dbits);
        device const float *xx = x + uint64_t(b) * 256ull;
        for (uint n = 0; n < 256u; n += 128u) {
            for (uint l = 0; l < 32u; l++) {
                uint is = l / 16u;
                int q1 = int((uint(ql[l +  0u]) & 15u) | (((uint(qh[l]) >> 0u) & 3u) << 4u)) - 32;
                int q2 = int((uint(ql[l + 32u]) & 15u) | (((uint(qh[l]) >> 2u) & 3u) << 4u)) - 32;
                int q3 = int((uint(ql[l +  0u]) >> 4u) | (((uint(qh[l]) >> 4u) & 3u) << 4u)) - 32;
                int q4 = int((uint(ql[l + 32u]) >> 4u) | (((uint(qh[l]) >> 6u) & 3u) << 4u)) - 32;
                sum += d * float(sc[is + 0u]) * float(q1) * xx[n + l +  0u];
                sum += d * float(sc[is + 2u]) * float(q2) * xx[n + l + 32u];
                sum += d * float(sc[is + 4u]) * float(q3) * xx[n + l + 64u];
                sum += d * float(sc[is + 6u]) * float(q4) * xx[n + l + 96u];
            }
            ql += 64u;
            qh += 32u;
            sc += 8u;
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) sh[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane < ((uint(nt) + 31u) / 32u) ? sh[lane] : 0.0f;
    sum = simd_sum(sum);
    if (tid == 0) x0[row] = x0[row] + sum * scale;
}
kernel void qw3_matvec_q6_k_swiglu_add_x0(constant qw3_matvec_q8_0_args &args,
                                          constant float &scale,
                                          device const uchar *weights,
                                          device const float *scratch,
                                          device float *x0,
                                          threadgroup float *sh,
                                          uint row [[threadgroup_position_in_grid]],
                                          ushort tid [[thread_index_in_threadgroup]],
                                          ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                          ushort lane [[thread_index_in_simdgroup]],
                                          ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);
    float sum = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = tid; b < n_blocks; b += nt) {
        device const uchar *blk = wr + uint64_t(b) * 210ull;
        device const uchar *ql = blk; device const uchar *qh = ql + 128u; device const uchar *scb = qh + 64u; device const char *sc = (device const char *)scb;
        ushort dbits = ushort(scb[16u]) | (ushort(scb[17u]) << 8u); float d = qw3_f16_to_f32(dbits); uint xb = b * 256u;
        for (uint n = 0; n < 256u; n += 128u) {
            for (uint l = 0; l < 32u; l++) {
                uint is = l / 16u;
                int q1 = int((uint(ql[l +  0u]) & 15u) | (((uint(qh[l]) >> 0u) & 3u) << 4u)) - 32;
                int q2 = int((uint(ql[l + 32u]) & 15u) | (((uint(qh[l]) >> 2u) & 3u) << 4u)) - 32;
                int q3 = int((uint(ql[l +  0u]) >> 4u) | (((uint(qh[l]) >> 4u) & 3u) << 4u)) - 32;
                int q4 = int((uint(ql[l + 32u]) >> 4u) | (((uint(qh[l]) >> 6u) & 3u) << 4u)) - 32;
                sum += d * float(sc[is + 0u]) * float(q1) * qw3_swiglu_val(scratch, args.n_in, xb + n + l +  0u);
                sum += d * float(sc[is + 2u]) * float(q2) * qw3_swiglu_val(scratch, args.n_in, xb + n + l + 32u);
                sum += d * float(sc[is + 4u]) * float(q3) * qw3_swiglu_val(scratch, args.n_in, xb + n + l + 64u);
                sum += d * float(sc[is + 6u]) * float(q4) * qw3_swiglu_val(scratch, args.n_in, xb + n + l + 96u);
            }
            ql += 64u; qh += 32u; sc += 8u;
        }
    }
    sum = simd_sum(sum); if (lane == 0) sh[simd_idx] = sum; threadgroup_barrier(mem_flags::mem_threadgroup); sum = lane < ((uint(nt) + 31u) / 32u) ? sh[lane] : 0.0f; sum = simd_sum(sum); if (tid == 0) x0[row] = x0[row] + sum * scale;
}
inline float qw3_iq3s_grid_val(device const ushort *kgrid, uint idx, uint j) {
    ushort packed = kgrid[idx & 511u];
    return float(2u * ((uint(packed) >> (3u * j)) & 7u) + 1u);
}
kernel void qw3_matvec_iq3_s(constant qw3_matvec_q8_0_args &args,
                             device const uchar *weights,
                             device const float *x,
                             device float *out,
                             device const ushort *kgrid,
                             threadgroup float *sh,
                             uint row [[threadgroup_position_in_grid]],
                             ushort tid [[thread_index_in_threadgroup]],
                             ushort simd_idx [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]],
                             ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);
    float sum = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = tid; b < n_blocks; b += nt) {
        device const uchar *blk = wr + uint64_t(b) * 110ull;
        half d = *((device const half *)blk);
        device const uchar *qs = blk + 2;
        device const uchar *qh = qs + 64;
        device const uchar *signs = qh + 8;
        device const uchar *scales = signs + 32;
        device const float *xx = x + uint64_t(b) * 256ull;
        uint xo = 0;
        for (uint ib32 = 0; ib32 < 8u; ib32 += 2u) {
            float db1 = float(d) * float(1u + 2u * (uint(scales[ib32 / 2u]) & 15u));
            float db2 = float(d) * float(1u + 2u * (uint(scales[ib32 / 2u]) >> 4u));
            uchar qh0 = qh[0];
            uchar qh1 = qh[1];
            for (uint l = 0; l < 4u; l++) {
                uint idx1 = uint(qs[2u * l + 0u]) | ((uint(qh0) << (8u - 2u * l)) & 256u);
                uint idx2 = uint(qs[2u * l + 1u]) | ((uint(qh0) << (7u - 2u * l)) & 256u);
                uchar s = signs[l];
                for (uint j = 0; j < 4u; j++) {
                    float sign1 = (uint(s) & (1u << j)) ? -1.0f : 1.0f;
                    float sign2 = (uint(s) & (1u << (j + 4u))) ? -1.0f : 1.0f;
                    sum += db1 * qw3_iq3s_grid_val(kgrid, idx1, j) * sign1 * xx[xo + j + 0u];
                    sum += db1 * qw3_iq3s_grid_val(kgrid, idx2, j) * sign2 * xx[xo + j + 4u];
                }
                xo += 8u;
            }
            qs += 8;
            signs += 4;
            for (uint l = 0; l < 4u; l++) {
                uint idx1 = uint(qs[2u * l + 0u]) | ((uint(qh1) << (8u - 2u * l)) & 256u);
                uint idx2 = uint(qs[2u * l + 1u]) | ((uint(qh1) << (7u - 2u * l)) & 256u);
                uchar s = signs[l];
                for (uint j = 0; j < 4u; j++) {
                    float sign1 = (uint(s) & (1u << j)) ? -1.0f : 1.0f;
                    float sign2 = (uint(s) & (1u << (j + 4u))) ? -1.0f : 1.0f;
                    sum += db2 * qw3_iq3s_grid_val(kgrid, idx1, j) * sign1 * xx[xo + j + 0u];
                    sum += db2 * qw3_iq3s_grid_val(kgrid, idx2, j) * sign2 * xx[xo + j + 4u];
                }
                xo += 8u;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) sh[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane < ((uint(nt) + 31u) / 32u) ? sh[lane] : 0.0f;
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}
struct qw3_matvec_f32_args { uint n_in; uint n_out; };
kernel void qw3_matvec_f32(constant qw3_matvec_f32_args &args,
                           device const float *weights,
                           device const float *x,
                           device float *out,
                           threadgroup float *sh,
                           uint row [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]],
                           ushort simd_idx [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    device const float *wr = weights + uint64_t(row) * args.n_in;
    float sum = 0.0f;
    for (uint i = tid; i < args.n_in; i += nt) sum += wr[i] * x[i];
    sum = simd_sum(sum);
    if (lane == 0) sh[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane < ((uint(nt) + 31u) / 32u) ? sh[lane] : 0.0f;
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}
struct qw3_matvec_f32_pair_args { uint n_in; uint n_out; uint out_a_offset; uint out_b_offset; };
kernel void qw3_matvec_f32_pair(constant qw3_matvec_f32_pair_args &args,
                                device const float *weights_a,
                                device const float *weights_b,
                                device const float *x,
                                device float *out,
                                threadgroup float *sh,
                                uint row [[threadgroup_position_in_grid]],
                                ushort tid [[thread_index_in_threadgroup]],
                                ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]],
                                ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    device const float *wa = weights_a + uint64_t(row) * args.n_in;
    device const float *wb = weights_b + uint64_t(row) * args.n_in;
    float suma = 0.0f;
    float sumb = 0.0f;
    for (uint i = tid; i < args.n_in; i += nt) { float xv = x[i]; suma += wa[i] * xv; sumb += wb[i] * xv; }
    suma = simd_sum(suma);
    sumb = simd_sum(sumb);
    if (lane == 0) { sh[simd_idx] = suma; sh[simd_idx + 32] = sumb; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    suma = lane < 32 ? sh[lane] : 0.0f;
    sumb = lane < 32 ? sh[lane + 32] : 0.0f;
    suma = simd_sum(suma);
    sumb = simd_sum(sumb);
    if (tid == 0) { out[args.out_a_offset + row] = suma; out[args.out_b_offset + row] = sumb; }
}
struct qw3_matmul_f32_batch_args { uint n_in; uint n_out; uint n_tokens; uint in_offset; uint in_stride; uint out_offset; uint out_stride; };
kernel void qw3_matmul_f32_batch4(constant qw3_matmul_f32_batch_args &args,
                                  device const float *weights,
                                  device const float *x,
                                  device float *out,
                                  uint2 group [[threadgroup_position_in_grid]],
                                  ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                  ushort lane [[thread_index_in_simdgroup]]) {
    uint row = group.x * 4u + uint(simd_idx);
    if (row >= args.n_out) return;
    uint t0 = group.y * 4u;
    device const float *wr = weights + uint64_t(row) * args.n_in;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (uint i = uint(lane); i < args.n_in; i += 32u) {
        float wv = wr[i];
        if (t0 + 0u < args.n_tokens) s0 += wv * x[uint64_t(t0 + 0u) * args.in_stride + args.in_offset + i];
        if (t0 + 1u < args.n_tokens) s1 += wv * x[uint64_t(t0 + 1u) * args.in_stride + args.in_offset + i];
        if (t0 + 2u < args.n_tokens) s2 += wv * x[uint64_t(t0 + 2u) * args.in_stride + args.in_offset + i];
        if (t0 + 3u < args.n_tokens) s3 += wv * x[uint64_t(t0 + 3u) * args.in_stride + args.in_offset + i];
    }
    s0 = simd_sum(s0); s1 = simd_sum(s1); s2 = simd_sum(s2); s3 = simd_sum(s3);
    if (lane == 0) {
        if (t0 + 0u < args.n_tokens) out[uint64_t(t0 + 0u) * args.out_stride + args.out_offset + row] = s0;
        if (t0 + 1u < args.n_tokens) out[uint64_t(t0 + 1u) * args.out_stride + args.out_offset + row] = s1;
        if (t0 + 2u < args.n_tokens) out[uint64_t(t0 + 2u) * args.out_stride + args.out_offset + row] = s2;
        if (t0 + 3u < args.n_tokens) out[uint64_t(t0 + 3u) * args.out_stride + args.out_offset + row] = s3;
    }
}
struct qw3_matmul_f32_pair_batch_args { uint n_in; uint n_out; uint n_tokens; uint out_a_offset; uint out_b_offset; uint out_stride; };
kernel void qw3_matmul_f32_pair_batch4(constant qw3_matmul_f32_pair_batch_args &args,
                                       device const float *weights_a,
                                       device const float *weights_b,
                                       device const float *x,
                                       device float *out,
                                       uint2 group [[threadgroup_position_in_grid]],
                                       ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                       ushort lane [[thread_index_in_simdgroup]]) {
    uint row = group.x * 4u + uint(simd_idx);
    if (row >= args.n_out) return;
    uint t0 = group.y * 4u;
    device const float *wa = weights_a + uint64_t(row) * args.n_in;
    device const float *wb = weights_b + uint64_t(row) * args.n_in;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
    for (uint i = uint(lane); i < args.n_in; i += 32u) {
        float wva = wa[i];
        float wvb = wb[i];
        if (t0 + 0u < args.n_tokens) { float xv = x[uint64_t(t0 + 0u) * args.n_in + i]; a0 += wva * xv; b0 += wvb * xv; }
        if (t0 + 1u < args.n_tokens) { float xv = x[uint64_t(t0 + 1u) * args.n_in + i]; a1 += wva * xv; b1 += wvb * xv; }
        if (t0 + 2u < args.n_tokens) { float xv = x[uint64_t(t0 + 2u) * args.n_in + i]; a2 += wva * xv; b2 += wvb * xv; }
        if (t0 + 3u < args.n_tokens) { float xv = x[uint64_t(t0 + 3u) * args.n_in + i]; a3 += wva * xv; b3 += wvb * xv; }
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);
    b0 = simd_sum(b0); b1 = simd_sum(b1); b2 = simd_sum(b2); b3 = simd_sum(b3);
    if (lane == 0) {
        uint64_t base = uint64_t(t0 + 0u) * args.out_stride;
        if (t0 + 0u < args.n_tokens) { out[base + args.out_a_offset + row] = a0; out[base + args.out_b_offset + row] = b0; }
        base = uint64_t(t0 + 1u) * args.out_stride;
        if (t0 + 1u < args.n_tokens) { out[base + args.out_a_offset + row] = a1; out[base + args.out_b_offset + row] = b1; }
        base = uint64_t(t0 + 2u) * args.out_stride;
        if (t0 + 2u < args.n_tokens) { out[base + args.out_a_offset + row] = a2; out[base + args.out_b_offset + row] = b2; }
        base = uint64_t(t0 + 3u) * args.out_stride;
        if (t0 + 3u < args.n_tokens) { out[base + args.out_a_offset + row] = a3; out[base + args.out_b_offset + row] = b3; }
    }
}
kernel void qw3_matvec_f32_fast(constant qw3_matvec_f32_args &args,
                               device const float *weights,
                               device const float *x,
                               device float *out,
                               threadgroup float *sh,
                               uint group [[threadgroup_position_in_grid]],
                               ushort simd_idx [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    const uint nr0 = 2u;
    const uint nsg = 4u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (uint i = uint(lane); i < args.n_in; i += 32u) {
        float xv = x[i];
        uint row = first_row;
        if (row < args.n_out) sum0 += weights[uint64_t(row) * args.n_in + i] * xv;
        row = first_row + 1u;
        if (row < args.n_out) sum1 += weights[uint64_t(row) * args.n_in + i] * xv;
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
