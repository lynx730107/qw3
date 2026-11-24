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
    constexpr int STAGE_K = 64;
    const int K = int(args.n_in);
    const int M = int(args.n_out);
    const int N = int(args.n_tokens);
    const int r0 = int(group.y) * NR0;
    const int r1 = int(group.x) * NR1;
    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(STAGE_K, NR0));
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
    const int row = int(tid) >> 1;
    const int blk_half = int(tid) & 1;
    const bool valid_row = (r0 + row < M);
    device const qw3_block_q8_0 *row_ptr =
        (device const qw3_block_q8_0 *)(weights + uint64_t(valid_row ? (r0 + row) : 0) * uint64_t(args.row_bytes));
    threadgroup half4 *sa4 = (threadgroup half4 *)(sa + row * STAGE_K + blk_half * 32);
    for (int stage_k = 0; stage_k < K; stage_k += STAGE_K) {
        if (valid_row) {
            device const qw3_block_q8_0 *blk = row_ptr + (stage_k >> 5) + blk_half;
            const half dh = blk->d;
            device const packed_char4 *qp = (device const packed_char4 *)blk->qs;
            sa4[0] = half4(char4(qp[0])) * dh;
            sa4[1] = half4(char4(qp[1])) * dh;
            sa4[2] = half4(char4(qp[2])) * dh;
            sa4[3] = half4(char4(qp[3])) * dh;
            sa4[4] = half4(char4(qp[4])) * dh;
            sa4[5] = half4(char4(qp[5])) * dh;
            sa4[6] = half4(char4(qp[6])) * dh;
            sa4[7] = half4(char4(qp[7])) * dh;
        } else {
            for (int i = 0; i < 8; i++) sa4[i] = half4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            auto mA0 = tA.slice(0, 0);
            auto mB0 = tB.slice(stage_k, r1);
            mm.run(mB0, mA0, cT);
            auto mA1 = tA.slice(32, 0);
            auto mB1 = tB.slice(stage_k + 32, r1);
            mm.run(mB1, mA1, cT);
        }
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
/* Load eight packed quants as two words; reuse the four activation vectors
 * across output rows and the gate/up pair. Magnitudes remain exact integers. */
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
kernel void qw3_matvec_iq4_xs_fast(constant qw3_matvec_q8_0_args &args,
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
    if (simd_idx == 0u && lane < 16u) sh[lane] = qw3_iq4nl_val(uint(lane));
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
kernel void qw3_matvec_iq4_xs_pair_swiglu_fast(
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
        if (row < args.n_out)
            out[row] = (gate0 / (1.0f + exp(-gate0))) * up0;
        row = first_row + 1u;
        if (row < args.n_out)
            out[row] = (gate1 / (1.0f + exp(-gate1))) * up1;
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
    if (simd_idx == 0u && lane < 16u) sh[lane] = qw3_iq4nl_val(uint(lane));
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
kernel void qw3_matvec_q5_k_fast(constant qw3_matvec_q8_0_args &args,
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
        float4 acc1 = float4(0.0f), acc2 = float4(0.0f);
#pragma unroll
        for (uint l = 0; l < 8u; ++l) {
            const uchar h = qh[l];
            acc1[0] += yl[l]      * float(q1[l] & 0x0fu);
            acc1[1] += yl[l + 8u] * float(q1[l] & 0xf0u);
            acc1[2] += yh[l]      * float(q2[l] & 0x0fu);
            acc1[3] += yh[l + 8u] * float(q2[l] & 0xf0u);
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

kernel void qw3_matvec_q4_k_words(constant qw3_matvec_q8_0_args &args,
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
            for (uint l = 0; l < 32u; l += 4u) {
                uint word = uint(*((device const uint *)(qs + qbase + l)));
                for (uint j = 0; j < 4u; j++) {
                    uint packed = (word >> (8u * j)) & 255u;
                    uint q = (g & 1u) == 0u ? (packed & 15u) : (packed >> 4u);
                    sum += (dl * float(q) - ml) * xx[xbase + l + j];
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
inline float qw3_q4_k_dot32_pair(device const uchar *blk,
                                 thread const float4 *xlo,
                                 thread const float4 *xhi,
                                 float sum_xlo, float sum_xhi,
                                 uint c, uint h) {
    const float d = float(*((device const half *)blk));
    const float dmin = float(*((device const half *)(blk + 2u)));
    device const uchar *scales = blk + 4u;
    const uint g0 = c * 2u;
    const uint g1 = g0 + 1u;
    uint sc0, mn0, sc1, mn1;
    if (c < 2u) {
        sc0 = uint(scales[g0]) & 63u;
        mn0 = uint(scales[g0 + 4u]) & 63u;
        sc1 = uint(scales[g1]) & 63u;
        mn1 = uint(scales[g1 + 4u]) & 63u;
    } else {
        sc0 = (uint(scales[g0 + 4u]) & 15u) | ((uint(scales[g0 - 4u]) >> 6u) << 4u);
        mn0 = (uint(scales[g0 + 4u]) >> 4u) | ((uint(scales[g0]) >> 6u) << 4u);
        sc1 = (uint(scales[g1 + 4u]) & 15u) | ((uint(scales[g1 - 4u]) >> 6u) << 4u);
        mn1 = (uint(scales[g1 + 4u]) >> 4u) | ((uint(scales[g1]) >> 6u) << 4u);
    }
    device const uint *qptr = (device const uint *)(blk + 16u + c * 32u + h * 16u);
    float dot_lo = 0.0f;
    float dot_hi = 0.0f;
    for (uint k = 0u; k < 4u; k++) {
        const uint w = qptr[k];
        const float4 qlo = float4(float(w & 15u), float((w >> 8u) & 15u),
                                  float((w >> 16u) & 15u), float((w >> 24u) & 15u));
        const float4 qhi = float4(float((w >> 4u) & 15u), float((w >> 12u) & 15u),
                                  float((w >> 20u) & 15u), float((w >> 28u) & 15u));
        dot_lo += dot(qlo, xlo[k]);
        dot_hi += dot(qhi, xhi[k]);
    }
    return (d * float(sc0)) * dot_lo - (dmin * float(mn0)) * sum_xlo +
           (d * float(sc1)) * dot_hi - (dmin * float(mn1)) * sum_xhi;
}
kernel void qw3_matvec_q4_k_fast(constant qw3_matvec_q8_0_args &args,
                                 device const uchar *weights,
                                 device const float *x,
                                 device float *out,
                                 threadgroup float *sh,
                                 uint group [[threadgroup_position_in_grid]],
                                 ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    const uint nr0 = 4u;
    const uint nsg = 2u;
    const uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    const uint nb32 = (args.n_in / 256u) * 8u;
    for (uint ib32 = uint(lane); ib32 < nb32; ib32 += 32u) {
        const uint b = ib32 >> 3u;
        const uint c = (ib32 >> 1u) & 3u;
        const uint h = ib32 & 1u;
        const uint64_t xbase = uint64_t(b) * 256ull + uint64_t(c) * 64ull + uint64_t(h) * 16ull;
        device const float4 *x4_lo = (device const float4 *)(x + xbase);
        device const float4 *x4_hi = (device const float4 *)(x + xbase + 32ull);
        float4 xlo[4] = { x4_lo[0], x4_lo[1], x4_lo[2], x4_lo[3] };
        float4 xhi[4] = { x4_hi[0], x4_hi[1], x4_hi[2], x4_hi[3] };
        const float4 sl = (xlo[0] + xlo[1]) + (xlo[2] + xlo[3]);
        const float4 shv = (xhi[0] + xhi[1]) + (xhi[2] + xhi[3]);
        const float sum_xlo = (sl.x + sl.y) + (sl.z + sl.w);
        const float sum_xhi = (shv.x + shv.y) + (shv.z + shv.w);
        const uint64_t blk_off = uint64_t(b) * 144ull;
        uint row = first_row;
        if (row < args.n_out)
            sum0 += qw3_q4_k_dot32_pair(weights + uint64_t(row) * uint64_t(args.row_bytes) + blk_off,
                                        xlo, xhi, sum_xlo, sum_xhi, c, h);
        row = first_row + 1u;
        if (row < args.n_out)
            sum1 += qw3_q4_k_dot32_pair(weights + uint64_t(row) * uint64_t(args.row_bytes) + blk_off,
                                        xlo, xhi, sum_xlo, sum_xhi, c, h);
        row = first_row + 2u;
        if (row < args.n_out)
            sum2 += qw3_q4_k_dot32_pair(weights + uint64_t(row) * uint64_t(args.row_bytes) + blk_off,
                                        xlo, xhi, sum_xlo, sum_xhi, c, h);
        row = first_row + 3u;
        if (row < args.n_out)
            sum3 += qw3_q4_k_dot32_pair(weights + uint64_t(row) * uint64_t(args.row_bytes) + blk_off,
                                        xlo, xhi, sum_xlo, sum_xhi, c, h);
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2);
    sum3 = simd_sum(sum3);
    if (lane == 0u) {
        uint row = first_row;
        if (row < args.n_out) out[row] = sum0;
        row = first_row + 1u;
        if (row < args.n_out) out[row] = sum1;
        row = first_row + 2u;
        if (row < args.n_out) out[row] = sum2;
        row = first_row + 3u;
        if (row < args.n_out) out[row] = sum3;
    }
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
/* Decode four output rows per 64-thread group. This follows the lane mapping
 * used by llama.cpp's Q3_K matvec while retaining our row-major GGUF input. */
kernel void qw3_matvec_q3_k_fast(constant qw3_matvec_q8_0_args &args,
                                 device const uchar *weights,
                                 device const float *x,
                                 device float *out,
                                 threadgroup float *sh,
                                 uint group [[threadgroup_position_in_grid]],
                                 ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    const uint first_row = (group * 2u + uint(simd_idx)) * 2u;
    const ushort tid = lane / 4u;
    const ushort ix = lane & 3u;
    const ushort ip = tid / 4u;
    const ushort il = 2u * ((tid & 3u) / 2u);
    const ushort ir = tid & 1u;
    const ushort l0 = 8u * ir;
    const ushort4 mm[4] = {
        ushort4(0x0001, 0x0100, 0x0002, 0x0200),
        ushort4(0x0004, 0x0400, 0x0008, 0x0800),
        ushort4(0x0010, 0x1000, 0x0020, 0x2000),
        ushort4(0x0040, 0x4000, 0x0080, 0x8000)
    };
    const int4 qm[2] = {
        int4(0x0003, 0x0300, 0x000c, 0x0c00),
        int4(0x0030, 0x3000, 0x00c0, 0xc000)
    };
    const ushort4 hm = mm[2u * ip + il / 2u];
    const ushort shift = 2u * il;
    const float v1 = il == 0u ? 4.0f : 64.0f;
    const float v2 = 4.0f * v1;
    const ushort s_shift1 = 4u * ip;
    const ushort s_shift2 = s_shift1 + il;
    const ushort q_offset = 32u * ip + l0;
    const ushort y_offset = 128u * ip + 32u * il + l0;
    const uint nb = args.n_in / 256u;
    device const float *y1 = x + uint(ix) * 256u + y_offset;
    float sumf1[2] = {0.0f, 0.0f};
    float sumf2[2] = {0.0f, 0.0f};

    for (uint ib = uint(ix); ib < nb; ib += 4u) {
        float yl[32];
#pragma unroll
        for (ushort l = 0u; l < 8u; ++l) {
            yl[l] = y1[l];
            yl[l + 8u] = y1[l + 16u];
            yl[l + 16u] = y1[l + 32u];
            yl[l + 24u] = y1[l + 48u];
        }
#pragma unroll
        for (ushort row_off = 0u; row_off < 2u; ++row_off) {
            const uint row = first_row + uint(row_off);
            if (row >= args.n_out) continue;
            device const uchar *blk = weights +
                uint64_t(row) * uint64_t(args.row_bytes) +
                uint64_t(ib) * 110ull;
            device const ushort *q =
                (device const ushort *)(blk + 32u + q_offset);
            device const ushort *h =
                (device const ushort *)(blk + l0);
            device const ushort *a =
                (device const ushort *)(blk + 96u);
            uint scales32 = uint(a[4]) | (uint(a[5]) << 16u);
            const uint aux32 = ((scales32 >> s_shift2) << 4u) &
                               0x30303030u;
            scales32 = uint(a[il]) | (uint(a[il + 1u]) << 16u);
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0fu) | aux32;
            thread const char *scales = (thread const char *)&scales32;
            float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f;
#pragma unroll
            for (ushort l = 0u; l < 8u; l += 2u) {
                const int qs = int(q[l / 2u]);
                s1 += yl[l] * float(qs & qm[il / 2u][0]);
                s2 += yl[l + 1u] * float(qs & qm[il / 2u][1]);
                s3 += (h[l / 2u] & hm[0]) ? 0.0f : yl[l];
                s3 += (h[l / 2u] & hm[1]) ? 0.0f : yl[l + 1u];
                s4 += yl[l + 16u] * float(qs & qm[il / 2u][2]);
                s5 += yl[l + 17u] * float(qs & qm[il / 2u][3]);
                s6 += (h[l / 2u] & hm[2]) ? 0.0f : yl[l + 16u];
                s6 += (h[l / 2u] & hm[3]) ? 0.0f : yl[l + 17u];
            }
            const float d = float(*((device const half *)(blk + 108u)));
            float d1 = d * (s1 + s2 / 256.0f - s3 * v1);
            float d2 = d * (s4 + s5 / 256.0f - s6 * v2);
            sumf1[row_off] += d1 * float(scales[0] - 32);
            sumf2[row_off] += d2 * float(scales[2] - 32);

            s1 = s2 = s3 = s4 = s5 = s6 = 0.0f;
#pragma unroll
            for (ushort l = 0u; l < 8u; l += 2u) {
                const int qs = int(q[l / 2u + 8u]);
                s1 += yl[l + 8u] * float(qs & qm[il / 2u][0]);
                s2 += yl[l + 9u] * float(qs & qm[il / 2u][1]);
                s3 += (h[l / 2u + 8u] & hm[0]) ? 0.0f : yl[l + 8u];
                s3 += (h[l / 2u + 8u] & hm[1]) ? 0.0f : yl[l + 9u];
                s4 += yl[l + 24u] * float(qs & qm[il / 2u][2]);
                s5 += yl[l + 25u] * float(qs & qm[il / 2u][3]);
                s6 += (h[l / 2u + 8u] & hm[2]) ? 0.0f : yl[l + 24u];
                s6 += (h[l / 2u + 8u] & hm[3]) ? 0.0f : yl[l + 25u];
            }
            d1 = d * (s1 + s2 / 256.0f - s3 * v1);
            d2 = d * (s4 + s5 / 256.0f - s6 * v2);
            sumf1[row_off] += d1 * float(scales[1] - 32);
            sumf2[row_off] += d2 * float(scales[3] - 32);
        }
        y1 += 4u * 256u;
    }
#pragma unroll
    for (ushort row_off = 0u; row_off < 2u; ++row_off) {
        const float sum = (sumf1[row_off] + 0.25f * sumf2[row_off]) /
                          float(1u << shift);
        sumf1[row_off] = simd_sum(sum);
    }
    if (lane == 0u) {
#pragma unroll
        for (ushort row_off = 0u; row_off < 2u; ++row_off) {
            const uint row = first_row + uint(row_off);
            if (row < args.n_out) out[row] = sumf1[row_off];
        }
    }
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
#ifdef QW3_ENABLE_PARTIAL_TILES
#define QW3_PARTIAL_LINEAR_TILE(row, count) ((row) < int(count))
#else
#define QW3_PARTIAL_LINEAR_TILE(row, count) true
#endif
kernel void qw3_matmul_iq4_nl_mpp_direct(
    constant qw3_matmul_q8_0_batch_args &args,
    device const uchar *weights,
    device const float *x,
    device float *out,
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int NR1 = 64;
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
    constexpr int NR1 = 64;
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

kernel void qw3_matmul_q2_k_mpp_staged128(
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
    const int token0 = r1 + int(simd_group) * ROWS_PER_SG;
    const bool token_group_active = QW3_PARTIAL_LINEAR_TILE(token0, args.n_tokens);
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
    for (uint16_t i = 0; i < cT.get_capacity(); i++)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    const int row = int(tid) / 2;
    const int half_k = int(tid) & 1;
    const bool valid_row = r0 + row < int(args.n_out);
    device const uchar *wrow = weights + uint64_t(r0 + row) * args.row_bytes;
    for (uint stage_k = 0u; stage_k < args.n_in; stage_k += STAGE_K) {
        if (valid_row) {
            device const uchar *blk = wrow + uint64_t(stage_k >> 8u) * 84ull;
            device const uchar *scales = blk;
            device const uchar *qs = blk + 16u;
            const float d = qw3_f16_to_f32(
                ushort(blk[80]) | (ushort(blk[81]) << 8u));
            const float dmin = qw3_f16_to_f32(
                ushort(blk[82]) | (ushort(blk[83]) << 8u));
            for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
                const uint k0 = stage_k + sub_k + uint(half_k * 16);
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
                    sa[row * STAGE_K + int(sub_k) + half_k * 16 + i] =
                        half(d * float(sc & 15u) * float(q) -
                             dmin * float(sc >> 4u));
                }
            }
        } else {
            for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK)
                for (int i = 0; i < 16; i++)
                    sa[row * STAGE_K + int(sub_k) + half_k * 16 + i] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
            auto mA = tA.slice(int(sub_k), 0);
            if (token_group_active) {
                auto mB = tB.slice(int(stage_k + sub_k), token0);
                mm.run(mB, mA, cT);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out), int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    if (token_group_active) cT.store(tD.slice(r0, token0));
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
    constexpr int NR1 = 64;
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

/* Stage each 64x32 weight tile once, then let four independent simdgroups
 * multiply it by 32 input rows each.  This mirrors the high-reuse prefill
 * shape used by Splash while retaining the native GGUF Q3_K layout. */
kernel void qw3_matmul_q3_k_mpp_staged128(
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
    const int token0 = r1 + int(simd_group) * ROWS_PER_SG;
    const bool token_group_active = QW3_PARTIAL_LINEAR_TILE(token0, args.n_tokens);
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
    for (uint16_t i = 0; i < cT.get_capacity(); i++)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0f0f0f0fu;
    const int row = int(tid) / 2;
    const int half_k = int(tid) & 1;
    const bool valid_row = r0 + row < int(args.n_out);
    device const uchar *wrow = weights + uint64_t(r0 + row) * args.row_bytes;
    for (uint stage_k = 0u; stage_k < args.n_in; stage_k += STAGE_K) {
        if (valid_row) {
            device const uchar *blk = wrow + uint64_t(stage_k >> 8u) * 110ull;
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
            for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
                const uint k0 = stage_k + sub_k + uint(half_k * 16);
                const uint local0 = k0 & 255u;
                const uint half_block = local0 >> 7u;
                const uint within = local0 & 127u;
                const uint grp = within >> 5u;
                const uint upper = (within & 31u) >= 16u ? 1u : 0u;
                const uint scale_index = half_block * 8u + grp * 2u + upper;
                const uint scale = (aux[scale_index >> 2u] >>
                                    (8u * (scale_index & 3u))) & 255u;
                const float ds = d * float(int(scale) - 32);
                const uint shift = grp * 2u;
                const uint mask = 1u << (half_block * 4u + grp);
                const uint q_base = half_block * 32u + upper * 16u;
                const uint h_base = upper * 16u;
                for (int i = 0; i < 16; i++) {
                    const uint l = uint(i);
                    const int low = int((uint(blk[32u + q_base + l]) >> shift) & 3u);
                    const int q = low - ((uint(blk[h_base + l]) & mask) ? 0 : 4);
                    sa[row * STAGE_K + int(sub_k) + half_k * 16 + i] =
                        half(ds * float(q));
                }
            }
        } else {
            for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK)
                for (int i = 0; i < 16; i++)
                    sa[row * STAGE_K + int(sub_k) + half_k * 16 + i] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
            auto mA = tA.slice(int(sub_k), 0);
            if (token_group_active) {
                auto mB = tB.slice(int(stage_k + sub_k), token0);
                mm.run(mB, mA, cT);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out), int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    if (token_group_active) cT.store(tD.slice(r0, token0));
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
    constexpr int NR1 = 64;
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
    const bool is_q5 = args.is_q5 != 0u;
    const uint64_t block_bytes = is_q5 ? 176ull : 144ull;
    const uint ql_offset = is_q5 ? 48u : 16u;
    for (uint loop_k = 0; loop_k < args.n_in; loop_k += NK) {
        const int row = int(tid) / 2;
        const int half_k = int(tid) % 2;
        if (r0 + row < int(args.n_out)) {
            const uint k0 = loop_k + uint(half_k * 16);
            const uint g = (loop_k & 255u) >> 5u;
            const uint l0 = uint(half_k * 16);
            device const uchar *wrow = weights +
                uint64_t(r0 + row) * args.row_bytes;
            device const uchar *blk = wrow +
                uint64_t(k0 >> 8u) * block_bytes;
            const float d = qw3_f16_to_f32(
                ushort(blk[0]) | (ushort(blk[1]) << 8u));
            const float dmin = qw3_f16_to_f32(
                ushort(blk[2]) | (ushort(blk[3]) << 8u));
            device const uchar *scales = blk + 4u;
            device const uchar *qh = blk + 16u;
            device const uchar *ql = blk + ql_offset;
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
            const float dl = d * float(sc);
            const float ml = dmin * float(mn);
            device const uchar *qgroup = ql + (g >> 1u) * 32u + l0;
            const uint high_mask = 1u << g;
            const bool high_nibble = (g & 1u) != 0u;
            for (int i = 0; i < 16; i++) {
                const uint l = l0 + uint(i);
                const uint packed = uint(qgroup[i]);
                uint q = high_nibble ? packed >> 4u : packed & 15u;
                if (is_q5 && (uint(qh[l]) & high_mask)) q += 16u;
                sa[row * NK + half_k * 16 + i] =
                    half(dl * float(q) - ml);
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

kernel void qw3_matmul_q4_q5_k_mpp_staged128(
    constant qw3_matmul_k_batch_args &args,
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
    const int token0 = r1 + int(simd_group) * ROWS_PER_SG;
    const bool token_group_active = QW3_PARTIAL_LINEAR_TILE(token0, args.n_tokens);
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
    for (uint16_t i = 0; i < cT.get_capacity(); i++)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    const bool is_q5 = args.is_q5 != 0u;
    const uint64_t block_bytes = is_q5 ? 176ull : 144ull;
    const uint ql_offset = is_q5 ? 48u : 16u;
    const int row = int(tid) / 2;
    const int half_k = int(tid) & 1;
    const bool valid_row = r0 + row < int(args.n_out);
    device const uchar *wrow = weights + uint64_t(r0 + row) * args.row_bytes;
    for (uint stage_k = 0u; stage_k < args.n_in; stage_k += STAGE_K) {
        if (valid_row) {
            device const uchar *blk = wrow + uint64_t(stage_k >> 8u) * block_bytes;
            const float d = qw3_f16_to_f32(
                ushort(blk[0]) | (ushort(blk[1]) << 8u));
            const float dmin = qw3_f16_to_f32(
                ushort(blk[2]) | (ushort(blk[3]) << 8u));
            device const uchar *scales = blk + 4u;
            device const uchar *qh = blk + 16u;
            device const uchar *ql = blk + ql_offset;
            const uint l0 = uint(half_k * 16);
            const uint g0 = (stage_k & 255u) >> 5u;
            device const uchar *qgroup = ql + (g0 >> 1u) * 32u + l0;
            for (uint sub = 0u; sub < 2u; sub++) {
                const uint g = g0 + sub;
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
                const float dl = d * float(sc);
                const float ml = dmin * float(mn);
                const uint high_mask = 1u << g;
                const bool high_nibble = (g & 1u) != 0u;
                for (int i = 0; i < 16; i++) {
                    const uint l = l0 + uint(i);
                    const uint packed = uint(qgroup[i]);
                    uint q = high_nibble ? packed >> 4u : packed & 15u;
                    if (is_q5 && (uint(qh[l]) & high_mask)) q += 16u;
                    sa[row * STAGE_K + int(sub * 32u) + half_k * 16 + i] =
                        half(dl * float(q) - ml);
                }
            }
        } else {
            for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK)
                for (int i = 0; i < 16; i++)
                    sa[row * STAGE_K + int(sub_k) + half_k * 16 + i] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
            auto mA = tA.slice(int(sub_k), 0);
            if (token_group_active) {
                auto mB = tB.slice(int(stage_k + sub_k), token0);
                mm.run(mB, mA, cT);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out), int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    if (token_group_active) cT.store(tD.slice(r0, token0));
}
#endif
inline float qw3_iq2_grid_value(ushort packed, uint j) {
    const uint k = (uint(packed) >> (2u * j)) & 3u;
    return k == 3u ? 0.0f : float(8u + k * 17u + (k >> 1u));
}
inline float qw3_iq3xxs_grid_value(ushort packed, uint j) {
    const uint k = (uint(packed) >> (3u * j)) & 7u;
    return float((k << 3u) + 4u + (k == 7u ? 2u : 0u));
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
/* Decode IQ2 in 32-value units, with two output rows per SIMD group.
 * The lookup magnitudes are exact bytes; expand them to FP32 only when used.
 * This halves the workgroup cache, without rounding weights or activations. */
template<ushort Kind>
inline float qw3_iq2_dot32_vector(device const uchar *blk,
                                  thread const float4 *xx,
                                  threadgroup const uchar4 *grid,
                                  threadgroup const uchar *signs, uint ib) {
    const float d = float(*((device const half *)blk));
    device const uchar *qs = blk + 2u;
    uint packed = 0u, scale = 0u;
    if (Kind == 0u) {
        device const uchar *sp = qs + 8u * ib + 4u;
        packed = uint(sp[0]) | (uint(sp[1]) << 8u) |
                 (uint(sp[2]) << 16u) | (uint(sp[3]) << 24u);
    } else {
        scale = uint(blk[(Kind == 1u ? 66u : 74u) + ib]);
    }
    float sum = 0.0f;
    for (uint l = 0; l < 4u; l++) {
        uint idx, s;
        if (Kind == 0u) {
            idx = uint(qs[8u * ib + l]);
            s = uint(signs[(packed >> (7u * l)) & 127u]);
        } else if (Kind == 1u) {
            const uint off = 2u * (4u * ib + l);
            const uint q = uint(qs[off]) | (uint(qs[off + 1u]) << 8u);
            idx = q & 511u;
            s = uint(signs[q >> 9u]);
        } else {
            idx = uint(qs[4u * ib + l]) |
                  ((uint(blk[66u + ib]) << (8u - 2u * l)) & 0x300u);
            s = uint(blk[34u + 4u * ib + l]);
        }
        const float4 sign1 = select(float4(1.0f), float4(-1.0f),
                                   (uint4(s) & uint4(1u, 2u, 4u, 8u)) != 0u);
        const float4 sign2 = select(float4(1.0f), float4(-1.0f),
                                   (uint4(s) & uint4(16u, 32u, 64u, 128u)) != 0u);
        const float v = dot(float4(grid[2u * idx]) * sign1, xx[2u * l]) +
                        dot(float4(grid[2u * idx + 1u]) * sign2, xx[2u * l + 1u]);
        if (Kind == 0u) sum += v;
        else sum += (0.5f + float((scale >> (4u * (l >> 1u))) & 15u)) * v;
    }
    return d * 0.25f * (Kind == 0u ? (0.5f + float(packed >> 28u)) * sum : sum);
}
template<ushort Kind, ushort Simdgroups = 2, bool AddX0 = false>
kernel void qw3_matvec_iq2_fast(
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
    constexpr uint NG = Kind == 0u ? 256u : Kind == 1u ? 512u : 1024u;
    constexpr uint NB = Kind == 0u ? 66u : Kind == 1u ? 74u : 82u;
    threadgroup uchar4 grid[NG * 2u];
    threadgroup uchar signs[128];
    const uint tid = uint(simd_idx) * 32u + uint(lane);
    for (uint i = tid; i < NG; i += Simdgroups * 32u) {
        const ushort packed = grid_table[i];
        grid[2u * i] = uchar4(qw3_iq2_grid_value(packed, 0u),
                             qw3_iq2_grid_value(packed, 1u),
                             qw3_iq2_grid_value(packed, 2u),
                             qw3_iq2_grid_value(packed, 3u));
        grid[2u * i + 1u] = uchar4(qw3_iq2_grid_value(packed, 4u),
                                  qw3_iq2_grid_value(packed, 5u),
                                  qw3_iq2_grid_value(packed, 6u),
                                  qw3_iq2_grid_value(packed, 7u));
    }
    if (Kind != 2u)
        for (uint i = tid; i < 128u; i += Simdgroups * 32u) signs[i] = sign_table[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row0 = (group * Simdgroups + uint(simd_idx)) * 2u;
    const uint row1 = row0 + 1u;
    float sum0 = 0.0f, sum1 = 0.0f;
    for (uint b32 = uint(lane); b32 < args.n_in / 32u; b32 += 32u) {
        const uint b = b32 / 8u, ib = b32 & 7u;
        device const float4 *x4 = (device const float4 *)(x + uint64_t(b32) * 32ull);
        float4 xx[8];
        for (uint j = 0; j < 8u; j++) xx[j] = x4[j];
        if (row0 < args.n_out)
            sum0 += qw3_iq2_dot32_vector<Kind>(
                weights + uint64_t(row0) * args.row_bytes + uint64_t(b) * NB,
                xx, grid, signs, ib);
        if (row1 < args.n_out)
            sum1 += qw3_iq2_dot32_vector<Kind>(
                weights + uint64_t(row1) * args.row_bytes + uint64_t(b) * NB,
                xx, grid, signs, ib);
    }
    sum0 = simd_sum(sum0); sum1 = simd_sum(sum1);
    if (lane == 0) {
        if (row0 < args.n_out) out[row0] = AddX0 ? (out[row0] + sum0) : sum0;
        if (row1 < args.n_out) out[row1] = AddX0 ? (out[row1] + sum1) : sum1;
    }
}
typedef decltype(qw3_matvec_iq2_fast<0>) qw3_matvec_iq2_fast_t;
template [[host_name("qw3_matvec_iq2_xxs_fast")]] kernel qw3_matvec_iq2_fast_t qw3_matvec_iq2_fast<0>;
template [[host_name("qw3_matvec_iq2_xs_fast")]] kernel qw3_matvec_iq2_fast_t qw3_matvec_iq2_fast<1>;
template [[host_name("qw3_matvec_iq2_s_fast")]] kernel qw3_matvec_iq2_fast_t qw3_matvec_iq2_fast<2>;
/* Amortize the 4/8 KiB lookup cache over eight SIMD groups on M5, without
 * increasing per-thread register work or changing either dot-product sum. */
template [[host_name("qw3_matvec_iq2_xs_sg8")]] kernel qw3_matvec_iq2_fast_t qw3_matvec_iq2_fast<1, 8>;
template [[host_name("qw3_matvec_iq2_s_sg8")]] kernel qw3_matvec_iq2_fast_t qw3_matvec_iq2_fast<2, 8>;
template [[host_name("qw3_matvec_iq2_s_add_x0_sg8")]] kernel qw3_matvec_iq2_fast_t qw3_matvec_iq2_fast<2, 8, true>;

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
struct qw3_iq1_group {
    float d, bias;
    uint packed;
};

// Each lattice entry describes eight consecutive weights of an IQ1 block.
inline qw3_iq1_group qw3_unpack_iq1_group(device const uchar *block, uint block_bytes,
                                         uint lane, device const ushort *grid) {
    uint index, scale;
    bool negative;
    ushort bits;
    if (block_bytes == 50u) {
        uint h = ((device const ushort *)(block + 34))[lane / 4u];
        index = block[2 + lane] | (((h >> (3u * (lane % 4u))) & 7u) << 8u);
        scale = (h >> 12u) & 7u;
        negative = (h & 0x8000u) != 0u;
        bits = *((device const ushort *)block);
    } else {
        device const ushort *sc = (device const ushort *)(block + 48);
        bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0xf0) |
               ((sc[2] >> 4) & 0xf00) | (sc[3] & 0xf000);
        uint h = block[32 + lane / 2u] >> (4u * (lane % 2u));
        index = block[lane] | ((h & 7u) << 8u);
        scale = (sc[lane / 8u] >> (3u * ((lane / 2u) % 4u))) & 7u;
        negative = (h & 8u) != 0u;
    }
    float d = float(as_type<half>(bits)) * float(2u * scale + 1u);
    float bias = negative ? -1.125f : -0.875f;
    uint packed = grid[index];
    return {d, bias, packed};
}

inline float qw3_iq1_weight(qw3_iq1_group group, uint j) {
    return group.d * (float((group.packed >> (2u*j)) & 3u) + group.bias);
}

#ifdef QW3_METAL_HAS_TENSOR
template<bool IQ1>
kernel void qw3_matmul_lowbit_mpp_direct(
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
    constexpr int NR1 = 64;
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
            if (IQ1) {
                const uint block_bytes = args.row_bytes / (args.n_in / 256u);
                device const uchar *block = wrow + ulong(k0 / 256u) * block_bytes;
                for (uint sub = 0; sub < 2; sub++) {
                    uint lane = (k0 % 256u) / 8u + sub;
                    qw3_iq1_group values = qw3_unpack_iq1_group(block, block_bytes, lane, grid_table);
                    for (uint j = 0; j < 8; j++)
                        sa[row*NK + half_k*16 + sub*8 + j] = half(qw3_iq1_weight(values, j));
                }
            } else {
                device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 66ull;
                const float d = qw3_f16_to_f32(
                    ushort(blk[0]) | (ushort(blk[1]) << 8u));
                device const uchar *qs = blk + 2u;
                const uint il = (k0 & 255u) >> 4u;
                const uint ib32 = il >> 1u;
                const uint half16 = il & 1u;
                device const uchar *q = qs + ib32 * 8u;
                const uint packed = uint(q[4]) | (uint(q[5]) << 8u) |
                    (uint(q[6]) << 16u) | (uint(q[7]) << 24u);
                const float db = d *
                    (0.5f + float(packed >> 28u)) * 0.25f;
                const ushort grid0 = grid_table[uint(q[2u * half16])];
                const ushort grid1 = grid_table[uint(q[2u * half16 + 1u])];
                const uint sign0 = uint(sign_table[
                    (packed >> (14u * half16)) & 127u]);
                const uint sign1 = uint(sign_table[
                    (packed >> (14u * half16 + 7u)) & 127u]);
                for (uint i = 0; i < 8u; i++) {
                    const float s0 = (sign0 & (1u << i)) ? -1.0f : 1.0f;
                    const float s1 = (sign1 & (1u << i)) ? -1.0f : 1.0f;
                    sa[row * NK + half_k * 16 + i] = half(
                        db * qw3_iq2_grid_value(grid0, i) * s0);
                    sa[row * NK + half_k * 16 + 8u + i] = half(
                        db * qw3_iq2_grid_value(grid1, i) * s1);
                }
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

typedef decltype(qw3_matmul_lowbit_mpp_direct<false>) qw3_matmul_lowbit_mpp_t;
template [[host_name("qw3_matmul_iq2_xxs_mpp_direct")]] kernel qw3_matmul_lowbit_mpp_t qw3_matmul_lowbit_mpp_direct<false>;
template [[host_name("qw3_matmul_iq1_mpp_direct")]] kernel qw3_matmul_lowbit_mpp_t qw3_matmul_lowbit_mpp_direct<true>;

template<bool IQ1>
kernel void qw3_matmul_lowbit_mpp_staged128(
    constant qw3_matmul_q8_0_batch_args &args,
    device const uchar *weights,
    device const float *x,
    device float *out,
    device const ushort *grid_table,
    device const uchar *sign_table,
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
    const int token0 = r1 + int(simd_group) * ROWS_PER_SG;
    const bool token_group_active = QW3_PARTIAL_LINEAR_TILE(token0, args.n_tokens);
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
    for (uint16_t i = 0; i < cT.get_capacity(); i++)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    const int row = int(tid) / 2;
    const int half_k = int(tid) & 1;
    const bool valid_row = r0 + row < int(args.n_out);
    device const uchar *wrow = weights + uint64_t(r0 + row) * args.row_bytes;
    const uint block_bytes = IQ1 ? (args.row_bytes / (args.n_in / 256u)) : 66u;
    for (uint stage_k = 0u; stage_k < args.n_in; stage_k += STAGE_K) {
        if (valid_row) {
            device const uchar *blk = wrow + uint64_t(stage_k >> 8u) * uint64_t(block_bytes);
            if (IQ1) {
                for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
                    const uint k0 = stage_k + sub_k + uint(half_k * 16);
                    for (uint sub = 0; sub < 2u; sub++) {
                        uint lane = (k0 & 255u) / 8u + sub;
                        qw3_iq1_group values = qw3_unpack_iq1_group(blk, block_bytes, lane, grid_table);
                        for (uint j = 0; j < 8u; j++)
                            sa[row * STAGE_K + int(sub_k) + half_k * 16 + int(sub * 8u + j)] =
                                half(qw3_iq1_weight(values, j));
                    }
                }
            } else {
                const float d = qw3_f16_to_f32(
                    ushort(blk[0]) | (ushort(blk[1]) << 8u));
                device const uchar *qs = blk + 2u;
                for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
                    const uint k0 = stage_k + sub_k + uint(half_k * 16);
                    const uint il = (k0 & 255u) >> 4u;
                    const uint ib32 = il >> 1u;
                    const uint half16 = il & 1u;
                    device const uchar *q = qs + ib32 * 8u;
                    const uint packed = uint(q[4]) | (uint(q[5]) << 8u) |
                        (uint(q[6]) << 16u) | (uint(q[7]) << 24u);
                    const float db = d * (0.5f + float(packed >> 28u)) * 0.25f;
                    const ushort grid0 = grid_table[uint(q[2u * half16])];
                    const ushort grid1 = grid_table[uint(q[2u * half16 + 1u])];
                    const uint sign0 = uint(sign_table[(packed >> (14u * half16)) & 127u]);
                    const uint sign1 = uint(sign_table[(packed >> (14u * half16 + 7u)) & 127u]);
                    for (uint i = 0; i < 8u; i++) {
                        const float s0 = (sign0 & (1u << i)) ? -1.0f : 1.0f;
                        const float s1 = (sign1 & (1u << i)) ? -1.0f : 1.0f;
                        sa[row * STAGE_K + int(sub_k) + half_k * 16 + int(i)] =
                            half(db * qw3_iq2_grid_value(grid0, i) * s0);
                        sa[row * STAGE_K + int(sub_k) + half_k * 16 + 8 + int(i)] =
                            half(db * qw3_iq2_grid_value(grid1, i) * s1);
                    }
                }
            }
        } else {
            for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK)
                for (int i = 0; i < 16; i++)
                    sa[row * STAGE_K + int(sub_k) + half_k * 16 + i] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
            auto mA = tA.slice(int(sub_k), 0);
            if (token_group_active) {
                auto mB = tB.slice(int(stage_k + sub_k), token0);
                mm.run(mB, mA, cT);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out), int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    if (token_group_active) cT.store(tD.slice(r0, token0));
}

typedef decltype(qw3_matmul_lowbit_mpp_staged128<false>) qw3_matmul_lowbit_mpp_staged128_t;
template [[host_name("qw3_matmul_iq2_xxs_mpp_staged128")]] kernel qw3_matmul_lowbit_mpp_staged128_t qw3_matmul_lowbit_mpp_staged128<false>;
template [[host_name("qw3_matmul_iq1_mpp_staged128")]] kernel qw3_matmul_lowbit_mpp_staged128_t qw3_matmul_lowbit_mpp_staged128<true>;

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
    constexpr int NR1 = 64;
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
            const uint il = (k0 & 255u) >> 4u;
            const uint ib32 = il >> 1u;
            const uint half16 = il & 1u;
            const uint off = 2u * (4u * ib32 + 2u * half16);
            const uint q0 = uint(qs[off]) | (uint(qs[off + 1u]) << 8u);
            const uint q1 = uint(qs[off + 2u]) |
                (uint(qs[off + 3u]) << 8u);
            const float db = d *
                (0.5f + float((uint(scales[ib32]) >>
                              (4u * half16)) & 15u)) * 0.25f;
            const ushort grid0 = grid_table[q0 & 511u];
            const ushort grid1 = grid_table[q1 & 511u];
            const uint sign0 = uint(sign_table[q0 >> 9u]);
            const uint sign1 = uint(sign_table[q1 >> 9u]);
            for (uint i = 0; i < 8u; i++) {
                const float s0 = (sign0 & (1u << i)) ? -1.0f : 1.0f;
                const float s1 = (sign1 & (1u << i)) ? -1.0f : 1.0f;
                sa[row * NK + half_k * 16 + i] = half(
                    db * qw3_iq2_grid_value(grid0, i) * s0);
                sa[row * NK + half_k * 16 + 8u + i] = half(
                    db * qw3_iq2_grid_value(grid1, i) * s1);
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

kernel void qw3_matmul_iq2_xs_mpp_staged128(
    constant qw3_matmul_q8_0_batch_args &args,
    device const uchar *weights,
    device const float *x,
    device float *out,
    device const ushort *grid_table,
    device const uchar *sign_table,
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
    const int token0 = r1 + int(simd_group) * ROWS_PER_SG;
    const bool token_group_active = QW3_PARTIAL_LINEAR_TILE(token0, args.n_tokens);
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
    for (uint16_t i = 0; i < cT.get_capacity(); i++)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    const int row = int(tid) / 2;
    const int half_k = int(tid) & 1;
    const bool valid_row = r0 + row < int(args.n_out);
    device const uchar *wrow = weights + uint64_t(r0 + row) * args.row_bytes;
    for (uint stage_k = 0u; stage_k < args.n_in; stage_k += STAGE_K) {
        if (valid_row) {
            device const uchar *blk = wrow + uint64_t(stage_k >> 8u) * 74ull;
            const float d = qw3_f16_to_f32(
                ushort(blk[0]) | (ushort(blk[1]) << 8u));
            device const uchar *qs = blk + 2u;
            device const uchar *scales = qs + 64u;
            for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
                const uint k0 = stage_k + sub_k + uint(half_k * 16);
                const uint il = (k0 & 255u) >> 4u;
                const uint ib32 = il >> 1u;
                const uint half16 = il & 1u;
                const uint off = 2u * (4u * ib32 + 2u * half16);
                const uint q0 = uint(qs[off]) | (uint(qs[off + 1u]) << 8u);
                const uint q1 = uint(qs[off + 2u]) | (uint(qs[off + 3u]) << 8u);
                const float db = d *
                    (0.5f + float((uint(scales[ib32]) >> (4u * half16)) & 15u)) * 0.25f;
                const ushort grid0 = grid_table[q0 & 511u];
                const ushort grid1 = grid_table[q1 & 511u];
                const uint sign0 = uint(sign_table[q0 >> 9u]);
                const uint sign1 = uint(sign_table[q1 >> 9u]);
                for (uint i = 0; i < 8u; i++) {
                    const float s0 = (sign0 & (1u << i)) ? -1.0f : 1.0f;
                    const float s1 = (sign1 & (1u << i)) ? -1.0f : 1.0f;
                    sa[row * STAGE_K + int(sub_k) + half_k * 16 + int(i)] =
                        half(db * qw3_iq2_grid_value(grid0, i) * s0);
                    sa[row * STAGE_K + int(sub_k) + half_k * 16 + 8 + int(i)] =
                        half(db * qw3_iq2_grid_value(grid1, i) * s1);
                }
            }
        } else {
            for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK)
                for (int i = 0; i < 16; i++)
                    sa[row * STAGE_K + int(sub_k) + half_k * 16 + i] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
            auto mA = tA.slice(int(sub_k), 0);
            if (token_group_active) {
                auto mB = tB.slice(int(stage_k + sub_k), token0);
                mm.run(mB, mA, cT);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out), int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    if (token_group_active) cT.store(tD.slice(r0, token0));
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
    constexpr int NR1 = 64;
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
            const uint il = (k0 & 255u) >> 4u;
            const uint ib32 = il >> 1u;
            const uint half16 = il & 1u;
            device const uchar *q = qs + 4u * ib32 + 2u * half16;
            device const uchar *sgn = signs + 4u * ib32 + 2u * half16;
            const uint qh4 = uint(qh[ib32]) >> (4u * half16);
            const float db = d *
                (0.5f + float((uint(scales[ib32]) >>
                              (4u * half16)) & 15u)) * 0.25f;
            const ushort grid0 = grid_table[
                uint(q[0]) | ((qh4 << 8u) & 0x300u)];
            const ushort grid1 = grid_table[
                uint(q[1]) | ((qh4 << 6u) & 0x300u)];
            const uint sign0 = uint(sgn[0]);
            const uint sign1 = uint(sgn[1]);
            for (uint i = 0; i < 8u; i++) {
                const float s0 = (sign0 & (1u << i)) ? -1.0f : 1.0f;
                const float s1 = (sign1 & (1u << i)) ? -1.0f : 1.0f;
                sa[row * NK + half_k * 16 + i] = half(
                    db * qw3_iq2_grid_value(grid0, i) * s0);
                sa[row * NK + half_k * 16 + 8u + i] = half(
                    db * qw3_iq2_grid_value(grid1, i) * s1);
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

kernel void qw3_matmul_iq2_s_mpp_staged128(
    constant qw3_matmul_q8_0_batch_args &args,
    device const uchar *weights,
    device const float *x,
    device float *out,
    device const ushort *grid_table,
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
    const int token0 = r1 + int(simd_group) * ROWS_PER_SG;
    const bool token_group_active = QW3_PARTIAL_LINEAR_TILE(token0, args.n_tokens);
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
    for (uint16_t i = 0; i < cT.get_capacity(); i++)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    const int row = int(tid) / 2;
    const int half_k = int(tid) & 1;
    const bool valid_row = r0 + row < int(args.n_out);
    device const uchar *wrow = weights + uint64_t(r0 + row) * args.row_bytes;
    for (uint stage_k = 0u; stage_k < args.n_in; stage_k += STAGE_K) {
        if (valid_row) {
            device const uchar *blk = wrow + uint64_t(stage_k >> 8u) * 82ull;
            const float d = qw3_f16_to_f32(
                ushort(blk[0]) | (ushort(blk[1]) << 8u));
            device const uchar *qs = blk + 2u;
            device const uchar *signs = qs + 32u;
            device const uchar *qh = qs + 64u;
            device const uchar *scales = qh + 8u;
            for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
                const uint k0 = stage_k + sub_k + uint(half_k * 16);
                const uint il = (k0 & 255u) >> 4u;
                const uint ib32 = il >> 1u;
                const uint half16 = il & 1u;
                device const uchar *q = qs + 4u * ib32 + 2u * half16;
                device const uchar *sgn = signs + 4u * ib32 + 2u * half16;
                const uint qh4 = uint(qh[ib32]) >> (4u * half16);
                const float db = d *
                    (0.5f + float((uint(scales[ib32]) >> (4u * half16)) & 15u)) * 0.25f;
                const ushort grid0 = grid_table[uint(q[0]) | ((qh4 << 8u) & 0x300u)];
                const ushort grid1 = grid_table[uint(q[1]) | ((qh4 << 6u) & 0x300u)];
                const uint sign0 = uint(sgn[0]);
                const uint sign1 = uint(sgn[1]);
                for (uint i = 0; i < 8u; i++) {
                    const float s0 = (sign0 & (1u << i)) ? -1.0f : 1.0f;
                    const float s1 = (sign1 & (1u << i)) ? -1.0f : 1.0f;
                    sa[row * STAGE_K + int(sub_k) + half_k * 16 + int(i)] =
                        half(db * qw3_iq2_grid_value(grid0, i) * s0);
                    sa[row * STAGE_K + int(sub_k) + half_k * 16 + 8 + int(i)] =
                        half(db * qw3_iq2_grid_value(grid1, i) * s1);
                }
            }
        } else {
            for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK)
                for (int i = 0; i < 16; i++)
                    sa[row * STAGE_K + int(sub_k) + half_k * 16 + i] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
            auto mA = tA.slice(int(sub_k), 0);
            if (token_group_active) {
                auto mB = tB.slice(int(stage_k + sub_k), token0);
                mm.run(mB, mA, cT);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out), int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    if (token_group_active) cT.store(tD.slice(r0, token0));
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
inline float4 qw3_iq3xxs_float_grid(ushort packed) {
    constexpr float vals[8] = {4.0f, 12.0f, 20.0f, 28.0f, 36.0f, 44.0f, 52.0f, 62.0f};
    const uint g = uint(packed);
    return float4(vals[g & 7u], vals[(g >> 3u) & 7u],
                  vals[(g >> 6u) & 7u], vals[(g >> 9u) & 7u]);
}
inline float qw3_iq3xxs_dot32_vector(device const uchar *blk,
                                    thread const float4 *xx,
                                    threadgroup const float4 *grid,
                                    threadgroup const uchar *signs, uint ib) {
    const float d = float(*((device const half *)blk));
    device const uchar *qs = blk + 2u + 8u * ib;
    device const uchar *sp = blk + 66u + 4u * ib;
    const uint packed = uint(sp[0]) | (uint(sp[1]) << 8u) |
                        (uint(sp[2]) << 16u) | (uint(sp[3]) << 24u);
    const float db = d * (0.5f + float(packed >> 28u)) * 0.5f;
    float sum = 0.0f;
    for (uint l = 0; l < 4u; l++) {
        const uint s = uint(signs[(packed >> (7u * l)) & 127u]);
        const float4 sign1 = select(float4(1.0f), float4(-1.0f),
                                   (uint4(s) & uint4(1u, 2u, 4u, 8u)) != 0u);
        const float4 sign2 = select(float4(1.0f), float4(-1.0f),
                                   (uint4(s) & uint4(16u, 32u, 64u, 128u)) != 0u);
        sum += dot(grid[uint(qs[2u * l])] * sign1, xx[2u * l]) +
               dot(grid[uint(qs[2u * l + 1u])] * sign2, xx[2u * l + 1u]);
    }
    return db * sum;
}
template<ushort Simdgroups = 2, bool AddX0 = false>
kernel void qw3_matvec_iq3_xxs_fast_tpl(
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
    /* Split quant blocks into 32-value units so all lanes participate, even
     * for 5120 inputs (only 20 whole blocks). Pre-expanded float4 grid avoids
     * inner-loop byte-to-float unpack overhead. */
    threadgroup float4 grid[256];
    threadgroup uchar signs[128];
    const uint tid = uint(simd_idx) * 32u + uint(lane);
    for (uint i = tid; i < 256u; i += Simdgroups * 32u)
        grid[i] = qw3_iq3xxs_float_grid(grid_table[i]);
    for (uint i = tid; i < 128u; i += Simdgroups * 32u) signs[i] = sign_table[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row0 = (group * Simdgroups + uint(simd_idx)) * 2u;
    uint row1 = row0 + 1u;
    float sum0 = 0.0f, sum1 = 0.0f;
    uint nb32 = args.n_in / 32u;
    for (uint b32 = uint(lane); b32 < nb32; b32 += 32u) {
        const uint b = b32 / 8u, ib = b32 & 7u;
        device const float4 *x4 = (device const float4 *)(x + uint64_t(b32) * 32ull);
        float4 xx[8];
        for (uint j = 0; j < 8u; j++) xx[j] = x4[j];
        if (row0 < args.n_out)
            sum0 += qw3_iq3xxs_dot32_vector(
                weights + uint64_t(row0) * uint64_t(args.row_bytes) +
                uint64_t(b) * 98ull, xx, grid, signs, ib);
        if (row1 < args.n_out)
            sum1 += qw3_iq3xxs_dot32_vector(
                weights + uint64_t(row1) * uint64_t(args.row_bytes) +
                uint64_t(b) * 98ull, xx, grid, signs, ib);
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (lane == 0) {
        if (row0 < args.n_out) out[row0] = AddX0 ? (out[row0] + sum0) : sum0;
        if (row1 < args.n_out) out[row1] = AddX0 ? (out[row1] + sum1) : sum1;
    }
}
typedef decltype(qw3_matvec_iq3_xxs_fast_tpl<2, false>) qw3_matvec_iq3_xxs_fast_t;
template [[host_name("qw3_matvec_iq3_xxs_fast")]] kernel qw3_matvec_iq3_xxs_fast_t qw3_matvec_iq3_xxs_fast_tpl<2, false>;
template [[host_name("qw3_matvec_iq3_xxs_sg8")]] kernel qw3_matvec_iq3_xxs_fast_t qw3_matvec_iq3_xxs_fast_tpl<8, false>;
template [[host_name("qw3_matvec_iq3_xxs_add_x0_sg8")]] kernel qw3_matvec_iq3_xxs_fast_t qw3_matvec_iq3_xxs_fast_tpl<8, true>;
// Same FP32 dot32 grouping as the production IQ3_XXS vector GEMV.
// Decode each 32-weight unit once for up to four prompt rows; no half weights.
kernel void qw3_matmul_iq3_xxs_short(
    constant qw3_matmul_q8_0_batch_args &args [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *out [[buffer(3)]],
    device const ushort *grid_table [[buffer(4)]],
    device const uchar *sign_table [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort simd_idx [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float4 grid[256];
    threadgroup uchar signs[128];
    const uint tid = uint(simd_idx) * 32u + uint(lane);
    constexpr float values[8] = {4.0f,12.0f,20.0f,28.0f,36.0f,44.0f,52.0f,62.0f};
    for (uint i = tid; i < 256u; i += 128u) {
        const ushort p = grid_table[i];
        grid[i] = float4(values[p & 7u], values[(p >> 3u) & 7u],
                         values[(p >> 6u) & 7u], values[(p >> 9u) & 7u]);
    }
    if (tid < 128u) signs[tid] = sign_table[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row = group.x * 4u + uint(simd_idx);
    if (row >= args.n_out) return;
    const uint t0 = group.y * 4u;
    if (t0 >= args.n_tokens) return;
    const uint count = min(4u, args.n_tokens - t0);
    float4 sums = float4(0.0f);
    for (uint b32 = uint(lane); b32 < args.n_in / 32u; b32 += 32u) {
        const uint ib = b32 & 7u;
        device const uchar *blk = weights + uint64_t(row) * args.row_bytes +
                                 uint64_t(b32 / 8u) * 98ull;
        device const uchar *qs = blk + 2u + 8u * ib;
        device const uchar *sp = blk + 66u + 4u * ib;
        const uint packed = uint(sp[0]) | (uint(sp[1]) << 8u) |
                            (uint(sp[2]) << 16u) | (uint(sp[3]) << 24u);
        const float db = float(*((device const half *)blk)) *
                         (0.5f + float(packed >> 28u)) * 0.5f;
        float4 dots = float4(0.0f);
        for (uint l = 0; l < 4u; l++) {
            const uint s = uint(signs[(packed >> (7u * l)) & 127u]);
            const float4 w0 = grid[uint(qs[2u*l])] *
                select(float4(1.0f), float4(-1.0f),
                       (uint4(s) & uint4(1u,2u,4u,8u)) != 0u);
            const float4 w1 = grid[uint(qs[2u*l+1u])] *
                select(float4(1.0f), float4(-1.0f),
                       (uint4(s) & uint4(16u,32u,64u,128u)) != 0u);
            for (uint r = 0; r < count; r++) {
                device const float *xp = x + uint64_t(t0+r) * args.in_stride +
                    args.in_offset + b32*32u + l*8u;
                const float4 x0 = float4(xp[0],xp[1],xp[2],xp[3]);
                const float4 x1 = float4(xp[4],xp[5],xp[6],xp[7]);
                dots[r] += dot(w0,x0) + dot(w1,x1);
            }
        }
        sums += db * dots;
    }
    sums.x = simd_sum(sums.x); sums.y = simd_sum(sums.y);
    sums.z = simd_sum(sums.z); sums.w = simd_sum(sums.w);
    if (lane == 0u) {
        for (uint r = 0; r < count; r++)
            out[uint64_t(t0+r) * args.out_stride + args.out_offset + row] = sums[r];
    }
}

kernel void qw3_matmul_iq2_s_short(
    constant qw3_matmul_q8_0_batch_args &args [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *out [[buffer(3)]],
    device const ushort *grid_table [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort simd_idx [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup uchar4 grid[2048];
    const uint tid = uint(simd_idx) * 32u + uint(lane);
    constexpr uchar values[4] = {8,25,43,0};
    for (uint i = tid; i < 1024u; i += 128u) {
        const ushort p = grid_table[i];
        grid[2u*i] = uchar4(values[p&3u],values[(p>>2u)&3u],
                            values[(p>>4u)&3u],values[(p>>6u)&3u]);
        grid[2u*i+1u] = uchar4(values[(p>>8u)&3u],values[(p>>10u)&3u],
                               values[(p>>12u)&3u],values[(p>>14u)&3u]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row = group.x * 4u + uint(simd_idx);
    if (row >= args.n_out) return;
    const uint t0 = group.y * 4u;
    if (t0 >= args.n_tokens) return;
    const uint count = min(4u,args.n_tokens-t0);
    float4 sums = float4(0.0f);
    for (uint b32 = uint(lane); b32 < args.n_in/32u; b32 += 32u) {
        const uint ib = b32 & 7u;
        device const uchar *blk = weights + uint64_t(row)*args.row_bytes +
                                 uint64_t(b32/8u)*82ull;
        device const uchar *qs = blk + 2u;
        const uint scale = uint(blk[74u+ib]);
        const float d = float(*((device const half *)blk));
        float4 dots = float4(0.0f);
        for (uint l = 0; l < 4u; l++) {
            const uint idx = uint(qs[4u*ib+l]) |
                ((uint(blk[66u+ib]) << (8u-2u*l)) & 0x300u);
            const uint s = uint(blk[34u+4u*ib+l]);
            const float4 w0 = float4(grid[2u*idx]) * select(float4(1.0f),float4(-1.0f),
                (uint4(s)&uint4(1u,2u,4u,8u)) != 0u);
            const float4 w1 = float4(grid[2u*idx+1u]) * select(float4(1.0f),float4(-1.0f),
                (uint4(s)&uint4(16u,32u,64u,128u)) != 0u);
            const float sc = 0.5f+float((scale >> (4u*(l>>1u)))&15u);
            for (uint r = 0; r < count; r++) {
                device const float *xp = x + uint64_t(t0+r)*args.in_stride +
                    args.in_offset+b32*32u+l*8u;
                const float4 x0 = float4(xp[0],xp[1],xp[2],xp[3]);
                const float4 x1 = float4(xp[4],xp[5],xp[6],xp[7]);
                dots[r] += sc*(dot(w0,x0)+dot(w1,x1));
            }
        }
        sums += d*0.25f*dots;
    }
    sums.x = simd_sum(sums.x); sums.y = simd_sum(sums.y);
    sums.z = simd_sum(sums.z); sums.w = simd_sum(sums.w);
    if (lane == 0u) for (uint r = 0; r < count; r++)
        out[uint64_t(t0+r)*args.out_stride+args.out_offset+row] = sums[r];
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
    constexpr int NR1 = 64;
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
            const uint il = (k0 & 255u) >> 4u;
            const uint ib32 = il >> 1u;
            const uint half16 = il & 1u;
            device const uchar *q3 = qs + 8u * ib32;
            device const uchar *sp = ss + 4u * ib32;
            const uint packed = uint(sp[0]) | (uint(sp[1]) << 8u) |
                (uint(sp[2]) << 16u) | (uint(sp[3]) << 24u);
            const float db = d *
                (0.5f + float(packed >> 28u)) * 0.5f;
            const uint sign0 = uint(sign_table[
                (packed >> (14u * half16)) & 127u]);
            const uint sign1 = uint(sign_table[
                (packed >> (14u * half16 + 7u)) & 127u]);
            const ushort grid0 = grid_table[uint(q3[4u * half16])];
            const ushort grid1 = grid_table[uint(q3[4u * half16 + 1u])];
            const ushort grid2 = grid_table[uint(q3[4u * half16 + 2u])];
            const ushort grid3 = grid_table[uint(q3[4u * half16 + 3u])];
            for (uint i = 0; i < 4u; i++) {
                const float s00 = (sign0 & (1u << i)) ? -1.0f : 1.0f;
                const float s01 = (sign0 & (1u << (i + 4u))) ? -1.0f : 1.0f;
                const float s10 = (sign1 & (1u << i)) ? -1.0f : 1.0f;
                const float s11 = (sign1 & (1u << (i + 4u))) ? -1.0f : 1.0f;
                sa[row * NK + half_k * 16 + i] = half(
                    db * qw3_iq3xxs_grid_value(grid0, i) * s00);
                sa[row * NK + half_k * 16 + 4u + i] = half(
                    db * qw3_iq3xxs_grid_value(grid1, i) * s01);
                sa[row * NK + half_k * 16 + 8u + i] = half(
                    db * qw3_iq3xxs_grid_value(grid2, i) * s10);
                sa[row * NK + half_k * 16 + 12u + i] = half(
                    db * qw3_iq3xxs_grid_value(grid3, i) * s11);
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

kernel void qw3_matmul_iq3_xxs_mpp_staged128(
    constant qw3_matmul_q8_0_batch_args &args,
    device const uchar *weights,
    device const float *x,
    device float *out,
    device const ushort *grid_table,
    device const uchar *sign_table,
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
    const int token0 = r1 + int(simd_group) * ROWS_PER_SG;
    const bool token_group_active = QW3_PARTIAL_LINEAR_TILE(token0, args.n_tokens);
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
    for (uint16_t i = 0; i < cT.get_capacity(); i++)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    const int row = int(tid) / 2;
    const int half_k = int(tid) & 1;
    const bool valid_row = r0 + row < int(args.n_out);
    device const uchar *wrow = weights + uint64_t(r0 + row) * args.row_bytes;
    for (uint stage_k = 0u; stage_k < args.n_in; stage_k += STAGE_K) {
        if (valid_row) {
            device const uchar *blk = wrow + uint64_t(stage_k >> 8u) * 98ull;
            const float d = qw3_f16_to_f32(
                ushort(blk[0]) | (ushort(blk[1]) << 8u));
            device const uchar *qs = blk + 2u;
            device const uchar *ss = qs + 64u;
            for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
                const uint k0 = stage_k + sub_k + uint(half_k * 16);
                const uint il = (k0 & 255u) >> 4u;
                const uint ib32 = il >> 1u;
                const uint half16 = il & 1u;
                device const uchar *q3 = qs + 8u * ib32;
                device const uchar *sp = ss + 4u * ib32;
                const uint packed = uint(sp[0]) | (uint(sp[1]) << 8u) |
                    (uint(sp[2]) << 16u) | (uint(sp[3]) << 24u);
                const float db = d * (0.5f + float(packed >> 28u)) * 0.5f;
                const uint sign0 = uint(sign_table[(packed >> (14u * half16)) & 127u]);
                const uint sign1 = uint(sign_table[(packed >> (14u * half16 + 7u)) & 127u]);
                const ushort grid0 = grid_table[uint(q3[4u * half16])];
                const ushort grid1 = grid_table[uint(q3[4u * half16 + 1u])];
                const ushort grid2 = grid_table[uint(q3[4u * half16 + 2u])];
                const ushort grid3 = grid_table[uint(q3[4u * half16 + 3u])];
                for (uint i = 0; i < 4u; i++) {
                    const float s00 = (sign0 & (1u << i)) ? -1.0f : 1.0f;
                    const float s01 = (sign0 & (1u << (i + 4u))) ? -1.0f : 1.0f;
                    const float s10 = (sign1 & (1u << i)) ? -1.0f : 1.0f;
                    const float s11 = (sign1 & (1u << (i + 4u))) ? -1.0f : 1.0f;
                    sa[row * STAGE_K + int(sub_k) + half_k * 16 + int(i)] =
                        half(db * qw3_iq3xxs_grid_value(grid0, i) * s00);
                    sa[row * STAGE_K + int(sub_k) + half_k * 16 + 4 + int(i)] =
                        half(db * qw3_iq3xxs_grid_value(grid1, i) * s01);
                    sa[row * STAGE_K + int(sub_k) + half_k * 16 + 8 + int(i)] =
                        half(db * qw3_iq3xxs_grid_value(grid2, i) * s10);
                    sa[row * STAGE_K + int(sub_k) + half_k * 16 + 12 + int(i)] =
                        half(db * qw3_iq3xxs_grid_value(grid3, i) * s11);
                }
            }
        } else {
            for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK)
                for (int i = 0; i < 16; i++)
                    sa[row * STAGE_K + int(sub_k) + half_k * 16 + i] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
            auto mA = tA.slice(int(sub_k), 0);
            if (token_group_active) {
                auto mB = tB.slice(int(stage_k + sub_k), token0);
                mm.run(mB, mA, cT);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out), int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    if (token_group_active) cT.store(tD.slice(r0, token0));
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
    constexpr int NR1 = 64;
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

kernel void qw3_matmul_q6_k_mpp_staged128(
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
    const int token0 = r1 + int(simd_group) * ROWS_PER_SG;
    const bool token_group_active = QW3_PARTIAL_LINEAR_TILE(token0, args.n_tokens);
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
    for (uint16_t i = 0; i < cT.get_capacity(); i++)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    const int row = int(tid) >> 1;
    const uint sub = uint(tid) & 1u;
    const bool valid_row = r0 + row < int(args.n_out);
    device const uchar *wrow = weights + uint64_t(valid_row ? (r0 + row) : 0) * args.row_bytes;
    threadgroup half4 *sa4 = (threadgroup half4 *)(sa + row * STAGE_K + int(sub * 32u));
    for (uint stage_k = 0u; stage_k < args.n_in; stage_k += STAGE_K) {
        if (valid_row) {
            device const uchar *blk = wrow + uint64_t(stage_k >> 8u) * 210ull;
            const float d = qw3_f16_to_f32(ushort(blk[208]) | (ushort(blk[209]) << 8u));
            const uint local0 = stage_k & 255u;
            const uint half_block = local0 >> 7u;
            const uint part = ((local0 >> 5u) & 2u) + sub;
            const uint high_shift = 2u * part;
            const bool hi_nib = (part >= 2u);
            device const uchar *ql = blk + half_block * 64u + (part & 1u) * 32u;
            device const uchar *qh = blk + 128u + half_block * 32u;
            device const char *sc = (device const char *)(blk + 192u + half_block * 8u + 2u * part);
            const float dl0 = d * float(sc[0]);
            const float dl1 = d * float(sc[1]);
            for (uint g = 0u; g < 8u; g++) {
                const float dl = (g < 4u) ? dl0 : dl1;
                const uint l0 = g * 4u;
                const uint p0 = uint(ql[l0 + 0u]), p1 = uint(ql[l0 + 1u]);
                const uint p2 = uint(ql[l0 + 2u]), p3 = uint(ql[l0 + 3u]);
                const int q0 = int((hi_nib ? (p0 >> 4u) : (p0 & 15u)) | (((uint(qh[l0 + 0u]) >> high_shift) & 3u) << 4u)) - 32;
                const int q1 = int((hi_nib ? (p1 >> 4u) : (p1 & 15u)) | (((uint(qh[l0 + 1u]) >> high_shift) & 3u) << 4u)) - 32;
                const int q2 = int((hi_nib ? (p2 >> 4u) : (p2 & 15u)) | (((uint(qh[l0 + 2u]) >> high_shift) & 3u) << 4u)) - 32;
                const int q3 = int((hi_nib ? (p3 >> 4u) : (p3 & 15u)) | (((uint(qh[l0 + 3u]) >> high_shift) & 3u) << 4u)) - 32;
                sa4[g] = half4(float4(float(q0), float(q1), float(q2), float(q3)) * dl);
            }
        } else {
            for (int i = 0; i < 8; i++) sa4[i] = half4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint sub_k = 0u; sub_k < STAGE_K; sub_k += NK) {
            auto mA = tA.slice(int(sub_k), 0);
            if (token_group_active) {
                auto mB = tB.slice(int(stage_k + sub_k), token0);
                mm.run(mB, mA, cT);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                    dextents<int32_t, 2>(int(args.n_out), int(args.n_tokens)),
                    array<int, 2>({1, int(args.out_stride)}));
    if (token_group_active) cT.store(tD.slice(r0, token0));
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

// One SIMD group computes a row. Each lane decodes eight weights per block.
kernel void qw3_matvec_iq1(constant qw3_matvec_q8_0_args &args [[buffer(0)]],
                           device const uchar *weights [[buffer(1)]],
                           device const float *x [[buffer(2)]],
                           device float *out [[buffer(3)]],
                           device const ushort *grid [[buffer(4)]],
                           uint group [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint groups_per_tg [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    uint row = group * groups_per_tg + sg;
    if (row >= args.n_out) return;
    uint blocks = args.n_in / 256u;
    uint block_bytes = args.row_bytes / blocks;
    device const uchar *block = weights + ulong(row) * args.row_bytes;
    float sum = 0.0f;
    for (uint k = 0; k < blocks; k++, block += block_bytes) {
        qw3_iq1_group values = qw3_unpack_iq1_group(block, block_bytes, lane, grid);
        for (uint j = 0; j < 8; j++)
            sum += x[256u*k + 8u*lane + j] * qw3_iq1_weight(values, j);
    }
    sum = simd_sum(sum);
    if (!lane) out[row] = sum;
}

// Inkling uses 64-value NVFP4 blocks with four UE4M3 scales and E2M1 payloads.
constant char qw3_nvfp4_twice_e2m1[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};
constant float qw3_nvfp4_e2m1_f[16] = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
    0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f
};
constant half qw3_nvfp4_e2m1_h[16] = {
    0.0h, 1.0h, 2.0h, 3.0h, 4.0h, 6.0h, 8.0h, 12.0h,
    0.0h, -1.0h, -2.0h, -3.0h, -4.0h, -6.0h, -8.0h, -12.0h
};

static inline float qw3_nvfp4_scale(uint bits) {
    if (bits == 0x7fu) return 0.0f;
    const uint e = (bits >> 3u) & 15u;
    const uint m = bits & 7u;
    return e ? as_type<float>(((e + 119u) << 23u) | (m << 20u))
             : (float(m) * (1.0f / 1024.0f));
}
constant half qw3_nvfp4_scale_h[256] = {
    0.0h, 0.0009765625h, 0.001953125h, 0.0029296875h, 0.00390625h, 0.0048828125h, 0.005859375h, 0.0068359375h,
    0.0078125h, 0.0087890625h, 0.009765625h, 0.0107421875h, 0.01171875h, 0.0126953125h, 0.013671875h, 0.0146484375h,
    0.015625h, 0.017578125h, 0.01953125h, 0.021484375h, 0.0234375h, 0.025390625h, 0.02734375h, 0.029296875h,
    0.03125h, 0.03515625h, 0.0390625h, 0.04296875h, 0.046875h, 0.05078125h, 0.0546875h, 0.05859375h,
    0.0625h, 0.0703125h, 0.078125h, 0.0859375h, 0.09375h, 0.1015625h, 0.109375h, 0.1171875h,
    0.125h, 0.140625h, 0.15625h, 0.171875h, 0.1875h, 0.203125h, 0.21875h, 0.234375h,
    0.25h, 0.28125h, 0.3125h, 0.34375h, 0.375h, 0.40625h, 0.4375h, 0.46875h,
    0.5h, 0.5625h, 0.625h, 0.6875h, 0.75h, 0.8125h, 0.875h, 0.9375h,
    1.0h, 1.125h, 1.25h, 1.375h, 1.5h, 1.625h, 1.75h, 1.875h,
    2.0h, 2.25h, 2.5h, 2.75h, 3.0h, 3.25h, 3.5h, 3.75h,
    4.0h, 4.5h, 5.0h, 5.5h, 6.0h, 6.5h, 7.0h, 7.5h,
    8.0h, 9.0h, 10.0h, 11.0h, 12.0h, 13.0h, 14.0h, 15.0h,
    16.0h, 18.0h, 20.0h, 22.0h, 24.0h, 26.0h, 28.0h, 30.0h,
    32.0h, 36.0h, 40.0h, 44.0h, 48.0h, 52.0h, 56.0h, 60.0h,
    64.0h, 72.0h, 80.0h, 88.0h, 96.0h, 104.0h, 112.0h, 120.0h,
    128.0h, 144.0h, 160.0h, 176.0h, 192.0h, 208.0h, 224.0h, 0.0h,
    0.0h, 0.0009765625h, 0.001953125h, 0.0029296875h, 0.00390625h, 0.0048828125h, 0.005859375h, 0.0068359375h,
    0.0078125h, 0.0087890625h, 0.009765625h, 0.0107421875h, 0.01171875h, 0.0126953125h, 0.013671875h, 0.0146484375h,
    0.015625h, 0.017578125h, 0.01953125h, 0.021484375h, 0.0234375h, 0.025390625h, 0.02734375h, 0.029296875h,
    0.03125h, 0.03515625h, 0.0390625h, 0.04296875h, 0.046875h, 0.05078125h, 0.0546875h, 0.05859375h,
    0.0625h, 0.0703125h, 0.078125h, 0.0859375h, 0.09375h, 0.1015625h, 0.109375h, 0.1171875h,
    0.125h, 0.140625h, 0.15625h, 0.171875h, 0.1875h, 0.203125h, 0.21875h, 0.234375h,
    0.25h, 0.28125h, 0.3125h, 0.34375h, 0.375h, 0.40625h, 0.4375h, 0.46875h,
    0.5h, 0.5625h, 0.625h, 0.6875h, 0.75h, 0.8125h, 0.875h, 0.9375h,
    1.0h, 1.125h, 1.25h, 1.375h, 1.5h, 1.625h, 1.75h, 1.875h,
    2.0h, 2.25h, 2.5h, 2.75h, 3.0h, 3.25h, 3.5h, 3.75h,
    4.0h, 4.5h, 5.0h, 5.5h, 6.0h, 6.5h, 7.0h, 7.5h,
    8.0h, 9.0h, 10.0h, 11.0h, 12.0h, 13.0h, 14.0h, 15.0h,
    16.0h, 18.0h, 20.0h, 22.0h, 24.0h, 26.0h, 28.0h, 30.0h,
    32.0h, 36.0h, 40.0h, 44.0h, 48.0h, 52.0h, 56.0h, 60.0h,
    64.0h, 72.0h, 80.0h, 88.0h, 96.0h, 104.0h, 112.0h, 120.0h,
    128.0h, 144.0h, 160.0h, 176.0h, 192.0h, 208.0h, 224.0h, 240.0h,
};

// Bit-exact IEEE-754 half4 decode of 4 NVFP4 (2*E2M1) nibbles with zero LUT/memory lookup.
static inline half4 qw3_nvfp4_decode4_h(uint c0, uint c1, uint c2, uint c3) {
    return half4(qw3_nvfp4_e2m1_h[c0], qw3_nvfp4_e2m1_h[c1],
                 qw3_nvfp4_e2m1_h[c2], qw3_nvfp4_e2m1_h[c3]);
}

static inline float qw3_nvfp4_dot8_f32(device const uchar *wrow, uint b, uint sub,
                                       uint half_sub, float4 x_lo, float4 x_hi) {
    device const uchar *blk = wrow + ulong(b) * 36ull;
    const float sc = float(qw3_nvfp4_scale_h[uint(blk[sub])]);
    const uint word = *(device const uint *)(blk + 4u + 8u * sub + 4u * half_sub);
    const float4 w_lo = float4(
        qw3_nvfp4_e2m1_f[word & 15u],
        qw3_nvfp4_e2m1_f[(word >> 8u) & 15u],
        qw3_nvfp4_e2m1_f[(word >> 16u) & 15u],
        qw3_nvfp4_e2m1_f[(word >> 24u) & 15u]);
    const float4 w_hi = float4(
        qw3_nvfp4_e2m1_f[(word >> 4u) & 15u],
        qw3_nvfp4_e2m1_f[(word >> 12u) & 15u],
        qw3_nvfp4_e2m1_f[(word >> 20u) & 15u],
        qw3_nvfp4_e2m1_f[(word >> 28u) & 15u]);
    return sc * (dot(w_lo, x_lo) + dot(w_hi, x_hi));
}

static inline float qw3_nvfp4_dot4_f32(device const uchar *wrow, uint b, uint sub,
                                       uint q_pair, float2 x_lo, float2 x_hi) {
    device const uchar *blk = wrow + ulong(b) * 36ull;
    const float sc = float(qw3_nvfp4_scale_h[uint(blk[sub])]);
    const uint word = uint(*(device const ushort *)(blk + 4u + 8u * sub + 2u * q_pair));
    const float2 w_lo = float2(qw3_nvfp4_e2m1_f[word & 15u],
                               qw3_nvfp4_e2m1_f[(word >> 8u) & 15u]);
    const float2 w_hi = float2(qw3_nvfp4_e2m1_f[(word >> 4u) & 15u],
                               qw3_nvfp4_e2m1_f[(word >> 12u) & 15u]);
    return sc * (dot(w_lo, x_lo) + dot(w_hi, x_hi));
}

kernel void qw3_matvec_nvfp4(constant qw3_matvec_q8_0_args &args [[buffer(0)]],
                            device const uchar *weights [[buffer(1)]],
                            device const float *x [[buffer(2)]],
                            device float *out [[buffer(3)]],
                            uint group [[threadgroup_position_in_grid]],
                            uint simd_idx [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    const uint row = 4u * group + simd_idx;
    if (row >= args.n_out) return;
    device const uchar *w = weights + ulong(row) * args.row_bytes;
    float sum = 0.0f;
    const uint nb = args.n_in / 64u;
    const uint blk_lane = lane >> 3u;
    const uint sub = (lane >> 1u) & 3u;
    const uint half_sub = lane & 1u;
    for (uint b = blk_lane; b < nb; b += 4u) {
        const uint xb = b * 64u + sub * 16u + half_sub * 4u;
        const float4 x_lo = *(device const float4 *)(x + xb);
        const float4 x_hi = *(device const float4 *)(x + xb + 8u);
        sum += qw3_nvfp4_dot8_f32(w, b, sub, half_sub, x_lo, x_hi);
    }
    sum = simd_sum(sum);
    if (lane == 0u) out[row] = sum;
}

// Fast Inkling decode Q8_0 matvec: 4 lanes/block (8 blocks = 256 elems/iter), 2 rows/simdgroup.
// Fast Q8_0 matvec: 4 lanes/block, 2 rows/simdgroup (8 rows/TG), single 64-bit ushort4 load per 8 elements.
static inline float qw3_inkling_q8_dot8(device const uchar *blk, uint sub, float4 x0, float4 x1) {
    const float d = float(*(device const half *)blk);
    const ushort4 raw = *(device const ushort4 *)(blk + 2u + sub * 8u);
    const char4 c0 = as_type<char4>(raw.xy);
    const char4 c1 = as_type<char4>(raw.zw);
    return d * (dot(float4(c0), x0) + dot(float4(c1), x1));
}

kernel void qw3_inkling_matvec_q8_0(constant qw3_matvec_q8_0_args &args [[buffer(0)]],
                                    device const uchar *weights [[buffer(1)]],
                                    device const float *x [[buffer(2)]],
                                    device float *out [[buffer(3)]],
                                    uint group [[threadgroup_position_in_grid]],
                                    uint simd_idx [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    const uint first_row = (group * 4u + simd_idx) * 2u;
    if (first_row >= args.n_out) return;
    const bool has_r1 = (first_row + 1u) < args.n_out;
    device const uchar *wr0 = weights + ulong(first_row) * args.row_bytes;
    device const uchar *wr1 = has_r1 ? (wr0 + args.row_bytes) : wr0;
    const uint n_blocks = args.n_in / 32u;
    const uint blk_lane = lane >> 2u;
    const uint sub = lane & 3u;
    float sum0 = 0.0f, sum1 = 0.0f;
    for (uint b = blk_lane; b < n_blocks; b += 8u) {
        const uint x_off = b * 32u + sub * 8u;
        const float4 x0 = *(device const float4 *)(x + x_off);
        const float4 x1 = *(device const float4 *)(x + x_off + 4u);
        const ulong boff = ulong(b) * 34ull;
        sum0 += qw3_inkling_q8_dot8(wr0 + boff, sub, x0, x1);
        if (has_r1) sum1 += qw3_inkling_q8_dot8(wr1 + boff, sub, x0, x1);
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (lane == 0u) {
        out[first_row] = sum0;
        if (has_r1) out[first_row + 1u] = sum1;
    }
}

// Fused Q8_0 pair matvec (e.g. Q + gate sharing the same input vector x):
// 4 lanes/block (256 elems/iter), 2 rows/simdgroup per matrix -> reads x ONCE for both matrices.
struct qw3_inkling_q8_pair_args { uint n_in, n_out, row_bytes_a, row_bytes_b; };
kernel void qw3_inkling_matvec_q8_0_pair(constant qw3_inkling_q8_pair_args &args [[buffer(0)]],
                                         device const uchar *weights_a [[buffer(1)]],
                                         device const uchar *weights_b [[buffer(2)]],
                                         device const float *x [[buffer(3)]],
                                         device float *out_a [[buffer(4)]],
                                         device float *out_b [[buffer(5)]],
                                         uint group [[threadgroup_position_in_grid]],
                                         ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                         ushort lane [[thread_index_in_simdgroup]]) {
    const uint first_row = (group * 4u + uint(simd_idx)) * 2u;
    if (first_row >= args.n_out) return;
    const bool has_r1 = (first_row + 1u) < args.n_out;
    device const uchar *wa0 = weights_a + ulong(first_row) * args.row_bytes_a;
    device const uchar *wa1 = has_r1 ? (wa0 + args.row_bytes_a) : wa0;
    device const uchar *wb0 = weights_b + ulong(first_row) * args.row_bytes_b;
    device const uchar *wb1 = has_r1 ? (wb0 + args.row_bytes_b) : wb0;
    const uint n_blocks = args.n_in / 32u;
    const uint blk_lane = uint(lane) >> 2u;
    const uint sub = uint(lane) & 3u;
    float sa0 = 0.0f, sa1 = 0.0f, sb0 = 0.0f, sb1 = 0.0f;
    for (uint b = blk_lane; b < n_blocks; b += 8u) {
        const uint x_off = b * 32u + sub * 8u;
        const float4 x0 = *(device const float4 *)(x + x_off);
        const float4 x1 = *(device const float4 *)(x + x_off + 4u);
        const ulong boff = ulong(b) * 34ull;
        sa0 += qw3_inkling_q8_dot8(wa0 + boff, sub, x0, x1);
        sb0 += qw3_inkling_q8_dot8(wb0 + boff, sub, x0, x1);
        if (has_r1) {
            sa1 += qw3_inkling_q8_dot8(wa1 + boff, sub, x0, x1);
            sb1 += qw3_inkling_q8_dot8(wb1 + boff, sub, x0, x1);
        }
    }
    sa0 = simd_sum(sa0); sb0 = simd_sum(sb0);
    sa1 = simd_sum(sa1); sb1 = simd_sum(sb1);
    if (lane == 0u) {
        out_a[first_row] = sa0;
        out_b[first_row] = sb0;
        if (has_r1) {
            out_a[first_row + 1u] = sa1;
            out_b[first_row + 1u] = sb1;
        }
    }
}

// High-occupancy 1-row-per-threadgroup F32 matvec for small n_out (router 128, loop_down 64).
kernel void qw3_inkling_matvec_f32_small(constant qw3_matvec_q8_0_args &args [[buffer(0)]],
                                         device const float4 *weights [[buffer(1)]],
                                         device const float4 *x [[buffer(2)]],
                                         device float *out [[buffer(3)]],
                                         uint row [[threadgroup_position_in_grid]],
                                         ushort tid [[thread_index_in_threadgroup]],
                                         ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                         ushort lane [[thread_index_in_simdgroup]]) {
    if (row >= args.n_out) return;
    const uint n4 = args.n_in >> 2u;
    device const float4 *w = weights + ulong(row) * n4;
    float sum = 0.0f;
    for (uint i = tid; i < n4; i += 128u) sum += dot(w[i], x[i]);
    sum = simd_sum(sum);
    threadgroup float sh[4];
    if (lane == 0u) sh[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_idx == 0u) {
        float tot = lane < 4u ? sh[lane] : 0.0f;
        tot = simd_sum(tot);
        if (lane == 0u) out[row] = tot;
    }
}

// High-occupancy 1-row-per-threadgroup BF16/F16 matvec for small n_out (sink 24, taps 32).
kernel void qw3_inkling_matvec_float16_small(constant qw3_matvec_q8_0_args &args [[buffer(0)]],
                                             device const ushort4 *weights [[buffer(1)]],
                                             device const float4 *x [[buffer(2)]],
                                             device float *out [[buffer(3)]],
                                             constant uint &type [[buffer(4)]],
                                             uint row [[threadgroup_position_in_grid]],
                                             ushort tid [[thread_index_in_threadgroup]],
                                             ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                             ushort lane [[thread_index_in_simdgroup]]) {
    if (row >= args.n_out) return;
    const uint n4 = args.n_in >> 2u;
    device const ushort4 *w = weights + ulong(row) * n4;
    float sum = 0.0f;
    if (type == 30u) {
        for (uint i = tid; i < n4; i += 128u) {
            const float4 wf = as_type<float4>(uint4(w[i]) << 16u);
            sum += dot(wf, x[i]);
        }
    } else {
        for (uint i = tid; i < n4; i += 128u) {
            const float4 wf = float4(as_type<half4>(w[i]));
            sum += dot(wf, x[i]);
        }
    }
    sum = simd_sum(sum);
    threadgroup float sh[4];
    if (lane == 0u) sh[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_idx == 0u) {
        float tot = lane < 4u ? sh[lane] : 0.0f;
        tot = simd_sum(tot);
        if (lane == 0u) out[row] = tot;
    }
}

kernel void qw3_matvec_float16(constant qw3_matvec_q8_0_args &args [[buffer(0)]],
                                 device const ushort *weights [[buffer(1)]],
                                 device const float *x [[buffer(2)]],
                                 device float *out [[buffer(3)]],
                                 constant uint &type [[buffer(4)]],
                                 uint group [[threadgroup_position_in_grid]],
                                 uint simd_idx [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint row = 4u * group + simd_idx;
    if (row >= args.n_out) return;
    device const ushort *w = weights + ulong(row) * args.n_in;
    float sum = 0.0f;
    for (uint i = lane; i < args.n_in; i += 32u) {
        const float value = type == 30u ? as_type<float>(uint(w[i]) << 16u) :
                                         float(as_type<half>(w[i]));
        sum += value * x[i];
    }
    sum = simd_sum(sum);
    if (lane == 0u) out[row] = sum;
}

// Fused plain RMSNorm of (state, message) into contiguous [2*D] scratch in 1 dispatch.
kernel void qw3_inkling_rmsnorm_pair(constant uint &width [[buffer(0)]],
                                     device const float4 *a [[buffer(1)]],
                                     device const float4 *b [[buffer(2)]],
                                     device float4 *out [[buffer(3)]],
                                     uint which [[threadgroup_position_in_grid]],
                                     ushort tid [[thread_index_in_threadgroup]],
                                     ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                     ushort lane [[thread_index_in_simdgroup]]) {
    const uint n4 = width >> 2u;
    device const float4 *src = which == 0u ? a : b;
    device float4 *dst = out + which * n4;
    float ss = 0.0f;
    for (uint i = tid; i < n4; i += 128u) {
        const float4 v = src[i];
        ss += dot(v, v);
    }
    ss = simd_sum(ss);
    threadgroup float sh[4];
    if (lane == 0u) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_idx == 0u) {
        float tot = lane < 4u ? sh[lane] : 0.0f;
        tot = simd_sum(tot);
        if (lane == 0u) sh[0] = rsqrt(tot / float(width) + 1.0e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = sh[0];
    for (uint i = tid; i < n4; i += 128u) dst[i] = src[i] * scale;
}

// Vectorized low-rank loop gate apply: 16 rows/TG (4 rows/simd, 8 lanes/row), float4 loads,
// separate src_state and dst_state to avoid copies on sliding repeat layers.
kernel void qw3_inkling_control_apply_fast(constant uint2 &shape [[buffer(0)]],
                                           device const float4 *gain [[buffer(1)]],
                                           device const float4 *forget [[buffer(2)]],
                                           device const float4 *code [[buffer(3)]],
                                           device const float *src_state [[buffer(4)]],
                                           device const float *message [[buffer(5)]],
                                           device float *dst_state [[buffer(6)]],
                                           uint group [[threadgroup_position_in_grid]],
                                           ushort tid [[thread_index_in_threadgroup]],
                                           ushort simd [[simdgroup_index_in_threadgroup]],
                                           ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float4 c_sh[16];
    if (tid < 16u) c_sh[tid] = code[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint sub_row = uint(lane) >> 3u;
    const uint k_lane = uint(lane) & 7u;
    const uint row = group * 16u + uint(simd) * 4u + sub_row;
    if (row >= shape.x) return;
    const uint base = row * 16u;
    float g = dot(gain[base + k_lane], c_sh[k_lane]) +
              dot(gain[base + k_lane + 8u], c_sh[k_lane + 8u]);
    float f = dot(forget[base + k_lane], c_sh[k_lane]) +
              dot(forget[base + k_lane + 8u], c_sh[k_lane + 8u]);
    g += simd_shuffle_xor(g, 4u);
    g += simd_shuffle_xor(g, 2u);
    g += simd_shuffle_xor(g, 1u);
    f += simd_shuffle_xor(f, 4u);
    f += simd_shuffle_xor(f, 2u);
    f += simd_shuffle_xor(f, 1u);
    if (k_lane == 0u) {
        dst_state[row] = (1.0f - tanh(f)) * src_state[row] + tanh(g) * message[row];
    }
}

// Apply both low-rank loop gates using the code produced by the preceding matvec.
kernel void qw3_inkling_control_apply(constant uint2 &shape [[buffer(0)]],
                                      device const float *gain [[buffer(1)]],
                                      device const float *forget [[buffer(2)]],
                                      device const float *code [[buffer(3)]],
                                      device float *state [[buffer(4)]],
                                      device const float *message [[buffer(5)]],
                                      uint group [[threadgroup_position_in_grid]],
                                      uint simd [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    const uint row = 4u * group + simd;
    const uint width = shape.x, rank = shape.y;
    if (row >= width) return;
    const uint offset = row * rank;
    float g = 0.0f, f = 0.0f;
    for (uint i = lane; i < rank; i += 32u) {
        g += gain[offset + i] * code[i];
        f += forget[offset + i] * code[i];
    }
    g = simd_sum(g);
    f = simd_sum(f);
    if (lane == 0u)
        state[row] = (1.0f - tanh(f)) * state[row] + tanh(g) * message[row];
}

// Zero-sync GPU-indexed MoE decode kernels for Inkling 2 Mini.
struct qw3_inkling_moe_dec_args {
    uint n_in, n_out, row_bytes, expert_bytes;
    float beta, limit;
};

// Dedicated NVFP4 fused gate_up + SiTU activation: 1 (gate,up) pair/simdgroup = 4 pairs/TG
// (higher occupancy, low register pressure), 8 lanes/block -> exact 9 iters for n_in=2304.
kernel void qw3_inkling_moe_gate_up_situ_nvfp4(
    constant qw3_inkling_moe_dec_args &args [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device const uint *route_ids [[buffer(3)]],
    device const float *route_coeffs [[buffer(4)]],
    device float *hidden_out [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort simd_idx [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint n_ff = args.n_out;
    const uint slot = group.y;
    const uint r0 = group.x * 4u + uint(simd_idx);
    if (r0 >= n_ff) return;
    const uint expert = route_ids[slot];
    device const uchar *wexp = weights + ulong(expert) * ulong(args.expert_bytes);
    device const uchar *wg0 = wexp + ulong(r0) * args.row_bytes;
    device const uchar *wu0 = wexp + ulong(r0 + n_ff) * args.row_bytes;
    const uint nb = args.n_in / 64u;
    const uint blk_lane = uint(lane) >> 3u;
    const uint sub = (uint(lane) >> 1u) & 3u;
    const uint half_sub = uint(lane) & 1u;
    float g0 = 0.0f, u0 = 0.0f;
    for (uint b = blk_lane; b < nb; b += 4u) {
        const uint xb = b * 64u + sub * 16u + half_sub * 4u;
        const float4 x_lo = *(device const float4 *)(x + xb);
        const float4 x_hi = *(device const float4 *)(x + xb + 8u);
        g0 += qw3_nvfp4_dot8_f32(wg0, b, sub, half_sub, x_lo, x_hi);
        u0 += qw3_nvfp4_dot8_f32(wu0, b, sub, half_sub, x_lo, x_hi);
    }
    g0 = simd_sum(g0);
    u0 = simd_sum(u0);
    if (lane == 0u) {
        const float sc = route_coeffs[slot];
        const float ga = g0 * sc, up = u0 * sc;
        const float beta = args.beta, limit = args.limit;
        hidden_out[ulong(slot) * n_ff + r0] =
            beta * tanh(ga / beta) / (1.0f + exp(-ga)) * limit * tanh(up / limit);
    }
}

// Dedicated NVFP4 indexed down projection: 2 rows/simdgroup = 8 rows/TG,
// 8 lanes/block -> uint 4-byte loads + float4 x loads (only 5 iters for n_in=1152!).
kernel void qw3_inkling_moe_down_nvfp4(
    constant qw3_inkling_moe_dec_args &args [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const float *hidden_in [[buffer(2)]],
    device const uint *route_ids [[buffer(3)]],
    device float *down_out [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort simd_idx [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint slot = group.y;
    const uint r0 = (group.x * 4u + uint(simd_idx)) * 2u;
    if (r0 >= args.n_out) return;
    const bool h1 = (r0 + 1u) < args.n_out;
    const uint expert = route_ids[slot];
    device const uchar *wexp = weights + ulong(expert) * ulong(args.expert_bytes);
    device const float *xs = hidden_in + ulong(slot) * args.n_in;
    const uint nb = args.n_in / 64u;
    const uint blk_lane = uint(lane) >> 3u;
    const uint sub = (uint(lane) >> 1u) & 3u;
    const uint half_sub = uint(lane) & 1u;
    float s0 = 0.0f, s1 = 0.0f;
    device const uchar *wr0 = wexp + ulong(r0) * args.row_bytes;
    device const uchar *wr1 = h1 ? (wr0 + args.row_bytes) : wr0;
    for (uint b = blk_lane; b < nb; b += 4u) {
        const uint xb = b * 64u + sub * 16u + half_sub * 4u;
        const float4 x_lo = *(device const float4 *)(xs + xb);
        const float4 x_hi = *(device const float4 *)(xs + xb + 8u);
        s0 += qw3_nvfp4_dot8_f32(wr0, b, sub, half_sub, x_lo, x_hi);
        if (h1) s1 += qw3_nvfp4_dot8_f32(wr1, b, sub, half_sub, x_lo, x_hi);
    }
    s0 = simd_sum(s0);
    s1 = simd_sum(s1);
    if (lane == 0u) {
        device float *dst = down_out + ulong(slot) * args.n_out;
        dst[r0] = s0;
        if (h1) dst[r0 + 1u] = s1;
    }
}

// Fast 32-lane-per-superblock (8 weights/lane, exact 9 iters for n_in=2304, zero idle lanes)
// FP32 dot helpers for IQ3_XXS, IQ2_S, IQ3_S, IQ2_XS decode gate_up_situ!
static inline float qw3_iq3xxs_dot8_f32(device const uchar *blk, uint ib32, uint q8,
        float4 x0, float4 x1, device const ushort *grid16, device const uchar *signs) {
    const float d = float(*(device const half *)blk);
    device const uchar *q3 = blk + 2u + 8u * ib32;
    device const uchar *sp = blk + 66u + 4u * ib32;
    const uint packed = uint(*(device const ushort *)sp) | (uint(*(device const ushort *)(sp + 2u)) << 16u);
    const float db = d * (0.5f + float(packed >> 28u)) * 0.5f;
    const uint sg = uint(signs[(packed >> (7u * q8)) & 127u]);
    const ushort g0 = grid16[uint(q3[2u * q8])];
    const ushort g1 = grid16[uint(q3[2u * q8 + 1u])];
    const float4 v0 = float4(
        qw3_iq3xxs_grid_value(g0, 0u) * ((sg & 1u) ? -1.0f : 1.0f),
        qw3_iq3xxs_grid_value(g0, 1u) * ((sg & 2u) ? -1.0f : 1.0f),
        qw3_iq3xxs_grid_value(g0, 2u) * ((sg & 4u) ? -1.0f : 1.0f),
        qw3_iq3xxs_grid_value(g0, 3u) * ((sg & 8u) ? -1.0f : 1.0f));
    const float4 v1 = float4(
        qw3_iq3xxs_grid_value(g1, 0u) * ((sg & 16u) ? -1.0f : 1.0f),
        qw3_iq3xxs_grid_value(g1, 1u) * ((sg & 32u) ? -1.0f : 1.0f),
        qw3_iq3xxs_grid_value(g1, 2u) * ((sg & 64u) ? -1.0f : 1.0f),
        qw3_iq3xxs_grid_value(g1, 3u) * ((sg & 128u) ? -1.0f : 1.0f));
    return db * (dot(v0, x0) + dot(v1, x1));
}

static inline float qw3_iq2s_dot8_f32(device const uchar *blk, uint ib32, uint q8,
        float4 x0, float4 x1, device const ushort *grid16) {
    const float d = float(*(device const half *)blk);
    device const uchar *qs = blk + 2u;
    const uint half16 = q8 >> 1u;
    const float db = d * (0.5f + float((uint(blk[74u + ib32]) >> (4u * half16)) & 15u)) * 0.25f;
    const uint qh_bits = (uint(blk[66u + ib32]) >> (2u * q8)) & 3u;
    const uint off = 4u * ib32 + q8;
    const ushort g = grid16[uint(qs[off]) | (qh_bits << 8u)];
    const uint sg = uint(qs[32u + off]);
    const float4 v0 = float4(
        qw3_iq2_grid_value(g, 0u) * ((sg & 1u) ? -1.0f : 1.0f),
        qw3_iq2_grid_value(g, 1u) * ((sg & 2u) ? -1.0f : 1.0f),
        qw3_iq2_grid_value(g, 2u) * ((sg & 4u) ? -1.0f : 1.0f),
        qw3_iq2_grid_value(g, 3u) * ((sg & 8u) ? -1.0f : 1.0f));
    const float4 v1 = float4(
        qw3_iq2_grid_value(g, 4u) * ((sg & 16u) ? -1.0f : 1.0f),
        qw3_iq2_grid_value(g, 5u) * ((sg & 32u) ? -1.0f : 1.0f),
        qw3_iq2_grid_value(g, 6u) * ((sg & 64u) ? -1.0f : 1.0f),
        qw3_iq2_grid_value(g, 7u) * ((sg & 128u) ? -1.0f : 1.0f));
    return db * (dot(v0, x0) + dot(v1, x1));
}

static inline float qw3_iq3s_dot8_f32(device const uchar *blk, uint ib32, uint q8,
        float4 x0, float4 x1, device const uchar *iq3s_kgrid) {
    const float d = float(*(device const half *)blk);
    const float dl = d * float(1u + 2u * ((uint(blk[106u + (ib32 >> 1u)]) >> (4u * (ib32 & 1u))) & 15u));
    const uint qh = uint(blk[66u + ib32]);
    device const uchar *q2 = blk + 2u + 8u * ib32 + 2u * q8;
    const uint idx0 = uint(q2[0]) | (((qh >> (2u * q8)) & 1u) << 8u);
    const uint idx1 = uint(q2[1]) | (((qh >> (2u * q8 + 1u)) & 1u) << 8u);
    device const uchar *g0 = iq3s_kgrid + idx0 * 4u;
    device const uchar *g1 = iq3s_kgrid + idx1 * 4u;
    const uint sb = uint(blk[74u + 4u * ib32 + q8]);
    const float4 v0 = float4(
        float(g0[0]) * ((sb & 1u) ? -1.0f : 1.0f),
        float(g0[1]) * ((sb & 2u) ? -1.0f : 1.0f),
        float(g0[2]) * ((sb & 4u) ? -1.0f : 1.0f),
        float(g0[3]) * ((sb & 8u) ? -1.0f : 1.0f));
    const float4 v1 = float4(
        float(g1[0]) * ((sb & 16u) ? -1.0f : 1.0f),
        float(g1[1]) * ((sb & 32u) ? -1.0f : 1.0f),
        float(g1[2]) * ((sb & 64u) ? -1.0f : 1.0f),
        float(g1[3]) * ((sb & 128u) ? -1.0f : 1.0f));
    return dl * (dot(v0, x0) + dot(v1, x1));
}

static inline float qw3_iq2xs_dot8_f32(device const uchar *blk, uint ib32, uint q8,
        float4 x0, float4 x1, device const ushort *grid16, device const uchar *signs) {
    const float d = float(*(device const half *)blk);
    const uint half16 = q8 >> 1u;
    const float db = d * (0.5f + float((uint(blk[66u + ib32]) >> (4u * half16)) & 15u)) * 0.25f;
    const uint q16 = uint(*(device const ushort *)(blk + 2u + 8u * ib32 + 2u * q8));
    const ushort g = grid16[q16 & 511u];
    const uint sg = uint(signs[q16 >> 9u]);
    const float4 v0 = float4(
        qw3_iq2_grid_value(g, 0u) * ((sg & 1u) ? -1.0f : 1.0f),
        qw3_iq2_grid_value(g, 1u) * ((sg & 2u) ? -1.0f : 1.0f),
        qw3_iq2_grid_value(g, 2u) * ((sg & 4u) ? -1.0f : 1.0f),
        qw3_iq2_grid_value(g, 3u) * ((sg & 8u) ? -1.0f : 1.0f));
    const float4 v1 = float4(
        qw3_iq2_grid_value(g, 4u) * ((sg & 16u) ? -1.0f : 1.0f),
        qw3_iq2_grid_value(g, 5u) * ((sg & 32u) ? -1.0f : 1.0f),
        qw3_iq2_grid_value(g, 6u) * ((sg & 64u) ? -1.0f : 1.0f),
        qw3_iq2_grid_value(g, 7u) * ((sg & 128u) ? -1.0f : 1.0f));
    return db * (dot(v0, x0) + dot(v1, x1));
}

static inline float qw3_q3k_dot8_f32(device const uchar *blk, uint ib32, uint q8,
        float4 x0, float4 x1) {
    device const uchar *packed = blk + 96u;
    const uint aux0 = uint(*(device const ushort *)packed) | (uint(*(device const ushort *)(packed + 2u)) << 16u);
    const uint aux1 = uint(*(device const ushort *)(packed + 4u)) | (uint(*(device const ushort *)(packed + 6u)) << 16u);
    const uint tmp  = uint(*(device const ushort *)(packed + 8u)) | (uint(*(device const ushort *)(packed + 10u)) << 16u);
    const uint kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    const uint half_b = ib32 >> 2u;
    const uint within = (ib32 & 3u) * 32u + q8 * 8u;
    const uint grp = within >> 5u;
    const uint upper = (within & 31u) >> 4u;
    const uint scale_index = half_b * 8u + grp * 2u + upper;
    const uint ax = (scale_index >> 2u) == 0u ? ((aux0 & kmask2) | (((tmp >> 0u) & kmask1) << 4u)) :
                    (scale_index >> 2u) == 1u ? ((aux1 & kmask2) | (((tmp >> 2u) & kmask1) << 4u)) :
                    (scale_index >> 2u) == 2u ? (((aux0 >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u)) :
                                                (((aux1 >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u));
    const float d = float(*(device const half *)(blk + 108u));
    const uint scale = (ax >> (8u * (scale_index & 3u))) & 255u;
    const float dl = d * float(int(scale) - 32);
    const uint shift = grp * 2u;
    const uint mask = 1u << (half_b * 4u + grp);
    const uint qbase = upper * 16u + (q8 & 1u) * 8u;
    const uint q_index = half_b * 32u + qbase;
    const uint2 qraw = uint2(*(device const uint *)(blk + 32u + q_index),
                             *(device const uint *)(blk + 32u + q_index + 4u));
    const uint2 hraw = uint2(*(device const uint *)(blk + qbase),
                             *(device const uint *)(blk + qbase + 4u));
    const uchar4 ql0 = as_type<uchar4>(qraw.x);
    const uchar4 ql1 = as_type<uchar4>(qraw.y);
    const uchar4 qh0 = as_type<uchar4>(hraw.x);
    const uchar4 qh1 = as_type<uchar4>(hraw.y);
    const float4 v0 = float4(
        float(int((ql0.x >> shift) & 3u) - ((qh0.x & mask) ? 0 : 4)),
        float(int((ql0.y >> shift) & 3u) - ((qh0.y & mask) ? 0 : 4)),
        float(int((ql0.z >> shift) & 3u) - ((qh0.z & mask) ? 0 : 4)),
        float(int((ql0.w >> shift) & 3u) - ((qh0.w & mask) ? 0 : 4)));
    const float4 v1 = float4(
        float(int((ql1.x >> shift) & 3u) - ((qh1.x & mask) ? 0 : 4)),
        float(int((ql1.y >> shift) & 3u) - ((qh1.y & mask) ? 0 : 4)),
        float(int((ql1.z >> shift) & 3u) - ((qh1.z & mask) ? 0 : 4)),
        float(int((ql1.w >> shift) & 3u) - ((qh1.w & mask) ? 0 : 4)));
    return dl * (dot(v0, x0) + dot(v1, x1));
}

static inline float qw3_iq4nl_dot4_f32(device const uchar *blk, uint lane_in_blk,
        float4 x) {
    const float d = float(*(device const half *)blk);
    const uint qs_off = (lane_in_blk & 3u) * 4u;
    const uint packed = uint(*(device const ushort *)(blk + 2u + qs_off)) |
                        (uint(*(device const ushort *)(blk + 4u + qs_off)) << 16u);
    const bool hi = (lane_in_blk & 4u) != 0u;
    const uint s = hi ? (packed >> 4u) : packed;
    const float4 w = float4(
        qw3_iq4nl_table[s & 15u],
        qw3_iq4nl_table[(s >> 8u) & 15u],
        qw3_iq4nl_table[(s >> 16u) & 15u],
        qw3_iq4nl_table[(s >> 24u) & 15u]);
    return d * dot(w, x);
}

// Compile-time specialized FP32 16-element dot product for all non-NVFP4 Inkling expert types.
template<uint TYPE>
static inline float qw3_inkling_dot16_f32(
    device const uchar *wrow, uint k0, device const float *x,
    device const ushort *grid16, device const uchar *signs,
    device const uchar *iq3s_kgrid) {
    device const float *xx = x + k0;
    if (TYPE == 20u) {
        device const uchar *blk = wrow + uint64_t(k0 >> 5u) * 18ull;
        const float d = float(*((device const half *)blk));
        device const uchar *qs = blk + 2u;
        const bool hi = (k0 & 16u) != 0u;
        float acc = 0.0f;
        for (uint i = 0; i < 16u; i++) {
            const uint v = uint(qs[i]);
            acc += qw3_iq4nl_table[hi ? (v >> 4u) : (v & 15u)] * xx[i];
        }
        return d * acc;
    } else if (TYPE == 11u) {
        device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 110ull;
        device const uchar *packed = blk + 96u;
        const uint aux0 = uint(packed[0]) | (uint(packed[1]) << 8u) |
                          (uint(packed[2]) << 16u) | (uint(packed[3]) << 24u);
        const uint aux1 = uint(packed[4]) | (uint(packed[5]) << 8u) |
                          (uint(packed[6]) << 16u) | (uint(packed[7]) << 24u);
        const uint tmp = uint(packed[8]) | (uint(packed[9]) << 8u) |
                         (uint(packed[10]) << 16u) | (uint(packed[11]) << 24u);
        const uint kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
        uint aux[4];
        aux[2] = ((aux0 >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u);
        aux[3] = ((aux1 >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u);
        aux[0] = (aux0 & kmask2) | (((tmp >> 0u) & kmask1) << 4u);
        aux[1] = (aux1 & kmask2) | (((tmp >> 2u) & kmask1) << 4u);
        const float d = float(*((device const half *)(blk + 108u)));
        const uint local0 = k0 & 255u;
        const uint half_block = local0 >> 7u;
        const uint within = local0 & 127u;
        const uint grp = within >> 5u;
        const uint upper = (within & 31u) >= 16u ? 1u : 0u;
        const uint scale_index = half_block * 8u + grp * 2u + upper;
        const uint scale = (aux[scale_index >> 2u] >> (8u * (scale_index & 3u))) & 255u;
        const float dl = d * float(int(scale) - 32);
        const uint shift = grp * 2u;
        const uint mask = 1u << (half_block * 4u + grp);
        float acc = 0.0f;
        for (uint l = 0; l < 16u; l++) {
            const uint q_index = half_block * 32u + upper * 16u + l;
            const int low = int((uint(blk[32u + q_index]) >> shift) & 3u);
            const int q = low - ((uint(blk[upper * 16u + l]) & mask) ? 0 : 4);
            acc += float(q) * xx[l];
        }
        return dl * acc;
    } else if (TYPE == 17u) {
        device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 74ull;
        const float d = qw3_f16_to_f32(ushort(blk[0]) | (ushort(blk[1]) << 8u));
        device const uchar *qs = blk + 2u;
        device const uchar *scales = qs + 64u;
        const uint il = (k0 & 255u) >> 4u;
        const uint ib32 = il >> 1u;
        const uint half16 = il & 1u;
        const uint off = 2u * (4u * ib32 + 2u * half16);
        const uint q0 = uint(qs[off]) | (uint(qs[off + 1u]) << 8u);
        const uint q1 = uint(qs[off + 2u]) | (uint(qs[off + 3u]) << 8u);
        const float db = d * (0.5f + float((uint(scales[ib32]) >> (4u * half16)) & 15u)) * 0.25f;
        const ushort grid0 = grid16[q0 & 511u];
        const ushort grid1 = grid16[q1 & 511u];
        const uint sign0 = uint(signs[q0 >> 9u]);
        const uint sign1 = uint(signs[q1 >> 9u]);
        float acc = 0.0f;
        for (uint i = 0; i < 8u; i++) {
            const float s0 = (sign0 & (1u << i)) ? -1.0f : 1.0f;
            const float s1 = (sign1 & (1u << i)) ? -1.0f : 1.0f;
            acc += qw3_iq2_grid_value(grid0, i) * s0 * xx[i] +
                   qw3_iq2_grid_value(grid1, i) * s1 * xx[8u + i];
        }
        return db * acc;
    } else if (TYPE == 18u) {
        device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 98ull;
        const float d = qw3_f16_to_f32(ushort(blk[0]) | (ushort(blk[1]) << 8u));
        device const uchar *qs = blk + 2u;
        device const uchar *ss = qs + 64u;
        const uint il = (k0 & 255u) >> 4u;
        const uint ib32 = il >> 1u;
        const uint half16 = il & 1u;
        device const uchar *q3 = qs + 8u * ib32;
        device const uchar *sp = ss + 4u * ib32;
        const uint packed = uint(sp[0]) | (uint(sp[1]) << 8u) |
                            (uint(sp[2]) << 16u) | (uint(sp[3]) << 24u);
        const float db = d * (0.5f + float(packed >> 28u)) * 0.5f;
        const uint sign0 = uint(signs[(packed >> (14u * half16)) & 127u]);
        const uint sign1 = uint(signs[(packed >> (14u * half16 + 7u)) & 127u]);
        const ushort g0 = grid16[uint(q3[4u * half16])];
        const ushort g1 = grid16[uint(q3[4u * half16 + 1u])];
        const ushort g2 = grid16[uint(q3[4u * half16 + 2u])];
        const ushort g3 = grid16[uint(q3[4u * half16 + 3u])];
        float acc = 0.0f;
        for (uint i = 0; i < 4u; i++) {
            const float s00 = (sign0 & (1u << i)) ? -1.0f : 1.0f;
            const float s01 = (sign0 & (1u << (i + 4u))) ? -1.0f : 1.0f;
            const float s10 = (sign1 & (1u << i)) ? -1.0f : 1.0f;
            const float s11 = (sign1 & (1u << (i + 4u))) ? -1.0f : 1.0f;
            acc += qw3_iq3xxs_grid_value(g0, i) * s00 * xx[i] +
                   qw3_iq3xxs_grid_value(g1, i) * s01 * xx[4u + i] +
                   qw3_iq3xxs_grid_value(g2, i) * s10 * xx[8u + i] +
                   qw3_iq3xxs_grid_value(g3, i) * s11 * xx[12u + i];
        }
        return db * acc;
    } else if (TYPE == 21u) {
        device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 110ull;
        const uint il = (k0 & 255u) >> 4u;
        const uint ib32 = il >> 1u;
        const uint half_il = il & 1u;
        const float d = float(*((device const half *)blk));
        device const uchar *qs = blk + 2u;
        device const uchar *qh = qs + 64u;
        device const uchar *sbuf = qh + 8u;
        device const uchar *scales = sbuf + 32u;
        const float dl = d * float(1u + 2u * ((uint(scales[ib32 >> 1u]) >> (4u * (ib32 & 1u))) & 15u));
        device const uchar *qs16 = qs + 8u * ib32 + 4u * half_il;
        device const uchar *signs16 = sbuf + 4u * ib32 + 2u * half_il;
        const uchar qh16 = uchar(uint(qh[ib32]) >> (4u * half_il));
        float acc = 0.0f;
        for (uint qn = 0; qn < 4u; qn++) {
            const uint idx = uint(qs16[qn]) | ((uint(qh16) << (8u - qn)) & 256u);
            device const uchar *g = iq3s_kgrid + (idx & 511u) * 4u;
            const uchar sb = signs16[qn >> 1u];
            const uint sshift = 4u * (qn & 1u);
            for (uint j = 0; j < 4u; j++) {
                const float sg = (uint(sb) & (1u << (j + sshift))) ? -1.0f : 1.0f;
                acc += float(g[j]) * sg * xx[qn * 4u + j];
            }
        }
        return dl * acc;
    } else if (TYPE == 22u) {
        device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 82ull;
        const float d = qw3_f16_to_f32(ushort(blk[0]) | (ushort(blk[1]) << 8u));
        device const uchar *qs = blk + 2u;
        device const uchar *qh = qs + 64u;
        device const uchar *scales = qh + 8u;
        const uint il = (k0 & 255u) >> 4u;
        const uint ib32 = il >> 1u;
        const uint half16 = il & 1u;
        const float db = d * (0.5f + float((uint(scales[ib32]) >> (4u * half16)) & 15u)) * 0.25f;
        const uint qh_byte = uint(qh[ib32]) >> (4u * half16);
        device const uchar *q2 = qs + 4u * ib32 + 2u * half16;
        device const uchar *s2 = q2 + 32u;
        const uint idx0 = uint(q2[0]) | ((qh_byte & 3u) << 8u);
        const uint idx1 = uint(q2[1]) | (((qh_byte >> 2u) & 3u) << 8u);
        const ushort g0 = grid16[idx0];
        const ushort g1 = grid16[idx1];
        const uint sg0 = uint(s2[0]);
        const uint sg1 = uint(s2[1]);
        float acc = 0.0f;
        for (uint i = 0; i < 8u; i++) {
            acc += qw3_iq2_grid_value(g0, i) * ((sg0 & (1u << i)) ? -1.0f : 1.0f) * xx[i] +
                   qw3_iq2_grid_value(g1, i) * ((sg1 & (1u << i)) ? -1.0f : 1.0f) * xx[8u + i];
        }
        return db * acc;
    } else if (TYPE == 12u || TYPE == 13u) {
        const bool is_q5 = (TYPE == 13u);
        device const uchar *blk = wrow + uint64_t(k0 >> 8u) * (is_q5 ? 176ull : 144ull);
        const float d = qw3_f16_to_f32(ushort(blk[0]) | (ushort(blk[1]) << 8u));
        const float dmin = qw3_f16_to_f32(ushort(blk[2]) | (ushort(blk[3]) << 8u));
        device const uchar *scales = blk + 4u;
        device const uchar *qh = blk + 16u;
        device const uchar *ql = blk + (is_q5 ? 48u : 16u);
        const uint g = (k0 & 255u) >> 5u;
        const uint l0 = k0 & 16u;
        uint sc, mn;
        if (g < 4u) {
            sc = uint(scales[g]) & 63u;
            mn = uint(scales[g + 4u]) & 63u;
        } else {
            sc = (uint(scales[g + 4u]) & 15u) | ((uint(scales[g - 4u]) >> 6u) << 4u);
            mn = (uint(scales[g + 4u]) >> 4u) | ((uint(scales[g]) >> 6u) << 4u);
        }
        const float dl = d * float(sc), ml = dmin * float(mn);
        device const uchar *qgroup = ql + (g >> 1u) * 32u + l0;
        const uint high_mask = 1u << g;
        const bool high_nibble = (g & 1u) != 0u;
        float acc = 0.0f, sumx = 0.0f;
        for (uint i = 0; i < 16u; i++) {
            const uint packed = uint(qgroup[i]);
            uint q = high_nibble ? (packed >> 4u) : (packed & 15u);
            if (is_q5 && (uint(qh[l0 + i]) & high_mask)) q += 16u;
            acc += float(q) * xx[i];
            sumx += xx[i];
        }
        return dl * acc - ml * sumx;
    } else if (TYPE == 14u) {
        device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 210ull;
        const float d = qw3_f16_to_f32(ushort(blk[208]) | (ushort(blk[209]) << 8u));
        float acc = 0.0f;
        for (uint i = 0; i < 16u; i++) {
            const uint local = (k0 + i) & 255u;
            const uint half_block = local >> 7u;
            const uint within = local & 127u;
            const uint part = within >> 5u;
            const uint l = within & 31u;
            const uint ql_index = half_block * 64u + (part & 1u) * 32u + l;
            const uint packed = uint(blk[ql_index]);
            const uint low = part < 2u ? (packed & 15u) : (packed >> 4u);
            const uint high = (uint(blk[128u + half_block * 32u + l]) >> (2u * part)) & 3u;
            const int q = int(low | (high << 4u)) - 32;
            const int scale = int(((device const char *)(blk + 192u))[half_block * 8u + 2u * part + l / 16u]);
            acc += float(scale * q) * xx[i];
        }
        return d * acc;
    } else if (TYPE == 8u) {
        device const uchar *blk = wrow + uint64_t(k0 >> 5u) * 34ull;
        const float d = float(*((device const half *)blk));
        device const char *qs = (device const char *)(blk + 2u) + (k0 & 16u);
        float acc = 0.0f;
        for (uint i = 0; i < 16u; i++) acc += float(qs[i]) * xx[i];
        return d * acc;
    } else if (TYPE == 23u) {
        device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 136ull;
        const float d = float(*((device const half *)blk));
        const ushort scales_h = *((device const ushort *)(blk + 2u));
        device const uchar *scales_l = blk + 4u;
        device const uchar *qs = blk + 8u;
        const uint ib = (k0 & 255u) >> 5u;
        const uint ls = ((uint(scales_l[ib >> 1u]) >> (4u * (ib & 1u))) & 15u) |
                        (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);
        const float dl = d * float(int(ls) - 32);
        device const uchar *q = qs + ib * 16u;
        const bool hi = (k0 & 16u) != 0u;
        float acc = 0.0f;
        for (uint i = 0; i < 16u; i++) {
            const uint v = uint(q[i]);
            acc += qw3_iq4nl_table[hi ? (v >> 4u) : (v & 15u)] * xx[i];
        }
        return dl * acc;
    }
    return 0.0f;
}

template<uint TYPE>
kernel void qw3_inkling_moe_gate_up_situ_id(
    constant qw3_inkling_moe_dec_args &args [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device const uint *route_ids [[buffer(3)]],
    device const float *route_coeffs [[buffer(4)]],
    device float *hidden_out [[buffer(5)]],
    device const ushort *grid16 [[buffer(6)]],
    device const uchar *signs [[buffer(7)]],
    device const uchar *iq3s_kgrid [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort simd_idx [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint n_ff = args.n_out;
    const uint slot = group.y;
    const uint row = group.x * 4u + uint(simd_idx);
    if (row >= n_ff) return;
    const uint expert = route_ids[slot];
    device const uchar *wexp = weights + ulong(expert) * ulong(args.expert_bytes);
    device const uchar *wg = wexp + ulong(row) * args.row_bytes;
    device const uchar *wu = wexp + ulong(row + n_ff) * args.row_bytes;
    float g = 0.0f, u = 0.0f;
    if (TYPE == 18u || TYPE == 22u || TYPE == 21u || TYPE == 17u || TYPE == 11u) {
        // Fast 32-lane-per-superblock path: 8 weights/lane, exact 9 iters for n_in=2304, zero idle lanes!
        const uint nb = args.n_in >> 8u;
        const uint ib32 = uint(lane) >> 2u;
        const uint q8 = uint(lane) & 3u;
        const uint x_lane_off = ib32 * 32u + q8 * 8u;
        const ulong bsize = (TYPE == 18u) ? 98ull : ((TYPE == 22u) ? 82ull : ((TYPE == 21u || TYPE == 11u) ? 110ull : 74ull));
        for (uint ib = 0u; ib < nb; ib++) {
            const uint xb = ib * 256u + x_lane_off;
            const float4 x0 = *(device const float4 *)(x + xb);
            const float4 x1 = *(device const float4 *)(x + xb + 4u);
            device const uchar *bg = wg + ulong(ib) * bsize;
            device const uchar *bu = wu + ulong(ib) * bsize;
            if (TYPE == 18u) {
                g += qw3_iq3xxs_dot8_f32(bg, ib32, q8, x0, x1, grid16, signs);
                u += qw3_iq3xxs_dot8_f32(bu, ib32, q8, x0, x1, grid16, signs);
            } else if (TYPE == 22u) {
                g += qw3_iq2s_dot8_f32(bg, ib32, q8, x0, x1, grid16);
                u += qw3_iq2s_dot8_f32(bu, ib32, q8, x0, x1, grid16);
            } else if (TYPE == 21u) {
                g += qw3_iq3s_dot8_f32(bg, ib32, q8, x0, x1, iq3s_kgrid);
                u += qw3_iq3s_dot8_f32(bu, ib32, q8, x0, x1, iq3s_kgrid);
            } else if (TYPE == 11u) {
                g += qw3_q3k_dot8_f32(bg, ib32, q8, x0, x1);
                u += qw3_q3k_dot8_f32(bu, ib32, q8, x0, x1);
            } else {
                g += qw3_iq2xs_dot8_f32(bg, ib32, q8, x0, x1, grid16, signs);
                u += qw3_iq2xs_dot8_f32(bu, ib32, q8, x0, x1, grid16, signs);
            }
        }
    } else {
        for (uint k0 = uint(lane) * 16u; k0 < args.n_in; k0 += 512u) {
            g += qw3_inkling_dot16_f32<TYPE>(wg, k0, x, grid16, signs, iq3s_kgrid);
            u += qw3_inkling_dot16_f32<TYPE>(wu, k0, x, grid16, signs, iq3s_kgrid);
        }
    }
    g = simd_sum(g);
    u = simd_sum(u);
    if (lane == 0u) {
        const float sc = route_coeffs[slot];
        const float ga = g * sc, up = u * sc;
        const float beta = args.beta, limit = args.limit;
        hidden_out[ulong(slot) * n_ff + row] =
            beta * tanh(ga / beta) / (1.0f + exp(-ga)) * limit * tanh(up / limit);
    }
}

template<uint TYPE>
kernel void qw3_inkling_moe_down_id(
    constant qw3_inkling_moe_dec_args &args [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const float *hidden_in [[buffer(2)]],
    device const uint *route_ids [[buffer(3)]],
    device float *down_out [[buffer(4)]],
    device const ushort *grid16 [[buffer(5)]],
    device const uchar *signs [[buffer(6)]],
    device const uchar *iq3s_kgrid [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort simd_idx [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint slot = group.y;
    const uint r0 = (group.x * 4u + uint(simd_idx)) * 2u;
    if (r0 >= args.n_out) return;
    const bool has_r1 = (r0 + 1u) < args.n_out;
    const uint expert = route_ids[slot];
    device const uchar *wexp = weights + ulong(expert) * ulong(args.expert_bytes);
    device const uchar *wr0 = wexp + ulong(r0) * args.row_bytes;
    device const uchar *wr1 = has_r1 ? (wr0 + args.row_bytes) : wr0;
    device const float *xs = hidden_in + ulong(slot) * args.n_in;
    float s0 = 0.0f, s1 = 0.0f;
    if (TYPE == 20u) {
        // 8 lanes/block = 4 blocks/iter -> exact 9 iters for n_in=1152, zero idle lanes!
        const uint nb = args.n_in >> 5u;
        const uint blk_lane = uint(lane) >> 3u;
        const uint in_blk = uint(lane) & 7u;
        for (uint b = blk_lane; b < nb; b += 4u) {
            const float4 x4 = *(device const float4 *)(xs + b * 32u + in_blk * 4u);
            const ulong boff = ulong(b) * 18ull;
            s0 += qw3_iq4nl_dot4_f32(wr0 + boff, in_blk, x4);
            if (has_r1) s1 += qw3_iq4nl_dot4_f32(wr1 + boff, in_blk, x4);
        }
    } else {
        for (uint k0 = uint(lane) * 16u; k0 < args.n_in; k0 += 512u) {
            s0 += qw3_inkling_dot16_f32<TYPE>(wr0, k0, xs, grid16, signs, iq3s_kgrid);
            if (has_r1) s1 += qw3_inkling_dot16_f32<TYPE>(wr1, k0, xs, grid16, signs, iq3s_kgrid);
        }
    }
    s0 = simd_sum(s0);
    s1 = simd_sum(s1);
    if (lane == 0u) {
        device float *dst = down_out + ulong(slot) * args.n_out;
        dst[r0] = s0;
        if (has_r1) dst[r0 + 1u] = s1;
    }
}

#define QW3_INST_INK_MOE_DEC(T) \
    template [[host_name("qw3_inkling_moe_gate_up_situ_t" #T)]] kernel void qw3_inkling_moe_gate_up_situ_id<T##u>( \
        constant qw3_inkling_moe_dec_args &, device const uchar *, device const float *, \
        device const uint *, device const float *, device float *, \
        device const ushort *, device const uchar *, device const uchar *, uint2, ushort, ushort); \
    template [[host_name("qw3_inkling_moe_down_t" #T)]] kernel void qw3_inkling_moe_down_id<T##u>( \
        constant qw3_inkling_moe_dec_args &, device const uchar *, device const float *, \
        device const uint *, device float *, \
        device const ushort *, device const uchar *, device const uchar *, uint2, ushort, ushort);

QW3_INST_INK_MOE_DEC(8)
QW3_INST_INK_MOE_DEC(11)
QW3_INST_INK_MOE_DEC(12)
QW3_INST_INK_MOE_DEC(13)
QW3_INST_INK_MOE_DEC(14)
QW3_INST_INK_MOE_DEC(17)
QW3_INST_INK_MOE_DEC(18)
QW3_INST_INK_MOE_DEC(20)
QW3_INST_INK_MOE_DEC(21)
QW3_INST_INK_MOE_DEC(22)
QW3_INST_INK_MOE_DEC(23)
#undef QW3_INST_INK_MOE_DEC

// Fused Q5_K output head projection + logit softcapping (1 row/simdgroup, 2 simdgroups = 64 threads/TG).
struct qw3_inkling_q5k_cap_args { uint n_in, n_out, row_bytes; float cap; };
kernel void qw3_inkling_matvec_q5_k_cap(constant qw3_inkling_q5k_cap_args &args [[buffer(0)]],
                                        device const uchar *weights [[buffer(1)]],
                                        device const float *x [[buffer(2)]],
                                        device float *out [[buffer(3)]],
                                        uint group [[threadgroup_position_in_grid]],
                                        ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                        ushort lane [[thread_index_in_simdgroup]]) {
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
    device const uchar *wr = weights + ulong(row) * ulong(args.row_bytes);
    device const float *x1 = x + ix * 256u + x_offset;
    float sum = 0.0f;

    for (uint ib = ix; ib < nb; ib += 4u) {
        device const uchar *blk = wr + ulong(ib) * 176ull;
        device const uchar *q1 = blk + 48u + q_offset;
        device const uchar *q2 = q1 + 64u;
        device const uchar *qh = blk + 16u + l0;
        device const ushort *a = (device const ushort *)(blk + 4u) + iq;
        device const float *x2 = x1 + 128u;
        float yl[16], yh[16];
        float4 sumx = float4(0.0f);
#pragma unroll
        for (uint l = 0; l < 8u; ++l) {
            yl[l] = x1[l];             sumx[0] += yl[l];
            yl[l + 8u] = x1[l + 32u];  sumx[1] += yl[l + 8u];
            yh[l] = x2[l];             sumx[2] += yh[l];
            yh[l + 8u] = x2[l + 32u];  sumx[3] += yh[l + 8u];
        }
        ushort sc16[4];
        sc16[0] = a[0] & kmask1;
        sc16[1] = a[2] & kmask1;
        sc16[2] = ((a[4] >> 0u) & kmask2) | ((a[0] & kmask3) >> 2u);
        sc16[3] = ((a[4] >> 4u) & kmask2) | ((a[2] & kmask3) >> 2u);
        thread const uchar *sc8 = (thread const uchar *)sc16;
        float4 acc1 = float4(0.0f), acc2 = float4(0.0f);
#pragma unroll
        for (uint l = 0; l < 8u; ++l) {
            const uchar h = qh[l];
            acc1[0] += yl[l]      * float(q1[l] & 0x0fu);
            acc1[1] += yl[l + 8u] * float(q1[l] & 0xf0u);
            acc1[2] += yh[l]      * float(q2[l] & 0x0fu);
            acc1[3] += yh[l + 8u] * float(q2[l] & 0xf0u);
            acc2[0] += (h & hm1) ? yl[l] : 0.0f;
            acc2[1] += (h & hm2) ? yl[l + 8u] : 0.0f;
            acc2[2] += (h & hm3) ? yh[l] : 0.0f;
            acc2[3] += (h & hm4) ? yh[l + 8u] : 0.0f;
        }
        const float d = float(*((device const half *)blk));
        const float dmin = float(*((device const half *)(blk + 2u)));
        sum += d *
            (float(sc8[0]) * (acc1[0] + 16.0f * acc2[0]) +
             float(sc8[1]) * (acc1[1] * (1.0f / 16.0f) + 16.0f * acc2[1]) +
             float(sc8[4]) * (acc1[2] + 16.0f * acc2[2]) +
             float(sc8[5]) * (acc1[3] * (1.0f / 16.0f) + 16.0f * acc2[3])) -
            dmin * (sumx[0] * float(sc8[2]) + sumx[1] * float(sc8[3]) +
                    sumx[2] * float(sc8[6]) + sumx[3] * float(sc8[7]));
        x1 += 4u * 256u;
    }
    sum = simd_sum(sum);
    if (lane == 0u) {
        const float cap = args.cap;
        out[row] = cap > 0.0f ? (cap * tanh(sum / cap)) : sum;
    }
}

kernel void qw3_inkling_expert_activation(constant uint2 &shape [[buffer(0)]],
                                           constant float2 &limits [[buffer(1)]],
                                           device const float *scales [[buffer(2)]],
                                           device float *gate_up [[buffer(3)]],
                                           uint2 pos [[thread_position_in_grid]]) {
    const uint width = shape.x;
    if (pos.x >= width) return;
    device float *expert = gate_up + pos.y * shape.y;
    const float gate = expert[pos.x] * scales[pos.y];
    const float up = expert[width + pos.x] * scales[pos.y];
    const float beta = limits.x, limit = limits.y;
    expert[pos.x] = beta * tanh(gate / beta) / (1.0f + exp(-gate)) * limit * tanh(up / limit);
}

kernel void qw3_inkling_expert_reduce(constant uint4 &shape [[buffer(0)]],
                                       device const float *weights [[buffer(1)]],
                                       device const float *down [[buffer(2)]],
                                       device float *out [[buffer(3)]],
                                       uint col [[thread_position_in_grid]]) {
    if (col >= shape.x) return;
    float value = 0.0f;
    for (uint i = 0; i < shape.z; i++) value += weights[i] * down[i * shape.y + col];
    out[col] = value;
}

// Batched Inkling prefill projections and mapped MoE kernels.
#ifdef QW3_METAL_HAS_TENSOR
struct qw3_inkling_matmul_args {
    uint n_in, n_out, row_bytes, n_tokens;
    uint in_offset, in_stride, out_offset, out_stride;
    uint type;
};

static inline half qw3_inkling_nvfp4_val(device const uchar *wrow, uint k) {
    device const uchar *blk = wrow + uint64_t(k >> 6u) * 36ull;
    const uint local = k & 63u;
    const uint sub = local >> 4u;
    const uint bits = uint(blk[sub]);
    const uint exponent = (bits >> 3u) & 15u;
    const uint mantissa = bits & 7u;
    const float scale = bits == 127u ? 0.0f :
        ldexp(float(exponent ? mantissa + 8u : mantissa),
              exponent ? int(exponent) - 11 : -10);
    const uint within = local & 15u;
    const uint code = blk[4u + 8u * sub + (within & 7u)];
    const int q = int(qw3_nvfp4_twice_e2m1[within < 8u ? (code & 15u) : (code >> 4u)]);
    return half(scale * float(q));
}

kernel void qw3_matmul_nvfp4_mpp_direct(
    constant qw3_inkling_matmul_args &args [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *out [[buffer(3)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int NR1 = 64;
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
        const int half_k = int(tid) & 1;
        if (r0 + row < int(args.n_out)) {
            const uint k0 = loop_k + uint(half_k * 16);
            device const uchar *wrow = weights + uint64_t(r0 + row) * args.row_bytes;
            device const uchar *blk = wrow + uint64_t(k0 >> 6u) * 36ull;
            const uint sub = (k0 & 63u) >> 4u;
            const float scale = qw3_nvfp4_scale(uint(blk[sub]));
            device const uint *vals = (device const uint *)(blk + 4u + 8u * sub);
            const uint w0 = vals[0], w1 = vals[1];
            threadgroup half4 *sa4 = (threadgroup half4 *)(sa + row * NK + half_k * 16);
            sa4[0] = half4(float4(qw3_nvfp4_e2m1_f[w0 & 15u], qw3_nvfp4_e2m1_f[(w0 >> 8u) & 15u],
                                  qw3_nvfp4_e2m1_f[(w0 >> 16u) & 15u], qw3_nvfp4_e2m1_f[(w0 >> 24u) & 15u]) * scale);
            sa4[1] = half4(float4(qw3_nvfp4_e2m1_f[w1 & 15u], qw3_nvfp4_e2m1_f[(w1 >> 8u) & 15u],
                                  qw3_nvfp4_e2m1_f[(w1 >> 16u) & 15u], qw3_nvfp4_e2m1_f[(w1 >> 24u) & 15u]) * scale);
            sa4[2] = half4(float4(qw3_nvfp4_e2m1_f[(w0 >> 4u) & 15u], qw3_nvfp4_e2m1_f[(w0 >> 12u) & 15u],
                                  qw3_nvfp4_e2m1_f[(w0 >> 20u) & 15u], qw3_nvfp4_e2m1_f[(w0 >> 28u) & 15u]) * scale);
            sa4[3] = half4(float4(qw3_nvfp4_e2m1_f[(w1 >> 4u) & 15u], qw3_nvfp4_e2m1_f[(w1 >> 12u) & 15u],
                                  qw3_nvfp4_e2m1_f[(w1 >> 20u) & 15u], qw3_nvfp4_e2m1_f[(w1 >> 28u) & 15u]) * scale);
        } else {
            threadgroup half4 *sa4 = (threadgroup half4 *)(sa + row * NK + half_k * 16);
            sa4[0] = half4(0.0f); sa4[1] = half4(0.0f); sa4[2] = half4(0.0f); sa4[3] = half4(0.0f);
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

kernel void qw3_matmul_float_mpp_direct(
    constant qw3_inkling_matmul_args &args [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *out [[buffer(3)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int NR1 = 64;
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
        const int half_k = int(tid) & 1;
        if (r0 + row < int(args.n_out)) {
            const uint k0 = loop_k + uint(half_k * 16);
            device const uchar *wrow = weights + uint64_t(r0 + row) * args.row_bytes;
            if (args.type == 0u) {
                device const float *wf = (device const float *)wrow + k0;
                for (int i = 0; i < 16; i++)
                    sa[row * NK + half_k * 16 + i] = k0 + uint(i) < args.n_in ? half(wf[i]) : half(0.0f);
            } else if (args.type == 30u) {
                device const ushort *wb = (device const ushort *)wrow + k0;
                for (int i = 0; i < 16; i++)
                    sa[row * NK + half_k * 16 + i] = k0 + uint(i) < args.n_in ?
                        half(as_type<float>(uint(wb[i]) << 16u)) : half(0.0f);
            } else {
                device const half *wh = (device const half *)wrow + k0;
                for (int i = 0; i < 16; i++)
                    sa[row * NK + half_k * 16 + i] = k0 + uint(i) < args.n_in ? wh[i] : half(0.0f);
            }
        } else {
            for (int i = 0; i < 16; i++) sa[row * NK + half_k * 16 + i] = half(0.0f);
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

// High-occupancy small-tile F32/BF16/F16 MPP matmul for small n_out (router 128, loop_down 64, sink 24, taps 32).
template<int NR0, int NR1>
kernel void qw3_matmul_float_mpp_small(
    constant qw3_inkling_matmul_args &args [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *out [[buffer(3)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
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
    // 128 threads load NR0 * 32 halfs = NR0 * 8 half4s
    constexpr int TOTAL_H4 = NR0 * (NK / 4);
    for (uint loop_k = 0; loop_k < args.n_in; loop_k += uint(NK)) {
        for (int idx = int(tid); idx < TOTAL_H4; idx += 128) {
            const int row = idx >> 3;
            const int k4 = idx & 7;
            const uint k0 = loop_k + uint(k4 * 4);
            threadgroup half4 *sa4 = (threadgroup half4 *)sa;
            if (r0 + row < int(args.n_out) && k0 < args.n_in) {
                device const uchar *wrow = weights + uint64_t(r0 + row) * args.row_bytes;
                if (args.type == 0u) {
                    sa4[idx] = half4(*(device const float4 *)((device const float *)wrow + k0));
                } else if (args.type == 30u) {
                    const ushort4 wb = *(device const ushort4 *)((device const ushort *)wrow + k0);
                    sa4[idx] = half4(as_type<float4>(uint4(wb) << 16u));
                } else {
                    sa4[idx] = *(device const half4 *)((device const half *)wrow + k0);
                }
            } else {
                sa4[idx] = half4(0.0f);
            }
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
template [[host_name("qw3_matmul_float_mpp_s32")]] kernel void qw3_matmul_float_mpp_small<32, 32>(
    constant qw3_inkling_matmul_args &, device const uchar *, device const float *, device float *,
    threadgroup char *, uint2, ushort);
template [[host_name("qw3_matmul_float_mpp_s16")]] kernel void qw3_matmul_float_mpp_small<16, 32>(
    constant qw3_inkling_matmul_args &, device const uchar *, device const float *, device float *,
    threadgroup char *, uint2, ushort);

// STAGE_K=64 float/bf16/f16 MPP for small n_out (control_down=64, router=128, sink=24) at n_tokens>=64:
// 2x fewer weight reads & 2x fewer barriers.
template<int NR0, int NR1>
kernel void qw3_matmul_float_mpp_small_k64(
    constant qw3_inkling_matmul_args &args [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *out [[buffer(3)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NK = 32;
    constexpr int STAGE_K = 64;
    const int r0 = int(group.y) * NR0;
    const int r1 = int(group.x) * NR1;
    if (r0 >= int(args.n_out) || r1 >= int(args.n_tokens)) return;
    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(STAGE_K, NR0));
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
    constexpr int TOTAL_H4 = NR0 * (STAGE_K / 4);
    threadgroup half4 *sa4 = (threadgroup half4 *)sa;
    for (uint stage_k = 0; stage_k < args.n_in; stage_k += uint(STAGE_K)) {
        for (int idx = int(tid); idx < TOTAL_H4; idx += 128) {
            const int row = idx >> 4;
            const int k4 = idx & 15;
            const uint k0 = stage_k + uint(k4 * 4);
            if (r0 + row < int(args.n_out) && k0 < args.n_in) {
                device const uchar *wrow = weights + uint64_t(r0 + row) * args.row_bytes;
                if (args.type == 0u) {
                    sa4[idx] = half4(*(device const float4 *)((device const float *)wrow + k0));
                } else if (args.type == 30u) {
                    const ushort4 wb = *(device const ushort4 *)((device const ushort *)wrow + k0);
                    sa4[idx] = half4(as_type<float4>(uint4(wb) << 16u));
                } else {
                    sa4[idx] = *(device const half4 *)((device const half *)wrow + k0);
                }
            } else {
                sa4[idx] = half4(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            auto mA0 = tA.slice(0, 0);
            auto mB0 = tB.slice(int(stage_k), r1);
            mm.run(mB0, mA0, cT);
            auto mA1 = tA.slice(32, 0);
            auto mB1 = tB.slice(int(stage_k + 32u), r1);
            mm.run(mB1, mA1, cT);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tD = tensor(out + args.out_offset,
                     dextents<int32_t, 2>(int(args.n_out), int(args.n_tokens)),
                     array<int, 2>({1, int(args.out_stride)}));
    cT.store(tD.slice(r0, r1));
}
template [[host_name("qw3_matmul_float_mpp_s32_n64")]] kernel void qw3_matmul_float_mpp_small_k64<32, 64>(
    constant qw3_inkling_matmul_args &, device const uchar *, device const float *, device float *,
    threadgroup char *, uint2, ushort);
template [[host_name("qw3_matmul_float_mpp_s16_n64")]] kernel void qw3_matmul_float_mpp_small_k64<16, 64>(
    constant qw3_inkling_matmul_args &, device const uchar *, device const float *, device float *,
    threadgroup char *, uint2, ushort);

struct qw3_inkling_moe_mapped_args {
    uint n_in, n_out, row_bytes;
    uint expert_bytes;
    uint n_tokens, n_active;
    uint type;
    uint input_is_slot;
    uint in_stride, out_stride;
};

static inline void qw3_inkling_dequant_16_generic(
    device const uchar *wrow, uint type, uint k0,
    device const ushort *grid16, device const uchar *signs,
    device const uchar *iq3s_kgrid, thread half *dst) {
    switch (type) {
    case 40u: {
        device const uchar *blk = wrow + uint64_t(k0 >> 6u) * 36ull;
        const uint sub = (k0 & 63u) >> 4u;
        const uint bits = uint(blk[sub]);
        const uint exponent = (bits >> 3u) & 15u;
        const uint mantissa = bits & 7u;
        const float scale = bits == 127u ? 0.0f :
            ldexp(float(exponent ? mantissa + 8u : mantissa),
                  exponent ? int(exponent) - 11 : -10);
        device const uchar *vals = blk + 4u + 8u * sub;
        for (int i = 0; i < 8; i++) {
            const uint code = vals[i];
            dst[i] = half(scale * float(qw3_nvfp4_twice_e2m1[code & 15u]));
            dst[8 + i] = half(scale * float(qw3_nvfp4_twice_e2m1[code >> 4u]));
        }
        return;
    }
    case 20u: {
        device const uchar *blk = wrow + uint64_t(k0 >> 5u) * 18ull;
        const float d = float(*((device const half *)blk));
        device const uchar *qs = blk + 2u;
        const uint local0 = k0 & 31u;
        for (int i = 0; i < 16; i++) {
            const uint local = local0 + uint(i);
            const uchar packed = qs[local & 15u];
            const uint q = local < 16u ? (uint(packed) & 15u) : (uint(packed) >> 4u);
            dst[i] = half(d * qw3_iq4nl_table[q]);
        }
        return;
    }
    case 11u: {
        device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 110ull;
        device const uchar *packed = blk + 96u;
        const uint aux0 = uint(packed[0]) | (uint(packed[1]) << 8u) |
                          (uint(packed[2]) << 16u) | (uint(packed[3]) << 24u);
        const uint aux1 = uint(packed[4]) | (uint(packed[5]) << 8u) |
                          (uint(packed[6]) << 16u) | (uint(packed[7]) << 24u);
        const uint tmp = uint(packed[8]) | (uint(packed[9]) << 8u) |
                         (uint(packed[10]) << 16u) | (uint(packed[11]) << 24u);
        const uint kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
        uint aux[4];
        aux[2] = ((aux0 >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u);
        aux[3] = ((aux1 >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u);
        aux[0] = (aux0 & kmask2) | (((tmp >> 0u) & kmask1) << 4u);
        aux[1] = (aux1 & kmask2) | (((tmp >> 2u) & kmask1) << 4u);
        const float d = float(*((device const half *)(blk + 108u)));
        const uint local0 = k0 & 255u;
        const uint half_block = local0 >> 7u;
        const uint within = local0 & 127u;
        const uint grp = within >> 5u;
        const uint upper = (within & 31u) >= 16u ? 1u : 0u;
        const uint scale_index = half_block * 8u + grp * 2u + upper;
        const uint scale = (aux[scale_index >> 2u] >> (8u * (scale_index & 3u))) & 255u;
        const float dl = d * float(int(scale) - 32);
        const uint shift = grp * 2u;
        const uint mask = 1u << (half_block * 4u + grp);
        for (int i = 0; i < 16; i++) {
            const uint l = uint(i);
            const uint q_index = half_block * 32u + upper * 16u + l;
            const int low = int((uint(blk[32u + q_index]) >> shift) & 3u);
            const int q = low - ((uint(blk[upper * 16u + l]) & mask) ? 0 : 4);
            dst[i] = half(dl * float(q));
        }
        return;
    }
    case 17u: {
        device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 74ull;
        const float d = qw3_f16_to_f32(ushort(blk[0]) | (ushort(blk[1]) << 8u));
        device const uchar *qs = blk + 2u;
        device const uchar *scales = qs + 64u;
        const uint il = (k0 & 255u) >> 4u;
        const uint ib32 = il >> 1u;
        const uint half16 = il & 1u;
        const uint off = 2u * (4u * ib32 + 2u * half16);
        const uint q0 = uint(qs[off]) | (uint(qs[off + 1u]) << 8u);
        const uint q1 = uint(qs[off + 2u]) | (uint(qs[off + 3u]) << 8u);
        const float db = d * (0.5f + float((uint(scales[ib32]) >> (4u * half16)) & 15u)) * 0.25f;
        const ushort grid0 = grid16[q0 & 511u];
        const ushort grid1 = grid16[q1 & 511u];
        const uint sign0 = uint(signs[q0 >> 9u]);
        const uint sign1 = uint(signs[q1 >> 9u]);
        for (uint i = 0; i < 8u; i++) {
            const float s0 = (sign0 & (1u << i)) ? -1.0f : 1.0f;
            const float s1 = (sign1 & (1u << i)) ? -1.0f : 1.0f;
            dst[i] = half(db * qw3_iq2_grid_value(grid0, i) * s0);
            dst[8u + i] = half(db * qw3_iq2_grid_value(grid1, i) * s1);
        }
        return;
    }
    case 18u: {
        device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 98ull;
        const float d = qw3_f16_to_f32(ushort(blk[0]) | (ushort(blk[1]) << 8u));
        device const uchar *qs = blk + 2u;
        device const uchar *ss = qs + 64u;
        const uint il = (k0 & 255u) >> 4u;
        const uint ib32 = il >> 1u;
        const uint half16 = il & 1u;
        device const uchar *q3 = qs + 8u * ib32;
        device const uchar *sp = ss + 4u * ib32;
        const uint packed = uint(sp[0]) | (uint(sp[1]) << 8u) |
                            (uint(sp[2]) << 16u) | (uint(sp[3]) << 24u);
        const float db = d * (0.5f + float(packed >> 28u)) * 0.5f;
        const uint sign0 = uint(signs[(packed >> (14u * half16)) & 127u]);
        const uint sign1 = uint(signs[(packed >> (14u * half16 + 7u)) & 127u]);
        const ushort g0 = grid16[uint(q3[4u * half16])];
        const ushort g1 = grid16[uint(q3[4u * half16 + 1u])];
        const ushort g2 = grid16[uint(q3[4u * half16 + 2u])];
        const ushort g3 = grid16[uint(q3[4u * half16 + 3u])];
        for (uint i = 0; i < 4u; i++) {
            const float s00 = (sign0 & (1u << i)) ? -1.0f : 1.0f;
            const float s01 = (sign0 & (1u << (i + 4u))) ? -1.0f : 1.0f;
            const float s10 = (sign1 & (1u << i)) ? -1.0f : 1.0f;
            const float s11 = (sign1 & (1u << (i + 4u))) ? -1.0f : 1.0f;
            dst[i]       = half(db * qw3_iq3xxs_grid_value(g0, i) * s00);
            dst[4u + i]  = half(db * qw3_iq3xxs_grid_value(g1, i) * s01);
            dst[8u + i]  = half(db * qw3_iq3xxs_grid_value(g2, i) * s10);
            dst[12u + i] = half(db * qw3_iq3xxs_grid_value(g3, i) * s11);
        }
        return;
    }
    case 21u: {
        device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 110ull;
        const uint il = (k0 & 255u) >> 4u;
        const uint ib32 = il >> 1u;
        const uint half_il = il & 1u;
        const float d = float(*((device const half *)blk));
        device const uchar *qs = blk + 2u;
        device const uchar *qh = qs + 64u;
        device const uchar *sbuf = qh + 8u;
        device const uchar *scales = sbuf + 32u;
        const float dl = d * float(1u + 2u * ((uint(scales[ib32 >> 1u]) >> (4u * (ib32 & 1u))) & 15u));
        device const uchar *qs16 = qs + 8u * ib32 + 4u * half_il;
        device const uchar *signs16 = sbuf + 4u * ib32 + 2u * half_il;
        const uchar qh16 = uchar(uint(qh[ib32]) >> (4u * half_il));
        for (uint qn = 0; qn < 4u; qn++) {
            const uint idx = uint(qs16[qn]) | ((uint(qh16) << (8u - qn)) & 256u);
            device const uchar *g = iq3s_kgrid + (idx & 511u) * 4u;
            const uchar sb = signs16[qn >> 1u];
            const uint sshift = 4u * (qn & 1u);
            for (uint j = 0; j < 4u; j++) {
                const float sg = (uint(sb) & (1u << (j + sshift))) ? -1.0f : 1.0f;
                dst[qn * 4u + j] = half(dl * float(g[j]) * sg);
            }
        }
        return;
    }
    case 22u: {
        device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 82ull;
        const float d = qw3_f16_to_f32(ushort(blk[0]) | (ushort(blk[1]) << 8u));
        device const uchar *qs = blk + 2u;
        device const uchar *qh = qs + 64u;
        device const uchar *scales = qh + 8u;
        const uint il = (k0 & 255u) >> 4u;
        const uint ib32 = il >> 1u;
        const uint half16 = il & 1u;
        const float db = d * (0.5f + float((uint(scales[ib32]) >> (4u * half16)) & 15u)) * 0.25f;
        const uint qh_byte = uint(qh[ib32]) >> (4u * half16);
        device const uchar *q2 = qs + 4u * ib32 + 2u * half16;
        device const uchar *s2 = q2 + 32u;
        const uint idx0 = uint(q2[0]) | ((qh_byte & 3u) << 8u);
        const uint idx1 = uint(q2[1]) | (((qh_byte >> 2u) & 3u) << 8u);
        const ushort g0 = grid16[idx0];
        const ushort g1 = grid16[idx1];
        const uint sg0 = uint(s2[0]);
        const uint sg1 = uint(s2[1]);
        for (uint i = 0; i < 8u; i++) {
            dst[i] = half(db * qw3_iq2_grid_value(g0, i) * ((sg0 & (1u << i)) ? -1.0f : 1.0f));
            dst[8u + i] = half(db * qw3_iq2_grid_value(g1, i) * ((sg1 & (1u << i)) ? -1.0f : 1.0f));
        }
        return;
    }
    case 12u: case 13u: {
        const bool is_q5 = type == 13u;
        device const uchar *blk = wrow + uint64_t(k0 >> 8u) * (is_q5 ? 176ull : 144ull);
        const float d = qw3_f16_to_f32(ushort(blk[0]) | (ushort(blk[1]) << 8u));
        const float dmin = qw3_f16_to_f32(ushort(blk[2]) | (ushort(blk[3]) << 8u));
        device const uchar *scales = blk + 4u;
        device const uchar *qh = blk + 16u;
        device const uchar *ql = blk + (is_q5 ? 48u : 16u);
        const uint g = (k0 & 255u) >> 5u;
        const uint l0 = k0 & 16u;
        uint sc, mn;
        if (g < 4u) {
            sc = uint(scales[g]) & 63u;
            mn = uint(scales[g + 4u]) & 63u;
        } else {
            sc = (uint(scales[g + 4u]) & 15u) | ((uint(scales[g - 4u]) >> 6u) << 4u);
            mn = (uint(scales[g + 4u]) >> 4u) | ((uint(scales[g]) >> 6u) << 4u);
        }
        const float dl = d * float(sc), ml = dmin * float(mn);
        device const uchar *qgroup = ql + (g >> 1u) * 32u + l0;
        const uint high_mask = 1u << g;
        const bool high_nibble = (g & 1u) != 0u;
        for (int i = 0; i < 16; i++) {
            const uint l = l0 + uint(i);
            const uint packed = uint(qgroup[i]);
            uint q = high_nibble ? (packed >> 4u) : (packed & 15u);
            if (is_q5 && (uint(qh[l]) & high_mask)) q += 16u;
            dst[i] = half(dl * float(q) - ml);
        }
        return;
    }
    case 14u: {
        device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 210ull;
        const float d = qw3_f16_to_f32(ushort(blk[208]) | (ushort(blk[209]) << 8u));
        for (int i = 0; i < 16; i++) {
            const uint local = (k0 + uint(i)) & 255u;
            const uint half_block = local >> 7u;
            const uint within = local & 127u;
            const uint part = within >> 5u;
            const uint l = within & 31u;
            const uint ql_index = half_block * 64u + (part & 1u) * 32u + l;
            const uint packed = uint(blk[ql_index]);
            const uint low = part < 2u ? (packed & 15u) : (packed >> 4u);
            const uint high = (uint(blk[128u + half_block * 32u + l]) >> (2u * part)) & 3u;
            const int q = int(low | (high << 4u)) - 32;
            const int scale = int(((device const char *)(blk + 192u))[half_block * 8u + 2u * part + l / 16u]);
            dst[i] = half(d * float(scale) * float(q));
        }
        return;
    }
    case 8u: {
        device const uchar *blk = wrow + uint64_t(k0 >> 5u) * 34ull;
        const float d = float(*((device const half *)blk));
        device const char *qs = (device const char *)(blk + 2u) + (k0 & 16u);
        for (int i = 0; i < 16; i++) dst[i] = half(d * float(qs[i]));
        return;
    }
    case 23u: {
        device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 136ull;
        const float d = float(*((device const half *)blk));
        const ushort scales_h = *((device const ushort *)(blk + 2u));
        device const uchar *scales_l = blk + 4u;
        device const uchar *qs = blk + 8u;
        const uint ib = (k0 & 255u) >> 5u;
        const uint ls = ((uint(scales_l[ib >> 1u]) >> (4u * (ib & 1u))) & 15u) |
                        (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);
        const float dl = d * float(int(ls) - 32);
        device const uchar *q = qs + ib * 16u;
        const bool hi = (k0 & 16u) != 0u;
        for (int i = 0; i < 16; i++) {
            const uint v = uint(q[i]);
            dst[i] = half(dl * qw3_iq4nl_table[hi ? (v >> 4u) : (v & 15u)]);
        }
        return;
    }
    default: {
        if (type == 0u) {
            device const float *wf = (device const float *)wrow + k0;
            for (int i = 0; i < 16; i++) dst[i] = half(wf[i]);
        } else if (type == 30u) {
            device const ushort *wb = (device const ushort *)wrow + k0;
            for (int i = 0; i < 16; i++) dst[i] = half(as_type<float>(uint(wb[i]) << 16u));
        } else {
            device const half *wh = (device const half *)wrow + k0;
            for (int i = 0; i < 16; i++) dst[i] = wh[i];
        }
        return;
    }
    }
}

kernel void qw3_inkling_moe_mapped_mpp(
    constant qw3_inkling_moe_mapped_args &args [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const float *xin [[buffer(2)]],
    device float *out_slots [[buffer(3)]],
    device const uint *counts [[buffer(4)]],
    device const int *pair_ids [[buffer(5)]],
    device const uint *block_ids [[buffer(6)]],
    device const ushort *grid16 [[buffer(7)]],
    device const uchar *signs [[buffer(8)]],
    device const uchar *iq3s_kgrid [[buffer(9)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    const uint r0u = group.x * uint(NR0);
    const uint block = block_ids[group.y];
    const uint expert = block & 255u;
    const uint r1u = block >> 8u;
    const uint count = counts[expert];
    if (r0u >= args.n_out || r1u >= count) return;
    const int nr0 = int(min(uint(NR0), args.n_out - r0u));
    const int nr1 = int(min(uint(NR1), count - r1u));
    const uint map_base = expert * args.n_tokens + r1u;
    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);
    threadgroup float *sc = (threadgroup float *)shmem;
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        sa, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        sb, dextents<int32_t, 2>(NK, NR1));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<
        decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); i++)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    device const uchar *wexp = weights + uint64_t(expert) * uint64_t(args.expert_bytes);
    device const half *xin_h = (device const half *)xin;
    for (uint loop_k = 0; loop_k < args.n_in; loop_k += uint(NK)) {
        const int row = int(tid) / 2;
        const int half_k = int(tid) & 1;
        if (row < nr0) {
            half vals[16];
            qw3_inkling_dequant_16_generic(
                wexp + uint64_t(r0u + uint(row)) * args.row_bytes,
                args.type, loop_k + uint(half_k * 16),
                grid16, signs, iq3s_kgrid, vals);
            for (int i = 0; i < 16; i++) sa[row * NK + half_k * 16 + i] = vals[i];
        } else {
            for (int i = 0; i < 16; i++) sa[row * NK + half_k * 16 + i] = half(0.0f);
        }
        const int tok = int(tid) / 4;
        const int k_part = int(tid) & 3;
        if (tok < nr1) {
            const uint pid = uint(pair_ids[map_base + uint(tok)]);
            if (args.input_is_slot != 0u) {
                device const half *xrow = xin_h + uint64_t(pid) * args.in_stride + loop_k + uint(k_part * 8);
                for (int i = 0; i < 8; i++) sb[tok * NK + k_part * 8 + i] = xrow[i];
            } else {
                const uint row_idx = pid / args.n_active;
                device const float *xrow = xin + uint64_t(row_idx) * args.in_stride + loop_k + uint(k_part * 8);
                for (int i = 0; i < 8; i++) sb[tok * NK + k_part * 8 + i] = half(xrow[i]);
            }
        } else {
            for (int i = 0; i < 8; i++) sb[tok * NK + k_part * 8 + i] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mB = tB.slice(0, 0);
        auto mA = tA.slice(0, 0);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        sc, dextents<int32_t, 2>(NR0, NR1));
    cT.store(tC.slice(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = int(sgitg); j < nr1; j += 4) {
        const uint pid = uint(pair_ids[map_base + uint(j)]);
        device float *dst = out_slots + uint64_t(pid) * args.out_stride + r0u;
        threadgroup float *src = sc + j * NR0;
        for (int i = int(tid & 31u); i < nr0; i += 32) dst[i] = src[i];
    }
}

// Gather+convert FP32 tokens into expert-contiguous FP16 buffer so gate_up reads tB directly via NAX tensor.
kernel void qw3_inkling_moe_gather_x_f16(
    constant uint2 &args [[buffer(0)]],
    device const float4 *xin [[buffer(1)]],
    device const int *flat_pair_ids [[buffer(2)]],
    device half4 *xout [[buffer(3)]],
    uint slot [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    if (slot >= args.y) return;
    const uint n4 = args.x >> 2u;
    const uint tok = uint(flat_pair_ids[slot]) >> 3u;
    device const float4 *src = xin + uint64_t(tok) * n4;
    device half4 *dst = xout + uint64_t(slot) * n4;
    for (uint i = uint(tid); i < n4; i += 128u) {
        dst[i] = half4(src[i]);
    }
}

// Direct-tensor zero-sb NVFP4 mapped MoE down kernel: 8 KiB shmem for ALL NR1 (16/32/64),
// reads expert-contiguous FP16 hidden directly via Metal 4 NAX tensor, pre-weights FP16 output slots.
template<int NR1>
kernel void qw3_inkling_moe_mapped_nvfp4_mpp(
    constant qw3_inkling_moe_mapped_args &args [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const half *xin [[buffer(2)]],
    device half *out_slots [[buffer(3)]],
    device const uint *counts [[buffer(4)]],
    device const int *pair_ids [[buffer(5)]],
    device const uint *block_ids [[buffer(6)]],
    device const float *down_weights [[buffer(7)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int NK_STEP = 64;
    constexpr int NK = 32;
    const uint r0u = group.x * uint(NR0);
    const uint block = block_ids[group.y];
    const uint expert = block & 255u;
    const uint r1u = block >> 8u;
    const uint count = counts[expert];
    if (r0u >= args.n_out || r1u >= count) return;
    const uint exp_off = counts[128u + expert];
    const int nr0 = int(min(uint(NR0), args.n_out - r0u));
    const int nr1 = int(min(uint(NR1), count - r1u));
    const uint map_base = expert * args.n_tokens + r1u;
    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup float *sc = (threadgroup float *)shmem;
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        sa, dextents<int32_t, 2>(NK_STEP, NR0));
    device half *ptrB = (device half *)xin + uint64_t(exp_off) * args.in_stride;
    auto tB = tensor(ptrB, dextents<int32_t, 2>(int(args.n_in), int(count)),
                     array<int, 2>({1, int(args.in_stride)}));
    device const uchar *wexp = weights + uint64_t(expert) * uint64_t(args.expert_bytes);
    const int row = int(tid) >> 1;
    const int half_blk = int(tid) & 1;
    device const uchar *wrow = (row < nr0) ? (wexp + uint64_t(r0u + uint(row)) * args.row_bytes) : wexp;
    threadgroup half4 *sa4 = (threadgroup half4 *)(sa + row * NK_STEP + half_blk * 32);
    const int lane = int(tid & 31u);

    if (NR1 >= 64) {
        constexpr int ROWS_PER_SG = (NR1 == 128) ? 32 : 16;
        const int sg_r1 = int(r1u) + int(sgitg) * ROWS_PER_SG;
        const int sg_nr1 = clamp(int(count) - sg_r1, 0, ROWS_PER_SG);
        const bool sg_active = sg_nr1 > 0;
        matmul2d<matmul2d_descriptor(ROWS_PER_SG, NR0, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
            execution_simdgroups<1>> mm;
        auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
        for (uint16_t i = 0; i < cT.get_capacity(); i++)
            if (cT.is_valid_element(i)) cT[i] = 0.0f;
        for (uint loop_k = 0; loop_k < args.n_in; loop_k += uint(NK_STEP)) {
            if (row < nr0) {
                device const uchar *blk = wrow + uint64_t(loop_k >> 6u) * 36ull;
                const uint sub0 = uint(half_blk * 2);
                const half sc0 = qw3_nvfp4_scale_h[blk[sub0]];
                const half sc1 = qw3_nvfp4_scale_h[blk[sub0 + 1u]];
                device const uint *vp = (device const uint *)(blk + 4u + 8u * sub0);
                const uint w0 = vp[0], w1 = vp[1], w2 = vp[2], w3 = vp[3];
                sa4[0] = qw3_nvfp4_decode4_h(w0 & 15u, (w0 >> 8u) & 15u, (w0 >> 16u) & 15u, (w0 >> 24u) & 15u) * sc0;
                sa4[1] = qw3_nvfp4_decode4_h(w1 & 15u, (w1 >> 8u) & 15u, (w1 >> 16u) & 15u, (w1 >> 24u) & 15u) * sc0;
                sa4[2] = qw3_nvfp4_decode4_h((w0 >> 4u) & 15u, (w0 >> 12u) & 15u, (w0 >> 20u) & 15u, (w0 >> 28u) & 15u) * sc0;
                sa4[3] = qw3_nvfp4_decode4_h((w1 >> 4u) & 15u, (w1 >> 12u) & 15u, (w1 >> 20u) & 15u, (w1 >> 28u) & 15u) * sc0;
                sa4[4] = qw3_nvfp4_decode4_h(w2 & 15u, (w2 >> 8u) & 15u, (w2 >> 16u) & 15u, (w2 >> 24u) & 15u) * sc1;
                sa4[5] = qw3_nvfp4_decode4_h(w3 & 15u, (w3 >> 8u) & 15u, (w3 >> 16u) & 15u, (w3 >> 24u) & 15u) * sc1;
                sa4[6] = qw3_nvfp4_decode4_h((w2 >> 4u) & 15u, (w2 >> 12u) & 15u, (w2 >> 20u) & 15u, (w2 >> 28u) & 15u) * sc1;
                sa4[7] = qw3_nvfp4_decode4_h((w3 >> 4u) & 15u, (w3 >> 12u) & 15u, (w3 >> 20u) & 15u, (w3 >> 28u) & 15u) * sc1;
            } else {
                for (int i = 0; i < 8; i++) sa4[i] = half4(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg_active) {
                auto mA0 = tA.slice(0, 0);
                auto mB0 = tB.slice(int(loop_k), sg_r1);
                mm.run(mB0, mA0, cT);
                auto mA1 = tA.slice(32, 0);
                auto mB1 = tB.slice(int(loop_k + 32u), sg_r1);
                mm.run(mB1, mA1, cT);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        constexpr int WAVES = (NR1 == 128) ? 4 : 2;
        constexpr int SGS_PER_WAVE = 4 / WAVES;
        for (int w = 0; w < WAVES; w++) {
            const int sg_in_w = int(sgitg) - w * SGS_PER_WAVE;
            if (sg_in_w >= 0 && sg_in_w < SGS_PER_WAVE && sg_active) {
                threadgroup float *sc_sg = sc + sg_in_w * (ROWS_PER_SG * NR0);
                auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
                    sc_sg, dextents<int32_t, 2>(NR0, ROWS_PER_SG));
                cT.store(tC.slice(0, 0));
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (int j = 0; j < sg_nr1; j++) {
                    const uint pid = uint(pair_ids[expert * args.n_tokens + uint(sg_r1 + j)]);
                    const float dw = down_weights[pid];
                    device half *dst = out_slots + uint64_t(pid) * args.out_stride + r0u;
                    threadgroup float *src = sc_sg + j * NR0;
                    if (lane < nr0 / 4) {
                        ((device half4 *)dst)[lane] = half4(((threadgroup float4 *)src)[lane] * dw);
                    } else if (4 * (nr0 / 4) + lane < nr0) {
                        dst[4 * (nr0 / 4) + lane] = half(src[4 * (nr0 / 4) + lane] * dw);
                    }
                }
            }
            if (w + 1 < WAVES) threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        return;
    }

    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); i++)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    for (uint loop_k = 0; loop_k < args.n_in; loop_k += uint(NK_STEP)) {
        if (row < nr0) {
            device const uchar *blk = wrow + uint64_t(loop_k >> 6u) * 36ull;
            const uint sub0 = uint(half_blk * 2);
            const half sc0 = qw3_nvfp4_scale_h[blk[sub0]];
            const half sc1 = qw3_nvfp4_scale_h[blk[sub0 + 1u]];
            device const uint *vp = (device const uint *)(blk + 4u + 8u * sub0);
            const uint w0 = vp[0], w1 = vp[1], w2 = vp[2], w3 = vp[3];
            sa4[0] = qw3_nvfp4_decode4_h(w0 & 15u, (w0 >> 8u) & 15u, (w0 >> 16u) & 15u, (w0 >> 24u) & 15u) * sc0;
            sa4[1] = qw3_nvfp4_decode4_h(w1 & 15u, (w1 >> 8u) & 15u, (w1 >> 16u) & 15u, (w1 >> 24u) & 15u) * sc0;
            sa4[2] = qw3_nvfp4_decode4_h((w0 >> 4u) & 15u, (w0 >> 12u) & 15u, (w0 >> 20u) & 15u, (w0 >> 28u) & 15u) * sc0;
            sa4[3] = qw3_nvfp4_decode4_h((w1 >> 4u) & 15u, (w1 >> 12u) & 15u, (w1 >> 20u) & 15u, (w1 >> 28u) & 15u) * sc0;
            sa4[4] = qw3_nvfp4_decode4_h(w2 & 15u, (w2 >> 8u) & 15u, (w2 >> 16u) & 15u, (w2 >> 24u) & 15u) * sc1;
            sa4[5] = qw3_nvfp4_decode4_h(w3 & 15u, (w3 >> 8u) & 15u, (w3 >> 16u) & 15u, (w3 >> 24u) & 15u) * sc1;
            sa4[6] = qw3_nvfp4_decode4_h((w2 >> 4u) & 15u, (w2 >> 12u) & 15u, (w2 >> 20u) & 15u, (w2 >> 28u) & 15u) * sc1;
            sa4[7] = qw3_nvfp4_decode4_h((w3 >> 4u) & 15u, (w3 >> 12u) & 15u, (w3 >> 20u) & 15u, (w3 >> 28u) & 15u) * sc1;
        } else {
            for (int i = 0; i < 8; i++) sa4[i] = half4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            auto mA0 = tA.slice(0, 0);
            auto mB0 = tB.slice(int(loop_k), int(r1u));
            mm.run(mB0, mA0, cT);
            auto mA1 = tA.slice(32, 0);
            auto mB1 = tB.slice(int(loop_k + 32u), int(r1u));
            mm.run(mB1, mA1, cT);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        sc, dextents<int32_t, 2>(NR0, NR1));
    cT.store(tC.slice(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = int(sgitg); j < nr1; j += 4) {
        const uint pid = uint(pair_ids[map_base + uint(j)]);
        const float dw = down_weights[pid];
        device half *dst = out_slots + uint64_t(pid) * args.out_stride + r0u;
        threadgroup float *src = sc + j * NR0;
        if (lane < nr0 / 4) {
            ((device half4 *)dst)[lane] = half4(((threadgroup float4 *)src)[lane] * dw);
        } else if (4 * (nr0 / 4) + lane < nr0) {
            dst[4 * (nr0 / 4) + lane] = half(src[4 * (nr0 / 4) + lane] * dw);
        }
    }
}

template [[host_name("qw3_inkling_moe_mapped_nvfp4_n16")]] kernel void qw3_inkling_moe_mapped_nvfp4_mpp<16>(
    constant qw3_inkling_moe_mapped_args &, device const uchar *, device const half *, device half *,
    device const uint *, device const int *, device const uint *, device const float *, threadgroup char *, uint2, ushort, ushort);
template [[host_name("qw3_inkling_moe_mapped_nvfp4_n32")]] kernel void qw3_inkling_moe_mapped_nvfp4_mpp<32>(
    constant qw3_inkling_moe_mapped_args &, device const uchar *, device const half *, device half *,
    device const uint *, device const int *, device const uint *, device const float *, threadgroup char *, uint2, ushort, ushort);
template [[host_name("qw3_inkling_moe_mapped_nvfp4_n64")]] kernel void qw3_inkling_moe_mapped_nvfp4_mpp<64>(
    constant qw3_inkling_moe_mapped_args &, device const uchar *, device const half *, device half *,
    device const uint *, device const int *, device const uint *, device const float *, threadgroup char *, uint2, ushort, ushort);
template [[host_name("qw3_inkling_moe_mapped_nvfp4_n128")]] kernel void qw3_inkling_moe_mapped_nvfp4_mpp<128>(
    constant qw3_inkling_moe_mapped_args &, device const uchar *, device const half *, device half *,
    device const uint *, device const int *, device const uint *, device const float *, threadgroup char *, uint2, ushort, ushort);

// Compile-time specialized non-NVFP4 mapped MoE MPP kernels: NK_STEP=64 (1 full ib32 sub-block
// per thread for IQ3_XXS/IQ2_S/IQ3_S/IQ2_XS/IQ4_NL, 2x fewer barriers, direct half4 writes).
template<uint TYPE>
static inline void qw3_inkling_dequant_32_sa4(
    device const uchar *wrow, uint k0,
    device const ushort *grid16, device const uchar *signs,
    device const uchar *iq3s_kgrid, threadgroup half4 *sa4) {
    if (TYPE == 18u) {
        device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 98ull;
        const float d = float(*(device const half *)blk);
        const uint ib32 = (k0 & 255u) >> 5u;
        device const uchar *q3 = blk + 2u + 8u * ib32;
        device const uchar *sp = blk + 66u + 4u * ib32;
        const uint packed = uint(*(device const ushort *)sp) | (uint(*(device const ushort *)(sp + 2u)) << 16u);
        const float db = d * (0.5f + float(packed >> 28u)) * 0.5f;
        for (uint q8 = 0u; q8 < 4u; q8++) {
            const uint sg = uint(signs[(packed >> (7u * q8)) & 127u]);
            const ushort g0 = grid16[uint(q3[2u * q8])];
            const ushort g1 = grid16[uint(q3[2u * q8 + 1u])];
            sa4[2u * q8] = half4(float4(
                qw3_iq3xxs_grid_value(g0, 0u) * ((sg & 1u) ? -db : db),
                qw3_iq3xxs_grid_value(g0, 1u) * ((sg & 2u) ? -db : db),
                qw3_iq3xxs_grid_value(g0, 2u) * ((sg & 4u) ? -db : db),
                qw3_iq3xxs_grid_value(g0, 3u) * ((sg & 8u) ? -db : db)));
            sa4[2u * q8 + 1u] = half4(float4(
                qw3_iq3xxs_grid_value(g1, 0u) * ((sg & 16u) ? -db : db),
                qw3_iq3xxs_grid_value(g1, 1u) * ((sg & 32u) ? -db : db),
                qw3_iq3xxs_grid_value(g1, 2u) * ((sg & 64u) ? -db : db),
                qw3_iq3xxs_grid_value(g1, 3u) * ((sg & 128u) ? -db : db)));
        }
        return;
    } else if (TYPE == 22u) {
        device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 82ull;
        const float d = float(*(device const half *)blk);
        const uint ib32 = (k0 & 255u) >> 5u;
        const uint scb = uint(blk[74u + ib32]);
        const float db0 = d * (0.5f + float(scb & 15u)) * 0.25f;
        const float db1 = d * (0.5f + float(scb >> 4u)) * 0.25f;
        const uint qh = uint(blk[66u + ib32]);
        device const uchar *qs = blk + 2u + 4u * ib32;
        device const uchar *s2 = qs + 32u;
        for (uint q8 = 0u; q8 < 4u; q8++) {
            const float db = q8 < 2u ? db0 : db1;
            const ushort g = grid16[uint(qs[q8]) | (((qh >> (2u * q8)) & 3u) << 8u)];
            const uint sg = uint(s2[q8]);
            sa4[2u * q8] = half4(float4(
                qw3_iq2_grid_value(g, 0u) * ((sg & 1u) ? -db : db),
                qw3_iq2_grid_value(g, 1u) * ((sg & 2u) ? -db : db),
                qw3_iq2_grid_value(g, 2u) * ((sg & 4u) ? -db : db),
                qw3_iq2_grid_value(g, 3u) * ((sg & 8u) ? -db : db)));
            sa4[2u * q8 + 1u] = half4(float4(
                qw3_iq2_grid_value(g, 4u) * ((sg & 16u) ? -db : db),
                qw3_iq2_grid_value(g, 5u) * ((sg & 32u) ? -db : db),
                qw3_iq2_grid_value(g, 6u) * ((sg & 64u) ? -db : db),
                qw3_iq2_grid_value(g, 7u) * ((sg & 128u) ? -db : db)));
        }
        return;
    } else if (TYPE == 21u) {
        device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 110ull;
        const float d = float(*(device const half *)blk);
        const uint ib32 = (k0 & 255u) >> 5u;
        const float dl = d * float(1u + 2u * ((uint(blk[106u + (ib32 >> 1u)]) >> (4u * (ib32 & 1u))) & 15u));
        const uint qh = uint(blk[66u + ib32]);
        device const uchar *qs = blk + 2u + 8u * ib32;
        device const uchar *sb_p = blk + 74u + 4u * ib32;
        for (uint q8 = 0u; q8 < 4u; q8++) {
            const uint idx0 = uint(qs[2u * q8]) | (((qh >> (2u * q8)) & 1u) << 8u);
            const uint idx1 = uint(qs[2u * q8 + 1u]) | (((qh >> (2u * q8 + 1u)) & 1u) << 8u);
            device const uchar *g0 = iq3s_kgrid + idx0 * 4u;
            device const uchar *g1 = iq3s_kgrid + idx1 * 4u;
            const uint sb = uint(sb_p[q8]);
            sa4[2u * q8] = half4(float4(
                float(g0[0]) * ((sb & 1u) ? -dl : dl),
                float(g0[1]) * ((sb & 2u) ? -dl : dl),
                float(g0[2]) * ((sb & 4u) ? -dl : dl),
                float(g0[3]) * ((sb & 8u) ? -dl : dl)));
            sa4[2u * q8 + 1u] = half4(float4(
                float(g1[0]) * ((sb & 16u) ? -dl : dl),
                float(g1[1]) * ((sb & 32u) ? -dl : dl),
                float(g1[2]) * ((sb & 64u) ? -dl : dl),
                float(g1[3]) * ((sb & 128u) ? -dl : dl)));
        }
        return;
    } else if (TYPE == 17u) {
        device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 74ull;
        const float d = float(*(device const half *)blk);
        const uint ib32 = (k0 & 255u) >> 5u;
        const uint scb = uint(blk[66u + ib32]);
        const float db0 = d * (0.5f + float(scb & 15u)) * 0.25f;
        const float db1 = d * (0.5f + float(scb >> 4u)) * 0.25f;
        device const ushort *q16p = (device const ushort *)(blk + 2u + 8u * ib32);
        for (uint q8 = 0u; q8 < 4u; q8++) {
            const float db = q8 < 2u ? db0 : db1;
            const uint q16 = uint(q16p[q8]);
            const ushort g = grid16[q16 & 511u];
            const uint sg = uint(signs[q16 >> 9u]);
            sa4[2u * q8] = half4(float4(
                qw3_iq2_grid_value(g, 0u) * ((sg & 1u) ? -db : db),
                qw3_iq2_grid_value(g, 1u) * ((sg & 2u) ? -db : db),
                qw3_iq2_grid_value(g, 2u) * ((sg & 4u) ? -db : db),
                qw3_iq2_grid_value(g, 3u) * ((sg & 8u) ? -db : db)));
            sa4[2u * q8 + 1u] = half4(float4(
                qw3_iq2_grid_value(g, 4u) * ((sg & 16u) ? -db : db),
                qw3_iq2_grid_value(g, 5u) * ((sg & 32u) ? -db : db),
                qw3_iq2_grid_value(g, 6u) * ((sg & 64u) ? -db : db),
                qw3_iq2_grid_value(g, 7u) * ((sg & 128u) ? -db : db)));
        }
        return;
    } else if (TYPE == 20u) {
        device const uchar *blk = wrow + uint64_t(k0 >> 5u) * 18ull;
        const float d = float(*(device const half *)blk);
        device const uchar *qs = blk + 2u;
        for (uint i = 0u; i < 4u; i++) {
            const uint b0 = uint(qs[4u * i]), b1 = uint(qs[4u * i + 1u]);
            const uint b2 = uint(qs[4u * i + 2u]), b3 = uint(qs[4u * i + 3u]);
            sa4[i] = half4(float4(qw3_iq4nl_table[b0 & 15u], qw3_iq4nl_table[b1 & 15u],
                                  qw3_iq4nl_table[b2 & 15u], qw3_iq4nl_table[b3 & 15u]) * d);
            sa4[4u + i] = half4(float4(qw3_iq4nl_table[b0 >> 4u], qw3_iq4nl_table[b1 >> 4u],
                                       qw3_iq4nl_table[b2 >> 4u], qw3_iq4nl_table[b3 >> 4u]) * d);
        }
        return;
    } else if (TYPE == 11u) {
        device const uchar *blk = wrow + uint64_t(k0 >> 8u) * 110ull;
        device const ushort *pk = (device const ushort *)(blk + 96u);
        const uint aux0 = uint(pk[0]) | (uint(pk[1]) << 16u);
        const uint aux1 = uint(pk[2]) | (uint(pk[3]) << 16u);
        const uint tmp  = uint(pk[4]) | (uint(pk[5]) << 16u);
        const uint kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
        uint aux[4];
        aux[0] = (aux0 & kmask2) | (((tmp >> 0u) & kmask1) << 4u);
        aux[1] = (aux1 & kmask2) | (((tmp >> 2u) & kmask1) << 4u);
        aux[2] = ((aux0 >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u);
        aux[3] = ((aux1 >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u);
        const float d = float(*(device const half *)(blk + 108u));
        const uint local0 = k0 & 255u;
        const uint half_block = local0 >> 7u;
        const uint grp = (local0 & 127u) >> 5u;
        const uint si0 = half_block * 8u + grp * 2u;
        const uint si1 = si0 + 1u;
        const float dl0 = d * float(int((aux[si0 >> 2u] >> (8u * (si0 & 3u))) & 255u) - 32);
        const float dl1 = d * float(int((aux[si1 >> 2u] >> (8u * (si1 & 3u))) & 255u) - 32);
        const uint shift = grp * 2u;
        const uint mask = 1u << (half_block * 4u + grp);
        device const uchar *qs = blk + 32u + half_block * 32u;
        for (uint g = 0u; g < 4u; g++) {
            const uint l0 = g * 4u;
            sa4[g] = half4(float4(
                float(int((uint(qs[l0 + 0u]) >> shift) & 3u) - ((uint(blk[l0 + 0u]) & mask) ? 0 : 4)),
                float(int((uint(qs[l0 + 1u]) >> shift) & 3u) - ((uint(blk[l0 + 1u]) & mask) ? 0 : 4)),
                float(int((uint(qs[l0 + 2u]) >> shift) & 3u) - ((uint(blk[l0 + 2u]) & mask) ? 0 : 4)),
                float(int((uint(qs[l0 + 3u]) >> shift) & 3u) - ((uint(blk[l0 + 3u]) & mask) ? 0 : 4))) * dl0);
            const uint u0 = 16u + l0;
            sa4[4u + g] = half4(float4(
                float(int((uint(qs[u0 + 0u]) >> shift) & 3u) - ((uint(blk[u0 + 0u]) & mask) ? 0 : 4)),
                float(int((uint(qs[u0 + 1u]) >> shift) & 3u) - ((uint(blk[u0 + 1u]) & mask) ? 0 : 4)),
                float(int((uint(qs[u0 + 2u]) >> shift) & 3u) - ((uint(blk[u0 + 2u]) & mask) ? 0 : 4)),
                float(int((uint(qs[u0 + 3u]) >> shift) & 3u) - ((uint(blk[u0 + 3u]) & mask) ? 0 : 4))) * dl1);
        }
        return;
    }
    half v0[16], v1[16];
    qw3_inkling_dequant_16_generic(wrow, TYPE, k0, grid16, signs, iq3s_kgrid, v0);
    qw3_inkling_dequant_16_generic(wrow, TYPE, k0 + 16u, grid16, signs, iq3s_kgrid, v1);
    for (int i = 0; i < 4; i++) {
        sa4[i] = half4(v0[4 * i], v0[4 * i + 1], v0[4 * i + 2], v0[4 * i + 3]);
        sa4[4 + i] = half4(v1[4 * i], v1[4 * i + 1], v1[4 * i + 2], v1[4 * i + 3]);
    }
}

template<uint TYPE, int NR1>
kernel void qw3_inkling_moe_mapped_typed_mpp(
    constant qw3_inkling_moe_mapped_args &args [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const half *xin [[buffer(2)]],
    device half *out_slots [[buffer(3)]],
    device const uint *counts [[buffer(4)]],
    device const int *pair_ids [[buffer(5)]],
    device const uint *block_ids [[buffer(6)]],
    device const ushort *grid16 [[buffer(7)]],
    device const uchar *signs [[buffer(8)]],
    device const uchar *iq3s_kgrid [[buffer(9)]],
    device const float *down_weights [[buffer(10)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int NK_STEP = 64;
    constexpr int NK = 32;
    const uint r0u = group.x * uint(NR0);
    const uint block = block_ids[group.y];
    const uint expert = block & 255u;
    const uint r1u = block >> 8u;
    const uint count = counts[expert];
    if (r0u >= args.n_out || r1u >= count) return;
    const uint exp_off = counts[128u + expert];
    const int nr0 = int(min(uint(NR0), args.n_out - r0u));
    const int nr1 = int(min(uint(NR1), count - r1u));
    const uint map_base = expert * args.n_tokens + r1u;
    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup float *sc = (threadgroup float *)shmem;
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        sa, dextents<int32_t, 2>(NK_STEP, NR0));
    device half *ptrB = (device half *)xin + uint64_t(exp_off) * args.in_stride;
    auto tB = tensor(ptrB, dextents<int32_t, 2>(int(args.n_in), int(count)),
                     array<int, 2>({1, int(args.in_stride)}));
    device const uchar *wexp = weights + uint64_t(expert) * uint64_t(args.expert_bytes);
    const int row = int(tid) >> 1;
    const int half_k = int(tid) & 1;
    device const uchar *wrow = (row < nr0) ? (wexp + uint64_t(r0u + uint(row)) * args.row_bytes) : wexp;
    threadgroup half4 *sa4 = (threadgroup half4 *)(sa + row * NK_STEP + half_k * 32);
    const int lane = int(tid & 31u);

    if (NR1 >= 64) {
        constexpr int ROWS_PER_SG = (NR1 == 128) ? 32 : 16;
        const int sg_r1 = int(r1u) + int(sgitg) * ROWS_PER_SG;
        const int sg_nr1 = clamp(int(count) - sg_r1, 0, ROWS_PER_SG);
        const bool sg_active = sg_nr1 > 0;
        matmul2d<matmul2d_descriptor(ROWS_PER_SG, NR0, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
            execution_simdgroups<1>> mm;
        auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
        for (uint16_t i = 0; i < cT.get_capacity(); i++)
            if (cT.is_valid_element(i)) cT[i] = 0.0f;
        for (uint loop_k = 0; loop_k < args.n_in; loop_k += uint(NK_STEP)) {
            if (row < nr0) {
                qw3_inkling_dequant_32_sa4<TYPE>(
                    wrow, loop_k + uint(half_k * 32), grid16, signs, iq3s_kgrid, sa4);
            } else {
                for (int i = 0; i < 8; i++) sa4[i] = half4(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg_active) {
                auto mA0 = tA.slice(0, 0);
                auto mB0 = tB.slice(int(loop_k), sg_r1);
                mm.run(mB0, mA0, cT);
                auto mA1 = tA.slice(32, 0);
                auto mB1 = tB.slice(int(loop_k + 32u), sg_r1);
                mm.run(mB1, mA1, cT);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        constexpr int WAVES = (NR1 == 128) ? 4 : 2;
        constexpr int SGS_PER_WAVE = 4 / WAVES;
        for (int w = 0; w < WAVES; w++) {
            const int sg_in_w = int(sgitg) - w * SGS_PER_WAVE;
            if (sg_in_w >= 0 && sg_in_w < SGS_PER_WAVE && sg_active) {
                threadgroup float *sc_sg = sc + sg_in_w * (ROWS_PER_SG * NR0);
                auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
                    sc_sg, dextents<int32_t, 2>(NR0, ROWS_PER_SG));
                cT.store(tC.slice(0, 0));
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (int j = 0; j < sg_nr1; j++) {
                    const uint pid = uint(pair_ids[expert * args.n_tokens + uint(sg_r1 + j)]);
                    const float dw = down_weights[pid];
                    device half *dst = out_slots + uint64_t(pid) * args.out_stride + r0u;
                    threadgroup float *src = sc_sg + j * NR0;
                    if (lane < nr0 / 4) {
                        ((device half4 *)dst)[lane] = half4(((threadgroup float4 *)src)[lane] * dw);
                    } else if (4 * (nr0 / 4) + lane < nr0) {
                        dst[4 * (nr0 / 4) + lane] = half(src[4 * (nr0 / 4) + lane] * dw);
                    }
                }
            }
            if (w + 1 < WAVES) threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        return;
    }

    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); i++)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    for (uint loop_k = 0; loop_k < args.n_in; loop_k += uint(NK_STEP)) {
        if (row < nr0) {
            qw3_inkling_dequant_32_sa4<TYPE>(
                wrow, loop_k + uint(half_k * 32), grid16, signs, iq3s_kgrid, sa4);
        } else {
            for (int i = 0; i < 8; i++) sa4[i] = half4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            auto mA0 = tA.slice(0, 0);
            auto mB0 = tB.slice(int(loop_k), int(r1u));
            mm.run(mB0, mA0, cT);
            auto mA1 = tA.slice(32, 0);
            auto mB1 = tB.slice(int(loop_k + 32u), int(r1u));
            mm.run(mB1, mA1, cT);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        sc, dextents<int32_t, 2>(NR0, NR1));
    cT.store(tC.slice(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = int(sgitg); j < nr1; j += 4) {
        const uint pid = uint(pair_ids[map_base + uint(j)]);
        const float dw = down_weights[pid];
        device half *dst = out_slots + uint64_t(pid) * args.out_stride + r0u;
        threadgroup float *src = sc + j * NR0;
        if (lane < nr0 / 4) {
            ((device half4 *)dst)[lane] = half4(((threadgroup float4 *)src)[lane] * dw);
        } else if (4 * (nr0 / 4) + lane < nr0) {
            dst[4 * (nr0 / 4) + lane] = half(src[4 * (nr0 / 4) + lane] * dw);
        }
    }
}

#define QW3_INST_INK_MOE_MPP(T) \
    template [[host_name("qw3_inkling_moe_mapped_t" #T "_n16")]] kernel void qw3_inkling_moe_mapped_typed_mpp<T##u, 16>( \
        constant qw3_inkling_moe_mapped_args &, device const uchar *, device const half *, device half *, \
        device const uint *, device const int *, device const uint *, \
        device const ushort *, device const uchar *, device const uchar *, device const float *, threadgroup char *, uint2, ushort, ushort); \
    template [[host_name("qw3_inkling_moe_mapped_t" #T "_n32")]] kernel void qw3_inkling_moe_mapped_typed_mpp<T##u, 32>( \
        constant qw3_inkling_moe_mapped_args &, device const uchar *, device const half *, device half *, \
        device const uint *, device const int *, device const uint *, \
        device const ushort *, device const uchar *, device const uchar *, device const float *, threadgroup char *, uint2, ushort, ushort); \
    template [[host_name("qw3_inkling_moe_mapped_t" #T "_n64")]] kernel void qw3_inkling_moe_mapped_typed_mpp<T##u, 64>( \
        constant qw3_inkling_moe_mapped_args &, device const uchar *, device const half *, device half *, \
        device const uint *, device const int *, device const uint *, \
        device const ushort *, device const uchar *, device const uchar *, device const float *, threadgroup char *, uint2, ushort, ushort); \
    template [[host_name("qw3_inkling_moe_mapped_t" #T "_n128")]] kernel void qw3_inkling_moe_mapped_typed_mpp<T##u, 128>( \
        constant qw3_inkling_moe_mapped_args &, device const uchar *, device const half *, device half *, \
        device const uint *, device const int *, device const uint *, \
        device const ushort *, device const uchar *, device const uchar *, device const float *, threadgroup char *, uint2, ushort, ushort);

QW3_INST_INK_MOE_MPP(8)
QW3_INST_INK_MOE_MPP(11)
QW3_INST_INK_MOE_MPP(12)
QW3_INST_INK_MOE_MPP(13)
QW3_INST_INK_MOE_MPP(14)
QW3_INST_INK_MOE_MPP(17)
QW3_INST_INK_MOE_MPP(18)
QW3_INST_INK_MOE_MPP(20)
QW3_INST_INK_MOE_MPP(21)
QW3_INST_INK_MOE_MPP(22)
QW3_INST_INK_MOE_MPP(23)
#undef QW3_INST_INK_MOE_MPP

// Direct-tensor zero-sb fused gate_up + SiTU prefill kernels: 8 KiB shmem for ALL NR1 (16/32/64),
// reads expert-contiguous FP16 x directly via Metal 4 NAX tensor, writes expert-contiguous FP16 hidden!
struct qw3_inkling_moe_gate_up_situ_args {
    uint n_in, n_ff, row_bytes, expert_bytes;
    uint n_tokens, n_active, type, in_stride;
    float beta, limit;
};

template<int NR1>
kernel void qw3_inkling_moe_gate_up_situ_nvfp4_mpp(
    constant qw3_inkling_moe_gate_up_situ_args &args [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const half *xin [[buffer(2)]],
    device half *hidden_out [[buffer(3)]],
    device const uint *counts [[buffer(4)]],
    device const int *pair_ids [[buffer(5)]],
    device const uint *block_ids [[buffer(6)]],
    device const float *gate_scales [[buffer(7)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0_H = 32;
    constexpr int NR0 = 64;
    constexpr int NK_STEP = 64;
    constexpr int NK = 32;
    const uint r0u = group.x * uint(NR0_H);
    const uint block = block_ids[group.y];
    const uint expert = block & 255u;
    const uint r1u = block >> 8u;
    const uint count = counts[expert];
    if (r0u >= args.n_ff || r1u >= count) return;
    const uint exp_off = counts[128u + expert];
    const int nr0 = int(min(uint(NR0_H), args.n_ff - r0u));
    const int nr1 = int(min(uint(NR1), count - r1u));
    const uint map_base = expert * args.n_tokens + r1u;
    threadgroup half *sa   = (threadgroup half *)shmem;
    threadgroup half *sa_g = sa;
    threadgroup half *sa_u = sa + NR0_H * NK_STEP;
    threadgroup float *sc  = (threadgroup float *)shmem;
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        sa, dextents<int32_t, 2>(NK_STEP, NR0));
    device half *ptrB = (device half *)xin + uint64_t(exp_off) * args.in_stride;
    auto tB = tensor(ptrB, dextents<int32_t, 2>(int(args.n_in), int(count)),
                     array<int, 2>({1, int(args.in_stride)}));
    device const uchar *wexp = weights + uint64_t(expert) * uint64_t(args.expert_bytes);
    const int row = int(tid) >> 2;
    const uint sub = uint(tid) & 3u;
    device const uchar *wrow_g = (row < nr0) ? (wexp + uint64_t(r0u + uint(row)) * args.row_bytes) : wexp;
    device const uchar *wrow_u = (row < nr0) ? (wexp + uint64_t(args.n_ff + r0u + uint(row)) * args.row_bytes) : wexp;
    threadgroup half4 *sag4 = (threadgroup half4 *)(sa_g + row * NK_STEP + int(sub) * 16);
    threadgroup half4 *sau4 = (threadgroup half4 *)(sa_u + row * NK_STEP + int(sub) * 16);
    const int lane = int(tid & 31u);
    const float beta = args.beta, limit = args.limit;

    if (NR1 >= 64) {
        constexpr int ROWS_PER_SG = (NR1 == 128) ? 32 : 16;
        const int sg_r1 = int(r1u) + int(sgitg) * ROWS_PER_SG;
        const int sg_nr1 = clamp(int(count) - sg_r1, 0, ROWS_PER_SG);
        const bool sg_active = sg_nr1 > 0;
        matmul2d<matmul2d_descriptor(ROWS_PER_SG, NR0, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
            execution_simdgroups<1>> mm;
        auto cGU = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
        for (uint16_t i = 0; i < cGU.get_capacity(); i++)
            if (cGU.is_valid_element(i)) cGU[i] = 0.0f;
        for (uint loop_k = 0; loop_k < args.n_in; loop_k += uint(NK_STEP)) {
            if (row < nr0) {
                const uint64_t boff = uint64_t(loop_k >> 6u) * 36ull;
                {
                    device const uchar *blk = wrow_g + boff;
                    const half sc0 = qw3_nvfp4_scale_h[blk[sub]];
                    device const uint *vp = (device const uint *)(blk + 4u + 8u * sub);
                    const uint w0 = vp[0], w1 = vp[1];
                    sag4[0] = qw3_nvfp4_decode4_h(w0 & 15u, (w0 >> 8u) & 15u, (w0 >> 16u) & 15u, (w0 >> 24u) & 15u) * sc0;
                    sag4[1] = qw3_nvfp4_decode4_h(w1 & 15u, (w1 >> 8u) & 15u, (w1 >> 16u) & 15u, (w1 >> 24u) & 15u) * sc0;
                    sag4[2] = qw3_nvfp4_decode4_h((w0 >> 4u) & 15u, (w0 >> 12u) & 15u, (w0 >> 20u) & 15u, (w0 >> 28u) & 15u) * sc0;
                    sag4[3] = qw3_nvfp4_decode4_h((w1 >> 4u) & 15u, (w1 >> 12u) & 15u, (w1 >> 20u) & 15u, (w1 >> 28u) & 15u) * sc0;
                }
                {
                    device const uchar *blk = wrow_u + boff;
                    const half sc0 = qw3_nvfp4_scale_h[blk[sub]];
                    device const uint *vp = (device const uint *)(blk + 4u + 8u * sub);
                    const uint w0 = vp[0], w1 = vp[1];
                    sau4[0] = qw3_nvfp4_decode4_h(w0 & 15u, (w0 >> 8u) & 15u, (w0 >> 16u) & 15u, (w0 >> 24u) & 15u) * sc0;
                    sau4[1] = qw3_nvfp4_decode4_h(w1 & 15u, (w1 >> 8u) & 15u, (w1 >> 16u) & 15u, (w1 >> 24u) & 15u) * sc0;
                    sau4[2] = qw3_nvfp4_decode4_h((w0 >> 4u) & 15u, (w0 >> 12u) & 15u, (w0 >> 20u) & 15u, (w0 >> 28u) & 15u) * sc0;
                    sau4[3] = qw3_nvfp4_decode4_h((w1 >> 4u) & 15u, (w1 >> 12u) & 15u, (w1 >> 20u) & 15u, (w1 >> 28u) & 15u) * sc0;
                }
            } else {
                for (int i = 0; i < 4; i++) { sag4[i] = half4(0.0f); sau4[i] = half4(0.0f); }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg_active) {
                auto mA0 = tA.slice(0, 0);
                auto mB0 = tB.slice(int(loop_k), sg_r1);
                mm.run(mB0, mA0, cGU);
                auto mA1 = tA.slice(32, 0);
                auto mB1 = tB.slice(int(loop_k + 32u), sg_r1);
                mm.run(mB1, mA1, cGU);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        constexpr int WAVES = (NR1 == 128) ? 4 : 2;
        constexpr int SGS_PER_WAVE = 4 / WAVES;
        for (int w = 0; w < WAVES; w++) {
            const int sg_in_w = int(sgitg) - w * SGS_PER_WAVE;
            if (sg_in_w >= 0 && sg_in_w < SGS_PER_WAVE && sg_active) {
                threadgroup float *sc_sg = sc + sg_in_w * (ROWS_PER_SG * NR0);
                auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
                    sc_sg, dextents<int32_t, 2>(NR0, ROWS_PER_SG));
                cGU.store(tC.slice(0, 0));
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (int j = 0; j < sg_nr1; j++) {
                    const uint pid = uint(pair_ids[expert * args.n_tokens + uint(sg_r1 + j)]);
                    const float sc_w = gate_scales[pid];
                    if (lane < nr0) {
                        const float ga = sc_sg[j * NR0 + lane] * sc_w;
                        const float up = sc_sg[j * NR0 + NR0_H + lane] * sc_w;
                        hidden_out[uint64_t(exp_off + uint(sg_r1 + j)) * args.n_ff + r0u + uint(lane)] =
                            half(beta * tanh(ga / beta) / (1.0f + exp(-ga)) * limit * tanh(up / limit));
                    }
                }
            }
            if (w + 1 < WAVES) threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        return;
    }

    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cGU = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cGU.get_capacity(); i++)
        if (cGU.is_valid_element(i)) cGU[i] = 0.0f;
    for (uint loop_k = 0; loop_k < args.n_in; loop_k += uint(NK_STEP)) {
        if (row < nr0) {
            const uint64_t boff = uint64_t(loop_k >> 6u) * 36ull;
            {
                device const uchar *blk = wrow_g + boff;
                const half sc0 = qw3_nvfp4_scale_h[blk[sub]];
                device const uint *vp = (device const uint *)(blk + 4u + 8u * sub);
                const uint w0 = vp[0], w1 = vp[1];
                sag4[0] = qw3_nvfp4_decode4_h(w0 & 15u, (w0 >> 8u) & 15u, (w0 >> 16u) & 15u, (w0 >> 24u) & 15u) * sc0;
                sag4[1] = qw3_nvfp4_decode4_h(w1 & 15u, (w1 >> 8u) & 15u, (w1 >> 16u) & 15u, (w1 >> 24u) & 15u) * sc0;
                sag4[2] = qw3_nvfp4_decode4_h((w0 >> 4u) & 15u, (w0 >> 12u) & 15u, (w0 >> 20u) & 15u, (w0 >> 28u) & 15u) * sc0;
                sag4[3] = qw3_nvfp4_decode4_h((w1 >> 4u) & 15u, (w1 >> 12u) & 15u, (w1 >> 20u) & 15u, (w1 >> 28u) & 15u) * sc0;
            }
            {
                device const uchar *blk = wrow_u + boff;
                const half sc0 = qw3_nvfp4_scale_h[blk[sub]];
                device const uint *vp = (device const uint *)(blk + 4u + 8u * sub);
                const uint w0 = vp[0], w1 = vp[1];
                sau4[0] = qw3_nvfp4_decode4_h(w0 & 15u, (w0 >> 8u) & 15u, (w0 >> 16u) & 15u, (w0 >> 24u) & 15u) * sc0;
                sau4[1] = qw3_nvfp4_decode4_h(w1 & 15u, (w1 >> 8u) & 15u, (w1 >> 16u) & 15u, (w1 >> 24u) & 15u) * sc0;
                sau4[2] = qw3_nvfp4_decode4_h((w0 >> 4u) & 15u, (w0 >> 12u) & 15u, (w0 >> 20u) & 15u, (w0 >> 28u) & 15u) * sc0;
                sau4[3] = qw3_nvfp4_decode4_h((w1 >> 4u) & 15u, (w1 >> 12u) & 15u, (w1 >> 20u) & 15u, (w1 >> 28u) & 15u) * sc0;
            }
        } else {
            for (int i = 0; i < 4; i++) { sag4[i] = half4(0.0f); sau4[i] = half4(0.0f); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            auto mA0 = tA.slice(0, 0);
            auto mB0 = tB.slice(int(loop_k), int(r1u));
            mm.run(mB0, mA0, cGU);
            auto mA1 = tA.slice(32, 0);
            auto mB1 = tB.slice(int(loop_k + 32u), int(r1u));
            mm.run(mB1, mA1, cGU);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        sc, dextents<int32_t, 2>(NR0, NR1));
    cGU.store(tC.slice(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = int(sgitg); j < nr1; j += 4) {
        const uint pid = uint(pair_ids[map_base + uint(j)]);
        const float sc_w = gate_scales[pid];
        if (lane < nr0) {
            const float ga = sc[j * NR0 + lane] * sc_w;
            const float up = sc[j * NR0 + NR0_H + lane] * sc_w;
            hidden_out[uint64_t(exp_off + r1u + uint(j)) * args.n_ff + r0u + uint(lane)] =
                half(beta * tanh(ga / beta) / (1.0f + exp(-ga)) * limit * tanh(up / limit));
        }
    }
}

template [[host_name("qw3_inkling_moe_gate_up_situ_nvfp4_n16")]] kernel void qw3_inkling_moe_gate_up_situ_nvfp4_mpp<16>(
    constant qw3_inkling_moe_gate_up_situ_args &, device const uchar *, device const half *, device half *,
    device const uint *, device const int *, device const uint *, device const float *, threadgroup char *, uint2, ushort, ushort);
template [[host_name("qw3_inkling_moe_gate_up_situ_nvfp4_n32")]] kernel void qw3_inkling_moe_gate_up_situ_nvfp4_mpp<32>(
    constant qw3_inkling_moe_gate_up_situ_args &, device const uchar *, device const half *, device half *,
    device const uint *, device const int *, device const uint *, device const float *, threadgroup char *, uint2, ushort, ushort);
template [[host_name("qw3_inkling_moe_gate_up_situ_nvfp4_n64")]] kernel void qw3_inkling_moe_gate_up_situ_nvfp4_mpp<64>(
    constant qw3_inkling_moe_gate_up_situ_args &, device const uchar *, device const half *, device half *,
    device const uint *, device const int *, device const uint *, device const float *, threadgroup char *, uint2, ushort, ushort);
template [[host_name("qw3_inkling_moe_gate_up_situ_nvfp4_n128")]] kernel void qw3_inkling_moe_gate_up_situ_nvfp4_mpp<128>(
    constant qw3_inkling_moe_gate_up_situ_args &, device const uchar *, device const half *, device half *,
    device const uint *, device const int *, device const uint *, device const float *, threadgroup char *, uint2, ushort, ushort);

template<uint TYPE, int NR1>
kernel void qw3_inkling_moe_gate_up_situ_typed_mpp(
    constant qw3_inkling_moe_gate_up_situ_args &args [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const half *xin [[buffer(2)]],
    device half *hidden_out [[buffer(3)]],
    device const uint *counts [[buffer(4)]],
    device const int *pair_ids [[buffer(5)]],
    device const uint *block_ids [[buffer(6)]],
    device const ushort *grid16 [[buffer(7)]],
    device const uchar *signs [[buffer(8)]],
    device const uchar *iq3s_kgrid [[buffer(9)]],
    device const float *gate_scales [[buffer(10)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0_H = 32;
    constexpr int NR0 = 64;
    constexpr int NK_STEP = 64;
    constexpr int NK = 32;
    const uint r0u = group.x * uint(NR0_H);
    const uint block = block_ids[group.y];
    const uint expert = block & 255u;
    const uint r1u = block >> 8u;
    const uint count = counts[expert];
    if (r0u >= args.n_ff || r1u >= count) return;
    const uint exp_off = counts[128u + expert];
    const int nr0 = int(min(uint(NR0_H), args.n_ff - r0u));
    const int nr1 = int(min(uint(NR1), count - r1u));
    const uint map_base = expert * args.n_tokens + r1u;
    threadgroup half *sa   = (threadgroup half *)shmem;
    threadgroup half *sa_g = sa;
    threadgroup half *sa_u = sa + NR0_H * NK_STEP;
    threadgroup float *sc  = (threadgroup float *)shmem;
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        sa, dextents<int32_t, 2>(NK_STEP, NR0));
    device half *ptrB = (device half *)xin + uint64_t(exp_off) * args.in_stride;
    auto tB = tensor(ptrB, dextents<int32_t, 2>(int(args.n_in), int(count)),
                     array<int, 2>({1, int(args.in_stride)}));
    device const uchar *wexp = weights + uint64_t(expert) * uint64_t(args.expert_bytes);
    const int row_all = int(tid) >> 1;
    const int half_k = int(tid) & 1;
    const bool is_up = row_all >= NR0_H;
    const int row_h = is_up ? (row_all - NR0_H) : row_all;
    const uint w_row_idx = (is_up ? args.n_ff : 0u) + r0u + uint(row_h);
    device const uchar *wrow = (row_h < nr0) ? (wexp + uint64_t(w_row_idx) * args.row_bytes) : wexp;
    threadgroup half *sa_dst = is_up ? sa_u : sa_g;
    threadgroup half4 *sa4 = (threadgroup half4 *)(sa_dst + row_h * NK_STEP + half_k * 32);
    const int lane = int(tid & 31u);
    const float beta = args.beta, limit = args.limit;

    if (NR1 >= 64) {
        constexpr int ROWS_PER_SG = (NR1 == 128) ? 32 : 16;
        const int sg_r1 = int(r1u) + int(sgitg) * ROWS_PER_SG;
        const int sg_nr1 = clamp(int(count) - sg_r1, 0, ROWS_PER_SG);
        const bool sg_active = sg_nr1 > 0;
        matmul2d<matmul2d_descriptor(ROWS_PER_SG, NR0, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
            execution_simdgroups<1>> mm;
        auto cGU = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
        for (uint16_t i = 0; i < cGU.get_capacity(); i++)
            if (cGU.is_valid_element(i)) cGU[i] = 0.0f;
        for (uint loop_k = 0; loop_k < args.n_in; loop_k += uint(NK_STEP)) {
            if (row_h < nr0) {
                qw3_inkling_dequant_32_sa4<TYPE>(
                    wrow, loop_k + uint(half_k * 32), grid16, signs, iq3s_kgrid, sa4);
            } else {
                for (int i = 0; i < 8; i++) sa4[i] = half4(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg_active) {
                auto mA0 = tA.slice(0, 0);
                auto mB0 = tB.slice(int(loop_k), sg_r1);
                mm.run(mB0, mA0, cGU);
                auto mA1 = tA.slice(32, 0);
                auto mB1 = tB.slice(int(loop_k + 32u), sg_r1);
                mm.run(mB1, mA1, cGU);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        constexpr int WAVES = (NR1 == 128) ? 4 : 2;
        constexpr int SGS_PER_WAVE = 4 / WAVES;
        for (int w = 0; w < WAVES; w++) {
            const int sg_in_w = int(sgitg) - w * SGS_PER_WAVE;
            if (sg_in_w >= 0 && sg_in_w < SGS_PER_WAVE && sg_active) {
                threadgroup float *sc_sg = sc + sg_in_w * (ROWS_PER_SG * NR0);
                auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
                    sc_sg, dextents<int32_t, 2>(NR0, ROWS_PER_SG));
                cGU.store(tC.slice(0, 0));
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (int j = 0; j < sg_nr1; j++) {
                    const uint pid = uint(pair_ids[expert * args.n_tokens + uint(sg_r1 + j)]);
                    const float sc_w = gate_scales[pid];
                    if (lane < nr0) {
                        const float ga = sc_sg[j * NR0 + lane] * sc_w;
                        const float up = sc_sg[j * NR0 + NR0_H + lane] * sc_w;
                        hidden_out[uint64_t(exp_off + uint(sg_r1 + j)) * args.n_ff + r0u + uint(lane)] =
                            half(beta * tanh(ga / beta) / (1.0f + exp(-ga)) * limit * tanh(up / limit));
                    }
                }
            }
            if (w + 1 < WAVES) threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        return;
    }

    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cGU = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cGU.get_capacity(); i++)
        if (cGU.is_valid_element(i)) cGU[i] = 0.0f;
    for (uint loop_k = 0; loop_k < args.n_in; loop_k += uint(NK_STEP)) {
        if (row_h < nr0) {
            qw3_inkling_dequant_32_sa4<TYPE>(
                wrow, loop_k + uint(half_k * 32), grid16, signs, iq3s_kgrid, sa4);
        } else {
            for (int i = 0; i < 8; i++) sa4[i] = half4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            auto mA0 = tA.slice(0, 0);
            auto mB0 = tB.slice(int(loop_k), int(r1u));
            mm.run(mB0, mA0, cGU);
            auto mA1 = tA.slice(32, 0);
            auto mB1 = tB.slice(int(loop_k + 32u), int(r1u));
            mm.run(mB1, mA1, cGU);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        sc, dextents<int32_t, 2>(NR0, NR1));
    cGU.store(tC.slice(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = int(sgitg); j < nr1; j += 4) {
        const uint pid = uint(pair_ids[map_base + uint(j)]);
        const float sc_w = gate_scales[pid];
        if (lane < nr0) {
            const float ga = sc[j * NR0 + lane] * sc_w;
            const float up = sc[j * NR0 + NR0_H + lane] * sc_w;
            hidden_out[uint64_t(exp_off + r1u + uint(j)) * args.n_ff + r0u + uint(lane)] =
                half(beta * tanh(ga / beta) / (1.0f + exp(-ga)) * limit * tanh(up / limit));
        }
    }
}

#define QW3_INST_INK_MOE_GU(T) \
    template [[host_name("qw3_inkling_moe_gate_up_situ_t" #T "_n16")]] kernel void qw3_inkling_moe_gate_up_situ_typed_mpp<T##u, 16>( \
        constant qw3_inkling_moe_gate_up_situ_args &, device const uchar *, device const half *, device half *, \
        device const uint *, device const int *, device const uint *, \
        device const ushort *, device const uchar *, device const uchar *, device const float *, threadgroup char *, uint2, ushort, ushort); \
    template [[host_name("qw3_inkling_moe_gate_up_situ_t" #T "_n32")]] kernel void qw3_inkling_moe_gate_up_situ_typed_mpp<T##u, 32>( \
        constant qw3_inkling_moe_gate_up_situ_args &, device const uchar *, device const half *, device half *, \
        device const uint *, device const int *, device const uint *, \
        device const ushort *, device const uchar *, device const uchar *, device const float *, threadgroup char *, uint2, ushort, ushort); \
    template [[host_name("qw3_inkling_moe_gate_up_situ_t" #T "_n64")]] kernel void qw3_inkling_moe_gate_up_situ_typed_mpp<T##u, 64>( \
        constant qw3_inkling_moe_gate_up_situ_args &, device const uchar *, device const half *, device half *, \
        device const uint *, device const int *, device const uint *, \
        device const ushort *, device const uchar *, device const uchar *, device const float *, threadgroup char *, uint2, ushort, ushort); \
    template [[host_name("qw3_inkling_moe_gate_up_situ_t" #T "_n128")]] kernel void qw3_inkling_moe_gate_up_situ_typed_mpp<T##u, 128>( \
        constant qw3_inkling_moe_gate_up_situ_args &, device const uchar *, device const half *, device half *, \
        device const uint *, device const int *, device const uint *, \
        device const ushort *, device const uchar *, device const uchar *, device const float *, threadgroup char *, uint2, ushort, ushort);

QW3_INST_INK_MOE_GU(8)
QW3_INST_INK_MOE_GU(11)
QW3_INST_INK_MOE_GU(12)
QW3_INST_INK_MOE_GU(13)
QW3_INST_INK_MOE_GU(14)
QW3_INST_INK_MOE_GU(17)
QW3_INST_INK_MOE_GU(18)
QW3_INST_INK_MOE_GU(20)
QW3_INST_INK_MOE_GU(21)
QW3_INST_INK_MOE_GU(22)
QW3_INST_INK_MOE_GU(23)
#undef QW3_INST_INK_MOE_GU

// Metal 4 NAX tiled loop control apply: NR1=32, STAGE_K=64 (entire rank=64 in 1 barrier/mat),
// direct tB (code in device memory), 1-wave parallel epilogue, zero register spill.
kernel void qw3_inkling_control_apply_batch_mpp(
    constant uint4 &shape [[buffer(0)]],
    device const float *gain [[buffer(1)]],
    device const float *forget [[buffer(2)]],
    device const float *code [[buffer(3)]],
    device const float *src_state [[buffer(4)]],
    device const float *message [[buffer(5)]],
    device float *dst_state [[buffer(6)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int STAGE_K = 64;
    const uint width = shape.x, rank = shape.y, n_tokens = shape.z;
    const uint r0 = group.x * uint(NR0);
    const uint t0 = group.y * uint(NR1);
    if (r0 >= width || t0 >= n_tokens) return;
    const int nr0 = int(min(uint(NR0), width - r0));
    const int nr1 = int(min(uint(NR1), n_tokens - t0));
    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        sa, dextents<int32_t, 2>(STAGE_K, NR0));
    device float *ptrB = (device float *)code;
    auto tB = tensor(ptrB, dextents<int32_t, 2>(int(rank), int(n_tokens)),
                     array<int, 2>({1, int(rank)}));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cG = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    auto cF = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cG.get_capacity(); i++) {
        if (cG.is_valid_element(i)) cG[i] = 0.0f;
        if (cF.is_valid_element(i)) cF[i] = 0.0f;
    }
    const int row = int(tid) >> 1;
    const int half_k = int(tid) & 1;
    threadgroup half4 *sa4 = (threadgroup half4 *)(sa + row * STAGE_K + half_k * 32);
    for (uint stage_k = 0; stage_k < rank; stage_k += uint(STAGE_K)) {
        if (row < nr0) {
            device const float4 *gf = (device const float4 *)(gain + uint64_t(r0 + uint(row)) * rank + stage_k + uint(half_k * 32));
            for (int i = 0; i < 8; i++) sa4[i] = half4(gf[i]);
        } else {
            for (int i = 0; i < 8; i++) sa4[i] = half4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            auto mB0 = tB.slice(int(stage_k), int(t0));
            auto mA0 = tA.slice(0, 0);
            mm.run(mB0, mA0, cG);
            auto mB1 = tB.slice(int(stage_k + 32u), int(t0));
            auto mA1 = tA.slice(32, 0);
            mm.run(mB1, mA1, cG);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < nr0) {
            device const float4 *ff = (device const float4 *)(forget + uint64_t(r0 + uint(row)) * rank + stage_k + uint(half_k * 32));
            for (int i = 0; i < 8; i++) sa4[i] = half4(ff[i]);
        } else {
            for (int i = 0; i < 8; i++) sa4[i] = half4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            auto mB0 = tB.slice(int(stage_k), int(t0));
            auto mA0 = tA.slice(0, 0);
            mm.run(mB0, mA0, cF);
            auto mB1 = tB.slice(int(stage_k + 32u), int(t0));
            auto mA1 = tA.slice(32, 0);
            mm.run(mB1, mA1, cF);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float *sc_g = (threadgroup float *)shmem;
    threadgroup float *sc_f = (threadgroup float *)(shmem + 8192);
    auto tCG = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        sc_g, dextents<int32_t, 2>(NR0, NR1));
    auto tCF = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        sc_f, dextents<int32_t, 2>(NR0, NR1));
    cG.store(tCG.slice(0, 0));
    cF.store(tCF.slice(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int lane = int(tid & 31u);
    for (int j = int(sgitg); j < nr1; j += 4) {
        const uint64_t row_off = uint64_t(t0 + uint(j)) * width + r0;
        threadgroup const float4 *g4 = (threadgroup const float4 *)(sc_g + j * NR0);
        threadgroup const float4 *f4 = (threadgroup const float4 *)(sc_f + j * NR0);
        if (lane < nr0 / 4) {
            const float4 s = ((device const float4 *)(src_state + row_off))[lane];
            const float4 m = ((device const float4 *)(message + row_off))[lane];
            ((device float4 *)(dst_state + row_off))[lane] = (1.0f - tanh(f4[lane])) * s + tanh(g4[lane]) * m;
        }
    }
}
#endif

kernel void qw3_inkling_control_apply_batch(
    constant uint4 &shape [[buffer(0)]],
    device const float *gain [[buffer(1)]],
    device const float *forget [[buffer(2)]],
    device const float *code [[buffer(3)]],
    device float *state [[buffer(4)]],
    device const float *message [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = 4u * group.x + simd;
    const uint t = group.y;
    const uint width = shape.x, rank = shape.y, n_tokens = shape.z;
    if (row >= width || t >= n_tokens) return;
    const uint offset = row * rank;
    device const float *ct = code + uint64_t(t) * rank;
    float g = 0.0f, f = 0.0f;
    for (uint i = lane; i < rank; i += 32u) {
        g += gain[offset + i] * ct[i];
        f += forget[offset + i] * ct[i];
    }
    g = simd_sum(g);
    f = simd_sum(f);
    if (lane == 0u) {
        const uint64_t idx = uint64_t(t) * width + row;
        state[idx] = (1.0f - tanh(f)) * state[idx] + tanh(g) * message[idx];
    }
}

kernel void qw3_inkling_expert_activation_batch(
    constant uint4 &shape [[buffer(0)]],
    constant float2 &limits [[buffer(1)]],
    device const float *scales [[buffer(2)]],
    device const float *gate_up [[buffer(3)]],
    device half *hidden [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]) {
    const uint width = shape.x, total_slots = shape.z;
    if (pos.x >= width || pos.y >= total_slots) return;
    device const float *src = gate_up + uint64_t(pos.y) * shape.y;
    device half *dst = hidden + uint64_t(pos.y) * width;
    const float sc = scales[pos.y];
    const float gate = src[pos.x] * sc;
    const float up = src[width + pos.x] * sc;
    const float beta = limits.x, limit = limits.y;
    dst[pos.x] = half(beta * tanh(gate / beta) / (1.0f + exp(-gate)) * limit * tanh(up / limit));
}

kernel void qw3_inkling_expert_reduce_batch(
    constant uint4 &shape [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *down [[buffer(2)]],
    device float *out [[buffer(3)]],
    uint2 pos [[thread_position_in_grid]]) {
    const uint width = shape.x, n_active = shape.z, n_tokens = shape.w;
    if (pos.x >= width || pos.y >= n_tokens) return;
    const uint base_slot = pos.y * n_active;
    float value = 0.0f;
    for (uint i = 0; i < n_active; i++) {
        value += weights[base_slot + i] * down[uint64_t(base_slot + i) * shape.y + pos.x];
    }
    out[uint64_t(pos.y) * width + pos.x] = value;
}

kernel void qw3_inkling_rmsnorm_pair_batch(
    constant uint2 &args [[buffer(0)]],
    device const float *a [[buffer(1)]],
    device const float *b [[buffer(2)]],
    device float *out [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd_idx [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint t = group.x, which = group.y;
    const uint width = args.x, n_tokens = args.y;
    if (t >= n_tokens) return;
    device const float *src = (which == 0u ? a : b) + uint64_t(t) * width;
    device float *dst = out + uint64_t(t) * (2u * width) + which * width;
    threadgroup float sh[4];
    float ss = 0.0f;
    for (uint i = tid; i < width; i += 128u) ss += src[i] * src[i];
    ss = simd_sum(ss);
    if (lane == 0u) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 4u ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    const float scale = rsqrt(ss / float(width) + 1.0e-6f);
    for (uint i = tid; i < width; i += 128u) dst[i] = src[i] * scale;
}
