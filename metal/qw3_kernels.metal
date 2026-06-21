#include <metal_stdlib>
#ifdef QW3_METAL_HAS_TENSOR
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;
#ifdef QW3_METAL_HAS_TENSOR
using namespace mpp::tensor_ops;
#endif
constant float qw3_iq4nl_table[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f,
     -49.0f,  -35.0f, -22.0f, -10.0f,
       1.0f,   13.0f,  25.0f,  38.0f,
      53.0f,   69.0f,  89.0f, 113.0f
};
inline float qw3_f16_to_f32(ushort h) {
    uint s = uint(h >> 15u);
    uint e = (uint(h) >> 10u) & 31u;
    uint f = uint(h) & 1023u;
    float sign = s ? -1.0f : 1.0f;
    if (e == 0u) return f == 0u ? sign * 0.0f : sign * float(f) * exp2(-24.0f);
    if (e == 31u) return f == 0u ? sign * INFINITY : NAN;
    return sign * (1.0f + float(f) / 1024.0f) * exp2(float(e) - 15.0f);
}
struct qw3_rmsnorm_args { uint n; float eps; };
kernel void qw3_rmsnorm_plain(constant qw3_rmsnorm_args &args,
                              device const float *x,
                              device float *y,
                              threadgroup float *sh,
                              ushort tid [[thread_index_in_threadgroup]],
                              ushort simd_idx [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]],
                              ushort nt [[threads_per_threadgroup]]) {
    float ss = 0.0f;
    for (uint i = tid; i < args.n; i += nt) ss += x[i] * x[i];
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.n) + args.eps);
    for (uint i = tid; i < args.n; i += nt) y[i] = x[i] * scale;
}
kernel void qw3_rmsnorm_weight_f32(constant qw3_rmsnorm_args &args,
                                   device const float *x,
                                   device const float *w,
                                   device float *y,
                                   threadgroup float *sh,
                                   ushort tid [[thread_index_in_threadgroup]],
                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]],
                                   ushort nt [[threads_per_threadgroup]]) {
    float ss = 0.0f;
    for (uint i = tid; i < args.n; i += nt) ss += x[i] * x[i];
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.n) + args.eps);
    for (uint i = tid; i < args.n; i += nt) y[i] = x[i] * scale * w[i];
}
struct qw3_rmsnorm_rows_args { uint n; float eps; uint n_rows; };
kernel void qw3_rmsnorm_weight_f32_rows(constant qw3_rmsnorm_rows_args &args,
                                        device float *x,
                                        device const float *w,
                                        threadgroup float *sh,
                                        uint row [[threadgroup_position_in_grid]],
                                        ushort tid [[thread_index_in_threadgroup]],
                                        ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                        ushort lane [[thread_index_in_simdgroup]],
                                        ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_rows) return;
    device float *xr = x + uint64_t(row) * args.n;
    float ss = 0.0f;
    for (uint i = tid; i < args.n; i += nt) ss += xr[i] * xr[i];
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.n) + args.eps);
    for (uint i = tid; i < args.n; i += nt) xr[i] = xr[i] * scale * w[i];
}
kernel void qw3_rmsnorm_weight_f32_rows_to_out(constant qw3_rmsnorm_rows_args &args,
                                               device const float *x,
                                               device const float *w,
                                               device float *out,
                                               threadgroup float *sh,
                                               uint row [[threadgroup_position_in_grid]],
                                               ushort tid [[thread_index_in_threadgroup]],
                                               ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                               ushort lane [[thread_index_in_simdgroup]],
                                               ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_rows) return;
    device const float *xr = x + uint64_t(row) * args.n;
    device float *yr = out + uint64_t(row) * args.n;
    float ss = 0.0f;
    for (uint i = tid; i < args.n; i += nt) ss += xr[i] * xr[i];
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.n) + args.eps);
    for (uint i = tid; i < args.n; i += nt) yr[i] = xr[i] * scale * w[i];
}
struct qw3_embed_q8_0_args { uint n_embd; uint row_bytes; };
kernel void qw3_embed_q8_0(constant qw3_embed_q8_0_args &args,
                           device const uchar *weights,
                           device float *out,
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n_embd) return;
    uint block = gid / 32;
    uint lane = gid % 32;
    device const uchar *blk = weights + block * 34;
    half d = *((device const half *)blk);
    char q = *((device const char *)(blk + 2 + lane));
    out[gid] = float(d) * float(q);
}
struct qw3_embed_q8_0_batch_args { uint n_embd; uint row_bytes; uint n_tokens; };
kernel void qw3_embed_q8_0_batch(constant qw3_embed_q8_0_batch_args &args,
                                 device const uchar *weights,
                                 device const uint *tokens,
                                 device float *out,
                                 uint gid [[thread_position_in_grid]]) {
    uint total = args.n_tokens * args.n_embd;
    if (gid >= total) return;
    uint t = gid / args.n_embd;
    uint i = gid - t * args.n_embd;
    uint token = tokens[t];
    device const uchar *row = weights + uint64_t(token) * uint64_t(args.row_bytes);
    uint block = i / 32u;
    uint lane = i & 31u;
    device const uchar *blk = row + uint64_t(block) * 34ull;
    float d = float(*((device const half *)blk));
    char q = *((device const char *)(blk + 2u + lane));
    out[uint64_t(t) * args.n_embd + i] = d * float(q);
}
struct qw3_kv_quant_q8_args { uint n; };
kernel void qw3_gqa_kv_quant_q8(constant qw3_kv_quant_q8_args &args,
                                device const float *k,
                                device const float *v,
                                device uchar *k_cache,
                                device uchar *v_cache,
                                uint block [[threadgroup_position_in_grid]],
                                ushort tid [[thread_index_in_threadgroup]]) {
    uint base = block * 32u;
    if (base >= args.n || tid >= 32u) return;
    float ka = fabs(k[base + uint(tid)]);
    float va = fabs(v[base + uint(tid)]);
    ka = simd_max(ka);
    va = simd_max(va);
    float kd = ka > 0.0f ? ka / 127.0f : 0.0f;
    float vd = va > 0.0f ? va / 127.0f : 0.0f;
    device uchar *kb = k_cache + uint64_t(block) * 34ull;
    device uchar *vb = v_cache + uint64_t(block) * 34ull;
    if (tid == 0) {
        *((device half *)kb) = half(kd);
        *((device half *)vb) = half(vd);
    }
    float kq = kd > 0.0f ? rint(k[base + uint(tid)] / kd) : 0.0f;
    float vq = vd > 0.0f ? rint(v[base + uint(tid)] / vd) : 0.0f;
    *((device char *)(kb + 2u + uint(tid))) = char(clamp(kq, -127.0f, 127.0f));
    *((device char *)(vb + 2u + uint(tid))) = char(clamp(vq, -127.0f, 127.0f));
}
kernel void qw3_gqa_store_token_cache_f16(constant qw3_kv_quant_q8_args &args,
                                          device const float *k,
                                          device const float *v,
                                          device half *k_cache,
                                          device half *v_cache,
                                          uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n) return;
    k_cache[gid] = half(k[gid]);
    v_cache[gid] = half(v[gid]);
}
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
    sum = lane < 32 ? sh[lane] : 0.0f;
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
    sum = lane < 32 ? sh[lane] : 0.0f;
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
    sum = lane < 32 ? sh[lane] : 0.0f;
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
    sum = lane < 32 ? sh[lane] : 0.0f;
    sum = simd_sum(sum);
    if (tid == 0) x0[row] = x0[row] + sum * scale;
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
    sum = simd_sum(sum); if (lane == 0) sh[simd_idx] = sum; threadgroup_barrier(mem_flags::mem_threadgroup); sum = lane < 32 ? sh[lane] : 0.0f; sum = simd_sum(sum); if (tid == 0) x0[row] = x0[row] + sum * scale;
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
    sum = lane < 32 ? sh[lane] : 0.0f;
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}
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
    sum = lane < 32 ? sh[lane] : 0.0f;
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
    sum = simd_sum(sum); if (lane == 0) sh[simd_idx] = sum; threadgroup_barrier(mem_flags::mem_threadgroup); sum = lane < 32 ? sh[lane] : 0.0f; sum = simd_sum(sum); if (tid == 0) x0[row] = x0[row] + sum * scale;
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
    sum = lane < 32 ? sh[lane] : 0.0f;
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
    sum = lane < 32 ? sh[lane] : 0.0f;
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
struct qw3_conv1d_args { uint n_channels; };
kernel void qw3_deltanet_conv1d_zero(constant qw3_conv1d_args &args,
                                     device const float *w,
                                     device const float *qkv,
                                     device float *out,
                                     uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n_channels) return;
    float x = qkv[gid] * w[gid * 4 + 3];
    out[gid] = x / (1.0f + exp(-x));
}
kernel void qw3_deltanet_conv1d_step(constant qw3_conv1d_args &args,
                                     device const float *w,
                                     device const float *qkv,
                                     device const float *state_in,
                                     device float *out,
                                     device float *state_out,
                                     uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n_channels) return;
    device const float *st = state_in + uint64_t(gid) * 3u;
    float x = st[0] * w[gid * 4 + 0] + st[1] * w[gid * 4 + 1] +
              st[2] * w[gid * 4 + 2] + qkv[gid] * w[gid * 4 + 3];
    out[gid] = x / (1.0f + exp(-x));
    device float *so = state_out + uint64_t(gid) * 3u;
    so[0] = st[1];
    so[1] = st[2];
    so[2] = qkv[gid];
}
struct qw3_conv1d_batch_args { uint n_channels; uint n_tokens; uint qkv_offset; uint conv_offset; uint stride; };
kernel void qw3_deltanet_conv1d_batch(constant qw3_conv1d_batch_args &args,
                                      device const float *w,
                                      device float *scratch,
                                      device float *state,
                                      uint gid [[thread_position_in_grid]]) {
    uint total = args.n_channels * args.n_tokens;
    if (gid >= total) return;
    uint t = gid / args.n_channels;
    uint ch = gid - t * args.n_channels;
    device const float *wr = w + uint64_t(ch) * 4ull;
    device float *st = state + uint64_t(ch) * 3ull;
    float sum = 0.0f;
    for (int k = 0; k < 4; k++) {
        int idx = int(t) + k - 3;
        float xv = idx < 0 ? st[3 + idx] : scratch[uint64_t(uint(idx)) * args.stride + args.qkv_offset + ch];
        sum += xv * wr[k];
    }
    float y = sum / (1.0f + exp(-sum));
    scratch[uint64_t(t) * args.stride + args.conv_offset + ch] = y;
}
kernel void qw3_deltanet_conv1d_batch_state(constant qw3_conv1d_batch_args &args,
                                            device const float *scratch,
                                            device float *state,
                                            uint ch [[thread_position_in_grid]]) {
    if (ch >= args.n_channels) return;
    device float *st = state + uint64_t(ch) * 3ull;
    float s0 = st[0], s1 = st[1], s2 = st[2];
    for (int j = 0; j < 3; j++) {
        int idx = int(args.n_tokens) - 3 + j;
        float v = 0.0f;
        if (idx < 0) {
            v = (idx == -3) ? s0 : ((idx == -2) ? s1 : s2);
        } else {
            v = scratch[uint64_t(uint(idx)) * args.stride + args.qkv_offset + ch];
        }
        st[j] = v;
    }
}
struct qw3_l2norm_args { uint head_dim; float eps; };
kernel void qw3_l2norm_heads(constant qw3_l2norm_args &args,
                             device const float *x,
                             device float *out,
                             threadgroup float *sh,
                             uint head [[threadgroup_position_in_grid]],
                             ushort tid [[thread_index_in_threadgroup]],
                             ushort simd_idx [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]],
                             ushort nt [[threads_per_threadgroup]]) {
    device const float *xh = x + uint64_t(head) * args.head_dim;
    device float *yh = out + uint64_t(head) * args.head_dim;
    float ss = 0.0f;
    for (uint i = tid; i < args.head_dim; i += nt) ss += xh[i] * xh[i];
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = 1.0f / max(sqrt(ss), args.eps);
    for (uint i = tid; i < args.head_dim; i += nt) yh[i] = xh[i] * scale;
}
struct qw3_l2norm_qk_batch_args { uint n_tokens; uint conv_offset; uint stride; uint n_qk_heads; uint head_dim; float eps; };
kernel void qw3_l2norm_qk_batch(constant qw3_l2norm_qk_batch_args &args,
                                device float *scratch,
                                threadgroup float *sh,
                                uint group [[threadgroup_position_in_grid]],
                                ushort tid [[thread_index_in_threadgroup]],
                                ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]],
                                ushort nt [[threads_per_threadgroup]]) {
    uint heads_per_qk = args.n_qk_heads * args.n_tokens;
    uint qk = group / heads_per_qk;
    uint rem = group - qk * heads_per_qk;
    uint t = rem / args.n_qk_heads;
    uint head = rem - t * args.n_qk_heads;
    if (head >= args.n_qk_heads || t >= args.n_tokens || qk >= 2u) return;
    uint qk_n = args.n_qk_heads * args.head_dim;
    uint off = args.conv_offset + qk * qk_n + head * args.head_dim;
    device float *xh = scratch + uint64_t(t) * args.stride + off;
    float ss = 0.0f;
    for (uint i = tid; i < args.head_dim; i += nt) ss += xh[i] * xh[i];
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = 1.0f / max(sqrt(ss), args.eps);
    for (uint i = tid; i < args.head_dim; i += nt) xh[i] *= scale;
}
struct qw3_gqa_norm_args { uint n_heads; uint head_dim; float eps; };
kernel void qw3_gqa_q_norm_gate(constant qw3_gqa_norm_args &args,
                                device const float *qg,
                                device const float *w,
                                device float *q_out,
                                device float *gate_out,
                                threadgroup float *sh,
                                uint head [[threadgroup_position_in_grid]],
                                ushort tid [[thread_index_in_threadgroup]],
                                ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]],
                                ushort nt [[threads_per_threadgroup]]) {
    if (head >= args.n_heads) return;
    device const float *qh = qg + uint64_t(head) * uint64_t(args.head_dim) * 2ull;
    device const float *gh = qh + args.head_dim;
    device float *yo = q_out + uint64_t(head) * args.head_dim;
    device float *go = gate_out + uint64_t(head) * args.head_dim;
    float ss = 0.0f;
    for (uint i = tid; i < args.head_dim; i += nt) ss += qh[i] * qh[i];
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.head_dim) + args.eps);
    for (uint i = tid; i < args.head_dim; i += nt) { yo[i] = qh[i] * scale * w[i]; go[i] = gh[i]; }
}
kernel void qw3_gqa_k_norm(constant qw3_gqa_norm_args &args,
                           device const float *k,
                           device const float *w,
                           device float *k_out,
                           threadgroup float *sh,
                           uint head [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]],
                           ushort simd_idx [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           ushort nt [[threads_per_threadgroup]]) {
    if (head >= args.n_heads) return;
    device const float *kh = k + uint64_t(head) * args.head_dim;
    device float *yo = k_out + uint64_t(head) * args.head_dim;
    float ss = 0.0f;
    for (uint i = tid; i < args.head_dim; i += nt) ss += kh[i] * kh[i];
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.head_dim) + args.eps);
    for (uint i = tid; i < args.head_dim; i += nt) yo[i] = kh[i] * scale * w[i];
}
struct qw3_gqa_norm_batch_args { uint n_tokens; uint n_heads; uint head_dim; uint in_offset; uint out_offset; uint gate_offset; uint stride; float eps; };
kernel void qw3_gqa_q_norm_gate_batch(constant qw3_gqa_norm_batch_args &args,
                                      device float *scratch,
                                      device const float *w,
                                      threadgroup float *sh,
                                      uint group [[threadgroup_position_in_grid]],
                                      ushort tid [[thread_index_in_threadgroup]],
                                      ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                      ushort lane [[thread_index_in_simdgroup]],
                                      ushort nt [[threads_per_threadgroup]]) {
    uint t = group / args.n_heads;
    uint head = group - t * args.n_heads;
    if (t >= args.n_tokens || head >= args.n_heads) return;
    device float *row = scratch + uint64_t(t) * args.stride;
    device const float *qh = row + args.in_offset + uint64_t(head) * uint64_t(args.head_dim) * 2ull;
    device const float *gh = qh + args.head_dim;
    device float *yo = row + args.out_offset + uint64_t(head) * args.head_dim;
    device float *go = row + args.gate_offset + uint64_t(head) * args.head_dim;
    float ss = 0.0f;
    for (uint i = tid; i < args.head_dim; i += nt) ss += qh[i] * qh[i];
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.head_dim) + args.eps);
    for (uint i = tid; i < args.head_dim; i += nt) { yo[i] = qh[i] * scale * w[i]; go[i] = gh[i]; }
}
kernel void qw3_gqa_k_norm_batch(constant qw3_gqa_norm_batch_args &args,
                                 device float *scratch,
                                 device const float *w,
                                 threadgroup float *sh,
                                 uint group [[threadgroup_position_in_grid]],
                                 ushort tid [[thread_index_in_threadgroup]],
                                 ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]],
                                 ushort nt [[threads_per_threadgroup]]) {
    uint t = group / args.n_heads;
    uint head = group - t * args.n_heads;
    if (t >= args.n_tokens || head >= args.n_heads) return;
    device float *row = scratch + uint64_t(t) * args.stride;
    device const float *kh = row + args.in_offset + uint64_t(head) * args.head_dim;
    device float *yo = row + args.out_offset + uint64_t(head) * args.head_dim;
    float ss = 0.0f;
    for (uint i = tid; i < args.head_dim; i += nt) ss += kh[i] * kh[i];
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.head_dim) + args.eps);
    for (uint i = tid; i < args.head_dim; i += nt) yo[i] = kh[i] * scale * w[i];
}
struct qw3_gqa_norm_rope_batch_args { uint n_tokens; uint n_heads; uint head_dim; uint rope_dim; uint pos0; uint in_offset; uint out_offset; uint gate_offset; uint stride; float theta; float eps; };
kernel void qw3_gqa_q_norm_gate_rope_batch(constant qw3_gqa_norm_rope_batch_args &args,
                                           device float *scratch,
                                           device const float *w,
                                           threadgroup float *sh,
                                           uint group [[threadgroup_position_in_grid]],
                                           ushort tid [[thread_index_in_threadgroup]],
                                           ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                           ushort lane [[thread_index_in_simdgroup]],
                                           ushort nt [[threads_per_threadgroup]]) {
    uint t = group / args.n_heads;
    uint head = group - t * args.n_heads;
    if (t >= args.n_tokens || head >= args.n_heads) return;
    device float *row = scratch + uint64_t(t) * args.stride;
    device const float *qh = row + args.in_offset + uint64_t(head) * uint64_t(args.head_dim) * 2ull;
    device const float *gh = qh + args.head_dim;
    device float *yo = row + args.out_offset + uint64_t(head) * args.head_dim;
    device float *go = row + args.gate_offset + uint64_t(head) * args.head_dim;
    float ss = 0.0f;
    for (uint i = tid; i < args.head_dim; i += nt) ss += qh[i] * qh[i];
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.head_dim) + args.eps);
    for (uint i = tid; i < args.head_dim; i += nt) {
        go[i] = gh[i];
        if (i >= args.rope_dim) { yo[i] = qh[i] * scale * w[i]; continue; }
        uint p = i & ~1u;
        float freq = pow(args.theta, -float(p) / float(args.rope_dim));
        float ang = float(args.pos0 + t) * freq;
        float c = cos(ang);
        float s = sin(ang);
        float x0 = qh[p + 0u] * scale * w[p + 0u];
        float x1 = qh[p + 1u] * scale * w[p + 1u];
        yo[i] = (i & 1u) ? (x0 * s + x1 * c) : (x0 * c - x1 * s);
    }
}
kernel void qw3_gqa_k_norm_rope_batch(constant qw3_gqa_norm_rope_batch_args &args,
                                      device float *scratch,
                                      device const float *w,
                                      threadgroup float *sh,
                                      uint group [[threadgroup_position_in_grid]],
                                      ushort tid [[thread_index_in_threadgroup]],
                                      ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                      ushort lane [[thread_index_in_simdgroup]],
                                      ushort nt [[threads_per_threadgroup]]) {
    uint t = group / args.n_heads;
    uint head = group - t * args.n_heads;
    if (t >= args.n_tokens || head >= args.n_heads) return;
    device float *row = scratch + uint64_t(t) * args.stride;
    device const float *kh = row + args.in_offset + uint64_t(head) * args.head_dim;
    device float *yo = row + args.out_offset + uint64_t(head) * args.head_dim;
    float ss = 0.0f;
    for (uint i = tid; i < args.head_dim; i += nt) ss += kh[i] * kh[i];
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.head_dim) + args.eps);
    for (uint i = tid; i < args.head_dim; i += nt) {
        if (i >= args.rope_dim) { yo[i] = kh[i] * scale * w[i]; continue; }
        uint p = i & ~1u;
        float freq = pow(args.theta, -float(p) / float(args.rope_dim));
        float ang = float(args.pos0 + t) * freq;
        float c = cos(ang);
        float s = sin(ang);
        float x0 = kh[p + 0u] * scale * w[p + 0u];
        float x1 = kh[p + 1u] * scale * w[p + 1u];
        yo[i] = (i & 1u) ? (x0 * s + x1 * c) : (x0 * c - x1 * s);
    }
}
struct qw3_rope_args { uint n_heads; uint head_dim; uint rope_dim; int pos; float theta; };
kernel void qw3_rope_heads(constant qw3_rope_args &args,
                           device const float *x,
                           device float *out,
                           uint gid [[thread_position_in_grid]]) {
    uint total = args.n_heads * args.head_dim;
    if (gid >= total) return;
    uint h = gid / args.head_dim;
    uint i = gid - h * args.head_dim;
    device const float *xh = x + uint64_t(h) * args.head_dim;
    device float *yh = out + uint64_t(h) * args.head_dim;
    if (i >= args.rope_dim) { yh[i] = xh[i]; return; }
    uint p = i & ~1u;
    float freq = pow(args.theta, -float(p) / float(args.rope_dim));
    float ang = float(args.pos) * freq;
    float c = cos(ang);
    float s = sin(ang);
    float x0 = xh[p + 0u];
    float x1 = xh[p + 1u];
    yh[i] = (i & 1u) ? (x0 * s + x1 * c) : (x0 * c - x1 * s);
}
struct qw3_rope_batch_args { uint n_tokens; uint n_heads; uint head_dim; uint rope_dim; uint pos0; uint in_offset; uint out_offset; uint stride; float theta; };
kernel void qw3_rope_heads_batch(constant qw3_rope_batch_args &args,
                                 device float *scratch,
                                 uint gid [[thread_position_in_grid]]) {
    uint per_tok = args.n_heads * args.head_dim;
    uint total = args.n_tokens * per_tok;
    if (gid >= total) return;
    uint t = gid / per_tok;
    uint rem = gid - t * per_tok;
    uint h = rem / args.head_dim;
    uint i = rem - h * args.head_dim;
    device const float *xh = scratch + uint64_t(t) * args.stride + args.in_offset + uint64_t(h) * args.head_dim;
    device float *yh = scratch + uint64_t(t) * args.stride + args.out_offset + uint64_t(h) * args.head_dim;
    if (i >= args.rope_dim) { yh[i] = xh[i]; return; }
    uint p = i & ~1u;
    float freq = pow(args.theta, -float(p) / float(args.rope_dim));
    float ang = float(args.pos0 + t) * freq;
    float c = cos(ang);
    float s = sin(ang);
    float x0 = xh[p + 0u];
    float x1 = xh[p + 1u];
    yh[i] = (i & 1u) ? (x0 * s + x1 * c) : (x0 * c - x1 * s);
}
struct qw3_gqa_inner_args { uint n_heads; uint n_kv_heads; uint head_dim; };
kernel void qw3_gqa_single_token_inner(constant qw3_gqa_inner_args &args,
                                       device const float *gate,
                                       device const float *v,
                                       device float *out,
                                       uint gid [[thread_position_in_grid]]) {
    uint total = args.n_heads * args.head_dim;
    if (gid >= total) return;
    uint h = gid / args.head_dim;
    uint i = gid - h * args.head_dim;
    uint kvh = h / (args.n_heads / args.n_kv_heads);
    float g = gate[gid];
    float sig = 1.0f / (1.0f + exp(-g));
    out[gid] = v[uint64_t(kvh) * args.head_dim + i] * sig;
}
kernel void qw3_gqa_attend2_inner(constant qw3_gqa_inner_args &args,
                                  device const float *q,
                                  device const float *gate,
                                  device const float *k_cache,
                                  device const float *v_cache,
                                  device float *out,
                                  uint gid [[thread_position_in_grid]]) {
    uint total = args.n_heads * args.head_dim;
    if (gid >= total) return;
    uint h = gid / args.head_dim;
    uint i = gid - h * args.head_dim;
    uint kvh = h / (args.n_heads / args.n_kv_heads);
    device const float *qh = q + uint64_t(h) * args.head_dim;
    device const float *k0 = k_cache + uint64_t(kvh) * args.head_dim;
    device const float *k1 = k_cache + (uint64_t(args.n_kv_heads) + kvh) * args.head_dim;
    float d0 = 0.0f;
    float d1 = 0.0f;
    for (uint j = 0; j < args.head_dim; j++) { d0 += qh[j] * k0[j]; d1 += qh[j] * k1[j]; }
    float scale = rsqrt(float(args.head_dim));
    d0 *= scale;
    d1 *= scale;
    float m = max(d0, d1);
    float e0 = exp(d0 - m);
    float e1 = exp(d1 - m);
    float w0 = e0 / (e0 + e1);
    float w1 = e1 / (e0 + e1);
    float v0 = v_cache[uint64_t(kvh) * args.head_dim + i];
    float v1 = v_cache[(uint64_t(args.n_kv_heads) + kvh) * args.head_dim + i];
    float sig = 1.0f / (1.0f + exp(-gate[gid]));
    out[gid] = (w0 * v0 + w1 * v1) * sig;
}
struct qw3_gqa_n_args { uint n_ctx; uint n_heads; uint n_kv_heads; uint head_dim; uint kv_type; };
inline float qw3_gqa_cache_load(device const float *f32_cache, device const half *f16_cache, uint64_t idx, uint kv_type) {
    return kv_type == 1u ? float(f16_cache[idx]) : f32_cache[idx];
}
kernel void qw3_gqa_attend_n_inner(constant qw3_gqa_n_args &args,
                                  device const float *q,
                                  device const float *gate,
                                  device const float *k_cache,
                                  device const float *v_cache,
                                  device float *out,
                                  device const half *k_cache_f16,
                                  device const half *v_cache_f16,
                                  threadgroup float *sh,
                                  uint h [[threadgroup_position_in_grid]],
                                  ushort tid [[thread_index_in_threadgroup]],
                                  ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                  ushort lane [[thread_index_in_simdgroup]],
                                  ushort nt [[threads_per_threadgroup]]) {
    if (h >= args.n_kv_heads || args.n_ctx == 0 || args.head_dim > uint(nt)) return;
    uint i = uint(tid);
    uint kvh = h;
    uint group_heads = args.n_heads / args.n_kv_heads;
    if (group_heads == 0u || group_heads > 8u) return;
    uint first_qh = kvh * group_heads;
    float scale = rsqrt(float(args.head_dim));
    float qv[8];
    float max_score[8];
    float denom[8];
    float acc[8];
    for (uint gh = 0; gh < 8u; gh++) {
        uint qh = first_qh + gh;
        qv[gh] = (gh < group_heads && i < args.head_dim && qh < args.n_heads) ? q[uint64_t(qh) * args.head_dim + i] : 0.0f;
        max_score[gh] = -FLT_MAX;
        denom[gh] = 0.0f;
        acc[gh] = 0.0f;
    }
    uint n_simd = (uint(nt) + 31u) >> 5u;
    for (uint t = 0; t < args.n_ctx; t++) {
        uint64_t kv_idx = (uint64_t(t) * args.n_kv_heads + kvh) * args.head_dim + i;
        float kval = (i < args.head_dim) ? qw3_gqa_cache_load(k_cache, k_cache_f16, kv_idx, args.kv_type) : 0.0f;
        for (uint gh = 0; gh < 8u; gh++) {
            float part = (gh < group_heads && i < args.head_dim) ? qv[gh] * kval : 0.0f;
            part = simd_sum(part);
            if (lane == 0) sh[gh * 8u + uint(simd_idx)] = part;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint gh = 0; gh < 8u; gh++) {
            float dot = (gh < group_heads && uint(tid) < n_simd) ? sh[gh * 8u + uint(tid)] : 0.0f;
            dot = simd_sum(dot);
            if (tid == 0) sh[gh] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float vv = (i < args.head_dim) ? qw3_gqa_cache_load(v_cache, v_cache_f16, kv_idx, args.kv_type) : 0.0f;
        for (uint gh = 0; gh < 8u; gh++) {
            if (gh < group_heads) {
                float score = sh[gh] * scale;
                float next_max = max(max_score[gh], score);
                float prev_scale = exp(max_score[gh] - next_max);
                float cur_scale = exp(score - next_max);
                acc[gh] = acc[gh] * prev_scale + vv * cur_scale;
                denom[gh] = denom[gh] * prev_scale + cur_scale;
                max_score[gh] = next_max;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i < args.head_dim) {
        for (uint gh = 0; gh < 8u; gh++) {
            uint qh = first_qh + gh;
            if (gh < group_heads && qh < args.n_heads) {
                uint gid = qh * args.head_dim + i;
                float sig = 1.0f / (1.0f + exp(-gate[gid]));
                out[gid] = (acc[gh] / denom[gh]) * sig;
            }
        }
    }
}
struct qw3_gqa_split_args { uint n_ctx; uint n_heads; uint n_kv_heads; uint head_dim; uint n_splits; uint kv_type; };
kernel void qw3_gqa_attend_n_split_partial(constant qw3_gqa_split_args &args,
                                           device const float *q,
                                           device const float *k_cache,
                                           device const float *v_cache,
                                           device float *partial,
                                           device const half *k_cache_f16,
                                           device const half *v_cache_f16,
                                           threadgroup float *sh,
                                           uint group [[threadgroup_position_in_grid]],
                                           ushort tid [[thread_index_in_threadgroup]],
                                           ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                           ushort lane [[thread_index_in_simdgroup]],
                                           ushort nt [[threads_per_threadgroup]]) {
    uint kvh = group % args.n_kv_heads;
    uint split = group / args.n_kv_heads;
    if (kvh >= args.n_kv_heads || split >= args.n_splits || args.head_dim > uint(nt)) return;
    uint t0 = uint((uint64_t(split) * uint64_t(args.n_ctx)) / uint64_t(args.n_splits));
    uint t1 = uint((uint64_t(split + 1u) * uint64_t(args.n_ctx)) / uint64_t(args.n_splits));
    if (t0 >= t1) return;
    uint i = uint(tid);
    uint group_heads = args.n_heads / args.n_kv_heads;
    if (group_heads == 0u || group_heads > 8u) return;
    uint first_qh = kvh * group_heads;
    float scale = rsqrt(float(args.head_dim));
    float qv[8];
    float max_score[8];
    float denom[8];
    float acc[8];
    for (uint gh = 0; gh < 8u; gh++) {
        uint qh = first_qh + gh;
        qv[gh] = (gh < group_heads && i < args.head_dim && qh < args.n_heads) ? q[uint64_t(qh) * args.head_dim + i] : 0.0f;
        max_score[gh] = -FLT_MAX;
        denom[gh] = 0.0f;
        acc[gh] = 0.0f;
    }
    uint n_simd = (uint(nt) + 31u) >> 5u;
    for (uint t = t0; t < t1; t++) {
        uint64_t kv_idx = (uint64_t(t) * args.n_kv_heads + kvh) * args.head_dim + i;
        float kval = (i < args.head_dim) ? qw3_gqa_cache_load(k_cache, k_cache_f16, kv_idx, args.kv_type) : 0.0f;
        for (uint gh = 0; gh < 8u; gh++) {
            float part = (gh < group_heads && i < args.head_dim) ? qv[gh] * kval : 0.0f;
            part = simd_sum(part);
            if (lane == 0) sh[gh * 8u + uint(simd_idx)] = part;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint gh = 0; gh < 8u; gh++) {
            float dot = (gh < group_heads && uint(tid) < n_simd) ? sh[gh * 8u + uint(tid)] : 0.0f;
            dot = simd_sum(dot);
            if (tid == 0) sh[gh] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float vv = (i < args.head_dim) ? qw3_gqa_cache_load(v_cache, v_cache_f16, kv_idx, args.kv_type) : 0.0f;
        for (uint gh = 0; gh < 8u; gh++) {
            if (gh < group_heads) {
                float score = sh[gh] * scale;
                float next_max = max(max_score[gh], score);
                float prev_scale = exp(max_score[gh] - next_max);
                float cur_scale = exp(score - next_max);
                acc[gh] = acc[gh] * prev_scale + vv * cur_scale;
                denom[gh] = denom[gh] * prev_scale + cur_scale;
                max_score[gh] = next_max;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i < args.head_dim) {
        uint stride = args.head_dim + 2u;
        for (uint gh = 0; gh < group_heads; gh++) {
            uint qh = first_qh + gh;
            uint64_t base = (uint64_t(split) * args.n_heads + qh) * stride;
            partial[base + i] = acc[gh];
            if (i == 0u) {
                partial[base + args.head_dim] = denom[gh];
                partial[base + args.head_dim + 1u] = max_score[gh];
            }
        }
    }
}
kernel void qw3_gqa_attend_n_split_reduce(constant qw3_gqa_split_args &args,
                                          device const float *gate,
                                          device const float *partial,
                                          device float *out,
                                          uint qh [[threadgroup_position_in_grid]],
                                          ushort tid [[thread_index_in_threadgroup]]) {
    if (qh >= args.n_heads || uint(tid) >= args.head_dim) return;
    uint i = uint(tid);
    uint stride = args.head_dim + 2u;
    float max_score = -FLT_MAX;
    float denom = 0.0f;
    float acc = 0.0f;
    for (uint split = 0; split < args.n_splits; split++) {
        uint64_t base = (uint64_t(split) * args.n_heads + qh) * stride;
        float local_denom = partial[base + args.head_dim];
        float local_max = partial[base + args.head_dim + 1u];
        float next_max = max(max_score, local_max);
        float prev_scale = exp(max_score - next_max);
        float cur_scale = exp(local_max - next_max);
        acc = acc * prev_scale + partial[base + i] * cur_scale;
        denom = denom * prev_scale + local_denom * cur_scale;
        max_score = next_max;
    }
    uint gid = qh * args.head_dim + i;
    float sig = 1.0f / (1.0f + exp(-gate[gid]));
    out[gid] = (acc / denom) * sig;
}
struct qw3_gqa_prefill_attn_args { uint n_tokens; uint n_heads; uint n_kv_heads; uint head_dim; uint q_offset; uint gate_offset; uint k_offset; uint v_offset; uint out_offset; uint stride; };
kernel void qw3_gqa_prefill_attend_inner(constant qw3_gqa_prefill_attn_args &args,
                                        device float *scratch,
                                        threadgroup float *sh,
                                        uint group [[threadgroup_position_in_grid]],
                                        ushort tid [[thread_index_in_threadgroup]],
                                        ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                        ushort lane [[thread_index_in_simdgroup]],
                                        ushort nt [[threads_per_threadgroup]]) {
    uint query = group / args.n_kv_heads;
    uint kvh = group - query * args.n_kv_heads;
    if (query >= args.n_tokens || kvh >= args.n_kv_heads || args.head_dim > uint(nt)) return;
    uint i = uint(tid);
    uint group_heads = args.n_heads / args.n_kv_heads;
    if (group_heads == 0u || group_heads > 8u) return;
    uint first_qh = kvh * group_heads;
    device float *qrow = scratch + uint64_t(query) * args.stride;
    float scale = rsqrt(float(args.head_dim));
    threadgroup float *dots = sh;
    threadgroup float *tg_max = sh + 64u;
    threadgroup float *tg_denom = sh + 72u;
    threadgroup float *tg_prev = sh + 80u;
    threadgroup float *tg_cur = sh + 88u;
    float qv[8];
    float acc[8];
    for (uint gh = 0; gh < 8u; gh++) {
        uint qh = first_qh + gh;
        qv[gh] = (gh < group_heads && i < args.head_dim && qh < args.n_heads) ? qrow[args.q_offset + uint64_t(qh) * args.head_dim + i] : 0.0f;
        acc[gh] = 0.0f;
    }
    if (tid < 8u) { tg_max[tid] = -FLT_MAX; tg_denom[tid] = 0.0f; tg_prev[tid] = 0.0f; tg_cur[tid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n_simd = (uint(nt) + 31u) >> 5u;
    for (uint src = 0; src <= query; src++) {
        device float *srow = scratch + uint64_t(src) * args.stride;
        float kval = (i < args.head_dim) ? srow[args.k_offset + uint64_t(kvh) * args.head_dim + i] : 0.0f;
        for (uint gh = 0; gh < 8u; gh++) {
            float part = (gh < group_heads && i < args.head_dim) ? qv[gh] * kval : 0.0f;
            part = simd_sum(part);
            if (lane == 0) dots[gh * 8u + uint(simd_idx)] = part;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint gh = 0; gh < 8u; gh++) {
            float dot = (gh < group_heads && uint(tid) < n_simd) ? dots[gh * 8u + uint(tid)] : 0.0f;
            dot = simd_sum(dot);
            if (tid == 0) dots[gh] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (uint(tid) < group_heads) {
            uint gh = uint(tid);
            float score = dots[gh] * scale;
            float next_max = max(tg_max[gh], score);
            float prev_scale = exp(tg_max[gh] - next_max);
            float cur_scale = exp(score - next_max);
            tg_denom[gh] = tg_denom[gh] * prev_scale + cur_scale;
            tg_max[gh] = next_max;
            tg_prev[gh] = prev_scale;
            tg_cur[gh] = cur_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float vv = (i < args.head_dim) ? srow[args.v_offset + uint64_t(kvh) * args.head_dim + i] : 0.0f;
        for (uint gh = 0; gh < 8u; gh++) {
            if (gh < group_heads) {
                acc[gh] = acc[gh] * tg_prev[gh] + vv * tg_cur[gh];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i < args.head_dim) {
        for (uint gh = 0; gh < 8u; gh++) {
            uint qh = first_qh + gh;
            if (gh < group_heads && qh < args.n_heads) {
                uint gid = qh * args.head_dim + i;
                float sig = 1.0f / (1.0f + exp(-qrow[args.gate_offset + gid]));
                qrow[args.out_offset + gid] = (acc[gh] / tg_denom[gh]) * sig;
            }
        }
    }
}
struct qw3_gqa_prefill_cache_args { uint n_tokens; uint n_kv_heads; uint head_dim; uint pos0; uint ctx_size; uint k_offset; uint v_offset; uint stride; uint kv_type; };
kernel void qw3_gqa_prefill_write_cache(constant qw3_gqa_prefill_cache_args &args,
                                        device const float *scratch,
                                        device float *k_cache,
                                        device float *v_cache,
                                        device half *k_cache_f16,
                                        device half *v_cache_f16,
                                        uint gid [[thread_position_in_grid]]) {
    uint kv_n = args.n_kv_heads * args.head_dim;
    uint total = args.n_tokens * kv_n;
    if (gid >= total) return;
    uint t = gid / kv_n;
    uint i = gid - t * kv_n;
    uint pos = args.pos0 + t;
    if (pos >= args.ctx_size) return;
    device const float *row = scratch + uint64_t(t) * args.stride;
    uint64_t dst = uint64_t(pos) * kv_n + i;
    float kv = row[args.k_offset + i];
    float vv = row[args.v_offset + i];
    if (args.kv_type == 1u) {
        k_cache_f16[dst] = half(kv);
        v_cache_f16[dst] = half(vv);
    } else {
        k_cache[dst] = kv;
        v_cache[dst] = vv;
    }
}
struct qw3_gqa_prefill_cached_attn_args { uint n_tokens; uint n_heads; uint n_kv_heads; uint head_dim; uint pos0; uint ctx_size; uint q_offset; uint gate_offset; uint out_offset; uint stride; uint kv_type; };
struct qw3_gqa_flash_gate_args { uint n_tokens; uint n_heads; uint head_dim; uint gate_offset; uint out_offset; uint stride; };
kernel void qw3_gqa_flash_gate_from_compact(constant qw3_gqa_flash_gate_args &args,
                                           device const float *flash_out,
                                           device float *scratch,
                                           uint gid [[thread_position_in_grid]]) {
    uint per_tok = args.n_heads * args.head_dim;
    uint total = args.n_tokens * per_tok;
    if (gid >= total) return;
    uint t = gid / per_tok;
    uint rem = gid - t * per_tok;
    uint h = rem / args.head_dim;
    uint i = rem - h * args.head_dim;
    device float *row = scratch + uint64_t(t) * args.stride;
    uint head_off = h * args.head_dim + i;
    float g = row[args.gate_offset + head_off];
    row[args.out_offset + head_off] = flash_out[gid] / (1.0f + exp(-g));
}
struct qw3_gqa_flash_causal_mask_args { uint n_tokens; uint n_keys; uint pos0; uint n_q_blocks; uint n_k_blocks; };
kernel void qw3_gqa_flash_causal_mask_block(constant qw3_gqa_flash_causal_mask_args &args,
                                          device half *mask,
                                          device char *blk,
                                          uint3 group [[threadgroup_position_in_grid]],
                                          ushort tid [[thread_index_in_threadgroup]],
                                          ushort3 ntg [[threads_per_threadgroup]]) {
    const uint Q = 8u;
    const uint C = 64u;
    uint kblk = group.x;
    uint qblk = group.y;
    if (qblk >= args.n_q_blocks || kblk >= args.n_k_blocks) return;
    uint q0 = qblk * Q;
    uint q1 = min(q0 + Q, args.n_tokens);
    uint k0 = kblk * C;
    uint k1 = min(k0 + C, args.n_keys);
    uint allowed_first = args.pos0 + q0 + 1u;
    uint allowed_last = args.pos0 + (q1 > 0u ? q1 - 1u : q0) + 1u;
    bool final_partial = k0 + C > args.n_keys;
    char b = 1;
    if (!final_partial && k0 >= allowed_last) {
        b = 0;
    } else if (!final_partial && (k1 - 1u) < allowed_first) {
        b = 2;
    }
    if (tid == 0) blk[qblk * args.n_k_blocks + kblk] = b;
    if (b != 1) return;
    const half zero = half(0.0f);
    const half neg = half(-65504.0f);
    for (uint idx = uint(tid); idx < Q * C; idx += uint(ntg.x)) {
        uint q = idx / C;
        uint k = idx - q * C;
        uint qt = q0 + q;
        uint kk = k0 + k;
        if (qt < args.n_tokens && kk < args.n_keys) {
            uint allowed = args.pos0 + qt + 1u;
            mask[uint64_t(qt) * uint64_t(args.n_keys) + uint64_t(kk)] = kk < allowed ? zero : neg;
        }
    }
}
struct qw3_gqa_flash_pad_args { int ne11; int ne_12_2; int ne_12_3; ulong nb11; ulong nb12; ulong nb13; ulong nb21; ulong nb22; ulong nb23; int ne31; int ne32; int ne33; ulong nb31; ulong nb32; ulong nb33; };
kernel void qw3_gqa_flash_pad_interleaved(constant qw3_gqa_flash_pad_args &args,
                                          device const char *k,
                                          device const char *v,
                                          device const char *mask,
                                          device char *dst,
                                          uint group [[threadgroup_position_in_grid]],
                                          ushort tid [[thread_index_in_threadgroup]],
                                          ushort nt [[threads_per_threadgroup]]) {
    const uint C = 64u;
    uint n_kv_heads = uint(args.ne_12_2);
    uint i1 = group % C;
    uint kvh = group / C;
    uint n_keys = uint(args.ne11);
    uint n_tokens = uint(args.ne31);
    uint head_dim = uint(args.nb12 / sizeof(half));
    uint kv_stride = uint(args.nb11 / sizeof(half));
    uint icp = n_keys % C;
    uint ic0 = n_keys - icp;
    device half *k_pad = (device half *)dst;
    device half *v_pad = (device half *)(dst + args.nb11 * C * n_kv_heads);
    device half *mask_pad = (device half *)(dst + args.nb11 * C * n_kv_heads + args.nb21 * C * n_kv_heads);
    if (i1 < C && kvh < n_kv_heads) {
        device half *kd = k_pad + uint64_t(kvh) * C * kv_stride + uint64_t(i1) * kv_stride;
        device half *vd = v_pad + uint64_t(kvh) * C * kv_stride + uint64_t(i1) * kv_stride;
        bool valid = i1 < icp;
        device const half *ks = (device const half *)(k + uint64_t(ic0 + i1) * args.nb11 + uint64_t(kvh) * args.nb12);
        device const half *vs = (device const half *)(v + uint64_t(ic0 + i1) * args.nb21 + uint64_t(kvh) * args.nb22);
        for (uint d = uint(tid); d < head_dim; d += uint(nt)) {
            kd[d] = valid ? ks[d] : half(0.0f);
            vd[d] = valid ? vs[d] : half(0.0f);
        }
    }
    if (i1 < C && kvh == 0u) {
        for (uint q = i1; q < n_tokens; q += C) {
            device const half *ms = (device const half *)(mask + uint64_t(q) * args.nb31 + uint64_t(ic0) * sizeof(half));
            device half *md = mask_pad + uint64_t(q) * C;
            for (uint d = uint(tid); d < C; d += uint(nt)) {
                md[d] = d < icp ? ms[d] : half(-65504.0f);
            }
        }
    }
}
kernel void qw3_gqa_prefill_cached_attend_inner(constant qw3_gqa_prefill_cached_attn_args &args,
                                               device float *scratch,
                                               device const float *k_cache,
                                               device const float *v_cache,
                                               device const half *k_cache_f16,
                                               device const half *v_cache_f16,
                                               threadgroup float *sh,
                                               uint group [[threadgroup_position_in_grid]],
                                               ushort tid [[thread_index_in_threadgroup]],
                                               ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                               ushort lane [[thread_index_in_simdgroup]],
                                               ushort nt [[threads_per_threadgroup]]) {
    uint query = group / args.n_kv_heads;
    uint kvh = group - query * args.n_kv_heads;
    if (query >= args.n_tokens || kvh >= args.n_kv_heads || args.head_dim > uint(nt)) return;
    uint n_ctx = args.pos0 + query + 1u;
    if (n_ctx > args.ctx_size) return;
    uint i = uint(tid);
    uint group_heads = args.n_heads / args.n_kv_heads;
    if (group_heads == 0u || group_heads > 8u) return;
    uint first_qh = kvh * group_heads;
    uint kv_n = args.n_kv_heads * args.head_dim;
    device float *qrow = scratch + uint64_t(query) * args.stride;
    float scale = rsqrt(float(args.head_dim));
    threadgroup float *dots = sh;
    threadgroup float *tg_max = sh + 64u;
    threadgroup float *tg_denom = sh + 72u;
    threadgroup float *tg_prev = sh + 80u;
    threadgroup float *tg_cur = sh + 88u;
    float qv[8];
    float acc[8];
    for (uint gh = 0; gh < 8u; gh++) {
        uint qh = first_qh + gh;
        qv[gh] = (gh < group_heads && i < args.head_dim && qh < args.n_heads) ? qrow[args.q_offset + uint64_t(qh) * args.head_dim + i] : 0.0f;
        acc[gh] = 0.0f;
    }
    if (tid < 8u) { tg_max[tid] = -FLT_MAX; tg_denom[tid] = 0.0f; tg_prev[tid] = 0.0f; tg_cur[tid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n_simd = (uint(nt) + 31u) >> 5u;
    for (uint src = 0; src < n_ctx; src++) {
        uint64_t kv_idx = uint64_t(src) * kv_n + uint64_t(kvh) * args.head_dim + i;
        float kval = (i < args.head_dim) ? qw3_gqa_cache_load(k_cache, k_cache_f16, kv_idx, args.kv_type) : 0.0f;
        for (uint gh = 0; gh < 8u; gh++) {
            float part = (gh < group_heads && i < args.head_dim) ? qv[gh] * kval : 0.0f;
            part = simd_sum(part);
            if (lane == 0) dots[gh * 8u + uint(simd_idx)] = part;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint gh = 0; gh < 8u; gh++) {
            float dot = (gh < group_heads && uint(tid) < n_simd) ? dots[gh * 8u + uint(tid)] : 0.0f;
            dot = simd_sum(dot);
            if (tid == 0) dots[gh] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (uint(tid) < group_heads) {
            uint gh = uint(tid);
            float score = dots[gh] * scale;
            float next_max = max(tg_max[gh], score);
            float prev_scale = exp(tg_max[gh] - next_max);
            float cur_scale = exp(score - next_max);
            tg_denom[gh] = tg_denom[gh] * prev_scale + cur_scale;
            tg_max[gh] = next_max;
            tg_prev[gh] = prev_scale;
            tg_cur[gh] = cur_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float vv = (i < args.head_dim) ? qw3_gqa_cache_load(v_cache, v_cache_f16, kv_idx, args.kv_type) : 0.0f;
        for (uint gh = 0; gh < 8u; gh++) {
            if (gh < group_heads) {
                acc[gh] = acc[gh] * tg_prev[gh] + vv * tg_cur[gh];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i < args.head_dim) {
        for (uint gh = 0; gh < 8u; gh++) {
            uint qh = first_qh + gh;
            if (gh < group_heads && qh < args.n_heads) {
                uint gid = qh * args.head_dim + i;
                float sig = 1.0f / (1.0f + exp(-qrow[args.gate_offset + gid]));
                qrow[args.out_offset + gid] = (acc[gh] / tg_denom[gh]) * sig;
            }
        }
    }
}
kernel void qw3_gqa_prefill_cached_attend_block2(constant qw3_gqa_prefill_cached_attn_args &args,
                                                device float *scratch,
                                                device const float *k_cache,
                                                device const float *v_cache,
                                                device const half *k_cache_f16,
                                                device const half *v_cache_f16,
                                                threadgroup float *sh,
                                                uint group [[threadgroup_position_in_grid]],
                                                ushort tid [[thread_index_in_threadgroup]],
                                                ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                                ushort lane [[thread_index_in_simdgroup]],
                                                ushort nt [[threads_per_threadgroup]]) {
    uint query0 = (group / args.n_kv_heads) * 2u;
    uint kvh = group - (group / args.n_kv_heads) * args.n_kv_heads;
    if (query0 >= args.n_tokens || kvh >= args.n_kv_heads || args.head_dim > uint(nt)) return;
    uint query1 = query0 + 1u;
    bool valid1 = query1 < args.n_tokens;
    uint max_query = valid1 ? query1 : query0;
    uint n_ctx = args.pos0 + max_query + 1u;
    if (n_ctx > args.ctx_size) return;
    uint i = uint(tid);
    uint group_heads = args.n_heads / args.n_kv_heads;
    if (group_heads == 0u || group_heads > 8u) return;
    uint first_qh = kvh * group_heads;
    uint kv_n = args.n_kv_heads * args.head_dim;
    device float *qrow0 = scratch + uint64_t(query0) * args.stride;
    device float *qrow1 = valid1 ? (scratch + uint64_t(query1) * args.stride) : qrow0;
    float scale = rsqrt(float(args.head_dim));
    threadgroup float *dots = sh;
    threadgroup float *tg_max = sh + 128u;
    threadgroup float *tg_denom = sh + 144u;
    threadgroup float *tg_prev = sh + 160u;
    threadgroup float *tg_cur = sh + 176u;
    float qv0[8];
    float qv1[8];
    float acc0[8];
    float acc1[8];
    for (uint gh = 0; gh < 8u; gh++) {
        uint qh = first_qh + gh;
        bool active = gh < group_heads && i < args.head_dim && qh < args.n_heads;
        qv0[gh] = active ? qrow0[args.q_offset + uint64_t(qh) * args.head_dim + i] : 0.0f;
        qv1[gh] = (active && valid1) ? qrow1[args.q_offset + uint64_t(qh) * args.head_dim + i] : 0.0f;
        acc0[gh] = 0.0f;
        acc1[gh] = 0.0f;
    }
    if (tid < 16u) {
        tg_max[tid] = -FLT_MAX;
        tg_denom[tid] = 0.0f;
        tg_prev[tid] = 1.0f;
        tg_cur[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n_simd = (uint(nt) + 31u) >> 5u;
    for (uint src = 0; src < n_ctx; src++) {
        uint64_t kv_idx = uint64_t(src) * kv_n + uint64_t(kvh) * args.head_dim + i;
        float kval = (i < args.head_dim) ? qw3_gqa_cache_load(k_cache, k_cache_f16, kv_idx, args.kv_type) : 0.0f;
        bool causal0 = src <= args.pos0 + query0;
        bool causal1 = valid1 && src <= args.pos0 + query1;
        for (uint gh = 0; gh < 8u; gh++) {
            bool active = gh < group_heads && i < args.head_dim;
            float part0 = (active && causal0) ? qv0[gh] * kval : 0.0f;
            float part1 = (active && causal1) ? qv1[gh] * kval : 0.0f;
            part0 = simd_sum(part0);
            part1 = simd_sum(part1);
            if (lane == 0) {
                dots[gh * 8u + uint(simd_idx)] = part0;
                dots[64u + gh * 8u + uint(simd_idx)] = part1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint gh = 0; gh < 8u; gh++) {
            float dot0 = (gh < group_heads && uint(tid) < n_simd) ? dots[gh * 8u + uint(tid)] : 0.0f;
            float dot1 = (gh < group_heads && uint(tid) < n_simd) ? dots[64u + gh * 8u + uint(tid)] : 0.0f;
            dot0 = simd_sum(dot0);
            dot1 = simd_sum(dot1);
            if (tid == 0) {
                dots[gh] = dot0;
                dots[64u + gh] = dot1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (uint(tid) < 16u) {
            uint idx = uint(tid);
            uint b = idx >> 3u;
            uint gh = idx & 7u;
            bool causal = (b == 0u) ? causal0 : causal1;
            if (gh < group_heads && causal) {
                float score = dots[b * 64u + gh] * scale;
                float next_max = max(tg_max[idx], score);
                float prev_scale = exp(tg_max[idx] - next_max);
                float cur_scale = exp(score - next_max);
                tg_denom[idx] = tg_denom[idx] * prev_scale + cur_scale;
                tg_max[idx] = next_max;
                tg_prev[idx] = prev_scale;
                tg_cur[idx] = cur_scale;
            } else {
                tg_prev[idx] = 1.0f;
                tg_cur[idx] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float vv = (i < args.head_dim) ? qw3_gqa_cache_load(v_cache, v_cache_f16, kv_idx, args.kv_type) : 0.0f;
        for (uint gh = 0; gh < 8u; gh++) {
            if (gh < group_heads) {
                acc0[gh] = acc0[gh] * tg_prev[gh] + vv * tg_cur[gh];
                acc1[gh] = acc1[gh] * tg_prev[8u + gh] + vv * tg_cur[8u + gh];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i < args.head_dim) {
        for (uint gh = 0; gh < 8u; gh++) {
            uint qh = first_qh + gh;
            if (gh < group_heads && qh < args.n_heads) {
                uint gid = qh * args.head_dim + i;
                float sig0 = 1.0f / (1.0f + exp(-qrow0[args.gate_offset + gid]));
                qrow0[args.out_offset + gid] = (acc0[gh] / tg_denom[gh]) * sig0;
                if (valid1) {
                    float sig1 = 1.0f / (1.0f + exp(-qrow1[args.gate_offset + gid]));
                    qrow1[args.out_offset + gid] = (acc1[gh] / tg_denom[8u + gh]) * sig1;
                }
            }
        }
    }
}
kernel void qw3_gqa_prefill_cached_attend_block4(constant qw3_gqa_prefill_cached_attn_args &args,
                                                device float *scratch,
                                                device const float *k_cache,
                                                device const float *v_cache,
                                                device const half *k_cache_f16,
                                                device const half *v_cache_f16,
                                                threadgroup float *sh,
                                                uint group [[threadgroup_position_in_grid]],
                                                ushort tid [[thread_index_in_threadgroup]],
                                                ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                                ushort lane [[thread_index_in_simdgroup]],
                                                ushort nt [[threads_per_threadgroup]]) {
    uint query0 = (group / args.n_kv_heads) * 4u;
    uint kvh = group - (group / args.n_kv_heads) * args.n_kv_heads;
    if (query0 >= args.n_tokens || kvh >= args.n_kv_heads || args.head_dim > uint(nt)) return;
    uint query1 = query0 + 1u;
    uint query2 = query0 + 2u;
    uint query3 = query0 + 3u;
    bool valid1 = query1 < args.n_tokens;
    bool valid2 = query2 < args.n_tokens;
    bool valid3 = query3 < args.n_tokens;
    uint max_query = valid3 ? query3 : (valid2 ? query2 : (valid1 ? query1 : query0));
    uint n_ctx = args.pos0 + max_query + 1u;
    if (n_ctx > args.ctx_size) return;
    uint i = uint(tid);
    uint group_heads = args.n_heads / args.n_kv_heads;
    if (group_heads == 0u || group_heads > 8u) return;
    uint first_qh = kvh * group_heads;
    uint kv_n = args.n_kv_heads * args.head_dim;
    device float *qrow0 = scratch + uint64_t(query0) * args.stride;
    device float *qrow1 = valid1 ? (scratch + uint64_t(query1) * args.stride) : qrow0;
    device float *qrow2 = valid2 ? (scratch + uint64_t(query2) * args.stride) : qrow0;
    device float *qrow3 = valid3 ? (scratch + uint64_t(query3) * args.stride) : qrow0;
    float scale = rsqrt(float(args.head_dim));
    threadgroup float *dots = sh;
    threadgroup float *tg_max = sh + 256u;
    threadgroup float *tg_denom = sh + 288u;
    threadgroup float *tg_prev = sh + 320u;
    threadgroup float *tg_cur = sh + 352u;
    float qv0[8]; float qv1[8]; float qv2[8]; float qv3[8];
    float acc0[8]; float acc1[8]; float acc2[8]; float acc3[8];
    for (uint gh = 0; gh < 8u; gh++) {
        uint qh = first_qh + gh;
        bool active = gh < group_heads && i < args.head_dim && qh < args.n_heads;
        uint64_t off = args.q_offset + uint64_t(qh) * args.head_dim + i;
        qv0[gh] = active ? qrow0[off] : 0.0f;
        qv1[gh] = (active && valid1) ? qrow1[off] : 0.0f;
        qv2[gh] = (active && valid2) ? qrow2[off] : 0.0f;
        qv3[gh] = (active && valid3) ? qrow3[off] : 0.0f;
        acc0[gh] = 0.0f; acc1[gh] = 0.0f; acc2[gh] = 0.0f; acc3[gh] = 0.0f;
    }
    if (tid < 32u) {
        tg_max[tid] = -FLT_MAX;
        tg_denom[tid] = 0.0f;
        tg_prev[tid] = 1.0f;
        tg_cur[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n_simd = (uint(nt) + 31u) >> 5u;
    for (uint src = 0; src < n_ctx; src++) {
        uint64_t kv_idx = uint64_t(src) * kv_n + uint64_t(kvh) * args.head_dim + i;
        float kval = (i < args.head_dim) ? qw3_gqa_cache_load(k_cache, k_cache_f16, kv_idx, args.kv_type) : 0.0f;
        bool causal0 = src <= args.pos0 + query0;
        bool causal1 = valid1 && src <= args.pos0 + query1;
        bool causal2 = valid2 && src <= args.pos0 + query2;
        bool causal3 = valid3 && src <= args.pos0 + query3;
        for (uint gh = 0; gh < 8u; gh++) {
            bool active = gh < group_heads && i < args.head_dim;
            float part0 = (active && causal0) ? qv0[gh] * kval : 0.0f;
            float part1 = (active && causal1) ? qv1[gh] * kval : 0.0f;
            float part2 = (active && causal2) ? qv2[gh] * kval : 0.0f;
            float part3 = (active && causal3) ? qv3[gh] * kval : 0.0f;
            part0 = simd_sum(part0); part1 = simd_sum(part1);
            part2 = simd_sum(part2); part3 = simd_sum(part3);
            if (lane == 0) {
                uint base = gh * 8u + uint(simd_idx);
                dots[base] = part0;
                dots[64u + base] = part1;
                dots[128u + base] = part2;
                dots[192u + base] = part3;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint gh = 0; gh < 8u; gh++) {
            uint base = gh * 8u + uint(tid);
            float dot0 = (gh < group_heads && uint(tid) < n_simd) ? dots[base] : 0.0f;
            float dot1 = (gh < group_heads && uint(tid) < n_simd) ? dots[64u + base] : 0.0f;
            float dot2 = (gh < group_heads && uint(tid) < n_simd) ? dots[128u + base] : 0.0f;
            float dot3 = (gh < group_heads && uint(tid) < n_simd) ? dots[192u + base] : 0.0f;
            dot0 = simd_sum(dot0); dot1 = simd_sum(dot1);
            dot2 = simd_sum(dot2); dot3 = simd_sum(dot3);
            if (tid == 0) {
                dots[gh] = dot0;
                dots[64u + gh] = dot1;
                dots[128u + gh] = dot2;
                dots[192u + gh] = dot3;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (uint(tid) < 32u) {
            uint idx = uint(tid);
            uint b = idx >> 3u;
            uint gh = idx & 7u;
            bool causal = b == 0u ? causal0 : (b == 1u ? causal1 : (b == 2u ? causal2 : causal3));
            if (gh < group_heads && causal) {
                float score = dots[b * 64u + gh] * scale;
                float next_max = max(tg_max[idx], score);
                float prev_scale = exp(tg_max[idx] - next_max);
                float cur_scale = exp(score - next_max);
                tg_denom[idx] = tg_denom[idx] * prev_scale + cur_scale;
                tg_max[idx] = next_max;
                tg_prev[idx] = prev_scale;
                tg_cur[idx] = cur_scale;
            } else {
                tg_prev[idx] = 1.0f;
                tg_cur[idx] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float vv = (i < args.head_dim) ? qw3_gqa_cache_load(v_cache, v_cache_f16, kv_idx, args.kv_type) : 0.0f;
        for (uint gh = 0; gh < 8u; gh++) {
            if (gh < group_heads) {
                acc0[gh] = acc0[gh] * tg_prev[gh] + vv * tg_cur[gh];
                acc1[gh] = acc1[gh] * tg_prev[8u + gh] + vv * tg_cur[8u + gh];
                acc2[gh] = acc2[gh] * tg_prev[16u + gh] + vv * tg_cur[16u + gh];
                acc3[gh] = acc3[gh] * tg_prev[24u + gh] + vv * tg_cur[24u + gh];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i < args.head_dim) {
        for (uint gh = 0; gh < 8u; gh++) {
            uint qh = first_qh + gh;
            if (gh < group_heads && qh < args.n_heads) {
                uint gid = qh * args.head_dim + i;
                float sig0 = 1.0f / (1.0f + exp(-qrow0[args.gate_offset + gid]));
                qrow0[args.out_offset + gid] = (acc0[gh] / tg_denom[gh]) * sig0;
                if (valid1) { float sig1 = 1.0f / (1.0f + exp(-qrow1[args.gate_offset + gid])); qrow1[args.out_offset + gid] = (acc1[gh] / tg_denom[8u + gh]) * sig1; }
                if (valid2) { float sig2 = 1.0f / (1.0f + exp(-qrow2[args.gate_offset + gid])); qrow2[args.out_offset + gid] = (acc2[gh] / tg_denom[16u + gh]) * sig2; }
                if (valid3) { float sig3 = 1.0f / (1.0f + exp(-qrow3[args.gate_offset + gid])); qrow3[args.out_offset + gid] = (acc3[gh] / tg_denom[24u + gh]) * sig3; }
            }
        }
    }
}
kernel void qw3_gqa_prefill_cached_attend_src8(constant qw3_gqa_prefill_cached_attn_args &args,
                                               device float *scratch,
                                               device const float *k_cache,
                                               device const float *v_cache,
                                               device const half *k_cache_f16,
                                               device const half *v_cache_f16,
                                               threadgroup float *sh,
                                               uint group [[threadgroup_position_in_grid]],
                                               ushort tid [[thread_index_in_threadgroup]],
                                               ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                               ushort lane [[thread_index_in_simdgroup]],
                                               ushort nt [[threads_per_threadgroup]]) {
    uint query0 = (group / args.n_kv_heads) * 4u;
    uint kvh = group - (group / args.n_kv_heads) * args.n_kv_heads;
    if (query0 >= args.n_tokens || kvh >= args.n_kv_heads || args.head_dim > uint(nt)) return;
    uint query1 = query0 + 1u;
    uint query2 = query0 + 2u;
    uint query3 = query0 + 3u;
    bool valid1 = query1 < args.n_tokens;
    bool valid2 = query2 < args.n_tokens;
    bool valid3 = query3 < args.n_tokens;
    uint max_query = valid3 ? query3 : (valid2 ? query2 : (valid1 ? query1 : query0));
    uint n_ctx = args.pos0 + max_query + 1u;
    if (n_ctx > args.ctx_size) return;
    uint i = uint(tid);
    uint group_heads = args.n_heads / args.n_kv_heads;
    if (group_heads == 0u || group_heads > 8u) return;
    uint first_qh = kvh * group_heads;
    uint kv_n = args.n_kv_heads * args.head_dim;
    device float *qrow0 = scratch + uint64_t(query0) * args.stride;
    device float *qrow1 = valid1 ? (scratch + uint64_t(query1) * args.stride) : qrow0;
    device float *qrow2 = valid2 ? (scratch + uint64_t(query2) * args.stride) : qrow0;
    device float *qrow3 = valid3 ? (scratch + uint64_t(query3) * args.stride) : qrow0;
    float scale = rsqrt(float(args.head_dim));
    threadgroup float *dots = sh;
    threadgroup float *scores = sh + 2048u;
    threadgroup float *tg_max = sh + 2304u;
    threadgroup float *tg_denom = sh + 2336u;
    threadgroup float *tg_prev = sh + 2368u;
    threadgroup float *tg_w = sh + 2400u;
    float qv0[8]; float qv1[8]; float qv2[8]; float qv3[8];
    float acc0[8]; float acc1[8]; float acc2[8]; float acc3[8];
    for (uint gh = 0; gh < 8u; gh++) {
        uint qh = first_qh + gh;
        bool active = gh < group_heads && i < args.head_dim && qh < args.n_heads;
        uint64_t off = args.q_offset + uint64_t(qh) * args.head_dim + i;
        qv0[gh] = active ? qrow0[off] : 0.0f;
        qv1[gh] = (active && valid1) ? qrow1[off] : 0.0f;
        qv2[gh] = (active && valid2) ? qrow2[off] : 0.0f;
        qv3[gh] = (active && valid3) ? qrow3[off] : 0.0f;
        acc0[gh] = 0.0f; acc1[gh] = 0.0f; acc2[gh] = 0.0f; acc3[gh] = 0.0f;
    }
    if (tid < 32u) {
        tg_max[tid] = -FLT_MAX;
        tg_denom[tid] = 0.0f;
        tg_prev[tid] = 1.0f;
    }
    if (tid < 256u) tg_w[tid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n_simd = (uint(nt) + 31u) >> 5u;
    for (uint src0 = 0; src0 < n_ctx; src0 += 8u) {
        for (uint so = 0u; so < 8u; so++) {
            uint src = src0 + so;
            bool valid_src = src < n_ctx;
            uint64_t kv_idx = uint64_t(src) * kv_n + uint64_t(kvh) * args.head_dim + i;
            float kval = (valid_src && i < args.head_dim) ? qw3_gqa_cache_load(k_cache, k_cache_f16, kv_idx, args.kv_type) : 0.0f;
            bool causal0 = valid_src && src <= args.pos0 + query0;
            bool causal1 = valid1 && valid_src && src <= args.pos0 + query1;
            bool causal2 = valid2 && valid_src && src <= args.pos0 + query2;
            bool causal3 = valid3 && valid_src && src <= args.pos0 + query3;
            for (uint gh = 0; gh < 8u; gh++) {
                bool active = gh < group_heads && i < args.head_dim;
                float part0 = (active && causal0) ? qv0[gh] * kval : 0.0f;
                float part1 = (active && causal1) ? qv1[gh] * kval : 0.0f;
                float part2 = (active && causal2) ? qv2[gh] * kval : 0.0f;
                float part3 = (active && causal3) ? qv3[gh] * kval : 0.0f;
                part0 = simd_sum(part0); part1 = simd_sum(part1);
                part2 = simd_sum(part2); part3 = simd_sum(part3);
                if (lane == 0) {
                    uint base = ((gh * 8u + so) * 8u) + uint(simd_idx);
                    dots[base] = part0;
                    dots[512u + base] = part1;
                    dots[1024u + base] = part2;
                    dots[1536u + base] = part3;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint qblk = 0u; qblk < 4u; qblk++) {
            for (uint gh = 0u; gh < 8u; gh++) {
                for (uint so = 0u; so < 8u; so++) {
                    uint base = qblk * 512u + (gh * 8u + so) * 8u + uint(tid);
                    float dot = (gh < group_heads && uint(tid) < n_simd) ? dots[base] : 0.0f;
                    dot = simd_sum(dot);
                    if (tid == 0) scores[(qblk * 8u + gh) * 8u + so] = dot;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32u) {
            uint idx = uint(tid);
            uint qblk = idx >> 3u;
            uint gh = idx & 7u;
            bool qvalid = qblk == 0u || (qblk == 1u ? valid1 : (qblk == 2u ? valid2 : valid3));
            uint query = query0 + qblk;
            float local_max = -FLT_MAX;
            if (qvalid && gh < group_heads) {
                for (uint so = 0u; so < 8u; so++) {
                    uint src = src0 + so;
                    if (src < n_ctx && src <= args.pos0 + query) {
                        local_max = max(local_max, scores[idx * 8u + so] * scale);
                    }
                }
            }
            if (local_max > -FLT_MAX / 4.0f) {
                float next_max = max(tg_max[idx], local_max);
                float prev_scale = exp(tg_max[idx] - next_max);
                float add = 0.0f;
                for (uint so = 0u; so < 8u; so++) {
                    uint src = src0 + so;
                    float w = 0.0f;
                    if (src < n_ctx && src <= args.pos0 + query) {
                        w = exp(scores[idx * 8u + so] * scale - next_max);
                        add += w;
                    }
                    tg_w[idx * 8u + so] = w;
                }
                tg_denom[idx] = tg_denom[idx] * prev_scale + add;
                tg_max[idx] = next_max;
                tg_prev[idx] = prev_scale;
            } else {
                tg_prev[idx] = 1.0f;
                for (uint so = 0u; so < 8u; so++) tg_w[idx * 8u + so] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint gh = 0u; gh < 8u; gh++) {
            if (gh < group_heads) {
                acc0[gh] *= tg_prev[gh];
                acc1[gh] *= tg_prev[8u + gh];
                acc2[gh] *= tg_prev[16u + gh];
                acc3[gh] *= tg_prev[24u + gh];
            }
        }
        for (uint so = 0u; so < 8u; so++) {
            uint src = src0 + so;
            bool valid_src = src < n_ctx;
            uint64_t kv_idx = uint64_t(src) * kv_n + uint64_t(kvh) * args.head_dim + i;
            float vv = (valid_src && i < args.head_dim) ? qw3_gqa_cache_load(v_cache, v_cache_f16, kv_idx, args.kv_type) : 0.0f;
            for (uint gh = 0u; gh < 8u; gh++) {
                if (gh < group_heads) {
                    acc0[gh] += vv * tg_w[gh * 8u + so];
                    acc1[gh] += vv * tg_w[(8u + gh) * 8u + so];
                    acc2[gh] += vv * tg_w[(16u + gh) * 8u + so];
                    acc3[gh] += vv * tg_w[(24u + gh) * 8u + so];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i < args.head_dim) {
        for (uint gh = 0; gh < 8u; gh++) {
            uint qh = first_qh + gh;
            if (gh < group_heads && qh < args.n_heads) {
                uint gid = qh * args.head_dim + i;
                float sig0 = 1.0f / (1.0f + exp(-qrow0[args.gate_offset + gid]));
                qrow0[args.out_offset + gid] = (acc0[gh] / tg_denom[gh]) * sig0;
                if (valid1) { float sig1 = 1.0f / (1.0f + exp(-qrow1[args.gate_offset + gid])); qrow1[args.out_offset + gid] = (acc1[gh] / tg_denom[8u + gh]) * sig1; }
                if (valid2) { float sig2 = 1.0f / (1.0f + exp(-qrow2[args.gate_offset + gid])); qrow2[args.out_offset + gid] = (acc2[gh] / tg_denom[16u + gh]) * sig2; }
                if (valid3) { float sig3 = 1.0f / (1.0f + exp(-qrow3[args.gate_offset + gid])); qrow3[args.out_offset + gid] = (acc3[gh] / tg_denom[24u + gh]) * sig3; }
            }
        }
    }
}
kernel void qw3_gqa_attend_n_q8_inner(constant qw3_gqa_n_args &args,
                                     device const float *q,
                                     device const float *gate,
                                     device const uchar *k_cache,
                                     device const uchar *v_cache,
                                     device float *out,
                                     threadgroup float *sh,
                                     uint h [[threadgroup_position_in_grid]],
                                     ushort tid [[thread_index_in_threadgroup]],
                                     ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                     ushort lane [[thread_index_in_simdgroup]],
                                     ushort nt [[threads_per_threadgroup]]) {
    if (h >= args.n_kv_heads || args.n_ctx == 0 || args.head_dim > uint(nt)) return;
    uint i = uint(tid);
    uint kvh = h;
    uint group_heads = args.n_heads / args.n_kv_heads;
    if (group_heads == 0u || group_heads > 8u) return;
    uint first_qh = kvh * group_heads;
    uint blocks_per_head = args.head_dim / 32u;
    float scale = rsqrt(float(args.head_dim));
    float qv[8];
    float max_score[8];
    float denom[8];
    float acc[8];
    for (uint gh = 0; gh < 8u; gh++) {
        uint qh = first_qh + gh;
        qv[gh] = (gh < group_heads && i < args.head_dim && qh < args.n_heads) ? q[uint64_t(qh) * args.head_dim + i] : 0.0f;
        max_score[gh] = -FLT_MAX;
        denom[gh] = 0.0f;
        acc[gh] = 0.0f;
    }
    uint n_simd = (uint(nt) + 31u) >> 5u;
    for (uint t = 0; t < args.n_ctx; t++) {
        uint64_t b = (uint64_t(t) * args.n_kv_heads + kvh) * blocks_per_head + i / 32u;
        device const uchar *kb = k_cache + b * 34ull;
        device const uchar *vb = v_cache + b * 34ull;
        half kd = *((device const half *)kb);
        half vd = *((device const half *)vb);
        char kq = *((device const char *)(kb + 2u + i % 32u));
        char vq = *((device const char *)(vb + 2u + i % 32u));
        float kval = (i < args.head_dim) ? float(kd) * float(kq) : 0.0f;
        float vv = (i < args.head_dim) ? float(vd) * float(vq) : 0.0f;
        for (uint gh = 0; gh < 8u; gh++) {
            float part = (gh < group_heads && i < args.head_dim) ? qv[gh] * kval : 0.0f;
            part = simd_sum(part);
            if (lane == 0) sh[gh * 8u + uint(simd_idx)] = part;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint gh = 0; gh < 8u; gh++) {
            float dot = (gh < group_heads && uint(tid) < n_simd) ? sh[gh * 8u + uint(tid)] : 0.0f;
            dot = simd_sum(dot);
            if (tid == 0) sh[gh] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint gh = 0; gh < 8u; gh++) {
            if (gh < group_heads) {
                float score = sh[gh] * scale;
                float next_max = max(max_score[gh], score);
                float prev_scale = exp(max_score[gh] - next_max);
                float cur_scale = exp(score - next_max);
                acc[gh] = acc[gh] * prev_scale + vv * cur_scale;
                denom[gh] = denom[gh] * prev_scale + cur_scale;
                max_score[gh] = next_max;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i < args.head_dim) {
        for (uint gh = 0; gh < 8u; gh++) {
            uint qh = first_qh + gh;
            if (gh < group_heads && qh < args.n_heads) {
                uint gid = qh * args.head_dim + i;
                float sig = 1.0f / (1.0f + exp(-gate[gid]));
                out[gid] = (acc[gh] / denom[gh]) * sig;
            }
        }
    }
}
struct qw3_gqa_q8_split_args { uint n_ctx; uint n_heads; uint n_kv_heads; uint head_dim; uint n_splits; };
kernel void qw3_gqa_attend_n_q8_split_partial(constant qw3_gqa_q8_split_args &args,
                                             device const float *q,
                                             device const uchar *k_cache,
                                             device const uchar *v_cache,
                                             device float *partial,
                                             threadgroup float *sh,
                                             uint group [[threadgroup_position_in_grid]],
                                             ushort tid [[thread_index_in_threadgroup]],
                                             ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                             ushort lane [[thread_index_in_simdgroup]],
                                             ushort nt [[threads_per_threadgroup]]) {
    uint kvh = group % args.n_kv_heads;
    uint split = group / args.n_kv_heads;
    if (kvh >= args.n_kv_heads || split >= args.n_splits || args.head_dim > uint(nt)) return;
    uint t0 = uint((uint64_t(split) * uint64_t(args.n_ctx)) / uint64_t(args.n_splits));
    uint t1 = uint((uint64_t(split + 1u) * uint64_t(args.n_ctx)) / uint64_t(args.n_splits));
    if (t0 >= t1) return;
    uint i = uint(tid);
    uint group_heads = args.n_heads / args.n_kv_heads;
    if (group_heads == 0u || group_heads > 8u) return;
    uint first_qh = kvh * group_heads;
    uint blocks_per_head = args.head_dim / 32u;
    float scale = rsqrt(float(args.head_dim));
    float qv[8];
    float max_score[8];
    float denom[8];
    float acc[8];
    for (uint gh = 0; gh < 8u; gh++) {
        uint qh = first_qh + gh;
        qv[gh] = (gh < group_heads && i < args.head_dim && qh < args.n_heads) ? q[uint64_t(qh) * args.head_dim + i] : 0.0f;
        max_score[gh] = -FLT_MAX;
        denom[gh] = 0.0f;
        acc[gh] = 0.0f;
    }
    uint n_simd = (uint(nt) + 31u) >> 5u;
    for (uint t = t0; t < t1; t++) {
        uint64_t b = (uint64_t(t) * args.n_kv_heads + kvh) * blocks_per_head + i / 32u;
        device const uchar *kb = k_cache + b * 34ull;
        device const uchar *vb = v_cache + b * 34ull;
        half kd = *((device const half *)kb);
        half vd = *((device const half *)vb);
        char kq = *((device const char *)(kb + 2u + i % 32u));
        char vq = *((device const char *)(vb + 2u + i % 32u));
        float kval = (i < args.head_dim) ? float(kd) * float(kq) : 0.0f;
        float vv = (i < args.head_dim) ? float(vd) * float(vq) : 0.0f;
        for (uint gh = 0; gh < 8u; gh++) {
            float part = (gh < group_heads && i < args.head_dim) ? qv[gh] * kval : 0.0f;
            part = simd_sum(part);
            if (lane == 0) sh[gh * 8u + uint(simd_idx)] = part;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint gh = 0; gh < 8u; gh++) {
            float dot = (gh < group_heads && uint(tid) < n_simd) ? sh[gh * 8u + uint(tid)] : 0.0f;
            dot = simd_sum(dot);
            if (tid == 0) sh[gh] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint gh = 0; gh < 8u; gh++) {
            if (gh < group_heads) {
                float score = sh[gh] * scale;
                float next_max = max(max_score[gh], score);
                float prev_scale = exp(max_score[gh] - next_max);
                float cur_scale = exp(score - next_max);
                acc[gh] = acc[gh] * prev_scale + vv * cur_scale;
                denom[gh] = denom[gh] * prev_scale + cur_scale;
                max_score[gh] = next_max;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i < args.head_dim) {
        uint stride = args.head_dim + 2u;
        for (uint gh = 0; gh < group_heads; gh++) {
            uint qh = first_qh + gh;
            uint64_t base = (uint64_t(split) * args.n_heads + qh) * stride;
            partial[base + i] = acc[gh];
            if (i == 0u) {
                partial[base + args.head_dim] = denom[gh];
                partial[base + args.head_dim + 1u] = max_score[gh];
            }
        }
    }
}
kernel void qw3_gqa_attend_n_q8_split_reduce(constant qw3_gqa_q8_split_args &args,
                                            device const float *gate,
                                            device const float *partial,
                                            device float *out,
                                            uint qh [[threadgroup_position_in_grid]],
                                            ushort tid [[thread_index_in_threadgroup]]) {
    if (qh >= args.n_heads || uint(tid) >= args.head_dim) return;
    uint i = uint(tid);
    uint stride = args.head_dim + 2u;
    float max_score = -FLT_MAX;
    float denom = 0.0f;
    float acc = 0.0f;
    for (uint split = 0; split < args.n_splits; split++) {
        uint64_t base = (uint64_t(split) * args.n_heads + qh) * stride;
        float local_denom = partial[base + args.head_dim];
        float local_max = partial[base + args.head_dim + 1u];
        float next_max = max(max_score, local_max);
        float prev_scale = exp(max_score - next_max);
        float cur_scale = exp(local_max - next_max);
        acc = acc * prev_scale + partial[base + i] * cur_scale;
        denom = denom * prev_scale + local_denom * cur_scale;
        max_score = next_max;
    }
    uint gid = qh * args.head_dim + i;
    float sig = 1.0f / (1.0f + exp(-gate[gid]));
    out[gid] = (acc / denom) * sig;
}
struct qw3_recur_zero_args { uint q_heads; uint v_heads; uint head_dim; };
kernel void qw3_deltanet_recur_zero(constant qw3_recur_zero_args &args,
                                     device const float *q,
                                     device const float *k,
                                     device const float *v,
                                     device const float *beta,
                                     device float *state_out,
                                     device float *core_out,
                                     threadgroup float *sh,
                                     uint hv [[threadgroup_position_in_grid]],
                                     ushort tid [[thread_index_in_threadgroup]],
                                     ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                     ushort lane [[thread_index_in_simdgroup]],
                                     ushort nt [[threads_per_threadgroup]]) {
    if (hv >= args.v_heads) return;
    uint hk = hv % args.q_heads;
    device const float *qh = q + uint64_t(hk) * args.head_dim;
    device const float *kh = k + uint64_t(hk) * args.head_dim;
    device const float *vh = v + uint64_t(hv) * args.head_dim;
    float dot = 0.0f;
    for (uint i = tid; i < args.head_dim; i += nt) dot += qh[i] * kh[i];
    dot = simd_sum(dot);
    if (lane == 0) sh[simd_idx] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dot = lane < 32 ? sh[lane] : 0.0f;
    dot = simd_sum(dot);
    float b = beta[hv];
    float scale = dot * rsqrt(float(args.head_dim)) * b;
    for (uint j = tid; j < args.head_dim; j += nt) core_out[uint64_t(hv) * args.head_dim + j] = scale * vh[j];
    uint state_n = args.head_dim * args.head_dim;
    device float *shv = state_out + uint64_t(hv) * state_n;
    for (uint idx = tid; idx < state_n; idx += nt) {
        uint i = idx / args.head_dim;
        uint j = idx - i * args.head_dim;
        shv[idx] = kh[i] * b * vh[j];
    }
}
kernel void qw3_deltanet_recur(constant qw3_recur_zero_args &args,
                                device const float *state_in,
                                device const float *q,
                                device const float *k,
                                device const float *v,
                                device const float *beta,
                                device const float *gamma,
                                device float *state_out,
                                device float *core_out,
                                uint2 pos [[thread_position_in_grid]]) {
    uint j = pos.x;
    uint hv = pos.y;
    if (hv >= args.v_heads || j >= args.head_dim) return;
    uint hk = hv % args.q_heads;
    uint state_n = args.head_dim * args.head_dim;
    device const float *qh = q + uint64_t(hk) * args.head_dim;
    device const float *kh = k + uint64_t(hk) * args.head_dim;
    device const float *vh = v + uint64_t(hv) * args.head_dim;
    device const float *sin = state_in + uint64_t(hv) * state_n;
    device float *sout = state_out + uint64_t(hv) * state_n;
    float g = gamma[hv];
    float sk = 0.0f;
    for (uint i = 0; i < args.head_dim; i++) sk += sin[i * args.head_dim + j] * g * kh[i];
    float d = beta[hv] * (vh[j] - sk);
    float out = 0.0f;
    for (uint i = 0; i < args.head_dim; i++) {
        uint idx = i * args.head_dim + j;
        float sv = sin[idx] * g + kh[i] * d;
        sout[idx] = sv;
        out += sv * qh[i];
    }
    core_out[uint64_t(hv) * args.head_dim + j] = out * rsqrt(float(args.head_dim));
}
struct qw3_recur_scratch_args { uint q_heads; uint v_heads; uint head_dim; uint alpha_offset; uint beta_offset; };
kernel void qw3_deltanet_recur_scratch_gates(constant qw3_recur_scratch_args &args,
                                             device const float *state_in,
                                             device const float *q,
                                             device const float *k,
                                             device const float *v,
                                             device const float *scratch,
                                             device const float *dt_bias,
                                             device const float *a,
                                             device float *state_out,
                                             device float *core_out,
                                             uint2 pos [[thread_position_in_grid]]) {
    uint j = pos.x;
    uint hv = pos.y;
    if (hv >= args.v_heads || j >= args.head_dim) return;
    uint hk = hv % args.q_heads;
    uint state_n = args.head_dim * args.head_dim;
    device const float *qh = q + uint64_t(hk) * args.head_dim;
    device const float *kh = k + uint64_t(hk) * args.head_dim;
    device const float *vh = v + uint64_t(hv) * args.head_dim;
    device const float *sin = state_in + uint64_t(hv) * state_n;
    device float *sout = state_out + uint64_t(hv) * state_n;
    float beta_raw = scratch[args.beta_offset + hv];
    float b = 1.0f / (1.0f + exp(-beta_raw));
    float alpha_raw = scratch[args.alpha_offset + hv] + dt_bias[hv];
    float sp = alpha_raw > 20.0f ? alpha_raw : (alpha_raw < -20.0f ? exp(alpha_raw) : log(1.0f + exp(alpha_raw)));
    float g = exp(sp * a[hv]);
    float sk = 0.0f;
    for (uint i = 0; i < args.head_dim; i++) sk += sin[i * args.head_dim + j] * g * kh[i];
    float d = b * (vh[j] - sk);
    float out = 0.0f;
    for (uint i = 0; i < args.head_dim; i++) {
        uint idx = i * args.head_dim + j;
        float sv = sin[idx] * g + kh[i] * d;
        sout[idx] = sv;
        out += sv * qh[i];
    }
    core_out[uint64_t(hv) * args.head_dim + j] = out * rsqrt(float(args.head_dim));
}
kernel void qw3_deltanet_prepare_scratch_gates(constant qw3_recur_scratch_args &args,
                                                device float *scratch,
                                                device const float *dt_bias,
                                                device const float *a,
                                                uint hv [[thread_position_in_grid]]) {
    if (hv >= args.v_heads) return;
    float beta_raw = scratch[args.beta_offset + hv];
    scratch[args.beta_offset + hv] = 1.0f / (1.0f + exp(-beta_raw));
    float alpha_raw = scratch[args.alpha_offset + hv] + dt_bias[hv];
    float sp = alpha_raw > 20.0f ? alpha_raw : (alpha_raw < -20.0f ? exp(alpha_raw) : log(1.0f + exp(alpha_raw)));
    scratch[args.alpha_offset + hv] = exp(sp * a[hv]);
}
kernel void qw3_deltanet_recur_scratch_gates_tiled(constant qw3_recur_scratch_args &args,
                                                   device const float *state_in,
                                                   device const float *q,
                                                   device const float *k,
                                                   device const float *v,
                                                   device const float *scratch,
                                                   device const float *dt_bias,
                                                   device const float *a,
                                                   device float *state_out,
                                                   device float *core_out,
                                                   uint2 group [[threadgroup_position_in_grid]],
                                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                                   ushort lane [[thread_index_in_simdgroup]]) {
    uint hv = group.y;
    uint j = group.x * 4u + uint(simd_idx);
    if (hv >= args.v_heads || j >= args.head_dim) return;
    uint hk = hv % args.q_heads;
    uint state_n = args.head_dim * args.head_dim;
    device const float *qh = q + uint64_t(hk) * args.head_dim;
    device const float *kh = k + uint64_t(hk) * args.head_dim;
    device const float *vh = v + uint64_t(hv) * args.head_dim;
    device const float *sin = state_in + uint64_t(hv) * state_n;
    device float *sout = state_out + uint64_t(hv) * state_n;
    float b = 0.0f;
    float g = 0.0f;
    if (lane == 0) {
        b = scratch[args.beta_offset + hv];
        g = scratch[args.alpha_offset + hv];
    }
    b = simd_broadcast(b, 0);
    g = simd_broadcast(g, 0);
    uint i0 = uint(lane) * 4u;
    uint state_col = j * args.head_dim + i0;
    float4 sv = *((device const float4 *)(sin + state_col));
    float4 kv = *((device const float4 *)(kh + i0));
    float sk = simd_sum(dot(sv, kv));
    float d = b * (vh[j] - sk * g);
    sv = sv * g + kv * d;
    *((device float4 *)(sout + state_col)) = sv;
    float4 qv = *((device const float4 *)(qh + i0));
    float out = simd_sum(dot(sv, qv));
    if (lane == 0) core_out[uint64_t(hv) * args.head_dim + j] = out * rsqrt(float(args.head_dim));
}
struct qw3_fused_gdn_args { uint q_heads; uint v_heads; uint head_dim; uint alpha_offset; uint beta_offset; uint z_offset; float eps; };
kernel void qw3_deltanet_fused_gdn_scratch(constant qw3_fused_gdn_args &args,
                                            device const float *state_in,
                                            device const float *q,
                                            device const float *k,
                                            device const float *v,
                                            device const float *scratch,
                                            device const float *dt_bias,
                                            device const float *a,
                                            device const float *w,
                                            device float *state_out,
                                            device float *inner_out,
                                            threadgroup float *sh,
                                            uint hv [[threadgroup_position_in_grid]],
                                            ushort j [[thread_index_in_threadgroup]],
                                            ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                            ushort lane [[thread_index_in_simdgroup]]) {
    if (hv >= args.v_heads || j >= args.head_dim) return;
    uint hk = hv % args.q_heads;
    uint state_n = args.head_dim * args.head_dim;
    device const float *qh = q + uint64_t(hk) * args.head_dim;
    device const float *kh = k + uint64_t(hk) * args.head_dim;
    device const float *vh = v + uint64_t(hv) * args.head_dim;
    device const float *sin = state_in + uint64_t(hv) * state_n;
    device float *sout = state_out + uint64_t(hv) * state_n;
    if (j == 0) {
        float beta_raw = scratch[args.beta_offset + hv];
        sh[0] = 1.0f / (1.0f + exp(-beta_raw));
        float alpha_raw = scratch[args.alpha_offset + hv] + dt_bias[hv];
        float sp = alpha_raw > 20.0f ? alpha_raw : (alpha_raw < -20.0f ? exp(alpha_raw) : log(1.0f + exp(alpha_raw)));
        sh[1] = exp(sp * a[hv]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float b = sh[0];
    float g = sh[1];
    float sk = 0.0f;
    for (uint i = 0; i < args.head_dim; i++) sk += sin[i * args.head_dim + j] * kh[i];
    float d = b * (vh[j] - sk * g);
    float sum = 0.0f;
    for (uint i = 0; i < args.head_dim; i++) {
        uint idx = i * args.head_dim + j;
        float sv = sin[idx] * g + kh[i] * d;
        sout[idx] = sv;
        sum += sv * qh[i];
    }
    float core = sum * rsqrt(float(args.head_dim));
    float ss = simd_sum(core * core);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.head_dim) + args.eps);
    float zi = scratch[args.z_offset + uint64_t(hv) * args.head_dim + j];
    float gate = zi / (1.0f + exp(-zi));
    inner_out[uint64_t(hv) * args.head_dim + j] = core * scale * w[j] * gate;
}
struct qw3_batch_gdn_args { uint q_heads; uint v_heads; uint head_dim; uint n_tokens; uint conv_offset; uint z_offset; uint alpha_offset; uint beta_offset; uint inner_offset; uint stride; float eps; };
kernel void qw3_deltanet_batch_fused_gdn(constant qw3_batch_gdn_args &args,
                                        device float *state,
                                        device float *scratch,
                                        device const float *dt_bias,
                                        device const float *a,
                                        device const float *w,
                                        threadgroup float *sh,
                                        uint hv [[threadgroup_position_in_grid]],
                                        ushort j [[thread_index_in_threadgroup]],
                                        ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                        ushort lane [[thread_index_in_simdgroup]]) {
    if (hv >= args.v_heads || j >= args.head_dim) return;
    uint hk = hv % args.q_heads;
    uint qk_n = args.q_heads * args.head_dim;
    uint state_n = args.head_dim * args.head_dim;
    device float *st = state + uint64_t(hv) * state_n;
    for (uint t = 0; t < args.n_tokens; t++) {
        uint64_t base = uint64_t(t) * args.stride;
        device const float *qh = scratch + base + args.conv_offset + uint64_t(hk) * args.head_dim;
        device const float *kh = scratch + base + args.conv_offset + qk_n + uint64_t(hk) * args.head_dim;
        device const float *vh = scratch + base + args.conv_offset + 2u * qk_n + uint64_t(hv) * args.head_dim;
        if (j == 0) {
            float beta_raw = scratch[base + args.beta_offset + hv];
            sh[0] = 1.0f / (1.0f + exp(-beta_raw));
            float alpha_raw = scratch[base + args.alpha_offset + hv] + dt_bias[hv];
            float sp = alpha_raw > 20.0f ? alpha_raw : (alpha_raw < -20.0f ? exp(alpha_raw) : log(1.0f + exp(alpha_raw)));
            sh[1] = exp(sp * a[hv]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float b = sh[0];
        float g = sh[1];
        float sk = 0.0f;
        for (uint i = 0; i < args.head_dim; i++) sk += st[j * args.head_dim + i] * kh[i];
        float d = b * (vh[j] - sk * g);
        float sum = 0.0f;
        for (uint i = 0; i < args.head_dim; i++) {
            uint idx = j * args.head_dim + i;
            float sv = st[idx] * g + kh[i] * d;
            st[idx] = sv;
            sum += sv * qh[i];
        }
        float core = sum * rsqrt(float(args.head_dim));
        float ss = simd_sum(core * core);
        if (lane == 0) sh[2u + simd_idx] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ss = lane < 32 ? sh[2u + lane] : 0.0f;
        ss = simd_sum(ss);
        float scale = rsqrt(ss / float(args.head_dim) + args.eps);
        float zi = scratch[base + args.z_offset + uint64_t(hv) * args.head_dim + j];
        float gate = zi / (1.0f + exp(-zi));
        scratch[base + args.inner_offset + uint64_t(hv) * args.head_dim + j] = core * scale * w[j] * gate;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
kernel void qw3_deltanet_batch_recur_tiled(constant qw3_batch_gdn_args &args,
                                           device float *state,
                                           device float *scratch,
                                           device const float *dt_bias,
                                           device const float *a,
                                           uint2 group [[threadgroup_position_in_grid]],
                                           ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                           ushort lane [[thread_index_in_simdgroup]]) {
    uint hv = group.y;
    uint j = group.x * 4u + uint(simd_idx);
    if (hv >= args.v_heads || j >= args.head_dim) return;
    uint hk = hv % args.q_heads;
    uint qk_n = args.q_heads * args.head_dim;
    uint state_n = args.head_dim * args.head_dim;
    device float *st = state + uint64_t(hv) * state_n;
    uint i0 = uint(lane) * 4u;
    float a_hv = a[hv];
    float dt_hv = dt_bias[hv];
    float norm_scale = rsqrt(float(args.head_dim));
    for (uint t = 0; t < args.n_tokens; t++) {
        uint64_t base = uint64_t(t) * args.stride;
        device const float *qh = scratch + base + args.conv_offset + uint64_t(hk) * args.head_dim;
        device const float *kh = scratch + base + args.conv_offset + qk_n + uint64_t(hk) * args.head_dim;
        device const float *vh = scratch + base + args.conv_offset + 2u * qk_n + uint64_t(hv) * args.head_dim;
        float b = 0.0f;
        float g = 0.0f;
        if (lane == 0) {
            float beta_raw = scratch[base + args.beta_offset + hv];
            b = 1.0f / (1.0f + exp(-beta_raw));
            float alpha_raw = scratch[base + args.alpha_offset + hv] + dt_hv;
            float sp = alpha_raw > 20.0f ? alpha_raw : (alpha_raw < -20.0f ? exp(alpha_raw) : log(1.0f + exp(alpha_raw)));
            g = exp(sp * a_hv);
        }
        b = simd_broadcast(b, 0);
        g = simd_broadcast(g, 0);
        uint state_col = j * args.head_dim + i0;
        float4 sv = *((device const float4 *)(st + state_col));
        float4 kv = *((device const float4 *)(kh + i0));
        float sk = simd_sum(dot(sv, kv));
        float d = b * (vh[j] - sk * g);
        sv = sv * g + kv * d;
        *((device float4 *)(st + state_col)) = sv;
        float4 qv = *((device const float4 *)(qh + i0));
        float out = simd_sum(dot(sv, qv));
        if (lane == 0) scratch[base + args.inner_offset + uint64_t(hv) * args.head_dim + j] = out * norm_scale;
    }
}
kernel void qw3_deltanet_batch_recur_tiled2(constant qw3_batch_gdn_args &args,
                                            device float *state,
                                            device float *scratch,
                                            device const float *dt_bias,
                                            device const float *a,
                                            uint2 group [[threadgroup_position_in_grid]],
                                            ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                            ushort lane [[thread_index_in_simdgroup]]) {
    uint hv = group.y;
    uint j0 = group.x * 8u + uint(simd_idx) * 2u;
    uint j1 = j0 + 1u;
    if (hv >= args.v_heads || j0 >= args.head_dim) return;
    uint hk = hv % args.q_heads;
    uint qk_n = args.q_heads * args.head_dim;
    uint state_n = args.head_dim * args.head_dim;
    device float *st = state + uint64_t(hv) * state_n;
    uint i0 = uint(lane) * 4u;
    float a_hv = a[hv];
    float dt_hv = dt_bias[hv];
    float norm_scale = rsqrt(float(args.head_dim));
    for (uint t = 0; t < args.n_tokens; t++) {
        uint64_t base = uint64_t(t) * args.stride;
        device const float *qh = scratch + base + args.conv_offset + uint64_t(hk) * args.head_dim;
        device const float *kh = scratch + base + args.conv_offset + qk_n + uint64_t(hk) * args.head_dim;
        device const float *vh = scratch + base + args.conv_offset + 2u * qk_n + uint64_t(hv) * args.head_dim;
        float b = 0.0f;
        float g = 0.0f;
        if (lane == 0) {
            float beta_raw = scratch[base + args.beta_offset + hv];
            b = 1.0f / (1.0f + exp(-beta_raw));
            float alpha_raw = scratch[base + args.alpha_offset + hv] + dt_hv;
            float sp = alpha_raw > 20.0f ? alpha_raw : (alpha_raw < -20.0f ? exp(alpha_raw) : log(1.0f + exp(alpha_raw)));
            g = exp(sp * a_hv);
        }
        b = simd_broadcast(b, 0);
        g = simd_broadcast(g, 0);
        float4 kv = *((device const float4 *)(kh + i0));
        float4 qv = *((device const float4 *)(qh + i0));
        uint state_col0 = j0 * args.head_dim + i0;
        float4 sv0 = *((device const float4 *)(st + state_col0));
        float sk0 = simd_sum(dot(sv0, kv));
        float d0 = b * (vh[j0] - sk0 * g);
        sv0 = sv0 * g + kv * d0;
        *((device float4 *)(st + state_col0)) = sv0;
        float out0 = simd_sum(dot(sv0, qv));
        if (lane == 0) scratch[base + args.inner_offset + uint64_t(hv) * args.head_dim + j0] = out0 * norm_scale;
        uint state_col1 = j1 * args.head_dim + i0;
        float4 sv1 = *((device const float4 *)(st + state_col1));
        float sk1 = simd_sum(dot(sv1, kv));
        float d1 = b * (vh[j1] - sk1 * g);
        sv1 = sv1 * g + kv * d1;
        *((device float4 *)(st + state_col1)) = sv1;
        float out1 = simd_sum(dot(sv1, qv));
        if (lane == 0) scratch[base + args.inner_offset + uint64_t(hv) * args.head_dim + j1] = out1 * norm_scale;
    }
}
kernel void qw3_deltanet_batch_recur_tiled4(constant qw3_batch_gdn_args &args,
                                            device float *state,
                                            device float *scratch,
                                            device const float *dt_bias,
                                            device const float *a,
                                            uint2 group [[threadgroup_position_in_grid]],
                                            ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                            ushort lane [[thread_index_in_simdgroup]]) {
    uint hv = group.y;
    uint j0 = group.x * 16u + uint(simd_idx) * 4u;
    uint j1 = j0 + 1u;
    uint j2 = j0 + 2u;
    uint j3 = j0 + 3u;
    if (hv >= args.v_heads || j0 >= args.head_dim) return;
    uint hk = hv % args.q_heads;
    uint qk_n = args.q_heads * args.head_dim;
    uint state_n = args.head_dim * args.head_dim;
    device float *st = state + uint64_t(hv) * state_n;
    uint i0 = uint(lane) * 4u;
    for (uint t = 0; t < args.n_tokens; t++) {
        uint64_t base = uint64_t(t) * args.stride;
        device const float *qh = scratch + base + args.conv_offset + uint64_t(hk) * args.head_dim;
        device const float *kh = scratch + base + args.conv_offset + qk_n + uint64_t(hk) * args.head_dim;
        device const float *vh = scratch + base + args.conv_offset + 2u * qk_n + uint64_t(hv) * args.head_dim;
        float b = 0.0f;
        float g = 0.0f;
        if (lane == 0) {
            float beta_raw = scratch[base + args.beta_offset + hv];
            b = 1.0f / (1.0f + exp(-beta_raw));
            float alpha_raw = scratch[base + args.alpha_offset + hv] + dt_bias[hv];
            float sp = alpha_raw > 20.0f ? alpha_raw : (alpha_raw < -20.0f ? exp(alpha_raw) : log(1.0f + exp(alpha_raw)));
            g = exp(sp * a[hv]);
        }
        b = simd_broadcast(b, 0);
        g = simd_broadcast(g, 0);
        float4 kv = *((device const float4 *)(kh + i0));
        float4 qv = *((device const float4 *)(qh + i0));
        uint state_col0 = j0 * args.head_dim + i0;
        float4 sv0 = *((device const float4 *)(st + state_col0));
        float sk0 = simd_sum(dot(sv0, kv));
        float d0 = b * (vh[j0] - sk0 * g);
        sv0 = sv0 * g + kv * d0;
        *((device float4 *)(st + state_col0)) = sv0;
        float out0 = simd_sum(dot(sv0, qv));
        if (lane == 0) scratch[base + args.inner_offset + uint64_t(hv) * args.head_dim + j0] = out0 * rsqrt(float(args.head_dim));
        if (j1 < args.head_dim) {
            uint state_col1 = j1 * args.head_dim + i0;
            float4 sv1 = *((device const float4 *)(st + state_col1));
            float sk1 = simd_sum(dot(sv1, kv));
            float d1 = b * (vh[j1] - sk1 * g);
            sv1 = sv1 * g + kv * d1;
            *((device float4 *)(st + state_col1)) = sv1;
            float out1 = simd_sum(dot(sv1, qv));
            if (lane == 0) scratch[base + args.inner_offset + uint64_t(hv) * args.head_dim + j1] = out1 * rsqrt(float(args.head_dim));
        }
        if (j2 < args.head_dim) {
            uint state_col2 = j2 * args.head_dim + i0;
            float4 sv2 = *((device const float4 *)(st + state_col2));
            float sk2 = simd_sum(dot(sv2, kv));
            float d2 = b * (vh[j2] - sk2 * g);
            sv2 = sv2 * g + kv * d2;
            *((device float4 *)(st + state_col2)) = sv2;
            float out2 = simd_sum(dot(sv2, qv));
            if (lane == 0) scratch[base + args.inner_offset + uint64_t(hv) * args.head_dim + j2] = out2 * rsqrt(float(args.head_dim));
        }
        if (j3 < args.head_dim) {
            uint state_col3 = j3 * args.head_dim + i0;
            float4 sv3 = *((device const float4 *)(st + state_col3));
            float sk3 = simd_sum(dot(sv3, kv));
            float d3 = b * (vh[j3] - sk3 * g);
            sv3 = sv3 * g + kv * d3;
            *((device float4 *)(st + state_col3)) = sv3;
            float out3 = simd_sum(dot(sv3, qv));
            if (lane == 0) scratch[base + args.inner_offset + uint64_t(hv) * args.head_dim + j3] = out3 * rsqrt(float(args.head_dim));
        }
    }
}
kernel void qw3_deltanet_batch_gated_rmsnorm(constant qw3_batch_gdn_args &args,
                                             device float *scratch,
                                             device const float *w,
                                             threadgroup float *sh,
                                             uint group [[threadgroup_position_in_grid]],
                                             ushort tid [[thread_index_in_threadgroup]],
                                             ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                             ushort lane [[thread_index_in_simdgroup]],
                                             ushort nt [[threads_per_threadgroup]]) {
    uint t = group / args.v_heads;
    uint hv = group - t * args.v_heads;
    if (t >= args.n_tokens || hv >= args.v_heads) return;
    uint64_t base = uint64_t(t) * args.stride;
    device float *core = scratch + base + args.inner_offset + uint64_t(hv) * args.head_dim;
    device const float *z = scratch + base + args.z_offset + uint64_t(hv) * args.head_dim;
    float ss = 0.0f;
    for (uint i = tid; i < args.head_dim; i += nt) ss += core[i] * core[i];
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.head_dim) + args.eps);
    for (uint i = tid; i < args.head_dim; i += nt) {
        float zi = z[i];
        float gate = zi / (1.0f + exp(-zi));
        core[i] = core[i] * scale * w[i] * gate;
    }
}
struct qw3_gated_rmsnorm_args { uint v_heads; uint head_dim; float eps; };
kernel void qw3_deltanet_gated_rmsnorm(constant qw3_gated_rmsnorm_args &args,
                                      device const float *w,
                                      device const float *core,
                                      device const float *z,
                                      device float *out,
                                      threadgroup float *sh,
                                      uint hv [[threadgroup_position_in_grid]],
                                      ushort tid [[thread_index_in_threadgroup]],
                                      ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                      ushort lane [[thread_index_in_simdgroup]],
                                      ushort nt [[threads_per_threadgroup]]) {
    if (hv >= args.v_heads) return;
    device const float *src = core + uint64_t(hv) * args.head_dim;
    device const float *zg = z + uint64_t(hv) * args.head_dim;
    device float *dst = out + uint64_t(hv) * args.head_dim;
    float ss = 0.0f;
    for (uint i = tid; i < args.head_dim; i += nt) ss += src[i] * src[i];
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.head_dim) + args.eps);
    for (uint i = tid; i < args.head_dim; i += nt) {
        float zi = zg[i];
        float gate = zi / (1.0f + exp(-zi));
        dst[i] = src[i] * scale * w[i] * gate;
    }
}
kernel void qw3_residual_rmsnorm_weight_f32(constant qw3_rmsnorm_args &args,
                                           device const float *x,
                                           device const float *residual,
                                           device const float *w,
                                           device float *y,
                                           threadgroup float *sh,
                                           ushort tid [[thread_index_in_threadgroup]],
                                           ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                           ushort lane [[thread_index_in_simdgroup]],
                                           ushort nt [[threads_per_threadgroup]]) {
    float ss = 0.0f;
    for (uint i = tid; i < args.n; i += nt) {
        float v = x[i] + residual[i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.n) + args.eps);
    for (uint i = tid; i < args.n; i += nt) y[i] = (x[i] + residual[i]) * scale * w[i];
}
kernel void qw3_residual_rmsnorm_update_x0(constant qw3_rmsnorm_args &args,
                                           device float *x0,
                                           device const float *residual,
                                           device const float *w,
                                           device float *y,
                                           threadgroup float *sh,
                                           ushort tid [[thread_index_in_threadgroup]],
                                           ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                           ushort lane [[thread_index_in_simdgroup]],
                                           ushort nt [[threads_per_threadgroup]]) {
    float ss = 0.0f;
    for (uint i = tid; i < args.n; i += nt) {
        float v = x0[i] + residual[i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.n) + args.eps);
    for (uint i = tid; i < args.n; i += nt) {
        float v = x0[i] + residual[i];
        x0[i] = v;
        y[i] = v * scale * w[i];
    }
}
struct qw3_residual_batch_args { uint n; float eps; uint n_tokens; uint residual_offset; uint residual_stride; };
kernel void qw3_residual_rmsnorm_batch_update_x0(constant qw3_residual_batch_args &args,
                                                 device float *x0,
                                                 device const float *residual,
                                                 device const float *w,
                                                 device float *y,
                                                 threadgroup float *sh,
                                                 uint row [[threadgroup_position_in_grid]],
                                                 ushort tid [[thread_index_in_threadgroup]],
                                                 ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                                 ushort lane [[thread_index_in_simdgroup]],
                                                 ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_tokens) return;
    device float *xr = x0 + uint64_t(row) * args.n;
    device const float *rr = residual + uint64_t(row) * args.residual_stride + args.residual_offset;
    device float *yr = y + uint64_t(row) * args.n;
    float ss = 0.0f;
    for (uint i = tid; i < args.n; i += nt) { float v = xr[i] + rr[i]; ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.n) + args.eps);
    for (uint i = tid; i < args.n; i += nt) { float v = xr[i] + rr[i]; xr[i] = v; yr[i] = v * scale * w[i]; }
}
struct qw3_unary_args { uint n; float scale; };
kernel void qw3_silu_mul(constant qw3_unary_args &args,
                         device const float *a,
                         device const float *b,
                         device float *out,
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n) return;
    float x = a[gid];
    out[gid] = (x / (1.0f + exp(-x))) * b[gid];
}
kernel void qw3_scale(constant qw3_unary_args &args,
                       device const float *x,
                       device float *out,
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n) return;
    out[gid] = x[gid] * args.scale;
}
struct qw3_offset_args { uint n; uint a_offset; uint b_offset; };
kernel void qw3_add_moe_to_x0(constant qw3_unary_args &args,
                              device float *x0,
                              device const float *x1,
                              device const float *moe,
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n) return;
    x0[gid] = x0[gid] + moe[gid];
}
kernel void qw3_silu_mul_offsets(constant qw3_offset_args &args,
                                  device const float *scratch,
                                  device float *out,
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n) return;
    float x = scratch[args.a_offset + gid];
    float y = scratch[args.b_offset + gid];
    out[gid] = (x / (1.0f + exp(-x))) * y;
}
struct qw3_rows_offset_args { uint n; uint n_rows; uint stride; uint a_offset; uint b_offset; uint out_offset; };
kernel void qw3_silu_mul_rows_offsets(constant qw3_rows_offset_args &args,
                                       device const float *scratch,
                                       device float *out,
                                       uint gid [[thread_position_in_grid]]) {
    uint total = args.n * args.n_rows;
    if (gid >= total) return;
    uint row = gid / args.n;
    uint col = gid - row * args.n;
    uint base = row * args.stride;
    float x = scratch[base + args.a_offset + col];
    float y = scratch[base + args.b_offset + col];
    out[base + args.out_offset + col] = (x / (1.0f + exp(-x))) * y;
}
kernel void qw3_scale_x1_scalar_add_x0(constant qw3_offset_args &args,
                                      device float *x0,
                                      device const float *x1,
                                      device const float *scratch,
                                      uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n) return;
    float raw = scratch[args.a_offset];
    float scale = 1.0f / (1.0f + exp(-raw));
    x0[gid] = x0[gid] + x1[gid] * scale;
}
kernel void qw3_scale_x1_add_x0(constant qw3_unary_args &args,
                                device float *x0,
                                device const float *x1,
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n) return;
    x0[gid] = x0[gid] + x1[gid] * args.scale;
}
kernel void qw3_scale_scratch_add_x0(constant qw3_offset_args &args,
                                     constant float &scale,
                                     device float *x0,
                                     device const float *scratch,
                                     uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n) return;
    x0[gid] = x0[gid] + scratch[args.a_offset + gid] * scale;
}
kernel void qw3_sigmoid_scale_scratch_add_x0_rows(constant qw3_rows_offset_args &args,
                                                 device float *x0,
                                                 device const float *scratch,
                                                 uint gid [[thread_position_in_grid]]) {
    uint total = args.n * args.n_rows;
    if (gid >= total) return;
    uint row = gid / args.n;
    uint col = gid - row * args.n;
    uint base = row * args.stride;
    float raw = scratch[base + args.b_offset];
    float scale = 1.0f / (1.0f + exp(-raw));
    x0[row * args.n + col] = x0[row * args.n + col] + scratch[base + args.a_offset + col] * scale;
}
kernel void qw3_scale_scratch_add_x0_slot(constant qw3_offset_args &args,
                                          constant uint &slot,
                                          device float *x0,
                                          device const float *scratch,
                                          device const float *weights,
                                          uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n) return;
    x0[gid] = x0[gid] + scratch[args.a_offset + gid] * weights[slot];
}
kernel void qw3_router_top8(device const float *router,
                            device int *ids,
                            device float *weights,
                            uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float vals[256];
    threadgroup int best[256];
    threadgroup float selected[8];
    threadgroup int selected_ids[8];
    for (uint rank = 0; rank < 8u; rank++) {
        float v = router[tid];
        for (uint k = 0; k < rank; k++) if (selected_ids[k] == int(tid)) v = -INFINITY;
        vals[tid] = v;
        best[tid] = int(tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                float rv = vals[tid + stride];
                int ri = best[tid + stride];
                if (rv > vals[tid] || (rv == vals[tid] && ri < best[tid])) {
                    vals[tid] = rv;
                    best[tid] = ri;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) { ids[rank] = best[0]; selected_ids[rank] = best[0]; selected[rank] = vals[0]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float sum = 0.0f;
        for (uint k = 0; k < 8u; k++) { weights[k] = exp(selected[k] - selected[0]); sum += weights[k]; }
        for (uint k = 0; k < 8u; k++) weights[k] /= sum;
    }
}
struct qw3_router_batch_args { uint n_tokens; uint router_offset; uint stride; };
kernel void qw3_router_top8_batch(constant qw3_router_batch_args &args,
                                  device const float *scratch,
                                  device int *ids,
                                  device float *weights,
                                  uint token [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
    if (token >= args.n_tokens) return;
    threadgroup float vals[256];
    threadgroup int best[256];
    threadgroup float selected[8];
    threadgroup int selected_ids[8];
    device const float *router = scratch + uint64_t(token) * uint64_t(args.stride) + uint64_t(args.router_offset);
    device int *row_ids = ids + uint64_t(token) * 8ull;
    device float *row_weights = weights + uint64_t(token) * 8ull;
    for (uint rank = 0; rank < 8u; rank++) {
        float v = router[tid];
        for (uint k = 0; k < rank; k++) if (selected_ids[k] == int(tid)) v = -INFINITY;
        vals[tid] = v;
        best[tid] = int(tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                float rv = vals[tid + stride];
                int ri = best[tid + stride];
                if (rv > vals[tid] || (rv == vals[tid] && ri < best[tid])) {
                    vals[tid] = rv;
                    best[tid] = ri;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) { row_ids[rank] = best[0]; selected_ids[rank] = best[0]; selected[rank] = vals[0]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float sum = 0.0f;
        for (uint k = 0; k < 8u; k++) { row_weights[k] = exp(selected[k] - selected[0]); sum += row_weights[k]; }
        for (uint k = 0; k < 8u; k++) row_weights[k] /= sum;
    }
}
struct qw3_expert_slot_args { uint n_in; uint n_out; uint row_bytes; uint expert_bytes; uint slot; };
inline float qw3_iq3s_dot_row(device const uchar *wr,
                              device const float *x,
                              device const ushort *kgrid,
                              uint n_in,
                              ushort tid,
                              ushort nt) {
    float sum = 0.0f;
    uint n_blocks = n_in / 256u;
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
            uchar qh0 = qh[0]; uchar qh1 = qh[1];
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
            qs += 8; signs += 4;
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
            qh += 2; qs += 8; signs += 4;
        }
    }
    return sum;
}
inline float qw3_iq3s_dot32(device const uchar *blk,
                           device const float *xx,
                           device const ushort *kgrid,
                           uint ib) {
    half d = *((device const half *)blk);
    device const uchar *qs = blk + 2;
    device const uchar *qh = qs + 64;
    device const uchar *signs = qh + 8;
    device const uchar *scales = signs + 32;
    float db = float(d) * float(1u + 2u * ((uint(scales[ib / 2u]) >> (4u * (ib & 1u))) & 15u));
    uchar qhb = qh[ib];
    qs += ib * 8u;
    signs += ib * 4u;
    float sum = 0.0f;
    for (uint l = 0; l < 4u; l++) {
        uint idx1 = uint(qs[2u * l + 0u]) | ((uint(qhb) << (8u - 2u * l)) & 256u);
        uint idx2 = uint(qs[2u * l + 1u]) | ((uint(qhb) << (7u - 2u * l)) & 256u);
        uchar s = signs[l];
        for (uint j = 0; j < 4u; j++) {
            float sign1 = (uint(s) & (1u << j)) ? -1.0f : 1.0f;
            float sign2 = (uint(s) & (1u << (j + 4u))) ? -1.0f : 1.0f;
            sum += db * qw3_iq3s_grid_val(kgrid, idx1, j) * sign1 * xx[8u * l + j + 0u];
            sum += db * qw3_iq3s_grid_val(kgrid, idx2, j) * sign2 * xx[8u * l + j + 4u];
        }
    }
    return sum;
}
inline float2 qw3_iq3s_dot32_pair(device const uchar *gate_blk,
                                  device const uchar *up_blk,
                                  device const float *xx,
                                  device const uchar *kgrid,
                                  uint ib) {
    half gate_d = *((device const half *)gate_blk);
    device const uchar *gate_qs = gate_blk + 2;
    device const uchar *gate_qh = gate_qs + 64;
    device const uchar *gate_signs = gate_qh + 8;
    device const uchar *gate_scales = gate_signs + 32;
    float gate_db = float(gate_d) * float(1u + 2u * ((uint(gate_scales[ib / 2u]) >> (4u * (ib & 1u))) & 15u));
    uchar gate_qhb = gate_qh[ib];
    gate_qs += ib * 8u;
    gate_signs += ib * 4u;
    half up_d = *((device const half *)up_blk);
    device const uchar *up_qs = up_blk + 2;
    device const uchar *up_qh = up_qs + 64;
    device const uchar *up_signs = up_qh + 8;
    device const uchar *up_scales = up_signs + 32;
    float up_db = float(up_d) * float(1u + 2u * ((uint(up_scales[ib / 2u]) >> (4u * (ib & 1u))) & 15u));
    uchar up_qhb = up_qh[ib];
    up_qs += ib * 8u;
    up_signs += ib * 4u;
    float2 sum = float2(0.0f);
    for (uint l = 0; l < 4u; l++) {
        uint gate_idx1 = uint(gate_qs[2u * l + 0u]) | ((uint(gate_qhb) << (8u - 2u * l)) & 256u);
        uint gate_idx2 = uint(gate_qs[2u * l + 1u]) | ((uint(gate_qhb) << (7u - 2u * l)) & 256u);
        uint up_idx1 = uint(up_qs[2u * l + 0u]) | ((uint(up_qhb) << (8u - 2u * l)) & 256u);
        uint up_idx2 = uint(up_qs[2u * l + 1u]) | ((uint(up_qhb) << (7u - 2u * l)) & 256u);
        uchar gate_s = gate_signs[l];
        uchar up_s = up_signs[l];
        for (uint j = 0; j < 4u; j++) {
            float x1 = xx[8u * l + j + 0u];
            float x2 = xx[8u * l + j + 4u];
            float gate_sign1 = (uint(gate_s) & (1u << j)) ? -1.0f : 1.0f;
            float gate_sign2 = (uint(gate_s) & (1u << (j + 4u))) ? -1.0f : 1.0f;
            float up_sign1 = (uint(up_s) & (1u << j)) ? -1.0f : 1.0f;
            float up_sign2 = (uint(up_s) & (1u << (j + 4u))) ? -1.0f : 1.0f;
            sum.x += gate_db * float(kgrid[gate_idx1 * 4u + j]) * gate_sign1 * x1;
            sum.x += gate_db * float(kgrid[gate_idx2 * 4u + j]) * gate_sign2 * x2;
            sum.y += up_db * float(kgrid[up_idx1 * 4u + j]) * up_sign1 * x1;
            sum.y += up_db * float(kgrid[up_idx2 * 4u + j]) * up_sign2 * x2;
        }
    }
    return sum;
}
kernel void qw3_matvec_iq3_s_pair(constant qw3_matvec_q8_0_args &args,
                                  device const uchar *gate_weights,
                                  device const uchar *up_weights,
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
    device const uchar *gate_wr = gate_weights + uint64_t(row) * uint64_t(args.row_bytes);
    device const uchar *up_wr = up_weights + uint64_t(row) * uint64_t(args.row_bytes);
    float gate_sum = qw3_iq3s_dot_row(gate_wr, x, kgrid, args.n_in, tid, nt);
    float up_sum = qw3_iq3s_dot_row(up_wr, x, kgrid, args.n_in, tid, nt);
    gate_sum = simd_sum(gate_sum);
    up_sum = simd_sum(up_sum);
    if (lane == 0) { sh[simd_idx] = gate_sum; sh[32u + simd_idx] = up_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = (uint(nt) + 31u) / 32u;
    gate_sum = lane < nsg ? sh[lane] : 0.0f;
    up_sum = lane < nsg ? sh[32u + lane] : 0.0f;
    gate_sum = simd_sum(gate_sum);
    up_sum = simd_sum(up_sum);
    if (tid == 0) { out[row] = gate_sum; out[args.n_out + row] = up_sum; }
}
kernel void qw3_matvec_iq3_s_pair_fast(constant qw3_matvec_q8_0_args &args,
                                       device const uchar *gate_weights,
                                       device const uchar *up_weights,
                                       device const float *x,
                                       device float *out,
                                       device const ushort *kgrid,
                                       threadgroup float *sh,
                                       uint group [[threadgroup_position_in_grid]],
                                       ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                       ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    const uint nr0 = 4u;
    const uint nsg = 2u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    float gate_sum0 = 0.0f, gate_sum1 = 0.0f, gate_sum2 = 0.0f, gate_sum3 = 0.0f;
    float up_sum0 = 0.0f, up_sum1 = 0.0f, up_sum2 = 0.0f, up_sum3 = 0.0f;
    uint nb32 = (args.n_in / 256u) * 8u;
    for (uint ib32 = uint(lane); ib32 < nb32; ib32 += 32u) {
        uint ibl = ib32 / 8u;
        uint ib = ib32 - ibl * 8u;
        device const float *xx = x + uint64_t(ib32) * 32ull;
        uint row = first_row;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;
            gate_sum0 += qw3_iq3s_dot32(gate_weights + off, xx, kgrid, ib);
            up_sum0 += qw3_iq3s_dot32(up_weights + off, xx, kgrid, ib);
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;
            gate_sum1 += qw3_iq3s_dot32(gate_weights + off, xx, kgrid, ib);
            up_sum1 += qw3_iq3s_dot32(up_weights + off, xx, kgrid, ib);
        }
        row = first_row + 2u;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;
            gate_sum2 += qw3_iq3s_dot32(gate_weights + off, xx, kgrid, ib);
            up_sum2 += qw3_iq3s_dot32(up_weights + off, xx, kgrid, ib);
        }
        row = first_row + 3u;
        if (row < args.n_out) {
            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;
            gate_sum3 += qw3_iq3s_dot32(gate_weights + off, xx, kgrid, ib);
            up_sum3 += qw3_iq3s_dot32(up_weights + off, xx, kgrid, ib);
        }
    }
    gate_sum0 = simd_sum(gate_sum0); up_sum0 = simd_sum(up_sum0);
    gate_sum1 = simd_sum(gate_sum1); up_sum1 = simd_sum(up_sum1);
    gate_sum2 = simd_sum(gate_sum2); up_sum2 = simd_sum(up_sum2);
    gate_sum3 = simd_sum(gate_sum3); up_sum3 = simd_sum(up_sum3);
    if (lane == 0) {
        uint row = first_row;
        if (row < args.n_out) { out[row] = gate_sum0; out[args.n_out + row] = up_sum0; }
        row = first_row + 1u;
        if (row < args.n_out) { out[row] = gate_sum1; out[args.n_out + row] = up_sum1; }
        row = first_row + 2u;
        if (row < args.n_out) { out[row] = gate_sum2; out[args.n_out + row] = up_sum2; }
        row = first_row + 3u;
        if (row < args.n_out) { out[row] = gate_sum3; out[args.n_out + row] = up_sum3; }
    }
}
kernel void qw3_matvec_iq3_s_expert_slot_pair(constant qw3_expert_slot_args &args,
                                              device const uchar *gate_weights,
                                              device const uchar *up_weights,
                                              device const float *x,
                                              device float *out,
                                              device const ushort *kgrid,
                                              device const int *ids,
                                              threadgroup float *sh,
                                              uint row [[threadgroup_position_in_grid]],
                                              ushort tid [[thread_index_in_threadgroup]],
                                              ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                              ushort lane [[thread_index_in_simdgroup]],
                                              ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    uint expert = uint(ids[args.slot]);
    uint64_t off = uint64_t(expert) * uint64_t(args.expert_bytes) + uint64_t(row) * uint64_t(args.row_bytes);
    float gate_sum = qw3_iq3s_dot_row(gate_weights + off, x, kgrid, args.n_in, tid, nt);
    float up_sum = qw3_iq3s_dot_row(up_weights + off, x, kgrid, args.n_in, tid, nt);
    gate_sum = simd_sum(gate_sum);
    up_sum = simd_sum(up_sum);
    if (lane == 0) { sh[simd_idx] = gate_sum; sh[32u + simd_idx] = up_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    gate_sum = lane < 32 ? sh[lane] : 0.0f;
    up_sum = lane < 32 ? sh[32u + lane] : 0.0f;
    gate_sum = simd_sum(gate_sum);
    up_sum = simd_sum(up_sum);
    if (tid == 0) { out[row] = gate_sum; out[args.n_out + row] = up_sum; }
}
kernel void qw3_matvec_iq3_s_expert_slot_pair_fast(constant qw3_expert_slot_args &args,
                                                   device const uchar *gate_weights,
                                                   device const uchar *up_weights,
                                                   device const float *x,
                                                   device float *out,
                                                   device const ushort *kgrid,
                                                   device const int *ids,
                                                   threadgroup float *sh,
                                                   uint group [[threadgroup_position_in_grid]],
                                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                                   ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    const uint nr0 = 4u;
    const uint nsg = 2u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    uint expert = uint(ids[args.slot]);
    uint64_t expert_off = uint64_t(expert) * uint64_t(args.expert_bytes);
    float gate_sum0 = 0.0f, gate_sum1 = 0.0f, gate_sum2 = 0.0f, gate_sum3 = 0.0f;
    float up_sum0 = 0.0f, up_sum1 = 0.0f, up_sum2 = 0.0f, up_sum3 = 0.0f;
    uint nb32 = (args.n_in / 256u) * 8u;
    for (uint ib32 = uint(lane); ib32 < nb32; ib32 += 32u) {
        uint ibl = ib32 / 8u;
        uint ib = ib32 - ibl * 8u;
        device const float *xx = x + uint64_t(ib32) * 32ull;
        uint row = first_row;
        if (row < args.n_out) {
            uint64_t off = expert_off + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;
            gate_sum0 += qw3_iq3s_dot32(gate_weights + off, xx, kgrid, ib);
            up_sum0 += qw3_iq3s_dot32(up_weights + off, xx, kgrid, ib);
        }
        row = first_row + 1u;
        if (row < args.n_out) {
            uint64_t off = expert_off + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;
            gate_sum1 += qw3_iq3s_dot32(gate_weights + off, xx, kgrid, ib);
            up_sum1 += qw3_iq3s_dot32(up_weights + off, xx, kgrid, ib);
        }
        row = first_row + 2u;
        if (row < args.n_out) {
            uint64_t off = expert_off + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;
            gate_sum2 += qw3_iq3s_dot32(gate_weights + off, xx, kgrid, ib);
            up_sum2 += qw3_iq3s_dot32(up_weights + off, xx, kgrid, ib);
        }
        row = first_row + 3u;
        if (row < args.n_out) {
            uint64_t off = expert_off + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;
            gate_sum3 += qw3_iq3s_dot32(gate_weights + off, xx, kgrid, ib);
            up_sum3 += qw3_iq3s_dot32(up_weights + off, xx, kgrid, ib);
        }
    }
    gate_sum0 = simd_sum(gate_sum0); up_sum0 = simd_sum(up_sum0);
    gate_sum1 = simd_sum(gate_sum1); up_sum1 = simd_sum(up_sum1);
    gate_sum2 = simd_sum(gate_sum2); up_sum2 = simd_sum(up_sum2);
    gate_sum3 = simd_sum(gate_sum3); up_sum3 = simd_sum(up_sum3);
    if (lane == 0) {
        uint row = first_row;
        if (row < args.n_out) { out[row] = gate_sum0; out[args.n_out + row] = up_sum0; }
        row = first_row + 1u;
        if (row < args.n_out) { out[row] = gate_sum1; out[args.n_out + row] = up_sum1; }
        row = first_row + 2u;
        if (row < args.n_out) { out[row] = gate_sum2; out[args.n_out + row] = up_sum2; }
        row = first_row + 3u;
        if (row < args.n_out) { out[row] = gate_sum3; out[args.n_out + row] = up_sum3; }
    }
}
struct qw3_moe_batch_args { uint n_in; uint n_ff; uint n_embd; uint n_active; uint iq3_row_bytes; uint iq3_expert_bytes; uint down_row_bytes; uint down_expert_bytes; uint gateup_base; uint hidden_base; uint down_base; };
kernel void qw3_moe_iq3_s_pair_batch(constant qw3_moe_batch_args &args,
                                      device const uchar *gate_weights,
                                      device const uchar *up_weights,
                                      device const float *x,
                                      device float *scratch,
                                      device const ushort *kgrid,
                                      constant int *ids,
                                      threadgroup float *sh,
                                      uint group [[threadgroup_position_in_grid]],
                                      ushort tid [[thread_index_in_threadgroup]],
                                      ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                      ushort lane [[thread_index_in_simdgroup]],
                                      ushort nt [[threads_per_threadgroup]]) {
    uint slot = group / args.n_ff;
    uint row = group - slot * args.n_ff;
    if (row >= args.n_ff || slot >= args.n_active) return;
    uint expert = uint(ids[slot]);
    uint64_t off = uint64_t(expert) * uint64_t(args.iq3_expert_bytes) + uint64_t(row) * uint64_t(args.iq3_row_bytes);
    float gate_sum = qw3_iq3s_dot_row(gate_weights + off, x, kgrid, args.n_in, tid, nt);
    float up_sum = qw3_iq3s_dot_row(up_weights + off, x, kgrid, args.n_in, tid, nt);
    gate_sum = simd_sum(gate_sum);
    up_sum = simd_sum(up_sum);
    if (lane == 0) { sh[simd_idx] = gate_sum; sh[32u + simd_idx] = up_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    gate_sum = lane < 32 ? sh[lane] : 0.0f;
    up_sum = lane < 32 ? sh[32u + lane] : 0.0f;
    gate_sum = simd_sum(gate_sum);
    up_sum = simd_sum(up_sum);
    if (tid == 0) {
        uint base = args.gateup_base + slot * (2u * args.n_ff);
        scratch[base + row] = gate_sum;
        scratch[base + args.n_ff + row] = up_sum;
    }
}
kernel void qw3_moe_down_iq4_xs_batch(constant qw3_moe_batch_args &args,
                                      device const uchar *weights,
                                      device const float *scratch,
                                      device float *out,
                                      constant int *ids,
                                      constant float *router_weights,
                                      threadgroup float *sh,
                                      uint group [[threadgroup_position_in_grid]],
                                      ushort tid [[thread_index_in_threadgroup]],
                                      ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                      ushort lane [[thread_index_in_simdgroup]],
                                      ushort nt [[threads_per_threadgroup]]) {
    uint slot = group / args.n_embd;
    uint row = group - slot * args.n_embd;
    if (row >= args.n_embd || slot >= args.n_active) return;
    uint expert = uint(ids[slot]);
    device const uchar *wr = weights + uint64_t(expert) * uint64_t(args.down_expert_bytes) + uint64_t(row) * uint64_t(args.down_row_bytes);
    device const float *x = scratch + args.hidden_base + slot * args.n_ff;
    float sum = 0.0f;
    uint n_blocks = args.n_ff / 256u;
    for (uint b = tid; b < n_blocks; b += nt) {
        device const uchar *blk = wr + uint64_t(b) * 136ull;
        half d = *((device const half *)blk);
        ushort scales_h = *((device const ushort *)(blk + 2));
        device const uchar *scales_l = blk + 4;
        device const uchar *qs = scales_l + 4;
        device const float *xx = x + uint64_t(b) * 256ull;
        for (uint ib = 0; ib < 8u; ib++) {
            uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);
            float dl = float(d) * float(int(ls) - 32);
            device const uchar *q = qs + ib * 16u;
            device const float *xg = xx + ib * 32u;
            for (uint j = 0; j < 16u; j++) { uchar v = q[j]; sum += dl * qw3_iq4nl_val(uint(v) & 15u) * xg[j]; sum += dl * qw3_iq4nl_val(uint(v) >> 4u) * xg[j + 16u]; }
        }
    }
    sum = simd_sum(sum); if (lane == 0) sh[simd_idx] = sum; threadgroup_barrier(mem_flags::mem_threadgroup); uint nsg = (uint(nt) + 31u) / 32u; sum = lane < nsg ? sh[lane] : 0.0f; sum = simd_sum(sum);
    if (tid == 0) out[args.down_base + slot * args.n_embd + row] = sum * router_weights[slot];
}
kernel void qw3_moe_iq3_s_swiglu_batch_fast(constant qw3_moe_batch_args &args,
                                           device const uchar *gate_weights,
                                           device const uchar *up_weights,
                                           device const float *x,
                                           device float *scratch,
                                           device const uchar *kgrid,
                                           constant int *ids,
                                           threadgroup float *sh,
                                           uint group [[threadgroup_position_in_grid]],
                                           ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                           ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    const uint nr0 = 4u;
    const uint nsg = 2u;
    uint groups_per_slot = (args.n_ff + 7u) / 8u;
    uint slot = group / groups_per_slot;
    uint row_group = group - slot * groups_per_slot;
    if (slot >= args.n_active) return;
    uint first_row = (row_group * nsg + uint(simd_idx)) * nr0;
    uint expert = uint(ids[slot]);
    uint64_t expert_off = uint64_t(expert) * uint64_t(args.iq3_expert_bytes);
    float gate_sum0 = 0.0f, gate_sum1 = 0.0f, gate_sum2 = 0.0f, gate_sum3 = 0.0f;
    float up_sum0 = 0.0f, up_sum1 = 0.0f, up_sum2 = 0.0f, up_sum3 = 0.0f;
    uint nb32 = (args.n_in / 256u) * 8u;
    for (uint ib32 = uint(lane); ib32 < nb32; ib32 += 32u) {
        uint ibl = ib32 / 8u;
        uint ib = ib32 - ibl * 8u;
        device const float *xx = x + uint64_t(ib32) * 32ull;
        uint row = first_row;
        if (row < args.n_ff) {
            uint64_t off = expert_off + uint64_t(row) * uint64_t(args.iq3_row_bytes) + uint64_t(ibl) * 110ull;
            float2 pair_sum = qw3_iq3s_dot32_pair(gate_weights + off, up_weights + off, xx, kgrid, ib);
            gate_sum0 += pair_sum.x; up_sum0 += pair_sum.y;
        }
        row = first_row + 1u;
        if (row < args.n_ff) {
            uint64_t off = expert_off + uint64_t(row) * uint64_t(args.iq3_row_bytes) + uint64_t(ibl) * 110ull;
            float2 pair_sum = qw3_iq3s_dot32_pair(gate_weights + off, up_weights + off, xx, kgrid, ib);
            gate_sum1 += pair_sum.x; up_sum1 += pair_sum.y;
        }
        row = first_row + 2u;
        if (row < args.n_ff) {
            uint64_t off = expert_off + uint64_t(row) * uint64_t(args.iq3_row_bytes) + uint64_t(ibl) * 110ull;
            float2 pair_sum = qw3_iq3s_dot32_pair(gate_weights + off, up_weights + off, xx, kgrid, ib);
            gate_sum2 += pair_sum.x; up_sum2 += pair_sum.y;
        }
        row = first_row + 3u;
        if (row < args.n_ff) {
            uint64_t off = expert_off + uint64_t(row) * uint64_t(args.iq3_row_bytes) + uint64_t(ibl) * 110ull;
            float2 pair_sum = qw3_iq3s_dot32_pair(gate_weights + off, up_weights + off, xx, kgrid, ib);
            gate_sum3 += pair_sum.x; up_sum3 += pair_sum.y;
        }
    }
    gate_sum0 = simd_sum(gate_sum0); up_sum0 = simd_sum(up_sum0);
    gate_sum1 = simd_sum(gate_sum1); up_sum1 = simd_sum(up_sum1);
    gate_sum2 = simd_sum(gate_sum2); up_sum2 = simd_sum(up_sum2);
    gate_sum3 = simd_sum(gate_sum3); up_sum3 = simd_sum(up_sum3);
    if (lane == 0) {
        uint base = args.hidden_base + slot * args.n_ff;
        uint row = first_row;
        if (row < args.n_ff) scratch[base + row] = (gate_sum0 / (1.0f + exp(-gate_sum0))) * up_sum0;
        row = first_row + 1u;
        if (row < args.n_ff) scratch[base + row] = (gate_sum1 / (1.0f + exp(-gate_sum1))) * up_sum1;
        row = first_row + 2u;
        if (row < args.n_ff) scratch[base + row] = (gate_sum2 / (1.0f + exp(-gate_sum2))) * up_sum2;
        row = first_row + 3u;
        if (row < args.n_ff) scratch[base + row] = (gate_sum3 / (1.0f + exp(-gate_sum3))) * up_sum3;
    }
}
kernel void qw3_moe_down_iq4_xs_batch_fast(constant qw3_moe_batch_args &args,
                                          device const uchar *weights,
                                          device const float *scratch,
                                          device float *out,
                                          constant int *ids,
                                          constant float *router_weights,
                                          threadgroup float *sh,
                                          uint group [[threadgroup_position_in_grid]],
                                          ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                          ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    const uint nr0 = 2u;
    const uint nsg = 2u;
    uint groups_per_slot = (args.n_embd + 3u) / 4u;
    uint slot = group / groups_per_slot;
    uint row_group = group - slot * groups_per_slot;
    if (slot >= args.n_active) return;
    uint first_row = (row_group * nsg + uint(simd_idx)) * nr0;
    uint expert = uint(ids[slot]);
    uint64_t expert_off = uint64_t(expert) * uint64_t(args.down_expert_bytes);
    device const float *x = scratch + args.hidden_base + slot * args.n_ff;
    uint ix = uint(lane) >> 4u;
    uint it = uint(lane) & 15u;
    uint ib = it >> 1u;
    uint il = it & 1u;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    uint n_blocks = args.n_ff / 256u;
    for (uint b = ix; b < n_blocks; b += 2u) {
        device const float *xg = x + uint64_t(b) * 256ull + uint64_t(ib) * 32ull + uint64_t(il) * 8ull;
        uint row = first_row;
        if (row < args.n_embd) {
            device const uchar *blk = weights + expert_off + uint64_t(row) * uint64_t(args.down_row_bytes) + uint64_t(b) * 136ull;
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
        if (row < args.n_embd) {
            device const uchar *blk = weights + expert_off + uint64_t(row) * uint64_t(args.down_row_bytes) + uint64_t(b) * 136ull;
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
    float scale = router_weights[slot];
    if (lane == 0) {
        uint base = args.down_base + slot * args.n_embd;
        uint row = first_row;
        if (row < args.n_embd) out[base + row] = sum0 * scale;
        row = first_row + 1u;
        if (row < args.n_embd) out[base + row] = sum1 * scale;
    }
}
kernel void qw3_moe_down_iq4_xs_pair_fast(constant qw3_moe_batch_args &args,
                                         device const uchar *weights,
                                         device const float *scratch,
                                         device float *out,
                                         constant int *ids,
                                         constant float *router_weights,
                                         threadgroup float *sh,
                                         uint group [[threadgroup_position_in_grid]],
                                         ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                         ushort lane [[thread_index_in_simdgroup]]) {
    if (lane < 16u) sh[lane] = qw3_iq4nl_val(uint(lane));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nr0 = 2u;
    const uint nsg = 2u;
    uint groups_per_pair = (args.n_embd + 3u) / 4u;
    uint pair = group / groups_per_pair;
    uint row_group = group - pair * groups_per_pair;
    uint slot0 = pair * 2u;
    uint slot1 = slot0 + 1u;
    if (slot0 >= args.n_active) return;
    uint first_row = (row_group * nsg + uint(simd_idx)) * nr0;
    uint expert0 = uint(ids[slot0]);
    uint expert1 = slot1 < args.n_active ? uint(ids[slot1]) : expert0;
    uint64_t expert_off0 = uint64_t(expert0) * uint64_t(args.down_expert_bytes);
    uint64_t expert_off1 = uint64_t(expert1) * uint64_t(args.down_expert_bytes);
    device const float *x0 = scratch + args.hidden_base + slot0 * args.n_ff;
    device const float *x1 = scratch + args.hidden_base + (slot1 < args.n_active ? slot1 : slot0) * args.n_ff;
    uint ix = uint(lane) >> 4u;
    uint it = uint(lane) & 15u;
    uint ib = it >> 1u;
    uint il = it & 1u;
    float sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;
    uint n_blocks = args.n_ff / 256u;
    for (uint b = ix; b < n_blocks; b += 2u) {
        device const float *xg0 = x0 + uint64_t(b) * 256ull + uint64_t(ib) * 32ull + uint64_t(il) * 8ull;
        device const float *xg1 = x1 + uint64_t(b) * 256ull + uint64_t(ib) * 32ull + uint64_t(il) * 8ull;
        uint row = first_row;
        if (row < args.n_embd) {
            device const uchar *ba = weights + expert_off0 + uint64_t(row) * uint64_t(args.down_row_bytes) + uint64_t(b) * 136ull;
            device const uchar *bb = weights + expert_off1 + uint64_t(row) * uint64_t(args.down_row_bytes) + uint64_t(b) * 136ull;
            half da = *((device const half *)ba); half db = *((device const half *)bb);
            ushort sha = *((device const ushort *)(ba + 2)); ushort shb = *((device const ushort *)(bb + 2));
            device const uchar *sla = ba + 4; device const uchar *slb = bb + 4;
            device const uchar *qsa = sla + 4 + ib * 16u + il * 8u; device const uchar *qsb = slb + 4 + ib * 16u + il * 8u;
            uint lsa = ((uint(sla[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(sha) >> (2u * ib)) & 3u) << 4u);
            uint lsb = ((uint(slb[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(shb) >> (2u * ib)) & 3u) << 4u);
            float aca = 0.0f, acb = 0.0f;
            for (uint j = 0; j < 8u; j++) { uchar va = qsa[j]; uchar vb = qsb[j]; aca += sh[uint(va) & 15u] * xg0[j] + sh[uint(va) >> 4u] * xg0[j + 16u]; acb += sh[uint(vb) & 15u] * xg1[j] + sh[uint(vb) >> 4u] * xg1[j + 16u]; }
            sum00 += float(da) * float(int(lsa) - 32) * aca; sum10 += float(db) * float(int(lsb) - 32) * acb;
        }
        row = first_row + 1u;
        if (row < args.n_embd) {
            device const uchar *ba = weights + expert_off0 + uint64_t(row) * uint64_t(args.down_row_bytes) + uint64_t(b) * 136ull;
            device const uchar *bb = weights + expert_off1 + uint64_t(row) * uint64_t(args.down_row_bytes) + uint64_t(b) * 136ull;
            half da = *((device const half *)ba); half db = *((device const half *)bb);
            ushort sha = *((device const ushort *)(ba + 2)); ushort shb = *((device const ushort *)(bb + 2));
            device const uchar *sla = ba + 4; device const uchar *slb = bb + 4;
            device const uchar *qsa = sla + 4 + ib * 16u + il * 8u; device const uchar *qsb = slb + 4 + ib * 16u + il * 8u;
            uint lsa = ((uint(sla[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(sha) >> (2u * ib)) & 3u) << 4u);
            uint lsb = ((uint(slb[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(shb) >> (2u * ib)) & 3u) << 4u);
            float aca = 0.0f, acb = 0.0f;
            for (uint j = 0; j < 8u; j++) { uchar va = qsa[j]; uchar vb = qsb[j]; aca += sh[uint(va) & 15u] * xg0[j] + sh[uint(va) >> 4u] * xg0[j + 16u]; acb += sh[uint(vb) & 15u] * xg1[j] + sh[uint(vb) >> 4u] * xg1[j + 16u]; }
            sum01 += float(da) * float(int(lsa) - 32) * aca; sum11 += float(db) * float(int(lsb) - 32) * acb;
        }
    }
    sum00 = simd_sum(sum00); sum01 = simd_sum(sum01);
    sum10 = simd_sum(sum10); sum11 = simd_sum(sum11);
    if (lane == 0) {
        float scale0 = router_weights[slot0];
        float scale1 = slot1 < args.n_active ? router_weights[slot1] : 0.0f;
        uint base = args.down_base + pair * args.n_embd;
        uint row = first_row;
        if (row < args.n_embd) out[base + row] = sum00 * scale0 + sum10 * scale1;
        row = first_row + 1u;
        if (row < args.n_embd) out[base + row] = sum01 * scale0 + sum11 * scale1;
    }
}
kernel void qw3_moe_down_iq4_xs_batch_reduce_fast(constant qw3_moe_batch_args &args,
                                                 device const uchar *weights,
                                                 device const float *scratch,
                                                 device float *x0,
                                                 constant int *ids,
                                                 constant float *router_weights,
                                                 threadgroup float *sh,
                                                 uint group [[threadgroup_position_in_grid]],
                                                 ushort tid [[thread_index_in_threadgroup]],
                                                 ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                                 ushort lane [[thread_index_in_simdgroup]]) {
    uint slot = uint(simd_idx);
    uint row0 = group * 2u;
    uint row1 = row0 + 1u;
    bool active = slot < args.n_active;
    uint expert = active ? uint(ids[slot]) : 0u;
    uint64_t expert_off = uint64_t(expert) * uint64_t(args.down_expert_bytes);
    device const float *x = scratch + args.hidden_base + slot * args.n_ff;
    uint ix = uint(lane) >> 4u;
    uint it = uint(lane) & 15u;
    uint ib = it >> 1u;
    uint il = it & 1u;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    uint n_blocks = args.n_ff / 256u;
    if (active) {
        for (uint b = ix; b < n_blocks; b += 2u) {
            device const float *xg = x + uint64_t(b) * 256ull + uint64_t(ib) * 32ull + uint64_t(il) * 8ull;
            if (row0 < args.n_embd) {
                device const uchar *blk = weights + expert_off + uint64_t(row0) * uint64_t(args.down_row_bytes) + uint64_t(b) * 136ull;
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
            if (row1 < args.n_embd) {
                device const uchar *blk = weights + expert_off + uint64_t(row1) * uint64_t(args.down_row_bytes) + uint64_t(b) * 136ull;
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
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    float scale = active ? router_weights[slot] : 0.0f;
    if (lane == 0) { sh[slot] = sum0 * scale; sh[8u + slot] = sum1 * scale; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total0 = 0.0f;
        float total1 = 0.0f;
        for (uint s = 0; s < args.n_active; s++) { total0 += sh[s]; total1 += sh[8u + s]; }
        if (row0 < args.n_embd) x0[row0] += total0;
        if (row1 < args.n_embd) x0[row1] += total1;
    }
}
kernel void qw3_moe_down_q6_k_batch(constant qw3_moe_batch_args &args,
                                    device const uchar *weights,
                                    device const float *scratch,
                                    device float *out,
                                    constant int *ids,
                                    constant float *router_weights,
                                    threadgroup float *sh,
                                    uint group [[threadgroup_position_in_grid]],
                                    ushort tid [[thread_index_in_threadgroup]],
                                    ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                    ushort lane [[thread_index_in_simdgroup]],
                                    ushort nt [[threads_per_threadgroup]]) {
    uint slot = group / args.n_embd;
    uint row = group - slot * args.n_embd;
    if (row >= args.n_embd || slot >= args.n_active) return;
    uint expert = uint(ids[slot]);
    device const uchar *wr = weights + uint64_t(expert) * uint64_t(args.down_expert_bytes) + uint64_t(row) * uint64_t(args.down_row_bytes);
    device const float *x = scratch + args.hidden_base + slot * args.n_ff;
    float sum = 0.0f;
    uint n_blocks = args.n_ff / 256u;
    for (uint b = tid; b < n_blocks; b += nt) {
        device const uchar *blk = wr + uint64_t(b) * 210ull;
        device const uchar *ql = blk; device const uchar *qh = ql + 128u; device const uchar *scb = qh + 64u; device const char *sc = (device const char *)scb;
        ushort dbits = ushort(scb[16u]) | (ushort(scb[17u]) << 8u); float d = qw3_f16_to_f32(dbits); device const float *xx = x + uint64_t(b) * 256ull;
        for (uint n = 0; n < 256u; n += 128u) {
            for (uint l = 0; l < 32u; l++) {
                uint is = l / 16u;
                int q1 = int((uint(ql[l +  0u]) & 15u) | (((uint(qh[l]) >> 0u) & 3u) << 4u)) - 32;
                int q2 = int((uint(ql[l + 32u]) & 15u) | (((uint(qh[l]) >> 2u) & 3u) << 4u)) - 32;
                int q3 = int((uint(ql[l +  0u]) >> 4u) | (((uint(qh[l]) >> 4u) & 3u) << 4u)) - 32;
                int q4 = int((uint(ql[l + 32u]) >> 4u) | (((uint(qh[l]) >> 6u) & 3u) << 4u)) - 32;
                sum += d * float(sc[is + 0u]) * float(q1) * xx[n + l +  0u]; sum += d * float(sc[is + 2u]) * float(q2) * xx[n + l + 32u]; sum += d * float(sc[is + 4u]) * float(q3) * xx[n + l + 64u]; sum += d * float(sc[is + 6u]) * float(q4) * xx[n + l + 96u];
            }
            ql += 64u; qh += 32u; sc += 8u;
        }
    }
    sum = simd_sum(sum); if (lane == 0) sh[simd_idx] = sum; threadgroup_barrier(mem_flags::mem_threadgroup); uint nsg = (uint(nt) + 31u) / 32u; sum = lane < nsg ? sh[lane] : 0.0f; sum = simd_sum(sum);
    if (tid == 0) out[args.down_base + slot * args.n_embd + row] = sum * router_weights[slot];
}
kernel void qw3_moe_down_q6_k_batch_fast(constant qw3_moe_batch_args &args,
                                        device const uchar *weights,
                                        device const float *scratch,
                                        device float *out,
                                        constant int *ids,
                                        constant float *router_weights,
                                        threadgroup float *sh,
                                        uint group [[threadgroup_position_in_grid]],
                                        ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                        ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    const uint nr0 = 2u;
    const uint nsg = 2u;
    uint groups_per_slot = (args.n_embd + 3u) / 4u;
    uint slot = group / groups_per_slot;
    uint row_group = group - slot * groups_per_slot;
    if (slot >= args.n_active) return;
    uint first_row = (row_group * nsg + uint(simd_idx)) * nr0;
    uint expert = uint(ids[slot]);
    uint64_t expert_off = uint64_t(expert) * uint64_t(args.down_expert_bytes);
    device const float *x = scratch + args.hidden_base + slot * args.n_ff;
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
    uint n_blocks = args.n_ff / 256u;
    for (uint b = ix; b < n_blocks; b += 2u) {
        device const float *yy = x + uint64_t(b) * 256ull + y_offset;
        uint row = first_row;
        if (row < args.n_embd) {
            device const uchar *blk = weights + expert_off + uint64_t(row) * uint64_t(args.down_row_bytes) + uint64_t(b) * 210ull;
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
        if (row < args.n_embd) {
            device const uchar *blk = weights + expert_off + uint64_t(row) * uint64_t(args.down_row_bytes) + uint64_t(b) * 210ull;
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
    float scale = router_weights[slot];
    if (lane == 0) {
        uint base = args.down_base + slot * args.n_embd;
        uint row = first_row;
        if (row < args.n_embd) out[base + row] = sum0 * scale;
        row = first_row + 1u;
        if (row < args.n_embd) out[base + row] = sum1 * scale;
    }
}
kernel void qw3_moe_reduce_batch(constant qw3_moe_batch_args &args,
                                  device const float *scratch,
                                  device float *x0,
                                  uint gid [[thread_position_in_grid]]) {
    uint n4 = args.n_embd / 4u;
    if (gid >= n4) return;
    float4 sum = float4(0.0f);
    for (uint slot = 0; slot < args.n_active; slot++) {
        device const float4 *src4 = (device const float4 *)(scratch + args.down_base + slot * args.n_embd);
        sum += src4[gid];
    }
    device float4 *x04 = (device float4 *)x0;
    x04[gid] += sum;
}
struct qw3_moe_prefill_batch_args { uint n_in; uint n_ff; uint n_embd; uint n_tokens; uint n_active; uint iq3_row_bytes; uint iq3_expert_bytes; uint down_row_bytes; uint down_expert_bytes; uint stride; uint hidden_offset; uint compact_blocks; uint mid_preweighted; };
kernel void qw3_moe_iq3_s_swiglu_prefill_batch_fast(constant qw3_moe_prefill_batch_args &args,
                                                   device const uchar *gate_weights,
                                                   device const uchar *up_weights,
                                                   device const float *x1,
                                                   device float *scratch,
                                                   device const uchar *kgrid,
                                                   constant int *ids,
                                                   threadgroup float *sh,
                                                   uint group [[threadgroup_position_in_grid]],
                                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                                   ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    const uint nr0 = 4u;
    const uint nsg = 2u;
    uint groups_per_pair = (args.n_ff + 7u) / 8u;
    uint pair = group / groups_per_pair;
    uint row_group = group - pair * groups_per_pair;
    uint token = pair / args.n_active;
    uint slot = pair - token * args.n_active;
    if (token >= args.n_tokens || slot >= args.n_active) return;
    uint first_row = (row_group * nsg + uint(simd_idx)) * nr0;
    uint expert = uint(ids[pair]);
    uint64_t expert_off = uint64_t(expert) * uint64_t(args.iq3_expert_bytes);
    device const float *x = x1 + uint64_t(token) * uint64_t(args.n_in);
    float gate_sum0 = 0.0f, gate_sum1 = 0.0f, gate_sum2 = 0.0f, gate_sum3 = 0.0f;
    float up_sum0 = 0.0f, up_sum1 = 0.0f, up_sum2 = 0.0f, up_sum3 = 0.0f;
    uint nb32 = (args.n_in / 256u) * 8u;
    for (uint ib32 = uint(lane); ib32 < nb32; ib32 += 32u) {
        uint ibl = ib32 / 8u;
        uint ib = ib32 - ibl * 8u;
        device const float *xx = x + uint64_t(ib32) * 32ull;
        uint row = first_row;
        if (row < args.n_ff) {
            uint64_t off = expert_off + uint64_t(row) * uint64_t(args.iq3_row_bytes) + uint64_t(ibl) * 110ull;
            float2 pair_sum = qw3_iq3s_dot32_pair(gate_weights + off, up_weights + off, xx, kgrid, ib);
            gate_sum0 += pair_sum.x; up_sum0 += pair_sum.y;
        }
        row = first_row + 1u;
        if (row < args.n_ff) {
            uint64_t off = expert_off + uint64_t(row) * uint64_t(args.iq3_row_bytes) + uint64_t(ibl) * 110ull;
            float2 pair_sum = qw3_iq3s_dot32_pair(gate_weights + off, up_weights + off, xx, kgrid, ib);
            gate_sum1 += pair_sum.x; up_sum1 += pair_sum.y;
        }
        row = first_row + 2u;
        if (row < args.n_ff) {
            uint64_t off = expert_off + uint64_t(row) * uint64_t(args.iq3_row_bytes) + uint64_t(ibl) * 110ull;
            float2 pair_sum = qw3_iq3s_dot32_pair(gate_weights + off, up_weights + off, xx, kgrid, ib);
            gate_sum2 += pair_sum.x; up_sum2 += pair_sum.y;
        }
        row = first_row + 3u;
        if (row < args.n_ff) {
            uint64_t off = expert_off + uint64_t(row) * uint64_t(args.iq3_row_bytes) + uint64_t(ibl) * 110ull;
            float2 pair_sum = qw3_iq3s_dot32_pair(gate_weights + off, up_weights + off, xx, kgrid, ib);
            gate_sum3 += pair_sum.x; up_sum3 += pair_sum.y;
        }
    }
    gate_sum0 = simd_sum(gate_sum0); up_sum0 = simd_sum(up_sum0);
    gate_sum1 = simd_sum(gate_sum1); up_sum1 = simd_sum(up_sum1);
    gate_sum2 = simd_sum(gate_sum2); up_sum2 = simd_sum(up_sum2);
    gate_sum3 = simd_sum(gate_sum3); up_sum3 = simd_sum(up_sum3);
    if (lane == 0) {
        uint base = token * args.stride + args.hidden_offset + slot * args.n_ff;
        uint row = first_row;
        if (row < args.n_ff) scratch[base + row] = (gate_sum0 / (1.0f + exp(-gate_sum0))) * up_sum0;
        row = first_row + 1u;
        if (row < args.n_ff) scratch[base + row] = (gate_sum1 / (1.0f + exp(-gate_sum1))) * up_sum1;
        row = first_row + 2u;
        if (row < args.n_ff) scratch[base + row] = (gate_sum2 / (1.0f + exp(-gate_sum2))) * up_sum2;
        row = first_row + 3u;
        if (row < args.n_ff) scratch[base + row] = (gate_sum3 / (1.0f + exp(-gate_sum3))) * up_sum3;
    }
}
kernel void qw3_moe_down_iq4_xs_prefill_batch_reduce_fast(constant qw3_moe_prefill_batch_args &args,
                                                         device const uchar *weights,
                                                         device const float *scratch,
                                                         device float *x0,
                                                         constant int *ids,
                                                         constant float *router_weights,
                                                         threadgroup float *sh,
                                                         uint group [[threadgroup_position_in_grid]],
                                                         ushort tid [[thread_index_in_threadgroup]],
                                                         ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                                         ushort lane [[thread_index_in_simdgroup]]) {
    uint row_pair_count = (args.n_embd + 1u) / 2u;
    uint token = group / row_pair_count;
    uint row0 = (group - token * row_pair_count) * 2u;
    uint row1 = row0 + 1u;
    uint slot = uint(simd_idx);
    bool active = token < args.n_tokens && slot < args.n_active;
    uint pair = token * args.n_active + slot;
    uint expert = active ? uint(ids[pair]) : 0u;
    uint64_t expert_off = uint64_t(expert) * uint64_t(args.down_expert_bytes);
    device const float *x = scratch + uint64_t(token) * uint64_t(args.stride) + uint64_t(args.hidden_offset) + uint64_t(slot) * uint64_t(args.n_ff);
    uint ix = uint(lane) >> 4u;
    uint it = uint(lane) & 15u;
    uint ib = it >> 1u;
    uint il = it & 1u;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    uint n_blocks = args.n_ff / 256u;
    if (active) {
        for (uint b = ix; b < n_blocks; b += 2u) {
            device const float *xg = x + uint64_t(b) * 256ull + uint64_t(ib) * 32ull + uint64_t(il) * 8ull;
            if (row0 < args.n_embd) {
                device const uchar *blk = weights + expert_off + uint64_t(row0) * uint64_t(args.down_row_bytes) + uint64_t(b) * 136ull;
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
            if (row1 < args.n_embd) {
                device const uchar *blk = weights + expert_off + uint64_t(row1) * uint64_t(args.down_row_bytes) + uint64_t(b) * 136ull;
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
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    float scale = active ? router_weights[pair] : 0.0f;
    if (lane == 0) { sh[slot] = sum0 * scale; sh[8u + slot] = sum1 * scale; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0 && token < args.n_tokens) {
        float total0 = 0.0f;
        float total1 = 0.0f;
        for (uint s = 0; s < args.n_active; s++) { total0 += sh[s]; total1 += sh[8u + s]; }
        device float *row = x0 + uint64_t(token) * uint64_t(args.n_embd);
        if (row0 < args.n_embd) row[row0] += total0;
        if (row1 < args.n_embd) row[row1] += total1;
    }
}
kernel void qw3_moe_down_q6_k_prefill_batch_reduce_fast(constant qw3_moe_prefill_batch_args &args,
                                                       device const uchar *weights,
                                                       device const float *scratch,
                                                       device float *x0,
                                                       constant int *ids,
                                                       constant float *router_weights,
                                                       threadgroup float *sh,
                                                       uint group [[threadgroup_position_in_grid]],
                                                       ushort tidx [[thread_index_in_threadgroup]],
                                                       ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                                       ushort lane [[thread_index_in_simdgroup]]) {
    uint row_pair_count = (args.n_embd + 1u) / 2u;
    uint token = group / row_pair_count;
    uint row0 = (group - token * row_pair_count) * 2u;
    uint row1 = row0 + 1u;
    uint slot = uint(simd_idx);
    bool active = token < args.n_tokens && slot < args.n_active;
    uint pair = token * args.n_active + slot;
    uint expert = active ? uint(ids[pair]) : 0u;
    uint64_t expert_off = uint64_t(expert) * uint64_t(args.down_expert_bytes);
    device const float *x = scratch + uint64_t(token) * uint64_t(args.stride) + uint64_t(args.hidden_offset) + uint64_t(slot) * uint64_t(args.n_ff);
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
    uint n_blocks = args.n_ff / 256u;
    if (active) {
        for (uint b = ix; b < n_blocks; b += 2u) {
            device const float *yy = x + uint64_t(b) * 256ull + y_offset;
            if (row0 < args.n_embd) {
                device const uchar *blk = weights + expert_off + uint64_t(row0) * uint64_t(args.down_row_bytes) + uint64_t(b) * 210ull;
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
            if (row1 < args.n_embd) {
                device const uchar *blk = weights + expert_off + uint64_t(row1) * uint64_t(args.down_row_bytes) + uint64_t(b) * 210ull;
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
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    float scale = active ? router_weights[pair] : 0.0f;
    if (lane == 0) { sh[slot] = sum0 * scale; sh[8u + slot] = sum1 * scale; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tidx == 0 && token < args.n_tokens) {
        float total0 = 0.0f;
        float total1 = 0.0f;
        for (uint s = 0; s < args.n_active; s++) { total0 += sh[s]; total1 += sh[8u + s]; }
        device float *row = x0 + uint64_t(token) * uint64_t(args.n_embd);
        if (row0 < args.n_embd) row[row0] += total0;
        if (row1 < args.n_embd) row[row1] += total1;
    }
}
struct qw3_moe_expert_map_args { uint n_tokens; uint n_active; uint n_expert; uint pair_capacity; uint n_ff; uint n_embd; };
kernel void qw3_moe_topk_expert_map(constant qw3_moe_expert_map_args &args,
                                    device const int *ids,
                                    device uint *counts,
                                    device int *pair_ids,
                                    device atomic_uint *block_count,
                                    device uint *block_ids,
                                    device uint *dispatch_ff,
                                    device uint *dispatch_embd,
                                    uint expert [[thread_position_in_grid]]) {
    if (expert >= args.n_expert) return;
    if (expert == 0u) {
        atomic_store_explicit(block_count, 0u, memory_order_relaxed);
        dispatch_ff[1] = (args.n_ff + 63u) / 64u;
        dispatch_ff[2] = 1u;
        dispatch_embd[1] = (args.n_embd + 63u) / 64u;
        dispatch_embd[2] = 1u;
    }
    threadgroup_barrier(mem_flags::mem_device);
    uint n = 0u;
    for (uint t = 0u; t < args.n_tokens; t++) {
        int found = -1;
        for (uint slot = 0u; slot < args.n_active; slot++) {
            int eid = ids[t * args.n_active + slot];
            if (eid == int(expert)) { found = int(slot); break; }
        }
        if (found >= 0 && n < args.pair_capacity) {
            pair_ids[expert * args.pair_capacity + n] = int(t * args.n_active + uint(found));
            n++;
        }
    }
    counts[expert] = n;
    for (uint r1u = 0u; r1u < n; r1u += 32u) {
        uint dst = atomic_fetch_add_explicit(block_count, 1u, memory_order_relaxed);
        block_ids[dst] = expert | (r1u << 8u);
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (expert == 0u) {
        uint n_blocks = atomic_load_explicit(block_count, memory_order_relaxed);
        dispatch_ff[0] = n_blocks;
        dispatch_embd[0] = n_blocks;
    }
}
inline half qw3_iq3s_dequant_k_expanded(device const uchar *row, device const uchar *kgrid, uint k) {
    uint block = k >> 8u;
    uint local = k & 255u;
    uint ib = local >> 5u;
    uint within = local & 31u;
    uint l = within >> 3u;
    uint j = within & 3u;
    bool second = (within & 4u) != 0u;
    device const uchar *blk = row + uint64_t(block) * 110ull;
    float d = float(*((device const half *)blk));
    device const uchar *qs = blk + 2u;
    device const uchar *qh = qs + 64u;
    device const uchar *signs = qh + 8u;
    device const uchar *scales = signs + 32u;
    float db = d * float(1u + 2u * ((uint(scales[ib >> 1u]) >> (4u * (ib & 1u))) & 15u));
    uchar qhb = qh[ib];
    uchar qsb = qs[ib * 8u + 2u * l + (second ? 1u : 0u)];
    uint idx = uint(qsb) | ((uint(qhb) << ((second ? 7u : 8u) - 2u * l)) & 256u);
    uchar s = signs[ib * 4u + l];
    float sign = (uint(s) & (1u << (j + (second ? 4u : 0u)))) ? -1.0f : 1.0f;
    return half(db * float(kgrid[(idx & 511u) * 4u + j]) * sign);
}
inline half qw3_iq3s_dequant16_expanded(float dl, device const uchar *qs16, uchar qh16, device const uchar *signs16, device const uchar *kgrid, uint i) {
    uint qn = i >> 2u;
    uint j = i & 3u;
    uint idx = uint(qs16[qn]) | ((uint(qh16) << (8u - qn)) & 256u);
    uchar sb = signs16[qn >> 1u];
    float sign = (uint(sb) & (1u << (j + 4u * (qn & 1u)))) ? -1.0f : 1.0f;
    return half(dl * float(kgrid[(idx & 511u) * 4u + j]) * sign);
}
inline half4x4 qw3_iq3s_dequant4x4_expanded(float dl, device const uchar *qs16, uchar qh16, device const uchar *signs16, device const uchar *kgrid) {
    half4x4 reg;
    uint idx0 = uint(qs16[0]) | ((uint(qh16) << 8u) & 256u);
    uint idx1 = uint(qs16[1]) | ((uint(qh16) << 7u) & 256u);
    uint idx2 = uint(qs16[2]) | ((uint(qh16) << 6u) & 256u);
    uint idx3 = uint(qs16[3]) | ((uint(qh16) << 5u) & 256u);
    device const uchar *g0 = kgrid + (idx0 & 511u) * 4u;
    device const uchar *g1 = kgrid + (idx1 & 511u) * 4u;
    device const uchar *g2 = kgrid + (idx2 & 511u) * 4u;
    device const uchar *g3 = kgrid + (idx3 & 511u) * 4u;
    uchar s0 = signs16[0];
    uchar s1 = signs16[1];
    reg[0][0] = half(dl * float(g0[0]) * ((uint(s0) & 1u) ? -1.0f : 1.0f));
    reg[0][1] = half(dl * float(g0[1]) * ((uint(s0) & 2u) ? -1.0f : 1.0f));
    reg[0][2] = half(dl * float(g0[2]) * ((uint(s0) & 4u) ? -1.0f : 1.0f));
    reg[0][3] = half(dl * float(g0[3]) * ((uint(s0) & 8u) ? -1.0f : 1.0f));
    reg[1][0] = half(dl * float(g1[0]) * ((uint(s0) & 16u) ? -1.0f : 1.0f));
    reg[1][1] = half(dl * float(g1[1]) * ((uint(s0) & 32u) ? -1.0f : 1.0f));
    reg[1][2] = half(dl * float(g1[2]) * ((uint(s0) & 64u) ? -1.0f : 1.0f));
    reg[1][3] = half(dl * float(g1[3]) * ((uint(s0) & 128u) ? -1.0f : 1.0f));
    reg[2][0] = half(dl * float(g2[0]) * ((uint(s1) & 1u) ? -1.0f : 1.0f));
    reg[2][1] = half(dl * float(g2[1]) * ((uint(s1) & 2u) ? -1.0f : 1.0f));
    reg[2][2] = half(dl * float(g2[2]) * ((uint(s1) & 4u) ? -1.0f : 1.0f));
    reg[2][3] = half(dl * float(g2[3]) * ((uint(s1) & 8u) ? -1.0f : 1.0f));
    reg[3][0] = half(dl * float(g3[0]) * ((uint(s1) & 16u) ? -1.0f : 1.0f));
    reg[3][1] = half(dl * float(g3[1]) * ((uint(s1) & 32u) ? -1.0f : 1.0f));
    reg[3][2] = half(dl * float(g3[2]) * ((uint(s1) & 64u) ? -1.0f : 1.0f));
    reg[3][3] = half(dl * float(g3[3]) * ((uint(s1) & 128u) ? -1.0f : 1.0f));
    return reg;
}
kernel void qw3_moe_iq3_s_prefill_mapped(constant qw3_moe_prefill_batch_args &args,
                                         device const uchar *weights,
                                         device const float *x1,
                                         device float *out_slots,
                                         device const uchar *kgrid,
                                         device const uint *counts,
                                         device const int *pair_ids,
                                         device const uint *block_ids,
                                         threadgroup char *shmem [[threadgroup(0)]],
                                         uint3 group [[threadgroup_position_in_grid]],
                                         ushort tid [[thread_index_in_threadgroup]],
                                         ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;
    uint r0u = group.y * NR0;
    uint expert = group.z;
    uint r1u = group.x * NR1;
    if (args.compact_blocks != 0u) {
        uint block = block_ids[group.x];
        expert = block & 255u;
        r1u = block >> 8u;
    }
    uint count = counts[expert];
    if (r0u >= args.n_ff || r1u >= count) return;
    int nr0 = int(min(uint(NR0), args.n_ff - r0u));
    int nr1 = int(min(uint(NR1), count - r1u));
    int lr0 = min(int(tid) / NL0, nr0 - 1);
    int lr1 = min(int(tid) / NL1, nr1 - 1);
    short il0 = short(tid % NL0);
    uint row = r0u + uint(lr0);
    uint map_base = expert * args.n_tokens + r1u;
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint loop_k = 0u; loop_k < args.n_in; loop_k += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const uchar *wrow = weights + uint64_t(expert) * uint64_t(args.iq3_expert_bytes) + uint64_t(row) * uint64_t(args.iq3_row_bytes);
        uint il_base = (loop_k & 255u) >> 4u;
        uint block_k = loop_k >> 8u;
        device const uchar *blk = wrow + uint64_t(block_k) * 110ull;
        float d = float(*((device const half *)blk));
        device const uchar *qs = blk + 2u;
        device const uchar *qh = qs + 64u;
        device const uchar *signs = qh + 8u;
        device const uchar *scales = signs + 32u;
        uint il = il_base + uint(il0);
        uint ib32 = il >> 1u;
        uint half_il = il & 1u;
        float dl = d * float(1u + 2u * ((uint(scales[ib32 >> 1u]) >> (4u * (ib32 & 1u))) & 15u));
        device const uchar *qs16 = qs + 8u * ib32 + 4u * half_il;
        device const uchar *signs16 = signs + 4u * ib32 + 2u * half_il;
        uchar qh16 = uchar(uint(qh[ib32]) >> (4u * half_il));
        for (short i = 0; i < 16; i++) {
            uint k = loop_k + uint(16 * il0 + i);
            const short sx = short(2 * il0 + i / 8);
            const short sy = short((tid / NL0) / 8);
            const short lx = short((tid / NL0) % 8);
            const short ly = short(i % 8);
            const short ib = short(8 * sx + sy);
            *(sa + 64 * ib + 8 * ly + lx) = k < args.n_in ? qw3_iq3s_dequant16_expanded(dl, qs16, qh16, signs16, kgrid, uint(i)) : half(0.0f);
        }
        int pid = pair_ids[map_base + uint(lr1)];
        uint token = uint(pid) / args.n_active;
        device const float *x = x1 + uint64_t(token) * uint64_t(args.n_embd) + uint64_t(loop_k);
        for (short i = 0; i < 8; i++) {
            const short sx = short(tid % NL1);
            const short sy = short((tid / NL1) / 8);
            const short lx = i;
            const short ly = short((tid / NL1) % 8);
            const short ib = short(4 * sx + sy);
            uint kk = uint(8 * sx + i);
            *(sb + 64 * ib + 8 * ly + lx) = (uint(lr1) < uint(nr1) && loop_k + kk < args.n_in) ? half(x[kk]) : half(0.0f);
        }
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *tmp = ((threadgroup float *)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
    for (short i = 0; i < 8; i++) simdgroup_store(mc[i], tmp + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short j = short(sgitg); j < nr1; j += 4) {
        int pid = pair_ids[map_base + uint(j)];
        device float *dst = out_slots + uint64_t(uint(pid)) * uint64_t(args.n_ff) + uint64_t(r0u);
        threadgroup float *src = ((threadgroup float *)shmem) + int(j) * NR0;
        int i = int(tid & 31u);
        device float4 *dst4 = (device float4 *)dst;
        threadgroup float4 *src4 = (threadgroup float4 *)src;
        for (; i < nr0 / 4; i += 32) dst4[i] = src4[i];
        i = 4 * (nr0 / 4) + int(tid & 31u);
        for (; i < nr0; i += 32) dst[i] = src[i];
    }
}
#ifdef QW3_METAL_HAS_TENSOR
kernel void qw3_moe_iq3_s_prefill_mapped_mpp(constant qw3_moe_prefill_batch_args &args,
                                             device const uchar *weights,
                                             device const float *x1,
                                             device float *out_slots,
                                             device const uchar *kgrid,
                                             device const uint *counts,
                                             device const int *pair_ids,
                                             device const uint *block_ids,
                                             threadgroup char *shmem [[threadgroup(0)]],
                                             uint3 group [[threadgroup_position_in_grid]],
                                             ushort tid [[thread_index_in_threadgroup]],
                                             ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);
    threadgroup float *sc = (threadgroup float *)shmem;
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;
    uint r0u = group.y * NR0;
    uint expert = group.z;
    uint r1u = group.x * NR1;
    if (args.compact_blocks != 0u) {
        uint block = block_ids[group.x];
        expert = block & 255u;
        r1u = block >> 8u;
    }
    uint count = counts[expert];
    if (r0u >= args.n_ff || r1u >= count) return;
    int nr0 = int(min(uint(NR0), args.n_ff - r0u));
    int nr1 = int(min(uint(NR1), count - r1u));
    int lr0 = min(int(tid) / NL0, nr0 - 1);
    int lr1 = min(int(tid) / NL1, nr1 - 1);
    short il0 = short(tid % NL0);
    uint row = r0u + uint(lr0);
    uint map_base = expert * args.n_tokens + r1u;
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
    for (uint loop_k = 0u; loop_k < args.n_in; loop_k += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const uchar *wrow = weights + uint64_t(expert) * uint64_t(args.iq3_expert_bytes) + uint64_t(row) * uint64_t(args.iq3_row_bytes);
        uint il_base = (loop_k & 255u) >> 4u;
        uint block_k = loop_k >> 8u;
        device const uchar *blk = wrow + uint64_t(block_k) * 110ull;
        float d = float(*((device const half *)blk));
        device const uchar *qs = blk + 2u;
        device const uchar *qh = qs + 64u;
        device const uchar *signs = qh + 8u;
        device const uchar *scales = signs + 32u;
        uint il = il_base + uint(il0);
        uint ib32 = il >> 1u;
        uint half_il = il & 1u;
        float dl = d * float(1u + 2u * ((uint(scales[ib32 >> 1u]) >> (4u * (ib32 & 1u))) & 15u));
        device const uchar *qs16 = qs + 8u * ib32 + 4u * half_il;
        device const uchar *signs16 = signs + 4u * ib32 + 2u * half_il;
        uchar qh16 = uchar(uint(qh[ib32]) >> (4u * half_il));
        for (short i = 0; i < 16; i++) {
            uint k = loop_k + uint(16 * il0 + i);
            const short sx = short(2 * il0 + i / 8);
            const short sy = short((tid / NL0) / 8);
            const short lx = short(i % 8);
            const short ly = short((tid / NL0) % 8);
            *(sa + NK * (8 * sy + ly) + 8 * sx + lx) = k < args.n_in ? qw3_iq3s_dequant16_expanded(dl, qs16, qh16, signs16, kgrid, uint(i)) : half(0.0f);
        }
        int pid = pair_ids[map_base + uint(lr1)];
        uint token = uint(pid) / args.n_active;
        device const float *x = x1 + uint64_t(token) * uint64_t(args.n_embd) + uint64_t(loop_k);
        for (short i = 0; i < 8; i++) {
            const short sx = short(tid % NL1);
            const short sy = short((tid / NL1) / 8);
            const short lx = i;
            const short ly = short((tid / NL1) % 8);
            uint kk = uint(8 * sx + i);
            *(sb + NK * (8 * sy + ly) + 8 * sx + lx) = (uint(lr1) < uint(nr1) && loop_k + kk < args.n_in) ? half(x[kk]) : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);
        mm.run(sB, sA, cT);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, NR1));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short j = short(sgitg); j < nr1; j += 4) {
        int pid = pair_ids[map_base + uint(j)];
        device float *dst = out_slots + uint64_t(uint(pid)) * uint64_t(args.n_ff) + uint64_t(r0u);
        threadgroup float *src = sc + int(j) * NR0;
        int i = int(tid & 31u);
        device float4 *dst4 = (device float4 *)dst;
        threadgroup float4 *src4 = (threadgroup float4 *)src;
        for (; i < nr0 / 4; i += 32) dst4[i] = src4[i];
        i = 4 * (nr0 / 4) + int(tid & 31u);
        for (; i < nr0; i += 32) dst[i] = src[i];
    }
}
#endif
kernel void qw3_moe_iq3_s_swiglu_prefill_pair_mapped(constant qw3_moe_prefill_batch_args &args,
                                                       device const uchar *gate_weights,
                                                       device const uchar *up_weights,
                                                       device const float *x1,
                                                       device float *scratch,
                                                       device const uchar *kgrid,
                                                       device const uint *counts,
                                                       device const int *pair_ids,
                                                       device const uint *block_ids,
                                                       threadgroup char *shmem [[threadgroup(0)]],
                                                       uint3 group [[threadgroup_position_in_grid]],
                                                       ushort tid [[thread_index_in_threadgroup]],
                                                       ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;
    uint r0u = group.y * NR0;
    uint expert = group.z;
    uint r1u = group.x * NR1;
    if (args.compact_blocks != 0u) {
        uint block = block_ids[group.x];
        expert = block & 255u;
        r1u = block >> 8u;
    }
    uint count = counts[expert];
    if (r0u >= args.n_ff || r1u >= count) return;
    int nr0 = int(min(uint(NR0), args.n_ff - r0u));
    int nr1 = int(min(uint(NR1), count - r1u));
    int lr0 = min(int(tid) / NL0, nr0 - 1);
    int lr1 = min(int(tid) / NL1, nr1 - 1);
    short il0 = short(tid % NL0);
    uint row = r0u + uint(lr0);
    uint map_base = expert * args.n_tokens + r1u;
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc_gate[8];
    simdgroup_float8x8 mc_up[8];
    for (short i = 0; i < 8; i++) {
        mc_gate[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_up[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }
    for (uint loop_k = 0u; loop_k < args.n_in; loop_k += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const uchar *wrow_gate = gate_weights + uint64_t(expert) * uint64_t(args.iq3_expert_bytes) + uint64_t(row) * uint64_t(args.iq3_row_bytes);
        uint il_base = (loop_k & 255u) >> 4u;
        uint block_k = loop_k >> 8u;
        uint il = il_base + uint(il0);
        uint ib32 = il >> 1u;
        uint half_il = il & 1u;
        device const uchar *blk_gate = wrow_gate + uint64_t(block_k) * 110ull;
        float d_gate = float(*((device const half *)blk_gate));
        device const uchar *qs_gate = blk_gate + 2u;
        device const uchar *qh_gate = qs_gate + 64u;
        device const uchar *signs_gate = qh_gate + 8u;
        device const uchar *scales_gate = signs_gate + 32u;
        float dl_gate = d_gate * float(1u + 2u * ((uint(scales_gate[ib32 >> 1u]) >> (4u * (ib32 & 1u))) & 15u));
        device const uchar *qs16_gate = qs_gate + 8u * ib32 + 4u * half_il;
        device const uchar *signs16_gate = signs_gate + 4u * ib32 + 2u * half_il;
        uchar qh16_gate = uchar(uint(qh_gate[ib32]) >> (4u * half_il));
        for (short i = 0; i < 16; i++) {
            uint k = loop_k + uint(16 * il0 + i);
            const short sx = short(2 * il0 + i / 8);
            const short sy = short((tid / NL0) / 8);
            const short lx = short((tid / NL0) % 8);
            const short ly = short(i % 8);
            const short ib = short(8 * sx + sy);
            *(sa + 64 * ib + 8 * ly + lx) = k < args.n_in ? qw3_iq3s_dequant16_expanded(dl_gate, qs16_gate, qh16_gate, signs16_gate, kgrid, uint(i)) : half(0.0f);
        }
        int pid = pair_ids[map_base + uint(lr1)];
        uint token = uint(pid) / args.n_active;
        device const float *x = x1 + uint64_t(token) * uint64_t(args.n_embd) + uint64_t(loop_k);
        for (short i = 0; i < 8; i++) {
            const short sx = short(tid % NL1);
            const short sy = short((tid / NL1) / 8);
            const short lx = i;
            const short ly = short((tid / NL1) % 8);
            const short ib = short(4 * sx + sy);
            uint kk = uint(8 * sx + i);
            *(sb + 64 * ib + 8 * ly + lx) = (uint(lr1) < uint(nr1) && loop_k + kk < args.n_in) ? half(x[kk]) : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half *lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half *lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc_gate[i], mb[i / 4], ma[i % 4], mc_gate[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const uchar *wrow_up = up_weights + uint64_t(expert) * uint64_t(args.iq3_expert_bytes) + uint64_t(row) * uint64_t(args.iq3_row_bytes);
        device const uchar *blk_up = wrow_up + uint64_t(block_k) * 110ull;
        float d_up = float(*((device const half *)blk_up));
        device const uchar *qs_up = blk_up + 2u;
        device const uchar *qh_up = qs_up + 64u;
        device const uchar *signs_up = qh_up + 8u;
        device const uchar *scales_up = signs_up + 32u;
        float dl_up = d_up * float(1u + 2u * ((uint(scales_up[ib32 >> 1u]) >> (4u * (ib32 & 1u))) & 15u));
        device const uchar *qs16_up = qs_up + 8u * ib32 + 4u * half_il;
        device const uchar *signs16_up = signs_up + 4u * ib32 + 2u * half_il;
        uchar qh16_up = uchar(uint(qh_up[ib32]) >> (4u * half_il));
        for (short i = 0; i < 16; i++) {
            uint k = loop_k + uint(16 * il0 + i);
            const short sx = short(2 * il0 + i / 8);
            const short sy = short((tid / NL0) / 8);
            const short lx = short((tid / NL0) % 8);
            const short ly = short(i % 8);
            const short ib = short(8 * sx + sy);
            *(sa + 64 * ib + 8 * ly + lx) = k < args.n_in ? qw3_iq3s_dequant16_expanded(dl_up, qs16_up, qh16_up, signs16_up, kgrid, uint(i)) : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        lsma = sa + 4 * 64 * (sgitg % 2);
        lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc_up[i], mb[i / 4], ma[i % 4], mc_up[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *tmp_gate = (threadgroup float *)shmem;
    threadgroup float *tmp_up = tmp_gate + NR0 * NR1;
    threadgroup float *gate_dst = tmp_gate + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
    threadgroup float *up_dst = tmp_up + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc_gate[i], gate_dst + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
        simdgroup_store(mc_up[i], up_dst + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short j = short(sgitg); j < nr1; j += 4) {
        int pid = pair_ids[map_base + uint(j)];
        uint token = uint(pid) / args.n_active;
        uint slot = uint(pid) - token * args.n_active;
        device float *dst = scratch + uint64_t(token) * uint64_t(args.stride) + uint64_t(args.hidden_offset) + uint64_t(slot) * uint64_t(args.n_ff) + uint64_t(r0u);
        threadgroup float *src_gate = tmp_gate + int(j) * NR0;
        threadgroup float *src_up = tmp_up + int(j) * NR0;
        int i = int(tid & 31u);
        device float4 *dst4 = (device float4 *)dst;
        threadgroup float4 *gate4 = (threadgroup float4 *)src_gate;
        threadgroup float4 *up4 = (threadgroup float4 *)src_up;
        for (; i < nr0 / 4; i += 32) {
            float4 g = gate4[i];
            dst4[i] = (g / (float4(1.0f) + exp(-g))) * up4[i];
        }
        i = 4 * (nr0 / 4) + int(tid & 31u);
        for (; i < nr0; i += 32) {
            float g = src_gate[i];
            dst[i] = (g / (1.0f + exp(-g))) * src_up[i];
        }
    }
}
#ifdef QW3_METAL_HAS_TENSOR
kernel void qw3_moe_iq3_s_swiglu_prefill_pair_mapped_mpp(constant qw3_moe_prefill_batch_args &args,
                                                           device const uchar *gate_weights,
                                                           device const uchar *up_weights,
                                                           device const float *x1,
                                                           device float *scratch,
                                                           device const uchar *kgrid,
                                                           device const uint *counts,
                                                           device const int *pair_ids,
                                                           device const uint *block_ids,
                                                           device const float *router_weights,
                                                           threadgroup char *shmem [[threadgroup(0)]],
                                                           uint3 group [[threadgroup_position_in_grid]],
                                                           ushort tid [[thread_index_in_threadgroup]],
                                                           ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);
    threadgroup float *sc = (threadgroup float *)shmem;
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;
    uint r0u = group.y * NR0;
    uint expert = group.z;
    uint r1u = group.x * NR1;
    if (args.compact_blocks != 0u) {
        uint block = block_ids[group.x];
        expert = block & 255u;
        r1u = block >> 8u;
    }
    uint count = counts[expert];
    if (r0u >= args.n_ff || r1u >= count) return;
    int nr0 = int(min(uint(NR0), args.n_ff - r0u));
    int nr1 = int(min(uint(NR1), count - r1u));
    int lr0 = min(int(tid) / NL0, nr0 - 1);
    int lr1 = min(int(tid) / NL1, nr1 - 1);
    short il0 = short(tid % NL0);
    uint row = r0u + uint(lr0);
    uint map_base = expert * args.n_tokens + r1u;
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto cGate = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
    auto cUp = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
    for (uint loop_k = 0u; loop_k < args.n_in; loop_k += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint il_base = (loop_k & 255u) >> 4u;
        uint block_k = loop_k >> 8u;
        uint il = il_base + uint(il0);
        uint ib32 = il >> 1u;
        uint half_il = il & 1u;
        device const uchar *wrow_gate = gate_weights + uint64_t(expert) * uint64_t(args.iq3_expert_bytes) + uint64_t(row) * uint64_t(args.iq3_row_bytes);
        device const uchar *blk_gate = wrow_gate + uint64_t(block_k) * 110ull;
        float d_gate = float(*((device const half *)blk_gate));
        device const uchar *qs_gate = blk_gate + 2u;
        device const uchar *qh_gate = qs_gate + 64u;
        device const uchar *signs_gate = qh_gate + 8u;
        device const uchar *scales_gate = signs_gate + 32u;
        float dl_gate = d_gate * float(1u + 2u * ((uint(scales_gate[ib32 >> 1u]) >> (4u * (ib32 & 1u))) & 15u));
        device const uchar *qs16_gate = qs_gate + 8u * ib32 + 4u * half_il;
        device const uchar *signs16_gate = signs_gate + 4u * ib32 + 2u * half_il;
        uchar qh16_gate = uchar(uint(qh_gate[ib32]) >> (4u * half_il));
        const short sy_a = short((tid / NL0) / 8);
        const short ly_a = short((tid / NL0) % 8);
        half4x4 gate4 = qw3_iq3s_dequant4x4_expanded(dl_gate, qs16_gate, qh16_gate, signs16_gate, kgrid);
        for (short i = 0; i < 16; i++) {
            const short sx = short(2 * il0 + i / 8);
            const short lx = short(i % 8);
            *(sa + NK * (8 * sy_a + ly_a) + 8 * sx + lx) = gate4[i / 4][i & 3];
        }
        int pid = pair_ids[map_base + uint(lr1)];
        uint token = uint(pid) / args.n_active;
        device const float *x = x1 + uint64_t(token) * uint64_t(args.n_embd) + uint64_t(loop_k);
        for (short i = 0; i < 8; i++) {
            const short sx = short(tid % NL1);
            const short sy = short((tid / NL1) / 8);
            const short lx = i;
            const short ly = short((tid / NL1) % 8);
            uint kk = uint(8 * sx + i);
            *(sb + NK * (8 * sy + ly) + 8 * sx + lx) = half(x[kk]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);
        mm.run(sB, sA, cGate);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const uchar *wrow_up = up_weights + uint64_t(expert) * uint64_t(args.iq3_expert_bytes) + uint64_t(row) * uint64_t(args.iq3_row_bytes);
        device const uchar *blk_up = wrow_up + uint64_t(block_k) * 110ull;
        float d_up = float(*((device const half *)blk_up));
        device const uchar *qs_up = blk_up + 2u;
        device const uchar *qh_up = qs_up + 64u;
        device const uchar *signs_up = qh_up + 8u;
        device const uchar *scales_up = signs_up + 32u;
        float dl_up = d_up * float(1u + 2u * ((uint(scales_up[ib32 >> 1u]) >> (4u * (ib32 & 1u))) & 15u));
        device const uchar *qs16_up = qs_up + 8u * ib32 + 4u * half_il;
        device const uchar *signs16_up = signs_up + 4u * ib32 + 2u * half_il;
        uchar qh16_up = uchar(uint(qh_up[ib32]) >> (4u * half_il));
        half4x4 up4 = qw3_iq3s_dequant4x4_expanded(dl_up, qs16_up, qh16_up, signs16_up, kgrid);
        for (short i = 0; i < 16; i++) {
            const short sx = short(2 * il0 + i / 8);
            const short lx = short(i % 8);
            *(sa + NK * (8 * sy_a + ly_a) + 8 * sx + lx) = up4[i / 4][i & 3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sA = tA.slice(0, 0);
        sB = tB.slice(0, 0);
        mm.run(sB, sA, cUp);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tGate = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, NR1));
    auto tUp = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc + NR0 * NR1, dextents<int32_t, 2>(NR0, NR1));
    cGate.store(tGate);
    cUp.store(tUp);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short j = short(sgitg); j < nr1; j += 4) {
        int pid = pair_ids[map_base + uint(j)];
        device float *dst = scratch + uint64_t(uint(pid)) * uint64_t(args.n_ff) + uint64_t(r0u);
        threadgroup float *src_gate = sc + int(j) * NR0;
        threadgroup float *src_up = sc + NR0 * NR1 + int(j) * NR0;
        float scale = args.mid_preweighted != 0u ? router_weights[uint(pid)] : 1.0f;
        int i = int(tid & 31u);
        device float4 *dst4 = (device float4 *)dst;
        threadgroup float4 *gate4 = (threadgroup float4 *)src_gate;
        threadgroup float4 *up4 = (threadgroup float4 *)src_up;
        for (; i < nr0 / 4; i += 32) {
            float4 g = gate4[i];
            dst4[i] = ((g / (float4(1.0f) + exp(-g))) * up4[i]) * scale;
        }
        i = 4 * (nr0 / 4) + int(tid & 31u);
        for (; i < nr0; i += 32) {
            float g = src_gate[i];
            dst[i] = ((g / (1.0f + exp(-g))) * src_up[i]) * scale;
        }
    }
}
#endif
kernel void qw3_moe_swiglu_slots_to_hidden(constant qw3_moe_prefill_batch_args &args,
                                          device const float *gate_slots,
                                          device const float *up_slots,
                                          device float *scratch,
                                          uint gid [[thread_position_in_grid]]) {
    uint total = args.n_tokens * args.n_active * args.n_ff;
    if (gid >= total) return;
    uint row = gid % args.n_ff;
    uint pair = gid / args.n_ff;
    uint token = pair / args.n_active;
    uint slot = pair - token * args.n_active;
    float g = gate_slots[gid];
    scratch[uint64_t(token) * uint64_t(args.stride) + uint64_t(args.hidden_offset) + uint64_t(slot) * uint64_t(args.n_ff) + row] = (g / (1.0f + exp(-g))) * up_slots[gid];
}
kernel void qw3_moe_swiglu_slots_to_mid_f32(constant qw3_moe_prefill_batch_args &args,
                                           device const float *gate_slots,
                                           device const float *up_slots,
                                           device float *mid,
                                           uint gid [[thread_position_in_grid]]) {
    uint total = args.n_tokens * args.n_active * args.n_ff;
    if (gid >= total) return;
    float g = gate_slots[gid];
    mid[gid] = (g / (1.0f + exp(-g))) * up_slots[gid];
}
kernel void qw3_moe_swiglu_slots_to_hidden_f16(constant qw3_moe_prefill_batch_args &args,
                                              device const float *gate_slots,
                                              device const float *up_slots,
                                              device half *mid,
                                              uint gid [[thread_position_in_grid]]) {
    uint total = args.n_tokens * args.n_active * args.n_ff;
    if (gid >= total) return;
    float g = gate_slots[gid];
    mid[gid] = half((g / (1.0f + exp(-g))) * up_slots[gid]);
}
inline half qw3_iq4xs_dequant_k(device const uchar *row, uint k) {
    uint block = k >> 8u;
    uint local = k & 255u;
    uint ib = local >> 5u;
    uint within = local & 31u;
    uint il = (within & 15u) >> 3u;
    uint j = within & 7u;
    bool hi = within >= 16u;
    device const uchar *blk = row + uint64_t(block) * 136ull;
    float d = float(*((device const half *)blk));
    ushort scales_h = *((device const ushort *)(blk + 2u));
    device const uchar *scales_l = blk + 4u;
    uint ls = ((uint(scales_l[ib >> 1u]) >> (4u * (ib & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);
    uchar packed = *(scales_l + 4u + ib * 16u + il * 8u + j);
    uint q = hi ? (uint(packed) >> 4u) : (uint(packed) & 15u);
    return half(d * float(int(ls) - 32) * qw3_iq4nl_val(q));
}
inline half qw3_iq4xs_dequant16(float dl, device const uchar *q16, bool hi, uint i) {
    uchar packed = q16[8u * (i >> 3u) + (i & 7u)];
    uint q = hi ? (uint(packed) >> 4u) : (uint(packed) & 15u);
    return half(dl * qw3_iq4nl_val(q));
}
inline half4x4 qw3_iq4xs_dequant4x4(float dl, device const uchar *q16, bool hi) {
    half4x4 reg;
    for (uint i = 0u; i < 16u; i++) {
        uchar packed = q16[8u * (i >> 3u) + (i & 7u)];
        uint q = hi ? (uint(packed) >> 4u) : (uint(packed) & 15u);
        reg[i >> 2u][i & 3u] = half(dl * qw3_iq4nl_val(q));
    }
    return reg;
}
kernel void qw3_moe_down_iq4_xs_prefill_mapped(constant qw3_moe_prefill_batch_args &args,
                                                 device const uchar *weights,
                                                 device const float *scratch,
                                                 device float *down_slots,
                                                 device const uint *counts,
                                                 device const int *pair_ids,
                                                 device const float *router_weights,
                                                 device const uint *block_ids,
                                                 threadgroup char *shmem [[threadgroup(0)]],
                                                 uint3 group [[threadgroup_position_in_grid]],
                                                 ushort tid [[thread_index_in_threadgroup]],
                                                 ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;
    uint r0u = group.y * NR0;
    uint expert = group.z;
    uint r1u = group.x * NR1;
    if (args.compact_blocks != 0u) {
        uint block = block_ids[group.x];
        expert = block & 255u;
        r1u = block >> 8u;
    }
    uint count = counts[expert];
    if (r0u >= args.n_embd || r1u >= count) return;
    int nr0 = int(min(uint(NR0), args.n_embd - r0u));
    int nr1 = int(min(uint(NR1), count - r1u));
    int lr0 = min(int(tid) / NL0, nr0 - 1);
    int lr1 = min(int(tid) / NL1, nr1 - 1);
    short il0 = short(tid % NL0);
    uint row = r0u + uint(lr0);
    uint map_base = expert * args.n_tokens + r1u;
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint loop_k = 0u; loop_k < args.n_ff; loop_k += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const uchar *wrow = weights + uint64_t(expert) * uint64_t(args.down_expert_bytes) + uint64_t(row) * uint64_t(args.down_row_bytes);
        uint il_base = (loop_k & 255u) >> 4u;
        uint block_k = loop_k >> 8u;
        device const uchar *blk = wrow + uint64_t(block_k) * 136ull;
        float d = float(*((device const half *)blk));
        ushort scales_h = *((device const ushort *)(blk + 2u));
        device const uchar *scales_l = blk + 4u;
        uint il = il_base + uint(il0);
        uint ib32 = il >> 1u;
        uint half_il = il & 1u;
        uint ls = ((uint(scales_l[ib32 >> 1u]) >> (4u * (ib32 & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib32)) & 3u) << 4u);
        float dl = d * float(int(ls) - 32);
        device const uchar *q16 = scales_l + 4u + ib32 * 16u;
        bool hi = half_il != 0u;
        for (short i = 0; i < 16; i++) {
            uint k = loop_k + uint(16 * il0 + i);
            const short sx = short(2 * il0 + i / 8);
            const short sy = short((tid / NL0) / 8);
            const short lx = short((tid / NL0) % 8);
            const short ly = short(i % 8);
            const short ib = short(8 * sx + sy);
            *(sa + 64 * ib + 8 * ly + lx) = k < args.n_ff ? qw3_iq4xs_dequant16(dl, q16, hi, uint(i)) : half(0.0f);
        }
        int pid = pair_ids[map_base + uint(lr1)];
        uint token = uint(pid) / args.n_active;
        uint slot = uint(pid) - token * args.n_active;
        device const float *hidden = scratch + uint64_t(token) * uint64_t(args.stride) + uint64_t(args.hidden_offset) + uint64_t(slot) * uint64_t(args.n_ff) + uint64_t(loop_k);
        for (short i = 0; i < 8; i++) {
            const short sx = short(tid % NL1);
            const short sy = short((tid / NL1) / 8);
            const short lx = i;
            const short ly = short((tid / NL1) % 8);
            const short ib = short(4 * sx + sy);
            uint kk = uint(8 * sx + i);
            *(sb + 64 * ib + 8 * ly + lx) = (uint(lr1) < uint(nr1) && loop_k + kk < args.n_ff) ? half(hidden[kk]) : half(0.0f);
        }
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *tmp = ((threadgroup float *)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
    for (short i = 0; i < 8; i++) simdgroup_store(mc[i], tmp + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short j = short(sgitg); j < nr1; j += 4) {
        int pid = pair_ids[map_base + uint(j)];
        device float *dst = down_slots + uint64_t(uint(pid)) * uint64_t(args.n_embd) + uint64_t(r0u);
        threadgroup float *src = ((threadgroup float *)shmem) + int(j) * NR0;
        float scale = args.mid_preweighted != 0u ? 1.0f : router_weights[uint(pid)];
        int i = int(tid & 31u);
        device float4 *dst4 = (device float4 *)dst;
        threadgroup float4 *src4 = (threadgroup float4 *)src;
        for (; i < nr0 / 4; i += 32) dst4[i] = src4[i] * scale;
        i = 4 * (nr0 / 4) + int(tid & 31u);
        for (; i < nr0; i += 32) dst[i] = src[i] * scale;
    }
}
kernel void qw3_moe_down_iq4_xs_prefill_mapped_mid_f32(constant qw3_moe_prefill_batch_args &args,
                                                         device const uchar *weights,
                                                         device const float *mid,
                                                         device float *down_slots,
                                                         device const uint *counts,
                                                         device const int *pair_ids,
                                                         device const float *router_weights,
                                                         device const uint *block_ids,
                                                         threadgroup char *shmem [[threadgroup(0)]],
                                                         uint3 group [[threadgroup_position_in_grid]],
                                                         ushort tid [[thread_index_in_threadgroup]],
                                                         ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;
    uint r0u = group.y * NR0;
    uint expert = group.z;
    uint r1u = group.x * NR1;
    if (args.compact_blocks != 0u) {
        uint block = block_ids[group.x];
        expert = block & 255u;
        r1u = block >> 8u;
    }
    uint count = counts[expert];
    if (r0u >= args.n_embd || r1u >= count) return;
    int nr0 = int(min(uint(NR0), args.n_embd - r0u));
    int nr1 = int(min(uint(NR1), count - r1u));
    int lr0 = min(int(tid) / NL0, nr0 - 1);
    int lr1 = min(int(tid) / NL1, nr1 - 1);
    short il0 = short(tid % NL0);
    uint row = r0u + uint(lr0);
    uint map_base = expert * args.n_tokens + r1u;
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint loop_k = 0u; loop_k < args.n_ff; loop_k += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const uchar *wrow = weights + uint64_t(expert) * uint64_t(args.down_expert_bytes) + uint64_t(row) * uint64_t(args.down_row_bytes);
        uint il_base = (loop_k & 255u) >> 4u;
        uint block_k = loop_k >> 8u;
        device const uchar *blk = wrow + uint64_t(block_k) * 136ull;
        float d = float(*((device const half *)blk));
        ushort scales_h = *((device const ushort *)(blk + 2u));
        device const uchar *scales_l = blk + 4u;
        uint il = il_base + uint(il0);
        uint ib32 = il >> 1u;
        uint half_il = il & 1u;
        uint ls = ((uint(scales_l[ib32 >> 1u]) >> (4u * (ib32 & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib32)) & 3u) << 4u);
        float dl = d * float(int(ls) - 32);
        device const uchar *q16 = scales_l + 4u + ib32 * 16u;
        bool hi = half_il != 0u;
        for (short i = 0; i < 16; i++) {
            uint k = loop_k + uint(16 * il0 + i);
            const short sx = short(2 * il0 + i / 8);
            const short sy = short((tid / NL0) / 8);
            const short lx = short((tid / NL0) % 8);
            const short ly = short(i % 8);
            const short ib = short(8 * sx + sy);
            *(sa + 64 * ib + 8 * ly + lx) = k < args.n_ff ? qw3_iq4xs_dequant16(dl, q16, hi, uint(i)) : half(0.0f);
        }
        int pid = pair_ids[map_base + uint(lr1)];
        device const float *hidden = mid + uint64_t(uint(pid)) * uint64_t(args.n_ff) + uint64_t(loop_k);
        for (short i = 0; i < 8; i++) {
            const short sx = short(tid % NL1);
            const short sy = short((tid / NL1) / 8);
            const short lx = i;
            const short ly = short((tid / NL1) % 8);
            const short ib = short(4 * sx + sy);
            uint kk = uint(8 * sx + i);
            *(sb + 64 * ib + 8 * ly + lx) = (uint(lr1) < uint(nr1) && loop_k + kk < args.n_ff) ? half(hidden[kk]) : half(0.0f);
        }
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *tmp = ((threadgroup float *)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
    for (short i = 0; i < 8; i++) simdgroup_store(mc[i], tmp + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short j = short(sgitg); j < nr1; j += 4) {
        int pid = pair_ids[map_base + uint(j)];
        device float *dst = down_slots + uint64_t(uint(pid)) * uint64_t(args.n_embd) + uint64_t(r0u);
        threadgroup float *src = ((threadgroup float *)shmem) + int(j) * NR0;
        float scale = router_weights[uint(pid)];
        int i = int(tid & 31u);
        device float4 *dst4 = (device float4 *)dst;
        threadgroup float4 *src4 = (threadgroup float4 *)src;
        for (; i < nr0 / 4; i += 32) dst4[i] = src4[i] * scale;
        i = 4 * (nr0 / 4) + int(tid & 31u);
        for (; i < nr0; i += 32) dst[i] = src[i] * scale;
    }
}
#ifdef QW3_METAL_HAS_TENSOR
kernel void qw3_moe_down_iq4_xs_prefill_mapped_mid_f32_mpp(constant qw3_moe_prefill_batch_args &args,
                                                             device const uchar *weights,
                                                             device const float *mid,
                                                             device float *down_slots,
                                                             device const uint *counts,
                                                             device const int *pair_ids,
                                                             device const float *router_weights,
                                                             device const uint *block_ids,
                                                             threadgroup char *shmem [[threadgroup(0)]],
                                                             uint3 group [[threadgroup_position_in_grid]],
                                                             ushort tid [[thread_index_in_threadgroup]],
                                                             ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);
    threadgroup float *sc = (threadgroup float *)shmem;
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;
    uint r0u = group.y * NR0;
    uint expert = group.z;
    uint r1u = group.x * NR1;
    if (args.compact_blocks != 0u) {
        uint block = block_ids[group.x];
        expert = block & 255u;
        r1u = block >> 8u;
    }
    uint count = counts[expert];
    if (r0u >= args.n_embd || r1u >= count) return;
    int nr0 = int(min(uint(NR0), args.n_embd - r0u));
    int nr1 = int(min(uint(NR1), count - r1u));
    int lr0 = min(int(tid) / NL0, nr0 - 1);
    int lr1 = min(int(tid) / NL1, nr1 - 1);
    short il0 = short(tid % NL0);
    uint row = r0u + uint(lr0);
    uint map_base = expert * args.n_tokens + r1u;
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
    for (uint loop_k = 0u; loop_k < args.n_ff; loop_k += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const uchar *wrow = weights + uint64_t(expert) * uint64_t(args.down_expert_bytes) + uint64_t(row) * uint64_t(args.down_row_bytes);
        uint il_base = (loop_k & 255u) >> 4u;
        uint block_k = loop_k >> 8u;
        device const uchar *blk = wrow + uint64_t(block_k) * 136ull;
        float d = float(*((device const half *)blk));
        ushort scales_h = *((device const ushort *)(blk + 2u));
        device const uchar *scales_l = blk + 4u;
        uint il = il_base + uint(il0);
        uint ib32 = il >> 1u;
        uint half_il = il & 1u;
        uint ls = ((uint(scales_l[ib32 >> 1u]) >> (4u * (ib32 & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib32)) & 3u) << 4u);
        float dl = d * float(int(ls) - 32);
        device const uchar *q16 = scales_l + 4u + ib32 * 16u;
        bool hi = half_il != 0u;
        half4x4 w4 = qw3_iq4xs_dequant4x4(dl, q16, hi);
        for (short i = 0; i < 16; i++) {
            const short sx = short(2 * il0 + i / 8);
            const short sy = short((tid / NL0) / 8);
            const short lx = short(i % 8);
            const short ly = short((tid / NL0) % 8);
            *(sa + NK * (8 * sy + ly) + 8 * sx + lx) = w4[i / 4][i & 3];
        }
        int pid = pair_ids[map_base + uint(lr1)];
        device const float *hidden = mid + uint64_t(uint(pid)) * uint64_t(args.n_ff) + uint64_t(loop_k);
        for (short i = 0; i < 8; i++) {
            const short sx = short(tid % NL1);
            const short sy = short((tid / NL1) / 8);
            const short lx = i;
            const short ly = short((tid / NL1) % 8);
            uint kk = uint(8 * sx + i);
            *(sb + NK * (8 * sy + ly) + 8 * sx + lx) = half(hidden[kk]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);
        mm.run(sB, sA, cT);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, NR1));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short j = short(sgitg); j < nr1; j += 4) {
        int pid = pair_ids[map_base + uint(j)];
        device float *dst = down_slots + uint64_t(uint(pid)) * uint64_t(args.n_embd) + uint64_t(r0u);
        threadgroup float *src = sc + int(j) * NR0;
        float scale = args.mid_preweighted != 0u ? 1.0f : router_weights[uint(pid)];
        int i = int(tid & 31u);
        device float4 *dst4 = (device float4 *)dst;
        threadgroup float4 *src4 = (threadgroup float4 *)src;
        for (; i < nr0 / 4; i += 32) dst4[i] = src4[i] * scale;
        i = 4 * (nr0 / 4) + int(tid & 31u);
        for (; i < nr0; i += 32) dst[i] = src[i] * scale;
    }
}
#endif
#ifdef QW3_METAL_HAS_TENSOR
kernel void qw3_moe_down_iq4_xs_prefill_mapped_f16_mpp(constant qw3_moe_prefill_batch_args &args,
                                                         device const uchar *weights,
                                                         device const half *mid,
                                                         device float *down_slots,
                                                         device const uint *counts,
                                                         device const int *pair_ids,
                                                         device const float *router_weights,
                                                         device const uint *block_ids,
                                                         threadgroup char *shmem [[threadgroup(0)]],
                                                         uint3 group [[threadgroup_position_in_grid]],
                                                         ushort tid [[thread_index_in_threadgroup]],
                                                         ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);
    threadgroup float *sc = (threadgroup float *)shmem;
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;
    uint r0u = group.y * NR0;
    uint expert = group.z;
    uint r1u = group.x * NR1;
    if (args.compact_blocks != 0u) {
        uint block = block_ids[group.x];
        expert = block & 255u;
        r1u = block >> 8u;
    }
    uint count = counts[expert];
    if (r0u >= args.n_embd || r1u >= count) return;
    int nr0 = int(min(uint(NR0), args.n_embd - r0u));
    int nr1 = int(min(uint(NR1), count - r1u));
    int lr0 = min(int(tid) / NL0, nr0 - 1);
    int lr1 = min(int(tid) / NL1, nr1 - 1);
    short il0 = short(tid % NL0);
    uint row = r0u + uint(lr0);
    uint map_base = expert * args.n_tokens + r1u;
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
    for (uint loop_k = 0u; loop_k < args.n_ff; loop_k += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const uchar *wrow = weights + uint64_t(expert) * uint64_t(args.down_expert_bytes) + uint64_t(row) * uint64_t(args.down_row_bytes);
        uint il_base = (loop_k & 255u) >> 4u;
        uint block_k = loop_k >> 8u;
        device const uchar *blk = wrow + uint64_t(block_k) * 136ull;
        float d = float(*((device const half *)blk));
        ushort scales_h = *((device const ushort *)(blk + 2u));
        device const uchar *scales_l = blk + 4u;
        uint il = il_base + uint(il0);
        uint ib32 = il >> 1u;
        uint half_il = il & 1u;
        uint ls = ((uint(scales_l[ib32 >> 1u]) >> (4u * (ib32 & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib32)) & 3u) << 4u);
        float dl = d * float(int(ls) - 32);
        device const uchar *q16 = scales_l + 4u + ib32 * 16u;
        bool hi = half_il != 0u;
        for (short i = 0; i < 16; i++) {
            const short sx = short(2 * il0 + i / 8);
            const short sy = short((tid / NL0) / 8);
            const short lx = short(i % 8);
            const short ly = short((tid / NL0) % 8);
            *(sa + NK * (8 * sy + ly) + 8 * sx + lx) = qw3_iq4xs_dequant16(dl, q16, hi, uint(i));
        }
        int pid = pair_ids[map_base + uint(lr1)];
        device const half *hidden = mid + uint64_t(uint(pid)) * uint64_t(args.n_ff) + uint64_t(loop_k);
        for (short i = 0; i < 8; i++) {
            const short sx = short(tid % NL1);
            const short sy = short((tid / NL1) / 8);
            const short lx = i;
            const short ly = short((tid / NL1) % 8);
            uint kk = uint(8 * sx + i);
            *(sb + NK * (8 * sy + ly) + 8 * sx + lx) = hidden[kk];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);
        mm.run(sB, sA, cT);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, NR1));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short j = short(sgitg); j < nr1; j += 4) {
        int pid = pair_ids[map_base + uint(j)];
        device float *dst = down_slots + uint64_t(uint(pid)) * uint64_t(args.n_embd) + uint64_t(r0u);
        threadgroup float *src = sc + int(j) * NR0;
        float scale = router_weights[uint(pid)];
        int i = int(tid & 31u);
        device float4 *dst4 = (device float4 *)dst;
        threadgroup float4 *src4 = (threadgroup float4 *)src;
        for (; i < nr0 / 4; i += 32) dst4[i] = src4[i] * scale;
        i = 4 * (nr0 / 4) + int(tid & 31u);
        for (; i < nr0; i += 32) dst[i] = src[i] * scale;
    }
}
#endif
kernel void qw3_moe_down_iq4_xs_prefill_mapped_f16(constant qw3_moe_prefill_batch_args &args,
                                                     device const uchar *weights,
                                                     device const half *mid,
                                                     device float *down_slots,
                                                     device const uint *counts,
                                                     device const int *pair_ids,
                                                     device const float *router_weights,
                                                     device const uint *block_ids,
                                                     threadgroup char *shmem [[threadgroup(0)]],
                                                     uint3 group [[threadgroup_position_in_grid]],
                                                     ushort tid [[thread_index_in_threadgroup]],
                                                     ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;
    uint r0u = group.y * NR0;
    uint expert = group.z;
    uint r1u = group.x * NR1;
    if (args.compact_blocks != 0u) {
        uint block = block_ids[group.x];
        expert = block & 255u;
        r1u = block >> 8u;
    }
    uint count = counts[expert];
    if (r0u >= args.n_embd || r1u >= count) return;
    int nr0 = int(min(uint(NR0), args.n_embd - r0u));
    int nr1 = int(min(uint(NR1), count - r1u));
    int lr0 = min(int(tid) / NL0, nr0 - 1);
    int lr1 = min(int(tid) / NL1, nr1 - 1);
    short il0 = short(tid % NL0);
    uint row = r0u + uint(lr0);
    uint map_base = expert * args.n_tokens + r1u;
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint loop_k = 0u; loop_k < args.n_ff; loop_k += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const uchar *wrow = weights + uint64_t(expert) * uint64_t(args.down_expert_bytes) + uint64_t(row) * uint64_t(args.down_row_bytes);
        uint il_base = (loop_k & 255u) >> 4u;
        uint block_k = loop_k >> 8u;
        device const uchar *blk = wrow + uint64_t(block_k) * 136ull;
        float d = float(*((device const half *)blk));
        ushort scales_h = *((device const ushort *)(blk + 2u));
        device const uchar *scales_l = blk + 4u;
        uint il = il_base + uint(il0);
        uint ib32 = il >> 1u;
        uint half_il = il & 1u;
        uint ls = ((uint(scales_l[ib32 >> 1u]) >> (4u * (ib32 & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib32)) & 3u) << 4u);
        float dl = d * float(int(ls) - 32);
        device const uchar *q16 = scales_l + 4u + ib32 * 16u;
        bool hi = half_il != 0u;
        for (short i = 0; i < 16; i++) {
            uint k = loop_k + uint(16 * il0 + i);
            const short sx = short(2 * il0 + i / 8);
            const short sy = short((tid / NL0) / 8);
            const short lx = short((tid / NL0) % 8);
            const short ly = short(i % 8);
            const short ib = short(8 * sx + sy);
            *(sa + 64 * ib + 8 * ly + lx) = k < args.n_ff ? qw3_iq4xs_dequant16(dl, q16, hi, uint(i)) : half(0.0f);
        }
        int pid = pair_ids[map_base + uint(lr1)];
        device const half *hidden = mid + uint64_t(uint(pid)) * uint64_t(args.n_ff) + uint64_t(loop_k);
        for (short i = 0; i < 8; i++) {
            const short sx = short(tid % NL1);
            const short sy = short((tid / NL1) / 8);
            const short lx = i;
            const short ly = short((tid / NL1) % 8);
            const short ib = short(4 * sx + sy);
            uint kk = uint(8 * sx + i);
            *(sb + 64 * ib + 8 * ly + lx) = (uint(lr1) < uint(nr1) && loop_k + kk < args.n_ff) ? hidden[kk] : half(0.0f);
        }
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *tmp = ((threadgroup float *)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
    for (short i = 0; i < 8; i++) simdgroup_store(mc[i], tmp + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short j = short(sgitg); j < nr1; j += 4) {
        int pid = pair_ids[map_base + uint(j)];
        device float *dst = down_slots + uint64_t(uint(pid)) * uint64_t(args.n_embd) + uint64_t(r0u);
        threadgroup float *src = ((threadgroup float *)shmem) + int(j) * NR0;
        float scale = router_weights[uint(pid)];
        int i = int(tid & 31u);
        device float4 *dst4 = (device float4 *)dst;
        threadgroup float4 *src4 = (threadgroup float4 *)src;
        for (; i < nr0 / 4; i += 32) dst4[i] = src4[i] * scale;
        i = 4 * (nr0 / 4) + int(tid & 31u);
        for (; i < nr0; i += 32) dst[i] = src[i] * scale;
    }
}
inline half qw3_q6k_dequant_k(device const uchar *row, uint k) {
    uint block = k >> 8u;
    uint local = k & 255u;
    device const uchar *blk = row + uint64_t(block) * 210ull;
    uint half_block = local >> 7u;
    uint rem = local & 127u;
    uint seg = rem >> 5u;
    uint l = rem & 31u;
    device const uchar *ql = blk + half_block * 64u;
    device const uchar *qh = blk + 128u + half_block * 32u;
    device const char *sc = (device const char *)(blk + 192u + half_block * 8u);
    ushort dbits = ushort(blk[208u]) | (ushort(blk[209u]) << 8u);
    float d = qw3_f16_to_f32(dbits);
    uint is = l >> 4u;
    int q = 0;
    int s = 0;
    if (seg == 0u) {
        q = int((uint(ql[l]) & 15u) | (((uint(qh[l]) >> 0u) & 3u) << 4u)) - 32;
        s = int(sc[is + 0u]);
    } else if (seg == 1u) {
        q = int((uint(ql[l + 32u]) & 15u) | (((uint(qh[l]) >> 2u) & 3u) << 4u)) - 32;
        s = int(sc[is + 2u]);
    } else if (seg == 2u) {
        q = int((uint(ql[l]) >> 4u) | (((uint(qh[l]) >> 4u) & 3u) << 4u)) - 32;
        s = int(sc[is + 4u]);
    } else {
        q = int((uint(ql[l + 32u]) >> 4u) | (((uint(qh[l]) >> 6u) & 3u) << 4u)) - 32;
        s = int(sc[is + 6u]);
    }
    return half(d * float(s) * float(q));
}
#ifdef QW3_METAL_HAS_TENSOR
kernel void qw3_moe_down_q6_k_prefill_mapped_mpp(constant qw3_moe_prefill_batch_args &args,
                                                     device const uchar *weights,
                                                     device const float *scratch,
                                                     device float *down_slots,
                                                     device const uint *counts,
                                                     device const int *pair_ids,
                                                     device const float *router_weights,
                                                     device const uint *block_ids,
                                                     threadgroup char *shmem [[threadgroup(0)]],
                                                     uint3 group [[threadgroup_position_in_grid]],
                                                     ushort tid [[thread_index_in_threadgroup]],
                                                     ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);
    threadgroup float *sc = (threadgroup float *)shmem;
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;
    uint r0u = group.y * NR0;
    uint expert = group.z;
    uint r1u = group.x * NR1;
    if (args.compact_blocks != 0u) {
        uint block = block_ids[group.x];
        expert = block & 255u;
        r1u = block >> 8u;
    }
    uint count = counts[expert];
    if (r0u >= args.n_embd || r1u >= count) return;
    int nr0 = int(min(uint(NR0), args.n_embd - r0u));
    int nr1 = int(min(uint(NR1), count - r1u));
    int lr0 = min(int(tid) / NL0, nr0 - 1);
    int lr1 = min(int(tid) / NL1, nr1 - 1);
    short il0 = short(tid % NL0);
    uint row = r0u + uint(lr0);
    uint map_base = expert * args.n_tokens + r1u;
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
    for (uint loop_k = 0u; loop_k < args.n_ff; loop_k += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const uchar *wrow = weights + uint64_t(expert) * uint64_t(args.down_expert_bytes) + uint64_t(row) * uint64_t(args.down_row_bytes);
        uint k_base = loop_k + uint(16 * il0);
        const short sy_a = short((tid / NL0) / 8);
        const short ly_a = short((tid / NL0) % 8);
        uint block = k_base >> 8u;
        uint local_base = k_base & 255u;
        device const uchar *blk = wrow + uint64_t(block) * 210ull;
        uint half_block = local_base >> 7u;
        uint rem_base = local_base & 127u;
        uint seg = rem_base >> 5u;
        uint l_base = rem_base & 31u;
        device const uchar *ql = blk + half_block * 64u;
        device const uchar *qh = blk + 128u + half_block * 32u;
        device const char *qsc = (device const char *)(blk + 192u + half_block * 8u);
        ushort dbits = ushort(blk[208u]) | (ushort(blk[209u]) << 8u);
        float dscale = qw3_f16_to_f32(dbits);
        uint is = l_base >> 4u;
        int scale = int(qsc[is + 2u * seg]);
        for (short i = 0; i < 16; i++) {
            uint l = l_base + uint(i);
            int q;
            if (seg == 0u) {
                q = int((uint(ql[l]) & 15u) | (((uint(qh[l]) >> 0u) & 3u) << 4u)) - 32;
            } else if (seg == 1u) {
                q = int((uint(ql[l + 32u]) & 15u) | (((uint(qh[l]) >> 2u) & 3u) << 4u)) - 32;
            } else if (seg == 2u) {
                q = int((uint(ql[l]) >> 4u) | (((uint(qh[l]) >> 4u) & 3u) << 4u)) - 32;
            } else {
                q = int((uint(ql[l + 32u]) >> 4u) | (((uint(qh[l]) >> 6u) & 3u) << 4u)) - 32;
            }
            const short sx = short(2 * il0 + i / 8);
            const short lx = short(i % 8);
            *(sa + NK * (8 * sy_a + ly_a) + 8 * sx + lx) = half(dscale * float(scale) * float(q));
        }
        int pid = pair_ids[map_base + uint(lr1)];
        uint token = uint(pid) / args.n_active;
        uint slot = uint(pid) - token * args.n_active;
        device const float *hidden = scratch + uint64_t(token) * uint64_t(args.stride) + uint64_t(args.hidden_offset) + uint64_t(slot) * uint64_t(args.n_ff) + uint64_t(loop_k);
        for (short i = 0; i < 8; i++) {
            const short sx = short(tid % NL1);
            const short sy = short((tid / NL1) / 8);
            const short lx = i;
            const short ly = short((tid / NL1) % 8);
            uint kk = uint(8 * sx + i);
            *(sb + NK * (8 * sy + ly) + 8 * sx + lx) = half(hidden[kk]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);
        mm.run(sB, sA, cT);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, NR1));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short j = short(sgitg); j < nr1; j += 4) {
        int pid = pair_ids[map_base + uint(j)];
        device float *dst = down_slots + uint64_t(uint(pid)) * uint64_t(args.n_embd) + uint64_t(r0u);
        threadgroup float *src = sc + int(j) * NR0;
        float scale = router_weights[uint(pid)];
        int i = int(tid & 31u);
        device float4 *dst4 = (device float4 *)dst;
        threadgroup float4 *src4 = (threadgroup float4 *)src;
        for (; i < nr0 / 4; i += 32) dst4[i] = src4[i] * scale;
        i = 4 * (nr0 / 4) + int(tid & 31u);
        for (; i < nr0; i += 32) dst[i] = src[i] * scale;
    }
}
#endif
kernel void qw3_moe_down_q6_k_prefill_mapped(constant qw3_moe_prefill_batch_args &args,
                                                device const uchar *weights,
                                                device const float *scratch,
                                                device float *down_slots,
                                                device const uint *counts,
                                                device const int *pair_ids,
                                                device const float *router_weights,
                                                device const uint *block_ids,
                                                threadgroup char *shmem [[threadgroup(0)]],
                                                uint3 group [[threadgroup_position_in_grid]],
                                                ushort tid [[thread_index_in_threadgroup]],
                                                ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;
    uint r0u = group.y * NR0;
    uint expert = group.z;
    uint r1u = group.x * NR1;
    if (args.compact_blocks != 0u) {
        uint block = block_ids[group.x];
        expert = block & 255u;
        r1u = block >> 8u;
    }
    uint count = counts[expert];
    if (r0u >= args.n_embd || r1u >= count) return;
    int nr0 = int(min(uint(NR0), args.n_embd - r0u));
    int nr1 = int(min(uint(NR1), count - r1u));
    int lr0 = min(int(tid) / NL0, nr0 - 1);
    int lr1 = min(int(tid) / NL1, nr1 - 1);
    short il0 = short(tid % NL0);
    uint row = r0u + uint(lr0);
    uint map_base = expert * args.n_tokens + r1u;
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint loop_k = 0u; loop_k < args.n_ff; loop_k += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const uchar *wrow = weights + uint64_t(expert) * uint64_t(args.down_expert_bytes) + uint64_t(row) * uint64_t(args.down_row_bytes);
        for (short i = 0; i < 16; i++) {
            uint k = loop_k + uint(16 * il0 + i);
            const short sx = short(2 * il0 + i / 8);
            const short sy = short((tid / NL0) / 8);
            const short lx = short((tid / NL0) % 8);
            const short ly = short(i % 8);
            const short ib = short(8 * sx + sy);
            *(sa + 64 * ib + 8 * ly + lx) = k < args.n_ff ? qw3_q6k_dequant_k(wrow, k) : half(0.0f);
        }
        int pid = pair_ids[map_base + uint(lr1)];
        uint token = uint(pid) / args.n_active;
        uint slot = uint(pid) - token * args.n_active;
        device const float *hidden = scratch + uint64_t(token) * uint64_t(args.stride) + uint64_t(args.hidden_offset) + uint64_t(slot) * uint64_t(args.n_ff) + uint64_t(loop_k);
        for (short i = 0; i < 8; i++) {
            const short sx = short(tid % NL1);
            const short sy = short((tid / NL1) / 8);
            const short lx = i;
            const short ly = short((tid / NL1) % 8);
            const short ib = short(4 * sx + sy);
            uint kk = uint(8 * sx + i);
            *(sb + 64 * ib + 8 * ly + lx) = (uint(lr1) < uint(nr1) && loop_k + kk < args.n_ff) ? half(hidden[kk]) : half(0.0f);
        }
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *tmp = ((threadgroup float *)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
    for (short i = 0; i < 8; i++) simdgroup_store(mc[i], tmp + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short j = short(sgitg); j < nr1; j += 4) {
        int pid = pair_ids[map_base + uint(j)];
        device float *dst = down_slots + uint64_t(uint(pid)) * uint64_t(args.n_embd) + uint64_t(r0u);
        threadgroup float *src = ((threadgroup float *)shmem) + int(j) * NR0;
        float scale = router_weights[uint(pid)];
        int i = int(tid & 31u);
        device float4 *dst4 = (device float4 *)dst;
        threadgroup float4 *src4 = (threadgroup float4 *)src;
        for (; i < nr0 / 4; i += 32) dst4[i] = src4[i] * scale;
        i = 4 * (nr0 / 4) + int(tid & 31u);
        for (; i < nr0; i += 32) dst[i] = src[i] * scale;
    }
}
struct qw3_moe_down_slot_reduce_args { uint n_tokens; uint n_active; uint n_embd; };
kernel void qw3_moe_down_prefill_reduce_slots(constant qw3_moe_down_slot_reduce_args &args,
                                             device const float *down_slots,
                                             device const float *router_weights,
                                             device float *x0,
                                             uint gid [[thread_position_in_grid]]) {
    uint total = args.n_tokens * args.n_embd;
    if (gid >= total) return;
    uint token = gid / args.n_embd;
    uint row = gid - token * args.n_embd;
    float sum = 0.0f;
    for (uint slot = 0u; slot < args.n_active; slot++) {
        uint pair = token * args.n_active + slot;
        sum += down_slots[uint64_t(pair) * uint64_t(args.n_embd) + row];
    }
    x0[uint64_t(token) * uint64_t(args.n_embd) + row] += sum;
}
kernel void qw3_matvec_iq3_s_expert_slot(constant qw3_expert_slot_args &args,
                                         device const uchar *weights,
                                         device const float *x,
                                         device float *out,
                                         device const ushort *kgrid,
                                         device const int *ids,
                                         threadgroup float *sh,
                                         uint row [[threadgroup_position_in_grid]],
                                         ushort tid [[thread_index_in_threadgroup]],
                                         ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                         ushort lane [[thread_index_in_simdgroup]],
                                         ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    uint expert = uint(ids[args.slot]);
    device const uchar *wr = weights + uint64_t(expert) * uint64_t(args.expert_bytes) + uint64_t(row) * uint64_t(args.row_bytes);
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
            uchar qh0 = qh[0]; uchar qh1 = qh[1];
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
            qs += 8; signs += 4;
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
            qh += 2; qs += 8; signs += 4;
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) sh[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane < 32 ? sh[lane] : 0.0f;
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}
kernel void qw3_matvec_iq4_xs_expert_slot(constant qw3_expert_slot_args &args,
                                           device const uchar *weights,
                                           device const float *x,
                                           device float *out,
                                           device const int *ids,
                                           threadgroup float *sh,
                                           uint row [[threadgroup_position_in_grid]],
                                           ushort tid [[thread_index_in_threadgroup]],
                                           ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                           ushort lane [[thread_index_in_simdgroup]],
                                           ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    uint expert = uint(ids[args.slot]);
    device const uchar *wr = weights + uint64_t(expert) * uint64_t(args.expert_bytes) + uint64_t(row) * uint64_t(args.row_bytes);
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
            uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);
            float dl = float(d) * float(int(ls) - 32);
            device const uchar *q = qs + ib * 16u;
            device const float *xg = xx + ib * 32u;
            for (uint j = 0; j < 16u; j++) { uchar v = q[j]; sum += dl * qw3_iq4nl_val(uint(v) & 15u) * xg[j]; sum += dl * qw3_iq4nl_val(uint(v) >> 4u) * xg[j + 16u]; }
        }
    }
    sum = simd_sum(sum); if (lane == 0) sh[simd_idx] = sum; threadgroup_barrier(mem_flags::mem_threadgroup); sum = lane < 32 ? sh[lane] : 0.0f; sum = simd_sum(sum); if (tid == 0) out[row] = sum;
}
kernel void qw3_matvec_iq4_xs_expert_slot_fast(constant qw3_expert_slot_args &args,
                                                device const uchar *weights,
                                                device const float *x,
                                                device float *out,
                                                device const int *ids,
                                                threadgroup float *sh,
                                                uint group [[threadgroup_position_in_grid]],
                                                ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                                ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    const uint nr0 = 2u;
    const uint nsg = 2u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    uint expert = uint(ids[args.slot]);
    uint64_t expert_off = uint64_t(expert) * uint64_t(args.expert_bytes);
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
            device const uchar *blk = weights + expert_off + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
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
            device const uchar *blk = weights + expert_off + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
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
        if (row < args.n_out) out[row] = sum0;
        row = first_row + 1u;
        if (row < args.n_out) out[row] = sum1;
    }
}
kernel void qw3_matvec_iq4_xs_expert_slot_add_x0_fast(constant qw3_expert_slot_args &args,
                                                       device const uchar *weights,
                                                       device const float *x,
                                                       device float *x0,
                                                       device const int *ids,
                                                       device const float *router_weights,
                                                       threadgroup float *sh,
                                                       uint group [[threadgroup_position_in_grid]],
                                                       ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                                       ushort lane [[thread_index_in_simdgroup]]) {
    (void)sh;
    const uint nr0 = 2u;
    const uint nsg = 2u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    uint expert = uint(ids[args.slot]);
    uint64_t expert_off = uint64_t(expert) * uint64_t(args.expert_bytes);
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
            device const uchar *blk = weights + expert_off + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
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
            device const uchar *blk = weights + expert_off + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;
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
    float scale = router_weights[args.slot];
    if (lane == 0) {
        uint row = first_row;
        if (row < args.n_out) x0[row] = x0[row] + sum0 * scale;
        row = first_row + 1u;
        if (row < args.n_out) x0[row] = x0[row] + sum1 * scale;
    }
}
kernel void qw3_matvec_q6_k_expert_slot(constant qw3_expert_slot_args &args,
                                         device const uchar *weights,
                                         device const float *x,
                                         device float *out,
                                         device const int *ids,
                                         threadgroup float *sh,
                                         uint row [[threadgroup_position_in_grid]],
                                         ushort tid [[thread_index_in_threadgroup]],
                                         ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                         ushort lane [[thread_index_in_simdgroup]],
                                         ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_out) return;
    uint expert = uint(ids[args.slot]);
    device const uchar *wr = weights + uint64_t(expert) * uint64_t(args.expert_bytes) + uint64_t(row) * uint64_t(args.row_bytes);
    float sum = 0.0f;
    uint n_blocks = args.n_in / 256u;
    for (uint b = tid; b < n_blocks; b += nt) {
        device const uchar *blk = wr + uint64_t(b) * 210ull;
        device const uchar *ql = blk; device const uchar *qh = ql + 128u; device const uchar *scb = qh + 64u; device const char *sc = (device const char *)scb;
        ushort dbits = ushort(scb[16u]) | (ushort(scb[17u]) << 8u); float d = qw3_f16_to_f32(dbits); device const float *xx = x + uint64_t(b) * 256ull;
        for (uint n = 0; n < 256u; n += 128u) {
            for (uint l = 0; l < 32u; l++) {
                uint is = l / 16u;
                int q1 = int((uint(ql[l +  0u]) & 15u) | (((uint(qh[l]) >> 0u) & 3u) << 4u)) - 32;
                int q2 = int((uint(ql[l + 32u]) & 15u) | (((uint(qh[l]) >> 2u) & 3u) << 4u)) - 32;
                int q3 = int((uint(ql[l +  0u]) >> 4u) | (((uint(qh[l]) >> 4u) & 3u) << 4u)) - 32;
                int q4 = int((uint(ql[l + 32u]) >> 4u) | (((uint(qh[l]) >> 6u) & 3u) << 4u)) - 32;
                sum += d * float(sc[is + 0u]) * float(q1) * xx[n + l +  0u]; sum += d * float(sc[is + 2u]) * float(q2) * xx[n + l + 32u]; sum += d * float(sc[is + 4u]) * float(q3) * xx[n + l + 64u]; sum += d * float(sc[is + 6u]) * float(q4) * xx[n + l + 96u];
            }
            ql += 64u; qh += 32u; sc += 8u;
        }
    }
    sum = simd_sum(sum); if (lane == 0) sh[simd_idx] = sum; threadgroup_barrier(mem_flags::mem_threadgroup); sum = lane < 32 ? sh[lane] : 0.0f; sum = simd_sum(sum); if (tid == 0) out[row] = sum;
}
struct qw3_argmax_args { uint n; };
kernel void qw3_argmax_blocks(constant qw3_argmax_args &args,
                              device const float *x,
                              device float *out_vals,
                              device uint *out_idxs,
                              threadgroup float *sh_vals,
                              threadgroup uint *sh_idxs,
                              uint block [[threadgroup_position_in_grid]],
                              ushort tid [[thread_index_in_threadgroup]],
                              ushort nt [[threads_per_threadgroup]]) {
    uint idx = block * uint(nt) + uint(tid);
    float v = idx < args.n ? x[idx] : -FLT_MAX;
    sh_vals[tid] = v;
    sh_idxs[tid] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = uint(nt) >> 1; stride > 0; stride >>= 1) {
        if (uint(tid) < stride) {
            float ov = sh_vals[tid + stride];
            uint oi = sh_idxs[tid + stride];
            float cv = sh_vals[tid];
            uint ci = sh_idxs[tid];
            if (ov > cv || (ov == cv && oi < ci)) {
                sh_vals[tid] = ov;
                sh_idxs[tid] = oi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out_vals[block] = sh_vals[0];
        out_idxs[block] = sh_idxs[0];
    }
}
