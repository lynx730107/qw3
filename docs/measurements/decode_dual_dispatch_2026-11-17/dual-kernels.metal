
inline void dual_run_iq3(constant qw3_matvec_q8_0_args &args,
    device const uchar *weights, device const float *x, device float *out,
    device const ushort *kgrid, threadgroup ushort *grid, uint group, ushort simd_idx, ushort lane) {
    const uint tid = uint(simd_idx) * 32u + uint(lane);
    for (uint i = tid; i < 512u; i += 64u) grid[i] = kgrid[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nr0 = 4u;
    const uint nsg = 2u;
    uint first_row = (group * nsg + uint(simd_idx)) * nr0;
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    uint nb32 = (args.n_in / 256u) * 8u;
    for (uint ib32 = uint(lane); ib32 < nb32; ib32 += 32u) {
        uint ibl = ib32 / 8u;
        uint ib = ib32 - ibl * 8u;
        float xx[32];
        for (uint j = 0; j < 32u; j++) xx[j] = x[uint64_t(ib32) * 32ull + j];
        uint64_t block_off = uint64_t(ibl) * 110ull;
        uint row = first_row;
        if (row < args.n_out)
            sum0 += qw3_iq3s_dot32_shared(weights +
                uint64_t(row) * uint64_t(args.row_bytes) + block_off,
                xx, grid, ib);
        row = first_row + 1u;
        if (row < args.n_out)
            sum1 += qw3_iq3s_dot32_shared(weights +
                uint64_t(row) * uint64_t(args.row_bytes) + block_off,
                xx, grid, ib);
        row = first_row + 2u;
        if (row < args.n_out)
            sum2 += qw3_iq3s_dot32_shared(weights +
                uint64_t(row) * uint64_t(args.row_bytes) + block_off,
                xx, grid, ib);
        row = first_row + 3u;
        if (row < args.n_out)
            sum3 += qw3_iq3s_dot32_shared(weights +
                uint64_t(row) * uint64_t(args.row_bytes) + block_off,
                xx, grid, ib);
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2);
    sum3 = simd_sum(sum3);
    if (lane == 0) {
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
inline void dual_run_iq4(constant qw3_matvec_q8_0_args &args,
    device const uchar *weights, device const float *x, device float *out,
    threadgroup float *sh, uint group, ushort simd_idx, ushort lane) {
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
struct dual_matvec_args {
    qw3_matvec_q8_0_args a, b;
    uint groups_a, type_a, type_b;
};

kernel void dual_iq3(constant dual_matvec_args &params [[buffer(0)]],
    device const uchar *wa [[buffer(1)]], device const uchar *wb [[buffer(2)]],
    device const float *x [[buffer(3)]], device float *out [[buffer(4)]],
    device const ushort *kgrid [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], ushort simd_idx [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup ushort grid[512];
    const bool first = group < params.groups_a;
    constant qw3_matvec_q8_0_args &args = first ? params.a : params.b;
    device const uchar *weights = first ? wa : wb;
    const uint local_group = first ? group : group - params.groups_a;
    device float *dst = first ? out : out + params.a.n_out;
    dual_run_iq3(args, weights, x, dst, kgrid, grid, local_group, simd_idx, lane);
}

kernel void dual_iq4(constant dual_matvec_args &params [[buffer(0)]],
    device const uchar *wa [[buffer(1)]], device const uchar *wb [[buffer(2)]],
    device const float *x [[buffer(3)]], device float *out [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]], ushort simd_idx [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[32];
    const bool first = group < params.groups_a;
    constant qw3_matvec_q8_0_args &args = first ? params.a : params.b;
    device const uchar *weights = first ? wa : wb;
    const uint local_group = first ? group : group - params.groups_a;
    device float *dst = first ? out : out + params.a.n_out;
    dual_run_iq4(args, weights, x, dst, sh, local_group, simd_idx, lane);
}

kernel void dual_mixed(constant dual_matvec_args &params [[buffer(0)]],
    device const uchar *wa [[buffer(1)]], device const uchar *wb [[buffer(2)]],
    device const float *x [[buffer(3)]], device float *out [[buffer(4)]],
    device const ushort *kgrid [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], ushort simd_idx [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup ushort grid[512];
    threadgroup float sh[32];
    const bool first = group < params.groups_a;
    constant qw3_matvec_q8_0_args &args = first ? params.a : params.b;
    device const uchar *weights = first ? wa : wb;
    const uint local_group = first ? group : group - params.groups_a;
    device float *dst = first ? out : out + params.a.n_out;
    const uint type = first ? params.type_a : params.type_b;
    if (type == 21u) dual_run_iq3(args, weights, x, dst, kgrid, grid, local_group, simd_idx, lane);
    else dual_run_iq4(args, weights, x, dst, sh, local_group, simd_idx, lane);
}
