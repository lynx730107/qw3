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
