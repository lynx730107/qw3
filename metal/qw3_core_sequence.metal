/* Copy raw FP32 storage without arithmetic, including NaN/zero payload bits. */
struct qw3_copy_rows_args {
    uint n, rows, src_stride, dst_stride, src_offset, dst_offset;
};
kernel void qw3_copy_rows(constant qw3_copy_rows_args &args [[buffer(0)]],
                          device const uint *src [[buffer(1)]],
                          device uint *dst [[buffer(2)]],
                          uint2 p [[thread_position_in_grid]]) {
    if (p.x >= args.n || p.y >= args.rows) return;
    dst[uint64_t(p.y) * args.dst_stride + args.dst_offset + p.x] =
        src[uint64_t(p.y) * args.src_stride + args.src_offset + p.x];
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
        uint half_dim = args.rope_dim / 2u;
        uint p = i % half_dim;
        float freq = pow(args.theta, -float(2u * p) / float(args.rope_dim));
        float ang = float(args.pos0 + t) * freq;
        float c = cos(ang);
        float s = sin(ang);
        float x0 = qh[p] * scale * w[p];
        float x1 = qh[p + half_dim] * scale * w[p + half_dim];
        yo[i] = i < half_dim ? (x0 * c - x1 * s) : (x0 * s + x1 * c);
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
        uint half_dim = args.rope_dim / 2u;
        uint p = i % half_dim;
        float freq = pow(args.theta, -float(2u * p) / float(args.rope_dim));
        float ang = float(args.pos0 + t) * freq;
        float c = cos(ang);
        float s = sin(ang);
        float x0 = kh[p] * scale * w[p];
        float x1 = kh[p + half_dim] * scale * w[p + half_dim];
        yo[i] = i < half_dim ? (x0 * c - x1 * s) : (x0 * s + x1 * c);
    }
}
struct qw3_rope_args { uint n_heads; uint head_dim; uint rope_dim; int pos; float theta; };
// Qwen IMRoPE's text positions are identical across sections; pairs are split-half.
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
    uint half_dim = args.rope_dim / 2u;
    uint p = i % half_dim;
    float freq = pow(args.theta, -float(2u * p) / float(args.rope_dim));
    float ang = float(args.pos) * freq;
    float c = cos(ang);
    float s = sin(ang);
    float x0 = xh[p];
    float x1 = xh[p + half_dim];
    yh[i] = i < half_dim ? (x0 * c - x1 * s) : (x0 * s + x1 * c);
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
    uint half_dim = args.rope_dim / 2u;
    uint p = i % half_dim;
    float freq = pow(args.theta, -float(2u * p) / float(args.rope_dim));
    float ang = float(args.pos0 + t) * freq;
    float c = cos(ang);
    float s = sin(ang);
    float x0 = xh[p];
    float x1 = xh[p + half_dim];
    yh[i] = i < half_dim ? (x0 * c - x1 * s) : (x0 * s + x1 * c);
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
                                        device half *tail,
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
        if (t + 32u >= args.n_tokens) {
            uint kvh = i / args.head_dim;
            uint d = i - kvh * args.head_dim;
            uint tail_row = pos % 32u;
            uint64_t plane = 32ull * args.n_kv_heads * kv_n;
            uint64_t tail_dst =
                (uint64_t(kvh) * 32ull + tail_row) * kv_n + d;
            tail[tail_dst] = half(kv);
            tail[plane + tail_dst] = half(vv);
        }
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
struct qw3_gqa_flash_decode_pad_args { uint n_ctx; uint n_kv_heads; uint head_dim; };
kernel void qw3_gqa_flash_decode_pad_f16(
        constant qw3_gqa_flash_decode_pad_args &args,
        device const half *k_cache,
        device const half *v_cache,
        device half *pad,
        uint group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint nt [[threads_per_threadgroup]]) {
    const uint C = 32u;
    const uint row = group % C;
    const uint kvh = group / C;
    if (kvh >= args.n_kv_heads) return;
    const uint valid = args.n_ctx % C;
    const uint src_row = args.n_ctx - valid + row;
    const uint kv_stride = args.n_kv_heads * args.head_dim;
    const uint64_t plane = uint64_t(C) * args.n_kv_heads * kv_stride;
    device half *kd = pad + (uint64_t(kvh) * C + row) * kv_stride;
    device half *vd = pad + plane + (uint64_t(kvh) * C + row) * kv_stride;
    for (uint d = tid; d < args.head_dim; d += nt) {
        uint64_t src = uint64_t(src_row) * kv_stride + uint64_t(kvh) * args.head_dim + d;
        kd[d] = row < valid ? k_cache[src] : half(0.0f);
        vd[d] = row < valid ? v_cache[src] : half(0.0f);
    }
}
struct qw3_gqa_flash_decode_gate_args { uint n; };
kernel void qw3_gqa_flash_decode_gate(
        constant qw3_gqa_flash_decode_gate_args &args,
        device float *attn,
        device const float *gate,
        uint i [[thread_position_in_grid]]) {
    if (i < args.n) attn[i] /= 1.0f + exp(-gate[i]);
}
struct qw3_gqa_flash_group_reduce_args {
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint n_partials;
    uint n_bulk;
};
kernel void qw3_gqa_flash_group_reduce(
        constant qw3_gqa_flash_group_reduce_args &args,
        device const float *partial,
        device const float *gate,
        device float *out,
        uint qh [[threadgroup_position_in_grid]],
        ushort i [[thread_index_in_threadgroup]]) {
    if (qh >= args.n_heads || uint(i) >= args.head_dim) return;
    const uint group_heads = args.n_heads / args.n_kv_heads;
    const uint kvh = qh / group_heads;
    const uint gh = qh - kvh * group_heads;
    const uint row = gh * args.n_kv_heads + kvh;
    const uint nrows = args.n_heads;
    float acc = 0.0f;
    float denom = 0.0f;
    float max_score = -FLT_MAX;
    for (uint split = 0; split < args.n_partials; split++) {
        uint out_index;
        uint stat_index;
        if (split < args.n_bulk) {
            out_index = (split * nrows + row) * args.head_dim + uint(i);
            stat_index = args.n_bulk * nrows * args.head_dim +
                         2u * (split * nrows + row);
        } else {
            const uint tail_base =
                args.n_bulk * nrows * (args.head_dim + 2u);
            const uint tail_stride = args.head_dim + 2u;
            out_index = tail_base + qh * tail_stride + uint(i);
            stat_index = tail_base + qh * tail_stride + args.head_dim;
        }
        const float local_denom = partial[stat_index + 0u];
        const float local_max = partial[stat_index + 1u];
        const float next_max = max(max_score, local_max);
        const float prev_scale = exp(max_score - next_max);
        const float cur_scale = exp(local_max - next_max);
        acc = acc * prev_scale +
              partial[out_index] * cur_scale;
        denom = denom * prev_scale + local_denom * cur_scale;
        max_score = next_max;
    }
    const uint dst = qh * args.head_dim + uint(i);
    const float sig = 1.0f / (1.0f + exp(-gate[dst]));
    out[dst] = (acc / denom) * sig;
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

/* Independent query-head partitions keep the original per-head reduction and time order. */
kernel void qw3_gqa_attend_n_inner_heads3(constant qw3_gqa_n_args &args,
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
    if (args.n_kv_heads == 0u) return;
    const uint full_heads = args.n_heads / args.n_kv_heads;
    if (full_heads == 0u || full_heads > 8u) return;
    const uint chunks = (full_heads + 2u) / 3u;
    const uint kvh = h / chunks;
    const uint head_chunk = h % chunks;
    if (kvh >= args.n_kv_heads || args.n_ctx == 0 || args.head_dim > uint(nt)) return;
    uint i = uint(tid);
    uint group_heads = min(3u, full_heads - head_chunk * 3u);
    if (group_heads == 0u || group_heads > 8u) return;
    uint first_qh = kvh * full_heads + head_chunk * 3u;
    float scale = rsqrt(float(args.head_dim));
    float qv[3];
    float max_score[3];
    float denom[3];
    float acc[3];
    for (uint gh = 0; gh < 3u; gh++) {
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
        for (uint gh = 0; gh < 3u; gh++) {
            float part = (gh < group_heads && i < args.head_dim) ? qv[gh] * kval : 0.0f;
            part = simd_sum(part);
            if (lane == 0) sh[gh * 8u + uint(simd_idx)] = part;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint gh = 0; gh < 3u; gh++) {
            float dot = (gh < group_heads && uint(tid) < n_simd) ? sh[gh * 8u + uint(tid)] : 0.0f;
            dot = simd_sum(dot);
            if (tid == 0) sh[gh] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float vv = (i < args.head_dim) ? qw3_gqa_cache_load(v_cache, v_cache_f16, kv_idx, args.kv_type) : 0.0f;
        for (uint gh = 0; gh < 3u; gh++) {
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
        for (uint gh = 0; gh < 3u; gh++) {
            uint qh = first_qh + gh;
            if (gh < group_heads && qh < args.n_heads) {
                uint gid = qh * args.head_dim + i;
                float sig = 1.0f / (1.0f + exp(-gate[gid]));
                out[gid] = (acc[gh] / denom[gh]) * sig;
            }
        }
    }
}
// Inkling 2 Mini uses 24 query heads, eight KV heads, and 128 channels per head.
constant uint inkling_head_width = 128u;
constant uint inkling_kv_width = 8u * inkling_head_width;
constant uint inkling_state_width = 2u * inkling_kv_width + 2u * 2304u;
constant uint inkling_partition_rows = 128u;

struct qw3_inkling_attention_args {
    uint pos, capacity, extent, partitions;
    uint q, k, v, gate, relative, sink, k_taps, v_taps;
    uint mixed, sliding;
    float qk_limit, relative_limit, tau;
};

static inline float qw3_inkling_short_conv(float value, float4 weights, float4 taps,
        bool dynamic, uint pos, device float *history, uint channel) {
    const float a = pos ? history[channel] : 0.0f;
    const float b = pos ? history[channel + inkling_state_width] : 0.0f;
    const float c = pos ? history[channel + 2u * inkling_state_width] : 0.0f;
    history[channel] = b;
    history[channel + inkling_state_width] = c;
    history[channel + 2u * inkling_state_width] = value;
    float residual = dot(float4(a, b, c, value), weights);
    if (dynamic) residual += dot(float4(value, c, b, a), taps);
    return (value + residual) * (dynamic ? (1.0f / 3.0f) : rsqrt(5.0f));
}

kernel void qw3_inkling_attention_prepare(constant qw3_inkling_attention_args &args [[buffer(0)]],
        device float *projected [[buffer(1)]], device half4 *cache [[buffer(2)]],
        device float *history [[buffer(3)]], device const float4 *k_conv [[buffer(4)]],
        device const float4 *v_conv [[buffer(5)]], device const float4 *q_clip [[buffer(6)]],
        device const float *relative_clip [[buffer(7)]],
        uint head [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (head < 24u) {
        const uint vec = 32u * head + lane;
        device float4 *q = (device float4 *)(projected + args.q);
        const float4 value = q[vec] * q_clip[vec];
        const float norm = sqrt(simd_sum(dot(value, value)));
        q[vec] = value * min(1.0f, args.qk_limit / (norm + 1.0e-6f));

        const uint ri = 16u * head + lane;
        const float relative = lane < 16u ? projected[args.relative + ri] * relative_clip[ri] : 0.0f;
        const float rel_norm = sqrt(simd_sum(relative * relative));
        if (lane < 16u) {
            projected[args.relative + ri] = relative * min(1.0f, args.relative_limit / (rel_norm + 1.0e-6f));
        }
    } else {
        const uint kv_head = head - 24u;
        const bool dynamic = !args.sliding;
        const uint start = kv_head * inkling_head_width + lane * 4u;
        const float4 kt = dynamic ? ((device const float4 *)(projected + args.k_taps))[kv_head] : float4(0);
        const float4 vt = dynamic ? ((device const float4 *)(projected + args.v_taps))[kv_head] : float4(0);
        float4 key, val;
        for (uint i = 0; i < 4u; i++) {
            const uint j = start + i;
            key[i] = qw3_inkling_short_conv(projected[args.k + j], k_conv[j], kt, dynamic,
                                            args.pos, history, j);
            val[i] = qw3_inkling_short_conv(projected[args.v + j], v_conv[j], vt, dynamic,
                                            args.pos, history, inkling_kv_width + j);
        }
        const float k_norm = sqrt(simd_sum(dot(key, key)));
        const float v_scale = rsqrt(simd_sum(dot(val, val)) / float(inkling_head_width) + 1.0e-6f);
        const uint slot = (args.pos % args.capacity) * (2u * inkling_kv_width / 4u) + start / 4u;
        cache[slot] = half4(key * min(1.0f, args.qk_limit / (k_norm + 1.0e-6f)));
        cache[slot + inkling_kv_width / 4u] = half4(val * v_scale);
    }
}

// One SIMD group per KV head (grouping all 3 GQA query heads) and partition.
// When partitions == 1, applies sink and gate directly to mixed (skipping merge).
kernel void qw3_inkling_attention_partition(constant qw3_inkling_attention_args &args [[buffer(0)]],
        device float *projected [[buffer(1)]], device const half4 *cache [[buffer(2)]],
        device const float *bias [[buffer(3)]], device float4 *partial [[buffer(4)]],
        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint kv_head = group.x, part = group.y;
    const uint h0 = kv_head * 3u;
    const uint h1 = h0 + 1u;
    const uint h2 = h0 + 2u;
    const uint rows = min(args.capacity, args.pos + 1u);
    const uint first = args.pos + 1u - rows;
    const uint begin = first + part * inkling_partition_rows;
    const uint end = min(args.pos + 1u, begin + inkling_partition_rows);
    const uint kv_vec = kv_head * 32u + lane;
    device const float4 *q_ptr = (device const float4 *)(projected + args.q);
    const float4 q0 = q_ptr[h0 * 32u + lane];
    const float4 q1 = q_ptr[h1 * 32u + lane];
    const float4 q2 = q_ptr[h2 * 32u + lane];
    const float rel0 = lane < 16u ? projected[args.relative + h0 * 16u + lane] : 0.0f;
    const float rel1 = lane < 16u ? projected[args.relative + h1 * 16u + lane] : 0.0f;
    const float rel2 = lane < 16u ? projected[args.relative + h2 * 16u + lane] : 0.0f;
    const bool single = (args.partitions == 1u);
    float max0 = single ? projected[args.sink + h0] : -3.402823e38f;
    float max1 = single ? projected[args.sink + h1] : -3.402823e38f;
    float max2 = single ? projected[args.sink + h2] : -3.402823e38f;
    float sum0 = single ? 1.0f : 0.0f;
    float sum1 = single ? 1.0f : 0.0f;
    float sum2 = single ? 1.0f : 0.0f;
    float4 val0 = float4(0.0f), val1 = float4(0.0f), val2 = float4(0.0f);
    const float inv_sqrt_d = rsqrt(float(inkling_head_width));
    uint slot = begin % args.capacity;
    for (uint j = begin; j < end; j++) {
        const uint offset = slot * (2u * inkling_kv_width / 4u) + kv_vec;
        const uint distance = args.pos - j;
        const float4 kv_k = float4(cache[offset]);
        float s0 = dot(q0, kv_k) * inv_sqrt_d;
        float s1 = dot(q1, kv_k) * inv_sqrt_d;
        float s2 = dot(q2, kv_k) * inv_sqrt_d;
        if (distance < args.extent && lane < 16u) {
            const float b = bias[distance * 16u + lane];
            s0 += rel0 * b;
            s1 += rel1 * b;
            s2 += rel2 * b;
        }
        s0 = simd_sum(s0) * args.tau;
        s1 = simd_sum(s1) * args.tau;
        s2 = simd_sum(s2) * args.tau;
        const float4 kv_v = float4(cache[offset + inkling_kv_width / 4u]);
        {
            const float next = max(max0, s0);
            const float keep = exp(max0 - next), add = exp(s0 - next);
            sum0 = sum0 * keep + add;
            val0 = val0 * keep + kv_v * add;
            max0 = next;
        }
        {
            const float next = max(max1, s1);
            const float keep = exp(max1 - next), add = exp(s1 - next);
            sum1 = sum1 * keep + add;
            val1 = val1 * keep + kv_v * add;
            max1 = next;
        }
        {
            const float next = max(max2, s2);
            const float keep = exp(max2 - next), add = exp(s2 - next);
            sum2 = sum2 * keep + add;
            val2 = val2 * keep + kv_v * add;
            max2 = next;
        }
        slot++;
        if (slot == args.capacity) slot = 0;
    }
    if (single) {
        device const float4 *gate_ptr = (device const float4 *)(projected + args.gate);
        device float4 *mixed_ptr = (device float4 *)(projected + args.mixed);
        const uint o0 = h0 * 32u + lane;
        const uint o1 = h1 * 32u + lane;
        const uint o2 = h2 * 32u + lane;
        mixed_ptr[o0] = val0 / (sum0 * (1.0f + exp(-gate_ptr[o0])));
        mixed_ptr[o1] = val1 / (sum1 * (1.0f + exp(-gate_ptr[o1])));
        mixed_ptr[o2] = val2 / (sum2 * (1.0f + exp(-gate_ptr[o2])));
        return;
    }
    device float4 *out0 = partial + (h0 * args.partitions + part) * 33u;
    device float4 *out1 = partial + (h1 * args.partitions + part) * 33u;
    device float4 *out2 = partial + (h2 * args.partitions + part) * 33u;
    out0[lane] = val0;
    out1[lane] = val1;
    out2[lane] = val2;
    if (lane == 0u) {
        out0[32] = float4(max0, sum0, 0.0f, 0.0f);
        out1[32] = float4(max1, sum1, 0.0f, 0.0f);
        out2[32] = float4(max2, sum2, 0.0f, 0.0f);
    }
}

kernel void qw3_inkling_attention_merge(constant qw3_inkling_attention_args &args [[buffer(0)]],
        device float *projected [[buffer(1)]], device const float4 *partial [[buffer(2)]],
        uint head [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    // A sink contributes to the denominator with a zero value.
    float maximum = projected[args.sink + head], sum = 1.0f;
    float4 value = float4(0.0f);
    for (uint i = 0; i < args.partitions; i++) {
        device const float4 *part = partial + (head * args.partitions + i) * 33u;
        const float2 stats = part[32].xy;
        const float next = max(maximum, stats.x);
        const float keep = exp(maximum - next), add = exp(stats.x - next);
        sum = sum * keep + stats.y * add;
        value = value * keep + part[lane] * add;
        maximum = next;
    }
    const uint offset = head * 32u + lane;
    const float4 gate = ((device const float4 *)(projected + args.gate))[offset];
    ((device float4 *)(projected + args.mixed))[offset] = value / (sum * (1.0f + exp(-gate)));
}

kernel void qw3_inkling_post_conv(constant uint4 &args [[buffer(0)]],
        device float *value [[buffer(1)]], device const float4 *weights [[buffer(2)]],
        device float *history [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= args.x) return;
    value[i] = qw3_inkling_short_conv(value[i], weights[i], float4(0.0f), false,
                                     args.y, history, args.z + i);
}

// Fused post-conv + optional residual add (args.w != 0 -> dst[i] += conv(src[i]), else dst[i] = conv(src[i])).
kernel void qw3_inkling_post_conv_add(constant uint4 &args [[buffer(0)]],
        device const float *src [[buffer(1)]], device float *dst [[buffer(2)]],
        device const float4 *weights [[buffer(3)]], device float *history [[buffer(4)]],
        uint i [[thread_position_in_grid]]) {
    if (i >= args.x) return;
    const float v = qw3_inkling_short_conv(src[i], weights[i], float4(0.0f), false,
                                           args.y, history, args.z + i);
    dst[i] = args.w ? (dst[i] + v) : v;
}

// Fused MoE top-8 weighted reduction + post-FFN short conv + residual add into cur.
kernel void qw3_inkling_moe_reduce_conv_add(constant uint4 &args [[buffer(0)]],
        device const float *down_weights [[buffer(1)]], device const float *down [[buffer(2)]],
        device const float4 *conv_weights [[buffer(3)]], device float *history [[buffer(4)]],
        device float *cur [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    const uint width = args.x;
    if (i >= width) return;
    float sum = 0.0f;
    for (uint e = 0u; e < 8u; e++) {
        sum += down_weights[e] * down[e * width + i];
    }
    const float v = qw3_inkling_short_conv(sum, conv_weights[i], float4(0.0f), false,
                                           args.y, history, args.z + i);
    cur[i] += v;
}

kernel void qw3_inkling_embed(constant uint2 &args [[buffer(0)]],
        device const uchar *weights [[buffer(1)]], device float *out [[buffer(2)]],
        uint i [[thread_position_in_grid]]) {
    if (i >= args.x) return;
    switch (args.y) {
    case 0u: out[i] = ((device const float *)weights)[i]; return;
    case 1u: out[i] = float(((device const half *)weights)[i]); return;
    case 8u: {
        device const uchar *block = weights + (i / 32u) * 34u;
        out[i] = float(*(device const half *)block) * float(((device const char *)(block + 2))[i % 32u]);
        return;
    }
    case 10u: out[i] = qw3_embed_q2_k_value(weights, i); return;
    case 11u: out[i] = qw3_embed_q3_k_value(weights, i); return;
    case 14u: {
        device const uchar *block = weights + (i / 256u) * 210u;
        const uint sub = (i / 128u) % 2u, group = (i / 32u) % 4u, lane = i % 32u;
        const uint lower = (block[sub * 64u + (group % 2u) * 32u + lane] >> (4u * (group / 2u))) & 15u;
        const uint upper = (block[128u + sub * 32u + lane] >> (2u * group)) & 3u;
        const float scale = ((device const char *)block)[192u + (i % 256u) / 16u];
        const float d = float(*(device const half *)(block + 208));
        out[i] = d * scale * float(int(lower | (upper << 4u)) - 32);
        return;
    }
    case 20u: {
        device const uchar *block = weights + (i / 32u) * 18u;
        const uint j = i % 32u;
        const uint code = (block[2u + (j % 16u)] >> (4u * (j / 16u))) & 15u;
        out[i] = float(*(device const half *)block) * qw3_iq4nl_table[code];
        return;
    }
    case 30u: out[i] = as_type<float>(uint(((device const ushort *)weights)[i]) << 16u); return;
    default: break;
    }
    const bool q5 = args.y == 13u;
    device const uchar *block = weights + (i / 256u) * (q5 ? 176u : 144u);
    const uint j = i % 256u;
    const uint group = j / 32u;
    device const uchar *s = block + 4u;
    const uint scale = group < 4u ? s[group] & 63u :
        (s[group + 4u] & 15u) | ((s[group - 4u] >> 6u) << 4u);
    const uint minimum = group < 4u ? s[group + 4u] & 63u :
        (s[group + 4u] >> 4u) | ((s[group] >> 6u) << 4u);
    const uint packed = block[(q5 ? 48u : 16u) + (group / 2u) * 32u + j % 32u];
    const uint high = q5 ? ((block[16u + j % 32u] >> group) & 1u) * 16u : 0u;
    const uint code = ((packed >> (4u * (group % 2u))) & 15u) + high;
    device const half *d = (device const half *)block;
    out[i] = float(d[0]) * float(scale) * float(code) - float(d[1]) * float(minimum);
}

kernel void qw3_inkling_mix(constant uint &banks [[buffer(0)]],
        device const float4 *query [[buffer(1)]], device const float4 *cur [[buffer(2)]],
        device const float4 *history [[buffer(3)]], device float4 *out [[buffer(4)]],
        ushort simd [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr uint width = 2304u / 4u;
    threadgroup float scores[5], scales[5], coefficients[5];
    for (uint bank = simd; bank <= banks; bank += 4u) {
        device const float4 *src = bank == banks ? cur : history + bank * width;
        float energy = 0.0f, dot_query = 0.0f;
        for (uint i = lane; i < width; i += 32u) {
            const float4 x = src[i];
            energy += dot(x, x);
            dot_query += dot(x, query[i]);
        }
        energy = simd_sum(energy);
        dot_query = simd_sum(dot_query);
        if (lane == 0u) {
            scales[bank] = rsqrt(energy / 2304.0f + 1.0e-6f);
            scores[bank] = dot_query * scales[bank] * rsqrt(2304.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float max_score = scores[banks];
        for (uint i = 0; i < banks; i++) max_score = max(max_score, scores[i]);
        float denom = 0.0f;
        for (uint i = 0; i <= banks; i++) {
            const float weight = exp(scores[i] - max_score);
            coefficients[i] = weight * scales[i];
            denom += weight;
        }
        for (uint i = 0; i <= banks; i++) coefficients[i] /= denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < width; i += 128u) {
        float4 sum = cur[i] * coefficients[banks];
        for (uint j = 0; j < banks; j++) sum += history[j * width + i] * coefficients[j];
        out[i] = sum;
    }
}

// Fused residual mix + RMSNorm with weight: keeps mixed float4 values in registers (5/thread),
// eliminating 57 norm dispatches and 57 vector roundtrips per token!
kernel void qw3_inkling_mix_norm(constant uint &banks [[buffer(0)]],
        device const float4 *query [[buffer(1)]], device const float4 *norm_w [[buffer(2)]],
        device const float4 *cur [[buffer(3)]], device const float4 *history [[buffer(4)]],
        device float4 *out [[buffer(5)]],
        ushort simd [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr uint width = 2304u / 4u;
    threadgroup float scores[5], scales[5], coefficients[5], sh_rms[4];
    for (uint bank = simd; bank <= banks; bank += 4u) {
        device const float4 *src = bank == banks ? cur : history + bank * width;
        float energy = 0.0f, dot_query = 0.0f;
        for (uint i = lane; i < width; i += 32u) {
            const float4 x = src[i];
            energy += dot(x, x);
            dot_query += dot(x, query[i]);
        }
        energy = simd_sum(energy);
        dot_query = simd_sum(dot_query);
        if (lane == 0u) {
            scales[bank] = rsqrt(energy / 2304.0f + 1.0e-6f);
            scores[bank] = dot_query * scales[bank] * rsqrt(2304.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float max_score = scores[banks];
        for (uint i = 0; i < banks; i++) max_score = max(max_score, scores[i]);
        float denom = 0.0f;
        for (uint i = 0; i <= banks; i++) {
            const float weight = exp(scores[i] - max_score);
            coefficients[i] = weight * scales[i];
            denom += weight;
        }
        for (uint i = 0; i <= banks; i++) coefficients[i] /= denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 vals[5];
    float ss = 0.0f;
    for (uint k = 0u; k < 5u; k++) {
        const uint i = uint(tid) + k * 128u;
        if (i < width) {
            float4 sum = cur[i] * coefficients[banks];
            for (uint j = 0; j < banks; j++) sum += history[j * width + i] * coefficients[j];
            vals[k] = sum;
            ss += dot(sum, sum);
        }
    }
    ss = simd_sum(ss);
    if (lane == 0u) sh_rms[simd] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        float tot = lane < 4u ? sh_rms[lane] : 0.0f;
        tot = simd_sum(tot);
        if (lane == 0u) sh_rms[0] = rsqrt(tot / 2304.0f + 1.0e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = sh_rms[0];
    for (uint k = 0u; k < 5u; k++) {
        const uint i = uint(tid) + k * 128u;
        if (i < width) out[i] = vals[k] * (inv_rms * norm_w[i]);
    }
}

struct qw3_inkling_mix4_batch_args { uint banks, n_tokens, slot_stride_vec4; };
kernel void qw3_inkling_mix_norm_batch(constant qw3_inkling_mix4_batch_args &args [[buffer(0)]],
        device const float4 *query [[buffer(1)]], device const float4 *norm_w [[buffer(2)]],
        device const float4 *cur [[buffer(3)]], device const float4 *history [[buffer(4)]],
        device float4 *out [[buffer(5)]],
        uint t [[threadgroup_position_in_grid]],
        ushort simd [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
        ushort tid [[thread_index_in_threadgroup]]) {
    if (t >= args.n_tokens) return;
    constexpr uint width = 2304u / 4u;
    const uint banks = args.banks;
    const uint64_t t_off = uint64_t(t) * width;
    device const float4 *cur_t = cur + t_off;
    device float4 *out_t = out + t_off;
    threadgroup float scores[5], scales[5], coefficients[5], sh_rms[4];
    for (uint bank = simd; bank <= banks; bank += 4u) {
        device const float4 *src = bank == banks ? cur_t :
            history + uint64_t(bank) * args.slot_stride_vec4 + t_off;
        float energy = 0.0f, dot_query = 0.0f;
        for (uint i = lane; i < width; i += 32u) {
            const float4 x = src[i];
            energy += dot(x, x);
            dot_query += dot(x, query[i]);
        }
        energy = simd_sum(energy);
        dot_query = simd_sum(dot_query);
        if (lane == 0u) {
            scales[bank] = rsqrt(energy / 2304.0f + 1.0e-6f);
            scores[bank] = dot_query * scales[bank] * rsqrt(2304.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float max_score = scores[banks];
        for (uint i = 0; i < banks; i++) max_score = max(max_score, scores[i]);
        float denom = 0.0f;
        for (uint i = 0; i <= banks; i++) {
            const float weight = exp(scores[i] - max_score);
            coefficients[i] = weight * scales[i];
            denom += weight;
        }
        for (uint i = 0; i <= banks; i++) coefficients[i] /= denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 vals[5];
    float ss = 0.0f;
    const uint64_t stride = uint64_t(args.slot_stride_vec4);
    for (uint k = 0u; k < 5u; k++) {
        const uint i = uint(tid) + k * 128u;
        if (i < width) {
            float4 sum = cur_t[i] * coefficients[banks];
            for (uint j = 0; j < banks; j++) sum += history[uint64_t(j) * stride + t_off + i] * coefficients[j];
            vals[k] = sum;
            ss += dot(sum, sum);
        }
    }
    ss = simd_sum(ss);
    if (lane == 0u) sh_rms[simd] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        float tot = lane < 4u ? sh_rms[lane] : 0.0f;
        tot = simd_sum(tot);
        if (lane == 0u) sh_rms[0] = rsqrt(tot / 2304.0f + 1.0e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = sh_rms[0];
    for (uint k = 0u; k < 5u; k++) {
        const uint i = uint(tid) + k * 128u;
        if (i < width) out_t[i] = vals[k] * (inv_rms * norm_w[i]);
    }
}

// Fused 4-query residual mix for repeat-visit messages (reads cur & history ONCE for all 4 queries).
// q_attn has queries [0] and [3] at rows 0 and 1; q_kv has queries [1] and [2] at rows 0 and 1.
kernel void qw3_inkling_mix4(constant uint &banks [[buffer(0)]],
        device const float4 *q_attn [[buffer(1)]], device const float4 *q_kv [[buffer(2)]],
        device const float4 *cur [[buffer(3)]], device const float4 *history [[buffer(4)]],
        device float4 *out [[buffer(5)]],
        ushort simd [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr uint width = 2304u / 4u;
    threadgroup float scores[4][5], scales[5], coeffs[4][5];
    for (uint bank = simd; bank <= banks; bank += 4u) {
        device const float4 *src = bank == banks ? cur : history + bank * width;
        float energy = 0.0f, d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        for (uint i = lane; i < width; i += 32u) {
            const float4 x = src[i];
            energy += dot(x, x);
            d0 += dot(x, q_attn[i]);
            d1 += dot(x, q_kv[i]);
            d2 += dot(x, q_kv[width + i]);
            d3 += dot(x, q_attn[width + i]);
        }
        energy = simd_sum(energy);
        d0 = simd_sum(d0); d1 = simd_sum(d1); d2 = simd_sum(d2); d3 = simd_sum(d3);
        if (lane == 0u) {
            const float sc = rsqrt(energy / 2304.0f + 1.0e-6f);
            const float factor = sc * rsqrt(2304.0f);
            scales[bank] = sc;
            scores[0][bank] = d0 * factor;
            scores[1][bank] = d1 * factor;
            scores[2][bank] = d2 * factor;
            scores[3][bank] = d3 * factor;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4u) {
        const uint q = uint(tid);
        float max_score = scores[q][banks];
        for (uint i = 0; i < banks; i++) max_score = max(max_score, scores[q][i]);
        float denom = 0.0f;
        for (uint i = 0; i <= banks; i++) {
            const float w = exp(scores[q][i] - max_score);
            coeffs[q][i] = w * scales[i];
            denom += w;
        }
        for (uint i = 0; i <= banks; i++) coeffs[q][i] /= denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < width; i += 128u) {
        const float4 c = cur[i];
        float4 s0 = c * coeffs[0][banks];
        float4 s1 = c * coeffs[1][banks];
        float4 s2 = c * coeffs[2][banks];
        float4 s3 = c * coeffs[3][banks];
        for (uint j = 0; j < banks; j++) {
            const float4 h = history[j * width + i];
            s0 += h * coeffs[0][j];
            s1 += h * coeffs[1][j];
            s2 += h * coeffs[2][j];
            s3 += h * coeffs[3][j];
        }
        out[i] = s0;
        out[width + i] = s1;
        out[2u * width + i] = s2;
        out[3u * width + i] = s3;
    }
}

kernel void qw3_inkling_mix4_batch(constant qw3_inkling_mix4_batch_args &args [[buffer(0)]],
        device const float4 *q_attn [[buffer(1)]], device const float4 *q_kv [[buffer(2)]],
        device const float4 *cur [[buffer(3)]], device const float4 *history [[buffer(4)]],
        device float4 *out [[buffer(5)]],
        uint t [[threadgroup_position_in_grid]],
        ushort simd [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
        ushort tid [[thread_index_in_threadgroup]]) {
    if (t >= args.n_tokens) return;
    constexpr uint width = 2304u / 4u;
    const uint banks = args.banks;
    const uint64_t t_off = uint64_t(t) * width;
    device const float4 *cur_t = cur + t_off;
    threadgroup float scores[4][5], scales[5], coeffs[4][5];
    for (uint bank = simd; bank <= banks; bank += 4u) {
        device const float4 *src = bank == banks ? cur_t :
            history + uint64_t(bank) * args.slot_stride_vec4 + t_off;
        float energy = 0.0f, d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        for (uint i = lane; i < width; i += 32u) {
            const float4 x = src[i];
            energy += dot(x, x);
            d0 += dot(x, q_attn[i]);
            d1 += dot(x, q_kv[i]);
            d2 += dot(x, q_kv[width + i]);
            d3 += dot(x, q_attn[width + i]);
        }
        energy = simd_sum(energy);
        d0 = simd_sum(d0); d1 = simd_sum(d1); d2 = simd_sum(d2); d3 = simd_sum(d3);
        if (lane == 0u) {
            const float sc = rsqrt(energy / 2304.0f + 1.0e-6f);
            const float factor = sc * rsqrt(2304.0f);
            scales[bank] = sc;
            scores[0][bank] = d0 * factor;
            scores[1][bank] = d1 * factor;
            scores[2][bank] = d2 * factor;
            scores[3][bank] = d3 * factor;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4u) {
        const uint q = uint(tid);
        float max_score = scores[q][banks];
        for (uint i = 0; i < banks; i++) max_score = max(max_score, scores[q][i]);
        float denom = 0.0f;
        for (uint i = 0; i <= banks; i++) {
            const float w = exp(scores[q][i] - max_score);
            coeffs[q][i] = w * scales[i];
            denom += w;
        }
        for (uint i = 0; i <= banks; i++) coeffs[q][i] /= denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint64_t stride = uint64_t(args.slot_stride_vec4);
    for (uint i = tid; i < width; i += 128u) {
        const float4 c = cur_t[i];
        float4 s0 = c * coeffs[0][banks];
        float4 s1 = c * coeffs[1][banks];
        float4 s2 = c * coeffs[2][banks];
        float4 s3 = c * coeffs[3][banks];
        for (uint j = 0; j < banks; j++) {
            const float4 h = history[uint64_t(j) * stride + t_off + i];
            s0 += h * coeffs[0][j];
            s1 += h * coeffs[1][j];
            s2 += h * coeffs[2][j];
            s3 += h * coeffs[3][j];
        }
        out[t_off + i] = s0;
        out[stride + t_off + i] = s1;
        out[2u * stride + t_off + i] = s2;
        out[3u * stride + t_off + i] = s3;
    }
}

kernel void qw3_inkling_vector_op(constant uint2 &args [[buffer(0)]],
        device float *dst [[buffer(1)]], device const float *src [[buffer(2)]],
        uint i [[thread_position_in_grid]]) {
    if (i < args.x) dst[i] = args.y ? dst[i] + src[i] : src[i];
}

struct qw3_inkling_route_args { float scale; uint flags; };
kernel void qw3_inkling_route(constant qw3_inkling_route_args &args [[buffer(0)]],
        device const float *router [[buffer(1)]], device const float *bias [[buffer(2)]],
        device const float *gate_scale [[buffer(3)]], device const float *down_scale [[buffer(4)]],
        device uint *ids [[buffer(5)]], device float *out [[buffer(6)]],
        ushort i [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float probabilities[128], sh_max[4];
    threadgroup uint sh_idx[4], sh_sel[8];
    const float prob = 1.0f / (1.0f + exp(-router[i]));
    probabilities[i] = prob;
    float my_score = prob + ((args.flags & 1u) ? bias[i] : 0.0f);
    for (uint j = 0u; j < 8u; j++) {
        const float sm = simd_max(my_score);
        const uint si = simd_min(my_score == sm ? uint(i) : 128u);
        if (lane == 0u) { sh_max[sg] = sm; sh_idx[sg] = si; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (i == 0u) {
            float best_s = sh_max[0];
            uint best_i = sh_idx[0];
            for (uint s = 1u; s < 4u; s++) {
                if (sh_max[s] > best_s) { best_s = sh_max[s]; best_i = sh_idx[s]; }
            }
            sh_sel[j] = best_i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (uint(i) == sh_sel[j]) my_score = -3.402823e38f;
    }
    if (i < 8u) {
        float sum = 1.0e-20f;
        for (uint j = 0u; j < 8u; j++) sum += probabilities[sh_sel[j]];
        const uint id = sh_sel[i];
        ids[i] = id;
        out[i] = (args.flags & 2u) ? gate_scale[id] : 1.0f;
        out[8u + i] = (probabilities[id] / sum) * args.scale *
                      ((args.flags & 4u) ? down_scale[id] : 1.0f);
    }
}

struct qw3_inkling_logit_args { uint n; float cap; };
kernel void qw3_inkling_logit_cap(constant qw3_inkling_logit_args &args [[buffer(0)]],
        device float *logits [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < args.n) logits[i] = args.cap * tanh(logits[i] / args.cap);
}

// Batched Inkling prefill kernels: embedding, mix, routing, causal convolutions, and tiled attention.
struct qw3_inkling_embed_batch_args { uint n_embd, type, row_bytes, n_tokens; };
kernel void qw3_inkling_embed_batch(constant qw3_inkling_embed_batch_args &args [[buffer(0)]],
        device const uchar *weights [[buffer(1)]], device const uint *tokens [[buffer(2)]],
        device float *out [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, t = gid.y;
    if (i >= args.n_embd || t >= args.n_tokens) return;
    device const uchar *row = weights + uint64_t(tokens[t]) * args.row_bytes;
    device float *dst = out + uint64_t(t) * args.n_embd;
    switch (args.type) {
    case 0u: dst[i] = ((device const float *)row)[i]; return;
    case 1u: dst[i] = float(((device const half *)row)[i]); return;
    case 8u: {
        device const uchar *block = row + (i / 32u) * 34u;
        dst[i] = float(*(device const half *)block) * float(((device const char *)(block + 2))[i % 32u]);
        return;
    }
    case 10u: dst[i] = qw3_embed_q2_k_value(row, i); return;
    case 11u: dst[i] = qw3_embed_q3_k_value(row, i); return;
    case 14u: {
        device const uchar *block = row + (i / 256u) * 210u;
        const uint sub = (i / 128u) % 2u, group = (i / 32u) % 4u, lane = i % 32u;
        const uint lower = (block[sub * 64u + (group % 2u) * 32u + lane] >> (4u * (group / 2u))) & 15u;
        const uint upper = (block[128u + sub * 32u + lane] >> (2u * group)) & 3u;
        const float scale = ((device const char *)block)[192u + (i % 256u) / 16u];
        const float d = float(*(device const half *)(block + 208));
        dst[i] = d * scale * float(int(lower | (upper << 4u)) - 32);
        return;
    }
    case 20u: {
        device const uchar *block = row + (i / 32u) * 18u;
        const uint j = i % 32u;
        const uint code = (block[2u + (j % 16u)] >> (4u * (j / 16u))) & 15u;
        dst[i] = float(*(device const half *)block) * qw3_iq4nl_table[code];
        return;
    }
    case 30u: dst[i] = as_type<float>(uint(((device const ushort *)row)[i]) << 16u); return;
    default: break;
    }
    const bool q5 = args.type == 13u;
    device const uchar *block = row + (i / 256u) * (q5 ? 176u : 144u);
    const uint j = i % 256u;
    const uint group = j / 32u;
    device const uchar *s = block + 4u;
    const uint scale = group < 4u ? s[group] & 63u :
        (s[group + 4u] & 15u) | ((s[group - 4u] >> 6u) << 4u);
    const uint minimum = group < 4u ? s[group + 4u] & 63u :
        (s[group + 4u] >> 4u) | ((s[group] >> 6u) << 4u);
    const uint packed = block[(q5 ? 48u : 16u) + (group / 2u) * 32u + j % 32u];
    const uint high = q5 ? ((block[16u + j % 32u] >> group) & 1u) * 16u : 0u;
    const uint code = ((packed >> (4u * (group % 2u))) & 15u) + high;
    device const half *d = (device const half *)block;
    dst[i] = float(d[0]) * float(scale) * float(code) - float(d[1]) * float(minimum);
}

struct qw3_inkling_mix_batch_args { uint banks, n_tokens, slot_stride_vec4; };
kernel void qw3_inkling_mix_batch(constant qw3_inkling_mix_batch_args &args [[buffer(0)]],
        device const float4 *query [[buffer(1)]], device const float4 *cur [[buffer(2)]],
        device const float4 *history [[buffer(3)]], device float4 *out [[buffer(4)]],
        uint t [[threadgroup_position_in_grid]],
        ushort simd [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
        ushort tid [[thread_index_in_threadgroup]]) {
    if (t >= args.n_tokens) return;
    constexpr uint width = 2304u / 4u;
    const uint banks = args.banks;
    device const float4 *cur_t = cur + uint64_t(t) * width;
    device float4 *out_t = out + uint64_t(t) * width;
    threadgroup float scores[5], scales[5], coefficients[5];
    for (uint bank = simd; bank <= banks; bank += 4u) {
        device const float4 *src = bank == banks ? cur_t :
            history + uint64_t(bank) * args.slot_stride_vec4 + uint64_t(t) * width;
        float energy = 0.0f, dot_query = 0.0f;
        for (uint i = lane; i < width; i += 32u) {
            const float4 x = src[i];
            energy += dot(x, x);
            dot_query += dot(x, query[i]);
        }
        energy = simd_sum(energy);
        dot_query = simd_sum(dot_query);
        if (lane == 0u) {
            scales[bank] = rsqrt(energy / 2304.0f + 1.0e-6f);
            scores[bank] = dot_query * scales[bank] * rsqrt(2304.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float max_score = scores[banks];
        for (uint i = 0; i < banks; i++) max_score = max(max_score, scores[i]);
        float denom = 0.0f;
        for (uint i = 0; i <= banks; i++) {
            const float weight = exp(scores[i] - max_score);
            coefficients[i] = weight * scales[i];
            denom += weight;
        }
        for (uint i = 0; i <= banks; i++) coefficients[i] /= denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < width; i += 128u) {
        float4 sum = cur_t[i] * coefficients[banks];
        for (uint j = 0; j < banks; j++)
            sum += history[uint64_t(j) * args.slot_stride_vec4 + uint64_t(t) * width + i] * coefficients[j];
        out_t[i] = sum;
    }
}

struct qw3_inkling_route_batch_args { float scale; uint flags, n_tokens; };
kernel void qw3_inkling_route_batch(constant qw3_inkling_route_batch_args &args [[buffer(0)]],
        device const float *router [[buffer(1)]], device const float *bias [[buffer(2)]],
        device const float *gate_scale [[buffer(3)]], device const float *down_scale [[buffer(4)]],
        device int *ids [[buffer(5)]], device float *gate_out [[buffer(6)]],
        device float *down_out [[buffer(7)]],
        uint t [[threadgroup_position_in_grid]],
        ushort i [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    if (t >= args.n_tokens) return;
    threadgroup float probabilities[128], sh_max[4];
    threadgroup uint sh_idx[4], sh_sel[8];
    const float r = router[uint64_t(t) * 128u + i];
    const float prob = 1.0f / (1.0f + exp(-r));
    probabilities[i] = prob;
    float my_score = prob + ((args.flags & 1u) ? bias[i] : 0.0f);
    for (uint j = 0u; j < 8u; j++) {
        const float sm = simd_max(my_score);
        const uint si = simd_min(my_score == sm ? uint(i) : 128u);
        if (lane == 0u) { sh_max[sg] = sm; sh_idx[sg] = si; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (i == 0u) {
            float best_s = sh_max[0];
            uint best_i = sh_idx[0];
            for (uint s = 1u; s < 4u; s++) {
                if (sh_max[s] > best_s) { best_s = sh_max[s]; best_i = sh_idx[s]; }
            }
            sh_sel[j] = best_i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (uint(i) == sh_sel[j]) my_score = -3.402823e38f;
    }
    if (i < 8u) {
        float sum = 1.0e-20f;
        for (uint j = 0u; j < 8u; j++) sum += probabilities[sh_sel[j]];
        const uint base = t * 8u;
        const uint id = sh_sel[i];
        ids[base + i] = int(id);
        gate_out[base + i] = (args.flags & 2u) ? gate_scale[id] : 1.0f;
        down_out[base + i] = (probabilities[id] / sum) * args.scale *
                             ((args.flags & 4u) ? down_scale[id] : 1.0f);
    }
}

struct qw3_inkling_post_conv_batch_args { uint width, pos0, channel, n_tokens; };
kernel void qw3_inkling_post_conv_batch(constant qw3_inkling_post_conv_batch_args &args [[buffer(0)]],
        device float *value [[buffer(1)]], device const float4 *weights [[buffer(2)]],
        device float *history [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= args.width) return;
    const uint ch = args.channel + i;
    const float4 w = weights[i];
    const float scale = rsqrt(5.0f);
    float a = args.pos0 ? history[ch] : 0.0f;
    float b = args.pos0 ? history[ch + inkling_state_width] : 0.0f;
    float c = args.pos0 ? history[ch + 2u * inkling_state_width] : 0.0f;
    for (uint t = 0; t < args.n_tokens; t++) {
        const uint64_t idx = uint64_t(t) * args.width + i;
        const float d = value[idx];
        const float residual = dot(float4(a, b, c, d), w);
        value[idx] = (d + residual) * scale;
        a = b;
        b = c;
        c = d;
    }
    history[ch] = a;
    history[ch + inkling_state_width] = b;
    history[ch + 2u * inkling_state_width] = c;
}

kernel void qw3_inkling_post_conv_add_batch(
        constant qw3_inkling_post_conv_batch_args &args [[buffer(0)]],
        device const float *src [[buffer(1)]], device float *cur [[buffer(2)]],
        device const float4 *weights [[buffer(3)]], device float *history [[buffer(4)]],
        uint i [[thread_position_in_grid]]) {
    if (i >= args.width) return;
    const uint ch = args.channel + i;
    const float4 w = weights[i];
    const float scale = rsqrt(5.0f);
    float a = args.pos0 ? history[ch] : 0.0f;
    float b = args.pos0 ? history[ch + inkling_state_width] : 0.0f;
    float c = args.pos0 ? history[ch + 2u * inkling_state_width] : 0.0f;
    for (uint t = 0u; t < args.n_tokens; t++) {
        const uint64_t idx = uint64_t(t) * args.width + i;
        const float d = src[idx];
        const float residual = dot(float4(a, b, c, d), w);
        cur[idx] += (d + residual) * scale;
        a = b;
        b = c;
        c = d;
    }
    history[ch] = a;
    history[ch + inkling_state_width] = b;
    history[ch + 2u * inkling_state_width] = c;
}

kernel void qw3_inkling_moe_reduce_conv_add_batch(
        constant qw3_inkling_post_conv_batch_args &args [[buffer(0)]],
        device const float *down_weights [[buffer(1)]],
        device const half *down [[buffer(2)]],
        device const float4 *conv_weights [[buffer(3)]],
        device float *history [[buffer(4)]],
        device float *cur [[buffer(5)]],
        uint i [[thread_position_in_grid]]) {
    const uint width = args.width;
    if (i >= width) return;
    const uint ch = args.channel + i;
    const float4 w = conv_weights[i];
    const float scale = rsqrt(5.0f);
    float a = args.pos0 ? history[ch] : 0.0f;
    float b = args.pos0 ? history[ch + inkling_state_width] : 0.0f;
    float c = args.pos0 ? history[ch + 2u * inkling_state_width] : 0.0f;
    (void)down_weights;
    for (uint t = 0u; t < args.n_tokens; t++) {
        const uint64_t row_base = uint64_t(t * 8u) * width + i;
        const float d =
            float(down[row_base + 0u * width]) +
            float(down[row_base + 1u * width]) +
            float(down[row_base + 2u * width]) +
            float(down[row_base + 3u * width]) +
            float(down[row_base + 4u * width]) +
            float(down[row_base + 5u * width]) +
            float(down[row_base + 6u * width]) +
            float(down[row_base + 7u * width]);
        const float residual = dot(float4(a, b, c, d), w);
        cur[uint64_t(t) * width + i] += (d + residual) * scale;
        a = b;
        b = c;
        c = d;
    }
    history[ch] = a;
    history[ch + inkling_state_width] = b;
    history[ch + 2u * inkling_state_width] = c;
}

struct qw3_inkling_batch_attn_args {
    uint pos0, n_tokens, capacity, extent;
    uint sliding;
    float qk_limit, relative_limit, log_alpha, log_floor;
};

kernel void qw3_inkling_attention_prepare_q_batch(
        constant qw3_inkling_batch_attn_args &args [[buffer(0)]],
        device float4 *q_buf [[buffer(1)]], device float *rel_buf [[buffer(2)]],
        device const float4 *q_clip [[buffer(3)]], device const float *rel_clip [[buffer(4)]],
        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint head = group.x, t = group.y;
    if (head >= 24u || t >= args.n_tokens) return;
    const uint vec = 32u * head + lane;
    device float4 *qt = q_buf + uint64_t(t) * (24u * 32u);
    const float4 value = qt[vec] * q_clip[vec];
    const float norm = sqrt(simd_sum(dot(value, value)));
    qt[vec] = value * min(1.0f, args.qk_limit / (norm + 1.0e-6f));

    const uint ri = 16u * head + lane;
    device float *rt = rel_buf + uint64_t(t) * (24u * 16u);
    const float relative = lane < 16u ? rt[ri] * rel_clip[ri] : 0.0f;
    const float rel_norm = sqrt(simd_sum(relative * relative));
    if (lane < 16u) {
        rt[ri] = relative * min(1.0f, args.relative_limit / (rel_norm + 1.0e-6f));
    }
}

kernel void qw3_inkling_attention_prepare_kv_batch(
        constant qw3_inkling_batch_attn_args &args [[buffer(0)]],
        device const float4 *k_in [[buffer(1)]], device const float4 *v_in [[buffer(2)]],
        device const float4 *k_taps [[buffer(3)]], device const float4 *v_taps [[buffer(4)]],
        device half4 *incoming [[buffer(5)]], device float *history [[buffer(6)]],
        device const float4 *k_conv [[buffer(7)]], device const float4 *v_conv [[buffer(8)]],
        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint kv_head = group.x;
    const uint is_v = group.y;
    if (kv_head >= 8u || is_v >= 2u) return;
    const bool dynamic = !args.sliding;
    const float conv_scale = dynamic ? (1.0f / 3.0f) : rsqrt(5.0f);
    const uint start = kv_head * inkling_head_width + lane * 4u;
    const uint start4 = start >> 2u;
    const uint hist_base = (is_v ? inkling_kv_width : 0u) + start;
    device const float4 *in_src = is_v ? v_in : k_in;
    device const float4 *taps_src = is_v ? v_taps : k_taps;
    device const float4 *conv_src = is_v ? v_conv : k_conv;
    const uint inc_off = is_v ? (inkling_kv_width / 4u) : 0u;
    float4 a, b, c;
    float4 w[4];
    for (uint i = 0; i < 4u; i++) {
        w[i] = conv_src[start + i];
        a[i] = args.pos0 ? history[hist_base + i] : 0.0f;
        b[i] = args.pos0 ? history[hist_base + i + inkling_state_width] : 0.0f;
        c[i] = args.pos0 ? history[hist_base + i + 2u * inkling_state_width] : 0.0f;
    }
    const uint kv4_stride = inkling_kv_width / 4u;
    const uint inc_stride = 2u * kv4_stride;
    for (uint t = 0; t < args.n_tokens; t++) {
        const float4 tp = dynamic ? taps_src[t * 8u + kv_head] : float4(0.0f);
        const float4 d = in_src[uint64_t(t) * kv4_stride + start4];
        float4 outv;
        for (uint i = 0; i < 4u; i++) {
            float res = dot(float4(a[i], b[i], c[i], d[i]), w[i]);
            if (dynamic) res += dot(float4(d[i], c[i], b[i], a[i]), tp);
            outv[i] = (d[i] + res) * conv_scale;
        }
        a = b; b = c; c = d;
        const float ss = simd_sum(dot(outv, outv));
        const float scale = is_v
            ? rsqrt(ss / float(inkling_head_width) + 1.0e-6f)
            : min(1.0f, args.qk_limit / (sqrt(ss) + 1.0e-6f));
        incoming[uint64_t(t) * inc_stride + start4 + inc_off] = half4(outv * scale);
    }
    for (uint i = 0; i < 4u; i++) {
        history[hist_base + i] = a[i];
        history[hist_base + i + inkling_state_width] = b[i];
        history[hist_base + i + 2u * inkling_state_width] = c[i];
    }
}

// Multi-query tiled causal attention: 4 simdgroups (1 query each) share each loaded K/V tile
// of 16 positions in threadgroup memory, applying relative bias, per-position tau, and sink.
kernel void qw3_inkling_attention_batch_tiled(
        constant qw3_inkling_batch_attn_args &args [[buffer(0)]],
        device const float4 *q_buf [[buffer(1)]], device const float *rel_buf [[buffer(2)]],
        device const float *sink_buf [[buffer(3)]], device const float4 *gate_buf [[buffer(4)]],
        device const half4 *cache [[buffer(5)]], device const float *bias [[buffer(6)]],
        device float4 *mixed_buf [[buffer(7)]],
        device const half4 *incoming [[buffer(8)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr uint Q_TILE = 4u;
    constexpr uint K_TILE = 16u;
    const uint head = group.x;
    const uint t0 = group.y * Q_TILE;
    if (head >= 24u || t0 >= args.n_tokens) return;
    const uint t = t0 + uint(sg);
    const bool active = t < args.n_tokens;
    const uint pos = args.pos0 + (active ? t : (args.n_tokens - 1u));
    const uint kv_head = head / 3u;

    const uint pos_min = args.pos0 + t0;
    const uint pos_max = args.pos0 + min(args.n_tokens, t0 + Q_TILE) - 1u;
    const uint first_min = pos_min + 1u > args.capacity ? pos_min + 1u - args.capacity : 0u;

    const float4 q = active ? q_buf[uint64_t(t) * (24u * 32u) + head * 32u + lane] : float4(0.0f);
    const float rel = (active && lane < 16u) ? rel_buf[uint64_t(t) * (24u * 16u) + head * 16u + lane] : 0.0f;
    const float tau = args.sliding ? 1.0f :
        1.0f + args.log_alpha * log(max(1.0f, float(pos + 1u) / args.log_floor));
    const uint my_first = pos + 1u > args.capacity ? pos + 1u - args.capacity : 0u;
    const float inv_sqrt_d = rsqrt(float(inkling_head_width));

    float maximum = active ? sink_buf[uint64_t(t) * 24u + head] : 0.0f;
    float sum = 1.0f;
    float4 value = float4(0.0f);

    threadgroup half4 k_sh[K_TILE * 32u];
    threadgroup half4 v_sh[K_TILE * 32u];

    for (uint j0 = first_min; j0 <= pos_max; j0 += K_TILE) {
        const uint k_count = min(K_TILE, pos_max + 1u - j0);
        for (uint idx = uint(tid); idx < k_count * 32u; idx += 128u) {
            const uint k_local = idx >> 5u;
            const uint k_lane = idx & 31u;
            const uint pos = j0 + k_local;
            const bool in_batch = pos >= args.pos0;
            const uint slot = in_batch ? pos - args.pos0 : pos % args.capacity;
            const uint offset = slot * (2u * inkling_kv_width / 4u) + kv_head * 32u + k_lane;
            device const half4 *rows = in_batch ? incoming : cache;
            k_sh[idx] = rows[offset];
            v_sh[idx] = rows[offset + inkling_kv_width / 4u];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint k_local = 0u; k_local < k_count; k_local++) {
                const uint j = j0 + k_local;
                if (j < my_first || j > pos) continue;
                const uint idx = k_local * 32u + lane;
                const uint distance = pos - j;
                float score = dot(q, float4(k_sh[idx])) * inv_sqrt_d;
                if (distance < args.extent && lane < 16u) {
                    score += rel * bias[distance * 16u + lane];
                }
                score = simd_sum(score) * tau;
                const float next = max(maximum, score);
                const float keep = exp(maximum - next);
                const float add = exp(score - next);
                sum = sum * keep + add;
                value = value * keep + float4(v_sh[idx]) * add;
                maximum = next;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        const uint offset = uint64_t(t) * (24u * 32u) + head * 32u + lane;
        const float4 gate = gate_buf[offset];
        mixed_buf[offset] = value / (sum * (1.0f + exp(-gate)));
    }
}

kernel void qw3_inkling_attention_commit_batch(
        constant qw3_inkling_batch_attn_args &args [[buffer(0)]],
        device half4 *cache [[buffer(1)]], device const half4 *incoming [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint row = 2u * inkling_kv_width / 4u;
    if (gid.x >= row || gid.y >= args.n_tokens) return;
    // When a block exceeds the ring, only its newest row per slot survives.
    if (gid.y + args.capacity < args.n_tokens) return;
    const uint slot = (args.pos0 + gid.y) % args.capacity;
    cache[slot * row + gid.x] = incoming[gid.y * row + gid.x];
}

#ifdef QW3_METAL_HAS_TENSOR
// Metal 4 NAX FlashAttention for Inkling prefill: Q_M=32, K_N=32, D=128 using matmul2d
// for both Q*K^T and P*V, with online tile softmax, relative bias, per-position tau, sink, and gate.
kernel void qw3_inkling_attention_batch_nax(
        constant qw3_inkling_batch_attn_args &args [[buffer(0)]],
        device const float4 *q_buf [[buffer(1)]], device const float *rel_buf [[buffer(2)]],
        device const float *sink_buf [[buffer(3)]], device const float4 *gate_buf [[buffer(4)]],
        device const half4 *cache [[buffer(5)]], device const float *bias [[buffer(6)]],
        device float4 *mixed_buf [[buffer(7)]],
        device const half4 *incoming [[buffer(8)]],
        threadgroup char *shmem [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr uint Q_M = 32u;
    constexpr uint K_N = 32u;
    constexpr uint D = 128u;
    const uint head = group.x;
    const uint t0 = group.y * Q_M;
    if (head >= 24u || t0 >= args.n_tokens) return;
    const uint q_count = min(Q_M, args.n_tokens - t0);
    const uint kv_head = head / 3u;
    const uint pos_min = args.pos0 + t0;
    const uint pos_max = args.pos0 + t0 + q_count - 1u;
    const uint first_min = pos_min + 1u > args.capacity ? pos_min + 1u - args.capacity : 0u;
    const float inv_sqrt_d = rsqrt(float(D));

    threadgroup half *q_sh = (threadgroup half *)(shmem + 0);          // 8192 B
    threadgroup half *kv_sh = (threadgroup half *)(shmem + 8192);      // 8192 B
    threadgroup float *s_sh = (threadgroup float *)(shmem + 16384);    // 4096 B
    threadgroup half *p_sh = (threadgroup half *)(shmem + 20480);      // 2048 B
    threadgroup float *rel_sh = (threadgroup float *)(shmem + 22528);  // 2048 B
    threadgroup float *rescale_sh = (threadgroup float *)(shmem + 24576); // 128 B
    threadgroup float *sum_sh = (threadgroup float *)(shmem + 24704);     // 128 B

    // Load Q[q_count, 128] into q_sh as half4
    threadgroup half4 *q_sh4 = (threadgroup half4 *)q_sh;
    for (uint idx = tid; idx < Q_M * 32u; idx += 128u) {
        const uint ql = idx >> 5u;
        const uint qlane = idx & 31u;
        if (ql < q_count) {
            q_sh4[idx] = half4(q_buf[uint64_t(t0 + ql) * (24u * 32u) + head * 32u + qlane]);
        } else {
            q_sh4[idx] = half4(0.0f);
        }
    }
    // Load rel[q_count, 16] into rel_sh as float4
    threadgroup float4 *rel_sh4 = (threadgroup float4 *)rel_sh;
    if (tid < Q_M * 4u) {
        const uint ql = tid >> 2u;
        const uint r4 = tid & 3u;
        if (ql < q_count) {
            device const float4 *rp = (device const float4 *)(rel_buf + uint64_t(t0 + ql) * (24u * 16u) + head * 16u);
            rel_sh4[tid] = rp[r4];
        } else {
            rel_sh4[tid] = float4(0.0f);
        }
    }

    float row_max[8], row_sum[8], my_tau[8];
    uint my_pos[8], my_first[8];
    for (uint r = 0u; r < 8u; r++) {
        const uint ql = uint(sg) * 8u + r;
        if (ql < q_count) {
            const uint t = t0 + ql;
            row_max[r] = sink_buf[uint64_t(t) * 24u + head];
            row_sum[r] = 1.0f;
            const uint p = args.pos0 + t;
            my_pos[r] = p;
            my_first[r] = p + 1u > args.capacity ? p + 1u - args.capacity : 0u;
            my_tau[r] = args.sliding ? 1.0f :
                (1.0f + args.log_alpha * log(max(1.0f, float(p + 1u) / args.log_floor)));
        } else {
            row_max[r] = 0.0f;
            row_sum[r] = 1.0f;
            my_pos[r] = 0u;
            my_first[r] = 1u;
            my_tau[r] = 1.0f;
        }
    }

    auto tQ = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        q_sh, dextents<int32_t, 2>(int(D), int(Q_M)));
    auto tKV = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        kv_sh, dextents<int32_t, 2>(int(D), int(K_N)));
    auto tS = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        s_sh, dextents<int32_t, 2>(int(K_N), int(Q_M)));
    auto tP = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        p_sh, dextents<int32_t, 2>(int(K_N), int(Q_M)));

    matmul2d<matmul2d_descriptor(int(Q_M), int(K_N), int(D), false, true, true,
        matmul2d_descriptor::mode::multiply),
        execution_simdgroups<4>> mmQK;
    matmul2d<matmul2d_descriptor(int(Q_M), int(D), int(K_N), false, false, true,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mmPV;

    auto cO = mmPV.template get_destination_cooperative_tensor<decltype(tP), decltype(tKV), float>();
    for (uint16_t i = 0; i < cO.get_capacity(); i++) {
        if (cO.is_valid_element(i)) cO[i] = 0.0f;
    }

    threadgroup half4 *kv_sh4 = (threadgroup half4 *)kv_sh;
    for (uint j0 = first_min; j0 <= pos_max; j0 += K_N) {
        const uint k_count = min(K_N, pos_max + 1u - j0);
        // 1. Load K tile into kv_sh
        for (uint idx = tid; idx < K_N * 32u; idx += 128u) {
            const uint kl = idx >> 5u;
            const uint klane = idx & 31u;
            if (kl < k_count) {
                const uint jpos = j0 + kl;
                const bool in_batch = jpos >= args.pos0;
                const uint slot = in_batch ? (jpos - args.pos0) : (jpos % args.capacity);
                const uint off = slot * (2u * inkling_kv_width / 4u) + kv_head * 32u + klane;
                kv_sh4[idx] = (in_batch ? incoming : cache)[off];
            } else {
                kv_sh4[idx] = half4(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. S = Q * K^T via NAX matmul2d
        auto cS = mmQK.template get_destination_cooperative_tensor<decltype(tQ), decltype(tKV), float>();
        {
            auto mQ = tQ.slice(0, 0);
            auto mK = tKV.slice(0, 0);
            mmQK.run(mQ, mK, cS);
        }
        cS.store(tS.slice(0, 0));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Tile online softmax with relative bias, tau, causal+window mask -> p_sh & rescale_sh
        const uint j = j0 + uint(lane);
        for (uint r = 0u; r < 8u; r++) {
            const uint ql = uint(sg) * 8u + r;
            if (ql < q_count) {
                const bool valid = (uint(lane) < k_count) && (j >= my_first[r]) && (j <= my_pos[r]);
                float score = -3.402823e38f;
                if (valid) {
                    score = s_sh[ql * K_N + uint(lane)] * inv_sqrt_d;
                    const uint dist = my_pos[r] - j;
                    if (dist < args.extent) {
                        threadgroup const float4 *r4 = rel_sh4 + ql * 4u;
                        device const float4 *b4 = (device const float4 *)(bias + dist * 16u);
                        score += dot(r4[0], b4[0]) + dot(r4[1], b4[1]) +
                                 dot(r4[2], b4[2]) + dot(r4[3], b4[3]);
                    }
                    score *= my_tau[r];
                }
                const float tile_max = simd_max(score);
                const float next_max = max(row_max[r], tile_max);
                const float rescale = exp(row_max[r] - next_max);
                const float p = valid ? exp(score - next_max) : 0.0f;
                const float tile_sum = simd_sum(p);
                row_sum[r] = row_sum[r] * rescale + tile_sum;
                row_max[r] = next_max;
                if (lane == 0u) rescale_sh[ql] = rescale;
                p_sh[ql * K_N + uint(lane)] = half(p);
            } else {
                if (lane == 0u) rescale_sh[ql] = 1.0f;
                p_sh[ql * K_N + uint(lane)] = half(0.0f);
            }
        }

        // 4. Load V tile into kv_sh (safe: mmQK completed before barrier 2)
        for (uint idx = tid; idx < K_N * 32u; idx += 128u) {
            const uint kl = idx >> 5u;
            const uint klane = idx & 31u;
            if (kl < k_count) {
                const uint jpos = j0 + kl;
                const bool in_batch = jpos >= args.pos0;
                const uint slot = in_batch ? (jpos - args.pos0) : (jpos % args.capacity);
                const uint off = slot * (2u * inkling_kv_width / 4u) + kv_head * 32u + klane;
                kv_sh4[idx] = (in_batch ? incoming : cache)[off + inkling_kv_width / 4u];
            } else {
                kv_sh4[idx] = half4(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 5. Rescale cO and accumulate P * V via NAX matmul2d
        for (uint16_t i = 0; i < cO.get_capacity(); i++) {
            if (cO.is_valid_element(i)) {
                auto idx = cO.get_multidimensional_index(i);
                cO[i] *= rescale_sh[idx[1]];
            }
        }
        {
            auto mP = tP.slice(0, 0);
            auto mV = tKV.slice(0, 0);
            mmPV.run(mP, mV, cO);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store row sums and O tile (reusing shmem[0..16383]), apply norm & gate
    for (uint r = 0u; r < 8u; r++) {
        if (lane == 0u) sum_sh[uint(sg) * 8u + r] = row_sum[r];
    }
    threadgroup float *o_sh = (threadgroup float *)shmem;
    auto tO = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        o_sh, dextents<int32_t, 2>(int(D), int(Q_M)));
    cO.store(tO.slice(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup const float4 *o_sh4 = (threadgroup const float4 *)o_sh;
    for (uint r = 0u; r < 8u; r++) {
        const uint ql = uint(sg) * 8u + r;
        if (ql < q_count) {
            const uint t = t0 + ql;
            const uint offset = uint64_t(t) * (24u * 32u) + head * 32u + uint(lane);
            const float4 gate = gate_buf[offset];
            mixed_buf[offset] = o_sh4[ql * 32u + uint(lane)] / (sum_sh[ql] * (1.0f + exp(-gate)));
        }
    }
}
#endif
