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
