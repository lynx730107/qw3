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
