/* Private FP32 prefill candidates. Packed reduction geometry inspired by
 * mlx-lm gated_delta.py; retain qw3 state layout, gates and update arithmetic.
 * Compile in the same library/options as the OFF comparator. */
/* Keep only the reduction tree precise; dot(float4) and the state update
 * retain the native comparator's fast-math arithmetic. */
inline float probe_tree8(thread const float *p) {
#pragma clang fp reassociate(off)
    return ((p[0]+p[1])+(p[2]+p[3]))+((p[4]+p[5])+(p[6]+p[7]));
}
inline float probe_tree4(thread const float *p) {
#pragma clang fp reassociate(off)
    return (p[0]+p[1])+(p[2]+p[3]);
}
kernel void probe_gdn_packed4(constant qw3_batch_gdn_args &args,
    device float *state, device float *scratch, device const float *dt_bias,
    device const float *a, uint2 group [[threadgroup_position_in_grid]],
    ushort simd_idx [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint hv = group.y;
    const uint j = group.x * 32u + uint(simd_idx) * 8u + uint(lane) / 4u;
    const uint row_lane = uint(lane) % 4u;
    if (hv >= args.v_heads || j >= args.head_dim) return;
    const uint hk = hv % args.q_heads;
    const uint qk_n = args.q_heads * args.head_dim;
    device float *st = state + uint64_t(hv) * args.head_dim * args.head_dim;
    const uint i0 = row_lane * 32u;
    const float norm_scale = rsqrt(float(args.head_dim));
    const float dt_hv = dt_bias[hv], a_hv = a[hv];
    float4 sv[8];
    #pragma unroll
    for (uint p = 0; p < 8u; p++) {
        uint i = i0 + p * 4u;
        sv[p] = float4(st[(i+0u)*args.head_dim+j],st[(i+1u)*args.head_dim+j],
                       st[(i+2u)*args.head_dim+j],st[(i+3u)*args.head_dim+j]);
    }
    for (uint t = 0; t < args.n_tokens; t++) {
        uint64_t base = uint64_t(t) * args.stride;
        device const float *qh = scratch + base + args.conv_offset + uint64_t(hk)*args.head_dim;
        device const float *kh = qh + qk_n;
        device const float *vh = scratch + base + args.conv_offset + 2u*qk_n + uint64_t(hv)*args.head_dim;
        float b = 0.0f, g = 0.0f;
        if (lane == 0) {
            float beta_raw = scratch[base + args.beta_offset + hv];
            b = 1.0f / (1.0f + exp(-beta_raw));
            float alpha_raw = scratch[base + args.alpha_offset + hv] + dt_hv;
            float sp = alpha_raw > 20.0f ? alpha_raw : (alpha_raw < -20.0f ? exp(alpha_raw) : log(1.0f + exp(alpha_raw)));
            g = exp(sp * a_hv);
        }
        b = simd_broadcast(b, 0); g = simd_broadcast(g, 0);
        float4 kv[8], qv[8];
        float part[8];
        #pragma unroll
        for (uint p = 0; p < 8u; p++) {
            kv[p] = *((device const float4 *)(kh + i0 + p*4u));
            qv[p] = *((device const float4 *)(qh + i0 + p*4u));
            part[p] = dot(sv[p], kv[p]);
        }
        float sk = probe_tree8(part);
        sk += simd_shuffle_xor(sk, 1);
        sk += simd_shuffle_xor(sk, 2);
        float d = b * (vh[j] - sk*g);
        #pragma unroll
        for (uint p = 0; p < 8u; p++) {
            sv[p] = sv[p]*g + kv[p]*d;
            part[p] = dot(sv[p], qv[p]);
        }
        float value = probe_tree8(part);
        value += simd_shuffle_xor(value, 1);
        value += simd_shuffle_xor(value, 2);
        if (row_lane == 0) scratch[base+args.inner_offset+uint64_t(hv)*args.head_dim+j] = value*norm_scale;
    }
    #pragma unroll
    for (uint p = 0; p < 8u; p++) {
        uint i = i0 + p*4u;
        st[(i+0u)*args.head_dim+j] = sv[p].x; st[(i+1u)*args.head_dim+j] = sv[p].y;
        st[(i+2u)*args.head_dim+j] = sv[p].z; st[(i+3u)*args.head_dim+j] = sv[p].w;
    }
}
kernel void probe_gdn_packed8(constant qw3_batch_gdn_args &args,
    device float *state, device float *scratch, device const float *dt_bias,
    device const float *a, uint2 group [[threadgroup_position_in_grid]],
    ushort simd_idx [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint hv = group.y;
    const uint j = group.x * 16u + uint(simd_idx) * 4u + uint(lane) / 8u;
    const uint row_lane = uint(lane) % 8u;
    if (hv >= args.v_heads || j >= args.head_dim) return;
    const uint hk = hv % args.q_heads;
    const uint qk_n = args.q_heads * args.head_dim;
    device float *st = state + uint64_t(hv) * args.head_dim * args.head_dim;
    const uint i0 = row_lane * 16u;
    const float norm_scale = rsqrt(float(args.head_dim));
    const float dt_hv = dt_bias[hv], a_hv = a[hv];
    float4 sv[4];
    #pragma unroll
    for (uint p = 0; p < 4u; p++) {
        uint i = i0 + p * 4u;
        sv[p] = float4(st[(i+0u)*args.head_dim+j],st[(i+1u)*args.head_dim+j],
                       st[(i+2u)*args.head_dim+j],st[(i+3u)*args.head_dim+j]);
    }
    for (uint t = 0; t < args.n_tokens; t++) {
        uint64_t base = uint64_t(t) * args.stride;
        device const float *qh = scratch + base + args.conv_offset + uint64_t(hk)*args.head_dim;
        device const float *kh = qh + qk_n;
        device const float *vh = scratch + base + args.conv_offset + 2u*qk_n + uint64_t(hv)*args.head_dim;
        float b = 0.0f, g = 0.0f;
        if (lane == 0) {
            float beta_raw = scratch[base + args.beta_offset + hv];
            b = 1.0f / (1.0f + exp(-beta_raw));
            float alpha_raw = scratch[base + args.alpha_offset + hv] + dt_hv;
            float sp = alpha_raw > 20.0f ? alpha_raw : (alpha_raw < -20.0f ? exp(alpha_raw) : log(1.0f + exp(alpha_raw)));
            g = exp(sp * a_hv);
        }
        b = simd_broadcast(b, 0); g = simd_broadcast(g, 0);
        float4 kv[4], qv[4];
        float part[4];
        #pragma unroll
        for (uint p = 0; p < 4u; p++) {
            kv[p] = *((device const float4 *)(kh + i0 + p*4u));
            qv[p] = *((device const float4 *)(qh + i0 + p*4u));
            part[p] = dot(sv[p], kv[p]);
        }
        float sk = probe_tree4(part);
        sk += simd_shuffle_xor(sk, 1);
        sk += simd_shuffle_xor(sk, 2);
        sk += simd_shuffle_xor(sk, 4);
        float d = b * (vh[j] - sk*g);
        #pragma unroll
        for (uint p = 0; p < 4u; p++) {
            sv[p] = sv[p]*g + kv[p]*d;
            part[p] = dot(sv[p], qv[p]);
        }
        float value = probe_tree4(part);
        value += simd_shuffle_xor(value, 1);
        value += simd_shuffle_xor(value, 2);
        value += simd_shuffle_xor(value, 4);
        if (row_lane == 0) scratch[base+args.inner_offset+uint64_t(hv)*args.head_dim+j] = value*norm_scale;
    }
    #pragma unroll
    for (uint p = 0; p < 4u; p++) {
        uint i = i0 + p*4u;
        st[(i+0u)*args.head_dim+j] = sv[p].x; st[(i+1u)*args.head_dim+j] = sv[p].y;
        st[(i+2u)*args.head_dim+j] = sv[p].z; st[(i+3u)*args.head_dim+j] = sv[p].w;
    }
}
kernel void probe_gdn_pipeline(constant qw3_batch_gdn_args &args,
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
    /* Each SIMD owns two independent state columns for the entire chunk.
     * Keep them in FP32 registers; no other group consumes intermediate
     * state. Only the per-token core output and the final state are shared. */
    float4 sv0 = float4(st[(i0 + 0u) * args.head_dim + j0],
                       st[(i0 + 1u) * args.head_dim + j0],
                       st[(i0 + 2u) * args.head_dim + j0],
                       st[(i0 + 3u) * args.head_dim + j0]);
    float4 sv1 = float4(st[(i0 + 0u) * args.head_dim + j1],
                       st[(i0 + 1u) * args.head_dim + j1],
                       st[(i0 + 2u) * args.head_dim + j1],
                       st[(i0 + 3u) * args.head_dim + j1]);
    uint load_t = 0;
    uint64_t load_base = uint64_t(load_t) * args.stride;
    device const float *qh = scratch + load_base + args.conv_offset + uint64_t(hk) * args.head_dim;
    device const float *kh = scratch + load_base + args.conv_offset + qk_n + uint64_t(hk) * args.head_dim;
    device const float *vh = scratch + load_base + args.conv_offset + 2u * qk_n + uint64_t(hv) * args.head_dim;
    float next_b = 0.0f;
    float next_g = 0.0f;
    if (lane == 0) {
        float beta_raw = scratch[load_base + args.beta_offset + hv];
        next_b = 1.0f / (1.0f + exp(-beta_raw));
        float alpha_raw = scratch[load_base + args.alpha_offset + hv] + dt_hv;
        float sp = alpha_raw > 20.0f ? alpha_raw : (alpha_raw < -20.0f ? exp(alpha_raw) : log(1.0f + exp(alpha_raw)));
        next_g = exp(sp * a_hv);
    }
    next_b = simd_broadcast(next_b, 0);
    next_g = simd_broadcast(next_g, 0);
    float4 next_kv = *((device const float4 *)(kh + i0));
    float4 next_qv = *((device const float4 *)(qh + i0));
    float next_v0 = vh[j0], next_v1 = vh[j1];
    for (uint t = 0; t < args.n_tokens; t++) {
        uint64_t base = uint64_t(t) * args.stride;
        float4 kv = next_kv, qv = next_qv;
        float b = next_b, g = next_g, v0 = next_v0, v1 = next_v1;
        if (t + 1u < args.n_tokens) {
            load_t = t + 1u;
            uint64_t load_base = uint64_t(load_t) * args.stride;
            device const float *qh = scratch + load_base + args.conv_offset + uint64_t(hk) * args.head_dim;
            device const float *kh = scratch + load_base + args.conv_offset + qk_n + uint64_t(hk) * args.head_dim;
            device const float *vh = scratch + load_base + args.conv_offset + 2u * qk_n + uint64_t(hv) * args.head_dim;
            next_b = 0.0f;
            next_g = 0.0f;
            if (lane == 0) {
                float beta_raw = scratch[load_base + args.beta_offset + hv];
                next_b = 1.0f / (1.0f + exp(-beta_raw));
                float alpha_raw = scratch[load_base + args.alpha_offset + hv] + dt_hv;
                float sp = alpha_raw > 20.0f ? alpha_raw : (alpha_raw < -20.0f ? exp(alpha_raw) : log(1.0f + exp(alpha_raw)));
                next_g = exp(sp * a_hv);
            }
            next_b = simd_broadcast(next_b, 0);
            next_g = simd_broadcast(next_g, 0);
            next_kv = *((device const float4 *)(kh + i0));
            next_qv = *((device const float4 *)(qh + i0));
            next_v0 = vh[j0]; next_v1 = vh[j1];
        }
        float sk0 = simd_sum(dot(sv0, kv));
        float d0 = b * (v0 - sk0 * g);
        sv0 = sv0 * g + kv * d0;
        float out0 = simd_sum(dot(sv0, qv));
        if (lane == 0) scratch[base + args.inner_offset + uint64_t(hv) * args.head_dim + j0] = out0 * norm_scale;
        float sk1 = simd_sum(dot(sv1, kv));
        float d1 = b * (v1 - sk1 * g);
        sv1 = sv1 * g + kv * d1;
        float out1 = simd_sum(dot(sv1, qv));
        if (lane == 0) scratch[base + args.inner_offset + uint64_t(hv) * args.head_dim + j1] = out1 * norm_scale;
    }
    st[(i0 + 0u) * args.head_dim + j0] = sv0.x;
    st[(i0 + 1u) * args.head_dim + j0] = sv0.y;
    st[(i0 + 2u) * args.head_dim + j0] = sv0.z;
    st[(i0 + 3u) * args.head_dim + j0] = sv0.w;
    st[(i0 + 0u) * args.head_dim + j1] = sv1.x;
    st[(i0 + 1u) * args.head_dim + j1] = sv1.y;
    st[(i0 + 2u) * args.head_dim + j1] = sv1.z;
    st[(i0 + 3u) * args.head_dim + j1] = sv1.w;
}