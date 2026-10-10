kernel void private_gdn_local_all(constant qw3_fused_gdn_args &args,
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
    if (hv >= args.v_heads || j >= args.head_dim || args.head_dim != 128u) return;
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
    constexpr uint CACHE_N = 64u;
    float cached[CACHE_N];
    float sk = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 128u; i++) {
        const float prior = sin[i * args.head_dim + j];
        if (i < CACHE_N) cached[i] = prior;
        sk += prior * kh[i];
    }
    float d = b * (vh[j] - sk * g);
    float sum = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 128u; i++) {
        uint idx = i * args.head_dim + j;
        float sv = (i < CACHE_N ? cached[i] : sin[idx]) * g + kh[i] * d;
        sout[idx] = sv;
        sum += sv * qh[i];
    }
    float core = sum * rsqrt(float(args.head_dim));
    float ss = simd_sum(core * core);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < ((args.head_dim + 31u) / 32u) ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.head_dim) + args.eps);
    float zi = scratch[args.z_offset + uint64_t(hv) * args.head_dim + j];
    float gate = zi / (1.0f + exp(-zi));
    inner_out[uint64_t(hv) * args.head_dim + j] = core * scale * w[j] * gate;
}

kernel void private_gdn_local_simd(constant qw3_fused_gdn_args &args,
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
    if (hv >= args.v_heads || j >= args.head_dim || args.head_dim != 128u) return;
    uint hk = hv % args.q_heads;
    uint state_n = args.head_dim * args.head_dim;
    device const float *qh = q + uint64_t(hk) * args.head_dim;
    device const float *kh = k + uint64_t(hk) * args.head_dim;
    device const float *vh = v + uint64_t(hv) * args.head_dim;
    device const float *sin = state_in + uint64_t(hv) * state_n;
    device float *sout = state_out + uint64_t(hv) * state_n;
    float b = 0.0f, g = 0.0f;
    if (lane == 0) {
        float beta_raw = scratch[args.beta_offset + hv];
        b = 1.0f / (1.0f + exp(-beta_raw));
        float alpha_raw = scratch[args.alpha_offset + hv] + dt_bias[hv];
        float sp = alpha_raw > 20.0f ? alpha_raw : (alpha_raw < -20.0f ? exp(alpha_raw) : log(1.0f + exp(alpha_raw)));
        g = exp(sp * a[hv]);
    }
    b = simd_shuffle(b, 0u);
    g = simd_shuffle(g, 0u);
    constexpr uint CACHE_N = 64u;
    float cached[CACHE_N];
    float sk = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 128u; i++) {
        const float prior = sin[i * args.head_dim + j];
        if (i < CACHE_N) cached[i] = prior;
        sk += prior * kh[i];
    }
    float d = b * (vh[j] - sk * g);
    float sum = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 128u; i++) {
        uint idx = i * args.head_dim + j;
        float sv = (i < CACHE_N ? cached[i] : sin[idx]) * g + kh[i] * d;
        sout[idx] = sv;
        sum += sv * qh[i];
    }
    float core = sum * rsqrt(float(args.head_dim));
    float ss = simd_sum(core * core);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < ((args.head_dim + 31u) / 32u) ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.head_dim) + args.eps);
    float zi = scratch[args.z_offset + uint64_t(hv) * args.head_dim + j];
    float gate = zi / (1.0f + exp(-zi));
    inner_out[uint64_t(hv) * args.head_dim + j] = core * scale * w[j] * gate;
}
