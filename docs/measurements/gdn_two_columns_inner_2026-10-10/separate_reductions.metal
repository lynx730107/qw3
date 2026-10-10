kernel void private_gdn_two_cache32(constant qw3_fused_gdn_args &args,
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
    if (hv >= args.v_heads || j >= 64u || args.head_dim != 128u) return;
    uint j1 = uint(j) + 64u;
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
    constexpr uint CACHE_N = 32u;
    float2 cached[CACHE_N];
    float sk0 = 0.0f, sk1 = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 128u; i++) {
        const float prior0 = sin[i * args.head_dim + j];
        const float prior1 = sin[i * args.head_dim + j1];
        if (i < CACHE_N) cached[i] = float2(prior0, prior1);
        sk0 = fma(prior0, kh[i], sk0);
        sk1 = fma(prior1, kh[i], sk1);
    }
    float d0 = b * fma(-sk0, g, vh[j]);
    float d1 = b * fma(-sk1, g, vh[j1]);
    float sum0 = 0.0f, sum1 = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 128u; i++) {
        uint idx = i * args.head_dim + j;
        float sv0 = fma(kh[i], d0, (i < CACHE_N ? cached[i].x : sin[idx]) * g);
        float sv1 = fma(kh[i], d1, (i < CACHE_N ? cached[i].y : sin[idx + 64u]) * g);
        sout[idx] = sv0;
        sout[idx + 64u] = sv1;
        sum0 = fma(sv0, qh[i], sum0);
        sum1 = fma(sv1, qh[i], sum1);
    }
    float core0 = sum0 * rsqrt(float(args.head_dim));
    float core1 = sum1 * rsqrt(float(args.head_dim));
    float ss = simd_sum(core0 * core0);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss1 = simd_sum(core1 * core1);
    if (lane == 0) sh[simd_idx + 2u] = ss1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < ((args.head_dim + 31u) / 32u) ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.head_dim) + args.eps);
    float zi0 = scratch[args.z_offset + uint64_t(hv) * args.head_dim + j];
    float zi1 = scratch[args.z_offset + uint64_t(hv) * args.head_dim + j1];
    float gate0 = zi0 / (1.0f + exp(-zi0));
    float gate1 = zi1 / (1.0f + exp(-zi1));
    inner_out[uint64_t(hv) * args.head_dim + j] = core0 * scale * w[j] * gate0;
    inner_out[uint64_t(hv) * args.head_dim + j1] = core1 * scale * w[j1] * gate1;
}

kernel void private_gdn_two_cache48(constant qw3_fused_gdn_args &args,
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
    if (hv >= args.v_heads || j >= 64u || args.head_dim != 128u) return;
    uint j1 = uint(j) + 64u;
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
    constexpr uint CACHE_N = 48u;
    float2 cached[CACHE_N];
    float sk0 = 0.0f, sk1 = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 128u; i++) {
        const float prior0 = sin[i * args.head_dim + j];
        const float prior1 = sin[i * args.head_dim + j1];
        if (i < CACHE_N) cached[i] = float2(prior0, prior1);
        sk0 = fma(prior0, kh[i], sk0);
        sk1 = fma(prior1, kh[i], sk1);
    }
    float d0 = b * fma(-sk0, g, vh[j]);
    float d1 = b * fma(-sk1, g, vh[j1]);
    float sum0 = 0.0f, sum1 = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 128u; i++) {
        uint idx = i * args.head_dim + j;
        float sv0 = fma(kh[i], d0, (i < CACHE_N ? cached[i].x : sin[idx]) * g);
        float sv1 = fma(kh[i], d1, (i < CACHE_N ? cached[i].y : sin[idx + 64u]) * g);
        sout[idx] = sv0;
        sout[idx + 64u] = sv1;
        sum0 = fma(sv0, qh[i], sum0);
        sum1 = fma(sv1, qh[i], sum1);
    }
    float core0 = sum0 * rsqrt(float(args.head_dim));
    float core1 = sum1 * rsqrt(float(args.head_dim));
    float ss = simd_sum(core0 * core0);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss1 = simd_sum(core1 * core1);
    if (lane == 0) sh[simd_idx + 2u] = ss1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < ((args.head_dim + 31u) / 32u) ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.head_dim) + args.eps);
    float zi0 = scratch[args.z_offset + uint64_t(hv) * args.head_dim + j];
    float zi1 = scratch[args.z_offset + uint64_t(hv) * args.head_dim + j1];
    float gate0 = zi0 / (1.0f + exp(-zi0));
    float gate1 = zi1 / (1.0f + exp(-zi1));
    inner_out[uint64_t(hv) * args.head_dim + j] = core0 * scale * w[j] * gate0;
    inner_out[uint64_t(hv) * args.head_dim + j1] = core1 * scale * w[j1] * gate1;
}

kernel void private_gdn_two_cache64(constant qw3_fused_gdn_args &args,
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
    if (hv >= args.v_heads || j >= 64u || args.head_dim != 128u) return;
    uint j1 = uint(j) + 64u;
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
    constexpr uint CACHE_N = 64u;
    float2 cached[CACHE_N];
    float sk0 = 0.0f, sk1 = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 128u; i++) {
        const float prior0 = sin[i * args.head_dim + j];
        const float prior1 = sin[i * args.head_dim + j1];
        if (i < CACHE_N) cached[i] = float2(prior0, prior1);
        sk0 = fma(prior0, kh[i], sk0);
        sk1 = fma(prior1, kh[i], sk1);
    }
    float d0 = b * fma(-sk0, g, vh[j]);
    float d1 = b * fma(-sk1, g, vh[j1]);
    float sum0 = 0.0f, sum1 = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 128u; i++) {
        uint idx = i * args.head_dim + j;
        float sv0 = fma(kh[i], d0, (i < CACHE_N ? cached[i].x : sin[idx]) * g);
        float sv1 = fma(kh[i], d1, (i < CACHE_N ? cached[i].y : sin[idx + 64u]) * g);
        sout[idx] = sv0;
        sout[idx + 64u] = sv1;
        sum0 = fma(sv0, qh[i], sum0);
        sum1 = fma(sv1, qh[i], sum1);
    }
    float core0 = sum0 * rsqrt(float(args.head_dim));
    float core1 = sum1 * rsqrt(float(args.head_dim));
    float ss = simd_sum(core0 * core0);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss1 = simd_sum(core1 * core1);
    if (lane == 0) sh[simd_idx + 2u] = ss1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < ((args.head_dim + 31u) / 32u) ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.head_dim) + args.eps);
    float zi0 = scratch[args.z_offset + uint64_t(hv) * args.head_dim + j];
    float zi1 = scratch[args.z_offset + uint64_t(hv) * args.head_dim + j1];
    float gate0 = zi0 / (1.0f + exp(-zi0));
    float gate1 = zi1 / (1.0f + exp(-zi1));
    inner_out[uint64_t(hv) * args.head_dim + j] = core0 * scale * w[j] * gate0;
    inner_out[uint64_t(hv) * args.head_dim + j1] = core1 * scale * w[j1] * gate1;
}
