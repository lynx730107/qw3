/* Original declarations and cache helper come from production source. */
kernel void private_gqa_barrier1(constant qw3_gqa_n_args &args,
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
    if (group_heads == 0u || group_heads > 7u) return;
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
            if (lane == 0 && gh < group_heads) sh[gh * 8u + uint(simd_idx)] = part;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint gh = 0; gh < 8u; gh++) {
            float dot = (gh < group_heads && uint(tid) < n_simd) ? sh[gh * 8u + uint(tid)] : 0.0f;
            dot = simd_sum(dot);
            if (tid == 0 && gh < group_heads) sh[56u + gh] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float vv = (i < args.head_dim) ? qw3_gqa_cache_load(v_cache, v_cache_f16, kv_idx, args.kv_type) : 0.0f;
        for (uint gh = 0; gh < 8u; gh++) {
            if (gh < group_heads) {
                float score = sh[56u + gh] * scale;
                float next_max = max(max_score[gh], score);
                float prev_scale = exp(max_score[gh] - next_max);
                float cur_scale = exp(score - next_max);
                acc[gh] = acc[gh] * prev_scale + vv * cur_scale;
                denom[gh] = denom[gh] * prev_scale + cur_scale;
                max_score[gh] = next_max;
            }
        }
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
kernel void private_gqa_barrier2(constant qw3_gqa_n_args &args,
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
    if (group_heads == 0u || group_heads > 7u) return;
    uint first_qh = kvh * group_heads;
    float scale = rsqrt(float(args.head_dim));
    float qv[8];
    float max_score[8];
    float denom[8];
    float acc[8];
    for (uint gh = 0; gh < group_heads; gh++) {
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
        for (uint gh = 0; gh < group_heads; gh++) {
            float part = (gh < group_heads && i < args.head_dim) ? qv[gh] * kval : 0.0f;
            part = simd_sum(part);
            if (lane == 0 && gh < group_heads) sh[gh * 8u + uint(simd_idx)] = part;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint gh = 0; gh < group_heads; gh++) {
            float dot = (gh < group_heads && uint(tid) < n_simd) ? sh[gh * 8u + uint(tid)] : 0.0f;
            dot = simd_sum(dot);
            if (tid == 0 && gh < group_heads) sh[56u + gh] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float vv = (i < args.head_dim) ? qw3_gqa_cache_load(v_cache, v_cache_f16, kv_idx, args.kv_type) : 0.0f;
        for (uint gh = 0; gh < group_heads; gh++) {
            if (gh < group_heads) {
                float score = sh[56u + gh] * scale;
                float next_max = max(max_score[gh], score);
                float prev_scale = exp(max_score[gh] - next_max);
                float cur_scale = exp(score - next_max);
                acc[gh] = acc[gh] * prev_scale + vv * cur_scale;
                denom[gh] = denom[gh] * prev_scale + cur_scale;
                max_score[gh] = next_max;
            }
        }
    }
    if (i < args.head_dim) {
        for (uint gh = 0; gh < group_heads; gh++) {
            uint qh = first_qh + gh;
            if (gh < group_heads && qh < args.n_heads) {
                uint gid = qh * args.head_dim + i;
                float sig = 1.0f / (1.0f + exp(-gate[gid]));
                out[gid] = (acc[gh] / denom[gh]) * sig;
            }
        }
    }
}
