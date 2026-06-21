struct qw3_argmax_args { uint n; };
struct qw3_argmax_penalty_args { uint n; float repeat_penalty; };
struct qw3_topk_penalty_args {
    uint n;
    uint k;
    float repeat_penalty;
    uint apply_penalty;
};

static inline float qw3_repeat_penalty_logit(float v, float penalty) {
    return v < 0.0f ? v * penalty : v / penalty;
}

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

kernel void qw3_argmax_penalty_blocks(constant qw3_argmax_penalty_args &args,
                                      device const float *x,
                                      device const uchar *seen,
                                      device float *out_vals,
                                      device uint *out_idxs,
                                      threadgroup float *sh_vals,
                                      threadgroup uint *sh_idxs,
                                      uint block [[threadgroup_position_in_grid]],
                                      ushort tid [[thread_index_in_threadgroup]],
                                      ushort nt [[threads_per_threadgroup]]) {
    uint idx = block * uint(nt) + uint(tid);
    float v = -FLT_MAX;
    if (idx < args.n) {
        v = x[idx];
        if (seen[idx] != 0) {
            v = qw3_repeat_penalty_logit(v, args.repeat_penalty);
        }
    }
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

kernel void qw3_topk_penalty_blocks(constant qw3_topk_penalty_args &args,
                                    device const float *x,
                                    device const uchar *seen,
                                    device float *out_vals,
                                    device uint *out_idxs,
                                    uint block [[threadgroup_position_in_grid]],
                                    ushort tid [[thread_index_in_threadgroup]]) {
    if (tid != 0) return;
    constexpr uint max_k = 64;
    float vals[max_k];
    uint idxs[max_k];
    uint k = args.k > max_k ? max_k : args.k;
    for (uint i = 0; i < max_k; i++) {
        vals[i] = -FLT_MAX;
        idxs[i] = 0xffffffffu;
    }
    uint start = block * 256u;
    uint end = start + 256u;
    if (end > args.n) end = args.n;
    for (uint idx = start; idx < end; idx++) {
        float v = x[idx];
        if (args.apply_penalty && seen[idx] != 0) {
            v = qw3_repeat_penalty_logit(v, args.repeat_penalty);
        }
        for (uint j = 0; j < k; j++) {
            if (v > vals[j] || (v == vals[j] && idx < idxs[j])) {
                for (uint m = k - 1u; m > j; m--) {
                    vals[m] = vals[m - 1u];
                    idxs[m] = idxs[m - 1u];
                }
                vals[j] = v;
                idxs[j] = idx;
                break;
            }
        }
    }
    uint base = block * k;
    for (uint i = 0; i < k; i++) {
        out_vals[base + i] = vals[i];
        out_idxs[base + i] = idxs[i];
    }
}
