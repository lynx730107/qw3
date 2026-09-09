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
    threadgroup float vals[256];
    threadgroup uint idxs[256];
    uint idx = block * 256u + uint(tid);
    float v = idx < args.n ? x[idx] : -FLT_MAX;
    if (idx < args.n && args.apply_penalty && seen[idx] != 0)
        v = qw3_repeat_penalty_logit(v, args.repeat_penalty);
    vals[tid] = v;
    idxs[tid] = idx < args.n ? idx : 0xffffffffu;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each comparator owns both entries, so one barrier per stage suffices.
    for (uint size = 2u; size <= 256u; size <<= 1u) {
        for (uint stride = size >> 1u; stride > 0u; stride >>= 1u) {
            uint partner = uint(tid) ^ stride;
            if (partner > uint(tid)) {
                float a = vals[tid], b = vals[partner];
                uint ai = idxs[tid], bi = idxs[partner];
                bool b_first = b > a || (b == a && bi < ai);
                bool descending = (uint(tid) & size) == 0u;
                if (b_first == descending) {
                    vals[tid] = b; idxs[tid] = bi;
                    vals[partner] = a; idxs[partner] = ai;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    uint k = min(args.k, 64u);
    if (uint(tid) < k) {
        out_vals[block * k + tid] = vals[tid];
        out_idxs[block * k + tid] = idxs[tid];
    }
}
