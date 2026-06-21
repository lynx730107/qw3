struct qw3_argmax_args { uint n; };
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
