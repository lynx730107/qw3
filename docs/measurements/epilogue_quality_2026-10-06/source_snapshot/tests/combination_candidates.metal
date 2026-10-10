struct combo_norm_args { uint n; float eps; uint rows; uint flags; };
kernel void combo_residual_norm_round(
    constant combo_norm_args &a,
    device float *x0, device const float *residual, device const float *w,
    device float *x1, threadgroup float *sh,
    uint row [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd_idx [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort nt [[threads_per_threadgroup]]) {
    if (row >= a.rows) return;
    uint64_t base = uint64_t(row) * a.n;
    float ss = 0.0f;
    for (uint i=tid; i<a.n; i+=nt) {
        float v = x0[base+i];
        if (a.flags & 1u) v = v + residual[base+i];
        ss += v*v;
    }
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < ((uint(nt)+31u)/32u) ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(a.n) + a.eps);
    for (uint i=tid; i<a.n; i+=nt) {
        float v = x0[base+i];
        if (a.flags & 1u) { v = v + residual[base+i]; x0[base+i] = v; }
        float y = v * scale * w[i];
        x1[base+i] = (a.flags & 2u) ? float(half(y)) : y;
    }
}
