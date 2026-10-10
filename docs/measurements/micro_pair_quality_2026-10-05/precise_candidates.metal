// Same per-output FP32 accumulation and SIMD reduction as batch4; share input.
struct micro_batch_pair_args {
    uint n_in, n_out, row_bytes, n_tokens, in_stride, out_stride, out_a, out_b;
};
kernel void micro_q8_batch_pair(constant micro_batch_pair_args &args,
    device const uchar *wa, device const uchar *wb, device const float *x,
    device float *out, uint2 group [[threadgroup_position_in_grid]],
    ushort simd_idx [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint row = group.x * 4u + uint(simd_idx);
    if (row >= args.n_out) return;
    uint t0 = group.y * 4u;
    device const uchar *wra = wa + uint64_t(row) * args.row_bytes;
    device const uchar *wrb = wb + uint64_t(row) * args.row_bytes;
    float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;
    float b0=0.0f,b1=0.0f,b2=0.0f,b3=0.0f;
    for (uint b=uint(lane); b<args.n_in/32u; b+=32u) {
        device const uchar *ba=wra+uint64_t(b)*34ull;
        device const uchar *bb=wrb+uint64_t(b)*34ull;
        float da=float(*((device const half *)ba));
        float db=float(*((device const half *)bb));
        for (uint i=0;i<32u;i++) {
            float av=da*float(*((device const char *)(ba+2u+i)));
            float bv=db*float(*((device const char *)(bb+2u+i)));
            uint xi=b*32u+i;
            if (t0+0u<args.n_tokens) { float xv=x[uint64_t(t0+0u)*args.in_stride+xi]; a0+=av*xv; b0+=bv*xv; }
            if (t0+1u<args.n_tokens) { float xv=x[uint64_t(t0+1u)*args.in_stride+xi]; a1+=av*xv; b1+=bv*xv; }
            if (t0+2u<args.n_tokens) { float xv=x[uint64_t(t0+2u)*args.in_stride+xi]; a2+=av*xv; b2+=bv*xv; }
            if (t0+3u<args.n_tokens) { float xv=x[uint64_t(t0+3u)*args.in_stride+xi]; a3+=av*xv; b3+=bv*xv; }
        }
    }
    a0=simd_sum(a0); a1=simd_sum(a1); a2=simd_sum(a2); a3=simd_sum(a3);
    b0=simd_sum(b0); b1=simd_sum(b1); b2=simd_sum(b2); b3=simd_sum(b3);
    if (lane==0) {
        if (t0+0u<args.n_tokens) { out[uint64_t(t0+0u)*args.out_stride+args.out_a+row]=a0; out[uint64_t(t0+0u)*args.out_stride+args.out_b+row]=b0; }
        if (t0+1u<args.n_tokens) { out[uint64_t(t0+1u)*args.out_stride+args.out_a+row]=a1; out[uint64_t(t0+1u)*args.out_stride+args.out_b+row]=b1; }
        if (t0+2u<args.n_tokens) { out[uint64_t(t0+2u)*args.out_stride+args.out_a+row]=a2; out[uint64_t(t0+2u)*args.out_stride+args.out_b+row]=b2; }
        if (t0+3u<args.n_tokens) { out[uint64_t(t0+3u)*args.out_stride+args.out_a+row]=a3; out[uint64_t(t0+3u)*args.out_stride+args.out_b+row]=b3; }
    }
}
// Match the separate matvec's thread count and mask unused SIMD sums.
#pragma clang fp reassociate(off)
kernel void micro_q8_decode_pair(constant qw3_matvec_q8_0_pair_args &args,
    device const uchar *wa, device const uchar *wb, device const float *x,
    device float *out, threadgroup float *sh,
    uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort simd_idx [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    ushort nt [[threads_per_threadgroup]]) {
    if (row>=args.n_out) return;
    device const uchar *wra=wa+uint64_t(row)*args.row_bytes;
    device const uchar *wrb=wb+uint64_t(row)*args.row_bytes;
    float a=0.0f,b=0.0f;
    for (uint block=tid;block<args.n_in/32u;block+=nt) {
        device const uchar *ba=wra+block*34u;
        device const uchar *bb=wrb+block*34u;
        half da=*((device const half *)ba),db=*((device const half *)bb);
        for (uint i=0;i<32u;i++) {
            char qa=*((device const char *)(ba+2u+i)),qb=*((device const char *)(bb+2u+i));
            float xv=x[block*32u+i];
            a+=float(da)*float(qa)*xv;
            b+=float(db)*float(qb)*xv;
        }
    }
    a=simd_sum(a); b=simd_sum(b);
    if (lane==0) { sh[simd_idx]=a; sh[simd_idx+32u]=b; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    a=lane<((uint(nt)+31u)/32u) ? sh[lane] : 0.0f;
    b=lane<((uint(nt)+31u)/32u) ? sh[lane+32u] : 0.0f;
    a=simd_sum(a); b=simd_sum(b);
    if (tid==0) { out[args.out_a_offset+row]=a; out[args.out_b_offset+row]=b; }
}
