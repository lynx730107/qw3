// Same per-output FP32 accumulation and SIMD reduction as batch4; share input.
struct micro_batch_pair_args {
    uint n_in, n_out, row_bytes, n_tokens, in_stride, out_stride, out_a, out_b;
};
struct micro_conv_norm_args {
    uint n_channels,n_tokens,qkv_offset,conv_offset,stride,q_heads,head_dim;
    float eps;
};
kernel void micro_conv_norm(constant micro_conv_norm_args &args,
    device const float *w, device float *scratch, device const float *state,
    threadgroup float *sh, uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]], ushort simd_idx [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint t=group.y, ch=group.x*args.head_dim+uint(tid);
    float y=0.0f;
    if (uint(tid)<args.head_dim && ch<args.n_channels) {
        device const float *wr=w+uint64_t(ch)*4ull;
        device const float *st=state+uint64_t(ch)*3ull;
        float sum=0.0f;
        for (int k=0;k<4;k++) {
            int idx=int(t)+k-3;
            float xv=idx<0 ? st[3+idx] : scratch[uint64_t(uint(idx))*args.stride+args.qkv_offset+ch];
            sum+=xv*wr[k];
        }
        y=sum/(1.0f+exp(-sum));
    }
    if (group.x<2u*args.q_heads) {
        float ss=0.0f;
        if (uint(tid)<args.head_dim) ss+=y*y;
        ss=simd_sum(ss);
        if (lane==0) sh[simd_idx]=ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ss=lane<32 ? sh[lane] : 0.0f;
        ss=simd_sum(ss);
        float scale=1.0f/max(sqrt(ss),args.eps);
        y*=scale;
    }
    if (uint(tid)<args.head_dim && ch<args.n_channels)
        scratch[uint64_t(t)*args.stride+args.conv_offset+ch]=y;
}
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
// Two independent outputs in one dispatch; original matvec arithmetic verbatim.
kernel void micro_q8_decode_pair(constant qw3_matvec_q8_0_pair_args &args,
                            device const uchar *weights_a,
                            device const uchar *weights_b,
                            device const float *x,
                            device float *out,
                            threadgroup float *sh,
                            uint group [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort simd_idx [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]],
                            ushort nt [[threads_per_threadgroup]]) {
    uint which = group / args.n_out;
    uint row = group - which * args.n_out;
    device const uchar *weights = which ? weights_b : weights_a;
    out += which ? args.out_b_offset : args.out_a_offset;
    if (row >= args.n_out) return;
    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);
    float sum = 0.0f;
    uint n_blocks = args.n_in / 32;
    for (uint b = tid; b < n_blocks; b += nt) {
        device const uchar *blk = wr + b * 34;
        half d = *((device const half *)blk);
        for (uint i = 0; i < 32; i++) {
            char q = *((device const char *)(blk + 2 + i));
            sum += float(d) * float(q) * x[b * 32 + i];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) sh[simd_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane < ((uint(nt) + 31u) / 32u) ? sh[lane] : 0.0f;
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}
