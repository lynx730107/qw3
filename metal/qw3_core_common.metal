#include <metal_stdlib>
#ifdef QW3_METAL_HAS_TENSOR
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;
#ifdef QW3_METAL_HAS_TENSOR
using namespace mpp::tensor_ops;
#endif
constant float qw3_iq4nl_table[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f,
     -49.0f,  -35.0f, -22.0f, -10.0f,
       1.0f,   13.0f,  25.0f,  38.0f,
      53.0f,   69.0f,  89.0f, 113.0f
};
inline float qw3_f16_to_f32(ushort h) {
    uint s = uint(h >> 15u);
    uint e = (uint(h) >> 10u) & 31u;
    uint f = uint(h) & 1023u;
    float sign = s ? -1.0f : 1.0f;
    if (e == 0u) return f == 0u ? sign * 0.0f : sign * float(f) * exp2(-24.0f);
    if (e == 31u) return f == 0u ? sign * INFINITY : NAN;
    return sign * (1.0f + float(f) / 1024.0f) * exp2(float(e) - 15.0f);
}
struct qw3_rmsnorm_args { uint n; float eps; };
kernel void qw3_rmsnorm_plain(constant qw3_rmsnorm_args &args,
                              device const float *x,
                              device float *y,
                              threadgroup float *sh,
                              ushort tid [[thread_index_in_threadgroup]],
                              ushort simd_idx [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]],
                              ushort nt [[threads_per_threadgroup]]) {
    float ss = 0.0f;
    for (uint i = tid; i < args.n; i += nt) ss += x[i] * x[i];
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.n) + args.eps);
    for (uint i = tid; i < args.n; i += nt) y[i] = x[i] * scale;
}
kernel void qw3_rmsnorm_weight_f32(constant qw3_rmsnorm_args &args,
                                   device const float *x,
                                   device const float *w,
                                   device float *y,
                                   threadgroup float *sh,
                                   ushort tid [[thread_index_in_threadgroup]],
                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]],
                                   ushort nt [[threads_per_threadgroup]]) {
    float ss = 0.0f;
    for (uint i = tid; i < args.n; i += nt) ss += x[i] * x[i];
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.n) + args.eps);
    for (uint i = tid; i < args.n; i += nt) y[i] = x[i] * scale * w[i];
}
struct qw3_rmsnorm_rows_args { uint n; float eps; uint n_rows; };
kernel void qw3_rmsnorm_weight_f32_rows(constant qw3_rmsnorm_rows_args &args,
                                        device float *x,
                                        device const float *w,
                                        threadgroup float *sh,
                                        uint row [[threadgroup_position_in_grid]],
                                        ushort tid [[thread_index_in_threadgroup]],
                                        ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                        ushort lane [[thread_index_in_simdgroup]],
                                        ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_rows) return;
    device float *xr = x + uint64_t(row) * args.n;
    float ss = 0.0f;
    for (uint i = tid; i < args.n; i += nt) ss += xr[i] * xr[i];
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.n) + args.eps);
    for (uint i = tid; i < args.n; i += nt) xr[i] = xr[i] * scale * w[i];
}
kernel void qw3_rmsnorm_weight_f32_rows_to_out(constant qw3_rmsnorm_rows_args &args,
                                               device const float *x,
                                               device const float *w,
                                               device float *out,
                                               threadgroup float *sh,
                                               uint row [[threadgroup_position_in_grid]],
                                               ushort tid [[thread_index_in_threadgroup]],
                                               ushort simd_idx [[simdgroup_index_in_threadgroup]],
                                               ushort lane [[thread_index_in_simdgroup]],
                                               ushort nt [[threads_per_threadgroup]]) {
    if (row >= args.n_rows) return;
    device const float *xr = x + uint64_t(row) * args.n;
    device float *yr = out + uint64_t(row) * args.n;
    float ss = 0.0f;
    for (uint i = tid; i < args.n; i += nt) ss += xr[i] * xr[i];
    ss = simd_sum(ss);
    if (lane == 0) sh[simd_idx] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = lane < 32 ? sh[lane] : 0.0f;
    ss = simd_sum(ss);
    float scale = rsqrt(ss / float(args.n) + args.eps);
    for (uint i = tid; i < args.n; i += nt) yr[i] = xr[i] * scale * w[i];
}
struct qw3_embed_q8_0_args { uint n_embd; uint row_bytes; };
kernel void qw3_embed_q8_0(constant qw3_embed_q8_0_args &args,
                           device const uchar *weights,
                           device float *out,
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n_embd) return;
    uint block = gid / 32;
    uint lane = gid % 32;
    device const uchar *blk = weights + block * 34;
    half d = *((device const half *)blk);
    char q = *((device const char *)(blk + 2 + lane));
    out[gid] = float(d) * float(q);
}
struct qw3_embed_q8_0_batch_args { uint n_embd; uint row_bytes; uint n_tokens; };
kernel void qw3_embed_q8_0_batch(constant qw3_embed_q8_0_batch_args &args,
                                 device const uchar *weights,
                                 device const uint *tokens,
                                 device float *out,
                                 uint gid [[thread_position_in_grid]]) {
    uint total = args.n_tokens * args.n_embd;
    if (gid >= total) return;
    uint t = gid / args.n_embd;
    uint i = gid - t * args.n_embd;
    uint token = tokens[t];
    device const uchar *row = weights + uint64_t(token) * uint64_t(args.row_bytes);
    uint block = i / 32u;
    uint lane = i & 31u;
    device const uchar *blk = row + uint64_t(block) * 34ull;
    float d = float(*((device const half *)blk));
    char q = *((device const char *)(blk + 2u + lane));
    out[uint64_t(t) * args.n_embd + i] = d * float(q);
}
struct qw3_kv_quant_q8_args { uint n; };
kernel void qw3_gqa_kv_quant_q8(constant qw3_kv_quant_q8_args &args,
                                device const float *k,
                                device const float *v,
                                device uchar *k_cache,
                                device uchar *v_cache,
                                uint block [[threadgroup_position_in_grid]],
                                ushort tid [[thread_index_in_threadgroup]]) {
    uint base = block * 32u;
    if (base >= args.n || tid >= 32u) return;
    float ka = fabs(k[base + uint(tid)]);
    float va = fabs(v[base + uint(tid)]);
    ka = simd_max(ka);
    va = simd_max(va);
    float kd = ka > 0.0f ? ka / 127.0f : 0.0f;
    float vd = va > 0.0f ? va / 127.0f : 0.0f;
    device uchar *kb = k_cache + uint64_t(block) * 34ull;
    device uchar *vb = v_cache + uint64_t(block) * 34ull;
    if (tid == 0) {
        *((device half *)kb) = half(kd);
        *((device half *)vb) = half(vd);
    }
    float kq = kd > 0.0f ? rint(k[base + uint(tid)] / kd) : 0.0f;
    float vq = vd > 0.0f ? rint(v[base + uint(tid)] / vd) : 0.0f;
    *((device char *)(kb + 2u + uint(tid))) = char(clamp(kq, -127.0f, 127.0f));
    *((device char *)(vb + 2u + uint(tid))) = char(clamp(vq, -127.0f, 127.0f));
}
kernel void qw3_gqa_store_token_cache_f16(constant qw3_kv_quant_q8_args &args,
                                          device const float *k,
                                          device const float *v,
                                          device half *k_cache,
                                          device half *v_cache,
                                          uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n) return;
    k_cache[gid] = half(k[gid]);
    v_cache[gid] = half(v[gid]);
}
