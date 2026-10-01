#include <metal_stdlib>
using namespace metal;
// Isolated experiments: no precision change or reassociated FP32 dot products.
struct iq3_args { uint n_in, n_out, row_bytes; };
inline uchar iq3_magnitude(ushort packed, uint j) {
    constexpr uchar values[8] = {4,12,20,28,36,44,52,62};
    return values[(uint(packed) >> (3u*j)) & 7u];
}
struct iq3_payload { float d; ushort q[4]; uint signs; };
template<bool WordLoads>
inline iq3_payload iq3_load(device const uchar *blk, uint ib) {
    iq3_payload p;
    p.d = float(*reinterpret_cast<device const half *>(blk));
    device const uchar *q = blk + 2u + 8u*ib;
    device const uchar *sp = blk + 66u + 4u*ib;
    // GGUF IQ3 blocks/rows are two-byte aligned, not necessarily four-byte
    // aligned. Do not issue uint/ushort2 vector reads from this payload.
    if (WordLoads) {
        device const ushort *qw = reinterpret_cast<device const ushort *>(q);
        device const ushort *sw = reinterpret_cast<device const ushort *>(sp);
        for (uint l=0;l<4u;l++) p.q[l]=qw[l];
        p.signs=uint(sw[0]) | (uint(sw[1]) << 16u);
    } else {
        for (uint l=0;l<4u;l++) p.q[l]=ushort(q[2u*l]) | (ushort(q[2u*l+1u]) << 8u);
        p.signs=uint(sp[0]) | (uint(sp[1]) << 8u) | (uint(sp[2]) << 16u) | (uint(sp[3]) << 24u);
    }
    return p;
}
template<class Grid>
inline float iq3_dot(thread const iq3_payload &p, thread const float4 *xx,
                     threadgroup const Grid *grid, threadgroup const uchar *signs) {
    const float db=p.d*(0.5f+float(p.signs >> 28u))*0.5f;
    float sum=0.0f;
    for (uint l=0;l<4u;l++) {
        const uint s=uint(signs[(p.signs >> (7u*l)) & 127u]);
        const float4 sign1=select(float4(1.0f),float4(-1.0f),
                                 (uint4(s)&uint4(1u,2u,4u,8u))!=0u);
        const float4 sign2=select(float4(1.0f),float4(-1.0f),
                                 (uint4(s)&uint4(16u,32u,64u,128u))!=0u);
        sum+=dot(float4(grid[uint(p.q[l])&255u])*sign1,xx[2u*l])+
             dot(float4(grid[uint(p.q[l])>>8u])*sign2,xx[2u*l+1u]);
    }
    return db*sum;
}
template<class Grid, bool WordLoads, bool Prefetch>
kernel void iq3_stream(constant iq3_args &args [[buffer(0)]],
    device const uchar *weights [[buffer(1)]], device const float *x [[buffer(2)]],
    device float *out [[buffer(3)]], device const ushort *grid_table [[buffer(4)]],
    device const uchar *sign_table [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    ushort simd_idx [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup Grid grid[256];
    threadgroup uchar signs[128];
    const uint tid=uint(simd_idx)*32u+uint(lane);
    for (uint i=tid;i<256u;i+=64u) {
        const ushort packed=grid_table[i];
        grid[i]=Grid(iq3_magnitude(packed,0u),iq3_magnitude(packed,1u),
                     iq3_magnitude(packed,2u),iq3_magnitude(packed,3u));
    }
    for (uint i=tid;i<128u;i+=64u) signs[i]=sign_table[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row0=(group*2u+uint(simd_idx))*2u,row1=row0+1u,nb32=args.n_in/32u;
    device const uchar *w0=weights+uint64_t(row0)*args.row_bytes;
    device const uchar *w1=weights+uint64_t(row1)*args.row_bytes;
    float sum0=0.0f,sum1=0.0f;
    iq3_payload next0={},next1={};
    if (Prefetch && uint(lane)<nb32) {
        if (row0<args.n_out) next0=iq3_load<WordLoads>(w0+uint64_t(uint(lane)/8u)*98u,uint(lane)&7u);
        if (row1<args.n_out) next1=iq3_load<WordLoads>(w1+uint64_t(uint(lane)/8u)*98u,uint(lane)&7u);
    }
    for (uint b32=uint(lane);b32<nb32;b32+=32u) {
        const uint b=b32/8u,ib=b32&7u;
        float4 xx[8];
        device const float4 *x4=reinterpret_cast<device const float4 *>(x+uint64_t(b32)*32u);
        for (uint j=0;j<8u;j++) xx[j]=x4[j];
        if (Prefetch) {
            iq3_payload now0=next0,now1=next1;
            if (b32+32u<nb32) {
                if (row0<args.n_out) next0=iq3_load<WordLoads>(w0+uint64_t(b+4u)*98u,ib);
                if (row1<args.n_out) next1=iq3_load<WordLoads>(w1+uint64_t(b+4u)*98u,ib);
            }
            if (row0<args.n_out) sum0+=iq3_dot(now0,xx,grid,signs);
            if (row1<args.n_out) sum1+=iq3_dot(now1,xx,grid,signs);
        } else {
            if (row0<args.n_out) { const auto p=iq3_load<WordLoads>(w0+uint64_t(b)*98u,ib); sum0+=iq3_dot(p,xx,grid,signs); }
            if (row1<args.n_out) { const auto p=iq3_load<WordLoads>(w1+uint64_t(b)*98u,ib); sum1+=iq3_dot(p,xx,grid,signs); }
        }
    }
    sum0=simd_sum(sum0);sum1=simd_sum(sum1);
    if (lane==0u) { if (row0<args.n_out) out[row0]=sum0; if (row1<args.n_out) out[row1]=sum1; }
}
typedef decltype(iq3_stream<uchar4,false,false>) iq3_stream_t;
template [[host_name("iq3_bytes")]] kernel iq3_stream_t iq3_stream<uchar4,false,false>;
template [[host_name("iq3_words")]] kernel iq3_stream_t iq3_stream<uchar4,true,false>;
template [[host_name("iq3_prefetch")]] kernel iq3_stream_t iq3_stream<uchar4,true,true>;
template [[host_name("iq3_float_prefetch")]] kernel iq3_stream_t iq3_stream<float4,true,true>;
template [[host_name("iq3_float_words")]] kernel iq3_stream_t iq3_stream<float4,true,false>;

// Exact byte codebook for the larger IQ3_S gate/up workload (8 KiB -> 2 KiB).
inline float iq3s_byte_dot(device const uchar *blk,thread const float4 *xx,
                          threadgroup const uchar4 *grid,uint ib) {
    const float d=float(*reinterpret_cast<device const half *>(blk));
    device const uchar *qs=blk+2u+ib*8u;
    const uint qhb=uint(blk[66u+ib]),scales=uint(blk[106u+ib/2u]);
    const float db=d*float(1u+2u*((scales>>(4u*(ib&1u)))&15u));
    float sum=0.0f;
    for (uint l=0;l<4u;l++) {
        const uint idx1=uint(qs[2u*l])|((qhb<<(8u-2u*l))&256u);
        const uint idx2=uint(qs[2u*l+1u])|((qhb<<(7u-2u*l))&256u);
        const uint s=uint(blk[74u+ib*4u+l]);
        const float4 sign1=select(float4(1.0f),float4(-1.0f),(uint4(s)&uint4(1u,2u,4u,8u))!=0u);
        const float4 sign2=select(float4(1.0f),float4(-1.0f),(uint4(s)&uint4(16u,32u,64u,128u))!=0u);
        sum+=dot(float4(grid[idx1])*sign1,xx[2u*l])+dot(float4(grid[idx2])*sign2,xx[2u*l+1u]);
    }
    return db*sum;
}
kernel void iq3s_bytes(constant iq3_args &args [[buffer(0)]],
    device const uchar *gate_weights [[buffer(1)]],device const uchar *up_weights [[buffer(2)]],
    device const float *x [[buffer(3)]],device float *out [[buffer(4)]],
    device const ushort *kgrid [[buffer(5)]],uint group [[threadgroup_position_in_grid]],
    ushort simd_idx [[simdgroup_index_in_threadgroup]],ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup uchar4 grid[512];
    const uint tid=uint(simd_idx)*32u+uint(lane);
    for (uint i=tid;i<512u;i+=64u) {
        const uint g=uint(kgrid[i]);
        grid[i]=uchar4(2u*(g&7u)+1u,2u*((g>>3u)&7u)+1u,2u*((g>>6u)&7u)+1u,2u*((g>>9u)&7u)+1u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row0=(group*2u+uint(simd_idx))*4u;
    float gate_sum[4]={0,0,0,0},up_sum[4]={0,0,0,0};
    for (uint b32=uint(lane);b32<args.n_in/32u;b32+=32u) {
        float4 xx[8];
        device const float4 *x4=reinterpret_cast<device const float4 *>(x+uint64_t(b32)*32u);
        for (uint j=0;j<8u;j++) xx[j]=x4[j];
#pragma unroll
        for (uint r=0;r<4u;r++) if (row0+r<args.n_out) {
            const uint64_t offset=uint64_t(row0+r)*args.row_bytes+uint64_t(b32/8u)*110u;
            gate_sum[r]+=iq3s_byte_dot(gate_weights+offset,xx,grid,b32&7u);
            up_sum[r]+=iq3s_byte_dot(up_weights+offset,xx,grid,b32&7u);
        }
    }
#pragma unroll
    for (uint r=0;r<4u;r++) {
        const float gate=simd_sum(gate_sum[r]),up=simd_sum(up_sum[r]);
        if (lane==0u && row0+r<args.n_out) {out[row0+r]=gate;out[args.n_out+row0+r]=up;}
    }
}
