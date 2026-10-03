/* Private short-verifier kernels: no half activation/weight tiles.
 * One SIMD group/output row shares quantized weights across up to four tokens.
 * Offsets/strides are float elements and partial rows preserve output guards. */
#include <metal_stdlib>
using namespace metal;
struct mtp_gdn_args {uint qh,vh,hd,nt,conv,z,alpha,beta,inner,stride; float eps;};
/* Private 128-wide register-state trial. Preserve scalar ordered column
 * reductions, but retain a column between tokens instead of rereading state. */
kernel void mtp_gdn_cached128(
    constant mtp_gdn_args &a [[buffer(0)]],
    device float *state [[buffer(1)]],
    device float *scratch [[buffer(2)]],
    device const float *dt [[buffer(3)]],
    device const float *decay [[buffer(4)]],
    device const float *weight [[buffer(5)]],
    threadgroup float *sh [[threadgroup(0)]],
    uint h [[threadgroup_position_in_grid]],
    ushort j [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (a.hd != 128u || h >= a.vh || j >= 128u) return;
    device float *st = state+uint64_t(h)*128u*128u;
    float column[128];
    #pragma unroll
    for (uint i = 0; i < 128u; i++) column[i] = st[i*128u+j];
    for (uint t = 0; t < a.nt; t++) {
        const uint64_t base = uint64_t(t)*a.stride;
        device const float *q = scratch+base+a.conv+(h%a.qh)*128u;
        device const float *k = q+a.qh*128u;
        device const float *v = scratch+base+a.conv+2u*a.qh*128u+h*128u;
        if (j == 0) {
            const float alpha = scratch[base+a.alpha+h]+dt[h];
            const float sp = alpha>20 ? alpha : alpha < -20 ? exp(alpha) : log(1.0f+exp(alpha));
            sh[0] = 1.0f/(1.0f+exp(-scratch[base+a.beta+h]));
            sh[1] = exp(sp*decay[h]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float b = sh[0], g = sh[1];
        float sk = 0.0f;
        #pragma unroll
        for (uint i = 0; i < 128u; i++) sk += column[i]*k[i];
        const float d = b*(v[j]-sk*g);
        float sum = 0.0f;
        #pragma unroll
        for (uint i = 0; i < 128u; i++) {
            column[i] = column[i]*g+k[i]*d;
            sum += column[i]*q[i];
        }
        const float core = sum*rsqrt(128.0f);
        float ss = simd_sum(core*core);
        if (lane == 0) sh[2u+sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ss = simd_sum(lane < 4u ? sh[2u+lane] : 0.0f);
        const float scale = rsqrt(ss/128.0f+a.eps);
        const float z = scratch[base+a.z+h*128u+j];
        scratch[base+a.inner+h*128u+j] = core*scale*weight[j]*(z/(1.0f+exp(-z)));
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for (uint i = 0; i < 128u; i++) st[i*128u+j] = column[i];
}

struct mtp_short_args {
    uint ni, no, rb, nt, io, is, oo, os;
};

/* Match ordinary Q5_K GEMV's per-token FP32 accumulation order, while
 * decoding each weight once for up to four token rows. No half tiles. */
template <uint Tokens>
inline void mtp_q5k_head_rows(constant mtp_short_args &a,
    device const uchar *weights, device const float *x, device float *y,
    uint2 group, ushort sg, ushort lane) {
    const uint row = group.x*2u+sg, t0 = group.y*Tokens;
    if (row >= a.no || t0 >= a.nt) return;
    const uint count = min(Tokens,a.nt-t0);
    const uint tid = uint(lane)/4u, ix = uint(lane)&3u;
    const uint iq = tid/4u, ir = tid&3u, l0 = ir*8u;
    const uint qoff = 32u*iq+l0, xoff = 64u*iq+l0;
    const uchar hm1 = uchar(1u<<(2u*iq)), hm2 = uchar(uint(hm1)<<1u);
    const uchar hm3 = uchar(uint(hm1)<<4u), hm4 = uchar(uint(hm2)<<4u);
    float4 sums = float4(0.0f);
    for (uint ib = ix; ib < a.ni/256u; ib += 4u) {
        device const uchar *b = weights+uint64_t(row)*a.rb+uint64_t(ib)*176ull;
        device const uchar *q1 = b+48u+qoff, *q2 = q1+64u, *qh = b+16u+l0;
        device const ushort *sc = (device const ushort *)(b+4u)+iq;
        ushort sc16[4];
        sc16[0] = sc[0]&0x3f3f;
        sc16[1] = sc[2]&0x3f3f;
        sc16[2] = (sc[4]&0x0f0f)|((sc[0]&0xc0c0)>>2u);
        sc16[3] = ((sc[4]>>4u)&0x0f0f)|((sc[2]&0xc0c0)>>2u);
        thread const uchar *sc8 = (thread const uchar *)sc16;
        const float d = float(*((device const half *)b));
        const float dm = float(*((device const half *)(b+2u)));
        uchar high[8], low[8], upper[8];
        #pragma unroll
        for (uint l = 0; l < 8u; l++) { high[l]=qh[l]; low[l]=q1[l]; upper[l]=q2[l]; }
        #pragma unroll
        for (uint r = 0; r < Tokens; r++) if (r < count) {
            device const float *p = x+uint64_t(t0+r)*a.is+a.io+ib*256u+xoff;
            float yl[16], yh[16];
            float4 sumx=float4(0), acc1=float4(0), acc2=float4(0);
            #pragma unroll
            for (uint l = 0; l < 8u; l++) {
                yl[l]=p[l]; sumx.x+=yl[l];
                yl[l+8u]=p[l+32u]; sumx.y+=yl[l+8u];
                yh[l]=p[l+128u]; sumx.z+=yh[l];
                yh[l+8u]=p[l+160u]; sumx.w+=yh[l+8u];
            }
            #pragma unroll
            for (uint l = 0; l < 8u; l++) {
                const uchar h=high[l];
                acc1.x+=yl[l]*float(low[l]&15u);
                acc1.y+=yl[l+8u]*float(low[l]&240u);
                acc1.z+=yh[l]*float(upper[l]&15u);
                acc1.w+=yh[l+8u]*float(upper[l]&240u);
                acc2.x+=(h&hm1) ? yl[l] : 0.0f;
                acc2.y+=(h&hm2) ? yl[l+8u] : 0.0f;
                acc2.z+=(h&hm3) ? yh[l] : 0.0f;
                acc2.w+=(h&hm4) ? yh[l+8u] : 0.0f;
            }
            sums[r] += d*(float(sc8[0])*(acc1.x+16.0f*acc2.x)+
                float(sc8[1])*(acc1.y*(1.0f/16.0f)+16.0f*acc2.y)+
                float(sc8[4])*(acc1.z+16.0f*acc2.z)+
                float(sc8[5])*(acc1.w*(1.0f/16.0f)+16.0f*acc2.w))-
                dm*(sumx.x*float(sc8[2])+sumx.y*float(sc8[3])+
                    sumx.z*float(sc8[6])+sumx.w*float(sc8[7]));
        }
    }
    sums.x=simd_sum(sums.x); sums.y=simd_sum(sums.y);
    sums.z=simd_sum(sums.z); sums.w=simd_sum(sums.w);
    if (lane == 0) for (uint r = 0; r < count; r++)
        y[uint64_t(t0+r)*a.os+a.oo+row] = sums[r];
}

#define MTP_HEAD_KERNEL(T) \
kernel void mtp_q5k_head_short##T(constant mtp_short_args &a [[buffer(0)]], \
    device const uchar *w [[buffer(1)]], device const float *x [[buffer(2)]], \
    device float *y [[buffer(3)]], uint2 g [[threadgroup_position_in_grid]], \
    ushort sg [[simdgroup_index_in_threadgroup]], ushort l [[thread_index_in_simdgroup]]) { \
    mtp_q5k_head_rows<T>(a,w,x,y,g,sg,l); \
}
MTP_HEAD_KERNEL(1)
MTP_HEAD_KERNEL(2)
MTP_HEAD_KERNEL(4)
#undef MTP_HEAD_KERNEL

kernel void mtp_iq3s_short(
    constant mtp_short_args &a [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *y [[buffer(3)]],
    device const ushort *packed_grid [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup ushort grid[512];
    const uint tid = uint(sg)*32u+lane;
    for (uint i = tid; i < 512u; i += 128u) {
        grid[i] = packed_grid[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row = group.x*4u+sg, t0 = group.y*4u;
    if (row >= a.no || t0 >= a.nt) return;
    const uint count = min(4u,a.nt-t0);
    float4 sums = float4(0.0f);
    for (uint b32 = lane; b32 < a.ni/32u; b32 += 32u) {
        const uint ib = b32&7u;
        device const uchar *b = weights+uint64_t(row)*a.rb+uint64_t(b32/8u)*110ull;
        const uint qh = b[66u+ib], scales = b[106u+ib/2u];
        const float scale = float(*((device const half *)b))*float(1u+2u*((scales>>(4u*(ib&1u)))&15u));
        float4 dots = float4(0.0f);
        #pragma unroll
        for (uint l = 0; l < 4u; l++) {
            const uint signs = b[74u+ib*4u+l];
            const uint i0 = uint(b[2u+ib*8u+l*2u])|((qh<<(8u-2u*l))&256u);
            const uint i1 = uint(b[3u+ib*8u+l*2u])|((qh<<(7u-2u*l))&256u);
            const uint p0 = grid[i0], p1 = grid[i1];
            const float4 g0 = float4(1u+2u*(p0&7u),1u+2u*((p0>>3u)&7u),1u+2u*((p0>>6u)&7u),1u+2u*((p0>>9u)&7u));
            const float4 g1 = float4(1u+2u*(p1&7u),1u+2u*((p1>>3u)&7u),1u+2u*((p1>>6u)&7u),1u+2u*((p1>>9u)&7u));
            const float4 w0 = g0*select(float4(1),float4(-1),(uint4(signs)&uint4(1,2,4,8)) != 0u);
            const float4 w1 = g1*select(float4(1),float4(-1),(uint4(signs)&uint4(16,32,64,128)) != 0u);
            #pragma unroll
            for (uint r = 0; r < 4u; r++) if (r < count) {
                device const float *p = x+uint64_t(t0+r)*a.is+a.io+b32*32u+l*8u;
                dots[r] += dot(w0,float4(p[0],p[1],p[2],p[3]))+
                           dot(w1,float4(p[4],p[5],p[6],p[7]));
            }
        }
        sums += scale*dots;
    }
    sums.x = simd_sum(sums.x); sums.y = simd_sum(sums.y);
    sums.z = simd_sum(sums.z); sums.w = simd_sum(sums.w);
    if (lane == 0) for (uint r = 0; r < count; r++)
        y[uint64_t(t0+r)*a.os+a.oo+row] = sums[r];
}

kernel void mtp_iq4xs_short(
    constant mtp_short_args &a [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *y [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr float values[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    threadgroup float table[16];
    if (sg == 0u && lane < 16u) table[lane] = values[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row = group.x*4u+sg, t0 = group.y*4u;
    if (row >= a.no || t0 >= a.nt) return;
    const uint count = min(4u,a.nt-t0), ix = lane>>4u, it = lane&15u;
    const uint ib = it>>1u, il = it&1u;
    float4 sums = float4(0.0f);
    for (uint block = ix; block < a.ni/256u; block += 2u) {
        device const uchar *b = weights+uint64_t(row)*a.rb+uint64_t(block)*136ull;
        const uint sh = uint(b[2])|(uint(b[3])<<8u);
        const uint scale_bits = ((uint(b[4u+ib/2u])>>(4u*(ib&1u)))&15u)|(((sh>>(2u*ib))&3u)<<4u);
        const float scale = float(*((device const half *)b))*float(int(scale_bits)-32);
        float4 lo[4], hi[4];
        #pragma unroll
        for (uint r = 0; r < 4u; r++) { lo[r] = float4(0); hi[r] = float4(0); }
        #pragma unroll
        for (uint j = 0; j < 2u; j++) {
            const uint qoff = 8u+ib*16u+il*8u+j*4u;
            const uchar4 q = uchar4(b[qoff],b[qoff+1u],b[qoff+2u],b[qoff+3u]);
            const uchar4 ql = q&uchar4(15u), qh = q>>4u;
            const float4 wl = float4(table[ql.x],table[ql.y],table[ql.z],table[ql.w]);
            const float4 wh = float4(table[qh.x],table[qh.y],table[qh.z],table[qh.w]);
            #pragma unroll
            for (uint r = 0; r < 4u; r++) if (r < count) {
                device const float *p = x+uint64_t(t0+r)*a.is+a.io+block*256u+ib*32u+il*8u+j*4u;
                lo[r] += float4(p[0],p[1],p[2],p[3])*wl;
                hi[r] += float4(p[16],p[17],p[18],p[19])*wh;
            }
        }
        #pragma unroll
        for (uint r = 0; r < 4u; r++) if (r < count) {
            const float4 v = lo[r]+hi[r];
            sums[r] += scale*(v.x+v.y+v.z+v.w);
        }
    }
    sums.x = simd_sum(sums.x); sums.y = simd_sum(sums.y);
    sums.z = simd_sum(sums.z); sums.w = simd_sum(sums.w);
    if (lane == 0) for (uint r = 0; r < count; r++)
        y[uint64_t(t0+r)*a.os+a.oo+row] = sums[r];
}
