#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

static inline float qw3_vis_erff(float x) {
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float poly = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                   - 0.284496736f) * t + 0.254829592f) * t;
    float r = 1.0f - poly * exp(-ax * ax);
    return x < 0.0f ? -r : r;
}

static inline float qw3_vis_gelu_erf(float x) {
    return 0.5f * x * (1.0f + qw3_vis_erff(x * 0.7071067811865475244f));
}

static inline float qw3_vis_read_w(const device uchar *base, ulong idx, uint w_type) {
    if (w_type == 1u) {
        return float(((const device half *)base)[idx]);
    } else if (w_type == 30u) {
        ushort b = ((const device ushort *)base)[idx];
        return as_type<float>(uint(b) << 16);
    }
    return ((const device float *)base)[idx];
}

struct VisPatchArgs {
    uint W;
    uint H;
    uint P;
    uint W_p;
    uint H_p;
    uint D;
    uint pos_grid;
    uint w_type;
    uint has_bias;
};

/* Patch embedding Conv2D (stride=P) + bilinear position embedding + CLS prepend.
 * Grid: (D, n_patches + 1, 1) */
kernel void qw3_vis_patch_embed(
    constant VisPatchArgs &a            [[buffer(0)]],
    const device float *img_planar      [[buffer(1)]],
    const device uchar *patch_w         [[buffer(2)]],
    const device float *patch_bias      [[buffer(3)]],
    const device float *pos_embd        [[buffer(4)]],
    const device float *class_embd      [[buffer(5)]],
    const device float *class_pos_embd  [[buffer(6)]],
    device float *out                   [[buffer(7)]],
    uint2 gid                           [[thread_position_in_grid]])
{
    uint oc = gid.x;
    uint t  = gid.y;
    uint n_patches = a.W_p * a.H_p;
    if (oc >= a.D || t > n_patches) return;

    if (t == 0u) {
        out[oc] = class_embd[oc] + class_pos_embd[oc];
        return;
    }

    uint p  = t - 1u;
    uint px = p % a.W_p;
    uint py = p / a.W_p;

    float sum = a.has_bias ? patch_bias[oc] : 0.0f;
    ulong w_oc_base = ulong(oc) * 3ul * ulong(a.P) * ulong(a.P);
    ulong plane_sz  = ulong(a.W) * ulong(a.H);

    for (uint c = 0u; c < 3u; c++) {
        ulong w_c_base = w_oc_base + ulong(c) * ulong(a.P) * ulong(a.P);
        ulong img_c_base = ulong(c) * plane_sz;
        for (uint ky = 0u; ky < a.P; ky++) {
            ulong img_row = img_c_base + ulong(py * a.P + ky) * ulong(a.W) + ulong(px * a.P);
            ulong w_row   = w_c_base + ulong(ky) * ulong(a.P);
            for (uint kx = 0u; kx < a.P; kx++) {
                sum += qw3_vis_read_w(patch_w, w_row + kx, a.w_type) * img_planar[img_row + kx];
            }
        }
    }

    if (a.W_p == a.pos_grid && a.H_p == a.pos_grid) {
        sum += pos_embd[ulong(p) * ulong(a.D) + oc];
    } else {
        float sf_x = float(a.W_p) / float(a.pos_grid);
        float sf_y = float(a.H_p) / float(a.pos_grid);
        float fx = (float(px) + 0.5f) / sf_x - 0.5f;
        float fy = (float(py) + 0.5f) / sf_y - 0.5f;
        int pg_max = int(a.pos_grid) - 1;
        int x0 = int(floor(fx));
        int x1 = x0 + 1;
        int y0 = int(floor(fy));
        int y1 = y0 + 1;
        x0 = clamp(x0, 0, pg_max);
        x1 = clamp(x1, 0, pg_max);
        y0 = clamp(y0, 0, pg_max);
        y1 = clamp(y1, 0, pg_max);
        float dx = clamp(fx - float(x0), 0.0f, 1.0f);
        float dy = clamp(fy - float(y0), 0.0f, 1.0f);

        float p00 = pos_embd[(ulong(y0) * a.pos_grid + ulong(x0)) * a.D + oc];
        float p10 = pos_embd[(ulong(y0) * a.pos_grid + ulong(x1)) * a.D + oc];
        float p01 = pos_embd[(ulong(y1) * a.pos_grid + ulong(x0)) * a.D + oc];
        float p11 = pos_embd[(ulong(y1) * a.pos_grid + ulong(x1)) * a.D + oc];
        sum += p00 * (1.0f - dx) * (1.0f - dy)
             + p10 * dx * (1.0f - dy)
             + p01 * (1.0f - dx) * dy
             + p11 * dx * dy;
    }

    out[ulong(t) * ulong(a.D) + oc] = sum;
}

struct VisLNArgs {
    uint N;
    uint D;
    float eps;
};

/* Standard LayerNorm: 1 threadgroup (256 threads) per row. */
kernel void qw3_vis_layernorm(
    constant VisLNArgs &a          [[buffer(0)]],
    const device float *x          [[buffer(1)]],
    const device float *w          [[buffer(2)]],
    const device float *b          [[buffer(3)]],
    device float *y                [[buffer(4)]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]])
{
    if (tgid >= a.N) return;
    threadgroup float partial[8];

    const device float *row_in = x + ulong(tgid) * ulong(a.D);
    device float *row_out      = y + ulong(tgid) * ulong(a.D);

    float s = 0.0f;
    for (uint i = tid; i < a.D; i += 256u) {
        s += row_in[i];
    }
    s = simd_sum(s);
    if (simd_lane == 0u) partial[simd_id] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = 0.0f;
    if (tid == 0u) {
        float tot = 0.0f;
        for (uint k = 0u; k < 8u; k++) tot += partial[k];
        partial[0] = tot / float(a.D);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mean = partial[0];

    float v = 0.0f;
    for (uint i = tid; i < a.D; i += 256u) {
        float diff = row_in[i] - mean;
        v += diff * diff;
    }
    v = simd_sum(v);
    if (simd_lane == 0u) partial[simd_id] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float tot = 0.0f;
        for (uint k = 0u; k < 8u; k++) tot += partial[k];
        partial[0] = rsqrt(tot / float(a.D) + a.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_std = partial[0];

    for (uint i = tid; i < a.D; i += 256u) {
        row_out[i] = (row_in[i] - mean) * inv_std * w[i] + b[i];
    }
}

struct VisGemmArgs {
    uint M;
    uint N_out;
    uint K;
    uint w_type;
    uint has_bias;
    uint act_gelu;
};

/* Tiled simdgroup_matrix GEMM: Y[M, N_out] = X[M, K] @ W[N_out, K]^T + bias
 * Fast F16 weight path using half8x8 * half8x8 -> float8x8 accumulation.
 * Threadgroup: 128 threads (4 simdgroups), tile BM=64, BN=64, BK=32, vectorized 4-wide loads. */
kernel void qw3_vis_gemm_f16(
    constant VisGemmArgs &a        [[buffer(0)]],
    const device float *X          [[buffer(1)]],
    const device half  *W          [[buffer(2)]],
    const device float *bias       [[buffer(3)]],
    device float *Y                [[buffer(4)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]])
{
    uint m0 = tgid.y * 64u;
    uint n0 = tgid.x * 64u;

    threadgroup half  As[64][36];
    threadgroup half  Bs[64][36];
    threadgroup float Cs[64][64];

    uint sg_m = simd_id >> 1;  /* 0..1 -> 32 rows each */
    uint sg_n = simd_id & 1u;  /* 0..1 -> 32 cols each */

    simdgroup_float8x8 acc[4][4];
    for (uint im = 0u; im < 4u; im++) {
        for (uint in = 0u; in < 4u; in++) {
            acc[im][in] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    uint vec_r = tid >> 3;        /* 0..15 */
    uint vec_c = (tid & 7u) * 4u; /* 0,4,8..28 */

    for (uint k0 = 0u; k0 < a.K; k0 += 32u) {
        /* 64x32 = 512 float4/half4 loads -> 4 iterations of 128 threads */
        for (uint r_off = 0u; r_off < 64u; r_off += 16u) {
            uint r = r_off + vec_r;
            uint gm = m0 + r;
            uint gk = k0 + vec_c;
            if (gm < a.M && gk + 3u < a.K) {
                float4 xv = *(const device float4 *)(X + ulong(gm) * ulong(a.K) + gk);
                *(threadgroup half4 *)(&As[r][vec_c]) = half4(xv);
            } else {
                for (uint j = 0u; j < 4u; j++) {
                    As[r][vec_c + j] = (gm < a.M && gk + j < a.K) ? half(X[ulong(gm) * ulong(a.K) + gk + j]) : half(0.0h);
                }
            }
            uint gn = n0 + r;
            if (gn < a.N_out && gk + 3u < a.K) {
                half4 wv = *(const device half4 *)(W + ulong(gn) * ulong(a.K) + gk);
                *(threadgroup half4 *)(&Bs[r][vec_c]) = wv;
            } else {
                for (uint j = 0u; j < 4u; j++) {
                    Bs[r][vec_c + j] = (gn < a.N_out && gk + j < a.K) ? W[ulong(gn) * ulong(a.K) + gk + j] : half(0.0h);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_half8x8 a_mat[4];
            simdgroup_half8x8 b_mat[4];
            for (uint im = 0u; im < 4u; im++) {
                simdgroup_load(a_mat[im], &As[sg_m * 32u + im * 8u][kk], 36, 0, false);
            }
            for (uint in = 0u; in < 4u; in++) {
                simdgroup_load(b_mat[in], &Bs[sg_n * 32u + in * 8u][kk], 36, 0, true);
            }
            for (uint im = 0u; im < 4u; im++) {
                for (uint in = 0u; in < 4u; in++) {
                    simdgroup_multiply_accumulate(acc[im][in], a_mat[im], b_mat[in], acc[im][in]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint im = 0u; im < 4u; im++) {
        for (uint in = 0u; in < 4u; in++) {
            simdgroup_store(acc[im][in], &Cs[sg_m * 32u + im * 8u][sg_n * 32u + in * 8u], 64, 0, false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < 64u * 64u; idx += 128u) {
        uint r = idx >> 6;
        uint c = idx & 63u;
        uint gm = m0 + r;
        uint gn = n0 + c;
        if (gm < a.M && gn < a.N_out) {
            float v = Cs[r][c];
            if (a.has_bias) v += bias[gn];
            if (a.act_gelu) v = qw3_vis_gelu_erf(v);
            Y[ulong(gm) * ulong(a.N_out) + gn] = v;
        }
    }
}

/* Generic F32/BF16 weight path using float8x8 * float8x8 -> float8x8 accumulation. */
kernel void qw3_vis_gemm_generic(
    constant VisGemmArgs &a        [[buffer(0)]],
    const device float *X          [[buffer(1)]],
    const device uchar *W          [[buffer(2)]],
    const device float *bias       [[buffer(3)]],
    device float *Y                [[buffer(4)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]])
{
    uint m0 = tgid.y * 32u;
    uint n0 = tgid.x * 64u;

    threadgroup float As[32][36];
    threadgroup float Bs[64][36];
    threadgroup float Cs[32][64];

    uint sg_m = simd_id >> 1;
    uint sg_n = simd_id & 1u;

    simdgroup_float8x8 acc[2][4];
    for (uint im = 0u; im < 2u; im++) {
        for (uint in = 0u; in < 4u; in++) {
            acc[im][in] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (uint k0 = 0u; k0 < a.K; k0 += 32u) {
        for (uint idx = tid; idx < 32u * 32u; idx += 128u) {
            uint r = idx >> 5;
            uint c = idx & 31u;
            uint gm = m0 + r;
            uint gk = k0 + c;
            As[r][c] = (gm < a.M && gk < a.K) ? X[ulong(gm) * ulong(a.K) + gk] : 0.0f;
        }
        for (uint idx = tid; idx < 64u * 32u; idx += 128u) {
            uint r = idx >> 5;
            uint c = idx & 31u;
            uint gn = n0 + r;
            uint gk = k0 + c;
            Bs[r][c] = (gn < a.N_out && gk < a.K) ? qw3_vis_read_w(W, ulong(gn) * ulong(a.K) + gk, a.w_type) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_float8x8 a_mat[2];
            simdgroup_float8x8 b_mat[4];
            for (uint im = 0u; im < 2u; im++) {
                simdgroup_load(a_mat[im], &As[sg_m * 16u + im * 8u][kk], 36, 0, false);
            }
            for (uint in = 0u; in < 4u; in++) {
                simdgroup_load(b_mat[in], &Bs[sg_n * 32u + in * 8u][kk], 36, 0, true);
            }
            for (uint im = 0u; im < 2u; im++) {
                for (uint in = 0u; in < 4u; in++) {
                    simdgroup_multiply_accumulate(acc[im][in], a_mat[im], b_mat[in], acc[im][in]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint im = 0u; im < 2u; im++) {
        for (uint in = 0u; in < 4u; in++) {
            simdgroup_store(acc[im][in], &Cs[sg_m * 16u + im * 8u][sg_n * 32u + in * 8u], 64, 0, false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < 32u * 64u; idx += 128u) {
        uint r = idx >> 6;
        uint c = idx & 63u;
        uint gm = m0 + r;
        uint gn = n0 + c;
        if (gm < a.M && gn < a.N_out) {
            float v = Cs[r][c];
            if (a.has_bias) v += bias[gn];
            if (a.act_gelu) v = qw3_vis_gelu_erf(v);
            Y[ulong(gm) * ulong(a.N_out) + gn] = v;
        }
    }
}

struct VisRopeArgs {
    uint N;
    uint W_p;
    uint n_head;
    uint d_head;
    float theta;
};

/* 2D RoPE on Q and K in-place (normal interleaved pairs, pos_w on first half, pos_h on second half).
 * CLS (t == 0) has pos_w = 0, pos_h = 0 (identity).
 * Grid: (n_head * (d_head / 2), N, 1) */
kernel void qw3_vis_rope2d(
    constant VisRopeArgs &a        [[buffer(0)]],
    device float *Q                [[buffer(1)]],
    device float *K                [[buffer(2)]],
    uint2 gid                      [[thread_position_in_grid]])
{
    uint half_d = a.d_head >> 1;
    uint total_pairs = a.n_head * half_d;
    uint pair_idx = gid.x;
    uint t = gid.y;
    if (pair_idx >= total_pairs || t >= a.N || t == 0u) return;

    uint h = pair_idx / half_d;
    uint i = pair_idx % half_d; /* 0 .. d_head/2 - 1 */
    uint quarter_d = half_d >> 1; /* d_head / 4 = 16 for d_head=64 */

    uint p = t - 1u;
    uint pos_w = (p % a.W_p) + 1u;
    uint pos_h = (p / a.W_p) + 1u;

    uint dim_i = (i < quarter_d) ? i : (i - quarter_d);
    float pos  = (i < quarter_d) ? float(pos_w) : float(pos_h);
    float inv_freq = pow(a.theta, -2.0f * float(dim_i) / float(half_d));
    float angle = pos * inv_freq;
    float c = cos(angle);
    float s = sin(angle);

    ulong off = ulong(t) * ulong(a.n_head * a.d_head) + ulong(h * a.d_head) + ulong(i * 2u);
    float q0 = Q[off];
    float q1 = Q[off + 1u];
    Q[off]      = q0 * c - q1 * s;
    Q[off + 1u] = q0 * s + q1 * c;

    float k0 = K[off];
    float k1 = K[off + 1u];
    K[off]      = k0 * c - k1 * s;
    K[off + 1u] = k0 * s + k1 * c;
}

struct VisAttnArgs {
    uint N;
    uint n_head;
    uint d_head;
    float scale;
};

/* Tensor-Core Bidirectional Multi-Head FlashAttention (d_head == 64).
 * 4 simdgroups (128 threads) per threadgroup -> BQ=32 query rows (8 per simdgroup), BK=32 keys/tile.
 * Grid threadgroups: ((N + 31) / 32, n_head, 1), threadsPerThreadgroup: (128, 1, 1). */
kernel void qw3_vis_attn_flash64(
    constant VisAttnArgs &a        [[buffer(0)]],
    const device float *Q          [[buffer(1)]],
    const device float *K          [[buffer(2)]],
    const device float *V          [[buffer(3)]],
    device float *Out              [[buffer(4)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]])
{
    uint h = tgid.y;
    uint q0_tg = tgid.x * 32u;
    uint sg_r = simd_id * 8u;
    uint D = a.n_head * 64u;
    uint h_off = h * 64u;

    threadgroup half  K_s[32][68];
    threadgroup half  V_s[32][68];
    threadgroup float S_s[32][36];
    threadgroup half  P_s[32][36];
    threadgroup float O_s[32][64];

    /* Cooperatively load Q[q0_tg..q0_tg+31][0..63] into K_s temporarily, then load q_mat[8] into registers */
    uint lr = tid >> 4;          /* 0..7 */
    uint lc = (tid & 15u) * 4u;  /* 0,4..60 */
    for (uint r_off = 0u; r_off < 32u; r_off += 8u) {
        uint r = r_off + lr;
        uint tq = q0_tg + r;
        if (tq < a.N) {
            float4 qv = *(const device float4 *)(Q + ulong(tq) * ulong(D) + h_off + lc);
            *(threadgroup half4 *)(&K_s[r][lc]) = half4(qv);
        } else {
            *(threadgroup half4 *)(&K_s[r][lc]) = half4(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 q_mat[8];
    simdgroup_float8x8 o_acc[8];
    for (uint d = 0u; d < 8u; d++) {
        simdgroup_load(q_mat[d], &K_s[sg_r][d * 8u], 68, 0, false);
        o_acc[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_row[8];
    float l_row[8];
    for (uint r = 0u; r < 8u; r++) {
        m_row[r] = -1.0e30f;
        l_row[r] = 0.0f;
    }

    for (uint k0 = 0u; k0 < a.N; k0 += 32u) {
        /* Cooperatively load K_s and V_s (32x64 = 512 float4 loads -> 4 iterations of 128 threads) */
        for (uint r_off = 0u; r_off < 32u; r_off += 8u) {
            uint r = r_off + lr;
            uint tk = k0 + r;
            if (tk < a.N) {
                ulong kv_off = ulong(tk) * ulong(D) + h_off + lc;
                *(threadgroup half4 *)(&K_s[r][lc]) = half4(*(const device float4 *)(K + kv_off));
                *(threadgroup half4 *)(&V_s[r][lc]) = half4(*(const device float4 *)(V + kv_off));
            } else {
                *(threadgroup half4 *)(&K_s[r][lc]) = half4(0.0h);
                *(threadgroup half4 *)(&V_s[r][lc]) = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* Compute S[8][32] = Q[8][64] @ K[32][64]^T via simdgroup_matrix */
        simdgroup_float8x8 s_acc[4];
        for (uint j = 0u; j < 4u; j++) {
            s_acc[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        for (uint d = 0u; d < 8u; d++) {
            for (uint j = 0u; j < 4u; j++) {
                simdgroup_half8x8 k_mat;
                simdgroup_load(k_mat, &K_s[j * 8u][d * 8u], 68, 0, true);
                simdgroup_multiply_accumulate(s_acc[j], q_mat[d], k_mat, s_acc[j]);
            }
        }
        for (uint j = 0u; j < 4u; j++) {
            simdgroup_store(s_acc[j], &S_s[sg_r][j * 8u], 36, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* Online softmax across 32 keys (lane = key col 0..31) for each of the 8 query rows */
        float alpha_row[8];
        for (uint r = 0u; r < 8u; r++) {
            bool valid_k = (k0 + simd_lane < a.N);
            float sc = valid_k ? (S_s[sg_r + r][simd_lane] * a.scale) : -1.0e30f;
            float m_blk = simd_max(sc);
            float m_new = (k0 == 0u) ? m_blk : max(m_row[r], m_blk);
            float alpha = (k0 == 0u) ? 0.0f : exp(m_row[r] - m_new);
            float p = valid_k ? exp(sc - m_new) : 0.0f;
            l_row[r] = l_row[r] * alpha + simd_sum(p);
            m_row[r] = m_new;
            alpha_row[r] = alpha;
            P_s[sg_r + r][simd_lane] = half(p);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* Rescale existing o_acc by alpha_row when k0 > 0 */
        if (k0 > 0u) {
            for (uint d = 0u; d < 8u; d++) {
                simdgroup_store(o_acc[d], &O_s[sg_r][d * 8u], 64, 0, false);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0u; r < 8u; r++) {
                float al = alpha_row[r];
                O_s[sg_r + r][simd_lane]       *= al;
                O_s[sg_r + r][simd_lane + 32u] *= al;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint d = 0u; d < 8u; d++) {
                simdgroup_load(o_acc[d], &O_s[sg_r][d * 8u], 64, 0, false);
            }
        }

        /* Accumulate o_acc[8][64] += P[8][32] @ V[32][64] via simdgroup_matrix */
        simdgroup_half8x8 p_mat[4];
        for (uint j = 0u; j < 4u; j++) {
            simdgroup_load(p_mat[j], &P_s[sg_r][j * 8u], 36, 0, false);
        }
        for (uint d = 0u; d < 8u; d++) {
            for (uint j = 0u; j < 4u; j++) {
                simdgroup_half8x8 v_mat;
                simdgroup_load(v_mat, &V_s[j * 8u][d * 8u], 68, 0, false);
                simdgroup_multiply_accumulate(o_acc[d], p_mat[j], v_mat, o_acc[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint d = 0u; d < 8u; d++) {
        simdgroup_store(o_acc[d], &O_s[sg_r][d * 8u], 64, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = 0u; r < 8u; r++) {
        uint tq = q0_tg + sg_r + r;
        if (tq < a.N) {
            float inv_l = 1.0f / l_row[r];
            ulong o_off = ulong(tq) * ulong(D) + h_off;
            Out[o_off + simd_lane]       = O_s[sg_r + r][simd_lane] * inv_l;
            Out[o_off + simd_lane + 32u] = O_s[sg_r + r][simd_lane + 32u] * inv_l;
        }
    }
}

struct VisResArgs {
    uint N;
    uint D;
};

/* X[t, d] += Y[t, d] * ls[d] */
kernel void qw3_vis_residual_scale(
    constant VisResArgs &a         [[buffer(0)]],
    device float *X                [[buffer(1)]],
    const device float *Y          [[buffer(2)]],
    const device float *ls         [[buffer(3)]],
    uint2 gid                      [[thread_position_in_grid]])
{
    uint d = gid.x;
    uint t = gid.y;
    if (d >= a.D || t >= a.N) return;
    ulong idx = ulong(t) * ulong(a.D) + d;
    X[idx] += Y[idx] * ls[d];
}

struct VisMergeArgs {
    uint W_p;
    uint H_p;
    uint D;
    uint merge;
};

/* Drop CLS (row 0) and merge merge x merge spatial patches into [out_h * out_w, merge*merge*D].
 * Order inside merged vector: row-major (dy=0..merge-1, dx=0..merge-1).
 * Grid: (merge * merge * D, (W_p/merge) * (H_p/merge), 1) */
kernel void qw3_vis_patch_merge(
    constant VisMergeArgs &a       [[buffer(0)]],
    const device float *X          [[buffer(1)]],
    device float *Out              [[buffer(2)]],
    uint2 gid                      [[thread_position_in_grid]])
{
    uint out_w = a.W_p / a.merge;
    uint out_h = a.H_p / a.merge;
    uint n_out = out_w * out_h;
    uint out_dim = a.merge * a.merge * a.D;
    uint c = gid.x;
    uint oi = gid.y;
    if (c >= out_dim || oi >= n_out) return;

    uint sub = c / a.D;
    uint d   = c % a.D;
    uint dy  = sub / a.merge;
    uint dx  = sub % a.merge;

    uint bx = oi % out_w;
    uint by = oi / out_w;
    uint src_x = bx * a.merge + dx;
    uint src_y = by * a.merge + dy;
    uint src_t = 1u + src_y * a.W_p + src_x; /* +1 skips CLS token */

    Out[ulong(oi) * ulong(out_dim) + c] = X[ulong(src_t) * ulong(a.D) + d];
}
