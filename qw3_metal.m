#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#include "qw3_metal.h"

/*
 * Minimal Metal bridge for qw3.
 *
 * This intentionally starts much smaller than ds4_metal.m: initialize the
 * device/queue and expose the GGUF tensor-data region as a shared Metal buffer.
 * Actual graph kernels will grow from this boundary one primitive at a time.
 */

static id<MTLDevice> g_device;
static id<MTLCommandQueue> g_queue;
static id<MTLCommandBuffer> g_batch_cb;
static id<MTLComputeCommandEncoder> g_batch_enc;
static NSMutableArray<id<MTLCommandBuffer>> *g_pending_cbs;

static NSMutableArray<id<MTLBuffer>> *g_model_buffers;
static NSMutableDictionary<NSString *, id<MTLBuffer>> *g_model_temp_buffers;
static id<MTLBuffer> g_iq3s_kgrid_buffer;
static id<MTLLibrary> g_library;
static id<MTLComputePipelineState> g_rmsnorm_plain_pipeline;
static id<MTLComputePipelineState> g_rmsnorm_weight_f32_pipeline;
static id<MTLComputePipelineState> g_embed_q8_0_pipeline;
static id<MTLComputePipelineState> g_matvec_q8_0_pipeline;
static id<MTLComputePipelineState> g_matvec_q8_0_pair_pipeline;
static id<MTLComputePipelineState> g_matvec_q8_0_pair_silu_pipeline;
static id<MTLComputePipelineState> g_shared_gate_up_silu_pipeline;
static id<MTLComputePipelineState> g_matvec_q8_0_inner_scale_add_x0_pipeline;
static id<MTLComputePipelineState> g_matvec_iq4_xs_pipeline;
static id<MTLComputePipelineState> g_matvec_q6_k_pipeline;
static id<MTLComputePipelineState> g_matvec_iq4_xs_add_x0_pipeline;
static id<MTLComputePipelineState> g_matvec_q6_k_add_x0_pipeline;
static id<MTLComputePipelineState> g_matvec_iq4_xs_swiglu_add_x0_pipeline;
static id<MTLComputePipelineState> g_matvec_q6_k_swiglu_add_x0_pipeline;
static id<MTLComputePipelineState> g_matvec_iq3_s_pipeline;
static id<MTLComputePipelineState> g_matvec_iq3_s_pair_pipeline;
static id<MTLComputePipelineState> g_moe_iq3_s_pair_batch_pipeline;
static id<MTLComputePipelineState> g_moe_down_iq4_xs_batch_pipeline;
static id<MTLComputePipelineState> g_moe_down_iq4_xs_pair_pipeline;
static id<MTLComputePipelineState> g_moe_down_iq4_xs_batch_reduce_pipeline;
static id<MTLComputePipelineState> g_moe_down_q6_k_batch_pipeline;
static id<MTLComputePipelineState> g_moe_reduce_batch_pipeline;
static id<MTLComputePipelineState> g_matvec_f32_pipeline;
static id<MTLComputePipelineState> g_matvec_f32_pair_pipeline;
static id<MTLComputePipelineState> g_matvec_f32_fast_pipeline;
static id<MTLComputePipelineState> g_deltanet_conv1d_zero_pipeline;
static id<MTLComputePipelineState> g_deltanet_conv1d_step_pipeline;
static id<MTLComputePipelineState> g_l2norm_heads_pipeline;
static id<MTLComputePipelineState> g_gqa_q_norm_gate_pipeline;
static id<MTLComputePipelineState> g_gqa_k_norm_pipeline;
static id<MTLComputePipelineState> g_rope_heads_pipeline;
static id<MTLComputePipelineState> g_gqa_single_token_inner_pipeline;
static id<MTLComputePipelineState> g_gqa_attend2_inner_pipeline;
static id<MTLComputePipelineState> g_gqa_attend_n_inner_pipeline;
static id<MTLComputePipelineState> g_gqa_kv_quant_q8_pipeline;
static id<MTLComputePipelineState> g_gqa_attend_n_q8_inner_pipeline;
static id<MTLComputePipelineState> g_deltanet_recur_zero_pipeline;
static id<MTLComputePipelineState> g_deltanet_recur_pipeline;
static id<MTLComputePipelineState> g_deltanet_recur_scratch_gates_pipeline;
static id<MTLComputePipelineState> g_deltanet_fused_gdn_scratch_pipeline;
static id<MTLComputePipelineState> g_deltanet_gated_rmsnorm_pipeline;
static id<MTLComputePipelineState> g_residual_rmsnorm_weight_f32_pipeline;
static id<MTLComputePipelineState> g_residual_rmsnorm_update_x0_pipeline;
static id<MTLComputePipelineState> g_silu_mul_pipeline;
static id<MTLComputePipelineState> g_scale_pipeline;
static id<MTLComputePipelineState> g_argmax_blocks_pipeline;
static id<MTLComputePipelineState> g_add_moe_to_x0_pipeline;
static id<MTLComputePipelineState> g_silu_mul_offsets_pipeline;
static id<MTLComputePipelineState> g_scale_x1_scalar_add_x0_pipeline;
static id<MTLComputePipelineState> g_scale_x1_add_x0_pipeline;
static id<MTLComputePipelineState> g_scale_scratch_add_x0_pipeline;
static id<MTLComputePipelineState> g_router_top8_pipeline;
static id<MTLComputePipelineState> g_matvec_iq3_s_expert_slot_pipeline;
static id<MTLComputePipelineState> g_matvec_iq3_s_expert_slot_pair_pipeline;
static id<MTLComputePipelineState> g_matvec_iq4_xs_expert_slot_pipeline;
static id<MTLComputePipelineState> g_matvec_q6_k_expert_slot_pipeline;
static id<MTLComputePipelineState> g_scale_scratch_add_x0_slot_pipeline;
static const uint8_t *g_model_view_ptrs[32];
static uint64_t g_model_view_offsets[32];
static uint64_t g_model_view_sizes[32];
static uint32_t g_model_view_count;
static char g_device_name[256];
static const void *g_model_map_ptr;
static uint64_t g_model_map_size;
static uint64_t g_model_offset;
static uint64_t g_model_size;
static int g_initialized;

static id<MTLCommandBuffer> qw3_metal_command_buffer(int *owned) {
    if (g_batch_cb) {
        if (owned) *owned = 0;
        return g_batch_cb;
    }
    if (owned) *owned = 1;
    if (getenv("QW3_METAL_UNRETAINED_COMMAND_BUFFERS") != NULL) {
        return [g_queue commandBufferWithUnretainedReferences];
    }
    return [g_queue commandBuffer];
}

static id<MTLComputeCommandEncoder> qw3_metal_compute_encoder(id<MTLCommandBuffer> cb) {
    if (g_batch_cb && cb == g_batch_cb) {
        if (!g_batch_enc) g_batch_enc = [cb computeCommandEncoder];
        return g_batch_enc;
    }
    return [cb computeCommandEncoder];
}

static void qw3_metal_end_compute_encoder(id<MTLCommandBuffer> cb,
                                          id<MTLComputeCommandEncoder> enc) {
    if (!enc) return;
    if (g_batch_cb && cb == g_batch_cb && enc == g_batch_enc) return;
    [enc endEncoding];
}

static void qw3_metal_close_batch_encoder(void) {
    if (!g_batch_enc) return;
    [g_batch_enc endEncoding];
    g_batch_enc = nil;
}

static int qw3_metal_wait_command_buffer(id<MTLCommandBuffer> cb,
                                         const char *label) {
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal %s command failed: %s\n",
                label ? label : "batch",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

static int qw3_metal_wait_pending_command_buffers(const char *label) {
    int ok = 1;
    for (id<MTLCommandBuffer> pending in g_pending_cbs) {
        if (!qw3_metal_wait_command_buffer(pending, label)) ok = 0;
    }
    [g_pending_cbs removeAllObjects];
    return ok;
}

static int qw3_metal_finish_command_buffer(id<MTLCommandBuffer> cb,
                                           int owned,
                                           const char *label) {
    if (!cb) return 0;
    if (!owned) return 1;
    [cb commit];
    int ok = qw3_metal_wait_pending_command_buffers(label);
    if (!qw3_metal_wait_command_buffer(cb, label)) ok = 0;
    return ok;
}

@interface QW3MetalSessionObj : NSObject
@property(nonatomic, strong) id<MTLBuffer> gqaK;
@property(nonatomic, strong) id<MTLBuffer> gqaV;
@property(nonatomic, strong) id<MTLBuffer> deltanetState;
@property(nonatomic, strong) id<MTLBuffer> convState;
@property(nonatomic, strong) id<MTLBuffer> logits;
@property(nonatomic, strong) id<MTLBuffer> x0;
@property(nonatomic, strong) id<MTLBuffer> x1;
@property(nonatomic, strong) id<MTLBuffer> scratch;
@property(nonatomic, strong) id<MTLBuffer> qkvConv;
@property(nonatomic, strong) id<MTLBuffer> qNorm;
@property(nonatomic, strong) id<MTLBuffer> kNorm;
@property(nonatomic, strong) id<MTLBuffer> core;
@property(nonatomic, strong) id<MTLBuffer> inner;
@property(nonatomic, strong) id<MTLBuffer> gqaTmpQ;
@property(nonatomic, strong) id<MTLBuffer> gqaTmpK;
@property(nonatomic, strong) id<MTLBuffer> gqaTokenQ;
@property(nonatomic, strong) id<MTLBuffer> gqaTokenK;
@property(nonatomic, strong) id<MTLBuffer> gqaTokenV;
@property(nonatomic, strong) id<MTLBuffer> gqaTokenGate;
@property(nonatomic, strong) id<MTLBuffer> routerIds;
@property(nonatomic, strong) id<MTLBuffer> routerWeights;
@property(nonatomic, strong) id<MTLBuffer> argmaxVals;
@property(nonatomic, strong) id<MTLBuffer> argmaxIdxs;
@property(nonatomic) qw3_metal_session_info info;
@property(nonatomic) uint32_t ctxSize;
@property(nonatomic) uint32_t vocabSize;
@property(nonatomic) uint32_t pos;
@property(nonatomic) BOOL gqaKvQ8;
@end

@implementation QW3MetalSessionObj
@end

struct qw3_metal_session {
    void *obj;
};

enum {
    QW3_METAL_N_FULL_ATTN_LAYERS = 10,
    QW3_METAL_N_LINEAR_LAYERS = 30,
    QW3_METAL_N_EMBD = 2048,
    QW3_METAL_N_HEAD_KV = 2,
    QW3_METAL_N_HEAD_DIM = 256,
    QW3_METAL_N_HEAD = 16,
    QW3_METAL_ROPE_DIM = 64,
    QW3_METAL_N_LINEAR_QK_HEADS = 16,
    QW3_METAL_N_LINEAR_V_HEADS = 32,
    QW3_METAL_N_LINEAR_HEAD_DIM = 128,
    QW3_METAL_LINEAR_QKV = 8192,
    QW3_METAL_LINEAR_INNER = 4096,
    QW3_METAL_LINEAR_CONV_K = 4,
};

static const uint16_t g_iq3s_kgrid[512] = {
       0,    1,    2,    5,    7,    8,    9,   10,   12,   14,   16,   17,   21,   27,   32,   34,
      37,   39,   41,   43,   48,   50,   57,   60,   63,   64,   65,   66,   68,   72,   73,   77,
      80,   83,   87,   89,   93,  100,  113,  117,  122,  128,  129,  133,  135,  136,  139,  142,
     145,  149,  152,  156,  162,  165,  167,  169,  171,  184,  187,  195,  201,  205,  208,  210,
     217,  219,  222,  228,  232,  234,  247,  249,  253,  256,  267,  271,  273,  276,  282,  288,
     291,  297,  312,  322,  324,  336,  338,  342,  347,  353,  357,  359,  374,  379,  390,  393,
     395,  409,  426,  441,  448,  450,  452,  464,  466,  470,  475,  488,  492,  512,  513,  514,
     516,  520,  521,  523,  525,  527,  528,  530,  537,  540,  542,  556,  558,  561,  570,  576,
     577,  579,  582,  584,  588,  593,  600,  603,  609,  616,  618,  632,  638,  640,  650,  653,
     655,  656,  660,  666,  672,  675,  685,  688,  698,  705,  708,  711,  712,  715,  721,  727,
     728,  732,  737,  754,  760,  771,  773,  778,  780,  793,  795,  802,  806,  808,  812,  833,
     840,  843,  849,  856,  858,  873,  912,  916,  919,  932,  934,  961,  963,  968,  970,  977,
     989,  993, 1010, 1016, 1024, 1025, 1027, 1029, 1031, 1032, 1034, 1036, 1038, 1041, 1043, 1047,
    1048, 1050, 1057, 1059, 1061, 1064, 1066, 1079, 1080, 1083, 1085, 1088, 1090, 1096, 1099, 1103,
    1106, 1109, 1113, 1116, 1122, 1129, 1153, 1156, 1159, 1169, 1171, 1176, 1183, 1185, 1195, 1199,
    1209, 1212, 1216, 1218, 1221, 1225, 1234, 1236, 1241, 1243, 1250, 1256, 1270, 1281, 1287, 1296,
    1299, 1306, 1309, 1313, 1338, 1341, 1348, 1353, 1362, 1375, 1376, 1387, 1400, 1408, 1410, 1415,
    1425, 1453, 1457, 1477, 1481, 1494, 1496, 1507, 1512, 1538, 1545, 1547, 1549, 1551, 1554, 1561,
    1563, 1565, 1570, 1572, 1575, 1577, 1587, 1593, 1601, 1603, 1605, 1612, 1617, 1619, 1632, 1648,
    1658, 1662, 1664, 1674, 1680, 1690, 1692, 1704, 1729, 1736, 1740, 1745, 1747, 1751, 1752, 1761,
    1763, 1767, 1773, 1787, 1795, 1801, 1806, 1810, 1817, 1834, 1840, 1844, 1857, 1864, 1866, 1877,
    1882, 1892, 1902, 1915, 1934, 1953, 1985, 1987, 2000, 2002, 2013, 2048, 2052, 2058, 2064, 2068,
    2071, 2074, 2081, 2088, 2104, 2114, 2119, 2121, 2123, 2130, 2136, 2141, 2147, 2153, 2157, 2177,
    2179, 2184, 2189, 2193, 2203, 2208, 2223, 2226, 2232, 2244, 2249, 2251, 2256, 2258, 2265, 2269,
    2304, 2306, 2324, 2335, 2336, 2361, 2373, 2375, 2385, 2418, 2443, 2460, 2480, 2504, 2509, 2520,
    2531, 2537, 2562, 2568, 2572, 2578, 2592, 2596, 2599, 2602, 2614, 2620, 2625, 2627, 2629, 2634,
    2641, 2650, 2682, 2688, 2697, 2707, 2712, 2718, 2731, 2754, 2759, 2760, 2775, 2788, 2793, 2805,
    2811, 2817, 2820, 2832, 2842, 2854, 2890, 2902, 2921, 2923, 2978, 3010, 3012, 3026, 3081, 3083,
    3085, 3097, 3099, 3120, 3136, 3152, 3159, 3188, 3210, 3228, 3234, 3245, 3250, 3256, 3264, 3276,
    3281, 3296, 3349, 3363, 3378, 3392, 3395, 3420, 3440, 3461, 3488, 3529, 3531, 3584, 3588, 3591,
    3600, 3602, 3614, 3616, 3628, 3634, 3650, 3657, 3668, 3683, 3685, 3713, 3716, 3720, 3726, 3729,
    3736, 3753, 3778, 3802, 3805, 3819, 3841, 3845, 3851, 3856, 3880, 3922, 3938, 3970, 3993, 4032,
};

static uint64_t round_up_u64(uint64_t v, uint64_t align) {
    return (v + align - 1) & ~(align - 1);
}

static NSString *qw3_metal_kernel_source(void) {
    return @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "inline float qw3_f16_to_f32(ushort h) {\n"
            "    uint s = uint(h >> 15u);\n"
            "    uint e = (uint(h) >> 10u) & 31u;\n"
            "    uint f = uint(h) & 1023u;\n"
            "    float sign = s ? -1.0f : 1.0f;\n"
            "    if (e == 0u) return f == 0u ? sign * 0.0f : sign * float(f) * exp2(-24.0f);\n"
            "    if (e == 31u) return f == 0u ? sign * INFINITY : NAN;\n"
            "    return sign * (1.0f + float(f) / 1024.0f) * exp2(float(e) - 15.0f);\n"
            "}\n"
            "struct qw3_rmsnorm_args { uint n; float eps; };\n"
            "kernel void qw3_rmsnorm_plain(constant qw3_rmsnorm_args &args,\n"
            "                              device const float *x,\n"
            "                              device float *y,\n"
            "                              threadgroup float *sh,\n"
            "                              ushort tid [[thread_index_in_threadgroup]],\n"
            "                              ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                              ushort lane [[thread_index_in_simdgroup]],\n"
            "                              ushort nt [[threads_per_threadgroup]]) {\n"
            "    float ss = 0.0f;\n"
            "    for (uint i = tid; i < args.n; i += nt) ss += x[i] * x[i];\n"
            "    ss = simd_sum(ss);\n"
            "    if (lane == 0) sh[simd_idx] = ss;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    ss = lane < 32 ? sh[lane] : 0.0f;\n"
            "    ss = simd_sum(ss);\n"
            "    float scale = rsqrt(ss / float(args.n) + args.eps);\n"
            "    for (uint i = tid; i < args.n; i += nt) y[i] = x[i] * scale;\n"
            "}\n"
            "kernel void qw3_rmsnorm_weight_f32(constant qw3_rmsnorm_args &args,\n"
            "                                   device const float *x,\n"
            "                                   device const float *w,\n"
            "                                   device float *y,\n"
            "                                   threadgroup float *sh,\n"
            "                                   ushort tid [[thread_index_in_threadgroup]],\n"
            "                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                   ushort lane [[thread_index_in_simdgroup]],\n"
            "                                   ushort nt [[threads_per_threadgroup]]) {\n"
            "    float ss = 0.0f;\n"
            "    for (uint i = tid; i < args.n; i += nt) ss += x[i] * x[i];\n"
            "    ss = simd_sum(ss);\n"
            "    if (lane == 0) sh[simd_idx] = ss;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    ss = lane < 32 ? sh[lane] : 0.0f;\n"
            "    ss = simd_sum(ss);\n"
            "    float scale = rsqrt(ss / float(args.n) + args.eps);\n"
            "    for (uint i = tid; i < args.n; i += nt) y[i] = x[i] * scale * w[i];\n"
            "}\n"
            "struct qw3_embed_q8_0_args { uint n_embd; uint row_bytes; };\n"
            "kernel void qw3_embed_q8_0(constant qw3_embed_q8_0_args &args,\n"
            "                           device const uchar *weights,\n"
            "                           device float *out,\n"
            "                           uint gid [[thread_position_in_grid]]) {\n"
            "    if (gid >= args.n_embd) return;\n"
            "    uint block = gid / 32;\n"
            "    uint lane = gid % 32;\n"
            "    device const uchar *blk = weights + block * 34;\n"
            "    half d = *((device const half *)blk);\n"
            "    char q = *((device const char *)(blk + 2 + lane));\n"
            "    out[gid] = float(d) * float(q);\n"
            "}\n"
            "struct qw3_kv_quant_q8_args { uint n; };\n"
            "kernel void qw3_gqa_kv_quant_q8(constant qw3_kv_quant_q8_args &args,\n"
            "                                device const float *k,\n"
            "                                device const float *v,\n"
            "                                device uchar *k_cache,\n"
            "                                device uchar *v_cache,\n"
            "                                uint block [[threadgroup_position_in_grid]],\n"
            "                                ushort tid [[thread_index_in_threadgroup]]) {\n"
            "    uint base = block * 32u;\n"
            "    if (base >= args.n || tid >= 32u) return;\n"
            "    float ka = fabs(k[base + uint(tid)]);\n"
            "    float va = fabs(v[base + uint(tid)]);\n"
            "    ka = simd_max(ka);\n"
            "    va = simd_max(va);\n"
            "    float kd = ka > 0.0f ? ka / 127.0f : 0.0f;\n"
            "    float vd = va > 0.0f ? va / 127.0f : 0.0f;\n"
            "    device uchar *kb = k_cache + uint64_t(block) * 34ull;\n"
            "    device uchar *vb = v_cache + uint64_t(block) * 34ull;\n"
            "    if (tid == 0) {\n"
            "        *((device half *)kb) = half(kd);\n"
            "        *((device half *)vb) = half(vd);\n"
            "    }\n"
            "    float kq = kd > 0.0f ? rint(k[base + uint(tid)] / kd) : 0.0f;\n"
            "    float vq = vd > 0.0f ? rint(v[base + uint(tid)] / vd) : 0.0f;\n"
            "    *((device char *)(kb + 2u + uint(tid))) = char(clamp(kq, -127.0f, 127.0f));\n"
            "    *((device char *)(vb + 2u + uint(tid))) = char(clamp(vq, -127.0f, 127.0f));\n"
            "}\n"
            "struct qw3_matvec_q8_0_args { uint n_in; uint n_out; uint row_bytes; };\n"
            "kernel void qw3_matvec_q8_0(constant qw3_matvec_q8_0_args &args,\n"
            "                            device const uchar *weights,\n"
            "                            device const float *x,\n"
            "                            device float *out,\n"
            "                            threadgroup float *sh,\n"
            "                            uint row [[threadgroup_position_in_grid]],\n"
            "                            ushort tid [[thread_index_in_threadgroup]],\n"
            "                            ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                            ushort lane [[thread_index_in_simdgroup]],\n"
            "                            ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (row >= args.n_out) return;\n"
            "    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);\n"
            "    float sum = 0.0f;\n"
            "    uint n_blocks = args.n_in / 32;\n"
            "    for (uint b = tid; b < n_blocks; b += nt) {\n"
            "        device const uchar *blk = wr + b * 34;\n"
            "        half d = *((device const half *)blk);\n"
            "        for (uint i = 0; i < 32; i++) {\n"
            "            char q = *((device const char *)(blk + 2 + i));\n"
            "            sum += float(d) * float(q) * x[b * 32 + i];\n"
            "        }\n"
            "    }\n"
            "    sum = simd_sum(sum);\n"
            "    if (lane == 0) sh[simd_idx] = sum;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    sum = lane < 32 ? sh[lane] : 0.0f;\n"
            "    sum = simd_sum(sum);\n"
            "    if (tid == 0) out[row] = sum;\n"
            "}\n"
            "struct qw3_matvec_q8_0_pair_args { uint n_in; uint n_out; uint row_bytes; uint out_a_offset; uint out_b_offset; };\n"
            "kernel void qw3_matvec_q8_0_pair(constant qw3_matvec_q8_0_pair_args &args,\n"
            "                                device const uchar *weights_a,\n"
            "                                device const uchar *weights_b,\n"
            "                                device const float *x,\n"
            "                                device float *out,\n"
            "                                threadgroup float *sh,\n"
            "                                uint row [[threadgroup_position_in_grid]],\n"
            "                                ushort tid [[thread_index_in_threadgroup]],\n"
            "                                ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                ushort lane [[thread_index_in_simdgroup]],\n"
            "                                ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (row >= args.n_out) return;\n"
            "    device const uchar *wra = weights_a + uint64_t(row) * uint64_t(args.row_bytes);\n"
            "    device const uchar *wrb = weights_b + uint64_t(row) * uint64_t(args.row_bytes);\n"
            "    float suma = 0.0f;\n"
            "    float sumb = 0.0f;\n"
            "    uint n_blocks = args.n_in / 32;\n"
            "    for (uint b = tid; b < n_blocks; b += nt) {\n"
            "        device const float *xx = x + b * 32;\n"
            "        device const uchar *blka = wra + b * 34;\n"
            "        device const uchar *blkb = wrb + b * 34;\n"
            "        half da = *((device const half *)blka);\n"
            "        half db = *((device const half *)blkb);\n"
            "        for (uint i = 0; i < 32; i++) {\n"
            "            char qa = *((device const char *)(blka + 2 + i));\n"
            "            char qb = *((device const char *)(blkb + 2 + i));\n"
            "            float xv = xx[i];\n"
            "            suma += float(da) * float(qa) * xv;\n"
            "            sumb += float(db) * float(qb) * xv;\n"
            "        }\n"
            "    }\n"
            "    suma = simd_sum(suma);\n"
            "    sumb = simd_sum(sumb);\n"
            "    if (lane == 0) {\n"
            "        sh[simd_idx] = suma;\n"
            "        sh[simd_idx + 32] = sumb;\n"
            "    }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    suma = lane < 32 ? sh[lane] : 0.0f;\n"
            "    sumb = lane < 32 ? sh[lane + 32] : 0.0f;\n"
            "    suma = simd_sum(suma);\n"
            "    sumb = simd_sum(sumb);\n"
            "    if (tid == 0) {\n"
            "        out[args.out_a_offset + row] = suma;\n"
            "        out[args.out_b_offset + row] = sumb;\n"
            "    }\n"
            "}\n"
            "kernel void qw3_matvec_q8_0_pair_silu_fast(constant qw3_matvec_q8_0_pair_args &args,\n"
            "                                             device const uchar *weights_a,\n"
            "                                             device const uchar *weights_b,\n"
            "                                             device const float *x,\n"
            "                                             device float *inner,\n"
            "                                             threadgroup float *sh,\n"
            "                                             uint group [[threadgroup_position_in_grid]],\n"
            "                                             ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                             ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    (void)sh;\n"
            "    const uint nr0 = 2u;\n"
            "    const uint nsg = 4u;\n"
            "    uint first_row = (group * nsg + uint(simd_idx)) * nr0;\n"
            "    float ga0 = 0.0f, up0 = 0.0f, ga1 = 0.0f, up1 = 0.0f;\n"
            "    uint n_blocks = args.n_in / 32u;\n"
            "    for (uint b = uint(lane); b < n_blocks; b += 32u) {\n"
            "        device const float *xx = x + uint64_t(b) * 32ull;\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) {\n"
            "            device const uchar *ba = weights_a + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;\n"
            "            device const uchar *bb = weights_b + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;\n"
            "            float da = float(*((device const half *)ba));\n"
            "            float db = float(*((device const half *)bb));\n"
            "            for (uint i = 0; i < 32u; i++) { float xv = xx[i]; ga0 += da * float(*((device const char *)(ba + 2u + i))) * xv; up0 += db * float(*((device const char *)(bb + 2u + i))) * xv; }\n"
            "        }\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) {\n"
            "            device const uchar *ba = weights_a + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;\n"
            "            device const uchar *bb = weights_b + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;\n"
            "            float da = float(*((device const half *)ba));\n"
            "            float db = float(*((device const half *)bb));\n"
            "            for (uint i = 0; i < 32u; i++) { float xv = xx[i]; ga1 += da * float(*((device const char *)(ba + 2u + i))) * xv; up1 += db * float(*((device const char *)(bb + 2u + i))) * xv; }\n"
            "        }\n"
            "    }\n"
            "    ga0 = simd_sum(ga0); up0 = simd_sum(up0);\n"
            "    ga1 = simd_sum(ga1); up1 = simd_sum(up1);\n"
            "    if (lane == 0) {\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) inner[row] = (ga0 / (1.0f + exp(-ga0))) * up0;\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) inner[row] = (ga1 / (1.0f + exp(-ga1))) * up1;\n"
            "    }\n"
            "}\n"
            "struct qw3_shared_gate_up_args { uint n_in; uint n_out; uint row_bytes; uint scalar_offset; };\n"
            "kernel void qw3_shared_gate_up_silu_fast(constant qw3_shared_gate_up_args &args,\n"
            "                                         device const uchar *weights_a,\n"
            "                                         device const uchar *weights_b,\n"
            "                                         device const float *scalar_weights,\n"
            "                                         device const float *x,\n"
            "                                         device float *inner,\n"
            "                                         device float *scratch,\n"
            "                                         threadgroup float *sh,\n"
            "                                         uint group [[threadgroup_position_in_grid]],\n"
            "                                         ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                         ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    const uint nr0 = 2u;\n"
            "    const uint nsg = 4u;\n"
            "    const uint pair_groups = (args.n_out + nr0 * nsg - 1u) / (nr0 * nsg);\n"
            "    if (group == pair_groups) {\n"
            "        uint tid = uint(simd_idx) * 32u + uint(lane);\n"
            "        float sum = 0.0f;\n"
            "        for (uint i = tid; i < args.n_in; i += 128u) sum += scalar_weights[i] * x[i];\n"
            "        sum = simd_sum(sum);\n"
            "        if (lane == 0) sh[simd_idx] = sum;\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        sum = lane < 4u ? sh[lane] : 0.0f;\n"
            "        sum = simd_sum(sum);\n"
            "        if (tid == 0u) scratch[args.scalar_offset] = sum;\n"
            "        return;\n"
            "    }\n"
            "    uint first_row = (group * nsg + uint(simd_idx)) * nr0;\n"
            "    float ga0 = 0.0f, up0 = 0.0f, ga1 = 0.0f, up1 = 0.0f;\n"
            "    uint n_blocks = args.n_in / 32u;\n"
            "    for (uint b = uint(lane); b < n_blocks; b += 32u) {\n"
            "        device const float *xx = x + uint64_t(b) * 32ull;\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) {\n"
            "            device const uchar *ba = weights_a + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;\n"
            "            device const uchar *bb = weights_b + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;\n"
            "            float da = float(*((device const half *)ba));\n"
            "            float db = float(*((device const half *)bb));\n"
            "            for (uint i = 0; i < 32u; i++) { float xv = xx[i]; ga0 += da * float(*((device const char *)(ba + 2u + i))) * xv; up0 += db * float(*((device const char *)(bb + 2u + i))) * xv; }\n"
            "        }\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) {\n"
            "            device const uchar *ba = weights_a + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;\n"
            "            device const uchar *bb = weights_b + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;\n"
            "            float da = float(*((device const half *)ba));\n"
            "            float db = float(*((device const half *)bb));\n"
            "            for (uint i = 0; i < 32u; i++) { float xv = xx[i]; ga1 += da * float(*((device const char *)(ba + 2u + i))) * xv; up1 += db * float(*((device const char *)(bb + 2u + i))) * xv; }\n"
            "        }\n"
            "    }\n"
            "    ga0 = simd_sum(ga0); up0 = simd_sum(up0);\n"
            "    ga1 = simd_sum(ga1); up1 = simd_sum(up1);\n"
            "    if (lane == 0) {\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) inner[row] = (ga0 / (1.0f + exp(-ga0))) * up0;\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) inner[row] = (ga1 / (1.0f + exp(-ga1))) * up1;\n"
            "    }\n"
            "}\n"
            "struct qw3_matvec_q8_0_scale_args { uint n_in; uint n_out; uint row_bytes; uint scalar_offset; };\n"
            "kernel void qw3_matvec_q8_0_inner_scale_add_x0(constant qw3_matvec_q8_0_scale_args &args,\n"
            "                                               device const uchar *weights,\n"
            "                                               device const float *x,\n"
            "                                               device const float *scratch,\n"
            "                                               device float *x0,\n"
            "                                               threadgroup float *sh,\n"
            "                                               uint row [[threadgroup_position_in_grid]],\n"
            "                                               ushort tid [[thread_index_in_threadgroup]],\n"
            "                                               ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                               ushort lane [[thread_index_in_simdgroup]],\n"
            "                                               ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (row >= args.n_out) return;\n"
            "    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);\n"
            "    float sum = 0.0f;\n"
            "    uint n_blocks = args.n_in / 32;\n"
            "    for (uint b = tid; b < n_blocks; b += nt) {\n"
            "        device const uchar *blk = wr + b * 34;\n"
            "        half d = *((device const half *)blk);\n"
            "        for (uint i = 0; i < 32; i++) {\n"
            "            char q = *((device const char *)(blk + 2 + i));\n"
            "            sum += float(d) * float(q) * x[b * 32 + i];\n"
            "        }\n"
            "    }\n"
            "    sum = simd_sum(sum);\n"
            "    if (lane == 0) sh[simd_idx] = sum;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    sum = lane < 32 ? sh[lane] : 0.0f;\n"
            "    sum = simd_sum(sum);\n"
            "    if (tid == 0) {\n"
            "        float raw = scratch[args.scalar_offset];\n"
            "        float scale = 1.0f / (1.0f + exp(-raw));\n"
            "        x0[row] = x0[row] + sum * scale;\n"
            "    }\n"
            "}\n"
            "kernel void qw3_matvec_q8_0_inner_scale_add_x0_fast(constant qw3_matvec_q8_0_scale_args &args,\n"
            "                                                    device const uchar *weights,\n"
            "                                                    device const float *x,\n"
            "                                                    device const float *scratch,\n"
            "                                                    device float *x0,\n"
            "                                                    threadgroup float *sh,\n"
            "                                                    uint group [[threadgroup_position_in_grid]],\n"
            "                                                    ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                                    ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    (void)sh;\n"
            "    const uint nr0 = 2u;\n"
            "    const uint nsg = 4u;\n"
            "    uint first_row = (group * nsg + uint(simd_idx)) * nr0;\n"
            "    float sum0 = 0.0f;\n"
            "    float sum1 = 0.0f;\n"
            "    uint n_blocks = args.n_in / 32u;\n"
            "    for (uint b = uint(lane); b < n_blocks; b += 32u) {\n"
            "        device const float *xx = x + uint64_t(b) * 32ull;\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) {\n"
            "            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;\n"
            "            float d = float(*((device const half *)blk));\n"
            "            for (uint i = 0; i < 32u; i++) sum0 += d * float(*((device const char *)(blk + 2u + i))) * xx[i];\n"
            "        }\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) {\n"
            "            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;\n"
            "            float d = float(*((device const half *)blk));\n"
            "            for (uint i = 0; i < 32u; i++) sum1 += d * float(*((device const char *)(blk + 2u + i))) * xx[i];\n"
            "        }\n"
            "    }\n"
            "    sum0 = simd_sum(sum0);\n"
            "    sum1 = simd_sum(sum1);\n"
            "    if (lane == 0) {\n"
            "        float raw = scratch[args.scalar_offset];\n"
            "        float scale = 1.0f / (1.0f + exp(-raw));\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) x0[row] = x0[row] + sum0 * scale;\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) x0[row] = x0[row] + sum1 * scale;\n"
            "    }\n"
            "}\n"
            "kernel void qw3_matvec_q8_0_fast(constant qw3_matvec_q8_0_args &args,\n"
            "                                 device const uchar *weights,\n"
            "                                 device const float *x,\n"
            "                                 device float *out,\n"
            "                                 threadgroup float *sh,\n"
            "                                 uint group [[threadgroup_position_in_grid]],\n"
            "                                 ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                 ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    (void)sh;\n"
            "    const uint nr0 = 2u;\n"
            "    const uint nsg = 4u;\n"
            "    uint first_row = (group * nsg + uint(simd_idx)) * nr0;\n"
            "    float sum0 = 0.0f;\n"
            "    float sum1 = 0.0f;\n"
            "    uint n_blocks = args.n_in / 32u;\n"
            "    for (uint b = uint(lane); b < n_blocks; b += 32u) {\n"
            "        device const float *xx = x + uint64_t(b) * 32ull;\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) {\n"
            "            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;\n"
            "            half d = *((device const half *)blk);\n"
            "            float ds = float(d);\n"
            "            for (uint i = 0; i < 32u; i++) { char q = *((device const char *)(blk + 2u + i)); sum0 += ds * float(q) * xx[i]; }\n"
            "        }\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) {\n"
            "            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 34ull;\n"
            "            half d = *((device const half *)blk);\n"
            "            float ds = float(d);\n"
            "            for (uint i = 0; i < 32u; i++) { char q = *((device const char *)(blk + 2u + i)); sum1 += ds * float(q) * xx[i]; }\n"
            "        }\n"
            "    }\n"
            "    sum0 = simd_sum(sum0);\n"
            "    sum1 = simd_sum(sum1);\n"
            "    if (lane == 0) {\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) out[row] = sum0;\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) out[row] = sum1;\n"
            "    }\n"
            "}\n"
            "inline float qw3_iq4nl_val(uint q) {\n"
            "    switch (q & 15u) {\n"
            "        case 0u: return -127.0f; case 1u: return -104.0f;\n"
            "        case 2u: return -83.0f;  case 3u: return -65.0f;\n"
            "        case 4u: return -49.0f;  case 5u: return -35.0f;\n"
            "        case 6u: return -22.0f;  case 7u: return -10.0f;\n"
            "        case 8u: return 1.0f;    case 9u: return 13.0f;\n"
            "        case 10u: return 25.0f;  case 11u: return 38.0f;\n"
            "        case 12u: return 53.0f;  case 13u: return 69.0f;\n"
            "        case 14u: return 89.0f;  default: return 113.0f;\n"
            "    }\n"
            "}\n"
            "kernel void qw3_matvec_iq4_xs(constant qw3_matvec_q8_0_args &args,\n"
            "                              device const uchar *weights,\n"
            "                              device const float *x,\n"
            "                              device float *out,\n"
            "                              threadgroup float *sh,\n"
            "                              uint row [[threadgroup_position_in_grid]],\n"
            "                              ushort tid [[thread_index_in_threadgroup]],\n"
            "                              ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                              ushort lane [[thread_index_in_simdgroup]],\n"
            "                              ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (row >= args.n_out) return;\n"
            "    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);\n"
            "    float sum = 0.0f;\n"
            "    uint n_blocks = args.n_in / 256u;\n"
            "    for (uint b = tid; b < n_blocks; b += nt) {\n"
            "        device const uchar *blk = wr + uint64_t(b) * 136ull;\n"
            "        half d = *((device const half *)blk);\n"
            "        ushort scales_h = *((device const ushort *)(blk + 2));\n"
            "        device const uchar *scales_l = blk + 4;\n"
            "        device const uchar *qs = scales_l + 4;\n"
            "        device const float *xx = x + uint64_t(b) * 256ull;\n"
            "        for (uint ib = 0; ib < 8u; ib++) {\n"
            "            uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) |\n"
            "                      (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);\n"
            "            float dl = float(d) * float(int(ls) - 32);\n"
            "            device const uchar *q = qs + ib * 16u;\n"
            "            device const float *xg = xx + ib * 32u;\n"
            "            for (uint j = 0; j < 16u; j++) {\n"
            "                uchar v = q[j];\n"
            "                sum += dl * qw3_iq4nl_val(uint(v) & 15u) * xg[j];\n"
            "                sum += dl * qw3_iq4nl_val(uint(v) >> 4u) * xg[j + 16u];\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    sum = simd_sum(sum);\n"
            "    if (lane == 0) sh[simd_idx] = sum;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    sum = lane < 32 ? sh[lane] : 0.0f;\n"
            "    sum = simd_sum(sum);\n"
            "    if (tid == 0) out[row] = sum;\n"
            "}\n"
            "kernel void qw3_matvec_iq4_xs_add_x0(constant qw3_matvec_q8_0_args &args,\n"
            "                                     constant float &scale,\n"
            "                                     device const uchar *weights,\n"
            "                                     device const float *x,\n"
            "                                     device float *x0,\n"
            "                                     threadgroup float *sh,\n"
            "                                     uint row [[threadgroup_position_in_grid]],\n"
            "                                     ushort tid [[thread_index_in_threadgroup]],\n"
            "                                     ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                     ushort lane [[thread_index_in_simdgroup]],\n"
            "                                     ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (row >= args.n_out) return;\n"
            "    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);\n"
            "    float sum = 0.0f;\n"
            "    uint n_blocks = args.n_in / 256u;\n"
            "    for (uint b = tid; b < n_blocks; b += nt) {\n"
            "        device const uchar *blk = wr + uint64_t(b) * 136ull;\n"
            "        half d = *((device const half *)blk);\n"
            "        ushort scales_h = *((device const ushort *)(blk + 2));\n"
            "        device const uchar *scales_l = blk + 4;\n"
            "        device const uchar *qs = scales_l + 4;\n"
            "        device const float *xx = x + uint64_t(b) * 256ull;\n"
            "        for (uint ib = 0; ib < 8u; ib++) {\n"
            "            uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) |\n"
            "                      (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);\n"
            "            float dl = float(d) * float(int(ls) - 32);\n"
            "            device const uchar *q = qs + ib * 16u;\n"
            "            device const float *xg = xx + ib * 32u;\n"
            "            for (uint j = 0; j < 16u; j++) {\n"
            "                uchar v = q[j];\n"
            "                sum += dl * qw3_iq4nl_val(uint(v) & 15u) * xg[j];\n"
            "                sum += dl * qw3_iq4nl_val(uint(v) >> 4u) * xg[j + 16u];\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    sum = simd_sum(sum);\n"
            "    if (lane == 0) sh[simd_idx] = sum;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    sum = lane < 32 ? sh[lane] : 0.0f;\n"
            "    sum = simd_sum(sum);\n"
            "    if (tid == 0) x0[row] = x0[row] + sum * scale;\n"
            "}\n"
            "kernel void qw3_matvec_iq4_xs_add_x0_fast(constant qw3_matvec_q8_0_args &args,\n"
            "                                          constant float &scale,\n"
            "                                          device const uchar *weights,\n"
            "                                          device const float *x,\n"
            "                                          device float *x0,\n"
            "                                          threadgroup float *sh,\n"
            "                                          uint group [[threadgroup_position_in_grid]],\n"
            "                                          ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                          ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    (void)sh;\n"
            "    const uint nr0 = 2u;\n"
            "    const uint nsg = 2u;\n"
            "    uint first_row = (group * nsg + uint(simd_idx)) * nr0;\n"
            "    uint ix = uint(lane) >> 4u;\n"
            "    uint it = uint(lane) & 15u;\n"
            "    uint ib = it >> 1u;\n"
            "    uint il = it & 1u;\n"
            "    float sum0 = 0.0f;\n"
            "    float sum1 = 0.0f;\n"
            "    uint n_blocks = args.n_in / 256u;\n"
            "    for (uint b = ix; b < n_blocks; b += 2u) {\n"
            "        device const float *xg = x + uint64_t(b) * 256ull + uint64_t(ib) * 32ull + uint64_t(il) * 8ull;\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) {\n"
            "            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;\n"
            "            half d = *((device const half *)blk);\n"
            "            ushort scales_h = *((device const ushort *)(blk + 2));\n"
            "            device const uchar *scales_l = blk + 4;\n"
            "            device const uchar *qs = scales_l + 4 + ib * 16u + il * 8u;\n"
            "            uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);\n"
            "            float dl = float(d) * float(int(ls) - 32);\n"
            "            float acc = 0.0f;\n"
            "            for (uint j = 0; j < 8u; j++) { uchar v = qs[j]; acc += qw3_iq4nl_val(uint(v) & 15u) * xg[j]; acc += qw3_iq4nl_val(uint(v) >> 4u) * xg[j + 16u]; }\n"
            "            sum0 += dl * acc;\n"
            "        }\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) {\n"
            "            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;\n"
            "            half d = *((device const half *)blk);\n"
            "            ushort scales_h = *((device const ushort *)(blk + 2));\n"
            "            device const uchar *scales_l = blk + 4;\n"
            "            device const uchar *qs = scales_l + 4 + ib * 16u + il * 8u;\n"
            "            uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);\n"
            "            float dl = float(d) * float(int(ls) - 32);\n"
            "            float acc = 0.0f;\n"
            "            for (uint j = 0; j < 8u; j++) { uchar v = qs[j]; acc += qw3_iq4nl_val(uint(v) & 15u) * xg[j]; acc += qw3_iq4nl_val(uint(v) >> 4u) * xg[j + 16u]; }\n"
            "            sum1 += dl * acc;\n"
            "        }\n"
            "    }\n"
            "    sum0 = simd_sum(sum0);\n"
            "    sum1 = simd_sum(sum1);\n"
            "    if (lane == 0) {\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) x0[row] = x0[row] + sum0 * scale;\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) x0[row] = x0[row] + sum1 * scale;\n"
            "    }\n"
            "}\n"
            "inline float qw3_swiglu_val(device const float *scratch, uint n, uint idx) {\n"
            "    float g = scratch[idx];\n"
            "    return (g / (1.0f + exp(-g))) * scratch[n + idx];\n"
            "}\n"
            "kernel void qw3_matvec_iq4_xs_swiglu_add_x0(constant qw3_matvec_q8_0_args &args,\n"
            "                                            constant float &scale,\n"
            "                                            device const uchar *weights,\n"
            "                                            device const float *scratch,\n"
            "                                            device float *x0,\n"
            "                                            threadgroup float *sh,\n"
            "                                            uint row [[threadgroup_position_in_grid]],\n"
            "                                            ushort tid [[thread_index_in_threadgroup]],\n"
            "                                            ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                            ushort lane [[thread_index_in_simdgroup]],\n"
            "                                            ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (row >= args.n_out) return;\n"
            "    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);\n"
            "    float sum = 0.0f;\n"
            "    uint n_blocks = args.n_in / 256u;\n"
            "    for (uint b = tid; b < n_blocks; b += nt) {\n"
            "        device const uchar *blk = wr + uint64_t(b) * 136ull;\n"
            "        half d = *((device const half *)blk);\n"
            "        ushort scales_h = *((device const ushort *)(blk + 2));\n"
            "        device const uchar *scales_l = blk + 4;\n"
            "        device const uchar *qs = scales_l + 4;\n"
            "        uint xb = b * 256u;\n"
            "        for (uint ib = 0; ib < 8u; ib++) {\n"
            "            uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);\n"
            "            float dl = float(d) * float(int(ls) - 32);\n"
            "            device const uchar *q = qs + ib * 16u;\n"
            "            uint xg = xb + ib * 32u;\n"
            "            for (uint j = 0; j < 16u; j++) { uchar v = q[j]; sum += dl * qw3_iq4nl_val(uint(v) & 15u) * qw3_swiglu_val(scratch, args.n_in, xg + j); sum += dl * qw3_iq4nl_val(uint(v) >> 4u) * qw3_swiglu_val(scratch, args.n_in, xg + j + 16u); }\n"
            "        }\n"
            "    }\n"
            "    sum = simd_sum(sum); if (lane == 0) sh[simd_idx] = sum; threadgroup_barrier(mem_flags::mem_threadgroup); sum = lane < 32 ? sh[lane] : 0.0f; sum = simd_sum(sum); if (tid == 0) x0[row] = x0[row] + sum * scale;\n"
            "}\n"
            "kernel void qw3_matvec_q6_k(constant qw3_matvec_q8_0_args &args,\n"
            "                            device const uchar *weights,\n"
            "                            device const float *x,\n"
            "                            device float *out,\n"
            "                            threadgroup float *sh,\n"
            "                            uint row [[threadgroup_position_in_grid]],\n"
            "                            ushort tid [[thread_index_in_threadgroup]],\n"
            "                            ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                            ushort lane [[thread_index_in_simdgroup]],\n"
            "                            ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (row >= args.n_out) return;\n"
            "    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);\n"
            "    float sum = 0.0f;\n"
            "    uint n_blocks = args.n_in / 256u;\n"
            "    for (uint b = tid; b < n_blocks; b += nt) {\n"
            "        device const uchar *blk = wr + uint64_t(b) * 210ull;\n"
            "        device const uchar *ql = blk;\n"
            "        device const uchar *qh = ql + 128u;\n"
            "        device const uchar *scb = qh + 64u;\n"
            "        device const char *sc = (device const char *)scb;\n"
            "        ushort dbits = ushort(scb[16u]) | (ushort(scb[17u]) << 8u);\n"
            "        float d = qw3_f16_to_f32(dbits);\n"
            "        device const float *xx = x + uint64_t(b) * 256ull;\n"
            "        for (uint n = 0; n < 256u; n += 128u) {\n"
            "            for (uint l = 0; l < 32u; l++) {\n"
            "                uint is = l / 16u;\n"
            "                int q1 = int((uint(ql[l +  0u]) & 15u) | (((uint(qh[l]) >> 0u) & 3u) << 4u)) - 32;\n"
            "                int q2 = int((uint(ql[l + 32u]) & 15u) | (((uint(qh[l]) >> 2u) & 3u) << 4u)) - 32;\n"
            "                int q3 = int((uint(ql[l +  0u]) >> 4u) | (((uint(qh[l]) >> 4u) & 3u) << 4u)) - 32;\n"
            "                int q4 = int((uint(ql[l + 32u]) >> 4u) | (((uint(qh[l]) >> 6u) & 3u) << 4u)) - 32;\n"
            "                sum += d * float(sc[is + 0u]) * float(q1) * xx[n + l +  0u];\n"
            "                sum += d * float(sc[is + 2u]) * float(q2) * xx[n + l + 32u];\n"
            "                sum += d * float(sc[is + 4u]) * float(q3) * xx[n + l + 64u];\n"
            "                sum += d * float(sc[is + 6u]) * float(q4) * xx[n + l + 96u];\n"
            "            }\n"
            "            ql += 64u;\n"
            "            qh += 32u;\n"
            "            sc += 8u;\n"
            "        }\n"
            "    }\n"
            "    sum = simd_sum(sum);\n"
            "    if (lane == 0) sh[simd_idx] = sum;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    sum = lane < 32 ? sh[lane] : 0.0f;\n"
            "    sum = simd_sum(sum);\n"
            "    if (tid == 0) out[row] = sum;\n"
            "}\n"
            "kernel void qw3_matvec_q6_k_fast(constant qw3_matvec_q8_0_args &args,\n"
            "                                 device const uchar *weights,\n"
            "                                 device const float *x,\n"
            "                                 device float *out,\n"
            "                                 threadgroup float *sh,\n"
            "                                 uint group [[threadgroup_position_in_grid]],\n"
            "                                 ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                 ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    (void)sh;\n"
            "    const uint nr0 = 2u;\n"
            "    const uint nsg = 2u;\n"
            "    uint first_row = (group * nsg + uint(simd_idx)) * nr0;\n"
            "    uint tid = uint(lane) >> 1u;\n"
            "    uint ix = uint(lane) & 1u;\n"
            "    uint ip = tid >> 3u;\n"
            "    uint il = tid & 7u;\n"
            "    uint l0 = 4u * il;\n"
            "    uint is = 8u * ip + l0 / 16u;\n"
            "    uint y_offset = 128u * ip + l0;\n"
            "    uint q_offset_l = 64u * ip + l0;\n"
            "    uint q_offset_h = 32u * ip + l0;\n"
            "    float sum0 = 0.0f;\n"
            "    float sum1 = 0.0f;\n"
            "    uint n_blocks = args.n_in / 256u;\n"
            "    for (uint b = ix; b < n_blocks; b += 2u) {\n"
            "        device const float *yy = x + uint64_t(b) * 256ull + y_offset;\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) {\n"
            "            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 210ull;\n"
            "            device const uchar *q1 = blk + q_offset_l;\n"
            "            device const uchar *q2 = q1 + 32u;\n"
            "            device const uchar *qh = blk + 128u + q_offset_h;\n"
            "            device const char *sc = (device const char *)(blk + 192u + is);\n"
            "            ushort dbits = ushort(blk[208u]) | (ushort(blk[209u]) << 8u);\n"
            "            float d = qw3_f16_to_f32(dbits);\n"
            "            float acc = 0.0f;\n"
            "            for (uint l = 0; l < 4u; l++) {\n"
            "                int qv1 = int((uint(q1[l]) & 15u) | (((uint(qh[l]) >> 0u) & 3u) << 4u)) - 32;\n"
            "                int qv2 = int((uint(q2[l]) & 15u) | (((uint(qh[l]) >> 2u) & 3u) << 4u)) - 32;\n"
            "                int qv3 = int((uint(q1[l]) >> 4u) | (((uint(qh[l]) >> 4u) & 3u) << 4u)) - 32;\n"
            "                int qv4 = int((uint(q2[l]) >> 4u) | (((uint(qh[l]) >> 6u) & 3u) << 4u)) - 32;\n"
            "                acc += float(sc[0]) * float(qv1) * yy[l + 0u];\n"
            "                acc += float(sc[2]) * float(qv2) * yy[l + 32u];\n"
            "                acc += float(sc[4]) * float(qv3) * yy[l + 64u];\n"
            "                acc += float(sc[6]) * float(qv4) * yy[l + 96u];\n"
            "            }\n"
            "            sum0 += d * acc;\n"
            "        }\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) {\n"
            "            device const uchar *blk = weights + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 210ull;\n"
            "            device const uchar *q1 = blk + q_offset_l;\n"
            "            device const uchar *q2 = q1 + 32u;\n"
            "            device const uchar *qh = blk + 128u + q_offset_h;\n"
            "            device const char *sc = (device const char *)(blk + 192u + is);\n"
            "            ushort dbits = ushort(blk[208u]) | (ushort(blk[209u]) << 8u);\n"
            "            float d = qw3_f16_to_f32(dbits);\n"
            "            float acc = 0.0f;\n"
            "            for (uint l = 0; l < 4u; l++) {\n"
            "                int qv1 = int((uint(q1[l]) & 15u) | (((uint(qh[l]) >> 0u) & 3u) << 4u)) - 32;\n"
            "                int qv2 = int((uint(q2[l]) & 15u) | (((uint(qh[l]) >> 2u) & 3u) << 4u)) - 32;\n"
            "                int qv3 = int((uint(q1[l]) >> 4u) | (((uint(qh[l]) >> 4u) & 3u) << 4u)) - 32;\n"
            "                int qv4 = int((uint(q2[l]) >> 4u) | (((uint(qh[l]) >> 6u) & 3u) << 4u)) - 32;\n"
            "                acc += float(sc[0]) * float(qv1) * yy[l + 0u];\n"
            "                acc += float(sc[2]) * float(qv2) * yy[l + 32u];\n"
            "                acc += float(sc[4]) * float(qv3) * yy[l + 64u];\n"
            "                acc += float(sc[6]) * float(qv4) * yy[l + 96u];\n"
            "            }\n"
            "            sum1 += d * acc;\n"
            "        }\n"
            "    }\n"
            "    sum0 = simd_sum(sum0);\n"
            "    sum1 = simd_sum(sum1);\n"
            "    if (lane == 0) {\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) out[row] = sum0;\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) out[row] = sum1;\n"
            "    }\n"
            "}\n"
            "kernel void qw3_matvec_q6_k_add_x0(constant qw3_matvec_q8_0_args &args,\n"
            "                                   constant float &scale,\n"
            "                                   device const uchar *weights,\n"
            "                                   device const float *x,\n"
            "                                   device float *x0,\n"
            "                                   threadgroup float *sh,\n"
            "                                   uint row [[threadgroup_position_in_grid]],\n"
            "                                   ushort tid [[thread_index_in_threadgroup]],\n"
            "                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                   ushort lane [[thread_index_in_simdgroup]],\n"
            "                                   ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (row >= args.n_out) return;\n"
            "    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);\n"
            "    float sum = 0.0f;\n"
            "    uint n_blocks = args.n_in / 256u;\n"
            "    for (uint b = tid; b < n_blocks; b += nt) {\n"
            "        device const uchar *blk = wr + uint64_t(b) * 210ull;\n"
            "        device const uchar *ql = blk;\n"
            "        device const uchar *qh = ql + 128u;\n"
            "        device const uchar *scb = qh + 64u;\n"
            "        device const char *sc = (device const char *)scb;\n"
            "        ushort dbits = ushort(scb[16u]) | (ushort(scb[17u]) << 8u);\n"
            "        float d = qw3_f16_to_f32(dbits);\n"
            "        device const float *xx = x + uint64_t(b) * 256ull;\n"
            "        for (uint n = 0; n < 256u; n += 128u) {\n"
            "            for (uint l = 0; l < 32u; l++) {\n"
            "                uint is = l / 16u;\n"
            "                int q1 = int((uint(ql[l +  0u]) & 15u) | (((uint(qh[l]) >> 0u) & 3u) << 4u)) - 32;\n"
            "                int q2 = int((uint(ql[l + 32u]) & 15u) | (((uint(qh[l]) >> 2u) & 3u) << 4u)) - 32;\n"
            "                int q3 = int((uint(ql[l +  0u]) >> 4u) | (((uint(qh[l]) >> 4u) & 3u) << 4u)) - 32;\n"
            "                int q4 = int((uint(ql[l + 32u]) >> 4u) | (((uint(qh[l]) >> 6u) & 3u) << 4u)) - 32;\n"
            "                sum += d * float(sc[is + 0u]) * float(q1) * xx[n + l +  0u];\n"
            "                sum += d * float(sc[is + 2u]) * float(q2) * xx[n + l + 32u];\n"
            "                sum += d * float(sc[is + 4u]) * float(q3) * xx[n + l + 64u];\n"
            "                sum += d * float(sc[is + 6u]) * float(q4) * xx[n + l + 96u];\n"
            "            }\n"
            "            ql += 64u;\n"
            "            qh += 32u;\n"
            "            sc += 8u;\n"
            "        }\n"
            "    }\n"
            "    sum = simd_sum(sum);\n"
            "    if (lane == 0) sh[simd_idx] = sum;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    sum = lane < 32 ? sh[lane] : 0.0f;\n"
            "    sum = simd_sum(sum);\n"
            "    if (tid == 0) x0[row] = x0[row] + sum * scale;\n"
            "}\n"
            "kernel void qw3_matvec_q6_k_swiglu_add_x0(constant qw3_matvec_q8_0_args &args,\n"
            "                                          constant float &scale,\n"
            "                                          device const uchar *weights,\n"
            "                                          device const float *scratch,\n"
            "                                          device float *x0,\n"
            "                                          threadgroup float *sh,\n"
            "                                          uint row [[threadgroup_position_in_grid]],\n"
            "                                          ushort tid [[thread_index_in_threadgroup]],\n"
            "                                          ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                          ushort lane [[thread_index_in_simdgroup]],\n"
            "                                          ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (row >= args.n_out) return;\n"
            "    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);\n"
            "    float sum = 0.0f;\n"
            "    uint n_blocks = args.n_in / 256u;\n"
            "    for (uint b = tid; b < n_blocks; b += nt) {\n"
            "        device const uchar *blk = wr + uint64_t(b) * 210ull;\n"
            "        device const uchar *ql = blk; device const uchar *qh = ql + 128u; device const uchar *scb = qh + 64u; device const char *sc = (device const char *)scb;\n"
            "        ushort dbits = ushort(scb[16u]) | (ushort(scb[17u]) << 8u); float d = qw3_f16_to_f32(dbits); uint xb = b * 256u;\n"
            "        for (uint n = 0; n < 256u; n += 128u) {\n"
            "            for (uint l = 0; l < 32u; l++) {\n"
            "                uint is = l / 16u;\n"
            "                int q1 = int((uint(ql[l +  0u]) & 15u) | (((uint(qh[l]) >> 0u) & 3u) << 4u)) - 32;\n"
            "                int q2 = int((uint(ql[l + 32u]) & 15u) | (((uint(qh[l]) >> 2u) & 3u) << 4u)) - 32;\n"
            "                int q3 = int((uint(ql[l +  0u]) >> 4u) | (((uint(qh[l]) >> 4u) & 3u) << 4u)) - 32;\n"
            "                int q4 = int((uint(ql[l + 32u]) >> 4u) | (((uint(qh[l]) >> 6u) & 3u) << 4u)) - 32;\n"
            "                sum += d * float(sc[is + 0u]) * float(q1) * qw3_swiglu_val(scratch, args.n_in, xb + n + l +  0u);\n"
            "                sum += d * float(sc[is + 2u]) * float(q2) * qw3_swiglu_val(scratch, args.n_in, xb + n + l + 32u);\n"
            "                sum += d * float(sc[is + 4u]) * float(q3) * qw3_swiglu_val(scratch, args.n_in, xb + n + l + 64u);\n"
            "                sum += d * float(sc[is + 6u]) * float(q4) * qw3_swiglu_val(scratch, args.n_in, xb + n + l + 96u);\n"
            "            }\n"
            "            ql += 64u; qh += 32u; sc += 8u;\n"
            "        }\n"
            "    }\n"
            "    sum = simd_sum(sum); if (lane == 0) sh[simd_idx] = sum; threadgroup_barrier(mem_flags::mem_threadgroup); sum = lane < 32 ? sh[lane] : 0.0f; sum = simd_sum(sum); if (tid == 0) x0[row] = x0[row] + sum * scale;\n"
            "}\n"
            "inline float qw3_iq3s_grid_val(device const ushort *kgrid, uint idx, uint j) {\n"
            "    ushort packed = kgrid[idx & 511u];\n"
            "    return float(2u * ((uint(packed) >> (3u * j)) & 7u) + 1u);\n"
            "}\n"
            "kernel void qw3_matvec_iq3_s(constant qw3_matvec_q8_0_args &args,\n"
            "                             device const uchar *weights,\n"
            "                             device const float *x,\n"
            "                             device float *out,\n"
            "                             device const ushort *kgrid,\n"
            "                             threadgroup float *sh,\n"
            "                             uint row [[threadgroup_position_in_grid]],\n"
            "                             ushort tid [[thread_index_in_threadgroup]],\n"
            "                             ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                             ushort lane [[thread_index_in_simdgroup]],\n"
            "                             ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (row >= args.n_out) return;\n"
            "    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);\n"
            "    float sum = 0.0f;\n"
            "    uint n_blocks = args.n_in / 256u;\n"
            "    for (uint b = tid; b < n_blocks; b += nt) {\n"
            "        device const uchar *blk = wr + uint64_t(b) * 110ull;\n"
            "        half d = *((device const half *)blk);\n"
            "        device const uchar *qs = blk + 2;\n"
            "        device const uchar *qh = qs + 64;\n"
            "        device const uchar *signs = qh + 8;\n"
            "        device const uchar *scales = signs + 32;\n"
            "        device const float *xx = x + uint64_t(b) * 256ull;\n"
            "        uint xo = 0;\n"
            "        for (uint ib32 = 0; ib32 < 8u; ib32 += 2u) {\n"
            "            float db1 = float(d) * float(1u + 2u * (uint(scales[ib32 / 2u]) & 15u));\n"
            "            float db2 = float(d) * float(1u + 2u * (uint(scales[ib32 / 2u]) >> 4u));\n"
            "            uchar qh0 = qh[0];\n"
            "            uchar qh1 = qh[1];\n"
            "            for (uint l = 0; l < 4u; l++) {\n"
            "                uint idx1 = uint(qs[2u * l + 0u]) | ((uint(qh0) << (8u - 2u * l)) & 256u);\n"
            "                uint idx2 = uint(qs[2u * l + 1u]) | ((uint(qh0) << (7u - 2u * l)) & 256u);\n"
            "                uchar s = signs[l];\n"
            "                for (uint j = 0; j < 4u; j++) {\n"
            "                    float sign1 = (uint(s) & (1u << j)) ? -1.0f : 1.0f;\n"
            "                    float sign2 = (uint(s) & (1u << (j + 4u))) ? -1.0f : 1.0f;\n"
            "                    sum += db1 * qw3_iq3s_grid_val(kgrid, idx1, j) * sign1 * xx[xo + j + 0u];\n"
            "                    sum += db1 * qw3_iq3s_grid_val(kgrid, idx2, j) * sign2 * xx[xo + j + 4u];\n"
            "                }\n"
            "                xo += 8u;\n"
            "            }\n"
            "            qs += 8;\n"
            "            signs += 4;\n"
            "            for (uint l = 0; l < 4u; l++) {\n"
            "                uint idx1 = uint(qs[2u * l + 0u]) | ((uint(qh1) << (8u - 2u * l)) & 256u);\n"
            "                uint idx2 = uint(qs[2u * l + 1u]) | ((uint(qh1) << (7u - 2u * l)) & 256u);\n"
            "                uchar s = signs[l];\n"
            "                for (uint j = 0; j < 4u; j++) {\n"
            "                    float sign1 = (uint(s) & (1u << j)) ? -1.0f : 1.0f;\n"
            "                    float sign2 = (uint(s) & (1u << (j + 4u))) ? -1.0f : 1.0f;\n"
            "                    sum += db2 * qw3_iq3s_grid_val(kgrid, idx1, j) * sign1 * xx[xo + j + 0u];\n"
            "                    sum += db2 * qw3_iq3s_grid_val(kgrid, idx2, j) * sign2 * xx[xo + j + 4u];\n"
            "                }\n"
            "                xo += 8u;\n"
            "            }\n"
            "            qh += 2;\n"
            "            qs += 8;\n"
            "            signs += 4;\n"
            "        }\n"
            "    }\n"
            "    sum = simd_sum(sum);\n"
            "    if (lane == 0) sh[simd_idx] = sum;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    sum = lane < 32 ? sh[lane] : 0.0f;\n"
            "    sum = simd_sum(sum);\n"
            "    if (tid == 0) out[row] = sum;\n"
            "}\n"
            "struct qw3_matvec_f32_args { uint n_in; uint n_out; };\n"
            "kernel void qw3_matvec_f32(constant qw3_matvec_f32_args &args,\n"
            "                           device const float *weights,\n"
            "                           device const float *x,\n"
            "                           device float *out,\n"
            "                           threadgroup float *sh,\n"
            "                           uint row [[threadgroup_position_in_grid]],\n"
            "                           ushort tid [[thread_index_in_threadgroup]],\n"
            "                           ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                           ushort lane [[thread_index_in_simdgroup]],\n"
            "                           ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (row >= args.n_out) return;\n"
            "    device const float *wr = weights + uint64_t(row) * args.n_in;\n"
            "    float sum = 0.0f;\n"
            "    for (uint i = tid; i < args.n_in; i += nt) sum += wr[i] * x[i];\n"
            "    sum = simd_sum(sum);\n"
            "    if (lane == 0) sh[simd_idx] = sum;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    sum = lane < 32 ? sh[lane] : 0.0f;\n"
            "    sum = simd_sum(sum);\n"
            "    if (tid == 0) out[row] = sum;\n"
            "}\n"
            "struct qw3_matvec_f32_pair_args { uint n_in; uint n_out; uint out_a_offset; uint out_b_offset; };\n"
            "kernel void qw3_matvec_f32_pair(constant qw3_matvec_f32_pair_args &args,\n"
            "                                device const float *weights_a,\n"
            "                                device const float *weights_b,\n"
            "                                device const float *x,\n"
            "                                device float *out,\n"
            "                                threadgroup float *sh,\n"
            "                                uint row [[threadgroup_position_in_grid]],\n"
            "                                ushort tid [[thread_index_in_threadgroup]],\n"
            "                                ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                ushort lane [[thread_index_in_simdgroup]],\n"
            "                                ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (row >= args.n_out) return;\n"
            "    device const float *wa = weights_a + uint64_t(row) * args.n_in;\n"
            "    device const float *wb = weights_b + uint64_t(row) * args.n_in;\n"
            "    float suma = 0.0f;\n"
            "    float sumb = 0.0f;\n"
            "    for (uint i = tid; i < args.n_in; i += nt) { float xv = x[i]; suma += wa[i] * xv; sumb += wb[i] * xv; }\n"
            "    suma = simd_sum(suma);\n"
            "    sumb = simd_sum(sumb);\n"
            "    if (lane == 0) { sh[simd_idx] = suma; sh[simd_idx + 32] = sumb; }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    suma = lane < 32 ? sh[lane] : 0.0f;\n"
            "    sumb = lane < 32 ? sh[lane + 32] : 0.0f;\n"
            "    suma = simd_sum(suma);\n"
            "    sumb = simd_sum(sumb);\n"
            "    if (tid == 0) { out[args.out_a_offset + row] = suma; out[args.out_b_offset + row] = sumb; }\n"
            "}\n"
            "kernel void qw3_matvec_f32_fast(constant qw3_matvec_f32_args &args,\n"
            "                               device const float *weights,\n"
            "                               device const float *x,\n"
            "                               device float *out,\n"
            "                               threadgroup float *sh,\n"
            "                               uint group [[threadgroup_position_in_grid]],\n"
            "                               ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                               ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    (void)sh;\n"
            "    const uint nr0 = 2u;\n"
            "    const uint nsg = 4u;\n"
            "    uint first_row = (group * nsg + uint(simd_idx)) * nr0;\n"
            "    float sum0 = 0.0f;\n"
            "    float sum1 = 0.0f;\n"
            "    for (uint i = uint(lane); i < args.n_in; i += 32u) {\n"
            "        float xv = x[i];\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) sum0 += weights[uint64_t(row) * args.n_in + i] * xv;\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) sum1 += weights[uint64_t(row) * args.n_in + i] * xv;\n"
            "    }\n"
            "    sum0 = simd_sum(sum0);\n"
            "    sum1 = simd_sum(sum1);\n"
            "    if (lane == 0) {\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) out[row] = sum0;\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) out[row] = sum1;\n"
            "    }\n"
            "}\n"
            "struct qw3_conv1d_args { uint n_channels; };\n"
            "kernel void qw3_deltanet_conv1d_zero(constant qw3_conv1d_args &args,\n"
            "                                     device const float *w,\n"
            "                                     device const float *qkv,\n"
            "                                     device float *out,\n"
            "                                     uint gid [[thread_position_in_grid]]) {\n"
            "    if (gid >= args.n_channels) return;\n"
            "    float x = qkv[gid] * w[gid * 4 + 3];\n"
            "    out[gid] = x / (1.0f + exp(-x));\n"
            "}\n"
            "kernel void qw3_deltanet_conv1d_step(constant qw3_conv1d_args &args,\n"
            "                                     device const float *w,\n"
            "                                     device const float *qkv,\n"
            "                                     device const float *state_in,\n"
            "                                     device float *out,\n"
            "                                     device float *state_out,\n"
            "                                     uint gid [[thread_position_in_grid]]) {\n"
            "    if (gid >= args.n_channels) return;\n"
            "    device const float *st = state_in + uint64_t(gid) * 3u;\n"
            "    float x = st[0] * w[gid * 4 + 0] + st[1] * w[gid * 4 + 1] +\n"
            "              st[2] * w[gid * 4 + 2] + qkv[gid] * w[gid * 4 + 3];\n"
            "    out[gid] = x / (1.0f + exp(-x));\n"
            "    device float *so = state_out + uint64_t(gid) * 3u;\n"
            "    so[0] = st[1];\n"
            "    so[1] = st[2];\n"
            "    so[2] = qkv[gid];\n"
            "}\n"
            "struct qw3_l2norm_args { uint head_dim; float eps; };\n"
            "kernel void qw3_l2norm_heads(constant qw3_l2norm_args &args,\n"
            "                             device const float *x,\n"
            "                             device float *out,\n"
            "                             threadgroup float *sh,\n"
            "                             uint head [[threadgroup_position_in_grid]],\n"
            "                             ushort tid [[thread_index_in_threadgroup]],\n"
            "                             ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                             ushort lane [[thread_index_in_simdgroup]],\n"
            "                             ushort nt [[threads_per_threadgroup]]) {\n"
            "    device const float *xh = x + uint64_t(head) * args.head_dim;\n"
            "    device float *yh = out + uint64_t(head) * args.head_dim;\n"
            "    float ss = 0.0f;\n"
            "    for (uint i = tid; i < args.head_dim; i += nt) ss += xh[i] * xh[i];\n"
            "    ss = simd_sum(ss);\n"
            "    if (lane == 0) sh[simd_idx] = ss;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    ss = lane < 32 ? sh[lane] : 0.0f;\n"
            "    ss = simd_sum(ss);\n"
            "    float scale = 1.0f / max(sqrt(ss), args.eps);\n"
            "    for (uint i = tid; i < args.head_dim; i += nt) yh[i] = xh[i] * scale;\n"
            "}\n"
            "struct qw3_gqa_norm_args { uint n_heads; uint head_dim; float eps; };\n"
            "kernel void qw3_gqa_q_norm_gate(constant qw3_gqa_norm_args &args,\n"
            "                                device const float *qg,\n"
            "                                device const float *w,\n"
            "                                device float *q_out,\n"
            "                                device float *gate_out,\n"
            "                                threadgroup float *sh,\n"
            "                                uint head [[threadgroup_position_in_grid]],\n"
            "                                ushort tid [[thread_index_in_threadgroup]],\n"
            "                                ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                ushort lane [[thread_index_in_simdgroup]],\n"
            "                                ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (head >= args.n_heads) return;\n"
            "    device const float *qh = qg + uint64_t(head) * uint64_t(args.head_dim) * 2ull;\n"
            "    device const float *gh = qh + args.head_dim;\n"
            "    device float *yo = q_out + uint64_t(head) * args.head_dim;\n"
            "    device float *go = gate_out + uint64_t(head) * args.head_dim;\n"
            "    float ss = 0.0f;\n"
            "    for (uint i = tid; i < args.head_dim; i += nt) ss += qh[i] * qh[i];\n"
            "    ss = simd_sum(ss);\n"
            "    if (lane == 0) sh[simd_idx] = ss;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    ss = lane < 32 ? sh[lane] : 0.0f;\n"
            "    ss = simd_sum(ss);\n"
            "    float scale = rsqrt(ss / float(args.head_dim) + args.eps);\n"
            "    for (uint i = tid; i < args.head_dim; i += nt) { yo[i] = qh[i] * scale * w[i]; go[i] = gh[i]; }\n"
            "}\n"
            "kernel void qw3_gqa_k_norm(constant qw3_gqa_norm_args &args,\n"
            "                           device const float *k,\n"
            "                           device const float *w,\n"
            "                           device float *k_out,\n"
            "                           threadgroup float *sh,\n"
            "                           uint head [[threadgroup_position_in_grid]],\n"
            "                           ushort tid [[thread_index_in_threadgroup]],\n"
            "                           ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                           ushort lane [[thread_index_in_simdgroup]],\n"
            "                           ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (head >= args.n_heads) return;\n"
            "    device const float *kh = k + uint64_t(head) * args.head_dim;\n"
            "    device float *yo = k_out + uint64_t(head) * args.head_dim;\n"
            "    float ss = 0.0f;\n"
            "    for (uint i = tid; i < args.head_dim; i += nt) ss += kh[i] * kh[i];\n"
            "    ss = simd_sum(ss);\n"
            "    if (lane == 0) sh[simd_idx] = ss;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    ss = lane < 32 ? sh[lane] : 0.0f;\n"
            "    ss = simd_sum(ss);\n"
            "    float scale = rsqrt(ss / float(args.head_dim) + args.eps);\n"
            "    for (uint i = tid; i < args.head_dim; i += nt) yo[i] = kh[i] * scale * w[i];\n"
            "}\n"
            "struct qw3_rope_args { uint n_heads; uint head_dim; uint rope_dim; int pos; float theta; };\n"
            "kernel void qw3_rope_heads(constant qw3_rope_args &args,\n"
            "                           device const float *x,\n"
            "                           device float *out,\n"
            "                           uint gid [[thread_position_in_grid]]) {\n"
            "    uint total = args.n_heads * args.head_dim;\n"
            "    if (gid >= total) return;\n"
            "    uint h = gid / args.head_dim;\n"
            "    uint i = gid - h * args.head_dim;\n"
            "    device const float *xh = x + uint64_t(h) * args.head_dim;\n"
            "    device float *yh = out + uint64_t(h) * args.head_dim;\n"
            "    if (i >= args.rope_dim) { yh[i] = xh[i]; return; }\n"
            "    uint p = i & ~1u;\n"
            "    float freq = pow(args.theta, -float(p) / float(args.rope_dim));\n"
            "    float ang = float(args.pos) * freq;\n"
            "    float c = cos(ang);\n"
            "    float s = sin(ang);\n"
            "    float x0 = xh[p + 0u];\n"
            "    float x1 = xh[p + 1u];\n"
            "    yh[i] = (i & 1u) ? (x0 * s + x1 * c) : (x0 * c - x1 * s);\n"
            "}\n"
            "struct qw3_gqa_inner_args { uint n_heads; uint n_kv_heads; uint head_dim; };\n"
            "kernel void qw3_gqa_single_token_inner(constant qw3_gqa_inner_args &args,\n"
            "                                       device const float *gate,\n"
            "                                       device const float *v,\n"
            "                                       device float *out,\n"
            "                                       uint gid [[thread_position_in_grid]]) {\n"
            "    uint total = args.n_heads * args.head_dim;\n"
            "    if (gid >= total) return;\n"
            "    uint h = gid / args.head_dim;\n"
            "    uint i = gid - h * args.head_dim;\n"
            "    uint kvh = h / (args.n_heads / args.n_kv_heads);\n"
            "    float g = gate[gid];\n"
            "    float sig = 1.0f / (1.0f + exp(-g));\n"
            "    out[gid] = v[uint64_t(kvh) * args.head_dim + i] * sig;\n"
            "}\n"
            "kernel void qw3_gqa_attend2_inner(constant qw3_gqa_inner_args &args,\n"
            "                                  device const float *q,\n"
            "                                  device const float *gate,\n"
            "                                  device const float *k_cache,\n"
            "                                  device const float *v_cache,\n"
            "                                  device float *out,\n"
            "                                  uint gid [[thread_position_in_grid]]) {\n"
            "    uint total = args.n_heads * args.head_dim;\n"
            "    if (gid >= total) return;\n"
            "    uint h = gid / args.head_dim;\n"
            "    uint i = gid - h * args.head_dim;\n"
            "    uint kvh = h / (args.n_heads / args.n_kv_heads);\n"
            "    device const float *qh = q + uint64_t(h) * args.head_dim;\n"
            "    device const float *k0 = k_cache + uint64_t(kvh) * args.head_dim;\n"
            "    device const float *k1 = k_cache + (uint64_t(args.n_kv_heads) + kvh) * args.head_dim;\n"
            "    float d0 = 0.0f;\n"
            "    float d1 = 0.0f;\n"
            "    for (uint j = 0; j < args.head_dim; j++) { d0 += qh[j] * k0[j]; d1 += qh[j] * k1[j]; }\n"
            "    float scale = rsqrt(float(args.head_dim));\n"
            "    d0 *= scale;\n"
            "    d1 *= scale;\n"
            "    float m = max(d0, d1);\n"
            "    float e0 = exp(d0 - m);\n"
            "    float e1 = exp(d1 - m);\n"
            "    float w0 = e0 / (e0 + e1);\n"
            "    float w1 = e1 / (e0 + e1);\n"
            "    float v0 = v_cache[uint64_t(kvh) * args.head_dim + i];\n"
            "    float v1 = v_cache[(uint64_t(args.n_kv_heads) + kvh) * args.head_dim + i];\n"
            "    float sig = 1.0f / (1.0f + exp(-gate[gid]));\n"
            "    out[gid] = (w0 * v0 + w1 * v1) * sig;\n"
            "}\n"
            "struct qw3_gqa_n_args { uint n_ctx; uint n_heads; uint n_kv_heads; uint head_dim; };\n"
            "kernel void qw3_gqa_attend_n_inner(constant qw3_gqa_n_args &args,\n"
            "                                  device const float *q,\n"
            "                                  device const float *gate,\n"
            "                                  device const float *k_cache,\n"
            "                                  device const float *v_cache,\n"
            "                                  device float *out,\n"
            "                                  threadgroup float *sh,\n"
            "                                  uint h [[threadgroup_position_in_grid]],\n"
            "                                  ushort tid [[thread_index_in_threadgroup]],\n"
            "                                  ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                  ushort lane [[thread_index_in_simdgroup]],\n"
            "                                  ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (h >= args.n_kv_heads || args.n_ctx == 0 || args.head_dim > uint(nt)) return;\n"
            "    uint i = uint(tid);\n"
            "    uint kvh = h;\n"
            "    uint group_heads = args.n_heads / args.n_kv_heads;\n"
            "    if (group_heads == 0u || group_heads > 8u) return;\n"
            "    uint first_qh = kvh * group_heads;\n"
            "    float scale = rsqrt(float(args.head_dim));\n"
            "    float qv[8];\n"
            "    float max_score[8];\n"
            "    float denom[8];\n"
            "    float acc[8];\n"
            "    for (uint gh = 0; gh < 8u; gh++) {\n"
            "        uint qh = first_qh + gh;\n"
            "        qv[gh] = (gh < group_heads && i < args.head_dim && qh < args.n_heads) ? q[uint64_t(qh) * args.head_dim + i] : 0.0f;\n"
            "        max_score[gh] = -FLT_MAX;\n"
            "        denom[gh] = 0.0f;\n"
            "        acc[gh] = 0.0f;\n"
            "    }\n"
            "    uint n_simd = (uint(nt) + 31u) >> 5u;\n"
            "    for (uint t = 0; t < args.n_ctx; t++) {\n"
            "        device const float *kh = k_cache + (uint64_t(t) * args.n_kv_heads + kvh) * args.head_dim;\n"
            "        float kval = (i < args.head_dim) ? kh[i] : 0.0f;\n"
            "        for (uint gh = 0; gh < 8u; gh++) {\n"
            "            float part = (gh < group_heads && i < args.head_dim) ? qv[gh] * kval : 0.0f;\n"
            "            part = simd_sum(part);\n"
            "            if (lane == 0) sh[gh * 8u + uint(simd_idx)] = part;\n"
            "        }\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        for (uint gh = 0; gh < 8u; gh++) {\n"
            "            float dot = (gh < group_heads && uint(tid) < n_simd) ? sh[gh * 8u + uint(tid)] : 0.0f;\n"
            "            dot = simd_sum(dot);\n"
            "            if (tid == 0) sh[gh] = dot;\n"
            "        }\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        float vv = (i < args.head_dim) ? v_cache[(uint64_t(t) * args.n_kv_heads + kvh) * args.head_dim + i] : 0.0f;\n"
            "        for (uint gh = 0; gh < 8u; gh++) {\n"
            "            if (gh < group_heads) {\n"
            "                float score = sh[gh] * scale;\n"
            "                float next_max = max(max_score[gh], score);\n"
            "                float prev_scale = exp(max_score[gh] - next_max);\n"
            "                float cur_scale = exp(score - next_max);\n"
            "                acc[gh] = acc[gh] * prev_scale + vv * cur_scale;\n"
            "                denom[gh] = denom[gh] * prev_scale + cur_scale;\n"
            "                max_score[gh] = next_max;\n"
            "            }\n"
            "        }\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    }\n"
            "    if (i < args.head_dim) {\n"
            "        for (uint gh = 0; gh < 8u; gh++) {\n"
            "            uint qh = first_qh + gh;\n"
            "            if (gh < group_heads && qh < args.n_heads) {\n"
            "                uint gid = qh * args.head_dim + i;\n"
            "                float sig = 1.0f / (1.0f + exp(-gate[gid]));\n"
            "                out[gid] = (acc[gh] / denom[gh]) * sig;\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}\n"
            "kernel void qw3_gqa_attend_n_q8_inner(constant qw3_gqa_n_args &args,\n"
            "                                     device const float *q,\n"
            "                                     device const float *gate,\n"
            "                                     device const uchar *k_cache,\n"
            "                                     device const uchar *v_cache,\n"
            "                                     device float *out,\n"
            "                                     threadgroup float *sh,\n"
            "                                     uint h [[threadgroup_position_in_grid]],\n"
            "                                     ushort tid [[thread_index_in_threadgroup]],\n"
            "                                     ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                     ushort lane [[thread_index_in_simdgroup]],\n"
            "                                     ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (h >= args.n_kv_heads || args.n_ctx == 0 || args.head_dim > uint(nt)) return;\n"
            "    uint i = uint(tid);\n"
            "    uint kvh = h;\n"
            "    uint group_heads = args.n_heads / args.n_kv_heads;\n"
            "    if (group_heads == 0u || group_heads > 8u) return;\n"
            "    uint first_qh = kvh * group_heads;\n"
            "    uint blocks_per_head = args.head_dim / 32u;\n"
            "    float scale = rsqrt(float(args.head_dim));\n"
            "    float qv[8];\n"
            "    float max_score[8];\n"
            "    float denom[8];\n"
            "    float acc[8];\n"
            "    for (uint gh = 0; gh < 8u; gh++) {\n"
            "        uint qh = first_qh + gh;\n"
            "        qv[gh] = (gh < group_heads && i < args.head_dim && qh < args.n_heads) ? q[uint64_t(qh) * args.head_dim + i] : 0.0f;\n"
            "        max_score[gh] = -FLT_MAX;\n"
            "        denom[gh] = 0.0f;\n"
            "        acc[gh] = 0.0f;\n"
            "    }\n"
            "    uint n_simd = (uint(nt) + 31u) >> 5u;\n"
            "    for (uint t = 0; t < args.n_ctx; t++) {\n"
            "        uint64_t b = (uint64_t(t) * args.n_kv_heads + kvh) * blocks_per_head + i / 32u;\n"
            "        device const uchar *kb = k_cache + b * 34ull;\n"
            "        device const uchar *vb = v_cache + b * 34ull;\n"
            "        half kd = *((device const half *)kb);\n"
            "        half vd = *((device const half *)vb);\n"
            "        char kq = *((device const char *)(kb + 2u + i % 32u));\n"
            "        char vq = *((device const char *)(vb + 2u + i % 32u));\n"
            "        float kval = (i < args.head_dim) ? float(kd) * float(kq) : 0.0f;\n"
            "        float vv = (i < args.head_dim) ? float(vd) * float(vq) : 0.0f;\n"
            "        for (uint gh = 0; gh < 8u; gh++) {\n"
            "            float part = (gh < group_heads && i < args.head_dim) ? qv[gh] * kval : 0.0f;\n"
            "            part = simd_sum(part);\n"
            "            if (lane == 0) sh[gh * 8u + uint(simd_idx)] = part;\n"
            "        }\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        for (uint gh = 0; gh < 8u; gh++) {\n"
            "            float dot = (gh < group_heads && uint(tid) < n_simd) ? sh[gh * 8u + uint(tid)] : 0.0f;\n"
            "            dot = simd_sum(dot);\n"
            "            if (tid == 0) sh[gh] = dot;\n"
            "        }\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        for (uint gh = 0; gh < 8u; gh++) {\n"
            "            if (gh < group_heads) {\n"
            "                float score = sh[gh] * scale;\n"
            "                float next_max = max(max_score[gh], score);\n"
            "                float prev_scale = exp(max_score[gh] - next_max);\n"
            "                float cur_scale = exp(score - next_max);\n"
            "                acc[gh] = acc[gh] * prev_scale + vv * cur_scale;\n"
            "                denom[gh] = denom[gh] * prev_scale + cur_scale;\n"
            "                max_score[gh] = next_max;\n"
            "            }\n"
            "        }\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    }\n"
            "    if (i < args.head_dim) {\n"
            "        for (uint gh = 0; gh < 8u; gh++) {\n"
            "            uint qh = first_qh + gh;\n"
            "            if (gh < group_heads && qh < args.n_heads) {\n"
            "                uint gid = qh * args.head_dim + i;\n"
            "                float sig = 1.0f / (1.0f + exp(-gate[gid]));\n"
            "                out[gid] = (acc[gh] / denom[gh]) * sig;\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}\n"
            "struct qw3_recur_zero_args { uint q_heads; uint v_heads; uint head_dim; };\n"
            "kernel void qw3_deltanet_recur_zero(constant qw3_recur_zero_args &args,\n"
            "                                     device const float *q,\n"
            "                                     device const float *k,\n"
            "                                     device const float *v,\n"
            "                                     device const float *beta,\n"
            "                                     device float *state_out,\n"
            "                                     device float *core_out,\n"
            "                                     threadgroup float *sh,\n"
            "                                     uint hv [[threadgroup_position_in_grid]],\n"
            "                                     ushort tid [[thread_index_in_threadgroup]],\n"
            "                                     ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                     ushort lane [[thread_index_in_simdgroup]],\n"
            "                                     ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (hv >= args.v_heads) return;\n"
            "    uint hk = hv % args.q_heads;\n"
            "    device const float *qh = q + uint64_t(hk) * args.head_dim;\n"
            "    device const float *kh = k + uint64_t(hk) * args.head_dim;\n"
            "    device const float *vh = v + uint64_t(hv) * args.head_dim;\n"
            "    float dot = 0.0f;\n"
            "    for (uint i = tid; i < args.head_dim; i += nt) dot += qh[i] * kh[i];\n"
            "    dot = simd_sum(dot);\n"
            "    if (lane == 0) sh[simd_idx] = dot;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    dot = lane < 32 ? sh[lane] : 0.0f;\n"
            "    dot = simd_sum(dot);\n"
            "    float b = beta[hv];\n"
            "    float scale = dot * rsqrt(float(args.head_dim)) * b;\n"
            "    for (uint j = tid; j < args.head_dim; j += nt) core_out[uint64_t(hv) * args.head_dim + j] = scale * vh[j];\n"
            "    uint state_n = args.head_dim * args.head_dim;\n"
            "    device float *shv = state_out + uint64_t(hv) * state_n;\n"
            "    for (uint idx = tid; idx < state_n; idx += nt) {\n"
            "        uint i = idx / args.head_dim;\n"
            "        uint j = idx - i * args.head_dim;\n"
            "        shv[idx] = kh[i] * b * vh[j];\n"
            "    }\n"
            "}\n"
            "kernel void qw3_deltanet_recur(constant qw3_recur_zero_args &args,\n"
            "                                device const float *state_in,\n"
            "                                device const float *q,\n"
            "                                device const float *k,\n"
            "                                device const float *v,\n"
            "                                device const float *beta,\n"
            "                                device const float *gamma,\n"
            "                                device float *state_out,\n"
            "                                device float *core_out,\n"
            "                                uint2 pos [[thread_position_in_grid]]) {\n"
            "    uint j = pos.x;\n"
            "    uint hv = pos.y;\n"
            "    if (hv >= args.v_heads || j >= args.head_dim) return;\n"
            "    uint hk = hv % args.q_heads;\n"
            "    uint state_n = args.head_dim * args.head_dim;\n"
            "    device const float *qh = q + uint64_t(hk) * args.head_dim;\n"
            "    device const float *kh = k + uint64_t(hk) * args.head_dim;\n"
            "    device const float *vh = v + uint64_t(hv) * args.head_dim;\n"
            "    device const float *sin = state_in + uint64_t(hv) * state_n;\n"
            "    device float *sout = state_out + uint64_t(hv) * state_n;\n"
            "    float g = gamma[hv];\n"
            "    float sk = 0.0f;\n"
            "    for (uint i = 0; i < args.head_dim; i++) sk += sin[i * args.head_dim + j] * g * kh[i];\n"
            "    float d = beta[hv] * (vh[j] - sk);\n"
            "    float out = 0.0f;\n"
            "    for (uint i = 0; i < args.head_dim; i++) {\n"
            "        uint idx = i * args.head_dim + j;\n"
            "        float sv = sin[idx] * g + kh[i] * d;\n"
            "        sout[idx] = sv;\n"
            "        out += sv * qh[i];\n"
            "    }\n"
            "    core_out[uint64_t(hv) * args.head_dim + j] = out * rsqrt(float(args.head_dim));\n"
            "}\n"
            "struct qw3_recur_scratch_args { uint q_heads; uint v_heads; uint head_dim; uint alpha_offset; uint beta_offset; };\n"
            "kernel void qw3_deltanet_recur_scratch_gates(constant qw3_recur_scratch_args &args,\n"
            "                                             device const float *state_in,\n"
            "                                             device const float *q,\n"
            "                                             device const float *k,\n"
            "                                             device const float *v,\n"
            "                                             device const float *scratch,\n"
            "                                             device const float *dt_bias,\n"
            "                                             device const float *a,\n"
            "                                             device float *state_out,\n"
            "                                             device float *core_out,\n"
            "                                             uint2 pos [[thread_position_in_grid]]) {\n"
            "    uint j = pos.x;\n"
            "    uint hv = pos.y;\n"
            "    if (hv >= args.v_heads || j >= args.head_dim) return;\n"
            "    uint hk = hv % args.q_heads;\n"
            "    uint state_n = args.head_dim * args.head_dim;\n"
            "    device const float *qh = q + uint64_t(hk) * args.head_dim;\n"
            "    device const float *kh = k + uint64_t(hk) * args.head_dim;\n"
            "    device const float *vh = v + uint64_t(hv) * args.head_dim;\n"
            "    device const float *sin = state_in + uint64_t(hv) * state_n;\n"
            "    device float *sout = state_out + uint64_t(hv) * state_n;\n"
            "    float beta_raw = scratch[args.beta_offset + hv];\n"
            "    float b = 1.0f / (1.0f + exp(-beta_raw));\n"
            "    float alpha_raw = scratch[args.alpha_offset + hv] + dt_bias[hv];\n"
            "    float sp = alpha_raw > 20.0f ? alpha_raw : (alpha_raw < -20.0f ? exp(alpha_raw) : log(1.0f + exp(alpha_raw)));\n"
            "    float g = exp(sp * a[hv]);\n"
            "    float sk = 0.0f;\n"
            "    for (uint i = 0; i < args.head_dim; i++) sk += sin[i * args.head_dim + j] * g * kh[i];\n"
            "    float d = b * (vh[j] - sk);\n"
            "    float out = 0.0f;\n"
            "    for (uint i = 0; i < args.head_dim; i++) {\n"
            "        uint idx = i * args.head_dim + j;\n"
            "        float sv = sin[idx] * g + kh[i] * d;\n"
            "        sout[idx] = sv;\n"
            "        out += sv * qh[i];\n"
            "    }\n"
            "    core_out[uint64_t(hv) * args.head_dim + j] = out * rsqrt(float(args.head_dim));\n"
            "}\n"
            "struct qw3_fused_gdn_args { uint q_heads; uint v_heads; uint head_dim; uint alpha_offset; uint beta_offset; uint z_offset; float eps; };\n"
            "kernel void qw3_deltanet_fused_gdn_scratch(constant qw3_fused_gdn_args &args,\n"
            "                                            device const float *state_in,\n"
            "                                            device const float *q,\n"
            "                                            device const float *k,\n"
            "                                            device const float *v,\n"
            "                                            device const float *scratch,\n"
            "                                            device const float *dt_bias,\n"
            "                                            device const float *a,\n"
            "                                            device const float *w,\n"
            "                                            device float *state_out,\n"
            "                                            device float *inner_out,\n"
            "                                            threadgroup float *sh,\n"
            "                                            uint hv [[threadgroup_position_in_grid]],\n"
            "                                            ushort j [[thread_index_in_threadgroup]],\n"
            "                                            ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                            ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    if (hv >= args.v_heads || j >= args.head_dim) return;\n"
            "    uint hk = hv % args.q_heads;\n"
            "    uint state_n = args.head_dim * args.head_dim;\n"
            "    device const float *qh = q + uint64_t(hk) * args.head_dim;\n"
            "    device const float *kh = k + uint64_t(hk) * args.head_dim;\n"
            "    device const float *vh = v + uint64_t(hv) * args.head_dim;\n"
            "    device const float *sin = state_in + uint64_t(hv) * state_n;\n"
            "    device float *sout = state_out + uint64_t(hv) * state_n;\n"
            "    if (j == 0) {\n"
            "        float beta_raw = scratch[args.beta_offset + hv];\n"
            "        sh[0] = 1.0f / (1.0f + exp(-beta_raw));\n"
            "        float alpha_raw = scratch[args.alpha_offset + hv] + dt_bias[hv];\n"
            "        float sp = alpha_raw > 20.0f ? alpha_raw : (alpha_raw < -20.0f ? exp(alpha_raw) : log(1.0f + exp(alpha_raw)));\n"
            "        sh[1] = exp(sp * a[hv]);\n"
            "    }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    float b = sh[0];\n"
            "    float g = sh[1];\n"
            "    float sk = 0.0f;\n"
            "    for (uint i = 0; i < args.head_dim; i++) sk += sin[i * args.head_dim + j] * g * kh[i];\n"
            "    float d = b * (vh[j] - sk);\n"
            "    float sum = 0.0f;\n"
            "    for (uint i = 0; i < args.head_dim; i++) {\n"
            "        uint idx = i * args.head_dim + j;\n"
            "        float sv = sin[idx] * g + kh[i] * d;\n"
            "        sout[idx] = sv;\n"
            "        sum += sv * qh[i];\n"
            "    }\n"
            "    float core = sum * rsqrt(float(args.head_dim));\n"
            "    float ss = simd_sum(core * core);\n"
            "    if (lane == 0) sh[simd_idx] = ss;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    ss = lane < 32 ? sh[lane] : 0.0f;\n"
            "    ss = simd_sum(ss);\n"
            "    float scale = rsqrt(ss / float(args.head_dim) + args.eps);\n"
            "    float zi = scratch[args.z_offset + uint64_t(hv) * args.head_dim + j];\n"
            "    float gate = zi / (1.0f + exp(-zi));\n"
            "    inner_out[uint64_t(hv) * args.head_dim + j] = core * scale * w[j] * gate;\n"
            "}\n"
            "struct qw3_gated_rmsnorm_args { uint v_heads; uint head_dim; float eps; };\n"
            "kernel void qw3_deltanet_gated_rmsnorm(constant qw3_gated_rmsnorm_args &args,\n"
            "                                      device const float *w,\n"
            "                                      device const float *core,\n"
            "                                      device const float *z,\n"
            "                                      device float *out,\n"
            "                                      threadgroup float *sh,\n"
            "                                      uint hv [[threadgroup_position_in_grid]],\n"
            "                                      ushort tid [[thread_index_in_threadgroup]],\n"
            "                                      ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                      ushort lane [[thread_index_in_simdgroup]],\n"
            "                                      ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (hv >= args.v_heads) return;\n"
            "    device const float *src = core + uint64_t(hv) * args.head_dim;\n"
            "    device const float *zg = z + uint64_t(hv) * args.head_dim;\n"
            "    device float *dst = out + uint64_t(hv) * args.head_dim;\n"
            "    float ss = 0.0f;\n"
            "    for (uint i = tid; i < args.head_dim; i += nt) ss += src[i] * src[i];\n"
            "    ss = simd_sum(ss);\n"
            "    if (lane == 0) sh[simd_idx] = ss;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    ss = lane < 32 ? sh[lane] : 0.0f;\n"
            "    ss = simd_sum(ss);\n"
            "    float scale = rsqrt(ss / float(args.head_dim) + args.eps);\n"
            "    for (uint i = tid; i < args.head_dim; i += nt) {\n"
            "        float zi = zg[i];\n"
            "        float gate = zi / (1.0f + exp(-zi));\n"
            "        dst[i] = src[i] * scale * w[i] * gate;\n"
            "    }\n"
            "}\n"
            "kernel void qw3_residual_rmsnorm_weight_f32(constant qw3_rmsnorm_args &args,\n"
            "                                           device const float *x,\n"
            "                                           device const float *residual,\n"
            "                                           device const float *w,\n"
            "                                           device float *y,\n"
            "                                           threadgroup float *sh,\n"
            "                                           ushort tid [[thread_index_in_threadgroup]],\n"
            "                                           ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                           ushort lane [[thread_index_in_simdgroup]],\n"
            "                                           ushort nt [[threads_per_threadgroup]]) {\n"
            "    float ss = 0.0f;\n"
            "    for (uint i = tid; i < args.n; i += nt) {\n"
            "        float v = x[i] + residual[i];\n"
            "        ss += v * v;\n"
            "    }\n"
            "    ss = simd_sum(ss);\n"
            "    if (lane == 0) sh[simd_idx] = ss;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    ss = lane < 32 ? sh[lane] : 0.0f;\n"
            "    ss = simd_sum(ss);\n"
            "    float scale = rsqrt(ss / float(args.n) + args.eps);\n"
            "    for (uint i = tid; i < args.n; i += nt) y[i] = (x[i] + residual[i]) * scale * w[i];\n"
            "}\n"
            "kernel void qw3_residual_rmsnorm_update_x0(constant qw3_rmsnorm_args &args,\n"
            "                                           device float *x0,\n"
            "                                           device const float *residual,\n"
            "                                           device const float *w,\n"
            "                                           device float *y,\n"
            "                                           threadgroup float *sh,\n"
            "                                           ushort tid [[thread_index_in_threadgroup]],\n"
            "                                           ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                           ushort lane [[thread_index_in_simdgroup]],\n"
            "                                           ushort nt [[threads_per_threadgroup]]) {\n"
            "    float ss = 0.0f;\n"
            "    for (uint i = tid; i < args.n; i += nt) {\n"
            "        float v = x0[i] + residual[i];\n"
            "        ss += v * v;\n"
            "    }\n"
            "    ss = simd_sum(ss);\n"
            "    if (lane == 0) sh[simd_idx] = ss;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    ss = lane < 32 ? sh[lane] : 0.0f;\n"
            "    ss = simd_sum(ss);\n"
            "    float scale = rsqrt(ss / float(args.n) + args.eps);\n"
            "    for (uint i = tid; i < args.n; i += nt) {\n"
            "        float v = x0[i] + residual[i];\n"
            "        x0[i] = v;\n"
            "        y[i] = v * scale * w[i];\n"
            "    }\n"
            "}\n"
            "struct qw3_unary_args { uint n; float scale; };\n"
            "kernel void qw3_silu_mul(constant qw3_unary_args &args,\n"
            "                         device const float *a,\n"
            "                         device const float *b,\n"
            "                         device float *out,\n"
            "                         uint gid [[thread_position_in_grid]]) {\n"
            "    if (gid >= args.n) return;\n"
            "    float x = a[gid];\n"
            "    out[gid] = (x / (1.0f + exp(-x))) * b[gid];\n"
            "}\n"
            "kernel void qw3_scale(constant qw3_unary_args &args,\n"
            "                       device const float *x,\n"
            "                       device float *out,\n"
            "                       uint gid [[thread_position_in_grid]]) {\n"
            "    if (gid >= args.n) return;\n"
            "    out[gid] = x[gid] * args.scale;\n"
            "}\n"
            "struct qw3_offset_args { uint n; uint a_offset; uint b_offset; };\n"
            "kernel void qw3_add_moe_to_x0(constant qw3_unary_args &args,\n"
            "                              device float *x0,\n"
            "                              device const float *x1,\n"
            "                              device const float *moe,\n"
            "                              uint gid [[thread_position_in_grid]]) {\n"
            "    if (gid >= args.n) return;\n"
            "    x0[gid] = x0[gid] + moe[gid];\n"
            "}\n"
            "kernel void qw3_silu_mul_offsets(constant qw3_offset_args &args,\n"
            "                                  device const float *scratch,\n"
            "                                  device float *out,\n"
            "                                  uint gid [[thread_position_in_grid]]) {\n"
            "    if (gid >= args.n) return;\n"
            "    float x = scratch[args.a_offset + gid];\n"
            "    float y = scratch[args.b_offset + gid];\n"
            "    out[gid] = (x / (1.0f + exp(-x))) * y;\n"
            "}\n"
            "kernel void qw3_scale_x1_scalar_add_x0(constant qw3_offset_args &args,\n"
            "                                      device float *x0,\n"
            "                                      device const float *x1,\n"
            "                                      device const float *scratch,\n"
            "                                      uint gid [[thread_position_in_grid]]) {\n"
            "    if (gid >= args.n) return;\n"
            "    float raw = scratch[args.a_offset];\n"
            "    float scale = 1.0f / (1.0f + exp(-raw));\n"
            "    x0[gid] = x0[gid] + x1[gid] * scale;\n"
            "}\n"
            "kernel void qw3_scale_x1_add_x0(constant qw3_unary_args &args,\n"
            "                                device float *x0,\n"
            "                                device const float *x1,\n"
            "                                uint gid [[thread_position_in_grid]]) {\n"
            "    if (gid >= args.n) return;\n"
            "    x0[gid] = x0[gid] + x1[gid] * args.scale;\n"
            "}\n"
            "kernel void qw3_scale_scratch_add_x0(constant qw3_offset_args &args,\n"
            "                                     constant float &scale,\n"
            "                                     device float *x0,\n"
            "                                     device const float *scratch,\n"
            "                                     uint gid [[thread_position_in_grid]]) {\n"
            "    if (gid >= args.n) return;\n"
            "    x0[gid] = x0[gid] + scratch[args.a_offset + gid] * scale;\n"
            "}\n"
            "kernel void qw3_scale_scratch_add_x0_slot(constant qw3_offset_args &args,\n"
            "                                          constant uint &slot,\n"
            "                                          device float *x0,\n"
            "                                          device const float *scratch,\n"
            "                                          device const float *weights,\n"
            "                                          uint gid [[thread_position_in_grid]]) {\n"
            "    if (gid >= args.n) return;\n"
            "    x0[gid] = x0[gid] + scratch[args.a_offset + gid] * weights[slot];\n"
            "}\n"
            "kernel void qw3_router_top8(device const float *router,\n"
            "                            device int *ids,\n"
            "                            device float *weights,\n"
            "                            uint tid [[thread_index_in_threadgroup]]) {\n"
            "    threadgroup float vals[256];\n"
            "    threadgroup int best[256];\n"
            "    threadgroup float selected[8];\n"
            "    threadgroup int selected_ids[8];\n"
            "    for (uint rank = 0; rank < 8u; rank++) {\n"
            "        float v = router[tid];\n"
            "        for (uint k = 0; k < rank; k++) if (selected_ids[k] == int(tid)) v = -INFINITY;\n"
            "        vals[tid] = v;\n"
            "        best[tid] = int(tid);\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        for (uint stride = 128u; stride > 0u; stride >>= 1u) {\n"
            "            if (tid < stride) {\n"
            "                float rv = vals[tid + stride];\n"
            "                int ri = best[tid + stride];\n"
            "                if (rv > vals[tid] || (rv == vals[tid] && ri < best[tid])) {\n"
            "                    vals[tid] = rv;\n"
            "                    best[tid] = ri;\n"
            "                }\n"
            "            }\n"
            "            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        }\n"
            "        if (tid == 0) { ids[rank] = best[0]; selected_ids[rank] = best[0]; selected[rank] = vals[0]; }\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    }\n"
            "    if (tid == 0) {\n"
            "        float sum = 0.0f;\n"
            "        for (uint k = 0; k < 8u; k++) { weights[k] = exp(selected[k] - selected[0]); sum += weights[k]; }\n"
            "        for (uint k = 0; k < 8u; k++) weights[k] /= sum;\n"
            "    }\n"
            "}\n"
            "struct qw3_expert_slot_args { uint n_in; uint n_out; uint row_bytes; uint expert_bytes; uint slot; };\n"
            "inline float qw3_iq3s_dot_row(device const uchar *wr,\n"
            "                              device const float *x,\n"
            "                              device const ushort *kgrid,\n"
            "                              uint n_in,\n"
            "                              ushort tid,\n"
            "                              ushort nt) {\n"
            "    float sum = 0.0f;\n"
            "    uint n_blocks = n_in / 256u;\n"
            "    for (uint b = tid; b < n_blocks; b += nt) {\n"
            "        device const uchar *blk = wr + uint64_t(b) * 110ull;\n"
            "        half d = *((device const half *)blk);\n"
            "        device const uchar *qs = blk + 2;\n"
            "        device const uchar *qh = qs + 64;\n"
            "        device const uchar *signs = qh + 8;\n"
            "        device const uchar *scales = signs + 32;\n"
            "        device const float *xx = x + uint64_t(b) * 256ull;\n"
            "        uint xo = 0;\n"
            "        for (uint ib32 = 0; ib32 < 8u; ib32 += 2u) {\n"
            "            float db1 = float(d) * float(1u + 2u * (uint(scales[ib32 / 2u]) & 15u));\n"
            "            float db2 = float(d) * float(1u + 2u * (uint(scales[ib32 / 2u]) >> 4u));\n"
            "            uchar qh0 = qh[0]; uchar qh1 = qh[1];\n"
            "            for (uint l = 0; l < 4u; l++) {\n"
            "                uint idx1 = uint(qs[2u * l + 0u]) | ((uint(qh0) << (8u - 2u * l)) & 256u);\n"
            "                uint idx2 = uint(qs[2u * l + 1u]) | ((uint(qh0) << (7u - 2u * l)) & 256u);\n"
            "                uchar s = signs[l];\n"
            "                for (uint j = 0; j < 4u; j++) {\n"
            "                    float sign1 = (uint(s) & (1u << j)) ? -1.0f : 1.0f;\n"
            "                    float sign2 = (uint(s) & (1u << (j + 4u))) ? -1.0f : 1.0f;\n"
            "                    sum += db1 * qw3_iq3s_grid_val(kgrid, idx1, j) * sign1 * xx[xo + j + 0u];\n"
            "                    sum += db1 * qw3_iq3s_grid_val(kgrid, idx2, j) * sign2 * xx[xo + j + 4u];\n"
            "                }\n"
            "                xo += 8u;\n"
            "            }\n"
            "            qs += 8; signs += 4;\n"
            "            for (uint l = 0; l < 4u; l++) {\n"
            "                uint idx1 = uint(qs[2u * l + 0u]) | ((uint(qh1) << (8u - 2u * l)) & 256u);\n"
            "                uint idx2 = uint(qs[2u * l + 1u]) | ((uint(qh1) << (7u - 2u * l)) & 256u);\n"
            "                uchar s = signs[l];\n"
            "                for (uint j = 0; j < 4u; j++) {\n"
            "                    float sign1 = (uint(s) & (1u << j)) ? -1.0f : 1.0f;\n"
            "                    float sign2 = (uint(s) & (1u << (j + 4u))) ? -1.0f : 1.0f;\n"
            "                    sum += db2 * qw3_iq3s_grid_val(kgrid, idx1, j) * sign1 * xx[xo + j + 0u];\n"
            "                    sum += db2 * qw3_iq3s_grid_val(kgrid, idx2, j) * sign2 * xx[xo + j + 4u];\n"
            "                }\n"
            "                xo += 8u;\n"
            "            }\n"
            "            qh += 2; qs += 8; signs += 4;\n"
            "        }\n"
            "    }\n"
            "    return sum;\n"
            "}\n"
            "inline float qw3_iq3s_dot32(device const uchar *blk,\n"
            "                           device const float *xx,\n"
            "                           device const ushort *kgrid,\n"
            "                           uint ib) {\n"
            "    half d = *((device const half *)blk);\n"
            "    device const uchar *qs = blk + 2;\n"
            "    device const uchar *qh = qs + 64;\n"
            "    device const uchar *signs = qh + 8;\n"
            "    device const uchar *scales = signs + 32;\n"
            "    float db = float(d) * float(1u + 2u * ((uint(scales[ib / 2u]) >> (4u * (ib & 1u))) & 15u));\n"
            "    uchar qhb = qh[ib];\n"
            "    qs += ib * 8u;\n"
            "    signs += ib * 4u;\n"
            "    float sum = 0.0f;\n"
            "    for (uint l = 0; l < 4u; l++) {\n"
            "        uint idx1 = uint(qs[2u * l + 0u]) | ((uint(qhb) << (8u - 2u * l)) & 256u);\n"
            "        uint idx2 = uint(qs[2u * l + 1u]) | ((uint(qhb) << (7u - 2u * l)) & 256u);\n"
            "        uchar s = signs[l];\n"
            "        for (uint j = 0; j < 4u; j++) {\n"
            "            float sign1 = (uint(s) & (1u << j)) ? -1.0f : 1.0f;\n"
            "            float sign2 = (uint(s) & (1u << (j + 4u))) ? -1.0f : 1.0f;\n"
            "            sum += db * qw3_iq3s_grid_val(kgrid, idx1, j) * sign1 * xx[8u * l + j + 0u];\n"
            "            sum += db * qw3_iq3s_grid_val(kgrid, idx2, j) * sign2 * xx[8u * l + j + 4u];\n"
            "        }\n"
            "    }\n"
            "    return sum;\n"
            "}\n"
            "inline float2 qw3_iq3s_dot32_pair(device const uchar *gate_blk,\n"
            "                                  device const uchar *up_blk,\n"
            "                                  device const float *xx,\n"
            "                                  device const ushort *kgrid,\n"
            "                                  uint ib) {\n"
            "    half gate_d = *((device const half *)gate_blk);\n"
            "    device const uchar *gate_qs = gate_blk + 2;\n"
            "    device const uchar *gate_qh = gate_qs + 64;\n"
            "    device const uchar *gate_signs = gate_qh + 8;\n"
            "    device const uchar *gate_scales = gate_signs + 32;\n"
            "    float gate_db = float(gate_d) * float(1u + 2u * ((uint(gate_scales[ib / 2u]) >> (4u * (ib & 1u))) & 15u));\n"
            "    uchar gate_qhb = gate_qh[ib];\n"
            "    gate_qs += ib * 8u;\n"
            "    gate_signs += ib * 4u;\n"
            "    half up_d = *((device const half *)up_blk);\n"
            "    device const uchar *up_qs = up_blk + 2;\n"
            "    device const uchar *up_qh = up_qs + 64;\n"
            "    device const uchar *up_signs = up_qh + 8;\n"
            "    device const uchar *up_scales = up_signs + 32;\n"
            "    float up_db = float(up_d) * float(1u + 2u * ((uint(up_scales[ib / 2u]) >> (4u * (ib & 1u))) & 15u));\n"
            "    uchar up_qhb = up_qh[ib];\n"
            "    up_qs += ib * 8u;\n"
            "    up_signs += ib * 4u;\n"
            "    float2 sum = float2(0.0f);\n"
            "    for (uint l = 0; l < 4u; l++) {\n"
            "        uint gate_idx1 = uint(gate_qs[2u * l + 0u]) | ((uint(gate_qhb) << (8u - 2u * l)) & 256u);\n"
            "        uint gate_idx2 = uint(gate_qs[2u * l + 1u]) | ((uint(gate_qhb) << (7u - 2u * l)) & 256u);\n"
            "        uint up_idx1 = uint(up_qs[2u * l + 0u]) | ((uint(up_qhb) << (8u - 2u * l)) & 256u);\n"
            "        uint up_idx2 = uint(up_qs[2u * l + 1u]) | ((uint(up_qhb) << (7u - 2u * l)) & 256u);\n"
            "        uchar gate_s = gate_signs[l];\n"
            "        uchar up_s = up_signs[l];\n"
            "        for (uint j = 0; j < 4u; j++) {\n"
            "            float x1 = xx[8u * l + j + 0u];\n"
            "            float x2 = xx[8u * l + j + 4u];\n"
            "            float gate_sign1 = (uint(gate_s) & (1u << j)) ? -1.0f : 1.0f;\n"
            "            float gate_sign2 = (uint(gate_s) & (1u << (j + 4u))) ? -1.0f : 1.0f;\n"
            "            float up_sign1 = (uint(up_s) & (1u << j)) ? -1.0f : 1.0f;\n"
            "            float up_sign2 = (uint(up_s) & (1u << (j + 4u))) ? -1.0f : 1.0f;\n"
            "            sum.x += gate_db * qw3_iq3s_grid_val(kgrid, gate_idx1, j) * gate_sign1 * x1;\n"
            "            sum.x += gate_db * qw3_iq3s_grid_val(kgrid, gate_idx2, j) * gate_sign2 * x2;\n"
            "            sum.y += up_db * qw3_iq3s_grid_val(kgrid, up_idx1, j) * up_sign1 * x1;\n"
            "            sum.y += up_db * qw3_iq3s_grid_val(kgrid, up_idx2, j) * up_sign2 * x2;\n"
            "        }\n"
            "    }\n"
            "    return sum;\n"
            "}\n"
            "kernel void qw3_matvec_iq3_s_pair(constant qw3_matvec_q8_0_args &args,\n"
            "                                  device const uchar *gate_weights,\n"
            "                                  device const uchar *up_weights,\n"
            "                                  device const float *x,\n"
            "                                  device float *out,\n"
            "                                  device const ushort *kgrid,\n"
            "                                  threadgroup float *sh,\n"
            "                                  uint row [[threadgroup_position_in_grid]],\n"
            "                                  ushort tid [[thread_index_in_threadgroup]],\n"
            "                                  ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                  ushort lane [[thread_index_in_simdgroup]],\n"
            "                                  ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (row >= args.n_out) return;\n"
            "    device const uchar *gate_wr = gate_weights + uint64_t(row) * uint64_t(args.row_bytes);\n"
            "    device const uchar *up_wr = up_weights + uint64_t(row) * uint64_t(args.row_bytes);\n"
            "    float gate_sum = qw3_iq3s_dot_row(gate_wr, x, kgrid, args.n_in, tid, nt);\n"
            "    float up_sum = qw3_iq3s_dot_row(up_wr, x, kgrid, args.n_in, tid, nt);\n"
            "    gate_sum = simd_sum(gate_sum);\n"
            "    up_sum = simd_sum(up_sum);\n"
            "    if (lane == 0) { sh[simd_idx] = gate_sum; sh[32u + simd_idx] = up_sum; }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    uint nsg = (uint(nt) + 31u) / 32u;\n"
            "    gate_sum = lane < nsg ? sh[lane] : 0.0f;\n"
            "    up_sum = lane < nsg ? sh[32u + lane] : 0.0f;\n"
            "    gate_sum = simd_sum(gate_sum);\n"
            "    up_sum = simd_sum(up_sum);\n"
            "    if (tid == 0) { out[row] = gate_sum; out[args.n_out + row] = up_sum; }\n"
            "}\n"
            "kernel void qw3_matvec_iq3_s_pair_fast(constant qw3_matvec_q8_0_args &args,\n"
            "                                       device const uchar *gate_weights,\n"
            "                                       device const uchar *up_weights,\n"
            "                                       device const float *x,\n"
            "                                       device float *out,\n"
            "                                       device const ushort *kgrid,\n"
            "                                       threadgroup float *sh,\n"
            "                                       uint group [[threadgroup_position_in_grid]],\n"
            "                                       ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                       ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    (void)sh;\n"
            "    const uint nr0 = 4u;\n"
            "    const uint nsg = 2u;\n"
            "    uint first_row = (group * nsg + uint(simd_idx)) * nr0;\n"
            "    float gate_sum0 = 0.0f, gate_sum1 = 0.0f, gate_sum2 = 0.0f, gate_sum3 = 0.0f;\n"
            "    float up_sum0 = 0.0f, up_sum1 = 0.0f, up_sum2 = 0.0f, up_sum3 = 0.0f;\n"
            "    uint nb32 = (args.n_in / 256u) * 8u;\n"
            "    for (uint ib32 = uint(lane); ib32 < nb32; ib32 += 32u) {\n"
            "        uint ibl = ib32 / 8u;\n"
            "        uint ib = ib32 - ibl * 8u;\n"
            "        device const float *xx = x + uint64_t(ib32) * 32ull;\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) {\n"
            "            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;\n"
            "            gate_sum0 += qw3_iq3s_dot32(gate_weights + off, xx, kgrid, ib);\n"
            "            up_sum0 += qw3_iq3s_dot32(up_weights + off, xx, kgrid, ib);\n"
            "        }\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) {\n"
            "            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;\n"
            "            gate_sum1 += qw3_iq3s_dot32(gate_weights + off, xx, kgrid, ib);\n"
            "            up_sum1 += qw3_iq3s_dot32(up_weights + off, xx, kgrid, ib);\n"
            "        }\n"
            "        row = first_row + 2u;\n"
            "        if (row < args.n_out) {\n"
            "            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;\n"
            "            gate_sum2 += qw3_iq3s_dot32(gate_weights + off, xx, kgrid, ib);\n"
            "            up_sum2 += qw3_iq3s_dot32(up_weights + off, xx, kgrid, ib);\n"
            "        }\n"
            "        row = first_row + 3u;\n"
            "        if (row < args.n_out) {\n"
            "            uint64_t off = uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;\n"
            "            gate_sum3 += qw3_iq3s_dot32(gate_weights + off, xx, kgrid, ib);\n"
            "            up_sum3 += qw3_iq3s_dot32(up_weights + off, xx, kgrid, ib);\n"
            "        }\n"
            "    }\n"
            "    gate_sum0 = simd_sum(gate_sum0); up_sum0 = simd_sum(up_sum0);\n"
            "    gate_sum1 = simd_sum(gate_sum1); up_sum1 = simd_sum(up_sum1);\n"
            "    gate_sum2 = simd_sum(gate_sum2); up_sum2 = simd_sum(up_sum2);\n"
            "    gate_sum3 = simd_sum(gate_sum3); up_sum3 = simd_sum(up_sum3);\n"
            "    if (lane == 0) {\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) { out[row] = gate_sum0; out[args.n_out + row] = up_sum0; }\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) { out[row] = gate_sum1; out[args.n_out + row] = up_sum1; }\n"
            "        row = first_row + 2u;\n"
            "        if (row < args.n_out) { out[row] = gate_sum2; out[args.n_out + row] = up_sum2; }\n"
            "        row = first_row + 3u;\n"
            "        if (row < args.n_out) { out[row] = gate_sum3; out[args.n_out + row] = up_sum3; }\n"
            "    }\n"
            "}\n"
            "kernel void qw3_matvec_iq3_s_expert_slot_pair(constant qw3_expert_slot_args &args,\n"
            "                                              device const uchar *gate_weights,\n"
            "                                              device const uchar *up_weights,\n"
            "                                              device const float *x,\n"
            "                                              device float *out,\n"
            "                                              device const ushort *kgrid,\n"
            "                                              device const int *ids,\n"
            "                                              threadgroup float *sh,\n"
            "                                              uint row [[threadgroup_position_in_grid]],\n"
            "                                              ushort tid [[thread_index_in_threadgroup]],\n"
            "                                              ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                              ushort lane [[thread_index_in_simdgroup]],\n"
            "                                              ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (row >= args.n_out) return;\n"
            "    uint expert = uint(ids[args.slot]);\n"
            "    uint64_t off = uint64_t(expert) * uint64_t(args.expert_bytes) + uint64_t(row) * uint64_t(args.row_bytes);\n"
            "    float gate_sum = qw3_iq3s_dot_row(gate_weights + off, x, kgrid, args.n_in, tid, nt);\n"
            "    float up_sum = qw3_iq3s_dot_row(up_weights + off, x, kgrid, args.n_in, tid, nt);\n"
            "    gate_sum = simd_sum(gate_sum);\n"
            "    up_sum = simd_sum(up_sum);\n"
            "    if (lane == 0) { sh[simd_idx] = gate_sum; sh[32u + simd_idx] = up_sum; }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    gate_sum = lane < 32 ? sh[lane] : 0.0f;\n"
            "    up_sum = lane < 32 ? sh[32u + lane] : 0.0f;\n"
            "    gate_sum = simd_sum(gate_sum);\n"
            "    up_sum = simd_sum(up_sum);\n"
            "    if (tid == 0) { out[row] = gate_sum; out[args.n_out + row] = up_sum; }\n"
            "}\n"
            "kernel void qw3_matvec_iq3_s_expert_slot_pair_fast(constant qw3_expert_slot_args &args,\n"
            "                                                   device const uchar *gate_weights,\n"
            "                                                   device const uchar *up_weights,\n"
            "                                                   device const float *x,\n"
            "                                                   device float *out,\n"
            "                                                   device const ushort *kgrid,\n"
            "                                                   device const int *ids,\n"
            "                                                   threadgroup float *sh,\n"
            "                                                   uint group [[threadgroup_position_in_grid]],\n"
            "                                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                                   ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    (void)sh;\n"
            "    const uint nr0 = 4u;\n"
            "    const uint nsg = 2u;\n"
            "    uint first_row = (group * nsg + uint(simd_idx)) * nr0;\n"
            "    uint expert = uint(ids[args.slot]);\n"
            "    uint64_t expert_off = uint64_t(expert) * uint64_t(args.expert_bytes);\n"
            "    float gate_sum0 = 0.0f, gate_sum1 = 0.0f, gate_sum2 = 0.0f, gate_sum3 = 0.0f;\n"
            "    float up_sum0 = 0.0f, up_sum1 = 0.0f, up_sum2 = 0.0f, up_sum3 = 0.0f;\n"
            "    uint nb32 = (args.n_in / 256u) * 8u;\n"
            "    for (uint ib32 = uint(lane); ib32 < nb32; ib32 += 32u) {\n"
            "        uint ibl = ib32 / 8u;\n"
            "        uint ib = ib32 - ibl * 8u;\n"
            "        device const float *xx = x + uint64_t(ib32) * 32ull;\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) {\n"
            "            uint64_t off = expert_off + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;\n"
            "            gate_sum0 += qw3_iq3s_dot32(gate_weights + off, xx, kgrid, ib);\n"
            "            up_sum0 += qw3_iq3s_dot32(up_weights + off, xx, kgrid, ib);\n"
            "        }\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) {\n"
            "            uint64_t off = expert_off + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;\n"
            "            gate_sum1 += qw3_iq3s_dot32(gate_weights + off, xx, kgrid, ib);\n"
            "            up_sum1 += qw3_iq3s_dot32(up_weights + off, xx, kgrid, ib);\n"
            "        }\n"
            "        row = first_row + 2u;\n"
            "        if (row < args.n_out) {\n"
            "            uint64_t off = expert_off + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;\n"
            "            gate_sum2 += qw3_iq3s_dot32(gate_weights + off, xx, kgrid, ib);\n"
            "            up_sum2 += qw3_iq3s_dot32(up_weights + off, xx, kgrid, ib);\n"
            "        }\n"
            "        row = first_row + 3u;\n"
            "        if (row < args.n_out) {\n"
            "            uint64_t off = expert_off + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(ibl) * 110ull;\n"
            "            gate_sum3 += qw3_iq3s_dot32(gate_weights + off, xx, kgrid, ib);\n"
            "            up_sum3 += qw3_iq3s_dot32(up_weights + off, xx, kgrid, ib);\n"
            "        }\n"
            "    }\n"
            "    gate_sum0 = simd_sum(gate_sum0); up_sum0 = simd_sum(up_sum0);\n"
            "    gate_sum1 = simd_sum(gate_sum1); up_sum1 = simd_sum(up_sum1);\n"
            "    gate_sum2 = simd_sum(gate_sum2); up_sum2 = simd_sum(up_sum2);\n"
            "    gate_sum3 = simd_sum(gate_sum3); up_sum3 = simd_sum(up_sum3);\n"
            "    if (lane == 0) {\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) { out[row] = gate_sum0; out[args.n_out + row] = up_sum0; }\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) { out[row] = gate_sum1; out[args.n_out + row] = up_sum1; }\n"
            "        row = first_row + 2u;\n"
            "        if (row < args.n_out) { out[row] = gate_sum2; out[args.n_out + row] = up_sum2; }\n"
            "        row = first_row + 3u;\n"
            "        if (row < args.n_out) { out[row] = gate_sum3; out[args.n_out + row] = up_sum3; }\n"
            "    }\n"
            "}\n"
            "struct qw3_moe_batch_args { uint n_in; uint n_ff; uint n_embd; uint n_active; uint iq3_row_bytes; uint iq3_expert_bytes; uint down_row_bytes; uint down_expert_bytes; uint gateup_base; uint hidden_base; uint down_base; };\n"
            "kernel void qw3_moe_iq3_s_pair_batch(constant qw3_moe_batch_args &args,\n"
            "                                      device const uchar *gate_weights,\n"
            "                                      device const uchar *up_weights,\n"
            "                                      device const float *x,\n"
            "                                      device float *scratch,\n"
            "                                      device const ushort *kgrid,\n"
            "                                      constant int *ids,\n"
            "                                      threadgroup float *sh,\n"
            "                                      uint group [[threadgroup_position_in_grid]],\n"
            "                                      ushort tid [[thread_index_in_threadgroup]],\n"
            "                                      ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                      ushort lane [[thread_index_in_simdgroup]],\n"
            "                                      ushort nt [[threads_per_threadgroup]]) {\n"
            "    uint slot = group / args.n_ff;\n"
            "    uint row = group - slot * args.n_ff;\n"
            "    if (row >= args.n_ff || slot >= args.n_active) return;\n"
            "    uint expert = uint(ids[slot]);\n"
            "    uint64_t off = uint64_t(expert) * uint64_t(args.iq3_expert_bytes) + uint64_t(row) * uint64_t(args.iq3_row_bytes);\n"
            "    float gate_sum = qw3_iq3s_dot_row(gate_weights + off, x, kgrid, args.n_in, tid, nt);\n"
            "    float up_sum = qw3_iq3s_dot_row(up_weights + off, x, kgrid, args.n_in, tid, nt);\n"
            "    gate_sum = simd_sum(gate_sum);\n"
            "    up_sum = simd_sum(up_sum);\n"
            "    if (lane == 0) { sh[simd_idx] = gate_sum; sh[32u + simd_idx] = up_sum; }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    gate_sum = lane < 32 ? sh[lane] : 0.0f;\n"
            "    up_sum = lane < 32 ? sh[32u + lane] : 0.0f;\n"
            "    gate_sum = simd_sum(gate_sum);\n"
            "    up_sum = simd_sum(up_sum);\n"
            "    if (tid == 0) {\n"
            "        uint base = args.gateup_base + slot * (2u * args.n_ff);\n"
            "        scratch[base + row] = gate_sum;\n"
            "        scratch[base + args.n_ff + row] = up_sum;\n"
            "    }\n"
            "}\n"
            "kernel void qw3_moe_down_iq4_xs_batch(constant qw3_moe_batch_args &args,\n"
            "                                      device const uchar *weights,\n"
            "                                      device const float *scratch,\n"
            "                                      device float *out,\n"
            "                                      constant int *ids,\n"
            "                                      constant float *router_weights,\n"
            "                                      threadgroup float *sh,\n"
            "                                      uint group [[threadgroup_position_in_grid]],\n"
            "                                      ushort tid [[thread_index_in_threadgroup]],\n"
            "                                      ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                      ushort lane [[thread_index_in_simdgroup]],\n"
            "                                      ushort nt [[threads_per_threadgroup]]) {\n"
            "    uint slot = group / args.n_embd;\n"
            "    uint row = group - slot * args.n_embd;\n"
            "    if (row >= args.n_embd || slot >= args.n_active) return;\n"
            "    uint expert = uint(ids[slot]);\n"
            "    device const uchar *wr = weights + uint64_t(expert) * uint64_t(args.down_expert_bytes) + uint64_t(row) * uint64_t(args.down_row_bytes);\n"
            "    device const float *x = scratch + args.hidden_base + slot * args.n_ff;\n"
            "    float sum = 0.0f;\n"
            "    uint n_blocks = args.n_ff / 256u;\n"
            "    for (uint b = tid; b < n_blocks; b += nt) {\n"
            "        device const uchar *blk = wr + uint64_t(b) * 136ull;\n"
            "        half d = *((device const half *)blk);\n"
            "        ushort scales_h = *((device const ushort *)(blk + 2));\n"
            "        device const uchar *scales_l = blk + 4;\n"
            "        device const uchar *qs = scales_l + 4;\n"
            "        device const float *xx = x + uint64_t(b) * 256ull;\n"
            "        for (uint ib = 0; ib < 8u; ib++) {\n"
            "            uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);\n"
            "            float dl = float(d) * float(int(ls) - 32);\n"
            "            device const uchar *q = qs + ib * 16u;\n"
            "            device const float *xg = xx + ib * 32u;\n"
            "            for (uint j = 0; j < 16u; j++) { uchar v = q[j]; sum += dl * qw3_iq4nl_val(uint(v) & 15u) * xg[j]; sum += dl * qw3_iq4nl_val(uint(v) >> 4u) * xg[j + 16u]; }\n"
            "        }\n"
            "    }\n"
            "    sum = simd_sum(sum); if (lane == 0) sh[simd_idx] = sum; threadgroup_barrier(mem_flags::mem_threadgroup); uint nsg = (uint(nt) + 31u) / 32u; sum = lane < nsg ? sh[lane] : 0.0f; sum = simd_sum(sum);\n"
            "    if (tid == 0) out[args.down_base + slot * args.n_embd + row] = sum * router_weights[slot];\n"
            "}\n"
            "kernel void qw3_moe_iq3_s_swiglu_batch_fast(constant qw3_moe_batch_args &args,\n"
            "                                           device const uchar *gate_weights,\n"
            "                                           device const uchar *up_weights,\n"
            "                                           device const float *x,\n"
            "                                           device float *scratch,\n"
            "                                           device const ushort *kgrid,\n"
            "                                           constant int *ids,\n"
            "                                           threadgroup float *sh,\n"
            "                                           uint group [[threadgroup_position_in_grid]],\n"
            "                                           ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                           ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    (void)sh;\n"
            "    const uint nr0 = 4u;\n"
            "    const uint nsg = 2u;\n"
            "    uint groups_per_slot = (args.n_ff + 7u) / 8u;\n"
            "    uint slot = group / groups_per_slot;\n"
            "    uint row_group = group - slot * groups_per_slot;\n"
            "    if (slot >= args.n_active) return;\n"
            "    uint first_row = (row_group * nsg + uint(simd_idx)) * nr0;\n"
            "    uint expert = uint(ids[slot]);\n"
            "    uint64_t expert_off = uint64_t(expert) * uint64_t(args.iq3_expert_bytes);\n"
            "    float gate_sum0 = 0.0f, gate_sum1 = 0.0f, gate_sum2 = 0.0f, gate_sum3 = 0.0f;\n"
            "    float up_sum0 = 0.0f, up_sum1 = 0.0f, up_sum2 = 0.0f, up_sum3 = 0.0f;\n"
            "    uint nb32 = (args.n_in / 256u) * 8u;\n"
            "    for (uint ib32 = uint(lane); ib32 < nb32; ib32 += 32u) {\n"
            "        uint ibl = ib32 / 8u;\n"
            "        uint ib = ib32 - ibl * 8u;\n"
            "        device const float *xx = x + uint64_t(ib32) * 32ull;\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_ff) {\n"
            "            uint64_t off = expert_off + uint64_t(row) * uint64_t(args.iq3_row_bytes) + uint64_t(ibl) * 110ull;\n"
            "            float2 pair_sum = qw3_iq3s_dot32_pair(gate_weights + off, up_weights + off, xx, kgrid, ib);\n"
            "            gate_sum0 += pair_sum.x; up_sum0 += pair_sum.y;\n"
            "        }\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_ff) {\n"
            "            uint64_t off = expert_off + uint64_t(row) * uint64_t(args.iq3_row_bytes) + uint64_t(ibl) * 110ull;\n"
            "            float2 pair_sum = qw3_iq3s_dot32_pair(gate_weights + off, up_weights + off, xx, kgrid, ib);\n"
            "            gate_sum1 += pair_sum.x; up_sum1 += pair_sum.y;\n"
            "        }\n"
            "        row = first_row + 2u;\n"
            "        if (row < args.n_ff) {\n"
            "            uint64_t off = expert_off + uint64_t(row) * uint64_t(args.iq3_row_bytes) + uint64_t(ibl) * 110ull;\n"
            "            float2 pair_sum = qw3_iq3s_dot32_pair(gate_weights + off, up_weights + off, xx, kgrid, ib);\n"
            "            gate_sum2 += pair_sum.x; up_sum2 += pair_sum.y;\n"
            "        }\n"
            "        row = first_row + 3u;\n"
            "        if (row < args.n_ff) {\n"
            "            uint64_t off = expert_off + uint64_t(row) * uint64_t(args.iq3_row_bytes) + uint64_t(ibl) * 110ull;\n"
            "            float2 pair_sum = qw3_iq3s_dot32_pair(gate_weights + off, up_weights + off, xx, kgrid, ib);\n"
            "            gate_sum3 += pair_sum.x; up_sum3 += pair_sum.y;\n"
            "        }\n"
            "    }\n"
            "    gate_sum0 = simd_sum(gate_sum0); up_sum0 = simd_sum(up_sum0);\n"
            "    gate_sum1 = simd_sum(gate_sum1); up_sum1 = simd_sum(up_sum1);\n"
            "    gate_sum2 = simd_sum(gate_sum2); up_sum2 = simd_sum(up_sum2);\n"
            "    gate_sum3 = simd_sum(gate_sum3); up_sum3 = simd_sum(up_sum3);\n"
            "    if (lane == 0) {\n"
            "        uint base = args.hidden_base + slot * args.n_ff;\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_ff) scratch[base + row] = (gate_sum0 / (1.0f + exp(-gate_sum0))) * up_sum0;\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_ff) scratch[base + row] = (gate_sum1 / (1.0f + exp(-gate_sum1))) * up_sum1;\n"
            "        row = first_row + 2u;\n"
            "        if (row < args.n_ff) scratch[base + row] = (gate_sum2 / (1.0f + exp(-gate_sum2))) * up_sum2;\n"
            "        row = first_row + 3u;\n"
            "        if (row < args.n_ff) scratch[base + row] = (gate_sum3 / (1.0f + exp(-gate_sum3))) * up_sum3;\n"
            "    }\n"
            "}\n"
            "kernel void qw3_moe_down_iq4_xs_batch_fast(constant qw3_moe_batch_args &args,\n"
            "                                          device const uchar *weights,\n"
            "                                          device const float *scratch,\n"
            "                                          device float *out,\n"
            "                                          constant int *ids,\n"
            "                                          constant float *router_weights,\n"
            "                                          threadgroup float *sh,\n"
            "                                          uint group [[threadgroup_position_in_grid]],\n"
            "                                          ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                          ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    (void)sh;\n"
            "    const uint nr0 = 2u;\n"
            "    const uint nsg = 2u;\n"
            "    uint groups_per_slot = (args.n_embd + 3u) / 4u;\n"
            "    uint slot = group / groups_per_slot;\n"
            "    uint row_group = group - slot * groups_per_slot;\n"
            "    if (slot >= args.n_active) return;\n"
            "    uint first_row = (row_group * nsg + uint(simd_idx)) * nr0;\n"
            "    uint expert = uint(ids[slot]);\n"
            "    uint64_t expert_off = uint64_t(expert) * uint64_t(args.down_expert_bytes);\n"
            "    device const float *x = scratch + args.hidden_base + slot * args.n_ff;\n"
            "    uint ix = uint(lane) >> 4u;\n"
            "    uint it = uint(lane) & 15u;\n"
            "    uint ib = it >> 1u;\n"
            "    uint il = it & 1u;\n"
            "    float sum0 = 0.0f;\n"
            "    float sum1 = 0.0f;\n"
            "    uint n_blocks = args.n_ff / 256u;\n"
            "    for (uint b = ix; b < n_blocks; b += 2u) {\n"
            "        device const float *xg = x + uint64_t(b) * 256ull + uint64_t(ib) * 32ull + uint64_t(il) * 8ull;\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_embd) {\n"
            "            device const uchar *blk = weights + expert_off + uint64_t(row) * uint64_t(args.down_row_bytes) + uint64_t(b) * 136ull;\n"
            "            half d = *((device const half *)blk);\n"
            "            ushort scales_h = *((device const ushort *)(blk + 2));\n"
            "            device const uchar *scales_l = blk + 4;\n"
            "            device const uchar *qs = scales_l + 4 + ib * 16u + il * 8u;\n"
            "            uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);\n"
            "            float dl = float(d) * float(int(ls) - 32);\n"
            "            float acc = 0.0f;\n"
            "            for (uint j = 0; j < 8u; j++) { uchar v = qs[j]; acc += qw3_iq4nl_val(uint(v) & 15u) * xg[j]; acc += qw3_iq4nl_val(uint(v) >> 4u) * xg[j + 16u]; }\n"
            "            sum0 += dl * acc;\n"
            "        }\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_embd) {\n"
            "            device const uchar *blk = weights + expert_off + uint64_t(row) * uint64_t(args.down_row_bytes) + uint64_t(b) * 136ull;\n"
            "            half d = *((device const half *)blk);\n"
            "            ushort scales_h = *((device const ushort *)(blk + 2));\n"
            "            device const uchar *scales_l = blk + 4;\n"
            "            device const uchar *qs = scales_l + 4 + ib * 16u + il * 8u;\n"
            "            uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);\n"
            "            float dl = float(d) * float(int(ls) - 32);\n"
            "            float acc = 0.0f;\n"
            "            for (uint j = 0; j < 8u; j++) { uchar v = qs[j]; acc += qw3_iq4nl_val(uint(v) & 15u) * xg[j]; acc += qw3_iq4nl_val(uint(v) >> 4u) * xg[j + 16u]; }\n"
            "            sum1 += dl * acc;\n"
            "        }\n"
            "    }\n"
            "    sum0 = simd_sum(sum0);\n"
            "    sum1 = simd_sum(sum1);\n"
            "    float scale = router_weights[slot];\n"
            "    if (lane == 0) {\n"
            "        uint base = args.down_base + slot * args.n_embd;\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_embd) out[base + row] = sum0 * scale;\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_embd) out[base + row] = sum1 * scale;\n"
            "    }\n"
            "}\n"
            "kernel void qw3_moe_down_iq4_xs_pair_fast(constant qw3_moe_batch_args &args,\n"
            "                                         device const uchar *weights,\n"
            "                                         device const float *scratch,\n"
            "                                         device float *out,\n"
            "                                         constant int *ids,\n"
            "                                         constant float *router_weights,\n"
            "                                         threadgroup float *sh,\n"
            "                                         uint group [[threadgroup_position_in_grid]],\n"
            "                                         ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                         ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    (void)sh;\n"
            "    const uint nr0 = 2u;\n"
            "    const uint nsg = 2u;\n"
            "    uint groups_per_pair = (args.n_embd + 3u) / 4u;\n"
            "    uint pair = group / groups_per_pair;\n"
            "    uint row_group = group - pair * groups_per_pair;\n"
            "    uint slot0 = pair * 2u;\n"
            "    uint slot1 = slot0 + 1u;\n"
            "    if (slot0 >= args.n_active) return;\n"
            "    uint first_row = (row_group * nsg + uint(simd_idx)) * nr0;\n"
            "    uint expert0 = uint(ids[slot0]);\n"
            "    uint expert1 = slot1 < args.n_active ? uint(ids[slot1]) : expert0;\n"
            "    uint64_t expert_off0 = uint64_t(expert0) * uint64_t(args.down_expert_bytes);\n"
            "    uint64_t expert_off1 = uint64_t(expert1) * uint64_t(args.down_expert_bytes);\n"
            "    device const float *x0 = scratch + args.hidden_base + slot0 * args.n_ff;\n"
            "    device const float *x1 = scratch + args.hidden_base + (slot1 < args.n_active ? slot1 : slot0) * args.n_ff;\n"
            "    uint ix = uint(lane) >> 4u;\n"
            "    uint it = uint(lane) & 15u;\n"
            "    uint ib = it >> 1u;\n"
            "    uint il = it & 1u;\n"
            "    float sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;\n"
            "    uint n_blocks = args.n_ff / 256u;\n"
            "    for (uint b = ix; b < n_blocks; b += 2u) {\n"
            "        device const float *xg0 = x0 + uint64_t(b) * 256ull + uint64_t(ib) * 32ull + uint64_t(il) * 8ull;\n"
            "        device const float *xg1 = x1 + uint64_t(b) * 256ull + uint64_t(ib) * 32ull + uint64_t(il) * 8ull;\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_embd) {\n"
            "            device const uchar *ba = weights + expert_off0 + uint64_t(row) * uint64_t(args.down_row_bytes) + uint64_t(b) * 136ull;\n"
            "            device const uchar *bb = weights + expert_off1 + uint64_t(row) * uint64_t(args.down_row_bytes) + uint64_t(b) * 136ull;\n"
            "            half da = *((device const half *)ba); half db = *((device const half *)bb);\n"
            "            ushort sha = *((device const ushort *)(ba + 2)); ushort shb = *((device const ushort *)(bb + 2));\n"
            "            device const uchar *sla = ba + 4; device const uchar *slb = bb + 4;\n"
            "            device const uchar *qsa = sla + 4 + ib * 16u + il * 8u; device const uchar *qsb = slb + 4 + ib * 16u + il * 8u;\n"
            "            uint lsa = ((uint(sla[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(sha) >> (2u * ib)) & 3u) << 4u);\n"
            "            uint lsb = ((uint(slb[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(shb) >> (2u * ib)) & 3u) << 4u);\n"
            "            float aca = 0.0f, acb = 0.0f;\n"
            "            for (uint j = 0; j < 8u; j++) { uchar va = qsa[j]; uchar vb = qsb[j]; aca += qw3_iq4nl_val(uint(va) & 15u) * xg0[j] + qw3_iq4nl_val(uint(va) >> 4u) * xg0[j + 16u]; acb += qw3_iq4nl_val(uint(vb) & 15u) * xg1[j] + qw3_iq4nl_val(uint(vb) >> 4u) * xg1[j + 16u]; }\n"
            "            sum00 += float(da) * float(int(lsa) - 32) * aca; sum10 += float(db) * float(int(lsb) - 32) * acb;\n"
            "        }\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_embd) {\n"
            "            device const uchar *ba = weights + expert_off0 + uint64_t(row) * uint64_t(args.down_row_bytes) + uint64_t(b) * 136ull;\n"
            "            device const uchar *bb = weights + expert_off1 + uint64_t(row) * uint64_t(args.down_row_bytes) + uint64_t(b) * 136ull;\n"
            "            half da = *((device const half *)ba); half db = *((device const half *)bb);\n"
            "            ushort sha = *((device const ushort *)(ba + 2)); ushort shb = *((device const ushort *)(bb + 2));\n"
            "            device const uchar *sla = ba + 4; device const uchar *slb = bb + 4;\n"
            "            device const uchar *qsa = sla + 4 + ib * 16u + il * 8u; device const uchar *qsb = slb + 4 + ib * 16u + il * 8u;\n"
            "            uint lsa = ((uint(sla[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(sha) >> (2u * ib)) & 3u) << 4u);\n"
            "            uint lsb = ((uint(slb[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(shb) >> (2u * ib)) & 3u) << 4u);\n"
            "            float aca = 0.0f, acb = 0.0f;\n"
            "            for (uint j = 0; j < 8u; j++) { uchar va = qsa[j]; uchar vb = qsb[j]; aca += qw3_iq4nl_val(uint(va) & 15u) * xg0[j] + qw3_iq4nl_val(uint(va) >> 4u) * xg0[j + 16u]; acb += qw3_iq4nl_val(uint(vb) & 15u) * xg1[j] + qw3_iq4nl_val(uint(vb) >> 4u) * xg1[j + 16u]; }\n"
            "            sum01 += float(da) * float(int(lsa) - 32) * aca; sum11 += float(db) * float(int(lsb) - 32) * acb;\n"
            "        }\n"
            "    }\n"
            "    sum00 = simd_sum(sum00); sum01 = simd_sum(sum01);\n"
            "    sum10 = simd_sum(sum10); sum11 = simd_sum(sum11);\n"
            "    if (lane == 0) {\n"
            "        float scale0 = router_weights[slot0];\n"
            "        float scale1 = slot1 < args.n_active ? router_weights[slot1] : 0.0f;\n"
            "        uint base = args.down_base + pair * args.n_embd;\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_embd) out[base + row] = sum00 * scale0 + sum10 * scale1;\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_embd) out[base + row] = sum01 * scale0 + sum11 * scale1;\n"
            "    }\n"
            "}\n"
            "kernel void qw3_moe_down_iq4_xs_batch_reduce_fast(constant qw3_moe_batch_args &args,\n"
            "                                                 device const uchar *weights,\n"
            "                                                 device const float *scratch,\n"
            "                                                 device float *x0,\n"
            "                                                 constant int *ids,\n"
            "                                                 constant float *router_weights,\n"
            "                                                 threadgroup float *sh,\n"
            "                                                 uint group [[threadgroup_position_in_grid]],\n"
            "                                                 ushort tid [[thread_index_in_threadgroup]],\n"
            "                                                 ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                                 ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    uint slot = uint(simd_idx);\n"
            "    uint row0 = group * 2u;\n"
            "    uint row1 = row0 + 1u;\n"
            "    bool active = slot < args.n_active;\n"
            "    uint expert = active ? uint(ids[slot]) : 0u;\n"
            "    uint64_t expert_off = uint64_t(expert) * uint64_t(args.down_expert_bytes);\n"
            "    device const float *x = scratch + args.hidden_base + slot * args.n_ff;\n"
            "    uint ix = uint(lane) >> 4u;\n"
            "    uint it = uint(lane) & 15u;\n"
            "    uint ib = it >> 1u;\n"
            "    uint il = it & 1u;\n"
            "    float sum0 = 0.0f;\n"
            "    float sum1 = 0.0f;\n"
            "    uint n_blocks = args.n_ff / 256u;\n"
            "    if (active) {\n"
            "        for (uint b = ix; b < n_blocks; b += 2u) {\n"
            "            device const float *xg = x + uint64_t(b) * 256ull + uint64_t(ib) * 32ull + uint64_t(il) * 8ull;\n"
            "            if (row0 < args.n_embd) {\n"
            "                device const uchar *blk = weights + expert_off + uint64_t(row0) * uint64_t(args.down_row_bytes) + uint64_t(b) * 136ull;\n"
            "                half d = *((device const half *)blk);\n"
            "                ushort scales_h = *((device const ushort *)(blk + 2));\n"
            "                device const uchar *scales_l = blk + 4;\n"
            "                device const uchar *qs = scales_l + 4 + ib * 16u + il * 8u;\n"
            "                uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);\n"
            "                float dl = float(d) * float(int(ls) - 32);\n"
            "                float acc = 0.0f;\n"
            "                for (uint j = 0; j < 8u; j++) { uchar v = qs[j]; acc += qw3_iq4nl_val(uint(v) & 15u) * xg[j]; acc += qw3_iq4nl_val(uint(v) >> 4u) * xg[j + 16u]; }\n"
            "                sum0 += dl * acc;\n"
            "            }\n"
            "            if (row1 < args.n_embd) {\n"
            "                device const uchar *blk = weights + expert_off + uint64_t(row1) * uint64_t(args.down_row_bytes) + uint64_t(b) * 136ull;\n"
            "                half d = *((device const half *)blk);\n"
            "                ushort scales_h = *((device const ushort *)(blk + 2));\n"
            "                device const uchar *scales_l = blk + 4;\n"
            "                device const uchar *qs = scales_l + 4 + ib * 16u + il * 8u;\n"
            "                uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);\n"
            "                float dl = float(d) * float(int(ls) - 32);\n"
            "                float acc = 0.0f;\n"
            "                for (uint j = 0; j < 8u; j++) { uchar v = qs[j]; acc += qw3_iq4nl_val(uint(v) & 15u) * xg[j]; acc += qw3_iq4nl_val(uint(v) >> 4u) * xg[j + 16u]; }\n"
            "                sum1 += dl * acc;\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    sum0 = simd_sum(sum0);\n"
            "    sum1 = simd_sum(sum1);\n"
            "    float scale = active ? router_weights[slot] : 0.0f;\n"
            "    if (lane == 0) { sh[slot] = sum0 * scale; sh[8u + slot] = sum1 * scale; }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    if (tid == 0) {\n"
            "        float total0 = 0.0f;\n"
            "        float total1 = 0.0f;\n"
            "        for (uint s = 0; s < args.n_active; s++) { total0 += sh[s]; total1 += sh[8u + s]; }\n"
            "        if (row0 < args.n_embd) x0[row0] += total0;\n"
            "        if (row1 < args.n_embd) x0[row1] += total1;\n"
            "    }\n"
            "}\n"
            "kernel void qw3_moe_down_q6_k_batch(constant qw3_moe_batch_args &args,\n"
            "                                    device const uchar *weights,\n"
            "                                    device const float *scratch,\n"
            "                                    device float *out,\n"
            "                                    constant int *ids,\n"
            "                                    constant float *router_weights,\n"
            "                                    threadgroup float *sh,\n"
            "                                    uint group [[threadgroup_position_in_grid]],\n"
            "                                    ushort tid [[thread_index_in_threadgroup]],\n"
            "                                    ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                    ushort lane [[thread_index_in_simdgroup]],\n"
            "                                    ushort nt [[threads_per_threadgroup]]) {\n"
            "    uint slot = group / args.n_embd;\n"
            "    uint row = group - slot * args.n_embd;\n"
            "    if (row >= args.n_embd || slot >= args.n_active) return;\n"
            "    uint expert = uint(ids[slot]);\n"
            "    device const uchar *wr = weights + uint64_t(expert) * uint64_t(args.down_expert_bytes) + uint64_t(row) * uint64_t(args.down_row_bytes);\n"
            "    device const float *x = scratch + args.hidden_base + slot * args.n_ff;\n"
            "    float sum = 0.0f;\n"
            "    uint n_blocks = args.n_ff / 256u;\n"
            "    for (uint b = tid; b < n_blocks; b += nt) {\n"
            "        device const uchar *blk = wr + uint64_t(b) * 210ull;\n"
            "        device const uchar *ql = blk; device const uchar *qh = ql + 128u; device const uchar *scb = qh + 64u; device const char *sc = (device const char *)scb;\n"
            "        ushort dbits = ushort(scb[16u]) | (ushort(scb[17u]) << 8u); float d = qw3_f16_to_f32(dbits); device const float *xx = x + uint64_t(b) * 256ull;\n"
            "        for (uint n = 0; n < 256u; n += 128u) {\n"
            "            for (uint l = 0; l < 32u; l++) {\n"
            "                uint is = l / 16u;\n"
            "                int q1 = int((uint(ql[l +  0u]) & 15u) | (((uint(qh[l]) >> 0u) & 3u) << 4u)) - 32;\n"
            "                int q2 = int((uint(ql[l + 32u]) & 15u) | (((uint(qh[l]) >> 2u) & 3u) << 4u)) - 32;\n"
            "                int q3 = int((uint(ql[l +  0u]) >> 4u) | (((uint(qh[l]) >> 4u) & 3u) << 4u)) - 32;\n"
            "                int q4 = int((uint(ql[l + 32u]) >> 4u) | (((uint(qh[l]) >> 6u) & 3u) << 4u)) - 32;\n"
            "                sum += d * float(sc[is + 0u]) * float(q1) * xx[n + l +  0u]; sum += d * float(sc[is + 2u]) * float(q2) * xx[n + l + 32u]; sum += d * float(sc[is + 4u]) * float(q3) * xx[n + l + 64u]; sum += d * float(sc[is + 6u]) * float(q4) * xx[n + l + 96u];\n"
            "            }\n"
            "            ql += 64u; qh += 32u; sc += 8u;\n"
            "        }\n"
            "    }\n"
            "    sum = simd_sum(sum); if (lane == 0) sh[simd_idx] = sum; threadgroup_barrier(mem_flags::mem_threadgroup); uint nsg = (uint(nt) + 31u) / 32u; sum = lane < nsg ? sh[lane] : 0.0f; sum = simd_sum(sum);\n"
            "    if (tid == 0) out[args.down_base + slot * args.n_embd + row] = sum * router_weights[slot];\n"
            "}\n"
            "kernel void qw3_moe_down_q6_k_batch_fast(constant qw3_moe_batch_args &args,\n"
            "                                        device const uchar *weights,\n"
            "                                        device const float *scratch,\n"
            "                                        device float *out,\n"
            "                                        constant int *ids,\n"
            "                                        constant float *router_weights,\n"
            "                                        threadgroup float *sh,\n"
            "                                        uint group [[threadgroup_position_in_grid]],\n"
            "                                        ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                        ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    (void)sh;\n"
            "    const uint nr0 = 2u;\n"
            "    const uint nsg = 2u;\n"
            "    uint groups_per_slot = (args.n_embd + 3u) / 4u;\n"
            "    uint slot = group / groups_per_slot;\n"
            "    uint row_group = group - slot * groups_per_slot;\n"
            "    if (slot >= args.n_active) return;\n"
            "    uint first_row = (row_group * nsg + uint(simd_idx)) * nr0;\n"
            "    uint expert = uint(ids[slot]);\n"
            "    uint64_t expert_off = uint64_t(expert) * uint64_t(args.down_expert_bytes);\n"
            "    device const float *x = scratch + args.hidden_base + slot * args.n_ff;\n"
            "    uint tid = uint(lane) >> 1u;\n"
            "    uint ix = uint(lane) & 1u;\n"
            "    uint ip = tid >> 3u;\n"
            "    uint il = tid & 7u;\n"
            "    uint l0 = 4u * il;\n"
            "    uint is = 8u * ip + l0 / 16u;\n"
            "    uint y_offset = 128u * ip + l0;\n"
            "    uint q_offset_l = 64u * ip + l0;\n"
            "    uint q_offset_h = 32u * ip + l0;\n"
            "    float sum0 = 0.0f;\n"
            "    float sum1 = 0.0f;\n"
            "    uint n_blocks = args.n_ff / 256u;\n"
            "    for (uint b = ix; b < n_blocks; b += 2u) {\n"
            "        device const float *yy = x + uint64_t(b) * 256ull + y_offset;\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_embd) {\n"
            "            device const uchar *blk = weights + expert_off + uint64_t(row) * uint64_t(args.down_row_bytes) + uint64_t(b) * 210ull;\n"
            "            device const uchar *q1 = blk + q_offset_l;\n"
            "            device const uchar *q2 = q1 + 32u;\n"
            "            device const uchar *qh = blk + 128u + q_offset_h;\n"
            "            device const char *sc = (device const char *)(blk + 192u + is);\n"
            "            ushort dbits = ushort(blk[208u]) | (ushort(blk[209u]) << 8u);\n"
            "            float d = qw3_f16_to_f32(dbits);\n"
            "            float acc = 0.0f;\n"
            "            for (uint l = 0; l < 4u; l++) {\n"
            "                int qv1 = int((uint(q1[l]) & 15u) | (((uint(qh[l]) >> 0u) & 3u) << 4u)) - 32;\n"
            "                int qv2 = int((uint(q2[l]) & 15u) | (((uint(qh[l]) >> 2u) & 3u) << 4u)) - 32;\n"
            "                int qv3 = int((uint(q1[l]) >> 4u) | (((uint(qh[l]) >> 4u) & 3u) << 4u)) - 32;\n"
            "                int qv4 = int((uint(q2[l]) >> 4u) | (((uint(qh[l]) >> 6u) & 3u) << 4u)) - 32;\n"
            "                acc += float(sc[0]) * float(qv1) * yy[l + 0u];\n"
            "                acc += float(sc[2]) * float(qv2) * yy[l + 32u];\n"
            "                acc += float(sc[4]) * float(qv3) * yy[l + 64u];\n"
            "                acc += float(sc[6]) * float(qv4) * yy[l + 96u];\n"
            "            }\n"
            "            sum0 += d * acc;\n"
            "        }\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_embd) {\n"
            "            device const uchar *blk = weights + expert_off + uint64_t(row) * uint64_t(args.down_row_bytes) + uint64_t(b) * 210ull;\n"
            "            device const uchar *q1 = blk + q_offset_l;\n"
            "            device const uchar *q2 = q1 + 32u;\n"
            "            device const uchar *qh = blk + 128u + q_offset_h;\n"
            "            device const char *sc = (device const char *)(blk + 192u + is);\n"
            "            ushort dbits = ushort(blk[208u]) | (ushort(blk[209u]) << 8u);\n"
            "            float d = qw3_f16_to_f32(dbits);\n"
            "            float acc = 0.0f;\n"
            "            for (uint l = 0; l < 4u; l++) {\n"
            "                int qv1 = int((uint(q1[l]) & 15u) | (((uint(qh[l]) >> 0u) & 3u) << 4u)) - 32;\n"
            "                int qv2 = int((uint(q2[l]) & 15u) | (((uint(qh[l]) >> 2u) & 3u) << 4u)) - 32;\n"
            "                int qv3 = int((uint(q1[l]) >> 4u) | (((uint(qh[l]) >> 4u) & 3u) << 4u)) - 32;\n"
            "                int qv4 = int((uint(q2[l]) >> 4u) | (((uint(qh[l]) >> 6u) & 3u) << 4u)) - 32;\n"
            "                acc += float(sc[0]) * float(qv1) * yy[l + 0u];\n"
            "                acc += float(sc[2]) * float(qv2) * yy[l + 32u];\n"
            "                acc += float(sc[4]) * float(qv3) * yy[l + 64u];\n"
            "                acc += float(sc[6]) * float(qv4) * yy[l + 96u];\n"
            "            }\n"
            "            sum1 += d * acc;\n"
            "        }\n"
            "    }\n"
            "    sum0 = simd_sum(sum0);\n"
            "    sum1 = simd_sum(sum1);\n"
            "    float scale = router_weights[slot];\n"
            "    if (lane == 0) {\n"
            "        uint base = args.down_base + slot * args.n_embd;\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_embd) out[base + row] = sum0 * scale;\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_embd) out[base + row] = sum1 * scale;\n"
            "    }\n"
            "}\n"
            "kernel void qw3_moe_reduce_batch(constant qw3_moe_batch_args &args,\n"
            "                                  device const float *scratch,\n"
            "                                  device float *x0,\n"
            "                                  uint gid [[thread_position_in_grid]]) {\n"
            "    uint n4 = args.n_embd / 4u;\n"
            "    if (gid >= n4) return;\n"
            "    float4 sum = float4(0.0f);\n"
            "    for (uint slot = 0; slot < args.n_active; slot++) {\n"
            "        device const float4 *src4 = (device const float4 *)(scratch + args.down_base + slot * args.n_embd);\n"
            "        sum += src4[gid];\n"
            "    }\n"
            "    device float4 *x04 = (device float4 *)x0;\n"
            "    x04[gid] += sum;\n"
            "}\n"
            "kernel void qw3_matvec_iq3_s_expert_slot(constant qw3_expert_slot_args &args,\n"
            "                                         device const uchar *weights,\n"
            "                                         device const float *x,\n"
            "                                         device float *out,\n"
            "                                         device const ushort *kgrid,\n"
            "                                         device const int *ids,\n"
            "                                         threadgroup float *sh,\n"
            "                                         uint row [[threadgroup_position_in_grid]],\n"
            "                                         ushort tid [[thread_index_in_threadgroup]],\n"
            "                                         ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                         ushort lane [[thread_index_in_simdgroup]],\n"
            "                                         ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (row >= args.n_out) return;\n"
            "    uint expert = uint(ids[args.slot]);\n"
            "    device const uchar *wr = weights + uint64_t(expert) * uint64_t(args.expert_bytes) + uint64_t(row) * uint64_t(args.row_bytes);\n"
            "    float sum = 0.0f;\n"
            "    uint n_blocks = args.n_in / 256u;\n"
            "    for (uint b = tid; b < n_blocks; b += nt) {\n"
            "        device const uchar *blk = wr + uint64_t(b) * 110ull;\n"
            "        half d = *((device const half *)blk);\n"
            "        device const uchar *qs = blk + 2;\n"
            "        device const uchar *qh = qs + 64;\n"
            "        device const uchar *signs = qh + 8;\n"
            "        device const uchar *scales = signs + 32;\n"
            "        device const float *xx = x + uint64_t(b) * 256ull;\n"
            "        uint xo = 0;\n"
            "        for (uint ib32 = 0; ib32 < 8u; ib32 += 2u) {\n"
            "            float db1 = float(d) * float(1u + 2u * (uint(scales[ib32 / 2u]) & 15u));\n"
            "            float db2 = float(d) * float(1u + 2u * (uint(scales[ib32 / 2u]) >> 4u));\n"
            "            uchar qh0 = qh[0]; uchar qh1 = qh[1];\n"
            "            for (uint l = 0; l < 4u; l++) {\n"
            "                uint idx1 = uint(qs[2u * l + 0u]) | ((uint(qh0) << (8u - 2u * l)) & 256u);\n"
            "                uint idx2 = uint(qs[2u * l + 1u]) | ((uint(qh0) << (7u - 2u * l)) & 256u);\n"
            "                uchar s = signs[l];\n"
            "                for (uint j = 0; j < 4u; j++) {\n"
            "                    float sign1 = (uint(s) & (1u << j)) ? -1.0f : 1.0f;\n"
            "                    float sign2 = (uint(s) & (1u << (j + 4u))) ? -1.0f : 1.0f;\n"
            "                    sum += db1 * qw3_iq3s_grid_val(kgrid, idx1, j) * sign1 * xx[xo + j + 0u];\n"
            "                    sum += db1 * qw3_iq3s_grid_val(kgrid, idx2, j) * sign2 * xx[xo + j + 4u];\n"
            "                }\n"
            "                xo += 8u;\n"
            "            }\n"
            "            qs += 8; signs += 4;\n"
            "            for (uint l = 0; l < 4u; l++) {\n"
            "                uint idx1 = uint(qs[2u * l + 0u]) | ((uint(qh1) << (8u - 2u * l)) & 256u);\n"
            "                uint idx2 = uint(qs[2u * l + 1u]) | ((uint(qh1) << (7u - 2u * l)) & 256u);\n"
            "                uchar s = signs[l];\n"
            "                for (uint j = 0; j < 4u; j++) {\n"
            "                    float sign1 = (uint(s) & (1u << j)) ? -1.0f : 1.0f;\n"
            "                    float sign2 = (uint(s) & (1u << (j + 4u))) ? -1.0f : 1.0f;\n"
            "                    sum += db2 * qw3_iq3s_grid_val(kgrid, idx1, j) * sign1 * xx[xo + j + 0u];\n"
            "                    sum += db2 * qw3_iq3s_grid_val(kgrid, idx2, j) * sign2 * xx[xo + j + 4u];\n"
            "                }\n"
            "                xo += 8u;\n"
            "            }\n"
            "            qh += 2; qs += 8; signs += 4;\n"
            "        }\n"
            "    }\n"
            "    sum = simd_sum(sum);\n"
            "    if (lane == 0) sh[simd_idx] = sum;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    sum = lane < 32 ? sh[lane] : 0.0f;\n"
            "    sum = simd_sum(sum);\n"
            "    if (tid == 0) out[row] = sum;\n"
            "}\n"
            "kernel void qw3_matvec_iq4_xs_expert_slot(constant qw3_expert_slot_args &args,\n"
            "                                           device const uchar *weights,\n"
            "                                           device const float *x,\n"
            "                                           device float *out,\n"
            "                                           device const int *ids,\n"
            "                                           threadgroup float *sh,\n"
            "                                           uint row [[threadgroup_position_in_grid]],\n"
            "                                           ushort tid [[thread_index_in_threadgroup]],\n"
            "                                           ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                           ushort lane [[thread_index_in_simdgroup]],\n"
            "                                           ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (row >= args.n_out) return;\n"
            "    uint expert = uint(ids[args.slot]);\n"
            "    device const uchar *wr = weights + uint64_t(expert) * uint64_t(args.expert_bytes) + uint64_t(row) * uint64_t(args.row_bytes);\n"
            "    float sum = 0.0f;\n"
            "    uint n_blocks = args.n_in / 256u;\n"
            "    for (uint b = tid; b < n_blocks; b += nt) {\n"
            "        device const uchar *blk = wr + uint64_t(b) * 136ull;\n"
            "        half d = *((device const half *)blk);\n"
            "        ushort scales_h = *((device const ushort *)(blk + 2));\n"
            "        device const uchar *scales_l = blk + 4;\n"
            "        device const uchar *qs = scales_l + 4;\n"
            "        device const float *xx = x + uint64_t(b) * 256ull;\n"
            "        for (uint ib = 0; ib < 8u; ib++) {\n"
            "            uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);\n"
            "            float dl = float(d) * float(int(ls) - 32);\n"
            "            device const uchar *q = qs + ib * 16u;\n"
            "            device const float *xg = xx + ib * 32u;\n"
            "            for (uint j = 0; j < 16u; j++) { uchar v = q[j]; sum += dl * qw3_iq4nl_val(uint(v) & 15u) * xg[j]; sum += dl * qw3_iq4nl_val(uint(v) >> 4u) * xg[j + 16u]; }\n"
            "        }\n"
            "    }\n"
            "    sum = simd_sum(sum); if (lane == 0) sh[simd_idx] = sum; threadgroup_barrier(mem_flags::mem_threadgroup); sum = lane < 32 ? sh[lane] : 0.0f; sum = simd_sum(sum); if (tid == 0) out[row] = sum;\n"
            "}\n"
            "kernel void qw3_matvec_iq4_xs_expert_slot_fast(constant qw3_expert_slot_args &args,\n"
            "                                                device const uchar *weights,\n"
            "                                                device const float *x,\n"
            "                                                device float *out,\n"
            "                                                device const int *ids,\n"
            "                                                threadgroup float *sh,\n"
            "                                                uint group [[threadgroup_position_in_grid]],\n"
            "                                                ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                                ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    (void)sh;\n"
            "    const uint nr0 = 2u;\n"
            "    const uint nsg = 2u;\n"
            "    uint first_row = (group * nsg + uint(simd_idx)) * nr0;\n"
            "    uint expert = uint(ids[args.slot]);\n"
            "    uint64_t expert_off = uint64_t(expert) * uint64_t(args.expert_bytes);\n"
            "    uint ix = uint(lane) >> 4u;\n"
            "    uint it = uint(lane) & 15u;\n"
            "    uint ib = it >> 1u;\n"
            "    uint il = it & 1u;\n"
            "    float sum0 = 0.0f;\n"
            "    float sum1 = 0.0f;\n"
            "    uint n_blocks = args.n_in / 256u;\n"
            "    for (uint b = ix; b < n_blocks; b += 2u) {\n"
            "        device const float *xg = x + uint64_t(b) * 256ull + uint64_t(ib) * 32ull + uint64_t(il) * 8ull;\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) {\n"
            "            device const uchar *blk = weights + expert_off + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;\n"
            "            half d = *((device const half *)blk);\n"
            "            ushort scales_h = *((device const ushort *)(blk + 2));\n"
            "            device const uchar *scales_l = blk + 4;\n"
            "            device const uchar *qs = scales_l + 4 + ib * 16u + il * 8u;\n"
            "            uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);\n"
            "            float dl = float(d) * float(int(ls) - 32);\n"
            "            float acc = 0.0f;\n"
            "            for (uint j = 0; j < 8u; j++) { uchar v = qs[j]; acc += qw3_iq4nl_val(uint(v) & 15u) * xg[j]; acc += qw3_iq4nl_val(uint(v) >> 4u) * xg[j + 16u]; }\n"
            "            sum0 += dl * acc;\n"
            "        }\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) {\n"
            "            device const uchar *blk = weights + expert_off + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;\n"
            "            half d = *((device const half *)blk);\n"
            "            ushort scales_h = *((device const ushort *)(blk + 2));\n"
            "            device const uchar *scales_l = blk + 4;\n"
            "            device const uchar *qs = scales_l + 4 + ib * 16u + il * 8u;\n"
            "            uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);\n"
            "            float dl = float(d) * float(int(ls) - 32);\n"
            "            float acc = 0.0f;\n"
            "            for (uint j = 0; j < 8u; j++) { uchar v = qs[j]; acc += qw3_iq4nl_val(uint(v) & 15u) * xg[j]; acc += qw3_iq4nl_val(uint(v) >> 4u) * xg[j + 16u]; }\n"
            "            sum1 += dl * acc;\n"
            "        }\n"
            "    }\n"
            "    sum0 = simd_sum(sum0);\n"
            "    sum1 = simd_sum(sum1);\n"
            "    if (lane == 0) {\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) out[row] = sum0;\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) out[row] = sum1;\n"
            "    }\n"
            "}\n"
            "kernel void qw3_matvec_iq4_xs_expert_slot_add_x0_fast(constant qw3_expert_slot_args &args,\n"
            "                                                       device const uchar *weights,\n"
            "                                                       device const float *x,\n"
            "                                                       device float *x0,\n"
            "                                                       device const int *ids,\n"
            "                                                       device const float *router_weights,\n"
            "                                                       threadgroup float *sh,\n"
            "                                                       uint group [[threadgroup_position_in_grid]],\n"
            "                                                       ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                                       ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    (void)sh;\n"
            "    const uint nr0 = 2u;\n"
            "    const uint nsg = 2u;\n"
            "    uint first_row = (group * nsg + uint(simd_idx)) * nr0;\n"
            "    uint expert = uint(ids[args.slot]);\n"
            "    uint64_t expert_off = uint64_t(expert) * uint64_t(args.expert_bytes);\n"
            "    uint ix = uint(lane) >> 4u;\n"
            "    uint it = uint(lane) & 15u;\n"
            "    uint ib = it >> 1u;\n"
            "    uint il = it & 1u;\n"
            "    float sum0 = 0.0f;\n"
            "    float sum1 = 0.0f;\n"
            "    uint n_blocks = args.n_in / 256u;\n"
            "    for (uint b = ix; b < n_blocks; b += 2u) {\n"
            "        device const float *xg = x + uint64_t(b) * 256ull + uint64_t(ib) * 32ull + uint64_t(il) * 8ull;\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) {\n"
            "            device const uchar *blk = weights + expert_off + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;\n"
            "            half d = *((device const half *)blk);\n"
            "            ushort scales_h = *((device const ushort *)(blk + 2));\n"
            "            device const uchar *scales_l = blk + 4;\n"
            "            device const uchar *qs = scales_l + 4 + ib * 16u + il * 8u;\n"
            "            uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);\n"
            "            float dl = float(d) * float(int(ls) - 32);\n"
            "            float acc = 0.0f;\n"
            "            for (uint j = 0; j < 8u; j++) { uchar v = qs[j]; acc += qw3_iq4nl_val(uint(v) & 15u) * xg[j]; acc += qw3_iq4nl_val(uint(v) >> 4u) * xg[j + 16u]; }\n"
            "            sum0 += dl * acc;\n"
            "        }\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) {\n"
            "            device const uchar *blk = weights + expert_off + uint64_t(row) * uint64_t(args.row_bytes) + uint64_t(b) * 136ull;\n"
            "            half d = *((device const half *)blk);\n"
            "            ushort scales_h = *((device const ushort *)(blk + 2));\n"
            "            device const uchar *scales_l = blk + 4;\n"
            "            device const uchar *qs = scales_l + 4 + ib * 16u + il * 8u;\n"
            "            uint ls = ((uint(scales_l[ib / 2u]) >> (4u * (ib & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);\n"
            "            float dl = float(d) * float(int(ls) - 32);\n"
            "            float acc = 0.0f;\n"
            "            for (uint j = 0; j < 8u; j++) { uchar v = qs[j]; acc += qw3_iq4nl_val(uint(v) & 15u) * xg[j]; acc += qw3_iq4nl_val(uint(v) >> 4u) * xg[j + 16u]; }\n"
            "            sum1 += dl * acc;\n"
            "        }\n"
            "    }\n"
            "    sum0 = simd_sum(sum0);\n"
            "    sum1 = simd_sum(sum1);\n"
            "    float scale = router_weights[args.slot];\n"
            "    if (lane == 0) {\n"
            "        uint row = first_row;\n"
            "        if (row < args.n_out) x0[row] = x0[row] + sum0 * scale;\n"
            "        row = first_row + 1u;\n"
            "        if (row < args.n_out) x0[row] = x0[row] + sum1 * scale;\n"
            "    }\n"
            "}\n"
            "kernel void qw3_matvec_q6_k_expert_slot(constant qw3_expert_slot_args &args,\n"
            "                                         device const uchar *weights,\n"
            "                                         device const float *x,\n"
            "                                         device float *out,\n"
            "                                         device const int *ids,\n"
            "                                         threadgroup float *sh,\n"
            "                                         uint row [[threadgroup_position_in_grid]],\n"
            "                                         ushort tid [[thread_index_in_threadgroup]],\n"
            "                                         ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                         ushort lane [[thread_index_in_simdgroup]],\n"
            "                                         ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (row >= args.n_out) return;\n"
            "    uint expert = uint(ids[args.slot]);\n"
            "    device const uchar *wr = weights + uint64_t(expert) * uint64_t(args.expert_bytes) + uint64_t(row) * uint64_t(args.row_bytes);\n"
            "    float sum = 0.0f;\n"
            "    uint n_blocks = args.n_in / 256u;\n"
            "    for (uint b = tid; b < n_blocks; b += nt) {\n"
            "        device const uchar *blk = wr + uint64_t(b) * 210ull;\n"
            "        device const uchar *ql = blk; device const uchar *qh = ql + 128u; device const uchar *scb = qh + 64u; device const char *sc = (device const char *)scb;\n"
            "        ushort dbits = ushort(scb[16u]) | (ushort(scb[17u]) << 8u); float d = qw3_f16_to_f32(dbits); device const float *xx = x + uint64_t(b) * 256ull;\n"
            "        for (uint n = 0; n < 256u; n += 128u) {\n"
            "            for (uint l = 0; l < 32u; l++) {\n"
            "                uint is = l / 16u;\n"
            "                int q1 = int((uint(ql[l +  0u]) & 15u) | (((uint(qh[l]) >> 0u) & 3u) << 4u)) - 32;\n"
            "                int q2 = int((uint(ql[l + 32u]) & 15u) | (((uint(qh[l]) >> 2u) & 3u) << 4u)) - 32;\n"
            "                int q3 = int((uint(ql[l +  0u]) >> 4u) | (((uint(qh[l]) >> 4u) & 3u) << 4u)) - 32;\n"
            "                int q4 = int((uint(ql[l + 32u]) >> 4u) | (((uint(qh[l]) >> 6u) & 3u) << 4u)) - 32;\n"
            "                sum += d * float(sc[is + 0u]) * float(q1) * xx[n + l +  0u]; sum += d * float(sc[is + 2u]) * float(q2) * xx[n + l + 32u]; sum += d * float(sc[is + 4u]) * float(q3) * xx[n + l + 64u]; sum += d * float(sc[is + 6u]) * float(q4) * xx[n + l + 96u];\n"
            "            }\n"
            "            ql += 64u; qh += 32u; sc += 8u;\n"
            "        }\n"
            "    }\n"
            "    sum = simd_sum(sum); if (lane == 0) sh[simd_idx] = sum; threadgroup_barrier(mem_flags::mem_threadgroup); sum = lane < 32 ? sh[lane] : 0.0f; sum = simd_sum(sum); if (tid == 0) out[row] = sum;\n"
            "}\n"
            "struct qw3_argmax_args { uint n; };\n"
            "kernel void qw3_argmax_blocks(constant qw3_argmax_args &args,\n"
            "                              device const float *x,\n"
            "                              device float *out_vals,\n"
            "                              device uint *out_idxs,\n"
            "                              threadgroup float *sh_vals,\n"
            "                              threadgroup uint *sh_idxs,\n"
            "                              uint block [[threadgroup_position_in_grid]],\n"
            "                              ushort tid [[thread_index_in_threadgroup]],\n"
            "                              ushort nt [[threads_per_threadgroup]]) {\n"
            "    uint idx = block * uint(nt) + uint(tid);\n"
            "    float v = idx < args.n ? x[idx] : -FLT_MAX;\n"
            "    sh_vals[tid] = v;\n"
            "    sh_idxs[tid] = idx;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    for (uint stride = uint(nt) >> 1; stride > 0; stride >>= 1) {\n"
            "        if (uint(tid) < stride) {\n"
            "            float ov = sh_vals[tid + stride];\n"
            "            uint oi = sh_idxs[tid + stride];\n"
            "            float cv = sh_vals[tid];\n"
            "            uint ci = sh_idxs[tid];\n"
            "            if (ov > cv || (ov == cv && oi < ci)) {\n"
            "                sh_vals[tid] = ov;\n"
            "                sh_idxs[tid] = oi;\n"
            "            }\n"
            "        }\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    }\n"
            "    if (tid == 0) {\n"
            "        out_vals[block] = sh_vals[0];\n"
            "        out_idxs[block] = sh_idxs[0];\n"
            "    }\n"
            "}\n";
}

static int qw3_metal_compile_kernels(void) {
    if (g_library && g_rmsnorm_plain_pipeline &&
        g_rmsnorm_weight_f32_pipeline && g_embed_q8_0_pipeline &&
        g_matvec_q8_0_pipeline && g_matvec_q8_0_pair_pipeline &&
        g_matvec_q8_0_pair_silu_pipeline &&
        g_shared_gate_up_silu_pipeline &&
        g_matvec_q8_0_inner_scale_add_x0_pipeline &&
        g_matvec_iq4_xs_pipeline &&
        g_matvec_q6_k_pipeline && g_matvec_iq4_xs_add_x0_pipeline &&
        g_matvec_q6_k_add_x0_pipeline &&
        g_matvec_iq4_xs_swiglu_add_x0_pipeline &&
        g_matvec_q6_k_swiglu_add_x0_pipeline &&
        g_matvec_iq3_s_pipeline && g_matvec_iq3_s_pair_pipeline &&
        g_moe_iq3_s_pair_batch_pipeline &&
        g_moe_down_iq4_xs_batch_pipeline &&
        g_moe_down_iq4_xs_pair_pipeline &&
        g_moe_down_iq4_xs_batch_reduce_pipeline &&
        g_moe_down_q6_k_batch_pipeline &&
        g_moe_reduce_batch_pipeline &&
        g_matvec_f32_pipeline && g_matvec_f32_pair_pipeline &&
        g_matvec_f32_fast_pipeline &&
        g_deltanet_conv1d_zero_pipeline &&
        g_deltanet_conv1d_step_pipeline &&
        g_l2norm_heads_pipeline &&
        g_gqa_q_norm_gate_pipeline && g_gqa_k_norm_pipeline &&
        g_rope_heads_pipeline &&
        g_gqa_single_token_inner_pipeline &&
        g_gqa_attend2_inner_pipeline &&
        g_gqa_attend_n_inner_pipeline &&
        g_gqa_kv_quant_q8_pipeline &&
        g_gqa_attend_n_q8_inner_pipeline &&
        g_deltanet_recur_zero_pipeline &&
        g_deltanet_recur_pipeline && g_deltanet_recur_scratch_gates_pipeline &&
        g_deltanet_fused_gdn_scratch_pipeline &&
        g_deltanet_gated_rmsnorm_pipeline &&
        g_residual_rmsnorm_weight_f32_pipeline &&
        g_residual_rmsnorm_update_x0_pipeline && g_silu_mul_pipeline &&
        g_scale_pipeline && g_argmax_blocks_pipeline &&
        g_add_moe_to_x0_pipeline && g_silu_mul_offsets_pipeline &&
        g_scale_x1_scalar_add_x0_pipeline && g_scale_x1_add_x0_pipeline &&
        g_scale_scratch_add_x0_pipeline && g_router_top8_pipeline &&
        g_matvec_iq3_s_expert_slot_pipeline &&
        g_matvec_iq3_s_expert_slot_pair_pipeline &&
        g_matvec_iq4_xs_expert_slot_pipeline &&
        g_matvec_q6_k_expert_slot_pipeline &&
        g_scale_scratch_add_x0_slot_pipeline) return 1;
    NSError *error = nil;
    g_library = [g_device newLibraryWithSource:qw3_metal_kernel_source()
                                       options:nil
                                         error:&error];
    if (!g_library) {
        fprintf(stderr, "qw3: Metal library compile failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    id<MTLFunction> fn = [g_library newFunctionWithName:@"qw3_rmsnorm_plain"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_rmsnorm_plain not found\n");
        return 0;
    }
    g_rmsnorm_plain_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                       error:&error];
    if (!g_rmsnorm_plain_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_rmsnorm_plain failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_rmsnorm_weight_f32"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_rmsnorm_weight_f32 not found\n");
        return 0;
    }
    g_rmsnorm_weight_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                            error:&error];
    if (!g_rmsnorm_weight_f32_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_rmsnorm_weight_f32 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_embed_q8_0"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_embed_q8_0 not found\n");
        return 0;
    }
    g_embed_q8_0_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                     error:&error];
    if (!g_embed_q8_0_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_embed_q8_0 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_matvec_q8_0"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matvec_q8_0 not found\n");
        return 0;
    }
    g_matvec_q8_0_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                      error:&error];
    if (!g_matvec_q8_0_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matvec_q8_0 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_matvec_q8_0_pair"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matvec_q8_0_pair not found\n");
        return 0;
    }
    g_matvec_q8_0_pair_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                          error:&error];
    if (!g_matvec_q8_0_pair_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matvec_q8_0_pair failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_matvec_q8_0_pair_silu_fast"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matvec_q8_0_pair_silu_fast not found\n");
        return 0;
    }
    g_matvec_q8_0_pair_silu_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                                error:&error];
    if (!g_matvec_q8_0_pair_silu_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matvec_q8_0_pair_silu_fast failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_shared_gate_up_silu_fast"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_shared_gate_up_silu_fast not found\n");
        return 0;
    }
    g_shared_gate_up_silu_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                              error:&error];
    if (!g_shared_gate_up_silu_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_shared_gate_up_silu_fast failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_matvec_q8_0_inner_scale_add_x0_fast"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matvec_q8_0_inner_scale_add_x0_fast not found\n");
        return 0;
    }
    g_matvec_q8_0_inner_scale_add_x0_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                                        error:&error];
    if (!g_matvec_q8_0_inner_scale_add_x0_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matvec_q8_0_inner_scale_add_x0_fast failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_matvec_iq4_xs"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matvec_iq4_xs not found\n");
        return 0;
    }
    g_matvec_iq4_xs_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                       error:&error];
    if (!g_matvec_iq4_xs_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matvec_iq4_xs failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_matvec_iq4_xs_add_x0_fast"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matvec_iq4_xs_add_x0_fast not found\n");
        return 0;
    }
    g_matvec_iq4_xs_add_x0_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                              error:&error];
    if (!g_matvec_iq4_xs_add_x0_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matvec_iq4_xs_add_x0_fast failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_matvec_iq4_xs_swiglu_add_x0"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matvec_iq4_xs_swiglu_add_x0 not found\n");
        return 0;
    }
    g_matvec_iq4_xs_swiglu_add_x0_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                                     error:&error];
    if (!g_matvec_iq4_xs_swiglu_add_x0_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matvec_iq4_xs_swiglu_add_x0 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_matvec_q6_k_fast"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matvec_q6_k_fast not found\n");
        return 0;
    }
    g_matvec_q6_k_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                      error:&error];
    if (!g_matvec_q6_k_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matvec_q6_k_fast failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_matvec_q6_k_add_x0"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matvec_q6_k_add_x0 not found\n");
        return 0;
    }
    g_matvec_q6_k_add_x0_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                            error:&error];
    if (!g_matvec_q6_k_add_x0_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matvec_q6_k_add_x0 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_matvec_q6_k_swiglu_add_x0"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matvec_q6_k_swiglu_add_x0 not found\n");
        return 0;
    }
    g_matvec_q6_k_swiglu_add_x0_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                                  error:&error];
    if (!g_matvec_q6_k_swiglu_add_x0_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matvec_q6_k_swiglu_add_x0 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_matvec_iq3_s"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matvec_iq3_s not found\n");
        return 0;
    }
    g_matvec_iq3_s_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                      error:&error];
    if (!g_matvec_iq3_s_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matvec_iq3_s failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_matvec_iq3_s_pair_fast"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matvec_iq3_s_pair_fast not found\n");
        return 0;
    }
    g_matvec_iq3_s_pair_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                           error:&error];
    if (!g_matvec_iq3_s_pair_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matvec_iq3_s_pair_fast failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_moe_iq3_s_swiglu_batch_fast"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_moe_iq3_s_swiglu_batch_fast not found\n");
        return 0;
    }
    g_moe_iq3_s_pair_batch_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                              error:&error];
    if (!g_moe_iq3_s_pair_batch_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_moe_iq3_s_swiglu_batch_fast failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_moe_down_iq4_xs_batch_fast"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_moe_down_iq4_xs_batch_fast not found\n");
        return 0;
    }
    g_moe_down_iq4_xs_batch_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                               error:&error];
    if (!g_moe_down_iq4_xs_batch_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_moe_down_iq4_xs_batch_fast failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_moe_down_iq4_xs_pair_fast"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_moe_down_iq4_xs_pair_fast not found\n");
        return 0;
    }
    g_moe_down_iq4_xs_pair_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                               error:&error];
    if (!g_moe_down_iq4_xs_pair_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_moe_down_iq4_xs_pair_fast failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_moe_down_iq4_xs_batch_reduce_fast"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_moe_down_iq4_xs_batch_reduce_fast not found\n");
        return 0;
    }
    g_moe_down_iq4_xs_batch_reduce_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                                      error:&error];
    if (!g_moe_down_iq4_xs_batch_reduce_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_moe_down_iq4_xs_batch_reduce_fast failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_moe_down_q6_k_batch_fast"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_moe_down_q6_k_batch_fast not found\n");
        return 0;
    }
    g_moe_down_q6_k_batch_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                             error:&error];
    if (!g_moe_down_q6_k_batch_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_moe_down_q6_k_batch_fast failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_moe_reduce_batch"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_moe_reduce_batch not found\n");
        return 0;
    }
    g_moe_reduce_batch_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                          error:&error];
    if (!g_moe_reduce_batch_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_moe_reduce_batch failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_matvec_f32"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matvec_f32 not found\n");
        return 0;
    }
    g_matvec_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                    error:&error];
    if (!g_matvec_f32_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matvec_f32 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_matvec_f32_pair"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matvec_f32_pair not found\n");
        return 0;
    }
    g_matvec_f32_pair_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                          error:&error];
    if (!g_matvec_f32_pair_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matvec_f32_pair failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_matvec_f32_fast"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matvec_f32_fast not found\n");
        return 0;
    }
    g_matvec_f32_fast_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                         error:&error];
    if (!g_matvec_f32_fast_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matvec_f32_fast failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_deltanet_conv1d_zero"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_deltanet_conv1d_zero not found\n");
        return 0;
    }
    g_deltanet_conv1d_zero_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                              error:&error];
    if (!g_deltanet_conv1d_zero_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_deltanet_conv1d_zero failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_deltanet_conv1d_step"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_deltanet_conv1d_step not found\n");
        return 0;
    }
    g_deltanet_conv1d_step_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                              error:&error];
    if (!g_deltanet_conv1d_step_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_deltanet_conv1d_step failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_l2norm_heads"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_l2norm_heads not found\n");
        return 0;
    }
    g_l2norm_heads_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                      error:&error];
    if (!g_l2norm_heads_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_l2norm_heads failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_gqa_q_norm_gate"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_q_norm_gate not found\n");
        return 0;
    }
    g_gqa_q_norm_gate_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                         error:&error];
    if (!g_gqa_q_norm_gate_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_q_norm_gate failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_gqa_k_norm"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_k_norm not found\n");
        return 0;
    }
    g_gqa_k_norm_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                    error:&error];
    if (!g_gqa_k_norm_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_k_norm failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_rope_heads"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_rope_heads not found\n");
        return 0;
    }
    g_rope_heads_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                    error:&error];
    if (!g_rope_heads_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_rope_heads failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_gqa_single_token_inner"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_single_token_inner not found\n");
        return 0;
    }
    g_gqa_single_token_inner_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                                error:&error];
    if (!g_gqa_single_token_inner_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_single_token_inner failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_gqa_attend2_inner"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_attend2_inner not found\n");
        return 0;
    }
    g_gqa_attend2_inner_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                           error:&error];
    if (!g_gqa_attend2_inner_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_attend2_inner failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_gqa_attend_n_inner"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_attend_n_inner not found\n");
        return 0;
    }
    g_gqa_attend_n_inner_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                            error:&error];
    if (!g_gqa_attend_n_inner_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_attend_n_inner failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_gqa_kv_quant_q8"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_kv_quant_q8 not found\n");
        return 0;
    }
    g_gqa_kv_quant_q8_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                          error:&error];
    if (!g_gqa_kv_quant_q8_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_kv_quant_q8 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_gqa_attend_n_q8_inner"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_attend_n_q8_inner not found\n");
        return 0;
    }
    g_gqa_attend_n_q8_inner_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_gqa_attend_n_q8_inner_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_attend_n_q8_inner failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_deltanet_recur_zero"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_deltanet_recur_zero not found\n");
        return 0;
    }
    g_deltanet_recur_zero_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                             error:&error];
    if (!g_deltanet_recur_zero_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_deltanet_recur_zero failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_deltanet_recur"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_deltanet_recur not found\n");
        return 0;
    }
    g_deltanet_recur_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                        error:&error];
    if (!g_deltanet_recur_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_deltanet_recur failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_deltanet_recur_scratch_gates"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_deltanet_recur_scratch_gates not found\n");
        return 0;
    }
    g_deltanet_recur_scratch_gates_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                                      error:&error];
    if (!g_deltanet_recur_scratch_gates_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_deltanet_recur_scratch_gates failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_deltanet_fused_gdn_scratch"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_deltanet_fused_gdn_scratch not found\n");
        return 0;
    }
    g_deltanet_fused_gdn_scratch_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_deltanet_fused_gdn_scratch_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_deltanet_fused_gdn_scratch failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_deltanet_gated_rmsnorm"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_deltanet_gated_rmsnorm not found\n");
        return 0;
    }
    g_deltanet_gated_rmsnorm_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                                error:&error];
    if (!g_deltanet_gated_rmsnorm_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_deltanet_gated_rmsnorm failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_residual_rmsnorm_weight_f32"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_residual_rmsnorm_weight_f32 not found\n");
        return 0;
    }
    g_residual_rmsnorm_weight_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                                     error:&error];
    if (!g_residual_rmsnorm_weight_f32_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_residual_rmsnorm_weight_f32 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_residual_rmsnorm_update_x0"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_residual_rmsnorm_update_x0 not found\n");
        return 0;
    }
    g_residual_rmsnorm_update_x0_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                                    error:&error];
    if (!g_residual_rmsnorm_update_x0_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_residual_rmsnorm_update_x0 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_silu_mul"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_silu_mul not found\n");
        return 0;
    }
    g_silu_mul_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                  error:&error];
    if (!g_silu_mul_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_silu_mul failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_scale"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_scale not found\n");
        return 0;
    }
    g_scale_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                               error:&error];
    if (!g_scale_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_scale failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_add_moe_to_x0"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_add_moe_to_x0 not found\n");
        return 0;
    }
    g_add_moe_to_x0_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                       error:&error];
    if (!g_add_moe_to_x0_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_add_moe_to_x0 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_silu_mul_offsets"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_silu_mul_offsets not found\n");
        return 0;
    }
    g_silu_mul_offsets_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                          error:&error];
    if (!g_silu_mul_offsets_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_silu_mul_offsets failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_scale_x1_scalar_add_x0"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_scale_x1_scalar_add_x0 not found\n");
        return 0;
    }
    g_scale_x1_scalar_add_x0_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                                error:&error];
    if (!g_scale_x1_scalar_add_x0_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_scale_x1_scalar_add_x0 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_scale_x1_add_x0"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_scale_x1_add_x0 not found\n");
        return 0;
    }
    g_scale_x1_add_x0_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                         error:&error];
    if (!g_scale_x1_add_x0_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_scale_x1_add_x0 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_scale_scratch_add_x0"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_scale_scratch_add_x0 not found\n");
        return 0;
    }
    g_scale_scratch_add_x0_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                              error:&error];
    if (!g_scale_scratch_add_x0_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_scale_scratch_add_x0 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_router_top8"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_router_top8 not found\n");
        return 0;
    }
    g_router_top8_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                     error:&error];
    if (!g_router_top8_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_router_top8 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_matvec_iq3_s_expert_slot"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matvec_iq3_s_expert_slot not found\n");
        return 0;
    }
    g_matvec_iq3_s_expert_slot_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                                  error:&error];
    if (!g_matvec_iq3_s_expert_slot_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matvec_iq3_s_expert_slot failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_matvec_iq3_s_expert_slot_pair_fast"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matvec_iq3_s_expert_slot_pair_fast not found\n");
        return 0;
    }
    g_matvec_iq3_s_expert_slot_pair_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                                       error:&error];
    if (!g_matvec_iq3_s_expert_slot_pair_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matvec_iq3_s_expert_slot_pair_fast failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_matvec_iq4_xs_expert_slot_add_x0_fast"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matvec_iq4_xs_expert_slot_add_x0_fast not found\n");
        return 0;
    }
    g_matvec_iq4_xs_expert_slot_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                                   error:&error];
    if (!g_matvec_iq4_xs_expert_slot_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matvec_iq4_xs_expert_slot_add_x0_fast failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_matvec_q6_k_expert_slot"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matvec_q6_k_expert_slot not found\n");
        return 0;
    }
    g_matvec_q6_k_expert_slot_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                                 error:&error];
    if (!g_matvec_q6_k_expert_slot_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matvec_q6_k_expert_slot failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_scale_scratch_add_x0_slot"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_scale_scratch_add_x0_slot not found\n");
        return 0;
    }
    g_scale_scratch_add_x0_slot_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                                   error:&error];
    if (!g_scale_scratch_add_x0_slot_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_scale_scratch_add_x0_slot failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_argmax_blocks"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_argmax_blocks not found\n");
        return 0;
    }
    g_argmax_blocks_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                       error:&error];
    if (!g_argmax_blocks_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_argmax_blocks failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

static id<MTLBuffer> qw3_metal_model_view_for(uint64_t offset, uint64_t bytes,
                                              uint64_t *inner_offset) {
    if (!g_model_map_ptr || offset > g_model_map_size ||
        bytes > g_model_map_size - offset) {
        return nil;
    }
    const uint8_t *ptr = (const uint8_t *)g_model_map_ptr + offset;
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        uint64_t vs = g_model_view_sizes[i];
        const uint8_t *base = g_model_view_ptrs[i];
        if (!base) continue;
        uintptr_t delta = (uintptr_t)(ptr - base);
        if (ptr >= base && (uint64_t)delta <= vs &&
            bytes <= vs - (uint64_t)delta) {
            if (inner_offset) *inner_offset = (uint64_t)delta;
            return [g_model_buffers objectAtIndex:i];
        }
    }
    return nil;
}

static id<MTLBuffer> qw3_metal_model_temp_buffer_for(uint64_t offset,
                                                     uint64_t bytes,
                                                     uint64_t *inner_offset) {
    if (!g_model_map_ptr || offset > g_model_map_size ||
        bytes > g_model_map_size - offset) {
        return nil;
    }
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t page_offset = offset & ~(page - 1);
    const uint64_t leading = offset - page_offset;
    const uint64_t wrap_size = round_up_u64(leading + bytes, page);
    if (wrap_size > (uint64_t)[g_device maxBufferLength]) return nil;
    if (!g_model_temp_buffers) {
        g_model_temp_buffers = [[NSMutableDictionary alloc] init];
    }
    NSString *key = [NSString stringWithFormat:@"%llu:%llu",
                     (unsigned long long)page_offset,
                     (unsigned long long)wrap_size];
    id<MTLBuffer> cached = [g_model_temp_buffers objectForKey:key];
    if (cached) {
        if (inner_offset) *inner_offset = leading;
        return cached;
    }
    id<MTLBuffer> b = [g_device newBufferWithBytesNoCopy:(void *)((const uint8_t *)g_model_map_ptr + page_offset)
                                                  length:(NSUInteger)wrap_size
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
    if (b) [g_model_temp_buffers setObject:b forKey:key];
    if (inner_offset) *inner_offset = leading;
    return b;
}

const char *qw3_metal_device_name(void) {
    return g_device_name[0] ? g_device_name : "unknown Metal device";
}

int qw3_metal_begin_commands(void) {
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (g_batch_cb) return 0;
    if (getenv("QW3_METAL_UNRETAINED_COMMAND_BUFFERS") != NULL) {
        g_batch_cb = [g_queue commandBufferWithUnretainedReferences];
    } else {
        g_batch_cb = [g_queue commandBuffer];
    }
    return g_batch_cb != nil;
}

int qw3_metal_flush_commands(void) {
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!g_batch_cb) return 0;
    qw3_metal_close_batch_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    [cb commit];
    [g_pending_cbs addObject:cb];
    if (getenv("QW3_METAL_UNRETAINED_COMMAND_BUFFERS") != NULL) {
        g_batch_cb = [g_queue commandBufferWithUnretainedReferences];
    } else {
        g_batch_cb = [g_queue commandBuffer];
    }
    if (!g_batch_cb) {
        (void)qw3_metal_wait_pending_command_buffers("command batch");
        return 0;
    }
    return 1;
}

int qw3_metal_end_commands(void) {
    if (!g_batch_cb) return 0;
    qw3_metal_close_batch_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    return qw3_metal_finish_command_buffer(cb, 1, "command batch");
}

int qw3_metal_synchronize(void) {
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (g_batch_cb) return qw3_metal_end_commands();
    if ([g_pending_cbs count] != 0) {
        return qw3_metal_wait_pending_command_buffers("synchronize");
    }
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    return qw3_metal_finish_command_buffer(cb, owned, "synchronize");
}

static uint64_t qw3_metal_session_align(uint64_t n) {
    const uint64_t align = 256;
    return (n + align - 1) & ~(align - 1);
}

static id<MTLBuffer> qw3_metal_new_private_buffer(uint64_t bytes) {
    if (!g_device || bytes == 0) return nil;
    return [g_device newBufferWithLength:(NSUInteger)qw3_metal_session_align(bytes)
                                 options:MTLResourceStorageModePrivate];
}

static id<MTLBuffer> qw3_metal_iq3s_kgrid_buffer(void) {
    if (!g_device) return nil;
    if (!g_iq3s_kgrid_buffer) {
        g_iq3s_kgrid_buffer = [g_device newBufferWithBytes:g_iq3s_kgrid
                                                    length:sizeof(g_iq3s_kgrid)
                                                   options:MTLResourceStorageModeShared];
    }
    return g_iq3s_kgrid_buffer;
}

qw3_metal_session *qw3_metal_session_create(uint32_t ctx_size,
                                            uint32_t vocab_size) {
    if (!g_initialized || !g_device || ctx_size == 0 || vocab_size == 0) {
        return NULL;
    }

    const char *kv_q8_env = getenv("QW3_METAL_KV_Q8_0");
    const BOOL gqa_kv_q8 = kv_q8_env && strcmp(kv_q8_env, "0") != 0;
    const uint64_t gqa_cache_token_bytes = gqa_kv_q8 ?
        (uint64_t)QW3_METAL_N_HEAD_KV * (QW3_METAL_N_HEAD_DIM / 32u) * 34ull :
        (uint64_t)QW3_METAL_N_HEAD_KV * QW3_METAL_N_HEAD_DIM * sizeof(float);
    const uint64_t gqa_kv_bytes =
        (uint64_t)QW3_METAL_N_FULL_ATTN_LAYERS * ctx_size *
        gqa_cache_token_bytes;
    const uint64_t deltanet_state_bytes =
        (uint64_t)QW3_METAL_N_LINEAR_LAYERS * QW3_METAL_N_LINEAR_V_HEADS *
        QW3_METAL_N_LINEAR_HEAD_DIM * QW3_METAL_N_LINEAR_HEAD_DIM *
        sizeof(float);
    const uint64_t conv_state_bytes =
        (uint64_t)QW3_METAL_N_LINEAR_LAYERS * QW3_METAL_LINEAR_QKV *
        (QW3_METAL_LINEAR_CONV_K - 1) * sizeof(float);
    const uint64_t logits_bytes = (uint64_t)vocab_size * sizeof(float);
    const uint64_t scratch_bytes = 32ull * QW3_METAL_N_EMBD * sizeof(float);
    const uint64_t qkv_conv_bytes = (uint64_t)QW3_METAL_LINEAR_QKV * sizeof(float);
    const uint64_t qk_norm_bytes =
        (uint64_t)QW3_METAL_N_LINEAR_QK_HEADS * QW3_METAL_N_LINEAR_HEAD_DIM *
        sizeof(float);
    const uint64_t inner_bytes = (uint64_t)QW3_METAL_LINEAR_INNER * sizeof(float);
    const uint64_t gqa_q_bytes =
        (uint64_t)QW3_METAL_N_HEAD * QW3_METAL_N_HEAD_DIM * sizeof(float);
    const uint64_t gqa_kv_token_bytes =
        (uint64_t)QW3_METAL_N_HEAD_KV * QW3_METAL_N_HEAD_DIM * sizeof(float);

    QW3MetalSessionObj *obj = [[QW3MetalSessionObj alloc] init];
    obj.ctxSize = ctx_size;
    obj.vocabSize = vocab_size;
    obj.gqaKvQ8 = gqa_kv_q8;
    qw3_metal_session_info info = {
        .gqa_kv_bytes = 2 * gqa_kv_bytes,
        .deltanet_state_bytes = deltanet_state_bytes,
        .conv_state_bytes = conv_state_bytes,
        .logits_bytes = logits_bytes,
        .scratch_bytes = scratch_bytes + qkv_conv_bytes + 2 * qk_norm_bytes +
                         2 * inner_bytes + 4 * gqa_q_bytes + 2 * gqa_kv_token_bytes,
    };
    info.total_bytes = info.gqa_kv_bytes +
                       info.deltanet_state_bytes +
                       info.conv_state_bytes +
                       info.logits_bytes +
                       info.scratch_bytes;
    obj.info = info;

    obj.gqaK = qw3_metal_new_private_buffer(gqa_kv_bytes);
    obj.gqaV = qw3_metal_new_private_buffer(gqa_kv_bytes);
    obj.deltanetState = qw3_metal_new_private_buffer(deltanet_state_bytes);
    obj.convState = qw3_metal_new_private_buffer(conv_state_bytes);
    obj.logits = qw3_metal_new_private_buffer(logits_bytes);
    obj.x0 = qw3_metal_new_private_buffer((uint64_t)QW3_METAL_N_EMBD * sizeof(float));
    obj.x1 = qw3_metal_new_private_buffer((uint64_t)QW3_METAL_N_EMBD * sizeof(float));
    obj.scratch = qw3_metal_new_private_buffer(scratch_bytes);
    obj.qkvConv = qw3_metal_new_private_buffer(qkv_conv_bytes);
    obj.qNorm = qw3_metal_new_private_buffer(qk_norm_bytes);
    obj.kNorm = qw3_metal_new_private_buffer(qk_norm_bytes);
    obj.core = qw3_metal_new_private_buffer(inner_bytes);
    obj.inner = qw3_metal_new_private_buffer(inner_bytes);
    obj.gqaTmpQ = qw3_metal_new_private_buffer(gqa_q_bytes);
    obj.gqaTmpK = qw3_metal_new_private_buffer(gqa_kv_token_bytes);
    obj.gqaTokenQ = qw3_metal_new_private_buffer(gqa_q_bytes);
    obj.gqaTokenK = qw3_metal_new_private_buffer(gqa_kv_token_bytes);
    obj.gqaTokenV = qw3_metal_new_private_buffer(gqa_kv_token_bytes);
    obj.gqaTokenGate = qw3_metal_new_private_buffer(gqa_q_bytes);
    obj.routerIds = qw3_metal_new_private_buffer(8ull * sizeof(int32_t));
    obj.routerWeights = qw3_metal_new_private_buffer(8ull * sizeof(float));
    const uint64_t argmax_blocks = ((uint64_t)vocab_size + 255ull) / 256ull;
    obj.argmaxVals = [g_device newBufferWithLength:(NSUInteger)(argmax_blocks * sizeof(float))
                                           options:MTLResourceStorageModeShared];
    obj.argmaxIdxs = [g_device newBufferWithLength:(NSUInteger)(argmax_blocks * sizeof(uint32_t))
                                           options:MTLResourceStorageModeShared];

    if (!obj.gqaK || !obj.gqaV || !obj.deltanetState || !obj.convState ||
        !obj.logits || !obj.x0 || !obj.x1 || !obj.scratch ||
        !obj.qkvConv || !obj.qNorm || !obj.kNorm || !obj.core || !obj.inner ||
        !obj.gqaTmpQ || !obj.gqaTmpK || !obj.gqaTokenQ || !obj.gqaTokenK ||
        !obj.gqaTokenV || !obj.gqaTokenGate || !obj.routerIds ||
        !obj.routerWeights || !obj.argmaxVals || !obj.argmaxIdxs) {
        return NULL;
    }

    qw3_metal_session *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->obj = (__bridge_retained void *)obj;
    if (!qw3_metal_session_clear(s)) {
        qw3_metal_session_free(s);
        return NULL;
    }
    return s;
}

void qw3_metal_session_free(qw3_metal_session *s) {
    if (!s) return;
    if (s->obj) {
        CFBridgingRelease(s->obj);
        s->obj = NULL;
    }
    free(s);
}

int qw3_metal_session_clear(qw3_metal_session *s) {
    if (!s || !s->obj || !g_queue) return 0;
    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    if (!cb) return 0;
    qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!blit) return 0;
    NSArray<id<MTLBuffer>> *buffers = @[
        obj.gqaK, obj.gqaV, obj.deltanetState, obj.convState,
        obj.logits, obj.x0, obj.x1, obj.scratch,
        obj.qkvConv, obj.qNorm, obj.kNorm, obj.core, obj.inner,
        obj.gqaTmpQ, obj.gqaTmpK, obj.gqaTokenQ, obj.gqaTokenK,
        obj.gqaTokenV, obj.gqaTokenGate, obj.routerIds, obj.routerWeights
    ];
    for (id<MTLBuffer> b in buffers) {
        if (b.length > 0) {
            [blit fillBuffer:b range:NSMakeRange(0, b.length) value:0];
        }
    }
    [blit endEncoding];
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    obj.pos = 0;
    return cb.status == MTLCommandBufferStatusCompleted;
}

qw3_metal_session_info qw3_metal_session_get_info(qw3_metal_session *s) {
    if (!s || !s->obj) return (qw3_metal_session_info){0};
    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    return obj.info;
}

int qw3_metal_session_embed_q8_0(qw3_metal_session *s, uint64_t tensor_offset,
                                 uint32_t token, uint32_t n_embd,
                                 float *out) {
    if (!s || !s->obj || n_embd == 0 || (n_embd % 32) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t out_bytes = (uint64_t)n_embd * sizeof(float);
    if (!obj.x0 || obj.x0.length < out_bytes) return 0;

    const uint64_t row_bytes = (uint64_t)(n_embd / 32) * 34ull;
    const uint64_t row_offset = tensor_offset + (uint64_t)token * row_bytes;
    uint64_t inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(row_offset, row_bytes, &inner);
    if (!wb) {
        fprintf(stderr, "qw3: Metal session embedding row is outside mapped model views\n");
        return 0;
    }

    struct {
        uint32_t n_embd;
        uint32_t row_bytes;
    } args = { n_embd, (uint32_t)row_bytes };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_embed_q8_0_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)inner atIndex:1];
    [enc setBuffer:obj.x0 offset:0 atIndex:2];
    NSUInteger threads = g_embed_q8_0_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake(n_embd, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    id<MTLBuffer> readback = nil;
    if (out) {
        readback = [g_device newBufferWithLength:(NSUInteger)out_bytes
                                         options:MTLResourceStorageModeShared];
        if (!readback) return 0;
        qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:obj.x0 sourceOffset:0
                    toBuffer:readback destinationOffset:0
                        size:(NSUInteger)out_bytes];
        [blit endEncoding];
    }

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session embedding command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    if (out) memcpy(out, readback.contents, (size_t)out_bytes);
    return 1;
}

int qw3_metal_session_write_x0(qw3_metal_session *s, const float *x,
                               uint32_t n) {
    if (!s || !s->obj || !x || n == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    if (!obj.x0 || obj.x0.length < bytes) return 0;

    id<MTLBuffer> xb = [g_device newBufferWithBytes:x
                                             length:(NSUInteger)bytes
                                            options:MTLResourceStorageModeShared];
    if (!xb) return 0;
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:xb sourceOffset:0
                toBuffer:obj.x0 destinationOffset:0
                    size:(NSUInteger)bytes];
    [blit endEncoding];
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session write x0 command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_read_x0(qw3_metal_session *s, float *out,
                              uint32_t n) {
    if (!s || !s->obj || !out || n == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    if (!obj.x0 || obj.x0.length < bytes) return 0;

    id<MTLBuffer> readback = [g_device newBufferWithLength:(NSUInteger)bytes
                                                    options:MTLResourceStorageModeShared];
    if (!readback) return 0;
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:obj.x0 sourceOffset:0
                toBuffer:readback destinationOffset:0
                    size:(NSUInteger)bytes];
    [blit endEncoding];
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session read x0 command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, readback.contents, (size_t)bytes);
    return 1;
}

int qw3_metal_session_add_moe_to_x0(qw3_metal_session *s, const float *moe,
                                    uint32_t n) {
    if (!s || !s->obj || !moe || n == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    if (!obj.x0 || !obj.x1 || obj.x0.length < bytes || obj.x1.length < bytes) {
        return 0;
    }
    id<MTLBuffer> mb = [g_device newBufferWithBytes:moe
                                             length:(NSUInteger)bytes
                                            options:MTLResourceStorageModeShared];
    if (!mb) return 0;
    struct {
        uint32_t n;
        float scale;
    } args = { n, 1.0f };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_add_moe_to_x0_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.x0 offset:0 atIndex:1];
    [enc setBuffer:obj.x1 offset:0 atIndex:2];
    [enc setBuffer:mb offset:0 atIndex:3];
    NSUInteger threads = g_add_moe_to_x0_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session add MoE command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_silu_mul_scratch_to_inner(qw3_metal_session *s,
                                                uint32_t a_offset,
                                                uint32_t b_offset,
                                                uint32_t n) {
    if (!s || !s->obj || n == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t a_bytes = (uint64_t)a_offset * sizeof(float);
    const uint64_t b_bytes = (uint64_t)b_offset * sizeof(float);
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    if (!obj.scratch || !obj.inner ||
        obj.scratch.length < a_bytes + bytes ||
        obj.scratch.length < b_bytes + bytes ||
        obj.inner.length < bytes) {
        return 0;
    }
    struct {
        uint32_t n;
        uint32_t a_offset;
        uint32_t b_offset;
    } args = { n, a_offset, b_offset };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_silu_mul_offsets_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.scratch offset:0 atIndex:1];
    [enc setBuffer:obj.inner offset:0 atIndex:2];
    NSUInteger threads = g_silu_mul_offsets_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session scratch silu_mul command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_scale_x1_by_scratch_scalar_add_x0(qw3_metal_session *s,
                                                        uint32_t scalar_offset,
                                                        uint32_t n) {
    if (!s || !s->obj || n == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    const uint64_t scalar_bytes = (uint64_t)scalar_offset * sizeof(float);
    if (!obj.x0 || !obj.x1 || !obj.scratch ||
        obj.x0.length < bytes || obj.x1.length < bytes ||
        obj.scratch.length < scalar_bytes + sizeof(float)) {
        return 0;
    }
    struct {
        uint32_t n;
        uint32_t a_offset;
        uint32_t b_offset;
    } args = { n, scalar_offset, 0 };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_scale_x1_scalar_add_x0_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.x0 offset:0 atIndex:1];
    [enc setBuffer:obj.x1 offset:0 atIndex:2];
    [enc setBuffer:obj.scratch offset:0 atIndex:3];
    NSUInteger threads = g_scale_x1_scalar_add_x0_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session shared add command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_scale_x1_add_x0(qw3_metal_session *s,
                                      float scale,
                                      uint32_t n) {
    if (!s || !s->obj || n == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    if (!obj.x0 || !obj.x1 || obj.x0.length < bytes || obj.x1.length < bytes) {
        return 0;
    }
    struct {
        uint32_t n;
        float scale;
    } args = { n, scale };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_scale_x1_add_x0_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.x0 offset:0 atIndex:1];
    [enc setBuffer:obj.x1 offset:0 atIndex:2];
    NSUInteger threads = g_scale_x1_add_x0_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session scale-add command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_scale_scratch_add_x0(qw3_metal_session *s,
                                           uint32_t scratch_offset,
                                           float scale,
                                           uint32_t n) {
    if (!s || !s->obj || n == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    const uint64_t scratch_offset_bytes = (uint64_t)scratch_offset * sizeof(float);
    if (!obj.x0 || !obj.scratch || obj.x0.length < bytes ||
        obj.scratch.length < scratch_offset_bytes ||
        obj.scratch.length - scratch_offset_bytes < bytes) {
        return 0;
    }
    struct {
        uint32_t n;
        uint32_t a_offset;
        uint32_t b_offset;
    } args = { n, scratch_offset, 0 };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_scale_scratch_add_x0_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBytes:&scale length:sizeof(scale) atIndex:1];
    [enc setBuffer:obj.x0 offset:0 atIndex:2];
    [enc setBuffer:obj.scratch offset:0 atIndex:3];
    NSUInteger threads = g_scale_scratch_add_x0_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session scratch scale-add command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_rmsnorm_weight_f32(qw3_metal_session *s,
                                         uint64_t weight_offset,
                                         uint32_t n, float eps,
                                         float *out) {
    if (!s || !s->obj || n == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    if (!obj.x0 || !obj.x1 || obj.x0.length < bytes || obj.x1.length < bytes) {
        return 0;
    }

    const uint64_t weight_bytes = (uint64_t)n * sizeof(float);
    if (!g_model_map_ptr || weight_offset > g_model_map_size ||
        weight_bytes > g_model_map_size - weight_offset) {
        fprintf(stderr, "qw3: Metal session RMSNorm weight is outside mapped model\n");
        return 0;
    }
    uint64_t weight_inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(weight_offset, weight_bytes, &weight_inner);
    if (!wb) return 0;

    struct {
        uint32_t n;
        float eps;
    } args = { n, eps };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_rmsnorm_weight_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.x0 offset:0 atIndex:1];
    [enc setBuffer:wb offset:(NSUInteger)weight_inner atIndex:2];
    [enc setBuffer:obj.x1 offset:0 atIndex:3];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_rmsnorm_weight_f32_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    id<MTLBuffer> readback = nil;
    if (out) {
        readback = [g_device newBufferWithLength:(NSUInteger)bytes
                                         options:MTLResourceStorageModeShared];
        if (!readback) return 0;
        qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:obj.x1 sourceOffset:0
                    toBuffer:readback destinationOffset:0
                        size:(NSUInteger)bytes];
        [blit endEncoding];
    }

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session RMSNorm command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    if (out) memcpy(out, readback.contents, (size_t)bytes);
    return 1;
}

int qw3_metal_session_matvec_q8_0_x1(qw3_metal_session *s,
                                     uint64_t tensor_offset,
                                     uint32_t n_in, uint32_t n_out,
                                     float *out) {
    return qw3_metal_session_matvec_q8_0_x1_to_scratch(
        s, tensor_offset, n_in, n_out, 0, out);
}

int qw3_metal_session_matvec_q8_0_x1_to_scratch(qw3_metal_session *s,
                                                uint64_t tensor_offset,
                                                uint32_t n_in,
                                                uint32_t n_out,
                                                uint32_t out_offset,
                                                float *out) {
    if (!s || !s->obj || n_in == 0 || n_out == 0 || (n_in % 32) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_in * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_out * sizeof(float);
    const uint64_t out_offset_bytes = (uint64_t)out_offset * sizeof(float);
    if (!obj.x1 || !obj.scratch ||
        obj.x1.length < x_bytes ||
        obj.scratch.length < out_offset_bytes ||
        obj.scratch.length - out_offset_bytes < out_bytes) {
        return 0;
    }

    const uint64_t row_bytes = (uint64_t)(n_in / 32) * 34ull;
    const uint64_t tensor_bytes = row_bytes * (uint64_t)n_out;
    uint64_t inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(tensor_offset, tensor_bytes, &inner);
    if (!wb) {
        fprintf(stderr, "qw3: Metal session q8_0 matvec tensor is outside mapped model views\n");
        return 0;
    }

    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
    } args = { n_in, n_out, (uint32_t)row_bytes };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_matvec_q8_0_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)inner atIndex:1];
    [enc setBuffer:obj.x1 offset:0 atIndex:2];
    [enc setBuffer:obj.scratch offset:(NSUInteger)out_offset_bytes atIndex:3];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_matvec_f32_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(n_out, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    id<MTLBuffer> readback = nil;
    if (out) {
        readback = [g_device newBufferWithLength:(NSUInteger)out_bytes
                                         options:MTLResourceStorageModeShared];
        if (!readback) return 0;
        qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:obj.scratch sourceOffset:(NSUInteger)out_offset_bytes
                    toBuffer:readback destinationOffset:0
                        size:(NSUInteger)out_bytes];
        [blit endEncoding];
    }

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session q8_0 matvec command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    if (out) memcpy(out, readback.contents, (size_t)out_bytes);
    return 1;
}

int qw3_metal_session_matvec_q8_0_pair_x1_to_scratch(
    qw3_metal_session *s, uint64_t tensor_a_offset, uint64_t tensor_b_offset,
    uint32_t n_in, uint32_t n_out, uint32_t out_a_offset,
    uint32_t out_b_offset) {
    if (!s || !s->obj || n_in == 0 || n_out == 0 || (n_in % 32) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_in * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_out * sizeof(float);
    const uint64_t out_a_offset_bytes = (uint64_t)out_a_offset * sizeof(float);
    const uint64_t out_b_offset_bytes = (uint64_t)out_b_offset * sizeof(float);
    if (!obj.x1 || !obj.scratch ||
        obj.x1.length < x_bytes ||
        obj.scratch.length < out_a_offset_bytes ||
        obj.scratch.length - out_a_offset_bytes < out_bytes ||
        obj.scratch.length < out_b_offset_bytes ||
        obj.scratch.length - out_b_offset_bytes < out_bytes) {
        return 0;
    }

    const uint64_t row_bytes = (uint64_t)(n_in / 32) * 34ull;
    const uint64_t tensor_bytes = row_bytes * (uint64_t)n_out;
    uint64_t inner_a = 0, inner_b = 0;
    id<MTLBuffer> wa = qw3_metal_model_view_for(tensor_a_offset, tensor_bytes, &inner_a);
    id<MTLBuffer> wb = qw3_metal_model_view_for(tensor_b_offset, tensor_bytes, &inner_b);
    if (!wa || !wb) {
        fprintf(stderr, "qw3: Metal session q8_0 pair tensor is outside mapped model views\n");
        return 0;
    }

    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
        uint32_t out_a_offset;
        uint32_t out_b_offset;
    } args = { n_in, n_out, (uint32_t)row_bytes, out_a_offset, out_b_offset };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_matvec_q8_0_pair_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wa offset:(NSUInteger)inner_a atIndex:1];
    [enc setBuffer:wb offset:(NSUInteger)inner_b atIndex:2];
    [enc setBuffer:obj.x1 offset:0 atIndex:3];
    [enc setBuffer:obj.scratch offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:64 * sizeof(float) atIndex:0];
    NSUInteger threads = g_matvec_q8_0_pair_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(n_out, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session q8_0 pair command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_matvec_q8_0_pair_silu_x1_to_inner(
    qw3_metal_session *s, uint64_t tensor_a_offset, uint64_t tensor_b_offset,
    uint32_t n_in, uint32_t n_out) {
    if (!s || !s->obj || n_in == 0 || n_out == 0 || (n_in % 32) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_in * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_out * sizeof(float);
    if (!obj.x1 || !obj.inner ||
        obj.x1.length < x_bytes || obj.inner.length < out_bytes) {
        return 0;
    }

    const uint64_t row_bytes = (uint64_t)(n_in / 32) * 34ull;
    const uint64_t tensor_bytes = row_bytes * (uint64_t)n_out;
    uint64_t inner_a = 0, inner_b = 0;
    id<MTLBuffer> wa = qw3_metal_model_view_for(tensor_a_offset, tensor_bytes, &inner_a);
    id<MTLBuffer> wb = qw3_metal_model_view_for(tensor_b_offset, tensor_bytes, &inner_b);
    if (!wa || !wb) {
        fprintf(stderr, "qw3: Metal session q8_0 pair silu tensor is outside mapped model views\n");
        return 0;
    }

    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
        uint32_t out_a_offset;
        uint32_t out_b_offset;
    } args = { n_in, n_out, (uint32_t)row_bytes, 0, 0 };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_matvec_q8_0_pair_silu_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wa offset:(NSUInteger)inner_a atIndex:1];
    [enc setBuffer:wb offset:(NSUInteger)inner_b atIndex:2];
    [enc setBuffer:obj.x1 offset:0 atIndex:3];
    [enc setBuffer:obj.inner offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((n_out + 7u) / 8u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session q8_0 pair silu command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_shared_gate_up_silu_x1_to_inner(
    qw3_metal_session *s, uint64_t tensor_a_offset, uint64_t tensor_b_offset,
    uint64_t scalar_weight_offset, uint32_t n_in, uint32_t n_out,
    uint32_t scalar_offset) {
    if (!s || !s->obj || n_in == 0 || n_out == 0 || (n_in % 32) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_in * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_out * sizeof(float);
    const uint64_t scalar_bytes = (uint64_t)scalar_offset * sizeof(float);
    if (!obj.x1 || !obj.inner || !obj.scratch ||
        obj.x1.length < x_bytes || obj.inner.length < out_bytes ||
        obj.scratch.length < scalar_bytes + sizeof(float)) {
        return 0;
    }

    const uint64_t row_bytes = (uint64_t)(n_in / 32) * 34ull;
    const uint64_t tensor_bytes = row_bytes * (uint64_t)n_out;
    const uint64_t scalar_weight_bytes = (uint64_t)n_in * sizeof(float);
    uint64_t inner_a = 0, inner_b = 0, inner_scalar = 0;
    id<MTLBuffer> wa = qw3_metal_model_view_for(tensor_a_offset, tensor_bytes, &inner_a);
    id<MTLBuffer> wb = qw3_metal_model_view_for(tensor_b_offset, tensor_bytes, &inner_b);
    id<MTLBuffer> ws = qw3_metal_model_view_for(scalar_weight_offset,
                                                scalar_weight_bytes,
                                                &inner_scalar);
    if (!wa || !wb || !ws) {
        fprintf(stderr, "qw3: Metal session shared gate/up tensor is outside mapped model views\n");
        return 0;
    }

    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
        uint32_t scalar_offset;
    } args = { n_in, n_out, (uint32_t)row_bytes, scalar_offset };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_shared_gate_up_silu_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wa offset:(NSUInteger)inner_a atIndex:1];
    [enc setBuffer:wb offset:(NSUInteger)inner_b atIndex:2];
    [enc setBuffer:ws offset:(NSUInteger)inner_scalar atIndex:3];
    [enc setBuffer:obj.x1 offset:0 atIndex:4];
    [enc setBuffer:obj.inner offset:0 atIndex:5];
    [enc setBuffer:obj.scratch offset:0 atIndex:6];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((n_out + 7u) / 8u + 1u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session shared gate/up command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_conv1d_zero_from_scratch(qw3_metal_session *s,
                                               uint64_t weight_offset,
                                               uint32_t n_channels,
                                               float *out) {
    if (!s || !s->obj || n_channels == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t bytes = (uint64_t)n_channels * sizeof(float);
    if (!obj.scratch || !obj.qkvConv ||
        obj.scratch.length < bytes || obj.qkvConv.length < bytes) {
        return 0;
    }

    const uint64_t weight_bytes = (uint64_t)n_channels * 4ull * sizeof(float);
    if (!g_model_map_ptr || weight_offset > g_model_map_size ||
        weight_bytes > g_model_map_size - weight_offset) {
        fprintf(stderr, "qw3: Metal session DeltaNet conv weight is outside mapped model\n");
        return 0;
    }
    uint64_t weight_inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(weight_offset, weight_bytes, &weight_inner);
    if (!wb) return 0;

    struct {
        uint32_t n_channels;
    } args = { n_channels };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_deltanet_conv1d_zero_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)weight_inner atIndex:1];
    [enc setBuffer:obj.scratch offset:0 atIndex:2];
    [enc setBuffer:obj.qkvConv offset:0 atIndex:3];
    NSUInteger threads = g_deltanet_conv1d_zero_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake(n_channels, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    id<MTLBuffer> readback = nil;
    if (out) {
        readback = [g_device newBufferWithLength:(NSUInteger)bytes
                                         options:MTLResourceStorageModeShared];
        if (!readback) return 0;
        qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:obj.qkvConv sourceOffset:0
                    toBuffer:readback destinationOffset:0
                        size:(NSUInteger)bytes];
        [blit endEncoding];
    }

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session DeltaNet conv command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    if (out) memcpy(out, readback.contents, (size_t)bytes);
    return 1;
}

int qw3_metal_session_conv1d_step_from_scratch(qw3_metal_session *s,
                                               uint64_t weight_offset,
                                               uint32_t layer_slot,
                                               uint32_t n_channels,
                                               float *out,
                                               float *state_out) {
    if (!s || !s->obj || n_channels == 0 ||
        layer_slot >= QW3_METAL_N_LINEAR_LAYERS) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t qkv_bytes = (uint64_t)n_channels * sizeof(float);
    const uint64_t state_bytes = (uint64_t)n_channels * 3ull * sizeof(float);
    const uint64_t state_offset = (uint64_t)layer_slot * state_bytes;
    if (!obj.scratch || !obj.qkvConv || !obj.convState ||
        obj.scratch.length < qkv_bytes ||
        obj.qkvConv.length < qkv_bytes ||
        obj.convState.length < state_offset ||
        obj.convState.length - state_offset < state_bytes) {
        return 0;
    }

    const uint64_t weight_bytes = (uint64_t)n_channels * 4ull * sizeof(float);
    if (!g_model_map_ptr || weight_offset > g_model_map_size ||
        weight_bytes > g_model_map_size - weight_offset) {
        fprintf(stderr, "qw3: Metal session DeltaNet conv-step weight is outside mapped model\n");
        return 0;
    }
    uint64_t weight_inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(weight_offset, weight_bytes, &weight_inner);
    if (!wb) return 0;

    struct {
        uint32_t n_channels;
    } args = { n_channels };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_deltanet_conv1d_step_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)weight_inner atIndex:1];
    [enc setBuffer:obj.scratch offset:0 atIndex:2];
    [enc setBuffer:obj.convState offset:(NSUInteger)state_offset atIndex:3];
    [enc setBuffer:obj.qkvConv offset:0 atIndex:4];
    /* Every thread owns one channel and consumes its old three values first. */
    [enc setBuffer:obj.convState offset:(NSUInteger)state_offset atIndex:5];
    NSUInteger threads = g_deltanet_conv1d_step_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake(n_channels, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    id<MTLBuffer> out_readback = nil;
    id<MTLBuffer> state_readback = nil;
    if (out || state_out) {
        qw3_metal_close_batch_encoder();
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        if (!blit) return 0;
        if (out) {
            out_readback = [g_device newBufferWithLength:(NSUInteger)qkv_bytes
                                                 options:MTLResourceStorageModeShared];
            if (!out_readback) return 0;
            [blit copyFromBuffer:obj.qkvConv sourceOffset:0
                        toBuffer:out_readback destinationOffset:0
                            size:(NSUInteger)qkv_bytes];
        }
        if (state_out) {
            state_readback = [g_device newBufferWithLength:(NSUInteger)state_bytes
                                                   options:MTLResourceStorageModeShared];
            if (!state_readback) return 0;
            [blit copyFromBuffer:obj.convState sourceOffset:(NSUInteger)state_offset
                        toBuffer:state_readback destinationOffset:0
                            size:(NSUInteger)state_bytes];
        }
        [blit endEncoding];
    }

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session DeltaNet conv-step command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    if (out) memcpy(out, out_readback.contents, (size_t)qkv_bytes);
    if (state_out) memcpy(state_out, state_readback.contents, (size_t)state_bytes);
    return 1;
}

static int qw3_metal_session_l2norm_one(id<MTLCommandBuffer> cb,
                                        id<MTLBuffer> in,
                                        NSUInteger in_offset,
                                        id<MTLBuffer> out,
                                        uint32_t n_heads,
                                        uint32_t head_dim,
                                        float eps) {
    struct {
        uint32_t head_dim;
        float eps;
    } args = { head_dim, eps };

    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    if (!enc) return 0;
    [enc setComputePipelineState:g_l2norm_heads_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:in offset:in_offset atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_l2norm_heads_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    return 1;
}

int qw3_metal_session_l2norm_qk_from_conv(qw3_metal_session *s,
                                          uint32_t n_heads,
                                          uint32_t head_dim, float eps,
                                          float *q_out, float *k_out) {
    if (!s || !s->obj || n_heads == 0 || head_dim == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint32_t n = n_heads * head_dim;
    const uint64_t qk_bytes = (uint64_t)n * sizeof(float);
    const uint64_t k_offset = qk_bytes;
    if (!obj.qkvConv || !obj.qNorm || !obj.kNorm ||
        obj.qkvConv.length < k_offset + qk_bytes ||
        obj.qNorm.length < qk_bytes || obj.kNorm.length < qk_bytes) {
        return 0;
    }

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    if (!cb) return 0;
    if (!qw3_metal_session_l2norm_one(cb, obj.qkvConv, 0, obj.qNorm,
                                      n_heads, head_dim, eps) ||
        !qw3_metal_session_l2norm_one(cb, obj.qkvConv, (NSUInteger)k_offset,
                                      obj.kNorm, n_heads, head_dim, eps)) {
        return 0;
    }

    id<MTLBuffer> q_readback = nil;
    id<MTLBuffer> k_readback = nil;
    if (q_out || k_out) {
        qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        if (!blit) return 0;
        if (q_out) {
            q_readback = [g_device newBufferWithLength:(NSUInteger)qk_bytes
                                               options:MTLResourceStorageModeShared];
            if (!q_readback) return 0;
            [blit copyFromBuffer:obj.qNorm sourceOffset:0
                        toBuffer:q_readback destinationOffset:0
                            size:(NSUInteger)qk_bytes];
        }
        if (k_out) {
            k_readback = [g_device newBufferWithLength:(NSUInteger)qk_bytes
                                               options:MTLResourceStorageModeShared];
            if (!k_readback) return 0;
            [blit copyFromBuffer:obj.kNorm sourceOffset:0
                        toBuffer:k_readback destinationOffset:0
                            size:(NSUInteger)qk_bytes];
        }
        [blit endEncoding];
    }

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session L2Norm Q/K command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    if (q_out) memcpy(q_out, q_readback.contents, (size_t)qk_bytes);
    if (k_out) memcpy(k_out, k_readback.contents, (size_t)qk_bytes);
    return 1;
}

int qw3_metal_session_matvec_f32_x1_to_scratch(qw3_metal_session *s,
                                               uint64_t tensor_offset,
                                               uint32_t n_in,
                                               uint32_t n_out,
                                               uint32_t out_offset,
                                               float *out) {
    if (!s || !s->obj || n_in == 0 || n_out == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_in * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_out * sizeof(float);
    const uint64_t out_offset_bytes = (uint64_t)out_offset * sizeof(float);
    if (!obj.x1 || !obj.scratch ||
        obj.x1.length < x_bytes ||
        obj.scratch.length < out_offset_bytes ||
        obj.scratch.length - out_offset_bytes < out_bytes) {
        return 0;
    }

    const uint64_t tensor_bytes = (uint64_t)n_in * (uint64_t)n_out * sizeof(float);
    if (!g_model_map_ptr || tensor_offset > g_model_map_size ||
        tensor_bytes > g_model_map_size - tensor_offset) {
        fprintf(stderr, "qw3: Metal session f32 matvec tensor is outside mapped model\n");
        return 0;
    }
    uint64_t tensor_inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(tensor_offset, tensor_bytes, &tensor_inner);
    if (!wb) return 0;

    struct {
        uint32_t n_in;
        uint32_t n_out;
    } args = { n_in, n_out };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    const int use_fast = getenv("QW3_METAL_ROUTER_F32_FAST") != NULL &&
                         n_in == QW3_METAL_N_EMBD && n_out == 256 && out == NULL;
    id<MTLComputePipelineState> pipeline =
        use_fast ? g_matvec_f32_fast_pipeline : g_matvec_f32_pipeline;
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)tensor_inner atIndex:1];
    [enc setBuffer:obj.x1 offset:0 atIndex:2];
    [enc setBuffer:obj.scratch offset:(NSUInteger)out_offset_bytes atIndex:3];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    if (use_fast) {
        [enc dispatchThreadgroups:MTLSizeMake((n_out + 7u) / 8u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
        NSUInteger threads = pipeline.maxTotalThreadsPerThreadgroup;
        if (threads > 256) threads = 256;
        if (threads < 32) threads = 32;
        [enc dispatchThreadgroups:MTLSizeMake(n_out, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    }
    qw3_metal_end_compute_encoder(cb, enc);

    id<MTLBuffer> readback = nil;
    if (out) {
        readback = [g_device newBufferWithLength:(NSUInteger)out_bytes
                                         options:MTLResourceStorageModeShared];
        if (!readback) return 0;
        qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:obj.scratch sourceOffset:(NSUInteger)out_offset_bytes
                    toBuffer:readback destinationOffset:0
                        size:(NSUInteger)out_bytes];
        [blit endEncoding];
    }

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session f32 matvec command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    if (out) memcpy(out, readback.contents, (size_t)out_bytes);
    return 1;
}

int qw3_metal_session_matvec_f32_pair_x1_to_scratch(
    qw3_metal_session *s, uint64_t tensor_a_offset, uint64_t tensor_b_offset,
    uint32_t n_in, uint32_t n_out, uint32_t out_a_offset,
    uint32_t out_b_offset) {
    if (!s || !s->obj || n_in == 0 || n_out == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_in * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_out * sizeof(float);
    const uint64_t out_a_offset_bytes = (uint64_t)out_a_offset * sizeof(float);
    const uint64_t out_b_offset_bytes = (uint64_t)out_b_offset * sizeof(float);
    if (!obj.x1 || !obj.scratch ||
        obj.x1.length < x_bytes ||
        obj.scratch.length < out_a_offset_bytes ||
        obj.scratch.length - out_a_offset_bytes < out_bytes ||
        obj.scratch.length < out_b_offset_bytes ||
        obj.scratch.length - out_b_offset_bytes < out_bytes) {
        return 0;
    }

    const uint64_t tensor_bytes =
        (uint64_t)n_in * (uint64_t)n_out * sizeof(float);
    uint64_t tensor_a_inner = 0, tensor_b_inner = 0;
    id<MTLBuffer> wa =
        qw3_metal_model_view_for(tensor_a_offset, tensor_bytes, &tensor_a_inner);
    id<MTLBuffer> wb =
        qw3_metal_model_view_for(tensor_b_offset, tensor_bytes, &tensor_b_inner);
    if (!wa || !wb) {
        fprintf(stderr, "qw3: Metal session f32 pair tensor is outside mapped model views\n");
        return 0;
    }

    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t out_a_offset;
        uint32_t out_b_offset;
    } args = { n_in, n_out, out_a_offset, out_b_offset };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_matvec_f32_pair_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wa offset:(NSUInteger)tensor_a_inner atIndex:1];
    [enc setBuffer:wb offset:(NSUInteger)tensor_b_inner atIndex:2];
    [enc setBuffer:obj.x1 offset:0 atIndex:3];
    [enc setBuffer:obj.scratch offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:64 * sizeof(float) atIndex:0];
    NSUInteger threads = g_matvec_f32_pair_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(n_out, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session f32 pair command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_router_topk_from_scratch(qw3_metal_session *s,
                                               uint32_t router_offset,
                                               uint32_t n_router,
                                               uint32_t n_top,
                                               int *ids_out,
                                               float *weights_out) {
    if (!s || !s->obj || !ids_out || !weights_out ||
        n_router != 256 || n_top != 8) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t router_offset_bytes = (uint64_t)router_offset * sizeof(float);
    if (!obj.scratch || !obj.routerIds || !obj.routerWeights ||
        obj.scratch.length < router_offset_bytes ||
        obj.scratch.length - router_offset_bytes < (uint64_t)n_router * sizeof(float) ||
        obj.routerIds.length < (uint64_t)n_top * sizeof(int32_t) ||
        obj.routerWeights.length < (uint64_t)n_top * sizeof(float)) {
        return 0;
    }

    id<MTLBuffer> ids_readback = [g_device newBufferWithLength:(NSUInteger)n_top * sizeof(int32_t)
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> weights_readback = [g_device newBufferWithLength:(NSUInteger)n_top * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    if (!ids_readback || !weights_readback) return 0;

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    if (!cb) return 0;
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_router_top8_pipeline];
    [enc setBuffer:obj.scratch offset:(NSUInteger)router_offset_bytes atIndex:0];
    [enc setBuffer:obj.routerIds offset:0 atIndex:1];
    [enc setBuffer:obj.routerWeights offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(256, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!blit) return 0;
    [blit copyFromBuffer:obj.routerIds sourceOffset:0
                toBuffer:ids_readback destinationOffset:0
                    size:(NSUInteger)n_top * sizeof(int32_t)];
    [blit copyFromBuffer:obj.routerWeights sourceOffset:0
                toBuffer:weights_readback destinationOffset:0
                    size:(NSUInteger)n_top * sizeof(float)];
    [blit endEncoding];

    int ok = owned ? qw3_metal_finish_command_buffer(cb, owned, "operation")
                   : qw3_metal_end_commands();
    if (!ok) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session router top-k command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(ids_out, ids_readback.contents, (size_t)n_top * sizeof(int32_t));
    memcpy(weights_out, weights_readback.contents, (size_t)n_top * sizeof(float));
    return 1;
}

int qw3_metal_session_deltanet_recur_zero_from_buffers(qw3_metal_session *s,
                                                       const float *beta,
                                                       uint32_t q_heads,
                                                       uint32_t v_heads,
                                                       uint32_t head_dim,
                                                       float *state_out,
                                                       float *core_out) {
    if (!s || !s->obj || !beta || q_heads == 0 || v_heads == 0 ||
        head_dim == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t qk_bytes = (uint64_t)q_heads * head_dim * sizeof(float);
    const uint64_t v_bytes = (uint64_t)v_heads * head_dim * sizeof(float);
    const uint64_t state_bytes = (uint64_t)v_heads * head_dim * head_dim * sizeof(float);
    const uint64_t v_offset = 2 * qk_bytes;
    if (!obj.qNorm || !obj.kNorm || !obj.qkvConv || !obj.deltanetState ||
        !obj.core || obj.qNorm.length < qk_bytes ||
        obj.kNorm.length < qk_bytes || obj.qkvConv.length < v_offset + v_bytes ||
        obj.deltanetState.length < state_bytes || obj.core.length < v_bytes) {
        return 0;
    }

    id<MTLBuffer> bb = [g_device newBufferWithBytes:beta
                                             length:(NSUInteger)v_heads * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    if (!bb) return 0;
    struct {
        uint32_t q_heads;
        uint32_t v_heads;
        uint32_t head_dim;
    } args = { q_heads, v_heads, head_dim };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_deltanet_recur_zero_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.qNorm offset:0 atIndex:1];
    [enc setBuffer:obj.kNorm offset:0 atIndex:2];
    [enc setBuffer:obj.qkvConv offset:(NSUInteger)v_offset atIndex:3];
    [enc setBuffer:bb offset:0 atIndex:4];
    [enc setBuffer:obj.deltanetState offset:0 atIndex:5];
    [enc setBuffer:obj.core offset:0 atIndex:6];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_deltanet_recur_zero_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(v_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    id<MTLBuffer> state_readback = nil;
    id<MTLBuffer> core_readback = nil;
    if (state_out || core_out) {
        qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        if (!blit) return 0;
        if (state_out) {
            state_readback = [g_device newBufferWithLength:(NSUInteger)state_bytes
                                                   options:MTLResourceStorageModeShared];
            if (!state_readback) return 0;
            [blit copyFromBuffer:obj.deltanetState sourceOffset:0
                        toBuffer:state_readback destinationOffset:0
                            size:(NSUInteger)state_bytes];
        }
        if (core_out) {
            core_readback = [g_device newBufferWithLength:(NSUInteger)v_bytes
                                                  options:MTLResourceStorageModeShared];
            if (!core_readback) return 0;
            [blit copyFromBuffer:obj.core sourceOffset:0
                        toBuffer:core_readback destinationOffset:0
                            size:(NSUInteger)v_bytes];
        }
        [blit endEncoding];
    }

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session DeltaNet recurrence command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    if (state_out) memcpy(state_out, state_readback.contents, (size_t)state_bytes);
    if (core_out) memcpy(core_out, core_readback.contents, (size_t)v_bytes);
    return 1;
}

int qw3_metal_session_deltanet_recur_from_buffers(qw3_metal_session *s,
                                                  const float *beta,
                                                  const float *gamma,
                                                  uint32_t layer_slot,
                                                  uint32_t q_heads,
                                                  uint32_t v_heads,
                                                  uint32_t head_dim,
                                                  float *state_out,
                                                  float *core_out) {
    if (!s || !s->obj || !beta || !gamma || q_heads == 0 || v_heads == 0 ||
        head_dim == 0 || layer_slot >= QW3_METAL_N_LINEAR_LAYERS) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t qk_bytes = (uint64_t)q_heads * head_dim * sizeof(float);
    const uint64_t v_bytes = (uint64_t)v_heads * head_dim * sizeof(float);
    const uint64_t gates_bytes = (uint64_t)v_heads * sizeof(float);
    const uint64_t state_bytes = (uint64_t)v_heads * head_dim * head_dim * sizeof(float);
    const uint64_t state_offset = (uint64_t)layer_slot * state_bytes;
    const uint64_t v_offset = 2 * qk_bytes;
    if (!obj.qNorm || !obj.kNorm || !obj.qkvConv || !obj.deltanetState ||
        !obj.core || obj.qNorm.length < qk_bytes ||
        obj.kNorm.length < qk_bytes || obj.qkvConv.length < v_offset + v_bytes ||
        obj.deltanetState.length < state_offset ||
        obj.deltanetState.length - state_offset < state_bytes ||
        obj.core.length < v_bytes) {
        return 0;
    }

    id<MTLBuffer> bb = [g_device newBufferWithBytes:beta
                                             length:(NSUInteger)gates_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> gb = [g_device newBufferWithBytes:gamma
                                             length:(NSUInteger)gates_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> next_state = [g_device newBufferWithLength:(NSUInteger)state_bytes
                                                     options:MTLResourceStorageModePrivate];
    if (!bb || !gb || !next_state) return 0;

    struct {
        uint32_t q_heads;
        uint32_t v_heads;
        uint32_t head_dim;
    } args = { q_heads, v_heads, head_dim };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_deltanet_recur_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.deltanetState offset:(NSUInteger)state_offset atIndex:1];
    [enc setBuffer:obj.qNorm offset:0 atIndex:2];
    [enc setBuffer:obj.kNorm offset:0 atIndex:3];
    [enc setBuffer:obj.qkvConv offset:(NSUInteger)v_offset atIndex:4];
    [enc setBuffer:bb offset:0 atIndex:5];
    [enc setBuffer:gb offset:0 atIndex:6];
    [enc setBuffer:next_state offset:0 atIndex:7];
    [enc setBuffer:obj.core offset:0 atIndex:8];
    NSUInteger threads = g_deltanet_recur_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 128) threads = 128;
    [enc dispatchThreads:MTLSizeMake(head_dim, v_heads, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!blit) return 0;
    [blit copyFromBuffer:next_state sourceOffset:0
                toBuffer:obj.deltanetState destinationOffset:(NSUInteger)state_offset
                    size:(NSUInteger)state_bytes];
    id<MTLBuffer> state_readback = nil;
    id<MTLBuffer> core_readback = nil;
    if (state_out) {
        state_readback = [g_device newBufferWithLength:(NSUInteger)state_bytes
                                               options:MTLResourceStorageModeShared];
        if (!state_readback) return 0;
        [blit copyFromBuffer:next_state sourceOffset:0
                    toBuffer:state_readback destinationOffset:0
                        size:(NSUInteger)state_bytes];
    }
    if (core_out) {
        core_readback = [g_device newBufferWithLength:(NSUInteger)v_bytes
                                              options:MTLResourceStorageModeShared];
        if (!core_readback) return 0;
        [blit copyFromBuffer:obj.core sourceOffset:0
                    toBuffer:core_readback destinationOffset:0
                        size:(NSUInteger)v_bytes];
    }
    [blit endEncoding];

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session DeltaNet recurrent-step command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    if (state_out) memcpy(state_out, state_readback.contents, (size_t)state_bytes);
    if (core_out) memcpy(core_out, core_readback.contents, (size_t)v_bytes);
    return 1;
}

int qw3_metal_session_deltanet_recur_from_scratch_gates(qw3_metal_session *s,
                                                        uint64_t dt_bias_offset,
                                                        uint64_t a_offset,
                                                        uint32_t alpha_offset,
                                                        uint32_t beta_offset,
                                                        uint32_t layer_slot,
                                                        uint32_t q_heads,
                                                        uint32_t v_heads,
                                                        uint32_t head_dim,
                                                        float *state_out,
                                                        float *core_out) {
    if (!s || !s->obj || q_heads == 0 || v_heads == 0 ||
        head_dim == 0 || layer_slot >= QW3_METAL_N_LINEAR_LAYERS) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t qk_bytes = (uint64_t)q_heads * head_dim * sizeof(float);
    const uint64_t v_bytes = (uint64_t)v_heads * head_dim * sizeof(float);
    const uint64_t gates_bytes = (uint64_t)v_heads * sizeof(float);
    const uint64_t state_bytes = (uint64_t)v_heads * head_dim * head_dim * sizeof(float);
    const uint64_t state_offset = (uint64_t)layer_slot * state_bytes;
    const uint64_t v_offset = 2 * qk_bytes;
    const uint64_t alpha_bytes = (uint64_t)alpha_offset * sizeof(float);
    const uint64_t beta_bytes = (uint64_t)beta_offset * sizeof(float);
    if (!obj.qNorm || !obj.kNorm || !obj.qkvConv || !obj.deltanetState ||
        !obj.scratch || !obj.core || obj.qNorm.length < qk_bytes ||
        obj.kNorm.length < qk_bytes || obj.qkvConv.length < v_offset + v_bytes ||
        obj.deltanetState.length < state_offset ||
        obj.deltanetState.length - state_offset < state_bytes ||
        obj.scratch.length < alpha_bytes + gates_bytes ||
        obj.scratch.length < beta_bytes + gates_bytes ||
        obj.core.length < v_bytes) {
        return 0;
    }
    if (!g_model_map_ptr || dt_bias_offset > g_model_map_size ||
        a_offset > g_model_map_size ||
        gates_bytes > g_model_map_size - dt_bias_offset ||
        gates_bytes > g_model_map_size - a_offset) {
        fprintf(stderr, "qw3: Metal session DeltaNet gate weights are outside mapped model\n");
        return 0;
    }

    uint64_t dt_inner = 0;
    uint64_t a_inner = 0;
    id<MTLBuffer> dtb = qw3_metal_model_view_for(dt_bias_offset, gates_bytes, &dt_inner);
    id<MTLBuffer> ab = qw3_metal_model_view_for(a_offset, gates_bytes, &a_inner);
    id<MTLBuffer> next_state = [g_device newBufferWithLength:(NSUInteger)state_bytes
                                                     options:MTLResourceStorageModePrivate];
    if (!dtb || !ab || !next_state) return 0;

    struct {
        uint32_t q_heads;
        uint32_t v_heads;
        uint32_t head_dim;
        uint32_t alpha_offset;
        uint32_t beta_offset;
    } args = { q_heads, v_heads, head_dim, alpha_offset, beta_offset };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_deltanet_recur_scratch_gates_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.deltanetState offset:(NSUInteger)state_offset atIndex:1];
    [enc setBuffer:obj.qNorm offset:0 atIndex:2];
    [enc setBuffer:obj.kNorm offset:0 atIndex:3];
    [enc setBuffer:obj.qkvConv offset:(NSUInteger)v_offset atIndex:4];
    [enc setBuffer:obj.scratch offset:0 atIndex:5];
    [enc setBuffer:dtb offset:(NSUInteger)dt_inner atIndex:6];
    [enc setBuffer:ab offset:(NSUInteger)a_inner atIndex:7];
    [enc setBuffer:next_state offset:0 atIndex:8];
    [enc setBuffer:obj.core offset:0 atIndex:9];
    NSUInteger threads = g_deltanet_recur_scratch_gates_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 128) threads = 128;
    [enc dispatchThreads:MTLSizeMake(head_dim, v_heads, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!blit) return 0;
    [blit copyFromBuffer:next_state sourceOffset:0
                toBuffer:obj.deltanetState destinationOffset:(NSUInteger)state_offset
                    size:(NSUInteger)state_bytes];
    id<MTLBuffer> state_readback = nil;
    id<MTLBuffer> core_readback = nil;
    if (state_out) {
        state_readback = [g_device newBufferWithLength:(NSUInteger)state_bytes
                                               options:MTLResourceStorageModeShared];
        if (!state_readback) return 0;
        [blit copyFromBuffer:next_state sourceOffset:0
                    toBuffer:state_readback destinationOffset:0
                        size:(NSUInteger)state_bytes];
    }
    if (core_out) {
        core_readback = [g_device newBufferWithLength:(NSUInteger)v_bytes
                                              options:MTLResourceStorageModeShared];
        if (!core_readback) return 0;
        [blit copyFromBuffer:obj.core sourceOffset:0
                    toBuffer:core_readback destinationOffset:0
                        size:(NSUInteger)v_bytes];
    }
    [blit endEncoding];

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session DeltaNet scratch-gate recurrent command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    if (state_out) memcpy(state_out, state_readback.contents, (size_t)state_bytes);
    if (core_out) memcpy(core_out, core_readback.contents, (size_t)v_bytes);
    return 1;
}

int qw3_metal_session_deltanet_fused_gdn_from_scratch(
    qw3_metal_session *s, uint64_t dt_bias_offset, uint64_t a_offset,
    uint64_t norm_weight_offset, uint32_t z_offset, uint32_t alpha_offset,
    uint32_t beta_offset, uint32_t layer_slot, uint32_t q_heads,
    uint32_t v_heads, uint32_t head_dim, float eps) {
    if (!s || !s->obj || q_heads == 0 || v_heads == 0 || head_dim == 0 ||
        layer_slot >= QW3_METAL_N_LINEAR_LAYERS) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t qk_bytes = (uint64_t)q_heads * head_dim * sizeof(float);
    const uint64_t inner_bytes = (uint64_t)v_heads * head_dim * sizeof(float);
    const uint64_t gates_bytes = (uint64_t)v_heads * sizeof(float);
    const uint64_t state_bytes =
        (uint64_t)v_heads * head_dim * head_dim * sizeof(float);
    const uint64_t state_offset = (uint64_t)layer_slot * state_bytes;
    const uint64_t v_offset = 2 * qk_bytes;
    const uint64_t z_bytes = (uint64_t)z_offset * sizeof(float);
    const uint64_t alpha_bytes = (uint64_t)alpha_offset * sizeof(float);
    const uint64_t beta_bytes = (uint64_t)beta_offset * sizeof(float);
    if (!obj.qNorm || !obj.kNorm || !obj.qkvConv || !obj.deltanetState ||
        !obj.scratch || !obj.inner || obj.qNorm.length < qk_bytes ||
        obj.kNorm.length < qk_bytes ||
        obj.qkvConv.length < v_offset + inner_bytes ||
        obj.deltanetState.length < state_offset ||
        obj.deltanetState.length - state_offset < state_bytes ||
        obj.scratch.length < z_bytes + inner_bytes ||
        obj.scratch.length < alpha_bytes + gates_bytes ||
        obj.scratch.length < beta_bytes + gates_bytes ||
        obj.inner.length < inner_bytes) {
        return 0;
    }

    const uint64_t weight_bytes = (uint64_t)head_dim * sizeof(float);
    if (!g_model_map_ptr || dt_bias_offset > g_model_map_size ||
        a_offset > g_model_map_size || norm_weight_offset > g_model_map_size ||
        gates_bytes > g_model_map_size - dt_bias_offset ||
        gates_bytes > g_model_map_size - a_offset ||
        weight_bytes > g_model_map_size - norm_weight_offset) {
        fprintf(stderr, "qw3: Metal fused Gated DeltaNet weights are outside mapped model\n");
        return 0;
    }

    uint64_t dt_inner = 0;
    uint64_t a_inner = 0;
    uint64_t weight_inner = 0;
    id<MTLBuffer> dtb = qw3_metal_model_view_for(dt_bias_offset, gates_bytes,
                                                  &dt_inner);
    id<MTLBuffer> ab = qw3_metal_model_view_for(a_offset, gates_bytes, &a_inner);
    id<MTLBuffer> wb = qw3_metal_model_view_for(norm_weight_offset, weight_bytes,
                                                &weight_inner);
    if (!dtb || !ab || !wb ||
        head_dim > g_deltanet_fused_gdn_scratch_pipeline.maxTotalThreadsPerThreadgroup) {
        return 0;
    }

    struct {
        uint32_t q_heads;
        uint32_t v_heads;
        uint32_t head_dim;
        uint32_t alpha_offset;
        uint32_t beta_offset;
        uint32_t z_offset;
        float eps;
    } args = { q_heads, v_heads, head_dim, alpha_offset, beta_offset,
               z_offset, eps };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_deltanet_fused_gdn_scratch_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.deltanetState offset:(NSUInteger)state_offset atIndex:1];
    [enc setBuffer:obj.qNorm offset:0 atIndex:2];
    [enc setBuffer:obj.kNorm offset:0 atIndex:3];
    [enc setBuffer:obj.qkvConv offset:(NSUInteger)v_offset atIndex:4];
    [enc setBuffer:obj.scratch offset:0 atIndex:5];
    [enc setBuffer:dtb offset:(NSUInteger)dt_inner atIndex:6];
    [enc setBuffer:ab offset:(NSUInteger)a_inner atIndex:7];
    [enc setBuffer:wb offset:(NSUInteger)weight_inner atIndex:8];
    /* Columns are independent in autoregressive decode, so state can update in place. */
    [enc setBuffer:obj.deltanetState offset:(NSUInteger)state_offset atIndex:9];
    [enc setBuffer:obj.inner offset:0 atIndex:10];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(v_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal fused Gated DeltaNet command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_deltanet_gated_rmsnorm_from_buffers(qw3_metal_session *s,
                                                          uint64_t norm_weight_offset,
                                                          uint32_t z_offset,
                                                          uint32_t v_heads,
                                                          uint32_t head_dim,
                                                          float eps,
                                                          float *out) {
    if (!s || !s->obj || v_heads == 0 || head_dim == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t bytes = (uint64_t)v_heads * head_dim * sizeof(float);
    const uint64_t z_offset_bytes = (uint64_t)z_offset * sizeof(float);
    if (!obj.core || !obj.scratch || !obj.inner ||
        obj.core.length < bytes || obj.inner.length < bytes ||
        obj.scratch.length < z_offset_bytes ||
        obj.scratch.length - z_offset_bytes < bytes) {
        return 0;
    }

    const uint64_t weight_bytes = (uint64_t)head_dim * sizeof(float);
    if (!g_model_map_ptr || norm_weight_offset > g_model_map_size ||
        weight_bytes > g_model_map_size - norm_weight_offset) {
        fprintf(stderr, "qw3: Metal session DeltaNet norm weight is outside mapped model\n");
        return 0;
    }
    uint64_t weight_inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(norm_weight_offset, weight_bytes, &weight_inner);
    if (!wb) return 0;
    struct {
        uint32_t v_heads;
        uint32_t head_dim;
        float eps;
    } args = { v_heads, head_dim, eps };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_deltanet_gated_rmsnorm_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)weight_inner atIndex:1];
    [enc setBuffer:obj.core offset:0 atIndex:2];
    [enc setBuffer:obj.scratch offset:(NSUInteger)z_offset_bytes atIndex:3];
    [enc setBuffer:obj.inner offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_deltanet_gated_rmsnorm_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(v_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    id<MTLBuffer> readback = nil;
    if (out) {
        readback = [g_device newBufferWithLength:(NSUInteger)bytes
                                         options:MTLResourceStorageModeShared];
        if (!readback) return 0;
        qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:obj.inner sourceOffset:0
                    toBuffer:readback destinationOffset:0
                        size:(NSUInteger)bytes];
        [blit endEncoding];
    }

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session DeltaNet gated RMSNorm command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    if (out) memcpy(out, readback.contents, (size_t)bytes);
    return 1;
}

int qw3_metal_session_matvec_q8_0_inner_to_x1(qw3_metal_session *s,
                                              uint64_t tensor_offset,
                                              uint32_t n_in,
                                              uint32_t n_out,
                                              float *out) {
    if (!s || !s->obj || n_in == 0 || n_out == 0 || (n_in % 32) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_in * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_out * sizeof(float);
    if (!obj.inner || !obj.x1 ||
        obj.inner.length < x_bytes || obj.x1.length < out_bytes) {
        return 0;
    }

    const uint64_t row_bytes = (uint64_t)(n_in / 32) * 34ull;
    const uint64_t tensor_bytes = row_bytes * (uint64_t)n_out;
    uint64_t inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(tensor_offset, tensor_bytes, &inner);
    if (!wb) {
        fprintf(stderr, "qw3: Metal session inner q8_0 matvec tensor is outside mapped model views\n");
        return 0;
    }

    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
    } args = { n_in, n_out, (uint32_t)row_bytes };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_matvec_q8_0_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)inner atIndex:1];
    [enc setBuffer:obj.inner offset:0 atIndex:2];
    [enc setBuffer:obj.x1 offset:0 atIndex:3];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_matvec_q8_0_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(n_out, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    id<MTLBuffer> readback = nil;
    if (out) {
        readback = [g_device newBufferWithLength:(NSUInteger)out_bytes
                                         options:MTLResourceStorageModeShared];
        if (!readback) return 0;
        qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:obj.x1 sourceOffset:0
                    toBuffer:readback destinationOffset:0
                        size:(NSUInteger)out_bytes];
        [blit endEncoding];
    }

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session inner q8_0 matvec command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    if (out) memcpy(out, readback.contents, (size_t)out_bytes);
    return 1;
}

int qw3_metal_session_matvec_q8_0_inner_scale_add_x0(qw3_metal_session *s,
                                                     uint64_t tensor_offset,
                                                     uint32_t n_in,
                                                     uint32_t n_out,
                                                     uint32_t scalar_offset) {
    if (!s || !s->obj || n_in == 0 || n_out == 0 || (n_in % 32) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_in * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_out * sizeof(float);
    const uint64_t scalar_bytes = (uint64_t)scalar_offset * sizeof(float);
    if (!obj.inner || !obj.x0 || !obj.scratch ||
        obj.inner.length < x_bytes || obj.x0.length < out_bytes ||
        obj.scratch.length < scalar_bytes + sizeof(float)) {
        return 0;
    }

    const uint64_t row_bytes = (uint64_t)(n_in / 32) * 34ull;
    const uint64_t tensor_bytes = row_bytes * (uint64_t)n_out;
    uint64_t inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(tensor_offset, tensor_bytes, &inner);
    if (!wb) {
        fprintf(stderr, "qw3: Metal session shared q8_0 add tensor is outside mapped model views\n");
        return 0;
    }

    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
        uint32_t scalar_offset;
    } args = { n_in, n_out, (uint32_t)row_bytes, scalar_offset };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_matvec_q8_0_inner_scale_add_x0_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)inner atIndex:1];
    [enc setBuffer:obj.inner offset:0 atIndex:2];
    [enc setBuffer:obj.scratch offset:0 atIndex:3];
    [enc setBuffer:obj.x0 offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((n_out + 7u) / 8u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session shared q8_0 add command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_matvec_iq3_s_expert_x1_to_scratch(qw3_metal_session *s,
                                                        uint64_t tensor_offset,
                                                        uint32_t expert,
                                                        uint32_t n_in,
                                                        uint32_t n_out,
                                                        uint32_t out_offset) {
    if (!s || !s->obj || n_in == 0 || n_out == 0 || (n_in % 256) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_in * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_out * sizeof(float);
    const uint64_t out_offset_bytes = (uint64_t)out_offset * sizeof(float);
    if (!obj.x1 || !obj.scratch || obj.x1.length < x_bytes ||
        obj.scratch.length < out_offset_bytes ||
        obj.scratch.length - out_offset_bytes < out_bytes) {
        return 0;
    }

    const uint64_t row_bytes = (uint64_t)(n_in / 256) * 110ull;
    const uint64_t expert_bytes = row_bytes * (uint64_t)n_out;
    const uint64_t expert_offset = tensor_offset + (uint64_t)expert * expert_bytes;
    uint64_t inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(expert_offset, expert_bytes, &inner);
    id<MTLBuffer> kgb = qw3_metal_iq3s_kgrid_buffer();
    if (!wb || !kgb) return 0;
    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
    } args = { n_in, n_out, (uint32_t)row_bytes };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_matvec_iq3_s_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)inner atIndex:1];
    [enc setBuffer:obj.x1 offset:0 atIndex:2];
    [enc setBuffer:obj.scratch offset:(NSUInteger)out_offset_bytes atIndex:3];
    [enc setBuffer:kgb offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_matvec_iq3_s_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(n_out, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session iq3_s expert matvec command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

static int qw3_metal_session_matvec_k_expert_inner_to_x1(qw3_metal_session *s,
                                                         uint64_t tensor_offset,
                                                         uint32_t expert,
                                                         uint32_t n_in,
                                                         uint32_t n_out,
                                                         uint64_t row_bytes,
                                                         id<MTLComputePipelineState> pipeline,
                                                         const char *name) {
    if (!s || !s->obj || n_in == 0 || n_out == 0 || (n_in % 256) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_in * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_out * sizeof(float);
    if (!obj.inner || !obj.x1 ||
        obj.inner.length < x_bytes || obj.x1.length < out_bytes) {
        return 0;
    }
    const uint64_t expert_bytes = row_bytes * (uint64_t)n_out;
    const uint64_t expert_offset = tensor_offset + (uint64_t)expert * expert_bytes;
    uint64_t inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(expert_offset, expert_bytes, &inner);
    if (!wb) return 0;
    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
    } args = { n_in, n_out, (uint32_t)row_bytes };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)inner atIndex:1];
    [enc setBuffer:obj.inner offset:0 atIndex:2];
    [enc setBuffer:obj.x1 offset:0 atIndex:3];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(n_out, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session %s expert matvec command failed: %s\n",
                name, [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_matvec_iq4_xs_expert_inner_to_x1(qw3_metal_session *s,
                                                       uint64_t tensor_offset,
                                                       uint32_t expert,
                                                       uint32_t n_in,
                                                       uint32_t n_out) {
    const uint64_t row_bytes = (uint64_t)(n_in / 256) * 136ull;
    return qw3_metal_session_matvec_k_expert_inner_to_x1(
        s, tensor_offset, expert, n_in, n_out, row_bytes,
        g_matvec_iq4_xs_pipeline, "iq4_xs");
}

int qw3_metal_session_matvec_q6_k_expert_inner_to_x1(qw3_metal_session *s,
                                                     uint64_t tensor_offset,
                                                     uint32_t expert,
                                                     uint32_t n_in,
                                                     uint32_t n_out) {
    const uint64_t row_bytes = (uint64_t)(n_in / 256) * 210ull;
    return qw3_metal_session_matvec_k_expert_inner_to_x1(
        s, tensor_offset, expert, n_in, n_out, row_bytes,
        g_matvec_q6_k_pipeline, "q6_K");
}

static int qw3_metal_session_matvec_k_expert_inner_to_scratch(qw3_metal_session *s,
                                                              uint64_t tensor_offset,
                                                              uint32_t expert,
                                                              uint32_t n_in,
                                                              uint32_t n_out,
                                                              uint32_t out_offset,
                                                              uint64_t row_bytes,
                                                              id<MTLComputePipelineState> pipeline,
                                                              const char *name) {
    if (!s || !s->obj || n_in == 0 || n_out == 0 || (n_in % 256) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_in * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_out * sizeof(float);
    const uint64_t out_offset_bytes = (uint64_t)out_offset * sizeof(float);
    if (!obj.inner || !obj.scratch ||
        obj.inner.length < x_bytes ||
        obj.scratch.length < out_offset_bytes ||
        obj.scratch.length - out_offset_bytes < out_bytes) {
        return 0;
    }
    const uint64_t expert_bytes = row_bytes * (uint64_t)n_out;
    const uint64_t expert_offset = tensor_offset + (uint64_t)expert * expert_bytes;
    uint64_t inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(expert_offset, expert_bytes, &inner);
    if (!wb) return 0;
    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
    } args = { n_in, n_out, (uint32_t)row_bytes };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)inner atIndex:1];
    [enc setBuffer:obj.inner offset:0 atIndex:2];
    [enc setBuffer:obj.scratch offset:(NSUInteger)out_offset_bytes atIndex:3];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(n_out, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session %s expert scratch matvec command failed: %s\n",
                name, [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_matvec_iq4_xs_expert_inner_to_scratch(qw3_metal_session *s,
                                                            uint64_t tensor_offset,
                                                            uint32_t expert,
                                                            uint32_t n_in,
                                                            uint32_t n_out,
                                                            uint32_t out_offset) {
    const uint64_t row_bytes = (uint64_t)(n_in / 256) * 136ull;
    return qw3_metal_session_matvec_k_expert_inner_to_scratch(
        s, tensor_offset, expert, n_in, n_out, out_offset, row_bytes,
        g_matvec_iq4_xs_pipeline, "iq4_xs");
}

int qw3_metal_session_matvec_q6_k_expert_inner_to_scratch(qw3_metal_session *s,
                                                          uint64_t tensor_offset,
                                                          uint32_t expert,
                                                          uint32_t n_in,
                                                          uint32_t n_out,
                                                          uint32_t out_offset) {
    const uint64_t row_bytes = (uint64_t)(n_in / 256) * 210ull;
    return qw3_metal_session_matvec_k_expert_inner_to_scratch(
        s, tensor_offset, expert, n_in, n_out, out_offset, row_bytes,
        g_matvec_q6_k_pipeline, "q6_K");
}

static int qw3_metal_session_sparse_moe_topk_batch(qw3_metal_session *s,
                                                   uint64_t gate_offset,
                                                   uint64_t up_offset,
                                                   uint64_t down_offset,
                                                   uint32_t down_type,
                                                   const int *ids,
                                                   const float *weights,
                                                   id<MTLBuffer> ids_buffer,
                                                   id<MTLBuffer> weights_buffer,
                                                   uint32_t n_active,
                                                   uint32_t n_embd,
                                                   uint32_t n_ff) {
    if (!s || !s->obj || (!ids && !ids_buffer) ||
        (!weights && !weights_buffer) || n_active == 0 || n_active > 8 ||
        n_embd == 0 || n_ff == 0 || (n_embd % 256) != 0 ||
        (n_ff % 256) != 0) {
        return 0;
    }
    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t embd_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t iq3_row_bytes = (uint64_t)(n_embd / 256) * 110ull;
    const uint64_t iq3_expert_bytes = iq3_row_bytes * (uint64_t)n_ff;
    const uint64_t iq4_row_bytes = (uint64_t)(n_ff / 256) * 136ull;
    const uint64_t q6_row_bytes = (uint64_t)(n_ff / 256) * 210ull;
    const uint64_t down_row_bytes = down_type == 23 ? iq4_row_bytes :
                                    down_type == 14 ? q6_row_bytes : 0;
    id<MTLComputePipelineState> down_batch_pipeline =
        down_type == 23 ? g_moe_down_iq4_xs_batch_pipeline :
        down_type == 14 ? g_moe_down_q6_k_batch_pipeline : nil;
    if (!down_row_bytes || !down_batch_pipeline) return 0;

    const uint64_t down_expert_bytes = down_row_bytes * (uint64_t)n_embd;
    const uint32_t gateup_base = 0;
    const uint32_t hidden_base = 2u * n_ff * n_active;
    const uint32_t down_base = hidden_base + n_ff * n_active;
    const uint64_t scratch_floats =
        (uint64_t)down_base + (uint64_t)n_active * (uint64_t)n_embd;
    const uint64_t scratch_bytes = scratch_floats * sizeof(float);
    if (!obj.x0 || !obj.x1 || !obj.scratch ||
        obj.x0.length < embd_bytes || obj.x1.length < embd_bytes ||
        obj.scratch.length < scratch_bytes) {
        return 0;
    }

    const uint64_t gate_tensor_bytes = iq3_expert_bytes * 256ull;
    const uint64_t up_tensor_bytes = iq3_expert_bytes * 256ull;
    const uint64_t down_tensor_bytes = down_expert_bytes * 256ull;
    uint64_t gate_inner = 0, up_inner = 0, down_inner = 0;
    id<MTLBuffer> gate_w = qw3_metal_model_view_for(
        gate_offset, gate_tensor_bytes, &gate_inner);
    id<MTLBuffer> up_w = qw3_metal_model_view_for(
        up_offset, up_tensor_bytes, &up_inner);
    id<MTLBuffer> down_w = qw3_metal_model_view_for(
        down_offset, down_tensor_bytes, &down_inner);
    if (!gate_w) {
        gate_w = qw3_metal_model_temp_buffer_for(
            gate_offset, gate_tensor_bytes, &gate_inner);
    }
    if (!up_w) {
        up_w = qw3_metal_model_temp_buffer_for(
            up_offset, up_tensor_bytes, &up_inner);
    }
    if (!down_w) {
        down_w = qw3_metal_model_temp_buffer_for(
            down_offset, down_tensor_bytes, &down_inner);
    }
    id<MTLBuffer> kgb = qw3_metal_iq3s_kgrid_buffer();
    if (!gate_w || !up_w || !down_w || !kgb) return 0;

    struct {
        uint32_t n_in;
        uint32_t n_ff;
        uint32_t n_embd;
        uint32_t n_active;
        uint32_t iq3_row_bytes;
        uint32_t iq3_expert_bytes;
        uint32_t down_row_bytes;
        uint32_t down_expert_bytes;
        uint32_t gateup_base;
        uint32_t hidden_base;
        uint32_t down_base;
    } args = {
        n_embd, n_ff, n_embd, n_active,
        (uint32_t)iq3_row_bytes, (uint32_t)iq3_expert_bytes,
        (uint32_t)down_row_bytes, (uint32_t)down_expert_bytes,
        gateup_base, hidden_base, down_base
    };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    if (!cb) return 0;
    const int profile_moe_sync =
        getenv("QW3_METAL_PROFILE_MOE_SYNC") != NULL && g_batch_cb != nil;
    double t_moe_sync = profile_moe_sync ? [NSDate timeIntervalSinceReferenceDate] : 0.0;

    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_moe_iq3_s_pair_batch_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:gate_w offset:(NSUInteger)gate_inner atIndex:1];
    [enc setBuffer:up_w offset:(NSUInteger)up_inner atIndex:2];
    [enc setBuffer:obj.x1 offset:0 atIndex:3];
    [enc setBuffer:obj.scratch offset:0 atIndex:4];
    [enc setBuffer:kgb offset:0 atIndex:5];
    if (ids_buffer) {
        [enc setBuffer:ids_buffer offset:0 atIndex:6];
    } else {
        [enc setBytes:ids length:(NSUInteger)n_active * sizeof(int) atIndex:6];
    }
    [enc setThreadgroupMemoryLength:64 * sizeof(float) atIndex:0];
    NSUInteger threads = 64;
    [enc dispatchThreadgroups:MTLSizeMake(((n_ff + 7u) / 8u) * n_active, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (profile_moe_sync) {
        if (!qw3_metal_synchronize()) return 0;
        fprintf(stderr, "qw3 metal moe profile down_type=%u gateup_swiglu_ms=%.3f\n",
                down_type, ([NSDate timeIntervalSinceReferenceDate] - t_moe_sync) * 1000.0);
        if (!qw3_metal_begin_commands()) return 0;
        cb = qw3_metal_command_buffer(&owned);
        if (!cb) return 0;
        t_moe_sync = [NSDate timeIntervalSinceReferenceDate];
    }

    enc = qw3_metal_compute_encoder(cb);
    const int fuse_down_reduce =
        down_type == 23 && getenv("QW3_METAL_FUSED_DOWN_REDUCE") != NULL;
    const int pair_down_reduce =
        down_type == 23 && !fuse_down_reduce &&
        getenv("QW3_METAL_LEGACY_PAIR_DOWN") == NULL;
    [enc setComputePipelineState:fuse_down_reduce ?
     g_moe_down_iq4_xs_batch_reduce_pipeline :
     pair_down_reduce ? g_moe_down_iq4_xs_pair_pipeline : down_batch_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:down_w offset:(NSUInteger)down_inner atIndex:1];
    [enc setBuffer:obj.scratch offset:0 atIndex:2];
    [enc setBuffer:fuse_down_reduce ? obj.x0 : obj.scratch offset:0 atIndex:3];
    if (ids_buffer) {
        [enc setBuffer:ids_buffer offset:0 atIndex:4];
    } else {
        [enc setBytes:ids length:(NSUInteger)n_active * sizeof(int) atIndex:4];
    }
    if (weights_buffer) {
        [enc setBuffer:weights_buffer offset:0 atIndex:5];
    } else {
        [enc setBytes:weights length:(NSUInteger)n_active * sizeof(float) atIndex:5];
    }
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    if (fuse_down_reduce) {
        [enc setThreadgroupMemoryLength:16 * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((n_embd + 1u) / 2u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else if (pair_down_reduce) {
        threads = 64;
        [enc dispatchThreadgroups:MTLSizeMake(((n_embd + 3u) / 4u) * ((n_active + 1u) / 2u), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    } else if (down_type == 23) {
        threads = 64;
        [enc dispatchThreadgroups:MTLSizeMake(((n_embd + 3u) / 4u) * n_active, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    } else {
        threads = 64;
        [enc dispatchThreadgroups:MTLSizeMake(((n_embd + 3u) / 4u) * n_active, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    }
    qw3_metal_end_compute_encoder(cb, enc);
    if (profile_moe_sync) {
        if (!qw3_metal_synchronize()) return 0;
        fprintf(stderr, "qw3 metal moe profile down_type=%u down_ms=%.3f\n",
                down_type, ([NSDate timeIntervalSinceReferenceDate] - t_moe_sync) * 1000.0);
        if (!qw3_metal_begin_commands()) return 0;
        cb = qw3_metal_command_buffer(&owned);
        if (!cb) return 0;
        t_moe_sync = [NSDate timeIntervalSinceReferenceDate];
    }

    if (!fuse_down_reduce) {
        __typeof__(args) reduce_args = args;
        if (pair_down_reduce) reduce_args.n_active = (n_active + 1u) / 2u;
        enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:g_moe_reduce_batch_pipeline];
        [enc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
        [enc setBuffer:obj.scratch offset:0 atIndex:1];
        [enc setBuffer:obj.x0 offset:0 atIndex:2];
        threads = g_moe_reduce_batch_pipeline.maxTotalThreadsPerThreadgroup;
        if (threads > 256) threads = 256;
        [enc dispatchThreads:MTLSizeMake(n_embd / 4u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);
        if (profile_moe_sync) {
            if (!qw3_metal_synchronize()) return 0;
            fprintf(stderr, "qw3 metal moe profile down_type=%u reduce_ms=%.3f\n",
                    down_type, ([NSDate timeIntervalSinceReferenceDate] - t_moe_sync) * 1000.0);
            if (!qw3_metal_begin_commands()) return 0;
            cb = qw3_metal_command_buffer(&owned);
            if (!cb) return 0;
        }
    }

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session batch sparse MoE command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_sparse_moe_topk(qw3_metal_session *s,
                                      uint64_t gate_offset,
                                      uint64_t up_offset,
                                      uint64_t down_offset,
                                      uint32_t down_type,
                                      const int *ids,
                                      const float *weights,
                                      uint32_t n_active,
                                      uint32_t n_embd,
                                      uint32_t n_ff) {
    if (!s || !s->obj || !ids || !weights || n_active == 0 ||
        n_embd == 0 || n_ff == 0 || (n_embd % 256) != 0 ||
        (n_ff % 256) != 0) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;
    if (getenv("QW3_METAL_BATCH_MOE") != NULL &&
        getenv("QW3_METAL_LEGACY_MOE") == NULL) {
        int ok = qw3_metal_session_sparse_moe_topk_batch(
            s, gate_offset, up_offset, down_offset, down_type,
            ids, weights, nil, nil, n_active, n_embd, n_ff);
        if (ok) return 1;
        if (getenv("QW3_METAL_REQUIRE_BATCH_MOE") != NULL) return 0;
    }

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t embd_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t ff_bytes = (uint64_t)n_ff * sizeof(float);
    if (!obj.x0 || !obj.x1 || !obj.scratch || !obj.inner ||
        obj.x0.length < embd_bytes || obj.x1.length < embd_bytes ||
        obj.inner.length < ff_bytes || obj.scratch.length < 2ull * ff_bytes ||
        obj.scratch.length < embd_bytes) {
        return 0;
    }

    id<MTLBuffer> kgb = qw3_metal_iq3s_kgrid_buffer();
    if (!kgb) return 0;

    const uint64_t iq3_row_bytes = (uint64_t)(n_embd / 256) * 110ull;
    const uint64_t iq3_expert_bytes = iq3_row_bytes * (uint64_t)n_ff;
    const uint64_t iq4_row_bytes = (uint64_t)(n_ff / 256) * 136ull;
    const uint64_t q6_row_bytes = (uint64_t)(n_ff / 256) * 210ull;
    const uint64_t down_row_bytes = down_type == 23 ? iq4_row_bytes :
                                    down_type == 14 ? q6_row_bytes : 0;
    id<MTLComputePipelineState> down_pipeline =
        down_type == 23 ? g_matvec_iq4_xs_pipeline :
        down_type == 14 ? g_matvec_q6_k_pipeline : nil;
    id<MTLComputePipelineState> down_add_pipeline =
        down_type == 23 ? g_matvec_iq4_xs_add_x0_pipeline :
        down_type == 14 ? g_matvec_q6_k_add_x0_pipeline : nil;
    if (!down_row_bytes || !down_pipeline || !down_add_pipeline) return 0;
    const uint64_t down_expert_bytes = down_row_bytes * (uint64_t)n_embd;

    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
    } iq3_args = { n_embd, n_ff, (uint32_t)iq3_row_bytes };
    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
    } down_args = { n_ff, n_embd, (uint32_t)down_row_bytes };
    struct {
        uint32_t n;
        uint32_t a_offset;
        uint32_t b_offset;
    } silu_args = { n_ff, 0, n_ff };
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    if (!cb) return 0;
    for (uint32_t kk = 0; kk < n_active; kk++) {
        const uint32_t expert = (uint32_t)ids[kk];
        const uint64_t gate_expert_offset = gate_offset + (uint64_t)expert * iq3_expert_bytes;
        const uint64_t up_expert_offset = up_offset + (uint64_t)expert * iq3_expert_bytes;
        const uint64_t down_expert_offset = down_offset + (uint64_t)expert * down_expert_bytes;
        uint64_t gate_inner = 0, up_inner = 0, down_inner = 0;
        id<MTLBuffer> gate_w = qw3_metal_model_view_for(gate_expert_offset,
                                                        iq3_expert_bytes,
                                                        &gate_inner);
        id<MTLBuffer> up_w = qw3_metal_model_view_for(up_expert_offset,
                                                      iq3_expert_bytes,
                                                      &up_inner);
        id<MTLBuffer> down_w = qw3_metal_model_view_for(down_expert_offset,
                                                        down_expert_bytes,
                                                        &down_inner);
        if (!gate_w || !up_w || !down_w) return 0;

        id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:g_matvec_iq3_s_pair_pipeline];
        [enc setBytes:&iq3_args length:sizeof(iq3_args) atIndex:0];
        [enc setBuffer:gate_w offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:up_w offset:(NSUInteger)up_inner atIndex:2];
        [enc setBuffer:obj.x1 offset:0 atIndex:3];
        [enc setBuffer:obj.scratch offset:0 atIndex:4];
        [enc setBuffer:kgb offset:0 atIndex:5];
        [enc setThreadgroupMemoryLength:64 * sizeof(float) atIndex:0];
        NSUInteger threads = 64;
        [enc dispatchThreadgroups:MTLSizeMake((n_ff + 7u) / 8u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);

        enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:g_silu_mul_offsets_pipeline];
        [enc setBytes:&silu_args length:sizeof(silu_args) atIndex:0];
        [enc setBuffer:obj.scratch offset:0 atIndex:1];
        [enc setBuffer:obj.inner offset:0 atIndex:2];
        threads = g_silu_mul_offsets_pipeline.maxTotalThreadsPerThreadgroup;
        if (threads > 256) threads = 256;
        [enc dispatchThreads:MTLSizeMake(n_ff, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);

        float scale = weights[kk];
        enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:down_add_pipeline];
        [enc setBytes:&down_args length:sizeof(down_args) atIndex:0];
        [enc setBytes:&scale length:sizeof(scale) atIndex:1];
        [enc setBuffer:down_w offset:(NSUInteger)down_inner atIndex:2];
        [enc setBuffer:obj.inner offset:0 atIndex:3];
        [enc setBuffer:obj.x0 offset:0 atIndex:4];
        [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
        if (down_type == 23) {
            threads = 64;
            [enc dispatchThreadgroups:MTLSizeMake((n_embd + 3u) / 4u, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        } else {
            threads = down_add_pipeline.maxTotalThreadsPerThreadgroup;
            if (threads > 256) threads = 256;
            if (threads < 32) threads = 32;
            [enc dispatchThreadgroups:MTLSizeMake(n_embd, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
        qw3_metal_end_compute_encoder(cb, enc);
    }
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session sparse MoE top-k command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_sparse_moe_topk_from_router_scratch(qw3_metal_session *s,
                                                          uint64_t gate_offset,
                                                          uint64_t up_offset,
                                                          uint64_t down_offset,
                                                          uint32_t down_type,
                                                          uint32_t n_active,
                                                          uint32_t n_embd,
                                                          uint32_t n_ff) {
    if (!s || !s->obj || n_active == 0 || n_active > 8 ||
        n_embd == 0 || n_ff == 0 || (n_embd % 256) != 0 ||
        (n_ff % 256) != 0) {
        fprintf(stderr, "qw3: Metal dynamic sparse MoE invalid args\n");
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t embd_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t ff_bytes = (uint64_t)n_ff * sizeof(float);
    if (!obj.x0 || !obj.x1 || !obj.scratch || !obj.inner ||
        !obj.routerIds || !obj.routerWeights ||
        obj.x0.length < embd_bytes || obj.x1.length < embd_bytes ||
        obj.inner.length < ff_bytes || obj.scratch.length < 2ull * ff_bytes ||
        obj.scratch.length < embd_bytes ||
        obj.routerIds.length < 8ull * sizeof(int32_t) ||
        obj.routerWeights.length < 8ull * sizeof(float)) {
        fprintf(stderr, "qw3: Metal dynamic sparse MoE session buffers too small\n");
        return 0;
    }

    id<MTLBuffer> kgb = qw3_metal_iq3s_kgrid_buffer();
    if (!kgb) {
        fprintf(stderr, "qw3: Metal dynamic sparse MoE kgrid allocation failed\n");
        return 0;
    }

    const uint64_t iq3_row_bytes = (uint64_t)(n_embd / 256) * 110ull;
    const uint64_t iq3_expert_bytes = iq3_row_bytes * (uint64_t)n_ff;
    const uint64_t iq4_row_bytes = (uint64_t)(n_ff / 256) * 136ull;
    const uint64_t q6_row_bytes = (uint64_t)(n_ff / 256) * 210ull;
    const uint64_t down_row_bytes = down_type == 23 ? iq4_row_bytes :
                                    down_type == 14 ? q6_row_bytes : 0;
    id<MTLComputePipelineState> down_pipeline =
        down_type == 23 ? g_matvec_iq4_xs_expert_slot_pipeline :
        down_type == 14 ? g_matvec_q6_k_expert_slot_pipeline : nil;
    if (!down_row_bytes || !down_pipeline) {
        fprintf(stderr, "qw3: Metal dynamic sparse MoE unsupported down type %u\n", down_type);
        return 0;
    }
    const uint64_t down_expert_bytes = down_row_bytes * (uint64_t)n_embd;

    const uint64_t gate_tensor_bytes = iq3_expert_bytes * 256ull;
    const uint64_t up_tensor_bytes = iq3_expert_bytes * 256ull;
    const uint64_t down_tensor_bytes = down_expert_bytes * 256ull;
    uint64_t gate_inner = 0, up_inner = 0, down_inner = 0;
    id<MTLBuffer> gate_w = qw3_metal_model_view_for(
        gate_offset, gate_tensor_bytes, &gate_inner);
    id<MTLBuffer> up_w = qw3_metal_model_view_for(
        up_offset, up_tensor_bytes, &up_inner);
    id<MTLBuffer> down_w = qw3_metal_model_view_for(
        down_offset, down_tensor_bytes, &down_inner);
    if (!gate_w) {
        gate_w = qw3_metal_model_temp_buffer_for(
            gate_offset, gate_tensor_bytes, &gate_inner);
    }
    if (!up_w) {
        up_w = qw3_metal_model_temp_buffer_for(
            up_offset, up_tensor_bytes, &up_inner);
    }
    if (!down_w) {
        down_w = qw3_metal_model_temp_buffer_for(
            down_offset, down_tensor_bytes, &down_inner);
    }
    if (!gate_w || !up_w || !down_w) {
        fprintf(stderr,
                "qw3: Metal dynamic sparse MoE tensor view failed gate=%p up=%p down=%p sizes=%llu/%llu\n",
                gate_w, up_w, down_w,
                (unsigned long long)gate_tensor_bytes,
                (unsigned long long)down_tensor_bytes);
        return 0;
    }

    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
        uint32_t expert_bytes;
        uint32_t slot;
    } iq3_args = { n_embd, n_ff, (uint32_t)iq3_row_bytes,
                   (uint32_t)iq3_expert_bytes, 0 };
    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
        uint32_t expert_bytes;
        uint32_t slot;
    } down_args = { n_ff, n_embd, (uint32_t)down_row_bytes,
                    (uint32_t)down_expert_bytes, 0 };
    struct {
        uint32_t n;
        uint32_t a_offset;
        uint32_t b_offset;
    } silu_args = { n_ff, 0, n_ff };
    struct {
        uint32_t n;
        uint32_t a_offset;
        uint32_t b_offset;
    } add_args = { n_embd, 0, 0 };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    if (!cb) return 0;

    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_router_top8_pipeline];
    [enc setBuffer:obj.scratch offset:0 atIndex:0];
    [enc setBuffer:obj.routerIds offset:0 atIndex:1];
    [enc setBuffer:obj.routerWeights offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(256, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    if (g_batch_cb && getenv("QW3_METAL_DYNAMIC_LEGACY_MOE") == NULL &&
        getenv("QW3_METAL_NO_BATCH_MOE") == NULL) {
        int ok = qw3_metal_session_sparse_moe_topk_batch(
            s, gate_offset, up_offset, down_offset, down_type,
            NULL, NULL, obj.routerIds, obj.routerWeights,
            n_active, n_embd, n_ff);
        if (ok) return 1;
        if (getenv("QW3_METAL_REQUIRE_BATCH_MOE") != NULL) return 0;
    }

    for (uint32_t kk = 0; kk < n_active; kk++) {
        iq3_args.slot = kk;
        down_args.slot = kk;

        enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:g_matvec_iq3_s_expert_slot_pair_pipeline];
        [enc setBytes:&iq3_args length:sizeof(iq3_args) atIndex:0];
        [enc setBuffer:gate_w offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:up_w offset:(NSUInteger)up_inner atIndex:2];
        [enc setBuffer:obj.x1 offset:0 atIndex:3];
        [enc setBuffer:obj.scratch offset:0 atIndex:4];
        [enc setBuffer:kgb offset:0 atIndex:5];
        [enc setBuffer:obj.routerIds offset:0 atIndex:6];
        [enc setThreadgroupMemoryLength:64 * sizeof(float) atIndex:0];
        NSUInteger threads = 64;
        [enc dispatchThreadgroups:MTLSizeMake((n_ff + 7u) / 8u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);

        enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:g_silu_mul_offsets_pipeline];
        [enc setBytes:&silu_args length:sizeof(silu_args) atIndex:0];
        [enc setBuffer:obj.scratch offset:0 atIndex:1];
        [enc setBuffer:obj.inner offset:0 atIndex:2];
        threads = g_silu_mul_offsets_pipeline.maxTotalThreadsPerThreadgroup;
        if (threads > 256) threads = 256;
        [enc dispatchThreads:MTLSizeMake(n_ff, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);

        if (down_type == 23) {
            enc = qw3_metal_compute_encoder(cb);
            [enc setComputePipelineState:down_pipeline];
            [enc setBytes:&down_args length:sizeof(down_args) atIndex:0];
            [enc setBuffer:down_w offset:(NSUInteger)down_inner atIndex:1];
            [enc setBuffer:obj.inner offset:0 atIndex:2];
            [enc setBuffer:obj.x0 offset:0 atIndex:3];
            [enc setBuffer:obj.routerIds offset:0 atIndex:4];
            [enc setBuffer:obj.routerWeights offset:0 atIndex:5];
            [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
            threads = 64;
            [enc dispatchThreadgroups:MTLSizeMake((n_embd + 3u) / 4u, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            qw3_metal_end_compute_encoder(cb, enc);
        } else {
            enc = qw3_metal_compute_encoder(cb);
            [enc setComputePipelineState:down_pipeline];
            [enc setBytes:&down_args length:sizeof(down_args) atIndex:0];
            [enc setBuffer:down_w offset:(NSUInteger)down_inner atIndex:1];
            [enc setBuffer:obj.inner offset:0 atIndex:2];
            [enc setBuffer:obj.scratch offset:0 atIndex:3];
            [enc setBuffer:obj.routerIds offset:0 atIndex:4];
            [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
            threads = down_pipeline.maxTotalThreadsPerThreadgroup;
            if (threads > 256) threads = 256;
            if (threads < 32) threads = 32;
            [enc dispatchThreadgroups:MTLSizeMake(n_embd, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            qw3_metal_end_compute_encoder(cb, enc);

            enc = qw3_metal_compute_encoder(cb);
            [enc setComputePipelineState:g_scale_scratch_add_x0_slot_pipeline];
            [enc setBytes:&add_args length:sizeof(add_args) atIndex:0];
            [enc setBytes:&kk length:sizeof(kk) atIndex:1];
            [enc setBuffer:obj.x0 offset:0 atIndex:2];
            [enc setBuffer:obj.scratch offset:0 atIndex:3];
            [enc setBuffer:obj.routerWeights offset:0 atIndex:4];
            threads = g_scale_scratch_add_x0_slot_pipeline.maxTotalThreadsPerThreadgroup;
            if (threads > 256) threads = 256;
            [enc dispatchThreads:MTLSizeMake(n_embd, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            qw3_metal_end_compute_encoder(cb, enc);
        }
    }
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session dynamic sparse MoE command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_matvec_q8_0_x1_to_logits(qw3_metal_session *s,
                                               uint64_t tensor_offset,
                                               uint32_t n_in,
                                               uint32_t n_out,
                                               float *out) {
    if (!s || !s->obj || n_in == 0 || n_out == 0 || (n_in % 32) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_in * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_out * sizeof(float);
    if (!obj.x1 || !obj.logits || obj.x1.length < x_bytes ||
        obj.logits.length < out_bytes) {
        return 0;
    }

    const uint64_t row_bytes = (uint64_t)(n_in / 32) * 34ull;
    const uint64_t tensor_bytes = row_bytes * (uint64_t)n_out;
    uint64_t inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(tensor_offset, tensor_bytes, &inner);
    if (!wb) {
        fprintf(stderr, "qw3: Metal session q8_0 logits matvec tensor is outside mapped model views\n");
        return 0;
    }

    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
    } args = { n_in, n_out, (uint32_t)row_bytes };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_matvec_q8_0_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)inner atIndex:1];
    [enc setBuffer:obj.x1 offset:0 atIndex:2];
    [enc setBuffer:obj.logits offset:0 atIndex:3];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_matvec_q8_0_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(n_out, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    id<MTLBuffer> readback = nil;
    if (out) {
        readback = [g_device newBufferWithLength:(NSUInteger)out_bytes
                                         options:MTLResourceStorageModeShared];
        if (!readback) return 0;
        qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:obj.logits sourceOffset:0
                    toBuffer:readback destinationOffset:0
                        size:(NSUInteger)out_bytes];
        [blit endEncoding];
    }

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session q8_0 logits matvec command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    if (out) memcpy(out, readback.contents, (size_t)out_bytes);
    return 1;
}

int qw3_metal_session_matvec_q6_k_x1_to_logits(qw3_metal_session *s,
                                               uint64_t tensor_offset,
                                               uint32_t n_in,
                                               uint32_t n_out,
                                               float *out) {
    if (!s || !s->obj || n_in == 0 || n_out == 0 || (n_in % 256) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_in * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_out * sizeof(float);
    if (!obj.x1 || !obj.logits || obj.x1.length < x_bytes ||
        obj.logits.length < out_bytes) {
        return 0;
    }

    const uint64_t row_bytes = (uint64_t)(n_in / 256) * 210ull;
    const uint64_t tensor_bytes = row_bytes * (uint64_t)n_out;
    uint64_t inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(tensor_offset, tensor_bytes, &inner);
    if (!wb) {
        fprintf(stderr, "qw3: Metal session q6_K logits matvec tensor is outside mapped model views\n");
        return 0;
    }

    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
    } args = { n_in, n_out, (uint32_t)row_bytes };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_matvec_q6_k_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)inner atIndex:1];
    [enc setBuffer:obj.x1 offset:0 atIndex:2];
    [enc setBuffer:obj.logits offset:0 atIndex:3];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = 64;
    [enc dispatchThreadgroups:MTLSizeMake((n_out + 3u) / 4u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    id<MTLBuffer> readback = nil;
    if (out) {
        readback = [g_device newBufferWithLength:(NSUInteger)out_bytes
                                         options:MTLResourceStorageModeShared];
        if (!readback) return 0;
        qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:obj.logits sourceOffset:0
                    toBuffer:readback destinationOffset:0
                        size:(NSUInteger)out_bytes];
        [blit endEncoding];
    }

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session q6_K logits matvec command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    if (out) memcpy(out, readback.contents, (size_t)out_bytes);
    return 1;
}

int qw3_metal_session_argmax_logits(qw3_metal_session *s, uint32_t n,
                                    uint32_t *idx_out, float *val_out) {
    if (!s || !s->obj || !idx_out || !val_out || n == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    if (!obj.logits || obj.logits.length < (NSUInteger)n * sizeof(float)) {
        return 0;
    }

    const uint32_t threads_u32 = 256;
    const uint32_t n_blocks = (n + threads_u32 - 1) / threads_u32;
    const NSUInteger vals_bytes = (NSUInteger)n_blocks * sizeof(float);
    const NSUInteger idxs_bytes = (NSUInteger)n_blocks * sizeof(uint32_t);
    id<MTLBuffer> valsb = obj.argmaxVals;
    id<MTLBuffer> idxsb = obj.argmaxIdxs;
    if (!valsb || !idxsb || valsb.length < vals_bytes || idxsb.length < idxs_bytes) {
        valsb = [g_device newBufferWithLength:vals_bytes
                                      options:MTLResourceStorageModeShared];
        idxsb = [g_device newBufferWithLength:idxs_bytes
                                      options:MTLResourceStorageModeShared];
    }
    if (!valsb || !idxsb) {
        fprintf(stderr, "qw3: Metal session argmax buffer allocation failed\n");
        return 0;
    }

    struct {
        uint32_t n;
    } args = { n };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_argmax_blocks_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.logits offset:0 atIndex:1];
    [enc setBuffer:valsb offset:0 atIndex:2];
    [enc setBuffer:idxsb offset:0 atIndex:3];
    [enc setThreadgroupMemoryLength:(NSUInteger)threads_u32 * sizeof(float)
                            atIndex:0];
    [enc setThreadgroupMemoryLength:(NSUInteger)threads_u32 * sizeof(uint32_t)
                            atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake(n_blocks, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads_u32, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session argmax command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }

    const float *vals = (const float *)valsb.contents;
    const uint32_t *idxs = (const uint32_t *)idxsb.contents;
    uint32_t best_idx = idxs[0];
    float best_val = vals[0];
    for (uint32_t i = 1; i < n_blocks; i++) {
        uint32_t idx = idxs[i];
        float val = vals[i];
        if (val > best_val || (val == best_val && idx < best_idx)) {
            best_val = val;
            best_idx = idx;
        }
    }
    *idx_out = best_idx;
    *val_out = best_val;
    return 1;
}

int qw3_metal_session_residual_rmsnorm_x0_x1(qw3_metal_session *s,
                                             uint64_t weight_offset,
                                             uint32_t n, float eps,
                                             float *out) {
    if (!s || !s->obj || n == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    if (!obj.x0 || !obj.x1 || obj.x0.length < bytes || obj.x1.length < bytes) {
        return 0;
    }
    if (!g_model_map_ptr || weight_offset > g_model_map_size ||
        bytes > g_model_map_size - weight_offset) {
        fprintf(stderr, "qw3: Metal session residual RMSNorm weight is outside mapped model\n");
        return 0;
    }
    uint64_t weight_inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(weight_offset, bytes, &weight_inner);
    if (!wb) return 0;
    struct {
        uint32_t n;
        float eps;
    } args = { n, eps };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_residual_rmsnorm_weight_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.x0 offset:0 atIndex:1];
    [enc setBuffer:obj.x1 offset:0 atIndex:2];
    [enc setBuffer:wb offset:(NSUInteger)weight_inner atIndex:3];
    [enc setBuffer:obj.x1 offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_residual_rmsnorm_weight_f32_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    id<MTLBuffer> readback = nil;
    if (out) {
        readback = [g_device newBufferWithLength:(NSUInteger)bytes
                                         options:MTLResourceStorageModeShared];
        if (!readback) return 0;
        qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:obj.x1 sourceOffset:0
                    toBuffer:readback destinationOffset:0
                        size:(NSUInteger)bytes];
        [blit endEncoding];
    }

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session residual RMSNorm command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    if (out) memcpy(out, readback.contents, (size_t)bytes);
    return 1;
}

int qw3_metal_session_residual_rmsnorm_update_x0_x1(qw3_metal_session *s,
                                                    uint64_t weight_offset,
                                                    uint32_t n, float eps,
                                                    float *out) {
    if (!s || !s->obj || n == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    if (!obj.x0 || !obj.x1 || obj.x0.length < bytes || obj.x1.length < bytes) {
        return 0;
    }
    if (!g_model_map_ptr || weight_offset > g_model_map_size ||
        bytes > g_model_map_size - weight_offset) {
        fprintf(stderr, "qw3: Metal session residual update RMSNorm weight is outside mapped model\n");
        return 0;
    }
    uint64_t weight_inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(weight_offset, bytes, &weight_inner);
    if (!wb) return 0;
    struct {
        uint32_t n;
        float eps;
    } args = { n, eps };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_residual_rmsnorm_update_x0_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.x0 offset:0 atIndex:1];
    [enc setBuffer:obj.x1 offset:0 atIndex:2];
    [enc setBuffer:wb offset:(NSUInteger)weight_inner atIndex:3];
    [enc setBuffer:obj.x1 offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_residual_rmsnorm_update_x0_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    id<MTLBuffer> readback = nil;
    if (out) {
        readback = [g_device newBufferWithLength:(NSUInteger)bytes
                                         options:MTLResourceStorageModeShared];
        if (!readback) return 0;
        qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:obj.x1 sourceOffset:0
                    toBuffer:readback destinationOffset:0
                        size:(NSUInteger)bytes];
        [blit endEncoding];
    }

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session residual update RMSNorm command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    if (out) memcpy(out, readback.contents, (size_t)bytes);
    return 1;
}

static int qw3_metal_encode_rope_heads(id<MTLCommandBuffer> cb,
                                       id<MTLBuffer> x,
                                       id<MTLBuffer> out,
                                       uint32_t n_heads,
                                       uint32_t head_dim,
                                       uint32_t rope_dim,
                                       uint32_t pos,
                                       float theta) {
    struct {
        uint32_t n_heads;
        uint32_t head_dim;
        uint32_t rope_dim;
        int32_t pos;
        float theta;
    } args = { n_heads, head_dim, rope_dim, (int32_t)pos, theta };
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    if (!enc) return 0;
    [enc setComputePipelineState:g_rope_heads_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:x offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    NSUInteger threads = g_rope_heads_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake(n_heads * head_dim, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    return 1;
}

int qw3_metal_session_gqa_project_cache(qw3_metal_session *s,
                                        uint64_t q_weight_offset,
                                        uint64_t k_weight_offset,
                                        uint64_t v_weight_offset,
                                        uint64_t q_norm_weight_offset,
                                        uint64_t k_norm_weight_offset,
                                        uint32_t qg_n, uint32_t q_n,
                                        uint32_t kv_n, uint32_t n_heads,
                                        uint32_t n_kv_heads,
                                        uint32_t head_dim,
                                        uint32_t rope_dim,
                                        uint32_t layer_slot,
                                        uint32_t pos,
                                        float rope_theta,
                                        float eps,
                                        float *q_out, float *k_out,
                                        float *v_out, float *gate_out) {
    if (!s || !s->obj || qg_n == 0 || q_n == 0 || kv_n == 0 ||
        n_heads == 0 || n_kv_heads == 0 || head_dim == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    if (obj.gqaKvQ8 && ((head_dim % 32u) != 0u || (kv_n % 32u) != 0u)) {
        return 0;
    }
    const uint32_t k_offset = qg_n;
    const uint32_t v_offset = qg_n + kv_n;
    const uint64_t q_bytes = (uint64_t)q_n * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)kv_n * sizeof(float);
    const uint64_t cache_kv_bytes = obj.gqaKvQ8 ?
        (uint64_t)(kv_n / 32u) * 34ull : kv_bytes;
    const uint64_t scratch_needed = (uint64_t)(qg_n + 2u * kv_n) * sizeof(float);
    const uint64_t cache_layer_bytes = (uint64_t)obj.ctxSize * cache_kv_bytes;
    const uint64_t cache_offset =
        ((uint64_t)layer_slot * (uint64_t)obj.ctxSize + (uint64_t)pos) * cache_kv_bytes;
    if (pos >= obj.ctxSize || obj.scratch.length < scratch_needed ||
        obj.gqaTmpQ.length < q_bytes || obj.gqaTokenQ.length < q_bytes ||
        obj.gqaTokenGate.length < q_bytes ||
        obj.gqaTmpK.length < kv_bytes || obj.gqaTokenK.length < kv_bytes ||
        obj.gqaTokenV.length < kv_bytes ||
        obj.gqaK.length < cache_offset + cache_kv_bytes ||
        obj.gqaV.length < cache_offset + cache_kv_bytes) {
        (void)cache_layer_bytes;
        return 0;
    }

    if (!qw3_metal_session_matvec_q8_0_x1_to_scratch(
            s, q_weight_offset, QW3_METAL_N_EMBD, qg_n, 0, NULL) ||
        !qw3_metal_session_matvec_q8_0_x1_to_scratch(
            s, k_weight_offset, QW3_METAL_N_EMBD, kv_n, k_offset, NULL) ||
        !qw3_metal_session_matvec_q8_0_x1_to_scratch(
            s, v_weight_offset, QW3_METAL_N_EMBD, kv_n, v_offset, NULL)) {
        return 0;
    }

    const uint64_t norm_weight_bytes = (uint64_t)head_dim * sizeof(float);
    if (!g_model_map_ptr ||
        q_norm_weight_offset > g_model_map_size ||
        norm_weight_bytes > g_model_map_size - q_norm_weight_offset ||
        k_norm_weight_offset > g_model_map_size ||
        norm_weight_bytes > g_model_map_size - k_norm_weight_offset) {
        fprintf(stderr, "qw3: Metal session GQA norm weight is outside mapped model\n");
        return 0;
    }
    uint64_t q_norm_inner = 0;
    uint64_t k_norm_inner = 0;
    id<MTLBuffer> qw = qw3_metal_model_view_for(q_norm_weight_offset,
                                                norm_weight_bytes,
                                                &q_norm_inner);
    id<MTLBuffer> kw = qw3_metal_model_view_for(k_norm_weight_offset,
                                                norm_weight_bytes,
                                                &k_norm_inner);
    if (!qw || !kw) return 0;

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    if (!cb) return 0;

    struct {
        uint32_t n_heads;
        uint32_t head_dim;
        float eps;
    } q_norm_args = { n_heads, head_dim, eps };
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    if (!enc) return 0;
    [enc setComputePipelineState:g_gqa_q_norm_gate_pipeline];
    [enc setBytes:&q_norm_args length:sizeof(q_norm_args) atIndex:0];
    [enc setBuffer:obj.scratch offset:0 atIndex:1];
    [enc setBuffer:qw offset:(NSUInteger)q_norm_inner atIndex:2];
    [enc setBuffer:obj.gqaTmpQ offset:0 atIndex:3];
    [enc setBuffer:obj.gqaTokenGate offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_gqa_q_norm_gate_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    struct {
        uint32_t n_heads;
        uint32_t head_dim;
        float eps;
    } k_norm_args = { n_kv_heads, head_dim, eps };
    enc = qw3_metal_compute_encoder(cb);
    if (!enc) return 0;
    [enc setComputePipelineState:g_gqa_k_norm_pipeline];
    [enc setBytes:&k_norm_args length:sizeof(k_norm_args) atIndex:0];
    [enc setBuffer:obj.scratch offset:(NSUInteger)k_offset * sizeof(float) atIndex:1];
    [enc setBuffer:kw offset:(NSUInteger)k_norm_inner atIndex:2];
    [enc setBuffer:obj.gqaTmpK offset:0 atIndex:3];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    threads = g_gqa_k_norm_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(n_kv_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    if (!qw3_metal_encode_rope_heads(cb, obj.gqaTmpQ, obj.gqaTokenQ,
                                     n_heads, head_dim, rope_dim, pos,
                                     rope_theta) ||
        !qw3_metal_encode_rope_heads(cb, obj.gqaTmpK, obj.gqaTokenK,
                                     n_kv_heads, head_dim, rope_dim, pos,
                                     rope_theta)) {
        return 0;
    }

    qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!blit) return 0;
    [blit copyFromBuffer:obj.scratch sourceOffset:(NSUInteger)v_offset * sizeof(float)
                toBuffer:obj.gqaTokenV destinationOffset:0
                    size:(NSUInteger)kv_bytes];
    if (!obj.gqaKvQ8) {
        [blit copyFromBuffer:obj.gqaTokenK sourceOffset:0
                    toBuffer:obj.gqaK destinationOffset:(NSUInteger)cache_offset
                        size:(NSUInteger)kv_bytes];
        [blit copyFromBuffer:obj.gqaTokenV sourceOffset:0
                    toBuffer:obj.gqaV destinationOffset:(NSUInteger)cache_offset
                        size:(NSUInteger)kv_bytes];
    }

    id<MTLBuffer> q_readback = nil, k_readback = nil, v_readback = nil, gate_readback = nil;
    if (q_out) {
        q_readback = [g_device newBufferWithLength:(NSUInteger)q_bytes
                                           options:MTLResourceStorageModeShared];
        if (!q_readback) return 0;
        [blit copyFromBuffer:obj.gqaTokenQ sourceOffset:0
                    toBuffer:q_readback destinationOffset:0
                        size:(NSUInteger)q_bytes];
    }
    if (k_out) {
        k_readback = [g_device newBufferWithLength:(NSUInteger)kv_bytes
                                           options:MTLResourceStorageModeShared];
        if (!k_readback) return 0;
        [blit copyFromBuffer:obj.gqaTokenK sourceOffset:0
                    toBuffer:k_readback destinationOffset:0
                        size:(NSUInteger)kv_bytes];
    }
    if (v_out) {
        v_readback = [g_device newBufferWithLength:(NSUInteger)kv_bytes
                                           options:MTLResourceStorageModeShared];
        if (!v_readback) return 0;
        [blit copyFromBuffer:obj.gqaTokenV sourceOffset:0
                    toBuffer:v_readback destinationOffset:0
                        size:(NSUInteger)kv_bytes];
    }
    if (gate_out) {
        gate_readback = [g_device newBufferWithLength:(NSUInteger)q_bytes
                                              options:MTLResourceStorageModeShared];
        if (!gate_readback) return 0;
        [blit copyFromBuffer:obj.gqaTokenGate sourceOffset:0
                    toBuffer:gate_readback destinationOffset:0
                        size:(NSUInteger)q_bytes];
    }
    [blit endEncoding];

    if (obj.gqaKvQ8) {
        struct {
            uint32_t n;
        } quant_args = { kv_n };
        enc = qw3_metal_compute_encoder(cb);
        if (!enc) return 0;
        [enc setComputePipelineState:g_gqa_kv_quant_q8_pipeline];
        [enc setBytes:&quant_args length:sizeof(quant_args) atIndex:0];
        [enc setBuffer:obj.gqaTokenK offset:0 atIndex:1];
        [enc setBuffer:obj.gqaTokenV offset:0 atIndex:2];
        [enc setBuffer:obj.gqaK offset:(NSUInteger)cache_offset atIndex:3];
        [enc setBuffer:obj.gqaV offset:(NSUInteger)cache_offset atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(kv_n / 32u, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);
    }

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session GQA project/cache command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    if (q_out) memcpy(q_out, q_readback.contents, (size_t)q_bytes);
    if (k_out) memcpy(k_out, k_readback.contents, (size_t)kv_bytes);
    if (v_out) memcpy(v_out, v_readback.contents, (size_t)kv_bytes);
    if (gate_out) memcpy(gate_out, gate_readback.contents, (size_t)q_bytes);
    return 1;
}

int qw3_metal_session_gqa_single_attn_out(qw3_metal_session *s,
                                          uint64_t out_weight_offset,
                                          uint32_t n_heads,
                                          uint32_t n_kv_heads,
                                          uint32_t head_dim,
                                          uint32_t n_embd,
                                          float *out) {
    if (!s || !s->obj || n_heads == 0 || n_kv_heads == 0 ||
        head_dim == 0 || n_embd == 0 || (n_heads % n_kv_heads) != 0) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint32_t inner_n = n_heads * head_dim;
    const uint32_t kv_n = n_kv_heads * head_dim;
    const uint64_t inner_bytes = (uint64_t)inner_n * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)kv_n * sizeof(float);
    if (!obj.gqaTokenGate || !obj.gqaTokenV || !obj.inner ||
        obj.gqaTokenGate.length < inner_bytes ||
        obj.gqaTokenV.length < kv_bytes ||
        obj.inner.length < inner_bytes) {
        return 0;
    }

    struct {
        uint32_t n_heads;
        uint32_t n_kv_heads;
        uint32_t head_dim;
    } args = { n_heads, n_kv_heads, head_dim };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    if (!enc) return 0;
    [enc setComputePipelineState:g_gqa_single_token_inner_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.gqaTokenGate offset:0 atIndex:1];
    [enc setBuffer:obj.gqaTokenV offset:0 atIndex:2];
    [enc setBuffer:obj.inner offset:0 atIndex:3];
    NSUInteger threads = g_gqa_single_token_inner_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake(inner_n, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session GQA single-token inner command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }

    return qw3_metal_session_matvec_q8_0_inner_to_x1(
        s, out_weight_offset, inner_n, n_embd, out);
}

int qw3_metal_session_gqa_cached_attn_out(qw3_metal_session *s,
                                          uint64_t out_weight_offset,
                                          uint32_t n_ctx,
                                          uint32_t layer_slot,
                                          uint32_t n_heads,
                                          uint32_t n_kv_heads,
                                          uint32_t head_dim,
                                          uint32_t n_embd,
                                          float *out) {
    if (!s || !s->obj || n_ctx == 0 || n_heads == 0 || n_kv_heads == 0 ||
        head_dim == 0 || n_embd == 0 || (n_heads % n_kv_heads) != 0) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    if (n_ctx > obj.ctxSize) return 0;
    if (obj.gqaKvQ8 && (head_dim % 32u) != 0u) return 0;
    const uint32_t inner_n = n_heads * head_dim;
    const uint32_t kv_n = n_kv_heads * head_dim;
    const uint64_t inner_bytes = (uint64_t)inner_n * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)kv_n * sizeof(float);
    const uint64_t cache_kv_bytes = obj.gqaKvQ8 ?
        (uint64_t)(kv_n / 32u) * 34ull : kv_bytes;
    const uint64_t cache_offset =
        (uint64_t)layer_slot * (uint64_t)obj.ctxSize * cache_kv_bytes;
    const uint64_t cache_bytes = (uint64_t)n_ctx * cache_kv_bytes;
    if (!obj.gqaTokenQ || !obj.gqaTokenGate || !obj.gqaK || !obj.gqaV ||
        !obj.inner || obj.gqaTokenQ.length < inner_bytes ||
        obj.gqaTokenGate.length < inner_bytes ||
        obj.gqaK.length < cache_offset + cache_bytes ||
        obj.gqaV.length < cache_offset + cache_bytes ||
        obj.inner.length < inner_bytes) {
        return 0;
    }

    struct {
        uint32_t n_ctx;
        uint32_t n_heads;
        uint32_t n_kv_heads;
        uint32_t head_dim;
    } args = { n_ctx, n_heads, n_kv_heads, head_dim };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    if (!enc) return 0;
    id<MTLComputePipelineState> pipeline = obj.gqaKvQ8 ?
        g_gqa_attend_n_q8_inner_pipeline : g_gqa_attend_n_inner_pipeline;
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.gqaTokenQ offset:0 atIndex:1];
    [enc setBuffer:obj.gqaTokenGate offset:0 atIndex:2];
    [enc setBuffer:obj.gqaK offset:(NSUInteger)cache_offset atIndex:3];
    [enc setBuffer:obj.gqaV offset:(NSUInteger)cache_offset atIndex:4];
    [enc setBuffer:obj.inner offset:0 atIndex:5];
    NSUInteger threads = ((NSUInteger)head_dim + 31u) & ~(NSUInteger)31u;
    if (threads < 32u) threads = 32u;
    if (threads > 256u ||
        threads > pipeline.maxTotalThreadsPerThreadgroup) {
        qw3_metal_end_compute_encoder(cb, enc);
        return 0;
    }
    [enc setThreadgroupMemoryLength:64u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(n_kv_heads, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session GQA cached attention command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }

    return qw3_metal_session_matvec_q8_0_inner_to_x1(
        s, out_weight_offset, inner_n, n_embd, out);
}

int qw3_metal_init(void) {
    if (g_initialized) return 1;

    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        if (devices.count > 0) {
            g_device = devices[0];
            fprintf(stderr,
                    "qw3: Metal default device unavailable, using first listed device (%lu total)\n",
                    (unsigned long)devices.count);
        }
    }
    if (!g_device) {
        fprintf(stderr, "qw3: Metal device unavailable\n");
        return 0;
    }

    g_queue = [g_device newCommandQueue];
    if (!g_queue) {
        fprintf(stderr, "qw3: Metal command queue creation failed\n");
        g_device = nil;
        return 0;
    }
    g_model_buffers = [[NSMutableArray alloc] init];
    g_pending_cbs = [[NSMutableArray alloc] init];
    if (!qw3_metal_compile_kernels()) {
        g_model_buffers = nil;
        g_pending_cbs = nil;
        g_queue = nil;
        g_device = nil;
        return 0;
    }

    const char *name = g_device.name ? [g_device.name UTF8String] : "unknown Metal device";
    snprintf(g_device_name, sizeof(g_device_name), "%s", name);
    g_initialized = 1;
    fprintf(stderr, "qw3: Metal device %s\n", qw3_metal_device_name());
    return 1;
}

void qw3_metal_cleanup(void) {
    if (g_batch_cb) {
        qw3_metal_close_batch_encoder();
        [g_batch_cb commit];
        [g_batch_cb waitUntilCompleted];
        g_batch_cb = nil;
    }
    (void)qw3_metal_wait_pending_command_buffers("cleanup");
    g_pending_cbs = nil;
    [g_model_buffers removeAllObjects];
    g_model_buffers = nil;
    [g_model_temp_buffers removeAllObjects];
    g_model_temp_buffers = nil;
    g_iq3s_kgrid_buffer = nil;
    g_rmsnorm_plain_pipeline = nil;
    g_rmsnorm_weight_f32_pipeline = nil;
    g_embed_q8_0_pipeline = nil;
    g_matvec_q8_0_pipeline = nil;
    g_matvec_q8_0_pair_pipeline = nil;
    g_matvec_q8_0_pair_silu_pipeline = nil;
    g_shared_gate_up_silu_pipeline = nil;
    g_matvec_q8_0_inner_scale_add_x0_pipeline = nil;
    g_matvec_iq4_xs_pipeline = nil;
    g_matvec_q6_k_pipeline = nil;
    g_matvec_iq4_xs_add_x0_pipeline = nil;
    g_matvec_q6_k_add_x0_pipeline = nil;
    g_matvec_iq4_xs_swiglu_add_x0_pipeline = nil;
    g_matvec_q6_k_swiglu_add_x0_pipeline = nil;
    g_matvec_iq3_s_pipeline = nil;
    g_matvec_iq3_s_pair_pipeline = nil;
    g_moe_iq3_s_pair_batch_pipeline = nil;
    g_moe_down_iq4_xs_batch_pipeline = nil;
    g_moe_down_iq4_xs_pair_pipeline = nil;
    g_moe_down_iq4_xs_batch_reduce_pipeline = nil;
    g_moe_down_q6_k_batch_pipeline = nil;
    g_moe_reduce_batch_pipeline = nil;
    g_matvec_f32_pipeline = nil;
    g_matvec_f32_pair_pipeline = nil;
    g_matvec_f32_fast_pipeline = nil;
    g_deltanet_conv1d_zero_pipeline = nil;
    g_deltanet_conv1d_step_pipeline = nil;
    g_l2norm_heads_pipeline = nil;
    g_gqa_q_norm_gate_pipeline = nil;
    g_gqa_k_norm_pipeline = nil;
    g_rope_heads_pipeline = nil;
    g_gqa_single_token_inner_pipeline = nil;
    g_gqa_attend2_inner_pipeline = nil;
    g_gqa_attend_n_inner_pipeline = nil;
    g_gqa_kv_quant_q8_pipeline = nil;
    g_gqa_attend_n_q8_inner_pipeline = nil;
    g_deltanet_recur_zero_pipeline = nil;
    g_deltanet_recur_pipeline = nil;
    g_deltanet_recur_scratch_gates_pipeline = nil;
    g_deltanet_fused_gdn_scratch_pipeline = nil;
    g_deltanet_gated_rmsnorm_pipeline = nil;
    g_residual_rmsnorm_weight_f32_pipeline = nil;
    g_residual_rmsnorm_update_x0_pipeline = nil;
    g_silu_mul_pipeline = nil;
    g_scale_pipeline = nil;
    g_add_moe_to_x0_pipeline = nil;
    g_silu_mul_offsets_pipeline = nil;
    g_scale_x1_scalar_add_x0_pipeline = nil;
    g_scale_x1_add_x0_pipeline = nil;
    g_scale_scratch_add_x0_pipeline = nil;
    g_router_top8_pipeline = nil;
    g_matvec_iq3_s_expert_slot_pipeline = nil;
    g_matvec_iq3_s_expert_slot_pair_pipeline = nil;
    g_matvec_iq4_xs_expert_slot_pipeline = nil;
    g_matvec_q6_k_expert_slot_pipeline = nil;
    g_scale_scratch_add_x0_slot_pipeline = nil;
    g_library = nil;
    g_queue = nil;
    g_device = nil;
    g_batch_enc = nil;
    g_model_map_ptr = NULL;
    g_model_map_size = 0;
    g_model_offset = 0;
    g_model_size = 0;
    g_model_view_count = 0;
    memset(g_model_view_ptrs, 0, sizeof(g_model_view_ptrs));
    memset(g_model_view_offsets, 0, sizeof(g_model_view_offsets));
    memset(g_model_view_sizes, 0, sizeof(g_model_view_sizes));
    g_device_name[0] = '\0';
    g_initialized = 0;
}

int qw3_metal_set_model_map_range(const void *model_map, uint64_t model_size,
                                  uint64_t map_offset, uint64_t map_size) {
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!model_map || map_offset > model_size || map_size > model_size - map_offset) {
        fprintf(stderr, "qw3: invalid Metal model map range\n");
        return 0;
    }

    const uint64_t page = (uint64_t)getpagesize();
    const uintptr_t base = (uintptr_t)model_map;
    if ((base & (uintptr_t)(page - 1)) != 0) {
        fprintf(stderr, "qw3: Metal model mmap base is not page aligned\n");
        return 0;
    }

    const uint64_t page_offset = map_offset & ~(page - 1);
    const uint64_t leading = map_offset - page_offset;
    const uint64_t mapped_size = round_up_u64(leading + map_size, page);
    uint64_t max_buffer = (uint64_t)[g_device maxBufferLength] & ~(page - 1);
    if (max_buffer < page) {
        fprintf(stderr, "qw3: Metal maxBufferLength is too small\n");
        return 0;
    }

    [g_model_buffers removeAllObjects];
    [g_model_temp_buffers removeAllObjects];
    uint64_t done = 0;
    uint32_t n_views = 0;
    while (done < mapped_size) {
        uint64_t view_size = mapped_size - done;
        if (view_size > max_buffer) view_size = max_buffer;

        id<MTLBuffer> buffer = [g_device newBufferWithBytesNoCopy:(void *)(base + page_offset + done)
                                                           length:(NSUInteger)view_size
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        if (!buffer) {
            fprintf(stderr,
                    "qw3: Metal could not wrap GGUF view %u at %.2f GiB size %.2f GiB\n",
                    n_views,
                    (double)(page_offset + done) / (1024.0 * 1024.0 * 1024.0),
                    (double)view_size / (1024.0 * 1024.0 * 1024.0));
            [g_model_buffers removeAllObjects];
            return 0;
        }
        buffer.label = [NSString stringWithFormat:@"qw3_model_view_%u", n_views];
        [g_model_buffers addObject:buffer];
        if (n_views < 32) {
            g_model_view_ptrs[n_views] = (const uint8_t *)(base + page_offset + done);
            g_model_view_offsets[n_views] = page_offset + done;
            g_model_view_sizes[n_views] = view_size;
        }
        done += view_size;
        n_views++;
    }
    g_model_map_ptr = model_map;
    g_model_map_size = model_size;
    g_model_offset = page_offset;
    g_model_size = mapped_size;
    g_model_view_count = n_views > 32 ? 32 : n_views;

    fprintf(stderr,
            "qw3: Metal model mapped %.2f MiB from GGUF offset %.2f MiB in %u view(s)\n",
            (double)g_model_size / (1024.0 * 1024.0),
            (double)g_model_offset / (1024.0 * 1024.0),
            n_views);
    return 1;
}

int qw3_metal_rmsnorm_plain(const float *x, float *out, uint32_t n, float eps) {
    if (!x || !out || n == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const NSUInteger bytes = (NSUInteger)n * sizeof(float);
    id<MTLBuffer> xb = [g_device newBufferWithBytes:x
                                             length:bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> yb = [g_device newBufferWithLength:bytes
                                             options:MTLResourceStorageModeShared];
    if (!xb || !yb) {
        fprintf(stderr, "qw3: Metal rmsnorm buffer allocation failed\n");
        return 0;
    }

    struct {
        uint32_t n;
        float eps;
    } args = { n, eps };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_rmsnorm_plain_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:xb offset:0 atIndex:1];
    [enc setBuffer:yb offset:0 atIndex:2];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_rmsnorm_plain_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal rmsnorm command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }

    memcpy(out, yb.contents, bytes);
    return 1;
}

int qw3_metal_rmsnorm_weight_f32(const float *x, uint64_t weight_offset,
                                 float *out, uint32_t n, float eps) {
    if (!x || !out || n == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const uint64_t weight_bytes = (uint64_t)n * sizeof(float);
    if (!g_model_map_ptr || weight_offset > g_model_map_size ||
        weight_bytes > g_model_map_size - weight_offset) {
        fprintf(stderr, "qw3: Metal RMSNorm weight is outside mapped model\n");
        return 0;
    }
    id<MTLBuffer> wb = [g_device newBufferWithBytes:(const uint8_t *)g_model_map_ptr + weight_offset
                                             length:(NSUInteger)weight_bytes
                                            options:MTLResourceStorageModeShared];

    const NSUInteger bytes = (NSUInteger)n * sizeof(float);
    id<MTLBuffer> xb = [g_device newBufferWithBytes:x
                                             length:bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> yb = [g_device newBufferWithLength:bytes
                                             options:MTLResourceStorageModeShared];
    if (!xb || !yb) {
        fprintf(stderr, "qw3: Metal weighted RMSNorm buffer allocation failed\n");
        return 0;
    }

    struct {
        uint32_t n;
        float eps;
    } args = { n, eps };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_rmsnorm_weight_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:xb offset:0 atIndex:1];
    [enc setBuffer:wb offset:0 atIndex:2];
    [enc setBuffer:yb offset:0 atIndex:3];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_rmsnorm_weight_f32_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal weighted RMSNorm command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }

    memcpy(out, yb.contents, bytes);
    return 1;
}

int qw3_metal_embed_q8_0(uint64_t tensor_offset, uint32_t token,
                         uint32_t n_embd, float *out) {
    if (!out || n_embd == 0 || (n_embd % 32) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const uint64_t row_bytes = (uint64_t)(n_embd / 32) * 34ull;
    const uint64_t row_offset = tensor_offset + (uint64_t)token * row_bytes;
    uint64_t inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(row_offset, row_bytes, &inner);
    if (!wb) {
        fprintf(stderr, "qw3: Metal embedding row is outside mapped model views\n");
        return 0;
    }

    id<MTLBuffer> outb = [g_device newBufferWithLength:(NSUInteger)n_embd * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!outb) {
        fprintf(stderr, "qw3: Metal embedding output allocation failed\n");
        return 0;
    }
    struct {
        uint32_t n_embd;
        uint32_t row_bytes;
    } args = { n_embd, (uint32_t)row_bytes };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_embed_q8_0_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)inner atIndex:1];
    [enc setBuffer:outb offset:0 atIndex:2];
    NSUInteger threads = g_embed_q8_0_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake(n_embd, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal embedding command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, outb.contents, (size_t)n_embd * sizeof(float));
    return 1;
}

int qw3_metal_matvec_q8_0(uint64_t tensor_offset, const float *x,
                          uint32_t n_in, uint32_t n_out, float *out) {
    if (!x || !out || n_in == 0 || n_out == 0 || (n_in % 32) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const uint64_t row_bytes = (uint64_t)(n_in / 32) * 34ull;
    const uint64_t tensor_bytes = row_bytes * (uint64_t)n_out;
    if (!g_model_map_ptr || tensor_offset > g_model_map_size ||
        tensor_bytes > g_model_map_size - tensor_offset) {
        fprintf(stderr, "qw3: Metal q8_0 matvec tensor is outside mapped model\n");
        return 0;
    }
    id<MTLBuffer> wb = [g_device newBufferWithBytes:(const uint8_t *)g_model_map_ptr + tensor_offset
                                             length:(NSUInteger)tensor_bytes
                                            options:MTLResourceStorageModeShared];

    id<MTLBuffer> xb = [g_device newBufferWithBytes:x
                                             length:(NSUInteger)n_in * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> outb = [g_device newBufferWithLength:(NSUInteger)n_out * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!xb || !outb) {
        fprintf(stderr, "qw3: Metal q8_0 matvec buffer allocation failed\n");
        return 0;
    }
    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
    } args = { n_in, n_out, (uint32_t)row_bytes };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_matvec_q8_0_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:0 atIndex:1];
    [enc setBuffer:xb offset:0 atIndex:2];
    [enc setBuffer:outb offset:0 atIndex:3];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_matvec_q8_0_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(n_out, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal q8_0 matvec command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, outb.contents, (size_t)n_out * sizeof(float));
    return 1;
}

int qw3_metal_matvec_iq4_xs_expert(uint64_t tensor_offset, uint32_t expert,
                                   const float *x, uint32_t n_in,
                                   uint32_t n_out, float *out) {
    if (!x || !out || n_in == 0 || n_out == 0 || (n_in % 256) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const uint64_t row_bytes = (uint64_t)(n_in / 256) * 136ull;
    const uint64_t expert_bytes = row_bytes * (uint64_t)n_out;
    const uint64_t expert_offset = tensor_offset + (uint64_t)expert * expert_bytes;
    if (!g_model_map_ptr || expert_offset > g_model_map_size ||
        expert_bytes > g_model_map_size - expert_offset) {
        fprintf(stderr, "qw3: Metal iq4_xs expert tensor is outside mapped model\n");
        return 0;
    }
    uint64_t expert_inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(expert_offset,
                                                expert_bytes,
                                                &expert_inner);

    id<MTLBuffer> xb = [g_device newBufferWithBytes:x
                                             length:(NSUInteger)n_in * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> outb = [g_device newBufferWithLength:(NSUInteger)n_out * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!wb || !xb || !outb) {
        fprintf(stderr, "qw3: Metal iq4_xs expert buffer allocation failed\n");
        return 0;
    }
    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
    } args = { n_in, n_out, (uint32_t)row_bytes };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_matvec_iq4_xs_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)expert_inner atIndex:1];
    [enc setBuffer:xb offset:0 atIndex:2];
    [enc setBuffer:outb offset:0 atIndex:3];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_matvec_iq4_xs_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(n_out, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal iq4_xs expert command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, outb.contents, (size_t)n_out * sizeof(float));
    return 1;
}

int qw3_metal_matvec_q6_k(uint64_t tensor_offset, const float *x,
                          uint32_t n_in, uint32_t n_out, float *out) {
    if (!x || !out || n_in == 0 || n_out == 0 || (n_in % 256) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const uint64_t row_bytes = (uint64_t)(n_in / 256) * 210ull;
    const uint64_t tensor_bytes = row_bytes * (uint64_t)n_out;
    if (!g_model_map_ptr || tensor_offset > g_model_map_size ||
        tensor_bytes > g_model_map_size - tensor_offset) {
        fprintf(stderr, "qw3: Metal q6_K matvec tensor is outside mapped model\n");
        return 0;
    }
    id<MTLBuffer> wb = [g_device newBufferWithBytes:(const uint8_t *)g_model_map_ptr + tensor_offset
                                             length:(NSUInteger)tensor_bytes
                                            options:MTLResourceStorageModeShared];

    id<MTLBuffer> xb = [g_device newBufferWithBytes:x
                                             length:(NSUInteger)n_in * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> outb = [g_device newBufferWithLength:(NSUInteger)n_out * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!wb || !xb || !outb) {
        fprintf(stderr, "qw3: Metal q6_K matvec buffer allocation failed\n");
        return 0;
    }
    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
    } args = { n_in, n_out, (uint32_t)row_bytes };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_matvec_q6_k_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:0 atIndex:1];
    [enc setBuffer:xb offset:0 atIndex:2];
    [enc setBuffer:outb offset:0 atIndex:3];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = 64;
    [enc dispatchThreadgroups:MTLSizeMake((n_out + 3u) / 4u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal q6_K matvec command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, outb.contents, (size_t)n_out * sizeof(float));
    return 1;
}

int qw3_metal_matvec_q6_k_expert(uint64_t tensor_offset, uint32_t expert,
                                 const float *x, uint32_t n_in,
                                 uint32_t n_out, float *out) {
    if (!x || !out || n_in == 0 || n_out == 0 || (n_in % 256) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const uint64_t row_bytes = (uint64_t)(n_in / 256) * 210ull;
    const uint64_t expert_bytes = row_bytes * (uint64_t)n_out;
    const uint64_t expert_offset = tensor_offset + (uint64_t)expert * expert_bytes;
    if (!g_model_map_ptr || expert_offset > g_model_map_size ||
        expert_bytes > g_model_map_size - expert_offset) {
        fprintf(stderr, "qw3: Metal q6_K expert tensor is outside mapped model\n");
        return 0;
    }
    uint64_t expert_inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(expert_offset,
                                                expert_bytes,
                                                &expert_inner);

    id<MTLBuffer> xb = [g_device newBufferWithBytes:x
                                             length:(NSUInteger)n_in * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> outb = [g_device newBufferWithLength:(NSUInteger)n_out * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!wb || !xb || !outb) {
        fprintf(stderr, "qw3: Metal q6_K expert buffer allocation failed\n");
        return 0;
    }
    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
    } args = { n_in, n_out, (uint32_t)row_bytes };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_matvec_q6_k_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)expert_inner atIndex:1];
    [enc setBuffer:xb offset:0 atIndex:2];
    [enc setBuffer:outb offset:0 atIndex:3];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = 64;
    [enc dispatchThreadgroups:MTLSizeMake((n_out + 3u) / 4u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal q6_K expert command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, outb.contents, (size_t)n_out * sizeof(float));
    return 1;
}

int qw3_metal_matvec_iq3_s_expert(uint64_t tensor_offset, uint32_t expert,
                                  const float *x, uint32_t n_in,
                                  uint32_t n_out, float *out) {
    if (!x || !out || n_in == 0 || n_out == 0 || (n_in % 256) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const uint64_t row_bytes = (uint64_t)(n_in / 256) * 110ull;
    const uint64_t expert_bytes = row_bytes * (uint64_t)n_out;
    const uint64_t expert_offset = tensor_offset + (uint64_t)expert * expert_bytes;
    if (!g_model_map_ptr || expert_offset > g_model_map_size ||
        expert_bytes > g_model_map_size - expert_offset) {
        fprintf(stderr, "qw3: Metal iq3_s expert tensor is outside mapped model\n");
        return 0;
    }
    uint64_t expert_inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(expert_offset,
                                                expert_bytes,
                                                &expert_inner);

    id<MTLBuffer> xb = [g_device newBufferWithBytes:x
                                             length:(NSUInteger)n_in * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> outb = [g_device newBufferWithLength:(NSUInteger)n_out * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> kgb = qw3_metal_iq3s_kgrid_buffer();
    if (!wb || !xb || !outb || !kgb) {
        fprintf(stderr, "qw3: Metal iq3_s expert buffer allocation failed\n");
        return 0;
    }
    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
    } args = { n_in, n_out, (uint32_t)row_bytes };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_matvec_iq3_s_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)expert_inner atIndex:1];
    [enc setBuffer:xb offset:0 atIndex:2];
    [enc setBuffer:outb offset:0 atIndex:3];
    [enc setBuffer:kgb offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_matvec_iq3_s_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(n_out, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal iq3_s expert command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, outb.contents, (size_t)n_out * sizeof(float));
    return 1;
}

int qw3_metal_matvec_f32(uint64_t tensor_offset, const float *x,
                         uint32_t n_in, uint32_t n_out, float *out) {
    if (!x || !out || n_in == 0 || n_out == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const uint64_t tensor_bytes = (uint64_t)n_in * (uint64_t)n_out * sizeof(float);
    if (!g_model_map_ptr || tensor_offset > g_model_map_size ||
        tensor_bytes > g_model_map_size - tensor_offset) {
        fprintf(stderr, "qw3: Metal f32 matvec tensor is outside mapped model\n");
        return 0;
    }
    id<MTLBuffer> wb = [g_device newBufferWithBytes:(const uint8_t *)g_model_map_ptr + tensor_offset
                                             length:(NSUInteger)tensor_bytes
                                            options:MTLResourceStorageModeShared];

    id<MTLBuffer> xb = [g_device newBufferWithBytes:x
                                             length:(NSUInteger)n_in * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> outb = [g_device newBufferWithLength:(NSUInteger)n_out * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!xb || !outb) {
        fprintf(stderr, "qw3: Metal f32 matvec buffer allocation failed\n");
        return 0;
    }
    struct {
        uint32_t n_in;
        uint32_t n_out;
    } args = { n_in, n_out };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_matvec_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:0 atIndex:1];
    [enc setBuffer:xb offset:0 atIndex:2];
    [enc setBuffer:outb offset:0 atIndex:3];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_matvec_q8_0_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(n_out, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal f32 matvec command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, outb.contents, (size_t)n_out * sizeof(float));
    return 1;
}

int qw3_metal_deltanet_conv1d_zero(uint64_t weight_offset, const float *qkv,
                                   uint32_t n_channels, float *out) {
    if (!qkv || !out || n_channels == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const uint64_t weight_bytes = (uint64_t)n_channels * 4ull * sizeof(float);
    if (!g_model_map_ptr || weight_offset > g_model_map_size ||
        weight_bytes > g_model_map_size - weight_offset) {
        fprintf(stderr, "qw3: Metal DeltaNet conv weight is outside mapped model\n");
        return 0;
    }
    id<MTLBuffer> wb = [g_device newBufferWithBytes:(const uint8_t *)g_model_map_ptr + weight_offset
                                             length:(NSUInteger)weight_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> qkvb = [g_device newBufferWithBytes:qkv
                                                length:(NSUInteger)n_channels * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> outb = [g_device newBufferWithLength:(NSUInteger)n_channels * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!qkvb || !outb) {
        fprintf(stderr, "qw3: Metal DeltaNet conv buffer allocation failed\n");
        return 0;
    }
    struct {
        uint32_t n_channels;
    } args = { n_channels };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_deltanet_conv1d_zero_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:0 atIndex:1];
    [enc setBuffer:qkvb offset:0 atIndex:2];
    [enc setBuffer:outb offset:0 atIndex:3];
    NSUInteger threads = g_deltanet_conv1d_zero_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake(n_channels, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal DeltaNet conv command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, outb.contents, (size_t)n_channels * sizeof(float));
    return 1;
}

int qw3_metal_deltanet_conv1d_step(uint64_t weight_offset, const float *qkv,
                                   const float *state_in, uint32_t n_channels,
                                   float *out, float *state_out) {
    if (!qkv || !state_in || !out || !state_out || n_channels == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const uint32_t state_n = n_channels * 3u;
    const uint64_t weight_bytes = (uint64_t)n_channels * 4ull * sizeof(float);
    if (!g_model_map_ptr || weight_offset > g_model_map_size ||
        weight_bytes > g_model_map_size - weight_offset) {
        fprintf(stderr, "qw3: Metal DeltaNet conv-step weight is outside mapped model\n");
        return 0;
    }
    id<MTLBuffer> wb = [g_device newBufferWithBytes:(const uint8_t *)g_model_map_ptr + weight_offset
                                             length:(NSUInteger)weight_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> qkvb = [g_device newBufferWithBytes:qkv
                                                length:(NSUInteger)n_channels * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> sb = [g_device newBufferWithBytes:state_in
                                             length:(NSUInteger)state_n * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> outb = [g_device newBufferWithLength:(NSUInteger)n_channels * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> soutb = [g_device newBufferWithLength:(NSUInteger)state_n * sizeof(float)
                                                options:MTLResourceStorageModeShared];
    if (!qkvb || !sb || !outb || !soutb) {
        fprintf(stderr, "qw3: Metal DeltaNet conv-step buffer allocation failed\n");
        return 0;
    }
    struct {
        uint32_t n_channels;
    } args = { n_channels };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_deltanet_conv1d_step_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:0 atIndex:1];
    [enc setBuffer:qkvb offset:0 atIndex:2];
    [enc setBuffer:sb offset:0 atIndex:3];
    [enc setBuffer:outb offset:0 atIndex:4];
    [enc setBuffer:soutb offset:0 atIndex:5];
    NSUInteger threads = g_deltanet_conv1d_step_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake(n_channels, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal DeltaNet conv-step command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, outb.contents, (size_t)n_channels * sizeof(float));
    memcpy(state_out, soutb.contents, (size_t)state_n * sizeof(float));
    return 1;
}

int qw3_metal_l2norm_heads(const float *x, uint32_t n_heads,
                           uint32_t head_dim, float eps, float *out) {
    if (!x || !out || n_heads == 0 || head_dim == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const uint32_t n = n_heads * head_dim;
    id<MTLBuffer> xb = [g_device newBufferWithBytes:x
                                             length:(NSUInteger)n * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> outb = [g_device newBufferWithLength:(NSUInteger)n * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!xb || !outb) {
        fprintf(stderr, "qw3: Metal L2Norm heads buffer allocation failed\n");
        return 0;
    }
    struct {
        uint32_t head_dim;
        float eps;
    } args = { head_dim, eps };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_l2norm_heads_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:xb offset:0 atIndex:1];
    [enc setBuffer:outb offset:0 atIndex:2];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_l2norm_heads_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal L2Norm heads command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, outb.contents, (size_t)n * sizeof(float));
    return 1;
}

int qw3_metal_rope_heads(const float *x, uint32_t n_heads, uint32_t head_dim,
                         uint32_t rope_dim, int32_t pos, float theta,
                         float *out) {
    if (!x || !out || n_heads == 0 || head_dim == 0 || rope_dim == 0 ||
        rope_dim > head_dim || (rope_dim % 2) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const uint32_t n = n_heads * head_dim;
    id<MTLBuffer> xb = [g_device newBufferWithBytes:x
                                             length:(NSUInteger)n * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> outb = [g_device newBufferWithLength:(NSUInteger)n * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!xb || !outb) {
        fprintf(stderr, "qw3: Metal RoPE heads buffer allocation failed\n");
        return 0;
    }
    struct {
        uint32_t n_heads;
        uint32_t head_dim;
        uint32_t rope_dim;
        int32_t pos;
        float theta;
    } args = { n_heads, head_dim, rope_dim, pos, theta };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_rope_heads_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:xb offset:0 atIndex:1];
    [enc setBuffer:outb offset:0 atIndex:2];
    NSUInteger threads = g_rope_heads_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal RoPE heads command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, outb.contents, (size_t)n * sizeof(float));
    return 1;
}

int qw3_metal_gqa_single_token_inner(const float *gate, const float *v,
                                     uint32_t n_heads, uint32_t n_kv_heads,
                                     uint32_t head_dim, float *out) {
    if (!gate || !v || !out || n_heads == 0 || n_kv_heads == 0 ||
        head_dim == 0 || (n_heads % n_kv_heads) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const uint32_t gate_n = n_heads * head_dim;
    const uint32_t v_n = n_kv_heads * head_dim;
    id<MTLBuffer> gb = [g_device newBufferWithBytes:gate
                                             length:(NSUInteger)gate_n * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> vb = [g_device newBufferWithBytes:v
                                             length:(NSUInteger)v_n * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> outb = [g_device newBufferWithLength:(NSUInteger)gate_n * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!gb || !vb || !outb) {
        fprintf(stderr, "qw3: Metal GQA single-token inner buffer allocation failed\n");
        return 0;
    }
    struct {
        uint32_t n_heads;
        uint32_t n_kv_heads;
        uint32_t head_dim;
    } args = { n_heads, n_kv_heads, head_dim };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_gqa_single_token_inner_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:gb offset:0 atIndex:1];
    [enc setBuffer:vb offset:0 atIndex:2];
    [enc setBuffer:outb offset:0 atIndex:3];
    NSUInteger threads = g_gqa_single_token_inner_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake(gate_n, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal GQA single-token inner command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, outb.contents, (size_t)gate_n * sizeof(float));
    return 1;
}

int qw3_metal_gqa_attend2_inner(const float *q, const float *gate,
                                const float *k_cache, const float *v_cache,
                                uint32_t n_heads, uint32_t n_kv_heads,
                                uint32_t head_dim, float *out) {
    if (!q || !gate || !k_cache || !v_cache || !out || n_heads == 0 ||
        n_kv_heads == 0 || head_dim == 0 || (n_heads % n_kv_heads) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const uint32_t q_n = n_heads * head_dim;
    const uint32_t kv_n = 2u * n_kv_heads * head_dim;
    id<MTLBuffer> qb = [g_device newBufferWithBytes:q
                                             length:(NSUInteger)q_n * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> gb = [g_device newBufferWithBytes:gate
                                             length:(NSUInteger)q_n * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> kb = [g_device newBufferWithBytes:k_cache
                                             length:(NSUInteger)kv_n * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> vb = [g_device newBufferWithBytes:v_cache
                                             length:(NSUInteger)kv_n * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> outb = [g_device newBufferWithLength:(NSUInteger)q_n * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!qb || !gb || !kb || !vb || !outb) {
        fprintf(stderr, "qw3: Metal GQA attend2 inner buffer allocation failed\n");
        return 0;
    }
    struct {
        uint32_t n_heads;
        uint32_t n_kv_heads;
        uint32_t head_dim;
    } args = { n_heads, n_kv_heads, head_dim };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_gqa_attend2_inner_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:qb offset:0 atIndex:1];
    [enc setBuffer:gb offset:0 atIndex:2];
    [enc setBuffer:kb offset:0 atIndex:3];
    [enc setBuffer:vb offset:0 atIndex:4];
    [enc setBuffer:outb offset:0 atIndex:5];
    NSUInteger threads = g_gqa_attend2_inner_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake(q_n, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal GQA attend2 inner command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, outb.contents, (size_t)q_n * sizeof(float));
    return 1;
}

int qw3_metal_gqa_attend_n_inner(const float *q, const float *gate,
                                 const float *k_cache, const float *v_cache,
                                 uint32_t n_ctx, uint32_t n_heads,
                                 uint32_t n_kv_heads, uint32_t head_dim,
                                 float *out) {
    if (!q || !gate || !k_cache || !v_cache || !out || n_ctx == 0 ||
        n_heads == 0 || n_kv_heads == 0 || head_dim == 0 ||
        (n_heads % n_kv_heads) != 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const uint32_t q_n = n_heads * head_dim;
    const uint32_t kv_n = n_ctx * n_kv_heads * head_dim;
    id<MTLBuffer> qb = [g_device newBufferWithBytes:q
                                             length:(NSUInteger)q_n * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> gb = [g_device newBufferWithBytes:gate
                                             length:(NSUInteger)q_n * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> kb = [g_device newBufferWithBytes:k_cache
                                             length:(NSUInteger)kv_n * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> vb = [g_device newBufferWithBytes:v_cache
                                             length:(NSUInteger)kv_n * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> outb = [g_device newBufferWithLength:(NSUInteger)q_n * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!qb || !gb || !kb || !vb || !outb) {
        fprintf(stderr, "qw3: Metal GQA attend-n inner buffer allocation failed\n");
        return 0;
    }
    struct {
        uint32_t n_ctx;
        uint32_t n_heads;
        uint32_t n_kv_heads;
        uint32_t head_dim;
    } args = { n_ctx, n_heads, n_kv_heads, head_dim };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_gqa_attend_n_inner_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:qb offset:0 atIndex:1];
    [enc setBuffer:gb offset:0 atIndex:2];
    [enc setBuffer:kb offset:0 atIndex:3];
    [enc setBuffer:vb offset:0 atIndex:4];
    [enc setBuffer:outb offset:0 atIndex:5];
    NSUInteger threads = ((NSUInteger)head_dim + 31u) & ~(NSUInteger)31u;
    if (threads < 32u) threads = 32u;
    if (threads > 256u ||
        threads > g_gqa_attend_n_inner_pipeline.maxTotalThreadsPerThreadgroup) {
        qw3_metal_end_compute_encoder(cb, enc);
        return 0;
    }
    [enc setThreadgroupMemoryLength:64u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(n_kv_heads, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal GQA attend-n inner command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, outb.contents, (size_t)q_n * sizeof(float));
    return 1;
}

int qw3_metal_deltanet_recur_zero(const float *q, const float *k,
                                  const float *v, const float *beta,
                                  uint32_t q_heads, uint32_t v_heads,
                                  uint32_t head_dim, float *state_out,
                                  float *core_out) {
    if (!q || !k || !v || !beta || !state_out || !core_out ||
        q_heads == 0 || v_heads == 0 || head_dim == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const NSUInteger qk_bytes = (NSUInteger)q_heads * head_dim * sizeof(float);
    const NSUInteger v_bytes = (NSUInteger)v_heads * head_dim * sizeof(float);
    const NSUInteger beta_bytes = (NSUInteger)v_heads * sizeof(float);
    const NSUInteger state_bytes = (NSUInteger)v_heads * head_dim * head_dim * sizeof(float);
    id<MTLBuffer> qb = [g_device newBufferWithBytes:q
                                             length:qk_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> kb = [g_device newBufferWithBytes:k
                                             length:qk_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> vb = [g_device newBufferWithBytes:v
                                             length:v_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [g_device newBufferWithBytes:beta
                                             length:beta_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> sb = [g_device newBufferWithLength:state_bytes
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> cb_out = [g_device newBufferWithLength:v_bytes
                                                 options:MTLResourceStorageModeShared];
    if (!qb || !kb || !vb || !bb || !sb || !cb_out) {
        fprintf(stderr, "qw3: Metal DeltaNet recurrence buffer allocation failed\n");
        return 0;
    }
    struct {
        uint32_t q_heads;
        uint32_t v_heads;
        uint32_t head_dim;
    } args = { q_heads, v_heads, head_dim };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_deltanet_recur_zero_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:qb offset:0 atIndex:1];
    [enc setBuffer:kb offset:0 atIndex:2];
    [enc setBuffer:vb offset:0 atIndex:3];
    [enc setBuffer:bb offset:0 atIndex:4];
    [enc setBuffer:sb offset:0 atIndex:5];
    [enc setBuffer:cb_out offset:0 atIndex:6];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_deltanet_recur_zero_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(v_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal DeltaNet recurrence command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(state_out, sb.contents, state_bytes);
    memcpy(core_out, cb_out.contents, v_bytes);
    return 1;
}

int qw3_metal_deltanet_recur(const float *state_in, const float *q,
                             const float *k, const float *v,
                             const float *beta, const float *gamma,
                             uint32_t q_heads, uint32_t v_heads,
                             uint32_t head_dim, float *state_out,
                             float *core_out) {
    if (!state_in || !q || !k || !v || !beta || !gamma ||
        !state_out || !core_out || q_heads == 0 || v_heads == 0 ||
        head_dim == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const NSUInteger qk_bytes = (NSUInteger)q_heads * head_dim * sizeof(float);
    const NSUInteger v_bytes = (NSUInteger)v_heads * head_dim * sizeof(float);
    const NSUInteger gates_bytes = (NSUInteger)v_heads * sizeof(float);
    const NSUInteger state_bytes = (NSUInteger)v_heads * head_dim * head_dim * sizeof(float);
    id<MTLBuffer> sib = [g_device newBufferWithBytes:state_in
                                              length:state_bytes
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> qb = [g_device newBufferWithBytes:q
                                             length:qk_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> kb = [g_device newBufferWithBytes:k
                                             length:qk_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> vb = [g_device newBufferWithBytes:v
                                             length:v_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [g_device newBufferWithBytes:beta
                                             length:gates_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> gb = [g_device newBufferWithBytes:gamma
                                             length:gates_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> sob = [g_device newBufferWithLength:state_bytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> cb_out = [g_device newBufferWithLength:v_bytes
                                                 options:MTLResourceStorageModeShared];
    if (!sib || !qb || !kb || !vb || !bb || !gb || !sob || !cb_out) {
        fprintf(stderr, "qw3: Metal DeltaNet recurrent buffer allocation failed\n");
        return 0;
    }
    struct {
        uint32_t q_heads;
        uint32_t v_heads;
        uint32_t head_dim;
    } args = { q_heads, v_heads, head_dim };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_deltanet_recur_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:sib offset:0 atIndex:1];
    [enc setBuffer:qb offset:0 atIndex:2];
    [enc setBuffer:kb offset:0 atIndex:3];
    [enc setBuffer:vb offset:0 atIndex:4];
    [enc setBuffer:bb offset:0 atIndex:5];
    [enc setBuffer:gb offset:0 atIndex:6];
    [enc setBuffer:sob offset:0 atIndex:7];
    [enc setBuffer:cb_out offset:0 atIndex:8];
    NSUInteger threads = g_deltanet_recur_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 128) threads = 128;
    [enc dispatchThreads:MTLSizeMake(head_dim, v_heads, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal DeltaNet recurrent command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(state_out, sob.contents, state_bytes);
    memcpy(core_out, cb_out.contents, v_bytes);
    return 1;
}

int qw3_metal_deltanet_gated_rmsnorm(uint64_t norm_weight_offset,
                                     const float *core, const float *z,
                                     uint32_t v_heads, uint32_t head_dim,
                                     float eps, float *out) {
    if (!core || !z || !out || v_heads == 0 || head_dim == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const uint64_t weight_bytes = (uint64_t)head_dim * sizeof(float);
    if (!g_model_map_ptr || norm_weight_offset > g_model_map_size ||
        weight_bytes > g_model_map_size - norm_weight_offset) {
        fprintf(stderr, "qw3: Metal DeltaNet norm weight is outside mapped model\n");
        return 0;
    }
    id<MTLBuffer> wb = [g_device newBufferWithBytes:(const uint8_t *)g_model_map_ptr + norm_weight_offset
                                             length:(NSUInteger)weight_bytes
                                            options:MTLResourceStorageModeShared];
    const NSUInteger bytes = (NSUInteger)v_heads * head_dim * sizeof(float);
    id<MTLBuffer> cb = [g_device newBufferWithBytes:core
                                             length:bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> zb = [g_device newBufferWithBytes:z
                                             length:bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> outb = [g_device newBufferWithLength:bytes
                                               options:MTLResourceStorageModeShared];
    if (!cb || !zb || !outb) {
        fprintf(stderr, "qw3: Metal DeltaNet gated RMSNorm buffer allocation failed\n");
        return 0;
    }
    struct {
        uint32_t v_heads;
        uint32_t head_dim;
        float eps;
    } args = { v_heads, head_dim, eps };

    int owned = 0;
    id<MTLCommandBuffer> cmd = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cmd);
    [enc setComputePipelineState:g_deltanet_gated_rmsnorm_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:0 atIndex:1];
    [enc setBuffer:cb offset:0 atIndex:2];
    [enc setBuffer:zb offset:0 atIndex:3];
    [enc setBuffer:outb offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_deltanet_gated_rmsnorm_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(v_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cmd, enc);
    if (!qw3_metal_finish_command_buffer(cmd, owned, "operation")) return 0;
    if (cmd.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal DeltaNet gated RMSNorm command failed: %s\n",
                [[cmd.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, outb.contents, bytes);
    return 1;
}

int qw3_metal_residual_rmsnorm_weight_f32(const float *x, const float *residual,
                                          uint64_t weight_offset, float *out,
                                          uint32_t n, float eps) {
    if (!x || !residual || !out || n == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const uint64_t weight_bytes = (uint64_t)n * sizeof(float);
    if (!g_model_map_ptr || weight_offset > g_model_map_size ||
        weight_bytes > g_model_map_size - weight_offset) {
        fprintf(stderr, "qw3: Metal residual RMSNorm weight is outside mapped model\n");
        return 0;
    }
    id<MTLBuffer> wb = [g_device newBufferWithBytes:(const uint8_t *)g_model_map_ptr + weight_offset
                                             length:(NSUInteger)weight_bytes
                                            options:MTLResourceStorageModeShared];

    const NSUInteger bytes = (NSUInteger)n * sizeof(float);
    id<MTLBuffer> xb = [g_device newBufferWithBytes:x
                                             length:bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> rb = [g_device newBufferWithBytes:residual
                                             length:bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> yb = [g_device newBufferWithLength:bytes
                                             options:MTLResourceStorageModeShared];
    if (!xb || !rb || !yb) {
        fprintf(stderr, "qw3: Metal residual RMSNorm buffer allocation failed\n");
        return 0;
    }
    struct {
        uint32_t n;
        float eps;
    } args = { n, eps };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_residual_rmsnorm_weight_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:xb offset:0 atIndex:1];
    [enc setBuffer:rb offset:0 atIndex:2];
    [enc setBuffer:wb offset:0 atIndex:3];
    [enc setBuffer:yb offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_residual_rmsnorm_weight_f32_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal residual RMSNorm command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, yb.contents, bytes);
    return 1;
}

int qw3_metal_silu_mul(const float *a, const float *b, uint32_t n, float *out) {
    if (!a || !b || !out || n == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const NSUInteger bytes = (NSUInteger)n * sizeof(float);
    id<MTLBuffer> ab = [g_device newBufferWithBytes:a
                                             length:bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [g_device newBufferWithBytes:b
                                             length:bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> ob = [g_device newBufferWithLength:bytes
                                             options:MTLResourceStorageModeShared];
    if (!ab || !bb || !ob) {
        fprintf(stderr, "qw3: Metal silu_mul buffer allocation failed\n");
        return 0;
    }
    struct {
        uint32_t n;
        float scale;
    } args = { n, 1.0f };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_silu_mul_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:ab offset:0 atIndex:1];
    [enc setBuffer:bb offset:0 atIndex:2];
    [enc setBuffer:ob offset:0 atIndex:3];
    NSUInteger threads = g_silu_mul_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal silu_mul command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, ob.contents, bytes);
    return 1;
}

int qw3_metal_scale(const float *x, uint32_t n, float scale, float *out) {
    if (!x || !out || n == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const NSUInteger bytes = (NSUInteger)n * sizeof(float);
    id<MTLBuffer> xb = [g_device newBufferWithBytes:x
                                             length:bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> ob = [g_device newBufferWithLength:bytes
                                             options:MTLResourceStorageModeShared];
    if (!xb || !ob) {
        fprintf(stderr, "qw3: Metal scale buffer allocation failed\n");
        return 0;
    }
    struct {
        uint32_t n;
        float scale;
    } args = { n, scale };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_scale_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:xb offset:0 atIndex:1];
    [enc setBuffer:ob offset:0 atIndex:2];
    NSUInteger threads = g_scale_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal scale command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, ob.contents, bytes);
    return 1;
}

int qw3_metal_argmax(const float *x, uint32_t n, uint32_t *idx_out,
                     float *val_out) {
    if (!x || !idx_out || !val_out || n == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    const uint32_t threads_u32 = 256;
    const uint32_t n_blocks = (n + threads_u32 - 1) / threads_u32;
    const NSUInteger input_bytes = (NSUInteger)n * sizeof(float);
    const NSUInteger vals_bytes = (NSUInteger)n_blocks * sizeof(float);
    const NSUInteger idxs_bytes = (NSUInteger)n_blocks * sizeof(uint32_t);

    id<MTLBuffer> xb = [g_device newBufferWithBytes:x
                                             length:input_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> valsb = [g_device newBufferWithLength:vals_bytes
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> idxsb = [g_device newBufferWithLength:idxs_bytes
                                                options:MTLResourceStorageModeShared];
    if (!xb || !valsb || !idxsb) {
        fprintf(stderr, "qw3: Metal argmax buffer allocation failed\n");
        return 0;
    }
    struct {
        uint32_t n;
    } args = { n };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_argmax_blocks_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:xb offset:0 atIndex:1];
    [enc setBuffer:valsb offset:0 atIndex:2];
    [enc setBuffer:idxsb offset:0 atIndex:3];
    [enc setThreadgroupMemoryLength:(NSUInteger)threads_u32 * sizeof(float)
                            atIndex:0];
    [enc setThreadgroupMemoryLength:(NSUInteger)threads_u32 * sizeof(uint32_t)
                            atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake(n_blocks, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads_u32, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal argmax command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }

    const float *vals = (const float *)valsb.contents;
    const uint32_t *idxs = (const uint32_t *)idxsb.contents;
    uint32_t best_idx = idxs[0];
    float best_val = vals[0];
    for (uint32_t i = 1; i < n_blocks; i++) {
        uint32_t idx = idxs[i];
        float val = vals[i];
        if (val > best_val || (val == best_val && idx < best_idx)) {
            best_val = val;
            best_idx = idx;
        }
    }
    *idx_out = best_idx;
    *val_out = best_val;
    return 1;
}
