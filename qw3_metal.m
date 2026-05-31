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
static int g_batch_concurrent;
static NSMutableArray<id<MTLCommandBuffer>> *g_pending_cbs;

static NSMutableArray<id<MTLBuffer>> *g_model_buffers;
static NSMutableDictionary<NSString *, id<MTLBuffer>> *g_model_temp_buffers;
static id<MTLBuffer> g_iq3s_kgrid_buffer;
static id<MTLBuffer> g_iq3s_expanded_kgrid_buffer;
static id<MTLLibrary> g_library;
static id<MTLComputePipelineState> g_rmsnorm_plain_pipeline;
static id<MTLComputePipelineState> g_rmsnorm_weight_f32_pipeline;
static id<MTLComputePipelineState> g_rmsnorm_weight_f32_rows_pipeline;
static id<MTLComputePipelineState> g_rmsnorm_weight_f32_rows_to_out_pipeline;
static id<MTLComputePipelineState> g_embed_q8_0_pipeline;
static id<MTLComputePipelineState> g_embed_q8_0_batch_pipeline;
static id<MTLComputePipelineState> g_matvec_q8_0_pipeline;
static id<MTLComputePipelineState> g_matmul_q8_0_batch4_pipeline;
static id<MTLComputePipelineState> g_matmul_q8_0_mm_pipeline;
static id<MTLComputePipelineState> g_matmul_q8_0_mm_bc_pipeline;
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
static id<MTLComputePipelineState> g_moe_iq3_s_prefill_batch_pipeline;
static id<MTLComputePipelineState> g_moe_down_iq4_xs_prefill_reduce_pipeline;
static id<MTLComputePipelineState> g_moe_down_q6_k_prefill_reduce_pipeline;
static id<MTLComputePipelineState> g_moe_topk_expert_map_pipeline;
static id<MTLComputePipelineState> g_moe_iq3_s_prefill_mapped_pipeline;
static id<MTLComputePipelineState> g_moe_iq3_s_prefill_pair_mapped_pipeline;
static id<MTLComputePipelineState> g_moe_swiglu_slots_to_hidden_pipeline;
static id<MTLComputePipelineState> g_moe_swiglu_slots_to_hidden_f16_pipeline;
static id<MTLComputePipelineState> g_moe_down_iq4_xs_prefill_mapped_pipeline;
static id<MTLComputePipelineState> g_moe_down_iq4_xs_prefill_mapped_f16_pipeline;
static id<MTLComputePipelineState> g_moe_down_prefill_reduce_slots_pipeline;
static id<MTLComputePipelineState> g_matvec_f32_pipeline;
static id<MTLComputePipelineState> g_matvec_f32_pair_pipeline;
static id<MTLComputePipelineState> g_matvec_f32_fast_pipeline;
static id<MTLComputePipelineState> g_matmul_f32_batch4_pipeline;
static id<MTLComputePipelineState> g_matmul_f32_pair_batch4_pipeline;
static id<MTLComputePipelineState> g_deltanet_conv1d_zero_pipeline;
static id<MTLComputePipelineState> g_deltanet_conv1d_step_pipeline;
static id<MTLComputePipelineState> g_deltanet_conv1d_batch_pipeline;
static id<MTLComputePipelineState> g_deltanet_conv1d_batch_state_pipeline;
static id<MTLComputePipelineState> g_l2norm_heads_pipeline;
static id<MTLComputePipelineState> g_l2norm_qk_batch_pipeline;
static id<MTLComputePipelineState> g_gqa_q_norm_gate_pipeline;
static id<MTLComputePipelineState> g_gqa_k_norm_pipeline;
static id<MTLComputePipelineState> g_gqa_q_norm_gate_batch_pipeline;
static id<MTLComputePipelineState> g_gqa_k_norm_batch_pipeline;
static id<MTLComputePipelineState> g_rope_heads_pipeline;
static id<MTLComputePipelineState> g_rope_heads_batch_pipeline;
static id<MTLComputePipelineState> g_gqa_single_token_inner_pipeline;
static id<MTLComputePipelineState> g_gqa_attend2_inner_pipeline;
static id<MTLComputePipelineState> g_gqa_attend_n_inner_pipeline;
static id<MTLComputePipelineState> g_gqa_prefill_attend_inner_pipeline;
static id<MTLComputePipelineState> g_gqa_prefill_write_cache_pipeline;
static id<MTLComputePipelineState> g_gqa_prefill_cached_attend_inner_pipeline;
static id<MTLComputePipelineState> g_gqa_kv_quant_q8_pipeline;
static id<MTLComputePipelineState> g_gqa_attend_n_q8_inner_pipeline;
static id<MTLComputePipelineState> g_gqa_attend_n_q8_split_partial_pipeline;
static id<MTLComputePipelineState> g_gqa_attend_n_q8_split_reduce_pipeline;
static id<MTLComputePipelineState> g_deltanet_recur_zero_pipeline;
static id<MTLComputePipelineState> g_deltanet_recur_pipeline;
static id<MTLComputePipelineState> g_deltanet_recur_scratch_gates_pipeline;
static id<MTLComputePipelineState> g_deltanet_prepare_scratch_gates_pipeline;
static id<MTLComputePipelineState> g_deltanet_recur_scratch_gates_tiled_pipeline;
static id<MTLComputePipelineState> g_deltanet_fused_gdn_scratch_pipeline;
static id<MTLComputePipelineState> g_deltanet_batch_fused_gdn_pipeline;
static id<MTLComputePipelineState> g_deltanet_gated_rmsnorm_pipeline;
static id<MTLComputePipelineState> g_residual_rmsnorm_weight_f32_pipeline;
static id<MTLComputePipelineState> g_residual_rmsnorm_update_x0_pipeline;
static id<MTLComputePipelineState> g_residual_rmsnorm_batch_update_x0_pipeline;
static id<MTLComputePipelineState> g_silu_mul_pipeline;
static id<MTLComputePipelineState> g_scale_pipeline;
static id<MTLComputePipelineState> g_argmax_blocks_pipeline;
static id<MTLComputePipelineState> g_add_moe_to_x0_pipeline;
static id<MTLComputePipelineState> g_silu_mul_offsets_pipeline;
static id<MTLComputePipelineState> g_silu_mul_rows_offsets_pipeline;
static id<MTLComputePipelineState> g_scale_x1_scalar_add_x0_pipeline;
static id<MTLComputePipelineState> g_scale_x1_add_x0_pipeline;
static id<MTLComputePipelineState> g_scale_scratch_add_x0_pipeline;
static id<MTLComputePipelineState> g_sigmoid_scale_scratch_add_x0_rows_pipeline;
static id<MTLComputePipelineState> g_router_top8_pipeline;
static id<MTLComputePipelineState> g_router_top8_batch_pipeline;
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
    if (getenv("QW3_METAL_RETAINED_COMMAND_BUFFERS") == NULL) {
        return [g_queue commandBufferWithUnretainedReferences];
    }
    return [g_queue commandBuffer];
}

static id<MTLComputeCommandEncoder> qw3_metal_compute_encoder(id<MTLCommandBuffer> cb) {
    if (g_batch_cb && cb == g_batch_cb) {
        if (!g_batch_enc) {
            g_batch_enc = g_batch_concurrent ?
                [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] :
                [cb computeCommandEncoder];
        }
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
@property(nonatomic, strong) id<MTLBuffer> gqaAttnPartial;
@property(nonatomic, strong) id<MTLBuffer> routerIds;
@property(nonatomic, strong) id<MTLBuffer> routerWeights;
@property(nonatomic, strong) id<MTLBuffer> argmaxVals;
@property(nonatomic, strong) id<MTLBuffer> argmaxIdxs;
@property(nonatomic, strong) id<MTLBuffer> prefillTokens;
@property(nonatomic, strong) id<MTLBuffer> prefillX0;
@property(nonatomic, strong) id<MTLBuffer> prefillX1;
@property(nonatomic, strong) id<MTLBuffer> prefillScratch;
@property(nonatomic, strong) id<MTLBuffer> moeExpertCounts;
@property(nonatomic, strong) id<MTLBuffer> moePairIds;
@property(nonatomic, strong) id<MTLBuffer> prefillMoeGate;
@property(nonatomic, strong) id<MTLBuffer> prefillMoeUp;
@property(nonatomic, strong) id<MTLBuffer> prefillMoeMidF16;
@property(nonatomic, strong) id<MTLBuffer> prefillMoeDown;
@property(nonatomic) qw3_metal_session_info info;
@property(nonatomic) uint32_t ctxSize;
@property(nonatomic) uint32_t vocabSize;
@property(nonatomic) uint32_t pos;
@property(nonatomic) uint32_t prefillCap;
@property(nonatomic) BOOL gqaKvQ8;
@property(nonatomic) BOOL gqaSplitQ8;
@property(nonatomic) uint32_t gqaMaxQ8Splits;
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
    QW3_METAL_N_EXPERT = 256,
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
            "struct qw3_rmsnorm_rows_args { uint n; float eps; uint n_rows; };\n"
            "kernel void qw3_rmsnorm_weight_f32_rows(constant qw3_rmsnorm_rows_args &args,\n"
            "                                        device float *x,\n"
            "                                        device const float *w,\n"
            "                                        threadgroup float *sh,\n"
            "                                        uint row [[threadgroup_position_in_grid]],\n"
            "                                        ushort tid [[thread_index_in_threadgroup]],\n"
            "                                        ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                        ushort lane [[thread_index_in_simdgroup]],\n"
            "                                        ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (row >= args.n_rows) return;\n"
            "    device float *xr = x + uint64_t(row) * args.n;\n"
            "    float ss = 0.0f;\n"
            "    for (uint i = tid; i < args.n; i += nt) ss += xr[i] * xr[i];\n"
            "    ss = simd_sum(ss);\n"
            "    if (lane == 0) sh[simd_idx] = ss;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    ss = lane < 32 ? sh[lane] : 0.0f;\n"
            "    ss = simd_sum(ss);\n"
            "    float scale = rsqrt(ss / float(args.n) + args.eps);\n"
            "    for (uint i = tid; i < args.n; i += nt) xr[i] = xr[i] * scale * w[i];\n"
            "}\n"
            "kernel void qw3_rmsnorm_weight_f32_rows_to_out(constant qw3_rmsnorm_rows_args &args,\n"
            "                                               device const float *x,\n"
            "                                               device const float *w,\n"
            "                                               device float *out,\n"
            "                                               threadgroup float *sh,\n"
            "                                               uint row [[threadgroup_position_in_grid]],\n"
            "                                               ushort tid [[thread_index_in_threadgroup]],\n"
            "                                               ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                               ushort lane [[thread_index_in_simdgroup]],\n"
            "                                               ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (row >= args.n_rows) return;\n"
            "    device const float *xr = x + uint64_t(row) * args.n;\n"
            "    device float *yr = out + uint64_t(row) * args.n;\n"
            "    float ss = 0.0f;\n"
            "    for (uint i = tid; i < args.n; i += nt) ss += xr[i] * xr[i];\n"
            "    ss = simd_sum(ss);\n"
            "    if (lane == 0) sh[simd_idx] = ss;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    ss = lane < 32 ? sh[lane] : 0.0f;\n"
            "    ss = simd_sum(ss);\n"
            "    float scale = rsqrt(ss / float(args.n) + args.eps);\n"
            "    for (uint i = tid; i < args.n; i += nt) yr[i] = xr[i] * scale * w[i];\n"
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
            "struct qw3_embed_q8_0_batch_args { uint n_embd; uint row_bytes; uint n_tokens; };\n"
            "kernel void qw3_embed_q8_0_batch(constant qw3_embed_q8_0_batch_args &args,\n"
            "                                 device const uchar *weights,\n"
            "                                 device const uint *tokens,\n"
            "                                 device float *out,\n"
            "                                 uint gid [[thread_position_in_grid]]) {\n"
            "    uint total = args.n_tokens * args.n_embd;\n"
            "    if (gid >= total) return;\n"
            "    uint t = gid / args.n_embd;\n"
            "    uint i = gid - t * args.n_embd;\n"
            "    uint token = tokens[t];\n"
            "    device const uchar *row = weights + uint64_t(token) * uint64_t(args.row_bytes);\n"
            "    uint block = i / 32u;\n"
            "    uint lane = i & 31u;\n"
            "    device const uchar *blk = row + uint64_t(block) * 34ull;\n"
            "    float d = float(*((device const half *)blk));\n"
            "    char q = *((device const char *)(blk + 2u + lane));\n"
            "    out[uint64_t(t) * args.n_embd + i] = d * float(q);\n"
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
            "struct qw3_matmul_q8_0_batch_args { uint n_in; uint n_out; uint row_bytes; uint n_tokens; uint in_offset; uint in_stride; uint out_offset; uint out_stride; };\n"
            "kernel void qw3_matmul_q8_0_batch4(constant qw3_matmul_q8_0_batch_args &args,\n"
            "                                   device const uchar *weights,\n"
            "                                   device const float *x,\n"
            "                                   device float *out,\n"
            "                                   uint2 group [[threadgroup_position_in_grid]],\n"
            "                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                   ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    uint row = group.x * 4u + uint(simd_idx);\n"
            "    if (row >= args.n_out) return;\n"
            "    uint t0 = group.y * 4u;\n"
            "    device const uchar *wr = weights + uint64_t(row) * uint64_t(args.row_bytes);\n"
            "    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;\n"
            "    uint n_blocks = args.n_in / 32u;\n"
            "    for (uint b = uint(lane); b < n_blocks; b += 32u) {\n"
            "        device const uchar *blk = wr + uint64_t(b) * 34ull;\n"
            "        float d = float(*((device const half *)blk));\n"
            "        uint xb = b * 32u;\n"
            "        for (uint i = 0; i < 32u; i++) {\n"
            "            float wv = d * float(*((device const char *)(blk + 2u + i)));\n"
            "            uint xi = xb + i;\n"
            "            if (t0 + 0u < args.n_tokens) s0 += wv * x[uint64_t(t0 + 0u) * args.in_stride + args.in_offset + xi];\n"
            "            if (t0 + 1u < args.n_tokens) s1 += wv * x[uint64_t(t0 + 1u) * args.in_stride + args.in_offset + xi];\n"
            "            if (t0 + 2u < args.n_tokens) s2 += wv * x[uint64_t(t0 + 2u) * args.in_stride + args.in_offset + xi];\n"
            "            if (t0 + 3u < args.n_tokens) s3 += wv * x[uint64_t(t0 + 3u) * args.in_stride + args.in_offset + xi];\n"
            "        }\n"
            "    }\n"
            "    s0 = simd_sum(s0); s1 = simd_sum(s1); s2 = simd_sum(s2); s3 = simd_sum(s3);\n"
            "    if (lane == 0) {\n"
            "        if (t0 + 0u < args.n_tokens) out[uint64_t(t0 + 0u) * args.out_stride + args.out_offset + row] = s0;\n"
            "        if (t0 + 1u < args.n_tokens) out[uint64_t(t0 + 1u) * args.out_stride + args.out_offset + row] = s1;\n"
            "        if (t0 + 2u < args.n_tokens) out[uint64_t(t0 + 2u) * args.out_stride + args.out_offset + row] = s2;\n"
            "        if (t0 + 3u < args.n_tokens) out[uint64_t(t0 + 3u) * args.out_stride + args.out_offset + row] = s3;\n"
            "    }\n"
            "}\n"
            "constant bool qw3_mm_q8_bc_out [[function_constant(700)]];\n"
            "struct qw3_block_q8_0 { half d; char qs[32]; };\n"
            "struct qw3_matmul_q8_0_mm_args { uint n_in; uint n_out; uint row_bytes; uint n_tokens; uint in_stride; uint out_stride; };\n"
            "static inline void qw3_dequant_q8_0_16(device const qw3_block_q8_0 *xb, short il, thread half4x4 &reg) {\n"
            "    const float d = float(xb->d);\n"
            "    float4x4 tmp;\n"
            "    for (short i = 0; i < 16; i++) tmp[i / 4][i % 4] = float(xb->qs[i + 16 * il]) * d;\n"
            "    reg = half4x4(tmp);\n"
            "}\n"
            "kernel void qw3_matmul_q8_0_mm(constant qw3_matmul_q8_0_mm_args &args,\n"
            "                                device const char *weights,\n"
            "                                device const char *xin,\n"
            "                                device char *yout,\n"
            "                                threadgroup char *shmem [[threadgroup(0)]],\n"
            "                                uint3 tgpig [[threadgroup_position_in_grid]],\n"
            "                                ushort tiitg [[thread_index_in_threadgroup]],\n"
            "                                ushort sgitg [[simdgroup_index_in_threadgroup]]) {\n"
            "    threadgroup half *sa = (threadgroup half *)shmem;\n"
            "    threadgroup half *sb = (threadgroup half *)(shmem + 4096);\n"
            "    constexpr int NR0 = 64;\n"
            "    constexpr int NR1 = 32;\n"
            "    constexpr int NK = 32;\n"
            "    constexpr int NL0 = NK / 16;\n"
            "    constexpr int NL1 = NK / 8;\n"
            "    const int K = int(args.n_in);\n"
            "    const int M = int(args.n_out);\n"
            "    const int N = int(args.n_tokens);\n"
            "    const int r0 = int(tgpig.y) * NR0;\n"
            "    const int r1 = int(tgpig.x) * NR1;\n"
            "    const int nr0 = min(M - r0, NR0);\n"
            "    const int nr1 = min(N - r1, NR1);\n"
            "    const int lr0 = min(int(tiitg) / NL0, nr0 - 1);\n"
            "    const int lr1 = min(int(tiitg) / NL1, nr1 - 1);\n"
            "    const short il0 = short(tiitg % NL0);\n"
            "    short il = il0;\n"
            "    device const qw3_block_q8_0 *wblk = (device const qw3_block_q8_0 *)(weights + uint64_t(args.row_bytes) * uint64_t(r0 + lr0));\n"
            "    const short iy = short(8 * (tiitg % NL1));\n"
            "    device const float *yin = (device const float *)xin + uint64_t(args.in_stride) * uint64_t(r1 + lr1) + uint64_t(iy);\n"
            "    simdgroup_half8x8 ma[4];\n"
            "    simdgroup_half8x8 mb[2];\n"
            "    simdgroup_float8x8 mc[8];\n"
            "    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);\n"
            "    for (int loop_k = 0; loop_k < K; loop_k += NK) {\n"
            "        half4x4 temp_a;\n"
            "        qw3_dequant_q8_0_16(wblk, il, temp_a);\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        for (short i = 0; i < 16; i++) {\n"
            "            const short sx = short(2 * il0 + i / 8);\n"
            "            const short sy = short((tiitg / NL0) / 8);\n"
            "            const short lx = short((tiitg / NL0) % 8);\n"
            "            const short ly = short(i % 8);\n"
            "            const short ib = short(8 * sx + sy);\n"
            "            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];\n"
            "        }\n"
            "        for (short i = 0; i < 8; i++) {\n"
            "            const short sx = short(tiitg % NL1);\n"
            "            const short sy = short((tiitg / NL1) / 8);\n"
            "            const short lx = i;\n"
            "            const short ly = short((tiitg / NL1) % 8);\n"
            "            const short ib = short(4 * sx + sy);\n"
            "            *(sb + 64 * ib + 8 * ly + lx) = half(yin[i]);\n"
            "        }\n"
            "        il = short((il + 2 < 2) ? il + 2 : il % 2);\n"
            "        wblk = (il < 2) ? wblk + 1 : wblk;\n"
            "        yin += NK;\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        threadgroup const half *lsma = sa + 4 * 64 * (sgitg % 2);\n"
            "        threadgroup const half *lsmb = sb + 2 * 64 * (sgitg / 2);\n"
            "        for (short ik = 0; ik < NK / 8; ik++) {\n"
            "            simdgroup_barrier(mem_flags::mem_none);\n"
            "            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);\n"
            "            simdgroup_barrier(mem_flags::mem_none);\n"
            "            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);\n"
            "            simdgroup_barrier(mem_flags::mem_none);\n"
            "            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);\n"
            "            lsma += 8 * 64;\n"
            "            lsmb += 4 * 64;\n"
            "        }\n"
            "    }\n"
            "    device float *dst = (device float *)yout;\n"
            "    if (!qw3_mm_q8_bc_out || (r0 + NR0 <= M && r1 + NR1 <= N)) {\n"
            "        device float *C = dst + uint64_t(r0 + 32 * (sgitg & 1)) + uint64_t(r1 + 16 * (sgitg >> 1)) * uint64_t(args.out_stride);\n"
            "        for (short i = 0; i < 8; i++) simdgroup_store(mc[i], C + 8 * (i % 4) + uint64_t(8 * (i / 4)) * uint64_t(args.out_stride), args.out_stride, 0, false);\n"
            "    } else {\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        threadgroup float *tmp = ((threadgroup float *)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;\n"
            "        for (short i = 0; i < 8; i++) simdgroup_store(mc[i], tmp + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        if (sgitg == 0) {\n"
            "            for (int j = int(tiitg); j < nr1; j += NR1) {\n"
            "                device float *D = dst + uint64_t(r0) + uint64_t(r1 + j) * uint64_t(args.out_stride);\n"
            "                threadgroup float *C = ((threadgroup float *)shmem) + j * NR0;\n"
            "                int i = 0;\n"
            "                device float4 *D4 = (device float4 *)D;\n"
            "                threadgroup float4 *C4 = (threadgroup float4 *)C;\n"
            "                for (; i < nr0 / 4; i++) D4[i] = C4[i];\n"
            "                i *= 4;\n"
            "                for (; i < nr0; i++) D[i] = C[i];\n"
            "            }\n"
            "        }\n"
            "    }\n"
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
            "struct qw3_matmul_f32_batch_args { uint n_in; uint n_out; uint n_tokens; uint in_offset; uint in_stride; uint out_offset; uint out_stride; };\n"
            "kernel void qw3_matmul_f32_batch4(constant qw3_matmul_f32_batch_args &args,\n"
            "                                  device const float *weights,\n"
            "                                  device const float *x,\n"
            "                                  device float *out,\n"
            "                                  uint2 group [[threadgroup_position_in_grid]],\n"
            "                                  ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                  ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    uint row = group.x * 4u + uint(simd_idx);\n"
            "    if (row >= args.n_out) return;\n"
            "    uint t0 = group.y * 4u;\n"
            "    device const float *wr = weights + uint64_t(row) * args.n_in;\n"
            "    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;\n"
            "    for (uint i = uint(lane); i < args.n_in; i += 32u) {\n"
            "        float wv = wr[i];\n"
            "        if (t0 + 0u < args.n_tokens) s0 += wv * x[uint64_t(t0 + 0u) * args.in_stride + args.in_offset + i];\n"
            "        if (t0 + 1u < args.n_tokens) s1 += wv * x[uint64_t(t0 + 1u) * args.in_stride + args.in_offset + i];\n"
            "        if (t0 + 2u < args.n_tokens) s2 += wv * x[uint64_t(t0 + 2u) * args.in_stride + args.in_offset + i];\n"
            "        if (t0 + 3u < args.n_tokens) s3 += wv * x[uint64_t(t0 + 3u) * args.in_stride + args.in_offset + i];\n"
            "    }\n"
            "    s0 = simd_sum(s0); s1 = simd_sum(s1); s2 = simd_sum(s2); s3 = simd_sum(s3);\n"
            "    if (lane == 0) {\n"
            "        if (t0 + 0u < args.n_tokens) out[uint64_t(t0 + 0u) * args.out_stride + args.out_offset + row] = s0;\n"
            "        if (t0 + 1u < args.n_tokens) out[uint64_t(t0 + 1u) * args.out_stride + args.out_offset + row] = s1;\n"
            "        if (t0 + 2u < args.n_tokens) out[uint64_t(t0 + 2u) * args.out_stride + args.out_offset + row] = s2;\n"
            "        if (t0 + 3u < args.n_tokens) out[uint64_t(t0 + 3u) * args.out_stride + args.out_offset + row] = s3;\n"
            "    }\n"
            "}\n"
            "struct qw3_matmul_f32_pair_batch_args { uint n_in; uint n_out; uint n_tokens; uint out_a_offset; uint out_b_offset; uint out_stride; };\n"
            "kernel void qw3_matmul_f32_pair_batch4(constant qw3_matmul_f32_pair_batch_args &args,\n"
            "                                       device const float *weights_a,\n"
            "                                       device const float *weights_b,\n"
            "                                       device const float *x,\n"
            "                                       device float *out,\n"
            "                                       uint2 group [[threadgroup_position_in_grid]],\n"
            "                                       ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                       ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    uint row = group.x * 4u + uint(simd_idx);\n"
            "    if (row >= args.n_out) return;\n"
            "    uint t0 = group.y * 4u;\n"
            "    device const float *wa = weights_a + uint64_t(row) * args.n_in;\n"
            "    device const float *wb = weights_b + uint64_t(row) * args.n_in;\n"
            "    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;\n"
            "    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;\n"
            "    for (uint i = uint(lane); i < args.n_in; i += 32u) {\n"
            "        float wva = wa[i];\n"
            "        float wvb = wb[i];\n"
            "        if (t0 + 0u < args.n_tokens) { float xv = x[uint64_t(t0 + 0u) * args.n_in + i]; a0 += wva * xv; b0 += wvb * xv; }\n"
            "        if (t0 + 1u < args.n_tokens) { float xv = x[uint64_t(t0 + 1u) * args.n_in + i]; a1 += wva * xv; b1 += wvb * xv; }\n"
            "        if (t0 + 2u < args.n_tokens) { float xv = x[uint64_t(t0 + 2u) * args.n_in + i]; a2 += wva * xv; b2 += wvb * xv; }\n"
            "        if (t0 + 3u < args.n_tokens) { float xv = x[uint64_t(t0 + 3u) * args.n_in + i]; a3 += wva * xv; b3 += wvb * xv; }\n"
            "    }\n"
            "    a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);\n"
            "    b0 = simd_sum(b0); b1 = simd_sum(b1); b2 = simd_sum(b2); b3 = simd_sum(b3);\n"
            "    if (lane == 0) {\n"
            "        uint64_t base = uint64_t(t0 + 0u) * args.out_stride;\n"
            "        if (t0 + 0u < args.n_tokens) { out[base + args.out_a_offset + row] = a0; out[base + args.out_b_offset + row] = b0; }\n"
            "        base = uint64_t(t0 + 1u) * args.out_stride;\n"
            "        if (t0 + 1u < args.n_tokens) { out[base + args.out_a_offset + row] = a1; out[base + args.out_b_offset + row] = b1; }\n"
            "        base = uint64_t(t0 + 2u) * args.out_stride;\n"
            "        if (t0 + 2u < args.n_tokens) { out[base + args.out_a_offset + row] = a2; out[base + args.out_b_offset + row] = b2; }\n"
            "        base = uint64_t(t0 + 3u) * args.out_stride;\n"
            "        if (t0 + 3u < args.n_tokens) { out[base + args.out_a_offset + row] = a3; out[base + args.out_b_offset + row] = b3; }\n"
            "    }\n"
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
            "struct qw3_conv1d_batch_args { uint n_channels; uint n_tokens; uint qkv_offset; uint conv_offset; uint stride; };\n"
            "kernel void qw3_deltanet_conv1d_batch(constant qw3_conv1d_batch_args &args,\n"
            "                                      device const float *w,\n"
            "                                      device float *scratch,\n"
            "                                      device float *state,\n"
            "                                      uint gid [[thread_position_in_grid]]) {\n"
            "    uint total = args.n_channels * args.n_tokens;\n"
            "    if (gid >= total) return;\n"
            "    uint t = gid / args.n_channels;\n"
            "    uint ch = gid - t * args.n_channels;\n"
            "    device const float *wr = w + uint64_t(ch) * 4ull;\n"
            "    device float *st = state + uint64_t(ch) * 3ull;\n"
            "    float sum = 0.0f;\n"
            "    for (int k = 0; k < 4; k++) {\n"
            "        int idx = int(t) + k - 3;\n"
            "        float xv = idx < 0 ? st[3 + idx] : scratch[uint64_t(uint(idx)) * args.stride + args.qkv_offset + ch];\n"
            "        sum += xv * wr[k];\n"
            "    }\n"
            "    float y = sum / (1.0f + exp(-sum));\n"
            "    scratch[uint64_t(t) * args.stride + args.conv_offset + ch] = y;\n"
            "}\n"
            "kernel void qw3_deltanet_conv1d_batch_state(constant qw3_conv1d_batch_args &args,\n"
            "                                            device const float *scratch,\n"
            "                                            device float *state,\n"
            "                                            uint ch [[thread_position_in_grid]]) {\n"
            "    if (ch >= args.n_channels) return;\n"
            "    device float *st = state + uint64_t(ch) * 3ull;\n"
            "    float s0 = st[0], s1 = st[1], s2 = st[2];\n"
            "    for (int j = 0; j < 3; j++) {\n"
            "        int idx = int(args.n_tokens) - 3 + j;\n"
            "        float v = 0.0f;\n"
            "        if (idx < 0) {\n"
            "            v = (idx == -3) ? s0 : ((idx == -2) ? s1 : s2);\n"
            "        } else {\n"
            "            v = scratch[uint64_t(uint(idx)) * args.stride + args.qkv_offset + ch];\n"
            "        }\n"
            "        st[j] = v;\n"
            "    }\n"
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
            "struct qw3_l2norm_qk_batch_args { uint n_tokens; uint conv_offset; uint stride; uint n_qk_heads; uint head_dim; float eps; };\n"
            "kernel void qw3_l2norm_qk_batch(constant qw3_l2norm_qk_batch_args &args,\n"
            "                                device float *scratch,\n"
            "                                threadgroup float *sh,\n"
            "                                uint group [[threadgroup_position_in_grid]],\n"
            "                                ushort tid [[thread_index_in_threadgroup]],\n"
            "                                ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                ushort lane [[thread_index_in_simdgroup]],\n"
            "                                ushort nt [[threads_per_threadgroup]]) {\n"
            "    uint heads_per_qk = args.n_qk_heads * args.n_tokens;\n"
            "    uint qk = group / heads_per_qk;\n"
            "    uint rem = group - qk * heads_per_qk;\n"
            "    uint t = rem / args.n_qk_heads;\n"
            "    uint head = rem - t * args.n_qk_heads;\n"
            "    if (head >= args.n_qk_heads || t >= args.n_tokens || qk >= 2u) return;\n"
            "    uint qk_n = args.n_qk_heads * args.head_dim;\n"
            "    uint off = args.conv_offset + qk * qk_n + head * args.head_dim;\n"
            "    device float *xh = scratch + uint64_t(t) * args.stride + off;\n"
            "    float ss = 0.0f;\n"
            "    for (uint i = tid; i < args.head_dim; i += nt) ss += xh[i] * xh[i];\n"
            "    ss = simd_sum(ss);\n"
            "    if (lane == 0) sh[simd_idx] = ss;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    ss = lane < 32 ? sh[lane] : 0.0f;\n"
            "    ss = simd_sum(ss);\n"
            "    float scale = 1.0f / max(sqrt(ss), args.eps);\n"
            "    for (uint i = tid; i < args.head_dim; i += nt) xh[i] *= scale;\n"
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
            "struct qw3_gqa_norm_batch_args { uint n_tokens; uint n_heads; uint head_dim; uint in_offset; uint out_offset; uint gate_offset; uint stride; float eps; };\n"
            "kernel void qw3_gqa_q_norm_gate_batch(constant qw3_gqa_norm_batch_args &args,\n"
            "                                      device float *scratch,\n"
            "                                      device const float *w,\n"
            "                                      threadgroup float *sh,\n"
            "                                      uint group [[threadgroup_position_in_grid]],\n"
            "                                      ushort tid [[thread_index_in_threadgroup]],\n"
            "                                      ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                      ushort lane [[thread_index_in_simdgroup]],\n"
            "                                      ushort nt [[threads_per_threadgroup]]) {\n"
            "    uint t = group / args.n_heads;\n"
            "    uint head = group - t * args.n_heads;\n"
            "    if (t >= args.n_tokens || head >= args.n_heads) return;\n"
            "    device float *row = scratch + uint64_t(t) * args.stride;\n"
            "    device const float *qh = row + args.in_offset + uint64_t(head) * uint64_t(args.head_dim) * 2ull;\n"
            "    device const float *gh = qh + args.head_dim;\n"
            "    device float *yo = row + args.out_offset + uint64_t(head) * args.head_dim;\n"
            "    device float *go = row + args.gate_offset + uint64_t(head) * args.head_dim;\n"
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
            "kernel void qw3_gqa_k_norm_batch(constant qw3_gqa_norm_batch_args &args,\n"
            "                                 device float *scratch,\n"
            "                                 device const float *w,\n"
            "                                 threadgroup float *sh,\n"
            "                                 uint group [[threadgroup_position_in_grid]],\n"
            "                                 ushort tid [[thread_index_in_threadgroup]],\n"
            "                                 ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                 ushort lane [[thread_index_in_simdgroup]],\n"
            "                                 ushort nt [[threads_per_threadgroup]]) {\n"
            "    uint t = group / args.n_heads;\n"
            "    uint head = group - t * args.n_heads;\n"
            "    if (t >= args.n_tokens || head >= args.n_heads) return;\n"
            "    device float *row = scratch + uint64_t(t) * args.stride;\n"
            "    device const float *kh = row + args.in_offset + uint64_t(head) * args.head_dim;\n"
            "    device float *yo = row + args.out_offset + uint64_t(head) * args.head_dim;\n"
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
            "struct qw3_rope_batch_args { uint n_tokens; uint n_heads; uint head_dim; uint rope_dim; uint pos0; uint in_offset; uint out_offset; uint stride; float theta; };\n"
            "kernel void qw3_rope_heads_batch(constant qw3_rope_batch_args &args,\n"
            "                                 device float *scratch,\n"
            "                                 uint gid [[thread_position_in_grid]]) {\n"
            "    uint per_tok = args.n_heads * args.head_dim;\n"
            "    uint total = args.n_tokens * per_tok;\n"
            "    if (gid >= total) return;\n"
            "    uint t = gid / per_tok;\n"
            "    uint rem = gid - t * per_tok;\n"
            "    uint h = rem / args.head_dim;\n"
            "    uint i = rem - h * args.head_dim;\n"
            "    device const float *xh = scratch + uint64_t(t) * args.stride + args.in_offset + uint64_t(h) * args.head_dim;\n"
            "    device float *yh = scratch + uint64_t(t) * args.stride + args.out_offset + uint64_t(h) * args.head_dim;\n"
            "    if (i >= args.rope_dim) { yh[i] = xh[i]; return; }\n"
            "    uint p = i & ~1u;\n"
            "    float freq = pow(args.theta, -float(p) / float(args.rope_dim));\n"
            "    float ang = float(args.pos0 + t) * freq;\n"
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
            "struct qw3_gqa_prefill_attn_args { uint n_tokens; uint n_heads; uint n_kv_heads; uint head_dim; uint q_offset; uint gate_offset; uint k_offset; uint v_offset; uint out_offset; uint stride; };\n"
            "kernel void qw3_gqa_prefill_attend_inner(constant qw3_gqa_prefill_attn_args &args,\n"
            "                                        device float *scratch,\n"
            "                                        threadgroup float *sh,\n"
            "                                        uint group [[threadgroup_position_in_grid]],\n"
            "                                        ushort tid [[thread_index_in_threadgroup]],\n"
            "                                        ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                        ushort lane [[thread_index_in_simdgroup]],\n"
            "                                        ushort nt [[threads_per_threadgroup]]) {\n"
            "    uint query = group / args.n_kv_heads;\n"
            "    uint kvh = group - query * args.n_kv_heads;\n"
            "    if (query >= args.n_tokens || kvh >= args.n_kv_heads || args.head_dim > uint(nt)) return;\n"
            "    uint i = uint(tid);\n"
            "    uint group_heads = args.n_heads / args.n_kv_heads;\n"
            "    if (group_heads == 0u || group_heads > 8u) return;\n"
            "    uint first_qh = kvh * group_heads;\n"
            "    device float *qrow = scratch + uint64_t(query) * args.stride;\n"
            "    float scale = rsqrt(float(args.head_dim));\n"
            "    float qv[8];\n"
            "    float max_score[8];\n"
            "    float denom[8];\n"
            "    float acc[8];\n"
            "    for (uint gh = 0; gh < 8u; gh++) {\n"
            "        uint qh = first_qh + gh;\n"
            "        qv[gh] = (gh < group_heads && i < args.head_dim && qh < args.n_heads) ? qrow[args.q_offset + uint64_t(qh) * args.head_dim + i] : 0.0f;\n"
            "        max_score[gh] = -FLT_MAX;\n"
            "        denom[gh] = 0.0f;\n"
            "        acc[gh] = 0.0f;\n"
            "    }\n"
            "    uint n_simd = (uint(nt) + 31u) >> 5u;\n"
            "    for (uint src = 0; src <= query; src++) {\n"
            "        device float *srow = scratch + uint64_t(src) * args.stride;\n"
            "        float kval = (i < args.head_dim) ? srow[args.k_offset + uint64_t(kvh) * args.head_dim + i] : 0.0f;\n"
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
            "        float vv = (i < args.head_dim) ? srow[args.v_offset + uint64_t(kvh) * args.head_dim + i] : 0.0f;\n"
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
            "                float sig = 1.0f / (1.0f + exp(-qrow[args.gate_offset + gid]));\n"
            "                qrow[args.out_offset + gid] = (acc[gh] / denom[gh]) * sig;\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}\n"
            "struct qw3_gqa_prefill_cache_args { uint n_tokens; uint n_kv_heads; uint head_dim; uint pos0; uint ctx_size; uint k_offset; uint v_offset; uint stride; };\n"
            "kernel void qw3_gqa_prefill_write_cache(constant qw3_gqa_prefill_cache_args &args,\n"
            "                                        device const float *scratch,\n"
            "                                        device float *k_cache,\n"
            "                                        device float *v_cache,\n"
            "                                        uint gid [[thread_position_in_grid]]) {\n"
            "    uint kv_n = args.n_kv_heads * args.head_dim;\n"
            "    uint total = args.n_tokens * kv_n;\n"
            "    if (gid >= total) return;\n"
            "    uint t = gid / kv_n;\n"
            "    uint i = gid - t * kv_n;\n"
            "    uint pos = args.pos0 + t;\n"
            "    if (pos >= args.ctx_size) return;\n"
            "    device const float *row = scratch + uint64_t(t) * args.stride;\n"
            "    uint64_t dst = uint64_t(pos) * kv_n + i;\n"
            "    k_cache[dst] = row[args.k_offset + i];\n"
            "    v_cache[dst] = row[args.v_offset + i];\n"
            "}\n"
            "struct qw3_gqa_prefill_cached_attn_args { uint n_tokens; uint n_heads; uint n_kv_heads; uint head_dim; uint pos0; uint ctx_size; uint q_offset; uint gate_offset; uint out_offset; uint stride; };\n"
            "kernel void qw3_gqa_prefill_cached_attend_inner(constant qw3_gqa_prefill_cached_attn_args &args,\n"
            "                                               device float *scratch,\n"
            "                                               device const float *k_cache,\n"
            "                                               device const float *v_cache,\n"
            "                                               threadgroup float *sh,\n"
            "                                               uint group [[threadgroup_position_in_grid]],\n"
            "                                               ushort tid [[thread_index_in_threadgroup]],\n"
            "                                               ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                               ushort lane [[thread_index_in_simdgroup]],\n"
            "                                               ushort nt [[threads_per_threadgroup]]) {\n"
            "    uint query = group / args.n_kv_heads;\n"
            "    uint kvh = group - query * args.n_kv_heads;\n"
            "    if (query >= args.n_tokens || kvh >= args.n_kv_heads || args.head_dim > uint(nt)) return;\n"
            "    uint n_ctx = args.pos0 + query + 1u;\n"
            "    if (n_ctx > args.ctx_size) return;\n"
            "    uint i = uint(tid);\n"
            "    uint group_heads = args.n_heads / args.n_kv_heads;\n"
            "    if (group_heads == 0u || group_heads > 8u) return;\n"
            "    uint first_qh = kvh * group_heads;\n"
            "    uint kv_n = args.n_kv_heads * args.head_dim;\n"
            "    device float *qrow = scratch + uint64_t(query) * args.stride;\n"
            "    float scale = rsqrt(float(args.head_dim));\n"
            "    float qv[8];\n"
            "    float max_score[8];\n"
            "    float denom[8];\n"
            "    float acc[8];\n"
            "    for (uint gh = 0; gh < 8u; gh++) {\n"
            "        uint qh = first_qh + gh;\n"
            "        qv[gh] = (gh < group_heads && i < args.head_dim && qh < args.n_heads) ? qrow[args.q_offset + uint64_t(qh) * args.head_dim + i] : 0.0f;\n"
            "        max_score[gh] = -FLT_MAX;\n"
            "        denom[gh] = 0.0f;\n"
            "        acc[gh] = 0.0f;\n"
            "    }\n"
            "    uint n_simd = (uint(nt) + 31u) >> 5u;\n"
            "    for (uint src = 0; src < n_ctx; src++) {\n"
            "        float kval = (i < args.head_dim) ? k_cache[uint64_t(src) * kv_n + uint64_t(kvh) * args.head_dim + i] : 0.0f;\n"
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
            "        float vv = (i < args.head_dim) ? v_cache[uint64_t(src) * kv_n + uint64_t(kvh) * args.head_dim + i] : 0.0f;\n"
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
            "                float sig = 1.0f / (1.0f + exp(-qrow[args.gate_offset + gid]));\n"
            "                qrow[args.out_offset + gid] = (acc[gh] / denom[gh]) * sig;\n"
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
            "struct qw3_gqa_q8_split_args { uint n_ctx; uint n_heads; uint n_kv_heads; uint head_dim; uint n_splits; };\n"
            "kernel void qw3_gqa_attend_n_q8_split_partial(constant qw3_gqa_q8_split_args &args,\n"
            "                                             device const float *q,\n"
            "                                             device const uchar *k_cache,\n"
            "                                             device const uchar *v_cache,\n"
            "                                             device float *partial,\n"
            "                                             threadgroup float *sh,\n"
            "                                             uint group [[threadgroup_position_in_grid]],\n"
            "                                             ushort tid [[thread_index_in_threadgroup]],\n"
            "                                             ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                             ushort lane [[thread_index_in_simdgroup]],\n"
            "                                             ushort nt [[threads_per_threadgroup]]) {\n"
            "    uint kvh = group % args.n_kv_heads;\n"
            "    uint split = group / args.n_kv_heads;\n"
            "    if (kvh >= args.n_kv_heads || split >= args.n_splits || args.head_dim > uint(nt)) return;\n"
            "    uint span = (args.n_ctx + args.n_splits - 1u) / args.n_splits;\n"
            "    uint t0 = split * span;\n"
            "    uint t1 = min(args.n_ctx, t0 + span);\n"
            "    if (t0 >= t1) return;\n"
            "    uint i = uint(tid);\n"
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
            "    for (uint t = t0; t < t1; t++) {\n"
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
            "        uint stride = args.head_dim + 2u;\n"
            "        for (uint gh = 0; gh < group_heads; gh++) {\n"
            "            uint qh = first_qh + gh;\n"
            "            uint64_t base = (uint64_t(split) * args.n_heads + qh) * stride;\n"
            "            partial[base + i] = acc[gh];\n"
            "            if (i == 0u) {\n"
            "                partial[base + args.head_dim] = denom[gh];\n"
            "                partial[base + args.head_dim + 1u] = max_score[gh];\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}\n"
            "kernel void qw3_gqa_attend_n_q8_split_reduce(constant qw3_gqa_q8_split_args &args,\n"
            "                                            device const float *gate,\n"
            "                                            device const float *partial,\n"
            "                                            device float *out,\n"
            "                                            uint qh [[threadgroup_position_in_grid]],\n"
            "                                            ushort tid [[thread_index_in_threadgroup]]) {\n"
            "    if (qh >= args.n_heads || uint(tid) >= args.head_dim) return;\n"
            "    uint i = uint(tid);\n"
            "    uint stride = args.head_dim + 2u;\n"
            "    float max_score = -FLT_MAX;\n"
            "    float denom = 0.0f;\n"
            "    float acc = 0.0f;\n"
            "    for (uint split = 0; split < args.n_splits; split++) {\n"
            "        uint64_t base = (uint64_t(split) * args.n_heads + qh) * stride;\n"
            "        float local_denom = partial[base + args.head_dim];\n"
            "        float local_max = partial[base + args.head_dim + 1u];\n"
            "        float next_max = max(max_score, local_max);\n"
            "        float prev_scale = exp(max_score - next_max);\n"
            "        float cur_scale = exp(local_max - next_max);\n"
            "        acc = acc * prev_scale + partial[base + i] * cur_scale;\n"
            "        denom = denom * prev_scale + local_denom * cur_scale;\n"
            "        max_score = next_max;\n"
            "    }\n"
            "    uint gid = qh * args.head_dim + i;\n"
            "    float sig = 1.0f / (1.0f + exp(-gate[gid]));\n"
            "    out[gid] = (acc / denom) * sig;\n"
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
            "kernel void qw3_deltanet_prepare_scratch_gates(constant qw3_recur_scratch_args &args,\n"
            "                                                device float *scratch,\n"
            "                                                device const float *dt_bias,\n"
            "                                                device const float *a,\n"
            "                                                uint hv [[thread_position_in_grid]]) {\n"
            "    if (hv >= args.v_heads) return;\n"
            "    float beta_raw = scratch[args.beta_offset + hv];\n"
            "    scratch[args.beta_offset + hv] = 1.0f / (1.0f + exp(-beta_raw));\n"
            "    float alpha_raw = scratch[args.alpha_offset + hv] + dt_bias[hv];\n"
            "    float sp = alpha_raw > 20.0f ? alpha_raw : (alpha_raw < -20.0f ? exp(alpha_raw) : log(1.0f + exp(alpha_raw)));\n"
            "    scratch[args.alpha_offset + hv] = exp(sp * a[hv]);\n"
            "}\n"
            "kernel void qw3_deltanet_recur_scratch_gates_tiled(constant qw3_recur_scratch_args &args,\n"
            "                                                   device const float *state_in,\n"
            "                                                   device const float *q,\n"
            "                                                   device const float *k,\n"
            "                                                   device const float *v,\n"
            "                                                   device const float *scratch,\n"
            "                                                   device const float *dt_bias,\n"
            "                                                   device const float *a,\n"
            "                                                   device float *state_out,\n"
            "                                                   device float *core_out,\n"
            "                                                   uint2 group [[threadgroup_position_in_grid]],\n"
            "                                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                                   ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    uint hv = group.y;\n"
            "    uint j = group.x * 4u + uint(simd_idx);\n"
            "    if (hv >= args.v_heads || j >= args.head_dim) return;\n"
            "    uint hk = hv % args.q_heads;\n"
            "    uint state_n = args.head_dim * args.head_dim;\n"
            "    device const float *qh = q + uint64_t(hk) * args.head_dim;\n"
            "    device const float *kh = k + uint64_t(hk) * args.head_dim;\n"
            "    device const float *vh = v + uint64_t(hv) * args.head_dim;\n"
            "    device const float *sin = state_in + uint64_t(hv) * state_n;\n"
            "    device float *sout = state_out + uint64_t(hv) * state_n;\n"
            "    float b = 0.0f;\n"
            "    float g = 0.0f;\n"
            "    if (lane == 0) {\n"
            "        b = scratch[args.beta_offset + hv];\n"
            "        g = scratch[args.alpha_offset + hv];\n"
            "    }\n"
            "    b = simd_broadcast(b, 0);\n"
            "    g = simd_broadcast(g, 0);\n"
            "    uint i0 = uint(lane) * 4u;\n"
            "    uint state_col = j * args.head_dim + i0;\n"
            "    float4 sv = *((device const float4 *)(sin + state_col));\n"
            "    float4 kv = *((device const float4 *)(kh + i0));\n"
            "    float sk = simd_sum(dot(sv, kv));\n"
            "    float d = b * (vh[j] - sk * g);\n"
            "    sv = sv * g + kv * d;\n"
            "    *((device float4 *)(sout + state_col)) = sv;\n"
            "    float4 qv = *((device const float4 *)(qh + i0));\n"
            "    float out = simd_sum(dot(sv, qv));\n"
            "    if (lane == 0) core_out[uint64_t(hv) * args.head_dim + j] = out * rsqrt(float(args.head_dim));\n"
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
            "    for (uint i = 0; i < args.head_dim; i++) sk += sin[i * args.head_dim + j] * kh[i];\n"
            "    float d = b * (vh[j] - sk * g);\n"
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
            "struct qw3_batch_gdn_args { uint q_heads; uint v_heads; uint head_dim; uint n_tokens; uint conv_offset; uint z_offset; uint alpha_offset; uint beta_offset; uint inner_offset; uint stride; float eps; };\n"
            "kernel void qw3_deltanet_batch_fused_gdn(constant qw3_batch_gdn_args &args,\n"
            "                                        device float *state,\n"
            "                                        device float *scratch,\n"
            "                                        device const float *dt_bias,\n"
            "                                        device const float *a,\n"
            "                                        device const float *w,\n"
            "                                        threadgroup float *sh,\n"
            "                                        uint hv [[threadgroup_position_in_grid]],\n"
            "                                        ushort j [[thread_index_in_threadgroup]],\n"
            "                                        ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                        ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    if (hv >= args.v_heads || j >= args.head_dim) return;\n"
            "    uint hk = hv % args.q_heads;\n"
            "    uint qk_n = args.q_heads * args.head_dim;\n"
            "    uint state_n = args.head_dim * args.head_dim;\n"
            "    device float *st = state + uint64_t(hv) * state_n;\n"
            "    for (uint t = 0; t < args.n_tokens; t++) {\n"
            "        uint64_t base = uint64_t(t) * args.stride;\n"
            "        device const float *qh = scratch + base + args.conv_offset + uint64_t(hk) * args.head_dim;\n"
            "        device const float *kh = scratch + base + args.conv_offset + qk_n + uint64_t(hk) * args.head_dim;\n"
            "        device const float *vh = scratch + base + args.conv_offset + 2u * qk_n + uint64_t(hv) * args.head_dim;\n"
            "        if (j == 0) {\n"
            "            float beta_raw = scratch[base + args.beta_offset + hv];\n"
            "            sh[0] = 1.0f / (1.0f + exp(-beta_raw));\n"
            "            float alpha_raw = scratch[base + args.alpha_offset + hv] + dt_bias[hv];\n"
            "            float sp = alpha_raw > 20.0f ? alpha_raw : (alpha_raw < -20.0f ? exp(alpha_raw) : log(1.0f + exp(alpha_raw)));\n"
            "            sh[1] = exp(sp * a[hv]);\n"
            "        }\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        float b = sh[0];\n"
            "        float g = sh[1];\n"
            "        float sk = 0.0f;\n"
            "        for (uint i = 0; i < args.head_dim; i++) sk += st[j * args.head_dim + i] * kh[i];\n"
            "        float d = b * (vh[j] - sk * g);\n"
            "        float sum = 0.0f;\n"
            "        for (uint i = 0; i < args.head_dim; i++) {\n"
            "            uint idx = j * args.head_dim + i;\n"
            "            float sv = st[idx] * g + kh[i] * d;\n"
            "            st[idx] = sv;\n"
            "            sum += sv * qh[i];\n"
            "        }\n"
            "        float core = sum * rsqrt(float(args.head_dim));\n"
            "        float ss = simd_sum(core * core);\n"
            "        if (lane == 0) sh[2u + simd_idx] = ss;\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        ss = lane < 32 ? sh[2u + lane] : 0.0f;\n"
            "        ss = simd_sum(ss);\n"
            "        float scale = rsqrt(ss / float(args.head_dim) + args.eps);\n"
            "        float zi = scratch[base + args.z_offset + uint64_t(hv) * args.head_dim + j];\n"
            "        float gate = zi / (1.0f + exp(-zi));\n"
            "        scratch[base + args.inner_offset + uint64_t(hv) * args.head_dim + j] = core * scale * w[j] * gate;\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    }\n"
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
            "struct qw3_residual_batch_args { uint n; float eps; uint n_tokens; uint residual_offset; uint residual_stride; };\n"
            "kernel void qw3_residual_rmsnorm_batch_update_x0(constant qw3_residual_batch_args &args,\n"
            "                                                 device float *x0,\n"
            "                                                 device const float *residual,\n"
            "                                                 device const float *w,\n"
            "                                                 device float *y,\n"
            "                                                 threadgroup float *sh,\n"
            "                                                 uint row [[threadgroup_position_in_grid]],\n"
            "                                                 ushort tid [[thread_index_in_threadgroup]],\n"
            "                                                 ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                                 ushort lane [[thread_index_in_simdgroup]],\n"
            "                                                 ushort nt [[threads_per_threadgroup]]) {\n"
            "    if (row >= args.n_tokens) return;\n"
            "    device float *xr = x0 + uint64_t(row) * args.n;\n"
            "    device const float *rr = residual + uint64_t(row) * args.residual_stride + args.residual_offset;\n"
            "    device float *yr = y + uint64_t(row) * args.n;\n"
            "    float ss = 0.0f;\n"
            "    for (uint i = tid; i < args.n; i += nt) { float v = xr[i] + rr[i]; ss += v * v; }\n"
            "    ss = simd_sum(ss);\n"
            "    if (lane == 0) sh[simd_idx] = ss;\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    ss = lane < 32 ? sh[lane] : 0.0f;\n"
            "    ss = simd_sum(ss);\n"
            "    float scale = rsqrt(ss / float(args.n) + args.eps);\n"
            "    for (uint i = tid; i < args.n; i += nt) { float v = xr[i] + rr[i]; xr[i] = v; yr[i] = v * scale * w[i]; }\n"
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
            "struct qw3_rows_offset_args { uint n; uint n_rows; uint stride; uint a_offset; uint b_offset; uint out_offset; };\n"
            "kernel void qw3_silu_mul_rows_offsets(constant qw3_rows_offset_args &args,\n"
            "                                       device const float *scratch,\n"
            "                                       device float *out,\n"
            "                                       uint gid [[thread_position_in_grid]]) {\n"
            "    uint total = args.n * args.n_rows;\n"
            "    if (gid >= total) return;\n"
            "    uint row = gid / args.n;\n"
            "    uint col = gid - row * args.n;\n"
            "    uint base = row * args.stride;\n"
            "    float x = scratch[base + args.a_offset + col];\n"
            "    float y = scratch[base + args.b_offset + col];\n"
            "    out[base + args.out_offset + col] = (x / (1.0f + exp(-x))) * y;\n"
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
            "kernel void qw3_sigmoid_scale_scratch_add_x0_rows(constant qw3_rows_offset_args &args,\n"
            "                                                 device float *x0,\n"
            "                                                 device const float *scratch,\n"
            "                                                 uint gid [[thread_position_in_grid]]) {\n"
            "    uint total = args.n * args.n_rows;\n"
            "    if (gid >= total) return;\n"
            "    uint row = gid / args.n;\n"
            "    uint col = gid - row * args.n;\n"
            "    uint base = row * args.stride;\n"
            "    float raw = scratch[base + args.b_offset];\n"
            "    float scale = 1.0f / (1.0f + exp(-raw));\n"
            "    x0[row * args.n + col] = x0[row * args.n + col] + scratch[base + args.a_offset + col] * scale;\n"
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
            "struct qw3_router_batch_args { uint n_tokens; uint router_offset; uint stride; };\n"
            "kernel void qw3_router_top8_batch(constant qw3_router_batch_args &args,\n"
            "                                  device const float *scratch,\n"
            "                                  device int *ids,\n"
            "                                  device float *weights,\n"
            "                                  uint token [[threadgroup_position_in_grid]],\n"
            "                                  uint tid [[thread_index_in_threadgroup]]) {\n"
            "    if (token >= args.n_tokens) return;\n"
            "    threadgroup float vals[256];\n"
            "    threadgroup int best[256];\n"
            "    threadgroup float selected[8];\n"
            "    threadgroup int selected_ids[8];\n"
            "    device const float *router = scratch + uint64_t(token) * uint64_t(args.stride) + uint64_t(args.router_offset);\n"
            "    device int *row_ids = ids + uint64_t(token) * 8ull;\n"
            "    device float *row_weights = weights + uint64_t(token) * 8ull;\n"
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
            "        if (tid == 0) { row_ids[rank] = best[0]; selected_ids[rank] = best[0]; selected[rank] = vals[0]; }\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    }\n"
            "    if (tid == 0) {\n"
            "        float sum = 0.0f;\n"
            "        for (uint k = 0; k < 8u; k++) { row_weights[k] = exp(selected[k] - selected[0]); sum += row_weights[k]; }\n"
            "        for (uint k = 0; k < 8u; k++) row_weights[k] /= sum;\n"
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
            "                                  device const uchar *kgrid,\n"
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
            "            sum.x += gate_db * float(kgrid[gate_idx1 * 4u + j]) * gate_sign1 * x1;\n"
            "            sum.x += gate_db * float(kgrid[gate_idx2 * 4u + j]) * gate_sign2 * x2;\n"
            "            sum.y += up_db * float(kgrid[up_idx1 * 4u + j]) * up_sign1 * x1;\n"
            "            sum.y += up_db * float(kgrid[up_idx2 * 4u + j]) * up_sign2 * x2;\n"
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
            "                                           device const uchar *kgrid,\n"
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
            "    if (lane < 16u) sh[lane] = qw3_iq4nl_val(uint(lane));\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
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
            "            for (uint j = 0; j < 8u; j++) { uchar va = qsa[j]; uchar vb = qsb[j]; aca += sh[uint(va) & 15u] * xg0[j] + sh[uint(va) >> 4u] * xg0[j + 16u]; acb += sh[uint(vb) & 15u] * xg1[j] + sh[uint(vb) >> 4u] * xg1[j + 16u]; }\n"
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
            "            for (uint j = 0; j < 8u; j++) { uchar va = qsa[j]; uchar vb = qsb[j]; aca += sh[uint(va) & 15u] * xg0[j] + sh[uint(va) >> 4u] * xg0[j + 16u]; acb += sh[uint(vb) & 15u] * xg1[j] + sh[uint(vb) >> 4u] * xg1[j + 16u]; }\n"
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
            "struct qw3_moe_prefill_batch_args { uint n_in; uint n_ff; uint n_embd; uint n_tokens; uint n_active; uint iq3_row_bytes; uint iq3_expert_bytes; uint down_row_bytes; uint down_expert_bytes; uint stride; uint hidden_offset; };\n"
            "kernel void qw3_moe_iq3_s_swiglu_prefill_batch_fast(constant qw3_moe_prefill_batch_args &args,\n"
            "                                                   device const uchar *gate_weights,\n"
            "                                                   device const uchar *up_weights,\n"
            "                                                   device const float *x1,\n"
            "                                                   device float *scratch,\n"
            "                                                   device const uchar *kgrid,\n"
            "                                                   constant int *ids,\n"
            "                                                   threadgroup float *sh,\n"
            "                                                   uint group [[threadgroup_position_in_grid]],\n"
            "                                                   ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                                   ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    (void)sh;\n"
            "    const uint nr0 = 4u;\n"
            "    const uint nsg = 2u;\n"
            "    uint groups_per_pair = (args.n_ff + 7u) / 8u;\n"
            "    uint pair = group / groups_per_pair;\n"
            "    uint row_group = group - pair * groups_per_pair;\n"
            "    uint token = pair / args.n_active;\n"
            "    uint slot = pair - token * args.n_active;\n"
            "    if (token >= args.n_tokens || slot >= args.n_active) return;\n"
            "    uint first_row = (row_group * nsg + uint(simd_idx)) * nr0;\n"
            "    uint expert = uint(ids[pair]);\n"
            "    uint64_t expert_off = uint64_t(expert) * uint64_t(args.iq3_expert_bytes);\n"
            "    device const float *x = x1 + uint64_t(token) * uint64_t(args.n_in);\n"
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
            "        uint base = token * args.stride + args.hidden_offset + slot * args.n_ff;\n"
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
            "kernel void qw3_moe_down_iq4_xs_prefill_batch_reduce_fast(constant qw3_moe_prefill_batch_args &args,\n"
            "                                                         device const uchar *weights,\n"
            "                                                         device const float *scratch,\n"
            "                                                         device float *x0,\n"
            "                                                         constant int *ids,\n"
            "                                                         constant float *router_weights,\n"
            "                                                         threadgroup float *sh,\n"
            "                                                         uint group [[threadgroup_position_in_grid]],\n"
            "                                                         ushort tid [[thread_index_in_threadgroup]],\n"
            "                                                         ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                                         ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    uint row_pair_count = (args.n_embd + 1u) / 2u;\n"
            "    uint token = group / row_pair_count;\n"
            "    uint row0 = (group - token * row_pair_count) * 2u;\n"
            "    uint row1 = row0 + 1u;\n"
            "    uint slot = uint(simd_idx);\n"
            "    bool active = token < args.n_tokens && slot < args.n_active;\n"
            "    uint pair = token * args.n_active + slot;\n"
            "    uint expert = active ? uint(ids[pair]) : 0u;\n"
            "    uint64_t expert_off = uint64_t(expert) * uint64_t(args.down_expert_bytes);\n"
            "    device const float *x = scratch + uint64_t(token) * uint64_t(args.stride) + uint64_t(args.hidden_offset) + uint64_t(slot) * uint64_t(args.n_ff);\n"
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
            "    float scale = active ? router_weights[pair] : 0.0f;\n"
            "    if (lane == 0) { sh[slot] = sum0 * scale; sh[8u + slot] = sum1 * scale; }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    if (tid == 0 && token < args.n_tokens) {\n"
            "        float total0 = 0.0f;\n"
            "        float total1 = 0.0f;\n"
            "        for (uint s = 0; s < args.n_active; s++) { total0 += sh[s]; total1 += sh[8u + s]; }\n"
            "        device float *row = x0 + uint64_t(token) * uint64_t(args.n_embd);\n"
            "        if (row0 < args.n_embd) row[row0] += total0;\n"
            "        if (row1 < args.n_embd) row[row1] += total1;\n"
            "    }\n"
            "}\n"
            "kernel void qw3_moe_down_q6_k_prefill_batch_reduce_fast(constant qw3_moe_prefill_batch_args &args,\n"
            "                                                       device const uchar *weights,\n"
            "                                                       device const float *scratch,\n"
            "                                                       device float *x0,\n"
            "                                                       constant int *ids,\n"
            "                                                       constant float *router_weights,\n"
            "                                                       threadgroup float *sh,\n"
            "                                                       uint group [[threadgroup_position_in_grid]],\n"
            "                                                       ushort tidx [[thread_index_in_threadgroup]],\n"
            "                                                       ushort simd_idx [[simdgroup_index_in_threadgroup]],\n"
            "                                                       ushort lane [[thread_index_in_simdgroup]]) {\n"
            "    uint row_pair_count = (args.n_embd + 1u) / 2u;\n"
            "    uint token = group / row_pair_count;\n"
            "    uint row0 = (group - token * row_pair_count) * 2u;\n"
            "    uint row1 = row0 + 1u;\n"
            "    uint slot = uint(simd_idx);\n"
            "    bool active = token < args.n_tokens && slot < args.n_active;\n"
            "    uint pair = token * args.n_active + slot;\n"
            "    uint expert = active ? uint(ids[pair]) : 0u;\n"
            "    uint64_t expert_off = uint64_t(expert) * uint64_t(args.down_expert_bytes);\n"
            "    device const float *x = scratch + uint64_t(token) * uint64_t(args.stride) + uint64_t(args.hidden_offset) + uint64_t(slot) * uint64_t(args.n_ff);\n"
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
            "    if (active) {\n"
            "        for (uint b = ix; b < n_blocks; b += 2u) {\n"
            "            device const float *yy = x + uint64_t(b) * 256ull + y_offset;\n"
            "            if (row0 < args.n_embd) {\n"
            "                device const uchar *blk = weights + expert_off + uint64_t(row0) * uint64_t(args.down_row_bytes) + uint64_t(b) * 210ull;\n"
            "                device const uchar *q1 = blk + q_offset_l;\n"
            "                device const uchar *q2 = q1 + 32u;\n"
            "                device const uchar *qh = blk + 128u + q_offset_h;\n"
            "                device const char *sc = (device const char *)(blk + 192u + is);\n"
            "                ushort dbits = ushort(blk[208u]) | (ushort(blk[209u]) << 8u);\n"
            "                float d = qw3_f16_to_f32(dbits);\n"
            "                float acc = 0.0f;\n"
            "                for (uint l = 0; l < 4u; l++) {\n"
            "                    int qv1 = int((uint(q1[l]) & 15u) | (((uint(qh[l]) >> 0u) & 3u) << 4u)) - 32;\n"
            "                    int qv2 = int((uint(q2[l]) & 15u) | (((uint(qh[l]) >> 2u) & 3u) << 4u)) - 32;\n"
            "                    int qv3 = int((uint(q1[l]) >> 4u) | (((uint(qh[l]) >> 4u) & 3u) << 4u)) - 32;\n"
            "                    int qv4 = int((uint(q2[l]) >> 4u) | (((uint(qh[l]) >> 6u) & 3u) << 4u)) - 32;\n"
            "                    acc += float(sc[0]) * float(qv1) * yy[l + 0u];\n"
            "                    acc += float(sc[2]) * float(qv2) * yy[l + 32u];\n"
            "                    acc += float(sc[4]) * float(qv3) * yy[l + 64u];\n"
            "                    acc += float(sc[6]) * float(qv4) * yy[l + 96u];\n"
            "                }\n"
            "                sum0 += d * acc;\n"
            "            }\n"
            "            if (row1 < args.n_embd) {\n"
            "                device const uchar *blk = weights + expert_off + uint64_t(row1) * uint64_t(args.down_row_bytes) + uint64_t(b) * 210ull;\n"
            "                device const uchar *q1 = blk + q_offset_l;\n"
            "                device const uchar *q2 = q1 + 32u;\n"
            "                device const uchar *qh = blk + 128u + q_offset_h;\n"
            "                device const char *sc = (device const char *)(blk + 192u + is);\n"
            "                ushort dbits = ushort(blk[208u]) | (ushort(blk[209u]) << 8u);\n"
            "                float d = qw3_f16_to_f32(dbits);\n"
            "                float acc = 0.0f;\n"
            "                for (uint l = 0; l < 4u; l++) {\n"
            "                    int qv1 = int((uint(q1[l]) & 15u) | (((uint(qh[l]) >> 0u) & 3u) << 4u)) - 32;\n"
            "                    int qv2 = int((uint(q2[l]) & 15u) | (((uint(qh[l]) >> 2u) & 3u) << 4u)) - 32;\n"
            "                    int qv3 = int((uint(q1[l]) >> 4u) | (((uint(qh[l]) >> 4u) & 3u) << 4u)) - 32;\n"
            "                    int qv4 = int((uint(q2[l]) >> 4u) | (((uint(qh[l]) >> 6u) & 3u) << 4u)) - 32;\n"
            "                    acc += float(sc[0]) * float(qv1) * yy[l + 0u];\n"
            "                    acc += float(sc[2]) * float(qv2) * yy[l + 32u];\n"
            "                    acc += float(sc[4]) * float(qv3) * yy[l + 64u];\n"
            "                    acc += float(sc[6]) * float(qv4) * yy[l + 96u];\n"
            "                }\n"
            "                sum1 += d * acc;\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    sum0 = simd_sum(sum0);\n"
            "    sum1 = simd_sum(sum1);\n"
            "    float scale = active ? router_weights[pair] : 0.0f;\n"
            "    if (lane == 0) { sh[slot] = sum0 * scale; sh[8u + slot] = sum1 * scale; }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    if (tidx == 0 && token < args.n_tokens) {\n"
            "        float total0 = 0.0f;\n"
            "        float total1 = 0.0f;\n"
            "        for (uint s = 0; s < args.n_active; s++) { total0 += sh[s]; total1 += sh[8u + s]; }\n"
            "        device float *row = x0 + uint64_t(token) * uint64_t(args.n_embd);\n"
            "        if (row0 < args.n_embd) row[row0] += total0;\n"
            "        if (row1 < args.n_embd) row[row1] += total1;\n"
            "    }\n"
            "}\n"
            "struct qw3_moe_expert_map_args { uint n_tokens; uint n_active; uint n_expert; uint pair_capacity; };\n"
            "kernel void qw3_moe_topk_expert_map(constant qw3_moe_expert_map_args &args,\n"
            "                                    device const int *ids,\n"
            "                                    device uint *counts,\n"
            "                                    device int *pair_ids,\n"
            "                                    uint expert [[thread_position_in_grid]]) {\n"
            "    if (expert >= args.n_expert) return;\n"
            "    uint n = 0u;\n"
            "    for (uint t = 0u; t < args.n_tokens; t++) {\n"
            "        int found = -1;\n"
            "        for (uint slot = 0u; slot < args.n_active; slot++) {\n"
            "            int eid = ids[t * args.n_active + slot];\n"
            "            if (eid == int(expert)) { found = int(slot); break; }\n"
            "        }\n"
            "        if (found >= 0 && n < args.pair_capacity) {\n"
            "            pair_ids[expert * args.pair_capacity + n] = int(t * args.n_active + uint(found));\n"
            "            n++;\n"
            "        }\n"
            "    }\n"
            "    counts[expert] = n;\n"
            "}\n"
            "inline half qw3_iq3s_dequant_k_expanded(device const uchar *row, device const uchar *kgrid, uint k) {\n"
            "    uint block = k >> 8u;\n"
            "    uint local = k & 255u;\n"
            "    uint ib = local >> 5u;\n"
            "    uint within = local & 31u;\n"
            "    uint l = within >> 3u;\n"
            "    uint j = within & 3u;\n"
            "    bool second = (within & 4u) != 0u;\n"
            "    device const uchar *blk = row + uint64_t(block) * 110ull;\n"
            "    float d = float(*((device const half *)blk));\n"
            "    device const uchar *qs = blk + 2u;\n"
            "    device const uchar *qh = qs + 64u;\n"
            "    device const uchar *signs = qh + 8u;\n"
            "    device const uchar *scales = signs + 32u;\n"
            "    float db = d * float(1u + 2u * ((uint(scales[ib >> 1u]) >> (4u * (ib & 1u))) & 15u));\n"
            "    uchar qhb = qh[ib];\n"
            "    uchar qsb = qs[ib * 8u + 2u * l + (second ? 1u : 0u)];\n"
            "    uint idx = uint(qsb) | ((uint(qhb) << ((second ? 7u : 8u) - 2u * l)) & 256u);\n"
            "    uchar s = signs[ib * 4u + l];\n"
            "    float sign = (uint(s) & (1u << (j + (second ? 4u : 0u)))) ? -1.0f : 1.0f;\n"
            "    return half(db * float(kgrid[(idx & 511u) * 4u + j]) * sign);\n"
            "}\n"
            "kernel void qw3_moe_iq3_s_prefill_mapped(constant qw3_moe_prefill_batch_args &args,\n"
            "                                         device const uchar *weights,\n"
            "                                         device const float *x1,\n"
            "                                         device float *out_slots,\n"
            "                                         device const uchar *kgrid,\n"
            "                                         device const uint *counts,\n"
            "                                         device const int *pair_ids,\n"
            "                                         threadgroup char *shmem [[threadgroup(0)]],\n"
            "                                         uint3 group [[threadgroup_position_in_grid]],\n"
            "                                         ushort tid [[thread_index_in_threadgroup]],\n"
            "                                         ushort sgitg [[simdgroup_index_in_threadgroup]]) {\n"
            "    threadgroup half *sa = (threadgroup half *)shmem;\n"
            "    threadgroup half *sb = (threadgroup half *)(shmem + 4096);\n"
            "    constexpr int NR0 = 64;\n"
            "    constexpr int NR1 = 32;\n"
            "    constexpr int NK = 32;\n"
            "    constexpr int NL0 = NK / 16;\n"
            "    constexpr int NL1 = NK / 8;\n"
            "    uint expert = group.z;\n"
            "    uint r0u = group.y * NR0;\n"
            "    uint r1u = group.x * NR1;\n"
            "    uint count = counts[expert];\n"
            "    if (r0u >= args.n_ff || r1u >= count) return;\n"
            "    int nr0 = int(min(uint(NR0), args.n_ff - r0u));\n"
            "    int nr1 = int(min(uint(NR1), count - r1u));\n"
            "    int lr0 = min(int(tid) / NL0, nr0 - 1);\n"
            "    int lr1 = min(int(tid) / NL1, nr1 - 1);\n"
            "    short il0 = short(tid % NL0);\n"
            "    uint row = r0u + uint(lr0);\n"
            "    uint map_base = expert * args.n_tokens + r1u;\n"
            "    simdgroup_half8x8 ma[4];\n"
            "    simdgroup_half8x8 mb[2];\n"
            "    simdgroup_float8x8 mc[8];\n"
            "    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);\n"
            "    for (uint loop_k = 0u; loop_k < args.n_in; loop_k += NK) {\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        device const uchar *wrow = weights + uint64_t(expert) * uint64_t(args.iq3_expert_bytes) + uint64_t(row) * uint64_t(args.iq3_row_bytes);\n"
            "        for (short i = 0; i < 16; i++) {\n"
            "            uint k = loop_k + uint(16 * il0 + i);\n"
            "            const short sx = short(2 * il0 + i / 8);\n"
            "            const short sy = short((tid / NL0) / 8);\n"
            "            const short lx = short((tid / NL0) % 8);\n"
            "            const short ly = short(i % 8);\n"
            "            const short ib = short(8 * sx + sy);\n"
            "            *(sa + 64 * ib + 8 * ly + lx) = k < args.n_in ? qw3_iq3s_dequant_k_expanded(wrow, kgrid, k) : half(0.0f);\n"
            "        }\n"
            "        int pid = pair_ids[map_base + uint(lr1)];\n"
            "        uint token = uint(pid) / args.n_active;\n"
            "        device const float *x = x1 + uint64_t(token) * uint64_t(args.n_embd) + uint64_t(loop_k);\n"
            "        for (short i = 0; i < 8; i++) {\n"
            "            const short sx = short(tid % NL1);\n"
            "            const short sy = short((tid / NL1) / 8);\n"
            "            const short lx = i;\n"
            "            const short ly = short((tid / NL1) % 8);\n"
            "            const short ib = short(4 * sx + sy);\n"
            "            uint kk = uint(8 * sx + i);\n"
            "            *(sb + 64 * ib + 8 * ly + lx) = (uint(lr1) < uint(nr1) && loop_k + kk < args.n_in) ? half(x[kk]) : half(0.0f);\n"
            "        }\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        threadgroup const half *lsma = sa + 4 * 64 * (sgitg % 2);\n"
            "        threadgroup const half *lsmb = sb + 2 * 64 * (sgitg / 2);\n"
            "        for (short ik = 0; ik < NK / 8; ik++) {\n"
            "            simdgroup_barrier(mem_flags::mem_none);\n"
            "            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);\n"
            "            simdgroup_barrier(mem_flags::mem_none);\n"
            "            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);\n"
            "            simdgroup_barrier(mem_flags::mem_none);\n"
            "            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);\n"
            "            lsma += 8 * 64;\n"
            "            lsmb += 4 * 64;\n"
            "        }\n"
            "    }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    threadgroup float *tmp = ((threadgroup float *)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;\n"
            "    for (short i = 0; i < 8; i++) simdgroup_store(mc[i], tmp + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    for (short j = short(sgitg); j < nr1; j += 4) {\n"
            "        int pid = pair_ids[map_base + uint(j)];\n"
            "        device float *dst = out_slots + uint64_t(uint(pid)) * uint64_t(args.n_ff) + uint64_t(r0u);\n"
            "        threadgroup float *src = ((threadgroup float *)shmem) + int(j) * NR0;\n"
            "        int i = int(tid & 31u);\n"
            "        device float4 *dst4 = (device float4 *)dst;\n"
            "        threadgroup float4 *src4 = (threadgroup float4 *)src;\n"
            "        for (; i < nr0 / 4; i += 32) dst4[i] = src4[i];\n"
            "        i = 4 * (nr0 / 4) + int(tid & 31u);\n"
            "        for (; i < nr0; i += 32) dst[i] = src[i];\n"
            "    }\n"
            "}\n"
            "kernel void qw3_moe_iq3_s_swiglu_prefill_pair_mapped(constant qw3_moe_prefill_batch_args &args,\n"
            "                                                       device const uchar *gate_weights,\n"
            "                                                       device const uchar *up_weights,\n"
            "                                                       device const float *x1,\n"
            "                                                       device float *scratch,\n"
            "                                                       device const uchar *kgrid,\n"
            "                                                       device const uint *counts,\n"
            "                                                       device const int *pair_ids,\n"
            "                                                       threadgroup char *shmem [[threadgroup(0)]],\n"
            "                                                       uint3 group [[threadgroup_position_in_grid]],\n"
            "                                                       ushort tid [[thread_index_in_threadgroup]],\n"
            "                                                       ushort sgitg [[simdgroup_index_in_threadgroup]]) {\n"
            "    threadgroup half *sa = (threadgroup half *)shmem;\n"
            "    threadgroup half *sb = (threadgroup half *)(shmem + 4096);\n"
            "    constexpr int NR0 = 64;\n"
            "    constexpr int NR1 = 32;\n"
            "    constexpr int NK = 32;\n"
            "    constexpr int NL0 = NK / 16;\n"
            "    constexpr int NL1 = NK / 8;\n"
            "    uint expert = group.z;\n"
            "    uint r0u = group.y * NR0;\n"
            "    uint r1u = group.x * NR1;\n"
            "    uint count = counts[expert];\n"
            "    if (r0u >= args.n_ff || r1u >= count) return;\n"
            "    int nr0 = int(min(uint(NR0), args.n_ff - r0u));\n"
            "    int nr1 = int(min(uint(NR1), count - r1u));\n"
            "    int lr0 = min(int(tid) / NL0, nr0 - 1);\n"
            "    int lr1 = min(int(tid) / NL1, nr1 - 1);\n"
            "    short il0 = short(tid % NL0);\n"
            "    uint row = r0u + uint(lr0);\n"
            "    uint map_base = expert * args.n_tokens + r1u;\n"
            "    simdgroup_half8x8 ma[4];\n"
            "    simdgroup_half8x8 mb[2];\n"
            "    simdgroup_float8x8 mc_gate[8];\n"
            "    simdgroup_float8x8 mc_up[8];\n"
            "    for (short i = 0; i < 8; i++) {\n"
            "        mc_gate[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);\n"
            "        mc_up[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);\n"
            "    }\n"
            "    for (uint loop_k = 0u; loop_k < args.n_in; loop_k += NK) {\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        device const uchar *wrow_gate = gate_weights + uint64_t(expert) * uint64_t(args.iq3_expert_bytes) + uint64_t(row) * uint64_t(args.iq3_row_bytes);\n"
            "        for (short i = 0; i < 16; i++) {\n"
            "            uint k = loop_k + uint(16 * il0 + i);\n"
            "            const short sx = short(2 * il0 + i / 8);\n"
            "            const short sy = short((tid / NL0) / 8);\n"
            "            const short lx = short((tid / NL0) % 8);\n"
            "            const short ly = short(i % 8);\n"
            "            const short ib = short(8 * sx + sy);\n"
            "            *(sa + 64 * ib + 8 * ly + lx) = k < args.n_in ? qw3_iq3s_dequant_k_expanded(wrow_gate, kgrid, k) : half(0.0f);\n"
            "        }\n"
            "        int pid = pair_ids[map_base + uint(lr1)];\n"
            "        uint token = uint(pid) / args.n_active;\n"
            "        device const float *x = x1 + uint64_t(token) * uint64_t(args.n_embd) + uint64_t(loop_k);\n"
            "        for (short i = 0; i < 8; i++) {\n"
            "            const short sx = short(tid % NL1);\n"
            "            const short sy = short((tid / NL1) / 8);\n"
            "            const short lx = i;\n"
            "            const short ly = short((tid / NL1) % 8);\n"
            "            const short ib = short(4 * sx + sy);\n"
            "            uint kk = uint(8 * sx + i);\n"
            "            *(sb + 64 * ib + 8 * ly + lx) = (uint(lr1) < uint(nr1) && loop_k + kk < args.n_in) ? half(x[kk]) : half(0.0f);\n"
            "        }\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        threadgroup const half *lsma = sa + 4 * 64 * (sgitg % 2);\n"
            "        threadgroup const half *lsmb = sb + 2 * 64 * (sgitg / 2);\n"
            "        for (short ik = 0; ik < NK / 8; ik++) {\n"
            "            simdgroup_barrier(mem_flags::mem_none);\n"
            "            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);\n"
            "            simdgroup_barrier(mem_flags::mem_none);\n"
            "            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);\n"
            "            simdgroup_barrier(mem_flags::mem_none);\n"
            "            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc_gate[i], mb[i / 4], ma[i % 4], mc_gate[i]);\n"
            "            lsma += 8 * 64;\n"
            "            lsmb += 4 * 64;\n"
            "        }\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        device const uchar *wrow_up = up_weights + uint64_t(expert) * uint64_t(args.iq3_expert_bytes) + uint64_t(row) * uint64_t(args.iq3_row_bytes);\n"
            "        for (short i = 0; i < 16; i++) {\n"
            "            uint k = loop_k + uint(16 * il0 + i);\n"
            "            const short sx = short(2 * il0 + i / 8);\n"
            "            const short sy = short((tid / NL0) / 8);\n"
            "            const short lx = short((tid / NL0) % 8);\n"
            "            const short ly = short(i % 8);\n"
            "            const short ib = short(8 * sx + sy);\n"
            "            *(sa + 64 * ib + 8 * ly + lx) = k < args.n_in ? qw3_iq3s_dequant_k_expanded(wrow_up, kgrid, k) : half(0.0f);\n"
            "        }\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        lsma = sa + 4 * 64 * (sgitg % 2);\n"
            "        lsmb = sb + 2 * 64 * (sgitg / 2);\n"
            "        for (short ik = 0; ik < NK / 8; ik++) {\n"
            "            simdgroup_barrier(mem_flags::mem_none);\n"
            "            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);\n"
            "            simdgroup_barrier(mem_flags::mem_none);\n"
            "            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);\n"
            "            simdgroup_barrier(mem_flags::mem_none);\n"
            "            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc_up[i], mb[i / 4], ma[i % 4], mc_up[i]);\n"
            "            lsma += 8 * 64;\n"
            "            lsmb += 4 * 64;\n"
            "        }\n"
            "    }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    threadgroup float *tmp_gate = (threadgroup float *)shmem;\n"
            "    threadgroup float *tmp_up = tmp_gate + NR0 * NR1;\n"
            "    threadgroup float *gate_dst = tmp_gate + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;\n"
            "    threadgroup float *up_dst = tmp_up + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;\n"
            "    for (short i = 0; i < 8; i++) {\n"
            "        simdgroup_store(mc_gate[i], gate_dst + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);\n"
            "        simdgroup_store(mc_up[i], up_dst + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);\n"
            "    }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    for (short j = short(sgitg); j < nr1; j += 4) {\n"
            "        int pid = pair_ids[map_base + uint(j)];\n"
            "        uint token = uint(pid) / args.n_active;\n"
            "        uint slot = uint(pid) - token * args.n_active;\n"
            "        device float *dst = scratch + uint64_t(token) * uint64_t(args.stride) + uint64_t(args.hidden_offset) + uint64_t(slot) * uint64_t(args.n_ff) + uint64_t(r0u);\n"
            "        threadgroup float *src_gate = tmp_gate + int(j) * NR0;\n"
            "        threadgroup float *src_up = tmp_up + int(j) * NR0;\n"
            "        int i = int(tid & 31u);\n"
            "        device float4 *dst4 = (device float4 *)dst;\n"
            "        threadgroup float4 *gate4 = (threadgroup float4 *)src_gate;\n"
            "        threadgroup float4 *up4 = (threadgroup float4 *)src_up;\n"
            "        for (; i < nr0 / 4; i += 32) {\n"
            "            float4 g = gate4[i];\n"
            "            dst4[i] = (g / (float4(1.0f) + exp(-g))) * up4[i];\n"
            "        }\n"
            "        i = 4 * (nr0 / 4) + int(tid & 31u);\n"
            "        for (; i < nr0; i += 32) {\n"
            "            float g = src_gate[i];\n"
            "            dst[i] = (g / (1.0f + exp(-g))) * src_up[i];\n"
            "        }\n"
            "    }\n"
            "}\n"
            "kernel void qw3_moe_swiglu_slots_to_hidden(constant qw3_moe_prefill_batch_args &args,\n"
            "                                          device const float *gate_slots,\n"
            "                                          device const float *up_slots,\n"
            "                                          device float *scratch,\n"
            "                                          uint gid [[thread_position_in_grid]]) {\n"
            "    uint total = args.n_tokens * args.n_active * args.n_ff;\n"
            "    if (gid >= total) return;\n"
            "    uint row = gid % args.n_ff;\n"
            "    uint pair = gid / args.n_ff;\n"
            "    uint token = pair / args.n_active;\n"
            "    uint slot = pair - token * args.n_active;\n"
            "    float g = gate_slots[gid];\n"
            "    scratch[uint64_t(token) * uint64_t(args.stride) + uint64_t(args.hidden_offset) + uint64_t(slot) * uint64_t(args.n_ff) + row] = (g / (1.0f + exp(-g))) * up_slots[gid];\n"
            "}\n"
            "kernel void qw3_moe_swiglu_slots_to_hidden_f16(constant qw3_moe_prefill_batch_args &args,\n"
            "                                              device const float *gate_slots,\n"
            "                                              device const float *up_slots,\n"
            "                                              device half *mid,\n"
            "                                              uint gid [[thread_position_in_grid]]) {\n"
            "    uint total = args.n_tokens * args.n_active * args.n_ff;\n"
            "    if (gid >= total) return;\n"
            "    float g = gate_slots[gid];\n"
            "    mid[gid] = half((g / (1.0f + exp(-g))) * up_slots[gid]);\n"
            "}\n"
            "inline half qw3_iq4xs_dequant_k(device const uchar *row, uint k) {\n"
            "    uint block = k >> 8u;\n"
            "    uint local = k & 255u;\n"
            "    uint ib = local >> 5u;\n"
            "    uint within = local & 31u;\n"
            "    uint il = (within & 15u) >> 3u;\n"
            "    uint j = within & 7u;\n"
            "    bool hi = within >= 16u;\n"
            "    device const uchar *blk = row + uint64_t(block) * 136ull;\n"
            "    float d = float(*((device const half *)blk));\n"
            "    ushort scales_h = *((device const ushort *)(blk + 2u));\n"
            "    device const uchar *scales_l = blk + 4u;\n"
            "    uint ls = ((uint(scales_l[ib >> 1u]) >> (4u * (ib & 1u))) & 15u) | (((uint(scales_h) >> (2u * ib)) & 3u) << 4u);\n"
            "    uchar packed = *(scales_l + 4u + ib * 16u + il * 8u + j);\n"
            "    uint q = hi ? (uint(packed) >> 4u) : (uint(packed) & 15u);\n"
            "    return half(d * float(int(ls) - 32) * qw3_iq4nl_val(q));\n"
            "}\n"
            "kernel void qw3_moe_down_iq4_xs_prefill_mapped(constant qw3_moe_prefill_batch_args &args,\n"
            "                                                 device const uchar *weights,\n"
            "                                                 device const float *scratch,\n"
            "                                                 device float *down_slots,\n"
            "                                                 device const uint *counts,\n"
            "                                                 device const int *pair_ids,\n"
            "                                                 device const float *router_weights,\n"
            "                                                 threadgroup char *shmem [[threadgroup(0)]],\n"
            "                                                 uint3 group [[threadgroup_position_in_grid]],\n"
            "                                                 ushort tid [[thread_index_in_threadgroup]],\n"
            "                                                 ushort sgitg [[simdgroup_index_in_threadgroup]]) {\n"
            "    threadgroup half *sa = (threadgroup half *)shmem;\n"
            "    threadgroup half *sb = (threadgroup half *)(shmem + 4096);\n"
            "    constexpr int NR0 = 64;\n"
            "    constexpr int NR1 = 32;\n"
            "    constexpr int NK = 32;\n"
            "    constexpr int NL0 = NK / 16;\n"
            "    constexpr int NL1 = NK / 8;\n"
            "    uint expert = group.z;\n"
            "    uint r0u = group.y * NR0;\n"
            "    uint r1u = group.x * NR1;\n"
            "    uint count = counts[expert];\n"
            "    if (r0u >= args.n_embd || r1u >= count) return;\n"
            "    int nr0 = int(min(uint(NR0), args.n_embd - r0u));\n"
            "    int nr1 = int(min(uint(NR1), count - r1u));\n"
            "    int lr0 = min(int(tid) / NL0, nr0 - 1);\n"
            "    int lr1 = min(int(tid) / NL1, nr1 - 1);\n"
            "    short il0 = short(tid % NL0);\n"
            "    uint row = r0u + uint(lr0);\n"
            "    uint map_base = expert * args.n_tokens + r1u;\n"
            "    simdgroup_half8x8 ma[4];\n"
            "    simdgroup_half8x8 mb[2];\n"
            "    simdgroup_float8x8 mc[8];\n"
            "    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);\n"
            "    for (uint loop_k = 0u; loop_k < args.n_ff; loop_k += NK) {\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        device const uchar *wrow = weights + uint64_t(expert) * uint64_t(args.down_expert_bytes) + uint64_t(row) * uint64_t(args.down_row_bytes);\n"
            "        for (short i = 0; i < 16; i++) {\n"
            "            uint k = loop_k + uint(16 * il0 + i);\n"
            "            const short sx = short(2 * il0 + i / 8);\n"
            "            const short sy = short((tid / NL0) / 8);\n"
            "            const short lx = short((tid / NL0) % 8);\n"
            "            const short ly = short(i % 8);\n"
            "            const short ib = short(8 * sx + sy);\n"
            "            *(sa + 64 * ib + 8 * ly + lx) = k < args.n_ff ? qw3_iq4xs_dequant_k(wrow, k) : half(0.0f);\n"
            "        }\n"
            "        int pid = pair_ids[map_base + uint(lr1)];\n"
            "        uint token = uint(pid) / args.n_active;\n"
            "        uint slot = uint(pid) - token * args.n_active;\n"
            "        device const float *hidden = scratch + uint64_t(token) * uint64_t(args.stride) + uint64_t(args.hidden_offset) + uint64_t(slot) * uint64_t(args.n_ff) + uint64_t(loop_k);\n"
            "        for (short i = 0; i < 8; i++) {\n"
            "            const short sx = short(tid % NL1);\n"
            "            const short sy = short((tid / NL1) / 8);\n"
            "            const short lx = i;\n"
            "            const short ly = short((tid / NL1) % 8);\n"
            "            const short ib = short(4 * sx + sy);\n"
            "            uint kk = uint(8 * sx + i);\n"
            "            *(sb + 64 * ib + 8 * ly + lx) = (uint(lr1) < uint(nr1) && loop_k + kk < args.n_ff) ? half(hidden[kk]) : half(0.0f);\n"
            "        }\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        threadgroup const half *lsma = sa + 4 * 64 * (sgitg % 2);\n"
            "        threadgroup const half *lsmb = sb + 2 * 64 * (sgitg / 2);\n"
            "        for (short ik = 0; ik < NK / 8; ik++) {\n"
            "            simdgroup_barrier(mem_flags::mem_none);\n"
            "            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);\n"
            "            simdgroup_barrier(mem_flags::mem_none);\n"
            "            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);\n"
            "            simdgroup_barrier(mem_flags::mem_none);\n"
            "            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);\n"
            "            lsma += 8 * 64;\n"
            "            lsmb += 4 * 64;\n"
            "        }\n"
            "    }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    threadgroup float *tmp = ((threadgroup float *)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;\n"
            "    for (short i = 0; i < 8; i++) simdgroup_store(mc[i], tmp + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    for (short j = short(sgitg); j < nr1; j += 4) {\n"
            "        int pid = pair_ids[map_base + uint(j)];\n"
            "        device float *dst = down_slots + uint64_t(uint(pid)) * uint64_t(args.n_embd) + uint64_t(r0u);\n"
            "        threadgroup float *src = ((threadgroup float *)shmem) + int(j) * NR0;\n"
            "        float scale = router_weights[uint(pid)];\n"
            "        int i = int(tid & 31u);\n"
            "        device float4 *dst4 = (device float4 *)dst;\n"
            "        threadgroup float4 *src4 = (threadgroup float4 *)src;\n"
            "        for (; i < nr0 / 4; i += 32) dst4[i] = src4[i] * scale;\n"
            "        i = 4 * (nr0 / 4) + int(tid & 31u);\n"
            "        for (; i < nr0; i += 32) dst[i] = src[i] * scale;\n"
            "    }\n"
            "}\n"
            "kernel void qw3_moe_down_iq4_xs_prefill_mapped_f16(constant qw3_moe_prefill_batch_args &args,\n"
            "                                                     device const uchar *weights,\n"
            "                                                     device const half *mid,\n"
            "                                                     device float *down_slots,\n"
            "                                                     device const uint *counts,\n"
            "                                                     device const int *pair_ids,\n"
            "                                                     device const float *router_weights,\n"
            "                                                     threadgroup char *shmem [[threadgroup(0)]],\n"
            "                                                     uint3 group [[threadgroup_position_in_grid]],\n"
            "                                                     ushort tid [[thread_index_in_threadgroup]],\n"
            "                                                     ushort sgitg [[simdgroup_index_in_threadgroup]]) {\n"
            "    threadgroup half *sa = (threadgroup half *)shmem;\n"
            "    threadgroup half *sb = (threadgroup half *)(shmem + 4096);\n"
            "    constexpr int NR0 = 64;\n"
            "    constexpr int NR1 = 32;\n"
            "    constexpr int NK = 32;\n"
            "    constexpr int NL0 = NK / 16;\n"
            "    constexpr int NL1 = NK / 8;\n"
            "    uint expert = group.z;\n"
            "    uint r0u = group.y * NR0;\n"
            "    uint r1u = group.x * NR1;\n"
            "    uint count = counts[expert];\n"
            "    if (r0u >= args.n_embd || r1u >= count) return;\n"
            "    int nr0 = int(min(uint(NR0), args.n_embd - r0u));\n"
            "    int nr1 = int(min(uint(NR1), count - r1u));\n"
            "    int lr0 = min(int(tid) / NL0, nr0 - 1);\n"
            "    int lr1 = min(int(tid) / NL1, nr1 - 1);\n"
            "    short il0 = short(tid % NL0);\n"
            "    uint row = r0u + uint(lr0);\n"
            "    uint map_base = expert * args.n_tokens + r1u;\n"
            "    simdgroup_half8x8 ma[4];\n"
            "    simdgroup_half8x8 mb[2];\n"
            "    simdgroup_float8x8 mc[8];\n"
            "    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);\n"
            "    for (uint loop_k = 0u; loop_k < args.n_ff; loop_k += NK) {\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        device const uchar *wrow = weights + uint64_t(expert) * uint64_t(args.down_expert_bytes) + uint64_t(row) * uint64_t(args.down_row_bytes);\n"
            "        for (short i = 0; i < 16; i++) {\n"
            "            uint k = loop_k + uint(16 * il0 + i);\n"
            "            const short sx = short(2 * il0 + i / 8);\n"
            "            const short sy = short((tid / NL0) / 8);\n"
            "            const short lx = short((tid / NL0) % 8);\n"
            "            const short ly = short(i % 8);\n"
            "            const short ib = short(8 * sx + sy);\n"
            "            *(sa + 64 * ib + 8 * ly + lx) = k < args.n_ff ? qw3_iq4xs_dequant_k(wrow, k) : half(0.0f);\n"
            "        }\n"
            "        int pid = pair_ids[map_base + uint(lr1)];\n"
            "        device const half *hidden = mid + uint64_t(uint(pid)) * uint64_t(args.n_ff) + uint64_t(loop_k);\n"
            "        for (short i = 0; i < 8; i++) {\n"
            "            const short sx = short(tid % NL1);\n"
            "            const short sy = short((tid / NL1) / 8);\n"
            "            const short lx = i;\n"
            "            const short ly = short((tid / NL1) % 8);\n"
            "            const short ib = short(4 * sx + sy);\n"
            "            uint kk = uint(8 * sx + i);\n"
            "            *(sb + 64 * ib + 8 * ly + lx) = (uint(lr1) < uint(nr1) && loop_k + kk < args.n_ff) ? hidden[kk] : half(0.0f);\n"
            "        }\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        threadgroup const half *lsma = sa + 4 * 64 * (sgitg % 2);\n"
            "        threadgroup const half *lsmb = sb + 2 * 64 * (sgitg / 2);\n"
            "        for (short ik = 0; ik < NK / 8; ik++) {\n"
            "            simdgroup_barrier(mem_flags::mem_none);\n"
            "            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);\n"
            "            simdgroup_barrier(mem_flags::mem_none);\n"
            "            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);\n"
            "            simdgroup_barrier(mem_flags::mem_none);\n"
            "            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);\n"
            "            lsma += 8 * 64;\n"
            "            lsmb += 4 * 64;\n"
            "        }\n"
            "    }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    threadgroup float *tmp = ((threadgroup float *)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;\n"
            "    for (short i = 0; i < 8; i++) simdgroup_store(mc[i], tmp + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    for (short j = short(sgitg); j < nr1; j += 4) {\n"
            "        int pid = pair_ids[map_base + uint(j)];\n"
            "        device float *dst = down_slots + uint64_t(uint(pid)) * uint64_t(args.n_embd) + uint64_t(r0u);\n"
            "        threadgroup float *src = ((threadgroup float *)shmem) + int(j) * NR0;\n"
            "        float scale = router_weights[uint(pid)];\n"
            "        int i = int(tid & 31u);\n"
            "        device float4 *dst4 = (device float4 *)dst;\n"
            "        threadgroup float4 *src4 = (threadgroup float4 *)src;\n"
            "        for (; i < nr0 / 4; i += 32) dst4[i] = src4[i] * scale;\n"
            "        i = 4 * (nr0 / 4) + int(tid & 31u);\n"
            "        for (; i < nr0; i += 32) dst[i] = src[i] * scale;\n"
            "    }\n"
            "}\n"
            "struct qw3_moe_down_slot_reduce_args { uint n_tokens; uint n_active; uint n_embd; };\n"
            "kernel void qw3_moe_down_prefill_reduce_slots(constant qw3_moe_down_slot_reduce_args &args,\n"
            "                                             device const float *down_slots,\n"
            "                                             device const float *router_weights,\n"
            "                                             device float *x0,\n"
            "                                             uint gid [[thread_position_in_grid]]) {\n"
            "    uint total = args.n_tokens * args.n_embd;\n"
            "    if (gid >= total) return;\n"
            "    uint token = gid / args.n_embd;\n"
            "    uint row = gid - token * args.n_embd;\n"
            "    float sum = 0.0f;\n"
            "    for (uint slot = 0u; slot < args.n_active; slot++) {\n"
            "        uint pair = token * args.n_active + slot;\n"
            "        sum += down_slots[uint64_t(pair) * uint64_t(args.n_embd) + row];\n"
            "    }\n"
            "    x0[uint64_t(token) * uint64_t(args.n_embd) + row] += sum;\n"
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
        g_rmsnorm_weight_f32_pipeline &&
        g_rmsnorm_weight_f32_rows_pipeline &&
        g_rmsnorm_weight_f32_rows_to_out_pipeline &&
        g_embed_q8_0_pipeline &&
        g_embed_q8_0_batch_pipeline &&
        g_matvec_q8_0_pipeline && g_matmul_q8_0_batch4_pipeline &&
        g_matmul_q8_0_mm_pipeline && g_matmul_q8_0_mm_bc_pipeline &&
        g_matvec_q8_0_pair_pipeline &&
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
        g_moe_iq3_s_prefill_batch_pipeline &&
        g_moe_down_iq4_xs_prefill_reduce_pipeline &&
        g_moe_down_q6_k_prefill_reduce_pipeline &&
        g_moe_topk_expert_map_pipeline &&
        g_moe_iq3_s_prefill_mapped_pipeline &&
        g_moe_iq3_s_prefill_pair_mapped_pipeline &&
        g_moe_swiglu_slots_to_hidden_pipeline &&
        g_moe_swiglu_slots_to_hidden_f16_pipeline &&
        g_moe_down_iq4_xs_prefill_mapped_pipeline &&
        g_moe_down_iq4_xs_prefill_mapped_f16_pipeline &&
        g_moe_down_prefill_reduce_slots_pipeline &&
        g_matvec_f32_pipeline && g_matvec_f32_pair_pipeline &&
        g_matvec_f32_fast_pipeline &&
        g_matmul_f32_batch4_pipeline &&
        g_matmul_f32_pair_batch4_pipeline &&
        g_deltanet_conv1d_zero_pipeline &&
        g_deltanet_conv1d_step_pipeline &&
        g_deltanet_conv1d_batch_pipeline &&
        g_deltanet_conv1d_batch_state_pipeline &&
        g_l2norm_heads_pipeline &&
        g_l2norm_qk_batch_pipeline &&
        g_gqa_q_norm_gate_pipeline && g_gqa_k_norm_pipeline &&
        g_gqa_q_norm_gate_batch_pipeline && g_gqa_k_norm_batch_pipeline &&
        g_rope_heads_pipeline && g_rope_heads_batch_pipeline &&
        g_gqa_single_token_inner_pipeline &&
        g_gqa_attend2_inner_pipeline &&
        g_gqa_attend_n_inner_pipeline &&
        g_gqa_prefill_attend_inner_pipeline &&
        g_gqa_prefill_write_cache_pipeline &&
        g_gqa_prefill_cached_attend_inner_pipeline &&
        g_gqa_kv_quant_q8_pipeline &&
        g_gqa_attend_n_q8_inner_pipeline &&
        g_gqa_attend_n_q8_split_partial_pipeline &&
        g_gqa_attend_n_q8_split_reduce_pipeline &&
        g_deltanet_recur_zero_pipeline &&
        g_deltanet_recur_pipeline && g_deltanet_recur_scratch_gates_pipeline &&
        g_deltanet_prepare_scratch_gates_pipeline &&
        g_deltanet_recur_scratch_gates_tiled_pipeline &&
        g_deltanet_fused_gdn_scratch_pipeline &&
        g_deltanet_batch_fused_gdn_pipeline &&
        g_deltanet_gated_rmsnorm_pipeline &&
        g_residual_rmsnorm_weight_f32_pipeline &&
        g_residual_rmsnorm_update_x0_pipeline &&
        g_residual_rmsnorm_batch_update_x0_pipeline &&
        g_silu_mul_pipeline &&
        g_scale_pipeline && g_argmax_blocks_pipeline &&
        g_add_moe_to_x0_pipeline && g_silu_mul_offsets_pipeline &&
        g_silu_mul_rows_offsets_pipeline &&
        g_scale_x1_scalar_add_x0_pipeline && g_scale_x1_add_x0_pipeline &&
        g_scale_scratch_add_x0_pipeline &&
        g_sigmoid_scale_scratch_add_x0_rows_pipeline &&
        g_router_top8_pipeline &&
        g_router_top8_batch_pipeline &&
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
    fn = [g_library newFunctionWithName:@"qw3_rmsnorm_weight_f32_rows"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_rmsnorm_weight_f32_rows not found\n");
        return 0;
    }
    g_rmsnorm_weight_f32_rows_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                                 error:&error];
    if (!g_rmsnorm_weight_f32_rows_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_rmsnorm_weight_f32_rows failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_rmsnorm_weight_f32_rows_to_out"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_rmsnorm_weight_f32_rows_to_out not found\n");
        return 0;
    }
    g_rmsnorm_weight_f32_rows_to_out_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_rmsnorm_weight_f32_rows_to_out_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_rmsnorm_weight_f32_rows_to_out failed: %s\n",
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
    fn = [g_library newFunctionWithName:@"qw3_embed_q8_0_batch"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_embed_q8_0_batch not found\n");
        return 0;
    }
    g_embed_q8_0_batch_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                          error:&error];
    if (!g_embed_q8_0_batch_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_embed_q8_0_batch failed: %s\n",
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
    fn = [g_library newFunctionWithName:@"qw3_matmul_q8_0_batch4"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matmul_q8_0_batch4 not found\n");
        return 0;
    }
    g_matmul_q8_0_batch4_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                            error:&error];
    if (!g_matmul_q8_0_batch4_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matmul_q8_0_batch4 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    {
        const BOOL bc_values[2] = { NO, YES };
        const char *labels[2] = {
            "qw3_matmul_q8_0_mm",
            "qw3_matmul_q8_0_mm(boundary)"
        };
        for (int i = 0; i < 2; i++) {
            MTLFunctionConstantValues *constants =
                [[MTLFunctionConstantValues alloc] init];
            BOOL bc_out = bc_values[i];
            [constants setConstantValue:&bc_out
                                    type:MTLDataTypeBool
                                 atIndex:700];
            fn = [g_library newFunctionWithName:@"qw3_matmul_q8_0_mm"
                                  constantValues:constants
                                           error:&error];
            if (!fn) {
                fprintf(stderr, "qw3: Metal function %s not found: %s\n",
                        labels[i], [[error localizedDescription] UTF8String]);
                return 0;
            }
            id<MTLComputePipelineState> pipeline =
                [g_device newComputePipelineStateWithFunction:fn
                                                        error:&error];
            if (!pipeline) {
                fprintf(stderr, "qw3: Metal pipeline %s failed: %s\n",
                        labels[i], [[error localizedDescription] UTF8String]);
                return 0;
            }
            if (i == 0) {
                g_matmul_q8_0_mm_pipeline = pipeline;
            } else {
                g_matmul_q8_0_mm_bc_pipeline = pipeline;
            }
        }
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
    fn = [g_library newFunctionWithName:@"qw3_moe_iq3_s_swiglu_prefill_batch_fast"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_moe_iq3_s_swiglu_prefill_batch_fast not found\n");
        return 0;
    }
    g_moe_iq3_s_prefill_batch_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                                 error:&error];
    if (!g_moe_iq3_s_prefill_batch_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_moe_iq3_s_swiglu_prefill_batch_fast failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_moe_down_iq4_xs_prefill_batch_reduce_fast"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_moe_down_iq4_xs_prefill_batch_reduce_fast not found\n");
        return 0;
    }
    g_moe_down_iq4_xs_prefill_reduce_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                                        error:&error];
    if (!g_moe_down_iq4_xs_prefill_reduce_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_moe_down_iq4_xs_prefill_batch_reduce_fast failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_moe_down_q6_k_prefill_batch_reduce_fast"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_moe_down_q6_k_prefill_batch_reduce_fast not found\n");
        return 0;
    }
    g_moe_down_q6_k_prefill_reduce_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                                      error:&error];
    if (!g_moe_down_q6_k_prefill_reduce_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_moe_down_q6_k_prefill_batch_reduce_fast failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_moe_topk_expert_map"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_moe_topk_expert_map not found\n");
        return 0;
    }
    g_moe_topk_expert_map_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                             error:&error];
    if (!g_moe_topk_expert_map_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_moe_topk_expert_map failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_moe_iq3_s_prefill_mapped"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_moe_iq3_s_prefill_mapped not found\n");
        return 0;
    }
    g_moe_iq3_s_prefill_mapped_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_iq3_s_prefill_mapped_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_moe_iq3_s_prefill_mapped failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_moe_iq3_s_swiglu_prefill_pair_mapped"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_moe_iq3_s_swiglu_prefill_pair_mapped not found\n");
        return 0;
    }
    g_moe_iq3_s_prefill_pair_mapped_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_iq3_s_prefill_pair_mapped_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_moe_iq3_s_swiglu_prefill_pair_mapped failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_moe_swiglu_slots_to_hidden"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_moe_swiglu_slots_to_hidden not found\n");
        return 0;
    }
    g_moe_swiglu_slots_to_hidden_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_swiglu_slots_to_hidden_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_moe_swiglu_slots_to_hidden failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_moe_swiglu_slots_to_hidden_f16"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_moe_swiglu_slots_to_hidden_f16 not found\n");
        return 0;
    }
    g_moe_swiglu_slots_to_hidden_f16_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_swiglu_slots_to_hidden_f16_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_moe_swiglu_slots_to_hidden_f16 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_moe_down_iq4_xs_prefill_mapped"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_moe_down_iq4_xs_prefill_mapped not found\n");
        return 0;
    }
    g_moe_down_iq4_xs_prefill_mapped_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_down_iq4_xs_prefill_mapped_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_moe_down_iq4_xs_prefill_mapped failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_moe_down_iq4_xs_prefill_mapped_f16"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_moe_down_iq4_xs_prefill_mapped_f16 not found\n");
        return 0;
    }
    g_moe_down_iq4_xs_prefill_mapped_f16_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_down_iq4_xs_prefill_mapped_f16_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_moe_down_iq4_xs_prefill_mapped_f16 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_moe_down_prefill_reduce_slots"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_moe_down_prefill_reduce_slots not found\n");
        return 0;
    }
    g_moe_down_prefill_reduce_slots_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_down_prefill_reduce_slots_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_moe_down_prefill_reduce_slots failed: %s\n",
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
    fn = [g_library newFunctionWithName:@"qw3_matmul_f32_batch4"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matmul_f32_batch4 not found\n");
        return 0;
    }
    g_matmul_f32_batch4_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                           error:&error];
    if (!g_matmul_f32_batch4_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matmul_f32_batch4 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_matmul_f32_pair_batch4"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_matmul_f32_pair_batch4 not found\n");
        return 0;
    }
    g_matmul_f32_pair_batch4_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_matmul_f32_pair_batch4_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_matmul_f32_pair_batch4 failed: %s\n",
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
    fn = [g_library newFunctionWithName:@"qw3_deltanet_conv1d_batch"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_deltanet_conv1d_batch not found\n");
        return 0;
    }
    g_deltanet_conv1d_batch_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_deltanet_conv1d_batch_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_deltanet_conv1d_batch failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_deltanet_conv1d_batch_state"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_deltanet_conv1d_batch_state not found\n");
        return 0;
    }
    g_deltanet_conv1d_batch_state_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_deltanet_conv1d_batch_state_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_deltanet_conv1d_batch_state failed: %s\n",
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
    fn = [g_library newFunctionWithName:@"qw3_l2norm_qk_batch"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_l2norm_qk_batch not found\n");
        return 0;
    }
    g_l2norm_qk_batch_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                         error:&error];
    if (!g_l2norm_qk_batch_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_l2norm_qk_batch failed: %s\n",
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
    fn = [g_library newFunctionWithName:@"qw3_gqa_q_norm_gate_batch"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_q_norm_gate_batch not found\n");
        return 0;
    }
    g_gqa_q_norm_gate_batch_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_gqa_q_norm_gate_batch_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_q_norm_gate_batch failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_gqa_k_norm_batch"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_k_norm_batch not found\n");
        return 0;
    }
    g_gqa_k_norm_batch_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_gqa_k_norm_batch_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_k_norm_batch failed: %s\n",
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
    fn = [g_library newFunctionWithName:@"qw3_rope_heads_batch"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_rope_heads_batch not found\n");
        return 0;
    }
    g_rope_heads_batch_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_rope_heads_batch_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_rope_heads_batch failed: %s\n",
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
    fn = [g_library newFunctionWithName:@"qw3_gqa_prefill_attend_inner"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_prefill_attend_inner not found\n");
        return 0;
    }
    g_gqa_prefill_attend_inner_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_gqa_prefill_attend_inner_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_prefill_attend_inner failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_gqa_prefill_write_cache"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_prefill_write_cache not found\n");
        return 0;
    }
    g_gqa_prefill_write_cache_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_gqa_prefill_write_cache_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_prefill_write_cache failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_gqa_prefill_cached_attend_inner"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_prefill_cached_attend_inner not found\n");
        return 0;
    }
    g_gqa_prefill_cached_attend_inner_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_gqa_prefill_cached_attend_inner_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_prefill_cached_attend_inner failed: %s\n",
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
    fn = [g_library newFunctionWithName:@"qw3_gqa_attend_n_q8_split_partial"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_attend_n_q8_split_partial not found\n");
        return 0;
    }
    g_gqa_attend_n_q8_split_partial_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_gqa_attend_n_q8_split_partial_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_attend_n_q8_split_partial failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_gqa_attend_n_q8_split_reduce"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_attend_n_q8_split_reduce not found\n");
        return 0;
    }
    g_gqa_attend_n_q8_split_reduce_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_gqa_attend_n_q8_split_reduce_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_attend_n_q8_split_reduce failed: %s\n",
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
    fn = [g_library newFunctionWithName:@"qw3_deltanet_prepare_scratch_gates"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_deltanet_prepare_scratch_gates not found\n");
        return 0;
    }
    g_deltanet_prepare_scratch_gates_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_deltanet_prepare_scratch_gates_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_deltanet_prepare_scratch_gates failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_deltanet_recur_scratch_gates_tiled"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_deltanet_recur_scratch_gates_tiled not found\n");
        return 0;
    }
    g_deltanet_recur_scratch_gates_tiled_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_deltanet_recur_scratch_gates_tiled_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_deltanet_recur_scratch_gates_tiled failed: %s\n",
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
    fn = [g_library newFunctionWithName:@"qw3_deltanet_batch_fused_gdn"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_deltanet_batch_fused_gdn not found\n");
        return 0;
    }
    g_deltanet_batch_fused_gdn_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_deltanet_batch_fused_gdn_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_deltanet_batch_fused_gdn failed: %s\n",
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
    fn = [g_library newFunctionWithName:@"qw3_residual_rmsnorm_batch_update_x0"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_residual_rmsnorm_batch_update_x0 not found\n");
        return 0;
    }
    g_residual_rmsnorm_batch_update_x0_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_residual_rmsnorm_batch_update_x0_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_residual_rmsnorm_batch_update_x0 failed: %s\n",
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
    fn = [g_library newFunctionWithName:@"qw3_silu_mul_rows_offsets"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_silu_mul_rows_offsets not found\n");
        return 0;
    }
    g_silu_mul_rows_offsets_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                           error:&error];
    if (!g_silu_mul_rows_offsets_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_silu_mul_rows_offsets failed: %s\n",
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
    fn = [g_library newFunctionWithName:@"qw3_sigmoid_scale_scratch_add_x0_rows"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_sigmoid_scale_scratch_add_x0_rows not found\n");
        return 0;
    }
    g_sigmoid_scale_scratch_add_x0_rows_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_sigmoid_scale_scratch_add_x0_rows_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_sigmoid_scale_scratch_add_x0_rows failed: %s\n",
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
    fn = [g_library newFunctionWithName:@"qw3_router_top8_batch"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_router_top8_batch not found\n");
        return 0;
    }
    g_router_top8_batch_pipeline = [g_device newComputePipelineStateWithFunction:fn
                                                                           error:&error];
    if (!g_router_top8_batch_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_router_top8_batch failed: %s\n",
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
    g_batch_concurrent = 0;
    if (getenv("QW3_METAL_RETAINED_COMMAND_BUFFERS") == NULL) {
        g_batch_cb = [g_queue commandBufferWithUnretainedReferences];
    } else {
        g_batch_cb = [g_queue commandBuffer];
    }
    return g_batch_cb != nil;
}

int qw3_metal_begin_commands_concurrent(void) {
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (g_batch_cb) return 0;
    g_batch_concurrent = 1;
    if (getenv("QW3_METAL_RETAINED_COMMAND_BUFFERS") == NULL) {
        g_batch_cb = [g_queue commandBufferWithUnretainedReferences];
    } else {
        g_batch_cb = [g_queue commandBuffer];
    }
    if (!g_batch_cb) g_batch_concurrent = 0;
    return g_batch_cb != nil;
}

int qw3_metal_batch_barrier(void) {
    if (!g_batch_cb || !g_batch_concurrent) return 1;
    if (!g_batch_enc) return 1;
    [g_batch_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    return 1;
}

int qw3_metal_flush_commands(void) {
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!g_batch_cb) return 0;
    qw3_metal_close_batch_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    [cb commit];
    [g_pending_cbs addObject:cb];
    if (getenv("QW3_METAL_RETAINED_COMMAND_BUFFERS") == NULL) {
        g_batch_cb = [g_queue commandBufferWithUnretainedReferences];
    } else {
        g_batch_cb = [g_queue commandBuffer];
    }
    if (!g_batch_cb) {
        (void)qw3_metal_wait_pending_command_buffers("command batch");
        g_batch_concurrent = 0;
        return 0;
    }
    return 1;
}

int qw3_metal_commit_commands(void) {
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!g_batch_cb) return 1;
    qw3_metal_close_batch_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    g_batch_concurrent = 0;
    [cb commit];
    [g_pending_cbs addObject:cb];
    return 1;
}

int qw3_metal_end_commands(void) {
    if (!g_batch_cb) return 0;
    qw3_metal_close_batch_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    g_batch_concurrent = 0;
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

static id<MTLBuffer> qw3_metal_iq3s_expanded_kgrid_buffer(void) {
    if (!g_device) return nil;
    if (!g_iq3s_expanded_kgrid_buffer) {
        uint8_t grid[512 * 4];
        for (uint32_t i = 0; i < 512; ++i) {
            const uint16_t packed = g_iq3s_kgrid[i];
            for (uint32_t j = 0; j < 4; ++j) {
                grid[4 * i + j] =
                    (uint8_t)(2 * ((packed >> (3 * j)) & 7) + 1);
            }
        }
        g_iq3s_expanded_kgrid_buffer =
            [g_device newBufferWithBytes:grid
                                   length:sizeof(grid)
                                  options:MTLResourceStorageModeShared];
    }
    return g_iq3s_expanded_kgrid_buffer;
}

static int qw3_metal_session_ensure_prefill_buffers(QW3MetalSessionObj *obj,
                                                     uint32_t n_tokens,
                                                     uint64_t x0_bytes,
                                                     uint64_t x1_bytes) {
    if (!obj || n_tokens == 0) return 0;
    const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(uint32_t);
    if (!obj.prefillTokens || obj.prefillTokens.length < token_bytes) {
        obj.prefillTokens = [g_device newBufferWithLength:(NSUInteger)qw3_metal_session_align(token_bytes)
                                                  options:MTLResourceStorageModeShared];
    }
    if (x0_bytes && (!obj.prefillX0 || obj.prefillX0.length < x0_bytes)) {
        obj.prefillX0 = qw3_metal_new_private_buffer(x0_bytes);
    }
    if (x1_bytes && (!obj.prefillX1 || obj.prefillX1.length < x1_bytes)) {
        obj.prefillX1 = qw3_metal_new_private_buffer(x1_bytes);
    }
    if (!obj.prefillTokens ||
        (x0_bytes && !obj.prefillX0) ||
        (x1_bytes && !obj.prefillX1)) {
        return 0;
    }
    if (n_tokens > obj.prefillCap) obj.prefillCap = n_tokens;
    return 1;
}

static int qw3_metal_session_ensure_prefill_scratch(QW3MetalSessionObj *obj,
                                                     uint32_t n_tokens,
                                                     uint64_t bytes) {
    if (!obj || n_tokens == 0 || bytes == 0) return 0;
    if (!obj.prefillScratch || obj.prefillScratch.length < bytes) {
        obj.prefillScratch = qw3_metal_new_private_buffer(bytes);
    }
    if (!obj.prefillScratch) return 0;
    if (n_tokens > obj.prefillCap) obj.prefillCap = n_tokens;
    return 1;
}

static int qw3_metal_session_ensure_router_buffers(QW3MetalSessionObj *obj,
                                                    uint32_t n_slots) {
    if (!obj || n_slots == 0) return 0;
    const uint64_t ids_bytes = (uint64_t)n_slots * sizeof(int32_t);
    const uint64_t weights_bytes = (uint64_t)n_slots * sizeof(float);
    if (!obj.routerIds || obj.routerIds.length < ids_bytes) {
        obj.routerIds = qw3_metal_new_private_buffer(ids_bytes);
    }
    if (!obj.routerWeights || obj.routerWeights.length < weights_bytes) {
        obj.routerWeights = qw3_metal_new_private_buffer(weights_bytes);
    }
    return obj.routerIds && obj.routerWeights;
}

static int qw3_metal_session_ensure_moe_map_buffers(QW3MetalSessionObj *obj,
                                                     uint32_t n_tokens,
                                                     uint32_t n_active,
                                                     uint32_t n_expert,
                                                     uint32_t n_embd,
                                                     uint32_t n_ff,
                                                     int need_gateup,
                                                     int need_down,
                                                     int need_mid_f16) {
    if (!obj || n_tokens == 0 || n_active == 0 ||
        n_expert == 0 || (need_down && n_embd == 0) ||
        (need_gateup && n_ff == 0)) {
        return 0;
    }
    const uint64_t counts_bytes = (uint64_t)n_expert * sizeof(uint32_t);
    const uint64_t pair_ids_bytes =
        (uint64_t)n_expert * (uint64_t)n_tokens * sizeof(int32_t);
    const uint64_t down_bytes = need_down ?
        (uint64_t)n_tokens * (uint64_t)n_active *
        (uint64_t)n_embd * sizeof(float) : 0;
    const uint64_t gateup_bytes = need_gateup ?
        (uint64_t)n_tokens * (uint64_t)n_active *
        (uint64_t)n_ff * sizeof(float) : 0;
    const uint64_t mid_f16_bytes = need_mid_f16 ?
        (uint64_t)n_tokens * (uint64_t)n_active *
        (uint64_t)n_ff * sizeof(uint16_t) : 0;
    if (!obj.moeExpertCounts || obj.moeExpertCounts.length < counts_bytes) {
        obj.moeExpertCounts = qw3_metal_new_private_buffer(counts_bytes);
    }
    if (!obj.moePairIds || obj.moePairIds.length < pair_ids_bytes) {
        obj.moePairIds = qw3_metal_new_private_buffer(pair_ids_bytes);
    }
    if (need_gateup &&
        (!obj.prefillMoeGate || obj.prefillMoeGate.length < gateup_bytes)) {
        obj.prefillMoeGate = qw3_metal_new_private_buffer(gateup_bytes);
    }
    if (need_gateup &&
        (!obj.prefillMoeUp || obj.prefillMoeUp.length < gateup_bytes)) {
        obj.prefillMoeUp = qw3_metal_new_private_buffer(gateup_bytes);
    }
    if (need_mid_f16 &&
        (!obj.prefillMoeMidF16 || obj.prefillMoeMidF16.length < mid_f16_bytes)) {
        obj.prefillMoeMidF16 = qw3_metal_new_private_buffer(mid_f16_bytes);
    }
    if (need_down &&
        (!obj.prefillMoeDown || obj.prefillMoeDown.length < down_bytes)) {
        obj.prefillMoeDown = qw3_metal_new_private_buffer(down_bytes);
    }
    return obj.moeExpertCounts && obj.moePairIds &&
           (!need_gateup || (obj.prefillMoeGate && obj.prefillMoeUp)) &&
           (!need_mid_f16 || obj.prefillMoeMidF16) &&
           (!need_down || obj.prefillMoeDown);
}

qw3_metal_session *qw3_metal_session_create(uint32_t ctx_size,
                                            uint32_t vocab_size) {
    if (!g_initialized || !g_device || ctx_size == 0 || vocab_size == 0) {
        return NULL;
    }

    const char *kv_q8_env = getenv("QW3_METAL_KV_Q8_0");
    const BOOL gqa_kv_q8 = kv_q8_env && strcmp(kv_q8_env, "0") != 0;
    const BOOL gqa_split_q8 =
        gqa_kv_q8 && getenv("QW3_METAL_LEGACY_Q8_ATTN") == NULL;
    const uint32_t gqa_max_q8_splits = !gqa_split_q8 ||
        getenv("QW3_METAL_Q8_SPLIT_32") != NULL ? 32u :
        (getenv("QW3_METAL_Q8_SPLIT_64") != NULL ? 64u :
         (getenv("QW3_METAL_Q8_SPLIT_128") != NULL ? 128u : 256u));
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
    const uint64_t gqa_attn_partial_bytes = gqa_split_q8 ?
        (uint64_t)gqa_max_q8_splits * QW3_METAL_N_HEAD *
            (QW3_METAL_N_HEAD_DIM + 2ull) * sizeof(float) :
        0ull;

    QW3MetalSessionObj *obj = [[QW3MetalSessionObj alloc] init];
    obj.ctxSize = ctx_size;
    obj.vocabSize = vocab_size;
    obj.gqaKvQ8 = gqa_kv_q8;
    obj.gqaSplitQ8 = gqa_split_q8;
    obj.gqaMaxQ8Splits = gqa_max_q8_splits;
    qw3_metal_session_info info = {
        .gqa_kv_bytes = 2 * gqa_kv_bytes,
        .deltanet_state_bytes = deltanet_state_bytes,
        .conv_state_bytes = conv_state_bytes,
        .logits_bytes = logits_bytes,
        .scratch_bytes = scratch_bytes + qkv_conv_bytes + 2 * qk_norm_bytes +
                         2 * inner_bytes + 4 * gqa_q_bytes + 2 * gqa_kv_token_bytes +
                         gqa_attn_partial_bytes,
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
    if (gqa_attn_partial_bytes > 0) {
        obj.gqaAttnPartial = qw3_metal_new_private_buffer(gqa_attn_partial_bytes);
    }
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
        (gqa_split_q8 && !obj.gqaAttnPartial) ||
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
    if (obj.gqaAttnPartial.length > 0) {
        [blit fillBuffer:obj.gqaAttnPartial
                   range:NSMakeRange(0, obj.gqaAttnPartial.length) value:0];
    }
    if (obj.prefillX0.length > 0) {
        [blit fillBuffer:obj.prefillX0
                   range:NSMakeRange(0, obj.prefillX0.length) value:0];
    }
    if (obj.prefillX1.length > 0) {
        [blit fillBuffer:obj.prefillX1
                   range:NSMakeRange(0, obj.prefillX1.length) value:0];
    }
    if (obj.prefillScratch.length > 0) {
        [blit fillBuffer:obj.prefillScratch
                   range:NSMakeRange(0, obj.prefillScratch.length) value:0];
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

int qw3_metal_session_batch_embed_q8_0(qw3_metal_session *s,
                                       uint64_t tensor_offset,
                                       const uint32_t *tokens,
                                       uint32_t n_tokens,
                                       uint32_t n_embd) {
    if (!s || !s->obj || !tokens || n_tokens == 0 ||
        n_embd == 0 || (n_embd % 32) != 0) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t row_bytes = (uint64_t)(n_embd / 32u) * 34ull;
    const uint64_t out_bytes = (uint64_t)n_tokens * n_embd * sizeof(float);
    if (!qw3_metal_session_ensure_prefill_buffers(obj, n_tokens,
                                                  out_bytes, 0)) {
        return 0;
    }
    memcpy(obj.prefillTokens.contents, tokens,
           (size_t)n_tokens * sizeof(uint32_t));

    const uint64_t tensor_bytes = row_bytes * (uint64_t)obj.vocabSize;
    uint64_t inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(tensor_offset,
                                                tensor_bytes, &inner);
    if (!wb) {
        fprintf(stderr, "qw3: Metal batch embedding tensor is outside mapped model views\n");
        return 0;
    }

    struct {
        uint32_t n_embd;
        uint32_t row_bytes;
        uint32_t n_tokens;
    } args = { n_embd, (uint32_t)row_bytes, n_tokens };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_embed_q8_0_batch_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)inner atIndex:1];
    [enc setBuffer:obj.prefillTokens offset:0 atIndex:2];
    [enc setBuffer:obj.prefillX0 offset:0 atIndex:3];
    NSUInteger threads = g_embed_q8_0_batch_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n_tokens * n_embd, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch embedding command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_batch_rmsnorm_weight_f32_x0_inplace(
    qw3_metal_session *s, uint64_t weight_offset, uint32_t n_tokens,
    uint32_t n, float eps) {
    if (!s || !s->obj || n_tokens == 0 || n == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_tokens * n * sizeof(float);
    if (!qw3_metal_session_ensure_prefill_buffers(obj, n_tokens,
                                                  x_bytes, 0)) {
        return 0;
    }
    uint64_t inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(weight_offset,
                                                (uint64_t)n * sizeof(float),
                                                &inner);
    if (!wb) return 0;
    struct {
        uint32_t n;
        float eps;
        uint32_t n_rows;
    } args = { n, eps, n_tokens };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_rmsnorm_weight_f32_rows_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.prefillX0 offset:0 atIndex:1];
    [enc setBuffer:wb offset:(NSUInteger)inner atIndex:2];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_rmsnorm_weight_f32_rows_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreadgroups:MTLSizeMake(n_tokens, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch rmsnorm command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_batch_rmsnorm_weight_f32_x0_to_x1(
    qw3_metal_session *s, uint64_t weight_offset, uint32_t n_tokens,
    uint32_t n, float eps) {
    if (!s || !s->obj || n_tokens == 0 || n == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_tokens * n * sizeof(float);
    if (!qw3_metal_session_ensure_prefill_buffers(obj, n_tokens,
                                                  x_bytes, x_bytes)) {
        return 0;
    }
    uint64_t inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(weight_offset,
                                                (uint64_t)n * sizeof(float),
                                                &inner);
    if (!wb) return 0;
    struct {
        uint32_t n;
        float eps;
        uint32_t n_rows;
    } args = { n, eps, n_tokens };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_rmsnorm_weight_f32_rows_to_out_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.prefillX0 offset:0 atIndex:1];
    [enc setBuffer:wb offset:(NSUInteger)inner atIndex:2];
    [enc setBuffer:obj.prefillX1 offset:0 atIndex:3];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads =
        g_rmsnorm_weight_f32_rows_to_out_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreadgroups:MTLSizeMake(n_tokens, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch rmsnorm x0->x1 command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

static int qw3_metal_use_q8_mm(uint32_t n_tokens, uint32_t n_in,
                               uint32_t n_out) {
    if (getenv("QW3_METAL_Q8_MM_DISABLE")) return 0;
    if (n_tokens < 32u) return 0;
    if ((n_in % 32u) != 0) return 0;
    if (n_out < 64u) return 0;
    return 1;
}

static int qw3_metal_encode_batch_matmul_q8_0(
    id<MTLBuffer> wbuf, NSUInteger woff, id<MTLBuffer> xbuf,
    NSUInteger xoff, id<MTLBuffer> outbuf, NSUInteger outoff,
    uint32_t n_tokens, uint32_t n_in, uint32_t n_out, uint32_t in_stride,
    uint32_t out_stride, uint32_t row_bytes) {
    if (qw3_metal_use_q8_mm(n_tokens, n_in, n_out)) {
        struct {
            uint32_t n_in;
            uint32_t n_out;
            uint32_t row_bytes;
            uint32_t n_tokens;
            uint32_t in_stride;
            uint32_t out_stride;
        } args = {
            n_in, n_out, row_bytes, n_tokens, in_stride, out_stride
        };
        const int bc_out = ((n_out % 64u) != 0 || (n_tokens % 32u) != 0);
        id<MTLComputePipelineState> pipeline =
            bc_out ? g_matmul_q8_0_mm_bc_pipeline : g_matmul_q8_0_mm_pipeline;
        int owned = 0;
        id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
        id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:woff atIndex:1];
        [enc setBuffer:xbuf offset:xoff atIndex:2];
        [enc setBuffer:outbuf offset:outoff atIndex:3];
        [enc setThreadgroupMemoryLength:(bc_out ? 8192u : 6144u) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((n_tokens + 31u) / 32u,
                                              (n_out + 63u) / 64u, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);
        if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
        if (cb.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "qw3: Metal Q8_0 prefill mm command failed: %s\n",
                    [[cb.error localizedDescription] UTF8String]);
            return 0;
        }
        return 1;
    }

    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
        uint32_t n_tokens;
        uint32_t in_offset;
        uint32_t in_stride;
        uint32_t out_offset;
        uint32_t out_stride;
    } args = {
        n_in, n_out, row_bytes, n_tokens, 0, in_stride, 0, out_stride
    };
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_matmul_q8_0_batch4_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wbuf offset:woff atIndex:1];
    [enc setBuffer:xbuf offset:xoff atIndex:2];
    [enc setBuffer:outbuf offset:outoff atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((n_out + 3u) / 4u,
                                          (n_tokens + 3u) / 4u, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch q8_0 matmul command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_batch_matmul_q8_0_x0_to_x1(qw3_metal_session *s,
                                                 uint64_t tensor_offset,
                                                 uint32_t n_tokens,
                                                 uint32_t n_in,
                                                 uint32_t n_out) {
    if (!s || !s->obj || n_tokens == 0 || n_in == 0 || n_out == 0 ||
        (n_in % 32) != 0) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_tokens * n_in * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tokens * n_out * sizeof(float);
    if (!qw3_metal_session_ensure_prefill_buffers(obj, n_tokens,
                                                  x_bytes, out_bytes)) {
        return 0;
    }

    const uint64_t row_bytes = (uint64_t)(n_in / 32u) * 34ull;
    const uint64_t tensor_bytes = row_bytes * (uint64_t)n_out;
    uint64_t inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(tensor_offset,
                                                tensor_bytes, &inner);
    if (!wb) {
        fprintf(stderr, "qw3: Metal batch q8_0 tensor is outside mapped model views\n");
        return 0;
    }

    return qw3_metal_encode_batch_matmul_q8_0(
        wb, (NSUInteger)inner, obj.prefillX0, 0, obj.prefillX1, 0,
        n_tokens, n_in, n_out, n_in, n_out, (uint32_t)row_bytes);
}

int qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(
    qw3_metal_session *s, uint64_t tensor_offset, uint32_t n_tokens,
    uint32_t n_in, uint32_t n_out, uint32_t out_offset,
    uint32_t out_stride) {
    if (!s || !s->obj || n_tokens == 0 || n_in == 0 || n_out == 0 ||
        (n_in % 32) != 0 || out_stride == 0 ||
        out_offset > out_stride || n_out > out_stride - out_offset) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_tokens * n_in * sizeof(float);
    const uint64_t out_bytes =
        (uint64_t)n_tokens * out_stride * sizeof(float);
    if (!obj.prefillX1 || obj.prefillX1.length < x_bytes) return 0;
    if (!qw3_metal_session_ensure_prefill_scratch(obj, n_tokens, out_bytes)) {
        return 0;
    }

    const uint64_t row_bytes = (uint64_t)(n_in / 32u) * 34ull;
    const uint64_t tensor_bytes = row_bytes * (uint64_t)n_out;
    uint64_t inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(tensor_offset,
                                                tensor_bytes, &inner);
    if (!wb) {
        fprintf(stderr, "qw3: Metal batch q8_0 tensor is outside mapped model views\n");
        return 0;
    }

    return qw3_metal_encode_batch_matmul_q8_0(
        wb, (NSUInteger)inner, obj.prefillX1, 0, obj.prefillScratch,
        (NSUInteger)out_offset * sizeof(float), n_tokens, n_in, n_out,
        n_in, out_stride, (uint32_t)row_bytes);
}

int qw3_metal_session_batch_matmul_q8_0_scratch_to_scratch(
    qw3_metal_session *s, uint64_t tensor_offset, uint32_t n_tokens,
    uint32_t n_in, uint32_t n_out, uint32_t in_offset,
    uint32_t out_offset, uint32_t stride) {
    if (!s || !s->obj || n_tokens == 0 || n_in == 0 || n_out == 0 ||
        (n_in % 32) != 0 || stride == 0 ||
        in_offset > stride || out_offset > stride ||
        n_in > stride - in_offset ||
        n_out > stride - out_offset) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t scratch_bytes = (uint64_t)n_tokens * stride * sizeof(float);
    if (!obj.prefillScratch || obj.prefillScratch.length < scratch_bytes) {
        return 0;
    }

    const uint64_t row_bytes = (uint64_t)(n_in / 32u) * 34ull;
    const uint64_t tensor_bytes = row_bytes * (uint64_t)n_out;
    uint64_t inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(tensor_offset,
                                                tensor_bytes, &inner);
    if (!wb) {
        fprintf(stderr, "qw3: Metal batch q8_0 tensor is outside mapped model views\n");
        return 0;
    }

    return qw3_metal_encode_batch_matmul_q8_0(
        wb, (NSUInteger)inner, obj.prefillScratch,
        (NSUInteger)in_offset * sizeof(float), obj.prefillScratch,
        (NSUInteger)out_offset * sizeof(float), n_tokens, n_in, n_out,
        stride, stride, (uint32_t)row_bytes);
}

int qw3_metal_session_batch_matmul_f32_x0_to_x1(qw3_metal_session *s,
                                                uint64_t tensor_offset,
                                                uint32_t n_tokens,
                                                uint32_t n_in,
                                                uint32_t n_out) {
    if (!s || !s->obj || n_tokens == 0 || n_in == 0 || n_out == 0) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_tokens * n_in * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tokens * n_out * sizeof(float);
    if (!qw3_metal_session_ensure_prefill_buffers(obj, n_tokens,
                                                  x_bytes, out_bytes)) {
        return 0;
    }

    const uint64_t tensor_bytes =
        (uint64_t)n_in * (uint64_t)n_out * sizeof(float);
    uint64_t inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(tensor_offset,
                                                tensor_bytes, &inner);
    if (!wb) {
        fprintf(stderr, "qw3: Metal batch f32 tensor is outside mapped model views\n");
        return 0;
    }

    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t n_tokens;
        uint32_t in_offset;
        uint32_t in_stride;
        uint32_t out_offset;
        uint32_t out_stride;
    } args = { n_in, n_out, n_tokens, 0, n_in, 0, n_out };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_matmul_f32_batch4_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)inner atIndex:1];
    [enc setBuffer:obj.prefillX0 offset:0 atIndex:2];
    [enc setBuffer:obj.prefillX1 offset:0 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((n_out + 3u) / 4u,
                                          (n_tokens + 3u) / 4u, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch f32 matmul command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_batch_matmul_f32_x1_to_scratch(
    qw3_metal_session *s, uint64_t tensor_offset, uint32_t n_tokens,
    uint32_t n_in, uint32_t n_out, uint32_t out_offset,
    uint32_t out_stride) {
    if (!s || !s->obj || n_tokens == 0 || n_in == 0 || n_out == 0 ||
        out_stride == 0 || out_offset > out_stride ||
        n_out > out_stride - out_offset) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_tokens * n_in * sizeof(float);
    const uint64_t out_bytes =
        (uint64_t)n_tokens * out_stride * sizeof(float);
    if (!obj.prefillX1 || obj.prefillX1.length < x_bytes) return 0;
    if (!qw3_metal_session_ensure_prefill_scratch(obj, n_tokens, out_bytes)) {
        return 0;
    }

    const uint64_t tensor_bytes =
        (uint64_t)n_in * (uint64_t)n_out * sizeof(float);
    uint64_t inner = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(tensor_offset,
                                                tensor_bytes, &inner);
    if (!wb) {
        fprintf(stderr, "qw3: Metal batch f32 tensor is outside mapped model views\n");
        return 0;
    }

    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t n_tokens;
        uint32_t in_offset;
        uint32_t in_stride;
        uint32_t out_offset;
        uint32_t out_stride;
    } args = { n_in, n_out, n_tokens, 0, n_in, out_offset, out_stride };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_matmul_f32_batch4_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)inner atIndex:1];
    [enc setBuffer:obj.prefillX1 offset:0 atIndex:2];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((n_out + 3u) / 4u,
                                          (n_tokens + 3u) / 4u, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch f32 scratch matmul command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_batch_matmul_f32_pair_x0_to_x1(
    qw3_metal_session *s, uint64_t tensor_a_offset, uint64_t tensor_b_offset,
    uint32_t n_tokens, uint32_t n_in, uint32_t n_out,
    uint32_t out_a_offset, uint32_t out_b_offset, uint32_t out_stride) {
    if (!s || !s->obj || n_tokens == 0 || n_in == 0 || n_out == 0 ||
        out_stride == 0 ||
        out_a_offset > out_stride || out_b_offset > out_stride ||
        n_out > out_stride - out_a_offset ||
        n_out > out_stride - out_b_offset) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_tokens * n_in * sizeof(float);
    const uint64_t out_bytes =
        (uint64_t)n_tokens * out_stride * sizeof(float);
    if (!qw3_metal_session_ensure_prefill_buffers(obj, n_tokens,
                                                  x_bytes, out_bytes)) {
        return 0;
    }

    const uint64_t tensor_bytes =
        (uint64_t)n_in * (uint64_t)n_out * sizeof(float);
    uint64_t inner_a = 0, inner_b = 0;
    id<MTLBuffer> wa =
        qw3_metal_model_view_for(tensor_a_offset, tensor_bytes, &inner_a);
    id<MTLBuffer> wb =
        qw3_metal_model_view_for(tensor_b_offset, tensor_bytes, &inner_b);
    if (!wa || !wb) {
        fprintf(stderr, "qw3: Metal batch f32 pair tensor is outside mapped model views\n");
        return 0;
    }

    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t n_tokens;
        uint32_t out_a_offset;
        uint32_t out_b_offset;
        uint32_t out_stride;
    } args = {
        n_in, n_out, n_tokens, out_a_offset, out_b_offset, out_stride
    };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_matmul_f32_pair_batch4_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wa offset:(NSUInteger)inner_a atIndex:1];
    [enc setBuffer:wb offset:(NSUInteger)inner_b atIndex:2];
    [enc setBuffer:obj.prefillX0 offset:0 atIndex:3];
    [enc setBuffer:obj.prefillX1 offset:0 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((n_out + 3u) / 4u,
                                          (n_tokens + 3u) / 4u, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch f32 pair matmul command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_batch_matmul_f32_pair_x1_to_scratch(
    qw3_metal_session *s, uint64_t tensor_a_offset, uint64_t tensor_b_offset,
    uint32_t n_tokens, uint32_t n_in, uint32_t n_out,
    uint32_t out_a_offset, uint32_t out_b_offset, uint32_t out_stride) {
    if (!s || !s->obj || n_tokens == 0 || n_in == 0 || n_out == 0 ||
        out_stride == 0 ||
        out_a_offset > out_stride || out_b_offset > out_stride ||
        n_out > out_stride - out_a_offset ||
        n_out > out_stride - out_b_offset) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_tokens * n_in * sizeof(float);
    const uint64_t out_bytes =
        (uint64_t)n_tokens * out_stride * sizeof(float);
    if (!obj.prefillX1 || obj.prefillX1.length < x_bytes) return 0;
    if (!qw3_metal_session_ensure_prefill_scratch(obj, n_tokens, out_bytes)) {
        return 0;
    }

    const uint64_t tensor_bytes =
        (uint64_t)n_in * (uint64_t)n_out * sizeof(float);
    uint64_t inner_a = 0, inner_b = 0;
    id<MTLBuffer> wa =
        qw3_metal_model_view_for(tensor_a_offset, tensor_bytes, &inner_a);
    id<MTLBuffer> wb =
        qw3_metal_model_view_for(tensor_b_offset, tensor_bytes, &inner_b);
    if (!wa || !wb) {
        fprintf(stderr, "qw3: Metal batch f32 pair tensor is outside mapped model views\n");
        return 0;
    }

    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t n_tokens;
        uint32_t out_a_offset;
        uint32_t out_b_offset;
        uint32_t out_stride;
    } args = {
        n_in, n_out, n_tokens, out_a_offset, out_b_offset, out_stride
    };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_matmul_f32_pair_batch4_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wa offset:(NSUInteger)inner_a atIndex:1];
    [enc setBuffer:wb offset:(NSUInteger)inner_b atIndex:2];
    [enc setBuffer:obj.prefillX1 offset:0 atIndex:3];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((n_out + 3u) / 4u,
                                          (n_tokens + 3u) / 4u, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch f32 pair scratch matmul command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_batch_conv1d_step_from_scratch(
    qw3_metal_session *s, uint64_t weight_offset, uint32_t layer_slot,
    uint32_t n_tokens, uint32_t n_channels, uint32_t qkv_offset,
    uint32_t conv_offset, uint32_t stride) {
    if (!s || !s->obj || n_tokens == 0 || n_channels == 0 ||
        layer_slot >= QW3_METAL_N_LINEAR_LAYERS ||
        stride == 0 || qkv_offset > stride || conv_offset > stride ||
        n_channels > stride - qkv_offset ||
        n_channels > stride - conv_offset) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t scratch_bytes =
        (uint64_t)n_tokens * stride * sizeof(float);
    const uint64_t state_bytes = (uint64_t)n_channels * 3ull * sizeof(float);
    const uint64_t state_offset = (uint64_t)layer_slot * state_bytes;
    if (!obj.prefillScratch || obj.prefillScratch.length < scratch_bytes ||
        !obj.convState || obj.convState.length < state_offset ||
        obj.convState.length - state_offset < state_bytes) {
        return 0;
    }

    const uint64_t weight_bytes = (uint64_t)n_channels * 4ull * sizeof(float);
    uint64_t weight_inner = 0;
    id<MTLBuffer> wb =
        qw3_metal_model_view_for(weight_offset, weight_bytes, &weight_inner);
    if (!wb) {
        fprintf(stderr, "qw3: Metal batch DeltaNet conv weight is outside mapped model views\n");
        return 0;
    }

    struct {
        uint32_t n_channels;
        uint32_t n_tokens;
        uint32_t qkv_offset;
        uint32_t conv_offset;
        uint32_t stride;
    } args = { n_channels, n_tokens, qkv_offset, conv_offset, stride };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_deltanet_conv1d_batch_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)weight_inner atIndex:1];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:2];
    [enc setBuffer:obj.convState offset:(NSUInteger)state_offset atIndex:3];
    NSUInteger threads = g_deltanet_conv1d_batch_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n_tokens * n_channels, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [enc setComputePipelineState:g_deltanet_conv1d_batch_state_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:1];
    [enc setBuffer:obj.convState offset:(NSUInteger)state_offset atIndex:2];
    NSUInteger state_threads =
        g_deltanet_conv1d_batch_state_pipeline.maxTotalThreadsPerThreadgroup;
    if (state_threads > 256) state_threads = 256;
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n_channels, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(state_threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch DeltaNet conv command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_batch_l2norm_qk_from_scratch(
    qw3_metal_session *s, uint32_t n_tokens, uint32_t conv_offset,
    uint32_t stride, uint32_t n_qk_heads, uint32_t head_dim, float eps) {
    if (!s || !s->obj || n_tokens == 0 || stride == 0 ||
        n_qk_heads == 0 || head_dim == 0) {
        return 0;
    }
    const uint32_t qk_n = n_qk_heads * head_dim;
    if (conv_offset > stride || qk_n > stride - conv_offset ||
        qk_n > stride - (conv_offset + qk_n)) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t bytes = (uint64_t)n_tokens * stride * sizeof(float);
    if (!obj.prefillScratch || obj.prefillScratch.length < bytes) return 0;

    struct {
        uint32_t n_tokens;
        uint32_t conv_offset;
        uint32_t stride;
        uint32_t n_qk_heads;
        uint32_t head_dim;
        float eps;
    } args = { n_tokens, conv_offset, stride, n_qk_heads, head_dim, eps };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_l2norm_qk_batch_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:1];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads = g_l2norm_qk_batch_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_qk_heads * n_tokens * 2u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch DeltaNet q/k l2norm command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_batch_deltanet_fused_gdn_from_scratch(
    qw3_metal_session *s, uint64_t dt_bias_offset, uint64_t a_offset,
    uint64_t norm_weight_offset, uint32_t layer_slot, uint32_t n_tokens,
    uint32_t conv_offset, uint32_t z_offset, uint32_t alpha_offset,
    uint32_t beta_offset, uint32_t inner_offset, uint32_t stride,
    uint32_t q_heads, uint32_t v_heads, uint32_t head_dim, float eps) {
    if (!s || !s->obj || n_tokens == 0 || stride == 0 ||
        q_heads == 0 || v_heads == 0 || head_dim == 0 ||
        layer_slot >= QW3_METAL_N_LINEAR_LAYERS ||
        head_dim > 256) {
        return 0;
    }
    const uint32_t qk_n = q_heads * head_dim;
    const uint32_t inner_n = v_heads * head_dim;
    if (conv_offset > stride || z_offset > stride ||
        alpha_offset > stride || beta_offset > stride ||
        inner_offset > stride ||
        2u * qk_n + inner_n > stride - conv_offset ||
        inner_n > stride - z_offset ||
        v_heads > stride - alpha_offset ||
        v_heads > stride - beta_offset ||
        inner_n > stride - inner_offset) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t scratch_bytes =
        (uint64_t)n_tokens * stride * sizeof(float);
    const uint64_t state_bytes =
        (uint64_t)v_heads * head_dim * head_dim * sizeof(float);
    const uint64_t state_offset = (uint64_t)layer_slot * state_bytes;
    if (!obj.prefillScratch || obj.prefillScratch.length < scratch_bytes ||
        !obj.deltanetState || obj.deltanetState.length < state_offset ||
        obj.deltanetState.length - state_offset < state_bytes) {
        return 0;
    }

    const uint64_t gates_bytes = (uint64_t)v_heads * sizeof(float);
    const uint64_t weight_bytes = (uint64_t)head_dim * sizeof(float);
    uint64_t dt_inner = 0, a_inner = 0, w_inner = 0;
    id<MTLBuffer> dtb =
        qw3_metal_model_view_for(dt_bias_offset, gates_bytes, &dt_inner);
    id<MTLBuffer> ab =
        qw3_metal_model_view_for(a_offset, gates_bytes, &a_inner);
    id<MTLBuffer> wb =
        qw3_metal_model_view_for(norm_weight_offset, weight_bytes, &w_inner);
    if (!dtb || !ab || !wb ||
        head_dim > g_deltanet_batch_fused_gdn_pipeline.maxTotalThreadsPerThreadgroup) {
        fprintf(stderr, "qw3: Metal batch Gated DeltaNet weights are outside mapped model views\n");
        return 0;
    }

    struct {
        uint32_t q_heads;
        uint32_t v_heads;
        uint32_t head_dim;
        uint32_t n_tokens;
        uint32_t conv_offset;
        uint32_t z_offset;
        uint32_t alpha_offset;
        uint32_t beta_offset;
        uint32_t inner_offset;
        uint32_t stride;
        float eps;
    } args = {
        q_heads, v_heads, head_dim, n_tokens, conv_offset, z_offset,
        alpha_offset, beta_offset, inner_offset, stride, eps
    };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_deltanet_batch_fused_gdn_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.deltanetState offset:(NSUInteger)state_offset atIndex:1];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:2];
    [enc setBuffer:dtb offset:(NSUInteger)dt_inner atIndex:3];
    [enc setBuffer:ab offset:(NSUInteger)a_inner atIndex:4];
    [enc setBuffer:wb offset:(NSUInteger)w_inner atIndex:5];
    [enc setThreadgroupMemoryLength:64 * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(v_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch Gated DeltaNet command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_batch_residual_rmsnorm_update_x0_from_scratch(
    qw3_metal_session *s, uint64_t weight_offset, uint32_t n_tokens,
    uint32_t n, uint32_t residual_offset, uint32_t residual_stride,
    float eps) {
    if (!s || !s->obj || n_tokens == 0 || n == 0 ||
        residual_stride == 0 || residual_offset > residual_stride ||
        n > residual_stride - residual_offset) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_tokens * n * sizeof(float);
    const uint64_t scratch_bytes =
        (uint64_t)n_tokens * residual_stride * sizeof(float);
    if (!obj.prefillX0 || obj.prefillX0.length < x_bytes ||
        !obj.prefillScratch || obj.prefillScratch.length < scratch_bytes) {
        return 0;
    }
    if (!qw3_metal_session_ensure_prefill_buffers(obj, n_tokens, x_bytes,
                                                  x_bytes)) {
        return 0;
    }

    uint64_t inner = 0;
    id<MTLBuffer> wb =
        qw3_metal_model_view_for(weight_offset, (uint64_t)n * sizeof(float),
                                 &inner);
    if (!wb) {
        fprintf(stderr, "qw3: Metal batch residual RMSNorm weight is outside mapped model views\n");
        return 0;
    }

    struct {
        uint32_t n;
        float eps;
        uint32_t n_tokens;
        uint32_t residual_offset;
        uint32_t residual_stride;
    } args = { n, eps, n_tokens, residual_offset, residual_stride };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_residual_rmsnorm_batch_update_x0_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.prefillX0 offset:0 atIndex:1];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:2];
    [enc setBuffer:wb offset:(NSUInteger)inner atIndex:3];
    [enc setBuffer:obj.prefillX1 offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads =
        g_residual_rmsnorm_batch_update_x0_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreadgroups:MTLSizeMake(n_tokens, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch residual RMSNorm command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_batch_silu_mul_scratch_to_scratch(
    qw3_metal_session *s, uint32_t n_tokens, uint32_t n,
    uint32_t a_offset, uint32_t b_offset, uint32_t out_offset,
    uint32_t stride) {
    if (!s || !s->obj || n_tokens == 0 || n == 0 || stride == 0 ||
        a_offset > stride || b_offset > stride || out_offset > stride ||
        n > stride - a_offset || n > stride - b_offset ||
        n > stride - out_offset) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t bytes = (uint64_t)n_tokens * stride * sizeof(float);
    if (!obj.prefillScratch || obj.prefillScratch.length < bytes) return 0;

    struct {
        uint32_t n;
        uint32_t n_rows;
        uint32_t stride;
        uint32_t a_offset;
        uint32_t b_offset;
        uint32_t out_offset;
    } args = { n, n_tokens, stride, a_offset, b_offset, out_offset };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_silu_mul_rows_offsets_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:1];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:2];
    NSUInteger threads = g_silu_mul_rows_offsets_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n_tokens * n, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch scratch SwiGLU command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_batch_sigmoid_scale_scratch_add_x0(
    qw3_metal_session *s, uint32_t n_tokens, uint32_t n,
    uint32_t src_offset, uint32_t scalar_offset, uint32_t stride) {
    if (!s || !s->obj || n_tokens == 0 || n == 0 || stride == 0 ||
        src_offset > stride || scalar_offset >= stride ||
        n > stride - src_offset) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_tokens * n * sizeof(float);
    const uint64_t scratch_bytes = (uint64_t)n_tokens * stride * sizeof(float);
    if (!obj.prefillX0 || !obj.prefillScratch ||
        obj.prefillX0.length < x_bytes ||
        obj.prefillScratch.length < scratch_bytes) {
        return 0;
    }

    struct {
        uint32_t n;
        uint32_t n_rows;
        uint32_t stride;
        uint32_t a_offset;
        uint32_t b_offset;
        uint32_t out_offset;
    } args = { n, n_tokens, stride, src_offset, scalar_offset, 0 };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_sigmoid_scale_scratch_add_x0_rows_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.prefillX0 offset:0 atIndex:1];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:2];
    NSUInteger threads =
        g_sigmoid_scale_scratch_add_x0_rows_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n_tokens * n, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch shared expert add command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_read_batch_x0(qw3_metal_session *s, float *out,
                                    uint32_t n_tokens, uint32_t n_out) {
    if (!s || !s->obj || !out || n_tokens == 0 || n_out == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t bytes = (uint64_t)n_tokens * n_out * sizeof(float);
    if (!obj.prefillX0 || obj.prefillX0.length < bytes) return 0;
    id<MTLBuffer> readback = [g_device newBufferWithLength:(NSUInteger)bytes
                                                    options:MTLResourceStorageModeShared];
    if (!readback) return 0;
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:obj.prefillX0 sourceOffset:0
                toBuffer:readback destinationOffset:0
                    size:(NSUInteger)bytes];
    [blit endEncoding];
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch x0 read command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, readback.contents, (size_t)bytes);
    return 1;
}

int qw3_metal_session_read_batch_x1(qw3_metal_session *s, float *out,
                                    uint32_t n_tokens, uint32_t n_out) {
    if (!s || !s->obj || !out || n_tokens == 0 || n_out == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t bytes = (uint64_t)n_tokens * n_out * sizeof(float);
    if (!obj.prefillX1 || obj.prefillX1.length < bytes) return 0;
    id<MTLBuffer> readback = [g_device newBufferWithLength:(NSUInteger)bytes
                                                    options:MTLResourceStorageModeShared];
    if (!readback) return 0;
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:obj.prefillX1 sourceOffset:0
                toBuffer:readback destinationOffset:0
                    size:(NSUInteger)bytes];
    [blit endEncoding];
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch x1 read command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, readback.contents, (size_t)bytes);
    return 1;
}

int qw3_metal_session_read_batch_scratch(qw3_metal_session *s, float *out,
                                         uint32_t n_tokens,
                                         uint32_t n_out) {
    if (!s || !s->obj || !out || n_tokens == 0 || n_out == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t bytes = (uint64_t)n_tokens * n_out * sizeof(float);
    if (!obj.prefillScratch || obj.prefillScratch.length < bytes) return 0;
    id<MTLBuffer> readback = [g_device newBufferWithLength:(NSUInteger)bytes
                                                    options:MTLResourceStorageModeShared];
    if (!readback) return 0;
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:obj.prefillScratch sourceOffset:0
                toBuffer:readback destinationOffset:0
                    size:(NSUInteger)bytes];
    [blit endEncoding];
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch scratch read command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, readback.contents, (size_t)bytes);
    return 1;
}

int qw3_metal_session_read_conv_state(qw3_metal_session *s,
                                      uint32_t layer_slot,
                                      uint32_t n_channels, float *out) {
    if (!s || !s->obj || !out || n_channels == 0 ||
        layer_slot >= QW3_METAL_N_LINEAR_LAYERS) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t bytes = (uint64_t)n_channels * 3ull * sizeof(float);
    const uint64_t offset = (uint64_t)layer_slot * bytes;
    if (!obj.convState || obj.convState.length < offset ||
        obj.convState.length - offset < bytes) {
        return 0;
    }

    id<MTLBuffer> readback = [g_device newBufferWithLength:(NSUInteger)bytes
                                                    options:MTLResourceStorageModeShared];
    if (!readback) return 0;
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:obj.convState sourceOffset:(NSUInteger)offset
                toBuffer:readback destinationOffset:0
                    size:(NSUInteger)bytes];
    [blit endEncoding];
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session conv-state read command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, readback.contents, (size_t)bytes);
    return 1;
}

int qw3_metal_session_read_deltanet_state(qw3_metal_session *s,
                                          uint32_t layer_slot,
                                          uint32_t v_heads,
                                          uint32_t head_dim, float *out) {
    if (!s || !s->obj || !out || v_heads == 0 || head_dim == 0 ||
        layer_slot >= QW3_METAL_N_LINEAR_LAYERS) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t bytes =
        (uint64_t)v_heads * head_dim * head_dim * sizeof(float);
    const uint64_t offset = (uint64_t)layer_slot * bytes;
    if (!obj.deltanetState || obj.deltanetState.length < offset ||
        obj.deltanetState.length - offset < bytes) {
        return 0;
    }

    id<MTLBuffer> readback = [g_device newBufferWithLength:(NSUInteger)bytes
                                                    options:MTLResourceStorageModeShared];
    if (!readback) return 0;
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:obj.deltanetState sourceOffset:(NSUInteger)offset
                toBuffer:readback destinationOffset:0
                    size:(NSUInteger)bytes];
    [blit endEncoding];
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal session DeltaNet-state read command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    memcpy(out, readback.contents, (size_t)bytes);
    return 1;
}

int qw3_metal_session_copy_batch_x0_to_x0(qw3_metal_session *s,
                                          uint32_t row, uint32_t n) {
    if (!s || !s->obj || n == 0) return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    const uint64_t src_offset = (uint64_t)row * row_bytes;
    if (!obj.prefillX0 || !obj.x0 ||
        obj.prefillX0.length < src_offset + row_bytes ||
        obj.x0.length < row_bytes) {
        return 0;
    }

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:obj.prefillX0 sourceOffset:(NSUInteger)src_offset
                toBuffer:obj.x0 destinationOffset:0
                    size:(NSUInteger)row_bytes];
    [blit endEncoding];
    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch x0 copy command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
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

int qw3_metal_session_batch_router_topk_from_scratch(
    qw3_metal_session *s, uint32_t n_tokens, uint32_t router_offset,
    uint32_t stride, uint32_t n_router, uint32_t n_top,
    int *ids_out, float *weights_out) {
    if (!s || !s->obj || n_tokens == 0 || stride == 0 ||
        n_router != 256 || n_top != 8 ||
        router_offset > stride || n_router > stride - router_offset) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t scratch_bytes = (uint64_t)n_tokens * stride * sizeof(float);
    if (!obj.prefillScratch || obj.prefillScratch.length < scratch_bytes ||
        !qw3_metal_session_ensure_router_buffers(obj, n_tokens * n_top)) {
        return 0;
    }

    struct {
        uint32_t n_tokens;
        uint32_t router_offset;
        uint32_t stride;
    } args = { n_tokens, router_offset, stride };

    id<MTLBuffer> ids_readback = nil;
    id<MTLBuffer> weights_readback = nil;
    const uint64_t ids_bytes = (uint64_t)n_tokens * n_top * sizeof(int32_t);
    const uint64_t weights_bytes = (uint64_t)n_tokens * n_top * sizeof(float);
    if (ids_out) {
        ids_readback = [g_device newBufferWithLength:(NSUInteger)ids_bytes
                                             options:MTLResourceStorageModeShared];
        if (!ids_readback) return 0;
    }
    if (weights_out) {
        weights_readback = [g_device newBufferWithLength:(NSUInteger)weights_bytes
                                                 options:MTLResourceStorageModeShared];
        if (!weights_readback) return 0;
    }

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    if (!cb) return 0;
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_router_top8_batch_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:1];
    [enc setBuffer:obj.routerIds offset:0 atIndex:2];
    [enc setBuffer:obj.routerWeights offset:0 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(n_tokens, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    if (ids_readback || weights_readback) {
        qw3_metal_close_batch_encoder();
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        if (!blit) return 0;
        if (ids_readback) {
            [blit copyFromBuffer:obj.routerIds sourceOffset:0
                        toBuffer:ids_readback destinationOffset:0
                            size:(NSUInteger)ids_bytes];
        }
        if (weights_readback) {
            [blit copyFromBuffer:obj.routerWeights sourceOffset:0
                        toBuffer:weights_readback destinationOffset:0
                            size:(NSUInteger)weights_bytes];
        }
        [blit endEncoding];
    }

    int ok = owned ? qw3_metal_finish_command_buffer(cb, owned, "operation")
                   : qw3_metal_end_commands();
    if (!ok) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch router top-k command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    if (ids_out) memcpy(ids_out, ids_readback.contents, (size_t)ids_bytes);
    if (weights_out) memcpy(weights_out, weights_readback.contents,
                            (size_t)weights_bytes);
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

int qw3_metal_session_deltanet_tiled_gdn_from_scratch(
    qw3_metal_session *s, uint64_t dt_bias_offset, uint64_t a_offset,
    uint64_t norm_weight_offset, uint32_t z_offset, uint32_t alpha_offset,
    uint32_t beta_offset, uint32_t layer_slot, uint32_t q_heads,
    uint32_t v_heads, uint32_t head_dim, float eps) {
    if (!s || !s->obj || q_heads == 0 || v_heads == 0 || head_dim == 0 ||
        layer_slot >= QW3_METAL_N_LINEAR_LAYERS || head_dim % 4u != 0u) {
        return 0;
    }
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
    const uint64_t alpha_bytes = (uint64_t)alpha_offset * sizeof(float);
    const uint64_t beta_bytes = (uint64_t)beta_offset * sizeof(float);
    if (!obj.qNorm || !obj.kNorm || !obj.qkvConv || !obj.deltanetState ||
        !obj.scratch || !obj.core || obj.qNorm.length < qk_bytes ||
        obj.kNorm.length < qk_bytes ||
        obj.qkvConv.length < v_offset + inner_bytes ||
        obj.deltanetState.length < state_offset ||
        obj.deltanetState.length - state_offset < state_bytes ||
        obj.scratch.length < alpha_bytes + gates_bytes ||
        obj.scratch.length < beta_bytes + gates_bytes ||
        obj.core.length < inner_bytes) {
        return 0;
    }
    if (!g_model_map_ptr || dt_bias_offset > g_model_map_size ||
        a_offset > g_model_map_size ||
        gates_bytes > g_model_map_size - dt_bias_offset ||
        gates_bytes > g_model_map_size - a_offset) {
        fprintf(stderr, "qw3: Metal tiled Gated DeltaNet weights are outside mapped model\n");
        return 0;
    }

    uint64_t dt_inner = 0;
    uint64_t a_inner = 0;
    id<MTLBuffer> dtb = qw3_metal_model_view_for(dt_bias_offset, gates_bytes,
                                                  &dt_inner);
    id<MTLBuffer> ab = qw3_metal_model_view_for(a_offset, gates_bytes, &a_inner);
    if (!dtb || !ab ||
        128u > g_deltanet_recur_scratch_gates_tiled_pipeline.maxTotalThreadsPerThreadgroup) {
        return 0;
    }

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
    [enc setComputePipelineState:g_deltanet_prepare_scratch_gates_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.scratch offset:0 atIndex:1];
    [enc setBuffer:dtb offset:(NSUInteger)dt_inner atIndex:2];
    [enc setBuffer:ab offset:(NSUInteger)a_inner atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_deltanet_recur_scratch_gates_tiled_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.deltanetState offset:(NSUInteger)state_offset atIndex:1];
    [enc setBuffer:obj.qNorm offset:0 atIndex:2];
    [enc setBuffer:obj.kNorm offset:0 atIndex:3];
    [enc setBuffer:obj.qkvConv offset:(NSUInteger)v_offset atIndex:4];
    [enc setBuffer:obj.scratch offset:0 atIndex:5];
    [enc setBuffer:dtb offset:(NSUInteger)dt_inner atIndex:6];
    [enc setBuffer:ab offset:(NSUInteger)a_inner atIndex:7];
    [enc setBuffer:obj.deltanetState offset:(NSUInteger)state_offset atIndex:8];
    [enc setBuffer:obj.core offset:0 atIndex:9];
    [enc dispatchThreadgroups:MTLSizeMake((head_dim + 3u) / 4u, v_heads, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal tiled Gated DeltaNet command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return qw3_metal_session_deltanet_gated_rmsnorm_from_buffers(
        s, norm_weight_offset, z_offset, v_heads, head_dim, eps, NULL);
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
    if (threads > head_dim) threads = head_dim;
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
    id<MTLBuffer> kgb = qw3_metal_iq3s_expanded_kgrid_buffer();
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

int qw3_metal_session_batch_sparse_moe_topk_from_router_scratch(
    qw3_metal_session *s, uint64_t gate_offset, uint64_t up_offset,
    uint64_t down_offset, uint32_t down_type, uint32_t n_tokens,
    uint32_t n_active, uint32_t n_embd, uint32_t n_ff,
    uint32_t router_offset, uint32_t hidden_offset, uint32_t stride) {
    if (!s || !s->obj || n_tokens == 0 || n_active == 0 || n_active > 8 ||
        n_embd == 0 || n_ff == 0 || (n_embd % 256) != 0 ||
        (n_ff % 256) != 0 || stride == 0 ||
        router_offset > stride || QW3_METAL_N_EXPERT > stride - router_offset ||
        hidden_offset > stride ||
        n_active > (stride - hidden_offset) / n_ff) {
        return 0;
    }
    if (down_type != 23 && down_type != 14) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_tokens * n_embd * sizeof(float);
    const uint64_t scratch_bytes = (uint64_t)n_tokens * stride * sizeof(float);
    const int use_mapped_gateup =
        n_tokens >= 32 && getenv("QW3_METAL_MOE_MAP_GATEUP_DISABLE") == NULL;
    const int use_mapped_gateup_pair =
        use_mapped_gateup && getenv("QW3_METAL_MOE_MAP_GATEUP_PAIR") != NULL &&
        getenv("QW3_METAL_MOE_MAP_GATEUP_PAIR_DISABLE") == NULL;
    const int use_mapped_down =
        down_type == 23 && n_tokens >= 32 &&
        getenv("QW3_METAL_MOE_MAP_DOWN_DISABLE") == NULL;
    const int use_mapped_mid_f16 =
        use_mapped_gateup && use_mapped_down && !use_mapped_gateup_pair &&
        getenv("QW3_METAL_MOE_MID_F16") != NULL &&
        getenv("QW3_METAL_MOE_MID_F16_DISABLE") == NULL;
    const int use_mapped_moe = use_mapped_gateup || use_mapped_down;
    if (!obj.prefillX0 || !obj.prefillX1 || !obj.prefillScratch ||
        obj.prefillX0.length < x_bytes || obj.prefillX1.length < x_bytes ||
        obj.prefillScratch.length < scratch_bytes ||
        !qw3_metal_session_ensure_router_buffers(obj, n_tokens * n_active) ||
        (use_mapped_moe &&
         !qw3_metal_session_ensure_moe_map_buffers(
             obj, n_tokens, n_active, QW3_METAL_N_EXPERT, n_embd, n_ff,
             use_mapped_gateup, use_mapped_down, use_mapped_mid_f16))) {
        return 0;
    }

    if (!qw3_metal_session_batch_router_topk_from_scratch(
            s, n_tokens, router_offset, stride, QW3_METAL_N_EXPERT,
            n_active, NULL, NULL)) {
        return 0;
    }
    if (!qw3_metal_batch_barrier()) return 0;

    const uint64_t iq3_row_bytes = (uint64_t)(n_embd / 256u) * 110ull;
    const uint64_t iq3_expert_bytes = iq3_row_bytes * (uint64_t)n_ff;
    const uint64_t down_row_bytes = down_type == 23 ?
        (uint64_t)(n_ff / 256u) * 136ull :
        (uint64_t)(n_ff / 256u) * 210ull;
    const uint64_t down_expert_bytes = down_row_bytes * (uint64_t)n_embd;
    const uint64_t gate_tensor_bytes = iq3_expert_bytes * 256ull;
    const uint64_t up_tensor_bytes = iq3_expert_bytes * 256ull;
    const uint64_t down_tensor_bytes = down_expert_bytes * 256ull;
    uint64_t gate_inner = 0, up_inner = 0, down_inner = 0;
    id<MTLBuffer> gate_w =
        qw3_metal_model_view_for(gate_offset, gate_tensor_bytes, &gate_inner);
    id<MTLBuffer> up_w =
        qw3_metal_model_view_for(up_offset, up_tensor_bytes, &up_inner);
    id<MTLBuffer> down_w =
        qw3_metal_model_view_for(down_offset, down_tensor_bytes, &down_inner);
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
    id<MTLBuffer> kgb = qw3_metal_iq3s_expanded_kgrid_buffer();
    if (!gate_w || !up_w || !down_w || !kgb) return 0;

    struct {
        uint32_t n_in;
        uint32_t n_ff;
        uint32_t n_embd;
        uint32_t n_tokens;
        uint32_t n_active;
        uint32_t iq3_row_bytes;
        uint32_t iq3_expert_bytes;
        uint32_t down_row_bytes;
        uint32_t down_expert_bytes;
        uint32_t stride;
        uint32_t hidden_offset;
    } args = {
        n_embd, n_ff, n_embd, n_tokens, n_active,
        (uint32_t)iq3_row_bytes, (uint32_t)iq3_expert_bytes,
        (uint32_t)down_row_bytes, (uint32_t)down_expert_bytes,
        stride, hidden_offset
    };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    if (!cb) return 0;
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    if (use_mapped_moe) {
        struct {
            uint32_t n_tokens;
            uint32_t n_active;
            uint32_t n_expert;
            uint32_t pair_capacity;
        } map_args = {
            n_tokens, n_active, QW3_METAL_N_EXPERT, n_tokens
        };
        [enc setComputePipelineState:g_moe_topk_expert_map_pipeline];
        [enc setBytes:&map_args length:sizeof(map_args) atIndex:0];
        [enc setBuffer:obj.routerIds offset:0 atIndex:1];
        [enc setBuffer:obj.moeExpertCounts offset:0 atIndex:2];
        [enc setBuffer:obj.moePairIds offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(QW3_METAL_N_EXPERT, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(QW3_METAL_N_EXPERT, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);
        if (!qw3_metal_batch_barrier()) return 0;
        enc = qw3_metal_compute_encoder(cb);
    }
    if (use_mapped_gateup_pair) {
        [enc setComputePipelineState:g_moe_iq3_s_prefill_pair_mapped_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gate_w offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:up_w offset:(NSUInteger)up_inner atIndex:2];
        [enc setBuffer:obj.prefillX1 offset:0 atIndex:3];
        [enc setBuffer:obj.prefillScratch offset:0 atIndex:4];
        [enc setBuffer:kgb offset:0 atIndex:5];
        [enc setBuffer:obj.moeExpertCounts offset:0 atIndex:6];
        [enc setBuffer:obj.moePairIds offset:0 atIndex:7];
        [enc setThreadgroupMemoryLength:16384u atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((n_tokens + 31u) / 32u,
                                              (n_ff + 63u) / 64u,
                                              QW3_METAL_N_EXPERT)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);
        if (!qw3_metal_batch_barrier()) return 0;
    } else if (use_mapped_gateup) {
        [enc setComputePipelineState:g_moe_iq3_s_prefill_mapped_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gate_w offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:obj.prefillX1 offset:0 atIndex:2];
        [enc setBuffer:obj.prefillMoeGate offset:0 atIndex:3];
        [enc setBuffer:kgb offset:0 atIndex:4];
        [enc setBuffer:obj.moeExpertCounts offset:0 atIndex:5];
        [enc setBuffer:obj.moePairIds offset:0 atIndex:6];
        [enc setThreadgroupMemoryLength:8192u atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((n_tokens + 31u) / 32u,
                                              (n_ff + 63u) / 64u,
                                              QW3_METAL_N_EXPERT)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);

        enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:g_moe_iq3_s_prefill_mapped_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:up_w offset:(NSUInteger)up_inner atIndex:1];
        [enc setBuffer:obj.prefillX1 offset:0 atIndex:2];
        [enc setBuffer:obj.prefillMoeUp offset:0 atIndex:3];
        [enc setBuffer:kgb offset:0 atIndex:4];
        [enc setBuffer:obj.moeExpertCounts offset:0 atIndex:5];
        [enc setBuffer:obj.moePairIds offset:0 atIndex:6];
        [enc setThreadgroupMemoryLength:8192u atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((n_tokens + 31u) / 32u,
                                              (n_ff + 63u) / 64u,
                                              QW3_METAL_N_EXPERT)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);

        if (!qw3_metal_batch_barrier()) return 0;

        enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:use_mapped_mid_f16 ?
         g_moe_swiglu_slots_to_hidden_f16_pipeline :
         g_moe_swiglu_slots_to_hidden_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:obj.prefillMoeGate offset:0 atIndex:1];
        [enc setBuffer:obj.prefillMoeUp offset:0 atIndex:2];
        [enc setBuffer:use_mapped_mid_f16 ? obj.prefillMoeMidF16 : obj.prefillScratch
               offset:0 atIndex:3];
        NSUInteger threads = (use_mapped_mid_f16 ?
                              g_moe_swiglu_slots_to_hidden_f16_pipeline :
                              g_moe_swiglu_slots_to_hidden_pipeline).maxTotalThreadsPerThreadgroup;
        if (threads > 256) threads = 256;
        if (threads < 32) threads = 32;
        [enc dispatchThreads:MTLSizeMake(n_tokens * n_active * n_ff, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);
        if (!qw3_metal_batch_barrier()) return 0;
    } else {
        [enc setComputePipelineState:g_moe_iq3_s_prefill_batch_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gate_w offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:up_w offset:(NSUInteger)up_inner atIndex:2];
        [enc setBuffer:obj.prefillX1 offset:0 atIndex:3];
        [enc setBuffer:obj.prefillScratch offset:0 atIndex:4];
        [enc setBuffer:kgb offset:0 atIndex:5];
        [enc setBuffer:obj.routerIds offset:0 atIndex:6];
        [enc setThreadgroupMemoryLength:64 * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((n_ff + 7u) / 8u) *
                                              n_active * n_tokens, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);
        if (!qw3_metal_batch_barrier()) return 0;
    }

    if (use_mapped_down) {
        enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:use_mapped_mid_f16 ?
         g_moe_down_iq4_xs_prefill_mapped_f16_pipeline :
         g_moe_down_iq4_xs_prefill_mapped_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:down_w offset:(NSUInteger)down_inner atIndex:1];
        [enc setBuffer:use_mapped_mid_f16 ? obj.prefillMoeMidF16 : obj.prefillScratch
               offset:0 atIndex:2];
        [enc setBuffer:obj.prefillMoeDown offset:0 atIndex:3];
        [enc setBuffer:obj.moeExpertCounts offset:0 atIndex:4];
        [enc setBuffer:obj.moePairIds offset:0 atIndex:5];
        [enc setBuffer:obj.routerWeights offset:0 atIndex:6];
        [enc setThreadgroupMemoryLength:8192u atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((n_tokens + 31u) / 32u,
                                              (n_embd + 63u) / 64u,
                                              QW3_METAL_N_EXPERT)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);

        if (!qw3_metal_batch_barrier()) return 0;

        struct {
            uint32_t n_tokens;
            uint32_t n_active;
            uint32_t n_embd;
        } reduce_args = { n_tokens, n_active, n_embd };
        enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:g_moe_down_prefill_reduce_slots_pipeline];
        [enc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
        [enc setBuffer:obj.prefillMoeDown offset:0 atIndex:1];
        [enc setBuffer:obj.routerWeights offset:0 atIndex:2];
        [enc setBuffer:obj.prefillX0 offset:0 atIndex:3];
        NSUInteger threads = g_moe_down_prefill_reduce_slots_pipeline.maxTotalThreadsPerThreadgroup;
        if (threads > 256) threads = 256;
        if (threads < 32) threads = 32;
        [enc dispatchThreads:MTLSizeMake(n_tokens * n_embd, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);
        if (!qw3_metal_batch_barrier()) return 0;
    } else {
        enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:down_type == 23 ?
         g_moe_down_iq4_xs_prefill_reduce_pipeline :
         g_moe_down_q6_k_prefill_reduce_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:down_w offset:(NSUInteger)down_inner atIndex:1];
        [enc setBuffer:obj.prefillScratch offset:0 atIndex:2];
        [enc setBuffer:obj.prefillX0 offset:0 atIndex:3];
        [enc setBuffer:obj.routerIds offset:0 atIndex:4];
        [enc setBuffer:obj.routerWeights offset:0 atIndex:5];
        [enc setThreadgroupMemoryLength:16 * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((n_embd + 1u) / 2u) * n_tokens,
                                              1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);
        if (!qw3_metal_batch_barrier()) return 0;
    }

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch sparse MoE prefill command failed: %s\n",
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

int qw3_metal_session_batch_gqa_norm_rope_from_scratch(
    qw3_metal_session *s, uint64_t q_norm_weight_offset,
    uint64_t k_norm_weight_offset, uint32_t n_tokens, uint32_t n_heads,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t rope_dim,
    uint32_t pos0, float rope_theta, float eps, uint32_t qg_offset,
    uint32_t k_offset, uint32_t q_tmp_offset, uint32_t k_tmp_offset,
    uint32_t q_rope_offset, uint32_t k_rope_offset, uint32_t gate_offset,
    uint32_t stride) {
    if (!s || !s->obj || n_tokens == 0 || n_heads == 0 ||
        n_kv_heads == 0 || head_dim == 0 || rope_dim == 0 ||
        rope_dim > head_dim || (rope_dim % 2u) != 0u || stride == 0) {
        return 0;
    }
    const uint32_t q_n = n_heads * head_dim;
    const uint32_t qg_n = 2u * q_n;
    const uint32_t kv_n = n_kv_heads * head_dim;
    if (qg_offset > stride || k_offset > stride ||
        q_tmp_offset > stride || k_tmp_offset > stride ||
        q_rope_offset > stride || k_rope_offset > stride ||
        gate_offset > stride ||
        qg_n > stride - qg_offset ||
        kv_n > stride - k_offset ||
        q_n > stride - q_tmp_offset ||
        kv_n > stride - k_tmp_offset ||
        q_n > stride - q_rope_offset ||
        kv_n > stride - k_rope_offset ||
        q_n > stride - gate_offset) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t scratch_bytes = (uint64_t)n_tokens * stride * sizeof(float);
    if (!obj.prefillScratch || obj.prefillScratch.length < scratch_bytes) {
        return 0;
    }
    const uint64_t norm_weight_bytes = (uint64_t)head_dim * sizeof(float);
    uint64_t q_inner = 0, k_inner = 0;
    id<MTLBuffer> qw = qw3_metal_model_view_for(q_norm_weight_offset,
                                                norm_weight_bytes,
                                                &q_inner);
    id<MTLBuffer> kw = qw3_metal_model_view_for(k_norm_weight_offset,
                                                norm_weight_bytes,
                                                &k_inner);
    if (!qw || !kw) {
        fprintf(stderr, "qw3: Metal batch GQA norm weights are outside mapped model views\n");
        return 0;
    }

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    if (!cb) return 0;

    struct {
        uint32_t n_tokens;
        uint32_t n_heads;
        uint32_t head_dim;
        uint32_t in_offset;
        uint32_t out_offset;
        uint32_t gate_offset;
        uint32_t stride;
        float eps;
    } q_args = {
        n_tokens, n_heads, head_dim, qg_offset, q_tmp_offset,
        gate_offset, stride, eps
    };
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    if (!enc) return 0;
    [enc setComputePipelineState:g_gqa_q_norm_gate_batch_pipeline];
    [enc setBytes:&q_args length:sizeof(q_args) atIndex:0];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:1];
    [enc setBuffer:qw offset:(NSUInteger)q_inner atIndex:2];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    NSUInteger threads =
        g_gqa_q_norm_gate_batch_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens * n_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    struct {
        uint32_t n_tokens;
        uint32_t n_heads;
        uint32_t head_dim;
        uint32_t in_offset;
        uint32_t out_offset;
        uint32_t gate_offset;
        uint32_t stride;
        float eps;
    } k_args = {
        n_tokens, n_kv_heads, head_dim, k_offset, k_tmp_offset,
        0, stride, eps
    };
    enc = qw3_metal_compute_encoder(cb);
    if (!enc) return 0;
    [enc setComputePipelineState:g_gqa_k_norm_batch_pipeline];
    [enc setBytes:&k_args length:sizeof(k_args) atIndex:0];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:1];
    [enc setBuffer:kw offset:(NSUInteger)k_inner atIndex:2];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    threads = g_gqa_k_norm_batch_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens * n_kv_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    if (!qw3_metal_batch_barrier()) return 0;

    struct {
        uint32_t n_tokens;
        uint32_t n_heads;
        uint32_t head_dim;
        uint32_t rope_dim;
        uint32_t pos0;
        uint32_t in_offset;
        uint32_t out_offset;
        uint32_t stride;
        float theta;
    } rq_args = {
        n_tokens, n_heads, head_dim, rope_dim, pos0, q_tmp_offset,
        q_rope_offset, stride, rope_theta
    };
    enc = qw3_metal_compute_encoder(cb);
    if (!enc) return 0;
    [enc setComputePipelineState:g_rope_heads_batch_pipeline];
    [enc setBytes:&rq_args length:sizeof(rq_args) atIndex:0];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:1];
    threads = g_rope_heads_batch_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n_tokens * q_n, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    struct {
        uint32_t n_tokens;
        uint32_t n_heads;
        uint32_t head_dim;
        uint32_t rope_dim;
        uint32_t pos0;
        uint32_t in_offset;
        uint32_t out_offset;
        uint32_t stride;
        float theta;
    } rk_args = {
        n_tokens, n_kv_heads, head_dim, rope_dim, pos0, k_tmp_offset,
        k_rope_offset, stride, rope_theta
    };
    enc = qw3_metal_compute_encoder(cb);
    if (!enc) return 0;
    [enc setComputePipelineState:g_rope_heads_batch_pipeline];
    [enc setBytes:&rk_args length:sizeof(rk_args) atIndex:0];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:1];
    threads = g_rope_heads_batch_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n_tokens * kv_n, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch GQA norm/RoPE command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_batch_gqa_causal_attn_from_scratch(
    qw3_metal_session *s, uint32_t n_tokens, uint32_t n_heads,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t q_offset,
    uint32_t gate_offset, uint32_t k_offset, uint32_t v_offset,
    uint32_t out_offset, uint32_t stride) {
    if (!s || !s->obj || n_tokens == 0 || n_heads == 0 ||
        n_kv_heads == 0 || head_dim == 0 || stride == 0 ||
        (n_heads % n_kv_heads) != 0u) {
        return 0;
    }
    const uint32_t q_n = n_heads * head_dim;
    const uint32_t kv_n = n_kv_heads * head_dim;
    if (q_offset > stride || gate_offset > stride ||
        k_offset > stride || v_offset > stride || out_offset > stride ||
        q_n > stride - q_offset ||
        q_n > stride - gate_offset ||
        kv_n > stride - k_offset ||
        kv_n > stride - v_offset ||
        q_n > stride - out_offset) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t scratch_bytes = (uint64_t)n_tokens * stride * sizeof(float);
    if (!obj.prefillScratch || obj.prefillScratch.length < scratch_bytes) {
        return 0;
    }
    NSUInteger threads = ((NSUInteger)head_dim + 31u) & ~(NSUInteger)31u;
    if (threads < 32u) threads = 32u;
    if (threads > 256u ||
        threads > g_gqa_prefill_attend_inner_pipeline.maxTotalThreadsPerThreadgroup) {
        return 0;
    }

    struct {
        uint32_t n_tokens;
        uint32_t n_heads;
        uint32_t n_kv_heads;
        uint32_t head_dim;
        uint32_t q_offset;
        uint32_t gate_offset;
        uint32_t k_offset;
        uint32_t v_offset;
        uint32_t out_offset;
        uint32_t stride;
    } args = {
        n_tokens, n_heads, n_kv_heads, head_dim, q_offset, gate_offset,
        k_offset, v_offset, out_offset, stride
    };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    if (!enc) return 0;
    [enc setComputePipelineState:g_gqa_prefill_attend_inner_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:1];
    [enc setThreadgroupMemoryLength:64u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens * n_kv_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch GQA causal attention command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_batch_gqa_write_cache_from_scratch(
    qw3_metal_session *s, uint32_t layer_slot, uint32_t pos0,
    uint32_t n_tokens, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t k_offset, uint32_t v_offset, uint32_t stride) {
    if (!s || !s->obj || n_tokens == 0 || n_kv_heads == 0 ||
        head_dim == 0 || stride == 0) {
        return 0;
    }
    const uint32_t kv_n = n_kv_heads * head_dim;
    if (k_offset > stride || v_offset > stride ||
        kv_n > stride - k_offset || kv_n > stride - v_offset) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    if (obj.gqaKvQ8 || pos0 > obj.ctxSize ||
        n_tokens > obj.ctxSize - pos0) {
        return 0;
    }
    const uint64_t scratch_bytes = (uint64_t)n_tokens * stride * sizeof(float);
    const uint64_t cache_token_bytes = (uint64_t)kv_n * sizeof(float);
    const uint64_t cache_layer_bytes = (uint64_t)obj.ctxSize * cache_token_bytes;
    const uint64_t cache_offset = (uint64_t)layer_slot * cache_layer_bytes;
    if (!obj.prefillScratch || obj.prefillScratch.length < scratch_bytes ||
        !obj.gqaK || !obj.gqaV ||
        obj.gqaK.length < cache_offset + cache_layer_bytes ||
        obj.gqaV.length < cache_offset + cache_layer_bytes) {
        return 0;
    }

    struct {
        uint32_t n_tokens;
        uint32_t n_kv_heads;
        uint32_t head_dim;
        uint32_t pos0;
        uint32_t ctx_size;
        uint32_t k_offset;
        uint32_t v_offset;
        uint32_t stride;
    } args = {
        n_tokens, n_kv_heads, head_dim, pos0, obj.ctxSize,
        k_offset, v_offset, stride
    };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    if (!enc) return 0;
    [enc setComputePipelineState:g_gqa_prefill_write_cache_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:1];
    [enc setBuffer:obj.gqaK offset:(NSUInteger)cache_offset atIndex:2];
    [enc setBuffer:obj.gqaV offset:(NSUInteger)cache_offset atIndex:3];
    NSUInteger threads = g_gqa_prefill_write_cache_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n_tokens * kv_n, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch GQA cache write command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

int qw3_metal_session_batch_gqa_cached_attn_from_scratch(
    qw3_metal_session *s, uint32_t layer_slot, uint32_t pos0,
    uint32_t n_tokens, uint32_t n_heads, uint32_t n_kv_heads,
    uint32_t head_dim, uint32_t q_offset, uint32_t gate_offset,
    uint32_t out_offset, uint32_t stride) {
    if (!s || !s->obj || n_tokens == 0 || n_heads == 0 ||
        n_kv_heads == 0 || head_dim == 0 || stride == 0 ||
        (n_heads % n_kv_heads) != 0u) {
        return 0;
    }
    const uint32_t q_n = n_heads * head_dim;
    const uint32_t kv_n = n_kv_heads * head_dim;
    if (q_offset > stride || gate_offset > stride || out_offset > stride ||
        q_n > stride - q_offset ||
        q_n > stride - gate_offset ||
        q_n > stride - out_offset) {
        return 0;
    }
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    if (obj.gqaKvQ8 || pos0 > obj.ctxSize ||
        n_tokens > obj.ctxSize - pos0) {
        return 0;
    }
    const uint64_t scratch_bytes = (uint64_t)n_tokens * stride * sizeof(float);
    const uint64_t cache_token_bytes = (uint64_t)kv_n * sizeof(float);
    const uint64_t cache_layer_bytes = (uint64_t)obj.ctxSize * cache_token_bytes;
    const uint64_t cache_offset = (uint64_t)layer_slot * cache_layer_bytes;
    if (!obj.prefillScratch || obj.prefillScratch.length < scratch_bytes ||
        !obj.gqaK || !obj.gqaV ||
        obj.gqaK.length < cache_offset + cache_layer_bytes ||
        obj.gqaV.length < cache_offset + cache_layer_bytes) {
        return 0;
    }
    NSUInteger threads = ((NSUInteger)head_dim + 31u) & ~(NSUInteger)31u;
    if (threads < 32u) threads = 32u;
    if (threads > 256u ||
        threads > g_gqa_prefill_cached_attend_inner_pipeline.maxTotalThreadsPerThreadgroup) {
        return 0;
    }

    struct {
        uint32_t n_tokens;
        uint32_t n_heads;
        uint32_t n_kv_heads;
        uint32_t head_dim;
        uint32_t pos0;
        uint32_t ctx_size;
        uint32_t q_offset;
        uint32_t gate_offset;
        uint32_t out_offset;
        uint32_t stride;
    } args = {
        n_tokens, n_heads, n_kv_heads, head_dim, pos0, obj.ctxSize,
        q_offset, gate_offset, out_offset, stride
    };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    if (!enc) return 0;
    [enc setComputePipelineState:g_gqa_prefill_cached_attend_inner_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:1];
    [enc setBuffer:obj.gqaK offset:(NSUInteger)cache_offset atIndex:2];
    [enc setBuffer:obj.gqaV offset:(NSUInteger)cache_offset atIndex:3];
    [enc setThreadgroupMemoryLength:64u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens * n_kv_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch GQA cached attention command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
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
    NSUInteger threads = ((NSUInteger)head_dim + 31u) & ~(NSUInteger)31u;
    if (threads < 32u) threads = 32u;
    if (threads > 256u) {
        return 0;
    }
    const BOOL split_q8 = obj.gqaKvQ8 && obj.gqaSplitQ8 &&
        (n_ctx >= 256u || getenv("QW3_METAL_Q8_SPLIT_FORCE") != NULL);
    if (split_q8) {
        uint32_t active_max_splits = obj.gqaMaxQ8Splits;
        if (active_max_splits > 32u && n_ctx < 2048u) {
            active_max_splits = 32u;
        } else if (active_max_splits > 64u && n_ctx < 4096u) {
            active_max_splits = 64u;
        } else if (active_max_splits > 128u && n_ctx < 16384u) {
            active_max_splits = 128u;
        }
        const uint32_t n_splits =
            n_ctx < active_max_splits ? n_ctx : active_max_splits;
        const uint64_t split_bytes =
            (uint64_t)n_splits * n_heads * (head_dim + 2u) * sizeof(float);
        if (!obj.gqaAttnPartial || obj.gqaAttnPartial.length < split_bytes ||
            threads > g_gqa_attend_n_q8_split_partial_pipeline.maxTotalThreadsPerThreadgroup ||
            threads > g_gqa_attend_n_q8_split_reduce_pipeline.maxTotalThreadsPerThreadgroup) {
            return 0;
        }
        struct {
            uint32_t n_ctx;
            uint32_t n_heads;
            uint32_t n_kv_heads;
            uint32_t head_dim;
            uint32_t n_splits;
        } split_args = { n_ctx, n_heads, n_kv_heads, head_dim, n_splits };
        id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
        if (!enc) return 0;
        [enc setComputePipelineState:g_gqa_attend_n_q8_split_partial_pipeline];
        [enc setBytes:&split_args length:sizeof(split_args) atIndex:0];
        [enc setBuffer:obj.gqaTokenQ offset:0 atIndex:1];
        [enc setBuffer:obj.gqaK offset:(NSUInteger)cache_offset atIndex:2];
        [enc setBuffer:obj.gqaV offset:(NSUInteger)cache_offset atIndex:3];
        [enc setBuffer:obj.gqaAttnPartial offset:0 atIndex:4];
        [enc setThreadgroupMemoryLength:64u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_kv_heads * n_splits, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);

        enc = qw3_metal_compute_encoder(cb);
        if (!enc) return 0;
        [enc setComputePipelineState:g_gqa_attend_n_q8_split_reduce_pipeline];
        [enc setBytes:&split_args length:sizeof(split_args) atIndex:0];
        [enc setBuffer:obj.gqaTokenGate offset:0 atIndex:1];
        [enc setBuffer:obj.gqaAttnPartial offset:0 atIndex:2];
        [enc setBuffer:obj.inner offset:0 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);
    } else {
        id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
        if (!enc) return 0;
        id<MTLComputePipelineState> pipeline = obj.gqaKvQ8 ?
            g_gqa_attend_n_q8_inner_pipeline : g_gqa_attend_n_inner_pipeline;
        if (threads > pipeline.maxTotalThreadsPerThreadgroup) {
            qw3_metal_end_compute_encoder(cb, enc);
            return 0;
        }
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:obj.gqaTokenQ offset:0 atIndex:1];
        [enc setBuffer:obj.gqaTokenGate offset:0 atIndex:2];
        [enc setBuffer:obj.gqaK offset:(NSUInteger)cache_offset atIndex:3];
        [enc setBuffer:obj.gqaV offset:(NSUInteger)cache_offset atIndex:4];
        [enc setBuffer:obj.inner offset:0 atIndex:5];
        [enc setThreadgroupMemoryLength:64u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_kv_heads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);
    }
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
    g_iq3s_expanded_kgrid_buffer = nil;
    g_rmsnorm_plain_pipeline = nil;
    g_rmsnorm_weight_f32_pipeline = nil;
    g_rmsnorm_weight_f32_rows_pipeline = nil;
    g_rmsnorm_weight_f32_rows_to_out_pipeline = nil;
    g_embed_q8_0_pipeline = nil;
    g_embed_q8_0_batch_pipeline = nil;
    g_matvec_q8_0_pipeline = nil;
    g_matmul_q8_0_batch4_pipeline = nil;
    g_matmul_q8_0_mm_pipeline = nil;
    g_matmul_q8_0_mm_bc_pipeline = nil;
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
    g_moe_iq3_s_prefill_batch_pipeline = nil;
    g_moe_down_iq4_xs_prefill_reduce_pipeline = nil;
    g_moe_down_q6_k_prefill_reduce_pipeline = nil;
    g_moe_topk_expert_map_pipeline = nil;
    g_moe_iq3_s_prefill_mapped_pipeline = nil;
    g_moe_iq3_s_prefill_pair_mapped_pipeline = nil;
    g_moe_swiglu_slots_to_hidden_pipeline = nil;
    g_moe_swiglu_slots_to_hidden_f16_pipeline = nil;
    g_moe_down_iq4_xs_prefill_mapped_pipeline = nil;
    g_moe_down_iq4_xs_prefill_mapped_f16_pipeline = nil;
    g_moe_down_prefill_reduce_slots_pipeline = nil;
    g_matvec_f32_pipeline = nil;
    g_matvec_f32_pair_pipeline = nil;
    g_matvec_f32_fast_pipeline = nil;
    g_matmul_f32_batch4_pipeline = nil;
    g_matmul_f32_pair_batch4_pipeline = nil;
    g_deltanet_conv1d_zero_pipeline = nil;
    g_deltanet_conv1d_step_pipeline = nil;
    g_deltanet_conv1d_batch_pipeline = nil;
    g_deltanet_conv1d_batch_state_pipeline = nil;
    g_l2norm_heads_pipeline = nil;
    g_l2norm_qk_batch_pipeline = nil;
    g_gqa_q_norm_gate_pipeline = nil;
    g_gqa_k_norm_pipeline = nil;
    g_gqa_q_norm_gate_batch_pipeline = nil;
    g_gqa_k_norm_batch_pipeline = nil;
    g_rope_heads_pipeline = nil;
    g_rope_heads_batch_pipeline = nil;
    g_gqa_single_token_inner_pipeline = nil;
    g_gqa_attend2_inner_pipeline = nil;
    g_gqa_attend_n_inner_pipeline = nil;
    g_gqa_prefill_attend_inner_pipeline = nil;
    g_gqa_prefill_write_cache_pipeline = nil;
    g_gqa_prefill_cached_attend_inner_pipeline = nil;
    g_gqa_kv_quant_q8_pipeline = nil;
    g_gqa_attend_n_q8_inner_pipeline = nil;
    g_gqa_attend_n_q8_split_partial_pipeline = nil;
    g_gqa_attend_n_q8_split_reduce_pipeline = nil;
    g_deltanet_recur_zero_pipeline = nil;
    g_deltanet_recur_pipeline = nil;
    g_deltanet_recur_scratch_gates_pipeline = nil;
    g_deltanet_prepare_scratch_gates_pipeline = nil;
    g_deltanet_recur_scratch_gates_tiled_pipeline = nil;
    g_deltanet_fused_gdn_scratch_pipeline = nil;
    g_deltanet_batch_fused_gdn_pipeline = nil;
    g_deltanet_gated_rmsnorm_pipeline = nil;
    g_residual_rmsnorm_weight_f32_pipeline = nil;
    g_residual_rmsnorm_update_x0_pipeline = nil;
    g_residual_rmsnorm_batch_update_x0_pipeline = nil;
    g_silu_mul_pipeline = nil;
    g_scale_pipeline = nil;
    g_add_moe_to_x0_pipeline = nil;
    g_silu_mul_offsets_pipeline = nil;
    g_silu_mul_rows_offsets_pipeline = nil;
    g_scale_x1_scalar_add_x0_pipeline = nil;
    g_scale_x1_add_x0_pipeline = nil;
    g_scale_scratch_add_x0_pipeline = nil;
    g_sigmoid_scale_scratch_add_x0_rows_pipeline = nil;
    g_router_top8_pipeline = nil;
    g_router_top8_batch_pipeline = nil;
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
