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
static id<MTLComputePipelineState> g_matmul_q8_0_nax_pipeline;
static id<MTLComputePipelineState> g_matmul_q8_0_nax_n64_pipeline;
static id<MTLComputePipelineState> g_matmul_q8_0_nax_n128_pipeline;
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
static id<MTLComputePipelineState> g_moe_iq3_s_prefill_mapped_mpp_pipeline;
static id<MTLComputePipelineState> g_moe_iq3_s_prefill_pair_mapped_pipeline;
static id<MTLComputePipelineState> g_moe_iq3_s_prefill_pair_mapped_mpp_pipeline;
static id<MTLComputePipelineState> g_moe_swiglu_slots_to_hidden_pipeline;
static id<MTLComputePipelineState> g_moe_swiglu_slots_to_mid_f32_pipeline;
static id<MTLComputePipelineState> g_moe_swiglu_slots_to_hidden_f16_pipeline;
static id<MTLComputePipelineState> g_moe_down_iq4_xs_prefill_mapped_pipeline;
static id<MTLComputePipelineState> g_moe_down_iq4_xs_prefill_mapped_mid_f32_pipeline;
static id<MTLComputePipelineState> g_moe_down_iq4_xs_prefill_mapped_mid_f32_mpp_pipeline;
static id<MTLComputePipelineState> g_moe_down_iq4_xs_prefill_mapped_f16_pipeline;
static id<MTLComputePipelineState> g_moe_down_iq4_xs_prefill_mapped_f16_mpp_pipeline;
static id<MTLComputePipelineState> g_moe_down_q6_k_prefill_mapped_pipeline;
static id<MTLComputePipelineState> g_moe_down_q6_k_prefill_mapped_mpp_pipeline;
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
static id<MTLComputePipelineState> g_gqa_q_norm_gate_rope_batch_pipeline;
static id<MTLComputePipelineState> g_gqa_k_norm_rope_batch_pipeline;
static id<MTLComputePipelineState> g_rope_heads_pipeline;
static id<MTLComputePipelineState> g_rope_heads_batch_pipeline;
static id<MTLComputePipelineState> g_gqa_single_token_inner_pipeline;
static id<MTLComputePipelineState> g_gqa_attend2_inner_pipeline;
static id<MTLComputePipelineState> g_gqa_attend_n_inner_pipeline;
static id<MTLComputePipelineState> g_gqa_prefill_attend_inner_pipeline;
static id<MTLComputePipelineState> g_gqa_prefill_write_cache_pipeline;
static id<MTLComputePipelineState> g_gqa_prefill_cached_attend_inner_pipeline;
static id<MTLComputePipelineState> g_gqa_prefill_cached_attend_block2_pipeline;
static id<MTLComputePipelineState> g_gqa_prefill_cached_attend_block4_pipeline;
static id<MTLComputePipelineState> g_gqa_prefill_cached_attend_src8_pipeline;
static id<MTLComputePipelineState> g_gqa_flash_gate_pipeline;
static id<MTLComputePipelineState> g_gqa_flash_causal_mask_pipeline;
static id<MTLComputePipelineState> g_gqa_flash_pad_pipeline;
static id<MTLComputePipelineState> g_gqa_flash_blk_pipeline;
static id<MTLComputePipelineState> g_gqa_flash_attn_pipeline;
static int g_gqa_flash_attn_external_enabled;
static id<MTLComputePipelineState> g_gqa_store_token_cache_f16_pipeline;
static id<MTLComputePipelineState> g_gqa_kv_quant_q8_pipeline;
static id<MTLComputePipelineState> g_gqa_attend_n_q8_inner_pipeline;
static id<MTLComputePipelineState> g_gqa_attend_n_split_partial_pipeline;
static id<MTLComputePipelineState> g_gqa_attend_n_split_reduce_pipeline;
static id<MTLComputePipelineState> g_gqa_attend_n_q8_split_partial_pipeline;
static id<MTLComputePipelineState> g_gqa_attend_n_q8_split_reduce_pipeline;
static id<MTLComputePipelineState> g_deltanet_recur_zero_pipeline;
static id<MTLComputePipelineState> g_deltanet_recur_pipeline;
static id<MTLComputePipelineState> g_deltanet_recur_scratch_gates_pipeline;
static id<MTLComputePipelineState> g_deltanet_prepare_scratch_gates_pipeline;
static id<MTLComputePipelineState> g_deltanet_recur_scratch_gates_tiled_pipeline;
static id<MTLComputePipelineState> g_deltanet_fused_gdn_scratch_pipeline;
static id<MTLComputePipelineState> g_deltanet_batch_fused_gdn_pipeline;
static id<MTLComputePipelineState> g_deltanet_batch_recur_tiled_pipeline;
static id<MTLComputePipelineState> g_deltanet_batch_recur_tiled2_pipeline;
static id<MTLComputePipelineState> g_deltanet_batch_recur_tiled4_pipeline;
static id<MTLComputePipelineState> g_deltanet_batch_gated_rmsnorm_pipeline;
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
static int g_metal4_runtime_available;
static int g_metal4_family_supported;
static int g_metal4_queue_supported;
static int g_metal4_m5_neural_accelerators_hint;
static int g_metal4_tensor_api_enabled;
static int g_metal4_tensor_api_compile_supported;

static int qw3_metal_env_bool(const char *name) {
    const char *v = getenv(name);
    if (!v || !v[0]) return 0;
    if (strcmp(v, "0") == 0 ||
        strcmp(v, "false") == 0 ||
        strcmp(v, "FALSE") == 0 ||
        strcmp(v, "no") == 0 ||
        strcmp(v, "NO") == 0) return 0;
    return 1;
}

static int qw3_metal_layer_is_full_attention(uint32_t il) {
    return ((il + 1u) % 4u) == 0u;
}

static uint32_t qw3_metal_env_n_gpu_layers(void) {
    const char *env = getenv("QW3_METAL_NGL");
    if (!env || !env[0]) return 40u;
    char *end = NULL;
    long v = strtol(env, &end, 10);
    if (end == env) return 40u;
    if (v < 0) return 40u;
    if (v > 40) return 40u;
    return (uint32_t)v;
}

static int qw3_metal_llama_split_enabled(void) {
    const char *env = getenv("QW3_METAL_LLAMACPP_SPLIT");
    if (env && env[0]) return strcmp(env, "0") != 0;
    return 1;
}

static uint32_t qw3_metal_llama_split_start(uint32_t n_gpu_layers) {
    if (n_gpu_layers == 0) return 40u;
    if (n_gpu_layers >= 41u) return 0;
    return 41u - n_gpu_layers;
}

static void qw3_metal_count_layer_types_before(uint32_t n_layers,
                                               uint32_t *n_full,
                                               uint32_t *n_linear) {
    if (n_layers > 40u) n_layers = 40u;
    uint32_t full = 0;
    uint32_t linear = 0;
    for (uint32_t il = 0; il < n_layers; il++) {
        if (qw3_metal_layer_is_full_attention(il)) full++;
        else linear++;
    }
    if (n_full) *n_full = full;
    if (n_linear) *n_linear = linear;
}

static void qw3_metal_count_layer_types_from(uint32_t first_layer,
                                             uint32_t *n_full,
                                             uint32_t *n_linear) {
    if (first_layer > 40u) first_layer = 40u;
    uint32_t full = 0;
    uint32_t linear = 0;
    for (uint32_t il = first_layer; il < 40u; il++) {
        if (qw3_metal_layer_is_full_attention(il)) full++;
        else linear++;
    }
    if (n_full) *n_full = full;
    if (n_linear) *n_linear = linear;
}

static uint64_t qw3_metal_alloc_size(uint64_t bytes) {
    return bytes ? bytes : 1ull;
}

static int qw3_metal_device_name_contains(const char *needle) {
    return g_device_name[0] != '\0' && strstr(g_device_name, needle) != NULL;
}

static int qw3_metal_compile_tensor_probe(void) {
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (!g_device) return 0;
    if (@available(macOS 26.0, *)) {
        const char *src =
            "#include <metal_stdlib>\n"
            "#include <metal_tensor>\n"
            "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
            "using namespace metal;\n"
            "using namespace mpp::tensor_ops;\n"
            "kernel void qw3_tensor_probe(\n"
            "        tensor<device half,  dextents<int32_t, 2>> A [[buffer(0)]],\n"
            "        tensor<device half,  dextents<int32_t, 2>> B [[buffer(1)]],\n"
            "        device float *C [[buffer(2)]],\n"
            "        uint2 tgid [[threadgroup_position_in_grid]]) {\n"
            "    auto tA = A.slice(0, (int)tgid.y);\n"
            "    auto tB = B.slice((int)tgid.x, 0);\n"
            "    matmul2d<matmul2d_descriptor(16, 16, dynamic_extent), execution_simdgroups<4>> mm;\n"
            "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();\n"
            "    auto sA = tA.slice(0, 0);\n"
            "    auto sB = tB.slice(0, 0);\n"
            "    mm.run(sB, sA, cT);\n"
            "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(16, 16));\n"
            "    cT.store(tC);\n"
            "}\n";

        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:src];
        id<MTLLibrary> probe_library =
            [g_device newLibraryWithSource:source options:[MTLCompileOptions new] error:&error];
        if (!probe_library) {
            fprintf(stderr, "qw3: Metal 4 tensor API probe compile failed: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "(unknown)");
            return 0;
        }
        id<MTLFunction> fn = [probe_library newFunctionWithName:@"qw3_tensor_probe"];
        if (!fn) {
            fprintf(stderr, "qw3: Metal 4 tensor API probe function missing\n");
            return 0;
        }
        error = nil;
        id<MTLComputePipelineState> pipeline =
            [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!pipeline) {
            fprintf(stderr, "qw3: Metal 4 tensor API probe pipeline failed: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "(unknown)");
            return 0;
        }
        return 1;
    }
#endif
    return 0;
}

static void qw3_metal_detect_metal4_features(void) {
    g_metal4_runtime_available = 0;
    g_metal4_family_supported = 0;
    g_metal4_queue_supported = 0;
    g_metal4_m5_neural_accelerators_hint = 0;
    g_metal4_tensor_api_enabled = 0;
    g_metal4_tensor_api_compile_supported = 0;

    if (!g_device) return;

    const int metal4_disabled = qw3_metal_env_bool("QW3_METAL_DISABLE_METAL4");

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (@available(macOS 26.0, *)) {
        g_metal4_runtime_available = 1;
        g_metal4_family_supported =
            !metal4_disabled && [g_device supportsFamily:MTLGPUFamilyMetal4] ? 1 : 0;
        g_metal4_queue_supported =
            [g_device respondsToSelector:@selector(newMTL4CommandQueue)] ? 1 : 0;

        if (g_metal4_family_supported && qw3_metal_device_name_contains("M5")) {
            g_metal4_m5_neural_accelerators_hint = 1;
        }

        if (g_metal4_family_supported) {
            const int default_enable =
                qw3_metal_device_name_contains("M5") ||
                qw3_metal_device_name_contains("M6") ||
                qw3_metal_device_name_contains("A19") ||
                qw3_metal_device_name_contains("A20");
            if (default_enable) {
                g_metal4_tensor_api_compile_supported = qw3_metal_compile_tensor_probe();
                g_metal4_tensor_api_enabled = g_metal4_tensor_api_compile_supported;
                if (!g_metal4_tensor_api_enabled) {
                    fprintf(stderr, "qw3: Metal 4 tensor API probe failed; using legacy Metal kernels\n");
                }
            } else {
                fprintf(stderr, "qw3: Metal 4 tensor API disabled for pre-M5/pre-A19 devices\n");
            }
        }
    }
#else
    (void)metal4_disabled;
#endif
}

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
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *gqaKLayers;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *gqaVLayers;
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
@property(nonatomic, strong) id<MTLBuffer> flashAttnOut;
@property(nonatomic, strong) id<MTLBuffer> flashAttnMask;
@property(nonatomic, strong) id<MTLBuffer> flashAttnBlock;
@property(nonatomic, strong) id<MTLBuffer> flashAttnPad;
@property(nonatomic, strong) id<MTLBuffer> moeExpertCounts;
@property(nonatomic, strong) id<MTLBuffer> moePairIds;
@property(nonatomic, strong) id<MTLBuffer> moeBlockCount;
@property(nonatomic, strong) id<MTLBuffer> moeBlockIds;
@property(nonatomic, strong) id<MTLBuffer> moeBlockDispatchFF;
@property(nonatomic, strong) id<MTLBuffer> moeBlockDispatchEmbd;
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
@property(nonatomic) BOOL gqaKvF16;
@property(nonatomic) BOOL gqaSplitQ8;
@property(nonatomic) BOOL gqaSplitAttn;
@property(nonatomic) uint32_t gqaMaxQ8Splits;
@property(nonatomic) uint32_t gqaMaxAttnSplits;
@property(nonatomic) uint32_t flashAttnMaskPos0;
@property(nonatomic) uint32_t flashAttnMaskTokens;
@property(nonatomic) uint32_t flashAttnMaskKeys;
@property(nonatomic) uint32_t metalFullLayers;
@property(nonatomic) uint32_t metalLinearLayers;
@end

@implementation QW3MetalSessionObj
@end

struct qw3_metal_session {
    void *obj;
};

enum {
    QW3_METAL_N_LAYER = 40,
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

static NSString *qw3_metal_read_text_file(NSString *path, NSError **error) {
    return [NSString stringWithContentsOfFile:path
                                    encoding:NSUTF8StringEncoding
                                       error:error];
}

static NSString *qw3_metal_join_path(NSString *dir, NSString *name) {
    return [dir stringByAppendingPathComponent:name];
}

static NSString *qw3_metal_kernel_source_from_dir(NSString *dir) {
    NSArray<NSString *> *parts = @[
        @"qw3_core_common.metal",
        @"qw3_core_linear.metal",
        @"qw3_core_sequence.metal",
        @"qw3_core_deltanet.metal",
        @"qw3_core_moe.metal",
        @"qw3_core_argmax.metal",
    ];
    NSMutableString *source = [NSMutableString string];
    for (NSString *part in parts) {
        NSError *read_error = nil;
        NSString *path = qw3_metal_join_path(dir, part);
        NSString *part_source = qw3_metal_read_text_file(path, &read_error);
        if (!part_source) return nil;
        [source appendString:part_source];
    }
    return source;
}

static NSString *qw3_metal_kernel_source(void) {
    const char *kernel_path_env = getenv("QW3_METAL_KERNEL_SOURCE");
    NSError *read_error = nil;
    if (kernel_path_env && kernel_path_env[0]) {
        NSString *path = [NSString stringWithUTF8String:kernel_path_env];
        NSString *source = qw3_metal_read_text_file(path, &read_error);
        if (source) return source;
        fprintf(stderr, "qw3: failed to read Metal kernel source %s: %s\n",
                [path UTF8String],
                read_error ? [[read_error localizedDescription] UTF8String] : "(unknown)");
        return nil;
    }

    NSArray<NSString *> *dirs = @[
        @"metal",
        @"../metal",
    ];
    for (NSString *dir in dirs) {
        NSString *source = qw3_metal_kernel_source_from_dir(dir);
        if (source) return source;
    }

    NSArray<NSString *> *monolithic_candidates = @[
        @"metal/qw3_kernels.metal",
        @"../metal/qw3_kernels.metal",
    ];
    for (NSString *candidate in monolithic_candidates) {
        read_error = nil;
        NSString *source = qw3_metal_read_text_file(candidate, &read_error);
        if (source) return source;
    }
    fprintf(stderr,
            "qw3: failed to read split Metal kernel sources from metal/ "
            "(set QW3_METAL_KERNEL_SOURCE or run from the project root)\n");
    return nil;
}

static NSString *qw3_metal_full_kernel_source(void) {
    NSString *core_source = qw3_metal_kernel_source();
    if (!core_source) return nil;
    NSMutableString *source = [NSMutableString stringWithString:core_source];
    g_gqa_flash_attn_external_enabled = 0;
    const char *flash_env = getenv("QW3_METAL_FLASH_ATTN");
    const char *gqa_flash_env = getenv("QW3_METAL_GQA_FLASH_ATTN");
    const char *gqa_flash_disable_env = getenv("QW3_METAL_GQA_FLASH_ATTN_DISABLE");
    const char *flash_path_env = getenv("QW3_METAL_FLASH_ATTN_SOURCE");
    const int gqa_flash_disabled =
        gqa_flash_disable_env != NULL ||
        (gqa_flash_env && gqa_flash_env[0] && strcmp(gqa_flash_env, "0") == 0);
    if (gqa_flash_disabled && (!flash_env || !flash_env[0]) &&
        (!flash_path_env || !flash_path_env[0])) {
        return source;
    }
    const int explicit_flash =
        (flash_env && flash_env[0]) ||
        (gqa_flash_env && gqa_flash_env[0] && strcmp(gqa_flash_env, "0") != 0) ||
        (flash_path_env && flash_path_env[0]);

    NSError *read_error = nil;
    NSString *path = nil;
    NSString *flash_source = nil;
    if (flash_path_env && flash_path_env[0]) {
        path = [NSString stringWithUTF8String:flash_path_env];
        flash_source = [NSString stringWithContentsOfFile:path
                                                 encoding:NSUTF8StringEncoding
                                                    error:&read_error];
    } else {
        NSArray<NSString *> *candidates = @[
            @"metal/flash_attn.metal",
            @"../ds4/metal/flash_attn.metal",
            @"../metal/flash_attn.metal",
        ];
        for (NSString *candidate in candidates) {
            read_error = nil;
            NSString *candidate_source =
                [NSString stringWithContentsOfFile:candidate
                                          encoding:NSUTF8StringEncoding
                                             error:&read_error];
            if (candidate_source) {
                path = candidate;
                flash_source = candidate_source;
                break;
            }
        }
        if (!path) path = [candidates lastObject];
    }
    if (!flash_source) {
        if (explicit_flash) {
            fprintf(stderr, "qw3: failed to read Metal flash attention source %s: %s\n",
                    [path UTF8String],
                    read_error ? [[read_error localizedDescription] UTF8String] : "(unknown)");
        }
        return explicit_flash ? nil : source;
    }
    g_gqa_flash_attn_external_enabled = 1;

    [source appendString:
        @"\n// QW3 FlashAttention prelude\n"
         "#ifndef MAX\n"
         "#define MAX(x, y) ((x) > (y) ? (x) : (y))\n"
         "#endif\n"
         "#ifndef MIN\n"
         "#define MIN(x, y) ((x) < (y) ? (x) : (y))\n"
         "#endif\n"
         "#ifndef N_SIMDWIDTH\n"
         "#define N_SIMDWIDTH 32\n"
         "#endif\n"
         "#ifndef FOR_UNROLL\n"
         "#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
         "#endif\n"];
    [source appendFormat:@"\n// appended %@\n%@\n", path, flash_source];
    [source appendString:
        @"\n// QW3 Qwen3 full-attention prefill instantiation, head_dim = 256.\n"
         "#define QW3_FA_NONVEC_TYPES \\\n"
         "    half,   half4,     simdgroup_half8x8,  \\\n"
         "    half,   half4x4,   simdgroup_half8x8,  \\\n"
         "    half,   half4x4,   simdgroup_half8x8,  \\\n"
         "    float,             simdgroup_float8x8, \\\n"
         "    float,  float2,    simdgroup_float8x8, \\\n"
         "    float,  float4,    simdgroup_float8x8\n"
         "typedef decltype(kernel_flash_attn_ext<QW3_FA_NONVEC_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 256, 256>) qw3_flash_attn_ext_dk256_t;\n"
         "template [[host_name(\"qw3_kernel_flash_attn_ext_f16_dk256_dv256\")]]\n"
         "kernel qw3_flash_attn_ext_dk256_t kernel_flash_attn_ext<QW3_FA_NONVEC_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 256, 256>;\n"
         "#undef QW3_FA_NONVEC_TYPES\n"];
    return source;
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
        g_moe_swiglu_slots_to_mid_f32_pipeline &&
        g_moe_swiglu_slots_to_hidden_f16_pipeline &&
        g_moe_down_iq4_xs_prefill_mapped_pipeline &&
        g_moe_down_iq4_xs_prefill_mapped_mid_f32_pipeline &&
        g_moe_down_iq4_xs_prefill_mapped_f16_pipeline &&
        g_moe_down_q6_k_prefill_mapped_pipeline &&
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
        g_gqa_q_norm_gate_rope_batch_pipeline &&
        g_gqa_k_norm_rope_batch_pipeline &&
        g_rope_heads_pipeline && g_rope_heads_batch_pipeline &&
        g_gqa_single_token_inner_pipeline &&
        g_gqa_attend2_inner_pipeline &&
        g_gqa_attend_n_inner_pipeline &&
        g_gqa_prefill_attend_inner_pipeline &&
        g_gqa_prefill_write_cache_pipeline &&
        g_gqa_prefill_cached_attend_inner_pipeline &&
        g_gqa_prefill_cached_attend_block2_pipeline &&
        g_gqa_prefill_cached_attend_block4_pipeline &&
        g_gqa_prefill_cached_attend_src8_pipeline &&
        g_gqa_flash_gate_pipeline &&
        g_gqa_flash_causal_mask_pipeline &&
        g_gqa_store_token_cache_f16_pipeline &&
        g_gqa_kv_quant_q8_pipeline &&
        g_gqa_attend_n_q8_inner_pipeline &&
        g_gqa_attend_n_split_partial_pipeline &&
        g_gqa_attend_n_split_reduce_pipeline &&
        g_gqa_attend_n_q8_split_partial_pipeline &&
        g_gqa_attend_n_q8_split_reduce_pipeline &&
        g_deltanet_recur_zero_pipeline &&
        g_deltanet_recur_pipeline && g_deltanet_recur_scratch_gates_pipeline &&
        g_deltanet_prepare_scratch_gates_pipeline &&
        g_deltanet_recur_scratch_gates_tiled_pipeline &&
        g_deltanet_fused_gdn_scratch_pipeline &&
        g_deltanet_batch_fused_gdn_pipeline &&
        g_deltanet_batch_recur_tiled_pipeline &&
        g_deltanet_batch_recur_tiled2_pipeline &&
        g_deltanet_batch_recur_tiled4_pipeline &&
        g_deltanet_batch_gated_rmsnorm_pipeline &&
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
    MTLCompileOptions *options = [MTLCompileOptions new];
    if (g_metal4_tensor_api_enabled) {
        options.preprocessorMacros = @{ @"QW3_METAL_HAS_TENSOR": @"1" };
    }
    NSString *kernel_source = qw3_metal_full_kernel_source();
    if (!kernel_source) return 0;
    g_library = [g_device newLibraryWithSource:kernel_source
                                       options:options
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
    g_moe_iq3_s_prefill_mapped_mpp_pipeline = nil;
    if (g_metal4_tensor_api_enabled) {
        fn = [g_library newFunctionWithName:@"qw3_moe_iq3_s_prefill_mapped_mpp"];
        if (fn) {
            g_moe_iq3_s_prefill_mapped_mpp_pipeline =
                [g_device newComputePipelineStateWithFunction:fn error:&error];
            if (!g_moe_iq3_s_prefill_mapped_mpp_pipeline) {
                fprintf(stderr, "qw3: Metal pipeline qw3_moe_iq3_s_prefill_mapped_mpp failed: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        } else {
            fprintf(stderr, "qw3: Metal function qw3_moe_iq3_s_prefill_mapped_mpp not found\n");
        }
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
    g_moe_iq3_s_prefill_pair_mapped_mpp_pipeline = nil;
    if (g_metal4_tensor_api_enabled) {
        fn = [g_library newFunctionWithName:@"qw3_moe_iq3_s_swiglu_prefill_pair_mapped_mpp"];
        if (fn) {
            g_moe_iq3_s_prefill_pair_mapped_mpp_pipeline =
                [g_device newComputePipelineStateWithFunction:fn error:&error];
            if (!g_moe_iq3_s_prefill_pair_mapped_mpp_pipeline) {
                fprintf(stderr, "qw3: Metal pipeline qw3_moe_iq3_s_swiglu_prefill_pair_mapped_mpp failed: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        } else {
            fprintf(stderr, "qw3: Metal function qw3_moe_iq3_s_swiglu_prefill_pair_mapped_mpp not found\n");
        }
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
    fn = [g_library newFunctionWithName:@"qw3_moe_swiglu_slots_to_mid_f32"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_moe_swiglu_slots_to_mid_f32 not found\n");
        return 0;
    }
    g_moe_swiglu_slots_to_mid_f32_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_swiglu_slots_to_mid_f32_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_moe_swiglu_slots_to_mid_f32 failed: %s\n",
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
    fn = [g_library newFunctionWithName:@"qw3_moe_down_iq4_xs_prefill_mapped_mid_f32"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_moe_down_iq4_xs_prefill_mapped_mid_f32 not found\n");
        return 0;
    }
    g_moe_down_iq4_xs_prefill_mapped_mid_f32_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_down_iq4_xs_prefill_mapped_mid_f32_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_moe_down_iq4_xs_prefill_mapped_mid_f32 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    g_moe_down_iq4_xs_prefill_mapped_mid_f32_mpp_pipeline = nil;
    if (g_metal4_tensor_api_enabled) {
        fn = [g_library newFunctionWithName:@"qw3_moe_down_iq4_xs_prefill_mapped_mid_f32_mpp"];
        if (fn) {
            g_moe_down_iq4_xs_prefill_mapped_mid_f32_mpp_pipeline =
                [g_device newComputePipelineStateWithFunction:fn error:&error];
            if (!g_moe_down_iq4_xs_prefill_mapped_mid_f32_mpp_pipeline) {
                fprintf(stderr, "qw3: Metal pipeline qw3_moe_down_iq4_xs_prefill_mapped_mid_f32_mpp failed: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        } else {
            fprintf(stderr, "qw3: Metal function qw3_moe_down_iq4_xs_prefill_mapped_mid_f32_mpp not found\n");
        }
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
    g_moe_down_iq4_xs_prefill_mapped_f16_mpp_pipeline = nil;
    if (g_metal4_tensor_api_enabled) {
        fn = [g_library newFunctionWithName:@"qw3_moe_down_iq4_xs_prefill_mapped_f16_mpp"];
        if (fn) {
            g_moe_down_iq4_xs_prefill_mapped_f16_mpp_pipeline =
                [g_device newComputePipelineStateWithFunction:fn error:&error];
            if (!g_moe_down_iq4_xs_prefill_mapped_f16_mpp_pipeline) {
                fprintf(stderr, "qw3: Metal pipeline qw3_moe_down_iq4_xs_prefill_mapped_f16_mpp failed: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        } else {
            fprintf(stderr, "qw3: Metal function qw3_moe_down_iq4_xs_prefill_mapped_f16_mpp not found\n");
        }
    }
    fn = [g_library newFunctionWithName:@"qw3_moe_down_q6_k_prefill_mapped"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_moe_down_q6_k_prefill_mapped not found\n");
        return 0;
    }
    g_moe_down_q6_k_prefill_mapped_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_moe_down_q6_k_prefill_mapped_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_moe_down_q6_k_prefill_mapped failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    g_moe_down_q6_k_prefill_mapped_mpp_pipeline = nil;
    if (g_metal4_tensor_api_enabled) {
        fn = [g_library newFunctionWithName:@"qw3_moe_down_q6_k_prefill_mapped_mpp"];
        if (fn) {
            g_moe_down_q6_k_prefill_mapped_mpp_pipeline =
                [g_device newComputePipelineStateWithFunction:fn error:&error];
            if (!g_moe_down_q6_k_prefill_mapped_mpp_pipeline) {
                fprintf(stderr, "qw3: Metal pipeline qw3_moe_down_q6_k_prefill_mapped_mpp failed: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        } else {
            fprintf(stderr, "qw3: Metal function qw3_moe_down_q6_k_prefill_mapped_mpp not found\n");
        }
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
    fn = [g_library newFunctionWithName:@"qw3_gqa_q_norm_gate_rope_batch"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_q_norm_gate_rope_batch not found\n");
        return 0;
    }
    g_gqa_q_norm_gate_rope_batch_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_gqa_q_norm_gate_rope_batch_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_q_norm_gate_rope_batch failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_gqa_k_norm_rope_batch"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_k_norm_rope_batch not found\n");
        return 0;
    }
    g_gqa_k_norm_rope_batch_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_gqa_k_norm_rope_batch_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_k_norm_rope_batch failed: %s\n",
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
    fn = [g_library newFunctionWithName:@"qw3_gqa_prefill_cached_attend_block2"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_prefill_cached_attend_block2 not found\n");
        return 0;
    }
    g_gqa_prefill_cached_attend_block2_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_gqa_prefill_cached_attend_block2_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_prefill_cached_attend_block2 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_gqa_prefill_cached_attend_block4"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_prefill_cached_attend_block4 not found\n");
        return 0;
    }
    g_gqa_prefill_cached_attend_block4_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_gqa_prefill_cached_attend_block4_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_prefill_cached_attend_block4 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_gqa_prefill_cached_attend_src8"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_prefill_cached_attend_src8 not found\n");
        return 0;
    }
    g_gqa_prefill_cached_attend_src8_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_gqa_prefill_cached_attend_src8_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_prefill_cached_attend_src8 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_gqa_flash_gate_from_compact"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_flash_gate_from_compact not found\n");
        return 0;
    }
    g_gqa_flash_gate_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_gqa_flash_gate_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_flash_gate_from_compact failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_gqa_flash_causal_mask_block"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_flash_causal_mask_block not found\n");
        return 0;
    }
    g_gqa_flash_causal_mask_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_gqa_flash_causal_mask_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_flash_causal_mask_block failed: %s\n",
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
    fn = [g_library newFunctionWithName:@"qw3_gqa_store_token_cache_f16"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_store_token_cache_f16 not found\n");
        return 0;
    }
    g_gqa_store_token_cache_f16_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_gqa_store_token_cache_f16_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_store_token_cache_f16 failed: %s\n",
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
    fn = [g_library newFunctionWithName:@"qw3_gqa_attend_n_split_partial"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_attend_n_split_partial not found\n");
        return 0;
    }
    g_gqa_attend_n_split_partial_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_gqa_attend_n_split_partial_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_attend_n_split_partial failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_gqa_attend_n_split_reduce"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_gqa_attend_n_split_reduce not found\n");
        return 0;
    }
    g_gqa_attend_n_split_reduce_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_gqa_attend_n_split_reduce_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_gqa_attend_n_split_reduce failed: %s\n",
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
    fn = [g_library newFunctionWithName:@"qw3_deltanet_batch_recur_tiled"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_deltanet_batch_recur_tiled not found\n");
        return 0;
    }
    g_deltanet_batch_recur_tiled_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_deltanet_batch_recur_tiled_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_deltanet_batch_recur_tiled failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_deltanet_batch_recur_tiled2"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_deltanet_batch_recur_tiled2 not found\n");
        return 0;
    }
    g_deltanet_batch_recur_tiled2_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_deltanet_batch_recur_tiled2_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_deltanet_batch_recur_tiled2 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_deltanet_batch_recur_tiled4"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_deltanet_batch_recur_tiled4 not found\n");
        return 0;
    }
    g_deltanet_batch_recur_tiled4_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_deltanet_batch_recur_tiled4_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_deltanet_batch_recur_tiled4 failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return 0;
    }
    fn = [g_library newFunctionWithName:@"qw3_deltanet_batch_gated_rmsnorm"];
    if (!fn) {
        fprintf(stderr, "qw3: Metal function qw3_deltanet_batch_gated_rmsnorm not found\n");
        return 0;
    }
    g_deltanet_batch_gated_rmsnorm_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_deltanet_batch_gated_rmsnorm_pipeline) {
        fprintf(stderr, "qw3: Metal pipeline qw3_deltanet_batch_gated_rmsnorm failed: %s\n",
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
    if (!cb) return 0;
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

static NSArray<id<MTLBuffer>> *qw3_metal_new_private_buffer_layers(uint32_t n_layers,
                                                                    uint64_t bytes) {
    if (!g_device || n_layers == 0 || bytes == 0 ||
        bytes > (uint64_t)NSUIntegerMax) {
        return nil;
    }
    NSMutableArray<id<MTLBuffer>> *layers =
        [NSMutableArray arrayWithCapacity:n_layers];
    for (uint32_t i = 0; i < n_layers; i++) {
        id<MTLBuffer> b = qw3_metal_new_private_buffer(bytes);
        if (!b) return nil;
        [layers addObject:b];
    }
    return [layers copy];
}

static int qw3_metal_gqa_layer_buffers(QW3MetalSessionObj *obj,
                                       uint32_t layer_slot,
                                       uint64_t cache_layer_bytes,
                                       id<MTLBuffer> *k_buffer,
                                       id<MTLBuffer> *v_buffer,
                                       NSUInteger *offset) {
    if (!obj || layer_slot >= obj.metalFullLayers ||
        !k_buffer || !v_buffer || !offset) {
        return 0;
    }
    if (obj.gqaKLayers.count > 0 || obj.gqaVLayers.count > 0) {
        if (layer_slot >= obj.gqaKLayers.count ||
            layer_slot >= obj.gqaVLayers.count) {
            return 0;
        }
        id<MTLBuffer> kb = obj.gqaKLayers[layer_slot];
        id<MTLBuffer> vb = obj.gqaVLayers[layer_slot];
        if (!kb || !vb || kb.length < cache_layer_bytes ||
            vb.length < cache_layer_bytes) {
            return 0;
        }
        *k_buffer = kb;
        *v_buffer = vb;
        *offset = 0;
        return 1;
    }
    const uint64_t cache_offset = (uint64_t)layer_slot * cache_layer_bytes;
    if (!obj.gqaK || !obj.gqaV ||
        obj.gqaK.length < cache_offset + cache_layer_bytes ||
        obj.gqaV.length < cache_offset + cache_layer_bytes ||
        cache_offset > (uint64_t)NSUIntegerMax) {
        return 0;
    }
    *k_buffer = obj.gqaK;
    *v_buffer = obj.gqaV;
    *offset = (NSUInteger)cache_offset;
    return 1;
}

typedef struct {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
    float    logit_softcap;
} qw3_metal_flash_attn_args;

typedef struct {
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
} qw3_metal_flash_attn_pad_args;

typedef struct {
    int32_t  ne01;
    int32_t  ne30;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
} qw3_metal_flash_attn_blk_args;

static id<MTLComputePipelineState> qw3_metal_gqa_flash_pad_pipeline(void) {
    static int unavailable;
    if (g_gqa_flash_pad_pipeline) return g_gqa_flash_pad_pipeline;
    if (unavailable || !g_library) return nil;

    NSError *error = nil;
    id<MTLFunction> fn =
        [g_library newFunctionWithName:@"qw3_gqa_flash_pad_interleaved"];
    if (!fn) {
        unavailable = 1;
        return nil;
    }
    g_gqa_flash_pad_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_gqa_flash_pad_pipeline) {
        unavailable = 1;
        return nil;
    }
    return g_gqa_flash_pad_pipeline;
}

static id<MTLComputePipelineState> qw3_metal_gqa_flash_blk_pipeline(void) {
    static int unavailable;
    if (g_gqa_flash_blk_pipeline) return g_gqa_flash_blk_pipeline;
    if (unavailable || !g_library) return nil;

    const int32_t nqptg = 8;
    const int32_t ncpsg = 64;
    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&nqptg type:MTLDataTypeInt atIndex:224];
    [constants setConstantValue:&ncpsg type:MTLDataTypeInt atIndex:225];

    NSError *error = nil;
    id<MTLFunction> fn =
        [g_library newFunctionWithName:@"kernel_flash_attn_ext_blk"
                        constantValues:constants
                                 error:&error];
    if (!fn) {
        unavailable = 1;
        return nil;
    }
    g_gqa_flash_blk_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_gqa_flash_blk_pipeline) {
        unavailable = 1;
        return nil;
    }
    return g_gqa_flash_blk_pipeline;
}

static id<MTLComputePipelineState> qw3_metal_gqa_flash_attn_pipeline(uint32_t kv_stride_elems,
                                                                      int has_kvpad,
                                                                      int bc_mask) {
    static uint32_t cached_kv_stride_elems;
    static int cached_has_kvpad;
    static int cached_bc_mask;
    static int unavailable;
    if (g_gqa_flash_attn_pipeline &&
        cached_kv_stride_elems == kv_stride_elems &&
        cached_has_kvpad == has_kvpad &&
        cached_bc_mask == bc_mask) {
        return g_gqa_flash_attn_pipeline;
    }
    if (g_gqa_flash_attn_pipeline) {
        g_gqa_flash_attn_pipeline = nil;
    }
    if (unavailable || !g_library || kv_stride_elems == 0) return nil;

    const bool has_mask = true;
    const bool has_sinks = false;
    const bool has_bias = false;
    const bool has_scap = false;
    const bool has_kvpad_bool = has_kvpad != 0;
    const bool bc_mask_bool = bc_mask != 0;
    const int32_t ns10 = (int32_t)kv_stride_elems;
    const int32_t ns20 = (int32_t)kv_stride_elems;
    const int32_t nsg = 4;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&has_mask  type:MTLDataTypeBool atIndex:300];
    [constants setConstantValue:&has_sinks type:MTLDataTypeBool atIndex:301];
    [constants setConstantValue:&has_bias  type:MTLDataTypeBool atIndex:302];
    [constants setConstantValue:&has_scap  type:MTLDataTypeBool atIndex:303];
    [constants setConstantValue:&has_kvpad_bool type:MTLDataTypeBool atIndex:304];
    [constants setConstantValue:&bc_mask_bool   type:MTLDataTypeBool atIndex:310];
    [constants setConstantValue:&ns10 type:MTLDataTypeInt atIndex:320];
    [constants setConstantValue:&ns20 type:MTLDataTypeInt atIndex:321];
    [constants setConstantValue:&nsg  type:MTLDataTypeInt atIndex:322];

    NSError *error = nil;
    id<MTLFunction> fn =
        [g_library newFunctionWithName:@"qw3_kernel_flash_attn_ext_f16_dk256_dv256"
                        constantValues:constants
                                 error:&error];
    if (!fn) {
        unavailable = 1;
        fprintf(stderr, "qw3: Metal FlashAttention function not found: %s\n",
                error ? [[error localizedDescription] UTF8String] : "(unknown)");
        return nil;
    }

    g_gqa_flash_attn_pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!g_gqa_flash_attn_pipeline) {
        unavailable = 1;
        fprintf(stderr, "qw3: Metal FlashAttention pipeline failed: %s\n",
                error ? [[error localizedDescription] UTF8String] : "(unknown)");
        return nil;
    }
    cached_kv_stride_elems = kv_stride_elems;
    cached_has_kvpad = has_kvpad;
    cached_bc_mask = bc_mask;
    return g_gqa_flash_attn_pipeline;
}

static int qw3_metal_session_ensure_flash_attn_buffers(QW3MetalSessionObj *obj,
                                                        uint32_t pos0,
                                                        uint32_t n_tokens,
                                                        uint32_t n_keys,
                                                        uint32_t n_heads,
                                                        uint32_t n_kv_heads,
                                                        uint32_t head_dim) {
    if (!obj || !g_device || n_tokens == 0 || n_keys == 0 ||
        n_heads == 0 || n_kv_heads == 0 || head_dim == 0) {
        return 0;
    }
    const uint32_t ncpsg = 64u;
    const int has_kvpad = (n_keys % ncpsg) != 0u;
    const uint64_t out_bytes =
        (uint64_t)n_tokens * n_heads * head_dim * sizeof(float);
    const uint64_t mask_bytes =
        (uint64_t)n_tokens * n_keys * sizeof(uint16_t);
    const uint64_t block_bytes =
        (((uint64_t)n_tokens + 7ull) / 8ull) *
        (((uint64_t)n_keys + 63ull) / 64ull);
    const uint64_t kv_stride = (uint64_t)n_kv_heads * head_dim;
    const uint64_t pad_bytes = has_kvpad ?
        (uint64_t)ncpsg *
            (2ull * (uint64_t)n_kv_heads * kv_stride * sizeof(uint16_t) +
             (uint64_t)n_tokens * sizeof(uint16_t)) :
        1ull;
    if (out_bytes > (uint64_t)NSUIntegerMax ||
        mask_bytes > (uint64_t)NSUIntegerMax ||
        block_bytes > (uint64_t)NSUIntegerMax ||
        pad_bytes > (uint64_t)NSUIntegerMax) {
        return 0;
    }

    if (!obj.flashAttnOut || obj.flashAttnOut.length < out_bytes) {
        obj.flashAttnOut = qw3_metal_new_private_buffer(out_bytes);
        if (!obj.flashAttnOut) return 0;
    }
    int refresh = obj.flashAttnMaskPos0 != pos0 ||
        obj.flashAttnMaskTokens != n_tokens ||
        obj.flashAttnMaskKeys != n_keys;
    if (!obj.flashAttnMask || obj.flashAttnMask.length < mask_bytes) {
        obj.flashAttnMask =
            [g_device newBufferWithLength:(NSUInteger)qw3_metal_session_align(mask_bytes)
                                  options:MTLResourceStorageModeShared];
        if (!obj.flashAttnMask) return 0;
        refresh = 1;
    }
    if (!obj.flashAttnBlock || obj.flashAttnBlock.length < block_bytes) {
        obj.flashAttnBlock =
            [g_device newBufferWithLength:(NSUInteger)qw3_metal_session_align(block_bytes)
                                  options:MTLResourceStorageModeShared];
        if (!obj.flashAttnBlock) return 0;
        refresh = 1;
    }
    if (!obj.flashAttnPad || obj.flashAttnPad.length < pad_bytes) {
        obj.flashAttnPad = qw3_metal_new_private_buffer(pad_bytes);
        if (!obj.flashAttnPad) return 0;
    }
    if (!refresh) return 1;

    const char *gpu_mask_env = getenv("QW3_METAL_GQA_FLASH_GPU_MASK");
    const int gpu_mask =
        gpu_mask_env && gpu_mask_env[0] && strcmp(gpu_mask_env, "0") != 0;
    if (gpu_mask) {
        obj.flashAttnMaskPos0 = pos0;
        obj.flashAttnMaskTokens = n_tokens;
        obj.flashAttnMaskKeys = n_keys;
        return 2;
    }

    if (g_batch_cb && !qw3_metal_synchronize()) return 0;

    uint16_t *mask = (uint16_t *)obj.flashAttnMask.contents;
    if (!mask) return 0;
    const uint16_t neg_inf_half = 0xfbffu;
    for (uint32_t q = 0; q < n_tokens; q++) {
        uint16_t *row = mask + (uint64_t)q * n_keys;
        uint32_t allowed = pos0 + q + 1u;
        if (allowed > n_keys) allowed = n_keys;
        for (uint32_t k = 0; k < allowed; k++) row[k] = 0;
        for (uint32_t k = allowed; k < n_keys; k++) row[k] = neg_inf_half;
    }
    [obj.flashAttnMask didModifyRange:NSMakeRange(0, (NSUInteger)mask_bytes)];

    obj.flashAttnMaskPos0 = pos0;
    obj.flashAttnMaskTokens = n_tokens;
    obj.flashAttnMaskKeys = n_keys;
    return 1;
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
    const uint64_t block_ids_bytes =
        (uint64_t)n_expert * (uint64_t)((n_tokens + 31u) / 32u) *
        sizeof(uint32_t);
    const uint64_t block_count_bytes = sizeof(uint32_t);
    const uint64_t dispatch_bytes = 3u * sizeof(uint32_t);
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
    if (!obj.moeBlockCount || obj.moeBlockCount.length < block_count_bytes) {
        obj.moeBlockCount = qw3_metal_new_private_buffer(block_count_bytes);
    }
    if (!obj.moeBlockIds || obj.moeBlockIds.length < block_ids_bytes) {
        obj.moeBlockIds = qw3_metal_new_private_buffer(block_ids_bytes);
    }
    if (!obj.moeBlockDispatchFF ||
        obj.moeBlockDispatchFF.length < dispatch_bytes) {
        obj.moeBlockDispatchFF = qw3_metal_new_private_buffer(dispatch_bytes);
    }
    if (!obj.moeBlockDispatchEmbd ||
        obj.moeBlockDispatchEmbd.length < dispatch_bytes) {
        obj.moeBlockDispatchEmbd = qw3_metal_new_private_buffer(dispatch_bytes);
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
           obj.moeBlockCount && obj.moeBlockIds &&
           obj.moeBlockDispatchFF && obj.moeBlockDispatchEmbd &&
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
    const char *kv_f16_env = getenv("QW3_METAL_KV_F16");
    const BOOL gqa_kv_f16 =
        !gqa_kv_q8 && (!kv_f16_env || strcmp(kv_f16_env, "0") != 0);
    const BOOL gqa_split_q8 =
        gqa_kv_q8 && getenv("QW3_METAL_LEGACY_Q8_ATTN") == NULL;
    const char *split_attn_env = getenv("QW3_METAL_GQA_SPLIT_ATTN");
    const BOOL gqa_split_attn =
        !gqa_kv_q8 && (!split_attn_env || strcmp(split_attn_env, "0") != 0);
    const uint32_t gqa_max_q8_splits = !gqa_split_q8 ||
        getenv("QW3_METAL_Q8_SPLIT_32") != NULL ? 32u :
        (getenv("QW3_METAL_Q8_SPLIT_64") != NULL ? 64u :
         (getenv("QW3_METAL_Q8_SPLIT_128") != NULL ? 128u : 256u));
    const uint32_t gqa_max_attn_splits = gqa_split_attn ? 256u : 32u;
    uint32_t metal_full_layers = QW3_METAL_N_FULL_ATTN_LAYERS;
    uint32_t metal_linear_layers = QW3_METAL_N_LINEAR_LAYERS;
    const uint32_t ngl = qw3_metal_env_n_gpu_layers();
    if (ngl < 40u && qw3_metal_llama_split_enabled()) {
        qw3_metal_count_layer_types_from(qw3_metal_llama_split_start(ngl),
                                         &metal_full_layers,
                                         &metal_linear_layers);
    } else {
        qw3_metal_count_layer_types_before(ngl, &metal_full_layers,
                                           &metal_linear_layers);
    }
    const uint64_t gqa_cache_token_bytes = gqa_kv_q8 ?
        (uint64_t)QW3_METAL_N_HEAD_KV * (QW3_METAL_N_HEAD_DIM / 32u) * 34ull :
        (gqa_kv_f16 ?
         (uint64_t)QW3_METAL_N_HEAD_KV * QW3_METAL_N_HEAD_DIM * sizeof(uint16_t) :
         (uint64_t)QW3_METAL_N_HEAD_KV * QW3_METAL_N_HEAD_DIM * sizeof(float));
    const uint64_t gqa_kv_bytes =
        (uint64_t)metal_full_layers * ctx_size *
        gqa_cache_token_bytes;
    const uint64_t deltanet_state_bytes =
        (uint64_t)metal_linear_layers * QW3_METAL_N_LINEAR_V_HEADS *
        QW3_METAL_N_LINEAR_HEAD_DIM * QW3_METAL_N_LINEAR_HEAD_DIM *
        sizeof(float);
    const uint64_t conv_state_bytes =
        (uint64_t)metal_linear_layers * QW3_METAL_LINEAR_QKV *
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
    const uint32_t gqa_max_splits =
        gqa_max_q8_splits > gqa_max_attn_splits ?
        gqa_max_q8_splits : gqa_max_attn_splits;
    const uint64_t gqa_attn_partial_bytes = (gqa_split_q8 || gqa_split_attn) ?
        (uint64_t)gqa_max_splits * QW3_METAL_N_HEAD *
            (QW3_METAL_N_HEAD_DIM + 2ull) * sizeof(float) :
        0ull;

    QW3MetalSessionObj *obj = [[QW3MetalSessionObj alloc] init];
    obj.ctxSize = ctx_size;
    obj.vocabSize = vocab_size;
    obj.gqaKvQ8 = gqa_kv_q8;
    obj.gqaKvF16 = gqa_kv_f16;
    obj.gqaSplitQ8 = gqa_split_q8;
    obj.gqaSplitAttn = gqa_split_attn;
    obj.gqaMaxQ8Splits = gqa_max_q8_splits;
    obj.gqaMaxAttnSplits = gqa_max_attn_splits;
    obj.metalFullLayers = metal_full_layers;
    obj.metalLinearLayers = metal_linear_layers;
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

    const char *gqa_layer_buffers_env = getenv("QW3_METAL_GQA_LAYER_BUFFERS");
    const BOOL gqa_layer_buffers = metal_full_layers > 0 &&
        (!gqa_layer_buffers_env || strcmp(gqa_layer_buffers_env, "0") != 0);
    if (gqa_layer_buffers) {
        const uint64_t gqa_layer_bytes =
            (uint64_t)ctx_size * gqa_cache_token_bytes;
        obj.gqaKLayers =
            qw3_metal_new_private_buffer_layers(metal_full_layers,
                                                qw3_metal_alloc_size(gqa_layer_bytes));
        obj.gqaVLayers =
            qw3_metal_new_private_buffer_layers(metal_full_layers,
                                                qw3_metal_alloc_size(gqa_layer_bytes));
    } else {
        obj.gqaK =
            qw3_metal_new_private_buffer(qw3_metal_alloc_size(gqa_kv_bytes));
        obj.gqaV =
            qw3_metal_new_private_buffer(qw3_metal_alloc_size(gqa_kv_bytes));
    }
    obj.deltanetState =
        qw3_metal_new_private_buffer(qw3_metal_alloc_size(deltanet_state_bytes));
    obj.convState =
        qw3_metal_new_private_buffer(qw3_metal_alloc_size(conv_state_bytes));
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

    if (((gqa_layer_buffers && (!obj.gqaKLayers || !obj.gqaVLayers)) ||
         (!gqa_layer_buffers && (!obj.gqaK || !obj.gqaV))) ||
        !obj.deltanetState || !obj.convState ||
        !obj.logits || !obj.x0 || !obj.x1 || !obj.scratch ||
        !obj.qkvConv || !obj.qNorm || !obj.kNorm || !obj.core || !obj.inner ||
        !obj.gqaTmpQ || !obj.gqaTmpK || !obj.gqaTokenQ || !obj.gqaTokenK ||
        !obj.gqaTokenV || !obj.gqaTokenGate || !obj.routerIds ||
        ((gqa_split_q8 || gqa_split_attn) && !obj.gqaAttnPartial) ||
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
    const char *force_kv_clear_env = getenv("QW3_METAL_FORCE_KV_CLEAR");
    const int clear_kv =
        force_kv_clear_env && strcmp(force_kv_clear_env, "0") != 0;
    const char *force_prefill_clear_env = getenv("QW3_METAL_FORCE_PREFILL_CLEAR");
    const int clear_prefill =
        force_prefill_clear_env && strcmp(force_prefill_clear_env, "0") != 0;
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    if (!cb) return 0;
    qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!blit) return 0;
    if (clear_kv) {
        if (obj.gqaKLayers.count > 0 || obj.gqaVLayers.count > 0) {
            for (id<MTLBuffer> b in obj.gqaKLayers) {
                if (b.length > 0) {
                    [blit fillBuffer:b range:NSMakeRange(0, b.length) value:0];
                }
            }
            for (id<MTLBuffer> b in obj.gqaVLayers) {
                if (b.length > 0) {
                    [blit fillBuffer:b range:NSMakeRange(0, b.length) value:0];
                }
            }
        } else {
            if (obj.gqaK.length > 0) {
                [blit fillBuffer:obj.gqaK range:NSMakeRange(0, obj.gqaK.length) value:0];
            }
            if (obj.gqaV.length > 0) {
                [blit fillBuffer:obj.gqaV range:NSMakeRange(0, obj.gqaV.length) value:0];
            }
        }
    }
    NSArray<id<MTLBuffer>> *buffers = @[
        obj.deltanetState, obj.convState,
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
    if (clear_prefill) {
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

static id<MTLComputePipelineState> qw3_metal_q8_nax_pipeline(uint32_t tile_n) {
    if (!g_metal4_tensor_api_enabled ||
        getenv("QW3_METAL_Q8_NAX_DISABLE") != NULL) {
        return nil;
    }

    __strong id<MTLComputePipelineState> *slot = NULL;
    const char *fn_name = NULL;
    if (tile_n == 128u) {
        slot = &g_matmul_q8_0_nax_n128_pipeline;
        fn_name = "qw3_matmul_q8_0_nax_direct_rhs_n128";
    } else if (tile_n == 64u) {
        slot = &g_matmul_q8_0_nax_n64_pipeline;
        fn_name = "qw3_matmul_q8_0_nax_direct_rhs_n64";
    } else {
        slot = &g_matmul_q8_0_nax_pipeline;
        fn_name = "qw3_matmul_q8_0_nax_direct_rhs";
    }
    if (*slot) return *slot;
    if (!g_library || !fn_name) return nil;

    NSError *error = nil;
    id<MTLFunction> fn =
        [g_library newFunctionWithName:[NSString stringWithUTF8String:fn_name]];
    if (!fn) {
        static int warned;
        if (!warned && getenv("QW3_METAL_Q8_NAX") != NULL) {
            fprintf(stderr, "qw3: Metal Q8_0 NAX function %s not found\n", fn_name);
            warned = 1;
        }
        return nil;
    }
    *slot = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!*slot) {
        static int warned;
        if (!warned && getenv("QW3_METAL_Q8_NAX") != NULL) {
            fprintf(stderr, "qw3: Metal Q8_0 NAX pipeline %s failed: %s\n",
                    fn_name, error ? [[error localizedDescription] UTF8String] : "(unknown)");
            warned = 1;
        }
        return nil;
    }
    return *slot;
}

static int qw3_metal_encode_batch_matmul_q8_0(
    id<MTLBuffer> wbuf, NSUInteger woff, id<MTLBuffer> xbuf,
    NSUInteger xoff, id<MTLBuffer> outbuf, NSUInteger outoff,
    uint32_t n_tokens, uint32_t n_in, uint32_t n_out, uint32_t in_stride,
    uint32_t out_stride, uint32_t row_bytes) {
    if (getenv("QW3_METAL_Q8_NAX_DISABLE") == NULL &&
        n_tokens >= 32u && (n_tokens % 32u) == 0u &&
        (n_in % 64u) == 0u && (n_out % 64u) == 0u) {
        uint32_t tile_n = 32u;
        if ((n_tokens % 128u) == 0u) {
            tile_n = 128u;
        } else if ((n_tokens % 64u) == 0u) {
            tile_n = 64u;
        }
        const char *tile_env = getenv("QW3_METAL_Q8_NAX_TILE");
        if (tile_env && tile_env[0]) {
            const long forced = strtol(tile_env, NULL, 10);
            if ((forced == 32 || forced == 64 || forced == 128) &&
                (n_tokens % (uint32_t)forced) == 0u) {
                tile_n = (uint32_t)forced;
            }
        }
        id<MTLComputePipelineState> pipeline =
            qw3_metal_q8_nax_pipeline(tile_n);
        if (pipeline) {
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
            int owned = 0;
            id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
            id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:wbuf offset:woff atIndex:1];
            [enc setBuffer:xbuf offset:xoff atIndex:2];
            [enc setBuffer:outbuf offset:outoff atIndex:3];
            [enc setThreadgroupMemoryLength:64u * 32u * sizeof(uint16_t)
                                    atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(n_tokens / tile_n,
                                                  n_out / 64u, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            qw3_metal_end_compute_encoder(cb, enc);
            if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
            if (cb.status == MTLCommandBufferStatusError) {
                fprintf(stderr, "qw3: Metal Q8_0 NAX prefill mm command failed: %s\n",
                        [[cb.error localizedDescription] UTF8String]);
                return 0;
            }
            return 1;
        }
    }

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

    const int use_tiled =
        getenv("QW3_METAL_BATCH_GDN_LEGACY") == NULL &&
        (head_dim % 4u) == 0u;
    const int use_tiled4 =
        use_tiled && g_deltanet_batch_recur_tiled4_pipeline &&
        getenv("QW3_METAL_BATCH_GDN_TILED4") != NULL;
    const int use_tiled2 =
        use_tiled && !use_tiled4 && g_deltanet_batch_recur_tiled2_pipeline &&
        (head_dim % 8u) == 0u &&
        getenv("QW3_METAL_BATCH_GDN_TILED2_DISABLE") == NULL;
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    if (use_tiled) {
        [enc setComputePipelineState:use_tiled4 ?
         g_deltanet_batch_recur_tiled4_pipeline :
         use_tiled2 ?
         g_deltanet_batch_recur_tiled2_pipeline :
         g_deltanet_batch_recur_tiled_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:obj.deltanetState offset:(NSUInteger)state_offset atIndex:1];
        [enc setBuffer:obj.prefillScratch offset:0 atIndex:2];
        [enc setBuffer:dtb offset:(NSUInteger)dt_inner atIndex:3];
        [enc setBuffer:ab offset:(NSUInteger)a_inner atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(use_tiled4 ?
                                              (NSUInteger)(head_dim + 15u) / 16u :
                                              use_tiled2 ?
                                              (NSUInteger)(head_dim + 7u) / 8u :
                                              (NSUInteger)(head_dim + 3u) / 4u,
                                              v_heads, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:g_deltanet_batch_gated_rmsnorm_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:obj.prefillScratch offset:0 atIndex:1];
        [enc setBuffer:wb offset:(NSUInteger)w_inner atIndex:2];
        [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
        NSUInteger norm_threads =
            g_deltanet_batch_gated_rmsnorm_pipeline.maxTotalThreadsPerThreadgroup;
        if (norm_threads > 256) norm_threads = 256;
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens * v_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(norm_threads, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);
        if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
        if (cb.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "qw3: Metal batch tiled Gated DeltaNet command failed: %s\n",
                    [[cb.error localizedDescription] UTF8String]);
            return 0;
        }
        return 1;
    }

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
    const int use_mapped_mpp =
        g_metal4_tensor_api_enabled &&
        getenv("QW3_METAL_MOE_MPP_DISABLE") == NULL;
    const int use_mapped_gateup_pair_auto =
        use_mapped_gateup && use_mapped_mpp &&
        g_moe_iq3_s_prefill_pair_mapped_mpp_pipeline &&
        getenv("QW3_METAL_MOE_MPP_GATEUP_DISABLE") == NULL &&
        getenv("QW3_METAL_MOE_MPP_GATEUP_PAIR_DISABLE") == NULL;
    const int use_mapped_gateup_pair =
        use_mapped_gateup &&
        getenv("QW3_METAL_MOE_MAP_GATEUP_PAIR_DISABLE") == NULL &&
        (getenv("QW3_METAL_MOE_MAP_GATEUP_PAIR") != NULL ||
         use_mapped_gateup_pair_auto);
    const int use_mapped_gateup_mpp =
        use_mapped_gateup && !use_mapped_gateup_pair &&
        use_mapped_mpp && g_moe_iq3_s_prefill_mapped_mpp_pipeline &&
        getenv("QW3_METAL_MOE_MPP_GATEUP_DISABLE") == NULL;
    const int use_mapped_gateup_pair_mpp =
        use_mapped_gateup_pair && use_mapped_mpp &&
        g_moe_iq3_s_prefill_pair_mapped_mpp_pipeline &&
        getenv("QW3_METAL_MOE_MPP_GATEUP_DISABLE") == NULL &&
        getenv("QW3_METAL_MOE_MPP_GATEUP_PAIR_DISABLE") == NULL;
    const int use_mapped_down =
        (down_type == 23 || down_type == 14) && n_tokens >= 32 &&
        getenv("QW3_METAL_MOE_MAP_DOWN_DISABLE") == NULL;
    const int use_mapped_mid_f16 =
        down_type == 23 && use_mapped_gateup && use_mapped_down &&
        !use_mapped_gateup_pair &&
        getenv("QW3_METAL_MOE_MID_F16") != NULL &&
        getenv("QW3_METAL_MOE_MID_F16_DISABLE") == NULL;
    const int use_mapped_mid_f32 =
        down_type == 23 && use_mapped_gateup && use_mapped_down &&
        (!use_mapped_gateup_pair || use_mapped_gateup_pair_mpp) &&
        !use_mapped_mid_f16 &&
        getenv("QW3_METAL_MOE_MID_F32_DISABLE") == NULL;
    const int use_mapped_mid_f32_mpp =
        use_mapped_mid_f32 && use_mapped_mpp &&
        g_moe_down_iq4_xs_prefill_mapped_mid_f32_mpp_pipeline &&
        getenv("QW3_METAL_MOE_MPP_DOWN_DISABLE") == NULL;
    const int use_mapped_mid_f16_mpp =
        use_mapped_mid_f16 && use_mapped_mpp &&
        g_moe_down_iq4_xs_prefill_mapped_f16_mpp_pipeline &&
        getenv("QW3_METAL_MOE_MPP_DOWN_DISABLE") == NULL;
    const int use_mapped_q6_mpp =
        down_type == 14 && use_mapped_down && use_mapped_mpp &&
        g_moe_down_q6_k_prefill_mapped_mpp_pipeline &&
        getenv("QW3_METAL_MOE_MPP_DOWN_DISABLE") == NULL &&
        getenv("QW3_METAL_MOE_Q6_MPP_DISABLE") == NULL;
    const int use_mapped_moe = use_mapped_gateup || use_mapped_down;
    const int use_compact_moe_blocks =
        use_mapped_moe && getenv("QW3_METAL_MOE_BLOCKS_DISABLE") == NULL;
    const int use_mapped_mid_preweighted =
        use_mapped_gateup_pair_mpp && use_mapped_mid_f32_mpp;
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
        uint32_t compact_blocks;
        uint32_t mid_preweighted;
    } args = {
        n_embd, n_ff, n_embd, n_tokens, n_active,
        (uint32_t)iq3_row_bytes, (uint32_t)iq3_expert_bytes,
        (uint32_t)down_row_bytes, (uint32_t)down_expert_bytes,
        stride, hidden_offset, (uint32_t)use_compact_moe_blocks,
        (uint32_t)use_mapped_mid_preweighted
    };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    if (!cb) return 0;
    const int profile_prefill_moe =
        getenv("QW3_METAL_PROFILE_PREFILL_MOE_SYNC") != NULL;
    const int profile_prefill_moe_batched = g_batch_cb != nil;
    const int profile_prefill_moe_concurrent = g_batch_concurrent;
    double profile_prefill_moe_t0 =
        profile_prefill_moe ? [NSDate timeIntervalSinceReferenceDate] : 0.0;
#define QW3_PROFILE_PREFILL_MOE_STAGE(stage_name) do {                              \
        if (profile_prefill_moe) {                                                  \
            if (profile_prefill_moe_batched) {                                      \
                if (!qw3_metal_synchronize()) return 0;                             \
            } else if (!qw3_metal_finish_command_buffer(cb, owned,                  \
                                                        "prefill MoE profile")) {   \
                return 0;                                                           \
            }                                                                       \
            const double profile_prefill_moe_t1 = [NSDate timeIntervalSinceReferenceDate]; \
            fprintf(stderr,                                                         \
                    "qw3 metal prefill moe stage tokens=%u active=%u down_type=%u " \
                    "mapped=%u/%u pair=%u gate_mpp=%u mid=%s compact=%u "          \
                    "prew=%u stage=%s ms=%.3f\n",                                 \
                    n_tokens, n_active, down_type,                                  \
                    (uint32_t)use_mapped_gateup, (uint32_t)use_mapped_down,         \
                    (uint32_t)use_mapped_gateup_pair,                               \
                    (uint32_t)(use_mapped_gateup_mpp || use_mapped_gateup_pair_mpp), \
                    use_mapped_mid_f32_mpp ? "f32c_mpp" :                           \
                    use_mapped_mid_f16_mpp ? "f16_mpp" :                            \
                    use_mapped_mid_f16 ? "f16" :                                    \
                    (use_mapped_mid_f32 ? "f32c" : "f32"),                         \
                    (uint32_t)use_compact_moe_blocks,                               \
                    (uint32_t)use_mapped_mid_preweighted,                           \
                    (stage_name),                                                   \
                    (profile_prefill_moe_t1 - profile_prefill_moe_t0) * 1000.0);    \
            if (profile_prefill_moe_batched &&                                      \
                !(profile_prefill_moe_concurrent ?                                  \
                  qw3_metal_begin_commands_concurrent() :                           \
                  qw3_metal_begin_commands())) return 0;                            \
            cb = qw3_metal_command_buffer(&owned);                                  \
            if (!cb) return 0;                                                      \
            profile_prefill_moe_t0 = [NSDate timeIntervalSinceReferenceDate];        \
        }                                                                           \
    } while (0)
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    if (use_mapped_moe) {
        struct {
            uint32_t n_tokens;
            uint32_t n_active;
            uint32_t n_expert;
            uint32_t pair_capacity;
            uint32_t n_ff;
            uint32_t n_embd;
        } map_args = {
            n_tokens, n_active, QW3_METAL_N_EXPERT, n_tokens, n_ff, n_embd
        };
        [enc setComputePipelineState:g_moe_topk_expert_map_pipeline];
        [enc setBytes:&map_args length:sizeof(map_args) atIndex:0];
        [enc setBuffer:obj.routerIds offset:0 atIndex:1];
        [enc setBuffer:obj.moeExpertCounts offset:0 atIndex:2];
        [enc setBuffer:obj.moePairIds offset:0 atIndex:3];
        [enc setBuffer:obj.moeBlockCount offset:0 atIndex:4];
        [enc setBuffer:obj.moeBlockIds offset:0 atIndex:5];
        [enc setBuffer:obj.moeBlockDispatchFF offset:0 atIndex:6];
        [enc setBuffer:obj.moeBlockDispatchEmbd offset:0 atIndex:7];
        [enc dispatchThreads:MTLSizeMake(QW3_METAL_N_EXPERT, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(QW3_METAL_N_EXPERT, 1, 1)];
        if (use_compact_moe_blocks) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        qw3_metal_end_compute_encoder(cb, enc);
        if (!qw3_metal_batch_barrier()) return 0;
        QW3_PROFILE_PREFILL_MOE_STAGE("map");
        enc = qw3_metal_compute_encoder(cb);
    }
    if (use_mapped_gateup_pair) {
        [enc setComputePipelineState:use_mapped_gateup_pair_mpp ?
         g_moe_iq3_s_prefill_pair_mapped_mpp_pipeline :
         g_moe_iq3_s_prefill_pair_mapped_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gate_w offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:up_w offset:(NSUInteger)up_inner atIndex:2];
        [enc setBuffer:obj.prefillX1 offset:0 atIndex:3];
        [enc setBuffer:use_mapped_gateup_pair_mpp ? obj.prefillMoeGate :
         obj.prefillScratch offset:0 atIndex:4];
        [enc setBuffer:kgb offset:0 atIndex:5];
        [enc setBuffer:obj.moeExpertCounts offset:0 atIndex:6];
        [enc setBuffer:obj.moePairIds offset:0 atIndex:7];
        [enc setBuffer:obj.moeBlockIds offset:0 atIndex:8];
        if (use_mapped_gateup_pair_mpp) {
            [enc setBuffer:obj.routerWeights offset:0 atIndex:9];
        }
        [enc setThreadgroupMemoryLength:16384u atIndex:0];
        if (use_compact_moe_blocks) {
            [enc dispatchThreadgroupsWithIndirectBuffer:obj.moeBlockDispatchFF
                                   indirectBufferOffset:0
                                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake((n_tokens + 31u) / 32u,
                                                  (n_ff + 63u) / 64u,
                                                  QW3_METAL_N_EXPERT)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        qw3_metal_end_compute_encoder(cb, enc);
        if (!qw3_metal_batch_barrier()) return 0;
        QW3_PROFILE_PREFILL_MOE_STAGE(use_mapped_gateup_pair_mpp ?
                                      "gate_up_pair_mpp" :
                                      "gate_up_pair");
    } else if (use_mapped_gateup) {
        [enc setComputePipelineState:use_mapped_gateup_mpp ?
         g_moe_iq3_s_prefill_mapped_mpp_pipeline :
         g_moe_iq3_s_prefill_mapped_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gate_w offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:obj.prefillX1 offset:0 atIndex:2];
        [enc setBuffer:obj.prefillMoeGate offset:0 atIndex:3];
        [enc setBuffer:kgb offset:0 atIndex:4];
        [enc setBuffer:obj.moeExpertCounts offset:0 atIndex:5];
        [enc setBuffer:obj.moePairIds offset:0 atIndex:6];
        [enc setBuffer:obj.moeBlockIds offset:0 atIndex:7];
        [enc setThreadgroupMemoryLength:8192u atIndex:0];
        if (use_compact_moe_blocks) {
            [enc dispatchThreadgroupsWithIndirectBuffer:obj.moeBlockDispatchFF
                                   indirectBufferOffset:0
                                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake((n_tokens + 31u) / 32u,
                                                  (n_ff + 63u) / 64u,
                                                  QW3_METAL_N_EXPERT)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        qw3_metal_end_compute_encoder(cb, enc);
        QW3_PROFILE_PREFILL_MOE_STAGE(use_mapped_gateup_mpp ? "gate_mpp" : "gate");

        enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:use_mapped_gateup_mpp ?
         g_moe_iq3_s_prefill_mapped_mpp_pipeline :
         g_moe_iq3_s_prefill_mapped_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:up_w offset:(NSUInteger)up_inner atIndex:1];
        [enc setBuffer:obj.prefillX1 offset:0 atIndex:2];
        [enc setBuffer:obj.prefillMoeUp offset:0 atIndex:3];
        [enc setBuffer:kgb offset:0 atIndex:4];
        [enc setBuffer:obj.moeExpertCounts offset:0 atIndex:5];
        [enc setBuffer:obj.moePairIds offset:0 atIndex:6];
        [enc setBuffer:obj.moeBlockIds offset:0 atIndex:7];
        [enc setThreadgroupMemoryLength:8192u atIndex:0];
        if (use_compact_moe_blocks) {
            [enc dispatchThreadgroupsWithIndirectBuffer:obj.moeBlockDispatchFF
                                   indirectBufferOffset:0
                                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake((n_tokens + 31u) / 32u,
                                                  (n_ff + 63u) / 64u,
                                                  QW3_METAL_N_EXPERT)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        qw3_metal_end_compute_encoder(cb, enc);

        if (!qw3_metal_batch_barrier()) return 0;
        QW3_PROFILE_PREFILL_MOE_STAGE(use_mapped_gateup_mpp ? "up_mpp" : "up");

        enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:use_mapped_mid_f16 ?
         g_moe_swiglu_slots_to_hidden_f16_pipeline :
         use_mapped_mid_f32 ?
         g_moe_swiglu_slots_to_mid_f32_pipeline :
         g_moe_swiglu_slots_to_hidden_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:obj.prefillMoeGate offset:0 atIndex:1];
        [enc setBuffer:obj.prefillMoeUp offset:0 atIndex:2];
        [enc setBuffer:use_mapped_mid_f16 ? obj.prefillMoeMidF16 :
         use_mapped_mid_f32 ? obj.prefillMoeGate : obj.prefillScratch
               offset:0 atIndex:3];
        NSUInteger threads = (use_mapped_mid_f16 ?
                              g_moe_swiglu_slots_to_hidden_f16_pipeline :
                              use_mapped_mid_f32 ?
                              g_moe_swiglu_slots_to_mid_f32_pipeline :
                              g_moe_swiglu_slots_to_hidden_pipeline).maxTotalThreadsPerThreadgroup;
        if (threads > 256) threads = 256;
        if (threads < 32) threads = 32;
        [enc dispatchThreads:MTLSizeMake(n_tokens * n_active * n_ff, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);
        if (!qw3_metal_batch_barrier()) return 0;
        QW3_PROFILE_PREFILL_MOE_STAGE("activation");
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
        QW3_PROFILE_PREFILL_MOE_STAGE("gate_up_legacy");
    }

    if (use_mapped_down) {
        enc = qw3_metal_compute_encoder(cb);
        [enc setComputePipelineState:down_type == 14 ?
         (use_mapped_q6_mpp ?
          g_moe_down_q6_k_prefill_mapped_mpp_pipeline :
          g_moe_down_q6_k_prefill_mapped_pipeline) :
         use_mapped_mid_f32_mpp ?
         g_moe_down_iq4_xs_prefill_mapped_mid_f32_mpp_pipeline :
         use_mapped_mid_f16_mpp ?
         g_moe_down_iq4_xs_prefill_mapped_f16_mpp_pipeline :
         use_mapped_mid_f16 ?
         g_moe_down_iq4_xs_prefill_mapped_f16_pipeline :
         use_mapped_mid_f32 ?
         g_moe_down_iq4_xs_prefill_mapped_mid_f32_pipeline :
         g_moe_down_iq4_xs_prefill_mapped_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:down_w offset:(NSUInteger)down_inner atIndex:1];
        [enc setBuffer:use_mapped_mid_f16 ? obj.prefillMoeMidF16 :
         use_mapped_mid_f32 ? obj.prefillMoeGate : obj.prefillScratch
               offset:0 atIndex:2];
        [enc setBuffer:obj.prefillMoeDown offset:0 atIndex:3];
        [enc setBuffer:obj.moeExpertCounts offset:0 atIndex:4];
        [enc setBuffer:obj.moePairIds offset:0 atIndex:5];
        [enc setBuffer:obj.routerWeights offset:0 atIndex:6];
        [enc setBuffer:obj.moeBlockIds offset:0 atIndex:7];
        [enc setThreadgroupMemoryLength:8192u atIndex:0];
        if (use_compact_moe_blocks) {
            [enc dispatchThreadgroupsWithIndirectBuffer:obj.moeBlockDispatchEmbd
                                   indirectBufferOffset:0
                                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake((n_tokens + 31u) / 32u,
                                                  (n_embd + 63u) / 64u,
                                                  QW3_METAL_N_EXPERT)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        qw3_metal_end_compute_encoder(cb, enc);

        if (!qw3_metal_batch_barrier()) return 0;
        QW3_PROFILE_PREFILL_MOE_STAGE((use_mapped_mid_f32_mpp ||
                                       use_mapped_mid_f16_mpp ||
                                       use_mapped_q6_mpp) ? "down_mpp" : "down");

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
        QW3_PROFILE_PREFILL_MOE_STAGE("reduce");
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
        QW3_PROFILE_PREFILL_MOE_STAGE("down_reduce_legacy");
    }

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal batch sparse MoE prefill command failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
#undef QW3_PROFILE_PREFILL_MOE_STAGE
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

    if (getenv("QW3_METAL_GQA_NORM_ROPE_SPLIT") == NULL) {
        struct {
            uint32_t n_tokens;
            uint32_t n_heads;
            uint32_t head_dim;
            uint32_t rope_dim;
            uint32_t pos0;
            uint32_t in_offset;
            uint32_t out_offset;
            uint32_t gate_offset;
            uint32_t stride;
            float theta;
            float eps;
        } q_fused_args = {
            n_tokens, n_heads, head_dim, rope_dim, pos0, qg_offset,
            q_rope_offset, gate_offset, stride, rope_theta, eps
        };
        id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
        if (!enc) return 0;
        [enc setComputePipelineState:g_gqa_q_norm_gate_rope_batch_pipeline];
        [enc setBytes:&q_fused_args length:sizeof(q_fused_args) atIndex:0];
        [enc setBuffer:obj.prefillScratch offset:0 atIndex:1];
        [enc setBuffer:qw offset:(NSUInteger)q_inner atIndex:2];
        [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
        NSUInteger threads =
            g_gqa_q_norm_gate_rope_batch_pipeline.maxTotalThreadsPerThreadgroup;
        if (threads > 256) threads = 256;
        if (threads < 32) threads = 32;
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens * n_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);

        __typeof__(q_fused_args) k_fused_args = {
            n_tokens, n_kv_heads, head_dim, rope_dim, pos0, k_offset,
            k_rope_offset, 0, stride, rope_theta, eps
        };
        enc = qw3_metal_compute_encoder(cb);
        if (!enc) return 0;
        [enc setComputePipelineState:g_gqa_k_norm_rope_batch_pipeline];
        [enc setBytes:&k_fused_args length:sizeof(k_fused_args) atIndex:0];
        [enc setBuffer:obj.prefillScratch offset:0 atIndex:1];
        [enc setBuffer:kw offset:(NSUInteger)k_inner atIndex:2];
        [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
        threads = g_gqa_k_norm_rope_batch_pipeline.maxTotalThreadsPerThreadgroup;
        if (threads > 256) threads = 256;
        if (threads < 32) threads = 32;
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens * n_kv_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);

        if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
        if (cb.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "qw3: Metal batch fused GQA norm/RoPE command failed: %s\n",
                    [[cb.error localizedDescription] UTF8String]);
            return 0;
        }
        return 1;
    }

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
    [enc setThreadgroupMemoryLength:96u * sizeof(float) atIndex:0];
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
    const uint64_t cache_token_bytes =
        (uint64_t)kv_n * (obj.gqaKvF16 ? sizeof(uint16_t) : sizeof(float));
    const uint64_t cache_layer_bytes = (uint64_t)obj.ctxSize * cache_token_bytes;
    id<MTLBuffer> cache_k = nil;
    id<MTLBuffer> cache_v = nil;
    NSUInteger cache_offset = 0;
    if (!obj.prefillScratch || obj.prefillScratch.length < scratch_bytes ||
        !qw3_metal_gqa_layer_buffers(obj, layer_slot, cache_layer_bytes,
                                     &cache_k, &cache_v, &cache_offset)) {
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
        uint32_t kv_type;
    } args = {
        n_tokens, n_kv_heads, head_dim, pos0, obj.ctxSize,
        k_offset, v_offset, stride, obj.gqaKvF16 ? 1u : 0u
    };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    if (!enc) return 0;
    [enc setComputePipelineState:g_gqa_prefill_write_cache_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:1];
    [enc setBuffer:cache_k offset:cache_offset atIndex:2];
    [enc setBuffer:cache_v offset:cache_offset atIndex:3];
    [enc setBuffer:cache_k offset:cache_offset atIndex:4];
    [enc setBuffer:cache_v offset:cache_offset atIndex:5];
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

static int qw3_metal_session_try_batch_gqa_flash_attn_from_scratch(
    qw3_metal_session *s, uint32_t layer_slot, uint32_t pos0,
    uint32_t n_tokens, uint32_t n_heads, uint32_t n_kv_heads,
    uint32_t head_dim, uint32_t q_offset, uint32_t gate_offset,
    uint32_t out_offset, uint32_t stride) {
    if (!s || !s->obj || n_tokens == 0 || n_heads == 0 ||
        n_kv_heads == 0 || head_dim == 0 || stride == 0 ||
        (n_heads % n_kv_heads) != 0u) {
        return -1;
    }
    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint32_t n_keys = pos0 + n_tokens;
    const uint32_t q_n = n_heads * head_dim;
    const uint32_t kv_n = n_kv_heads * head_dim;
    const uint32_t nqptg = 8u;
    const uint32_t ncpsg = 64u;
    const int has_kvpad = (n_keys % ncpsg) != 0u;
    const int bc_mask = (n_tokens % nqptg) != 0u;
    if (!obj.gqaKvF16 || obj.gqaKvQ8 || head_dim != 256u ||
        n_tokens < 8u || n_keys == 0 ||
        pos0 > obj.ctxSize || n_tokens > obj.ctxSize - pos0 ||
        q_offset > stride || gate_offset > stride || out_offset > stride ||
        q_n > stride - q_offset ||
        q_n > stride - gate_offset ||
        q_n > stride - out_offset) {
        return -1;
    }
    const uint64_t scratch_bytes = (uint64_t)n_tokens * stride * sizeof(float);
    const uint64_t cache_token_bytes = (uint64_t)kv_n * sizeof(uint16_t);
    const uint64_t cache_layer_bytes = (uint64_t)obj.ctxSize * cache_token_bytes;
    id<MTLBuffer> cache_k = nil;
    id<MTLBuffer> cache_v = nil;
    NSUInteger cache_offset = 0;
    if (!obj.prefillScratch || obj.prefillScratch.length < scratch_bytes ||
        !qw3_metal_gqa_layer_buffers(obj, layer_slot, cache_layer_bytes,
                                     &cache_k, &cache_v, &cache_offset)) {
        return 0;
    }
    if (cache_k.length < (uint64_t)cache_offset + (uint64_t)n_keys * cache_token_bytes ||
        cache_v.length < (uint64_t)cache_offset + (uint64_t)n_keys * cache_token_bytes) {
        return 0;
    }
    const int flash_mask_status =
        qw3_metal_session_ensure_flash_attn_buffers(obj, pos0, n_tokens,
                                                    n_keys, n_heads,
                                                    n_kv_heads, head_dim);
    if (!flash_mask_status) {
        return 0;
    }
    const int gpu_causal_mask = flash_mask_status == 2;

    id<MTLComputePipelineState> pad_pipeline =
        has_kvpad ? qw3_metal_gqa_flash_pad_pipeline() : nil;
    id<MTLComputePipelineState> blk_pipeline =
        qw3_metal_gqa_flash_blk_pipeline();
    id<MTLComputePipelineState> attn_pipeline =
        qw3_metal_gqa_flash_attn_pipeline(n_kv_heads * head_dim,
                                          has_kvpad, bc_mask);
    if ((has_kvpad && !pad_pipeline) || !blk_pipeline ||
        !attn_pipeline || !g_gqa_flash_gate_pipeline) return -1;

    qw3_metal_flash_attn_args args = {
        .ne01 = (int32_t)n_tokens,
        .ne02 = (int32_t)n_heads,
        .ne03 = 1,
        .nb01 = (uint64_t)stride * sizeof(float),
        .nb02 = (uint64_t)head_dim * sizeof(float),
        .nb03 = (uint64_t)n_tokens * stride * sizeof(float),
        .ne11 = (int32_t)n_keys,
        .ne_12_2 = (int32_t)n_kv_heads,
        .ne_12_3 = 1,
        .ns10 = (int32_t)(n_kv_heads * head_dim),
        .nb11 = (uint64_t)n_kv_heads * head_dim * sizeof(uint16_t),
        .nb12 = (uint64_t)head_dim * sizeof(uint16_t),
        .nb13 = (uint64_t)n_keys * n_kv_heads * head_dim * sizeof(uint16_t),
        .ns20 = (int32_t)(n_kv_heads * head_dim),
        .nb21 = (uint64_t)n_kv_heads * head_dim * sizeof(uint16_t),
        .nb22 = (uint64_t)head_dim * sizeof(uint16_t),
        .nb23 = (uint64_t)n_keys * n_kv_heads * head_dim * sizeof(uint16_t),
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
        .nb32 = (uint64_t)n_tokens * n_keys * sizeof(uint16_t),
        .nb33 = (uint64_t)n_tokens * n_keys * sizeof(uint16_t),
        .ne1 = (int32_t)n_heads,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    if (gpu_causal_mask) {
        struct {
            uint32_t n_tokens;
            uint32_t n_keys;
            uint32_t pos0;
            uint32_t n_q_blocks;
            uint32_t n_k_blocks;
        } mask_args = {
            n_tokens, n_keys, pos0,
            (n_tokens + nqptg - 1u) / nqptg,
            (n_keys + ncpsg - 1u) / ncpsg
        };
        id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
        if (!enc) return 0;
        [enc setComputePipelineState:g_gqa_flash_causal_mask_pipeline];
        [enc setBytes:&mask_args length:sizeof(mask_args) atIndex:0];
        [enc setBuffer:obj.flashAttnMask offset:0 atIndex:1];
        [enc setBuffer:obj.flashAttnBlock offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(mask_args.n_k_blocks,
                                              mask_args.n_q_blocks,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);
    }
    if (has_kvpad) {
        qw3_metal_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_keys,
            .ne_12_2 = (int32_t)n_kv_heads,
            .ne_12_3 = 1,
            .nb11 = (uint64_t)n_kv_heads * head_dim * sizeof(uint16_t),
            .nb12 = (uint64_t)head_dim * sizeof(uint16_t),
            .nb13 = (uint64_t)n_keys * n_kv_heads * head_dim * sizeof(uint16_t),
            .nb21 = (uint64_t)n_kv_heads * head_dim * sizeof(uint16_t),
            .nb22 = (uint64_t)head_dim * sizeof(uint16_t),
            .nb23 = (uint64_t)n_keys * n_kv_heads * head_dim * sizeof(uint16_t),
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
            .nb32 = (uint64_t)n_tokens * n_keys * sizeof(uint16_t),
            .nb33 = (uint64_t)n_tokens * n_keys * sizeof(uint16_t),
        };
        id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
        if (!enc) return 0;
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:cache_k offset:cache_offset atIndex:1];
        [enc setBuffer:cache_v offset:cache_offset atIndex:2];
        [enc setBuffer:obj.flashAttnMask offset:0 atIndex:3];
        [enc setBuffer:obj.flashAttnPad offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)ncpsg * n_kv_heads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);
    }

    if (!gpu_causal_mask) {
        qw3_metal_flash_attn_blk_args blk_args = {
            .ne01 = (int32_t)n_tokens,
            .ne30 = (int32_t)n_keys,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
            .nb32 = (uint64_t)n_tokens * n_keys * sizeof(uint16_t),
            .nb33 = (uint64_t)n_tokens * n_keys * sizeof(uint16_t),
        };
        id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
        if (!enc) return 0;
        [enc setComputePipelineState:blk_pipeline];
        [enc setBytes:&blk_args length:sizeof(blk_args) atIndex:0];
        [enc setBuffer:obj.flashAttnMask offset:0 atIndex:1];
        [enc setBuffer:obj.flashAttnBlock offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_keys + ncpsg - 1u) / ncpsg,
                                              ((NSUInteger)n_tokens + nqptg - 1u) / nqptg,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);
    }

    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    if (!enc) return 0;
    const NSUInteger padded_v = ((NSUInteger)head_dim + 63u) & ~(NSUInteger)63u;
    const NSUInteger shared_elems = 8u *
        ((NSUInteger)head_dim + 2u * padded_v + 2u * (2u * 64u));
    const NSUInteger shared_bytes =
        ((shared_elems * sizeof(uint16_t)) + 15u) & ~(NSUInteger)15u;
    [enc setComputePipelineState:attn_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.prefillScratch
             offset:(NSUInteger)q_offset * sizeof(float)
            atIndex:1];
    [enc setBuffer:cache_k offset:cache_offset atIndex:2];
    [enc setBuffer:cache_v offset:cache_offset atIndex:3];
    [enc setBuffer:obj.flashAttnMask offset:0 atIndex:4];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:5];
    [enc setBuffer:obj.flashAttnPad offset:0 atIndex:6];
    [enc setBuffer:obj.flashAttnBlock offset:0 atIndex:7];
    [enc setBuffer:obj.flashAttnOut offset:0 atIndex:8];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_tokens + nqptg - 1u) / nqptg,
                                          n_heads, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    qw3_metal_end_compute_encoder(cb, enc);

    struct {
        uint32_t n_tokens;
        uint32_t n_heads;
        uint32_t head_dim;
        uint32_t gate_offset;
        uint32_t out_offset;
        uint32_t stride;
    } gate_args = {
        n_tokens, n_heads, head_dim, gate_offset, out_offset, stride
    };
    enc = qw3_metal_compute_encoder(cb);
    if (!enc) return 0;
    [enc setComputePipelineState:g_gqa_flash_gate_pipeline];
    [enc setBytes:&gate_args length:sizeof(gate_args) atIndex:0];
    [enc setBuffer:obj.flashAttnOut offset:0 atIndex:1];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:2];
    NSUInteger threads = g_gqa_flash_gate_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256u) threads = 256u;
    if (threads < 32u) threads = 32u;
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n_tokens * n_heads * head_dim, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);

    if (!qw3_metal_finish_command_buffer(cb, owned, "operation")) return 0;
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "qw3: Metal FlashAttention GQA prefill command failed: %s\n",
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
    const char *flash_attn_env = getenv("QW3_METAL_GQA_FLASH_ATTN");
    const int flash_attn_disabled =
        getenv("QW3_METAL_GQA_FLASH_ATTN_DISABLE") != NULL ||
        !g_gqa_flash_attn_external_enabled ||
        (flash_attn_env && flash_attn_env[0] &&
         strcmp(flash_attn_env, "0") == 0);
    if (!flash_attn_disabled) {
        int flash_status =
            qw3_metal_session_try_batch_gqa_flash_attn_from_scratch(
                s, layer_slot, pos0, n_tokens, n_heads, n_kv_heads,
                head_dim, q_offset, gate_offset, out_offset, stride);
        if (flash_status > 0) return 1;
        if (flash_status == 0) return 0;
    }
    if (obj.gqaKvQ8 || pos0 > obj.ctxSize ||
        n_tokens > obj.ctxSize - pos0) {
        return 0;
    }
    const uint64_t scratch_bytes = (uint64_t)n_tokens * stride * sizeof(float);
    const uint64_t cache_token_bytes =
        (uint64_t)kv_n * (obj.gqaKvF16 ? sizeof(uint16_t) : sizeof(float));
    const uint64_t cache_layer_bytes = (uint64_t)obj.ctxSize * cache_token_bytes;
    id<MTLBuffer> cache_k = nil;
    id<MTLBuffer> cache_v = nil;
    NSUInteger cache_offset = 0;
    if (!obj.prefillScratch || obj.prefillScratch.length < scratch_bytes ||
        !qw3_metal_gqa_layer_buffers(obj, layer_slot, cache_layer_bytes,
                                     &cache_k, &cache_v, &cache_offset)) {
        return 0;
    }
    const int force_block1 = getenv("QW3_METAL_GQA_ATTEND_BLOCK1") != NULL;
    const int force_block2 = getenv("QW3_METAL_GQA_ATTEND_BLOCK2") != NULL;
    const int force_block4 = getenv("QW3_METAL_GQA_ATTEND_BLOCK4") != NULL;
    const int use_src8 = !force_block1 && !force_block2 && !force_block4 && n_tokens >= 4u;
    const int use_block4 = !use_src8 && !force_block1 && !force_block2 && n_tokens >= 4u;
    const int use_block2 =
        !force_block1 && !use_block4 && n_tokens >= 2u;
    id<MTLComputePipelineState> attend_pipeline = use_src8 ?
        g_gqa_prefill_cached_attend_src8_pipeline : (use_block4 ?
        g_gqa_prefill_cached_attend_block4_pipeline :
        (use_block2 ? g_gqa_prefill_cached_attend_block2_pipeline :
         g_gqa_prefill_cached_attend_inner_pipeline));
    NSUInteger threads = ((NSUInteger)head_dim + 31u) & ~(NSUInteger)31u;
    if (threads < 32u) threads = 32u;
    if (threads > 256u ||
        threads > attend_pipeline.maxTotalThreadsPerThreadgroup) {
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
        uint32_t kv_type;
    } args = {
        n_tokens, n_heads, n_kv_heads, head_dim, pos0, obj.ctxSize,
        q_offset, gate_offset, out_offset, stride, obj.gqaKvF16 ? 1u : 0u
    };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    if (!enc) return 0;
    [enc setComputePipelineState:attend_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:obj.prefillScratch offset:0 atIndex:1];
    [enc setBuffer:cache_k offset:cache_offset atIndex:2];
    [enc setBuffer:cache_v offset:cache_offset atIndex:3];
    [enc setBuffer:cache_k offset:cache_offset atIndex:4];
    [enc setBuffer:cache_v offset:cache_offset atIndex:5];
    [enc setThreadgroupMemoryLength:
        (use_src8 ? 2656u : (use_block4 ? 384u :
         (use_block2 ? 192u : 96u))) * sizeof(float)
                            atIndex:0];
    const NSUInteger query_groups = (use_src8 || use_block4) ?
        ((NSUInteger)n_tokens + 3u) / 4u :
        (use_block2 ? ((NSUInteger)n_tokens + 1u) / 2u :
         (NSUInteger)n_tokens);
    [enc dispatchThreadgroups:MTLSizeMake(query_groups * n_kv_heads, 1, 1)
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
        (uint64_t)(kv_n / 32u) * 34ull :
        (obj.gqaKvF16 ? (uint64_t)kv_n * sizeof(uint16_t) : kv_bytes);
    const uint64_t scratch_needed = (uint64_t)(qg_n + 2u * kv_n) * sizeof(float);
    const uint64_t cache_layer_bytes = (uint64_t)obj.ctxSize * cache_kv_bytes;
    id<MTLBuffer> cache_k = nil;
    id<MTLBuffer> cache_v = nil;
    NSUInteger cache_layer_offset = 0;
    if (!qw3_metal_gqa_layer_buffers(obj, layer_slot, cache_layer_bytes,
                                     &cache_k, &cache_v, &cache_layer_offset)) {
        return 0;
    }
    const uint64_t cache_offset_u64 =
        (uint64_t)cache_layer_offset + (uint64_t)pos * cache_kv_bytes;
    if (cache_offset_u64 > (uint64_t)NSUIntegerMax) return 0;
    const NSUInteger cache_offset = (NSUInteger)cache_offset_u64;
    if (pos >= obj.ctxSize || obj.scratch.length < scratch_needed ||
        obj.gqaTmpQ.length < q_bytes || obj.gqaTokenQ.length < q_bytes ||
        obj.gqaTokenGate.length < q_bytes ||
        obj.gqaTmpK.length < kv_bytes || obj.gqaTokenK.length < kv_bytes ||
        obj.gqaTokenV.length < kv_bytes ||
        cache_k.length < cache_offset_u64 + cache_kv_bytes ||
        cache_v.length < cache_offset_u64 + cache_kv_bytes) {
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
    if (!obj.gqaKvQ8 && !obj.gqaKvF16) {
        [blit copyFromBuffer:obj.gqaTokenK sourceOffset:0
                    toBuffer:cache_k destinationOffset:cache_offset
                        size:(NSUInteger)kv_bytes];
        [blit copyFromBuffer:obj.gqaTokenV sourceOffset:0
                    toBuffer:cache_v destinationOffset:cache_offset
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

    if (obj.gqaKvF16) {
        struct {
            uint32_t n;
        } store_args = { kv_n };
        enc = qw3_metal_compute_encoder(cb);
        if (!enc) return 0;
        [enc setComputePipelineState:g_gqa_store_token_cache_f16_pipeline];
        [enc setBytes:&store_args length:sizeof(store_args) atIndex:0];
        [enc setBuffer:obj.gqaTokenK offset:0 atIndex:1];
        [enc setBuffer:obj.gqaTokenV offset:0 atIndex:2];
        [enc setBuffer:cache_k offset:cache_offset atIndex:3];
        [enc setBuffer:cache_v offset:cache_offset atIndex:4];
        NSUInteger threads = g_gqa_store_token_cache_f16_pipeline.maxTotalThreadsPerThreadgroup;
        if (threads > 256) threads = 256;
        [enc dispatchThreads:MTLSizeMake(kv_n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        qw3_metal_end_compute_encoder(cb, enc);
    } else if (obj.gqaKvQ8) {
        struct {
            uint32_t n;
        } quant_args = { kv_n };
        enc = qw3_metal_compute_encoder(cb);
        if (!enc) return 0;
        [enc setComputePipelineState:g_gqa_kv_quant_q8_pipeline];
        [enc setBytes:&quant_args length:sizeof(quant_args) atIndex:0];
        [enc setBuffer:obj.gqaTokenK offset:0 atIndex:1];
        [enc setBuffer:obj.gqaTokenV offset:0 atIndex:2];
        [enc setBuffer:cache_k offset:cache_offset atIndex:3];
        [enc setBuffer:cache_v offset:cache_offset atIndex:4];
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
        (uint64_t)(kv_n / 32u) * 34ull :
        (obj.gqaKvF16 ? (uint64_t)kv_n * sizeof(uint16_t) : kv_bytes);
    const uint64_t cache_layer_bytes = (uint64_t)obj.ctxSize * cache_kv_bytes;
    const uint64_t cache_bytes = (uint64_t)n_ctx * cache_kv_bytes;
    id<MTLBuffer> cache_k = nil;
    id<MTLBuffer> cache_v = nil;
    NSUInteger cache_offset = 0;
    if (!obj.gqaTokenQ || !obj.gqaTokenGate ||
        !qw3_metal_gqa_layer_buffers(obj, layer_slot, cache_layer_bytes,
                                     &cache_k, &cache_v, &cache_offset) ||
        !obj.inner || obj.gqaTokenQ.length < inner_bytes ||
        obj.gqaTokenGate.length < inner_bytes ||
        cache_k.length < (uint64_t)cache_offset + cache_bytes ||
        cache_v.length < (uint64_t)cache_offset + cache_bytes ||
        obj.inner.length < inner_bytes) {
        return 0;
    }

    struct {
        uint32_t n_ctx;
        uint32_t n_heads;
        uint32_t n_kv_heads;
        uint32_t head_dim;
        uint32_t kv_type;
    } args = { n_ctx, n_heads, n_kv_heads, head_dim, obj.gqaKvF16 ? 1u : 0u };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    NSUInteger threads = ((NSUInteger)head_dim + 31u) & ~(NSUInteger)31u;
    if (threads < 32u) threads = 32u;
    if (threads > 256u) {
        return 0;
    }
    const BOOL split_attn = !obj.gqaKvQ8 && obj.gqaSplitAttn && n_ctx >= 1024u;
    const BOOL split_q8 = obj.gqaKvQ8 && obj.gqaSplitQ8 &&
        (n_ctx >= 256u || getenv("QW3_METAL_Q8_SPLIT_FORCE") != NULL);
    if (split_attn) {
        uint32_t active_max_splits = obj.gqaMaxAttnSplits;
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
            threads > g_gqa_attend_n_split_partial_pipeline.maxTotalThreadsPerThreadgroup ||
            threads > g_gqa_attend_n_split_reduce_pipeline.maxTotalThreadsPerThreadgroup) {
            return 0;
        }
        struct {
            uint32_t n_ctx;
            uint32_t n_heads;
            uint32_t n_kv_heads;
            uint32_t head_dim;
            uint32_t n_splits;
            uint32_t kv_type;
        } split_args = {
            n_ctx, n_heads, n_kv_heads, head_dim, n_splits,
            obj.gqaKvF16 ? 1u : 0u
        };
        id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
        if (!enc) return 0;
        [enc setComputePipelineState:g_gqa_attend_n_split_partial_pipeline];
        [enc setBytes:&split_args length:sizeof(split_args) atIndex:0];
        [enc setBuffer:obj.gqaTokenQ offset:0 atIndex:1];
        [enc setBuffer:cache_k offset:cache_offset atIndex:2];
        [enc setBuffer:cache_v offset:cache_offset atIndex:3];
        [enc setBuffer:obj.gqaAttnPartial offset:0 atIndex:4];
        [enc setBuffer:cache_k offset:cache_offset atIndex:5];
        [enc setBuffer:cache_v offset:cache_offset atIndex:6];
        [enc setThreadgroupMemoryLength:64u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_kv_heads * n_splits, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        qw3_metal_end_compute_encoder(cb, enc);

        enc = qw3_metal_compute_encoder(cb);
        if (!enc) return 0;
        [enc setComputePipelineState:g_gqa_attend_n_split_reduce_pipeline];
        [enc setBytes:&split_args length:sizeof(split_args) atIndex:0];
        [enc setBuffer:obj.gqaTokenGate offset:0 atIndex:1];
        [enc setBuffer:obj.gqaAttnPartial offset:0 atIndex:2];
        [enc setBuffer:obj.inner offset:0 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        qw3_metal_end_compute_encoder(cb, enc);
    } else if (split_q8) {
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
        [enc setBuffer:cache_k offset:cache_offset atIndex:2];
        [enc setBuffer:cache_v offset:cache_offset atIndex:3];
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
        [enc setBuffer:cache_k offset:cache_offset atIndex:3];
        [enc setBuffer:cache_v offset:cache_offset atIndex:4];
        [enc setBuffer:obj.inner offset:0 atIndex:5];
        if (!obj.gqaKvQ8) {
            [enc setBuffer:cache_k offset:cache_offset atIndex:6];
            [enc setBuffer:cache_v offset:cache_offset atIndex:7];
        }
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

    const char *name = g_device.name ? [g_device.name UTF8String] : "unknown Metal device";
    snprintf(g_device_name, sizeof(g_device_name), "%s", name);
    fprintf(stderr, "qw3: Metal device %s\n", qw3_metal_device_name());
    qw3_metal_detect_metal4_features();

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

    g_initialized = 1;
    if (g_metal4_tensor_api_enabled) {
        fprintf(stderr, "qw3: Metal 4 tensor API available for Tensor kernels\n");
    }
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
    g_matmul_q8_0_nax_pipeline = nil;
    g_matmul_q8_0_nax_n64_pipeline = nil;
    g_matmul_q8_0_nax_n128_pipeline = nil;
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
    g_moe_iq3_s_prefill_mapped_mpp_pipeline = nil;
    g_moe_iq3_s_prefill_pair_mapped_pipeline = nil;
    g_moe_iq3_s_prefill_pair_mapped_mpp_pipeline = nil;
    g_moe_swiglu_slots_to_hidden_pipeline = nil;
    g_moe_swiglu_slots_to_mid_f32_pipeline = nil;
    g_moe_swiglu_slots_to_hidden_f16_pipeline = nil;
    g_moe_down_iq4_xs_prefill_mapped_pipeline = nil;
    g_moe_down_iq4_xs_prefill_mapped_mid_f32_pipeline = nil;
    g_moe_down_iq4_xs_prefill_mapped_mid_f32_mpp_pipeline = nil;
    g_moe_down_iq4_xs_prefill_mapped_f16_pipeline = nil;
    g_moe_down_iq4_xs_prefill_mapped_f16_mpp_pipeline = nil;
    g_moe_down_q6_k_prefill_mapped_pipeline = nil;
    g_moe_down_q6_k_prefill_mapped_mpp_pipeline = nil;
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
    g_gqa_q_norm_gate_rope_batch_pipeline = nil;
    g_gqa_k_norm_rope_batch_pipeline = nil;
    g_rope_heads_pipeline = nil;
    g_rope_heads_batch_pipeline = nil;
    g_gqa_single_token_inner_pipeline = nil;
    g_gqa_attend2_inner_pipeline = nil;
    g_gqa_attend_n_inner_pipeline = nil;
    g_gqa_prefill_attend_inner_pipeline = nil;
    g_gqa_prefill_write_cache_pipeline = nil;
    g_gqa_prefill_cached_attend_inner_pipeline = nil;
    g_gqa_prefill_cached_attend_block2_pipeline = nil;
    g_gqa_prefill_cached_attend_block4_pipeline = nil;
    g_gqa_prefill_cached_attend_src8_pipeline = nil;
    g_gqa_flash_gate_pipeline = nil;
    g_gqa_flash_causal_mask_pipeline = nil;
    g_gqa_flash_pad_pipeline = nil;
    g_gqa_flash_blk_pipeline = nil;
    g_gqa_flash_attn_pipeline = nil;
    g_gqa_store_token_cache_f16_pipeline = nil;
    g_gqa_kv_quant_q8_pipeline = nil;
    g_gqa_attend_n_q8_inner_pipeline = nil;
    g_gqa_attend_n_split_partial_pipeline = nil;
    g_gqa_attend_n_split_reduce_pipeline = nil;
    g_gqa_attend_n_q8_split_partial_pipeline = nil;
    g_gqa_attend_n_q8_split_reduce_pipeline = nil;
    g_deltanet_recur_zero_pipeline = nil;
    g_deltanet_recur_pipeline = nil;
    g_deltanet_recur_scratch_gates_pipeline = nil;
    g_deltanet_prepare_scratch_gates_pipeline = nil;
    g_deltanet_recur_scratch_gates_tiled_pipeline = nil;
    g_deltanet_fused_gdn_scratch_pipeline = nil;
    g_deltanet_batch_fused_gdn_pipeline = nil;
    g_deltanet_batch_recur_tiled_pipeline = nil;
    g_deltanet_batch_recur_tiled2_pipeline = nil;
    g_deltanet_batch_recur_tiled4_pipeline = nil;
    g_deltanet_batch_gated_rmsnorm_pipeline = nil;
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
    g_metal4_runtime_available = 0;
    g_metal4_family_supported = 0;
    g_metal4_queue_supported = 0;
    g_metal4_m5_neural_accelerators_hint = 0;
    g_metal4_tensor_api_enabled = 0;
    g_metal4_tensor_api_compile_supported = 0;
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
        uint32_t kv_type;
    } args = { n_ctx, n_heads, n_kv_heads, head_dim, 0u };

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
    [enc setBuffer:kb offset:0 atIndex:6];
    [enc setBuffer:vb offset:0 atIndex:7];
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
