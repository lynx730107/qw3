#ifndef QW3_METAL_H
#define QW3_METAL_H

#include <stdint.h>

/* Narrow Objective-C/Metal boundary.
 *
 * qw3.c owns model semantics and scheduling.  This bridge only owns Metal
 * device setup and shared GGUF model mapping until real Qwen kernels are added.
 */

int qw3_metal_init(void);
void qw3_metal_cleanup(void);
int qw3_metal_begin_commands(void);
int qw3_metal_begin_commands_concurrent(void);
int qw3_metal_batch_barrier(void);
int qw3_metal_flush_commands(void);
int qw3_metal_commit_commands(void);
int qw3_metal_end_commands(void);
int qw3_metal_synchronize(void);

/* SSD streaming */
void qw3_metal_set_ssd_streaming(int enabled);
void qw3_metal_set_streaming_expert_cache_budget(uint32_t experts);
int qw3_metal_stream_expert_cache_reset_route_hotness(void);
int qw3_metal_stream_expert_ensure_loaded(
    uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
    uint32_t gate_type, uint32_t up_type, uint32_t down_type,
    uint32_t expert, uint32_t n_in, uint32_t n_ff,
    uint32_t layer, uint32_t priority);
uint64_t qw3_gpu_recommended_working_set_size(void);

typedef struct qw3_metal_session qw3_metal_session;

typedef struct {
  uint64_t total_bytes;
  uint64_t gqa_kv_bytes;
  uint64_t deltanet_state_bytes;
  uint64_t conv_state_bytes;
  uint64_t logits_bytes;
  uint64_t scratch_bytes;
} qw3_metal_session_info;

qw3_metal_session *qw3_metal_session_create(uint32_t ctx_size,
                                            uint32_t vocab_size);
void qw3_metal_session_free(qw3_metal_session *s);
int qw3_metal_session_clear(qw3_metal_session *s);
qw3_metal_session_info qw3_metal_session_get_info(qw3_metal_session *s);
int qw3_metal_session_embed_q8_0(qw3_metal_session *s, uint64_t tensor_offset,
                                 uint32_t token, uint32_t n_embd, float *out);
int qw3_metal_session_batch_embed_q8_0(qw3_metal_session *s,
                                       uint64_t tensor_offset,
                                       const uint32_t *tokens,
                                       uint32_t n_tokens,
                                       uint32_t n_embd);
int qw3_metal_session_batch_rmsnorm_weight_f32_x0_inplace(
    qw3_metal_session *s, uint64_t weight_offset, uint32_t n_tokens,
    uint32_t n, float eps);
int qw3_metal_session_batch_rmsnorm_weight_f32_x0_to_x1(
    qw3_metal_session *s, uint64_t weight_offset, uint32_t n_tokens,
    uint32_t n, float eps);
int qw3_metal_session_batch_matmul_q8_0_x0_to_x1(qw3_metal_session *s,
                                                 uint64_t tensor_offset,
                                                 uint32_t n_tokens,
                                                 uint32_t n_in,
                                                 uint32_t n_out);
int qw3_metal_session_batch_matmul_q8_0_x1_to_scratch(
    qw3_metal_session *s, uint64_t tensor_offset, uint32_t n_tokens,
    uint32_t n_in, uint32_t n_out, uint32_t out_offset,
    uint32_t out_stride);
int qw3_metal_session_batch_matmul_q8_0_scratch_to_scratch(
    qw3_metal_session *s, uint64_t tensor_offset, uint32_t n_tokens,
    uint32_t n_in, uint32_t n_out, uint32_t in_offset,
    uint32_t out_offset, uint32_t stride);
int qw3_metal_session_batch_matmul_f32_x0_to_x1(qw3_metal_session *s,
                                                uint64_t tensor_offset,
                                                uint32_t n_tokens,
                                                uint32_t n_in,
                                                uint32_t n_out);
int qw3_metal_session_batch_matmul_f32_x1_to_scratch(
    qw3_metal_session *s, uint64_t tensor_offset, uint32_t n_tokens,
    uint32_t n_in, uint32_t n_out, uint32_t out_offset,
    uint32_t out_stride);
int qw3_metal_session_batch_matmul_f32_pair_x0_to_x1(
    qw3_metal_session *s, uint64_t tensor_a_offset, uint64_t tensor_b_offset,
    uint32_t n_tokens, uint32_t n_in, uint32_t n_out,
    uint32_t out_a_offset, uint32_t out_b_offset, uint32_t out_stride);
int qw3_metal_session_batch_matmul_f32_pair_x1_to_scratch(
    qw3_metal_session *s, uint64_t tensor_a_offset, uint64_t tensor_b_offset,
    uint32_t n_tokens, uint32_t n_in, uint32_t n_out,
    uint32_t out_a_offset, uint32_t out_b_offset, uint32_t out_stride);
int qw3_metal_session_batch_conv1d_step_from_scratch(
    qw3_metal_session *s, uint64_t weight_offset, uint32_t layer_slot,
    uint32_t n_tokens, uint32_t n_channels, uint32_t qkv_offset,
    uint32_t conv_offset, uint32_t stride);
int qw3_metal_session_batch_l2norm_qk_from_scratch(
    qw3_metal_session *s, uint32_t n_tokens, uint32_t conv_offset,
    uint32_t stride, uint32_t n_qk_heads, uint32_t head_dim, float eps);
int qw3_metal_session_batch_deltanet_fused_gdn_from_scratch(
    qw3_metal_session *s, uint64_t dt_bias_offset, uint64_t a_offset,
    uint64_t norm_weight_offset, uint32_t layer_slot, uint32_t n_tokens,
    uint32_t conv_offset, uint32_t z_offset, uint32_t alpha_offset,
    uint32_t beta_offset, uint32_t inner_offset, uint32_t stride,
    uint32_t q_heads, uint32_t v_heads, uint32_t head_dim, float eps);
int qw3_metal_session_batch_residual_rmsnorm_update_x0_from_scratch(
    qw3_metal_session *s, uint64_t weight_offset, uint32_t n_tokens,
    uint32_t n, uint32_t residual_offset, uint32_t residual_stride,
    float eps);
int qw3_metal_session_batch_gqa_norm_rope_from_scratch(
    qw3_metal_session *s, uint64_t q_norm_weight_offset,
    uint64_t k_norm_weight_offset, uint32_t n_tokens, uint32_t n_heads,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t rope_dim,
    uint32_t pos0, float rope_theta, float eps, uint32_t qg_offset,
    uint32_t k_offset, uint32_t q_tmp_offset, uint32_t k_tmp_offset,
    uint32_t q_rope_offset, uint32_t k_rope_offset, uint32_t gate_offset,
    uint32_t stride);
int qw3_metal_session_batch_gqa_causal_attn_from_scratch(
    qw3_metal_session *s, uint32_t n_tokens, uint32_t n_heads,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t q_offset,
    uint32_t gate_offset, uint32_t k_offset, uint32_t v_offset,
    uint32_t out_offset, uint32_t stride);
int qw3_metal_session_batch_gqa_write_cache_from_scratch(
    qw3_metal_session *s, uint32_t layer_slot, uint32_t pos0,
    uint32_t n_tokens, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t k_offset, uint32_t v_offset, uint32_t stride);
int qw3_metal_session_batch_gqa_cached_attn_from_scratch(
    qw3_metal_session *s, uint32_t layer_slot, uint32_t pos0,
    uint32_t n_tokens, uint32_t n_heads, uint32_t n_kv_heads,
    uint32_t head_dim, uint32_t q_offset, uint32_t gate_offset,
    uint32_t out_offset, uint32_t stride);
int qw3_metal_session_batch_router_topk_from_scratch(
    qw3_metal_session *s, uint32_t n_tokens, uint32_t router_offset,
    uint32_t stride, uint32_t n_router, uint32_t n_top,
    int *ids_out, float *weights_out);
int qw3_metal_session_batch_sparse_moe_topk_from_router_scratch(
    qw3_metal_session *s, uint64_t gate_offset, uint64_t up_offset,
    uint64_t down_offset, uint32_t down_type, uint32_t n_tokens,
    uint32_t n_active, uint32_t n_embd, uint32_t n_ff,
    uint32_t router_offset, uint32_t hidden_offset, uint32_t stride);
int qw3_metal_session_batch_silu_mul_scratch_to_scratch(
    qw3_metal_session *s, uint32_t n_tokens, uint32_t n,
    uint32_t a_offset, uint32_t b_offset, uint32_t out_offset,
    uint32_t stride);
int qw3_metal_session_batch_sigmoid_scale_scratch_add_x0(
    qw3_metal_session *s, uint32_t n_tokens, uint32_t n,
    uint32_t src_offset, uint32_t scalar_offset, uint32_t stride);
int qw3_metal_session_read_batch_x0(qw3_metal_session *s, float *out,
                                    uint32_t n_tokens, uint32_t n_out);
int qw3_metal_session_read_batch_x1(qw3_metal_session *s, float *out,
                                    uint32_t n_tokens, uint32_t n_out);
int qw3_metal_session_read_batch_scratch(qw3_metal_session *s, float *out,
                                         uint32_t n_tokens,
                                         uint32_t n_out);
int qw3_metal_session_read_conv_state(qw3_metal_session *s,
                                      uint32_t layer_slot,
                                      uint32_t n_channels, float *out);
int qw3_metal_session_read_deltanet_state(qw3_metal_session *s,
                                          uint32_t layer_slot,
                                          uint32_t v_heads,
                                          uint32_t head_dim, float *out);
int qw3_metal_session_copy_batch_x0_to_x0(qw3_metal_session *s,
                                          uint32_t row, uint32_t n);
int qw3_metal_session_write_x0(qw3_metal_session *s, const float *x,
                               uint32_t n);
int qw3_metal_session_read_x0(qw3_metal_session *s, float *out, uint32_t n);
int qw3_metal_session_add_moe_to_x0(qw3_metal_session *s, const float *moe,
                                    uint32_t n);
int qw3_metal_session_silu_mul_scratch_to_inner(qw3_metal_session *s,
                                                uint32_t a_offset,
                                                uint32_t b_offset, uint32_t n);
int qw3_metal_session_scale_x1_by_scratch_scalar_add_x0(qw3_metal_session *s,
                                                        uint32_t scalar_offset,
                                                        uint32_t n);
int qw3_metal_session_scale_x1_add_x0(qw3_metal_session *s, float scale,
                                      uint32_t n);
int qw3_metal_session_scale_scratch_add_x0(qw3_metal_session *s,
                                           uint32_t scratch_offset, float scale,
                                           uint32_t n);
int qw3_metal_session_rmsnorm_weight_f32(qw3_metal_session *s,
                                         uint64_t weight_offset, uint32_t n,
                                         float eps, float *out);
int qw3_metal_session_matvec_q8_0_x1(qw3_metal_session *s,
                                     uint64_t tensor_offset, uint32_t n_in,
                                     uint32_t n_out, float *out);
int qw3_metal_session_matvec_q8_0_x1_to_scratch(qw3_metal_session *s,
                                                uint64_t tensor_offset,
                                                uint32_t n_in, uint32_t n_out,
                                                uint32_t out_offset,
                                                float *out);
int qw3_metal_session_matvec_q8_0_pair_x1_to_scratch(
    qw3_metal_session *s, uint64_t tensor_a_offset, uint64_t tensor_b_offset,
    uint32_t n_in, uint32_t n_out, uint32_t out_a_offset,
    uint32_t out_b_offset);
int qw3_metal_session_matvec_q8_0_pair_silu_x1_to_inner(
    qw3_metal_session *s, uint64_t tensor_a_offset, uint64_t tensor_b_offset,
    uint32_t n_in, uint32_t n_out);
int qw3_metal_session_shared_gate_up_silu_x1_to_inner(
    qw3_metal_session *s, uint64_t tensor_a_offset, uint64_t tensor_b_offset,
    uint64_t scalar_weight_offset, uint32_t n_in, uint32_t n_out,
    uint32_t scalar_offset);
int qw3_metal_session_conv1d_zero_from_scratch(qw3_metal_session *s,
                                               uint64_t weight_offset,
                                               uint32_t n_channels, float *out);
int qw3_metal_session_conv1d_step_from_scratch(qw3_metal_session *s,
                                               uint64_t weight_offset,
                                               uint32_t layer_slot,
                                               uint32_t n_channels, float *out,
                                               float *state_out);
int qw3_metal_session_l2norm_qk_from_conv(qw3_metal_session *s,
                                          uint32_t n_heads, uint32_t head_dim,
                                          float eps, float *q_out,
                                          float *k_out);
int qw3_metal_session_matvec_f32_x1_to_scratch(qw3_metal_session *s,
                                               uint64_t tensor_offset,
                                               uint32_t n_in, uint32_t n_out,
                                               uint32_t out_offset, float *out);
int qw3_metal_session_matvec_f32_pair_x1_to_scratch(
    qw3_metal_session *s, uint64_t tensor_a_offset, uint64_t tensor_b_offset,
    uint32_t n_in, uint32_t n_out, uint32_t out_a_offset,
    uint32_t out_b_offset);
int qw3_metal_session_router_topk_from_scratch(qw3_metal_session *s,
                                               uint32_t router_offset,
                                               uint32_t n_router,
                                               uint32_t n_top,
                                               int *ids_out,
                                               float *weights_out);
int qw3_metal_session_deltanet_recur_zero_from_buffers(
    qw3_metal_session *s, const float *beta, uint32_t q_heads, uint32_t v_heads,
    uint32_t head_dim, float *state_out, float *core_out);
int qw3_metal_session_deltanet_recur_from_buffers(
    qw3_metal_session *s, const float *beta, const float *gamma,
    uint32_t layer_slot, uint32_t q_heads, uint32_t v_heads, uint32_t head_dim,
    float *state_out, float *core_out);
int qw3_metal_session_deltanet_recur_from_scratch_gates(
    qw3_metal_session *s, uint64_t dt_bias_offset, uint64_t a_offset,
    uint32_t alpha_offset, uint32_t beta_offset, uint32_t layer_slot,
    uint32_t q_heads, uint32_t v_heads, uint32_t head_dim, float *state_out,
    float *core_out);
int qw3_metal_session_deltanet_fused_gdn_from_scratch(
    qw3_metal_session *s, uint64_t dt_bias_offset, uint64_t a_offset,
    uint64_t norm_weight_offset, uint32_t z_offset, uint32_t alpha_offset,
    uint32_t beta_offset, uint32_t layer_slot, uint32_t q_heads,
    uint32_t v_heads, uint32_t head_dim, float eps);
int qw3_metal_session_deltanet_tiled_gdn_from_scratch(
    qw3_metal_session *s, uint64_t dt_bias_offset, uint64_t a_offset,
    uint64_t norm_weight_offset, uint32_t z_offset, uint32_t alpha_offset,
    uint32_t beta_offset, uint32_t layer_slot, uint32_t q_heads,
    uint32_t v_heads, uint32_t head_dim, float eps);
int qw3_metal_session_deltanet_gated_rmsnorm_from_buffers(
    qw3_metal_session *s, uint64_t norm_weight_offset, uint32_t z_offset,
    uint32_t v_heads, uint32_t head_dim, float eps, float *out);
int qw3_metal_session_matvec_q8_0_inner_to_x1(qw3_metal_session *s,
                                              uint64_t tensor_offset,
                                              uint32_t n_in, uint32_t n_out,
                                              float *out);
int qw3_metal_session_matvec_q8_0_inner_scale_add_x0(qw3_metal_session *s,
                                                     uint64_t tensor_offset,
                                                     uint32_t n_in,
                                                     uint32_t n_out,
                                                     uint32_t scalar_offset);
int qw3_metal_session_matvec_iq3_s_expert_x1_to_scratch(
    qw3_metal_session *s, uint64_t tensor_offset, uint32_t expert,
    uint32_t n_in, uint32_t n_out, uint32_t out_offset);
int qw3_metal_session_matvec_iq4_xs_expert_inner_to_x1(qw3_metal_session *s,
                                                       uint64_t tensor_offset,
                                                       uint32_t expert,
                                                       uint32_t n_in,
                                                       uint32_t n_out);
int qw3_metal_session_matvec_q6_k_expert_inner_to_x1(qw3_metal_session *s,
                                                     uint64_t tensor_offset,
                                                     uint32_t expert,
                                                     uint32_t n_in,
                                                     uint32_t n_out);
int qw3_metal_session_matvec_iq4_xs_expert_inner_to_scratch(
    qw3_metal_session *s, uint64_t tensor_offset, uint32_t expert,
    uint32_t n_in, uint32_t n_out, uint32_t out_offset);
int qw3_metal_session_matvec_q6_k_expert_inner_to_scratch(
    qw3_metal_session *s, uint64_t tensor_offset, uint32_t expert,
    uint32_t n_in, uint32_t n_out, uint32_t out_offset);
int qw3_metal_session_sparse_moe_topk(qw3_metal_session *s,
                                      uint64_t gate_offset,
                                      uint64_t up_offset,
                                      uint64_t down_offset,
                                      uint32_t down_type,
                                      const int *ids,
                                      const float *weights,
                                      uint32_t n_active,
                                      uint32_t n_embd,
                                      uint32_t n_ff);
int qw3_metal_session_sparse_moe_topk_from_router_scratch(qw3_metal_session *s,
                                                          uint64_t gate_offset,
                                                          uint64_t up_offset,
                                                          uint64_t down_offset,
                                                          uint32_t down_type,
                                                          uint32_t layer,
                                                          uint32_t n_active,
                                                          uint32_t n_embd,
                                                          uint32_t n_ff);
int qw3_metal_session_matvec_q8_0_x1_to_logits(qw3_metal_session *s,
                                               uint64_t tensor_offset,
                                               uint32_t n_in, uint32_t n_out,
                                               float *out);
int qw3_metal_session_matvec_q6_k_x1_to_logits(qw3_metal_session *s,
                                               uint64_t tensor_offset,
                                               uint32_t n_in, uint32_t n_out,
                                               float *out);
int qw3_metal_session_argmax_logits(qw3_metal_session *s, uint32_t n,
                                    uint32_t *idx_out, float *val_out);
int qw3_metal_session_residual_rmsnorm_x0_x1(qw3_metal_session *s,
                                             uint64_t weight_offset, uint32_t n,
                                             float eps, float *out);
int qw3_metal_session_residual_rmsnorm_update_x0_x1(qw3_metal_session *s,
                                                    uint64_t weight_offset,
                                                    uint32_t n, float eps,
                                                    float *out);
int qw3_metal_session_gqa_project_cache(
    qw3_metal_session *s, uint64_t q_weight_offset, uint64_t k_weight_offset,
    uint64_t v_weight_offset, uint64_t q_norm_weight_offset,
    uint64_t k_norm_weight_offset, uint32_t qg_n, uint32_t q_n, uint32_t kv_n,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim, uint32_t rope_dim,
    uint32_t layer_slot, uint32_t pos, float rope_theta, float eps,
    float *q_out, float *k_out, float *v_out, float *gate_out);
int qw3_metal_session_gqa_single_attn_out(qw3_metal_session *s,
                                          uint64_t out_weight_offset,
                                          uint32_t n_heads, uint32_t n_kv_heads,
                                          uint32_t head_dim, uint32_t n_embd,
                                          float *out);
int qw3_metal_session_gqa_cached_attn_out(qw3_metal_session *s,
                                          uint64_t out_weight_offset,
                                          uint32_t n_ctx, uint32_t layer_slot,
                                          uint32_t n_heads, uint32_t n_kv_heads,
                                          uint32_t head_dim, uint32_t n_embd,
                                          float *out);

int qw3_metal_set_model_map_range(const void *model_map, uint64_t model_size,
                                  uint64_t map_offset, uint64_t map_size);
int qw3_metal_set_model_map_spans(const void *model_map, uint64_t model_size,
                                  const uint64_t *offsets,
                                  const uint64_t *sizes,
                                  uint32_t count);
const char *qw3_metal_device_name(void);
int qw3_metal_rmsnorm_plain(const float *x, float *out, uint32_t n, float eps);
int qw3_metal_rmsnorm_weight_f32(const float *x, uint64_t weight_offset,
                                 float *out, uint32_t n, float eps);
int qw3_metal_embed_q8_0(uint64_t tensor_offset, uint32_t token,
                         uint32_t n_embd, float *out);
int qw3_metal_matvec_q8_0(uint64_t tensor_offset, const float *x, uint32_t n_in,
                          uint32_t n_out, float *out);
int qw3_metal_matvec_q6_k(uint64_t tensor_offset, const float *x, uint32_t n_in,
                          uint32_t n_out, float *out);
int qw3_metal_matvec_iq4_xs_expert(uint64_t tensor_offset, uint32_t expert,
                                   const float *x, uint32_t n_in,
                                   uint32_t n_out, float *out);
int qw3_metal_matvec_q6_k_expert(uint64_t tensor_offset, uint32_t expert,
                                 const float *x, uint32_t n_in, uint32_t n_out,
                                 float *out);
int qw3_metal_matvec_iq3_s_expert(uint64_t tensor_offset, uint32_t expert,
                                  const float *x, uint32_t n_in, uint32_t n_out,
                                  float *out);
int qw3_metal_matvec_f32(uint64_t tensor_offset, const float *x, uint32_t n_in,
                         uint32_t n_out, float *out);
int qw3_metal_deltanet_conv1d_zero(uint64_t weight_offset, const float *qkv,
                                   uint32_t n_channels, float *out);
int qw3_metal_deltanet_conv1d_step(uint64_t weight_offset, const float *qkv,
                                   const float *state_in, uint32_t n_channels,
                                   float *out, float *state_out);
int qw3_metal_l2norm_heads(const float *x, uint32_t n_heads, uint32_t head_dim,
                           float eps, float *out);
int qw3_metal_rope_heads(const float *x, uint32_t n_heads, uint32_t head_dim,
                         uint32_t rope_dim, int32_t pos, float theta,
                         float *out);
int qw3_metal_gqa_single_token_inner(const float *gate, const float *v,
                                     uint32_t n_heads, uint32_t n_kv_heads,
                                     uint32_t head_dim, float *out);
int qw3_metal_gqa_attend2_inner(const float *q, const float *gate,
                                const float *k_cache, const float *v_cache,
                                uint32_t n_heads, uint32_t n_kv_heads,
                                uint32_t head_dim, float *out);
int qw3_metal_gqa_attend_n_inner(const float *q, const float *gate,
                                 const float *k_cache, const float *v_cache,
                                 uint32_t n_ctx, uint32_t n_heads,
                                 uint32_t n_kv_heads, uint32_t head_dim,
                                 float *out);
int qw3_metal_deltanet_recur_zero(const float *q, const float *k,
                                  const float *v, const float *beta,
                                  uint32_t q_heads, uint32_t v_heads,
                                  uint32_t head_dim, float *state_out,
                                  float *core_out);
int qw3_metal_deltanet_recur(const float *state_in, const float *q,
                             const float *k, const float *v, const float *beta,
                             const float *gamma, uint32_t q_heads,
                             uint32_t v_heads, uint32_t head_dim,
                             float *state_out, float *core_out);
int qw3_metal_deltanet_gated_rmsnorm(uint64_t norm_weight_offset,
                                     const float *core, const float *z,
                                     uint32_t v_heads, uint32_t head_dim,
                                     float eps, float *out);
int qw3_metal_residual_rmsnorm_weight_f32(const float *x, const float *residual,
                                          uint64_t weight_offset, float *out,
                                          uint32_t n, float eps);
int qw3_metal_silu_mul(const float *a, const float *b, uint32_t n, float *out);
int qw3_metal_scale(const float *x, uint32_t n, float scale, float *out);
int qw3_metal_argmax(const float *x, uint32_t n, uint32_t *idx_out,
                     float *val_out);

#endif
