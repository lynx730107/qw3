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
int qw3_metal_flush_commands(void);
int qw3_metal_end_commands(void);
int qw3_metal_synchronize(void);

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
int qw3_metal_session_deltanet_gated_rmsnorm_from_buffers(
    qw3_metal_session *s, uint64_t norm_weight_offset, uint32_t z_offset,
    uint32_t v_heads, uint32_t head_dim, float eps, float *out);
int qw3_metal_session_matvec_q8_0_inner_to_x1(qw3_metal_session *s,
                                              uint64_t tensor_offset,
                                              uint32_t n_in, uint32_t n_out,
                                              float *out);
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
