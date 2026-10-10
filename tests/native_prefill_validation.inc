/* Exercise production dispatch against its runtime compact-path opt-out. */
#ifdef QW3_PRIVATE_COPY_CONTROL
static int probe_restore_copies;
#define qw3_metal_dense_prefill_strided_enabled probe_base_strided_enabled
#endif
#include "../qw3_metal.m"
#ifdef QW3_PRIVATE_COPY_CONTROL
#undef qw3_metal_dense_prefill_strided_enabled
int qw3_metal_dense_prefill_strided_enabled(uint32_t type, uint32_t n_tokens) {
    return !probe_restore_copies && probe_base_strided_enabled(type, n_tokens);
}
#endif
static int scratch_mode;
static uint64_t scratch_projection_calls, scratch_output_calls, scratch_copies_skipped;
static void scratch_select(int mode) {
    scratch_mode = mode;
#ifdef QW3_PRIVATE_COPY_CONTROL
    probe_restore_copies = mode != 0;
    setenv("QW3_METAL_DENSE_PREFILL_FAST", "1", 1);
#else
    setenv("QW3_METAL_DENSE_PREFILL_FAST", mode ? "1" : "0", 1);
#endif
}
#define qw3_metal_session_batch_dense_deltanet_projections_layout(...) \
    (scratch_projection_calls += direct_dn_projections ? 2 : 0, \
     scratch_copies_skipped += direct_dn_projections ? 2 : 0, \
     qw3_metal_session_batch_dense_deltanet_projections_layout(__VA_ARGS__))
#define qw3_metal_session_batch_dense_gqa_projections_layout(...) \
    (scratch_projection_calls += direct_gqa_projections ? 3 : 0, \
     scratch_copies_skipped += direct_gqa_projections ? 3 : 0, \
     qw3_metal_session_batch_dense_gqa_projections_layout(__VA_ARGS__))
#define qw3_metal_session_batch_dense_gqa_output_add_layout(...) \
    (scratch_output_calls += direct_output ? 1 : 0, \
     scratch_copies_skipped += direct_output ? 1 : 0, \
     qw3_metal_session_batch_dense_gqa_output_add_layout(__VA_ARGS__))
#define qw3_metal_env_n_gpu_layers native_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled native_core_llama_split_enabled
#define qw3_metal_llama_split_start native_core_llama_split_start
#include "../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
#undef qw3_metal_session_batch_dense_deltanet_projections_layout
#undef qw3_metal_session_batch_dense_gqa_projections_layout
#undef qw3_metal_session_batch_dense_gqa_output_add_layout
#define scratch_encode_matmul qw3_metal_encode_dense_batch_matmul_strided
#define scratch_matmul_available(t, n) qw3_metal_dense_prefill_strided_enabled((t), (n))
