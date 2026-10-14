/* Original shaders, weights, data buffers and command buffer; two FFN dispatches overlap. */
#include "../../qw3_metal.m"
static int private_ffn_enabled;
static int private_ffn_active;
static uint64_t private_ffn_calls;
static int private_ffn_window_begin(void) {
    if (!private_ffn_enabled) return 1;
    if (private_ffn_active || !g_batch_cb || g_batch_concurrent || g_batch_separate_encoders) return 0;
    qw3_metal_close_batch_encoder();
    g_batch_concurrent=1;
    private_ffn_active=1;
    private_ffn_calls++;
    return 1;
}
static void private_ffn_window_end(void) {
    if (!private_ffn_active) return;
    qw3_metal_close_batch_encoder();
    g_batch_concurrent=0;
    private_ffn_active=0;
}
#define qw3_metal_env_n_gpu_layers ffn_concurrent_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled ffn_concurrent_core_llama_split_enabled
#define qw3_metal_llama_split_start ffn_concurrent_core_llama_split_start
#include "qw3_private.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
