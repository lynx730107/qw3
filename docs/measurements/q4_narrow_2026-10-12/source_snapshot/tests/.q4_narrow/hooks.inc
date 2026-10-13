#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
static int q4_narrow_mode;
static uint64_t q4_narrow_calls,q4_narrow_fallbacks;
#include "core.inc"
#define qw3_metal_env_n_gpu_layers q4_narrow_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled q4_narrow_core_llama_split_enabled
#define qw3_metal_llama_split_start q4_narrow_core_llama_split_start
#include "../../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
static int q4_narrow_compile(void) {
    if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 0;
    id<MTLComputePipelineState> p=g_matvec_q4_k_pipeline;
    if (p.threadExecutionWidth!=32u || p.maxTotalThreadsPerThreadgroup<64u) return 0;
    fprintf(stderr,"q4_narrow_metadata width=%lu max_threads=%lu static_threadgroup_bytes=%lu dynamic_threadgroup_bytes=128 candidate_threads=32 native_min_threads=64 shader=original\n",(unsigned long)p.threadExecutionWidth,(unsigned long)p.maxTotalThreadsPerThreadgroup,(unsigned long)p.staticThreadgroupMemoryLength);
    return 1;
}
static void scratch_select(int mode) {q4_narrow_mode=mode!=0;}
static void q4_narrow_print_counts(void) {
    fprintf(stderr,"q4_narrow_dispatch_counts calls=%llu fallbacks=%llu\n",(unsigned long long)q4_narrow_calls,(unsigned long long)q4_narrow_fallbacks);
}
