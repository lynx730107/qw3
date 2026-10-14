#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static int private_ffn_enabled,private_decode_active,private_ffn_active,private_error;
static uint64_t private_ffn_calls,private_dispatches,private_barriers,private_skips;
static unsigned private_window_dispatches;
static void private_before_dispatch(id<MTLComputeCommandEncoder> enc) {
    if (!private_decode_active) return;
    private_dispatches++;
    if (private_ffn_active && private_window_dispatches==1u) private_skips++;
    else { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];private_barriers++; }
    if (private_ffn_active && ++private_window_dispatches>2u) private_error=1;
}
@protocol QW3FFNBarrierHooks
- (void)privateDispatchThreadgroups:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads;
- (void)privateDispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)group;
@end
@interface NSObject (QW3FFNBarrierProbe) <QW3FFNBarrierHooks>
@end
@implementation NSObject (QW3FFNBarrierProbe)
- (void)privateDispatchThreadgroups:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads {
    private_before_dispatch((id<MTLComputeCommandEncoder>)self);
    [(id<MTLComputeCommandEncoder>)self dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}
- (void)privateDispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)group {
    private_before_dispatch((id<MTLComputeCommandEncoder>)self);
    [(id<MTLComputeCommandEncoder>)self dispatchThreads:threads threadsPerThreadgroup:group];
}
@end
#include "qw3_private_metal.m"
static int private_decode_begin(void) {
    private_decode_active=private_ffn_enabled;
    return private_ffn_enabled ? qw3_metal_begin_commands_concurrent() : qw3_metal_begin_commands();
}
static void private_decode_end(void) {
    if (private_ffn_active) private_error=1;
    private_decode_active=0;private_ffn_active=0;
}
static int private_ffn_window_begin(void) {
    if (!private_decode_active) return 1;
    if (private_ffn_active || !g_batch_cb || !g_batch_concurrent || g_batch_separate_encoders) {private_error=1;return 0;}
    private_ffn_active=1;private_window_dispatches=0;private_ffn_calls++;
    return 1;
}
static void private_ffn_window_end(void) {
    if (!private_ffn_active) return;
    if (private_window_dispatches!=2u) private_error=1;
    private_ffn_active=0;
}
#define qw3_metal_env_n_gpu_layers ffn_barriers_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled ffn_barriers_core_llama_split_enabled
#define qw3_metal_llama_split_start ffn_barriers_core_llama_split_start
#include "qw3_private.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
