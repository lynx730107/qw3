/* One view set per process, preserving the stock benchmark protocol. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static int weight_native_untracked;
static uint64_t weight_native_views,weight_native_bytes;
static MTLResourceOptions weight_native_options(void) {
    return MTLResourceStorageModeShared |
        (weight_native_untracked ? MTLResourceHazardTrackingModeUntracked : 0);
}
#define qw3_metal_set_model_map_range weight_native_original_map
#include ".weight_tracking/native_core.inc"
#undef qw3_metal_set_model_map_range
int qw3_metal_set_model_map_range(const void *map,uint64_t size,uint64_t offset,uint64_t length) {
    if (!weight_native_original_map(map,size,offset,length)) return 0;
    for (id<MTLBuffer> b in g_model_buffers) {
        if (b.hazardTrackingMode != (weight_native_untracked ? MTLHazardTrackingModeUntracked : MTLHazardTrackingModeTracked)) return 0;
        weight_native_views++;weight_native_bytes+=b.length;
    }
    return weight_native_views!=0;
}
#define qw3_metal_env_n_gpu_layers weight_native_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled weight_native_core_llama_split_enabled
#define qw3_metal_llama_split_start weight_native_core_llama_split_start
#include "../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
#define main weight_native_original_main
#include "../qw3_bench.c"
#undef main
int main(int argc,char **argv) {
    const char *env=getenv("QW3_PRIVATE_WEIGHT_UNTRACKED");
    weight_native_untracked=env && !strcmp(env,"1");
    int rc=weight_native_original_main(argc,argv);
    fprintf(stderr,"private dispatches pair=0 add=0 gdn=0 weight_untracked=%d weight_views=%llu weight_bytes=%llu\n",
        weight_native_untracked,(unsigned long long)weight_native_views,(unsigned long long)weight_native_bytes);
    return rc;
}
