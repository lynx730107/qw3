/* Private read-only GGUF views. Session buffers and kernels remain native. */
#define qw3_metal_set_model_map_range weight_original_map
#define qw3_metal_cleanup weight_original_cleanup
#include "../qw3_metal.m"
#undef qw3_metal_set_model_map_range
#undef qw3_metal_cleanup
static NSMutableArray<id<MTLBuffer>> *weight_views[2];
static NSMutableDictionary<NSString *, id<MTLBuffer>> *weight_temp[2];
static uint64_t weight_switches[2], weight_view_count, weight_view_bytes;
static int weight_select(int mode) {
    int v=mode!=0;
    if (!weight_views[v] || !qw3_metal_synchronize()) return 0;
    g_model_buffers=weight_views[v];g_model_temp_buffers=weight_temp[v];
    weight_switches[v]++;
    for (id<MTLBuffer> b in g_model_buffers)
        if (b.hazardTrackingMode != (v ? MTLHazardTrackingModeUntracked : MTLHazardTrackingModeTracked)) return 0;
    return 1;
}
static void weight_drop(void) {
    for (int v=0;v<2;v++) { weight_views[v]=nil;weight_temp[v]=nil; }
}
void qw3_metal_cleanup(void) {
    weight_drop();weight_original_cleanup();
}
int qw3_metal_set_model_map_range(const void *map,uint64_t size,uint64_t offset,uint64_t length) {
    weight_drop();
    if (!weight_original_map(map,size,offset,length)) return 0;
    weight_views[0]=g_model_buffers;
    weight_views[1]=[NSMutableArray new];
    for (id<MTLBuffer> native in weight_views[0]) {
        id<MTLBuffer> b=[g_device newBufferWithBytesNoCopy:native.contents length:native.length
            options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked deallocator:nil];
        if (!b || b.hazardTrackingMode!=MTLHazardTrackingModeUntracked ||
            native.hazardTrackingMode!=MTLHazardTrackingModeTracked) return 0;
        [weight_views[1] addObject:b];weight_view_count++;weight_view_bytes+=b.length;
    }
    for (int v=0;v<2;v++) weight_temp[v]=[NSMutableDictionary new];
    const char *env=getenv("QW3_PRIVATE_WEIGHT_UNTRACKED");
    return weight_select(env && !strcmp(env,"1"));
}
#define qw3_metal_env_n_gpu_layers weight_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled weight_core_llama_split_enabled
#define qw3_metal_llama_split_start weight_core_llama_split_start
#include "../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
