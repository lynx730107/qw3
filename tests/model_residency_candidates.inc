/* Private model-buffer residency probe. Original GGUF buffers and shaders. */
#define qw3_metal_set_model_map_range residency_original_map
#define qw3_metal_cleanup residency_original_cleanup
#include "../qw3_metal.m"
#undef qw3_metal_set_model_map_range
#undef qw3_metal_cleanup
static id<MTLResidencySet> model_residency_set;
static uint64_t residency_requests,residency_views,residency_bytes;
static int residency_active;
static int residency_select(int enabled) {
    if (@available(macOS 15.0,*)) {
        if (!model_residency_set) return !enabled;
        if (enabled && !residency_active) {
            [g_queue addResidencySet:model_residency_set];
            [model_residency_set requestResidency];
            residency_active=1;residency_requests++;
        } else if (!enabled && residency_active) {
            [g_queue removeResidencySet:model_residency_set];
            [model_residency_set endResidency];
            residency_active=0;
        }
        return 1;
    }
    return !enabled;
}
static void residency_drop(void) {
    if (@available(macOS 15.0,*)) {
        if (model_residency_set) {
            residency_select(0);
            model_residency_set=nil;
        }
    }
}
void qw3_metal_cleanup(void) {
    residency_drop();residency_original_cleanup();
}
int qw3_metal_set_model_map_range(const void *map,uint64_t size,uint64_t offset,uint64_t length) {
    residency_drop();
    if (!residency_original_map(map,size,offset,length)) return 0;
    const char *env=getenv("QW3_PRIVATE_MODEL_RESIDENCY");
    if (!env || strcmp(env,"1")) return 1;
    if (@available(macOS 15.0,*)) {
        MTLResidencySetDescriptor *desc=[MTLResidencySetDescriptor new];
        desc.label=@"private_qw3_model_residency";desc.initialCapacity=g_model_buffers.count;
        NSError *error=nil;
        model_residency_set=[g_device newResidencySetWithDescriptor:desc error:&error];
        if (!model_residency_set) {
            fprintf(stderr,"model residency unavailable: %s\n",error.localizedDescription.UTF8String);return 0;
        }
        for (id<MTLBuffer> buffer in g_model_buffers) [model_residency_set addAllocation:buffer];
        [model_residency_set commit];
        if (model_residency_set.allocationCount!=g_model_buffers.count || !g_model_buffers.count) return 0;
        if (!residency_select(1)) return 0;
        residency_views+=g_model_buffers.count;
        residency_bytes+=model_residency_set.allocatedSize;
        fprintf(stderr,"private model residency request=%llu allocations=%llu bytes=%llu\n",
            (unsigned long long)residency_requests,(unsigned long long)residency_views,(unsigned long long)residency_bytes);
        return 1;
    }
    return 0;
}
#define qw3_metal_env_n_gpu_layers residency_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled residency_core_llama_split_enabled
#define qw3_metal_llama_split_start residency_core_llama_split_start
#include "../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
