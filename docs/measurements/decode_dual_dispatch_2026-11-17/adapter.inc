#include "qw3_metal.m"

static unsigned dual_mode;
static unsigned dual_same_calls, dual_mixed_calls;
static id<MTLComputePipelineState> dual_pipelines[3];

static int dual_enabled(unsigned type_a, unsigned type_b) {
    if (!dual_mode || (type_a != 21 && type_a != 23) ||
        (type_b != 21 && type_b != 23)) return 0;
    return (dual_mode & (type_a == type_b ? 1u : 2u)) != 0;
}

static int dual_matvec(qw3_metal_session *s, uint64_t offset_a, uint64_t offset_b,
                       unsigned type_a, unsigned type_b, uint32_t n_in,
                       uint32_t rows_a, uint32_t rows_b) {
    if (!s || !s->obj || !n_in || n_in % 256u || !rows_a || !rows_b ||
        !qw3_metal_compile_kernels()) return 0;
    const unsigned slot = type_a != type_b ? 2u : type_a == 21u ? 0u : 1u;
    if (!dual_pipelines[slot]) {
        NSString *name = @[@"dual_iq3", @"dual_iq4", @"dual_mixed"][slot];
        NSError *error = nil;
        id<MTLFunction> fn = [g_library newFunctionWithName:name];
        if (fn) dual_pipelines[slot] = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!dual_pipelines[slot]) {
            fprintf(stderr, "dual pipeline %s failed: %s\n", name.UTF8String,
                    error ? error.localizedDescription.UTF8String : "function unavailable");
            return 0;
        }
    }
    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    if ((uint64_t)n_in * 4u > obj.x1.length ||
        ((uint64_t)rows_a + rows_b) * 4u > obj.scratch.length) return 0;

    uint32_t stride_a = (n_in / 256u) * (type_a == 21u ? 110u : 136u);
    uint32_t stride_b = (n_in / 256u) * (type_b == 21u ? 110u : 136u);
    uint64_t inner_a = 0, inner_b = 0;
    id<MTLBuffer> wa = qw3_metal_model_view_for(offset_a, (uint64_t)stride_a * rows_a, &inner_a);
    id<MTLBuffer> wb = qw3_metal_model_view_for(offset_b, (uint64_t)stride_b * rows_b, &inner_b);
    id<MTLBuffer> grid = slot == 1u ? nil : qw3_metal_iq3s_kgrid_buffer();
    if (!wa || !wb || (slot != 1u && !grid)) return 0;
    uint32_t rows_per_group_a = type_a == 21u ? 8u : 4u;
    uint32_t rows_per_group_b = type_b == 21u ? 8u : 4u;
    uint32_t groups_a = (rows_a + rows_per_group_a - 1u) / rows_per_group_a;
    uint32_t groups_b = (rows_b + rows_per_group_b - 1u) / rows_per_group_b;
    struct {
        struct { uint32_t n_in, n_out, row_bytes; } a, b;
        uint32_t groups_a, type_a, type_b;
    } args = {{n_in, rows_a, stride_a}, {n_in, rows_b, stride_b}, groups_a, type_a, type_b};
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    if (!cb) return 0;
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:dual_pipelines[slot]];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wa offset:(NSUInteger)inner_a atIndex:1];
    [enc setBuffer:wb offset:(NSUInteger)inner_b atIndex:2];
    [enc setBuffer:obj.x1 offset:0 atIndex:3];
    [enc setBuffer:obj.scratch offset:0 atIndex:4];
    if (grid) [enc setBuffer:grid offset:0 atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(groups_a + groups_b, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    qw3_metal_end_compute_encoder(cb, enc);
    if (slot == 2u) dual_mixed_calls++;
    else dual_same_calls++;
    return qw3_metal_finish_command_buffer(cb, owned, "dual matvec");
}

#define qw3_metal_env_n_gpu_layers dual_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled dual_core_llama_split_enabled
#define qw3_metal_llama_split_start dual_core_llama_split_start
#include "qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
