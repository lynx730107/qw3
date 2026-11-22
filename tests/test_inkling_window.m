#include "../qw3_metal.m"
#include <math.h>

static void fill_value(id<MTLBuffer> buffer, unsigned row, unsigned value) {
    _Float16 *data = buffer.contents;
    for (unsigned i = 0; i < QW3_INK_GPU_KV; i++) {
        data[(2 * row + 1) * QW3_INK_GPU_KV + i] = (_Float16)value;
    }
}

int main(void) {
    @autoreleasepool {
        if (!qw3_metal_init() || !qw3_metal_compile_kernels()) return 1;
        NSError *error = nil;
        id<MTLComputePipelineState> attention = [g_device newComputePipelineStateWithFunction:
            [g_library newFunctionWithName:@"qw3_inkling_attention_batch_tiled"] error:&error];
        id<MTLComputePipelineState> commit = [g_device newComputePipelineStateWithFunction:
            [g_library newFunctionWithName:@"qw3_inkling_attention_commit_batch"] error:&error];
        if (!attention || !commit) {
            fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
            return 1;
        }

        enum { WINDOW = 4, TOKENS = 2, Q_WIDTH = QW3_INK_GPU_HEADS * 128 };
        const NSUInteger sizes[] = {
            TOKENS * Q_WIDTH * sizeof(float),
            TOKENS * QW3_INK_GPU_HEADS * 16 * sizeof(float),
            TOKENS * QW3_INK_GPU_HEADS * sizeof(float),
            TOKENS * Q_WIDTH * sizeof(float),
            WINDOW * QW3_INK_GPU_KV_ROW_BYTES,
            16 * sizeof(float),
            TOKENS * Q_WIDTH * sizeof(float),
            TOKENS * QW3_INK_GPU_KV_ROW_BYTES,
        };
        id<MTLBuffer> buf[8];
        for (unsigned i = 0; i < 8; i++) {
            buf[i] = [g_device newBufferWithLength:sizes[i] options:MTLResourceStorageModeShared];
            if (!buf[i]) return 1;
            memset(buf[i].contents, 0, sizes[i]);
        }
        for (unsigned row = 0; row < WINDOW; row++) fill_value(buf[4], row, row);
        for (unsigned row = 0; row < TOKENS; row++) fill_value(buf[7], row, WINDOW + row);

        const struct {
            uint32_t pos0, n_tokens, capacity, extent, sliding;
            float qk_limit, relative_limit, log_alpha, log_floor;
        } args = {WINDOW, TOKENS, WINDOW, 0, 1, 1.0f, 1.0f, 0.0f, 1.0f};
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:attention];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        for (unsigned i = 0; i < 8; i++) [enc setBuffer:buf[i] offset:0 atIndex:i + 1];
        [enc dispatchThreadgroups:MTLSizeMake(QW3_INK_GPU_HEADS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc setComputePipelineState:commit];
        [enc setBuffer:buf[4] offset:0 atIndex:1];
        [enc setBuffer:buf[7] offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(QW3_INK_GPU_KV / 256, TOKENS, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "%s\n", cb.error.localizedDescription.UTF8String);
            return 1;
        }

        // Uniform logits give a denominator of five (four rows and the sink).
        // The zero gate halves the value. The two queries see [1..4] and [2..5].
        const float *out = buf[6].contents;
        for (unsigned i = 0; i < Q_WIDTH; i++) {
            const float first = out[i], second = out[Q_WIDTH + i];
            if (!isfinite(first) || !isfinite(second) ||
                fabsf(first - 1.0f) > 1e-6f || fabsf(second - 1.4f) > 1e-6f)
                return 1;
        }
        const _Float16 *cache = buf[4].contents;
        for (unsigned row = 0; row < WINDOW; row++) {
            const unsigned expected = row < TOKENS ? WINDOW + row : row;
            for (unsigned i = 0; i < QW3_INK_GPU_KV; i++) {
                if (cache[(2 * row + 1) * QW3_INK_GPU_KV + i] != (_Float16)expected) return 1;
            }
        }
        // Also verify Metal 4 NAX FlashAttention on the same sliding-window wraparound case
        if (g_metal4_tensor_api_enabled) {
            id<MTLFunction> naxFn = [g_library newFunctionWithName:@"qw3_inkling_attention_batch_nax"];
            id<MTLComputePipelineState> naxAttn = naxFn ?
                [g_device newComputePipelineStateWithFunction:naxFn error:&error] : nil;
            if (!naxAttn) {
                fprintf(stderr, "nax: %s\n", error ? error.localizedDescription.UTF8String : "missing");
                return 1;
            }
            for (unsigned row = 0; row < WINDOW; row++) fill_value(buf[4], row, row);
            memset(buf[6].contents, 0, sizes[6]);
            id<MTLCommandBuffer> cb2 = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc2 = [cb2 computeCommandEncoder];
            [enc2 setComputePipelineState:naxAttn];
            [enc2 setBytes:&args length:sizeof(args) atIndex:0];
            for (unsigned i = 0; i < 8; i++) [enc2 setBuffer:buf[i] offset:0 atIndex:i + 1];
            [enc2 setThreadgroupMemoryLength:25088u atIndex:0];
            [enc2 dispatchThreadgroups:MTLSizeMake(QW3_INK_GPU_HEADS, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc2 endEncoding];
            [cb2 commit];
            [cb2 waitUntilCompleted];
            if (cb2.status != MTLCommandBufferStatusCompleted) {
                fprintf(stderr, "nax: %s\n", cb2.error.localizedDescription.UTF8String);
                return 1;
            }
            const float *nout = buf[6].contents;
            for (unsigned i = 0; i < Q_WIDTH; i++) {
                const float first = nout[i], second = nout[Q_WIDTH + i];
                if (!isfinite(first) || !isfinite(second) ||
                    fabsf(first - 1.0f) > 1e-3f || fabsf(second - 1.4f) > 1e-3f) {
                    fprintf(stderr, "nax mismatch at %u: %g %g\n", i, first, second);
                    return 1;
                }
            }
        }
        puts("test-inkling-window: causal attention and final ring contents ok");
        return 0;
    }
}
