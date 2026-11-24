#include "../qw3_metal.m"
#include <math.h>

static void fill_value(id<MTLBuffer> buffer, unsigned row, unsigned value) {
    _Float16 *data = buffer.contents;
    for (unsigned i = 0; i < QW3_INK_GPU_KV; i++) {
        data[(2 * row + 1) * QW3_INK_GPU_KV + i] = (_Float16)value;
    }
}

static void fill_value_q8(id<MTLBuffer> buffer, unsigned row, unsigned value) {
    uint8_t *base = (uint8_t *)buffer.contents + (size_t)row * QW3_INK_GPU_KV_Q8_ROW_BYTES;
    int8_t *vq = (int8_t *)(base + 1024);
    _Float16 *vs = (_Float16 *)(base + 2112);
    const float sc = value ? ((float)value / 127.0f) : 0.0f;
    const int8_t q = value ? 127 : 0;
    for (unsigned i = 0; i < QW3_INK_GPU_KV; i++) vq[i] = q;
    for (unsigned i = 0; i < 32; i++) vs[i] = (_Float16)sc;
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
        // Also verify Q8_0 KV cache wraparound for tiled, commit, and NAX FlashAttention
        {
            id<MTLComputePipelineState> attnQ8 = [g_device newComputePipelineStateWithFunction:
                [g_library newFunctionWithName:@"qw3_inkling_attention_batch_tiled_q8"] error:&error];
            id<MTLComputePipelineState> commitQ8 = [g_device newComputePipelineStateWithFunction:
                [g_library newFunctionWithName:@"qw3_inkling_attention_commit_batch_q8"] error:&error];
            if (!attnQ8 || !commitQ8) {
                fprintf(stderr, "q8: %s\n", error ? error.localizedDescription.UTF8String : "missing");
                return 1;
            }
            id<MTLBuffer> q8Cache = [g_device newBufferWithLength:WINDOW * QW3_INK_GPU_KV_Q8_ROW_BYTES
                                                          options:MTLResourceStorageModeShared];
            if (!q8Cache) return 1;
            memset(q8Cache.contents, 0, q8Cache.length);
            for (unsigned row = 0; row < WINDOW; row++) fill_value_q8(q8Cache, row, row);
            memset(buf[6].contents, 0, sizes[6]);

            id<MTLCommandBuffer> cbq = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> encq = [cbq computeCommandEncoder];
            [encq setComputePipelineState:attnQ8];
            [encq setBytes:&args length:sizeof(args) atIndex:0];
            for (unsigned i = 0; i < 8; i++) [encq setBuffer:(i == 4 ? q8Cache : buf[i]) offset:0 atIndex:i + 1];
            [encq dispatchThreadgroups:MTLSizeMake(QW3_INK_GPU_HEADS, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encq setComputePipelineState:commitQ8];
            [encq setBuffer:q8Cache offset:0 atIndex:1];
            [encq setBuffer:buf[7] offset:0 atIndex:2];
            [encq dispatchThreadgroups:MTLSizeMake(16, TOKENS, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [encq endEncoding];
            [cbq commit];
            [cbq waitUntilCompleted];
            if (cbq.status != MTLCommandBufferStatusCompleted) {
                fprintf(stderr, "q8: %s\n", cbq.error.localizedDescription.UTF8String);
                return 1;
            }
            const float *qout = buf[6].contents;
            for (unsigned i = 0; i < Q_WIDTH; i++) {
                const float first = qout[i], second = qout[Q_WIDTH + i];
                if (!isfinite(first) || !isfinite(second) ||
                    fabsf(first - 1.0f) > 5e-3f || fabsf(second - 1.4f) > 5e-3f) {
                    fprintf(stderr, "q8 tiled mismatch at %u: %g %g\n", i, first, second);
                    return 1;
                }
            }
            for (unsigned row = 0; row < WINDOW; row++) {
                const float expected = (float)(row < TOKENS ? WINDOW + row : row);
                const uint8_t *base = (const uint8_t *)q8Cache.contents + (size_t)row * QW3_INK_GPU_KV_Q8_ROW_BYTES;
                const int8_t *vq = (const int8_t *)(base + 1024);
                const _Float16 *vs = (const _Float16 *)(base + 2112);
                for (unsigned i = 0; i < QW3_INK_GPU_KV; i++) {
                    const float got = (float)vq[i] * (float)vs[i / 32];
                    if (fabsf(got - expected) > 5e-2f) {
                        fprintf(stderr, "q8 ring mismatch row=%u i=%u got=%g exp=%g\n", row, i, got, expected);
                        return 1;
                    }
                }
            }
            if (g_metal4_tensor_api_enabled) {
                id<MTLFunction> naxQ8Fn = [g_library newFunctionWithName:@"qw3_inkling_attention_batch_nax_q8"];
                id<MTLComputePipelineState> naxQ8 = naxQ8Fn ?
                    [g_device newComputePipelineStateWithFunction:naxQ8Fn error:&error] : nil;
                if (!naxQ8) {
                    fprintf(stderr, "nax q8: %s\n", error ? error.localizedDescription.UTF8String : "missing");
                    return 1;
                }
                memset(q8Cache.contents, 0, q8Cache.length);
                for (unsigned row = 0; row < WINDOW; row++) fill_value_q8(q8Cache, row, row);
                memset(buf[6].contents, 0, sizes[6]);
                id<MTLCommandBuffer> cbq2 = [g_queue commandBuffer];
                id<MTLComputeCommandEncoder> encq2 = [cbq2 computeCommandEncoder];
                [encq2 setComputePipelineState:naxQ8];
                [encq2 setBytes:&args length:sizeof(args) atIndex:0];
                for (unsigned i = 0; i < 8; i++) [encq2 setBuffer:(i == 4 ? q8Cache : buf[i]) offset:0 atIndex:i + 1];
                [encq2 setThreadgroupMemoryLength:25088u atIndex:0];
                [encq2 dispatchThreadgroups:MTLSizeMake(QW3_INK_GPU_HEADS, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [encq2 endEncoding];
                [cbq2 commit];
                [cbq2 waitUntilCompleted];
                if (cbq2.status != MTLCommandBufferStatusCompleted) {
                    fprintf(stderr, "nax q8: %s\n", cbq2.error.localizedDescription.UTF8String);
                    return 1;
                }
                const float *nqout = buf[6].contents;
                for (unsigned i = 0; i < Q_WIDTH; i++) {
                    const float first = nqout[i], second = nqout[Q_WIDTH + i];
                    if (!isfinite(first) || !isfinite(second) ||
                        fabsf(first - 1.0f) > 1e-2f || fabsf(second - 1.4f) > 1e-2f) {
                        fprintf(stderr, "nax q8 mismatch at %u: %g %g\n", i, first, second);
                        return 1;
                    }
                }
            }
        }
        puts("test-inkling-window: causal attention and final ring contents ok (fp16 + q8_0)");
        return 0;
    }
}
