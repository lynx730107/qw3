#include "native_prefill_validation.inc"

/* Synthetic activations, real bounded GGUF weight rows. Compare the unchanged
 * compact dispatch against the strided dispatch, including every output and
 * the gaps around the rows. No timing or independent arithmetic is introduced. */
static int scratch_matrix_case(QW3MetalSessionObj *obj, const qw3_tensor *weight,
                              uint32_t tokens, uint32_t n_out) {
    const uint32_t n_in = (uint32_t)weight->dim[0];
    const uint32_t input_offset = 3, input_stride = n_in + 9;
    const uint32_t output_offset = 5, output_stride = n_out + 13;
    const size_t in_bytes = (size_t)tokens * input_stride * sizeof(float);
    const size_t out_bytes = (size_t)tokens * output_stride * sizeof(float);
    id<MTLBuffer> input = [g_device newBufferWithLength:in_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> compact = [g_device newBufferWithLength:(size_t)tokens * n_in * sizeof(float)
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [g_device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> reference = [g_device newBufferWithLength:(size_t)tokens * n_out * sizeof(float)
        options:MTLResourceStorageModeShared];
    if (!input || !compact || !output || !reference) return 0;
    memset(input.contents, 0x5a, in_bytes);
    memset(output.contents, 0x5a, out_bytes);
    float *x = input.contents, *c = compact.contents;
    uint32_t rng = 13;
    for (uint32_t t = 0; t < tokens; t++) for (uint32_t i = 0; i < n_in; i++) {
        rng = rng * 1664525u + 1013904223u;
        const float value = (float)((int)(rng >> 16u) - 32768) / 32768.0f;
        c[(size_t)t * n_in + i] = x[(size_t)t * input_stride + input_offset + i] = value;
    }
    NSData *input_before = [NSData dataWithBytes:input.contents length:in_bytes];
    if (!qw3_metal_dense_batch_prepare_map(obj, tokens)) return 0;
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    if (!qw3_metal_encode_dense_batch_matmul(obj, cb, weight->offset,
            weight->type, tokens, n_in, n_out, compact, reference) ||
        !scratch_encode_matmul(obj, cb, weight->offset, weight->type,
            tokens, n_in, n_out, input, output,
            input_offset, input_stride, output_offset, output_stride)) return 0;
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted ||
        memcmp(input_before.bytes, input.contents, in_bytes)) return 0;
    const float *a = reference.contents, *b = output.contents;
    for (uint32_t t = 0; t < tokens; t++) for (uint32_t i = 0; i < output_stride; i++) {
        const uint32_t index = t * output_stride + i;
        if (i >= output_offset && i < output_offset + n_out) {
            const float *want = a + (size_t)t * n_out + i - output_offset;
            if (!isfinite(*want) || !isfinite(b[index]) || memcmp(want, b + index, sizeof(float))) {
                fprintf(stderr, "strided output mismatch type=%u tokens=%u n_in=%u n_out=%u row=%u col=%u\n",
                    weight->type, tokens, n_in, n_out, t, i - output_offset);
                return 0;
            }
        } else {
            uint32_t bits;
            memcpy(&bits, b + index, sizeof(bits));
            if (bits != 0x5a5a5a5au) {
                fprintf(stderr, "strided output guard overwritten type=%u row=%u col=%u\n", weight->type, t, i);
                return 0;
            }
        }
    }
    /* Reject a last row that would exceed the allocation and invalid layouts
     * before encoding any GPU work. */
    id<MTLCommandBuffer> invalid = [g_queue commandBuffer];
    if (scratch_encode_matmul(obj, invalid, weight->offset, weight->type,
            tokens, n_in, n_out, input, output, input_offset,
            input_offset + n_in - 1, output_offset, output_stride) ||
        scratch_encode_matmul(obj, invalid, weight->offset, weight->type,
            tokens, n_in, n_out, input, output, input_offset,
            input_stride, output_stride + 1, output_stride) ||
        scratch_encode_matmul(obj, invalid, weight->offset, weight->type,
            tokens + 1, n_in, n_out, input, output, input_offset,
            input_stride, output_offset, output_stride)) return 0;
    printf("matrix_exact type=%u tokens=%u n_in=%u n_out=%u guards=exact input=unchanged invalid_layout=rejected\n",
        weight->type, tokens, n_in, n_out);
    fflush(stdout);
    return 1;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 2) return 2;
        qw3_engine *engine = NULL;
        qw3_session *session = NULL;
        QW3MetalSessionObj *obj = nil;
        NSMutableSet *seen = nil;
        qw3_engine_options options = {.model_path=argv[1], .backend=QW3_BACKEND_METAL, .n_threads=4};
        int rc = 1, cases = 0;
        scratch_select(3);
        if (qw3_engine_open(&engine, &options) || !qw3_metal_compile_kernels() ||
            qw3_session_create(&session, engine, 512)) goto done;
        obj = (__bridge QW3MetalSessionObj *)session->metal->obj;
        seen = [NSMutableSet new];
        for (int l = 0; l < qw3_engine_layer_count(engine); l++) {
            qw3_layer_weights *w = &engine->weights.layer[l];
            const qw3_tensor *weights[] = {w->attn_q_proj, w->attn_k_proj,
                w->attn_v_proj, w->attn_o_proj, w->linear_qkv_proj,
                w->linear_gate_proj, w->linear_ssm_out};
            for (size_t i = 0; i < sizeof(weights) / sizeof(weights[0]); i++) {
                const qw3_tensor *t = weights[i];
                if (!t) continue;
                NSString *key = [NSString stringWithFormat:@"%u-%llu-%llu", t->type,
                    (unsigned long long)t->dim[0], (unsigned long long)t->dim[1]];
                if ([seen containsObject:key]) continue;
                if (!scratch_matmul_available(t->type, 128)) {
                    fprintf(stderr, "missing native strided pipeline %s\n", key.UTF8String);
                    goto done;
                }
                [seen addObject:key];
                const uint32_t tokens[] = {16, 31, 128};
                const uint32_t rows[] = {1, 33, (uint32_t)t->dim[1]};
                for (int n = 0; n < 3; n++) for (int r = 0; r < 3; r++) {
                    if (rows[r] > t->dim[1] || !scratch_matrix_case(obj, t, tokens[n], rows[r])) goto done;
                    cases++;
                }
            }
        }
        if (!cases) goto done;
        printf("PASS shapes=%lu cases=%d\n", (unsigned long)seen.count, cases);
        rc = 0;
done:
        if (rc) fprintf(stderr, "direct-scratch matrix quality failed after %d cases\n", cases);
        if (session) qw3_session_free(session);
        if (engine) qw3_engine_close(engine);
        return rc;
    }
}
