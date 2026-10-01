/* Q5_K head experiments only. The normal bridge is included privately;
 * its head entry point is renamed so the linked core can use our wrapper.
 * Nothing changes in production/CLI/agent. */
#define qw3_metal_session_matvec_q5_k_x1_to_logits qw3_bench_q5_baseline_head
#include "../qw3_metal.m"
#undef qw3_metal_session_matvec_q5_k_x1_to_logits
#include "../qw3.h"
#include <inttypes.h>
#include <time.h>

static double qw3_now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static id<MTLComputePipelineState> g_q5_pipeline;
static uint32_t g_q5_rows = 2, g_q5_sg = 2;
static int g_q5_candidate;
static qw3_metal_session *g_q5_session;
static uint64_t g_q5_offset;
static uint32_t g_q5_n_in, g_q5_n_out;
int qw3_metal_session_matvec_q5_k_x1_to_logits(qw3_metal_session *s,
                                               uint64_t tensor_offset,
                                               uint32_t n_in,
                                               uint32_t n_out,
                                               float *out) {
    g_q5_session = s; g_q5_offset = tensor_offset;
    g_q5_n_in = n_in; g_q5_n_out = n_out;
    if (!g_q5_candidate) return qw3_bench_q5_baseline_head(s, tensor_offset, n_in, n_out, out);
    if (!s || !s->obj || n_in == 0 || n_out == 0 || (n_in % 256u) != 0u)
        return 0;
    if (!g_initialized && !qw3_metal_init()) return 0;
    if (!qw3_metal_compile_kernels()) return 0;

    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)s->obj;
    const uint64_t x_bytes = (uint64_t)n_in * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_out * sizeof(float);
    if (!obj.x1 || !obj.logits || obj.x1.length < x_bytes ||
        obj.logits.length < out_bytes) return 0;

    const uint64_t row_bytes = (uint64_t)(n_in / 256u) * 176ull;
    const uint64_t tensor_bytes = row_bytes * (uint64_t)n_out;
    uint64_t model_offset = 0;
    id<MTLBuffer> wb = qw3_metal_model_view_for(
        tensor_offset, tensor_bytes, &model_offset);
    if (!wb) return 0;
    struct {
        uint32_t n_in;
        uint32_t n_out;
        uint32_t row_bytes;
    } args = { n_in, n_out, (uint32_t)row_bytes };

    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
    [enc setComputePipelineState:g_q5_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:wb offset:(NSUInteger)model_offset atIndex:1];
    [enc setBuffer:obj.x1 offset:0 atIndex:2];
    [enc setBuffer:obj.logits offset:0 atIndex:3];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((n_out + g_q5_rows * g_q5_sg - 1u) / (g_q5_rows * g_q5_sg), 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(32u * g_q5_sg, 1u, 1u)];
    qw3_metal_end_compute_encoder(cb, enc);

    id<MTLBuffer> readback = nil;
    if (out) {
        readback = [g_device newBufferWithLength:(NSUInteger)out_bytes
                                         options:MTLResourceStorageModeShared];
        if (!readback) return 0;
        qw3_metal_close_batch_encoder();
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:obj.logits sourceOffset:0
                    toBuffer:readback destinationOffset:0
                        size:(NSUInteger)out_bytes];
        [blit endEncoding];
    }
    if (out && !owned) {
        if (!qw3_metal_synchronize()) return 0;
    } else if (!qw3_metal_finish_command_buffer(cb, owned, "q5_K logits")) {
        return 0;
    }
    if (cb.status == MTLCommandBufferStatusError) return 0;
    if (out) memcpy(out, readback.contents, (size_t)out_bytes);
    return 1;
}

static id<MTLComputePipelineState> q5_candidate(uint32_t rows, uint32_t sg) {
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@"tests/q5_decode_candidates.metal"
                                               encoding:NSUTF8StringEncoding error:&error];
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    options.mathMode = MTLMathModeFast;
    id<MTLLibrary> library = source ? [g_device newLibraryWithSource:source
                                                           options:options error:&error] : nil;
    NSString *name = [NSString stringWithFormat:@"qw3_q5_r%u_sg%u", rows, sg];
    id<MTLFunction> function = [library newFunctionWithName:name];
    id<MTLComputePipelineState> pipeline = function ?
        [g_device newComputePipelineStateWithFunction:function error:&error] : nil;
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < sg * 32u)
        fprintf(stderr, "Q5 candidate unavailable: %s %s\n", name.UTF8String,
                error ? error.localizedDescription.UTF8String : "");
    return pipeline;
}

static int q5_compare(qw3_metal_session *s, uint64_t offset, uint32_t n_in,
                       uint32_t n_out, float *a, float *b) {
    g_q5_candidate = 0;
    if (!qw3_metal_session_matvec_q5_k_x1_to_logits(s, offset, n_in, n_out, a)) return 0;
    g_q5_candidate = 1;
    if (!qw3_metal_session_matvec_q5_k_x1_to_logits(s, offset, n_in, n_out, b)) return 0;
    double maxdiff = 0.0, ss = 0.0;
    for (uint32_t i = 0; i < n_out; i++) {
        double d = fabs((double)a[i] - b[i]);
        if (!isfinite(a[i]) || !isfinite(b[i])) return 0;
        if (d > maxdiff) maxdiff = d;
        ss += d * d;
    }
    double rms = sqrt(ss / n_out);
    printf("q5_compare n_in=%u n_out=%u maxdiff=%.9g rmsdiff=%.9g\n",
           n_in, n_out, maxdiff, rms);
    return maxdiff <= 0.0001 && rms <= 0.00001;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 4 && argc != 5) {
            fprintf(stderr, "usage: %s MODEL ROWS_PER_SIMD SIMD_GROUPS [--full]\n", argv[0]);
            return 1;
        }
        char *end = NULL;
        const long rows = strtol(argv[2], &end, 10);
        if (end == argv[2] || *end) return 1;
        const long sg = strtol(argv[3], &end, 10);
        if (end == argv[3] || *end) return 1;
        const int full = argc == 5 && !strcmp(argv[4], "--full");
        if ((argc == 5 && !full) || (rows != 1 && rows != 2 && rows != 4) ||
            (sg != 2 && sg != 4 && sg != 8) || (rows == 4 && sg == 8)) return 1;
        qw3_engine *engine = NULL;
        qw3_session *warm = NULL, *sessions[2] = {NULL, NULL};
        qw3_tokens prompt = {0};
        qw3_engine_options options = {.model_path = argv[1], .backend = QW3_BACKEND_METAL};
        float *a = NULL, *b = NULL;
        char error[512] = {0};
        int status = 1;
        if (qw3_engine_open(&engine, &options) != 0) goto done;
        g_q5_rows = rows; g_q5_sg = sg;
        g_q5_pipeline = q5_candidate(rows, sg);
        if (!g_q5_pipeline) goto done;
        qw3_encode_chat_prompt(engine, NULL,
            "Spiega dettagliatamente come implementare una tabella hash in C.",
            QW3_THINK_NONE, &prompt);
        if (qw3_session_create(&warm, engine, 512) != 0 ||
            qw3_session_sync(warm, &prompt, error, sizeof(error)) != 0 ||
            !g_q5_session || g_q5_n_in != 5120 || g_q5_n_out != 248320) goto done;
        qw3_metal_session *metal = g_q5_session;
        const uint64_t offset = g_q5_offset;
        const uint32_t n_in = g_q5_n_in, n_out = g_q5_n_out;
        a = calloc(n_out, sizeof(float)); b = calloc(n_out, sizeof(float));
        if (!a || !b) goto done;
        const uint32_t shapes[][2] = {
            {5120,248320}, {5120,248319}, {5120,4097},
            {5120,4111}, {4352,4097}, {4352,4111}
        };
        for (size_t i = 0; i < sizeof(shapes)/sizeof(shapes[0]); i++)
            if (!q5_compare(metal, offset, shapes[i][0], shapes[i][1], a, b)) goto done;
        double head[2] = {0};
        for (int i = 0; i < 80; i++) {
            for (int order = 0; order < 2; order++) {
                const int variant = order ^ (i & 1);
                g_q5_candidate = variant;
                const double start = qw3_now_sec();
                if (!qw3_metal_session_matvec_q5_k_x1_to_logits(metal, offset,
                                                                n_in, n_out, NULL)) goto done;
                if (i >= 8) head[variant] += qw3_now_sec() - start;
            }
        }
        printf("q5_micro rows=%ld sg=%ld baseline_ms=%.4f candidate_ms=%.4f gain_pct=%.2f\n",
               rows, sg, head[0] * 1000 / 72, head[1] * 1000 / 72,
               (head[0] / head[1] - 1) * 100);
        fflush(stdout);
        qw3_session_free(warm); warm = NULL;
        g_q5_session = NULL;
        if (!full) { status = 0; goto done; }
        for (int v = 0; v < 2; v++) {
            g_q5_candidate = v;
            if (qw3_session_create(&sessions[v], engine, 512) != 0 ||
                qw3_session_sync(sessions[v], &prompt, error, sizeof(error)) != 0) goto done;
        }
        uint64_t rng[2] = {42,42}, hash = 1469598103934665603ULL;
        double forward[2] = {0}, sampling[2] = {0}, block[2] = {0};
        for (int i = 0; i < 128; i++) {
            int tokens[2];
            for (int v = 0; v < 2; v++) {
                const qw3_tokens *history = qw3_session_tokens(sessions[v]);
                const int count = history->len > 1024 ? 1024 : history->len;
                const double start = qw3_now_sec();
                tokens[v] = qw3_session_sample_repetition(sessions[v],
                    0.6f, 20, 0.95f, 0, &rng[v],
                    history->v + history->len - count, count, 1.06f);
                if (i >= 8) sampling[v] += qw3_now_sec() - start;
            }
            if (tokens[0] < 0 || tokens[0] != tokens[1]) {
                fprintf(stderr, "token mismatch at %d: %d/%d\n", i, tokens[0], tokens[1]); goto done;
            }
            hash = (hash ^ (uint32_t)tokens[0]) * 1099511628211ULL;
            for (int order = 0; order < 2; order++) {
                const int v = order ^ (i & 1);
                g_q5_candidate = v;
                const double start = qw3_now_sec();
                if (qw3_session_eval_gpu_logits(sessions[v], tokens[v], error, sizeof(error)) != 0) goto done;
                const double elapsed = qw3_now_sec() - start;
                if (i >= 8) { forward[v] += elapsed; block[v] += elapsed; }
            }
            if (i < 4) {
                if (qw3_session_copy_logits(sessions[0], a, n_out) != (int)n_out ||
                    qw3_session_copy_logits(sessions[1], b, n_out) != (int)n_out) goto done;
                double maxdiff = 0, ss = 0;
                for (uint32_t j = 0; j < n_out; j++) {
                    if (!isfinite(a[j]) || !isfinite(b[j])) goto done;
                    double d = fabs((double)a[j] - b[j]);
                    if (d > maxdiff) maxdiff = d;
                    ss += d * d;
                }
                double rms = sqrt(ss / n_out);
                printf("q5_full_logits step=%d maxdiff=%.9g rmsdiff=%.9g\n", i, maxdiff, rms);
                if (maxdiff > 0.0001 || rms > 0.00001) goto done;
            }
            if (i >= 8 && ((i-7)%16 == 0 || i == 127)) {
                printf("block_end=%d baseline_ms=%.2f candidate_ms=%.2f gain_pct=%.2f\n",
                       i+1, block[0]*1000, block[1]*1000, (block[0]/block[1]-1)*100);
                block[0]=block[1]=0; fflush(stdout);
            }
        }
        printf("q5_full rows=%ld sg=%ld tokens=128 baseline=%.3f candidate=%.3f gain_pct=%.2f hash=%" PRIx64 "\n",
               rows, sg, 120/(forward[0]+sampling[0]), 120/(forward[1]+sampling[1]),
               (forward[0]/forward[1]-1)*100, hash);
        status = 0;
done:
        if (status) fprintf(stderr, "Q5 benchmark failed: %s\n", error);
        free(a); free(b); qw3_tokens_free(&prompt);
        if (warm) qw3_session_free(warm);
        for (int i=0;i<2;i++) if (sessions[i]) qw3_session_free(sessions[i]);
        if (engine) qw3_engine_close(engine);
        return status;
    }
}
