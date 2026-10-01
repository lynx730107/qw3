/* Private pipeline access for quant checks; paired production-policy selection.
 * No CLI/agent integration or speculative generation. */
#include "../qw3_metal.m"
#include "../qw3.h"

static int short_quant_checks(qw3_engine *engine);

static id<MTLComputePipelineState> short_baseline, short_candidate;
static id<MTLComputePipelineState> short_iq2_baseline, short_iq2_candidate;

static int short_policy_checks(void) {
    char saved_device[sizeof(g_device_name)];
    memcpy(saved_device,g_device_name,sizeof(saved_device));
    const char *original = getenv("QW3_METAL_DENSE_SHORT_VECTOR");
    char *saved_env = original ? strdup(original) : NULL;
    if (original && !saved_env) return 0;
    int ok = 1;
    setenv("QW3_METAL_DENSE_SHORT_VECTOR","1",1);
    ok &= qw3_metal_dense_short_vector_enabled(2,4352,4111);
    ok &= qw3_metal_dense_short_vector_enabled(8,5120,17408);
    ok &= !qw3_metal_dense_short_vector_enabled(1,5120,17408);
    ok &= !qw3_metal_dense_short_vector_enabled(9,5120,17408);
    ok &= !qw3_metal_dense_short_vector_enabled(16,5120,17408);
    ok &= !qw3_metal_dense_short_vector_enabled(4,256,17408);
    ok &= !qw3_metal_dense_short_vector_enabled(4,5120,33);
    setenv("QW3_METAL_DENSE_SHORT_VECTOR","0",1);
    ok &= !qw3_metal_dense_short_vector_enabled(8,5120,17408);
    unsetenv("QW3_METAL_DENSE_SHORT_VECTOR");
    ok &= qw3_metal_dense_short_vector_enabled(8,5120,17408) ==
          (strcmp(g_device_name,"Apple M5") == 0);
    snprintf(g_device_name,sizeof(g_device_name),"diagnostic other GPU");
    ok &= !qw3_metal_dense_short_vector_enabled(8,5120,17408);
    setenv("QW3_METAL_DENSE_SHORT_VECTOR","1",1);
    ok &= qw3_metal_dense_short_vector_enabled(8,5120,17408);
    memcpy(g_device_name,saved_device,sizeof(saved_device));
    if (saved_env) { setenv("QW3_METAL_DENSE_SHORT_VECTOR",saved_env,1); free(saved_env); }
    else unsetenv("QW3_METAL_DENSE_SHORT_VECTOR");
    printf("short_policy_checks=%s\n",ok ? "ok" : "failed");
    return ok;
}

static int short_initialize(void) {
    if (!qw3_metal_compile_kernels() || !short_policy_checks()) return 0;
    short_baseline = g_matmul_iq3_xxs_batch4_pipeline;
    short_iq2_baseline = g_matmul_iq2_s_batch4_pipeline;
    short_candidate = g_matmul_iq3_xxs_short_pipeline;
    short_iq2_candidate = g_matmul_iq2_s_short_pipeline;
    if (!short_baseline || !short_candidate || !short_iq2_baseline || !short_iq2_candidate) {
        fprintf(stderr, "short batch pipeline unavailable\n");
        return 0;
    }
    return 1;
}

static void short_select(int variant) {
    setenv("QW3_METAL_DENSE_SHORT_VECTOR", variant ? "1" : "0", 1);
}

#define QW3_BLOCK_INIT short_initialize
#define QW3_BLOCK_SELECT short_select
#define QW3_BLOCK_QUANT_CHECK short_quant_checks
#define qw3_metal_env_n_gpu_layers qw3_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled qw3_core_llama_split_enabled
#define qw3_metal_llama_split_start qw3_core_llama_split_start
#include "bench_verify_blocks.c"

static int short_quant_checks(qw3_engine *engine) {
    const uint32_t shapes[][2] = {
        {256,1}, {4352,33}, {5120,4097}, {5120,4111},
        {5120,17408}, {17408,5120}
    };
    const uint32_t counts[] = {1,2,3,4,5,8,15};
    id<MTLBuffer> signs = nil, iq2grid = nil, iq3grid = nil;
    if (!qw3_metal_dense_iq_buffers(&signs,NULL,&iq2grid,&iq3grid)) return 0;
    int checks = 0;
    for (int kind = 0; kind < 2; kind++) {
        const uint32_t type = kind ? QW3_TENSOR_IQ2_S : QW3_TENSOR_IQ3_XXS;
        const qw3_tensor *weight = NULL;
        for (int layer = 0; layer < engine->model.spec->n_layer && !weight; layer++) {
            const qw3_layer_weights *lw = &engine->weights.layer[layer];
            const qw3_tensor *possibilities[] = {lw->ffn_gate,lw->ffn_up,lw->ffn_down};
            for (int p = 0; p < 3; p++) if (possibilities[p] && possibilities[p]->type == type) {
                weight = possibilities[p]; break;
            }
        }
        if (!weight) return 0;
        for (int shape = 0; shape < 6; shape++) for (int count = 0; count < 7; count++) {
            const uint32_t ni = shapes[shape][0], no = shapes[shape][1], nt = counts[count];
            const uint32_t row_bytes = ni/256u*(kind ? 82u : 98u);
            const uint64_t bytes = (uint64_t)no*row_bytes;
            if ((uint64_t)ni*no > weight->elements) return 0;
            uint64_t offset = 0;
            id<MTLBuffer> wb = qw3_metal_model_view_for(weight->offset,bytes,&offset);
            const uint32_t in_offset = 7u, in_stride = ni+13u, out_offset = 5u, out_stride = no+11u;
            const uint64_t input_count = (uint64_t)nt*in_stride;
            const uint64_t output_count = (uint64_t)nt*out_stride;
            id<MTLBuffer> x = [g_device newBufferWithLength:input_count*sizeof(float)
                                                 options:MTLResourceStorageModeShared];
            id<MTLBuffer> outputs[2];
            for (int v=0;v<2;v++) outputs[v] =
                [g_device newBufferWithLength:output_count*sizeof(float)
                                     options:MTLResourceStorageModeShared];
            if (!wb || !x || !outputs[0] || !outputs[1]) return 0;
            float *input = x.contents;
            for (uint64_t i=0;i<input_count;i++)
                input[i] = ((int)((i*37u+count*19u)%101u)-50)/31.0f;
            struct { uint32_t ni,no,rb,nt,io,is,oo,os; } args = {
                ni,no,row_bytes,nt,in_offset,in_stride,out_offset,out_stride};
            for (int v=0;v<2;v++) {
                float *out = outputs[v].contents;
                for (uint64_t i=0;i<output_count;i++) out[i] = 123456.0f;
                int owned=0;
                id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
                id<MTLComputeCommandEncoder> enc = qw3_metal_compute_encoder(cb);
                [enc setComputePipelineState:kind ? (v ? short_iq2_candidate : short_iq2_baseline) :
                                                       (v ? short_candidate : short_baseline)];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:wb offset:offset atIndex:1];
                [enc setBuffer:x offset:0 atIndex:2];
                [enc setBuffer:outputs[v] offset:0 atIndex:3];
                [enc setBuffer:kind ? iq2grid : iq3grid offset:0 atIndex:4];
                if (!kind) [enc setBuffer:signs offset:0 atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((no+3u)/4u,(nt+3u)/4u,1)
                    threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                qw3_metal_end_compute_encoder(cb,enc);
                if (!qw3_metal_finish_command_buffer(cb,owned,"short quant check")) return 0;
            }
            const float *a = outputs[0].contents, *b = outputs[1].contents;
            double maximum = 0.0, ss = 0.0;
            for (uint32_t t=0;t<nt;t++) for (uint32_t j=0;j<out_stride;j++) {
                const uint64_t i = (uint64_t)t*out_stride+j;
                if (j<out_offset || j>=out_offset+no) {
                    if (a[i] != 123456.0f || b[i] != 123456.0f) return 0;
                } else {
                    if (!isfinite(a[i]) || !isfinite(b[i])) return 0;
                    const double d = fabs((double)a[i]-b[i]);
                    if (d>maximum) maximum=d;
                    ss += d*d;
                }
            }
            const double rms = sqrt(ss/((uint64_t)nt*no));
            // Independent double dot against CPU dequantization at boundary rows.
            qw3_tensor view = *weight;
            view.dim[0]=ni; view.dim[1]=no; view.elements=(uint64_t)ni*no;
            float *row = malloc(ni*sizeof(float));
            if (!row) return 0;
            const uint32_t rows[] = {0u,no/2u,no-1u};
            double cpu_max = 0;
            for (int r=0;r<3;r++) {
                if (!tensor_read_dense_row(&engine->model,&view,rows[r],row)) {free(row);return 0;}
                for (uint32_t t=0;t<nt;t++) {
                    double sum=0.0;
                    for (uint32_t j=0;j<ni;j++) sum += (double)row[j]*input[(uint64_t)t*in_stride+in_offset+j];
                    const double d = fabs(sum-b[(uint64_t)t*out_stride+out_offset+rows[r]]);
                    if (!isfinite(sum) || !isfinite(d)) {free(row);return 0;}
                    if (d>cpu_max) cpu_max=d;
                }
            }
            free(row);
            printf("short_quant type=%u n_in=%u n_out=%u tokens=%u max=%.9g rms=%.9g cpu_max=%.9g guards=ok\n",
                   type,ni,no,nt,maximum,rms,cpu_max);
            if (maximum>0.0001 || rms>0.00001 || cpu_max>0.0001) return 0;
            checks++;
        }
    }
    printf("short_quant_checks=ok checks=%d\n",checks);
    fflush(stdout);
    return 1;
}
