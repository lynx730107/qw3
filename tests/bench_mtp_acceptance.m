/* Feasibility only: qw3 target + native Metal MTP, llama.cpp MTP reference.
 * Teacher-forced greedy comparison, not speculative acceptance or speedup.
 * No sampler/rollback/agent integration. QW3_MTP_TRACE invalidates timing. */
#include "../qw3_metal.m"
/* The core and bridge normally have separate translation-unit statics. */
#define qw3_metal_env_n_gpu_layers qw3_core_env_n_gpu_layers
#define qw3_metal_llama_split_enabled qw3_core_llama_split_enabled
#define qw3_metal_llama_split_start qw3_core_llama_split_start
#include "../qw3.c"
#undef qw3_metal_env_n_gpu_layers
#undef qw3_metal_llama_split_enabled
#undef qw3_metal_llama_split_start
#include "llama.h"
#include "ggml-backend.h"

static int mtp_trace_active;
static struct { const char *name; float *values; uint32_t n; } mtp_traces[16];
static int mtp_trace_count;

static int mtp_native_trace(const char *name, id<MTLBuffer> buffer, uint32_t n) {
    if (!mtp_trace_active) return 1;
    if (mtp_trace_count >= 16) return 0;
    id<MTLBuffer> host = [g_device newBufferWithLength:(NSUInteger)n * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    if (!host) return 0;
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:buffer sourceOffset:0 toBuffer:host destinationOffset:0
                   size:(NSUInteger)n * sizeof(float)];
    [blit endEncoding];
    if (!qw3_metal_finish_command_buffer(cb, owned, "MTP trace") ||
        !qw3_metal_synchronize()) return 0;
    float *values = malloc((size_t)n * sizeof(float));
    if (!values) return 0;
    memcpy(values, host.contents, (size_t)n * sizeof(float));
    mtp_traces[mtp_trace_count].name = name;
    mtp_traces[mtp_trace_count].values = values;
    mtp_traces[mtp_trace_count++].n = n;
    return qw3_metal_begin_commands();
}

static bool mtp_reference_trace(struct ggml_tensor *tensor, bool ask, void *data) {
    (void)data;
    if (!mtp_trace_active) return !ask;
    for (int i = 0; i < mtp_trace_count; i++) {
        if (strncmp(tensor->name, mtp_traces[i].name, strlen(mtp_traces[i].name))) continue;
        if (ask) return true;
        const uint32_t n = mtp_traces[i].n;
        if (tensor->type != GGML_TYPE_F32 || ggml_nelements(tensor) != n) return true;
        float *reference = malloc((size_t)n * sizeof(float));
        if (!reference) return false;
        ggml_backend_tensor_get(tensor, reference, 0, (size_t)n * sizeof(float));
        double max_abs = 0.0, ss = 0.0;
        for (uint32_t j = 0; j < n; j++) {
            const double d = (double)reference[j] - mtp_traces[i].values[j];
            if (fabs(d) > max_abs) max_abs = fabs(d);
            ss += d * d;
        }
        printf("trace %s rmse=%.9g max_abs=%.9g first_native=%.9g first_reference=%.9g\n",
            tensor->name, sqrt(ss / n), max_abs, mtp_traces[i].values[0], reference[0]);
        free(reference);
        return true;
    }
    return !ask;
}

static int mtp_norm(qw3_metal_session *session, uint64_t offset, uint32_t n,
                     const char *name) {
    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)session->obj;
    return qw3_metal_session_rmsnorm_weight_f32(session, offset, n, QW3_RMS_EPS, NULL) &&
           mtp_native_trace(name, obj.x1, n);
}

static void mtp_quiet_log(enum ggml_log_level level, const char *message, void *data) {
    (void)data;
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) fputs(message, stderr);
}

/* Diagnostic-only graph assembled from existing, unchanged Metal kernels.
 * Independent one-layer KV state: never overwrite the target's recurrent state. */
static int mtp_copy(id<MTLBuffer> source, NSUInteger source_offset,
                    id<MTLBuffer> destination, NSUInteger destination_offset,
                    NSUInteger bytes) {
    if (!source || !destination || source_offset > source.length ||
        bytes > source.length - source_offset || destination_offset > destination.length ||
        bytes > destination.length - destination_offset) return 0;
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:source sourceOffset:source_offset toBuffer:destination
       destinationOffset:destination_offset size:bytes];
    [blit endEncoding];
    return qw3_metal_finish_command_buffer(cb, owned, "MTP diagnostic copy");
}

static int mtp_native_eval(qw3_engine *engine, qw3_metal_session *session,
                           int token, uint32_t pos, const float *previous_hidden,
                           id<MTLBuffer> upload, id<MTLBuffer> concat,
                           id<MTLBuffer> readback, float *logits) {
    const qw3_model_spec *spec = engine->model.spec;
    const qw3_layer_weights *lw = &engine->weights.layer[spec->n_layer];
    const uint32_t n = spec->n_embd, ff = spec->n_ff_dense;
    const NSUInteger bytes = (NSUInteger)n * sizeof(float);
    QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)session->obj;
    id<MTLBuffer> original_x1 = obj.x1;
    memcpy(upload.contents, previous_hidden, bytes);
    int ok = qw3_metal_begin_commands() &&
        engine->weights.token_embd->type == QW3_TENSOR_Q3_K &&
        qw3_metal_session_embed_q3_k(session, engine->weights.token_embd->offset,
                                     token, n, NULL) &&
        mtp_norm(session, lw->nextn_enorm->offset, n, "mtp_enorm") &&
        mtp_copy(obj.x1, 0, concat, 0, bytes) &&
        mtp_copy(upload, 0, obj.x0, 0, bytes) &&
        mtp_norm(session, lw->nextn_hnorm->offset, n, "mtp_hnorm") &&
        mtp_copy(obj.x1, 0, concat, bytes, bytes);
    /* GGUF nextn.eh_proj takes [normalized embedding, normalized hidden]. */
    obj.x1 = concat;
    if (ok) ok = qw3_metal_session_matvec_quant_x1_to_scratch(session,
        lw->nextn_eh_proj->offset, lw->nextn_eh_proj->type, 2u * n, n, 0, NULL);
    obj.x1 = original_x1; /* Keep both buffers alive until the batch completes. */
    if (ok) ok = mtp_copy(obj.scratch, 0, obj.x0, 0, bytes) &&
        mtp_native_trace("mtp_eh_proj", obj.x0, n) &&
        mtp_norm(session, lw->attn_norm->offset, n, "mtp_attn_norm") &&
        qw3_metal_session_gqa_project_cache_typed(session,
            lw->attn_q_proj->offset, lw->attn_k_proj->offset, lw->attn_v_proj->offset,
            lw->attn_q_proj->type, lw->attn_k_proj->type, lw->attn_v_proj->type, n,
            lw->attn_q_norm->offset, lw->attn_k_norm->offset,
            2u * spec->n_head * spec->n_head_dim, spec->n_head * spec->n_head_dim,
            spec->n_head_kv * spec->n_head_dim, spec->n_head, spec->n_head_kv,
            spec->n_head_dim, spec->rope_dim, 0, pos, spec->rope_theta,
            QW3_RMS_EPS, NULL, NULL, NULL, NULL);
    if (ok) ok = mtp_native_trace("mtp_Qcur_normed", obj.gqaTmpQ,
        spec->n_head * spec->n_head_dim) &&
        mtp_native_trace("mtp_Kcur_normed", obj.gqaTmpK,
        spec->n_head_kv * spec->n_head_dim) &&
        mtp_native_trace("mtp_Vcur", obj.gqaTokenV,
        spec->n_head_kv * spec->n_head_dim) &&
        mtp_native_trace("mtp_gate", obj.gqaTokenGate,
        spec->n_head * spec->n_head_dim);
    if (ok) ok = pos == 0 ?
        qw3_metal_session_gqa_single_attn_out_typed(session, lw->attn_o_proj->offset,
            lw->attn_o_proj->type, spec->n_head, spec->n_head_kv, spec->n_head_dim, n, NULL) :
        qw3_metal_session_gqa_cached_attn_out_typed(session, lw->attn_o_proj->offset,
            lw->attn_o_proj->type, pos + 1u, 0, spec->n_head, spec->n_head_kv,
            spec->n_head_dim, n, NULL);
    if (ok) ok = mtp_native_trace("mtp_attn_out", obj.x1, n) &&
        qw3_metal_session_residual_rmsnorm_update_x0_x1(session,
        lw->ffn_norm->offset, n, QW3_RMS_EPS, NULL) &&
        mtp_native_trace("mtp_attn_residual", obj.x0, n) &&
        mtp_native_trace("mtp_attn_post_norm", obj.x1, n);
    if (ok && lw->ffn_gate->type == QW3_TENSOR_IQ3_S &&
        lw->ffn_up->type == QW3_TENSOR_IQ3_S) {
        ok = qw3_metal_session_matvec_iq3_s_pair_x1_to_scratch(session,
            lw->ffn_gate->offset, lw->ffn_up->offset, n, ff);
    } else if (ok) {
        ok = qw3_metal_session_matvec_quant_x1_to_scratch(session,
            lw->ffn_gate->offset, lw->ffn_gate->type, n, ff, 0, NULL) &&
            qw3_metal_session_matvec_quant_x1_to_scratch(session,
            lw->ffn_up->offset, lw->ffn_up->type, n, ff, ff, NULL);
    }
    if (ok) ok = qw3_metal_session_silu_mul_scratch_to_inner(session, 0, ff, ff) &&
        qw3_metal_session_matvec_quant_inner_to_x1(session,
            lw->ffn_down->offset, lw->ffn_down->type, ff, n, NULL) &&
        qw3_metal_session_scale_x1_add_x0(session, 1.0f, n) &&
        mtp_native_trace("mtp_post_ffn", obj.x0, n) &&
        mtp_norm(session, lw->nextn_shared_head_norm->offset, n, "h_nextn");
    /* Catch-up needs the hidden and KV, not an unused full vocabulary head. */
    if (ok && logits) ok = engine->weights.output->type == QW3_TENSOR_Q5_K &&
        qw3_metal_session_matvec_q5_k_x1_to_logits(session,
            engine->weights.output->offset, n, spec->n_vocab, NULL) &&
        mtp_copy(obj.logits, 0, readback, 0, (NSUInteger)spec->n_vocab * sizeof(float));
    const int completed = qw3_metal_synchronize();
    if (!ok || !completed) return 0;
    if (logits) memcpy(logits, readback.contents, (size_t)spec->n_vocab * sizeof(float));
    return 1;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 2 || argc > 4) {
            fprintf(stderr, "usage: %s MODEL [STEPS=32] [PROMPT]\n", argv[0]);
            return 1;
        }
        char *end = NULL;
        const long requested_steps = argc >= 3 ? strtol(argv[2], &end, 10) : 32;
        if (requested_steps < 1 || requested_steps > 128 ||
            (argc >= 3 && (!argv[2][0] || *end))) return 1;
        const int steps = (int)requested_steps;
        qw3_engine *engine = NULL;
        qw3_session *target = NULL;
        qw3_tokens prompt = {0};
        qw3_engine_options options = {.model_path = argv[1], .backend = QW3_BACKEND_METAL};
        struct llama_model *model = NULL;
        struct llama_context *draft = NULL;
        qw3_metal_session *native = NULL;
        float *hidden = NULL;
        float *native_logits = NULL;
        double *native_time = NULL;
        int *tokens = NULL, *expected = NULL;
        id<MTLBuffer> readback = nil;
        id<MTLBuffer> upload = nil, concat = nil, native_readback = nil;
        char error[512] = {0};
        int status = 1;
        int n_rows = 0, n_embd = 0, n_vocab = 0, checked = 0, accepted = 0;
        int native_agreement = 0;
        double max_abs = 0.0, squared_error = 0.0, native_total = 0.0;
        int current = 0;
        uint64_t rng = 42;
        double draft_time = 0.0;
        if (qw3_engine_open(&engine, &options) != 0) goto done;
        if (!engine->model.spec || engine->model.spec->n_layer_nextn != 1 ||
            engine->model.spec->ffn_kind != QW3_FFN_DENSE) {
            fprintf(stderr, "probe requires the supported dense model with one existing MTP head\n");
            goto done;
        }
        n_embd = engine->model.spec->n_embd;
        n_vocab = engine->model.spec->n_vocab;
        char name[128];
        snprintf(name, sizeof(name), "blk.%d.nextn.embed_tokens.weight", engine->model.spec->n_layer);
        if (model_find_tensor(&engine->model, name)) {
            fprintf(stderr, "probe does not support a separate MTP embedding\n"); goto done;
        }
        snprintf(name, sizeof(name), "blk.%d.nextn.shared_head_head.weight", engine->model.spec->n_layer);
        if (model_find_tensor(&engine->model, name)) {
            fprintf(stderr, "probe does not support a separate MTP output head\n"); goto done;
        }
        printf("mode=teacher_forced_greedy timing=%s\n",
               getenv("QW3_MTP_TRACE") ? "trace_instrumented_not_benchmark" : "synchronized_readback");
        qw3_encode_chat_prompt(engine, NULL, argc == 4 ? argv[3] :
            "Spiega dettagliatamente come implementare una tabella hash in C.",
            QW3_THINK_NONE, &prompt);
        n_rows = prompt.len + steps - 1;
        if (prompt.len < 1 || n_rows >= 512) goto done;
        hidden = calloc((size_t)(n_rows + 1) * n_embd, sizeof(float));
        tokens = calloc((size_t)n_rows, sizeof(int));
        expected = calloc((size_t)n_rows, sizeof(int));
        readback = [g_device newBufferWithLength:(NSUInteger)n_embd * sizeof(float)
                                         options:MTLResourceStorageModeShared];
        if (!hidden || !tokens || !expected || !readback ||
            qw3_session_create(&target, engine, 512) != 0) goto done;
        /* MTP token t consumes the normalized target hidden from t-1.
         * x1 after the LM head is precisely the target output-norm result;
         * reading x0 would pass the wrong (unnormalized) representation. */
        for (int i = 0; i < n_rows; i++) {
            if (i < prompt.len) current = prompt.v[i];
            tokens[i] = current;
            if (qw3_session_eval_gpu_logits(target, current, error, sizeof(error)) != 0) goto done;
            current = qw3_session_sample(target, 0.0f, 0, 1.0f, 0, &rng);
            if (current < 0) goto done;
            expected[i] = current;
            QW3MetalSessionObj *obj = (__bridge QW3MetalSessionObj *)target->metal->obj;
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit copyFromBuffer:obj.x1 sourceOffset:0 toBuffer:readback
               destinationOffset:0 size:(NSUInteger)n_embd * sizeof(float)];
            [blit endEncoding];
            if (!qw3_metal_finish_command_buffer(cb, 1, "MTP hidden capture")) goto done;
            memcpy(hidden + (size_t)(i + 1) * n_embd, readback.contents,
                   (size_t)n_embd * sizeof(float));
        }
        printf("target_capture rows=%d prompt=%d checked=%d hidden=output_norm shift=1\n",
            n_rows, prompt.len, steps);
        fflush(stdout);
        qw3_session_free(target); target = NULL;
        const qw3_model_spec *spec = engine->model.spec;
        qw3_metal_model_config config = {
            .n_layer = 1, .n_embd = n_embd, .vocab_size = n_vocab,
            .n_head = spec->n_head, .n_head_kv = spec->n_head_kv,
            .n_head_dim = spec->n_head_dim, .n_ff_dense = spec->n_ff_dense,
            .n_linear_qk_heads = spec->n_linear_qk_heads,
            .n_linear_v_heads = spec->n_linear_v_heads,
            .n_linear_head_dim = spec->n_linear_head_dim,
            .n_linear_qkv = spec_tensor_linear_qkv(spec),
            .n_linear_inner = spec->n_linear_inner,
            .n_linear_conv_k = spec->n_linear_conv_k, .full_attn_interval = 1
        };
        native = qw3_metal_session_create(512, &config);
        qw3_metal_state_info info = {0};
        if (!native || !qw3_metal_session_get_state_info(native, &info) ||
            info.kv_type != 1u || info.n_full_layers != 1u || info.n_linear_layers != 0u)
            goto done;
        upload = [g_device newBufferWithLength:(NSUInteger)n_embd * sizeof(float)
                                       options:MTLResourceStorageModeShared];
        concat = qw3_metal_new_private_buffer((uint64_t)2u * n_embd * sizeof(float));
        native_readback = [g_device newBufferWithLength:(NSUInteger)n_vocab * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        native_logits = calloc((size_t)steps * n_vocab, sizeof(float));
        native_time = calloc((size_t)steps, sizeof(double));
        if (!upload || !concat || !native_readback || !native_logits || !native_time) goto done;
        for (int i = 0; i < n_rows; i++) {
            const int row = i >= prompt.len - 1 ? i - prompt.len + 1 : 0;
            mtp_trace_active = getenv("QW3_MTP_TRACE") && i == n_rows - 1;
            const double start = qw3_now_sec();
            if (!mtp_native_eval(engine, native, tokens[i], i, hidden + (size_t)i * n_embd,
                                  upload, concat, native_readback,
                                  native_logits + (size_t)row * n_vocab)) goto done;
            if (i >= prompt.len - 1) native_time[row] = qw3_now_sec() - start;
        }
        qw3_metal_session_free(native); native = NULL;
        upload = nil; concat = nil; native_readback = nil;
        /* Open the runtimes sequentially: do not keep two full models resident. */
        qw3_engine_close(engine); engine = NULL;
        readback = nil;
        llama_log_set(mtp_quiet_log, NULL);
        ggml_log_set(mtp_quiet_log, NULL);
        llama_backend_init();
        struct llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        mp.load_mtp = true;
        model = llama_model_load_from_file(argv[1], mp);
        if (!model || llama_model_n_embd(model) != n_embd) goto done;
        struct llama_context_params cp = llama_context_default_params();
        cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cp.n_ctx = 512;
        cp.n_batch = 512;
        cp.n_ubatch = 1;
        cp.n_threads = cp.n_threads_batch = 10;
        cp.type_k = cp.type_v = GGML_TYPE_F16;
        if (getenv("QW3_MTP_TRACE")) cp.cb_eval = mtp_reference_trace;
        draft = llama_init_from_model(model, cp);
        if (!draft) goto done;
        for (int i = 0; i < n_rows; i++) {
            mtp_trace_active = getenv("QW3_MTP_TRACE") && i == n_rows - 1;
            llama_token token = tokens[i];
            llama_pos pos = i;
            int32_t seq_count = 1;
            llama_seq_id seq_id = 0;
            llama_seq_id *seq = &seq_id;
            int8_t want_logits = 1;
            struct llama_batch batch = {
                .n_tokens = 1, .token = &token, .embd = hidden + (size_t)i * n_embd,
                .pos = &pos, .n_seq_id = &seq_count, .seq_id = &seq, .logits = &want_logits
            };
            const double start = qw3_now_sec();
            if (llama_decode(draft, batch) != 0) goto done;
            llama_synchronize(draft);
            const double elapsed = qw3_now_sec() - start;
            if (i < prompt.len - 1) continue;
            const float *logits = llama_get_logits_ith(draft, 0);
            if (!logits) goto done;
            int top = 0;
            const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
            if (vocab != n_vocab) goto done;
            const float *candidate = native_logits + (size_t)checked * n_vocab;
            int native_top = 0;
            for (int j = 0; j < vocab; j++) {
                if (!isfinite(logits[j]) || !isfinite(candidate[j])) goto done;
                if (logits[j] > logits[top]) top = j;
                if (candidate[j] > candidate[native_top]) native_top = j;
                const double diff = (double)candidate[j] - logits[j];
                if (fabs(diff) > max_abs) max_abs = fabs(diff);
                squared_error += diff * diff;
            }
            native_agreement += native_top == top;
            native_total += native_time[checked];
            checked++;
            accepted += top == expected[i];
            draft_time += elapsed;
            printf("step=%d target=%d draft=%d match=%d reference_mtp_ms=%.3f\n",
                checked, expected[i], top, top == expected[i], elapsed * 1000);
            fflush(stdout);
        }
        if (checked != steps) goto done;
        printf("greedy_agreement=%d/%d (%.2f%%) reference_mtp_ms=%.3f\n",
            accepted, checked, 100.0 * accepted / checked, draft_time * 1000 / checked);
        const double rmse = sqrt(squared_error / ((double)steps * n_vocab));
        printf("native_reference greedy_match=%d/%d rmse=%.9g max_abs=%.9g "
               "native_mtp_readback_ms=%.3f\n", native_agreement, steps, rmse, max_abs,
               native_total * 1000 / steps);
        /* Same strict limits as the existing dense llama differential test. */
        status = native_agreement != steps || rmse > 0.001 || max_abs > 0.005;
        if (status) snprintf(error, sizeof(error), "native MTP differs from reference beyond dense-logit limits");
done:
        if (status) fprintf(stderr, "MTP feasibility probe failed: %s\n", error);
        if (draft) llama_free(draft);
        if (model) llama_model_free(model);
        if (native) qw3_metal_session_free(native);
        if (target) qw3_session_free(target);
        if (engine) qw3_engine_close(engine);
        qw3_tokens_free(&prompt);
        free(expected); free(tokens); free(hidden);
        free(native_logits); free(native_time);
        for (int i = 0; i < mtp_trace_count; i++) free(mtp_traces[i].values);
        return status;
    }
}
