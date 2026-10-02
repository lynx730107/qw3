/* Experimental greedy MTP decoder, deliberately separate from CLI/agent.
 * Reuse the reference-checked native draft graph and both private engines.
 * Batched verification exposes EVERY hidden/logit row. Rejection restores
 * the entire GPU state and replays only the accepted prefix. This is a
 * correctness-first baseline. serial-replay records compact recurrent inputs
 * and restores accepted state without embeddings/projections/FFN/head replay.
 * Sampling, EOS stopping, cancellation and public session APIs are absent. */
#include "../qw3.h"
static int decode_replay_record(qw3_session *s, uint32_t slot, uint32_t count);
#define QW3_METAL_DENSE_REPLAY_RECORD(s, slot, count) decode_replay_record(s, slot, count)
#define main qw3_mtp_teacher_forced_entrypoint
#include "bench_mtp_acceptance.m"
#undef main
#include <inttypes.h>

/* Private scalar-verifier log. No production session allocations or switches.
 * Projected QKV/Z/alpha/beta are recorded before convolution modifies state. */
static qw3_session *replay_capture_session;
static uint32_t replay_row, replay_width, replay_slots, replay_count;
static uint64_t replay_seen[8];
static id<MTLBuffer> replay_inputs, replay_kv, replay_heads;

static int decode_replay_record(qw3_session *s, uint32_t slot, uint32_t count) {
    if (s != replay_capture_session) return 1;
    if (replay_row >= replay_width || slot >= replay_slots || count != replay_count ||
        (replay_seen[replay_row] & (1ull << slot))) return 0;
    QW3MetalSessionObj *o = (__bridge QW3MetalSessionObj *)s->metal->obj;
    const NSUInteger bytes = (NSUInteger)count * sizeof(float);
    const NSUInteger dst = ((NSUInteger)slot*replay_width+replay_row)*bytes;
    if (!replay_inputs || o.scratch.length < bytes ||
        dst > replay_inputs.length || bytes > replay_inputs.length-dst) return 0;
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!blit) return 0;
    [blit copyFromBuffer:o.scratch sourceOffset:0 toBuffer:replay_inputs
       destinationOffset:dst size:bytes];
    [blit endEncoding];
    if (!qw3_metal_finish_command_buffer(cb, owned, "record recurrent inputs")) return 0;
    replay_seen[replay_row] |= 1ull << slot;
    return 1;
}

static int decode_replay_record_end(qw3_session *s, int row) {
    if (s != replay_capture_session) return 1;
    if (row < 0 || (uint32_t)row >= replay_width ||
        replay_seen[row] != ((1ull << replay_slots) - 1ull)) return 0;
    const qw3_model_spec *sp = s->engine->model.spec;
    qw3_metal_state_info info;
    if (!qw3_metal_session_get_state_info(s->metal, &info) || info.kv_type != 1u ||
        !qw3_metal_begin_commands()) return 0;
    QW3MetalSessionObj *o = (__bridge QW3MetalSessionObj *)s->metal->obj;
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!blit) return 0;
    const NSUInteger kv_bytes = (NSUInteger)info.kv_row_bytes;
    for (uint32_t slot = 0; slot < info.n_full_layers; slot++) {
        for (uint32_t plane = 0; plane < 2; plane++) {
            id<MTLBuffer> b = nil;
            uint64_t base = 0, length = 0;
            if (!qw3_metal_session_state_span(o, plane ? QW3_METAL_STATE_GQA_V : QW3_METAL_STATE_GQA_K,
                                              slot, &b, &base, &length)) {
                [blit endEncoding]; (void)qw3_metal_synchronize(); return 0;
            }
            [blit copyFromBuffer:b sourceOffset:base + (NSUInteger)(s->kv.pos - 1)*kv_bytes
                toBuffer:replay_kv destinationOffset:
                    (((NSUInteger)slot * replay_width + row)*2 + plane)*kv_bytes size:kv_bytes];
        }
    }
    const NSUInteger d_bytes = (NSUInteger)sp->n_embd * sizeof(float);
    const NSUInteger v_bytes = (NSUInteger)sp->n_vocab * sizeof(float);
    const NSUInteger base = (NSUInteger)row * (2*d_bytes + v_bytes);
    [blit copyFromBuffer:o.x0 sourceOffset:0 toBuffer:replay_heads destinationOffset:base size:d_bytes];
    [blit copyFromBuffer:o.x1 sourceOffset:0 toBuffer:replay_heads destinationOffset:base+d_bytes size:d_bytes];
    [blit copyFromBuffer:o.logits sourceOffset:0 toBuffer:replay_heads destinationOffset:base+2*d_bytes size:v_bytes];
    [blit endEncoding];
    return qw3_metal_synchronize();
}

/* Replay only the state transition, using unchanged scalar Metal arithmetic.
 * Each layer's projected inputs were computed from the original true prefix.
 * This does not run embeddings, attention projections, FFN or the LM head. */
static int decode_replay_apply(qw3_session *s, int pos, int commit) {
    const qw3_model_spec *sp = s->engine->model.spec;
    QW3MetalSessionObj *o = (__bridge QW3MetalSessionObj *)s->metal->obj;
    qw3_metal_state_info info;
    if (commit < 1 || commit > (int)replay_width ||
        !qw3_metal_session_get_state_info(s->metal, &info) || info.kv_type != 1u ||
        !qw3_metal_begin_commands()) return 0;
    const uint32_t qkv = spec_tensor_linear_qkv(sp);
    const uint32_t alpha = qkv + sp->n_linear_inner;
    const uint32_t beta = alpha + sp->n_linear_v_heads;
    const NSUInteger bytes = (NSUInteger)replay_count * sizeof(float);
    int ok = 1;
    for (int il = 0; ok && il < sp->n_layer; il++) {
        if (qw3_spec_layer_is_full_attention(sp, il)) continue;
        const uint32_t slot = s->dn.linear_layer_map[il];
        const qw3_layer_weights *lw = &s->engine->weights.layer[il];
        for (int t = 0; ok && t < commit; t++) {
            int owned = 0;
            id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
            qw3_metal_close_batch_encoder();
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            if (!blit) { ok = 0; break; }
            [blit copyFromBuffer:replay_inputs sourceOffset:((NSUInteger)slot*replay_width+t)*bytes
                toBuffer:o.scratch destinationOffset:0 size:bytes];
            [blit endEncoding];
            ok = qw3_metal_session_conv1d_step_from_scratch(s->metal,
                     lw->linear_conv_weight->offset, slot, qkv, NULL, NULL) &&
                 qw3_metal_session_l2norm_qk_from_conv(s->metal,
                     sp->n_linear_qk_heads, sp->n_linear_head_dim, QW3_RMS_EPS, NULL, NULL) &&
                 qw3_metal_session_deltanet_fused_gdn_from_scratch(s->metal,
                     lw->linear_ssm_dt_bias->offset, lw->linear_ssm_a->offset,
                     lw->linear_ssm_norm->offset, qkv, alpha, beta, slot,
                     sp->n_linear_qk_heads, sp->n_linear_v_heads, sp->n_linear_head_dim, QW3_RMS_EPS);
        }
    }
    int owned = 0;
    id<MTLCommandBuffer> cb = qw3_metal_command_buffer(&owned);
    qw3_metal_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!blit) ok = 0;
    const NSUInteger kv_bytes = (NSUInteger)info.kv_row_bytes;
    const NSUInteger head_bytes = (NSUInteger)sp->n_head_dim * sizeof(uint16_t);
    const NSUInteger tail_plane = 32u * sp->n_head_kv * kv_bytes;
    for (uint32_t slot = 0; ok && slot < info.n_full_layers; slot++) {
        id<MTLBuffer> tail = qw3_metal_gqa_flash_tail_buffer(o, slot);
        if (!tail) { ok = 0; break; }
        for (uint32_t plane = 0; ok && plane < 2; plane++) {
            id<MTLBuffer> b = nil;
            uint64_t base = 0, length = 0;
            ok = qw3_metal_session_state_span(o, plane ? QW3_METAL_STATE_GQA_V : QW3_METAL_STATE_GQA_K,
                                             slot, &b, &base, &length);
            if (!ok) break;
            for (int t = 0; t < commit; t++) {
                const NSUInteger src = (((NSUInteger)slot*replay_width+t)*2+plane)*kv_bytes;
                [blit copyFromBuffer:replay_kv sourceOffset:src toBuffer:b
                    destinationOffset:base+(NSUInteger)(pos+t)*kv_bytes size:kv_bytes];
                for (int h = 0; h < sp->n_head_kv; h++) {
                    const NSUInteger dst = plane*tail_plane + ((NSUInteger)h*32+(pos+t)%32)*kv_bytes;
                    [blit copyFromBuffer:replay_kv sourceOffset:src+h*head_bytes
                        toBuffer:tail destinationOffset:dst size:head_bytes];
                }
            }
        }
    }
    if (ok) {
        const NSUInteger d_bytes = (NSUInteger)sp->n_embd * sizeof(float);
        const NSUInteger v_bytes = (NSUInteger)sp->n_vocab * sizeof(float);
        const NSUInteger src = (NSUInteger)(commit-1)*(2*d_bytes+v_bytes);
        [blit copyFromBuffer:replay_heads sourceOffset:src toBuffer:o.x0 destinationOffset:0 size:d_bytes];
        [blit copyFromBuffer:replay_heads sourceOffset:src+d_bytes toBuffer:o.x1 destinationOffset:0 size:d_bytes];
        [blit copyFromBuffer:replay_heads sourceOffset:src+2*d_bytes toBuffer:o.logits destinationOffset:0 size:v_bytes];
    }
    if (blit) [blit endEncoding];
    return qw3_metal_synchronize() && ok;
}

static int decode_top(const float *p, int n) {
    int best = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(p[i])) return -1;
        if (p[i] > p[best]) best = i;
    }
    return best;
}

static NSArray<id<MTLBuffer>> *decode_state_buffers(qw3_metal_session *s) {
    QW3MetalSessionObj *o = (__bridge QW3MetalSessionObj *)s->obj;
    NSMutableArray<id<MTLBuffer>> *v = [NSMutableArray array];
    if (o.gqaKLayers.count) {
        [v addObjectsFromArray:o.gqaKLayers];
        [v addObjectsFromArray:o.gqaVLayers];
    } else {
        if (o.gqaK) [v addObject:o.gqaK];
        if (o.gqaV) [v addObject:o.gqaV];
    }
    /* The F16 attention tail can overwrite older ring slots while drafting:
     * trimming position alone is not a rollback. Preserve it too. */
    if (o.gqaFlashTailLayers) [v addObjectsFromArray:o.gqaFlashTailLayers];
    [v addObject:o.deltanetState];
    [v addObject:o.convState];
    [v addObject:o.logits];
    [v addObject:o.x0];
    [v addObject:o.x1];
    return v;
}

static NSArray<id<MTLBuffer>> *decode_backup(NSArray<id<MTLBuffer>> *live) {
    NSMutableArray<id<MTLBuffer>> *v = [NSMutableArray array];
    for (id<MTLBuffer> b in live) {
        id<MTLBuffer> copy = qw3_metal_new_private_buffer(b.length);
        if (!copy) return nil;
        [v addObject:copy];
    }
    return v;
}

static int decode_copy_state(NSArray<id<MTLBuffer>> *src,
                             NSArray<id<MTLBuffer>> *dst) {
    if (!src || !dst || src.count != dst.count || !qw3_metal_begin_commands()) return 0;
    int ok = 1;
    for (NSUInteger i = 0; ok && i < src.count; i++) {
        ok = src[i].length == dst[i].length &&
             mtp_copy(src[i], 0, dst[i], 0, src[i].length);
    }
    return qw3_metal_synchronize() && ok;
}

static void decode_rewind(qw3_session *s, int pos) {
    s->kv.pos = (uint64_t)pos;
    s->tokens.len = pos;
    s->valid = true;
    s->cpu_logits_valid = false;
}

static int decode_hidden(qw3_metal_session *s, id<MTLBuffer> host, float *out, int n) {
    QW3MetalSessionObj *o = (__bridge QW3MetalSessionObj *)s->obj;
    if (!mtp_copy(o.x1, 0, host, 0, (NSUInteger)n * sizeof(float)) ||
        !qw3_metal_synchronize()) return 0;
    memcpy(out, host.contents, (size_t)n * sizeof(float));
    return 1;
}

/* Serial heads intentionally retain the ordinary FP32 GEMV arithmetic.
 * Head batching is a subsequent optimization, not hidden in the cost probe. */
static int decode_verify(qw3_session *s, const int *tokens, int n, int batch,
                          id<MTLBuffer> log_host, id<MTLBuffer> h_host,
                          float *logits, float *hidden, char *err, size_t errlen) {
    const qw3_model_spec *spec = s->engine->model.spec;
    const int d = spec->n_embd, v = spec->n_vocab;
    QW3MetalSessionObj *o = (__bridge QW3MetalSessionObj *)s->metal->obj;
    if (batch && n > 1 && qw3_metal_session_eval_dense_prefill_batch(
            s, tokens, n, err, errlen, QW3_METAL_LOGITS_DEFER) != 0) return 0;
    if (batch && n > 1 && !qw3_metal_begin_commands()) return 0;
    for (int i = 0; i < n; i++) {
        if (s == replay_capture_session) {
            if (batch || i >= (int)replay_width) return 0;
            replay_row = (uint32_t)i;
        }
        if (batch && n > 1) {
            if (!qw3_metal_session_copy_batch_x0_to_x0(s->metal, i, d) ||
                !qw3_metal_session_rmsnorm_weight_f32(s->metal,
                    s->engine->weights.output_norm->offset, d, QW3_RMS_EPS, NULL) ||
                !qw3_metal_session_matvec_q5_k_x1_to_logits(s->metal,
                    s->engine->weights.output->offset, d, v, NULL)) return 0;
        } else if (qw3_session_eval_gpu_logits(s, tokens[i], err, errlen) != 0) return 0;
        if (!decode_replay_record_end(s, i)) return 0;
        if (!mtp_copy(o.logits, 0, log_host, (NSUInteger)i * v * sizeof(float),
                      (NSUInteger)v * sizeof(float)) ||
            !mtp_copy(o.x1, 0, h_host, (NSUInteger)i * d * sizeof(float),
                      (NSUInteger)d * sizeof(float))) return 0;
    }
    if (!qw3_metal_synchronize()) return 0;
    memcpy(logits, log_host.contents, (size_t)n * v * sizeof(float));
    memcpy(hidden, h_host.contents, (size_t)n * d * sizeof(float));
    s->cpu_logits_valid = false;
    return 1;
}

static int decode_vector_error(const float *a, const float *b, int n,
                                double *maximum, double *rms) {
    double ss = 0;
    *maximum = 0;
    *rms = INFINITY;
    for (int i = 0; i < n; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i])) return 0;
        double diff = (double)a[i] - b[i];
        if (fabs(diff) > *maximum) *maximum = fabs(diff);
        ss += diff * diff;
    }
    *rms = sqrt(ss / n);
    return 1;
}

static int decode_compare(const float *a, const float *b, int n,
                            double *maximum, double *rms) {
    return decode_vector_error(a, b, n, maximum, rms) &&
           *maximum <= 0.005 && *rms <= 0.001 &&
           decode_top(a, n) == decode_top(b, n);
}

/* After rollback + serial replay, every persistent buffer must be identical
 * to an independent restore + replay, including unused KV and Flash tails. */
static int decode_state_equal(NSArray<id<MTLBuffer>> *a,
                              NSArray<id<MTLBuffer>> *b) {
    const NSUInteger chunk = 1024u * 1024u;
    id<MTLBuffer> ha = [g_device newBufferWithLength:chunk options:MTLResourceStorageModeShared];
    id<MTLBuffer> hb = [g_device newBufferWithLength:chunk options:MTLResourceStorageModeShared];
    if (!ha || !hb || a.count != b.count) return 0;
    for (NSUInteger i = 0; i < a.count; i++) {
        if (a[i].length != b[i].length) return 0;
        for (NSUInteger off = 0; off < a[i].length; off += chunk) {
            NSUInteger bytes = MIN(chunk, a[i].length - off);
            if (!qw3_metal_begin_commands() || !mtp_copy(a[i], off, ha, 0, bytes) ||
                !mtp_copy(b[i], off, hb, 0, bytes) || !qw3_metal_synchronize() ||
                memcmp(ha.contents, hb.contents, bytes)) {
                fprintf(stderr, "rollback state differs buffer=%lu offset=%lu\n",
                        (unsigned long)i, (unsigned long)off);
                return 0;
            }
        }
    }
    return 1;
}

static int decode_int(const char *s, int lo, int hi) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    return !*s || *end || v < lo || v > hi ? -1 : (int)v;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 2 || argc > 7) {
            fprintf(stderr, "usage: %s MODEL [STEPS=32] [BLOCK=8] [serial|serial-replay|batch] [PROMPT] [FORCE_REJECT_INDEX]\n", argv[0]);
            return 1;
        }
        const int steps = argc > 2 ? decode_int(argv[2], 1, 128) : 32;
        const int block = argc > 3 ? decode_int(argv[3], 2, 8) : 8;
        const int batch = argc > 4 && !strcmp(argv[4], "batch");
        const int compact = argc > 4 && !strcmp(argv[4], "serial-replay");
        const int force = argc > 6 ? decode_int(argv[6], 1, block - 1) : -1;
        if (steps < 0 || block < 0 || (argc > 4 && !batch && !compact && strcmp(argv[4], "serial")) ||
            (argc > 6 && (force < 0 || steps < block))) return 1;
        qw3_engine *e = NULL;
        qw3_session *target = NULL, *baseline = NULL, *oracle = NULL;
        qw3_metal_session *draft = NULL;
        id<MTLBuffer> upload = nil, concat = nil, draft_log = nil;
        id<MTLBuffer> h_one = nil, log_host = nil, h_host = nil;
        NSArray<id<MTLBuffer>> *live = nil, *backup = nil, *oracle_live = nil;
        NSArray<id<MTLBuffer>> *draft_live = nil, *draft_backup = nil;
        qw3_tokens prompt = {0};
        float *prev = NULL, *roll = NULL, *proposal_logits = NULL;
        float *rows = NULL, *hidden = NULL, *check = NULL, *check_hidden = NULL;
        int *expected = NULL;
        char err[512] = {0};
        int status = 1, produced = 0, cycles = 0, rejected = 0, accepted_drafts = 0;
        int forced_seen = 0, rollback_checks = 0, checked_rows = 0;
        double baseline_time = 0, total = 0, draft_time = 0, verify_time = 0;
        double save_time = 0, restore_time = 0, catchup_time = 0;
        double scalar_restore_time = 0;
        double worst_max = 0, worst_rms = 0, hidden_max = 0, hidden_rms = 0;
        qw3_engine_options options = {.model_path = argv[1], .backend = QW3_BACKEND_METAL};
        if (qw3_engine_open(&e, &options) != 0) goto done;
        const qw3_model_spec *sp = e->model.spec;
        if (!sp || sp->ffn_kind != QW3_FFN_DENSE || sp->n_layer_nextn != 1 ||
            e->weights.token_embd->type != QW3_TENSOR_Q3_K ||
            e->weights.output->type != QW3_TENSOR_Q5_K) goto done;
        char name[128];
        snprintf(name, sizeof(name), "blk.%d.nextn.embed_tokens.weight", sp->n_layer);
        if (model_find_tensor(&e->model, name)) goto done;
        snprintf(name, sizeof(name), "blk.%d.nextn.shared_head_head.weight", sp->n_layer);
        if (model_find_tensor(&e->model, name)) goto done;
        const int d = sp->n_embd, v = sp->n_vocab;
        qw3_encode_chat_prompt(e, NULL, argc > 5 ? argv[5] :
            "Spiega dettagliatamente come implementare una tabella hash in C.",
            QW3_THINK_NONE, &prompt);
        if (prompt.len < 1 || prompt.len + steps + block > 512 ||
            qw3_session_create(&target, e, 512) != 0 ||
            qw3_session_create(&baseline, e, 512) != 0 ||
            qw3_session_create(&oracle, e, 512) != 0 ||
            target->metal_n_gpu_layers != sp->n_layer) goto done;
        qw3_metal_state_info info = {0};
        if (!qw3_metal_session_get_state_info(target->metal, &info) || info.kv_type != 1) goto done;
        qw3_metal_model_config config = {
            .n_layer=1, .n_embd=d, .vocab_size=v, .n_head=sp->n_head,
            .n_head_kv=sp->n_head_kv, .n_head_dim=sp->n_head_dim,
            .n_ff_dense=sp->n_ff_dense, .n_linear_qk_heads=sp->n_linear_qk_heads,
            .n_linear_v_heads=sp->n_linear_v_heads, .n_linear_head_dim=sp->n_linear_head_dim,
            .n_linear_qkv=spec_tensor_linear_qkv(sp), .n_linear_inner=sp->n_linear_inner,
            .n_linear_conv_k=sp->n_linear_conv_k, .full_attn_interval=1
        };
        draft = qw3_metal_session_create(512, &config);
        if (!draft) goto done;
        prev = calloc(d, sizeof(float)); roll = calloc(d, sizeof(float));
        proposal_logits = calloc(v, sizeof(float)); expected = calloc(steps, sizeof(int));
        rows = calloc((size_t)block * v, sizeof(float)); check = calloc((size_t)block * v, sizeof(float));
        hidden = calloc((size_t)block * d, sizeof(float)); check_hidden = calloc((size_t)block * d, sizeof(float));
        upload = [g_device newBufferWithLength:(NSUInteger)d * sizeof(float) options:MTLResourceStorageModeShared];
        concat = qw3_metal_new_private_buffer((NSUInteger)2 * d * sizeof(float));
        draft_log = [g_device newBufferWithLength:(NSUInteger)v * sizeof(float) options:MTLResourceStorageModeShared];
        h_one = [g_device newBufferWithLength:(NSUInteger)d * sizeof(float) options:MTLResourceStorageModeShared];
        log_host = [g_device newBufferWithLength:(NSUInteger)block * v * sizeof(float) options:MTLResourceStorageModeShared];
        h_host = [g_device newBufferWithLength:(NSUInteger)block * d * sizeof(float) options:MTLResourceStorageModeShared];
        if (!prev || !roll || !proposal_logits || !expected || !rows || !check ||
            !hidden || !check_hidden || !upload || !concat || !draft_log || !h_one || !log_host || !h_host) goto done;
        live = decode_state_buffers(target->metal);
        backup = decode_backup(live);
        oracle_live = decode_state_buffers(oracle->metal);
        draft_live = decode_state_buffers(draft);
        draft_backup = decode_backup(draft_live);
        if (!backup || !draft_backup) goto done;
        if (compact) {
            replay_width = block; replay_slots = info.n_linear_layers;
            replay_count = spec_tensor_linear_qkv(sp) + sp->n_linear_inner + 2*sp->n_linear_v_heads;
            if (!replay_slots || replay_slots >= 64) goto done;
            replay_inputs = qw3_metal_new_private_buffer((NSUInteger)block*replay_slots*replay_count*sizeof(float));
            replay_kv = qw3_metal_new_private_buffer((NSUInteger)block*info.n_full_layers*2*info.kv_row_bytes);
            replay_heads = qw3_metal_new_private_buffer((NSUInteger)block*(sp->n_vocab+2*sp->n_embd)*sizeof(float));
            if (!replay_inputs || !replay_kv || !replay_heads) goto done;
        }
        NSUInteger snapshot_bytes = 0;
        for (id<MTLBuffer> b in backup) snapshot_bytes += b.length;
        /* Prefix catch-up uses true target hidden rows, shifted right by one. */
        for (int i = 0; i < prompt.len; i++) {
            if (!mtp_native_eval(e, draft, prompt.v[i], i, prev, upload, concat, draft_log, NULL) ||
                qw3_session_eval_gpu_logits(target, prompt.v[i], err, sizeof(err)) != 0 ||
                !decode_hidden(target->metal, h_one, prev, d)) goto done;
        }
        if (!decode_copy_state(live, decode_state_buffers(baseline->metal))) goto done;
        for (int i = 0; i < prompt.len; i++) token_vec_push(&baseline->tokens, prompt.v[i]);
        decode_rewind(baseline, prompt.len);
        double t = qw3_now_sec();
        for (int i = 0; i < steps; i++) {
            expected[i] = qw3_session_argmax(baseline);
            if (expected[i] < 0 || qw3_session_eval_gpu_logits(baseline, expected[i], err, sizeof(err)) != 0) goto done;
        }
        baseline_time = qw3_now_sec() - t;
        printf("mtp_decode mode=%s steps=%d block=%d prefix=%d force_reject=%d snapshot_mib=%.2f replay_log_mib=%.2f "
               "greedy_only=1 correctness_instrumented=1 production_enabled=0\n",
            batch ? "batch" : compact ? "serial-replay" : "serial", steps, block, prompt.len, force, snapshot_bytes / 1048576.0,
            (replay_inputs.length + replay_kv.length + replay_heads.length) / 1048576.0);
        fflush(stdout);
        uint64_t hash = 1469598103934665603ULL;
        while (produced < steps) {
            const int n = MIN(block, steps - produced), pos = (int)target->kv.pos;
            int inputs[8];
            t = qw3_now_sec();
            inputs[0] = qw3_session_argmax(target);
            if (inputs[0] < 0) goto done;
            if (!decode_copy_state(live, backup) || !decode_copy_state(draft_live, draft_backup)) goto done;
            double dt = qw3_now_sec() - t; save_time += dt; total += dt;
            memcpy(roll, prev, (size_t)d * sizeof(float));
            t = qw3_now_sec();
            for (int i = 1; i < n; i++) {
                if (!mtp_native_eval(e, draft, inputs[i-1], pos+i-1, roll, upload, concat, draft_log, proposal_logits) ||
                    !decode_hidden(draft, h_one, roll, d)) goto done;
                inputs[i] = decode_top(proposal_logits, v);
                if (inputs[i] < 0) goto done;
            }
            dt = qw3_now_sec() - t; draft_time += dt; total += dt;
            /* Deterministic fault injection tests a specific rollback boundary.
             * It is explicitly NOT autonomous acceptance/performance evidence. */
            if (force >= 0 && cycles == 0) {
                for (int i = 0; i < n; i++) inputs[i] = expected[i];
                inputs[force] = (inputs[force] + 1) % v;
            }
            t = qw3_now_sec();
            if (compact) {
                memset(replay_seen, 0, sizeof(replay_seen));
                replay_capture_session = target;
            }
            if (!decode_verify(target, inputs, n, batch, log_host, h_host, rows, hidden, err, sizeof(err))) goto done;
            replay_capture_session = NULL;
            int commit = 1;
            while (commit < n && decode_top(rows + (size_t)(commit-1)*v, v) == inputs[commit]) commit++;
            dt = qw3_now_sec() - t; verify_time += dt; total += dt;
            if (force >= 0 && cycles == 0) {
                if (commit != force) { snprintf(err, sizeof(err), "forced rejection boundary was not reached"); goto done; }
                forced_seen = 1;
            }
            /* Independent all-row serial verification is outside timing. It
             * sees the same proposals, including rows after a rejection. */
            if (!decode_copy_state(backup, oracle_live)) goto done;
            for (int i = oracle->tokens.len; i < pos; i++) token_vec_push(&oracle->tokens, target->tokens.v[i]);
            decode_rewind(oracle, pos);
            if (!decode_verify(oracle, inputs, n, 0, log_host, h_host, check, check_hidden, err, sizeof(err))) goto done;
            for (int i = 0; i < n; i++) {
                double maximum, rms;
                if (!decode_compare(rows + (size_t)i*v, check + (size_t)i*v, v, &maximum, &rms)) {
                    snprintf(err, sizeof(err), "all-row verification differs cycle=%d row=%d max=%.9g rms=%.9g", cycles, i, maximum, rms);
                    goto done;
                }
                if (maximum > worst_max) worst_max = maximum;
                if (rms > worst_rms) worst_rms = rms;
                if (!decode_vector_error(hidden + (size_t)i*d, check_hidden + (size_t)i*d,
                                         d, &maximum, &rms) || maximum > 0.001 || rms > 0.0001) {
                    snprintf(err, sizeof(err), "hidden verification differs cycle=%d row=%d max=%.9g rms=%.9g", cycles, i, maximum, rms);
                    goto done;
                }
                if (maximum > hidden_max) hidden_max = maximum;
                if (rms > hidden_rms) hidden_rms = rms;
                checked_rows++;
            }
            if (commit < n) {
                rejected++;
                t = qw3_now_sec();
                if (!decode_copy_state(backup, live)) goto done;
                if (compact) {
                    if (!decode_replay_apply(target, pos, commit)) goto done;
                    decode_rewind(target, pos+commit);
                } else {
                    decode_rewind(target, pos);
                    if (!decode_verify(target, inputs, commit, 0, log_host, h_host, rows, hidden, err, sizeof(err))) goto done;
                }
                dt = qw3_now_sec() - t; restore_time += dt; total += dt;
                t = qw3_now_sec();
                if (!decode_copy_state(backup, oracle_live)) goto done;
                decode_rewind(oracle, pos);
                if (!decode_verify(oracle, inputs, commit, 0, log_host, h_host, check, check_hidden, err, sizeof(err))) goto done;
                scalar_restore_time += qw3_now_sec()-t;
                if (!decode_state_equal(live, oracle_live)) goto done;
                rollback_checks++;
            }
            t = qw3_now_sec();
            if (!decode_copy_state(draft_backup, draft_live)) goto done;
            for (int i = 0; i < commit; i++) {
                if (!mtp_native_eval(e, draft, inputs[i], pos+i, prev, upload, concat, draft_log, NULL)) goto done;
                memcpy(prev, hidden + (size_t)i*d, (size_t)d * sizeof(float));
            }
            dt = qw3_now_sec() - t; catchup_time += dt; total += dt;
            for (int i = 0; i < commit; i++) {
                if (inputs[i] != expected[produced+i]) {
                    snprintf(err, sizeof(err), "greedy output differs at %d", produced+i); goto done;
                }
                hash = (hash ^ (uint32_t)inputs[i]) * 1099511628211ULL;
            }
            accepted_drafts += commit - 1;
            produced += commit;
            cycles++;
            if (target->kv.pos != prompt.len + produced ||
                target->tokens.len != prompt.len + produced ||
                memcmp(target->tokens.v, prompt.v, (size_t)prompt.len*sizeof(int)) ||
                memcmp(target->tokens.v + prompt.len, expected, (size_t)produced*sizeof(int))) goto done;
            printf("mtp_cycle=%d proposed=%d committed=%d rejected=%d produced=%d\n",
                cycles, n-1, commit, commit < n, produced);
            fflush(stdout);
        }
        if (force >= 0 && !forced_seen) goto done;
        if (!batch && !decode_state_equal(live, decode_state_buffers(baseline->metal))) {
            snprintf(err, sizeof(err), "final state differs from ordinary serial decode"); goto done;
        }
        /* The all-row oracle starts from the speculative state. Also compare
         * against the untouched serial generator to catch accumulated drift. */
        for (int i = 0; i <= 8; i++) {
            double maximum, rms;
            if (qw3_session_copy_logits(target, rows, v) != v ||
                qw3_session_copy_logits(baseline, check, v) != v ||
                !decode_compare(rows, check, v, &maximum, &rms)) {
                snprintf(err, sizeof(err), "serial continuation differs at %d", i); goto done;
            }
            if (i == 8) break;
            const int next = decode_top(check, v);
            if (qw3_session_eval_gpu_logits(target, next, err, sizeof(err)) != 0 ||
                qw3_session_eval_gpu_logits(baseline, next, err, sizeof(err)) != 0) goto done;
        }
        printf("mtp_summary tokens=%d matched=%d hash=%" PRIx64 " cycles=%d accepted_drafts=%d "
               "rejected_cycles=%d rollback_checks=%d all_rows=%d max=%.9g rms=%.9g hidden_max=%.9g hidden_rms=%.9g continuation=8 serial_state_exact=%d "
               "baseline_tps=%.3f experimental_tps=%.3f ratio=%.3f "
               "save_ms=%.3f draft_ms=%.3f verify_ms=%.3f restore_replay_ms=%.3f catchup_ms=%.3f scalar_restore_reference_ms=%.3f compact_replay=%d "
               "not_promotion_benchmark=1\n",
               produced, steps, hash, cycles, accepted_drafts, rejected, rollback_checks,
               checked_rows, worst_max, worst_rms, hidden_max, hidden_rms, !batch, steps/baseline_time, steps/total,
               baseline_time/total, save_time*1000, draft_time*1000, verify_time*1000,
               restore_time*1000, catchup_time*1000, scalar_restore_time*1000, compact);
        status = 0;
done:
        if (status) fprintf(stderr, "MTP decode probe failed: %s\n", err);
        /* A failed graph can leave an encoded command batch in flight. */
        if (e) (void)qw3_metal_synchronize();
        replay_capture_session = NULL;
        replay_inputs = nil; replay_kv = nil; replay_heads = nil;
        if (draft) qw3_metal_session_free(draft);
        if (oracle) qw3_session_free(oracle);
        if (baseline) qw3_session_free(baseline);
        if (target) qw3_session_free(target);
        if (e) qw3_engine_close(e);
        qw3_tokens_free(&prompt);
        free(prev); free(roll); free(proposal_logits); free(expected);
        free(rows); free(hidden); free(check); free(check_hidden);
        return status;
    }
}
